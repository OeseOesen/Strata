#!/usr/bin/env python3
"""m4serve - an OpenAI-compatible HTTP server in front of strata-m4 (Mistral Small 4).

    python3 m4serve.py --engine ./strata-m4 --model /models/Mistral-Small-4-Q4_K_M.gguf \
        --tokenizer /models/tekken.json [--mmproj mmproj.gguf --vision ./m4-vision] [--port 8080] \
        [--engine-args "--ctx 65536 --spec 6"]

Prompts are built by mistral-common (Mistral's own reference for the chat format: system prompt, tools,
[THINK] reasoning with reasoning_effort, images); the engine only sees token ids.  Answers are split back into
content, reasoning_content and tool_calls.

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions (stream or not), POST /v1/completions,
GET /v1/telemetry and /v1/status (Strata's read-only telemetry contract, schema 1: `strata monitor --url http://host:port`
shows this server like a Strata server; run it on port 8095 and the monitor needs no --url).
One request runs at a time (the engine keeps one prompt cache); others wait in line.  With the engine option
--parallel N, up to N requests run together (each in a slot with its own prompt cache, decoded in shared windows).
"""
from __future__ import annotations

import argparse
import signal
import base64
import collections
import hashlib
import json
import re
import os
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


def log(*a):
    print("[m4serve]", *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------- the engine process
class Engine:
    def __init__(self, exe: str, model: str, extra: list[str]):
        self.args = [exe, "-m", model, "--serve", *extra]
        log("starting", " ".join(shlex.quote(a) for a in self.args))
        self.p = subprocess.Popen(self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lock = threading.Lock()
        self.info = {}
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("strata-m4 exited before READY (see its log above)")
            if line.startswith("READY"):
                self.info = json.loads(line[6:] or "{}")
                break
        log("engine ready:", json.dumps(self.info))
        self.parallel = max(1, int(self.info.get("parallel") or 1))
        if self.parallel > 1:   # tagged lines (ADMIT/PP/W/T/DONE/ERR <id> ...), routed to their request by a reader
            self.wlock, self.plock = threading.Lock(), threading.Lock()
            self.pending: dict[str, queue.Queue] = {}
            self.dead = False
            threading.Thread(target=self._reader, daemon=True).start()

    def _send(self, line: str):
        with self.wlock:
            self.p.stdin.write(line + "\n")
            self.p.stdin.flush()

    def _reader(self):
        for line in self.p.stdout:
            f = line.rstrip("\n").split(" ", 2)
            if f[0] in ("ADMIT", "PP", "W", "T", "DONE", "ERR") and len(f) >= 2:
                with self.plock:
                    q = self.pending.get(f[1])
                if q is not None:
                    q.put((f[0], f[2] if len(f) > 2 else ""))
                    continue
            if f[0] == "ERR":
                log("engine:", line.strip())
        with self.plock:
            self.dead = True
            for q in self.pending.values():
                q.put(None)

    def generate(self, req: dict, on_tokens, cancelled, on_progress=None, on_window=None, on_admit=None,
                 admit_timeout: float | None = None) -> dict:
        """Runs one GEN; on_tokens(list[int]) per accepted window; on_progress(read, total, reused) per prompt chunk,
        on_window(rows, taken) per decode window; on_admit(slot, reused) when it gets a slot (--parallel); returns
        the DONE stats.  --parallel: a request without a slot after admit_timeout s is withdrawn (QueueTimeout)."""
        if self.parallel > 1:
            return self._generate_tagged(req, on_tokens, cancelled, on_progress, on_window, on_admit, admit_timeout)
        if on_admit:
            on_admit(0, None)
        self.p.stdin.write("GEN " + json.dumps(req) + "\n")
        self.p.stdin.flush()
        stopped = False
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("strata-m4 exited")
            if line.startswith("T "):
                ids = [int(x) for x in line[2:].split()]
                if not stopped and cancelled():
                    self.p.stdin.write("STOP\n")
                    self.p.stdin.flush()
                    stopped = True
                if not stopped:
                    on_tokens(ids)
            elif line.startswith("PP ") or line.startswith("W "):
                try:
                    f = [int(x) for x in line.split()[1:]]
                    if line[0] == "P" and on_progress and len(f) == 3:
                        on_progress(*f)
                    elif line[0] == "W" and on_window and len(f) == 2:
                        on_window(*f)
                except ValueError:
                    pass
            elif line.startswith("DONE"):
                return json.loads(line[5:] or "{}")
            elif line.startswith("ERR"):
                raise RuntimeError(line[4:].strip())

    def _generate_tagged(self, req, on_tokens, cancelled, on_progress, on_window, on_admit, admit_timeout):
        rid = uuid.uuid4().hex[:12]
        q: queue.Queue = queue.Queue()
        with self.plock:
            if self.dead:
                raise RuntimeError("strata-m4 exited")
            self.pending[rid] = q
        try:
            self._send("GEN " + json.dumps({**req, "id": rid}))
            t0 = time.time()
            admitted = stopped = timed_out = False
            while True:
                try:
                    item = q.get(timeout=0.25)
                except queue.Empty:
                    item = ()
                if item is None:
                    raise RuntimeError("strata-m4 exited")
                if not stopped:
                    late = not admitted and admit_timeout is not None and time.time() - t0 > admit_timeout
                    if late or cancelled():
                        timed_out = late
                        self._send("STOP " + rid)
                        stopped = True
                if not item:
                    continue
                tag, rest = item
                if tag == "T":
                    if not stopped:
                        on_tokens([int(x) for x in rest.split()])
                elif tag in ("PP", "W", "ADMIT"):
                    try:
                        f = [int(x) for x in rest.split()]
                    except ValueError:
                        continue
                    if tag == "ADMIT" and len(f) == 2:
                        admitted = True
                        if on_admit:
                            on_admit(f[0], f[1])
                    elif tag == "PP" and on_progress and len(f) == 3:
                        on_progress(*f)
                    elif tag == "W" and on_window and len(f) == 2:
                        on_window(*f)
                elif tag == "DONE":
                    st = json.loads(rest or "{}")
                    if timed_out and not admitted:
                        raise QueueTimeout()
                    return st
                elif tag == "ERR":
                    raise RuntimeError(rest.strip())
        finally:
            with self.plock:
                self.pending.pop(rid, None)


class QueueTimeout(Exception):
    """--parallel: the request found no free slot within --queue-timeout s."""


# ---------------------------------------------------------------------------------------------- images
class Vision:
    """The resident m4-vision helper (llama.cpp's mtmd with the model's mmproj)."""

    def __init__(self, exe: str, mmproj: str, model: str, cpu: bool):
        args = [exe, "--mmproj", mmproj, "--model", model] + (["--cpu"] if cpu else [])
        log("starting vision:", " ".join(args))
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lock = threading.Lock()
        self.dir = tempfile.mkdtemp(prefix="m4img-")
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("m4-vision exited before READY")
            if line.startswith("READY"):
                break
        self.cache: dict[str, tuple[str, dict]] = {}

    def encode(self, image_bytes: bytes) -> tuple[str, dict]:
        key = hashlib.sha256(image_bytes).hexdigest()
        if key in self.cache:
            return self.cache[key]
        src = os.path.join(self.dir, key + ".img")
        out = os.path.join(self.dir, key + ".f32")
        with open(src, "wb") as f:
            f.write(image_bytes)
        with self.lock:
            self.p.stdin.write(f"ENC {src} {out}\n")
            self.p.stdin.flush()
            line = self.p.stdout.readline()
        if not line.startswith("OK"):
            raise RuntimeError("image encoder: " + line.strip())
        meta = json.loads(line[3:])
        self.cache[key] = (out, meta)
        return out, meta


def image_bytes_of(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    if url.startswith("file://"):
        with open(url[7:], "rb") as f:
            return f.read()
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


# ---------------------------------------------------------------------------------------------- the chat format
_ID9 = __import__("re").compile(r"^[A-Za-z0-9]{9}$")
_ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _id9(raw) -> str:
    """Mistral's tool call ids are 9 letters / digits; agents send OpenAI's (call_...) or Anthropic's (toolu_...)."""
    raw = str(raw or "")
    if _ID9.match(raw):
        return raw
    h = hashlib.sha256(raw.encode()).digest()
    return "".join(_ALNUM[b % len(_ALNUM)] for b in h[:9])


def _parts(content):
    """Content parts mistral-common reads (text, image_url); other spellings of text become text, the rest goes."""
    if not isinstance(content, list):
        return content
    out = []
    for p in content:
        if not isinstance(p, dict):
            continue
        t = p.get("type")
        if t in ("text", "input_text", "output_text") and isinstance(p.get("text"), str):
            out.append({"type": "text", "text": p["text"]})
        elif t == "image_url":
            out.append({"type": "image_url", "image_url": p["image_url"]})
    return out


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text"))
    return ""


def normalize_messages(messages: list) -> list:
    """What agents send, in the shape Mistral's validator accepts: the leading system / developer messages merged into
    one system message, later ones as user turns marked [System]; tool call ids as 9 letters / digits (the same mapping for a call and its result); a tool
    result without an id gets the id of the next open call; empty assistant turns dropped; only the fields
    mistral-common reads."""
    systems, rest, open_ids = [], [], []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("system", "developer"):
            t = _text_of(m.get("content"))
            if t.strip():
                if rest:   # a reminder later in the chat: a user turn there (moving it up would change the prompt's
                    rest.append({"role": "user", "content": "[System]\n" + t})   # prefix and void the KV cache)
                else:
                    systems.append(t)
            continue
        if role == "assistant":
            calls = []
            for c in m.get("tool_calls") or []:
                f = c.get("function") or {}
                args = f.get("arguments", "{}")
                calls.append({"id": _id9(c.get("id")), "type": "function",
                              "function": {"name": f.get("name", ""), "arguments": args if isinstance(args, str) else json.dumps(args)}})
            content = _parts(m.get("content"))
            if isinstance(content, list) and not any(p.get("type") == "image_url" for p in content):
                content = _text_of(content)
            if calls:
                open_ids = [c["id"] for c in calls]
                msg = {"role": "assistant", "content": content if (isinstance(content, str) and content.strip()) else None,
                       "tool_calls": calls}
            elif isinstance(content, str) and content.strip():
                msg = {"role": "assistant", "content": content}
            else:
                continue
            rest.append(msg)
        elif role == "tool":
            tid = m.get("tool_call_id")
            tid = _id9(tid) if tid else (open_ids[0] if open_ids else _id9("missing"))
            if tid in open_ids:
                open_ids.remove(tid)
            msg = {"role": "tool", "tool_call_id": tid, "content": _text_of(m.get("content"))}
            if m.get("name"):
                msg["name"] = m["name"]
            rest.append(msg)
        elif role == "user":
            content = _parts(m.get("content"))
            rest.append({"role": "user", "content": content if content not in (None, []) else ""})
    if systems:
        rest.insert(0, {"role": "system", "content": "\n\n".join(systems)})
    return rest


def normalize_tools(tools):
    out = []
    for t in tools or []:
        f = (t or {}).get("function") if isinstance(t, dict) else None
        if not f or not f.get("name"):
            continue
        out.append({"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                                     "parameters": f.get("parameters") or {"type": "object", "properties": {}}}})
    return out or None


def load_default_system_prompt(spec: str, repo: str) -> str | None:
    """Mistral's own system prompt (SYSTEM_PROMPT.txt of the model repo, as their vLLM examples load it), used when a
    request brings none.  spec: auto (from the repo), none, or a file."""
    if not spec or spec == "none":
        return None
    try:
        if spec == "auto":
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo, "SYSTEM_PROMPT.txt")
        else:
            path = spec
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        log("no default system prompt:", e)
        return None
    import datetime
    today = datetime.date.today()
    for k, v in (("{name}", repo.split("/")[-1]), ("{today}", today.isoformat()),
                 ("{yesterday}", (today - datetime.timedelta(days=1)).isoformat())):
        text = text.replace(k, v)
    log(f"default system prompt: {len(text)} characters (requests without a system message get it)")
    return text


class Chat:
    default_system: str | None = None
    default_effort: str | None = None   # for requests that name none: "none", "high" (None: the model's own default)

    def __init__(self, tokenizer: str):
        if os.path.exists(tokenizer):
            self.mt = MistralTokenizer.from_file(tokenizer)
        else:
            self.mt = MistralTokenizer.from_hf_hub(tokenizer)
        self.tok = self.mt.instruct_tokenizer.tokenizer
        self.eos = self.tok.eos_id
        self.ids = {}
        for name in ("[THINK]", "[/THINK]", "[TOOL_CALLS]", "[ARGS]", "[CALL_ID]", "[IMG]", "[IMG_BREAK]", "[IMG_END]"):
            self.ids[name] = self._special(name)
        log("tokenizer:", tokenizer, "eos", self.eos, {k: v for k, v in self.ids.items()})

    def _special(self, s: str):
        for fn in ("get_special_token", "get_control_token"):
            f = getattr(self.tok, fn, None)
            if f:
                try:
                    return int(f(s))
                except Exception:
                    pass
        return None

    def encode(self, body: dict, vision: Vision | None):
        """-> (ids, embd list for the engine, embd_key)"""
        messages = body.get("messages") or []
        images: list[bytes] = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                for part in c:
                    if part.get("type") == "image_url":
                        u = part["image_url"]["url"] if isinstance(part["image_url"], dict) else part["image_url"]
                        images.append(image_bytes_of(u))
        kw = {}
        eff = effort_of(body, self.default_effort)
        if eff:
            kw["reasoning_effort"] = eff
        msgs, tools = normalize_messages(messages), normalize_tools(body.get("tools"))
        if self.default_system and not (msgs and msgs[0].get("role") == "system"):
            msgs.insert(0, {"role": "system", "content": self.default_system})

        def build(ms):
            r = ChatCompletionRequest.from_openai(ms, tools=tools, **kw)
            if r.messages and getattr(r.messages[-1], "role", None) == "assistant" and hasattr(r.messages[-1], "prefix"):
                r.messages[-1].prefix = True   # a trailing assistant turn is continued, not answered
            return self.mt.encode_chat_completion(r)
        try:
            tokd = build(msgs)
        except Exception as e:
            if "not both" not in str(e):
                raise
            # older validators: an assistant turn has text or tool calls, not both - keep the calls
            tokd = build([{**m, "content": None} if m.get("role") == "assistant" and m.get("tool_calls") else m for m in msgs])
        ids = list(tokd.tokens)
        embd, key = [], ""
        if images:
            if vision is None:
                raise ValueError("this server runs without an image encoder (--mmproj / --vision)")
            ids, embd = self._lay_out_images(ids, images, vision)
            key = hashlib.sha256(b"".join(hashlib.sha256(b).digest() for b in images)).hexdigest()
        return ids, embd, key

    def _lay_out_images(self, ids, images, vision):
        """Each image span mistral-common wrote ([IMG]... [IMG_END]) becomes mtmd's layout for that image: its `pre`
        tokens, one placeholder per embedding row, its `post` tokens."""
        img, end = self.ids["[IMG]"], self.ids["[IMG_END]"]
        brk = self.ids["[IMG_BREAK]"]
        out, embd, i, k = [], [], 0, 0
        while i < len(ids):
            if ids[i] == img and k < len(images):
                j = i
                while j < len(ids) and ids[j] != end:
                    j += 1
                path, meta = vision.encode(images[k])
                k += 1
                out.extend(meta.get("pre", []))
                rows = list(range(len(out), len(out) + meta["n_tokens"]))
                out.extend([img] * meta["n_tokens"])
                post = meta.get("post", [])
                out.extend(post if post else [end])
                embd.append({"file": path, "rows": rows})
                i = j + 1
                continue
            out.append(ids[i])
            i += 1
        return out, embd

    def _dec(self, x: list[int]) -> str:
        return self.tok.decode(x, special_token_policy=SpecialTokenPolicy.IGNORE) if x else ""

    def split(self, ids: list[int], finish: str | None = None, tools: list | None = None):
        """-> (content, reasoning, tool_calls).  finish: the engine's reason the answer ended ("stop": the model ended
        it itself - then a call whose JSON lacks closing brackets is completed)."""
        th, eth, tc = self.ids["[THINK]"], self.ids["[/THINK]"], self.ids["[TOOL_CALLS]"]
        reasoning, rest, inside = [], [], False
        for t in ids:
            if t == th:
                inside = True
            elif t == eth:
                inside = False
            elif inside:
                reasoning.append(t)
            else:
                rest.append(t)
        calls = []
        if tc is not None and tc in rest:
            i = rest.index(tc)
            try:
                calls = self.decode_calls(rest[i:], finish == "stop", tool_params(tools))
                rest = rest[:i]
            except Exception as e:   # malformed call: it stays text (and the log says why)
                raw = self._dec(rest[i:])
                log(f"tool call not parsed (answer ended by {finish or '?'}, {len(rest) - i} tokens): {e}; "
                    f"start {raw[:120]!r} ... end {raw[-160:]!r}")
                calls = []
        content = self._dec(rest)
        if not calls and finish == "stop" and tools:
            bare = bare_call(content, tools)
            if bare:
                log(f"tool call {bare['function']['name']!r}: the answer was the call's arguments as plain text "
                    "(no [TOOL_CALLS]); turned into the call")
                return "", self._dec(reasoning), [bare]
        return content, self._dec(reasoning), calls

    def decode_calls(self, ids: list[int], may_close: bool, params: dict | None = None) -> list[dict]:
        """[TOOL_CALLS]name([CALL_ID]id)?[ARGS]{json} ... -> OpenAI tool calls.  Tolerates text after a call's JSON
        (the next call's name when the model left out [TOOL_CALLS]); may_close: the answer's last JSON may lack its
        closing brackets (the model ended the answer itself - never for a cut-off answer)."""
        tc, arg, cid = self.ids["[TOOL_CALLS]"], self.ids["[ARGS]"], self.ids.get("[CALL_ID]")
        groups, cur = [], None
        for t in ids:
            if t == tc:
                cur = []
                groups.append(cur)
            elif cur is not None:
                cur.append(t)
        calls = []

        def add(name, args, call_id=None):
            name = name.strip()
            if not name or not isinstance(args, dict):
                raise ValueError(f"call {name!r}: arguments are not a JSON object")
            calls.append({"id": call_id or "call_" + uuid.uuid4().hex[:9], "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})

        for gi, g in enumerate(groups):
            last_group = gi == len(groups) - 1
            if arg is None or arg not in g:   # the older format: [TOOL_CALLS][{"name": ..., "arguments": {...}}, ...]
                obj, tail, _ = json_prefix(self._dec(g), may_close and last_group)
                for c in obj if isinstance(obj, list) else [obj]:
                    if not isinstance(c, dict):
                        raise ValueError("a call is not a JSON object")
                    add(str(c.get("name", "")), c.get("arguments"), c.get("id"))
                continue
            segs, cur = [[]], None
            for t in g:
                if t == arg:
                    segs.append([])
                else:
                    segs[-1].append(t)
            head = segs[0]
            call_id = None
            if cid is not None and cid in head:
                k = head.index(cid)
                call_id, head = self._dec(head[k + 1:]).strip() or None, head[:k]
            name = self._dec(head)
            for k, seg in enumerate(segs[1:], 1):
                last = last_group and k == len(segs) - 1
                obj, tail, closed = json_prefix(self._dec(seg), may_close and last, (params or {}).get(name.strip()))
                if closed:
                    log(f"tool call {name.strip()!r}: repaired (the model ended it with invalid JSON)")
                add(name, obj, call_id)
                name, call_id = tail, None   # text after the JSON: the next call's name
            if name.strip():
                log(f"tool call: text after the last call ignored: {name.strip()[:80]!r}")
        if not calls:
            raise ValueError("no call found")
        return calls


def effort_of(body: dict, default: str | None) -> str | None:
    """The request's reasoning effort as Mistral Small 4 knows it: "none" or "high" (agents also send minimal / low /
    medium / xhigh); a request without one gets default (None: the model's own default)."""
    eff = body.get("reasoning_effort")
    if eff is None and isinstance(body.get("reasoning"), dict):   # the Responses-style spelling
        eff = body["reasoning"].get("effort")
    if eff is None:
        return default
    return "none" if str(eff).strip().lower() in ("none", "off", "minimal", "false", "0", "") else "high"


def may_be_bare_call(text: str) -> bool:
    """While streaming: could this answer still turn out to be a tool call written as plain JSON?"""
    t = text.lstrip()
    if t.startswith("{"):
        return True
    if t.startswith("```"):
        first = t.split("\n", 1)[0].strip("`").strip().lower()
        return first in ("", "json") or ("\n" not in t and "json".startswith(first))
    return t == "" or "```".startswith(t)


def bare_call(text: str, tools) -> dict | None:
    """An answer that is nothing but one JSON object - a tool call written as plain text, without [TOOL_CALLS] -
    -> that call, if it fits exactly one of the request's tools: {"name": tool, "arguments": {...}}, or the arguments
    alone whose names are that tool's (its required ones all there).  Otherwise None (the answer stays text)."""
    t = text.strip()
    if t.startswith("```"):   # a fenced block: ```json ... ```
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0].strip()
    if not (t.startswith("{") and t.endswith("}")):
        return None
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not obj:
        return None
    specs = {}
    for tl in tools or []:
        f = tl.get("function", tl) if isinstance(tl, dict) else {}
        if isinstance(f, dict) and f.get("name"):
            p = f.get("parameters") or {}
            specs[f["name"]] = (set((p.get("properties") or {})), set(p.get("required") or []))
    def call(name, args):
        return {"id": "call_" + uuid.uuid4().hex[:9], "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
    if set(obj) <= {"name", "arguments", "parameters"} and obj.get("name") in specs:
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        return call(obj["name"], args) if isinstance(args, dict) else None
    fits = [n for n, (props, req) in specs.items() if props and set(obj) <= props and req <= set(obj)]
    return call(fits[0], obj) if len(fits) == 1 else None


def tool_params(tools) -> dict:
    """{tool name: its parameter names} from a request's tools"""
    out = {}
    for t in tools or []:
        f = t.get("function", t) if isinstance(t, dict) else {}
        props = ((f.get("parameters") or {}).get("properties") or {}) if isinstance(f, dict) else {}
        if f.get("name") and props:
            out[f["name"]] = list(props)
    return out


def request_line(rid: str, st: dict, outcome: str, req: dict | None = None) -> str:
    """One log line per request: what was read (and how), what was written."""
    n, reused, gen = int(st.get("n_prompt") or 0), int(st.get("n_reused") or 0), int(st.get("n_gen") or 0)
    read, pms, gms = n - reused, float(st.get("prompt_ms") or 0), float(st.get("gen_ms") or 0)
    gib = float(st.get("streamed_gib") or 0)
    s = f"request {rid[:12]}: prompt {n} tokens ({reused} cached)"
    if read > 1 and pms > 0:
        s += f", read {read} in {pms / 1000:.1f} s = {read / (pms / 1000):.0f} tok/s " + \
             (f"({gib:.1f} GiB of experts streamed)" if gib > 0 else "(experts on the CPU)")
    if gen:
        s += f"; answer {gen} tokens" + (f" in {gms / 1000:.1f} s = {gen / (gms / 1000):.1f} tok/s" if gms > 0 else "")
        if st.get("eagle_windows"):
            s += f" (eagle {st.get('eagle_accepted', 0)}/{st.get('eagle_drafted', 0)})"
    if st.get("slot") is not None and int(st.get("slot", -1)) >= 0:
        s += f"; slot {st['slot']}"
    if req:   # how the request asked to be answered
        s += f"; thinking {req.get('_think') or 'model default'}, temperature {req.get('temp')}" + \
             (f" (tool calls {req['tool_temp']})" if "tool_temp" in req else "") + \
             f", limit {req.get('max_new')}" + (" (client)" if req.get("_limit_client") else "")
    return s + f"; {outcome}"


def repair_by_keys(s: str, keys: list[str]) -> dict | None:
    """An arguments object that is no valid JSON because a string value holds unescaped quotes or raw line breaks
    (code written into an argument: open("x", "r")) -> the object, read along the tool's own parameter names; None if
    no reading fits.  A value is the text between one '"name":' and the next (or the final '}'); a value in quotes
    that does not parse is taken as the raw string it was meant to be."""
    s = s.strip()
    if not keys or not s.startswith("{") or not s.endswith("}"):
        return None
    pat = re.compile(r'[{,]\s*"(' + "|".join(re.escape(k) for k in keys) + r')"\s*:\s*')
    marks = [m for m in pat.finditer(s)]
    if not marks or marks[0].start() != 0:
        return None

    def value(t: str):
        t = t.strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"':   # the raw string: escape what the model left bare
            body = t[1:-1]
            return json.loads('"' + re.sub(r'(?<!\\)"', r'\\"', body).replace("\n", "\\n").replace("\r", "\\r")
                              .replace("\t", "\\t") + '"')
        raise ValueError

    def walk(k: int, seen: frozenset):   # marks[k] starts a key; returns the object from here or None
        m = marks[k]
        name = m.group(1)
        if name in seen:
            return None
        for nxt in range(k + 1, len(marks) + 1):   # where this value ends: before a later mark or the final '}'
            end = marks[nxt].start() if nxt < len(marks) else len(s) - 1
            try:
                v = value(s[m.end():end])
            except (ValueError, json.JSONDecodeError):
                continue
            if nxt == len(marks):
                return {name: v}
            rest = walk(nxt, seen | {name})
            if rest is not None:
                return {name: v, **rest}
        return None
    try:
        return walk(0, frozenset())
    except RecursionError:
        return None


def json_prefix(text: str, may_close: bool, keys: list[str] | None = None):
    """The JSON value at the start of text -> (value, text after it, closed).  may_close: a value that ends early
    (closing brackets missing, nothing cut inside a string) is completed, and with the tool's parameter names (keys)
    unescaped quotes in its strings are escaped."""
    s = text.lstrip()
    try:
        obj, end = json.JSONDecoder().raw_decode(s)
        return obj, s[end:], False
    except json.JSONDecodeError as e:
        if not may_close:
            raise ValueError(f"invalid JSON ({e.msg} at {e.pos} of {len(s)} characters)") from None
        err = e
    obj = repair_by_keys(s, keys or [])   # code in an argument with its quotes not escaped (needs the final '}')
    if obj is not None:
        log("tool call: unescaped quotes / line breaks inside an argument escaped (the model wrote code into it)")
        return obj, "", True
    stack, in_str, esc = [], False, False   # the closing brackets left out
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                stack = None
                break
    if not in_str and stack:
        try:
            obj, end = json.JSONDecoder().raw_decode(s.rstrip().rstrip(",") + "".join(reversed(stack)))
            return obj, "", True
        except json.JSONDecodeError:
            pass
    raise ValueError(f"invalid JSON ({err.msg} at {err.pos} of {len(s)} characters)")


# ---------------------------------------------------------------------------------------------- telemetry
class Telemetry:
    """What `strata monitor` reads (GET /v1/telemetry, schema 1; see terminal/strata_cli/client.py sanitize_telemetry):
    the slots (one, or the engine's --parallel N), the queue, request transitions, rates, the GPU (nvidia-smi) and the host's RAM.  Memory in bytes."""

    GPU_FIELDS = ("index", "name", "memory.used", "memory.total", "utilization.gpu", "temperature.gpu", "power.draw",
                  "power.limit", "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current",
                  "pcie.link.width.max")

    def __init__(self, model_id: str):
        self.lock = threading.Lock()
        self.instance = uuid.uuid4().hex[:16]
        self.boot = time.time()
        self.model_id = model_id
        self.context = None
        self.seq = 0
        self.events = collections.deque(maxlen=64)
        self.queue: dict[str, dict] = {}          # waiting requests, in order
        self.active: dict[str, dict] = {}         # the requests on the engine (each in a slot)
        self.slots_n = 1
        self.recent = collections.deque(maxlen=50)
        self.totals = {"since": self.boot, "requests": 0, "prompt_tokens": 0, "output_tokens": 0}
        self.max_queue, self.queue_timeout = 16, 900.0
        self.rejected = self.timeouts = self.disconnects = 0
        self.last_answer = None
        self.last_prefill = None
        self.last_prompt: dict | None = None
        self.devices: list[dict] = []
        self.windows = collections.deque(maxlen=512)   # (time, rows, taken) per decode window
        threading.Thread(target=self._sample_gpus, daemon=True).start()

    # -- request transitions
    def _event(self, rid: str, event: str, outcome: str | None = None):
        self.seq += 1
        self.events.append({"seq": self.seq, "at": time.time(), "request_id": rid, "event": event, "outcome": outcome})

    def queued(self, prompt_tokens: int) -> str:
        rid = uuid.uuid4().hex[:12]
        with self.lock:
            self.queue[rid] = {"id": rid, "state": "queued", "queued_at": time.time(), "admitted_at": None,
                               "prompt_tokens": prompt_tokens}
            self._event(rid, "queued")
        return rid

    def admitted(self, rid: str, slot: int = 0):
        with self.lock:
            row = self.queue.pop(rid, None) or {"id": rid, "queued_at": time.time(), "prompt_tokens": None}
            now = time.time()
            self.active[rid] = {**row, "state": "reading", "admitted_at": now, "generated": 0, "first_at": None,
                                "slot": slot}
            self._event(rid, "admitted")
            self._event(rid, "reading")

    def reject(self):
        with self.lock:
            self.rejected += 1

    def waiting(self) -> int:
        with self.lock:
            return len(self.queue)

    def inflight(self) -> int:
        with self.lock:
            return len(self.queue) + len(self.active)

    def progress(self, rid: str, read: int, total: int, reused: int):
        with self.lock:
            c = self.active.get(rid)
            if c is not None:
                c["prompt_tokens"] = total + reused + 1   # + the last prompt token (read by the first window)
                c["prompt_read"] = reused + read

    def window(self, rows: int, taken: int):
        with self.lock:
            self.windows.append((time.time(), min(8, max(1, rows)), min(8, max(1, taken))))

    def _batch(self, now: float) -> dict:
        """The decode windows of the last 10 s (or the last 64 of the last minute): rows per window (1 + drafts) and
        rows taken (1 + accepted drafts), Strata's batch-window sample fields."""
        w = [x for x in self.windows if now - x[0] <= 10] or [x for x in list(self.windows)[-64:] if now - x[0] <= 60]
        if len(w) < 2:
            return {}
        return {"batch_sample_windows": len(w), "batch_sample_rows": sum(x[1] for x in w),
                "batch_sample_active_rows": sum(x[2] for x in w),
                "batch_sample_span_s": round(max(1e-3, w[-1][0] - w[0][0]), 3),
                "batch_sample_age_s": round(max(0.0, now - w[-1][0]), 3)}

    def tokens(self, rid: str, n: int):
        with self.lock:
            c = self.active.get(rid)
            if c is None:
                return
            if c["first_at"] is None:
                c["first_at"] = time.time()
                c["state"] = "decoding"
                self._event(c["id"], "first_token")
            c["generated"] += n

    def finished(self, rid: str, st: dict | None, outcome: str):
        with self.lock:
            self.queue.pop(rid, None)
            c = self.active.pop(rid, None)
            now = time.time()
            st = st or {}
            event = {"stop": "finished", "length": "finished", "cancel": "cancelled", "disconnect": "disconnected",
                     "timeout": "timeout"}.get(outcome, "error")
            self._event(rid, event, outcome)
            if outcome == "timeout":
                self.timeouts += 1
            elif outcome == "disconnect":
                self.disconnects += 1
            if c is not None:
                n_prompt, n_gen = int(st.get("n_prompt", c.get("prompt_tokens") or 0)), int(st.get("n_gen", c["generated"]))
                pms, gms = float(st.get("prompt_ms", 0) or 0), float(st.get("gen_ms", 0) or 0)
                self.recent.appendleft({"id": rid, "time": now, "duration_s": round(now - c["admitted_at"], 3), "finish": outcome,
                                        "prompt_tokens": n_prompt, "reused": int(st.get("n_reused", 0)), "output_tokens": n_gen,
                                        "prompt_ms": round(pms, 1), "decode_ms": round(gms, 1),
                                        "first_engine_token_s": round(c["first_at"] - c["admitted_at"], 3) if c["first_at"] else None})
                self.totals["requests"] += 1
                self.totals["prompt_tokens"] += n_prompt
                self.totals["output_tokens"] += n_gen
                if gms > 0 and n_gen > 1:
                    self.last_answer = (n_gen / (gms / 1000.0), gms / 1000.0)
                read = n_prompt - int(st.get("n_reused", 0))
                if pms > 0 and read > 16:
                    self.last_prefill = read / (pms / 1000.0)
                    self.last_prompt = {"read": read, "cached": int(st.get("n_reused", 0)), "s": round(pms / 1000.0, 2),
                                        "tok_s": round(self.last_prefill, 1),
                                        "streamed_gib": float(st.get("streamed_gib") or 0)}

    # -- hardware
    def _sample_gpus(self):
        while True:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(self.GPU_FIELDS), "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                rows = []
                for line in out.strip().splitlines()[:16]:
                    f = [x.strip() for x in line.split(",")]
                    if len(f) != len(self.GPU_FIELDS):
                        continue
                    num = lambda x: (float(x) if "." in x else int(x)) if x.replace(".", "", 1).isdigit() else None
                    mib = lambda x: int(float(x) * 2**20) if num(x) is not None else None
                    rows.append({"index": num(f[0]), "name": f[1], "mem_used": mib(f[2]), "mem_total": mib(f[3]),
                                 "util": num(f[4]), "temp": num(f[5]), "power": num(f[6]), "power_limit": num(f[7]),
                                 "pcie_gen": num(f[8]), "pcie_gen_max": num(f[9]), "pcie_width": num(f[10]),
                                 "pcie_width_max": num(f[11])})
                with self.lock:
                    self.devices = rows
            except Exception:
                pass
            time.sleep(1.0)

    @staticmethod
    def _ram():
        try:
            m = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    m[k] = int(v.split()[0]) * 1024
            return {"used": m["MemTotal"] - m.get("MemAvailable", m.get("MemFree", 0)), "total": m["MemTotal"]}
        except Exception:
            return {"used": None, "total": None}

    # -- the snapshot
    def snapshot(self) -> dict:
        now = time.time()
        with self.lock:
            act = [dict(c) for c in self.active.values()]
            queue = [dict(r) for r in self.queue.values()]
            S = max(self.slots_n, 1 + max((c["slot"] for c in act), default=0))
            slots = [{"slot": i, "group": 0, "state": "idle", "held_tokens": None} for i in range(S)]
            requests, rates = [], []
            for c in act:
                live = (c["generated"] / (now - c["first_at"])) if c["first_at"] and now > c["first_at"] and c["generated"] > 1 else None
                if live:
                    rates.append(live)
                read = c.get("prompt_tokens") if c["first_at"] else c.get("prompt_read")
                slots[c["slot"]] = {"slot": c["slot"], "group": 0, "replica": None, "request_id": c["id"], "state": c["state"],
                                    "held_tokens": None, "prompt_tokens": c.get("prompt_tokens"), "prompt_read": read,
                                    "generated": c["generated"], "elapsed_s": round(now - c["admitted_at"], 2),
                                    "tok_s": round(live, 1) if live else None}
                requests.append({"id": c["id"], "state": c["state"], "queued_at": c["queued_at"], "admitted_at": c["admitted_at"],
                                 "slot": c["slot"], "group": 0, "replica": None, "prompt_tokens": c.get("prompt_tokens"),
                                 "prompt_read": read, "generated": c["generated"],
                                 "first_engine_token_s": round(c["first_at"] - c["admitted_at"], 3) if c["first_at"] else None})
            requests += [{"id": r["id"], "state": "queued", "queued_at": r["queued_at"], "admitted_at": None, "slot": None,
                          "group": None, "replica": None, "prompt_tokens": r.get("prompt_tokens"), "prompt_read": None,
                          "generated": None, "first_engine_token_s": None} for r in queue]
            live_rate = sum(rates) if rates else None   # all conversations together
            answer = (live_rate, None) if live_rate else (self.last_answer or (None, None))
            return {
                "schema_version": 1, "source": "m4serve", "sampled_at": now,
                "instance": {"id": self.instance, "boot_time": self.boot, "server": "m4serve (strata-m4)", "state": "ready"},
                "model": {"id": self.model_id, "loaded": True, "context": self.context},
                "work": {"slots_requested": S, "slots_effective": S, "active": len(act), "queued": len(queue),
                         "waiting": len(queue), "outside_slots": 0, "groups": 1, "requests_omitted": max(0, len(requests) - 50),
                         "max_queue": self.max_queue, "admission_limit": S + self.max_queue,
                         "admission_reserved": len(act) + len(queue), "admission_rejected": self.rejected,
                         "queue_timeout_s": self.queue_timeout, "queue_timeouts": self.timeouts,
                         "client_disconnect_cancels": self.disconnects,
                         "oldest_queue_s": round(max((now - r["queued_at"] for r in queue), default=0), 2),
                         "slots": slots, "requests": requests[:50], **self._batch(now)},
                "rates": {"answer_tok_s": round(answer[0], 1) if answer[0] else None,
                          "answer_window_s": round(answer[1], 2) if answer[1] else None,
                          "prefill_tok_s": round(self.last_prefill, 1) if self.last_prefill else None,
                          "output_tokens_server_counter": self.totals["output_tokens"] + sum(c["generated"] for c in act)},
                "topology": {"mode": "single", "selected_devices": [d["index"] for d in self.devices if d.get("index") is not None][:1],
                             "stages": [], "source": "m4serve"},
                "devices": list(self.devices),
                "ram": self._ram(),
                "recent": list(self.recent),
                "totals": dict(self.totals),
                "activity": {"last_seq": self.seq, "events": list(self.events)[-32:]},
                "quality": {"request_correlation": "reported", "per_device": "reported"},
                "m4": {"last_prompt": dict(self.last_prompt) if self.last_prompt else None},
            }

    def status(self) -> dict:
        with self.lock:
            busy = bool(self.active)
        return {"service": "m4serve (strata-m4)", "model": self.model_id, "loaded": True, "started": self.boot,
                "cache_max_tokens": self.context, "busy": busy, "concurrency": {"serving": self.slots_n}}


# ---------------------------------------------------------------------------------------------- HTTP
class State:
    engine: Engine
    chat: Chat
    vision: Vision | None
    model_name: str
    api_key: str | None
    defaults: dict
    tele: Telemetry


def make_handler(S: State):
    gate = threading.Lock()

    class Busy(Exception):
        def __init__(self, code: int, message: str):
            super().__init__(message)
            self.code, self.message = code, message

    def run(req: dict, on_tokens, cancelled) -> dict:
        """One engine request behind the line (--queue-timeout s at most), with its telemetry transitions."""
        rid = S.tele.queued(len(req.get("ids") or []))
        st, outcome = None, "error"

        def tok(ids_):
            S.tele.tokens(rid, len(ids_))
            on_tokens(ids_)

        def prog(read, total, reused):
            S.tele.progress(rid, read, total, reused)
        try:
            if S.engine.parallel > 1:   # the engine keeps the line: a request waits there for a slot
                try:
                    st = S.engine.generate(req, tok, cancelled, prog, S.tele.window,
                                           on_admit=lambda slot, reused: S.tele.admitted(rid, slot),
                                           admit_timeout=S.tele.queue_timeout)
                except QueueTimeout:
                    outcome = "timeout"
                    raise Busy(503, f"waited {S.tele.queue_timeout:.0f} s in the queue; the server is busy")
            else:
                if not gate.acquire(timeout=S.tele.queue_timeout):
                    outcome = "timeout"
                    raise Busy(503, f"waited {S.tele.queue_timeout:.0f} s in the queue; the server is busy")
                try:
                    S.tele.admitted(rid)
                    st = S.engine.generate(req, tok, cancelled, prog, S.tele.window)
                finally:
                    gate.release()
            fin = st.get("finish")
            outcome = "stop" if fin == "stop" else "length" if fin in ("length", "context") else \
                "disconnect" if cancelled() else "cancel" if fin == "abort" else "error"
            return st
        finally:
            S.tele.finished(rid, st, outcome)
            if st:
                log(request_line(rid, st, outcome, req))

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _json(self, code: int, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _auth(self) -> bool:
            if not S.api_key:
                return True
            if self.headers.get("Authorization", "") == "Bearer " + S.api_key or self.headers.get("x-api-key", "") == S.api_key:
                return True
            self._json(401, {"error": {"message": "invalid API key"}})
            return False

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                return self._json(200, {"status": "ok", "engine": S.engine.info})
            if not self._auth():
                return
            if self.path == "/v1/telemetry":
                t0 = time.time()
                snap = S.tele.snapshot()
                self._json(200, snap)
                if time.time() - t0 > 1.0:   # the monitor gives up after 3 s
                    log(f"slow telemetry answer: {time.time() - t0:.1f} s")
                return
            if self.path == "/v1/status":
                return self._json(200, S.tele.status())
            if self.path == "/v1/models":
                ctx = S.engine.info.get("ctx")
                return self._json(200, {"object": "list", "data": [{"id": S.model_name, "object": "model", "owned_by": "local",
                                                                    "max_model_len": ctx, "context_length": ctx,
                                                                    "context_window": ctx}]})
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self._auth():
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                return self._json(400, {"error": {"message": f"bad request: {e}"}})
            if self.path == "/v1/chat/completions":
                return self._chat(body)
            if self.path == "/v1/completions":
                return self._completion(body)
            log(f"404: POST {self.path} (this server speaks /v1/chat/completions and /v1/completions)")
            self._json(404, {"error": {"message": f"not found: {self.path}; use /v1/chat/completions"}})

        def _full(self) -> bool:
            """--max-queue requests waiting already: refused at once (503), as Strata's admission does."""
            if S.engine.parallel > 1:
                full = S.tele.inflight() >= S.engine.parallel + S.tele.max_queue
            else:
                full = gate.locked() and S.tele.waiting() >= S.tele.max_queue
            if full:
                S.tele.reject()
                self._json(503, {"error": {"message": f"server busy: {S.tele.max_queue} requests are waiting",
                                           "type": "server_error", "code": "queue_full"}})
                return True
            return False

        def _too_long(self, n: int) -> bool:
            """A prompt that does not fit the engine's context is refused the way OpenAI does it (agents read the code
            and compact their history) - never answered with an empty 'length'."""
            ctx = int(S.engine.info.get("ctx") or 0)
            if ctx and n + 32 >= ctx:
                log(f"400: prompt of {n} tokens, context {ctx}")
                self._json(400, {"error": {"message": f"This model's maximum context length is {ctx} tokens. However, your "
                                                      f"messages resulted in {n} tokens. Please reduce the length of the messages.",
                                           "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}})
                return True
            return False

        # -- the engine request
        def _gen_req(self, body: dict, ids: list[int], embd, key: str) -> dict:
            d = S.defaults
            ctx = int(S.engine.info.get("ctx") or 0)
            auto = d["max_tokens"] or (max(256, min(32768, ctx - len(ids) - 64)) if ctx else 4096)
            think = effort_of(body, S.chat.default_effort)
            temp = d["temperature"]
            if temp is None:   # Mistral's advice: 0.7 when the model thinks, low (0.0 .. 0.7 by task) when it does not
                temp = 0.7 if think == "high" else 0.1
            limit = body.get("max_tokens") or body.get("max_completion_tokens")
            if body.get("tools") and S.chat.ids.get("[TOOL_CALLS]") is not None and d.get("tool_temperature", -1) >= 0:
                r_tool = {"tool_start": S.chat.ids["[TOOL_CALLS]"], "tool_temp": float(d["tool_temperature"])}
            else:
                r_tool = {}
            r = {"ids": ids, "max_new": int(limit or auto), "_think": think, "_limit_client": bool(limit), **r_tool,
                 "temp": float(body["temperature"]) if body.get("temperature") is not None else temp,
                 "top_p": float(body.get("top_p", d["top_p"])),
                 "top_k": int(body.get("top_k", 0)), "min_p": float(body.get("min_p", 0.0)),
                 "stop_ids": [S.chat.eos], "seed": int(body.get("seed") or 0)}
            if "tool_temp" in r:   # never above the request's own
                r["tool_temp"] = min(r["tool_temp"], r["temp"])
                if r["tool_temp"] >= r["temp"]:
                    del r["tool_start"], r["tool_temp"]
            if "spec" in body:
                r["spec"] = int(body["spec"])
            if embd:
                r["embd"], r["embd_key"] = embd, key
            return r

        def _chat(self, body: dict):
            try:
                ids, embd, key = S.chat.encode(body, S.vision)
            except Exception as e:
                log(f"400 on /v1/chat/completions: {type(e).__name__}: {e}")
                return self._json(400, {"error": {"message": f"{type(e).__name__}: {e}", "type": "invalid_request_error"}})
            if self._too_long(len(ids)) or self._full():
                return
            req = self._gen_req(body, ids, embd, key)
            stream = bool(body.get("stream"))
            rid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            out: list[int] = []
            if not stream:
                try:
                    st = run(req, out.extend, lambda: False)
                except Busy as b:
                    return self._json(b.code, {"error": {"message": b.message, "type": "server_error", "code": "queue_timeout"}})
                content, reasoning, calls = S.chat.split([t for t in out if t != S.chat.eos], st.get("finish"), body.get("tools"))
                msg = {"role": "assistant", "content": content or (None if calls else "")}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                if calls:
                    msg["tool_calls"] = calls
                finish = "tool_calls" if calls else ("stop" if st.get("finish") == "stop" else "length")
                return self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": S.model_name,
                                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                        "usage": {"prompt_tokens": st.get("n_prompt", 0), "completion_tokens": st.get("n_gen", 0),
                                                  "total_tokens": st.get("n_prompt", 0) + st.get("n_gen", 0)},
                                        "timings": st})
            # streaming: content and reasoning deltas as the windows arrive; tool calls once complete
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            gone = [False]
            sent = {"content": "", "reasoning": ""}

            def send(obj):
                try:
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()
                except Exception:
                    gone[0] = True

            def chunk(delta, finish=None):
                send({"id": rid, "object": "chat.completion.chunk", "created": created, "model": S.model_name,
                      "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

            chunk({"role": "assistant"})
            tc = S.chat.ids["[TOOL_CALLS]"]

            def on_tokens(ids_):
                out.extend(ids_)
                toks = [t for t in out if t != S.chat.eos]
                if tc in toks:   # a tool call is being written: hold until the end
                    toks = toks[: toks.index(tc)]
                content, reasoning, _ = S.chat.split(toks)
                # hold back a trailing replacement char (an unfinished UTF-8 sequence)
                content, reasoning = content.rstrip("�"), reasoning.rstrip("�")
                if body.get("tools") and may_be_bare_call(content):   # maybe a call as plain JSON: decided at the end
                    content = sent["content"]
                if reasoning.startswith(sent["reasoning"]) and len(reasoning) > len(sent["reasoning"]):
                    chunk({"reasoning_content": reasoning[len(sent["reasoning"]):]})
                    sent["reasoning"] = reasoning
                if content.startswith(sent["content"]) and len(content) > len(sent["content"]):
                    chunk({"content": content[len(sent["content"]):]})
                    sent["content"] = content

            try:
                st = run(req, on_tokens, lambda: gone[0])
            except Busy as b:
                send({"error": {"message": b.message, "type": "server_error", "code": "queue_timeout"}})
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
                return
            content, reasoning, calls = S.chat.split([t for t in out if t != S.chat.eos], st.get("finish"), body.get("tools"))
            if content.startswith(sent["content"]) and len(content) > len(sent["content"]):
                chunk({"content": content[len(sent["content"]):]})
            for i, c in enumerate(calls):
                chunk({"tool_calls": [{"index": i, **c}]})
            finish = "tool_calls" if calls else ("stop" if st.get("finish") == "stop" else "length")
            chunk({}, finish)
            send({"id": rid, "object": "chat.completion.chunk", "created": created, "model": S.model_name, "choices": [],
                  "usage": {"prompt_tokens": st.get("n_prompt", 0), "completion_tokens": st.get("n_gen", 0)}, "timings": st})
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass

        def _completion(self, body: dict):
            prompt = body.get("prompt", "")
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
                ids = prompt
            else:
                ids = S.chat.tok.encode(str(prompt), bos=True, eos=False)
            if self._too_long(len(ids)) or self._full():
                return
            req = self._gen_req(body, ids, None, "")
            out: list[int] = []
            try:
                st = run(req, out.extend, lambda: False)
            except Busy as b:
                return self._json(b.code, {"error": {"message": b.message, "type": "server_error", "code": "queue_timeout"}})
            text = S.chat.tok.decode([t for t in out if t != S.chat.eos], special_token_policy=SpecialTokenPolicy.IGNORE)
            self._json(200, {"id": "cmpl-" + uuid.uuid4().hex[:24], "object": "text_completion", "created": int(time.time()),
                             "model": S.model_name, "choices": [{"index": 0, "text": text,
                                                                 "finish_reason": "stop" if st.get("finish") == "stop" else "length"}],
                             "usage": {"prompt_tokens": st.get("n_prompt", 0), "completion_tokens": st.get("n_gen", 0)},
                             "timings": st})

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "strata-m4"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="mistralai/Mistral-Small-4-119B-2603",
                    help="tekken.json, or a Hugging Face repo id to fetch it from")
    ap.add_argument("--engine-args", default="", help="extra strata-m4 options, e.g. \"--ctx 65536 --spec 6\"")
    ap.add_argument("--mmproj", default="", help="the model's mmproj GGUF (enables images)")
    ap.add_argument("--vision", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "m4-vision"))
    ap.add_argument("--vision-cpu", action="store_true", help="run the image encoder on the CPU (saves ~1.5 GiB VRAM)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("M4_API_KEY") or os.environ.get("STRATA_API_KEY"),
                    help="require this key (Authorization: Bearer ... or x-api-key); default $M4_API_KEY, else "
                         "$STRATA_API_KEY - the variable `strata monitor` sends by itself; 'none' = no key")
    ap.add_argument("--model-name", default="mistral-small-4")
    ap.add_argument("--system-prompt", default="auto",
                    help="for requests without a system message: auto (Mistral's SYSTEM_PROMPT.txt from the --tokenizer "
                         "repo), none, or a file")
    ap.add_argument("--parallel", type=int, default=None,
                    help="conversations at once (the engine's --parallel; 1-8; overrides one in --engine-args)")
    ap.add_argument("--max-queue", type=int, default=16, help="requests that may wait for the engine (more: 503 at once)")
    ap.add_argument("--queue-timeout", type=float, default=900.0, help="seconds a request may wait for the engine (then 503)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="answer limit when a request sets none (default 0: the rest of the context, at most 32768)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="for requests that name none (default: 0.7 when the model thinks, else 0.1 - Mistral's advice)")
    ap.add_argument("--tool-temperature", type=float, default=0.1,
                    help="temperature inside a tool call, from [TOOL_CALLS] on (default 0.1; -1: the request's own)")
    ap.add_argument("--reasoning-effort", choices=("none", "high"), default=None,
                    help="thinking for requests that name none (default: the model's own default)")
    ap.add_argument("--top-p", type=float, default=1.0)
    a = ap.parse_args()

    S = State()
    S.chat = Chat(a.tokenizer)
    S.chat.default_effort = a.reasoning_effort
    S.chat.default_system = load_default_system_prompt(
        a.system_prompt, a.tokenizer if not os.path.exists(a.tokenizer) else "mistralai/Mistral-Small-4-119B-2603")
    S.vision = Vision(a.vision, a.mmproj, a.model, a.vision_cpu) if a.mmproj else None
    eargs = shlex.split(a.engine_args)
    if a.parallel is not None:
        eargs += ["--parallel", str(max(1, min(8, a.parallel)))]
    S.engine = Engine(a.engine, a.model, eargs)
    S.model_name, S.api_key = a.model_name, (None if (a.api_key or "").strip().lower() in ("", "none") else a.api_key.strip())
    log("API key:", "required" if S.api_key else "none (every client on the network may use this server)")
    S.tele = Telemetry(a.model_name + " (" + os.path.basename(a.model) + ")")
    S.tele.context = S.engine.info.get("ctx")
    S.tele.slots_n = S.engine.parallel
    S.tele.max_queue, S.tele.queue_timeout = max(0, a.max_queue), max(1.0, a.queue_timeout)
    S.defaults = {"max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p,
                  "tool_temperature": a.tool_temperature}
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(S))

    def _stop(signum, frame):   # SIGINT and SIGTERM both end the server cleanly (a shell's background job ignores
        raise KeyboardInterrupt  # SIGINT unless the program sets its own handler: the "zombie" m4serve of AUFTRAG-18)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    log(f"listening on http://{a.host}:{a.port}/v1 (model name '{a.model_name}'"
        + (f", {S.engine.parallel} requests at once)" if S.engine.parallel > 1 else ")"))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        t0 = time.time()
        try:
            S.engine.p.stdin.write("QUIT\n")
            S.engine.p.stdin.flush()
            S.engine.p.stdin.close()   # end of input as well: the engine's line reader lets go of stdin
            S.engine.p.wait(timeout=120)   # the engine logs its own release steps; let it finish
            log(f"engine ended in {time.time() - t0:.1f} s")
        except Exception:
            S.engine.p.kill()
            log(f"engine did not end in {time.time() - t0:.0f} s; killed")


if __name__ == "__main__":
    main()
