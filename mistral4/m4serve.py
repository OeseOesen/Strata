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
One request runs at a time (the engine keeps one prompt cache); others wait in line.
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
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

try:  # private helpers of mistral-common; the server falls back to plain text when they move
    from mistral_common.experimental.tools import _decode_tool_calls, _split_content_and_tool_calls
except Exception:  # pragma: no cover
    _decode_tool_calls = _split_content_and_tool_calls = None


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

    def generate(self, req: dict, on_tokens, cancelled) -> dict:
        """Runs one GEN; on_tokens(list[int]) per accepted window; returns the DONE stats."""
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
            elif line.startswith("DONE"):
                return json.loads(line[5:] or "{}")
            elif line.startswith("ERR"):
                raise RuntimeError(line[4:].strip())


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
    """What agents send, in the shape Mistral's validator accepts: every system / developer message merged into one
    leading system message; tool call ids as 9 letters / digits (the same mapping for a call and its result); a tool
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


class Chat:
    def __init__(self, tokenizer: str):
        if os.path.exists(tokenizer):
            self.mt = MistralTokenizer.from_file(tokenizer)
        else:
            self.mt = MistralTokenizer.from_hf_hub(tokenizer)
        self.tok = self.mt.instruct_tokenizer.tokenizer
        self.eos = self.tok.eos_id
        self.ids = {}
        for name in ("[THINK]", "[/THINK]", "[TOOL_CALLS]", "[IMG]", "[IMG_BREAK]", "[IMG_END]"):
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
        if body.get("reasoning_effort") is not None:
            kw["reasoning_effort"] = body["reasoning_effort"]
        msgs, tools = normalize_messages(messages), normalize_tools(body.get("tools"))

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

    def split(self, ids: list[int]):
        """-> (content, reasoning, tool_calls)"""
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
        if tc is not None and tc in rest and _split_content_and_tool_calls:
            content_ids, call_ids = _split_content_and_tool_calls(rest, tc)
            try:
                for c in _decode_tool_calls(call_ids, self.tok):
                    calls.append({"id": c.id if getattr(c, "id", None) and c.id != "null" else "call_" + uuid.uuid4().hex[:9],
                                  "type": "function",
                                  "function": {"name": c.function.name, "arguments": c.function.arguments
                                               if isinstance(c.function.arguments, str) else json.dumps(c.function.arguments)}})
                rest = content_ids
            except Exception as e:  # malformed call: leave it as text
                log("tool call not parsed:", e)
        dec = lambda x: self.tok.decode(x, special_token_policy=SpecialTokenPolicy.IGNORE) if x else ""
        return dec(rest), dec(reasoning), calls


# ---------------------------------------------------------------------------------------------- telemetry
class Telemetry:
    """What `strata monitor` reads (GET /v1/telemetry, schema 1; see terminal/strata_cli/client.py sanitize_telemetry):
    the one slot, the queue, request transitions, rates, the GPU (nvidia-smi) and the host's RAM.  Memory in bytes."""

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
        self.cur: dict | None = None              # the request on the engine
        self.recent = collections.deque(maxlen=50)
        self.totals = {"since": self.boot, "requests": 0, "prompt_tokens": 0, "output_tokens": 0}
        self.last_answer = None
        self.last_prefill = None
        self.devices: list[dict] = []
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

    def admitted(self, rid: str):
        with self.lock:
            row = self.queue.pop(rid, None) or {"id": rid, "queued_at": time.time(), "prompt_tokens": None}
            now = time.time()
            self.cur = {**row, "state": "reading", "admitted_at": now, "generated": 0, "first_at": None}
            self._event(rid, "admitted")
            self._event(rid, "reading")

    def tokens(self, n: int):
        with self.lock:
            c = self.cur
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
            c = self.cur if self.cur and self.cur["id"] == rid else None
            now = time.time()
            st = st or {}
            event = {"stop": "finished", "length": "finished", "cancel": "cancelled", "disconnect": "disconnected"}.get(outcome, "error")
            self._event(rid, event, outcome)
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
                self.cur = None

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
            c = dict(self.cur) if self.cur else None
            queue = [dict(r) for r in self.queue.values()]
            if c:
                live_rate = (c["generated"] / (now - c["first_at"])) if c["first_at"] and now > c["first_at"] and c["generated"] > 1 else None
                slot = {"slot": 0, "group": None, "replica": None, "request_id": c["id"], "state": c["state"],
                        "held_tokens": None, "prompt_tokens": c.get("prompt_tokens"),
                        "prompt_read": c.get("prompt_tokens") if c["first_at"] else None, "generated": c["generated"],
                        "elapsed_s": round(now - c["admitted_at"], 2), "tok_s": round(live_rate, 1) if live_rate else None}
            else:
                live_rate = None
                slot = {"slot": 0, "state": "idle", "held_tokens": None}
            requests = [{"id": r["id"], "state": "queued", "queued_at": r["queued_at"], "admitted_at": None, "slot": None,
                         "group": None, "replica": None, "prompt_tokens": r.get("prompt_tokens"), "prompt_read": None,
                         "generated": None, "first_engine_token_s": None} for r in queue]
            if c:
                requests.insert(0, {"id": c["id"], "state": c["state"], "queued_at": c["queued_at"], "admitted_at": c["admitted_at"],
                                    "slot": 0, "group": None, "replica": None, "prompt_tokens": c.get("prompt_tokens"),
                                    "prompt_read": slot["prompt_read"], "generated": c["generated"],
                                    "first_engine_token_s": round(c["first_at"] - c["admitted_at"], 3) if c["first_at"] else None})
            answer = (live_rate, None) if live_rate else (self.last_answer or (None, None))
            return {
                "schema_version": 1, "source": "m4serve", "sampled_at": now,
                "instance": {"id": self.instance, "boot_time": self.boot, "server": "m4serve (strata-m4)", "state": "ready"},
                "model": {"id": self.model_id, "loaded": True, "context": self.context},
                "work": {"slots_requested": 1, "slots_effective": 1, "active": 1 if c else 0, "queued": len(queue),
                         "waiting": len(queue), "outside_slots": 0, "groups": None, "requests_omitted": 0,
                         "oldest_queue_s": round(max((now - r["queued_at"] for r in queue), default=0), 2),
                         "slots": [slot], "requests": requests[:50]},
                "rates": {"answer_tok_s": round(answer[0], 1) if answer[0] else None,
                          "answer_window_s": round(answer[1], 2) if answer[1] else None,
                          "prefill_tok_s": round(self.last_prefill, 1) if self.last_prefill else None,
                          "output_tokens_server_counter": self.totals["output_tokens"] + (c["generated"] if c else 0)},
                "topology": {"mode": "single", "selected_devices": [d["index"] for d in self.devices if d.get("index") is not None][:1],
                             "stages": [], "source": "m4serve"},
                "devices": list(self.devices),
                "ram": self._ram(),
                "recent": list(self.recent),
                "totals": dict(self.totals),
                "activity": {"last_seq": self.seq, "events": list(self.events)[-32:]},
                "quality": {"request_correlation": "reported", "per_device": "reported"},
            }

    def status(self) -> dict:
        with self.lock:
            busy = self.cur is not None
        return {"service": "m4serve (strata-m4)", "model": self.model_id, "loaded": True, "started": self.boot,
                "cache_max_tokens": self.context, "busy": busy, "concurrency": {"serving": 1}}


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

    def run(req: dict, on_tokens, cancelled) -> dict:
        """One engine request behind the line, with its telemetry transitions."""
        rid = S.tele.queued(len(req.get("ids") or []))
        st, outcome = None, "error"
        try:
            with gate:
                S.tele.admitted(rid)

                def tok(ids_):
                    S.tele.tokens(len(ids_))
                    on_tokens(ids_)
                st = S.engine.generate(req, tok, cancelled)
            fin = st.get("finish")
            outcome = "stop" if fin == "stop" else "length" if fin in ("length", "context") else \
                "disconnect" if cancelled() else "cancel" if fin == "abort" else "error"
            return st
        finally:
            S.tele.finished(rid, st, outcome)

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
            if self.headers.get("Authorization", "") == "Bearer " + S.api_key:
                return True
            self._json(401, {"error": {"message": "invalid API key"}})
            return False

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                return self._json(200, {"status": "ok", "engine": S.engine.info})
            if not self._auth():
                return
            if self.path == "/v1/telemetry":
                return self._json(200, S.tele.snapshot())
            if self.path == "/v1/status":
                return self._json(200, S.tele.status())
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": S.model_name, "object": "model", "owned_by": "local"}]})
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

        # -- the engine request
        def _gen_req(self, body: dict, ids: list[int], embd, key: str) -> dict:
            d = S.defaults
            r = {"ids": ids, "max_new": int(body.get("max_tokens") or body.get("max_completion_tokens") or d["max_tokens"]),
                 "temp": float(body.get("temperature", d["temperature"])), "top_p": float(body.get("top_p", d["top_p"])),
                 "top_k": int(body.get("top_k", 0)), "min_p": float(body.get("min_p", 0.0)),
                 "stop_ids": [S.chat.eos], "seed": int(body.get("seed") or 0)}
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
            req = self._gen_req(body, ids, embd, key)
            stream = bool(body.get("stream"))
            rid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            out: list[int] = []
            if not stream:
                st = run(req, out.extend, lambda: False)
                content, reasoning, calls = S.chat.split([t for t in out if t != S.chat.eos])
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
                if reasoning.startswith(sent["reasoning"]) and len(reasoning) > len(sent["reasoning"]):
                    chunk({"reasoning_content": reasoning[len(sent["reasoning"]):]})
                    sent["reasoning"] = reasoning
                if content.startswith(sent["content"]) and len(content) > len(sent["content"]):
                    chunk({"content": content[len(sent["content"]):]})
                    sent["content"] = content

            st = run(req, on_tokens, lambda: gone[0])
            content, reasoning, calls = S.chat.split([t for t in out if t != S.chat.eos])
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
            req = self._gen_req(body, ids, None, "")
            out: list[int] = []
            st = run(req, out.extend, lambda: False)
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
    ap.add_argument("--api-key", default=os.environ.get("M4_API_KEY"))
    ap.add_argument("--model-name", default="mistral-small-4")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top-p", type=float, default=1.0)
    a = ap.parse_args()

    S = State()
    S.chat = Chat(a.tokenizer)
    S.vision = Vision(a.vision, a.mmproj, a.model, a.vision_cpu) if a.mmproj else None
    S.engine = Engine(a.engine, a.model, shlex.split(a.engine_args))
    S.model_name, S.api_key = a.model_name, a.api_key
    S.tele = Telemetry(a.model_name + " (" + os.path.basename(a.model) + ")")
    S.tele.context = S.engine.info.get("ctx")
    S.defaults = {"max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p}
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(S))
    log(f"listening on http://{a.host}:{a.port}/v1 (model name '{a.model_name}')")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            S.engine.p.stdin.write("QUIT\n")
            S.engine.p.stdin.flush()
            S.engine.p.wait(timeout=30)
        except Exception:
            S.engine.p.kill()


if __name__ == "__main__":
    main()
