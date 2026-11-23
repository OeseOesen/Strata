#!/usr/bin/env python3
"""m4serve - an OpenAI-compatible HTTP server in front of strata-m4 (Mistral Small 4).

    python3 m4serve.py --engine ./strata-m4 --model /models/Mistral-Small-4-Q4_K_M.gguf \
        --tokenizer /models/tekken.json [--mmproj mmproj.gguf --vision ./m4-vision] [--port 8080] \
        [--engine-args "--ctx 65536 --spec 6"]

Prompts are built by mistral-common (Mistral's own reference for the chat format: system prompt, tools,
[THINK] reasoning with reasoning_effort, images); the engine only sees token ids.  Answers are split back into
content, reasoning_content and tool_calls.

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions (stream or not), POST /v1/completions.
One request runs at a time (the engine keeps one prompt cache); others wait in line.
"""
from __future__ import annotations

import argparse
import base64
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
        req = ChatCompletionRequest.from_openai(messages, tools=body.get("tools"), **kw)
        tokd = self.mt.encode_chat_completion(req)
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


# ---------------------------------------------------------------------------------------------- HTTP
class State:
    engine: Engine
    chat: Chat
    vision: Vision | None
    model_name: str
    api_key: str | None
    defaults: dict


def make_handler(S: State):
    gate = threading.Lock()

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
            self._json(404, {"error": {"message": "not found"}})

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
                return self._json(400, {"error": {"message": str(e)}})
            req = self._gen_req(body, ids, embd, key)
            stream = bool(body.get("stream"))
            rid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            out: list[int] = []
            if not stream:
                with gate:
                    st = S.engine.generate(req, out.extend, lambda: False)
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

            with gate:
                st = S.engine.generate(req, on_tokens, lambda: gone[0])
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
            with gate:
                st = S.engine.generate(req, out.extend, lambda: False)
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
