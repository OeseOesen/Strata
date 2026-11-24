#!/usr/bin/env python3
"""parallel_check.py - checks a running m4serve started with --parallel N.

    python3 parallel_check.py --url http://127.0.0.1:8095 [--n 4] [--max-tokens 200]

1. the N prompts one after another (greedy), 2. the same N at once.  Reports whether each answer is the same both
times (greedy decoding: the rows of a shared window are computed independently, so they should match), the slots
the telemetry showed busy, and the output rate of both runs.  The key comes from $M4_API_KEY or $STRATA_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request

PROMPTS = [
    "List the planets of the solar system with one fact each.",
    "Explain in a short paragraph how a hash table handles collisions.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Translate into German: 'The train was late, so we walked the last two kilometres in the rain.'",
    "Give five tips for keeping a sourdough starter healthy.",
    "What is the difference between TCP and UDP? Answer in four bullet points.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Name three sorting algorithms and their average time complexity.",
]


def post(url: str, key: str | None, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url: str, key: str | None, path: str) -> dict:
    req = urllib.request.Request(url + path, headers={"Authorization": "Bearer " + key} if key else {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8095")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()
    key = os.environ.get("M4_API_KEY") or os.environ.get("STRATA_API_KEY")
    info = get(a.url, None, "/health").get("engine", {})
    par = int(info.get("parallel") or 1)
    print(f"server: parallel {par}, ctx {info.get('ctx')}, eagle {info.get('eagle')}")
    if par < 2:
        print("note: the server runs one request at a time (start it with --parallel N)")
    prompts = (PROMPTS * 2)[: a.n]

    def body(p):
        return {"model": "m4", "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": a.max_tokens,
                "reasoning_effort": "none"}

    def one(i, p, out):
        t0 = time.time()
        r = post(a.url, key, body(p))
        out[i] = (r["choices"][0]["message"].get("content") or "", r["usage"]["completion_tokens"], time.time() - t0,
                  r.get("timings", {}))

    seq: dict = {}
    t0 = time.time()
    for i, p in enumerate(prompts):
        one(i, p, seq)
        print(f"  alone   #{i}: {seq[i][1]:4d} tokens in {seq[i][2]:5.1f} s (slot {seq[i][3].get('slot', '-')})")
    t_seq = time.time() - t0

    busy_max, stop = [0], [False]

    def watch():
        while not stop[0]:
            try:
                w = get(a.url, key, "/v1/telemetry")["work"]
                busy_max[0] = max(busy_max[0], int(w.get("active") or 0))
            except Exception:
                pass
            time.sleep(0.3)
    threading.Thread(target=watch, daemon=True).start()
    par_out: dict = {}
    th = [threading.Thread(target=one, args=(i, p, par_out)) for i, p in enumerate(prompts)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    t_par = time.time() - t0
    stop[0] = True
    same = 0
    for i in range(len(prompts)):
        s_txt, p_txt = seq[i][0], par_out[i][0]
        k = next((j for j in range(min(len(s_txt), len(p_txt))) if s_txt[j] != p_txt[j]), min(len(s_txt), len(p_txt)))
        ok = s_txt == p_txt
        same += ok
        print(f"  at once #{i}: {par_out[i][1]:4d} tokens in {par_out[i][2]:5.1f} s (slot {par_out[i][3].get('slot', '-')}) "
              + ("same answer" if ok else f"DIFFERENT after {k} of {len(s_txt)} chars"))
    n_seq, n_par = sum(v[1] for v in seq.values()), sum(v[1] for v in par_out.values())
    print(f"one after another: {n_seq} tokens in {t_seq:.1f} s = {n_seq / t_seq:.1f} tok/s")
    print(f"at once:           {n_par} tokens in {t_par:.1f} s = {n_par / t_par:.1f} tok/s  (x{(n_par / t_par) / (n_seq / t_seq):.2f}), "
          f"busy slots seen: {busy_max[0]}")
    print(f"PARALLEL-CHECK same {same}/{len(prompts)} busy-max {busy_max[0]} seq {n_seq / t_seq:.1f} par {n_par / t_par:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
