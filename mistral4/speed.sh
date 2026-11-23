#!/usr/bin/env bash
# speed.sh - decode speed on realistic requests through the API, with and without n-gram speculation.
#   ./speed.sh [--port 8092] [--engine-args "..."]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/m4.env"
PORT=8092
EARGS="--ctx 16384 --spec 6"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --engine-args) EARGS="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
LOG="$HERE/logs/speed-$(date +%Y%m%d-%H%M%S).log"
"$HERE/venv/bin/python" "$HERE/m4serve.py" --engine "$HERE/bin/strata-m4" --model "$M4_MODEL" \
  --tokenizer "$M4_TOKENIZER" --engine-args "$EARGS --heat $HERE/mistral4.m4heat" \
  --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
PID=$!
for i in $(seq 1 180); do
  curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "m4serve exited during start:"; tail -n 40 "$LOG"; exit 1; }
  sleep 2
done
"$HERE/venv/bin/python" - "$PORT" <<'PY'
import json, sys, time, urllib.request
port = sys.argv[1]
code = "\n".join(f"def step_{i}(values, factor):\n    total = 0\n    for v in values:\n        total += v * factor + {i}\n    return total\n" for i in range(12))
tasks = [
    ("code-edit", None, f"Here is a Python module:\n\n{code}\nReturn the complete module unchanged except that every function gets a one-line docstring. Output only the code."),
    ("reasoning", "high", "A train leaves at 09:40 and arrives at 13:05 after two stops of 7 minutes each. How many minutes does it actually drive? Think it through."),
    ("prose", None, "Write about 200 words explaining NUMA memory placement to a system administrator."),
]
def ask(prompt, effort, spec):
    body = {"model": "m", "temperature": 0, "max_tokens": 400, "messages": [{"role": "user", "content": prompt}], "spec": spec}
    if effort: body["reasoning_effort"] = effort
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    t = r.get("timings", {})
    return t, time.time() - t0
print(f"{'task':10} {'spec':>4} {'tokens':>6} {'decode tok/s':>12} {'drafted':>7} {'accepted':>8} {'windows':>7}")
for name, effort, prompt in tasks:
    for spec in (6, 0):   # drafted run first: the n-gram table learns each answer at its end
        t, wall = ask(prompt, effort, spec)
        n, ms = t.get("n_gen", 0), t.get("gen_ms", 1)
        print(f"{name:10} {spec:>4} {n:>6} {n / (ms / 1000):>12.2f} {t.get('drafted', 0):>7} {t.get('accepted', 0):>8} {t.get('windows', 0):>7}", flush=True)
PY
kill -INT $PID 2>/dev/null; sleep 5; kill $PID 2>/dev/null
echo "server log: $LOG"
