#!/usr/bin/env bash
# sanity.sh - m4serve end to end: start it on a spare port, ask three short questions (greedy), stop it.
#   ./sanity.sh [--port 8091]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/m4.env"
PORT=8091
[[ "${1:-}" == "--port" ]] && PORT="$2"
LOG="$HERE/logs/sanity-$(date +%Y%m%d-%H%M%S).log"
"$HERE/venv/bin/python" "$HERE/m4serve.py" --engine "$HERE/bin/strata-m4" --model "$M4_MODEL" \
  --tokenizer "$M4_TOKENIZER" --engine-args "--ctx 16384 --spec 6 --heat $HERE/mistral4.m4heat" \
  --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
PID=$!
for i in $(seq 1 180); do
  curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "m4serve exited during start:"; tail -n 40 "$LOG"; exit 1; }
  sleep 2
done
ask() {
  local effort="$1" q="$2"
  echo "--- Q ($effort): $q"
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d "$(
    "$HERE/venv/bin/python" -c 'import json,sys; print(json.dumps({"model":"m","temperature":0,"max_tokens":400,
      "reasoning_effort":sys.argv[1] if sys.argv[1]!="default" else None,
      "messages":[{"role":"user","content":sys.argv[2]}]}))' "$effort" "$q")" |
    "$HERE/venv/bin/python" -c 'import json,sys; r=json.load(sys.stdin)
if "error" in r: print("ERROR:", r["error"]); sys.exit()
m=r["choices"][0]["message"]; t=r.get("timings",{})
if m.get("reasoning_content"): print("[reasoning]", m["reasoning_content"][:600])
print("[answer]", m.get("content")); print("[timings]", json.dumps(t))'
}
ask default "What is the capital of France? Answer in one word."
ask default "Schreibe einen kurzen Satz über den Herbst in Norddeutschland."
ask high "What is 17 * 23? Think step by step, then give the number."
kill -INT $PID 2>/dev/null; sleep 5; kill $PID 2>/dev/null
echo "--- server log (last 30 lines): $LOG"
tail -n 30 "$LOG"
