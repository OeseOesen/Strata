#!/usr/bin/env bash
# longctx.sh - long prompts through the API with the production settings (m4.env): three facts hidden at 10 %, 50 %
# and 90 % of a long document (llama.cpp's docs as filler), asked one by one (the document stays in the prefix cache),
# then a summary for the decode speed at that depth.  Sizes in characters (~3.5 per token).
#   ./longctx.sh [--port 8094] [--sizes "30000 90000 170000"] [--engine-args "..."]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/m4.env"
PORT=8094
SIZES="30000 90000 170000"
EARGS="$M4_ENGINE_ARGS"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --sizes) SIZES="$2"; shift 2 ;;
    --engine-args) EARGS="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
mkdir -p "$HERE/logs"
LOG="$HERE/logs/longctx-$(date +%Y%m%d-%H%M%S).log"
FILLER="$HERE/logs/longctx-filler.txt"
find "$HERE/src/llama.cpp/docs" -name "*.md" -print0 2>/dev/null | sort -z | xargs -0 cat > "$FILLER"
echo "filler: $(wc -c < "$FILLER") characters; engine args: $EARGS"
"$HERE/venv/bin/python" "$HERE/m4serve.py" --engine "$HERE/bin/strata-m4" --model "$M4_MODEL" \
  --tokenizer "$M4_TOKENIZER" --engine-args "$EARGS --heat $HERE/mistral4.m4heat" \
  --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
PID=$!
for i in $(seq 1 180); do
  curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "m4serve exited during start:"; tail -n 40 "$LOG"; exit 1; }
  sleep 2
done
"$HERE/venv/bin/python" - "$PORT" "$FILLER" $SIZES <<'PY'
import json, sys, time, urllib.request
port, filler, sizes = sys.argv[1], open(sys.argv[2], errors="replace").read(), [int(x) for x in sys.argv[3:]]
facts = [
    (0.10, "Note for the archive: the maintenance password of the cold-storage room is heron-7319.",
     "What is the maintenance password of the cold-storage room?", ["heron-7319"]),
    (0.50, "Remember: the backup tape for project Lindwurm is stored in cabinet 42-B.",
     "In which cabinet is the backup tape for project Lindwurm stored?", ["42-B", "42B"]),
    (0.90, "Recorded here for completeness: the server named Kranich was installed on 3 March 2019.",
     "On which date was the server named Kranich installed?", ["2019"]),
]
def ask(content, max_tokens):
    body = {"model": "m", "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    m = r["choices"][0]["message"]
    return (m.get("content") or "").strip(), r.get("timings", {}), time.time() - t0
total_ok = total = 0
for size in sizes:
    while len(filler) < size: filler += filler
    doc = filler[:size]
    for frac, fact, _, _ in sorted(facts, reverse=True):   # insert from the back so the earlier offsets stay valid
        cut = doc.rfind("\n", 0, int(size * frac)) + 1
        doc = doc[:cut] + "\n" + fact + "\n\n" + doc[cut:]
    head = "Here is a long document. Read it carefully; questions follow.\n\n<document>\n" + doc + "\n</document>\n\n"
    print(f"== document of {len(doc)} characters", flush=True)
    for frac, fact, q, want in facts:
        a, t, wall = ask(head + q + " Answer in one short sentence.", 120)
        ok = any(w.lower() in a.lower() for w in want)
        total_ok += ok; total += 1
        print(f"  [{int(frac*100):2d} %] {'OK  ' if ok else 'FAIL'} prompt {t.get('n_prompt',0)} tok ({t.get('n_reused',0)} reused) "
              f"in {t.get('prompt_ms',0)/1000:.1f} s = {(t.get('n_prompt',0)-t.get('n_reused',0))/max(1e-3,t.get('prompt_ms',1)/1000):.0f} tok/s | {a[:160]!r}",
              flush=True)
    a, t, wall = ask(head + "Summarize the document in about 150 words.", 300)
    print(f"  [summary] {t.get('n_gen',0)} tok at {t.get('n_gen',0)/max(1e-3,t.get('gen_ms',1)/1000):.1f} tok/s "
          f"(eagle {t.get('eagle_drafted',0)}/{t.get('eagle_accepted',0)}, n-gram {t.get('drafted',0)-t.get('eagle_drafted',0)}"
          f"/{t.get('accepted',0)-t.get('eagle_accepted',0)}) | {a[:400]!r}", flush=True)
print(f"facts found: {total_ok}/{total}")
PY
kill -INT $PID 2>/dev/null; sleep 5; kill $PID 2>/dev/null
echo "server log: $LOG"
grep -E "error|EAGLE head|attention temperature|experts:" "$LOG"
