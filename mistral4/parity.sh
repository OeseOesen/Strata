#!/usr/bin/env bash
# parity.sh - is strata-m4 computing the same model as llama.cpp?  Perplexity of both on the same token ids.
#
#   ./parity.sh [--chunks N] [--text FILE] [--skip-llama] [--kv "f16 int8"] [--extra "strata-m4 options"]
#
# The text (default: llama.cpp's own docs, ~300k tokens of English prose, code and tables) is tokenized once with
# llama.cpp's tokenizer; both engines score the second half of every 512-token chunk (llama-perplexity's method, BOS at
# each chunk start).  With --kv f16 the two should agree to about 0.5 %; the int8 latent cache adds a little (expect
# < 1.5 %).  A difference of several percent or more means a bug, not rounding: send the log.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/m4.env"
TEXT=""
CHUNKS=10
DO_LLAMA=1
KVS="f16 int8"
EXTRA=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunks) CHUNKS="$2"; shift 2 ;;
    --text) TEXT="$2"; shift 2 ;;
    --skip-llama) DO_LLAMA=0; shift ;;
    --kv) KVS="$2"; shift 2 ;;
    --extra) EXTRA="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
WORK="$HERE/logs/parity-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$WORK"
if [[ -z "$TEXT" ]]; then
  TEXT="$WORK/corpus.txt"
  find "$HERE/src/llama.cpp/docs" "$HERE/src/llama.cpp/tools" -name "*.md" -print0 2>/dev/null | sort -z | xargs -0 cat > "$TEXT"
fi
echo "text: $TEXT ($(wc -c < "$TEXT") bytes), $CHUNKS chunks of 512"
"$M4_LLAMA/llama-tokenize" -m "$M4_MODEL" -f "$TEXT" --ids 2>/dev/null | tail -n 1 | tr -d '[],' > "$WORK/ids.txt"
# the first CHUNKS*512 ids (awk reads everything: no early-closed pipe, no SIGPIPE)
awk -v need=$((CHUNKS * 512)) '{ for (i = 1; i <= NF && n < need; ++i) { printf "%s ", $i; ++n } } END { print "" }' \
  "$WORK/ids.txt" > "$WORK/ids_cut.txt"
echo "$(wc -w < "$WORK/ids.txt") tokens in the text, $(wc -w < "$WORK/ids_cut.txt") used"

if [[ $DO_LLAMA -eq 1 ]]; then
  echo "== llama.cpp llama-perplexity (-c 512, --chunks $CHUNKS)"
  "$M4_LLAMA/llama-perplexity" -m "$M4_MODEL" -f "$TEXT" -c 512 --chunks "$CHUNKS" -ngl 99 --n-cpu-moe 24 -fa 1 \
    > "$WORK/llama.log" 2>&1
  echo "exit=$?"
  grep -E "Final estimate|^\[[0-9]" "$WORK/llama.log" | tail -n 3
fi
for KV in $KVS; do
  echo "== strata-m4 --ppl (--kv $KV)"
  ARGS="$(echo " $M4_ENGINE_ARGS " | sed 's/ --kv [a-z0-9]* / /; s/ --spec [0-9]* / /; s/ --ctx [0-9]* / /')"
  # shellcheck disable=SC2086
  "$HERE/bin/strata-m4" -m "$M4_MODEL" $ARGS --kv "$KV" --spec 0 --ctx 8192 --heat "$HERE/mistral4.m4heat" $EXTRA \
    --ppl "$WORK/ids_cut.txt" > "$WORK/m4-$KV.log" 2>&1
  echo "exit=$?"
  grep -E "^PPL|ppl chunk|error" "$WORK/m4-$KV.log" | tail -n 4
done
echo "logs: $WORK"
