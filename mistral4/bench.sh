#!/usr/bin/env bash
# bench.sh - strata-m4 against llama.cpp on the same GGUF: prompt and decode speed.
#
#   ./bench.sh [--pp 2048] [--tg 256] [--llama-ncmoe N] [--skip-llama] [--skip-m4]
#
# llama.cpp runs the usual way for a model larger than VRAM: attention and shared tensors on the GPU, the routed
# experts of the first N layers on the CPU (--n-cpu-moe), N chosen so that the rest fills the card (default: 24,
# override with --llama-ncmoe).  strata-m4 runs with the options in m4.env.  Both: one sequence, greedy, no drafts
# (strata-m4 --spec 0 in the bench mode), so the numbers compare the engines, not the drafter.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/m4.env"
PP=2048; TG=256; NCMOE=24; DO_LLAMA=1; DO_M4=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pp) PP="$2"; shift 2 ;;
    --tg) TG="$2"; shift 2 ;;
    --llama-ncmoe) NCMOE="$2"; shift 2 ;;
    --skip-llama) DO_LLAMA=0; shift ;;
    --skip-m4) DO_M4=0; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
OUT="$HERE/logs/bench-$(date +%Y%m%d-%H%M%S).txt"
mkdir -p "$HERE/logs"
{
  echo "# $(date -Is)  model: $M4_MODEL  pp=$PP tg=$TG"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader
  lscpu | grep -E "^(Model name|NUMA node[0-9]* CPU)"
} | tee "$OUT"
if [[ $DO_LLAMA -eq 1 ]]; then
  echo "== llama.cpp llama-bench (-ngl 99 --n-cpu-moe $NCMOE -fa 1 -t $(nproc --all))" | tee -a "$OUT"
  "$M4_LLAMA/llama-bench" -m "$M4_MODEL" -ngl 99 --n-cpu-moe "$NCMOE" -fa 1 -p "$PP" -n "$TG" -r 2 2>&1 | tee -a "$OUT" | grep -E "^\|" || true
fi
if [[ $DO_M4 -eq 1 ]]; then
  echo "== strata-m4 --bench ($M4_ENGINE_ARGS)" | tee -a "$OUT"
  # shellcheck disable=SC2086
  "$HERE/bin/strata-m4" -m "$M4_MODEL" $M4_ENGINE_ARGS --bench --bench-pp "$PP" --bench-tg "$TG" 2>&1 | tee -a "$OUT" | grep -E "bench:|experts:|pool|error" || true
  echo "(second run: the expert heat file from the first run now decides what sits in VRAM)" | tee -a "$OUT"
  # shellcheck disable=SC2086
  "$HERE/bin/strata-m4" -m "$M4_MODEL" $M4_ENGINE_ARGS --bench --bench-pp "$PP" --bench-tg "$TG" 2>&1 | tee -a "$OUT" | grep -E "bench:|error" || true
fi
echo "results: $OUT"
