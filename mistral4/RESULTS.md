# Mistral Small 4 (119B) on one Tesla V100 — measured results

All numbers produced on a single Dell R740: Tesla V100-PCIe 32 GB, 2× Xeon
Gold 6154 (18C/36T each, dual-NUMA DDR4, 192 GB), CUDA 12.6, model
`mistralai/Mistral-Small-4-119B-2603` in UD-Q4_K_S GGUF. Every figure below
is reproducible with the scripts in this folder on the same box.

## Decode speed

| Configuration | tok/s |
|---|---|
| llama.cpp baseline (same GGUF, same GPU) | ~17 |
| strata-m4, NUMA `split`, no speculation | 35 |
| strata-m4 + n-gram speculation (`--spec 6`) | 44–72 by task |
| strata-m4 + EAGLE draft head | 55–72, **lossless** at greedy (52.9 % draft agreement) |
| Long context (50k tokens in), TC attention kernel | **43** (was 18 before the kernel; kernel itself 2.1–2.9× faster, selftest max diff < 1e-6) |

## Prefill / prompt streaming

| Configuration | tok/s |
|---|---|
| Default (experts resident/split, no streaming) | 270 |
| + `--pf-gemm` (f16 tensor-core expert GEMMs) | 318 |
| + `--pf-stream --chunk 2048` (experts DMA'd per prompt chunk) | 432 |
| + `--pf-stream --chunk 4096` | **711** |

A 50k-token document reads end-to-end in **153 s** (was 260 s), with all
9/9 needle facts still found and perplexity unchanged (5.3003 off vs 5.2969
on). Streaming is on by default now (`--chunk 4096` in `m4.env`).

Early streaming segfaulted in serve mode: the BOS token produces massive
activations that overflowed the new f16 paths (inf → NaN → router id
`0x7fffffff` → bad expert index). Fixed by per-row scaling around the f16
GEMMs and a host-side router-id check; the bench never caught it because it
uses random tokens without BOS.

## Per-head prompt attention (`--pf-attn fa`)

One kernel computes prompt attention per head (2.25× less work, no score
matrix in memory): +10 % on 8k prompts and **+42 % on 32k prompts** vs the
`gemm` default; PPL moved 0.13 % (5.2910 → 5.2978, gate ≤0.5 %), 9/9 facts.
On by default after the gate passed.

## 100k-token context

One request reads 97,073 tokens in **222.5 s (436 tok/s)** at production
settings (`--ctx 131072 --kv int8 --chunk 4096 --pf-attn fa`); all **3/3**
needle facts found at 10/50/90 % depth; follow-ups answer in 0.9 s via KV
reuse; decode at 97k holds ~21 tok/s with EAGLE.

## Serving behaviour

* Thinking is the default (`M4_REASONING=high`): requests without a
  preference get `high` plus Mistral's recommended sampling (0.7 thinking /
  0.1 plain); every request logs thinking, temperature, and the answer limit.
* Tool calls: arguments holding code with unescaped quotes are repaired
  (name-aware, only when the model ended its own answer); a bare-JSON answer
  matching exactly one offered tool is mapped back into a call; temperature
  drops to 0.1 inside `[TOOL_CALLS]` generation.
* Clean shutdown in **3.6 s**. Earlier builds hung 30–120 s on stop: the
  engine's stdin reader thread held the lock the quit path waited on (and
  `--selftest-pin` proved the 40 GiB pin/unpin was innocent at 4.6 s total).

## Quality gates

* Perplexity (wikitext-style probe): strata-m4 8.14–8.23 vs llama.cpp 8.06;
  at 16k positions 2.78 (t=0) / 2.85 (t=0.1).
* 50k long-context needle test: 9/9 facts, including depth 10 % (right after
  BOS-near positions — the exact regression zone of the f16 bug).
* Prefix/KV reuse: repeat questions on a resident document answer in 0.2 s.

## Operations

* `strata-m4` CLI: `launch`, `serve`, `stop`, `restart`, `status`, `monitor`,
  `logs`, `config`, `doctor`, `bench` — installed to `~/.local/bin`.
* `m4serve.py`: OpenAI-compatible API, tool calls, `reasoning_effort`
  (`medium`→`high` translated server-side), queue limit (16) + 900 s
  timeout, admission/group telemetry.
* `strata monitor` (upstream terminal UI) works live against
  `/v1/telemetry`; needs `textual==8.2.8` (installed by `install.sh`).

*Engine development was paired work: Claude Opus 5.5 (architecture) and an
autonomous agent (build/measure pipeline). Numbers are from this specific
rig — treat them as research notes, not benchmarks.*
