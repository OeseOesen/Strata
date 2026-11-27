# ⚠️ This fork: `mistral4-engine` — **Mistral Small 4 (119B)** on a Strata-style engine

> **Fork of [Niko1221/Strata](https://github.com/Niko1221/Strata), research
> prototype — not a product or support channel.** This branch adds a
> standalone Mistral Small 4 engine (`strata-m4`) for old server hardware —
> one Tesla V100 32 GB + 2× Xeon Gold (dual-NUMA DDR4) — as preliminary work
> toward serving **Mistral Large** on modern rigs (Threadripper + ≥512 GB RAM
> + multi-GPU Blackwell).
>
> The upstream Strata app (Qwen/Hy4/Inkling, one-click install) is described
> in [README-upstream.md](README-upstream.md); nothing on this branch
> changes it.

## Measured on that one V100 box (all reproducible)

| | |
|---|---|
| Decode, warm, no speculation | **35 tok/s** (llama.cpp, same GGUF: ~17) |
| Decode, NUMA split + n-gram speculation | **44–72 tok/s** by task |
| Decode, EAGLE draft head | **55–72 tok/s**, verified **lossless** |
| Perplexity vs llama.cpp | 8.14–8.23 vs 8.06 · at 16k positions: 2.78 vs 2.80 |
| 50k-token document | **9/9 needle facts**; repeat questions via KV reuse: 0.2 s |
| Long-context decode (tensor-core attention) | **43 tok/s at 50k** (was 18; kernel 2.1–2.9×, selftest max diff < 1e-6) |
| Prompt streaming (experts DMA'd to GPU, f16 tensor-core GEMM) | **Prefill 270 → 702 tok/s**; 50k doc read 260 s → **153 s**, 9/9 facts, PPL unchanged |
| Per-head prompt attention (`--pf-attn fa`) | +10 % at 8k, **+42 % at 32k prompts**; PPL within 0.13 %, 9/9 facts |
| 100k-token context (97k read in one request) | **3/3 needle facts**; first read 222 s (436 tok/s), follow-ups 0.9 s via KV reuse |
| Thinking by default (`M4_REASONING=high`) | reasoning content + Mistral-recommended temperature (0.7 thinking / 0.1 plain), logged per request |
| Tool calls | repaired when args hold unescaped quotes; bare-JSON answers mapped back to calls; queue limit + timeout |
| Clean shutdown | **3.6 s** (stdin-lock deadlock in teardown found & fixed; `--selftest-pin` proves pinning was innocent) |
| Agent traffic (tools, tool ids, `reasoning_effort`) | OpenAI-format compatible; `medium`→`high` translated server-side |
| `strata-m4` CLI | launch / serve / stop / restart / status / monitor / logs / config / doctor; queue limit + timeout |
| `strata monitor` | works live against the engine via `/v1/telemetry` |

Architecture walkthrough (memory hierarchy, token flow, flags):
**[mistral4/README.md](mistral4/README.md)** (German). Scripts that produced
every number above: `mistral4/` — `install.sh`, `m4serve.py`, `parity.sh`,
`speed.sh`, `longctx.sh`, `m4-detcheck.sh`.

## Try it (Linux, CUDA 12)

```sh
bash mistral4/install.sh
strata-m4 launch                 # OpenAI-compatible server + CLI (stop/status/monitor/logs), see mistral4/README.md
```

*Engine development was paired work: Claude Opus 5.5 (planning) and an
autonomous agent (execution). All numbers measured on the rig described, not
simulated. Goal: port to Blackwell-class hardware — contributions from people
with that hardware welcome.*

[MIT license](LICENSE) · [Upstream Strata](README-upstream.md) ·
[Upstream repo](https://github.com/Niko1221/Strata)
