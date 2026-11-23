// engine/src/mistral4/m4_kernels.cuh - strata-m4's own CUDA kernels: the parts of Mistral Small 4 that Strata's
// kernel library has no shape for (MLA attention over a compressed latent cache, the 128/top-4 router, the host
// coupling of the hybrid MoE).  Everything quantized (dense GEMVs, grouped experts, the head) runs on Strata's
// native kernels, which are llama.cpp's block formats bit for bit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace m4k {

constexpr int kLat = 256;      // kv_lora_rank
constexpr int kRope = 64;      // qk_rope_head_dim
constexpr int kKD = kLat + kRope;   // one cached key row: [latent | rope]
constexpr int kMaxHeads = 32;
constexpr int kAttnChunks = 64;     // decode attention: key ranges per query token (graph-stable grid)

enum KvMode : int { KV_I8 = 0, KV_F16 = 1 };

/// y[r] = x[r] / rms(x[r]) * w, rows of n (x and y may alias); strides in floats.
void rmsnorm(const float* x, const float* w, float* y, int n, int rows, int x_stride, int y_stride, float eps,
             cudaStream_t s);
/// x[i] += a[i]
void add_inplace(float* x, const float* a, size_t n, cudaStream_t s);

/// The YaRN rope table, host-computed exactly as ggml's CPU rope (ggml_rope_cache_init): for every position p in
/// [0, n_pos) and pair i in [0, n_rot/2): cos, sin (already multiplied by the magnitude scale).
void rope_table_host(float* out, int n_pos, int n_rot, float freq_base, float freq_scale, float ext_factor,
                     float attn_factor, float beta_fast, float beta_slow, int n_ctx_orig);

/// MLA, step 1 (per token t of the window):
///   kv latent = rmsnorm(kvp[t][0:256]) * kv_a_norm ; k_pe = rope(kvp[t][256:320], pos[t])
///   written to the layer's cache row pos[t] (int8 with a scale per half, or f16).
void mla_kv_write(const float* kvp, const float* kv_norm, const float* rope_tab, const int32_t* pos, int T,
                  int kv_mode, void* cache, float* scales, float eps, cudaStream_t s);
/// MLA, step 2: q_pe = rope(q[t][h][64:128]) written to Qf[t][h][256:320]; Qf[t][h][0:256] holds q_nope absorbed
/// through wk_b (a cuBLAS batched GEMM before this call).  Optional llama-4 style temperature scaling of the whole
/// row: s = log(floor(pos/floor_scale)+1)*temp_scale + 1 (temp_scale 0 = off; llama.cpp's form).
void mla_q_rope(const float* q, float* Qf, const float* rope_tab, const int32_t* pos, int T, int n_head,
                float temp_scale, float temp_floor, cudaStream_t s);
/// MLA, step 3 (decode / verify windows, T <= 16): O[t][h][0:256] = softmax_k(scale * Qf[t][h] . K[k]) V[k] over
/// cache rows k <= pos[t].  Split over kAttnChunks key ranges and merged; `part` is scratch of
/// mla_decode_part_floats(T, n_head) floats.
size_t mla_decode_part_floats(int T, int n_head);
/// impl: ATTN_TC (default) both products on the tensor cores (WMMA, f16 in, f32 accumulate; needs sm_70+), ATTN_SIMT
/// the plain CUDA-core kernel.
enum AttnImpl : int { ATTN_SIMT = 0, ATTN_TC = 1 };
void mla_attn_decode(const float* Qf, const void* cache, const float* scales, int kv_mode, const int32_t* pos, int T,
                     int n_head, float kq_scale, float* part, float* O, cudaStream_t s, int impl = ATTN_TC);
/// Prompt path helpers (cuBLAS does the products): the cache rows [0, n) as f16 K rows (ld 320), Qf as f16 scaled by
/// `scale`, the causal softmax over scores S (column c = token c / n_head; keys <= pos[t]) into f16 P.
void mla_cache_to_f16(const void* cache, const float* scales, int kv_mode, int n, __half* K16, cudaStream_t s);
void f32_to_f16_scaled(const float* x, __half* y, size_t n, float scale, cudaStream_t s);
/// Column c is query token c / n_head of the sub-chunk, at position pos0 + c / n_head: keys beyond it get 0.
void mla_softmax_causal(const float* S, __half* P, int n_kv, int cols, int n_head, int pos0, cudaStream_t s);

/// Router: probs = softmax(logits) (gating 1) or sigmoid (gating 2); selection on probs (+ bias); weights = probs of the
/// selected, normalized when `norm`, times `scale`.  ids/w: [T][k].
void router_topk(const float* logits, const float* bias, int n_expert, int k, int T, int gating, bool norm, float scale,
                 int32_t* ids, float* w, cudaStream_t s);

/// Groups the window's routes whose expert has a VRAM slot (slot_of[e] >= 0) by expert for native_expert_grouped:
/// grp_ptr[g] = slot_base + slot * slot_stride; entries ent_dst = route index (t*k + j), ent_tok = t.  Misses are left
/// out (the host computes them).  One block.
void group_hits(const int32_t* ids, int n_routes, int k, const int32_t* slot_of, int n_expert,
                const uint8_t* slot_base, size_t slot_stride, unsigned long long* grp_ptr, int32_t* grp_start,
                int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok, cudaStream_t s);

/// x[t] += sum_j w[t][j] * part(t*k+j) + shared[t], where a route whose expert has a slot reads gpu_parts[route] and
/// any other reads cpu_parts[cpu_row ? cpu_row[route] : route].
void moe_combine(float* x, const float* gpu_parts, const float* cpu_parts, const int32_t* cpu_row, const float* shared,
                 const int32_t* ids, const float* w, const int32_t* slot_of, int T, int k, int n_embd, cudaStream_t s);

/// Host coupling (mapped host memory): `publish` copies the MoE input rows and the routes to the host buffers and then
/// sets *flag = *seq; `wait_host` spins until *flag == *seq (a timeout sets *err and returns).
void publish(const float* x, int T, int n_embd, const int32_t* ids, const float* w, int n_routes, float* hx,
             int32_t* hids, float* hw, volatile int* flag, const volatile int* seq, cudaStream_t s);
void wait_host(const volatile int* flag, const volatile int* seq, volatile int* err, cudaStream_t s);

/// Greedy pick per row (the first maximum).
void argmax_rows(const float* x, int n, int rows, int32_t* out, cudaStream_t s);
/// Embedding rows given as floats (an image's rows): out[row[i]] = src[i].
void scatter_rows(const float* src, const int32_t* rows, int n_rows, int n_embd, float* out, cudaStream_t s);

/// EAGLE input rows: out[t] = [a[t] | b[t]], rows of n floats each.
void concat2(const float* a, const float* b, float* out, int n, int rows, cudaStream_t s);
/// One row: the first maximum and its softmax probability.
void argmax_prob(const float* x, int n, int32_t* out, float* prob, cudaStream_t s);

/// Prompt experts on the tensor cores: out[i] = f16(x[rows[i]]) (rows of `cols` floats).
void gather_rows_f16(const float* x, const int32_t* rows, int n, int cols, __half* out, cudaStream_t s);
/// h[i][f] = f16(silu(gu[i][f]) * gu[i][F + f]) for rows of [gate | up] (2F floats).
void swiglu_f16(const float* gu, int n, int F, __half* h, cudaStream_t s);

}  // namespace m4k
