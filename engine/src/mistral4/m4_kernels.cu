// engine/src/mistral4/m4_kernels.cu - see m4_kernels.cuh.
#include "m4_kernels.cuh"

#include <mma.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>

namespace m4k {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum / max for blockDim.x a multiple of 32, <= 1024; `red` has 32 floats
__device__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = threadIdx.x < (unsigned) nw ? red[threadIdx.x] : 0.f;
    if (w == 0) r = warp_sum(r);
    if (threadIdx.x == 0) red[0] = r;
    __syncthreads();
    return red[0];
}
__device__ float block_max(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = threadIdx.x < (unsigned) nw ? red[threadIdx.x] : -FLT_MAX;
    if (w == 0) r = warp_max(r);
    if (threadIdx.x == 0) red[0] = r;
    __syncthreads();
    return red[0];
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "strata-m4: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ---------------------------------------------------------------------------------------------- norms, adds
__global__ void rmsnorm_kernel(const float* x, const float* w, float* y, int n, int xs, int ys, float eps) {
    __shared__ float red[32];
    const float* xr = x + (size_t) blockIdx.x * xs;
    float* yr = y + (size_t) blockIdx.x * ys;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}

__global__ void add_kernel(float* x, const float* a, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a[i];
}

// ---------------------------------------------------------------------------------------------- MLA
// one block per token, 256 threads: thread d < 256 owns latent value d; threads < 32 own rope pair d
__global__ void mla_kv_write_kernel(const float* kvp, const float* kv_norm, const float2* rope_tab, const int32_t* pos,
                                    int kv_mode, void* cache, float* scales, float eps, KvMap map) {
    __shared__ float red[32];
    const int t = blockIdx.x, d = threadIdx.x;
    const float* r = kvp + (size_t) t * kKD;
    const int p = pos[t];
    const size_t prow = kv_row(map, t, p);   // the cache row (rope still takes the position)
    const float v = r[d];
    const float ss = block_sum(v * v, red);
    const float lat = v * rsqrtf(ss / (float) kLat + eps) * kv_norm[d];
    float rp = 0.f;   // this thread's rope value (threads < 64)
    if (d < kRope) {
        const int pair = d >> 1;
        const float2 cs = rope_tab[(size_t) p * (kRope / 2) + pair];
        const float x0 = r[kLat + 2 * pair], x1 = r[kLat + 2 * pair + 1];
        rp = (d & 1) ? (x0 * cs.y + x1 * cs.x) : (x0 * cs.x - x1 * cs.y);
    }
    if (kv_mode == KV_F16) {
        __half* row = static_cast<__half*>(cache) + prow * kKD;
        row[d] = __float2half(lat);
        if (d < kRope) row[kLat + d] = __float2half(rp);
        return;
    }
    const float am = block_max(fabsf(lat), red);
    const float ar = block_max(d < kRope ? fabsf(rp) : 0.f, red);
    const float s0 = am > 0.f ? am / 127.f : 1.f, s1 = ar > 0.f ? ar / 127.f : 1.f;
    int8_t* row = static_cast<int8_t*>(cache) + prow * kKD;
    row[d] = (int8_t) __float2int_rn(fminf(127.f, fmaxf(-127.f, lat / s0)));
    if (d < kRope) row[kLat + d] = (int8_t) __float2int_rn(fminf(127.f, fmaxf(-127.f, rp / s1)));
    if (d == 0) {
        scales[prow * 2 + 0] = s0;
        scales[prow * 2 + 1] = s1;
    }
}

// grid (T, n_head), 64 threads
__global__ void mla_q_rope_kernel(const float* q, float* Qf, const float2* rope_tab, const int32_t* pos, int n_head,
                                  float temp_scale, float temp_floor) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;   // d < 64
    const int p = pos[t];
    const float* qr = q + ((size_t) t * n_head + h) * 128 + 64;   // the rope half of the head
    float* o = Qf + ((size_t) t * n_head + h) * kKD;
    const int pair = d >> 1;
    const float2 cs = rope_tab[(size_t) p * (kRope / 2) + pair];
    const float x0 = qr[2 * pair], x1 = qr[2 * pair + 1];
    float v = (d & 1) ? (x0 * cs.y + x1 * cs.x) : (x0 * cs.x - x1 * cs.y);
    if (temp_scale != 0.f) {
        const float sc = logf(floorf((float) p / temp_floor) + 1.f) * temp_scale + 1.f;   // llama.cpp, vLLM: offset 0
        v *= sc;
        for (int i = d; i < kLat; i += 64) o[i] *= sc;
    }
    o[kLat + d] = v;
}

// grid (kAttnChunks, T), 256 threads.  One block: one query token t, a range of keys, all heads (the heads share
// every key row: MLA with absorption is MQA).  Per tile of 32 keys:
//   1. the tile's rows as half2 into shared memory
//   2. scores: 32 heads x 32 keys, thread tid takes (head, key) pairs tid, tid+256, tid+512, tid+768
//   3. softmax bookkeeping: thread h < n_head walks its 32 scores serially (running max m, sum l, rescale a)
//   4. output: thread (oh = tid/8, slice = tid%8) accumulates O[oh][slice*32 .. +32]
// No infinities anywhere (kNeg is a finite sentinel), no warp shuffles: every value has one writer and a barrier
// between its write and its reads.
constexpr int TILE = 32;
constexpr int QW = kKD / 2 + 1;   // half2 words per shared row (161: odd, so rows fall in different banks)
constexpr float kNeg = -1.0e30f;

__global__ void __launch_bounds__(256) mla_decode_kernel(const float* Qf, const void* cache, const float* scales,
                                                         int kv_mode, const int32_t* pos, int n_head, float kq_scale,
                                                         float* part, KvMap map) {
    __shared__ __half2 q2[kMaxHeads][QW];
    __shared__ __half2 k2[TILE][QW];
    __shared__ float S[kMaxHeads][TILE + 1];
    __shared__ float m_s[kMaxHeads], l_s[kMaxHeads], a_s[kMaxHeads];
    const int c = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const int n_kv = pos[t] + 1;
    const int per = ((n_kv + kAttnChunks - 1) / kAttnChunks + TILE - 1) / TILE * TILE;
    const int k0 = c * per;
    const int k1 = min(n_kv, k0 + per);
    const int oh = tid >> 3, slice = tid & 7;
    float* out = part + (((size_t) t * kAttnChunks + c) * n_head) * (kLat + 2);
    if (k0 >= k1) {   // an empty range: l = 0 marks it for the merge
        if (oh < n_head) {
            float* o = out + (size_t) oh * (kLat + 2);
            for (int j = 0; j < 32; ++j) o[slice * 32 + j] = 0.f;
            if (slice == 0) {
                o[kLat] = kNeg;
                o[kLat + 1] = 0.f;
            }
        }
        return;
    }
    for (int i = tid; i < n_head * (kKD / 2); i += 256) {
        const int h = i / (kKD / 2), w = i - h * (kKD / 2);
        const float* qr = Qf + ((size_t) t * n_head + h) * kKD;
        q2[h][w] = __floats2half2_rn(qr[2 * w] * kq_scale, qr[2 * w + 1] * kq_scale);
    }
    if (tid < kMaxHeads) {
        m_s[tid] = kNeg;
        l_s[tid] = 0.f;
        a_s[tid] = 1.f;
    }
    float acc[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) acc[j] = 0.f;

    for (int kt = k0; kt < k1; kt += TILE) {
        const int nk = min(TILE, k1 - kt);   // valid keys in this tile
        const size_t base = kv_row(map, t, kt);   // the tile's first cache row (a tile stays within one page)
        __syncthreads();                     // (a) the previous tile's readers are done; q2 / m_s are written
        for (int i = tid; i < TILE * (kKD / 2); i += 256) {
            const int r = i / (kKD / 2), w = i - r * (kKD / 2);
            float2 f = make_float2(0.f, 0.f);
            if (r < nk) {
                const size_t key = base + (size_t) r;
                if (kv_mode == KV_F16) {
                    const __half* row = static_cast<const __half*>(cache) + key * kKD;
                    f = make_float2(__half2float(row[2 * w]), __half2float(row[2 * w + 1]));
                } else {
                    const int8_t* row = static_cast<const int8_t*>(cache) + key * kKD;
                    const float sc = scales[key * 2 + (2 * w >= kLat ? 1 : 0)];
                    f = make_float2((float) row[2 * w] * sc, (float) row[2 * w + 1] * sc);
                }
            }
            k2[r][w] = __floats2half2_rn(f.x, f.y);
        }
        __syncthreads();                     // (b) the tile is loaded
        for (int pi = tid; pi < kMaxHeads * TILE; pi += 256) {
            const int h = pi / TILE, key = pi - h * TILE;
            if (h >= n_head) continue;
            float sacc = 0.f;
            for (int w = 0; w < kKD / 2; ++w) {
                const float2 qa = __half22float2(q2[h][w]);
                const float2 kb = __half22float2(k2[key][w]);
                sacc = fmaf(qa.x, kb.x, sacc);
                sacc = fmaf(qa.y, kb.y, sacc);
            }
            S[h][key] = key < nk ? sacc : kNeg;
        }
        __syncthreads();                     // (c) the scores are written
        if (tid < n_head) {
            const int h = tid;
            float mx = m_s[h];
            for (int key = 0; key < nk; ++key) mx = fmaxf(mx, S[h][key]);
            const float a = __expf(m_s[h] - mx);   // the first tile: exp(-1e30 - mx) = 0 (l and acc are 0 anyway)
            float sum = 0.f;
            for (int key = 0; key < TILE; ++key) {
                const float pk = key < nk ? __expf(S[h][key] - mx) : 0.f;
                S[h][key] = pk;
                sum += pk;
            }
            l_s[h] = l_s[h] * a + sum;
            m_s[h] = mx;
            a_s[h] = a;
        }
        __syncthreads();                     // (d) the probabilities and the rescale are written
        if (oh < n_head) {
            const float a = a_s[oh];
#pragma unroll
            for (int j = 0; j < 32; ++j) acc[j] *= a;
            for (int key = 0; key < nk; ++key) {
                const float pk = S[oh][key];
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const float2 v = __half22float2(k2[key][slice * 16 + j]);
                    acc[2 * j] = fmaf(pk, v.x, acc[2 * j]);
                    acc[2 * j + 1] = fmaf(pk, v.y, acc[2 * j + 1]);
                }
            }
        }
    }
    __syncthreads();
    if (oh < n_head) {
        float* o = out + (size_t) oh * (kLat + 2);
#pragma unroll
        for (int j = 0; j < 32; ++j) o[slice * 32 + j] = acc[j];
        if (slice == 0) {
            o[kLat] = m_s[oh];
            o[kLat + 1] = l_s[oh];
        }
    }
}

// ---------------------------------------------------------------------------------------------- decode on tensor cores
// The same work split as mla_decode_kernel (grid (kAttnChunks, T): one query token, one key range, all heads) and the
// same partial-result layout for mla_merge_kernel, but both products on the tensor cores (WMMA 16x16x16, f16 in, f32
// accumulate): per tile of 32 keys S[32 heads][32 keys] = Q K^T, an online softmax per head row (a warp per 4 rows,
// a lane per key), the output O[32][256] (in shared memory, rescaled per row) += P V.  256 threads, 8 warps.
namespace tc {
constexpr int WT = 32;            // keys per tile
constexpr int QLD = kKD + 8;      // halves per Q / K row in shared memory (328: 16-byte multiple, staggered banks)
constexpr int SLD = WT + 4;       // floats per score row
constexpr int PLD = WT + 8;       // halves per probability row
constexpr int OLD = kLat + 4;     // floats per output row
constexpr size_t kSmem = (size_t) kMaxHeads * QLD * 2 + (size_t) WT * QLD * 2 + (size_t) kMaxHeads * SLD * 4 +
                         (size_t) kMaxHeads * PLD * 2 + (size_t) kMaxHeads * OLD * 4 + 3 * kMaxHeads * 4;
}  // namespace tc

__global__ void __launch_bounds__(256) mla_decode_tc_kernel(const float* Qf, const void* cache, const float* scales,
                                                            int kv_mode, const int32_t* pos, int n_head, float kq_scale,
                                                            float* part, KvMap map) {
    using namespace nvcuda;
    using namespace tc;
    extern __shared__ __align__(128) unsigned char smem[];
    __half* qh = reinterpret_cast<__half*>(smem);                       // [32][QLD]
    __half* kh = qh + kMaxHeads * QLD;                                 // [WT][QLD]
    float* S = reinterpret_cast<float*>(kh + WT * QLD);                // [32][SLD]
    __half* P = reinterpret_cast<__half*>(S + kMaxHeads * SLD);        // [32][PLD]
    float* O = reinterpret_cast<float*>(P + kMaxHeads * PLD);          // [32][OLD]
    float* m_s = O + kMaxHeads * OLD;
    float* l_s = m_s + kMaxHeads;
    float* a_s = l_s + kMaxHeads;
    const int c = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n_kv = pos[t] + 1;
    const int per = ((n_kv + kAttnChunks - 1) / kAttnChunks + WT - 1) / WT * WT;
    const int k0 = c * per;
    const int k1 = min(n_kv, k0 + per);
    float* out = part + (((size_t) t * kAttnChunks + c) * n_head) * (kLat + 2);
    if (k0 >= k1) {   // an empty range: l = 0 marks it for the merge
        for (int i = tid; i < n_head * (kLat + 2); i += 256) {
            const int d = i % (kLat + 2);
            out[i] = d == kLat ? kNeg : 0.f;
        }
        return;
    }
    for (int i = tid; i < kMaxHeads * kKD; i += 256) {
        const int h = i / kKD, d = i - h * kKD;
        qh[h * QLD + d] = __float2half(h < n_head ? Qf[((size_t) t * n_head + h) * kKD + d] * kq_scale : 0.f);
    }
    for (int i = tid; i < kMaxHeads * OLD; i += 256) O[i] = 0.f;
    if (tid < kMaxHeads) {
        m_s[tid] = kNeg;
        l_s[tid] = 0.f;
        a_s[tid] = 1.f;
    }
    for (int kt = k0; kt < k1; kt += WT) {
        const int nk = min(WT, k1 - kt);
        const size_t base = kv_row(map, t, kt);   // a tile stays within one page
        __syncthreads();   // (a) the previous tile's readers are done
        // 1. the key rows as f16 (4 values per item; rows beyond the range are zero)
        for (int i = tid; i < WT * (kKD / 4); i += 256) {
            const int r = i / (kKD / 4), w = (i - r * (kKD / 4)) * 4;
            __half2 lo = __floats2half2_rn(0.f, 0.f), hi = lo;
            if (r < nk) {
                const size_t key = base + (size_t) r;
                if (kv_mode == KV_F16) {
                    const __half2* row = reinterpret_cast<const __half2*>(static_cast<const __half*>(cache) + key * kKD + w);
                    lo = row[0];
                    hi = row[1];
                } else {
                    const char4 q = *reinterpret_cast<const char4*>(static_cast<const int8_t*>(cache) + key * kKD + w);
                    const float sc = scales[key * 2 + (w >= kLat ? 1 : 0)];
                    lo = __floats2half2_rn((float) q.x * sc, (float) q.y * sc);
                    hi = __floats2half2_rn((float) q.z * sc, (float) q.w * sc);
                }
            }
            __half2* dst = reinterpret_cast<__half2*>(kh + r * QLD + w);
            dst[0] = lo;
            dst[1] = hi;
        }
        __syncthreads();   // (b) the tile is loaded
        // 2. S = Q K^T: 2 x 2 fragments, warps 0-3
        if (warp < 4) {
            const int r = warp >> 1, cc = warp & 1;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
#pragma unroll 4
            for (int k = 0; k < kKD / 16; ++k) {
                wmma::load_matrix_sync(a, qh + r * 16 * QLD + k * 16, QLD);
                wmma::load_matrix_sync(b, kh + cc * 16 * QLD + k * 16, QLD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(S + r * 16 * SLD + cc * 16, acc, SLD, wmma::mem_row_major);
        }
        __syncthreads();   // (c) the scores are written
        // 3. online softmax: warp w takes rows 4w .. 4w+3, lane = key
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            const int row = warp * 4 + rr;
            const float mold = m_s[row];
            const float sv = lane < nk ? S[row * SLD + lane] : kNeg;
            const float mx = fmaxf(mold, warp_max(sv));
            const float pk = lane < nk ? __expf(sv - mx) : 0.f;
            const float sum = warp_sum(pk);
            P[row * PLD + lane] = __float2half(pk);
            __syncwarp();
            if (lane == 0) {
                const float a = __expf(mold - mx);
                l_s[row] = l_s[row] * a + sum;
                m_s[row] = mx;
                a_s[row] = a;
            }
        }
        __syncthreads();   // (d) P and the rescale factors are written
        for (int i = tid; i < kMaxHeads * kLat; i += 256) {
            const int row = i / kLat, d = i - row * kLat;
            O[row * OLD + d] *= a_s[row];
        }
        __syncthreads();   // (e) O is rescaled
        // 4. O += P V: 2 x 16 fragments, 4 per warp
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            const int fi = warp * 4 + f, r = fi >> 4, cc = fi & 15;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, O + r * 16 * OLD + cc * 16, OLD, wmma::mem_row_major);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
#pragma unroll
            for (int k = 0; k < WT / 16; ++k) {
                wmma::load_matrix_sync(a, P + r * 16 * PLD + k * 16, PLD);
                wmma::load_matrix_sync(b, kh + k * 16 * QLD + cc * 16, QLD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(O + r * 16 * OLD + cc * 16, acc, OLD, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int i = tid; i < n_head * kLat; i += 256) {
        const int h = i / kLat, d = i - h * kLat;
        out[(size_t) h * (kLat + 2) + d] = O[h * OLD + d];
    }
    if (tid < n_head) {
        out[(size_t) tid * (kLat + 2) + kLat] = m_s[tid];
        out[(size_t) tid * (kLat + 2) + kLat + 1] = l_s[tid];
    }
}

// grid (T, n_head), 256 threads: merge the key ranges (a range with l == 0 holds nothing)
__global__ void mla_merge_kernel(const float* part, float* O, int n_head) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    float M = kNeg;
    for (int c = 0; c < kAttnChunks; ++c) {
        const float* pr = part + (((size_t) t * kAttnChunks + c) * n_head + h) * (kLat + 2);
        if (pr[kLat + 1] > 0.f) M = fmaxf(M, pr[kLat]);
    }
    float L = 0.f, acc = 0.f;
    for (int c = 0; c < kAttnChunks; ++c) {
        const float* pr = part + (((size_t) t * kAttnChunks + c) * n_head + h) * (kLat + 2);
        if (!(pr[kLat + 1] > 0.f)) continue;
        const float e = __expf(pr[kLat] - M);
        L += pr[kLat + 1] * e;
        acc += pr[d] * e;
    }
    O[((size_t) t * n_head + h) * kLat + d] = L > 0.f ? acc / L : 0.f;
}

__global__ void cache_to_f16_kernel(const void* cache, const float* scales, int kv_mode, int n, __half* K16,
                                    KvMap map) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * kKD) return;
    const size_t k = i / kKD, d = i % kKD;
    const size_t r = kv_row(map, 0, (int) k);   // key k's cache row
    const size_t src = r * kKD + d;
    if (kv_mode == KV_F16) {
        K16[i] = static_cast<const __half*>(cache)[src];
        return;
    }
    K16[i] = __float2half((float) static_cast<const int8_t*>(cache)[src] * scales[r * 2 + (d >= (size_t) kLat ? 1 : 0)]);
}

__global__ void f32_to_f16_kernel(const float* x, __half* y, size_t n, float scale) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i] * scale);
}

// one block per column
__global__ void softmax_causal_kernel(const float* S, __half* P, int n_kv, int n_head, int pos0) {
    __shared__ float red[32];
    const int c = blockIdx.x;
    const int lim = min(n_kv, pos0 + c / n_head + 1);
    const float* s = S + (size_t) c * n_kv;
    __half* p = P + (size_t) c * n_kv;
    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < lim; i += blockDim.x) m = fmaxf(m, s[i]);
    m = block_max(m, red);
    float sum = 0.f;
    for (int i = threadIdx.x; i < lim; i += blockDim.x) sum += __expf(s[i] - m);
    sum = block_sum(sum, red);
    const float inv = 1.f / sum;
    for (int i = threadIdx.x; i < n_kv; i += blockDim.x) p[i] = __float2half(i < lim ? __expf(s[i] - m) * inv : 0.f);
}

// ---------------------------------------------------------------------------------------------- router, MoE
// one block per token
__global__ void router_kernel(const float* logits, const float* bias, int n_expert, int k, int gating, bool norm,
                              float scale, int32_t* ids, float* w) {
    extern __shared__ float sm[];
    float* prob = sm;               // n_expert
    float* sel = sm + n_expert;     // n_expert
    __shared__ float red[32];
    __shared__ int pick;
    const int t = blockIdx.x;
    const float* lg = logits + (size_t) t * n_expert;
    if (gating == 2) {
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x) prob[e] = 1.f / (1.f + __expf(-lg[e]));
    } else {
        float m = -FLT_MAX;
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x) m = fmaxf(m, lg[e]);
        m = block_max(m, red);
        float s = 0.f;
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
            prob[e] = __expf(lg[e] - m);
            s += prob[e];
        }
        s = block_sum(s, red);
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x) prob[e] /= s;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) sel[e] = prob[e] + (bias ? bias[e] : 0.f);
    __syncthreads();
    float wsum = 0.f;
    for (int j = 0; j < k; ++j) {
        // block argmax, the lowest index on ties
        float best = -FLT_MAX;
        int bi = 0x7fffffff;
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x)
            if (sel[e] > best || (sel[e] == best && e < bi)) {
                best = sel[e];
                bi = e;
            }
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) {
                best = ob;
                bi = oi;
            }
        }
        __shared__ float wb[32];
        __shared__ int wi[32];
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        if (lane == 0) {
            wb[warp] = best;
            wi[warp] = bi;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int i = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (wb[q] > b || (wb[q] == b && wi[q] < i)) {
                    b = wb[q];
                    i = wi[q];
                }
            pick = i;
            ids[(size_t) t * k + j] = i;
            w[(size_t) t * k + j] = prob[i];
            sel[i] = -FLT_MAX;
        }
        __syncthreads();
        wsum += prob[pick];
    }
    if (threadIdx.x == 0) {
        const float div = norm ? fmaxf(wsum, 6.103515625e-5f) : 1.f;
        for (int j = 0; j < k; ++j) {
            float v = w[(size_t) t * k + j];
            if (norm) v /= div;
            if (scale != 0.f && scale != 1.f) v *= scale;
            w[(size_t) t * k + j] = v;
        }
    }
}

__global__ void group_hits_kernel(const int32_t* ids, int n_routes, int k, const int32_t* slot_of, int n_expert,
                                  const uint8_t* slot_base, size_t slot_stride, unsigned long long* grp_ptr,
                                  int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    extern __shared__ int cnt[];   // n_expert counts, then n_expert starts, then n_expert group ids
    int* start = cnt + n_expert;
    int* gid = start + n_expert;
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        int c = 0;
        if (slot_of[e] >= 0)
            for (int r = 0; r < n_routes; ++r) c += ids[r] == e;
        cnt[e] = c;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, off = 0;
        for (int e = 0; e < n_expert; ++e) {
            start[e] = off;
            gid[e] = -1;
            if (cnt[e] > 0) {
                gid[e] = g;
                grp_start[g] = off;
                grp_ptr[g] = (unsigned long long) (slot_base + (size_t) slot_of[e] * slot_stride);
                ++g;
                off += cnt[e];
            }
        }
        grp_start[g] = off;
        *n_groups = g;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        if (gid[e] < 0) continue;
        int idx = start[e];
        for (int r = 0; r < n_routes; ++r)
            if (ids[r] == e) {
                ent_dst[idx] = r;
                ent_tok[idx] = r / k;
                ++idx;
            }
    }
}

__global__ void moe_combine_kernel(float* x, const float* gpu_parts, const float* cpu_parts, const int32_t* cpu_row,
                                   const float* shared, const int32_t* ids, const float* w, const int32_t* slot_of, int k,
                                   int n_embd) {
    const int t = blockIdx.x;
    __shared__ const float* src[16];
    __shared__ float ww[16];
    if (threadIdx.x < (unsigned) k) {
        const int r = t * k + threadIdx.x;
        src[threadIdx.x] = slot_of[ids[r]] >= 0 ? gpu_parts + (size_t) r * n_embd
                                                : cpu_parts + (size_t) (cpu_row ? cpu_row[r] : r) * n_embd;
        ww[threadIdx.x] = w[r];
    }
    __syncthreads();
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) {
        float s = 0.f;
        for (int j = 0; j < k; ++j) s += ww[j] * src[j][d];
        s += shared[(size_t) t * n_embd + d];
        x[(size_t) t * n_embd + d] += s;
    }
}

__global__ void publish_kernel(const float* x, int n, const int32_t* ids, const float* w, int n_routes, float* hx,
                               int32_t* hids, float* hw, volatile int* flag, const volatile int* seq) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* h4 = reinterpret_cast<float4*>(hx);
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) h4[i] = x4[i];
    for (int i = threadIdx.x; i < n_routes; i += blockDim.x) {
        hids[i] = ids[i];
        hw[i] = w[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *flag = *seq;
        __threadfence_system();
    }
}

__global__ void wait_host_kernel(const volatile int* flag, const volatile int* seq, volatile int* err) {
    const int want = *seq;
    const long long t0 = clock64();
    while (*flag != want) {
        __nanosleep(200);
        if (clock64() - t0 > 30000000000ll) {   // ~20 s at 1.5 GHz: the host is gone
            *err = 1;
            break;
        }
    }
    __threadfence_system();
}

__global__ void argmax_kernel(const float* x, int n, int32_t* out) {
    const float* r = x + (size_t) blockIdx.x * n;
    float best = -FLT_MAX;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (r[i] > best) {
            best = r[i];
            bi = i;
        }
    __shared__ float wb[32];
    __shared__ int wi[32];
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ob > best || (ob == best && oi < bi)) {
            best = ob;
            bi = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        wb[threadIdx.x >> 5] = best;
        wi[threadIdx.x >> 5] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        best = wb[0];
        bi = wi[0];
        for (int q = 1; q < (int) (blockDim.x >> 5); ++q)
            if (wb[q] > best || (wb[q] == best && wi[q] < bi)) {
                best = wb[q];
                bi = wi[q];
            }
        out[blockIdx.x] = bi;
    }
}

__global__ void scatter_rows_kernel(const float* src, const int32_t* rows, int n_embd, float* out) {
    const int i = blockIdx.x;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) out[(size_t) rows[i] * n_embd + d] = src[(size_t) i * n_embd + d];
}

__global__ void concat2_kernel(const float* a, const float* b, float* out, int n) {
    const int t = blockIdx.x;
    float* o = out + (size_t) t * 2 * n;
    for (int d = threadIdx.x; d < n; d += blockDim.x) {
        o[d] = a[(size_t) t * n + d];
        o[n + d] = b[(size_t) t * n + d];
    }
}

// one block of 1024: argmax (first maximum) and its softmax probability
__global__ void argmax_prob_kernel(const float* x, int n, int32_t* out, float* prob) {
    __shared__ float red[32];
    __shared__ float wb[32];
    __shared__ int wi[32];
    float best = -FLT_MAX;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > best) { best = x[i]; bi = i; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { wb[threadIdx.x >> 5] = best; wi[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int q = 1; q < (int) (blockDim.x >> 5); ++q)
            if (wb[q] > wb[0] || (wb[q] == wb[0] && wi[q] < wi[0])) { wb[0] = wb[q]; wi[0] = wi[q]; }
    }
    __syncthreads();
    const float m = wb[0];
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += __expf(x[i] - m);
    sum = block_sum(sum, red);
    if (threadIdx.x == 0) {
        *out = wi[0];
        *prob = 1.f / sum;
    }
}

// f16 GEMM operands: a row whose largest value would not fit f16 (the first token's "massive activations" reach
// 1e4 and more in some FFN channels; f16 ends at 65504) is scaled down by a power-free factor and the GEMM's output row
// scaled back (inv[i]); every other row is converted as it is (inv 1).
constexpr float kF16Room = 16384.f;

__global__ void rows_f16_scaled_kernel(const float* x, const int32_t* rows, int cols, __half* out, float* inv) {
    __shared__ float red[32];
    const int i = blockIdx.x;
    const float* src = x + (size_t) (rows ? rows[i] : i) * cols;
    float m = 0.f;
    for (int d = threadIdx.x; d < cols; d += blockDim.x) m = fmaxf(m, fabsf(src[d]));
    m = block_max(m, red);
    const float sc = m > kF16Room ? kF16Room / m : 1.f;
    __half* dst = out + (size_t) i * cols;
    for (int d = threadIdx.x; d < cols; d += blockDim.x) dst[d] = __float2half(src[d] * sc);
    if (threadIdx.x == 0) inv[i] = 1.f / sc;
}

// h = silu(g) * u of the gate/up GEMM's rows (scaled by inv_in, which this undoes), written as f16 with its own scale
__global__ void swiglu_f16_scaled_kernel(const float* gu, const float* inv_in, int F, __half* h, float* inv_out) {
    __shared__ float red[32];
    const int i = blockIdx.x;
    const float* g = gu + (size_t) i * 2 * F;
    const float s = inv_in ? inv_in[i] : 1.f;
    float m = 0.f;
    for (int f = threadIdx.x; f < F; f += blockDim.x) {
        const float a = g[f] * s;
        m = fmaxf(m, fabsf(a / (1.f + __expf(-a)) * g[F + f] * s));
    }
    m = block_max(m, red);
    const float sc = m > kF16Room ? kF16Room / m : 1.f;
    for (int f = threadIdx.x; f < F; f += blockDim.x) {
        const float a = g[f] * s;
        h[(size_t) i * F + f] = __float2half(a / (1.f + __expf(-a)) * g[F + f] * s * sc);
    }
    if (threadIdx.x == 0) inv_out[i] = 1.f / sc;
}

__global__ void scale_rows_kernel(float* y, int cols, const float* inv) {
    const float s = inv[blockIdx.x];
    if (s == 1.f) return;
    float* r = y + (size_t) blockIdx.x * cols;
    for (int d = threadIdx.x; d < cols; d += blockDim.x) r[d] *= s;
}

}  // namespace

// ================================================================================================ launchers
void rmsnorm(const float* x, const float* w, float* y, int n, int rows, int xs, int ys, float eps, cudaStream_t s) {
    if (rows <= 0) return;
    rmsnorm_kernel<<<rows, 256, 0, s>>>(x, w, y, n, xs, ys, eps);
    check("rmsnorm");
}

void add_inplace(float* x, const float* a, size_t n, cudaStream_t s) {
    add_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, a, n);
    check("add");
}

void rope_table_host(float* out, int n_pos, int n_rot, float freq_base, float freq_scale, float ext_factor,
                     float attn_factor, float beta_fast, float beta_slow, int n_ctx_orig) {
    // ggml_rope_yarn_corr_dims / rope_yarn / ggml_rope_cache_init, the same float operations in the same order
    auto corr_dim = [&](float n_rot_b) {
        return (float) n_rot * logf((float) n_ctx_orig / (n_rot_b * 2 * (float) M_PI)) / (2 * logf(freq_base));
    };
    float corr[2];
    corr[0] = std::max(0.f, floorf(corr_dim(beta_fast)));
    corr[1] = std::min((float) (n_rot - 1), ceilf(corr_dim(beta_slow)));
    const float theta_scale = powf(freq_base, -2.0f / (float) n_rot);
    for (int p = 0; p < n_pos; ++p) {
        float theta = (float) p;
        for (int i0 = 0; i0 < n_rot; i0 += 2) {
            const float theta_extrap = theta;
            const float theta_interp = freq_scale * theta_extrap;
            float th = theta_interp, mscale = attn_factor;
            if (ext_factor != 0.0f) {
                const float y = ((float) (i0 / 2) - corr[0]) / std::max(0.001f, corr[1] - corr[0]);
                const float ramp = 1 - std::min(1.f, std::max(0.f, y));
                const float mix = ramp * ext_factor;
                th = theta_interp * (1 - mix) + theta_extrap * mix;
                mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }
            out[((size_t) p * (n_rot / 2) + i0 / 2) * 2 + 0] = cosf(th) * mscale;
            out[((size_t) p * (n_rot / 2) + i0 / 2) * 2 + 1] = sinf(th) * mscale;
            theta *= theta_scale;
        }
    }
}

void mla_kv_write(const float* kvp, const float* kv_norm, const float* rope_tab, const int32_t* pos, int T, int kv_mode,
                  void* cache, float* scales, float eps, cudaStream_t s, KvMap map) {
    if (T <= 0) return;
    mla_kv_write_kernel<<<T, kLat, 0, s>>>(kvp, kv_norm, reinterpret_cast<const float2*>(rope_tab), pos, kv_mode, cache,
                                            scales, eps, map);
    check("mla_kv_write");
}

void mla_q_rope(const float* q, float* Qf, const float* rope_tab, const int32_t* pos, int T, int n_head, float temp_scale,
                float temp_floor, cudaStream_t s) {
    if (T <= 0) return;
    mla_q_rope_kernel<<<dim3(T, n_head), kRope, 0, s>>>(q, Qf, reinterpret_cast<const float2*>(rope_tab), pos, n_head,
                                                         temp_scale, temp_floor);
    check("mla_q_rope");
}

// ------------------------------------------------------------------------------------------------ prompt attention per head
// Prompts without absorption: per head q = [q_nope (64) | q_pe (64)], k = [wk_b . c (64) | k_pe (64)], v = wv_b . c
// (128).  2.25x fewer products than the absorbed form (rows 320 / 256 wide), and one fused kernel - online softmax,
// the scores never leave the SM - instead of a score matrix in memory.

// Qm[t][h][0:64] = q_nope (x the attention temperature), [64:128] = q_pe as mla_q_rope left it in Qf; all x scale
__global__ void mla_q_mha_kernel(const float* q, const float* Qf, const int32_t* pos, int n_head, float scale,
                                 float temp_scale, float temp_floor, __half* Qm) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;   // d < 128
    const size_t th = (size_t) t * n_head + h;
    float v;
    if (d < 64) {
        v = q[th * 128 + d];
        if (temp_scale != 0.f) v *= logf(floorf((float) pos[t] / temp_floor) + 1.f) * temp_scale + 1.f;
    } else {
        v = Qf[th * kKD + kLat + (d - 64)];
    }
    Qm[th * 128 + d] = __float2half(v * scale);
}

// Kh[n][h][64:128] = the key's rope part (shared by the heads), from the f16 latent rows
__global__ void mla_k_rope_bcast_kernel(const __half* K16, int n_head, __half* Kh) {
    const int n = blockIdx.x;
    const __half2* src = reinterpret_cast<const __half2*>(K16 + (size_t) n * kKD + kLat);
    for (int i = threadIdx.x; i < n_head * 32; i += blockDim.x) {
        const int h = i >> 5, w = i & 31;
        reinterpret_cast<__half2*>(Kh + ((size_t) n * n_head + h) * 128 + 64)[w] = src[w];
    }
}

namespace fa {
constexpr int BQ = 64;          // query rows per block: 4 warps x 16
constexpr int BK = 32;          // keys per tile
constexpr int D = 128;          // head width (q.k and v)
constexpr int LDH = D + 8;      // halves per shared row
constexpr int LDS = BK + 4;     // floats per score row
constexpr int LDP = BK + 8;     // halves per P row (aliases the score rows)
constexpr int NW = 4;
}  // namespace fa

// grid (ceil(T / 64), n_head), 128 threads.  Block: 64 query rows of one head; warp w: rows 16w .. 16w+15.
// Per tile of 32 keys: K, V into shared memory; per warp S = Q K^T (16 x 32, tensor cores) into shared memory;
// online softmax (lanes 2r, 2r+1: row r, 16 keys each; log2 domain - Qm carries log2 e); O = O * alpha + P V with O
// in registers.  The accumulator layout is unspecified, so each thread learns once which row each of its elements
// is (rowof) and rescales its own elements.  Keys kb0 .. kb0+nb-1 per launch (Kh, Vh hold just them); between
// launches O (unnormalized), max and sum wait in Oacc / ML; the last launch writes O / sum to out [T][n_head * 128].
__global__ void __launch_bounds__(128, 2) mla_fa_prefill_kernel(const __half* __restrict__ Qm, const __half* __restrict__ Kh,
                                                               const __half* __restrict__ Vh, int T, int n_head, int pos0,
                                                               int kb0, int nb, int first, int last, float* __restrict__ Oacc,
                                                               float* __restrict__ ML, float* __restrict__ out) {
    using namespace nvcuda;
    using namespace fa;
    __shared__ __align__(32) __half Qs[BQ * LDH];
    __shared__ __align__(32) __half Ks[BK * LDH];
    __shared__ __align__(32) __half Vs[BK * LDH];
    __shared__ __align__(32) float Sw[NW][16 * LDS];
    __shared__ float m_s[NW][16], l_s[NW][16], a_s[NW][16];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h = blockIdx.y, q0 = blockIdx.x * BQ, r0 = q0 + warp * 16;
    const size_t rs = (size_t) n_head * D;   // row stride (elements) of Qm, Kh, Vh, Oacc, out
    for (int i = tid; i < BQ * (D / 8); i += 128) {   // the query tile (rows past T: zeros)
        const int r = i / (D / 8), c = (i % (D / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (q0 + r < T) v = *reinterpret_cast<const uint4*>(Qm + (size_t) (q0 + r) * rs + (size_t) h * D + c);
        *reinterpret_cast<uint4*>(Qs + r * LDH + c) = v;
    }
    float* S = Sw[warp];
    __half* Pw = reinterpret_cast<__half*>(S);
    using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    constexpr int NE = Acc::num_elements;
    int rowof[NE];
    {
        for (int i = lane; i < 256; i += 32) S[(i >> 4) * LDS + (i & 15)] = (float) (i >> 4);
        __syncwarp();
        Acc probe;
        wmma::load_matrix_sync(probe, S, LDS, wmma::mem_row_major);
        #pragma unroll
        for (int i = 0; i < NE; ++i) rowof[i] = (int) probe.x[i];
        __syncwarp();
    }
    if (lane < 16) {
        const int t = r0 + lane;
        float m = kNeg, l = 0.f;
        if (!first && t < T) {
            m = ML[((size_t) t * n_head + h) * 2];
            l = ML[((size_t) t * n_head + h) * 2 + 1];
        }
        m_s[warp][lane] = m;
        l_s[warp][lane] = l;
    }
    Acc O[D / 16];
    #pragma unroll
    for (int f = 0; f < D / 16; ++f) {
        if (first) {
            wmma::fill_fragment(O[f], 0.f);
            continue;
        }
        for (int i = lane; i < 256; i += 32) {   // staged: rows past T read nothing
            const int r = i >> 4, c = i & 15, t = r0 + r;
            S[r * LDS + c] = t < T ? Oacc[(size_t) t * rs + (size_t) h * D + f * 16 + c] : 0.f;
        }
        __syncwarp();
        wmma::load_matrix_sync(O[f], S, LDS, wmma::mem_row_major);
        __syncwarp();
    }
    const bool live = r0 < T;
    const int wlast = pos0 + min(r0 + 16, T) - 1;            // this warp's last query position
    const int kend = min(kb0 + nb, pos0 + min(q0 + BQ, T));  // keys past the block's last query: none
    for (int kt = kb0; kt < kend; kt += BK) {
        __syncthreads();   // the previous tile is used up
        for (int i = tid; i < BK * (D / 8); i += 128) {
            const int r = i / (D / 8), c = (i % (D / 8)) * 8;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
            if (kt + r < kb0 + nb) {
                const size_t o = (size_t) (kt - kb0 + r) * rs + (size_t) h * D + c;
                kv = *reinterpret_cast<const uint4*>(Kh + o);
                vv = *reinterpret_cast<const uint4*>(Vh + o);
            }
            *reinterpret_cast<uint4*>(Ks + r * LDH + c) = kv;
            *reinterpret_cast<uint4*>(Vs + r * LDH + c) = vv;
        }
        __syncthreads();
        if (!live || kt > wlast) continue;   // no key of the tile is visible to this warp's rows
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
            Acc s0, s1;
            wmma::fill_fragment(s0, 0.f);
            wmma::fill_fragment(s1, 0.f);
            #pragma unroll
            for (int k = 0; k < D / 16; ++k) {
                wmma::load_matrix_sync(a, Qs + warp * 16 * LDH + k * 16, LDH);
                wmma::load_matrix_sync(b, Ks + k * 16, LDH);
                wmma::mma_sync(s0, a, b, s0);
                wmma::load_matrix_sync(b, Ks + 16 * LDH + k * 16, LDH);
                wmma::mma_sync(s1, a, b, s1);
            }
            wmma::store_matrix_sync(S, s0, LDS, wmma::mem_row_major);
            wmma::store_matrix_sync(S + 16, s1, LDS, wmma::mem_row_major);
        }
        __syncwarp();
        {
            const int r = lane >> 1, hf = lane & 1, t = r0 + r, qpos = pos0 + t;
            float sv[16];
            unsigned mk = 0;
            float mx = kNeg;
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                const int col = hf * 16 + c, kpos = kt + col;
                sv[c] = S[r * LDS + col];
                if (t < T && kpos < kb0 + nb && kpos <= qpos) {
                    mk |= 1u << c;
                    mx = fmaxf(mx, sv[c]);
                }
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            const float m_old = m_s[warp][r], m_new = fmaxf(m_old, mx), alpha = exp2f(m_old - m_new);
            const float l_old = l_s[warp][r];
            __syncwarp();   // every lane holds its scores (and the row's state) before P and the state change
            float sum = 0.f;
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                const float p = (mk >> c & 1u) ? exp2f(sv[c] - m_new) : 0.f;
                sum += p;
                Pw[r * LDP + hf * 16 + c] = __float2half(p);
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            if (hf == 0) {
                m_s[warp][r] = m_new;
                l_s[warp][r] = l_old * alpha + sum;
                a_s[warp][r] = alpha;
            }
        }
        __syncwarp();
        float al[NE];
        #pragma unroll
        for (int i = 0; i < NE; ++i) al[i] = a_s[warp][rowof[i]];
        #pragma unroll
        for (int f = 0; f < D / 16; ++f)
            #pragma unroll
            for (int i = 0; i < NE; ++i) O[f].x[i] *= al[i];
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
            #pragma unroll
            for (int k = 0; k < BK / 16; ++k) {
                wmma::load_matrix_sync(a, Pw + k * 16, LDP);
                #pragma unroll
                for (int f = 0; f < D / 16; ++f) {
                    wmma::load_matrix_sync(b, Vs + k * 16 * LDH + f * 16, LDH);
                    wmma::mma_sync(O[f], a, b, O[f]);
                }
            }
        }
        __syncwarp();
    }
    if (!live) return;
    if (last) {
        float inv[NE];
        #pragma unroll
        for (int i = 0; i < NE; ++i) {
            const float l = l_s[warp][rowof[i]];
            inv[i] = l > 0.f ? 1.f / l : 0.f;
        }
        #pragma unroll
        for (int f = 0; f < D / 16; ++f)
            #pragma unroll
            for (int i = 0; i < NE; ++i) O[f].x[i] *= inv[i];
    }
    float* dst = last ? out : Oacc;
    #pragma unroll
    for (int f = 0; f < D / 16; ++f) {
        wmma::store_matrix_sync(S, O[f], LDS, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
            const int r = i >> 4, c = i & 15, t = r0 + r;
            if (t < T) dst[(size_t) t * rs + (size_t) h * D + f * 16 + c] = S[r * LDS + c];
        }
        __syncwarp();
    }
    if (!last && lane < 16 && r0 + lane < T) {
        ML[((size_t) (r0 + lane) * n_head + h) * 2] = m_s[warp][lane];
        ML[((size_t) (r0 + lane) * n_head + h) * 2 + 1] = l_s[warp][lane];
    }
}

size_t mla_decode_part_floats(int T, int n_head) { return (size_t) T * kAttnChunks * n_head * (kLat + 2); }

void mla_attn_decode(const float* Qf, const void* cache, const float* scales, int kv_mode, const int32_t* pos, int T,
                     int n_head, float kq_scale, float* part, float* O, cudaStream_t s, int impl, KvMap map) {
    if (n_head > kMaxHeads) {
        std::fprintf(stderr, "strata-m4: %d heads, the decode attention takes %d\n", n_head, kMaxHeads);
        std::exit(1);
    }
    if (impl == ATTN_TC) {
        static bool attr = false;   // > 48 KiB of dynamic shared memory needs the opt-in (once, before any capture)
        if (!attr) {
            const cudaError_t e = cudaFuncSetAttribute(mla_decode_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       (int) tc::kSmem);
            if (e != cudaSuccess) {
                std::fprintf(stderr, "strata-m4: tensor-core attention needs %zu bytes of shared memory: %s\n", tc::kSmem,
                             cudaGetErrorString(e));
                std::exit(1);
            }
            attr = true;
        }
        mla_decode_tc_kernel<<<dim3(kAttnChunks, T), 256, tc::kSmem, s>>>(Qf, cache, scales, kv_mode, pos, n_head,
                                                                          kq_scale, part, map);
        check("mla_decode_tc");
    } else {
        mla_decode_kernel<<<dim3(kAttnChunks, T), 256, 0, s>>>(Qf, cache, scales, kv_mode, pos, n_head, kq_scale, part,
                                                                map);
        check("mla_decode");
    }
    mla_merge_kernel<<<dim3(T, n_head), kLat, 0, s>>>(part, O, n_head);
    check("mla_merge");
}

void mla_cache_to_f16(const void* cache, const float* scales, int kv_mode, int n, __half* K16, cudaStream_t s,
                      KvMap map) {
    const size_t tot = (size_t) n * kKD;
    if (tot == 0) return;
    cache_to_f16_kernel<<<(unsigned) ((tot + 255) / 256), 256, 0, s>>>(cache, scales, kv_mode, n, K16, map);
    check("cache_to_f16");
}

void mla_q_mha(const float* q, const float* Qf, const int32_t* pos, int T, int n_head, float scale, float temp_scale,
               float temp_floor, __half* Qm, cudaStream_t s) {
    if (T <= 0) return;
    mla_q_mha_kernel<<<dim3(T, n_head), 128, 0, s>>>(q, Qf, pos, n_head, scale, temp_scale, temp_floor, Qm);
    check("mla_q_mha");
}

void mla_k_rope_bcast(const __half* K16, int nb, int n_head, __half* Kh, cudaStream_t s) {
    if (nb <= 0) return;
    mla_k_rope_bcast_kernel<<<nb, std::min(1024, n_head * 32), 0, s>>>(K16, n_head, Kh);
    check("mla_k_rope_bcast");
}

void mla_fa_prefill(const __half* Qm, const __half* Kh, const __half* Vh, int T, int n_head, int pos0, int kb0, int nb,
                    bool first, bool last, float* Oacc, float* ML, float* out, cudaStream_t s) {
    if (T <= 0) return;
    mla_fa_prefill_kernel<<<dim3((T + fa::BQ - 1) / fa::BQ, n_head), 128, 0, s>>>(Qm, Kh, Vh, T, n_head, pos0, kb0, nb,
                                                                                    first, last, Oacc, ML, out);
    check("mla_fa_prefill");
}

void f32_to_f16_scaled(const float* x, __half* y, size_t n, float scale, cudaStream_t s) {
    if (n == 0) return;
    f32_to_f16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, y, n, scale);
    check("f32_to_f16");
}

void mla_softmax_causal(const float* S, __half* P, int n_kv, int cols, int n_head, int pos0, cudaStream_t s) {
    if (cols <= 0) return;
    softmax_causal_kernel<<<cols, 256, 0, s>>>(S, P, n_kv, n_head, pos0);
    check("softmax_causal");
}

void router_topk(const float* logits, const float* bias, int n_expert, int k, int T, int gating, bool norm, float scale,
                 int32_t* ids, float* w, cudaStream_t s) {
    if (T <= 0) return;
    router_kernel<<<T, 128, 2 * n_expert * sizeof(float), s>>>(logits, bias, n_expert, k, gating, norm, scale, ids, w);
    check("router");
}

void group_hits(const int32_t* ids, int n_routes, int k, const int32_t* slot_of, int n_expert, const uint8_t* slot_base,
                size_t slot_stride, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                int32_t* ent_tok, cudaStream_t s) {
    group_hits_kernel<<<1, 256, 3 * n_expert * sizeof(int), s>>>(ids, n_routes, k, slot_of, n_expert, slot_base,
                                                                  slot_stride, grp_ptr, grp_start, n_groups, ent_dst,
                                                                  ent_tok);
    check("group_hits");
}

void moe_combine(float* x, const float* gpu_parts, const float* cpu_parts, const int32_t* cpu_row, const float* shared,
                 const int32_t* ids, const float* w, const int32_t* slot_of, int T, int k, int n_embd, cudaStream_t s) {
    if (T <= 0) return;
    moe_combine_kernel<<<T, 256, 0, s>>>(x, gpu_parts, cpu_parts, cpu_row, shared, ids, w, slot_of, k, n_embd);
    check("moe_combine");
}

void publish(const float* x, int T, int n_embd, const int32_t* ids, const float* w, int n_routes, float* hx,
             int32_t* hids, float* hw, volatile int* flag, const volatile int* seq, cudaStream_t s) {
    publish_kernel<<<1, 1024, 0, s>>>(x, T * n_embd, ids, w, n_routes, hx, hids, hw, flag, seq);
    check("publish");
}

void wait_host(const volatile int* flag, const volatile int* seq, volatile int* err, cudaStream_t s) {
    wait_host_kernel<<<1, 1, 0, s>>>(flag, seq, err);
    check("wait_host");
}

void argmax_rows(const float* x, int n, int rows, int32_t* out, cudaStream_t s) {
    if (rows <= 0) return;
    argmax_kernel<<<rows, 1024, 0, s>>>(x, n, out);
    check("argmax");
}

void scatter_rows(const float* src, const int32_t* rows, int n_rows, int n_embd, float* out, cudaStream_t s) {
    if (n_rows <= 0) return;
    scatter_rows_kernel<<<n_rows, 256, 0, s>>>(src, rows, n_embd, out);
    check("scatter_rows");
}

void concat2(const float* a, const float* b, float* out, int n, int rows, cudaStream_t s) {
    if (rows <= 0) return;
    concat2_kernel<<<rows, 256, 0, s>>>(a, b, out, n);
    check("concat2");
}

void argmax_prob(const float* x, int n, int32_t* out, float* prob, cudaStream_t s) {
    argmax_prob_kernel<<<1, 1024, 0, s>>>(x, n, out, prob);
    check("argmax_prob");
}

void rows_f16_scaled(const float* x, const int32_t* rows, int n, int cols, __half* out, float* inv, cudaStream_t s) {
    if (n <= 0) return;
    rows_f16_scaled_kernel<<<n, 256, 0, s>>>(x, rows, cols, out, inv);
    check("rows_f16_scaled");
}

void swiglu_f16_scaled(const float* gu, const float* inv_in, int n, int F, __half* h, float* inv_out, cudaStream_t s) {
    if (n <= 0) return;
    swiglu_f16_scaled_kernel<<<n, 256, 0, s>>>(gu, inv_in, F, h, inv_out);
    check("swiglu_f16_scaled");
}

void scale_rows(float* y, int n, int cols, const float* inv, cudaStream_t s) {
    if (n <= 0) return;
    scale_rows_kernel<<<n, 256, 0, s>>>(y, cols, inv);
    check("scale_rows");
}

}  // namespace m4k
