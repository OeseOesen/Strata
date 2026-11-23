// engine/src/mistral4/m4_kernels.cu - see m4_kernels.cuh.
#include "m4_kernels.cuh"

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
                                    int kv_mode, void* cache, float* scales, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.x, d = threadIdx.x;
    const float* r = kvp + (size_t) t * kKD;
    const int p = pos[t];
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
        __half* row = static_cast<__half*>(cache) + (size_t) p * kKD;
        row[d] = __float2half(lat);
        if (d < kRope) row[kLat + d] = __float2half(rp);
        return;
    }
    const float am = block_max(fabsf(lat), red);
    const float ar = block_max(d < kRope ? fabsf(rp) : 0.f, red);
    const float s0 = am > 0.f ? am / 127.f : 1.f, s1 = ar > 0.f ? ar / 127.f : 1.f;
    int8_t* row = static_cast<int8_t*>(cache) + (size_t) p * kKD;
    row[d] = (int8_t) __float2int_rn(fminf(127.f, fmaxf(-127.f, lat / s0)));
    if (d < kRope) row[kLat + d] = (int8_t) __float2int_rn(fminf(127.f, fmaxf(-127.f, rp / s1)));
    if (d == 0) {
        scales[(size_t) p * 2 + 0] = s0;
        scales[(size_t) p * 2 + 1] = s1;
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
                                                         float* part) {
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
        __syncthreads();                     // (a) the previous tile's readers are done; q2 / m_s are written
        for (int i = tid; i < TILE * (kKD / 2); i += 256) {
            const int r = i / (kKD / 2), w = i - r * (kKD / 2);
            float2 f = make_float2(0.f, 0.f);
            if (r < nk) {
                const size_t key = (size_t) (kt + r);
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

__global__ void cache_to_f16_kernel(const void* cache, const float* scales, int kv_mode, int n, __half* K16) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * kKD) return;
    if (kv_mode == KV_F16) {
        K16[i] = static_cast<const __half*>(cache)[i];
        return;
    }
    const size_t r = i / kKD, d = i % kKD;
    K16[i] = __float2half((float) static_cast<const int8_t*>(cache)[i] * scales[r * 2 + (d >= (size_t) kLat ? 1 : 0)]);
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
                  void* cache, float* scales, float eps, cudaStream_t s) {
    if (T <= 0) return;
    mla_kv_write_kernel<<<T, kLat, 0, s>>>(kvp, kv_norm, reinterpret_cast<const float2*>(rope_tab), pos, kv_mode, cache,
                                            scales, eps);
    check("mla_kv_write");
}

void mla_q_rope(const float* q, float* Qf, const float* rope_tab, const int32_t* pos, int T, int n_head, float temp_scale,
                float temp_floor, cudaStream_t s) {
    if (T <= 0) return;
    mla_q_rope_kernel<<<dim3(T, n_head), kRope, 0, s>>>(q, Qf, reinterpret_cast<const float2*>(rope_tab), pos, n_head,
                                                         temp_scale, temp_floor);
    check("mla_q_rope");
}

size_t mla_decode_part_floats(int T, int n_head) { return (size_t) T * kAttnChunks * n_head * (kLat + 2); }

void mla_attn_decode(const float* Qf, const void* cache, const float* scales, int kv_mode, const int32_t* pos, int T,
                     int n_head, float kq_scale, float* part, float* O, cudaStream_t s) {
    if (n_head > kMaxHeads) {
        std::fprintf(stderr, "strata-m4: %d heads, the decode attention takes %d\n", n_head, kMaxHeads);
        std::exit(1);
    }
    mla_decode_kernel<<<dim3(kAttnChunks, T), 256, 0, s>>>(Qf, cache, scales, kv_mode, pos, n_head, kq_scale, part);
    check("mla_decode");
    mla_merge_kernel<<<dim3(T, n_head), kLat, 0, s>>>(part, O, n_head);
    check("mla_merge");
}

void mla_cache_to_f16(const void* cache, const float* scales, int kv_mode, int n, __half* K16, cudaStream_t s) {
    const size_t tot = (size_t) n * kKD;
    if (tot == 0) return;
    cache_to_f16_kernel<<<(unsigned) ((tot + 255) / 256), 256, 0, s>>>(cache, scales, kv_mode, n, K16);
    check("cache_to_f16");
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

}  // namespace m4k
