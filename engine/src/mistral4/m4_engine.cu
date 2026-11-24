// engine/src/mistral4/m4_engine.cu - strata-m4: Mistral Small 4 (GGUF architecture "mistral4", 119B-A6.5B MoE, MLA) on
// one GPU plus a two-socket host, built on Strata's quantized kernels.
//
//   GPU (VRAM):  every dense weight (MLA projections, shared expert, router, head, embeddings), the MLA latent KV cache
//                in int8 (or f16), the hottest routed experts as native expert blobs ("slots"), the scratch.
//   GPU node:    the next-hottest experts, in memory bound to the GPU's NUMA node, computed by that node's cores.
//   Other node:  the coldest experts and the global n-gram table, computed by that node's own cores (near-data:
//                only activations and results cross the socket link).
//   File tier:   experts that fit nowhere stay in the mapped GGUF (page cache / NVMe), computed on the other node.
//
// A decode window (1 token + up to 15 n-gram drafts) is one CUDA graph: per layer the GPU publishes the routes to
// mapped host memory, computes its own experts and the shared expert while the host's NUMA pool computes the misses,
// and waits on a host flag before the combine.  Prompts run in chunks, eagerly, with cuBLAS attention.
//
// Reference semantics: llama.cpp's deepseek2 graph (which mistral4 reuses): MLA with absorbed wk_b / wv_b,
// YaRN rope (NORM pairs) on the 64 rope dims, softmax router top-4 (+ norm / scale from the GGUF), shared expert added.
#include "m4_kernels.cuh"
#include "m4_ngram.hpp"
#include "m4_numa.hpp"

#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <functional>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using json = nlohmann::json;
namespace K = strata::kernels;
namespace C = strata::kernels::cpu;

namespace m4 {
namespace {

#define CK(x)                                                                                                   \
    do {                                                                                                        \
        const cudaError_t e_ = (x);                                                                             \
        if (e_ != cudaSuccess) {                                                                                \
            std::fprintf(stderr, "strata-m4: CUDA error %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, \
                         __LINE__, #x);                                                                         \
            std::exit(1);                                                                                       \
        }                                                                                                       \
    } while (0)
#define CB(x)                                                                                          \
    do {                                                                                               \
        const cublasStatus_t m4_st_ = (x);                                                                 \
        if (m4_st_ != CUBLAS_STATUS_SUCCESS) {                                                             \
            std::fprintf(stderr, "strata-m4: cuBLAS error %d at %s:%d: %s\n", (int) m4_st_, __FILE__, __LINE__, #x); \
            std::exit(1);                                                                              \
        }                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
constexpr int kMaxWin = 16;   // decode window: 1 + up to 15 drafts

void logf_(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void logf_(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "strata-m4: ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
}
[[noreturn]] void fail(const std::string& why) {
    logf_("error: %s", why.c_str());
    std::exit(1);
}

// q8_1 activations for T contiguous columns: Strata's quantizer takes at most 8 columns, but columns are contiguous and
// n_in is a multiple of 32, so T columns quantize as one vector of n_in * T values (no block crosses a column)
void q8(const float* x, void* xq, int n_in, int T, cudaStream_t s) { K::native_quantize_q8_1(x, xq, n_in * T, 1, s); }
size_t q8_bytes(int n_in, int T) { return (size_t) T * (size_t) (n_in / 32) * 36; }

// ================================================================================================ options
struct Options {
    std::string model, heat, mode = "serve", ids_file, dump_ids, dump_layers = "0,1";
    int ctx = 65536, chunk = 512, spec = 6, kv = m4k::KV_I8, workers = -1, host_core = -1, bench_pp = 512,
        bench_tg = 128, ngram_mib = 1024, seed = 0, gpu = 0, ppl_ctx = 512, attn_impl = m4k::ATTN_TC;
    double attn_temp_scale = -1;   // < 0: the GGUF's
    // prompt path: experts not in VRAM streamed to the GPU over PCIe (pinned arenas, a ring of slots on a copy stream)
    // and every product of a chunk on the tensor cores (weights dequantized to f16, cuBLAS) - Strata's prompt trick
    int pf_stream = 1, pf_min = 768, pf_ring = 12;
    bool pf_gemm = true, chunk_set = false;
    double vram_reserve = 1.5, vram_experts = -1, ram_gpu_node = -1, ram_other_node = -1;
    std::string eagle;            // EAGLE draft head: Mistral's consolidated.safetensors (or its folder)
    int eagle_k = 3;              // drafts per window at most (Mistral recommends 3)
    double eagle_p = 0.3;         // stop drafting when the chain's probability (product of the steps') falls below
    std::string numa = "split";   // measured on the R740 (2x Gold 6154): split 42.6 tok/s decode vs tiered 35.3
    bool graph = true, keep_file_tier = true, attn_gemm = false, attn_temp = true;
};

void usage() {
    std::fprintf(stderr,
                 "strata-m4: Mistral Small 4 on one GPU + a NUMA host\n"
                 "  -m FILE.gguf             the model (first shard of a split file is enough)\n"
                 "  --serve                  line protocol on stdin/stdout (default; mistral4/m4serve.py speaks it)\n"
                 "  --bench                  random prompt (--bench-pp N) then --bench-tg N tokens, greedy\n"
                 "  --ppl IDS.txt            perplexity over token ids (whitespace separated), chunks of --chunk\n"
                 "  --dump 1,2,3             debug: run these token ids as one prompt and print every intermediate of the\n"
                 "                           --dump-layers (default 0,1) in llama-eval-callback's names, then the top logits\n"
                 "  --ppl-ctx N              perplexity chunk length (default 512; the second half of each chunk is scored)\n"
                 "  --no-attn-temp           debug: ignore the GGUF's attention temperature scaling (llama-4 style, > 8k)\n"
                 "  --ctx N                  context (default 65536; the int8 latent cache costs 11.5 KiB per token)\n"
                 "  --kv int8|f16            latent cache format (default int8)\n"
                 "  --chunk N                prompt chunk (default 2048 with --pf-stream 1, else 512)\n"
                 "  --pf-stream 0|1          prompt chunks of --pf-min tokens or more stream the experts that are not in VRAM\n"
                 "                           to the GPU (pinned arenas, DMA on a copy stream) and run every expert there on\n"
                 "                           the tensor cores; 0 = the CPU pool computes them (default 1)\n"
                 "  --pf-min N               smallest chunk that streams (default 768; below it the CPU pool is faster)\n"
                 "  --pf-ring N              expert slots of the stream ring in VRAM (default 12)\n"
                 "  --pf-gemm 0|1            prompt chunks of 64+ tokens: dense products as f16 tensor-core GEMMs (default 1)\n"
                 "  --spec N                 n-gram draft tokens per window, 0 = off (default 6, max 15)\n"
                 "  --vram-reserve GiB       VRAM left free (default 1.5)\n"
                 "  --vram-experts GiB       cap on VRAM expert slots (default: all that is left)\n"
                 "  --numa split|tiered|off  CPU expert placement (default split: every CPU expert halved over both nodes)\n"
                 "  --ram-gpu-node GiB       expert arena on the GPU's node (default: free - 6 GiB)\n"
                 "  --ram-other-node GiB     expert arena on the other node (default: free - 4 GiB - n-gram table)\n"
                 "  --workers N              CPU workers per node (default: every physical core)\n"
                 "  --host-core N            the CPU the GPU-feeding host thread sits on (default: first core of the GPU node)\n"
                 "  --heat FILE              expert routing counts (default <model>.m4heat; written at exit)\n"
                 "  --eagle PATH             EAGLE draft head (Mistral-Small-4-119B-2603-eagle: consolidated.safetensors\n"
                 "                           or its folder); drafts when the n-gram lookup has none\n"
                 "  --eagle-k N              EAGLE drafts per window at most (default 3)\n"
                 "  --eagle-p P              stop the EAGLE chain below this probability (product, default 0.3)\n"
                 "  --eagle-acc IDS.txt      debug: teacher-forced top-1 accuracy of the model and of the EAGLE head\n"
                 "  --ngram-mib N            global n-gram table on the other node (default 1024, 0 = off)\n"
                 "  --no-graph               launch decode windows eagerly (debugging)\n"
                 "  --attn tc|kernel|gemm    decode attention: split-K on the tensor cores (default), split-K on the CUDA cores,\n"
                 "                           or the prompt path's cuBLAS route (no CUDA graphs then)\n"
                 "  --attn-temp-scale X      override the GGUF's llama-4 attention temperature (beta; 0.1 from 8192 on is what\n"
                 "                           Mistral's own params.json says; the unsloth GGUF carries none)\n"
                 "  --selftest-attn          check the attention kernels against a CPU reference on random data (no model)\n"
                 "  --gpu N                  CUDA device (default 0)\n");
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) fail("missing value after " + a);
            return argv[++i];
        };
        if (a == "-m" || a == "--model") o.model = next();
        else if (a == "--eagle") o.eagle = next();
        else if (a == "--eagle-k") o.eagle_k = std::stoi(next());
        else if (a == "--eagle-p") o.eagle_p = std::stod(next());
        else if (a == "--eagle-acc") { o.mode = "eagle-acc"; o.ids_file = next(); }
        else if (a == "--serve") o.mode = "serve";
        else if (a == "--bench") o.mode = "bench";
        else if (a == "--ppl") { o.mode = "ppl"; o.ids_file = next(); }
        else if (a == "--ppl-ctx") o.ppl_ctx = std::stoi(next());
        else if (a == "--no-attn-temp") o.attn_temp = false;
        else if (a == "--dump") { o.mode = "dump"; o.dump_ids = next(); }
        else if (a == "--dump-layers") o.dump_layers = next();
        else if (a == "--ctx") o.ctx = std::stoi(next());
        else if (a == "--kv") { const std::string v = next(); o.kv = v == "f16" ? m4k::KV_F16 : m4k::KV_I8; }
        else if (a == "--chunk") { o.chunk = std::stoi(next()); o.chunk_set = true; }
        else if (a == "--pf-stream") o.pf_stream = std::stoi(next());
        else if (a == "--pf-min") o.pf_min = std::stoi(next());
        else if (a == "--pf-ring") o.pf_ring = std::stoi(next());
        else if (a == "--pf-gemm") o.pf_gemm = std::stoi(next()) != 0;
        else if (a == "--spec") o.spec = std::stoi(next());
        else if (a == "--vram-reserve") o.vram_reserve = std::stod(next());
        else if (a == "--vram-experts") o.vram_experts = std::stod(next());
        else if (a == "--numa") o.numa = next();
        else if (a == "--ram-gpu-node") o.ram_gpu_node = std::stod(next());
        else if (a == "--ram-other-node") o.ram_other_node = std::stod(next());
        else if (a == "--workers") o.workers = std::stoi(next());
        else if (a == "--host-core") o.host_core = std::stoi(next());
        else if (a == "--heat") o.heat = next();
        else if (a == "--ngram-mib") o.ngram_mib = std::stoi(next());
        else if (a == "--no-graph") o.graph = false;
        else if (a == "--attn") {
            const std::string v = next();
            if (v != "gemm" && v != "tc" && v != "kernel") fail("--attn tc|kernel|gemm");
            o.attn_gemm = v == "gemm";
            o.attn_impl = v == "kernel" ? m4k::ATTN_SIMT : m4k::ATTN_TC;
        }
        else if (a == "--attn-temp-scale") o.attn_temp_scale = std::stod(next());
        else if (a == "--selftest-attn") o.mode = "selftest-attn";
        else if (a == "--bench-pp") o.bench_pp = std::stoi(next());
        else if (a == "--bench-tg") o.bench_tg = std::stoi(next());
        else if (a == "--seed") o.seed = std::stoi(next());
        else if (a == "--gpu") o.gpu = std::stoi(next());
        else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        else fail("unknown option " + a + " (--help)");
    }
    if (o.model.empty() && o.mode != "selftest-attn") { usage(); fail("no model (-m)"); }
    if (o.attn_gemm) o.graph = false;   // the cuBLAS route's sizes follow the context: not capturable
    o.spec = std::max(0, std::min(o.spec, kMaxWin - 1));
    o.eagle_k = std::max(1, std::min(o.eagle_k, kMaxWin - 1));
    if (o.mode == "eagle-acc" && o.eagle.empty()) fail("--eagle-acc needs --eagle");
    if (!o.chunk_set && o.pf_stream) o.chunk = 2048;
    o.chunk = std::max(kMaxWin + 1, o.chunk);
    o.pf_ring = std::max(2, std::min(o.pf_ring, 64));
    if (o.heat.empty()) o.heat = o.model + ".m4heat";
    return o;
}

// ================================================================================================ model
struct Hp {
    int n_layer = 0, n_embd = 0, n_head = 0, q_lora = 0, kv_lora = 0, nope = 0, rope = 0, v_dim = 0, n_vocab = 0;
    int n_expert = 0, top_k = 0, n_ff = 0, n_shexp = 0, dense_lead = 0, n_ff_dense = 0;
    float eps = 1e-6f, rope_base = 10000.f, freq_scale = 1.f, ext_factor = 0.f, attn_factor = 1.f, beta_fast = 32.f,
          beta_slow = 1.f, kq_scale = 0.f, w_scale = 1.f, temp_scale = 0.f, temp_floor = 8192.f;
    int n_ctx_orig = 0, gating = 1;
    bool w_norm = false;
};

struct Mat {
    const uint8_t* d = nullptr;   // quantized GGUF blocks on the device
    float* f = nullptr;           // or f32 [n_out][n_in] on the device
    int type = -1, n_in = 0, n_out = 0;
    size_t bytes() const { return f ? (size_t) n_in * n_out * 4 : K::native_mmvq_weight_bytes(type, n_in, n_out); }
};

struct Layer {
    float *attn_norm = nullptr, *q_a_norm = nullptr, *kv_a_norm = nullptr, *ffn_norm = nullptr;
    Mat wq_a, wq_b, wkv_a, wo;
    float *wk_b = nullptr, *wv_b = nullptr;   // f32 [h][kv_lora][nope], [h][v_dim][kv_lora]
    bool dense = false;
    Mat ffn_gate, ffn_up, ffn_down;           // the leading dense layers (none in Mistral Small 4)
    float* router = nullptr;                  // f32 [n_expert][n_embd]
    float* probs_b = nullptr;
    Mat sh_gate, sh_up, sh_down;
    // routed experts
    int gu_type = -1, d_type = -1;
    const uint8_t *src_gate = nullptr, *src_up = nullptr, *src_down = nullptr;   // mapped file, expert 0
    size_t src_gu_stride = 0, src_d_stride = 0;                                  // bytes between experts
    K::NativeExpertLayout L;
    C::NativeFmt fmt, half_fmt[2];
    size_t stride = 0;                        // bytes per slot (L.bytes rounded up)
    uint8_t* slots = nullptr;
    int n_slots = 0;
    std::vector<int> slot_of;                 // host copy
    int32_t* d_slot_of = nullptr;
    std::vector<CpuExpert> cpu;               // n_expert (unused entries for experts in VRAM)
    std::vector<std::unique_ptr<C::NativeFmt>> file_fmt;   // per-expert layouts of file-tier experts
    // cache
    void* kv = nullptr;
    float* kv_s = nullptr;
};

const uint8_t* tensor_ptr(const strata::GgufModel& gm, const strata::TensorInfo* t) {
    size_t sh = 0;
    gm.find(t->name, &sh);
    return gm.shard(sh).tensor_data(*t);
}

/// Any tensor as host f32 (dense small tensors: norms, router, wk_b, wv_b).
std::vector<float> host_f32(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h((size_t) t.elements());
    auto blocks = [&](size_t qk, size_t bb, void (*f)(const uint8_t*, float*)) {
        if (h.size() % qk) fail(t.name + ": not whole blocks");
        for (size_t b = 0; b < h.size() / qk; ++b) f(src + b * bb, h.data() + b * qk);
    };
    switch (t.type) {
        case 0: std::memcpy(h.data(), src, h.size() * 4); break;
        case 1: strata::dequantize_f16(src, h.data(), (int) h.size()); break;
        case 30: strata::dequantize_bf16(src, h.data(), (int) h.size()); break;
        case 8: blocks(32, 34, strata::dequantize_q8_0); break;
        case 2: blocks(32, 18, strata::dequantize_q4_0); break;
        case 6: blocks(32, 22, strata::dequantize_q5_0); break;
        case 7: blocks(32, 24, strata::dequantize_q5_1); break;
        case 11: blocks(256, 110, strata::dequantize_q3_K); break;
        case 12: blocks(256, 144, strata::dequantize_q4_K); break;
        case 13: blocks(256, 176, strata::dequantize_q5_K); break;
        case 14: blocks(256, 210, strata::dequantize_q6_K); break;
        case 20: blocks(32, 18, strata::dequantize_iq4_nl); break;
        case 23: blocks(256, 136, strata::dequantize_iq4_xs); break;
        default: fail(t.name + " is " + t.type_name() + ", which strata-m4 cannot widen to f32");
    }
    return h;
}

float* upload(const std::vector<float>& h) {
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

// ================================================================================================ safetensors (EAGLE head)
// Mistral ships the EAGLE head as consolidated.safetensors: every linear weight float8 e4m3 with a per-tensor
// bf16 scale (qscale_weight; the activation scale qscale_act is not needed: activations stay q8_1 here), norms bf16.
struct StTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t bytes = 0;
};

struct Safetensors {
    void* map = nullptr;
    size_t size = 0;
    std::map<std::string, StTensor> t;
    ~Safetensors() {
        if (map) munmap(map, size);
    }
    void open(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) fail("cannot open " + path);
        struct stat stt;
        if (fstat(fd, &stt) != 0) fail("cannot stat " + path);
        size = (size_t) stt.st_size;
        void* m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) fail("cannot map " + path);
        map = m;
        const uint8_t* b = (const uint8_t*) map;
        uint64_t hl = 0;
        if (size < 8) fail(path + ": not a safetensors file");
        std::memcpy(&hl, b, 8);
        if (8 + hl > size) fail(path + ": not a safetensors file");
        const json h = json::parse(std::string((const char*) b + 8, (size_t) hl));
        const uint8_t* base = b + 8 + hl;
        for (auto it = h.begin(); it != h.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            StTensor x;
            x.dtype = it.value().at("dtype").get<std::string>();
            x.shape = it.value().at("shape").get<std::vector<int64_t>>();
            const auto off = it.value().at("data_offsets").get<std::vector<size_t>>();
            if (off.size() != 2 || off[1] < off[0] || base + off[1] > b + size) fail(path + ": bad offsets of " + it.key());
            x.data = base + off[0];
            x.bytes = off[1] - off[0];
            t[it.key()] = x;
        }
    }
    const StTensor* find(const std::string& n) const {
        auto it = t.find(n);
        return it == t.end() ? nullptr : &it->second;
    }
};

float bf16_to_f32(uint16_t v) {
    const uint32_t u = (uint32_t) v << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float fp8_e4m3_to_f32(uint8_t v) {   // e4m3fn: bias 7, no infinities, S.1111.111 is NaN
    const int sgn = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float r;
    if (e == 15 && m == 7) r = NAN;
    else if (e == 0) r = std::ldexp((float) m / 8.f, -6);
    else r = std::ldexp(1.f + (float) m / 8.f, e - 7);
    return sgn ? -r : r;
}

std::vector<float> st_f32(const StTensor& x, const std::string& name) {
    size_t n = 1;
    for (int64_t d : x.shape) n *= (size_t) d;
    std::vector<float> h(n);
    if (x.dtype == "F8_E4M3") {
        if (x.bytes != n) fail(name + ": size");
        float lut[256];
        for (int i = 0; i < 256; ++i) lut[i] = fp8_e4m3_to_f32((uint8_t) i);
        for (size_t i = 0; i < n; ++i) h[i] = lut[x.data[i]];
    } else if (x.dtype == "BF16") {
        if (x.bytes != n * 2) fail(name + ": size");
        for (size_t i = 0; i < n; ++i) {
            uint16_t v;
            std::memcpy(&v, x.data + i * 2, 2);
            h[i] = bf16_to_f32(v);
        }
    } else if (x.dtype == "F16") {
        if (x.bytes != n * 2) fail(name + ": size");
        strata::dequantize_f16(x.data, h.data(), (int) n);
    } else if (x.dtype == "F32") {
        if (x.bytes != n * 4) fail(name + ": size");
        std::memcpy(h.data(), x.data, n * 4);
    } else {
        fail(name + " is " + x.dtype + ", which the EAGLE loader does not read");
    }
    return h;
}

/// f32 rows [n_out][n_in] as GGUF Q8_0 blocks (llama.cpp's quantize_row_q8_0_ref), uploaded with the native GEMV's tail.
uint8_t* upload_q8_0(const std::vector<float>& w, int n_in, int n_out) {
    const size_t nb = (size_t) n_in / 32, b = (size_t) n_out * nb * 34;
    if (b != K::native_mmvq_weight_bytes(8, n_in, n_out)) fail("Q8_0 layout mismatch");
    std::vector<uint8_t> q(b);
    for (size_t r = 0; r < (size_t) n_out; ++r)
        for (size_t k = 0; k < nb; ++k) {
            const float* x = w.data() + r * (size_t) n_in + k * 32;
            float amax = 0.f;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
            const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
            uint8_t* blk = q.data() + (r * nb + k) * 34;
            const __half hd = __float2half(d);
            std::memcpy(blk, &hd, 2);
            for (int j = 0; j < 32; ++j) blk[2 + j] = (uint8_t) (int8_t) std::lrintf(x[j] * id);
        }
    uint8_t* d = nullptr;
    CK(cudaMalloc(&d, b + 256));
    CK(cudaMemset(d + b, 0, 256));
    CK(cudaMemcpy(d, q.data(), b, cudaMemcpyHostToDevice));
    return d;
}

// ================================================================================================ the engine
struct HostShared {          // mapped, pinned host memory the decode graph talks through
    volatile int seq;
    volatile int err;
    volatile int pub[256];
    volatile int done[256];
};

class Engine {
public:
    explicit Engine(const Options& o) : opt_(o) {}
    void load();
    // generation
    struct GenParams {
        int max_new = 256;
        float temp = 0.f, top_p = 1.f, min_p = 0.f;
        int top_k = 0, spec = -1;
        uint64_t seed = 0;
        std::vector<int32_t> stop_ids;
        json embd;            // [{file, rows:[...]}] image rows (prompt positions)
        std::string embd_key; // identity of the images (prefix cache)
    };
    struct GenStats {
        int n_prompt = 0, n_reused = 0, n_gen = 0, drafted = 0, accepted = 0, windows = 0, e_drafted = 0,
            e_accepted = 0, e_windows = 0;
        double prompt_ms = 0, gen_ms = 0;
        std::string finish = "length";
    };
    struct GenHooks {   // optional live reports (the serve protocol's PP and W lines)
        std::function<void(int, int, int)> progress;   // prompt tokens read, to read, reused
        std::function<void(int, int)> window;          // rows in the window, rows taken
    };
    GenStats generate(const std::vector<int32_t>& prompt, const GenParams& gp,
                      const std::function<bool(const std::vector<int32_t>&)>& emit, const std::atomic<bool>* stop,
                      const GenHooks* hooks = nullptr);
    double perplexity(const std::vector<int32_t>& ids, int chunk);
    void save_heat() const;
    const Hp& hp() const { return hp_; }
    json info() const;
    std::string placement_summary() const { return placement_; }

private:
    // loading
    void read_hparams();
    Mat load_mat(const std::string& name, int n_in, int n_out, bool required = true);
    float* load_f32(const std::string& name, bool required = true);
    void load_layers();
    void alloc_buffers();
    void place_experts();
    // forward
    void mm(const Mat& m, const float* x, const void* xq, float* y, int T, cudaStream_t s);
    void attention(Layer& L, int T, bool window, int pos0, const int32_t* pos = nullptr, float temp_scale = -1.f,
                   float temp_floor = 0.f);
    void moe_window(int l, int T);
    void moe_prefill(int l, int T);
    void moe_prefill_stream(int l, int T);
    bool stream_ok(const Layer& L) const;
    void check_ids(int l, const int32_t* ids, int R) const;
    void ensure_w16(size_t elems);
    void ensure_x16(size_t elems);
    void register_arenas();
    void ffn_dense(Layer& L, int T);
    void embed(const int32_t* d_tokens, int T);
    void enqueue_window(int T);
    void run_window(const std::vector<int32_t>& toks, int pos0, bool all_logits);
    void serve_host_layers(int T);
    void prefill(const int32_t* toks, int T, int pos0, const GenParams* gp, int prompt_off, bool all_logits);
    void head(int rows_from, int n_rows);   // logits for rows [rows_from, +n) of x
    int sample(const float* logits, const GenParams& gp, std::mt19937_64& rng);
    void count_heat(int l, const int32_t* ids, int n);
    // EAGLE draft head
    void load_eagle();
    void eagle_ffn(Layer& L, int T);
    void eagle_forward(const int32_t* d_tokens, const float* hid, int T, int pos0, const int32_t* d_pos);
    void eagle_pick(const float* row, int slot);
    void eagle_follow(const int32_t* next, int T, int pos0, bool rows_normed);
    void eagle_chain(int P, int k, double pmin, std::vector<int32_t>& out);
    bool eagle_on_ = false, eagle_ready_ = false;
    int eagle_valid_ = 0;          // positions [0, eagle_valid_) have valid EAGLE KV rows
    Mat efc_;
    std::vector<Layer> elayers_;
    float* enorm_ = nullptr;
    float e_temp_ = 0.f, e_floor_ = 8192.f;
    int e_ff_ = 0, e_rows_ = 512;
    const float* e_last_ = nullptr;
    float *mh_ = nullptr, *eemb_ = nullptr, *ecat_ = nullptr, *eg_ = nullptr, *eu_ = nullptr, *eh2_ = nullptr,
          *eh_ = nullptr, *elogits_ = nullptr, *d_eprob_ = nullptr;
    void *ecatq_ = nullptr, *ehq_ = nullptr;
    int32_t *d_etok_ = nullptr, *d_edraft_ = nullptr, *d_epos_ = nullptr, *petok_ = nullptr;
public:
    double eagle_acc(const std::vector<int32_t>& ids);
    void dump(const std::vector<int32_t>& ids);
private:
    void dbg(const char* name, int l, const float* d, int rows, int cols);
    void dbg_i(const char* name, int l, const int32_t* d, int rows, int cols);
    bool dbg_on_ = false;
    std::vector<int> dbg_layers_;

    Options opt_;
    Hp hp_;
    std::shared_ptr<strata::GgufModel> gm_;
    std::vector<Layer> layers_;
    float* out_norm_ = nullptr;
    Mat head_;
    const uint8_t* embd_ = nullptr;
    int embd_type_ = -1;
    size_t embd_row_ = 0;
    int bos_ = -1;
    float* rope_ = nullptr;
    Topology topo_;
    int gpu_node_ = -1, other_node_ = -1, host_cpu_ = -1, host_sib_ = -1;
    int win_pos0_ = 0;   // the current window's first position (the --attn gemm route needs it on the host)
    std::unique_ptr<NumaPool> pool_;
    NgramDrafter ngram_;
    void* ngram_mem_ = nullptr;
    size_t ngram_bytes_ = 0;
    std::vector<std::pair<void*, size_t>> arenas_;
    // the streamed prompt path
    cudaStream_t cs_ = nullptr;                       // copy stream
    std::vector<cudaEvent_t> ev_copied_, ev_free_;     // per ring slot
    uint8_t* ring_ = nullptr;
    size_t ring_stride_ = 0;
    uint64_t ring_seq_ = 0;                            // ring uses so far (slot = seq % ring size)
    __half *W16_ = nullptr, *X16_ = nullptr, *H16_ = nullptr;
    size_t W16_elems_ = 0, X16_elems_ = 0;
    float* GU_ = nullptr;
    float *d_xinv_ = nullptr, *d_hinv_ = nullptr;   // per-row unscale factors of the f16 GEMM operands
    int32_t *d_rpos_ = nullptr, *d_noslot_ = nullptr, *d_rows_ = nullptr, *prpos_ = nullptr, *prows_ = nullptr;
    bool pinned_arenas_ = false;
    double pf_stream_bytes_ = 0;
    std::vector<std::vector<uint64_t>> heat_;
    std::string placement_;
    // device buffers
    cudaStream_t s_ = nullptr;
    cublasHandle_t cb_ = nullptr;
    void* cb_ws_ = nullptr;
    int Tm_ = 0;   // rows the buffers hold
    int32_t *d_tok_ = nullptr, *d_pos_ = nullptr, *d_ids_ = nullptr, *d_argmax_ = nullptr, *d_cpu_row_ = nullptr;
    float *x_ = nullptr, *xn_ = nullptr, *qa_ = nullptr, *kvp_ = nullptr, *q_ = nullptr, *Qf_ = nullptr, *Olat_ = nullptr,
          *attn_ = nullptr, *ao_ = nullptr, *rlog_ = nullptr, *w_ = nullptr, *g_ = nullptr, *u_ = nullptr, *h_ = nullptr,
          *sh_ = nullptr, *gpu_parts_ = nullptr, *cpu_parts_ = nullptr, *part_ = nullptr, *logits_ = nullptr, *S_ = nullptr,
          *img_ = nullptr;
    void *xq_ = nullptr, *qaq_ = nullptr, *aq_ = nullptr, *hq_ = nullptr, *xq_exp_ = nullptr, *exp_scratch_ = nullptr;
    __half *K16_ = nullptr, *Q16_ = nullptr, *P_ = nullptr;
    size_t S_floats_ = 0;
    unsigned long long* grp_ptr_ = nullptr;
    int32_t *grp_start_ = nullptr, *n_groups_ = nullptr, *ent_dst_ = nullptr, *ent_tok_ = nullptr;
    int32_t* d_img_rows_ = nullptr;
    // host side
    HostShared* hs_ = nullptr;
    HostShared* hs_dev_ = nullptr;
    float *hx_ = nullptr, *hw_ = nullptr, *hcpu_ = nullptr;      // mapped: MoE input rows, weights, CPU outputs
    float *hx_dev_ = nullptr, *hw_dev_ = nullptr, *hcpu_dev_ = nullptr;
    int32_t *hids_ = nullptr, *hids_dev_ = nullptr;
    float* px_ = nullptr;          // pinned (prefill): MoE input rows
    int32_t* pids_ = nullptr;      // pinned (prefill): routes
    float* pcpu_ = nullptr;        // pinned (prefill): compact CPU outputs
    int32_t* pcpu_row_ = nullptr;  // pinned (prefill): route -> compact row
    int32_t* ptok_ = nullptr;      // pinned: tokens and positions of a window
    std::vector<float> hlogits_;
    std::map<int, cudaGraphExec_t> graphs_;
    std::vector<int32_t> cached_;  // tokens whose KV is valid, positions 0..
    std::string cached_embd_key_;
    CpuCall call_;
    int seq_ = 0;
    // stats
    uint64_t routes_total_ = 0, routes_vram_ = 0, routes_gpu_node_ = 0, routes_other_ = 0;
};

// ------------------------------------------------------------------------------------------------ hyper-parameters
void Engine::read_hparams() {
    const auto& m = gm_->meta();
    const strata::MetaValue* a = m.get("general.architecture");
    const std::string arch = a ? a->s : "";
    if (arch != "mistral4" && arch != "deepseek2") fail("general.architecture is '" + arch + "', strata-m4 runs mistral4");
    auto num = [&](const std::string& k, double def, bool required = false) -> double {
        const strata::MetaValue* v = m.get(arch + "." + k);
        if (!v) {
            if (required) fail("GGUF key " + arch + "." + k + " missing");
            return def;
        }
        if (v->type == strata::MetaType::ARRAY) return v->items.empty() ? def : v->items[0].num();
        if (v->type == strata::MetaType::BOOL) return v->u ? 1 : 0;
        return v->num();
    };
    hp_.n_layer = (int) num("block_count", 0, true);
    hp_.n_embd = (int) num("embedding_length", 0, true);
    hp_.n_head = (int) num("attention.head_count", 0, true);
    hp_.q_lora = (int) num("attention.q_lora_rank", 0);
    hp_.kv_lora = (int) num("attention.kv_lora_rank", 0, true);
    const int k_mla = (int) num("attention.key_length_mla", 0);
    const int v_mla = (int) num("attention.value_length_mla", 0);
    hp_.rope = (int) num("rope.dimension_count", 64);
    hp_.nope = (k_mla ? k_mla : 128) - hp_.rope;
    hp_.v_dim = v_mla ? v_mla : 128;
    hp_.n_expert = (int) num("expert_count", 0, true);
    hp_.top_k = (int) num("expert_used_count", 0, true);
    hp_.n_ff = (int) num("expert_feed_forward_length", 0, true);
    hp_.n_shexp = (int) num("expert_shared_count", 1);
    hp_.dense_lead = (int) num("leading_dense_block_count", 0);
    hp_.n_ff_dense = (int) num("feed_forward_length", 0);
    hp_.eps = (float) num("attention.layer_norm_rms_epsilon", 1e-6);
    hp_.rope_base = (float) num("rope.freq_base", 10000.0);
    hp_.w_scale = (float) num("expert_weights_scale", 0.0);
    hp_.w_norm = num("expert_weights_norm", 0) != 0;
    const int gf = (int) num("expert_gating_func", 0);
    hp_.gating = gf == 2 ? 2 : 1;   // 1 softmax, 2 sigmoid (0 = unset: softmax, as llama.cpp's deepseek2 loader)
    if (gf != 0 && gf != 1 && gf != 2) fail("expert_gating_func " + std::to_string(gf) + " is not supported");
    hp_.temp_scale = (float) num("attention.temperature_scale", 0.0);
    hp_.temp_floor = (float) num("attention.temperature_length", 0.0);
    if (hp_.temp_scale != 0.f && hp_.temp_floor <= 0.f) hp_.temp_floor = 8192.f;
    if (opt_.attn_temp_scale >= 0) {
        hp_.temp_scale = (float) opt_.attn_temp_scale;
        if (hp_.temp_floor <= 0.f) hp_.temp_floor = 8192.f;
    }
    if (!opt_.attn_temp) hp_.temp_scale = 0.f;
    const strata::MetaValue* toks = m.get("tokenizer.ggml.tokens");
    if (const strata::MetaValue* b = m.get("tokenizer.ggml.bos_token_id")) bos_ = (int) b->num();
    hp_.n_vocab = (int) num("vocab_size", toks ? (double) toks->count : 0.0);
    if (hp_.kv_lora != m4k::kLat || hp_.rope != m4k::kRope || hp_.n_head > m4k::kMaxHeads || hp_.nope + hp_.rope != 128)
        fail("an MLA shape the kernels do not take (kv_lora 256, rope 64, nope 64, <= 32 heads)");
    if (hp_.top_k > 16) fail("more than 16 experts per token");

    // YaRN as llama.cpp's llama_context sets it up (see llama-context.cpp, TAG_DEEPSEEK2_YARN_LOG_MUL_FIX)
    const strata::MetaValue* st = m.get(arch + ".rope.scaling.type");
    const bool yarn = st && st->s == "yarn";
    const float factor = (float) num("rope.scaling.factor", 0.0);
    hp_.freq_scale = (yarn && factor > 0.f) ? 1.f / factor : 1.f;
    hp_.n_ctx_orig = (int) num("rope.scaling.original_context_length", num("context_length", 8192));
    hp_.beta_fast = (float) num("rope.scaling.yarn_beta_fast", 32.0);
    hp_.beta_slow = (float) num("rope.scaling.yarn_beta_slow", 1.0);
    const float log_mul = (float) num("rope.scaling.yarn_log_multiplier", 0.0) / 0.1f;   // the converter's 0.1 undone
    const float rope_attn_factor = (float) num("rope.scaling.attn_factor", 1.0);
    hp_.ext_factor = yarn ? 1.f : 0.f;
    float attn_factor = 1.f;
    auto get_mscale = [](float scale, float ms) { return scale <= 1.f ? 1.f : (0.1f * ms * logf(scale) + 1.f); };
    if (hp_.ext_factor != 0.f) {
        const float f = 1.f / hp_.freq_scale;
        if (log_mul != 0.f) {
            const float mscale = (arch == "deepseek2" && log_mul != 1.f) ? log_mul : 1.f;
            attn_factor = get_mscale(f, mscale) / get_mscale(f, log_mul);
        } else {
            attn_factor = get_mscale(f, 1.f);
        }
        attn_factor *= 1.f / (1.f + 0.1f * logf(f));
    }
    attn_factor *= rope_attn_factor;
    hp_.attn_factor = attn_factor;
    // deepseek2 graph: kq_scale = mscale^2 / sqrt(head_k)
    const float attn_factor_org = attn_factor * (1.f + 0.1f * logf(1.f / hp_.freq_scale));
    const float mscale = attn_factor_org * (1.f + 0.1f * log_mul * logf(1.f / hp_.freq_scale));
    hp_.kq_scale = mscale * mscale / sqrtf((float) (hp_.nope + hp_.rope));
    logf_("%s: %d layers, %d embd, %d heads, q_lora %d, kv_lora %d, %d experts top-%d (ff %d, shared %d), vocab %d, "
          "yarn x%.0f (orig %d), kq_scale %.5f, attention temperature %.3f / %.0f, router %s%s scale %.3f",
          arch.c_str(), hp_.n_layer, hp_.n_embd, hp_.n_head, hp_.q_lora, hp_.kv_lora, hp_.n_expert, hp_.top_k, hp_.n_ff,
          hp_.n_shexp, hp_.n_vocab, 1.f / hp_.freq_scale, hp_.n_ctx_orig, hp_.kq_scale, hp_.temp_scale, hp_.temp_floor,
          hp_.gating == 2 ? "sigmoid" : "softmax", hp_.w_norm ? " norm" : "", hp_.w_scale);
}

Mat Engine::load_mat(const std::string& name, int n_in, int n_out, bool required) {
    Mat m;
    const strata::TensorInfo* t = gm_->find(name);
    if (!t) {
        if (required) fail("tensor " + name + " missing");
        return m;
    }
    if (t->shape.size() < 2 || (int) t->shape[0] != n_in || (int) t->shape[1] != n_out)
        fail(name + ": shape " + std::to_string(t->shape[0]) + "x" + std::to_string(t->shape.size() > 1 ? t->shape[1] : 0) +
             ", expected " + std::to_string(n_in) + "x" + std::to_string(n_out));
    m.type = (int) t->type;
    m.n_in = n_in;
    m.n_out = n_out;
    const uint8_t* src = tensor_ptr(*gm_, t);
    if (K::native_mmvq_supported(m.type) && n_in % 256 == 0) {
        const size_t b = K::native_mmvq_weight_bytes(m.type, n_in, n_out);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, b + 256));
        CK(cudaMemset(d + b, 0, 256));
        CK(cudaMemcpy(d, src, b, cudaMemcpyHostToDevice));
        m.d = d;
    } else {
        m.f = upload(host_f32(*t, src));
        m.type = 0;
    }
    return m;
}

float* Engine::load_f32(const std::string& name, bool required) {
    const strata::TensorInfo* t = gm_->find(name);
    if (!t) {
        if (required) fail("tensor " + name + " missing");
        return nullptr;
    }
    return upload(host_f32(*t, tensor_ptr(*gm_, t)));
}

void Engine::load_layers() {
    const int E = hp_.n_embd, H = hp_.n_head;
    layers_.resize((size_t) hp_.n_layer);
    size_t dense_bytes = 0;
    for (int l = 0; l < hp_.n_layer; ++l) {
        Layer& L = layers_[(size_t) l];
        auto n = [&](const char* s) { return "blk." + std::to_string(l) + "." + s; };
        L.attn_norm = load_f32(n("attn_norm.weight"));
        L.ffn_norm = load_f32(n("ffn_norm.weight"));
        L.kv_a_norm = load_f32(n("attn_kv_a_norm.weight"));
        if (hp_.q_lora > 0) {
            L.q_a_norm = load_f32(n("attn_q_a_norm.weight"));
            L.wq_a = load_mat(n("attn_q_a.weight"), E, hp_.q_lora);
            L.wq_b = load_mat(n("attn_q_b.weight"), hp_.q_lora, H * 128);
        } else {
            L.wq_b = load_mat(n("attn_q.weight"), E, H * 128);
        }
        L.wkv_a = load_mat(n("attn_kv_a_mqa.weight"), E, hp_.kv_lora + hp_.rope);
        L.wo = load_mat(n("attn_output.weight"), H * hp_.v_dim, E);
        {
            const strata::TensorInfo* kb = gm_->find(n("attn_k_b.weight"));
            const strata::TensorInfo* vb = gm_->find(n("attn_v_b.weight"));
            if (!kb || !vb) fail(n("attn_k_b / attn_v_b") + " missing (a GGUF converted before the MLA split; re-convert)");
            L.wk_b = upload(host_f32(*kb, tensor_ptr(*gm_, kb)));   // {nope, kv_lora, H}: per head [kv_lora][nope]
            L.wv_b = upload(host_f32(*vb, tensor_ptr(*gm_, vb)));   // {kv_lora, v_dim, H}: per head [v_dim][kv_lora]
        }
        dense_bytes += L.wq_a.bytes() + L.wq_b.bytes() + L.wkv_a.bytes() + L.wo.bytes();
        if (l < hp_.dense_lead) {
            L.dense = true;
            L.ffn_gate = load_mat(n("ffn_gate.weight"), E, hp_.n_ff_dense);
            L.ffn_up = load_mat(n("ffn_up.weight"), E, hp_.n_ff_dense);
            L.ffn_down = load_mat(n("ffn_down.weight"), hp_.n_ff_dense, E);
            continue;
        }
        L.router = load_f32(n("ffn_gate_inp.weight"));
        L.probs_b = load_f32(n("exp_probs_b.bias"), false);
        const int fsh = hp_.n_ff * std::max(1, hp_.n_shexp);
        L.sh_gate = load_mat(n("ffn_gate_shexp.weight"), E, fsh);
        L.sh_up = load_mat(n("ffn_up_shexp.weight"), E, fsh);
        L.sh_down = load_mat(n("ffn_down_shexp.weight"), fsh, E);
        dense_bytes += L.sh_gate.bytes() + L.sh_up.bytes() + L.sh_down.bytes();
        // routed experts: pointers into the mapped file
        const strata::TensorInfo* gu = gm_->find(n("ffn_gate_up_exps.weight"));
        const strata::TensorInfo* tg = gm_->find(n("ffn_gate_exps.weight"));
        const strata::TensorInfo* tu = gm_->find(n("ffn_up_exps.weight"));
        const strata::TensorInfo* td = gm_->find(n("ffn_down_exps.weight"));
        if (!td || (!gu && (!tg || !tu))) fail(n("ffn_*_exps") + " missing");
        L.d_type = (int) td->type;
        L.gu_type = (int) (gu ? gu->type : tg->type);
        if (!gu && tg->type != tu->type) fail(n("ffn_gate_exps / ffn_up_exps") + " differ in type");
        std::string err;
        if (!C::native_fmt(L.gu_type, L.d_type, E, hp_.n_ff, L.fmt, err)) fail(n("experts: ") + err);
        if (gu) {   // fused [gate rows | up rows] per expert
            L.src_gate = tensor_ptr(*gm_, gu);
            L.src_up = L.src_gate + (size_t) hp_.n_ff * L.fmt.gu_row;
            L.src_gu_stride = 2 * (size_t) hp_.n_ff * L.fmt.gu_row;
        } else {
            L.src_gate = tensor_ptr(*gm_, tg);
            L.src_up = tensor_ptr(*gm_, tu);
            L.src_gu_stride = (size_t) hp_.n_ff * L.fmt.gu_row;
        }
        L.src_down = tensor_ptr(*gm_, td);
        L.src_d_stride = (size_t) E * L.fmt.d_row;
        for (int p = 0; p < 2; ++p) {   // the split policy's halves
            C::NativeFmt& f = L.half_fmt[p];
            f = L.fmt;
            f.up_off = (size_t) (hp_.n_ff / 2) * L.fmt.gu_row;
            f.down_off = 2 * f.up_off;
            f.bytes = f.down_off + (size_t) (E / 2) * L.fmt.d_row;
        }
        if (K::native_expert_supported(L.gu_type, L.d_type, E, hp_.n_ff)) {
            L.L = K::native_expert_layout(L.gu_type, L.d_type, E, hp_.n_ff);
            const size_t blk = 256;
            L.stride = (L.L.bytes + blk - 1) / blk * blk;
        } else {
            L.stride = 0;   // no GPU kernels for this pair: every expert of the layer on the CPU
            logf_("layer %d: no grouped GPU expert kernel for %s/%s, its experts run on the CPU", l,
                  strata::ggml_type_name((uint32_t) L.gu_type), strata::ggml_type_name((uint32_t) L.d_type));
        }
        L.slot_of.assign((size_t) hp_.n_expert, -1);
        L.cpu.resize((size_t) hp_.n_expert);
        L.file_fmt.resize((size_t) hp_.n_expert);
    }
    out_norm_ = load_f32("output_norm.weight");
    const strata::TensorInfo* te = gm_->find("token_embd.weight");
    if (!te) fail("token_embd.weight missing");
    embd_type_ = (int) te->type;
    if (!K::embed_type_supported(embd_type_)) fail(std::string("token_embd is ") + te->type_name() + ", not an embedding type the kernels take");
    embd_row_ = K::iq_row_bytes(embd_type_, hp_.n_embd) ? K::iq_row_bytes(embd_type_, hp_.n_embd)
                                                         : K::native_mmvq_weight_bytes(embd_type_, hp_.n_embd, 1);
    {
        const size_t b = embd_row_ * (size_t) te->shape[1];
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, b));
        CK(cudaMemcpy(d, tensor_ptr(*gm_, te), b, cudaMemcpyHostToDevice));
        embd_ = d;
        dense_bytes += b;
    }
    const strata::TensorInfo* to = gm_->find("output.weight");
    head_ = to ? load_mat("output.weight", hp_.n_embd, (int) to->shape[1]) : load_mat("token_embd.weight", hp_.n_embd, (int) te->shape[1]);
    if (hp_.n_vocab <= 0) hp_.n_vocab = head_.n_out;
    dense_bytes += head_.bytes();
    logf_("dense weights in VRAM: %.2f GiB (embeddings %s, head %s)", dense_bytes / GiB, te->type_name(),
          strata::ggml_type_name((uint32_t) (head_.f ? 0 : head_.type)));
}

// ------------------------------------------------------------------------------------------------ buffers
void Engine::alloc_buffers() {
    const int E = hp_.n_embd, H = hp_.n_head, k = hp_.top_k;
    Tm_ = std::max(opt_.chunk, kMaxWin);
    const size_t T = (size_t) Tm_;
    auto fa = [&](float** p, size_t n) { CK(cudaMalloc(p, n * 4)); };
    auto ba = [&](void** p, size_t n) { CK(cudaMalloc(p, n)); };
    CK(cudaMalloc(&d_tok_, T * 4));
    CK(cudaMalloc(&d_pos_, T * 4));
    CK(cudaMalloc(&d_ids_, T * k * 4));
    CK(cudaMalloc(&d_argmax_, T * 4));
    CK(cudaMalloc(&d_cpu_row_, T * k * 4));
    CK(cudaMalloc(&d_img_rows_, T * 4));
    fa(&x_, T * E);
    fa(&xn_, T * E);
    fa(&qa_, T * std::max(hp_.q_lora, 1));
    fa(&kvp_, T * m4k::kKD);
    fa(&q_, T * H * 128);
    fa(&Qf_, T * H * m4k::kKD);
    fa(&Olat_, T * H * m4k::kLat);
    fa(&attn_, T * H * hp_.v_dim);
    fa(&ao_, T * E);
    fa(&rlog_, T * hp_.n_expert);
    fa(&w_, T * k);
    const int ffmax = std::max(hp_.n_ff * std::max(1, hp_.n_shexp), hp_.n_ff_dense);
    fa(&g_, T * ffmax);
    fa(&u_, T * ffmax);
    fa(&h_, T * ffmax);
    fa(&sh_, T * E);
    fa(&gpu_parts_, T * k * E);
    fa(&cpu_parts_, T * k * E);
    fa(&img_, T * E);
    fa(&part_, m4k::mla_decode_part_floats(kMaxWin, H));
    fa(&logits_, (size_t) kMaxWin * hp_.n_vocab);
    ba(&xq_, q8_bytes(std::max(E, H * hp_.v_dim), (int) T) + 1024);
    ba(&qaq_, q8_bytes(std::max(hp_.q_lora, 256), (int) T) + 1024);
    ba(&aq_, q8_bytes(H * hp_.v_dim, (int) T) + 1024);
    ba(&hq_, q8_bytes(ffmax, (int) T) + 1024);
    ba(&xq_exp_, T * (E / 32) * 36 + 1024);
    ba(&exp_scratch_, K::native_expert_scratch_bytes((int64_t) T * k, hp_.n_ff) + 4096);
    CK(cudaMalloc(&grp_ptr_, (size_t) (hp_.n_expert + 1) * 8));
    CK(cudaMalloc(&grp_start_, (size_t) (hp_.n_expert + 2) * 4));
    CK(cudaMalloc(&n_groups_, 4));
    CK(cudaMalloc(&ent_dst_, T * k * 4));
    CK(cudaMalloc(&ent_tok_, T * k * 4));
    // prompt attention: f16 K rows for the whole context, Q, and the score budget
    CK(cudaMalloc(&K16_, (size_t) opt_.ctx * m4k::kKD * 2));
    CK(cudaMalloc(&Q16_, T * H * m4k::kKD * 2));
    S_floats_ = std::max<size_t>((size_t) 32 << 20, (size_t) opt_.ctx * H);
    CK(cudaMalloc(&S_, S_floats_ * 4));
    CK(cudaMalloc(&P_, S_floats_ * 2));
    // KV cache
    const size_t kv_row = opt_.kv == m4k::KV_I8 ? m4k::kKD : m4k::kKD * 2;
    for (Layer& L : layers_) {
        CK(cudaMalloc(&L.kv, (size_t) opt_.ctx * kv_row));
        if (opt_.kv == m4k::KV_I8) CK(cudaMalloc(&L.kv_s, (size_t) opt_.ctx * 2 * 4));
    }
    logf_("latent KV cache: %d tokens x %d layers, %s, %.2f GiB", opt_.ctx, hp_.n_layer,
          opt_.kv == m4k::KV_I8 ? "int8" : "f16", (double) opt_.ctx * hp_.n_layer * (kv_row + (opt_.kv == m4k::KV_I8 ? 8 : 0)) / GiB);
    // rope table
    {
        const int npos = opt_.ctx + kMaxWin;
        std::vector<float> tab((size_t) npos * hp_.rope);
        m4k::rope_table_host(tab.data(), npos, hp_.rope, hp_.rope_base, hp_.freq_scale, hp_.ext_factor, hp_.attn_factor,
                             hp_.beta_fast, hp_.beta_slow, hp_.n_ctx_orig);
        rope_ = upload(tab);
    }
    // mapped host memory for the decode graph, pinned host memory for prompts
    CK(cudaHostAlloc((void**) &hs_, sizeof(HostShared), cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset((void*) hs_, 0, sizeof(HostShared));
    CK(cudaHostGetDevicePointer((void**) &hs_dev_, (void*) hs_, 0));
    CK(cudaHostAlloc((void**) &hx_, (size_t) kMaxWin * E * 4, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**) &hw_, (size_t) kMaxWin * k * 4, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**) &hids_, (size_t) kMaxWin * k * 4, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**) &hcpu_, (size_t) kMaxWin * k * E * 4, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void**) &hx_dev_, hx_, 0));
    CK(cudaHostGetDevicePointer((void**) &hw_dev_, hw_, 0));
    CK(cudaHostGetDevicePointer((void**) &hids_dev_, hids_, 0));
    CK(cudaHostGetDevicePointer((void**) &hcpu_dev_, hcpu_, 0));
    CK(cudaHostAlloc((void**) &px_, T * E * 4, 0));
    CK(cudaHostAlloc((void**) &pids_, T * k * 4 * 2, 0));
    CK(cudaHostAlloc((void**) &pcpu_, T * k * E * 4, 0));
    CK(cudaHostAlloc((void**) &pcpu_row_, T * k * 4, 0));
    CK(cudaHostAlloc((void**) &ptok_, T * 4 * 2, 0));
    hlogits_.resize((size_t) kMaxWin * hp_.n_vocab);
    CB(cublasCreate(&cb_));
    CB(cublasSetStream(cb_, s_));
    CK(cudaMalloc(&cb_ws_, (size_t) 32 << 20));
    CB(cublasSetWorkspace(cb_, cb_ws_, (size_t) 32 << 20));
    CB(cublasSetMathMode(cb_, CUBLAS_DEFAULT_MATH));
    // the streamed prompt path: a ring of expert slots, f16 scratch for one expert's (or one dense) weights and a
    // chunk's rows, the routes' order tables
    if (opt_.pf_stream) {
        const int F = hp_.n_ff;
        for (const Layer& L : layers_) {
            if (L.dense) continue;
            ring_stride_ = std::max(ring_stride_, L.stride ? L.stride : (L.fmt.bytes + 255) / 256 * 256);
        }
        CK(cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking));
        CK(cudaMalloc(&ring_, (size_t) opt_.pf_ring * ring_stride_ + 4096));
        ev_copied_.resize((size_t) opt_.pf_ring);
        ev_free_.resize((size_t) opt_.pf_ring);
        for (int i = 0; i < opt_.pf_ring; ++i) {
            CK(cudaEventCreateWithFlags(&ev_copied_[(size_t) i], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_free_[(size_t) i], cudaEventDisableTiming));
        }
        CK(cudaMalloc(&GU_, T * 2 * F * 4));
        CK(cudaMalloc(&H16_, T * F * 2));
        CK(cudaMalloc(&d_rpos_, T * k * 4));
        CK(cudaMalloc(&d_rows_, T * k * 4));
        CK(cudaMalloc(&d_noslot_, (size_t) hp_.n_expert * 4));
        CK(cudaMemset(d_noslot_, 0xff, (size_t) hp_.n_expert * 4));   // -1: every route reads its sorted row
        CK(cudaHostAlloc((void**) &prpos_, T * k * 4, 0));
        CK(cudaHostAlloc((void**) &prows_, T * k * 4, 0));
        logf_("prompt streaming: chunks of %d+ tokens, ring of %d expert slots (%.2f GiB)", opt_.pf_min, opt_.pf_ring,
              (double) opt_.pf_ring * ring_stride_ / GiB);
    }
    if (opt_.pf_stream || opt_.pf_gemm) {
        CK(cudaMalloc(&d_xinv_, T * 4));
        CK(cudaMalloc(&d_hinv_, T * 4));
        ensure_w16((size_t) 2 * hp_.n_ff * E);
        ensure_x16((size_t) T * std::max(E, H * hp_.v_dim));
    }
}

void Engine::ensure_w16(size_t elems) {
    if (elems <= W16_elems_) return;
    if (W16_) CK(cudaFree(W16_));
    CK(cudaMalloc(&W16_, elems * 2));
    W16_elems_ = elems;
}

void Engine::ensure_x16(size_t elems) {
    if (elems <= X16_elems_) return;
    if (X16_) CK(cudaFree(X16_));
    CK(cudaMalloc(&X16_, elems * 2));
    X16_elems_ = elems;
}

/// The host arenas pinned for DMA (cudaHostRegister): the copy engine then reads the experts straight from both nodes'
/// memory.  A refusal is not fatal: the copies then go through the driver's staging (slower).
void Engine::register_arenas() {
    if (!opt_.pf_stream || arenas_.empty()) return;
    const auto t0 = Clock::now();
    double gib = 0;
    bool ok = true;
    for (const auto& a : arenas_) {
        const cudaError_t e = cudaHostRegister(a.first, a.second, cudaHostRegisterPortable);
        if (e != cudaSuccess) {
            (void) cudaGetLastError();
            logf_("prompt streaming: pinning %.2f GiB of expert arena failed (%s); copies go through staging",
                  a.second / GiB, cudaGetErrorString(e));
            ok = false;
            continue;
        }
        gib += a.second / GiB;
    }
    pinned_arenas_ = ok;
    logf_("prompt streaming: %.2f GiB of expert arenas pinned for DMA in %.1f s", gib, ms_since(t0) / 1000.0);
}

// ------------------------------------------------------------------------------------------------ expert placement
void Engine::place_experts() {
    const auto t0 = Clock::now();
    const int NE = hp_.n_expert, E = hp_.n_embd, F = hp_.n_ff;
    // heat: counts from earlier runs, or flat
    heat_.assign((size_t) hp_.n_layer, std::vector<uint64_t>((size_t) NE, 0));
    {
        std::ifstream f(opt_.heat, std::ios::binary);
        uint32_t nl = 0, ne = 0;
        if (f && f.read((char*) &nl, 4) && f.read((char*) &ne, 4) && (int) nl == hp_.n_layer && (int) ne == NE) {
            for (auto& row : heat_) f.read((char*) row.data(), (std::streamsize) (row.size() * 8));
            logf_("expert heat from %s", opt_.heat.c_str());
        } else {
            logf_("no expert heat file (%s): placement starts flat; it is learned while serving and saved at exit",
                  opt_.heat.c_str());
        }
    }
    struct Cand { int l, e; double score; };
    std::vector<Cand> c;
    for (int l = 0; l < hp_.n_layer; ++l) {
        if (layers_[(size_t) l].dense) continue;
        uint64_t tot = 0;
        for (uint64_t v : heat_[(size_t) l]) tot += v;
        for (int e = 0; e < NE; ++e) {
            // the share of the layer's routes this expert took (+ a flat prior), a tiny layer-spread tie breaker
            const double share = ((double) heat_[(size_t) l][(size_t) e] + 1.0) / ((double) tot + NE);
            c.push_back({l, e, share - 1e-12 * (double) ((e * 7 + l * 13) % 97)});
        }
    }
    std::stable_sort(c.begin(), c.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });

    // 1. VRAM slots
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const size_t reserve = (size_t) (opt_.vram_reserve * GiB);
    size_t budget = fr > reserve ? fr - reserve : 0;
    if (opt_.vram_experts >= 0) budget = std::min(budget, (size_t) (opt_.vram_experts * GiB));
    std::vector<int> n_slots((size_t) hp_.n_layer, 0);
    std::vector<char> in_vram(c.size(), 0);
    size_t used = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        const Layer& L = layers_[(size_t) c[i].l];
        if (L.stride == 0) continue;
        if (used + L.stride > budget) continue;
        used += L.stride;
        in_vram[i] = 1;
        ++n_slots[(size_t) c[i].l];
    }
    for (int l = 0; l < hp_.n_layer; ++l) {
        Layer& L = layers_[(size_t) l];
        L.n_slots = 0;
        if (n_slots[(size_t) l] > 0) CK(cudaMalloc(&L.slots, (size_t) n_slots[(size_t) l] * L.stride + 4096));
        CK(cudaMalloc(&L.d_slot_of, (size_t) NE * 4));
    }
    std::vector<uint8_t> blob;
    size_t n_vram = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        if (!in_vram[i]) continue;
        Layer& L = layers_[(size_t) c[i].l];
        const int e = c[i].e, s = L.n_slots++;
        blob.resize(L.L.bytes);
        std::memcpy(blob.data(), L.src_gate + (size_t) e * L.src_gu_stride, (size_t) F * L.fmt.gu_row);
        std::memcpy(blob.data() + L.L.up_off, L.src_up + (size_t) e * L.src_gu_stride, (size_t) F * L.fmt.gu_row);
        std::memcpy(blob.data() + L.L.down_off, L.src_down + (size_t) e * L.src_d_stride, (size_t) E * L.fmt.d_row);
        CK(cudaMemcpy(L.slots + (size_t) s * L.stride, blob.data(), L.L.bytes, cudaMemcpyHostToDevice));
        L.slot_of[(size_t) e] = s;
        ++n_vram;
    }
    for (Layer& L : layers_) CK(cudaMemcpy(L.d_slot_of, L.slot_of.data(), (size_t) NE * 4, cudaMemcpyHostToDevice));
    if (n_vram == 0) logf_("no expert fits in VRAM (%.2f GiB free after the reserve)", budget / GiB);

    // 2. host tiers
    const bool numa = opt_.numa != "off" && topo_.nodes.size() > 1 && gpu_node_ >= 0;
    const bool split = numa && opt_.numa == "split";
    auto node_free = [&](int node) -> double {
        const NumaNode* n = topo_.node(node);
        return n ? (double) n->mem_free : 0.0;
    };
    double cap_gpu = opt_.ram_gpu_node >= 0 ? opt_.ram_gpu_node * GiB : std::max(0.0, node_free(gpu_node_) - 6 * GiB);
    double cap_oth = !numa ? 0.0
                     : opt_.ram_other_node >= 0 ? opt_.ram_other_node * GiB
                                                : std::max(0.0, node_free(other_node_) - 4 * GiB - (double) ngram_bytes_);
    if (!numa) cap_gpu = opt_.ram_gpu_node >= 0 ? opt_.ram_gpu_node * GiB : std::max(0.0, node_free(topo_.nodes[0].id) - 6 * GiB);
    struct Pick { size_t i; int tier; };   // tier 0 GPU node, 1 other node, 2 file
    std::vector<Pick> picks;
    double need_gpu = 0, need_oth = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        if (in_vram[i]) continue;
        const Layer& L = layers_[(size_t) c[i].l];
        const double b = (double) L.fmt.bytes;
        if (split) {
            if (need_gpu + b / 2 <= cap_gpu && need_oth + b / 2 <= cap_oth) {
                need_gpu += b / 2;
                need_oth += b / 2;
                picks.push_back({i, 0});
            } else {
                picks.push_back({i, 2});
            }
        } else if (need_gpu + b <= cap_gpu) {
            need_gpu += b;
            picks.push_back({i, 0});
        } else if (numa && need_oth + b <= cap_oth) {
            need_oth += b;
            picks.push_back({i, 1});
        } else {
            picks.push_back({i, 2});
        }
    }
    // arenas
    const int node_g = numa ? gpu_node_ : -1, node_o = numa ? other_node_ : -1;
    std::string err;
    uint8_t* ar_g = need_gpu > 0 ? (uint8_t*) node_alloc((size_t) need_gpu + 4096, node_g, topo_, err) : nullptr;
    if (need_gpu > 0 && !ar_g) fail("expert arena on node " + std::to_string(node_g) + ": " + err);
    uint8_t* ar_o = need_oth > 0 ? (uint8_t*) node_alloc((size_t) need_oth + 4096, node_o, topo_, err) : nullptr;
    if (need_oth > 0 && !ar_o) fail("expert arena on node " + std::to_string(node_o) + ": " + err);
    if (ar_g) arenas_.push_back({ar_g, (size_t) need_gpu + 4096});
    if (ar_o) arenas_.push_back({ar_o, (size_t) need_oth + 4096});
    // fill: each node's arena copied by threads on that node (the first touch already placed the pages)
    struct Job { uint8_t* dst; const uint8_t* src; size_t n; };
    std::vector<Job> jobs_g, jobs_o;
    size_t off_g = 0, off_o = 0;
    int n_g = 0, n_o = 0, n_f = 0;
    for (const Pick& p : picks) {
        Layer& L = layers_[(size_t) c[p.i].l];
        const int e = c[p.i].e;
        CpuExpert& ce = L.cpu[(size_t) e];
        const uint8_t* sg = L.src_gate + (size_t) e * L.src_gu_stride;
        const uint8_t* su = L.src_up + (size_t) e * L.src_gu_stride;
        const uint8_t* sd = L.src_down + (size_t) e * L.src_d_stride;
        const size_t gub = (size_t) F * L.fmt.gu_row, db = (size_t) E * L.fmt.d_row;
        if (p.tier == 2) {
            // file tier: read in place from the mapping (gate rows, up and down reached by offsets)
            auto f = std::make_unique<C::NativeFmt>(L.fmt);
            f->up_off = (size_t) (su - sg);
            f->down_off = (size_t) (sd - sg);
            ce.n_parts = 1;
            ce.node[0] = numa ? other_node_ : (topo_.nodes.empty() ? 0 : topo_.nodes[0].id);
            ce.part[0] = {sg, f.get(), 0, F, 0, E};
            L.file_fmt[(size_t) e] = std::move(f);
            ++n_f;
            continue;
        }
        if (split) {
            const int fh = F / 2, eh = E / 2;
            for (int h = 0; h < 2; ++h) {
                uint8_t*& base = h == 0 ? ar_g : ar_o;
                size_t& off = h == 0 ? off_g : off_o;
                auto& jobs = h == 0 ? jobs_g : jobs_o;
                uint8_t* dst = base + off;
                const C::NativeFmt& hf = L.half_fmt[h];
                jobs.push_back({dst, sg + (size_t) h * fh * L.fmt.gu_row, (size_t) fh * L.fmt.gu_row});
                jobs.push_back({dst + hf.up_off, su + (size_t) h * fh * L.fmt.gu_row, (size_t) fh * L.fmt.gu_row});
                jobs.push_back({dst + hf.down_off, sd + (size_t) h * eh * L.fmt.d_row, (size_t) eh * L.fmt.d_row});
                off += hf.bytes;
                ce.part[h] = {dst, &L.half_fmt[h], h * fh, fh, h * eh, eh};
                ce.node[h] = h == 0 ? gpu_node_ : other_node_;
            }
            ce.n_parts = 2;
            ++n_g;
            continue;
        }
        uint8_t* base = p.tier == 0 ? ar_g : ar_o;
        size_t& off = p.tier == 0 ? off_g : off_o;
        auto& jobs = p.tier == 0 ? jobs_g : jobs_o;
        uint8_t* dst = base + off;
        jobs.push_back({dst, sg, gub});
        jobs.push_back({dst + L.fmt.up_off, su, gub});
        jobs.push_back({dst + L.fmt.down_off, sd, db});
        off += L.fmt.bytes;
        ce.n_parts = 1;
        ce.node[0] = p.tier == 0 ? (numa ? gpu_node_ : topo_.nodes[0].id) : other_node_;
        ce.part[0] = {dst, &L.fmt, 0, F, 0, E};
        (p.tier == 0 ? n_g : n_o) += 1;
    }
    auto run_jobs = [&](std::vector<Job>& jobs, int node) {
        if (jobs.empty()) return;
        std::atomic<size_t> next{0};
        std::vector<std::thread> th;
        const int nt = 8;
        for (int i = 0; i < nt; ++i)
            th.emplace_back([&] {
                if (node >= 0) pin_to_node(topo_, node);
                for (size_t j; (j = next.fetch_add(1)) < jobs.size();) std::memcpy(jobs[j].dst, jobs[j].src, jobs[j].n);
            });
        for (auto& t : th) t.join();
    };
    std::thread tg([&] { run_jobs(jobs_g, node_g); });
    std::thread to([&] { run_jobs(jobs_o, node_o); });
    tg.join();
    to.join();
    char buf[512];
    if (split)
        std::snprintf(buf, sizeof buf,
                      "experts: %zu in VRAM (%.2f GiB), %d split over both nodes (%.2f GiB on node %d + %.2f GiB on node %d), "
                      "%d in the file tier; placed in %.1f s",
                      n_vram, used / GiB, n_g, need_gpu / GiB, gpu_node_, need_oth / GiB, other_node_, n_f, ms_since(t0) / 1000.0);
    else
        std::snprintf(buf, sizeof buf,
                      "experts: %zu in VRAM (%.2f GiB), %d on node %d (%.2f GiB), %d on node %d (%.2f GiB), %d in the file "
                      "tier; placed in %.1f s",
                      n_vram, used / GiB, n_g, numa ? gpu_node_ : (topo_.nodes.empty() ? 0 : topo_.nodes[0].id), need_gpu / GiB,
                      n_o, other_node_, need_oth / GiB, n_f, ms_since(t0) / 1000.0);
    placement_ = buf;
    logf_("%s", buf);
    register_arenas();
}

void Engine::load() {
    const auto t0 = Clock::now();
    C::native_fmt_allow_large(true);
    // mapped host memory needs cudaDeviceMapHost; with unified addressing (every 64-bit Linux system) it is on anyway,
    // and a context made earlier refuses the flag: try it, never fail on it
    cudaSetDeviceFlags(cudaDeviceMapHost);
    (void) cudaGetLastError();
    CK(cudaSetDevice(opt_.gpu));
    CK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, opt_.gpu));
    char bus[32] = {};
    CK(cudaDeviceGetPCIBusId(bus, sizeof bus, opt_.gpu));
    topo_ = Topology::detect();
    gpu_node_ = pci_numa_node(bus);
    for (const auto& n : topo_.nodes)
        if (n.id != gpu_node_ && other_node_ < 0) other_node_ = n.id;
    if (gpu_node_ < 0 && !topo_.nodes.empty()) gpu_node_ = topo_.nodes[0].id;
    logf_("%s (sm_%d%d, %.1f GiB) on PCI %s, NUMA node %d; %s", prop.name, prop.major, prop.minor,
          prop.totalGlobalMem / GiB, bus, gpu_node_, topo_.describe().c_str());
    // the host thread: one core of the GPU node
    int host_cpu = opt_.host_core;
    if (host_cpu < 0) {
        const NumaNode* n = topo_.node(gpu_node_);
        host_cpu = n && !n->cores.empty() ? n->cores[0] : 0;
    }
    pin_thread({host_cpu});
    host_cpu_ = host_cpu;
    {   // its SMT sibling: the eager windows' layer server (run_window)
        const std::string sib = [&] {
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(host_cpu) + "/topology/thread_siblings_list");
            std::string l;
            std::getline(f, l);
            return l;
        }();
        for (size_t a = 0; a < sib.size();) {
            size_t b = sib.find_first_of(",-", a);
            const int c = std::atoi(sib.substr(a, b == std::string::npos ? std::string::npos : b - a).c_str());
            if (c != host_cpu) { host_sib_ = c; break; }
            if (b == std::string::npos) break;
            a = b + 1;
        }
    }

    gm_ = std::make_shared<strata::GgufModel>(strata::GgufModel::open(opt_.model));
    read_hparams();
    load_layers();
    alloc_buffers();
    if (!opt_.eagle.empty() && opt_.mode != "dump" && opt_.mode != "ppl") load_eagle();   // before the experts take VRAM
    // the n-gram table on the other node
    if (opt_.ngram_mib > 0) {
        ngram_bytes_ = (size_t) opt_.ngram_mib << 20;
        std::string err;
        const int nd = (topo_.nodes.size() > 1 && opt_.numa != "off") ? other_node_ : -1;
        ngram_mem_ = node_alloc(ngram_bytes_, nd, topo_, err);
        if (!ngram_mem_) logf_("n-gram table: %s (lookup drafts only)", err.c_str());
        ngram_.init(ngram_mem_, ngram_mem_ ? ngram_bytes_ : 0);
        if (ngram_mem_) logf_("n-gram table: %zu slots (%d MiB) on node %d", ngram_.slots(), opt_.ngram_mib, nd);
    }
    place_experts();
    NumaPool::Options po;
    po.workers_per_node = opt_.workers;
    po.host_cpu = host_cpu;
    if (opt_.numa == "off" || topo_.nodes.size() < 2) po.nodes = {gpu_node_};
    pool_ = std::make_unique<NumaPool>(topo_, po, hp_.n_embd, hp_.n_ff, Tm_, Tm_ * hp_.top_k);
    logf_("CPU expert pool: %s; host thread on CPU %d", pool_->describe().c_str(), host_cpu);
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    logf_("loaded in %.1f s, %.2f GiB VRAM free", ms_since(t0) / 1000.0, fr / GiB);
}

json Engine::info() const {
    return json{{"model", opt_.model},       {"ctx", opt_.ctx},          {"vocab", hp_.n_vocab},
                {"kv", opt_.kv == m4k::KV_I8 ? "int8" : "f16"},   {"spec", opt_.spec},
                {"placement", placement_},    {"numa", opt_.numa},       {"workers", pool_ ? pool_->workers() : 0},
                {"eagle", eagle_on_},         {"eagle_k", opt_.eagle_k}, {"eagle_p", opt_.eagle_p}};
}

void Engine::save_heat() const {
    if (heat_.empty()) return;
    std::ofstream f(opt_.heat + ".tmp", std::ios::binary);
    const uint32_t nl = (uint32_t) hp_.n_layer, ne = (uint32_t) hp_.n_expert;
    f.write((const char*) &nl, 4);
    f.write((const char*) &ne, 4);
    for (const auto& row : heat_) f.write((const char*) row.data(), (std::streamsize) (row.size() * 8));
    f.close();
    std::rename((opt_.heat + ".tmp").c_str(), opt_.heat.c_str());
}

void Engine::count_heat(int l, const int32_t* ids, int n) {
    auto& row = heat_[(size_t) l];
    const Layer& L = layers_[(size_t) l];
    for (int i = 0; i < n; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= hp_.n_expert) continue;
        ++row[(size_t) e];
        ++routes_total_;
        if (L.slot_of[(size_t) e] >= 0) ++routes_vram_;
        else if (L.cpu[(size_t) e].node[0] == gpu_node_) ++routes_gpu_node_;
        else ++routes_other_;
    }
}

// ------------------------------------------------------------------------------------------------ debug dumps
void Engine::dbg(const char* name, int l, const float* d, int rows, int cols) {
    if (!dbg_on_ || (l >= 0 && std::find(dbg_layers_.begin(), dbg_layers_.end(), l) == dbg_layers_.end())) return;
    CK(cudaStreamSynchronize(s_));
    std::vector<float> h((size_t) rows * cols);
    CK(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost));
    double sum = 0, asum = 0;
    for (float v : h) { sum += v; asum += std::fabs(v); }
    auto row = [&](int r) {
        const float* p = h.data() + (size_t) r * cols;
        std::fprintf(stderr, "    [%d]: %.4f %.4f %.4f ... %.4f %.4f %.4f\n", r, p[0], p[1], p[2], p[cols - 3], p[cols - 2], p[cols - 1]);
    };
    std::fprintf(stderr, "DUMP %s-%d = {%d, %d} sum = %.6f, abs = %.6f\n", name, l, cols, rows, sum, asum);
    row(0);
    if (rows > 1) row(rows - 1);
}

void Engine::dbg_i(const char* name, int l, const int32_t* d, int rows, int cols) {
    if (!dbg_on_ || (l >= 0 && std::find(dbg_layers_.begin(), dbg_layers_.end(), l) == dbg_layers_.end())) return;
    CK(cudaStreamSynchronize(s_));
    std::vector<int32_t> h((size_t) rows * cols);
    CK(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost));
    std::fprintf(stderr, "DUMP %s-%d = {%d, %d}:", name, l, cols, rows);
    for (int r : {0, rows - 1}) {
        std::fprintf(stderr, " [%d]:", r);
        for (int c = 0; c < cols; ++c) std::fprintf(stderr, " %d", h[(size_t) r * cols + c]);
        if (rows == 1) break;
    }
    std::fprintf(stderr, "\n");
}

// ------------------------------------------------------------------------------------------------ forward pieces
void Engine::mm(const Mat& m, const float* x, const void* xq, float* y, int T, cudaStream_t s) {
    if (m.f) {
        const float one = 1.f, zero = 0.f;
        CB(cublasSgemm(cb_, CUBLAS_OP_T, CUBLAS_OP_N, m.n_out, T, m.n_in, &one, m.f, m.n_in, x, m.n_in, &zero, y, m.n_out));
        return;
    }
    if (opt_.pf_gemm && d_xinv_ && T >= 64 && T <= Tm_ && K::dequant_bf16_supported(m.type)) {
        // a prompt chunk: the weights dequantized to f16 once, one tensor-core GEMM (the q8_1 path would read the
        // weights T/8 times)
        const float one = 1.f, zero = 0.f;
        ensure_w16((size_t) m.n_in * m.n_out);
        ensure_x16((size_t) T * m.n_in);
        K::dequant_f16(m.type, m.d, 0, m.n_out, m.n_in, (uint16_t*) W16_, s);
        m4k::rows_f16_scaled(x, nullptr, T, m.n_in, X16_, d_xinv_, s);   // rows past f16's range scaled down
        CB(cublasGemmEx(cb_, CUBLAS_OP_T, CUBLAS_OP_N, m.n_out, T, m.n_in, &one, W16_, CUDA_R_16F, m.n_in, X16_, CUDA_R_16F,
                        m.n_in, &zero, y, CUDA_R_32F, m.n_out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        m4k::scale_rows(y, T, m.n_out, d_xinv_, s);                    // ... and back
        return;
    }
    const size_t col = q8_bytes(m.n_in, 1);
    for (int c0 = 0; c0 < T; c0 += 8) {
        const int nc = std::min(8, T - c0);
        K::native_mmvq(m.type, m.d, (const uint8_t*) xq + (size_t) c0 * col, y + (size_t) c0 * m.n_out, m.n_in, m.n_out,
                       nc, s);
    }
}

__global__ void swiglu_kernel(const float* g, const float* u, float* h, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = g[i] / (1.f + __expf(-g[i])) * u[i];
}

void Engine::embed(const int32_t* d_tokens, int T) {
    K::iq_embed_rows(embd_type_, embd_, embd_row_, d_tokens, T, hp_.n_embd, x_, s_);
}

void Engine::attention(Layer& L, int T, bool window, int pos0, const int32_t* pos, float temp_scale, float temp_floor) {
    const int E = hp_.n_embd, H = hp_.n_head;
    const bool main_layer = &L >= layers_.data() && &L < layers_.data() + layers_.size();
    const int li = main_layer ? (int) (&L - layers_.data()) : -1000;   // EAGLE layers: no dumps
    const int32_t* P = pos ? pos : d_pos_;
    const float ts = temp_scale < 0.f ? hp_.temp_scale : temp_scale, tf = temp_scale < 0.f ? hp_.temp_floor : temp_floor;
    if (li == 0) dbg("inp_embd", -1, x_, T, E);
    m4k::rmsnorm(x_, L.attn_norm, xn_, E, T, E, E, hp_.eps, s_);
    dbg("attn_norm", li, xn_, T, E);
    q8(xn_, xq_, E, T, s_);
    if (hp_.q_lora > 0) {
        mm(L.wq_a, xn_, xq_, qa_, T, s_);
        dbg("q(wq_a)", li, qa_, T, hp_.q_lora);
        m4k::rmsnorm(qa_, L.q_a_norm, qa_, hp_.q_lora, T, hp_.q_lora, hp_.q_lora, hp_.eps, s_);
        dbg("q(q_a_norm)", li, qa_, T, hp_.q_lora);
        q8(qa_, qaq_, hp_.q_lora, T, s_);
        mm(L.wq_b, qa_, qaq_, q_, T, s_);
        dbg("q(wq_b)", li, q_, T, H * 128);
    } else {
        mm(L.wq_b, xn_, xq_, q_, T, s_);
    }
    mm(L.wkv_a, xn_, xq_, kvp_, T, s_);
    dbg("kv_cmpr_pe", li, kvp_, T, m4k::kKD);
    m4k::mla_kv_write(kvp_, L.kv_a_norm, rope_, P, T, opt_.kv, L.kv, L.kv_s, hp_.eps, s_);
    if (dbg_on_ && opt_.kv == m4k::KV_F16) {   // the cache rows this chunk wrote, as f32: [kv_cmpr (normed) | k_pe (roped)]
        m4k::mla_cache_to_f16(L.kv, L.kv_s, opt_.kv, pos0 + T, K16_, s_);
        std::vector<__half> hh((size_t) T * m4k::kKD);
        CK(cudaStreamSynchronize(s_));
        CK(cudaMemcpy(hh.data(), K16_ + (size_t) pos0 * m4k::kKD, hh.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<float> f(hh.size());
        for (size_t i = 0; i < f.size(); ++i) f[i] = __half2float(hh[i]);
        float* tmp = nullptr;
        CK(cudaMalloc(&tmp, f.size() * 4));
        CK(cudaMemcpy(tmp, f.data(), f.size() * 4, cudaMemcpyHostToDevice));
        dbg("Kcur[kv_cmpr|k_pe]", li, tmp, T, m4k::kKD);
        CK(cudaFree(tmp));
    }
    const float one = 1.f, zero = 0.f;
    // q_nope absorbed through wk_b: per head [kv_lora x nope] . q_nope -> Qf[t][h][0:256]
    CB(cublasSgemmStridedBatched(cb_, CUBLAS_OP_T, CUBLAS_OP_N, hp_.kv_lora, T, hp_.nope, &one, L.wk_b, hp_.nope,
                                 (long long) hp_.kv_lora * hp_.nope, q_, H * 128, 128, &zero, Qf_, H * m4k::kKD, m4k::kKD, H));
    m4k::mla_q_rope(q_, Qf_, rope_, P, T, H, ts, tf, s_);
    dbg("Qcur[absorbed|q_pe]", li, Qf_, T * H, m4k::kKD);
    if (window && !opt_.attn_gemm) {
        m4k::mla_attn_decode(Qf_, L.kv, L.kv_s, opt_.kv, P, T, H, hp_.kq_scale, part_, Olat_, s_, opt_.attn_impl);
    } else {
        // prompt chunk at positions pos0 .. pos0+T-1: K rows 0 .. pos0+T-1 as f16, S = K^T Q, causal softmax, O = V P
        const int n_kv = pos0 + T;
        m4k::mla_cache_to_f16(L.kv, L.kv_s, opt_.kv, n_kv, K16_, s_);
        m4k::f32_to_f16_scaled(Qf_, Q16_, (size_t) T * H * m4k::kKD, hp_.kq_scale, s_);
        const int per = std::max(1, (int) std::min<size_t>((size_t) T, S_floats_ / ((size_t) n_kv * H)));
        for (int a = 0; a < T; a += per) {
            const int b = std::min(T, a + per);
            const int cols = (b - a) * H, m = pos0 + b;   // keys beyond the sub-chunk's last token are all masked
            CB(cublasGemmEx(cb_, CUBLAS_OP_T, CUBLAS_OP_N, m, cols, m4k::kKD, &one, K16_, CUDA_R_16F, m4k::kKD,
                            Q16_ + (size_t) a * H * m4k::kKD, CUDA_R_16F, m4k::kKD, &zero, S_, CUDA_R_32F, m,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            m4k::mla_softmax_causal(S_, P_, m, cols, H, pos0 + a, s_);
            CB(cublasGemmEx(cb_, CUBLAS_OP_N, CUBLAS_OP_N, m4k::kLat, cols, m, &one, K16_, CUDA_R_16F, m4k::kKD, P_,
                            CUDA_R_16F, m, &zero, Olat_ + (size_t) a * H * m4k::kLat, CUDA_R_32F, m4k::kLat,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
    // per head: wv_b [v_dim x kv_lora] . O -> attn[t][h*v_dim ..]
    CB(cublasSgemmStridedBatched(cb_, CUBLAS_OP_T, CUBLAS_OP_N, hp_.v_dim, T, hp_.kv_lora, &one, L.wv_b, hp_.kv_lora,
                                 (long long) hp_.v_dim * hp_.kv_lora, Olat_, H * m4k::kLat, m4k::kLat, &zero, attn_,
                                 H * hp_.v_dim, hp_.v_dim, H));
    dbg("kqv_latent", li, Olat_, T * H, m4k::kLat);
    dbg("kqv_out(wv_b)", li, attn_, T, H * hp_.v_dim);
    q8(attn_, aq_, H * hp_.v_dim, T, s_);
    mm(L.wo, attn_, aq_, ao_, T, s_);
    dbg("attn_out(wo)", li, ao_, T, E);
    m4k::add_inplace(x_, ao_, (size_t) T * E, s_);
    dbg("ffn_inp", li, x_, T, E);
}

void Engine::ffn_dense(Layer& L, int T) {
    const int E = hp_.n_embd, F = hp_.n_ff_dense;
    m4k::rmsnorm(x_, L.ffn_norm, xn_, E, T, E, E, hp_.eps, s_);
    q8(xn_, xq_, E, T, s_);
    mm(L.ffn_gate, xn_, xq_, g_, T, s_);
    mm(L.ffn_up, xn_, xq_, u_, T, s_);
    const size_t n = (size_t) T * F;
    swiglu_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s_>>>(g_, u_, h_, n);
    q8(h_, hq_, F, T, s_);
    mm(L.ffn_down, h_, hq_, ao_, T, s_);
    m4k::add_inplace(x_, ao_, (size_t) T * E, s_);
}

void Engine::moe_window(int l, int T) {
    Layer& L = layers_[(size_t) l];
    const int E = hp_.n_embd, k = hp_.top_k, R = T * k, Fs = hp_.n_ff * std::max(1, hp_.n_shexp);
    const float one = 1.f, zero = 0.f;
    m4k::rmsnorm(x_, L.ffn_norm, xn_, E, T, E, E, hp_.eps, s_);
    CB(cublasSgemm(cb_, CUBLAS_OP_T, CUBLAS_OP_N, hp_.n_expert, T, E, &one, L.router, E, xn_, E, &zero, rlog_, hp_.n_expert));
    m4k::router_topk(rlog_, L.probs_b, hp_.n_expert, k, T, hp_.gating, hp_.w_norm, hp_.w_scale, d_ids_, w_, s_);
    dbg("ffn_norm", l, xn_, T, E);
    dbg("ffn_moe_logits", l, rlog_, T, hp_.n_expert);
    dbg_i("ffn_moe_topk", l, d_ids_, T, k);
    dbg("ffn_moe_weights", l, w_, T, k);
    m4k::publish(xn_, T, E, d_ids_, w_, R, hx_dev_, hids_dev_, hw_dev_, &hs_dev_->pub[l], &hs_dev_->seq, s_);
    // shared expert
    q8(xn_, xq_, E, T, s_);
    mm(L.sh_gate, xn_, xq_, g_, T, s_);
    mm(L.sh_up, xn_, xq_, u_, T, s_);
    const size_t n = (size_t) T * Fs;
    swiglu_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s_>>>(g_, u_, h_, n);
    q8(h_, hq_, Fs, T, s_);
    mm(L.sh_down, h_, hq_, sh_, T, s_);
    // the experts in VRAM
    if (L.n_slots > 0) {
        K::quantize_q8_1_rows(xn_, T, E, xq_exp_, s_);
        m4k::group_hits(d_ids_, R, k, L.d_slot_of, hp_.n_expert, L.slots, L.stride, grp_ptr_, grp_start_, n_groups_,
                        ent_dst_, ent_tok_, s_);
        K::native_expert_grouped(L.L, grp_ptr_, grp_start_, n_groups_, ent_dst_, ent_tok_, hp_.n_expert, R, xq_exp_,
                                 exp_scratch_, gpu_parts_, s_);
    }
    dbg("ffn_shexp", l, sh_, T, E);
    // the host's share
    m4k::wait_host(&hs_dev_->done[l], &hs_dev_->seq, &hs_dev_->err, s_);
    if (dbg_on_ && std::find(dbg_layers_.begin(), dbg_layers_.end(), l) != dbg_layers_.end()) {
        CK(cudaStreamSynchronize(s_));
        int ids[16];
        CK(cudaMemcpy(ids, d_ids_, (size_t) k * 4, cudaMemcpyDeviceToHost));
        for (int j = 0; j < k; ++j) {
            const bool g = L.slot_of[(size_t) ids[j]] >= 0;
            char nm[64];
            std::snprintf(nm, sizeof nm, "expert_out[t0,%d,e%d,%s]", j, ids[j], g ? "gpu" : "cpu");
            dbg(nm, l, g ? gpu_parts_ + (size_t) j * E : hcpu_dev_ + (size_t) j * E, 1, E);
        }
    }
    m4k::moe_combine(x_, gpu_parts_, hcpu_dev_, nullptr, sh_, d_ids_, w_, L.d_slot_of, T, k, E, s_);
    dbg("l_out", l, x_, T, E);
}

void Engine::moe_prefill(int l, int T) {
    Layer& L = layers_[(size_t) l];
    const int E = hp_.n_embd, k = hp_.top_k, R = T * k, Fs = hp_.n_ff * std::max(1, hp_.n_shexp);
    const float one = 1.f, zero = 0.f;
    m4k::rmsnorm(x_, L.ffn_norm, xn_, E, T, E, E, hp_.eps, s_);
    CB(cublasSgemm(cb_, CUBLAS_OP_T, CUBLAS_OP_N, hp_.n_expert, T, E, &one, L.router, E, xn_, E, &zero, rlog_, hp_.n_expert));
    m4k::router_topk(rlog_, L.probs_b, hp_.n_expert, k, T, hp_.gating, hp_.w_norm, hp_.w_scale, d_ids_, w_, s_);
    dbg("ffn_norm", l, xn_, T, E);
    dbg("ffn_moe_logits", l, rlog_, T, hp_.n_expert);
    dbg_i("ffn_moe_topk", l, d_ids_, T, k);
    dbg("ffn_moe_weights", l, w_, T, k);
    CK(cudaMemcpyAsync(pids_, d_ids_, (size_t) R * 4, cudaMemcpyDeviceToHost, s_));
    CK(cudaMemcpyAsync(px_, xn_, (size_t) T * E * 4, cudaMemcpyDeviceToHost, s_));
    CK(cudaStreamSynchronize(s_));
    check_ids(l, pids_, R);
    count_heat(l, pids_, R);
    // the CPU's routes, grouped by expert; compact output rows
    call_.clear();
    call_.x = px_;
    call_.n_tok = T;
    call_.fmt = &L.fmt;
    int n_cpu = 0;
    std::vector<std::vector<int>> by_e((size_t) hp_.n_expert);
    for (int r = 0; r < R; ++r) {
        pcpu_row_[r] = -1;
        const int e = pids_[r];
        if (L.slot_of[(size_t) e] < 0) by_e[(size_t) e].push_back(r);
    }
    for (int e = 0; e < hp_.n_expert; ++e) {
        if (by_e[(size_t) e].empty()) continue;
        CpuCall::Entry en;
        en.ex = &L.cpu[(size_t) e];
        en.off = (int) call_.tok.size();
        for (int r : by_e[(size_t) e]) {
            pcpu_row_[r] = n_cpu;
            call_.tok.push_back(r / k);
            call_.out.push_back(pcpu_ + (size_t) n_cpu * E);
            ++n_cpu;
        }
        en.nt = (int) by_e[(size_t) e].size();
        call_.entries.push_back(en);
    }
    if (n_cpu) pool_->start(call_);
    // meanwhile on the GPU: the shared expert and the experts in VRAM
    q8(xn_, xq_, E, T, s_);
    mm(L.sh_gate, xn_, xq_, g_, T, s_);
    mm(L.sh_up, xn_, xq_, u_, T, s_);
    const size_t n = (size_t) T * Fs;
    swiglu_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s_>>>(g_, u_, h_, n);
    q8(h_, hq_, Fs, T, s_);
    mm(L.sh_down, h_, hq_, sh_, T, s_);
    if (L.n_slots > 0) {
        K::quantize_q8_1_rows(xn_, T, E, xq_exp_, s_);
        m4k::group_hits(d_ids_, R, k, L.d_slot_of, hp_.n_expert, L.slots, L.stride, grp_ptr_, grp_start_, n_groups_,
                        ent_dst_, ent_tok_, s_);
        K::native_expert_grouped(L.L, grp_ptr_, grp_start_, n_groups_, ent_dst_, ent_tok_, hp_.n_expert, R, xq_exp_,
                                 exp_scratch_, gpu_parts_, s_);
    }
    if (n_cpu) {
        pool_->wait();
        CK(cudaMemcpyAsync(cpu_parts_, pcpu_, (size_t) n_cpu * E * 4, cudaMemcpyHostToDevice, s_));
    }
    CK(cudaMemcpyAsync(d_cpu_row_, pcpu_row_, (size_t) R * 4, cudaMemcpyHostToDevice, s_));
    dbg("ffn_shexp", l, sh_, T, E);
    if (dbg_on_ && std::find(dbg_layers_.begin(), dbg_layers_.end(), l) != dbg_layers_.end()) {
        int nv = 0;
        for (int r = 0; r < R; ++r) nv += L.slot_of[(size_t) pids_[r]] >= 0;
        std::fprintf(stderr, "DUMP routes-%d: %d of %d on the GPU, %d on the CPU\n", l, nv, R, R - nv);
        // each route's expert output (unweighted), GPU and CPU, for token 0
        for (int j = 0; j < k; ++j) {
            const bool g = L.slot_of[(size_t) pids_[j]] >= 0;
            char nm[64];
            std::snprintf(nm, sizeof nm, "expert_out[t0,%d,e%d,%s]", j, pids_[j], g ? "gpu" : "cpu");
            dbg(nm, l, g ? gpu_parts_ + (size_t) j * E : cpu_parts_ + (size_t) pcpu_row_[j] * E, 1, E);
        }
    }
    // combine: a CPU route reads its compact row (cpu_row)
    m4k::moe_combine(x_, gpu_parts_, cpu_parts_, d_cpu_row_, sh_, d_ids_, w_, L.d_slot_of, T, k, E, s_);
    dbg("l_out", l, x_, T, E);
}

void Engine::head(int from, int n) {
    const int E = hp_.n_embd;
    m4k::rmsnorm(x_ + (size_t) from * E, out_norm_, xn_, E, n, E, E, hp_.eps, s_);
    q8(xn_, xq_, E, n, s_);
    mm(head_, xn_, xq_, logits_, n, s_);
}

// ------------------------------------------------------------------------------------------------ streamed prompt MoE
// Strata's prompt trick for this machine: a chunk of hundreds or thousands of tokens routes to nearly every expert of
// every layer, which the CPU pool computes at ~200 tok/s.  Here every expert of the chunk runs on the GPU instead:
// the ones in VRAM from their slot, the others DMA'd from the pinned arenas of both nodes into a ring of VRAM slots on
// a copy stream (the split policy's halves reassembled on the way), while the compute stream dequantizes each expert
// to f16 and runs gate/up, SwiGLU and down as tensor-core GEMMs over the expert's rows.  PCIe Gen3 x16 moves the
// ~37 GB of CPU experts in ~3 s, so a 2048-token chunk reads at ~700 tok/s, 4096 at ~1300 (the copies dominate).
/// Router ids before the host indexes anything with them: a NaN in the layer's input makes every logit NaN and the
/// top-k's argmax keeps its 0x7fffffff - a clear message instead of a segfault in the grouping.
void Engine::check_ids(int l, const int32_t* ids, int R) const {
    for (int r = 0; r < R; ++r)
        if (ids[r] < 0 || ids[r] >= hp_.n_expert)
            fail("layer " + std::to_string(l) + ": the router returned expert " + std::to_string(ids[r]) + " for route " +
                 std::to_string(r) + " (NaN or inf in the hidden state)");
}

bool Engine::stream_ok(const Layer& L) const {
    return opt_.pf_stream && ring_ && !L.dense && K::dequant_bf16_supported(L.gu_type) &&
           K::dequant_bf16_supported(L.d_type);
}

void Engine::moe_prefill_stream(int l, int T) {
    Layer& L = layers_[(size_t) l];
    const int E = hp_.n_embd, k = hp_.top_k, R = T * k, F = hp_.n_ff, NE = hp_.n_expert;
    const float one = 1.f, zero = 0.f;
    m4k::rmsnorm(x_, L.ffn_norm, xn_, E, T, E, E, hp_.eps, s_);
    CB(cublasSgemm(cb_, CUBLAS_OP_T, CUBLAS_OP_N, NE, T, E, &one, L.router, E, xn_, E, &zero, rlog_, NE));
    m4k::router_topk(rlog_, L.probs_b, NE, k, T, hp_.gating, hp_.w_norm, hp_.w_scale, d_ids_, w_, s_);
    CK(cudaMemcpyAsync(pids_, d_ids_, (size_t) R * 4, cudaMemcpyDeviceToHost, s_));
    CK(cudaStreamSynchronize(s_));
    check_ids(l, pids_, R);
    count_heat(l, pids_, R);
    // the routes sorted by expert: prows_[i] = the token of sorted route i, prpos_[route] = its sorted row
    std::vector<int> cnt((size_t) NE + 1, 0), off((size_t) NE + 1, 0);
    for (int r = 0; r < R; ++r) ++cnt[(size_t) pids_[r]];
    for (int e = 0; e < NE; ++e) off[(size_t) e + 1] = off[(size_t) e] + cnt[(size_t) e];
    {
        std::vector<int> fill(off.begin(), off.end() - 1);
        for (int r = 0; r < R; ++r) {
            const int i = fill[(size_t) pids_[r]]++;
            prows_[i] = r / k;
            prpos_[r] = i;
        }
    }
    CK(cudaMemcpyAsync(d_rows_, prows_, (size_t) R * 4, cudaMemcpyHostToDevice, s_));
    CK(cudaMemcpyAsync(d_rpos_, prpos_, (size_t) R * 4, cudaMemcpyHostToDevice, s_));
    // the shared expert (dense GEMMs)
    const int Fs = F * std::max(1, hp_.n_shexp);
    q8(xn_, xq_, E, T, s_);
    mm(L.sh_gate, xn_, xq_, g_, T, s_);
    mm(L.sh_up, xn_, xq_, u_, T, s_);
    {
        const size_t n = (size_t) T * Fs;
        swiglu_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s_>>>(g_, u_, h_, n);
    }
    q8(h_, hq_, Fs, T, s_);
    mm(L.sh_down, h_, hq_, sh_, T, s_);
    // where each expert's bytes come from; the order: VRAM experts first (they need no copy), then the streamed ones
    const size_t gu_row = L.fmt.gu_row, d_row = L.fmt.d_row;
    const size_t up_off = L.stride ? L.L.up_off : (size_t) F * gu_row;
    const size_t down_off = L.stride ? L.L.down_off : 2 * (size_t) F * gu_row;
    std::vector<int> resident, streamed;
    for (int e = 0; e < NE; ++e) {
        if (!cnt[(size_t) e]) continue;
        (L.slot_of[(size_t) e] >= 0 ? resident : streamed).push_back(e);
    }
    const int RN = opt_.pf_ring;
    const uint64_t seq0 = ring_seq_;
    int issued = 0;
    auto issue = [&](int upto) {   // the copies of streamed experts [issued, upto)
        for (; issued < upto && issued < (int) streamed.size(); ++issued) {
            const int e = streamed[(size_t) issued];
            const int slot = (int) ((seq0 + (uint64_t) issued) % (uint64_t) RN);
            uint8_t* dst = ring_ + (size_t) slot * ring_stride_;
            CK(cudaStreamWaitEvent(cs_, ev_free_[(size_t) slot], 0));   // the slot's previous expert is computed
            const CpuExpert& ce = L.cpu[(size_t) e];
            for (int p = 0; p < ce.n_parts; ++p) {
                const PartRef& pr = ce.part[p];
                const size_t gb = (size_t) pr.gu_rows * gu_row, db = (size_t) pr.d_rows * d_row;
                CK(cudaMemcpyAsync(dst + (size_t) pr.gu_base * gu_row, pr.blob, gb, cudaMemcpyHostToDevice, cs_));
                CK(cudaMemcpyAsync(dst + up_off + (size_t) pr.gu_base * gu_row, pr.blob + pr.fmt->up_off, gb,
                                   cudaMemcpyHostToDevice, cs_));
                CK(cudaMemcpyAsync(dst + down_off + (size_t) pr.d_base * d_row, pr.blob + pr.fmt->down_off, db,
                                   cudaMemcpyHostToDevice, cs_));
                pf_stream_bytes_ += (double) (2 * gb + db);
            }
            CK(cudaEventRecord(ev_copied_[(size_t) slot], cs_));
        }
    };
    issue(RN);
    // one expert: rows [off[e], off[e]+n) of the sorted routes -> cpu_parts_ rows of the same indices
    auto compute = [&](int e, const uint8_t* blob, size_t uo, size_t dof) {
        const int n = cnt[(size_t) e], o = off[(size_t) e];
        m4k::rows_f16_scaled(xn_, d_rows_ + o, n, E, X16_, d_xinv_, s_);
        K::dequant_f16(L.gu_type, blob, 0, F, E, (uint16_t*) W16_, s_);
        K::dequant_f16(L.gu_type, blob + uo, 0, F, E, (uint16_t*) W16_ + (size_t) F * E, s_);
        // GU[n][2F] = X W_gu^T
        CB(cublasGemmEx(cb_, CUBLAS_OP_T, CUBLAS_OP_N, 2 * F, n, E, &one, W16_, CUDA_R_16F, E, X16_, CUDA_R_16F, E, &zero,
                        GU_, CUDA_R_32F, 2 * F, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        m4k::swiglu_f16_scaled(GU_, d_xinv_, n, F, H16_, d_hinv_, s_);
        K::dequant_f16(L.d_type, blob + dof, 0, E, F, (uint16_t*) W16_, s_);
        // D[n][E] = H W_down^T
        CB(cublasGemmEx(cb_, CUBLAS_OP_T, CUBLAS_OP_N, E, n, F, &one, W16_, CUDA_R_16F, F, H16_, CUDA_R_16F, F, &zero,
                        cpu_parts_ + (size_t) o * E, CUDA_R_32F, E, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        m4k::scale_rows(cpu_parts_ + (size_t) o * E, n, E, d_hinv_, s_);
    };
    for (int e : resident) compute(e, L.slots + (size_t) L.slot_of[(size_t) e] * L.stride, L.L.up_off, L.L.down_off);
    for (int j = 0; j < (int) streamed.size(); ++j) {
        const int slot = (int) ((seq0 + (uint64_t) j) % (uint64_t) RN);
        CK(cudaStreamWaitEvent(s_, ev_copied_[(size_t) slot], 0));
        compute(streamed[(size_t) j], ring_ + (size_t) slot * ring_stride_, up_off, down_off);
        CK(cudaEventRecord(ev_free_[(size_t) slot], s_));
        issue(j + 1 + RN);   // the slot just freed (in stream order) takes the next copy
    }
    ring_seq_ += streamed.size();
    // combine: every route reads its sorted row (slot_of all -1, cpu_row = the sorted position)
    m4k::moe_combine(x_, gpu_parts_, cpu_parts_, d_rpos_, sh_, d_ids_, w_, d_noslot_, T, k, E, s_);
}

// ------------------------------------------------------------------------------------------------ EAGLE draft head
// Mistral's EAGLE head for Mistral Small 4 (vLLM: EagleMistralLarge3ForCausalLM): two DeepSeek-V2 decoder layers (MLA as
// the model's, a dense SwiGLU FFN), fed per position p with fc([embed(token p+1) | h_p]) where h_p is the model's final
// hidden state after its output norm; its own output (after its norm) is both the next step's h and the input of the
// model's head (shared, as are the embeddings).  Positions and rope are the model's; the KV cache is its own.
void Engine::load_eagle() {
    const auto t0 = Clock::now();
    std::string path = opt_.eagle, dir;
    struct stat stt;
    if (stat(path.c_str(), &stt) == 0 && S_ISDIR(stt.st_mode)) {
        dir = path;
        path += "/consolidated.safetensors";
    } else {
        const size_t sl = path.find_last_of('/');
        dir = sl == std::string::npos ? "." : path.substr(0, sl);
    }
    const int E = hp_.n_embd, H = hp_.n_head;
    int n_layers = 2;
    float eps = hp_.eps;
    e_ff_ = 0;
    {
        std::ifstream f(dir + "/params.json");
        if (f) {
            const json pj = json::parse(f);
            auto need = [&](const char* k, int v) {
                if (pj.contains(k) && pj[k].get<int>() != v)
                    fail(std::string("EAGLE params.json: ") + k + " " + std::to_string(pj[k].get<int>()) + ", the model has " +
                         std::to_string(v));
            };
            need("dim", E);
            need("n_heads", H);
            need("q_lora_rank", hp_.q_lora);
            need("kv_lora_rank", hp_.kv_lora);
            need("qk_rope_head_dim", hp_.rope);
            need("qk_nope_head_dim", hp_.nope);
            need("v_head_dim", hp_.v_dim);
            n_layers = pj.value("n_layers", 2);
            e_ff_ = pj.value("hidden_dim", 0);
            eps = pj.value("norm_eps", (double) hp_.eps);
            if (pj.contains("llama_4_scaling") && pj["llama_4_scaling"].is_object()) {
                e_temp_ = pj["llama_4_scaling"].value("beta", 0.0);
                e_floor_ = pj["llama_4_scaling"].value("original_max_position_embeddings", 8192.0);
            }
            if (pj.contains("moe") && !pj["moe"].is_null()) fail("EAGLE head with MoE layers: not supported");
        } else {
            logf_("EAGLE: no params.json next to %s, shapes from the tensors", path.c_str());
        }
    }
    if (!opt_.attn_temp) e_temp_ = 0.f;
    if (std::fabs(eps - hp_.eps) > 1e-9) logf_("EAGLE: norm eps %g differs from the model's %g (the model's is used)", eps, hp_.eps);
    if (hp_.q_lora <= 0) fail("EAGLE: the model has no q_lora (the head needs the MLA q path)");
    Safetensors st;
    st.open(path);
    auto get = [&](const std::string& name, std::vector<int64_t>* shape = nullptr) -> std::vector<float> {
        const StTensor* t = st.find(name);
        if (!t) fail("EAGLE: tensor " + name + " missing in " + path);
        std::vector<float> h = st_f32(*t, name);
        if (shape) *shape = t->shape;
        const std::string base = name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0 ? name.substr(0, name.size() - 7) : name;
        if (const StTensor* sc = st.find(base + ".qscale_weight")) {
            const std::vector<float> v = st_f32(*sc, base + ".qscale_weight");
            if (v.size() != 1) fail("EAGLE: " + base + ".qscale_weight is not per tensor");
            for (float& x : h) x *= v[0];
        } else if (t->dtype == "F8_E4M3") {
            fail("EAGLE: " + name + " is float8 without a qscale_weight");
        }
        return h;
    };
    size_t bytes = 0;
    auto mat = [&](const std::string& name, int n_in, int n_out) -> Mat {
        std::vector<int64_t> sh;
        const std::vector<float> w = get(name, &sh);
        if (sh.size() != 2 || sh[0] != n_out || sh[1] != n_in)
            fail("EAGLE: " + name + " shape, expected [" + std::to_string(n_out) + ", " + std::to_string(n_in) + "]");
        if (name == "eagle_linear.weight" || name == "layers.0.feed_forward.w1.weight") {   // a wrong scale shows here
            double ss = 0;
            for (float v : w) ss += (double) v * v;
            logf_("EAGLE: %s rms %.5f", name.c_str(), std::sqrt(ss / (double) w.size()));
        }
        Mat m;
        m.type = 8;
        m.n_in = n_in;
        m.n_out = n_out;
        m.d = upload_q8_0(w, n_in, n_out);
        bytes += m.bytes();
        return m;
    };
    auto vec = [&](const std::string& name, int n) -> float* {
        const std::vector<float> v = get(name);
        if ((int) v.size() != n) fail("EAGLE: " + name + " has " + std::to_string(v.size()) + " values, expected " + std::to_string(n));
        bytes += v.size() * 4;
        return upload(v);
    };
    if (e_ff_ <= 0) {
        const StTensor* t = st.find("layers.0.feed_forward.w1.weight");
        if (!t || t->shape.size() != 2) fail("EAGLE: layers.0.feed_forward.w1.weight missing");
        e_ff_ = (int) t->shape[0];
    }
    if (e_ff_ % 256) fail("EAGLE: hidden_dim not a multiple of 256");
    efc_ = mat("eagle_linear.weight", 2 * E, E);
    elayers_.resize((size_t) n_layers);
    const size_t kv_row = opt_.kv == m4k::KV_I8 ? m4k::kKD : m4k::kKD * 2;
    for (int i = 0; i < n_layers; ++i) {
        Layer& L = elayers_[(size_t) i];
        const std::string p = "layers." + std::to_string(i) + ".";
        L.attn_norm = vec(p + "attention_norm.weight", E);
        L.ffn_norm = vec(p + "ffn_norm.weight", E);
        L.q_a_norm = vec(p + "attention.q_a_norm.weight", hp_.q_lora);
        L.kv_a_norm = vec(p + "attention.kv_a_norm.weight", hp_.kv_lora);
        L.wq_a = mat(p + "attention.wq_a.weight", E, hp_.q_lora);
        L.wq_b = mat(p + "attention.wq_b.weight", hp_.q_lora, H * 128);
        L.wkv_a = mat(p + "attention.wkv_a_with_mqa.weight", E, hp_.kv_lora + hp_.rope);
        L.wo = mat(p + "attention.wo.weight", H * hp_.v_dim, E);
        {   // kv_b [H * (nope + v_dim)][kv_lora] (per head: nope rows, then v rows) -> the absorbed wk_b, wv_b
            std::vector<int64_t> sh;
            const std::vector<float> w = get(p + "attention.wkv_b.weight", &sh);
            const int per = hp_.nope + hp_.v_dim, C = hp_.kv_lora;
            if (sh.size() != 2 || sh[0] != (int64_t) H * per || sh[1] != C) fail("EAGLE: " + p + "attention.wkv_b.weight shape");
            std::vector<float> kb((size_t) H * C * hp_.nope), vb((size_t) H * hp_.v_dim * C);
            for (int h = 0; h < H; ++h)
                for (int c = 0; c < C; ++c) {
                    for (int j = 0; j < hp_.nope; ++j)   // wk_b[h][c][j]
                        kb[((size_t) h * C + c) * hp_.nope + j] = w[((size_t) h * per + j) * C + c];
                    for (int v = 0; v < hp_.v_dim; ++v)   // wv_b[h][v][c]
                        vb[((size_t) h * hp_.v_dim + v) * C + c] = w[((size_t) h * per + hp_.nope + v) * C + c];
                }
            L.wk_b = upload(kb);
            L.wv_b = upload(vb);
            bytes += (kb.size() + vb.size()) * 4;
        }
        L.dense = true;
        L.ffn_gate = mat(p + "feed_forward.w1.weight", E, e_ff_);
        L.ffn_up = mat(p + "feed_forward.w3.weight", E, e_ff_);
        L.ffn_down = mat(p + "feed_forward.w2.weight", e_ff_, E);
        CK(cudaMalloc(&L.kv, (size_t) opt_.ctx * kv_row));
        if (opt_.kv == m4k::KV_I8) CK(cudaMalloc(&L.kv_s, (size_t) opt_.ctx * 2 * 4));
        bytes += (size_t) opt_.ctx * (kv_row + (opt_.kv == m4k::KV_I8 ? 8 : 0));
    }
    enorm_ = vec("norm.weight", E);
    // buffers
    const size_t T = (size_t) Tm_;
    auto fa = [&](float** pp, size_t n) { CK(cudaMalloc(pp, n * 4)); bytes += n * 4; };
    e_rows_ = std::min(Tm_, 512);   // the head runs a long chunk in pieces of this many rows (its buffers stay small)
    const size_t ET = (size_t) e_rows_;
    fa(&mh_, T * E);
    fa(&eemb_, ET * E);
    fa(&ecat_, ET * 2 * E);
    fa(&eg_, ET * e_ff_);
    fa(&eu_, ET * e_ff_);
    fa(&eh2_, ET * e_ff_);
    fa(&eh_, ET * E);
    fa(&elogits_, (size_t) hp_.n_vocab);
    fa(&d_eprob_, kMaxWin);
    CK(cudaMalloc(&ecatq_, q8_bytes(2 * E, (int) ET) + 1024));
    CK(cudaMalloc(&ehq_, q8_bytes(e_ff_, (int) ET) + 1024));
    if (opt_.pf_gemm) {   // the prompt GEMMs' f16 scratch, sized for the head's widest products now (no growth later)
        ensure_w16((size_t) e_ff_ * E);
        ensure_x16(ET * (size_t) std::max(e_ff_, 2 * E));
    }
    CK(cudaMalloc(&d_etok_, T * 4));
    CK(cudaMalloc(&d_edraft_, kMaxWin * 4));
    CK(cudaMalloc(&d_epos_, kMaxWin * 4));
    CK(cudaHostAlloc((void**) &petok_, (T + 2 * kMaxWin + 4) * 4, 0));   // tokens | chain positions | read-back
    eagle_on_ = true;
    logf_("EAGLE head: %d layers, ffn %d, %.2f GiB VRAM (Q8_0 from float8, KV %s), llama-4 scaling %.2f/%g, "
          "%d drafts max, chain probability >= %.2f, loaded in %.1f s",
          n_layers, e_ff_, bytes / GiB, opt_.kv == m4k::KV_I8 ? "int8" : "f16", e_temp_, e_floor_, opt_.eagle_k,
          opt_.eagle_p, ms_since(t0) / 1000.0);
}

void Engine::eagle_ffn(Layer& L, int T) {
    const int E = hp_.n_embd, F = e_ff_;
    m4k::rmsnorm(x_, L.ffn_norm, xn_, E, T, E, E, hp_.eps, s_);
    q8(xn_, xq_, E, T, s_);
    mm(L.ffn_gate, xn_, xq_, eg_, T, s_);
    mm(L.ffn_up, xn_, xq_, eu_, T, s_);
    const size_t n = (size_t) T * F;
    swiglu_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s_>>>(eg_, eu_, eh2_, n);
    q8(eh2_, ehq_, F, T, s_);
    mm(L.ffn_down, eh2_, ehq_, ao_, T, s_);
    m4k::add_inplace(x_, ao_, (size_t) T * E, s_);
}

/// T rows at positions d_pos (pos0 .. pos0+T-1): tokens d_tokens (the token after each position), hidden rows hid
/// [T][E]; leaves the head's output (after its norm) in eh_ and its KV rows in the EAGLE cache.
void Engine::eagle_forward(const int32_t* d_tokens, const float* hid, int T, int pos0, const int32_t* d_pos) {
    const int E = hp_.n_embd;
    K::iq_embed_rows(embd_type_, embd_, embd_row_, d_tokens, T, E, eemb_, s_);
    m4k::concat2(eemb_, hid, ecat_, E, T, s_);
    q8(ecat_, ecatq_, 2 * E, T, s_);
    mm(efc_, ecat_, ecatq_, x_, T, s_);
    for (Layer& L : elayers_) {
        attention(L, T, T <= kMaxWin, pos0, d_pos, e_temp_, e_floor_);
        eagle_ffn(L, T);
    }
    m4k::rmsnorm(x_, enorm_, eh_, E, T, E, E, hp_.eps, s_);
}

/// The model's head on one EAGLE row: greedy token and its probability into d_edraft_[slot], d_eprob_[slot].
void Engine::eagle_pick(const float* row, int slot) {
    q8(row, xq_, hp_.n_embd, 1, s_);
    mm(head_, row, xq_, elogits_, 1, s_);
    m4k::argmax_prob(elogits_, hp_.n_vocab, d_edraft_ + slot, d_eprob_ + slot, s_);
}

/// After the model ran T rows at positions pos0.. (rows_normed: a window, its head left the normed rows in xn_;
/// otherwise a prompt chunk with the residual stream in x_): the EAGLE rows for the same positions, with next[i] the
/// token at pos0+i+1.  Leaves the first draft (the token after next[T-1]) in d_edraft_[0].
void Engine::eagle_follow(const int32_t* next, int T, int pos0, bool rows_normed) {
    const int E = hp_.n_embd;
    if (rows_normed) CK(cudaMemcpyAsync(mh_, xn_, (size_t) T * E * 4, cudaMemcpyDeviceToDevice, s_));
    else m4k::rmsnorm(x_, out_norm_, mh_, E, T, E, E, hp_.eps, s_);
    for (int i = 0; i < T; ++i) petok_[i] = next[i];
    CK(cudaMemcpyAsync(d_etok_, petok_, (size_t) T * 4, cudaMemcpyHostToDevice, s_));
    int last = 0;
    for (int o = 0; o < T; o += e_rows_) {   // pieces of e_rows_ rows, in order (each attends to the ones before)
        last = std::min(e_rows_, T - o);
        eagle_forward(d_etok_ + o, mh_ + (size_t) o * E, last, pos0 + o, d_pos_ + o);
    }
    eagle_pick(eh_ + (size_t) (last - 1) * E, 0);
    e_last_ = eh_ + (size_t) (last - 1) * E;
    eagle_ready_ = true;
    if (pos0 <= eagle_valid_) eagle_valid_ = std::max(eagle_valid_, pos0 + T);
}

/// Drafts for the window whose first token sits at position P (eagle_follow just ran up to P-1): the first draft is
/// d_edraft_[0]; each further one is an EAGLE step at P, P+1, ... on its own output.  The chain stops at k drafts or
/// when the product of the steps' probabilities falls below pmin.
void Engine::eagle_chain(int P, int k, double pmin, std::vector<int32_t>& out) {
    out.clear();
    if (!eagle_ready_ || k <= 0) return;
    for (int j = 1; j < k; ++j) petok_[Tm_ + j] = P + j - 1;
    if (k > 1) CK(cudaMemcpyAsync(d_epos_ + 1, petok_ + Tm_ + 1, (size_t) (k - 1) * 4, cudaMemcpyHostToDevice, s_));
    double cum = 1.0;
    for (int j = 0; j < k; ++j) {
        if (j > 0) {
            eagle_forward(d_edraft_ + (j - 1), e_last_, 1, P + j - 1, d_epos_ + j);
            eagle_pick(eh_, j);
            e_last_ = eh_;
        }
        int32_t tok = 0;
        float pr = 0.f;
        CK(cudaMemcpyAsync(&petok_[Tm_ + kMaxWin], d_edraft_ + j, 4, cudaMemcpyDeviceToHost, s_));
        CK(cudaMemcpyAsync(&petok_[Tm_ + kMaxWin + 1], d_eprob_ + j, 4, cudaMemcpyDeviceToHost, s_));
        CK(cudaStreamSynchronize(s_));
        tok = petok_[Tm_ + kMaxWin];
        std::memcpy(&pr, &petok_[Tm_ + kMaxWin + 1], 4);
        cum *= pr;
        if (cum < pmin || tok < 0 || tok >= hp_.n_vocab) break;
        out.push_back(tok);
    }
    eagle_ready_ = false;   // the chain moved past the model's rows: the next window's eagle_follow renews it
}

// ------------------------------------------------------------------------------------------------ EAGLE check
// Teacher-forced top-1 accuracy over chunks of the ids: the model's (row p predicts token p+1) and the EAGLE head's
// (row p, fed token p+1 and the model's h_p, predicts token p+2), and how often the head agrees with the model's own
// next pick - the acceptance a greedy draft of one token would get.
double Engine::eagle_acc(const std::vector<int32_t>& ids) {
    const int E = hp_.n_embd, C = std::min(512, Tm_);
    long m_ok = 0, e_ok = 0, e_agree = 0, n_m = 0, n_e = 0, n_a = 0;
    std::vector<int32_t> mp((size_t) C), ep((size_t) C);
    const auto t0 = Clock::now();
    for (size_t a = 0; a + (size_t) C + 1 <= ids.size(); a += (size_t) C) {
        cached_.clear();
        eagle_valid_ = 0;
        const int T = C - 1;   // positions 0 .. C-2; the head reads token p+1 <= C-1
        std::vector<int32_t> w(ids.begin() + (long) a, ids.begin() + (long) a + C);
        if (bos_ >= 0) w[0] = bos_;
        prefill(w.data(), T, 0, nullptr, 0, false);
        for (int f = 0; f < T; f += kMaxWin) {   // the model's picks (head() normalizes x_ rows into xn_)
            const int nr = std::min(kMaxWin, T - f);
            head(f, nr);
            m4k::argmax_rows(logits_, hp_.n_vocab, nr, d_argmax_, s_);
            CK(cudaMemcpyAsync(mp.data() + f, d_argmax_, (size_t) nr * 4, cudaMemcpyDeviceToHost, s_));
        }
        eagle_follow(w.data() + 1, T, 0, false);
        for (int f = 0; f < T; f += kMaxWin) {
            const int nr = std::min(kMaxWin, T - f);
            q8(eh_ + (size_t) f * E, xq_, E, nr, s_);
            mm(head_, eh_ + (size_t) f * E, xq_, logits_, nr, s_);
            m4k::argmax_rows(logits_, hp_.n_vocab, nr, d_argmax_, s_);
            CK(cudaMemcpyAsync(ep.data() + f, d_argmax_, (size_t) nr * 4, cudaMemcpyDeviceToHost, s_));
        }
        CK(cudaStreamSynchronize(s_));
        for (int p = 0; p < T; ++p) {
            m_ok += mp[(size_t) p] == w[(size_t) p + 1];
            ++n_m;
            if (p + 2 < C) {
                e_ok += ep[(size_t) p] == w[(size_t) p + 2];
                ++n_e;
            }
            if (p + 1 < T) {
                e_agree += ep[(size_t) p] == mp[(size_t) p + 1];
                ++n_a;
            }
        }
        logf_("eagle-acc chunk %zu: model top-1 %.1f %%, EAGLE top-1 %.1f %%, EAGLE = model's next pick %.1f %% (%.1f s)",
              a / (size_t) C + 1, 100.0 * m_ok / std::max(1L, n_m), 100.0 * e_ok / std::max(1L, n_e),
              100.0 * e_agree / std::max(1L, n_a), ms_since(t0) / 1000.0);
    }
    std::printf("EAGLE-ACC model %.2f %% eagle %.2f %% agree %.2f %%\n", 100.0 * m_ok / std::max(1L, n_m),
                100.0 * e_ok / std::max(1L, n_e), 100.0 * e_agree / std::max(1L, n_a));
    cached_.clear();
    eagle_valid_ = 0;
    return (double) e_agree / std::max(1L, n_a);
}

// ------------------------------------------------------------------------------------------------ decode windows
void Engine::enqueue_window(int T) {
    embed(d_tok_, T);
    for (int l = 0; l < hp_.n_layer; ++l) {
        Layer& L = layers_[(size_t) l];
        attention(L, T, true, win_pos0_);
        if (L.dense) ffn_dense(L, T);
        else moe_window(l, T);
    }
    head(0, T);
}

void Engine::serve_host_layers(int T) {
    const int E = hp_.n_embd, k = hp_.top_k, R = T * k;
    const int want = seq_;
    for (int l = 0; l < hp_.n_layer; ++l) {
        Layer& L = layers_[(size_t) l];
        if (L.dense) continue;
        const auto t0 = Clock::now();
        while (hs_->pub[l] != want) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
            if (hs_->err) fail("the GPU gave up waiting for the host");
            if (ms_since(t0) > 60000) fail("no routes from the GPU for layer " + std::to_string(l) + " after 60 s");
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        check_ids(l, hids_, R);
        count_heat(l, hids_, R);
        call_.clear();
        call_.x = hx_;
        call_.n_tok = T;
        call_.fmt = &L.fmt;
        // group the CPU routes by expert; route r writes hcpu_[r]
        for (int r = 0; r < R; ++r) {
            const int e = hids_[r];
            if (L.slot_of[(size_t) e] >= 0) continue;
            bool found = false;
            for (auto& en : call_.entries)
                if (en.ex == &L.cpu[(size_t) e]) { found = true; break; }
            if (found) continue;
            CpuCall::Entry en;
            en.ex = &L.cpu[(size_t) e];
            en.off = (int) call_.tok.size();
            for (int r2 = r; r2 < R; ++r2)
                if (hids_[r2] == e) {
                    call_.tok.push_back(r2 / k);
                    call_.out.push_back(hcpu_ + (size_t) r2 * E);
                }
            en.nt = (int) call_.tok.size() - en.off;
            call_.entries.push_back(en);
        }
        if (!call_.entries.empty()) pool_->run(call_);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        hs_->done[l] = want;
    }
}

void Engine::run_window(const std::vector<int32_t>& toks, int pos0, bool all_logits) {
    (void) all_logits;
    const int T = (int) toks.size();
    win_pos0_ = pos0;
    for (int i = 0; i < T; ++i) {
        ptok_[i] = toks[(size_t) i];
        ptok_[kMaxWin + i] = pos0 + i;
    }
    CK(cudaMemcpyAsync(d_tok_, ptok_, (size_t) T * 4, cudaMemcpyHostToDevice, s_));
    CK(cudaMemcpyAsync(d_pos_, ptok_ + kMaxWin, (size_t) T * 4, cudaMemcpyHostToDevice, s_));
    hs_->seq = ++seq_;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // Eagerly enqueued windows (the first of each size, or --no-graph) are ~1000 launches: more than the stream's
    // launch queue holds.  The host would block in a launch while the GPU waits in layer 0 for the host's experts -
    // a deadlock.  So the layers are served by a second host thread (on the host core's SMT sibling) while this one
    // enqueues.  A graph launch is one call, so the graph path serves on this thread.
    auto eager = [&]() {
        std::thread srv([&] {
            if (host_sib_ >= 0) pin_thread({host_sib_});
            serve_host_layers(T);
        });
        enqueue_window(T);
        srv.join();
        CK(cudaStreamSynchronize(s_));
    };
    if (opt_.graph) {
        auto it = graphs_.find(T);
        if (it == graphs_.end()) {
            // the first window of a size runs eagerly (lazy inits, cuBLAS heuristics), and is captured after it
            eager();
            cudaGraph_t g;
            CK(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal));
            enqueue_window(T);
            CK(cudaStreamEndCapture(s_, &g));
            cudaGraphExec_t ex;
            CK(cudaGraphInstantiate(&ex, g, 0));
            CK(cudaGraphDestroy(g));
            graphs_[T] = ex;
            logf_("decode window of %d token%s captured as a CUDA graph", T, T == 1 ? "" : "s");
            return;   // the eager run produced the logits
        }
        CK(cudaGraphLaunch(it->second, s_));
        serve_host_layers(T);
        CK(cudaStreamSynchronize(s_));
    } else {
        eager();
    }
}

// ------------------------------------------------------------------------------------------------ prompts
void Engine::prefill(const int32_t* toks, int T, int pos0, const GenParams* gp, int prompt_off, bool all_logits) {
    (void) all_logits;
    for (int i = 0; i < T; ++i) {
        ptok_[i] = toks[i];
        ptok_[Tm_ + i] = pos0 + i;
    }
    CK(cudaMemcpyAsync(d_tok_, ptok_, (size_t) T * 4, cudaMemcpyHostToDevice, s_));
    CK(cudaMemcpyAsync(d_pos_, ptok_ + Tm_, (size_t) T * 4, cudaMemcpyHostToDevice, s_));
    embed(d_tok_, T);
    // image rows of this chunk
    if (gp && gp->embd.is_array()) {
        std::vector<int32_t> rows;
        std::vector<float> data;
        for (const auto& im : gp->embd) {
            const std::string file = im.value("file", "");
            const auto& pr = im["rows"];
            std::ifstream f(file, std::ios::binary);
            if (!f) fail("image embeddings " + file + " cannot be read");
            for (size_t j = 0; j < pr.size(); ++j) {
                const int p = pr[j].get<int>() - prompt_off;
                if (p < 0 || p >= T) continue;
                std::vector<float> row((size_t) hp_.n_embd);
                f.seekg((std::streamoff) (j * (size_t) hp_.n_embd * 4));
                f.read((char*) row.data(), (std::streamsize) (row.size() * 4));
                rows.push_back(p);
                data.insert(data.end(), row.begin(), row.end());
            }
        }
        if (!rows.empty()) {
            CK(cudaMemcpyAsync(img_, data.data(), data.size() * 4, cudaMemcpyHostToDevice, s_));
            CK(cudaMemcpyAsync(d_img_rows_, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice, s_));
            m4k::scatter_rows(img_, d_img_rows_, (int) rows.size(), hp_.n_embd, x_, s_);
            CK(cudaStreamSynchronize(s_));
        }
    }
    for (int l = 0; l < hp_.n_layer; ++l) {
        Layer& L = layers_[(size_t) l];
        attention(L, T, false, pos0);
        if (L.dense) ffn_dense(L, T);
        else if (T >= opt_.pf_min && stream_ok(L)) moe_prefill_stream(l, T);
        else moe_prefill(l, T);
    }
}

// ------------------------------------------------------------------------------------------------ sampling
int Engine::sample(const float* lg, const GenParams& gp, std::mt19937_64& rng) {
    const int n = hp_.n_vocab;
    if (gp.temp <= 0.f) return (int) (std::max_element(lg, lg + n) - lg);
    // the best 512 candidates (the mass beyond them is negligible for any sane temperature)
    const int cap = std::min(n, gp.top_k > 0 ? gp.top_k : 512);
    std::vector<int> idx((size_t) n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + cap, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) cap);
    std::vector<double> p((size_t) cap);
    const double m = lg[idx[0]];
    double sum = 0;
    for (int i = 0; i < cap; ++i) {
        p[(size_t) i] = std::exp(((double) lg[idx[(size_t) i]] - m) / gp.temp);
        sum += p[(size_t) i];
    }
    int keep = cap;
    if (gp.min_p > 0.f) {
        for (int i = 0; i < cap; ++i)
            if (p[(size_t) i] < gp.min_p * p[0]) { keep = std::max(1, i); break; }
    }
    if (gp.top_p < 1.f) {
        double c = 0;
        for (int i = 0; i < keep; ++i) {
            c += p[(size_t) i] / sum;
            if (c >= gp.top_p) { keep = i + 1; break; }
        }
    }
    double s2 = 0;
    for (int i = 0; i < keep; ++i) s2 += p[(size_t) i];
    std::uniform_real_distribution<double> U(0.0, s2);
    double r = U(rng);
    for (int i = 0; i < keep; ++i) {
        r -= p[(size_t) i];
        if (r <= 0) return idx[(size_t) i];
    }
    return idx[(size_t) keep - 1];
}

// ------------------------------------------------------------------------------------------------ debug run
void Engine::dump(const std::vector<int32_t>& ids) {
    dbg_layers_.clear();
    for (size_t a = 0; a < opt_.dump_layers.size();) {
        const size_t b = opt_.dump_layers.find(',', a);
        dbg_layers_.push_back(std::atoi(opt_.dump_layers.substr(a, b - a).c_str()));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    const int T = (int) ids.size();
    std::fprintf(stderr, "DUMP tokens (%d):", T);
    for (int t : ids) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "\n");
    dbg_on_ = true;
    prefill(ids.data(), T, 0, nullptr, 0, false);
    // final norm and head for every row (8 at a time)
    for (int a = 0; a < T; a += kMaxWin) {
        const int n = std::min(kMaxWin, T - a);
        head(a, n);
        if (a == 0) dbg("result_norm", -1, xn_, n, hp_.n_embd);
        CK(cudaMemcpy(hlogits_.data(), logits_, (size_t) n * hp_.n_vocab * 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            const float* lg = hlogits_.data() + (size_t) i * hp_.n_vocab;
            std::vector<int> idx((size_t) hp_.n_vocab);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int x, int y) { return lg[x] > lg[y]; });
            double s = 0;
            for (int v = 0; v < hp_.n_vocab; ++v) s += lg[v];
            std::fprintf(stderr, "DUMP logits row %d (after token %d): sum %.3f, top5:", a + i, ids[(size_t) (a + i)], s);
            for (int j = 0; j < 5; ++j) std::fprintf(stderr, " %d(%.3f)", idx[(size_t) j], lg[idx[(size_t) j]]);
            std::fprintf(stderr, "\n");
        }
    }
    dbg_on_ = false;
    auto top5 = [&](const char* what) {
        CK(cudaMemcpy(hlogits_.data(), logits_, (size_t) hp_.n_vocab * 4, cudaMemcpyDeviceToHost));
        const float* lg = hlogits_.data();
        std::vector<int> idx((size_t) hp_.n_vocab);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int x, int y) { return lg[x] > lg[y]; });
        std::fprintf(stderr, "DUMP %s logits after token %d: top5:", what, ids.back());
        for (int j = 0; j < 5; ++j) std::fprintf(stderr, " %d(%.3f)", idx[(size_t) j], lg[idx[(size_t) j]]);
        std::fprintf(stderr, "\n");
    };
    // the same prompt through the decode windows, one token at a time, launched eagerly; the last window prints its
    // intermediates: its row [0] must equal row [last] of the prompt pass above
    const bool graph = opt_.graph;
    opt_.graph = false;
    std::vector<int32_t> w{ids[0]};
    for (int t = 0; t < T; ++t) {
        w[0] = ids[(size_t) t];
        if (t == T - 1) {
            std::fprintf(stderr, "DUMP ===== decode window (eager), token %d at position %d =====\n", w[0], t);
            dbg_on_ = true;
        }
        run_window(w, t, false);
        dbg_on_ = false;
    }
    head(0, 1);
    top5("decode-eager");
    // and once more through the CUDA graphs
    opt_.graph = graph;
    for (int t = 0; t < T; ++t) {
        w[0] = ids[(size_t) t];
        run_window(w, t, false);
    }
    top5("decode-graph");
    // a window of all T tokens at once (the verify path of speculation)
    if (T <= kMaxWin) {
        std::vector<int32_t> all(ids.begin(), ids.end());
        opt_.graph = false;
        run_window(all, 0, false);
        CK(cudaMemcpy(hlogits_.data(), logits_ + (size_t) (T - 1) * hp_.n_vocab, (size_t) hp_.n_vocab * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(logits_, hlogits_.data(), (size_t) hp_.n_vocab * 4, cudaMemcpyHostToDevice));
        top5("window-of-all (last row)");
        opt_.graph = graph;
    }
}

// ------------------------------------------------------------------------------------------------ generation
Engine::GenStats Engine::generate(const std::vector<int32_t>& prompt, const GenParams& gp,
                                  const std::function<bool(const std::vector<int32_t>&)>& emit,
                                  const std::atomic<bool>* stop, const GenHooks* hooks) {
    GenStats st;
    const int n = (int) prompt.size();
    st.n_prompt = n;
    if (n == 0) { st.finish = "error"; return st; }
    if (n + gp.max_new + kMaxWin >= opt_.ctx) {
        if (n + kMaxWin >= opt_.ctx) { st.finish = "context"; return st; }
    }
    std::mt19937_64 rng(gp.seed ? gp.seed : (uint64_t) std::random_device{}());
    // the prefix whose KV is still valid
    int reuse = 0;
    while (reuse < (int) cached_.size() && reuse < n - 1 && cached_[(size_t) reuse] == prompt[(size_t) reuse]) ++reuse;
    if (gp.embd_key != cached_embd_key_ && gp.embd.is_array()) {
        int first = n;
        for (const auto& im : gp.embd)
            for (const auto& r : im["rows"]) first = std::min(first, r.get<int>());
        reuse = std::min(reuse, first);
    }
    const int spec = gp.spec >= 0 ? std::min(gp.spec, kMaxWin - 1) : opt_.spec;
    // EAGLE rows exist only where the head followed the model; a request without drafts does not keep them
    const bool use_eagle = eagle_on_ && spec > 0;
    if (use_eagle) reuse = std::min(reuse, eagle_valid_);
    else eagle_valid_ = std::min(eagle_valid_, reuse);
    eagle_ready_ = false;
    st.n_reused = reuse;
    const auto tp = Clock::now();
    const double streamed0 = pf_stream_bytes_;
    for (int a = reuse; a < n - 1; a += opt_.chunk) {
        const int T = std::min(opt_.chunk, n - 1 - a);
        const bool win = T <= kMaxWin && !(gp.embd.is_array() && !gp.embd.empty());
        if (win) {
            std::vector<int32_t> w(prompt.begin() + a, prompt.begin() + a + T);
            run_window(w, a, false);
        } else {
            prefill(prompt.data() + a, T, a, &gp, a, false);
        }
        if (use_eagle) eagle_follow(prompt.data() + a + 1, T, a, win);
        if (hooks && hooks->progress) {
            CK(cudaStreamSynchronize(s_));
            hooks->progress(a + T - reuse, n - 1 - reuse, reuse);
        }
    }
    CK(cudaStreamSynchronize(s_));
    st.prompt_ms = ms_since(tp);
    if (opt_.mode != "serve")
        logf_("prompt: %d tokens (%d reused) in %.0f ms, %.1f GiB of experts streamed", n, reuse, st.prompt_ms,
              (pf_stream_bytes_ - streamed0) / GiB);
    cached_.assign(prompt.begin(), prompt.end() - 1);
    cached_embd_key_ = gp.embd_key;

    std::vector<int32_t> hist(prompt.begin(), prompt.end());
    int n_past = n - 1;
    int32_t cur = prompt.back();
    const auto tg = Clock::now();
    std::vector<int32_t> drafts, out;
    double acc_rate = 0.5;
    bool done = false;
    while (!done) {
        if (stop && stop->load()) { st.finish = "abort"; break; }
        int src = 0;
        // adaptive drafting: a window of k drafts touches up to 4(k+1) experts per layer, so drafts that are mostly
        // rejected cost more than they save.  The draft length follows the acceptance rate of the request so far
        // (an exponential average per drafted token); the global table's chains (source 2) are capped shorter than
        // the lookup's; below 20 % the drafter pauses and probes with 2 tokens every 8 windows.
        int room = std::min(spec, std::max(0, gp.max_new - st.n_gen - 1));
        if (spec > 0) {
            if (acc_rate >= 0.6) room = std::min(room, spec);
            else if (acc_rate >= 0.4) room = std::min(room, std::max(1, spec / 2));
            else if (acc_rate >= 0.2) room = std::min(room, 2);
            else room = (st.windows % 8 == 0) ? std::min(room, 2) : 0;
        }
        ngram_.draft(hist, room, drafts, src);
        if (src == 2 && (int) drafts.size() > std::max(1, room / 2)) drafts.resize((size_t) std::max(1, room / 2));
        // EAGLE drafts wherever the history lookup (source 1: repeated text, code edits) has none
        bool from_eagle = false;
        if (use_eagle && !(src == 1 && !drafts.empty())) {
            std::vector<int32_t> ed;
            const int ke = std::max(0, std::min({opt_.eagle_k, gp.max_new - st.n_gen - 1, opt_.ctx - n_past - 2}));
            eagle_chain(n_past, ke, opt_.eagle_p, ed);
            if (!ed.empty() || src != 2) {
                drafts = ed;
                from_eagle = true;
                src = 3;
            }
        }
        if (n_past + 1 + (int) drafts.size() >= opt_.ctx) drafts.clear();
        if (n_past + 1 >= opt_.ctx) { st.finish = "context"; break; }
        std::vector<int32_t> win{cur};
        win.insert(win.end(), drafts.begin(), drafts.end());
        const int T = (int) win.size();
        run_window(win, n_past, true);
        ++st.windows;
        st.drafted += (int) drafts.size();
        // logits of every row only when sampling needs them; greedy: argmax on the GPU
        out.clear();
        if (gp.temp <= 0.f) {
            m4k::argmax_rows(logits_, hp_.n_vocab, T, d_argmax_, s_);
            int32_t am[kMaxWin];
            CK(cudaMemcpyAsync(am, d_argmax_, (size_t) T * 4, cudaMemcpyDeviceToHost, s_));
            CK(cudaStreamSynchronize(s_));
            for (int i = 0; i < T; ++i) {
                out.push_back(am[i]);
                if (i + 1 < T && am[i] == win[(size_t) i + 1]) continue;
                break;
            }
        } else {
            CK(cudaMemcpy(hlogits_.data(), logits_, (size_t) T * hp_.n_vocab * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < T; ++i) {
                const int y = sample(hlogits_.data() + (size_t) i * hp_.n_vocab, gp, rng);
                out.push_back(y);
                if (i + 1 < T && y == win[(size_t) i + 1]) continue;
                break;
            }
        }
        st.accepted += (int) out.size() - 1;
        if (from_eagle) {
            st.e_drafted += (int) drafts.size();
            st.e_accepted += (int) out.size() - 1;
            ++st.e_windows;
        } else if (!drafts.empty()) {   // the share of this window's n-gram drafts that were taken, over the request
            acc_rate = 0.7 * acc_rate + 0.3 * (double) ((int) out.size() - 1) / (double) drafts.size();
        }
        if (hooks && hooks->window) hooks->window(T, (int) out.size());
        // the EAGLE head follows the model over the rows that stay (cur and the accepted drafts)
        if (use_eagle) eagle_follow(out.data(), (int) out.size(), n_past, true);
        // KV rows of cur and the accepted drafts are final
        n_past += (int) out.size();
        std::vector<int32_t> emitted;
        for (int32_t y : out) {
            emitted.push_back(y);
            hist.push_back(y);
            ++st.n_gen;
            if (std::find(gp.stop_ids.begin(), gp.stop_ids.end(), y) != gp.stop_ids.end()) {
                st.finish = "stop";
                done = true;
                break;
            }
            if (st.n_gen >= gp.max_new) {
                st.finish = "length";
                done = true;
                break;
            }
        }
        cur = hist.back();
        if (!emit(emitted)) { st.finish = "abort"; break; }
    }
    st.gen_ms = ms_since(tg);
    // KV valid for hist[0 .. n_past)
    cached_.assign(hist.begin(), hist.begin() + std::min<size_t>((size_t) n_past, hist.size()));
    ngram_.learn(hist, (size_t) std::max(0, n - 3));
    return st;
}

double Engine::perplexity(const std::vector<int32_t>& ids, int chunk) {
    // llama-perplexity style: independent chunks of `chunk` tokens, the second half of each scored
    double nll = 0;
    int cnt = 0;
    const auto t0 = Clock::now();
    std::vector<float> row((size_t) hp_.n_vocab);
    eagle_valid_ = 0;
    for (size_t a = 0; a + (size_t) chunk <= ids.size(); a += (size_t) chunk) {
        cached_.clear();
        for (int b = 0; b < chunk; b += kMaxWin) {
            const int T = std::min(kMaxWin, chunk - b);
            if (b < chunk / 2 - kMaxWin) {   // the unscored first half as a prompt (one chunk at a time)
                const int T2 = std::min(opt_.chunk, chunk / 2 - b);
                std::vector<int32_t> w(ids.begin() + (long) (a + (size_t) b), ids.begin() + (long) (a + (size_t) b + (size_t) T2));
                if (b == 0 && bos_ >= 0) w[0] = bos_;   // llama-perplexity: every chunk starts with BOS
                prefill(w.data(), T2, b, nullptr, 0, false);
                b += T2 - kMaxWin;
                continue;
            }
            std::vector<int32_t> w(ids.begin() + (long) (a + (size_t) b), ids.begin() + (long) (a + (size_t) b + (size_t) T));
            if (b == 0 && bos_ >= 0) w[0] = bos_;
            run_window(w, b, true);
            CK(cudaMemcpy(hlogits_.data(), logits_, (size_t) T * hp_.n_vocab * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < T; ++i) {
                const int pos = b + i;
                if (pos < chunk / 2 || pos + 1 >= chunk) continue;
                const float* lg = hlogits_.data() + (size_t) i * hp_.n_vocab;
                const float m = *std::max_element(lg, lg + hp_.n_vocab);
                double s = 0;
                for (int v = 0; v < hp_.n_vocab; ++v) s += std::exp((double) lg[v] - m);
                const int y = ids[a + (size_t) pos + 1];
                nll += -((double) lg[y] - m - std::log(s));
                ++cnt;
            }
        }
        logf_("ppl chunk %zu: %.4f (%d tokens scored, %.1f s)", a / (size_t) chunk + 1, std::exp(nll / std::max(1, cnt)), cnt,
              ms_since(t0) / 1000.0);
    }
    return std::exp(nll / std::max(1, cnt));
}

// ================================================================================================ serving
struct LineReader {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::atomic<bool> stop{false}, eof{false};
    std::thread th;
    void start() {
        th = std::thread([this] {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line == "STOP") {
                    stop.store(true);
                    continue;
                }
                std::lock_guard<std::mutex> g(mu);
                q.push_back(line);
                cv.notify_one();
            }
            eof.store(true);
            stop.store(true);
            cv.notify_one();
        });
        th.detach();
    }
    bool next(std::string& out) {
        std::unique_lock<std::mutex> g(mu);
        cv.wait(g, [&] { return !q.empty() || eof.load(); });
        if (q.empty()) return false;
        out = q.front();
        q.pop_front();
        return true;
    }
};

void out_line(const std::string& s) {
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

int serve(Engine& eng) {
    LineReader rd;
    rd.start();
    json ready = eng.info();
    out_line("READY " + ready.dump());
    std::string line;
    auto last_heat = Clock::now();
    while (rd.next(line)) {
        if (line.rfind("GEN ", 0) == 0) {
            json req;
            try {
                req = json::parse(line.substr(4));
            } catch (const std::exception& e) {
                out_line(std::string("ERR bad json: ") + e.what());
                continue;
            }
            Engine::GenParams gp;
            gp.max_new = req.value("max_new", 256);
            gp.temp = req.value("temp", 0.f);
            gp.top_p = req.value("top_p", 1.f);
            gp.top_k = req.value("top_k", 0);
            gp.min_p = req.value("min_p", 0.f);
            gp.spec = req.value("spec", -1);
            gp.seed = req.value("seed", (uint64_t) 0);
            if (req.contains("stop_ids")) gp.stop_ids = req["stop_ids"].get<std::vector<int32_t>>();
            if (req.contains("embd")) gp.embd = req["embd"];
            gp.embd_key = req.value("embd_key", "");
            std::vector<int32_t> ids;
            try {
                ids = req.at("ids").get<std::vector<int32_t>>();
            } catch (const std::exception& e) {
                out_line(std::string("ERR ") + e.what());
                continue;
            }
            rd.stop.store(false);
            Engine::GenHooks hk;
            hk.progress = [](int done, int total, int reused) {
                out_line("PP " + std::to_string(done) + " " + std::to_string(total) + " " + std::to_string(reused));
            };
            hk.window = [](int rows, int taken) { out_line("W " + std::to_string(rows) + " " + std::to_string(taken)); };
            const auto st = eng.generate(ids, gp, [&](const std::vector<int32_t>& t) {
                std::string s = "T";
                for (int32_t v : t) s += " " + std::to_string(v);
                out_line(s);
                return true;
            }, &rd.stop, &hk);
            json d{{"n_prompt", st.n_prompt}, {"n_reused", st.n_reused}, {"n_gen", st.n_gen},
                   {"prompt_ms", st.prompt_ms}, {"gen_ms", st.gen_ms}, {"drafted", st.drafted},
                   {"accepted", st.accepted}, {"windows", st.windows}, {"eagle_drafted", st.e_drafted},
                   {"eagle_accepted", st.e_accepted}, {"eagle_windows", st.e_windows}, {"finish", st.finish}};
            out_line("DONE " + d.dump());
            if (ms_since(last_heat) > 300000) {
                eng.save_heat();
                last_heat = Clock::now();
            }
        } else if (line == "INFO") {
            out_line("INFO " + eng.info().dump());
        } else if (line == "SAVEHEAT") {
            eng.save_heat();
            out_line("OK");
        } else if (line == "QUIT") {
            break;
        } else if (!line.empty()) {
            out_line("ERR unknown command");
        }
    }
    eng.save_heat();
    return 0;
}

// ------------------------------------------------------------------------------------------------ attention self-test
// Random queries and a random latent cache; the split-K decode kernel and the cuBLAS route against a CPU reference
// in double.  No model needed.
int selftest_attn(int gpu) {
    CK(cudaSetDevice(gpu));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cublasHandle_t cb;
    CB(cublasCreate(&cb));
    CB(cublasSetStream(cb, s));
    const int H = 32, D = m4k::kKD, LAT = m4k::kLat;
    const float scale = 0.19497f;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;
    for (int kv_mode : {m4k::KV_F16, m4k::KV_I8}) {
        for (int T : {1, 6, 16}) {
            for (int n_kv : {1, 6, 33, 100, 1000, 5000}) {
                if (n_kv < T) continue;
                const int pos0 = n_kv - T;   // the window's tokens are the last T rows
                // cache rows: latent ~ N(0, 0.05), rope ~ N(0, 3) (the scales seen in the model)
                std::vector<float> Kf((size_t) n_kv * D);
                for (int r = 0; r < n_kv; ++r)
                    for (int d = 0; d < D; ++d) Kf[(size_t) r * D + d] = nd(rng) * (d < LAT ? 0.05f : 3.f);
                std::vector<float> Q((size_t) T * H * D);
                for (auto& v : Q) v = nd(rng) * 2.f;
                // the cache as the engine stores it, and the values it represents (what the reference must use)
                std::vector<__half> c16((size_t) n_kv * D);
                std::vector<int8_t> c8((size_t) n_kv * D);
                std::vector<float> sc((size_t) n_kv * 2), Kr((size_t) n_kv * D);
                for (int r = 0; r < n_kv; ++r) {
                    float am[2] = {0, 0};
                    for (int d = 0; d < D; ++d) am[d >= LAT] = std::max(am[d >= LAT], std::fabs(Kf[(size_t) r * D + d]));
                    for (int h2 = 0; h2 < 2; ++h2) sc[(size_t) r * 2 + h2] = am[h2] > 0 ? am[h2] / 127.f : 1.f;
                    for (int d = 0; d < D; ++d) {
                        const float v = Kf[(size_t) r * D + d];
                        c16[(size_t) r * D + d] = __float2half(v);
                        const int q = std::max(-127, std::min(127, (int) std::lrint(v / sc[(size_t) r * 2 + (d >= LAT)])));
                        c8[(size_t) r * D + d] = (int8_t) q;
                        Kr[(size_t) r * D + d] = kv_mode == m4k::KV_F16 ? __half2float(c16[(size_t) r * D + d])
                                                                       : (float) q * sc[(size_t) r * 2 + (d >= LAT)];
                    }
                }
                // reference
                std::vector<double> ref((size_t) T * H * LAT);
                for (int t = 0; t < T; ++t)
                    for (int h = 0; h < H; ++h) {
                        const int lim = pos0 + t + 1;
                        std::vector<double> sco((size_t) lim);
                        double m = -1e300;
                        for (int k = 0; k < lim; ++k) {
                            double a = 0;
                            for (int d = 0; d < D; ++d) a += (double) Q[((size_t) t * H + h) * D + d] * Kr[(size_t) k * D + d];
                            sco[(size_t) k] = a * scale;
                            m = std::max(m, sco[(size_t) k]);
                        }
                        double l = 0;
                        for (int k = 0; k < lim; ++k) l += (sco[(size_t) k] = std::exp(sco[(size_t) k] - m));
                        for (int d = 0; d < LAT; ++d) {
                            double o = 0;
                            for (int k = 0; k < lim; ++k) o += sco[(size_t) k] * Kr[(size_t) k * D + d];
                            ref[((size_t) t * H + h) * LAT + d] = o / l;
                        }
                    }
                // device data
                void* cache = nullptr;
                float *scl = nullptr, *dQ = nullptr, *part = nullptr, *O = nullptr, *O2 = nullptr, *O3 = nullptr, *S = nullptr;
                int32_t* dpos = nullptr;
                __half *K16 = nullptr, *Q16 = nullptr, *P = nullptr;
                const size_t cb_bytes = (size_t) n_kv * D * (kv_mode == m4k::KV_F16 ? 2 : 1);
                CK(cudaMalloc(&cache, cb_bytes));
                CK(cudaMemcpy(cache, kv_mode == m4k::KV_F16 ? (const void*) c16.data() : (const void*) c8.data(), cb_bytes, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&scl, sc.size() * 4));
                CK(cudaMemcpy(scl, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&dQ, Q.size() * 4));
                CK(cudaMemcpy(dQ, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice));
                std::vector<int32_t> pos((size_t) T);
                for (int t = 0; t < T; ++t) pos[(size_t) t] = pos0 + t;
                CK(cudaMalloc(&dpos, (size_t) T * 4));
                CK(cudaMemcpy(dpos, pos.data(), (size_t) T * 4, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&part, m4k::mla_decode_part_floats(T, H) * 4));
                CK(cudaMemset(part, 0x7f, m4k::mla_decode_part_floats(T, H) * 4));   // poison: stale data shows
                CK(cudaMalloc(&O, (size_t) T * H * LAT * 4));
                CK(cudaMalloc(&O2, (size_t) T * H * LAT * 4));
                CK(cudaMalloc(&O3, (size_t) T * H * LAT * 4));
                m4k::mla_attn_decode(dQ, cache, scl, kv_mode, dpos, T, H, scale, part, O, s, m4k::ATTN_SIMT);
                CK(cudaMemset(part, 0x7f, m4k::mla_decode_part_floats(T, H) * 4));
                m4k::mla_attn_decode(dQ, cache, scl, kv_mode, dpos, T, H, scale, part, O3, s, m4k::ATTN_TC);
                // the cuBLAS route (as attention() runs it for prompts)
                CK(cudaMalloc(&K16, (size_t) n_kv * D * 2));
                CK(cudaMalloc(&Q16, Q.size() * 2));
                CK(cudaMalloc(&S, (size_t) n_kv * T * H * 4));
                CK(cudaMalloc(&P, (size_t) n_kv * T * H * 2));
                m4k::mla_cache_to_f16(cache, scl, kv_mode, n_kv, K16, s);
                m4k::f32_to_f16_scaled(dQ, Q16, Q.size(), scale, s);
                const float one = 1.f, zero = 0.f;
                const int cols = T * H, m = n_kv;
                CB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, m, cols, D, &one, K16, CUDA_R_16F, D, Q16, CUDA_R_16F, D, &zero, S,
                                CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                m4k::mla_softmax_causal(S, P, m, cols, H, pos0, s);
                CB(cublasGemmEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, LAT, cols, m, &one, K16, CUDA_R_16F, D, P, CUDA_R_16F, m, &zero, O2,
                                CUDA_R_32F, LAT, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                CK(cudaStreamSynchronize(s));
                std::vector<float> o((size_t) T * H * LAT), o2(o.size()), o3(o.size());
                CK(cudaMemcpy(o.data(), O, o.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(o2.data(), O2, o2.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(o3.data(), O3, o3.size() * 4, cudaMemcpyDeviceToHost));
                double e1 = 0, e2 = 0, e3 = 0, mag = 0;
                int wt = -1, wh = -1;
                for (size_t i = 0; i < o.size(); ++i) {
                    mag = std::max(mag, std::fabs(ref[i]));
                    const double d1 = std::fabs(o[i] - ref[i]);
                    if (!(d1 <= e1)) { e1 = d1; wt = (int) (i / ((size_t) H * LAT)); wh = (int) (i / LAT % H); }
                    e2 = std::max(e2, std::fabs((double) o2[i] - ref[i]));
                    e3 = std::max(e3, std::fabs((double) o3[i] - ref[i]));
                }
                const bool ok = e1 <= 0.02 * mag + 1e-4, ok2 = e2 <= 0.02 * mag + 1e-4, ok3 = e3 <= 0.02 * mag + 1e-4;
                fails += !ok + !ok2 + !ok3;
                std::printf("attn %s T=%2d n_kv=%5d  |ref|max %.4f  kernel err %.2e %s (worst t%d h%d: %.5f vs %.5f)  gemm err %.2e %s  tc err %.2e %s\n",
                            kv_mode == m4k::KV_F16 ? "f16 " : "int8", T, n_kv, mag, e1, ok ? "ok  " : "FAIL", wt, wh,
                            wt >= 0 ? o[((size_t) wt * H + wh) * LAT] : 0.f, wt >= 0 ? (float) ref[((size_t) wt * H + wh) * LAT] : 0.f,
                            e2, ok2 ? "ok  " : "FAIL", e3, ok3 ? "ok" : "FAIL");
                for (void* p : {cache, (void*) scl, (void*) dQ, (void*) part, (void*) O, (void*) O2, (void*) O3, (void*) S, (void*) dpos,
                                (void*) K16, (void*) Q16, (void*) P})
                    CK(cudaFree(p));
            }
        }
    }
    // speed at long contexts (int8 cache, random rows): both decode kernels, per layer and per token (36 layers)
    for (int n_kv : {4096, 16384, 32768, 65536}) {
        for (int T : {1, 4}) {
            const int pos0 = n_kv - T;
            std::vector<int8_t> c8((size_t) n_kv * D);
            for (auto& v : c8) v = (int8_t) ((int) (rng() % 255) - 127);
            std::vector<float> sc((size_t) n_kv * 2, 0.0004f), Q((size_t) T * H * D);
            for (int r = 0; r < n_kv; ++r) sc[(size_t) r * 2 + 1] = 0.02f;
            for (auto& v : Q) v = nd(rng) * 2.f;
            std::vector<int32_t> pos((size_t) T);
            for (int t = 0; t < T; ++t) pos[(size_t) t] = pos0 + t;
            void* cache = nullptr;
            float *scl = nullptr, *dQ = nullptr, *part = nullptr, *O1 = nullptr, *O2 = nullptr;
            int32_t* dpos = nullptr;
            CK(cudaMalloc(&cache, c8.size()));
            CK(cudaMemcpy(cache, c8.data(), c8.size(), cudaMemcpyHostToDevice));
            CK(cudaMalloc(&scl, sc.size() * 4));
            CK(cudaMemcpy(scl, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dQ, Q.size() * 4));
            CK(cudaMemcpy(dQ, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dpos, (size_t) T * 4));
            CK(cudaMemcpy(dpos, pos.data(), (size_t) T * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&part, m4k::mla_decode_part_floats(T, H) * 4));
            CK(cudaMalloc(&O1, (size_t) T * H * LAT * 4));
            CK(cudaMalloc(&O2, (size_t) T * H * LAT * 4));
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            float ms[2] = {0, 0};
            for (int impl : {m4k::ATTN_SIMT, m4k::ATTN_TC}) {
                float* O = impl == m4k::ATTN_TC ? O2 : O1;
                m4k::mla_attn_decode(dQ, cache, scl, m4k::KV_I8, dpos, T, H, scale, part, O, s, impl);   // warm-up
                CK(cudaEventRecord(e0, s));
                for (int r = 0; r < 20; ++r) m4k::mla_attn_decode(dQ, cache, scl, m4k::KV_I8, dpos, T, H, scale, part, O, s, impl);
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms[impl == m4k::ATTN_TC], e0, e1));
            }
            std::vector<float> a((size_t) T * H * LAT), b(a.size());
            CK(cudaMemcpy(a.data(), O1, a.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), O2, b.size() * 4, cudaMemcpyDeviceToHost));
            double d = 0, mag = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                d = std::max(d, (double) std::fabs(a[i] - b[i]));
                mag = std::max(mag, (double) std::fabs(a[i]));
            }
            std::printf("speed n_kv=%5d T=%d: kernel %.3f ms/layer (%.1f ms/token), tc %.3f ms/layer (%.1f ms/token), "
                        "x%.1f, max diff %.2e of %.3f\n", n_kv, T, ms[0] / 20, ms[0] / 20 * 36, ms[1] / 20, ms[1] / 20 * 36,
                        ms[0] / std::max(1e-6f, ms[1]), d, mag);
            CK(cudaEventDestroy(e0));
            CK(cudaEventDestroy(e1));
            for (void* p : {cache, (void*) scl, (void*) dQ, (void*) part, (void*) O1, (void*) O2, (void*) dpos}) CK(cudaFree(p));
        }
    }
    std::printf("selftest-attn: %s (%d failed)\n", fails ? "FAILED" : "passed", fails);
    return fails ? 1 : 0;
}

}  // namespace

int run(int argc, char** argv) {
    const Options o = parse(argc, argv);
    if (o.mode == "selftest-attn") return selftest_attn(o.gpu);
    Engine eng(o);
    eng.load();
    if (o.mode == "serve") return serve(eng);
    if (o.mode == "eagle-acc") {
        std::ifstream f(o.ids_file);
        std::vector<int32_t> ids;
        for (int32_t v; f >> v;) ids.push_back(v);
        logf_("EAGLE accuracy over %zu tokens", ids.size());
        eng.eagle_acc(ids);
        return 0;
    }
    if (o.mode == "ppl") {
        std::ifstream f(o.ids_file);
        std::vector<int32_t> ids;
        for (int32_t v; f >> v;) ids.push_back(v);
        const int chunk = o.ppl_ctx;
        if (chunk + kMaxWin > o.ctx) fail("--ppl-ctx needs --ctx >= ppl-ctx + 16");
        logf_("perplexity over %zu tokens, chunks of %d", ids.size(), chunk);
        const double p = eng.perplexity(ids, chunk);
        std::printf("PPL %.4f\n", p);
        eng.save_heat();
        return 0;
    }
    if (o.mode == "dump") {
        std::vector<int32_t> ids;
        for (size_t a = 0; a < o.dump_ids.size();) {
            const size_t b = o.dump_ids.find_first_of(", ", a);
            const std::string t = o.dump_ids.substr(a, b == std::string::npos ? std::string::npos : b - a);
            if (!t.empty()) ids.push_back(std::atoi(t.c_str()));
            if (b == std::string::npos) break;
            a = b + 1;
        }
        eng.dump(ids);
        return 0;
    }
    if (o.mode == "bench") {
        std::mt19937 rng(1234);
        std::vector<int32_t> prompt((size_t) o.bench_pp);
        for (auto& t : prompt) t = 1000 + (int32_t) (rng() % 30000);
        Engine::GenParams gp;
        gp.max_new = o.bench_tg;
        gp.spec = 0;
        const auto st = eng.generate(prompt, gp, [](const std::vector<int32_t>&) { return true; }, nullptr);
        std::printf("bench: prompt %d tokens in %.0f ms (%.1f tok/s), generated %d tokens in %.0f ms (%.2f tok/s)\n",
                    st.n_prompt, st.prompt_ms, st.n_prompt / (st.prompt_ms / 1000.0), st.n_gen, st.gen_ms,
                    st.n_gen / (st.gen_ms / 1000.0));
        eng.save_heat();
        return 0;
    }
    fail("unknown mode " + o.mode);
}

}  // namespace m4
