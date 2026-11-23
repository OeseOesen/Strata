// engine/src/mistral4/m4_numa.hpp - strata-m4: NUMA topology, node-bound memory and the near-data CPU expert pool.
//
// The machine this engine is written for: two sockets (Skylake-SP, 6 DDR4 channels each, 96 GiB per node), one GPU on
// the PCIe root of ONE socket ("the GPU node"), and a slow UPI link between the sockets.  The rules that follow from it:
//
//   * Expert weights never cross UPI during decode.  Every expert that is not in VRAM lives on exactly one node (tiered
//     policy) or is split by rows over both nodes (split policy), and it is computed ONLY by worker threads of the node
//     that holds its bytes.  What crosses UPI is the activation (4096 floats) and the expert's partial result.
//   * The tiered policy (default, the user's spec): warm experts on the GPU node, the coldest ones (and the n-gram
//     table) on the other node; each node's workers compute the experts in their own memory.
//   * The split policy (opt-in, --numa-policy split): every CPU expert's gate/up rows and down rows are halved, one half per
//     node, so both memory controllers stream for every expert.  Measure both: it doubles the bandwidth on paper and
//     costs one cross-socket barrier per phase.
//
// Memory is bound with mbind(MPOL_BIND) through the raw syscall (no libnuma dependency) and first-touched by a thread
// pinned to that node, then advised for transparent huge pages.  Pinned for DMA (cudaHostRegister) by the caller.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "strata/kernels/cpu/native_expert.hpp"

namespace m4 {

// ------------------------------------------------------------------------------------------------ topology
struct NumaNode {
    int id = 0;
    std::vector<int> cores;       // first hardware thread of every physical core of this node, ascending
    std::vector<int> siblings;    // the other hardware threads
    size_t mem_total = 0, mem_free = 0;
};

struct Topology {
    std::vector<NumaNode> nodes;  // only nodes with CPUs
    int node_of_cpu(int cpu) const;
    const NumaNode* node(int id) const;
    static Topology detect();     // /sys/devices/system/node; one synthetic node when there is none
    std::string describe() const;
};

/// The NUMA node of a PCI device ("0000:3b:00.0"), -1 when unknown.
int pci_numa_node(const std::string& bus_id);

// ------------------------------------------------------------------------------------------------ memory
/// `bytes` of anonymous memory bound to `node` (MPOL_BIND), THP-advised and first-touched by a thread on that node.
/// Returns nullptr on failure (err says why).  node < 0: no binding.
void* node_alloc(size_t bytes, int node, const Topology& topo, std::string& err);
void node_free(void* p, size_t bytes);
/// Pins the calling thread to the given CPUs.
bool pin_thread(const std::vector<int>& cpus);
/// Pins the calling thread to every hardware thread of `node`.
bool pin_to_node(const Topology& topo, int node);

// ------------------------------------------------------------------------------------------------ the pool
/// One unit of CPU expert work: rows [r0, r1) of gate/up (phase 0) or of down (phase 2) of one expert part, for the
/// `nt` tokens routed to it.  Phase 1 quantizes h (n_ff values) for an entry.
enum class Phase : int { GateUp = 0, QuantH = 1, Down = 2 };

struct PartRef {
    const uint8_t* blob = nullptr;                   // this node's part: [gate rows | up rows | down rows]
    const strata::kernels::cpu::NativeFmt* fmt = nullptr;   // the part's layout (local row counts and offsets)
    int gu_base = 0, gu_rows = 0;                    // which global gate/up rows the part holds
    int d_base = 0, d_rows = 0;                      // which global down rows
};

/// A CPU expert: one part per node that holds some of it.
struct CpuExpert {
    PartRef part[2];
    int n_parts = 0;
    int node[2] = {-1, -1};
};

/// The per-call description the host fills: entries (an expert and the tokens routed to it) computed on the CPU.
/// Entry i's tokens are tok[off .. off+nt) (rows of `x`) and its outputs go to out[off .. off+nt) (n_embd floats each,
/// unweighted: the router weight is applied by the GPU's combine).
struct CpuCall {
    struct Entry {
        const CpuExpert* ex = nullptr;
        int off = 0, nt = 0;
    };
    std::vector<Entry> entries;
    std::vector<int> tok;
    std::vector<float*> out;
    const float* x = nullptr;    // [n_tok][n_embd], the normalized MoE input
    int n_tok = 0;
    const strata::kernels::cpu::NativeFmt* fmt = nullptr;   // the full expert layout (activation formats, n_ff)
    void clear() {
        entries.clear();
        tok.clear();
        out.clear();
    }
};

class NumaPool {
public:
    struct Options {
        int workers_per_node = -1;   // -1: every physical core of the node (minus the host's on the GPU node)
        int host_cpu = -1;           // the core the calling (host) thread sits on; skipped for workers
        std::vector<int> nodes;      // nodes that run workers (default: every node with CPUs)
        int rows_gu = 64, rows_down = 128;   // work-item granularity
    };
    /// max_tokens: most rows of x in one call; max_routes: most (entry, token) pairs in one call.
    NumaPool(const Topology& topo, const Options& o, int64_t n_embd, int64_t n_ff, int max_tokens, int max_routes);
    ~NumaPool();
    NumaPool(const NumaPool&) = delete;
    NumaPool& operator=(const NumaPool&) = delete;

    /// Runs `call` on the workers and returns when every output is written.  Blocking; the host thread does not
    /// compute (it is the one that talks to the GPU), it spins until done.
    void run(const CpuCall& call);
    /// Starts `call` and returns at once; wait() blocks until it is done.  One call in flight.
    void start(const CpuCall& call);
    void wait();
    bool busy() const { return busy_; }

    int workers() const { return (int) threads_.size(); }
    std::string describe() const;
    double last_ms() const { return last_ms_; }

private:
    struct Item { int entry; int part; int r0, r1; };   // phase 1: part = the route index
    struct NodeQueue {
        std::vector<Item> items[3];
        std::atomic<int> next[3];
        int node = -1;
    };
    void worker(int idx, int node_slot, std::vector<int> cpus);
    void build(const CpuCall& call);
    void do_phase(int node_slot, int phase);
    void barrier(int node_slot);

    const Topology& topo_;
    Options opt_;
    int64_t n_embd_, n_ff_;
    int max_tokens_, max_routes_;
    std::vector<std::thread> threads_;
    std::vector<int> thread_node_slot_;
    std::vector<NodeQueue*> queues_;     // one per node that has workers
    std::vector<int> node_ids_;          // queue slot -> node id
    // per call (built by the host before the epoch bumps)
    const CpuCall* call_ = nullptr;
    std::vector<uint8_t> act_;           // quantized activations [n_tok][act_bytes]
    std::vector<float> ff_;              // [route][n_ff]
    std::vector<uint8_t> hq_;            // [route][h_bytes]
    // synchronisation
    std::atomic<uint32_t> epoch_{0};
    // barriers: the whole pool (a call with split experts: both nodes read h, which both nodes wrote) or, when every
    // expert of the call lives on one node, one barrier per node - no cache line crosses the socket link per phase
    struct alignas(64) Bar {
        std::atomic<int> arrived{0};
        std::atomic<uint32_t> gen{0};
        int members = 0;
    };
    Bar bar_all_;
    std::vector<Bar*> bar_node_;
    bool global_bar_ = true;
    std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};
    bool busy_ = false;
    double last_ms_ = 0;
    int64_t t_start_ns_ = 0;
};

}  // namespace m4
