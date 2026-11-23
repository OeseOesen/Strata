// engine/src/mistral4/m4_numa.cpp - see m4_numa.hpp.
#include "m4_numa.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define M4_PAUSE() _mm_pause()
#else
#define M4_PAUSE() ((void) 0)
#endif

namespace m4 {

namespace C = strata::kernels::cpu;

// ------------------------------------------------------------------------------------------------ topology
namespace {
std::string read_file(const std::string& p) {
    std::ifstream f(p);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
// "0-3,8,10-11" -> {0,1,2,3,8,10,11}
std::vector<int> parse_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        while (!part.empty() && (part.back() == '\n' || part.back() == ' ')) part.pop_back();
        if (part.empty()) continue;
        const auto dash = part.find('-');
        if (dash == std::string::npos) {
            out.push_back(std::stoi(part));
        } else {
            const int a = std::stoi(part.substr(0, dash)), b = std::stoi(part.substr(dash + 1));
            for (int i = a; i <= b; ++i) out.push_back(i);
        }
    }
    return out;
}
size_t meminfo_kb(const std::string& text, const char* key) {
    const auto p = text.find(key);
    if (p == std::string::npos) return 0;
    size_t i = p + std::strlen(key);
    while (i < text.size() && (text[i] == ' ' || text[i] == ':')) ++i;
    return (size_t) std::strtoull(text.c_str() + i, nullptr, 10);
}
int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int Topology::node_of_cpu(int cpu) const {
    for (const auto& n : nodes) {
        if (std::find(n.cores.begin(), n.cores.end(), cpu) != n.cores.end()) return n.id;
        if (std::find(n.siblings.begin(), n.siblings.end(), cpu) != n.siblings.end()) return n.id;
    }
    return -1;
}

const NumaNode* Topology::node(int id) const {
    for (const auto& n : nodes)
        if (n.id == id) return &n;
    return nullptr;
}

Topology Topology::detect() {
    Topology t;
#ifdef __linux__
    const std::string root = "/sys/devices/system/node";
    const std::vector<int> online = parse_list(read_file(root + "/online"));
    for (int id : online) {
        const std::string dir = root + "/node" + std::to_string(id);
        const std::vector<int> cpus = parse_list(read_file(dir + "/cpulist"));
        if (cpus.empty()) continue;   // memory-only node
        NumaNode n;
        n.id = id;
        std::set<int> seen;
        for (int c : cpus) {
            if (seen.count(c)) continue;
            const std::vector<int> sib =
                parse_list(read_file("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
            const int first = sib.empty() ? c : *std::min_element(sib.begin(), sib.end());
            for (int s : sib) seen.insert(s);
            seen.insert(c);
            if (std::find(n.cores.begin(), n.cores.end(), first) == n.cores.end()) n.cores.push_back(first);
            for (int s : sib)
                if (s != first && std::find(cpus.begin(), cpus.end(), s) != cpus.end()) n.siblings.push_back(s);
        }
        std::sort(n.cores.begin(), n.cores.end());
        const std::string mi = read_file(dir + "/meminfo");
        n.mem_total = meminfo_kb(mi, "MemTotal") * 1024;
        // what the node can give: free pages plus the page cache (the model file just read lives there and is
        // reclaimable), minus shared memory that only looks like cache
        const size_t fp = meminfo_kb(mi, "FilePages"), sh = meminfo_kb(mi, "Shmem");
        n.mem_free = (meminfo_kb(mi, "MemFree") + (fp > sh ? fp - sh : 0)) * 1024;
        t.nodes.push_back(n);
    }
#endif
    if (t.nodes.empty()) {
        NumaNode n;
        const unsigned hc = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 0; i < hc; ++i) n.cores.push_back((int) i);
        t.nodes.push_back(n);
    }
    return t;
}

std::string Topology::describe() const {
    std::ostringstream o;
    for (const auto& n : nodes)
        o << "node " << n.id << ": " << n.cores.size() << " cores (+" << n.siblings.size() << " SMT), "
          << (n.mem_total >> 30) << " GiB (" << (n.mem_free >> 30) << " free); ";
    return o.str();
}

int pci_numa_node(const std::string& bus_id) {
    std::string id = bus_id;
    for (auto& c : id) c = (char) std::tolower((unsigned char) c);
    // cudaDeviceGetPCIBusId gives "0000:3B:00.0"; sysfs wants lower case, and some drivers report an 8-digit domain
    if (id.size() > 12 && id[8] == ':') id = id.substr(4);
    const std::string s = read_file("/sys/bus/pci/devices/" + id + "/numa_node");
    if (s.empty()) return -1;
    return std::atoi(s.c_str());
}

// ------------------------------------------------------------------------------------------------ memory
bool pin_thread(const std::vector<int>& cpus) {
#ifdef __linux__
    if (cpus.empty()) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) CPU_SET(c, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void) cpus;
    return false;
#endif
}

bool pin_to_node(const Topology& topo, int node) {
    const NumaNode* n = topo.node(node);
    if (!n) return false;
    std::vector<int> all = n->cores;
    all.insert(all.end(), n->siblings.begin(), n->siblings.end());
    return pin_thread(all);
}

void* node_alloc(size_t bytes, int node, const Topology& topo, std::string& err) {
#ifdef __linux__
    if (bytes == 0) return nullptr;
    const size_t page = (size_t) 2 << 20;
    const size_t len = (bytes + page - 1) / page * page;
    void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        err = "mmap of " + std::to_string(len >> 20) + " MiB failed";
        return nullptr;
    }
    madvise(p, len, MADV_HUGEPAGE);
    if (node >= 0 && topo.nodes.size() > 1) {
        unsigned long mask[16] = {};
        mask[node / 64] |= 1ul << (node % 64);
        constexpr int kMpolBind = 2;
        constexpr unsigned kMfStrict = 1u, kMfMove = 2u;
        if (syscall(SYS_mbind, p, len, kMpolBind, mask, 16 * 64, kMfStrict | kMfMove) != 0) {
            err = "mbind to node " + std::to_string(node) + " failed (" + std::strerror(errno) + ")";
            munmap(p, len);
            return nullptr;
        }
    }
    // first touch from threads on that node, in parallel (a 50 GiB arena touched by one thread takes ~20 s)
    const NumaNode* n = topo.node(node);
    const int nt = n ? std::max(1, std::min<int>((int) n->cores.size(), 16)) : 8;
    std::vector<std::thread> th;
    const size_t chunk = (len / nt + page - 1) / page * page;
    for (int i = 0; i < nt; ++i) {
        th.emplace_back([&, i] {
            if (node >= 0) pin_to_node(topo, node);
            const size_t a = (size_t) i * chunk, b = std::min(len, a + chunk);
            for (size_t o = a; o < b; o += 4096) static_cast<volatile uint8_t*>(p)[o] = 0;
        });
    }
    for (auto& t : th) t.join();
    return p;
#else
    (void) node;
    (void) topo;
    void* p = std::malloc(bytes);
    if (!p) err = "malloc failed";
    return p;
#endif
}

void node_free(void* p, size_t bytes) {
    if (!p) return;
#ifdef __linux__
    const size_t page = (size_t) 2 << 20;
    munmap(p, (bytes + page - 1) / page * page);
#else
    (void) bytes;
    std::free(p);
#endif
}

// ------------------------------------------------------------------------------------------------ the pool
NumaPool::NumaPool(const Topology& topo, const Options& o, int64_t n_embd, int64_t n_ff, int max_tokens, int max_routes)
    : topo_(topo), opt_(o), n_embd_(n_embd), n_ff_(n_ff), max_tokens_(max_tokens), max_routes_(max_routes) {
    std::vector<int> nodes = o.nodes;
    if (nodes.empty())
        for (const auto& n : topo.nodes) nodes.push_back(n.id);
    act_.resize((size_t) max_tokens * C::native_act_bytes_for(n_embd) + 64);
    ff_.resize((size_t) max_routes * (size_t) n_ff);
    hq_.resize((size_t) max_routes * C::native_act_bytes_for(n_ff) + 64);
    int idx = 0;
    for (int nid : nodes) {
        const NumaNode* n = topo.node(nid);
        if (!n) continue;
        std::vector<int> cores;
        for (int c : n->cores)
            if (c != o.host_cpu) cores.push_back(c);
        int want = o.workers_per_node < 0 ? (int) cores.size() : std::min<int>(o.workers_per_node, (int) cores.size());
        if (want <= 0) continue;
        auto* q = new NodeQueue();
        for (auto& a : q->next) a.store(0);
        q->node = nid;
        const int slot = (int) queues_.size();
        queues_.push_back(q);
        node_ids_.push_back(nid);
        bar_node_.push_back(new Bar());
        bar_node_.back()->members = want;
        for (int i = 0; i < want; ++i) {
            thread_node_slot_.push_back(slot);
            threads_.emplace_back(&NumaPool::worker, this, idx++, slot, std::vector<int>{cores[(size_t) i]});
        }
    }
    bar_all_.members = (int) threads_.size();
}

NumaPool::~NumaPool() {
    stop_.store(true);
    epoch_.fetch_add(1, std::memory_order_release);
    for (auto& t : threads_) t.join();
    for (auto* q : queues_) delete q;
    for (auto* b : bar_node_) delete b;
}

std::string NumaPool::describe() const {
    std::ostringstream o;
    for (size_t s = 0; s < queues_.size(); ++s) {
        const int n = (int) std::count(thread_node_slot_.begin(), thread_node_slot_.end(), (int) s);
        o << (s ? ", " : "") << n << " workers on node " << node_ids_[s];
    }
    return o.str();
}

void NumaPool::build(const CpuCall& call) {
    call_ = &call;
    for (auto* q : queues_)
        for (int ph = 0; ph < 3; ++ph) {
            q->items[ph].clear();
            q->next[ph].store(0, std::memory_order_relaxed);
        }
    auto slot_of = [&](int node) -> int {
        for (size_t s = 0; s < node_ids_.size(); ++s)
            if (node_ids_[s] == node) return (int) s;
        return 0;   // a node without workers: the first queue computes it (its bytes then cross the link)
    };
    // the activations: quantized once here (<= 16 x 4096 values, a few microseconds)
    const size_t ab = call.fmt->act_bytes;
    if ((size_t) call.n_tok * ab > act_.size()) act_.resize((size_t) call.n_tok * ab + 64);
    for (int t = 0; t < call.n_tok; ++t) C::native_quant_act(*call.fmt, call.x + (size_t) t * n_embd_, act_.data() + (size_t) t * ab);
    global_bar_ = false;
    for (int e = 0; e < (int) call.entries.size(); ++e) {
        const CpuCall::Entry& en = call.entries[(size_t) e];
        if (en.ex->n_parts > 1) global_bar_ = true;
        for (int p = 0; p < en.ex->n_parts; ++p) {
            const PartRef& pr = en.ex->part[p];
            NodeQueue* q = queues_[(size_t) slot_of(en.ex->node[p])];
            for (int r = 0; r < pr.gu_rows; r += opt_.rows_gu) q->items[0].push_back({e, p, r, std::min(pr.gu_rows, r + opt_.rows_gu)});
            for (int r = 0; r < pr.d_rows; r += opt_.rows_down) q->items[2].push_back({e, p, r, std::min(pr.d_rows, r + opt_.rows_down)});
        }
        // h is quantized once per route, by a node that holds (part of) the expert
        NodeQueue* q = queues_[(size_t) slot_of(en.ex->node[0])];
        for (int i = 0; i < en.nt; ++i) q->items[1].push_back({e, en.off + i, 0, 0});
    }
}

void NumaPool::do_phase(int slot, int phase) {
    NodeQueue* q = queues_[(size_t) slot];
    const auto& items = q->items[phase];
    const CpuCall& call = *call_;
    const size_t ab = call.fmt->act_bytes, hb = call.fmt->h_bytes;
    for (;;) {
        const int i = q->next[phase].fetch_add(1, std::memory_order_relaxed);
        if (i >= (int) items.size()) break;
        const Item& it = items[(size_t) i];
        if (phase == 1) {   // it.part is the route
            C::native_quant_h(*call.fmt, ff_.data() + (size_t) it.part * n_ff_, hq_.data() + (size_t) it.part * hb);
            continue;
        }
        const CpuCall::Entry& en = call.entries[(size_t) it.entry];
        const PartRef& pr = en.ex->part[it.part];
        // the native kernels take at most 8 tokens per call: larger groups go in pieces
        for (int t0 = 0; t0 < en.nt; t0 += 8) {
            const int nt = std::min(8, en.nt - t0);
            const void* in[8];
            float* out[8];
            for (int t = 0; t < nt; ++t) {
                const int route = en.off + t0 + t;
                if (phase == 0) {
                    in[t] = act_.data() + (size_t) call.tok[(size_t) route] * ab;
                    out[t] = ff_.data() + (size_t) route * n_ff_ + pr.gu_base;
                } else {
                    in[t] = hq_.data() + (size_t) route * hb;
                    out[t] = call.out[(size_t) route] + pr.d_base;
                }
            }
            if (phase == 0) C::native_gu_rows(*pr.fmt, pr.blob, in, nt, out, it.r0, it.r1);
            else C::native_down_rows(*pr.fmt, pr.blob, in, nt, out, it.r0, it.r1);
        }
    }
}

void NumaPool::barrier(int slot) {
    Bar& b = global_bar_ ? bar_all_ : *bar_node_[(size_t) slot];
    const uint32_t gen = b.gen.load(std::memory_order_acquire);
    if (b.arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == b.members) {
        b.arrived.store(0, std::memory_order_relaxed);
        b.gen.store(gen + 1, std::memory_order_release);
    } else {
        while (b.gen.load(std::memory_order_acquire) == gen) M4_PAUSE();
    }
}

void NumaPool::worker(int idx, int slot, std::vector<int> cpus) {
    (void) idx;
    pin_thread(cpus);
    uint32_t seen = 0;
    for (;;) {
        // wait for a new epoch: spin ~2 ms (a decode step calls every ~100 us), then sleep in steps that grow with
        // the idle time (50 us up to 2 ms): an idle server must not burn 36 cores, and a new request pays at most
        // one sleep once
        const int64_t spin_start = now_ns();
        uint32_t e;
        while ((e = epoch_.load(std::memory_order_acquire)) == seen) {
            const int64_t idle = now_ns() - spin_start;
            if (idle < 2'000'000) {
                M4_PAUSE();
                continue;
            }
            const int64_t us = idle < 50'000'000 ? 50 : idle < 500'000'000 ? 250 : 2000;
            std::this_thread::sleep_for(std::chrono::microseconds(us));
        }
        seen = e;
        if (stop_.load()) return;
        do_phase(slot, 0);
        barrier(slot);
        do_phase(slot, 1);
        barrier(slot);
        do_phase(slot, 2);
        done_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void NumaPool::start(const CpuCall& call) {
    t_start_ns_ = now_ns();
    if (threads_.empty() || call.entries.empty()) {
        busy_ = false;
        last_ms_ = 0;
        return;
    }
    if ((int) call.tok.size() > max_routes_ || call.n_tok > max_tokens_) {
        std::fprintf(stderr, "strata-m4: a CPU call of %zu routes / %d tokens, the pool was sized for %d / %d\n",
                     call.tok.size(), call.n_tok, max_routes_, max_tokens_);
        std::abort();
    }
    build(call);
    done_.store(0, std::memory_order_relaxed);
    busy_ = true;
    epoch_.fetch_add(1, std::memory_order_release);
}

void NumaPool::wait() {
    if (!busy_) return;
    const int n = (int) threads_.size();
    while (done_.load(std::memory_order_acquire) < n) M4_PAUSE();
    busy_ = false;
    last_ms_ = (double) (now_ns() - t_start_ns_) / 1e6;
}

void NumaPool::run(const CpuCall& call) {
    start(call);
    wait();
}

}  // namespace m4
