// engine/src/mistral4/m4_ngram.hpp - strata-m4's n-gram drafter.
//
// Two sources, tried in order:
//   1. Prompt / history lookup: the longest suffix (up to 8 tokens, at least `min_match`) of the current sequence
//      that occurred earlier in it; the tokens after that occurrence are the draft.  Code edits, tool-call JSON
//      and quoted documents make this the strongest source for agent workloads.
//   2. The global n-gram table: hashed 3-gram -> (next token, count), learned from every accepted token of every
//      request and kept for the life of the server.  It lives in memory the caller hands in (strata-m4 binds it to
//      the NUMA node that is NOT the GPU's: it is looked up once per draft token, latency-tolerant, and must not take
//      bandwidth from the expert arena).
//
// A draft is only a proposal: every token is verified by the model in the next window.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace m4 {

class NgramDrafter {
public:
    struct Slot {
        uint64_t key;     // 0 = empty
        int32_t token;
        uint32_t count;
    };

    /// `mem` / `bytes`: the table's memory (any node); nullptr disables the table, lookups still work.
    void init(void* mem, size_t bytes, int min_match = 3) {
        table_ = static_cast<Slot*>(mem);
        n_slots_ = mem ? bytes / sizeof(Slot) : 0;
        // power of two for masking
        size_t p = 1;
        while (p * 2 <= n_slots_) p *= 2;
        n_slots_ = mem ? p : 0;
        if (table_) std::memset(table_, 0, n_slots_ * sizeof(Slot));
        min_match_ = min_match;
    }

    /// Learn the transitions of `seq` from position `from` on (the tokens that became final).
    void learn(const std::vector<int32_t>& seq, size_t from) {
        if (!table_) return;
        for (size_t i = std::max<size_t>(from, 3); i < seq.size(); ++i) {
            const uint64_t k = key3(seq[i - 3], seq[i - 2], seq[i - 1]);
            Slot& s = table_[k & (n_slots_ - 1)];
            if (s.key == k) {
                if (s.token == seq[i]) {
                    if (s.count < 0xffffffffu) ++s.count;
                } else if (s.count <= 1) {
                    s.token = seq[i];
                    s.count = 1;
                } else {
                    --s.count;   // a competing continuation wears the old one down
                }
            } else if (s.key == 0 || s.count <= 1) {
                s = Slot{k, seq[i], 1};
            } else {
                --s.count;
            }
        }
    }

    /// Up to `max_draft` tokens that may follow `seq`.
    int draft(const std::vector<int32_t>& seq, int max_draft, std::vector<int32_t>& out, int& source) const {
        out.clear();
        source = 0;
        if (max_draft <= 0 || seq.size() < 2) return 0;
        // 1. lookup in the sequence itself, longest match first
        const int n = (int) seq.size();
        for (int m = std::min(8, n - 1); m >= min_match_; --m) {
            const int32_t* suf = seq.data() + n - m;
            // the most recent earlier occurrence
            for (int s = n - m - 1; s >= 0; --s) {
                if (seq[(size_t) s] != suf[0] || s + m > n - 1) continue;
                if (std::memcmp(seq.data() + s, suf, (size_t) m * sizeof(int32_t)) != 0) continue;
                const int start = s + m;
                const int take = std::min(max_draft, n - start);
                if (take <= 0) continue;
                out.assign(seq.begin() + start, seq.begin() + start + take);
                source = 1;
                return (int) out.size();
            }
        }
        // 2. the global table, chained
        if (!table_ || n < 3) return 0;
        int32_t a = seq[(size_t) n - 3], b = seq[(size_t) n - 2], c = seq[(size_t) n - 1];
        for (int i = 0; i < max_draft; ++i) {
            const uint64_t k = key3(a, b, c);
            const Slot& s = table_[k & (n_slots_ - 1)];
            if (s.key != k || s.count < 2) break;
            out.push_back(s.token);
            a = b;
            b = c;
            c = s.token;
        }
        if (!out.empty()) source = 2;
        return (int) out.size();
    }

    size_t slots() const { return n_slots_; }

private:
    static uint64_t key3(int32_t a, int32_t b, int32_t c) {
        uint64_t h = 0x9e3779b97f4a7c15ull;
        for (uint64_t v : {(uint64_t) (uint32_t) a, (uint64_t) (uint32_t) b, (uint64_t) (uint32_t) c}) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            h *= 0xff51afd7ed558ccdull;
        }
        return h | 1;   // never 0 (the empty mark)
    }
    Slot* table_ = nullptr;
    size_t n_slots_ = 0;
    int min_match_ = 3;
};

}  // namespace m4
