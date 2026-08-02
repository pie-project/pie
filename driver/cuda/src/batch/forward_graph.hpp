#pragma once

// `PIE_PREFILL_GRAPH=1` (default off) lets a wave carrying a prefill reach the
// graph cache instead of dropping to the eager path with all of its decode
// lanes. Declared here rather than in `batch/forward.hpp` because the float
// workspace has to be sized for graph-mode planning under the same flag, and
// `batch/workspace.cu` must not pull in the whole batch engine to ask.
// Defined in `batch/forward.cpp`.
namespace pie_cuda_driver {
bool prefill_graph_enabled();
}  // namespace pie_cuda_driver

// CUDA-graph cache for the decode forward body.
//
// Why: every fire_batch in a steady decode workload issues the same ~420
// kernel launches per layer × 28 layers + embed/lm_head/etc. Per-launch
// overhead dominates at small batch sizes. Capturing the launch sequence
// once into a `cudaGraphExec_t` and replaying it on subsequent fires of
// the same shape collapses N CPU-side forward launch invocations into a
// single `cudaGraphLaunch`. PTIR publication remains outside the graph because
// bound program/channel layouts vary independently from the forward shape.
//
// Constraints (correctness — gibberish if violated):
//   1. **Pointer stability.** All kernel arguments (`pi.tokens`, `ws.*`,
//      `kv_cache.k(L)`, etc.) must live at the same address every replay.
//      Persistent inputs (PersistentInputs) and the workspace allocations
//      satisfy this.
//   2. **No host-side work in the captured region.** Every host loop /
//      `std::vector` allocation / cuBLAS algo search inside capture
//      happens *only at capture time*; replay just re-issues the recorded
//      kernels. flashinfer's `DecodePlan` violates this — it allocates
//      host vectors, runs work estimation, fills `page_locked_int`. So we
//      hoist the plan out (`plan_attention_flashinfer_decode_bf16` runs
//      *before* graph launch); only the dispatch is captured.
//   3. **Stable kernel sequence.** The graph encodes the exact kernel
//      list of one fire shape. Different `(R, num_pages, …)` shapes need
//      different graphs, hence the bucket cache.
//
// Used only when the executor decides the fire is "pure decode"
// (every request has qo_len == 1) and the `--cuda-graphs` flag is on.

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace pie_cuda_driver {

// Bucket key for the graph cache. Two fires that match on this key share
// a captured graph. We choose the inputs that flashinfer's plan / our
// kernel-launch sequence depend on; per-token contents (token IDs,
// positions, KV page indices) flow through persistent buffers and don't
// affect graph topology.
//
// PTIR program identity and per-fire geometry remain device data or execute
// outside this graph. They therefore cannot expand the forward cache key.
struct ForwardGraphKey {
    int num_requests;
    int num_tokens;
    std::uint32_t variant = 0;

    bool operator==(const ForwardGraphKey& o) const noexcept {
        return num_requests == o.num_requests &&
               num_tokens == o.num_tokens &&
               variant == o.variant;
    }
};

// vLLM-style decode graph lattice. Runtime batches are padded upward to
// one of these request counts before graph capture/replay:
//   1, 2, 4, then multiples of 8 up to 256, then multiples of 16.
// The planner's max request count is also a legal bucket even when it is
// off-lattice, matching vLLM's "append max if it fits" behavior.
constexpr int forward_graph_request_bucket(int requests,
                                           int max_requests) noexcept {
    if (requests <= 0 || max_requests <= 0 || requests > max_requests) {
        return 0;
    }

    int bucket = requests;
    if (requests <= 1) {
        bucket = 1;
    } else if (requests <= 2) {
        bucket = 2;
    } else if (requests <= 4) {
        bucket = 4;
    } else if (requests < 256) {
        bucket = ((requests + 7) / 8) * 8;
    } else {
        bucket = ((requests + 15) / 16) * 16;
    }

    return bucket <= max_requests ? bucket : max_requests;
}

// Token lattice, the N-axis counterpart to the request lattice above. Only a
// prefill-carrying wave needs it: in pure decode N == R, so bucketing R
// already buckets N, and this is never consulted.
//
// Granularity is measured, not chosen. On the S cell (4096x64 @ c256) the 182
// prefill-carrying waves produce 150 distinct exact (R, N) keys -- 82% of them
// first-touch, each paying a full capture for a shape that never recurs, which
// is what `forward_graph_replay_eligible` means by "one-off shapes". Rounding
// N to a multiple of:
//
//     G=64  -> 81 keys, 55.5% reuse, 3.1% padding waste
//     G=128 -> 67 keys, 63.2% reuse, 6.9% padding waste
//     G=256 -> 56 keys, 69.2% reuse, 13.6% padding waste
//
// 128 is the finest grain whose key set still fits `kMaxEntries` beside the
// 51 upfront decode graphs (67 + 51 = 118). G=64 would need 132 entries and
// evict its own working set, spending the padding and losing the reuse that
// paid for it.
constexpr int kForwardGraphTokenGrain = 128;

constexpr int forward_graph_token_bucket(int tokens, int max_tokens) noexcept {
    if (tokens <= 0 || max_tokens <= 0 || tokens > max_tokens) return 0;
    const int bucket =
        ((tokens + kForwardGraphTokenGrain - 1) / kForwardGraphTokenGrain) *
        kForwardGraphTokenGrain;
    return bucket <= max_tokens ? bucket : max_tokens;
}

static_assert(forward_graph_token_bucket(1, 8192) == 128);
static_assert(forward_graph_token_bucket(128, 8192) == 128);
static_assert(forward_graph_token_bucket(129, 8192) == 256);
static_assert(forward_graph_token_bucket(6003, 8192) == 6016);
static_assert(forward_graph_token_bucket(8100, 8192) == 8192);
static_assert(forward_graph_token_bucket(9000, 8192) == 0);

static_assert(forward_graph_request_bucket(1, 512) == 1);
static_assert(forward_graph_request_bucket(3, 512) == 4);
static_assert(forward_graph_request_bucket(5, 512) == 8);
static_assert(forward_graph_request_bucket(255, 512) == 256);
static_assert(forward_graph_request_bucket(257, 512) == 272);
static_assert(forward_graph_request_bucket(506, 512) == 512);
static_assert(forward_graph_request_bucket(129, 130) == 130);

// Order-independent identity of the hook programs a fire carries: a
// commutative combine (sum of splitmix64 mixes) over the launch's
// `ptir_program_hashes`, plus the lane count. Two fires whose lanes carry the
// same program multiset — in any order — share it; snapkv and h2o at the same
// (R, N, variant) do not. Used to partition the per-key hook exec storage
// below so distinct program sets stop invalidating each other's captures.
// This is a cache-PARTITIONING hash only; the baked-state fingerprint each
// entry carries remains the correctness gate.
constexpr std::uint64_t hook_program_set_hash(
    const std::uint64_t* hashes, std::size_t count) noexcept {
    auto mix = [](std::uint64_t x) constexpr {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    };
    std::uint64_t h = mix(static_cast<std::uint64_t>(count));
    for (std::size_t i = 0; i < count; ++i) h += mix(hashes[i]);
    return h;
}

// Per-key bookkeeping for HOOK-carrying captures (stage 6 increment 4). A
// hook graph bakes addresses the plain body does not — stable per-occurrence
// stage buffers, sideband-arena blocks, channel-ring arrays, the host CSR a
// captured upload re-reads — so each cached exec carries the fingerprint of
// what it baked. The prepare pass recomputes the fingerprint before every
// launch; a mismatch (arena growth, stable-buffer growth, instance churn,
// data-sized grid drift) invalidates the exec and recaptures. An entry whose
// fingerprint churns on CONSECUTIVE fires — recapturing every fire costs more
// than it saves (~10 ms per capture) — is banned back to the eager body; a
// clean replay resets the churn count, so the legitimate one-recapture-per-
// new-instance cadence never accumulates into a ban.
//
// Storage is partitioned by `hook_program_set_hash`: two hook programs at one
// (R, N, variant) — snapkv and h2o alternating at R=1 was the live case —
// prepare DIFFERENT baked state, so a single exec slot per key ping-pongs
// fingerprint-mismatch recaptures until the churn ban forces both eager.
// Each program set gets its own entry (exec + fingerprint + churn counter),
// held MRU-first and capped at `kMaxProgramSets` per key; eviction destroys
// the exec, and a key whose program sets churn past `kMaxEvictions` is banned
// outright — adversarial program churn can neither grow memory nor buy a
// capture per fire. Unlike the plain lattice, hook execs live HERE, not in
// `ForwardGraphCache` (the plain-fire cache key must not fragment on program
// identity).
struct HookGraphKeyState {
    struct Entry {
        std::uint64_t program_set = 0;
        cudaGraphExec_t exec = nullptr;
        std::uint64_t fingerprint = 0;
        std::uint32_t mismatches = 0;
        bool banned = false;
    };
    static constexpr std::uint32_t kMaxMismatches = 8;
    static constexpr std::size_t kMaxProgramSets = 4;
    static constexpr std::uint32_t kMaxEvictions = 16;

    // MRU-first; size <= kMaxProgramSets.
    std::vector<Entry> entries;
    std::uint32_t evictions = 0;
    bool banned = false;

    HookGraphKeyState() = default;
    HookGraphKeyState(const HookGraphKeyState&) = delete;
    HookGraphKeyState& operator=(const HookGraphKeyState&) = delete;
    HookGraphKeyState(HookGraphKeyState&& o) noexcept
        : entries(std::move(o.entries)),
          evictions(o.evictions),
          banned(o.banned) {
        o.entries.clear();
    }
    HookGraphKeyState& operator=(HookGraphKeyState&&) = delete;
    ~HookGraphKeyState() noexcept {
        for (Entry& e : entries) {
            if (e.exec != nullptr) cudaGraphExecDestroy(e.exec);
        }
    }

    // Entry for `program_set`, moved to MRU position; nullptr if absent.
    Entry* find(std::uint64_t program_set) noexcept {
        for (std::size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].program_set == program_set) {
                if (i != 0) {
                    Entry hit = entries[i];
                    entries.erase(entries.begin() +
                                  static_cast<std::ptrdiff_t>(i));
                    entries.insert(entries.begin(), hit);
                }
                return &entries.front();
            }
        }
        return nullptr;
    }

    // Fresh MRU entry for `program_set` (caller checked find() == nullptr).
    // Evicts the LRU entry (destroying its exec) at capacity; sets `banned`
    // when evictions churn past the cap. Returns the inserted entry — valid
    // even on the banning insert, so the current fire still launches what it
    // captured (matching the mismatch ban's "this fire is fine" semantics).
    Entry* insert(std::uint64_t program_set) {
        if (entries.size() >= kMaxProgramSets) {
            Entry& victim = entries.back();
            if (victim.exec != nullptr) cudaGraphExecDestroy(victim.exec);
            entries.pop_back();
            if (++evictions > kMaxEvictions) banned = true;
        }
        entries.insert(entries.begin(), Entry{program_set});
        return &entries.front();
    }
};

struct ForwardGraphKeyHash {
    std::size_t operator()(const ForwardGraphKey& k) const noexcept {
        return static_cast<std::size_t>(k.num_requests) ^
               (static_cast<std::size_t>(k.num_tokens) << 12) ^
               (static_cast<std::size_t>(k.variant) << 24) ^
               (static_cast<std::size_t>(k.variant) >> 8);
    }
};

// Cache of executable graphs keyed by shape. Owned by BatchEngine;
// graphs are destroyed in the destructor. Wide, page-limited serving can
// create many decode batch sizes, and cudaGraphExec_t objects retain
// device-side resources, so keep this bounded.
class ForwardGraphCache {
public:
    struct Metrics {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t captures = 0;
        std::uint64_t evictions = 0;
    };
    ForwardGraphCache() = default;
    ~ForwardGraphCache() noexcept {
        for (auto& [_, exec] : execs_) cudaGraphExecDestroy(exec);
    }
    ForwardGraphCache(const ForwardGraphCache&) = delete;
    ForwardGraphCache& operator=(const ForwardGraphCache&) = delete;

    // Returns a captured graph for `key`, or nullptr if none cached.
    cudaGraphExec_t get(const ForwardGraphKey& key) const noexcept {
        auto it = execs_.find(key);
        if (it == execs_.end()) {
            ++metrics_.misses;
            return nullptr;
        }
        ++metrics_.hits;
        return it->second;
    }

    // Stores a captured graph. Caller transfers ownership.
    void put(const ForwardGraphKey& key, cudaGraphExec_t exec) {
        if (auto it = execs_.find(key); it != execs_.end()) {
            cudaGraphExecDestroy(it->second);
            it->second = exec;
            ++metrics_.captures;
            return;
        }

        if (execs_.size() >= kMaxEntries && !execs_.empty()) {
            auto victim = execs_.begin();
            cudaGraphExecDestroy(victim->second);
            execs_.erase(victim);
            ++metrics_.evictions;
        }
        execs_.emplace(key, exec);
        ++metrics_.captures;
    }

    std::size_t size() const noexcept { return execs_.size(); }
    Metrics metrics() const noexcept { return metrics_; }

private:
    static constexpr std::size_t kMaxEntries = 128;
    std::unordered_map<ForwardGraphKey, cudaGraphExec_t,
                       ForwardGraphKeyHash> execs_;
    mutable Metrics metrics_;
};

}  // namespace pie_cuda_driver
