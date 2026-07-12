#pragma once

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
#include <algorithm>
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
// PTIR: `program_set_hash` is the batching identity (contract C3)
// of the stage programs captured in-graph — the order-independent fold of the
// distinct program container hashes in the fire (see make_program_set_hash).
// It is 0 for a pure-trunk / non-PTIR fire, so today's decode path is
// bit-identical (the field defaults to 0 and the 3-field aggregate inits at the
// executor capture sites keep compiling unchanged). When tier-1 glue kernels are
// captured in-graph, two fires that agree on {num_requests, num_tokens, variant}
// but run different program sets emit different kernel sequences, so they MUST
// NOT share a graph — this field keeps them distinct.
struct ForwardGraphKey {
    int num_requests;
    int num_tokens;
    std::uint32_t variant = 0;
    std::uint64_t program_set_hash = 0;   // C3 batching identity (0 = no PTIR)

    bool operator==(const ForwardGraphKey& o) const noexcept {
        return num_requests == o.num_requests &&
               num_tokens == o.num_tokens &&
               variant == o.variant &&
               program_set_hash == o.program_set_hash;
    }
};

// Fold a batch's distinct program container hashes into ONE order-independent
// program-set identity (C3). Batching identity is a SET of stage-program traces
// (T5) — co-batched instances of the same program share a hash, and the union
// is what a captured graph's glue kernels encode — so the fold must be
// commutative + duplicate-insensitive. We sort+unique then FNV-1a64 over the
// canonical bytes (same FNV as program_hash / container_hash). Empty set → 0
// (matches the non-PTIR default so the trunk path is unchanged).
inline std::uint64_t make_program_set_hash(std::vector<std::uint64_t> hashes) {
    if (hashes.empty()) return 0;
    std::sort(hashes.begin(), hashes.end());
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::uint64_t x : hashes) {
        for (int b = 0; b < 8; ++b) { h ^= (x >> (b * 8)) & 0xff; h *= 0x100000001b3ULL; }
    }
    return h;
}

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

static_assert(forward_graph_request_bucket(1, 512) == 1);
static_assert(forward_graph_request_bucket(3, 512) == 4);
static_assert(forward_graph_request_bucket(5, 512) == 8);
static_assert(forward_graph_request_bucket(255, 512) == 256);
static_assert(forward_graph_request_bucket(257, 512) == 272);
static_assert(forward_graph_request_bucket(506, 512) == 512);
static_assert(forward_graph_request_bucket(129, 130) == 130);

struct ForwardGraphKeyHash {
    std::size_t operator()(const ForwardGraphKey& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k.num_requests) ^
               (static_cast<std::size_t>(k.num_tokens) << 12) ^
               (static_cast<std::size_t>(k.variant) << 24) ^
               (static_cast<std::size_t>(k.variant) >> 8);
        // Mix the C3 program-set hash (0 for non-PTIR fires → unchanged).
        h ^= static_cast<std::size_t>(k.program_set_hash) +
             0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

// Cache of executable graphs keyed by shape. Owned by BatchEngine;
// graphs are destroyed in the destructor. Wide, page-limited serving can
// create many decode batch sizes, and cudaGraphExec_t objects retain
// device-side resources, so keep this bounded.
class ForwardGraphCache {
public:
    ForwardGraphCache() = default;
    ~ForwardGraphCache() noexcept {
        for (auto& [_, exec] : execs_) cudaGraphExecDestroy(exec);
    }
    ForwardGraphCache(const ForwardGraphCache&) = delete;
    ForwardGraphCache& operator=(const ForwardGraphCache&) = delete;

    // Returns a captured graph for `key`, or nullptr if none cached.
    cudaGraphExec_t get(const ForwardGraphKey& key) const noexcept {
        auto it = execs_.find(key);
        return it == execs_.end() ? nullptr : it->second;
    }

    // Stores a captured graph. Caller transfers ownership.
    void put(const ForwardGraphKey& key, cudaGraphExec_t exec) {
        if (auto it = execs_.find(key); it != execs_.end()) {
            cudaGraphExecDestroy(it->second);
            it->second = exec;
            return;
        }

        if (execs_.size() >= kMaxEntries && !execs_.empty()) {
            auto victim = execs_.begin();
            cudaGraphExecDestroy(victim->second);
            execs_.erase(victim);
        }
        execs_.emplace(key, exec);
    }

    std::size_t size() const noexcept { return execs_.size(); }

private:
    static constexpr std::size_t kMaxEntries = 128;
    std::unordered_map<ForwardGraphKey, cudaGraphExec_t,
                       ForwardGraphKeyHash> execs_;
};

}  // namespace pie_cuda_driver
