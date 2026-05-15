#pragma once

// CUDA-graph cache for the decode forward pass.
//
// Why: every fire_batch in a steady decode workload issues the same ~420
// kernel launches per layer × 28 layers + embed/lm_head/etc. Per-launch
// overhead dominates at small batch sizes. Capturing the launch sequence
// once into a `cudaGraphExec_t` and replaying it on subsequent fires of
// the same shape collapses N CPU-side launch invocations into a single
// `cudaGraphLaunch`.
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

#include <cuda_runtime.h>

namespace pie_cuda_driver {

// Bucket key for the graph cache. Two fires that match on this key share
// a captured graph. We choose the inputs that flashinfer's plan / our
// kernel-launch sequence depend on; per-token contents (token IDs,
// positions, KV page indices) flow through persistent buffers and don't
// affect graph topology.
struct ForwardGraphKey {
    int num_requests;

    bool operator==(const ForwardGraphKey& o) const noexcept {
        return num_requests == o.num_requests;
    }
};

struct ForwardGraphKeyHash {
    std::size_t operator()(const ForwardGraphKey& k) const noexcept {
        return static_cast<std::size_t>(k.num_requests);
    }
};

// Cache of executable graphs keyed by shape. Owned by Executor;
// graphs are destroyed in the destructor. Bounded LRU is overkill at the
// shapes we see — a few buckets suffice; we let it grow unbounded.
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
        execs_[key] = exec;
    }

    std::size_t size() const noexcept { return execs_.size(); }

private:
    std::unordered_map<ForwardGraphKey, cudaGraphExec_t,
                       ForwardGraphKeyHash> execs_;
};

}  // namespace pie_cuda_driver
