#pragma once

// CUDA-graph cache for the decode forward body.
//
// Why: every fire_batch in a steady decode workload issues the same ~420
// kernel launches per layer × 28 layers + embed/lm_head/etc. Per-launch
// overhead dominates at small batch sizes. Capturing the launch sequence
// once into a `cudaGraphExec_t` and replaying it on subsequent fires of
// the same shape collapses N CPU-side forward launch invocations into a
// single `cudaGraphLaunch`. Sampling/probe response work is intentionally
// outside the graph because sampler layouts vary independently from the
// forward shape.
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
    std::uint32_t variant = 0;

    bool operator==(const ForwardGraphKey& o) const noexcept {
        return num_requests == o.num_requests && variant == o.variant;
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

static_assert(forward_graph_request_bucket(1, 512) == 1);
static_assert(forward_graph_request_bucket(3, 512) == 4);
static_assert(forward_graph_request_bucket(5, 512) == 8);
static_assert(forward_graph_request_bucket(255, 512) == 256);
static_assert(forward_graph_request_bucket(257, 512) == 272);
static_assert(forward_graph_request_bucket(506, 512) == 512);
static_assert(forward_graph_request_bucket(129, 130) == 130);

struct ForwardGraphKeyHash {
    std::size_t operator()(const ForwardGraphKey& k) const noexcept {
        return static_cast<std::size_t>(k.num_requests) ^
               (static_cast<std::size_t>(k.variant) << 24) ^
               (static_cast<std::size_t>(k.variant) >> 8);
    }
};

// Cache of executable graphs keyed by shape. Owned by Executor;
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
