#pragma once

// Forward executor for the Metal (MLX) driver.
//
// Owns the per-fire pipeline for `PIE_METHOD_FORWARD`:
//   1. plan   — read the wire `PieForwardRequestView` (SoA), derive the
//               per-token KV write slots from the paged page-table, and build
//               the per-slot sampler params.
//   2. stage  — upload the plan's index arrays into MLX `Tensor`s.
//   3. forward— invoke the arch `ModelGraph` (charlie) to build the lazy
//               logits graph, reading/writing delta's paged `KvCacheView`.
//   4. sample — evaluate + sample one token per slot (kernels/sampling).
//   5. pack   — group tokens per request into the `ResponseBuilder`.
//
// Mirrors driver/cuda/src/executor + driver/portable/src/executor, adapted to
// MLX's lazy-array model. alpha's InProcService constructs one Executor and
// calls `run_forward` from the server loop's Forward arm.

#include <cstdint>
#include <vector>

#include <pie_schema/response_builder.hpp>
#include <pie_schema/view.hpp>

#include "kernels/sampling.hpp"
#include "model/model_graph.hpp"
#include "model/model_kv.hpp"
#include "ops/tensor.hpp"

namespace pie_metal_driver {

class Executor {
public:
    Executor(model::ModelGraph& graph, KvCacheView& kv);

    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    // Run one forward + sampling pass. Fills `out` via `builder` (whose
    // scratch backs the response slices until the next build()).
    void run_forward(const pie_driver::PieForwardRequestView& req,
                     pie_driver::ResponseBuilder& builder,
                     pie_driver::PieForwardResponseView& out);

private:
    // Reusable host staging buffers (refilled per fire; their storage backs
    // the no-copy MLX input arrays for the duration of one run_forward).
    struct Staging {
        std::vector<int> token_ids;
        std::vector<int> positions;
        std::vector<int> logit_rows;
        std::vector<int> kv_page_indices;
        std::vector<int> kv_page_indptr;
        std::vector<int> kv_last_page_lens;
        std::vector<int> qo_indptr;
        std::vector<int> kv_write_indices;
        std::vector<int> slot_ids;
    };

    // Derive per-token physical KV write slots from the paged page-table.
    void compute_write_indices(const pie_driver::PieForwardRequestView& req);

    // Build the per-slot sampler params from the wire SoA sampler arrays.
    std::vector<sampling::SamplerParams> build_sampler_params(
        const pie_driver::PieForwardRequestView& req) const;

    model::ModelGraph& graph_;
    KvCacheView&       kv_;
    Staging            stg_;
    std::uint64_t      fire_counter_ = 0;
    std::vector<pie_driver::PerRequestOutput> per_req_;
    // qwen3.6 hybrid linear-attention state store (delta-owned). Null for
    // non-hybrid models; the runtime (entry.cpp) attaches it for hybrids so the
    // graph's gated_delta_net can gather/scatter per-request conv+recurrent
    // state. Borrowed — outlives the Executor.
    LinearStateCache*  lin_cache_ = nullptr;

public:
    // Attach the hybrid linear-attention state cache (qwen3.6). Threaded into
    // every ForwardBatch so the graph's linear layers can reach it.
    void set_linear_state_cache(LinearStateCache* cache) noexcept {
        lin_cache_ = cache;
    }
};

}  // namespace pie_metal_driver
