#pragma once

// Default (MLX-free) forward executor for the Metal driver.
//
// Drives the raw Metal-4 decode pipeline — heap_bind weights + DAG + PSOs +
// KV / linear-state — from the wire `PieForwardRequestView`. This is the path
// the worker runs e2e (the worker builds pie_driver_metal_lib with MLX OFF by
// default, so this is the default compute backend).
//
// OWNERSHIP: alpha owns this skeleton + the request-marshaling seam; delta
// fills `run_forward`'s body + the two state bridges:
//   * gemma4 (attention) → runtime paged-KV CSR → kv_append/sdpa
//     (phys_slot = kv_page_indices[page_base + pos/page_size]*page_size
//                  + pos%page_size).
//   * qwen3.6 (GDN/linear-attn) → runtime rs_slot_ids → conv/recurrent state
//     held in the executor's own heap buffers (no pages).
//
// SKELETON STATE: run_forward emits a sampler-shaped stub response so the
// pipeline compiles, links into the lib, and round-trips end-to-end before the
// real forward body lands.

#include <cstdint>
#include <memory>
#include <string>

#include "forward_executor.hpp"

namespace pie_metal_driver::raw_metal {

class RawMetalExecutor : public IForwardExecutor {
public:
    RawMetalExecutor(std::string checkpoint_dir,
                     std::string kernels_dir,
                     std::uint32_t vocab_size);
    ~RawMetalExecutor() override;

    RawMetalExecutor(const RawMetalExecutor&) = delete;
    RawMetalExecutor& operator=(const RawMetalExecutor&) = delete;

    // Run one forward + sampling pass over the raw-Metal pipeline. Fills `out`
    // via `builder` (whose scratch backs the response slices until the next
    // build()).
    void run_forward(const pie_driver::PieForwardRequestView& req,
                     pie_driver::ResponseBuilder& builder,
                     pie_driver::PieForwardResponseView& out) override;

private:
    struct Impl;                 // delta-owned compute state (ctx/dag/psos/kv).
    std::unique_ptr<Impl> impl_;
    std::string   checkpoint_dir_;
    std::string   kernels_dir_;
    std::uint32_t vocab_size_;
};

// Construct a RawMetalExecutor for the checkpoint at `checkpoint_dir`, or
// return nullptr (caller serves the stub Forward) if the model can't be
// loaded. `kernels_dir` locates the .metal sources for runtime PSO compilation.
std::unique_ptr<RawMetalExecutor> make_raw_metal_executor(
    const std::string& checkpoint_dir,
    const std::string& kernels_dir,
    std::uint32_t vocab_size);

}  // namespace pie_metal_driver::raw_metal
