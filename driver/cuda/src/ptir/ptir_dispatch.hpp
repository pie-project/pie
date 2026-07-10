#pragma once

// Driver-side PTIR (thrust-3) stage-program dispatcher — a CUDA-FREE façade over
// the tier-0 runtime (`program_runtime.hpp`). The impl + the `__global__` tier-0
// kernels live in `ptir_dispatch.cu`, so host `.cpp` translation units that
// include `executor.hpp` never pull device code (the tier-0 headers only compile
// under nvcc). This is the driver half of P2c: the executor calls `run()` when a
// forward request carries `ptir_program_*`; the dispatcher decodes (hash-cache),
// instantiates (persistent, by wire instance id), fires, and harvests outputs.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <pie_driver_abi.h>

#include "pie_native/launch_view.hpp"

#include "ptir/fire_geometry.hpp"

namespace pie_cuda_driver::ptir {

class PtirDispatch {
  public:
    PtirDispatch();
    ~PtirDispatch();
    PtirDispatch(const PtirDispatch&) = delete;
    PtirDispatch& operator=(const PtirDispatch&) = delete;

    int register_program(std::uint64_t program_hash,
                         pie_native::ByteSlice canonical,
                         pie_native::ByteSlice sidecar,
                         std::string* err);

    int register_channel(const PieChannelDesc& channel,
                         PieChannelEndpointBinding* binding,
                         std::string* err);

    int bind_instance(std::uint64_t instance_id,
                      std::uint64_t program_hash,
                      std::uint64_t pacing_wait_id,
                      const std::vector<std::uint64_t>& channel_ids,
                      const std::vector<PieChannelValueDesc>& seed_values,
                      PieInstanceBinding* binding,
                      std::string* err);

    int validate_launch(const pie_native::LaunchView& view, std::string* err);

    void close_instance(std::uint64_t instance_id);
    int close_channel(std::uint64_t channel_id, std::string* err);

    bool run(const pie_native::LaunchView& view,
             const void* logits, std::uint32_t vocab, cudaStream_t stream,
             const PieRuntimeCallbacks* runtime,
             PieCompletion completion);

    std::vector<std::pair<std::uint64_t, std::uint64_t>> settle_failed_launch(
        const pie_native::LaunchView& view,
        cudaStream_t execution_stream);

    // W1.1 PRE-FORWARD descriptor resolution: for the request's device-geometry
    // PTIR program (descriptor ports bind channels), decode + get-or-build the
    // instance (applying seeds on its first fire) and read its port channels'
    // current cells into `out` — the standard forward geometry the executor
    // feeds into batch assembly INSTEAD of the (empty) wire geometry fields.
    // Returns true iff a device-geometry program was resolved; false with an
    // empty `*err` if the request carries no such program, or false with a
    // non-empty `*err` if a descriptor channel is not ready (W1.6 — the executor
    // must fail the fire; the runtime's poison plumbing surfaces it to the guest).
    bool resolve_descriptors(const pie_native::LaunchView& view,
                             std::uint32_t page_size,
                             std::uint32_t device_pages,
                             FireGeometry& out,
                             std::string* err);

    struct Impl;

  private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace pie_cuda_driver::ptir
