#pragma once

// `pipeline::Dispatch`: driver-side PTIR stage-program dispatcher —
// a CUDA-FREE façade over the tier-0 runtime (`program_runtime.hpp`). The impl
// + the `__global__` tier-0 kernels live in `dispatch.cu`, so host `.cpp`
// translation units that include `batch/forward.hpp` never pull device code (the
// tier-0 headers only compile under nvcc). This is the driver half of the submission path: the
// executor calls `run()` when a forward request carries `ptir_program_*`; the
// dispatcher decodes (hash-cache), instantiates (persistent, by wire instance
// id), fires, and harvests outputs. Owned once by `pipeline::Registry`
// (`registry.hpp`), which is the single construction site.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <pie_driver_abi.h>

#include "pie_native/launch_view.hpp"

#include "pie_native/ptir/fire_geometry.hpp"

namespace pie_cuda_driver::pipeline {

// Shared pure-host PTIR decode model (trace/op-table/container/bound/
// fire-geometry) now lives in pie_native::ptir (driver/common); bring it into
// scope so the CUDA-side tier-0/1 code below can use it unqualified.
using namespace pie_native::ptir;

class Dispatch {
  public:
    Dispatch();
    ~Dispatch();
    Dispatch(const Dispatch&) = delete;
    Dispatch& operator=(const Dispatch&) = delete;

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

    // W1.1 PRE-FORWARD descriptor resolution, over EVERY device-geometry
    // program in the batch: for each program whose trace is device-geometry
    // (the runtime's `detect_device_geometry` mirror — WSlot/WOff write
    // descriptors + a channel-bound [B, P>1] `Pages` port), read its port
    // channels' current cells into `out.per_program[p]` and map the resolved
    // WorkingSet-relative page references through the program's
    // `kv_translation` segment. Wire (non-device-geometry) programs keep an
    // empty per-program entry; the executor composes both kinds into one
    // forward batch (`compose_forward_batch`). Each resolved geometry is
    // validated independently. Returns true iff at least one device-geometry
    // program was resolved; false with an empty `*err` if the batch carries
    // none, or false with a non-empty `*err` on failure (not-ready descriptor
    // channel (W1.6), bad geometry — the executor must fail the fire).
    bool resolve_descriptors(const pie_native::LaunchView& view,
                             std::uint32_t page_size,
                             std::uint32_t device_pages,
                             ResolvedPrograms& out,
                             std::string* err);

    struct Impl;

  private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace pie_cuda_driver::pipeline
