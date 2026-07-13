#pragma once

// `pipeline::Dispatch`: driver-side PTIR stage-program dispatcher —
// a CUDA-FREE façade over the tier-0 runtime (`program_runtime.hpp`). The impl
// + the `__global__` tier-0 kernels live in `dispatch.cu`, so host `.cpp`
// translation units that include `batch/forward.hpp` never pull device code (the
// tier-0 headers only compile under nvcc). This is the driver half of the submission path: the
// executor opens a launch before descriptor resolution, invokes the declared
// model phases at their anatomical hooks, then atomically finishes after
// lm_head. `run()` remains the boundary-stage convenience path used by focused
// tests. Programs are decoded from compiler-owned PTRP plans and instances stay
// persistent by wire id. Owned once by `pipeline::Registry`
// (`registry.hpp`), which is the single construction site.

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
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

class RetryableLaunchError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class StagedLaunch {
  public:
    ~StagedLaunch();
    StagedLaunch(const StagedLaunch&) = delete;
    StagedLaunch& operator=(const StagedLaunch&) = delete;

    struct State;

  private:
    StagedLaunch();
    std::unique_ptr<State> state_;
    friend class Dispatch;
};

struct DispatchStats {
    std::uint64_t grouped_tier0_groups = 0;
    std::uint64_t grouped_lanes = 0;
    std::uint64_t nucleus_library_groups = 0;
    std::uint64_t selection_library_groups = 0;
    std::uint64_t direct_bf16_groups = 0;
    std::uint64_t direct_bf16_solo_materializations = 0;
    std::uint64_t grouped_body_op_launches = 0;
    std::uint64_t overlapped_groups = 0;
    std::uint64_t ordered_alias_launches = 0;
    std::uint64_t structured_mask_direct = 0;
    std::uint64_t structured_mask_dense_fallback = 0;
    std::uint64_t large_nucleus_scalable_groups = 0;
    std::uint64_t shared_slot_exclusions = 0;
    std::uint64_t rs_exclusions = 0;
    std::uint64_t grouped_graph_cache_entries = 0;
    std::uint64_t grouped_graph_captures = 0;
    std::uint64_t grouped_graph_replays = 0;
};

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

    // Declared-phase execution. `begin` validates/pulls one logical fire and
    // executes Prologue before descriptor resolution. Model hook points invoke
    // `execute_attention_phase` for each layer. `finish` executes Epilogue and
    // performs the sole atomic channel publication.
    std::unique_ptr<StagedLaunch> begin(
        const pie_native::LaunchView& view,
        cudaStream_t stream);

    void update_launch_geometry(
        StagedLaunch& launch,
        const pie_native::LaunchView& resolved_view,
        std::span<const std::uint32_t> program_token_starts);

    void execute_attention_phase(
        StagedLaunch& launch,
        std::uint8_t phase,
        const void* query_data,
        std::uint32_t query_rows,
        std::uint32_t query_columns,
        std::uint32_t layer,
        cudaStream_t stream,
        bool query_is_f32 = false);

    bool finish(
        StagedLaunch& launch,
        const pie_native::LaunchView& view,
        const void* logits, std::uint32_t vocab, cudaStream_t stream,
        const PieRuntimeCallbacks* runtime,
        PieCompletion completion,
        const std::uint16_t* direct_bf16_logits = nullptr,
        const std::uint32_t* direct_row_indices = nullptr,
        std::span<const std::uint32_t> mtp_draft_row_starts = {},
        std::span<const std::uint32_t> mtp_draft_row_counts = {},
        std::uint32_t direct_bf16_row_capacity = 0);

    void abort(StagedLaunch& launch, cudaStream_t stream) noexcept;

    bool launch_has_attention_stages(
        const pie_native::LaunchView& view) const;
    void set_attention_hook_coverage(
        bool supported,
        std::uint32_t model_layers = 0);

    void close_instance(std::uint64_t instance_id);
    int close_channel(std::uint64_t channel_id, std::string* err);

    bool run(const pie_native::LaunchView& view,
             const void* logits, std::uint32_t vocab, cudaStream_t stream,
             const PieRuntimeCallbacks* runtime,
             PieCompletion completion,
             const std::uint16_t* direct_bf16_logits = nullptr,
             const std::uint32_t* direct_row_indices = nullptr,
             std::span<const std::uint32_t> mtp_draft_row_starts = {},
             std::span<const std::uint32_t> mtp_draft_row_counts = {},
             std::uint32_t direct_bf16_row_capacity = 0);

    std::vector<std::uint32_t> mtp_draft_rows(
        const pie_native::LaunchView& view) const;

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
                             std::string* err,
                             bool allow_structured_masks = false,
                             StagedLaunch* launch = nullptr);

    DispatchStats stats() const;
    std::uint64_t compiled_program_set_hash(
        const pie_native::LaunchView& view) const;

    struct Impl;

  private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace pie_cuda_driver::pipeline
