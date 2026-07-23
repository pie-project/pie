#pragma once

// `pipeline::Dispatch`: driver-side PTIR stage-program dispatcher —
// a CUDA-FREE façade over the generated runtime (`program_runtime.hpp`). The impl
// + generated/library launch integration live in `dispatch.cu`, so host `.cpp`
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

    // Fire-timing sub-breakdown of `begin` (microseconds; -1 = not
    // measured): the serial lookup/alloc pass, the parallel ticket
    // builds, the serial apply+staging pass, and the pull-validate
    // upload+launch (W6 attribution).
    struct BeginBreakdown {
        std::int64_t prologue_us = -1;
        std::int64_t pass_a_us = -1;
        std::int64_t tickets_us = -1;
        std::int64_t pass_c_us = -1;
        std::int64_t pull_validate_us = -1;
    };
    const BeginBreakdown& begin_breakdown() const { return begin_breakdown_; }

    struct State;

  private:
    StagedLaunch();
    std::unique_ptr<State> state_;
    BeginBreakdown begin_breakdown_;
    friend class Dispatch;
};

struct DispatchStats {
    std::uint64_t generated_fused_groups = 0;
    std::uint64_t generated_fused_body_launches = 0;
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
    std::uint64_t generated_compilations = 0;
    std::uint64_t generated_disk_hits = 0;
    std::uint64_t generated_disk_writes = 0;
    std::uint64_t generated_disk_errors = 0;
    std::uint64_t generated_negative_hits = 0;
    std::uint64_t generated_stage_cache_entries = 0;
    std::uint64_t generated_program_cache_entries = 0;
    std::uint64_t generated_negative_cache_entries = 0;
    std::uint64_t descriptor_readback_batches = 0;
    std::uint64_t descriptor_readback_cells = 0;
    std::uint64_t descriptor_readback_bytes = 0;
    std::uint64_t fixed_decode_batches = 0;
    std::uint64_t fixed_decode_lanes = 0;
    // Lanes the fixed-decode compose kernel fail-stopped (geometry or
    // containment inconsistency; NOT in-band -1 skips). Any nonzero count is
    // a chain kill that also logs loudly when detected.
    std::uint64_t fixed_decode_chain_kills = 0;
    std::uint64_t decode_envelope_batches = 0;
    std::uint64_t decode_envelope_lanes = 0;
    std::uint64_t decode_envelope_chain_kills = 0;
    // Channel-registry slot capacity: starts at kInitialChannelSlots and
    // doubles on demand (each doubling quiesces the device — RV-27). Lets
    // tests assert that a growth actually happened under their load.
    std::uint64_t channel_slot_capacity = 0;
};

struct FixedDecodeDeviceBuffers {
    std::uint32_t* token_ids = nullptr;
    std::uint32_t* position_ids = nullptr;
    std::uint32_t* qo_indptr = nullptr;
    std::uint32_t* kv_page_indices = nullptr;
    std::uint32_t* kv_page_indptr = nullptr;
    std::uint32_t* kv_last_page_lens = nullptr;
    std::uint32_t* w_page = nullptr;
    std::uint32_t* w_off = nullptr;
    std::uint8_t* row_valid = nullptr;
    std::int32_t* rs_slot_ids = nullptr;
    std::int32_t* sample_indices = nullptr;
    std::size_t token_capacity = 0;
    std::size_t request_capacity = 0;
    std::size_t page_capacity = 0;
    std::uint32_t dummy_page = 0;
};

struct DecodeEnvelopeDeviceBuffers {
    std::uint32_t* token_ids = nullptr;
    std::uint32_t* position_ids = nullptr;
    std::uint32_t* kv_page_indices = nullptr;
    std::uint32_t* kv_page_indptr = nullptr;
    std::uint32_t* kv_last_page_lens = nullptr;
    std::uint8_t* row_valid = nullptr;
    std::int32_t* rs_slot_ids = nullptr;
    std::uint32_t dummy_page = 0;
    std::uint32_t page_size = 0;
};

class Dispatch {
  public:
    Dispatch();
    ~Dispatch();
    Dispatch(const Dispatch&) = delete;
    Dispatch& operator=(const Dispatch&) = delete;

    // W2: size the channel registry for the fleet at model load (before
    // any registration traffic) so grow() never fires mid-ramp.
    void reserve_channel_slots(std::uint32_t min_slots);

    int register_program(std::uint64_t program_hash,
                         pie_native::ByteSlice canonical,
                         pie_native::ByteSlice sidecar,
                         std::string* err);

    int register_channel(const PieChannelDesc& channel,
                         PieChannelEndpointBinding* binding,
                         std::string* err);

    int bind_instance(std::uint64_t instance_id,
                      std::uint64_t program_hash,
                      std::uint32_t geometry_class,
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

    // Fire-timing sub-breakdown of `finish` (microseconds; -1 = not
    // measured): per-lane epilogue kernel enqueue, settlement-mutex
    // acquisition (contention with the settlement callback thread), and
    // settlement/publish preparation + enqueue.
    struct FinishBreakdown {
        std::int64_t epilogue_us = -1;
        std::int64_t settle_lock_us = -1;
        std::int64_t settle_prep_us = -1;
        // One level into the epilogue phase (fire-timing only): per-lane
        // task/binding assembly, signature grouping + independence check,
        // and generated-stage execution (metadata upload + body launches).
        std::int64_t epilogue_assemble_us = -1;
        std::int64_t epilogue_group_us = -1;
        std::int64_t epilogue_execute_us = -1;
        // Sections inside the generated-stage execution, summed across
        // groups: host table build, device workspace acquisition
        // (including the budget-fallback cudaMallocAsync path), staging
        // pack + H2D upload, and the launch region.
        std::int64_t epilogue_exec_build_us = -1;
        std::int64_t epilogue_exec_workspace_us = -1;
        std::int64_t epilogue_exec_upload_us = -1;
        std::int64_t epilogue_exec_launch_us = -1;
    };

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
        std::uint32_t direct_bf16_row_capacity = 0,
        const std::uint8_t* row_valid = nullptr,
        std::span<const std::uint32_t> row_valid_offsets = {},
        FinishBreakdown* breakdown = nullptr);

    void abort(StagedLaunch& launch, cudaStream_t stream) noexcept;

    /// Drain every parked settle_defer record (frame-group settlement) and
    /// arm the last in-flight callback to self-drain if some have not yet
    /// arrived. Called by the synchronous settle paths and the engine's
    /// truncation flush; spurious calls only settle early, never wrongly.
    void flush_deferred_settlement() noexcept;

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
    // When allowed, fixed one-token graph buckets return a shape-only template;
    // `enqueue_fixed_decode` resolves their values entirely on device.
    bool resolve_descriptors(const pie_native::LaunchView& view,
                             std::uint32_t page_size,
                             std::uint32_t device_pages,
                             ResolvedPrograms& out,
                             std::string* err,
                             bool allow_structured_masks = false,
                             StagedLaunch* launch = nullptr,
                             bool allow_device_composed = false);

    // Enqueue the fixed-capacity single-token decode lowering directly into
    // stable model input buffers. Returns false with an empty error when the
    // launched programs require general host composition.
    bool enqueue_fixed_decode(
        const pie_native::LaunchView& view,
        std::uint32_t page_size,
        std::uint32_t device_pages,
        const FixedDecodeDeviceBuffers& buffers,
        std::string* err,
        StagedLaunch& launch);

    // Resolve device-carried decode values into a host-owned shape template.
    bool enqueue_decode_envelopes(
        const pie_native::LaunchView& view,
        std::span<const std::uint32_t> program_token_starts,
        std::span<const std::uint32_t> program_request_starts,
        std::span<const std::uint32_t> template_kv_page_indptr,
        const DecodeEnvelopeDeviceBuffers& buffers,
        std::string* err,
        StagedLaunch& launch);

    bool has_decode_envelopes(
        const pie_native::LaunchView& view) const;

    // Per-request page counts for attention PLANNING of a device-composed
    // batch: wire counts raised to each envelope lane's host-known upper
    // bound (min of the trace's page-envelope width and its translation
    // span). Planning from these bounds — never from placeholder wire
    // geometry — keeps XQA bucket selection and FlashInfer plans safe for
    // any device-resolved length. Returns false if no lane needed a bound.
    bool envelope_plan_page_bounds(
        const pie_native::LaunchView& view,
        std::span<const std::uint32_t> program_request_starts,
        std::span<const std::uint32_t> wire_kv_page_indptr,
        std::vector<std::uint32_t>& per_request_pages) const;

    DispatchStats stats() const;

    struct Impl;

  private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace pie_cuda_driver::pipeline
