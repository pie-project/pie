#pragma once

// `pipeline::Dispatch`: driver-side PTIR stage-program dispatcher —
// a CUDA-FREE façade over the generated runtime (`program_runtime.hpp`). The impl
// + generated/library launch integration live in `dispatch.cu`, so host `.cpp`
// translation units that include `batch/forward.hpp` never pull device code (the
// tier-0 headers only compile under nvcc). This is the driver half of the submission path: the
// executor opens a launch before descriptor resolution, invokes the declared
// model phases at their anatomical hooks, then atomically finishes after
// lm_head. `run()` remains the boundary-stage convenience path used by focused
// tests. Programs arrive as a compiler-built launch package -- there is no plan
// format to decode here -- and instances stay persistent by wire id. Owned once by `pipeline::Registry`
// (`registry.hpp`), which is the single construction site.

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <pie_driver_abi.h>

#include "pie/driver/fire/view.hpp"

#include "pie/driver/fire/geometry.hpp"

#include "model/lora.hpp"
#include "model/stage_hooks.hpp"

namespace pie_cuda_driver::pipeline {

// Shared pure-host PTIR decode model (trace/op-table/container/bound/
// fire-geometry) now lives in pie::driver::launch (driver/common); bring it into
// scope so the CUDA-side tier-0/1 code below can use it unqualified.
using namespace pie::driver::fire;

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
    /// Fused regions compiled from host-generated source, versus regenerated
    /// in-driver. Distinguishes a live host path from a silent fallback.
    std::uint64_t generated_host_sources = 0;
    std::uint64_t generated_driver_sources = 0;
    std::uint64_t generated_stage_cache_entries = 0;
    std::uint64_t generated_program_cache_entries = 0;
    std::uint64_t generated_negative_cache_entries = 0;
    /// Regions whose analysis the host shipped. `region_support.hpp`'s copies
    /// are gone, so this is no longer half of a divergence gate -- it is the
    /// count of regions the driver was told about rather than worked out, and
    /// a program with fused regions that registers with this at zero would
    /// have failed in the module cache first.
    std::uint64_t region_host_supplied = 0;
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

    // `package` is the program in the shape this driver executes it; the
    // driver has no other source for it. `emitted` is the host's generated
    // kernels, which the driver cannot regenerate -- its emitters are gone.
    int register_program(std::uint64_t program_hash,
                         const PieLaunchPackage& package,
                         PieEmittedKernelSlice emitted,
                         PieRegionAnalysisSlice region_analysis,
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

    int validate_launch(const pie::driver::fire::LaunchView& view, std::string* err);

    // Declared-phase execution, split along the frame pipeline's two
    // tracks (venus decision: FramePrepare is host-only, StepEnqueue is
    // enqueue-only):
    //
    //   * `begin_host` — every host-side pass of wave admission: instance
    //     lookups, commit-snapshot claims, channel ticket builds, and the
    //     registry sequence applies (which MUST run in wave order — the
    //     frame driver calls begin_host for its steps in step order).
    //     Nothing reaches the stream. `stream` is recorded for the later
    //     enqueue half and for descriptor readbacks.
    //   * `begin_enqueue` — the stream half, in the original order:
    //     initialization/publication ordering waits, the pull-validate
    //     ticket upload + kernel, and the Prologue phase. Must be called
    //     once, after `begin_host`, at the step's position in the frame's
    //     enqueue sequence (step i+1's pull-validate reads ring state
    //     step i's settlement publishes — stream order carries that).
    //
    // `begin` is the fused convenience form (host + enqueue back to back)
    // for single-step callers like `run`. Model hook points invoke
    // `execute_attention_phase` for each layer. `finish` executes Epilogue
    // and performs the sole atomic channel publication.
    std::unique_ptr<StagedLaunch> begin_host(
        const pie::driver::fire::LaunchView& view,
        cudaStream_t stream);

    void begin_enqueue(StagedLaunch& launch);

    std::unique_ptr<StagedLaunch> begin(
        const pie::driver::fire::LaunchView& view,
        cudaStream_t stream);

    void update_launch_geometry(
        StagedLaunch& launch,
        const pie::driver::fire::LaunchView& resolved_view,
        std::span<const std::uint32_t> program_token_starts);

    // ── Hook prepared mode (stage 6 increment 4 + eager unification) ────
    // Fire-level prepare pass for a hook fire. Since the eager-path
    // unification the batch engine runs it for EVERY pure-decode hook fire
    // — eager and graph alike — so both modes drive the attention phases
    // through one prepare-then-launch seam; graph mode adds capture/replay
    // on top. Hoists EVERY attention-phase prepare — binding
    // assembly, grouping, `prepare_generated_stage` (stable per-occurrence
    // buffers), channel-effect application, score-sideband sizing — out of
    // the body, in the exact (layer-major, OnAttnProj-then-OnAttn) order the
    // body will consume them. After a nonzero return the launch is in
    // prepared mode: `execute_attention_phase` becomes a pure launch replay
    // against the stored cursor (loud throw on any order mismatch), which is
    // what a captured graph records and what a replayed graph re-executes.
    //
    // Returns a fingerprint over every address and grid the captured body
    // bakes (stable-buffer blocks, channel-ring arrays, sideband-arena
    // addresses, region launch geometry). The batch engine stores it per
    // graph key and recaptures when it changes — that is the generation /
    // growth / instance-churn invalidation in one value. Returns 0 — with NO
    // side effects on the launch — when this fire cannot run prepared (a
    // stage reads Query, needs per-fire device allocations, a non-decode
    // fire, …) and must take the legacy interleaved eager body.
    struct HookReplayPrepare {
        const model::AttentionObservation* observation = nullptr;
        model::HookSidebandArena* arena = nullptr;
        std::uint32_t num_q_heads = 0;
        std::uint32_t hook_free_prefix_rows = 0;
        bool wants_attn_score = false;
        bool wants_page_mask = false;
        cudaStream_t stream = nullptr;
    };
    std::uint64_t prepare_attention_phases(
        StagedLaunch& launch,
        const HookReplayPrepare& in);

    // Structural check: every prepared attention invocation was consumed by
    // the body that just ran. Run by the batch engine right after a
    // hook-graph capture AND after every prepared-eager body — the moments
    // the model's per-layer hook coverage is observable — and never after a
    // replay (a replayed body does not touch the cursors). Throws on
    // partial consumption.
    void verify_hook_capture_consumed(StagedLaunch& launch) const;

    // `sideband` carries what the model body published for exactly this hook
    // invocation — the fire's KV geometry, the layer's captured scores, the
    // page-mask destination. An empty sideband is "the body published
    // nothing", and a program that then names the corresponding intrinsic
    // fails loudly in the lane resolvers.
    void execute_attention_phase(
        StagedLaunch& launch,
        std::uint8_t phase,
        const void* query_data,
        std::uint32_t query_rows,
        std::uint32_t query_columns,
        std::uint32_t layer,
        cudaStream_t stream,
        bool query_is_f32 = false,
        const model::StageHookSideband& sideband = {});

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
        const pie::driver::fire::LaunchView& view,
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
        // Base of the forward's `[max_tokens]` i32 token buffer, non-null only
        // when the forward reduced the vocabulary as it produced it and so
        // never wrote `direct_bf16_logits` (§20.37). Must be paired with a
        // launch that `launch_epilogue_is_greedy_argmax` accepted.
        const std::int32_t* presampled_tokens = nullptr,
        FinishBreakdown* breakdown = nullptr);

    void abort(StagedLaunch& launch, cudaStream_t stream) noexcept;

    bool launch_has_attention_stages(
        const pie::driver::fire::LaunchView& view) const;

    // How many LEADING wire request rows are covered by no attention-stage
    // program. Rows [0, n) may take hook-free fast paths; everything at or
    // after row n must run hook-visible. Returns 0 — "no fast prefix" — when
    // a hook-carrying program has no wire span to locate it by
    // (device-resolved geometry) or when per-program row attribution is
    // absent. The count is in WIRE request rows, and it is conservative by
    // construction: rows past the wire span (composed device-geometry
    // suffix) are never counted into the prefix.
    //
    // Since B (the fire planner's first consumed lowering): when the
    // scheduler crossed its planned prefix
    // (`view.planned_hook_free_prefix_rows`), the PLAN owns the answer and
    // the compiled-plan derivation below becomes a cross-check — a
    // disagreement refuses the launch loudly.
    std::uint32_t launch_hook_free_prefix_rows(
        const pie::driver::fire::LaunchView& view) const;

    // The compiled-plan derivation alone (the pre-plan behavior; the
    // cross-check's second opinion).
    std::uint32_t derive_hook_free_prefix_rows(
        const pie::driver::fire::LaunchView& view) const;

    // Whether any program in this launch reads `AttnScore`. Capture is opt-in
    // per fire because it costs an extra `[num_q_heads, kv_len]` write inside
    // the attention kernel; a launch that does not observe scores must pay
    // nothing.
    bool launch_wants_attn_score(
        const pie::driver::fire::LaunchView& view) const;

    // Whether any program in this launch writes the `attn_page_mask` sink. The
    // model body allocates the keep buffer only when true, and only then does
    // it pay the per-layer compaction.
    bool launch_wants_page_mask(
        const pie::driver::fire::LaunchView& view) const;

    // Whether every epilogue in this launch is a bare greedy argmax over
    // `logits` that only publishes its token to a channel. That is the one
    // shape whose logits can be reduced while the LM head GEMM produces them
    // instead of being materialised (§20.37), because nothing else in the
    // stage can observe the values. Asked before the forward runs, since the
    // forward is what decides whether to materialise them. `vocab` is the
    // weight's row count: a program declaring a narrower one is rejected,
    // because the fused reduction cannot honour it.
    bool launch_epilogue_is_greedy_argmax(
        const pie::driver::fire::LaunchView& view,
        std::uint32_t vocab) const;

    // Whether this model's decode path can honour a page mask: the plan must
    // not depend on the page counts it was planned against, or substituting a
    // compacted list at launch is silently wrong.
    void set_attn_page_mask_available(bool available);

    // Whether any program in this launch calls the `lora` sink in its
    // prologue. The frame queries this to decide whether to fetch and thread
    // the resolved lora table into the model body.
    bool launch_wants_lora(
        const pie::driver::fire::LaunchView& view) const;

    // Whether the active model's projection path can honour the `lora` sink
    // (capability `has_lora`). Default FALSE: a program naming the sink is
    // refused at bind until the model opts in, because a bound-but-ignored
    // configuration sink is a program whose adapter silently never applied.
    void set_lora_available(bool available);

    // The launch's begin-time-resolved lora configuration: one entry per lane
    // whose program carries the sink (empty when none does). A borrowed view
    // into launch-owned storage — valid until the launch is finished or
    // aborted. Populated when the prologue executes in `begin`, which is why
    // the sink cannot use `attn_page_mask`'s body-owned-buffer shape: the
    // model body does not exist yet at that point.
    model::LoraTable launch_lora_table(const StagedLaunch& launch) const;

    // Whether this model + cache can honour the `envelope_dot` contract.
    // Mirrors the `has_kv_envelopes` driver capability; a program that names
    // the kernel is refused at bind when false.
    // `enable` is invoked once, lazily, when a program that names
    // `envelope_dot` is registered — envelopes cost 4/page_size of the KV
    // cache, so a model no program observes must not pay for them.
    void set_kv_envelopes_available(
        bool available,
        std::function<void()> enable = nullptr);

    void set_attention_hook_coverage(
        bool supported,
        std::uint32_t model_layers = 0);

    void close_instance(std::uint64_t instance_id);
    int close_channel(std::uint64_t channel_id, std::string* err);

    bool run(const pie::driver::fire::LaunchView& view,
             const void* logits, std::uint32_t vocab, cudaStream_t stream,
             const PieRuntimeCallbacks* runtime,
             PieCompletion completion,
             const std::uint16_t* direct_bf16_logits = nullptr,
             const std::uint32_t* direct_row_indices = nullptr,
             std::span<const std::uint32_t> mtp_draft_row_starts = {},
             std::span<const std::uint32_t> mtp_draft_row_counts = {},
             std::uint32_t direct_bf16_row_capacity = 0);

    std::vector<std::uint32_t> mtp_draft_rows(
        const pie::driver::fire::LaunchView& view) const;

    std::vector<std::pair<std::uint64_t, std::uint64_t>> settle_failed_launch(
        const pie::driver::fire::LaunchView& view,
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
    bool resolve_descriptors(const pie::driver::fire::LaunchView& view,
                             std::uint32_t page_size,
                             std::uint32_t device_pages,
                             ResolvedPrograms& out,
                             std::string* err,
                             bool allow_structured_masks = false,
                             StagedLaunch* launch = nullptr,
                             bool allow_device_composed = false);

    // STATIC v1 mask-scope admission check, run at frame admission — before
    // the arena commit and before any prepare-time state mutation. A
    // device-geometry program carrying a dense device `AttnMask` channel
    // composes only SOLO (batch_compose.hpp "out of scope" v1); the runtime
    // scheduler batches such fires alone, and `prepare_step`'s resolve-time
    // throw used to be the only defence — reachable AFTER `begin_host` had
    // applied the wave's channel tickets, so a violating step poisoned every
    // lane in the frame and leaked the dead lanes' pages. Returns the
    // offending program's index in `view.ptir_program_hashes` when the step
    // is MULTI-program and one of its device-geometry programs binds a
    // dense device mask, -1 otherwise. `allow_structured_masks` mirrors the
    // resolve path (`resolve_attention_mask`): a mask the structured-mask
    // recognizer lowers to a runtime window override is not a dense mask.
    int dense_mask_scope_violation(const pie::driver::fire::LaunchView& view,
                                   bool allow_structured_masks) const;

    // Device-composition lowering, split along the frame pipeline: the
    // `stage_*` half (FramePrepare) validates and builds the lane tables —
    // which read the live registry ring cursors and the wave's channel-
    // effect sets, valid only at this wave's position in `begin_host`
    // order — and pulls host-writer rings; the `enqueue_*` half
    // (StepEnqueue) claims the upload arena and launches the compose
    // kernel at the step's stream position.

    // Fixed-capacity single-token decode lowering directly into stable
    // model input buffers. `stage` returns false with an empty error when
    // the launched programs require general host composition.
    //
    // `scope` selects the ordered ENVELOPE sub-batch of a mixed
    // [wire][envelope] step: a contiguous program suffix and the wire
    // sub-batch's row/request/page totals (the compose kernel writes the
    // envelope rows in place after the ordinary wire refill). The
    // default all-zero scope is the whole-step all-envelope form.
    struct FixedDecodeScope {
        std::uint32_t program_begin = 0;
        std::uint32_t program_count = 0;  // 0 = every program
        std::uint32_t row_base = 0;
        std::uint32_t request_base = 0;
        std::uint32_t page_base = 0;
    };

    bool stage_fixed_decode(
        const pie::driver::fire::LaunchView& view,
        std::uint32_t page_size,
        std::uint32_t device_pages,
        const FixedDecodeDeviceBuffers& buffers,
        std::string* err,
        StagedLaunch& launch,
        const FixedDecodeScope& scope);

    bool enqueue_fixed_decode(
        const FixedDecodeDeviceBuffers& buffers,
        std::string* err,
        StagedLaunch& launch);

    // Resolve device-carried decode values into a host-owned shape template.
    bool stage_decode_envelopes(
        const pie::driver::fire::LaunchView& view,
        std::span<const std::uint32_t> program_token_starts,
        std::span<const std::uint32_t> program_request_starts,
        std::span<const std::uint32_t> template_kv_page_indptr,
        const DecodeEnvelopeDeviceBuffers& buffers,
        std::string* err,
        StagedLaunch& launch);

    bool enqueue_decode_envelopes(
        const DecodeEnvelopeDeviceBuffers& buffers,
        std::string* err,
        StagedLaunch& launch);

    bool has_decode_envelopes(
        const pie::driver::fire::LaunchView& view) const;

    // Per-request page counts for attention PLANNING of a device-composed
    // batch: wire counts raised to each envelope lane's host-known upper
    // bound (min of the trace's page-envelope width and its translation
    // span). Planning from these bounds — never from placeholder wire
    // geometry — keeps XQA bucket selection and FlashInfer plans safe for
    // any device-resolved length. Returns false if no lane needed a bound.
    bool envelope_plan_page_bounds(
        const pie::driver::fire::LaunchView& view,
        std::span<const std::uint32_t> program_request_starts,
        std::span<const std::uint32_t> wire_kv_page_indptr,
        std::vector<std::uint32_t>& per_request_pages) const;

    DispatchStats stats() const;

    struct Impl;

  private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace pie_cuda_driver::pipeline
