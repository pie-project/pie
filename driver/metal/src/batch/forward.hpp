#pragma once

// MetalExecutor — the forward seam between Context's PTIR launch path and
// the native Metal decode pipeline (metal_ptir_plan.md §5.1, D1: MetalExecutor
// backs it, not the MLX ops path). Deliberately narrow: `context.cpp` never
// includes MetalExecutor or any Metal/ObjC header directly — it owns one
// `std::unique_ptr<MetalExecutor>`, created lazily on the first
// forward-needing launch, and calls `setup()` / `forward()` through this
// plain-C++ interface only. The implementation is in `forward.cpp`;
// compiled only on Apple; a non-Apple build still links (Linux/CI stub
// builds keep validating the direct-ABI surface) but every call reports a
// clear "requires an Apple build" error instead of silently no-op'ing.
//
// Sealed M=1 scope (ordinary linear member): MetalExecutor holds
// exactly ONE resident linear KV/GDN sequence (the shipped single-stream
// path). `forward()` therefore accepts a fire only if it is either a fresh
// sequence for the SAME (or no) resident sequence (RS_FLAG_RESET / position
// 0), or the exact continuation of the currently resident one (matching
// sequence id, exact next position, and a KV page list that preserves the
// resident one as a prefix — physical page NUMBERING need not be
// arithmetically adjacent, e.g. {5, 9} is valid); anything else (a second
// concurrent sequence, a fork, a shared prefix, out-of-order positions, or
// duplicated pages) is rejected with a precise reason rather than silently
// corrupting state. `validate_linear_sequence_geometry` is the pure
// (host-testable, no Metal dependency) core of that check.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pie::metal {

struct BatchSchedule;
struct BatchStepInputs;
struct DecodeGeometry;
struct KvPagePool;
struct StepTiming;

}  // namespace pie::metal

namespace pie::metal::batch {

struct NativeAccess;

// One member's forward request for this fire: the NEW tokens/positions this
// fire adds (prefill chunk or a single decode token — never the full
// history; KV history lives in the decoder's resident ring), plus the
// recurrent-state slot bookkeeping the engine assigns per request
// (`RS_FLAG_RESET` mirrors `runtime/engine/src/driver/frame.rs`'s
// `RS_FLAG_RESET = 1`) and the this-fire KV page ids (used ONLY to validate
// the "single linear run" contract — Phase 1a never reads history through
// them; the decoder's own ring supplies it).
struct MemberForwardDesc {
    // The engine's stable identity for this PTIR instance (Context passes
    // `InstanceRecord::instance_id`). Distinguishes "the same conversation
    // continuing" from "a different conversation" — the physical-page
    // numbering alone cannot (the runtime's page free-list is reused across
    // sequences and is not required to hand out arithmetically adjacent
    // ids, e.g. {5, 9} is a perfectly valid two-page allocation).
    std::uint64_t sequence_id = 0;
    std::vector<std::uint32_t> token_ids;     // this fire's new tokens, in order
    std::vector<std::uint32_t> position_ids;  // absolute positions, parallel to token_ids
    std::vector<std::uint32_t> kv_pages;      // this member's full historical page list (flashinfer CSR convention)
    std::uint32_t kv_last_page_len = 0;       // final page fill count after this fire (0 => derive)

    bool has_rs_slot = false;   // false for a non-hybrid arch (no GDN / no rs_cache)
    std::uint32_t rs_slot_id = 0;
    bool rs_reset = false;      // RS_FLAG_RESET bit — fresh sequence, decoder resets

    // Explicit KV write descriptor (device-geometry `WSlot`/`WOff` ports,
    // Phase 2/1b review fix B): per-token PHYSICAL page id + in-page offset
    // to write this fire's new K/V into, instead of the decoder's implicit
    // append-at-next-position ring semantics. `has_write_desc` is set
    // whenever the resolved geometry carried a write descriptor AT ALL — it
    // must never be silently dropped; `MetalExecutor::forward` rejects it
    // explicitly (the paged-KV write kernel has no encoder integration in
    // this build, see forward.cpp) rather than ignoring it and running the
    // (wrong) implicit-append path.
    bool has_write_desc = false;
    std::vector<std::uint32_t> w_page;  // physical page id per token, size == token_ids.size()
    std::vector<std::uint32_t> w_off;   // in-page offset per token, size == token_ids.size()
    bool requires_paged = false;        // C3/device geometry even without explicit writes

    // Local indices into `token_ids` (NOT global) whose logits must be
    // materialized — the fire's `sampling_indices` slice, member-relative
    // (mirrors CUDA executor.cpp's `qo_begin + h_sidx[k]` convention, minus
    // the `qo_begin` offset since this desc is already member-scoped).
    std::vector<std::uint32_t> readout_local_indices;
};

// f32 logits materialized for this fire's readout rows, in
// `readout_local_indices` order — the exact buffer `PassInputs::logits`
// binds into interp Intrinsic(Logits)/Intrinsic(MtpLogits) roots.
struct LogitsOut {
    std::vector<float> data;  // [rows, vocab] row-major
    std::uint32_t rows = 0;
    std::uint32_t vocab = 0;
};

// Phase 1b (review fix B): the number of resident GDN conv+recurrent state
// slots `MetalExecutor::setup` really allocates (heap_layout.hpp `plan_heap`
// sizes the State region as `slots * per_slot_bytes`; the M=1 decode
// kernels always bind slot 0's fixed base offset, so growing this does not
// change the sealed M=1 decode path — see heap_bind.cpp's `bind::GdnCore`
// wiring). Small and fixed rather than derived from `max_forward_requests`:
// `recurrent_state` alone is ~1 MiB/slot/GDN-layer for qwen3.6 (18 GDN
// layers), so a large slot count would reserve hundreds of MiB to GiB of
// idle memory.  The paged command path uses these four slots concurrently;
// caps report exactly this value — never a larger, aspirational one — via
// `MetalExecutor::rs_slots()`.
inline constexpr std::uint32_t kPhase1bRsSlots = 4;
inline constexpr std::uint32_t kPagedMaxForwardRequests = kPhase1bRsSlots;
// Paged prompts run one correct N=1 GDN recurrence DAG per token inside one
// command buffer.  This bounds IO/scratch/logits allocation independently of
// the four concurrently-addressable recurrent-state slots.
inline constexpr std::uint32_t kPagedMaxForwardTokens = 64;

struct SetupConfig {
    std::string checkpoint_dir;  // HF snapshot dir (config.json + safetensors)
    std::string kernels_dir;     // compiled .metal library search dir
    std::string arch_name;       // read_model_facts() arch, for a truthful early reject
    std::uint32_t vocab_size = 0;      // config.json vocab_size, cross-checked vs the shipped geometry
    bool has_linear_attn = false;      // config-derived GDN/hybrid signal (qwen3.6 requires this)
    // Phase 1b/3 paged-KV bridge: the runtime's configured pool capacity
    // (cfg.batching.total_pages/kv_page_size) — MetalExecutor::setup()
    // allocates a REAL paged KV pool sized from these (see MetalExecutor::
    // setup_kv_pool), so copy_kv/resize_pool operate over genuine storage
    // matching what caps advertises, not an aspirational placeholder.
    std::uint32_t total_pages = 0;
    std::uint32_t kv_page_size = 0;
    std::uint32_t max_forward_tokens = 1;
    std::uint32_t max_forward_requests = 1;
    std::vector<std::uint8_t> load_plan;
    std::uint64_t compiler_version = 0;
    std::uint32_t storage_page_size = 1;
};

// Tracks one resident-state SLOT's logical-sequence bookkeeping. Exposed so
// tests can drive `validate_linear_sequence_geometry` without a live
// executor. Phase 1b state-slot fix: this is now tracked PER SLOT (a
// `slot_id -> LinearSequenceState` map in MetalExecutor) instead of one
// global record, because `copy_state` can now populate a DIFFERENT slot's
// metadata (sequence identity + next position + page-list prefix) without
// that slot being the one the shared M=1 KV ring currently backs.
struct LinearSequenceState {
    bool has_resident = false;
    std::uint64_t resident_sequence_id = 0;
    std::uint32_t resident_slot = 0;
    std::uint32_t resident_next_position = 0;
    // The full ordered KV page list backing the resident sequence, exactly
    // as last observed — a later fire's page list must carry this as a
    // literal prefix (§ below); ids need not be arithmetically adjacent.
    std::vector<std::uint32_t> resident_pages;
    // True iff this slot's GDN state is the one actually BACKING the shared
    // M=1 KV ring right now (reached via a real forward()/step() sequence
    // through this exact slot) — as opposed to merely holding valid,
    // correctly-tracked metadata because `MetalExecutor::copy_state` copied
    // it here. A slot can have `has_resident=true` with real, accurate
    // metadata (sequence id / next position / page-list prefix) yet
    // `ring_backed=false`: continuing THAT slot through the M=1 forward
    // path is impossible until its KV history is ALSO resident somewhere
    // (the not-yet-wired-into-forward paged-KV bridge) — the shared ring
    // only ever holds ONE sequence's actual K/V at a time, independent of
    // how many GDN state slots exist. `validate_linear_sequence_geometry`
    // rejects such a continuation attempt with a precise, distinct reason
    // rather than silently treating copied metadata as replay-ready.
    bool ring_backed = false;
    // True when this slot's history is backed by the NHD paged pool rather
    // than the legacy HND M=1 ring.  Both can be false for metadata copied
    // without a matching KV copy.
    bool paged_backed = false;
};

// Pure Phase 1a geometry gate (no Metal/decoder dependency — unit-testable).
// `state` is the caller-selected per-slot record for `desc.rs_slot_id` (or
// the implicit slot-0 record for a non-hybrid arch with no rs_slot at all).
// `other_slot_ring_backed_different_sequence` is precomputed by the caller
// (MetalExecutor::forward, which owns the full slot map) — true iff some
// OTHER slot besides this one is currently `ring_backed` for a DIFFERENT
// `sequence_id` than this fire's (only one slot may be ring-backed at a
// time, system-wide, since there is exactly one shared M=1 KV ring).
// Accepts exactly:
//   (a) a fresh sequence (`desc.rs_reset`, or — when the arch has no
//       rs_slot — `position_ids.front() == 0`): allowed only when
//       `!other_slot_ring_backed_different_sequence` — resetting while a
//       DIFFERENT sequence is ring-backed elsewhere would silently steal
//       the shared ring out from under it. The engine must
//       `close_sequence()` the old one first (Context's
//       `close_instance`) before a different sequence may go fresh.
//   (b) the exact contiguous continuation of the currently RING-BACKED
//       sequence at this slot: `state.ring_backed` must be true (a slot
//       whose metadata was merely copy_state'd here, never ring-backed,
//       cannot be "continued" — see `LinearSequenceState::ring_backed`),
//       same `sequence_id`, `position_ids.front() ==
//       state.resident_next_position`, and `desc.kv_pages` carries
//       `state.resident_pages` as a literal prefix (same ids, same order —
//       physical page NUMBERING is reused/non-adjacent across sequences by
//       design, e.g. {5, 9}; what must hold is that the prior pages are
//       still there, unmodified).
// Every page id across the fire's full list must be unique (no duplicates)
// — a repeated physical page within one sequence's list is a fork/share/
// corruption signal, not a valid single linear run. Positions within one
// fire must be contiguous ascending by 1 (`in-order positions`).
bool validate_linear_sequence_geometry(const LinearSequenceState& state,
                                       bool other_slot_ring_backed_different_sequence,
                                       const MemberForwardDesc& desc,
                                       std::string* reject_reason);

// Pure state transition backing `MetalExecutor::close_sequence` — releases
// residency in `state` if `sequence_id` is the one currently resident AND
// `state.ring_backed` (a no-op otherwise — in particular, a DIFFERENT
// slot's copy_state'd metadata that happens to share `sequence_id` is never
// touched, so "close of one sequence must not erase copied destination
// metadata"). Exposed standalone so it (and the "B accepted after A
// closes" sequence of events) is unit-testable without a live executor.
void close_linear_sequence(LinearSequenceState& state, std::uint64_t sequence_id);

// Phase 3 (metal_ptir_plan.md §7): the result of scheduling ONE launch batch
// of forward-needing members over the single shared M=1 KV ring. A batch may
// carry several members (mixed C1/C2, several C2). The ring holds exactly one
// sequence's KV at a time, so the members are SERIALIZED: the (at most one)
// member that CONTINUES the currently ring-backed sequence must run first
// (while the ring still holds its history), then every fresh member runs a
// reset+replay (each starting at position 0, so its causal SDPA reads only
// its own freshly-appended KV — independent of what any sibling member wrote,
// and of any stale higher-position ring bytes). Each member's readout logits
// are captured immediately after ITS OWN step run, before the next member
// clobbers the ring, so a serial pass produces correct per-member logits.
//
// What the single ring genuinely CANNOT serve (honestly gated, never faked):
//   * more than one member that must CONTINUE pre-existing KV (two distinct
//     sequences both needing their history resident at once), or a
//     continuation of a sequence that is not the currently ring-backed one —
//     these need the per-request paged-KV pool + a paged decode DAG (the
//     checkpoint/hardware-gated multi-request path), so they are marked not
//     serviceable with a precise reason instead of silently corrupting state;
//   * an explicit per-token KV write descriptor (w_page/w_off) — no encoder
//     integration dispatches the paged write kernel against a live forward in
//     this build (see MetalExecutor::forward), so it is gated too.
struct BatchExecPlan {
    // Member indices (into the input `descs`) to execute, in ring-safe order:
    // the leading ring-backed continuation (if any) first, then fresh members
    // in their original order. Only serviceable members appear here.
    std::vector<std::size_t> order;
    // Parallel to the input `descs`: 1 iff the member can be served by the
    // single-ring serial path (and therefore appears in `order`), else 0.
    std::vector<std::uint8_t> member_ok;
    // Parallel to the input `descs`: a precise reason when `member_ok[i]==0`.
    std::vector<std::string> member_reason;
};

// Pure (host-testable, no Metal/decoder dependency) batch scheduler. `slot_states`
// is the executor's current per-slot residency map (rs_slot_id -> state; at most
// one entry `ring_backed`). Decides ORDER + per-member serviceability per the
// contract in `BatchExecPlan`. Per-member GEOMETRY faults (empty span, non-
// monotone positions, duplicated pages, a continuation that does not extend the
// resident prefix) are NOT decided here — they are surfaced by
// `validate_linear_sequence_geometry` when the member actually runs; this
// scheduler only arbitrates the shared-ring CONCURRENCY question.
BatchExecPlan plan_batch_execution(
    const std::unordered_map<std::uint32_t, LinearSequenceState>& slot_states,
    const std::vector<MemberForwardDesc>& descs);

// CUDA-compatible member-local sampling indices become rows in the concatenated
// paged batch by adding the member's qo_indptr begin.
std::vector<std::uint32_t> global_readout_rows(
    std::uint32_t qo_begin, const std::vector<std::uint32_t>& local_indices);

class MetalExecutor {
  public:
    MetalExecutor();
    ~MetalExecutor();
    MetalExecutor(const MetalExecutor&) = delete;
    MetalExecutor& operator=(const MetalExecutor&) = delete;

    // One-time lifecycle: load the qwen3.6 checkpoint via MetalExecutor.
    // Refuses truthfully (no lying caps) when `cfg` does not match the
    // shipped geometry this increment supports.
    bool setup(const SetupConfig& cfg, std::string* err);

    bool ready() const;
    std::uint32_t vocab() const;

    // One member's forward for this fire: validates the Phase 1a linear-
    // sequence contract, advances the resident decoder (reset+replay for a
    // fresh sequence, or an incremental `step()` run for a continuation),
    // and materializes f32 logits for `desc.readout_local_indices`.
    bool forward(const MemberForwardDesc& desc, LogitsOut& out, std::string* err);

    // Forward an entire launch batch in one paged command buffer.  Member
    // geometry is concatenated into token/request CSR rows, state is selected
    // by SlotOfToken, and full-attention layers use the separate NHD page pool.
    // One ordinary linear M=1 member retains the sealed HND-ring fast path.
    //
    // Outputs are all parallel to `descs` (index i ↔ member i):
    //   * `outs[i]`   — that member's logits (valid iff `success[i]`).
    //   * `success[i]`— true iff member i's forward succeeded; false members
    //                   carry a precise reason in `errors[i]` and MUST be
    //                   poisoned per-member by the caller (never the batch).
    //   * `errors[i]` — reject/fault reason when `!success[i]` (empty on ok).
    // The sealed single-forward-member case is byte-identical to the old
    // per-member path (one member, no sibling to serialize against).
    void forward_batch(const std::vector<MemberForwardDesc>& descs,
                       std::vector<LogitsOut>& outs,
                       std::vector<std::uint8_t>& success,
                       std::vector<std::string>& errors);

    // Releases residency if `sequence_id` is the one currently ring-backed
    // (a no-op otherwise — closing a sequence that never ran, one that
    // isn't the currently ring-backed one, or a slot that merely holds
    // copy_state'd metadata for that sequence id, must not disturb
    // residency or erase that copied metadata). Call before erasing an
    // instance (`Context::close_instance`) so a later FRESH session is
    // not rejected as "another sequence is resident".
    void close_sequence(std::uint64_t sequence_id);

    // Phase 1b: how many resident recurrent-state (GDN) slots the decoder's
    // heap actually allocated (0 before `setup()`, or for a non-hybrid
    // checkpoint with no GDN layers at all). Real, not aspirational — the
    // heap genuinely reserves `rs_slots() * per_slot_bytes` for conv+
    // recurrent state per GDN layer (heap_layout.hpp `plan_heap`); caps
    // reports exactly this value, never a larger, unsupported one.
    std::uint32_t rs_slots() const;
    std::uint64_t rs_slot_bytes() const;

    // Copies one GDN layer's-worth (every GDN layer) resident conv+
    // recurrent state from `src_slot` to `dst_slot` (whole-slot; per-token
    // sub-ranges are not supported by either backend today — CUDA's own
    // `RsCache::copy_slot_d2d` is likewise whole-slot only). Real, tested
    // memory movement over Shared-storage (unified-memory) regions — not
    // gated on the (unimplemented) paged-KV forward bridge, since GDN state
    // resides in its own always-real region regardless of KV storage mode.
    // ALSO copies `src_slot`'s tracked sequence metadata (sequence id, next
    // position, page-list prefix) to `dst_slot` — WITHOUT marking `dst_slot`
    // ring-backed (copying bytes does not make the shared ring hold that
    // slot's KV history too) — so a later fire that presents `dst_slot`
    // with the correct next position is recognized (not silently treated
    // as garbage) even though it is honestly rejected as "not ring-backed"
    // until the paged-KV bridge can back it independently. If `src_slot`
    // has no tracked metadata (never forwarded/reset), any stale metadata
    // at `dst_slot` is cleared instead of copied (the destination's bytes
    // no longer correspond to whatever metadata used to be there).
    bool copy_state(std::uint32_t src_slot, std::uint32_t dst_slot, std::string* err);

    // Phase 1b/3 paged-KV bridge: real, page-addressable KV pool queries +
    // control ops — narrow methods so context.cpp never needs to include
    // MetalExecutor/Metal types directly.
    std::uint32_t kv_pool_total_pages() const;
    std::uint32_t kv_pool_page_size() const;

    // One per-token KV cell move (mirrors PieKvMoveCell exactly).
    struct KvMoveCell {
        std::uint32_t dst_page_id, dst_token_offset, src_page_id, src_token_offset;
    };

    // Whole-page copy: `src_pages[i] -> dst_pages[i]`, every full-attention
    // layer, K and V both. Real memcpy over the Shared-storage paged pool.
    bool copy_kv_pages(const std::vector<std::uint32_t>& src_pages,
                       const std::vector<std::uint32_t>& dst_pages, std::string* err);

    // Per-token cell copy (PieKvMoveCell semantics), every full-attention layer.
    bool copy_kv_cells(const std::vector<KvMoveCell>& cells, std::string* err);

    // Grow (always allowed, page ids stable) or logically shrink (only when
    // `unmapped_tail_pages` attests the truncated pages are free) the paged
    // KV pool to `new_total_pages`.
    bool resize_kv_pool(std::uint32_t new_total_pages, bool unmapped_tail_pages, std::string* err);

  private:
    friend struct NativeAccess;

    bool setup_native(
        const std::string& checkpoint_dir,
        const std::string& kernels_dir,
        const ::pie::metal::DecodeGeometry& geometry,
        std::string* error);
    bool setup_kv_pool_native(
        std::uint32_t total_pages,
        std::uint32_t page_size,
        std::string* error);
    void reset_state_native();
    void reset_state_native(std::uint32_t slot);
    bool copy_state_slot_native(
        std::uint32_t src_slot,
        std::uint32_t dst_slot,
        std::string* error);
    ::pie::metal::StepTiming step_native(
        std::uint32_t token_id,
        std::uint32_t position,
        std::uint32_t slot);
    bool run_batch_step_native(
        const ::pie::metal::BatchSchedule& schedule,
        const ::pie::metal::BatchStepInputs& inputs,
        std::string* error);
    std::uint64_t paged_bind_generation_native() const;
    const ::pie::metal::KvPagePool& kv_pool_native() const;
    int vocab_native() const;
    void copy_logits_f32_native(float* output) const;
    void copy_batch_logits_f32_native(
        std::uint32_t token_row,
        float* output) const;
    std::uint32_t argmax_native() const;

    // Shared body of `forward` (single member) and `forward_batch` (one member
    // at a time, in the scheduled order). `batch_serialized` = true forces the
    // cross-sequence "another sequence is ring-backed" gate OFF: within a batch
    // the ring is deliberately clobbered member-to-member, and the shared-ring
    // concurrency arbitration is `plan_batch_execution`'s job, not this pure
    // per-member geometry check's. The single-member `forward` passes false, so
    // its sealed cross-launch semantics are unchanged.
    bool run_member_forward(const MemberForwardDesc& desc, LogitsOut& out,
                            bool batch_serialized, std::string* err);
    bool run_paged_batch_forward(const std::vector<MemberForwardDesc>& descs,
                                 std::vector<LogitsOut>& outs,
                                 std::vector<std::uint8_t>& success,
                                 std::vector<std::string>& errors);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    // Keyed by rs_slot_id (or 0 for a non-hybrid arch with no rs_slot at
    // all). At most one entry may have `ring_backed == true` at a time.
    std::unordered_map<std::uint32_t, LinearSequenceState> slot_states_;
    std::uint32_t vocab_ = 0;
};

}  // namespace pie::metal::batch
