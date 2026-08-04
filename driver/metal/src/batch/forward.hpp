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
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "loader/load_plan.hpp"
#include "pie/driver/fire/geometry.hpp"

namespace pie::metal {

struct BatchSchedule;
struct BatchStepInputs;
struct DecodeGeometry;
struct KvPagePool;
struct StepTiming;
class StepEncoder;
class RawMetalContext;
struct SlotHandle;
struct WeightBytes;

}  // namespace pie::metal

namespace pie::metal::batch {


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
    std::vector<std::uint32_t> qo_indptr;
    std::vector<std::uint32_t> kv_page_indptr;
    std::vector<std::uint32_t> kv_last_page_lens;

    bool has_rs_slot = false;   // false for a non-hybrid arch (no GDN / no rs_cache)
    std::uint32_t rs_slot_id = 0;
    bool rs_reset = false;      // RS_FLAG_RESET bit — fresh sequence, decoder resets
    std::vector<std::uint32_t> request_rs_slot_ids;
    std::vector<std::uint8_t> request_rs_reset;
    std::vector<std::uint8_t> request_rs_read;
    std::vector<std::uint8_t> request_rs_write;

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
    bool has_attention_mask = false;
    std::uint32_t attention_mask_stride = 0;
    std::vector<std::uint8_t> attention_mask;
    pie::driver::fire::StructuredMaskDescriptor structured_mask;

    // Local indices into `token_ids` (NOT global) whose logits must be
    // materialized — the fire's `sampling_indices` slice, member-relative
    // (mirrors CUDA executor.cpp's `qo_begin + h_sidx[k]` convention, minus
    // the `qo_begin` offset since this desc is already member-scoped).
    std::vector<std::uint32_t> readout_local_indices;
    std::vector<std::uint32_t> sampling_indptr;
    std::uint32_t row_count = 1;
    std::uint32_t sampled_rows = 0;
    std::uint32_t token_count = 0;
    std::uint32_t page_count = 0;
    std::uint32_t query_len = 0;
    std::uint32_t key_len = 0;
    std::uint32_t kv_len = 0;
    // Prefer the device's exact greedy token for every readout row and skip
    // full-logits staging when the family supports it. Other families retain
    // the existing logits path, so this is an optimization request, not a
    // capability requirement.
    bool greedy_token_only = false;
};

// f32 logits materialized for this fire's readout rows, in
// `readout_local_indices` order. Production consumes the bf16 device view;
// this CPU form remains only for test support.
struct LogitsOut {
    // Test-only CPU materialization. Production M1 leaves this empty and binds
    // the bf16 Shared-storage view below directly.
    std::vector<float> data;
    std::uint32_t rows = 0;
    std::uint32_t vocab = 0;
    void* device_buffer = nullptr;
    void* device_contents = nullptr;
    std::uint64_t device_gpu_address = 0;
    std::uint64_t device_bytes = 0;
    std::uint32_t device_row_offset = 0;
    // Optional Shared-storage device-greedy result. Valid until the next fire,
    // like the logits view; `greedy_row_offset` selects this member's rows.
    const std::uint32_t* greedy_contents = nullptr;
    std::uint32_t greedy_row_offset = 0;
};

struct PtirCommandCallbacks {
    std::function<void(::pie::metal::StepEncoder&)> pre_forward;
    std::function<void(::pie::metal::StepEncoder&)> post_forward;
    std::function<void(std::uint32_t)> set_logits_row;
    std::function<void(const std::vector<std::uint32_t>&)>
        set_logits_rows;
    std::function<void()> finalize_group;
    bool consumes_logits_directly = false;
};

std::vector<PtirCommandCallbacks> compact_ptir_callbacks(
    const std::vector<MemberForwardDesc>& descs,
    const std::vector<std::size_t>& accepted_members,
    const std::vector<std::uint32_t>& accepted_token_bases,
    const std::vector<PtirCommandCallbacks>& callbacks);

bool validate_request_local_positions(
    const MemberForwardDesc& desc,
    std::string* error);

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


// Concurrent recurrent-state slots, and through `kPagedMaxForwardRequests` the
// driver's advertised `max_forward_requests` -- which bootstrap also adopts as
// the process admission cap, so this one number decides how wide a decode batch
// can get AND how many requests may be in flight.
//
// A slot is 21.4MB here (18 GDN layers x (2 x conv + recurrent)), so 32 costs
// 684MB against a 405MB checkpoint on a 34GB machine.  Worth it: 32 lanes turn
// in 812 tok/s against 16 lanes' 698 on the same binary, bit-identical output.
//
// It also has to be >= any concurrency the deployment expects: asking for more
// concurrent requests than this hangs the cold-start seal rather than queueing
// (32 requests at concurrency 16 are fine; 17 at concurrency 17 never starts).
// That is a separate defect -- oversubscription should queue -- but until it is
// fixed this bound is also a floor on usable concurrency.
inline constexpr std::uint32_t kPhase1bRsSlots = 64;
inline constexpr std::uint32_t kPagedMaxForwardRequests = kPhase1bRsSlots;

/// What the recurrent-state slots are allowed to cost, in bytes.
///
/// Sixty-four slots was a COUNT, chosen where a slot was 21 MB and sixty-four
/// of them 1.37 GB. A slot's size is the model's, not a constant: at forty
/// layers, thirty-two value heads and a 128-wide state, one slot is 67 MiB and
/// sixty-four are 4.3 GiB -- which is what stood between Qwen3.5-35B-A3B and a
/// machine that would otherwise have held it. So the BUDGET is the constant,
/// because it is the thing that was actually being chosen, and the count is
/// derived from what a slot costs for the checkpoint in hand.
inline constexpr std::uint64_t kRsSlotBudgetBytes = 1536ull << 20;

/// Bytes one recurrent-state slot costs for this geometry: conv in, conv out
/// and the recurrent state, for every GDN layer.
///
/// The same formula as `MetalExecutor::rs_slot_bytes()`, which needs a live
/// executor and so cannot answer before setup -- and answering before setup is
/// the whole point. Kept beside the slot count it feeds.
std::uint64_t rs_slot_bytes_for(const DecodeGeometry& g);

/// How many slots to reserve: as many as the budget buys, never more than
/// `kPhase1bRsSlots` and never fewer than `floor_slots`.
///
/// ONE function, called by the capabilities pass and by setup, because a
/// capability advertising more slots than setup allocates is a request the
/// driver accepts and cannot hold -- the same rule, and the same reason, as
/// `simple_family_max_forward_tokens`.
std::uint32_t rs_slots_for_budget(const DecodeGeometry& g, std::uint64_t budget_bytes,
                                  std::uint32_t floor_slots);

// Tokens the resident KV/GDN ring holds, across the WHOLE fleet -- it is one
// linear ring, not a per-request allocation, so the whole fleet shares it. At
// 4096 that ceiling was reached by sixteen requests generating ~230 tokens
// each, and the planner failed them all with `NoSwapRoom`; a concurrent fleet
// could not run a normal-length generation at all.
//
// Derived from the slot count rather than written down, because the two are the
// same statement about how many requests this driver serves: a ring sized for
// sixteen requests starves the moment the slot count allows sixty-four, which
// is exactly what happened -- 64 x 512 tokens needs ~35k and a 32768 ring
// failed every request with `NoSwapRoom`.  Each request gets 2048 tokens.
//
// The KV region is `n_full_attn * 2 * n_kv_heads * max_ctx * head_dim * 2B`,
// which is ~100MB for this checkpoint at 4096, ~800MB at 32768 and ~3.2GB at
// 131072 -- worth it against a 405MB weight set on a machine with tens of GB.
//
// ADVERTISED and ENFORCED from here: `context.cpp` builds the capabilities page
// count from this and `validate_fire_geometry` bounds page ids by the same
// number, so the two can never drift (they were separate 4096 literals).
inline constexpr std::uint32_t kMetalCtxTokensPerRequest = 2048;
inline constexpr std::uint32_t kMetalMaxCtxTokens =
    kPhase1bRsSlots * kMetalCtxTokensPerRequest;
// How many prompt rows one fire may carry.
//
// This is not just an allocation bound: it is advertised through capabilities
// as `max_forward_tokens`, and `FramePolicy` defers any lane whose fire does
// not fit the wave budget, so it is also what decides how many of a fleet's
// prompts can share a prefill.  At 64 -- the value this started at, sized for a
// single stream -- sixteen 34-token prompts could not pair up and ran as
// sixteen separate fires.
//
// So it has to come from what actually constrains it, which is memory per row.
// A row costs, per fire:
//
//   * `vocab * 2` bytes of logits -- 485KB for this checkpoint, and the term
//     that dominates every other by two orders of magnitude;
//   * `scratch_widest_elems * 2` bytes in each of the scratch pool's colors;
//   * a handful of 4-byte per-row IO scalars.
//
// `paged_max_forward_tokens` divides a budget by that, so a small-vocabulary
// model is allowed more rows and a large one fewer, instead of every model
// inheriting a number tuned against one checkpoint.  The floor keeps a single
// long prompt working.
//
// The ceiling is not a throughput optimum.  512 was, measured: sixteen 34-token
// prompts take 6 fires at 256 and 4 at both 512 and 1024, so nothing above 512
// buys a batch this wide anything.  But rows are also the longest prompt this
// driver will ACCEPT -- a longer one is refused, not chunked -- so the ceiling
// has to be sized for the longest prompt, not for the widest batch.  At 512 a
// 650-token prompt was refused by every family.  1024 is what the 1GB budget
// below affords this checkpoint (677KB per row), and it is free: measured
// through `pie serve`, 566 rows and 1024 rows give the same 2.35GB peak RSS and
// the same wall clock, because the pool is a heap RESERVATION and a fire
// touches only the rows it has.
//
// `kPrefillOrdinalMaxRows` must be raised with this: qwen3.5's prefill builds
// one DAG per row and claims an ordinal block for each, and PTIR's ordinal base
// is derived from where that range ends.
inline constexpr std::uint64_t kPagedForwardRowBudgetBytes = 1024ull << 20;
// The scratch coloring is computed from the DAG, which does not exist yet when
// capabilities are built.  A generous fixed count is fine here: at 12KB per
// color against 485KB of logits, the whole scratch term is noise in this
// division, and over-counting it can only make the answer conservative.
inline constexpr std::uint32_t kPagedScratchColors = 16;
inline constexpr std::uint32_t kPagedMinForwardTokens = 64;
inline constexpr std::uint32_t kPagedMaxForwardTokensCeiling = 1024;
/// The bound for a family whose row budget is DERIVED rather than priced.
///
/// 512 is what a per-row price of `vocab * 2` buys, and that price stopped
/// being true for these families when `Kind::RowGather` moved the LM head onto
/// the sampled rows only. A prompt longer than the ceiling is refused, not
/// chunked, so the ceiling is a hard bound on what can be run -- worth deriving
/// from the pool that will actually be allocated. This caps that derivation:
/// past a few thousand rows a fire is no longer a prompt, and the argument
/// tables are per-row.
inline constexpr std::uint32_t kPagedMaxForwardTokensHardCeiling = 4096;
// Every row claims a block of argument-table ordinals, so this ceiling also
// fixes where the prefill's ordinal space ends and PTIR's may begin; see
// `kPrefillOrdinalLimit`, which the setup path cross-checks against this.

inline std::uint32_t paged_max_forward_tokens(std::uint32_t vocab,
                                              std::uint32_t scratch_widest_elems,
                                              std::uint32_t scratch_colors,
                                              std::uint64_t budget_bytes =
                                                  kPagedForwardRowBudgetBytes) {
    const std::uint64_t per_row = std::uint64_t(vocab) * 2u +
                                  std::uint64_t(scratch_widest_elems) * 2u *
                                      std::max<std::uint32_t>(1, scratch_colors) +
                                  64u;  // per-row IO scalars
    const std::uint64_t rows =
        per_row == 0 ? kPagedMaxForwardTokensCeiling : budget_bytes / per_row;
    return std::clamp<std::uint32_t>(static_cast<std::uint32_t>(rows),
                                     kPagedMinForwardTokens,
                                     kPagedMaxForwardTokensCeiling);
}

/// The row budget for the families `SimpleFamilyEngine` serves.
///
/// They allocate their activation pool for `max_forward_tokens` ROWS at setup,
/// so what is advertised and what is allocated have to be the SAME number --
/// advertise more and the driver accepts a fire it cannot hold; allocate more
/// and an 11.8 GB checkpoint fails to create its heap over a fire nobody asked
/// for. One function, called from both places.
// The rows-per-fire bound for a `SimpleFamilyEngine` family is DERIVED from the
// activation pool it will actually allocate; see `simple_family_row_budget` in
// context.cpp. There is no fixed ceiling for them to consult here.

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
    // The load *request*, not a compiled plan: `setup()` calls the loader
    // itself, because only the driver knows the device
    // (`loader/architecture.md` §3).
    std::string snapshot_dir;
    // Page this model's routed experts in from a mapping instead of keeping
    // them resident in the heap.
    //
    // The same `[model].stream_routed_experts` the CUDA driver reads. It is
    // one switch because it is one decision -- a residency trade the operator
    // makes about a model, not about a backend. It used to be
    // `PIE_METAL_STREAM_EXPERTS` here, which meant setting the config on a
    // Metal backend did nothing and said nothing.
    bool stream_routed_experts = false;
    /// How many bytes the routed experts may occupy on the device. Zero means
    /// the whole bank stays resident, which is what every model that fits
    /// should do.
    ///
    /// Non-zero is the only setting under which a model can exceed the
    /// machine, and it is a different mechanism from `stream_routed_experts`
    /// rather than a stronger version of it: streaming maps the bank and every
    /// mapped page is WIRED, so it bounds nothing. A budget turns mapping off
    /// and pages the experts through a slab of this size instead, which costs
    /// a submit-and-wait per mixture layer. Set it only when the alternative
    /// is not running at all.
    std::uint64_t expert_slab_bytes = 0;
    // Which storage schema to author against. It selects a contract on this
    // side of the loader call and never crosses it (§10.4).
    std::string model_type;

    /// `config.json`'s `quantization` block. Not per-family, because it
    /// describes the FILE and not the architecture -- and not recoverable from
    /// the tensors, because 8 bits in groups of 64 and 4 bits in groups of 128
    /// pack identically. Zero means the config declared none.
    int quant_bits = 0;
    int quant_group_size = 0;
    /// Gemma 4's shape, when `model_type` says so. Zero means "not gemma4", so
    /// a config that never mentions this family cannot accidentally select it.
    struct Gemma4Facts {
        int n_layers = 0;
        int hidden = 0;
        int intermediate = 0;
        int n_q_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;         // sliding layers
        int global_head_dim = 0;  // full layers
        int sliding_window = 0;
        int num_kv_shared_layers = 0;
        int per_layer_emb_dim = 0;
        int full_attn_interval = 0;  // -1: `layer_types` is irregular, refuse
        bool double_wide_mlp = false;
        float final_softcap = 0.0f;
        float rope_theta_full = 1.0e6f;
        float rope_theta_sliding = 1.0e4f;
        float full_partial_rotary = 0.25f;
        // The mixture, and the k-eq-V attention that comes with it on the 26B.
        // Zero and false on every dense member, which is how the geometry tells
        // a dense gemma 4 from a routed one.
        bool enable_moe = false;
        int n_experts = 0;
        int experts_per_token = 0;
        int moe_intermediate = 0;
        bool attention_k_eq_v = false;
        int n_global_kv_heads = 0;
        bool present() const { return n_layers > 0 && hidden > 0; }
    } gemma4;
    /// GPT-OSS's shape, when `model_type` says so. Zero means "not gpt-oss",
    /// on the same principle: a config that never mentions this family cannot
    /// accidentally select it.
    struct GptOssFacts {
        int n_layers = 0;
        int hidden = 0;
        int vocab = 0;
        int n_q_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int sliding_window = 0;
        int n_experts = 0;
        int experts_per_token = 0;
        int intermediate = 0;
        int rope_original_max_position = 4096;
        float eps = 1e-5f;
        float swiglu_limit = 7.0f;
        float rope_theta = 150000.0f;
        float rope_factor = 32.0f;
        float rope_beta_fast = 32.0f;
        float rope_beta_slow = 1.0f;
        bool present() const { return n_layers > 0 && hidden > 0; }
    } gptoss;
    /// The llama-shaped families' shape, when `model_type` says so. Zero means
    /// "not one of them", on the same principle as the two above.
    ///
    /// One struct for `llama`, `llama3`, `mistral`, `qwen2`, `qwen3`,
    /// `qwen2_moe` and `qwen3_moe`, because those differ in two FIELDS and not
    /// in shape: whether q and k are normed, and whether the FFN is routed.
    /// Splitting them would be seven copies of the same fifteen integers.
    struct LlamaFacts {
        int n_layers = 0;
        int hidden = 0;
        int vocab = 0;
        int n_q_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int intermediate = 0;
        int n_experts = 0;
        int experts_per_token = 0;
        int moe_intermediate = 0;
        float eps = 1e-5f;
        float rope_theta = 500000.0f;
        float rope_scale = 1.0f;
        /// `rope_scaling.rope_type`, verbatim. Empty or "linear"/"default" is
        /// implemented; anything else -- Llama 3.1's piecewise schedule above
        /// all -- is REFUSED by the geometry rather than approximated by
        /// `rope_scale`, which runs and is wrong past the original context.
        std::string rope_scaling_kind;
        /// Llama 3.1's three extra knobs. `rope_scale` carries its `factor`.
        float rope_low_freq_factor = 1.0f;
        float rope_high_freq_factor = 4.0f;
        int rope_original_max_position = 8192;
        /// Set when the checkpoint ships `self_attn.q_norm`. A config fact
        /// rather than a model_type one: `qwen3` has it and `qwen2` does not,
        /// and both are this family.
        bool qk_norm = false;
        bool tied_embeddings = true;
        /// `norm_topk_prob`. True means the routing weights are renormalized
        /// over the selected experts, which is what `router_topk` computes.
        /// Defaults to true because a config that omits it (Mixtral, gpt-oss)
        /// means it; only an explicit false is a model this driver refuses.
        bool norm_topk_prob = true;
        bool present() const { return n_layers > 0 && hidden > 0; }
    } llama;

    /// Qwen3.5 / Qwen3-Next: the GDN hybrid's shape.
    ///
    /// This family used to have no facts at all. Its `DecodeGeometry` was
    /// default-constructed and the defaults were one preview checkpoint's
    /// dimensions, so the driver ran that checkpoint and mis-ran every other
    /// one -- silently, because nothing in the path ever compared a config
    /// against what it was about to execute.
    struct Qwen35Facts {
        int n_layers = 0;
        int hidden = 0;
        int vocab = 0;
        int n_q_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int intermediate = 0;
        /// The linear-attention block's head counts and dims. The convolution
        /// width and the value total are derived from them by the geometry.
        int gdn_k_heads = 0;
        int gdn_v_heads = 0;
        int gdn_k_dim = 0;
        int gdn_v_dim = 0;
        int gdn_conv_k = 0;
        /// One full-attention layer every `interval`. -1 means the config
        /// listed an irregular pattern, which the geometry refuses.
        int full_attn_interval = 0;
        int n_experts = 0;
        int experts_per_token = 0;
        int moe_intermediate = 0;
        int shared_expert_intermediate = 0;
        int decoder_sparse_step = 1;
        int mlp_only_layer_count = 0;
        float eps = 1e-6f;
        bool tied_embeddings = true;
        bool norm_topk_prob = true;
        bool present() const { return n_layers > 0 && hidden > 0; }
    } qwen35;

    /// How many tokens the KV ring must hold, across ALL resident requests.
    ///
    /// Zero means `kMetalMaxCtxTokens`: a ring sized for a full fleet, which is
    /// what `pie serve` wants and what every caller got when this was a
    /// constant. It stopped being affordable as a constant. The ring does not
    /// scale with the model, so at 48 layers it is 13 GiB of KV whatever the
    /// weights are -- fine beside a 405 MB checkpoint, and the difference
    /// between running and not beside a 17 GiB one. A caller that knows it
    /// drives ONE sequence should not pay for sixty-four.
    std::uint32_t max_ctx_tokens = 0;
    // `config.json`'s RoPE hyperparameters, read out of the nested
    // `rope_parameters` object this family uses (context.cpp). The defaults
    // below are Qwen3.5's, so a checkpoint that omits them still lands on the
    // values the reference implementation applies.
    float rope_theta = 1.0e7f;
    float partial_rotary_factor = 0.25f;
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

bool validate_paged_request_state(
    const std::unordered_map<std::uint32_t, LinearSequenceState>& states,
    const MemberForwardDesc& desc,
    std::size_t request,
    std::string* error);
void commit_paged_request_state(
    std::unordered_map<std::uint32_t, LinearSequenceState>& states,
    const MemberForwardDesc& desc,
    std::size_t request);

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

    /// Resident weight bytes, and how many of them one decoded token reads.
    /// A fact about the heap, not a formula over a config -- see
    /// `weight_bytes` in `loader/heap_bind_metal.hpp`.
    WeightBytes weight_bytes() const;

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
                       std::vector<std::string>& errors,
                       const std::vector<PtirCommandCallbacks>* ptir = nullptr);

    ::pie::metal::RawMetalContext* command_context();
    ::pie::metal::SlotHandle logits_device_slot() const;

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
    std::uint64_t elastic_page_bytes() const;
    std::uint64_t elastic_budget_pages() const;
    std::uint64_t elastic_committed_pages() const;
    bool ensure_launch_storage(
        std::uint32_t kv_pages,
        std::uint32_t state_slots,
        std::uint32_t token_rows,
        std::string* error);

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
    std::uint32_t kv_pool_committed_pages() const;
    std::uint32_t kv_pool_page_size() const;
    bool ensure_kv_pages(std::uint32_t pages, std::string* error);

    // One per-token KV cell move (mirrors PieKvMoveCell exactly).
    struct KvMoveCell {
        std::uint32_t dst_page_id, dst_token_offset, src_page_id, src_token_offset;
    };

    // Whole-page copy: `src_pages[i] -> dst_pages[i]`, every full-attention
    // layer, K and V both, through chunk alias views of the sparse pool.
    bool copy_kv_pages(const std::vector<std::uint32_t>& src_pages,
                       const std::vector<std::uint32_t>& dst_pages, std::string* err);

    // Per-token cell copy (PieKvMoveCell semantics), every full-attention layer.
    bool copy_kv_cells(const std::vector<KvMoveCell>& cells, std::string* err);

    // Grow or trim physical backing without replacing the sparse buffer or
    // rebinding its stable GPU address.
    bool resize_kv_pool(std::uint32_t new_total_pages, bool unmapped_tail_pages, std::string* err);
    bool resize_elastic_pool(
        std::uint64_t pool_id,
        std::uint64_t target_pages,
        std::string* err);

  private:
    // Shared body of `forward` (single member) and `forward_batch` (one member
    // at a time, in the scheduled order). `batch_serialized` = true forces the
    // cross-sequence "another sequence is ring-backed" gate OFF: within a batch
    // the ring is deliberately clobbered member-to-member, and the shared-ring
    // concurrency arbitration is `plan_batch_execution`'s job, not this pure
    // per-member geometry check's. The single-member `forward` passes false, so
    // its sealed cross-launch semantics are unchanged.
    bool run_member_forward(const MemberForwardDesc& desc, LogitsOut& out,
                            bool batch_serialized, std::string* err,
                            const PtirCommandCallbacks* ptir = nullptr);
    /// The paged batch path for the families `SimpleFamilyEngine` serves.
    ///
    /// Several requests share ONE fire: their tokens are concatenated, the CSR
    /// says who owns which rows, and each request's history is its own page
    /// list. Prefill rows and decode rows differ only in how many a request
    /// contributes, so a mixed batch needs nothing further.
    bool run_simple_batch_forward(const std::vector<MemberForwardDesc>& descs,
                                  std::vector<LogitsOut>& outs,
                                  std::vector<std::uint8_t>& success,
                                  std::vector<std::string>& errors,
                                  const std::vector<PtirCommandCallbacks>* ptir = nullptr);
    bool run_paged_batch_forward(const std::vector<MemberForwardDesc>& descs,
                                 std::vector<LogitsOut>& outs,
                                 std::vector<std::uint8_t>& success,
                                 std::vector<std::string>& errors,
                                 const std::vector<PtirCommandCallbacks>* ptir = nullptr);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    // Keyed by rs_slot_id (or 0 for a non-hybrid arch with no rs_slot at
    // all). At most one entry may have `ring_backed == true` at a time.
    std::unordered_map<std::uint32_t, LinearSequenceState> slot_states_;
    std::uint32_t vocab_ = 0;
};

}  // namespace pie::metal::batch
