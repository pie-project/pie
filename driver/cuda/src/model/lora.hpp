#pragma once

#include <cstdint>

namespace pie_cuda_driver::model {

/// The `lora` sink's SITES vocabulary: one bit per projection site, in the
/// model's own terms (PTIR overview §6.5 — "placement is structure"). The
/// vocabulary is model-level; the bit assignment below is the llama-like
/// family's, fixed here so a traced program and the consuming forward agree
/// on the same integers.
///
/// v0 consumes `q` and `v` only. The others are RESERVED: a lane naming one
/// of them binds fine (the bind gate checks only `has_lora`) but is refused
/// loudly at first use in the forward — a silently ignored site would be a
/// request whose adapter never applied while every sample still returned.
enum LoraSite : std::uint64_t {
    kLoraSiteQ      = 1ull << 0,  // q_proj      (consumed)
    kLoraSiteK      = 1ull << 1,  // k_proj      (reserved)
    kLoraSiteV      = 1ull << 2,  // v_proj      (consumed)
    kLoraSiteO      = 1ull << 3,  // o_proj      (reserved)
    kLoraSiteGateUp = 1ull << 4,  // gate/up     (reserved)
    kLoraSiteDown   = 1ull << 5,  // down_proj   (reserved)
};
inline constexpr std::uint64_t kLoraSitesKnown =
    kLoraSiteQ | kLoraSiteK | kLoraSiteV | kLoraSiteO |
    kLoraSiteGateUp | kLoraSiteDown;
inline constexpr std::uint64_t kLoraSitesConsumed = kLoraSiteQ | kLoraSiteV;

/// One lane's resolved `lora` sink: where this program's adapter weights live
/// and where the forward should apply them.
///
/// `lora` is a pass-wide *configuration* sink (PTIR overview §6.5): the
/// program hands the backend `lora(A, B, SITES)` in its PROLOGUE and the whole
/// forward applies the low-rank delta `W'x = Wx + B(Ax)` at the declared
/// projection sites. Unlike `attn_page_mask` — whose effect is a device write
/// into a body-owned buffer at the layer hook — this sink's effect is
/// host-side table construction: the prologue executes in `Dispatch::begin`,
/// BEFORE the model body runs, so the dispatch resolves the sink into this
/// launch-owned entry and the body reads it for the duration of the fire. No
/// device kernel runs for the sink itself.
struct LoraLaneView {
    /// Device address of the A channel's committed cell,
    /// `[num_layers, R, d]` in the adapter's trace-known geometry. Contents
    /// are per-instance data (an adapter swap is a channel re-seed, never a
    /// re-trace), so the address is the committed cell at begin time — the
    /// same cell the lane's prologue reads.
    const void* a = nullptr;

    /// Device address of the B channel's committed cell,
    /// `[num_layers, d_out, R]`. The LoRA scale `alpha/R` is folded into
    /// these contents; there is no scalar to carry here.
    const void* b = nullptr;

    /// The SITES placement constant — a trace-known bitmask over the model's
    /// site vocabulary (q/k/v/o/up/..). Placement is structure (part of the
    /// traced program), weights are contents; that is why this is a value and
    /// the other two are addresses.
    std::uint64_t sites_bits = 0;

    /// The lane's span in fire token rows, so the body can scope the delta to
    /// the requests this program governs. Resolved from the lane geometry at
    /// begin time — the prologue runs before the body publishes any KV
    /// observation, so there is no request CSR to consult yet.
    std::uint32_t token_start = 0;
    std::uint32_t token_count = 0;

    /// Adapter geometry, derived by the resolver from the sink arguments'
    /// trace-known value types (`A: [num_layers, R, d_in]`,
    /// `B: [num_layers, d_out, R]` — §6.5). The rank is deliberately NOT a
    /// sink argument: a different rank is a different traced program, so the
    /// shape carries it and the resolver throws when the two tensors
    /// disagree (or when any dim is symbolic). All four are element counts.
    ///
    /// Dtype note: `a`/`b` point at channel CELLS, and the PTIR channel
    /// vocabulary carries f32 (there is no bf16 wire dtype), so the contents
    /// are f32; a bf16 consumer casts once per fire before its GEMMs.
    std::uint32_t num_layers = 0;
    std::uint32_t rank = 0;
    std::uint32_t d_in = 0;
    std::uint32_t d_out = 0;

    /// The adapter FORM (the per-site rung's expression classes): the
    /// sink's ARITY selects it — 3 args = low-rank `y += B(Ax)` (`a`/`b`
    /// live, rank/d_in real), 2 args = SCALE `y = l ⊙ y` (IA3: `a`
    /// holds the l vector `[num_layers, d_out]`, `b` null, rank/d_in 0).
    enum class Form : std::uint32_t { LowRank = 0, Scale = 1 };
    Form form = Form::LowRank;
};

/// The launch's resolved lora configuration: one entry per lane whose program
/// carries the sink. Owned by the staged launch (rebuilt each time its
/// prologue executes); this is a borrowed view, valid for the fire.
struct LoraTable {
    const LoraLaneView* lanes = nullptr;
    std::uint32_t count = 0;

    bool usable() const noexcept { return lanes != nullptr && count > 0; }
};

}  // namespace pie_cuda_driver::model
