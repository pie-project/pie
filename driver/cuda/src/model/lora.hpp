#pragma once

#include <cstdint>

namespace pie_cuda_driver::model {

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
