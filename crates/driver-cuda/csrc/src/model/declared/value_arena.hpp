#pragma once

// The SSA VALUE ARENA — where a traced value's buffer comes from.
//
// The trace is pure SSA: `rmsnorm(x: &Val) -> Val`, and every op carries
// its `inputs` / `outputs` as value ids. What it deliberately does NOT
// carry is a buffer, because choosing one is a BACKEND job. Both family
// executors did that job as convention — "the normed activation lives in
// `ws.norm_y`" here, "in `ws.norm_x`" there — and since an arm then has to
// know whose convention it is serving, the executor could not be one file.
// (Measured at the start of this merge: the SSA edges are read 1 and 0
// times across the two executors; workspace fields are read 82 and 110.)
//
// This is the other answer: a value gets a slot, the arm asks for the slot
// by value id, and which physical bytes those are stops being anyone's
// convention.
//
// WHO CHOOSES THE BYTES: the HOST does, and this file no longer does.
//
// It used to. It walked the plan for last-use, kept a free list, and
// bump-allocated on first ask — the same job `Buffers::assign` does in
// `model-compiler/src/lower.rs`. Two allocators over one plan have to
// agree byte-for-byte forever, and these two already did not: this copy
// predates `Select` (a value that IS a window of another's bytes), the
// `kernel!` in-place table (an output that IS the operand it accumulates
// into) and `Dim::MoeAlignedRoutes` (a padded extent it cannot size). On
// a text using any of the three it placed or sized a value differently,
// and silently, because an allocator that hands back a plausible pointer
// reports nothing.
//
// So the host assigns and this maps. `PieForwardLowered::value_offsets`
// is a byte offset per value id, `arena_bytes` is the block they need,
// and `slot()` is an add. Everything the old allocator knew — liveness,
// reuse, alignment, the union of values that share bytes — is decided
// once on the host, where it is tested (`model/tests/arena_soundness.rs`
// walks a write trace per family and fails if any value lands on bytes a
// later op still reads). A mapper cannot notice an overlapping
// assignment, which is exactly why the overlap has to be impossible
// there rather than caught here.
//
// CAPTURE SAFETY comes out of the same change for free. A decode body
// runs inside `cudaStreamBeginCapture`, so it must allocate nothing and
// a value must land at the same address on every fire. That used to hold
// because the ask ORDER was deterministic — an emergent property, true
// until an arm asked in a different order. Now the address is a function
// of the plan and the fire's extents, so it is structural.
//
// PINS stay, for the values the host declines to place: the buffer some
// machinery OUTSIDE the traced ops reaches by name — LoRA captures the
// normed activation's pointer at fire setup, hook sites observe the
// query buffer, the sampler reads the logits. The host marks those
// `NAMED` off the SEAM statements, so which values they are is stated in
// the declaration rather than listed per family, and the pass that binds
// them is the only family-shaped thing left here.
//
// MIGRATION: arms move onto the arena one dataflow island at a time. An
// arm that has not moved keeps its convention, and the two coexist —
// which is only sound while every producer and consumer of a given value
// have moved TOGETHER. That is why the islands are converted whole.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "pie_forward.h"
#include "pie_forward/plan.hpp"

using pie_forward::PieForwardLowered;
using pie_forward::PieForwardValue;

namespace pie_cuda_driver::model::declared {

class ValueArena {
   public:
    // `offset[v] == kNamed` is the host declining to place a value: its
    // bytes are the backend's to bind, via `pin`.
    static constexpr std::size_t kNamed = static_cast<std::size_t>(-1);

    // `block` is workspace-owned memory, allocated once, outside capture;
    // `flat` is the lowering that says where in it every value lives.
    //
    // Refuses rather than truncates when the block is too small. An arena
    // that quietly hands out addresses past its end is a corruption whose
    // first symptom is somewhere else entirely, and the block is sized by
    // a workspace formula that nothing previously checked against the
    // plan's actual need.
    void reset(void* block, std::size_t capacity,
               const PieForwardLowered& flat) {
        if (flat.arena_bytes > capacity) {
            throw std::runtime_error(
                "declared value arena: this fire needs " +
                std::to_string(flat.arena_bytes) +
                " bytes of activation block, but the workspace holds " +
                std::to_string(capacity) +
                " — the arena may not grow inside a captured body");
        }
        block_ = static_cast<std::uint8_t*>(block);
        capacity_ = capacity;
        want_ = flat.arena_bytes;
        offsets_ = flat.value_offsets;
        count_ = flat.value_offsets_len;
        pinned_.assign(count_, nullptr);
    }

    // The same arena for an executor that has not moved onto the flat
    // list yet: every value is the backend's to bind, so `slot()` serves
    // exactly the pins and names any other value as unbound.
    //
    // This is the honest shape of a half-migrated executor. It walks OPS
    // rather than rectangles, so it holds no lowering to read offsets
    // from, and lowering per fire just to get one would put a plan walk
    // on the hot path for buffers it is not yet using. What it must not
    // do is assign its own — that is the second allocator this file just
    // stopped being.
    void reset_pins_only(std::size_t value_count) {
        block_ = nullptr;
        capacity_ = 0;
        offsets_ = nullptr;
        count_ = value_count;
        pinned_.assign(count_, nullptr);
    }

    // Point an already-pinned arena at a lowering's table, for the leg
    // that gets its rectangles after the pin pass has run. Keeps the
    // pins, which is what makes the two coexist.
    void bind_offsets(void* block, std::size_t capacity,
                      const PieForwardLowered& flat) {
        // NO capacity precheck here, unlike `reset`. The block is sized
        // for the islands that have MOVED (`ws.declared_values` is one
        // or two values wide, and its comment says every further island
        // widens it), while `arena_bytes` is the whole plan's peak. A
        // half-migrated executor asks for a handful of values and must
        // not be refused for the ones it never touches; `slot` bounds-
        // checks the ones it does.
        block_ = static_cast<std::uint8_t*>(block);
        want_ = flat.arena_bytes;
        capacity_ = capacity;
        offsets_ = flat.value_offsets;
        if (flat.value_offsets_len > count_) {
            count_ = flat.value_offsets_len;
            pinned_.resize(count_, nullptr);
        }
    }

    // PIN a value the host declined to place. Declared once per family by
    // a pass over the plan, so an ARM still just asks by value id and
    // stays family-blind; the convention lives in the pass, not in 82
    // scattered arm sites.
    //
    // A pin WINS over the host's table where both speak, and that is the
    // migration rule rather than a conflict: an arm that has not moved
    // still writes `ws.norm_y` by convention, so its consumers have to
    // read those bytes and not the ones the lowering set aside. The two
    // agree again when the island moves, because the arm stops naming a
    // workspace field and the pin for that value goes away with it.
    void pin(std::uint32_t value_id, void* ptr) {
        if (value_id >= count_ || ptr == nullptr) return;
        pinned_[value_id] = ptr;
    }

    // The bytes one value lives in — an add, plus the pin table for the
    // values the host left to the backend.
    void* slot(std::uint32_t value_id) const {
        if (value_id >= count_) {
            throw std::runtime_error(
                "declared value arena: value id " +
                std::to_string(value_id) + " is outside the plan's table");
        }
        // The pin first — see `pin` for why it outranks the table.
        if (pinned_[value_id] != nullptr) return pinned_[value_id];
        const std::size_t at = offsets_ == nullptr ? kNamed : offsets_[value_id];
        if (at == kNamed) {
            throw std::runtime_error(
                "declared value arena: value " + std::to_string(value_id) +
                " is one the lowering left to the backend, and no pin pass "
                "bound it");
        }
        if (block_ == nullptr || at >= capacity_) {
            throw std::runtime_error(
                "declared value arena: value " + std::to_string(value_id) +
                " sits at offset " + std::to_string(at) + ", past the " +
                std::to_string(capacity_) +
                "-byte block — this plan's arena wants " +
                std::to_string(want_) +
                " bytes, so `ws.declared_values` is sized for fewer islands "
                "than are asking");
        }
        return block_ + at;
    }

    // The epilogue's two intermediates, from the same block.
    //
    // They belong to no traced value, because they belong to no traced
    // STATEMENT: one `LmHead` lowers to a gather, a norm and a GEMM, and
    // whether the gather runs is a fact about the FIRE's rows. So the
    // lowering owns them and this hands them over — which is what every
    // executor's `ws.norm_y` apology was standing in for.
    void* epilogue_gather(const PieForwardLowered& flat) const {
        return at_offset(flat.epilogue_gather);
    }
    void* epilogue_norm(const PieForwardLowered& flat) const {
        return at_offset(flat.epilogue_norm);
    }

    // Whether a value is reached through the PIN table rather than the
    // host's placement. A bring-up probe: an arm mid-migration whose
    // operand answers `false` is reading bytes no unconverted arm
    // writes, which is a silently wrong buffer rather than a fault.
    bool is_pinned(std::uint32_t value_id) const {
        return value_id < count_ && pinned_[value_id] != nullptr;
    }

    // Overload kept for the call sites mid-migration, which pass the
    // value descriptor because the arena used to need it for SIZING. It
    // does not any more — the host sized it — so the descriptor is
    // ignored and the sites drop it as their island moves.
    void* slot(std::uint32_t value_id, const PieForwardValue&) const {
        return slot(value_id);
    }

   private:
    // A lowering-owned offset, bounds-checked like a value's. `kNamed`
    // means the fire needs none, and a caller that asks anyway gets
    // nullptr rather than an address into somebody else's bytes.
    void* at_offset(std::size_t at) const {
        if (at == kNamed || block_ == nullptr || at >= capacity_) {
            return nullptr;
        }
        return block_ + at;
    }

    std::uint8_t* block_ = nullptr;
    std::size_t capacity_ = 0;
    // What the whole plan's arena wants, for the refusal message: the
    // block being short is a SIZING fact, and naming the target is the
    // difference between a fix and a bisect.
    std::size_t want_ = 0;
    const std::size_t* offsets_ = nullptr;
    std::size_t count_ = 0;
    std::vector<void*> pinned_;
};

// How many elements one traced value holds at this fire's extents.
//
// An arm needs it wherever a kernel takes an ELEMENT COUNT rather than
// rows and a width — the casts, the elementwise GLU. Those counts used
// to be spelled `N * H` and `routes * I` at each site, which is the
// convention doing arithmetic the value descriptor already carries; and
// it is the same arithmetic in each, so it lives here once.
//
// Dims outside the closed kinds resolve to their stated value, which is
// what `Const` means and what a padded route count is not — a value on
// `MoeAlignedRoutes` is not asked for by any arm that uses this.
inline std::size_t value_elements(const pie_forward::ForwardPlan& plan,
                                  std::uint32_t id, int n_fire, int r_fire) {
    const PieForwardValue& val = plan.value(id);
    std::size_t elements = 1;
    for (std::uint32_t d = 0; d < val.rank; ++d) {
        switch (val.dims[d].kind) {
        case pie_forward::PieForwardDimKind::Tokens:
            elements *= static_cast<std::size_t>(n_fire);
            break;
        case pie_forward::PieForwardDimKind::Requests:
            elements *= static_cast<std::size_t>(r_fire);
            break;
        default:
            elements *= static_cast<std::size_t>(val.dims[d].value);
            break;
        }
    }
    return elements;
}

// The activation block one plan needs for the WIDEST fire a deployment
// admits — what `ws.declared_values` has to hold.
//
// The row shape matters more than it looks. Rows a request does not
// sample are INTERIOR rows of a multi-token request, and saying so is
// what makes `Buffers::assign` size the `[Requests, vocab]` logits by
// sampled rows instead of by tokens. Left unset, every row counts as its
// own request and the answer is inflated by the largest value in the
// plan times the batch.
inline std::size_t arena_bytes_for_widest(const pie_forward::ForwardPlan& plan,
                                          int max_tokens, int max_sampled) {
    const std::size_t tokens =
        static_cast<std::size_t>(std::max(1, max_tokens));
    const int sampled = std::max(1, max_sampled);
    std::vector<pie_forward::PieForwardRow> rows(tokens);
    for (std::size_t i = 0; i < tokens; ++i) {
        pie_forward::PieForwardRow& row = rows[i];
        row = {};
        row.depth_k = -1;
        const bool samples = i < static_cast<std::size_t>(sampled);
        row.samples = samples ? 1 : 0;
        row.multi_token = samples ? 0 : 1;
    }
    const PieForwardLowered out = plan.lower(rows.data(), rows.size());
    // A plan that refuses the widest fire asks for nothing: the caller
    // keeps whatever block it had, and the arena's bounds check is what
    // catches a family whose islands outgrow it.
    if (out.uncovered != pie_forward::PieForwardUncovered::None) return 0;
    return out.arena_bytes;
}

// `PIE_DECLARED_ARENA_TRACE=1`: print what THIS driver's lowering says
// its arena needs, and which values are asking.
//
// It exists because the host and the driver disagreed by 138x about a
// one-row gemma-4 decode -- 2167296 bytes computed from the declaration
// in `model/tests/arena_soundness.rs`, 299302912 reported by
// `lower()` here -- and the two plans are not the same object, so no
// host-side test can say which is wrong. This prints the driver's own
// side of that comparison: the extents it lowered at, the block it got,
// and the values holding the most bytes, by SHAPE, which is the thing
// that would make a number that large.
inline void trace_arena(const char* family,
                        const pie_forward::ForwardPlan& plan,
                        const PieForwardLowered& flat,
                        std::size_t capacity, int n_fire, int r_fire) {
    const char* v = std::getenv("PIE_DECLARED_ARENA_TRACE");
    if (v == nullptr || v[0] == '0' || v[0] == '\0') return;

    std::fprintf(stderr,
                 "[arena/%s] N=%d R=%d ops=%zu values=%zu table=%zu "
                 "arena_bytes=%zu block=%zu\n",
                 family, n_fire, r_fire, plan.op_count(), plan.value_count(),
                 flat.value_offsets_len, flat.arena_bytes, capacity);

    // The widest PLACED values, by the bytes their shape implies at this
    // fire's extents. Sized here rather than trusted from the offsets,
    // so a wrong SIZE and a wrong PLACEMENT tell themselves apart.
    struct Row {
        std::size_t bytes;
        std::size_t at;
        std::uint32_t id;
    };
    std::vector<Row> rows;
    rows.reserve(flat.value_offsets_len);
    std::size_t placed_total = 0;
    for (std::uint32_t id = 0; id < flat.value_offsets_len; ++id) {
        const std::size_t at = flat.value_offsets[id];
        if (at == ValueArena::kNamed) continue;
        if (id >= plan.value_count()) continue;
        const PieForwardValue& val = plan.value(id);
        std::size_t elements = 1;
        for (std::uint32_t d = 0; d < val.rank; ++d) {
            switch (val.dims[d].kind) {
            case pie_forward::PieForwardDimKind::Tokens:
                elements *= static_cast<std::size_t>(n_fire);
                break;
            case pie_forward::PieForwardDimKind::Requests:
                elements *= static_cast<std::size_t>(r_fire);
                break;
            default:
                elements *= static_cast<std::size_t>(val.dims[d].value);
                break;
            }
        }
        const std::size_t width =
            val.dtype == pie_forward::PieForwardDType::F32 ||
                    val.dtype == pie_forward::PieForwardDType::I32
                ? 4u
                : 2u;
        const std::size_t bytes = elements * width;
        placed_total += bytes;
        rows.push_back(Row{bytes, at, id});
    }
    std::fprintf(stderr, "[arena/%s] placed=%zu values, %zu bytes if none reused\n",
                 family, rows.size(), placed_total);
    // Name one value and every statement that touches it. Both queries
    // below end here: a bisect narrows to a KEY, and this is what turns
    // a key back into text.
    const auto describe = [&](const Row& r, const char* how, std::size_t key) {
        const PieForwardValue& val = plan.value(r.id);
        std::fprintf(stderr, "[arena/%s] %s %zu: v%u at %zu %zu bytes [",
                     family, how, key, r.id, r.at, r.bytes);
        for (std::uint32_t d = 0; d < val.rank; ++d) {
            std::fprintf(stderr, "%s%u:%u", d ? ", " : "",
                         static_cast<unsigned>(val.dims[d].kind),
                         val.dims[d].value);
        }
        std::fprintf(stderr, "] dtype=%u\n", static_cast<unsigned>(val.dtype));
        for (std::size_t i = 0; i < plan.op_count(); ++i) {
            const pie_forward::PieForwardOp& op = plan.op(i);
            for (const std::uint32_t o : plan.outputs(op)) {
                if (o != r.id) continue;
                std::fprintf(stderr,
                             "[arena/%s]    written by op %zu kind=%u '%.*s'\n",
                             family, i, static_cast<unsigned>(op.kind),
                             static_cast<int>(plan.weight_name(op).size()),
                             plan.weight_name(op).data());
            }
            for (const std::uint32_t in : plan.inputs(op)) {
                if (in != r.id) continue;
                std::fprintf(stderr,
                             "[arena/%s]    read by    op %zu kind=%u '%.*s'\n",
                             family, i, static_cast<unsigned>(op.kind),
                             static_cast<int>(plan.weight_name(op).size()),
                             plan.weight_name(op).data());
            }
        }
    };

    // `PIE_DECLARED_ARENA_AT=<offset>`: name every value living at one
    // offset. Kept, but it answers a WEAKER question than it looks: an
    // offset is reused over a fire, so this lists every chain that ever
    // sat there, not one chain.
    if (const char* want_at = std::getenv("PIE_DECLARED_ARENA_AT")) {
        const std::size_t at = static_cast<std::size_t>(std::atoll(want_at));
        for (const Row& r : rows) {
            if (r.at == at) describe(r, "AT", at);
        }
    }

    // `PIE_DECLARED_ARENA_OWNER=<value id>`: name every value in one
    // alias chain. This is the query the owner-keyed bisect leaves you
    // with, and unlike AT it names exactly the values that share a
    // buffer BECAUSE they share a buffer -- so what it prints is one
    // statement's stream, start to end.
    if (const char* want_owner = std::getenv("PIE_DECLARED_ARENA_OWNER")) {
        const std::size_t want = static_cast<std::size_t>(std::atoll(want_owner));
        for (const Row& r : rows) {
            const std::size_t owner =
                r.id < flat.value_owners_len
                    ? static_cast<std::size_t>(flat.value_owners[r.id])
                    : static_cast<std::size_t>(r.id);
            if (owner == want) describe(r, "OWNER", want);
        }
    }

    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.bytes > b.bytes; });
    for (std::size_t i = 0; i < rows.size() && i < 8; ++i) {
        const PieForwardValue& val = plan.value(rows[i].id);
        std::fprintf(stderr, "[arena/%s]   v%-5u at %10zu  %10zu bytes  [",
                     family, rows[i].id, rows[i].at, rows[i].bytes);
        for (std::uint32_t d = 0; d < val.rank; ++d) {
            std::fprintf(stderr, "%s%u:%u", d ? ", " : "",
                         static_cast<unsigned>(val.dims[d].kind),
                         val.dims[d].value);
        }
        std::fprintf(stderr, "] dtype=%u\n",
                     static_cast<unsigned>(val.dtype));
    }
}

}  // namespace pie_cuda_driver::model::declared
