#pragma once

#include "prim/device.hip"

namespace pie {

// Where a lane's buffered rows lie, resolved row by row on the device, so a
// fire stages no entry a row whatever its lanes carry. Buffer token `tok`
// (slot `base + tok`) is the row of slot `σ` on entry `(σ - base) / T` of
// `bases` (`bases_len` entries of `T` rows each, their slot addresses), row
// `(σ - base) mod T`. A lane alone on its set reads its own pages: `list`
// null and `σ` the slot. A beam on a hybrid (BEAM-PAGE-LISTS §11, §11.1)
// reads its set's entries through its row of its group's list, which the
// fire's `beam_step` points `list` at: `σ = (f0 + list[slot / S - f0]) S +
// slot mod S`, a slot below `f0 S` the set's own; `cap` bounds the list.
struct RsRows {
    const u32* list;
    const u64* bases;
    u32 cap;
    u32 f0;
    u32 page;
    u32 base;
    u32 tokens;
    u32 bases_len;

    // In u32: a slot lies within a sequence's context, and 64-bit division
    // costs a carried row several times its 32-bit one.
    __device__ __forceinline__ void locate(u32 tok, u64& slot, u32& row) const {
        const u32 T = tokens == 0u ? 1u : tokens;
        const u32 at_slot = base + tok;
        u32 sigma = at_slot;
        if (list != nullptr && at_slot >= f0 * page) {
            const u32 j = at_slot / page - f0;
            // A column past the list faulted the beam in its step; its rows
            // read the list's last column, never past it.
            const u32 q = list[j < cap ? j : cap - 1u];
            sigma = (f0 + q) * page + (at_slot - (f0 + j) * page);
        }
        const u32 rel = sigma - base;
        u32 entry = rel / T;
        if (entry >= bases_len) entry = bases_len - 1u;
        slot = bases[entry];
        row = rel - entry * T;
    }
};

static_assert(sizeof(RsRows) == 40, "RsRows: the Rust `channel::RsRows` mirrors this layout");

}
