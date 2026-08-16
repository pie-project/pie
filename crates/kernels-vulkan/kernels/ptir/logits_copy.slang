// PTIR logits row staging.
//
// Ported from Metal's batched copy: one dispatch can stage every row the fire
// needs, with the y lane choosing the per-row params record.
//
// The one kernel whose params binding is genuinely a runtime array -- there is
// one record per ROW and the row count is the dispatch's, not the module's --
// so this is the ordinary Slang `StructuredBuffer` and not `PIE_PARAMS`. The
// driver never looks for a scalar block here, because the row states none.

#include "common/bf16.slang"

struct PtirLogitsCopyParams { uint source_row; uint destination_row; uint vocab; uint reserved; };

PIE_BUFFER_RO(0, source);
PIE_BUFFER_RW(1, destination);
[[vk::binding(2)]] StructuredBuffer<PtirLogitsCopyParams> params;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    PtirLogitsCopyParams p = params[gid.y];
    if (gid.x >= p.vocab) return;
    destination[p.destination_row * p.vocab + gid.x] = source[p.source_row * p.vocab + gid.x];
}

// pie:instantiate copy_logits_bf16
