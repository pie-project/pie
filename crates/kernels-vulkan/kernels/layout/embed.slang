// The DENSE token-embedding gather: `y[n, :] = table[ids[n], :]`.
//
// This is `layout.embed` at a `dense` bank, which is the seed statement of
// every tower -- the one point in the whole declaration floor whose result is
// sized without reading an operand's rectangle, and therefore the only thing
// that can start one.
//
// # This is NOT `embed_gather.slang`
//
// That file gathers a row out of a TIED 4-BIT table and affine-dequantises it,
// and it takes two operands this statement does not carry (scales, biases).
// This is the other half of the pair: the bf16 table, gathered.
//
// **It was the only half this plane had**, and a whole tower is what found
// that: `kernels_vulkan::layout`'s `embed` body asked `Staged::bank` for the
// sidecar planes unconditionally, and `Staged::bank` on this plane is a
// refusal with no other arm. So `layout.embed` was in the claim table and
// could not fire for any SKU in the catalog -- every one of them states a
// `bf16` embedding -- and nothing said so, because nothing had ever fired the
// first statement of a real plan here.
//
// # The vocab clamp is not decoration
//
// `row = (raw >= 0 && raw < vocab) ? raw : 0`, the same guard
// `kernels-cuda/kernels/layout/embed.cuh` states and for the same reason: the
// ids arrive from a wire payload, and an out-of-range one is an OOB read into
// the largest tensor in the model rather than a wrong answer. Out of vocab
// reads row 0.
//
// # A gather moves bits, so nothing here widens a bf16
//
// The copy is element for element at `PIE_ACT`, which reproduces the bit
// pattern; rounding it through `float` and back would be lossless at best and
// a canonicalisation of a NaN at worst. `row_gather.slang` makes the same
// argument for the same reason and this is the same kind of move.
//
// The WGSL sibling copies a WORD at a time -- two bf16 columns per invocation
// -- and refuses an odd `hidden` for it. This does not, and the difference is
// the binding model rather than a decision: `PIE_BUFFER_RO` is a
// `StructuredBuffer<PIE_ACT>`, so an element IS addressable here and there is
// no parity to keep. Every hidden size the tree loads is a multiple of 64
// either way.
//
// The grid is `elementwise_rows(hidden, rows)`, which is the grid every layout
// point on this plane states.

#include "common/bf16.slang"

[[vk::binding(0)]] StructuredBuffer<int> id;
PIE_BUFFER_RO(1, table);
PIE_BUFFER_RW(2, out_);

// The row of both tensors, then the clamp bound. Both signed, because both are
// rectangle extents on the host side and `i32` is what a rectangle's width is
// there -- passing one as `uint` would only move the cast.
struct Push { int hidden; int vocab; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint hidden = uint(pc.hidden);
    uint c = gid.x;
    uint n = gid.y;
    if (c >= hidden) return;

    int raw = id[n];
    uint row = uint((raw >= 0 && raw < pc.vocab) ? raw : 0);
    out_[n * hidden + c] = table[row * hidden + c];
}

// pie:instantiate embed_bfloat16
