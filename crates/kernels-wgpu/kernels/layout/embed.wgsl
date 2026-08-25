// The DENSE token-embedding gather: `y[n, :] = table[ids[n], :]`.
//
// This is `layout.embed` at a `dense` bank, which is the seed statement of
// every tower -- the one point in the whole declaration floor whose result is
// sized without reading an operand's rectangle (`[fire, table.axis(1)]`), and
// therefore the only thing that can start one.
//
// # This is NOT `embed_gather.wgsl`
//
// That file gathers a row out of a TIED 4-BIT table and affine-dequantises it,
// and it takes three operands this statement does not carry (codes, scales,
// biases). This file is the other half of that pair: the bf16 table, gathered.
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
// `common/bf16.inc.wgsl` is not included. A copy reproduces the bit pattern,
// and rounding it through f32 and back would be a lossless operation at best
// and a canonicalisation of a NaN at worst. `row_gather.wgsl` makes the same
// argument for the same reason and this is the same kind of move.
//
// So one invocation owns one WORD -- columns `2j` and `2j+1` of its row -- both
// of which come from one word of the table because SOURCE AND DESTINATION SHARE
// A PITCH. That is what makes this a word copy and not a repack: `hidden` is
// the row of both, so `n * hidden + 2j` and `row * hidden + 2j` have the same
// parity for every `n` and every `row`.
//
// That parity is `hidden`'s. An odd `hidden` means row `n+1` starts in the
// middle of row `n`'s last word and no invocation can own it, so
// `kernels_wgpu::layout` refuses one by name rather than letting this body
// scramble it. Every hidden size the tree loads is a multiple of 64.
//
// The grid is `elementwise_rows(hidden, rows)` -- one lane per output ELEMENT,
// where half of them do the work and the other half fall out of the guard. That
// is `embed_gather.wgsl`'s arrangement too: an overshoot is harmless and the
// grid is then the same one every layout point states.

@group(0) @binding(0) var<storage, read_write> ids: array<i32>;
@group(0) @binding(1) var<storage, read_write> table: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;

// The row of both tensors, then the clamp bound. Both signed, because both are
// rectangle extents on the host side and `i32` is what a rectangle's width is
// there -- passing one as `u32` would only move the cast.
struct Params { hidden: i32, vocab: i32 }
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hidden = u32(params.hidden);
    // The word this invocation owns within the row: elements `2j` and `2j+1`.
    let c = gid.x * 2u;
    if c + 1u >= hidden {
        return;
    }

    // The row guard is the OUTPUT's own length, and it is what bounds `n`: the
    // row count is not an operand, so a grid rounded up on the row axis would
    // otherwise read `ids` past its end and -- worse -- write a clamped index,
    // which lands on the last element of `out_` rather than nowhere.
    let n = gid.y;
    let at = (n * hidden + c) >> 1u;
    if at >= arrayLength(&out_) {
        return;
    }

    let raw = ids[n];
    let row = select(0, raw, raw >= 0 && raw < params.vocab);
    out_[at] = table[(u32(row) * hidden + c) >> 1u];
}

// pie:instantiate embed_bfloat16
