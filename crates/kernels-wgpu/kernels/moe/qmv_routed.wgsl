// Routed quantized GEMV for MoE expert slots.
//
// One workgroup per (row, output block, slot); 32 lanes walk the reduction
// dimension and 8 rows of them share the workgroup, which is the same shape
// `moe/qmv_routed.comp` and the Metal original have. The formats are
// deliberately only affine g64/b4 and MXFP4 g32/b4: those are what the routed
// checkpoints in the tree actually pack, and every extra pair would be nine
// more modules to compile for weights nobody ships.
//
// Two things are this backend's.
//
// **The `expert < 0` early return is gone, and that is not cosmetic.** The GLSL
// sibling returns before the reduction when a slot is unrouted. Every lane of
// the workgroup reads the same `expert_ids[sel]` -- `sel` comes from workgroup
// ids alone -- so the return really is workgroup-uniform, but naga cannot know
// that: a value loaded from a storage buffer is non-uniform to its analysis, so
// the `workgroupBarrier()` after it sits in non-uniform control flow and the
// module is REJECTED. Even where it compiled, an early return in front of a
// barrier is a hang rather than a wrong number. The flag below guards the WORK
// and the STORE instead, which costs an unrouted slot one empty tree reduction.
//
// **`y` is `array<atomic<u32>>`.** A bf16 tensor is `array<u32>` with two
// values per word (`common/bf16.inc.wgsl`), and the two halves of the word at
// `sel * out_vec_size + out_row` belong to two DIFFERENT workgroups whenever
// `out_row` is 8-aligned-odd or the row is the last of a slot -- the y axis of
// the grid is blocks of 8 output rows, so `out_row` 7 and 8 are never in the
// same workgroup. A read-modify-write would then lose one of the two. See the
// store.

//#include "common/bf16.inc.wgsl"
//#include "common/affine.inc.wgsl"
//#include "common/mxfp4.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> w: array<u32>;
// `scales` is one bf16 per group in the affine arm and one E8M0 BYTE per
// 32-element block in the MXFP4 one -- same binding, same operand, two
// completely different element widths, which is why every read of it goes
// through a named accessor below rather than a bare subscript.
@group(0) @binding(1) var<storage, read_write> scales: array<u32>;
// The MXFP4 codec has no separate bias PLANE -- its codes are not linear, so
// there is nothing for a per-group bias to be -- and the row still lists the
// operand. Binding 2 is therefore simply absent from that arm: a bind group
// entry the module does not use is legal, and declaring an unread
// `array<u32>` here would only invite a reader to believe MXFP4 has one.
//#if !defined(PIE_MXFP4)
@group(0) @binding(2) var<storage, read_write> biases: array<u32>;
//#endif
@group(0) @binding(3) var<storage, read_write> x: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<atomic<u32>>;
// Read only by the `_bias` variants. The row lists the operand either way,
// because a row is positional and dropping the slot would shift `expert_ids`
// down a binding; a declared-and-unread buffer costs nothing.
@group(0) @binding(5) var<storage, read_write> bias: array<u32>;
@group(0) @binding(6) var<storage, read_write> expert_ids: array<i32>;

// The five scalars, in the row's operand order. `x_slot_stride` and
// `x_row_stride` are both here and are different numbers: a routed decode
// packs `slots_per_row` copies of the activation contiguously inside a row, so
// walking slots is the small stride and walking rows the large one.
struct Params {
    in_vec_size: i32,
    out_vec_size: i32,
    x_slot_stride: i32,
    x_row_stride: i32,
    slots_per_row: i32,
}
@group(1) @binding(0) var<uniform> params: Params;

var<workgroup> partial: array<array<f32, 32>, 8>;

// The bf16 half-index split, per buffer. `pie_load_bf16` would say this once
// for all of them and cannot be called: naga 30 refuses a `ptr<storage, ...>`
// function parameter, so a module that called it would parse and then fail
// `create_shader_module`. The widening itself still goes through the fragment.
fn load_x(i: u32) -> f32 {
    let word = x[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn load_bias(i: u32) -> f32 {
    let word = bias[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

//#if !defined(PIE_MXFP4)
fn load_scale(i: u32) -> f32 {
    let word = scales[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn load_qbias(i: u32) -> f32 {
    let word = biases[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}
//#endif

// See the header: two atomics rather than one read-modify-write, because the
// other half of this word is another workgroup's output row.
fn store_y(i: u32, v: f32) {
    let at = i >> 1u;
    let b = pie_f32_to_bf16(v);
    if ((i & 1u) == 1u) {
        atomicAnd(&y[at], 0x0000ffffu);
        atomicOr(&y[at], b << 16u);
    } else {
        atomicAnd(&y[at], 0xffff0000u);
        atomicOr(&y[at], b);
    }
}

// Element `k` of the weight row belonging to (expert `e`, output row
// `out_row`).
//
// The expert axis is the OUTER one -- the weights are `[E, out_vec_size, K]` --
// so a routed row index is `e * out_vec_size + out_row` and the two index
// helpers in `common/affine.inc.wgsl` then address it exactly as the dense
// kernels do. Folding the expert into the element offset instead is the classic
// way to read expert 0's weights for every expert.
fn dequant_at(e: u32, out_row: u32, k: u32) -> f32 {
    let row = e * u32(params.out_vec_size) + out_row;
    let k_len = u32(params.in_vec_size);
//#if defined(PIE_MXFP4)
    // Two codes per byte, four bytes per word: the byte index is halved and
    // then split, and BOTH halves have to come from the same statement or a
    // reader of the second nibble reads the next byte's first.
    let bi = row * (k_len / 2u) + (k >> 1u);
    let byte_ = pie_mxfp4_byte(w[bi >> 2u], bi);
    let code = select(pie_mxfp4_lo(byte_), pie_mxfp4_hi(byte_), (k & 1u) == 1u);
    // One E8M0 byte per `PIE_MXFP4_BLOCK` elements, in its own plane.
    let sg = row * (k_len / PIE_MXFP4_BLOCK) + k / PIE_MXFP4_BLOCK;
    return code * pie_mxfp4_block_scale(pie_mxfp4_byte(scales[sg >> 2u], sg));
//#else
    let word = w[pie_affine_word_of(row, k_len, k)];
    let sg = pie_affine_scale_of(row, k_len, k);
    return pie_affine_value(word, pie_affine_code_of(k), load_scale(sg), load_qbias(sg));
//#endif
}

@compute @workgroup_size(32, 8)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let lane = lid.x;
    let local_out = lid.y;
    let row = wid.x;
    let block = wid.y;
    let slot = wid.z;
    let out_row = block * 8u + local_out;
    // The grid rounds the output axis up to blocks of 8, so the last block runs
    // past `out_vec_size`.
    let active_out = out_row < u32(params.out_vec_size);

    let sel = row * u32(params.slots_per_row) + slot;
    let expert = expert_ids[sel];
    // A slot with no expert still runs the reduction; see the header for why
    // this is a flag and not a `return`.
    let routed = expert >= 0;
    let e = u32(max(expert, 0));
    let x_base = row * u32(params.x_row_stride) + slot * u32(params.x_slot_stride);

    var acc = 0.0;
    if (active_out && routed) {
        // Lane-strided over the reduction dimension: 32 lanes, `in_vec_size`
        // elements, so each lane owns every 32nd. `in_vec_size` is not bounded
        // by the workgroup width and never was.
        for (var k = lane; k < u32(params.in_vec_size); k = k + 32u) {
            acc = acc + load_x(x_base + k) * dequant_at(e, out_row, k);
        }
    }

    partial[local_out][lane] = acc;
    workgroupBarrier();
    // A halving tree over the 32 lanes of this output row. The bound is a
    // const-expression and the guard is on the ADD, so every invocation reaches
    // every barrier -- which is the whole requirement.
    for (var step = 16u; step > 0u; step = step >> 1u) {
        if (lane < step) {
            partial[local_out][lane] = partial[local_out][lane] + partial[local_out][lane + step];
        }
        workgroupBarrier();
    }
    if (lane == 0u && active_out && routed) {
        var out = partial[local_out][0];
//#if defined(PIE_BIASED)
        // The bias is per (expert, output row) -- the same `[E, out_vec_size]`
        // shape the weights' outer two axes have, not per slot.
        out = out + load_bias(e * u32(params.out_vec_size) + out_row);
//#endif
        store_y(sel * u32(params.out_vec_size) + out_row, out);
    }
}

// pie:instantiate affine_qmv_routed_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1
// pie:instantiate mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_MXFP4=1 PIE_BIASED=1
