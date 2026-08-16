// Affine GEMV projections, ported from `quant/qmv.comp` (itself from
// `quant/qmv.metal`).
//
// The port keeps MLX's little-endian affine packing, the four-rows-per-slot
// launch shape, and the bounds-checked K tail whose absence broke Gemma-4-31B
// at K=5376. WebGPU promises no subgroup width at all in the Baseline tier, so
// the final reduction is through workgroup memory over the 32 local-x lanes --
// the same shape Vulkan's port took, for the same reason.
//
// Two things are WGSL's alone and are the only real divergences from the
// `.comp`:
//
//   * every bf16 tensor is an `array<u32>` holding TWO values per word, low
//     half first, because WGSL has no 16-bit storage type even with `enable
//     f16`. Every index the GLSL uses against `uint16_t[]` is a half-index
//     here. See `common/bf16.inc.wgsl`.
//   * the output store is therefore a read-modify-write of a word this
//     invocation may only half own, and it goes through a CAS loop. See
//     `store_y` for which race that is and why it is not hypothetical.

//#include "common/bf16.inc.wgsl"
//#include "common/affine.inc.wgsl"

// ── Bindings ────────────────────────────────────────────────────────────────
//
// Numbered from the ROW in `src/quant.rs`, densely from zero over its
// buffer-kinded operands, with the scalars going to the uniform block instead.
// `.wiki/new-driver/vulkan.md` §3 records what happens when they are copied
// from Metal instead: Metal numbers scalars in the same run as buffers, so its
// `residual` is buffer 7 while the row puts it at 5, and a shader that
// transcribes the 7 binds a buffer the shell never wrote.

@group(0) @binding(0) var<storage, read_write> w: array<u32>;
@group(0) @binding(1) var<storage, read_write> scales: array<u32>;
@group(0) @binding(2) var<storage, read_write> biases: array<u32>;
@group(0) @binding(3) var<storage, read_write> x: array<u32>;
// `atomic` for the CAS in `store_y`, not because anything here accumulates.
@group(0) @binding(4) var<storage, read_write> y: array<atomic<u32>>;
// Declared only by the variants whose row has a sixth buffer operand:
// `affine_qmv_fast` has five and no sixth binding at all, so this cannot be
// declared unconditionally -- a module that binds past the row's buffer count
// is a pipeline-layout error, while a HOLE in the set is fine (§13).
//#if defined(PIE_RESIDUAL) || defined(PIE_BIAS)
@group(0) @binding(5) var<storage, read_write> extra: array<u32>;
//#endif

// The block is variant-shaped, and it has to be. `uniform_layout()` builds the
// buffer from what the ROW states -- `affine_qmv_fast` states two scalars -- so
// the fields here are exactly that row's scalars in that row's order, and
// `row_stride`/`m` appear only for the strided variant that reads them. The
// Vulkan file carried three more (`x_slot_stride`, `x_row_stride`,
// `slots_per_row`) that were the ROUTED row's shape, copied in and read by
// nothing.
struct Params {
    in_vec_size: i32,
    out_vec_size: i32,
//#if defined(PIE_WIDE_STRIDED)
    row_stride: i32,
    m: i32,
//#endif
}
@group(1) @binding(0) var<uniform> params: Params;

// 8 rows (two y-slots of four) x the 32 local-x lanes each row is split over.
var<workgroup> qmv_partials: array<f32, 8 * 32>;

// The `i`-th bf16 of a word already loaded. Not `pie_load_bf16`: that one takes
// a `ptr<storage, ...>`, which naga 30 parses and then refuses to validate
// ("which can't be passed into functions"), so it cannot appear in a module
// that has to reach a device. The conversion is still the include's.
fn qmv_bf16(word: u32, i: u32) -> f32 {
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

// Write one bf16 output element.
//
// The word at `i >> 1` holds outputs `i & ~1` and `i | 1`, and this invocation
// generally owns only one of them: with an odd `out_vec_size` -- a one-column
// gate projection is a real shape -- consecutive outputs of one row land in one
// word while belonging to different y-slots, and consecutive `vec`s land in one
// word while belonging to different WORKGROUPS. A plain read-modify-write
// therefore loses whichever half landed second, and no workgroup barrier can
// fix the cross-workgroup case. The CAS is device-scoped, which is exactly the
// scope of the race; it retries on the spurious failure `...Weak` is allowed.
fn store_y(i: u32, v: f32) {
    let at = i >> 1u;
    let shift = (i & 1u) * 16u;
    let keep = ~(0xffffu << shift);
    let bits = pie_f32_to_bf16(v) << shift;
    var old = atomicLoad(&y[at]);
    loop {
        let r = atomicCompareExchangeWeak(&y[at], old, (old & keep) | bits);
        if r.exchanged {
            break;
        }
        old = r.old_value;
    }
}

fn affine_value(row: u32, k: u32) -> f32 {
    let len = u32(params.in_vec_size);
    let word = w[pie_affine_word_of(row, len, k)];
    let g = pie_affine_scale_of(row, len, k);
    return pie_affine_value(
        word,
        pie_affine_code_of(k),
        qmv_bf16(scales[g >> 1u], g),
        qmv_bf16(biases[g >> 1u], g),
    );
}

fn x_at(vec_: u32, k: u32) -> f32 {
//#if defined(PIE_WIDE_STRIDED)
    let i = vec_ * u32(params.row_stride) + k;
//#else
    let i = vec_ * u32(params.in_vec_size) + k;
//#endif
    return qmv_bf16(x[i >> 1u], i);
}

// Values one lane pulls per pass: two words' worth, so 16 at four bits and 8 at
// eight. The GLSL passed this as a runtime argument and then wrote the inner
// loop to a constant 16 with `i < values_per_thread` inside; here it is a real
// const-expression -- `PIE_BITS` is a prelude const -- so the loop bound is the
// constant and the redundant guard is gone. The OTHER guard is not redundant:
// see `dot_lane`.
const PIE_QMV_VPT = PIE_CODES_PER_WORD * 2;

fn dot_lane(row: u32, vec_: u32, lid: u32) -> f32 {
    let n = u32(params.in_vec_size);
    let stride = u32(PIE_QMV_VPT) * 32u;
    var acc = 0.0;
    var k0 = lid * u32(PIE_QMV_VPT);
    while k0 < n {
        for (var i = 0u; i < u32(PIE_QMV_VPT); i = i + 1u) {
            // The K tail. K is NOT a whole number of `vpt * 32` blocks -- the
            // Metal comments record that assuming it was returned nonsense on
            // gemma-4-31b at K=5376 -- so the last pass runs partly out of
            // range and this is what keeps it out of the sum.
            if k0 + i < n {
                acc = acc + x_at(vec_, k0 + i) * affine_value(row, k0 + i);
            }
        }
        k0 = k0 + stride;
    }
    return acc;
}

//#if !defined(PIE_WIDE_STRIDED)
fn reduce_store(lid: u32, ly: u32, wg: vec3<u32>) {
    let out0 = wg.y * 8u + ly * 4u;
    let vec_ = wg.x;

    // Every lane runs all four rows and stores all four partials, out of range
    // or not, because the `barrier()` below is next: an early return in front
    // of a workgroup barrier is a HANG in WGSL, not a wrong number. The bound
    // is applied to the VALUE (zero) and again to the store, never to whether
    // an invocation arrives.
    for (var r = 0u; r < 4u; r = r + 1u) {
        let row = out0 + r;
        var v = 0.0;
        if row < u32(params.out_vec_size) {
            v = dot_lane(row, vec_, lid);
        }
        qmv_partials[(ly * 4u + r) * 32u + lid] = v;
    }
    workgroupBarrier();

    if lid == 0u {
        for (var r = 0u; r < 4u; r = r + 1u) {
            let row = out0 + r;
            if row < u32(params.out_vec_size) {
                var sum = 0.0;
                for (var i = 0u; i < 32u; i = i + 1u) {
                    sum = sum + qmv_partials[(ly * 4u + r) * 32u + i];
                }
//#if defined(PIE_BIAS)
                // Unconditional, where the GLSL asked `if (tail)`: every
                // variant that defines PIE_BIAS also defines PIE_TAIL, so the
                // runtime flag could only ever be true.
                sum = sum + qmv_bf16(extra[row >> 1u], row);
//#endif
//#if defined(PIE_RESIDUAL)
                let at = vec_ * u32(params.out_vec_size) + row;
                // Rounded to bf16 BEFORE the add, which is not an accident:
                // it makes this bit-identical to the two-kernel path (project,
                // then `residual_add`) that the fused variant replaces.
                let q = pie_bf16_to_f32(pie_f32_to_bf16(sum));
                store_y(at, q + qmv_bf16(extra[at >> 1u], at));
//#else
                store_y(vec_ * u32(params.out_vec_size) + row, sum);
//#endif
            }
        }
    }
}
//#endif

//#if defined(PIE_WIDE_STRIDED)
// One row per lane over the whole of K, four lanes per y-slot. `PIE_K_LANES` is
// in the entrypoint name and read by nothing: Metal splits K over eight lanes
// and finishes with `simd_shuffle_down`, and neither Vulkan's port nor this one
// does -- the name records the shape the launcher picked, and the body is a
// serial K loop that gets the same answer with 4 of 32 lanes busy. There is no
// barrier in here, which is why the early returns below are safe.
fn wide_strided(lid: vec3<u32>, wg: vec3<u32>) {
    let row_slot = lid.y * 4u + (lid.x & 3u);
    if lid.x >= 4u {
        return;
    }
    let row = wg.y * 8u + row_slot;
    let vec0 = wg.x * u32(PIE_VEC);
    if row >= u32(params.out_vec_size) {
        return;
    }
    for (var v = 0u; v < u32(PIE_VEC); v = v + 1u) {
        let vec_ = vec0 + v;
        if vec_ < u32(params.m) {
            var acc = 0.0;
            for (var k = 0u; k < u32(params.in_vec_size); k = k + 1u) {
                acc = acc + x_at(vec_, k) * affine_value(row, k);
            }
            store_y(vec_ * u32(params.row_stride) + row, acc);
        }
    }
}
//#endif

@compute @workgroup_size(32, 2, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
//#if defined(PIE_WIDE_STRIDED)
    wide_strided(lid, wg);
//#else
    reduce_store(lid.x, lid.y, wg);
//#endif
}

// pie:instantiate affine_qmv_fast_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8
// pie:instantiate affine_qmv_fast_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8
// pie:instantiate affine_qmv_fast_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_tail_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bias_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_BIAS=1 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bias_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_BIAS=1 PIE_TAIL=1
// pie:instantiate affine_qmv_wide_strided_bfloat16_gs_64_b_4_v_4_kl_8 PIE_GROUP=64 PIE_BITS=4 PIE_WIDE_STRIDED=1 PIE_VEC=4 PIE_K_LANES=8
// pie:instantiate affine_qmv_wide_strided_bfloat16_gs_64_b_8_v_4_kl_8 PIE_GROUP=64 PIE_BITS=8 PIE_WIDE_STRIDED=1 PIE_VEC=4 PIE_K_LANES=8
