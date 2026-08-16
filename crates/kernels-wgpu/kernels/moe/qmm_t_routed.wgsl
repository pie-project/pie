// Routed quantized GEMM for sorted MoE rows.
//
// A tile of `PIE_BM` sorted rows by `PIE_BN` output columns per workgroup, with
// the expert taken from `tile_expert[tile_m]` -- which is why `route_sort`
// rounds every expert's span up to a whole number of `tile_rows`: one tile is
// one expert, and a tile that straddled two would multiply its second half by
// the wrong weights.
//
// Scalar and tiled, like `moe/qmm_t_routed.comp`, and deliberately not shared
// with the dense `quant/qmm_t.wgsl`: the routed family stands alone so that the
// expert-major weight indexing lives in one body. There is no matrix unit here
// and no `@coopmat` tier to reach for -- `src/capability.rs` has Baseline, Fp16
// and Subgroup and nothing else, on purpose.
//
// `PIE_BM` and `PIE_BN` have no default and the expander has no `//#error`: a
// variant that forgot them fails to compile at the first `u32(PIE_BM)` with
// "unknown identifier", which names the missing define at the line that wanted
// it.

//#include "common/bf16.inc.wgsl"
//#include "common/affine.inc.wgsl"
//#include "common/mxfp4.inc.wgsl"

// The two arms bind DIFFERENT things at the same numbers, because the two rows
// list different operands: MXFP4 has an exponent plane and a bias where affine
// has a scale and a bias PAIR. Nothing warns when a binding number is reused
// for a different tensor -- it reads whatever is there -- so the numbering here
// is the Vulkan table's, operand for operand, and not a translation of Metal's
// buffer indices (that translation is what cost the Vulkan port sixty misbound
// entrypoints).
@group(0) @binding(0) var<storage, read_write> w: array<u32>;
//#if defined(PIE_MXFP4)
// E8M0, one BYTE per 32-element block: four to a `u32` word, lowest first.
@group(0) @binding(1) var<storage, read_write> exponents: array<u32>;
@group(0) @binding(2) var<storage, read_write> x: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> bias: array<u32>;
@group(0) @binding(5) var<storage, read_write> tile_expert: array<i32>;
//#else
@group(0) @binding(1) var<storage, read_write> scales: array<u32>;
@group(0) @binding(2) var<storage, read_write> biases: array<u32>;
@group(0) @binding(3) var<storage, read_write> x: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> tile_expert: array<i32>;
//#endif

// The reduction length and the output width. Both are the SORTED tensor's, so
// `N` is the expert's output size and not the model's.
struct Params {
    K: i32,
    N: i32,
}
@group(1) @binding(0) var<uniform> params: Params;

// The bf16 half-index split, per buffer: `pie_load_bf16` takes a
// `ptr<storage, ...>`, which naga 30 refuses as a function parameter, so the
// subscript is local and only the widening is shared.
fn load_x(i: u32) -> f32 {
    let word = x[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

//#if defined(PIE_MXFP4)
fn load_bias(i: u32) -> f32 {
    let word = bias[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}
//#else
fn load_scale(i: u32) -> f32 {
    let word = scales[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn load_qbias(i: u32) -> f32 {
    let word = biases[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}
//#endif

// Two atomics, not a read-modify-write.
//
// Adjacent output columns `n` and `n + 1` land in the two halves of one word
// and are held by two DIFFERENT invocations -- the `cc` loop strides by the
// workgroup width, so consecutive columns are consecutive lanes -- and at an
// odd `N` a row's last column shares a word with the next row's first, which
// crosses workgroups. WGSL has no sub-word atomic, so the read-modify-write
// `pie_store_bf16` performs would drop one of the two values. AND-then-OR
// touches only this writer's sixteen bits, in any interleaving.
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

// Element `k` of output row `n` of expert `e`.
//
// The weights are `[E, N, K]`, expert-major, so the flat row is `e * N + n` and
// the dense index helpers then apply unchanged. Adding the expert offset to `k`
// instead -- the shape of the mistake -- stays inside the buffer and reads a
// diagonal slice of the wrong expert.
fn weight_at(e: u32, n: u32, k: u32) -> f32 {
    let row = e * u32(params.N) + n;
    let k_len = u32(params.K);
//#if defined(PIE_MXFP4)
    let bi = row * (k_len / 2u) + (k >> 1u);
    let byte_ = pie_mxfp4_byte(w[bi >> 2u], bi);
    let code = select(pie_mxfp4_lo(byte_), pie_mxfp4_hi(byte_), (k & 1u) == 1u);
    let s = row * (k_len / PIE_MXFP4_BLOCK) + k / PIE_MXFP4_BLOCK;
    return code * pie_mxfp4_block_scale(pie_mxfp4_byte(exponents[s >> 2u], s));
//#else
    let word = w[pie_affine_word_of(row, k_len, k)];
    let sg = pie_affine_scale_of(row, k_len, k);
    return pie_affine_value(word, pie_affine_code_of(k), load_scale(sg), load_qbias(sg));
//#endif
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_m = wid.y;
    let expert = tile_expert[tile_m];
    // A tile no expert owns is padding the sort left behind. Returning here is
    // safe in a way it is not in `qmv_routed`: this body has no barrier at all,
    // so there is no reduction for an absent invocation to hang.
    if (expert < 0) {
        return;
    }
    let e = u32(expert);
    let col0 = wid.x * u32(PIE_BN);
    let row0 = tile_m * u32(PIE_BM);

    // 16x16 invocations walk a `PIE_BM` x `PIE_BN` tile in strides of the
    // workgroup width, which is the `gated_rms` lesson: at `PIE_BN` of 64 a
    // body that gave each lane one column would compute a quarter of the tile
    // and leave the rest holding the previous dispatch's outputs.
    for (var rr = lid.y; rr < u32(PIE_BM); rr = rr + 16u) {
        for (var cc = lid.x; cc < u32(PIE_BN); cc = cc + 16u) {
            let n = col0 + cc;
            // The column axis is padded to `PIE_BN` and `N` need not be. The
            // ROW axis is not guarded, and does not need to be: `route_sort`
            // rounds each expert's span up to whole tiles, so every row of an
            // owned tile is inside the padded permutation.
            if (n >= u32(params.N)) {
                continue;
            }
            var acc = 0.0;
            for (var k = 0u; k < u32(params.K); k = k + 1u) {
                acc = acc + load_x((row0 + rr) * u32(params.K) + k) * weight_at(e, n, k);
            }
//#if defined(PIE_MXFP4)
            // Per (expert, column), like the weights' outer two axes -- the
            // MXFP4 rows are the `_bias` ones and the affine rows are not.
            acc = acc + load_bias(e * u32(params.N) + n);
//#endif
            store_y((row0 + rr) * u32(params.N) + n, acc);
        }
    }
}

// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_16_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_32_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_64_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_16_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_32_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_32_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_64_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_16_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_32_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_64_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=64
// `PIE_FP16` below is INERT: no body in this file reads it, so these nine
// modules are byte-identical to their non-`fp16` siblings and the name is a
// promise the code does not keep. It is not a live defect, because
// `affine_qmm_t_routed_fp16`'s row states no operands and so cannot be bound by
// any driver -- but it is a trap laid for whoever states it. In the dense
// sibling the `_fp16_precast` variants are NOT cosmetic: they bind a separate
// pre-cast fp16 activation buffer. A driver that selects this name because it
// has pre-cast its activations would hand fp16 bytes to a shader reading bf16
// and get silent garbage.
//
// So: implement the pre-cast path here BEFORE the row is stated, or drop the
// nine names -- which cannot be done unilaterally, since `kernels-metal` has
// the same nine and this table is checked against it row for row.
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_FP16=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_MXFP4=1
