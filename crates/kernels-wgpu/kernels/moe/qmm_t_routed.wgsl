//#include "common/bf16.inc.wgsl"
//#if defined(PIE_MXFP4)
//#include "common/mxfp4.inc.wgsl"
//#else
//#include "common/affine.inc.wgsl"
//#endif

const PIE_BK = 32u;
const PIE_BN = 32u;

@group(0) @binding(0) var<storage, read> w: array<u32>;

@group(0) @binding(1) var<storage, read> scales: array<u32>;

@group(0) @binding(2) var<storage, read> biases: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<u32>;

@group(0) @binding(5) var<storage, read> bias: array<u32>;
@group(0) @binding(6) var<storage, read> tile_expert: array<i32>;

struct Params {
    k: i32,
    n: i32,
}
@group(0) @binding(7) var<uniform> params: Params;

//#if !defined(PIE_SCALARSTAGE)

const PIE_QUADS = PIE_BN / 4u;
const PIE_ROWS4 = u32(PIE_BM) / 4u;
const PIE_THREADS = PIE_ROWS4 * PIE_QUADS;

var<workgroup> xs: array<vec4<f32>, PIE_BK * PIE_ROWS4>;
var<workgroup> ws: array<vec4<f32>, PIE_BK * PIE_QUADS>;

@compute @workgroup_size(32, PIE_THREADS / 32u, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let flat = local.x + 32u * local.y;
    let tile = group.y;
    let expert = tile_expert[tile];

    if (expert < 0) {
        return;
    }
    let e = u32(expert);
    let row0 = tile * u32(PIE_BM);
    let col0 = group.x * PIE_BN;
    let k = u32(params.k);
    let n = u32(params.n);

    let cq = flat % PIE_QUADS;
    let rg = flat / PIE_QUADS;
    let r0 = rg * 4u;

    var acc = array<vec4<f32>, 4>();

    for (var k0 = 0u; k0 < k; k0 = k0 + PIE_BK) {
        workgroupBarrier();

        for (var i = flat; i < u32(PIE_BM) * (PIE_BK / 2u); i = i + PIE_THREADS) {
            let r = i / (PIE_BK / 2u);
            let kk = (i - r * (PIE_BK / 2u)) * 2u;
            let at = (row0 + r) * k + k0 + kk;
            xs[kk * PIE_ROWS4 + r / 4u][r & 3u] = pie_bf16_at(x[at >> 1u], at);
            xs[(kk + 1u) * PIE_ROWS4 + r / 4u][r & 3u] = pie_bf16_at(x[(at + 1u) >> 1u], at + 1u);
        }
//#if defined(PIE_MXFP4)

        for (var i = flat; i < PIE_QUADS * 4u; i = i + PIE_THREADS) {
            let cq4 = i >> 2u;
            let wi = i & 3u;
            let col = col0 + cq4 * 4u;
            var word: vec4<u32>;
            var sc: vec4<f32>;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col + j;
                if (c < n) {
                    let row = e * n + c;
                    word[j] = w[row * (k >> 3u) + (k0 >> 3u) + wi];
                    let sb = row * (k >> 5u) + (k0 >> 5u);
                    sc[j] = pie_mxfp4_block_scale(pie_mxfp4_byte(scales[sb >> 2u], sb));
                } else {
                    word[j] = 0u;
                    sc[j] = 0.0;
                }
            }
            for (var j = 0u; j < 8u; j = j + 1u) {
                let code = vec4<f32>(
                    pie_mxfp4_code(word.x, j),
                    pie_mxfp4_code(word.y, j),
                    pie_mxfp4_code(word.z, j),
                    pie_mxfp4_code(word.w, j),
                );
                ws[(wi * 8u + j) * PIE_QUADS + cq4] = code * sc;
            }
        }
//#else
        let cpw = u32(PIE_CODES_PER_WORD);
        let words_per_block = PIE_BK / cpw;
        let wpr = k * u32(PIE_BITS) / 32u;
        let gpr = k / u32(PIE_GROUP);
        let g = k0 / u32(PIE_GROUP);
        for (var i = flat; i < PIE_QUADS * words_per_block; i = i + PIE_THREADS) {
            let cq4 = i / words_per_block;
            let wi = i - cq4 * words_per_block;
            let col = col0 + cq4 * 4u;
            var word: vec4<u32>;
            var sc: vec4<f32>;
            var bi: vec4<f32>;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col + j;
                if (c < n) {
                    let row = e * n + c;
                    word[j] = w[row * wpr + k0 / cpw + wi];
                    let si = row * gpr + g;
                    sc[j] = pie_bf16_at(scales[si >> 1u], si);
                    bi[j] = pie_bf16_at(biases[si >> 1u], si);
                } else {
                    word[j] = 0u;
                    sc[j] = 0.0;
                    bi[j] = 0.0;
                }
            }
            for (var q = 0u; q < cpw; q = q + 1u) {
                let code = vec4<f32>(
                    pie_affine_code(word.x, q),
                    pie_affine_code(word.y, q),
                    pie_affine_code(word.z, q),
                    pie_affine_code(word.w, q),
                );
                ws[(wi * cpw + q) * PIE_QUADS + cq4] = code * sc + bi;
            }
        }
//#endif
        workgroupBarrier();
        for (var kk = 0u; kk < PIE_BK; kk = kk + 1u) {
            let wv = ws[kk * PIE_QUADS + cq];
            let xv = xs[kk * PIE_ROWS4 + rg];
            acc[0] = acc[0] + xv.x * wv;
            acc[1] = acc[1] + xv.y * wv;
            acc[2] = acc[2] + xv.z * wv;
            acc[3] = acc[3] + xv.w * wv;
        }
    }

    let col = col0 + cq * 4u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let row = row0 + r0 + i;
        var v = acc[i];
//#if defined(PIE_BIASED)
        let b0 = e * n + col;
        v = v + vec4<f32>(
            pie_bf16_at(bias[b0 >> 1u], b0),
            pie_bf16_at(bias[(b0 + 1u) >> 1u], b0 + 1u),
            pie_bf16_at(bias[(b0 + 2u) >> 1u], b0 + 2u),
            pie_bf16_at(bias[(b0 + 3u) >> 1u], b0 + 3u),
        );
//#endif
        let at = row * n + col;
        if (col < n) {
            y[at >> 1u] = pie_pack_bf16(v.x, v.y);
        }
        if (col + 2u < n) {
            y[(at + 2u) >> 1u] = pie_pack_bf16(v.z, v.w);
        }
    }
}
//#else
const PIE_THREADS = 128u;

const PIE_STRIDE = PIE_BK + 1u;

const PIE_COL_LANES = PIE_BN / 2u;
const PIE_ROW_LANES = PIE_THREADS / PIE_COL_LANES;
const PIE_RM = u32(PIE_BM) / PIE_ROW_LANES;

var<workgroup> xs: array<f32, u32(PIE_BM) * PIE_STRIDE>;
var<workgroup> ws: array<f32, PIE_BN * PIE_STRIDE>;

@compute @workgroup_size(32, 4, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let flat = local.x + 32u * local.y;
    let tile = group.y;
    let expert = tile_expert[tile];

    if (expert < 0) {
        return;
    }
    let e = u32(expert);
    let row0 = tile * u32(PIE_BM);
    let col0 = group.x * PIE_BN;
    let k = u32(params.k);
    let n = u32(params.n);

    let c2 = (flat % PIE_COL_LANES) << 1u;
    let ty = flat / PIE_COL_LANES;

    var acc0: array<f32, PIE_RM>;
    var acc1: array<f32, PIE_RM>;
    for (var m = 0u; m < PIE_RM; m = m + 1u) {
        acc0[m] = 0.0;
        acc1[m] = 0.0;
    }

    for (var k0 = 0u; k0 < k; k0 = k0 + PIE_BK) {
        workgroupBarrier();

        for (var i = flat; i < u32(PIE_BM) * PIE_BK; i = i + PIE_THREADS) {
            let r = i / PIE_BK;
            let kk = i - r * PIE_BK;
            let at = (row0 + r) * k + k0 + kk;
            xs[r * PIE_STRIDE + kk] = pie_bf16_at(x[at >> 1u], at);
        }
//#if defined(PIE_MXFP4)

        for (var i = flat; i < PIE_BN * 4u; i = i + PIE_THREADS) {
            let c = i >> 2u;
            let wi = i & 3u;
            let col = col0 + c;
            if (col < n) {
                let row = e * n + col;
                let word = w[row * (k >> 3u) + (k0 >> 3u) + wi];
                let sb = row * (k >> 5u) + (k0 >> 5u);
                let scale = pie_mxfp4_block_scale(pie_mxfp4_byte(scales[sb >> 2u], sb));
                for (var j = 0u; j < 8u; j = j + 1u) {
                    ws[c * PIE_STRIDE + wi * 8u + j] = pie_mxfp4_code(word, j) * scale;
                }
            } else {
                for (var j = 0u; j < 8u; j = j + 1u) {
                    ws[c * PIE_STRIDE + wi * 8u + j] = 0.0;
                }
            }
        }
//#else
        let cpw = u32(PIE_CODES_PER_WORD);
        let words_per_block = PIE_BK / cpw;
        let wpr = k * u32(PIE_BITS) / 32u;
        let gpr = k / u32(PIE_GROUP);
        let g = k0 / u32(PIE_GROUP);
        for (var i = flat; i < PIE_BN * words_per_block; i = i + PIE_THREADS) {
            let c = i / words_per_block;
            let wi = i - c * words_per_block;
            let col = col0 + c;
            if (col < n) {
                let row = e * n + col;
                let word = w[row * wpr + k0 / cpw + wi];
                let si = row * gpr + g;
                let s = pie_bf16_at(scales[si >> 1u], si);
                let b = pie_bf16_at(biases[si >> 1u], si);
                for (var q = 0u; q < cpw; q = q + 1u) {
                    ws[c * PIE_STRIDE + wi * cpw + q] = pie_affine_code(word, q) * s + b;
                }
            } else {
                for (var q = 0u; q < cpw; q = q + 1u) {
                    ws[c * PIE_STRIDE + wi * cpw + q] = 0.0;
                }
            }
        }
//#endif
        workgroupBarrier();

        for (var kk = 0u; kk < PIE_BK; kk = kk + 1u) {
            let w0 = ws[c2 * PIE_STRIDE + kk];
            let w1 = ws[(c2 + 1u) * PIE_STRIDE + kk];
            for (var m = 0u; m < PIE_RM; m = m + 1u) {
                let xv = xs[(ty + m * PIE_ROW_LANES) * PIE_STRIDE + kk];
                acc0[m] = acc0[m] + xv * w0;
                acc1[m] = acc1[m] + xv * w1;
            }
        }
    }

    let col = col0 + c2;
    if (col < n) {
        for (var m = 0u; m < PIE_RM; m = m + 1u) {
            let row = row0 + ty + m * PIE_ROW_LANES;
            var v0 = acc0[m];
            var v1 = acc1[m];
//#if defined(PIE_BIASED)
            let b0 = e * n + col;
            v0 = v0 + pie_bf16_at(bias[b0 >> 1u], b0);
            v1 = v1 + pie_bf16_at(bias[(b0 + 1u) >> 1u], b0 + 1u);
//#endif
            let at = row * n + col;
            y[at >> 1u] = pie_f32_to_bf16(v0) | (pie_f32_to_bf16(v1) << 16u);
        }
    }
}
//#endif

// pie:instantiate mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_16 PIE_BM=16 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_32 PIE_BM=32 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_64 PIE_BM=64 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_16 PIE_BM=16 PIE_MXFP4=1 PIE_BIASED=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_32 PIE_BM=32 PIE_MXFP4=1 PIE_BIASED=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_64 PIE_BM=64 PIE_MXFP4=1 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_4_bm_16 PIE_BM=16 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_4_bm_32 PIE_BM=32 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_4_bm_64 PIE_BM=64 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_16 PIE_BM=16 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_32 PIE_BM=32 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_64 PIE_BM=64 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_2_bm_16 PIE_BM=16 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_2_bm_32 PIE_BM=32 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_2_bm_64 PIE_BM=64 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_16 PIE_BM=16 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_32 PIE_BM=32 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_64 PIE_BM=64 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1

// pie:instantiate mxfp4_qmm_t_routed_bias_scalar_bf16_gs_32_b_4_bm_32 PIE_BM=32 PIE_MXFP4=1 PIE_BIASED=1 PIE_SCALARSTAGE=1
// pie:instantiate affine_qmm_t_routed_scalar_bf16_gs_64_b_4_bm_32 PIE_BM=32 PIE_GROUP=64 PIE_BITS=4 PIE_SCALARSTAGE=1
