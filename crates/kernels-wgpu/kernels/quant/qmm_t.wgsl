//#include "common/bf16.inc.wgsl"
//#include "common/affine.inc.wgsl"

const PIE_BK = 32u;
const PIE_THREADS = 64u * PIE_WM;

const PIE_PER = (PIE_BM * PIE_BN) / (2u * PIE_THREADS);

@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> biases: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<u32>;

struct Params {
    k: i32,
    n: i32,
    m: i32,
}
@group(0) @binding(5) var<uniform> params: Params;

//#if PIE_BM == 64 && PIE_BN == 32
const PIE_RPL = 4u;
//#elif PIE_BM == 64 && PIE_BN == 16
const PIE_RPL = 2u;
//#elif PIE_BM == 32 && PIE_BN == 32
const PIE_RPL = 2u;
//#elif PIE_BM == 32 && PIE_BN == 16
const PIE_RPL = 1u;
//#elif PIE_BM == 16 && PIE_BN == 32
const PIE_RPL = 1u;
//#elif PIE_BM == 8 && PIE_BN == 32
const PIE_RPL = 1u;
//#else
const PIE_RPL = 0u;
//#endif

//#if PIE_BM == 16 && PIE_BN == 16 || PIE_BM == 8 && PIE_BN == 16
var<workgroup> xs: array<f32, PIE_BM * PIE_BK>;
var<workgroup> ws: array<f32, PIE_BN * PIE_BK>;

@compute @workgroup_size(32, 2, PIE_WM)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let flat = local.x + 32u * local.y + 64u * local.z;
    let row0 = group.y * u32(PIE_BM);
    let col0 = group.x * u32(PIE_BN);
    let k = u32(max(params.k, 0));
    let n = u32(max(params.n, 0));
    let m = u32(max(params.m, 0));
    let cpw = u32(PIE_CODES_PER_WORD);
    let wpr = k / cpw;
    let gpr = k / u32(PIE_GROUP);
    let words_per_block = PIE_BK / cpw;
    let half_bn = u32(PIE_BN) / 2u;

    var acc = array<vec2<f32>, PIE_PER>();
    for (var k0 = 0u; k0 < k; k0 = k0 + PIE_BK) {
        workgroupBarrier();
        for (var i = flat; i < u32(PIE_BM) * PIE_BK; i = i + u32(PIE_THREADS)) {
            let r = i / PIE_BK;
            let kk = i - r * PIE_BK;
            let row = row0 + r;
            var v = 0.0;
            if (row < m) {
                let e = row * k + k0 + kk;
                v = pie_bf16_at(x[e >> 1u], e);
            }
            xs[i] = v;
        }
        let g = k0 / u32(PIE_GROUP);
        for (var i = flat; i < u32(PIE_BN) * words_per_block; i = i + u32(PIE_THREADS)) {
            let c = i / words_per_block;
            let wi = i - c * words_per_block;
            let col = col0 + c;
            if (col < n) {
                let word = w[col * wpr + k0 / cpw + wi];
                let sg = col * gpr + g;
                let s = pie_bf16_at(scales[sg >> 1u], sg);
                let b = pie_bf16_at(biases[sg >> 1u], sg);
                for (var q = 0u; q < cpw; q = q + 1u) {
                    ws[c * PIE_BK + wi * cpw + q] = pie_affine_code(word, q) * s + b;
                }
            } else {
                for (var q = 0u; q < cpw; q = q + 1u) {
                    ws[c * PIE_BK + wi * cpw + q] = 0.0;
                }
            }
        }
        workgroupBarrier();
        for (var p = 0u; p < u32(PIE_PER); p = p + 1u) {
            let idx = flat + p * u32(PIE_THREADS);
            let r = idx / half_bn;
            let c = 2u * (idx - r * half_bn);
            var sum = vec2<f32>(0.0);
            for (var kk = 0u; kk < PIE_BK; kk = kk + 1u) {
                let xv = xs[r * PIE_BK + kk];
                sum = sum + xv * vec2<f32>(ws[c * PIE_BK + kk], ws[(c + 1u) * PIE_BK + kk]);
            }
            acc[p] = acc[p] + sum;
        }
    }
    for (var p = 0u; p < u32(PIE_PER); p = p + 1u) {
        let idx = flat + p * u32(PIE_THREADS);
        let r = idx / half_bn;
        let c = 2u * (idx - r * half_bn);
        let row = row0 + r;
        let col = col0 + c;
        if (row < m && col < n) {
            y[(row * n + col) >> 1u] = pie_pack_bf16(acc[p].x, acc[p].y);
        }
    }
}
//#else
const PIE_QUADS = u32(PIE_BN) / 4u;

var<workgroup> xs: array<vec4<f32>, PIE_BK * (u32(PIE_BM) / 4u)>;
var<workgroup> ws: array<vec4<f32>, PIE_BK * (u32(PIE_BN) / 4u)>;

@compute @workgroup_size(32, 2, PIE_WM)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let flat = local.x + 32u * local.y + 64u * local.z;
    let row0 = group.y * u32(PIE_BM);
    let col0 = group.x * u32(PIE_BN);
    let k = u32(max(params.k, 0));
    let n = u32(max(params.n, 0));
    let m = u32(max(params.m, 0));
    let cpw = u32(PIE_CODES_PER_WORD);
    let wpr = k / cpw;
    let gpr = k / u32(PIE_GROUP);
    let words_per_block = PIE_BK / cpw;
    let cq = flat % PIE_QUADS;
    let rg = flat / PIE_QUADS;
    let rows4 = u32(PIE_BM) / 4u;

    var acc: array<vec4<f32>, PIE_RPL>;
    for (var i = 0u; i < PIE_RPL; i++) {
        acc[i] = vec4<f32>(0.0);
    }
    for (var k0 = 0u; k0 < k; k0 = k0 + PIE_BK) {
        workgroupBarrier();

        for (var i = flat; i < u32(PIE_BM) * (PIE_BK / 2u); i = i + u32(PIE_THREADS)) {
            let r = i / (PIE_BK / 2u);
            let kk = (i - r * (PIE_BK / 2u)) * 2u;
            let row = row0 + r;
            var lo = 0.0;
            var hi = 0.0;
            if (row < m) {
                let e = row * k + k0 + kk;
                lo = pie_bf16_at(x[e >> 1u], e);
                hi = pie_bf16_at(x[(e + 1u) >> 1u], e + 1u);
            }
            xs[kk * rows4 + r / 4u][r & 3u] = lo;
            xs[(kk + 1u) * rows4 + r / 4u][r & 3u] = hi;
        }
        let g = k0 / u32(PIE_GROUP);
//#if !defined(PIE_WCOMPONENT)

        for (var i = flat; i < PIE_QUADS * words_per_block; i = i + u32(PIE_THREADS)) {
            let cq4 = i / words_per_block;
            let wi = i - cq4 * words_per_block;
            let col = col0 + cq4 * 4u;
            var word: vec4<u32>;
            var sc: vec4<f32>;
            var bi: vec4<f32>;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col + j;
                if (c < n) {
                    word[j] = w[c * wpr + k0 / cpw + wi];
                    let sg = c * gpr + g;
                    sc[j] = pie_bf16_at(scales[sg >> 1u], sg);
                    bi[j] = pie_bf16_at(biases[sg >> 1u], sg);
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
//#else
        for (var i = flat; i < u32(PIE_BN) * words_per_block; i = i + u32(PIE_THREADS)) {
            let c = i / words_per_block;
            let wi = i - c * words_per_block;
            let col = col0 + c;
            if (col < n) {
                let word = w[col * wpr + k0 / cpw + wi];
                let sg = col * gpr + g;
                let s = pie_bf16_at(scales[sg >> 1u], sg);
                let b = pie_bf16_at(biases[sg >> 1u], sg);
                for (var q = 0u; q < cpw; q = q + 1u) {
                    ws[(wi * cpw + q) * PIE_QUADS + c / 4u][c & 3u] = pie_affine_code(word, q) * s + b;
                }
            } else {
                for (var q = 0u; q < cpw; q = q + 1u) {
                    ws[(wi * cpw + q) * PIE_QUADS + c / 4u][c & 3u] = 0.0;
                }
            }
        }
//#endif
        workgroupBarrier();
        for (var kk = 0u; kk < PIE_BK; kk = kk + 1u) {
            let wv = ws[kk * PIE_QUADS + cq];
//#if PIE_BM == 64 && PIE_BN == 32
            let xv = xs[kk * rows4 + rg];
            acc[0] = acc[0] + xv.x * wv;
            acc[1] = acc[1] + xv.y * wv;
            acc[2] = acc[2] + xv.z * wv;
            acc[3] = acc[3] + xv.w * wv;
//#else
            for (var i = 0u; i < PIE_RPL; i++) {
                let r = rg * PIE_RPL + i;
                acc[i] = acc[i] + xs[kk * rows4 + r / 4u][r & 3u] * wv;
            }
//#endif
        }
    }
    let col = col0 + cq * 4u;
    for (var i = 0u; i < PIE_RPL; i++) {
        let row = row0 + rg * PIE_RPL + i;
        if (row < m) {
            if (col < n) {
                y[(row * n + col) >> 1u] = pie_pack_bf16(acc[i].x, acc[i].y);
            }
            if (col + 2u < n) {
                y[(row * n + col + 2u) >> 1u] = pie_pack_bf16(acc[i].z, acc[i].w);
            }
        }
    }
}
//#endif

// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_8_bn_16 PIE_GROUP=32 PIE_BITS=2 PIE_BM=8 PIE_BN=16 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_8_bn_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=8 PIE_BN=32 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=2 PIE_BM=16 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=16 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=2 PIE_BM=32 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=32 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=2 PIE_BM=64 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=64 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_8_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=8 PIE_BN=16 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_8_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=8 PIE_BN=32 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_8_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=8 PIE_BN=16 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_8_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=8 PIE_BN=32 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_8_bn_16 PIE_GROUP=64 PIE_BITS=2 PIE_BM=8 PIE_BN=16 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_8_bn_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=8 PIE_BN=32 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=2 PIE_BM=16 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=16 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=2 PIE_BM=32 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=32 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=2 PIE_BM=64 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=64 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_8_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=8 PIE_BN=16 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_8_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=8 PIE_BN=32 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_8_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=8 PIE_BN=16 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_8_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=8 PIE_BN=32 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_8_bn_16 PIE_GROUP=128 PIE_BITS=2 PIE_BM=8 PIE_BN=16 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_8_bn_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=8 PIE_BN=32 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=2 PIE_BM=16 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=16 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=2 PIE_BM=32 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=32 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=2 PIE_BM=64 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=64 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_8_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=8 PIE_BN=16 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_8_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=8 PIE_BN=32 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_8_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=8 PIE_BN=16 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_8_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=8 PIE_BN=32 PIE_WM=1
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=16 PIE_WM=2
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_WM=2

// pie:instantiate affine_qmm_t_components_bf16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_WM=2 PIE_WCOMPONENT=1
