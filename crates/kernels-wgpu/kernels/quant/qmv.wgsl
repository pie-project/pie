//#include "common/bf16.inc.wgsl"
//#include "common/affine.inc.wgsl"
//#if defined(PIE_SUBGROUP)
//#include "common/subgroup.inc.wgsl"
//#endif

const PIE_LANES = 32u;
//#if defined(PIE_SUBGROUP)

const PIE_ROWS_PER_HALF = 2u;
const PIE_HALVES = 4u;
//#else
const PIE_ROWS_PER_HALF = 4u;
const PIE_HALVES = 2u;
//#endif

@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> biases: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<u32>;

struct Params {
    in_vec_size: i32,
    out_vec_size: i32,
}
@group(0) @binding(5) var<uniform> params: Params;

//#if defined(PIE_SUBGROUP)
const PIE_XS_WORDS = 4096u;
var<workgroup> xs_shared: array<u32, PIE_XS_WORDS>;
var<workgroup> xs_staged: u32;
//#else
var<workgroup> partials: array<f32, 256>;
//#endif

@compute @workgroup_size(32, PIE_HALVES, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let lid = local.x;
    let ly = local.y;
    let vec = group.x;
    let out0 = group.y * (PIE_HALVES * PIE_ROWS_PER_HALF) + ly * PIE_ROWS_PER_HALF;
    let k = u32(max(params.in_vec_size, 0));
    let n = u32(max(params.out_vec_size, 0));
    let cpw = u32(PIE_CODES_PER_WORD);
    let wpr = k / cpw;
    let gpr = k / u32(PIE_GROUP);

//#if defined(PIE_SUBGROUP)

    let x_words = (k + 1u) >> 1u;
    let x_base = vec * k;
    let staged = x_words <= PIE_XS_WORDS && (x_base & 1u) == 0u;
    if (staged) {
        for (var w = lid + PIE_LANES * ly; w < x_words; w = w + PIE_LANES * PIE_HALVES) {
            xs_shared[w] = x[(x_base >> 1u) + w];
        }
    }
    if (lid == 0u && ly == 0u) {
        xs_staged = select(0u, 1u, staged);
    }
    workgroupBarrier();
//#endif
    var acc = array<f32, PIE_ROWS_PER_HALF>();
    for (var j = lid; j < wpr; j = j + PIE_LANES) {
        let kb = j * cpw;

        var xw = array<f32, PIE_CODES_PER_WORD>();
        var xsum = 0.0;
        for (var i = 0u; i < cpw; i = i + 1u) {
            let e = vec * k + kb + i;
//#if defined(PIE_SUBGROUP)
            var v = 0.0;
            if (xs_staged == 1u) {
                v = pie_bf16_at(xs_shared[(kb + i) >> 1u], kb + i);
            } else {
                v = pie_bf16_at(x[e >> 1u], e);
            }
//#else
            let v = pie_bf16_at(x[e >> 1u], e);
//#endif
            xw[i] = v;
            xsum = xsum + v;
        }
        let g = kb / u32(PIE_GROUP);
        for (var r = 0u; r < PIE_ROWS_PER_HALF; r = r + 1u) {
            let row = out0 + r;
            if (row < n) {
                let word = w[row * wpr + j];
                let sg = row * gpr + g;
                let s = pie_bf16_at(scales[sg >> 1u], sg);
                let b = pie_bf16_at(biases[sg >> 1u], sg);
                var dot = 0.0;
                for (var i = 0u; i < cpw; i = i + 1u) {
                    dot = dot + xw[i] * pie_affine_code(word, i);
                }
                acc[r] = acc[r] + s * dot + b * xsum;
            }
        }
    }
//#if defined(PIE_SUBGROUP)

    for (var r = 0u; r < PIE_ROWS_PER_HALF; r = r + 1u) {
        acc[r] = pie_subgroup_sum32(acc[r]);
    }
    if (lid == 0u) {
        for (var r = 0u; r < PIE_ROWS_PER_HALF; r = r + 2u) {
            let row = out0 + r;
            if (row < n) {
                y[(vec * n + row) >> 1u] = pie_pack_bf16(acc[r], acc[r + 1u]);
            }
        }
    }
//#else
    for (var r = 0u; r < PIE_ROWS_PER_HALF; r = r + 1u) {
        partials[(ly * PIE_ROWS_PER_HALF + r) * PIE_LANES + lid] = acc[r];
    }
    workgroupBarrier();
    for (var s = PIE_LANES / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            for (var r = 0u; r < PIE_ROWS_PER_HALF; r = r + 1u) {
                let at = (ly * PIE_ROWS_PER_HALF + r) * PIE_LANES;
                partials[at + lid] = partials[at + lid] + partials[at + lid + s];
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        for (var r = 0u; r < PIE_ROWS_PER_HALF; r = r + 2u) {
            let row = out0 + r;
            if (row < n) {
                let lo = partials[(ly * PIE_ROWS_PER_HALF + r) * PIE_LANES];
                let hi = partials[(ly * PIE_ROWS_PER_HALF + r + 1u) * PIE_LANES];
                y[(vec * n + row) >> 1u] = pie_pack_bf16(lo, hi);
            }
        }
    }
//#endif
}

// pie:instantiate affine_qmv_bf16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2
// pie:instantiate affine_qmv_bf16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4
// pie:instantiate affine_qmv_bf16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8
// pie:instantiate affine_qmv_bf16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmv_bf16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_bf16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate affine_qmv_bf16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2
// pie:instantiate affine_qmv_bf16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4
// pie:instantiate affine_qmv_bf16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8
// pie:instantiate affine_qmv_bf16_gs_32_b_2 @subgroup PIE_GROUP=32 PIE_BITS=2
// pie:instantiate affine_qmv_bf16_gs_32_b_4 @subgroup PIE_GROUP=32 PIE_BITS=4
// pie:instantiate affine_qmv_bf16_gs_32_b_8 @subgroup PIE_GROUP=32 PIE_BITS=8
// pie:instantiate affine_qmv_bf16_gs_64_b_2 @subgroup PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmv_bf16_gs_64_b_4 @subgroup PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_bf16_gs_64_b_8 @subgroup PIE_GROUP=64 PIE_BITS=8
// pie:instantiate affine_qmv_bf16_gs_128_b_2 @subgroup PIE_GROUP=128 PIE_BITS=2
// pie:instantiate affine_qmv_bf16_gs_128_b_4 @subgroup PIE_GROUP=128 PIE_BITS=4
// pie:instantiate affine_qmv_bf16_gs_128_b_8 @subgroup PIE_GROUP=128 PIE_BITS=8
