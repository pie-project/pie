//#include "common/bf16.inc.wgsl"
//#if defined(PIE_SUBGROUP)
//#include "common/subgroup.inc.wgsl"
//#endif
//#if defined(PIE_MXFP4)
//#include "common/mxfp4.inc.wgsl"
const PIE_CHUNK = 8u;
//#elif defined(PIE_DENSE)
const PIE_CHUNK = 4u;
//#else
//#include "common/affine.inc.wgsl"
const PIE_CHUNK = u32(PIE_CODES_PER_WORD);
//#endif

const PIE_LANES = 32u;
const PIE_ROWS = 8u;

//#if defined(PIE_DENSE)
@group(0) @binding(0) var<storage, read> bank: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> expert_ids: array<i32>;

struct Params {
    in_vec_size: i32,
    out_vec_size: i32,
    x_slot_stride: i32,
    x_row_stride: i32,
    slots_per_row: i32,
}
@group(0) @binding(4) var<uniform> params: Params;
//#else
@group(0) @binding(0) var<storage, read> w: array<u32>;

@group(0) @binding(1) var<storage, read> scales: array<u32>;

@group(0) @binding(2) var<storage, read> biases: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<atomic<u32>>;

@group(0) @binding(5) var<storage, read> bias: array<u32>;
@group(0) @binding(6) var<storage, read> expert_ids: array<i32>;

struct Params {
    in_vec_size: i32,
    out_vec_size: i32,
    x_slot_stride: i32,
    x_row_stride: i32,
    slots_per_row: i32,
}
@group(0) @binding(7) var<uniform> params: Params;
//#endif

//#if !defined(PIE_SUBGROUP)
var<workgroup> partial: array<array<f32, PIE_LANES>, PIE_ROWS>;
//#endif

//#if defined(PIE_SUBGROUP)

const PIE_XS_WORDS = 4096u;
var<workgroup> xs_shared: array<u32, PIE_XS_WORDS>;
var<workgroup> xs_staged: u32;

fn load_x(base: u32, rel: u32) -> f32 {
    if (xs_staged == 1u) {
        return pie_bf16_at(xs_shared[rel >> 1u], rel);
    }
    let i = base + rel;
    return pie_bf16_at(x[i >> 1u], i);
}
//#else
fn load_x(base: u32, rel: u32) -> f32 {
    let i = base + rel;
    return pie_bf16_at(x[i >> 1u], i);
}
//#endif

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

fn chunk_dot(e: u32, out_row: u32, k: u32, x_base: u32) -> f32 {
    let in_size = u32(params.in_vec_size);
    let out_size = u32(params.out_vec_size);
    var acc = 0.0;
//#if defined(PIE_DENSE)
    let at = (e * out_size + out_row) * in_size + k;
    for (var j = 0u; j < PIE_CHUNK; j = j + 1u) {
        acc = acc + load_x(x_base, k + j) * pie_bf16_at(bank[(at + j) >> 1u], at + j);
    }
//#elif defined(PIE_MXFP4)
    let words_per_row = in_size / 8u;
    let word = w[(e * out_size + out_row) * words_per_row + k / 8u];
    let groups_per_row = in_size / PIE_MXFP4_BLOCK;
    let sg = (e * out_size + out_row) * groups_per_row + k / PIE_MXFP4_BLOCK;
    let scale = pie_mxfp4_block_scale(pie_mxfp4_byte(scales[sg >> 2u], sg));
    for (var j = 0u; j < 8u; j = j + 1u) {
        acc = acc + load_x(x_base, k + j) * pie_mxfp4_code(word, j);
    }
    acc = acc * scale;
//#else
    let words_per_row = in_size / PIE_CHUNK;
    let word = w[(e * out_size + out_row) * words_per_row + k / PIE_CHUNK];
    let groups_per_row = in_size / u32(PIE_GROUP);
    let sg = (e * out_size + out_row) * groups_per_row + k / u32(PIE_GROUP);
    let scale = pie_bf16_at(scales[sg >> 1u], sg);
    let zero = pie_bf16_at(biases[sg >> 1u], sg);
    var xs: array<f32, PIE_CODES_PER_WORD>;
    for (var j = 0u; j < PIE_CHUNK; j = j + 1u) {
        xs[j] = load_x(x_base, k + j);
    }
    acc = pie_affine_word_dot(word, xs, scale, zero);
//#endif
    return acc;
}

@compute @workgroup_size(32, 8, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    let local_out = local.y;
    let row = group.x;
    let block = group.y;
    let slot = group.z;
    let out_row = block * PIE_ROWS + local_out;
    let out_size = u32(params.out_vec_size);
    let in_size = u32(params.in_vec_size);
    let active_out = out_row < out_size;

    let sel = row * u32(params.slots_per_row) + slot;
    let expert = expert_ids[sel];

    let routed = expert >= 0;
    let e = u32(max(expert, 0));
    let x_base = row * u32(params.x_row_stride) + slot * u32(params.x_slot_stride);
//#if defined(PIE_SUBGROUP)
    let x_words = (in_size + 1u) >> 1u;
    let staged = x_words <= PIE_XS_WORDS && (x_base & 1u) == 0u;
    if (staged) {
        for (var w = lane + PIE_LANES * local_out; w < x_words; w = w + PIE_LANES * PIE_ROWS) {
            xs_shared[w] = x[(x_base >> 1u) + w];
        }
    }
    if (lane == 0u && local_out == 0u) {
        xs_staged = select(0u, 1u, staged);
    }
    workgroupBarrier();
//#endif

    var acc = 0.0;
    if (active_out && routed) {
        for (var k = lane * PIE_CHUNK; k < in_size; k = k + PIE_LANES * PIE_CHUNK) {
            acc = acc + chunk_dot(e, out_row, k, x_base);
        }
    }

//#if defined(PIE_SUBGROUP)

    let folded = pie_subgroup_sum32(acc);
    if (lane == 0u && active_out && routed) {
        var out = folded;
//#else
    partial[local_out][lane] = acc;
    workgroupBarrier();
    for (var step = 16u; step > 0u; step = step >> 1u) {
        if (lane < step) {
            partial[local_out][lane] = partial[local_out][lane] + partial[local_out][lane + step];
        }
        workgroupBarrier();
    }
    if (lane == 0u && active_out && routed) {
        var out = partial[local_out][0];
//#endif
//#if defined(PIE_BIASED)
        let b = e * out_size + out_row;
        out = out + pie_bf16_at(bias[b >> 1u], b);
//#endif
        store_y(sel * out_size + out_row, out);
    }
}

// pie:instantiate mxfp4_qmv_routed_bfloat16_gs_32_b_4 PIE_MXFP4=1
// pie:instantiate mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4 PIE_MXFP4=1 PIE_BIASED=1
// pie:instantiate affine_qmv_routed_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1
// pie:instantiate affine_qmv_routed_bfloat16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmv_routed_bfloat16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmv_routed_bfloat16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate select_gemv_bfloat16 PIE_DENSE=1
// pie:instantiate mxfp4_qmv_routed_bfloat16_gs_32_b_4 @subgroup PIE_MXFP4=1
// pie:instantiate mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4 @subgroup PIE_MXFP4=1 PIE_BIASED=1
// pie:instantiate affine_qmv_routed_bfloat16_gs_64_b_4 @subgroup PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_64_b_4 @subgroup PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1
// pie:instantiate affine_qmv_routed_bfloat16_gs_64_b_2 @subgroup PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_64_b_2 @subgroup PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmv_routed_bfloat16_gs_32_b_2 @subgroup PIE_GROUP=32 PIE_BITS=2
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_32_b_2 @subgroup PIE_GROUP=32 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmv_routed_bfloat16_gs_128_b_2 @subgroup PIE_GROUP=128 PIE_BITS=2
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_128_b_2 @subgroup PIE_GROUP=128 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate select_gemv_bfloat16 @subgroup PIE_DENSE=1
