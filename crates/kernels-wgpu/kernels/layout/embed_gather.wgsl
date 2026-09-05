//#include "common/bf16.inc.wgsl"
//#include "common/affine.inc.wgsl"

@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> biases: array<u32>;
@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(4) var<storage, read_write> out_: array<u32>;

struct Params {
    hidden: i32,
    vocab: i32,
}
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hidden = u32(max(params.hidden, 0));
    let k = 2u * gid.x;
    let m = gid.y;
    if (k >= hidden) {
        return;
    }
    let id = ids[m];
    let at = (m * hidden + k) >> 1u;
    if (id < 0 || id >= params.vocab) {
        out_[at] = 0u;
        return;
    }
    let row = u32(id);
    let word = w[pie_affine_word_of(row, hidden, k)];
    let g = pie_affine_scale_of(row, hidden, k);
    let s = pie_bf16_at(scales[g >> 1u], g);
    let b = pie_bf16_at(biases[g >> 1u], g);
    let c = pie_affine_code_of(k);
    out_[at] = pie_pack_bf16(
        pie_affine_value(word, c, s, b),
        pie_affine_value(word, c + 1u, s, b),
    );
}

// pie:instantiate embed_gather_mb_bf16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_GROUP_X=256
