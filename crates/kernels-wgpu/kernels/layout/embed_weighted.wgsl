//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read> ids: array<i32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> table: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<u32>;
struct Params {
    hidden: i32,
    vocab: i32,
    taps: i32,
    rows: i32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = i32(gid.x) * 2;
    let n = gid.y;
    if (c >= params.hidden || n >= u32(params.rows)) {
        return;
    }
    let row_base = n * u32(params.taps);
    let hidden = u32(params.hidden);
    var lo = 0.0;
    var hi = 0.0;
    for (var t = 0u; t < u32(params.taps); t = t + 1u) {
        let raw = ids[row_base + t];
        var at = 0u;
        if (raw >= 0 && raw < params.vocab) {
            at = u32(raw);
        }
        let w = weights[row_base + t];
        let word = table[(at * hidden + u32(c)) >> 1u];
        lo = lo + w * pie_bf16_to_f32(word & 0xffffu);
        hi = hi + w * pie_bf16_to_f32(word >> 16u);
    }
    y[(n * hidden + u32(c)) >> 1u] = pie_pack_bf16(lo, hi);
}

// pie:instantiate embed_weighted_bf16 PIE_GROUP_X=256
