//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

@group(0) @binding(0) var<storage, read> key: array<u32>;
@group(0) @binding(1) var<storage, read> query: array<u32>;
@group(0) @binding(2) var<storage, read> value: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<u32>;

struct Params {
    m: i32,
    h: i32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let M = u32(max(params.m, 1));
    let H = u32(max(params.h, 0));
    let n = group.x / M;
    let s = group.x - n * M;
    let stream = (n * M + s) * H;
    var acc = 0.0;
    for (var i = local.x; i < H; i = i + u32(PIE_GROUP_X)) {
        acc = acc + pie_bf16_at(key[(stream + i) >> 1u], stream + i)
            * pie_bf16_at(query[(stream + i) >> 1u], stream + i);
    }
    let dot = pie_workgroup_sum(local.x, u32(PIE_GROUP_X), acc) * inverseSqrt(f32(max(H, 1u)));
    var damped = sqrt(max(abs(dot), 1e-6));
    damped = select(select(0.0, -damped, dot < 0.0), damped, dot > 0.0);
    let gate = 1.0 / (1.0 + exp(-damped));
    for (var i = 2u * local.x; i < H; i = i + 2u * u32(PIE_GROUP_X)) {
        let word = value[(n * H + i) >> 1u];
        y[(stream + i) >> 1u] = pie_pack_bf16(
            gate * pie_bf16_to_f32(word & 0xffffu),
            gate * pie_bf16_to_f32(word >> 16u),
        );
    }
}

// pie:instantiate ple_gate_bf16 PIE_GROUP_X=256
