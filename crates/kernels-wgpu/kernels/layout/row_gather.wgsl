//#if defined(PIE_F32)
@group(0) @binding(0) var<storage, read> in_: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_: array<f32>;
//#else
@group(0) @binding(0) var<storage, read> in_: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;
//#endif
//#if defined(PIE_LIVE)
@group(0) @binding(2) var<storage, read> rows: array<i32>;
//#else
@group(0) @binding(2) var<storage, read> rows: array<u32>;
//#endif
struct Params {
    width: u32,
    count: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
//#if defined(PIE_F32)
    let c = gid.x;
    let pitch = params.width;
//#else

    let c = gid.x;
    let pitch = params.width >> 1u;
//#endif
    let i = gid.y;
    if (c >= pitch || i >= params.count) {
        return;
    }
//#if defined(PIE_LIVE)
    let at = rows[i];
    if (at < 0) {
        return;
    }
    out_[u32(at) * pitch + c] = in_[i * pitch + c];
//#elif defined(PIE_SCATTER)
    out_[rows[i] * pitch + c] = in_[i * pitch + c];
//#else
    out_[i * pitch + c] = in_[rows[i] * pitch + c];
//#endif
}

// pie:instantiate row_gather_bf16 PIE_GROUP_X=256
// pie:instantiate row_gather_f32 PIE_F32=1 PIE_GROUP_X=256
// pie:instantiate row_scatter_bf16 PIE_SCATTER=1 PIE_GROUP_X=256
// pie:instantiate row_scatter_f32 PIE_SCATTER=1 PIE_F32=1 PIE_GROUP_X=256
// pie:instantiate row_scatter_live_bf16 PIE_LIVE=1 PIE_GROUP_X=256
// pie:instantiate row_scatter_live_f32 PIE_LIVE=1 PIE_F32=1 PIE_GROUP_X=256
