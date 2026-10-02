@group(0) @binding(0) var<storage, read> qg: array<u32>;
@group(0) @binding(1) var<storage, read_write> q_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> gate_out: array<u32>;

struct Params {
    head_dim: i32,
    qg_width: i32,
    q_width: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hd = u32(max(params.head_dim, 0)) / 2u;
    let i = gid.x;
    let h = gid.y;
    let row = gid.z;
    if (i >= hd) {
        return;
    }
    let out_row = row * (u32(max(params.q_width, 0)) / 2u);
    let qg_row = row * (u32(max(params.qg_width, 0)) / 2u);
    q_out[out_row + h * hd + i] = qg[qg_row + h * 2u * hd + i];
    gate_out[out_row + h * hd + i] = qg[qg_row + h * 2u * hd + hd + i];
}

// pie:instantiate q_gate_split_bf16 PIE_GROUP_X=256
