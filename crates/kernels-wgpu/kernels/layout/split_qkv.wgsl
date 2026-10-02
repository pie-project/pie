@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read_write> q: array<u32>;
@group(0) @binding(2) var<storage, read_write> k: array<u32>;
@group(0) @binding(3) var<storage, read_write> v: array<u32>;

struct Params {
    q_width: u32,
    kv_width: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qw = params.q_width / 2u;
    let kvw = params.kv_width / 2u;
    let pw = qw + 2u * kvw;
    let c = gid.x;
    let row = gid.y;
    if (c >= pw) {
        return;
    }
    let value = packed[row * pw + c];
    if (c < qw) {
        q[row * qw + c] = value;
    } else if (c < qw + kvw) {
        k[row * kvw + (c - qw)] = value;
    } else {
        v[row * kvw + (c - qw - kvw)] = value;
    }
}

// pie:instantiate split_qkv_bf16
