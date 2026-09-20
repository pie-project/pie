@group(0) @binding(0) var<storage, read> ids: array<i32>;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;

struct Params {
    hidden: i32,
    vocab: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hidden = u32(max(params.hidden, 0));
    let words = hidden / 2u;
    let c = gid.x;
    let n = gid.y;
    if (c >= words) {
        return;
    }
    let raw = ids[n];
    var row = 0u;
    if (raw >= 0 && raw < params.vocab) {
        row = u32(raw);
    }
    out_[n * words + c] = table[row * words + c];
}

// pie:instantiate embed_bf16
