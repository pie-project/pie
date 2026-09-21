@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct Params {
    words: i32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u32(params.words)) {
        return;
    }
    dst[i] = src[i];
}

// pie:instantiate copy_words PIE_GROUP_X=256
