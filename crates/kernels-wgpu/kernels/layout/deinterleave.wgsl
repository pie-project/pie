//#if defined(PIE_SELECT)
@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;
struct Params {
    stride: i32,
    offset: i32,
    width: i32,
}
@group(0) @binding(2) var<uniform> params: Params;
//#else
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> left: array<u32>;
@group(0) @binding(2) var<storage, read_write> right: array<u32>;
struct Params {
    left_dim: i32,
    right_dim: i32,
}
@group(0) @binding(3) var<uniform> params: Params;
//#endif

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let row = gid.y;
//#if defined(PIE_SELECT)
    let width = u32(max(params.width, 0)) / 2u;
    if (c >= width) {
        return;
    }
    let stride = u32(max(params.stride, 0)) / 2u;
    let offset = u32(max(params.offset, 0)) / 2u;
    out_[row * width + c] = table[row * stride + offset + c];
//#else
    let left_dim = u32(max(params.left_dim, 0)) / 2u;
    let right_dim = u32(max(params.right_dim, 0)) / 2u;
    let total = left_dim + right_dim;
    if (c >= total) {
        return;
    }
    let value = src[row * total + c];
    if (c < left_dim) {
        left[row * left_dim + c] = value;
    } else {
        right[row * right_dim + (c - left_dim)] = value;
    }
//#endif
}

// pie:instantiate split_rows_bf16
// pie:instantiate select_slice_bf16 PIE_SELECT=1
