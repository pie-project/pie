@group(0) @binding(0) var<storage, read> k_new: array<u32>;
@group(0) @binding(1) var<storage, read> v_new: array<u32>;
@group(0) @binding(2) var<storage, read_write> k_pages: array<u32>;
@group(0) @binding(3) var<storage, read_write> v_pages: array<u32>;
@group(0) @binding(4) var<storage, read> w_page: array<u32>;
@group(0) @binding(5) var<storage, read> w_off: array<u32>;

struct Params {
    head_dim: i32,
    page_size: i32,
    n_kv_heads: i32,
    src_row_stride: i32,
}
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head_dim = u32(params.head_dim);
    let d = gid.x * 2u;
    let h = gid.y;
    let i = gid.z;
    if (d >= head_dim) {
        return;
    }
    let row_stride = u32(params.n_kv_heads) * head_dim;
    let slot = w_page[i] * u32(params.page_size) + w_off[i];
    let dst = slot * row_stride + h * head_dim + d;
    var src_row = row_stride;
    if (params.src_row_stride > 0) {
        src_row = u32(params.src_row_stride);
    }
    let src = i * src_row + h * head_dim + d;
    k_pages[dst >> 1u] = k_new[src >> 1u];
    v_pages[dst >> 1u] = v_new[src >> 1u];
}

// pie:instantiate kv_append_paged_bfloat16 PIE_GROUP_X=64
