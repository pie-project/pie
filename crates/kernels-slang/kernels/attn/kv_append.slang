#include "prim/params.slang"

// New key and value rows appended into a paged cache, a word (two head
// columns) a lane: row `i` lands at in-page slot `w_off[i]` of page
// `w_page[i]`. A source row is `n_kv_heads * head_dim` wide, or
// `src_row_stride` when that is positive. The grid is (head words, heads,
// rows).

[[vk::binding(0)]] StructuredBuffer<uint> k_new;
[[vk::binding(1)]] StructuredBuffer<uint> v_new;
[[vk::binding(2)]] RWStructuredBuffer<uint> k_pages;
[[vk::binding(3)]] RWStructuredBuffer<uint> v_pages;
[[vk::binding(4)]] StructuredBuffer<uint> w_page;
[[vk::binding(5)]] StructuredBuffer<uint> w_off;

struct Params {
    int head_dim;
    int page_size;
    int n_kv_heads;
    int src_row_stride;
};
PIE_PARAMS(6, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint head_dim = uint(params.head_dim);
    uint d = 2u * gid.x;
    uint h = gid.y;
    uint i = gid.z;
    if (d >= head_dim) {
        return;
    }
    uint row = uint(params.n_kv_heads) * head_dim;
    uint slot = w_page[i] * uint(params.page_size) + w_off[i];
    uint dst = slot * row + h * head_dim + d;
    uint src = i * (params.src_row_stride > 0 ? uint(params.src_row_stride) : row) + h * head_dim + d;
    k_pages[dst >> 1] = k_new[src >> 1];
    v_pages[dst >> 1] = v_new[src >> 1];
}

// pie:instantiate attn_kv_append_bf16 PIE_GROUP_X=64
