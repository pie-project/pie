#include "prim/params.slang"
#include "prim/pool.slang"

// New key and value rows appended into a paged cache, a word (two head
// columns) a lane: row `i` lands at in-page slot `w_off[i]` of page
// `w_page[i]`. A source row is `n_kv_heads * head_dim` wide, or
// `src_row_stride` when that is positive. The grid is (head words, heads,
// rows). The pages lie in the chunk pool (`prim/pool.slang`): page `p`'s
// keys from word `k_origin + p * k_page_stride`, its values likewise.

[[vk::binding(0)]] StructuredBuffer<uint> k_new;
[[vk::binding(1)]] StructuredBuffer<uint> v_new;
[[vk::binding(2)]] StructuredBuffer<uint> w_page;
[[vk::binding(3)]] StructuredBuffer<uint> w_off;
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint, 4)

struct Params {
    int head_dim;
    int page_size;
    int n_kv_heads;
    int src_row_stride;
    uint seg_words;
    uint k_origin;
    uint v_origin;
    uint k_page_stride;
    uint v_page_stride;
};
PIE_PARAMS(8, Params, params);
PIE_POOL_READ(uint, params.seg_words)
PIE_POOL_WRITE(uint, params.seg_words)
PIE_POOL_VIEW(k_pages, uint, params.k_origin)
PIE_POOL_VIEW(v_pages, uint, params.v_origin)

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
    uint dst = w_off[i] * row + h * head_dim + d;
    uint src = i * (params.src_row_stride > 0 ? uint(params.src_row_stride) : row) + h * head_dim + d;
    k_pages[w_page[i] * params.k_page_stride + (dst >> 1)] = k_new[src >> 1];
    v_pages[w_page[i] * params.v_page_stride + (dst >> 1)] = v_new[src >> 1];
}

// pie:instantiate attn_kv_append_bf16 PIE_GROUP_X=64
