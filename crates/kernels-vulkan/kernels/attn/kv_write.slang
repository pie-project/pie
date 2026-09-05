#include "common/bf16.slang"

#define PIE_GROUP 64

PIE_BUFFER_RO(0, k_new);
PIE_BUFFER_RO(1, v_new);
PIE_BUFFER_RW(2, k_pages);
PIE_BUFFER_RW(3, v_pages);
[[vk::binding(4)]] StructuredBuffer<uint> w_page;
[[vk::binding(5)]] StructuredBuffer<uint> w_off;

struct Push { int head_dim; int page_size; int n_kv_heads; int src_row_stride; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint d = gid.x;
    uint h = gid.y;
    uint i = gid.z;
    if (d >= uint(pc.head_dim)) return;
    uint row_stride = uint(pc.n_kv_heads) * uint(pc.head_dim);
    uint slot = w_page[i] * uint(pc.page_size) + w_off[i];
    uint dst = slot * row_stride + h * uint(pc.head_dim) + d;
    uint src_row = pc.src_row_stride > 0 ? uint(pc.src_row_stride) : row_stride;
    uint src = i * src_row + h * uint(pc.head_dim) + d;
    k_pages[dst] = k_new[src];
    v_pages[dst] = v_new[src];
}

// pie:instantiate kv_append_paged_bfloat16
