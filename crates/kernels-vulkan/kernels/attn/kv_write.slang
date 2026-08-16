#include "common/bf16.slang"

// The contiguous and paged KV appends share one binding contract and one scatter.
// Scalars move to push constants, which is what pulls the paged write page and
// offset down to bindings 10 and 11: the row keeps the Metal ring ABI's
// placeholder operands (`ring_4` .. `ring_15`) so the two tables stay
// comparable, but `kernels_vulkan::bindings()` numbers BUFFERS densely and
// sends `head_dim`, `page_size` and `n_kv_heads` to the push block instead, so
// the buffer run closes up by three.
//
// These read 9 and 10 until a SPIR-V-level audit compared the compiled
// `OpDecorate Binding` set against the table: off by one, which would have made
// the shader read `ring_11` as the write page and the write page as the offset.
// That audit is deleted, so nothing will catch the same mistake twice. Do not
// renumber these by hand: `kernels_vulkan::bindings` numbers the row's
// buffer-kinded operands densely from zero in operand order, and the row is
// `kv_append_paged` in `src/attn.rs`. Read those two, in that order.

#define PIE_GROUP 256

PIE_BUFFER_RO(0, k_new);
PIE_BUFFER_RO(1, v_new);
PIE_BUFFER_RW(2, k_dst);
PIE_BUFFER_RW(3, v_dst);

#if defined(PIE_PAGED)
[[vk::binding(10)]] StructuredBuffer<uint> w_page;
[[vk::binding(11)]] StructuredBuffer<uint> w_off;
struct Push { int head_dim; int page_size; int n_kv_heads; };
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
    uint src = i * row_stride + h * uint(pc.head_dim) + d;
    k_dst[dst] = k_new[src];
    v_dst[dst] = v_new[src];
}
#else
[[vk::binding(4)]] StructuredBuffer<int> pos;
struct Push { int head_dim; PIE_STRIDE k_head_stride; PIE_STRIDE k_seq_stride; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint d = gid.x;
    uint h = gid.y;
    if (d >= uint(pc.head_dim)) return;
    uint dst = h * PIE_LOW(pc.k_head_stride) + uint(pos[0]) * PIE_LOW(pc.k_seq_stride) + d;
    uint src = h * uint(pc.head_dim) + d;
    k_dst[dst] = k_new[src];
    v_dst[dst] = v_new[src];
}
#endif

// pie:instantiate kv_append_bfloat16
// pie:instantiate kv_append_paged_bfloat16 PIE_PAGED=1
