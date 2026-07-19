// Raw-Metal KV-append (scatter) for Phase-0 decode.
//
// Writes the new token's projected K/V into the resident KV cache at the token's
// sequence position. Token-major layout, head_dim LAST (decode_abi locked):
//   cache [n_kv_heads, max_ctx, head_dim], new [n_kv_heads, head_dim].
// Port notes (M=1 decode):
//   * position IS the per-token IO scalar -> read from a *buffer* (pos[0]) per
//     decode_abi I1, never setBytes. The CB is byte-identical; the executor
//     writes the new seq position into the slot each token.
//   * head/seq strides are STATIC paged-KV geometry -> constant buffers.
//   * One thread per (head_dim channel, kv_head). In-place into the resident
//     cache slot that sdpa_vector_decode then reads.
// Launch: dispatchThreads grid=(head_dim, n_kv_heads, 1). bfloat native.

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void kv_append(
    const device T* k_new   [[buffer(0)]],  // [n_kv_heads, head_dim]
    const device T* v_new   [[buffer(1)]],
    device T* k_cache       [[buffer(2)]],  // [n_kv_heads, max_ctx, head_dim]
    device T* v_cache       [[buffer(3)]],
    const device int* pos                [[buffer(4)]],  // IO scalar (I1)
    const constant int& head_dim         [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],  // max_ctx*head_dim
    const constant size_t& k_seq_stride  [[buffer(7)]],  // head_dim
    uint2 tid [[thread_position_in_grid]]) {
  const int d = int(tid.x);   // channel within head_dim
  const int h = int(tid.y);   // kv head
  if (d >= head_dim) return;

  const size_t dst = h * k_head_stride + size_t(pos[0]) * k_seq_stride + d;
  const int src = h * head_dim + d;
  k_cache[dst] = k_new[src];
  v_cache[dst] = v_new[src];
}

#define instantiate_kv_append(name, itype)                        \
  template [[host_name("kv_append_" #name)]]                      \
  [[kernel]] void kv_append<itype>(                               \
      const device itype*, const device itype*, device itype*,    \
      device itype*, const device int*, const constant int&,      \
      const constant size_t&, const constant size_t&, uint2);

instantiate_kv_append(float32, float)
instantiate_kv_append(float16, half)
instantiate_kv_append(bfloat16, bfloat)
