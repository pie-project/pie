// The KV write, contiguous and paged: one file because they are one BINDING
// CONTRACT and one scatter.
//
// Both take `(k_new, v_new, k_dst, v_dst)` at buffers 0..3 and write the step's
// K and V into the cache. What differs is how the destination is addressed --
// a contiguous `[n_kv_heads, max_ctx, head_dim]` cache, or a page table -- and
// that difference is the whole of the second kernel.

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

instantiate_kv_append(bfloat16, bfloat)

template <typename T>
[[kernel]] void kv_append_paged(
    const device T* k_new   [[buffer(0)]],   // K:  [N, n_kv_heads, head_dim]
    const device T* v_new   [[buffer(1)]],   // V
    device T* k_pages       [[buffer(2)]],   // KPages: [num_pages*page_size, n_kv_heads, head_dim]
    device T* v_pages       [[buffer(3)]],   // VPages
    const constant int& head_dim         [[buffer(5)]],   // HeadDim
    // Buffers 4, 6-9 and 11 belong to the shared ring/read ABI. The host
    // validates the normalized write against that CSR before encoding; this
    // kernel needs only the physical destination below.
    const constant int& page_size        [[buffer(10)]],  // PageSize
    const constant int& n_kv_heads       [[buffer(12)]],  // NKvHeads
    const device uint* w_page            [[buffer(13)]],  // explicit/normalized physical destination
    const device uint* w_off             [[buffer(14)]],  // explicit/normalized in-page offset
    // Elements between one token's k_new/v_new row and the next, 0 meaning the
    // packed [N, n_kv_heads, head_dim] layout.
    //
    // NOBODY PASSES A NONZERO ONE. `kernels-metal`'s `kv_append_paged` is the
    // only thing that binds this slot and it states 0; the Vulkan and WGSL
    // siblings of this kernel do not declare the parameter at all, and their
    // prefills append through the packed layout like everything else. So the
    // branch below has one live arm, and this is a hook rather than a feature.
    //
    // It was written for a prefill whose rows sit in the scratch arena at that
    // arena's one common pitch -- several times as wide as a kv row -- and
    // reading the note as though that case were HANDLED is the mistake to
    // avoid. Handling it needs a caller that knows the pitch, and on this
    // plane a routine's `In` carries `rows` and `width` and no stride; see
    // `kernels-cuda`'s `kimi_mla`, where the same distinction is a region
    // field and a `Refusal::Narrow`.
    const constant int& src_row_stride   [[buffer(15)]],
    uint3 tid [[thread_position_in_grid]]) {
  const int d = int(tid.x);   // channel within head_dim
  const int h = int(tid.y);   // kv head
  const int i = int(tid.z);   // token within the batch [0, N)
  if (d >= head_dim) return;

  const uint page = w_page[i];
  const size_t slot = size_t(page) * size_t(page_size) + size_t(w_off[i]);

  const size_t row_stride = size_t(n_kv_heads) * size_t(head_dim);  // NHD page row
  const size_t dst = slot * row_stride + size_t(h) * size_t(head_dim) + size_t(d);
  const size_t src_row = src_row_stride > 0 ? size_t(src_row_stride) : row_stride;
  const size_t src = size_t(i) * src_row + size_t(h) * size_t(head_dim) + size_t(d);

  k_pages[dst] = k_new[src];
  v_pages[dst] = v_new[src];
}

#define instantiate_kv_append_paged(name, itype)                  \
  template [[host_name("kv_append_paged_" #name)]]                \
  [[kernel]] void kv_append_paged<itype>(                         \
      const device itype*, const device itype*, device itype*,    \
      device itype*, const constant int&, const constant int&,    \
      const constant int&, const device uint*,                    \
      const device uint*, const constant int&, uint3);

instantiate_kv_append_paged(bfloat16, bfloat)
