#include <metal_stdlib>

using namespace metal;

template <typename T>
[[kernel]] void kv_append(
    const device T* k_new   [[buffer(0)]],
    const device T* v_new   [[buffer(1)]],
    device T* k_cache       [[buffer(2)]],
    device T* v_cache       [[buffer(3)]],
    const device int* pos                [[buffer(4)]],
    const constant int& head_dim         [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride  [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int d = int(tid.x);
  const int h = int(tid.y);
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
    const device T* k_new   [[buffer(0)]],
    const device T* v_new   [[buffer(1)]],
    device T* k_pages       [[buffer(2)]],
    device T* v_pages       [[buffer(3)]],
    const constant int& head_dim         [[buffer(5)]],

    const constant int& page_size        [[buffer(10)]],
    const constant int& n_kv_heads       [[buffer(12)]],
    const device uint* w_page            [[buffer(13)]],
    const device uint* w_off             [[buffer(14)]],

    const constant int& src_row_stride   [[buffer(15)]],
    uint3 tid [[thread_position_in_grid]]) {
  const int d = int(tid.x);
  const int h = int(tid.y);
  const int i = int(tid.z);
  if (d >= head_dim) return;

  const uint page = w_page[i];
  const size_t slot = size_t(page) * size_t(page_size) + size_t(w_off[i]);

  const size_t row_stride = size_t(n_kv_heads) * size_t(head_dim);
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
