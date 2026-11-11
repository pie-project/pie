#include <metal_stdlib>
using namespace metal;

// Where a paged kv plane keeps token `off` of page `page`:
//
//   plane + page * page_stride + off * row                   (bytes)
//
// `page_stride` is the pool's, per plane: a plane that is its own buffer of
// `page_size`-row pages has `page_stride = page_size * row`; a page that
// holds every layer's rows (layer-major) has a wider stride, and the pool
// binds each layer's plane at its offset into page 0, so `plane` already
// points at that layer's rows. The offsets stay in the arguments (always 0,
// `kernels_metal::tensor::PLANE_AT_BINDING`). Every paged kernel takes the
// four as its last arguments, in this order.
struct pie_kv_layout {
  ulong k_page;
  ulong v_page;
  ulong k_offset;
  ulong v_offset;
};

#define PIE_KV_LAYOUT_TYPES                                                    \
  const constant ulong&, const constant ulong&, const constant ulong&,         \
      const constant ulong&,

#define PIE_KV_LAYOUT_ARGS pie_kv_layout{k_page_stride, v_page_stride, k_offset, v_offset}

// A plane's rows: the pool binds each plane at its offset into a page
// (`setBuffer:offset:`), so a kernel reads it from 0 and the `offset`
// argument is always 0. Applying a runtime offset in every paged kernel cost
// paged decode attention ~2% of a gemma-4-e4b decode token (M1 Max).
template <typename T>
inline const device T* pie_kv_plane(const device T* plane, ulong offset) {
  (void)offset;
  return plane;
}

template <typename T>
inline device T* pie_kv_plane(device T* plane, ulong offset) {
  (void)offset;
  return plane;
}
