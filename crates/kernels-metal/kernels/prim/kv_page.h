#include <metal_stdlib>
using namespace metal;

// Where a paged kv plane keeps token `off` of page `page`:
//
//   plane + offset + page * page_stride + off * row          (bytes)
//
// `page_stride` and `offset` are the pool's, per plane: today a plane is its
// own buffer of `page_size`-row pages (`page_stride = page_size * row`,
// `offset = 0`); a page that holds every layer's rows (layer-major) keeps
// the same arithmetic with a wider stride and a layer's offset into it.
// Every paged kernel takes the four as its last arguments, in this order.
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

template <typename T>
inline const device T* pie_kv_plane(const device T* plane, ulong offset) {
  return reinterpret_cast<const device T*>(reinterpret_cast<const device uchar*>(plane) +
                                           offset);
}

template <typename T>
inline device T* pie_kv_plane(device T* plane, ulong offset) {
  return reinterpret_cast<device T*>(reinterpret_cast<device uchar*>(plane) + offset);
}
