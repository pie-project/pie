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
// The page strides as function constants where the pipeline was specialized
// on them (`Fire::page_strides`): fixed for a loaded model, so a kernel
// folds them; else the arguments' runtime values.
constant ulong pie_fc_k_page [[function_constant(0)]];
constant ulong pie_fc_v_page [[function_constant(1)]];
constant bool pie_fc_pages = is_function_constant_defined(pie_fc_k_page) &&
                             is_function_constant_defined(pie_fc_v_page);

// A pool past one buffer's length spans up to four (`engine-metal`'s
// `store::pool`): a page id then names its buffer in bits 29 and 30 and its
// page in that buffer under them (`PART_SHIFT`), so every id stays a
// non-negative `int` in the page tables, and a kernel reaches buffer `b` from
// the plane bound in the first by buffer `b`'s address less the first's
// (`pie_fc_pool_delta1..3`, bytes, wrapping). Function constants, fixed for
// a load, so a pool of one buffer compiles to `page * stride`, bit for bit.
constant bool pie_fc_pool_split [[function_constant(2)]];
constant ulong pie_fc_pool_delta1 [[function_constant(3)]];
constant ulong pie_fc_pool_delta2 [[function_constant(4)]];
constant ulong pie_fc_pool_delta3 [[function_constant(5)]];
constant bool pie_pool_split =
    is_function_constant_defined(pie_fc_pool_split) && pie_fc_pool_split;

// Where page `page` starts, from the plane bound in the first buffer, in
// units of `unit` bytes (an element's, or 1 for bytes): `stride` units a
// page.
inline ulong pie_page_at(uint page, ulong stride, ulong unit) {
  if (!pie_pool_split) {
    return ulong(page) * stride;
  }
  const uint part = page >> 29;
  const ulong delta = part == 0u   ? 0ul
                      : part == 1u ? pie_fc_pool_delta1
                      : part == 2u ? pie_fc_pool_delta2
                                   : pie_fc_pool_delta3;
  return ulong(page & 0x1fffffffu) * stride + ulong(long(delta) / long(unit));
}

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
