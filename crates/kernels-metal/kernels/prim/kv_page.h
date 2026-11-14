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

// A pool past one buffer's length spans as many buffers as it takes
// (`engine-metal`'s `store::pool`): a page id then names its buffer in the
// bits over `pie_fc_part_shift` (the bits that count the buffers, under the
// sign bit, so every id stays a non-negative `int` in the page tables) and
// its page in that buffer under them, and a kernel reaches buffer `b` from
// the plane bound in the first by buffer `b`'s address less the first's:
// entry `b` of the delta table at device address `pie_fc_part_table` (bytes,
// wrapping), which lies in the pool's first buffer. Function constants,
// fixed for a load, so a pool of one buffer compiles to `page * stride`,
// bit for bit.
constant bool pie_fc_pool_split [[function_constant(2)]];
constant uint pie_fc_part_shift [[function_constant(3)]];
constant ulong pie_fc_part_table [[function_constant(4)]];
constant bool pie_pool_split =
    is_function_constant_defined(pie_fc_pool_split) && pie_fc_pool_split;

// A page id names a slot (`engine-metal`'s `store::pool::PageIds`): under
// its buffer, its slot in the buffer where every paged shape's slots tile a
// chunk, else its chunk in the buffer over `pie_fc_slot_shift` bits and its
// index in that chunk under them, a chunk being `pie_fc_chunk_bytes`. Load
// constants: a pool whose slots tile its chunks compiles to `page * stride`.
constant bool pie_fc_slot_split [[function_constant(6)]];
constant uint pie_fc_slot_shift [[function_constant(7)]];
constant ulong pie_fc_chunk_bytes [[function_constant(8)]];
constant bool pie_slot_split =
    is_function_constant_defined(pie_fc_slot_split) && pie_fc_slot_split;

// Where page `page` starts, from the plane bound in the first buffer, in
// units of `unit` bytes (an element's, or 1 for bytes): `stride` units a
// page (a slot).
inline ulong pie_page_at(uint page, ulong stride, ulong unit) {
  if (!pie_pool_split && !pie_slot_split) {
    return ulong(page) * stride;
  }
  const uint local = pie_pool_split ? page & ((1u << pie_fc_part_shift) - 1u) : page;
  const ulong at =
      pie_slot_split
          ? ulong(local >> pie_fc_slot_shift) * (pie_fc_chunk_bytes / unit) +
                ulong(local & ((1u << pie_fc_slot_shift) - 1u)) * stride
          : ulong(local) * stride;
  if (!pie_pool_split) {
    return at;
  }
  const uint part = page >> pie_fc_part_shift;
  const ulong delta =
      part == 0u ? 0ul : reinterpret_cast<const device ulong*>(pie_fc_part_table)[part];
  return at + ulong(long(delta) / long(unit));
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
