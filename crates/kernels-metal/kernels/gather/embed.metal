#include <metal_stdlib>
using namespace metal;

// Where a block of a table's rows lies: the GPU address of its first row
// (the weight cache's pool slot, or the table's own copy), written by the
// host into a buffer of these, one a block.
struct RowAt {
  const device char* at;
};

// `block_rows > 0` reads row `r` through `rows_at[r / block_rows]`, row
// `r % block_rows` of that block, `hidden` elements a row as in the table;
// `table` is then not read. `block_rows == 0` reads the contiguous `table`
// (`rows_at` and `hits` unbound). `counting != 0` adds one to
// `hits[r / block_rows]` for each row a token reads: the weight cache's use
// record.
template <typename T, bool ZERO_OOB>
[[kernel]] void embed(
    const device int* ids           [[buffer(0)]],
    const device T* table           [[buffer(1)]],
    device T* y                     [[buffer(2)]],
    const constant int& hidden      [[buffer(3)]],
    const constant int& vocab       [[buffer(4)]],
    const device RowAt* rows_at     [[buffer(5)]],
    const constant int& block_rows  [[buffer(6)]],
    device atomic_uint* hits        [[buffer(7)]],
    const constant int& counting    [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int c = int(tid.x);

  if (c >= hidden) {
    return;
  }
  const size_t n = size_t(tid.y);
  const int raw = ids[n];
  const bool addressable = (raw >= 0 && raw < vocab);
  const size_t at = n * size_t(hidden) + size_t(c);

  if (ZERO_OOB && !addressable) {
    y[at] = T(0);
    return;
  }
  const int row = addressable ? raw : 0;
  if (block_rows > 0) {
    const int block = row / block_rows;
    const device T* first = reinterpret_cast<const device T*>(rows_at[block].at);
    y[at] = first[size_t(row % block_rows) * size_t(hidden) + size_t(c)];
    if (counting != 0 && c == 0) {
      atomic_fetch_add_explicit(&hits[block], 1u, memory_order_relaxed);
    }
    return;
  }
  y[at] = table[size_t(row) * size_t(hidden) + size_t(c)];
}

#define instantiate_embed_dense(name, itype)                                \
  template [[host_name("embed_" #name)]]                                    \
  [[kernel]] void embed<itype, false>(                                      \
      const device int*, const device itype*, device itype*,                \
      const constant int&, const constant int&, const device RowAt*,        \
      const constant int&, device atomic_uint*, const constant int&,        \
      uint2);                                                               \
  template [[host_name("embed_concat_" #name)]]                             \
  [[kernel]] void embed<itype, true>(                                       \
      const device int*, const device itype*, device itype*,                \
      const constant int&, const constant int&, const device RowAt*,        \
      const constant int&, device atomic_uint*, const constant int&,        \
      uint2);

instantiate_embed_dense(bfloat16, bfloat)
