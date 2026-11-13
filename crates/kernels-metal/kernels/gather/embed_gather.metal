#include <metal_stdlib>
using namespace metal;

template <int bits>
inline uint dequant_code(const device uint32_t* row, int k) {
  constexpr int per_word = 32 / bits;
  constexpr uint mask = (1u << bits) - 1u;
  return (row[k / per_word] >> ((k % per_word) * bits)) & mask;
}

// Where a block of a quantized table's rows lies in one plane: the GPU
// address of its first row (the weight cache's pool slot, or the plane's own
// copy), one a block.
struct BlockAt {
  const device char* at;
};

// `block_rows > 0` reads row `r`'s codes, scales and zero points through
// `codes_at`, `scales_at` and `biases_at[r / block_rows]`, row
// `r % block_rows` of that block, laid as in the table; the planes bound at
// 0..2 are then not read. `counting != 0` adds one to `hits[r / block_rows]`
// for each row a token reads: the weight cache's use record.
template <typename T, int group_size, int bits>
[[kernel]] void embed_gather_mb_4bit(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device int* id       [[buffer(3)]],
    device T* out              [[buffer(4)]],
    const constant int& hidden [[buffer(5)]],
    const constant int& vocab  [[buffer(6)]],
    const device BlockAt* codes_at  [[buffer(7)]],
    const device BlockAt* scales_at [[buffer(8)]],
    const device BlockAt* biases_at [[buffer(9)]],
    const constant int& block_rows  [[buffer(10)]],
    device atomic_uint* hits        [[buffer(11)]],
    const constant int& counting    [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int k = int(gid.x);
  const int m = int(gid.y);
  if (k >= hidden) return;
  const size_t out_at = size_t(m) * size_t(hidden) + size_t(k);
  const int row = id[m];
  if (row < 0 || row >= vocab) {
    out[out_at] = T(0);
    return;
  }
  const int packs_per_row = hidden / (32 / bits);
  const int groups_per_row = hidden / group_size;
  const int g = k / group_size;
  const device uint32_t* w_row = w + size_t(row) * size_t(packs_per_row);
  const device T* s_row = scales + size_t(row) * size_t(groups_per_row);
  const device T* b_row = biases + size_t(row) * size_t(groups_per_row);
  if (block_rows > 0) {
    const int block = row / block_rows, at = row % block_rows;
    w_row = reinterpret_cast<const device uint32_t*>(codes_at[block].at) +
            size_t(at) * size_t(packs_per_row);
    s_row = reinterpret_cast<const device T*>(scales_at[block].at) +
            size_t(at) * size_t(groups_per_row);
    b_row = reinterpret_cast<const device T*>(biases_at[block].at) +
            size_t(at) * size_t(groups_per_row);
    if (counting != 0 && k == 0) {
      atomic_fetch_add_explicit(&hits[block], 1u, memory_order_relaxed);
    }
  }
  const uint code = dequant_code<bits>(w_row, k);
  const float s = float(s_row[g]);
  const float b = float(b_row[g]);
  out[out_at] = static_cast<T>(s * float(code) + b);
}

#define instantiate_embed_mb(name, itype, gs, b)                            \
  template [[host_name("embed_gather_mb_4bit_" #name "_gs_" #gs "_b_" #b)]]  \
  [[kernel]] void embed_gather_mb_4bit<itype, gs, b>(                       \
      const device uint32_t*, const device itype*, const device itype*,     \
      const device int*, device itype*, const constant int&,                \
      const constant int&, const device BlockAt*, const device BlockAt*,    \
      const device BlockAt*, const constant int&, device atomic_uint*,      \
      const constant int&, uint2);

instantiate_embed_mb(bfloat16, bfloat, 64, 4)
instantiate_embed_mb(bfloat16, bfloat, 32, 4)
instantiate_embed_mb(bfloat16, bfloat, 128, 4)
instantiate_embed_mb(bfloat16, bfloat, 64, 8)
instantiate_embed_mb(bfloat16, bfloat, 32, 8)
instantiate_embed_mb(bfloat16, bfloat, 128, 8)
