#include <metal_stdlib>
using namespace metal;

template <int bits>
inline uint dequant_code(const device uint32_t* row, int k) {
  constexpr int per_word = 32 / bits;
  constexpr uint mask = (1u << bits) - 1u;
  return (row[k / per_word] >> ((k % per_word) * bits)) & mask;
}

template <typename T, int group_size, int bits, bool SCALED>
METAL_FUNC void embed_gather_body(
    const device uint32_t* w, const device T* scales, const device T* biases,
    device T* out, int hidden, int row, int k, size_t out_at,
    float embed_scale) {
  if (k >= hidden) return;
  const int packs_per_row = hidden / (32 / bits);
  const int groups_per_row = hidden / group_size;
  const int g = k / group_size;
  const uint code = dequant_code<bits>(w + row * packs_per_row, k);
  const float s = float(scales[row * groups_per_row + g]);
  const float b = float(biases[row * groups_per_row + g]);
  const float value = s * float(code) + b;
  out[out_at] = static_cast<T>(SCALED ? value * embed_scale : value);
}

template <typename T, int group_size, int bits>
[[kernel]] void embed_gather_4bit(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device int* id       [[buffer(3)]],
    device T* out              [[buffer(4)]],
    const constant int& hidden [[buffer(5)]],
    uint k [[thread_position_in_grid]]) {
  embed_gather_body<T, group_size, bits, false>(
      w, scales, biases, out, hidden, id[0], int(k), size_t(k), 1.0f);
}

#define instantiate_embed(name, itype, gs, b)                        \
  template [[host_name("embed_gather_4bit_" #name "_gs_" #gs "_b_" #b)]] \
  [[kernel]] void embed_gather_4bit<itype, gs, b>(                   \
      const device uint32_t*, const device itype*, const device itype*, \
      const device int*, device itype*, const constant int&, uint);

instantiate_embed(bfloat16, bfloat, 64, 4)
instantiate_embed(bfloat16, bfloat, 32, 4)
instantiate_embed(bfloat16, bfloat, 128, 4)
instantiate_embed(bfloat16, bfloat, 64, 8)
instantiate_embed(bfloat16, bfloat, 32, 8)
instantiate_embed(bfloat16, bfloat, 128, 8)

template <typename T, int group_size, int bits>
[[kernel]] void embed_gather_scaled_4bit(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device int* id       [[buffer(3)]],
    device T* out              [[buffer(4)]],
    const constant int& hidden [[buffer(5)]],
    const constant float& embed_scale [[buffer(6)]],
    uint k [[thread_position_in_grid]]) {
  embed_gather_body<T, group_size, bits, true>(
      w, scales, biases, out, hidden, id[0], int(k), size_t(k), embed_scale);
}

#define instantiate_embed_scaled(name, itype, gs, b)                            \
  template [[host_name("embed_gather_scaled_4bit_" #name "_gs_" #gs "_b_" #b)]] \
  [[kernel]] void embed_gather_scaled_4bit<itype, gs, b>(                       \
      const device uint32_t*, const device itype*, const device itype*,         \
      const device int*, device itype*, const constant int&,                    \
      const constant float&, uint);

instantiate_embed_scaled(bfloat16, bfloat, 64, 4)
instantiate_embed_scaled(bfloat16, bfloat, 32, 4)
instantiate_embed_scaled(bfloat16, bfloat, 128, 4)
instantiate_embed_scaled(bfloat16, bfloat, 64, 8)
instantiate_embed_scaled(bfloat16, bfloat, 32, 8)
instantiate_embed_scaled(bfloat16, bfloat, 128, 8)

template <typename T, int group_size, int bits>
[[kernel]] void embed_gather_mb_4bit(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device int* id       [[buffer(3)]],
    device T* out              [[buffer(4)]],
    const constant int& hidden [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int k = int(gid.x);
  const int m = int(gid.y);
  embed_gather_body<T, group_size, bits, false>(
      w, scales, biases, out, hidden, id[m], k,
      size_t(m) * size_t(hidden) + size_t(k), 1.0f);
}

#define instantiate_embed_mb(name, itype, gs, b)                            \
  template [[host_name("embed_gather_mb_4bit_" #name "_gs_" #gs "_b_" #b)]]  \
  [[kernel]] void embed_gather_mb_4bit<itype, gs, b>(                       \
      const device uint32_t*, const device itype*, const device itype*,     \
      const device int*, device itype*, const constant int&, uint2);

instantiate_embed_mb(bfloat16, bfloat, 64, 4)
instantiate_embed_mb(bfloat16, bfloat, 32, 4)
instantiate_embed_mb(bfloat16, bfloat, 128, 4)
instantiate_embed_mb(bfloat16, bfloat, 64, 8)
instantiate_embed_mb(bfloat16, bfloat, 32, 8)
instantiate_embed_mb(bfloat16, bfloat, 128, 8)

template <typename T, int group_size, int bits>
[[kernel]] void embed_gather_scaled_mb_4bit(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device int* id       [[buffer(3)]],
    device T* out              [[buffer(4)]],
    const constant int& hidden [[buffer(5)]],
    const constant float& embed_scale [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int k = int(gid.x);
  const int m = int(gid.y);
  embed_gather_body<T, group_size, bits, true>(
      w, scales, biases, out, hidden, id[m], k,
      size_t(m) * size_t(hidden) + size_t(k), embed_scale);
}

#define instantiate_embed_scaled_mb(name, itype, gs, b)                            \
  template [[host_name("embed_gather_scaled_mb_4bit_" #name "_gs_" #gs "_b_" #b)]] \
  [[kernel]] void embed_gather_scaled_mb_4bit<itype, gs, b>(                       \
      const device uint32_t*, const device itype*, const device itype*,           \
      const device int*, device itype*, const constant int&,                      \
      const constant float&, uint2);

instantiate_embed_scaled_mb(bfloat16, bfloat, 64, 4)
instantiate_embed_scaled_mb(bfloat16, bfloat, 32, 4)
instantiate_embed_scaled_mb(bfloat16, bfloat, 128, 4)
instantiate_embed_scaled_mb(bfloat16, bfloat, 64, 8)
instantiate_embed_scaled_mb(bfloat16, bfloat, 32, 8)
instantiate_embed_scaled_mb(bfloat16, bfloat, 128, 8)
