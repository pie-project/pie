// Raw-Metal port of the tied-quantized embedding gather for Phase-0 decode.
//
// Source semantics: weights.hpp apply_embedding (quantized path) ==
//   take(weight, id, axis0) + take(scales/biases) + dequantize(gs, bits).
// qwen3.6 uses a TIED 4-bit lm_head as the embed table (dense embed dropped for
// true-4-bit parity): gather + affine-dequant ONE row [hidden] for the token.
// Port notes (M=1 decode, group_size=64, bits=4, hidden=1024):
//   * token_id IS the per-token IO scalar -> read from a *buffer* (id[0]) per
//     decode_abi I1, never setBytes.
//   * Same 4-bit affine packing as qmv: 8 nibbles/uint32, per-group scale+bias.
//   * One thread per output channel k -> out[k] = scale[g]*nibble_k + bias[g].
//   * Output goes to the resident hidden slot (logits path is the separate qmv).
// Launch: dispatchThreads grid=(hidden, 1, 1). bfloat native on Metal 4.

#include <metal_stdlib>
using namespace metal;

template <typename T, int group_size, int bits>
[[kernel]] void embed_gather_4bit(
    const device uint32_t* w   [[buffer(0)]],  // [vocab, hidden/8] packed
    const device T* scales     [[buffer(1)]],  // [vocab, hidden/group_size]
    const device T* biases     [[buffer(2)]],  // [vocab, hidden/group_size]
    const device int* id       [[buffer(3)]],  // IO scalar (I1): id[0] = token
    device T* out              [[buffer(4)]],  // [hidden]
    const constant int& hidden [[buffer(5)]],
    uint k [[thread_position_in_grid]]) {
  if ((int)k >= hidden) return;
  const int row = id[0];
  const int packs_per_row = hidden / 8;        // 8 nibbles per uint32
  const int groups_per_row = hidden / group_size;

  const int g = (int)k / group_size;
  const uint32_t pack = w[row * packs_per_row + (int)k / 8];
  const uint nibble = (pack >> (((int)k % 8) * 4)) & 0xf;

  const float s = static_cast<float>(scales[row * groups_per_row + g]);
  const float b = static_cast<float>(biases[row * groups_per_row + g]);
  out[k] = static_cast<T>(s * static_cast<float>(nibble) + b);
}

#define instantiate_embed(name, itype, gs, b)                        \
  template [[host_name("embed_gather_4bit_" #name "_gs_" #gs "_b_" #b)]] \
  [[kernel]] void embed_gather_4bit<itype, gs, b>(                   \
      const device uint32_t*, const device itype*, const device itype*, \
      const device int*, device itype*, const constant int&, uint);

instantiate_embed(float32, float, 64, 4)
instantiate_embed(float16, half, 64, 4)
instantiate_embed(bfloat16, bfloat, 64, 4)

// ── Scaled variant (gemma4): out[k] = embed_scale * dequant(row, k). ─────────────
// gemma4 multiplies the gathered embedding by a constant (embed_tokens: sqrt(hidden);
// embed_tokens_per_layer: sqrt(per_layer_emb_dim)). Same tied 4-bit table as the
// unscaled gather + lm_head — the scale must NOT be folded into the shared weights
// (lm_head reuses them), so it is applied here on the embed path only. Extra buffer 6
// = const float scale; qwen's `embed_gather_4bit` (no buffer 6) is untouched.
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
  if ((int)k >= hidden) return;
  const int row = id[0];
  const int packs_per_row = hidden / 8;
  const int groups_per_row = hidden / group_size;
  const int g = (int)k / group_size;
  const uint32_t pack = w[row * packs_per_row + (int)k / 8];
  const uint nibble = (pack >> (((int)k % 8) * 4)) & 0xf;
  const float s = static_cast<float>(scales[row * groups_per_row + g]);
  const float b = static_cast<float>(biases[row * groups_per_row + g]);
  out[k] = static_cast<T>((s * static_cast<float>(nibble) + b) * embed_scale);
}

#define instantiate_embed_scaled(name, itype, gs, b)                            \
  template [[host_name("embed_gather_scaled_4bit_" #name "_gs_" #gs "_b_" #b)]] \
  [[kernel]] void embed_gather_scaled_4bit<itype, gs, b>(                       \
      const device uint32_t*, const device itype*, const device itype*,         \
      const device int*, device itype*, const constant int&,                    \
      const constant float&, uint);

instantiate_embed_scaled(float32, float, 64, 4)
instantiate_embed_scaled(float16, half, 64, 4)
instantiate_embed_scaled(bfloat16, bfloat, 64, 4)
