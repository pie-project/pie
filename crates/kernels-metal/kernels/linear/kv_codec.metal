#include <metal_stdlib>
using namespace metal;

// C2a — the low-bit KV codec, in isolation. This file is the codec MATH only:
// pack (f32/bf16 -> packed) and unpack (packed -> f32/bf16). No cache, no
// allocation, no forward path touches it; it is driven directly from a test.
//
// LOCKED v1 format: 4-bit SYMMETRIC absmax, block = N elements (the model's
// full-attention head_dim — rotation-alignment), one fp16 scale per block, NO
// bias, the same scheme for K and V. The codec is PARAMETERIZED on the block N
// (stamped per size below, mirroring the Hadamard op fwht_simd/fwht_tg): pie
// serves many models, not just the head_dim-256 27B.
//   absmax = max|x| over the N-block
//   scale  = absmax / 7
//   q      = round(clamp(x / scale, -7, 7))   in [-7, 7]
//   dequant = q * scale
//
// Packed block layout, N/2 + 2 bytes:
//   bytes [0, N/2)     : N nibbles, two 4-bit codes per byte. Element 2*b sits
//                        in the LOW nibble of byte b, element 2*b+1 in the HIGH
//                        nibble (little-nibble-first, matching quant_transcode).
//   bytes [N/2, N/2+2) : one fp16 scale, LITTLE-ENDIAN (byte N/2 = low 8 bits).
// (At N = 256 this is the original 130-byte block: 128 nibble bytes + fp16.)
//
// Signed-nibble convention: OFFSET-BINARY. The stored nibble is q + 8, so
// q in [-7, 7] maps to nibbles [1, 15]; unpack recovers q as (nibble - 8).
// (nibble 0, i.e. q = -8, is never produced by this encoder — the clamp floor
// is -7.) A wrong convention shows up immediately as a large round-trip error.

#define KV_LIM 7.0f        // symmetric 4-bit limit, (1 << (4 - 1)) - 1

template <typename T, uint N>
inline void kv_pack_block(
    device const T* input,
    device uchar* packed,
    uint blocks,
    uint gid) {
  if (gid >= blocks) return;
  constexpr uint NIB_BYTES = N / 2u;   // packed nibble bytes per block
  constexpr uint BYTES = NIB_BYTES + 2u;  // + one fp16 scale
  const uint first = gid * N;
  device uchar* out = packed + gid * BYTES;

  // Per-block absmax reduction.
  float absmax = 0.0f;
  for (uint i = 0; i < N; ++i) {
    absmax = max(absmax, fabs(float(input[first + i])));
  }
  const float scale = absmax / KV_LIM;
  const bool live = absmax > 0.0f;

  // Quantize + pack two nibbles per byte (offset-binary q + 8).
  for (uint b = 0; b < NIB_BYTES; ++b) {
    const float lo_v = float(input[first + 2 * b]);
    const float hi_v = float(input[first + 2 * b + 1]);
    const int lo_q = live ? int(round(clamp(lo_v / scale, -KV_LIM, KV_LIM))) : 0;
    const int hi_q = live ? int(round(clamp(hi_v / scale, -KV_LIM, KV_LIM))) : 0;
    const uint lo_n = uint(lo_q + 8);  // [1, 15]
    const uint hi_n = uint(hi_q + 8);  // [1, 15]
    out[b] = uchar(((hi_n & 0xf) << 4) | (lo_n & 0xf));
  }

  // Store the fp16 scale, little-endian, without assuming half alignment.
  const half hs = half(scale);
  const ushort bits = as_type<ushort>(hs);
  out[NIB_BYTES] = uchar(bits & 0xff);
  out[NIB_BYTES + 1] = uchar((bits >> 8) & 0xff);
}

template <typename T, uint N>
inline void kv_unpack_block(
    device const uchar* packed,
    device T* out,
    uint blocks,
    uint gid) {
  if (gid >= blocks) return;
  constexpr uint NIB_BYTES = N / 2u;
  constexpr uint BYTES = NIB_BYTES + 2u;
  const uint first = gid * N;
  device const uchar* in = packed + gid * BYTES;

  const ushort bits =
      ushort(in[NIB_BYTES]) | (ushort(in[NIB_BYTES + 1]) << 8);
  const float scale = float(as_type<half>(bits));

  for (uint b = 0; b < NIB_BYTES; ++b) {
    const uchar byte = in[b];
    const int lo_q = int(byte & 0xf) - 8;
    const int hi_q = int((byte >> 4) & 0xf) - 8;
    out[first + 2 * b] = T(float(lo_q) * scale);
    out[first + 2 * b + 1] = T(float(hi_q) * scale);
  }
}

// Stamp the pack/unpack kernels per (dtype, block N). The entry name carries the
// block so the Rust wrapper can select it from the head_dim; N ∈ {64, 128, 256}.
#define instantiate_kv_codec(n)                                        \
  kernel void kv_pack_sym4_f32_##n(                                    \
      device const float* input [[buffer(0)]],                        \
      device uchar* packed [[buffer(1)]],                             \
      const constant uint& blocks [[buffer(2)]],                      \
      uint gid [[thread_position_in_grid]]) {                         \
    kv_pack_block<float, n##u>(input, packed, blocks, gid);           \
  }                                                                    \
  kernel void kv_pack_sym4_bf16_##n(                                   \
      device const bfloat* input [[buffer(0)]],                       \
      device uchar* packed [[buffer(1)]],                             \
      const constant uint& blocks [[buffer(2)]],                      \
      uint gid [[thread_position_in_grid]]) {                         \
    kv_pack_block<bfloat, n##u>(input, packed, blocks, gid);          \
  }                                                                    \
  kernel void kv_unpack_sym4_f32_##n(                                  \
      device const uchar* packed [[buffer(0)]],                       \
      device float* out [[buffer(1)]],                                \
      const constant uint& blocks [[buffer(2)]],                      \
      uint gid [[thread_position_in_grid]]) {                         \
    kv_unpack_block<float, n##u>(packed, out, blocks, gid);           \
  }                                                                    \
  kernel void kv_unpack_sym4_bf16_##n(                                 \
      device const uchar* packed [[buffer(0)]],                       \
      device bfloat* out [[buffer(1)]],                               \
      const constant uint& blocks [[buffer(2)]],                      \
      uint gid [[thread_position_in_grid]]) {                         \
    kv_unpack_block<bfloat, n##u>(packed, out, blocks, gid);          \
  }

instantiate_kv_codec(64)
instantiate_kv_codec(128)
instantiate_kv_codec(256)
