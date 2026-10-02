#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

// PQ2_0 positional-2-bit decode-in-dot (Dtype::Pq2_0, spelled `g128_u2_f16_n`).
//
// A block is 34 bytes for 128 weights (2.125 bpw): a LEADING fp16 scale `d` at
// bytes 0-1 (not a trailing one, as PTQ1_0 has) + `qs[32]` holding positional
// 2-bit codes, 4 per byte. A weight is `value = (code - 1) * d`, code in
// {0,1,2,3} -> {-1,0,+1,+2}
// (an asymmetric quaternary with a `+2` outlier code, no zero-point). The scale
// is INLINE per block; there is NO separate scales plane. row_bytes =
// ceil(K/128) * 34.
//
// This mirrors the host oracle `checkpoint::codec::pq2_0::decode_block` byte for
// byte, which is itself bit-exact against the fork's own `dequantize_row_pq2_0`
// (PrismML-Eng llama.cpp, branch `prism`). Unlike PTQ1_0 there is NO base-3
// staging: element `j` lives in byte `j / 4` at bit offset `(j % 4) * 2`, so the
// layout is purely positional and decodes in natural element order 0..127. We
// decode a block the same way, inside the dot, so
// `y[row] = sum_i decoded_W[row][i] * x[i]`.
//
// We compute each decoded weight as `float(code - 1) * d` and then multiply by
// the activation, exactly as the host forms `(f32::from(code) - 1.0) * d`. With
// a one-hot activation only one term survives and the read-back is bit-exact
// against the host decoder (an integer in {-1,0,1,2} times a float is exact, and
// half->float is exact for every finite scale).

#define PQ2_0_BLOCK 128
#define PQ2_0_BLOCK_BYTES 34
#define PQ2_0_QS_OFFSET 2
#define SIMD_SIZE 32

// Dot one 34-byte block against the 128 activations at `xblk`, decoding weights
// positionally (element `j` -> byte `qs[j / 4]`, bit `(j % 4) * 2`). Returns
// `sum_j (code_j - 1) * d * x_j`.
template <typename Ti>
inline float pq2_0_block_dot(const device uint8_t* blk, const device Ti* xblk) {
  ushort dbits = ushort(uint(blk[0]) | (uint(blk[1]) << 8));
  half dh = as_type<half>(dbits);
  float d = float(dh);
  const device uint8_t* qs = blk + PQ2_0_QS_OFFSET;

  float acc = 0.0f;
  for (int j = 0; j < PQ2_0_BLOCK; j++) {
    int code = (int(qs[j >> 2]) >> ((j & 3) * 2)) & 0x03;
    float wv = float(code - 1) * d;
    acc += wv * float(xblk[j]);
  }
  return acc;
}

// Matrix-vector (and small-batch) decode-in-dot. Shape mirrors the affine
// `affine_qmv_fast` (and the PTQ1_0 `ptq1_0_qmv`): 2 simdgroups per threadgroup,
// 4 output rows per simdgroup, the 32 lanes split the K blocks and `simd_sum`
// reduces. Grid is `qmv_grid`: lanes [vecs*32, ceil(out/4), 1], group [32, 2, 1].
template <typename Ti, typename To>
METAL_FUNC void pq2_0_qmv_impl(
    const device uint8_t* w,
    const device Ti* x,
    device To* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;

  const int num_blocks = in_vec_size / PQ2_0_BLOCK;
  const int row_bytes = num_blocks * PQ2_0_BLOCK_BYTES;
  const int out_row = int(tid.y) * (num_simdgroups * results_per_simdgroup) +
      int(simd_gid) * results_per_simdgroup;

  const device Ti* x_vec = x + int(tid.x) * in_vec_size;

  float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int bk = int(simd_lid); bk < num_blocks; bk += SIMD_SIZE) {
    const device Ti* xblk = x_vec + bk * PQ2_0_BLOCK;
    for (int row = 0; row < results_per_simdgroup; row++) {
      int r = out_row + row;
      if (r >= out_vec_size) {
        continue;
      }
      const device uint8_t* blk = w + r * row_bytes + bk * PQ2_0_BLOCK_BYTES;
      result[row] += pq2_0_block_dot<Ti>(blk, xblk);
    }
  }

  device To* y_vec = y + int(tid.x) * out_vec_size + out_row;
  for (int row = 0; row < results_per_simdgroup; row++) {
    float v = simd_sum(result[row]);
    if (simd_lid == 0 && out_row + row < out_vec_size) {
      y_vec[row] = static_cast<To>(v);
    }
  }
}

template <typename Ti, typename To>
[[kernel]] void pq2_0_qmv(
    const device uint8_t* w          [[buffer(0)]],
    const device Ti* x               [[buffer(1)]],
    device To* y                     [[buffer(2)]],
    const constant int& in_vec_size  [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  pq2_0_qmv_impl<Ti, To>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

#define instantiate_pq2_0_qmv(name, itype, otype)                          \
  template [[host_name("pq2_0_qmv_" #name)]]                              \
  [[kernel]] void pq2_0_qmv<itype, otype>(                                 \
      const device uint8_t*, const device itype*, device otype*,           \
      const constant int&, const constant int&, uint3, uint, uint);

// Production path: bf16 activation in, bf16 out (pie's qwen_3 d27b decode).
instantiate_pq2_0_qmv(bfloat16, bfloat, bfloat)
// Bit-exact read-back path: bf16 activation in, f32 out (the M2b oracle test).
instantiate_pq2_0_qmv(bfloat16_f32, bfloat, float)
