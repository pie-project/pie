#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

// PTQ1_0 ternary decode-in-dot (Dtype::Ptq1_0, spelled `g128_t3_f16_n`).
//
// A block is 28 bytes for 128 weights (1.75 bpw): `qs[24]` (5 base-3 trits per
// byte) + `qh[2]` (4 trits per byte) + a trailing fp16 scale `d`. A weight is
// `value = (trit - 1) * d`, trit in {0,1,2} -> {-1,0,+1}. The scale is INLINE
// per block; there is NO separate scales plane. row_bytes = ceil(K/128) * 28.
//
// This mirrors the host oracle `checkpoint::codec::ptq1_0::decode_block` byte
// for byte, which is itself bit-exact against the fork's own
// `dequantize_row_ptq1_0` (PrismML-Eng llama.cpp, branch `prism`). The subtle
// part is the *staging*: for a 24-byte `qs` the fork walks chunks {16,8} (the
// 32-chunk never fits), emitting elements in NATURAL order 0..127; the `qh`
// tail carries elements 120..127. We decode a block the same way, inside the
// dot, so `y[row] = sum_i decoded_W[row][i] * x[i]`.
//
// We compute each decoded weight as `float(trit - 1) * d` and then multiply by
// the activation, exactly as the host forms `f32::from(trit-1) * d`. With a
// one-hot activation only one term survives and the read-back is bit-exact
// against the host decoder (an integer in {-1,0,1} times a float is exact, and
// half->float is exact for every finite scale).

#define PTQ1_0_BLOCK 128
#define PTQ1_0_BLOCK_BYTES 28
#define PTQ1_0_QS_LEN 24
#define SIMD_SIZE 32

// Base-3 place weights 3^n, as the fork/host use to shift a trit into the byte
// MSBs before reading it back: `q = (byte * 3^n) & 0xff; trit = (q * 3) >> 8`.
constant int kPow3[5] = {1, 3, 9, 27, 81};

// One base-3 trit from a packed byte at place `n` (0,1,2), matching the host's
// `trit()` arithmetic exactly.
inline int ptq1_0_trit(uint8_t byte, int n) {
  int q = (int(byte) * kPow3[n]) & 0xff;
  return (q * 3) >> 8;
}

// Dot one 28-byte block against the 128 activations at `xblk`, decoding weights
// in the fork's {16,8}+qh staging (natural element order). Returns
// `sum_i (trit_i - 1) * d * x_i`.
template <typename Ti>
inline float ptq1_0_block_dot(const device uint8_t* blk, const device Ti* xblk) {
  const device uint8_t* qs = blk;
  const device uint8_t* qh = blk + PTQ1_0_QS_LEN;
  ushort dbits = ushort(uint(blk[26]) | (uint(blk[27]) << 8));
  half dh = as_type<half>(dbits);
  float d = float(dh);

  float acc = 0.0f;
  int idx = 0;

  // qs stages: a 16-byte chunk (elements 0..79) then an 8-byte chunk (80..119).
  // trit-slot `n` outer, byte `m` inner, is what emits natural element order.
  for (int chunk = 0; chunk < 2; chunk++) {
    int base = chunk == 0 ? 0 : 16;
    int c = chunk == 0 ? 16 : 8;
    for (int n = 0; n < 5; n++) {
      for (int m = 0; m < c; m++) {
        int t = ptq1_0_trit(qs[base + m], n);
        float wv = float(t - 1) * d;
        acc += wv * float(xblk[idx]);
        idx++;
      }
    }
  }

  // qh tail: elements 120..127, 4 trits per byte.
  for (int n = 0; n < 4; n++) {
    for (int m = 0; m < 2; m++) {
      int t = ptq1_0_trit(qh[m], n);
      float wv = float(t - 1) * d;
      acc += wv * float(xblk[idx]);
      idx++;
    }
  }

  return acc;
}

// Matrix-vector (and small-batch) decode-in-dot. Shape mirrors the affine
// `affine_qmv_fast`: 2 simdgroups per threadgroup, 4 output rows per simdgroup,
// the 32 lanes split the K blocks and `simd_sum` reduces. Grid is `qmv_grid`:
// lanes [vecs*32, ceil(out/4), 1], group [32, 2, 1].
template <typename Ti, typename To>
METAL_FUNC void ptq1_0_qmv_impl(
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

  const int num_blocks = in_vec_size / PTQ1_0_BLOCK;
  const int row_bytes = num_blocks * PTQ1_0_BLOCK_BYTES;
  const int out_row = int(tid.y) * (num_simdgroups * results_per_simdgroup) +
      int(simd_gid) * results_per_simdgroup;

  const device Ti* x_vec = x + int(tid.x) * in_vec_size;

  float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int bk = int(simd_lid); bk < num_blocks; bk += SIMD_SIZE) {
    const device Ti* xblk = x_vec + bk * PTQ1_0_BLOCK;
    for (int row = 0; row < results_per_simdgroup; row++) {
      int r = out_row + row;
      if (r >= out_vec_size) {
        continue;
      }
      const device uint8_t* blk = w + r * row_bytes + bk * PTQ1_0_BLOCK_BYTES;
      result[row] += ptq1_0_block_dot<Ti>(blk, xblk);
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
[[kernel]] void ptq1_0_qmv(
    const device uint8_t* w          [[buffer(0)]],
    const device Ti* x               [[buffer(1)]],
    device To* y                     [[buffer(2)]],
    const constant int& in_vec_size  [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  ptq1_0_qmv_impl<Ti, To>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

#define instantiate_ptq1_0_qmv(name, itype, otype)                         \
  template [[host_name("ptq1_0_qmv_" #name)]]                              \
  [[kernel]] void ptq1_0_qmv<itype, otype>(                                \
      const device uint8_t*, const device itype*, device otype*,           \
      const constant int&, const constant int&, uint3, uint, uint);

// Production path: bf16 activation in, bf16 out (pie's qwen_3 d27b decode).
instantiate_ptq1_0_qmv(bfloat16, bfloat, bfloat)
// Bit-exact read-back path: bf16 activation in, f32 out (the M1b oracle test).
instantiate_ptq1_0_qmv(bfloat16_f32, bfloat, float)
