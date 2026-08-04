// Raw-Metal port of MLX affine_qmv_fast, scoped to Phase-0 decode (M=1, B=1).
//
// Source: mlx/backend/metal/kernels/quantized.h (qmv_fast_impl + helpers).
// Port notes:
//   * Decode is single-token GEMV -> only the qmv_fast path (M=1). The "fast"
//     variant requires N % 8 == 0 && K % 512 == 0, which holds for every qwen3.6
//     projection at group_size=64/4-bit (K in {1024,2048,3584}, all %512==0).
//   * batched branch DROPPED: B=1/M=1 decode never needs adjust_matrix_offsets,
//     so the 8 batch stride/shape buffers (7..14) are removed -> minimal binding
//     table (matches decode_abi bind::Qmv = {W, Scales, Biases, X, Y, K, N}).
//   * in_vec_size(K)/out_vec_size(N) are STATIC per weight (geometry, not per-token
//     IO scalars) -> safe as constant buffers under decode_abi I1.
//   * Helpers (get_pack_factor, get_bytes_per_pack, load_vector, qdot, qmv_fast_impl)
//     vendored verbatim from MLX so the math is bit-exact by construction.
//   * bfloat native on Metal 4 (no bf16.h).

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
MLX_MTL_CONST int SIMD_SIZE = 32;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}
template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

// The pre-division is how the 4-bit dot gets away with never unpacking: it
// multiplies the packed u16 word's nibbles in place, so the input carries the
// shift instead. At 8 bits a code already occupies a whole byte, so there is
// nothing to undo and the input is passed through.
template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(bits == 4 || bits == 8, "port covers the two widths mlx_lm ships");
  U sum = 0;
  if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(bits == 4 || bits == 8, "port covers the two widths mlx_lm ships");
  U accum = 0;
  if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}

// `packs_per_thread` is what sets the K stride: block_size = 32 * pack_factor *
// packs_per_thread, and the reduction loop assumes K is a whole number of blocks.
// At 4 bits that is 512 for the two-pack default and 256 for one pack, which is
// why gemma4's `per_layer_projection` (K=256) needs the narrow instantiation --
// the fast one reads 512 elements of a 256-element row.
template <typename T, int group_size, int bits, int packs_per_thread_ = 2>
METAL_FUNC void qmv_fast_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = packs_per_thread_;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_qmv_fast(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device T* x          [[buffer(3)]],
    device T* y                [[buffer(4)]],
    const constant int& in_vec_size  [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  qmv_fast_impl<T, group_size, bits>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// ── Residual-epilogue fusion (beta) ──────────────────────────────────────────
// y = matmul(x, W) + residual, folding the post-projection residual_add into the
// GEMV's write-back → removes one dispatch per residual edge (QmvO/QmvOut/QmvDown
// → 48 dispatches across the step). BIT-EXACT vs the 2-step qmv+residual_add: the
// intermediate `static_cast<T>(result)` reproduces the same per-element bf16 round
// the standalone qmv emits before residual_add re-reads it, so charlie's golden
// taps + argmax-264 are unperturbed. `residual` is the pre-block hidden stream,
// indexed identically to `y` (buffer(7); decode_abi bind::Qmv gains a Residual slot).
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast_residual_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const device T* residual,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;
  residual += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      T q = static_cast<T>(result[row]);                       // same bf16 round as standalone qmv
      y[row] = static_cast<T>(float(q) + float(residual[row]));  // then residual_add's round
    }
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_qmv_fast_residual(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device T* x          [[buffer(3)]],
    device T* y                [[buffer(4)]],
    const constant int& in_vec_size  [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const device T* residual   [[buffer(7)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  qmv_fast_residual_impl<T, group_size, bits>(
      w, scales, biases, x, y, residual, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

#define instantiate_qmv_fast(name, itype, gs, b)                         \
  template [[host_name("affine_qmv_fast_" #name "_gs_" #gs "_b_" #b)]]    \
  [[kernel]] void affine_qmv_fast<itype, gs, b>(                         \
      const device uint32_t*, const device itype*, const device itype*,  \
      const device itype*, device itype*, const constant int&,           \
      const constant int&, uint3, uint, uint);

instantiate_qmv_fast(bfloat16, bfloat, 64, 4)
instantiate_qmv_fast(bfloat16, bfloat, 32, 4)
instantiate_qmv_fast(bfloat16, bfloat, 128, 4)
instantiate_qmv_fast(bfloat16, bfloat, 64, 8)
instantiate_qmv_fast(bfloat16, bfloat, 32, 8)
instantiate_qmv_fast(bfloat16, bfloat, 128, 8)

#define instantiate_qmv_fast_residual(name, itype, gs, b)                       \
  template [[host_name("affine_qmv_fast_residual_" #name "_gs_" #gs "_b_" #b)]] \
  [[kernel]] void affine_qmv_fast_residual<itype, gs, b>(                       \
      const device uint32_t*, const device itype*, const device itype*,         \
      const device itype*, device itype*, const constant int&,                  \
      const constant int&, const device itype*, uint3, uint, uint);

instantiate_qmv_fast_residual(bfloat16, bfloat, 64, 4)
instantiate_qmv_fast_residual(bfloat16, bfloat, 32, 4)
instantiate_qmv_fast_residual(bfloat16, bfloat, 128, 4)
instantiate_qmv_fast_residual(bfloat16, bfloat, 64, 8)
instantiate_qmv_fast_residual(bfloat16, bfloat, 32, 8)
instantiate_qmv_fast_residual(bfloat16, bfloat, 128, 8)

// ── The two 4-bit codecs, as one axis ───────────────────────────────────────
//
// Affine-U4 and MXFP4 differ in four things and nothing else: the group width,
// how the group's scale is stored, whether there is a zero point, and what a
// code means. Everything around them -- the routing, the bounds-checked tail,
// the simd reduction, the bias epilogue -- is the same arithmetic on the same
// layout, and it is where the subtlety lives. So the codec is a template
// parameter rather than a second copy of the matvec.
#include "mxfp4_codec.h"
template <typename T, int BITS>
struct AffineQ {
  typedef T scale_t;
  MLX_MTL_CONST int bits = BITS;
  // Fixed, unlike the templated `group_size` of `qmv_fast` and `qmm_t` above.
  // This codec is only reached by the gpt-oss/routed matvec, and the only
  // checkpoints that dispatch it are published at 64 -- so a group axis here
  // would be one nothing on disk can exercise. A routed checkpoint at another
  // group fails when its pipeline is built, by name, which is where the width
  // has always failed too.
  MLX_MTL_CONST int group_size = 64;
  MLX_MTL_CONST bool zero_point = true;
  static METAL_FUNC float scale_of(scale_t s) { return float(s); }
  template <typename U, int VPT>
  static METAL_FUNC U prepare(const device T* x, thread U* x_thread) {
    return load_vector<T, U, VPT, BITS>(x, x_thread);
  }
  template <typename U, int VPT>
  static METAL_FUNC U dot(const device uint8_t* w, const thread U* x_thread, U scale,
                          U bias, U sum) {
    return qdot<U, VPT, BITS>(w, x_thread, scale, bias, sum);
  }
};
// The width is a third point on the same axis, not a third codec. `scale * code
// + bias` is linear at either width; only the unpacking differs, and that lives
// in `qdot`.
template <typename T> using AffineU4 = AffineQ<T, 4>;
template <typename T> using AffineU8 = AffineQ<T, 8>;

// The block exponent is E8M0: an unsigned power of two, 127-biased, with 0xff
// reserved for NaN. A code is a lookup and not a product, which is why MXFP4
// cannot borrow the affine dot -- that one multiplies the packed nibbles in
// place, and only gets away with it because `scale * code + bias` is LINEAR in
// the code. Nothing here is, so the nibbles are unpacked and the input is left
// alone rather than pre-divided by 16, 256 and 4096.
template <typename T>
struct Mxfp4 {
  typedef uint8_t scale_t;
  MLX_MTL_CONST int bits = 4;
  MLX_MTL_CONST int group_size = 32;
  MLX_MTL_CONST bool zero_point = false;
  static METAL_FUNC float scale_of(scale_t s) { return mxfp4_block_scale(s); }
  template <typename U, int VPT>
  static METAL_FUNC U prepare(const device T* x, thread U* x_thread) {
    for (int i = 0; i < VPT; i++) {
      x_thread[i] = U(x[i]);
    }
    return U(0);
  }
  template <typename U, int VPT>
  static METAL_FUNC U dot(const device uint8_t* w, const thread U* x_thread, U scale,
                          U, U) {
    U accum = 0;
    for (int i = 0; i < VPT / 2; i++) {
      const uint8_t byte = w[i];
      accum += x_thread[2 * i] * mxfp4_lo(byte) + x_thread[2 * i + 1] * mxfp4_hi(byte);
    }
    return scale * accum;
  }
};

// ── GPT-OSS matvec: arbitrary K, optional bias, optional expert routing ──────
//
// Three things this family needs that `affine_qmv_fast` cannot do.
//
// **K is 2880.** That is 64*45 -- a whole number of quantization groups, but NOT
// a multiple of the fast path's 512-wide reduction block, nor of the narrow
// one's 256. Every projection in the model is K=2880, so the reduction needs a
// bounds-checked tail: 11 full 256-blocks and 64 elements left over, which lanes
// 8..31 must sit out rather than read past the row.
//
// **Every projection is biased.** `attention_bias: true`, and the experts carry
// biases too, so the write-back has an additive epilogue.
//
// **Three of every layer's matvecs are ROUTED.** The MoE router picks 4 of 32
// experts ON THE GPU, so the weight this dispatch reads is not something the
// host can bind. The expert stack is one tensor per projection with the expert
// on axis 0, so the base offset is `expert_id * N * row_stride` -- and every
// stride is derivable from K and N, so routing costs one index buffer and no
// extra constants. `tid.z` selects which of the k slots this threadgroup serves;
// all of them read the same x and write disjoint rows of y.
template <typename T, typename Codec, bool BIASED, bool ROUTED>
METAL_FUNC void qmv_gptoss_impl(
    const device uint32_t* w,
    const device typename Codec::scale_t* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const device T* bias,
    const device int* expert_ids,
    int in_vec_size,
    int out_vec_size,
    int x_slot_stride,
    int x_row_stride,
    int slots_per_row,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int bits = Codec::bits;
  constexpr int group_size = Codec::group_size;
  constexpr int packs_per_thread = 1;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  // The routed base. Every stride is N * (per-row stride), so the expert axis
  // needs no constants of its own.
  //
  // `tid.x` is the token ROW. At M=1 it is always 0 and everything below is the
  // decode path unchanged; at M>1 each row picks its OWN k experts, so the
  // selection index is (row, slot) rather than slot -- the whole reason a
  // batched MoE is not just a wider launch.
  const int row = int(tid.x);
  const int slot = ROUTED ? int(tid.z) : 0;
  const int sel = row * slots_per_row + slot;
  if (ROUTED) {
    const size_t e = size_t(expert_ids[sel]);
    ws += e * size_t(out_vec_size) * size_t(in_vec_size_w);
    scales += e * size_t(out_vec_size) * size_t(in_vec_size_g);
    if (Codec::zero_point) {
      biases += e * size_t(out_vec_size) * size_t(in_vec_size_g);
    }
  }

  const device uint8_t* ws_row = ws + out_row * in_vec_size_w;
  const device typename Codec::scale_t* sc_row = scales + out_row * in_vec_size_g;
  const device T* bi_row = Codec::zero_point ? biases + out_row * in_vec_size_g : biases;
  // Whether the INPUT is per-expert too. `gate` and `up` read the one shared
  // norm output, so their stride is 0; `down` reads the SwiGLU's `[k,
  // intermediate]` stack, so its stride is K. Reading slot 0 for every expert
  // is not a crash -- it is four copies of the first expert's activation, which
  // survives all the way to a plausible wrong token.
  // `x_row_stride` is stated, not `in_vec_size`: `down` reads the SwiGLU's
  // [rows, k, intermediate] stack, whose row is k times as wide as its slot.
  const device T* x_row = x + row * x_row_stride + slot * x_slot_stride;

  for (int k = 0; k < in_vec_size; k += block_size) {
    const int base = k + int(simd_lid) * values_per_thread;
    // The tail: 2880 leaves 64 elements after eleven 256-blocks, so most lanes
    // have nothing to add. They must not read the next row's bytes.
    if (base + values_per_thread <= in_vec_size) {
      const device uint8_t* wl =
          ws_row + size_t(base) * size_t(bytes_per_pack) / size_t(pack_factor);
      U sum = Codec::template prepare<U, values_per_thread>(x_row + base, x_thread);
      const int g = base / group_size;
      for (int row = 0; row < results_per_simdgroup; row++) {
        const device uint8_t* wr = wl + row * in_vec_size_w;
        U s = Codec::scale_of(sc_row[row * in_vec_size_g + g]);
        U b = Codec::zero_point ? U(bi_row[row * in_vec_size_g + g]) : U(0);
        result[row] += Codec::template dot<U, values_per_thread>(wr, x_thread, s, b, sum);
      }
    }
  }

  device T* y_row = y + (ROUTED ? sel : row) * out_vec_size + out_row;
  const device T* bias_row = bias;
  if (BIASED && ROUTED) {
    bias_row += size_t(expert_ids[sel]) * size_t(out_vec_size);
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U v = simd_sum(result[row]);
    if (simd_lid == 0) {
      if (BIASED) v += U(bias_row[out_row + row]);
      y_row[row] = static_cast<T>(v);
    }
  }
}

// The codec is a template-template parameter rather than baked in by the
// macro, so a kernel's SHAPE (biased? routed?) and its ENCODING are chosen
// independently. That is what lets the same `qmv_bias` serve the 4-bit
// attention projections and the 8-bit router.
#define gptoss_qmv_kernel(name, BIASED, ROUTED)                                \
  template <typename T, template <typename> class Codec>                       \
  [[kernel]] void name(                                                        \
      const device uint32_t* w   [[buffer(0)]],                                \
      const device typename Codec<T>::scale_t* scales [[buffer(1)]],           \
      const device T* biases     [[buffer(2)]],                                \
      const device T* x          [[buffer(3)]],                                \
      device T* y                [[buffer(4)]],                                \
      const constant int& in_vec_size  [[buffer(5)]],                          \
      const constant int& out_vec_size [[buffer(6)]],                          \
      const device T* bias       [[buffer(7)]],                                \
      const device int* expert_ids [[buffer(8)]],                              \
      const constant int& x_slot_stride [[buffer(9)]],                         \
      const constant int& x_row_stride  [[buffer(10)]],                        \
      const constant int& slots_per_row [[buffer(11)]],                        \
      uint3 tid       [[threadgroup_position_in_grid]],                        \
      uint simd_gid   [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid   [[thread_index_in_simdgroup]]) {                         \
    qmv_gptoss_impl<T, Codec<T>, BIASED, ROUTED>(                              \
        w, scales, biases, x, y, bias, expert_ids, in_vec_size, out_vec_size,  \
        x_slot_stride, x_row_stride, slots_per_row, tid, simd_gid, simd_lid);  \
  }

gptoss_qmv_kernel(qmv_tail, false, false)
gptoss_qmv_kernel(qmv_tail_bias, true, false)
// Qwen3-MoE's experts carry no bias. Everything else about the routed path --
// the stacked weights indexed by `expert_ids`, `tid.z` selecting the slot -- is
// identical, so it is the same template with BIASED off rather than a kernel.
gptoss_qmv_kernel(qmv_routed_bias, true, true)
gptoss_qmv_kernel(qmv_routed, false, true)

#define instantiate_gptoss_qmv(host, fn, codec, name, itype, gs, b)           \
  template [[host_name(#host "_" #name "_gs_" #gs "_b_" #b)]]                 \
  [[kernel]] void fn<itype, codec>(                                           \
      const device uint32_t*, const device codec<itype>::scale_t*,            \
      const device itype*,                                                    \
      const device itype*, device itype*, const constant int&,                \
      const constant int&, const device itype*, const device int*,            \
      const constant int&, const constant int&, const constant int&,          \
      uint3, uint, uint);

instantiate_gptoss_qmv(affine_qmv_tail, qmv_tail, AffineU4, bfloat16, bfloat, 64, 4)
instantiate_gptoss_qmv(affine_qmv_tail_bias, qmv_tail_bias, AffineU4, bfloat16, bfloat, 64, 4)
instantiate_gptoss_qmv(affine_qmv_routed_bias, qmv_routed_bias, AffineU4, bfloat16, bfloat, 64, 4)
instantiate_gptoss_qmv(affine_qmv_routed, qmv_routed, AffineU4, bfloat16, bfloat, 64, 4)

// The same matvec against the checkpoint's own MXFP4 experts. Only gpt-oss has
// them, and only its experts -- attention, router, embedding and head are all
// published affine -- so the routed-and-biased shape is the only one MXFP4
// needs.
instantiate_gptoss_qmv(mxfp4_qmv_routed_bias, qmv_routed_bias, Mxfp4, bfloat16, bfloat, 32, 4)

// gpt-oss's router. `mlx_lm`'s predicate leaves it at 8 bits while everything
// around it goes to 4 -- 32 x 2880, small enough that the width costs nothing
// and sensitive enough that it matters. It used to be a hand-written kernel
// with its own launch geometry and its own special case in two dispatchers;
// once the width is a codec parameter it is the dense biased matvec at b=8,
// and all of that goes away.
instantiate_gptoss_qmv(affine_qmv_tail_bias, qmv_tail_bias, AffineU8, bfloat16, bfloat, 64, 8)

// gemma 4's unaligned K, at both of the widths one checkpoint can hold. The
// 26B is hidden 2816 and `intermediate` 2112: 2816 is 5.5 of the 4-bit fast
// path's 512-wide blocks, and 2112 is 8.25 of the 8-bit path's 256, so the
// aligned matvec reduces a block of the NEXT row into the last rows of every
// projection -- which past the end of the tensor is a NaN, and inside it is a
// wrong number that still looks like text.
instantiate_gptoss_qmv(affine_qmv_tail, qmv_tail, AffineU8, bfloat16, bfloat, 64, 8)
