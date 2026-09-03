#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
constant float kMxfp4Lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float mxfp4_lo(uint8_t byte) { return kMxfp4Lut[byte & 0xf]; }
inline float mxfp4_hi(uint8_t byte) { return kMxfp4Lut[byte >> 4]; }

inline float mxfp4_block_scale(uint8_t code) {
  return code == 0xff ? NAN : metal::ldexp(1.0f, int(code) - 127);
}

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
MLX_MTL_CONST int SIMD_SIZE = 32;

// The simdgroup split down M, as a function of the row block — the DEFAULT
// every template below takes for `WM`, so that a row block names its own
// warp shape instead of each instantiation restating it.
//
// `mlx::steel::BlockMMA` computes `TM = BM / (kFragSize * WM)` with
// `kFragSize = 8`, so `BM = 8` admits `WM = 1` AND NOTHING ELSE: at `WM = 2`
// every simdgroup gets a zero-row warp tile and the MMA computes nothing at
// all. That single constraint is why the 8 rung needed a source
// instantiation rather than another argument to the stamp — see `BM_RUNGS`
// in `kernels-metal/src/linear/quant.rs` for what it buys.
//
// The threadgroup shrinks with it: `WM * WN * SIMD_SIZE` is 64 lanes at the 8
// rung against 128 at every rung above, which the host has to launch to match
// (`quant::qmm_group`), because both block loaders divide their tile by
// `tgp_size` and a loader told 128 while 64 arrive leaves half the tile
// unwritten.
template <int BM>
inline constexpr int qmm_wm() {
  return BM < 16 ? 1 : 2;
}

// The lanes a row block's threadgroup holds. `WN = 2` throughout this file —
// the one place a column split of four appears is a hand-written
// instantiation at the bottom — so a block loader that carries no `WN` of its
// own reads its `tgp_size` from here rather than from a literal `4 *
// SIMD_SIZE`, which was correct for exactly the rungs that predate the 8.
template <int BM>
inline constexpr int qmm_tgp() {
  return qmm_wm<BM>() * 2 * SIMD_SIZE;
}

#include "../third_party/mlx_steel_prelude.metal"
#include "../third_party/mlx_steel_transforms.metal"
#include "../third_party/mlx_steel_mma.metal"
#include "../third_party/mlx_steel_loader.metal"
#include "../third_party/mlx_quantized_block.metal"

template <typename T, typename P, typename LoaderW, int BM, int BK, int BN,
          bool WITH_RESIDUAL, bool WITH_BIAS, int WM = qmm_wm<BM>(),
          int WN = 2>
METAL_FUNC void qmm_t_loaded_impl(
    const device T* x,
    device P* y,
    const device P* residual,
    threadgroup T* Xs,
    threadgroup T* Ws,
    int x_row_stride,
    int y_row_stride,
    int k_len,
    uint3 tid,
    uint simd_gid,
    uint simd_lid,
    thread LoaderW& loader_w) {
  constexpr int BK_padded = BK + 16 / sizeof(T);
  using mma_t = mlx::steel::
      BlockMMA<T, P, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;

  const int y_row = int(tid.y) * BM;
  const int y_col = int(tid.x) * BN;
  x += y_row * static_cast<int64_t>(x_row_stride);
  y += y_row * static_cast<int64_t>(y_row_stride) + y_col;

  loader_x_t loader_x(x, x_row_stride, Xs, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  for (int k = 0; k < k_len; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_unsafe();
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (WITH_RESIDUAL) {
    residual += y_row * static_cast<int64_t>(y_row_stride) + y_col;
    mma_op.store_result(y, y_row_stride);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint idx = simd_gid * 32u + simd_lid; idx < uint(BM * BN);
         idx += uint(WM * WN * SIMD_SIZE)) {
      const int r = int(idx) / BN;
      const int c = int(idx) % BN;
      y[r * static_cast<int64_t>(y_row_stride) + c] = P(
          float(y[r * static_cast<int64_t>(y_row_stride) + c]) +
          float(residual[r * static_cast<int64_t>(y_row_stride) + c]));
    }
  } else if (WITH_BIAS) {

    mma_op.store_result_bias(y, y_row_stride, residual + y_col);
  } else {
    mma_op.store_result(y, y_row_stride);
  }
}

template <typename T, typename LoaderW, int BM, int BK, int BN, bool WITH_BIAS,
          bool WITH_RESIDUAL = false, int WM = qmm_wm<BM>(), int WN = 2>
METAL_FUNC void qmm_t_cast_loaded_impl(
    const device T* x,
    device T* y,
    const device T* bias,
    threadgroup half* Xs,
    threadgroup half* Ws,
    const int K,
    const int N,
    uint3 tid,
    uint simd_gid,
    uint simd_lid,
    thread LoaderW& loader_w) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  using loader_x_t = mlx::steel::
      BlockLoaderCast<T, half, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using mma_t = mlx::steel::
      BlockMMA<half, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;

  const int y_row = int(tid.y) * BM;
  const int y_col = int(tid.x) * BN;

  loader_x_t loader_x(x + size_t(y_row) * size_t(K), K, Xs, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_unsafe();
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device T* yp = y + size_t(y_row) * size_t(N) + y_col;
  if (WITH_RESIDUAL) {

    const device T* rp = bias + size_t(y_row) * size_t(N) + y_col;
    mma_op.store_result(yp, N);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint idx = simd_gid * 32u + simd_lid; idx < uint(BM * BN);
         idx += uint(WM * WN * SIMD_SIZE)) {
      const int r = int(idx) / BN;
      const int c = int(idx) % BN;
      yp[r * static_cast<int64_t>(N) + c] = T(
          float(yp[r * static_cast<int64_t>(N) + c]) +
          float(rp[r * static_cast<int64_t>(N) + c]));
    }
  } else if (WITH_BIAS) {
    mma_op.store_result_bias(yp, N, bias);
  } else {
    mma_op.store_result(yp, N);
  }
}

template <typename T, int group_size, int bits, int BM, int BK, int BN,
          bool WITH_RESIDUAL, bool WITH_BIAS = false,
          int WM = qmm_wm<BM>(), int WN = 2>
METAL_FUNC void qmm_t_aligned_half_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const device T* residual,
    threadgroup half* Xs,
    threadgroup half* Ws,
    const constant int& K,
    const constant int& N,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = BK + 16 / sizeof(half);
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits, half>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_col = int(tid.x) * BN;

  auto wl = (const device uint8_t*)w;
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

  qmm_t_cast_loaded_impl<T, loader_w_t, BM, BK, BN, WITH_BIAS, WITH_RESIDUAL,
                         WM, WN>(
      x, y, residual + (WITH_BIAS ? y_col : 0), Xs, Ws, K, N, tid, simd_gid,
      simd_lid, loader_w);
}

template <typename T, int group_size, int bits, int BM, int BK, int BN,
          bool WITH_RESIDUAL, bool WITH_BIAS = false,
          int WM = qmm_wm<BM>(), int WN = 2>
METAL_FUNC void qmm_t_aligned_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const device T* residual,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_col = int(tid.x) * BN;

  auto wl = (const device uint8_t*)w;
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  qmm_t_loaded_impl<T, T, loader_w_t, BM, BK, BN, WITH_RESIDUAL, WITH_BIAS,
                    WM, WN>(
      x, y, residual, Xs, Ws, K, N, K, tid, simd_gid, simd_lid, loader_w);
}

template <typename P, int group_size, int bits, int BM, int BK, int BN,
          bool WITH_BIAS = false, bool WITH_RESIDUAL = false,
          int WM = qmm_wm<BM>(), int WN = 2>
METAL_FUNC void qmm_t_fp16_precast_impl(
    const device uint32_t* w,
    const device bfloat* scales,
    const device bfloat* biases,
    const device half* x,
    device P* y,
    const device P* bias,
    threadgroup half* Xs,
    threadgroup half* Ws,
    const constant int& K,
    const constant int& k_len,
    const constant int& N,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = BK + 16 / sizeof(half);

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_col = int(tid.x) * BN;

  auto wl = (const device uint8_t*)w;
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  using loader_w_t = QuantizedBlockLoader<
      bfloat, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits, half>;
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  qmm_t_loaded_impl<half, P, loader_w_t, BM, BK, BN, WITH_RESIDUAL, WITH_BIAS,
                    WM, WN>(
      x, y, bias, Xs, Ws, K, N, k_len,
      tid, simd_gid, simd_lid, loader_w);
}

[[kernel]] void cast_qmm_input_bfloat16_to_float16(
    const device bfloat* x [[buffer(3)]],
    device half* y [[buffer(12)]],
    const constant int& count [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
  if (int(gid) < count) y[gid] = half(x[gid]);
}

template <int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_fp16_precast(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device bfloat* biases [[buffer(2)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const device half* x [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];
  qmm_t_fp16_precast_impl<bfloat, group_size, bits, BM, BK, BN>(
      w, scales, biases, x, y, (const device bfloat*)nullptr, Xs, Ws, K, K, N,
      tid, simd_gid, simd_lid);
}

template <int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_bias_fp16_precast(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device bfloat* biases [[buffer(2)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const device bfloat* bias [[buffer(7)]],
    const device half* x [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];
  qmm_t_fp16_precast_impl<bfloat, group_size, bits, BM, BK, BN, true>(
      w, scales, biases, x, y, bias, Xs, Ws, K, K, N, tid, simd_gid, simd_lid);
}

template <int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_residual_fp16_precast(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device bfloat* biases [[buffer(2)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const device bfloat* residual [[buffer(7)]],
    const device half* x [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];
  qmm_t_fp16_precast_impl<bfloat, group_size, bits, BM, BK, BN, false, true>(
      w, scales, biases, x, y, residual, Xs, Ws, K, K, N, tid, simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_aligned_bias(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device T* x          [[buffer(3)]],
    device T* y                [[buffer(4)]],
    const constant int& K      [[buffer(5)]],
    const constant int& N      [[buffer(6)]],

    const device T* bias       [[buffer(7)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];
  qmm_t_aligned_half_impl<T, group_size, bits, BM, BK, BN, false, true>(
      w, scales, biases, x, y, bias, Xs, Ws, K, N, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, int BM, int BK, int BN,
          int WM = qmm_wm<BM>(), int WN = 2>
[[kernel]] void affine_qmm_t_aligned(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device T* x          [[buffer(3)]],
    device T* y                [[buffer(4)]],
    const constant int& K      [[buffer(5)]],
    const constant int& N      [[buffer(6)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];
  qmm_t_aligned_half_impl<T, group_size, bits, BM, BK, BN, false, false, WM, WN>(
      w, scales, biases, x, y, nullptr, Xs, Ws, K, N, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_aligned_residual(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device T* x          [[buffer(3)]],
    device T* y                [[buffer(4)]],
    const constant int& K      [[buffer(5)]],
    const constant int& N      [[buffer(6)]],
    const device T* residual   [[buffer(7)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];
  qmm_t_aligned_half_impl<T, group_size, bits, BM, BK, BN, true>(
      w, scales, biases, x, y, residual, Xs, Ws, K, N, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_routed(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device T* x          [[buffer(3)]],
    device T* y                [[buffer(4)]],
    const constant int& K      [[buffer(5)]],
    const constant int& N      [[buffer(6)]],

    const device int* tile_expert [[buffer(12)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  const int e = tile_expert[tid.y];
  if (e < 0) return;

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  const size_t w_bytes = size_t(e) * size_t(N) * size_t(K) *
                         size_t(bytes_per_pack) / size_t(pack_factor);
  const size_t g_off = size_t(e) * size_t(N) * size_t(K / group_size);

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  qmm_t_aligned_impl<T, group_size, bits, BM, BK, BN, false>(
      (const device uint32_t*)((const device uint8_t*)w + w_bytes),
      scales + g_off, biases + g_off, x, y, nullptr, Xs, Ws, K, N, tid,
      simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_routed_fp16(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device T* x          [[buffer(3)]],
    device T* y                [[buffer(4)]],
    const constant int& K      [[buffer(5)]],
    const constant int& N      [[buffer(6)]],
    const device int* tile_expert [[buffer(12)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  const int e = tile_expert[tid.y];
  if (e < 0) return;

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = BK + 16 / sizeof(half);
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, qmm_tgp<BM>(), group_size, bits, half>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_col = int(tid.x) * BN;
  const size_t w_bytes = size_t(e) * size_t(N) * size_t(K) *
                         size_t(bytes_per_pack) / size_t(pack_factor);
  const size_t g_off = size_t(e) * size_t(N) * size_t(K_g);

  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];
  loader_w_t loader_w(
      (const device uint8_t*)w + w_bytes + size_t(y_col) * size_t(K_w),
      scales + g_off + size_t(y_col) * size_t(K_g),
      biases + g_off + size_t(y_col) * size_t(K_g), K, Ws, simd_gid, simd_lid);
  qmm_t_cast_loaded_impl<T, loader_w_t, BM, BK, BN, false>(
      x, y, (const device T*)nullptr, Xs, Ws, K, N, tid, simd_gid, simd_lid,
      loader_w);
}

// ── the routed MXFP4 tile loader, in vectors ────────────────────────────────
//
// The same values in the same places as `mlx::steel::Mxfp4BlockLoader`, which
// is vendored and stays as it is: a thread still owns `n_reads` consecutive
// bytes of one weight row, still reads the one E8M0 exponent those bytes
// share, and still multiplies each nibble by it before staging. Nothing about
// the k order or the rounding moves, so this lands the SAME BITS as the
// loader it replaces and the quant tile family's fingerprint is untouched.
//
// **WHAT MOVES IS THE SHAPE OF THE TWO MEMORY ACCESSES AROUND THAT
// MULTIPLY.** The vendored loader spells its read `src[i]` over `n_reads` and
// its write `dst[2 * i]`, `dst[2 * i + 1]` — eight one-byte device loads and
// sixteen two-byte threadgroup stores per thread per k step at the 64-column
// tile. Both runs are contiguous and both are aligned; NEITHER FACT IS
// AVAILABLE TO THE COMPILER, because `src` is a `uint8_t*` whose alignment is
// one by declaration, so the eight loads cannot be merged and are not.
//
// The alignment is the CALLER's property and is stated here rather than
// assumed. `src` reaches a thread as `e * N * K / 2 + y_col * K / 2` plus
// `bi * K / 2 + bj`, and the k loop steps it by `BCOLS_PACKED`, which is 16.
// `qmm_grid` refuses to launch unless `K % BK == 0` at `BK = 32`, so every
// row offset is a multiple of 16 bytes; `bj` is a multiple of `n_reads` by
// construction. So each address is `n_reads`-aligned and the read is one
// vector. `dst` lands at `bi * BK_padded + 2 * bj` with `BK_padded = BK + 8`,
// a multiple of four halves both ways, so the store is one `vec<D, 4>`.
//
// Measured on an M1 Max over gpt-oss-20b's two expert shapes at 2048 sorted
// rows, the shipped 32x64 tile: 12.29 → 12.12 ms on gate/up and 6.19 → 6.09
// on down, +1.4% and +1.6%. It is a small number because the loader was never
// what bound this kernel — see the note under the stamps — and it is kept
// because it is free and because eight byte loads of an eight-byte aligned
// run is a defect whatever it costs.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    typename D = T>
struct Mxfp4VecLoader {
  static_assert(BCOLS == 32, "MXFP4 blocks are exactly 32 values");
  MLX_MTL_CONST short pack_factor = 2;
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  // The widest byte vector that divides a thread's run. Over the stamped
  // column tiles at this family's 128 lanes `n_reads` is 8, 4 or 2, so this
  // is 4 or 2 and never 1.
  MLX_MTL_CONST short VEC = (n_reads % 4 == 0) ? 4 : n_reads;
  static_assert(VEC == 2 || VEC == 4, "a thread reads two or four bytes");

  const int src_ld;
  const int tile_stride;
  const int group_stride;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup D* dst;
  const device uint8_t* src;
  const device uint8_t* exponents;

  Mxfp4VecLoader(
      const device uint8_t* src_,
      const device uint8_t* exponents_,
      const int src_ld_,
      threadgroup D* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS_PACKED : BROWS * src_ld / pack_factor),
        group_stride(BROWS * src_ld / 32),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld / pack_factor + bj),
        exponents(exponents_ + bi * src_ld / 32) {}

  void load_unsafe() const {
    const D scale = D(mxfp4_block_scale(*exponents));
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_reads; i += VEC) {
      const vec<uint8_t, VEC> run =
          *((const device vec<uint8_t, VEC>*)(src + i));
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < VEC; j += 2) {
        const uint8_t lo = run[j];
        const uint8_t hi = run[j + 1];
        *((threadgroup vec<D, 4>*)(dst + 2 * (i + j))) = vec<D, 4>(
            scale * D(mxfp4_lo(lo)), scale * D(mxfp4_hi(lo)),
            scale * D(mxfp4_lo(hi)), scale * D(mxfp4_hi(hi)));
      }
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1)
      ++exponents;
    else
      exponents += group_stride;
  }
};

// ── the routed MXFP4 GEMM, biased and not ──────────────────────────────────
//
// **BOTH FORMS ARE STAMPED, AND THE SECOND ONE IS WHY THIS IS A `_impl`.**
// gpt-oss is the family that ships mxfp4, and it biases its gate/up
// projection and does NOT bias its down projection: `down_bias` is folded
// after the weighted reduce, where a tensor-parallel split can still add it
// once, so the down GEMM itself reaches this file with no bias plane. When
// only the biased form existed, `batched_point` answered `None` for that
// half of every mixture layer and the driver fell through to the MATVEC arm
// -- one simdgroup per (row, four columns), reading each expert's down slice
// once per ROUTED ROW instead of once per tile. Measured on an M1 Max at 512
// prompt tokens that was 9.0 GB of weight traffic a layer against the tiled
// arm's 0.53. Stamping this form measures 1547.0 -> 784.9 ms, 331.0 -> 652.4
// tok/s. See `.wiki/macos-bench.md` section 18.
template <typename T, int BM, int BK, int BN, bool WITH_BIAS>
METAL_FUNC void mxfp4_qmm_t_routed_impl(
    const device uint32_t* w,
    const device uint8_t* exponents,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const device T* bias,
    const device int* tile_expert,
    threadgroup half* Xs,
    threadgroup half* Ws,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  const int e = tile_expert[tid.y];
  if (e < 0) return;

  constexpr int BK_padded = BK + 16 / sizeof(half);
  constexpr int tgp_size = qmm_tgp<BM>();
  const int y_col = int(tid.x) * BN;
  const size_t expert_w = size_t(e) * size_t(N) * size_t(K) / 2;
  const size_t expert_s = size_t(e) * size_t(N) * size_t(K / 32);
  const device uint8_t* wb = (const device uint8_t*)w + expert_w + y_col * K / 2;
  const device uint8_t* sb = exponents + expert_s + y_col * (K / 32);

  using loader_w_t = Mxfp4VecLoader<T, BN, BK, BK_padded, 1, tgp_size, half>;
  loader_w_t loader_w(wb, sb, K, Ws, simd_gid, simd_lid);
  qmm_t_cast_loaded_impl<T, loader_w_t, BM, BK, BN, WITH_BIAS>(
      x, y,
      WITH_BIAS ? bias + size_t(e) * size_t(N) + y_col : (const device T*)nullptr,
      Xs, Ws, K, N, tid, simd_gid, simd_lid, loader_w);
}

template <typename T, int BM, int BK, int BN>
[[kernel]] void mxfp4_qmm_t_routed_bias(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* exponents [[buffer(1)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const device T* bias [[buffer(7)]],
    const device int* tile_expert [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];
  mxfp4_qmm_t_routed_impl<T, BM, BK, BN, true>(
      w, exponents, x, y, K, N, bias, tile_expert, Xs, Ws, tid, simd_gid,
      simd_lid);
}

// The unbiased form does not DECLARE buffer 7. The driver binds a nil buffer
// at every seat the family leaves unbound, and a seat that is declared and
// never read is the trap this file already warns about one point down; not
// naming it is how the trap is avoided rather than tolerated.
template <typename T, int BM, int BK, int BN>
[[kernel]] void mxfp4_qmm_t_routed(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* exponents [[buffer(1)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const device int* tile_expert [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];
  mxfp4_qmm_t_routed_impl<T, BM, BK, BN, false>(
      w, exponents, x, y, K, N, (const device T*)nullptr, tile_expert, Xs, Ws,
      tid, simd_gid, simd_lid);
}

#define PIE_STAMP_qmm_t_routed(entry, gs, b, bm, bk, bn)                       \
  template [[host_name(entry)]]                                                \
  [[kernel]] void affine_qmm_t_routed<bfloat, gs, b, bm, bk, bn>(              \
      const device uint32_t*, const device bfloat*, const device bfloat*,      \
      const device bfloat*, device bfloat*, const constant int&,               \
      const constant int&, const device int*, uint3, uint, uint);

#define PIE_STAMP_qmm_t(entry, gs, b, bm, bk, bn)                              \
  template [[host_name(entry)]]                                                \
  [[kernel]] void affine_qmm_t_aligned<bfloat, gs, b, bm, bk, bn>(             \
      const device uint32_t*, const device bfloat*, const device bfloat*,      \
      const device bfloat*, device bfloat*, const constant int&,               \
      const constant int&, uint3, uint, uint);

#define PIE_STAMP_qmm_t_residual(entry, gs, b, bm, bk, bn)                     \
  template [[host_name(entry)]]                                                \
  [[kernel]] void affine_qmm_t_aligned_residual<bfloat, gs, b, bm, bk, bn>(    \
      const device uint32_t*, const device bfloat*, const device bfloat*,      \
      const device bfloat*, device bfloat*, const constant int&,               \
      const constant int&, const device bfloat*, uint3, uint, uint);

#define PIE_STAMP_qmm_t_bias(entry, gs, b, bm, bk, bn)                         \
  template [[host_name(entry)]]                                                \
  [[kernel]] void affine_qmm_t_aligned_bias<bfloat, gs, b, bm, bk, bn>(        \
      const device uint32_t*, const device bfloat*, const device bfloat*,      \
      const device bfloat*, device bfloat*, const constant int&,               \
      const constant int&, const device bfloat*, uint3, uint, uint);

#define instantiate_mxfp4_qmm_t_routed(bm, bn)                              \
  template [[host_name("mxfp4_qmm_t_routed_bias_bfloat16_bm_" #bm           \
                       "_bn_" #bn)]]                                         \
  [[kernel]] void mxfp4_qmm_t_routed_bias<bfloat, bm, 32, bn>(              \
      const device uint32_t*, const device uint8_t*, const device bfloat*,   \
      device bfloat*, const constant int&, const constant int&,              \
      const device bfloat*, const device int*, uint3, uint, uint);

instantiate_mxfp4_qmm_t_routed(16, 16)
instantiate_mxfp4_qmm_t_routed(16, 32)
instantiate_mxfp4_qmm_t_routed(16, 64)
instantiate_mxfp4_qmm_t_routed(32, 16)
instantiate_mxfp4_qmm_t_routed(32, 32)
instantiate_mxfp4_qmm_t_routed(32, 64)
instantiate_mxfp4_qmm_t_routed(64, 16)
instantiate_mxfp4_qmm_t_routed(64, 32)
instantiate_mxfp4_qmm_t_routed(64, 64)

#define instantiate_mxfp4_qmm_t_routed_plain(bm, bn)                        \
  template [[host_name("mxfp4_qmm_t_routed_bfloat16_bm_" #bm                \
                       "_bn_" #bn)]]                                        \
  [[kernel]] void mxfp4_qmm_t_routed<bfloat, bm, 32, bn>(                   \
      const device uint32_t*, const device uint8_t*, const device bfloat*,   \
      device bfloat*, const constant int&, const constant int&,              \
      const device int*, uint3, uint, uint);

instantiate_mxfp4_qmm_t_routed_plain(16, 16)
instantiate_mxfp4_qmm_t_routed_plain(16, 32)
instantiate_mxfp4_qmm_t_routed_plain(16, 64)
instantiate_mxfp4_qmm_t_routed_plain(32, 16)
instantiate_mxfp4_qmm_t_routed_plain(32, 32)
instantiate_mxfp4_qmm_t_routed_plain(32, 64)
instantiate_mxfp4_qmm_t_routed_plain(64, 16)
instantiate_mxfp4_qmm_t_routed_plain(64, 32)
instantiate_mxfp4_qmm_t_routed_plain(64, 64)

#define instantiate_qmm_t_routed_fp16(bm, bn)                                 \
  template [[host_name("affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_"      \
                       #bm "_bn_" #bn)]]                                      \
  [[kernel]] void affine_qmm_t_routed_fp16<bfloat, 64, 4, bm, 32, bn>(        \
      const device uint32_t*, const device bfloat*, const device bfloat*,     \
      const device bfloat*, device bfloat*, const constant int&,              \
      const constant int&, const device int*, uint3, uint, uint);

instantiate_qmm_t_routed_fp16(16, 16)
instantiate_qmm_t_routed_fp16(16, 32)
instantiate_qmm_t_routed_fp16(16, 64)
instantiate_qmm_t_routed_fp16(32, 16)
instantiate_qmm_t_routed_fp16(32, 32)
instantiate_qmm_t_routed_fp16(32, 64)
instantiate_qmm_t_routed_fp16(64, 16)
instantiate_qmm_t_routed_fp16(64, 32)
instantiate_qmm_t_routed_fp16(64, 64)

#define instantiate_qmm_t_fp16_precast(bm, bn)                              \
  template [[host_name("affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_"   \
                       #bm "_bn_" #bn)]]                                     \
  [[kernel]] void affine_qmm_t_fp16_precast<64, 4, bm, 64, bn>(             \
      const device uint32_t*, const device bfloat*, const device bfloat*,    \
      device bfloat*, const constant int&, const constant int&,              \
      const device half*, uint3, uint, uint);

instantiate_qmm_t_fp16_precast(16, 16)
instantiate_qmm_t_fp16_precast(16, 32)
instantiate_qmm_t_fp16_precast(16, 64)
instantiate_qmm_t_fp16_precast(32, 16)
instantiate_qmm_t_fp16_precast(32, 32)
instantiate_qmm_t_fp16_precast(32, 64)
instantiate_qmm_t_fp16_precast(64, 16)
instantiate_qmm_t_fp16_precast(64, 32)
instantiate_qmm_t_fp16_precast(64, 64)

// **THE 8 RUNG**, `WM = 1` by `qmm_wm` and 64 lanes to the threadgroup.
// Three of the file's families are stamped at it and the rest are not:
//
//   * PRE-CAST, here and in its `_bias`/`_residual` forms below, is the one
//     that matters. On a machine that emulates bfloat every dense projection
//     of a 4-bit/g64 checkpoint takes this family (`quant::act_x_wt` rung 2),
//     so a two-to-seven-lane decode is arithmetic that only this rung can cut.
//   * SPLIT-K follows it, because rung 3 is reachable at the same row block
//     whenever the pre-cast arm declines — and a `bm_rung` of 8 with no split
//     point stamped at 8 is a refusal at fire time rather than a slower path.
//   * PLAIN needs no line anywhere: it is minted by `PIE_STAMP_qmm_t`, and
//     `affine_qmm_t_aligned` reads its `WM` from `qmm_wm<BM>()` like
//     everything else, so the existing stamp mints the 8 rung unchanged and
//     the macro's arity did not move.
//
// STRIDED, ROUTED and MXFP4 are deliberately left out. No Rust caller composes
// a strided name at all, and the routed pair's row block comes from
// `linear::moe`'s own `MOE_TILE_ROWS = [16, 32, 64]` — an expert's run is
// `rows x top_k / experts`, which reaches a block this narrow only in a fleet
// too small to take the batched arm in the first place. Stamping them would
// compile entrypoints for a fire that cannot arrive.
instantiate_qmm_t_fp16_precast(8, 16)
instantiate_qmm_t_fp16_precast(8, 32)
instantiate_qmm_t_fp16_precast(8, 64)

#define instantiate_qmm_t_bias_fp16_precast(bm, bn)                          \
  template [[host_name("affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4"   \
                       "_bm_" #bm "_bn_" #bn)]]                              \
  [[kernel]] void affine_qmm_t_bias_fp16_precast<64, 4, bm, 64, bn>(         \
      const device uint32_t*, const device bfloat*, const device bfloat*,    \
      device bfloat*, const constant int&, const constant int&,              \
      const device bfloat*, const device half*, uint3, uint, uint);

#define instantiate_qmm_t_residual_fp16_precast(bm, bn)                      \
  template [[host_name("affine_qmm_t_residual_fp16_precast_bfloat16"         \
                       "_gs_64_b_4_bm_" #bm "_bn_" #bn)]]                    \
  [[kernel]] void affine_qmm_t_residual_fp16_precast<64, 4, bm, 64, bn>(     \
      const device uint32_t*, const device bfloat*, const device bfloat*,    \
      device bfloat*, const constant int&, const constant int&,              \
      const device bfloat*, const device half*, uint3, uint, uint);

instantiate_qmm_t_residual_fp16_precast(16, 16)
instantiate_qmm_t_residual_fp16_precast(16, 32)
instantiate_qmm_t_residual_fp16_precast(16, 64)
instantiate_qmm_t_residual_fp16_precast(32, 16)
instantiate_qmm_t_residual_fp16_precast(32, 32)
instantiate_qmm_t_residual_fp16_precast(32, 64)
instantiate_qmm_t_residual_fp16_precast(64, 16)
instantiate_qmm_t_residual_fp16_precast(64, 32)
instantiate_qmm_t_residual_fp16_precast(64, 64)
instantiate_qmm_t_residual_fp16_precast(8, 16)
instantiate_qmm_t_residual_fp16_precast(8, 32)
instantiate_qmm_t_residual_fp16_precast(8, 64)

instantiate_qmm_t_bias_fp16_precast(16, 16)
instantiate_qmm_t_bias_fp16_precast(16, 32)
instantiate_qmm_t_bias_fp16_precast(16, 64)
instantiate_qmm_t_bias_fp16_precast(32, 16)
instantiate_qmm_t_bias_fp16_precast(32, 32)
instantiate_qmm_t_bias_fp16_precast(32, 64)
instantiate_qmm_t_bias_fp16_precast(64, 16)
instantiate_qmm_t_bias_fp16_precast(64, 32)
instantiate_qmm_t_bias_fp16_precast(64, 64)
instantiate_qmm_t_bias_fp16_precast(8, 16)
instantiate_qmm_t_bias_fp16_precast(8, 32)
instantiate_qmm_t_bias_fp16_precast(8, 64)

template <typename T, int group_size, int bits, int BM, int BK, int BN,
          bool WITH_RESIDUAL>
METAL_FUNC void qmm_t_strided_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const device T* residual,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& row_stride,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, qmm_tgp<BM>(), group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_col = int(tid.x) * BN;

  auto wl = (const device uint8_t*)w;
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  qmm_t_loaded_impl<T, T, loader_w_t, BM, BK, BN, WITH_RESIDUAL, false>(
      x, y, residual, Xs, Ws, row_stride, row_stride, K,
      tid, simd_gid, simd_lid, loader_w);
  (void)N;
}

template <typename T, int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_strided(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device T* x          [[buffer(3)]],
    device T* y                [[buffer(4)]],
    const constant int& K      [[buffer(5)]],
    const constant int& N      [[buffer(6)]],
    const constant int& row_stride [[buffer(8)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  qmm_t_strided_impl<T, group_size, bits, BM, BK, BN, false>(
      w, scales, biases, x, y, nullptr, Xs, Ws, K, N, row_stride,
      tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_strided_residual(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device T* x          [[buffer(3)]],
    device T* y                [[buffer(4)]],
    const constant int& K      [[buffer(5)]],
    const constant int& N      [[buffer(6)]],
    const device T* residual   [[buffer(7)]],
    const constant int& row_stride [[buffer(8)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  qmm_t_strided_impl<T, group_size, bits, BM, BK, BN, true>(
      w, scales, biases, x, y, residual, Xs, Ws, K, N, row_stride,
      tid, simd_gid, simd_lid);
}

[[kernel]] void cast_qmm_input_strided_bfloat16_to_float16(
    const device bfloat* x [[buffer(3)]],
    const constant int& K [[buffer(5)]],
    const constant int& row_stride [[buffer(8)]],
    device half* y [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]) {
  y[size_t(gid.y) * row_stride + gid.x] =
      half(x[size_t(gid.y) * row_stride + gid.x]);
}

template <int group_size, int bits, int BM, int BK, int BN, bool WITH_RESIDUAL>
METAL_FUNC void qmm_t_strided_fp16_precast_impl(
    const device uint32_t* w,
    const device bfloat* scales,
    const device bfloat* biases,
    const device half* x,
    device bfloat* y,
    const device bfloat* residual,
    threadgroup half* Xs,
    threadgroup half* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& row_stride,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  using loader_w_t = QuantizedBlockLoader<
      bfloat, BN, BK, BK_padded, 1, qmm_tgp<BM>(), group_size, bits, half>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_col = int(tid.x) * BN;

  auto wl = (const device uint8_t*)w;
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  qmm_t_loaded_impl<half, bfloat, loader_w_t, BM, BK, BN,
                    WITH_RESIDUAL, false>(
      x, y, residual, Xs, Ws, row_stride, row_stride, K,
      tid, simd_gid, simd_lid, loader_w);
  (void)N;
}

template <int group_size, int bits, int BM, int BK, int BN, bool WITH_RESIDUAL>
[[kernel]] void affine_qmm_t_strided_fp16_precast(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device bfloat* biases [[buffer(2)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const device bfloat* residual [[buffer(7)]],
    const constant int& row_stride [[buffer(8)]],
    const device half* x [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];
  qmm_t_strided_fp16_precast_impl<group_size, bits, BM, BK, BN, WITH_RESIDUAL>(
      w, scales, biases, x, y, residual, Xs, Ws, K, N, row_stride,
      tid, simd_gid, simd_lid);
}

#define instantiate_qmm_t_strided(gs, bm, bk, bn, b)                                     \
  template [[host_name("affine_qmm_t_strided_bfloat16_gs_" #gs "_b_" #b "_bm_" #bm "_bn_" #bn)]] \
  [[kernel]] void affine_qmm_t_strided<bfloat, gs, b, bm, bk, bn>(                \
      const device uint32_t*, const device bfloat*, const device bfloat*,         \
      const device bfloat*, device bfloat*, const constant int&,                  \
      const constant int&, const constant int&, uint3, uint, uint);               \
  template [[host_name("affine_qmm_t_strided_residual_bfloat16_gs_" #gs "_b_" #b "_bm_" #bm "_bn_" #bn)]] \
  [[kernel]] void affine_qmm_t_strided_residual<bfloat, gs, b, bm, bk, bn>(       \
      const device uint32_t*, const device bfloat*, const device bfloat*,         \
      const device bfloat*, device bfloat*, const constant int&,                  \
      const constant int&, const device bfloat*, const constant int&,             \
      uint3, uint, uint);

instantiate_qmm_t_strided(64, 16, 32, 32, 4)
instantiate_qmm_t_strided(32, 16, 32, 32, 4)
instantiate_qmm_t_strided(128, 16, 32, 32, 4)
instantiate_qmm_t_strided(64, 16, 32, 32, 8)
instantiate_qmm_t_strided(32, 16, 32, 32, 8)
instantiate_qmm_t_strided(128, 16, 32, 32, 8)
instantiate_qmm_t_strided(64, 32, 32, 32, 4)
instantiate_qmm_t_strided(32, 32, 32, 32, 4)
instantiate_qmm_t_strided(128, 32, 32, 32, 4)
instantiate_qmm_t_strided(64, 32, 32, 32, 8)
instantiate_qmm_t_strided(32, 32, 32, 32, 8)
instantiate_qmm_t_strided(128, 32, 32, 32, 8)
instantiate_qmm_t_strided(64, 64, 32, 32, 4)
instantiate_qmm_t_strided(32, 64, 32, 32, 4)
instantiate_qmm_t_strided(128, 64, 32, 32, 4)
instantiate_qmm_t_strided(64, 64, 32, 32, 8)
instantiate_qmm_t_strided(32, 64, 32, 32, 8)
instantiate_qmm_t_strided(128, 64, 32, 32, 8)

#define instantiate_qmm_t_strided_fp16_precast(bm, residual, name)           \
  template [[host_name("affine_qmm_t_strided_fp16_precast" name              \
                       "_bfloat16_gs_64_b_4_bm_" #bm "_bn_32")]]              \
  [[kernel]] void affine_qmm_t_strided_fp16_precast<64, 4, bm, 32, 32, residual>(\
      const device uint32_t*, const device bfloat*, const device bfloat*,    \
      device bfloat*, const constant int&, const constant int&,              \
      const device bfloat*, const constant int&, const device half*,         \
      uint3, uint, uint);

instantiate_qmm_t_strided_fp16_precast(16, false, "")
instantiate_qmm_t_strided_fp16_precast(32, false, "")
instantiate_qmm_t_strided_fp16_precast(16, true, "_residual")
instantiate_qmm_t_strided_fp16_precast(32, true, "_residual")
instantiate_qmm_t_strided_fp16_precast(64, false, "")
instantiate_qmm_t_strided_fp16_precast(64, true, "_residual")

template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>
[[kernel]] void affine_qmv_wide_strided(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& row_stride [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = SIMD_SIZE / k_lanes;
  constexpr int sub = 8;

  const short k_lane = simd_lid % k_lanes;
  const short sg_row = simd_lid / k_lanes;
  const int out_row = int(tid.y) * (results_per_simdgroup * num_simdgroups) +
      results_per_simdgroup * int(simd_gid) + int(sg_row);
  const int vec0 = int(tid.x) * vecs_per_tg;
  const int row = min(out_row, N - 1);

  const int row_w = K * bits / 8;
  const int row_g = K / group_size;
  const device uint8_t* wrow = (const device uint8_t*)w + row * row_w;
  const device T* srow = scales + row * row_g;
  const device T* brow = biases + row * row_g;

  const device T* xv[vecs_per_tg];
  for (int v = 0; v < vecs_per_tg; ++v)
    xv[v] = x + min(vec0 + v, M - 1) * row_stride;

  float result[vecs_per_tg] = {0};
  for (int g = k_lane; g < row_g; g += k_lanes) {
    const float scale = float(srow[g]);
    const float bias = float(brow[g]);
    for (int sc = 0; sc < group_size / sub; ++sc) {
      const int k0 = g * group_size + sc * sub;
      const device uint8_t* wc = wrow + k0 * bits / 8;
      float wd[sub];
      dequantize<float, sub, bits>(wc, scale, bias, wd);
      for (int v = 0; v < vecs_per_tg; ++v) {
        float acc = 0;
        for (int i = 0; i < sub; ++i) acc += float(xv[v][k0 + i]) * wd[i];
        result[v] += acc;
      }
    }
  }

  for (int v = 0; v < vecs_per_tg; ++v) {
    if constexpr (k_lanes >= 16) result[v] += simd_shuffle_down(result[v], 8);
    if constexpr (k_lanes >= 8) result[v] += simd_shuffle_down(result[v], 4);
    if constexpr (k_lanes >= 4) result[v] += simd_shuffle_down(result[v], 2);
    if constexpr (k_lanes >= 2) result[v] += simd_shuffle_down(result[v], 1);
  }
  if (k_lane == 0 && out_row < N) {
    for (int v = 0; v < vecs_per_tg; ++v)
      if (vec0 + v < M)
        y[(vec0 + v) * row_stride + out_row] = T(result[v]);
  }
}

#define instantiate_qmv_wide_strided(b, v, kl)                               \
  template [[host_name("affine_qmv_wide_strided_bfloat16_gs_64_b_" #b "_v_" #v \
                       "_kl_" #kl)]]                                         \
  [[kernel]] void affine_qmv_wide_strided<bfloat, 64, b, v, kl>(            \
      const device uint32_t*, const device bfloat*, const device bfloat*,    \
      const device bfloat*, device bfloat*, const constant int&,             \
      const constant int&, const constant int&, const constant int&,         \
      uint3, uint, uint);

instantiate_qmv_wide_strided(4, 4, 8)

instantiate_qmv_wide_strided(8, 4, 8)

template <typename T, typename P, int group_size, int bits, int BM, int BK, int BN>
METAL_FUNC void qmm_t_splitk_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device P* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& k_len,
    const constant int& N,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, qmm_tgp<BM>(), group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_col = int(tid.x) * BN;

  auto wl = (const device uint8_t*)w;
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  qmm_t_loaded_impl<T, P, loader_w_t, BM, BK, BN, false, false>(
      x, y, nullptr, Xs, Ws, K, N, k_len,
      tid, simd_gid, simd_lid, loader_w);
}

template <typename T, typename P, int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_splitk(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales   [[buffer(1)]],
    const device T* biases   [[buffer(2)]],
    const device T* x        [[buffer(3)]],
    device P* y              [[buffer(8)]],
    const constant int& K    [[buffer(5)]],
    const constant int& N    [[buffer(6)]],
    const constant int& k_partition_size [[buffer(9)]],
    const constant int& split_k_partition_stride [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int k_start = int(tid.z) * k_partition_size;
  x += k_start;
  auto wl = (const device uint8_t*)w;
  wl += k_start * bytes_per_pack / pack_factor;
  scales += k_start / group_size;
  biases += k_start / group_size;
  y += int64_t(tid.z) * split_k_partition_stride;

  qmm_t_splitk_impl<T, P, group_size, bits, BM, BK, BN>(
      (const device uint32_t*)wl, scales, biases, x, y, Xs, Ws, K,
      k_partition_size, N, tid, simd_gid, simd_lid);
}

template <typename P, int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_splitk_fp16_precast(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device bfloat* biases [[buffer(2)]],
    device P* y [[buffer(8)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& k_partition_size [[buffer(9)]],
    const constant int& split_k_partition_stride [[buffer(10)]],
    const device half* x [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];

  const int k_start = int(tid.z) * k_partition_size;
  x += k_start;
  auto wl = (const device uint8_t*)w;
  wl += k_start * bytes_per_pack / pack_factor;
  scales += k_start / group_size;
  biases += k_start / group_size;
  y += int64_t(tid.z) * split_k_partition_stride;

  qmm_t_fp16_precast_impl<P, group_size, bits, BM, BK, BN>(
      (const device uint32_t*)wl, scales, biases, x, y, (const device P*)nullptr,
      Xs, Ws, K, k_partition_size, N, tid, simd_gid, simd_lid);
}

template <typename T, typename P>
[[kernel]] void qmm_splitk_reduce(
    device T* y                 [[buffer(4)]],
    const constant int& N       [[buffer(6)]],
    const device P* partial     [[buffer(8)]],
    const constant int& stride  [[buffer(10)]],

    const constant int& split_k [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int col = int(gid.x);
  if (col >= N) return;
  const int64_t o = int64_t(gid.y) * N + col;
  float acc = 0.0f;
  for (int s = 0; s < split_k; ++s)
    acc += float(partial[o + int64_t(s) * stride]);
  y[o] = static_cast<T>(acc);
}

#define instantiate_qmm_t_splitk_named(name, ptype, gs, bm, bk, bn, b)          \
  template [[host_name("affine_qmm_t_splitk_" #name "_gs_" #gs "_b_" #b        \
                       "_bm_" #bm "_bn_" #bn)]]                                  \
  [[kernel]] void affine_qmm_t_splitk<bfloat, ptype, gs, b, bm, bk, bn>(        \
      const device uint32_t*, const device bfloat*, const device bfloat*,       \
      const device bfloat*, device ptype*, const constant int&,                 \
      const constant int&, const constant int&, const constant int&,            \
      uint3, uint, uint);

#define instantiate_qmm_t_splitk(gs, bm, bk, bn, b)                              \
  instantiate_qmm_t_splitk_named(bfloat16, bfloat, gs, bm, bk, bn, b)           \
  instantiate_qmm_t_splitk_named(f32_bfloat16, float, gs, bm, bk, bn, b)

instantiate_qmm_t_splitk(64, 16, 32, 32, 4)
instantiate_qmm_t_splitk(32, 16, 32, 32, 4)
instantiate_qmm_t_splitk(128, 16, 32, 32, 4)
instantiate_qmm_t_splitk(64, 16, 32, 32, 8)
instantiate_qmm_t_splitk(32, 16, 32, 32, 8)
instantiate_qmm_t_splitk(128, 16, 32, 32, 8)
instantiate_qmm_t_splitk(64, 32, 32, 32, 4)
instantiate_qmm_t_splitk(32, 32, 32, 32, 4)
instantiate_qmm_t_splitk(128, 32, 32, 32, 4)
instantiate_qmm_t_splitk(64, 32, 32, 32, 8)
instantiate_qmm_t_splitk(32, 32, 32, 32, 8)
instantiate_qmm_t_splitk(128, 32, 32, 32, 8)
instantiate_qmm_t_splitk(64, 64, 32, 32, 4)
instantiate_qmm_t_splitk(32, 64, 32, 32, 4)
instantiate_qmm_t_splitk(128, 64, 32, 32, 4)
instantiate_qmm_t_splitk(64, 64, 32, 32, 8)
instantiate_qmm_t_splitk(32, 64, 32, 32, 8)
instantiate_qmm_t_splitk(128, 64, 32, 32, 8)
instantiate_qmm_t_splitk(64, 8, 32, 32, 4)
instantiate_qmm_t_splitk(32, 8, 32, 32, 4)
instantiate_qmm_t_splitk(128, 8, 32, 32, 4)
instantiate_qmm_t_splitk(64, 8, 32, 32, 8)
instantiate_qmm_t_splitk(32, 8, 32, 32, 8)
instantiate_qmm_t_splitk(128, 8, 32, 32, 8)

#define instantiate_qmm_t_splitk_fp16_precast(name, ptype, bm)              \
  template [[host_name("affine_qmm_t_splitk_fp16_precast_" #name            \
                       "_gs_64_b_4_bm_" #bm "_bn_32")]]                      \
  [[kernel]] void affine_qmm_t_splitk_fp16_precast<ptype, 64, 4, bm, 32, 32>(\
      const device uint32_t*, const device bfloat*, const device bfloat*,    \
      device ptype*, const constant int&, const constant int&,               \
      const constant int&, const constant int&, const device half*,          \
      uint3, uint, uint);

instantiate_qmm_t_splitk_fp16_precast(bfloat16, bfloat, 16)
instantiate_qmm_t_splitk_fp16_precast(bfloat16, bfloat, 32)
instantiate_qmm_t_splitk_fp16_precast(bfloat16, bfloat, 64)
instantiate_qmm_t_splitk_fp16_precast(f32_bfloat16, float, 16)
instantiate_qmm_t_splitk_fp16_precast(f32_bfloat16, float, 32)
instantiate_qmm_t_splitk_fp16_precast(f32_bfloat16, float, 64)
instantiate_qmm_t_splitk_fp16_precast(bfloat16, bfloat, 8)
instantiate_qmm_t_splitk_fp16_precast(f32_bfloat16, float, 8)

template [[host_name("qmm_splitk_reduce_bfloat16")]] [[kernel]] void
qmm_splitk_reduce<bfloat, bfloat>(
    device bfloat*, const constant int&, const device bfloat*,
    const constant int&, const constant int&, uint2);
template [[host_name("qmm_splitk_reduce_f32_bfloat16")]] [[kernel]] void
qmm_splitk_reduce<bfloat, float>(
    device bfloat*, const constant int&, const device float*,
    const constant int&, const constant int&, uint2);

template [[host_name("affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4")]]
[[kernel]] void affine_qmm_t_aligned<bfloat, 64, 4, 64, 32, 64, 2, 4>(
    const device uint32_t*, const device bfloat*, const device bfloat*,
    const device bfloat*, device bfloat*, const constant int&,
    const constant int&, uint3, uint, uint);

template [[host_name("affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4")]]
[[kernel]] void affine_qmm_t_aligned<bfloat, 64, 4, 128, 32, 32, 4, 2>(
    const device uint32_t*, const device bfloat*, const device bfloat*,
    const device bfloat*, device bfloat*, const constant int&,
    const constant int&, uint3, uint, uint);

template [[host_name("affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1")]]
[[kernel]] void affine_qmm_t_aligned<bfloat, 64, 4, 64, 32, 32, 2, 1>(
    const device uint32_t*, const device bfloat*, const device bfloat*,
    const device bfloat*, device bfloat*, const constant int&,
    const constant int&, uint3, uint, uint);

template [[host_name("affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2")]]
[[kernel]] void affine_qmm_t_aligned<bfloat, 64, 4, 64, 32, 32, 1, 2>(
    const device uint32_t*, const device bfloat*, const device bfloat*,
    const device bfloat*, device bfloat*, const constant int&,
    const constant int&, uint3, uint, uint);

template [[host_name("affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2")]]
[[kernel]] void affine_qmm_t_aligned<bfloat, 64, 4, 32, 32, 32, 1, 2>(
    const device uint32_t*, const device bfloat*, const device bfloat*,
    const device bfloat*, device bfloat*, const constant int&,
    const constant int&, uint3, uint, uint);
