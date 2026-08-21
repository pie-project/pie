// Raw-Metal port of MLX affine_qmm_t — the quantized projection as a GEMM on
// the simdgroup-matrix unit, for the batched decode.
//
// Source: mlx/backend/metal/kernels/quantized.h (QuantizedBlockLoader, qmm_t_impl)
// plus the steel GEMM primitives it builds on:
//   steel/defines.h, steel/utils/integral_constant.h, steel/gemm/transforms.h,
//   steel/gemm/loader.h (BlockLoader), steel/gemm/mma.h (BaseMMAFrag, MMATile,
//   BlockMMA).
// MLX is MIT licensed; these are vendored so the math matches by construction,
// exactly as `quantized_qmv.metal` vendors qmv_fast_impl.
//
// Port notes:
//   * ONE self-contained file: pie compiles kernels at run time through
//     `newLibraryWithSource`, which does no filesystem include resolution.
//   * The batched branch (`adjust_matrix_offsets` and its eight shape/stride
//     buffers) is dropped, as in the qmv port — the decode passes one matrix.
//   * `M` is not a kernel argument. The driver only selects this kernel when
//     M % BM == 0, N % BN == 0 and K % BK == 0, so every tile is full and the
//     `load_unsafe` path is the only one reachable; the row count lives in the
//     grid. That matters because pie binds the projection constants once at
//     setup, not per fire, so a per-fire M has nowhere to come from.
//   * The complex64_t specialisations are stripped.
//
// This is NOT bit-identical to the GEMV: the accumulation order differs. Gate it
// on token agreement with the reference, not on the GEMV's sha256.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
#include "mxfp4_codec.h"

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
MLX_MTL_CONST int SIMD_SIZE = 32;

#include "../third_party/mlx/steel_prelude.metal"
#include "../third_party/mlx/steel_transforms.metal"
#include "../third_party/mlx/steel_mma.metal"
#include "../third_party/mlx/steel_loader.metal"
#include "../third_party/mlx/quantized_block.metal"

// ── pie entry ────────────────────────────────────────────────────────────────
// The one full-tile MMA loop. Callers choose the weight loader, input/output
// row pitches, K span, and epilogue; every exported variant keeps the same
// unsafe full-tile contract and therefore the same hot loop.
template <typename T, typename P, typename LoaderW, int BM, int BK, int BN,
          bool WITH_RESIDUAL, bool WITH_BIAS, int WM = 2, int WN = 2>
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
  // Two fences a K-step, and nothing overlaps between them: the matrix units
  // idle through the weight read, the load units idle through the MMA. The
  // obvious repair is a second threadgroup tile so the next step's reads issue
  // under this step's MMA, and it was written and measured -- gpt-oss prefill,
  // one binary, tok/s at 128 / 448 / 1024 rows: 410.0 / 539.2 / 566.9 single
  // against 379.4 / 500.4 / 559.0 double. Slower at every length.
  //
  // The reason is occupancy, which is how this GPU hides the latency anyway:
  // the widest tile here is 10 KiB of threadgroup memory against a 32 KiB
  // budget, so three threadgroups sit on a core and one's load overlaps
  // another's MMA for free. Doubling the tile leaves room for one, and
  // hand-pipelining one threadgroup does not pay for the two it evicted. The
  // loss shrinks with row count (-7.5%, -7.2%, -1.4%) exactly as that story
  // predicts. Not a tuning knob: there is no shape here where it wins.
  //
  // This also settles THE TWO FENCES below, which read like separate targets
  // and are not. The first guards WAR (last iteration's `mma` still reading
  // Xs/Ws) and the second RAW (this iteration's loads not landed yet), and
  // the only way to drop the first is to give the next tile its own buffer --
  // which is the double buffering the table above already priced and lost. A
  // fence here is not a latency to remove; it is what lets three threadgroups
  // share a core, and that sharing is worth more than the stall it costs.
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
    // The bias is per-column and the accumulator is in registers, so it folds
    // into the store. This used to be a second pass through device memory --
    // store, device barrier, read back, add, store again -- which cost the
    // whole output tile two extra trips on every routed expert GEMM.
    mma_op.store_result_bias(y, y_row_stride, residual + y_col);
  } else {
    mma_op.store_result(y, y_row_stride);
  }
}

/// The same loop, staged to HALF when the device has no bfloat matrix unit.
///
/// A separate implementation and not a flag on the one above, because the two
/// differ in the x loader and the comment on `BlockLoaderCast` explains why
/// that is not a degenerate case of `BlockLoader`: the same-type path copies
/// whole `ReadVector`s and would lose that copy if it had to convert. What is
/// shared is everything that could drift -- the tile shape, the double-fence
/// K loop, the epilogue -- which is the same argument `affine_qmm_t_routed`
/// makes for being an entry point rather than an implementation.
///
/// Both callers are ROUTED. Nothing here is routing-aware: the expert is
/// already folded into the weight pointers the caller constructs, and `x` and
/// `y` are contiguous in the sorted order `moe_route_sort` produced. The dense
/// projections take a different road to the same instruction -- see
/// `qmm_t_fp16_precast_impl`, which stages the activations once in a separate
/// dispatch instead of converting each x tile N/BN times. That road is not
/// open here: the sorted buffer is written by a kernel whose row count only
/// the GPU knows, and paying the conversion per tile is what MXFP4 measured
/// as worth it anyway.
template <typename T, typename LoaderW, int BM, int BK, int BN, bool WITH_BIAS,
          bool WITH_RESIDUAL = false, int WM = 2, int WN = 2>
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
    // The same store, fence and read-back `qmm_t_loaded_impl` does, and for
    // the same reason: `BlockMMA` has no residual epilogue, and the fragment
    // layout that would let one thread find its own residual element is the
    // MMA's business. Measured free -- an o_proj with this epilogue times the
    // same as a q_proj without one.
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

/// The dense GEMM's body, with both threadgroup tiles staged as HALF.
///
/// Same buffers, same grid, same tile, same `BlockMMA`, same float
/// accumulator -- the only thing that moves is the element type the two
/// staging buffers hold, and therefore the type the matrix instruction runs
/// on. `BK_padded` is `BK + 16/sizeof(...)` either way, so the threadgroup
/// allocation and the occupancy it buys are unchanged.
///
/// This is not a tuning knob, it is a repair, and the measurement is the
/// argument. Llama-3.2-1B, one 2048-row prefill, gate_proj (8192x2048),
/// marginal cost of one more dispatch inside a fire, M1 Max:
///
///     bf16 tiles, this driver    18.81 ms   3.65 TFLOP/s
///     bf16 tiles, mlx-lm         19.89 ms   3.45 TFLOP/s
///     fp16 tiles, mlx-lm         11.34 ms   6.06 TFLOP/s
///
/// The two bf16 rows agreeing is the finding: this kernel was never behind
/// MLX's, and the whole 1.4x that a full prefill showed against mlx-lm was
/// the ELEMENT TYPE. Apple silicon before M3 has no bfloat matrix path, so
/// `simdgroup_matrix<bfloat>` lowers to conversions around a float multiply;
/// `simdgroup_matrix<half>` is native on every family this driver runs on,
/// and where bfloat is native too the two issue at the same rate. There is no
/// device on which staging bf16 is the faster choice.
///
/// The ABI does not move. `x`, `y`, the scales, the biases and the residual
/// are all still BF16, the accumulator is still float, and no name changes --
/// which matters because `kernels-vulkan` and `kernels-wgpu` diff their
/// entrypoint lists against this crate's, so a metal-only symbol would break
/// two sibling tables for a change that is an implementation detail.
///
/// What DOES move is precision, and only in the multiplicands: bf16 keeps
/// float's exponent range with 8 mantissa bits, half trades range for 11. The
/// operands here are a dequantized 4- or 8-bit weight, which spans nothing,
/// and an activation that is always a norm's output, an attention output or a
/// gated MLP's -- never the raw residual stream, which is added after the
/// store in BF16. `a_generation_agrees_with_mlx_token_for_token` runs over
/// real weights and is the gate on that claim; mlx-lm serves this checkpoint
/// family in fp16 end to end, which is a stronger version of the same bet.
///
/// `affine_qmm_t_routed` deliberately does NOT come here -- see its own
/// header: a routed model's next-layer top-k moved under FP16, so the mixture
/// keeps bfloat tiles and `affine_qmm_t_routed_fp16` stays the opt-in.
template <typename T, int group_size, int bits, int BM, int BK, int BN,
          bool WITH_RESIDUAL, bool WITH_BIAS = false, int WM = 2, int WN = 2>
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
  // The third value slot is the residual under `WITH_RESIDUAL` and the
  // projection's additive bias under `WITH_BIAS`, and the two are indexed
  // differently: the residual is a full [rows, N] block the epilogue walks
  // itself, the bias a length-N vector `store_result_bias` reads at the
  // TILE's column. `qmm_t_cast_loaded_impl`'s other callers pre-offset it,
  // so this one does too.
  qmm_t_cast_loaded_impl<T, loader_w_t, BM, BK, BN, WITH_BIAS, WITH_RESIDUAL,
                         WM, WN>(
      x, y, residual + (WITH_BIAS ? y_col : 0), Xs, Ws, K, N, tid, simd_gid,
      simd_lid, loader_w);
}

/// The same body with BFLOAT tiles, which one caller still wants.
///
/// `affine_qmm_t_routed` alone comes here. Its own header says why: a routed
/// model's next-layer top-k moved under FP16 in `llama_numerics_test`, so the
/// mixture keeps bfloat staging and pays the lowering, and
/// `affine_qmm_t_routed_fp16` stays the opt-in a host can gate. Every DENSE
/// projection takes [`qmm_t_aligned_half_impl`] above instead.
template <typename T, int group_size, int bits, int BM, int BK, int BN,
          bool WITH_RESIDUAL, bool WITH_BIAS = false, int WM = 2, int WN = 2>
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
          bool WITH_BIAS = false, bool WITH_RESIDUAL = false>
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

  using loader_w_t = QuantizedBlockLoader<
      bfloat, BN, BK, BK_padded, 1, 4 * SIMD_SIZE, group_size, bits, half>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_col = int(tid.x) * BN;

  auto wl = (const device uint8_t*)w;
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  qmm_t_loaded_impl<half, P, loader_w_t, BM, BK, BN, WITH_RESIDUAL, WITH_BIAS>(
      x, y, bias, Xs, Ws, K, N, k_len,
      tid, simd_gid, simd_lid, loader_w);
}

// Dense g64/b4 keeps the model ABI in BF16 but stages each projection source
// once. Casting inside every output tile repeated the same conversion N/BN
// times (128 times for gate/up); the tiny staging pass removes that multiplier.
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

/// The same GEMM with the projection's additive bias folded into the store.
///
/// gpt-oss biases every dense projection, so the plain precast kernel could
/// not serve it: the bias would have needed a second pass over the output
/// tile, which is exactly the trip `store_result_bias` exists to avoid. Buffer
/// 7 is `bind::GoQmv::Bias`, already bound there by the matvec this replaces.
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

/// The same GEMM adding the layer's residual row to its output.
///
/// Buffer 7 again, which the BF16 `affine_qmm_t_aligned_residual` already uses
/// for the same pointer -- so a family that had one has the other for free.
/// The epilogue is not the bias's: a residual is per-ROW and the accumulator
/// is not, so `qmm_t_loaded_impl` stores, fences and reads back, where the
/// bias folds into the store. Both are its code and neither is repeated here.
///
/// qwen3.5's fused-residual projections are 28% of its batched decode, and
/// without this they were the half of that path still on the emulated BF16
/// matrix unit while the other half had moved.
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
    // Buffer 7 is `bind::GoQmv::Bias`, which gpt-oss's matvec already binds
    // there -- so the batched path needs no host binding of its own.
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
          int WM = 2, int WN = 2>
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

/// The same matmul, against the expert slice this TILE was routed to.
///
/// The whole of the batched mixture is here: `moe_route_sort` has already put
/// the rows that share an expert next to each other and padded each run to a
/// tile, so a tile's rows agree on their expert and the only thing that changes
/// per tile is where the weights start. `x` and `y` are read and written
/// contiguously in that sorted order -- the permutation is the gather's job and
/// the scatter's, not this kernel's.
///
/// That is why this is an entry point and not an implementation. Nothing about
/// the inner loop differs from the dense case, and a routed copy of
/// `qmm_t_aligned_impl` would be a second place for the tile shape, the loader
/// and the epilogue to drift.
///
/// `tile_expert[tid.y] < 0` is a tile past the end of the routing. The grid is
/// the WORST CASE -- every expert claiming a partial tile -- because the real
/// tile count is a number the GPU computed and the host cannot see without a
/// stall, so the spare tiles have to be dispatched and then decline. The return
/// is uniform across the threadgroup, which is what makes it safe to take
/// before the barriers inside the impl.
///
/// The worst case is the ALLOCATION's problem and not the arithmetic's, which
/// is worth saying because the two are easy to confuse. `moe_sorted_rows` is
/// pessimistic by design -- at 448 rows of top-4 over 32 experts it asks for
/// 2784 sorted rows where 1792 carry a token -- but the surplus tiles decline
/// above and cost a threadgroup launch rather than a GEMM. What the arithmetic
/// actually pays is each expert's own run rounded up, which for a router that
/// spreads evenly is nearer 14% than the bound's 55%.
///
/// And the mixture's kernel is not the slow one. Measured with `roofline_probe`
/// at the expert shape K=N=2880, GFLOP/s, against the affine kernel it shares
/// an inner loop with:
///
///     BM=16    affine 2741    mxfp4 3141
///     BM=32    affine 3741    mxfp4 4002
///
/// and 4504 at BM=64. MXFP4 is ahead of affine at every width.
///
/// In situ it does not reach that. A 1024-row gpt-oss prefill puts 128 rows in
/// each of 32 experts, which BM=64 covers in two whole tiles with NO padding at
/// all, and the 72 dispatches then run 4892 GFLOP in 1.289 s: 3796 GFLOP/s,
/// 16% under the probe. The difference is not bandwidth -- the same fire moves
/// 10.2 GB of expert weight, which is 7.9 GB/s and nowhere near any roof -- it
/// is that the probe reads ONE expert over and over where a mixture reads
/// thirty-two. A threadgroup here reuses its weight slice across the tiles of
/// its own expert and no further, which at two tiles an expert is a reuse of
/// two.
///
/// So the remaining gap to a dense prefill is that reuse. The other thing it
/// used to be -- that the input could not be staged to FP16 -- is no longer
/// true of the kernel below it: `mxfp4_qmm_t_routed_bias` stages both tiles to
/// half, and `affine_qmm_t_routed_fp16` does the same for this format. What
/// keeps a model on THIS kernel is `fp16_qmm` saying no, which llama's routed
/// projections do because their next-layer top-k moved under FP16.
template <typename T, int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_routed(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device T* x          [[buffer(3)]],
    device T* y                [[buffer(4)]],
    const constant int& K      [[buffer(5)]],
    const constant int& N      [[buffer(6)]],
    // Buffer 12, clear of every slot `bind::GoQmv` uses: one argument table
    // ordinal serves this pipeline and the routed matvec both, and the host
    // binds all of an ordinal's slots whichever one the row count selects.
    const device int* tile_expert [[buffer(12)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  const int e = tile_expert[tid.y];
  if (e < 0) return;

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  // The stack is [n_experts * N, K] as the contract declared it: one flat
  // matrix, expert-major, so an expert's slice is a row offset and not a
  // separate allocation.
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

/// The routed affine matmul with the tiles and the MMA in HALF.
///
/// Same buffers, same grid, same `tile_expert` contract as the kernel above --
/// the host picks between them by NAME and changes nothing else. That is the
/// point: a routed GEMM's argument table is shared with the routed matvec, so
/// a variant that needed a different binding would have to be threaded through
/// the decode path as well for no reason.
///
/// The checkpoint and the output stay bfloat. Only the two threadgroup tiles
/// and the matrix instruction are half, which on a device without a native
/// bfloat matrix unit is the difference between an emulated sequence and one
/// instruction -- about 40% on the GEMM, measured on the dense projections
/// (see `gemma4_fp16_qmm`). The weight loader converts for free: it has to
/// dequantize into threadgroup memory anyway, so asking it for `half` costs
/// nothing over asking it for `bfloat`.
///
/// gemma-4-26b-a4b spends 47.9% of a 512-row prefill in the BF16 form of this
/// kernel, which is the largest single term in that model's prefill and the
/// reason this exists.
///
/// Correctness is the host's call and not the kernel's: `fp16_qmm` gates it,
/// and llama keeps its routed projections on the BF16 kernel because a routed
/// model's next-layer top-k moved under FP16 in `llama_numerics_test`.
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
      T, BN, BK, BK_padded, 1, 4 * SIMD_SIZE, group_size, bits, half>;

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
  const int e = tile_expert[tid.y];
  if (e < 0) return;

  constexpr int BK_padded = BK + 16 / sizeof(half);
  constexpr int tgp_size = 4 * SIMD_SIZE;
  const int y_col = int(tid.x) * BN;
  const size_t expert_w = size_t(e) * size_t(N) * size_t(K) / 2;
  const size_t expert_s = size_t(e) * size_t(N) * size_t(K / 32);
  const device uint8_t* wb = (const device uint8_t*)w + expert_w + y_col * K / 2;
  const device uint8_t* sb = exponents + expert_s + y_col * (K / 32);

  // The tiles and the MMA are HALF while the checkpoint and the output stay
  // bfloat, which is the same trick `fp16_qmm` plays on the DENSE projections
  // and the largest single win in this driver -- about 40% on the GEMM. The
  // MXFP4 loader gets it cheapest of all: it has to decode into threadgroup
  // memory anyway, so asking it for `half` instead of `bfloat` is free.
  //
  // The loop and the epilogue are shared with `affine_qmm_t_routed_fp16`; only
  // the weight loader differs.
  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];
  using loader_w_t =
      mlx::steel::Mxfp4BlockLoader<T, BN, BK, BK_padded, 1, tgp_size, half>;
  loader_w_t loader_w(wb, sb, K, Ws, simd_gid, simd_lid);
  qmm_t_cast_loaded_impl<T, loader_w_t, BM, BK, BN, true>(
      x, y, bias + size_t(e) * size_t(N) + y_col, Xs, Ws, K, N, tid, simd_gid,
      simd_lid, loader_w);
}

// ── THE FOUR FORMS THIS FILE CAN STAMP, AND NOT ONE POINT OF THEM ──────────
//
// `#define instantiate_qmm_t(gs, bm, bk, bn, b)` stood here with fifty-four
// calls under it, stamping four entry points each: 216 pipelines codegen'd on
// every load to reach the six a model actually fires.
//
// The calls are gone and the `#define`s are not, because they are two
// different things. A `#define` declares WHAT CAN BE STAMPED -- and holds the
// device signature, which is this file's to own and must stay written once. A
// call declares WHICH POINT, and the host is the only party that knows: it is
// the one that picked the tile.
//
// so the host composes one of these calls per fire and the driver appends it
// to this source before compiling (`Fire::stamp`). A point is checked by the
// host that picks it (`quant::qmm_point`), which is the party that knows the
// tile, and reached by being asked for.
//
// WHAT THIS DELETED, beyond the fifty-four lines: `moe.rs`'s fifty-four-entry
// table of these same names, and the fold that indexed it. What it did NOT
// delete is `build.rs` -- see `kernels_metal::kernel_of`, whose census the
// expander still feeds for the families that have not moved.
//
// `entry` IS A STRING LITERAL, composed host-side. The names are unchanged --
// `affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_32` is what `qmm_point` spells and
// what the driver's stem table already knows. Renaming them is a separate
// change and this one does not make it.

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

// ── WHAT THE DELETED CALL LIST KNEW, KEPT ──────────────────────────────────
//
// Two of the fifty-four lines carried findings rather than coordinates, and a
// coordinate is the only thing the host took over.
//
// **`(bm 32, bn 64)` is not optional.** `qmm_bn` takes the widest tile that
// DIVIDES the output and every projection in these checkpoints is a multiple
// of 64, so a 32-row batch asks for exactly that pair. It was once left out
// and ALIASED onto the BM=16 pipeline -- not a crash: the grid is built for 32
// rows per block and the pipeline computes 16, so half the batch is never
// written. At 32 rows gemma4's logits came back all zero.
//
// That failure is now unreachable rather than fixed. The host composes the
// name and the stamp from ONE set of numbers (`quant::qmm_point`), so the
// pipeline it gets is the pipeline it built the grid for; there is no list to
// be missing from and nothing to fall back onto.
//
// **`bm 64` earns its rung.** A prompt is not a 32-row batch: 128 tokens at
// BM=32 unpack every weight four times and the GEMM is memory-bound on exactly
// that unpacking. BM=64 halves it again at no cost in accumulators -- 64x32
// over 128 lanes is the same sixteen per lane that 32x64 already spends -- and
// buys 20% on a llama-1B prefill. Reached only when the batch divides by 64.


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

// Only g64/b4, which is what `fp16_qmm` gates on: an 8-bit routed bank has no
// FP16 pipeline anywhere in this driver, and instantiating one here would cost
// every load the compile without a caller.
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
  [[kernel]] void affine_qmm_t_fp16_precast<64, 4, bm, 32, bn>(             \
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

#define instantiate_qmm_t_bias_fp16_precast(bm, bn)                          \
  template [[host_name("affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4"   \
                       "_bm_" #bm "_bn_" #bn)]]                              \
  [[kernel]] void affine_qmm_t_bias_fp16_precast<64, 4, bm, 32, bn>(         \
      const device uint32_t*, const device bfloat*, const device bfloat*,    \
      device bfloat*, const constant int&, const constant int&,              \
      const device bfloat*, const device half*, uint3, uint, uint);

#define instantiate_qmm_t_residual_fp16_precast(bm, bn)                      \
  template [[host_name("affine_qmm_t_residual_fp16_precast_bfloat16"         \
                       "_gs_64_b_4_bm_" #bm "_bn_" #bn)]]                    \
  [[kernel]] void affine_qmm_t_residual_fp16_precast<64, 4, bm, 32, bn>(     \
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

instantiate_qmm_t_bias_fp16_precast(16, 16)
instantiate_qmm_t_bias_fp16_precast(16, 32)
instantiate_qmm_t_bias_fp16_precast(16, 64)
instantiate_qmm_t_bias_fp16_precast(32, 16)
instantiate_qmm_t_bias_fp16_precast(32, 32)
instantiate_qmm_t_bias_fp16_precast(32, 64)
instantiate_qmm_t_bias_fp16_precast(64, 16)
instantiate_qmm_t_bias_fp16_precast(64, 32)
instantiate_qmm_t_bias_fp16_precast(64, 64)

// ── Strided form, for the prefill ────────────────────────────────────────────
// Identical to the aligned kernel above except that the row pitch of `x`, `y`
// and `residual` is given separately from `K`/`N`.
//
// The prefill lays token t's slice of EVERY scratch tensor at
// `t * scratch_widest_elems`, a uniform pitch chosen so one pool slot can hold
// any of them -- not at `t * K`, which is what the decode's packed layout does
// and what the aligned kernel assumes. That pitch is a property of the model,
// not of the fire, so it binds once at setup like `K` and `N` do.
//
// With it, a prompt's projections stop being N sequential GEMVs over the same
// weights and become one GEMM: 34 reads of the checkpoint collapse to one.
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
      T, BN, BK, BK_padded, 1, 4 * SIMD_SIZE, group_size, bits>;

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

// Qwen3.5's prefill scratch uses one uniform row pitch. Mirror the live K
// columns into a half buffer once per projection; unlike tile-local casting,
// this conversion is independent of the number of output tiles.
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
      bfloat, BN, BK, BK_padded, 1, 4 * SIMD_SIZE, group_size, bits, half>;

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
// The 8-bit twin, for a checkpoint that spares individual tensors from the
// model-wide format. mlx-lm's quantization predicate can single out a tensor by
// NAME, and Qwen3.6-35B-A3B's build leaves the MoE router and the shared
// expert's gate at 8 bits while everything else is 4. Those two kinds had no
// batched shape to run in, so a prefill fell back to ONE MATVEC PER TOKEN for
// them -- 40 layers times two kinds times every row -- and that was the single
// largest line in the profile. The kernel body is parametric in `bits`; only
// the instantiation was missing.
instantiate_qmv_wide_strided(8, 4, 8)


// Split-K's affine-loader adapter. The common loop runs `k_len` columns from
// bases the caller advanced, while row pitches still come from the full K.
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
      T, BN, BK, BK_padded, 1, 4 * SIMD_SIZE, group_size, bits>;

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

// ── Split-K, following MLX's dispatch ────────────────────────────────────────
//
// MLX does not use the plain qmm for a transposed non-batched decode at all:
// `QuantizedMatmul::eval_gpu` sends that case to `qmm_splitk`
// (backend/metal/quantized.cpp:1518), and picks the split to land near 512
// threadgroups:
//
//     split_k = max(1, 512 / (n_tiles * m_tiles))
//
// which is the same saturation point `roofline_probe` measures independently --
// this GEMM's throughput tracks threadgroup count almost linearly up to ~200 and
// flattens past it.  A projection to hidden (N=1024) gets 32 tiles and so a
// split of 16; a wide one like gate/up (N=3584) gets 4; lm_head has 7760 tiles
// of its own and takes none.
//
// An earlier attempt here used a fixed split of 4 for every shape and measured
// slower end to end.  That is why: the shapes that needed it most were given a
// quarter of the parallelism they had room for.
//
// Each threadgroup takes one K partition and writes its own [M, N] slice; the
// reduce below sums the slices, in float, whatever the partials are stored as.
//
// The partials are FLOAT for every projection, and that is a correctness
// requirement rather than a preference. A bfloat partial carries seven mantissa
// bits, so each partition's accumulated dot product is rounded to about 0.4%
// before anything sums it, and the error on the total is that times
// `sum|p_i| / |sum p_i|` -- the cancellation factor. For a projection whose
// partitions largely cancel, which is the ordinary case for an attention
// output, that factor is one to two orders of magnitude and the split GEMM
// simply returns a different answer from the unsplit one.
//
// It shipped with bf16 partials for every kind but V, on the reading that the
// side-buffer traffic is otherwise a visible second pass over the output and
// that V was special because rounding it "moved a routed model's next-layer
// top-k choice in llama_numerics_test". V was not special. It was the first
// place the effect was large enough to cross a gate. The others cross it too:
// `llama_numerics_test`'s eight-row mixture diverges at layer 1's O projection
// at 0.0999 rel_l2 against a 0.06 tolerance, and a sixteen-wide llama-1b fleet
// generates visibly wrong text -- both only once the batch is wide enough to
// take this kernel at all, which is why lowering the GEMM crossover is what
// exposed it rather than what caused it.
//
// The traffic the bf16 partials were saving is real and is now paid in full.
// It is `split_k * M * N` extra bytes written and read once each, against a
// GEMM that reads `M * K * N / BN`; measured end to end it is inside the
// run-to-run spread on every checkpoint here.
//
// The K sum is reassociated into `split_k` contiguous blocks -- pairwise rather
// than strictly sequential, which is the better-conditioned order, but it does
// mean this is not bit-identical to the unsplit kernel.
template <typename T, typename P, int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void affine_qmm_t_splitk(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales   [[buffer(1)]],
    const device T* biases   [[buffer(2)]],
    const device T* x        [[buffer(3)]],
    device P* y              [[buffer(8)]],  // [split_k, M, N]
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

  // The impl walks `K` columns from the bases above, so pass the partition
  // length as its K -- the row pitches (K_w, K_g) come from the real K.
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

// Sum the split_k slices and write the activation type, folding in the block
// residual the fused projection would otherwise have applied.
template <typename T, typename P>
[[kernel]] void qmm_splitk_reduce(
    device T* y                 [[buffer(4)]],
    const constant int& N       [[buffer(6)]],
    const device P* partial     [[buffer(8)]],
    const constant int& stride  [[buffer(10)]],
    // NOT buffer(9). One argument table serves BOTH halves of a split
    // projection, and 9 is the GEMM's `k_partition_size`, the partition LENGTH
    // rather than the partition COUNT.
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


template [[host_name("qmm_splitk_reduce_bfloat16")]] [[kernel]] void
qmm_splitk_reduce<bfloat, bfloat>(
    device bfloat*, const constant int&, const device bfloat*,
    const constant int&, const constant int&, uint2);
template [[host_name("qmm_splitk_reduce_f32_bfloat16")]] [[kernel]] void
qmm_splitk_reduce<bfloat, float>(
    device bfloat*, const constant int&, const device float*,
    const constant int&, const constant int&, uint2);

// Warp-shape variants, kept for `roofline_probe`'s PROBE_WM/PROBE_WN.
//
// The K loop's note says occupancy is how this GPU hides the kernel's latency,
// and at 128 threads a threadgroup only three fit on a core. A 256-thread
// threadgroup puts twice the threads there for the same threadgroup memory and
// the same sixteen accumulators a lane already holds, so it is the one shape
// argument the tile sweep never tested. It loses. M=1024, M1 Max, GFLOP/s over
// the checkpoint's projections:
//
//     BM=64  BN=32  2x2  (128 thr)   4530   the shipping shape
//     BM=64  BN=64  2x4  (256 thr)   4090   -9.7%
//     BM=128 BN=32  4x2  (256 thr)   4470   -1.3%
//
// Fewer, fatter threadgroups is what it minds, which is the same thing BN=64
// says in the model (104.6 vs 107.2 tok/s) from the other direction. These two
// are instantiated and not dispatched ON PURPOSE: the sweep that closes an
// axis is worth as much as the one that opens it, and the next person to
// suspect occupancy should be able to re-run it rather than re-derive it.
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

// The other direction, which the note above argues for and never tested. If
// "fewer, fatter threadgroups is what it minds" then 128 threads is not the
// floor of that argument -- 64 is. A 2x1 or 1x2 warp shape halves the
// threadgroup at the same tile, so six fit on a core where three do, and a
// standalone `matmul2d` sweep at this checkpoint's prefill shape (M=128,
// K=N=5120) does put its best arm at two simdgroups: 6.33 TFLOP/s against
// 6.29 at four and 3.23 at eight.
//
// IT LOSES, and it loses to the cost the argument predicted. A lane holds
// TM*TN*2 accumulator fragments and halving the simdgroups doubles one of TM
// or TN, so 16 becomes 32 and the registers give back what the threadgroup
// count won. `roofline_probe <kernels> 128 32`, M1 Max, whole-step TFLOP/s:
//
//     BM=64  2x2  (128 thr)   3.88   the shipping shape
//     BM=64  2x1  ( 64 thr)   3.61   -7.0%
//     BM=64  1x2  ( 64 thr)   3.58   -7.7%
//     BM=32  1x2  ( 64 thr)   3.34   -13.9%
//
// So the occupancy axis is now closed in BOTH directions around 2x2 -- eight
// simdgroups is -9.7% by the table above, two is -7.0% by this one. That is
// worth more than either arm winning would have been: it means the shipping
// warp shape is a peak and not a default nobody questioned. Kept instantiated
// and undispatched, like the 256-thread pair, so the next person to have this
// idea can re-run it in a minute instead of re-deriving it.
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
