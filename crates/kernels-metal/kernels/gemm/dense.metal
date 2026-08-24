// The DENSE bf16 projection: `y[M, N] = act[M, K] @ w[N, K]^T`.
//
// This is `gemm.matmul`, `gemm.lm_head` and `gemm.attention_landing` — one
// arithmetic under three names, exactly as `kernels-cuda/src/gemm.rs` answers
// all three with one `act_x_wt_bf16`. The weight is stored TRANSPOSED
// (`[N, K]`, output-major, K contiguous), which is what the point declaration
// states by sizing the result `[act.rows, w.axis(0)]`: axis 0 of the weight is
// the output width, so the contraction runs down each weight ROW.
//
// # Why a shader and not a vendor call
//
// On CUDA this point is cuBLAS. There is no equivalent road on this plane:
// `kernels::plane::Fire` names a FILE, an ENTRYPOINT and a GRID, and
// `driver-metal` turns exactly that into `dispatchThreads`. Nothing in the
// crossing can carry "call MPSMatrixMultiplication instead", and `TIER2` is
// not that lever either — a tier-2 point is one this plane DECLARES under its
// own name (`metal::foo`), not an alternate answer to a floor point a model
// text spells `gemm.matmul`. So the choice here was own-shader or nothing.
//
// # Two entry points, and the rule that picks between them
//
//   * `dense_gemm_t` — the simdgroup-matrix tile loop, for M >= BM.
//   * `dense_gemv_t` — one simdgroup per output column, for M < BM.
//
// The split is not a tuning knob, it is the tile's own arithmetic. The MMA
// path computes a whole BM x BN tile whatever M is, so at M = 1 it does BM
// times the multiplies the answer needs; the vector path does exactly M*N*K
// and no more. BM is therefore the crossing point BY CONSTRUCTION, and the
// host states it in `kernels_metal::gemm::TILE_M`. Where inside that range
// the two actually cross on a given device is UNMEASURED — this file has
// never been compiled, let alone timed, and nothing here should be read as a
// performance claim.
//
// # The tiles are BFLOAT, and `quant/qmm_t.metal` argues the other way
//
// That file stages its threadgroup tiles as `half` and reports ~40% for it:
// Apple silicon before M3 has no bfloat matrix unit, so
// `simdgroup_matrix<bfloat>` lowers to conversions around a float multiply.
// It can do that because `BlockLoaderCast` converts on the way in — and
// `BlockLoaderCast` has NO `load_safe`, which is the whole reason this file
// does not follow it. A dense GEMM's M is the token count: 1 at decode, any
// integer at prefill, and never a multiple of the tile by construction. So
// this path must bounds-check, `BlockLoader` is the loader that can, and its
// element type is the tile's element type. Staging half here means either
// forking the vendored loader (which its own header forbids) or refusing
// every ragged shape, and a GEMM that refuses ragged M is not a GEMM.
//
// The accumulator is `float` either way — `BlockMMA`'s `AccumType` default —
// so what the element type costs is issue rate, not precision.
//
// # UNVERIFIED
//
// No Metal toolchain and no Apple device existed where this was written. It
// has not been compiled by `metal`, and no number it computes has been
// compared against anything. What IS checked is the shape contract on the
// host side (`kernels_metal::gemm`) and the entrypoint census `build.rs`
// stamps from the `instantiate_*` calls below.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

// `steel_transforms.metal` spells its constants `MLX_MTL_CONST` and expects
// the including file to have defined it, the way `quant/qmm_t.metal` does.
#define MLX_MTL_CONST static constant constexpr const
MLX_MTL_CONST int SIMD_SIZE = 32;

// THE MXFP4 CODEC IS HERE FOR A COMPILER RULE, NOT FOR THIS KERNEL. Nothing
// below decodes a 4-bit block. `steel_transforms.metal` is where the vendored
// tree keeps `TransformNone` — `BlockMMA`'s default epilogue, which this file
// does need — and the same file also holds `Mxfp4BlockLoader`, whose
// `load_unsafe` calls `mxfp4_block_scale`, `mxfp4_lo` and `mxfp4_hi` on a
// plain `uint8_t`. Those calls are NON-DEPENDENT names in a template, so
// two-phase lookup resolves them where the template is DEFINED and not where
// it is instantiated: leaving the codec out is an error at the include even
// though the loader is never instantiated. `quant/qmm_t.metal` splices the
// same header first for the same reason; the alternative is editing a
// vendored file to fit a caller, which its own header forbids.
#include "../quant/mxfp4_codec.h"

#include "../third_party/mlx/steel_prelude.metal"
#include "../third_party/mlx/steel_transforms.metal"
#include "../third_party/mlx/steel_mma.metal"
#include "../third_party/mlx/steel_loader.metal"

// The tile loop.
//
// `transpose_b = true` in the `BlockMMA` is what makes the weight's own
// storage the right one: it sets `B_str_k = 1` and `B_str_n = ldb_tgp`, so the
// staged W tile is `[BN rows, BK columns]` with K contiguous — which is
// `w[N, K]` read at `w + n*K + k`, no transpose pass anywhere.
//
// THE FULL TILE AND THE RAGGED ONE TAKE THE SAME LOOP. `whole` is derived from
// `tid`, which is the THREADGROUP's position, so every thread in the group
// agrees on it and the branch is uniform — the barriers below are reached by
// all threads or by none. Interior tiles get `load_unsafe`, which copies whole
// `ReadVector`s; only the edge tiles pay the per-element bound check, and the
// K tail is its own trailing block because `load_safe` zero-fills the columns
// past K and zeros contribute nothing to the accumulation.
template <typename T, int BM, int BK, int BN, int WM = 2, int WN = 2>
[[kernel]] void dense_gemm_t(
    const device T* act      [[buffer(0)]],  // [M, K]
    const device T* w        [[buffer(1)]],  // [N, K]
    device T* y              [[buffer(2)]],  // [M, N]
    const constant int& M    [[buffer(3)]],
    const constant int& N    [[buffer(4)]],
    const constant int& K    [[buffer(5)]],
    uint3 tid     [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(T);
  constexpr int tgp_size = WM * WN * SIMD_SIZE;
  using loader_a_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, tgp_size>;
  using loader_b_t = mlx::steel::BlockLoader<T, BN, BK, BK_padded, 1, tgp_size>;
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;

  threadgroup T As[BM * BK_padded];
  threadgroup T Bs[BN * BK_padded];

  const int y_row = int(tid.y) * BM;
  const int y_col = int(tid.x) * BN;
  const short rows_left = short(min(M - y_row, BM));
  const short cols_left = short(min(N - y_col, BN));
  const bool whole = rows_left == BM && cols_left == BN;

  loader_a_t loader_a(
      act + size_t(y_row) * size_t(K), K, As, simd_gid, simd_lid);
  loader_b_t loader_b(w + size_t(y_col) * size_t(K), K, Bs, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  int k = 0;
  for (; k + BK <= K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (whole) {
      loader_a.load_unsafe();
      loader_b.load_unsafe();
    } else {
      loader_a.load_safe(short2(short(BK), rows_left));
      loader_b.load_safe(short2(short(BK), cols_left));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.next();
    loader_b.next();
  }
  // The K tail. `next()` already walked both sources to column `k`, so the
  // only thing left to say is how many columns are really there.
  if (k < K) {
    const short tail = short(K - k);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_safe(short2(tail, rows_left));
    loader_b.load_safe(short2(tail, cols_left));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  device T* dst = y + size_t(y_row) * size_t(N) + size_t(y_col);
  if (whole) {
    mma_op.store_result(dst, N);
  } else {
    // `short2(x = columns, y = rows)`, which is the order `store_result_safe`
    // subtracts its own `(sn, sm)` in.
    mma_op.store_result_safe(dst, N, short2(cols_left, rows_left));
  }
}

#define instantiate_dense_gemm_t(name, itype, bm, bk, bn)                    \
  template [[host_name("dense_gemm_t_" #name "_bm_" #bm "_bn_" #bn)]]        \
  [[kernel]] void dense_gemm_t<itype, bm, bk, bn>(                           \
      const device itype*, const device itype*, device itype*,               \
      const constant int&, const constant int&, const constant int&,         \
      uint3, uint, uint);

instantiate_dense_gemm_t(bfloat16, bfloat, 32, 32, 32)

// The small-M road: ONE SIMDGROUP PER OUTPUT COLUMN.
//
// Each simdgroup owns one `(row, column)` result. Its 32 lanes stride the
// contraction, each accumulating in float, and `simd_sum` folds the 32 partial
// sums in registers — no threadgroup memory, no barrier, one store per
// simdgroup from lane 0.
//
// The reduction order is therefore NOT the tile kernel's, which is a real
// numeric difference and not a rounding footnote: `dense_gemm_t` accumulates
// through `simdgroup_matrix` in BK-sized steps while this walks K in strides
// of 32. Both are float accumulations of the same products; neither is a
// reference for the other bit-for-bit.
//
// `n >= N` IS UNIFORM WITHIN A SIMDGROUP and that is what makes the early
// return safe in front of `simd_sum`: `n` is `gid.x / 32`, so all 32 lanes of
// a simdgroup share it. The host rounds the x extent up to whole threadgroups
// (`GEMV_GROUP` threads = `GEMV_GROUP / 32` columns), so the surplus is always
// whole simdgroups and never a split one.
template <typename T>
[[kernel]] void dense_gemv_t(
    const device T* act      [[buffer(0)]],  // [M, K]
    const device T* w        [[buffer(1)]],  // [N, K]
    device T* y              [[buffer(2)]],  // [M, N]
    const constant int& M    [[buffer(3)]],
    const constant int& N    [[buffer(4)]],
    const constant int& K    [[buffer(5)]],
    uint2 gid  [[thread_position_in_grid]],
    uint lane  [[thread_index_in_simdgroup]]) {
  const int n = int(gid.x) / SIMD_SIZE;
  const int m = int(gid.y);
  if (n >= N || m >= M) {
    return;
  }
  const device T* act_row = act + size_t(m) * size_t(K);
  const device T* w_row = w + size_t(n) * size_t(K);
  float acc = 0.0f;
  for (int k = int(lane); k < K; k += SIMD_SIZE) {
    acc += float(act_row[k]) * float(w_row[k]);
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    y[size_t(m) * size_t(N) + size_t(n)] = static_cast<T>(acc);
  }
}

#define instantiate_dense_gemv_t(name, itype)                                \
  template [[host_name("dense_gemv_t_" #name)]]                              \
  [[kernel]] void dense_gemv_t<itype>(                                       \
      const device itype*, const device itype*, device itype*,               \
      const constant int&, const constant int&, const constant int&,         \
      uint2, uint);

instantiate_dense_gemv_t(bfloat16, bfloat)
