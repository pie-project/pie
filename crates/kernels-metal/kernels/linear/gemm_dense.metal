#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
MLX_MTL_CONST int SIMD_SIZE = 32;

constant float kMxfp4Lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float mxfp4_lo(uint8_t byte) { return kMxfp4Lut[byte & 0xf]; }
inline float mxfp4_hi(uint8_t byte) { return kMxfp4Lut[byte >> 4]; }

inline float mxfp4_block_scale(uint8_t code) {
  return code == 0xff ? NAN : metal::ldexp(1.0f, int(code) - 127);
}

#include "../third_party/mlx_steel_prelude.metal"
#include "../third_party/mlx_steel_transforms.metal"
#include "../third_party/mlx_steel_mma.metal"
#include "../third_party/mlx_steel_loader.metal"

template <typename T, int BM, int BK, int BN, int WM = 2, int WN = 2>
[[kernel]] void dense_gemm_t(
    const device T* act      [[buffer(0)]],
    const device T* w        [[buffer(1)]],
    device T* y              [[buffer(2)]],
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

template <typename T>
[[kernel]] void dense_gemv_t(
    const device T* act      [[buffer(0)]],
    const device T* w        [[buffer(1)]],
    device T* y              [[buffer(2)]],
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
