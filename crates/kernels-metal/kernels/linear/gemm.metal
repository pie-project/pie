// Portions derived from MLX (https://github.com/ml-explore/mlx, MIT License):
//   mlx/backend/metal/kernels/steel/gemm/gemm.h
// renamed into Pie's namespaces; crates/kernels-metal/NOTICE lists the changes.
//
// Copyright © 2023 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

#define PIE_CONST static constant constexpr const
PIE_CONST int SIMD_SIZE = 32;

constant float kMxfp4Lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float mxfp4_lo(uint8_t byte) { return kMxfp4Lut[byte & 0xf]; }
inline float mxfp4_hi(uint8_t byte) { return kMxfp4Lut[byte >> 4]; }

inline float mxfp4_block_scale(uint8_t code) {
  return code == 0xff ? NAN : metal::ldexp(1.0f, int(code) - 127);
}

#include "../prim/gemm_traits.h"
#include "../prim/gemm_transforms.h"
#include "../prim/gemm_mma.h"
#include "../prim/gemm_loader.h"

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
  using loader_a_t = pie::gemm::BlockLoader<T, BM, BK, BK_padded, 1, tgp_size>;
  using loader_b_t = pie::gemm::BlockLoader<T, BN, BK, BK_padded, 1, tgp_size>;
  using mma_t = pie::gemm::
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


#define instantiate_dense_gemm_t(name, itype, bm, bk, bn, wm, wn)            \
  template [[host_name(                                                      \
      "dense_gemm_t_" #name "_bm_" #bm "_bk_" #bk "_bn_" #bn)]]              \
  [[kernel]] void dense_gemm_t<itype, bm, bk, bn, wm, wn>(                   \
      const device itype*, const device itype*, device itype*,               \
      const constant int&, const constant int&, const constant int&,         \
      uint3, uint, uint);

instantiate_dense_gemm_t(bfloat16, bfloat, 32, 32, 32, 2, 2)

instantiate_dense_gemm_t(bfloat16, bfloat, 8, 64, 32, 1, 2)

inline float bf16x2_dot(uint a, uint b) {
  const float a_lo = as_type<float>(a << 16);
  const float a_hi = as_type<float>(a & 0xffff0000u);
  const float b_lo = as_type<float>(b << 16);
  const float b_hi = as_type<float>(b & 0xffff0000u);
  return a_lo * b_lo + a_hi * b_hi;
}

template <typename T>
inline float gemv_lane_dot(const device T* act_row, const device T* w_row, int K, int k0, int stride) {
  float acc = 0.0f;
  if ((K & 1) == 0) {
    const device uint* a2 = reinterpret_cast<const device uint*>(act_row);
    const device uint* w2 = reinterpret_cast<const device uint*>(w_row);
    const int pairs = K >> 1;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int p = k0;
    for (; p + 3 * stride < pairs; p += 4 * stride) {
      acc0 += bf16x2_dot(a2[p], w2[p]);
      acc1 += bf16x2_dot(a2[p + stride], w2[p + stride]);
      acc2 += bf16x2_dot(a2[p + 2 * stride], w2[p + 2 * stride]);
      acc3 += bf16x2_dot(a2[p + 3 * stride], w2[p + 3 * stride]);
    }
    for (; p < pairs; p += stride) {
      acc0 += bf16x2_dot(a2[p], w2[p]);
    }
    acc = (acc0 + acc1) + (acc2 + acc3);
  } else {
    for (int k = k0; k < K; k += stride) {
      acc += float(act_row[k]) * float(w_row[k]);
    }
  }
  return acc;
}

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
  float acc = gemv_lane_dot<T>(act_row, w_row, K, int(lane), SIMD_SIZE);
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

PIE_CONST int KSPLIT_GROUP = 128;
PIE_CONST int KSPLIT_SIMDGROUPS = KSPLIT_GROUP / SIMD_SIZE;

template <typename T>
[[kernel]] void dense_gemv_t_ksplit(
    const device T* act      [[buffer(0)]],
    const device T* w        [[buffer(1)]],
    device T* y              [[buffer(2)]],
    const constant int& M    [[buffer(3)]],
    const constant int& N    [[buffer(4)]],
    const constant int& K    [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg    [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[KSPLIT_SIMDGROUPS];
  const int n = int(tgid.x);
  const int m = int(tgid.y);
  if (n >= N || m >= M) {
    return;
  }
  const device T* act_row = act + size_t(m) * size_t(K);
  const device T* w_row = w + size_t(n) * size_t(K);
  float acc = gemv_lane_dot<T>(act_row, w_row, K, int(tid), KSPLIT_GROUP);
  acc = simd_sum(acc);
  if (lane == 0) {
    part[sg] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < KSPLIT_SIMDGROUPS; ++i) {
      total += part[i];
    }
    y[size_t(m) * size_t(N) + size_t(n)] = static_cast<T>(total);
  }
}

#define instantiate_dense_gemv_t_ksplit(name, itype)                         \
  template [[host_name("dense_gemv_t_ksplit_" #name)]]                       \
  [[kernel]] void dense_gemv_t_ksplit<itype>(                                \
      const device itype*, const device itype*, device itype*,               \
      const constant int&, const constant int&, const constant int&,         \
      uint2, uint, uint, uint);

instantiate_dense_gemv_t_ksplit(bfloat16, bfloat)
