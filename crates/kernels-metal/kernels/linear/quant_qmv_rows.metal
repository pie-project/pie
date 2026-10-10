


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

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(bits == 2 || bits == 4 || bits == 8,
                "port covers the widths mlx affine ships this box");
  U sum = 0;
  if (bits == 2) {

    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
      x_thread[i + 4] = x[i + 4] / 256.0f;
      x_thread[i + 5] = x[i + 5] / 1024.0f;
      x_thread[i + 6] = x[i + 6] / 4096.0f;
      x_thread[i + 7] = x[i + 7] / 16384.0f;
    }
  } else if (bits == 4) {
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

template <typename U, int values_per_thread, int bits, int packs>
inline U qdot_staged(
    const thread uint16_t* p,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(bits == 2 || bits == 4 || bits == 8,
                "port covers the widths mlx affine ships this box");
  U accum = 0;
  if (bits == 2) {

    for (int i = 0; i < (values_per_thread / 8); i++) {
      const uint16_t q = p[i];
      accum +=
          (x_thread[8 * i] * (q & 0x0003) + x_thread[8 * i + 1] * (q & 0x000c) +
           x_thread[8 * i + 2] * (q & 0x0030) + x_thread[8 * i + 3] * (q & 0x00c0) +
           x_thread[8 * i + 4] * (q & 0x0300) + x_thread[8 * i + 5] * (q & 0x0c00) +
           x_thread[8 * i + 6] * (q & 0x3000) + x_thread[8 * i + 7] * (q & 0xc000));
    }
  } else if (bits == 4) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      const uint16_t q = p[i];
      accum +=
          (x_thread[4 * i] * (q & 0x000f) + x_thread[4 * i + 1] * (q & 0x00f0) +
           x_thread[4 * i + 2] * (q & 0x0f00) + x_thread[4 * i + 3] * (q & 0xf000));
    }
  } else {
    const thread uint8_t* w = (const thread uint8_t*)p;
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}

template <typename T, typename U, int R, int values_per_thread, int words_per_thread>
METAL_FUNC void qmv_rows_block_unpacked(
    const thread uint16_t (&pack)[4][words_per_thread],
    const thread U (&s)[4],
    const thread U (&b)[4],
    const device T* xb,
    const thread int (&roff)[R],
    thread U (&result)[4][R]) {

  if (words_per_thread * 4 != values_per_thread) {
    return;
  }
  thread U wq[4][values_per_thread];
  #pragma clang loop unroll(full)
  for (int row = 0; row < 4; row++) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < words_per_thread; i++) {
      const uint16_t q = pack[row][i];
      wq[row][4 * i] = U(q & 0x000f);
      wq[row][4 * i + 1] = U((q >> 4) & 0x000f);
      wq[row][4 * i + 2] = U((q >> 8) & 0x000f);
      wq[row][4 * i + 3] = U((q >> 12) & 0x000f);
    }
  }
  #pragma clang loop unroll(full)
  for (int r = 0; r < R; r++) {
    const device T* xr = xb + roff[r];
    thread U xc[values_per_thread];
    U xsum = 0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < values_per_thread; i += 4) {
      xsum += xr[i] + xr[i + 1] + xr[i + 2] + xr[i + 3];
      xc[i] = xr[i];
      xc[i + 1] = xr[i + 1];
      xc[i + 2] = xr[i + 2];
      xc[i + 3] = xr[i + 3];
    }
    #pragma clang loop unroll(full)
    for (int row = 0; row < 4; row++) {
      U accum = 0;
      #pragma clang loop unroll(full)
      for (int i = 0; i < values_per_thread; i += 4) {
        accum += (xc[i] * wq[row][i] + xc[i + 1] * wq[row][i + 1] +
                  xc[i + 2] * wq[row][i + 2] + xc[i + 3] * wq[row][i + 3]);
      }
      result[row][r] += s[row] * accum + xsum * b[row];
    }
  }
}

template <typename T, int group_size, int bits, int rows_per_group, int packs_per_thread_>
METAL_FUNC void qmv_rows_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& ld,
    const constant int& out_vec_size,
    const constant int& row_count,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int R = rows_per_group;
  constexpr int packs_per_thread = packs_per_thread_;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  constexpr int words_per_thread = values_per_thread * bits / 16;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup][R];
  for (int row = 0; row < results_per_simdgroup; row++) {
    #pragma clang loop unroll(full)
    for (int r = 0; r < R; r++) {
      result[row][r] = 0;
    }
  }

  const int in_vec_size_w = ld * bytes_per_pack / pack_factor;
  const int in_vec_size_g = ld / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int row0 = int(tid.x) * R;

  constexpr bool UNPACKED_ROAD = bits == 4 && packs_per_thread == 1 && R == 4;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  const device T* xb = x + simd_lid * values_per_thread;
  int roff[R];
  #pragma clang loop unroll(full)
  for (int r = 0; r < R; r++) {
    roff[r] = min(row0 + r, row_count - 1) * in_vec_size;
  }
  y += row0 * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    if (k + int(simd_lid) * values_per_thread < in_vec_size) {

      thread uint16_t pack[results_per_simdgroup][words_per_thread];
      thread U s[results_per_simdgroup];
      thread U b[results_per_simdgroup];
      for (int row = 0; row < results_per_simdgroup; row++) {
        const device uint16_t* wl =
            (const device uint16_t*)(ws + row * in_vec_size_w);
        for (int i = 0; i < words_per_thread; i++) {
          pack[row][i] = wl[i];
        }
        s[row] = scales[row * in_vec_size_g];
        b[row] = biases[row * in_vec_size_g];
      }
      if (UNPACKED_ROAD) {
        qmv_rows_block_unpacked<T, U, R, values_per_thread, words_per_thread>(
            pack, s, b, xb, roff, result);
      } else {

// Bounding this row loop's unroll dodges a clang/Metal register-allocation
// miscompile: at p=2 (values_per_thread=16, ~28 live result[row][r]
// accumulators) a full unroll of the 7-trip loop corrupts the r=0
// accumulator. This is a back-end register-pressure bug, not source UB --
// unroll_count(4) and a simdgroup_barrier(mem_none) both make it correct.
// The p=1 rungs (R in {2,3}) trip fewer than 4 times, so unroll_count(4)
// still fully unrolls them: that hot path is byte-for-byte unchanged.
#pragma clang loop unroll_count(4)
      for (int r = 0; r < R; r++) {
        U sum = load_vector<T, U, values_per_thread, bits>(xb + roff[r], x_thread);
        for (int row = 0; row < results_per_simdgroup; row++) {
          result[row][r] += qdot_staged<U, values_per_thread, bits, packs_per_thread>(
              pack[row], x_thread, s[row], b[row], sum);
        }
      }
      }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    xb += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int r = 0; r < R; r++) {
      U v = simd_sum(result[row][r]);
      if (simd_lid == 0 && out_row + row < out_vec_size && row0 + r < row_count) {
        y[r * out_vec_size + row] = static_cast<T>(v);
      }
    }
  }
}

template <typename T, int group_size, int bits, int rows_per_group, int packs_per_thread>
[[kernel]] void affine_qmv_rows(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device T* x          [[buffer(3)]],
    device T* y                [[buffer(4)]],
    const constant int& in_vec_size  [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& row_count    [[buffer(7)]],
    const constant int& ld           [[buffer(8)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  qmv_rows_impl<T, group_size, bits, rows_per_group, packs_per_thread>(
      w, scales, biases, x, y, in_vec_size, ld, out_vec_size, row_count, tid,
      simd_gid, simd_lid);
}

#define PIE_STAMP_qmv_rows(entry, gs, b, r, p)                                 \
  template [[host_name(entry)]]                                                \
  [[kernel]] void affine_qmv_rows<bfloat, gs, b, r, p>(                        \
      const device uint32_t*, const device bfloat*, const device bfloat*,      \
      const device bfloat*, device bfloat*, const constant int&,               \
      const constant int&, const constant int&, const constant int&, uint3, uint, uint);
