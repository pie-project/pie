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
    // EIGHT two-bit codes ride one uint16, at bit offsets 0,2,..,14, and
    // `qdot` reads them UNSHIFTED — code j comes back as `code << 2j`, i.e.
    // multiplied by 4^j. So the activation is pre-divided by 4^j here, the
    // two-bit twin of the nibble point's {1,16,256,4096}. The eight divisors
    // are {1,4,16,64,256,1024,4096,16384}, one per code of the packed word.
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

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(bits == 2 || bits == 4 || bits == 8,
                "port covers the widths mlx affine ships this box");
  U accum = 0;
  if (bits == 2) {
    // Eight codes per uint16 word: two bits apiece, masked in place so the
    // pre-division `load_vector` did cancels the `<< 2j` the mask leaves on.
    // The masks are {0x0003,0x000c,0x0030,0x00c0,0x0300,0x0c00,0x3000,0xc000}.
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 8); i++) {
      accum +=
          (x_thread[8 * i] * (ws[i] & 0x0003) +
           x_thread[8 * i + 1] * (ws[i] & 0x000c) +
           x_thread[8 * i + 2] * (ws[i] & 0x0030) +
           x_thread[8 * i + 3] * (ws[i] & 0x00c0) +
           x_thread[8 * i + 4] * (ws[i] & 0x0300) +
           x_thread[8 * i + 5] * (ws[i] & 0x0c00) +
           x_thread[8 * i + 6] * (ws[i] & 0x3000) +
           x_thread[8 * i + 7] * (ws[i] & 0xc000));
    }
  } else if (bits == 4) {
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

    if (k + int(simd_lid) * values_per_thread < in_vec_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && out_row + row < out_vec_size) {
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
    if (k + int(simd_lid) * values_per_thread < in_vec_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && out_row + row < out_vec_size) {
      T q = static_cast<T>(result[row]);
      y[row] = static_cast<T>(float(q) + float(residual[row]));
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
// The 2-bit affine one-row point, at all three groups. Stamped in source
// beside the 4/8 twins even though `linear::quant::WIDTHS` does not yet list
// two — these are the directly-fired arms the 2-bit unit floor binds, ahead of
// the WIDTHS flip that would put them on the default warm-up ladder. At gs=32
// the point runs packs_per_thread=2 (its fixed default), giving
// values_per_thread=32=group and scale_step_per_thread=1, the tight edge of
// the format and still one whole group a thread.
instantiate_qmv_fast(bfloat16, bfloat, 64, 2)
instantiate_qmv_fast(bfloat16, bfloat, 32, 2)
instantiate_qmv_fast(bfloat16, bfloat, 128, 2)

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
instantiate_qmv_fast_residual(bfloat16, bfloat, 64, 2)
instantiate_qmv_fast_residual(bfloat16, bfloat, 32, 2)
instantiate_qmv_fast_residual(bfloat16, bfloat, 128, 2)

constant float kMxfp4Lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float mxfp4_lo(uint8_t byte) { return kMxfp4Lut[byte & 0xf]; }
inline float mxfp4_hi(uint8_t byte) { return kMxfp4Lut[byte >> 4]; }

inline float mxfp4_block_scale(uint8_t code) {
  return code == 0xff ? NAN : metal::ldexp(1.0f, int(code) - 127);
}
template <typename T, int BITS, int GROUP = 64>
struct AffineQ {
  typedef T scale_t;
  MLX_MTL_CONST int bits = BITS;

  MLX_MTL_CONST int group_size = GROUP;
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

template <typename T> using AffineU4 = AffineQ<T, 4>;
template <typename T> using AffineU8 = AffineQ<T, 8>;
// The two-bit affine codec, for the routed decode arm — now group-parametric,
// because the 2-bit expert banks this box runs are not uniform in their group.
// `AffineQ`'s third parameter is the group, defaulting to 64 so the four- and
// eight-bit twins above are unchanged. Qwen3.8-Flash keeps its 2-bit banks at
// group 128, DeepSeek-V4-Flash at group 32 (with one layer's gate at 64), so
// the routed matvec instantiates the codec at all three groups the artifacts
// carry. The routed impl reads the group straight off `Codec::group_size` for
// every scale/bias index, so the group IS the codec — the `gs` suffix on the
// host name below only names the point, it no longer silently rides 64.
template <typename T> using AffineU2 = AffineQ<T, 2, 64>;
template <typename T> using AffineU2_gs32 = AffineQ<T, 2, 32>;
template <typename T> using AffineU2_gs128 = AffineQ<T, 2, 128>;

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

template <typename T, typename Codec, bool BIASED, bool ROUTED, int PPT>
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
  constexpr int packs_per_thread = PPT;
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

  const device T* x_row = x + row * x_row_stride + slot * x_slot_stride;

  for (int k = 0; k < in_vec_size; k += block_size) {
    const int base = k + int(simd_lid) * values_per_thread;

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
    if (simd_lid == 0 && out_row + row < out_vec_size) {
      if (BIASED) v += U(bias_row[out_row + row]);
      y_row[row] = static_cast<T>(v);
    }
  }
}

#define gptoss_qmv_kernel(name, BIASED, ROUTED, PPT)                                \
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
    qmv_gptoss_impl<T, Codec<T>, BIASED, ROUTED, PPT>(                              \
        w, scales, biases, x, y, bias, expert_ids, in_vec_size, out_vec_size,  \
        x_slot_stride, x_row_stride, slots_per_row, tid, simd_gid, simd_lid);  \
  }

gptoss_qmv_kernel(qmv_tail, false, false, 2)
gptoss_qmv_kernel(qmv_tail_bias, true, false, 2)

gptoss_qmv_kernel(qmv_routed_bias, true, true, 1)
gptoss_qmv_kernel(qmv_routed, false, true, 1)

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

// The 2-bit routed arms — the switch_mlp expert banks the 2-bit checkpoints
// keep in the routed path, at all three groups the artifacts carry.
// `linear::moe::routed_point` names these for a bank that is affine, two bits,
// and one of {32,64,128}. PPT is 1 (the routed rung), so `pack_factor` is 16
// and `values_per_thread` is 16 = two packed uint16 words a thread, which
// `qdot`'s two-bit arm folds eight codes at a time. Sixteen values a thread
// divide 32, 64 and 128 alike, so a thread's span never straddles a group and
// `g = base / group_size` names the one scale/bias the span wants — the routed
// path carries no `scale_step_per_thread`, one group per thread-chunk instead,
// so the gs=32 packs-to-one edge the dense fast point rides does not arise here.
instantiate_gptoss_qmv(affine_qmv_routed_bias, qmv_routed_bias, AffineU2, bfloat16, bfloat, 64, 2)
instantiate_gptoss_qmv(affine_qmv_routed, qmv_routed, AffineU2, bfloat16, bfloat, 64, 2)
instantiate_gptoss_qmv(affine_qmv_routed_bias, qmv_routed_bias, AffineU2_gs32, bfloat16, bfloat, 32, 2)
instantiate_gptoss_qmv(affine_qmv_routed, qmv_routed, AffineU2_gs32, bfloat16, bfloat, 32, 2)
instantiate_gptoss_qmv(affine_qmv_routed_bias, qmv_routed_bias, AffineU2_gs128, bfloat16, bfloat, 128, 2)
instantiate_gptoss_qmv(affine_qmv_routed, qmv_routed, AffineU2_gs128, bfloat16, bfloat, 128, 2)

instantiate_gptoss_qmv(mxfp4_qmv_routed_bias, qmv_routed_bias, Mxfp4, bfloat16, bfloat, 32, 4)
instantiate_gptoss_qmv(mxfp4_qmv_routed, qmv_routed, Mxfp4, bfloat16, bfloat, 32, 4)

instantiate_gptoss_qmv(affine_qmv_tail_bias, qmv_tail_bias, AffineU8, bfloat16, bfloat, 64, 8)

instantiate_gptoss_qmv(affine_qmv_tail, qmv_tail, AffineU8, bfloat16, bfloat, 64, 8)
