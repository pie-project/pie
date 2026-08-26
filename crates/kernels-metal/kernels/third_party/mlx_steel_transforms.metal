// Vendored from MLX, MIT licensed. the epilogue transforms and the MXFP4 block loader
//
// Upstream: steel/gemm/transforms.h
//
// Kept verbatim so the math matches by construction. It lives here rather than
// pasted into the kernel that uses it because the driver's shader reader
// (`read_metal_source_at`) splices a quoted include, so a shared definition
// costs a line rather than a copy — which is what
// `quantized_qmm_t.metal` predated and why it grew to 2,970 lines.
//
// Do not edit to fit a caller. A caller that needs different behaviour wraps
// this; that is the difference between vendoring and forking.

#ifndef PIE_MLX_STEEL_TRANSFORMS_METAL
#define PIE_MLX_STEEL_TRANSFORMS_METAL

// ── steel/gemm/transforms.h ──




///////////////////////////////////////////////////////////////////////////////
// Transforms and Epilogues
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <typename OutT, typename InT>
struct TransformNone {
  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT) {
    return static_cast<OutT>(x);
  }
};

template <typename OutT, typename InT>
struct TransformAdd {
  TransformAdd(const float, const float) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT c) {
    return static_cast<OutT>(x) + c;
  }
};

template <typename OutT, typename InT>
struct TransformAxpby {
  const float alpha;
  const float beta;

  TransformAxpby(const float alpha_, const float beta_)
      : alpha(alpha_), beta(beta_) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  METAL_FUNC OutT apply(InT x, OutT c) const {
    return static_cast<OutT>(
        x * static_cast<InT>(alpha) + (static_cast<OutT>(beta) * c));
  }
};

template <typename T>
struct AccumHelper {
  typedef float accum_type;
};

struct BlockSwizzle {
  static METAL_FUNC int2
  swizzle(uint3 tid [[threadgroup_position_in_grid]], const int swizzle_log) {
    const int tid_x = (tid.x) >> swizzle_log;
    const int tid_y =
        ((tid.y) << swizzle_log) + ((tid.x) & ((1 << swizzle_log) - 1));
    return int2(tid_x, tid_y);
  }
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    typename D = T>
struct Mxfp4BlockLoader {
  static_assert(BCOLS == 32, "MXFP4 blocks are exactly 32 values");
  MLX_MTL_CONST short pack_factor = 2;
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;

  const int src_ld;
  const int tile_stride;
  const int group_stride;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup D* dst;
  const device uint8_t* src;
  const device uint8_t* exponents;

  Mxfp4BlockLoader(
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
    for (int i = 0; i < n_reads; ++i) {
      const uint8_t byte = src[i];
      dst[2 * i] = scale * D(mxfp4_lo(byte));
      dst[2 * i + 1] = scale * D(mxfp4_hi(byte));
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

} // namespace steel
} // namespace mlx

#endif
