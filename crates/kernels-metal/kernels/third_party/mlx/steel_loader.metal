// Vendored from MLX, MIT licensed. BlockLoader, the tile stager
//
// Upstream: steel/gemm/loader.h
//
// Kept verbatim so the math matches by construction. It lives here rather than
// pasted into the kernel that uses it because the driver's shader reader
// (`read_metal_source_at`) splices a quoted include, so a shared definition
// costs a line rather than a copy — which is what
// `quantized_qmm_t.metal` predated and why it grew to 2,970 lines.
//
// Do not edit to fit a caller. A caller that needs different behaviour wraps
// this; that is the difference between vendoring and forking.

#ifndef PIE_MLX_STEEL_LOADER_METAL
#define PIE_MLX_STEEL_LOADER_METAL

// ── steel/gemm/loader.h ──




///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short alignment = 1,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoader {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Leading dimension for src
  const int src_ld;
  const int tile_stride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  struct alignas(alignment * sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };

  /* Constructor */
  METAL_FUNC BlockLoader(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  /* Apply operation to threadgroup without bound checking */
  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp& op) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = op.apply(dst[i * dst_ld + j]);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
          *((const device ReadVector*)(&src[i * src_ld]));
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    // Skip loading if thread has no valid reads
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
      return;
    }

    // Use fast thread memory for bound checks
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      // Make sure tmp_idx only contains valid indices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }

      // Read valid indices into tmp_val
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }

      // Zero out unneeded values
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }

      // Copy values to threadgroup memory
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = tmp_val[j];
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    src += tile_stride;
  }
};

/* The x loader, reading one type and staging another.
 *
 * `BlockLoader` above copies whole `ReadVector`s, which is why it cannot
 * convert: the copy is raw bytes. On a machine with no native bfloat matrix
 * path the MMA wants half, and the activations on the device are bfloat, so
 * somebody has to convert. Doing it here costs the vector copy and buys the
 * whole K loop a native matrix instruction -- see
 * `mxfp4_qmm_t_routed_bias_fp16`.
 *
 * Deliberately NOT folded into `BlockLoader` as a degenerate case: the
 * same-type path would lose its `ReadVector` copy, and that path is every
 * other GEMM in this file.
 */
template <
    typename S,
    typename D,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct BlockLoaderCast {
  STEEL_CONST short n_reads = (BCOLS * BROWS) < tgp_size
                                  ? 1
                                  : (BCOLS * BROWS) / tgp_size;
  STEEL_CONST short TCOLS = BCOLS / n_reads;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  const int src_ld;
  const int tile_stride;
  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup D* dst;
  const device S* src;

  METAL_FUNC BlockLoaderCast(
      const device S* src_,
      const int src_ld_,
      threadgroup D* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(n_reads * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < n_reads; j++) {
        dst[i * dst_ld + j] = D(float(src[i * src_ld + j]));
      }
    }
  }

  METAL_FUNC void next() {
    src += tile_stride;
  }
};

} // namespace steel
} // namespace mlx

#endif
