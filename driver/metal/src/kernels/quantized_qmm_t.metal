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

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
MLX_MTL_CONST int SIMD_SIZE = 32;

// -- steel/utils/type_traits.h --


#include <metal_stdlib>

#pragma METAL internals : enable

namespace metal {

template <typename T>
struct is_empty : metal::bool_constant<__is_empty(T)> {};

#ifdef __cpp_variable_templates
template <typename T>
constexpr constant bool is_empty_v = is_empty<T>::value;
#endif

template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <class T>
struct is_static : metal::bool_constant<is_empty<remove_cv_t<T>>::value> {};

template <typename T>
struct pointer_element {};

template <typename T>
struct pointer_element<thread T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<device T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<constant T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<threadgroup T*> {
  using type = remove_cv_t<T>;
};

template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

} // namespace metal

#pragma METAL internals : disable


// ── steel/defines.h ──



#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")

// ── steel/utils/integral_constant.h ──



#include <metal_stdlib>

#pragma METAL internals : enable

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// Integral constant with casting
///////////////////////////////////////////////////////////////////////////////

template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;

  METAL_FUNC constexpr operator value_type() const noexcept {
    return value;
  }
};

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <class T>
struct is_integral : bool_constant<metal::is_integral<T>::value> {};

template <class T, T v>
struct is_integral<integral_constant<T, v>>
    : bool_constant<metal::is_integral<T>::value> {};

template <typename T>
constexpr constant bool is_integral_v = is_integral<T>::value;

template <int val>
using Int = integral_constant<int, val>;

///////////////////////////////////////////////////////////////////////////////
// Binary Operators on Integral constants
///////////////////////////////////////////////////////////////////////////////

#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    return integral_constant<decltype(res), res>{};         \
  }

integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);

integral_const_binop(==, operator==);
integral_const_binop(!=, operator!=);
integral_const_binop(<, operator<);
integral_const_binop(>, operator>);
integral_const_binop(<=, operator<=);
integral_const_binop(>=, operator>=);

integral_const_binop(&&, operator&&);
integral_const_binop(||, operator||);

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(true_type, T) {
  return true_type{};
}
template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(T, true_type) {
  return true_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(false_type, T) {
  return false_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(T, false_type) {
  return false_type{};
}

// Dispatch utilities
template <typename F>
void dispatch_bool(bool v, F f) {
  if (v) {
    f(true_type{});
  } else {
    f(false_type{});
  }
}

template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    constexpr auto idx = Int<start>{};
    f(idx);
    const_for_loop<start + step, stop, step, F>(f);
  }
}

#undef integral_const_binop

///////////////////////////////////////////////////////////////////////////////
// Reduction operators
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC constexpr T sum(T x) {
  return x;
}

template <typename T, typename... Us>
METAL_FUNC constexpr auto sum(T x, Us... us) {
  return x + sum(us...);
}

} // namespace steel
} // namespace mlx

#pragma METAL internals : disable

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

} // namespace steel
} // namespace mlx

// ── steel/gemm/mma.h ──



#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>


using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <typename T, int kFragRows_, int kFragCols_>
struct BaseMMAFrag {
  static_assert(
      kFragRows_ == 8,
      "Only 8 x 8 fragment matrices are currently supported");
  static_assert(
      kFragCols_ == 8,
      "Only 8 x 8 fragment matrices are currently supported");
};

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  STEEL_CONST int kFragRows = 8;
  STEEL_CONST int kFragCols = 8;

  STEEL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST int kElemRows = 1;
  STEEL_CONST int kElemCols = 2;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;

  METAL_FUNC static constexpr short2 get_coord(
      ushort simd_lane_id [[thread_index_in_simdgroup]]) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }

  template <typename SrcPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void
  load(thread frag_type& dst, SrcPtrType src, StrX str_x, StrY str_y) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] = static_cast<T>(src[i * str_x + j * str_y]);
      }
    }
  }

  template <
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void load_safe(
      thread frag_type& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[(off_x + i) * str_x + (off_y + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename DstPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void
  store(const thread frag_type& src, DstPtrType dst, StrX str_x, StrY str_y) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * str_x + j * str_y] = static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void store_safe(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename StartX,
      typename StopX,
      typename StartY,
      typename StopY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void store_slice(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      StartX start_x,
      StopX stop_x,
      StartY start_y,
      StopY stop_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < stop_x && (off_x + i) >= start_x &&
            (off_y + j) < stop_y && (off_y + j) >= start_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  METAL_FUNC static constexpr void mma(
      thread frag_type& D,
      thread frag_type& A,
      thread frag_type& B,
      thread frag_type& C) {
    mat_type D_mat;
    mat_type A_mat;
    mat_type B_mat;
    mat_type C_mat;

    reinterpret_cast<thread frag_type&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread frag_type&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread frag_type&>(C_mat.thread_elements()) = C;

    mma(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }

  METAL_FUNC static constexpr void mma(
      thread mat_type& D,
      thread mat_type& A,
      thread mat_type& B,
      thread mat_type& C) {
    simdgroup_multiply_accumulate(D, A, B, C);
  }
};

template <
    typename T,
    int kTileRows_,
    int kTileCols_,
    class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  STEEL_CONST int kFragRows = MMAFrag_t::kFragRows;
  STEEL_CONST int kFragCols = MMAFrag_t::kFragCols;
  STEEL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;

  STEEL_CONST int kTileRows = kTileRows_;
  STEEL_CONST int kTileCols = kTileCols_;

  STEEL_CONST int kRows = kTileRows * kFragRows;
  STEEL_CONST int kCols = kTileCols * kFragCols;

  STEEL_CONST int kNumFrags = kTileRows * kTileCols;
  STEEL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[kNumFrags] = {frag_type(0)};

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC mat_type mat_at(const short i, const short j) {
    mat_type val_mat;
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < kElemsPerFrag; ++ii) {
      val_mat.thread_elements()[ii] = frag_at(i, j)[ii];
    }
    return val_mat;
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  METAL_FUNC const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(
                src[(i * kFragRows) * w_x * str_x +
                    (j * kFragCols) * w_y * str_y]),
            Int<str_x>{},
            Int<str_y>{});
      }
    }
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void store(threadgroup U* dst) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(
                dst[(i * kFragRows) * w_x * str_x +
                    (j * kFragCols) * w_y * str_y]),
            Int<str_x>{},
            Int<str_y>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void load(const device U* src, const int ld) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(src[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld,
            Int<1>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld,
            Int<1>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void
  load_safe(const device U* src, const int ld, const short2 src_tile_dims) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load_safe(
            frag_at(i, j),
            src,
            ld,
            Int<1>{},
            src_tile_dims.y,
            src_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void
  store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_safe(
            frag_at(i, j),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store_slice(
      device U* dst,
      const int ld,
      const short2 start,
      const short2 stop) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_slice(
            frag_at(i, j),
            dst,
            ld,
            Int<1>{},
            start.y,
            stop.y,
            start.x,
            stop.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
    }
  }
};

template <typename T, typename U, int M, int N, int K>
METAL_FUNC void tile_matmad(
    thread MMATile<T, M, N>& D,
    thread MMATile<U, M, K>& A,
    thread MMATile<U, K, N>& B,
    thread MMATile<T, M, N>& C) {
  STEEL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short n_serp = (m % 2) ? (N - 1 - n) : n;
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMATile<T, M, N>::MMAFrag_t::mma(
            D.frag_at(m, n_serp),
            A.frag_at(m, k),
            B.frag_at(k, n_serp),
            C.frag_at(m, n_serp));
      }
    }
  }
}


template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    short lda_tgp,
    short ldb_tgp,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
struct BlockMMA {
  // MMAFrag size
  STEEL_CONST short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TM_stride = kFragSize * WM;
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TN_stride = kFragSize * WN;

  // Warp tile size along M
  STEEL_CONST short TM = BM / (kFragSize * WM);
  // Warp tile size along N
  STEEL_CONST short TN = BN / (kFragSize * WN);

  // Threadgroup A strides
  STEEL_CONST short A_str_m = transpose_a ? 1 : lda_tgp; // M
  STEEL_CONST short A_str_k = transpose_a ? lda_tgp : 1; // K

  // Threadgroup B strides
  STEEL_CONST short B_str_k = transpose_b ? 1 : ldb_tgp; // K
  STEEL_CONST short B_str_n = transpose_b ? ldb_tgp : 1; // N

  // Threadgroup strides along K
  STEEL_CONST short tile_stride_a = kFragSize * A_str_k;
  STEEL_CONST short tile_stride_b = kFragSize * B_str_k;

  // Simdgroup matrices
  MMATile<AccumType, TM, 1, MMAFrag_acc_t> Atile;
  MMATile<AccumType, 1, TN, MMAFrag_acc_t> Btile;
  MMATile<AccumType, TM, TN, MMAFrag_acc_t> Ctile;

  // Offsets within threadgroup
  short sm;
  short sn;

  short As_offset;
  short Bs_offset;

  /* Constructor */
  METAL_FUNC BlockMMA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) {
    // Determine thread position in simdgroup matrix
    short tm = kFragSize * (simd_group_id / WN);
    short tn = kFragSize * (simd_group_id % WN);

    short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    sm = simd_coord.y;
    sn = simd_coord.x;

    // Determine thread and simdgroup offset
    As_offset = (tm + sm) * A_str_m + (sn)*A_str_k; // M, K
    Bs_offset = (sm)*B_str_k + (tn + sn) * B_str_n; // K, N

    sm += tm;
    sn += tn;
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    // Adjust for simdgroup and thread location
    As += As_offset;
    Bs += Bs_offset;

    // Iterate over BK in blocks of kFragSize
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += kFragSize) {
      simdgroup_barrier(mem_flags::mem_none);

      Atile.template load<T, WM, 1, A_str_m, A_str_k>(As);

      simdgroup_barrier(mem_flags::mem_none);

      Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Ctile, Atile, Btile, Ctile);

      // Progress to next simdgroup tile
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* D, const int ldd) {
    // Apply epilogue
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;

    Ctile.template store<U, WM, WN>(D, ldd);
  }

  METAL_FUNC void
  store_result_slice(device U* D, const int ldd, short2 start, short2 stop) {
    // Apply epilogue
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    D += sm * ldd + sn;
    start -= short2(sn, sm);
    stop -= short2(sn, sm);

    // TODO: Check the start as well
    if (stop.y <= 0 || stop.x <= 0) {
      return;
    }

    Ctile.template store_slice<U, WM, WN>(D, ldd, start, stop);
  }

  METAL_FUNC void
  store_result_safe(device U* D, const int ldd, short2 dst_tile_dims) {
    // Apply epilogue
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Ctile.template store_safe<U, WM, WN>(D, ldd, dst_tile_dims);
  }

  /* Apply epilogue */
  template <typename UnaryEpilogue>
  METAL_FUNC void apply_epilogue(thread const UnaryEpilogue& epilogue_op) {
    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = epilogue_op.apply(Ctile.elems()[i]);
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int ldc,
      const int fdc,
      thread const BinaryEpilogue& epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Ctile)::kElemsPerFrag; k++) {
          accum[k] = epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
        }
      }
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue_safe(
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const BinaryEpilogue& epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        constexpr short kelems = decltype(Ctile)::kElemsPerFrag;

        // Read C
        U c_elems[kelems] = {0};

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          if ((j * TN_stride + k) < dst_tile_dims.x) {
            c_elems[k] = C[offset_c + k * fdc];
          }
        }

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          accum[k] = epilogue_op.apply(accum[k], c_elems[k]);
        }
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;

    constexpr short kelems = decltype(Ctile)::kElemsPerFrag;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          D[offset_d + k] = epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
        }
      }
    }
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    constexpr short kelems = decltype(Ctile)::kElemsPerFrag;

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = Ctile.frag_at(i, j);
          int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
          int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

          // Apply epilogue
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < kelems; k++) {
            if ((j * TN_stride + k) < dst_tile_dims.x) {
              D[offset_d + k] =
                  epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
            }
          }
        }
      }
    }
  }
};

} // namespace steel
} // namespace mlx

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

} // namespace steel
} // namespace mlx

// ── quantized.h helpers ──

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}


// ── quantized.h dequantize ──
template <typename U, int N, int bits, typename W>
inline void dequantize(const device uint8_t* w, U scale, U bias, W w_local) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  if (bits == 2) {
    U s[4] = {
        scale,
        scale / static_cast<U>(4.0f),
        scale / static_cast<U>(16.0f),
        scale / static_cast<U>(64.0f)};
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i] = s[0] * (w[i] & 0x03) + bias;
      w_local[4 * i + 1] = s[1] * (w[i] & 0x0c) + bias;
      w_local[4 * i + 2] = s[2] * (w[i] & 0x30) + bias;
      w_local[4 * i + 3] = s[3] * (w[i] & 0xc0) + bias;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 3 * i;

      w_local[0] = (w[0] & 0x7) * scale + bias;
      w_local[1] = ((w[0] & 0x38) >> 3) * scale + bias;
      w_local[2] = (((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * scale + bias;
      w_local[3] = ((w[1] & 0xe) >> 1) * scale + bias;
      w_local[4] = ((w[1] & 0x70) >> 4) * scale + bias;
      w_local[5] = (((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * scale + bias;
      w_local[6] = ((w[2] & 0x1c) >> 2) * scale + bias;
      w_local[7] = ((w[2] & 0xe0) >> 5) * scale + bias;
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / static_cast<U>(16.0f)};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = s[0] * (w[i] & 0x0f) + bias;
      w_local[2 * i + 1] = s[1] * (w[i] & 0xf0) + bias;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;

      w_local[0] = (w[0] & 0x1f) * scale + bias;
      w_local[1] = (((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * scale + bias;
      w_local[2] = ((w[1] & 0x7c) >> 2) * scale + bias;
      w_local[3] = (((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * scale + bias;
      w_local[4] = (((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * scale + bias;
      w_local[5] = ((w[3] & 0x3e) >> 1) * scale + bias;
      w_local[6] = (((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * scale + bias;
      w_local[7] = ((w[4] & 0xf8) >> 3) * scale + bias;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = (w[0] & 0x3f) * scale + bias;
      w_local[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
      w_local[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
      w_local[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = scale * w[i] + bias;
    }
  }
}

// ── quantized.h QuantizedBlockLoader ──
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits,
    typename D = T>
struct QuantizedBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup D* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup D* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    D scale = static_cast<D>(*scales);
    D bias = static_cast<D>(*biases);
    for (int i = 0; i < n_reads; i++) {
      dequantize<D, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = D(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = D(0);
      }
      return;
    }

    D scale = static_cast<D>(*scales);
    D bias = static_cast<D>(*biases);
    for (int i = 0; i < n_reads; i++) {
      dequantize<D, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

// ── pie entry ────────────────────────────────────────────────────────────────
// Aligned-only form of MLX's qmm_t_impl: every tile is full, so there is no
// `M`/`num_els`/`num_outs` bookkeeping and no safe-load branch.
template <typename T, int group_size, int bits, int BM, int BK, int BN,
          bool WITH_RESIDUAL, bool WITH_BIAS = false>
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
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = int(tid.y) * BM;
  const int y_col = int(tid.x) * BN;

  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
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
  if (WITH_RESIDUAL) {
    // The projection this replaces folds the block residual into its
    // write-back, so keep that here rather than paying another dispatch.
    residual += y_row * static_cast<int64_t>(N) + y_col;
    mma_op.store_result(y, N);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint idx = simd_gid * 32u + simd_lid; idx < uint(BM * BN);
         idx += uint(WM * WN * SIMD_SIZE)) {
      const int r = int(idx) / BN;
      const int c = int(idx) % BN;
      y[r * static_cast<int64_t>(N) + c] = static_cast<T>(
          float(y[r * static_cast<int64_t>(N) + c]) +
          float(residual[r * static_cast<int64_t>(N) + c]));
    }
  } else if (WITH_BIAS) {
    // A Linear's additive bias, broadcast down the tile's rows. gpt-oss biases
    // every projection, so without this the batched path is a GEMM plus a
    // dispatch that reads and rewrites the whole output to add one vector.
    // `residual` carries the bias here -- same slot, same load, one row of it.
    mma_op.store_result(y, N);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint idx = simd_gid * 32u + simd_lid; idx < uint(BM * BN);
         idx += uint(WM * WN * SIMD_SIZE)) {
      const int r = int(idx) / BN;
      const int c = int(idx) % BN;
      y[r * static_cast<int64_t>(N) + c] = static_cast<T>(
          float(y[r * static_cast<int64_t>(N) + c]) + float(residual[y_col + c]));
    }
  } else {
    mma_op.store_result(y, N);
  }
}

template <typename P, int group_size, int bits, int BM, int BK, int BN>
METAL_FUNC void qmm_t_fp16_precast_impl(
    const device uint32_t* w,
    const device bfloat* scales,
    const device bfloat* biases,
    const device half* x,
    device P* y,
    threadgroup half* Xs,
    threadgroup half* Ws,
    const constant int& K,
    const constant int& k_len,
    const constant int& N,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = BK + 16 / sizeof(half);

  using mma_t = mlx::steel::
      BlockMMA<half, P, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<half, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      bfloat, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits, half>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = int(tid.y) * BM;
  const int y_col = int(tid.x) * BN;

  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
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
  mma_op.store_result(y, N);
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
      w, scales, biases, x, y, Xs, Ws, K, K, N, tid, simd_gid, simd_lid);
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
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  qmm_t_aligned_impl<T, group_size, bits, BM, BK, BN, false, true>(
      w, scales, biases, x, y, bias, Xs, Ws, K, N, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, int BM, int BK, int BN>
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
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  qmm_t_aligned_impl<T, group_size, bits, BM, BK, BN, false>(
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
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  qmm_t_aligned_impl<T, group_size, bits, BM, BK, BN, true>(
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

#define instantiate_qmm_t(gs, bm, bk, bn, b)                                          \
  template [[host_name("affine_qmm_t_routed_bfloat16_gs_" #gs "_b_" #b "_bm_" #bm "_bn_" #bn)]] \
  [[kernel]] void affine_qmm_t_routed<bfloat, gs, b, bm, bk, bn>(              \
      const device uint32_t*, const device bfloat*, const device bfloat*,      \
      const device bfloat*, device bfloat*, const constant int&,               \
      const constant int&, const device int*, uint3, uint, uint);              \
  template [[host_name("affine_qmm_t_bfloat16_gs_" #gs "_b_" #b "_bm_" #bm "_bn_" #bn)]] \
  [[kernel]] void affine_qmm_t_aligned<bfloat, gs, b, bm, bk, bn>(             \
      const device uint32_t*, const device bfloat*, const device bfloat*,      \
      const device bfloat*, device bfloat*, const constant int&,               \
      const constant int&, uint3, uint, uint);                                 \
  template [[host_name("affine_qmm_t_residual_bfloat16_gs_" #gs "_b_" #b "_bm_" #bm "_bn_" #bn)]] \
  [[kernel]] void affine_qmm_t_aligned_residual<bfloat, gs, b, bm, bk, bn>(    \
      const device uint32_t*, const device bfloat*, const device bfloat*,      \
      const device bfloat*, device bfloat*, const constant int&,               \
      const constant int&, const device bfloat*, uint3, uint, uint);           \
  template [[host_name("affine_qmm_t_bias_bfloat16_gs_" #gs "_b_" #b "_bm_" #bm "_bn_" #bn)]] \
  [[kernel]] void affine_qmm_t_aligned_bias<bfloat, gs, b, bm, bk, bn>(        \
      const device uint32_t*, const device bfloat*, const device bfloat*,      \
      const device bfloat*, device bfloat*, const constant int&,               \
      const constant int&, const device bfloat*, uint3, uint, uint);

instantiate_qmm_t(64, 16, 32, 32, 4)
instantiate_qmm_t(32, 16, 32, 32, 4)
instantiate_qmm_t(128, 16, 32, 32, 4)
instantiate_qmm_t(64, 16, 32, 32, 8)
instantiate_qmm_t(32, 16, 32, 32, 8)
instantiate_qmm_t(128, 16, 32, 32, 8)
instantiate_qmm_t(64, 16, 32, 64, 4)
instantiate_qmm_t(32, 16, 32, 64, 4)
instantiate_qmm_t(128, 16, 32, 64, 4)
instantiate_qmm_t(64, 16, 32, 64, 8)
instantiate_qmm_t(32, 16, 32, 64, 8)
instantiate_qmm_t(128, 16, 32, 64, 8)
instantiate_qmm_t(64, 16, 32, 16, 4)
instantiate_qmm_t(32, 16, 32, 16, 4)
instantiate_qmm_t(128, 16, 32, 16, 4)
instantiate_qmm_t(64, 16, 32, 16, 8)
instantiate_qmm_t(32, 16, 32, 16, 8)
instantiate_qmm_t(128, 16, 32, 16, 8)
instantiate_qmm_t(64, 32, 32, 16, 4)
instantiate_qmm_t(32, 32, 32, 16, 4)
instantiate_qmm_t(128, 32, 32, 16, 4)
instantiate_qmm_t(64, 32, 32, 16, 8)
instantiate_qmm_t(32, 32, 32, 16, 8)
instantiate_qmm_t(128, 32, 32, 16, 8)
instantiate_qmm_t(64, 32, 32, 32, 4)
instantiate_qmm_t(32, 32, 32, 32, 4)
instantiate_qmm_t(128, 32, 32, 32, 4)
instantiate_qmm_t(64, 32, 32, 32, 8)
instantiate_qmm_t(32, 32, 32, 32, 8)
instantiate_qmm_t(128, 32, 32, 32, 8)
// The wide row block at the widest column tile. `qmm_bn` takes the widest tile
// that DIVIDES the output, and every projection in these checkpoints is a
// multiple of 64 -- so a 32-row batch asks for exactly this pair. It used to be
// left out and aliased onto the BM=16 pipeline, which is not a crash: the grid
// is built for 32 rows per block and the pipeline computes 16, so HALF the
// batch is never written. At 32 rows gemma4's logits came back all zero.
instantiate_qmm_t(64, 32, 32, 64, 4)
instantiate_qmm_t(32, 32, 32, 64, 4)
instantiate_qmm_t(128, 32, 32, 64, 4)
instantiate_qmm_t(64, 32, 32, 64, 8)
instantiate_qmm_t(32, 32, 32, 64, 8)
instantiate_qmm_t(128, 32, 32, 64, 8)

// The third row block. A prompt is not a 32-row batch: 128 tokens at BM=32
// unpack every weight four times, and the GEMM is memory-bound on exactly that
// unpacking. BM=64 halves it again at no cost in accumulators -- 64x32/128
// lanes is the same sixteen per lane that 32x64 already spends -- and buys 20%
// on a llama-1B prefill. It is a strict addition: `qmm_bm_index` only reaches
// this rung when the batch divides by 64, so nothing that used to run at 32
// moves.
instantiate_qmm_t(64, 64, 32, 16, 4)
instantiate_qmm_t(32, 64, 32, 16, 4)
instantiate_qmm_t(128, 64, 32, 16, 4)
instantiate_qmm_t(64, 64, 32, 16, 8)
instantiate_qmm_t(32, 64, 32, 16, 8)
instantiate_qmm_t(128, 64, 32, 16, 8)
instantiate_qmm_t(64, 64, 32, 32, 4)
instantiate_qmm_t(32, 64, 32, 32, 4)
instantiate_qmm_t(128, 64, 32, 32, 4)
instantiate_qmm_t(64, 64, 32, 32, 8)
instantiate_qmm_t(32, 64, 32, 32, 8)
instantiate_qmm_t(128, 64, 32, 32, 8)
instantiate_qmm_t(64, 64, 32, 64, 4)
instantiate_qmm_t(32, 64, 32, 64, 4)
instantiate_qmm_t(128, 64, 32, 64, 4)
instantiate_qmm_t(64, 64, 32, 64, 8)
instantiate_qmm_t(32, 64, 32, 64, 8)
instantiate_qmm_t(128, 64, 32, 64, 8)

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
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = int(tid.y) * BM;
  const int y_col = int(tid.x) * BN;

  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(row_stride);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(row_stride) + y_col;

  loader_x_t loader_x(x, row_stride, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
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
  mma_op.store_result(y, row_stride);
  if (WITH_RESIDUAL) {
    residual += y_row * static_cast<int64_t>(row_stride) + y_col;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint idx = simd_gid * 32u + simd_lid; idx < uint(BM * BN);
         idx += uint(WM * WN * SIMD_SIZE)) {
      const int r = int(idx) / BN;
      const int c = int(idx) % BN;
      const int64_t off = r * static_cast<int64_t>(row_stride) + c;
      y[off] = static_cast<T>(float(y[off]) + float(residual[off]));
    }
  }
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


// Same as `qmm_t_aligned_impl`, except the K loop runs `k_len` columns from the
// bases the caller advanced, while the row pitches still come from the full K.
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
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, P, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = int(tid.y) * BM;
  const int y_col = int(tid.x) * BN;

  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
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
  mma_op.store_result(y, N);
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
// reduce below sums the slices. Most projections use bf16 partials because the
// side-buffer traffic is otherwise a visible second pass over the output. The
// V projection uses the float instantiation: rounding each partition there
// moved a routed model's next-layer top-k choice in llama_numerics_test.
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
      (const device uint32_t*)wl, scales, biases, x, y, Xs, Ws, K,
      k_partition_size, N, tid, simd_gid, simd_lid);
}

// Sum the split_k slices and write the activation type, folding in the block
// residual the fused projection would otherwise have applied.
template <typename T, typename P, bool WITH_RESIDUAL>
[[kernel]] void qmm_splitk_reduce(
    device T* y                 [[buffer(4)]],
    const constant int& N       [[buffer(6)]],
    const device T* residual    [[buffer(7)]],
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
  if (WITH_RESIDUAL) acc += float(residual[o]);
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
qmm_splitk_reduce<bfloat, bfloat, false>(
    device bfloat*, const constant int&, const device bfloat*, const device bfloat*,
    const constant int&, const constant int&, uint2);
template [[host_name("qmm_splitk_reduce_residual_bfloat16")]] [[kernel]] void
qmm_splitk_reduce<bfloat, bfloat, true>(
    device bfloat*, const constant int&, const device bfloat*, const device bfloat*,
    const constant int&, const constant int&, uint2);
template [[host_name("qmm_splitk_reduce_f32_bfloat16")]] [[kernel]] void
qmm_splitk_reduce<bfloat, float, false>(
    device bfloat*, const constant int&, const device bfloat*, const device float*,
    const constant int&, const constant int&, uint2);
template [[host_name("qmm_splitk_reduce_residual_f32_bfloat16")]] [[kernel]] void
qmm_splitk_reduce<bfloat, float, true>(
    device bfloat*, const constant int&, const device bfloat*, const device float*,
    const constant int&, const constant int&, uint2);
