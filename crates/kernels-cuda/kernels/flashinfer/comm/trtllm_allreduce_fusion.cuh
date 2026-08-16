#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif

#include <cuda/std/optional>
// PIE: REMOVED -- `#include <tuple>`, and REPLACED by `<array>` on the line below, which
// is an EDIT rather than a plain removal. No `std::tuple` is named anywhere in this file.
// What upstream actually took from that directive is `std::array`, transitively:
// libstdc++'s `<tuple>` includes `<array>` for the `tuple_size`/`tuple_element`
// specialisations, and `allreduce_fusion_kernel_twoshot_sync`'s parameter list below is
// `std::array<int, NRanks>` twice. `csrc/shim` answers a directive by the literal string
// in it and carries no `tuple`, so the transitive route does not exist here and the
// header that carries the name is named directly. This marker is one a strip does NOT
// undo; see MODIFICATIONS.
#include <array>
#include <type_traits>

// PIE: REMOVED -- `#include "../exception.h"`. That file is deleted from this tree
// (host C++: `flashinfer::Error` derives from `std::exception` and every macro in it
// builds its message in a `std::ostringstream`), and `utils.cuh` records the same
// removal at its own include site. Its `FLASHINFER_CHECK` and `FLASHINFER_ERROR` were
// expanded ONLY inside `allreduce_fusion_kernel_launcher` and `allreduce_fusion_op`,
// both host functions removed at the bottom of this file, so this deletion changes what
// NVRTC sees by nothing at all. This marker is one a strip does NOT undo.
#include "../fp4_layout.cuh"
// PIE: REMOVED -- `#include "../logging.h"`. 100% host C++ and not even portable host
// C++: it is six `#define`s over `spdlog::` free functions plus a `set_log_level` built
// on `std::make_shared<spdlog::sinks::stdout_color_sink_mt>`, and `spdlog` is a
// third-party library this repository does not carry at all. Nothing in this file
// expands a `FLASHINFER_LOG_*` macro. This marker is one a strip does NOT undo; see
// MODIFICATIONS.
#include "../utils.cuh"
#include "../vec_dtypes.cuh"

namespace flashinfer {

namespace trtllm_allreduce_fusion {

using flashinfer::QuantizationSFLayout;

namespace details {

static constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
static constexpr int CVT_FP4_SF_VEC_SIZE = 16;
static constexpr int kBytesPerAccess = 16;
static constexpr int kOneShotMaxToken = 128;
static constexpr int kBarrierFlagCount = 256;

}  // namespace details

namespace maths {
// // ============================== Cast ==============================
template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <>
__device__ inline float2 cuda_cast<float2, int2>(int2 val) {
  return make_float2(val.x, val.y);
}

template <>
__device__ inline float2 cuda_cast<float2, float>(float val) {
  return make_float2(val, val);
}

template <>
__device__ inline float2 cuda_cast<float2, half2>(half2 val) {
  return __half22float2(val);
}

template <>
__device__ inline half2 cuda_cast<half2, float2>(float2 val) {
  return __float22half2_rn(val);
}

template <>
__device__ inline half2 cuda_cast<half2, float>(float val) {
  return __float2half2_rn(val);
}

template <>
__device__ inline half2 cuda_cast<half2, half>(half val) {
  return __half2half2(val);
}

// PIE: REMOVED -- `cuda_cast<int8_t, half>` and `cuda_cast<int16_t, half2>`,
// and with them the other four members of the integer<->half-precision cast
// family marked below. **This is the first removal in this tree that takes
// DEVICE text**, so it needs the whole argument rather than a line, and
// MODIFICATIONS carries it under a heading of its own.
//
// What they are: `cuda_cast` specialisations that convert between `int8_t`/
// `int16_t` and `half`/`half2`/`__nv_bfloat16`/`__nv_bfloat162`. TensorRT-LLM's
// SmoothQuant heritage, carried into this header with the rest of
// `namespace maths`.
//
// **Nothing in this file calls any of them.** Measured: the only consumers of
// `maths::` anywhere in this header are four lines inside
// `utils::cvt_warp_fp16_to_fp4`, and all four name `cuda_abs` and `cuda_max`.
// Every reference to the six removed specialisations is one of them calling
// another -- `cuda_cast<int16_t, half2>` calls `cuda_cast<int8_t, half>`,
// `cuda_cast<__nv_bfloat162, int16_t>` calls `cuda_cast<__nv_bfloat16>(int8_t)`,
// `cuda_cast<int16_t, __nv_bfloat162>` calls `bf1622int16`. The family is a
// closed component of the call graph with no edge into it.
//
// **Why they cannot stay.** Every one of them needs an IMPLICIT conversion
// between a built-in integer or floating type and the half-precision type --
// `make_half2(int8[0], int8[1])`, `return static_cast<float>(val);` from a
// function returning `__nv_bfloat16`, `static_cast<short>(val.x)` on a
// `__nv_bfloat16`. NVIDIA's `__half` and `__nv_bfloat16` are classes with a
// converting constructor per arithmetic type and they compile. Ours are
// `pie_cuda_driver::kernels::device::f16` and `::bf16`, whose every
// constructor and conversion operator is `explicit` ON PURPOSE:
// `pie_device.cuh:71-83` states the rule (`bf16 b = 5;` and `bf16 b = 1.0f;`
// stay refused) and MODIFICATIONS' "THE EDIT THAT IS NOT A REMOVAL" is a
// second, independent record of that explicitness being load-bearing --
// `xqa/mha.cuh:1455` had to change because of it.
//
// So the choice was: relax the prelude's canonical device types for six dead
// functions, or remove the six. Relaxing would reach every FA2 and XQA
// instantiation in the crate and turn a class of narrowing bugs from compile
// errors into silent conversions. `csrc/shim/README.md`'s rule points the
// same way -- "an untested conversion is a wrong answer that compiles".
//
// What a rebase does: put all six back and they will not compile until the
// prelude is widened. That is the correct failure. This marker is one a strip
// does NOT undo; see MODIFICATIONS.
template <>
__device__ inline int8_t cuda_cast<int8_t, float>(float val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };

  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
  return int8[0];
}

template <>
__device__ inline int16_t cuda_cast<int16_t, float2>(float2 val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };

  int8[0] = cuda_cast<int8_t>(val.x);
  int8[1] = cuda_cast<int8_t>(val.y);
  return int16;
}

// PIE: REMOVED -- `cuda_cast<half2, int16_t>`. Part of the integer<->half-precision `cuda_cast`
// family; the whole family's removal is argued at the FIRST marker of this
// kind above (the `cuda_cast<int8_t, half>` one). Nothing in this file calls
// it. This marker is one a strip does NOT undo; see MODIFICATIONS.
template <>
__device__ inline float2 cuda_cast<float2, int16_t>(int16_t val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };

  int16 = val;
  return make_float2(int8[0], int8[1]);
}

// PIE: REMOVED -- `cuda_cast<__nv_bfloat16>(int32_t)`, `cuda_cast<__nv_bfloat16>(int8_t)`
// and `cuda_cast<int8_t>(__nv_bfloat16)`. Part of the integer<->half-precision `cuda_cast`
// family; the whole family's removal is argued at the FIRST marker of this
// kind above (the `cuda_cast<int8_t, half>` one). Nothing in this file calls
// it. This marker is one a strip does NOT undo; see MODIFICATIONS.
template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = __low2float(val);
  f_val.y = __high2float(val);
  return f_val;
#else
  return __bfloat1622float2(val);
#endif
}

template <>
__device__ inline float2 cuda_cast<float2, __nv_bfloat162>(__nv_bfloat162 val) {
  return bf1622float2(val);
}

template <>
__device__ inline half cuda_cast<half, __nv_bfloat16>(__nv_bfloat16 val) {
  return __float2half(__bfloat162float(val));
}

// PIE: REMOVED -- `bf1622int16` and `cuda_cast<int16_t, __nv_bfloat162>`. Part of the integer<->half-precision `cuda_cast`
// family; the whole family's removal is argued at the FIRST marker of this
// kind above (the `cuda_cast<int8_t, half>` one). Nothing in this file calls
// it. This marker is one a strip does NOT undo; see MODIFICATIONS.

template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float val) {
  return __float2bfloat16(val);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, half>(half val) {
  return __float2bfloat16(__half2float(val));
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __nv_bfloat162 val2;
  val2.x = val;
  val2.y = val;
  return val2;
#else
  return __bfloat162bfloat162(val);
#endif
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, __nv_bfloat16>(__nv_bfloat16 val) {
  return bf162bf162(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float>(float val) {
  return __float2bfloat162_rn(val);
}

inline __device__ __nv_bfloat162 float22bf162(const float2 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __floats2bfloat162_rn(val.x, val.y);
#else
  return __float22bfloat162_rn(val);
#endif
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float2>(float2 val) {
  return float22bf162(val);
}

// PIE: REMOVED -- `cuda_cast<__nv_bfloat162, int16_t>`. Part of the integer<->half-precision `cuda_cast`
// family; the whole family's removal is argued at the FIRST marker of this
// kind above (the `cuda_cast<int8_t, half>` one). Nothing in this file calls
// it. This marker is one a strip does NOT undo; see MODIFICATIONS.

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, half2>(half2 val) {
  return float22bf162(__half22float2(val));
}

// // ============================== Abs ==============================
template <typename T>
__device__ inline T cuda_abs(T val) {
  assert(false);
  return {};
}

template <>
__device__ inline float cuda_abs(float val) {
  return fabs(val);
}

template <>
__device__ inline float2 cuda_abs(float2 val) {
  return make_float2(fabs(val.x), fabs(val.y));
}

template <>
__device__ inline half cuda_abs(half val) {
  return __habs(val);
}

template <>
__device__ inline half2 cuda_abs(half2 val) {
  return __habs2(val);
}

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ inline __nv_bfloat16 cuda_abs(__nv_bfloat16 val) {
  return __habs(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_abs(__nv_bfloat162 val) {
  return __habs2(val);
}
#endif

// // ============================== Max ==============================
template <typename To, typename Ti>
__device__ inline To cuda_max(Ti val) {
  return cuda_cast<To>(val);
};

template <>
__device__ inline float cuda_max(float2 val) {
  return fmaxf(val.x, val.y);
}

template <>
__device__ inline half cuda_max(half2 val) {
  return __hmax(val.x, val.y);
}

template <>
__device__ inline __nv_bfloat16 cuda_max(__nv_bfloat162 val) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  return __hmax(val.x, val.y);
#else
  assert(0);
  asm volatile("brkpt;\n" ::);
  return __nv_bfloat16(0);
#endif
}

// Binary maximum: compute the max of two values.
template <typename T>
__device__ inline T cuda_max(T val1, T val2) {
  return (val1 > val2) ? val1 : val2;
}

template <>
__device__ inline float2 cuda_max(float2 val1, float2 val2) {
  float2 out;
  out.x = fmaxf(val1.x, val2.x);
  out.y = fmaxf(val1.y, val2.y);
  return out;
}

template <>
__device__ inline half2 cuda_max(half2 val1, half2 val2) {
  return __hmax2(val1, val2);
}

template <>
__device__ inline __nv_bfloat162 cuda_max(__nv_bfloat162 val1, __nv_bfloat162 val2) {
  return __hmax2(val1, val2);
}

// // ============================== Reciprocal ==============================
// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}
}  // namespace maths

namespace utils {

#define FINAL_MASK 0xffffffff

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

// PIE: REMOVED -- `inline int getSMVersion()` and `inline int getSMRegisters()`.
// 20 lines of host C++: `cudaGetDevice` and `cudaDeviceGetAttribute` behind
// `FLASHINFER_CUDA_CALL`, neither with a `__device__` qualifier and neither callable
// from one. Their only callers were `allreduce_fusion_kernel_launcher` (removed at the
// bottom of this file), and under the JIT both questions are Rust's:
// `jit::Ctx::compute_capability_major` answers the first and
// `kernels_cuda::comm::CLUSTER_SIZE`'s note records why the second is not asked at all.
// This marker is one a strip does NOT undo; see MODIFICATIONS.

inline __device__ int64_t get_sf_out_offset_128x4(cuda::std::optional<int> batchIdx, int mIdx,
                                                  int kIdx, cuda::std::optional<int> numRows,
                                                  int numCols) {
  // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

  // batched tensor
  // SF layout [numBTiles, numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  // --> index [bTileIdx, mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

  int32_t innerKIdx = (kIdx % 4);
  int64_t innerKStride = 1;

  int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
  int64_t innerMStride = 4 * innerKStride;  // 4

  // M tile layout [32, 4] is column-major.
  int32_t outerMIdx = (mIdx % 32);
  int64_t outerMStride = 4 * innerMStride;  // 16

  int32_t kTileIdx = (kIdx / 4);
  int64_t kTileStride = 32 * outerMStride;  // 512

  // SF vector size 16. We round the "numCols" up to a multiple of 64.
  int factor = details::CVT_FP4_SF_VEC_SIZE * 4;
  int32_t numKTiles = (numCols + factor - 1) / factor;
  int32_t mTileIdx = mIdx / (32 * 4);
  int64_t mTileStride = numKTiles * kTileStride;

  // Each SF block has 128 rows so pad rows to the multiple of 128.
  int32_t numMTiles = (numRows.value_or(0) + 128 - 1) / 128;
  int64_t bTileStride = numMTiles * mTileStride;

  // Compute the global offset.
  int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride +
                     kTileIdx * kTileStride + outerMIdx * outerMStride + innerMIdx * innerMStride +
                     innerKIdx * innerKStride;

  return SFOffset;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(cuda::std::optional<int> batchIdx,
                                                       int rowIdx, int colIdx,
                                                       cuda::std::optional<int> numRows,
                                                       int numCols, SFType* SFout,
                                                       QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    if (layout == QuantizationSFLayout::SWIZZLED_128x4) {
      // SF vector index (16 elements share one SF in the K dimension).
      // numRows and numCols are unpadded.
      int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
      int32_t mIdx = rowIdx;

      auto SFOffset = get_sf_out_offset_128x4(batchIdx, mIdx, kIdx, numRows, numCols);
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else if (layout == QuantizationSFLayout::LINEAR) {
      // Linear row-major layout, no padding required.
      int32_t KTileIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;

      int32_t numKTiles = numCols / details::CVT_FP4_SF_VEC_SIZE;
      int64_t mTileStride = numKTiles;

      int64_t BTileStride = numRows.value_or(0) * mTileStride;

      int64_t SFOffset = batchIdx.value_or(0) * BTileStride + rowIdx * mTileStride + KTileIdx;
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else {
      return nullptr;
    }
  }
#endif
  return nullptr;
}

__forceinline__ __device__ uint32_t pack_bytes(uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3) {
  uint32_t val0 = c0;
  uint32_t val1 = c1;
  uint32_t val2 = c2;
  uint32_t val3 = c3;

  return (val3 << 24) | (val2 << 16) | (val1 << 8) | val0;
}

// Convert single float2 pair to e2m1 (2 float32 -> 2 e2m1, returns uint8_t)
// Optimization: allows pipelined processing to reduce register usage
// Note: "=r" constraint always allocates 32-bit register regardless of variable type
inline __device__ uint8_t fp32_pair_to_e2m1(float2 pair) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val32;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "mov.b32 %0, {byte0, 0, 0, 0};\n"
      "}"
      : "=r"(val32)
      : "f"(pair.x), "f"(pair.y));
  return static_cast<uint8_t>(val32 & 0xFF);  // Extract low 8 bits
#else
  return 0;
#endif
}

#if CUDA_VERSION >= 12080
// PIE: REMOVED -- both `fp32_vec_to_e2m1` overloads, `float (&)[8]` and
// `float2 (&)[4]`. 64 lines of DEVICE text, and the second removal of that
// kind in this file; the `cuda_cast` marker above carries the general
// argument and MODIFICATIONS the heading. This one is removed for a different
// reason and it is a decision this repository had already taken.
//
// Each overload is two bodies. Above sm_100 it is
// `cvt.rn.satfinite.e2m1x2.f32` inline PTX; below sm_100 it is four calls to
// `__nv_cvt_float2_to_fp4x2`, NVIDIA's software emulation of that
// instruction. `csrc/shim/cuda_fp4.h` answers `<cuda_fp4.h>` here, and it
// carries the fp4 STORAGE types and the `__NV_E2M1` enumerator and
// deliberately no conversion at all. Its banner states why, and the sentence
// is about exactly this situation: *"If a Blackwell path we never instantiate
// is one day switched on and reaches `__nv_fp4_e2m1(x)`, the build stops on a
// missing constructor -- which is the correct moment to decide what that
// conversion should do, on hardware that can be measured."*
//
// This is that build stopping. Supplying `__nv_cvt_float2_to_fp4x2` means
// writing a round-to-nearest-even over the eight E2M1 magnitudes
// {0, .5, 1, 1.5, 2, 3, 4, 6} with a saturating NaN rule, on a box whose one
// GPU is an sm_89 L40S that cannot execute the instruction the emulation is
// emulating -- so nothing here could check it against anything, and a
// conversion checked against nothing is the failure `csrc/shim/README.md`
// names in one line: *"an untested conversion is a wrong answer that
// compiles."*
//
// It costs nothing today. **Neither overload has a caller anywhere in this
// file** -- the FP4 epilogue reaches `fp32_pair_to_e2m1` (which is above and
// stands, because its sub-sm_100 body is `return 0;` and needs no shim), not
// `fp32_vec_to_e2m1` -- and `comm::INSTANTIATED` names one pattern,
// `kARResidualRMSNorm`, whose `GetQuantType` is `kNone`. The whole FP4 arm is
// unreachable from every instantiation this tree compiles.
//
// What a rebase does: put both back, and they compile the moment
// `csrc/shim/cuda_fp4.h` grows a measured `__nv_cvt_float2_to_fp4x2`. That is
// the right order. This marker is one a strip does NOT undo; see
// MODIFICATIONS.

// Quantizes the provided PackedVec into the uint32_t output
template <typename T, uint32_t VEC_SIZE, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(vec_t<T, VEC_SIZE>& vec, float SFScaleVal,
                                         uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Pre-compute constant: reciprocal of 6.0 (maximum value of e2m1)
  static constexpr float RECIPROCAL_6 = 1.0f / 6.0f;

  // Get absolute maximum values among the local 8 values.
  auto localMax = maths::cuda_abs(get_vec2_element(vec, 0));

#pragma unroll
  for (int i = 1; i < details::CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = maths::cuda_max(localMax, maths::cuda_abs(get_vec2_element(vec, i)));
  }

  // Get the absolute maximum among all 16 values (two threads).
  localMax = maths::cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // Get the final absolute maximum values.
  // Optimization: compute vecMax and reuse localMax space (localMax no longer needed)
  float vecMax = float(maths::cuda_max(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of e2m1).
  // maximum value of e2m1 = 6.0.
  // Optimization: compute quantized SF directly, avoid storing intermediate SFValue
  uint8_t fp8SFVal;
  float quantized_sf;
  if constexpr (UE8M0_SF) {
#if (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 12080)
    __nv_fp8_e8m0 tmp;
    float sf_value = SFScaleVal * (vecMax * RECIPROCAL_6);
    tmp.__x = __nv_cvt_float_to_e8m0(sf_value, __NV_SATFINITE, cudaRoundPosInf);
    quantized_sf = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
#else
#error "FP8 E8M0 support requires CUDA 12.8 or newer."
#endif
  } else {
    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFScaleVal * (vecMax * RECIPROCAL_6));
    fp8SFVal = tmp.__x;
    quantized_sf = static_cast<float>(tmp);
  }
  // Get the output scale directly (optimization: avoid storing intermediate SFValue)
  // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) * reciprocal(SFScaleVal))
  // Optimization: mathematically equivalent to SFScaleVal / quantized_sf, but more efficient
  // (reduces 1 reciprocal call and 1 multiply operation)
  float outputScale = quantized_sf != 0 ? SFScaleVal / quantized_sf : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float and quantize (pipelined to reduce register usage).
  // Optimization: use single float2 instead of array to reduce register pressure from 32 bytes to 8
  // bytes
  uint32_t e2m1Vec = 0;

#pragma unroll
  for (int i = 0; i < details::CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // Reuse single float2 register instead of array
    float2 fp2Val;
    if constexpr (std::is_same_v<T, half>) {
      fp2Val = __half22float2(get_vec2_element(vec, i));
    } else {
      fp2Val = __bfloat1622float2(get_vec2_element(vec, i));
    }
    fp2Val.x *= outputScale;
    fp2Val.y *= outputScale;

    // Convert pair immediately and pack into result
    uint8_t e2m1Pair = fp32_pair_to_e2m1(fp2Val);
    e2m1Vec |= (static_cast<uint32_t>(e2m1Pair) << (i * 8));
  }

  // Write the e2m1 values to global memory.
  return e2m1Vec;
#else
  return 0;
#endif
}

#endif

}  // namespace utils

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ vec_t<T, VEC_SIZE> vec_add(const vec_t<T, VEC_SIZE>& a,
                                                      const vec_t<T, VEC_SIZE>& b) {
  vec_t<T, VEC_SIZE> ret;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    ret[i] = static_cast<float>(a[i]) + static_cast<float>(b[i]);
  }
  return ret;
}

enum class AllReduceFusionPattern : int {
  kAllReduce = 0,
  kARResidualRMSNorm = 1,
  kARResidualRMSNormFP8Quant = 2,
  kARResidualRMSNormFP4Quant = 3,
  // The difference between these two and the standard version is that the NormOut version outputs
  // the result of the norm.
  kARResidualRMSNormOutFP8Quant = 4,
  kARResidualRMSNormOutFP4Quant = 5,
  // Per-token-group FP8 quantization with UE8M0 packed scales
  kARResidualRMSNormPerTokenGroupFP8PackedQuant = 8,
  // Same as above but also outputs the norm result
  kARResidualRMSNormOutPerTokenGroupFP8PackedQuant = 9,
};

enum class QuantType : int {
  kNone = 0,
  kFP8 = 1,
  kFP4 = 2,
  kPerTokenGroupFP8Packed = 3,  // Per-token-group FP8 with dynamic UE8M0 scales
};

template <AllReduceFusionPattern Pattern>
struct FusionPatternTraits;

#define DEFINE_FUSION_PATTERN_TRAITS(pattern, hasAllReduceOut, hasResidual, hasResidualOut, \
                                     hasRMSNorm, hasNormOut, quantType)                     \
  template <>                                                                               \
  struct FusionPatternTraits<pattern> {                                                     \
    static constexpr bool kHasAllReduceOut = hasAllReduceOut;                               \
    static constexpr bool kHasResidual = hasResidual;                                       \
    static constexpr bool kHasResidualOut = hasResidualOut;                                 \
    static constexpr bool kHasRMSNorm = hasRMSNorm;                                         \
    static constexpr bool kHasNormOut = hasNormOut;                                         \
    static constexpr QuantType kQuantType = quantType;                                      \
  };

DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kAllReduce, true, false, false, false, false,
                             QuantType::kNone);
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNorm, false, true, true, true,
                             true, QuantType::kNone);
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNormFP8Quant, false, true, true,
                             true, false, QuantType::kFP8);
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNormFP4Quant, false, true, true,
                             true, false, QuantType::kFP4);
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant, false, true,
                             true, true, true, QuantType::kFP8);
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant, false, true,
                             true, true, true, QuantType::kFP4);
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNormPerTokenGroupFP8PackedQuant,
                             false, true, true, true, false, QuantType::kPerTokenGroupFP8Packed);
DEFINE_FUSION_PATTERN_TRAITS(
    AllReduceFusionPattern::kARResidualRMSNormOutPerTokenGroupFP8PackedQuant, false, true, true,
    true, true, QuantType::kPerTokenGroupFP8Packed);
#undef DEFINE_FUSION_PATTERN_TRAITS

template <AllReduceFusionPattern Pattern>
constexpr bool HasResidual = FusionPatternTraits<Pattern>::kHasResidual;
template <AllReduceFusionPattern Pattern>
constexpr bool HasRMSNorm = FusionPatternTraits<Pattern>::kHasRMSNorm;
template <AllReduceFusionPattern Pattern>
constexpr bool HasAllReduceOut = FusionPatternTraits<Pattern>::kHasAllReduceOut;
template <AllReduceFusionPattern Pattern>
constexpr bool HasResidualOut = FusionPatternTraits<Pattern>::kHasResidualOut;
template <AllReduceFusionPattern Pattern>
constexpr bool HasNormOut = FusionPatternTraits<Pattern>::kHasNormOut;
template <AllReduceFusionPattern Pattern>
constexpr QuantType GetQuantType = FusionPatternTraits<Pattern>::kQuantType;

template <typename T>
struct AllReduceFusionParams {
  int nranks;
  int rank;
  int size;
  int hidden_dim;
  void** workspace;
  void* allreduce_in;
  void* allreduce_out;
  void* residual_in;
  void* residual_out;
  void* norm_out;
  void* quant_out;
  void* scale_out;
  void* rms_gamma;
  float rms_eps;
  // 0 for standard RMSNorm (out = gamma * x * rsqrt(...)),
  // 1 for Gemma / Qwen3.5 (out = (1 + gamma) * x * rsqrt(...)).
  float weight_bias = 0.f;
  float* scale_factor;
  bool use_oneshot;
  QuantizationSFLayout layout = QuantizationSFLayout::SWIZZLED_128x4;
  cudaStream_t stream;
  AllReduceFusionPattern pattern;
  bool trigger_completion_at_end = true;
  int block_quant_group_size = 0;
  int tma_aligned_mn = 0;
};

template <int NRanks>
struct SyncComm {
  __device__ __forceinline__ SyncComm(void** workspace) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[1];
    flag_value = *flag_ptr;
    for (int r = 0; r < NRanks; ++r) {
      comm_bufs[r] = workspace[r];
      barrier_flags[r] = workspace[NRanks + r];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_flag_value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = new_flag_value;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  void* comm_bufs[NRanks];
  void* barrier_flags[NRanks];
  int flag_value;
};

template <int NRanks>
struct LamportComm {
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
    clear_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[4];
    flag_value = *flag_ptr;
    int comm_size = reinterpret_cast<int*>(workspace[NRanks * 3])[3];
    clear_size = *clear_ptr;
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) +
                     static_cast<int64_t>(data_offset) * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int clear_size;
  int flag_value;
};

template <int NRanks>
class Barrier {
 public:
  __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const& comm) {
    if (threadIdx.x < NRanks) {
      m_flag_value = comm.flag_value;
      int current_rank = rank;
      int target_rank = threadIdx.x;
      m_target_flag = reinterpret_cast<int*>(comm.barrier_flags[target_rank]) + current_rank;
      m_current_flag = reinterpret_cast<int*>(comm.barrier_flags[current_rank]) +
                       blockIdx.x * NRanks + target_rank;
    }
  }

  __device__ __forceinline__ void sync() {
    __syncthreads();
    if (threadIdx.x < NRanks) {
      m_flag_value = next_flag(m_flag_value);
      // To avoid the ABA problem, we need to synchronize the correct flag value to all
      // barrier_flags, even if the corresponding CTA has not been launched.
      for (int flag_idx = blockIdx.x; flag_idx < details::kBarrierFlagCount;
           flag_idx += gridDim.x) {
        st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
      }
      while (ld_flag(m_current_flag) == prev_flag(m_flag_value)) {
      }
    }
    __syncthreads();
  }

 protected:
  __device__ __forceinline__ void st_flag(int* addr, int flag) {
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
  }

  __device__ __forceinline__ int ld_flag(int* addr) {
    int flag;
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(addr));
    return flag;
  }

  __device__ __forceinline__ int next_flag(int flag) { return flag == 2 ? 0 : flag + 1; }

  __device__ __forceinline__ int prev_flag(int flag) { return flag == 0 ? 2 : flag - 1; }

 public:
  int m_flag_value;

 private:
  int* m_target_flag;
  int* m_current_flag;
};

template <AllReduceFusionPattern Pattern, typename T>
class FusedOp {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

 public:
  __device__ __forceinline__ FusedOp(AllReduceFusionParams<T> const& params, int access_id,
                                     int access_id_in_token)
      : m_params(params), m_access_id(access_id), m_access_id_in_token(access_id_in_token) {
    if constexpr (HasRMSNorm<Pattern>) {
      m_gamma_val.load(reinterpret_cast<T*>(params.rms_gamma) + m_access_id_in_token * VEC_SIZE);
    }
    if constexpr (HasResidual<Pattern>) {
      m_residual_val.load(reinterpret_cast<T*>(params.residual_in) + m_access_id * VEC_SIZE);
    }
    if constexpr (GetQuantType<Pattern> == QuantType::kFP8) {
      m_scale_factor = 1.f / *(params.scale_factor);
    } else if constexpr (GetQuantType<Pattern> == QuantType::kFP4) {
      m_scale_factor = *(params.scale_factor);
    }
  }

  // template <typename T>
  __device__ __forceinline__ void update(int access_id) {
    if (m_access_id != access_id) {
      m_access_id = access_id;
      if constexpr (HasResidual<Pattern>) {
        m_residual_val.load(reinterpret_cast<T*>(m_params.residual_in) + m_access_id * VEC_SIZE);
      }
    }
  }

  // template <typename T, uint32_t VEC_SIZE>
  __device__ __forceinline__ void operator()(vec_t<T, VEC_SIZE> val, int token_id) {
    if constexpr (HasAllReduceOut<Pattern>) {
      val.store(reinterpret_cast<T*>(m_params.allreduce_out) + m_access_id * VEC_SIZE);
    }
    if constexpr (HasResidual<Pattern>) {
      val = vec_add<T, VEC_SIZE>(val, m_residual_val);
      if constexpr (HasResidualOut<Pattern>) {
        val.store(reinterpret_cast<T*>(m_params.residual_out) + m_access_id * VEC_SIZE);
      }
    }
    if constexpr (HasRMSNorm<Pattern>) {
      val = rms_norm(val, m_gamma_val);
      if constexpr (HasNormOut<Pattern>) {
        val.store(reinterpret_cast<T*>(m_params.norm_out) + m_access_id * VEC_SIZE);
      }
    }

#if CUDA_VERSION >= 12080
    if constexpr (GetQuantType<Pattern> == QuantType::kFP4) {
      // NOTE(Yingyi): might update later
      auto sf_out = utils::cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 2>(
          cuda::std::nullopt /* batchIdx */, token_id, m_access_id_in_token,
          cuda::std::nullopt /* numRows */, m_params.hidden_dim,
          reinterpret_cast<uint32_t*>(m_params.scale_out), m_params.layout);
      reinterpret_cast<uint32_t*>(m_params.quant_out)[m_access_id] =
          utils::cvt_warp_fp16_to_fp4<T, VEC_SIZE>(val, m_scale_factor, sf_out);
    } else
#endif
        if constexpr (GetQuantType<Pattern> == QuantType::kFP8) {
      using PackedQuantizedType = std::conditional_t<std::is_same_v<T, float>, float, float2>;
      PackedQuantizedType ret;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        reinterpret_cast<__nv_fp8_e4m3*>(&ret)[i] = static_cast<__nv_fp8_e4m3>(
            static_cast<float>(reinterpret_cast<T*>(&val)[i]) * m_scale_factor);
      }
      reinterpret_cast<PackedQuantizedType*>(m_params.quant_out)[m_access_id] = ret;
    } else if constexpr (GetQuantType<Pattern> == QuantType::kPerTokenGroupFP8Packed) {
      // Per-token-group FP8 quantization with UE8M0 packed scales.
      constexpr float FP8_E4M3_MAX = 448.0f;
      int group_size = m_params.block_quant_group_size;
      int groups_in_block = blockDim.x * VEC_SIZE / group_size;
      int block_elem_start = threadIdx.x * VEC_SIZE;

      // --- Group absmax reduction ---
      // use warp-shuffle reduce when group fits cleanly in a warp
      // (group_size divisible by VEC_SIZE, group_size_in_vecs is power of 2 and <= 32).
      // otherwise use shared-memory atomicMax
      int group_size_in_vecs = group_size / VEC_SIZE;
      bool use_warp_shuffle = (group_size % VEC_SIZE == 0) && (group_size_in_vecs <= 32) &&
                              (group_size_in_vecs & (group_size_in_vecs - 1)) == 0;

      // per-element group absmax
      float elem_group_absmax[VEC_SIZE];
      extern __shared__ unsigned int smem_group_absmax[];

      if (use_warp_shuffle) {
        float local_absmax = 0.0f;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          float v = fabsf(static_cast<float>(reinterpret_cast<T*>(&val)[i]));
          local_absmax = fmaxf(local_absmax, v);
        }
        // Butterfly all-reduce within the quantization group using warp shuffles.
        for (int offset = group_size_in_vecs / 2; offset > 0; offset /= 2) {
          local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, offset));
        }
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          elem_group_absmax[i] = local_absmax;
        }
      } else {
        // use shared-memory atomicMax for group max reduction
        for (int g = threadIdx.x; g < groups_in_block; g += blockDim.x) {
          smem_group_absmax[g] = 0;
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          int local_group = (block_elem_start + i) / group_size;
          float absval = fabsf(static_cast<float>(reinterpret_cast<T*>(&val)[i]));
          atomicMax(&smem_group_absmax[local_group], __float_as_uint(absval));
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          int local_group = (block_elem_start + i) / group_size;
          elem_group_absmax[i] = __uint_as_float(smem_group_absmax[local_group]);
        }
      }

      // compute UE8M0 scale and quantize to FP8
      auto compute_ue8m0_scale = [](float group_absmax) -> float {
        float y_s = fmaxf(group_absmax / FP8_E4M3_MAX, 1e-10f);
        unsigned int y_s_bits = __float_as_uint(y_s);
        if (y_s_bits & 0x7fffff) {
          y_s_bits = (y_s_bits + 0x800000) & 0x7f800000;
        }
        return __uint_as_float(y_s_bits);
      };

      using PackedQuantizedType = std::conditional_t<std::is_same_v<T, float>, float, float2>;
      PackedQuantizedType ret;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        float y_s = compute_ue8m0_scale(elem_group_absmax[i]);
        float q = static_cast<float>(reinterpret_cast<T*>(&val)[i]) / y_s;
        q = fminf(fmaxf(q, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        reinterpret_cast<__nv_fp8_e4m3*>(&ret)[i] = static_cast<__nv_fp8_e4m3>(q);
      }
      reinterpret_cast<PackedQuantizedType*>(m_params.quant_out)[m_access_id] = ret;

      // write packed UE8M0 scales
      // For warp-shuffle path: one thread per group (first thread in each group).
      // For smem path: one thread per group in the block (threadIdx.x < groups_in_block).
      int block_first_elem = (m_access_id_in_token - threadIdx.x) * VEC_SIZE;
      auto write_group_scale = [&](int group_idx_in_row, float group_absmax) {
        float y_s = compute_ue8m0_scale(group_absmax);
        int groups_per_row = m_params.hidden_dim / group_size;
        int k_num_packed = (groups_per_row + 3) / 4;
        int token_num = m_params.size / m_params.hidden_dim;
        int pack_idx = group_idx_in_row / 4;
        int pos = group_idx_in_row % 4;
        int elem_idx = pack_idx * m_params.tma_aligned_mn + token_id;

        // Write valid exponent
        unsigned int bits = __float_as_uint(y_s);
        uint8_t exponent = static_cast<uint8_t>((bits >> 23u) & 0xffu);
        reinterpret_cast<uint8_t*>(m_params.scale_out)[elem_idx * 4 + pos] = exponent;

        // K-padding: last valid group zeros trailing bytes in its pack
        if (group_idx_in_row == groups_per_row - 1) {
          for (int p = pos + 1; p < 4; p++) {
            reinterpret_cast<uint8_t*>(m_params.scale_out)[elem_idx * 4 + p] = 0;
          }
        }

        // MN-padding: on last valid token, first group zeros all packs
        // for padding tokens (token_num .. tma_aligned_mn - 1).
        // Skip the last packed column (pk = k_num_packed - 1) because
        // scale_out storage is (token_num + (k_num_packed-1)*tma_aligned_mn)
        // elements — the last column only has token_num rows allocated.
        if (token_id == token_num - 1 && group_idx_in_row == 0) {
          for (int pad_t = token_num; pad_t < m_params.tma_aligned_mn; pad_t++) {
            for (int pk = 0; pk < k_num_packed - 1; pk++) {
              int pad_elem = pk * m_params.tma_aligned_mn + pad_t;
              reinterpret_cast<uint32_t*>(m_params.scale_out)[pad_elem] = 0;
            }
          }
        }
      };

      if (use_warp_shuffle) {
        int lane_in_group = m_access_id_in_token % group_size_in_vecs;
        if (lane_in_group == 0) {
          int group_idx_in_row = m_access_id_in_token / group_size_in_vecs;
          write_group_scale(group_idx_in_row, elem_group_absmax[0]);
        }
      } else {
        // Loop: groups_in_block may exceed blockDim.x when group_size < VEC_SIZE
        for (int local_group = threadIdx.x; local_group < groups_in_block;
             local_group += blockDim.x) {
          float group_absmax = __uint_as_float(smem_group_absmax[local_group]);
          int group_idx_in_row = block_first_elem / group_size + local_group;
          write_group_scale(group_idx_in_row, group_absmax);
        }
      }
    } else {
      static_assert(GetQuantType<Pattern> == QuantType::kNone, "Invalid quant type");
    }
  }

 protected:
  __device__ __forceinline__ vec_t<T, VEC_SIZE> rms_norm(vec_t<T, VEC_SIZE> const& residual,
                                                         vec_t<T, VEC_SIZE> const& gamma) {
    __shared__ float s_val;
    vec_t<T, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float v = static_cast<float>(reinterpret_cast<T const*>(&residual)[i]);
      acc += v * v;
    }
    utils::blockReduceSumV2<float, 1>(&acc);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    if (cluster.num_blocks() > 1) {
      if (threadIdx.x == 0) {
        s_val = acc;
        acc = 0.f;
      }
      cluster.sync();
      if (threadIdx.x == 0) {
        for (int i = 0; i < cluster.num_blocks(); ++i) {
          acc += *cluster.map_shared_rank(&s_val, i);
        }
      }
      cluster.sync();
    }
#endif
    if (threadIdx.x == 0) {
      s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      reinterpret_cast<T*>(&norm_out)[i] = static_cast<T>(
          static_cast<float>(reinterpret_cast<T const*>(&residual)[i]) * s_val *
          (m_params.weight_bias + static_cast<float>(reinterpret_cast<T const*>(&gamma)[i])));
    }
    return norm_out;
  }

 private:
  AllReduceFusionParams<T> const& m_params;
  int m_access_id;
  int m_access_id_in_token;
  float m_scale_factor;
  vec_t<T, VEC_SIZE> m_residual_val;
  vec_t<T, VEC_SIZE> m_gamma_val;
};

template <typename T>
struct neg_zero {
  static constexpr T value = -T(0);
};

template <>
struct neg_zero<half> {
  static constexpr unsigned short neg_zero_bits = 0x8000U;
  static constexpr __half value = __half_raw{neg_zero_bits};
};

template <>
struct neg_zero<nv_bfloat16> {
  static constexpr unsigned short neg_zero_bits = 0x8000U;
  static constexpr __nv_bfloat16 value = __nv_bfloat16_raw{neg_zero_bits};
};

template <>
struct neg_zero<float> {
  static constexpr unsigned int neg_zero_bits = 0x80000000U;
  static constexpr float value = -0.0f;
};

template <typename T>
__device__ static constexpr T neg_zero_v = neg_zero<T>::value;

template <typename T>
__device__ bool is_negative_zero(T) {
  return false;
}

// float specialization
template <>
__device__ bool is_negative_zero<float>(float x) {
  return (__float_as_int(x) == 0x80000000);
}

// double specialization
template <>
__device__ bool is_negative_zero<double>(double x) {
  return (__double_as_longlong(x) == 0x8000000000000000ULL);
}

// __half specialization
template <>
__device__ bool is_negative_zero<__half>(__half x) {
  return (__half_as_ushort(x) == 0x8000);
}

// __nv_bfloat16 specialization
template <>
__device__ bool is_negative_zero<__nv_bfloat16>(__nv_bfloat16 x) {
  return (__bfloat16_as_ushort(x) == 0x8000);
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ bool has_neg_zero(const vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    if (is_negative_zero(vec[i])) {
      return true;
    }
  }
  return false;
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void remove_neg_zero(vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    vec[i] = (is_negative_zero(vec[i])) ? static_cast<T>(0.f) : vec[i];
  }
}

template <typename T>
__device__ __forceinline__ void set_neg_zero(T* addr) {
  vec_t<T, details::kBytesPerAccess / sizeof(T)> val;
  val.fill(neg_zero_v<T>);
  val.store_global_volatile(addr);
}

template <typename T, uint32_t VEC_SIZE, int NRanks, bool Fp32Acc>
__device__ __forceinline__ vec_t<T, VEC_SIZE> allreduce_sum(vec_t<T, VEC_SIZE>* vals) {
  if constexpr (Fp32Acc) {
    static_assert(!std::is_same_v<T, float>);
    // Optimization: process one element at a time to reduce register usage
    // Instead of storing acc_f32[VEC_SIZE] (32 bytes), process and convert immediately
    vec_t<T, VEC_SIZE> acc;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float acc_f32 = static_cast<float>(reinterpret_cast<T*>(&vals[0])[i]);
#pragma unroll
      for (int r = 1; r < NRanks; ++r) {
        acc_f32 += static_cast<float>(reinterpret_cast<T*>(&vals[r])[i]);
      }
      acc[i] = static_cast<T>(acc_f32);
    }
    return acc;
  } else {
    vec_t<T, VEC_SIZE> acc = vals[0];
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
      acc = vec_add<T, VEC_SIZE>(acc, vals[r]);
    }
    return acc;
  }
}

template <typename T>
class IndexHelper {
 public:
  __device__ __forceinline__ IndexHelper(AllReduceFusionParams<T> const& params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    token_id = grid.cluster_rank();
    access_id_in_token = cluster.thread_rank();
    token_stride = grid.num_clusters();
#else
    token_id = blockIdx.x;
    access_id_in_token = threadIdx.x;
    token_stride = gridDim.x;
#endif
    access_id = token_id * params.hidden_dim / VEC_SIZE + access_id_in_token;
    access_stride = token_stride * params.hidden_dim / VEC_SIZE;
    tot_access = params.size / VEC_SIZE;
  }

  int token_id;
  int access_id_in_token;
  int token_stride;
  int access_id;
  int access_stride;
  int tot_access;
};

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc,
          bool TriggerCompletionAtEnd = true>
__global__ void allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams<T> params) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  IndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  vec_t<T, VEC_SIZE> clear_vec;
  clear_vec.fill(neg_zero_v<T>);
  FusedOp<Pattern, T> fused_op(params, access_id, access_id_in_token);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  if constexpr (!TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
  LamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / VEC_SIZE;

  for (int idx = access_id; idx < tot_access; idx += access_stride) {
    vec_t<T, VEC_SIZE> val;
    val.load(reinterpret_cast<T*>(params.allreduce_in) + idx * VEC_SIZE);
    remove_neg_zero<T, VEC_SIZE>(val);
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      // Push data to other ranks
      val.store(reinterpret_cast<T*>(comm.data_bufs[r]) +
                (params.rank * tot_access + idx) * VEC_SIZE);
    }
  }
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    // Clear comm buffer that previous kernel used
    clear_vec.store(reinterpret_cast<T*>(comm.clear_buf) + idx * VEC_SIZE);
  }

  for (int idx = access_id, tidx = token_id; idx < tot_access;
       idx += access_stride, tidx += token_stride) {
    fused_op.update(idx);
    vec_t<T, VEC_SIZE> vals[NRanks];
    bool done = false;

    while (!done) {
      done = true;
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        // LDG.128 from local rank
        vals[r].load_global_volatile(reinterpret_cast<T*>(comm.data_bufs[params.rank]) +
                                     (r * tot_access + idx) * VEC_SIZE);
        done &= !has_neg_zero<T, VEC_SIZE>(vals[r]);
      }
    }
    vec_t<T, VEC_SIZE> sum_val = allreduce_sum<T, VEC_SIZE, NRanks, Fp32Acc>(vals);
    fused_op(sum_val, tidx);
  }

  comm.update(params.size * NRanks);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
__global__ void allreduce_fusion_kernel_twoshot_sync(AllReduceFusionParams<T> params,
                                                     std::array<int, NRanks> begin_tokens,
                                                     std::array<int, NRanks> token_num_per_ranks) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  IndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  FusedOp<Pattern, T> fused_op(params, access_id, access_id_in_token);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  SyncComm<NRanks> comm(params.workspace);
#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / VEC_SIZE;
    int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / VEC_SIZE;
    for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride) {
      reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[idx] =
          reinterpret_cast<float4*>(params.allreduce_in)[idx];
    }
  }
  Barrier<NRanks> barrier(params.rank, comm);
  barrier.sync();
  int comm_access_id = access_id + begin_tokens[params.rank] * params.hidden_dim / VEC_SIZE;
  int comm_tot_access =
      (begin_tokens[params.rank] + token_num_per_ranks[params.rank]) * params.hidden_dim / VEC_SIZE;
  for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride) {
    vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      vals[r].load(reinterpret_cast<T*>(comm.comm_bufs[r]) + idx * VEC_SIZE);
    }
    vec_t<T, VEC_SIZE> sum_val = allreduce_sum<T, VEC_SIZE, NRanks, Fp32Acc>(vals);
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      sum_val.store(reinterpret_cast<T*>(comm.comm_bufs[r]) + (tot_access + idx) * VEC_SIZE);
    }
  }
  barrier.sync();
#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / VEC_SIZE;
    int comm_token_id = token_id + begin_tokens[r];
    int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / VEC_SIZE;
    for (int idx = comm_access_id, tidx = comm_token_id; idx < comm_tot_access;
         idx += access_stride, tidx += token_stride) {
      fused_op.update(idx);
      vec_t<T, VEC_SIZE> sum_val;
      sum_val.load(reinterpret_cast<T*>(comm.comm_bufs[params.rank]) +
                   (tot_access + idx) * VEC_SIZE);
      fused_op(sum_val, tidx);
    }
  }
  comm.update(barrier.m_flag_value);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// PIE: REMOVED -- `get_sm_count`, `launch_oneshot_lamport`,
// `get_registers_per_thread_oneshot`, `launch_twoshot_sync`,
// `get_registers_per_thread_twoshot`, `use_oneshot`,
// `allreduce_fusion_kernel_launcher` and `allreduce_fusion_op`. 296 lines of host C++
// and the whole of what upstream calls a "launcher": `cudaGetDevice`,
// `cudaDeviceGetAttribute`, `cudaFuncGetAttributes`, `cudaLaunchConfig_t`,
// `cudaLaunchAttribute`, `cudaLaunchKernelEx`, `std::min`, five lambdas, twelve
// `FLASHINFER_CHECK`s and two nested dispatch macros. Not one line carries a
// `__device__` or `__global__` qualifier.
//
// It is the largest removal in this directory and the one that has a NAMED
// replacement rather than a deleted one: `kernels_cuda::comm` is all of it in Rust.
// The two `switch`es of `allreduce_fusion_op` are `comm::resolve`; the
// `#include "kernels.def"` that expanded into `DISPATCH_PATTERN`'s case labels is
// `comm::INSTANTIATED`; the `case 2/4/8/16` list is `comm::NRANKS`; the three
// `FLASHINFER_CHECK`s that this file turned into a `throw` are `comm::Decline`
// variants; and the grid/block arithmetic at `:1660-1685` (upstream numbering) is
// `comm::fusion_geometry`, whose doc records exactly which two of upstream's
// decisions it does NOT make and why -- the cluster dimension and the
// registers-per-thread clamp.
//
// **The two `__global__`s this file exists for stand above it, untouched**, and they
// are what NVRTC is handed: `allreduce_fusion_kernel_oneshot_lamport` and
// `allreduce_fusion_kernel_twoshot_sync`. That is the whole distinction this removal
// draws -- upstream's `allreduce_fusion_kernel_launcher` is a HOST function whose name
// no `nvrtcAddNameExpression` can lower, which is why `comm::Instantiation::\
// name_expression` names the `__global__` and not the launcher.
//
// This marker is one a strip does NOT undo; see MODIFICATIONS.

}  // namespace trtllm_allreduce_fusion

}  // namespace flashinfer
