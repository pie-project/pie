














#ifndef FLASHINFER_CP_ASYNC_CUH_
#define FLASHINFER_CP_ASYNC_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {

namespace cp_async {

enum class SharedMemFillMode {
  kFillZero,
  kNoFill
};

enum class PrefetchMode {
  kNoPrefetch,
  kPrefetch
};

#if (__CUDACC_VER_MAJOR__ >= 11)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define FLASHINFER_CP_ASYNC_ENABLED
#endif
#endif




__device__ __forceinline__ void commit_group() {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}




template <size_t n>
__device__ __forceinline__ void wait_group() {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
#endif
}








template <PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load_128b(T* smem_ptr, const T* gmem_ptr) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(16), "r"(16));
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(16), "r"(16));
  }
#else
  *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
#endif
}











template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load_128b(T* smem_ptr, const T* gmem_ptr, bool predicate) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    int src_in_bytes = predicate ? 16 : 0;
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
      asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                   "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    } else {
      asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                   "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    }
  } else {
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    } else {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    }
  }
#else
  if (predicate) {
    *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
  } else {
    if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
      *((uint4*)smem_ptr) = make_uint4(0, 0, 0, 0);
    }
  }
#endif
}








template <size_t num_bits, PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load(T* smem_ptr, const T* gmem_ptr) {
  static_assert(num_bits == 128 || num_bits == 256, "num_bits must be 128 or 256");
  if constexpr (num_bits == 128) {
    load_128b<prefetch_mode>(smem_ptr, gmem_ptr);
  } else {
    load_128b<prefetch_mode>(smem_ptr, gmem_ptr);
    load_128b<prefetch_mode>(smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T));
  }
}












template <size_t num_bits, PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  static_assert(num_bits == 128 || num_bits == 256, "num_bits must be 128 or 256");
  if constexpr (num_bits == 128) {
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
  } else {
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T),
                                             predicate);
  }
}






template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load_128b_from_64b(T* smem_ptr, const T* gmem_ptr,
                                                        bool predicate) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    int src_in_bytes = predicate ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(8), "r"(src_in_bytes));

  } else {

    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %0, 0;\n"
        " @p cp.async.ca.shared.global [%1], [%2], %3, %4;\n"
        "}\n" ::"r"((int)predicate),
        "r"(smem_int_ptr), "l"(gmem_ptr), "n"(8), "n"(8));
  }
#else
  if (predicate) {
    uint64_t* smem_u64 = reinterpret_cast<uint64_t*>(smem_ptr);
    smem_u64[0] = *reinterpret_cast<const uint64_t*>(gmem_ptr);
    smem_u64[1] = 0;
  } else {
    if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
      *((uint4*)smem_ptr) = make_uint4(0, 0, 0, 0);
    }
  }
#endif
}










template <SharedMemFillMode fill_mode>
__device__ __forceinline__ void pred_load_32b(uint32_t* smem_ptr, const uint32_t* gmem_ptr,
                                              bool predicate) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    int src_in_bytes = predicate ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(4), "r"(src_in_bytes));
  } else {
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %0, 0;\n"
        " @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int)predicate),
        "r"(smem_int_ptr), "l"(gmem_ptr), "n"(4));
  }
#else
  if (predicate) {
    *smem_ptr = *gmem_ptr;
  } else if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    *smem_ptr = 0;
  }
#endif
}

}

}

#endif
