














#ifndef FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_
#define FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_

#include <cuda_runtime.h>

#include <cstdint>

__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_8b(uint32_t x) {
  uint32_t tmp = __shfl_xor_sync(0xffffffff, x, 0x1);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x1) == 0) ? 0x5410 : 0x3276);
  tmp = __shfl_xor_sync(0xffffffff, x, 0x2);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x2) == 0) ? 0x5410 : 0x3276);
  return x;
}

__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_8b_trans(uint32_t x) {
  uint32_t tmp = __shfl_xor_sync(0xffffffff, x, 0x4);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x4) == 0) ? 0x6420 : 0x3175);
  tmp = __shfl_xor_sync(0xffffffff, x, 0x8);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x8) == 0) ? 0x5410 : 0x3276);
  tmp = __shfl_xor_sync(0xffffffff, x, 0x10);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x10) == 0) ? 0x5410 : 0x3276);
  return x;
}

__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_4b(uint32_t x) {

  uint32_t tmp0 = __shfl_sync(0xffffffff, x, threadIdx.x & ~0x3u);

  uint32_t tmp1 = __shfl_sync(0xffffffff, x, (threadIdx.x & ~0x3u) + 1);

  uint32_t byte_idx = threadIdx.x & 0x3u;
  x = __byte_perm(tmp0, tmp1, byte_idx * 0x0101u + 0x0400u);
  return x;
}

__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_4b_trans(uint32_t x) {

  unsigned src_thrd = (threadIdx.x & ~0x1cu) + ((threadIdx.x & 0x10u) >> 2);
  uint32_t tmp0 = __shfl_sync(0xffffffff, x, src_thrd);
  uint32_t tmp1 = __shfl_sync(0xffffffff, x, src_thrd + 8u);

  uint32_t select_code = (threadIdx.x & 0x8u) ? 0x7531u : 0x6420u;
  uint32_t tmp = __byte_perm(tmp0, tmp1, select_code);

  tmp = tmp >> (threadIdx.x & 0x4u);

  tmp = tmp & 0x0F0F0F0F;
  tmp = tmp | (tmp >> 4);
  return tmp;
}

#endif
