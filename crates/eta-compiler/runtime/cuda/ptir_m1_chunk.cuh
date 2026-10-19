
// NVRTC's static shared memory stops at 48 KB, which the exact cut's state
// passes beside a region's other shared arrays: on CUDA it sits in the
// launch's dynamic shared memory, this many bytes of it
// (`eta_compiler::codegen::cuda::CUT_STATE_BYTES`, the launcher's).
#define M1_CUT_DYNAMIC 1
#define M1_CUT_STATE_BYTES 34816u

// The 32-lane chunk the reductions' numeric contract is written over (a
// width-32 tree; see `canonical_reduce`), as CUDA's warp holds it.
template <class T>
__device__ __forceinline__ T m1_chunk_down(T v, m1_u32 offset) {
  return __shfl_down_sync(0xffffffffu, v, offset);
}
template <class T>
__device__ __forceinline__ T m1_chunk_up(T v, m1_u32 offset) {
  return __shfl_up_sync(0xffffffffu, v, offset);
}
template <class T>
__device__ __forceinline__ T m1_chunk_from(T v, m1_u32 lane) {
  return __shfl_sync(0xffffffffu, v, lane);
}
__device__ __forceinline__ m1_u32 m1_chunk_ballot(bool predicate) {
  return __ballot_sync(0xffffffffu, predicate);
}

// The inclusive prefix sum of `v` over the warp.
template <class T>
__device__ __forceinline__ T m1_chunk_scan(T v) {
  const unsigned lane = threadIdx.x & 31u;
  for (unsigned offset = 1u; offset < 32u; offset <<= 1) {
    const T other = __shfl_up_sync(0xffffffffu, v, offset);
    if (lane >= offset) v += other;
  }
  return v;
}

// `v` of lane `lane` of the block's first warp, read by it.
template <class T>
__device__ __forceinline__ T m1_lead_from(T v, m1_u32 lane) {
  return __shfl_sync(0xffffffffu, v, lane);
}

// `v` of the lane `lane ^ mask` of the warp.
template <class T>
__device__ __forceinline__ T m1_chunk_xor(T v, m1_u32 mask) {
  return __shfl_xor_sync(0xffffffffu, v, mask);
}
