
// The 32-lane chunk the reductions' numeric contract is written over (a
// width-32 tree; see `canonical_reduce`), as CUDA's warp holds it.
template <class T>
__device__ __forceinline__ T m1_chunk_down(T v, m1_u32 offset) {
  return __shfl_down_sync(0xffffffffu, v, offset);
}
__device__ __forceinline__ m1_u32 m1_chunk_ballot(bool predicate) {
  return __ballot_sync(0xffffffffu, predicate);
}
