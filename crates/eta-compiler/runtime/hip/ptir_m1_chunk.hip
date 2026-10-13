
// The 32-lane chunk the reductions' numeric contract is written over (a
// width-32 tree; see `canonical_reduce`). A CDNA wavefront is 64 lanes and
// carries two chunks side by side: every shuffle stays within its chunk
// (width 32), and a chunk's ballot is its half of the wavefront's.
template <class T>
__device__ __forceinline__ T m1_chunk_down(T v, m1_u32 offset) {
  return __shfl_down(v, offset, 32);
}
__device__ __forceinline__ m1_u32 m1_chunk_ballot(bool predicate) {
  const unsigned long long wave = __ballot(predicate);
  return static_cast<m1_u32>(wave >> (__lane_id() & 32u));
}
