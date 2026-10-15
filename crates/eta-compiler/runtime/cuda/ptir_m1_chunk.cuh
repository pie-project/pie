
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

// A slot of `counter` for each lane of the warp that wants one (in lane
// order): one atomic a warp, not one a lane.
__device__ __forceinline__ m1_u32 m1_wave_slot(m1_u32* counter, bool want) {
  const unsigned active = __activemask();
  const unsigned mask = __ballot_sync(active, want);
  if (mask == 0u) return 0u;
  const unsigned lane = threadIdx.x & 31u;
  const unsigned leader = (unsigned)(__ffs((int)mask) - 1);
  m1_u32 base = 0u;
  if (lane == leader) base = atomicAdd(counter, (m1_u32)__popc(mask));
  base = __shfl_sync(active, base, leader);
  return base + (m1_u32)__popc(mask & ((1u << lane) - 1u));
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
