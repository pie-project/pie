
// ------------------------------------------------------------- split rows --
//
// A generated region whose rows run across several blocks of their lane
// (`splits_rows`): each block streams its stretch of its row, the stretches
// cut at 128-element bounds so a 32-element chunk is one block's alone. A
// reduction's chunk partials land in the row's work at their chunk; the
// row's blocks meet (a word of the lane arenas a row, zeroed before the
// launch, counting arrivals); then every block folds the partials in the
// reference's order, so each holds the one-block answer.

// Where part `part` of `split` begins in a `width`-wide row.
__device__ __forceinline__ m1_u32 ptir_stretch(m1_u32 width, m1_u32 part, m1_u32 split) {
  if (part >= split) return width;
  const m1_u32 at = (m1_u32)(((m1_u64)width * part / split) & ~(m1_u64)127u);
  return at < width ? at : width;
}

// The row's blocks meet: `target` arrivals at `word`, this block's among them.
__device__ __forceinline__ void ptir_meet(m1_u32* word, m1_u32 target) {
  __syncthreads();
  if (threadIdx.x == 0u) {
    M1_LANE_RELEASE();
    atomicAdd(word, 1u);
    while (M1_LANE_LOAD(word) < target) M1_LANE_PAUSE();
    M1_LANE_ACQUIRE();
  }
  __syncthreads();
}

// `ptir_canonical_reduce` over a first level other blocks wrote: read past
// this block's cache, folded into this block's `out` and `spare`.
__device__ __forceinline__ float ptir_canonical_reduce_met(
    m1_u32 tag, float* level, m1_u32 count, float* out, float* spare) {
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  if (count == 0u) return m1_reduce_identity(tag);
  if (count == 1u) {
    const float answer = __uint_as_float(M1_LANE_GET32(reinterpret_cast<m1_u32*>(level)));
    __syncthreads();
    return answer;
  }
  const m1_u32 chunks = (count + 31u) / 32u;
  for (m1_u32 chunk = warp; chunk < chunks; chunk += warps) {
    const m1_u32 index = chunk * 32u + lane;
    const float value = m1_reduce_chunk(
        tag,
        index < count
            ? __uint_as_float(M1_LANE_GET32(reinterpret_cast<m1_u32*>(level + index)))
            : m1_reduce_identity(tag));
    if (lane == 0u) out[chunk] = value;
  }
  __syncthreads();
  return ptir_canonical_reduce(tag, out, chunks, spare, out);
}

