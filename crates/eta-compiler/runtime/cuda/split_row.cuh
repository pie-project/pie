
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

// `ptir_parallel_copy`'s stretch `part` of `split`: sixteen bytes a thread
// where both ends allow it, cut at whole quads, the tail bytewise by the
// last part.
__device__ __forceinline__ void ptir_split_copy(
    const m1_u8* input, m1_u8* output, m1_u32 len, m1_u32 dtype, m1_u32 part, m1_u32 split) {
  if (input == output) return;
  const m1_u32 bytes = dtype == 3u ? len : len * 4u;
  m1_u32 head = 0u;
  if (((reinterpret_cast<m1_u64>(input) | reinterpret_cast<m1_u64>(output)) & 15ull) == 0ull) {
    const m1_u32 quads = bytes >> 4;
    const m1_u32 lo = (m1_u32)((m1_u64)quads * part / split);
    const m1_u32 hi = (m1_u32)((m1_u64)quads * (part + 1u) / split);
    const uint4* in4 = reinterpret_cast<const uint4*>(input);
    uint4* out4 = reinterpret_cast<uint4*>(output);
    for (m1_u32 q = lo + threadIdx.x; q < hi; q += blockDim.x) out4[q] = in4[q];
    head = quads << 4;
  }
  if (part + 1u != split) return;
  for (m1_u32 index = head + threadIdx.x; index < bytes; index += blockDim.x)
    output[index] = input[index];
}

