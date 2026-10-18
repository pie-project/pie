
// The 32-lane chunk the reductions' numeric contract is written over (a
// width-32 tree; see `canonical_reduce`). A CDNA wavefront is 64 lanes and
// carries two chunks side by side: every shuffle stays within its chunk
// (width 32), and a chunk's ballot is its half of the wavefront's.
template <class T>
__device__ __forceinline__ T m1_chunk_down(T v, m1_u32 offset) {
  return __shfl_down(v, offset, 32);
}
template <class T>
__device__ __forceinline__ T m1_chunk_up(T v, m1_u32 offset) {
  return __shfl_up(v, offset, 32);
}
template <class T>
__device__ __forceinline__ T m1_chunk_from(T v, m1_u32 lane) {
  return __shfl(v, lane, 32);
}
__device__ __forceinline__ m1_u32 m1_chunk_ballot(bool predicate) {
  const unsigned long long wave = __ballot(predicate);
  return static_cast<m1_u32>(wave >> (__lane_id() & 32u));
}

// The inclusive prefix sum of `v` over the lane's 32-lane chunk, by DPP row
// shifts within each 16-lane row and row 0's (2's) last lane broadcast
// into row 1 (3).
__device__ __forceinline__ m1_u32 m1_chunk_scan(m1_u32 v) {
  v += (m1_u32)__builtin_amdgcn_update_dpp(0, (int)v, 0x111, 0xF, 0xF, false);
  v += (m1_u32)__builtin_amdgcn_update_dpp(0, (int)v, 0x112, 0xF, 0xF, false);
  v += (m1_u32)__builtin_amdgcn_update_dpp(0, (int)v, 0x114, 0xF, 0xF, false);
  v += (m1_u32)__builtin_amdgcn_update_dpp(0, (int)v, 0x118, 0xF, 0xF, false);
  v += (m1_u32)__builtin_amdgcn_update_dpp(0, (int)v, 0x142, 0xA, 0xF, false);
  return v;
}
__device__ __forceinline__ m1_u64 m1_chunk_scan(m1_u64 v) {
#define M1_DPP64(ctrl, rows)                                                          \
  {                                                                                   \
    const m1_u32 lo =                                                                 \
        (m1_u32)__builtin_amdgcn_update_dpp(0, (int)(m1_u32)v, ctrl, rows, 0xF, false); \
    const m1_u32 hi = (m1_u32)__builtin_amdgcn_update_dpp(                            \
        0, (int)(m1_u32)(v >> 32), ctrl, rows, 0xF, false);                           \
    v += ((m1_u64)hi << 32) | lo;                                                     \
  }
  M1_DPP64(0x111, 0xF)
  M1_DPP64(0x112, 0xF)
  M1_DPP64(0x114, 0xF)
  M1_DPP64(0x118, 0xF)
  M1_DPP64(0x142, 0xA)
#undef M1_DPP64
  return v;
}

// `v` of lane `lane` (uniform) of the block's first 32 lanes, read by them.
__device__ __forceinline__ m1_u32 m1_lead_from(m1_u32 v, m1_u32 lane) {
  return (m1_u32)__builtin_amdgcn_readlane((int)v, (int)lane);
}
__device__ __forceinline__ int m1_lead_from(int v, m1_u32 lane) {
  return __builtin_amdgcn_readlane(v, (int)lane);
}
__device__ __forceinline__ float m1_lead_from(float v, m1_u32 lane) {
  return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), (int)lane));
}
__device__ __forceinline__ m1_u64 m1_lead_from(m1_u64 v, m1_u32 lane) {
  const m1_u32 lo = (m1_u32)__builtin_amdgcn_readlane((int)(m1_u32)v, (int)lane);
  const m1_u32 hi = (m1_u32)__builtin_amdgcn_readlane((int)(m1_u32)(v >> 32), (int)lane);
  return ((m1_u64)hi << 32) | lo;
}

// `v` of the lane `lane ^ mask` of the lane's 32-lane chunk.
template <class T>
__device__ __forceinline__ T m1_chunk_xor(T v, m1_u32 mask) {
  return __shfl_xor(v, mask, 32);
}
