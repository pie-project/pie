u) {
    if (threadIdx.x == 0u) *commit = 0u;
    return;
  }
  const m1_u8* lane_row_valid =
      reinterpret_cast<const m1_u8*>(lane.row_valid);
  const bool lane_active =
      lane_row_valid == nullptr ||
      lane_row_valid[lane.row_valid_offset] != 0u;
  __shared__ M1Status status;
  if (threadIdx.x == 0u)
    status = M1Status{1u, 0u, 0u, 0u};
  __syncthreads();
  const M1ValueDesc* descriptors =
      all_descriptors + (m1_u64)dispatch_lane * value_count;
  m1_u8* scratch =
      all_scratch + (m1_u64)dispatch_lane * scratch_stride;
  // Each of a lane's blocks owns `temporary_stride` bytes of the arena.
  m1_u8* temporary = scratch + temporary_offset + (m1_u64)lane_row * temporary_stride;
  (void)temporary;
