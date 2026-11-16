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
  // The lane's scratch is its instance's, wherever the pool holds it: the
  // first piece of its base table (`lane.scratch_table`, a device address
  // a piece). Its 64-byte line of the fire's tables, zeroed each fire, is
  // its own too (a lane sampler's barrier words).
  m1_u8* scratch = reinterpret_cast<m1_u8*>(
      reinterpret_cast<const m1_u64*>(lane.scratch_table)[0]);
  m1_u8* lane_line = lane_lines + (m1_u64)dispatch_lane * 64u;
  (void)lane_line;

  m1_u8* temporary = scratch + temporary_offset + (m1_u64)lane_row * temporary_stride;
  (void)temporary;
