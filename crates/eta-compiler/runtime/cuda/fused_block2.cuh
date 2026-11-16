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
  // The lane's scratch is its instance's, wherever the pool holds it, in
  // pieces no value crosses: value `v` lies `offsets[v]` into piece
  // `pieces[v]`, whose address is the lane's base table's entry
  // (`lane.scratch_table`, a device address a piece). Its 64-byte line of
  // the fire's tables, zeroed each fire, is its own too (a lane sampler's
  // barrier words).
  const m1_u64* const scratch_bases =
      reinterpret_cast<const m1_u64*>(lane.scratch_table);
#define ptir_at(v) \
  (reinterpret_cast<m1_u8*>(scratch_bases[pieces[(v)]]) + offsets[(v)])
  // Piece 0, which opens with the dummy bytes: what an op's slot it has no
  // value for names (`codegen::slots::Slots`), never read.
  m1_u8* const scratch = reinterpret_cast<m1_u8*>(scratch_bases[0]);
  (void)scratch;
  m1_u8* lane_line = lane_lines + (m1_u64)dispatch_lane * 64u;
  (void)lane_line;

  m1_u8* temporary = reinterpret_cast<m1_u8*>(scratch_bases[temporary_piece]) +
      temporary_offset + (m1_u64)lane_row * temporary_stride;
  (void)temporary;
