(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const PtirLaneChannelSlot* channels,
    const M1ValueDesc* all_descriptors,
    const M1OpParams* params,
    const m1_u32* offsets,
    m1_u8* all_scratch,
    m1_u32 value_count,
    m1_u32 scratch_stride,
    m1_u32 temporary_offset,
    m1_u8* pending_flags,
    const m1_u64* intrinsic_bases,
    const m1_u32* intrinsic_modes,
    const m1_u32* intrinsic_widths,
    const m1_u32* intrinsic_strides,
    const m1_u32* intrinsic_offsets) {
  const m1_u32 dispatch_lane = blockIdx.x;
  if (header == nullptr || dispatch_lane >= header->lane_count) return;
  const PtirLaneRecord lane = lanes[dispatch_lane];
  m1_u32* commit = reinterpret_cast<m1_u32*>(lane.commit_slot);
  if (commit == nullptr || *commit == 0u) return;
  if (header->abi_version != 