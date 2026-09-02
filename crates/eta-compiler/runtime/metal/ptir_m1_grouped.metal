
struct M3LaneHeader { uint abi_version; uint lane_count; uint channel_count; uint flags; };
struct M3LaneRecord {
  ulong logits_base;
  uint logits_row_offset;
  uint logits_row_count;
  uint kv_len;
  uint page_count;
  uint row_count;
  uint token_count;
  uint sampled_rows;
  uint query_len;
  uint key_len;
  uint channel_slot_offset;
  ulong rng_state;
  ulong commit_slot;
  ulong active_row_mask;
  ulong sample_output_channel_mask;
  ulong row_valid;
  uint row_valid_offset;
  uint reserved0;
  ulong attn_score_base;
  uint attn_score_row_stride;
  uint reserved1;
  ulong mtp_drafts_base;
  uint mtp_drafts_depth;
  uint reserved2;
};
struct M3LaneChannelSlot {
  ulong committed_cell;
  ulong pending_cell;
  ulong expected_head;
  ulong expected_tail;
};
struct M3ChannelMeta {
  ulong words;
  uint capacity;
  uint flags;
};
struct M3GroupLayout {
  uint lane_count;
  uint value_count;
  uint scratch_stride;
  uint temporary_offset;
  uint vocab;
  uint reserved0;
  uint reserved1;
  uint reserved2;
};
struct M3RowMeta {
  uint offset;
  uint count;
  uint mtp_offset;
  uint reserved;
};
