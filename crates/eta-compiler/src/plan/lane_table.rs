pub const LANE_TABLE_ABI_VERSION: u32 = 5;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneTableHeader {
    pub abi_version: u32,
    pub lane_count: u32,
    pub channel_slots_per_lane: u32,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneRecord {
    pub logits_base: u64,
    pub logits_row_offset: u32,
    pub logits_row_count: u32,
    pub row_count: u32,
    pub token_count: u32,
    pub sampled_rows: u32,
    pub query_len: u32,
    pub key_len: u32,
    pub channel_slot_offset: u32,
    pub rng_state: u64,
    pub commit_slot: u64,
    pub active_row_mask: u64,
    pub sample_output_channel_mask: u64,
    pub row_valid: u64,
    pub row_valid_offset: u32,
    pub reserved0: u32,
    pub attn_score_base: u64,
    pub attn_score_row_stride: u32,
    pub reserved1: u32,
    pub mtp_drafts_base: u64,
    pub mtp_drafts_depth: u32,
    pub reserved2: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneChannelSlot {
    pub committed_cell: u64,
    pub pending_cell: u64,
    pub expected_head: u64,
    pub expected_tail: u64,
}
