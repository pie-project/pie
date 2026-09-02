//! The lane-table ABI: the structs the host fills and the kernels read.
//! Declarations, not a compilation pass. `crate::codegen::layout` mirrors the
//! layout into the generated C header and both MSL preambles, pinning these
//! structs with `offset_of!` so the copies cannot drift.
//!
//! [`LaneRecord`] and [`LaneChannelSlot`] carry three kinds of `u64`
//! (addresses, ring tickets, opaque values), each labeled in its doc comment
//! since nothing else distinguishes them at the type level.
//!
//! [`LaneRecord::attn_score_base`] is declared and zeroed on both backends
//! even though only the grouped Metal form reads it, so the fields stay
//! pinned identically on both planes; a field declared on one side only is
//! the reinterpretation [`crate::codegen::layout`] exists to rule out.

/// Stamped into [`LaneTableHeader::abi_version`]; every backend decoder checks it.
pub const LANE_TABLE_ABI_VERSION: u32 = 4;

/// Runtime extents never enter a stage signature.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeExtents {
    /// Number of live KV-cache entries.
    pub kv_len: u32,
    /// Number of KV-cache pages.
    pub page_count: u32,
    /// Number of rows (requests) in the batch.
    pub row_count: u32,
    /// Number of input tokens in the pass.
    pub token_count: u32,
    /// Number of rows read out for sampling.
    pub sampled_rows: u32,
    /// Attention query length.
    pub query_len: u32,
    /// Attention key length.
    pub key_len: u32,
}

/// Stable grouped-dispatch header. Lane-record address fields are device virtual addresses, `u64` on both backends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneTableHeader {
    /// Copy of [`LANE_TABLE_ABI_VERSION`]; a decoder rejects a table that differs.
    pub abi_version: u32,
    /// Number of [`LaneRecord`]s that follow the header.
    pub lane_count: u32,
    /// [`LaneChannelSlot`]s per lane, the stride between lanes. The MSL preamble spells it `channel_count`.
    pub channel_slots_per_lane: u32,
    /// Flag bits: table validity and the stage's runtime requirements, read before the grouped path is taken.
    pub flags: u32,
}

/// One lane's dispatch state: buffer bases, runtime extents, and its window into the flat channel-slot array.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneRecord {
    /// *Address.* Base of this lane's logits buffer.
    pub logits_base: u64,
    /// Where this lane starts in [`logits_base`](Self::logits_base) — a row count, not a byte offset.
    pub logits_row_offset: u32,
    /// How many rows of [`logits_base`](Self::logits_base) belong to this lane.
    pub logits_row_count: u32,
    /// The runtime extents, matching [`RuntimeExtents`] field for field; never in a stage signature, which is what lets one plan serve many batch shapes.
    pub kv_len: u32,
    /// KV-cache pages for this lane.
    pub page_count: u32,
    /// Rows for this lane.
    pub row_count: u32,
    /// Input tokens for this lane.
    pub token_count: u32,
    /// Rows this lane reads out for sampling.
    pub sampled_rows: u32,
    /// Attention query length for this lane.
    pub query_len: u32,
    /// Attention key length for this lane.
    pub key_len: u32,
    /// This lane's first [`LaneChannelSlot`]; stage-local channel `n` is at `channel_slot_offset + n`.
    pub channel_slot_offset: u32,
    /// *Opaque value.* Counter-mode RNG key. Not an address, though it sits next to one — swapping the two preserves every size.
    pub rng_state: u64,
    /// *Address.* Where the kernel writes its commit record.
    pub commit_slot: u64,
    /// *Address.* Optional bitset of active rows; zero means all `logits_row_count` are active.
    pub active_row_mask: u64,
    /// *Opaque value.* Channel bits whose puts publish the sampled token — a bitset by value, not a pointer.
    pub sample_output_channel_mask: u64,
    /// *Address.* Optional byte mask for model rows; [`row_valid_offset`](Self::row_valid_offset) selects this program's first row.
    pub row_valid: u64,
    /// Row index into [`row_valid`](Self::row_valid).
    pub row_valid_offset: u32,
    /// Padding to 8-byte alignment. Must be zero; a non-zero reserved word is a corrupt table.
    pub reserved0: u32,
    /// *Address.* Base of this lane's block of the attention-score slab, or
    /// zero for a lane that captured nothing. Zero is honest absence: the
    /// emitted grouped gather faults rather than dereferencing it.
    pub attn_score_base: u64,
    /// Row pitch of the rectangle [`attn_score_base`](Self::attn_score_base)
    /// points at, in F32 elements (CUDA's `intrinsic_row_stride`).
    pub attn_score_row_stride: u32,
    /// Padding to 8-byte alignment. Must be zero, for [`reserved0`](Self::reserved0)'s reason.
    pub reserved1: u32,
    /// *Address.* Base of this lane's block of the `mtp.drafts` plane — the
    /// draft head's `[rows, depth]` i32 token ids, at the lane's first readout
    /// row — or zero for a load with no draft head. Zero is honest absence,
    /// as for [`attn_score_base`](Self::attn_score_base).
    pub mtp_drafts_base: u64,
    /// Row pitch of that plane in i32 elements: the head's chain depth.
    pub mtp_drafts_depth: u32,
    /// Padding to 8-byte alignment. Must be zero, for [`reserved0`](Self::reserved0)'s reason.
    pub reserved2: u32,
}

/// One stage-local channel's per-lane state: the cells a put/take touches and the ring tickets the kernel checks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneChannelSlot {
    /// *Address.* The cell a `take`/`read` reads from.
    pub committed_cell: u64,
    /// *Address.* The cell a `put` writes into before commit advances the ring.
    pub pending_cell: u64,
    /// *Ticket.* The ring head the host observed; a mismatch is a stale table and the kernel refuses. `CHANNEL_TICKET_NONE` means "not consuming".
    pub expected_head: u64,
    /// *Ticket.* Ring tail counterpart of [`expected_head`](Self::expected_head); `CHANNEL_TICKET_NONE` means "not publishing".
    pub expected_tail: u64,
}
