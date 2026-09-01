//! The lane-table ABI: the structs the host fills and the kernels read.
//! Declarations, not a compilation pass. `crate::codegen::layout` mirrors the
//! layout into the generated C header and both MSL preambles, pinning these
//! structs with `offset_of!` so the copies cannot drift.
//!
//! **Three kinds of `u64`, one spelling.** [`LaneRecord`] and
//! [`LaneChannelSlot`] carry ten: addresses a kernel dereferences, ring tickets
//! it only compares, opaque values. Each states its kind because nothing else
//! does — a ticket written where an address belongs compiles, passes the
//! `offset_of!` pinning, and produces a wild dereference. Newtypes are declined:
//! ~100 sites across four crates outside `compiler/` would wrap and unwrap.
//!
//! **THE RECORD CARRIES A SECOND RECTANGLE NOW, AND ONLY ONE SHELL READS IT.**
//! [`LaneRecord::attn_score_base`] is the grouped Metal form's whole answer to
//! the attention-score plane: that form binds no per-intrinsic buffer, so a
//! rectangle it must reach has to arrive as an ADDRESS on this record, and the
//! score slab is not the logits allocation at any displacement. The CUDA shell
//! reaches the same rectangle through its five per-(lane, intrinsic) side
//! arrays and never reads these two words — but the record is one struct on
//! both planes, so the fields are declared, pinned and zeroed on both. A field
//! declared on one side only is the silent reinterpretation
//! [`crate::codegen::layout`] exists to rule out.

/// Stamped into [`LaneTableHeader::abi_version`]; every backend decoder checks it.
pub const LANE_TABLE_ABI_VERSION: u32 = 3;

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
    /// *Address.* Base of this lane's block of the attention-score slab, or zero for a lane that captured nothing.
    ///
    /// **APPENDED, AND DELIBERATELY NOT FOLDED IN BESIDE `logits_base`.** The
    /// two are not one rectangle at a displacement: the readout lives in the
    /// arena a fire wrote and the score slab is the shell's own reservation
    /// (`engine_metal::scores`), which is exactly why the grouped form could
    /// not reach it by counting rows off the trunk the way it reaches the
    /// draft column. It is a second address or it is nothing.
    ///
    /// Zero is the honest absence: the emitted grouped gather faults rather
    /// than dereferencing it, because a lane that did not capture has no
    /// block and the last fire's mass is a wrong answer rather than a missing
    /// one.
    pub attn_score_base: u64,
    /// Row pitch of the rectangle [`attn_score_base`](Self::attn_score_base) points at, in F32 ELEMENTS.
    ///
    /// The CUDA twin's `intrinsic_row_stride` under the one name this record
    /// can give it. A plane is `ATTN_SCORE_KV_MAX` wide whatever a reader
    /// declares, and a reader's declared row is a CEILING on it — the same
    /// relation `ptir_m1_runtime_body.cuh`'s `0xA0` arm checks — so the pitch
    /// has to arrive as its own number rather than be inferred from the
    /// reader.
    pub attn_score_row_stride: u32,
    /// Padding to 8-byte alignment. Must be zero, for [`reserved0`](Self::reserved0)'s reason.
    pub reserved1: u32,
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
