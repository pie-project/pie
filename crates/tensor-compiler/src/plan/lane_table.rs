//! The lane-table ABI: the structs the host fills and the kernels read.
//!
//! Declarations, not a compilation pass, which is why they sit at the crate
//! root rather than under `compile/`. Nothing here reads a stage or produces a
//! plan; every consumer is downstream — `crate::codegen::layout`, `header`, and
//! the two backends' emitters.
//!
//! The field layout is mirrored into the generated C header and both MSL
//! preambles by `crate::codegen::layout`, which pins these structs with
//! `offset_of!` so the copies cannot drift.
//!
//! ## Three kinds of `u64`, one spelling
//!
//! [`LaneRecord`] and [`LaneChannelSlot`] carry nine `u64` fields between
//! them, and they are not the same kind of number:
//!
//! * **device addresses** — [`LaneRecord::logits_base`],
//!   [`commit_slot`](LaneRecord::commit_slot),
//!   [`active_row_mask`](LaneRecord::active_row_mask),
//!   [`row_valid`](LaneRecord::row_valid),
//!   [`LaneChannelSlot::committed_cell`] and
//!   [`pending_cell`](LaneChannelSlot::pending_cell). A kernel dereferences
//!   these.
//! * **ring tickets** — [`LaneChannelSlot::expected_head`] and
//!   [`expected_tail`](LaneChannelSlot::expected_tail), monotonic counters the
//!   kernel compares, never dereferences.
//! * **opaque values** — [`LaneRecord::rng_state`] (a counter-mode key) and
//!   [`sample_output_channel_mask`](LaneRecord::sample_output_channel_mask)
//!   (a bitset over stage-local channels).
//!
//! Every field is documented with its kind because nothing else distinguishes
//! them. Writing a ticket where an address belongs compiles, passes the
//! `offset_of!` pinning, and produces a wild dereference on the device;
//! `crate::codegen::cuda::fused`'s `cuda_runtime_lane_table_matches_layout` says
//! the same thing about the CUDA-side copy.
//!
//! Newtypes (`DeviceAddress(u64)`, `Ticket(u64)`) would make that a compile
//! error and are the right answer in the abstract. They are not applied here
//! because these structs are an ABI read by `runtime/engine`,
//! `interface/driver` and the generated C and MSL — roughly a hundred sites
//! across four crates outside `compiler/`, all of which would have to wrap
//! and unwrap at a boundary where the value is genuinely a bare `u64` again.
//! The cost is real and the benefit stops at the crate edge, so the kind is
//! carried in documentation instead. If these fields ever move behind a
//! constructor that the host must call, that is the moment to revisit it.

/// The lane-table ABI version stamped into [`LaneTableHeader::abi_version`]
/// and checked by every backend decoder before it reads the table.
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

// ## Why there is no executable-cache key here
//
// The obvious companion to these declarations would be an
// "executable-cache identity" — backend, device arch, compiler version,
// stage signature — encoded to a fixed byte array. There is deliberately
// none, because a key defined on this side that no driver reads is not a
// cache key, only a description of a cache nobody built.
//
// The real one is the CUDA driver's, in
// `crates/driver-cuda/csrc/src/pipeline/generated/module_cache.hpp`. It is a different
// shape from anything derivable here: it folds in the real `sm_` arch, the
// NVRTC version the module was compiled by, and a fingerprint of the
// generated source bytes themselves. Those are properties of the machine and
// of the toolchain, which the host planning these tables does not know. So if
// a host-side key is ever wanted, it has to be *generated into* the driver's
// header the way the op table is — otherwise it is a comment that compiles.
//
// This is also why nothing here worries about 64-bit signature collisions.
// Every cache that a signature indexes re-compares the full material on a hit:
// the driver keys on the whole key string, and `pipeline::program`'s registry
// re-compares the container bytes and returns `RegisterError::HashCollision`
// rather than serving the wrong program
// (`crates/engine/src/pipeline/program.rs`). A hash is an index into those
// caches, never the identity — and any new cache keyed on a signature owes the
// same re-check.

/// Stable grouped-dispatch header. Address fields in the lane records are
/// device virtual addresses represented as `u64` on both supported backends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneTableHeader {
    /// Copy of [`LANE_TABLE_ABI_VERSION`]; a decoder rejects the table when it
    /// differs.
    pub abi_version: u32,
    /// Number of [`LaneRecord`]s that follow the header.
    pub lane_count: u32,
    /// [`LaneChannelSlot`]s per lane; the stride from one lane's slots to the
    /// next. The MSL preamble spells this field `channel_count`.
    pub channel_slots_per_lane: u32,
    /// Grouped-dispatch flag bits: table validity and the stage's runtime
    /// requirements, consulted before the driver takes the grouped path.
    pub flags: u32,
}

/// One lane's dispatch state: buffer bases, its runtime extents, and the
/// window into the flat channel-slot array.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneRecord {
    /// *Address.* Base of this lane's logits buffer.
    pub logits_base: u64,
    /// Row index within [`logits_base`](Self::logits_base) where this lane
    /// starts — a row count, not a byte offset.
    pub logits_row_offset: u32,
    /// How many rows of [`logits_base`](Self::logits_base) belong to this lane.
    pub logits_row_count: u32,
    /// The runtime extents, matching [`RuntimeExtents`] field for field. These
    /// never enter a stage signature — that is what makes one plan serve many
    /// batch shapes.
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
    /// Index of this lane's first [`LaneChannelSlot`] in the flat slot array;
    /// stage-local channel `n` is at `channel_slot_offset + n`.
    pub channel_slot_offset: u32,
    /// *Opaque value.* The counter-mode RNG key for this lane. Not an address
    /// — it sits next to one, and the CUDA runtime's lane-table copy is
    /// checked against this declaration precisely because swapping the two
    /// preserves every size.
    pub rng_state: u64,
    /// *Address.* Where the kernel writes its commit record.
    pub commit_slot: u64,
    /// *Address.* Optional device bitset for active rows in a ragged lane;
    /// zero means all `logits_row_count` rows are active.
    pub active_row_mask: u64,
    /// *Opaque value.* Stage-local channel bits whose puts publish the sampled
    /// token value — a bitset by value, not a pointer to one.
    pub sample_output_channel_mask: u64,
    /// *Address.* Optional device byte mask for model rows.
    /// [`row_valid_offset`](Self::row_valid_offset) selects the first row
    /// belonging to this program.
    pub row_valid: u64,
    /// Row index into [`row_valid`](Self::row_valid).
    pub row_valid_offset: u32,
    /// Padding to keep the record 8-byte aligned. Must be zero — the drivers
    /// treat a non-zero reserved word as a corrupt table.
    pub reserved0: u32,
}

/// One stage-local channel's per-lane state: the cells a put/take touches and
/// the ring tickets the kernel checks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneChannelSlot {
    /// *Address.* The cell a `take`/`read` reads from.
    pub committed_cell: u64,
    /// *Address.* The cell a `put` writes into before commit advances the ring.
    pub pending_cell: u64,
    /// *Ticket.* The ring head the host observed when it built this table. The
    /// kernel compares it and refuses on a mismatch (a stale lane table); it
    /// never dereferences it. `CHANNEL_TICKET_NONE` means "not consuming".
    pub expected_head: u64,
    /// *Ticket.* The ring tail counterpart of
    /// [`expected_head`](Self::expected_head). `CHANNEL_TICKET_NONE` means
    /// "not publishing".
    pub expected_tail: u64,
}
