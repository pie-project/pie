//! The lane table and its grouped-dispatch sidecars: the records every launch
//! path writes and every effect kernel reads.
//!
//! A fire hands the GPU one *lane table*: a [`Header`], then one [`Record`]
//! per lane, then a flat array of [`ChannelSlot`]s — `channel_slots_per_lane`
//! of them per lane, contiguously. The single-lane M1 fire and the 64-lane M3
//! group write the same layout; only the counts differ. The grouped path adds
//! three sidecar buffers beside the table — [`ChannelMeta`], [`GroupLayout`]
//! and [`RowMeta`] — that the grouped kernels index per lane.
//!
//! The table structs are declared authoritatively in
//! `tensor_compiler::plan::lane_table` and pinned into the generated C header
//! and both MSL preambles by the compiler's own `codegen::layout`. This crate
//! deliberately does not depend on the compiler at build time — the same
//! reason `identity::Versions` is a parameter — so the structs are *mirrored*
//! here and a dev-dependency test compares every field offset against the
//! compiler's, the same arrangement `status::FAULT_CLASSES` uses. A
//! hand-copied ABI that nothing checks drifts.
//!
//! ## What the C++ got wrong, and this module is shaped against
//!
//! **Three load-bearing fields are named `reserved`, on both sides of the
//! ABI.** The C++ fills `M3GroupLayout::reserved[3]` with the per-lane channel
//! binding stride, the row count of the parallel-selection grid, and the
//! per-lane op stride — and the emitted kernels read all three:
//! `channel_bindings[dispatch_lane * layout->reserved0 + n]`,
//! `group_position / layout->reserved1`,
//! `dispatch_lane * layout->reserved2`. Nothing on either side says the
//! fields are live; a reader deleting "unused padding" or a writer forgetting
//! one gets a kernel indexing another lane's bindings, and every size check
//! still passes. The mirror here names them — [`GroupLayout::binding_stride`],
//! [`GroupLayout::rows_per_lane`], [`GroupLayout::op_stride`] — at the same
//! offsets, and the offsets are tested.
//!
//! **The layout arithmetic was walked by hand, twice.** `prepare` computes
//! `sizeof(header) + sizeof(record) + n * sizeof(slot)` and casts three raw
//! pointers into the buffer; `prepare_m3_group` writes the same walk again for
//! N lanes. Neither checks a lane index — and the record array and the slot
//! array are contiguous, so `records[lane_count]` does not fault, it reads
//! channel slots reinterpreted as a lane record, every field plausible
//! garbage. [`Shape`] computes the sizes and offsets once, checked.
//!
//! **The `static_assert`s pin sizes and nothing else.** `sizeof == 16/32/16`
//! holds under any permutation of same-width fields; swapping
//! `ChannelMeta::capacity` with its `flags` preserves every assertion the C++
//! makes and misdirects every take. The tests here pin each field's offset.

use core::mem::size_of;

/// The lane-table ABI version.
///
/// Mirror of `tensor_compiler::plan::lane_table::LANE_TABLE_ABI_VERSION`; the
/// effect kernels reject a table stamped with anything else, recording the
/// version they saw in the status word's guard fields.
pub const ABI_VERSION: u32 = 3;

/// [`Header::flags`] bit recorded when lanes disagree on their logits row
/// count.
///
/// The C++ writes this bit as a bare `1u`, and writes it twice — once when
/// the raggedness is discovered and once more, redundantly, after the stages
/// are built. No shipped kernel reads it (ragged lanes are consumed through
/// [`Record::active_row_mask`] instead); it is recorded for the table's other
/// reader, a human with a debugger.
pub const FLAG_RAGGED: u32 = 1;

/// Table header: the counts a decoder needs before it can address anything.
///
/// Mirror of the compiler's `LaneTableHeader`. The MSL preamble spells
/// `channel_slots_per_lane` as `channel_count`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Header {
    /// Copy of [`ABI_VERSION`]; the kernels refuse a table that differs.
    pub abi_version: u32,
    /// Number of [`Record`]s that follow.
    pub lane_count: u32,
    /// [`ChannelSlot`]s per lane — the stride from one lane's slots to the
    /// next.
    pub channel_slots_per_lane: u32,
    /// Flag bits; see [`FLAG_RAGGED`].
    pub flags: u32,
}

/// One lane's dispatch state: buffer bases, runtime extents, and the window
/// into the flat channel-slot array.
///
/// Mirror of the compiler's `LaneRecord`. The `u64` fields are three
/// different kinds of number wearing one type — *addresses* the kernel
/// dereferences, *tickets* it only compares, and *opaque values* — and each
/// field says which it is, because nothing else distinguishes a wild
/// dereference from a comparison.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Record {
    /// *Address.* Base of this lane's logits buffer.
    pub logits_base: u64,
    /// First row of [`logits_base`](Self::logits_base) belonging to this lane
    /// — a row count, not a byte offset.
    pub logits_row_offset: u32,
    /// Rows of [`logits_base`](Self::logits_base) belonging to this lane.
    pub logits_row_count: u32,
    /// Live KV-cache entries. This and the six extents after it match
    /// [`Extents`](crate::Extents) field for field; they never enter
    /// a stage signature, which is what lets one plan serve many batch shapes.
    pub kv_len: u32,
    /// KV-cache pages for this lane.
    pub page_count: u32,
    /// Batch rows for this lane.
    pub row_count: u32,
    /// Input tokens for this lane.
    pub token_count: u32,
    /// Rows this lane reads out for sampling.
    pub sampled_rows: u32,
    /// Attention query length for this lane.
    pub query_len: u32,
    /// Attention key length for this lane.
    pub key_len: u32,
    /// Index of this lane's first [`ChannelSlot`] in the flat slot array;
    /// stage-local channel `n` is at `channel_slot_offset + n`. See
    /// [`Shape::slot_index`].
    pub channel_slot_offset: u32,
    /// *Opaque value.* The lane's counter-mode RNG key. Not an address — it
    /// sits next to several, and swapping it with one preserves every size.
    pub rng_state: u64,
    /// *Address.* Where the kernel writes this lane's status record.
    pub commit_slot: u64,
    /// *Address.* Optional device bitset of active rows in a ragged lane;
    /// zero means every row is active.
    pub active_row_mask: u64,
    /// *Opaque value.* Bitset over stage-local channels whose puts publish
    /// the sampled token.
    pub sample_output_channel_mask: u64,
    /// *Address.* Optional device byte mask for model rows.
    pub row_valid: u64,
    /// Row index into [`row_valid`](Self::row_valid).
    pub row_valid_offset: u32,
    /// Must be zero; the drivers treat a non-zero reserved word as a corrupt
    /// table.
    pub reserved0: u32,
}

/// One stage-local channel's per-lane state: the cells a put/take touches and
/// the ring tickets the kernel checks.
///
/// Mirror of the compiler's `LaneChannelSlot`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct ChannelSlot {
    /// *Address.* The cell a `take`/`read` reads from.
    pub committed_cell: u64,
    /// *Address.* The cell a `put` writes before commit advances the ring.
    pub pending_cell: u64,
    /// *Ticket.* The ring head the host observed when it built the table;
    /// the kernel compares and refuses a stale table, never dereferences.
    /// [`NO_TICKET`](crate::NO_TICKET) means "not consuming".
    pub expected_head: u64,
    /// *Ticket.* The tail counterpart of
    /// [`expected_head`](Self::expected_head); `NO_TICKET` means "not
    /// publishing".
    pub expected_tail: u64,
}

/// Per-`(lane, channel)` grouped-dispatch metadata, one per [`ChannelSlot`].
///
/// The grouped effect kernels read the ring through this rather than through
/// per-program bindings: `words` is where the ring's head/tail live, and
/// `flags` carries the channel-effect bits
/// ([`channel_flags`](crate::channel_flags) writes them,
/// `CHANNEL_VALID` through `CHANNEL_RETRY_INELIGIBLE`).
///
/// Declared only as MSL text on the kernel side — the compiler's own preamble
/// comment concedes there is "nothing to pin them to". This struct is the
/// something; the drift test holds the MSL declaration against it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct ChannelMeta {
    /// *Address.* The channel's head/tail word pair.
    pub words: u64,
    /// The ring's capacity in cells.
    pub capacity: u32,
    /// `CHANNEL_*` effect bits for this lane's use of the channel.
    pub flags: u32,
}

/// One grouped stage's launch geometry, read by every kernel in the stage.
///
/// MSL declares the last three fields `reserved0`/`reserved1`/`reserved2`,
/// and the C++ host fills them through a field literally named `reserved` —
/// but the kernels read all three. They are named here for what they are; the
/// offsets are identical and tested.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct GroupLayout {
    /// Lanes dispatched in this stage group.
    pub lane_count: u32,
    /// Values per lane; the stride of the per-lane descriptor block.
    pub value_count: u32,
    /// Bytes of scratch per lane; lane `n`'s scratch starts at
    /// `n * scratch_stride`.
    pub scratch_stride: u32,
    /// Offset of the shared temporary region within a lane's scratch.
    pub temporary_offset: u32,
    /// The model's vocabulary width.
    pub vocab: u32,
    /// MSL `reserved0`: channel bindings per lane. The kernels index
    /// `channel_bindings[dispatch_lane * binding_stride + slot]`.
    pub binding_stride: u32,
    /// MSL `reserved1`: rows per lane in the parallel-selection grid. The
    /// nucleus/top-k kernels recover `(lane, row)` from the flat threadgroup
    /// position by dividing by this; zero disables the parallel path.
    pub rows_per_lane: u32,
    /// MSL `reserved2`: ops per lane; the stride of the per-lane
    /// `DeviceOpParams` block.
    pub op_stride: u32,
}

/// One lane's window into the grouped logits-row index array.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct RowMeta {
    /// First entry in the flat row-index array belonging to this lane.
    pub offset: u32,
    /// Entries belonging to this lane.
    pub count: u32,
    /// The lane's MTP draft-row base, floored at zero.
    pub mtp_offset: u32,
    /// Must be zero.
    pub reserved: u32,
}

/// Bytes of one [`Header`].
pub const HEADER_BYTES: u64 = size_of::<Header>() as u64;
/// Bytes of one [`Record`].
pub const RECORD_BYTES: u64 = size_of::<Record>() as u64;
/// Bytes of one [`ChannelSlot`].
pub const SLOT_BYTES: u64 = size_of::<ChannelSlot>() as u64;

/// The dimensions of one lane table, and the offset arithmetic the C++ walked
/// by hand at two call sites.
///
/// The single-lane M1 fire is `Shape::of(1, channels)`; the grouped path is
/// `Shape::of(lanes, channel_stride)`. Every method is checked: an offset
/// this type returns is inside a buffer of [`bytes`](Shape::bytes), and a
/// lane or slot index past the shape is `None` rather than the neighbouring
/// array misread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Lanes in the table.
    pub lanes: u32,
    /// [`ChannelSlot`]s per lane.
    pub channel_slots_per_lane: u32,
}

impl Shape {
    /// The shape with `lanes` lanes of `channel_slots_per_lane` slots each.
    #[must_use]
    pub const fn of(lanes: u32, channel_slots_per_lane: u32) -> Self {
        Self {
            lanes,
            channel_slots_per_lane,
        }
    }

    /// Total bytes of the table: the header, `lanes` records, then the flat
    /// slot array.
    ///
    /// `None` when the product leaves `u64` — a table no allocator could
    /// satisfy, refused here rather than wrapped into a small allocation the
    /// writes would then walk off the end of.
    #[must_use]
    pub fn bytes(&self) -> Option<u64> {
        let lanes = u64::from(self.lanes);
        let records = lanes.checked_mul(RECORD_BYTES)?;
        let slots = lanes
            .checked_mul(u64::from(self.channel_slots_per_lane))?
            .checked_mul(SLOT_BYTES)?;
        HEADER_BYTES.checked_add(records)?.checked_add(slots)
    }

    /// Byte offset of lane `lane`'s [`Record`].
    ///
    /// `None` when `lane` is not in the table. The record array and the slot
    /// array are contiguous, so the C++'s unchecked `records[lane]` for an
    /// out-of-range lane does not fault — it reads channel slots
    /// reinterpreted as a lane record.
    #[must_use]
    pub fn record_offset(&self, lane: u32) -> Option<u64> {
        if lane >= self.lanes {
            return None;
        }
        HEADER_BYTES.checked_add(u64::from(lane).checked_mul(RECORD_BYTES)?)
    }

    /// Byte offset of the start of the flat [`ChannelSlot`] array.
    #[must_use]
    pub fn slots_offset(&self) -> Option<u64> {
        HEADER_BYTES.checked_add(u64::from(self.lanes).checked_mul(RECORD_BYTES)?)
    }

    /// Byte offset of slot `slot` of lane `lane` in the flat slot array.
    ///
    /// `None` when either index leaves the shape.
    #[must_use]
    pub fn slot_offset(&self, lane: u32, slot: u32) -> Option<u64> {
        if lane >= self.lanes || slot >= self.channel_slots_per_lane {
            return None;
        }
        let index = u64::from(lane)
            .checked_mul(u64::from(self.channel_slots_per_lane))?
            .checked_add(u64::from(slot))?;
        self.slots_offset()?
            .checked_add(index.checked_mul(SLOT_BYTES)?)
    }

    /// The value lane `lane`'s [`Record::channel_slot_offset`] must carry:
    /// its first index in the flat slot array.
    ///
    /// The C++ computes this as `static_cast<uint32_t>(lane * stride)` — an
    /// unchecked truncation of a `size_t` product into the `u32` the record
    /// declares. `None` when the product does not fit, or the lane is not in
    /// the table.
    #[must_use]
    pub fn slot_index(&self, lane: u32) -> Option<u32> {
        if lane >= self.lanes {
            return None;
        }
        u32::try_from(u64::from(lane) * u64::from(self.channel_slots_per_lane)).ok()
    }
}

#[cfg(test)]
mod tests {
    use core::mem::offset_of;

    use super::*;

    #[test]
    fn the_struct_sizes_match_the_static_asserts_the_cpp_makes() {
        assert_eq!(size_of::<Header>(), 16);
        assert_eq!(size_of::<Record>(), 96);
        assert_eq!(size_of::<ChannelSlot>(), 32);
        assert_eq!(size_of::<ChannelMeta>(), 16);
        assert_eq!(size_of::<GroupLayout>(), 32);
        assert_eq!(size_of::<RowMeta>(), 16);
    }

    /// The C++ asserts only sizes, and every struct here has same-width
    /// neighbours a permutation of which preserves every size. The offsets
    /// are the actual contract with the MSL declarations.
    #[test]
    fn the_sidecar_field_offsets_match_the_msl_declarations() {
        assert_eq!(offset_of!(ChannelMeta, words), 0);
        assert_eq!(offset_of!(ChannelMeta, capacity), 8);
        assert_eq!(offset_of!(ChannelMeta, flags), 12);

        assert_eq!(offset_of!(GroupLayout, lane_count), 0);
        assert_eq!(offset_of!(GroupLayout, value_count), 4);
        assert_eq!(offset_of!(GroupLayout, scratch_stride), 8);
        assert_eq!(offset_of!(GroupLayout, temporary_offset), 12);
        assert_eq!(offset_of!(GroupLayout, vocab), 16);
        // The three the MSL calls reserved0/1/2, at those exact offsets.
        assert_eq!(offset_of!(GroupLayout, binding_stride), 20);
        assert_eq!(offset_of!(GroupLayout, rows_per_lane), 24);
        assert_eq!(offset_of!(GroupLayout, op_stride), 28);

        assert_eq!(offset_of!(RowMeta, offset), 0);
        assert_eq!(offset_of!(RowMeta, count), 4);
        assert_eq!(offset_of!(RowMeta, mtp_offset), 8);
        assert_eq!(offset_of!(RowMeta, reserved), 12);
    }

    /// The compiler owns the table declarations; this mirror exists because
    /// the driver refuses a build dependency on it. Field by field, offset by
    /// offset, or the mirror has drifted.
    #[test]
    fn the_mirror_still_matches_the_compilers_lane_table() {
        use tensor_compiler::plan::lane_table as theirs;

        assert_eq!(ABI_VERSION, theirs::LANE_TABLE_ABI_VERSION);

        macro_rules! same {
            ($ours:ty, $their:ty, $($field:ident),+ $(,)?) => {
                assert_eq!(size_of::<$ours>(), size_of::<$their>());
                $(assert_eq!(
                    offset_of!($ours, $field),
                    offset_of!($their, $field),
                    concat!("field `", stringify!($field), "` has drifted"),
                );)+
            };
        }

        same!(
            Header,
            theirs::LaneTableHeader,
            abi_version,
            lane_count,
            channel_slots_per_lane,
            flags,
        );
        same!(
            Record,
            theirs::LaneRecord,
            logits_base,
            logits_row_offset,
            logits_row_count,
            kv_len,
            page_count,
            row_count,
            token_count,
            sampled_rows,
            query_len,
            key_len,
            channel_slot_offset,
            rng_state,
            commit_slot,
            active_row_mask,
            sample_output_channel_mask,
            row_valid,
            row_valid_offset,
            reserved0,
        );
        same!(
            ChannelSlot,
            theirs::LaneChannelSlot,
            committed_cell,
            pending_cell,
            expected_head,
            expected_tail,
        );
    }

    /// The sidecars exist on the kernel side only as text in the emitted
    /// preamble — the compiler's own comment concedes there is "nothing to
    /// pin them to". Holding the exact declarations against this mirror is
    /// the pin: a reordered or renamed field on either side fails here.
    #[test]
    fn the_sidecar_mirrors_still_match_the_emitted_msl_text() {
        let msl = tensor_compiler::codegen::metal::preamble::grouped_preamble();
        for declaration in [
            "struct M3ChannelMeta {\n  ulong words;\n  uint capacity;\n  uint flags;\n};",
            "struct M3GroupLayout {\n  uint lane_count;\n  uint value_count;\n  \
             uint scratch_stride;\n  uint temporary_offset;\n  uint vocab;\n  \
             uint reserved0;\n  uint reserved1;\n  uint reserved2;\n};",
            "struct M3RowMeta {\n  uint offset;\n  uint count;\n  uint mtp_offset;\n  \
             uint reserved;\n};",
        ] {
            assert!(
                msl.contains(declaration),
                "the emitted MSL no longer declares:\n{declaration}\n\
                 the sidecar mirror layout must be re-checked against the preamble"
            );
        }
    }

    #[test]
    fn a_single_lane_table_is_the_size_prepare_computes() {
        // sizeof(header) + sizeof(record) + channels * sizeof(slot).
        assert_eq!(Shape::of(1, 3).bytes(), Some(16 + 96 + 3 * 32));
        assert_eq!(Shape::of(1, 0).bytes(), Some(16 + 96));
    }

    #[test]
    fn a_grouped_table_is_the_size_the_group_walk_computes() {
        // header + lanes * record + lanes * stride * slot.
        assert_eq!(Shape::of(4, 2).bytes(), Some(16 + 4 * 96 + 4 * 2 * 32));
    }

    #[test]
    fn record_and_slot_offsets_tile_the_table_without_overlap() {
        let shape = Shape::of(4, 2);
        assert_eq!(shape.record_offset(0), Some(16));
        assert_eq!(shape.record_offset(3), Some(16 + 3 * 96));
        assert_eq!(shape.slots_offset(), Some(16 + 4 * 96));
        assert_eq!(shape.slot_offset(0, 0), shape.slots_offset());
        assert_eq!(
            shape.slot_offset(3, 1),
            Some(16 + 4 * 96 + (3 * 2 + 1) * 32)
        );
        // The last slot ends exactly at the table's end.
        assert_eq!(
            shape.slot_offset(3, 1).map(|o| o + SLOT_BYTES),
            shape.bytes()
        );
    }

    #[test]
    fn a_lane_past_the_table_is_refused_because_the_next_bytes_are_slots() {
        let shape = Shape::of(2, 3);
        assert_eq!(shape.record_offset(2), None);
        assert_eq!(shape.slot_offset(2, 0), None);
        assert_eq!(shape.slot_offset(0, 3), None);
        assert_eq!(shape.slot_index(2), None);
    }

    #[test]
    fn slot_index_is_the_records_channel_slot_offset() {
        let shape = Shape::of(64, 29);
        assert_eq!(shape.slot_index(0), Some(0));
        assert_eq!(shape.slot_index(63), Some(63 * 29));
    }

    #[test]
    fn a_shape_too_large_for_u64_is_refused_rather_than_wrapped() {
        assert_eq!(Shape::of(u32::MAX, u32::MAX).bytes(), None);
    }
}
