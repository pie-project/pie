//! The device records, in CUDA's spelling.
//!
//! # Why this file exists at all
//!
//! [`driver_pipeline`] mirrors three structs the emitted kernels read:
//! `Status` (16 bytes), `ValueDesc` (36) and `OpParams` (64). Two of those are
//! byte-identical to what CUDA's generated kernels declare and cross as
//! themselves. The third is not, and the difference is not a detail:
//!
//! | record | Metal / `driver-pipeline` | CUDA `M1OpParams` |
//! |---|---|---|
//! | `Status` | 16 B | 16 B — identical |
//! | `ValueDesc` | 36 B | 36 B — identical |
//! | `OpParams` | **64 B**, sixteen `u32` | **88 B**, twenty `u32` + one `u64` |
//!
//! `driver-pipeline`'s own comment says why it is sixteen: *"Sixteen words is
//! the agreement with the emitted kernels."* That is Metal's agreement. CUDA's
//! `ptir_m1_runtime_prologue.cuh` declares the same sixteen words **in the
//! same order** and then four more — `intrinsic_dtype`, `bool_storage`,
//! `intrinsic_row_stride`, `intrinsic_row_offset` — followed by a `u64`
//! `rng_seed`, whose alignment is what makes the record 88 and not 84.
//!
//! # Why widening, and not a second mirror in the shared crate
//!
//! The shared crate could have carried the union of both layouts, and that
//! would be worse. `OpParams::of` is where an op's *meaning* is decided —
//! which slot is the predicate, what a result-less op points its output at —
//! and that reasoning is identical on both backends and belongs in one place.
//! The extra five fields are not meaning; they are CUDA's binding of the
//! intrinsic side tables. So the shared crate keeps the decision, and this
//! file keeps the encoding, and [`CudaOpParams::widen`] is the seam.
//!
//! # The hazard this is written against
//!
//! Two records whose first sixteen words agree and whose sizes differ is the
//! worst shape a layout bug can take: writing 64 bytes into an 88-byte stride
//! leaves every op after the first reading its predecessor's tail as its own
//! head, and every field is a plausible small integer. Nothing faults. The
//! kernel samples a wrong token and the model still speaks fluently. So the
//! offsets here are asserted individually rather than the size being asserted
//! once — a size check passes under any permutation of same-width fields.

use driver_pipeline::OpParams;

/// One op's parameters, in the layout CUDA's generated kernels read.
///
/// `#[repr(C)]` and field-for-field with `M1OpParams`. The order is the
/// header's and may not be tidied: the kernels index this by offset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CudaOpParams {
    /// `PTIR_OP_*`.
    pub tag: u32,
    /// First argument's value slot.
    pub a0: u32,
    /// Second argument's value slot, or `pivot_threshold`'s predicate.
    pub a1: u32,
    /// Third argument's value slot.
    pub a2: u32,
    /// First result's value slot, or `a0` for an op with no results.
    pub o0: u32,
    /// Second result's value slot, or `o0` for an op with fewer than two.
    pub o1: u32,
    /// The op's immediate, or the vocabulary for an intrinsic.
    pub imm: u32,
    /// The op's second immediate, or the MTP draft row.
    pub imm2: u32,
    /// The op's third immediate.
    pub imm3: u32,
    /// RNG kind: 0 uniform, 1 gumbel.
    pub kind: u32,
    /// `pivot_threshold`'s predicate tag.
    pub pred_tag: u32,
    /// A const literal's dtype.
    pub lit_dtype: u32,
    /// A const literal's raw bits.
    pub lit_bits: u32,
    /// The channel slot a channel op targets.
    pub channel_slot: u32,
    /// `PTIR_INTR_*`, for `intrinsic_val`.
    pub intr: u32,
    /// The fixed cell size a `chan_put` writes into.
    pub sink_bytes: u32,

    // ── Past here is CUDA's alone; `driver-pipeline`'s record ends above. ──
    /// How the intrinsic's buffer stores its elements:
    /// [`INTRINSIC_STORAGE_F32`] or [`INTRINSIC_STORAGE_RAW_BF16`].
    ///
    /// The driver's, not the program's: the same `intrinsic_val` op reads f32
    /// logits from one deployment and raw bf16 from another, and which it is
    /// is a fact about the buffer the fire binds, not about the trace.
    pub intrinsic_dtype: u32,
    /// How a bool cell is stored: [`BOOL_STORAGE_NATIVE_BYTES`] or
    /// [`BOOL_STORAGE_WIRE_PACKED`].
    ///
    /// The C++ writes a bare `0` here for every op with the comment that the
    /// device side is always native bytes — the bit-packing happens at the
    /// host boundary, on the way into the pinned mirror. Named rather than
    /// zeroed so that a future packed path has somewhere to say so.
    pub bool_storage: u32,
    /// Elements between one row and the next in the intrinsic's buffer.
    ///
    /// The vocabulary for logits, which is what makes a lane read *its* row.
    pub intrinsic_row_stride: u32,
    /// Which row of the intrinsic's buffer this op reads.
    pub intrinsic_row_offset: u32,
    /// The per-op RNG seed.
    ///
    /// A `u64`, and it is the reason the record is 88 bytes rather than 84:
    /// it forces eight-byte alignment, so the compiler pads after
    /// `intrinsic_row_offset`. A port that laid the fields out by hand and
    /// summed their widths would get 84 and shift every op after the first.
    pub rng_seed: u64,
}

/// `IntrinsicStorageMode::F32` — the buffer holds `f32` elements.
pub const INTRINSIC_STORAGE_F32: u32 = 0;
/// `IntrinsicStorageMode::RawBf16` — the buffer holds raw `bf16` elements the
/// kernel widens as it reads.
pub const INTRINSIC_STORAGE_RAW_BF16: u32 = 1;
/// `BoolStorageMode::NativeBytes` — one byte per lane, which is what every
/// device-side bool cell is.
pub const BOOL_STORAGE_NATIVE_BYTES: u32 = 0;
/// `BoolStorageMode::WirePacked` — one bit per lane, which is what a bool cell
/// becomes on the way to the host mirror.
pub const BOOL_STORAGE_WIRE_PACKED: u32 = 1;

/// The record's size, as `ptir_m1_runtime_prologue.cuh` asserts it.
const _: () = assert!(size_of::<CudaOpParams>() == 88);

/// Every field's offset, pinned individually.
///
/// Not one size assertion. `sizeof == 88` holds under any permutation of the
/// twenty `u32`s, and a permutation is exactly what a hand-transcribed field
/// list produces — swapping `channel_slot` with `intr` costs nothing at
/// compile time and misdirects every channel op at run time.
const _: () = {
    assert!(std::mem::offset_of!(CudaOpParams, tag) == 0);
    assert!(std::mem::offset_of!(CudaOpParams, a0) == 4);
    assert!(std::mem::offset_of!(CudaOpParams, a1) == 8);
    assert!(std::mem::offset_of!(CudaOpParams, a2) == 12);
    assert!(std::mem::offset_of!(CudaOpParams, o0) == 16);
    assert!(std::mem::offset_of!(CudaOpParams, o1) == 20);
    assert!(std::mem::offset_of!(CudaOpParams, imm) == 24);
    assert!(std::mem::offset_of!(CudaOpParams, imm2) == 28);
    assert!(std::mem::offset_of!(CudaOpParams, imm3) == 32);
    assert!(std::mem::offset_of!(CudaOpParams, kind) == 36);
    assert!(std::mem::offset_of!(CudaOpParams, pred_tag) == 40);
    assert!(std::mem::offset_of!(CudaOpParams, lit_dtype) == 44);
    assert!(std::mem::offset_of!(CudaOpParams, lit_bits) == 48);
    assert!(std::mem::offset_of!(CudaOpParams, channel_slot) == 52);
    assert!(std::mem::offset_of!(CudaOpParams, intr) == 56);
    assert!(std::mem::offset_of!(CudaOpParams, sink_bytes) == 60);
    assert!(std::mem::offset_of!(CudaOpParams, intrinsic_dtype) == 64);
    assert!(std::mem::offset_of!(CudaOpParams, bool_storage) == 68);
    assert!(std::mem::offset_of!(CudaOpParams, intrinsic_row_stride) == 72);
    assert!(std::mem::offset_of!(CudaOpParams, intrinsic_row_offset) == 76);
    // 80..84 is the padding the `u64` forces. Asserting 84 here would be the
    // bug this whole block exists to catch.
    assert!(std::mem::offset_of!(CudaOpParams, rng_seed) == 80);
};

impl CudaOpParams {
    /// The shared record, widened; CUDA's five extra fields left at their
    /// defaults for the caller to fill.
    ///
    /// The first sixteen words are copied by NAME rather than by a
    /// transmute of the 64-byte prefix. A transmute would be correct today
    /// and would keep being "correct" after a field was inserted in the
    /// middle of `driver_pipeline::OpParams` — silently, with every op after
    /// the insertion point reading a neighbour. Copying by name makes that a
    /// compile error.
    #[must_use]
    pub const fn widen(shared: OpParams) -> Self {
        Self {
            tag: shared.tag,
            a0: shared.a0,
            a1: shared.a1,
            a2: shared.a2,
            o0: shared.o0,
            o1: shared.o1,
            imm: shared.imm,
            imm2: shared.imm2,
            imm3: shared.imm3,
            kind: shared.kind,
            pred_tag: shared.pred_tag,
            lit_dtype: shared.lit_dtype,
            lit_bits: shared.lit_bits,
            channel_slot: shared.channel_slot,
            intr: shared.intr,
            sink_bytes: shared.sink_bytes,
            intrinsic_dtype: INTRINSIC_STORAGE_F32,
            bool_storage: BOOL_STORAGE_NATIVE_BYTES,
            intrinsic_row_stride: 0,
            intrinsic_row_offset: 0,
            rng_seed: 0,
        }
    }

    /// Bind this op to a row of an intrinsic's buffer.
    ///
    /// Separate from [`Self::widen`] because it is the only part of the record
    /// that depends on the FIRE rather than on the program: the same op reads
    /// row 3 of one launch's logits and row 0 of the next.
    #[must_use]
    pub const fn with_intrinsic(mut self, dtype: u32, row_stride: u32, row_offset: u32) -> Self {
        self.intrinsic_dtype = dtype;
        self.intrinsic_row_stride = row_stride;
        self.intrinsic_row_offset = row_offset;
        self
    }

    /// This record as the bytes a device upload copies.
    ///
    /// # Safety of the cast
    ///
    /// `#[repr(C)]` over twenty `u32` and a `u64` has no padding except the
    /// four bytes before `rng_seed`, and those are initialised — the struct is
    /// only ever built by [`Self::widen`], which writes every field, so no
    /// uninitialised byte is read.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `Self` is `#[repr(C)]`, `Copy`, and contains no padding that
        // was not written by a field store; reading it as bytes reads only
        // initialised memory, and the lifetime is the borrow's.
        unsafe {
            std::slice::from_raw_parts(std::ptr::from_ref(self).cast::<u8>(), size_of::<Self>())
        }
    }
}

/// A slice of records as the flat bytes one upload copies.
#[must_use]
pub fn params_bytes(params: &[CudaOpParams]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_of_val(params));
    for record in params {
        bytes.extend_from_slice(record.as_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use driver_pipeline::{STATUS_BYTES, ValueDesc};

    /// The two records that ARE byte-identical, asserted here rather than
    /// assumed. If either ever diverges the way `OpParams` did, this is where
    /// it is found out — and the remedy would be another widening, not a
    /// silently wrong upload.
    #[test]
    fn the_status_and_value_records_need_no_widening() {
        assert_eq!(STATUS_BYTES, 16, "M1Status is 16 bytes");
        assert_eq!(size_of::<ValueDesc>(), 36, "M1ValueDesc is 36 bytes");
    }

    /// The record CUDA's kernels index. A port that summed the field widths
    /// by hand would get 84 and shift every op after the first.
    #[test]
    fn the_cuda_record_is_eighty_eight_bytes_because_of_the_u64() {
        assert_eq!(size_of::<CudaOpParams>(), 88);
        assert_eq!(align_of::<CudaOpParams>(), 8, "the u64 sets the alignment");
        assert_eq!(
            20 * size_of::<u32>() + size_of::<u64>(),
            88,
            "twenty words and a u64 sum to 88 only because the padding before \
             the u64 is exactly four bytes"
        );
    }

    /// Widening must move every one of the sixteen shared words into the field
    /// of the same name. Each is given a DISTINCT value, so a transposition
    /// fails rather than passing because two neighbours happened to agree.
    #[test]
    fn widening_carries_all_sixteen_shared_words_to_their_own_fields() {
        let shared = OpParams {
            tag: 1,
            a0: 2,
            a1: 3,
            a2: 4,
            o0: 5,
            o1: 6,
            imm: 7,
            imm2: 8,
            imm3: 9,
            kind: 10,
            pred_tag: 11,
            lit_dtype: 12,
            lit_bits: 13,
            channel_slot: 14,
            intr: 15,
            sink_bytes: 16,
        };
        let cuda = CudaOpParams::widen(shared);
        assert_eq!(
            (
                cuda.tag, cuda.a0, cuda.a1, cuda.a2, cuda.o0, cuda.o1, cuda.imm, cuda.imm2
            ),
            (1, 2, 3, 4, 5, 6, 7, 8)
        );
        assert_eq!(
            (
                cuda.imm3,
                cuda.kind,
                cuda.pred_tag,
                cuda.lit_dtype,
                cuda.lit_bits,
                cuda.channel_slot,
                cuda.intr,
                cuda.sink_bytes
            ),
            (9, 10, 11, 12, 13, 14, 15, 16)
        );
    }

    /// The five CUDA-only fields start at the values the C++ writes for an op
    /// that binds no intrinsic: native bool bytes, f32 storage, no row.
    #[test]
    fn the_cuda_only_fields_default_to_what_the_cpp_writes() {
        let cuda = CudaOpParams::widen(OpParams::default());
        assert_eq!(cuda.intrinsic_dtype, INTRINSIC_STORAGE_F32);
        assert_eq!(cuda.bool_storage, BOOL_STORAGE_NATIVE_BYTES);
        assert_eq!(cuda.intrinsic_row_stride, 0);
        assert_eq!(cuda.intrinsic_row_offset, 0);
        assert_eq!(cuda.rng_seed, 0);
    }

    /// The bytes a lane reads are at `index * 88`. This is the assertion that
    /// would have caught a 64-byte stride: it checks that the SECOND record
    /// starts where the kernel will look for it, which a per-record test
    /// cannot.
    #[test]
    fn records_are_packed_at_the_stride_the_kernel_indexes() {
        let params = vec![
            CudaOpParams::widen(OpParams {
                tag: 0xAA,
                ..OpParams::default()
            }),
            CudaOpParams::widen(OpParams {
                tag: 0xBB,
                ..OpParams::default()
            }),
        ];
        let bytes = params_bytes(&params);
        assert_eq!(bytes.len(), 176);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 0xAA);
        assert_eq!(
            u32::from_le_bytes(bytes[88..92].try_into().unwrap()),
            0xBB,
            "the second op must begin at byte 88; at 64 the kernel would read \
             the first record's tail as the second's head, and every field \
             would still be a plausible small integer"
        );
    }

    /// The intrinsic binding is per-fire, so it must be settable without
    /// rebuilding the shared half.
    #[test]
    fn an_intrinsic_binding_can_be_attached_after_widening() {
        let cuda = CudaOpParams::widen(OpParams {
            intr: 3,
            ..OpParams::default()
        })
        .with_intrinsic(INTRINSIC_STORAGE_RAW_BF16, 128, 7);
        assert_eq!(cuda.intr, 3, "the program's half is untouched");
        assert_eq!(cuda.intrinsic_dtype, INTRINSIC_STORAGE_RAW_BF16);
        assert_eq!(cuda.intrinsic_row_stride, 128);
        assert_eq!(cuda.intrinsic_row_offset, 7);
    }
}
