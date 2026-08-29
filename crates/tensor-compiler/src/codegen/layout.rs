//! The device-side struct layouts, declared once.
//!
//! The lane table is the ABI between the host planner and the kernels it
//! emits, and it was written out seven times. Nothing tied the copies
//! together. Adding a field meant editing all seven, and getting it wrong did
//! not fail to compile — it shifted every field after it, so the kernel read a
//! different offset than the host wrote. Wrong numbers, no error.
//!
//! Here the field list is data, and each copy is either printed from it or
//! read back and compared against it:
//!
//! | copy | how it is tied here |
//! |---|---|
//! | `#[repr(C)]` structs in `tensor-compiler` | `offset_of!` in `static_assertions` |
//! | MSL `M1*` in `metal::preamble` | printed by [`DeviceStruct::emit_msl`] |
//! | MSL `M3*` in `metal::preamble` | printed by [`DeviceStruct::emit_msl`] |
//! | MSL `M1Status` in the effect emitters | printed by [`DeviceStruct::emit_msl`] |
//! | `runtime/cuda/fused_block0.cuh` | [`DeviceStruct::emit_cuda`], compared in `cuda::fused` |
//! | `runtime/metal/ptir_m1_grouped.metal` | `metal::preamble::tests::file_matches_emitted_text` |
//!
//! The two runtime files are hand-written C++/MSL that cannot be generated
//! wholesale — they are compiled by NVRTC and the Metal compiler as text and
//! carry far more than the struct declarations — so they are checked rather
//! than produced. A seventh row stood above until the C++ drivers were
//! deleted: a plain-C printing (`DeviceStruct::emit_c`) into the generated
//! `ptir_abi.h`, which those engines `#include`d instead of retyping the
//! table. The Rust engines that replaced them read this module directly, and a
//! printer whose only consumer was a deleted header is not a tie to anything.
//!
//! A field added on one side and not the other is now a compile error or a
//! test failure rather than a silent reinterpretation.
//!
//! The emitted text is byte-identical to what was hand-written, which the MSL
//! goldens (and the `# @grouped:` length+hash pin in `emit_grouped_*.txt`)
//! check on every run. Those bytes came from the deleted C++ oracle, so they
//! are an independent witness that this generator reproduces the ABI the
//! engine was built against.

use alloc::format;
use alloc::string::String;

/// The two scalar widths the lane table uses. Addresses are `u64` on both
/// supported backends, so there is no pointer-shaped case.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FieldType {
    /// A 32-bit unsigned field (`uint32_t` / `uint` / `m1_u32`).
    U32,
    /// A 64-bit unsigned field (`uint64_t` / `ulong` / `m1_u64`); also every
    /// address, since both backends use 64-bit pointers.
    U64,
}

impl FieldType {
    /// Spelling in Metal Shading Language.
    const fn msl(self) -> &'static str {
        match self {
            FieldType::U32 => "uint",
            FieldType::U64 => "ulong",
        }
    }

    /// Spelling in the CUDA runtime headers, which typedef their own widths
    /// rather than including `<cstdint>` — NVRTC compiles these as a string.
    const fn cuda(self) -> &'static str {
        match self {
            FieldType::U32 => "m1_u32",
            FieldType::U64 => "m1_u64",
        }
    }

    /// Size in bytes. Used by the layout self-check, not by the emitters.
    pub const fn size(self) -> usize {
        match self {
            FieldType::U32 => 4,
            FieldType::U64 => 8,
        }
    }
}

/// One field of a device struct: its C name, its MSL spelling, and its scalar
/// width.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    /// The field's name in the generated C header and in `tensor-compiler`'s
    /// `#[repr(C)]` struct.
    pub name: &'static str,
    /// The name MSL uses, when it differs. It differs in exactly one place —
    /// see [`LANE_TABLE_HEADER`] — and carrying the difference here is what
    /// lets the generator reproduce the existing bytes instead of quietly
    /// renaming a field the goldens and the engine already agree on.
    pub msl_name: &'static str,
    /// The field's scalar width.
    pub ty: FieldType,
}

impl Field {
    const fn u32(name: &'static str) -> Self {
        Field {
            name,
            msl_name: name,
            ty: FieldType::U32,
        }
    }

    const fn u64(name: &'static str) -> Self {
        Field {
            name,
            msl_name: name,
            ty: FieldType::U64,
        }
    }

    const fn renamed_u32(name: &'static str, msl_name: &'static str) -> Self {
        Field {
            name,
            msl_name,
            ty: FieldType::U32,
        }
    }
}

/// How a struct is printed in MSL. Cosmetic, but the goldens record the exact
/// bytes, so the choice has to be recorded too.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MslStyle {
    /// All fields on the declaration line.
    Inline,
    /// One field per line, two-space indent.
    Block,
}

/// One device struct: its name in each dialect, its MSL layout style, and its
/// ordered field list.
#[derive(Clone, Copy, Debug)]
pub struct DeviceStruct {
    /// Name in the generated C header, `Ptir`-prefixed.
    pub c_name: &'static str,
    /// Name in MSL, minus the `M1`/`M3` prefix the caller supplies. The two
    /// dialects do not agree on every suffix (`LaneTableHeader` is
    /// `LaneHeader` in MSL), so both are spelled out.
    pub msl_suffix: &'static str,
    /// How the MSL printer lays the fields out.
    pub msl_style: MslStyle,
    /// The struct's fields in declaration order; offsets follow from their
    /// widths.
    pub fields: &'static [Field],
}

impl DeviceStruct {
    /// `struct M1Name { ... };` followed by a newline.
    pub fn emit_msl(&self, prefix: &str) -> String {
        let name = format!("{prefix}{}", self.msl_suffix);
        match self.msl_style {
            MslStyle::Inline => {
                let mut out = format!("struct {name} {{");
                for field in self.fields {
                    out.push_str(&format!(" {} {};", field.ty.msl(), field.msl_name));
                }
                out.push_str(" };\n");
                out
            }
            MslStyle::Block => {
                let mut out = format!("struct {name} {{\n");
                for field in self.fields {
                    out.push_str(&format!("  {} {};\n", field.ty.msl(), field.msl_name));
                }
                out.push_str("};\n");
                out
            }
        }
    }

    /// `struct Name { ... };` in the CUDA runtime's dialect: the C names and
    /// declaration order, the CUDA width spellings, two-space indent.
    pub fn emit_cuda(&self) -> String {
        let mut out = format!("struct {} {{\n", self.c_name);
        for field in self.fields {
            out.push_str(&format!("  {} {};\n", field.ty.cuda(), field.name));
        }
        out.push_str("};\n");
        out
    }

    /// `static_assert(sizeof(Name) == N, "...");` — the size check the CUDA
    /// runtime carries. It is not enough on its own (reordering two fields of
    /// equal width keeps the size), which is why `emit_cuda` exists.
    pub fn emit_cuda_size_assert(&self, note: &str) -> String {
        format!(
            "static_assert(sizeof({}) == {}, \"{note}\");\n",
            self.c_name,
            self.size_bytes()
        )
    }

    /// Total size with the natural alignment, from the same walk `offsets`
    /// does.
    pub fn size_bytes(&self) -> usize {
        let mut end = 0usize;
        let mut alignment = 1usize;
        for field in self.fields {
            let size = field.ty.size();
            alignment = alignment.max(size);
            end = end.next_multiple_of(size) + size;
        }
        end.next_multiple_of(alignment)
    }

    /// Byte offset of each field, assuming the natural C alignment both
    /// backends use. Only meaningful because every field is 4 or 8 bytes and
    /// the declarations are already ordered to avoid padding; the
    /// `offset_of!` assertions below are what prove that.
    pub fn offsets(&self) -> impl Iterator<Item = (&'static str, usize)> + '_ {
        let mut offset = 0usize;
        self.fields.iter().map(move |field| {
            let size = field.ty.size();
            offset = offset.next_multiple_of(size);
            let at = offset;
            offset += size;
            (field.name, at)
        })
    }
}

/// The status word a lane's commit slot points at. Written by the readiness
/// and commit kernels, read by the engine. It has no `tensor-compiler` counterpart —
/// the host never builds one, it only hands out the address — so it is pinned
/// by the goldens alone.
pub const STATUS: DeviceStruct = DeviceStruct {
    c_name: "PtirStatus",
    msl_suffix: "Status",
    msl_style: MslStyle::Inline,
    fields: &[
        Field::u32("state"),
        Field::u32("fault"),
        Field::u32("reserved0"),
        Field::u32("reserved1"),
    ],
};

/// Header of the grouped-dispatch lane table. MSL calls the third field
/// `channel_count`; the host and the C header call it
/// `channel_slots_per_lane`. Same offset, same width, different word — a drift
/// that survived precisely because nothing compared the copies.
pub const LANE_TABLE_HEADER: DeviceStruct = DeviceStruct {
    c_name: "PtirLaneTableHeader",
    msl_suffix: "LaneHeader",
    msl_style: MslStyle::Inline,
    fields: &[
        Field::u32("abi_version"),
        Field::u32("lane_count"),
        Field::renamed_u32("channel_slots_per_lane", "channel_count"),
        Field::u32("flags"),
    ],
};

/// One lane's worth of dispatch state.
pub const LANE_RECORD: DeviceStruct = DeviceStruct {
    c_name: "PtirLaneRecord",
    msl_suffix: "LaneRecord",
    msl_style: MslStyle::Block,
    fields: &[
        Field::u64("logits_base"),
        Field::u32("logits_row_offset"),
        Field::u32("logits_row_count"),
        Field::u32("kv_len"),
        Field::u32("page_count"),
        Field::u32("row_count"),
        Field::u32("token_count"),
        Field::u32("sampled_rows"),
        Field::u32("query_len"),
        Field::u32("key_len"),
        Field::u32("channel_slot_offset"),
        Field::u64("rng_state"),
        Field::u64("commit_slot"),
        Field::u64("active_row_mask"),
        Field::u64("sample_output_channel_mask"),
        Field::u64("row_valid"),
        Field::u32("row_valid_offset"),
        Field::u32("reserved0"),
    ],
};

/// One channel slot within a lane.
pub const LANE_CHANNEL_SLOT: DeviceStruct = DeviceStruct {
    c_name: "PtirLaneChannelSlot",
    msl_suffix: "LaneChannelSlot",
    msl_style: MslStyle::Block,
    fields: &[
        Field::u64("committed_cell"),
        Field::u64("pending_cell"),
        Field::u64("expected_head"),
        Field::u64("expected_tail"),
    ],
};

/// The three structs the host also builds, in the order the C header declares
/// them.
pub const HOST_SHARED: &[DeviceStruct] = &[LANE_TABLE_HEADER, LANE_RECORD, LANE_CHANNEL_SLOT];

/// Compile-time proof that this table describes the `tensor-compiler` structs the
/// host actually writes.
///
/// Without these, a field added to `LaneRecord` and not to [`LANE_RECORD`]
/// would compile, emit a kernel that reads the old layout, and produce wrong
/// numbers at run time. `offset_of!` is a constant, so the mismatch is caught
/// before anything runs.
mod static_assertions {
    use super::*;
    use crate::plan::{LaneChannelSlot, LaneRecord, LaneTableHeader};
    use core::mem::{offset_of, size_of};

    /// Byte offset of field `index`, laid out the way both backends lay out
    /// a `#[repr(C)]` struct of 4- and 8-byte scalars.
    const fn field_offset(table: &DeviceStruct, index: usize) -> usize {
        let mut offset = 0usize;
        let mut i = 0usize;
        while i <= index {
            let size = table.fields[i].ty.size();
            offset = offset.next_multiple_of(size);
            if i < index {
                offset += size;
            }
            i += 1;
        }
        offset
    }

    /// Total size including trailing padding to the widest member.
    const fn table_size(table: &DeviceStruct) -> usize {
        let mut offset = 0usize;
        let mut align = 1usize;
        let mut i = 0usize;
        while i < table.fields.len() {
            let size = table.fields[i].ty.size();
            if size > align {
                align = size;
            }
            offset = offset.next_multiple_of(size);
            offset += size;
            i += 1;
        }
        offset.next_multiple_of(align)
    }

    /// `$rust` must have exactly the fields `$table` lists, in order, at the
    /// offsets `$table` implies, and no others.
    macro_rules! pin_layout {
        ($rust:ty, $table:expr, $($field:ident),+ $(,)?) => {
            const _: () = {
                let mut index = 0usize;
                $(
                    assert!(index < $table.fields.len(), "layout table is missing a field");
                    assert!(
                        offset_of!($rust, $field) == field_offset(&$table, index),
                        "field offset disagrees with the layout table",
                    );
                    index += 1;
                )+
                assert!(index == $table.fields.len(), "layout table has extra fields");
                // Catches a field appended past the last one the table knows
                // about, which the per-field offsets alone would not see.
                assert!(
                    size_of::<$rust>() == table_size(&$table),
                    "struct size disagrees with the layout table",
                );
            };
        };
    }

    pin_layout!(
        LaneTableHeader,
        LANE_TABLE_HEADER,
        abi_version,
        lane_count,
        channel_slots_per_lane,
        flags,
    );

    pin_layout!(
        LaneRecord,
        LANE_RECORD,
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

    pin_layout!(
        LaneChannelSlot,
        LANE_CHANNEL_SLOT,
        committed_cell,
        pending_cell,
        expected_head,
        expected_tail,
    );
}
