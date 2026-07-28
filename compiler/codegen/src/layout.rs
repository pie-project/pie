//! The device-side struct layouts, declared once.
//!
//! The lane table is the ABI between the host planner and the kernels it
//! emits, and it used to be written out five times: as `#[repr(C)]` structs in
//! `pie-plan`, as C text in [`crate::header`], and as MSL text three times over
//! in `metal::preamble` (`M1*`, `M3*`, and `M1Status` again in two effect
//! emitters). Nothing tied the copies together. Adding a field meant editing
//! five places, and getting it wrong did not fail to compile — it shifted every
//! field after it, so the kernel read a different offset than the host wrote.
//! Wrong numbers, no error.
//!
//! Here the field list is data. The C and MSL texts are printed from it, and
//! [`static_assertions`] ties it back to the `pie-plan` structs with
//! `offset_of!`, so a field added on one side and not the other is a
//! compile error rather than a silent reinterpretation.
//!
//! The emitted text is byte-identical to what was hand-written, which the MSL
//! goldens (and the `# @grouped:` length+hash pin in `emit_grouped_*.txt`)
//! check on every run. Those bytes came from the deleted C++ oracle, so they
//! are an independent witness that this generator reproduces the ABI the
//! driver was built against.

use alloc::format;
use alloc::string::String;

/// The two scalar widths the lane table uses. Addresses are `u64` on both
/// supported backends, so there is no pointer-shaped case.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FieldType {
    U32,
    U64,
}

impl FieldType {
    /// Spelling in the generated C header.
    const fn c(self) -> &'static str {
        match self {
            FieldType::U32 => "uint32_t",
            FieldType::U64 => "uint64_t",
        }
    }

    /// Spelling in Metal Shading Language.
    const fn msl(self) -> &'static str {
        match self {
            FieldType::U32 => "uint",
            FieldType::U64 => "ulong",
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

#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub name: &'static str,
    /// The name MSL uses, when it differs. It differs in exactly one place —
    /// see [`LANE_TABLE_HEADER`] — and carrying the difference here is what
    /// lets the generator reproduce the existing bytes instead of quietly
    /// renaming a field the goldens and the driver already agree on.
    pub msl_name: &'static str,
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

#[derive(Clone, Copy, Debug)]
pub struct DeviceStruct {
    /// Name in the generated C header, `Ptir`-prefixed.
    pub c_name: &'static str,
    /// Name in MSL, minus the `M1`/`M3` prefix the caller supplies. The two
    /// dialects do not agree on every suffix (`LaneTableHeader` is
    /// `LaneHeader` in MSL), so both are spelled out.
    pub msl_suffix: &'static str,
    pub msl_style: MslStyle,
    pub fields: &'static [Field],
}

impl DeviceStruct {
    /// `typedef struct Name { ... } Name;` followed by a blank line.
    ///
    /// Members are not indented, which matches the rest of `ptir_abi.h`: the
    /// hand-written parts are Rust string literals using `\` continuations,
    /// and a `\` continuation eats the next line's leading whitespace. The
    /// source looks indented and the output never was. Reproducing that here
    /// keeps the generated header byte-identical to the checked-in one, so
    /// `ptir_header.rs` proves this generator is a pure refactor. Indenting
    /// the whole file is a separate change.
    pub fn emit_c(&self) -> String {
        let mut out = format!("typedef struct {} {{\n", self.c_name);
        for field in self.fields {
            out.push_str(&format!("{} {};\n", field.ty.c(), field.name));
        }
        out.push_str(&format!("}} {};\n\n", self.c_name));
        out
    }

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
/// and commit kernels, read by the driver. It has no `pie-plan` counterpart —
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

/// Compile-time proof that this table describes the `pie-plan` structs the
/// host actually writes.
///
/// Without these, a field added to `LaneRecord` and not to [`LANE_RECORD`]
/// would compile, emit a kernel that reads the old layout, and produce wrong
/// numbers at run time. `offset_of!` is a constant, so the mismatch is caught
/// before anything runs.
mod static_assertions {
    use super::*;
    use core::mem::{offset_of, size_of};
    use pie_plan::{LaneChannelSlot, LaneRecord, LaneTableHeader};

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
