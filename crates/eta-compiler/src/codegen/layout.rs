use alloc::format;
use alloc::string::String;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FieldType {
    U32,
    U64,
}

impl FieldType {
    const fn msl(self) -> &'static str {
        match self {
            FieldType::U32 => "uint",
            FieldType::U64 => "ulong",
        }
    }

    const fn cuda(self) -> &'static str {
        match self {
            FieldType::U32 => "m1_u32",
            FieldType::U64 => "m1_u64",
        }
    }

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MslStyle {
    Inline,
    Block,
}

#[derive(Clone, Copy, Debug)]
pub struct DeviceStruct {
    pub c_name: &'static str,
    pub msl_suffix: &'static str,
    pub msl_style: MslStyle,
    pub fields: &'static [Field],
}

impl DeviceStruct {
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

    pub fn emit_cuda(&self) -> String {
        let mut out = format!("struct {} {{\n", self.c_name);
        for field in self.fields {
            out.push_str(&format!("  {} {};\n", field.ty.cuda(), field.name));
        }
        out.push_str("};\n");
        out
    }

    pub fn emit_cuda_size_assert(&self, note: &str) -> String {
        format!(
            "static_assert(sizeof({}) == {}, \"{note}\");\n",
            self.c_name,
            self.size_bytes()
        )
    }

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
}

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

pub const LANE_RECORD: DeviceStruct = DeviceStruct {
    c_name: "PtirLaneRecord",
    msl_suffix: "LaneRecord",
    msl_style: MslStyle::Block,
    fields: &[
        Field::u64("logits_base"),
        Field::u32("logits_row_offset"),
        Field::u32("logits_row_count"),
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
        Field::u64("attn_score_base"),
        Field::u32("attn_score_row_stride"),
        Field::u32("reserved1"),
        Field::u64("mtp_drafts_base"),
        Field::u32("mtp_drafts_depth"),
        Field::u32("reserved2"),
    ],
};

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

pub const HOST_SHARED: &[DeviceStruct] = &[LANE_TABLE_HEADER, LANE_RECORD, LANE_CHANNEL_SLOT];

mod static_assertions {
    use super::*;
    use crate::plan::{LaneChannelSlot, LaneRecord, LaneTableHeader};
    use core::mem::{offset_of, size_of};

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
        attn_score_base,
        attn_score_row_stride,
        reserved1,
        mtp_drafts_base,
        mtp_drafts_depth,
        reserved2,
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
