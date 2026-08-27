use core::mem::size_of;

pub use tensor_compiler::plan::lane_table::{
    LANE_TABLE_ABI_VERSION as ABI_VERSION, LaneChannelSlot as ChannelSlot, LaneRecord as Record,
    LaneTableHeader as Header,
};

pub const FLAG_RAGGED: u32 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct ChannelMeta {
    pub words: u64,

    pub capacity: u32,

    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct GroupLayout {
    pub lane_count: u32,

    pub value_count: u32,

    pub scratch_stride: u32,

    pub temporary_offset: u32,

    pub vocab: u32,

    pub binding_stride: u32,

    pub rows_per_lane: u32,

    pub op_stride: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct RowMeta {
    pub offset: u32,

    pub count: u32,

    pub mtp_offset: u32,

    pub reserved: u32,
}

pub const HEADER_BYTES: u64 = size_of::<Header>() as u64;

pub const RECORD_BYTES: u64 = size_of::<Record>() as u64;

pub const SLOT_BYTES: u64 = size_of::<ChannelSlot>() as u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub lanes: u32,

    pub channel_slots_per_lane: u32,
}

impl Shape {
    #[must_use]
    pub const fn of(lanes: u32, channel_slots_per_lane: u32) -> Self {
        Self {
            lanes,
            channel_slots_per_lane,
        }
    }

    #[must_use]
    pub fn bytes(&self) -> Option<u64> {
        let lanes = u64::from(self.lanes);
        let records = lanes.checked_mul(RECORD_BYTES)?;
        let slots = lanes
            .checked_mul(u64::from(self.channel_slots_per_lane))?
            .checked_mul(SLOT_BYTES)?;
        HEADER_BYTES.checked_add(records)?.checked_add(slots)
    }

    #[must_use]
    pub fn record_offset(&self, lane: u32) -> Option<u64> {
        if lane >= self.lanes {
            return None;
        }
        HEADER_BYTES.checked_add(u64::from(lane).checked_mul(RECORD_BYTES)?)
    }

    #[must_use]
    pub fn slots_offset(&self) -> Option<u64> {
        HEADER_BYTES.checked_add(u64::from(self.lanes).checked_mul(RECORD_BYTES)?)
    }

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

    #[must_use]
    pub fn slot_index(&self, lane: u32) -> Option<u32> {
        if lane >= self.lanes {
            return None;
        }
        u32::try_from(u64::from(lane) * u64::from(self.channel_slots_per_lane)).ok()
    }
}
