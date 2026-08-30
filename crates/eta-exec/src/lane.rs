use core::mem::size_of;

pub use eta_compiler::plan::lane_table::{
    LANE_TABLE_ABI_VERSION as ABI_VERSION, LaneChannelSlot as ChannelSlot, LaneRecord as Record,
    LaneTableHeader as Header,
};

// `FLAG_RAGGED`, `ChannelMeta`, `GroupLayout` and `RowMeta` stood here: four
// `#[repr(C)]` mirrors of a device lane table's side structures. Nothing
// constructs or reads one — `eta_compiler::plan::lane_table` owns the ABI
// the shells actually bind, and these were a second spelling of part of it
// (alto E). The three `*_BYTES` constants below are the live half: they size
// the types re-exported above, which are that crate's.

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
