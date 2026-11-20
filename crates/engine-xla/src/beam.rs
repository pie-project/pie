//! Beam groups' page lists on XLA (`state-geometry.parents`,
//! `new-memory/designs/BEAM-PAGE-LISTS.md`): what an instance whose program
//! states each beam's parent keeps on the device, laid from the load's
//! paging and frames in flight. The step itself is `kernels_xla::beam`,
//! run in the fire's own program ahead of its first append or read.

use eta_compiler::codegen::launch::LaunchPackage;
use kernels_xla::beam::Layout;

/// A program's beam group: its beams (the parents channel's elements) and
/// the parents channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group {
    pub lanes: u32,
    pub channel: u32,
}

/// The program's beam group, where it binds `Port::Parents` from a channel.
#[must_use]
pub fn group_of(package: &LaunchPackage) -> Option<Group> {
    let binding = package
        .ports
        .iter()
        .find(|binding| binding.port == eta_ir::registry::Port::Parents && !binding.is_const)?;
    let channel = package.channels.get(binding.channel as usize)?;
    let lanes = channel.shape.iter().map(|&d| d as u64).product::<u64>();
    Some(Group {
        lanes: u32::try_from(lanes).ok()?,
        channel: binding.channel,
    })
}

/// The group's state layout on this load: its pool holds its beams' pages
/// (each at most a sequence's, `pages_per_slot`), the pages one fire hands
/// out, those pending over the frames in flight and the prompt's own page
/// (§3, as engine-cuda lays it).
#[must_use]
pub fn layout(group: Group, page_size: u32, pages_per_slot: u32, frames: u32) -> Layout {
    let lanes = i64::from(group.lanes);
    let frames = i64::from(frames.max(1));
    Layout {
        lanes,
        cap: lanes * i64::from(pages_per_slot) + lanes * (frames + 1) + 1,
        frames,
        page: i64::from(page_size),
    }
}

/// The state's bytes in the pool's `Instance` slots: its i32 words, whole
/// slots of `slot_bytes`.
#[must_use]
pub fn state_slots(layout: Layout, slot_bytes: u64) -> (u64, u32) {
    let bytes = u64::try_from(layout.words()).unwrap_or(u64::MAX) * 4;
    let slots = bytes.div_ceil(slot_bytes.max(1));
    (slots * slot_bytes, u32::try_from(slots).unwrap_or(u32::MAX))
}

/// A bound instance's beam group on the device: its group, its state's
/// layout, and the pool row its state's `Instance` run starts at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seat {
    pub group: Group,
    pub layout: Layout,
    pub row: u32,
}
