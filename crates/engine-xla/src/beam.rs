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

/// The width rung a group of `lanes` beams runs on: the smallest power of
/// two that holds it (the head program is laid once a rung at load, the
/// rung's beams past the group's own dead).
#[must_use]
pub fn rung(lanes: u32) -> u32 {
    lanes.max(1).next_power_of_two()
}

/// The group's state layout on this load, at its width rung: its pool holds
/// its beams' pages (each at most a sequence's, `pages_per_slot`), the pages
/// one fire hands out, those pending over the frames in flight and the
/// prompt's own page (§3, as engine-cuda lays it).
#[must_use]
pub fn layout(group: Group, page_size: u32, pages_per_slot: u32, frames: u32) -> Layout {
    rung_layout(rung(group.lanes), page_size, pages_per_slot, frames)
}

/// The state layout of width rung `rung` on this load.
#[must_use]
pub fn rung_layout(rung: u32, page_size: u32, pages_per_slot: u32, frames: u32) -> Layout {
    let lanes = i64::from(rung);
    let frames = i64::from(frames.max(1));
    Layout {
        lanes,
        cap: lanes * i64::from(pages_per_slot) + lanes * (frames + 1) + 1,
        frames,
        page: i64::from(page_size),
    }
}

/// The group's `Instance` run: its state's rows and its results' (each
/// beam's resolved entries, write page and fault sets,
/// [`crate::store::beam::results_words`]), whole slots of `slot_bytes`.
#[must_use]
pub fn state_slots(layout: Layout, slot_bytes: u64) -> (u64, u32) {
    let rows = u64::try_from(crate::store::beam::run_rows(layout)).unwrap_or(u64::MAX);
    let bytes = rows * crate::store::layout::ROW_BYTES;
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
    /// The heads run for it: its next fire's index (its fault set's).
    pub fires: u64,
}

/// A beam of a group in this fire, as its fire reads it: where its group's
/// head landed the resolved entries (the first row past its state), its
/// beam and the rung's beams and columns, the set's first pool page (`f0`)
/// and the fire's column (`c`). Entry `f0 + j` of its table (`j <= c`) and
/// its row's write page are the head's, not the staged words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeamLane {
    pub results_row: u32,
    pub beam: u32,
    pub lanes: u32,
    pub cap: u32,
    pub first_page: u32,
    pub column: u32,
}

impl BeamLane {
    /// The pool row and lane of word `w` of its group's results.
    fn word(&self, w: u64) -> (i32, i32) {
        let row = u64::from(self.results_row) + w / 128;
        (i32::try_from(row).unwrap_or(i32::MAX), (w % 128) as i32)
    }

    /// Where entry `j` of its table lies, where the head resolved it.
    #[must_use]
    pub fn entry(&self, j: u32) -> Option<(i32, i32)> {
        let k = j.checked_sub(self.first_page)?;
        (k <= self.column && k < self.cap)
            .then(|| self.word(u64::from(self.beam) * u64::from(self.cap) + u64::from(k)))
    }

    /// Where its row's write page lies.
    #[must_use]
    pub fn write(&self) -> (i32, i32) {
        self.word(u64::from(self.lanes) * u64::from(self.cap) + u64::from(self.beam))
    }
}

/// Why a beam's fault word (`BEAM-PAGE-LISTS.md` §7) fails its lane, named.
#[must_use]
pub fn fault_reason(code: u32, lane: u32) -> String {
    use kernels_xla::beam::fault;
    let why = match i64::from(code) {
        fault::PAST_RESERVE => "names a page past its working set's pool",
        fault::SHORT => "is shorter than the pages the lane reads",
        fault::BELOW_FLOOR => {
            "would land a row below its working set's frozen slots or past its reserve"
        }
        fault::COPY_BOUNDS => {
            "states a page copy past its working set's pool, longer than a page, or into frozen \
             slots"
        }
        fault::PARENT => "continues a parent past its beams",
        fault::RESERVE_EXHAUSTED => {
            tracing::error!(
                lane,
                "engine-xla: BUG: a beam group found fewer free pages than it hands out; its pool \
                 is sized exactly (BEAM-PAGE-LISTS.md §3)"
            );
            "found no free page of its working set's pool left (a bug: the pool is sized exactly)"
        }
        _ => "faulted",
    };
    format!(
        "Overreach: lane {lane}'s page list {why}; nothing was written outside its own working \
         set, and the lane is failed"
    )
}
