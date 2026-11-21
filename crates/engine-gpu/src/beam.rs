//! Beam groups' page lists (`state-geometry.parents`,
//! `new-memory/designs/BEAM-PAGE-LISTS.md`), planned on the host from what
//! the host knows (each group's working set's reserve and frozen slots, the
//! slot its beams write, its lanes' staged tables) and run on the device
//! from the group's resident state and the parents its last epilogue wrote
//! (`kernels_slang::runtime::beam`: the step, then the page copies), ahead
//! of the fire's forward in queue order: the fire's feed frame
//! (`Shell::feed`), behind the passes that wrote the parents.
//!
//! The host stages each beam's page table as for any envelope lane; the
//! group's step reorders its lists by the parents (or primes them on its
//! first fire) and overwrites each beam's entries past `f0` and its rows'
//! write pages in the arm's input store with the pages its list names,
//! bounds-checked against the group's pool and frozen slots; the copies it
//! leaves in its state run next. A lane out of bounds keeps the staged
//! words (its own set's pages) and is faulted by name once the fire lands
//! (`crate::settle::FedCheck`). The port of engine-cuda's `serve::pages`.
//!
//! The group's state is the instance's resident run (`GuestState::resident`,
//! [`Resident`]): live from the bind to the close, cleared once at the
//! bind, copied by the runtime wherever a park moves the instance. It names
//! pool pages only, never a physical page, so a restored search continues
//! as it was.

use engine::program::ScratchPlace;
use eta_compiler::codegen::launch::LaunchPackage;
use kernels_slang::runtime::beam as kb;

use crate::backend::{Backend, Buffer as _};
use crate::error::{Fault, Result};
use crate::guest::state::{BeamLayout, Placed};
use crate::inputs::BeamRoom;
use crate::serve::DeviceCell;
use crate::store::Pools;
use crate::store::layout::INSTANCE;

fn refused(why: String) -> Fault {
    Fault::Program { at: "beam", why }
}

/// The beams of a program that states each beam's parent (`Port::Parents`,
/// a `[lanes]` u32 channel), or `None`.
#[must_use]
pub fn lanes_of(package: &LaunchPackage) -> Option<u32> {
    let binding = package
        .ports
        .iter()
        .find(|binding| binding.port == eta_ir::registry::Port::Parents && !binding.is_const)?;
    let channel = package.channels.get(binding.channel as usize)?;
    let numel = channel.shape.iter().map(|&dim| u64::from(dim)).product::<u64>();
    u32::try_from(numel).ok()
}

/// A beam group's resident state where the runtime placed it.
#[derive(Debug)]
pub struct Resident<B: Backend> {
    placed: Placed<B>,
    layout: BeamLayout,
}

impl<B: Backend> Resident<B> {
    /// The state of `layout` at `place` (the binding's resident run),
    /// cleared: room another owner left until then, and the first fire
    /// primes a state of zeros.
    pub fn bind(pools: &Pools<B>, place: &ScratchPlace, layout: BeamLayout) -> Result<Resident<B>> {
        let placed = Placed::at(pools, place, layout.bytes())?;
        if !matches!(place, ScratchPlace::Run { .. }) {
            return Err(refused(format!(
                "a beam group's state lies in one run of Instance slots, and the binding \
                 names {place:?}"
            )));
        }
        placed.clear()?;
        Ok(Resident { placed, layout })
    }

    #[must_use]
    pub fn layout(&self) -> BeamLayout {
        self.layout
    }

    /// Points the state at where its slots sit now (an evacuation).
    pub fn rebind(&mut self, pools: &Pools<B>) -> Result<()> {
        self.placed.rebind(pools)
    }

    /// Moves the state to `place`, the run the runtime copied its bytes
    /// into at an unpark: nothing is cleared.
    pub fn retarget(&mut self, pools: &Pools<B>, place: &ScratchPlace) -> Result<()> {
        if !matches!(place, ScratchPlace::Run { .. }) {
            return Err(refused(format!(
                "a beam group's state lies in one run of Instance slots, and the rebind names \
                 {place:?}"
            )));
        }
        self.placed.retarget(pools, place)
    }

    /// The state's first word in the pool, as a kernel addresses it.
    pub fn word(&self, pools: &Pools<B>) -> Result<u32> {
        match self.placed.place() {
            &ScratchPlace::Run { first, .. } => pools.slot_word(INSTANCE, first),
            other => Err(refused(format!(
                "a beam group's state at {other:?}, not a run of Instance slots"
            ))),
        }
    }

    /// Beam `beam`'s list as the last fire left it (§6.6): the lists are
    /// kept twice by the fire count's parity, the last fire's at its
    /// fires' parity; `cap` columns a beam. Read once every fire that
    /// writes it landed.
    pub fn list(&self, pools: &Pools<B>, beam: u32) -> Result<Vec<u32>> {
        let BeamLayout { lanes, cap, .. } = self.layout;
        if beam >= lanes {
            return Err(refused(format!("beam {beam} of a group of {lanes}")));
        }
        let word = u64::from(self.word(pools)?);
        let read = |at: u64, words: usize| -> Result<Vec<u32>> {
            let (buffer, offset) = pools.locate(at * 4)?;
            let mut bytes = vec![0u8; words * 4];
            buffer.read(offset, &mut bytes)?;
            Ok(bytes
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect())
        };
        let header = read(word, kb::HEADER as usize)?;
        if header[0] == 0 {
            return Err(refused(
                "the beam group has fired no step yet, so it holds no list".to_string(),
            ));
        }
        let row = kb::map_word(lanes, cap, header[1]) + u64::from(beam) * u64::from(cap);
        read(word + row, cap as usize)
    }
}

/// Where a group's parents come from at a fire: the host's ring (the pass
/// that put them landed into it), or the device cell a pass still on the
/// queue writes, copied into the store behind it.
#[derive(Debug)]
pub enum Parents<B: Backend> {
    Host(Vec<u32>),
    Cell(DeviceCell<B>),
}

/// A beam group of a fire: one instance's lanes, beam `b` its lane
/// `lanes[b]` of the fire.
#[derive(Debug)]
pub struct Group<B: Backend> {
    pub instance: engine::InstanceId,
    /// Its state's first word in the pool, and its layout.
    pub state: u32,
    pub layout: BeamLayout,
    pub parents: Parents<B>,
    pub lanes: Vec<usize>,
    /// Its working set's whole kv list table (slot refs, `Lane::reserve`)
    /// and its frozen slots, as its lanes state them.
    pub reserve: Vec<u32>,
    pub folded: u32,
}

/// A beam's lane as the fire stages it: its seat in the staged tables, its
/// first row, the slots it held before the fire and the rows it lands.
#[derive(Clone, Copy, Debug)]
pub struct Seat {
    pub seat: usize,
    pub row_offset: u32,
    pub have: u32,
    pub rows: u32,
}

/// What the fire stages that a plan reads.
pub struct Staged<'a> {
    pub room: BeamRoom,
    /// Per staged table: its seats' `indptr`, and its indices', write
    /// pages' and write offsets' first words in the store.
    pub tables: Vec<(&'a [i32], (u32, u32, u32))>,
    /// Whether any staged table is a windowed one (refused).
    pub windowed: bool,
    pub page: u32,
    /// The kv planes a page holds (their first word in a page and the
    /// words a slot holds there) and the pool words a page id counts.
    pub planes: Vec<[u32; 2]>,
    pub unit: u32,
}

/// What a fire's beam groups have the device run, and the store words the
/// host writes for them.
#[derive(Debug)]
pub struct Planned<'a, B: Backend> {
    /// Words the host writes, each run from its first word in the store.
    pub writes: Vec<(u32, Vec<u32>)>,
    /// Device cells copied into the store ahead of the step: the cell and
    /// its word there.
    pub cells: Vec<(&'a DeviceCell<B>, u32)>,
    pub steps: u32,
    pub copies: u32,
    pub planes: u32,
    /// Per fault word: its word in the store, the lane it faults and its
    /// instance.
    pub faults: Vec<(u32, usize, engine::InstanceId)>,
    /// The highest pool page id the groups stage.
    pub top_page: Option<u32>,
}

impl<B: Backend> Planned<'_, B> {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps == 0
    }
}

/// The pool pages a group of `lanes` beams can name at the fire whose
/// beams write `slot` (§3): the step hands out the lowest free pages, so
/// every page named or pending lies below `B (c + 2) + B F + 1`. Only
/// those are staged.
#[must_use]
pub fn backed(lanes: u32, slot: u32, folded: u32, page: u32, frames: u32, reserve: u32) -> u32 {
    let page = page.max(1);
    let column = (slot / page).saturating_sub(folded / page);
    lanes
        .saturating_mul(column.saturating_add(2))
        .saturating_add(lanes.saturating_mul(frames.max(1)))
        .saturating_add(1)
        .min(reserve)
}

/// Why a lane's fault word faults it (§7).
#[must_use]
pub fn fault_reason(code: u32, lane: usize) -> String {
    let why = match code {
        kb::PAGES_PAST_RESERVE => "names a page past its working set's pool",
        kb::PAGES_SHORT => "is shorter than the pages the lane reads",
        kb::PAGES_BELOW_FLOOR => {
            "would land a row below its working set's frozen slots or past its reserve"
        }
        kb::PAGES_COPY_BOUNDS => {
            "states a page copy past its working set's pool, longer than a page, or into frozen \
             slots"
        }
        kb::PAGES_PARENT => "continues a parent past its beams",
        kb::PAGES_EXHAUSTED => {
            tracing::error!(
                lane,
                "engine-gpu: BUG: a beam group found fewer free pages than it hands out; its \
                 pool is sized exactly (BEAM-PAGE-LISTS.md §3)"
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

/// A part of the room, filled from its first word: refused by name past
/// what the load laid.
struct Part {
    name: &'static str,
    at: u32,
    cap: u32,
    words: Vec<u32>,
}

impl Part {
    fn of(name: &'static str, (at, cap): (u32, u32)) -> Part {
        Part {
            name,
            at,
            cap,
            words: Vec::new(),
        }
    }

    /// The word the next push lands at.
    fn next(&self) -> u32 {
        self.at + u32::try_from(self.words.len()).unwrap_or(u32::MAX)
    }

    fn push(&mut self, words: &[u32]) -> Result<u32> {
        let at = self.next();
        if self.words.len() + words.len() > self.cap as usize {
            return Err(Fault::Ceiling {
                what: self.name,
                need: (self.words.len() + words.len()) as u64,
                have: u64::from(self.cap),
            });
        }
        self.words.extend_from_slice(words);
        Ok(at)
    }
}

/// Plans `groups` over the fire as `staged`: `seat(lane)` a beam's lane as
/// staged, `translation(refs)` a working set's table's slot refs as page
/// ids.
pub fn plan<'a, B: Backend>(
    groups: &'a [Group<B>],
    staged: &Staged<'_>,
    seat: &dyn Fn(usize) -> Option<Seat>,
    translation: &mut dyn FnMut(&[u32]) -> Result<Vec<u32>>,
) -> Result<Planned<'a, B>> {
    let mut planned = Planned {
        writes: Vec::new(),
        cells: Vec::new(),
        steps: 0,
        copies: 0,
        planes: u32::try_from(staged.planes.len()).unwrap_or(u32::MAX),
        faults: Vec::new(),
        top_page: None,
    };
    if groups.is_empty() {
        return Ok(planned);
    }
    if staged.windowed {
        return Err(refused(
            "a beam group's page lists are resolved over full-context tables only, and this \
             load lays windowed ones"
                .to_string(),
        ));
    }
    let room = staged.room;
    let page = staged.page.max(1);
    let mut parents = Part::of("beam groups' parents in one fire", room.parents);
    let mut steps = Part::of("beam groups' steps in one fire", room.steps);
    let mut lists = Part::of("beam lanes' page table runs in one fire", room.lists);
    let mut copies = Part::of("beam lanes' page copies in one fire", room.copies);
    let mut planes = Part::of("kv planes a beam copy moves", room.planes);
    let mut trans = Part::of("pool pages beam groups stage in one fire", room.trans);
    for plane in &staged.planes {
        planes.push(plane)?;
    }
    for group in groups {
        let layout = group.layout;
        let lanes = layout.lanes;
        kb::admit_lanes(lanes).map_err(|why| refused(why.to_string()))?;
        if group.lanes.len() != lanes as usize {
            return Err(refused(format!(
                "instance {}'s beam group binds {} lane(s) in this fire and states parents for \
                 {lanes}; every beam runs every fire",
                group.instance,
                group.lanes.len()
            )));
        }
        let seats = group
            .lanes
            .iter()
            .map(|&lane| {
                seat(lane).ok_or_else(|| {
                    refused(format!("instance {}'s lane {lane} has no seat", group.instance))
                })
            })
            .collect::<Result<Vec<Seat>>>()?;
        let slot = seats[0].have;
        if seats.iter().any(|seat| seat.have != slot || seat.rows != 1) {
            return Err(refused(format!(
                "instance {}'s beams write different slots or more than one row; every beam \
                 writes its depth, one row a fire",
                group.instance
            )));
        }
        let table_pages = translation(&group.reserve)?;
        let folded = group.folded;
        let f0 = folded / page;
        let whole = table_pages.get(f0 as usize..).unwrap_or(&[]);
        let reserve = u32::try_from(whole.len()).unwrap_or(u32::MAX);
        if reserve > layout.cap {
            return Err(refused(format!(
                "instance {}'s working set reserves {reserve} pool page(s) past its frozen ones \
                 and its beam state is laid for {}",
                group.instance, layout.cap
            )));
        }
        let backed = backed(lanes, slot, folded, page, layout.frames, reserve);
        let pool = &whole[..backed as usize];
        planned.top_page = pool.iter().copied().chain(planned.top_page).max();
        let trans_at = trans.push(pool)?;
        // The group's parents and its lanes' fault words share an index.
        let parents_at = parents.next();
        let faults_at = room.faults.0 + (parents_at - room.parents.0);
        match &group.parents {
            Parents::Host(values) => {
                if values.len() != lanes as usize {
                    return Err(refused(format!(
                        "instance {}'s parents cell holds {} beam(s) for a group of {lanes}",
                        group.instance,
                        values.len()
                    )));
                }
                parents.push(values)?;
            }
            Parents::Cell(cell) => {
                if cell.bytes != u64::from(lanes) * 4 {
                    return Err(refused(format!(
                        "instance {}'s parents cell holds {} byte(s) for a group of {lanes}",
                        group.instance, cell.bytes
                    )));
                }
                // Laid by the copy; the host writes zeros it overwrites.
                parents.push(&vec![0; lanes as usize])?;
                planned.cells.push((cell, parents_at));
            }
        }
        for (beam, &lane) in group.lanes.iter().enumerate() {
            planned.faults.push((
                faults_at + u32::try_from(beam).unwrap_or(u32::MAX),
                lane,
                group.instance,
            ));
        }
        let lists_at = lists.next();
        for &(indptr, (indices, write_page, write_offset)) in &staged.tables {
            for (beam, at) in seats.iter().enumerate() {
                let from = indptr.get(at.seat).copied().ok_or_else(|| {
                    refused(format!("instance {}'s beam {beam} has no seat", group.instance))
                })?;
                let to = indptr.get(at.seat + 1).copied().unwrap_or(from);
                let (from, to) = (from.max(0) as u32, to.max(0) as u32);
                let count = to.saturating_sub(from);
                lists.push(&[
                    indices + from + f0.min(count),
                    write_page + at.row_offset,
                    write_offset + at.row_offset,
                    u32::try_from(beam).unwrap_or(u32::MAX),
                    count.saturating_sub(f0),
                    at.rows,
                    0,
                    0,
                ])?;
            }
        }
        let list_count = (lists.next() - lists_at) / kb::LIST_WORDS;
        steps.push(&[
            group.state,
            parents_at,
            trans_at,
            lists_at,
            faults_at,
            lanes,
            layout.cap,
            reserve,
            page,
            folded,
            slot,
            list_count,
            layout.frames,
            backed,
            0,
            0,
        ])?;
        planned.steps += 1;
        let copy_word = kb::copies_word(lanes, layout.cap, layout.frames);
        for beam in 0..lanes {
            let copy = u64::from(group.state) + copy_word + 3 * u64::from(beam);
            copies.push(&[
                u32::try_from(copy).map_err(|_| Fault::Ceiling {
                    what: "pool words a beam copy names",
                    need: copy,
                    have: u64::from(u32::MAX),
                })?,
                trans_at,
                faults_at + beam,
                backed,
                page,
                folded - f0 * page,
                staged.unit,
                planned.planes,
            ])?;
            planned.copies += 1;
        }
    }
    for part in [parents, steps, lists, copies, planes, trans] {
        if !part.words.is_empty() {
            planned.writes.push((part.at, part.words));
        }
    }
    Ok(planned)
}
