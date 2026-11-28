//! Where a guest instance's state lies on the device (REDESIGN step 4,
//! `new-memory/STATE-ROLES.md`): its bind-time launch tables in pool slots
//! of role `Instance`, and the cells a channel keeps on the device in slots
//! of role `Channel`, which the runtime took for them. An item past one slot
//! lies in a run of adjacent slots of one chunk (`Footprint::contiguous`);
//! an item past a chunk is refused. An instance's tables lie in one run
//! whatever their sizes, so in one segment: a backend binds that segment
//! once for every instance whose tables lie in it.
//!
//! A [`Held`] is a span of one slot ref (or one run), addressed through the
//! segment buffer its chunk sits in. Evacuation moves the chunk under a ref
//! and leaves the ref, so a span is [`Held::rebind`]-ed after it; a park
//! and unpark moves it to other refs ([`Held::retarget`]).

use engine::chunk::ShapeId;

use crate::backend::{Backend, Buffer as _};
use crate::error::{Fault, Result};
use crate::store::Pools;

/// The least alignment of a table in its slot: the widest word the guest
/// kernels read.
pub const TABLE_ALIGN: u64 = 16;

fn refused(why: String) -> Fault {
    Fault::Program {
        at: "guest::held",
        why,
    }
}

/// Bytes `[at, at + bytes)` of slot ref `slot` of `shape` (the first of a
/// run of `run` slots), as a kernel binds them: `buffer` at `offset`.
#[derive(Debug, Clone)]
pub struct Held<B: Backend> {
    shape: ShapeId,
    slot: u32,
    run: u32,
    at: u64,
    bytes: u64,
    buffer: B::Buffer,
    offset: u64,
}

impl<B: Backend> Held<B> {
    /// Bytes `[at, at + bytes)` of slot ref `slot` of `shape`.
    pub fn in_slot(
        pools: &Pools<B>,
        shape: ShapeId,
        slot: u32,
        at: u64,
        bytes: u64,
    ) -> Result<Self> {
        Held::in_run(pools, shape, slot, 1, at, bytes)
    }

    /// Bytes `[at, at + bytes)` of the run of `count` slots of `shape` from
    /// `first`: adjacent slots of one chunk, so one span of one segment.
    pub fn in_run(
        pools: &Pools<B>,
        shape: ShapeId,
        first: u32,
        count: u32,
        at: u64,
        bytes: u64,
    ) -> Result<Self> {
        let per_chunk = pools.slots_per_chunk(shape)?.max(1);
        let slot_bytes = pools.slot_bytes(shape)?;
        let count = count.max(1);
        let last = first.checked_add(count - 1);
        if last.is_none_or(|last| last / per_chunk != first / per_chunk)
            || at
                .checked_add(bytes)
                .is_none_or(|end| end > u64::from(count) * slot_bytes)
        {
            return Err(refused(format!(
                "bytes {at}..{} of a run of {count} slot(s) of shape {shape} from {first}: a run \
                 lies inside one chunk of {per_chunk} slots and holds its item whole, and a \
                 table or cell never straddles a slot",
                at.saturating_add(bytes)
            )));
        }
        let (buffer, offset) = pools.slot_place(shape, first)?;
        Ok(Held {
            shape,
            slot: first,
            run: count,
            at,
            bytes,
            buffer: buffer.clone(),
            offset: offset + at,
        })
    }

    /// Points the span at the chunk its ref sits in now.
    pub fn rebind(&mut self, pools: &Pools<B>) -> Result<()> {
        let (buffer, offset) = pools.slot_place(self.shape, self.slot)?;
        self.buffer = buffer.clone();
        self.offset = offset + self.at;
        Ok(())
    }

    /// Moves the span to slot ref `to` (the first of as long a run), whose
    /// bytes the runtime already copied there.
    pub fn retarget(&mut self, pools: &Pools<B>, to: u32) -> Result<()> {
        let per_chunk = pools.slots_per_chunk(self.shape)?.max(1);
        if to
            .checked_add(self.run - 1)
            .is_none_or(|last| last / per_chunk != to / per_chunk)
        {
            return Err(refused(format!(
                "a run of {} slot(s) rebound to {to}, past its chunk",
                self.run
            )));
        }
        self.slot = to;
        self.rebind(pools)
    }

    /// The slot ref the span lies in (the first of its run).
    #[must_use]
    pub fn slot(&self) -> u32 {
        self.slot
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub fn buffer(&self) -> &B::Buffer {
        &self.buffer
    }

    /// The span's first byte in [`Held::buffer`].
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The bytes of [`Held::buffer`] its slot (or run) spans, of slots of
    /// `slot_bytes`: what the instance holds there, the span among it.
    #[must_use]
    pub fn run_span(&self, slot_bytes: u64) -> (u64, u64) {
        let start = self.offset - self.at;
        (start, start + u64::from(self.run) * slot_bytes)
    }

    /// Writes `from` at `at` of the span through the queue (the segment is
    /// device memory), before the work submitted next.
    pub fn write(&self, at: u64, from: &[u8]) -> Result<()> {
        if at
            .checked_add(from.len() as u64)
            .is_none_or(|end| end > self.bytes)
        {
            return Err(refused(format!(
                "bytes {at}..{} of a {}-byte span",
                at.saturating_add(from.len() as u64),
                self.bytes
            )));
        }
        self.buffer.write_shared(self.offset + at, from)
    }
}

/// How an instance's tables lie in its `Instance` slots: one after another,
/// each aligned, in one run of adjacent slots of one chunk
/// (`Footprint::contiguous` where it takes more than one), so in one
/// segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableLayout {
    /// Each table's byte in the run.
    pub places: Vec<u64>,
    pub slots: u32,
}

impl TableLayout {
    #[must_use]
    pub fn of(sizes: &[u64], slot_bytes: u64, align: u64) -> TableLayout {
        let mut places = Vec::with_capacity(sizes.len());
        let mut end = 0u64;
        for &bytes in sizes {
            let at = end.next_multiple_of(align.max(1));
            places.push(at);
            end = at + bytes.max(1);
        }
        TableLayout {
            places,
            slots: u32::try_from(end.div_ceil(slot_bytes.max(1))).unwrap_or(u32::MAX),
        }
    }

    /// Whether the run fits one chunk of `per_chunk` slots: it must.
    #[must_use]
    pub fn fits(&self, per_chunk: u32) -> bool {
        self.slots <= per_chunk
    }

    /// Whether the runtime must take the slots as one run
    /// (`Footprint::contiguous`): wherever they are more than one, whatever
    /// each table's size.
    #[must_use]
    pub fn contiguous(&self) -> bool {
        self.slots > 1
    }
}

/// Whether `slots` are refs `first .. first + n` in order.
#[must_use]
pub fn consecutive(slots: &[u32]) -> bool {
    slots
        .windows(2)
        .all(|pair| pair[0].checked_add(1) == Some(pair[1]))
}

/// The slots an item of `bytes` takes in slots of `slot_bytes`, and
/// whether they are one run (the item is larger than a slot); `None` where
/// the run would not fit one chunk of `per_chunk` slots.
#[must_use]
pub fn item_slots(bytes: u64, slot_bytes: u64, per_chunk: u32) -> Option<(u32, bool)> {
    let slots = u32::try_from(bytes.max(1).div_ceil(slot_bytes.max(1))).ok()?;
    (slots <= per_chunk).then_some((slots, slots > 1))
}

/// What an instance's tables of `sizes` take in `shape`'s slots; refused
/// where they would not fit one chunk.
pub fn table_footprint<B: Backend>(
    pools: &Pools<B>,
    shape: ShapeId,
    sizes: &[u64],
    align: u64,
) -> Result<engine::program::Footprint> {
    let layout = TableLayout::of(sizes, pools.slot_bytes(shape)?, align);
    if !layout.fits(pools.slots_per_chunk(shape)?) {
        return Err(refused(format!(
            "an instance's launch tables take a run of {} slots, past a chunk",
            layout.slots
        )));
    }
    Ok(engine::program::Footprint {
        bytes: sizes.iter().sum(),
        slots: layout.slots,
        contiguous: layout.contiguous(),
    })
}

/// Places `tables` (sizes in [`TableLayout`] order) in `slots`, slot refs of
/// `shape` (one run, in order), and writes their bytes in one write through
/// the queue. Each table's span.
pub fn place_tables<B: Backend>(
    pools: &Pools<B>,
    shape: ShapeId,
    slots: &[u32],
    tables: &[Vec<u8>],
    align: u64,
) -> Result<Vec<Held<B>>> {
    let sizes: Vec<u64> = tables.iter().map(|t| t.len() as u64).collect();
    let layout = TableLayout::of(&sizes, pools.slot_bytes(shape)?, align);
    if layout.slots as usize != slots.len() || !consecutive(slots) {
        return Err(refused(format!(
            "this instance's tables take a run of {} instance slot(s) and the binding names \
             {slots:?}",
            layout.slots,
        )));
    }
    let count = u32::try_from(slots.len()).unwrap_or(u32::MAX);
    let held = layout
        .places
        .iter()
        .zip(&sizes)
        .map(|(&at, &bytes)| Held::in_run(pools, shape, slots[0], count, at, bytes.max(1)))
        .collect::<Result<Vec<_>>>()?;
    // One image of the run, from its first byte, written once.
    let mut image = Vec::new();
    for (&at, table) in layout.places.iter().zip(tables) {
        let at = usize::try_from(at).unwrap_or(usize::MAX);
        let end = (at + table.len()).next_multiple_of(4);
        if image.len() < end {
            image.resize(end, 0);
        }
        image[at..at + table.len()].copy_from_slice(table);
    }
    if let Some(first) = held.first() {
        first
            .buffer()
            .write_shared(first.offset() - first.at, &image)?;
    }
    Ok(held)
}
