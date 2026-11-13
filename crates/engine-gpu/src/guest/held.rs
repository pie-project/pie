//! Where a guest instance's state lies on the device (REDESIGN step 4,
//! `new-memory/STATE-ROLES.md`): its bind-time launch tables in pool slots
//! of role `Instance`, and the cells a channel keeps on the device in slots
//! of role `Channel`, which the runtime took for them. An item past one slot
//! lies in a run of adjacent slots of one chunk (`Footprint::contiguous`);
//! an item past a chunk is refused.
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

/// Lays tables of `sizes` out in slots of `slot_bytes`: each aligned to
/// `align` and never straddling a slot. Each table's slot index and byte
/// there, and the slots taken; `None` where a table is larger than a slot.
#[must_use]
pub fn pack(sizes: &[u64], slot_bytes: u64, align: u64) -> Option<(Vec<(u32, u64)>, u32)> {
    let align = align.max(1);
    let mut places = Vec::with_capacity(sizes.len());
    let (mut slot, mut cursor) = (0u32, 0u64);
    for &bytes in sizes {
        let bytes = bytes.max(1);
        if bytes > slot_bytes {
            return None;
        }
        let at = cursor.next_multiple_of(align);
        if at + bytes > slot_bytes {
            slot += 1;
            cursor = 0;
        } else {
            cursor = at;
        }
        places.push((slot, cursor));
        cursor += bytes;
    }
    let slots = if sizes.is_empty() { 0 } else { slot + 1 };
    Some((places, slots))
}

/// How an instance's tables lie in its `Instance` slots: packed a slot
/// ([`pack`]), or, where a table is larger than a slot, one after another
/// in one run of adjacent slots (`Footprint::contiguous`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableLayout {
    /// Each table's slot index (0 in a run) and byte there.
    pub places: Vec<(u32, u64)>,
    pub slots: u32,
    pub run: bool,
}

impl TableLayout {
    #[must_use]
    pub fn of(sizes: &[u64], slot_bytes: u64, align: u64) -> TableLayout {
        if let Some((places, slots)) = pack(sizes, slot_bytes, align) {
            return TableLayout {
                places,
                slots,
                run: false,
            };
        }
        let mut places = Vec::with_capacity(sizes.len());
        let mut end = 0u64;
        for &bytes in sizes {
            let at = end.next_multiple_of(align.max(1));
            places.push((0, at));
            end = at + bytes.max(1);
        }
        TableLayout {
            places,
            slots: u32::try_from(end.div_ceil(slot_bytes)).unwrap_or(u32::MAX),
            run: true,
        }
    }

    /// Whether the slots fit one chunk of `per_chunk` slots: a run must.
    #[must_use]
    pub fn fits(&self, per_chunk: u32) -> bool {
        !self.run || self.slots <= per_chunk
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
        contiguous: layout.run,
    })
}

/// Places `tables` (sizes in [`TableLayout`] order) in `slots`, slot refs of
/// `shape`, and writes each one's bytes: one write per slot or run, so a
/// bind costs one host copy through the queue for every slot it fills.
/// Each table's span, and whether they lie in one run.
pub fn place_tables<B: Backend>(
    pools: &Pools<B>,
    shape: ShapeId,
    slots: &[u32],
    tables: &[Vec<u8>],
    align: u64,
) -> Result<(Vec<Held<B>>, bool)> {
    let sizes: Vec<u64> = tables.iter().map(|t| t.len() as u64).collect();
    let layout = TableLayout::of(&sizes, pools.slot_bytes(shape)?, align);
    if layout.slots as usize != slots.len() || (layout.run && !consecutive(slots)) {
        return Err(refused(format!(
            "this instance's tables take {} instance slot(s){} and the binding names {slots:?}",
            layout.slots,
            if layout.run { " in one run" } else { "" },
        )));
    }
    let count = u32::try_from(slots.len()).unwrap_or(u32::MAX);
    let held = layout
        .places
        .iter()
        .zip(&sizes)
        .map(|(&(slot, at), &bytes)| {
            if layout.run {
                Held::in_run(pools, shape, slots[0], count, at, bytes.max(1))
            } else {
                Held::in_slot(pools, shape, slots[slot as usize], at, bytes.max(1))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    // One image a slot (or the run), from the slot's first byte, written
    // once.
    let mut images: Vec<(usize, &Held<B>, Vec<u8>)> = Vec::new();
    for ((&(slot, at), table), span) in layout.places.iter().zip(tables).zip(&held) {
        let index = if layout.run { 0 } else { slot as usize };
        let image = match images.iter().position(|(k, _, _)| *k == index) {
            Some(found) => &mut images[found].2,
            None => {
                images.push((index, span, Vec::new()));
                &mut images.last_mut().expect("just pushed").2
            }
        };
        let at = usize::try_from(at).unwrap_or(usize::MAX);
        let end = (at + table.len()).next_multiple_of(4);
        if image.len() < end {
            image.resize(end, 0);
        }
        image[at..at + table.len()].copy_from_slice(table);
    }
    for (_, span, image) in images {
        span.buffer()
            .write_shared(span.offset() - span.at, &image)?;
    }
    Ok((held, layout.run))
}

#[cfg(test)]
mod tests {
    use super::{TableLayout, consecutive, item_slots, pack};

    #[test]
    fn tables_pack_aligned_and_never_straddle_a_slot() {
        let (places, slots) = pack(&[100, 40, 4000, 8], 4096, 16).expect("every table fits");
        assert_eq!(places, vec![(0, 0), (0, 112), (1, 0), (1, 4000)]);
        assert_eq!(slots, 2);
        let (places, _) = pack(&[100, 40], 4096, 64).expect("both fit");
        assert_eq!(places, vec![(0, 0), (0, 128)]);
        assert_eq!(pack(&[], 4096, 16), Some((Vec::new(), 0)));
        assert_eq!(pack(&[5000], 4096, 16), None);
    }

    #[test]
    fn tables_past_a_slot_lie_one_after_another_in_one_run() {
        let packed = TableLayout::of(&[100, 40], 4096, 16);
        assert!(!packed.run);
        assert_eq!(packed.slots, 1);
        let run = TableLayout::of(&[5000, 100, 8], 4096, 16);
        assert!(run.run, "a table past a slot makes the tables a run");
        assert_eq!(run.places, vec![(0, 0), (0, 5008), (0, 5120)]);
        assert_eq!(run.slots, 2);
        assert!(run.fits(2) && !run.fits(1));
    }

    #[test]
    fn an_item_past_a_slot_takes_a_run_inside_a_chunk() {
        assert_eq!(item_slots(4, 1024, 4096), Some((1, false)));
        assert_eq!(item_slots(1024, 1024, 4096), Some((1, false)));
        // A vocabulary-wide f32 cell.
        assert_eq!(item_slots(151_936 * 4, 1024, 4096), Some((594, true)));
        assert_eq!(item_slots(151_936 * 4, 1024, 512), None);
        assert!(consecutive(&[7, 8, 9]));
        assert!(!consecutive(&[7, 9]));
    }
}
