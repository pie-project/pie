//! Where a guest instance's state lives (REDESIGN step 4,
//! `new-memory/STATE-ROLES.md`): its bind-time launch tables and its rings,
//! either in buffers of its own or in pool slots of role `Instance` and
//! `Channel` the runtime took for it.
//!
//! A pooled [`Held`] is one slot ref's bytes, addressed through the pool's
//! view (a placement-sparse buffer) and written on the host through the heap
//! under it. Evacuation moves the chunk under a ref and leaves the ref, so a
//! region is [`Held::rebind`]-ed after it: its view offset and host
//! address are those of the chunk the ref sits in now.
#![allow(unsafe_code)]

use std::ptr::NonNull;

use engine::chunk::ShapeId;

use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};
use crate::store::Pools;
use crate::store::layout::CHANNEL;

/// A table's alignment inside an `Instance` slot: the widest field the
/// kernels read from one (`device` address space).
pub const TABLE_ALIGN: u64 = 16;

/// A host address into a heap. The heap outlives the region: a slot's chunk
/// is given back only after the runtime evacuated it and the engine rebound.
#[derive(Debug, Clone, Copy)]
struct HostAt(NonNull<u8>);

// SAFETY: a plain address into a Shared heap, written and read only under
// the plane's `&mut` / `&` discipline, as `device::Buffer::contents` is.
unsafe impl Send for HostAt {}
// SAFETY: as above.
unsafe impl Sync for HostAt {}

#[derive(Debug, Clone)]
enum Place {
    /// A buffer of its own, from offset 0.
    Own,
    /// Bytes `[at, at + bytes)` of slot `slot` of shape `shape`, at view
    /// offset `offset` and host address `host`.
    Slot {
        shape: ShapeId,
        slot: u32,
        at: u64,
        host: HostAt,
    },
}

/// A table's or a ring's bytes as a kernel binds them: `buffer` at
/// `offset`.
#[derive(Debug, Clone)]
pub struct Held {
    buffer: Buffer,
    offset: u64,
    bytes: u64,
    place: Place,
}

impl Held {
    /// `bytes` zeroed bytes in a buffer of their own.
    pub fn own(device: &Context, bytes: u64) -> Result<Held> {
        Ok(Held {
            buffer: Buffer::zeroed(device, bytes.max(1))?,
            offset: 0,
            bytes,
            place: Place::Own,
        })
    }

    /// Bytes `[at, at + bytes)` of slot ref `slot` of `shape`, zeroed.
    pub fn in_slot(pools: &Pools, shape: ShapeId, slot: u32, at: u64, bytes: u64) -> Result<Held> {
        let (offset, host, slot_bytes) = pools.slot_place(shape, slot)?;
        if at.checked_add(bytes).is_none_or(|end| end > slot_bytes) {
            return Err(Fault::program(
                "program::held",
                format!(
                    "bytes {at}..{} of a {slot_bytes}-byte slot of shape {shape}: a table \
                     or a cell never straddles a slot",
                    at.saturating_add(bytes)
                ),
            ));
        }
        // SAFETY: `at < slot_bytes`, inside the slot's chunk.
        let host = HostAt(unsafe { host.add(usize_of(at)) });
        let mut region = Held {
            buffer: pools.view().clone(),
            offset: offset + at,
            bytes,
            place: Place::Slot {
                shape,
                slot,
                at,
                host,
            },
        };
        region.zero()?;
        Ok(region)
    }

    /// `bytes` zeroed bytes over the run of `count` slots of `shape` from
    /// `first`: adjacent slots of one chunk, so one span in the view and in
    /// the heap (STATE-ROLES.md option D).
    pub fn in_run(
        pools: &Pools,
        shape: ShapeId,
        first: u32,
        count: u32,
        bytes: u64,
    ) -> Result<Held> {
        let per_chunk = pools.slots_per_chunk(shape)?;
        let slot_bytes = pools.slot_bytes(shape)?;
        let last = first.checked_add(count.max(1) - 1);
        if last.is_none_or(|last| last / per_chunk != first / per_chunk)
            || bytes > u64::from(count) * slot_bytes
        {
            return Err(Fault::program(
                "program::held",
                format!(
                    "a run of {count} slot(s) of shape {shape} from {first} holding {bytes} \
                     bytes: a run lies inside one chunk of {per_chunk} slots and holds its \
                     item whole"
                ),
            ));
        }
        let (offset, host, _) = pools.slot_place(shape, first)?;
        let mut region = Held {
            buffer: pools.view().clone(),
            offset,
            bytes,
            place: Place::Slot {
                shape,
                slot: first,
                at: 0,
                host: HostAt(host),
            },
        };
        region.zero()?;
        Ok(region)
    }

    /// Points a pooled region at the chunk its ref sits in now.
    pub fn rebind(&mut self, pools: &Pools) -> Result<()> {
        if let Place::Slot {
            shape,
            slot,
            at,
            host,
        } = &mut self.place
        {
            let (offset, base, _) = pools.slot_place(*shape, *slot)?;
            self.offset = offset + *at;
            // SAFETY: as in `in_slot`.
            *host = HostAt(unsafe { base.add(usize_of(*at)) });
        }
        Ok(())
    }

    /// The slot ref a pooled region lies in.
    #[must_use]
    pub fn slot(&self) -> Option<u32> {
        match self.place {
            Place::Own => None,
            Place::Slot { slot, .. } => Some(slot),
        }
    }

    /// Moves a pooled region to slot ref `to` of its shape, whose bytes
    /// the runtime already copied there (`Engine::rebind_slots`).
    pub fn retarget(&mut self, pools: &Pools, to: u32) -> Result<()> {
        if let Place::Slot { slot, .. } = &mut self.place {
            *slot = to;
        }
        self.rebind(pools)
    }

    /// The device bytes this region holds apart from the pool.
    #[must_use]
    pub fn own_bytes(&self) -> u64 {
        match self.place {
            Place::Own => self.buffer.bytes(),
            Place::Slot { .. } => 0,
        }
    }

    #[must_use]
    pub fn is_pooled(&self) -> bool {
        matches!(self.place, Place::Slot { .. })
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) fn raw(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer> {
        self.buffer.raw()
    }

    fn check(&self, at: u64, len: usize) -> Result<()> {
        if at
            .checked_add(len as u64)
            .is_none_or(|end| end > self.bytes)
        {
            return Err(Fault::program(
                "program::held",
                format!(
                    "bytes {at}..{} of a {}-byte region",
                    at.saturating_add(len as u64),
                    self.bytes
                ),
            ));
        }
        Ok(())
    }

    /// Writes `from` at `at` on the host; as `Buffer::write`, the caller
    /// writes only bytes no frame in flight reads.
    pub fn write(&mut self, at: u64, from: &[u8]) -> Result<()> {
        self.check(at, from.len())?;
        match self.place {
            Place::Own => self.buffer.write(self.offset + at, from),
            Place::Slot { host, .. } => {
                // SAFETY: `check` keeps the span inside the region, which
                // lies inside one slot of a mapped heap.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        from.as_ptr(),
                        host.0.as_ptr().add(usize_of(at)),
                        from.len(),
                    );
                }
                Ok(())
            }
        }
    }

    pub fn read(&self, at: u64, into: &mut [u8]) -> Result<()> {
        self.check(at, into.len())?;
        match self.place {
            Place::Own => self.buffer.read(self.offset + at, into),
            Place::Slot { host, .. } => {
                // SAFETY: as `write`.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        host.0.as_ptr().add(usize_of(at)),
                        into.as_mut_ptr(),
                        into.len(),
                    );
                }
                Ok(())
            }
        }
    }

    fn zero(&mut self) -> Result<()> {
        let bytes = usize_of(self.bytes);
        if bytes > 0 {
            self.write(0, &vec![0u8; bytes])?;
        }
        Ok(())
    }

    /// The GPU address of byte `at`.
    pub fn address_at(&self, at: u64) -> Option<u64> {
        self.buffer.address_at(self.offset + at)
    }
}

fn usize_of(n: u64) -> usize {
    usize::try_from(n).unwrap_or(usize::MAX)
}

/// Lays tables out in `Instance` slots of `slot_bytes`: each aligned to
/// [`TABLE_ALIGN`] and never straddling a slot. Returns each table's slot
/// index and offset, and the slots taken; `None` when a table is larger
/// than a slot.
#[must_use]
pub fn pack(sizes: &[u64], slot_bytes: u64) -> Option<(Vec<(u32, u64)>, u32)> {
    let mut places = Vec::with_capacity(sizes.len());
    let (mut slot, mut cursor) = (0u32, 0u64);
    let mut used = false;
    for &bytes in sizes {
        let bytes = bytes.max(1);
        if bytes > slot_bytes {
            return None;
        }
        let at = cursor.next_multiple_of(TABLE_ALIGN);
        if at + bytes > slot_bytes {
            slot += 1;
            cursor = 0;
        } else {
            cursor = at;
        }
        places.push((slot, cursor));
        cursor += bytes;
        used = true;
    }
    Some((places, if used { slot + 1 } else { 0 }))
}

/// Where an instance's bind-time tables go, taken in the order
/// [`super::launch::Prepared::table_sizes`] states them, stage by stage.
pub enum Tables<'a> {
    Own(&'a Context),
    Pooled {
        pools: &'a Pools,
        shape: ShapeId,
        slots: &'a [u32],
        places: std::vec::IntoIter<(u32, u64)>,
    },
}

impl<'a> Tables<'a> {
    /// Tables of `sizes` in `slots` (slot refs of `shape`).
    pub fn pooled(
        pools: &'a Pools,
        shape: ShapeId,
        slots: &'a [u32],
        sizes: &[u64],
    ) -> Result<Tables<'a>> {
        let slot_bytes = pools.slot_bytes(shape)?;
        let (places, need) = pack(sizes, slot_bytes).ok_or_else(|| {
            Fault::program(
                "program::held",
                format!("a launch table is larger than a {slot_bytes}-byte instance slot"),
            )
        })?;
        if need as usize != slots.len() {
            return Err(Fault::program(
                "program::held",
                format!(
                    "this instance's tables take {need} instance slot(s) and the binding \
                     names {}",
                    slots.len()
                ),
            ));
        }
        Ok(Tables::Pooled {
            pools,
            shape,
            slots,
            places: places.into_iter(),
        })
    }

    /// The next table, of `bytes`, zeroed.
    pub fn take(&mut self, bytes: u64) -> Result<Held> {
        match self {
            Tables::Own(device) => Held::own(device, bytes),
            Tables::Pooled {
                pools,
                shape,
                slots,
                places,
            } => {
                let (slot, at) = places.next().ok_or_else(|| {
                    Fault::program(
                        "program::held",
                        "a stage took more launch tables than its program's facts state",
                    )
                })?;
                Held::in_slot(pools, *shape, slots[slot as usize], at, bytes.max(1))
            }
        }
    }

    /// Refuses tables the facts stated and the stages did not take.
    pub fn finish(self) -> Result<()> {
        if let Tables::Pooled { mut places, .. } = self
            && places.next().is_some()
        {
            return Err(Fault::program(
                "program::held",
                "a program's facts state more launch tables than its stages took",
            ));
        }
        Ok(())
    }
}

/// Where a ring's cells go: `cells` cells of `stride` bytes, `per_span` of
/// them in each region (one region for a ring of its own, one a slot for a
/// pooled ring).
#[derive(Debug, Clone)]
pub struct Cells {
    spans: Vec<Held>,
    per_span: u64,
    stride: u64,
    cells: u64,
    /// The slots of a ring held as one run (a cell past a slot), whose one
    /// span starts at the run's first slot.
    run: Option<u32>,
}

/// The `Channel` slots a ring takes as one run where its cell is larger
/// than a slot (`Footprint::contiguous`), or `None` where the run would
/// not fit one chunk of `per_chunk` slots.
#[must_use]
pub fn ring_run(cells: u64, stride: u64, slot_bytes: u64, per_chunk: u32) -> Option<u32> {
    let slots = u32::try_from(cells.checked_mul(stride)?.div_ceil(slot_bytes)).ok()?;
    (slots <= per_chunk).then_some(slots)
}

/// Whether `slots` are refs `first .. first + n` in order.
fn consecutive(slots: &[u32]) -> bool {
    slots
        .windows(2)
        .all(|pair| pair[0].checked_add(1) == Some(pair[1]))
}

/// The `Channel` slots a ring of `cells` cells of `stride` bytes takes, or
/// `None` where a cell is larger than a slot (the engine keeps that ring).
#[must_use]
pub fn ring_slots(cells: u64, stride: u64, slot_bytes: u64) -> Option<u32> {
    let per = slot_bytes.checked_div(stride).filter(|&per| per > 0)?;
    u32::try_from(cells.div_ceil(per)).ok()
}

impl Cells {
    pub fn own(device: &Context, cells: u64, stride: u64) -> Result<Cells> {
        let bytes = cells
            .checked_mul(stride)
            .ok_or_else(|| Fault::program("program::held", "a ring past what a u64 counts"))?;
        Ok(Cells {
            spans: vec![Held::own(device, bytes)?],
            per_span: cells.max(1),
            stride,
            cells,
            run: None,
        })
    }

    /// The ring in `slots` (`Channel` slot refs), cells packed whole.
    pub fn pooled(
        pools: &Pools,
        shape: ShapeId,
        slots: &[u32],
        cells: u64,
        stride: u64,
    ) -> Result<Cells> {
        let slot_bytes = pools.slot_bytes(shape)?;
        let Some(need) = ring_slots(cells, stride, slot_bytes) else {
            return Cells::run(pools, shape, slots, cells, stride);
        };
        if slots.len() != need as usize {
            return Err(Fault::program(
                "program::held",
                format!(
                    "a ring of {cells} cells of {stride} bytes takes {need} channel slot(s) \
                     and the binding names {}",
                    slots.len()
                ),
            ));
        }
        let per_span = slot_bytes / stride;
        let spans = slots
            .iter()
            .enumerate()
            .map(|(k, &slot)| {
                let here = (cells - k as u64 * per_span).min(per_span);
                Held::in_slot(pools, shape, slot, 0, here * stride)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Cells {
            spans,
            per_span,
            stride,
            cells,
            run: None,
        })
    }

    /// A ring whose cell is past a slot, in one run of `slots`.
    fn run(pools: &Pools, shape: ShapeId, slots: &[u32], cells: u64, stride: u64) -> Result<Cells> {
        let slot_bytes = pools.slot_bytes(shape)?;
        let per_chunk = pools.slots_per_chunk(shape)?;
        let bytes = cells
            .checked_mul(stride)
            .ok_or_else(|| Fault::program("program::held", "a ring past what a u64 counts"))?;
        let need = ring_run(cells, stride, slot_bytes, per_chunk).ok_or_else(|| {
            Fault::program(
                "program::held",
                format!(
                    "a ring of {bytes} bytes is larger than a chunk of {per_chunk} \
                     {slot_bytes}-byte channel slots, and is the engine's to keep"
                ),
            )
        })?;
        if slots.len() != need as usize || !consecutive(slots) {
            return Err(Fault::program(
                "program::held",
                format!(
                    "a ring of {cells} cells of {stride} bytes takes a run of {need} adjacent \
                     channel slots and the binding names {slots:?}"
                ),
            ));
        }
        Ok(Cells {
            spans: vec![Held::in_run(pools, shape, slots[0], need, bytes)?],
            per_span: cells.max(1),
            stride,
            cells,
            run: Some(need),
        })
    }

    /// Cell `index`'s region and the byte it starts at there.
    pub fn locate(&self, index: u64) -> Result<(&Held, u64)> {
        if index >= self.cells {
            return Err(Fault::program(
                "program::held",
                format!("cell {index} of a ring of {}", self.cells),
            ));
        }
        let span = &self.spans[usize_of(index / self.per_span)];
        Ok((span, (index % self.per_span) * self.stride))
    }

    pub fn write(&mut self, index: u64, from: &[u8]) -> Result<()> {
        let (_, at) = self.locate(index)?;
        let span = usize_of(index / self.per_span);
        self.spans[span].write(at, from)
    }

    pub fn read(&self, index: u64, into: &mut [u8]) -> Result<()> {
        let (span, at) = self.locate(index)?;
        span.read(at, into)
    }

    pub fn rebind(&mut self, pools: &Pools) -> Result<()> {
        self.spans
            .iter_mut()
            .try_for_each(|span| span.rebind(pools))
    }

    /// Moves a pooled ring to `slots`, one a span, in order.
    pub fn retarget(&mut self, pools: &Pools, slots: &[u32]) -> Result<()> {
        if let Some(need) = self.run {
            if slots.len() != need as usize || !consecutive(slots) {
                return Err(Fault::program(
                    "program::held",
                    format!(
                        "a ring in a run of {need} slot(s) rebound to {slots:?}, which is \
                         not a run of as many"
                    ),
                ));
            }
            let per_chunk = pools.slots_per_chunk(CHANNEL)?;
            let (first, last) = (slots[0], slots[slots.len() - 1]);
            if first / per_chunk != last / per_chunk {
                return Err(Fault::program(
                    "program::held",
                    format!("a ring's run {first}..={last} crosses a chunk"),
                ));
            }
            return self.spans[0].retarget(pools, first);
        }
        if self.spans.iter().any(|span| !span.is_pooled()) || slots.len() != self.spans.len() {
            return Err(Fault::program(
                "program::held",
                format!(
                    "a ring in {} span(s) rebound to {} channel slot(s)",
                    self.spans.len(),
                    slots.len()
                ),
            ));
        }
        self.spans
            .iter_mut()
            .zip(slots)
            .try_for_each(|(span, &slot)| span.retarget(pools, slot))
    }

    #[must_use]
    pub fn own_bytes(&self) -> u64 {
        self.spans.iter().map(Held::own_bytes).sum()
    }

    #[must_use]
    pub fn cells(&self) -> u64 {
        self.cells
    }

    /// The buffers the cells lie in, each once (a pooled ring's are all the
    /// pool's view).
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        self.spans[0].buffer()
    }
}

#[cfg(test)]
mod tests {
    use super::{pack, ring_slots};

    #[test]
    fn tables_pack_aligned_and_never_straddle_a_slot() {
        let (places, slots) = pack(&[100, 40, 4000, 8], 4096).expect("every table fits a slot");
        assert_eq!(places, vec![(0, 0), (0, 112), (1, 0), (1, 4000)]);
        assert_eq!(slots, 2);
        assert_eq!(pack(&[], 4096), Some((Vec::new(), 0)));
        assert_eq!(pack(&[5000], 4096), None);
    }

    #[test]
    fn ring_cells_pack_whole_into_channel_slots() {
        assert_eq!(ring_slots(2, 16, 1024), Some(1));
        assert_eq!(ring_slots(12, 192, 1024), Some(3));
        assert_eq!(ring_slots(5, 600_016, 1024), None);
    }

    #[test]
    fn a_cell_past_a_slot_takes_one_run_inside_a_chunk() {
        use super::{consecutive, ring_run};
        // A vocab-wide probabilities cell, capacity 4: five cells.
        assert_eq!(ring_run(5, 607_744, 1024, 16_384), Some(2968));
        assert_eq!(ring_run(5, 607_744, 1024, 2048), None);
        assert!(consecutive(&[7, 8, 9]));
        assert!(!consecutive(&[7, 9]));
    }
}
