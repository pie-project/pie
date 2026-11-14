use kernels_metal::Tensor;
use poem_compiler::{ArenaMap, FireRows};
use poem_exec::store::arena::rect;
use poem_ir::ValueId;

use crate::device::{Buffer, Handles};
use crate::error::Result;
use crate::run::SlotTable;

/// Where a carve's bytes and the kernel scratch plane start inside the span
/// are kept on this boundary, so every value's offset keeps the alignment it
/// was carved with.
pub const ALIGN: u64 = 64 << 10;

/// A fire's activation: the top of the pool's range (REDESIGN 3c), the
/// activation span the runtime keeps state out of at each rung.
///
/// The kernel scratch plane takes the span's top `scratch` bytes for every
/// fire; under it, a fire carves the smallest rung's arena that holds it,
/// laid so it ends where the plane starts. A rung's activation is the plane
/// and its carve, so a lower rung touches only the span's top and state may
/// take the chunks under it. The place each carve takes is fixed, so a fire
/// stays good at every rung it fits.
#[derive(Debug)]
pub struct Arena {
    /// The pool's buffer.
    store: Buffer,
    /// The span's end in the pool's buffer, where the plane ends.
    end: u64,
    /// The bytes the kernel scratch plane takes at the span's top.
    scratch: u64,
    /// For fires of up to so many tokens (and lanes), the carve laid for
    /// them, ascending; the last the whole arena, for every fire.
    rungs: Vec<(u32, ArenaMap)>,
    /// Each rung's staged regions, a frame in flight each, under the plane:
    /// `(inputs, readout)` bytes of one frame's, on the [`ALIGN`] boundary.
    io: Vec<(u64, u64)>,
    /// The frames in flight.
    arms: u64,
    /// The rung the runtime set: whose regions frames stage in now.
    current: usize,
}

/// `bytes` on the [`ALIGN`] boundary.
#[must_use]
pub fn room(bytes: u64) -> u64 {
    bytes.next_multiple_of(ALIGN)
}

/// One frame's staged regions at a rung, `(inputs, readout)`, as the span
/// lays them.
#[must_use]
pub fn io_room((inputs, readout): (u64, u64)) -> u64 {
    room(inputs) + room(readout)
}

/// The span's chunks of `chunk_bytes` each rung takes: the `scratch` plane,
/// `arms` frames' staged regions `io` and its carve, `(tokens, chunks)`
/// ascending.
#[must_use]
pub fn act_chunks(
    rungs: &[(u32, ArenaMap)],
    scratch: u64,
    io: &[(u64, u64)],
    arms: u64,
    chunk_bytes: u64,
) -> Vec<(u32, u32)> {
    rungs
        .iter()
        .zip(io)
        .map(|((tokens, map), &io)| {
            let bytes = room(scratch) + arms * io_room(io) + room(map.bytes);
            (
                *tokens,
                u32::try_from(bytes.div_ceil(chunk_bytes.max(1))).unwrap_or(u32::MAX),
            )
        })
        .collect()
}

impl Arena {
    /// The arena over the span `[offset, offset + bytes)` of `store`, the
    /// pool's buffer: `scratch` plane bytes at its top, then each rung's
    /// staged regions (`io`, `arms` frames' of them), then its carve, the
    /// last rung the whole arena. It starts at the smallest rung.
    #[must_use]
    pub fn over(
        store: Buffer,
        (offset, bytes): (u64, u64),
        scratch: u64,
        rungs: Vec<(u32, ArenaMap)>,
        io: Vec<(u64, u64)>,
        arms: u64,
    ) -> Arena {
        Arena {
            store,
            end: offset + bytes,
            scratch: room(scratch),
            rungs,
            io,
            arms: arms.max(1),
            current: 0,
        }
    }

    /// The bytes the kernel scratch plane takes at the span's top.
    #[must_use]
    pub fn scratch_room(&self) -> u64 {
        self.scratch
    }

    /// The rung that carries a fire of `tokens`: the smallest at or above
    /// it, else the widest.
    #[must_use]
    pub fn index(&self, tokens: u32) -> usize {
        self.rungs
            .iter()
            .position(|(rung, _)| *rung >= tokens)
            .unwrap_or(self.rungs.len().saturating_sub(1))
    }

    /// Frames stage in rung `tokens`'s regions from now on: the caller has
    /// drained every frame staged in the rung before.
    pub fn set_current(&mut self, tokens: u32) {
        self.current = self.index(tokens);
    }

    /// The rung frames stage in now.
    #[must_use]
    pub fn current(&self) -> usize {
        self.current
    }

    /// The staged regions' bytes at rung `index`, every frame's.
    fn io_total(&self, index: usize) -> u64 {
        self.arms * self.io.get(index).copied().map_or(0, io_room)
    }

    /// Where frame `arm`'s staged inputs and readout lie at rung `index` in
    /// the span's buffer, and their bytes: `((inputs, bytes), (readout,
    /// bytes))`.
    #[must_use]
    pub fn regions(&self, index: usize, arm: u64) -> ((u64, u64), (u64, u64)) {
        let (inputs, readout) = self.io.get(index).copied().unwrap_or((0, 0));
        let at = self.scratch_at() - self.io_total(index) + arm * io_room((inputs, readout));
        ((at, inputs), (at + room(inputs), readout))
    }

    /// Where the kernel scratch plane starts in the span's buffer.
    #[must_use]
    pub fn scratch_at(&self) -> u64 {
        self.end - self.scratch
    }

    /// The pool's buffer, every carve's.
    #[must_use]
    pub fn store(&self) -> &Buffer {
        &self.store
    }

    /// The staged regions' bytes at the widest rung, every frame's.
    #[must_use]
    pub fn io_bytes(&self) -> u64 {
        self.io_total(self.rungs.len().saturating_sub(1))
    }

    /// The rungs, their staged regions laid.
    #[must_use]
    pub fn rung_count(&self) -> usize {
        self.rungs.len()
    }

    /// The whole arena's bytes: the widest carve.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.rungs.last().map_or(0, |(_, map)| map.bytes)
    }

    /// The carve a fire of `rows` takes: the smallest rung that holds its
    /// tokens and lanes, else the whole arena.
    #[must_use]
    pub fn carve_for(&self, rows: FireRows) -> &ArenaMap {
        let last = self.rungs.len().saturating_sub(1);
        let at = self
            .rungs
            .iter()
            .position(|(tokens, _)| {
                u64::from(*tokens) >= rows.tokens && u64::from(*tokens) >= rows.lanes
            })
            .unwrap_or(last);
        &self.rungs[at].1
    }

    /// Where a fire's carve starts under the current rung's staged regions.
    fn carve_top(&self) -> u64 {
        self.scratch_at() - self.io_total(self.current)
    }

    /// The bytes from the span's end a fire of `rows` takes at the current
    /// rung: the plane, the rung's staged regions and its carve.
    #[must_use]
    pub fn reach(&self, rows: FireRows) -> u64 {
        self.scratch + self.io_total(self.current) + room(self.carve_for(rows).bytes)
    }

    /// The bytes from the span's end a rung of `tokens` takes: the plane,
    /// its staged regions and the carve its widest fire lays.
    #[must_use]
    pub fn rung_reach(&self, tokens: u32) -> u64 {
        let index = self.index(tokens);
        self.scratch
            + self.io_total(index)
            + self.rungs.get(index).map_or(0, |(_, map)| room(map.bytes))
    }

    /// The bytes from the span's end the smallest rung takes: the floor a
    /// refused raise falls back to.
    #[must_use]
    pub fn floor_reach(&self) -> u64 {
        self.rung_reach(0)
    }

    /// A fire's slots: [`Arena::carve_for`]'s carve, laid ending where the
    /// current rung's staged regions start.
    pub fn fire_slots(&self, handles: &Handles, rows: FireRows) -> Result<SlotTable> {
        let map = self.carve_for(rows);
        carve(
            handles,
            &self.store,
            self.carve_top() - room(map.bytes),
            map,
            rows,
        )
    }
}

/// `map`'s rectangles for `fire`, laid from `base` in `store`.
pub fn carve(
    handles: &Handles,
    store: &Buffer,
    base: u64,
    map: &ArenaMap,
    fire: FireRows,
) -> Result<SlotTable> {
    let mut rows: Vec<Option<Tensor>> = Vec::with_capacity(map.placements.len());
    for value in 0..map.placements.len() {
        let value = ValueId(value as u32);
        rows.push(match rect(map, value, fire) {
            Some(rect) => Some(Tensor::new(
                handles.bind(store, base + rect.offset, rect.bytes)?,
                rect.rows,
                rect.width,
                rect.dtype,
            )),
            None => None,
        });
    }
    Ok(SlotTable(rows))
}

#[must_use]
pub fn capacities(map: &ArenaMap) -> Vec<u32> {
    (0..map.placements.len())
        .map(|at| {
            let root = map.root(ValueId(at as u32));
            let Some(poem_compiler::Placement::Arena {
                bytes,
                width,
                dtype,
                ..
            }) = map.placements.get(root.0 as usize)
            else {
                return 0;
            };
            let row = width.saturating_mul(poem_compiler::arena::elem_bytes(*dtype).unwrap_or(0));
            if row == 0 {
                return 0;
            }
            u32::try_from(bytes / row).unwrap_or(u32::MAX)
        })
        .collect()
}
