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
}

/// `bytes` on the [`ALIGN`] boundary.
#[must_use]
pub fn room(bytes: u64) -> u64 {
    bytes.next_multiple_of(ALIGN)
}

/// The span's chunks of `chunk_bytes` each rung takes: its carve under the
/// `scratch` plane, `(tokens, chunks)` ascending.
#[must_use]
pub fn act_chunks(rungs: &[(u32, ArenaMap)], scratch: u64, chunk_bytes: u64) -> Vec<(u32, u32)> {
    rungs
        .iter()
        .map(|(tokens, map)| {
            let bytes = room(scratch) + room(map.bytes);
            (
                *tokens,
                u32::try_from(bytes.div_ceil(chunk_bytes.max(1))).unwrap_or(u32::MAX),
            )
        })
        .collect()
}

impl Arena {
    /// The arena over the span `[offset, offset + bytes)` of `store`, the
    /// pool's buffer: `scratch` plane bytes at its top and `rungs` laid
    /// under them, the last the whole arena.
    #[must_use]
    pub fn over(
        store: Buffer,
        (offset, bytes): (u64, u64),
        scratch: u64,
        rungs: Vec<(u32, ArenaMap)>,
    ) -> Arena {
        Arena {
            store,
            end: offset + bytes,
            scratch: room(scratch),
            rungs,
        }
    }

    /// The bytes the kernel scratch plane takes at the span's top.
    #[must_use]
    pub fn scratch_room(&self) -> u64 {
        self.scratch
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

    /// The smallest rung's tokens: where the runtime starts an on-demand
    /// pool's rung.
    #[must_use]
    pub fn floor_tokens(&self) -> u32 {
        self.rungs.first().map_or(1, |(tokens, _)| *tokens)
    }

    /// Where the span ends in its buffer: the plane's end.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.end
    }

    /// Where a fire of `rows` lays its guests' scratch from, down: under
    /// its own carve.
    #[must_use]
    pub fn guest_top(&self, rows: FireRows) -> u64 {
        self.scratch_at() - room(self.carve_for(rows).bytes)
    }

    /// [`Arena::guest_top`] of the widest fire a rung of `tokens` carries.
    #[must_use]
    pub fn rung_guest_top(&self, tokens: u32) -> u64 {
        self.guest_top(FireRows {
            tokens: u64::from(tokens),
            lanes: u64::from(tokens),
            ..FireRows::default()
        })
    }

    /// The bytes from the span's end a fire of `rows` takes: the scratch
    /// plane and its carve.
    #[must_use]
    pub fn reach(&self, rows: FireRows) -> u64 {
        self.scratch + room(self.carve_for(rows).bytes)
    }

    /// The bytes from the span's end a rung of `tokens` takes: the plane
    /// and the carve its widest fire lays, never under the floor rung's.
    #[must_use]
    pub fn rung_reach(&self, tokens: u32) -> u64 {
        let rows = FireRows {
            tokens: u64::from(tokens),
            lanes: u64::from(tokens),
            ..FireRows::default()
        };
        self.reach(rows).max(self.floor_reach())
    }

    /// The bytes from the span's end the smallest rung's fire takes: the
    /// floor a refused raise falls back to.
    #[must_use]
    pub fn floor_reach(&self) -> u64 {
        self.scratch + self.rungs.first().map_or(0, |(_, map)| room(map.bytes))
    }

    /// A fire's slots: [`Arena::carve_for`]'s carve, laid ending where the
    /// scratch plane starts.
    pub fn fire_slots(&self, handles: &Handles, rows: FireRows) -> Result<SlotTable> {
        let map = self.carve_for(rows);
        carve(
            handles,
            &self.store,
            self.scratch_at() - room(map.bytes),
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
