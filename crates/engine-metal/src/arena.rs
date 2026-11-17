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
    /// Each rung's rs scratch, under its staged regions and over its
    /// carve: what a fire of the rung's tokens and lanes replays its
    /// recurrent rows in (0 for a model with none).
    rs: Vec<u64>,
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
/// `arms` frames' staged regions `io`, its rs scratch `rs` and its carve,
/// `(tokens, chunks)` ascending. A rung's span holds every narrower rung's
/// (the spans nest from the range's top), so its chunks are the most any
/// rung up to it takes: an arena map is not monotone in rows (Qwen3-14B's
/// 8192-token carve packs 98 MB under its 4096-token one), and a fire at a
/// rung touches only its own map, inside that span.
#[must_use]
pub fn act_chunks(
    rungs: &[(u32, ArenaMap)],
    scratch: u64,
    io: &[(u64, u64)],
    rs: &[u64],
    arms: u64,
    chunk_bytes: u64,
) -> Vec<(u32, u32)> {
    let mut most = 0u32;
    rungs
        .iter()
        .zip(io)
        .zip(rs)
        .map(|(((tokens, map), &io), &rs)| {
            let bytes = room(scratch) + arms * io_room(io) + room(rs) + room(map.bytes);
            let chunks = u32::try_from(bytes.div_ceil(chunk_bytes.max(1))).unwrap_or(u32::MAX);
            most = most.max(chunks);
            (*tokens, most)
        })
        .collect()
}

impl Arena {
    /// The arena over the span `[offset, offset + bytes)` of `store`, the
    /// pool's buffer: `scratch` plane bytes at its top, then each rung's
    /// staged regions (`io`, `arms` frames' of them), its rs scratch (`rs`)
    /// and then its carve, the last rung the whole arena. It starts at the
    /// largest rung, the one the runtime's pool starts at.
    #[must_use]
    pub fn over(
        store: Buffer,
        (offset, bytes): (u64, u64),
        scratch: u64,
        rungs: Vec<(u32, ArenaMap)>,
        io: Vec<(u64, u64)>,
        rs: Vec<u64>,
        arms: u64,
    ) -> Arena {
        let rungs_len = rungs.len();
        Arena {
            store,
            end: offset + bytes,
            scratch: room(scratch),
            rungs,
            io,
            arms: arms.max(1),
            current: rungs_len.saturating_sub(1),
            rs,
        }
    }

    /// The bytes the kernel scratch plane takes at the span's top.
    #[must_use]
    pub fn scratch_room(&self) -> u64 {
        self.scratch
    }

    /// Every rung's tokens, narrowest first.
    #[must_use]
    pub fn rung_tokens(&self) -> Vec<u32> {
        self.rungs.iter().map(|(tokens, _)| *tokens).collect()
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

    /// Each rung's tokens and the bytes of its rs scratch.
    #[must_use]
    pub fn rs_rooms(&self) -> Vec<(u32, u64)> {
        self.rungs
            .iter()
            .zip(&self.rs)
            .map(|((tokens, _), &bytes)| (*tokens, bytes))
            .collect()
    }

    /// The tokens of the rung frames stage in now: what its rs scratch
    /// holds of the rows a fire's lanes carry in.
    #[must_use]
    pub fn current_tokens(&self) -> u32 {
        self.rungs.get(self.current).map_or(0, |(tokens, _)| *tokens)
    }

    /// The staged regions' bytes at rung `index`, every frame's.
    fn io_total(&self, index: usize) -> u64 {
        self.arms * self.io.get(index).copied().map_or(0, io_room)
    }

    /// What lies over rung `index`'s carve under the plane: its staged
    /// regions and its rs scratch.
    fn over_carve(&self, index: usize) -> u64 {
        self.io_total(index) + room(self.rs.get(index).copied().unwrap_or(0))
    }

    /// Where the current rung's rs scratch lies in the span's buffer, and
    /// its bytes.
    #[must_use]
    pub fn rs_region(&self) -> (u64, u64) {
        let bytes = self.rs.get(self.current).copied().unwrap_or(0);
        (
            self.scratch_at() - self.io_total(self.current) - room(bytes),
            bytes,
        )
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

    /// Where a fire's carve starts under the current rung's staged regions
    /// and rs scratch.
    fn carve_top(&self) -> u64 {
        self.scratch_at() - self.over_carve(self.current)
    }

    /// The largest rung's tokens: where the runtime starts the pool's rung,
    /// and the shell stands from the load.
    #[must_use]
    pub fn top_tokens(&self) -> u32 {
        self.rungs.last().map_or(1, |(tokens, _)| *tokens)
    }

    /// Where the span ends in its buffer: the plane's end.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.end
    }

    /// The bytes from the span's end a fire of `rows` takes at the current
    /// rung: the plane, the rung's staged regions and its carve.
    #[must_use]
    pub fn reach(&self, rows: FireRows) -> u64 {
        self.scratch + self.over_carve(self.current) + room(self.carve_for(rows).bytes)
    }

    /// The bytes from the span's end a rung of `tokens` takes: the plane,
    /// its staged regions and the carve its widest fire lays.
    #[must_use]
    pub fn rung_reach(&self, tokens: u32) -> u64 {
        let index = self.index(tokens);
        self.scratch
            + self.over_carve(index)
            + self.rungs.get(index).map_or(0, |(_, map)| room(map.bytes))
    }

    /// The bytes from the span's end the largest rung takes: what the
    /// load backs first, the rung the shell stands at from it.
    #[must_use]
    pub fn top_reach(&self) -> u64 {
        self.rung_reach(self.top_tokens())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn carve(bytes: u64) -> ArenaMap {
        ArenaMap {
            placements: Vec::new(),
            spans: Vec::new(),
            live_in: Vec::new(),
            bytes,
        }
    }

    /// Qwen3-14B's rungs on an M5 Pro (chunk 2688000, two frames): the
    /// 8192-token carve packs under the 4096-token one. Each rung states the
    /// most any rung up to it takes, so the spans ascend and nest, and a
    /// rung whose own carve is smaller keeps the larger span below it.
    #[test]
    fn a_rung_states_the_most_any_rung_up_to_it_takes() {
        let rungs = vec![
            (4096, carve(8_000_503_808)),
            (8192, carve(7_902_068_736)),
            (10240, carve(8_199_864_320)),
        ];
        let io = [(173_801_472, 0), (190_775_296, 0), (199_294_976, 0)];
        let chunks = act_chunks(&rungs, 0, &io, &[0, 0, 0], 2, 2_688_000);
        let own = |at: usize| {
            (room(rungs[at].1.bytes) + 2 * io_room(io[at])).div_ceil(2_688_000) as u32
        };
        assert!(own(1) < own(0), "the 8192 rung's own span is the smaller");
        assert_eq!(chunks[0], (4096, own(0)));
        assert_eq!(chunks[1], (8192, own(0)), "it keeps the 4096 rung's span");
        assert_eq!(chunks[2], (10240, own(2)));
        assert!(chunks.windows(2).all(|pair| pair[0].0 < pair[1].0 && pair[0].1 <= pair[1].1));
    }
}
