#![allow(unsafe_code)]
use kernels_cuda::Tensor;
use poem_compiler::ArenaMap;
use poem_exec::store::arena::rect;
use poem_ir::ValueId;

use crate::error::{Fault, Result};
use crate::run::SlotTable;
use crate::store::Pools;

/// A fire's activation: the span under the state pool's end
/// ([`Pools::activation_base`]), backed as far as the load's fires reach.
#[derive(Debug)]
pub struct Arena {
    committed: u64,
    reserved: u64,
    /// The activation rungs (SPEC decision 6): for fires of at most so many
    /// tokens (and lanes), the carve laid out for them, smallest first.
    /// Each lies at the reservation's end, so a rung's fires touch only the
    /// top of the span and state may take what lies under it.
    rungs: Vec<(u32, ArenaMap)>,
    /// The lanes every fire carves for, whatever its own: a fire that
    /// replays a recorded body lays its lane tables at the body's capacity
    /// (`window::lane_capacity`), so its carve is the smallest rung's that
    /// holds that many lanes. 0 where fires carve for their own lanes.
    lane_reach: u32,
    /// The bytes kept at the reservation's end above every carve: a load
    /// that keeps rings lays its fires' own-row pages there
    /// (`store::OwnRows`).
    top: u64,
}

impl Arena {
    /// `bytes`, the most any arena map this load carves asks for: what the
    /// pool was reserved its activation chunks for.
    #[must_use]
    pub fn reserve(bytes: u64) -> Arena {
        Arena {
            committed: 0,
            reserved: bytes,
            rungs: Vec::new(),
            lane_reach: 0,
            top: 0,
        }
    }

    /// With `bytes` of the reservation's end kept above every carve.
    #[must_use]
    pub fn keeping_top(mut self, bytes: u64) -> Arena {
        self.top = bytes;
        self
    }

    /// Every fire of this load carves for `lanes` lanes at least: what the
    /// rungs state to the runtime follows the carve a fire really lays.
    pub fn reach_lanes(&mut self, lanes: u32) {
        self.lane_reach = lanes;
    }

    /// With `rungs`, each a carve no larger than the reservation, smallest
    /// first: a fire carves the smallest that holds it, at the
    /// reservation's end ([`Arena::fire_slots`]).
    #[must_use]
    pub fn with_rungs(mut self, mut rungs: Vec<(u32, ArenaMap)>) -> Arena {
        rungs.retain(|(_, map)| map.bytes + self.top <= self.reserved);
        rungs.sort_by_key(|&(tokens, _)| tokens);
        self.rungs = rungs;
        self
    }

    /// Whether this load lays carves at the reservation's end.
    #[must_use]
    pub fn has_rungs(&self) -> bool {
        !self.rungs.is_empty()
    }

    /// Each rung's tokens and the bytes of the span under what a fire of
    /// it takes at the top (its carve, or `forked`'s where a fire that forks
    /// is wider) that state may take at that rung: no more than the prefix
    /// backed from the span's start ([`Arena::ensure`]), so state never
    /// lands where nothing is backed.
    #[must_use]
    pub fn rungs(&self, forked: Option<&ArenaMap>) -> Vec<(u32, u64)> {
        self.rungs
            .iter()
            .map(|(tokens, _)| {
                // The carve a fire cut to this rung lays: the smallest
                // rung's that holds its tokens and the lanes it carves
                // for, the whole arena past every rung
                // ([`Arena::fire_slots`]).
                let carve = self
                    .rungs
                    .iter()
                    .find(|(rung, _)| rung >= tokens && *rung >= self.lane_reach)
                    .map_or(self.reserved - self.top, |(_, map)| map.bytes);
                let bytes = carve.max(forked.map_or(0, |forked| forked.bytes)) + self.top;
                (
                    *tokens,
                    self.reserved.saturating_sub(bytes).min(self.committed),
                )
            })
            .collect()
    }

    /// The rung a fire of `tokens` tokens and `lanes` lanes carves (its
    /// tokens; `top` where it carves the whole arena) and the bytes of the
    /// span under that carve, as [`Arena::rungs`] states them.
    #[must_use]
    pub fn rung_of(
        &self,
        forked: Option<&ArenaMap>,
        tokens: u64,
        lanes: u64,
        top: u32,
    ) -> (u32, u64) {
        self.rungs(forked)
            .into_iter()
            .find(|&(rung, _)| u64::from(rung) >= tokens && u64::from(rung) >= lanes)
            .unwrap_or((top, 0))
    }

    /// Every rung's tokens and the bytes under its carve, the whole arena's
    /// (`top` tokens, nothing under it) last.
    #[must_use]
    pub fn every_rung(&self, forked: Option<&ArenaMap>, top: u32) -> Vec<(u32, u64)> {
        let mut rungs = self.rungs(forked);
        rungs.push((top, 0));
        rungs
    }

    /// The bytes of the span a fire of `tokens` tokens and `lanes` lanes
    /// carves (its rung's carve, or `forked`'s where a fire that forks is
    /// wider; the whole reservation past every rung): what the span must be
    /// backed to before the fire runs ([`Arena::ensure`]).
    #[must_use]
    pub fn carve_of(&self, forked: Option<&ArenaMap>, tokens: u64, lanes: u64) -> u64 {
        let carve = self
            .rungs
            .iter()
            .find(|(rung, _)| {
                u64::from(*rung) >= tokens && u64::from(*rung) >= lanes && *rung >= self.lane_reach
            })
            .map_or(self.reserved - self.top, |(_, map)| map.bytes);
        carve.max(forked.map_or(0, |forked| forked.bytes)) + self.top
    }

    /// No smaller rung: every fire's arena at the span's start.
    pub fn drop_rungs(&mut self) {
        self.rungs.clear();
    }

    /// The reservation's bytes: the most any carve this load lays asks for.
    #[must_use]
    pub fn reserved(&self) -> u64 {
        self.reserved
    }

    pub fn ensure(&mut self, pools: &mut Pools, bytes: u64) -> Result<()> {
        if bytes <= self.committed {
            return Ok(());
        }
        pools.hold_activation(bytes, false)?;
        self.committed = bytes;
        Ok(())
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.committed
    }

    #[must_use]
    pub fn slots(
        &self,
        pools: &Pools,
        map: &ArenaMap,
        rows: poem_compiler::FireRows,
    ) -> Result<SlotTable> {
        carve(pools.activation_base(), map, rows)
    }

    /// A fire's slots: `map` (the arena in order, or the forked one) as
    /// laid at the span's start, or, where this load states rungs, the
    /// smallest rung's carve that holds `rows` (a fire that forks keeps its
    /// own) at the reservation's end. The place each carve takes is fixed,
    /// so a body recorded at any rung stays good at every rung its fire
    /// fits.
    #[must_use]
    pub fn fire_slots(
        &self,
        pools: &Pools,
        map: &ArenaMap,
        forked: bool,
        rows: poem_compiler::FireRows,
    ) -> Result<SlotTable> {
        if self.rungs.is_empty() {
            return self.slots(pools, map, rows);
        }
        let chosen = if forked {
            map
        } else {
            self.rungs
                .iter()
                .find(|(tokens, _)| {
                    u64::from(*tokens) >= rows.tokens && u64::from(*tokens) >= rows.lanes
                })
                .map_or(map, |(_, rung)| rung)
        };
        let base =
            pools.activation_base() + self.reserved.saturating_sub(self.top + chosen.bytes);
        carve(base, chosen, rows)
    }

    /// Reads `into.len()` bytes at `at` on `stream` (the shell's own), and
    /// waits for it. Not on the legacy stream: that waits on every blocking
    /// stream of the context, a peer rank's on the same device too, and a
    /// peer spinning in its next step's collective waits on this rank's
    /// launch, which queues behind the legacy copy.
    pub fn read(
        &self,
        pools: &Pools,
        stream: *mut core::ffi::c_void,
        at: u64,
        into: &mut [u8],
    ) -> Result<()> {
        let base = pools.activation_base();
        let offset = at.checked_sub(base).ok_or(Fault::Ceiling {
            what: "bytes into the arena",
            need: at,
            have: base,
        })?;
        let backed = self.committed;
        if offset.saturating_add(into.len() as u64) > backed {
            return Err(Fault::Ceiling {
                what: "bytes into the arena's committed prefix",
                need: offset.saturating_add(into.len() as u64),
                have: backed,
            });
        }
        #[cfg(feature = "device")]
        {
            use crate::device::rt;
            // SAFETY: `at..at + len` lies inside the committed prefix, and
            // `stream` is the caller's live stream of this device.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemcpyAsync",
                    rt::cudaMemcpyAsync(
                        into.as_mut_ptr().cast(),
                        at as *const core::ffi::c_void,
                        into.len(),
                        rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                        stream.cast(),
                    ),
                )?;
            }
            crate::device::ctx::sync(stream)
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = (stream, into);
            Err(Fault::Runtimeless)
        }
    }
}

#[must_use]
/// `map`'s slots for a fire of `rows`, each value's rectangle at its
/// place. A rectangle past the bytes the map laid its slot for is refused
/// by name before anything is written: a carve wider than the layout (more
/// rows, more lanes) would land its rows in a neighbour's value.
pub fn carve(base: u64, map: &ArenaMap, rows: poem_compiler::FireRows) -> Result<SlotTable> {
    let mut cells: Vec<Option<Tensor>> = Vec::with_capacity(map.placements.len());
    for value in 0..map.placements.len() {
        let value = ValueId(value as u32);
        let Some(rect) = rect(map, value, rows) else {
            cells.push(None);
            continue;
        };
        let laid = map.placements[map.root(value).0 as usize].bytes();
        if rect.bytes > laid {
            return Err(Fault::Overcarved {
                value: value.0,
                rows: rect.rows,
                need: rect.bytes,
                laid,
            });
        }
        cells.push(Some(Tensor::new(base + rect.offset, rect.rows, rect.width, rect.dtype)));
    }
    Ok(SlotTable(cells))
}
