#![allow(unsafe_code)]
use kernels_cuda::Tensor;
use poem_compiler::ArenaMap;
use poem_exec::store::arena::rect;
use poem_ir::ValueId;

use crate::error::{Fault, Result};
use crate::run::SlotTable;
use crate::store::Pools;

/// A fire's activation: the span under the state pool's end
/// ([`Pools::activation_base`]), backed whole at load ([`Arena::back`]).
///
/// The span is the largest carve. A capturing lane's score rows take none
/// of it (`poem_compiler::Placement::Capture`): they are its reading
/// program's instance's state.
#[derive(Debug)]
pub struct Arena {
    committed: u64,
    reserved: u64,
    /// The activation rungs (SPEC decision 6): for fires of at most so many
    /// tokens (and lanes), the carve laid out for them, smallest first.
    /// Each lies at the reservation's end, so a rung's fires touch only the
    /// top of the span and state may take what lies under it.
    rungs: Vec<(u32, ArenaMap)>,
    /// The classes that capture scores: a lane of one has its score rows
    /// in its reading program's instance.
    scoring: poem_ir::ClassSet,
}

/// The classes whose nodes capture scores.
#[must_use]
pub fn scoring_classes(
    trace: &poem_ir::Trace,
    compiled: &poem_compiler::CompiledModel,
) -> poem_ir::ClassSet {
    let mut scoring = poem_ir::ClassSet::default();
    for (node, at) in trace.nodes.iter().zip(&compiled.classes.node_mask) {
        if matches!(
            node.op,
            poem_ir::Operation::Attn(poem_ir::Attn::ScoreCapture { .. })
        ) {
            for class in at.iter() {
                scoring.insert(class);
            }
        }
    }
    scoring
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
            scoring: poem_ir::ClassSet::default(),
        }
    }

    /// The classes that capture scores ([`scoring_classes`]).
    #[must_use]
    pub fn with_scoring(mut self, scoring: poem_ir::ClassSet) -> Arena {
        self.scoring = scoring;
        self
    }

    /// The rung whose carve a fire of `tokens` tokens carving for `lanes`
    /// lanes lays ([`Arena::fire_slots`]); `None`: the whole arena's.
    #[must_use]
    pub fn carve_rung(&self, tokens: u64, lanes: u64) -> Option<u32> {
        self.rungs
            .iter()
            .map(|&(rung, _)| rung)
            .find(|&rung| u64::from(rung) >= tokens && u64::from(rung) >= lanes)
    }

    /// Whether a lane of `class` captures scores.
    #[must_use]
    pub fn captures(&self, class: usize) -> bool {
        self.scoring.contains(class)
    }

    /// With `rungs`, each a carve no larger than the reservation, smallest
    /// first: a fire carves the smallest that
    /// holds it, at the reservation's end ([`Arena::fire_slots`]).
    #[must_use]
    pub fn with_rungs(mut self, mut rungs: Vec<(u32, ArenaMap)>) -> Arena {
        rungs.retain(|(_, map)| map.bytes <= self.reserved);
        rungs.sort_by_key(|&(tokens, _)| tokens);
        self.rungs = rungs;
        self
    }

    /// Each rung's tokens and its carve's bytes: what the span weighs a
    /// rung by.
    #[must_use]
    pub fn carves(&self) -> Vec<(u32, u64)> {
        self.rungs
            .iter()
            .map(|(tokens, map)| (*tokens, map.bytes))
            .collect()
    }

    /// Whether this load lays carves at the reservation's end.
    #[must_use]
    pub fn has_rungs(&self) -> bool {
        !self.rungs.is_empty()
    }

    /// Each rung's tokens and the bytes of the span under what a fire of
    /// it takes at the top (its carve, or `forked`'s where a fire that forks
    /// is wider) that state may take at that rung.
    #[must_use]
    pub fn rungs(&self, forked: Option<&ArenaMap>) -> Vec<(u32, u64)> {
        let size = |map: &ArenaMap| map.bytes;
        self.rungs
            .iter()
            .map(|(tokens, map)| {
                // The carve a fire cut to this rung lays: its own. A fire
                // replaying a body carves for the body's lanes, which are
                // no more than its bucket's rows (`window::body_lanes`),
                // so no wider rung's ([`Arena::fire_slots`]).
                let bytes = size(map).max(forked.map_or(0, size));
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

    /// No smaller rung: every fire's arena at the span's start.
    pub fn drop_rungs(&mut self) {
        self.rungs.clear();
    }

    /// The reservation's bytes: the most any carve this load lays asks for,
    /// the span stated to the runtime.
    #[must_use]
    pub fn reserved(&self) -> u64 {
        self.reserved
    }

    /// Backs the whole reservation, at load: every chunk of the span the
    /// runtime may open state in under a rung's carve is then backed, and
    /// nothing backs the arena after.
    pub fn back(&mut self, pools: &mut Pools) -> Result<()> {
        if self.committed >= self.reserved {
            return Ok(());
        }
        pools.hold_activation(self.reserved, false)?;
        self.committed = self.reserved;
        Ok(())
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.committed
    }

    #[must_use]
    pub fn slots(&self, pools: &Pools, map: &ArenaMap, rows: poem_compiler::FireRows) -> SlotTable {
        carve(pools.activation_base(), map, rows)
    }

    /// A fire's slots over the span ([`Arena::fire_slots_at`] its start).
    pub fn fire_slots(
        &self,
        pools: &Pools,
        map: &ArenaMap,
        forked: bool,
        rows: poem_compiler::FireRows,
    ) -> SlotTable {
        self.fire_slots_at(pools.activation_base(), map, forked, rows)
    }

    /// A fire's slots over a span starting at `base`: `map` (the arena in
    /// order, or the forked one) as laid at the span's start, or, where
    /// this load states rungs, the smallest rung's carve that holds `rows`
    /// (a fire that forks keeps its own) at the reservation's end. The
    /// place each carve takes is fixed, so a body recorded at any rung stays
    /// good at every rung its fire fits.
    pub fn fire_slots_at(
        &self,
        base: u64,
        map: &ArenaMap,
        forked: bool,
        rows: poem_compiler::FireRows,
    ) -> SlotTable {
        let chosen = if forked || self.rungs.is_empty() {
            map
        } else {
            self.rungs
                .iter()
                .find(|(tokens, _)| {
                    u64::from(*tokens) >= rows.tokens && u64::from(*tokens) >= rows.lanes
                })
                .map_or(map, |(_, rung)| rung)
        };
        let at = if self.rungs.is_empty() {
            0
        } else {
            self.reserved.saturating_sub(chosen.bytes)
        };
        carve(base + at, chosen, rows)
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
        // The span is backed whole: every carve lies
        // inside it.
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

/// `map`'s slots from `base`; a value with no arena bytes (a capturing
/// lane's score row, `Placement::Capture`) has none.
#[must_use]
pub fn carve(base: u64, map: &ArenaMap, rows: poem_compiler::FireRows) -> SlotTable {
    let cells = (0..map.placements.len())
        .map(|value| {
            rect(map, ValueId(value as u32), rows).map(|rect| {
                Tensor::new(
                    base.wrapping_add(rect.offset),
                    rect.rows,
                    rect.width,
                    rect.dtype,
                )
            })
        })
        .collect();
    SlotTable(cells)
}

#[cfg(test)]
mod tests {
    use poem::Platform;
    use poem_compiler::{Budget, DeviceProfile, compile};

    use super::*;

    const DEPLOYMENT: &str = "qwen35-d0.8b-bf16-kv-bf16";

    fn compiled() -> (poem_ir::Trace, poem_compiler::CompiledModel) {
        let trace = poem_compiler::catalog::deployment(DEPLOYMENT)
            .expect("the catalog ships the smoke's deployment")
            .trace(Platform::Cuda);
        let compiled = compile(&trace, &Budget::new(4, 64), &DeviceProfile::default())
            .expect("the smoke's deployment bakes");
        (trace, compiled)
    }

    #[test]
    fn the_carve_fits_the_allocation_it_asks_for() {
        let (_, compiled) = compiled();
        let slots = carve(
            0,
            &compiled.arena,
            poem_compiler::FireRows::text_only(64, 4),
        );
        for handle in slots.0.iter().flatten() {
            let end = handle.ptr
                + handle.elements() * poem_compiler::arena::elem_bytes(handle.dtype).unwrap_or(0);
            assert!(
                end <= compiled.arena.bytes,
                "a rectangle ending at {end} in an arena of {} bytes",
                compiled.arena.bytes
            );
        }
    }
}
