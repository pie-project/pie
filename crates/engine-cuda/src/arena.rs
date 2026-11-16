#![allow(unsafe_code)]
use kernels_cuda::Tensor;
use poem_compiler::{ArenaMap, Placement};
use poem_exec::store::arena::rect;
use poem_ir::ValueId;

use crate::error::{Fault, Result};
use crate::run::SlotTable;
use crate::store::Pools;

/// A fire's activation: the span under the state pool's end
/// ([`Pools::activation_base`]), backed whole at load ([`Arena::back`]).
///
/// The span is the largest carve without its `attn.scores` planes: no load
/// reserves or backs score planes. A fire that captures scores (a program
/// reading them is in it) lays its planes in the state of a pair whose
/// program reads them (its `engine::GuestState::per_pair`, sized by
/// [`Arena::widest_planes`]), never in the span.
#[derive(Debug)]
pub struct Arena {
    committed: u64,
    reserved: u64,
    /// The activation rungs (SPEC decision 6): for fires of at most so many
    /// tokens (and lanes), the carve laid out for them, smallest first.
    /// Each lies at the reservation's end, so a rung's fires touch only the
    /// top of the span and state may take what lies under it.
    rungs: Vec<(u32, ArenaMap)>,
    /// The trace's `attn.scores` outputs: in every carve, the planes on top
    /// of every other slot (`ArenaMap::scores_from`).
    scores: Vec<ValueId>,
    /// The classes that capture scores: a fire holding a lane of one lays
    /// its carve's score planes ([`Arena::fire_slots`]).
    scoring: poem_ir::ClassSet,
}

/// The `attn.scores` outputs of `trace`.
#[must_use]
pub fn score_values(trace: &poem_ir::Trace) -> Vec<ValueId> {
    trace
        .outputs
        .iter()
        .filter(|output| output.name == SCORES)
        .map(|output| output.value)
        .collect()
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

const SCORES: &str = "attn.scores";

impl Arena {
    /// `bytes`, the most any arena map this load carves asks for without
    /// its score planes: what the pool was reserved its activation chunks
    /// for. `scores`: the trace's `attn.scores` outputs ([`score_values`]).
    #[must_use]
    pub fn reserve(bytes: u64, scores: Vec<ValueId>) -> Arena {
        Arena {
            committed: 0,
            reserved: bytes,
            rungs: Vec::new(),
            scores,
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

    /// The bytes of `map`'s `attn.scores` planes, laid aligned for its
    /// lanes.
    #[must_use]
    pub fn planes(&self, map: &ArenaMap) -> u64 {
        map.bytes - self.unscored(map)
    }

    /// The planes a fire of `tokens` tokens carving for `lanes` lanes lays
    /// under its carve ([`Arena::fire_slots`]), `forked`'s where wider.
    #[must_use]
    pub fn planes_at(&self, forked: Option<&ArenaMap>, tokens: u64, lanes: u64) -> u64 {
        let rung = self
            .rungs
            .iter()
            .find(|(rung, _)| u64::from(*rung) >= tokens && u64::from(*rung) >= lanes)
            .map_or(0, |(_, map)| self.planes(map));
        rung.max(forked.map_or(0, |forked| self.planes(forked)))
    }

    /// The most any of `maps`' carves (the in-order and forked arenas,
    /// widest of all) lays for its planes.
    #[must_use]
    pub fn widest_planes<'a>(&self, maps: impl IntoIterator<Item = &'a ArenaMap>) -> u64 {
        maps.into_iter()
            .map(|map| self.planes(map))
            .max()
            .unwrap_or(0)
    }

    /// Whether a lane of `class` captures scores.
    #[must_use]
    pub fn captures(&self, class: usize) -> bool {
        self.scoring.contains(class)
    }

    /// With `rungs`, each a carve no larger than the reservation without
    /// its score planes, smallest first: a fire carves the smallest that
    /// holds it, at the reservation's end ([`Arena::fire_slots`]).
    #[must_use]
    pub fn with_rungs(mut self, mut rungs: Vec<(u32, ArenaMap)>) -> Arena {
        rungs.retain(|(_, map)| self.unscored(map) <= self.reserved);
        rungs.sort_by_key(|&(tokens, _)| tokens);
        self.rungs = rungs;
        self
    }

    /// Each rung's tokens and its carve's bytes without its score planes:
    /// what the span weighs a rung by.
    #[must_use]
    pub fn carves(&self) -> Vec<(u32, u64)> {
        self.rungs
            .iter()
            .map(|(tokens, map)| (*tokens, self.unscored(map)))
            .collect()
    }

    /// Whether this load lays carves at the reservation's end.
    #[must_use]
    pub fn has_rungs(&self) -> bool {
        !self.rungs.is_empty()
    }

    /// The bytes of `map` under its `attn.scores` planes.
    #[must_use]
    pub fn unscored(&self, map: &ArenaMap) -> u64 {
        self.scores
            .iter()
            .filter_map(
                |&value| match map.placements.get(map.root(value).0 as usize) {
                    Some(Placement::Arena { offset, .. }) => Some(*offset),
                    _ => None,
                },
            )
            .min()
            .unwrap_or(map.bytes)
    }

    /// Each rung's tokens and the bytes of the span under what a fire of
    /// it takes at the top (its carve, or `forked`'s where a fire that forks
    /// is wider) that state may take at that rung, a carve its bytes
    /// without its score planes: a fire's planes, under it, lie in the
    /// term the runtime sets aside below.
    #[must_use]
    pub fn rungs(&self, forked: Option<&ArenaMap>) -> Vec<(u32, u64)> {
        let size = |map: &ArenaMap| self.unscored(map);
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

    /// The reservation's bytes: the most any carve this load lays asks for
    /// without its score planes, the span stated to the runtime.
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
        planes: Option<u64>,
    ) -> SlotTable {
        self.fire_slots_at(pools.activation_base(), map, forked, rows, planes)
    }

    /// A fire's slots over a span starting at `base`: `map` (the arena in
    /// order, or the forked one) as laid at the span's start, or, where
    /// this load states rungs, the smallest rung's carve that holds `rows`
    /// (a fire that forks keeps its own) at the reservation's end. A fire
    /// capturing scores lays its carve's score planes at `planes`, in a
    /// reader's pair's state, which holds the widest carve's; one that
    /// captures none gives them no slot. The place each carve takes is
    /// fixed, so a body recorded at any rung stays good at every rung its
    /// fire fits.
    pub fn fire_slots_at(
        &self,
        base: u64,
        map: &ArenaMap,
        forked: bool,
        rows: poem_compiler::FireRows,
        planes: Option<u64>,
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
        let unscored = self.unscored(chosen);
        let at = if self.rungs.is_empty() {
            0
        } else {
            self.reserved.saturating_sub(unscored)
        };
        let roots: Vec<ValueId> = self
            .scores
            .iter()
            .map(|&value| chosen.root(value))
            .collect();
        carve_split(base + at, planes, unscored, &roots, chosen, rows)
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
        // The span is backed whole: every carve, its score planes too, lies
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

#[must_use]
pub fn carve(base: u64, map: &ArenaMap, rows: poem_compiler::FireRows) -> SlotTable {
    carve_split(base, Some(base), 0, &[], map, rows)
}

/// `map`'s slots: the score planes (rooted in `scores`, every slot from
/// `unscored` on) from `planes`, none where it is `None`; the rest from
/// `base`.
fn carve_split(
    base: u64,
    planes: Option<u64>,
    unscored: u64,
    scores: &[ValueId],
    map: &ArenaMap,
    rows: poem_compiler::FireRows,
) -> SlotTable {
    let mut cells: Vec<Option<Tensor>> = Vec::with_capacity(map.placements.len());
    for value in 0..map.placements.len() {
        let value = ValueId(value as u32);
        let from = if scores.contains(&map.root(value)) {
            planes.map(|planes| planes.wrapping_sub(unscored))
        } else {
            Some(base)
        };
        cells.push(from.and_then(|from| {
            rect(map, value, rows).map(|rect| {
                Tensor::new(
                    from.wrapping_add(rect.offset),
                    rect.rows,
                    rect.width,
                    rect.dtype,
                )
            })
        }));
    }
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

    fn end(handle: &Tensor) -> u64 {
        handle.ptr + handle.elements() * poem_compiler::arena::elem_bytes(handle.dtype).unwrap_or(0)
    }

    /// The span is the widest carve without its score planes, backed whole
    /// and stated whole: under every rung's carve, state may take what the
    /// span holds below it. A fire capturing scores lays its rung's planes
    /// under that carve, inside the span, where it has room; one past it is
    /// refused by name; and the planes are no more than the term a program
    /// reading them states (the widest carve's planes).
    #[test]
    fn a_scored_carve_lays_its_planes_where_a_readers_pair_holds_them() {
        let (trace, compiled) = compiled();
        let budgets = poem_compiler::Budgets::of(Budget::new(4, 64));
        let span = compiled.arena.scores_from(&trace);
        assert!(
            span < compiled.arena.bytes,
            "the smoke's deployment captures scores"
        );
        let rungs: Vec<(u32, ArenaMap)> = [1u32, 2, 4, 8, 16, 32]
            .into_iter()
            .map(|tokens| {
                let map = poem_compiler::arena_rung(&trace, &budgets, &compiled, tokens)
                    .expect("a rung's carve");
                (tokens, map)
            })
            .collect();
        let mut arena = Arena::reserve(span, score_values(&trace)).with_rungs(rungs.clone());
        // What `Arena::back` leaves it at.
        arena.committed = arena.reserved;
        let widest = arena.widest_planes([&compiled.arena]);
        assert!(widest > 0);
        let scores = score_values(&trace);
        let mut fitted = 0;
        for (&(tokens, ref map), (stated, under)) in rungs.iter().zip(arena.rungs(None)) {
            assert_eq!(stated, tokens);
            let unscored = arena.unscored(map);
            assert_eq!(unscored, map.scores_from(&trace));
            assert_eq!(
                under,
                span - unscored,
                "{tokens} tokens: what state may take"
            );
            let planes = map.bytes - unscored;
            assert_eq!(planes, arena.planes(map));
            assert!(
                planes <= widest,
                "{tokens} tokens: {planes} bytes of planes past the term's {widest}"
            );
            let rows =
                poem_compiler::FireRows::text_only(u64::from(tokens), u64::from(tokens.min(4)));
            let roots: Vec<ValueId> = scores.iter().map(|&value| map.root(value)).collect();
            let plane = |value: usize| roots.contains(&map.root(ValueId(value as u32)));
            let plain = arena.fire_slots_at(0, &compiled.arena, false, rows, None);
            for (value, handle) in plain.0.iter().enumerate() {
                if plane(value) {
                    assert!(handle.is_none(), "an unscored fire gives a plane no slot");
                    continue;
                }
                let Some(handle) = handle else { continue };
                assert!(handle.ptr >= under && end(handle) <= span);
            }
            // The planes in a pair's state, far from the span.
            let held = 1u64 << 40;
            let scored = arena.fire_slots_at(0, &compiled.arena, false, rows, Some(held));
            fitted += 1;
            for (value, handle) in scored.0.iter().enumerate() {
                let Some(handle) = handle else { continue };
                if plane(value) {
                    assert!(
                        handle.ptr >= held && end(handle) <= held + widest,
                        "{tokens} tokens: a score plane outside the pair's room"
                    );
                } else {
                    assert!(handle.ptr >= span - unscored && end(handle) <= span);
                }
            }
            assert_eq!(arena.planes_at(None, u64::from(tokens), 1), planes);
        }
        assert!(fitted > 0, "no rung was walked");
    }
}
