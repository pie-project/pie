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
/// reading them is in it) lays its planes under its carve, in the room the
/// runtime sets aside for that program's term (`engine::FireFacts::a_row`),
/// and a fire whose planes would not lie in the span is refused by name.
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
    /// The lanes every fire carves for, whatever its own: a fire that
    /// replays a recorded body lays its lane tables at the body's capacity
    /// (`window::lane_capacity`), so its carve is the smallest rung's that
    /// holds that many lanes. 0 where fires carve for their own lanes.
    lane_reach: u32,
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

/// What a fire capturing scores takes of its span for them, as `(bytes a
/// row of the fire, bytes a fire)`: a row of every `attn.scores` plane for
/// each lane the fire holds (a plane's rows are the fire's lanes, at most
/// its tokens), at the width the trace lays it, and the most the planes'
/// alignment adds. Zero where the trace captures none.
#[must_use]
pub fn score_term(trace: &poem_ir::Trace) -> (u64, u64) {
    let mut planes = 0u64;
    let row: u64 = score_values(trace)
        .into_iter()
        .filter_map(|value| match &trace.values.get(value.0 as usize)?.ty {
            poem_ir::Ty::Tensor { shape, dtype } => {
                planes += 1;
                let width: u64 = shape
                    .iter()
                    .skip(1)
                    .map(|dim| match dim {
                        poem_ir::Dim::Const(n) => *n,
                        _ => 1,
                    })
                    .product();
                Some(width * poem_compiler::arena::elem_bytes(*dtype).unwrap_or(4))
            }
            _ => None,
        })
        .sum();
    (row, planes * PLANE_ALIGN)
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

/// What a plane's offset is aligned to in a carve.
const PLANE_ALIGN: u64 = 256;

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
            lane_reach: 0,
        }
    }

    /// The classes that capture scores ([`scoring_classes`]).
    #[must_use]
    pub fn with_scoring(mut self, scoring: poem_ir::ClassSet) -> Arena {
        self.scoring = scoring;
        self
    }

    /// Whether a lane of `class` captures scores.
    #[must_use]
    pub fn captures(&self, class: usize) -> bool {
        self.scoring.contains(class)
    }

    /// Every fire of this load carves for `lanes` lanes at least: what the
    /// rungs state to the runtime follows the carve a fire really lays.
    pub fn reach_lanes(&mut self, lanes: u32) {
        self.lane_reach = lanes;
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
    /// is wider) that state may take at that rung. A carve is its bytes
    /// without its score planes; with `scored` (a program reading them is
    /// bound), with them: what lies under the planes.
    #[must_use]
    pub fn rungs(&self, forked: Option<&ArenaMap>, scored: bool) -> Vec<(u32, u64)> {
        let size = |map: &ArenaMap| {
            if scored {
                map.bytes
            } else {
                self.unscored(map)
            }
        };
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
                    .map_or(self.reserved, |(_, map)| size(map));
                let bytes = carve.max(forked.map_or(0, size));
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
        scored: bool,
    ) -> (u32, u64) {
        self.rungs(forked, scored)
            .into_iter()
            .find(|&(rung, _)| u64::from(rung) >= tokens && u64::from(rung) >= lanes)
            .unwrap_or((top, 0))
    }

    /// Every rung's tokens and the bytes under its carve, the whole arena's
    /// (`top` tokens, nothing under it) last.
    #[must_use]
    pub fn every_rung(&self, forked: Option<&ArenaMap>, top: u32, scored: bool) -> Vec<(u32, u64)> {
        let mut rungs = self.rungs(forked, scored);
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
        scored: bool,
    ) -> Result<SlotTable> {
        self.fire_slots_at(pools.activation_base(), map, forked, rows, scored)
    }

    /// A fire's slots over a span starting at `base`: `map` (the arena in
    /// order, or the forked one) as laid at the span's start, or, where
    /// this load states rungs, the smallest rung's carve that holds `rows`
    /// (a fire that forks keeps its own) at the reservation's end. The
    /// carve's score planes lie under the rest of it, inside the span where
    /// it has room for them; else they get no slot, and a `scored` fire is
    /// refused. The place each carve takes is fixed, so a body recorded at
    /// any rung stays good at every rung its fire fits.
    pub fn fire_slots_at(
        &self,
        base: u64,
        map: &ArenaMap,
        forked: bool,
        rows: poem_compiler::FireRows,
        scored: bool,
    ) -> Result<SlotTable> {
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
        // The carve's start and, where they lie in the span, its planes'.
        let (at, planes) = if self.rungs.is_empty() {
            (0, (chosen.bytes <= self.reserved).then_some(unscored))
        } else {
            let at = self.reserved.saturating_sub(unscored);
            (at, self.reserved.checked_sub(chosen.bytes))
        };
        if scored && planes.is_none() {
            return Err(Fault::Ceiling {
                what: "bytes of the activation's span a fire capturing attn.scores takes, its \
                       carve and the score planes under it",
                need: chosen.bytes,
                have: self.reserved,
            });
        }
        let roots: Vec<ValueId> = self
            .scores
            .iter()
            .map(|&value| chosen.root(value))
            .collect();
        Ok(carve_split(
            base + at,
            planes.map(|planes| base + planes),
            unscored,
            &roots,
            chosen,
            rows,
        ))
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
    /// reading them states (`a_row` a row of the rung's tokens, `slack`).
    #[test]
    fn a_scored_carve_lies_in_the_span_its_unscored_top_backs() {
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
        let (a_row, slack) = score_term(&trace);
        assert!(a_row > 0);
        let scores = score_values(&trace);
        let mut fitted = 0;
        for (&(tokens, ref map), (stated, under)) in rungs.iter().zip(arena.rungs(None, false)) {
            assert_eq!(stated, tokens);
            let unscored = arena.unscored(map);
            assert_eq!(unscored, map.scores_from(&trace));
            assert_eq!(
                under,
                span - unscored,
                "{tokens} tokens: what state may take"
            );
            let planes = map.bytes - unscored;
            assert!(
                planes <= u64::from(tokens) * a_row + slack,
                "{tokens} tokens: {planes} bytes of planes past the term {a_row} a row, {slack}"
            );
            let rows =
                poem_compiler::FireRows::text_only(u64::from(tokens), u64::from(tokens.min(4)));
            let roots: Vec<ValueId> = scores.iter().map(|&value| map.root(value)).collect();
            let plane = |value: usize| roots.contains(&map.root(ValueId(value as u32)));
            let plain = arena
                .fire_slots_at(0, &compiled.arena, false, rows, false)
                .expect("an unscored fire carves at every rung");
            for (value, handle) in plain.0.iter().enumerate() {
                let Some(handle) = handle.filter(|_| !plane(value)) else {
                    continue;
                };
                assert!(handle.ptr >= under && end(&handle) <= span);
            }
            let scored = arena.fire_slots_at(0, &compiled.arena, false, rows, true);
            if map.bytes > span {
                let refused = scored.expect_err("planes past the span are refused");
                assert!(refused.to_string().contains("attn.scores"), "{refused}");
                continue;
            }
            fitted += 1;
            let scored = scored.expect("planes inside the span");
            for (value, handle) in scored.0.iter().enumerate() {
                let Some(handle) = handle else { continue };
                assert!(end(handle) <= span, "a slot ending past the span");
                if plane(value) {
                    assert!(
                        handle.ptr >= span - map.bytes && end(handle) <= span - unscored,
                        "{tokens} tokens: a score plane outside the room under the carve"
                    );
                } else {
                    assert!(handle.ptr >= span - unscored);
                }
            }
            // With a reader bound, state is stated under the planes.
            let (_, scored_under) = arena.rung_of(None, u64::from(tokens), 1, 64, true);
            assert_eq!(scored_under, span - map.bytes);
        }
        assert!(fitted > 0, "no rung's planes fit the span");
        // The whole arena's planes lie past the span: its fires are cut.
        let top = poem_compiler::FireRows::text_only(64, 4);
        assert!(
            arena
                .fire_slots_at(0, &compiled.arena, false, top, true)
                .is_err()
        );
    }
}
