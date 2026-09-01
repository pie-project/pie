//! **The prefetch schedule** (alto streaming §2 "Dense (static)", build-order
//! item 4): which params a fire reads, in what order, and how a fixed set of
//! device slots can serve them all.
//!
//! # The demand shape this module is for
//!
//! Streaming §7 splits weight demand in two. The ROUTED shape is dynamic —
//! routing is computed on device, so no host decision can precede a fire and
//! say which experts it needs — and its answer is an indirection table and a
//! popularity vote (`engine_cuda::experts`). The DENSE shape is the opposite
//! in every respect: **the compiler knows which regions read which params, and
//! the answer never changes.** Every fire reads the same planes in the same
//! order, so the schedule is fire-invariant and can be computed once, here,
//! off the plan alone.
//!
//! That invariance is the whole asset. It means a copy can be issued AHEAD of
//! the read that wants it rather than in response to it, which is what makes
//! a spilled dense plane a bandwidth cost instead of a stall.
//!
//! # Why it is DERIVED and not a field of `CompiledModel`
//!
//! Because it is derivable, and because the artifact is hashed. Alto's G4
//! invariant is that every pre-campaign SKU compiles to a BIT-IDENTICAL
//! artifact; a new serialized field changes every artifact in the catalog to
//! carry a table that is a pure function of two things the artifact already
//! holds. So this is a function over `Trace` (and, for the region-granular
//! projection, `CompiledModel`), and storing it would buy load-time work and
//! no information at all. If it is ever stored, that is a cache and should be
//! argued as one.
//!
//! # Two granularities, and why the coarse one is off the trace alone
//!
//! [`Schedule::of`] reads NODE indices, because a residency plan is decided
//! before the model is compiled — `experts::Plan::of` runs against the trace
//! and the two budgets, with no `CompiledModel` in hand — and the ORDER is the
//! same either way: a region is a maximal run of ADJACENT nodes, so ordering
//! params by first-reading node and ordering them by first-reading region give
//! the same sequence. [`Schedule::against`] projects onto regions for the
//! consumer that pumps copies at region boundaries.

use std::collections::BTreeMap;
use std::ops::Range;

use model_ir::{Def, Operands, Trace, ValueId};

use crate::compiled::CompiledModel;

/// **Where in the fire one param is read.**
///
/// Half-open in nodes, inclusive in count: `reads` is how many distinct nodes
/// name it, which is what says whether a plane is a norm scale touched once or
/// an embedding touched at both ends of the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reads {
    /// Index into `Trace::params`.
    pub param: usize,
    /// The first node that reads it, and the last, half-open — so
    /// `span.start` is when its bytes must have arrived and `span.end` is when
    /// its slot may be reused.
    pub span: Range<u32>,
    /// How many nodes read it.
    pub reads: u32,
}

impl Reads {
    /// Is this plane read by nothing?
    ///
    /// A real answer and not a degenerate one: a plan may declare a param no
    /// region names — a registered adapter bank nothing corrects through, an
    /// arm the bake dropped — and a schedule that pretended it was read at
    /// node zero would pin it first.
    #[must_use]
    pub const fn unread(&self) -> bool {
        self.reads == 0
    }
}

/// **THE FIRE-INVARIANT PREFETCH SCHEDULE**: every param, in the order the
/// fire reaches it.
///
/// Ascending by first read and then by param index, so two compiles of one
/// plan produce the same sequence and a slot assignment derived from it is a
/// compile-time constant.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Schedule {
    reads: Vec<Reads>,
    of: BTreeMap<usize, usize>,
    nodes: u32,
}

impl Schedule {
    /// **The schedule for `trace`.**
    ///
    /// One pass over the nodes, resolving each input that is a `Def::Weight`
    /// to its `Trace::params` row. Params nothing reads are carried with
    /// `reads == 0` and sort LAST, which is where a residency plan wants them:
    /// a plane no fire touches is the first thing a budget should give up.
    #[must_use]
    pub fn of(trace: &Trace) -> Schedule {
        let mut span: Vec<Option<Range<u32>>> = vec![None; trace.params.len()];
        let mut reads: Vec<u32> = vec![0; trace.params.len()];
        let mut inputs: Vec<ValueId> = Vec::new();
        for (at, node) in trace.nodes.iter().enumerate() {
            let at = u32::try_from(at).unwrap_or(u32::MAX);
            inputs.clear();
            node.op.inputs(&mut inputs);
            for id in &inputs {
                let Some(param) = weight_of(trace, *id) else {
                    continue;
                };
                reads[param] += 1;
                span[param] = Some(match span[param].clone() {
                    None => at..at + 1,
                    Some(had) => had.start.min(at)..had.end.max(at + 1),
                });
            }
        }
        let mut rows: Vec<Reads> = (0..trace.params.len())
            .map(|param| Reads {
                param,
                span: span[param].clone().unwrap_or(0..0),
                reads: reads[param],
            })
            .collect();
        // Unread planes last, then by first read, then by param — a total
        // order with no ties left to the sort's stability.
        rows.sort_by_key(|row| (row.unread(), row.span.start, row.param));
        let of = rows
            .iter()
            .enumerate()
            .map(|(at, row)| (row.param, at))
            .collect();
        Schedule {
            reads: rows,
            of,
            nodes: u32::try_from(trace.nodes.len()).unwrap_or(u32::MAX),
        }
    }

    /// Every param, in schedule order.
    #[must_use]
    pub fn reads(&self) -> &[Reads] {
        &self.reads
    }

    /// Where one param is read.
    #[must_use]
    pub fn read_of(&self, param: usize) -> Option<Reads> {
        self.of.get(&param).map(|at| self.reads[*at].clone())
    }

    /// **The order a prefetch issues copies in**: earliest-read first.
    #[must_use]
    pub fn order(&self) -> Vec<usize> {
        self.reads.iter().map(|row| row.param).collect()
    }

    /// **The order a BUDGET gives planes up in**: the reverse.
    ///
    /// A plane nothing reads goes first, then the plane read latest, and the
    /// embedding — read at node zero by every fire — goes last. That is the
    /// same statement the prefetch makes, read from the other end: the plane
    /// with the most fire between now and its read is the one a copy has the
    /// most time to deliver, and the one a UVA read has the most warps to hide
    /// behind.
    #[must_use]
    pub fn spill_order(&self) -> Vec<usize> {
        let mut out = self.order();
        out.reverse();
        out
    }

    /// **Project onto regions** — the granularity a pump works at.
    ///
    /// A region is a maximal run of adjacent nodes, so this is a lookup and
    /// not a second scan, and the ORDER is unchanged by it (which is the
    /// property that lets a residency plan use the node-granular schedule
    /// before the model is compiled).
    #[must_use]
    pub fn against(&self, compiled: &CompiledModel) -> Vec<Range<u32>> {
        self.reads
            .iter()
            .map(|row| {
                if row.unread() {
                    return 0..0;
                }
                let first = region_of(compiled, row.span.start);
                let last = region_of(compiled, row.span.end.saturating_sub(1));
                first..last + 1
            })
            .collect()
    }

    /// How many nodes the plan has — the horizon a span is read against.
    #[must_use]
    pub const fn nodes(&self) -> u32 {
        self.nodes
    }

    /// **THE SLOT ASSIGNMENT, AS A COMPILE-TIME CONSTANT** (streaming §2:
    /// *"layer→slot assignment is a compile-time constant, so captured graphs
    /// read fixed slot addresses — contents rotate, addresses never"*).
    ///
    /// `spilled` is the params a budget did not seat, in any order; they are
    /// walked in SCHEDULE order and dealt round-robin into `slots` slots. The
    /// assignment is a function of the plan and the slot count alone, so two
    /// boots of one deployment place the same plane in the same slot and a
    /// graph recorded against slot `k`'s address stays correct forever.
    ///
    /// **AND IT IS PROVED, NOT ASSUMED.** Dealing round-robin is only correct
    /// if a slot's next tenant does not arrive before its current one is
    /// finished being read: plane `i` and plane `i + slots` share a slot, so
    /// the schedule must satisfy `last_read(i) <= first_read(i + slots)`. That
    /// is exactly `experts.rs`' uniformity proof restated for a static demand
    /// shape, and like it, it is CHECKED off the plan rather than assumed —
    /// the day a plan arrives whose live ranges overlap, the arithmetic below
    /// would silently read a plane that had already been overwritten.
    ///
    /// # Errors
    ///
    /// [`Overlap`] naming both planes and both spans, for a slot count too
    /// small to keep every tenant alive to its last read. The fix is more
    /// slots, and the error carries the smallest count that would work.
    pub fn slotting(&self, spilled: &[usize], slots: u32) -> Result<Slotting, Overlap> {
        if slots == 0 {
            return Err(Overlap {
                slots,
                want: 1,
                evicted: 0,
                by: 0,
                live: 0..0,
                arrives: 0,
            });
        }
        let mut queue: Vec<Reads> = spilled
            .iter()
            .filter_map(|param| self.read_of(*param))
            .collect();
        queue.sort_by_key(|row| (row.unread(), row.span.start, row.param));

        let mut of: BTreeMap<usize, u32> = BTreeMap::new();
        for (at, row) in queue.iter().enumerate() {
            of.insert(row.param, (at as u32) % slots);
        }
        // THE PROOF. Walk each slot's tenants in order and require each one's
        // last read to fall at or before the next one's first.
        for pair in queue.windows(slots as usize + 1) {
            let (evicted, by) = (&pair[0], &pair[slots as usize]);
            if evicted.unread() {
                continue;
            }
            if evicted.span.end > by.span.start {
                // The smallest slot count that keeps this pair apart is the
                // number of planes live across the conflict, which is how far
                // apart they have to be dealt.
                let want = queue
                    .iter()
                    .filter(|row| !row.unread() && row.span.start < evicted.span.end)
                    .count();
                return Err(Overlap {
                    slots,
                    want: u32::try_from(want).unwrap_or(u32::MAX),
                    evicted: evicted.param,
                    by: by.param,
                    live: evicted.span.clone(),
                    arrives: by.span.start,
                });
            }
        }
        Ok(Slotting {
            of,
            slots,
            order: queue.into_iter().map(|row| row.param).collect(),
        })
    }
}

/// **A proved slot assignment**: which slot holds which plane, and in what
/// order the pump fills them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slotting {
    of: BTreeMap<usize, u32>,
    slots: u32,
    order: Vec<usize>,
}

impl Slotting {
    /// Which slot holds `param`, or `None` for a plane this slotting does not
    /// cover.
    #[must_use]
    pub fn slot_of(&self, param: usize) -> Option<u32> {
        self.of.get(&param).copied()
    }

    /// How many slots.
    #[must_use]
    pub const fn slots(&self) -> u32 {
        self.slots
    }

    /// The planes, in the order the pump fills them — schedule order.
    #[must_use]
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// **The slot's own size**: the largest plane it ever holds.
    ///
    /// A slot is one rectangle for its whole life, so it is sized for its
    /// biggest tenant and the smaller ones use a prefix. `plane` answers a
    /// param's bytes; the caller owns that arithmetic because the compiler
    /// does not know a backend's alignment.
    #[must_use]
    pub fn slot_bytes(&self, slot: u32, plane: impl Fn(usize) -> u64) -> u64 {
        self.of
            .iter()
            .filter(|(_, which)| **which == slot)
            .map(|(param, _)| plane(*param))
            .max()
            .unwrap_or(0)
    }
}

/// **Why a slot count cannot serve a schedule**: a plane would be overwritten
/// before its last read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Overlap {
    /// The slot count that was asked for.
    pub slots: u32,
    /// The smallest count that would have worked.
    pub want: u32,
    /// The plane that would be overwritten too early.
    pub evicted: usize,
    /// The plane that would overwrite it.
    pub by: usize,
    /// The evicted plane's live range, in nodes.
    pub live: Range<u32>,
    /// The node at which its successor is first read.
    pub arrives: u32,
}

impl std::fmt::Display for Overlap {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "{} slots cannot serve this schedule: param {} is read through node {} \
             and param {}, which shares its slot, is first read at node {}. A slot's \
             tenant must be finished before its successor arrives; {} slots would \
             hold this plan",
            self.slots, self.evicted, self.live.end, self.by, self.arrives, self.want,
        )
    }
}

impl std::error::Error for Overlap {}

/// The `Trace::params` row a value id names, or `None` for a value that is not
/// a weight.
fn weight_of(trace: &Trace, id: ValueId) -> Option<usize> {
    match trace.values.get(id.0 as usize).map(|decl| &decl.def) {
        Some(Def::Weight(w)) => Some(*w as usize),
        _ => None,
    }
}

/// Which region covers `node`. Regions are a partition of the node range in
/// program order, so this is a binary search.
fn region_of(compiled: &CompiledModel, node: u32) -> u32 {
    let found = compiled
        .regions
        .binary_search_by(|region| {
            if region.nodes.end <= node {
                std::cmp::Ordering::Less
            } else if region.nodes.start > node {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .unwrap_or(0);
    u32::try_from(found).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use model_dsl::Platform;

    use super::*;

    fn d0_8b() -> Trace {
        let trace = models::trace_of("qwen35-d0.8b-bf16-kv-bf16").expect("the catalog ships it");
        trace(Platform::Cuda)
    }

    #[test]
    fn the_schedule_is_a_function_of_the_plan_and_nothing_else() {
        // FIRE-INVARIANCE, as an assertion rather than a claim. Streaming §2's
        // whole premise for the static demand shape is that the same layers
        // are read in the same order every step; a schedule that varied with
        // anything at all would be a prefetch that could arrive late for a
        // reason nobody could name.
        let one = Schedule::of(&d0_8b());
        let two = Schedule::of(&d0_8b());
        assert_eq!(one, two, "two traces of one SKU schedule identically");
        assert!(!one.reads().is_empty());
        assert_eq!(one.reads().len(), d0_8b().params.len());
    }

    #[test]
    fn every_read_plane_has_a_span_and_they_ascend() {
        let trace = d0_8b();
        let schedule = Schedule::of(&trace);
        let mut last = 0u32;
        let mut seen = 0usize;
        for row in schedule.reads() {
            if row.unread() {
                continue;
            }
            seen += 1;
            assert!(row.span.start < row.span.end, "param {} is read at nothing", row.param);
            assert!(row.span.end <= schedule.nodes());
            assert!(
                row.span.start >= last,
                "the schedule is not ascending at param {}",
                row.param
            );
            last = row.span.start;
        }
        assert!(seen > 0, "a dense plan reads its own weights");
        // The embedding is param 0 and is read first, so it is what a budget
        // gives up LAST.
        assert_eq!(
            schedule.spill_order().last().copied(),
            Some(schedule.order()[0]),
            "the spill order is the read order reversed"
        );
    }

    #[test]
    fn a_slot_count_that_would_overwrite_a_live_plane_is_refused_with_the_count_that_works() {
        let trace = d0_8b();
        let schedule = Schedule::of(&trace);
        // Every plane spilled, one slot: the first plane is still live when
        // the second arrives, so one slot cannot serve it.
        let all = schedule.order();
        let why = schedule
            .slotting(&all, 1)
            .expect_err("one slot cannot hold a whole plan's planes in turn");
        assert!(why.want > 1, "and it says how many would: {why}");
        assert!(
            format!("{why}").contains("must be finished before its successor arrives"),
            "{why}"
        );

        // And the count it named does work, which is what makes the error an
        // instruction rather than a complaint.
        let works = schedule
            .slotting(&all, why.want)
            .expect("the count the refusal named serves the schedule");
        assert_eq!(works.slots(), why.want);
        assert_eq!(works.order().len(), all.len());
        // The assignment is round-robin over the schedule, which is what makes
        // it a compile-time constant: same plan, same slot, every boot.
        for (at, param) in works.order().iter().enumerate() {
            assert_eq!(works.slot_of(*param), Some((at as u32) % why.want));
        }
        assert_eq!(
            works,
            schedule.slotting(&all, why.want).expect("twice"),
            "and it is stable"
        );
    }

    #[test]
    fn a_slotting_sizes_each_slot_for_its_largest_tenant() {
        let trace = d0_8b();
        let schedule = Schedule::of(&trace);
        let all = schedule.order();
        let slots = schedule
            .slotting(&all, 1)
            .map_or_else(|why| why.want, |ok| ok.slots());
        let slotting = schedule.slotting(&all, slots).expect("it serves");
        let plane = |param: usize| -> u64 {
            let shape = &trace.params[param].shape;
            shape.iter().product::<u64>()
        };
        for slot in 0..slots {
            let sized = slotting.slot_bytes(slot, plane);
            for param in &all {
                if slotting.slot_of(*param) == Some(slot) {
                    assert!(
                        plane(*param) <= sized,
                        "slot {slot} is {sized} and param {param} wants {}",
                        plane(*param)
                    );
                }
            }
        }
    }
}
