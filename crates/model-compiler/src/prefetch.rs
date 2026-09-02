//! The prefetch schedule: which params a fire reads, in what order, and how
//! a fixed set of device slots can serve them all — computed once from the
//! trace alone, since a dense plan's read order is fire-invariant. Not
//! stored on `CompiledModel`: it is a pure function of data already hashed
//! into the artifact, so storing it would buy nothing.

use std::collections::BTreeMap;
use std::ops::Range;

use model_ir::{Def, Operands, Trace, ValueId};

use crate::compiled::CompiledModel;

/// Where in the fire one param is read. `span` is half-open in nodes;
/// `reads` is how many distinct nodes name it.
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
    /// Is this plane read by nothing? A real case: a plan may declare a
    /// param no region names.
    #[must_use]
    pub const fn unread(&self) -> bool {
        self.reads == 0
    }
}

/// Every param, in the order the fire reaches it: ascending by first read
/// then param index, so two compiles of one plan produce the same sequence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Schedule {
    reads: Vec<Reads>,
    of: BTreeMap<usize, usize>,
    nodes: u32,
}

impl Schedule {
    /// The schedule for `trace`. Params nothing reads sort last, since a
    /// plane no fire touches is the first thing a budget should give up.
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

    /// The order a prefetch issues copies in: earliest-read first.
    #[must_use]
    pub fn order(&self) -> Vec<usize> {
        self.reads.iter().map(|row| row.param).collect()
    }

    /// The order a budget gives planes up in: the reverse of [`order`](Self::order)
    /// — unread first, embedding (read at node zero) last.
    #[must_use]
    pub fn spill_order(&self) -> Vec<usize> {
        let mut out = self.order();
        out.reverse();
        out
    }

    /// Projects onto regions, the granularity a pump works at. Order is
    /// unchanged: a region is a maximal run of adjacent nodes.
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

    /// The slot assignment, as a compile-time constant: `spilled` params are
    /// walked in schedule order and dealt round-robin into `slots` slots, so
    /// a graph recorded against slot `k`'s address stays correct forever.
    /// This is checked rather than assumed: plane `i` and plane `i + slots`
    /// share a slot, so the schedule must satisfy
    /// `last_read(i) <= first_read(i + slots)`.
    ///
    /// # Errors
    ///
    /// [`Overlap`] when no slot count that small keeps every tenant alive to
    /// its last read; the error carries the smallest count that would work.
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
        // Walk each slot's tenants in order; each one's last read must fall
        // at or before the next one's first.
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

/// A proved slot assignment: which slot holds which plane, and in what
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

    /// The slot's own size: the largest plane it ever holds (smaller
    /// tenants use a prefix). `plane` answers a param's bytes.
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

/// Why a slot count cannot serve a schedule: a plane would be overwritten
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
        let trace = models::sku("qwen35-d0.8b-bf16-kv-bf16").expect("the catalog ships it").trace;
        trace(Platform::Cuda)
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

}
