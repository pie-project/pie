use std::collections::BTreeMap;
use std::ops::Range;

use model_ir::{Def, Operands, Trace, ValueId};

use crate::compiled::CompiledModel;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reads {
    pub param: usize,
    pub span: Range<u32>,
    pub reads: u32,
}

impl Reads {
    #[must_use]
    pub const fn unread(&self) -> bool {
        self.reads == 0
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Schedule {
    reads: Vec<Reads>,
    of: BTreeMap<usize, usize>,
    nodes: u32,
}

impl Schedule {
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

    #[must_use]
    pub fn reads(&self) -> &[Reads] {
        &self.reads
    }

    #[must_use]
    pub fn read_of(&self, param: usize) -> Option<Reads> {
        self.of.get(&param).map(|at| self.reads[*at].clone())
    }

    #[must_use]
    pub fn order(&self) -> Vec<usize> {
        self.reads.iter().map(|row| row.param).collect()
    }

    #[must_use]
    pub fn spill_order(&self) -> Vec<usize> {
        let mut out = self.order();
        out.reverse();
        out
    }

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

    #[must_use]
    pub const fn nodes(&self) -> u32 {
        self.nodes
    }

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
        for pair in queue.windows(slots as usize + 1) {
            let (evicted, by) = (&pair[0], &pair[slots as usize]);
            if evicted.unread() {
                continue;
            }
            if evicted.span.end > by.span.start {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slotting {
    of: BTreeMap<usize, u32>,
    slots: u32,
    order: Vec<usize>,
}

impl Slotting {
    #[must_use]
    pub fn slot_of(&self, param: usize) -> Option<u32> {
        self.of.get(&param).copied()
    }

    #[must_use]
    pub const fn slots(&self) -> u32 {
        self.slots
    }

    #[must_use]
    pub fn order(&self) -> &[usize] {
        &self.order
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Overlap {
    pub slots: u32,
    pub want: u32,
    pub evicted: usize,
    pub by: usize,
    pub live: Range<u32>,
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

fn weight_of(trace: &Trace, id: ValueId) -> Option<usize> {
    match trace.values.get(id.0 as usize).map(|decl| &decl.def) {
        Some(Def::Weight(w)) => Some(*w as usize),
        _ => None,
    }
}

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
        let trace = models::sku("qwen35-d0.8b-bf16-kv-bf16")
            .expect("the catalog ships it")
            .trace;
        trace(Platform::Cuda)
    }

    #[test]
    fn a_slot_count_that_would_overwrite_a_live_plane_is_refused_with_the_count_that_works() {
        let trace = d0_8b();
        let schedule = Schedule::of(&trace);
        let all = schedule.order();
        let why = schedule
            .slotting(&all, 1)
            .expect_err("one slot cannot hold a whole plan's planes in turn");
        assert!(why.want > 1, "and it says how many would: {why}");
        assert!(
            format!("{why}").contains("must be finished before its successor arrives"),
            "{why}"
        );

        let works = schedule
            .slotting(&all, why.want)
            .expect("the count the refusal named serves the schedule");
        assert_eq!(works.slots(), why.want);
        assert_eq!(works.order().len(), all.len());
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
