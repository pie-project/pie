//! The calls a stepwise guest stage dispatches, whatever the backend: each
//! step of a stage is one call of a closed set ([`calls`]) on one node, at
//! one rung of its ladder, with one op selector, and the call reads all
//! three from the step's uniform ([`StepUniform`]). No call names a program,
//! so every pipeline (or module) a guest stage can dispatch is one of
//! [`calls`], and a backend builds them all before it serves: a program
//! registered while serving builds nothing. Each backend spells the calls
//! in its own language (`codegen::wgsl` spells them in WGSL).

use alloc::string::String;
use alloc::vec::Vec;

use eta_ir::op::tags;

use crate::codegen::launch::LaunchStagePlan;

/// The most fold levels a reduce's ladder takes: one per 32-fold.
pub const REDUCE_LEVELS: u32 = 7;

/// The most merge rounds a sort's ladder takes (even, so the ping-pong ends
/// in the buffer the finish reads).
pub const SORT_ROUNDS: u32 = 28;

/// The bytes of a step's record ([`StepUniform`]), in whole sixteens.
pub const STEP_BYTES: u32 = 48;

/// Where a step's record holds its node, rung, selector and its node's
/// temporary area, and the bases of its instance's tables, status and heap.
pub const NODE_AT: u32 = 0;
pub const RUNG_AT: u32 = 4;
pub const SELECTOR_AT: u32 = 8;
pub const AREA_AT: u32 = 12;
pub const BASES_AT: u32 = 16;

/// What a step's call reads: the node it runs (its op's params), the rung
/// of the node's ladder (a reduce's level, a sort's round, a select's
/// pass; 0 where it has none), and the op selector of a call that serves a
/// family of ops (which unary, which compare, whether a scatter adds; 0
/// where it serves one).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepUniform {
    pub node: u32,
    pub rung: u32,
    pub selector: u32,
    /// The word its node's temporary area begins at in the stage's heap, as
    /// a bind lays it (`eta_exec::tables::StageTables::areas`); 0 where the
    /// node keeps none.
    pub area: u32,
}

/// Where a stage's instance lies, in words of the segments a backend binds:
/// its status word and its descriptor, op param, offset and config tables
/// in the segment holding its slots, and its heap in the segment holding
/// its state. The host writes them, and a bind checks each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bases {
    pub status: u32,
    pub descs: u32,
    pub params: u32,
    pub offs: u32,
    pub cfg: u32,
    pub heap: u32,
}

impl StepUniform {
    /// The record's bytes, little-endian u32s: node, rung, selector and
    /// area at [`NODE_AT`], [`RUNG_AT`], [`SELECTOR_AT`] and [`AREA_AT`],
    /// then `bases` from [`BASES_AT`] (status, descriptors, params, offsets,
    /// config, heap), padded to [`STEP_BYTES`].
    #[must_use]
    pub fn bytes(self, bases: Bases) -> [u8; STEP_BYTES as usize] {
        let mut out = [0u8; STEP_BYTES as usize];
        let words = [
            (NODE_AT, self.node),
            (RUNG_AT, self.rung),
            (SELECTOR_AT, self.selector),
            (AREA_AT, self.area),
            (BASES_AT, bases.status),
            (BASES_AT + 4, bases.descs),
            (BASES_AT + 8, bases.params),
            (BASES_AT + 12, bases.offs),
            (BASES_AT + 16, bases.cfg),
            (BASES_AT + 20, bases.heap),
        ];
        for (at, word) in words {
            let at = at as usize;
            out[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reduce {
    Sum,
    Max,
    Min,
    Argmax,
    /// `gumbel_argmax`: an argmax whose lanes are the Gumbel-perturbed
    /// logits, made in the reduction's first level.
    GumbelArgmax,
}

/// What a sort ladder's finishing step lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Finish {
    SortDesc,
    TopK,
    Pivot,
}

/// A call of the closed set. A `guarded` call of a one-row select's ladder
/// runs only for a row the select handed over.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Call {
    /// Selector: which unary (the tag less `EXP`).
    Unary,
    /// Selector: which arithmetic (the tag less `ADD`; 6 for the last).
    Arith,
    /// Selector: which compare (the tag less `GT`).
    Compare,
    /// Selector: which logic op (the tag less `AND`).
    Logic,
    Select,
    /// Selector: 1 for a product.
    Cumulative,
    Broadcast,
    Copy,
    Transpose,
    Gather,
    GatherRow,
    /// Selector: 1 where it adds.
    Scatter,
    Iota,
    MaskApply,
    /// Selector: which structured mask (the tag less `CAUSAL_MASK`).
    StructMask,
    /// Selector: 1 where it is keyed.
    Rng,
    Matmul,
    Const,
    /// Rung: the level.
    ReduceLevel(Reduce),
    ReduceFinish(Reduce),
    SortSeed {
        guarded: bool,
    },
    /// Rung: the round.
    SortRound {
        guarded: bool,
    },
    SortPre {
        guarded: bool,
    },
    Finish {
        finish: Finish,
        guarded: bool,
    },
    PivotPack {
        guarded: bool,
    },
    SelInit,
    /// Rung: the pass.
    SelHist,
    /// Rung: the pass.
    SelPick,
    SelRankMask,
    SelTally,
    SelPlace,
    SelWalk,
    /// A top-k's candidates, sorted, its first `k` written as its results.
    SelTopK,
}

/// How a node's steps are laid: none (a sink call, the engine's own), one
/// call, the one-row select, the sort ladder, or the reduce ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ladder {
    None,
    Plain,
    Select { rank: bool, rounds: u32 },
    Sort { rounds: u32 },
    Reduce { levels: u32 },
}

/// One step of a node: its call, a reader's name for it (`sel`, `h0`,
/// `l2`, `step`), and its rung and selector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeStep {
    pub call: Call,
    pub suffix: String,
    pub rung: u32,
    pub selector: u32,
}

#[must_use]
pub const fn is_reduce(tag: u8) -> bool {
    matches!(
        tag,
        tags::REDUCE_SUM
            | tags::REDUCE_MAX
            | tags::REDUCE_MIN
            | tags::REDUCE_ARGMAX
            | tags::GUMBEL_ARGMAX
    )
}

#[must_use]
pub const fn is_sort(tag: u8) -> bool {
    matches!(tag, tags::SORT_DESC | tags::TOP_K | tags::PIVOT_THRESHOLD)
}

const fn reduce_of(tag: u8) -> Option<Reduce> {
    match tag {
        tags::REDUCE_SUM => Some(Reduce::Sum),
        tags::REDUCE_MAX => Some(Reduce::Max),
        tags::REDUCE_MIN => Some(Reduce::Min),
        tags::REDUCE_ARGMAX => Some(Reduce::Argmax),
        tags::GUMBEL_ARGMAX => Some(Reduce::GumbelArgmax),
        _ => None,
    }
}

const fn finish_of(tag: u8) -> Option<Finish> {
    match tag {
        tags::SORT_DESC => Some(Finish::SortDesc),
        tags::TOP_K => Some(Finish::TopK),
        tags::PIVOT_THRESHOLD => Some(Finish::Pivot),
        _ => None,
    }
}

/// The call a node of `tag` that takes one step runs, and its selector;
/// `None` for a tag no call serves.
#[must_use]
pub const fn plain_of(tag: u8) -> Option<(Call, u32)> {
    Some(match tag {
        0x01..=0x0B => (Call::Unary, (tag - 0x01) as u32),
        0x10..=0x15 => (Call::Arith, (tag - 0x10) as u32),
        0x1F => (Call::Arith, 6),
        0x16..=0x1B => (Call::Compare, (tag - 0x16) as u32),
        0x1C..=0x1E => (Call::Logic, (tag - 0x1C) as u32),
        0x20 => (Call::Select, 0),
        0x40 => (Call::Cumulative, 0),
        0x41 => (Call::Cumulative, 1),
        0x38 => (Call::Broadcast, 0),
        // A reshape a source does not cover, and a kernel call, copy.
        0x39 | 0xA1 => (Call::Copy, 0),
        0x3A => (Call::Transpose, 0),
        0x60 => (Call::Gather, 0),
        0x61 => (Call::GatherRow, 0),
        0x62 => (Call::Scatter, 1),
        0x63 => (Call::Scatter, 0),
        0x64 => (Call::Iota, 0),
        0x65 => (Call::MaskApply, 0),
        0x66..=0x68 => (Call::StructMask, (tag - 0x66) as u32),
        0x70 => (Call::Rng, 0),
        0x71 => (Call::Rng, 1),
        0x55 => (Call::Matmul, 0),
        0x81 => (Call::Const, 0),
        _ => match finish_of(tag) {
            Some(finish) => (
                Call::Finish {
                    finish,
                    guarded: false,
                },
                0,
            ),
            None => return None,
        },
    })
}

fn arg_axes(plan: &LaunchStagePlan, at: usize) -> Option<&[crate::plan::Dimension]> {
    let arg = *plan.ops.get(at)?.args.first()?;
    Some(&plan.value_types.get(arg as usize)?.axes)
}

fn last_extent(plan: &LaunchStagePlan, at: usize) -> Option<u64> {
    match arg_axes(plan, at)?.last() {
        Some(crate::plan::Dimension::Static(extent)) => Some(u64::from(*extent)),
        None => Some(1),
        _ => None,
    }
}

/// The merge rounds a sort over `at`'s argument needs: enough to merge its
/// whole row, even; a row of unknown length gets the full ladder.
#[must_use]
pub fn sort_rounds(plan: &LaunchStagePlan, at: usize) -> u32 {
    let Some(op) = plan.ops.get(at) else {
        return SORT_ROUNDS;
    };
    let Some(&arg) = op.args.first() else {
        return SORT_ROUNDS;
    };
    let Some(value) = plan.value_types.get(arg as usize) else {
        return SORT_ROUNDS;
    };
    let mut total: u64 = 1;
    for axis in &value.axes {
        let crate::plan::Dimension::Static(extent) = axis else {
            return SORT_ROUNDS;
        };
        total *= u64::from(*extent);
    }
    let mut need = 0u32;
    while (1u64 << need) < total {
        need += 1;
        if need >= SORT_ROUNDS {
            return SORT_ROUNDS;
        }
    }
    let rounds = need + (need & 1);
    rounds.min(SORT_ROUNDS)
}

/// The fold levels a reduce over `at`'s argument needs: one per 32-fold
/// until a row is one cell. A row of unknown length gets the full ladder.
#[must_use]
pub fn reduce_levels(plan: &LaunchStagePlan, at: usize) -> u32 {
    let Some(mut len) = last_extent(plan, at) else {
        return REDUCE_LEVELS;
    };
    let first = plan.ops.get(at).map_or(32, |op| first_fan(op.tag));
    let mut levels = 0;
    while len > 1 && levels < REDUCE_LEVELS {
        len = len.div_ceil(if levels == 0 { first } else { 32 });
        levels += 1;
    }
    levels
}

/// The lanes a job of a reduction's first level folds (the runtime's
/// `red_fan0`): 32, but 8 for a `gumbel_argmax`, whose lanes each hash the
/// rng and take two logs, so a row spreads over four times the
/// invocations. Every level above folds 32.
#[must_use]
pub const fn first_fan(tag: u8) -> u64 {
    if tag == tags::GUMBEL_ARGMAX { 8 } else { 32 }
}

/// Whether `at`'s argument is one row, which the one-row select handles.
#[must_use]
pub fn one_row(plan: &LaunchStagePlan, at: usize) -> bool {
    let Some(axes) = arg_axes(plan, at) else {
        return false;
    };
    let Some((_, leading)) = axes.split_last() else {
        return true;
    };
    leading
        .iter()
        .all(|axis| matches!(axis, crate::plan::Dimension::Static(1)))
}

/// The ladder node `at` (of `tag`) takes.
#[must_use]
pub fn ladder_of(plan: &LaunchStagePlan, at: usize, tag: u8) -> Ladder {
    if tag == tags::SINK_CALL {
        return Ladder::None;
    }
    let pivot = (tag == tags::PIVOT_THRESHOLD).then(|| plan.ops[at].pred_tag);
    if pivot.is_some_and(|pred| pred > 1) {
        // ProbGe compares each lane with the threshold; nothing to order.
        Ladder::Plain
    } else if pivot.is_some() && one_row(plan, at) {
        Ladder::Select {
            rank: pivot == Some(0),
            rounds: sort_rounds(plan, at),
        }
    } else if tag == tags::TOP_K && one_row(plan, at) {
        // One row's top `k`: its `k`-th key selected, the keys past it
        // sorted; the row's sort where the select hands it over.
        Ladder::Select {
            rank: false,
            rounds: sort_rounds(plan, at),
        }
    } else if is_sort(tag) {
        Ladder::Sort {
            rounds: sort_rounds(plan, at),
        }
    } else if is_reduce(tag) {
        Ladder::Reduce {
            levels: reduce_levels(plan, at),
        }
    } else {
        Ladder::Plain
    }
}

/// Every ladder a node of `tag` can take: what [`calls`] closes over.
#[must_use]
pub fn ladders_of(tag: u8) -> Vec<Ladder> {
    if tag == tags::SINK_CALL {
        // The engine's own: the device runs nothing for it.
        alloc::vec![Ladder::None]
    } else if tag == tags::PIVOT_THRESHOLD {
        alloc::vec![
            Ladder::Plain,
            Ladder::Select {
                rank: true,
                rounds: 1
            },
            Ladder::Select {
                rank: false,
                rounds: 1
            },
            Ladder::Sort { rounds: 1 },
        ]
    } else if tag == tags::TOP_K {
        alloc::vec![
            Ladder::Select {
                rank: false,
                rounds: 1
            },
            Ladder::Sort { rounds: 1 },
        ]
    } else if is_sort(tag) {
        alloc::vec![Ladder::Sort { rounds: 1 }]
    } else if is_reduce(tag) {
        alloc::vec![Ladder::Reduce { levels: 1 }]
    } else {
        alloc::vec![Ladder::Plain]
    }
}

/// A node of `tag`'s steps on `ladder`; `None` where `tag` takes one step
/// and no call serves it.
#[must_use]
pub fn steps_of(tag: u8, ladder: Ladder) -> Option<Vec<NodeStep>> {
    let mut steps = Vec::new();
    let mut push = |call: Call, suffix: String, rung: u32, selector: u32| {
        steps.push(NodeStep {
            call,
            suffix,
            rung,
            selector,
        });
    };
    match ladder {
        Ladder::None => {}
        Ladder::Plain => {
            let (call, selector) = plain_of(tag)?;
            push(call, "step".into(), 0, selector);
        }
        Ladder::Select { rank, rounds } => {
            push(Call::SelInit, "sel".into(), 0, 0);
            for pass in 0..4 {
                push(Call::SelHist, alloc::format!("h{pass}"), pass, 0);
                push(Call::SelPick, alloc::format!("p{pass}"), pass, 0);
            }
            if rank {
                push(Call::SelRankMask, "mask".into(), 0, 0);
            } else {
                push(Call::SelTally, "tally".into(), 0, 0);
                push(Call::SelPlace, "place".into(), 0, 0);
                if tag == tags::TOP_K {
                    push(Call::SelTopK, "top".into(), 0, 0);
                } else {
                    push(Call::SelWalk, "walk".into(), 0, 0);
                }
                sort_steps(&mut push, tag, rounds, true)?;
            }
        }
        Ladder::Sort { rounds } => sort_steps(&mut push, tag, rounds, false)?,
        Ladder::Reduce { levels } => {
            let reduce = reduce_of(tag)?;
            for level in 0..levels {
                push(
                    Call::ReduceLevel(reduce),
                    alloc::format!("l{level}"),
                    level,
                    0,
                );
            }
            push(Call::ReduceFinish(reduce), "fin".into(), 0, 0);
        }
    }
    Some(steps)
}

fn sort_steps(
    push: &mut impl FnMut(Call, String, u32, u32),
    tag: u8,
    rounds: u32,
    guarded: bool,
) -> Option<()> {
    let finish = finish_of(tag)?;
    push(Call::SortSeed { guarded }, "seed".into(), 0, 0);
    for round in 0..rounds {
        push(
            Call::SortRound { guarded },
            alloc::format!("r{round}"),
            round,
            0,
        );
    }
    push(Call::SortPre { guarded }, "pre".into(), 0, 0);
    push(Call::Finish { finish, guarded }, "step".into(), 0, 0);
    push(Call::PivotPack { guarded }, "pack".into(), 0, 0);
    Some(())
}

/// The calls the steps of every ladder of every tag in `stepped` take,
/// each once, in a fixed order: a stage's steps index it.
#[must_use]
pub fn calls_of(stepped: impl Fn(u8) -> bool) -> Vec<Call> {
    let mut calls: Vec<Call> = Vec::new();
    for tag in 0..=u8::MAX {
        if !stepped(tag) {
            continue;
        }
        for ladder in ladders_of(tag) {
            for step in steps_of(tag, ladder).into_iter().flatten() {
                if !calls.contains(&step.call) {
                    calls.push(step.call);
                }
            }
        }
    }
    calls
}

/// The first tag in `stepped` and its call that `calls` does not hold, or
/// whose single step no call serves (as `None`): a set that passes serves
/// every step of every ladder of every tag a backend steps.
#[must_use]
pub fn unlisted(calls: &[Call], stepped: impl Fn(u8) -> bool) -> Option<(u8, Option<Call>)> {
    for tag in 0..=u8::MAX {
        if !stepped(tag) {
            continue;
        }
        for ladder in ladders_of(tag) {
            let Some(steps) = steps_of(tag, ladder) else {
                return Some((tag, None));
            };
            if let Some(step) = steps.iter().find(|step| !calls.contains(&step.call)) {
                return Some((tag, Some(step.call)));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_lays_its_words_where_it_says() {
        let bases = Bases {
            status: 11,
            descs: 13,
            params: 17,
            offs: 19,
            cfg: 23,
            heap: 29,
        };
        let bytes = StepUniform {
            node: 7,
            rung: 3,
            selector: 2,
            area: 9,
        }
        .bytes(bases);
        let word = |at: u32| {
            let at = at as usize;
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        assert_eq!(
            (word(NODE_AT), word(RUNG_AT), word(SELECTOR_AT), word(AREA_AT)),
            (7, 3, 2, 9)
        );
        let laid: Vec<u32> = (0..6).map(|k| word(BASES_AT + 4 * k)).collect();
        assert_eq!(laid, [11, 13, 17, 19, 23, 29]);
        assert_eq!((word(40), word(44)), (0, 0), "the pad is zero");
        assert_eq!(STEP_BYTES % 16, 0, "a record is whole sixteens");
    }

    #[test]
    fn a_family_of_ops_is_one_call_and_a_sink_call_none() {
        assert_eq!(plain_of(tags::EXP), Some((Call::Unary, 0)));
        assert_eq!(plain_of(tags::LOG).map(|(call, _)| call), Some(Call::Unary));
        assert_eq!(steps_of(tags::SINK_CALL, Ladder::None), Some(Vec::new()));
        assert_eq!(plain_of(tags::SINK_CALL), None);
    }

    /// The closed set serves every tag it was drawn from; short of any one
    /// call it does not (the negative control).
    #[test]
    fn the_set_closes_over_its_tags_and_no_call_is_spare() {
        let stepped = |tag: u8| eta_ir::op::spec(tag).is_some() && tag != tags::OUTPUT;
        let every: Vec<u8> = (0..=u8::MAX)
            .filter(|&tag| {
                stepped(tag)
                    && ladders_of(tag)
                        .into_iter()
                        .all(|ladder| steps_of(tag, ladder).is_some())
            })
            .collect();
        let served = |tag: u8| every.contains(&tag);
        let calls = calls_of(served);
        assert_eq!(unlisted(&calls, served), None);
        for gone in 0..calls.len() {
            let mut short = calls.clone();
            let call = short.remove(gone);
            assert!(
                unlisted(&short, served).is_some(),
                "the set without {call:?} still serves every tag"
            );
        }
    }
}
