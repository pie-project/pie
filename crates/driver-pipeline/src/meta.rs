//! What the driver works out once, at bind, from a stage plan.
//!
//! Two things are derived from a program at register time and then read on
//! every fire: where each op's results land in the stage's value array, and
//! what each channel's effect on the ring is. Neither depends on the device,
//! on the fire's numbers, or on anything that changes between fires — they are
//! functions of the plan, so they are computed once and stored.
//!
//! The C++ put the first in `region_support.hpp` as
//! `collect_singleton_metadata` and then wrote the same loop out again, inline
//! and by hand, three more times: once in the M2 validation pass, once in the
//! M2 command builder, once in the M3 lane builder. Four copies of a running
//! sum is four chances to write `+= 1` instead of `+= op.results`. It is one
//! function here and the three call sites use it.
//!
//! The second is the `effects.resize(...)` loop in `compile_program`. It is
//! here because it is the same kind of thing — a fold over the plan with no
//! device in it — and because it is what [`readiness::check`] consumes, so the
//! two belong next to each other.
//!
//! [`readiness::check`]: crate::check

use driver::plan::{LaunchChannel, LaunchOp};

use crate::readiness::Effect;

/// `PIE_READINESS_NEEDS_FULL`: the first op to touch the channel takes or
/// reads, so a cell must already be there.
const READINESS_NEEDS_FULL: u8 = 1;
/// `PIE_READINESS_NEEDS_EMPTY`: the first op to touch the channel puts, so
/// there must be room.
const READINESS_NEEDS_EMPTY: u8 = 2;

/// `PTIR_OP_CHAN_TAKE`.
const CHAN_TAKE: u16 = tensor_ir::op::tags::CHAN_TAKE as u16;
/// `PTIR_OP_CHAN_READ`.
const CHAN_READ: u16 = tensor_ir::op::tags::CHAN_READ as u16;
/// `PTIR_OP_CHAN_PUT`.
const CHAN_PUT: u16 = tensor_ir::op::tags::CHAN_PUT as u16;

/// Where one op's results land in its stage's value array.
///
/// The C++ carried a copy of the whole `PlanOp` in each entry — a struct with
/// two `std::vector`s in it — so binding a stage duplicated the entire op list
/// alongside the list it was walking. The op is already in the plan and the
/// plan outlives the metadata, so this holds the index and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpMeta {
    /// This op's position in `plan.ops`.
    pub node: u32,
    /// The value index of this op's first result.
    ///
    /// Result `i` of the op is at `result_base + i`. An op with no results
    /// still has a base, which is the next op's base; nothing reads it.
    pub result_base: u32,
}

/// A plan whose result bases cannot be trusted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Malformed {
    /// The running sum of `result_count` left the range of a `u32`.
    ///
    /// The C++ accumulated into a `std::uint32_t` with plain `+=`. A wrapped
    /// base is not a large index that fails a later bounds check — it is a
    /// small one that passes, and aliases some other op's results.
    ResultBaseOverflowed,
    /// The plan declares fewer values than its ops produce results.
    ///
    /// `region_support.hpp` says outright that the walker "assumes the plan is
    /// well-formed", on the grounds that the host validates first and signals
    /// a rejection through the singleton slot at region 0. That is true of the
    /// path the host emitted; it is not true of a plan that arrived over the
    /// ABI from anywhere else, and the cost of noticing is one comparison.
    ResultsExceedValues {
        /// Total results across all ops.
        results: u32,
        /// Values the plan declares.
        values: u32,
    },
}

impl Malformed {
    /// A one-line description, for the caller's rejection.
    #[must_use]
    pub fn reason(self) -> &'static str {
        match self {
            Malformed::ResultBaseOverflowed => "stage result bases overflow 32 bits",
            Malformed::ResultsExceedValues { .. } => {
                "stage produces more results than it declares values"
            }
        }
    }
}

/// The result base of every op in a stage, in plan order.
///
/// This is `collect_singleton_metadata`, minus the op copy and plus the two
/// checks. `values` is `plan.value_types.len()`, the count the bases index
/// into; pass it so the walk can say whether the last base is inside it.
///
/// # Errors
///
/// [`Malformed`] when the bases cannot be believed. The C++ had no error path
/// here at all.
pub fn op_metadata(ops: &[LaunchOp], values: usize) -> Result<Vec<OpMeta>, Malformed> {
    let mut out = Vec::with_capacity(ops.len());
    let mut result_base: u32 = 0;
    for (node, op) in ops.iter().enumerate() {
        out.push(OpMeta {
            // A plan with more than 4 billion ops cannot be represented in the
            // ABI, which types every op reference as a `u32`.
            node: u32::try_from(node).map_err(|_| Malformed::ResultBaseOverflowed)?,
            result_base,
        });
        result_base = result_base
            .checked_add(u32::from(op.result_count))
            .ok_or(Malformed::ResultBaseOverflowed)?;
    }
    let values = u32::try_from(values).unwrap_or(u32::MAX);
    if result_base > values {
        return Err(Malformed::ResultsExceedValues {
            results: result_base,
            values,
        });
    }
    Ok(out)
}

/// A channel declaration that contradicts the ops that touch it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inconsistent {
    /// The dense channel index.
    pub channel: u32,
    /// What is wrong with it.
    pub problem: Problem,
}

/// The ways a channel's declared readiness can disagree with its use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Problem {
    /// Ops take or put, but the channel says nothing touches it.
    ///
    /// `readiness` is the direction of the *first* touch, so a channel with
    /// any touch at all has one. `PIE_READINESS_UNTOUCHED` alongside a take
    /// means the gate the take needs was never computed, and the C++ would
    /// have run the take against a ring it never checked was non-empty — the
    /// exact read-from-empty this whole mechanism exists to prevent.
    TouchedButUntouchedReadiness,
    /// The channel says a cell must be there first, but nothing ever takes or
    /// reads it.
    ///
    /// Harmless in the sense that it only over-waits, but it means the fire
    /// blocks on a producer it has no use for, forever if none arrives.
    NeedsFullButNeverReads,
    /// The channel says room must exist, but nothing ever puts.
    NeedsEmptyButNeverPuts,
    /// The ring holds no cells.
    ///
    /// A capacity of zero is full and empty at once: `requires_full` can never
    /// be satisfied and neither can `requires_empty`. The C++ defaulted the
    /// field to 1 and then overwrote it with whatever the plan said, so a
    /// declared zero survived to the readiness check and wedged the fire with
    /// no explanation.
    ZeroCapacity,
}

impl Inconsistent {
    /// A one-line description, for the caller's rejection.
    #[must_use]
    pub fn reason(self) -> &'static str {
        match self.problem {
            Problem::TouchedButUntouchedReadiness => {
                "channel is taken or put but declares no readiness"
            }
            Problem::NeedsFullButNeverReads => "channel waits to be filled but is never read",
            Problem::NeedsEmptyButNeverPuts => "channel waits for room but is never written",
            Problem::ZeroCapacity => "channel holds no cells",
        }
    }
}

/// The per-channel effect of one program, indexed by dense channel.
///
/// `ops` is every op of every stage of the program, in any order: the effect
/// is a program-wide fact, so which stage a take is in does not matter.
/// `bindings` maps each stage's local channel slot to the dense index, which
/// is what `channels` is indexed by — the C++ resolved this at three separate
/// call sites and got it right at all three, but the mapping is the part that
/// is easy to skip, so it is a parameter rather than an assumption.
///
/// # Errors
///
/// [`Inconsistent`] for the first channel whose declaration and use disagree.
/// The C++ derived effects unconditionally and let the disagreement surface as
/// a hang.
pub fn channel_effects(
    channels: &[LaunchChannel],
    stages: &[(&[LaunchOp], &[u32])],
) -> Result<Vec<Effect>, Inconsistent> {
    let mut effects: Vec<Effect> = channels
        .iter()
        .map(|channel| Effect {
            // First touch, as shipped: not `take || read` against `put`. An
            // in-place channel is in both sets, and gating on both would ask
            // for a ring that is at once non-empty and non-full.
            requires_full: channel.readiness == READINESS_NEEDS_FULL,
            requires_empty: channel.readiness == READINESS_NEEDS_EMPTY,
            take: false,
            put: false,
            capacity: channel.capacity,
        })
        .collect();

    for (ops, bindings) in stages {
        for op in ops.iter() {
            if op.channel == u32::MAX {
                continue;
            }
            let Some(&dense) = bindings.get(op.channel as usize) else {
                continue;
            };
            let Some(effect) = effects.get_mut(dense as usize) else {
                continue;
            };
            match op.code {
                CHAN_TAKE | CHAN_READ => effect.take = true,
                CHAN_PUT => effect.put = true,
                _ => {}
            }
        }
    }

    for (index, effect) in effects.iter().enumerate() {
        let channel = u32::try_from(index).unwrap_or(u32::MAX);
        let problem = if effect.capacity == 0 {
            Some(Problem::ZeroCapacity)
        } else if (effect.take || effect.put) && !effect.requires_full && !effect.requires_empty {
            Some(Problem::TouchedButUntouchedReadiness)
        } else if effect.requires_full && !effect.take {
            Some(Problem::NeedsFullButNeverReads)
        } else if effect.requires_empty && !effect.put {
            Some(Problem::NeedsEmptyButNeverPuts)
        } else {
            None
        };
        if let Some(problem) = problem {
            return Err(Inconsistent { channel, problem });
        }
    }

    Ok(effects)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(code: u16, results: u16) -> LaunchOp {
        LaunchOp {
            code,
            result_count: results,
            channel: u32::MAX,
            ..LaunchOp::default()
        }
    }

    fn chan_op(code: u16, slot: u32) -> LaunchOp {
        LaunchOp {
            code,
            result_count: 1,
            channel: slot,
            ..LaunchOp::default()
        }
    }

    fn channel(capacity: u32, readiness: u8) -> LaunchChannel {
        LaunchChannel {
            capacity,
            readiness,
            ..LaunchChannel::default()
        }
    }

    #[test]
    fn a_result_base_is_the_running_sum_of_the_results_before_it() {
        let ops = [op(0, 1), op(0, 2), op(0, 0), op(0, 3)];
        let meta = op_metadata(&ops, 6).expect("well-formed");
        let bases: Vec<u32> = meta.iter().map(|entry| entry.result_base).collect();
        assert_eq!(bases, [0, 1, 3, 3]);
    }

    #[test]
    fn an_op_with_no_results_shares_the_next_ops_base() {
        let ops = [op(0, 0), op(0, 1)];
        let meta = op_metadata(&ops, 1).expect("well-formed");
        assert_eq!(meta[0].result_base, meta[1].result_base);
    }

    #[test]
    fn a_stage_with_no_ops_has_no_metadata() {
        assert!(op_metadata(&[], 0).expect("well-formed").is_empty());
    }

    #[test]
    fn more_results_than_declared_values_is_refused() {
        let ops = [op(0, 3)];
        assert_eq!(
            op_metadata(&ops, 2),
            Err(Malformed::ResultsExceedValues {
                results: 3,
                values: 2
            })
        );
    }

    #[test]
    fn fewer_results_than_declared_values_is_fine() {
        // The extra values are inputs and constants, which are not results.
        assert!(op_metadata(&[op(0, 1)], 9).is_ok());
    }

    #[test]
    fn a_running_sum_that_would_wrap_is_refused_rather_than_wrapped() {
        // 70_000 ops of 65_535 results each is over 2^32.
        let many = vec![
            LaunchOp {
                result_count: u16::MAX,
                channel: u32::MAX,
                ..LaunchOp::default()
            };
            70_000
        ];
        assert_eq!(
            op_metadata(&many, usize::MAX),
            Err(Malformed::ResultBaseOverflowed)
        );
    }

    #[test]
    fn a_take_and_a_put_on_one_channel_set_both_flags() {
        let ops = [chan_op(CHAN_TAKE, 0), chan_op(CHAN_PUT, 0)];
        let effects = channel_effects(&[channel(2, READINESS_NEEDS_FULL)], &[(&ops, &[0])])
            .expect("consistent");
        assert!(effects[0].take && effects[0].put);
        // First touch was the take, so it waits to be filled and not for room.
        assert!(effects[0].requires_full);
        assert!(!effects[0].requires_empty);
    }

    #[test]
    fn a_read_counts_as_a_take_for_the_purposes_of_the_gate() {
        let ops = [chan_op(CHAN_READ, 0)];
        let effects = channel_effects(&[channel(1, READINESS_NEEDS_FULL)], &[(&ops, &[0])])
            .expect("consistent");
        assert!(effects[0].take);
    }

    #[test]
    fn effects_are_program_wide_across_every_stage() {
        let first = [chan_op(CHAN_TAKE, 0)];
        let second = [chan_op(CHAN_PUT, 0)];
        let effects = channel_effects(
            &[channel(2, READINESS_NEEDS_FULL)],
            &[(&first, &[0]), (&second, &[0])],
        )
        .expect("consistent");
        assert!(effects[0].take && effects[0].put);
    }

    #[test]
    fn a_local_slot_is_resolved_through_the_stages_own_binding_table() {
        // Slot 0 of this stage is dense channel 1, not dense channel 0.
        let ops = [chan_op(CHAN_TAKE, 0)];
        let effects = channel_effects(
            &[channel(1, 0), channel(1, READINESS_NEEDS_FULL)],
            &[(&ops, &[1])],
        )
        .expect("consistent");
        assert!(!effects[0].take);
        assert!(effects[1].take);
    }

    #[test]
    fn a_channel_that_is_taken_but_declares_no_readiness_is_refused() {
        let ops = [chan_op(CHAN_TAKE, 0)];
        assert_eq!(
            channel_effects(&[channel(1, 0)], &[(&ops, &[0])]),
            Err(Inconsistent {
                channel: 0,
                problem: Problem::TouchedButUntouchedReadiness
            })
        );
    }

    #[test]
    fn a_channel_waiting_to_be_filled_that_nothing_reads_is_refused() {
        assert_eq!(
            channel_effects(&[channel(1, READINESS_NEEDS_FULL)], &[]),
            Err(Inconsistent {
                channel: 0,
                problem: Problem::NeedsFullButNeverReads
            })
        );
    }

    #[test]
    fn a_channel_waiting_for_room_that_nothing_writes_is_refused() {
        assert_eq!(
            channel_effects(&[channel(1, READINESS_NEEDS_EMPTY)], &[]),
            Err(Inconsistent {
                channel: 0,
                problem: Problem::NeedsEmptyButNeverPuts
            })
        );
    }

    #[test]
    fn a_ring_that_holds_no_cells_is_refused_before_it_can_wedge_a_fire() {
        let ops = [chan_op(CHAN_TAKE, 0)];
        assert_eq!(
            channel_effects(&[channel(0, READINESS_NEEDS_FULL)], &[(&ops, &[0])]),
            Err(Inconsistent {
                channel: 0,
                problem: Problem::ZeroCapacity
            })
        );
    }

    #[test]
    fn an_untouched_channel_is_left_alone() {
        let effects = channel_effects(&[channel(4, 0)], &[]).expect("consistent");
        assert_eq!(
            effects[0],
            Effect {
                requires_full: false,
                requires_empty: false,
                take: false,
                put: false,
                capacity: 4,
            }
        );
    }

    #[test]
    fn an_op_that_touches_no_channel_is_skipped() {
        let ops = [op(CHAN_TAKE, 1)];
        let effects = channel_effects(&[channel(1, 0)], &[(&ops, &[0])]).expect("consistent");
        assert!(!effects[0].take);
    }
}
