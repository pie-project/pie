//! What the WGSL arm decides BEFORE it emits, and what a shell must do with
//! those decisions.
//!
//! The CUDA arm runs two analyses ahead of its fused emitter
//! ([`super::cuda::fused`]): reshape aliasing, and direct argmax. Both are
//! decisions about which nodes need no code, and both are wrong in the same
//! dangerous way if got wrong — an elided node whose value nothing else
//! produces does not fault, it reads a zeroed slot, and a sampler over a
//! zeroed row draws token 0 forever. So the invariant CUDA states is the one
//! kept here, verbatim: **only ever elide a node whose value some other
//! emission still produces.**
//!
//! ## Why the shell has to participate
//!
//! In the CUDA arm an operand is spelled into the generated source as
//! `scratch + offsets[aliases.resolve(value)]`, so eliding is entirely the
//! emitter's business. The WGSL runtime does not work that way: `ptir_step`
//! reads its operands out of `params[node]` as value ids and finds their
//! bytes through `base(v) = offs[v] >> 2`, and both tables are built by the
//! SHELL. The emitter can therefore decide an elision but cannot enact one.
//!
//! That is not a limitation, it is the cleaner split: [`analyze_stage`] is a
//! pure function of the plan, the emitter drops the elided nodes' calls, and
//! the shell points `offs[result]` at `offs[source]`. A consumer reading the
//! elided value then reads the source's bytes, while `descs[result]` still
//! carries the result's own shape — which is exactly what a reshape means.
//!
//! ## Why direct argmax is smaller here than on CUDA
//!
//! CUDA's `analyze_direct_argmax` elides the whole chain from a logits
//! intrinsic to an `argmax`: the reshapes AND the `intrinsic_val`. On this
//! arm `intrinsic_val` is a boundary op the emitter already skips and the
//! shell already stages, and the reshapes between are elided by ordinary
//! aliasing, whose `resolve` walks the chain to the staged logits value. So
//! the analysis has nothing left to elide, and what it produces is the
//! RECORD: a shell that recognises "this pass is an argmax of the logits" may
//! answer it with its own kernel and skip the guest dispatch entirely, which
//! is what CUDA's `RegionAnalysis::direct_argmax` exists to enable.
//!
//! Expressing the chain through the alias table rather than through a second
//! elision list is also what keeps the invariant above true by construction:
//! every elided value resolves to one the shell writes.

use alloc::vec::Vec;

use eta_ir::op::{IntrinsicId, intrinsic_tags, tags};

use crate::codegen::alias::{AliasTable, covers};
use crate::codegen::launch::{LaunchPlanValue, LaunchStagePlan};
use crate::plan::{Dimension, SymbolicType};

/// An `argmax` a shell may answer without running the emitted pass, by
/// reading a logits intrinsic's buffer directly.
///
/// The typed, sparse form — the dense per-node arrays CUDA carries are an
/// implementation detail of its emitter, and a shell only ever wants the
/// handful of rows that apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectArgmax {
    /// The `argmax` node, by index into the stage's ops.
    pub node: u32,
    /// The value id whose bytes it reads — the logits intrinsic's result,
    /// which the shell stages.
    pub source_value: u32,
    /// Which intrinsic that is. An id no intrinsic claims is dropped rather
    /// than reported.
    pub intrinsic: IntrinsicId,
    /// Whether the path is legal only for a single-row fire: the source's
    /// rows are statically one and the reduction's are symbolic, so a fire
    /// that brought more than one row would fold the wrong extent.
    pub requires_single_row: bool,
}

/// Everything decided about one stage before a line of it is emitted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageFusion {
    /// `(result, source)`: the shell must make `offs[result]` equal
    /// `offs[source]`, in the order given — a later entry may name an earlier
    /// entry's result, so applying them in order resolves a chain.
    pub aliases: Vec<(u32, u32)>,
    /// Nodes the emitter writes no call for, ascending. Every one is a
    /// `reshape` whose result appears in [`Self::aliases`].
    pub elided: Vec<u32>,
    /// The direct-argmax fast paths this stage admits; empty when none do.
    pub direct_argmax: Vec<DirectArgmax>,
}

impl StageFusion {
    /// Whether the emitter should write a call for `node`.
    #[must_use]
    pub fn emits_node(&self, node: usize) -> bool {
        !self.elided.contains(&(node as u32))
    }
}

/// A launch plan's value table in the shape [`covers`] reads.
///
/// `LaunchPlanValue` and `SymbolicType` are the same two fields under two
/// names; converting rather than duplicating `covers` is what keeps one
/// answer to "may this reshape be elided" across the arms.
fn symbolic_types(values: &[LaunchPlanValue]) -> Vec<SymbolicType> {
    values
        .iter()
        .map(|value| SymbolicType {
            dtype: value.dtype,
            dims: value.axes.clone(),
        })
        .collect()
}

/// Rows and width of a value, as the direct-argmax shape test needs them.
///
/// Transcribed from `cuda::fused::row_shape`, whose contract this must match:
/// `fixed_rows` is the static product of every leading axis, `row_extent` is
/// the one symbolic leading axis if there is one, and `width` is the trailing
/// axis, which must be static. `None` where the shape cannot be read that way.
#[derive(Clone, Copy, PartialEq, Eq)]
struct RowShape {
    fixed_rows: u64,
    row_extent: u32,
    width: u32,
}

fn row_shape(dims: &[Dimension]) -> Option<RowShape> {
    let mut shape = RowShape {
        fixed_rows: 1,
        row_extent: u32::MAX,
        width: 1,
    };
    if dims.len() >= 2 {
        for dimension in &dims[..dims.len() - 1] {
            match dimension {
                Dimension::Symbolic(role) => {
                    if shape.row_extent != u32::MAX {
                        return None;
                    }
                    shape.row_extent = *role as u32;
                }
                Dimension::Static(value) => {
                    if *value == 0 || shape.fixed_rows > u64::MAX / u64::from(*value) {
                        return None;
                    }
                    shape.fixed_rows *= u64::from(*value);
                }
            }
        }
    }
    if let Some(last) = dims.last() {
        let Dimension::Static(width) = last else {
            return None;
        };
        if *width == 0 {
            return None;
        }
        shape.width = *width;
    }
    Some(shape)
}

/// Who produces each value, and how many nodes read it.
struct Uses {
    producer: Vec<u32>,
    consumers: Vec<u32>,
}

fn uses(plan: &LaunchStagePlan) -> Uses {
    let count = plan.value_types.len();
    let mut producer = alloc::vec![u32::MAX; count];
    let mut consumers = alloc::vec![0u32; count];
    for (node, op) in plan.ops.iter().enumerate() {
        for result in 0..u32::from(op.result_count) {
            if let Some(slot) = producer.get_mut((op.result_id + result) as usize) {
                *slot = node as u32;
            }
        }
        for &argument in &op.args {
            if let Some(slot) = consumers.get_mut(argument as usize) {
                *slot += 1;
            }
        }
    }
    Uses {
        producer,
        consumers,
    }
}

/// Both analyses, over the shape a shell holds.
///
/// # The elision rule
///
/// A `reshape` is elided when its result covers no more bytes than its source
/// ([`covers`]) and its result is not bound to a channel — a bound value is
/// one the shell may WRITE rather than read, and pointing it at another
/// value's bytes would make that write land somewhere else. Everything else
/// is safe because the shell routes every read through `offs`, so an alias is
/// honoured uniformly wherever the value is read.
///
/// A boundary op's result is never a reshape's result, so no elision can
/// steal a value the shell stages.
#[must_use]
pub fn analyze_stage(plan: &LaunchStagePlan) -> StageFusion {
    let types = symbolic_types(&plan.value_types);
    let uses = uses(plan);
    let bound: Vec<bool> = {
        let mut bound = alloc::vec![false; plan.value_types.len()];
        for &value in &plan.channel_bindings {
            if let Some(slot) = bound.get_mut(value as usize) {
                *slot = true;
            }
        }
        bound
    };

    let mut table = AliasTable::new();
    let mut fusion = StageFusion::default();

    for (node, op) in plan.ops.iter().enumerate() {
        if op.tag != tags::RESHAPE || op.args.is_empty() || op.result_count == 0 {
            continue;
        }
        let result = op.result_id;
        let source = op.args[0];
        if bound.get(result as usize).copied().unwrap_or(true) {
            continue;
        }
        if !covers(&types, source, result) {
            continue;
        }
        table.elide(result, source);
        fusion.aliases.push((result, table.resolve(result)));
        fusion.elided.push(node as u32);
    }

    fusion.direct_argmax = direct_argmax(plan, &uses, &table);
    fusion
}

/// Which `argmax` nodes read a logits intrinsic through nothing but reshapes.
///
/// The walk is CUDA's: from the reduction's operand, back through producers,
/// following a `reshape` only while the value it produced has exactly one
/// consumer and that consumer is the node we came from — a value read twice
/// is a value some other op still needs at its own shape.
fn direct_argmax(plan: &LaunchStagePlan, uses: &Uses, table: &AliasTable) -> Vec<DirectArgmax> {
    let mut found = Vec::new();
    for (node, op) in plan.ops.iter().enumerate() {
        if op.tag != tags::REDUCE_ARGMAX || op.args.is_empty() {
            continue;
        }
        let mut value = op.args[0];
        while let Some(&producer) = uses.producer.get(value as usize) {
            if producer == u32::MAX {
                break;
            }
            // One consumer, and we are standing on a consumer of this value —
            // so the one consumer IS the node we came from, and no other op
            // still wants the value at the shape this link would skip.
            if uses.consumers.get(value as usize).copied().unwrap_or(0) != 1 {
                break;
            }
            let Some(source) = plan.ops.get(producer as usize) else {
                break;
            };
            if source.tag == tags::RESHAPE && !source.args.is_empty() {
                value = source.args[0];
                continue;
            }
            if source.tag != tags::INTRINSIC_VAL {
                break;
            }
            let Some(intrinsic) = source.intrinsic else {
                break;
            };
            let wire = intrinsic as u16;
            if wire != intrinsic_tags::LOGITS && wire != intrinsic_tags::MTP_LOGITS {
                break;
            }
            let source_dims = plan
                .value_types
                .get(source.result_id as usize)
                .map(|value| value.axes.as_slice());
            let reduced_dims = plan
                .value_types
                .get(op.args[0] as usize)
                .map(|value| value.axes.as_slice());
            let (Some(source_dims), Some(reduced_dims)) = (source_dims, reduced_dims) else {
                break;
            };
            let source_shape = row_shape(source_dims);
            let reduced_shape = row_shape(reduced_dims);
            let exact = source_shape.is_some() && source_shape == reduced_shape;
            let single_row = match (source_shape, reduced_shape) {
                (Some(source), Some(target)) => {
                    source.width == target.width
                        && source.fixed_rows == 1
                        && target.fixed_rows == 1
                        && source.row_extent != u32::MAX
                        && target.row_extent == u32::MAX
                }
                _ => false,
            };
            if exact || single_row {
                found.push(DirectArgmax {
                    node: node as u32,
                    // The chain's reshapes are already aliased, so the value
                    // the shell will actually read is the resolved root.
                    source_value: table.resolve(op.args[0]),
                    intrinsic,
                    requires_single_row: single_row,
                });
            }
            break;
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::{DirectArgmax, analyze_stage};
    use crate::codegen::launch::{LaunchOp, LaunchPlanValue, LaunchStagePlan};
    use crate::plan::Dimension;
    use alloc::vec;
    use alloc::vec::Vec;
    use eta_ir::op::{IntrinsicId, tags};
    use eta_ir::types::Dtype;

    fn value(dims: &[Dimension]) -> LaunchPlanValue {
        LaunchPlanValue {
            dtype: Dtype::F32,
            axes: dims.to_vec(),
        }
    }

    fn op(tag: u8, result_id: u32, args: &[u32]) -> LaunchOp {
        LaunchOp {
            tag,
            result_count: 1,
            result_id,
            args: args.to_vec(),
            ..LaunchOp::default()
        }
    }

    fn plan(ops: Vec<LaunchOp>, values: Vec<LaunchPlanValue>) -> LaunchStagePlan {
        LaunchStagePlan {
            ops,
            value_types: values,
            ..LaunchStagePlan::default()
        }
    }

    /// A reshape whose result is no larger than its source and which nothing
    /// binds is elided, and its consumers are told to read the source.
    #[test]
    fn a_plain_reshape_is_elided() {
        let stage = plan(
            vec![
                op(tags::IOTA, 0, &[]),
                op(tags::RESHAPE, 1, &[0]),
                op(tags::EXP, 2, &[1]),
            ],
            vec![
                value(&[Dimension::Static(4), Dimension::Static(8)]),
                value(&[Dimension::Static(32)]),
                value(&[Dimension::Static(32)]),
            ],
        );
        let fusion = analyze_stage(&stage);
        assert_eq!(fusion.elided, [1], "the reshape is the only elided node");
        assert_eq!(fusion.aliases, [(1, 0)], "value 1 reads value 0's bytes");
        assert!(!fusion.emits_node(1));
        assert!(fusion.emits_node(0) && fusion.emits_node(2));
    }

    /// **THE ANALYSIS DECLINING IS AS LOAD-BEARING AS IT FIRING.** A reshape
    /// whose result is WIDER than its source would read past the source's
    /// bytes, and one whose result a channel binds is a value the shell may
    /// write. Neither is elided.
    #[test]
    fn a_reshape_that_does_not_cover_or_that_escapes_is_not_elided() {
        let widening = plan(
            vec![op(tags::IOTA, 0, &[]), op(tags::RESHAPE, 1, &[0])],
            vec![
                value(&[Dimension::Static(8)]),
                value(&[Dimension::Static(32)]),
            ],
        );
        assert!(
            analyze_stage(&widening).elided.is_empty(),
            "a reshape to a wider value must keep its own bytes"
        );

        let mut bound = plan(
            vec![op(tags::IOTA, 0, &[]), op(tags::RESHAPE, 1, &[0])],
            vec![
                value(&[Dimension::Static(32)]),
                value(&[Dimension::Static(32)]),
            ],
        );
        bound.channel_bindings = vec![1];
        assert!(
            analyze_stage(&bound).elided.is_empty(),
            "a bound value is one the shell may write; it keeps its own bytes"
        );
    }

    /// A chain of reshapes resolves to the root, so a consumer three hops
    /// down reads the value the shell actually staged rather than a slot
    /// nothing writes — which is the whole of the "only elide a node whose
    /// value some other emission still produces" invariant.
    #[test]
    fn a_chain_of_reshapes_resolves_to_the_root() {
        let stage = plan(
            vec![
                op(tags::IOTA, 0, &[]),
                op(tags::RESHAPE, 1, &[0]),
                op(tags::RESHAPE, 2, &[1]),
                op(tags::EXP, 3, &[2]),
            ],
            vec![
                value(&[Dimension::Static(32)]),
                value(&[Dimension::Static(32)]),
                value(&[Dimension::Static(32)]),
                value(&[Dimension::Static(32)]),
            ],
        );
        let fusion = analyze_stage(&stage);
        assert_eq!(fusion.elided, [1, 2]);
        assert_eq!(
            fusion.aliases,
            [(1, 0), (2, 0)],
            "both hops name the root, so the shell applies them in any order"
        );
    }

    /// An `argmax` fed by the logits intrinsic through reshapes is reported,
    /// with the root the shell stages as its source.
    #[test]
    fn an_argmax_of_the_logits_is_reported() {
        let mut logits = op(tags::INTRINSIC_VAL, 0, &[]);
        logits.intrinsic = Some(IntrinsicId::Logits);
        let stage = plan(
            vec![
                logits,
                op(tags::RESHAPE, 1, &[0]),
                op(tags::REDUCE_ARGMAX, 2, &[1]),
            ],
            vec![
                value(&[Dimension::Static(128)]),
                value(&[Dimension::Static(128)]),
                value(&[Dimension::Static(1)]),
            ],
        );
        let fusion = analyze_stage(&stage);
        assert_eq!(
            fusion.direct_argmax,
            [DirectArgmax {
                node: 2,
                source_value: 0,
                intrinsic: IntrinsicId::Logits,
                requires_single_row: false,
            }],
            "the argmax reads the staged logits value, not the elided reshape"
        );
    }

    /// **THE DECLINE THAT MATTERS.** An `argmax` fed through anything but a
    /// reshape is not a direct read: the op between it and the intrinsic
    /// changes the values, and answering the argmax off the logits buffer
    /// would ignore that op entirely. A Gumbel sampler is exactly this shape.
    #[test]
    fn an_argmax_through_an_op_that_is_not_a_reshape_is_refused() {
        let mut logits = op(tags::INTRINSIC_VAL, 0, &[]);
        logits.intrinsic = Some(IntrinsicId::Logits);
        let stage = plan(
            vec![
                logits,
                op(tags::EXP, 1, &[0]),
                op(tags::REDUCE_ARGMAX, 2, &[1]),
            ],
            vec![
                value(&[Dimension::Static(128)]),
                value(&[Dimension::Static(128)]),
                value(&[Dimension::Static(1)]),
            ],
        );
        assert!(
            analyze_stage(&stage).direct_argmax.is_empty(),
            "an op between the intrinsic and the argmax must be run"
        );
    }

    /// A value read twice is not a chain: the second reader still needs it at
    /// its own shape, so the reshape stays and the fast path is refused.
    #[test]
    fn a_reshape_read_twice_breaks_the_chain() {
        let mut logits = op(tags::INTRINSIC_VAL, 0, &[]);
        logits.intrinsic = Some(IntrinsicId::Logits);
        let stage = plan(
            vec![
                logits,
                op(tags::RESHAPE, 1, &[0]),
                op(tags::REDUCE_ARGMAX, 2, &[1]),
                op(tags::EXP, 3, &[1]),
            ],
            vec![
                value(&[Dimension::Static(128)]),
                value(&[Dimension::Static(128)]),
                value(&[Dimension::Static(1)]),
                value(&[Dimension::Static(128)]),
            ],
        );
        assert!(
            analyze_stage(&stage).direct_argmax.is_empty(),
            "the reshape's result has two readers; the chain does not hold"
        );
    }
}
