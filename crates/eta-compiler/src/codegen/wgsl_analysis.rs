use alloc::vec::Vec;

use eta_ir::op::{IntrinsicId, intrinsic_tags, tags};

use crate::codegen::alias::{AliasTable, covers};
use crate::codegen::launch::{LaunchPlanValue, LaunchStagePlan};
use crate::plan::{Dimension, SymbolicType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectArgmax {
    pub node: u32,
    pub source_value: u32,
    pub intrinsic: IntrinsicId,
    pub requires_single_row: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageFusion {
    pub aliases: Vec<(u32, u32)>,
    pub elided: Vec<u32>,
    pub direct_argmax: Vec<DirectArgmax>,
}

impl StageFusion {
    #[must_use]
    pub fn emits_node(&self, node: usize) -> bool {
        !self.elided.contains(&(node as u32))
    }
}

fn symbolic_types(values: &[LaunchPlanValue]) -> Vec<SymbolicType> {
    values
        .iter()
        .map(|value| SymbolicType {
            dtype: value.dtype,
            dims: value.axes.clone(),
        })
        .collect()
}

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

    #[test]
    fn wgsl_analysis_every_case() {
        a_plain_reshape_is_elided();
        a_reshape_that_does_not_cover_or_that_escapes_is_not_elided();
        a_chain_of_reshapes_resolves_to_the_root();
        an_argmax_of_the_logits_is_reported();
        an_argmax_through_an_op_that_is_not_a_reshape_is_refused();
        a_reshape_read_twice_breaks_the_chain();
    }

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
