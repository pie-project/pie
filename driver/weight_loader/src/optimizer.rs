use crate::error::CompileError;
use crate::ir::{LayoutExpr, LayoutPlan};
use crate::typecheck::typecheck;
use crate::types::ExprId;
use std::collections::HashMap;

pub fn optimize(mut plan: LayoutPlan) -> Result<LayoutPlan, CompileError> {
    typecheck(&plan)?;
    plan = normalize_live_plan(&plan)?;
    let mut changed = true;
    let mut iterations = 0;
    while changed {
        iterations += 1;
        if iterations > 32 {
            return Err(CompileError::Internal(
                "layout optimizer did not converge".to_string(),
            ));
        }
        changed = false;
        changed |= collapse_selects(&mut plan);
        changed |= cancel_partition_join(&mut plan);
        changed |= coalesce_reorders(&mut plan);
        changed |= elide_identity_views(&mut plan);
        typecheck(&plan)?;
    }
    Ok(plan)
}

fn normalize_live_plan(plan: &LayoutPlan) -> Result<LayoutPlan, CompileError> {
    let mut out = LayoutPlan::new();
    let mut memo = HashMap::new();
    for output in &plan.outputs {
        let new_output = normalize_expr(plan, *output, &mut out, &mut memo)?;
        out.outputs.push(new_output);
    }
    typecheck(&out)?;
    Ok(out)
}

fn normalize_expr(
    plan: &LayoutPlan,
    id: ExprId,
    out: &mut LayoutPlan,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Result<ExprId, CompileError> {
    if let Some(existing) = memo.get(&id) {
        return Ok(*existing);
    }
    let expr = plan.expr(id).ok_or_else(|| {
        CompileError::InvalidInput(format!("optimizer expr {} is out of range", id.0))
    })?;
    let new_id = match expr {
        LayoutExpr::Source { .. } => out.push(expr.clone()),
        LayoutExpr::Select {
            input,
            axis,
            start,
            length,
            decl,
        } => normalize_select(plan, *input, *axis, *start, *length, decl, out, memo)?,
        LayoutExpr::Partition {
            input,
            axis,
            parts,
            index,
            decl,
        } => normalize_partition(plan, *input, *axis, *parts, *index, decl, out, memo)?,
        LayoutExpr::Join { inputs, axis, decl } => {
            let inputs = normalize_inputs(plan, inputs, out, memo)?;
            out.push(LayoutExpr::Join {
                inputs,
                axis: *axis,
                decl: decl.clone(),
            })
        }
        LayoutExpr::Stack { inputs, axis, decl } => {
            let inputs = normalize_inputs(plan, inputs, out, memo)?;
            out.push(LayoutExpr::Stack {
                inputs,
                axis: *axis,
                decl: decl.clone(),
            })
        }
        LayoutExpr::Unzip {
            input,
            axis,
            outputs,
        } => {
            let input = normalize_expr(plan, *input, out, memo)?;
            out.push(LayoutExpr::Unzip {
                input,
                axis: *axis,
                outputs: outputs.clone(),
            })
        }
        LayoutExpr::Reorder { input, perm, decl } => {
            let input = normalize_expr(plan, *input, out, memo)?;
            out.push(LayoutExpr::Reorder {
                input,
                perm: perm.clone(),
                decl: decl.clone(),
            })
        }
        LayoutExpr::View {
            input,
            layout,
            axis,
            start,
            length,
            decl,
        } => {
            let input = normalize_expr(plan, *input, out, memo)?;
            out.push(LayoutExpr::View {
                input,
                layout: layout.clone(),
                axis: *axis,
                start: *start,
                length: *length,
                decl: decl.clone(),
            })
        }
        LayoutExpr::Cast { input, dtype, decl } => {
            let input = normalize_expr(plan, *input, out, memo)?;
            out.push(LayoutExpr::Cast {
                input,
                dtype: *dtype,
                decl: decl.clone(),
            })
        }
        LayoutExpr::Decode {
            scheme,
            data,
            metadata,
            decl,
        } => {
            let data = normalize_expr(plan, *data, out, memo)?;
            let metadata = normalize_inputs(plan, metadata, out, memo)?;
            out.push(LayoutExpr::Decode {
                scheme: *scheme,
                data,
                metadata,
                decl: decl.clone(),
            })
        }
        LayoutExpr::Encode {
            scheme,
            input,
            metadata_outputs,
            decl,
        } => {
            let input = normalize_expr(plan, *input, out, memo)?;
            out.push(LayoutExpr::Encode {
                scheme: *scheme,
                input,
                metadata_outputs: metadata_outputs.clone(),
                decl: decl.clone(),
            })
        }
        LayoutExpr::Transcode {
            from,
            to,
            data,
            metadata,
            metadata_outputs,
            decl,
        } => {
            let data = normalize_expr(plan, *data, out, memo)?;
            let metadata = normalize_inputs(plan, metadata, out, memo)?;
            out.push(LayoutExpr::Transcode {
                from: *from,
                to: *to,
                data,
                metadata,
                metadata_outputs: metadata_outputs.clone(),
                decl: decl.clone(),
            })
        }
        LayoutExpr::Attach {
            data,
            metadata,
            decl,
        } => {
            let data = normalize_expr(plan, *data, out, memo)?;
            let metadata = normalize_inputs(plan, metadata, out, memo)?;
            out.push(LayoutExpr::Attach {
                data,
                metadata,
                decl: decl.clone(),
            })
        }
        LayoutExpr::Realize {
            input,
            runtime_name,
            decl,
        } => {
            let input = normalize_expr(plan, *input, out, memo)?;
            out.push(LayoutExpr::Realize {
                input,
                runtime_name: runtime_name.clone(),
                decl: decl.clone(),
            })
        }
    };
    memo.insert(id, new_id);
    Ok(new_id)
}

#[allow(clippy::too_many_arguments)]
fn normalize_select(
    plan: &LayoutPlan,
    input: ExprId,
    axis: crate::types::Axis,
    start: i64,
    length: i64,
    decl: &crate::types::TensorDecl,
    out: &mut LayoutPlan,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Result<ExprId, CompileError> {
    if let Some(LayoutExpr::Select {
        input: parent,
        axis: parent_axis,
        start: parent_start,
        ..
    }) = plan.expr(input)
        && *parent_axis == axis
    {
        return normalize_select(
            plan,
            *parent,
            axis,
            parent_start + start,
            length,
            decl,
            out,
            memo,
        );
    }
    if let Some(LayoutExpr::Join {
        inputs,
        axis: join_axis,
        ..
    }) = plan.expr(input)
        && *join_axis == axis
    {
        return push_select_through_join(plan, inputs, axis, start, length, decl, out, memo);
    }
    let input = normalize_expr(plan, input, out, memo)?;
    Ok(out.push(LayoutExpr::Select {
        input,
        axis,
        start,
        length,
        decl: decl.clone(),
    }))
}

#[allow(clippy::too_many_arguments)]
fn normalize_partition(
    plan: &LayoutPlan,
    input: ExprId,
    axis: crate::types::Axis,
    parts: u32,
    index: u32,
    decl: &crate::types::TensorDecl,
    out: &mut LayoutPlan,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Result<ExprId, CompileError> {
    if let Some(LayoutExpr::Join {
        inputs,
        axis: join_axis,
        ..
    }) = plan.expr(input)
        && *join_axis == axis
        && inputs.len() == parts as usize
    {
        let selected = inputs[index as usize];
        let selected = normalize_expr(plan, selected, out, memo)?;
        return Ok(out.push(LayoutExpr::View {
            input: selected,
            layout: decl.layout.clone(),
            axis: None,
            start: 0,
            length: 0,
            decl: decl.clone(),
        }));
    }
    let input = normalize_expr(plan, input, out, memo)?;
    Ok(out.push(LayoutExpr::Partition {
        input,
        axis,
        parts,
        index,
        decl: decl.clone(),
    }))
}

#[allow(clippy::too_many_arguments)]
fn push_select_through_join(
    plan: &LayoutPlan,
    inputs: &[ExprId],
    axis: crate::types::Axis,
    start: i64,
    length: i64,
    decl: &crate::types::TensorDecl,
    out: &mut LayoutPlan,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Result<ExprId, CompileError> {
    let end = start + length;
    let mut cursor = 0;
    let mut selected = Vec::new();
    for input in inputs {
        let input_decl = plan.decl(*input).ok_or_else(|| {
            CompileError::InvalidInput(format!("Join input {} has no decl", input.0))
        })?;
        let axis_len = input_decl.shape[axis.0 as usize];
        let input_start = cursor;
        let input_end = cursor + axis_len;
        let overlap_start = start.max(input_start);
        let overlap_end = end.min(input_end);
        if overlap_start < overlap_end {
            let normalized_input = normalize_expr(plan, *input, out, memo)?;
            if overlap_start == input_start && overlap_end == input_end {
                selected.push(normalized_input);
            } else {
                let mut slice_decl = input_decl.clone();
                slice_decl.shape[axis.0 as usize] = overlap_end - overlap_start;
                selected.push(out.push(LayoutExpr::Select {
                    input: normalized_input,
                    axis,
                    start: overlap_start - input_start,
                    length: overlap_end - overlap_start,
                    decl: slice_decl,
                }));
            }
        }
        cursor = input_end;
    }
    if selected.is_empty() {
        return Err(CompileError::InvalidInput(
            "Select through Join produced no inputs".to_string(),
        ));
    }
    if selected.len() == 1 {
        return Ok(selected[0]);
    }
    Ok(out.push(LayoutExpr::Join {
        inputs: selected,
        axis,
        decl: decl.clone(),
    }))
}

fn normalize_inputs(
    plan: &LayoutPlan,
    inputs: &[ExprId],
    out: &mut LayoutPlan,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Result<Vec<ExprId>, CompileError> {
    inputs
        .iter()
        .map(|input| normalize_expr(plan, *input, out, memo))
        .collect()
}

fn collapse_selects(plan: &mut LayoutPlan) -> bool {
    let mut changed = false;
    for id in 0..plan.exprs.len() {
        let replacement = match &plan.exprs[id] {
            LayoutExpr::Select {
                input,
                axis,
                start,
                length,
                decl,
            } => match plan.expr(*input) {
                Some(LayoutExpr::Select {
                    input: parent,
                    axis: parent_axis,
                    start: parent_start,
                    ..
                }) if axis == parent_axis => Some(LayoutExpr::Select {
                    input: *parent,
                    axis: *axis,
                    start: parent_start + start,
                    length: *length,
                    decl: decl.clone(),
                }),
                _ => None,
            },
            _ => None,
        };
        if let Some(expr) = replacement {
            plan.exprs[id] = expr;
            changed = true;
        }
    }
    changed
}

fn cancel_partition_join(plan: &mut LayoutPlan) -> bool {
    let mut changed = false;
    for id in 0..plan.exprs.len() {
        let replacement = match &plan.exprs[id] {
            LayoutExpr::Partition {
                input,
                axis,
                parts,
                index,
                decl,
            } => match plan.expr(*input) {
                Some(LayoutExpr::Join {
                    inputs,
                    axis: join_axis,
                    ..
                }) if axis == join_axis && *parts as usize == inputs.len() => inputs
                    .get(*index as usize)
                    .map(|selected| LayoutExpr::View {
                        input: *selected,
                        layout: decl.layout.clone(),
                        axis: None,
                        start: 0,
                        length: 0,
                        decl: decl.clone(),
                    }),
                _ => None,
            },
            _ => None,
        };
        if let Some(expr) = replacement {
            plan.exprs[id] = expr;
            changed = true;
        }
    }
    changed
}

fn coalesce_reorders(plan: &mut LayoutPlan) -> bool {
    let mut changed = false;
    for id in 0..plan.exprs.len() {
        let replacement = match &plan.exprs[id] {
            LayoutExpr::Reorder { input, perm, decl } => match plan.expr(*input) {
                Some(LayoutExpr::Reorder {
                    input: parent,
                    perm: parent_perm,
                    ..
                }) if parent_perm.len() == perm.len() => {
                    let mut composed = Vec::with_capacity(perm.len());
                    for axis in perm {
                        composed.push(parent_perm[*axis as usize]);
                    }
                    Some(LayoutExpr::Reorder {
                        input: *parent,
                        perm: composed,
                        decl: decl.clone(),
                    })
                }
                _ => None,
            },
            _ => None,
        };
        if let Some(expr) = replacement {
            plan.exprs[id] = expr;
            changed = true;
        }
    }
    changed
}

fn elide_identity_views(plan: &mut LayoutPlan) -> bool {
    let mut changed = false;
    for id in 0..plan.exprs.len() {
        let replacement = match &plan.exprs[id] {
            LayoutExpr::View {
                input,
                layout,
                axis,
                decl,
                ..
            } => match plan.decl(*input) {
                Some(input_decl)
                    if axis.is_none()
                        && input_decl.layout == *layout
                        && input_decl.shape == decl.shape =>
                {
                    Some(LayoutExpr::Attach {
                        data: *input,
                        metadata: Vec::new(),
                        decl: decl.clone(),
                    })
                }
                _ => None,
            },
            _ => None,
        };
        if let Some(expr) = replacement {
            plan.exprs[id] = expr;
            changed = true;
        }
    }
    changed
}
