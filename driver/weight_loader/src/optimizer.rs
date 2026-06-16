use crate::error::CompileError;
use crate::ir::{LayoutExpr, LayoutPlan};
use crate::typecheck::typecheck;
use crate::types::{Axis, DType, Encoding, ExprId, TensorDecl};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizerPassStats {
    pub name: String,
    pub exprs_before: usize,
    pub exprs_after: usize,
    pub rewrites: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizerReport {
    pub passes: Vec<OptimizerPassStats>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimizedLayoutPlan {
    pub plan: LayoutPlan,
    pub report: OptimizerReport,
}

pub fn optimize(plan: LayoutPlan) -> Result<LayoutPlan, CompileError> {
    Ok(optimize_with_report(plan)?.plan)
}

pub fn optimize_with_report(mut plan: LayoutPlan) -> Result<OptimizedLayoutPlan, CompileError> {
    let mut report = OptimizerReport::default();
    typecheck(&plan)?;
    let before = plan.exprs.len();
    let output_count_before = plan.outputs.len();
    plan = dedupe_realize_outputs(plan);
    let output_rewrites = output_count_before.saturating_sub(plan.outputs.len());
    plan = normalize_live_plan(&plan)?;
    report.passes.push(OptimizerPassStats {
        name: "live-normalization".to_string(),
        exprs_before: before,
        exprs_after: plan.exprs.len(),
        rewrites: output_rewrites + before.saturating_sub(plan.exprs.len()),
    });
    let mut changed = true;
    let mut iterations = 0;
    while changed {
        iterations += 1;
        if iterations > 32 {
            return Err(CompileError::Internal(
                "layout optimizer did not converge".to_string(),
            ));
        }
        let before = plan.exprs.len();
        changed = false;
        let mut rewrites = 0;
        rewrites += collapse_selects(&mut plan);
        rewrites += cancel_partition_join(&mut plan);
        rewrites += coalesce_reorders(&mut plan);
        rewrites += elide_identity_views(&mut plan);
        changed |= rewrites > 0;
        typecheck(&plan)?;
        if rewrites > 0 {
            report.passes.push(OptimizerPassStats {
                name: format!("fixed-point-normalization-{iterations}"),
                exprs_before: before,
                exprs_after: plan.exprs.len(),
                rewrites,
            });
        }
    }
    Ok(OptimizedLayoutPlan { plan, report })
}

fn dedupe_realize_outputs(mut plan: LayoutPlan) -> LayoutPlan {
    let mut seen = HashSet::new();
    let mut outputs = Vec::with_capacity(plan.outputs.len());
    for output in plan.outputs.iter().rev() {
        let key = match plan.expr(*output) {
            Some(LayoutExpr::Realize { runtime_name, .. }) => runtime_name.clone(),
            _ => format!("expr:{}", output.0),
        };
        if seen.insert(key) {
            outputs.push(*output);
        }
    }
    outputs.reverse();
    plan.outputs = outputs;
    plan
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
        LayoutExpr::Source { .. } | LayoutExpr::ByteSpans { .. } => out.push(expr.clone()),
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
            normalize_cast(plan, *input, *dtype, decl, out, memo)?
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
        } => normalize_encode(plan, *scheme, *input, metadata_outputs, decl, out, memo)?,
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
        LayoutExpr::Repack { input, spec, decl } => {
            let input = normalize_expr(plan, *input, out, memo)?;
            out.push(LayoutExpr::Repack {
                input,
                spec: *spec,
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

fn normalize_cast(
    plan: &LayoutPlan,
    input: ExprId,
    dtype: DType,
    decl: &TensorDecl,
    out: &mut LayoutPlan,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Result<ExprId, CompileError> {
    if let Some(input_decl) = plan.decl(input)
        && input_decl.encoding == Encoding::Raw(dtype)
    {
        return normalize_expr(plan, input, out, memo);
    }
    if let Some(LayoutExpr::Join { inputs, axis, .. }) = plan.expr(input) {
        let casted = normalize_cast_inputs(plan, inputs, dtype, out, memo)?;
        return Ok(out.push(LayoutExpr::Join {
            inputs: casted,
            axis: *axis,
            decl: decl.clone(),
        }));
    }
    if let Some(LayoutExpr::Stack { inputs, axis, .. }) = plan.expr(input) {
        let casted = normalize_cast_inputs(plan, inputs, dtype, out, memo)?;
        return Ok(out.push(LayoutExpr::Stack {
            inputs: casted,
            axis: *axis,
            decl: decl.clone(),
        }));
    }
    if let Some(LayoutExpr::Decode {
        scheme,
        data,
        metadata,
        ..
    }) = plan.expr(input)
    {
        let data = normalize_expr(plan, *data, out, memo)?;
        let metadata = normalize_inputs(plan, metadata, out, memo)?;
        return Ok(out.push(LayoutExpr::Decode {
            scheme: *scheme,
            data,
            metadata,
            decl: decl.clone(),
        }));
    }
    let input = normalize_expr(plan, input, out, memo)?;
    Ok(out.push(LayoutExpr::Cast {
        input,
        dtype,
        decl: decl.clone(),
    }))
}

fn normalize_cast_inputs(
    plan: &LayoutPlan,
    inputs: &[ExprId],
    dtype: DType,
    out: &mut LayoutPlan,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Result<Vec<ExprId>, CompileError> {
    inputs
        .iter()
        .map(|input| {
            let normalized = normalize_expr(plan, *input, out, memo)?;
            let source_decl = plan.decl(*input).ok_or_else(|| {
                CompileError::InvalidInput(format!("Cast input {} has no decl", input.0))
            })?;
            if source_decl.encoding == Encoding::Raw(dtype) {
                return Ok(normalized);
            }
            let mut cast_decl = source_decl.clone();
            cast_decl.name = format!("{}.cast.{dtype:?}", source_decl.name);
            cast_decl.encoding = Encoding::Raw(dtype);
            Ok(out.push(LayoutExpr::Cast {
                input: normalized,
                dtype,
                decl: cast_decl,
            }))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn normalize_select(
    plan: &LayoutPlan,
    input: ExprId,
    axis: Axis,
    start: i64,
    length: i64,
    decl: &TensorDecl,
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
    if let Some(LayoutExpr::Decode {
        scheme,
        data,
        metadata,
        ..
    }) = plan.expr(input)
    {
        return push_select_through_decode(
            plan, *scheme, *data, metadata, axis, start, length, decl, out, memo,
        );
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
    axis: Axis,
    parts: u32,
    index: u32,
    decl: &TensorDecl,
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
    if let Some(LayoutExpr::Join {
        inputs,
        axis: join_axis,
        ..
    }) = plan.expr(input)
        && *join_axis != axis
    {
        let mut partitioned = Vec::with_capacity(inputs.len());
        for input in inputs {
            let input_decl = plan.decl(*input).ok_or_else(|| {
                CompileError::InvalidInput(format!("Join input {} has no decl", input.0))
            })?;
            let axis_index = axis.0 as usize;
            if axis_index >= input_decl.shape.len()
                || input_decl.shape[axis_index] % i64::from(parts) != 0
            {
                return Err(CompileError::InvalidInput(format!(
                    "Partition through Join input {} cannot partition axis {} into {} parts",
                    input.0, axis.0, parts
                )));
            }
            let normalized_input = normalize_expr(plan, *input, out, memo)?;
            let mut part_decl = input_decl.clone();
            part_decl.shape[axis_index] /= i64::from(parts);
            part_decl.sharding = crate::types::Sharding {
                axis: Some(axis),
                world: parts,
                rank: index,
            };
            partitioned.push(out.push(LayoutExpr::Partition {
                input: normalized_input,
                axis,
                parts,
                index,
                decl: part_decl,
            }));
        }
        return Ok(out.push(LayoutExpr::Join {
            inputs: partitioned,
            axis: *join_axis,
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
    axis: Axis,
    start: i64,
    length: i64,
    decl: &TensorDecl,
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

#[allow(clippy::too_many_arguments)]
fn push_select_through_decode(
    plan: &LayoutPlan,
    scheme: crate::types::QuantScheme,
    data: ExprId,
    metadata: &[ExprId],
    axis: Axis,
    start: i64,
    length: i64,
    decl: &TensorDecl,
    out: &mut LayoutPlan,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Result<ExprId, CompileError> {
    let data_decl = plan
        .decl(data)
        .ok_or_else(|| CompileError::InvalidInput(format!("Decode data {} has no decl", data.0)))?;
    let selected_data_decl = select_decl(data_decl, axis, start, length, "select")?;
    let data = normalize_expr(plan, data, out, memo)?;
    let selected_data = out.push(LayoutExpr::Select {
        input: data,
        axis,
        start,
        length,
        decl: selected_data_decl,
    });
    let mut selected_metadata = Vec::with_capacity(metadata.len());
    for meta in metadata {
        let meta_decl = plan.decl(*meta).ok_or_else(|| {
            CompileError::InvalidInput(format!("Decode metadata {} has no decl", meta.0))
        })?;
        let axis_index = axis.0 as usize;
        let tracks_selected_axis = axis_index < meta_decl.shape.len()
            && axis_index < decl.shape.len()
            && axis_index < data_decl.shape.len()
            && meta_decl.shape[axis_index] == data_decl.shape[axis_index]
            && start + length <= meta_decl.shape[axis_index];
        let normalized_meta = normalize_expr(plan, *meta, out, memo)?;
        if tracks_selected_axis {
            selected_metadata.push(out.push(LayoutExpr::Select {
                input: normalized_meta,
                axis,
                start,
                length,
                decl: select_decl(meta_decl, axis, start, length, "select")?,
            }));
        } else {
            selected_metadata.push(normalized_meta);
        }
    }
    Ok(out.push(LayoutExpr::Decode {
        scheme,
        data: selected_data,
        metadata: selected_metadata,
        decl: decl.clone(),
    }))
}

fn normalize_encode(
    plan: &LayoutPlan,
    scheme: crate::types::QuantScheme,
    input: ExprId,
    metadata_outputs: &[TensorDecl],
    decl: &TensorDecl,
    out: &mut LayoutPlan,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Result<ExprId, CompileError> {
    if let Some(LayoutExpr::Decode {
        scheme: from,
        data,
        metadata,
        ..
    }) = plan.expr(input)
    {
        let data = normalize_expr(plan, *data, out, memo)?;
        let metadata = normalize_inputs(plan, metadata, out, memo)?;
        return Ok(out.push(LayoutExpr::Transcode {
            from: *from,
            to: scheme,
            data,
            metadata,
            metadata_outputs: metadata_outputs.to_vec(),
            decl: decl.clone(),
        }));
    }
    if let Some(LayoutExpr::Select {
        input: selected_input,
        axis,
        start,
        length,
        ..
    }) = plan.expr(input)
        && metadata_outputs.is_empty()
        && encode_select_is_aligned(decl, *axis, *start, *length)
    {
        let source_decl = plan.decl(*selected_input).ok_or_else(|| {
            CompileError::InvalidInput(format!("Encode input {} has no decl", selected_input.0))
        })?;
        let mut encoded_source_decl = source_decl.clone();
        encoded_source_decl.name = format!("{}.encoded.{scheme:?}", source_decl.name);
        encoded_source_decl.encoding = decl.encoding.clone();
        let source = normalize_expr(plan, *selected_input, out, memo)?;
        let encoded_source = out.push(LayoutExpr::Encode {
            scheme,
            input: source,
            metadata_outputs: Vec::new(),
            decl: encoded_source_decl,
        });
        return Ok(out.push(LayoutExpr::Select {
            input: encoded_source,
            axis: *axis,
            start: *start,
            length: *length,
            decl: decl.clone(),
        }));
    }
    let input = normalize_expr(plan, input, out, memo)?;
    Ok(out.push(LayoutExpr::Encode {
        scheme,
        input,
        metadata_outputs: metadata_outputs.to_vec(),
        decl: decl.clone(),
    }))
}

fn select_decl(
    source: &TensorDecl,
    axis: Axis,
    start: i64,
    length: i64,
    suffix: &str,
) -> Result<TensorDecl, CompileError> {
    let axis_index = axis.0 as usize;
    if axis_index >= source.shape.len()
        || start < 0
        || length <= 0
        || start + length > source.shape[axis_index]
    {
        return Err(CompileError::InvalidInput(format!(
            "cannot select axis {} range [{}:{}) from {:?}",
            axis.0,
            start,
            start + length,
            source.shape
        )));
    }
    let mut decl = source.clone();
    decl.name = format!("{}.{}", source.name, suffix);
    decl.shape[axis_index] = length;
    Ok(decl)
}

fn encode_select_is_aligned(decl: &TensorDecl, _axis: Axis, start: i64, length: i64) -> bool {
    match &decl.encoding {
        Encoding::Raw(_) => false,
        Encoding::Quant(spec) if spec.normalized_bits() >= 8 => true,
        Encoding::Quant(spec) => {
            let group = i64::from(spec.normalized_group_size());
            start % group == 0 && length % group == 0
        }
    }
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

fn collapse_selects(plan: &mut LayoutPlan) -> usize {
    let mut changed = 0;
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
            changed += 1;
        }
    }
    changed
}

fn cancel_partition_join(plan: &mut LayoutPlan) -> usize {
    let mut changed = 0;
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
            changed += 1;
        }
    }
    changed
}

fn coalesce_reorders(plan: &mut LayoutPlan) -> usize {
    let mut changed = 0;
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
            changed += 1;
        }
    }
    changed
}

fn elide_identity_views(plan: &mut LayoutPlan) -> usize {
    let mut changed = 0;
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
            changed += 1;
        }
    }
    changed
}
