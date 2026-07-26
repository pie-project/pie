//! Rewrites over the HIGH IR.
//!
//! There used to be a great deal more here: pushing selects through joins,
//! cancelling a partition against the join beneath it, collapsing chains of
//! selects, coalescing reorders, eliding views that changed nothing — all run
//! to a fixed point, with a convergence guard in case they fought each other.
//!
//! Every one of those rules existed to undo structure the frontend had just
//! built. Now the frontend builds none: `contract::compile` resolves the whole
//! affine fragment symbolically before an IR node is ever pushed, so there is
//! no select to push, no partition to cancel, and no identity view to elide.
//!
//! What remains are the two rewrites that are about *encoding*, which the
//! affine algebra deliberately does not model, plus dead-code elimination.

use crate::error::CompileError;
use crate::ir::{LayoutExpr, LayoutPlan};
use crate::typecheck::typecheck;
use crate::types::{DType, Encoding, ExprId, TensorDecl};
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
    let outputs_before = plan.outputs.len();
    plan = dedupe_realize_outputs(plan);
    let output_rewrites = outputs_before.saturating_sub(plan.outputs.len());
    plan = normalize_live_plan(&plan)?;
    report.passes.push(OptimizerPassStats {
        name: "live-normalization".to_string(),
        exprs_before: before,
        exprs_after: plan.exprs.len(),
        rewrites: output_rewrites + before.saturating_sub(plan.exprs.len()),
    });
    Ok(OptimizedLayoutPlan { plan, report })
}

/// Keep the last declaration of each runtime name.
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

/// Rebuild the plan from its outputs, dropping anything unreachable.
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
        LayoutExpr::Gather {
            inputs,
            pieces,
            zero_fill,
            decl,
        } => {
            let inputs = normalize_inputs(plan, inputs, out, memo)?;
            out.push(LayoutExpr::Gather {
                inputs,
                pieces: pieces.clone(),
                zero_fill: *zero_fill,
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

/// A cast to the encoding the value already has is nothing, and a cast over a
/// decode is the decode: decoding already names its output dtype.
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
        dtype,
        input,
        decl: decl.clone(),
    }))
}

/// Encoding what was just decoded is a transcode, and the transcode kernel can
/// do it without ever materializing the dequantized tensor.
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
    let input = normalize_expr(plan, input, out, memo)?;
    Ok(out.push(LayoutExpr::Encode {
        scheme,
        input,
        metadata_outputs: metadata_outputs.to_vec(),
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
