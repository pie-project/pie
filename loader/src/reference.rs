use std::collections::HashMap;

use crate::error::CompileError;
use crate::ir::{GatherPiece, LayoutExpr, LayoutPlan};
use crate::typecheck::typecheck;
use crate::types::{Encoding, ExprId, TensorDecl, TensorId, encoding_dense_element_bytes};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorValue {
    pub decl: TensorDecl,
    pub data: Vec<i64>,
}

impl TensorValue {
    pub fn new(decl: TensorDecl, data: Vec<i64>) -> Result<Self, CompileError> {
        let expected = element_count(&decl.shape)?;
        if expected != data.len() {
            return Err(CompileError::InvalidInput(format!(
                "reference tensor '{}' has {} values for shape {:?}",
                decl.name,
                data.len(),
                decl.shape
            )));
        }
        Ok(Self { decl, data })
    }
}

pub fn evaluate(
    plan: &LayoutPlan,
    sources: &HashMap<TensorId, TensorValue>,
) -> Result<HashMap<String, TensorValue>, CompileError> {
    typecheck(plan)?;
    let mut values: HashMap<ExprId, TensorValue> = HashMap::new();
    let mut outputs = HashMap::new();
    for (index, expr) in plan.exprs.iter().enumerate() {
        let id = ExprId(index as u32);
        let value = eval_expr(expr, &values, sources)?;
        if let LayoutExpr::Realize { runtime_name, .. } = expr {
            outputs.insert(runtime_name.clone(), value.clone());
        }
        values.insert(id, value);
    }
    Ok(outputs)
}

fn eval_expr(
    expr: &LayoutExpr,
    values: &HashMap<ExprId, TensorValue>,
    sources: &HashMap<TensorId, TensorValue>,
) -> Result<TensorValue, CompileError> {
    match expr {
        LayoutExpr::Source { tensor, decl } => {
            let mut value = sources.get(tensor).cloned().ok_or_else(|| {
                CompileError::InvalidInput(format!("missing reference source {}", tensor.0))
            })?;
            value.decl = decl.clone();
            Ok(value)
        }
        LayoutExpr::Gather {
            inputs,
            pieces,
            zero_fill,
            decl,
        } => {
            let sources = inputs
                .iter()
                .map(|input| value(values, *input))
                .collect::<Result<Vec<_>, _>>()?;
            gather(&sources, pieces, *zero_fill, decl)
        }
        LayoutExpr::Cast { input, decl, .. } | LayoutExpr::Encode { input, decl, .. } => {
            retag(value(values, *input)?, decl)
        }
        LayoutExpr::Decode { data, decl, .. } | LayoutExpr::Transcode { data, decl, .. } => {
            retag(value(values, *data)?, decl)
        }
        LayoutExpr::Repack { input, decl, .. } => retag(value(values, *input)?, decl),
        LayoutExpr::Realize { input, decl, .. } => retag(value(values, *input)?, decl),
    }
}

fn value(values: &HashMap<ExprId, TensorValue>, id: ExprId) -> Result<&TensorValue, CompileError> {
    values.get(&id).ok_or_else(|| {
        CompileError::InvalidInput(format!("reference expr {} is not available", id.0))
    })
}

fn retag(source: &TensorValue, decl: &TensorDecl) -> Result<TensorValue, CompileError> {
    TensorValue::new(decl.clone(), source.data.clone())
}

/// Replay a compiled affine expression one rectangular copy at a time.
///
/// This is the semantics the storage executors implement, spelled out with no
/// regard for speed: the point is that it is obviously right, so a plan that
/// disagrees with it is wrong. Offsets below the HIGH IR are in bytes, so the
/// evaluator only handles encodings with a whole number of bytes per element —
/// it has one `i64` per element and nowhere to put half of one.
fn gather(
    inputs: &[&TensorValue],
    pieces: &[GatherPiece],
    zero_fill: bool,
    decl: &TensorDecl,
) -> Result<TensorValue, CompileError> {
    let width = element_bytes(&decl.encoding)?;
    let mut out = vec![0i64; element_count(&decl.shape)?];
    let mut written = 0usize;
    for piece in pieces {
        let source = inputs.get(piece.input as usize).ok_or_else(|| {
            CompileError::InvalidInput(format!("reference gather names input {}", piece.input))
        })?;
        if element_bytes(&source.decl.encoding)? != width {
            return Err(CompileError::InvalidInput(
                "reference gather mixes element widths".to_string(),
            ));
        }
        let (inner, outer) = piece.dims.split_last().ok_or_else(|| {
            CompileError::InvalidInput("reference gather piece has no extent".to_string())
        })?;
        if inner.src_stride != 1 || inner.dst_stride != 1 {
            return Err(CompileError::InvalidInput(
                "reference gather piece has no contiguous inner block".to_string(),
            ));
        }
        let run = elements(inner.count, width, "gather run")?;
        let counts = outer.iter().map(|dim| dim.count).collect::<Vec<_>>();
        for index in iterate_indices(&counts)? {
            let mut src = piece.src_offset as i64;
            let mut dst = piece.dst_offset as i64;
            for (at, dim) in outer.iter().enumerate() {
                src += index[at] * dim.src_stride;
                dst += index[at] * dim.dst_stride;
            }
            let src = elements(src, width, "gather source offset")?;
            let dst = elements(dst, width, "gather destination offset")?;
            if src + run > source.data.len() || dst + run > out.len() {
                return Err(CompileError::InvalidInput(
                    "reference gather piece is out of range".to_string(),
                ));
            }
            out[dst..dst + run].copy_from_slice(&source.data[src..src + run]);
            written += run;
        }
    }
    if !zero_fill && written != out.len() {
        return Err(CompileError::InvalidInput(format!(
            "reference gather covers {written} of {} elements with no zero fill",
            out.len()
        )));
    }
    TensorValue::new(decl.clone(), out)
}

fn element_bytes(encoding: &Encoding) -> Result<usize, CompileError> {
    encoding_dense_element_bytes(encoding)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| {
            CompileError::InvalidInput(format!(
                "reference evaluator cannot address {encoding:?} by element"
            ))
        })
}

fn elements(bytes: i64, width: usize, what: &str) -> Result<usize, CompileError> {
    let bytes = usize::try_from(bytes)
        .map_err(|_| CompileError::InvalidInput(format!("negative {what}")))?;
    if bytes % width != 0 {
        return Err(CompileError::InvalidInput(format!(
            "{what} {bytes} does not land on an element boundary"
        )));
    }
    Ok(bytes / width)
}

fn element_count(shape: &[i64]) -> Result<usize, CompileError> {
    shape.iter().try_fold(1usize, |acc, dim| {
        let dim = usize::try_from(*dim).map_err(|_| {
            CompileError::InvalidInput(format!("negative reference dimension {}", dim))
        })?;
        acc.checked_mul(dim).ok_or_else(|| {
            CompileError::InvalidInput("reference element count overflow".to_string())
        })
    })
}

fn iterate_indices(shape: &[i64]) -> Result<Vec<Vec<i64>>, CompileError> {
    let total = element_count(shape)?;
    let mut out = Vec::with_capacity(total);
    for linear in 0..total {
        let mut remaining = linear;
        let mut index = vec![0; shape.len()];
        for axis in (0..shape.len()).rev() {
            let dim = usize::try_from(shape[axis]).map_err(|_| {
                CompileError::InvalidInput(format!("negative reference dimension {}", shape[axis]))
            })?;
            index[axis] = (remaining % dim) as i64;
            remaining /= dim;
        }
        out.push(index);
    }
    Ok(out)
}
