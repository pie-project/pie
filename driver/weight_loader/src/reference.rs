use std::collections::HashMap;

use crate::error::CompileError;
use crate::ir::{LayoutExpr, LayoutPlan};
use crate::typecheck::typecheck;
use crate::types::{Axis, ExprId, TensorDecl, TensorId};

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
        LayoutExpr::ByteSpans { .. } => Err(CompileError::InvalidInput(
            "reference evaluator does not model byte-span assembly".to_string(),
        )),
        LayoutExpr::Select {
            input,
            axis,
            start,
            length,
            decl,
        } => select(value(values, *input)?, *axis, *start, *length, decl),
        LayoutExpr::Partition {
            input,
            axis,
            parts,
            index,
            decl,
        } => {
            let source = value(values, *input)?;
            let axis_len = source.decl.shape[axis.0 as usize];
            let length = axis_len / i64::from(*parts);
            select(source, *axis, i64::from(*index) * length, length, decl)
        }
        LayoutExpr::Join { inputs, axis, decl } => {
            let tensors = inputs
                .iter()
                .map(|input| value(values, *input).cloned())
                .collect::<Result<Vec<_>, _>>()?;
            join(&tensors, *axis, decl)
        }
        LayoutExpr::Stack { inputs, axis, decl } => {
            let tensors = inputs
                .iter()
                .map(|input| value(values, *input).cloned())
                .collect::<Result<Vec<_>, _>>()?;
            stack(&tensors, *axis, decl)
        }
        LayoutExpr::Unzip { input, .. } => Ok(value(values, *input)?.clone()),
        LayoutExpr::Reorder { input, perm, decl } => reorder(value(values, *input)?, perm, decl),
        LayoutExpr::View {
            input,
            axis,
            start,
            length,
            decl,
            ..
        } => {
            let source = value(values, *input)?;
            if let Some(axis) = axis {
                select(source, *axis, *start, *length, decl)
            } else {
                retag(source, decl)
            }
        }
        LayoutExpr::Cast { input, decl, .. } | LayoutExpr::Encode { input, decl, .. } => {
            retag(value(values, *input)?, decl)
        }
        LayoutExpr::Decode { data, decl, .. } | LayoutExpr::Transcode { data, decl, .. } => {
            retag(value(values, *data)?, decl)
        }
        LayoutExpr::Attach { data, decl, .. }
        | LayoutExpr::Realize {
            input: data, decl, ..
        } => retag(value(values, *data)?, decl),
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

fn select(
    source: &TensorValue,
    axis: Axis,
    start: i64,
    length: i64,
    decl: &TensorDecl,
) -> Result<TensorValue, CompileError> {
    let mut out = Vec::with_capacity(element_count(&decl.shape)?);
    for mut index in iterate_indices(&decl.shape)? {
        index[axis.0 as usize] += start;
        out.push(source.data[linear_index(&source.decl.shape, &index)?]);
    }
    if length != decl.shape[axis.0 as usize] {
        return Err(CompileError::InvalidInput(
            "reference select length mismatch".to_string(),
        ));
    }
    TensorValue::new(decl.clone(), out)
}

fn join(
    tensors: &[TensorValue],
    axis: Axis,
    decl: &TensorDecl,
) -> Result<TensorValue, CompileError> {
    let axis = axis.0 as usize;
    let mut out = Vec::with_capacity(element_count(&decl.shape)?);
    for mut out_index in iterate_indices(&decl.shape)? {
        let mut offset = 0;
        let selected = tensors
            .iter()
            .find(|tensor| {
                let len = tensor.decl.shape[axis];
                let hit = out_index[axis] >= offset && out_index[axis] < offset + len;
                if !hit {
                    offset += len;
                }
                hit
            })
            .ok_or_else(|| CompileError::Internal("join index not covered".to_string()))?;
        out_index[axis] -= offset;
        out.push(selected.data[linear_index(&selected.decl.shape, &out_index)?]);
    }
    TensorValue::new(decl.clone(), out)
}

fn stack(
    tensors: &[TensorValue],
    axis: Axis,
    decl: &TensorDecl,
) -> Result<TensorValue, CompileError> {
    let axis = axis.0 as usize;
    let mut out = Vec::with_capacity(element_count(&decl.shape)?);
    for mut out_index in iterate_indices(&decl.shape)? {
        let selected = out_index.remove(axis) as usize;
        out.push(tensors[selected].data[linear_index(&tensors[selected].decl.shape, &out_index)?]);
    }
    TensorValue::new(decl.clone(), out)
}

fn reorder(
    source: &TensorValue,
    perm: &[u8],
    decl: &TensorDecl,
) -> Result<TensorValue, CompileError> {
    let mut out = Vec::with_capacity(element_count(&decl.shape)?);
    for out_index in iterate_indices(&decl.shape)? {
        let mut source_index = vec![0; out_index.len()];
        for (out_axis, source_axis) in perm.iter().enumerate() {
            source_index[*source_axis as usize] = out_index[out_axis];
        }
        out.push(source.data[linear_index(&source.decl.shape, &source_index)?]);
    }
    TensorValue::new(decl.clone(), out)
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

fn linear_index(shape: &[i64], index: &[i64]) -> Result<usize, CompileError> {
    let mut linear = 0usize;
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        let component = usize::try_from(index[axis])
            .map_err(|_| CompileError::InvalidInput("negative reference index".to_string()))?;
        linear = linear
            .checked_add(component.checked_mul(stride).ok_or_else(|| {
                CompileError::InvalidInput("reference linear index overflow".to_string())
            })?)
            .ok_or_else(|| {
                CompileError::InvalidInput("reference linear index overflow".to_string())
            })?;
        stride = stride
            .checked_mul(usize::try_from(shape[axis]).map_err(|_| {
                CompileError::InvalidInput(format!("negative reference dimension {}", shape[axis]))
            })?)
            .ok_or_else(|| CompileError::InvalidInput("reference stride overflow".to_string()))?;
    }
    Ok(linear)
}
