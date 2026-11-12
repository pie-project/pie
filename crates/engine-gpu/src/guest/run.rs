//! What a guest pass's run shares: the readouts it is offered, and values
//! to and from the session's heap.

use eta_compiler::codegen::launch::{LaunchPackage, LaunchValue, ValueOrigin};
use eta_exec::Value;
use eta_ir::Dtype;

use crate::backend::Backend;
use crate::error::{Fault, Result};

pub struct Readout<'a, B: Backend> {
    /// The output it holds.
    pub output: &'a str,

    pub seat: &'a B::Buffer,

    pub at: u64,

    pub width: u32,

    /// Bf16 values widen; f32 and i32 copy as they are.
    pub dtype: Dtype,
}

impl<B: Backend> Clone for Readout<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: Backend> Copy for Readout<'_, B> {}

pub fn local_of(package: &LaunchPackage, at: usize, global: u32) -> Result<Option<u32>> {
    match package.plan_local(global) {
        Some((stage, local)) if stage == at => Ok(local),
        Some((stage, _)) => Err(Fault::Program {
            at: "guest::run",
            why: format!("value {global} is stage {stage}'s and this is stage {at}"),
        }),
        None => Err(Fault::Program {
            at: "guest::run",
            why: format!("value {global} belongs to no stage of this package"),
        }),
    }
}

pub fn root_bytes(root: &LaunchValue, value: &Value, span: u64) -> Vec<u8> {
    let bytes = to_heap(value);
    if root.source != ValueOrigin::Const {
        return bytes;
    }
    let lane = if matches!(value, Value::Bool(_)) {
        1
    } else {
        4
    };
    let span = usize::try_from(span).unwrap_or(0);
    if bytes.len() != lane || span <= lane || !span.is_multiple_of(lane) {
        return bytes;
    }
    bytes.repeat(span / lane)
}

pub fn to_heap(value: &Value) -> Vec<u8> {
    match value {
        Value::F32(cells) => cells.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Value::I32(cells) => cells.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Value::U32(cells) => cells.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Value::Bool(cells) => cells.iter().map(|&b| u8::from(b != 0)).collect(),
    }
}

pub fn from_heap(raw: &[u8], dtype: Dtype, held: u64) -> Value {
    let word = |i: usize| {
        let at = i * 4;
        u32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]])
    };
    match dtype {
        Dtype::Bool => Value::Bool(raw.iter().map(|&b| u8::from(b != 0)).collect()),
        Dtype::I32 => Value::I32((0..(held / 4) as usize).map(|i| word(i) as i32).collect()),
        Dtype::U32 => Value::U32((0..(held / 4) as usize).map(word).collect()),
        _ => Value::F32(
            (0..(held / 4) as usize)
                .map(|i| f32::from_bits(word(i)))
                .collect(),
        ),
    }
}
