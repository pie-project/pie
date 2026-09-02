use eta_ir::Dtype;
use eta_ir::container::ChanDType;

/// The element type a channel's cells actually hold. `ChanDType::Act` is
/// the late-bound activation type; this host plane materializes it as
/// `F32`, which is what its reference interpreter computes in.
#[must_use]
pub fn concrete_dtype(dtype: ChanDType) -> Dtype {
    match dtype {
        ChanDType::Concrete(dtype) => dtype,
        ChanDType::Act => Dtype::F32,
    }
}

/// What a [`Dtype`] outside ETA's set means to this plane: nothing, and it
/// cannot get here. [`Value`]'s four variants are the four dtypes ETA
/// computes in. Panics rather than substituting `F32`: a plan reaching this
/// point already passed `eta_ir::infer::body_types`, which refuses an
/// unsupported result dtype by name.
///
/// # Panics
///
/// Always.
#[cold]
pub fn no_lane(dtype: Dtype) -> ! {
    panic!("{dtype:?} is not a dtype ETA computes in; this plane has no lane for it")
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    F32(Vec<f32>),

    I32(Vec<i32>),

    U32(Vec<u32>),

    Bool(Vec<u8>),
}

impl Value {
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        match self {
            Value::F32(_) => Dtype::F32,
            Value::I32(_) => Dtype::I32,
            Value::U32(_) => Dtype::U32,
            Value::Bool(_) => Dtype::Bool,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Value::F32(v) => v.len(),
            Value::I32(v) => v.len(),
            Value::U32(v) => v.len(),
            Value::Bool(v) => v.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn zeros(dtype: Dtype, numel: usize) -> Value {
        let n = numel.max(1);
        match dtype {
            Dtype::I32 => Value::I32(vec![0; n]),
            Dtype::U32 => Value::U32(vec![0; n]),
            Dtype::Bool => Value::Bool(vec![0; n]),
            Dtype::F32 => Value::F32(vec![0.0; n]),
            _ => no_lane(dtype),
        }
    }

    #[must_use]
    pub fn lanes_f32(&self) -> Vec<f32> {
        match self {
            Value::F32(v) => v.clone(),
            Value::I32(v) => v.iter().map(|&x| x as f32).collect(),
            Value::U32(v) => v.iter().map(|&x| x as f32).collect(),
            Value::Bool(v) => v.iter().map(|&x| if x != 0 { 1.0 } else { 0.0 }).collect(),
        }
    }

    #[must_use]
    pub fn lanes_i64(&self) -> Vec<i64> {
        match self {
            Value::I32(v) => v.iter().map(|&x| i64::from(x)).collect(),
            Value::U32(v) => v.iter().map(|&x| i64::from(x)).collect(),
            Value::Bool(v) => v.iter().map(|&x| i64::from(x != 0)).collect(),
            Value::F32(v) => v.iter().map(|&x| x as i64).collect(),
        }
    }

    #[must_use]
    pub fn from_i64(dtype: Dtype, x: &[i64]) -> Value {
        match dtype {
            Dtype::U32 => Value::U32(x.iter().map(|&v| v as u32).collect()),
            Dtype::Bool => Value::Bool(x.iter().map(|&v| u8::from(v != 0)).collect()),
            Dtype::F32 => Value::F32(x.iter().map(|&v| v as f32).collect()),
            Dtype::I32 => Value::I32(x.iter().map(|&v| v as i32).collect()),
            _ => no_lane(dtype),
        }
    }
}

#[must_use]
pub fn pick(len: usize, i: usize) -> usize {
    if len == 1 { 0 } else { i }
}

#[must_use]
pub fn value_matches(v: &Value, dtype: ChanDType, dims: &[u32]) -> bool {
    v.dtype() == concrete_dtype(dtype) && v.len() as u64 == crate::shape_numel(dims).max(1)
}

#[must_use]
pub fn wire_cell_bytes(dtype: Dtype, numel: usize) -> usize {
    if dtype == Dtype::Bool {
        numel.div_ceil(8)
    } else {
        numel * 4
    }
}

#[must_use]
pub fn decode_wire(bytes: &[u8], dtype: Dtype, numel: usize) -> Option<Value> {
    if bytes.len() != wire_cell_bytes(dtype, numel) {
        return None;
    }
    Some(match dtype {
        Dtype::Bool => {
            let mut out = vec![0u8; numel];
            for (j, lane) in out.iter_mut().enumerate() {
                *lane = (bytes[j / 8] >> (j % 8)) & 1;
            }
            Value::Bool(out)
        }
        Dtype::I32 => Value::I32(
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        Dtype::U32 => Value::U32(
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        Dtype::F32 => Value::F32(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        // a dtype this plane has no lane for; see `no_lane`.
        _ => return None,
    })
}

pub fn encode_wire(v: &Value, dst: &mut [u8]) {
    match v {
        Value::Bool(b) => {
            let packed = b.len().div_ceil(8);
            for byte in &mut dst[..packed] {
                *byte = 0;
            }
            for (j, &lane) in b.iter().enumerate() {
                if lane != 0 {
                    dst[j / 8] |= 1u8 << (j % 8);
                }
            }
        }
        Value::I32(v) => copy_le(v.iter().flat_map(|x| x.to_le_bytes()), dst),
        Value::U32(v) => copy_le(v.iter().flat_map(|x| x.to_le_bytes()), dst),
        Value::F32(v) => copy_le(v.iter().flat_map(|x| x.to_le_bytes()), dst),
    }
}

fn copy_le(bytes: impl Iterator<Item = u8>, dst: &mut [u8]) {
    for (slot, byte) in dst.iter_mut().zip(bytes) {
        *slot = byte;
    }
}
