use eta_ir::Dtype;
use eta_ir::container::ChanDType;

#[must_use]
pub fn concrete_dtype(dtype: ChanDType) -> Dtype {
    match dtype {
        ChanDType::Concrete(dtype) => dtype,
        ChanDType::Act => Dtype::F32,
    }
}

#[cold]
pub fn no_lane(dtype: Dtype) -> ! {
    panic!("{dtype:?} is not a dtype ETA computes in; this plane has no lane for it")
}

/// A value as the interpreter holds it: eta-compiler's, whose `eval_op`
/// evaluates every op here.
pub use eta_compiler::eval::interp::Value;

#[must_use]
pub fn value_matches(v: &Value, dtype: ChanDType, dims: &[u32]) -> bool {
    v.dtype() == concrete_dtype(dtype) && v.len() as u64 == crate::shape_numel(dims).max(1)
}

#[must_use]
pub fn wire_cell_bytes(dtype: Dtype, numel: usize) -> usize {
    match dtype {
        Dtype::Bool => numel.div_ceil(8),
        Dtype::Bf16 => numel * 2,
        _ => numel * 4,
    }
}

#[must_use]
pub fn decode_wire(bytes: &[u8], dtype: Dtype, numel: usize) -> Option<Value> {
    if bytes.len() != wire_cell_bytes(dtype, numel) {
        return None;
    }
    Some(match dtype {
        Dtype::Bool => Value::Bool(
            (0..numel)
                .map(|j| (bytes[j / 8] >> (j % 8)) & 1 != 0)
                .collect(),
        ),
        Dtype::I32 => Value::I32(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        Dtype::U32 => Value::U32(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        Dtype::F32 => Value::F32(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        Dtype::Bf16 => Value::Bf16(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect(),
        ),
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
                if lane {
                    dst[j / 8] |= 1u8 << (j % 8);
                }
            }
        }
        Value::I32(v) => copy_le(v.iter().flat_map(|x| x.to_le_bytes()), dst),
        Value::U32(v) => copy_le(v.iter().flat_map(|x| x.to_le_bytes()), dst),
        Value::F32(v) => copy_le(v.iter().flat_map(|x| x.to_le_bytes()), dst),
        Value::Bf16(v) => copy_le(v.iter().flat_map(|x| x.to_le_bytes()), dst),
    }
}

fn copy_le(bytes: impl Iterator<Item = u8>, dst: &mut [u8]) {
    for (slot, byte) in dst.iter_mut().zip(bytes) {
        *slot = byte;
    }
}
