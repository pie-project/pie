//! Between a dtype's bytes and `f64`.
//!
//! Every element in this module passes through `f64`, which holds every
//! value of every dtype here exactly — including `I32` and `U32`, which
//! `f32` does not. That is what makes a cast lossless except where the
//! DESTINATION is lossy, and the rounding is then the destination's own.

use half::{bf16, f16};

use crate::error::Error;
use crate::types::DType;

use super::invalid;

pub fn decode_values(bytes: &[u8], dtype: DType) -> Result<Vec<f64>, Error> {
    let width = dtype.bytes() as usize;
    if !bytes.len().is_multiple_of(width) {
        return Err(invalid("cast input byte count is not element-aligned"));
    }
    bytes
        .chunks_exact(width)
        .map(|chunk| {
            Ok(match dtype {
                DType::F32 => f32::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::F16 => {
                    f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32() as f64
                }
                DType::BF16 => {
                    bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32() as f64
                }
                DType::I32 => i32::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::I16 => i16::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::I8 => i8::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::U32 => u32::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::U16 => u16::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::U8 | DType::Bool => chunk[0] as f64,
                DType::E8M0 => (chunk[0] as f64 - 127.0).exp2(),
                DType::F8E4M3 | DType::F8E5M2 => {
                    return Err(invalid("host Cast does not implement FP8"));
                }
                // A 64-bit integer does not survive the f64 pivot this cast
                // is written around, and nothing asks it to: `I64`/`U64`
                // tensors are index tables that move byte-for-byte.
                DType::I64 | DType::U64 => {
                    return Err(invalid("host Cast does not implement 64-bit integers"));
                }
            })
        })
        .collect()
}

pub fn encode_values(values: &[f64], dtype: DType) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(values.len() * dtype.bytes() as usize);
    for &value in values {
        match dtype {
            DType::F32 => out.extend_from_slice(&(value as f32).to_le_bytes()),
            DType::F16 => {
                out.extend_from_slice(&f16::from_f32(value as f32).to_bits().to_le_bytes())
            }
            DType::BF16 => {
                out.extend_from_slice(&bf16::from_f32(value as f32).to_bits().to_le_bytes())
            }
            DType::I32 => out.extend_from_slice(&(value as i32).to_le_bytes()),
            DType::I16 => out.extend_from_slice(&(value as i16).to_le_bytes()),
            DType::I8 => out.push(value as i8 as u8),
            DType::U32 => out.extend_from_slice(&(value as u32).to_le_bytes()),
            DType::U16 => out.extend_from_slice(&(value as u16).to_le_bytes()),
            DType::U8 => out.push(value as u8),
            DType::Bool => out.push(u8::from(value != 0.0)),
            DType::E8M0 => {
                return Err(invalid("host Cast does not encode to E8M0"));
            }
            DType::F8E4M3 | DType::F8E5M2 => {
                return Err(invalid("host Cast does not implement FP8"));
            }
            DType::I64 | DType::U64 => {
                return Err(invalid("host Cast does not implement 64-bit integers"));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DType;

    #[test]
    fn half_casts_round_and_overflow_to_infinity() {
        let f16_bytes = encode_values(&[100_000.0], DType::F16).unwrap();
        let f16_value = f16::from_bits(u16::from_le_bytes(f16_bytes.try_into().unwrap()));
        assert!(f16_value.is_infinite() && !f16_value.is_nan());

        let input = f32::from_bits(0x3f80_8001);
        let bf16_bytes = encode_values(&[f64::from(input)], DType::BF16).unwrap();
        let actual = u16::from_le_bytes(bf16_bytes.try_into().unwrap());
        assert_eq!(actual, bf16::from_f32(input).to_bits());
    }
}
