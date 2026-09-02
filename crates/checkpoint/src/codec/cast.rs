//! Between a dtype's bytes and `f64`. Every element passes through `f64`, which holds every value of every dtype here exactly (including `I32`/`U32`, unlike `f32`), so a cast is lossless except where the destination is lossy.

use half::{bf16, f16};

use crate::error::Error;
use crate::types::DType;

use super::invalid;

pub fn decode_values(bytes: &[u8], dtype: DType) -> Result<Vec<f64>, Error> {
    // asked before the width is: a quantization term has no element width, and chunks_exact(0) panics rather than refusing.
    if dtype.elem().is_none() {
        return Err(invalid(
            "host Cast does not implement a quantization term: a block decodes \
             through its own scheme, not element by element",
        ));
    }
    let width = dtype.bytes_ceil() as usize;
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
                DType::Bf16 => {
                    bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32() as f64
                }
                DType::I32 => i32::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::I16 => i16::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::I8 => i8::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::U32 => u32::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::U16 => u16::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::U8 | DType::Bool => chunk[0] as f64,
                DType::E8m0 => (chunk[0] as f64 - 127.0).exp2(),
                DType::E4m3 | DType::E5m2 => {
                    return Err(invalid("host Cast does not implement FP8"));
                }
                // I64/U64 tensors are index tables that move byte-for-byte, not through the f64 pivot.
                DType::I64 | DType::U64 => {
                    return Err(invalid("host Cast does not implement 64-bit integers"));
                }
                // sub-byte codes have no element to chunk on; packing/unpacking is codec::mxfp4's, not a cast's.
                DType::E2m1
                | DType::Mxfp4
                | DType::U4g64
                | DType::U4g32
                | DType::U4g64tiled
                | DType::U2g32
                | DType::U2g64
                | DType::Nvfp4
                | DType::U2g16k
                | DType::I3g16k
                | DType::U4g32k
                | DType::U5g32k
                | DType::I6g16k
                | DType::E4m3row
                | DType::E4m3tile128
                | DType::U2g128 => {
                    return Err(invalid("host Cast does not implement the sub-byte codes"));
                }
                // an U8g64 byte is an affine code, meaningless without its group's scale and offset.
                DType::U8g64 => {
                    return Err(invalid(
                        "host Cast does not implement affine codes: an U8g64 byte means \
                         nothing without its group's scale and offset",
                    ));
                }
            })
        })
        .collect()
}

pub fn encode_values(values: &[f64], dtype: DType) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(values.len() * dtype.bytes_ceil() as usize);
    for &value in values {
        match dtype {
            DType::F32 => out.extend_from_slice(&(value as f32).to_le_bytes()),
            DType::F16 => {
                out.extend_from_slice(&f16::from_f32(value as f32).to_bits().to_le_bytes())
            }
            DType::Bf16 => {
                out.extend_from_slice(&bf16::from_f32(value as f32).to_bits().to_le_bytes())
            }
            DType::I32 => out.extend_from_slice(&(value as i32).to_le_bytes()),
            DType::I16 => out.extend_from_slice(&(value as i16).to_le_bytes()),
            DType::I8 => out.push(value as i8 as u8),
            DType::U32 => out.extend_from_slice(&(value as u32).to_le_bytes()),
            DType::U16 => out.extend_from_slice(&(value as u16).to_le_bytes()),
            DType::U8 => out.push(value as u8),
            DType::Bool => out.push(u8::from(value != 0.0)),
            DType::E8m0 => {
                return Err(invalid("host Cast does not encode to E8M0"));
            }
            DType::E4m3 | DType::E5m2 => {
                return Err(invalid("host Cast does not implement FP8"));
            }
            // sub-byte code is packed, not cast; see decode_values.
            DType::E2m1
            | DType::Mxfp4
            | DType::U4g64
            | DType::U4g32
            | DType::U4g64tiled
            | DType::U2g32
            | DType::U2g64
            | DType::Nvfp4
            | DType::U2g16k
            | DType::I3g16k
            | DType::U4g32k
            | DType::U5g32k
            | DType::I6g16k
            | DType::E4m3row
            | DType::E4m3tile128
            | DType::U2g128 => {
                return Err(invalid("host Cast does not implement the sub-byte codes"));
            }
            // affine code is quantized, not cast — see decode_values.
            DType::U8g64 => {
                return Err(invalid(
                    "host Cast does not encode to affine codes: choosing an U8g64 byte \
                     is a quantization, which picks the group's scale and offset too",
                ));
            }
            DType::I64 | DType::U64 => {
                return Err(invalid("host Cast does not implement 64-bit integers"));
            }
        }
    }
    Ok(out)
}

/// A `Cast`, dispatched on its dtype pair once and run across every core, threaded for large inputs.
/// The float pairs are bit-identical to the generic `decode_values`/`encode_values` pivot they replace (`f32 -> f64 -> f32` is the identity); every other pair still goes through the pivot, one chunk at a time.
pub fn cast_elements(bytes: &[u8], from: DType, to: DType) -> Result<Vec<u8>, Error> {
    let in_width = from.bytes_ceil() as usize;
    let out_width = to.bytes_ceil() as usize;
    if in_width == 0 || !bytes.len().is_multiple_of(in_width) {
        return Err(invalid("cast input byte count is not element-aligned"));
    }
    let elements = bytes.len() / in_width;
    let mut out = vec![0u8; elements * out_width];
    // below about a megabyte the threads cost more than they carry.
    let workers = if bytes.len() < (1 << 20) || elements == 0 {
        1
    } else {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(elements)
    };
    if workers <= 1 {
        return cast_chunk(bytes, &mut out, from, to).map(|()| out);
    }
    let per_worker = elements.div_ceil(workers);
    let failure = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        let mut out_rest = &mut out[..];
        let mut start = 0usize;
        while start < elements {
            let count = per_worker.min(elements - start);
            let (chunk, rest) = std::mem::take(&mut out_rest).split_at_mut(count * out_width);
            out_rest = rest;
            let source = &bytes[start * in_width..(start + count) * in_width];
            handles.push(scope.spawn(move || cast_chunk(source, chunk, from, to)));
            start += count;
        }
        handles
            .into_iter()
            .filter_map(|handle| handle.join().ok()?.err())
            .next()
    });
    match failure {
        Some(error) => Err(error),
        None => Ok(out),
    }
}

/// One chunk of a cast: `src.len() / from.bytes_ceil()` elements, converted into `dst`.
/// The match is on the pair, once, so each arm is a loop over two known widths the compiler can vectorize; arms cover the float conversions the loader performs, everything else keeps the general implementation.
fn cast_chunk(src: &[u8], dst: &mut [u8], from: DType, to: DType) -> Result<(), Error> {
    use DType::{Bf16, F16, F32};
    match (from, to) {
        (F32, Bf16) => map_elements(src, dst, |v: [u8; 4]| {
            bf16::from_f32(f32::from_le_bytes(v))
                .to_bits()
                .to_le_bytes()
        }),
        (F32, F16) => map_elements(src, dst, |v: [u8; 4]| {
            f16::from_f32(f32::from_le_bytes(v)).to_bits().to_le_bytes()
        }),
        (F16, Bf16) => map_elements(src, dst, |v: [u8; 2]| {
            bf16::from_f32(f16::from_bits(u16::from_le_bytes(v)).to_f32())
                .to_bits()
                .to_le_bytes()
        }),
        (Bf16, F16) => map_elements(src, dst, |v: [u8; 2]| {
            f16::from_f32(bf16::from_bits(u16::from_le_bytes(v)).to_f32())
                .to_bits()
                .to_le_bytes()
        }),
        (F16, F32) => map_elements(src, dst, |v: [u8; 2]| {
            f16::from_bits(u16::from_le_bytes(v)).to_f32().to_le_bytes()
        }),
        (Bf16, F32) => map_elements(src, dst, |v: [u8; 2]| {
            bf16::from_bits(u16::from_le_bytes(v))
                .to_f32()
                .to_le_bytes()
        }),
        _ => {
            let values = decode_values(src, from)?;
            let bytes = encode_values(&values, to)?;
            if bytes.len() != dst.len() {
                return Err(invalid(format!(
                    "cast produced {} bytes for a {}-byte chunk",
                    bytes.len(),
                    dst.len()
                )));
            }
            dst.copy_from_slice(&bytes);
            Ok(())
        }
    }
}

/// Element-wise `[u8; IN] -> [u8; OUT]` over two slices, with the widths in
/// the type so the loop carries neither a stride nor a bounds check.
fn map_elements<const IN: usize, const OUT: usize>(
    src: &[u8],
    dst: &mut [u8],
    convert: impl Fn([u8; IN]) -> [u8; OUT],
) -> Result<(), Error> {
    if !src.len().is_multiple_of(IN) || src.len() / IN * OUT != dst.len() {
        return Err(invalid("cast chunk widths do not match its buffers"));
    }
    for (input, output) in src.chunks_exact(IN).zip(dst.chunks_exact_mut(OUT)) {
        output.copy_from_slice(&convert(input.try_into().unwrap()));
    }
    Ok(())
}

