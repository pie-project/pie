//! Native Rust GGUF header parser for load planning.
//!
//! Ports the checkpoint-parsing half of the C++ `GgufCheckpointSource`
//! (`driver/cuda/src/loader/gguf_source.cpp`) to Rust. Like the safetensors
//! header parser, it reads **only** the GGUF framing — magic, version, the
//! metadata key/value table, and the tensor-info table — and computes each
//! tensor's absolute file offset and byte span. The bulk tensor payloads are
//! never read; the C++ reference slurps the whole file, but here we stream the
//! header region and bounds-check against the on-disk size, so the runtime
//! never maps weight bytes.
//!
//! GGUF is inherently block-quantized. Dense GGML types (`F32`/`F16`/`BF16`/
//! `I8`/`I32`) become [`Encoding::Raw`]; `Q4_0` becomes an
//! [`Encoding::Quant`] with [`QuantScheme::GgufQ4_0`] (32-element / 18-byte
//! blocks). The remaining GGUF block schemes (`Q4K`/`Q5`/`Q8`) exist in the
//! [`QuantScheme`] enum but, matching the C++ loader, are not yet mapped here.
//!
//! GAP: there is no GGUF checkpoint on the verification box, so this parser is
//! exercised only by synthetic fixtures (unit tests) — it is **not**
//! device-proven end to end. See `plan.md` §"RUNTIME REFACTOR".

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::CompileError;
use crate::source::{CheckpointFile, CheckpointMetadata, RawTensor};
use crate::types::{
    Axis, CheckpointFormat, DType, Encoding, FileId, Layout, QuantScheme, QuantSpec, TensorId,
};

/// GGUF metadata value type tags (`gguf_source.cpp` `GgufValueType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GgufValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

impl GgufValueType {
    fn from_u32(v: u32) -> Result<Self, CompileError> {
        Ok(match v {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            other => {
                return Err(CompileError::InvalidInput(format!(
                    "gguf: unknown metadata value type {other}"
                )));
            }
        })
    }
}

/// GGML tensor type ids we recognise (`gguf_source.cpp` `GgmlType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GgmlType {
    F32,
    F16,
    Q4_0,
    I8,
    I32,
    I64,
    Bf16,
}

impl GgmlType {
    fn from_u32(raw: u32, tensor_name: &str) -> Result<Self, CompileError> {
        Ok(match raw {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            24 => Self::I8,
            26 => Self::I32,
            27 => Self::I64,
            30 => Self::Bf16,
            other => {
                return Err(CompileError::InvalidInput(format!(
                    "gguf: tensor '{tensor_name}' uses unsupported GGUF/GGML type id {other}. \
                     Add a GGUF quant dialect adapter before loading this type."
                )));
            }
        })
    }
}

/// A resolved GGML tensor type: its loader [`Encoding`] plus, for dense types,
/// the storage byte width, or for block-quant types the (elements, bytes) block
/// geometry used to size the packed payload.
struct GgufTensorType {
    encoding: Encoding,
    /// Storage bytes per logical element for dense types (0 for block-quant).
    bytes_per_element: u64,
    /// Logical elements per quant block (0 for dense types).
    block_elements: u64,
    /// Packed storage bytes per quant block (0 for dense types).
    block_bytes: u64,
}

fn map_tensor_type(ty: GgmlType, tensor_name: &str) -> Result<GgufTensorType, CompileError> {
    Ok(match ty {
        GgmlType::F32 => dense(DType::F32, 4),
        GgmlType::F16 => dense(DType::F16, 2),
        GgmlType::Bf16 => dense(DType::BF16, 2),
        GgmlType::I8 => dense(DType::I8, 1),
        GgmlType::I32 => dense(DType::I32, 4),
        // The Rust loader dtype set has no 64-bit integer; GGUF `I64` tensors
        // are metadata-only (token types), never weights. Reject with the same
        // 64-bit rationale as the safetensors parser.
        GgmlType::I64 => {
            return Err(CompileError::InvalidInput(format!(
                "gguf: tensor '{tensor_name}' is I64 (64-bit), unsupported by the loader dtype set"
            )));
        }
        // Q4_0: 32 logical elements packed into an 18-byte block (one F16 scale
        // + 16 bytes of 4-bit weights). Stored as U8 bytes.
        GgmlType::Q4_0 => GgufTensorType {
            encoding: Encoding::Quant(
                QuantSpec {
                    scheme: QuantScheme::GgufQ4_0,
                    logical_dtype: DType::BF16,
                    bits_per_element: 4,
                    group_size: 32,
                    channel_axis: Some(Axis(0)),
                    scale_dtype: Some(DType::F16),
                    zero_point_dtype: None,
                    block_shape: vec![32],
                }
                .normalized(),
            ),
            bytes_per_element: 0,
            block_elements: 32,
            block_bytes: 18,
        },
    })
}

fn dense(dtype: DType, bytes_per_element: u64) -> GgufTensorType {
    GgufTensorType {
        encoding: Encoding::Raw(dtype),
        bytes_per_element,
        block_elements: 0,
        block_bytes: 0,
    }
}

/// A bounds-checked, seekable reader over the GGUF header region. Reads are
/// sequential little-endian; `skip` uses `Seek` so we never buffer skipped
/// metadata payloads.
struct GgufReader {
    inner: BufReader<File>,
    len: u64,
    pos: u64,
}

impl GgufReader {
    fn open(path: &Path) -> Result<Self, CompileError> {
        let file = File::open(path).map_err(|err| {
            CompileError::InvalidInput(format!("gguf: cannot open {}: {err}", path.display()))
        })?;
        let len = file
            .metadata()
            .map_err(|err| {
                CompileError::InvalidInput(format!("gguf: cannot stat {}: {err}", path.display()))
            })?
            .len();
        Ok(Self {
            inner: BufReader::new(file),
            len,
            pos: 0,
        })
    }

    fn require(&self, bytes: u64, context: &str) -> Result<(), CompileError> {
        if bytes > self.len || self.pos > self.len - bytes {
            return Err(CompileError::InvalidInput(format!(
                "gguf: truncated file while reading {context}"
            )));
        }
        Ok(())
    }

    fn read_exact(&mut self, buf: &mut [u8], context: &str) -> Result<(), CompileError> {
        self.require(buf.len() as u64, context)?;
        self.inner.read_exact(buf).map_err(|err| {
            CompileError::InvalidInput(format!("gguf: read failed for {context}: {err}"))
        })?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn read_u32(&mut self, context: &str) -> Result<u32, CompileError> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b, context)?;
        Ok(u32::from_le_bytes(b))
    }

    fn read_u64(&mut self, context: &str) -> Result<u64, CompileError> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b, context)?;
        Ok(u64::from_le_bytes(b))
    }

    fn read_string(&mut self, context: &str) -> Result<String, CompileError> {
        let len = self.read_u64(context)?;
        let len = usize::try_from(len).map_err(|_| {
            CompileError::InvalidInput(format!("gguf: string too large while reading {context}"))
        })?;
        self.require(len as u64, context)?;
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf).map_err(|err| {
            CompileError::InvalidInput(format!("gguf: read failed for {context}: {err}"))
        })?;
        self.pos += len as u64;
        String::from_utf8(buf)
            .map_err(|_| CompileError::InvalidInput(format!("gguf: {context} is not valid UTF-8")))
    }

    fn skip(&mut self, bytes: u64, context: &str) -> Result<(), CompileError> {
        self.require(bytes, context)?;
        self.inner
            .seek(SeekFrom::Current(bytes as i64))
            .map_err(|err| {
                CompileError::InvalidInput(format!("gguf: seek failed for {context}: {err}"))
            })?;
        self.pos += bytes;
        Ok(())
    }
}

fn skip_value(r: &mut GgufReader, ty: GgufValueType) -> Result<(), CompileError> {
    match ty {
        GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool => {
            r.skip(1, "metadata scalar")
        }
        GgufValueType::Uint16 | GgufValueType::Int16 => r.skip(2, "metadata scalar"),
        GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => {
            r.skip(4, "metadata scalar")
        }
        GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64 => {
            r.skip(8, "metadata scalar")
        }
        GgufValueType::String => r.read_string("metadata string").map(|_| ()),
        GgufValueType::Array => skip_array(r),
    }
}

fn skip_array(r: &mut GgufReader) -> Result<(), CompileError> {
    let item_type = GgufValueType::from_u32(r.read_u32("metadata array type")?)?;
    let count = r.read_u64("metadata array length")?;
    for _ in 0..count {
        skip_value(r, item_type)?;
    }
    Ok(())
}

fn read_alignment(r: &mut GgufReader, ty: GgufValueType) -> Result<u64, CompileError> {
    match ty {
        GgufValueType::Uint32 => Ok(u64::from(r.read_u32("general.alignment")?)),
        GgufValueType::Uint64 => r.read_u64("general.alignment"),
        _ => Err(CompileError::InvalidInput(
            "gguf: general.alignment must be uint32 or uint64".to_string(),
        )),
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, CompileError> {
    if alignment == 0 {
        return Err(CompileError::InvalidInput(
            "gguf: alignment must be non-zero".to_string(),
        ));
    }
    let rem = value % alignment;
    Ok(if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    })
}

fn checked_mul(a: u64, b: u64, tensor_name: &str) -> Result<u64, CompileError> {
    a.checked_mul(b).ok_or_else(|| {
        CompileError::InvalidInput(format!(
            "gguf: tensor byte size overflows for '{tensor_name}'"
        ))
    })
}

fn numel_for_shape(shape: &[i64], tensor_name: &str) -> Result<u64, CompileError> {
    let mut out: u64 = 1;
    for &dim in shape {
        if dim < 0 {
            return Err(CompileError::InvalidInput(format!(
                "gguf: negative dimension for '{tensor_name}'"
            )));
        }
        out = checked_mul(out, dim as u64, tensor_name)?;
    }
    Ok(out)
}

/// Default GGUF tensor-data alignment when `general.alignment` is absent
/// (`gguf_source.cpp` initializes `alignment_` to 32).
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

struct PendingTensor {
    name: String,
    shape: Vec<i64>,
    encoding: Encoding,
    relative_offset: u64,
    nbytes: u64,
}

/// Parse a single GGUF checkpoint file's header into a [`CheckpointMetadata`].
/// Only the header region is read; tensor payloads are never mapped.
pub fn parse_gguf_checkpoint(path: &Path) -> Result<CheckpointMetadata, CompileError> {
    let mut r = GgufReader::open(path)?;

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic, "magic")?;
    if &magic != b"GGUF" {
        return Err(CompileError::InvalidInput(format!(
            "gguf: {} is not a GGUF file",
            path.display()
        )));
    }
    let version = r.read_u32("version")?;
    if version != 2 && version != 3 {
        return Err(CompileError::InvalidInput(format!(
            "gguf: unsupported GGUF version {version}"
        )));
    }

    let tensor_count = r.read_u64("tensor count")?;
    let metadata_count = r.read_u64("metadata count")?;

    let mut alignment = GGUF_DEFAULT_ALIGNMENT;
    for _ in 0..metadata_count {
        let key = r.read_string("metadata key")?;
        let ty = GgufValueType::from_u32(r.read_u32("metadata type")?)?;
        if key == "general.alignment" {
            alignment = read_alignment(&mut r, ty)?;
        } else {
            skip_value(&mut r, ty)?;
        }
    }

    let mut pending: Vec<PendingTensor> = Vec::with_capacity(
        usize::try_from(tensor_count)
            .map_err(|_| CompileError::InvalidInput("gguf: tensor count too large".to_string()))?,
    );
    for _ in 0..tensor_count {
        let name = r.read_string("tensor name")?;
        let dim_count = r.read_u32("tensor dimension count")?;
        if dim_count > 16 {
            return Err(CompileError::InvalidInput(format!(
                "gguf: tensor '{name}' has unreasonable rank {dim_count}"
            )));
        }
        let mut shape = Vec::with_capacity(dim_count as usize);
        for _ in 0..dim_count {
            let dim = r.read_u64("tensor dimension")?;
            if dim > i64::MAX as u64 {
                return Err(CompileError::InvalidInput(format!(
                    "gguf: dimension too large for '{name}'"
                )));
            }
            shape.push(dim as i64);
        }
        let raw_type = r.read_u32("tensor type")?;
        let ty = map_tensor_type(GgmlType::from_u32(raw_type, &name)?, &name)?;
        let relative_offset = r.read_u64("tensor data offset")?;

        let logical_elements = numel_for_shape(&shape, &name)?;
        let nbytes = if ty.block_elements != 0 {
            if logical_elements % ty.block_elements != 0 {
                return Err(CompileError::InvalidInput(format!(
                    "gguf: quantized tensor '{name}' element count {logical_elements} is not \
                     divisible by block size {}",
                    ty.block_elements
                )));
            }
            checked_mul(logical_elements / ty.block_elements, ty.block_bytes, &name)?
        } else {
            checked_mul(logical_elements, ty.bytes_per_element, &name)?
        };

        pending.push(PendingTensor {
            name,
            shape,
            encoding: ty.encoding,
            relative_offset,
            nbytes,
        });
    }

    let data_base = align_up(r.pos, alignment)?;
    if data_base > r.len {
        return Err(CompileError::InvalidInput(
            "gguf: tensor data section is missing".to_string(),
        ));
    }

    // Deterministic ordering by name mirrors the C++ loader's sorted name list.
    pending.sort_by(|a, b| a.name.cmp(&b.name));

    let mut tensors = Vec::with_capacity(pending.len());
    for (index, t) in pending.into_iter().enumerate() {
        let absolute_offset = data_base.checked_add(t.relative_offset).ok_or_else(|| {
            CompileError::InvalidInput(format!("gguf: tensor '{}' offset overflows", t.name))
        })?;
        if absolute_offset > r.len || t.nbytes > r.len - absolute_offset {
            return Err(CompileError::InvalidInput(format!(
                "gguf: tensor '{}' points outside the file",
                t.name
            )));
        }
        tensors.push(RawTensor {
            id: TensorId(index as u32),
            name: t.name,
            file_id: FileId(0),
            file_offset: absolute_offset,
            span_bytes: t.nbytes,
            shape: t.shape,
            encoding: t.encoding,
            layout: Layout::dense(1),
        });
    }

    Ok(CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: path.to_string_lossy().into_owned(),
            size_bytes: r.len,
            format: CheckpointFormat::Gguf,
        }],
        tensors,
    })
}

/// Decode one GGUF `Q4_0` block (18 bytes → 32 f32 values), ported from the C++
/// `decode_gguf_q4_0_block`. Kept for parity testing of the block geometry; the
/// runtime materialization is the driver's job.
pub fn decode_gguf_q4_0_block(block: &[u8]) -> Result<Vec<f32>, CompileError> {
    if block.len() != 18 {
        return Err(CompileError::InvalidInput(
            "gguf: Q4_0 block decode expects exactly 18 bytes".to_string(),
        ));
    }
    let scale = half_to_float(u16::from_le_bytes([block[0], block[1]]));
    let mut values = vec![0.0f32; 32];
    for i in 0..16 {
        let packed = block[2 + i];
        let lo = (packed & 0x0f) as i32 - 8;
        let hi = ((packed >> 4) & 0x0f) as i32 - 8;
        values[i] = scale * lo as f32;
        values[i + 16] = scale * hi as f32;
    }
    Ok(values)
}

/// IEEE-754 half → f32 (`gguf_source.cpp` `half_to_float`).
fn half_to_float(half: u16) -> f32 {
    let sign = ((half >> 15) & 0x1) as u32;
    let exp = ((half >> 10) & 0x1f) as u32;
    let frac = (half & 0x3ff) as u32;
    let magnitude = if exp == 0 {
        if frac == 0 {
            0.0
        } else {
            (frac as f32 / 1024.0) * 2f32.powi(-14)
        }
    } else if exp == 31 {
        if frac == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + frac as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -magnitude } else { magnitude }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal single-tensor GGUF v3 file in memory.
    fn build_gguf(
        tensor_name: &str,
        ggml_type: u32,
        shape: &[u64],
        payload: &[u8],
        alignment: Option<u64>,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        let meta_count = if alignment.is_some() { 1u64 } else { 0 };
        out.extend_from_slice(&meta_count.to_le_bytes());
        if let Some(align) = alignment {
            let key = b"general.alignment";
            out.extend_from_slice(&(key.len() as u64).to_le_bytes());
            out.extend_from_slice(key);
            out.extend_from_slice(&4u32.to_le_bytes()); // Uint32
            out.extend_from_slice(&(align as u32).to_le_bytes());
        }
        // tensor info
        out.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        out.extend_from_slice(tensor_name.as_bytes());
        out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &d in shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&ggml_type.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // relative offset
        // align data section to `alignment` (default 32)
        let align = alignment.unwrap_or(GGUF_DEFAULT_ALIGNMENT) as usize;
        while out.len() % align != 0 {
            out.push(0);
        }
        out.extend_from_slice(payload);
        out
    }

    fn write_tmp(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "load_planner_gguf_{tag}_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn parses_dense_f32_tensor() {
        // 2x3 F32 = 24 bytes.
        let payload = vec![0u8; 24];
        let bytes = build_gguf("w", 0, &[2, 3], &payload, None);
        let path = write_tmp("dense", &bytes);
        let meta = parse_gguf_checkpoint(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(meta.files.len(), 1);
        assert_eq!(meta.files[0].format, CheckpointFormat::Gguf);
        assert_eq!(meta.tensors.len(), 1);
        let t = &meta.tensors[0];
        assert_eq!(t.name, "w");
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.encoding, Encoding::Raw(DType::F32));
        assert_eq!(t.span_bytes, 24);
    }

    #[test]
    fn parses_q4_0_block_quant_tensor() {
        // 32 logical elements → one 18-byte Q4_0 block.
        let payload = vec![0u8; 18];
        let bytes = build_gguf("blk", 2, &[32], &payload, None);
        let path = write_tmp("q40", &bytes);
        let meta = parse_gguf_checkpoint(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let t = &meta.tensors[0];
        assert_eq!(t.span_bytes, 18);
        match &t.encoding {
            Encoding::Quant(spec) => {
                assert_eq!(spec.scheme, QuantScheme::GgufQ4_0);
                assert_eq!(spec.group_size, 32);
                assert_eq!(spec.bits_per_element, 4);
            }
            other => panic!("expected Quant, got {other:?}"),
        }
    }

    #[test]
    fn rejects_q4_0_non_divisible_elements() {
        // 33 elements is not a multiple of the 32-element Q4_0 block.
        let payload = vec![0u8; 36];
        let bytes = build_gguf("blk", 2, &[33], &payload, None);
        let path = write_tmp("q40bad", &bytes);
        let err = parse_gguf_checkpoint(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(format!("{err}").contains("divisible"), "got: {err}");
    }

    #[test]
    fn rejects_unsupported_ggml_type() {
        let bytes = build_gguf("x", 99, &[4], &vec![0u8; 16], None);
        let path = write_tmp("badtype", &bytes);
        let err = parse_gguf_checkpoint(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(format!("{err}").contains("unsupported"), "got: {err}");
    }

    #[test]
    fn rejects_non_gguf_magic() {
        let path = write_tmp("nomagic", b"NOPExxxxxxxx");
        let err = parse_gguf_checkpoint(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(format!("{err}").contains("not a GGUF"), "got: {err}");
    }

    #[test]
    fn q4_0_block_decode_matches_reference() {
        // scale = 1.0 (half 0x3C00), all nibbles = 8 → (8-8)*scale = 0.
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C;
        for b in block.iter_mut().skip(2) {
            *b = 0x88; // both nibbles = 8
        }
        let values = decode_gguf_q4_0_block(&block).unwrap();
        assert_eq!(values.len(), 32);
        assert!(values.iter().all(|&v| v == 0.0));

        // nibble 0xF → (15-8)=7, scale 2.0 (half 0x4000) → 14.0
        let mut block2 = vec![0u8; 18];
        block2[0] = 0x00;
        block2[1] = 0x40; // 2.0
        block2[2] = 0xFF; // lo=15, hi=15
        let v2 = decode_gguf_q4_0_block(&block2).unwrap();
        assert_eq!(v2[0], 14.0);
        assert_eq!(v2[16], 14.0);
    }

    #[test]
    fn half_to_float_specials() {
        assert_eq!(half_to_float(0x3C00), 1.0);
        assert_eq!(half_to_float(0x4000), 2.0);
        assert_eq!(half_to_float(0x0000), 0.0);
        assert!(half_to_float(0x7C00).is_infinite());
    }
}
