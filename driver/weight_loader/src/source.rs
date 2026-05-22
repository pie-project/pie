use crate::error::CompileError;
use crate::ffi_types::{
    PieLoaderBytes, PieLoaderCheckpointFileView, PieLoaderCheckpointFormat,
    PieLoaderCheckpointTensorView, PieLoaderDType, PieLoaderEncodingKind, PieLoaderI64Slice,
    PieLoaderQuantScheme,
};
use crate::types::{
    CheckpointFormat, DType, Encoding, FileId, Layout, QuantScheme, QuantSpec, TensorId,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointMetadata {
    pub files: Vec<CheckpointFile>,
    pub tensors: Vec<RawTensor>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointFile {
    pub id: FileId,
    pub path: String,
    pub size_bytes: u64,
    pub format: CheckpointFormat,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawTensor {
    pub id: TensorId,
    pub name: String,
    pub file_id: FileId,
    pub file_offset: u64,
    pub span_bytes: u64,
    pub shape: Vec<i64>,
    pub encoding: Encoding,
    pub layout: Layout,
}

impl CheckpointMetadata {
    pub fn tensor(&self, id: TensorId) -> Option<&RawTensor> {
        self.tensors.iter().find(|tensor| tensor.id == id)
    }
}

pub fn files_from_ffi(
    files: &[PieLoaderCheckpointFileView],
) -> Result<Vec<CheckpointFile>, CompileError> {
    let mut out = Vec::with_capacity(files.len());
    for file in files {
        out.push(CheckpointFile {
            id: FileId(file.id),
            path: ffi_string(file.path, "file.path")?,
            size_bytes: file.size_bytes,
            format: checkpoint_format(file.format),
        });
    }
    out.sort_by_key(|file| file.id.0);
    Ok(out)
}

pub fn tensors_from_ffi(
    tensors: &[PieLoaderCheckpointTensorView],
) -> Result<Vec<RawTensor>, CompileError> {
    let mut out = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let shape = ffi_i64_slice(tensor.shape, "tensor.shape")?;
        for dim in &shape {
            if *dim < 0 {
                return Err(CompileError::InvalidInput(format!(
                    "tensor {} has negative dimension {}",
                    tensor.id, dim
                )));
            }
        }
        out.push(RawTensor {
            id: TensorId(tensor.id),
            name: ffi_string(tensor.name, "tensor.name")?,
            file_id: FileId(tensor.file_id),
            file_offset: tensor.file_offset,
            span_bytes: tensor.span_bytes,
            shape,
            encoding: ffi_encoding(
                tensor.encoding_kind,
                tensor.dtype,
                tensor.quant_scheme,
                tensor.quant_bits_per_element,
                tensor.quant_group_size,
                tensor.quant_channel_axis,
                tensor
                    .quant_has_scale_dtype
                    .then_some(tensor.quant_scale_dtype),
                tensor
                    .quant_has_zero_point_dtype
                    .then_some(tensor.quant_zero_point_dtype),
                tensor.quant_block_shape,
            )?,
            layout: Layout::dense(1),
        });
    }
    out.sort_by_key(|tensor| tensor.id.0);
    Ok(out)
}

pub fn ffi_string(bytes: PieLoaderBytes, field: &'static str) -> Result<String, CompileError> {
    if bytes.len == 0 {
        return Ok(String::new());
    }
    if bytes.ptr.is_null() {
        return Err(CompileError::NullArgument(field));
    }
    let slice = unsafe { std::slice::from_raw_parts(bytes.ptr, bytes.len) };
    std::str::from_utf8(slice)
        .map(str::to_owned)
        .map_err(|err| CompileError::InvalidInput(format!("{field} is not UTF-8: {err}")))
}

pub fn ffi_i64_slice(
    values: PieLoaderI64Slice,
    field: &'static str,
) -> Result<Vec<i64>, CompileError> {
    if values.len == 0 {
        return Ok(Vec::new());
    }
    if values.ptr.is_null() {
        return Err(CompileError::NullArgument(field));
    }
    Ok(unsafe { std::slice::from_raw_parts(values.ptr, values.len) }.to_vec())
}

pub fn ffi_dtype(dtype: PieLoaderDType) -> DType {
    match dtype {
        PieLoaderDType::F32 => DType::F32,
        PieLoaderDType::F16 => DType::F16,
        PieLoaderDType::BF16 => DType::BF16,
        PieLoaderDType::F8E4M3 => DType::F8E4M3,
        PieLoaderDType::F8E5M2 => DType::F8E5M2,
        PieLoaderDType::I32 => DType::I32,
        PieLoaderDType::I16 => DType::I16,
        PieLoaderDType::I8 => DType::I8,
        PieLoaderDType::U32 => DType::U32,
        PieLoaderDType::U16 => DType::U16,
        PieLoaderDType::U8 => DType::U8,
        PieLoaderDType::Bool => DType::Bool,
    }
}

pub fn ffi_quant_scheme(scheme: PieLoaderQuantScheme) -> QuantScheme {
    match scheme {
        PieLoaderQuantScheme::None => QuantScheme::None,
        PieLoaderQuantScheme::Fp8E4M3 => QuantScheme::Fp8E4M3,
        PieLoaderQuantScheme::Fp8E5M2 => QuantScheme::Fp8E5M2,
        PieLoaderQuantScheme::Int8Symmetric => QuantScheme::Int8Symmetric,
        PieLoaderQuantScheme::Int8Asymmetric => QuantScheme::Int8Asymmetric,
        PieLoaderQuantScheme::AwqInt4 => QuantScheme::AwqInt4,
        PieLoaderQuantScheme::GptqInt4 => QuantScheme::GptqInt4,
        PieLoaderQuantScheme::Mxfp4E2M1E8M0 => QuantScheme::Mxfp4E2M1E8M0,
        PieLoaderQuantScheme::GgufQ4_0 => QuantScheme::GgufQ4_0,
        PieLoaderQuantScheme::GgufQ4K => QuantScheme::GgufQ4K,
        PieLoaderQuantScheme::GgufQ5_0 => QuantScheme::GgufQ5_0,
        PieLoaderQuantScheme::GgufQ5K => QuantScheme::GgufQ5K,
        PieLoaderQuantScheme::GgufQ8_0 => QuantScheme::GgufQ8_0,
    }
}

fn checkpoint_format(format: PieLoaderCheckpointFormat) -> CheckpointFormat {
    match format {
        PieLoaderCheckpointFormat::Safetensors => CheckpointFormat::Safetensors,
        PieLoaderCheckpointFormat::Gguf => CheckpointFormat::Gguf,
        PieLoaderCheckpointFormat::Unknown => CheckpointFormat::Unknown,
    }
}

fn ffi_encoding(
    kind: PieLoaderEncodingKind,
    dtype: PieLoaderDType,
    scheme: PieLoaderQuantScheme,
    bits_per_element: u8,
    group_size: u32,
    channel_axis: i32,
    scale_dtype: Option<PieLoaderDType>,
    zero_point_dtype: Option<PieLoaderDType>,
    block_shape: PieLoaderI64Slice,
) -> Result<Encoding, CompileError> {
    let dtype = ffi_dtype(dtype);
    match kind {
        PieLoaderEncodingKind::Raw => Ok(Encoding::Raw(dtype)),
        PieLoaderEncodingKind::Quant => Ok(Encoding::Quant(
            QuantSpec {
                scheme: ffi_quant_scheme(scheme),
                logical_dtype: dtype,
                bits_per_element,
                group_size,
                channel_axis: ffi_optional_axis(channel_axis)?,
                scale_dtype: scale_dtype.map(ffi_dtype),
                zero_point_dtype: zero_point_dtype.map(ffi_dtype),
                block_shape: ffi_i64_slice(block_shape, "tensor.quant_block_shape")?,
            }
            .normalized(),
        )),
    }
}

pub fn ffi_optional_axis(axis: i32) -> Result<Option<crate::types::Axis>, CompileError> {
    if axis < 0 {
        return Ok(None);
    }
    Ok(Some(crate::types::Axis(u8::try_from(axis).map_err(
        |_| CompileError::InvalidInput(format!("axis {axis} is out of range")),
    )?)))
}
