use serde::{Deserialize, Serialize};

use crate::term::gguf_name;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FileId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InstrId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Axis(pub u8);

pub use dtype::Dtype as DType;

#[must_use]
pub fn is_block_scaled(dtype: DType) -> bool {
    matches!(dtype, DType::E4m3 | DType::E5m2)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointFormat {
    Safetensors,
    Gguf,
    Unknown,
    Zt,
    Npz,
    Pt,
    Hdf5,
    Onnx,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    Cuda,
    Metal,
    Vulkan,
    Wgpu,
    Xla,
    Cerebras,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantScheme {
    None,
    Fp8E4M3,
    Fp8E5M2,
    Int8Symmetric,
    Int8Asymmetric,
    AwqInt4,
    GptqInt4,
    Mxfp4E2M1E8M0,
    MlxAffineU4,
    GgufQ4_0,
    GgufQ2K,
    GgufQ3K,
    GgufQ4K,
    GgufQ5_0,
    GgufQ5K,
    GgufQ8_0,
    Int4B8,
    GgufQ6K,
    GgufQ4_1,
    GgufQ5_1,
    GgufIq4Nl,
    GgufIq4Xs,
    GgufMxfp4,
    GgufIq2Xxs,
    GgufIq2Xs,
    GgufIq2S,
    GgufIq3Xxs,
    GgufIq3S,
}

impl QuantScheme {
    pub fn default_bits(self) -> u8 {
        match self {
            Self::AwqInt4
            | Self::GptqInt4
            | Self::Mxfp4E2M1E8M0
            | Self::MlxAffineU4
            | Self::GgufQ4_0
            | Self::GgufQ4_1
            | Self::GgufQ4K
            | Self::GgufIq4Nl
            | Self::GgufIq4Xs
            | Self::GgufMxfp4
            | Self::Int4B8 => 4,
            Self::GgufQ2K | Self::GgufIq2Xxs | Self::GgufIq2Xs | Self::GgufIq2S => 2,
            Self::GgufQ3K | Self::GgufIq3Xxs | Self::GgufIq3S => 3,
            Self::GgufQ5_0 | Self::GgufQ5_1 | Self::GgufQ5K => 5,
            Self::GgufQ6K => 6,
            Self::Fp8E4M3
            | Self::Fp8E5M2
            | Self::Int8Symmetric
            | Self::Int8Asymmetric
            | Self::GgufQ8_0
            | Self::None => 8,
        }
    }

    pub fn block_layout(self) -> Option<(u64, u64)> {
        let row = ztensor::vocab::gguf::row_of(gguf_name(self)?)?;
        Some((row.elems_per_block, row.block_bytes))
    }

    #[must_use]
    pub fn is_self_contained(self) -> bool {
        gguf_name(self).is_some()
    }

    pub fn default_group_size(self) -> u32 {
        match self {
            Self::AwqInt4 | Self::GptqInt4 | Self::Mxfp4E2M1E8M0 | Self::Int4B8 => 32,
            Self::MlxAffineU4 => 64,
            Self::GgufQ4_0
            | Self::GgufQ4_1
            | Self::GgufQ4K
            | Self::GgufQ5_0
            | Self::GgufQ5_1
            | Self::GgufQ5K
            | Self::GgufIq4Nl
            | Self::GgufIq4Xs
            | Self::GgufMxfp4 => 32,
            Self::GgufQ2K | Self::GgufQ3K | Self::GgufQ6K => 16,
            Self::GgufIq2Xxs
            | Self::GgufIq2Xs
            | Self::GgufIq2S
            | Self::GgufIq3Xxs
            | Self::GgufIq3S => 32,
            Self::Fp8E4M3
            | Self::Fp8E5M2
            | Self::Int8Symmetric
            | Self::Int8Asymmetric
            | Self::GgufQ8_0
            | Self::None => 1,
        }
    }
}

pub use dtype::{TILED_BAND, TILED_STEP};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RepackLayout {
    MarlinMxfp4Weight,
    MarlinMxfp4Scale,
    TiledAffineU4Weight,
    TiledAffineFactor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RepackSpec {
    pub layout: RepackLayout,
    pub batch: u32,
    pub source_rows: u32,
    pub target_rows: u32,
    pub source_cols: u32,
    pub target_cols: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantSpec {
    pub scheme: QuantScheme,
    pub logical_dtype: DType,
    pub bits_per_element: u8,
    pub group_size: u32,
    pub channel_axis: Option<Axis>,
}

impl QuantSpec {
    pub fn normalized(mut self) -> Self {
        if self.bits_per_element == 0 {
            self.bits_per_element = self.scheme.default_bits();
        }
        if self.group_size == 0 {
            self.group_size = self.scheme.default_group_size();
        }
        self
    }

    pub fn dense_element_bytes(&self) -> Option<u64> {
        if self.scheme.is_self_contained() {
            return None;
        }
        let bits = self.normalized_bits();
        if bits.is_multiple_of(8) {
            Some(u64::from(bits / 8))
        } else {
            None
        }
    }

    pub fn block_layout(&self) -> Option<(u64, u64)> {
        self.scheme.block_layout()
    }

    pub fn normalized_bits(&self) -> u8 {
        if self.bits_per_element == 0 {
            self.scheme.default_bits()
        } else {
            self.bits_per_element
        }
    }

    pub fn normalized_group_size(&self) -> u32 {
        if self.group_size == 0 {
            self.scheme.default_group_size()
        } else {
            self.group_size
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Encoding {
    Raw(DType),
    Quant(QuantSpec),
}

impl Encoding {
    pub fn dtype(&self) -> DType {
        match self {
            Encoding::Raw(dtype) => *dtype,
            Encoding::Quant(spec) => spec.logical_dtype,
        }
    }
}

pub fn normalize_encoding(encoding: &Encoding) -> Encoding {
    match encoding {
        Encoding::Raw(dtype) => Encoding::Raw(*dtype),
        Encoding::Quant(spec) => Encoding::Quant(spec.clone().normalized()),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    #[default]
    Public,
    Internal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDecl {
    pub id: TensorId,
    pub name: String,
    pub shape: Vec<i64>,
    pub encoding: Encoding,
    pub alignment: u32,
    #[serde(default, skip_serializing_if = "Visibility::is_public")]
    pub visibility: Visibility,
}

impl Visibility {
    pub fn is_public(&self) -> bool {
        matches!(self, Visibility::Public)
    }
}

impl TensorDecl {
    pub fn dtype(&self) -> DType {
        self.encoding.dtype()
    }
}

#[must_use]
pub fn rectangle(shape: &[i64]) -> Option<(i64, i64)> {
    let (&cols, lead) = shape.split_last()?;
    if lead.is_empty() {
        return None;
    }
    let rows = lead
        .iter()
        .try_fold(1i64, |acc, dim| acc.checked_mul(*dim))?;
    Some((rows, cols))
}

#[must_use]
pub fn grouped_shape(lead: &[i64], groups: i64) -> Vec<i64> {
    let mut shape = lead.to_vec();
    shape.push(groups);
    shape
}

pub fn tensor_nbytes(shape: &[i64], element_bytes: u64) -> Option<u64> {
    tensor_elements(shape)?.checked_mul(element_bytes)
}

pub fn tensor_elements(shape: &[i64]) -> Option<u64> {
    let mut elements = 1u64;
    for dim in shape {
        let dim = u64::try_from(*dim).ok()?;
        elements = elements.checked_mul(dim)?;
    }
    Some(elements)
}

pub fn encoding_dense_element_bytes(encoding: &Encoding) -> Option<u64> {
    match encoding {
        Encoding::Raw(dtype) => Some(dtype.bytes_ceil()),
        Encoding::Quant(spec) => spec.dense_element_bytes(),
    }
}

pub fn encoding_nbytes(shape: &[i64], encoding: &Encoding) -> Option<u64> {
    match encoding {
        Encoding::Raw(dtype) => tensor_nbytes(shape, dtype.bytes_ceil()),
        Encoding::Quant(spec) => {
            let spec = spec.clone().normalized();
            if let Some((block_elements, block_bytes)) = spec.block_layout() {
                let elements = tensor_elements(shape)?;
                return elements.div_ceil(block_elements).checked_mul(block_bytes);
            }
            if let Some(element_bytes) = spec.dense_element_bytes() {
                return tensor_nbytes(shape, element_bytes);
            }
            let elements = tensor_elements(shape)?;
            let bits = elements.checked_mul(u64::from(spec.bits_per_element))?;
            Some(bits.div_ceil(8))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantGranularity {
    PerChannel,
    PerGroup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleForm {
    RawE8M0,
    F32Factors,
    Bf16AffineFactors,
}
