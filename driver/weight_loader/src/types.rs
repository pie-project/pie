use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExprId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FileId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InstrId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Axis(pub u8);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    I32,
    I16,
    I8,
    U32,
    U16,
    U8,
    Bool,
}

impl DType {
    pub fn bytes(self) -> u64 {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::F32 | Self::F16 | Self::BF16 | Self::F8E4M3 | Self::F8E5M2
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointFormat {
    Safetensors,
    Gguf,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    Cuda,
    Portable,
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
    GgufQ4_0,
    GgufQ4K,
    GgufQ5_0,
    GgufQ5K,
    GgufQ8_0,
}

impl QuantScheme {
    pub fn default_bits(self) -> u8 {
        match self {
            Self::AwqInt4
            | Self::GptqInt4
            | Self::Mxfp4E2M1E8M0
            | Self::GgufQ4_0
            | Self::GgufQ4K => 4,
            Self::GgufQ5_0 | Self::GgufQ5K => 5,
            Self::Fp8E4M3
            | Self::Fp8E5M2
            | Self::Int8Symmetric
            | Self::Int8Asymmetric
            | Self::GgufQ8_0
            | Self::None => 8,
        }
    }

    pub fn default_group_size(self) -> u32 {
        match self {
            Self::AwqInt4 | Self::GptqInt4 | Self::Mxfp4E2M1E8M0 => 32,
            Self::GgufQ4_0 | Self::GgufQ4K | Self::GgufQ5_0 | Self::GgufQ5K => 32,
            Self::Fp8E4M3
            | Self::Fp8E5M2
            | Self::Int8Symmetric
            | Self::Int8Asymmetric
            | Self::GgufQ8_0
            | Self::None => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mxfp4MoePolicy {
    RoutedDecode,
    NativeGemm,
    EagerBf16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RowMap {
    #[default]
    Identity,
    Even,
    Odd,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RepackLayout {
    #[default]
    None,
    MarlinMxfp4Weight,
    MarlinMxfp4Scale,
    DenseRowGather,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RepackSpec {
    pub layout: RepackLayout,
    pub row_map: RowMap,
    pub batch: u32,
    pub source_rows: u32,
    pub source_row_offset: u32,
    pub target_rows: u32,
    pub valid_rows: u32,
    pub source_stride_cols: u32,
    pub source_col_offset: u32,
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
    pub scale_dtype: Option<DType>,
    pub zero_point_dtype: Option<DType>,
    pub block_shape: Vec<i64>,
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
        let bits = self.normalized_bits();
        if bits % 8 == 0 {
            Some(u64::from(bits / 8))
        } else {
            None
        }
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

pub fn normalize_encoding(encoding: &Encoding) -> Encoding {
    match encoding {
        Encoding::Raw(dtype) => Encoding::Raw(*dtype),
        Encoding::Quant(spec) => Encoding::Quant(spec.clone().normalized()),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    pub strides: Vec<i64>,
    pub alignment: u32,
}

impl Layout {
    pub fn dense(alignment: u32) -> Self {
        Self {
            strides: Vec::new(),
            alignment,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sharding {
    pub axis: Option<Axis>,
    pub world: u32,
    pub rank: u32,
}

impl Sharding {
    pub fn replicated() -> Self {
        Self {
            axis: None,
            world: 1,
            rank: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDecl {
    pub id: TensorId,
    pub name: String,
    pub shape: Vec<i64>,
    pub encoding: Encoding,
    pub layout: Layout,
    pub sharding: Sharding,
    pub alignment: u32,
}

impl TensorDecl {
    pub fn same_runtime_contract(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.encoding == other.encoding
            && self.layout == other.layout
            && self.sharding == other.sharding
            && self.alignment == other.alignment
    }

    pub fn dtype(&self) -> DType {
        match &self.encoding {
            Encoding::Raw(dtype) => *dtype,
            Encoding::Quant(spec) => spec.logical_dtype,
        }
    }

    pub fn with_name_and_id(&self, id: TensorId, name: impl Into<String>) -> Self {
        let mut out = self.clone();
        out.id = id;
        out.name = name.into();
        out
    }
}

pub fn tensor_nbytes(shape: &[i64], element_bytes: u64) -> Option<u64> {
    let mut elements = 1u64;
    for dim in shape {
        let dim = u64::try_from(*dim).ok()?;
        elements = elements.checked_mul(dim)?;
    }
    elements.checked_mul(element_bytes)
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
        Encoding::Raw(dtype) => Some(dtype.bytes()),
        Encoding::Quant(spec) => spec.dense_element_bytes(),
    }
}

pub fn encoding_nbytes(shape: &[i64], encoding: &Encoding) -> Option<u64> {
    match encoding {
        Encoding::Raw(dtype) => tensor_nbytes(shape, dtype.bytes()),
        Encoding::Quant(spec) => {
            let spec = spec.clone().normalized();
            if let Some(element_bytes) = spec.dense_element_bytes() {
                return tensor_nbytes(shape, element_bytes);
            }
            let elements = tensor_elements(shape)?;
            let bits = elements.checked_mul(u64::from(spec.bits_per_element))?;
            Some(bits.div_ceil(8))
        }
    }
}
