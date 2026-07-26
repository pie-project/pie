use serde::{Deserialize, Serialize};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    /// OCP Microscaling's 8-bit exponent-only scale format: the stored byte
    /// `b` denotes `2^(b - 127)`. It carries no sign and no mantissa, so it
    /// only ever appears as the scale beside a block-scaled tensor -- which is
    /// why `QuantScheme` long knew it only as half of `Mxfp4E2M1E8M0`.
    /// DeepSeek-V4 pairs it with FP8-E4M3 weights instead, a combination that
    /// composite cannot name.
    E8M0,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
}

impl DType {
    pub fn bytes(self) -> u64 {
        match self {
            Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::E8M0 | Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    /// Whether a checkpoint storing this dtype ships a separate block-scale
    /// tensor alongside it.
    ///
    /// A *format* fact, not a device one: DeepSeek- and GLM-style FP8
    /// checkpoints carry one scale per `[B, B]` tile of the weight, so an FP8
    /// tensor is never self-describing. The block size `B` is what the
    /// consuming kernel fixes, and that is on the target
    /// ([`crate::plan::StorageTarget::block_scale_rows`]); which dtypes
    /// arrive that way is here, because it is true of the file no matter who
    /// reads it.
    pub fn is_block_scaled(self) -> bool {
        matches!(self, Self::F8E4M3 | Self::F8E5M2)
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
    Metal,
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
    GgufQ4K,
    GgufQ5_0,
    GgufQ5K,
    GgufQ8_0,
    /// 4-bit integers biased by 8, eight to a 32-bit word, low nibble first.
    ///
    /// An element is `nibble - 8`, so the stored range `0..=15` reads as
    /// `-8..=7`. The group scales are a separate tensor rather than part of
    /// this name -- a contract pairs the two with [`crate::contract::Expr`]'s
    /// `Scale` -- which is why, unlike [`Self::AwqInt4`] and
    /// [`Self::GptqInt4`], there is no zero-point tensor implied here: the
    /// zero point *is* the 8.
    ///
    /// New variants go on the end. The FFI discriminants follow declaration
    /// order and the C++ side reads them as integers, so inserting one in the
    /// middle renumbers every scheme after it.
    Int4B8,
}

impl QuantScheme {
    pub fn default_bits(self) -> u8 {
        match self {
            Self::AwqInt4
            | Self::GptqInt4
            | Self::Mxfp4E2M1E8M0
            | Self::MlxAffineU4
            | Self::GgufQ4_0
            | Self::GgufQ4K
            | Self::Int4B8 => 4,
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
            Self::AwqInt4 | Self::GptqInt4 | Self::Mxfp4E2M1E8M0 | Self::Int4B8 => 32,
            Self::MlxAffineU4 => 64,
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
        if bits.is_multiple_of(8) {
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

impl Encoding {
    /// The type one element reads as. For a quantized encoding this is the
    /// logical type the elements decode to, not their storage width.
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

/// Whether a declared tensor is something the driver binds, or a name the
/// contract needed for itself.
///
/// The algebra has no `let`: the only way to use a subexpression twice, or to
/// feed one entry's result into another's [`Expr::Scale`](crate::contract::Expr::Scale) factors, is to give it
/// a name. Without this, every such name is also a runtime weight — a stacked
/// slab of dequantization factors ends up in the persistent arena and stays
/// there for the life of the process, and the bind namespace fills with
/// tensors no kernel will ever ask for.
///
/// `Internal` is that name without those consequences. It resolves through
/// [`Expr::Out`](crate::contract::Expr::Out) like any other, but the plan emits no `Finalize` for it, so the
/// driver never sees it and its buffer stays a temporary the memory planner may
/// reuse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    /// A runtime weight. The driver binds it by name.
    #[default]
    Public,
    /// A name for the contract's own use. Not bound, not persistent.
    Internal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDecl {
    pub id: TensorId,
    pub name: String,
    pub shape: Vec<i64>,
    pub encoding: Encoding,
    pub alignment: u32,
    /// Whether the driver binds this name. See [`Visibility`].
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

/// How a scale tensor's entries map onto the tensor they scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantGranularity {
    /// One scale per row of `channel_axis`.
    PerChannel,
    /// One scale per `group_size` elements along the axis after `channel_axis`.
    PerGroup,
}

/// What the driver's kernels expect a scale tensor to hold by the time they read
/// it.
///
/// Not derivable from the scale tensor itself — its dtype says how the bytes are
/// stored, not how the kernel wants them — so whoever declared the scale states
/// it. The driver used to infer it from `group_size == 32`, which was true only
/// because MXFP4 is the one scheme with that group size today.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleForm {
    /// Consumed as raw E8M0 exponent bytes. The MXFP4 GEMM, the dequant kernels
    /// and `make_expert_weight_view` all require U8 and assert on anything else.
    RawE8M0,
    /// Consumed as F32 multipliers. Whatever the scales were stored as (E8M0
    /// bytes, BF16, or F32 already) is expanded before the GEMM sees them.
    F32Factors,
}
