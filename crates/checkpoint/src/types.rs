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

/// What a checkpoint tensor holds; re-export of [`dtype::Dtype`]. Old
/// spellings (`BF16`, `F8E4M3`, `F8E5M2`, `E8M0`) survive as serde aliases
/// so a plan recorded under them still reads.
pub use dtype::Dtype as DType;

/// Whether a checkpoint storing `dtype` ships a separate block-scale tensor
/// (one scale per `[B, B]` tile; `B` itself is on the consuming kernel's
/// target, [`crate::plan::StorageTarget::block_scale_rows`]).
#[must_use]
pub fn is_block_scaled(dtype: DType) -> bool {
    matches!(dtype, DType::E4m3 | DType::E5m2)
}

/// Which on-disk format a checkpoint file is. `Unknown` is what a newer
/// zTensor reports for a format this build has no name for, not a
/// "cannot read this" marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointFormat {
    Safetensors,
    Gguf,
    Unknown,
    /// The loader's own container (`.zt`), including a root that names shards.
    Zt,
    /// NumPy's zip archive (`.npz`).
    Npz,
    /// PyTorch's pickle archive (`.pt`).
    Pt,
    /// HDF5 (`.h5`), including Keras checkpoints.
    Hdf5,
    /// ONNX protobuf (`.onnx`), read for its initializers.
    Onnx,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    Cuda,
    Metal,
    Vulkan,
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
    /// Two bits per weight, affine, sixteen sub-blocks of sixteen; unlike
    /// [`Self::GgufQ4K`] the super-block scales trail the payload.
    GgufQ2K,
    /// Three bits per weight, symmetric; the third bit is a separate
    /// inverted mask (set keeps the two-bit value, clear subtracts four).
    GgufQ3K,
    GgufQ4K,
    GgufQ5_0,
    GgufQ5K,
    GgufQ8_0,
    /// 4-bit integers biased by 8, low nibble first (`nibble - 8`); group
    /// scales are a separate tensor, so the zero point is the bias itself.
    Int4B8,
    /// llama.cpp's `block_q6_K`: 128 low nibbles, 64 high pairs, per
    /// sub-block scales, one super-block scale.
    GgufQ6K,
    /// llama.cpp's `block_q4_1`: F16 scale, F16 offset, packed nibbles;
    /// element is `nibble * d + m` (offset added, not subtracted).
    GgufQ4_1,
    /// llama.cpp's `block_q5_1`: [`Self::GgufQ4_1`] plus a plane of fifth bits.
    GgufQ5_1,
    /// llama.cpp's `block_iq4_nl`: F16 scale and packed 4-bit indices into
    /// a non-linear 16-entry table (`kvalues_iq4nl`), not an offset nibble.
    GgufIq4Nl,
    /// llama.cpp's `block_iq4_xs`: sub-blocks of 32 over
    /// [`Self::GgufIq4Nl`]'s table, 6-bit sub-block scale read as `ls - 32`.
    GgufIq4Xs,
    /// llama.cpp's `block_mxfp4`: one E8M0 scale byte then packed E2M1
    /// nibbles, interleaved (unlike [`Self::Mxfp4E2M1E8M0`]'s separate planes).
    GgufMxfp4,
    /// llama.cpp's `block_iq2_xxs`: indexes a 256-entry point table
    /// compiled in from `gguf-py`'s `iq2xxs_grid` (see
    /// `checkpoint/src/executor/iq_grid.rs`), not stored in the file.
    GgufIq2Xxs,
    /// llama.cpp's `block_iq2_xs`: 512-entry grid, 9-bit index + 7-bit
    /// sign in one `u16`; per-16 scales instead of `IQ2_XXS`'s per-32.
    GgufIq2Xs,
    /// llama.cpp's `block_iq2_s`: 1024-entry grid (8 index bits in `qs`,
    /// 2 in `qh`); signs stored outright, not packed via parity.
    GgufIq2S,
    /// llama.cpp's `block_iq3_xxs`: four-component grid points, 64 indices
    /// per block, four 7-bit sign indices, a 4-bit scale doubled to match.
    GgufIq3Xxs,
    /// llama.cpp's `block_iq3_s`: four-component grid; scale is the odd
    /// integer `1 + 2s`; `qh` contributes one bit per point (`IQ2S` uses two).
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

    /// The block a GGUF-family scheme stores, as `(elements, bytes)`, read
    /// off the container's own `gguf.<type>/2` row; `None` for a plain
    /// bit-packing (GGUF blocks carry scales inside the payload, so size
    /// isn't `elements * bits / 8`).
    pub fn block_layout(self) -> Option<(u64, u64)> {
        let row = ztensor::vocab::gguf::row_of(gguf_name(self)?)?;
        Some((row.elems_per_block, row.block_bytes))
    }

    /// Whether this scheme keeps its scales inside its payload — see
    /// [`block_layout`](Self::block_layout).
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
            // Sixteen: Q6_K/Q2_K/Q3_K sub-block size. Inert for extents
            // since `block_layout` answers those.
            Self::GgufQ2K | Self::GgufQ3K | Self::GgufQ6K => 16,
            // IQ lattice schemes have no group size in the affine sense.
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

/// The tiled affine layout's two geometry constants, used by
/// [`DType::U4g64tiled`] and [`RepackLayout::TiledAffineU4Weight`].
pub use dtype::{TILED_BAND, TILED_STEP};

/// Which backend kernel a [`Expr::Repack`](crate::contract::Expr::Repack)
/// names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RepackLayout {
    MarlinMxfp4Weight,
    MarlinMxfp4Scale,
    /// Four-bit affine code plane in mma fragment order
    /// (`kernels_cuda::linear::tiled::repack_affine_tiled`); target rows
    /// pad to a 16-column band, tail decodes to a zero weight.
    TiledAffineU4Weight,
    /// Factor plane beside it (`repack_factors_tiled`): a transpose of the
    /// (column, group) rectangle within each 16-column band.
    TiledAffineFactor,
}

/// A repack as the executor needs it: layout plus geometry. `target_rows`/
/// `target_cols` may exceed the source's when the layout pads to a tile
/// quantum; the kernel zero-fills the tail.
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

    /// Element width when the payload is a plain array, `None` when the
    /// scheme is blocked. Checks `is_self_contained` first rather than bit
    /// divisibility alone — Q8_0's bits divide evenly by 8 too, but its
    /// payload still carries a block scale a bits-only check would miss.
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

    /// See [`QuantScheme::block_layout`].
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

/// Whether a declared tensor is bound by the engine, or just a name the
/// contract needed internally (the algebra has no `let`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    /// A runtime weight. The engine binds it by name.
    #[default]
    Public,
    /// A name for the contract's own use. Not bound, not persistent.
    Internal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDecl {
    pub id: TensorId,
    pub name: String,
    /// What this rank holds, in elements — the shard a
    /// [`Expr::Shard`](crate::contract::Expr::Shard) cut out, not the
    /// whole declared tensor.
    pub shape: Vec<i64>,
    pub encoding: Encoding,
    pub alignment: u32,
    /// Whether the engine binds this name. See [`Visibility`].
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

/// A declared shape read as the `[rows, cols]` rectangle every encode
/// kernel walks: the last axis is the contracted one, every axis before it
/// folds into the row count (row-major, so the fold costs nothing). Rank 0
/// and rank 1 return `None` rather than inventing a `rows = 1`.
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

/// A block-scaled scales shape: the payload's leading axes, then one entry
/// per group along the contracted axis. Shared by the plan compiler (which
/// builds it) and the host executor (which checks a buffer against it).
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
            // A blocked scheme's scales live inside the payload, so its span
            // is blocks × block bytes, not elements × bits.
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

/// How a scale tensor's entries map onto the tensor they scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantGranularity {
    /// One scale per row of `channel_axis`.
    PerChannel,
    /// One scale per `group_size` elements along the axis after `channel_axis`.
    PerGroup,
}

/// What the engine's kernels expect a scale tensor to hold, once read. Not
/// derivable from the scale tensor's own dtype, so the declaration states
/// it explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleForm {
    /// Consumed as raw E8M0 exponent bytes (MXFP4 GEMM, dequant kernels).
    RawE8M0,
    /// Consumed as F32 multipliers. Whatever the scales were stored as (E8M0
    /// bytes, BF16, or F32 already) is expanded before the GEMM sees them.
    F32Factors,
    /// Consumed as BF16 multipliers, half the dequantization: a second
    /// tensor ([`QuantAttachment::zero_point_tensor`]) holds the per-group
    /// zero point, so an element is `code * scale + zero`.
    Bf16AffineFactors,
}
