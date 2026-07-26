//! `#[repr(C)]` mirror of the LOW IR.
//!
//! These types are the loader's published vocabulary: the driver walks them in
//! place, so every field here is part of the ABI. They replace the hand-written
//! `PieLoader*` structs in `driver/common/include/pie_native/load_plan.hpp`,
//! which the generated header (`loader/include/pie_loader.h`) supersedes.
//!
//! Two rules govern this module:
//!
//! * **No owning types.** Every aggregate is a pointer + length pair into an
//!   arena owned by [`crate::ffi::arena::PlanArena`]. Slices stay valid exactly
//!   as long as the `PieLoaderPlan` that produced them.
//! * **No `Option`, no niches.** `Option<T>` is spelled as an explicit
//!   `has_*: bool` companion, matching the C++ views this replaces, so the
//!   layout is legible from C without knowing Rust's niche rules.

use crate::types::{BackendKind, DType, Mxfp4MoePolicy, QuantScheme, RepackLayout, RowMap};

/// Sentinel for "no buffer", mirroring the C++ `numeric_limits<uint32_t>::max()`
/// defaults on `PieLoaderStorageInstrView::buffer_id` and `slab_file_id`.
pub const PIE_LOADER_NO_BUFFER: u32 = u32::MAX;

/// The LOW IR version. A driver that reads a plan with a different version is
/// reading a layout it was not compiled against.
pub const PIE_LOADER_PLAN_VERSION: u32 = 5;

// Tile-map capability bits. A driver ORs together the transforms its kernels
// implement and passes the result as `PieLoaderTargetSpec::tile_map_mask`; the
// compiler then refuses to emit any transform outside that set rather than
// producing a plan the device cannot run.
//
// These live here, in the module that owns the C surface, because the header is
// where they have to be correct. `crate::load_plan` re-exports them under short
// names for use inside the compiler.
pub const PIE_LOADER_TILE_MAP_CAST: u32 = 1 << 0;
pub const PIE_LOADER_TILE_MAP_DECODE: u32 = 1 << 1;
pub const PIE_LOADER_TILE_MAP_ENCODE: u32 = 1 << 2;
pub const PIE_LOADER_TILE_MAP_TRANSCODE: u32 = 1 << 3;
pub const PIE_LOADER_TILE_MAP_REBLOCK: u32 = 1 << 4;
pub const PIE_LOADER_TILE_MAP_REORDER: u32 = 1 << 5;
pub const PIE_LOADER_TILE_MAP_REPACK: u32 = 1 << 6;

// Fused-chain capability bits, on the same principle: the *loader* knows what a
// fusion means — which two-step chain `PieLoaderTransformFusion::Fp8ToMxfp4`
// collapses, and that the collapsed form is bit-identical — and the *driver*
// knows whether it built the kernel. A bit set here says only the second thing.
//
// Before this was a bit, it was `fused_transcode: bool`, which conflated the two
// and had no room for a second fusion. A mask does, and it makes adding one a
// change to a table rather than to a signature.
pub const PIE_LOADER_FUSION_FP8_TO_MXFP4: u32 = 1 << 0;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderBackendKind {
    Cuda = 0,
    Metal = 1,
    Unknown = 255,
}

impl From<BackendKind> for PieLoaderBackendKind {
    fn from(value: BackendKind) -> Self {
        match value {
            BackendKind::Cuda => Self::Cuda,
            BackendKind::Metal => Self::Metal,
            BackendKind::Unknown => Self::Unknown,
        }
    }
}

impl TryFrom<u32> for PieLoaderBackendKind {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, u32> {
        match value {
            0 => Ok(Self::Cuda),
            1 => Ok(Self::Metal),
            255 => Ok(Self::Unknown),
            other => Err(other),
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    F8E4M3 = 3,
    F8E5M2 = 4,
    I32 = 5,
    I16 = 6,
    I8 = 7,
    U32 = 8,
    U16 = 9,
    U8 = 10,
    Bool = 11,
}

impl From<DType> for PieLoaderDType {
    fn from(value: DType) -> Self {
        match value {
            DType::F32 => Self::F32,
            DType::F16 => Self::F16,
            DType::BF16 => Self::BF16,
            DType::F8E4M3 => Self::F8E4M3,
            DType::F8E5M2 => Self::F8E5M2,
            DType::I32 => Self::I32,
            DType::I16 => Self::I16,
            DType::I8 => Self::I8,
            DType::U32 => Self::U32,
            DType::U16 => Self::U16,
            DType::U8 => Self::U8,
            DType::Bool => Self::Bool,
        }
    }
}

/// The inverse. Kept adjacent to the forward direction so the two cannot drift;
/// `dtype_survives_the_c_boundary` holds them to it.
impl From<PieLoaderDType> for DType {
    fn from(value: PieLoaderDType) -> Self {
        match value {
            PieLoaderDType::F32 => Self::F32,
            PieLoaderDType::F16 => Self::F16,
            PieLoaderDType::BF16 => Self::BF16,
            PieLoaderDType::F8E4M3 => Self::F8E4M3,
            PieLoaderDType::F8E5M2 => Self::F8E5M2,
            PieLoaderDType::I32 => Self::I32,
            PieLoaderDType::I16 => Self::I16,
            PieLoaderDType::I8 => Self::I8,
            PieLoaderDType::U32 => Self::U32,
            PieLoaderDType::U16 => Self::U16,
            PieLoaderDType::U8 => Self::U8,
            PieLoaderDType::Bool => Self::Bool,
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderEncodingKind {
    Raw = 0,
    Quant = 1,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderMxfp4MoePolicy {
    RoutedDecode = 0,
    NativeGemm = 1,
    EagerBf16 = 2,
}

impl From<Mxfp4MoePolicy> for PieLoaderMxfp4MoePolicy {
    fn from(value: Mxfp4MoePolicy) -> Self {
        match value {
            Mxfp4MoePolicy::RoutedDecode => Self::RoutedDecode,
            Mxfp4MoePolicy::NativeGemm => Self::NativeGemm,
            Mxfp4MoePolicy::EagerBf16 => Self::EagerBf16,
        }
    }
}

/// Discriminants follow `crate::types::QuantScheme` declaration order, which is
/// *not* the order of the hand-written C++ enum this replaces (`MlxAffineU4` is
/// eighth here and last there). The mismatch was invisible while the boundary
/// was JSON, because the C++ parser mapped by name. Now that the two sides share
/// integers, the generated header must be the only definition — deleting the
/// hand-written enum is part of the same change, not a follow-up.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PieLoaderQuantScheme {
    None = 0,
    Fp8E4M3 = 1,
    Fp8E5M2 = 2,
    Int8Symmetric = 3,
    Int8Asymmetric = 4,
    AwqInt4 = 5,
    GptqInt4 = 6,
    Mxfp4E2M1E8M0 = 7,
    MlxAffineU4 = 8,
    GgufQ4_0 = 9,
    GgufQ4K = 10,
    GgufQ5_0 = 11,
    GgufQ5K = 12,
    GgufQ8_0 = 13,
}

impl From<QuantScheme> for PieLoaderQuantScheme {
    fn from(value: QuantScheme) -> Self {
        match value {
            QuantScheme::None => Self::None,
            QuantScheme::Fp8E4M3 => Self::Fp8E4M3,
            QuantScheme::Fp8E5M2 => Self::Fp8E5M2,
            QuantScheme::Int8Symmetric => Self::Int8Symmetric,
            QuantScheme::Int8Asymmetric => Self::Int8Asymmetric,
            QuantScheme::AwqInt4 => Self::AwqInt4,
            QuantScheme::GptqInt4 => Self::GptqInt4,
            QuantScheme::Mxfp4E2M1E8M0 => Self::Mxfp4E2M1E8M0,
            QuantScheme::MlxAffineU4 => Self::MlxAffineU4,
            QuantScheme::GgufQ4_0 => Self::GgufQ4_0,
            QuantScheme::GgufQ4K => Self::GgufQ4K,
            QuantScheme::GgufQ5_0 => Self::GgufQ5_0,
            QuantScheme::GgufQ5K => Self::GgufQ5K,
            QuantScheme::GgufQ8_0 => Self::GgufQ8_0,
        }
    }
}

/// The inverse, for the same reason; `quant_scheme_survives_the_c_boundary`
/// holds the pair together.
impl From<PieLoaderQuantScheme> for QuantScheme {
    fn from(value: PieLoaderQuantScheme) -> Self {
        match value {
            PieLoaderQuantScheme::None => Self::None,
            PieLoaderQuantScheme::Fp8E4M3 => Self::Fp8E4M3,
            PieLoaderQuantScheme::Fp8E5M2 => Self::Fp8E5M2,
            PieLoaderQuantScheme::Int8Symmetric => Self::Int8Symmetric,
            PieLoaderQuantScheme::Int8Asymmetric => Self::Int8Asymmetric,
            PieLoaderQuantScheme::AwqInt4 => Self::AwqInt4,
            PieLoaderQuantScheme::GptqInt4 => Self::GptqInt4,
            PieLoaderQuantScheme::Mxfp4E2M1E8M0 => Self::Mxfp4E2M1E8M0,
            PieLoaderQuantScheme::MlxAffineU4 => Self::MlxAffineU4,
            PieLoaderQuantScheme::GgufQ4_0 => Self::GgufQ4_0,
            PieLoaderQuantScheme::GgufQ4K => Self::GgufQ4K,
            PieLoaderQuantScheme::GgufQ5_0 => Self::GgufQ5_0,
            PieLoaderQuantScheme::GgufQ5K => Self::GgufQ5K,
            PieLoaderQuantScheme::GgufQ8_0 => Self::GgufQ8_0,
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderRepackLayout {
    None = 0,
    MarlinMxfp4Weight = 1,
    MarlinMxfp4Scale = 2,
    DenseRowGather = 3,
}

impl From<RepackLayout> for PieLoaderRepackLayout {
    fn from(value: RepackLayout) -> Self {
        match value {
            RepackLayout::None => Self::None,
            RepackLayout::MarlinMxfp4Weight => Self::MarlinMxfp4Weight,
            RepackLayout::MarlinMxfp4Scale => Self::MarlinMxfp4Scale,
            RepackLayout::DenseRowGather => Self::DenseRowGather,
        }
    }
}

impl From<PieLoaderRepackLayout> for RepackLayout {
    fn from(value: PieLoaderRepackLayout) -> Self {
        match value {
            PieLoaderRepackLayout::None => Self::None,
            PieLoaderRepackLayout::MarlinMxfp4Weight => Self::MarlinMxfp4Weight,
            PieLoaderRepackLayout::MarlinMxfp4Scale => Self::MarlinMxfp4Scale,
            PieLoaderRepackLayout::DenseRowGather => Self::DenseRowGather,
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderRowMap {
    Identity = 0,
    Even = 1,
    Odd = 2,
}

impl From<RowMap> for PieLoaderRowMap {
    fn from(value: RowMap) -> Self {
        match value {
            RowMap::Identity => Self::Identity,
            RowMap::Even => Self::Even,
            RowMap::Odd => Self::Odd,
        }
    }
}

impl From<PieLoaderRowMap> for RowMap {
    fn from(value: PieLoaderRowMap) -> Self {
        match value {
            PieLoaderRowMap::Identity => Self::Identity,
            PieLoaderRowMap::Even => Self::Even,
            PieLoaderRowMap::Odd => Self::Odd,
        }
    }
}

/// Contract fields cross as `uint32_t`, so an out-of-range value is a
/// diagnosable request rather than an invalid Rust enum. These are the inverses.
impl TryFrom<u32> for PieLoaderDType {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::BF16,
            3 => Self::F8E4M3,
            4 => Self::F8E5M2,
            5 => Self::I32,
            6 => Self::I16,
            7 => Self::I8,
            8 => Self::U32,
            9 => Self::U16,
            10 => Self::U8,
            11 => Self::Bool,
            other => return Err(other),
        })
    }
}

impl TryFrom<u32> for PieLoaderEncodingKind {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::Raw,
            1 => Self::Quant,
            other => return Err(other),
        })
    }
}

impl TryFrom<u32> for PieLoaderQuantScheme {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::None,
            1 => Self::Fp8E4M3,
            2 => Self::Fp8E5M2,
            3 => Self::Int8Symmetric,
            4 => Self::Int8Asymmetric,
            5 => Self::AwqInt4,
            6 => Self::GptqInt4,
            7 => Self::Mxfp4E2M1E8M0,
            8 => Self::MlxAffineU4,
            9 => Self::GgufQ4_0,
            10 => Self::GgufQ4K,
            11 => Self::GgufQ5_0,
            12 => Self::GgufQ5K,
            13 => Self::GgufQ8_0,
            other => return Err(other),
        })
    }
}

impl TryFrom<u32> for PieLoaderRepackLayout {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::None,
            1 => Self::MarlinMxfp4Weight,
            2 => Self::MarlinMxfp4Scale,
            3 => Self::DenseRowGather,
            other => return Err(other),
        })
    }
}

impl TryFrom<u32> for PieLoaderRowMap {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::Identity,
            1 => Self::Even,
            2 => Self::Odd,
            other => return Err(other),
        })
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderStorageInstrKind {
    Allocate = 0,
    ExtentWrite = 1,
    TileMap = 2,
    CreateView = 3,
    Release = 4,
    Finalize = 5,
    BulkExtentWrite = 6,
    SlabScatter = 7,
}

/// `None` is the resting value for instructions that carry no tile map, so it
/// sorts last rather than first — matching the C++ enum and the default on
/// `PieLoaderStorageInstrView::tile_kind`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderTileMapKind {
    Cast = 0,
    Decode = 1,
    Encode = 2,
    Transcode = 3,
    Reblock = 4,
    Reorder = 5,
    Repack = 6,
    None = 7,
}

impl From<crate::load_plan::TileMapKind> for PieLoaderTileMapKind {
    fn from(value: crate::load_plan::TileMapKind) -> Self {
        use crate::load_plan::TileMapKind as K;
        match value {
            K::Cast => Self::Cast,
            K::Decode => Self::Decode,
            K::Encode => Self::Encode,
            K::Transcode => Self::Transcode,
            K::Reblock => Self::Reblock,
            K::Reorder => Self::Reorder,
            K::Repack => Self::Repack,
        }
    }
}

/// A transform chain the backend collapsed into a single kernel.
///
/// `None` is the resting value, so the driver's `switch` needs no separate
/// "is there a fusion" test.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderTransformFusion {
    None = 0,
    /// Encode an FP8 source straight to MXFP4, skipping the BF16 HBM
    /// round-trip. Bit-identical to the two-step path.
    Fp8ToMxfp4 = 1,
}

impl From<crate::load_plan::TransformFusion> for PieLoaderTransformFusion {
    fn from(value: crate::load_plan::TransformFusion) -> Self {
        use crate::load_plan::TransformFusion as F;
        match value {
            F::None => Self::None,
            F::Fp8ToMxfp4 => Self::Fp8ToMxfp4,
        }
    }
}

/// A borrowed UTF-8 string. Not NUL-terminated: plan strings come from Rust
/// `String`s, and copying them only to append a NUL would double the arena for
/// no reader that needs it. C++ consumers build a `string_view` from the pair.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderBytes {
    pub ptr: *const u8,
    pub len: usize,
}

impl Default for PieLoaderBytes {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderU32Slice {
    pub ptr: *const u32,
    pub len: usize,
}

impl Default for PieLoaderU32Slice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

/// One tensor the driver promises to bind after the load.
///
/// This is the driver's half of the contract (§10.2.1). Today the loader's
/// `arch/` passes decide *both* what the model needs and how to build it, so a
/// pass that invents an output name the driver never binds — or computes a shape
/// the driver disagrees with — is caught by nothing. Declaring the demand makes
/// `pie_loader_verify` a real check rather than a self-comparison.
///
/// State only what you know. `shape.len == 0` means "do not check the shape",
/// which is the honest answer for a tensor whose runtime layout the driver reads
/// off the loaded tensor rather than predicting. The encoding is deliberately
/// absent from this struct: it is the loader's choice under the runtime quant
/// policy, and binders probe it afterwards.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderTensorDemand {
    pub name: PieLoaderBytes,
    /// The shape *this rank* expects, already divided by the TP size. Empty to
    /// demand presence without a shape.
    pub shape: PieLoaderI64Slice,
    /// Absence is not a violation, e.g. `lm_head.weight` under
    /// `tie_word_embeddings`.
    pub optional: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderTensorDemandSlice {
    pub ptr: *const PieLoaderTensorDemand,
    pub len: usize,
}

impl Default for PieLoaderTensorDemandSlice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderI64Slice {
    pub ptr: *const i64,
    pub len: usize,
}

impl Default for PieLoaderI64Slice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderDimSpecView {
    pub count: i64,
    pub src_stride: i64,
    pub dst_stride: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderDimSpecSlice {
    pub ptr: *const PieLoaderDimSpecView,
    pub len: usize,
}

impl Default for PieLoaderDimSpecSlice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderStridedExtentView {
    pub base_offset: u64,
    pub element_bytes: u32,
    pub dims: PieLoaderDimSpecSlice,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderSourceExtentView {
    pub file_id: u32,
    pub tensor_id: u32,
    pub file_offset: u64,
    pub span_bytes: u64,
    pub stride: PieLoaderStridedExtentView,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderDestExtentView {
    pub buffer_id: u32,
    pub offset: u64,
    pub stride: PieLoaderStridedExtentView,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderTensorDeclView {
    pub id: u32,
    pub name: PieLoaderBytes,
    pub dtype: PieLoaderDType,
    pub encoding_kind: PieLoaderEncodingKind,
    pub quant_scheme: PieLoaderQuantScheme,
    pub quant_bits_per_element: u8,
    pub quant_group_size: u32,
    pub shape: PieLoaderI64Slice,
    pub alignment: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderTensorDeclSlice {
    pub ptr: *const PieLoaderTensorDeclView,
    pub len: usize,
}

impl Default for PieLoaderTensorDeclSlice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

/// How a scale tensor's entries map onto the tensor they scale.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderQuantGranularity {
    PerChannel = 0,
    PerGroup = 1,
}

/// What the driver's kernels expect a scale tensor to hold when they read it.
///
/// Not derivable from the scale tensor: its dtype says how the bytes are stored,
/// not how the kernel wants them. The driver used to infer this from
/// `group_size == 32`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderScaleForm {
    /// Raw E8M0 exponent bytes, consumed as-is.
    RawE8M0 = 0,
    /// F32 multipliers; expand before the GEMM sees them.
    F32Factors = 1,
}

/// A quantized tensor paired with the tensor holding its scales.
///
/// Both are entries in [`PieLoaderPlan::tensors`], named by `id`. The driver has
/// to know the pairing in order to attach the quant metadata its kernels read;
/// it used to rediscover it by matching name suffixes over the tensor list,
/// which guessed at something the loader states here (`load_plan.rs`'s
/// `derive_quant_attachments`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderQuantAttachmentView {
    pub tensor_id: u32,
    pub scale_tensor_id: u32,
    pub granularity: PieLoaderQuantGranularity,
    pub group_size: u32,
    pub channel_axis: u32,
    pub scale_form: PieLoaderScaleForm,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderQuantAttachmentSlice {
    pub ptr: *const PieLoaderQuantAttachmentView,
    pub len: usize,
}

impl Default for PieLoaderQuantAttachmentSlice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

/// How much of the caller's declared contract the plan delivers.
///
/// `covered < demanded` means the loader did not build something the driver said
/// it would bind. An absent *optional* demand is dropped from both counts rather
/// than only from `covered`, so a tied-embedding checkpoint that legitimately
/// has no `lm_head.weight` still reports full coverage.
///
/// A driver that declared nothing gets `0 / 0`, which passes — the declaration
/// is opt-in per model family, not a flag day.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderContractCoverageView {
    pub covered: usize,
    pub demanded: usize,
}

/// Which on-disk format a checkpoint file uses.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderCheckpointFormat {
    Safetensors = 0,
    Gguf = 1,
    Unknown = 2,
}

impl From<crate::types::CheckpointFormat> for PieLoaderCheckpointFormat {
    fn from(value: crate::types::CheckpointFormat) -> Self {
        match value {
            crate::types::CheckpointFormat::Safetensors => Self::Safetensors,
            crate::types::CheckpointFormat::Gguf => Self::Gguf,
            crate::types::CheckpointFormat::Unknown => Self::Unknown,
        }
    }
}

/// One file the plan reads from. `PieLoaderSourceTensorView::file_id` indexes
/// `PieLoaderPlan::files`, so the driver no longer has to re-derive the file
/// order for itself (§6).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderCheckpointFileView {
    pub id: u32,
    pub path: PieLoaderBytes,
    pub size_bytes: u64,
    pub format: PieLoaderCheckpointFormat,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderCheckpointFileSlice {
    pub ptr: *const PieLoaderCheckpointFileView,
    pub len: usize,
}

impl Default for PieLoaderCheckpointFileSlice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderSourceTensorView {
    pub id: u32,
    pub name: PieLoaderBytes,
    pub file_id: u32,
    pub file_offset: u64,
    pub span_bytes: u64,
    pub dtype: PieLoaderDType,
    pub encoding_kind: PieLoaderEncodingKind,
    pub quant_scheme: PieLoaderQuantScheme,
    pub quant_bits_per_element: u8,
    pub quant_group_size: u32,
    pub shape: PieLoaderI64Slice,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderSourceTensorSlice {
    pub ptr: *const PieLoaderSourceTensorView,
    pub len: usize,
}

impl Default for PieLoaderSourceTensorSlice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderBufferDeclView {
    pub id: u32,
    pub tensor_id: u32,
    pub has_tensor: bool,
    pub bytes: u64,
    pub alignment: u32,
    pub temporary: bool,
    pub has_persistent_offset: bool,
    pub persistent_offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderBufferDeclSlice {
    pub ptr: *const PieLoaderBufferDeclView,
    pub len: usize,
}

impl Default for PieLoaderBufferDeclSlice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderSlabPlacementView {
    pub src_offset: u64,
    pub dest_offset: u64,
    pub bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderSlabPlacementSlice {
    pub ptr: *const PieLoaderSlabPlacementView,
    pub len: usize,
}

impl Default for PieLoaderSlabPlacementSlice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

/// The flattened instruction. Rust's `StorageInstr` is a sum type whose variants
/// carry disjoint payloads; C has no such thing, so this is the union of all
/// variants with `kind` as the tag and `has_source`/`has_dest` marking which
/// optional members are live. Members a given `kind` does not use keep their
/// resting values.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderStorageInstrView {
    pub id: u32,
    pub kind: PieLoaderStorageInstrKind,
    pub buffer_id: u32,
    pub source: PieLoaderSourceExtentView,
    pub has_source: bool,
    pub dest: PieLoaderDestExtentView,
    pub has_dest: bool,
    pub input_buffers: PieLoaderU32Slice,
    pub output_buffers: PieLoaderU32Slice,
    pub tile_kind: PieLoaderTileMapKind,
    /// Rows of the output to transform per launch; `0` means the whole tensor in
    /// one pass.
    ///
    /// This is where the driver's `max_tile_bytes` budget ends up. The budget
    /// itself does not cross the boundary: the driver stated it in the request,
    /// the loader answered with a row count in `backend::lower`, and sending the
    /// question back alongside the answer would only invite the executor to
    /// re-derive it (§8.1).
    pub rows_per_tile: u32,
    /// A transform chain the backend collapsed into one kernel.
    pub transform_fusion: PieLoaderTransformFusion,
    pub transform_from: PieLoaderQuantScheme,
    pub transform_to: PieLoaderQuantScheme,
    pub repack_layout: PieLoaderRepackLayout,
    pub row_map: PieLoaderRowMap,
    pub transform_batch: u32,
    pub transform_source_rows: u32,
    pub transform_source_row_offset: u32,
    pub transform_target_rows: u32,
    pub transform_valid_rows: u32,
    pub transform_source_stride_cols: u32,
    pub transform_source_col_offset: u32,
    pub transform_source_cols: u32,
    pub transform_target_cols: u32,
    pub transform_scratch_bytes: u64,
    pub name: PieLoaderBytes,
    pub slab_file_id: u32,
    pub slab_file_offset: u64,
    pub slab_span_bytes: u64,
    pub slab_placements: PieLoaderSlabPlacementSlice,
}

impl Default for PieLoaderStorageInstrView {
    fn default() -> Self {
        Self {
            id: 0,
            kind: PieLoaderStorageInstrKind::Allocate,
            buffer_id: PIE_LOADER_NO_BUFFER,
            source: PieLoaderSourceExtentView::default(),
            has_source: false,
            dest: PieLoaderDestExtentView::default(),
            has_dest: false,
            input_buffers: PieLoaderU32Slice::default(),
            output_buffers: PieLoaderU32Slice::default(),
            tile_kind: PieLoaderTileMapKind::None,
            rows_per_tile: 0,
            transform_fusion: PieLoaderTransformFusion::None,
            transform_from: PieLoaderQuantScheme::None,
            transform_to: PieLoaderQuantScheme::None,
            repack_layout: PieLoaderRepackLayout::None,
            row_map: PieLoaderRowMap::Identity,
            transform_batch: 0,
            transform_source_rows: 0,
            transform_source_row_offset: 0,
            transform_target_rows: 0,
            transform_valid_rows: 0,
            transform_source_stride_cols: 0,
            transform_source_col_offset: 0,
            transform_source_cols: 0,
            transform_target_cols: 0,
            transform_scratch_bytes: 0,
            name: PieLoaderBytes::default(),
            slab_file_id: PIE_LOADER_NO_BUFFER,
            slab_file_offset: 0,
            slab_span_bytes: 0,
            slab_placements: PieLoaderSlabPlacementSlice::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderStorageInstrSlice {
    pub ptr: *const PieLoaderStorageInstrView,
    pub len: usize,
}

impl Default for PieLoaderStorageInstrSlice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderMemoryPlanView {
    pub persistent_bytes: u64,
    pub temporary_peak_bytes: u64,
    pub transform_scratch_peak_bytes: u64,
    pub checkpoint_read_bytes: u64,
    pub device_write_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderOptimizerPassStatsView {
    pub name: PieLoaderBytes,
    pub exprs_before: u64,
    pub exprs_after: u64,
    pub rewrites: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderOptimizerPassStatsSlice {
    pub ptr: *const PieLoaderOptimizerPassStatsView,
    pub len: usize,
}

impl Default for PieLoaderOptimizerPassStatsSlice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderOptimizerReportView {
    pub passes: PieLoaderOptimizerPassStatsSlice,
}

/// The target the plan was compiled against. The driver reads it back to assert
/// the plan it received is the plan it asked for — the same fields it supplied
/// in the request, plus the rank identity that makes a TP shard distinguishable
/// from its siblings (§6.2).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderTargetView {
    pub backend: PieLoaderBackendKind,
    pub tp_rank: u32,
    pub tp_size: u32,
    pub max_tile_bytes: u64,
    pub preferred_alignment: u32,
    pub tile_map_mask: u32,
    pub mxfp4_moe: PieLoaderMxfp4MoePolicy,
    pub native_mxfp4_moe: bool,
    pub fusion_mask: u32,
    pub encode_scratch_dtype: PieLoaderDType,
    pub block_scale_rows: u32,
}

impl From<&crate::load_plan::StorageTarget> for PieLoaderTargetView {
    fn from(value: &crate::load_plan::StorageTarget) -> Self {
        Self {
            backend: value.backend.into(),
            tp_rank: value.tp_rank,
            tp_size: value.tp_size,
            max_tile_bytes: value.max_tile_bytes,
            preferred_alignment: value.preferred_alignment,
            tile_map_mask: value.tile_map_mask,
            mxfp4_moe: value.mxfp4_moe.into(),
            native_mxfp4_moe: value.native_mxfp4_moe,
            fusion_mask: value.fusion_mask,
            encode_scratch_dtype: value.encode_scratch_dtype.into(),
            block_scale_rows: value.block_scale_rows,
        }
    }
}

/// The compiled plan, as the driver sees it.
///
/// The leading members reproduce the old `LoadPlanView` in order, so an executor
/// written against that view compiles unchanged against this struct. `target`
/// and `compiler_version` fold in the accessors `loaded_model.cpp` reached
/// through `LoadPlan` methods (`backend()`, `mxfp4_moe()`, `native_mxfp4_moe()`,
/// `preferred_alignment()`, `max_tile_bytes()`, `tile_map_mask()`,
/// `compiler_version()`), which have no method syntax to hide behind once the
/// type is POD.
///
/// `owner` is the opaque handle to the arena keeping every slice above alive. It
/// is consumed by `pie_loader_release`; the driver must not dereference it.
#[repr(C)]
#[derive(Debug)]
pub struct PieLoaderPlan {
    pub version: u32,
    pub files: PieLoaderCheckpointFileSlice,
    pub sources: PieLoaderSourceTensorSlice,
    pub tensors: PieLoaderTensorDeclSlice,
    pub buffers: PieLoaderBufferDeclSlice,
    pub instrs: PieLoaderStorageInstrSlice,
    pub schedule: PieLoaderU32Slice,
    pub memory: PieLoaderMemoryPlanView,
    pub optimizer: PieLoaderOptimizerReportView,
    pub compiler_version: u64,
    pub target: PieLoaderTargetView,
    pub attachments: PieLoaderQuantAttachmentSlice,
    /// Measured against `PieLoaderRequest::demands` at compile time.
    pub coverage: PieLoaderContractCoverageView,
    /// The name of the materialized weights this plan produces, as 16 hex
    /// digits. Stable for as long as nothing that decides the bytes changes, so
    /// a driver can use it to key an artifact cache.
    pub cache_key: PieLoaderBytes,
    /// One line describing the plan, for a boot log.
    pub summary: PieLoaderBytes,
    /// The plan's counts and instruction histograms as JSON, for an operator
    /// dump. Rendered by the loader so no driver keeps a second table of
    /// instruction names to fall out of step with this one.
    pub stats_json: PieLoaderBytes,
    pub owner: *mut std::ffi::c_void,
}
