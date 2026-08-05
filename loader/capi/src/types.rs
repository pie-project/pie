//! `#[repr(C)]` mirror of the LOW IR.
//!
//! These types are the loader's published vocabulary: the driver walks them in
//! place, so every field here is part of the ABI. They replace the hand-written
//! `PieLoader*` structs in `driver/{cuda,metal}/src/loader/load_plan.hpp`,
//! which the generated header (`loader/include/pie_loader.h`) supersedes.
//!
//! Two rules govern this module:
//!
//! * **No owning types.** Every aggregate is a pointer + length pair into an
//!   arena owned by [`crate::arena::PlanArena`]. Slices stay valid exactly
//!   as long as the `PieLoaderPlan` that produced them.
//! * **No `Option`, no niches.** `Option<T>` is spelled as an explicit
//!   `has_*: bool` companion, matching the C++ views this replaces, so the
//!   layout is legible from C without knowing Rust's niche rules.

use pie_loader::contract::Visibility;
use pie_loader::types::{
    BackendKind, DType, Encoding, QuantGranularity, QuantScheme, QuantSpec, RepackLayout, ScaleForm,
};

/// Sentinel for "no buffer", mirroring the C++ `numeric_limits<uint32_t>::max()`
/// default on `PieLoaderStorageInstrView::buffer_id`.
pub const PIE_LOADER_NO_BUFFER: u32 = u32::MAX;

/// Sentinel for "no source tensor", on the optional tensor-id fields.
pub const PIE_LOADER_NO_TENSOR: u32 = u32::MAX;

// Tile-map capability bits. A driver ORs together the transforms its kernels
// implement and passes the result as `PieLoaderTargetSpec::tile_map_mask`; the
// compiler then refuses to emit any transform outside that set rather than
// producing a plan the device cannot run.
//
// These live here, in the module that owns the C surface, because the header is
// where they have to be correct. `pie_loader::plan` states the same six bits.
// Restated here rather than aliased because cbindgen emits a literal and cannot
// follow a path — and because the arrow used to run the other way, with the
// compiler importing its own serialization format's constants. Two independent
// statements, checked below, is the same shape as the loader/driver cross-check
// in `ffi/mod.rs`.
pub const PIE_LOADER_TILE_MAP_CAST: u32 = 1 << 0;
pub const PIE_LOADER_TILE_MAP_DECODE: u32 = 1 << 1;
pub const PIE_LOADER_TILE_MAP_ENCODE: u32 = 1 << 2;
pub const PIE_LOADER_TILE_MAP_TRANSCODE: u32 = 1 << 3;
pub const PIE_LOADER_TILE_MAP_REBLOCK: u32 = 1 << 4;
pub const PIE_LOADER_TILE_MAP_REPACK: u32 = 1 << 6;
pub const PIE_LOADER_TILE_MAP_SCALE: u32 = 1 << 7;

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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PieLoaderDType {
    #[default]
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
    // Appended after `Bool` so existing discriminants keep their values: the
    // enum is an ABI, not a declaration order.
    E8M0 = 12,
    I64 = 13,
    U64 = 14,
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
            DType::E8M0 => Self::E8M0,
            DType::I64 => Self::I64,
            DType::U64 => Self::U64,
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
            PieLoaderDType::E8M0 => Self::E8M0,
            PieLoaderDType::I64 => Self::I64,
            PieLoaderDType::U64 => Self::U64,
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderEncodingKind {
    Raw = 0,
    Quant = 1,
}

/// Discriminants follow `pie_loader::types::QuantScheme` declaration order, which is
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
    Int4B8 = 14,
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
            QuantScheme::Int4B8 => Self::Int4B8,
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
            PieLoaderQuantScheme::Int4B8 => Self::Int4B8,
        }
    }
}

/// A repack's kernel, across the ABI.
///
/// `None` is the wire sentinel for a transform that ends in no kernel, the way
/// [`PieLoaderQuantScheme::None`] is for a transform that converts nothing. It
/// is deliberately *not* a member of [`RepackLayout`]: outbound it means "this
/// tile map is not a repack", and inbound — where the only reader is a contract
/// node that has already said it is one — it is rejected, because an all-zero
/// node is the shape a forgotten field has.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderRepackLayout {
    None = 0,
    MarlinMxfp4Weight = 1,
    MarlinMxfp4Scale = 2,
}

impl From<RepackLayout> for PieLoaderRepackLayout {
    fn from(value: RepackLayout) -> Self {
        match value {
            RepackLayout::MarlinMxfp4Weight => Self::MarlinMxfp4Weight,
            RepackLayout::MarlinMxfp4Scale => Self::MarlinMxfp4Scale,
        }
    }
}

impl TryFrom<PieLoaderRepackLayout> for RepackLayout {
    type Error = ();
    fn try_from(value: PieLoaderRepackLayout) -> Result<Self, ()> {
        match value {
            PieLoaderRepackLayout::None => Err(()),
            PieLoaderRepackLayout::MarlinMxfp4Weight => Ok(Self::MarlinMxfp4Weight),
            PieLoaderRepackLayout::MarlinMxfp4Scale => Ok(Self::MarlinMxfp4Scale),
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
            12 => Self::E8M0,
            13 => Self::I64,
            14 => Self::U64,
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
            14 => Self::Int4B8,
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
            other => return Err(other),
        })
    }
}

/// `None` is the resting value for instructions that carry no tile map, so it
/// sorts after the transforms that existed when it was chosen — matching the
/// C++ enum and the default on `PieLoaderStorageInstrView::tile_kind`.
///
/// These discriminants are not the `TILE_MAP_*` capability bits and are not
/// required to agree with them: the two happen to line up below `None` and stop
/// there, because a kind added afterwards has to take a free discriminant while
/// its bit is chosen from the free bits. The loader's `TileMapKind` states the
/// pairing in one place, by name, and nothing derives one from the other.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderTileMapKind {
    Cast = 0,
    Decode = 1,
    Encode = 2,
    Transcode = 3,
    Reblock = 4,
    Repack = 6,
    None = 7,
    Scale = 8,
}

impl From<pie_loader::plan::TileMapKind> for PieLoaderTileMapKind {
    fn from(value: pie_loader::plan::TileMapKind) -> Self {
        use pie_loader::plan::TileMapKind as K;
        match value {
            K::Cast => Self::Cast,
            K::Decode => Self::Decode,
            K::Encode => Self::Encode,
            K::Transcode => Self::Transcode,
            K::Reblock => Self::Reblock,
            K::Repack => Self::Repack,
            K::Scale => Self::Scale,
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

impl From<pie_loader::plan::TransformFusion> for PieLoaderTransformFusion {
    fn from(value: pie_loader::plan::TransformFusion) -> Self {
        use pie_loader::plan::TransformFusion as F;
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

pub type PieLoaderU32Slice = PieLoaderSlice<u32>;

pub type PieLoaderI64Slice = PieLoaderSlice<i64>;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderDimSpecView {
    pub count: i64,
    pub src_stride: i64,
    pub dst_stride: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderSlice<T> {
    pub ptr: *const T,
    pub len: usize,
}

pub type PieLoaderDimSpecSlice = PieLoaderSlice<PieLoaderDimSpecView>;

impl<T> Default for PieLoaderSlice<T> {
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
    /// The type these bytes are read as. Not necessarily
    /// `PieLoaderPlan::sources[tensor_id].dtype`: a contract that reinterprets
    /// a tensor with `Transmute` -- DeepSeek-V4's E8M0 block scales are stored
    /// as `U8` -- says so here, and an executor that consulted the source
    /// table instead would undo the reinterpretation.
    pub dtype: PieLoaderDType,
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
    pub visibility: PieLoaderVisibility,
}

pub type PieLoaderTensorDeclSlice = PieLoaderSlice<PieLoaderTensorDeclView>;

/// Whether a declared tensor is bound by the driver. Mirrors [`Visibility`](pie_loader::types::Visibility).
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PieLoaderVisibility {
    /// A runtime weight, bound by name.
    #[default]
    Public = 0,
    /// A name the contract needed for itself: not bound, not persistent.
    Internal = 1,
}

/// How a scale tensor's entries map onto the tensor they scale.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PieLoaderQuantGranularity {
    #[default]
    PerChannel = 0,
    PerGroup = 1,
}

/// What the driver's kernels expect a scale tensor to hold when they read it.
///
/// Not derivable from the scale tensor: its dtype says how the bytes are stored,
/// not how the kernel wants them. The driver used to infer this from
/// `group_size == 32`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PieLoaderScaleForm {
    /// Raw E8M0 exponent bytes, consumed as-is.
    #[default]
    RawE8M0 = 0,
    /// F32 multipliers; expand before the GEMM sees them.
    F32Factors = 1,
    /// BF16 multipliers paired with the zero points named by
    /// `zero_point_tensor_id`: an element is `code * scale + zero`.
    Bf16AffineFactors = 2,
}

impl From<QuantGranularity> for PieLoaderQuantGranularity {
    fn from(value: QuantGranularity) -> Self {
        match value {
            QuantGranularity::PerChannel => Self::PerChannel,
            QuantGranularity::PerGroup => Self::PerGroup,
        }
    }
}

impl From<PieLoaderQuantGranularity> for QuantGranularity {
    fn from(value: PieLoaderQuantGranularity) -> Self {
        match value {
            PieLoaderQuantGranularity::PerChannel => Self::PerChannel,
            PieLoaderQuantGranularity::PerGroup => Self::PerGroup,
        }
    }
}

impl TryFrom<u32> for PieLoaderQuantGranularity {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::PerChannel,
            1 => Self::PerGroup,
            other => return Err(other),
        })
    }
}

impl From<ScaleForm> for PieLoaderScaleForm {
    fn from(value: ScaleForm) -> Self {
        match value {
            ScaleForm::RawE8M0 => Self::RawE8M0,
            ScaleForm::F32Factors => Self::F32Factors,
            ScaleForm::Bf16AffineFactors => Self::Bf16AffineFactors,
        }
    }
}

impl From<PieLoaderScaleForm> for ScaleForm {
    fn from(value: PieLoaderScaleForm) -> Self {
        match value {
            PieLoaderScaleForm::RawE8M0 => Self::RawE8M0,
            PieLoaderScaleForm::F32Factors => Self::F32Factors,
            PieLoaderScaleForm::Bf16AffineFactors => Self::Bf16AffineFactors,
        }
    }
}

impl TryFrom<u32> for PieLoaderScaleForm {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::RawE8M0,
            1 => Self::F32Factors,
            2 => Self::Bf16AffineFactors,
            other => return Err(other),
        })
    }
}

/// A quantized tensor paired with the tensor holding its scales.
///
/// Both are entries in [`PieLoaderPlan::tensors`], named by `id`. The driver has
/// to know the pairing in order to attach the quant metadata its kernels read;
/// it used to rediscover it by matching name suffixes over the tensor list,
/// which guessed at something stated here.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderQuantAttachmentView {
    pub tensor_id: u32,
    pub scale_tensor_id: u32,
    /// The tensor holding this weight's zero points, for an affine scheme, or
    /// [`PIE_LOADER_NO_TENSOR`] for a symmetric one. An affine weight without it
    /// cannot be dequantized, so a driver reading a `Bf16AffineFactors`
    /// attachment is entitled to treat the sentinel as a malformed plan rather
    /// than as an offset of zero.
    pub zero_point_tensor_id: u32,
    pub granularity: PieLoaderQuantGranularity,
    pub group_size: u32,
    pub channel_axis: u32,
    pub scale_form: PieLoaderScaleForm,
}

pub type PieLoaderQuantAttachmentSlice = PieLoaderSlice<PieLoaderQuantAttachmentView>;

/// Where one instruction of a group's plan reads, for one instance.
///
/// The whole of what distinguishes instance `i` from instance 0: a group did
/// not compile unless every other field of every instruction agreed across all
/// of them. `instr_id` names the instruction in
/// [`PieLoaderGroupView::plan`](PieLoaderGroupView) whose source extent these
/// three fields replace.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieLoaderSourceBindingView {
    pub instr_id: u32,
    pub file_id: u32,
    pub tensor_id: u32,
    pub file_offset: u64,
}

pub type PieLoaderSourceBindingSlice = PieLoaderSlice<PieLoaderSourceBindingView>;

/// A set of interchangeable tensors: one plan, `arity` instances.
///
/// The driver decides what to do with it. Running `plan` `arity` times into
/// `arity` destinations makes the group resident, which is what a driver that
/// has never heard of streaming should do; running it on demand into a bounded
/// set of slots is streaming. The loader states only that the two are the same
/// program.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderGroupView {
    pub name: PieLoaderBytes,
    pub arity: u32,
    /// The program one instance runs, compiled at index 0. A whole plan, so it
    /// goes to the same executor the resident load already uses.
    pub plan: *const PieLoaderPlan,
    /// `arity * bindings_per_instance` entries, instance-major. Instance `i`
    /// owns `[i * bindings_per_instance, (i + 1) * bindings_per_instance)`.
    pub bindings: PieLoaderSourceBindingSlice,
    /// How many instructions of `plan` name a source. Zero only if the group's
    /// plan reads nothing, which no group does.
    pub bindings_per_instance: usize,
}

pub type PieLoaderGroupSlice = PieLoaderSlice<PieLoaderGroupView>;

/// Which on-disk format a checkpoint file uses.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderCheckpointFormat {
    Safetensors = 0,
    Gguf = 1,
    Unknown = 2,
    /// Appended: adding a value in the middle would renumber `Unknown` under
    /// every driver already compiled against this header.
    Zt = 3,
    Npz = 4,
    Pt = 5,
    Hdf5 = 6,
    Onnx = 7,
}

impl From<pie_loader::types::CheckpointFormat> for PieLoaderCheckpointFormat {
    fn from(value: pie_loader::types::CheckpointFormat) -> Self {
        match value {
            pie_loader::types::CheckpointFormat::Safetensors => Self::Safetensors,
            pie_loader::types::CheckpointFormat::Gguf => Self::Gguf,
            pie_loader::types::CheckpointFormat::Unknown => Self::Unknown,
            pie_loader::types::CheckpointFormat::Zt => Self::Zt,
            pie_loader::types::CheckpointFormat::Npz => Self::Npz,
            pie_loader::types::CheckpointFormat::Pt => Self::Pt,
            pie_loader::types::CheckpointFormat::Hdf5 => Self::Hdf5,
            pie_loader::types::CheckpointFormat::Onnx => Self::Onnx,
        }
    }
}

/// One file the plan reads from. `PieLoaderSourceTensorView::file_id` indexes
/// `PieLoaderPlan::files`, so the driver no longer has to re-derive the file
/// order for itself (`architecture.md` §6).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderCheckpointFileView {
    pub id: u32,
    pub path: PieLoaderBytes,
    pub size_bytes: u64,
    pub format: PieLoaderCheckpointFormat,
}

pub type PieLoaderCheckpointFileSlice = PieLoaderSlice<PieLoaderCheckpointFileView>;

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

pub type PieLoaderSourceTensorSlice = PieLoaderSlice<PieLoaderSourceTensorView>;

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

pub type PieLoaderBufferDeclSlice = PieLoaderSlice<PieLoaderBufferDeclView>;

/// One entry of the plan's instruction stream: an identity, and an operation.
///
/// `id` is the schedule's handle on this instruction and is the only thing every
/// instruction has. Everything else belongs to one operation, and lives inside
/// it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderStorageInstrView {
    pub id: u32,
    pub op: PieLoaderStorageOp,
}

/// What an instruction does, as a tagged union carrying only that operation's
/// operands.
///
/// This mirrors `pie_loader::plan::StorageInstr` variant for variant. It was a flat
/// struct of 32 members with a `kind` tag until every reader had grown a
/// defence against the members that tag left meaningless: `if (!instr.has_source
/// || !instr.has_dest)` in three executors, `inputs.size() != 1` around a
/// `CreateView` whose input count is one by construction, and a comment in
/// `ffi::view` explaining that a resting `source` "would look like a valid
/// reference to file 0". Those are invariants a union states and a product type
/// can only apologise for.
///
/// The discriminants are the wire tag and are written out for the same reason
/// the mirror enums' are — 4 and 7 are absent because retired instructions had
/// them, and renumbering to close a gap would silently move every tag above it.
#[repr(C, u32)]
#[derive(Clone, Copy, Debug)]
pub enum PieLoaderStorageOp {
    Allocate {
        buffer_id: u32,
    } = 0,
    ExtentWrite {
        source: PieLoaderSourceExtentView,
        dest: PieLoaderDestExtentView,
    } = 1,
    TileMap {
        tile_kind: PieLoaderTileMapKind,
        /// A tile map reads a checkpoint tensor, or transforms a buffer already
        /// on the device. `has_source` says which; the other seven operations no
        /// longer have to carry the question.
        source: PieLoaderSourceExtentView,
        has_source: bool,
        dest: PieLoaderDestExtentView,
        has_dest: bool,
        input_buffers: PieLoaderU32Slice,
        output_buffers: PieLoaderU32Slice,
        /// Rows of the output to transform per launch; `0` means the whole
        /// tensor in one pass.
        ///
        /// This is where the driver's `max_tile_bytes` budget ends up. The
        /// budget itself does not cross the boundary: the driver stated it in
        /// the request, the loader answered with a row count in
        /// `backend::lower`, and sending the question back alongside the answer
        /// would only invite the executor to re-derive it (`architecture.md`
        /// §8.1).
        rows_per_tile: u32,
        /// A transform chain the backend collapsed into one kernel.
        transform_fusion: PieLoaderTransformFusion,
        transform_from: PieLoaderQuantScheme,
        transform_to: PieLoaderQuantScheme,
        repack_layout: PieLoaderRepackLayout,
        transform_batch: u32,
        transform_source_rows: u32,
        transform_target_rows: u32,
        transform_source_cols: u32,
        transform_target_cols: u32,
        transform_scratch_bytes: u64,
        /// Source tensor holding this transform's input block scales, or
        /// [`PIE_LOADER_NO_TENSOR`].
        ///
        /// Index into `PieLoaderPlan::sources`, the same space
        /// `source.tensor_id` uses, so the executor reaches the scales' name and
        /// shape the way it reaches the payload's — instead of appending
        /// `_scale_inv` to a name and hoping the checkpoint agrees.
        transform_metadata_source: u32,
        /// The multiplier for a [`PieLoaderTileMapKind::Scale`], as the bit
        /// pattern of an IEEE-754 binary32; zero on every other kind.
        ///
        /// Bits rather than a `float` field so this union's layout does not
        /// depend on float ABI, and so the executor multiplies with exactly the
        /// constant the contract named — `__uint_as_float` on the CUDA side
        /// costs nothing and cannot round.
        transform_scale_factor_bits: u32,
        /// Elements of the operand per factor, on each axis, for a per-block
        /// [`PieLoaderTileMapKind::Scale`]; empty when the factor is the
        /// uniform constant above.
        ///
        /// Non-empty is what tells the executor to read its factors from
        /// `input_buffers[0]` — the operand the contract paired with the
        /// payload — instead of from `transform_scale_factor_bits`.
        ///
        /// One entry per axis of the destination, so a DeepSeek-style FP8
        /// checkpoint's two-dimensional block scale is `[128, 128]` and the
        /// ordinary row-wise case is `[1, 32]`. The executor already knows both
        /// shapes, so this is a statement it can check rather than one it must
        /// trust.
        transform_scale_blocks: PieLoaderI64Slice,
    } = 2,
    CreateView {
        input_buffer: u32,
        output_buffer: u32,
        view: PieLoaderDestExtentView,
    } = 3,
    Finalize {
        buffer_id: u32,
        name: PieLoaderBytes,
    } = 5,
    /// The destination is an offset into the persistent arena, not a buffer.
    /// The flat form had to fabricate a rank-1 `dest` extent per instruction —
    /// an arena allocation whose only content the executor ever read was
    /// `dest.offset`.
    BulkExtentWrite {
        source: PieLoaderSourceExtentView,
        dest_offset: u64,
    } = 6,
    Fill {
        buffer_id: u32,
    } = 8,
}

pub type PieLoaderStorageInstrSlice = PieLoaderSlice<PieLoaderStorageInstrView>;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderMemoryPlanView {
    pub persistent_bytes: u64,
    pub temporary_peak_bytes: u64,
    pub transform_scratch_peak_bytes: u64,
    pub checkpoint_read_bytes: u64,
    pub device_write_bytes: u64,
}

/// The target the plan was compiled against. The driver reads it back to assert
/// the plan it received is the plan it asked for — the same fields it supplied
/// in the request, plus the rank identity that makes a TP shard distinguishable
/// from its siblings (`architecture.md` §6.2).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderTargetView {
    pub backend: PieLoaderBackendKind,
    pub tp_rank: u32,
    pub tp_size: u32,
    pub max_tile_bytes: u64,
    pub preferred_alignment: u32,
    pub tile_map_mask: u32,
    pub native_mxfp4_moe: bool,
    pub fusion_mask: u32,
    pub encode_scratch_dtype: PieLoaderDType,
    pub block_scale_rows: u32,
}

impl From<&pie_loader::plan::StorageTarget> for PieLoaderTargetView {
    fn from(value: &pie_loader::plan::StorageTarget) -> Self {
        Self {
            backend: value.backend.into(),
            tp_rank: value.tp_rank,
            tp_size: value.tp_size,
            max_tile_bytes: value.max_tile_bytes,
            preferred_alignment: value.preferred_alignment,
            tile_map_mask: value.tile_map_mask,
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
/// through `LoadPlan` methods (`backend()`, `native_mxfp4_moe()`,
/// `preferred_alignment()`, `max_tile_bytes()`, `tile_map_mask()`,
/// `compiler_version()`), which have no method syntax to hide behind once the
/// type is POD.
///
/// `owner` is the opaque handle to the arena keeping every slice above alive. It
/// is consumed by `pie_loader_release`; the driver must not dereference it.
#[repr(C)]
#[derive(Debug)]
pub struct PieLoaderPlan {
    pub files: PieLoaderCheckpointFileSlice,
    pub sources: PieLoaderSourceTensorSlice,
    pub tensors: PieLoaderTensorDeclSlice,
    pub buffers: PieLoaderBufferDeclSlice,
    pub instrs: PieLoaderStorageInstrSlice,
    pub schedule: PieLoaderU32Slice,
    pub memory: PieLoaderMemoryPlanView,
    pub compiler_version: u64,
    pub target: PieLoaderTargetView,
    pub attachments: PieLoaderQuantAttachmentSlice,
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
    /// Interchangeable tensor sets, each compiled once. Empty for a plan whose
    /// contract declared no group, which is every contract that predates them.
    pub groups: PieLoaderGroupSlice,
    pub owner: *mut std::ffi::c_void,
}

/// The ABI's bits and the plan's are the same bits.
///
/// A build error here means someone added a transform on one side only, which
/// is the drift `PieLoaderTileMapKind::capability_bit` would otherwise turn
/// into a mis-dispatched kernel.
const _: () = {
    use pie_loader::plan as p;
    assert!(PIE_LOADER_TILE_MAP_CAST == p::TILE_MAP_CAST);
    assert!(PIE_LOADER_TILE_MAP_DECODE == p::TILE_MAP_DECODE);
    assert!(PIE_LOADER_TILE_MAP_ENCODE == p::TILE_MAP_ENCODE);
    assert!(PIE_LOADER_TILE_MAP_TRANSCODE == p::TILE_MAP_TRANSCODE);
    assert!(PIE_LOADER_TILE_MAP_REBLOCK == p::TILE_MAP_REBLOCK);
    assert!(PIE_LOADER_TILE_MAP_REPACK == p::TILE_MAP_REPACK);
    assert!(PIE_LOADER_TILE_MAP_SCALE == p::TILE_MAP_SCALE);
    assert!(PIE_LOADER_FUSION_FP8_TO_MXFP4 == p::FUSION_FP8_TO_MXFP4);
};

impl From<PieLoaderVisibility> for Visibility {
    fn from(value: PieLoaderVisibility) -> Self {
        match value {
            PieLoaderVisibility::Public => Self::Public,
            PieLoaderVisibility::Internal => Self::Internal,
        }
    }
}

/// `QuantSpec::channel_axis`, as a signed integer so that `-1` is "unset".
///
/// A sentinel rather than a second `bool` field keeps the struct a plain list
/// of numbers, which is what a designated initializer is good at.
pub const PIE_LOADER_NO_AXIS: i32 = -1;

/// [`pie_loader::types::QuantSpec`], flattened.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderQuantSpecView {
    /// A `PieLoaderQuantScheme` value, as `uint32_t`.
    pub scheme: u32,
    /// A `PieLoaderDType` value, as `uint32_t`.
    pub logical_dtype: u32,
    /// `0` asks for the scheme's default, exactly as the Rust side does.
    pub bits_per_element: u8,
    /// `0` asks for the scheme's default.
    pub group_size: u32,
    pub channel_axis: i32,
}

impl Default for PieLoaderQuantSpecView {
    fn default() -> Self {
        Self {
            scheme: PieLoaderQuantScheme::None as u32,
            logical_dtype: PieLoaderDType::BF16 as u32,
            bits_per_element: 0,
            group_size: 0,
            channel_axis: PIE_LOADER_NO_AXIS,
        }
    }
}

/// [`pie_loader::types::Encoding`], flattened. `kind` selects which half is read.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderEncodingSpec {
    /// A `PieLoaderEncodingKind` value, as `uint32_t`.
    pub kind: u32,
    /// A `PieLoaderDType` value, as `uint32_t`. Read when `kind == Raw`.
    pub dtype: u32,
    /// Read when `kind == Quant`.
    pub quant: PieLoaderQuantSpecView,
}

impl Default for PieLoaderEncodingSpec {
    fn default() -> Self {
        Self {
            kind: PieLoaderEncodingKind::Raw as u32,
            dtype: PieLoaderDType::BF16 as u32,
            quant: PieLoaderQuantSpecView::default(),
        }
    }
}

pub(crate) fn write_quant(spec: &QuantSpec) -> PieLoaderQuantSpecView {
    PieLoaderQuantSpecView {
        scheme: PieLoaderQuantScheme::from(spec.scheme) as u32,
        logical_dtype: PieLoaderDType::from(spec.logical_dtype) as u32,
        bits_per_element: spec.bits_per_element,
        group_size: spec.group_size,
        channel_axis: spec
            .channel_axis
            .map_or(PIE_LOADER_NO_AXIS, |axis| i32::from(axis.0)),
    }
}

pub(crate) fn write_encoding(encoding: &Encoding) -> PieLoaderEncodingSpec {
    let mut spec = PieLoaderEncodingSpec::default();
    match encoding {
        Encoding::Raw(dtype) => {
            spec.kind = PieLoaderEncodingKind::Raw as u32;
            spec.dtype = PieLoaderDType::from(*dtype) as u32;
        }
        Encoding::Quant(quant) => {
            spec.kind = PieLoaderEncodingKind::Quant as u32;
            spec.quant = write_quant(quant);
        }
    }
    spec
}
