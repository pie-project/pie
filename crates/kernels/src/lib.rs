pub mod bind;
pub mod routine;

pub mod jit;

pub mod canon;
pub mod points;
pub mod raises;
pub mod runtime;

pub mod shader;

pub use routine::{Answers, Arg, Asks, Backend, KernelFn, Refusal, Routine};

pub use routine::{Elem, Layout, Region, Stride};

pub use routine::{Addressed, Bind, BindMut, Fire, Geometry, Grid};

pub use routine::{Cache, Const, ConstRun, In, InOut, Out};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    Scores,

    PageMaskSink,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutRule {
    #[default]
    Unstated,

    Like {
        of: u8,
    },

    Shaped {
        rows_of: u8,

        width: OutWidth,
    },

    Split {
        of: u8,

        dim_param: u8,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutWidth {
    Half { of: u8 },

    Of { of: u8 },

    Weight { of: u8 },

    Param { of: u8 },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LaunchRule {
    #[default]
    Unstated,

    Qmv,

    Rms,

    Rope,

    Elementwise,

    ElementwiseRows,

    PerHead,

    SdpaVector,

    SdpaTiled,

    SdpaMma,

    PerHeadElementwise,

    GatedRms,

    RouterLane,

    RouterSort,

    RouteRows,

    RoutedQmv,

    SplitPacked,

    Qmm,

    RecurrentScan,

    PerRow,

    PerChannel,

    ElementwiseIn,

    RowScores,

    RowsPerHead,

    RowsFlat,

    Slab,

    Tile16,

    AxialRope,

    WarpTiledScan,

    PerRowNarrow,

    PagedScores,

    PagedScoresDecode,

    MlaPrepare,

    RowsPackedHeads,

    RowsPackedHeadsNarrow,

    WarpPackedHeads,

    RoutedQmvTransposed,

    AltUpStreams,

    RoutedQmvQuad,

    Single,

    SingleWarp,

    PerRequest,
}

impl LaunchRule {
    pub const ALL: &'static [Self] = &[
        Self::Unstated,
        Self::Qmv,
        Self::Rms,
        Self::Rope,
        Self::Elementwise,
        Self::ElementwiseRows,
        Self::PerHead,
        Self::SdpaVector,
        Self::SdpaTiled,
        Self::SdpaMma,
        Self::PerHeadElementwise,
        Self::GatedRms,
        Self::RouterLane,
        Self::RouterSort,
        Self::RouteRows,
        Self::RoutedQmv,
        Self::SplitPacked,
        Self::Qmm,
        Self::RecurrentScan,
        Self::PerRow,
        Self::PerChannel,
        Self::ElementwiseIn,
        Self::RowScores,
        Self::RowsPerHead,
        Self::RowsFlat,
        Self::Slab,
        Self::Tile16,
        Self::AxialRope,
        Self::WarpTiledScan,
        Self::PerRowNarrow,
        Self::PagedScores,
        Self::PagedScoresDecode,
        Self::MlaPrepare,
        Self::RowsPackedHeads,
        Self::RowsPackedHeadsNarrow,
        Self::WarpPackedHeads,
        Self::RoutedQmvTransposed,
        Self::AltUpStreams,
        Self::RoutedQmvQuad,
        Self::Single,
        Self::SingleWarp,
        Self::PerRequest,
    ];

    const fn index(self) -> usize {
        match self {
            Self::Unstated => 0,
            Self::Qmv => 1,
            Self::Rms => 2,
            Self::Rope => 3,
            Self::Elementwise => 4,
            Self::ElementwiseRows => 5,
            Self::PerHead => 6,
            Self::SdpaVector => 7,
            Self::SdpaTiled => 8,
            Self::SdpaMma => 9,
            Self::PerHeadElementwise => 10,
            Self::GatedRms => 11,
            Self::RouterLane => 12,
            Self::RouterSort => 13,
            Self::RouteRows => 14,
            Self::RoutedQmv => 15,
            Self::SplitPacked => 16,
            Self::Qmm => 17,
            Self::RecurrentScan => 18,
            Self::PerRow => 19,
            Self::PerChannel => 20,
            Self::ElementwiseIn => 21,
            Self::RowScores => 22,
            Self::RowsPerHead => 23,
            Self::RowsFlat => 24,
            Self::Slab => 25,
            Self::Tile16 => 26,
            Self::AxialRope => 27,
            Self::WarpTiledScan => 28,
            Self::PerRowNarrow => 29,
            Self::PagedScores => 30,
            Self::PagedScoresDecode => 31,
            Self::MlaPrepare => 32,
            Self::RowsPackedHeads => 33,
            Self::RowsPackedHeadsNarrow => 34,
            Self::WarpPackedHeads => 35,
            Self::RoutedQmvTransposed => 36,
            Self::AltUpStreams => 37,
            Self::RoutedQmvQuad => 38,
            Self::Single => 39,
            Self::SingleWarp => 40,
            Self::PerRequest => 41,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis {
    pub what: &'static str,

    pub points: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    BufMut,

    Buf,

    I32s,

    I64s,

    U32s,

    U8s,

    F32sMut,

    F32s,

    I32sMut,

    U32sMut,

    U8sMut,

    I32,

    U32,

    BufArray,

    BufArrayMut,

    BufArrayOut,

    BufArrayOutMut,

    U8Array,

    U16s,

    U16sMut,

    I8s,

    I8sMut,

    Bf16s,

    F16s,

    Bf16sMut,

    F16sMut,

    I32Array,

    MoeActivation,

    Mxfp4RowSelect,

    CustomAllReduce,

    Dtype,

    KvScheme,

    KvDType,

    Fp8Kind,

    I64,

    Usize,

    InPacked,

    F32,

    Bool,

    Stream,

    CublasHandle,

    AttentionWorkspaceView,

    KvCacheLayerView,

    MlaCacheLayerView,

    DecodePlanCache,

    PrefillPlanCache,

    MlaPlanCache,

    HopperPrefillPlan,

    YarnOriginalParams,

    Raised,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binds {
    Reads,

    Writes,

    Nothing,
}

impl Ty {
    pub const fn cpp(self) -> &'static str {
        match self {
            Ty::InPacked => "::std::uint32_t",
            Ty::BufMut => "void*",
            Ty::Buf => "const void*",
            Ty::I32s => "const ::std::int32_t*",
            Ty::I64s => "const ::std::int64_t*",
            Ty::BufArray => "const void* const*",
            Ty::BufArrayMut => "void* const*",
            Ty::BufArrayOut => "const void**",
            Ty::BufArrayOutMut => "void**",
            Ty::U8Array => "const ::std::uint8_t* const*",
            Ty::CustomAllReduce => "::pie::comm::CustomAllReduce*",
            Ty::I8s => "const ::std::int8_t*",
            Ty::I8sMut => "::std::int8_t*",
            Ty::Bf16s => "const ::pie::bf16*",
            Ty::F16s => "const ::pie::f16*",
            Ty::Bf16sMut => "::pie::bf16*",
            Ty::F16sMut => "::pie::f16*",
            Ty::I32Array => "const ::std::int32_t* const*",
            Ty::MoeActivation => "::pie::moe::MoeActivation",
            Ty::Mxfp4RowSelect => "::pie::quant::Mxfp4RowSelect",
            Ty::U16s => "const ::std::uint16_t*",
            Ty::U16sMut => "::std::uint16_t*",
            Ty::Dtype => "::pie_cuda_driver::DType",
            Ty::KvScheme => "::pie::attn::KvScheme",
            Ty::KvDType => "::pie::attn::KvDType",

            Ty::Fp8Kind => "::__nv_fp8_interpretation_t",
            Ty::I64 => "long long",
            Ty::U32s => "const ::std::uint32_t*",
            Ty::U8s => "const ::std::uint8_t*",
            Ty::F32sMut => "float*",
            Ty::F32s => "const float*",
            Ty::I32sMut => "::std::int32_t*",
            Ty::U32sMut => "::std::uint32_t*",
            Ty::U8sMut => "::std::uint8_t*",
            Ty::I32 => "int",
            Ty::U32 => "::std::uint32_t",
            Ty::Usize => "::std::size_t",
            Ty::F32 => "float",
            Ty::Bool => "bool",
            Ty::Stream => "cudaStream_t",
            Ty::CublasHandle => "cublasHandle_t",
            Ty::AttentionWorkspaceView => "::pie_cuda_driver::AttentionWorkspaceView",
            Ty::KvCacheLayerView => "::pie_cuda_driver::KvCacheLayerView",
            Ty::MlaCacheLayerView => "::pie_cuda_driver::MlaCacheLayerView",
            Ty::DecodePlanCache => "const ::pie::attn::DecodePlanCache&",
            Ty::PrefillPlanCache => "const ::pie::attn::PrefillPlanCache&",
            Ty::MlaPlanCache => "const ::pie::attn::MlaPlanCache&",
            Ty::HopperPrefillPlan => "const ::pie::attn::HopperPrefillPlan&",
            Ty::YarnOriginalParams => "const ::pie::attn::YarnOriginalParams*",

            Ty::Raised => "",
        }
    }

    pub const fn rust(self) -> &'static str {
        match self {
            Ty::InPacked => "u32",
            Ty::BufMut => "*mut ::core::ffi::c_void",
            Ty::Buf => "*const ::core::ffi::c_void",
            Ty::I32s => "*const i32",
            Ty::I64s => "*const i64",
            Ty::BufArray => "*const *const ::core::ffi::c_void",
            Ty::BufArrayMut => "*const *mut ::core::ffi::c_void",
            Ty::BufArrayOut => "*mut *const ::core::ffi::c_void",
            Ty::BufArrayOutMut => "*mut *mut ::core::ffi::c_void",
            Ty::U8Array => "*const *const u8",
            Ty::CustomAllReduce => "*mut ::core::ffi::c_void",
            Ty::I8s => "*const i8",
            Ty::I8sMut => "*mut i8",

            Ty::Bf16s | Ty::F16s => "*const u16",
            Ty::Bf16sMut | Ty::F16sMut => "*mut u16",
            Ty::I32Array => "*const *const i32",
            Ty::MoeActivation => "u32",
            Ty::Mxfp4RowSelect => "i32",
            Ty::U16s => "*const u16",
            Ty::U16sMut => "*mut u16",
            Ty::Dtype => "u8",

            Ty::KvScheme | Ty::KvDType => "u8",

            Ty::Fp8Kind => "u32",
            Ty::I64 => "::core::ffi::c_longlong",
            Ty::U32s => "*const u32",
            Ty::U8s => "*const u8",
            Ty::F32sMut => "*mut f32",
            Ty::F32s => "*const f32",
            Ty::I32sMut => "*mut i32",
            Ty::U32sMut => "*mut u32",
            Ty::U8sMut => "*mut u8",
            Ty::I32 => "::core::ffi::c_int",
            Ty::U32 => "u32",
            Ty::Usize => "usize",
            Ty::F32 => "f32",
            Ty::Bool => "bool",
            Ty::Stream | Ty::CublasHandle => "*mut ::core::ffi::c_void",

            Ty::AttentionWorkspaceView => "AttentionWorkspaceView",
            Ty::KvCacheLayerView => "KvCacheLayerView",
            Ty::MlaCacheLayerView => "MlaCacheLayerView",

            Ty::DecodePlanCache | Ty::PrefillPlanCache | Ty::MlaPlanCache => {
                "*const ::core::ffi::c_void"
            }
            Ty::HopperPrefillPlan => "*const HopperPrefillPlan",
            Ty::YarnOriginalParams => "*const YarnOriginalParams",

            Ty::Raised => "*const ::core::ffi::c_void",
        }
    }

    #[must_use]
    pub const fn binds(self) -> Binds {
        match self {
            Ty::BufMut
            | Ty::F32sMut
            | Ty::I32sMut
            | Ty::U32sMut
            | Ty::U8sMut
            | Ty::U16sMut
            | Ty::I8sMut
            | Ty::Bf16sMut
            | Ty::F16sMut
            | Ty::BufArrayMut
            | Ty::BufArrayOutMut => Binds::Writes,

            Ty::Buf
            | Ty::I32s
            | Ty::I64s
            | Ty::U32s
            | Ty::U8s
            | Ty::F32s
            | Ty::U16s
            | Ty::I8s
            | Ty::Bf16s
            | Ty::F16s
            | Ty::BufArray
            | Ty::BufArrayOut
            | Ty::U8Array
            | Ty::I32Array => Binds::Reads,

            Ty::I32
            | Ty::U32
            | Ty::I64
            | Ty::Usize
            | Ty::F32
            | Ty::Bool
            | Ty::InPacked
            | Ty::MoeActivation
            | Ty::Mxfp4RowSelect
            | Ty::Dtype
            | Ty::KvScheme
            | Ty::KvDType
            | Ty::Fp8Kind
            | Ty::Stream
            | Ty::CublasHandle
            | Ty::CustomAllReduce
            | Ty::AttentionWorkspaceView
            | Ty::KvCacheLayerView
            | Ty::MlaCacheLayerView
            | Ty::DecodePlanCache
            | Ty::PrefillPlanCache
            | Ty::MlaPlanCache
            | Ty::HopperPrefillPlan
            | Ty::YarnOriginalParams => Binds::Nothing,

            Ty::Raised => Binds::Reads,
        }
    }

    #[must_use]
    pub const fn needs_mirror(self) -> bool {
        matches!(
            self,
            Ty::AttentionWorkspaceView
                | Ty::KvCacheLayerView
                | Ty::MlaCacheLayerView
                | Ty::HopperPrefillPlan
                | Ty::YarnOriginalParams
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Derived {
    pub name: &'static str,

    pub nullable: bool,
}

pub trait Derivation {
    const DERIVED: &'static [Derived];

    const SOURCES: &'static [Option<Source>];
}

pub trait Signature {
    const NAMESPACE: &'static str;

    const NAME: &'static str;

    type Sig;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Kind {
    In,

    Out,

    Weight,

    Param,

    ParamF32,

    Aux,

    OutWidth,

    InWidth,

    OutElements,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    Lit(Lit),

    Slot(Kind, u8),

    Or(&'static Source, &'static Source),

    Times(&'static Source, &'static Source),

    Alias(u8, u8),

    Over(&'static Source, &'static Source),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lit {
    Null,

    Bool(bool),

    F32(f32),

    I32(i32),
}

pub struct KernelSig {
    pub name: &'static str,

    pub symbol: &'static str,

    pub whole: bool,

    pub depth_prefix_plan: bool,

    pub args: &'static [Ty],

    pub sources: &'static [Option<Source>],

    pub derived: &'static [Derived],

    pub axes: &'static [Axis],

    pub internal: bool,

    pub no_join: bool,

    pub driver: bool,

    pub canon: Option<&'static str>,

    pub point: &'static [&'static str],

    pub out_rule: &'static [OutRule],
}

impl KernelSig {
    pub fn covers_point(&self, symbol: &str) -> bool {
        if self.axes.is_empty() {
            return false;
        }
        let mut rest = symbol;
        for axis in self.axes.iter().rev() {
            match axis
                .points
                .iter()
                .find(|point| rest.len() > point.len() && rest.ends_with(**point))
            {
                Some(point) => rest = &rest[..rest.len() - point.len()],
                None => return false,
            }
        }
        rest == self.symbol
    }

    pub fn entrypoints(&self) -> Vec<String> {
        let mut out = vec![self.symbol.to_string()];
        for axis in self.axes {
            out = out
                .iter()
                .flat_map(|stem| {
                    axis.points
                        .iter()
                        .map(move |point| format!("{stem}{point}"))
                })
                .collect();
        }
        out
    }
}

pub fn sig_in(table: &'static [KernelSig], symbol: &str) -> Option<&'static KernelSig> {
    table
        .iter()
        .find(|k| k.symbol == symbol)
        .or_else(|| table.iter().find(|k| k.covers_point(symbol)))
}
