#![allow(clippy::missing_safety_doc)]

pub mod bind;
pub mod bound;
pub mod plane;

pub mod jit;

pub mod points;
pub mod raises;

pub mod shader;

pub use plane::{Answers, Arg, Asks, Backend, Refusal};

pub use plane::{Elem, Region, Stride};

pub use plane::{Addressed, Bind, BindMut, Fire, Geometry, Grid};

pub use plane::{Cache, Const, ConstRun, In, InOut, Out};

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

    KvScheme,

    KvDType,

    Fp8Kind,

    I64,

    Usize,

    InPacked,

    F32,

    Bool,

    Stream,

    KvCacheLayerView,

    DecodePlanCache,

    PrefillPlanCache,

    MlaPlanCache,

    Raised,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binds {
    Reads,

    Writes,

    Nothing,
}

impl Ty {
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
            | Ty::KvScheme
            | Ty::KvDType
            | Ty::Fp8Kind
            | Ty::Stream
            | Ty::KvCacheLayerView
            | Ty::DecodePlanCache
            | Ty::PrefillPlanCache
            | Ty::MlaPlanCache => Binds::Nothing,

            Ty::Raised => Binds::Reads,
        }
    }

    #[must_use]
    pub const fn needs_mirror(self) -> bool {
        matches!(self, Ty::KvCacheLayerView)
    }
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
