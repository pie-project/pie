use alloc::vec;
use alloc::vec::Vec;

use crate::types::{Dtype, Literal, Predicate, RngKind, Shape, ValueId};

pub type ChannelIndex = u32;
pub type NameIndex = u16;

macro_rules! declare_intrinsics {
    ($($(#[$doc:meta])* $variant:ident = $id:literal, $konst:ident, $name:literal;)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[repr(u16)]
        pub enum IntrinsicId {
            $($(#[$doc])* $variant = $id,)*
        }

        pub mod intrinsic_tags {
            $(
                #[doc = concat!("Wire id of the `", $name, "` intrinsic.")]
                pub const $konst: u16 = $id;
            )*
        }

        impl IntrinsicId {
            pub const ALL: &'static [IntrinsicId] = &[$(IntrinsicId::$variant,)*];

            pub const SLOTS: u32 = {
                let mut max = 0u32;
                $(if $id as u32 > max { max = $id as u32; })*
                max + 1
            };

            pub fn from_u16(v: u16) -> Option<Self> {
                Some(match v {
                    $($id => IntrinsicId::$variant,)*
                    _ => return None,
                })
            }

            pub fn name(self) -> &'static str {
                match self {
                    $(IntrinsicId::$variant => $name,)*
                }
            }
        }
    };
}

declare_intrinsics! {
    Logits = 0, LOGITS, "logits";
    MtpLogits = 1, MTP_LOGITS, "mtp_logits";
    Hidden = 2, HIDDEN, "hidden";
    Query = 3, QUERY, "query";
    ValueHead = 4, VALUE_HEAD, "value_head";
    Layer = 5, LAYER, "layer";
    MtpDrafts = 6, MTP_DRAFTS, "mtp_drafts";
    AttnScore = 7, ATTN_SCORE, "attn_score";
    Velocity = 8, VELOCITY, "velocity";
    Pixels = 9, PIXELS, "pixels";
}

impl IntrinsicId {
    /// Whether the intrinsic is one lane's of its pass (`velocity`, `hidden`
    /// and `pixels` name the lane); the rest are the pass's.
    pub fn per_lane(self) -> bool {
        matches!(
            self,
            IntrinsicId::Velocity | IntrinsicId::Hidden | IntrinsicId::Pixels
        )
    }
}

/// The lanes of its pass an epilogue reads per-lane intrinsics of.
pub const INTRINSIC_LANES: u32 = 16;

/// The side-table slots one program instance binds intrinsics at.
pub const INTRINSIC_PITCH: u32 = IntrinsicId::SLOTS * INTRINSIC_LANES;

/// The side-table slot an intrinsic of a pass's lane `lane` is bound at:
/// [`IntrinsicId::SLOTS`] of them a lane, lane by lane.
pub const fn intrinsic_slot(intr: IntrinsicId, lane: u32) -> u32 {
    intr as u32 + IntrinsicId::SLOTS * lane
}

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Const(Literal),

    Exp(ValueId),
    Log(ValueId),
    Neg(ValueId),
    Recip(ValueId),
    Sin(ValueId),
    Cos(ValueId),
    Sqrt(ValueId),
    Rsqrt(ValueId),
    Abs(ValueId),
    Sign(ValueId),
    Cast {
        value: ValueId,
        dtype: Dtype,
    },

    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    MaxElem(ValueId, ValueId),
    MinElem(ValueId, ValueId),
    Rem(ValueId, ValueId),

    Gt(ValueId, ValueId),
    Ge(ValueId, ValueId),
    Eq(ValueId, ValueId),
    Ne(ValueId, ValueId),
    Lt(ValueId, ValueId),
    Le(ValueId, ValueId),
    And(ValueId, ValueId),
    Or(ValueId, ValueId),
    Not(ValueId),

    Select {
        cond: ValueId,
        a: ValueId,
        b: ValueId,
    },

    ReduceSum(ValueId),
    ReduceMax(ValueId),
    ReduceMin(ValueId),
    ReduceArgmax(ValueId),

    Broadcast {
        value: ValueId,
        shape: Shape,
    },
    Reshape {
        value: ValueId,
        shape: Shape,
    },
    Transpose(ValueId),

    CumSum(ValueId),
    CumProd(ValueId),

    SortDesc(ValueId),
    TopK {
        input: ValueId,
        k: u32,
    },
    PivotThreshold {
        input: ValueId,
        predicate: Predicate,
    },

    MatMul(ValueId, ValueId),

    Gather {
        src: ValueId,
        idx: ValueId,
    },
    GatherRow {
        src: ValueId,
        idx: ValueId,
    },
    ScatterAdd {
        base: ValueId,
        idx: ValueId,
        vals: ValueId,
    },
    ScatterSet {
        base: ValueId,
        idx: ValueId,
        vals: ValueId,
    },
    Iota {
        len: u32,
    },
    MaskApply {
        logits: ValueId,
        mask: ValueId,
    },
    CausalMask {
        positions: ValueId,
        len: u32,
    },
    SlidingWindowMask {
        positions: ValueId,
        len: u32,
        window: u32,
    },
    SinkWindowMask {
        positions: ValueId,
        len: u32,
        sink: u32,
        window: u32,
    },
    Rng {
        stream: u32,
        shape: Shape,
        kind: RngKind,
    },
    RngKeyed {
        state: ValueId,
        shape: Shape,
        kind: RngKind,
    },

    ChanTake(ChannelIndex),
    ChanRead(ChannelIndex),
    ChanPut {
        chan: ChannelIndex,
        value: ValueId,
    },

    /// `lane` is the pass's lane a per-lane intrinsic reads
    /// ([`IntrinsicId::per_lane`]), 0 for the others.
    IntrinsicVal {
        intr: IntrinsicId,
        lane: u32,
        shape: Shape,
        dtype: Dtype,
    },
    KernelCall {
        name: NameIndex,
        args: Vec<ValueId>,
        shape: Shape,
        dtype: Dtype,
    },
    SinkCall {
        name: NameIndex,
        args: Vec<ValueId>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChannelUse {
    Take,
    Read,
    Put,
}

impl Op {
    pub fn channel_use(&self) -> Option<(ChannelUse, ChannelIndex)> {
        match self {
            Op::ChanTake(chan) => Some((ChannelUse::Take, *chan)),
            Op::ChanRead(chan) => Some((ChannelUse::Read, *chan)),
            Op::ChanPut { chan, .. } => Some((ChannelUse::Put, *chan)),
            _ => None,
        }
    }

    pub fn channel_mut(&mut self) -> Option<&mut ChannelIndex> {
        match self {
            Op::ChanTake(chan) | Op::ChanRead(chan) | Op::ChanPut { chan, .. } => Some(chan),
            _ => None,
        }
    }

    pub fn name_index_mut(&mut self) -> Option<&mut NameIndex> {
        match self {
            Op::KernelCall { name, .. } | Op::SinkCall { name, .. } => Some(name),
            _ => None,
        }
    }

    pub fn result_count(&self) -> u32 {
        match spec(self.tag()) {
            Some(row) => u32::from(row.results),
            None => unreachable!("op tag has no OP_TABLE row"),
        }
    }

    pub fn family(&self) -> Family {
        match family_of(self.tag()) {
            Some(family) => family,
            None => unreachable!("op tag has no OP_TABLE row"),
        }
    }

    pub fn is_effectful(&self) -> bool {
        match self {
            Op::ChanTake(..)
            | Op::ChanRead(..)
            | Op::ChanPut { .. }
            | Op::KernelCall { .. }
            | Op::SinkCall { .. } => true,

            Op::Const(..)
            | Op::Exp(..)
            | Op::Log(..)
            | Op::Neg(..)
            | Op::Recip(..)
            | Op::Sin(..)
            | Op::Cos(..)
            | Op::Sqrt(..)
            | Op::Rsqrt(..)
            | Op::Abs(..)
            | Op::Sign(..)
            | Op::Cast { .. }
            | Op::Add(..)
            | Op::Sub(..)
            | Op::Mul(..)
            | Op::Div(..)
            | Op::MaxElem(..)
            | Op::MinElem(..)
            | Op::Rem(..)
            | Op::Gt(..)
            | Op::Ge(..)
            | Op::Eq(..)
            | Op::Ne(..)
            | Op::Lt(..)
            | Op::Le(..)
            | Op::And(..)
            | Op::Or(..)
            | Op::Not(..)
            | Op::Select { .. }
            | Op::ReduceSum(..)
            | Op::ReduceMax(..)
            | Op::ReduceMin(..)
            | Op::ReduceArgmax(..)
            | Op::Broadcast { .. }
            | Op::Reshape { .. }
            | Op::Transpose(..)
            | Op::CumSum(..)
            | Op::CumProd(..)
            | Op::SortDesc(..)
            | Op::TopK { .. }
            | Op::PivotThreshold { .. }
            | Op::MatMul(..)
            | Op::Gather { .. }
            | Op::GatherRow { .. }
            | Op::ScatterAdd { .. }
            | Op::ScatterSet { .. }
            | Op::Iota { .. }
            | Op::MaskApply { .. }
            | Op::CausalMask { .. }
            | Op::SlidingWindowMask { .. }
            | Op::SinkWindowMask { .. }
            | Op::Rng { .. }
            | Op::RngKeyed { .. }
            | Op::IntrinsicVal { .. } => false,
        }
    }

    pub fn value_source(&self) -> ValueSource {
        match self {
            Op::KernelCall { .. }
            | Op::IntrinsicVal { .. }
            | Op::SinkCall { .. }
            | Op::Rng { .. } => ValueSource::Device,

            Op::ChanTake(..) | Op::ChanRead(..) | Op::ChanPut { .. } => ValueSource::Channel,

            Op::Const(..)
            | Op::Exp(..)
            | Op::Log(..)
            | Op::Neg(..)
            | Op::Recip(..)
            | Op::Sin(..)
            | Op::Cos(..)
            | Op::Sqrt(..)
            | Op::Rsqrt(..)
            | Op::Abs(..)
            | Op::Sign(..)
            | Op::Cast { .. }
            | Op::Add(..)
            | Op::Sub(..)
            | Op::Mul(..)
            | Op::Div(..)
            | Op::MaxElem(..)
            | Op::MinElem(..)
            | Op::Rem(..)
            | Op::Gt(..)
            | Op::Ge(..)
            | Op::Eq(..)
            | Op::Ne(..)
            | Op::Lt(..)
            | Op::Le(..)
            | Op::And(..)
            | Op::Or(..)
            | Op::Not(..)
            | Op::Select { .. }
            | Op::ReduceSum(..)
            | Op::ReduceMax(..)
            | Op::ReduceMin(..)
            | Op::ReduceArgmax(..)
            | Op::Broadcast { .. }
            | Op::Reshape { .. }
            | Op::Transpose(..)
            | Op::CumSum(..)
            | Op::CumProd(..)
            | Op::SortDesc(..)
            | Op::TopK { .. }
            | Op::PivotThreshold { .. }
            | Op::MatMul(..)
            | Op::Gather { .. }
            | Op::GatherRow { .. }
            | Op::ScatterAdd { .. }
            | Op::ScatterSet { .. }
            | Op::Iota { .. }
            | Op::MaskApply { .. }
            | Op::CausalMask { .. }
            | Op::SlidingWindowMask { .. }
            | Op::SinkWindowMask { .. }
            | Op::RngKeyed { .. } => ValueSource::Operands,
        }
    }
}

macro_rules! declare_operands {
    (
        fixed { $( $pat:pat => [$($slot:ident),*], )* }
        predicate { $ppat:pat => ($pinput:ident, $ppred:ident) }
        variadic { $( $vpat:pat => $vargs:ident, )* }
    ) => {
        impl Op {
            pub fn operands(&self) -> Vec<ValueId> {
                match self {
                    $( $pat => vec![$(*$slot),*], )*
                    $ppat => vec![*$pinput, $ppred.value()],
                    $( $vpat => $vargs.clone(), )*
                }
            }

            pub fn map_operands(&mut self, mut f: impl FnMut(ValueId) -> ValueId) {
                match self {
                    $( $pat => { $( *$slot = f(*$slot); )* } )*
                    $ppat => {
                        *$pinput = f(*$pinput);
                        let threshold = $ppred.value_slot();
                        *threshold = f(*threshold);
                    }
                    $( $vpat => {
                        for arg in $vargs.iter_mut() {
                            *arg = f(*arg);
                        }
                    } )*
                }
            }
        }
    };
}

declare_operands! {
    fixed {
        Op::Const(_)
        | Op::Iota { .. }
        | Op::Rng { .. }
        | Op::ChanTake(_)
        | Op::ChanRead(_)
        | Op::IntrinsicVal { .. } => [],

        Op::Exp(a)
        | Op::Log(a)
        | Op::Neg(a)
        | Op::Recip(a)
        | Op::Sin(a)
        | Op::Cos(a)
        | Op::Sqrt(a)
        | Op::Rsqrt(a)
        | Op::Abs(a)
        | Op::Sign(a)
        | Op::Cast { value: a, .. }
        | Op::Not(a)
        | Op::ReduceSum(a)
        | Op::ReduceMax(a)
        | Op::ReduceMin(a)
        | Op::ReduceArgmax(a)
        | Op::Broadcast { value: a, .. }
        | Op::Reshape { value: a, .. }
        | Op::Transpose(a)
        | Op::CumSum(a)
        | Op::CumProd(a)
        | Op::SortDesc(a)
        | Op::TopK { input: a, .. }
        | Op::CausalMask { positions: a, .. }
        | Op::SlidingWindowMask { positions: a, .. }
        | Op::SinkWindowMask { positions: a, .. }
        | Op::RngKeyed { state: a, .. }
        | Op::ChanPut { value: a, .. } => [a],

        Op::Add(a, b)
        | Op::Sub(a, b)
        | Op::Mul(a, b)
        | Op::Div(a, b)
        | Op::MaxElem(a, b)
        | Op::MinElem(a, b)
        | Op::Rem(a, b)
        | Op::Gt(a, b)
        | Op::Ge(a, b)
        | Op::Eq(a, b)
        | Op::Ne(a, b)
        | Op::Lt(a, b)
        | Op::Le(a, b)
        | Op::And(a, b)
        | Op::Or(a, b)
        | Op::MatMul(a, b)
        | Op::Gather { src: a, idx: b }
        | Op::GatherRow { src: a, idx: b }
        | Op::MaskApply { logits: a, mask: b } => [a, b],

        Op::Select { cond: a, a: b, b: c } => [a, b, c],

        Op::ScatterAdd { base: a, idx: b, vals: c }
        | Op::ScatterSet { base: a, idx: b, vals: c } => [a, b, c],
    }
    predicate {
        Op::PivotThreshold { input, predicate } => (input, predicate)
    }
    variadic {
        Op::KernelCall { args, .. } => args,
        Op::SinkCall { args, .. } => args,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueSource {
    Device,
    Channel,
    Operands,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Leaf,
    Map,
    CompareLogic,
    Choice,
    Shape,
    Index,
    ReduceScan,
    Order,
    Linear,
    Sampling,
    Channel,
    Intrinsic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireField {
    Value,
    Chan,
    Imm,
    Dtype,
    Shape,
    RngKind,
    Predicate,
    Literal,
    Name,
    Intrinsic,
    Args,
}

#[derive(Clone, Copy, Debug)]
pub struct OpSpec {
    pub tag: u8,
    pub name: &'static str,
    pub family: Family,
    pub val_operands: u8,
    pub results: u8,
    pub wire: &'static [WireField],
}

pub const VARIADIC: u8 = 0xFF;

macro_rules! declare_ops {
    ($($konst:ident = $tag:literal, $name:literal, $family:ident, $operands:expr, $results:expr,
       $rep:expr, $pat:pat, [$($wire:ident),*];)*) => {
        pub mod tags {
            $(
                #[doc = concat!("Wire tag of the `", $name, "` op.")]
                pub const $konst: u8 = $tag;
            )*
        }

        pub const OP_TABLE: &[OpSpec] = &[
            $(OpSpec {
                tag: tags::$konst,
                name: $name,
                family: Family::$family,
                val_operands: $operands,
                results: $results,
                wire: &[$(WireField::$wire,)*],
            },)*
        ];

        pub fn representatives() -> alloc::vec::Vec<Op> {
            alloc::vec![$($rep,)*]
        }

        impl Op {
            pub fn tag(&self) -> u8 {
                match self {
                    $($pat => tags::$konst,)*
                }
            }
        }
    };
}

declare_ops! {
    EXP = 0x01, "exp", Map, 1, 1, Op::Exp(0), Op::Exp(_), [Value];
    LOG = 0x02, "log", Map, 1, 1, Op::Log(0), Op::Log(_), [Value];
    NEG = 0x03, "neg", Map, 1, 1, Op::Neg(0), Op::Neg(_), [Value];
    RECIP = 0x04, "recip", Map, 1, 1, Op::Recip(0), Op::Recip(_), [Value];
    ABS = 0x05, "abs", Map, 1, 1, Op::Abs(0), Op::Abs(_), [Value];
    SIGN = 0x06, "sign", Map, 1, 1, Op::Sign(0), Op::Sign(_), [Value];
    CAST = 0x07, "cast", Map, 1, 1,
        Op::Cast { value: 0, dtype: Dtype::I32 }, Op::Cast { .. }, [Value, Dtype];
    SIN = 0x08, "sin", Map, 1, 1, Op::Sin(0), Op::Sin(_), [Value];
    COS = 0x09, "cos", Map, 1, 1, Op::Cos(0), Op::Cos(_), [Value];
    SQRT = 0x0A, "sqrt", Map, 1, 1, Op::Sqrt(0), Op::Sqrt(_), [Value];
    RSQRT = 0x0B, "rsqrt", Map, 1, 1, Op::Rsqrt(0), Op::Rsqrt(_), [Value];
    ADD = 0x10, "add", Map, 2, 1, Op::Add(0, 1), Op::Add(..), [Value, Value];
    SUB = 0x11, "sub", Map, 2, 1, Op::Sub(0, 1), Op::Sub(..), [Value, Value];
    MUL = 0x12, "mul", Map, 2, 1, Op::Mul(0, 1), Op::Mul(..), [Value, Value];
    DIV = 0x13, "div", Map, 2, 1, Op::Div(0, 1), Op::Div(..), [Value, Value];
    MAX_ELEM = 0x14, "max_elem", Map, 2, 1, Op::MaxElem(0, 1), Op::MaxElem(..), [Value, Value];
    MIN_ELEM = 0x15, "min_elem", Map, 2, 1, Op::MinElem(0, 1), Op::MinElem(..), [Value, Value];
    GT = 0x16, "gt", CompareLogic, 2, 1, Op::Gt(0, 1), Op::Gt(..), [Value, Value];
    GE = 0x17, "ge", CompareLogic, 2, 1, Op::Ge(0, 1), Op::Ge(..), [Value, Value];
    EQ = 0x18, "eq", CompareLogic, 2, 1, Op::Eq(0, 1), Op::Eq(..), [Value, Value];
    NE = 0x19, "ne", CompareLogic, 2, 1, Op::Ne(0, 1), Op::Ne(..), [Value, Value];
    LT = 0x1A, "lt", CompareLogic, 2, 1, Op::Lt(0, 1), Op::Lt(..), [Value, Value];
    LE = 0x1B, "le", CompareLogic, 2, 1, Op::Le(0, 1), Op::Le(..), [Value, Value];
    AND = 0x1C, "and", CompareLogic, 2, 1, Op::And(0, 1), Op::And(..), [Value, Value];
    OR = 0x1D, "or", CompareLogic, 2, 1, Op::Or(0, 1), Op::Or(..), [Value, Value];
    NOT = 0x1E, "not", CompareLogic, 1, 1, Op::Not(0), Op::Not(_), [Value];
    REM = 0x1F, "rem", Map, 2, 1, Op::Rem(0, 1), Op::Rem(..), [Value, Value];
    SELECT = 0x20, "select", Choice, 3, 1,
        Op::Select { cond: 0, a: 1, b: 2 }, Op::Select { .. }, [Value, Value, Value];
    REDUCE_SUM = 0x30, "reduce_sum", ReduceScan, 1, 1,
        Op::ReduceSum(0), Op::ReduceSum(_), [Value];
    REDUCE_MAX = 0x31, "reduce_max", ReduceScan, 1, 1,
        Op::ReduceMax(0), Op::ReduceMax(_), [Value];
    REDUCE_MIN = 0x32, "reduce_min", ReduceScan, 1, 1,
        Op::ReduceMin(0), Op::ReduceMin(_), [Value];
    REDUCE_ARGMAX = 0x33, "reduce_argmax", ReduceScan, 1, 1,
        Op::ReduceArgmax(0), Op::ReduceArgmax(_), [Value];
    BROADCAST = 0x38, "broadcast", Shape, 1, 1,
        Op::Broadcast { value: 0, shape: Shape::vector(4) }, Op::Broadcast { .. }, [Value, Shape];
    RESHAPE = 0x39, "reshape", Shape, 1, 1,
        Op::Reshape { value: 0, shape: Shape::vector(4) }, Op::Reshape { .. }, [Value, Shape];
    TRANSPOSE = 0x3A, "transpose", Shape, 1, 1, Op::Transpose(0), Op::Transpose(_), [Value];
    CUMSUM = 0x40, "cumsum", ReduceScan, 1, 1, Op::CumSum(0), Op::CumSum(_), [Value];
    CUMPROD = 0x41, "cumprod", ReduceScan, 1, 1, Op::CumProd(0), Op::CumProd(_), [Value];
    SORT_DESC = 0x50, "sort_desc", Order, 1, 2, Op::SortDesc(0), Op::SortDesc(_), [Value];
    TOP_K = 0x51, "top_k", Order, 1, 2,
        Op::TopK { input: 0, k: 4 }, Op::TopK { .. }, [Value, Imm];
    MATMUL = 0x55, "matmul", Linear, 2, 1, Op::MatMul(0, 1), Op::MatMul(..), [Value, Value];
    PIVOT_THRESHOLD = 0x58, "pivot_threshold", Order, 2, 1,
        Op::PivotThreshold { input: 0, predicate: Predicate::RankLe(1) },
        Op::PivotThreshold { .. }, [Value, Predicate];
    GATHER = 0x60, "gather", Index, 2, 1,
        Op::Gather { src: 0, idx: 1 }, Op::Gather { .. }, [Value, Value];
    GATHER_ROW = 0x61, "gather_row", Index, 2, 1,
        Op::GatherRow { src: 0, idx: 1 }, Op::GatherRow { .. }, [Value, Value];
    SCATTER_ADD = 0x62, "scatter_add", Index, 3, 1,
        Op::ScatterAdd { base: 0, idx: 1, vals: 2 }, Op::ScatterAdd { .. },
        [Value, Value, Value];
    SCATTER_SET = 0x63, "scatter_set", Index, 3, 1,
        Op::ScatterSet { base: 0, idx: 1, vals: 2 }, Op::ScatterSet { .. },
        [Value, Value, Value];
    IOTA = 0x64, "iota", Leaf, 0, 1, Op::Iota { len: 8 }, Op::Iota { .. }, [Imm];
    MASK_APPLY_PACKED = 0x65, "mask_apply_packed", Choice, 2, 1,
        Op::MaskApply { logits: 0, mask: 1 }, Op::MaskApply { .. }, [Value, Value];
    CAUSAL_MASK = 0x66, "causal_mask", Index, 1, 1,
        Op::CausalMask { positions: 0, len: 8 }, Op::CausalMask { .. }, [Value, Imm];
    SLIDING_WINDOW_MASK = 0x67, "sliding_window_mask", Index, 1, 1,
        Op::SlidingWindowMask { positions: 0, len: 8, window: 4 },
        Op::SlidingWindowMask { .. }, [Value, Imm, Imm];
    SINK_WINDOW_MASK = 0x68, "sink_window_mask", Index, 1, 1,
        Op::SinkWindowMask { positions: 0, len: 8, sink: 2, window: 4 },
        Op::SinkWindowMask { .. }, [Value, Imm, Imm, Imm];
    RNG = 0x70, "rng", Sampling, 0, 1,
        Op::Rng { stream: 0, shape: Shape::vector(4), kind: RngKind::Gumbel },
        Op::Rng { .. }, [Imm, Shape, RngKind];
    RNG_KEYED = 0x71, "rng_keyed", Sampling, 1, 1,
        Op::RngKeyed { state: 0, shape: Shape::vector(4), kind: RngKind::Uniform },
        Op::RngKeyed { .. }, [Value, Shape, RngKind];
    CONST = 0x81, "const", Leaf, 0, 1,
        Op::Const(Literal::F32(1.0)), Op::Const(_), [Literal];
    CHAN_TAKE = 0x90, "chan_take", Channel, 0, 1, Op::ChanTake(0), Op::ChanTake(_), [Chan];
    CHAN_READ = 0x91, "chan_read", Channel, 0, 1, Op::ChanRead(0), Op::ChanRead(_), [Chan];
    CHAN_PUT = 0x92, "chan_put", Channel, 1, 0,
        Op::ChanPut { chan: 0, value: 0 }, Op::ChanPut { .. }, [Chan, Value];
    INTRINSIC_VAL = 0xA0, "intrinsic_val", Intrinsic, 0, 1,
        Op::IntrinsicVal { intr: IntrinsicId::Logits, lane: 0, shape: Shape::matrix(1, 8), dtype: Dtype::F32 },
        Op::IntrinsicVal { .. }, [Intrinsic, Imm, Dtype, Shape];
    KERNEL_CALL = 0xA1, "kernel_call", Intrinsic, VARIADIC, 1,
        Op::KernelCall { name: 0, args: vec![0, 1], shape: Shape::vector(4), dtype: Dtype::F32 },
        Op::KernelCall { .. }, [Name, Dtype, Shape, Args];
    SINK_CALL = 0xA2, "sink_call", Intrinsic, VARIADIC, 0,
        Op::SinkCall { name: 0, args: vec![0] }, Op::SinkCall { .. }, [Name, Args];
}

pub fn spec(tag: u8) -> Option<&'static OpSpec> {
    let mut lo = 0usize;
    let mut hi = OP_TABLE.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let row = &OP_TABLE[mid];
        if row.tag == tag {
            return Some(row);
        } else if row.tag < tag {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    None
}

pub fn family_of(tag: u8) -> Option<Family> {
    spec(tag).map(|row| row.family)
}
