//! The PTIR op set — the closed first-party core (overview appendix) plus the
//! channel / intrinsic / kernel / sink carrier ops, with its **op table** (the
//! single source of truth for op ids, names, families, and arities; the C++
//! header `include/ptir_abi.h` is generated from [`OP_TABLE`]).
//!
//! ## Relation to PSIR v4
//!
//! Where an op coincides with a [`crate::types::Op`] variant, the **wire tag is
//! identical** (e.g. `Add` = 0x10, `Gather` = 0x60), so a driver-side decoder
//! extends its v4 table instead of forking. New tags occupy previously free
//! space; tag `0x80` (`Input`) is *reserved-unused* — PTIR stage bodies have no
//! input slots: values enter through channel ops ([`Op::ChanTake`] /
//! [`Op::ChanRead`]), intrinsics ([`Op::IntrinsicVal`]), and constants; effects
//! leave through [`Op::ChanPut`] and [`Op::SinkCall`]. A stage body is just
//! `Vec<Op>` — no separate inputs/outputs tables.
//!
//! ## Generalized index ops (superset semantics, same tags)
//!
//! `gather` / `scatter_set` / `scatter_add` operate along **axis 0**:
//! `gather(src[n, rest..], idx S) -> [S.., rest..]` (a rank-1 `src` with rank-1
//! `idx` is exactly the v4 element gather; a rank-2 `src` with rank-1 `idx` is
//! §6.2's row gather). `scatter_*(base[n, rest..], idx S, vals [S.., rest..])
//! -> base.shape`; duplicate indices resolve in index order, **last wins**
//! (`scatter_set`) / accumulate (`scatter_add`); an out-of-range index skips.
//! Valid v4 programs keep their exact meaning.
//!
//! ## SSA model
//!
//! One flat SSA space per stage body. Op at position `p` defines
//! `next_id .. next_id + result_count()`; `SortDesc`/`TopK` define 2 ids
//! (value-first), `ChanPut`/`SinkCall` define 0, everything else 1. Operands
//! reference earlier ids only.

use alloc::vec;
use alloc::vec::Vec;

use crate::types::{DType, Literal, Predicate, RngKind, Shape, ValueId};

/// Index of a channel in the container's channel-declaration table.
pub type ChannelIndex = u32;
/// Index into the container's name table (second-party kernel / sink names).
pub type NameIndex = u16;

/// First-party stage-scoped value intrinsics (overview §4, §5.3).
/// Wire tags are stable `u16` constants — see [`crate::registry`] for
/// scope + gating rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum IntrinsicId {
    /// `[n_out, vocab]` F32 — epilogue only.
    Logits = 0,
    /// `[K, vocab]` F32 — epilogue only; model-gated.
    MtpLogits = 1,
    /// `[n_out, d]` F32 — epilogue only.
    Hidden = 2,
    /// This layer's projected query — attn taps only.
    Query = 3,
    /// `[n_out]` F32 — epilogue only; model-gated.
    ValueHead = 4,
    /// Scalar U32 — the invocation's layer index; attn taps only. Replayable
    /// per-invocation value, not a register read (overview §5.3).
    Layer = 5,
    /// `[k]` I32 — epilogue only; model-gated. The MTP head's `k` draft token
    /// ids for the prior fire (device-resident spec-decode drafts channel).
    /// APPENDED (id 6) — existing ids 0..5 unchanged so every prior program's
    /// bytecode + identity hash stays byte-stable.
    MtpDrafts = 6,
}

impl IntrinsicId {
    pub fn from_u16(v: u16) -> Option<Self> {
        Some(match v {
            0 => IntrinsicId::Logits,
            1 => IntrinsicId::MtpLogits,
            2 => IntrinsicId::Hidden,
            3 => IntrinsicId::Query,
            4 => IntrinsicId::ValueHead,
            5 => IntrinsicId::Layer,
            6 => IntrinsicId::MtpDrafts,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            IntrinsicId::Logits => "logits",
            IntrinsicId::MtpLogits => "mtp_logits",
            IntrinsicId::Hidden => "hidden",
            IntrinsicId::Query => "query",
            IntrinsicId::ValueHead => "value_head",
            IntrinsicId::Layer => "layer",
            IntrinsicId::MtpDrafts => "mtp_drafts",
        }
    }
}

/// A PTIR stage-body op. See the module docs for the SSA model and the
/// PSIR-v4 tag-sharing rule.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// Trace-known constant scalar (`0x81`).
    Const(Literal),

    // ── map (unary) ──────────────────────────────────────────────────────
    Exp(ValueId),
    Log(ValueId),
    Neg(ValueId),
    Recip(ValueId),
    Abs(ValueId),
    Sign(ValueId),
    /// Element-wise dtype cast (`0x07`). numeric↔numeric; bool→numeric is
    /// `{0,1}`; numeric→bool is `x != 0`.
    Cast {
        value: ValueId,
        dtype: DType,
    },

    // ── map (binary; scalar operand broadcasts) ─────────────────────────
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    MaxElem(ValueId, ValueId),
    MinElem(ValueId, ValueId),
    /// Remainder (`0x1F`): integer `%` for I32/U32, `fmod` for F32.
    Rem(ValueId, ValueId),

    // ── compare / logic → Bool ───────────────────────────────────────────
    Gt(ValueId, ValueId),
    Ge(ValueId, ValueId),
    Eq(ValueId, ValueId),
    Ne(ValueId, ValueId),
    Lt(ValueId, ValueId),
    Le(ValueId, ValueId),
    And(ValueId, ValueId),
    Or(ValueId, ValueId),
    Not(ValueId),

    // ── choice ───────────────────────────────────────────────────────────
    Select {
        cond: ValueId,
        a: ValueId,
        b: ValueId,
    },

    // ── reduce (last axis; per-row for rank ≥ 2) ─────────────────────────
    ReduceSum(ValueId),
    ReduceMax(ValueId),
    ReduceMin(ValueId),
    ReduceArgmax(ValueId),

    // ── shape (metadata only) ────────────────────────────────────────────
    Broadcast {
        value: ValueId,
        shape: Shape,
    },
    /// Same numel, new dims (`0x39`). Dtype preserved.
    Reshape {
        value: ValueId,
        shape: Shape,
    },
    /// Rank-2 transpose `[m, n] → [n, m]` (`0x3A`).
    Transpose(ValueId),

    // ── scan (last axis; per-row for rank ≥ 2) ───────────────────────────
    CumSum(ValueId),
    CumProd(ValueId),

    // ── order ────────────────────────────────────────────────────────────
    /// Descending sort over `[n]` F32 → 2 results value-first (`0x50`).
    SortDesc(ValueId),
    /// Top-k over the last axis (`0x51`): `k` is a trace-known immediate
    /// (result shapes are trace-known, §5.1). 2 results value-first:
    /// values F32 `[.., k]`, indices U32 `[.., k]`. Ties → lower index.
    TopK {
        input: ValueId,
        k: u32,
    },
    /// Sort-free top-k/top-p/min-p mask (`0x58`), per-row for rank 2.
    PivotThreshold {
        input: ValueId,
        predicate: Predicate,
    },

    // ── linear ───────────────────────────────────────────────────────────
    /// `[m, k] × [k, n] → [m, n]`, F32 (`0x55`). A library kernel (T9).
    MatMul(ValueId, ValueId),

    // ── index (axis-0 generalized; see module docs) ──────────────────────
    Gather {
        src: ValueId,
        idx: ValueId,
    },
    /// Per-row column pick `out[i] = src[i, idx[i]]` (`0x61`, v4-exact).
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
    /// `iota(len)` → U32 `[len]` = `0..len` (`0x64`).
    Iota {
        len: u32,
    },
    /// Packed-bitmask apply (`0x65`, v4-exact): `out[j] = bit_j(mask) ?
    /// logits[j] : -inf`; `mask` `[ceil(n/32)]` U32. (The PTIR-level
    /// `mask_apply(logits, bool-mask)` composed op expands to `Select`;
    /// this packed form is the wire-efficient special case, kept core.)
    MaskApply {
        logits: ValueId,
        mask: ValueId,
    },

    // ── sampling ─────────────────────────────────────────────────────────
    /// Ambient-seed noise (`0x70`, v4-exact; per-fire seed folded by the
    /// runtime). Kept for epilogue-parity with shipped samplers.
    Rng {
        stream: u32,
        shape: Shape,
        kind: RngKind,
    },
    /// State-keyed noise (`0x71`): noise is a **pure function of the `[2]`
    /// U32 `state = [key, ctr]` tensor and the element index** — the §3 `rng`
    /// channel discipline; replay-deterministic (T8). Exact function pinned
    /// in PTIR-CONTAINER.md §5.
    RngKeyed {
        state: ValueId,
        shape: Shape,
        kind: RngKind,
    },

    // ── channels (the only effects) ──────────────────────────────────────
    /// Consume: full → value, set empty (`0x90`). In-pass register rule:
    /// a take after an in-pass put reads the pending value (§7.1).
    ChanTake(ChannelIndex),
    /// Peek: full → copy, stays full (`0x91`).
    ChanRead(ChannelIndex),
    /// Fill the pending cell (`0x92`); double-put = last wins (§7.1).
    /// Defines **0** result ids.
    ChanPut {
        chan: ChannelIndex,
        value: ValueId,
    },

    // ── intrinsics / second-party ────────────────────────────────────────
    /// Materialize a first-party stage-scoped value (`0xA0`). The shape and
    /// dtype are trace-known and declared inline; the validator cross-checks
    /// them against the registry rule and the stage scope.
    IntrinsicVal {
        intr: IntrinsicId,
        shape: Shape,
        dtype: DType,
    },
    /// Named second-party kernel call (`0xA1`): `intrinsics::kernel::*`.
    /// Name from the container's name table; availability + replayability
    /// (T10) checked at bind against the [`registry::ModelProfile`]. Declares
    /// its result type; no effects beyond it.
    KernelCall {
        name: NameIndex,
        args: Vec<ValueId>,
        shape: Shape,
        dtype: DType,
    },
    /// Named configuration sink (`0xA2`): takes tensors, returns nothing,
    /// configures THIS pass's forward (§4). Stage-precedence checked (T11).
    /// Defines **0** result ids.
    SinkCall {
        name: NameIndex,
        args: Vec<ValueId>,
    },
}

impl Op {
    /// Number of SSA ids this op defines.
    pub fn result_count(&self) -> u32 {
        match self {
            Op::SortDesc(_) | Op::TopK { .. } => 2,
            Op::ChanPut { .. } | Op::SinkCall { .. } => 0,
            _ => 1,
        }
    }

    /// The value ids this op reads, in a stable order (immediates excluded;
    /// the value-id predicate operands of `PivotThreshold` included).
    pub fn operands(&self) -> Vec<ValueId> {
        match *self {
            Op::Const(_)
            | Op::Iota { .. }
            | Op::Rng { .. }
            | Op::ChanTake(_)
            | Op::ChanRead(_)
            | Op::IntrinsicVal { .. } => Vec::new(),

            Op::Exp(a)
            | Op::Log(a)
            | Op::Neg(a)
            | Op::Recip(a)
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
            | Op::RngKeyed { state: a, .. }
            | Op::ChanPut { value: a, .. } => vec![a],

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
            | Op::MaskApply { logits: a, mask: b } => vec![a, b],

            Op::Select { cond, a, b } => vec![cond, a, b],
            Op::ScatterAdd { base, idx, vals } | Op::ScatterSet { base, idx, vals } => {
                vec![base, idx, vals]
            }

            Op::PivotThreshold { input, predicate } => match predicate {
                Predicate::RankLe(v) | Predicate::CummassLe(v) | Predicate::ProbGe(v) => {
                    vec![input, v]
                }
            },

            Op::KernelCall { ref args, .. } | Op::SinkCall { ref args, .. } => args.clone(),
        }
    }

    /// Rewrite this op's value-id operands in place — the mutable counterpart
    /// of [`Op::operands`], covering exactly the same ids (immediates
    /// untouched). For passes that renumber a stage's positional SSA space
    /// after inserting or removing ops.
    pub fn map_operands(&mut self, mut f: impl FnMut(ValueId) -> ValueId) {
        match self {
            Op::Const(_)
            | Op::Iota { .. }
            | Op::Rng { .. }
            | Op::ChanTake(_)
            | Op::ChanRead(_)
            | Op::IntrinsicVal { .. } => {}

            Op::Exp(a)
            | Op::Log(a)
            | Op::Neg(a)
            | Op::Recip(a)
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
            | Op::RngKeyed { state: a, .. }
            | Op::ChanPut { value: a, .. } => *a = f(*a),

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
            | Op::MaskApply { logits: a, mask: b } => {
                *a = f(*a);
                *b = f(*b);
            }

            Op::Select { cond, a, b } => {
                *cond = f(*cond);
                *a = f(*a);
                *b = f(*b);
            }
            Op::ScatterAdd { base, idx, vals } | Op::ScatterSet { base, idx, vals } => {
                *base = f(*base);
                *idx = f(*idx);
                *vals = f(*vals);
            }

            Op::PivotThreshold { input, predicate } => {
                *input = f(*input);
                match predicate {
                    Predicate::RankLe(v) | Predicate::CummassLe(v) | Predicate::ProbGe(v) => {
                        *v = f(*v)
                    }
                }
            }

            Op::KernelCall { args, .. } | Op::SinkCall { args, .. } => {
                for a in args.iter_mut() {
                    *a = f(*a);
                }
            }
        }
    }

    /// This op's wire tag (see [`OP_TABLE`]).
    pub fn tag(&self) -> u8 {
        match self {
            Op::Exp(_) => 0x01,
            Op::Log(_) => 0x02,
            Op::Neg(_) => 0x03,
            Op::Recip(_) => 0x04,
            Op::Abs(_) => 0x05,
            Op::Sign(_) => 0x06,
            Op::Cast { .. } => 0x07,
            Op::Add(..) => 0x10,
            Op::Sub(..) => 0x11,
            Op::Mul(..) => 0x12,
            Op::Div(..) => 0x13,
            Op::MaxElem(..) => 0x14,
            Op::MinElem(..) => 0x15,
            Op::Gt(..) => 0x16,
            Op::Ge(..) => 0x17,
            Op::Eq(..) => 0x18,
            Op::Ne(..) => 0x19,
            Op::Lt(..) => 0x1A,
            Op::Le(..) => 0x1B,
            Op::And(..) => 0x1C,
            Op::Or(..) => 0x1D,
            Op::Not(_) => 0x1E,
            Op::Rem(..) => 0x1F,
            Op::Select { .. } => 0x20,
            Op::ReduceSum(_) => 0x30,
            Op::ReduceMax(_) => 0x31,
            Op::ReduceMin(_) => 0x32,
            Op::ReduceArgmax(_) => 0x33,
            Op::Broadcast { .. } => 0x38,
            Op::Reshape { .. } => 0x39,
            Op::Transpose(_) => 0x3A,
            Op::CumSum(_) => 0x40,
            Op::CumProd(_) => 0x41,
            Op::SortDesc(_) => 0x50,
            Op::TopK { .. } => 0x51,
            Op::MatMul(..) => 0x55,
            Op::PivotThreshold { .. } => 0x58,
            Op::Gather { .. } => 0x60,
            Op::GatherRow { .. } => 0x61,
            Op::ScatterAdd { .. } => 0x62,
            Op::ScatterSet { .. } => 0x63,
            Op::Iota { .. } => 0x64,
            Op::MaskApply { .. } => 0x65,
            Op::Rng { .. } => 0x70,
            Op::RngKeyed { .. } => 0x71,
            Op::Const(_) => 0x81,
            Op::ChanTake(_) => 0x90,
            Op::ChanRead(_) => 0x91,
            Op::ChanPut { .. } => 0x92,
            Op::IntrinsicVal { .. } => 0xA0,
            Op::KernelCall { .. } => 0xA1,
            Op::SinkCall { .. } => 0xA2,
        }
    }
}

/// Op family (the overview appendix's row grouping).
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

/// One op-table row: the declarative identity charlie's C++ tables are
/// generated from. `operand layout` is documented per-op in
/// PTIR-CONTAINER.md §4; `val_operands` counts value-id operands
/// (`0xFF` = variadic, count byte on the wire).
#[derive(Clone, Copy, Debug)]
pub struct OpSpec {
    pub tag: u8,
    pub name: &'static str,
    pub family: Family,
    pub val_operands: u8,
    pub results: u8,
}

/// Variadic marker for [`OpSpec::val_operands`].
pub const VARIADIC: u8 = 0xFF;

/// The op table — one row per wire tag, sorted by tag. The generated C++
/// header and any driver-side dispatch table MUST be derived from this list.
pub const OP_TABLE: &[OpSpec] = &[
    OpSpec {
        tag: 0x01,
        name: "exp",
        family: Family::Map,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x02,
        name: "log",
        family: Family::Map,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x03,
        name: "neg",
        family: Family::Map,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x04,
        name: "recip",
        family: Family::Map,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x05,
        name: "abs",
        family: Family::Map,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x06,
        name: "sign",
        family: Family::Map,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x07,
        name: "cast",
        family: Family::Map,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x10,
        name: "add",
        family: Family::Map,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x11,
        name: "sub",
        family: Family::Map,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x12,
        name: "mul",
        family: Family::Map,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x13,
        name: "div",
        family: Family::Map,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x14,
        name: "max_elem",
        family: Family::Map,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x15,
        name: "min_elem",
        family: Family::Map,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x16,
        name: "gt",
        family: Family::CompareLogic,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x17,
        name: "ge",
        family: Family::CompareLogic,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x18,
        name: "eq",
        family: Family::CompareLogic,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x19,
        name: "ne",
        family: Family::CompareLogic,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x1A,
        name: "lt",
        family: Family::CompareLogic,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x1B,
        name: "le",
        family: Family::CompareLogic,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x1C,
        name: "and",
        family: Family::CompareLogic,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x1D,
        name: "or",
        family: Family::CompareLogic,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x1E,
        name: "not",
        family: Family::CompareLogic,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x1F,
        name: "rem",
        family: Family::Map,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x20,
        name: "select",
        family: Family::Choice,
        val_operands: 3,
        results: 1,
    },
    OpSpec {
        tag: 0x30,
        name: "reduce_sum",
        family: Family::ReduceScan,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x31,
        name: "reduce_max",
        family: Family::ReduceScan,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x32,
        name: "reduce_min",
        family: Family::ReduceScan,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x33,
        name: "reduce_argmax",
        family: Family::ReduceScan,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x38,
        name: "broadcast",
        family: Family::Shape,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x39,
        name: "reshape",
        family: Family::Shape,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x3A,
        name: "transpose",
        family: Family::Shape,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x40,
        name: "cumsum",
        family: Family::ReduceScan,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x41,
        name: "cumprod",
        family: Family::ReduceScan,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x50,
        name: "sort_desc",
        family: Family::Order,
        val_operands: 1,
        results: 2,
    },
    OpSpec {
        tag: 0x51,
        name: "top_k",
        family: Family::Order,
        val_operands: 1,
        results: 2,
    },
    OpSpec {
        tag: 0x55,
        name: "matmul",
        family: Family::Linear,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x58,
        name: "pivot_threshold",
        family: Family::Order,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x60,
        name: "gather",
        family: Family::Index,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x61,
        name: "gather_row",
        family: Family::Index,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x62,
        name: "scatter_add",
        family: Family::Index,
        val_operands: 3,
        results: 1,
    },
    OpSpec {
        tag: 0x63,
        name: "scatter_set",
        family: Family::Index,
        val_operands: 3,
        results: 1,
    },
    OpSpec {
        tag: 0x64,
        name: "iota",
        family: Family::Index,
        val_operands: 0,
        results: 1,
    },
    OpSpec {
        tag: 0x65,
        name: "mask_apply_packed",
        family: Family::Sampling,
        val_operands: 2,
        results: 1,
    },
    OpSpec {
        tag: 0x70,
        name: "rng",
        family: Family::Sampling,
        val_operands: 0,
        results: 1,
    },
    OpSpec {
        tag: 0x71,
        name: "rng_keyed",
        family: Family::Sampling,
        val_operands: 1,
        results: 1,
    },
    OpSpec {
        tag: 0x81,
        name: "const",
        family: Family::Leaf,
        val_operands: 0,
        results: 1,
    },
    OpSpec {
        tag: 0x90,
        name: "chan_take",
        family: Family::Channel,
        val_operands: 0,
        results: 1,
    },
    OpSpec {
        tag: 0x91,
        name: "chan_read",
        family: Family::Channel,
        val_operands: 0,
        results: 1,
    },
    OpSpec {
        tag: 0x92,
        name: "chan_put",
        family: Family::Channel,
        val_operands: 1,
        results: 0,
    },
    OpSpec {
        tag: 0xA0,
        name: "intrinsic_val",
        family: Family::Intrinsic,
        val_operands: 0,
        results: 1,
    },
    OpSpec {
        tag: 0xA1,
        name: "kernel_call",
        family: Family::Intrinsic,
        val_operands: VARIADIC,
        results: 1,
    },
    OpSpec {
        tag: 0xA2,
        name: "sink_call",
        family: Family::Intrinsic,
        val_operands: VARIADIC,
        results: 0,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_table_sorted_and_unique() {
        for w in OP_TABLE.windows(2) {
            assert!(w[0].tag < w[1].tag, "OP_TABLE must be sorted by tag");
        }
    }

    #[test]
    fn table_matches_op_metadata() {
        // One representative per variant; table row must agree with tag(),
        // result_count(), and operands().len().
        let reps: Vec<Op> = vec![
            Op::Exp(0),
            Op::Log(0),
            Op::Neg(0),
            Op::Recip(0),
            Op::Abs(0),
            Op::Sign(0),
            Op::Cast {
                value: 0,
                dtype: DType::I32,
            },
            Op::Add(0, 1),
            Op::Sub(0, 1),
            Op::Mul(0, 1),
            Op::Div(0, 1),
            Op::MaxElem(0, 1),
            Op::MinElem(0, 1),
            Op::Gt(0, 1),
            Op::Ge(0, 1),
            Op::Eq(0, 1),
            Op::Ne(0, 1),
            Op::Lt(0, 1),
            Op::Le(0, 1),
            Op::And(0, 1),
            Op::Or(0, 1),
            Op::Not(0),
            Op::Rem(0, 1),
            Op::Select {
                cond: 0,
                a: 1,
                b: 2,
            },
            Op::ReduceSum(0),
            Op::ReduceMax(0),
            Op::ReduceMin(0),
            Op::ReduceArgmax(0),
            Op::Broadcast {
                value: 0,
                shape: Shape::vector(4),
            },
            Op::Reshape {
                value: 0,
                shape: Shape::vector(4),
            },
            Op::Transpose(0),
            Op::CumSum(0),
            Op::CumProd(0),
            Op::SortDesc(0),
            Op::TopK { input: 0, k: 4 },
            Op::MatMul(0, 1),
            Op::PivotThreshold {
                input: 0,
                predicate: Predicate::RankLe(1),
            },
            Op::Gather { src: 0, idx: 1 },
            Op::GatherRow { src: 0, idx: 1 },
            Op::ScatterAdd {
                base: 0,
                idx: 1,
                vals: 2,
            },
            Op::ScatterSet {
                base: 0,
                idx: 1,
                vals: 2,
            },
            Op::Iota { len: 8 },
            Op::MaskApply { logits: 0, mask: 1 },
            Op::Rng {
                stream: 0,
                shape: Shape::vector(4),
                kind: RngKind::Gumbel,
            },
            Op::RngKeyed {
                state: 0,
                shape: Shape::vector(4),
                kind: RngKind::Uniform,
            },
            Op::Const(Literal::F32(1.0)),
            Op::ChanTake(0),
            Op::ChanRead(0),
            Op::ChanPut { chan: 0, value: 0 },
            Op::IntrinsicVal {
                intr: IntrinsicId::Logits,
                shape: Shape::matrix(1, 8),
                dtype: DType::F32,
            },
            Op::KernelCall {
                name: 0,
                args: vec![0, 1],
                shape: Shape::vector(4),
                dtype: DType::F32,
            },
            Op::SinkCall {
                name: 0,
                args: vec![0],
            },
        ];
        assert_eq!(
            reps.len(),
            OP_TABLE.len(),
            "one representative per table row"
        );
        for op in &reps {
            let spec = OP_TABLE
                .iter()
                .find(|s| s.tag == op.tag())
                .unwrap_or_else(|| panic!("no table row for {op:?}"));
            assert_eq!(
                spec.results as u32,
                op.result_count(),
                "results for {}",
                spec.name
            );
            if spec.val_operands != VARIADIC {
                assert_eq!(
                    spec.val_operands as usize,
                    op.operands().len(),
                    "arity for {}",
                    spec.name
                );
            }

            // map_operands must visit exactly operands(), in order, and a
            // rewrite must be readable back through operands().
            let mut visited: Vec<ValueId> = Vec::new();
            let mut rewritten = op.clone();
            rewritten.map_operands(|id| {
                visited.push(id);
                id + 100
            });
            assert_eq!(
                visited,
                op.operands(),
                "map_operands coverage for {}",
                spec.name
            );
            let shifted: Vec<ValueId> = op.operands().iter().map(|id| id + 100).collect();
            assert_eq!(
                rewritten.operands(),
                shifted,
                "map_operands rewrite for {}",
                spec.name
            );
        }
    }
}
