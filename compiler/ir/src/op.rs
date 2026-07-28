//! The PTIR op set — the closed first-party core (overview appendix) plus the
//! channel / intrinsic / kernel / sink carrier ops, with its **op table** (the
//! single source of truth for op ids, names, families, and arities; the C++
//! header `include/ptir_abi.h` is generated from [`OP_TABLE`]).
//!
//! ## Relation to PSIR v4
//!
//! Where an op coincides with a PSIR v4 op, the **wire tag is
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

/// Declares the first-party value intrinsics once and derives the enum, the
/// [`intrinsic_tags`] wire constants, [`IntrinsicId::ALL`], `from_u16` and
/// `name` from it.
///
/// Same rule as [`declare_ops`]: an intrinsic's id is spelled as a number on
/// exactly one line. `from_u16`, the name table and the generated C++ header's
/// `PtirIntrinsic` enum used to be three hand-kept copies of this list, and a
/// missed entry there is an intrinsic the driver never learns about.
macro_rules! declare_intrinsics {
    ($($(#[$doc:meta])* $variant:ident = $id:literal, $konst:ident, $name:literal;)*) => {
        /// First-party stage-scoped value intrinsics (overview §4, §5.3).
        /// Wire tags are stable `u16` constants — see [`crate::registry`] for
        /// scope + gating rules.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(u16)]
        pub enum IntrinsicId {
            $($(#[$doc])* $variant = $id,)*
        }

        /// The wire id of every intrinsic, by name — the `u16` counterpart of
        /// [`tags`], for downstream `match` arms that see a raw payload id.
        pub mod intrinsic_tags {
            $(pub const $konst: u16 = $id;)*
        }

        impl IntrinsicId {
            /// Every intrinsic, in wire-id order. Anything that must cover the
            /// whole set (the generated C++ header, a driver dispatch table)
            /// iterates this instead of repeating the list.
            pub const ALL: &'static [IntrinsicId] = &[$(IntrinsicId::$variant,)*];

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
    /// `[n_out, vocab]` F32 — epilogue only.
    Logits = 0, LOGITS, "logits";
    /// `[K, vocab]` F32 — epilogue only; model-gated.
    MtpLogits = 1, MTP_LOGITS, "mtp_logits";
    /// `[n_out, d]` F32 — epilogue only.
    Hidden = 2, HIDDEN, "hidden";
    /// This layer's projected query — attn taps only.
    Query = 3, QUERY, "query";
    /// `[n_out]` F32 — epilogue only; model-gated.
    ValueHead = 4, VALUE_HEAD, "value_head";
    /// Scalar U32 — the invocation's layer index; attn taps only. Replayable
    /// per-invocation value, not a register read (overview §5.3).
    Layer = 5, LAYER, "layer";
    /// `[k]` I32 — epilogue only; model-gated. The MTP head's `k` draft token
    /// ids for the prior fire (device-resident spec-decode drafts channel).
    /// APPENDED (id 6) — existing ids 0..5 unchanged so every prior program's
    /// bytecode + identity hash stays byte-stable.
    MtpDrafts = 6, MTP_DRAFTS, "mtp_drafts";
    /// `[num_heads, kv_len]` F32 — `OnAttn` only; model-gated. This layer's
    /// softmax attention weights over the request's live KV, the quantity
    /// H2O (arXiv:2306.14048) and TOVA (arXiv:2305.19370) evict on.
    /// Backend-shaped like `Query`, so the type rule stays loose.
    /// APPENDED (id 7) — ids 0..6 unchanged, same byte-stability contract.
    AttnScore = 7, ATTN_SCORE, "attn_score";
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
    /// Causal mask for query positions: output shape `positions.shape ++ [len]`.
    CausalMask {
        positions: ValueId,
        len: u32,
    },
    /// Causal sliding window: `key <= position && key + window > position`.
    SlidingWindowMask {
        positions: ValueId,
        len: u32,
        window: u32,
    },
    /// Causal sink + recent window mask.
    SinkWindowMask {
        positions: ValueId,
        len: u32,
        sink: u32,
        window: u32,
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

/// How an op touches a channel. See [`Op::channel_use`].
///
/// An enum rather than a bool because the three uses are not interchangeable:
/// readiness needs a full slot for `Take`/`Read` and an empty one for `Put`,
/// and SPSC endpoint counting treats `Take` and `Read` as consumers. Every
/// consumer matches this exhaustively, so a fourth channel op cannot be added
/// without each of them being asked what it means.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChannelUse {
    /// Consuming read (`chan_take`) — empties the slot.
    Take,
    /// Non-consuming read (`chan_read`) — requires a full slot, leaves it full.
    Read,
    /// Write (`chan_put`) — requires an empty slot.
    Put,
}

impl Op {
    /// How this op uses a channel, and which one, or `None` if it uses none.
    ///
    /// The single answer to "is this a channel op?". It replaced six separate
    /// `match op { Op::ChanTake(c) | Op::ChanRead(c) => .., Op::ChanPut { .. }
    /// => .., _ => {} }` scans across the planner, the validator and the DSL
    /// builder. Six copies is six chances to miss a new channel op, and the
    /// failure is not a crash: two of those scans rewrite channel ids, so a
    /// missed op keeps a stale id and reads the wrong channel.
    ///
    /// `table_matches_op_metadata` pins this to `Family::Channel`, so the op
    /// table and this accessor cannot disagree about what a channel op is.
    pub fn channel_use(&self) -> Option<(ChannelUse, ChannelIndex)> {
        match self {
            Op::ChanTake(chan) => Some((ChannelUse::Take, *chan)),
            Op::ChanRead(chan) => Some((ChannelUse::Read, *chan)),
            Op::ChanPut { chan, .. } => Some((ChannelUse::Put, *chan)),
            _ => None,
        }
    }

    /// The channel id this op names, for rewriting it. See [`Op::channel_use`].
    pub fn channel_mut(&mut self) -> Option<&mut ChannelIndex> {
        match self {
            Op::ChanTake(chan) | Op::ChanRead(chan) | Op::ChanPut { chan, .. } => Some(chan),
            _ => None,
        }
    }

    /// The name-table index this op names, for rewriting it when a stage's
    /// name table is localized. `intrinsic_val` shares `Family::Intrinsic`
    /// with these two but carries no name, so this set is pinned by tag rather
    /// than by family.
    pub fn name_index_mut(&mut self) -> Option<&mut NameIndex> {
        match self {
            Op::KernelCall { name, .. } | Op::SinkCall { name, .. } => Some(name),
            _ => None,
        }
    }

    /// Number of SSA ids this op defines.
    ///
    /// Read from [`OP_TABLE`] rather than re-derived here. The catch-all this
    /// replaces answered `1` for anything it did not recognise, so an op
    /// declared with two results but omitted from the match would define one
    /// id and **shift every later value id in the trace** — silently, because
    /// `Recorder::push` and the encoders all trust this number.
    pub fn result_count(&self) -> u32 {
        match spec(self.tag()) {
            Some(row) => u32::from(row.results),
            // `tag()` is exhaustive over `Op` and `declare_ops!` is the only
            // place a tag exists, so a variant whose tag has no row cannot be
            // built without also failing `table_matches_op_metadata`, which
            // pins one representative per row.
            None => unreachable!("op tag has no OP_TABLE row"),
        }
    }

    /// This op's [`Family`], read from [`OP_TABLE`].
    ///
    /// The family is a fact about the op's shape on the wire, so a pass that
    /// keys on "is this a channel op" should ask here rather than restate a
    /// variant list that can fall behind the table.
    pub fn family(&self) -> Family {
        match family_of(self.tag()) {
            Some(family) => family,
            // Same argument as `result_count`: `tag()` is exhaustive over
            // `Op`, and every tag has a row.
            None => unreachable!("op tag has no OP_TABLE row"),
        }
    }

    /// True when the op must survive dead-code elimination even if nothing
    /// reads its results, and must never be merged with an identical
    /// neighbour by CSE.
    ///
    /// Two things qualify: touching a channel (`take` pops; `read` and `put`
    /// order against its contents) and calling second-party code.
    ///
    /// `IntrinsicVal` is deliberately absent — it names a device-provided
    /// value such as the logits, so two of them *are* the same value and
    /// merging them is correct. So is `Rng`: within one fire it is a function
    /// of its stream and shape, which is why `pareval` can call it
    /// device-decided (the host does not know the ambient seed) while CSE
    /// still merges it. Taint and purity are different questions.
    ///
    /// Exhaustive on purpose. `fold::cse_candidate` and `normalize::live_ops`
    /// each carried this list; a new channel or call op missing from either
    /// would be quietly deleted by DCE or folded away by CSE.
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
            | Op::CausalMask { positions: a, .. }
            | Op::SlidingWindowMask { positions: a, .. }
            | Op::SinkWindowMask { positions: a, .. }
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
            | Op::CausalMask { positions: a, .. }
            | Op::SlidingWindowMask { positions: a, .. }
            | Op::SinkWindowMask { positions: a, .. }
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
            Op::Exp(_) => tags::EXP,
            Op::Log(_) => tags::LOG,
            Op::Neg(_) => tags::NEG,
            Op::Recip(_) => tags::RECIP,
            Op::Abs(_) => tags::ABS,
            Op::Sign(_) => tags::SIGN,
            Op::Cast { .. } => tags::CAST,
            Op::Add(..) => tags::ADD,
            Op::Sub(..) => tags::SUB,
            Op::Mul(..) => tags::MUL,
            Op::Div(..) => tags::DIV,
            Op::MaxElem(..) => tags::MAX_ELEM,
            Op::MinElem(..) => tags::MIN_ELEM,
            Op::Gt(..) => tags::GT,
            Op::Ge(..) => tags::GE,
            Op::Eq(..) => tags::EQ,
            Op::Ne(..) => tags::NE,
            Op::Lt(..) => tags::LT,
            Op::Le(..) => tags::LE,
            Op::And(..) => tags::AND,
            Op::Or(..) => tags::OR,
            Op::Not(_) => tags::NOT,
            Op::Rem(..) => tags::REM,
            Op::Select { .. } => tags::SELECT,
            Op::ReduceSum(_) => tags::REDUCE_SUM,
            Op::ReduceMax(_) => tags::REDUCE_MAX,
            Op::ReduceMin(_) => tags::REDUCE_MIN,
            Op::ReduceArgmax(_) => tags::REDUCE_ARGMAX,
            Op::Broadcast { .. } => tags::BROADCAST,
            Op::Reshape { .. } => tags::RESHAPE,
            Op::Transpose(_) => tags::TRANSPOSE,
            Op::CumSum(_) => tags::CUMSUM,
            Op::CumProd(_) => tags::CUMPROD,
            Op::SortDesc(_) => tags::SORT_DESC,
            Op::TopK { .. } => tags::TOP_K,
            Op::MatMul(..) => tags::MATMUL,
            Op::PivotThreshold { .. } => tags::PIVOT_THRESHOLD,
            Op::Gather { .. } => tags::GATHER,
            Op::GatherRow { .. } => tags::GATHER_ROW,
            Op::ScatterAdd { .. } => tags::SCATTER_ADD,
            Op::ScatterSet { .. } => tags::SCATTER_SET,
            Op::Iota { .. } => tags::IOTA,
            Op::MaskApply { .. } => tags::MASK_APPLY_PACKED,
            Op::CausalMask { .. } => tags::CAUSAL_MASK,
            Op::SlidingWindowMask { .. } => tags::SLIDING_WINDOW_MASK,
            Op::SinkWindowMask { .. } => tags::SINK_WINDOW_MASK,
            Op::Rng { .. } => tags::RNG,
            Op::RngKeyed { .. } => tags::RNG_KEYED,
            Op::Const(_) => tags::CONST,
            Op::ChanTake(_) => tags::CHAN_TAKE,
            Op::ChanRead(_) => tags::CHAN_READ,
            Op::ChanPut { .. } => tags::CHAN_PUT,
            Op::IntrinsicVal { .. } => tags::INTRINSIC_VAL,
            Op::KernelCall { .. } => tags::KERNEL_CALL,
            Op::SinkCall { .. } => tags::SINK_CALL,
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

/// One op-table row: the declarative identity the generated C++ tables are
/// built from. `operand layout` is documented per-op in PTIR-CONTAINER.md §4;
/// `val_operands` counts value-id operands (`0xFF` = variadic, count byte on
/// the wire).
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

/// Declares the op set once and derives [`tags`] and [`OP_TABLE`] from it.
///
/// A wire tag, its name, family, value-operand count and result count appear
/// on exactly one line each. Nothing downstream may re-spell a tag literal —
/// it imports [`tags`] instead.
macro_rules! declare_ops {
    ($($konst:ident = $tag:literal, $name:literal, $family:ident, $operands:expr, $results:expr;)*) => {
        /// The wire tag of every op, by name. **The only place a PTIR op tag
        /// is spelled as a number.** Downstream crates (`pie-plan`,
        /// `pie-codegen`, and drivers via the generated header) import these
        /// instead of keeping their own copies: a hand-copied tag that drifts
        /// by one hex digit is a silently wrong kernel, and nothing but a
        /// single definition can rule that out.
        pub mod tags {
            $(pub const $konst: u8 = $tag;)*
        }

        /// The op table — one row per wire tag, sorted by tag. The generated
        /// C++ header and any driver-side dispatch table MUST be derived from
        /// this list.
        pub const OP_TABLE: &[OpSpec] = &[
            $(OpSpec {
                tag: tags::$konst,
                name: $name,
                family: Family::$family,
                val_operands: $operands,
                results: $results,
            },)*
        ];
    };
}

declare_ops! {
    EXP = 0x01, "exp", Map, 1, 1;
    LOG = 0x02, "log", Map, 1, 1;
    NEG = 0x03, "neg", Map, 1, 1;
    RECIP = 0x04, "recip", Map, 1, 1;
    ABS = 0x05, "abs", Map, 1, 1;
    SIGN = 0x06, "sign", Map, 1, 1;
    CAST = 0x07, "cast", Map, 1, 1;
    ADD = 0x10, "add", Map, 2, 1;
    SUB = 0x11, "sub", Map, 2, 1;
    MUL = 0x12, "mul", Map, 2, 1;
    DIV = 0x13, "div", Map, 2, 1;
    MAX_ELEM = 0x14, "max_elem", Map, 2, 1;
    MIN_ELEM = 0x15, "min_elem", Map, 2, 1;
    GT = 0x16, "gt", CompareLogic, 2, 1;
    GE = 0x17, "ge", CompareLogic, 2, 1;
    EQ = 0x18, "eq", CompareLogic, 2, 1;
    NE = 0x19, "ne", CompareLogic, 2, 1;
    LT = 0x1A, "lt", CompareLogic, 2, 1;
    LE = 0x1B, "le", CompareLogic, 2, 1;
    AND = 0x1C, "and", CompareLogic, 2, 1;
    OR = 0x1D, "or", CompareLogic, 2, 1;
    NOT = 0x1E, "not", CompareLogic, 1, 1;
    REM = 0x1F, "rem", Map, 2, 1;
    SELECT = 0x20, "select", Choice, 3, 1;
    REDUCE_SUM = 0x30, "reduce_sum", ReduceScan, 1, 1;
    REDUCE_MAX = 0x31, "reduce_max", ReduceScan, 1, 1;
    REDUCE_MIN = 0x32, "reduce_min", ReduceScan, 1, 1;
    REDUCE_ARGMAX = 0x33, "reduce_argmax", ReduceScan, 1, 1;
    BROADCAST = 0x38, "broadcast", Shape, 1, 1;
    RESHAPE = 0x39, "reshape", Shape, 1, 1;
    TRANSPOSE = 0x3A, "transpose", Shape, 1, 1;
    CUMSUM = 0x40, "cumsum", ReduceScan, 1, 1;
    CUMPROD = 0x41, "cumprod", ReduceScan, 1, 1;
    SORT_DESC = 0x50, "sort_desc", Order, 1, 2;
    TOP_K = 0x51, "top_k", Order, 1, 2;
    MATMUL = 0x55, "matmul", Linear, 2, 1;
    PIVOT_THRESHOLD = 0x58, "pivot_threshold", Order, 2, 1;
    GATHER = 0x60, "gather", Index, 2, 1;
    GATHER_ROW = 0x61, "gather_row", Index, 2, 1;
    SCATTER_ADD = 0x62, "scatter_add", Index, 3, 1;
    SCATTER_SET = 0x63, "scatter_set", Index, 3, 1;
    IOTA = 0x64, "iota", Index, 0, 1;
    MASK_APPLY_PACKED = 0x65, "mask_apply_packed", Sampling, 2, 1;
    CAUSAL_MASK = 0x66, "causal_mask", Index, 1, 1;
    SLIDING_WINDOW_MASK = 0x67, "sliding_window_mask", Index, 1, 1;
    SINK_WINDOW_MASK = 0x68, "sink_window_mask", Index, 1, 1;
    RNG = 0x70, "rng", Sampling, 0, 1;
    RNG_KEYED = 0x71, "rng_keyed", Sampling, 1, 1;
    CONST = 0x81, "const", Leaf, 0, 1;
    CHAN_TAKE = 0x90, "chan_take", Channel, 0, 1;
    CHAN_READ = 0x91, "chan_read", Channel, 0, 1;
    CHAN_PUT = 0x92, "chan_put", Channel, 1, 0;
    INTRINSIC_VAL = 0xA0, "intrinsic_val", Intrinsic, 0, 1;
    KERNEL_CALL = 0xA1, "kernel_call", Intrinsic, VARIADIC, 1;
    SINK_CALL = 0xA2, "sink_call", Intrinsic, VARIADIC, 0;
}

/// The table row for a wire tag, or `None` when the tag is not a PTIR op.
///
/// [`OP_TABLE`] is sorted by tag, so this is a binary search. Downstream
/// dispatch asks this instead of matching hex ranges: a range literal silently
/// swallows any tag landing in one of its gaps, and the gaps are where new ops
/// go.
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

/// The family of a wire tag, or `None` when the tag is not a PTIR op.
pub fn family_of(tag: u8) -> Option<Family> {
    spec(tag).map(|row| row.family)
}

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
            Op::CausalMask {
                positions: 0,
                len: 8,
            },
            Op::SlidingWindowMask {
                positions: 0,
                len: 8,
                window: 4,
            },
            Op::SinkWindowMask {
                positions: 0,
                len: 8,
                sink: 2,
                window: 4,
            },
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

            // The channel accessors and the table must agree on what a
            // channel op is; six scans across three crates depend on it.
            let is_channel = spec.family == Family::Channel;
            assert_eq!(
                op.channel_use().is_some(),
                is_channel,
                "channel_use for {}",
                spec.name
            );
            assert_eq!(
                op.clone().channel_mut().is_some(),
                is_channel,
                "channel_mut for {}",
                spec.name
            );
            assert_eq!(
                op.clone().name_index_mut().is_some(),
                spec.tag == tags::KERNEL_CALL || spec.tag == tags::SINK_CALL,
                "name_index_mut for {}",
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
