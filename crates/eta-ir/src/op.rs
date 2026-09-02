//! The ETA op set: op ids, names, families, and arities, sourced from
//! [`OP_TABLE`].

use alloc::vec;
use alloc::vec::Vec;

use crate::types::{Dtype, Literal, Predicate, RngKind, Shape, ValueId};

/// Index of a channel in the container's channel-declaration table.
pub type ChannelIndex = u32;
/// Index into the container's name table (second-party kernel / sink names).
pub type NameIndex = u16;

/// Declares the first-party value intrinsics once and derives the enum,
/// [`intrinsic_tags`], [`IntrinsicId::ALL`], `from_u16`, and `name` from it.
macro_rules! declare_intrinsics {
    ($($(#[$doc:meta])* $variant:ident = $id:literal, $konst:ident, $name:literal;)*) => {
        /// First-party stage-scoped value intrinsics. Wire tags are stable
        /// `u16` constants — see [`crate::registry`] for scope and gating
        /// rules.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[repr(u16)]
        pub enum IntrinsicId {
            $($(#[$doc])* $variant = $id,)*
        }

        /// The wire id of every intrinsic, by name — the `u16` counterpart of
        /// [`tags`], for downstream `match` arms that see a raw payload id.
        pub mod intrinsic_tags {
            $(
                #[doc = concat!("Wire id of the `", $name, "` intrinsic.")]
                pub const $konst: u16 = $id;
            )*
        }

        impl IntrinsicId {
            /// Every intrinsic, in wire-id order.
            pub const ALL: &'static [IntrinsicId] = &[$(IntrinsicId::$variant,)*];

            /// One past the largest wire id, not `ALL.len()` — the per-lane
            /// stride of a table indexed by [`IntrinsicId`].
            pub const SLOTS: u32 = {
                let mut max = 0u32;
                $(if $id as u32 > max { max = $id as u32; })*
                max + 1
            };

            /// The intrinsic with wire id `v`, or `None` if no intrinsic
            /// claims it.
            pub fn from_u16(v: u16) -> Option<Self> {
                Some(match v {
                    $($id => IntrinsicId::$variant,)*
                    _ => return None,
                })
            }

            /// The intrinsic's spelling in diagnostics and generated sources.
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
    /// per-invocation value, not a register read.
    Layer = 5, LAYER, "layer";
    /// `[k]` I32 — epilogue only; model-gated. The MTP head's `k` draft token
    /// ids for the prior fire (device-resident spec-decode drafts channel).
    MtpDrafts = 6, MTP_DRAFTS, "mtp_drafts";
    /// `[num_heads, kv_len]` F32 — `OnAttn` only; model-gated. Softmax
    /// attention weights over live KV, for eviction policies.
    AttnScore = 7, ATTN_SCORE, "attn_score";
}

/// An ETA stage-body op. Docs here give meaning only; wire tags live on the
/// [`declare_ops!`] row, read via [`tags`].
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// Trace-known constant scalar.
    Const(Literal),

    // ── map (unary) ──────────────────────────────────────────────────────
    /// Element-wise `e^x`, F32 only.
    Exp(ValueId),
    /// Element-wise natural log, F32 only.
    Log(ValueId),
    /// Element-wise negation.
    Neg(ValueId),
    /// Element-wise `1/x`, F32 only.
    Recip(ValueId),
    /// Element-wise absolute value.
    Abs(ValueId),
    /// Element-wise sign as `-1` / `0` / `+1` in the input dtype.
    Sign(ValueId),
    /// Element-wise dtype cast. numeric↔numeric; bool→numeric is
    /// `{0,1}`; numeric→bool is `x != 0`.
    Cast {
        /// The tensor to reinterpret.
        value: ValueId,
        /// The result element type; the shape is unchanged.
        dtype: Dtype,
    },

    // ── map (binary; scalar operand broadcasts) ─────────────────────────
    /// Element-wise sum.
    Add(ValueId, ValueId),
    /// Element-wise difference.
    Sub(ValueId, ValueId),
    /// Element-wise product.
    Mul(ValueId, ValueId),
    /// Element-wise quotient; integer division truncates.
    Div(ValueId, ValueId),
    /// Element-wise maximum.
    MaxElem(ValueId, ValueId),
    /// Element-wise minimum.
    MinElem(ValueId, ValueId),
    /// Remainder: integer `%` for I32/U32, `fmod` for F32.
    Rem(ValueId, ValueId),

    // ── compare / logic → Bool ───────────────────────────────────────────
    /// Element-wise `a > b`.
    Gt(ValueId, ValueId),
    /// Element-wise `a >= b`.
    Ge(ValueId, ValueId),
    /// Element-wise `a == b`.
    Eq(ValueId, ValueId),
    /// Element-wise `a != b`.
    Ne(ValueId, ValueId),
    /// Element-wise `a < b`.
    Lt(ValueId, ValueId),
    /// Element-wise `a <= b`.
    Le(ValueId, ValueId),
    /// Element-wise conjunction, Bool operands.
    And(ValueId, ValueId),
    /// Element-wise disjunction, Bool operands.
    Or(ValueId, ValueId),
    /// Element-wise negation, Bool operand.
    Not(ValueId),

    // ── choice ───────────────────────────────────────────────────────────
    /// Element-wise `cond ? a : b`.
    Select {
        /// Bool selector.
        cond: ValueId,
        /// Taken where `cond` is true.
        a: ValueId,
        /// Taken where `cond` is false; same dtype as `a`.
        b: ValueId,
    },

    // ── reduce (last axis; per-row for rank ≥ 2) ─────────────────────────
    /// Sum over the last axis, dropping it.
    ReduceSum(ValueId),
    /// Maximum over the last axis, dropping it.
    ReduceMax(ValueId),
    /// Minimum over the last axis, dropping it.
    ReduceMin(ValueId),
    /// Index of the maximum along the last axis as **I32** (token channels
    /// are I32, unlike [`Op::TopK`]/[`Op::SortDesc`]'s U32). Lower index wins ties.
    ReduceArgmax(ValueId),

    // ── shape (metadata only) ────────────────────────────────────────────
    /// Stretch to a larger shape without moving data. Dtype
    /// preserved.
    Broadcast {
        /// The tensor to stretch.
        value: ValueId,
        /// The result shape; each of `value`'s dims must be `1` or already
        /// equal to the corresponding result dim.
        shape: Shape,
    },
    /// Same numel, new dims. Dtype preserved.
    Reshape {
        /// The tensor to reinterpret.
        value: ValueId,
        /// The result shape; its `numel` must equal `value`'s.
        shape: Shape,
    },
    /// Rank-2 transpose `[m, n] → [n, m]`.
    Transpose(ValueId),

    // ── scan (last axis; per-row for rank ≥ 2) ───────────────────────────
    /// Inclusive prefix sum along the last axis; shape and dtype preserved.
    /// Numeric, not F32-only — used to build U32 ragged-tensor row offsets.
    CumSum(ValueId),
    /// Inclusive prefix product along the last axis; shape and dtype
    /// preserved. Numeric, like [`Op::CumSum`]; integer overflow wraps.
    CumProd(ValueId),

    // ── order ────────────────────────────────────────────────────────────
    /// Descending sort over `[n]` F32 → 2 results value-first.
    SortDesc(ValueId),
    /// Top-k over the last axis, `k` a trace-known immediate. 2 results
    /// value-first: values F32 `[.., k]`, indices U32 `[.., k]`.
    TopK {
        /// The tensor to rank along its last axis.
        input: ValueId,
        /// How many entries to keep; must not exceed the last axis.
        k: u32,
    },
    /// Sort-free top-k/top-p/min-p mask, per-row for rank 2.
    PivotThreshold {
        /// F32 scores to threshold, per row for rank 2.
        input: ValueId,
        /// Which entries survive; see [`Predicate`].
        predicate: Predicate,
    },

    // ── linear ───────────────────────────────────────────────────────────
    /// `[m, k] × [k, n] → [m, n]`, F32. A library kernel.
    MatMul(ValueId, ValueId),

    // ── index (axis-0 generalized) ───────────────────────────────────────
    /// Axis-0 gather `src[n, rest..]` by `idx S` → `[S.., rest..]`.
    /// An out-of-range index yields zero.
    Gather {
        /// The tensor to read rows from.
        src: ValueId,
        /// U32 axis-0 indices.
        idx: ValueId,
    },
    /// Per-row column pick `out[i] = src[i, idx[i]]` (v4-exact).
    GatherRow {
        /// Rank-2 source.
        src: ValueId,
        /// One U32 column index per row of `src`.
        idx: ValueId,
    },
    /// Axis-0 scatter that accumulates into `base`. Duplicate indices all
    /// contribute; an out-of-range index is skipped.
    ScatterAdd {
        /// The tensor updates are applied to; unwritten rows pass through.
        base: ValueId,
        /// U32 axis-0 indices.
        idx: ValueId,
        /// Values to accumulate, shaped `idx.shape ++ base.shape[1..]`.
        vals: ValueId,
    },
    /// Axis-0 scatter that overwrites into `base`. Duplicates resolve last
    /// wins, in index order; an out-of-range index is skipped.
    ScatterSet {
        /// The tensor updates are applied to; unwritten rows pass through.
        base: ValueId,
        /// U32 axis-0 indices.
        idx: ValueId,
        /// Values to write, shaped `idx.shape ++ base.shape[1..]`.
        vals: ValueId,
    },
    /// `iota(len)` → U32 `[len]` = `0..len`.
    Iota {
        /// Number of elements produced.
        len: u32,
    },
    /// Packed-bitmask apply (v4-exact): `out[j] = bit_j(mask) ?
    /// logits[j] : -inf`; `mask` is `[ceil(n/32)]` U32.
    MaskApply {
        /// F32 scores to mask.
        logits: ValueId,
        /// Packed bitmask, one bit per logit.
        mask: ValueId,
    },
    /// Causal mask for query positions: output shape `positions.shape ++ [len]`.
    CausalMask {
        /// U32 query positions.
        positions: ValueId,
        /// Key-axis extent; the trailing dim of the result.
        len: u32,
    },
    /// Causal sliding window: `key <= position && key + window > position`.
    SlidingWindowMask {
        /// U32 query positions.
        positions: ValueId,
        /// Key-axis extent; the trailing dim of the result.
        len: u32,
        /// How many keys back from each position stay visible.
        window: u32,
    },
    /// Causal sink + recent window mask.
    SinkWindowMask {
        /// U32 query positions.
        positions: ValueId,
        /// Key-axis extent; the trailing dim of the result.
        len: u32,
        /// How many leading keys stay visible regardless of position.
        sink: u32,
        /// How many keys back from each position stay visible.
        window: u32,
    },
    // ── sampling ─────────────────────────────────────────────────────────
    /// Ambient-seed noise (v4-exact; per-fire seed folded by the
    /// runtime). Kept for epilogue-parity with shipped samplers.
    Rng {
        /// Decorrelates draws that share a fire; two ops with the same
        /// stream draw the same numbers.
        stream: u32,
        /// The result shape, trace-known.
        shape: Shape,
        /// Which distribution to draw from.
        kind: RngKind,
    },
    /// State-keyed noise: a pure function of `state = [key, ctr]` and the
    /// element index, fixed by [`crate::rng`] for bit-identical replay.
    RngKeyed {
        /// The `[2]` U32 `[key, ctr]` tensor the draw is keyed on.
        state: ValueId,
        /// The result shape, trace-known.
        shape: Shape,
        /// Which distribution to draw from.
        kind: RngKind,
    },

    // ── channels (the only effects) ──────────────────────────────────────
    /// Consume: full → value, set empty. In-pass register rule:
    /// a take after an in-pass put reads the pending value.
    ChanTake(ChannelIndex),
    /// Peek: full → copy, stays full.
    ChanRead(ChannelIndex),
    /// Fill the pending cell; double-put = last wins.
    /// Defines **0** result ids.
    ChanPut {
        /// The channel written.
        chan: ChannelIndex,
        /// The tensor to store; must match the channel's declared type.
        value: ValueId,
    },

    // ── intrinsics / second-party ────────────────────────────────────────
    /// Materialize a first-party stage-scoped value. Shape and dtype are
    /// trace-known, cross-checked at bind against the registry rule.
    IntrinsicVal {
        /// Which intrinsic to materialize.
        intr: IntrinsicId,
        /// The shape the trace expects; cross-checked at bind.
        shape: Shape,
        /// The dtype the trace expects; cross-checked at bind.
        dtype: Dtype,
    },
    /// Named second-party kernel call: `intrinsics::kernel::*`. Availability
    /// and replayability checked at bind against [`ModelProfile`](crate::registry::ModelProfile).
    KernelCall {
        /// Index into the container's name table.
        name: NameIndex,
        /// Operand tensors, in call order.
        args: Vec<ValueId>,
        /// The declared result shape; the call is opaque, so nothing else
        /// can derive it.
        shape: Shape,
        /// The declared result dtype.
        dtype: Dtype,
    },
    /// Named configuration sink: takes tensors, returns nothing, configures
    /// this pass's forward. Defines **0** result ids.
    SinkCall {
        /// Index into the container's name table.
        name: NameIndex,
        /// Operand tensors, in call order.
        args: Vec<ValueId>,
    },
}

/// How an op touches a channel. See [`Op::channel_use`]. Not a bool:
/// readiness needs a full slot for `Take`/`Read`, an empty one for `Put`.
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
    /// `table_matches_op_metadata` pins this to `Family::Channel`.
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

    /// The name-table index this op names, for rewriting it when a
    /// stage's name table is localized. Pinned by tag, not family.
    pub fn name_index_mut(&mut self) -> Option<&mut NameIndex> {
        match self {
            Op::KernelCall { name, .. } | Op::SinkCall { name, .. } => Some(name),
            _ => None,
        }
    }

    /// Number of SSA ids this op defines. A wrong count would silently shift
    /// every later value id in the trace.
    pub fn result_count(&self) -> u32 {
        match spec(self.tag()) {
            Some(row) => u32::from(row.results),
            None => unreachable!("op tag has no OP_TABLE row"),
        }
    }

    /// This op's [`Family`], read from [`OP_TABLE`].
    pub fn family(&self) -> Family {
        match family_of(self.tag()) {
            Some(family) => family,
            None => unreachable!("op tag has no OP_TABLE row"),
        }
    }

    /// True when the op must survive dead-code elimination and must never
    /// be CSE-merged: touching a channel, or calling second-party code.
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

    /// Where this op's result comes from, a different question than
    /// [`Self::is_effectful`]: [`Op::Rng`] is CSE-mergeable yet [`ValueSource::Device`].
    pub fn value_source(&self) -> ValueSource {
        match self {
            // SinkCall defines no results but is grouped here: not
            // something the host can perform either.
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

/// Declares each op's value-id operands once, deriving both [`Op::operands`]
/// and [`Op::map_operands`] from the same list.
macro_rules! declare_operands {
    (
        fixed { $( $pat:pat => [$($slot:ident),*], )* }
        predicate { $ppat:pat => ($pinput:ident, $ppred:ident) }
        variadic { $( $vpat:pat => $vargs:ident, )* }
    ) => {
        impl Op {
            /// The value ids this op reads, in a stable order (immediates
            /// excluded; `PivotThreshold`'s predicate operand included).
            pub fn operands(&self) -> Vec<ValueId> {
                match self {
                    $( $pat => vec![$(*$slot),*], )*
                    $ppat => vec![*$pinput, $ppred.value()],
                    $( $vpat => $vargs.clone(), )*
                }
            }

            /// Rewrite this op's value-id operands in place — the mutable
            /// counterpart of [`Op::operands`] (immediates untouched).
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

/// What decides an op's result value (not which trace construct defined its
/// SSA id — that's `eta-compiler`'s unrelated `ValueOrigin`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueSource {
    /// The device decides it: a second-party kernel, a device intrinsic, or
    /// the ambient RNG seed. No amount of host-known input produces it.
    Device,
    /// A channel's contents decide it, so the answer depends on what the host
    /// knows about that channel rather than on the op.
    Channel,
    /// A pure function of its operands: host-derivable exactly when they are.
    Operands,
}

/// Op family — the row grouping of the op table above. [`Family::Channel`]
/// is load-bearing: passes use it to ask "does this op have effects".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Takes no tensor operand: a trace-known constant, or the index ramp
    /// [`Op::Iota`] that stands in for one.
    Leaf,
    /// Element-wise arithmetic over matching shapes.
    Map,
    /// Element-wise comparison and boolean logic; always yields Bool.
    CompareLogic,
    /// Element-wise selection between two alternatives — [`Op::Select`]
    /// between two tensors, [`Op::MaskApply`] between a tensor and `-inf`.
    Choice,
    /// Rearranges metadata only, never element values.
    Shape,
    /// Reads or writes elements at computed positions, or derives the
    /// positions and position masks that address them.
    Index,
    /// Folds or scans along the last axis.
    ReduceScan,
    /// Ranks or reorders along the last axis.
    Order,
    /// Matrix products, dispatched to a library kernel.
    Linear,
    /// Draws noise.
    Sampling,
    /// Touches a channel — the only ops with effects.
    Channel,
    /// Names a first- or second-party extension rather than a fixed
    /// operation.
    Intrinsic,
}

/// One field of an op's wire payload, in write order. Each variant names
/// the [`OpWire`](crate::wire::OpWire) field it fills.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireField {
    /// A `u32` SSA value id operand. Appends to `args`.
    Value,
    /// A `u32` channel-table index. Fills `chan`.
    Chan,
    /// A `u32` trace-known immediate. Fills the next free immediate slot
    /// (`imm`, then `imm2`, then `imm3`).
    Imm,
    /// A `u8` [`Dtype`] tag. Fills `dtype`.
    Dtype,
    /// A rank byte followed by that many `u32` dims. Fills `shape`.
    Shape,
    /// A `u8` [`RngKind`] tag. Fills `kind`.
    RngKind,
    /// A `u8` predicate tag plus its `u32` value-id payload. Fills
    /// `pred_tag` / `pred_payload`.
    Predicate,
    /// A `u8` literal dtype tag plus 4 raw payload bytes. Fills
    /// `lit_dtype` / `lit_bits`.
    Literal,
    /// A `u16` name-table index. Fills `name_idx`.
    Name,
    /// A `u16` [`IntrinsicId`]. Fills `intr`.
    Intrinsic,
    /// A `u8` count followed by that many `u32` value ids. Appends to
    /// `args`. Variadic, so it is always the last field.
    Args,
}

/// One op-table row: the declarative identity the generated C++ tables are
/// built from. [`VARIADIC`] means the operand count is on the wire instead.
#[derive(Clone, Copy, Debug)]
pub struct OpSpec {
    /// The op's wire tag — its identity on the wire and in every generated
    /// dispatch table.
    pub tag: u8,
    /// The op's spelling in diagnostics and generated sources.
    pub name: &'static str,
    /// The row grouping this op belongs to.
    pub family: Family,
    /// How many value-id operands the op takes, or [`VARIADIC`] when the
    /// count is a byte on the wire.
    pub val_operands: u8,
    /// How many SSA ids the op defines.
    pub results: u8,
    /// The op body's fields, in encode order. See [`WireField`].
    pub wire: &'static [WireField],
}

/// Variadic marker for [`OpSpec::val_operands`].
pub const VARIADIC: u8 = 0xFF;

/// Declares the op set once and derives [`tags`], [`OP_TABLE`] and
/// [`Op::tag`] from it. Downstream imports [`tags`] rather than respelling a tag.
macro_rules! declare_ops {
    ($($konst:ident = $tag:literal, $name:literal, $family:ident, $operands:expr, $results:expr,
       $rep:expr, $pat:pat, [$($wire:ident),*];)*) => {
        /// The wire tag of every op, by name — the only place an ETA op
        /// tag is spelled as a number.
        pub mod tags {
            $(
                #[doc = concat!("Wire tag of the `", $name, "` op.")]
                pub const $konst: u8 = $tag;
            )*
        }

        /// The op table — one row per wire tag, sorted by tag. The
        /// generated C++ header and any dispatch table must derive from it.
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

        /// One constructible [`Op`] per [`OP_TABLE`] row, in table order.
        /// Not `#[cfg(test)]`, so any crate wanting "for every op" can use it.
        pub fn representatives() -> alloc::vec::Vec<Op> {
            alloc::vec![$($rep,)*]
        }

        impl Op {
            /// This op's wire tag (see [`OP_TABLE`]).
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
        Op::IntrinsicVal { intr: IntrinsicId::Logits, shape: Shape::matrix(1, 8), dtype: Dtype::F32 },
        Op::IntrinsicVal { .. }, [Intrinsic, Dtype, Shape];
    KERNEL_CALL = 0xA1, "kernel_call", Intrinsic, VARIADIC, 1,
        Op::KernelCall { name: 0, args: vec![0, 1], shape: Shape::vector(4), dtype: Dtype::F32 },
        Op::KernelCall { .. }, [Name, Dtype, Shape, Args];
    SINK_CALL = 0xA2, "sink_call", Intrinsic, VARIADIC, 0,
        Op::SinkCall { name: 0, args: vec![0] }, Op::SinkCall { .. }, [Name, Args];
}

/// The table row for a wire tag, or `None` when the tag is not an ETA op.
/// [`OP_TABLE`] is sorted by tag, so this is a binary search.
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

/// The family of a wire tag, or `None` when the tag is not an ETA op.
pub fn family_of(tag: u8) -> Option<Family> {
    spec(tag).map(|row| row.family)
}

