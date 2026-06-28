//! The typed SSA IR data model — **shape-typed, binding-free** (the new-interface
//! convergence; mirrors `interface/inferlet/core/wit/tensor.wit`).
//!
//! A value's type is `ValueType { shape, dtype }` where [`Shape`] is a dim list
//! (`rank = shape.rank()`): scalar = `[]`, vector = `[n]`, matrix = `[m, n]`.
//!
//! A [`SamplingProgram`] mirrors the WIT `program`:
//! * `inputs: Vec<InputDecl>` — **typed input slots** (shape + dtype, *no
//!   binding*). The binding (logits / tensor, readiness) is **attach-time** (the
//!   forward-pass `sampler`), keeping the compiled program reusable — "construct
//!   once, attach by handle". [`Binding`] / [`Readiness`] are shared vocab for
//!   that attach/carrier layer, not part of the program.
//! * `ops: Vec<Op>` — the flat SSA op list. [`Op::Input`]`(index)` materializes
//!   slot `index`; [`Op::Const`] a literal; the rest compute.
//! * `outputs: Vec<OutputDecl>` — value id + [`OutputKind`]. (The WIT front door
//!   carries bare value ids; the bytecode keeps `OutputKind` for the host's
//!   typed-channel marshaling until WS5.)
//!
//! Value ids are implicit: op at list position `p` defines `next_id .. next_id +
//! op.result_count()`, `next_id` starting at 0 (2 ids for [`Op::SortDesc`], 1
//! otherwise). Operands reference earlier ids.
//!
//! See `BYTECODE.md` and the canonical `Op ↔ op-kind` oracle in [`crate::witmap`].

use alloc::vec::Vec;

/// SSA value id.
pub type ValueId = u32;
/// Index of an input slot in [`SamplingProgram::inputs`] (the operand of
/// [`Op::Input`]).
pub type InputIndex = u32;
/// Opaque key identifying a host-visible `tensor` resource (attach/carrier side).
pub type TensorKey = u32;

/// Maximum tensor rank the IR represents inline. Scalar/vector/matrix need ≤ 2;
/// the headroom covers near-term batched shapes. The WIT `list<u32>` lowers to
/// this; lowering rejects rank `> MAX_RANK`.
pub const MAX_RANK: usize = 4;

/// Element type of a value. Tag bytes are stable wire constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    F32 = 0,
    I32 = 1,
    U32 = 2,
    Bool = 3,
}

impl DType {
    pub fn is_float(self) -> bool {
        matches!(self, DType::F32)
    }
    pub fn is_int(self) -> bool {
        matches!(self, DType::I32 | DType::U32)
    }
    pub fn is_numeric(self) -> bool {
        matches!(self, DType::F32 | DType::I32 | DType::U32)
    }
}

/// A logical shape: an ordered list of dimension sizes, `rank = len`.
///
/// Stored inline (fixed capacity [`MAX_RANK`]) so the type stays `Copy`. The
/// **last axis** is the reduce/scan/argmax/pivot axis; a rank-2 `[m, n]` is `m`
/// rows of length `n` (per-row ops iterate axis 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [u32; MAX_RANK],
    rank: u8,
}

impl Shape {
    pub const SCALAR: Shape = Shape { dims: [0; MAX_RANK], rank: 0 };

    /// Build a shape from a dim slice. `None` if `dims.len() > MAX_RANK`.
    pub fn new(dims: &[u32]) -> Option<Shape> {
        if dims.len() > MAX_RANK {
            return None;
        }
        let mut d = [0u32; MAX_RANK];
        d[..dims.len()].copy_from_slice(dims);
        Some(Shape { dims: d, rank: dims.len() as u8 })
    }
    pub fn vector(n: u32) -> Shape {
        Shape::new(&[n]).unwrap()
    }
    pub fn matrix(m: u32, n: u32) -> Shape {
        Shape::new(&[m, n]).unwrap()
    }

    pub fn dims(&self) -> &[u32] {
        &self.dims[..self.rank as usize]
    }
    pub fn rank(&self) -> usize {
        self.rank as usize
    }
    pub fn is_scalar(&self) -> bool {
        self.rank == 0
    }
    pub fn numel(&self) -> u64 {
        self.dims().iter().map(|&d| d as u64).product()
    }
    pub fn last_len(&self) -> Option<u32> {
        self.dims().last().copied()
    }
    pub fn rows(&self) -> u32 {
        match self.rank as usize {
            0 | 1 => 1,
            r => self.dims[..r - 1].iter().product(),
        }
    }
    /// The shape with the last axis dropped (a reduction's result), or `None`
    /// for a scalar.
    pub fn drop_last(&self) -> Option<Shape> {
        if self.rank == 0 {
            return None;
        }
        Shape::new(&self.dims[..self.rank as usize - 1])
    }
}

/// A value's full type: [`Shape`] + [`DType`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueType {
    pub shape: Shape,
    pub dtype: DType,
}

impl ValueType {
    pub const fn new(shape: Shape, dtype: DType) -> Self {
        Self { shape, dtype }
    }
    pub fn scalar(dtype: DType) -> Self {
        Self { shape: Shape::SCALAR, dtype }
    }
    pub fn vector(n: u32, dtype: DType) -> Self {
        Self { shape: Shape::vector(n), dtype }
    }
}

/// A typed input slot (the WIT `record input { shape, dtype, ready }`). The
/// source binding (which slot is logits vs a host tensor) stays attach-time, but
/// the **readiness** is a program property: a `Late` slot is injected per-fire
/// before its first consuming op (e.g. a grammar mask computed post-logits), so
/// a late-input program is a distinct recognized shape. Readiness is encoded
/// additively in **bit 7 of the dtype byte**, so v4 bytecode decodes as `Submit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputDecl {
    pub shape: Shape,
    pub dtype: DType,
    pub ready: Readiness,
}

impl InputDecl {
    /// A `Submit`-readiness input slot (the common case).
    pub const fn new(shape: Shape, dtype: DType) -> Self {
        Self { shape, dtype, ready: Readiness::Submit }
    }
    /// An input slot with explicit readiness (`Late` = injected per-fire before
    /// its first consuming op).
    pub const fn with_ready(shape: Shape, dtype: DType, ready: Readiness) -> Self {
        Self { shape, dtype, ready }
    }
    pub fn ty(&self) -> ValueType {
        ValueType::new(self.shape, self.dtype)
    }
}

/// The closed primitive operation set, shape-typed — **1:1 with the WIT
/// `op-kind`** (see [`crate::witmap`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    /// Materialize input slot `index` ([`SamplingProgram::inputs`]) as a value.
    Input(InputIndex),
    /// A compile-time constant scalar; its type is `scalar(literal.dtype())`.
    Const(Literal),

    // -- unary elementwise map --
    Exp(ValueId),
    Log(ValueId),
    Neg(ValueId),
    Recip(ValueId),
    Abs(ValueId),
    Sign(ValueId),

    // -- binary elementwise map (scalar operand broadcasts) --
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    MaxElem(ValueId, ValueId),
    MinElem(ValueId, ValueId),
    Gt(ValueId, ValueId),
    Ge(ValueId, ValueId),
    Eq(ValueId, ValueId),
    Select {
        cond: ValueId,
        a: ValueId,
        b: ValueId,
    },

    // -- reductions over the last axis (per-row for rank ≥ 2) --
    ReduceSum(ValueId),
    ReduceMax(ValueId),
    ReduceMin(ValueId),
    /// Argmax over the last axis → `I32` token id (per-row for rank ≥ 2).
    ReduceArgmax(ValueId),

    /// Shape-directed broadcast: replicate `value` to `shape`. `value`'s dims,
    /// **left-aligned** against `shape` (trailing axes padded with `1`), must
    /// equal the target or be `1` (replicated). Folds scalar-broadcast
    /// (`[] → [..]`) and per-row broadcast (`[m] → [m, n]`).
    Broadcast {
        value: ValueId,
        shape: Shape,
    },

    // -- scans over the last axis (per-row for rank ≥ 2) --
    CumSum(ValueId),
    CumProd(ValueId),

    /// Descending sort over a `[n]` vector → two consecutive results, value-
    /// first: `r` = sorted values (F32 `[n]`), `r+1` = original indices (U32).
    SortDesc(ValueId),
    /// Sort-free top-k / top-p / min-p mask (Bool, same shape; per-row for rank
    /// ≥ 2). See [`Predicate`].
    PivotThreshold {
        input: ValueId,
        predicate: Predicate,
    },

    // -- indexing --
    /// 1-D gather: `out[j] = src[idx[j]]`. `src` `[n]`, `idx` `[k]` int → `[k]`.
    /// Invalid index → fill-0.
    Gather { src: ValueId, idx: ValueId },
    /// Per-row column pick: `out[i] = src[i, idx[i]]`. `src` `[m, n]`, `idx`
    /// `[m]` int → `[m]`. The lossless accept-ratio `p[i, draft[i]]`. Invalid
    /// per-row column → fill-0. **(Not** whole-row select.)
    GatherRow { src: ValueId, idx: ValueId },
    /// `out = base; out[idx[k]] += vals[k]`; invalid index → skip; duplicates
    /// accumulate; `base` numeric.
    ScatterAdd {
        base: ValueId,
        idx: ValueId,
        vals: ValueId,
    },
    /// `out = base; out[idx[k]] = vals[k]`; invalid index → skip; last-write-wins.
    ScatterSet {
        base: ValueId,
        idx: ValueId,
        vals: ValueId,
    },

    /// Apply a packed allowed-token bitmask to logits: `out[j] = bit_j(mask) ?
    /// logits[j] : −∞`, where `bit_j = (mask[j>>5] >> (j&31)) & 1` (word
    /// `j/32`, bit `j%32`). **Bit 1 = allowed → pass-through, bit 0 =
    /// disallowed → `−∞`.** `logits` is `[n]`; `mask` is `[ceil(n/32)]` U32
    /// (word-indexed, NOT broadcast); result ≡ `logits` (shape + dtype). The
    /// de-hardwired grammar/constrained-decode mask channel (the matcher emits
    /// the packed bits; this op applies them in-program).
    MaskApply { logits: ValueId, mask: ValueId },

    /// Per-element noise of `shape` (F32). **No seed operand:** the per-fire seed
    /// is ambient (the runtime's per-row `sample_seed`), folded into every key by
    /// codegen. `stream` is a static per-op salt decorrelating multiple `Rng`
    /// ops (`stream = 0` ≡ `seed_eff = S ^ 0xA5A5A5A5`, `sample_temp.cu` parity).
    /// The **axis is a lowering choice**, not a program property: standard
    /// samplers lower batch-axis (per-row seed `S[r]` over in-row `col`,
    /// `sample_temp.cu` parity); the flattened `row*len+col` single-seed form is
    /// spec-verify-only.
    Rng {
        stream: u32,
        shape: Shape,
        kind: RngKind,
    },
}

impl Op {
    /// Number of SSA ids this op defines: 2 for [`Op::SortDesc`], 1 otherwise.
    pub fn result_count(&self) -> u32 {
        match self {
            Op::SortDesc(_) => 2,
            _ => 1,
        }
    }

    /// The [`ValueId`]s this op reads, in a stable order. Leaves (`Input`,
    /// `Const`) and immediates (input-index, shape, stream) are excluded; the
    /// value-id predicate operands (top-k `k`, top-p `p`, min-p `thr`) are
    /// included.
    pub fn operands(&self) -> Vec<ValueId> {
        use alloc::vec;
        match *self {
            Op::Input(_) | Op::Const(_) | Op::Rng { .. } => Vec::new(),

            Op::Exp(a)
            | Op::Log(a)
            | Op::Neg(a)
            | Op::Recip(a)
            | Op::Abs(a)
            | Op::Sign(a)
            | Op::ReduceSum(a)
            | Op::ReduceMax(a)
            | Op::ReduceMin(a)
            | Op::ReduceArgmax(a)
            | Op::CumSum(a)
            | Op::CumProd(a)
            | Op::SortDesc(a)
            | Op::Broadcast { value: a, .. } => vec![a],

            Op::Add(a, b)
            | Op::Sub(a, b)
            | Op::Mul(a, b)
            | Op::Div(a, b)
            | Op::MaxElem(a, b)
            | Op::MinElem(a, b)
            | Op::Gt(a, b)
            | Op::Ge(a, b)
            | Op::Eq(a, b)
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
        }
    }
}

/// Threshold predicate for [`Op::PivotThreshold`] (top-k / top-p / min-p).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predicate {
    /// top-k: keep the top `k` — a value id (host-submit `U32` scalar, or a
    /// per-row `[rows]` `U32` vector for a matrix input). De-hardwired like
    /// top-p `p` / min-p `thr`, so the program bytecode is k-invariant (`k` is
    /// supplied at submit, never a baked immediate).
    RankLe(ValueId),
    /// top-p: inclusive nucleus to mass `p` (a Scalar-F32 value id).
    CummassLe(ValueId),
    /// min-p: keep `>= thr` (a Scalar-F32 value id, e.g. `p·max_prob`).
    ProbGe(ValueId),
}

/// Distribution sampled by [`Op::Rng`]. Tag bytes are stable wire constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RngKind {
    Uniform = 0,
    Gumbel = 1,
}

/// A compile-time constant scalar (the payload of [`Op::Const`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Literal {
    F32(f32),
    I32(i32),
    U32(u32),
    Bool(bool),
}

impl Literal {
    pub fn dtype(self) -> DType {
        match self {
            Literal::F32(_) => DType::F32,
            Literal::I32(_) => DType::I32,
            Literal::U32(_) => DType::U32,
            Literal::Bool(_) => DType::Bool,
        }
    }
}

// ===========================================================================
// Attach / carrier vocab (NOT part of the program — see module docs)
// ===========================================================================

/// When a [`Binding::Tensor`] value becomes ready. An **attach/carrier** concern
/// (the program itself is binding-free); shared vocab for that layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Readiness {
    /// Ready at submit; refreshed per fire without recompile.
    Submit = 0,
    /// Ready after submit, before its first consuming op; miss = skip.
    Late = 1,
    /// Late-class, **driver-internal**: the verify's draft INPUT (#31 self-spec).
    /// The verify reads forward-(N−1)'s drafts that the host refed as forward-N's
    /// verify input, resident in the driver's token buffer at the `[k,vocab]`
    /// matrix base (`pi.tokens + sample_row + 1`) — the drafts ARE the verify
    /// input (populated at forward start, before the sampling-IR fires; no new
    /// materialization). The resolver binds flag-first; NO host upload, NO
    /// `sampling_late_device_*` ptr. Distinct from [`Late`] (#27 host-uploaded
    /// device-alias). Late-class for barriers/codegen (`HostLate`), so it rides
    /// `READY_LATE_BIT` in bytecode (degrades to `Late` for a stale reader); the
    /// structured manifest carries the precise role. (The reciprocal MTP
    /// draft-OUTPUT exposure is a drafter-populated program-output slot, NOT an
    /// IR-read readiness — an IR program fires upstream of the drafter.)
    SelfSpecDraftInput = 2,
}

/// How an input slot is bound at forward-pass attach time (the WIT
/// `input-binding`). **Attach/carrier vocab**, not part of [`SamplingProgram`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    /// The LM-head logits intrinsic (positions resolved host-side).
    Logits,
    /// The speculator's **draft** logits intrinsic (de-hardwired speculation).
    /// Source-selects the draft rows of `ws.logits` (M=1 ⇒ row 0) rather than a
    /// separate buffer — host→driver as `IntrinsicKind::MtpLogits`, resolving to
    /// a draft-row offset within `ws.logits`. A **distinct manifest variant**
    /// (not a kind-byte on `Logits`) so a stale reader loud-rejects rather than
    /// misparses a layout change. Manifest-only, additive — not in the bytecode.
    MtpLogits,
    /// A host-visible `tensor` resource, keyed, ready per [`Readiness`].
    Tensor {
        key: TensorKey,
        ready: Readiness,
    },
}

/// Semantic kind of a declared output. The host marshals the value into the
/// matching response channel; the driver reads and ignores it. The WIT front
/// door carries bare value ids (typed-tensor outputs); the bytecode keeps this
/// for host marshaling until WS5.
///
/// **Frozen wire discriminants — do not reorder/renumber** (mirror the SDK enum).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OutputKind {
    Token = 0,
    Distribution = 1,
    Logits = 2,
    Logprobs = 3,
    Entropy = 4,
    Scalar = 5,
    Embedding = 6,
}

impl OutputKind {
    pub fn to_u8(self) -> u8 {
        self as u8
    }
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => OutputKind::Token,
            1 => OutputKind::Distribution,
            2 => OutputKind::Logits,
            3 => OutputKind::Logprobs,
            4 => OutputKind::Entropy,
            5 => OutputKind::Scalar,
            6 => OutputKind::Embedding,
            _ => return None,
        })
    }
    /// Soft check: `Token` is an integer token id; every other kind is `F32`.
    pub fn accepts_dtype(self, dtype: DType) -> bool {
        match self {
            OutputKind::Token => dtype.is_int(),
            _ => dtype == DType::F32,
        }
    }

    /// Infer the cutover [`OutputKind`] from a value's dtype — the host decoder
    /// uses this when the WIT front door carries **bare value-id** outputs (no
    /// kind): an integer value is a [`Token`](Self::Token), an `F32` value a
    /// [`Scalar`](Self::Scalar). That is the Submit-for-MVP routing the host
    /// marshaling keys on (`Token → tokens`, `Scalar → entropies`); a `Bool`
    /// cannot be an output (`None`). The finer float kinds
    /// (`Distribution`/`Logits`/…) are the typed WS5 end-state, not inferable
    /// from dtype alone. Inverse-compatible with [`accepts_dtype`](Self::accepts_dtype).
    pub fn from_dtype(dtype: DType) -> Option<Self> {
        match dtype {
            DType::I32 | DType::U32 => Some(OutputKind::Token),
            DType::F32 => Some(OutputKind::Scalar),
            DType::Bool => None,
        }
    }
}

/// A declared output: which value id to expose plus its [`OutputKind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputDecl {
    pub value: ValueId,
    pub kind: OutputKind,
}

impl OutputDecl {
    pub const fn new(value: ValueId, kind: OutputKind) -> Self {
        Self { value, kind }
    }
}

/// A complete sampling program — mirrors the WIT `program`: typed input slots +
/// a flat SSA op list + declared outputs. Binding-free (binding is attach-time).
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingProgram {
    pub inputs: Vec<InputDecl>,
    pub ops: Vec<Op>,
    pub outputs: Vec<OutputDecl>,
}

impl SamplingProgram {
    /// Validate SSA well-formedness and per-op shape/dtype rules.
    pub fn validate(&self) -> Result<(), crate::validate::ValidationError> {
        crate::validate::validate(self)
    }

    /// Lower to the flat versioned bytecode.
    pub fn encode(&self) -> Vec<u8> {
        crate::bytecode::encode(self)
    }
}
