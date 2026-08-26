//! Values: `Def` × `Ty`. Where a value comes from and what it is are
//! orthogonal, and everything is a `ValueId` — no `WeightId`, no separate
//! cache handle type.

use serde::{Deserialize, Serialize};

use crate::cond::Cond;

/// One id space for every value in a plan: op outputs, weights, cache
/// bindings, runtime inputs, merges. Indexes `Plan::values`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// Element type as data, not a generic: monomorphization's guarantee moved to
/// the trace-time validator plus a launch-site match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
    I32,
    U32,
    U8,
}

/// The whole surviving shape algebra. Symbolic dims are sized by engine
/// budgets (`Tokens` → max_tokens, `Lanes` → max_lanes) when the arena is cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    Const(u64),
    /// This fire's token count.
    Tokens,
    /// MoE routed rows: tokens × top_k.
    TokensTimes(u32),
    /// Request count (geometry vectors).
    Lanes,
    /// Indptr-shaped: lanes + 1.
    LanesPlus(u32),
}

/// The kinds of host-owned plan objects an op may define. The payload is
/// backend-opaque; only the kind is IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructKind {
    AttnDecodePlan,
    AttnPrefillPlan,
    AttnPrefillPlanSm90,
    MlaPlan,
}

/// Which geometry vector of a cache space a runtime input binds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeomKind {
    Indptr,
    Indices,
    SeqLens,
    LastPageLen,
}

/// What the driver binds each fire. Geometry is a declared input, not implicit
/// driver state: cache ops become pure functions of visible inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeInput {
    Tokens,
    Positions,
    Geometry { cache: u32, kind: GeomKind },
}

/// Raggedness is not a `Ty` — a leading symbolic `Dim` means the value is
/// fire-aligned and viewable through the fire's shared indptr.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ty {
    Tensor { shape: Vec<Dim>, dtype: Dtype },
    /// Opaque, host-owned, outside the arena; sized at plan-build time.
    Struct(StructKind),
}

/// Where a value comes from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Def {
    /// Bound by the driver each fire.
    Input(RuntimeInput),
    /// Index into `Plan::params`. Weights are plain values — no `WeightId`;
    /// the compiler skips non-`Op` defs during allocation.
    Weight(u32),
    /// Index into `Plan::caches` — storage only; geometry arrives as `Input`.
    /// Distinct from `Weight` because caches are written during a fire.
    Cache(u32),
    /// Output of `Plan::nodes[i]`; the index is cross-checked by the validator.
    Op(u32),
    /// φ-node: data, never dispatched — the compiler resolves it to slot
    /// aliasing.
    Merge(Vec<(ValueId, Cond)>),
}

/// One row of `Plan::values`: provenance and type, orthogonal by construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueDecl {
    pub def: Def,
    pub ty: Ty,
}
