//! Values: `Def` × `Ty`. Where a value comes from and what it is are
//! orthogonal, and everything is a `ValueId` — no `WeightId`, no separate
//! cache handle type.

use serde::{Deserialize, Serialize};

use crate::guard::Guard;

/// One id space for every value in a plan: op outputs, weights, cache
/// bindings, runtime inputs, merges. Indexes `Trace::values`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// Element type as data, not a generic: monomorphization's guarantee moved to
/// the trace-time validator plus a launch-site match. The one such enum in the
/// stack — it names storage representations as well as compute elements, so a
/// weight plane, a kv page and a tensor all say what they hold in one spelling.
///
/// `Mxfp4` is a weight plane's 32-code block packed to 16 bytes; the companion
/// `.scales` plane beside it is `E8m0`, which is only ever that companion and
/// never something an author declares. `Fp8E4m3`, `I8` and `Fp4` name kv-page
/// quant schemes. What a scheme's granularity is (per-tensor vs per-token-head)
/// and how wide an fp4 block runs are not facts about the element — they are
/// sibling fields of the cache row that chose the scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
    I32,
    U32,
    U8,
    I8,
    Fp8E4m3,
    Fp4,
    Mxfp4,
    E8m0,
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

/// Which geometry vector of a cache space a runtime input binds. Each kind
/// says which op family reads it, so a fire owes exactly what its plan names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeomKind {
    /// Per-lane page-list bounds; read by the plan ops (`attention.plan_*`, `mla.plan`).
    Indptr,
    /// The flat page-id list the indptr bounds; read by the plan ops.
    Indices,
    /// Per-lane sequence lengths; read by the plan ops.
    SeqLens,
    /// Per-lane fill of the last page; read by the plan ops.
    LastPageLen,
    /// Per-lane total kv length; read by the plan builders (`attention.plan_*`, `mla.plan`).
    KvLen,
    /// Graph-padding row mask; read by the pool boundary ops (`pool.boundary_*`).
    RowValid,
    /// Token→lane map; read by `pool.attention_lse` (and the metal fire tables).
    RequestOfToken,
    /// Per-token destination page of a kv write; read by the `kv_append` ops.
    WritePage,
    /// Per-token in-page offset of a kv write; read by the `kv_append` ops.
    WriteOffset,
}

/// What the driver binds each fire. Geometry is a declared input, not implicit
/// driver state: cache ops become pure functions of visible inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeInput {
    Tokens,
    Positions,
    /// Custom attention mask bits for a kv space; read by `attention.masked`.
    Mask { space: u32 },
    /// One geometry vector of a cache space; `space` matches the group the
    /// caches declare (`CacheRow::Kv::space`).
    Geometry { space: u32, kind: GeomKind },
    /// Which adapter bank each token row routes to (design §8); read by
    /// `linear.lora_correct`. `i32`, one entry per token row, `-1` for the
    /// base model.
    ///
    /// **BARE, LIKE `Tokens` AND `Positions`, AND NOT KEYED BY ANYTHING.**
    /// `Mask` and `Geometry` carry a `space` because what they describe is a
    /// page-id space's own — one mask slab per readable extent, one indptr per
    /// page table. An adapter is a property of the REQUEST: a lane routes to
    /// one adapter and every correction site in the plan reads the same id for
    /// that lane's rows, so a per-site or per-bank spelling would be the same
    /// vector interned under `sites` names, free to disagree with itself. One
    /// vector, staged once, read by every site — which is also what makes the
    /// zero-adapter fire's cost exactly zero: nothing is staged when no lane
    /// carries one.
    AdapterRoutes,
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
    /// Index into `Trace::params`. Weights are plain values — no `WeightId`;
    /// the compiler skips non-`Op` defs during allocation.
    Weight(u32),
    /// Index into `Trace::caches` — storage only; geometry arrives as `Input`.
    /// Distinct from `Weight` because caches are written during a fire.
    Cache(u32),
    /// Output of `Trace::nodes[i]`; the index is cross-checked by the validator.
    Op(u32),
    /// φ-node: data, never dispatched — the compiler resolves it to slot
    /// aliasing.
    Merge(Vec<(ValueId, Guard)>),
}

/// One row of `Trace::values`: provenance and type, orthogonal by construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueDecl {
    pub def: Def,
    pub ty: Ty,
}
