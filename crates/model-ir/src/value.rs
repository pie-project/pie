//! Values: `Def` (provenance) × `Ty` (type), orthogonal; every value is a
//! `ValueId` — no `WeightId`, no separate cache handle type.

use serde::{Deserialize, Serialize};

use crate::guard::Guard;

/// One id space for every value in a plan: op outputs, weights, cache
/// bindings, runtime inputs, merges. Indexes `Trace::values`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// Element type as data — re-export of [`dtype::Dtype`], the IR's one name
/// for an element type.
pub use dtype::Dtype;
/// The tiled affine layout's geometry, beside the [`Dtype`] that names it.
/// `model_dsl::Weight::planes` sizes a repacked weight's three planes with
/// these; `checkpoint` checks a declared repack target against the same two
/// numbers.
pub use dtype::{TILED_BAND, TILED_STEP};

/// The shape algebra's symbolic dims, sized by runtime budgets (`Tokens` →
/// max_tokens, `Lanes` → max_lanes, `Patches` → max_patches) when the arena
/// is cut. Which axis a value lives on is read off its type
/// ([`Dim::axis`]) rather than declared beside it.
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
    /// This fire's patch count — the rows of the vision tower's window,
    /// concatenated over every image every lane submitted. Not a subset of
    /// the token rectangle: the patch axis gets its own seriation, bucket
    /// ladder and capture unit.
    Patches,
    /// The patch axis's own lane space: how many images this fire carries.
    /// `Images` is to [`Patches`](Dim::Patches) what [`Lanes`](Dim::Lanes)
    /// is to [`Tokens`](Dim::Tokens) — separate because a lane with no
    /// image contributes a lane and no image.
    Images,
    /// Indptr-shaped on the patch axis: `images + 1`.
    ImagesPlus(u32),
}

/// Which row space a symbolic dim sizes — the discriminator every per-axis
/// table is keyed by. Derived, never declared: [`Dim::axis`] reads it off
/// the type a model text already wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RowAxis {
    /// The token rectangle: `Tokens`, `TokensTimes(k)`, `Lanes`, `LanesPlus(k)`.
    Tokens,
    /// The patch rectangle: `Patches`, `Images`, `ImagesPlus(k)`.
    Patches,
}

impl RowAxis {
    /// The axis every plan has, and the one a plan that names no other is
    /// entirely made of.
    pub const PRIMARY: RowAxis = RowAxis::Tokens;

    /// Every axis, in discriminant order — what a pass that owes an answer
    /// per row space iterates, and what [`PerAxis`] is laid out along.
    /// `ALL[axis as usize] == axis`, so a `PerAxis` entry is reachable by
    /// the same integer the variant is.
    pub const ALL: [RowAxis; 2] = [RowAxis::Tokens, RowAxis::Patches];

    /// How many row spaces there are — [`ALL`](RowAxis::ALL)'s length, and
    /// the width of every [`PerAxis`].
    pub const COUNT: usize = RowAxis::ALL.len();

    /// The name a refusal or a ledger line spells this axis with.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            RowAxis::Tokens => "tokens",
            RowAxis::Patches => "patches",
        }
    }
}

/// One value per row axis, addressed by the axis, so "which axis" is an
/// index rather than a pair of hand-kept fields. Total over the axes: every
/// axis has a value always, so a text-only fire's patch entry is the zero
/// window rather than a missing one. `T` is opaque to this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PerAxis<T>([T; RowAxis::COUNT]);

impl<T> PerAxis<T> {
    /// One value per axis, in [`RowAxis::ALL`]'s order.
    ///
    /// An array rather than one argument per axis: adding a variant then
    /// fails to compile at the fill site, where the caller has the value
    /// in hand to state it.
    pub const fn new(values: [T; RowAxis::COUNT]) -> PerAxis<T> {
        PerAxis(values)
    }

    /// One value per axis, computed from the axis.
    pub fn from_fn(mut f: impl FnMut(RowAxis) -> T) -> PerAxis<T> {
        PerAxis(RowAxis::ALL.map(&mut f))
    }

    /// The same axes, carrying something else — `f` applied entry by entry,
    /// in [`RowAxis::ALL`]'s order.
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> PerAxis<U> {
        PerAxis(self.0.map(&mut f))
    }

    /// The entries in axis order, for a caller that wants to walk them
    /// without naming the axes.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// `(axis, value)` per axis, ascending — the loop a pass that owes one
    /// answer per row space writes instead of two blocks.
    pub fn iter(&self) -> impl Iterator<Item = (RowAxis, &T)> {
        RowAxis::ALL.into_iter().zip(self.0.iter())
    }
}

impl<T: Clone> PerAxis<T> {
    /// The same value on every axis.
    #[must_use]
    pub fn splat(value: T) -> PerAxis<T> {
        PerAxis::from_fn(|_| value.clone())
    }
}

impl<T> core::ops::Index<RowAxis> for PerAxis<T> {
    type Output = T;

    /// `table[axis]`, in bounds by construction: the array is
    /// [`RowAxis::COUNT`] wide.
    fn index(&self, axis: RowAxis) -> &T {
        &self.0[axis as usize]
    }
}

impl<T> core::ops::IndexMut<RowAxis> for PerAxis<T> {
    fn index_mut(&mut self, axis: RowAxis) -> &mut T {
        &mut self.0[axis as usize]
    }
}

impl Dim {
    /// The row space this dim sizes, or `None` for a [`Const`](Dim::Const)
    /// — a fixed block is not fire-aligned and so belongs to no axis.
    ///
    /// `Lanes`/`LanesPlus` answer `RowAxis::Tokens`;
    /// [`Images`](Dim::Images)/[`ImagesPlus`](Dim::ImagesPlus) answer
    /// `RowAxis::Patches` for the same reason.
    #[must_use]
    pub fn axis(self) -> Option<RowAxis> {
        match self {
            Dim::Const(_) => None,
            Dim::Tokens | Dim::TokensTimes(_) | Dim::Lanes | Dim::LanesPlus(_) => {
                Some(RowAxis::Tokens)
            }
            Dim::Patches | Dim::Images | Dim::ImagesPlus(_) => Some(RowAxis::Patches),
        }
    }
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

/// What the engine binds each fire. Geometry is a declared input, not implicit
/// engine state: cache ops become pure functions of visible inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeInput {
    Tokens,
    Positions,
    /// Trunk's triple-wide position stream: `[Dim::Tokens, 3]` i32, one
    /// `(t, h, w)` per token row, read by `elementwise.rope_mrope`. A
    /// second stream rather than a widened `Positions`, so a text that
    /// doesn't rotate this way pays nothing extra.
    MropePositions,
    /// Custom attention mask bits for a kv space; read by `attention.masked`.
    Mask { space: u32 },
    /// One geometry vector of a cache space; `space` matches the group the
    /// caches declare (`CacheRow::Kv::space`).
    Geometry { space: u32, kind: GeomKind },
    /// Which adapter bank each token row routes to; read by
    /// `linear.lora_correct`. `i32`, one entry per token row, `-1` for the
    /// base model. Bare (not keyed): an adapter is a property of the
    /// request, not of a page-id space.
    AdapterRoutes,
    /// The fire's patch rows, pre-unfolded: `[Dim::Patches, C·T·P²]`, one
    /// row per patch, every image of every lane concatenated in the fire's
    /// patch order. Decode/resize happen host-side.
    Patches,
    /// The patch axis's own indptr: `[Dim::ImagesPlus(1)]` `i32`, where
    /// image `i`'s patch rows are `[segments[i], segments[i + 1])` of the
    /// fire's patch rectangle. Read by `attention.dense`.
    PatchSegments,
    /// Where each row of the tower's output lands in the token rectangle:
    /// `i32`, one destination token row per tower row, read by
    /// `layout.scatter_rows`. The only vector crossing both axes, and the
    /// only one the fire path validates before launch (an out-of-range
    /// entry is an out-of-bounds device write nothing else would catch).
    PatchRoutes,
    /// The tower's own position stream: `[Dim::Patches, 3]` `i32`, one
    /// `(t, h, w)` per patch row, each patch's position in its own image's
    /// grid. The time column is zero for image input, reserved for video.
    PatchPositions,
    /// Which row of the learned position table each patch reads:
    /// `[Dim::Patches]` `i32` on the native grid, `[Dim::Patches, taps]`
    /// when resampled. Read as a gather through `layout.embed`; `taps` is
    /// the interpolation width (1 native, 4 bilinear, 16 bicubic).
    PatchEmbedRows,
    /// How much of each tap — `[Dim::Patches, taps]` `f32`, read by
    /// `layout.embed_weighted`. `f32` because this is geometry, not the
    /// activation element. A text on the native grid never declares this.
    PatchEmbedWeights,
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
    /// Bound by the engine each fire.
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

#[cfg(test)]
mod tests {
    use super::{Dim, PerAxis, RowAxis};

    /// The index is the variant's own integer, both ways: every axis reads
    /// back what was filled at it, and `Dim::axis` lands each symbolic dim
    /// on the entry its row space owns.
    #[test]
    fn a_per_axis_reads_back_what_each_axis_was_filled_with() {
        let mut table = PerAxis::new(["tokens", "patches"]);
        assert_eq!(table[RowAxis::Tokens], "tokens");
        assert_eq!(table[RowAxis::Patches], "patches");
        assert_eq!(table.as_slice().len(), RowAxis::COUNT);

        // The array's order is the enum's, so the index is direct.
        for (at, axis) in RowAxis::ALL.into_iter().enumerate() {
            assert_eq!(table.as_slice()[at], axis.name());
        }

        // Writing through the index is the same address.
        table[RowAxis::Patches] = "second";
        assert_eq!(table[RowAxis::Patches], "second");
        assert_eq!(table[RowAxis::Tokens], "tokens", "the other axis stood still");

        // `from_fn` is the same fill said once.
        let named = PerAxis::from_fn(RowAxis::name);
        assert_eq!(named[RowAxis::Tokens], "tokens");
        assert_eq!(named[RowAxis::Patches], "patches");

        // What a cut indexes with: every symbolic dim's own axis.
        let cut = PerAxis::new([10u32, 20]);
        for (dim, want) in [
            (Dim::Tokens, 10),
            (Dim::TokensTimes(2), 10),
            (Dim::Lanes, 10),
            (Dim::LanesPlus(1), 10),
            (Dim::Patches, 20),
            (Dim::Images, 20),
            (Dim::ImagesPlus(1), 20),
        ] {
            assert_eq!(cut[dim.axis().expect("a symbolic dim names a row space")], want);
        }
        assert_eq!(Dim::Const(8).axis(), None, "a fixed block belongs to no axis");
    }
}
