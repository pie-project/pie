//! Values: `Def` × `Ty`. Where a value comes from and what it is are
//! orthogonal, and everything is a `ValueId` — no `WeightId`, no separate
//! cache handle type.

use serde::{Deserialize, Serialize};

use crate::guard::Guard;

/// One id space for every value in a plan: op outputs, weights, cache
/// bindings, runtime inputs, merges. Indexes `Trace::values`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// Element type as data — [`dtype::Dtype`], re-exported at the path the IR has
/// always spelled it.
///
/// The enum ITSELF stood here, and it was one of two: the loader's
/// `checkpoint::types::DType` named the same thing over a wider vocabulary
/// (the wide ints, `Bool`) and a narrower one (no `Fp4`, no `Mxfp4`), so every
/// edge between the IR and a checkpoint was a hand-written table. There is one
/// enum now, in a leaf crate that deps nothing, and `model_ir::Dtype` is a name
/// for it rather than a second one.
pub use dtype::Dtype;
/// The tiled affine layout's geometry, beside the [`Dtype`] that names it —
/// `model_dsl::Weight::planes` sizes a repacked weight's three planes with
/// these, and `checkpoint` checks a declared repack target against the same
/// two numbers.
pub use dtype::{TILED_BAND, TILED_STEP};

/// The whole surviving shape algebra. Symbolic dims are sized by runtime
/// budgets (`Tokens` → max_tokens, `Lanes` → max_lanes, `Patches` →
/// max_patches) when the arena is cut.
///
/// **A VARIANT IS A ROW SPACE, AND THAT IS WHAT MAKES THE AXIS A SYMBOL**
/// (multimodal §5.1). `Tokens` and `Lanes` are two spellings of ONE rectangle
/// — a lane is a request of the token rectangle, an indptr closes over it —
/// so every value they size is windowed by the same seriation and carried by
/// the same descriptor row count. `Patches` is not: a lane may carry zero
/// images or three, so patch rows and token rows do not break at the same
/// places, and nothing about the token window can be read off a patch one.
/// Which axis a value lives on is therefore READ OFF ITS TYPE
/// ([`Dim::axis`]) rather than declared beside it, and the capture-unit
/// partition is derived from that and never from a flag.
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
    /// **THE SECOND ROW AXIS.** This fire's patch count — the rows of the
    /// vision tower's window, concatenated over every image every lane
    /// submitted.
    ///
    /// NOT A SUBSET OF THE TOKEN RECTANGLE, which is the whole of why it is a
    /// variant rather than a `TokensTimes`. The tower's "lanes" are IMAGES:
    /// one request contributes as many patch runs as it submitted images, and
    /// a text-only request contributes none. So the merged-prefix-sum
    /// invariant the token axis is composed under — rows and lanes break at
    /// the same places — does not hold here, and the patch axis gets its own
    /// seriation, its own bucket ladder and its own capture unit.
    Patches,
    /// **THE PATCH AXIS'S LANE SPACE: IMAGES.** How many images this fire
    /// carries, over every lane in it.
    ///
    /// M1 promised this variant in as many words — "the patch axis has its
    /// own lane space — images — and when it needs a bounds vector it will
    /// name one, in its own variant, for the same reason this one does" — and
    /// the bounds vector arrived with the tower's one real kernel:
    /// `attention.dense` is block-diagonal PER IMAGE and reads an indptr of
    /// `images + 1` entries to know where one image's patch run ends and the
    /// next begins.
    ///
    /// `Images` is to [`Patches`](Dim::Patches) exactly what [`Lanes`](Dim::Lanes)
    /// is to [`Tokens`](Dim::Tokens) — a count of the requests of a row
    /// rectangle, cut by the same window pair its rows are — and it is a
    /// SEPARATE variant from `Lanes` for the reason the row spaces are
    /// separate: a lane carrying no image contributes a lane and no image, so
    /// the two counts are two numbers in any mixed fire.
    Images,
    /// Indptr-shaped on the patch axis: `images + 1`.
    ImagesPlus(u32),
}

/// Which row space a symbolic dim sizes — the discriminator every per-axis
/// table is keyed by.
///
/// **DERIVED, NEVER DECLARED.** There is no axis field on a value, a node or
/// a region: [`Dim::axis`] reads it off the type a model text already wrote,
/// which is what keeps a second row axis from being a second vocabulary a
/// text could get wrong. A third instance is already spoken for — the
/// per-key attention-score extent (attn-score §6.1) — and it lands as one
/// more variant here plus one more ceiling, not as a parallel invention.
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

    /// **EVERY AXIS, IN DISCRIMINANT ORDER** — what a pass that owes an
    /// answer per row space iterates, and what [`PerAxis`] is laid out along.
    ///
    /// The order is the enum's own, so `ALL[axis as usize] == axis` and a
    /// `PerAxis` entry is reachable by the same integer the variant is. That
    /// equality is what makes the array an INDEX rather than a table with a
    /// lookup in front of it, and it is why a third row space costs a variant
    /// here and one more element at each fill site — never a second match arm
    /// in every file that holds a pair.
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

/// **ONE VALUE PER ROW AXIS, ADDRESSED BY THE AXIS** — the shape every
/// per-axis fact in this tree is carried in, so that "which axis" is an INDEX
/// and never a pair of hand-kept fields.
///
/// **WHY THIS EXISTS.** The second row axis arrived as a RECORD on the
/// compiler half — `FireRows` has four named counts, `AxisPlan` names its
/// axis — and as a MIRROR everywhere else: a `patch_classes` beside every
/// `classes`, a `patch_pad` beside every `pad`, a `RowAxis::Patches` arm
/// beside every `RowAxis::Tokens` one. A mirror is not a generalisation. It
/// is the same derivation written twice, free to disagree with itself, and
/// its cost is paid again in full by the third row space (the per-key
/// attention-score extent, attn-score §6.1) — which is a variant of
/// [`RowAxis`] and, with this type in the way, one more element at each fill
/// site rather than a second field in nine files.
///
/// **AN ARRAY AND NOT A MAP.** [`RowAxis::ALL`] is in discriminant order and
/// [`RowAxis::COUNT`] is its length, so the index is the variant's own
/// integer: no hashing, no `Option`, no absent entry. Every axis has a value
/// always — that is what makes a text-only fire's patch entry the ZERO
/// window rather than a missing one, which is the reading the whole campaign
/// rests on.
///
/// **AND IT KNOWS NOTHING ABOUT WHAT IT HOLDS.** `T` is a window table here,
/// a pad pair there, a carve somewhere else. The type owes those no
/// vocabulary and they owe it none; what it owns is the indexing and the
/// promise that a `PerAxis` is total over the axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PerAxis<T>([T; RowAxis::COUNT]);

impl<T> PerAxis<T> {
    /// One value per axis, IN [`RowAxis::ALL`]'s ORDER — the array fill a new
    /// row space costs one more element of.
    ///
    /// Written as an array rather than as one argument per axis on purpose:
    /// adding a variant then fails to compile HERE, at the fill, which is
    /// where a caller has the value in hand to state. A per-variant signature
    /// would fail to compile too, but only after the signature had been
    /// widened by hand in every file that calls it.
    pub const fn new(values: [T; RowAxis::COUNT]) -> PerAxis<T> {
        PerAxis(values)
    }

    /// One value per axis, computed from the axis.
    ///
    /// The door for a fill whose entries are the same derivation over
    /// different arguments — which is what a mirrored pair always was, said
    /// once.
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
    /// The same value on every axis — the honest fill for a fact that is not
    /// per-axis yet and is carried per-axis so that the day it becomes one
    /// costs a call site rather than a type.
    #[must_use]
    pub fn splat(value: T) -> PerAxis<T> {
        PerAxis::from_fn(|_| value.clone())
    }
}

impl<T> core::ops::Index<RowAxis> for PerAxis<T> {
    type Output = T;

    /// **THE WHOLE POINT.** `table[axis]` where a mirrored pair used to spell
    /// a two-arm match — in bounds by construction, because the array is
    /// [`RowAxis::COUNT`] wide and a `RowAxis` is one of exactly that many
    /// variants.
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
    /// The row space this dim sizes, or `None` for a [`Const`](Dim::Const) —
    /// a fixed block is not fire-aligned and so belongs to no axis.
    ///
    /// **`Lanes` IS THE TOKEN AXIS AND `Images` IS THE PATCH ONE.** A lane is
    /// a request of the token rectangle and an indptr is that rectangle's
    /// bounds vector; both are cut by the same window pair the token rows
    /// are. [`Images`](Dim::Images) and [`ImagesPlus`](Dim::ImagesPlus) say
    /// the same two sentences about the patch rectangle, and answer
    /// [`RowAxis::Patches`] for the same reason — which is what makes the
    /// second seriation's window pair the one that cuts them.
    ///
    /// **AND THE CUT NOW CONSUMES THIS.** `engine_cuda::run::Run::cut` picks
    /// which of a window's intervals a column is sliced at by asking this
    /// function and indexing a [`PerAxis`] with the answer, where it used to
    /// carry one arm per variant against one named field per axis. So a new
    /// row space's variants are cut correctly by the resolution the day they
    /// answer here — the ROW-versus-LANE reading beside it is what still has
    /// to be stated, because that is a fact about the DIM and not about its
    /// axis.
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
    /// **THE TRUNK'S TRIPLE-WIDE POSITION STREAM**: `[Dim::Tokens, 3]` `i32`,
    /// one `(t, h, w)` per TOKEN row (multimodal §6.3).
    ///
    /// [`Positions`](RuntimeInput::Positions) is one scalar per token row and
    /// `elementwise.rope_mrope` reads a `[rows, 3]` rectangle, so before this
    /// row existed THE MROPE OP WAS UNREACHABLE FROM ANY TEXT — it had no
    /// feeder at all. Both qwen SKUs' `text_config.rope_parameters` state
    /// `mrope_interleaved: true` with `mrope_section: [11, 11, 10]` (summing
    /// to 32 = `rotary_dim/2`, which is what identifies it as the TRUNK's
    /// rotation and not the tower's), and every lane carrying an image needs
    /// it.
    ///
    /// **A SECOND STREAM AND NOT A WIDENED FIRST ONE.** A text lane's triple
    /// is `(p, p, p)`, which is scalar rope to the last bit — so widening
    /// `Positions` would have cost every fire of every SKU three times the
    /// staging for a rectangle whose two extra columns repeat the first. It is
    /// declared by the texts that rotate this way and by no others, and a load
    /// whose plan does not name it reserves not one byte.
    ///
    /// **AND IT RIDES NO STAGING RING**, for [`Patches`](RuntimeInput::Patches)'
    /// reason (multimodal §5.4): device bytes below the staged prefix, written
    /// inside `enqueue` from a pageable vector and consumed by the rotation
    /// launched behind it on the same stream.
    MropePositions,
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
    /// The fire's patch rows, pre-unfolded: `[Dim::Patches, C·T·P²]`, one row
    /// per patch, every image of every lane concatenated in the fire's patch
    /// order.
    ///
    /// **BARE, FOR `AdapterRoutes`' REASON, AND ON THE OTHER AXIS.** There is
    /// one patch rectangle per fire the way there is one token rectangle, so
    /// there is nothing to key it by; what makes it the second row axis is its
    /// `Dim::Patches` leading dim and nothing else it carries.
    ///
    /// **PRE-UNFOLDED IS A CONTRACT DECISION, NOT AN OMISSION** (multimodal
    /// §2). The submission ships patch vectors rather than pixels, so the
    /// patch embed is the matmul+bias this IR already has and no convolution
    /// op is owed; v1's decode and resize happen host-side under the rung
    /// policy that fixes patches-per-image.
    ///
    /// **AND IT RIDES NO STAGING RING** (multimodal §5.4). A store layout may
    /// not depend on the plan, so a depth-multiplied pinned reservation for a
    /// vector a text-only load never fills would be paid by every load; patch
    /// bytes are written inside the enqueue and consumed by kernels behind
    /// them on the same stream. The same is true of the two vectors below,
    /// which are cut from the same submission at the same instant.
    Patches,
    /// The patch axis's own indptr: `[Dim::ImagesPlus(1)]` `i32`, where image
    /// `i`'s patch rows are `[segments[i], segments[i + 1])` of the fire's
    /// patch rectangle. Read by `attention.dense`.
    ///
    /// **THE BOUNDS VECTOR THE SECOND AXIS OWES ITSELF.** `GeomKind::Indptr`
    /// is the token rectangle's, keyed by a cache SPACE because a page table
    /// is a property of an extent; this one is keyed by nothing for
    /// [`AdapterRoutes`]'s reason — there is one patch rectangle per fire and
    /// therefore one way to cut it into images, and a second spelling would
    /// be the same vector interned twice and free to disagree with itself.
    ///
    /// [`AdapterRoutes`]: RuntimeInput::AdapterRoutes
    PatchSegments,
    /// Where each row of the tower's output lands in the TOKEN rectangle:
    /// `i32`, one destination token row per tower row, read by
    /// `layout.scatter_rows` as the embed merge.
    ///
    /// **THE ONE VECTOR THAT CROSSES THE TWO AXES, AND THE ONLY UNCHECKABLE
    /// ONE.** `scatter_rows` is a copy with an index and no arithmetic: an
    /// entry past the token rectangle is an out-of-bounds DEVICE WRITE that
    /// the kernel cannot see and the arena does not fault on. So the fire path
    /// validates this vector against the composition's row count before the
    /// launch — refusal (i) of multimodal M-1e — which is a check the plan
    /// cannot state and the kernel cannot make.
    PatchRoutes,
    /// **THE TOWER'S OWN POSITION STREAM**: `[Dim::Patches, 3]` `i32`, one
    /// `(t, h, w)` per PATCH row (multimodal §6.3).
    ///
    /// `Qwen3_5VisionAttention.forward` rotates, and what it rotates by is
    /// each patch's `(h, w)` in its own image's grid —
    /// `get_vision_position_ids` over `Qwen3_5VisionRotaryEmbedding`. That is
    /// the patch axis's fact and nothing on the token axis carries it: two
    /// patches of two images may share a `(h, w)` and sit in different lanes,
    /// and a lane's token positions say nothing about either.
    ///
    /// **BARE, LIKE ITS TWO NEIGHBOURS, AND CUT FROM THE SAME SUBMISSION.**
    /// One patch rectangle per fire means one position stream over it; it is
    /// staged beside [`Patches`](RuntimeInput::Patches),
    /// [`PatchSegments`](RuntimeInput::PatchSegments) and
    /// [`PatchRoutes`](RuntimeInput::PatchRoutes), in the same `enqueue`, on
    /// no ring.
    ///
    /// The third column is the time axis, and the towers this campaign serves
    /// leave it zero: an image is one frame, so `sections[0] == 0` in the
    /// tower's [`MropeForm::Blocked`](crate::MropeForm::Blocked) rotation and
    /// nothing reads the column. It is carried anyway because the rotation
    /// reads `[rows, 3]` on both axes, and a two-wide patch stream would be a
    /// second shape for one op to know about — the video case (`temporal_
    /// patch_size`, out of scope by §4) is where the column starts moving.
    PatchPositions,
    /// **WHICH ROW OF THE LEARNED POSITION TABLE EACH PATCH READS**:
    /// `[Dim::Patches]` `i32` on the native grid, `[Dim::Patches, taps]` when
    /// the table is resampled (multimodal §9.2, text-wave III).
    ///
    /// §6.4 proposed baking the position embedding into the patch-embed GEMM
    /// by widening the patch vector with a one-hot of the patch's index. That
    /// does not survive its own arithmetic — the one-hot is
    /// `num_position_embeddings` wide, so an image of 2304 patches would ship
    /// 10.6 MiB of bf16 zeros to address a 3.4 MiB table, which is §5.4's own
    /// objection word for word — and it could not express the resample at all,
    /// because an import places bytes and does not compute them.
    ///
    /// **SO THE POSITION EMBED IS A GATHER, AND THE GATHER ALREADY EXISTS.**
    /// `layout.embed` reads a table by an id vector and now types its output
    /// off THAT VECTOR'S row space, so the same op serves both axes and the
    /// exact-grid case costs one node:
    /// `residual_add(layout::embed(ids, pos_embed, vocab), y)`.
    ///
    /// `taps` is the interpolation's width — 1 on the native grid, 4 for
    /// bilinear, 16 for bicubic — and the text's own declaration states it, so
    /// the shell reserves what the plan asks for and a native-grid tower pays
    /// one i32 per patch row.
    ///
    /// Cut from the same submission as the four patch vectors beside it, and
    /// staged on no ring for their reason.
    PatchEmbedRows,
    /// **HOW MUCH OF EACH TAP** — `[Dim::Patches, taps]` `f32`, read by
    /// `layout.embed_weighted` (multimodal §9.2).
    ///
    /// The bilinear resample of the learned grid is four table rows summed
    /// under four weights; `_interpolation_axis_taps_weights` computes both
    /// per axis and the 2-D case is their outer product. The weights are the
    /// PREPROCESSOR'S arithmetic — the resize policy already owns the grid —
    /// so they arrive with the ids rather than being derived on the device.
    ///
    /// **A TEXT ON THE NATIVE GRID NEVER DECLARES THIS**, and then nothing is
    /// reserved, nothing staged, and the plan reads
    /// [`PatchEmbedRows`](RuntimeInput::PatchEmbedRows) through the plain
    /// `layout.embed`. That is the cheap path, and it is cheap because it is
    /// the absence of this row rather than a degenerate value of it.
    ///
    /// `f32` and not the activation element: these are geometry, the same way
    /// a position is, and a weight quantised to bf16 would move the resample
    /// by more than the gather it feeds.
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

    /// **THE INDEX IS THE VARIANT'S OWN INTEGER, BOTH WAYS.** Every axis
    /// reads back what was filled at it, `RowAxis::ALL` is in discriminant
    /// order, and `Dim::axis` lands each symbolic dim on the entry its row
    /// space owns — which is the whole contract every per-axis table in the
    /// tree indexes on.
    #[test]
    fn a_per_axis_reads_back_what_each_axis_was_filled_with() {
        let mut table = PerAxis::new(["tokens", "patches"]);
        assert_eq!(table[RowAxis::Tokens], "tokens");
        assert_eq!(table[RowAxis::Patches], "patches");
        assert_eq!(table.as_slice().len(), RowAxis::COUNT);

        // The array's order IS the enum's, which is what makes the index a
        // multiplication rather than a lookup.
        for (at, axis) in RowAxis::ALL.into_iter().enumerate() {
            assert_eq!(table.as_slice()[at], axis.name());
        }

        // Writing through the index is the same address.
        table[RowAxis::Patches] = "second";
        assert_eq!(table[RowAxis::Patches], "second");
        assert_eq!(table[RowAxis::Tokens], "tokens", "the other axis stood still");

        // And `from_fn` is the same fill said once.
        let named = PerAxis::from_fn(RowAxis::name);
        assert_eq!(named[RowAxis::Tokens], "tokens");
        assert_eq!(named[RowAxis::Patches], "patches");

        // Which is what a cut indexes with: every symbolic dim's own axis.
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
