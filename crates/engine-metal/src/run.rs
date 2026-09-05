//! One fire's dispatch state: resolves plan ids to device handles. Integrity failures panic rather than returning backend errors.

use kernels_metal::attn::mla::MlaPlan;
use kernels_metal::linear::moe::RoutedScratch;
use kernels_metal::{
    Bank, Ctx, DecodePlan, KvPool, PrefillPlan, RaggedTensor, RecurrentPool, Tensor,
};
use model_ir::{Def, Dim, GeomKind, Node, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};

use crate::device::Handles;
use crate::dispatch::copy::CopyPlan;
use crate::scratch::Scratch;
use crate::window::{At, Window, Windows};

/// One loader-resolved weight: most rows are a single dense handle; a quantized weight is 2-3
/// device planes under one `Def::Weight` id, with group size and bit width traveling per-bank
/// rather than model-wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRow {
    /// One dense handle, resolved by [`Run::tensor`].
    Dense(Tensor),

    /// A split-plane quantized bank, resolved by [`Run::planes`] or asked
    /// after by [`Run::banked`] — never as one tensor.
    Planes(Bank),
}

/// Loader-resolved weights, one row per `Trace::params` entry —
/// `Def::Weight(i)` resolves to row `i`.
/// `None` marks an unbound param; resolving it is a binding bug and panics.
#[derive(Clone, Debug, Default)]
pub struct WeightTable(pub Vec<Option<WeightRow>>);

/// Arena slots at the compiler's offsets, `ValueId`-indexed. A merge aliases onto its op's slot,
/// so both resolve the same row. `None` for ids with no arena slot (inputs, weights, caches,
/// structs).
#[derive(Clone, Debug, Default)]
pub struct SlotTable(pub Vec<Option<Tensor>>);

/// One resolved cache space: the storage pointer only; geometry rides in [`FireBindings`].
#[derive(Clone, Copy, Debug)]
pub enum CachePool {
    /// A paged kv space (`CacheRow::Kv`).
    Kv(KvPool),
    /// A recurrent state space (`CacheRow::State`).
    Recurrent(RecurrentPool),
}

/// Cache-index-indexed pools, aligned with `Trace::caches`.
#[derive(Clone, Debug, Default)]
pub struct CacheTable(pub Vec<CachePool>);

/// The geometry vectors one cache space declared. Only what the plan names gets bound, so every
/// seat is optional; resolving an unbound seat is a binding bug and panics.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheGeometry {
    /// `RuntimeInput::Geometry { kind: Indptr }`.
    pub indptr: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: Indices }`.
    pub indices: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: SeqLens }`.
    pub seq_lens: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: LastPageLen }`.
    pub last_page_len: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: KvLen }`.
    pub kv_len: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: RowValid }`.
    pub row_valid: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: RequestOfToken }`.
    pub request_of_token: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: WritePage }`.
    pub write_page: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: WriteOffset }`.
    pub write_offset: Option<Tensor>,
}

/// Per-fire tables the sdpa shaders read alongside the pool; positions live in
/// [`FireBindings::positions`].
#[derive(Clone, Copy, Debug)]
pub struct FireTables {
    /// `i32`, one per token: the owning request.
    pub request_of_token: Tensor,

    /// `u8` packed mask planes, one row per request — shared with `RuntimeInput::Mask`'s
    /// resolution.
    pub mask: Tensor,

    /// `u8`, one per request: whether its mask row is live.
    pub mask_enabled: Tensor,

    /// Elements from one request's mask row to the next.
    pub mask_stride: u32,
}

/// The dsv4 compressor state `attention.pool_gather` reads beside its cache. Reserved only when
/// the trace has a pooled layer, so a fire with none pays no bytes.
#[derive(Clone, Copy, Debug)]
pub struct PoolSlabs {
    /// The rolling kv window: `[the source pool's cells, coff * head_dim]`,
    /// in the pooled entry's element.
    pub state_kv: Tensor,

    /// The rolling gate logits, at `state_kv`'s shape and element.
    pub state_score: Tensor,
}

/// What the engine binds each fire, owned by the [`Run`] for its lifetime.
///
/// `tokens`, `positions`, and `geometry` are op-visible (`RuntimeInput` routes onto them in
/// [`Run::tensor`]); `tables` are ambient — read by the plan builders without an op naming them.
#[derive(Clone, Debug)]
pub struct FireBindings {
    /// `RuntimeInput::Tokens`: ragged `i32`, one id per token.
    pub tokens: Tensor,

    /// `RuntimeInput::Positions`: ragged `i32`, one absolute position per
    /// token — also the plan builders' causal-bound table.
    pub positions: Tensor,

    /// `RuntimeInput::AdapterRoutes`: `i32`, one adapter id per token row, `-1` for a row whose
    /// lane routes nowhere. `None` for a fire no lane carried an adapter into — costs zero bytes
    /// and zero launches.
    pub adapter_routes: Option<Tensor>,

    /// `RuntimeInput::Patches`: `[patch rows, C·T·P²]` in the plan's element.
    pub patches: Option<Tensor>,

    /// `RuntimeInput::PatchSegments`: `i32`, `[images + 1]` — the patch
    /// axis's own indptr, which `attention.dense` reads its image boundaries
    /// out of.
    pub patch_segments: Option<Tensor>,

    /// `RuntimeInput::PatchRoutes`: `i32`, `[patch rows]`, one destination token row per tower
    /// row, `-1` for a row the fold spends. Host-checked: an out-of-range entry is an OOB device
    /// write the arena doesn't catch.
    pub patch_routes: Option<Tensor>,

    /// `RuntimeInput::PatchPositions`: `i32`, `[patch rows, 3]`.
    pub patch_positions: Option<Tensor>,

    /// `RuntimeInput::PatchEmbedRows`: `i32`, `[patch rows, taps]`, `None`
    /// for a plan that reads the learned position table on its native grid.
    pub patch_embed_rows: Option<Tensor>,

    /// `RuntimeInput::PatchEmbedWeights`: `f32`, `[patch rows, taps]`.
    pub patch_embed_weights: Option<Tensor>,

    /// `RuntimeInput::MropePositions`: `i32`, `[rows, 3]` — staged for every fire of a plan that
    /// declares the rotation, image or no image; a lane with no stream of its own reads the
    /// scalar `(p, p, p)`.
    pub mrope_positions: Option<Tensor>,

    /// Per cache space, aligned with `Trace::caches`:
    /// `RuntimeInput::Geometry { space, kind }` routes to that space.
    pub geometry: Vec<CacheGeometry>,

    /// The fire tables the attention plan builders consume.
    pub tables: FireTables,

    /// Score-capture output slab. `None` unless the plan
    /// declares an `attn.scores` export and a lane in this fire asks for it.
    pub scores: Option<crate::scores::ScoreSeat>,

    /// **The recurrent-state seat** (`crate::rs`): `Some` only for a fire in which a lane
    /// buffers or replays recurrent state, and then the SSM and hasher ops take the committed
    /// arm (`crate::dispatch::rs`). `None` is every fire that folds in the forward, whose walk
    /// is byte for byte what it was before the seat existed.
    pub rs: Option<std::sync::Arc<crate::rs::Seat>>,
}

/// One built plan payload — closed enum over the three kinds this plane builds; a wrong kind is
/// a named panic.
#[derive(Clone, Copy, Debug)]
pub enum StructSlot {
    /// `StructKind::AttnDecodePlan`.
    Decode(DecodePlan),

    /// `StructKind::AttnPrefillPlan`.
    Prefill(PrefillPlan),

    /// `StructKind::MlaPlan` — empty: this engine reads the fire's position/owning-request
    /// tables and the pool page walk directly at each attention op, so no payload is needed.
    Mla(MlaPlan),
}

/// One fire's dispatch state: encode sink, resolution tables, fire bindings and plan payloads.
/// Constructed once per fire; the walk runs prepare phase first so every plan payload exists
/// before its consumers encode.
pub struct Run<'c> {
    /// The encode sink; everything device-facing goes through it — nothing here names Metal
    /// directly.
    ctx: &'c Ctx<'c>,

    /// Handle table every carve is minted into and every argument resolves through. A windowed
    /// cut is a new row here (Metal binds a buffer and an offset, so there is no address to add
    /// a stride to).
    handles: &'c Handles,

    /// The routing: `Trace::values`, read by [`Run::tensor`] to send each id to its table.
    values: &'c [ValueDecl],

    /// `Trace::nodes` — read only by `crate::dispatch::copy`, which walks a copied region's node
    /// range.
    nodes: &'c [Node],

    /// `Def::Weight` rows, loader-resolved.
    weights: &'c WeightTable,

    /// `Def::Op` / `Def::Merge` rows, carved at the compiler's offsets.
    arena: &'c SlotTable,

    /// `Def::Cache` rows — pool pointers, resolved through [`Run::pool`] and
    /// [`Run::recurrent`], never through [`Run::tensor`].
    caches: &'c CacheTable,

    /// Plan payloads, filled in the prepare phase and read in the capture phase. Keyed by
    /// `(run, value)` — flat at `run * values_wide + value` — since a region split into multiple
    /// window-runs needs one slot per run, not per value.
    structs: Vec<Option<StructSlot>>,
    /// How many values one run's slice of [`structs`](Run::structs) holds.
    values_wide: usize,

    /// This fire's bindings.
    fire: FireBindings,

    /// Every region's windows, resolved once per fire from the composition's
    /// class table. Indexed by [`place`](Run::place).
    windows: &'c Windows,

    /// Which region/run the walk is currently on, written by [`Cursor`](crate::window::Cursor)
    /// before each region's dispatch and before each encode within it.
    place: &'c At,

    /// The copied region the walk is inside, or the default when none is active. Carries the
    /// region index it was built for, so a stale plan (built for a different region) panics
    /// rather than reading the wrong offsets.
    copy: CopyPlan,

    /// Load-time scratch reservation: working rectangles no op names, plus arena slot capacities
    /// and router expert counts. Borrowed, not owned — built once at load; each fire mints its
    /// own handle row into it, like every other arena row.
    scratch: &'c Scratch,
}

impl<'c> Run<'c> {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &'c Ctx<'c>,
        handles: &'c Handles,
        values: &'c [ValueDecl],
        nodes: &'c [Node],
        weights: &'c WeightTable,
        arena: &'c SlotTable,
        caches: &'c CacheTable,
        fire: FireBindings,
        windows: &'c Windows,
        place: &'c At,
        scratch: &'c Scratch,
    ) -> Self {
        Self {
            ctx,
            handles,
            values,
            nodes,
            weights,
            arena,
            caches,
            structs: vec![None; values.len() * windows.max_runs() as usize],
            values_wide: values.len(),
            fire,
            windows,
            place,
            copy: CopyPlan::default(),
            scratch,
        }
    }

    /// The window of the region the walk is inside, cut at the run it is on.
    pub(crate) fn window(&self) -> &'c Window {
        self.windows.at(self.place.region.get(), self.place.run.get())
    }

    /// Where this run's payload for `id` sits in [`structs`](Run::structs); same run/window
    /// pairing as [`window`](Run::window).
    fn struct_at(&self, id: ValueId) -> usize {
        self.place.run.get() as usize * self.values_wide + id.0 as usize
    }

    /// This window's own qo boundaries, rebased to start at zero.
    pub(crate) fn qo_indptr(&self) -> Tensor {
        self.window().indptr
    }

    /// Host-side copy of the qo boundaries, for an arm that needs to read rather than bind them.
    #[allow(dead_code)]
    pub(crate) fn qo_indptr_host(&self) -> &'c [i32] {
        &self.window().indptr_host
    }

    /// How many token rows this window covers.
    #[allow(dead_code)]
    pub(crate) fn total_tokens(&self) -> u32 {
        self.window().span.rows
    }

    /// Whether any lane of this window carries more than one row — the
    /// question a chunked linear-attention entry asks to choose its arm.
    #[allow(dead_code)]
    pub(crate) fn multi_token(&self) -> bool {
        self.qo_indptr_host()
            .windows(2)
            .any(|pair| pair[1] - pair[0] > 1)
    }

    /// A raw ambient table (not a `ValueId`), cut to this window's rows. For a
    /// gathered window, `positions`/`request_of_token` are permuted rather
    /// than sliced (a gather's rows aren't contiguous); the mask is never permuted.
    pub(crate) fn cut_rows(&self, handle: Tensor) -> Tensor {
        if let Some(gathered) = &self.window().gathered {
            if handle.buf == self.fire.positions.buf {
                return gathered.positions;
            }
            if handle.buf == self.fire.tables.request_of_token.buf {
                return gathered.request_of_token;
            }
        }
        let span = self.window().span;
        self.slice(handle, span.row_offset, span.rows)
    }

    /// Which region of the template the walk is inside — the cursor's own
    /// index, read by `crate::dispatch::copy` so that a copy plan built for
    /// one region cannot be read inside another.
    pub(crate) fn at_region(&self) -> u32 {
        self.place.region.get()
    }

    /// `Trace::nodes`, for the one caller that walks a region's node range.
    pub(crate) fn nodes(&self) -> &'c [Node] {
        self.nodes
    }

    /// `Trace::values`, for the same caller: what a node's operand ids are
    /// declared as.
    pub(crate) fn values(&self) -> &'c [ValueDecl] {
        self.values
    }

    /// One value's fire-wide rectangle, uncut — what a copy plan compacts from, and what a
    /// scatter puts back.
    pub(crate) fn uncut(&self, id: ValueId) -> Tensor {
        self.whole(id)
    }

    /// Where a handle actually points: `(reservation, offset)` — the copy
    /// plan's key, since two values aliased onto one arena slot get
    /// different handle rows at the same offset. `None` for an unminted row.
    pub(crate) fn address(&self, handle: u32) -> Option<(u64, u64)> {
        let row = self.handles.get(handle)?;
        Some((crate::device::alloc::slab_id(row.slab()), row.offset()))
    }

    /// Seat the plan a copied region's gather just built — read back by
    /// [`Run::compacted`] for every operand until the region's scatter.
    pub(crate) fn set_copy(&mut self, plan: CopyPlan) {
        self.copy = plan;
    }

    /// The plan the current region's gather seated, for the scatter that
    /// closes the bracket.
    pub(crate) fn staged_copy(&self) -> &CopyPlan {
        &self.copy
    }

    /// One rectangle of the copy role, minted for this fire. `None` means the load reserved too
    /// little for it; a panic means the handle table itself is full.
    pub(crate) fn copy_room(&self, offset: u64, bytes: u64) -> Option<u32> {
        Some(
            self.scratch
                .copy(self.handles, offset, bytes)?
                .unwrap_or_else(|fault| {
                    panic!("the copy rectangle this load reserved does not mint: {fault}")
                }),
        )
    }

    /// The encode sink, for the arms.
    /// The fire's handle table, for an arm that mints its own cuts.
    pub(crate) fn handles(&self) -> &'c Handles {
        self.handles
    }

    /// This fire's recurrent seat (`crate::rs`), or `None` for a fire every
    /// lane of which folds in the forward — the ordinary path, untouched.
    pub(crate) fn rs_seat(&self) -> Option<std::sync::Arc<crate::rs::Seat>> {
        self.fire.rs.clone()
    }

    pub(crate) fn ctx(&self) -> &'c Ctx<'c> {
        self.ctx
    }

    /// The fire bindings, for the plan-building arms' seam.
    pub(crate) fn bindings(&self) -> &FireBindings {
        &self.fire
    }

    /// One rectangle, sliced to `keep` rows starting at `skip`; minted into
    /// [`Handles`], which bounds-checks it. A failing cut is an integrity
    /// failure (compiler carve vs. window table disagreement), and panics.
    fn slice(&self, handle: Tensor, skip: u32, keep: u32) -> Tensor {
        if skip == 0 && keep >= handle.rows {
            return handle;
        }
        let stride = u64::from(handle.width)
            * model_compiler::arena::elem_bytes(handle.dtype).unwrap_or_else(|| {
                panic!(
                    "a {:?} rectangle has no element size and so no row to step by",
                    handle.dtype
                )
            });
        let rows = keep.min(handle.rows.saturating_sub(skip));
        let cut = self
            .handles
            .cut(handle.buf, u64::from(skip) * stride, u64::from(rows) * stride)
            .unwrap_or_else(|fault| {
                panic!(
                    "the window's cut of handle {} at row {skip} for {rows} rows does \
                     not land: {fault}",
                    handle.buf
                )
            });
        Tensor::new(cut, rows, handle.width, handle.dtype)
    }

    /// One value's rectangle, cut to the window of the node asking for it.
    /// Row-shaped values are indexed by absolute fire row; a `Dim::Const`
    /// column is handed over whole. `GeomKind::Indices` and
    /// `RuntimeInput::Mask` are never fire-row-indexed and are never sliced here.
    fn cut(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
        // A gathered window's rows were compacted into scratch by the gather; resolve
        // through `compacted` instead of slicing the fire-wide column.
        if self.window().gathered.is_some() {
            return self.compacted(id, handle);
        }
        if matches!(
            self.values[at].def,
            Def::Input(RuntimeInput::Mask { .. })
                | Def::Input(RuntimeInput::Geometry {
                    kind: GeomKind::Indices,
                    ..
                })
        ) {
            return handle;
        }
        let Ty::Tensor { shape, .. } = &self.values[at].ty else {
            return handle;
        };
        let seated = self.window();
        let window = seated.span;
        // The patch axis has its own window (`Window::patch`), cut separately from the
        // token axis's `span` — needed because the embed merge is a token region that also
        // reads a patch column.
        let patch = seated.patch;
        let (skip, keep) = match shape.first() {
            Some(Dim::Tokens) => (window.row_offset, window.rows),
            Some(Dim::TokensTimes(k)) => (window.row_offset * k, window.rows * k),
            Some(Dim::Lanes) => (window.lane_offset, window.lanes),
            Some(Dim::LanesPlus(k)) => (window.lane_offset, window.lanes + k),
            Some(Dim::Const(_)) | None => return handle,
            Some(Dim::Patches) => (patch.row_offset, patch.rows),
            Some(Dim::Images) => (patch.lane_offset, patch.lanes),
            Some(Dim::ImagesPlus(k)) => (patch.lane_offset, patch.lanes + k),
        };
        self.slice(handle, skip, keep)
    }

    /// [`Run::cut`]'s counterpart for a gathered window: resolves to the
    /// gather's staging rectangle, a re-cut kv-geometry twin, or the value
    /// unchanged. The kv pool itself is never re-cut here — sdpa indexes it
    /// via the permuted `request_of_token`.
    fn compacted(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
        let gathered = self
            .window()
            .gathered
            .as_ref()
            .expect("`compacted` is reached only through a gathered window");
        if let Def::Input(RuntimeInput::Geometry { space, kind }) = &self.values[at].def {
            let Some(space) = gathered.spaces.get(*space as usize) else {
                return handle;
            };
            return match kind {
                GeomKind::Indptr => space.page_indptr,
                GeomKind::Indices => space.page_indices,
                GeomKind::LastPageLen => space.last_page_lens,
                GeomKind::KvLen => space.kv_len,
                // Unreachable in practice; falls back to the fire-wide vector rather than
                // guessing a window.
                _ => handle,
            };
        }
        let Ty::Tensor { shape, .. } = &self.values[at].ty else {
            return handle;
        };
        match shape.first() {
            Some(Dim::Tokens) => {
                assert_eq!(
                    self.copy.region,
                    self.place.region.get(),
                    "value {at} is being resolved inside a copied region whose gather \
                     has not run; `model_exec::fire::walk` brackets a copied region's \
                     nodes and this is what says the bracket was lost",
                );
                let Some(key) = self.address(handle.buf) else {
                    panic!(
                        "value {at} is row-shaped and its handle {} resolves to no \
                         binding; every operand of a copied region was minted by this \
                         same fire",
                        handle.buf
                    )
                };
                self.copy.tight(key).unwrap_or_else(|| {
                    panic!(
                        "value {at} is row-shaped and its column was not compacted; the \
                         copy plan is built from the same region's operands the walk is \
                         dispatching, so a miss is a plan and a template built apart"
                    )
                })
            }
            _ => handle,
        }
    }

    /// One plan id in, one device handle out. Cache ids and split-plane
    /// weights don't resolve to a tensor here; they panic (use
    /// [`Run::pool`]/[`Run::recurrent`]/[`Run::planes`] instead).
    pub(crate) fn tensor(&self, id: ValueId) -> Tensor {
        self.cut(id, self.whole(id))
    }

    /// The same resolution, uncut — the fire-wide rectangle a value names.
    fn whole(&self, id: ValueId) -> Tensor {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Input(RuntimeInput::Tokens) => self.fire.tokens,
            Def::Input(RuntimeInput::Positions) => self.fire.positions,
            // One mask per fire; the op-named mask resolves onto the same seat the plan
            // builders use.
            Def::Input(RuntimeInput::Mask { space: _ }) => self.fire.tables.mask,
            // Bound only if a lane carried an adapter; otherwise the correction's window is
            // empty and this arm is never reached, so the panic below is unreachable rather
            // than a real gap.
            Def::Input(RuntimeInput::AdapterRoutes) => {
                self.fire.adapter_routes.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's adapter ids, which no lane of it \
                         carried"
                    )
                })
            }
            // Bound only when a lane carried an image; named per-input so a panic message
            // says which one is missing.
            Def::Input(RuntimeInput::Patches) => self.fire.patches.unwrap_or_else(|| {
                panic!("value {at} reads this fire's patch rows, which no lane of it submitted")
            }),
            Def::Input(RuntimeInput::PatchSegments) => {
                self.fire.patch_segments.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's image boundaries, which no lane of \
                         it submitted"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchRoutes) => self.fire.patch_routes.unwrap_or_else(|| {
                panic!(
                    "value {at} reads where this fire's tower rows land, which no lane \
                     of it submitted"
                )
            }),
            Def::Input(RuntimeInput::PatchPositions) => {
                self.fire.patch_positions.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's patch grid positions, which no lane \
                         of it submitted"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchEmbedRows) => {
                self.fire.patch_embed_rows.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's position-table taps, and this load \
                         stages none — the plan reads the table on its native grid"
                    )
                })
            }
            Def::Input(RuntimeInput::SelfCondRows | RuntimeInput::SelfCondWeights) => {
                panic!(
                    "value {at} reads a self-conditioning input, which this shell stages none \
                     of; the load refuses such a plan"
                )
            }
            Def::Input(RuntimeInput::PatchEmbedWeights) => {
                self.fire.patch_embed_weights.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's interpolation weights, and this load \
                         stages none — the plan reads the table on its native grid"
                    )
                })
            }
            // Staged for every fire of a plan that declares rotation, image or not; a
            // text-only lane reads the scalar (p, p, p).
            Def::Input(RuntimeInput::MropePositions) => {
                self.fire.mrope_positions.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads the fire's (t, h, w) token positions, and this \
                         load reserved no triple — the plan declares no multimodal rotation"
                    )
                })
            }
            Def::Input(RuntimeInput::Geometry { space, kind }) => {
                let space = *space as usize;
                let seat = self.fire.geometry.get(space).unwrap_or_else(|| {
                    panic!(
                        "value {at} names cache space {space}, and this fire binds \
                         {} geometry spaces",
                        self.fire.geometry.len()
                    )
                });
                let bound = match kind {
                    GeomKind::Indptr => seat.indptr,
                    GeomKind::Indices => seat.indices,
                    GeomKind::SeqLens => seat.seq_lens,
                    GeomKind::LastPageLen => seat.last_page_len,
                    GeomKind::KvLen => seat.kv_len,
                    GeomKind::RowValid => seat.row_valid,
                    GeomKind::RequestOfToken => seat.request_of_token,
                    GeomKind::WritePage => seat.write_page,
                    GeomKind::WriteOffset => seat.write_offset,
                };
                bound.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads {kind:?} of cache space {space}, which this \
                         fire left unbound"
                    )
                })
            }
            Def::Weight(w) => {
                let row = *w as usize;
                match self.weights.0.get(row).copied().flatten() {
                    Some(WeightRow::Dense(handle)) => handle,
                    Some(WeightRow::Planes(_)) => panic!(
                        "value {at} is weight {row}, a split-plane bank; it resolves \
                         through `Run::planes`, never as one dense handle"
                    ),
                    None => panic!("value {at} is weight {row}, which the shell has not bound"),
                }
            }
            // A merge aliases onto its op's arena slot, so it resolves the same way as an
            // op output.
            Def::Op(_) | Def::Merge(_) => self
                .arena
                .0
                .get(at)
                .copied()
                .flatten()
                .unwrap_or_else(|| {
                    panic!("value {at} has no arena slot, which the compiler should have cut")
                }),
            Def::Cache(_) => panic!(
                "value {at} is a cache space; it resolves to a pool through `Run::pool`, \
                 never to a tensor"
            ),
        }
    }

    /// A fire-aligned value viewed through this window's boundaries; the indptr is rebased to
    /// start at zero, since a fire-wide indptr would run past a windowed rectangle's end.
    pub(crate) fn ragged(&self, id: ValueId) -> RaggedTensor {
        RaggedTensor {
            data: self.tensor(id),
            indptr: self.qo_indptr(),
        }
    }

    /// The planes of a split-plane bank. For ops whose IR variant unconditionally names a bank;
    /// an op serving both forms uses [`Run::banked`] instead.
    pub(crate) fn planes(&self, id: ValueId) -> Bank {
        self.banked(id).unwrap_or_else(|| {
            panic!(
                "value {} is bound as one dense handle, and this op reads a split-plane \
                 bank",
                id.0
            )
        })
    }

    /// The bank behind a weight id, or `None` when the row is one dense handle. An unbound row
    /// is still a binding bug and still panics.
    pub(crate) fn banked(&self, id: ValueId) -> Option<Bank> {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            panic!("value {at} is not a weight, and split-plane banks live in the weight table");
        };
        let row = *w as usize;
        match self.weights.0.get(row).copied().flatten() {
            Some(WeightRow::Planes(bank)) => Some(bank),
            Some(WeightRow::Dense(_)) => None,
            None => panic!("value {at} is weight {row}, which the shell has not bound"),
        }
    }

    /// How many experts the router that wrote `routes` declared; resolved once at load into a
    /// table indexed by `routes`. `0` if no router in this artifact wrote it.
    pub(crate) fn experts(&self, routes: ValueId) -> u32 {
        self.scratch.experts(routes)
    }

    /// The sorted MoE arm's working rectangles, minted into this fire. `None` if the load
    /// reserved none (no mixture, or one with no stated expert count); the matvec arm is used
    /// instead.
    pub(crate) fn routed_scratch(&self) -> Option<RoutedScratch> {
        Some(
            self.scratch
                .routed(self.handles)?
                .unwrap_or_else(|fault| {
                    panic!("the routed scratch this load reserved does not mint: {fault}")
                }),
        )
    }

    /// The NSA indexer's score slab, minted into this fire. `None` if the trace has no
    /// `attention.index_topk`.
    pub(crate) fn index_scores(&self) -> Option<Tensor> {
        Some(
            self.scratch
                .index_scores(self.handles)?
                .unwrap_or_else(|fault| {
                    panic!("the index score slab this load reserved does not mint: {fault}")
                }),
        )
    }

    /// The pooled-compressor rolling state for cache space `pages`, minted into this fire.
    /// `None` if the trace has no `attention.pool_gather` over that space.
    pub(crate) fn pool_state(&self, pages: ValueId) -> Option<PoolSlabs> {
        let at = pages.0 as usize;
        let Some(Def::Cache(space)) = self.values.get(at).map(|v| &v.def) else {
            panic!("value {at} is not a cache space; the pooled state is keyed by one")
        };
        Some(
            self.scratch
                .pool_state(self.handles, *space)?
                .unwrap_or_else(|fault| {
                    panic!("the compressor state this load reserved does not mint: {fault}")
                }),
        )
    }

    /// qwen4's PLE hash constants, minted into this fire. `None` if the trace has no
    /// `attention.ple_ngram_ids`.
    pub(crate) fn ple_hash(
        &self,
        mults: &[u64],
        primes: &[u64],
        offsets: &[u64],
    ) -> Option<Tensor> {
        Some(
            self.scratch
                .ple_hash(self.handles, mults, primes, offsets)?
                .unwrap_or_else(|fault| {
                    panic!("the PLE hash plane this load wrote does not mint: {fault}")
                }),
        )
    }

    /// Rows the arena slot behind `id` can hold at the budget's ceiling (not this fire's
    /// extent) — used by the dense quantized linear arms as launch capacity, so padding never
    /// overruns into the next value's slot.
    pub(crate) fn capacity(&self, id: ValueId) -> u32 {
        self.scratch.capacity(id)
    }

    /// The FP16 staging plane at `rows x contraction`, minted for the quantized linear pre-cast
    /// path. `None` if the load-time reservation doesn't hold that shape.
    ///
    /// On a mixture, this aliases the routed plane's bytes rather than costing extra.
    pub(crate) fn precast(&self, rows: u32, contraction: u32) -> Option<Tensor> {
        Some(
            self.scratch
                .precast(self.handles, rows, contraction)?
                .unwrap_or_else(|fault| {
                    panic!("the precast plane this load reserved does not mint: {fault}")
                }),
        )
    }

    /// The `StructKind` a plan op's output value declares, checked by the plan-building arms
    /// against the trace.
    pub(crate) fn declared(&self, id: ValueId) -> StructKind {
        match &self.values[id.0 as usize].ty {
            Ty::Struct(kind) => *kind,
            Ty::Tensor { .. } => panic!(
                "value {} declares a tensor, and a plan op defines a struct",
                id.0
            ),
        }
    }

    /// The paged kv pool a cache id names. Never sliced to the window: sdpa
    /// shaders index it via `request_of_token`'s absolute lane ids, so the
    /// fire-wide table is always correct.
    pub(crate) fn pool(&self, id: ValueId) -> &KvPool {
        match self.cache(id) {
            CachePool::Kv(pool) => pool,
            CachePool::Recurrent(_) => panic!(
                "value {} is a recurrent state space, and this op walks a paged kv pool",
                id.0
            ),
        }
    }

    /// The recurrent state pool a cache id names, with its slot map cut to the asking node's
    /// window — banks are addressed by slot, read as `slots[r]` from the launch's own zero.
    pub(crate) fn recurrent(&self, id: ValueId) -> RecurrentPool {
        match self.cache(id) {
            CachePool::Recurrent(pool) => RecurrentPool {
                slots: self.cut_rows(pool.slots),
                ..*pool
            },
            CachePool::Kv(_) => panic!(
                "value {} is a paged kv space, and this op scans a recurrent state pool",
                id.0
            ),
        }
    }

    fn cache(&self, id: ValueId) -> &CachePool {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Cache(c) => {
                let row = *c as usize;
                self.caches.0.get(row).unwrap_or_else(|| {
                    panic!(
                        "value {at} is cache space {row}, and the shell binds {} pools",
                        self.caches.0.len()
                    )
                })
            }
            _ => panic!("value {at} is not a cache space; tensors resolve through `Run::tensor`"),
        }
    }

    /// Store a plan payload a prepare-phase arm just built.
    pub(crate) fn put(&mut self, id: ValueId, built: StructSlot) {
        let at = self.struct_at(id);
        self.structs[at] = Some(built);
    }

    /// The decode plan a consuming arm names.
    pub(crate) fn decode_plan(&self, id: ValueId) -> &DecodePlan {
        match &self.structs[self.struct_at(id)] {
            Some(StructSlot::Decode(plan)) => plan,
            Some(_) => panic!(
                "value {} holds another plan kind, and this op consumes a decode plan",
                id.0
            ),
            None => panic!(
                "value {} holds no plan payload; its plan op has not fired, and the \
                 prepare phase runs first",
                id.0
            ),
        }
    }

    /// The prefill plan a consuming arm names.
    pub(crate) fn prefill_plan(&self, id: ValueId) -> &PrefillPlan {
        match &self.structs[self.struct_at(id)] {
            Some(StructSlot::Prefill(plan)) => plan,
            Some(_) => panic!(
                "value {} holds another plan kind, and this op consumes a prefill plan",
                id.0
            ),
            None => panic!(
                "value {} holds no plan payload; its plan op has not fired, and the \
                 prepare phase runs first",
                id.0
            ),
        }
    }
}
