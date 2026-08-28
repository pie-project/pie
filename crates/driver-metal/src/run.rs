//! The §8 `Run`, in metal's encode world: one fire's dispatch state, and the
//! one function that turns a plan id into a device handle.
//!
//! Long-lived state — the weight and arena tables, the cache pools — arrives
//! pre-built and borrowed: building those tables is the shell's binding
//! business (the successor of the old `bind/`), not this layer's. Fire-lived
//! state — the input bindings, the plan payloads — is owned: a `Run` is
//! constructed per fire and dropped with it.
//!
//! Everything here answers to one rule: a [`KernelError`] is about the
//! backend, never about the plan (`kernels::error`). A hole in a table,
//! a cache id in a tensor seat, a plan consumed before its plan op — those
//! are integrity failures of the shell or the compiler, and they panic with
//! a sentence instead of dressing up as a backend refusal.
//!
//! [`KernelError`]: kernels::KernelError

use kernels_metal::attn::mla::MlaPlan;
use kernels_metal::{Ctx, DecodePlan, KvPool, PrefillPlan, RaggedTensor, RecurrentPool, Tensor};
use model_ir::{Def, Dim, GeomKind, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};

use crate::device::Handles;
use crate::window::{At, Window, Windows};

/// One loader-resolved weight. Most rows are one dense handle; an mxfp4 bank
/// is two device planes under one `Def::Weight` id — the form this shell's
/// one-handle rows once refused. This plane stamps the mxfp4 routed matmul
/// (`linear.moe_matmul_select_bias`), so the table names the form instead of
/// refusing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRow {
    /// One dense handle, resolved by [`Run::tensor`].
    Dense(Tensor),

    /// A split-plane quantized bank — e2m1 `codes` beside e8m0 `scales` —
    /// resolved by [`Run::planes`], never as one tensor.
    Planes { codes: Tensor, scales: Tensor },
}

/// Loader-resolved weights, one row per `Plan::params` entry —
/// `Def::Weight(i)` resolves to row `i`. `None` marks a param the shell has
/// not bound; resolving such a row is a binding bug and panics.
#[derive(Clone, Debug, Default)]
pub struct WeightTable(pub Vec<Option<WeightRow>>);

/// Arena slots at the compiler's offsets, `ValueId`-indexed. Op outputs and
/// merges alike land here: the compiler aliased every merge arm onto one
/// slot and wrote that slot at the merged id's row too, so a φ resolves like
/// any op output. Rows for ids that own no arena slot (inputs, weights,
/// caches, structs) stay `None`.
#[derive(Clone, Debug, Default)]
pub struct SlotTable(pub Vec<Option<Tensor>>);

/// One resolved cache space — the storage pointer and nothing else (design
/// §7); its geometry rides in [`FireBindings`] as declared inputs.
#[derive(Clone, Copy, Debug)]
pub enum CachePool {
    /// A paged kv space (`CacheRow::Kv`).
    Kv(KvPool),
    /// A recurrent state space (`CacheRow::State`).
    Recurrent(RecurrentPool),
}

/// Cache-index-indexed pools, aligned with `Plan::caches`.
#[derive(Clone, Debug, Default)]
pub struct CacheTable(pub Vec<CachePool>);

/// The geometry vectors one cache space declared. Only what the plan names
/// gets bound, so every seat is optional; resolving an unbound seat is a
/// binding bug and panics.
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

/// The per-fire tables the sdpa shaders read beside the pool. Positions ride
/// in [`FireBindings::positions`]; these are the rest. `attention.plan_*`
/// names kv geometry instead of these, so the plan-building arms bind them
/// from here; `mask` alone is op-named too now (`attention.masked`), and
/// [`Run::tensor`] resolves `RuntimeInput::Mask` onto the same seat. This is
/// the driver side of the seam the `MENLO-SEAM` markers in
/// `kernels_metal::attn` describe.
#[derive(Clone, Copy, Debug)]
pub struct FireTables {
    /// `i32`, one per token: the owning request.
    pub request_of_token: Tensor,

    /// `u8` packed mask planes, one row per request — the plan builders'
    /// table and `RuntimeInput::Mask`'s resolution, one seat wearing both
    /// names.
    pub mask: Tensor,

    /// `u8`, one per request: whether its mask row is live.
    pub mask_enabled: Tensor,

    /// Elements from one request's mask row to the next.
    pub mask_stride: u32,
}

/// What the driver binds each fire, owned by the [`Run`] for its lifetime.
///
/// `tokens`, `positions`, and `geometry` are the op-visible inputs —
/// `RuntimeInput` routes onto them in [`Run::tensor`]. The `tables` are
/// ambient: the plan builders read them and no op names them.
///
/// **TWO FIELDS THE CUDA SIBLING'S TWIN CARRIED ARE GONE, AND FOR THE
/// REASONS THAT SIBLING RECORDED.** `facts` — the fire-wide fact word — was
/// deleted from the CUDA shell in palo build log 8 with zero readers, and
/// deleted here for the same reason: which classes run a node is
/// `Region::mask` resolved per region against the window table, which is
/// exactly what a fire-wide word cannot say (design §0's collapse). And the
/// shared `indptr` is gone because a ragged view is assembled from the
/// WINDOW's own rebased boundaries ([`Run::qo_indptr`]) — a fire-wide vector
/// handed to a windowed launch sends every entry past the end of the
/// rectangle it was given, by exactly the rows the classes before it hold.
#[derive(Clone, Debug)]
pub struct FireBindings {
    /// `RuntimeInput::Tokens`: ragged `i32`, one id per token.
    pub tokens: Tensor,

    /// `RuntimeInput::Positions`: ragged `i32`, one absolute position per
    /// token — also the plan builders' causal-bound table.
    pub positions: Tensor,

    /// Per cache space, aligned with `Plan::caches`:
    /// `RuntimeInput::Geometry { space, kind }` routes to that space.
    pub geometry: Vec<CacheGeometry>,

    /// The fire tables the attention plan builders consume.
    pub tables: FireTables,
}

/// One built plan payload. An enum over the three kinds this plane can be
/// asked to build, not `Box<dyn Any>`: the IR's `StructKind` is closed, and
/// this crate names every payload type at compile time — erasure would buy
/// no generality, only a silent-downcast failure mode. Here a wrong kind is
/// a named panic.
#[derive(Clone, Copy, Debug)]
pub enum StructSlot {
    /// `StructKind::AttnDecodePlan`.
    Decode(DecodePlan),

    /// `StructKind::AttnPrefillPlan`.
    Prefill(PrefillPlan),

    /// `StructKind::MlaPlan` — declared for shape; the metal builder refuses
    /// before one exists, so this variant is never stored.
    Mla(MlaPlan),
}

/// One fire's dispatch state: the encode sink, the resolution tables, the
/// fire bindings, and the plan payloads this fire builds. The shell
/// constructs one per fire and drives the substrate's walk
/// (`driver::fire::walk`) over it — prepare phase first, so every plan
/// payload exists before its consumers encode.
pub struct Run<'c> {
    /// The encode sink — the real shell behind `dyn Encode`. Everything this
    /// crate does to the device goes through it; nothing here names Metal.
    ctx: &'c Ctx<'c>,

    /// The handle table every carve is minted into and every argument is
    /// resolved through. A windowed cut IS a new row here — Metal binds a
    /// buffer and an offset, so there is no address to add a row stride to.
    handles: &'c Handles,

    /// The routing: `Plan::values`, read by [`Run::tensor`] to send each id
    /// to its table.
    values: &'c [ValueDecl],

    /// `Def::Weight` rows, loader-resolved.
    weights: &'c WeightTable,

    /// `Def::Op` / `Def::Merge` rows, carved at the compiler's offsets.
    arena: &'c SlotTable,

    /// `Def::Cache` rows — pool pointers, resolved through [`Run::pool`] and
    /// [`Run::recurrent`], never through [`Run::tensor`].
    caches: &'c CacheTable,

    /// Plan payloads: filled by the plan-building arms in the prepare phase,
    /// read by the consuming arms afterwards.
    ///
    /// **KEYED BY `(RUN, VALUE)`, NOT BY VALUE**, and the extra key is P4's
    /// split (design §3). A schedule is carved for ONE window, so a region the
    /// layout could not seat carves one per interval of it — all built in the
    /// prepare phase, all read in the capture phase. One slot per value would
    /// let run 1's builder overwrite run 0's, and run 0's encode would then
    /// read a schedule describing run 1's requests: not a fault, just wrong
    /// logits for the lanes in the first interval.
    ///
    /// Flat rather than nested, at `run * values + value`: a plan has
    /// thousands of values and a `Vec` per value would be thousands of
    /// allocations per fire, where this is one. The width is
    /// [`Windows::max_runs`] — `1` for every artifact P4 seated whole.
    structs: Vec<Option<StructSlot>>,
    /// How many values one run's slice of [`structs`](Run::structs) holds.
    values_wide: usize,

    /// This fire's bindings.
    fire: FireBindings,

    /// Every region's windows, resolved once per fire from the composition's
    /// class table. Indexed by [`place`](Run::place).
    windows: &'c Windows,

    /// Which region the walk is inside and which run of its window, written
    /// by [`Cursor`](crate::window::Cursor) on `region_begin` and on `run` —
    /// before that region's nodes are dispatched, and before each encode of
    /// them. **THIS IS THE WHOLE MIXED-FIRE MECHANISM**: the walk's `Dispatch`
    /// signature is fixed and carries no region, so the sink and the resolver
    /// share one cell instead.
    place: &'c At,
}

impl<'c> Run<'c> {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &'c Ctx<'c>,
        handles: &'c Handles,
        values: &'c [ValueDecl],
        weights: &'c WeightTable,
        arena: &'c SlotTable,
        caches: &'c CacheTable,
        fire: FireBindings,
        windows: &'c Windows,
        place: &'c At,
    ) -> Self {
        Self {
            ctx,
            handles,
            values,
            weights,
            arena,
            caches,
            structs: vec![None; values.len() * windows.max_runs() as usize],
            values_wide: values.len(),
            fire,
            windows,
            place,
        }
    }

    /// The window of the region the walk is inside, cut at the run it is on.
    pub(crate) fn window(&self) -> &'c Window {
        self.windows.at(self.place.region.get(), self.place.run.get())
    }

    /// Where this run's payload for `id` sits in [`structs`](Run::structs).
    ///
    /// The run comes off the same cell the window does, so a schedule is
    /// stored and read at the same key by construction — a builder cannot
    /// carve for one interval and an encode read another.
    fn struct_at(&self, id: ValueId) -> usize {
        self.place.run.get() as usize * self.values_wide + id.0 as usize
    }

    /// This window's own qo boundaries, rebased to start at zero.
    pub(crate) fn qo_indptr(&self) -> Tensor {
        self.window().indptr
    }

    /// The same vector, host-side — what an entry that needs to READ a
    /// boundary (rather than bind it) takes.
    ///
    /// Unread today: every metal entry that is boundary-aware takes the
    /// pair (`RaggedTensor`) and walks it on the device. Kept beside
    /// [`Run::multi_token`] and [`Run::total_tokens`], which are the two
    /// questions a host-side arm-picking builder would ask of it, because
    /// the answer is one line and re-deriving it in a dispatch arm is how
    /// two readings of one window come to disagree.
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

    /// A raw ambient table, cut to this window's ROWS.
    ///
    /// The plan builders' tables (`positions`, `request_of_token`, the mask
    /// planes) are staged once per fire at absolute fire rows, and the
    /// shaders index them by the LOCAL row of the launch — `req_of_token[row]`
    /// where `row` runs `0..q.rows`. So a windowed launch needs them cut the
    /// way its `q` is. They are not `ValueId`s, so [`Run::tensor`]'s
    /// declaration-driven cut cannot reach them and this is the same
    /// arithmetic, stated for a bare handle.
    ///
    /// What stays ABSOLUTE is what the shaders index by lane through a
    /// value the table itself holds: `request_of_token`'s entries are lane
    /// ids into the fire-wide `page_indptr`, and cutting the vector does not
    /// renumber its contents — which is exactly the property that makes one
    /// page table serve every window.
    pub(crate) fn cut_rows(&self, handle: Tensor) -> Tensor {
        let span = self.window().span;
        self.slice(handle, span.row_offset, span.rows)
    }

    /// The encode sink, for the arms.
    pub(crate) fn ctx(&self) -> &'c Ctx<'c> {
        self.ctx
    }

    /// The fire bindings, for the plan-building arms' seam.
    pub(crate) fn bindings(&self) -> &FireBindings {
        &self.fire
    }

    /// One rectangle, sliced to `keep` rows starting at `skip`.
    ///
    /// The one place a windowed cut becomes a handle. A CUDA shell answers
    /// this with `ptr + skip * stride` and no state; here the row is minted
    /// into [`Handles`], which is also where the bounds check lives — a cut
    /// past the end of the reservation is caught before a shader
    /// dereferences it.
    ///
    /// A cut that fails is an INTEGRITY failure, not a refusal: the offsets
    /// come from the compiler's carve and the composition's window table,
    /// and a disagreement between those two is a bug in this crate or in the
    /// bake. It panics with a sentence, per the file's rule.
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
    ///
    /// **EVERY ROW-SHAPED TABLE IN THIS SHELL IS INDEXED BY ABSOLUTE FIRE
    /// ROW** — the arena carve gives a `Dim::Tokens` value one column at the
    /// fire's row count, the geometry vectors one entry per fire lane — so a
    /// window is a slice, and which slice is read off the value's own leading
    /// `Dim`. A `Dim::Const` column (a weight plane, a bias) is not
    /// fire-aligned and is handed over whole.
    ///
    /// `GeomKind::Indices` is the one declared shape that is not what it
    /// says: the IR spells the flat page-id list `Dim::Lanes` because it has
    /// no page symbol, and its entries are pages rather than lanes. Slicing
    /// it by a lane offset would hand a windowed consumer somebody else's
    /// pages, so it is excluded — and its bounds vector stays absolute,
    /// which is what makes a sliced `Indptr` still address the whole list.
    ///
    /// `RuntimeInput::Mask` is the second of those, for the same reason: the
    /// IR spells the mask slab `Dim::Tokens` and its entries are (query,
    /// key) BITS, so a row offset is not a byte offset. This plane binds the
    /// mask through [`FireTables`] rather than through a declared input and
    /// cuts it with [`Run::cut_rows`], which knows the stride.
    fn cut(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
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
        let window = self.window().span;
        let (skip, keep) = match shape.first() {
            Some(Dim::Tokens) => (window.row_offset, window.rows),
            Some(Dim::TokensTimes(k)) => (window.row_offset * k, window.rows * k),
            Some(Dim::Lanes) => (window.lane_offset, window.lanes),
            Some(Dim::LanesPlus(k)) => (window.lane_offset, window.lanes + k),
            Some(Dim::Const(_)) | None => return handle,
        };
        self.slice(handle, skip, keep)
    }

    /// The crate's heart: one plan id in, one device handle out, routed on
    /// the id's `Def`. Every dispatch arm resolves through here, so
    /// provenance handling exists exactly once.
    ///
    /// Cache ids never resolve to a tensor — a cache is a pool pointer and
    /// resolves through [`Run::pool`] or [`Run::recurrent`]; a cache id
    /// arriving here is a dispatch-arm bug, answered with a panic. So is a
    /// split-plane weight: two planes resolve through [`Run::planes`].
    pub(crate) fn tensor(&self, id: ValueId) -> Tensor {
        self.cut(id, self.whole(id))
    }

    /// The same resolution, uncut — the fire-wide rectangle a value names.
    fn whole(&self, id: ValueId) -> Tensor {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Input(RuntimeInput::Tokens) => self.fire.tokens,
            Def::Input(RuntimeInput::Positions) => self.fire.positions,
            // The op-named mask (`attention.masked`) resolves onto the fire
            // table the plan builders already carry: this plane binds one
            // mask per fire — every sdpa launch reads its seats — so a
            // second seat would only exist to drift, and the space
            // collapses onto it.
            Def::Input(RuntimeInput::Mask { space: _ }) => self.fire.tables.mask,
            // THE ADAPTER AXIS IS NOT SERVED HERE, AND THE PANIC IS THE
            // HONEST ANSWER RATHER THAN A ZERO HANDLE. This plane binds no
            // adapter bank (`linear.lora_correct` answers
            // `KernelError::Unsupported` in this shell's dispatch arm), so
            // nothing can reach this id: a plan carrying a correction op is
            // refused at its first correction node, one dispatch earlier and
            // with a sentence naming the op. Resolving a bare zero here would
            // be routing every row to adapter zero of a bank that does not
            // exist.
            Def::Input(RuntimeInput::AdapterRoutes) => panic!(
                "value {at} reads the fire's adapter ids, which this plane binds no seat                  for; `linear.lora_correct` is what refuses the plan, by name"
            ),
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
                    Some(WeightRow::Planes { .. }) => panic!(
                        "value {at} is weight {row}, a split-plane bank; it resolves \
                         through `Run::planes`, never as one dense handle"
                    ),
                    None => panic!("value {at} is weight {row}, which the shell has not bound"),
                }
            }
            // A φ resolves like the op output it merges: the compiler
            // aliased every arm onto one arena slot, written at this id's
            // row — so `Merge` is the same read as `Op`.
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

    /// A fire-aligned value viewed through THIS WINDOW's boundaries. The
    /// indptr is ambient (design §5): no op names it, and this pairing is
    /// where it re-enters.
    ///
    /// The boundaries are the window's own, rebased: `data` already points
    /// at the window's first row, so a fire-wide vector would send every
    /// ragged entry past the end of the rectangle it was handed, by exactly
    /// the number of rows the classes before it occupy.
    pub(crate) fn ragged(&self, id: ValueId) -> RaggedTensor {
        RaggedTensor {
            data: self.tensor(id),
            indptr: self.qo_indptr(),
        }
    }

    /// The `(codes, scales)` planes of a split-plane bank — the resolution
    /// `linear.moe_matmul_select_bias` needs where [`Run::tensor`] would have
    /// to lie with one handle.
    pub(crate) fn planes(&self, id: ValueId) -> (Tensor, Tensor) {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            panic!("value {at} is not a weight, and split-plane banks live in the weight table");
        };
        let row = *w as usize;
        match self.weights.0.get(row).copied().flatten() {
            Some(WeightRow::Planes { codes, scales }) => (codes, scales),
            Some(WeightRow::Dense(_)) => panic!(
                "value {at} is weight {row}, bound as one dense handle, and this op reads \
                 a split-plane bank"
            ),
            None => panic!("value {at} is weight {row}, which the shell has not bound"),
        }
    }

    /// The `StructKind` a plan op's output value declares — how the
    /// plan-building arms check the trace against what this plane can build:
    /// the trace wrote the choice into `Plan::values`, the arm only follows
    /// it.
    pub(crate) fn declared(&self, id: ValueId) -> StructKind {
        match &self.values[id.0 as usize].ty {
            Ty::Struct(kind) => *kind,
            Ty::Tensor { .. } => panic!(
                "value {} declares a tensor, and a plan op defines a struct",
                id.0
            ),
        }
    }

    /// The paged kv pool a cache id names.
    ///
    /// **NOTHING IS CUT HERE, AND THAT IS A STATEMENT ABOUT THIS PLANE'S
    /// ABI.** The CUDA sibling slices its pool's `page_indptr`,
    /// `last_page_lens` and `row_valid` to the window, because its schedules
    /// number requests from the launch's own zero. The metal sdpa entries
    /// read the page table through `kv_page_indptr[req_of_token[row]]`, and
    /// `request_of_token` is staged with ABSOLUTE lane ids — so the table
    /// stays fire-wide and the two agree. Slicing it here would send every
    /// windowed launch to somebody else's pages. (`page_indices` is absolute
    /// on both planes, for the reason `Run::cut` gives.)
    pub(crate) fn pool(&self, id: ValueId) -> &KvPool {
        match self.cache(id) {
            CachePool::Kv(pool) => pool,
            CachePool::Recurrent(_) => panic!(
                "value {} is a recurrent state space, and this op walks a paged kv pool",
                id.0
            ),
        }
    }

    /// The recurrent state pool a cache id names, with its slot map cut to
    /// the asking node's window.
    ///
    /// A recurrent bank is addressed by SLOT, and every metal ssm shader
    /// reads its slot out of `slots[r]` where `r` counts from the LAUNCH's
    /// own zero — so a windowed scan gets the window's rows of that vector
    /// and nothing else. The banks themselves are the model's state, whole.
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
