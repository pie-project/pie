//! The §8 `Run`, on cuda's stream: one fire's dispatch state, and the one
//! function that turns a plan id into a device handle.
//!
//! Long-lived state — the weight and arena tables, the cache pools — arrives
//! pre-built and borrowed: building those tables is the shell's binding
//! business, not this layer's. Fire-lived state — the input bindings, the
//! plan payloads — is owned: a `Run` is constructed per fire and dropped
//! with it.
//!
//! Everything here answers to one rule: a [`KernelError`] is about the
//! backend, never about the plan (`new_kernels::error`). A hole in a table,
//! a cache id in a tensor seat, a plan consumed before its plan op — those
//! are integrity failures of the shell or the compiler, and they panic with
//! a sentence instead of dressing up as a backend refusal.
//!
//! [`KernelError`]: new_kernels::KernelError

use new_kernels_cuda::attn::plan::{
    DecodePlan, Device, Mask, MlaPlan, PrefillPlan, PrefillPlanSm90, Shape, Workspace,
};
use new_kernels_cuda::{Ctx, KvPool, RaggedTensor, RecurrentPool, Tensor};
use new_model_ir::{Def, GeomKind, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};

/// One loader-resolved weight. Most rows are one dense handle; an mxfp4 bank
/// is two device planes under one `Def::Weight` id — the form the metal
/// shell's one-handle rows could not seat (its `MENLO-SEAM` at
/// `moe.matmul_select_bias`). This plane stamps the mxfp4 kernels, so the
/// table names the form instead of refusing it.
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
/// §7); its geometry rides in [`FireBindings`] as declared inputs. On this
/// plane the [`KvPool`] row also carries the write tables
/// (`write_page`/`write_offset`) and the graph-padding `row_valid` the
/// appenders address by — the shell derives them from the same declared
/// inputs when it builds the row (the appenders' `MENLO-SEAM`).
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

/// The host half of one cache space's geometry — THE cuda duality. The IR
/// names `kv_indptr` as a device input, and [`Run::tensor`] serves it to
/// launches; but the plan builders are host functions that walk its
/// *contents*, and a device handle cannot be read host-side. So the shell
/// binds the same vector twice, and this seat is the host twin — plus the
/// carved facts the builders take beside it.
///
/// Bound only for cache spaces a plan op names; a plan op firing over a
/// space with no planning seat is a binding bug and panics.
#[derive(Clone, Debug)]
pub struct CachePlanning {
    /// Host copy of the space's `GeomKind::Indptr` contents — what
    /// `plan_decode`/`plan_prefill` walk (the builders' `MENLO-SEAM`).
    pub kv_indptr: Vec<i32>,

    /// Per-request kv lengths in tokens, read by the sm90 and mla builders.
    // MENLO-SEAM: no IR seat — the driver derives it host-side from
    // kv_indptr + last_page_len (design §7 keeps derived quantities
    // host-side and named), as the old plane did.
    pub kv_len: Vec<i32>,

    /// The kv-side shape this space's plans are carved at. The consuming
    /// ops restate `head_dim`/`kv_heads` and the entries refuse a
    /// disagreement; for a latent (mla) space, `head_dim` is the output
    /// head width the schedule sizes at.
    pub shape: Shape,

    /// The sliding window this space's schedules are carved for; the
    /// entries check each consumer's stated window against the plan.
    pub window: Option<u32>,

    /// The decode plan's workspace grant. Grants are disjoint carvings of
    /// the shell's bounded pool — one per plan kind, because their staged
    /// int images coexist within a fire.
    pub decode_workspace: Option<Workspace>,

    /// The prefill (fa2 or sm90) plan's workspace grant.
    pub prefill_workspace: Option<Workspace>,

    /// The mla plan's workspace grant.
    pub mla_workspace: Option<Workspace>,
}

impl CachePlanning {
    /// The decode grant, or a sentence: an ungranted workspace is a binding
    /// bug, not a backend refusal.
    #[must_use]
    pub fn decode_grant(&self) -> Workspace {
        self.decode_workspace.unwrap_or_else(|| {
            panic!("this cache space grants no decode-plan workspace, which the shell carves before a decode plan op can fire")
        })
    }

    /// The prefill grant.
    #[must_use]
    pub fn prefill_grant(&self) -> Workspace {
        self.prefill_workspace.unwrap_or_else(|| {
            panic!("this cache space grants no prefill-plan workspace, which the shell carves before a prefill plan op can fire")
        })
    }

    /// The mla grant.
    #[must_use]
    pub fn mla_grant(&self) -> Workspace {
        self.mla_workspace.unwrap_or_else(|| {
            panic!("this cache space grants no mla-plan workspace, which the shell carves before an mla plan op can fire")
        })
    }
}

/// The geometry one cache space declared: the device seats the ops read,
/// and the host planning twin beside them (the duality [`CachePlanning`]
/// names). Only what the plan names gets bound, so every seat is optional;
/// resolving an unbound seat is a binding bug and panics.
#[derive(Clone, Debug, Default)]
pub struct CacheGeometry {
    /// `RuntimeInput::Geometry { kind: Indptr }`.
    pub indptr: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: Indices }`.
    pub indices: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: SeqLens }`.
    pub seq_lens: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: LastPageLen }`.
    pub last_page_len: Option<Tensor>,

    /// The host twin the plan builders walk, bound for spaces a plan op
    /// names.
    pub planning: Option<CachePlanning>,
}

/// The dsv4 compressor state `pool.gather` reads beside its cache.
// MENLO-SEAM: no IR seat — the driver binds the slabs it staged for the
// pooled space (the marker at `new_kernels_cuda::attn::pool::gather`).
#[derive(Clone, Copy, Debug)]
pub struct PoolSlabs {
    /// The rolling kv window state.
    pub state_kv: Tensor,

    /// The rolling score state.
    pub state_score: Tensor,

    /// The absolute-position-embedding plane.
    pub ape: Tensor,
}

/// The driver-bound extras the cuda entries want beside the ops' named
/// operands. No op names these — every seat here is the driver side of a
/// `MENLO-SEAM` marker in `new_kernels_cuda`, bound from fire state by the
/// arm that carries the matching comment.
#[derive(Clone, Copy, Debug)]
pub struct FireTables {
    /// `i32`, one per token: the owning request — `pool.attention_lse`
    /// walks entries by it.
    pub request_of_token: Tensor,

    /// `u8` per row: the CUDA-graph padding mask the boundary kernels read
    /// before any pool row is named (`pool.boundary_*`'s marker).
    pub row_valid: Tensor,

    /// The packed mask pair bound onto the prefill plan at build, for
    /// `attention.masked` (`attn::masked`'s marker). `None` when this fire
    /// carries no mask; a masked consumer then gets the entry's own typed
    /// refusal, not a panic — mask-lessness is a run-time fact, not a
    /// binding hole.
    pub mask: Option<Mask>,

    /// The dsv4 compressor slabs, bound when a pooled space exists. One
    /// fire-wide seat: a plan carries at most one pooled space today; this
    /// moves onto [`CacheGeometry`] the day one carries two.
    pub pool_state: Option<PoolSlabs>,
}

/// What the driver binds each fire, owned by the [`Run`] for its lifetime.
///
/// `tokens`, `positions`, and `geometry` are the op-visible inputs —
/// `RuntimeInput` routes onto them in [`Run::tensor`]. The rest is ambient:
/// the shared `indptr` and its host twin (design §5 removed `qo_indptr` as
/// a named input, so ragged views assemble here and the prefill/mla
/// builders walk the twin), the seam `tables`, the pre-probed `device`
/// facts and the shell's `capture` policy word the builders take, and the
/// fact word the walk guards on.
#[derive(Clone, Debug)]
pub struct FireBindings {
    /// `RuntimeInput::Tokens`: ragged `i32`, one id per token.
    pub tokens: Tensor,

    /// `RuntimeInput::Positions`: ragged `i32`, one absolute position per
    /// token.
    pub positions: Tensor,

    /// The fire's one shared boundary vector — `i32`, `[lanes + 1]` —
    /// through which every fire-aligned value is viewed ([`Run::ragged`]).
    pub indptr: Tensor,

    /// The host twin of `indptr` — the `qo_indptr` slice the prefill and
    /// mla builders walk. The same duality [`CachePlanning::kv_indptr`]
    /// states per cache space, stated once more here because this vector is
    /// fire-wide.
    pub indptr_host: Vec<i32>,

    /// Per cache space, aligned with `Plan::caches`:
    /// `RuntimeInput::Geometry { cache, kind }` routes to row `cache`, and
    /// the plan-building arms route to row `cache`'s planning twin.
    pub geometry: Vec<CacheGeometry>,

    /// The seam extras the arms bind beside the ops' named operands.
    pub tables: FireTables,

    /// The device facts every builder takes — pre-probed by the shell
    /// (`Device::probe` once at boot, or a stated fallback); the builders
    /// themselves never probe, purity is their design.
    pub device: Device,

    /// The shell's graph policy word: whether this fire's capture phase
    /// will be captured as a CUDA graph. Builders carve graph-shaped,
    /// padded schedules under it; `PrefillPlan::graph_capturable` answers
    /// whether they managed. Policy stays the shell's — this word only
    /// carries it to the builders.
    pub capture: bool,

    /// The fire's fact word — `Cond::holds`'s input, read by the walk and
    /// never by dispatch; carried here so the shell hands one bundle per
    /// fire.
    pub facts: u64,
}

impl FireBindings {
    /// This fire's token count, as the last boundary of the host indptr —
    /// the `total_num_rows` the prefill builders take.
    #[must_use]
    pub fn total_tokens(&self) -> u32 {
        let last = self.indptr_host.last().copied().unwrap_or_else(|| {
            panic!("the host indptr is empty, and a fire's boundaries are at least [0]")
        });
        u32::try_from(last).unwrap_or_else(|_| {
            panic!("the host indptr ends at {last}, which is not a token count")
        })
    }

    /// Whether any lane carries more than one token this fire — the mla
    /// builder's `causal` word, derived rather than seated: multi-token
    /// lanes attend causally within themselves, single-token (decode) lanes
    /// have nothing to order.
    #[must_use]
    pub fn multi_token(&self) -> bool {
        self.indptr_host
            .windows(2)
            .any(|span| span[1] - span[0] > 1)
    }
}

/// One built plan payload. An enum over the four kinds this plane can be
/// asked to build, not `Box<dyn Any>`: the IR's `StructKind` is closed, and
/// this crate names every payload type at compile time — erasure would buy
/// no generality, only a silent-downcast failure mode. Here a wrong kind is
/// a named panic.
#[derive(Clone, Debug)]
pub enum StructSlot {
    /// `StructKind::AttnDecodePlan`.
    Decode(DecodePlan),

    /// `StructKind::AttnPrefillPlan`.
    Prefill(PrefillPlan),

    /// `StructKind::AttnPrefillPlanSm90` — built when a trace declares its
    /// prefill plan at this kind; the consumer entry (`attn::prefill_sm90`)
    /// still answers a typed refusal, as the old plane did.
    PrefillSm90(PrefillPlanSm90),

    /// `StructKind::MlaPlan`.
    Mla(MlaPlan),
}

/// One fire's dispatch state: the stream context, the resolution tables,
/// the fire bindings, and the plan payloads this fire builds. The shell
/// constructs one per fire and drives `new_driver::walk` over it — prepare
/// phase first (outside any capture), so every plan payload exists and is
/// staged before its consumers enqueue.
pub struct Run<'c> {
    /// The stream and its companions (cuBLAS handle, communicator, jit
    /// cache behind it). Everything this crate does to the device goes
    /// through it, enqueue only.
    ctx: &'c Ctx,

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

    /// `ValueId`-indexed plan payloads: filled by the plan-building arms in
    /// the prepare phase, read by the consuming arms afterwards.
    structs: Vec<Option<StructSlot>>,

    /// This fire's bindings.
    fire: FireBindings,
}

impl<'c> Run<'c> {
    #[must_use]
    pub fn new(
        ctx: &'c Ctx,
        values: &'c [ValueDecl],
        weights: &'c WeightTable,
        arena: &'c SlotTable,
        caches: &'c CacheTable,
        fire: FireBindings,
    ) -> Self {
        Self {
            ctx,
            values,
            weights,
            arena,
            caches,
            structs: vec![None; values.len()],
            fire,
        }
    }

    /// The fire's fact word, for the shell to hand `new_driver::walk`.
    #[must_use]
    pub fn facts(&self) -> u64 {
        self.fire.facts
    }

    /// The stream context, for the arms.
    pub(crate) fn ctx(&self) -> &'c Ctx {
        self.ctx
    }

    /// The fire bindings, for the plan-building arms' seam.
    pub(crate) fn bindings(&self) -> &FireBindings {
        &self.fire
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
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Input(RuntimeInput::Tokens) => self.fire.tokens,
            Def::Input(RuntimeInput::Positions) => self.fire.positions,
            Def::Input(RuntimeInput::Geometry { cache, kind }) => {
                let seat = self.geometry(at, *cache);
                let bound = match kind {
                    GeomKind::Indptr => seat.indptr,
                    GeomKind::Indices => seat.indices,
                    GeomKind::SeqLens => seat.seq_lens,
                    GeomKind::LastPageLen => seat.last_page_len,
                };
                bound.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads {kind:?} of cache space {cache}, which this \
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
            Def::Op(_) | Def::Merge(_) => {
                self.arena.0.get(at).copied().flatten().unwrap_or_else(|| {
                    panic!("value {at} has no arena slot, which the compiler should have cut")
                })
            }
            Def::Cache(_) => panic!(
                "value {at} is a cache space; it resolves to a pool through `Run::pool`, \
                 never to a tensor"
            ),
        }
    }

    /// A fire-aligned value viewed through the fire's shared boundaries. The
    /// indptr is ambient (design §5): no op names it, and this pairing is
    /// where it re-enters.
    pub(crate) fn ragged(&self, id: ValueId) -> RaggedTensor {
        RaggedTensor {
            data: self.tensor(id),
            indptr: self.fire.indptr,
        }
    }

    /// The `(codes, scales)` planes of a split-plane bank — the resolution
    /// `moe.matmul_select_bias` needs where [`Run::tensor`] would have to
    /// lie with one handle.
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

    /// The paged kv pool a cache id names.
    pub(crate) fn pool(&self, id: ValueId) -> &KvPool {
        match self.cache(id) {
            CachePool::Kv(pool) => pool,
            CachePool::Recurrent(_) => panic!(
                "value {} is a recurrent state space, and this op walks a paged kv pool",
                id.0
            ),
        }
    }

    /// The recurrent state pool a cache id names.
    pub(crate) fn recurrent(&self, id: ValueId) -> &RecurrentPool {
        match self.cache(id) {
            CachePool::Recurrent(pool) => pool,
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

    fn geometry(&self, at: usize, cache: u32) -> &CacheGeometry {
        let space = cache as usize;
        self.fire.geometry.get(space).unwrap_or_else(|| {
            panic!(
                "value {at} names cache space {space}, and this fire binds {} geometry spaces",
                self.fire.geometry.len()
            )
        })
    }

    /// The planning twin of the cache space a plan op's geometry input
    /// names. The op states `kv_indptr` as a device value; its `Def` says
    /// which space that is, and the space's [`CachePlanning`] holds what
    /// the builders actually walk — the duality, routed in one place.
    pub(crate) fn planning(&self, geom: ValueId) -> &CachePlanning {
        let at = geom.0 as usize;
        let Def::Input(RuntimeInput::Geometry { cache, .. }) = &self.values[at].def else {
            panic!(
                "value {at} is not declared cache geometry, and a plan op routes to its \
                 cache space through its geometry input"
            );
        };
        let seat = self.geometry(at, *cache);
        seat.planning.as_ref().unwrap_or_else(|| {
            panic!(
                "cache space {cache} carries no planning seat; the shell binds the host \
                 geometry and plan facts before a plan op can fire"
            )
        })
    }

    /// The `StructKind` a plan op's output value declares — how the
    /// prefill-building arm tells fa2 from sm90: the trace wrote the choice
    /// into `Plan::values`, the arm only follows it.
    pub(crate) fn declared(&self, id: ValueId) -> StructKind {
        match &self.values[id.0 as usize].ty {
            Ty::Struct(kind) => *kind,
            Ty::Tensor { .. } => panic!(
                "value {} declares a tensor, and a plan op defines a struct",
                id.0
            ),
        }
    }

    /// The dsv4 compressor slabs, for `pool.gather`'s seam.
    pub(crate) fn slabs(&self) -> PoolSlabs {
        self.fire.tables.pool_state.unwrap_or_else(|| {
            panic!("this fire binds no dsv4 compressor slabs, which `pool.gather` reads beside the pool")
        })
    }

    /// Store a plan payload a prepare-phase arm just built.
    pub(crate) fn put(&mut self, id: ValueId, built: StructSlot) {
        self.structs[id.0 as usize] = Some(built);
    }

    /// One built slot, whichever kind it holds — for the arm that routes on
    /// the kind (prefill's fa2/sm90 fork); the typed accessors below are the
    /// single-kind reads.
    pub(crate) fn slot(&self, id: ValueId) -> &StructSlot {
        self.structs[id.0 as usize].as_ref().unwrap_or_else(|| {
            panic!(
                "value {} holds no plan payload; its plan op has not fired, and the \
                 prepare phase runs first",
                id.0
            )
        })
    }

    /// The decode plan a consuming arm names.
    pub(crate) fn decode_plan(&self, id: ValueId) -> &DecodePlan {
        match self.slot(id) {
            StructSlot::Decode(plan) => plan,
            _ => panic!(
                "value {} holds another plan kind, and this op consumes a decode plan",
                id.0
            ),
        }
    }

    /// The fa2 prefill plan a consuming arm names.
    pub(crate) fn prefill_plan(&self, id: ValueId) -> &PrefillPlan {
        match self.slot(id) {
            StructSlot::Prefill(plan) => plan,
            _ => panic!(
                "value {} holds another plan kind, and this op consumes an fa2 prefill plan",
                id.0
            ),
        }
    }

    /// The mla plan a consuming arm names.
    pub(crate) fn mla_plan(&self, id: ValueId) -> &MlaPlan {
        match self.slot(id) {
            StructSlot::Mla(plan) => plan,
            _ => panic!(
                "value {} holds another plan kind, and this op consumes an mla plan",
                id.0
            ),
        }
    }
}
