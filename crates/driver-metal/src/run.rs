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
use model_ir::{Def, GeomKind, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};

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
/// `RuntimeInput` routes onto them in [`Run::tensor`]. The rest is ambient:
/// the shared `indptr` (design §5 removed `qo_indptr` as a named input, so
/// ragged views are assembled here), the fire `tables` for the plan
/// builders, and the fact word the walk guards on.
#[derive(Clone, Debug)]
pub struct FireBindings {
    /// `RuntimeInput::Tokens`: ragged `i32`, one id per token.
    pub tokens: Tensor,

    /// `RuntimeInput::Positions`: ragged `i32`, one absolute position per
    /// token — also the plan builders' causal-bound table.
    pub positions: Tensor,

    /// The fire's one shared boundary vector — `i32`, `[lanes + 1]` —
    /// through which every fire-aligned value is viewed ([`Run::ragged`]).
    pub indptr: Tensor,

    /// Per cache space, aligned with `Plan::caches`:
    /// `RuntimeInput::Geometry { space, kind }` routes to that space.
    pub geometry: Vec<CacheGeometry>,

    /// The fire tables the attention plan builders consume.
    pub tables: FireTables,

    /// The fire's fact word — `Cond::holds`'s input, read by the walk and
    /// never by dispatch; carried here so the shell hands one bundle per
    /// fire.
    pub facts: u64,
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
/// constructs one per fire and drives `kernels::walk` over it — prepare
/// phase first, so every plan payload exists before its consumers encode.
pub struct Run<'c> {
    /// The encode sink — the real shell behind `dyn Encode`. Everything this
    /// crate does to the device goes through it; nothing here names Metal.
    ctx: &'c Ctx<'c>,

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
        ctx: &'c Ctx<'c>,
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

    /// The fire's fact word, for the shell to hand `kernels::walk`.
    #[must_use]
    pub fn facts(&self) -> u64 {
        self.fire.facts
    }

    /// The encode sink, for the arms.
    pub(crate) fn ctx(&self) -> &'c Ctx<'c> {
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
            // The op-named mask (`attention.masked`) resolves onto the fire
            // table the plan builders already carry: this plane binds one
            // mask per fire — every sdpa launch reads its seats — so a
            // second seat would only exist to drift, and the space
            // collapses onto it.
            Def::Input(RuntimeInput::Mask { space: _ }) => self.fire.tables.mask,
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

    /// Store a plan payload a prepare-phase arm just built.
    pub(crate) fn put(&mut self, id: ValueId, built: StructSlot) {
        self.structs[id.0 as usize] = Some(built);
    }

    /// The decode plan a consuming arm names.
    pub(crate) fn decode_plan(&self, id: ValueId) -> &DecodePlan {
        match &self.structs[id.0 as usize] {
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
        match &self.structs[id.0 as usize] {
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
