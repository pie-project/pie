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
//! backend, never about the plan (`kernels::error`). A hole in a table,
//! a cache id in a tensor seat, a plan consumed before its plan op — those
//! are integrity failures of the shell or the compiler, and they panic with
//! a sentence instead of dressing up as a backend refusal.
//!
//! [`KernelError`]: kernels::KernelError

use std::cell::Cell;

use kernels_cuda::attn::plan::{
    DecodePlan, Device, MlaPlan, PrefillPlan, PrefillPlanSm90, Shape, Toggles, Workspace,
};
use kernels_cuda::linear::lora::Segments;
use kernels_cuda::{Ctx, KvPool, Pad, RaggedTensor, RecurrentPool, Tensor};
use model_ir::{Def, Dim, GeomKind, Node, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};

use crate::dispatch::copy::CopyPlan;
use crate::window::{At, Window, Windows};

/// One loader-resolved weight. Most rows are one dense handle; an mxfp4 bank
/// is two device planes under one `Def::Weight` id. Both shells seat the form
/// the same way now — the table names it instead of refusing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRow {
    /// One dense handle, resolved by [`Run::tensor`].
    Dense(Tensor),

    /// A split-plane quantized bank — e2m1 `codes` beside e8m0 `scales` —
    /// resolved by [`Run::planes`], never as one tensor.
    Planes { codes: Tensor, scales: Tensor },
}

/// Loader-resolved weights, one row per `Trace::params` entry —
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
/// plane the [`KvPool`] row also carries the graph-padding `row_valid` the
/// writer kernels read — the shell derives it from the same declared inputs
/// when it builds the row. (The write tables the row used to smuggle are
/// gone: `write_page`/`write_offset` are op-named inputs now.)
#[derive(Clone, Copy, Debug)]
pub enum CachePool {
    /// A paged kv space (`CacheRow::Kv`), and which geometry space's tables
    /// address it.
    ///
    /// **THE SPACE RIDES ALONG BECAUSE A GATHERED WINDOW HAS ITS OWN.** A
    /// pool's page bounds and last-page fills are the fire's, seated per
    /// geometry space at bind; a `Fallback::Copy` re-cuts them for its
    /// compacted lanes and seats the result per space too
    /// (`window::GatheredSpace`), so resolving one means knowing which space
    /// this row belongs to. The number is the cache row's own declaration
    /// (`CacheRow::Kv { space }`) — restated here rather than looked up,
    /// because `Run::pool` holds the pool and not the plan's cache table.
    Kv {
        /// The `CacheRow::Kv` space this row's geometry comes from.
        space: u32,
        /// The storage and the tables addressing it.
        pool: KvPool,
    },
    /// A recurrent state space (`CacheRow::State`).
    Recurrent(RecurrentPool),
}

/// Cache-index-indexed pools, aligned with `Trace::caches`.
#[derive(Clone, Debug, Default)]
pub struct CacheTable(pub Vec<CachePool>);

/// The host half of one cache space's geometry — THE cuda duality. The IR
/// names `kv_indptr` as a device input, and [`Run::tensor`] serves it to
/// launches; but the plan builders are host functions that walk its
/// *contents*, and a device handle cannot be read host-side. So the shell
/// binds the same vector twice, and this seat is the host twin.
///
/// **THE TWINS ARE THE SPACE'S; THE READING IS THE SCHEDULE'S.** Shape,
/// window and workspace used to sit here and moved to [`ScheduleSeat`] in
/// C1b, because a page-id space says which page a token lands in and nothing
/// about how wide that row is or how far back a reader may look — and gemma
/// states two answers to both over one sequence.
///
/// Bound only for cache spaces a plan op names; a plan op firing over a
/// space with no planning seat is a binding bug and panics.
#[derive(Clone, Debug)]
pub struct CachePlanning {
    /// Host copy of the space's `GeomKind::Indptr` contents — what
    /// `plan_decode`/`plan_prefill` walk (the builders' `MENLO-SEAM`).
    pub kv_indptr: Vec<i32>,

    /// Host copy of the space's `GeomKind::KvLen` contents — per-request kv
    /// lengths in tokens, the op-named input the sm90 and mla builders walk
    /// (the fa2 builders take it and leave it unread). The same duality as
    /// `kv_indptr`: the device tensor serves launches, this twin serves the
    /// host planners.
    pub kv_len: Vec<i32>,
}

/// What ONE attention schedule is carved for, and where it is staged.
///
/// **KEYED BY THE PLAN VALUE, NOT BY THE SPACE** (build log 20's first
/// blocker). A page-id space says which page a token lands in; it says
/// nothing about how wide that row is or how far back a reader may look, and
/// gemma states two answers to both over one sequence. So the reading and the
/// grant belong to the schedule that was carved for them, and a family that
/// carves two mints two plan values — which is also what makes each one's
/// staged int image its own.
///
/// The reading itself comes off the PLAN OP: `Attention::PlanDecode` and its
/// two siblings state the query heads, kv heads, head width and window their
/// schedule is carved for, and `store::kv::probe` seats what they say. The
/// launches restate their share and the shell refuses a disagreement.
#[derive(Clone, Copy, Debug)]
pub struct ScheduleSeat {
    /// The kv-side shape this schedule is carved at, at the FIRE's lanes;
    /// [`Run::planning`] narrows `num_requests` to the asking node's window.
    /// The consuming ops restate `head_dim`/`kv_heads` and the entries refuse
    /// a disagreement; for a latent (mla) schedule, `head_dim` is the output
    /// head width it sizes at.
    pub shape: Shape,

    /// The sliding window this schedule carved its kv spans for; the entries
    /// check each consumer's stated window against it.
    pub window: Option<u32>,

    /// Where the built schedule's staged image lands.
    pub workspace: Workspace,
}

/// One cache space's planning twin, CUT TO THE WINDOW of the node asking.
///
/// A plan build is per-window work: an all-decode fire's prefill schedule is
/// never built (the walk skips the empty window, design §5 step 4), and a
/// MIXED fire builds both — each over its own lanes. So the builders must not
/// see the fire's whole geometry, or the decode schedule would carve requests
/// for the prefill lanes and the launch would read a schedule wider than the
/// rectangle it was handed.
///
/// The slices borrow the fire-wide host twins rather than copying them: the
/// window is contiguous in lanes (seriation, design §3) and the builders read
/// DIFFERENCES of the boundary vectors, so a slice is the whole adaptation.
/// `shape` is the one field that is rewritten — `num_requests` is the window's
/// lanes, not the fire's.
#[derive(Clone, Copy, Debug)]
pub struct Planning<'a> {
    /// The window's slice of `GeomKind::Indptr`'s host contents.
    pub kv_indptr: &'a [i32],
    /// The window's slice of `GeomKind::KvLen`'s.
    pub kv_len: &'a [i32],
    /// The kv-side shape, at this window's request count.
    pub shape: Shape,
    /// The sliding window this schedule is carved for.
    pub window: Option<u32>,
    /// Where the built schedule's staged image lands.
    pub workspace: Workspace,
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

    /// `RuntimeInput::Mask`: this space's packed `u8` mask bits, for
    /// `attention.masked` — one `rows x (held + rows)` rectangle per masked
    /// lane, end to end, each starting on a byte boundary, with the causal
    /// bound already folded in ([`crate::mask`]). `None` for a fire no lane
    /// masked, which is what makes the entry's own refusal reachable.
    pub mask: Option<Tensor>,

    /// The host twin the plan builders walk, bound for spaces a plan op
    /// names.
    pub planning: Option<CachePlanning>,
}

/// The dsv4 compressor state `attention.pool_gather` reads beside its cache.
// MENLO-SEAM: no IR seat — the driver binds the slabs it staged for the
// pooled space (the marker at `kernels_cuda::attn::pool::gather`).
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
/// `MENLO-SEAM` marker in `kernels_cuda`, bound from fire state by the
/// arm that carries the matching comment. (The seats the IR reclaimed —
/// `row_valid`, `request_of_token`, the mask bits — resolve as declared
/// inputs on [`CacheGeometry`] now.)
#[derive(Clone, Copy, Debug)]
pub struct FireTables {
    /// `i32`, `[lanes + 1]`: the span table of the mask bits
    /// `attention.masked` names — each FIRE lane's byte offset into the slab
    /// [`CacheGeometry::mask`] holds — bound onto the prefill plan at build
    /// (`plan_prefill`'s `mask_indptr` seam). `None` when this fire carries
    /// no mask; a masked consumer then gets the entry's own typed refusal,
    /// not a panic — mask-lessness is a run-time fact, not a binding hole.
    ///
    /// **FIRE-WIDE HERE, WINDOW-SLICED AT THE SEAM.** The table is indexed by
    /// the SCHEDULE's request number, so the plan-building arm takes
    /// [`Run::mask_indptr`] — this vector cut to that node's lanes — and the
    /// byte offsets inside it stay ABSOLUTE, because the slab they address is
    /// handed to the launch whole. It is the shape `GeomKind::Indices` and
    /// its bounds vector already have, for the same reason: a table whose
    /// entries are not fire rows cannot be sliced by one.
    pub mask_indptr: Option<Tensor>,

    /// The dsv4 compressor slabs, bound when a pooled space exists. One
    /// fire-wide seat: a plan carries at most one pooled space today; this
    /// moves onto [`CacheGeometry`] the day one carries two.
    pub pool_state: Option<PoolSlabs>,
}

/// What the driver binds each fire, owned by the [`Run`] for its lifetime.
///
/// `tokens`, `positions`, and `geometry` are the op-visible inputs —
/// `RuntimeInput` routes onto them in [`Run::tensor`]. The rest is ambient:
/// the seam `tables`, the pre-probed `device` facts, the once-read
/// `toggles`, and the shell's `capture` policy word the builders take.
///
/// **THE QO BOUNDARIES ARE NOT HERE, AND THAT IS THE MIXED FIRE.** Design §5
/// removed `qo_indptr` as a named input, so ragged views assemble from an
/// ambient boundary vector — but a windowed consumer's boundaries are its
/// OWN, rebased to its sub-rectangle's zero, and one fire-wide vector would
/// be a lie in a fire whose lanes fall in more than one class. So the device
/// handle and the host twin the prefill/mla builders walk both live per
/// window ([`Window`](crate::window::Window)), reached through
/// [`Run::qo_indptr`] and [`Run::qo_indptr_host`].
///
/// **THE FACT WORD IS NOT HERE EITHER.** It used to ride along as one `u64`
/// for the whole fire, for a `Guard::holds` the walk stopped asking when
/// regions grew masks: which classes run a node is `Region::mask`, resolved
/// per region against the window table, and nothing on this side ever read
/// the word. A fire-wide word is exactly the collapse design §0's vocabulary
/// note warns about ("only the old execution contract collapsed the word to
/// per-fire"), so it is gone rather than windowed.
#[derive(Clone, Debug)]
pub struct FireBindings {
    /// `RuntimeInput::Tokens`: ragged `i32`, one id per token.
    pub tokens: Tensor,

    /// `RuntimeInput::Positions`: ragged `i32`, one absolute position per
    /// token.
    pub positions: Tensor,

    /// `RuntimeInput::AdapterRoutes`: `i32`, one adapter id per token row,
    /// `-1` for a row whose lane registered none.
    ///
    /// **HERE AND NOT IN [`FireTables`], BECAUSE AN OP NAMES IT.** Everything
    /// on `tables` is a seat no `Operands` impl mentions — the driver side of
    /// a `MENLO-SEAM` marker. This one is a declared `RuntimeInput` that
    /// `linear.lora_correct` lists among its inputs, so it stands beside
    /// `tokens` and `positions`, which are the other two fire-wide op-visible
    /// vectors keyed by nothing.
    ///
    /// `None` for a fire no lane carried an adapter into, and that absence is
    /// load-bearing: it is the same statement `mask` makes — nothing staged,
    /// no seat bound, and the axis costs the fire zero bytes and zero
    /// launches, because its window is empty and the walk skips it.
    pub adapter_routes: Option<Tensor>,

    /// Per cache space, aligned with `Trace::caches`:
    /// `RuntimeInput::Geometry { space, kind }` routes to that space, and
    /// the plan-building arms route to row `cache`'s planning twin.
    pub geometry: Vec<CacheGeometry>,

    /// Per (RUN, PLAN VALUE), flat at `run * plan_values + value`: the reading
    /// that schedule is carved for and the grant it stages into. `None` for
    /// every value that is not a plan struct some launch consumes.
    ///
    /// **THE RUN IS P4'S FALLBACK REACHING THE GRANT** (design §3). A region
    /// the layout could not seat carves one schedule per interval of its class
    /// set, every one of them built in the prepare pass and read in the
    /// capture pass, so every one of them needs its own staged image — see
    /// [`Inputs::plans`](crate::inputs) for which half of a grant is
    /// per-run and which is shared. For an artifact P4 seated whole this is
    /// one run wide and is the table it always was.
    pub schedules: Vec<Option<ScheduleSeat>>,
    /// How many plan values one run's slice of
    /// [`schedules`](FireBindings::schedules) holds.
    pub plan_values: usize,

    /// The seam extras the arms bind beside the ops' named operands.
    pub tables: FireTables,

    /// The device facts every builder takes — pre-probed by the shell
    /// (`Device::probe` once at boot, or a stated fallback); the builders
    /// themselves never probe, purity is their design.
    pub device: Device,

    /// The operator toggles `plan_decode` takes — resolved by the shell
    /// once ([`Toggles::from_env`], like `device`'s one probe) and carried
    /// here so no arm ever reads the environment per fire.
    pub toggles: Toggles,

    /// The shell's graph policy word: whether this fire's capture phase
    /// will be captured as a CUDA graph. Builders carve graph-shaped,
    /// padded schedules under it; `PrefillPlan::graph_capturable` answers
    /// whether they managed. Policy stays the shell's — this word only
    /// carries it to the builders.
    pub capture: bool,
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
/// constructs one per fire and drives the substrate's walk
/// (`driver::fire::walk`) over it — prepare phase first (outside any
/// capture), so every plan payload exists and is staged before its
/// consumers enqueue.
pub struct Run<'c> {
    /// The stream and its companions (cuBLAS handle, communicator, jit
    /// cache behind it). Everything this crate does to the device goes
    /// through it, enqueue only.
    ctx: &'c Ctx,

    /// The routing: `Trace::values`, read by [`Run::tensor`] to send each id
    /// to its table.
    values: &'c [ValueDecl],

    /// `Trace::nodes`, for the one thing a resolution cannot do from a value
    /// id alone: read a whole REGION's operands at once.
    ///
    /// A `Fallback::Copy` compacts every rectangle the region touches into
    /// one slab, so it has to enumerate them before the first node runs —
    /// which is a question about the region, not about any one op. Nothing
    /// else here reads it; the walk still hands each arm its own `&Node`.
    nodes: &'c [Node],

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
    /// **KEYED BY `(RUN, VALUE)`, NOT BY VALUE**, and the extra key is the
    /// split's other half. A schedule is carved for ONE window — how many
    /// requests it batches, where each one's query rows start, how its work
    /// items divide the kv — so a region P4 could not seat carves one per
    /// interval, and the prepare phase builds all of them before the capture
    /// phase reads any of them. One slot per value would let run 1's builder
    /// overwrite run 0's, and run 0's launch would then read a schedule
    /// describing run 1's requests: not a fault, just wrong logits for the
    /// lanes in the first interval.
    ///
    /// Flat rather than nested, at `run * values + value`: a plan has
    /// thousands of values and a `Vec` per value would be thousands of
    /// allocations per fire, where this is one. The width is
    /// [`Windows::max_runs`] — `1` for every artifact P4 seated whole, which
    /// is the layout this table had before the split existed.
    structs: Vec<Option<StructSlot>>,
    /// How many values one run's slice of [`structs`](Run::structs) holds.
    values_wide: usize,

    /// This fire's bindings.
    fire: FireBindings,

    /// Every region's window, resolved once per fire from the composition's
    /// class table.
    windows: &'c Windows,

    /// Which region the walk is inside and which run of its window, written
    /// by [`Cursor`](crate::window::Cursor) — the shell's `Sink` — before the
    /// region's nodes are dispatched and before each launch of them. **THIS
    /// IS THE WHOLE MIXED-FIRE MECHANISM**: it turns every resolution below
    /// from "the fire's rectangle" into "this node's window of it".
    place: &'c At,

    /// P6's side-stream contexts, in stream order: `side[0]` is stream 1.
    ///
    /// **EMPTY IS THE EAGER MODE AND IT IS NOT A DEGRADATION.** A `Run` built
    /// with no side contexts fires everything on the main one, which is the
    /// serialization of the compiler's dependency DAG — a legal schedule of it
    /// by construction, and the golden every recorded fire is diffed against.
    side: &'c [&'c Ctx],

    /// Which stream the walk is on, written by the same `Cursor` that writes
    /// [`place`](Run::place), at the same instant and for the same reason.
    stream: &'c Cell<u32>,

    /// The `Fallback::Copy` currently in force: which rectangles the region
    /// the walk is inside compacted, and where in the scratch slab each one
    /// landed.
    ///
    /// **WRITTEN BY `Serve::gather`, READ BY [`Run::cut`], AND IT CANNOT GO
    /// STALE.** The walk brackets a copied region's nodes with `gather` and
    /// `scatter`, so the plan is rebuilt immediately before the first node
    /// and is still the current one at the last; `CopyPlan::region` carries
    /// which region it was built for, and `cut` refuses a mismatch rather
    /// than reading another region's slab offsets. A region that is not
    /// copied never consults it — `Window::gathered` is `None` and the cut is
    /// the slice it always was.
    copy: CopyPlan,

    /// **D4'S TWO NUMBERS, AND THE WALK IS WHERE THEY ARE GATED**
    /// (`.wiki/palo/cuda-abi.md` §3, refined form): this fire's total rows and
    /// the lattice point above them. `Pad::default()` — a bucket that is not
    /// above the rows — is a shell that padded nothing, which is
    /// `PIE_CUDA_PAD=off`, a deployment with no lattice, and every `Run` built
    /// before this field existed.
    ///
    /// **HELD HERE RATHER THAN STAMPED ON THE CONTEXT ONCE PER FIRE**, because
    /// the question padding turns on is not "which fire" but "which REGION" —
    /// see [`Run::ctx`]. A pad the shell wrote onto the context at the top of
    /// the fire would still be there when the walk stepped into a windowed
    /// region, and the entry would then be left inferring from an extent
    /// whether its launch owns the fire's tail. An extent cannot answer that:
    /// a window's rows and a fire's rows are both `u32`, and they can be
    /// equal. The window can answer it, and the walk is holding one.
    pad: Pad,
}

impl<'c> Run<'c> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        ctx: &'c Ctx,
        values: &'c [ValueDecl],
        nodes: &'c [Node],
        weights: &'c WeightTable,
        arena: &'c SlotTable,
        caches: &'c CacheTable,
        fire: FireBindings,
        windows: &'c Windows,
        place: &'c At,
    ) -> Self {
        Self {
            ctx,
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
            side: &[],
            stream: &place.region,
            copy: CopyPlan::default(),
            pad: Pad::default(),
        }
    }

    /// The same `Run`, told this fire's row count and the bucket it rounds up
    /// to — D4's quantization, armed.
    ///
    /// ADDITIVE, AND THE SHELL CHOOSES: a `Run` never handed one pads nothing,
    /// which is the `PIE_CUDA_PAD=off` arm and every caller that predates the
    /// design. `pad.rows` must be the COMPOSITION's rows, not a window's —
    /// [`Run::ctx`] compares a window's span against it and the comparison is
    /// the whole safety argument.
    #[must_use]
    pub fn quantized(mut self, pad: Pad) -> Self {
        self.pad = pad;
        self
    }

    /// `Trace::values`, for the copy plan's own shape reading.
    pub(crate) fn values(&self) -> &'c [ValueDecl] {
        self.values
    }

    /// `Trace::nodes`, for the same.
    pub(crate) fn nodes(&self) -> &'c [Node] {
        self.nodes
    }

    /// Which region of the template the walk is inside.
    pub(crate) fn at_region(&self) -> u32 {
        self.place.region.get()
    }

    /// The FIRE-WIDE rectangle a value names, before any window is applied —
    /// what a copy's gather reads from and its scatter writes back to.
    pub(crate) fn uncut(&self, id: ValueId) -> Tensor {
        self.whole(id)
    }

    /// Take the copy plan the region's gather just built.
    pub(crate) fn seat_copy(&mut self, plan: CopyPlan) {
        self.copy = plan;
    }

    /// The same `Run`, told where P6's side streams are.
    ///
    /// ADDITIVE, AND THE CALLER CHOOSES PER PASS. `record.rs` builds one `Run`
    /// per fire and walks it twice — eagerly for the numbers, then capturing —
    /// and only the capturing walk is handed the streams, because a capture is
    /// where an event is a graph edge and an eager fire is where it would be a
    /// real synchronization bought for nothing.
    #[must_use]
    pub fn across(mut self, side: &'c [&'c Ctx], stream: &'c Cell<u32>) -> Self {
        self.side = side;
        self.stream = stream;
        self
    }

    /// The stream context the node being dispatched fires on, for the arms.
    ///
    /// **THE ONE PLACE A SIDE STREAM ENTERS THE DISPATCH PLANE**, and it is a
    /// lookup rather than a decision: which stream a region belongs on was
    /// decided once, at compile, by `model_compiler::stream`, and the `Cursor`
    /// wrote it into a cell before this region's first node. A stream the load
    /// never opened resolves back to the main one — the cursor has already
    /// stopped the walk with `Fault::Unbound` naming it, and answering a null
    /// handle here would turn that refusal into a launch.
    pub(crate) fn ctx(&self) -> &'c Ctx {
        // The eager reading, asked first and by the field that MEANS it: a
        // `Run` with no side contexts was never given a stream cell either,
        // and `new` seats the region cell there so the type has something to
        // hold. Reading a region index as a stream index is exactly the bug
        // this line exists to make impossible.
        let ctx = if self.side.is_empty() {
            self.ctx
        } else {
            match self.stream.get() {
                0 => self.ctx,
                n => self.side.get(n as usize - 1).copied().unwrap_or(self.ctx),
            }
        };
        // **AND THE SECOND THING THIS LOOKUP ANSWERS: MAY THIS REGION PAD?**
        // (`.wiki/palo/cuda-abi.md` §3's boundary.) Every dispatch arm reaches
        // its context through here and does so with the cursor already on the
        // node it is about to launch, so this is the last instant at which the
        // WINDOW is still in hand and the first at which an entry could ask.
        // Deciding it here rather than in the entry is not tidiness: an entry
        // sees an extent, and an extent cannot distinguish the fire's rows
        // from a window that happens to hold as many.
        ctx.arm(self.here());
        ctx
    }

    /// **THE PAD THIS REGION IS ALLOWED**, and the argument for every clause.
    ///
    /// D4 pads an opaque callee's `M` up to the bucket so that cuBLASLt's
    /// unpublished shape→kernel table stops following the batch. What that
    /// buys is bought with the rows `[rows, bucket)`, and those rows are safe
    /// to scribble on exactly when they are the FIRE's tail — reserved by the
    /// arena at `max_tokens`, promised by P0 to be above every bucket, and
    /// spoken for by nobody. A WINDOWED launch's tail is a different thing
    /// entirely: the rows above a window are the next class's rows of the same
    /// column, and under `ArenaMap::co_tenants` or a merge those are somebody's
    /// real bytes. Padding one is a clobber that computes and never faults.
    ///
    /// So a region pads only if its window IS the whole fire:
    ///
    /// * **`row_offset == 0` and `rows >= the fire's`** — the mask covers every
    ///   class that has rows, read off the window the shell built FROM the
    ///   mask (`Windows::of`). This is the clause the entry could not make: it
    ///   compares a window's span against the composition, where an entry can
    ///   only compare one extent against another.
    /// * **not gathered** — a `Fallback::Copy` compacted its rows into a
    ///   scratch slab cut to the rows it gathered, so past them is the next
    ///   thing in the slab rather than a reserved tail. (A gathered window
    ///   cannot span the whole fire anyway — gathering needs two intervals and
    ///   two intervals need a gap — but the clause is written because that is
    ///   an argument, and this is a bounds check.)
    /// * **no segment list** — a `Fallback::Grouped` window's span is the UNION
    ///   of its intervals and the gaps hold foreign rows, so its span covering
    ///   the fire says nothing about the rows it owns.
    ///
    /// A region that fails any clause gets `Pad::default()`, which
    /// [`Ctx::opaque_rows`] answers with the extent it was handed.
    fn here(&self) -> Pad {
        if self.pad.bucket <= self.pad.rows {
            return Pad::default();
        }
        let window = self.window();
        let whole = window.span.row_offset == 0
            && window.span.rows >= self.pad.rows
            && window.gathered.is_none()
            && window.segs() == 0;
        if whole { self.pad } else { Pad::default() }
    }

    /// The fire bindings, for the plan-building arms' seam.
    pub(crate) fn bindings(&self) -> &FireBindings {
        &self.fire
    }

    /// The window the node being dispatched runs over — this region's, cut at
    /// the run the walk is on.
    pub(crate) fn window(&self) -> &'c Window {
        self.windows.at(self.place.region.get(), self.place.run.get())
    }

    /// **THE ROWS OF THIS WINDOW THAT ARE ACTUALLY THE NODE'S**, for a region
    /// P4 answered `Fallback::Grouped` for — and `None` for every other
    /// region, which is every region of every artifact P4 seated whole.
    ///
    /// A grouped window's span is the UNION of the intervals its mask covers,
    /// so [`Run::cut`] hands the arm a rectangle with foreign rows standing in
    /// the gaps; this is the list that keeps the launch off them. An arm that
    /// takes it MUST honour it — reaching a grouped window without asking is
    /// running the correction over rows whose lanes never asked for one — and
    /// an arm whose kernel cannot honour it must never have been named in
    /// `DeviceProfile::grouped`, because that is the word that made the
    /// compiler write the row.
    pub(crate) fn segments(&self) -> Option<Segments> {
        let window = self.window();
        let count = window.segs();
        if count == 0 {
            return None;
        }
        Some(Segments {
            list: window.segments,
            count,
            cap: window.segment_cap,
            max_rows: window.segment_rows(),
        })
    }

    /// Where this run's payload for `id` sits in [`structs`](Run::structs).
    ///
    /// The run comes off the same cell the window does, so a schedule is
    /// stored and read at the same key by construction — a builder cannot
    /// carve for one interval and a launch read another.
    fn struct_at(&self, id: ValueId) -> usize {
        self.place.run.get() as usize * self.values_wide + id.0 as usize
    }

    /// This window's qo boundaries, staged — what a ragged view is cut by.
    pub(crate) fn qo_indptr(&self) -> Tensor {
        self.window().indptr
    }

    /// Their host twin, for the prefill and mla builders that walk the
    /// contents. Rebased: entry 0 is 0, because the rectangle they bound is
    /// this window's, not the fire's.
    pub(crate) fn qo_indptr_host(&self) -> &'c [i32] {
        &self.window().indptr_host
    }

    /// How many token rows this window carries — the `total_num_rows` the
    /// prefill builders take.
    pub(crate) fn total_tokens(&self) -> u32 {
        self.window().span.rows
    }

    /// The mask span table this window's schedule should carry, or `None`
    /// for a fire no lane masked.
    ///
    /// **SLICED BY LANE, ABSOLUTE IN VALUE** — the opposite of the qo
    /// boundaries beside it, and the difference is what each one bounds. A
    /// window's qo indptr cuts the window's OWN rectangle, so it is rebased;
    /// this one names byte offsets into a fire-wide slab the consumer takes
    /// whole, so rebasing it would send request 0 of a later window to the
    /// first lane's bits. Same shape as `GeomKind::Indices` and its bounds.
    ///
    /// `[lanes + 1]` entries, because the schedule's last request needs an
    /// upper bound as much as the ones before it.
    pub(crate) fn mask_indptr(&self) -> Option<Tensor> {
        let span = self.window().span;
        self.fire
            .tables
            .mask_indptr
            .map(|table| skip(table, span.lane_offset, span.lanes + 1))
    }

    /// Whether any lane of THIS WINDOW carries more than one token — the mla
    /// builder's `causal` word, derived rather than seated: multi-token lanes
    /// attend causally within themselves, single-token (decode) lanes have
    /// nothing to order.
    pub(crate) fn multi_token(&self) -> bool {
        self.qo_indptr_host()
            .windows(2)
            .any(|span| span[1] - span[0] > 1)
    }

    /// One value's rectangle, cut to the window of the node asking for it.
    ///
    /// **EVERY ROW-SHAPED TABLE IN THIS SHELL IS INDEXED BY ABSOLUTE FIRE
    /// ROW** — the arena carve gives a `Dim::Tokens` value one column at the
    /// fire's row count, the geometry vectors one entry per fire lane — so a
    /// window is a slice, and which slice is read off the value's own leading
    /// `Dim`. A `Dim::Const` column (a weight plane, a bias) is not fire
    /// -aligned and is handed over whole.
    ///
    /// `GeomKind::Indices` is the one declared shape that is not what it
    /// says: the IR spells the flat page-id list `Dim::Lanes` because it has
    /// no page symbol, and its entries are pages rather than lanes. Slicing it
    /// by a lane offset would hand a windowed consumer somebody else's pages,
    /// so it is excluded here — and its bounds vector stays absolute, which is
    /// exactly what makes a sliced `Indptr` still address the whole list.
    ///
    /// `RuntimeInput::Mask` is the SECOND of those, for the same reason and
    /// with the same remedy. The IR spells the custom-mask slab `Dim::Tokens`
    /// because it has no bit symbol either, and its entries are (query, key)
    /// BITS: one lane of the fire occupies `rows x (held + rows)` of them, so
    /// a row offset is not a byte offset and a slice would land mid-lane.
    /// The slab goes over whole and `FireTables::mask_indptr` — absolute byte
    /// offsets, sliced by lane — is what puts a windowed launch on its own
    /// rectangle ([`crate::mask`] argues both halves).
    fn cut(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
        // **THE OTHER ANSWER, ASKED FIRST** (design §3). A gathered window's
        // rows do not lie in the arena at all — they were compacted into a
        // scratch slab before the region's first node — so a slice of the
        // fire-wide column is not a narrower reading of the same bytes, it is
        // the wrong bytes. `compacted` is the whole of what a copy changes
        // about resolution.
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
        let window = self.window().span;
        let (skip, keep) = match shape.first() {
            Some(Dim::Tokens) => (window.row_offset, window.rows),
            Some(Dim::TokensTimes(k)) => (window.row_offset * k, window.rows * k),
            Some(Dim::Lanes) => (window.lane_offset, window.lanes),
            Some(Dim::LanesPlus(k)) => (window.lane_offset, window.lanes + k),
            Some(Dim::Const(_)) | None => return handle,
        };
        if skip == 0 && keep >= handle.rows {
            return handle;
        }
        let stride = u64::from(handle.width)
            * model_compiler::arena::elem_bytes(handle.dtype).unwrap_or_else(|| {
                panic!(
                    "value {at} is stored as {:?}, which has no element size and so no \
                     row to step by",
                    handle.dtype
                )
            });
        Tensor::new(
            handle.ptr + u64::from(skip) * stride,
            keep.min(handle.rows.saturating_sub(skip)),
            handle.width,
            handle.dtype,
        )
    }

    /// [`Run::cut`]'s other half: what a `Fallback::Copy` resolves to.
    ///
    /// **THREE ANSWERS, AND THEY ARE THE THREE `window::copyable` ADMITS.**
    /// A row-shaped value is the slab rectangle the region's gather laid it
    /// in; the four kv geometry vectors are the twins re-cut for the gathered
    /// lanes ([`GatheredSpace`](crate::window::GatheredSpace)); everything
    /// window-free is handed over whole, exactly as a split hands it over.
    /// Nothing else can arrive — `Windows::of` declines to gather a region
    /// naming anything else, and the region then takes the split, which is
    /// always correct.
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
                // `window::copyable` admits no other kind into a copied
                // region, so this is the arm nothing reaches — and it
                // answers the fire-wide vector rather than a wrong window,
                // which is the conservative direction.
                _ => handle,
            };
        }
        let Ty::Tensor { shape, .. } = &self.values[at].ty else {
            return handle;
        };
        match shape.first() {
            Some(Dim::Tokens | Dim::TokensTimes(_)) => {
                assert_eq!(
                    self.copy.region,
                    self.place.region.get(),
                    "value {at} is being resolved inside a copied region whose gather \
                     has not run; `driver::fire::walk` brackets a copied region's \
                     nodes and this is what says the bracket was lost",
                );
                self.copy.tight(handle.ptr).unwrap_or_else(|| {
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

    /// The crate's heart: one plan id in, one device handle out, routed on
    /// the id's `Def`, cut to the asking node's window ([`Run::cut`]). Every
    /// dispatch arm resolves through here, so provenance handling — and
    /// windowing — exists exactly once.
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
            Def::Input(RuntimeInput::Mask { space }) => {
                let seat = self.geometry(at, *space);
                seat.mask.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads the mask bits of cache space {space}, which \
                         this fire left unbound"
                    )
                })
            }
            // THE ADAPTER AXIS'S ONE RUNTIME INPUT (design §8). Bound only
            // when a lane of this fire carried an adapter — a fire none did
            // stages nothing, and nothing can reach this arm either, because
            // the correction's window is empty and the walk skips a zero-row
            // region before it dispatches a node. So the panic is not a hole:
            // it is the same "unbound seat" statement the mask makes one arm
            // up, and reaching it means a word said `has_adapter` where the
            // submission said no adapter — which `Fault::AdapterWord` refuses
            // before anything launches.
            Def::Input(RuntimeInput::AdapterRoutes) => {
                self.fire.adapter_routes.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's adapter ids, which no lane of it                          carried"
                    )
                })
            }
            Def::Input(RuntimeInput::Geometry { space, kind }) => {
                let seat = self.geometry(at, *space);
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

    /// A fire-aligned value viewed through THIS WINDOW's boundaries. The
    /// indptr is ambient (design §5): no op names it, and this pairing is
    /// where it re-enters.
    ///
    /// The boundaries are the window's own, rebased: `data` already points at
    /// the window's first row, so a fire-wide vector would send every ragged
    /// entry past the end of the rectangle it was handed by exactly the number
    /// of rows the classes before it occupy.
    pub(crate) fn ragged(&self, id: ValueId) -> RaggedTensor {
        RaggedTensor {
            data: self.tensor(id),
            indptr: self.qo_indptr(),
        }
    }

    /// The `(codes, scales)` planes of a split-plane bank — the resolution
    /// `linear.moe_matmul_select_bias` needs where [`Run::tensor`] would
    /// have to lie with one handle.
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

    /// The paged kv pool a cache id names, with its LANE-INDEXED tables cut
    /// to the asking node's window.
    ///
    /// The storage is the whole pool — pages are the model's state and outlive
    /// every fire — but the tables that address it are this fire's: the page
    /// bounds and last-page fills are one entry per lane, and the padding mask
    /// one per row. A windowed attention launches request `r` in `0..lanes` of
    /// ITS window, so those three are sliced and the page-id list they index
    /// is not (its bounds stay absolute).
    pub(crate) fn pool(&self, id: ValueId) -> KvPool {
        match self.cache(id) {
            CachePool::Kv { space, pool } => {
                let window = self.window().span;
                // **A GATHERED WINDOW'S TABLES ARE RE-CUT, NOT SLICED**, and
                // the page-id list is the reason: gathered lanes own spans of
                // it with other lanes' pages standing between them, and no
                // `[lanes + 1]` bounds vector over the whole list can name
                // two such spans as requests 0 and 1
                // (`window::GatheredSpace`). So the list is compacted with
                // the lanes and the bounds are a fresh prefix sum over it.
                //
                // `row_valid` is the one table a gather does NOT need: its
                // entries are 1 for a row this fire means and 0 for a
                // bucket's padding, every gathered row is one the fire
                // means, and the count is the gathered count — so the
                // window's own prefix of the fire-wide vector holds exactly
                // the values a gathered one would.
                if let Some(gathered) = &self.window().gathered {
                    let seat = gathered.spaces.get(*space as usize);
                    return KvPool {
                        page_indptr: seat.map_or(pool.page_indptr, |seat| seat.page_indptr),
                        page_indices: seat.map_or(pool.page_indices, |seat| seat.page_indices),
                        last_page_lens: seat.map_or(pool.last_page_lens, |seat| seat.last_page_lens),
                        row_valid: skip(pool.row_valid, 0, window.rows),
                        ..*pool
                    };
                }
                KvPool {
                    page_indptr: skip(pool.page_indptr, window.lane_offset, window.lanes + 1),
                    last_page_lens: skip(pool.last_page_lens, window.lane_offset, window.lanes),
                    row_valid: skip(pool.row_valid, window.row_offset, window.rows),
                    ..*pool
                }
            }
            CachePool::Recurrent(_) => panic!(
                "value {} is a recurrent state space, and this op walks a paged kv pool",
                id.0
            ),
        }
    }

    /// The recurrent state pool a cache id names, with its slot map cut to the
    /// asking node's window.
    ///
    /// A recurrent bank is addressed by SLOT and the scan reads its slot from
    /// `slot_ids[lane]` — where `lane` counts from the launch's own zero. So a
    /// windowed scan gets the window's lanes and nothing else; the slabs
    /// themselves are the model's state, whole.
    pub(crate) fn recurrent(&self, id: ValueId) -> RecurrentPool {
        match self.cache(id) {
            CachePool::Recurrent(pool) => {
                let window = self.window().span;
                RecurrentPool {
                    slot_ids: skip(pool.slot_ids, window.lane_offset, window.lanes),
                    ..*pool
                }
            }
            CachePool::Kv { .. } => panic!(
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

    fn geometry(&self, at: usize, space: u32) -> &CacheGeometry {
        let space = space as usize;
        self.fire.geometry.get(space).unwrap_or_else(|| {
            panic!(
                "value {at} names cache space {space}, and this fire binds {} geometry spaces",
                self.fire.geometry.len()
            )
        })
    }

    /// Everything a plan op's builder takes, in one place: the host geometry
    /// twins of the cache space its `kv_indptr` names, cut to the op's own
    /// window, beside the reading and the grant of the SCHEDULE it defines.
    ///
    /// Two keys, because they are two facts. `kv_indptr` is a device value
    /// whose `Def` says which space it is, and the space's
    /// [`CachePlanning`] holds the twin the builders actually walk — the
    /// duality, routed in one place and windowed in the same one. `plan` is
    /// the struct value the op DEFINES, and its [`ScheduleSeat`] holds the
    /// head width, the query heads and the window this carving is for; a
    /// family that reads its one page-id space at two of those mints two plan
    /// values (build log 20's first blocker).
    pub(crate) fn planning(&self, geom: ValueId, plan: ValueId) -> Planning<'_> {
        let at = geom.0 as usize;
        let Def::Input(RuntimeInput::Geometry { space, .. }) = &self.values[at].def else {
            panic!(
                "value {at} is not declared cache geometry, and a plan op routes to its \
                 cache space through its geometry input"
            );
        };
        let seat = self.geometry(at, *space);
        let seat = seat.planning.as_ref().unwrap_or_else(|| {
            panic!(
                "cache space {space} carries no planning seat; the shell binds the host \
                 geometry twins before a plan op can fire"
            )
        });
        let run = self.place.run.get() as usize;
        let schedule = self
            .fire
            .schedules
            .get(run * self.fire.plan_values + plan.0 as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| {
                panic!(
                    "plan value {} carries no schedule seat for run {run} of its \
                     window; the shell reads every schedule's reading off the plan op \
                     that defines it and carves one grant per run of the region that \
                     builds it, so a plan op firing without one is a value \
                     `store::kv::probe` never walked",
                    plan.0
                )
            });
        let window = self.window();
        let span = window.span;
        // **A GATHERED WINDOW'S TWINS ARE THE ONES RE-CUT WITH ITS LANES.**
        // The comment on `Planning` says a slice is the whole adaptation
        // because "the window is contiguous in lanes"; a copy's is not, so
        // the builder gets the union's vectors — the page bounds as a fresh
        // prefix sum over the compacted page-id list, the kv lengths lane by
        // lane in gathered order. `num_requests` is the union's lane count,
        // which is what makes the ONE schedule cover every run.
        let (kv_indptr, kv_len) = match window.gathered.as_ref().and_then(|g| g.spaces.get(*space as usize)) {
            Some(gathered) => (
                gathered.page_indptr_host.as_slice(),
                gathered.kv_len_host.as_slice(),
            ),
            None => {
                let first = span.lane_offset as usize;
                let lanes = span.lanes as usize;
                (
                    seat.kv_indptr
                        .get(first..=first + lanes)
                        .unwrap_or(&seat.kv_indptr),
                    seat.kv_len
                        .get(first..first + lanes)
                        .unwrap_or(&seat.kv_len),
                )
            }
        };
        Planning {
            kv_indptr,
            kv_len,
            shape: Shape {
                num_requests: span.lanes,
                ..schedule.shape
            },
            window: schedule.window,
            workspace: schedule.workspace,
        }
    }

    /// The `StructKind` a plan op's output value declares — how the
    /// prefill-building arm tells fa2 from sm90: the trace wrote the choice
    /// into `Trace::values`, the arm only follows it.
    pub(crate) fn declared(&self, id: ValueId) -> StructKind {
        match &self.values[id.0 as usize].ty {
            Ty::Struct(kind) => *kind,
            Ty::Tensor { .. } => panic!(
                "value {} declares a tensor, and a plan op defines a struct",
                id.0
            ),
        }
    }

    /// The dsv4 compressor slabs, for `attention.pool_gather`'s seam.
    pub(crate) fn slabs(&self) -> PoolSlabs {
        self.fire.tables.pool_state.unwrap_or_else(|| {
            panic!(
                "this fire binds no dsv4 compressor slabs, which `attention.pool_gather` reads beside the pool"
            )
        })
    }

    /// A hash of the SHAPE of every plan payload this fire built — every
    /// number off a plan struct that can reach a kernel argument, and none of
    /// the CONTENTS that reach the device through the workspace.
    ///
    /// **THE ONE THING A GRAPH KEY CANNOT SEE.** A recorded fire bakes the
    /// plan's offsets, its padded batch size and its tile width into the
    /// launches it recorded; the prepare phase rebuilds the plan every fire
    /// and the replay keeps reading the captured numbers. Under
    /// [`FireBindings::capture`] the builders carve graph-shaped schedules, so
    /// those numbers are a function of the fire's shape and the key holds them
    /// fixed — but that is a property of somebody else's arithmetic, and this
    /// is the fire path checking it rather than believing it. A disagreement
    /// is `Fault::Schedule`, not a slightly wrong logit.
    ///
    /// The Debug image is the hashed form on purpose: it covers every field
    /// the plan structs have TODAY and every field they grow, where a
    /// hand-listed hash would silently stop covering the one that was added.
    /// It allocates nothing — the formatter writes straight into the hasher.
    pub(crate) fn schedule_shape(&self) -> u64 {
        use core::fmt::Write;
        use std::hash::{DefaultHasher, Hasher};

        struct Sink(DefaultHasher);
        impl Write for Sink {
            fn write_str(&mut self, text: &str) -> core::fmt::Result {
                self.0.write(text.as_bytes());
                Ok(())
            }
        }

        let mut sink = Sink(DefaultHasher::new());
        for (at, slot) in self.structs.iter().enumerate() {
            let Some(slot) = slot else { continue };
            let _ = write!(sink, "{at}:");
            // `int_upload` is the one field deliberately left out: it is this
            // fire's schedule CONTENTS, staged into a pointer-stable
            // workspace, and it is supposed to differ every fire.
            let _ = match slot {
                StructSlot::Decode(p) => write!(
                    sink,
                    "d{:?}{:?}{:?}{:?}",
                    p.info, p.workspace, p.shape, p.window
                ),
                StructSlot::Prefill(p) => write!(
                    sink,
                    "p{:?}{:?}{:?}{:?}{}{}{}",
                    p.info, p.workspace, p.shape, p.window, p.total_tokens, p.causal,
                    p.graph_capturable
                ),
                StructSlot::PrefillSm90(p) => write!(
                    sink,
                    "s{:?}{:?}{:?}{}{}",
                    p.info, p.workspace, p.shape, p.total_tokens, p.causal
                ),
                StructSlot::Mla(p) => write!(sink, "m{:?}{:?}{}", p.info, p.workspace, p.num_heads),
            };
        }
        sink.0.finish()
    }

    /// Did every schedule this fire built keep its graph-shaped padding?
    ///
    /// The builders' answer OUT (`PrefillPlan::graph_capturable`): a
    /// graph-shaped prefill schedule that did not fit its workspace grant
    /// falls back to one that fits and is not capturable, and capturing that
    /// would bake this fire's row count into a graph the next fire replays at
    /// another. The shell reads this before it captures, and stays eager when
    /// it is false.
    pub(crate) fn capturable(&self) -> bool {
        self.structs.iter().flatten().all(|slot| match slot {
            StructSlot::Prefill(plan) => plan.graph_capturable,
            _ => true,
        })
    }

    /// Store a plan payload a prepare-phase arm just built.
    pub(crate) fn put(&mut self, id: ValueId, built: StructSlot) {
        let at = self.struct_at(id);
        self.structs[at] = Some(built);
    }

    /// One built slot, whichever kind it holds — for the arm that routes on
    /// the kind (prefill's fa2/sm90 fork); the typed accessors below are the
    /// single-kind reads.
    pub(crate) fn slot(&self, id: ValueId) -> &StructSlot {
        let at = self.struct_at(id);
        self.structs[at].as_ref().unwrap_or_else(|| {
            panic!(
                "value {} holds no plan payload for run {} of its window; its plan \
                 op has not fired, and the prepare phase runs first",
                id.0,
                self.place.run.get(),
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

/// A row-indexed handle, advanced past `skip` rows and cut to `keep` of them.
///
/// The one arithmetic every windowed table shares: a pointer plus an extent,
/// which is exactly what design §0 says a windowed kernel takes.
fn skip(handle: Tensor, skip: u32, keep: u32) -> Tensor {
    if skip == 0 && keep >= handle.rows {
        return handle;
    }
    let stride = u64::from(handle.width)
        * model_compiler::arena::elem_bytes(handle.dtype).unwrap_or_else(|| {
            panic!(
                "a {:?} table has no element size and so no row to step by",
                handle.dtype
            )
        });
    Tensor::new(
        handle.ptr + u64::from(skip) * stride,
        keep.min(handle.rows.saturating_sub(skip)),
        handle.width,
        handle.dtype,
    )
}
