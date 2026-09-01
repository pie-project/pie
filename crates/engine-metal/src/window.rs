//! The fire's windows: which rows and which lanes each region of the baked
//! template actually runs over, and the cursor that tells a [`Run`] which one
//! it is inside.
//!
//! **THIS IS DESIGN §0's DIAGRAM, RESOLVED.** A fire seriates its lanes by
//! class, so a node guarded on `qo_one` stands over one interval of rows and a
//! node guarded on `¬qo_one` over the interval beside it; the shared nodes
//! stand over both. Every table a [`Run`] resolves through is indexed by
//! ABSOLUTE fire row (or absolute fire lane) — the arena carve gives one
//! column per value at `Dim::Tokens` rows, and design §0's merge is exactly
//! "the arms write disjoint row ranges of it" — so a node's operands are that
//! node's window's SLICE of those columns, and nothing has to be re-carved.
//!
//! ```text
//! fire:            [ prefill lane 0 : 7 rows | prefill lane 1 : 3 | decode l2 ]
//! arena column x:  [·············· 11 rows, one rectangle ··················]
//! embed/norm/qkv    ─────────── window (0,11) lanes (0,3) ────────────
//! attention.prefill ──── window (0,10) lanes (0,2) ────┐
//! attention.decode                                     └── (10,1) (2,1) ──
//! ```
//!
//! # Why per REGION and not per value
//!
//! The obvious reading — give every value its own span, from the classes its
//! defining node runs in — is not enough, and the reason is in the IR. A
//! `split` does not mint a value: `Value::split` REFINES a value's cond and
//! hands back the same `ValueId` (`model_dsl::record`), so the `q` a decode op
//! reads and the `q` a prefill op reads are one id with one rectangle. The
//! window belongs to the READER, not to the value: the decode node takes rows
//! `[10,11)` of `q` and the prefill node takes `[0,10)` of the same `q`, and
//! only the node knows which. So the resolution is `value column ∩ this
//! node's window`, and the node's window is its region's — P2 coalesces
//! exactly the nodes whose class mask is equal, which is to say exactly the
//! nodes that share a window.
//!
//! Values still land where a per-value reading would put them, because the
//! two agree wherever the value has one reader: an arm of a merge is
//! written by its own guarded node and therefore over its own window, and the
//! merge column it is aliased onto is read by an unguarded consumer over the
//! union — which is the whole fire. That is design §0's zero-instruction φ,
//! and it falls out rather than being arranged.
//!
//! # When a region's window is NOT one interval
//!
//! P4 makes as many windowed consumers consecutive as one row order can, and
//! writes a `Fallback` row for each one it could not (design §3). A region
//! with such a row covers SEVERAL row intervals in a fire that carries the
//! classes between its own, and the answer this shell serves is
//! `Fallback::Split { r }`: the region holds `r` windows rather than one, the
//! walk encodes its nodes once per window, and each encode takes its own
//! offset, its own extent and — the part that is easy to get silently wrong —
//! its own rebased qo boundaries. A ragged view's `indptr` is offsets INTO
//! the rectangle it cuts, so the second run's must start at 0 again over the
//! second run's lanes; sharing the first run's would hand the encode a vector
//! that describes somebody else's requests.
//!
//! [`Fault::Fragmented`] survives, narrowed to what it always meant: a
//! fragmented window the artifact owes NO fallback row for, which is P4
//! having promised this mask consecutive and the fire finding it broken.
//!
//! # And when the table asks for a COPY instead
//!
//! `Fallback::Split` is what the menu writes ABOVE the copy/split crossover
//! (`model_compiler::layout`'s `CROSSOVER_ROWS`: at 64 rows a two-way split
//! measured 1.82x the ideal against a copy's 1.07x, and they converge by
//! 2048). Below it the table asks for `Fallback::Copy`, and on a fourteen-
//! point lattice that is ten of the fourteen buckets — every bucket a decode
//! fire lands in.
//!
//! A copy is the same window read as ONE rectangle. The runs are gathered
//! into a staging rectangle, the region's nodes run once over it, and the
//! answers are scattered back to the fire rows they came from
//! (`kernels_metal::layout::{gather_rows, scatter_rows}`). [`Gathered`] is
//! what such a window carries beyond a split's: the row map the two entries
//! read, the two ambient row tables re-laid under it, and — because a paged
//! consumer names its geometry by LANE and the gathered lanes are not
//! contiguous either — the pool tables re-cut for the union. Those are small,
//! host-computable and written beside the boundary vectors in the same packed
//! blob; only the activations move on the device, which is the whole reason a
//! copy is cheap.
//!
//! ```text
//! classes in fire order:  [ 4 : 3 rows | 0 : 5 rows | 5 : 2 rows ]
//! mask {4,5,6,7}:          ──run 0──                 ──run 1──
//! split:                   encode over [0,3)         encode over [8,10)
//! copy:                    gather rows 0 1 2 8 9  ->  ONE encode over [0,5)
//!                          scatter back to 0 1 2 8 9
//! ```
//!
//! **THERE IS NO FORK AND NO JOIN AROUND IT.** The CUDA sibling's copy rides
//! the region's own stream and is ordered against its producers by the same
//! events the region is; this shell has one serial compute pass, so the
//! gather, the region's nodes and the scatter are three points of it and the
//! ENCODER ORDER is the ordering (`crate::dispatch::copy`).
//!
//! **THE BUILDER TAKES THE SAME ANSWER AS ITS READERS.** An attention plan is
//! carried for one window, and a consumer standing over the union of two runs
//! must read a plan carried over that union — so the prepare region that
//! builds it is copied whenever its readers are, even though P4 owes it no
//! row of its own (`model_exec::fire::fallback::copies` argues why the
//! question is asked of the MASK). The two masks being equal is checked at
//! load, by name, in [`no_schedule_straddles_its_readers`].
//!
//! # A cut is a HANDLE here, not an address
//!
//! The one place this file's arithmetic meets the plane it runs on. A
//! `kernels_cuda::Tensor` carries a device address, so the sibling shell
//! seats a window's boundary vector by writing `base + offset` into a
//! `Tensor` and is done — the number IS the location. Metal binds a BUFFER
//! and an OFFSET, so [`Windows::bind`] mints a row in
//! [`Handles`](crate::device::Handles) per window and seats the row's index.
//! Same vector, same rebasing, same one staged copy; what changes is that
//! seating a window can now FAIL (a boundary vector that would leave the
//! reservation is refused where it is minted, not where a shader reads it),
//! which is why `bind` answers a `Result` and its twin answers nothing.
//!
//! # How a `Run` learns which region it is in
//!
//! `Dispatch::exec` takes a `&Node` and the walk's signature is fixed
//! (decision #11: one walk, generic over `Dispatch` × `Sink`). But the walk
//! announces every region to the SINK, in order, before dispatching its nodes
//! — and the sink is the shell's. So [`Cursor`] is this shell's `Sink`: it
//! counts regions into an [`At`] the `Run` also holds a shared reference to,
//! and writes the run index beside it for the same reason and at the same
//! instant. No signature moves, and the state involved is two `u32`s.
//!
//! [`Run`]: crate::run::Run

use std::cell::Cell;

use kernels_metal::Tensor;
use model_compiler::{CompiledModel, Lowering, Region};
use model_exec::fire::{EventId, MaskSpan, Sink, WindowTable, fallback};
use model_exec::store::check::{self, rebase};
use model_exec::store::kv::Geometry;
use model_ir::{Def, Dim, Dtype, GeomKind, Operands, Operation, RuntimeInput, Trace, Ty};

use crate::device::handles::NIL;
use crate::device::Handles;
use crate::error::{Fault, Result};

/// One window, and the qo boundaries that go with it.
///
/// The span is the arithmetic (rows and lanes, both, because the IR has both
/// symbols); the two indptrs are the one thing a window cannot slice, because
/// a ragged view's boundaries are OFFSETS INTO the rectangle they cut and a
/// sub-rectangle starts at zero. So each window carries its own rebased copy —
/// `[lanes + 1]` entries, the first of them 0 — device-side for the launches
/// and host-side for the plan builders that walk the contents (the duality
/// [`CacheGeometry`](crate::run::CacheGeometry)'s seats and their host twins
/// state per cache space).
#[derive(Debug, Clone)]
pub struct Window {
    /// The rows and lanes this window covers, in fire coordinates.
    pub span: MaskSpan,
    /// `[lanes + 1]`: the window's qo boundaries, rebased to start at 0.
    pub indptr_host: Vec<i32>,
    /// The same vector, staged. Carries [`NIL`] — the absent handle, never
    /// the first row — until [`Windows::bind`] has minted its view; on the
    /// CUDA plane the same seat holds a null ADDRESS, and 0 is a perfectly
    /// good handle here.
    pub indptr: Tensor,
    /// Present iff this window is a [`Fallback::Copy`](model_compiler::Fallback)
    /// — the runs it compacts, and everything a consumer needs to read them
    /// as one.
    ///
    /// **AND WHEN IT IS PRESENT, [`span`](Window::span) IS THE COMPACTED
    /// RECTANGLE** and not a fire interval: `row_offset` and `lane_offset`
    /// are 0 and the counts are the union's. That is the right reading and
    /// not a fudge — every consumer of a gathered window reads the staging
    /// rectangle, whose rows start at its own zero, and the map back to fire
    /// coordinates is [`Gathered::rows_host`], which is where it belongs.
    pub gathered: Option<Gathered>,
    /// **THE SAME MASK'S INTERVAL ON THE SECOND ROW AXIS** — patch rows where
    /// [`span`](Window::span) has token rows, and IMAGES where it has lanes
    /// (multimodal §5.1).
    ///
    /// **CARRIED ON EVERY WINDOW RATHER THAN CHOSEN BETWEEN**, which is the
    /// CUDA sibling's own line and holds here for the same reason: the embed
    /// merge is a TOKEN region that reads a patch column, so one resolution
    /// needs both pairs. A tower region's own rows come out of the patch
    /// table and land in `span` (its capture unit's axis is
    /// [`RowAxis::Patches`](model_ir::RowAxis::Patches)); `patch` then holds
    /// the same interval, which is what makes `span == patch` the tower's
    /// signature and their disagreement the trunk's.
    ///
    /// All-zero for every window of a fire whose lanes carried no image, and
    /// that is the property the text-lane invariance gate rests on: a
    /// patch-axis rectangle cut at `(0, 0)` has no rows, and the walk skips a
    /// zero-row region before it dispatches a node.
    pub patch: MaskSpan,
}

/// A `Fallback::Copy`'s window: which fire rows the compacted rectangle is
/// made of, the ambient row tables re-laid in that order, and the per-space
/// pool tables the gathered LANES address by.
///
/// **TWO GATHERS, AND ONLY ONE OF THEM IS ON THE DEVICE.** The activations
/// are big rectangles and move through
/// [`gather_rows`](kernels_metal::layout::gather_rows). Everything else a
/// windowed attention reads is a handful of `i32` per row or per lane —
/// which lane owns a row, where its pages are, how full the last one is —
/// and those are recomputed on the host for the union and written into the
/// same packed blob as the boundary vectors ([`crate::inputs`]). A device
/// gather of a per-lane vector would be three dispatches to move forty
/// bytes; on this plane it would not even be that, because
/// `kernels_metal::layout` stamps the row move for bf16 and f32 alone and an
/// `i32` map is not one of them ([`copyable`] declines what the entries
/// cannot carry).
#[derive(Debug, Clone)]
pub struct Gathered {
    /// The fire intervals this rectangle compacts, in order.
    pub runs: Vec<MaskSpan>,
    /// `[rows]`: the FIRE row each compacted row was read from — the map both
    /// halves of the copy read, in the two directions.
    pub rows_host: Vec<i32>,
    /// The same vector, staged.
    pub rows: Tensor,
    /// `[rows]`: `positions`, re-laid in gathered row order.
    ///
    /// **THE AMBIENT TABLES ARE THIS PLANE'S OWN LINE OF THE COPY**, and the
    /// CUDA sibling has no counterpart to them. There a plan builder walks
    /// the boundaries host-side and bakes a schedule; here
    /// `kernels_metal::attn::plan_prefill` is a pure carrier and the sdpa
    /// shaders read `position_ids[row]` and `req_of_token[row]` by the LOCAL
    /// row of the launch. A gathered launch's local row `i` is fire row
    /// `rows_host[i]`, so the two tables have to be permuted the way the
    /// activations are — and they are `i32`, which the row-move entries do
    /// not stamp, so the permutation is done here and staged.
    pub positions_host: Vec<i32>,
    /// The same vector, staged.
    pub positions: Tensor,
    /// `[rows]`: `request_of_token`, re-laid in gathered row order.
    ///
    /// **ITS ENTRIES STAY ABSOLUTE, WHICH IS WHAT KEEPS THE POOL FIRE-WIDE.**
    /// A permutation moves the rows and does not renumber what they hold, so
    /// a gathered row still names the fire lane that owns it — and
    /// [`Run::pool`](crate::run::Run::pool) hands the sdpa entries the same
    /// fire-wide page tables it hands every other window
    /// (its doc argues why this plane cuts nothing there). The re-cut
    /// [`GatheredSpace`] tables below are the OTHER reading, for the ops that
    /// name a geometry vector as an operand and index it by the launch's own
    /// lane; the two never meet, exactly as they already do not meet under a
    /// split.
    pub request_of_token_host: Vec<i32>,
    /// The same vector, staged.
    pub request_of_token: Tensor,
    /// One entry per kv geometry space, in space order.
    pub spaces: Vec<GatheredSpace>,
}

/// One kv space's geometry, re-cut for a gathered window's lanes.
///
/// **WHY THE PAGE-ID LIST IS COPIED AND NOT SLICED**, which is the whole
/// reason this struct exists. A window's `page_indptr` is ordinarily a SLICE
/// of the fire's, entries left absolute, because the page-id list it bounds
/// is handed over whole and a contiguous run of lanes owns a contiguous run
/// of it. Gathered lanes do not: lanes 0 and 2 own `indices[i0..i1]` and
/// `indices[i2..i3]` with lane 1's pages standing between them, and no
/// `[lanes + 1]` vector over the whole list can name both spans for requests
/// 0 and 1 — request 0's end and request 1's start are one entry. So the
/// LIST is compacted too, and the bounds over it become a fresh prefix sum.
///
/// These are what [`Run::tensor`](crate::run::Run::tensor) answers for the
/// four `GeomKind`s [`copyable`] admits, in place of the lane slice a split
/// window takes.
#[derive(Debug, Clone)]
pub struct GatheredSpace {
    /// `[lanes + 1]`: bounds over
    /// [`page_indices_host`](GatheredSpace::page_indices_host), a fresh
    /// prefix sum starting at 0.
    pub page_indptr_host: Vec<i32>,
    /// The gathered lanes' page ids, end to end.
    pub page_indices_host: Vec<i32>,
    /// `[lanes]`: how full each gathered lane's last page is.
    pub last_page_lens_host: Vec<i32>,
    /// `[lanes]`: each gathered lane's kv length.
    pub kv_len_host: Vec<i32>,
    /// The four device-side ones, staged.
    pub page_indptr: Tensor,
    /// See [`page_indptr`](GatheredSpace::page_indptr).
    pub page_indices: Tensor,
    /// See [`page_indptr`](GatheredSpace::page_indptr).
    pub last_page_lens: Tensor,
    /// See [`page_indptr`](GatheredSpace::page_indptr).
    pub kv_len: Tensor,
}

/// What one fire needs to know before it can decide to copy anything.
///
/// **THE BUCKET AND THE TOGGLE ARE BOTH THE DEPLOYMENT'S, NOT THE
/// ARTIFACT'S.** P4's menu is keyed by bucket range because the cost model
/// is, and turning `Composition::bucket` — a row COUNT — into the index that
/// range is over needs the `Budget` only the shell holds. `enabled` is the
/// A/B switch: `Fallback::Split` is green on this plane and is the oracle a
/// copy is diffed against, so the caller says which arm this fire runs.
///
/// The three vectors are the fire's own host state, borrowed for the length
/// of the call: what the gathered twins are re-laid and re-cut FROM.
#[derive(Debug, Clone, Copy)]
pub struct Copies<'a> {
    /// Which position of `Budget::buckets` this fire's rows land in; `0` for
    /// a deployment that declared no lattice and therefore has one bucket.
    pub bucket: u32,
    /// Does this shell serve `Fallback::Copy` at all?
    ///
    /// **AND ALSO: IS THIS A FIRE A COPY IS SAFE IN?** A masked fire is not,
    /// and the reason is the one [`GatheredSpace`] states about the page-id
    /// list, in a place the gather does not reach. This plane's mask is a
    /// row-major plane of BYTES at a stated stride (`crate::mask`), read as
    /// `attention_mask[row * stride + kp]` — so a gathered launch would need
    /// the PLANE permuted by row as well as the two `i32` tables, in an
    /// element the row-move entries do not stamp and at a width the packed
    /// blob is not carved for. It is the same problem and it has the same
    /// answer (compact the plane, re-lay the enable column); it is not solved
    /// here, so a fire that staged mask bits takes the split, which is always
    /// correct. That is also what lets the gathered window leave
    /// `mask_enabled` sliced rather than permuted: a fire no lane masked
    /// writes that column all zeros, and any `rows` of a zero vector are the
    /// same `rows` of zeros.
    pub enabled: bool,
    /// This fire's host geometry, one per kv space — what the gathered pool
    /// tables are re-cut from.
    pub spaces: &'a [Geometry],
    /// `[rows]`: this fire's absolute positions, in fire row order.
    pub positions: &'a [i32],
    /// `[rows]`: which lane owns each token row, in fire row order.
    pub request_of_token: &'a [i32],
}

impl Copies<'_> {
    /// The answer for a shell that does not copy: split everything, which is
    /// what this shell did before the copy existed.
    #[must_use]
    pub fn off() -> Copies<'static> {
        Copies {
            bucket: 0,
            enabled: false,
            spaces: &[],
            positions: &[],
            request_of_token: &[],
        }
    }
}

/// Every region's windows, deduplicated.
///
/// Deduplicated because a plan has hundreds of regions and at most a handful
/// of distinct windows — one per contiguous run of the class order — and the
/// rebased boundary vectors are staged one per DISTINCT window, in a single
/// copy, rather than one per region.
///
/// **A REGION HAS A LIST AND NOT A WINDOW**, because P4's fallback is a list:
/// a consumer it could not seat runs once per maximal interval of its class
/// set, and the interval is what an encode is cut at. One entry is the case P4
/// exists to produce and is what every region of every SKU the catalog seats
/// has; the empty window is one entry too, so that a region with no rows is
/// resolvable rather than special.
#[derive(Debug, Clone, Default)]
pub struct Windows {
    windows: Vec<Window>,
    /// Every region's runs end to end, as positions in
    /// [`windows`](Windows::windows) — region `r`'s are
    /// `runs[of_region[r].0 .. of_region[r].0 + of_region[r].1]`.
    runs: Vec<u32>,
    /// Region index → `(where its runs start, how many)`.
    of_region: Vec<(u32, u32)>,
}

/// Every value the region's nodes name — the inputs of all of them, then the
/// outputs of all of them.
///
/// **ONE WALK, THREE READERS**, which is the whole reason it is a function.
/// [`copyable`] asks whether the copy path can re-point them all,
/// `crate::scratch` asks how many bytes the widest copied region's rectangles
/// come to, and `crate::dispatch::copy` asks which of them are read and which
/// are written. Three walks over one node range is three chances for the
/// question "what does this region touch" to be answered three ways.
///
/// Flat rather than per node, and the two lists are still separate: a value
/// both read and written by the region ends up in both, which is exactly the
/// in-place case the copy plan has to fold onto ONE staging rectangle. Which
/// node did which is not a fact any of the three readers uses.
///
/// `None` for a template naming a node the plan lacks, which every caller
/// answers by declining rather than by guessing.
pub(crate) fn operands(
    nodes: &[model_ir::Node],
    region: &Region,
) -> Option<(Vec<model_ir::ValueId>, Vec<model_ir::ValueId>)> {
    let mut ins: Vec<model_ir::ValueId> = Vec::new();
    let mut outs: Vec<model_ir::ValueId> = Vec::new();
    for node in region.nodes.clone() {
        let node = nodes.get(node as usize)?;
        macro_rules! collect {
            ($op:expr) => {{
                $op.inputs(&mut ins);
                $op.outputs(&mut outs);
            }};
        }
        match &node.op {
            Operation::Attention(op) => collect!(op),
            Operation::Linear(op) => collect!(op),
            Operation::Elementwise(op) => collect!(op),
            Operation::Layout(op) => collect!(op),
            Operation::Collective(op) => collect!(op),
            Operation::CustomCuda(op) => collect!(op),
        }
    }
    Some((ins, outs))
}

/// Is this region's work something the copy path can actually serve?
///
/// **THE GATHER IS GENERAL AND THE RESOLUTION IS NOT**, which is why this
/// asks about operands rather than about ops. A copy re-points every operand
/// of every node at a compacted rectangle, and `Run::cut` can do that for
/// three shapes: a token-shaped tensor becomes a staging rectangle, a cache
/// binding keeps the fire-wide pool ([`Gathered::request_of_token_host`]
/// argues why), and the four geometry vectors [`GatheredSpace`] re-cuts
/// become their gathered twins. A region naming anything else — a lane-shaped
/// value nothing gathers, a mask slab whose entries are bits rather than rows
/// — would silently get the fire's whole vector where it asked for a
/// window's, so it is not copied at all and takes the split, which is always
/// correct.
///
/// **AND ONE EXCLUSION THE CUDA TWIN DOES NOT HAVE: THE ELEMENT.**
/// `kernels_cuda::layout::gather_rows` is dtype-blind and moves bytes;
/// `kernels_metal::layout::gather_rows` dispatches an entry per element and
/// `row_gather.metal` is stamped for bf16 and f32 alone — the two a copied
/// attention region holds, an activation and a log-sum-exp column. A region
/// naming a row-shaped `i32` (a token id, a write descriptor) would reach
/// that entry and be REFUSED, which is a fire that fails rather than one that
/// lies; it is excluded here so the two halves agree about what the copy path
/// supports, exactly as `TokensTimes` is. A third element is a third
/// `instantiate_row_gather` line and one more arm below.
///
/// Struct operands are exempt by construction: a plan payload is host state
/// resolved through `Run::put`/`Run::prefill_plan`, and its own window is the
/// region that BUILT it — which is copied whenever this one is, for the
/// reason `model_exec::fire::fallback::copies` states.
pub(crate) fn copyable(trace: &Trace, region: &Region) -> bool {
    let Some((ins, outs)) = operands(&trace.nodes, region) else {
        return false;
    };
    ins.iter().chain(outs.iter()).all(|id| {
        let Some(decl) = trace.values.get(id.0 as usize) else {
            return false;
        };
        match &decl.def {
            // The PAGED pool, and ONLY the paged one. A recurrent bank is
            // addressed by SLOT, and `Run::recurrent` reads its slot map with
            // `Run::cut_rows` — which a gathered window answers with the
            // permuted rows only for the two ambient tables it carries twins
            // of, and answers with a SLICE for everything else. That slice
            // would hand the scan the first `n` rows' banks instead of the
            // gathered rows' — the page-id list's problem again, in a table
            // `GatheredSpace` does not re-cut. Wrong state, no fault; so a
            // region that scans one does not copy.
            Def::Cache(c) => matches!(
                trace.caches.get(*c as usize),
                Some(model_ir::CacheRow::Kv { .. })
            ),
            // The four geometry vectors that re-cut goes through. `Indices`
            // is compacted rather than sliced; the rest are per-lane.
            Def::Input(RuntimeInput::Geometry { kind, .. }) => matches!(
                kind,
                GeomKind::Indptr | GeomKind::Indices | GeomKind::LastPageLen | GeomKind::KvLen
            ),
            // The mask plane, whose rows are (query, key) BYTES at a stated
            // stride rather than activations: permuting it is the page-id
            // list's problem again ([`Copies::enabled`]). A masked fire
            // disables copies wholesale for that reason; this says the same
            // thing per region, so a future fire-level relaxation cannot
            // quietly let one through.
            Def::Input(RuntimeInput::Mask { .. }) => false,
            _ => match &decl.ty {
                // A plan payload: host state, not a rectangle.
                Ty::Struct(_) => true,
                Ty::Tensor { shape, dtype } => match shape.first() {
                    // Row-shaped: the staging rectangle — if the row move is
                    // stamped for what it holds. See the item doc.
                    Some(Dim::Tokens) => matches!(dtype, Dtype::Bf16 | Dtype::F32),
                    // `TokensTimes(k)` is `k` rectangle rows per TOKEN row, and
                    // `Gathered::rows_host` names token rows — one index per
                    // `k` rows to move. `kernels_metal::layout::gather_rows`
                    // refuses the mismatch rather than moving the wrong bytes
                    // (`index.rows != tight.rows`), so this would be a fire
                    // that fails rather than one that lies; it is excluded here
                    // so that the two halves agree about what the copy path
                    // supports. Expanding the map by `k` is the fix when a
                    // withdrawn consumer needs one.
                    Some(Dim::TokensTimes(_)) => false,
                    // Window-free: handed over whole, gathered or not.
                    Some(Dim::Const(_)) | None => true,
                    Some(Dim::Lanes | Dim::LanesPlus(_)) => false,
                    // The second row axis, excluded for `TokensTimes`' reason
                    // and not for a weaker one: `Gathered::rows_host` is a map
                    // of TOKEN rows, and a patch column's rows are a different
                    // row space entirely (multimodal §5.1 — patches and tokens
                    // do not break at the same places). Copying one under a
                    // token map would move the wrong bytes, so a region that
                    // reads a patch rectangle does not copy. The patch axis's
                    // lane space (`Images`, `ImagesPlus`) answers the same
                    // `false` for the same reason: `Dim::axis` calls all three
                    // `RowAxis::Patches`, and a token-row map may not cut any
                    // of them.
                    //
                    // **THIS ARM IS REACHABLE NOW**, where it used to be
                    // argued unreachable ("this mirror binds no patch seat at
                    // all"). The seat is bound; the exclusion is the one the
                    // CUDA twin has always carried, and it stands on its own.
                    Some(Dim::Patches | Dim::Images | Dim::ImagesPlus(_)) => false,
                },
            },
        }
    })
}

/// The same question asked of a MASK, which is the one that may be answered
/// `true`.
///
/// **A COPY IS RESOLVED PER MASK AND `copyable` IS PER REGION, AND THAT GAP
/// IS THIS FUNCTION.** `model_exec::fire::fallback::copies` keys on the mask
/// precisely so a prepare region inherits its readers' answer — P4 offers it
/// no row of its own, and a builder that split while its reader gathered
/// would carry ONE set of tables where `r` were read. Asking `copyable` of
/// each region separately re-opens exactly that: a consumer whose operands
/// the copy path declines would split into `r` runs while its builder, whose
/// operands are four geometry vectors and a struct, gathered into one — and
/// the consumer's second run would find no plan payload at its key. So every
/// region standing over the mask has to admit the copy, or none of them does.
pub(crate) fn copyable_mask(
    trace: &Trace,
    compiled: &CompiledModel,
    mask: &model_ir::ClassSet,
) -> bool {
    compiled
        .template()
        .iter()
        .filter(|region| &region.mask == mask)
        .all(|region| copyable(trace, region))
}

/// How many DISTINCT windows this artifact can ever gather.
///
/// **THE NUMBER `crate::inputs` RESERVES STAGING ROOM PER**, and it is a
/// count of MASKS rather than of regions because that is what a gathered
/// window is one of: every region over one mask cuts the same run list, so
/// [`seat`] hands them all one window and one set of staged tables. `0` for
/// an artifact P4 withdrew no copyable region from, which is every SKU
/// outside the qwen family — and then the copy costs that load nothing at
/// all, in this plane or in `crate::scratch`.
#[must_use]
pub fn gathers(trace: &Trace, compiled: &CompiledModel) -> usize {
    let mut masks: Vec<&model_ir::ClassSet> = Vec::new();
    for region in compiled.template() {
        let owed = compiled.fallback.rows.iter().any(|row| {
            region.nodes.contains(&row.node) && row.fallback == model_compiler::Fallback::Copy
        });
        if !owed || masks.contains(&&region.mask) {
            continue;
        }
        if copyable_mask(trace, compiled, &region.mask) {
            masks.push(&region.mask);
        }
    }
    masks.len()
}

/// Give this window a position in the fire's deduplicated list.
///
/// **DEDUPLICATED ON EVERYTHING, NOT ONLY THE SPAN.** A gathered window and a
/// plain one can name the same compacted extent — `{0, 5, 0, 3}` is what a
/// copy of two runs looks like and also what the first three lanes of a fire
/// look like — and they are not the same window: one reads its rows where
/// they lie and the other reads them out of a staging rectangle. Comparing
/// the runs beside the span is what keeps a region from being handed the
/// other one's.
fn seat(windows: &mut Vec<Window>, window: Window) -> u32 {
    let same = |held: &Window| {
        held.span == window.span
            && held.gathered.as_ref().map(|g| &g.runs) == window.gathered.as_ref().map(|g| &g.runs)
    };
    let index = match windows.iter().position(same) {
        Some(index) => index,
        None => {
            windows.push(window);
            windows.len() - 1
        }
    };
    index as u32
}

/// Build the gathered window a list of runs compacts to.
///
/// Four things fall out of the run list and nothing else does:
///
/// - **the row map**, which is the runs' rows concatenated in run order. That
///   order is the one the compacted rectangle is in, so it is also the order
///   the lanes and their qo boundaries have to be in — a gather that laid the
///   rows down in one order and the boundaries in another would hand the
///   encode a ragged view of somebody else's requests.
/// - **the qo boundaries**, rebased over the union: each run's per-lane row
///   counts, appended, prefix-summed from 0. Not `rebase(indptr, span)` of any
///   one run — the union is what the single encode stands over.
/// - **the two ambient row tables**, re-laid under the same map
///   ([`Gathered::positions_host`] argues why this plane has them and the
///   CUDA one does not).
/// - **the per-space pool tables**, re-cut lane by lane
///   ([`GatheredSpace`] argues why the page-id list is copied).
fn gather_of(runs: &[MaskSpan], indptr_host: &[i32], copies: Copies<'_>) -> Window {
    let mut rows_host: Vec<i32> = Vec::new();
    let mut lanes: Vec<usize> = Vec::new();
    let mut bounds: Vec<i32> = vec![0];
    for run in runs {
        for row in run.row_offset..run.row_offset + run.rows {
            rows_host.push(row as i32);
        }
        for lane in run.lane_offset..run.lane_offset + run.lanes {
            let lane = lane as usize;
            // The lane's own row count, off the fire-wide boundaries, added
            // to the running total — which IS the rebase, done once over the
            // union instead of once per run.
            let width = indptr_host
                .get(lane + 1)
                .zip(indptr_host.get(lane))
                .map_or(0, |(end, start)| end - start);
            bounds.push(bounds.last().copied().unwrap_or(0) + width);
            lanes.push(lane);
        }
    }

    // The permutation, applied to the two tables the sdpa shaders index by
    // the launch's own row. `0` for a row the fire did not stage — which
    // cannot happen, since every gathered row is a fire row, and is answered
    // rather than panicked because a shorter vector is the caller's fact and
    // not this function's.
    let relay = |table: &[i32]| -> Vec<i32> {
        rows_host
            .iter()
            .map(|&row| table.get(row as usize).copied().unwrap_or(0))
            .collect()
    };
    let positions_host = relay(copies.positions);
    let request_of_token_host = relay(copies.request_of_token);

    let spaces = copies
        .spaces
        .iter()
        .map(|space| {
            let mut page_indptr_host: Vec<i32> = vec![0];
            let mut page_indices_host: Vec<i32> = Vec::new();
            let mut last_page_lens_host: Vec<i32> = Vec::new();
            let mut kv_len_host: Vec<i32> = Vec::new();
            for &lane in &lanes {
                let start = space.indptr.get(lane).copied().unwrap_or(0).max(0) as usize;
                let end = space.indptr.get(lane + 1).copied().unwrap_or(0).max(0) as usize;
                let pages = space.indices.get(start..end).unwrap_or(&[]);
                page_indices_host.extend_from_slice(pages);
                page_indptr_host.push(page_indices_host.len() as i32);
                last_page_lens_host.push(space.last_page_len.get(lane).copied().unwrap_or(0));
                kv_len_host.push(space.kv_len.get(lane).copied().unwrap_or(0));
            }
            GatheredSpace {
                page_indptr_host,
                page_indices_host,
                last_page_lens_host,
                kv_len_host,
                page_indptr: Tensor::new(NIL, 0, 1, Dtype::I32),
                page_indices: Tensor::new(NIL, 0, 1, Dtype::I32),
                last_page_lens: Tensor::new(NIL, 0, 1, Dtype::I32),
                kv_len: Tensor::new(NIL, 0, 1, Dtype::I32),
            }
        })
        .collect();

    Window {
        span: MaskSpan {
            row_offset: 0,
            rows: rows_host.len() as u32,
            lane_offset: 0,
            lanes: lanes.len() as u32,
        },
        indptr_host: bounds,
        indptr: Tensor::new(NIL, 0, 1, Dtype::I32),
        gathered: Some(Gathered {
            runs: runs.to_vec(),
            rows: Tensor::new(NIL, 0, 1, Dtype::I32),
            rows_host,
            positions: Tensor::new(NIL, 0, 1, Dtype::I32),
            positions_host,
            request_of_token: Tensor::new(NIL, 0, 1, Dtype::I32),
            request_of_token_host,
            spaces,
        }),
        // Filled by the caller, which is the only party that holds the patch
        // table — a gathered window's patch interval is the region's, not the
        // union's, because the copy path never gathers a patch rectangle
        // (`copyable` declines every `Dim::Patches` operand).
        patch: MaskSpan::default(),
    }
}

impl Windows {
    /// The windows of one fire: every region of the template resolved against
    /// this composition's class table, one per interval its mask covers.
    ///
    /// # Errors
    ///
    /// [`Fault::Fragmented`] for a region whose classes are not consecutive in
    /// the fire's class order AND which the artifact owes no `Fallback` row —
    /// a promise P4 made and this fire found broken, which is a bake-integrity
    /// failure rather than a slow path. A region P4 DID write a row for is the
    /// slow path, and is served here as `Fallback::Split { r }` — or, where
    /// [`copies`](Copies) says this fire's bucket asks for a copy and the
    /// region's operands admit one, as a single [`Gathered`] window.
    pub fn of(
        trace: &Trace,
        compiled: &CompiledModel,
        classes: &WindowTable,
        patches: &WindowTable,
        indptr_host: &[i32],
        copies: Copies<'_>,
    ) -> Result<Windows> {
        let mut windows: Vec<Window> = Vec::new();
        let mut runs: Vec<u32> = Vec::with_capacity(compiled.template().len());
        let mut of_region: Vec<(u32, u32)> = Vec::with_capacity(compiled.template().len());
        let mut spans: Vec<MaskSpan> = Vec::new();

        for (at, region) in compiled.template().iter().enumerate() {
            // **WHICH TABLE THIS REGION'S OWN ROWS COME OUT OF** is its
            // capture unit's axis, exactly as `model_exec::fire::walk` reads
            // it — the two have to agree about the run count, or the walk's
            // launch loop and this table's window list are cut at different
            // places. A single-unit artifact answers `RowAxis::PRIMARY` for
            // every region and this is the line it always was.
            let axis = compiled.axis_of(at);
            match axis {
                model_ir::RowAxis::Tokens => classes.spans_into(&region.mask, &mut spans),
                model_ir::RowAxis::Patches => patches.spans_into(&region.mask, &mut spans),
            }
            // **AND THE OTHER AXIS'S INTERVAL, COMPUTED FOR EVERY REGION** —
            // because the embed merge is a token region reading a patch
            // rectangle and one resolution needs both. A FRAGMENTED patch
            // window is refused rather than resolved to its first piece: P4's
            // fallback menu is the token axis's, so there is no split to fall
            // back to here and a silent first-run answer would move the wrong
            // rows.
            let patch = match patches.span(&region.mask) {
                Ok(span) => span.unwrap_or_default(),
                Err(runs) => {
                    return Err(Fault::Fragmented {
                        region: at as u32,
                        runs,
                        promised: None,
                    });
                }
            };
            // P4's menu is asked with the region's own `axis`: the table
            // answers per axis now (`model_exec::fire::fallback`'s header),
            // where this used to hold a hand-written `menu` word in front of
            // every call to keep the token seriation's answers off a patch
            // region. The rebased qo boundaries below are still the token
            // rectangle's alone — a patch region's `indptr_host` stays empty,
            // because that axis's bounds vector is
            // `RuntimeInput::PatchSegments` and not this one.
            if spans.len() > 1 {
                // The two integrity questions, asked of the artifact
                // rather than of the fire. Did P4 PROMISE this window
                // consecutive — a capture region it seated and wrote no
                // fallback row for? And is this fire's run count within the
                // one the shipped order breaks the mask into? A fire's order
                // is that order with the absent classes dropped, and dropping
                // a class can only close a gap, so neither can happen to a
                // `CompiledModel` and a `WindowTable` built from each other.
                let bound = fallback::bound(compiled, axis, &region.mask);
                if fallback::promised(compiled, axis, region) || spans.len() > bound as usize {
                    return Err(Fault::Fragmented {
                        region: at as u32,
                        runs: spans.len(),
                        promised: fallback::promised(compiled, axis, region).then_some(bound),
                    });
                }
                // **THIS PLANE SERVES `Fallback::Split` AND `Fallback::Copy`,
                // AND NOT `Fallback::Grouped`**, and the reason it need not
                // check for the third is that it cannot be handed one: P4
                // writes a `Grouped` row only for a region whose every op the
                // caller named in `DeviceProfile::grouped`, and this shell
                // names none (the CUDA one passes `engine_cuda::GROUPED`; see
                // `engine_cuda::window::Windows::of` for what honouring the
                // row costs). The day it names one, this is where the union
                // window and its segment list go — and until then a `Grouped`
                // row reaching here would be `model_exec::fire::walk` turning
                // its encode loop once against `r` windows cut below, which
                // computes only the first interval.
            }
            // An empty mask (a region no class demands) answers the zero
            // window, which is the same answer a composition without this
            // behavior gives — and the walk skips both for the same reason.
            if spans.is_empty() {
                spans.push(MaskSpan::default());
            }

            // P4'S OTHER ANSWER. A window in pieces whose bucket asks for a
            // copy, whose operands the copy can re-point, becomes ONE window
            // over the compacted rectangle — and the region then costs one
            // encode, which is the whole point.
            if spans.len() > 1
                && copies.enabled
                && fallback::copies(compiled, axis, &region.mask, copies.bucket)
                && copyable_mask(trace, compiled, &region.mask)
            {
                let mut gathered = gather_of(&spans, indptr_host, copies);
                gathered.patch = patch;
                of_region.push((runs.len() as u32, 1));
                runs.push(seat(&mut windows, gathered));
                continue;
            }

            of_region.push((runs.len() as u32, spans.len() as u32));
            for &span in &spans {
                let window = Window {
                    span,
                    indptr_host: match axis {
                        model_ir::RowAxis::Tokens => rebase(indptr_host, span),
                        model_ir::RowAxis::Patches => Vec::new(),
                    },
                    indptr: Tensor::new(NIL, 0, 1, Dtype::I32),
                    gathered: None,
                    patch,
                };
                runs.push(seat(&mut windows, window));
            }
        }

        Ok(Windows {
            windows,
            runs,
            of_region,
        })
    }

    /// How many distinct windows this fire has.
    #[must_use]
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Does it hold none? Only for a template with no regions at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Every window's `i32` vectors, end to end — what the shell writes in
    /// one copy.
    ///
    /// **ONE BLOB, AND [`bind`](Windows::bind) WALKS IT IN THE SAME ORDER.**
    /// A window contributes its rebased boundaries; a GATHERED one
    /// contributes its row map, its two re-laid ambient tables and its
    /// per-space pool tables behind them. The two functions are written as
    /// one traversal in two directions for exactly the reason the copy's own
    /// two halves are: an order that could drift is a handle that could point
    /// at another window's vector.
    #[must_use]
    pub fn packed(&self) -> Vec<i32> {
        let mut out: Vec<i32> = Vec::new();
        for window in &self.windows {
            out.extend_from_slice(&window.indptr_host);
            let Some(gathered) = &window.gathered else {
                continue;
            };
            out.extend_from_slice(&gathered.rows_host);
            out.extend_from_slice(&gathered.positions_host);
            out.extend_from_slice(&gathered.request_of_token_host);
            for space in &gathered.spaces {
                out.extend_from_slice(&space.page_indptr_host);
                out.extend_from_slice(&space.page_indices_host);
                out.extend_from_slice(&space.last_page_lens_host);
                out.extend_from_slice(&space.kv_len_host);
            }
        }
        out
    }

    /// Seat the staged boundaries: `base` is where
    /// [`packed`](Windows::packed) landed inside `buffer`.
    ///
    /// One handle per distinct window, minted in the order `packed` wrote
    /// them, which is the order this table holds them in — so the `n`th
    /// window's view starts `n` boundary vectors into the copy. The CUDA
    /// sibling adds those same byte counts to a device address and cannot
    /// fail; here each view is a row in the handle table and is bounds-checked
    /// against the reservation as it is minted.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] when a window's boundaries would leave `buffer` —
    /// a staging reservation too small for what `packed` produced — or when
    /// the handle table is full.
    pub fn bind(&mut self, handles: &Handles, packed: u32) -> Result<()> {
        let mut at = 0u64;
        let mut take = |vector: &[i32]| -> Result<Tensor> {
            let rows = vector.len() as u32;
            let bytes = u64::from(rows) * 4;
            let cut = handles.cut(packed, at, bytes)?;
            at += bytes;
            Ok(Tensor::new(cut, rows, 1, Dtype::I32))
        };
        for window in &mut self.windows {
            window.indptr = take(&window.indptr_host)?;
            let Some(gathered) = &mut window.gathered else {
                continue;
            };
            gathered.rows = take(&gathered.rows_host)?;
            gathered.positions = take(&gathered.positions_host)?;
            gathered.request_of_token = take(&gathered.request_of_token_host)?;
            for space in &mut gathered.spaces {
                space.page_indptr = take(&space.page_indptr_host)?;
                space.page_indices = take(&space.page_indices_host)?;
                space.last_page_lens = take(&space.last_page_lens_host)?;
                space.kv_len = take(&space.kv_len_host)?;
            }
        }
        Ok(())
    }

    /// How many encodes a region costs in this fire — `1` for a window P4
    /// seated, `r` for one it could not, and `1` for an empty window.
    ///
    /// THE SAME NUMBER `model_exec::fire::walk` LOOPS ON, and it is the same
    /// number because both read it off the same class table: the walk asks
    /// `WindowTable::spans_into` and this asked it once per region when the
    /// table was built. A disagreement would show up as
    /// [`at`](Windows::at)'s panic rather than as a wrong window.
    #[must_use]
    pub fn runs(&self, region: u32) -> u32 {
        self.of_region.get(region as usize).map_or(0, |held| held.1)
    }

    /// How many encodes this fire's walk makes over the whole template.
    ///
    /// **THE NUMBER A COPY EXISTS TO LOWER, AND THE ONLY ONE A CALLER CAN SEE
    /// FROM OUTSIDE.** `model_exec::fire::walk` loops `Windows::runs(region)`
    /// times per region — the same table, read the same way — so this is that
    /// loop's total, known before a single dispatch is encoded. A fire whose
    /// windows P4 all seated answers one per region; a split adds `r - 1` per
    /// fragmented region; a copy takes them back off.
    #[must_use]
    pub fn launches(&self) -> u32 {
        self.of_region.iter().map(|&(_, runs)| runs.max(1)).sum()
    }

    /// How many regions of this fire are served as a `Fallback::Copy`.
    ///
    /// Zero unless the shell was told to copy AND P4's table asked for one at
    /// this fire's bucket AND the region's operands admitted it — which is
    /// three questions with one visible answer, so it is worth being able to
    /// ask it.
    #[must_use]
    pub fn copied(&self) -> u32 {
        self.of_region
            .iter()
            .filter(|&&(start, _)| {
                self.runs
                    .get(start as usize)
                    .and_then(|&index| self.windows.get(index as usize))
                    .is_some_and(|window| window.gathered.is_some())
            })
            .count() as u32
    }

    /// The most encodes any region of this fire costs — what a per-run table
    /// is sized at.
    #[must_use]
    pub fn max_runs(&self) -> u32 {
        self.of_region
            .iter()
            .map(|&(_, runs)| runs)
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// One region's window, for one run of it.
    ///
    /// A region index this table does not hold, or a run past the ones it cut
    /// for that region, is an integrity failure of the shell — the cursor
    /// counts the same template the table was built from, and the walk loops
    /// over the same span list — so it panics with a sentence rather than
    /// dressing up as a window.
    #[must_use]
    pub fn at(&self, region: u32, run: u32) -> &Window {
        self.of_region
            .get(region as usize)
            .filter(|&&(_, runs)| run < runs)
            .and_then(|&(start, _)| self.runs.get((start + run) as usize))
            .and_then(|index| self.windows.get(*index as usize))
            .unwrap_or_else(|| {
                panic!(
                    "region {region} has no run {run}; this fire cut it into {} \
                     over a template of {}",
                    self.runs(region),
                    self.of_region.len()
                )
            })
    }
}

/// **THE BAKE-TIME HALF OF THE WINDOW ARGUMENT**: no attention schedule may
/// be built over more classes than the node consuming it runs in.
///
/// The whole argument, and the walk that carries it out, is
/// [`model_exec::store::check::no_schedule_straddles_its_readers`] — neutral IR
/// reasoning over a `Trace` and a `CompiledModel`, which is what it always
/// was, written once now instead of twice. This is the shell's door onto it:
/// same signature, this shell's [`Fault`].
///
/// # Errors
///
/// [`Fault::Straddled`], naming the value, the consuming node, and the two
/// class sets.
pub fn no_schedule_straddles_its_readers(trace: &Trace, compiled: &CompiledModel) -> Result<()> {
    Ok(check::no_schedule_straddles_its_readers(trace, compiled)?)
}

/// Where the walk is: which region of the template, and which run of that
/// region's window.
///
/// **TWO NUMBERS, ONE OBJECT, BECAUSE THEY ARE READ TOGETHER.** A `Run`
/// resolves every operand at `windows.at(region, run)`, and a pair that could
/// be handed in separately is a pair that could be handed in from two
/// different walks. The [`Cursor`] writes both — the region before the
/// region's first node, the run before each encode of it — and the `Run`
/// holds a shared reference to the same object; that is the whole mechanism,
/// and it is a `Cell` rather than a `&mut` because `walk` takes the sink and
/// the dispatch as two separate borrows.
#[derive(Debug, Default)]
pub struct At {
    /// The region index, in `CompiledModel::template` order.
    pub region: Cell<u32>,
    /// Which run of that region's window: `0` always, and `0..r` for a region
    /// P4 could not seat.
    pub run: Cell<u32>,
}

impl At {
    /// A cursor position at the top of the template.
    #[must_use]
    pub fn new() -> At {
        At::default()
    }
}

/// This shell's [`Sink`]: the region counter a [`Run`](crate::run::Run) reads
/// its window out of.
///
/// **THE EAGER CURSOR RECORDS NOTHING, LIKE `EagerSink`, AND CARRIES ONE
/// NUMBER.** The walk calls [`region_begin`](Sink::region_begin) for every
/// region of the template in order — including the ones this composition has
/// no rows for, which is what makes the count an index rather than a guess —
/// and a `Run` holding a shared reference to the same `Cell` then resolves
/// each operand at that region's window.
///
/// **THERE IS NO RECORDING CURSOR HERE, AND NO STREAM SWITCH — DESIGN §6's
/// "no record.rs".** The CUDA sibling carries a second constructor
/// (`Cursor::across`) that writes a per-region stream into a cell, records an
/// event at each fork and waits one at each join, because a captured graph
/// has to CARRY the DAG's parallelism: the capture is replayed later, so the
/// structure must be in it. This shell is eager from end to end — one command
/// buffer, encoded in walk order — so the DAG's serialization IS the schedule
/// (`model_exec::fire::EagerSink`'s doc argues why that is correct rather than
/// merely safe: the walk emits a topological order, and a topological order
/// of a dependency DAG is a legal execution of it). [`Sink::fork`] and
/// [`Sink::join`] are therefore no-ops that name their event and drop it, and
/// the whole `Lanes` bundle — side streams, event handles, the stream cell —
/// has no counterpart on this plane and is not ported.
///
/// What is lost is the OVERLAP, not the correctness: two independent regions
/// that a CUDA capture would run on two streams are encoded one after the
/// other here. That is a performance ceiling this shell has not needed to
/// lift, and the place to lift it is a second command buffer per fork arm
/// with an `MTLEvent` between them — a change to this type, not to the walk.
#[derive(Debug)]
pub struct Cursor<'a> {
    at: u32,
    place: &'a At,
}

impl<'a> Cursor<'a> {
    /// A cursor writing into `place`, counting from the template's first.
    #[must_use]
    pub fn new(place: &'a At) -> Cursor<'a> {
        place.region.set(0);
        place.run.set(0);
        Cursor { at: 0, place }
    }

    /// What the device refused during the walk, if anything.
    ///
    /// **NOTHING CAN, AND THE METHOD IS KEPT SAYING SO.** The CUDA twin
    /// exists because a `cudaEventRecord` inside a `Sink` method has nowhere
    /// to return an error to, so the first failure is held here and drained
    /// afterwards by the code that knows a half-recorded capture must not be
    /// instantiated. This cursor makes no device call at all — it writes a
    /// `u32` into a cell — so there is nothing to hold and nothing to drain.
    /// The signature stays because the CALLER's shape is the seam: the fire
    /// path settles its cursor before it commits, and a future cursor that
    /// does encode across events (see the type's doc) would fill this in
    /// without moving a line above it.
    ///
    /// # Errors
    ///
    /// None today, by construction. The `Result` is the seam, not a claim.
    #[allow(clippy::unnecessary_wraps, reason = "the seam: see the item doc")]
    pub fn settle(self) -> Result<()> {
        Ok(())
    }
}

impl Sink for Cursor<'_> {
    fn region_begin(&mut self, _region: &Region) {
        self.place.region.set(self.at);
        self.place.run.set(0);
        self.at += 1;
    }
    fn region_end(&mut self, _region: &Region) {}

    /// **THE SPLIT'S ONE PIECE OF STATE.** A region P4 could not seat runs
    /// once per interval of its class set, and every operand the `Run`
    /// resolves after this call is cut at THAT interval — its rows, its lanes,
    /// its rebased qo boundaries. A cursor that ignored this would hand every
    /// run the first one's window, which is not a fault: it is the first
    /// interval's rows encoded `r` times and the rest never encoded at all.
    fn run(&mut self, run: u32, _runs: u32) {
        self.place.run.set(run);
    }
    fn cond_begin(&mut self, _lowering: &Lowering) {}
    fn cond_arm(&mut self, _arm: u8) {}
    fn cond_end(&mut self) {}
    /// Nothing to record: an eager encode has already ordered this region
    /// against everything before it (see the type doc).
    fn fork(&mut self, _event: EventId) {}
    /// Nothing to wait on, for the same reason `fork` records nothing.
    fn join(&mut self, _event: EventId) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_exec::fire::ClassWindow;
    use model_compiler::Phase;
    use model_ir::ClassSet;

    /// The design's own diagram: 10 prefill rows over 2 lanes, then 3 decode
    /// rows over 3 lanes.
    fn table() -> WindowTable {
        WindowTable::new(vec![
            ClassWindow {
                row_offset: 0,
                rows: 10,
                lane_offset: 0,
                lanes: 2,
            },
            ClassWindow {
                row_offset: 10,
                rows: 3,
                lane_offset: 2,
                lanes: 3,
            },
        ])
    }

    #[test]
    fn a_mask_over_both_classes_is_the_whole_fire() {
        let span = table()
            .span(&ClassSet::of([0, 1]))
            .expect("consecutive")
            .expect("non-empty");
        assert_eq!(span.row_offset, 0);
        assert_eq!(span.rows, 13);
        assert_eq!(span.lane_offset, 0);
        assert_eq!(span.lanes, 5);
    }

    #[test]
    fn one_class_is_its_own_interval() {
        let span = table()
            .span(&ClassSet::of([1]))
            .expect("consecutive")
            .expect("non-empty");
        assert_eq!((span.row_offset, span.rows), (10, 3));
        assert_eq!((span.lane_offset, span.lanes), (2, 3));
    }

    #[test]
    fn the_boundaries_are_rebased_to_the_window_s_own_zero() {
        // qo boundaries of the whole fire: two prefills then three decodes.
        let indptr = [0, 7, 10, 11, 12, 13];
        let decode = table()
            .span(&ClassSet::of([1]))
            .expect("consecutive")
            .expect("non-empty");
        assert_eq!(rebase(&indptr, decode), vec![0, 1, 2, 3]);
        let prefill = table()
            .span(&ClassSet::of([0]))
            .expect("consecutive")
            .expect("non-empty");
        assert_eq!(rebase(&indptr, prefill), vec![0, 7, 10]);
    }

    /// The cursor is a counter and nothing else: every region the walk
    /// announces lands in the cell the `Run` reads, in template order — and
    /// every run of that region's window lands in the cell beside it.
    #[test]
    fn the_cursor_counts_regions_and_their_runs_into_the_cells() {
        let place = At::new();
        place.region.set(7);
        place.run.set(3);
        let cell = &place.region;
        let mut cursor = Cursor::new(&place);
        assert_eq!(cell.get(), 0, "a fresh cursor rebases the cell");
        assert_eq!(place.run.get(), 0, "and the run beside it");
        let region = Region {
            nodes: 0..0,
            mask: ClassSet::of([0]),
            phase: Phase::Capture,
            lowering: Lowering::AlwaysLaunch,
            stream: 0,
            wait: Vec::new(),
            open: None,
            close: None,
            sm_hint: None,
            collective: false,
        };
        cursor.region_begin(&region);
        assert_eq!(cell.get(), 0, "the first region is index 0");
        cursor.region_end(&region);
        cursor.region_begin(&region);
        assert_eq!(cell.get(), 1);
        // The fork points are announced and dropped — an eager encode has
        // already ordered them.
        cursor.fork(EventId(0));
        cursor.join(EventId(0));
        assert_eq!(cell.get(), 1, "an event point is not a region");

        // A region P4 could not seat announces one run per interval, and the
        // next region rebases the count — a run index that leaked across a
        // region boundary would resolve the next region's nodes at a window
        // it does not have.
        cursor.run(1, 2);
        assert_eq!(place.run.get(), 1);
        cursor.region_end(&region);
        cursor.region_begin(&region);
        assert_eq!(place.run.get(), 0, "a new region starts at its first run");
        cursor.settle().expect("an eager cursor never holds a fault");
    }
}
