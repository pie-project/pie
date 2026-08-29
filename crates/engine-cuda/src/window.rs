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
//! walk dispatches its nodes once per window, and each launch takes its own
//! pointer, its own extent and — the part that is easy to get silently wrong
//! — its own rebased qo boundaries. A ragged view's `indptr` is offsets INTO
//! the rectangle it cuts, so the second run's must start at 0 again over the
//! second run's lanes; sharing the first run's would hand the launch a vector
//! that describes somebody else's requests.
//!
//! ```text
//! classes in fire order:  [ 4 : 3 rows | 0 : 5 rows | 5 : 2 rows ]
//! mask {4,5,6,7}:          ──run 0──                 ──run 1──
//! qo indptr, fire-wide:   [0, 1, 3, 4, 6, 9, 10, 12]
//!            run 0:       [0, 1, 3]        run 1: [0, 2]
//! ```
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
//! into a scratch slab, the region's nodes run once over it, and the answers
//! are scattered back to the fire rows they came from
//! (`kernels_cuda::layout::gather_rows`). [`Gathered`] is what such a window
//! carries beyond a split's: the row map the two kernels read, and — because
//! a paged consumer addresses its kv by LANE and the gathered lanes are not
//! contiguous either — the pool tables and planning twins re-cut for the
//! union. Those are small, host-computable and staged beside the boundary
//! vectors; only the activations move on the device, which is the whole
//! reason a copy is cheap.
//!
//! ```text
//! classes in fire order:  [ 4 : 3 rows | 0 : 5 rows | 5 : 2 rows ]
//! mask {4,5,6,7}:          ──run 0──                 ──run 1──
//! split:                   launch over [0,3)         launch over [8,10)
//! copy:                    gather rows 0 1 2 8 9  ->  ONE launch over [0,5)
//!                          scatter back to 0 1 2 8 9
//! ```
//!
//! **THE BUILDER TAKES THE SAME ANSWER AS ITS READERS.** An attention
//! schedule is carved for one window, and a consumer standing over the union
//! of two runs must read a schedule carved over that union — so the prepare
//! region that builds it is copied whenever its readers are, even though P4
//! owes it no row of its own (`engine::fire::fallback::copies` argues why the
//! question is asked of the MASK). The two masks being equal is checked at
//! load, by name, in [`no_schedule_straddles_its_readers`].
//!
//! # And when the kernel can walk the pieces itself
//!
//! `Fallback::Grouped` is the other answer, and this table serves it by
//! cutting ONE window where the split cuts `r`: the span is the UNION of the
//! `r` intervals and [`Window::segments_host`] says which of its rows belong
//! to the consumer. That rectangle contains foreign rows — the classes P4
//! could not keep out of the gaps — so it is only legal for an op that touches
//! the rows it was told about and no others, which is a per-op fact the
//! compiler is handed as `DeviceProfile::grouped` and which the artifact
//! restates here as a `FallbackTable` row. Today one op has it:
//! `linear.lora_correct`, whose weight side was already runtime-indexed and
//! whose kernel is one file in this tree.
//!
//! ```text
//! classes in fire order:  [ 0 : 3 rows | 2 : 4 rows | 1 : 2 rows | 3 : 5 rows ]
//! adapter mask {2,3}:                    ──run 0──               ──run 1──
//! Split   → 2 windows:    (3,4) and (9,5), two launches
//! Grouped → 1 window:     (3,11), segments [(0,4), (6,5)], one launch
//! ```
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

use crate::device::graph::Event;
use engine::fire::{EventId, MaskSpan, Sink, WindowTable, fallback};
use engine::store::check::{self, rebase};
use kernels_cuda::Tensor;
use model_compiler::{CompiledModel, Lowering, Phase, Region};
use model_ir::{Def, Dim, Dtype, GeomKind, Operands, Operation, RuntimeInput, Trace, Ty};

use crate::error::{Fault, Result};
use crate::store::kv::Geometry;

/// One window, and the qo boundaries that go with it.
///
/// The span is the arithmetic (rows and lanes, both, because the IR has both
/// symbols); the two indptrs are the one thing a window cannot slice, because
/// a ragged view's boundaries are OFFSETS INTO the rectangle they cut and a
/// sub-rectangle starts at zero. So each window carries its own rebased copy —
/// `[lanes + 1]` entries, the first of them 0 — device-side for the launches
/// and host-side for the plan builders that walk the contents (the duality
/// [`CachePlanning`](crate::run::CachePlanning) states per cache space).
#[derive(Debug, Clone)]
pub struct Window {
    /// The rows and lanes this window covers, in fire coordinates.
    ///
    /// **FOR A GATHERED WINDOW THIS IS THE COMPACTED RECTANGLE**, not a
    /// fire interval: `row_offset` and `lane_offset` are 0 and the counts are
    /// the union's. That is the right reading and not a fudge — every
    /// consumer of a gathered window reads the scratch rectangle, whose rows
    /// start at its own zero, and the map back to fire coordinates is
    /// [`Gathered::rows_host`], which is where it belongs.
    pub span: MaskSpan,
    /// `[lanes + 1]`: the window's qo boundaries, rebased to start at 0.
    pub indptr_host: Vec<i32>,
    /// The same vector, staged. `Tensor::new(0, 0, 0, ..)` until
    /// [`Windows::bind`] has been given the staging base.
    pub indptr: Tensor,

    /// **`Fallback::Grouped`: WHICH ROWS OF THIS RECTANGLE ARE ACTUALLY THE
    /// CONSUMER'S** — `[segs][2]` as `(row offset within the span, rows)`,
    /// ascending. EMPTY for every ordinary window, which is every window of
    /// every artifact P4 seated whole.
    ///
    /// A region P4 answered `Fallback::Grouped` for gets ONE window rather
    /// than `r`, and that window's span is the UNION of the `r` intervals —
    /// so it contains rows belonging to classes the consumer's mask does not
    /// hold, standing in the gaps. This vector is what keeps the launch off
    /// them, and it is rebased to the span for the same reason
    /// [`indptr_host`](Window::indptr_host) is: the rectangle a launch is cut
    /// at starts at its own zero.
    pub segments_host: Vec<i32>,
    /// The same vector, staged — beside the boundaries, in the same copy.
    /// `Tensor::new(0, 0, 0, ..)` for a window with no segments.
    pub segments: Tensor,
    /// The artifact's load-time bound on the segment count
    /// (`engine::fire::max_runs`), carried here because it is what sizes the
    /// grid's segment axis and the fire is not allowed to size it
    /// (decision #15).
    pub segment_cap: u32,
    /// Present iff this window is a [`Fallback::Copy`](model_compiler::Fallback)
    /// — the runs it compacts, and everything a consumer needs to read them
    /// as one.
    pub gathered: Option<Gathered>,
}


impl Window {
    /// How many segments this window states — `0` for an ordinary one.
    #[must_use]
    pub fn segs(&self) -> u32 {
        self.segments_host.len() as u32 / 2
    }

    /// The longest segment's row count — the grid's row axis for a grouped
    /// launch, and `0` when there are none.
    #[must_use]
    pub fn segment_rows(&self) -> u32 {
        self.segments_host
            .chunks_exact(2)
            .map(|pair| pair[1].max(0) as u32)
            .max()
            .unwrap_or(0)
    }
}


/// A `Fallback::Copy`'s window: which fire rows the compacted rectangle is
/// made of, and the per-space tables the gathered LANES address the pool by.
///
/// **TWO GATHERS, AND ONLY ONE OF THEM IS ON THE DEVICE.** The activations
/// are big rectangles and move through
/// [`gather_rows`](kernels_cuda::layout::gather_rows). Everything else a
/// windowed attention reads is a handful of `i32` per lane — where its pages
/// are, how full the last one is, how long its kv is — and those are
/// recomputed on the host for the union and staged in the same copy as the
/// boundary vectors. A device gather of a per-lane vector would be three
/// launches to move forty bytes.
#[derive(Debug, Clone)]
pub struct Gathered {
    /// The fire intervals this rectangle compacts, in order.
    pub runs: Vec<MaskSpan>,
    /// `[rows]`: the FIRE row each compacted row was read from — the map both
    /// halves of the copy read, in the two directions.
    pub rows_host: Vec<i32>,
    /// The same vector, staged.
    pub rows: Tensor,
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
/// The host twins beside them are the same numbers for the plan builders,
/// which walk differences of `kv_indptr` and read `kv_len` per request
/// (`Run::planning`'s duality).
#[derive(Debug, Clone)]
pub struct GatheredSpace {
    /// `[lanes + 1]`: bounds over [`page_indices_host`](GatheredSpace::page_indices_host),
    /// a fresh prefix sum starting at 0.
    pub page_indptr_host: Vec<i32>,
    /// The gathered lanes' page ids, end to end.
    pub page_indices_host: Vec<i32>,
    /// `[lanes]`: how full each gathered lane's last page is.
    pub last_page_lens_host: Vec<i32>,
    /// `[lanes]`: each gathered lane's kv length.
    pub kv_len_host: Vec<i32>,
    /// The four device-side ones, staged.
    pub page_indptr: Tensor,
    pub page_indices: Tensor,
    pub last_page_lens: Tensor,
    pub kv_len: Tensor,
}

/// What one fire needs to know before it can decide to copy anything.
///
/// **THE BUCKET AND THE TOGGLE ARE BOTH THE DEPLOYMENT'S, NOT THE
/// ARTIFACT'S.** P4's menu is keyed by bucket range because the cost model
/// is, and turning `Composition::bucket` — a row COUNT — into the index that
/// range is over needs the `Budget` only the shell holds. `enabled` is the
/// A/B switch: `Fallback::Split` is green on device and is the oracle a copy
/// is diffed against, so the shell ships with copies OFF and a caller turns
/// them on.
#[derive(Debug, Clone, Copy)]
pub struct Copies<'a> {
    /// Which position of `Budget::buckets` this fire's rows land in; `0` for
    /// a deployment that declared no lattice and therefore has one bucket.
    pub bucket: u32,
    /// Does this shell serve `Fallback::Copy` at all?
    ///
    /// **AND ALSO: IS THIS A FIRE A COPY IS SAFE IN?** A masked fire is not,
    /// and the reason is the one [`GatheredSpace`] states about the page-id
    /// list, in a place the gather does not reach. `attention.masked`'s bits
    /// ride in one slab handed over whole, addressed by a per-lane vector of
    /// ABSOLUTE byte offsets that `plan_prefill` binds onto the schedule
    /// (`Run::mask_indptr`) — and gathered lanes own spans of that slab with
    /// other lanes' bits standing between them, exactly as they own spans of
    /// the page-id list. It is the same problem and it has the same answer
    /// (compact the slab, rebuild the offsets); it is not solved here, so a
    /// fire that staged mask bits takes the split, which is always correct.
    /// Today's qwen texts declare no masked axis at all, so nothing this file
    /// gates is affected; gemma's is what this line is for.
    pub enabled: bool,
    /// This fire's host geometry, one per kv space — what the gathered pool
    /// tables are re-cut from.
    pub spaces: &'a [Geometry],
}

impl Copies<'_> {
    /// The answer for a shell that does not copy: split everything, which is
    /// what every shell did before the copy existed.
    #[must_use]
    pub fn off() -> Copies<'static> {
        Copies {
            bucket: 0,
            enabled: false,
            spaces: &[],
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
/// set, and the interval is what a launch is cut at. One entry is the case P4
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

/// Is this region's work something the copy path can actually serve?
///
/// **THE GATHER IS GENERAL AND THE RESOLUTION IS NOT**, which is why this
/// asks about operands rather than about ops. A copy re-points every operand
/// of every node at a compacted rectangle, and `Run::cut` can do that for
/// three shapes: a token-shaped tensor becomes a slab rectangle, a cache
/// binding becomes the gathered pool tables, and the four geometry vectors
/// [`GatheredSpace`] re-cuts become their gathered twins. A region naming
/// anything else — a lane-shaped value nothing gathers, a mask slab whose
/// entries are bits rather than rows — would silently get the fire's whole
/// vector where it asked for a window's, so it is not copied at all and takes
/// the split, which is always correct.
///
/// Struct operands are exempt by construction: a plan payload is host state
/// resolved through `Run::slot`, and its own window is the region that BUILT
/// it — which is copied whenever this one is, for the reason
/// `engine::fire::fallback::copies` states.
fn copyable(trace: &Trace, region: &Region) -> bool {
    let mut operands: Vec<model_ir::ValueId> = Vec::new();
    for node in region.nodes.clone() {
        let Some(node) = trace.nodes.get(node as usize) else {
            return false;
        };
        macro_rules! collect {
            ($op:expr) => {{
                $op.inputs(&mut operands);
                $op.outputs(&mut operands);
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
    operands.iter().all(|id| {
        let Some(decl) = trace.values.get(id.0 as usize) else {
            return false;
        };
        match &decl.def {
            // The PAGED pool, whose lane tables `GatheredSpace` re-cuts — and
            // ONLY the paged one. A recurrent bank is addressed by SLOT, and
            // `Run::recurrent` reads `slot_ids[lane]` by slicing the window's
            // `lane_offset`/`lanes` out of the fire-wide vector. A gathered
            // window's span starts at lane zero and covers the compacted lane
            // count, so that slice would hand the scan the FIRST `n` lanes'
            // banks instead of the gathered lanes' — the page-id list's problem
            // again, in a table `GatheredSpace` does not re-cut. Wrong state,
            // no fault; so a region that scans one does not copy.
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
            // The mask slab, whose entries are BITS and not rows: the span
            // table addressing it is a per-lane vector of absolute offsets
            // and gathering it is the page-id list's problem again
            // ([`Copies::enabled`]). A masked fire disables copies wholesale
            // for that reason; this says the same thing per region, so a
            // future fire-level relaxation cannot quietly let one through.
            Def::Input(RuntimeInput::Mask { .. }) => false,
            _ => match &decl.ty {
                // A plan payload: host state, not a rectangle.
                Ty::Struct(_) => true,
                Ty::Tensor { shape, .. } => match shape.first() {
                    // Row-shaped: the slab.
                    Some(Dim::Tokens) => true,
                    // `TokensTimes(k)` is `k` rectangle rows per TOKEN row, and
                    // `Gathered::rows_host` names token rows — one index per
                    // `k` rows to move. `kernels_cuda::layout::move_rows`
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
                },
            },
        }
    })
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
    /// `copies` says this fire's bucket asks for a copy and the region's
    /// operands admit one, as a single [`Gathered`] window.
    pub fn of(
        trace: &Trace,
        compiled: &CompiledModel,
        classes: &WindowTable,
        indptr_host: &[i32],
        copies: Copies<'_>,
    ) -> Result<Windows> {
        let mut windows: Vec<Window> = Vec::new();
        let mut runs: Vec<u32> = Vec::with_capacity(compiled.template().len());
        let mut of_region: Vec<(u32, u32)> = Vec::with_capacity(compiled.template().len());
        let mut spans: Vec<MaskSpan> = Vec::new();
        // The grid's segment axis, once per fire rather than once per window:
        // it is a property of the ARTIFACT (how many intervals the shipped
        // order breaks any mask into) and a fire may not move it.
        let segment_cap = fallback::max_runs(compiled);

        for (at, region) in compiled.template().iter().enumerate() {
            classes.spans_into(&region.mask, &mut spans);
            let mut segments_host: Vec<i32> = Vec::new();
            if spans.len() > 1 {
                // The two integrity questions, asked of the artifact
                // rather than of the fire. Did P4 PROMISE this window
                // consecutive — a capture region it seated and wrote no
                // fallback row for? And is this fire's run count within the
                // one the shipped order breaks the mask into? A fire's order
                // is that order with the absent classes dropped, and dropping
                // a class can only close a gap, so neither can happen to a
                // `CompiledModel` and a `WindowTable` built from each other.
                let bound = fallback::bound(compiled, &region.mask);
                if fallback::promised(compiled, region) || spans.len() > bound as usize {
                    return Err(Fault::Fragmented {
                        region: at as u32,
                        runs: spans.len(),
                        promised: fallback::promised(compiled, region).then_some(bound),
                    });
                }
                // **P4'S OTHER ANSWER, AND THE ONE THAT CHANGES THE LAUNCH
                // COUNT** (design §3, decision #24). `Fallback::Split` is `r`
                // windows; `Fallback::Grouped` is ONE, over the union of the
                // `r` intervals, carrying the intervals themselves so the
                // kernel can skip the foreign rows between them. The walk
                // reads the same row of the same table
                // (`engine::fire::fallback::grouped`) and turns its launch
                // loop once, so the two cannot disagree about how many runs
                // this region has — which is what `at`'s panic would
                // otherwise be for.
                if fallback::grouped(compiled, region.nodes.clone()) {
                    let union = union_of(&spans);
                    segments_host = spans
                        .iter()
                        .flat_map(|span| {
                            [(span.row_offset - union.row_offset) as i32, span.rows as i32]
                        })
                        .collect();
                    spans.clear();
                    spans.push(union);
                }
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
            // launch, which is the whole point.
            if spans.len() > 1
                && copies.enabled
                && fallback::copies(compiled, &region.mask, copies.bucket)
                && copyable(trace, region)
            {
                let gathered = gather_of(&spans, indptr_host, copies.spaces);
                of_region.push((runs.len() as u32, 1));
                runs.push(seat(&mut windows, gathered));
                continue;
            }

            of_region.push((runs.len() as u32, spans.len() as u32));
            for &span in &spans {
                let window = Window {
                    span,
                    indptr_host: rebase(indptr_host, span),
                    indptr: Tensor::new(0, 0, 1, Dtype::I32),
                    gathered: None,
                    segments_host: segments_host.clone(),
                    segments: Tensor::new(0, 0, 2, Dtype::I32),
                    segment_cap,
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

    /// Every window's `i32` vectors, end to end — what the shell stages in
    /// one copy.
    ///
    /// **ONE BLOB, AND [`bind`](Windows::bind) WALKS IT IN THE SAME ORDER.**
    /// A window contributes its rebased boundaries; a GATHERED one
    /// contributes its row map and its per-space pool tables behind them.
    /// The two functions are written as one traversal in two directions for
    /// exactly the reason the copy's own two halves are: an order that could
    /// drift is an address that could point at another window's vector.
    #[must_use]
    pub fn packed(&self) -> Vec<i32> {
        let mut out: Vec<i32> = Vec::new();
        for window in &self.windows {
            out.extend_from_slice(&window.indptr_host);
            out.extend_from_slice(&window.segments_host);
            let Some(gathered) = &window.gathered else {
                continue;
            };
            out.extend_from_slice(&gathered.rows_host);
            for space in &gathered.spaces {
                out.extend_from_slice(&space.page_indptr_host);
                out.extend_from_slice(&space.page_indices_host);
                out.extend_from_slice(&space.last_page_lens_host);
                out.extend_from_slice(&space.kv_len_host);
            }
        }
        out
    }

    /// Seat the staged vectors: `base` is where [`packed`](Windows::packed)
    /// landed on the device.
    pub fn bind(&mut self, base: u64) {
        let mut at = base;
        // `cols` because a segment list is `[segs][2]` where every other
        // staged vector is `[n][1]`; the byte stride is the same and only the
        // shape the consumer reads it at differs.
        let mut take = |entries: usize, cols: u32| {
            let here = at;
            at += entries as u64 * 4;
            Tensor::new(here, entries as u32 / cols.max(1), cols, Dtype::I32)
        };
        for window in &mut self.windows {
            window.indptr = take(window.indptr_host.len(), 1);
            window.segments = take(window.segments_host.len(), 2);
            let Some(gathered) = &mut window.gathered else {
                continue;
            };
            gathered.rows = take(gathered.rows_host.len(), 1);
            for space in &mut gathered.spaces {
                space.page_indptr = take(space.page_indptr_host.len(), 1);
                space.page_indices = take(space.page_indices_host.len(), 1);
                space.last_page_lens = take(space.last_page_lens_host.len(), 1);
                space.kv_len = take(space.kv_len_host.len(), 1);
            }
        }
    }

    /// How many launches a region costs in this fire — `1` for a window P4
    /// seated, `r` for one it could not, and `1` for an empty window.
    ///
    /// THE SAME NUMBER `engine::fire::walk` LOOPS ON, and it is the same
    /// number because both read it off the same class table: the walk asks
    /// `WindowTable::spans_into` and this asked it once per region when the
    /// table was built. A disagreement would show up as
    /// [`at`](Windows::at)'s panic rather than as a wrong window.
    #[must_use]
    pub fn runs(&self, region: u32) -> u32 {
        self.of_region.get(region as usize).map_or(0, |held| held.1)
    }

    /// How many launches this fire's walk makes over the whole template.
    ///
    /// **THE NUMBER A COPY EXISTS TO LOWER, AND THE ONLY ONE A CALLER CAN SEE
    /// FROM OUTSIDE.** `engine::fire::walk` loops `Windows::runs(region)`
    /// times per region — the same table, read the same way — so this is that
    /// loop's total, known before a single kernel is enqueued. A fire whose
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

    /// The most launches any region of this fire costs — what a per-run table
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
/// [`engine::store::check::no_schedule_straddles_its_readers`] — neutral IR
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

/// **NO GROUPED CONSUMER SHARES ITS WINDOW WITH A PREPARE REGION.**
///
/// The sibling of [`no_schedule_straddles_its_readers`], and it guards the one
/// asymmetry between this shell's two ways out of a split. `Fallback::Copy` is
/// resolved per MASK (`engine::fire::fallback::copies`) precisely so a prepare
/// builder inherits its readers' answer: P4 offers only capture regions to its
/// C1P instance, so a builder standing over the same window is owed no row of
/// its own, and one that split while its reader gathered would hand the single
/// launch a schedule describing the first interval only. `Fallback::Grouped`
/// is resolved per REGION — the launch count is a property of the nodes'
/// kernels, not of the window — so it has no such inheritance, and a prepare
/// builder sharing a grouped consumer's mask would carve `r` schedules while
/// the consumer ran once at run zero.
///
/// It cannot happen today: [`crate::GROUPED`] names `linear.lora_correct`
/// alone, whose region holds no plan build. This refuses the load the day a
/// second op is named and that stops being true, rather than letting the
/// asymmetry be discovered as wrong numbers.
///
/// # Errors
///
/// [`Fault::Straddled`], naming the grouped region's first node and the two
/// class sets.
pub fn no_grouped_window_is_also_a_prepare_window(compiled: &CompiledModel) -> Result<()> {
    for region in compiled.template() {
        if !fallback::grouped(compiled, region.nodes.clone()) {
            continue;
        }
        let Some(builder) = compiled
            .template()
            .iter()
            .find(|other| other.phase == Phase::Prepare && other.mask == region.mask)
        else {
            continue;
        };
        return Err(Fault::Straddled {
            value: builder.nodes.start,
            node: region.nodes.start,
            planned: format!("{:?}", builder.mask.iter().collect::<Vec<_>>()),
            consumed: format!("{:?}", region.mask.iter().collect::<Vec<_>>()),
        });
    }
    Ok(())
}

/// The one rectangle that contains every one of these intervals.
///
/// **IT CONTAINS ROWS THE MASK DOES NOT**, and that is the point: a grouped
/// launch is cut at the union and told which of its rows are its own, so the
/// gaps are addressed by the launch and touched by nobody. Rows and lanes are
/// unioned together because the spans break at the same classes for both
/// (`WindowTable::spans` argues why), and the caller only ever asks this of a
/// list `spans_into` produced, which is ascending and non-empty.
fn union_of(spans: &[MaskSpan]) -> MaskSpan {
    let first = spans.first().copied().unwrap_or_default();
    let last = spans.last().copied().unwrap_or_default();
    MaskSpan {
        row_offset: first.row_offset,
        rows: (last.row_offset + last.rows) - first.row_offset,
        lane_offset: first.lane_offset,
        lanes: (last.lane_offset + last.lanes) - first.lane_offset,
    }
}

/// Give this window a position in the fire's deduplicated list.
///
/// **DEDUPLICATED ON EVERYTHING, NOT ONLY THE SPAN.** A gathered window and a
/// plain one can name the same compacted extent — `{0, 5, 0, 3}` is what a
/// copy of two runs looks like and also what the first three lanes of a fire
/// look like — and they are not the same window: one reads its rows where
/// they lie and the other reads them out of a slab. Comparing the runs beside
/// the span is what keeps a region from being handed the other one's.
fn seat(windows: &mut Vec<Window>, window: Window) -> u32 {
    let same = |held: &Window| {
        held.span == window.span
            && held.segments_host == window.segments_host
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
/// Three things fall out of the run list and nothing else does:
///
/// - **the row map**, which is the runs' rows concatenated in run order. That
///   order is the one the compacted rectangle is in, so it is also the order
///   the lanes and their qo boundaries have to be in — a gather that laid the
///   rows down in one order and the boundaries in another would hand the
///   launch a ragged view of somebody else's requests.
/// - **the qo boundaries**, rebased over the union: each run's per-lane row
///   counts, appended, prefix-summed from 0. Not `rebase(indptr, span)` of any
///   one run — the union is what the single launch stands over.
/// - **the per-space pool tables**, re-cut lane by lane
///   ([`GatheredSpace`] argues why the page-id list is copied).
fn gather_of(runs: &[MaskSpan], indptr_host: &[i32], spaces: &[Geometry]) -> Window {
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

    let gathered_spaces = spaces
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
                page_indptr: Tensor::new(0, 0, 1, Dtype::I32),
                page_indices: Tensor::new(0, 0, 1, Dtype::I32),
                last_page_lens: Tensor::new(0, 0, 1, Dtype::I32),
                kv_len: Tensor::new(0, 0, 1, Dtype::I32),
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
        indptr: Tensor::new(0, 0, 1, Dtype::I32),
        // A COPY AND A GROUPED WINDOW ARE THE TWO WAYS OUT OF A SPLIT AND
        // NEVER THE SAME WAY: a gathered rectangle holds only the consumer's
        // own rows, so there is nothing for a segment list to keep a launch
        // off (`walk`'s rule 4 gives `Grouped` the tie for the same reason).
        segments_host: Vec::new(),
        segments: Tensor::new(0, 0, 2, Dtype::I32),
        segment_cap: 0,
        gathered: Some(Gathered {
            runs: runs.to_vec(),
            rows: Tensor::new(0, 0, 1, Dtype::I32),
            rows_host,
            spaces: gathered_spaces,
        }),
    }
}

/// Where the walk is: which region of the template, and which run of that
/// region's window.
///
/// **TWO NUMBERS, ONE OBJECT, BECAUSE THEY ARE READ TOGETHER.** A `Run`
/// resolves every operand at `windows.at(region, run)`, and a pair that could
/// be handed in separately is a pair that could be handed in from two
/// different walks. The [`Cursor`] writes both — the region before the
/// region's first node, the run before each launch of it — and the `Run`
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

/// The stream handles and events a [`Cursor`] switches between — P6's half of
/// the sink.
///
/// **HANDED IN, NEVER OWNED.** The streams and the events are the context's,
/// opened once at load (`Context::open_lanes`); what this bundle adds is the
/// one cell the [`Run`](crate::run::Run) reads to know which of them the
/// launch it is about to make belongs on. Same mechanism as the region cell
/// beside it, for the same reason: the walk takes two `&mut` and the sink and
/// the dispatch cannot be one object.
#[derive(Debug, Clone, Copy)]
pub struct Lanes<'a> {
    /// The side streams, in stream order: `side[0]` is stream 1. The main
    /// stream is not here — a region on stream 0 is the ordinary case and
    /// needs no lookup.
    pub side: &'a [*mut core::ffi::c_void],
    /// The main stream, which is what an event on stream 0 is recorded on.
    pub main: *mut core::ffi::c_void,
    /// One event per `EventId`, in id order.
    pub events: &'a [Event],
    /// Which stream the walk is on now.
    pub at: &'a Cell<u32>,
}

/// This shell's [`Sink`]: the region counter a [`Run`](crate::run::Run) reads
/// its window out of, and — when the artifact forked — the stream switch and
/// the event points.
///
/// **THE EAGER CURSOR RECORDS NOTHING, LIKE `EagerSink`, AND CARRIES ONE
/// NUMBER.** The walk calls [`region_begin`](Sink::region_begin) for every
/// region of the template in order — including the ones this composition has
/// no rows for, which is what makes the count an index rather than a guess —
/// and a `Run` holding a shared reference to the same `Cell` then resolves
/// each operand at that region's window.
///
/// **[`Cursor::across`] IS THE RECORDING ONE, AND IT IS THE ONLY PLACE A
/// STREAM SWITCH HAPPENS.** A cursor built with [`Cursor::new`] leaves the
/// stream cell at zero forever, which is what makes the eager pass the
/// SERIALIZATION of P6's DAG (`engine::fire::EagerSink`'s doc argues why that
/// is correct rather than merely safe). A cursor built with `across` writes
/// each region's stream into the cell, waits the events the region waits on
/// and records the ones it records — the fork/join pattern
/// `.wiki/tart/evidence/green_contexts.md` Finding 3 measured, in the order
/// `engine::fire::walk` emits it.
///
/// A device call inside a `Sink` method has nowhere to return an error to, so
/// the first one is kept and [`Cursor::settle`] is where the caller asks. That
/// is not a swallowed error: a failed `cudaEventRecord` leaves the capture in
/// a state the caller must not instantiate, and the caller is the code that
/// knows it.
#[derive(Debug)]
pub struct Cursor<'a> {
    at: u32,
    place: &'a At,
    lanes: Option<Lanes<'a>>,
    /// Is this walk being WRITTEN DOWN?
    ///
    /// **NOT THE SAME QUESTION AS "does it have side streams".** A plan with
    /// no fork group captures through a cursor with no [`Lanes`], and it is
    /// still a capture — so the two are separate fields even though today's
    /// artifacts usually set both. What reads it is
    /// [`cond_begin`](Sink::cond_begin), where the difference between the two
    /// modes is the difference between ignoring a conditional (correct) and
    /// recording its body unconditionally (silently wrong).
    recording: bool,
    fault: Option<Fault>,
}

impl<'a> Cursor<'a> {
    /// A cursor writing into `place`, on the main stream from end to end.
    #[must_use]
    pub fn new(place: &'a At) -> Cursor<'a> {
        place.region.set(0);
        place.run.set(0);
        Cursor {
            at: 0,
            place,
            lanes: None,
            recording: false,
            fault: None,
        }
    }

    /// The same cursor, told that what it is walking is being recorded.
    ///
    /// The one thing a capture pass must say about itself that a stream
    /// assignment does not already say — see [`Cursor::recording`].
    #[must_use]
    pub fn writing(self) -> Cursor<'a> {
        Cursor {
            recording: true,
            ..self
        }
    }

    /// The same, plus P6: switch streams at every region boundary and put the
    /// baked event points on the device.
    #[must_use]
    pub fn across(place: &'a At, lanes: Lanes<'a>) -> Cursor<'a> {
        place.region.set(0);
        place.run.set(0);
        lanes.at.set(0);
        Cursor {
            at: 0,
            place,
            lanes: Some(lanes),
            recording: false,
            fault: None,
        }
    }

    /// What the device refused during the walk, if anything.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] from a `cudaEventRecord` or a `cudaStreamWaitEvent`,
    /// or [`Fault::Unbound`] for a template naming a stream or an event this
    /// load never opened — which is a `CompiledModel` and a `Context` that were not
    /// set up from each other.
    pub fn settle(self) -> Result<()> {
        match self.fault {
            Some(fault) => Err(fault),
            None => Ok(()),
        }
    }

    /// The stream the current region is on, or the fault for a region naming
    /// one this load did not open.
    fn stream(&self, lanes: &Lanes<'a>) -> core::result::Result<*mut core::ffi::c_void, Fault> {
        match lanes.at.get() {
            0 => Ok(lanes.main),
            n => lanes
                .side
                .get(n as usize - 1)
                .copied()
                .ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "region {} on stream {n}, and this load opened {}",
                        self.at.saturating_sub(1),
                        lanes.side.len(),
                    ),
                }),
        }
    }

    /// Record or wait one event on the current stream. `record` chooses which.
    fn event(&mut self, id: EventId, record: bool) {
        let Some(lanes) = self.lanes else {
            return;
        };
        // The first fault wins: a later call on a stream whose earlier event
        // failed says nothing new, and the caller wants the one that started
        // it.
        if self.fault.is_some() {
            return;
        }
        let outcome = self.stream(&lanes).and_then(|stream| {
            let Some(event) = lanes.events.get(id.0 as usize) else {
                return Err(Fault::Unbound {
                    what: format!(
                        "event {}, and this load created {}",
                        id.0,
                        lanes.events.len(),
                    ),
                });
            };
            if record {
                event.record(stream)
            } else {
                event.wait(stream)
            }
        });
        if let Err(fault) = outcome {
            self.fault = Some(fault);
        }
    }
}

impl Sink for Cursor<'_> {
    fn region_begin(&mut self, region: &Region) {
        self.place.region.set(self.at);
        self.place.run.set(0);
        self.at += 1;
        // The stream switch, and it is the whole of it: everything the `Run`
        // resolves afterwards fires on whatever this names, until the next
        // region says otherwise.
        if let Some(lanes) = self.lanes {
            lanes.at.set(region.stream);
        }
    }
    fn region_end(&mut self, _region: &Region) {}

    /// **THE SPLIT'S ONE PIECE OF STATE.** A region P4 could not seat runs
    /// once per interval of its class set, and every operand the `Run`
    /// resolves after this call is cut at THAT interval — its rows, its lanes,
    /// its rebased qo boundaries. A cursor that ignored this would hand every
    /// run the first one's window, which is not a fault: it is the first
    /// interval's rows computed `r` times and the rest never computed at all.
    fn run(&mut self, run: u32, _runs: u32) {
        self.place.run.set(run);
    }

    /// **THE EAGER CURSOR IGNORES IT AND THE RECORDING ONE REFUSES IT.**
    ///
    /// Ignoring is correct for an eager pass and it is not a shortcut: the
    /// walk's zero-row rule decides exactly what a conditional decides, at the
    /// same instant, so a fire that walks a conditional region eagerly runs
    /// the same nodes over the same rows (design §4 — conditionals are the
    /// optimization, zero-row always-launch is the semantics). That is what
    /// `engine::fire::EagerSink` says too, and why the two agree.
    ///
    /// A CAPTURE CANNOT IGNORE IT. The graph outlives the fire that recorded
    /// it, so a body recorded outside its conditional node is a body that runs
    /// under every composition the exec is replayed for — and it would
    /// compute. So the recording cursor answers [`Fault::Unlowered`], which
    /// names the region and says what is missing; see that variant for the two
    /// things this shell would need, neither of which is the cudarc binding.
    fn cond_begin(&mut self, lowering: &Lowering) {
        if !self.recording || self.fault.is_some() {
            return;
        }
        self.fault = Some(Fault::Unlowered {
            region: self.at.saturating_sub(1),
            lowering: format!("{lowering:?}"),
        });
    }
    fn cond_arm(&mut self, _arm: u8) {}
    fn cond_end(&mut self) {}
    fn fork(&mut self, event: EventId) {
        self.event(event, true);
    }
    fn join(&mut self, event: EventId) {
        self.event(event, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::fire::{ClassWindow, WindowTable};
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

    /// A region shaped like the one P3 picks: windowed, in the capture phase,
    /// and behind a conditional node.
    fn conditional() -> Region {
        Region {
            nodes: 0..26,
            mask: ClassSet::of([0]),
            phase: model_compiler::Phase::Capture,
            lowering: Lowering::If,
            stream: 0,
            wait: Vec::new(),
            open: None,
            close: None,
            sm_hint: None,
            collective: false,
        }
    }

    #[test]
    fn an_eager_cursor_ignores_a_conditional_and_a_recording_one_refuses_it() {
        let cell = At::new();
        let mut eager = Cursor::new(&cell);
        let region = conditional();
        eager.region_begin(&region);
        eager.cond_begin(&region.lowering);
        eager.cond_end();
        eager.region_end(&region);
        // Correct, and not a shortcut: the walk's zero-row rule decides what
        // the conditional decides, so an eager pass runs the same nodes over
        // the same rows (design §4).
        eager.settle().expect("an eager walk ignores the bracket");

        let cell = At::new();
        let mut recording = Cursor::new(&cell).writing();
        recording.region_begin(&region);
        recording.cond_begin(&region.lowering);
        let fault = recording
            .settle()
            .expect_err("a capture may not record a body outside its node");
        assert!(matches!(fault, Fault::Unlowered { region: 0, .. }), "{fault}");
        assert!(fault.to_string().contains("conditional nodes"));
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

    /// **THE EASIEST THING IN THE SPLIT TO GET SILENTLY WRONG.** A window's qo
    /// boundaries are offsets INTO the rectangle it cuts, so every run of a
    /// fragmented window needs its OWN vector, rebased to its own zero over
    /// its own lanes. Handing run 1 the vector rebased for run 0 does not
    /// fault: the schedule's work items index a boundary list that describes
    /// somebody else's requests, and the answer is wrong logits for every lane
    /// past the first interval.
    #[test]
    fn each_run_of_a_fragmented_window_rebases_its_own_boundaries() {
        // Three classes, and the middle one is not in the mask: 2 prefill
        // lanes of 3 rows, 1 lane of 5, 2 lanes of 4.
        let table = WindowTable::new(vec![
            ClassWindow {
                row_offset: 0,
                rows: 3,
                lane_offset: 0,
                lanes: 2,
            },
            ClassWindow {
                row_offset: 3,
                rows: 5,
                lane_offset: 2,
                lanes: 1,
            },
            ClassWindow {
                row_offset: 8,
                rows: 4,
                lane_offset: 3,
                lanes: 2,
            },
        ]);
        let mask = ClassSet::of([0, 2]);
        assert_eq!(table.span(&mask), Err(2), "class 1's rows stand between");

        let spans = table.spans(&mask);
        assert_eq!(spans.len(), 2);
        assert_eq!((spans[0].row_offset, spans[0].rows), (0, 3));
        assert_eq!((spans[1].row_offset, spans[1].rows), (8, 4));

        // The fire's boundaries, over all five lanes.
        let indptr = [0, 1, 3, 8, 10, 12];
        assert_eq!(rebase(&indptr, spans[0]), vec![0, 1, 3]);
        assert_eq!(
            rebase(&indptr, spans[1]),
            vec![0, 2, 4],
            "the second run starts at ITS zero, not the fire's",
        );
    }
}
