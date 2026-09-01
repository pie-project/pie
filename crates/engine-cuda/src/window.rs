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
//! owes it no row of its own (`model_exec::fire::fallback::copies` argues why the
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
//! # And the live-rows seat, which is a window fact on a different index
//!
//! A recorded graph is carved at a bucket, and a replay of it that means to
//! serve fewer rows than it was carved at cannot say so in a node parameter —
//! the parameters are baked. The bodies design's answer is a device word per
//! launch that the kernels READ (`kernels_cuda::Ctx::arm_stage`, and the
//! `if (win != nullptr && r >= win[0]) return;` every supporting entry
//! carries), and this table is where its host side lives, because the number
//! it holds is a WINDOW's row count and nothing else here knows one.
//!
//! **INDEXED BY REGION, NOT BY WINDOW**, which is why it is [`Windows::live`]
//! and not another vector inside [`Windows::packed`]. The packed blob is one
//! entry per DISTINCT window — deduplicated — and the seat's address has to be
//! a multiplication from the cursor's two `u32`s ([`Windows::live_at`]). Two
//! indices, two carves, one traversal each. Both are laid at a FIXED STRIDE
//! now, and for one reason ([`Slots`]).
//!
//! The value every window contributes is its FULL row count, so a launch that
//! reads it admits every row it was already going to run: arming the seat is
//! arithmetically the identity, and the fire path stages nothing into it until
//! a caller means something else by it.
//!
//! **AND THE SEAT IS FOUR WORDS, TWO PER AXIS** (the chunked-arm wave):
//! `[rows, row_offset, lanes, lane_offset]`. Words 0 and 1 are the row axis
//! and mean exactly what they always meant, which is why they stay first —
//! every guard shipped reads `win[0]` and every shift `win[1]`, and a seat
//! that moved either would turn a correct kernel into a wrong one without
//! touching it. Words 2 and 3 are their LANE twins, read only by a kernel
//! whose grid counts requests rather than rows: `win[2]` retires the lanes a
//! ceiling grid padded in, and `win[3]` turns a window request number into a
//! FIRE lane for the tables a body may not bake a slice of. A kernel that
//! reads neither is unaffected by their existence, which is what makes the
//! widening free.
//!
//! # And why the packed blob is laid out at a fixed stride
//!
//! **A BODY BAKES THESE ADDRESSES, SO THE LAYOUT HAS TO BE A FUNCTION OF THE
//! KEY AND NOT OF THE FIRE.** A recorded body is captured once and replayed
//! for every fire that keys to it (`record::BodyKey`: the bucket and which
//! classes have rows — and NOT the per-class counts). Two
//! consumers bake a window's device address into a graph node: a seated pie
//! kernel takes its window's `indptr` pointer as a launch argument, and
//! `Cursor::count_of` hands the same pointer to a conditional setter. If a
//! window's address were `base + Σ_{j<i} (that fire's j-th window's words)`,
//! only slot 0 would survive a replay — the lane counts move between fires of
//! one key, so every slot behind the first would name somebody else's CSR.
//!
//! So a slot's address is `base + slot * stride`, where the stride is the
//! CEILING one slot can ever need — `max_lanes + 1 + 2 * max_segs` — and the
//! per-window vectors are laid at their slot's own offset with the tail of the
//! slot left as padding. **THE PADDING IS BYTES THAT ALREADY EXISTED**:
//! `Inputs::reserve` has always carved the blob at `slots × that same
//! ceiling`, because the carve is a reserve and a reserve cannot measure a
//! fire it has not seen. [`Slots`] is that arithmetic, written once and read
//! by both halves, so the carve cannot become a ceiling the layout disagrees
//! with.
//!
//! And the SLOT itself is a function of the key: [`seat`] deduplicates on the
//! window's span, and within one `BodyKey` two masks have equal spans in one
//! fire exactly when they have equal spans in every fire of that key — the
//! argument is at [`seat`], and the two host tests at the bottom of this file
//! pin both halves.
//!
//! # And which regions of a fire a body may HOLD (tier 2)
//!
//! Everything above is the shell's answer to "what does this launch run
//! over". [`Windows::admits`] is the one question `record.rs` asks of that
//! answer: per template region, may a captured graph hold this region's
//! launches, or must the body re-issue them ([`Admit`])? The clauses are the
//! ones this header already argued — a gathered window is numbered from a
//! scratch slab's own zero, a grouped one's span is a union with foreign rows
//! in the gaps, and a windowed one whose ops do not all read the seat's
//! `start` wants a pointer the host advanced — and what tier 2 changed is
//! only what a caller SPENDS them on. A region the rule refuses is an
//! ISLAND, not a refused composition: `record::cuts` cuts the template into
//! maximal runs of one answer, the captured stretches become execs and the
//! islands are walked eagerly between them.
//!
//! **AND WHAT A CALLER SPENDS IS THIS TABLE WIDENED** (`record::widen`).
//! Some cuts are not legal — a boundary inside a fork group, one between two
//! arms of a conditional, a schedule on the far side of one from its readers
//! — and the answer to an illegal boundary is to GROW the island until it is
//! legal, never to refuse the composition. That is a pure function of this
//! table and of the template, so everything this header claims about the
//! table being a function of the `record::BodyKey` survives it, and the
//! shell derives it exactly once per key (`Shell::segmentation`) so the
//! `Run`, the capture loop and the cut script cannot read different
//! answers.
//!
//! Which makes one property of this table load-bearing, and it is argued in
//! full at [`Windows::admits`]: **every entry is a function of the
//! `record::BodyKey`** — the present SET and the BUCKET — because a body is
//! captured once and every fire of its key replays the same script. The
//! shell memoizes the table per key on that argument (`Shell::segments`), and
//! `record::Graphs::fire_body` asserts the island list on every hit rather
//! than believing it.
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
//! **AND "EVERY REGION, IN ORDER" IS LITERAL, WHICH IS WHAT MAKES A SEGMENTED
//! CAPTURE WORK.** `model_exec::fire::walk` filters DISPATCH and never
//! structure, so a pass restricted to one segment still announces every
//! region to this cursor — every stream switch, every event record, every
//! event wait. So a region's number means one thing in every pass of one
//! fire (this cursor's whole contract), and every fork pair a segment's
//! capture states is MATCHED, which is the condition
//! `cudaStreamEndCapture` will not finish a graph without
//! (`record::walk_capture_cut` states the cost of it).
//!
//! [`Run`]: crate::run::Run

use std::cell::Cell;

use crate::device::conditional::Kind;
use crate::device::graph::Event;
use kernels_cuda::Tensor;
use model_compiler::{CompiledModel, Lowering, Phase, Region};
use model_exec::fire::{EventId, MaskSpan, Sink, WindowTable, fallback};
use model_exec::store::check::{self, rebase};
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
///
/// **AND A WINDOW HAS A SECOND READING OF THOSE SAME BOUNDARIES, WHICH IS NOT
/// HERE** (bodies design, chunk 2c-a). The vectors above are rebased because
/// the pointer beside them is the WINDOW's first row. A region on
/// [`crate::SHIFTED`] under a body is handed the PLANE's base instead
/// (`Run::cut`, `Run::plane_base`), and a CSR counting from the window's zero
/// beside a pointer counting from the plane's would address rows the launch
/// does not own. That reading is one fire-wide `[lanes + 1]` vector
/// ([`Windows::qo_absolute`]) and it goes over WHOLE:
/// `Run::qo_indptr_absolute` hands back the base and cuts nothing — not even
/// by lane, because a body BAKES that pointer and `lane_offset` moves between
/// fires of one key, so a sliced absolute reading would be stale on every
/// replay but its recording one. Which entries are a launch's requests is the
/// SCHEDULE's business, not the pointer's.
///
/// Which consumer takes which: every seated pie kernel, `Cursor::count_of`
/// and both plan builders take the REBASED pair below; the FA2 params' q axis
/// takes the absolute one, and only where its region moves its own plane.
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
    /// (`model_exec::fire::max_runs`), carried here because it is what sizes the
    /// grid's segment axis and the fire is not allowed to size it
    /// (decision #15).
    pub segment_cap: u32,
    /// Present iff this window is a [`Fallback::Copy`](model_compiler::Fallback)
    /// — the runs it compacts, and everything a consumer needs to read them
    /// as one.
    pub gathered: Option<Gathered>,
    /// **THE SAME MASK'S INTERVAL ON THE SECOND ROW AXIS** — patch rows where
    /// [`span`](Window::span) has token rows, and IMAGES where it has lanes
    /// (multimodal §5.1).
    ///
    /// A REGION HAS BOTH, AND WHICH ONE A VALUE IS CUT AT IS READ OFF ITS
    /// LEADING `Dim`. A tower region's `span` IS this — its rows are the row
    /// count its kernels launch over — but the embed merge is a TOKEN region
    /// that reads a patch rectangle, so a shell that carried only one pair
    /// would have to hand that node the token interval of a patch column.
    /// Zero for every fire of every artifact with no patch axis, which is the
    /// same zero window a class no lane is in has.
    pub patch: MaskSpan,
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

/// **THE FIXED CARVE THE PACKED WINDOW BLOB IS LAID OUT IN** — one slot per
/// distinct window, every slot the same width, and the gathered payloads
/// behind all of them.
///
/// **ONE ARITHMETIC, TWO READERS.** [`Inputs::reserve`](crate::inputs::Inputs::reserve)
/// carves the blob's bytes from this object and [`Windows::packed`] /
/// [`Windows::bind`] place the vectors inside it from the same object, so the
/// reserve cannot become a ceiling the layout disagrees with. That is the
/// whole reason it is a type rather than two expressions that happen to
/// match.
///
/// **AND THE STRIDE IS THE CEILING THE RESERVE WAS ALREADY PAYING.** A window
/// slot holds a rebased `[lanes + 1]` boundary vector and a `[segs][2]`
/// segment list; the most either can ever be is the budget's lane count and
/// the artifact's own `model_exec::fire::max_runs`, and the reserve has always
/// multiplied that ceiling by the slot count because a reserve cannot measure
/// a fire it has not seen. Laying each slot AT that ceiling therefore costs no
/// device byte that was not already carved — it spends the padding the reserve
/// bought — and buys the one thing a recorded body needs: an address that is a
/// function of the slot and not of the fire (this file's header argues why).
///
/// **SLOT COUNT.** `k` classes give at most `k(k+1)/2` contiguous runs of the
/// class order plus the zero window every empty region shares, and a GATHERED
/// window is a slot beyond those — [`seat`] deliberately does not dedupe one
/// against a plain window of the same extent — so the count is
/// `k(k+1)/2 + 1 + fragmentable`. A fire that somehow wanted more is refused
/// by `Inputs::write_host`'s ceiling with a named fault rather than writing
/// past the carve.
///
/// `Slots::default()` is the empty carve, and only the derived
/// [`Windows::default`] mints one — every table a fire walks is handed the
/// load's, from [`Inputs::window_slots`](crate::inputs::Inputs::window_slots).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Slots {
    /// Words per slot: `max_lanes + 1 + 2 * max_segs`.
    stride: u64,
    /// How many slots the carve holds.
    slots: u64,
}

impl Slots {
    /// The carve for one load: `classes` from the artifact's class table,
    /// `lanes` from the budget, `segs` from `model_exec::fire::max_runs` and
    /// `gathered` from `model_exec::fire::fragmentable`.
    #[must_use]
    pub fn new(classes: usize, lanes: u64, segs: u32, gathered: usize) -> Slots {
        Slots {
            stride: lanes + 1 + 2 * u64::from(segs.max(1)),
            slots: (classes * (classes + 1) / 2 + 1 + gathered) as u64,
        }
    }

    /// Words per slot.
    #[must_use]
    pub fn stride(&self) -> u64 {
        self.stride
    }

    /// How many `i32`s the fixed slot region occupies — where the gathered
    /// payloads begin.
    #[must_use]
    pub fn tail(&self) -> u64 {
        self.slots * self.stride
    }

    /// The word offset of one slot's vectors, from the blob's base.
    #[must_use]
    pub fn at(&self, slot: usize) -> u64 {
        slot as u64 * self.stride
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
    /// **THE CARVE [`packed`](Windows::packed) LAYS ITSELF OUT IN** — the load's,
    /// handed down from `Inputs::reserve` so that the blob's offsets and the
    /// blob's bytes are the one arithmetic ([`Slots`]).
    slots: Slots,

    /// **THE LIVE-ROWS SEAT'S HOST SIDE**: FOUR `u32` per (REGION ORDINAL,
    /// run), flat at `4 * (region * max_runs() + run)`, holding that launch's
    /// full geometry — `[rows, row_offset, lanes, lane_offset]`.
    ///
    /// **NOT PART OF [`packed`](Windows::packed), AND THE INDEX IS THE WHOLE
    /// REASON.** The packed blob is one vector per DISTINCT window — [`seat`]
    /// deduplicates, so two regions with the same span share one boundary
    /// vector and the blob is addressed by SLOT. The seat is addressed by
    /// REGION: an address stamped onto a context per region has to be
    /// reachable by a multiplication from the region ordinal, and a
    /// deduplicated table has no such multiplication. Two indices, so two
    /// vectors — both at a fixed stride, for the one reason this file's header
    /// gives.
    ///
    /// **AND THE VALUE IS THE IDENTITY.** A window's full row count is the
    /// number that admits every row the launch already runs over, so a kernel
    /// reading `r >= win[0]` off it retires nothing — and the same holds one
    /// axis over, where `win[2]` is the window's own lane count. That is
    /// deliberate: the seat is built so that arming it changes no arithmetic,
    /// and only a caller that means to serve fewer rows than the graph was
    /// carved at writes anything else here.
    ///
    /// **AND THE CEILINGS DO NOT TOUCH THESE FOUR WORDS, WHICH WAS WORTH
    /// SAYING RATHER THAN ASSUMING.** The ceiling design's Option B carves a
    /// windowed class's SCHEDULE at the key's rungs — its rows, its lanes and
    /// the reach in front of it — and every one of those numbers rides the
    /// plan payload. What is written here stays the FIRE's own four, because
    /// that is what the word means on this side of the seam: the capture is
    /// made over the window's live grid and the seat is what a later, smaller
    /// fire retires it to. A seat that held a ceiling would be a launch told
    /// to run rows no fire brought, which is the exact inversion of what it
    /// is for.
    live_words: Vec<u32>,
    /// The stride [`live_words`](Windows::live_words) is addressed at —
    /// [`max_runs`](Windows::max_runs), held rather than recomputed because
    /// [`live_at`](Windows::live_at) is asked once per DISPATCHED NODE and
    /// `max_runs` is a scan of every region.
    live_stride: u32,
    /// Where [`live`](Windows::live) landed on the device, or `0` for a fire
    /// that staged none — [`bind_live`](Windows::bind_live) is what sets it,
    /// and `0` is the disarmed seat `kernels_cuda::Ctx::stage` answers
    /// `ArgValue::ABSENT` for.
    live_base: u64,

    /// **THE FIRE'S QO BOUNDARIES, UN-REBASED** — `[lanes + 1]`, exactly the
    /// vector [`Windows::of`] was handed and each window's own copy was
    /// rebased out of.
    ///
    /// Kept whole and kept ONCE, because there is nothing per-window about it:
    /// the DEVICE reading is the whole vector at the fire's own base, and the
    /// only lane slice anybody takes is a HOST one
    /// (`Run::qo_indptr_absolute_host`), read by the fire that made it and
    /// baked into nothing. [`qo_absolute`](Windows::qo_absolute) and
    /// [`qo_absolute_host`](Windows::qo_absolute_host) are the two sides.
    qo_absolute_host: Vec<i32>,
    /// Where [`qo_absolute_host`](Windows::qo_absolute_host) landed on the
    /// device, or `0` for a fire that staged none —
    /// [`bind_qo_absolute`](Windows::bind_qo_absolute) sets it, and `0` is the
    /// unbound reading that answers `None` rather than an address past a
    /// carve nobody wrote.
    qo_absolute_base: u64,
    /// **HOW MANY LANES THE STAGED READING OF THAT VECTOR COVERS**, or `0`
    /// for "exactly the ones [`qo_absolute_host`](Windows::qo_absolute_host)
    /// holds" — which is every fire but a bodied one.
    ///
    /// A bodied fire stages a COPY padded out to the key's ladder reach
    /// (`serve::prepare` step 4d), so the device vector reaches further than
    /// the table's own; and since the decode ceiling
    /// (`Run::planning`) a schedule may be carved at that far, which makes a
    /// launch's `q_indptr` a vector with `ceiling + 1` bounds in it. The
    /// table is the one place that states this vector's SHAPE
    /// ([`Handles::qo_absolute`](crate::inputs::Handles::qo_absolute) is a
    /// bare base for that reason), so the shell tells it what it staged
    /// ([`stage_qo_absolute`](Windows::stage_qo_absolute)) and
    /// [`qo_absolute`](Windows::qo_absolute) dresses the wider reading.
    qo_absolute_lanes: u32,
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
/// `model_exec::fire::fallback::copies` states.
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
                    // The second row axis, excluded for `TokensTimes`' reason
                    // and not for a weaker one: `Gathered::rows_host` is a map
                    // of TOKEN rows, and a patch column's rows are a different
                    // row space entirely (multimodal §5.1 — patches and tokens
                    // do not break at the same places). Copying one under a
                    // token map would move the wrong bytes, so a region that
                    // reads a patch rectangle does not copy. A patch-axis
                    // fallback needs a gather map of its own, which is what
                    // the second seriation brings.
                    Some(Dim::Patches | Dim::Images | Dim::ImagesPlus(_)) => false,
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
    /// `slots` is the load's window carve, from
    /// [`Inputs::window_slots`](crate::inputs::Inputs::window_slots) — the
    /// same object the blob's BYTES were reserved from, so that
    /// [`packed`](Windows::packed) lays a slot exactly where the reserve put
    /// one.
    pub fn of(
        trace: &Trace,
        compiled: &CompiledModel,
        classes: &WindowTable,
        patches: &WindowTable,
        indptr_host: &[i32],
        copies: Copies<'_>,
        slots: Slots,
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
            // **WHICH TABLE THIS REGION'S OWN ROWS COME OUT OF** is its
            // capture unit's axis, exactly as the walk reads it — the two
            // have to agree about the run count or the walk's launch loop and
            // this table's window list are cut at different places.
            let axis = compiled
                .units
                .get(compiled.unit_of(at) as usize)
                .copied()
                .unwrap_or(model_ir::RowAxis::PRIMARY);
            match axis {
                model_ir::RowAxis::Tokens => classes.spans_into(&region.mask, &mut spans),
                model_ir::RowAxis::Patches => patches.spans_into(&region.mask, &mut spans),
            }
            // And the OTHER axis's interval, which a token region needs
            // because the embed merge reads a patch rectangle from inside
            // one. A patch window this fire found in pieces has no single
            // rectangle for such a node to read, and that is refused here
            // rather than resolved to the first piece.
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
            let mut segments_host: Vec<i32> = Vec::new();
            // **P4'S MENU IS THE TOKEN AXIS'S MENU, AND IT IS NOT CONSULTED
            // OVER HERE.** `fallback::promised`, `::grouped` and `::copies`
            // all read `CompiledModel::fallback`, which is the row order of
            // the TOKEN seriation; the patch axis carries its own table
            // (`fallback_for(RowAxis::Patches)`), indexed into its own ladder.
            // Asking the token table about a patch region would be judging
            // one seriation's promise against the other's answers. So a patch
            // window this fire finds in pieces takes the plain split — one
            // window per interval, which is what the walk's launch loop turns
            // for either axis — and neither the grouped arm nor the copy arm
            // is offered, because the shell has no patch-axis gather map and
            // says so by name in `copyable`.
            let menu = axis == model_ir::RowAxis::Tokens;
            if menu && spans.len() > 1 {
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
                // (`model_exec::fire::fallback::grouped`) and turns its launch
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
            if menu
                && spans.len() > 1
                && copies.enabled
                && fallback::copies(compiled, &region.mask, copies.bucket)
                && copyable(trace, region)
            {
                let mut gathered = gather_of(&spans, indptr_host, copies.spaces);
                gathered.patch = patch;
                of_region.push((runs.len() as u32, 1));
                runs.push(seat(&mut windows, gathered));
                continue;
            }

            of_region.push((runs.len() as u32, spans.len() as u32));
            for &span in &spans {
                let window = Window {
                    span,
                    // **A PATCH REGION HAS NO REBASED QO BOUNDARIES**, and
                    // that is a statement rather than an omission: this vector
                    // is the TOKEN rectangle's per-lane bounds, and `span` for
                    // a tower region indexes images. Slicing one by the other
                    // would produce a vector that is the right shape and about
                    // the wrong thing. The patch axis's bounds vector is
                    // `RuntimeInput::PatchSegments` — it arrives in the
                    // submission, cut at this window by `Run::cut`, and no op
                    // on that axis asks a window for one.
                    indptr_host: if menu {
                        rebase(indptr_host, span)
                    } else {
                        Vec::new()
                    },
                    indptr: Tensor::new(0, 0, 1, Dtype::I32),
                    gathered: None,
                    segments_host: segments_host.clone(),
                    segments: Tensor::new(0, 0, 2, Dtype::I32),
                    segment_cap,
                    patch,
                };
                runs.push(seat(&mut windows, window));
            }
        }

        let mut table = Windows {
            windows,
            runs,
            of_region,
            slots,
            live_words: Vec::new(),
            live_stride: 0,
            live_base: 0,
            // **THE SAME VECTOR THE REBASE WAS TAKEN OUT OF, KEPT.** Every
            // window above subtracted its own first bound off a slice of this;
            // the second reading is that slice with the subtraction left out,
            // so holding the source is the whole of what it costs.
            qo_absolute_host: indptr_host.to_vec(),
            qo_absolute_base: 0,
            qo_absolute_lanes: 0,
        };
        // **THE LIVE-GEOMETRY SEAT, FILLED WITH THE IDENTITY.** FOUR words per
        // (region, run) at the stride [`Windows::live_at`] multiplies by, and
        // the ORDER IS A CONTRACT: word 0 is the window's row COUNT, word 1 its
        // row START, word 2 its LANE count and word 3 its LANE start. Count
        // first, because the device guards shipped ahead of this seat read
        // `win[0]` as "how many rows are live" and a seat that put the start
        // there would turn every guard into a wrong one; start second, read
        // only by an entry that has learned to address a plane from its base
        // rather than from a pre-shifted pointer.
        //
        // **AND THE LANE PAIR IS SECOND, NOT INTERLEAVED, FOR THE SAME
        // REASON.** Words 0 and 1 mean today exactly what they meant when the
        // seat held nothing else, so every kernel already reading them is
        // untouched by the widening; a kernel whose grid counts REQUESTS reads
        // 2 and 3 instead — `win[2]` retires the lanes a ceiling grid padded
        // in, and `win[3]` turns this window's request number into a fire lane
        // for the tables a body may not bake a slice of (`Run::recurrent_absolute`).
        // Filled with the identity on both axes — the counts are the window's
        // own and the starts its own offsets — so a kernel that reads the seat
        // admits exactly the work the launch was already going to do. Built
        // here, beside the windows it is read off, and staged only by a caller
        // that asks for it (`inputs::Fire::live`).
        let stride = table.max_runs();
        let wide = stride as usize;
        let mut live = vec![0u32; table.of_region.len() * wide * 4];
        for region in 0..table.of_region.len() as u32 {
            for run in 0..table.runs(region) {
                let span = table.at(region, run).span;
                let seat = 4 * (region as usize * wide + run as usize);
                live[seat] = span.rows;
                live[seat + 1] = span.row_offset;
                live[seat + 2] = span.lanes;
                live[seat + 3] = span.lane_offset;
            }
        }
        table.live_words = live;
        table.live_stride = stride;
        Ok(table)
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

    /// Every window's `i32` vectors, at their slot's own offset — what the
    /// shell stages in one copy.
    ///
    /// **SLOT `i` STARTS AT `i * stride` AND NOT WHERE SLOT `i-1` ENDED**, and
    /// the tail of each slot is padding. That is the H2 fix and this file's
    /// header carries the whole of it: a body bakes a window's `indptr`
    /// pointer into a graph node, the lane counts move between fires of one
    /// `record::BodyKey`, and a tightly packed blob therefore has exactly one
    /// address — slot 0's — that survives a replay. At a fixed stride every
    /// slot's address is `base + slot * stride`, which is a function of the
    /// KEY. The bytes are not new: [`Slots`] is the ceiling `Inputs::reserve`
    /// has always multiplied by the slot count.
    ///
    /// A GATHERED window's payload — its row map and its per-space pool
    /// tables — rides BEHIND every slot, packed tight from [`Slots::tail`],
    /// because a gathered region is refused a body outright
    /// ([`covers_fire`](Windows::covers_fire): its rows were compacted into a
    /// plane of their own and no seat word names them), so nothing bakes those
    /// addresses and the reserve pays them per fragmentable mask rather than
    /// per slot.
    ///
    /// **AND [`bind`](Windows::bind) WALKS THE SAME OFFSETS.** The two
    /// functions are written as one traversal in two directions for exactly
    /// the reason the copy's own two halves are: an arithmetic that could
    /// drift is an address that could point at another window's vector.
    ///
    /// **THE FIXED SLOT REGION IS STAGED WHOLE, TAIL AND ALL** — which it was
    /// not until the grid-at-ceiling wave, and the reason it is now is that a
    /// slot's tail acquired a reader. The four chunked recurrent arms count
    /// their requests off the LENGTH of the boundary vector they are handed,
    /// so a bodied fire declares that vector out to the key's lane ceiling
    /// (`Run::ragged_lanes`) — and while the words past `[lanes + 1]` are
    /// never dereferenced (`win[2]` retires those requests before the kernel
    /// reads the CSR), a vector whose declared bytes were last fire's is a
    /// promise this table should not be making. Zeroing them costs
    /// `slots * stride` `i32`s of H2D — a few hundred bytes on any bake — and
    /// buys the wider reading a defined tail. The GATHERED payloads behind
    /// [`Slots::tail`] are still cut off after their last written word, for
    /// the reason the whole blob used to be: nothing declares a length into
    /// them.
    #[must_use]
    pub fn packed(&self) -> Vec<i32> {
        let mut out: Vec<i32> = Vec::new();
        // Each slot is opened by padding out to its own offset. The windows
        // are walked in slot order, so `resize` only ever grows.
        let open = |out: &mut Vec<i32>, at: u64| {
            let at = at as usize;
            debug_assert!(out.len() <= at, "a window overran its slot's stride");
            // GROW ONLY. A table past its carve is a `Fault::Ceiling` at
            // `Inputs::write_host`, and a truncation here would be that fault
            // served as silently wrong bytes instead.
            if out.len() < at {
                out.resize(at, 0);
            }
        };
        for (slot, window) in self.windows.iter().enumerate() {
            open(&mut out, self.slots.at(slot));
            out.extend_from_slice(&window.indptr_host);
            out.extend_from_slice(&window.segments_host);
        }
        // **THE INVARIANT THE RESERVE AND THIS LAYOUT SHARE**: a slot holds a
        // `[lanes + 1]` boundary vector and a `[segs][2]` segment list, and
        // `Slots::stride` is the load's ceiling on exactly that sum. A fire
        // past it would be a fire past `Inputs::reserve`'s own carve, which
        // `Inputs::write_host` refuses by name.
        debug_assert!(
            self.windows.iter().enumerate().all(|(slot, window)| {
                let words = (window.indptr_host.len() + window.segments_host.len()) as u64;
                words <= self.slots.stride() && self.slots.at(slot) < self.slots.tail()
            }),
            "the packed window layout does not fit the carve it was reserved in",
        );
        // Unconditional since the grid-at-ceiling wave — see this function's
        // own note. It was `if any window gathered`, because the gathered
        // payloads are what used to need the fixed region closed behind them.
        open(&mut out, self.slots.tail());
        for window in &self.windows {
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
    ///
    /// The offsets are [`packed`](Windows::packed)'s, arrived at the same way
    /// — slot by slot at the fixed stride, then the gathered payloads behind
    /// the slots.
    pub fn bind(&mut self, base: u64) {
        let slots = self.slots;
        // `cols` because a segment list is `[segs][2]` where every other
        // staged vector is `[n][1]`; the byte stride is the same and only the
        // shape the consumer reads it at differs.
        let take = |at: &mut u64, entries: usize, cols: u32| {
            let here = *at;
            *at += entries as u64 * 4;
            Tensor::new(here, entries as u32 / cols.max(1), cols, Dtype::I32)
        };
        for (slot, window) in self.windows.iter_mut().enumerate() {
            // **THE SLOT'S OWN OFFSET, NOT WHERE THE LAST ONE ENDED.**
            let mut at = base + slots.at(slot) * 4;
            window.indptr = take(&mut at, window.indptr_host.len(), 1);
            window.segments = take(&mut at, window.segments_host.len(), 2);
        }
        let mut at = base + slots.tail() * 4;
        for window in &mut self.windows {
            let Some(gathered) = &mut window.gathered else {
                continue;
            };
            gathered.rows = take(&mut at, gathered.rows_host.len(), 1);
            for space in &mut gathered.spaces {
                space.page_indptr = take(&mut at, space.page_indptr_host.len(), 1);
                space.page_indices = take(&mut at, space.page_indices_host.len(), 1);
                space.last_page_lens = take(&mut at, space.last_page_lens_host.len(), 1);
                space.kv_len = take(&mut at, space.kv_len_host.len(), 1);
            }
        }
    }

    /// The carve this table lays its packed blob out in — what a caller that
    /// wants a window's offset without a device base asks.
    #[must_use]
    pub fn slots(&self) -> Slots {
        self.slots
    }

    /// The live-rows seat's words, in the order [`live_at`](Windows::live_at)
    /// addresses them — four per (region, run), `[rows, row_offset, lanes,
    /// lane_offset]` — what a shell that means to stage the seat hands
    /// `inputs::Fire::live`.
    ///
    /// A SECOND BLOB AND NOT A TAIL OF [`packed`](Windows::packed), and the
    /// index is the reason: this one is addressed by region ordinal and that
    /// one by deduplicated window slot.
    #[must_use]
    pub fn live(&self) -> &[u32] {
        &self.live_words
    }

    /// Seat the live-rows words: `base` is where [`live`](Windows::live)
    /// landed on the device, or `None` for a fire that staged none.
    ///
    /// Separate from [`bind`](Windows::bind) because it seats a separate
    /// carve — `bind` walks one packed blob in `packed`'s own order, and a
    /// second base threaded through that traversal would be a cursor with
    /// nothing to do.
    pub fn bind_live(&mut self, base: Option<u64>) {
        self.live_base = base.unwrap_or(0);
    }

    /// The fire's qo boundaries with nothing subtracted, host side — what a
    /// shell that means to stage the second reading stages, and what a
    /// window's lane slice of that reading is taken out of.
    ///
    /// **THIS VECTOR IS THE FIRE'S OWN LANES AND STAYS THEM.** What the shell
    /// hands `inputs::Fire::qo_absolute` on the bodies path is a COPY padded
    /// out to the key's LADDER REACH (`serve::prepare` step 4d): the
    /// device tail is what a ceiling plan will read, and this vector is what
    /// every window's rebased slice was cut from and what
    /// `Run::qo_indptr_absolute_host` slices — so the padded BYTES belong to
    /// the staging and not to the table. What the table does keep is the
    /// padded LENGTH ([`qo_absolute_lanes`](Windows::qo_absolute_lanes)),
    /// because the shape of the device reading is this table's to state and
    /// a ceiling-carved schedule reads out to that lane. So this vector
    /// stays `[fire lanes + 1]` — it is what tells `Run::planning` how many
    /// lanes the fire actually brought — and
    /// [`qo_absolute`](Windows::qo_absolute)'s rectangle is the wider one.
    #[must_use]
    pub fn qo_absolute_host(&self) -> &[i32] {
        &self.qo_absolute_host
    }

    /// Seat that vector: `base` is where
    /// [`qo_absolute_host`](Windows::qo_absolute_host) landed on the device,
    /// or `None` for a fire that staged none.
    ///
    /// Its own call for [`bind_live`](Windows::bind_live)'s reason — a third
    /// carve is a third base, and threading one through `bind`'s slot walk
    /// would be a cursor with nothing to advance.
    pub fn bind_qo_absolute(&mut self, base: Option<u64>) {
        self.qo_absolute_base = base.unwrap_or(0);
    }

    /// **AND HOW MANY LANES OF IT WENT OVER** — the key's ladder reach on the
    /// bodies path, and never asked at all on any other, where the staged
    /// vector is the table's own.
    ///
    /// Its own call rather than an argument to
    /// [`bind_qo_absolute`](Windows::bind_qo_absolute) because the two facts
    /// are known at two instants: the padding is decided in `prepare`, beside
    /// the lane tables it pads with, and the base is not known until the
    /// staging slot has committed. Never shrinks, for
    /// `model_exec::store::kv::Geometry::pad_to`'s reason — the vector the
    /// shell staged is this table's own vector plus a tail.
    pub fn stage_qo_absolute(&mut self, lanes: u32) {
        self.qo_absolute_lanes = self.qo_absolute_lanes.max(lanes);
    }

    /// **THE WHOLE FIRE'S BOUNDARIES, ABSOLUTE**, or `None` for a fire that
    /// staged none — and WHOLE is the contract rather than a convenience.
    /// `Run::qo_indptr_absolute` hands this straight to a launch without
    /// cutting it: a body bakes the pointer, this fire's LIVE `lane_offset`
    /// is not a function of a `record::BodyKey` (the ladder fixes a bound on
    /// it, not the number), and a sliced absolute vector would therefore be
    /// stale on every replay but its recording one. The fire's own base is the
    /// only address that holds.
    ///
    /// `[fire lanes + 1]` rows of one `i32` column, or `[ceiling + 1]` for a
    /// bodied fire that padded it (see
    /// [`qo_absolute_lanes`](Windows::qo_absolute_lanes)).
    #[must_use]
    pub fn qo_absolute(&self) -> Option<Tensor> {
        if self.qo_absolute_base == 0 || self.qo_absolute_host.is_empty() {
            return None;
        }
        Some(Tensor::new(
            self.qo_absolute_base,
            // **AT WHAT WAS STAGED, WHICH IS THE TABLE'S OWN LENGTH UNTIL A
            // BODIED FIRE PADS IT** — see
            // [`qo_absolute_lanes`](Windows::qo_absolute_lanes). A launch
            // whose schedule was carved at the key's lane ceiling reads
            // `q_indptr` out to that lane, and `kernels_cuda::attn`'s door
            // (`lanes_carry`) refuses a vector that does not say it reaches
            // there.
            (self.qo_absolute_lanes + 1).max(self.qo_absolute_host.len() as u32),
            1,
            Dtype::I32,
        ))
    }

    /// **THE ADDRESS OF ONE (REGION, RUN)'S LIVE-GEOMETRY WORDS** — its
    /// `[rows, row_offset, lanes, lane_offset]`, in that order — or `0` for a
    /// fire that bound no seat.
    ///
    /// A multiplication and not a lookup: `live_base + 16 * (region *
    /// max_runs + run)`. That is what lets [`Run::ctx`](crate::run::Run) stamp
    /// it onto a context with the walk's cursor already on the node — the
    /// cursor holds two `u32`s and this turns them into an address without
    /// consulting the deduplicated window table.
    ///
    /// `0` is the disarmed seat, and it is the same `0` `Ctx::stage` answers
    /// `ArgValue::ABSENT` with — so an unbound fire's launches take the null
    /// pointer they have always taken. A region or run this table does not
    /// hold answers `0` for the same reason rather than an address past the
    /// carve.
    #[must_use]
    pub fn live_at(&self, region: u32, run: u32) -> u64 {
        if self.live_base == 0 {
            return 0;
        }
        if region as usize >= self.of_region.len() || run >= self.live_stride {
            return 0;
        }
        self.live_base + 16 * (u64::from(region) * u64::from(self.live_stride) + u64::from(run))
    }

    /// How many launches a region costs in this fire — `1` for a window P4
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

    /// How many launches this fire's walk makes over the whole template.
    ///
    /// **THE NUMBER A COPY EXISTS TO LOWER, AND THE ONLY ONE A CALLER CAN SEE
    /// FROM OUTSIDE.** `model_exec::fire::walk` loops `Windows::runs(region)`
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

    /// **WHAT ONE RECORDED BODY MAY BE REPLAYED OVER** — the admissibility
    /// rule of the bodies path (`record::BodyKey`), asked of the WHOLE table
    /// and asked BEFORE a walk rather than during one.
    ///
    /// **THE SEAT HAS TWO WORDS AND THEY RETIRE TWO DIFFERENT THINGS.** A
    /// body is a graph recorded at one bucket and replayed for every fire
    /// that keys to it, and the only thing that may differ between those
    /// fires is what the staged `(count, start)` seat says. The COUNT retires
    /// a TAIL: a launch's grid was carved at the bucket, and a kernel that
    /// reads `win[0]` drops the blocks past the live rows, which is why a
    /// shorter fire is served as a prefix of a longer one. The START moves a
    /// BASE: a kernel that also reads `win[1]` addresses `start + r` off a
    /// pointer handed at the plane's own base, which is what lets a region
    /// that does NOT begin at the fire's row zero be replayed at all. Two
    /// words, two clauses, and a region is admissible under whichever of them
    /// its ops actually read.
    ///
    /// **SO THERE ARE TWO READINGS OF THIS QUESTION, AND THIS IS THE NARROW
    /// ONE.** [`covers_fire`](Windows::covers_fire) waives nothing: every
    /// present region's window must BE the whole fire — `row_offset == 0`,
    /// `rows >= the fire's` — which is `Run::whole_fire`'s clauses one for
    /// one, differing only in the quantifier and the instant. In practice it
    /// is `true` for a SINGLE-CLASS composition and false the moment two
    /// classes have rows, because a two-class fire is exactly a fire whose
    /// regions are windowed.
    /// [`covers_fire_shifted`](Windows::covers_fire_shifted) is the wide
    /// reading: it waives the offset and the rows for a region whose every op
    /// reads BOTH seat words, and waives nothing else.
    ///
    /// **AND TWO CLAUSES NO SEAT WORD CAN EXPRESS, WAIVED BY NEITHER.** A
    /// GATHERED window is its own plane: its rows were compacted out of the
    /// fire's into a fresh rectangle, so `start + r` names a row of that copy
    /// and not of the fire, and there is no offset into the fire that would
    /// make the two agree. A GROUPED window is a union with foreign gaps: its
    /// span covers rows belonging to somebody else, and one `(count, start)`
    /// pair describes an interval, which a union of intervals is not. Both are
    /// refusals about the SHAPE of the rows, and the seat only ever says where
    /// an interval of them begins and how many are live.
    ///
    /// A region with NO rows is not asked, in either reading — an absent
    /// window is not a window whose geometry a launch reads, and its absence
    /// is a fact about the composition, which is the thing a caller keys on
    /// rather than the thing it has to check.
    ///
    /// **WHO ASKS, AND WHY IT CANNOT WAIT FOR THE WALK.** The seat has to be
    /// staged in `prepare`, before the first stream touch, and
    /// `Run::whole_fire` first answers halfway through the walk. So the same
    /// question is asked twice, of one table, from two instants: this is the
    /// host's reading and `Run::whole_fire` is the launch's, and the day they
    /// disagreed the launch's would win — it arms the seat per region, and a
    /// region it declines simply arms nothing.
    #[must_use]
    pub fn covers_fire(&self, rows: u32) -> bool {
        (0..self.of_region.len() as u32).all(|region| {
            (0..self.runs(region)).all(|run| {
                let window = self.at(region, run);
                window.span.rows == 0
                    || (window.span.row_offset == 0
                        && window.span.rows >= rows
                        && window.gathered.is_none()
                        && window.segs() == 0)
            })
        })
    }

    /// **THE SAME QUESTION, ASKED OF A TABLE WHOSE REGIONS CAN MOVE THEIR OWN
    /// BASE** — [`covers_fire`](Windows::covers_fire) with the offset and the
    /// rows waived per region, and nothing else waived at all.
    ///
    /// `shifted[region]` is `exports::regions_shifting`'s answer:
    /// every op in that template region is on [`crate::SHIFTED`], so given the
    /// plane's base pointers and an armed seat it computes over rows
    /// `[start, start + count)` and touches no other row. For such a region
    /// the two INTERVAL clauses are what the seat says, not what the window
    /// has to be, so a windowed one is admissible. The two SHAPE clauses —
    /// gathered, and a segment list — are never waived, for the reason the
    /// doc above gives: neither is an interval, and the seat has no word for
    /// anything else. The zero-row exemption is unchanged.
    ///
    /// A region index past the end of `shifted` reads as NOT shifting. The
    /// slice is the load's, one entry per template region, and the table is
    /// cut from the same template; a short slice is a shell that has lost
    /// track of which is which, and the safe answer to that is the narrow one.
    ///
    /// **THIS IS THE `bodied` PREDICATE'S READING NOW** (chunk 2b-ii), and it
    /// is only sound because the launch plane moved with it. A windowed region
    /// admitted here is one `Run::plane_base` hands the PLANE's base pointers
    /// and `Run::live_at` arms the seat for, so `start + r` is what addresses
    /// its rows; admitting one while the launch plane still pre-shifted would
    /// have captured advanced pointers under a disarmed seat and replayed them
    /// at a different row split — the right number of rows read from the wrong
    /// place, silently. Both halves read this same `shifted` slice
    /// (`Shell::shifted`), which is what keeps them one answer.
    ///
    /// [`covers_fire`](Windows::covers_fire) stays as the narrow reading and
    /// as this one's own argument: it is what this waives clauses OF, and the
    /// tests below diff the two.
    ///
    /// **AND SINCE THE TIER-2 CAMPAIGN THIS IS A READING OF
    /// [`admits`](Windows::admits) RATHER THAN AN ARITHMETIC OF ITS OWN** —
    /// "every region is [`Admit::Captured`]". The clauses did not move and
    /// nothing about the answer changed; what changed is that a caller can
    /// now ask WHERE the table stops being uniform, which is where a
    /// segmented capture is cut. This form survives because a load still has
    /// one question that wants the collapse: whether the whole composition
    /// fits in one graph, which is the fire that pays no eager launch at all.
    #[must_use]
    pub fn covers_fire_shifted(&self, rows: u32, shifted: &[bool]) -> bool {
        (0..self.of_region.len() as u32)
            .all(|region| self.admit(region, rows, shifted) == Admit::Captured)
    }

    /// **WHICH REGIONS OF THIS FIRE A BODY MAY HOLD, AND WHICH ONES IT MUST
    /// RE-ISSUE** — the admissibility rule answered PER REGION instead of
    /// collapsed to one `bool` (the tier-2 campaign).
    ///
    /// [`covers_fire_shifted`](Windows::covers_fire_shifted) is this table
    /// read as "every entry is [`Admit::Captured`]", and until tier 2 that
    /// was the only reading anybody wanted: a composition with one
    /// unrecordable region was a composition no body served, so the fire
    /// walked and was counted. Tier 2 keeps the arithmetic and spends the
    /// answer differently — the CAPTURED stretches are recorded as graph
    /// segments and the ISLANDS between them are re-issued eagerly on the
    /// same stream — so what the callers need is not whether the table is
    /// uniform but WHERE it changes.
    ///
    /// # This table is a function of the [`record::BodyKey`](crate::record::BodyKey), which is what makes segmentation stable
    ///
    /// A body is captured once and replayed by every fire of its key, so the
    /// CUTS have to be the same cuts for every one of those fires — a body
    /// whose second fire wanted its islands somewhere else would be a body
    /// replaying somebody else's launches. They are, and each of the three
    /// clauses is a key function for its own reason:
    ///
    /// * **gathered** is `fallback::copies`' answer, which reads the artifact
    ///   and the fire's BUCKET (`model_exec::fire::fallback::copies` is
    ///   bucket-keyed, `model_compiler::layout`'s `CROSSOVER_ROWS`) — and the
    ///   bucket is the key's first coordinate. The `copyable` scan beside it
    ///   reads the trace alone;
    /// * **a segment list** is `fallback::grouped`'s answer, which reads the
    ///   artifact and nothing else;
    /// * **the interval clauses** — whether a window IS the whole fire, and
    ///   whether it is in pieces at all — are functions of the PRESENT SET
    ///   and of the artifact ([`seat`]'s note argues it: two masks resolve to
    ///   the same span exactly when their present classes are the same set),
    ///   and the present set is the key's second coordinate;
    /// * **`shifted`** is `exports::regions_shifting` read once at LOAD.
    ///
    /// So `(present set, bucket)` — which IS the [`record::BodyKey`](crate::record::BodyKey) — decides
    /// every entry here, and a capture may freeze the cuts it derives.
    /// `record::Graphs::fire_body` asserts it on every hit rather than merely
    /// believing it: the segments a fire derives are compared against the
    /// ones its resident body was cut at.
    ///
    /// # One clause of that proof has a hole, and it is [`Copies::enabled`]
    ///
    /// The gathered clause reads `fallback::copies`, and `fallback::copies` is
    /// asked only when this table was built with copies ENABLED — which is
    /// `[engine] fallback_copy`, a load constant, AND "did this fire stage
    /// mask bits", which is not one. A masked fire takes the split
    /// ([`Copies::enabled`]'s own note says why: the mask slab is addressed by
    /// absolute per-lane byte offsets, so a gather would have to compact it
    /// too). So on a deployment with a masked axis AND a P4 copy row at some
    /// bucket, two fires of ONE key can answer this table differently: the
    /// unmasked one calls the copy region an [`Admit::Island`], the masked one
    /// calls it [`Admit::Captured`], and a body captured under the second is
    /// one the first would replay whole over launches baked for the split.
    ///
    /// **IT IS UNREACHABLE TODAY AND IT IS NOT GUARDED.** `crate::GROUPED`
    /// and the copy table are exercised on qwen texts, which declare no masked
    /// axis at all, so no fire in the tree presents the pair — which is why
    /// this reads as a note rather than as a clause. Two answers are on the
    /// table for the day it does: put the word in the [`record::BodyKey`](crate::record::BodyKey), or
    /// refuse a body to a fire whose copy answer is the FIRE's rather than the
    /// LOAD's (`bodied && !(copies && any lane masked)`), which costs bodies
    /// on masked fires of a copying SKU and costs nothing anywhere else.
    /// `record::Graphs::fire_body`'s `debug_assert` is what would find it, and
    /// `Shell::segments` stores the word beside each memoized table so that a
    /// memo can never be the thing that hides it.
    ///
    /// **AND A REGION IS ONE OR THE OTHER, NEVER PART OF EACH.** The question
    /// is asked of every RUN and the region takes the conjunction, because a
    /// cut is a boundary between template regions and there is no way to
    /// record half of one. A split region's runs all answer alike anyway —
    /// `Windows::of` gives a gathered or grouped region exactly one run, so a
    /// region with several has neither shape — but the conjunction is written
    /// rather than inferred, because the day a fire produced a mixed region
    /// the safe answer is the island and the inferred one would be the
    /// segment.
    ///
    /// A region with NO rows is [`Admit::Captured`], for the reason the bool
    /// reading exempted it: an absent window is not a window whose geometry a
    /// launch reads. It dispatches nothing in either pass, so which side of a
    /// cut it lands on is immaterial — and putting it on the CAPTURED side is
    /// what keeps an empty region from splitting one segment into two.
    #[must_use]
    pub fn admits(&self, rows: u32, shifted: &[bool]) -> Vec<Admit> {
        (0..self.of_region.len() as u32)
            .map(|region| self.admit(region, rows, shifted))
            .collect()
    }

    /// One region's entry of [`admits`](Windows::admits) — the clauses, in one
    /// place, so that the table and the `bool` cannot part.
    #[must_use]
    fn admit(&self, region: u32, rows: u32, shifted: &[bool]) -> Admit {
        let moves = shifted.get(region as usize).copied().unwrap_or(false);
        let held = (0..self.runs(region)).all(|run| {
            let window = self.at(region, run);
            window.span.rows == 0
                || (window.gathered.is_none()
                    && window.segs() == 0
                    && (moves || (window.span.row_offset == 0 && window.span.rows >= rows)))
        });
        if held { Admit::Captured } else { Admit::Island }
    }
}

/// **MAY A BODY HOLD THIS REGION'S LAUNCHES, OR MUST IT RE-ISSUE THEM?** —
/// one entry of [`Windows::admits`], one per TEMPLATE region (the tier-2
/// campaign).
///
/// Two words and no third, because the seat has two words and the question is
/// exactly whether they can speak for this region's rows. What is not here is
/// a "why": a caller that has to act on the answer acts on it identically
/// whichever clause failed, and the clauses themselves are argued once, at
/// [`Windows::admits`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admit {
    /// **A GRAPH MAY HOLD IT.** Every run of this region either has no rows
    /// at all, or is a window the staged `(count, start)` seat can speak for:
    /// it IS the whole fire, or its every op reads the seat's start
    /// (`crate::SHIFTED`) and the launch takes the plane's base. A capture of
    /// such a region is replayable at every split of its key, because the
    /// only thing that moves between two fires of one key is what the seat
    /// says.
    Captured,
    /// **IT HAS TO BE RE-ISSUED, EVERY FIRE.** Some run of this region is
    /// gathered (its rows were compacted into a scratch slab and numbered
    /// from that slab's own zero), or carries a segment list (its span is a
    /// union with foreign rows in the gaps), or is windowed without every op
    /// reading the seat's start (so its launch wants a pre-shifted pointer,
    /// which is a fire's address and not a key's).
    ///
    /// **AND "RE-ISSUED" IS THE WHOLE OF WHAT IT COSTS.** The region walks
    /// eagerly between the execs around it, at this fire's own live geometry
    /// — no ceiling, no seat, no plane base (`Run::captured` is the one gate
    /// all three hang off) — which is byte for byte the launch the eager path
    /// would have made. What it gives up is the launch overhead of its own
    /// nodes and P6's overlap across its span; what it buys is every other
    /// region of the composition replaying from a graph.
    Island,
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

/// **NO GROUPED CONSUMER SHARES ITS WINDOW WITH A PREPARE REGION.**
///
/// The sibling of [`no_schedule_straddles_its_readers`], and it guards the one
/// asymmetry between this shell's two ways out of a split. `Fallback::Copy` is
/// resolved per MASK (`model_exec::fire::fallback::copies`) precisely so a prepare
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
///
/// **AND THE POSITION IT GIVES IS A FUNCTION OF THE `BodyKey`, NOT OF THE
/// FIRE** — which is the second half of the H2 fix, and it holds of the span
/// comparison as written rather than needing a different key. The argument:
///
/// * a `record::BodyKey` fixes WHICH CLASSES HAVE ROWS (and the bucket, and
///   the ceiling each class is carved to); it deliberately drops the
///   per-class COUNTS, which is the whole of what the bodies path buys. Only
///   the presence half is read below, so the ceiling design's Option B —
///   which added the rungs beside the presences — left this argument exactly
///   where it was, and the tier-1 collapse, which made those rungs functions
///   of the bucket, cannot touch it either;
/// * `WindowTable::spans_into` skips a zero-row class and merges the rest by
///   adjacency, so the span list a mask resolves to is determined by
///   `mask ∩ present` — the counts move the OFFSETS, never which classes
///   contribute or where the runs break;
/// * every present class has rows, so the prefix sums are strictly
///   increasing: two masks resolve to the same span exactly when their
///   present classes are the same set. That is a fact about the KEY. Two
///   masks that share a slot in one fire of a key share it in every fire of
///   that key, and two that do not, never do.
///
/// So the dedupe is a re-encoding of `mask ∩ present` and not a coincidence of
/// this fire's arithmetic — which is also why the slot COUNT stays inside the
/// `k(k+1)/2 + 1` the reserve carves: each equivalence class is one contiguous
/// run of the class order. Keying on the raw mask instead would split two
/// masks that differ only in classes this key has no rows for, and buy nothing
/// for slots the carve then has to grow for.
///
/// (The patch axis is the caveat: a patch region's spans come from the patch
/// table, whose class presence `BodyKey` does not carry. A fire with a patch
/// rectangle does not reach the bodies path today; the day it does, the key is
/// what has to learn about the second axis.)
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
        // The caller fills this from the patch table: `gather_of` compacts a
        // TOKEN rectangle and knows nothing about the second axis (a patch
        // column is not copyable at all — `copyable` says so by name).
        patch: MaskSpan::default(),
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

/// **THE SENTINEL `Lanes::at` CARRIES WHILE A CONDITIONAL BODY IS OPEN.**
///
/// A stream index, like every other value in that cell, and deliberately one
/// no artifact can name: `model_compiler::stream` numbers streams from zero
/// upward and a plan that asked for four billion of them would have been
/// refused at load. What reads it is [`Run::ctx`](crate::run::Run) — the same
/// lookup that picks a side stream — so a body's launches land on the stream
/// its capture was begun on without a second mechanism.
pub const BODY: u32 = u32::MAX;

/// What a [`Cursor`] needs to put a conditional node in the graph it is
/// recording (palo design §4).
///
/// **HANDED IN, NEVER OWNED**, exactly as [`Lanes`] is: the streams and the
/// context are the load's, the windows are this fire's, and the cell is the
/// one the [`Run`](crate::run::Run) already reads. A cursor without this
/// bundle is a cursor that still refuses a conditional by name — which is what
/// a shell whose artifact holds none never has to build.
#[derive(Clone, Copy)]
pub struct Conditionals<'a> {
    /// The stream the parent capture is on: where the handle is minted, the
    /// setter is launched and the node is placed.
    pub main: *mut core::ffi::c_void,
    /// The stream a body is captured on — opened at load by
    /// `Context::open_conditional`, and never enqueued on outside a
    /// `cuStreamBeginCaptureToGraph`.
    pub body: *mut core::ffi::c_void,
    /// The kernel context on [`main`](Conditionals::main), which is where the
    /// device-side setter's one launch goes.
    pub setter: &'a kernels_cuda::Ctx,
    /// This fire's windows: the setter reads a region's row count out of the
    /// staged boundary vector this table addresses.
    pub windows: &'a Windows,
    /// Which stream the walk is on — the same cell [`Lanes::at`] carries, so
    /// that a load with side streams and a load without one both have exactly
    /// one of them.
    pub at: &'a Cell<u32>,
}

/// What a [`Cursor`] needs to ROTATE A SLOT'S CONTENTS at a region boundary
/// (alto streaming §3 item 4, D2b).
///
/// **HANDED IN, NEVER OWNED**, exactly as [`Lanes`] and [`Conditionals`] are:
/// the rotor is the LOAD's — its slots are pointer-stable for the life of the
/// load, which is the whole reason a weight row may name one — and the stream
/// is this fire's compute stream, the one the region's launches are about to
/// go on.
///
/// **ONLY AN EAGER CURSOR IS GIVEN ONE.** A captured graph is replayed without
/// the walk that owns the pump's issue cursor, so a recording cursor that
/// pumped would bake one fire's copies into a graph replayed for every fire of
/// its key; the router refuses to record a rotating load at all instead
/// (`crate::rotate`'s header carries the argument, and `serve.rs` is where the
/// decline is taken).
#[derive(Clone, Copy)]
pub struct Pump<'a> {
    /// The load's rotor: the slots, the two event rings, and the copy stream.
    pub rotor: &'a crate::rotate::Rotor,
    /// The stream this fire's launches are on — where `free` is recorded and
    /// `ready` is waited.
    pub compute: *mut core::ffi::c_void,
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
/// SERIALIZATION of P6's DAG (`model_exec::fire::EagerSink`'s doc argues why that
/// is correct rather than merely safe). A cursor built with `across` writes
/// each region's stream into the cell, waits the events the region waits on
/// and records the ones it records — the fork/join pattern
/// `.wiki/tart/evidence/green_contexts.md` Finding 3 measured, in the order
/// `model_exec::fire::walk` emits it.
///
/// A device call inside a `Sink` method has nowhere to return an error to, so
/// the first one is kept and [`Cursor::settle`] is where the caller asks. That
/// is not a swallowed error: a failed `cudaEventRecord` leaves the capture in
/// a state the caller must not instantiate, and the caller is the code that
/// knows it.
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
    /// The conditional machinery, when this load opened any — see
    /// [`Conditionals`]. `None` is a cursor that refuses a conditional region
    /// by name, which is what every load whose artifact holds none gets.
    cond: Option<Conditionals<'a>>,
    /// The bracket currently open, the stream to put the walk back on when it
    /// closes, and whether a body capture is running right now.
    ///
    /// `Some` only between [`cond_begin`](Sink::cond_begin) and
    /// [`cond_end`](Sink::cond_end) — which for a `SWITCH` spans `arms`
    /// regions, because a group's arms are consecutive regions under one node.
    ///
    /// **THE THIRD FIELD IS NOT REDUNDANT WITH THE FIRST.** An `IF` opens its
    /// node and begins its body in one breath; a `SWITCH` opens its node in
    /// `cond_begin` and does not begin a body until `cond_arm(0)`, and between
    /// arms it is briefly open with no body. It is also what keeps the stream
    /// cell honest when a `begin_body` refuses: the cell says `BODY` only while
    /// a capture is actually running on that stream.
    open: Option<(crate::device::conditional::Conditional, u32, bool)>,
    /// The rotating dense pump, when this load armed one — see [`Pump`].
    /// `None` is every load whose weights are where the fire expects them,
    /// and the region seam below is then the line it was before D2b.
    pump: Option<Pump<'a>>,
    fault: Option<Fault>,
}

/// **BY HAND BECAUSE ONE FIELD IS A KERNEL CONTEXT**, which is a stream and
/// three opaque handles and derives nothing. What a reader wants from a cursor
/// is where it stands and what it is doing, and that is all four of these.
impl core::fmt::Debug for Cursor<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Cursor")
            .field("at", &self.at)
            .field("recording", &self.recording)
            .field("conditionals", &self.cond.is_some())
            .field("pump", &self.pump.is_some())
            .field("open", &self.open.is_some())
            .field("fault", &self.fault)
            .finish()
    }
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
            cond: None,
            open: None,
            pump: None,
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
            cond: None,
            open: None,
            pump: None,
            fault: None,
        }
    }

    /// The same cursor, told where to put a conditional node.
    ///
    /// **ADDITIVE, AND ONLY A RECORDING WALK IS GIVEN IT.** An eager pass
    /// ignores the bracket and is right to (design §4 — the walk's zero-row
    /// rule decides what the conditional decides, at the same instant), so
    /// handing it this bundle would be minting a graph handle on a stream that
    /// is not capturing. A recording walk that is NOT given it is the shell
    /// this campaign started with, and it still answers
    /// [`Fault::Unlowered`].
    #[must_use]
    pub fn conditionals(self, cond: Conditionals<'a>) -> Cursor<'a> {
        Cursor {
            cond: Some(cond),
            ..self
        }
    }

    /// **The same cursor, told to rotate this load's dense slots** (alto
    /// streaming §3 item 4).
    ///
    /// **ADDITIVE, AND ONLY AN EAGER PASS IS GIVEN IT.** The rotation's
    /// backpressure is a HOST cursor advanced at each region boundary, and a
    /// replayed graph has no walk to advance it; a recording cursor handed one
    /// would enqueue copies into the capture and bake a single fire's ring
    /// state into a graph that outlives it. `crate::rotate`'s header carries
    /// the whole argument; the decline itself is taken where the mode is
    /// chosen, so that a rotating load never reaches a recording walk at all.
    #[must_use]
    pub fn pumping(self, pump: Pump<'a>) -> Cursor<'a> {
        Cursor {
            pump: Some(pump),
            ..self
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
    pub fn settle(mut self) -> Result<()> {
        // **A BRACKET LEFT OPEN IS CLOSED HERE AND NOT LEFT TO THE DROP.** The
        // walk returns early on a plan that names a node the trace lacks, and
        // that path runs no `cond_end`; a body stream left mid-capture answers
        // every later call with `cudaErrorStreamCaptureUnjoined` for the rest
        // of the process, which would turn one bad artifact into a dead shell.
        self.cond_end();
        match self.fault {
            Some(fault) => Err(fault),
            None => Ok(()),
        }
    }

    /// **WHAT THE SETTER READS FOR ONE REGION**: the device address of its
    /// window's rebased row CSR, the lane count to index it at, and whether
    /// this region can state a count at all.
    ///
    /// The third value is the interesting one, and the two kinds do opposite
    /// things with it. `true` means "no readable count here", which happens two
    /// ways: a region P4 could not seat runs once per interval of its class set
    /// and there is no single count to read; and a region on the PATCH axis
    /// carries no rebased qo boundaries at all — `indptr_host` is empty by
    /// construction, because that vector is the token rectangle's per-lane
    /// bounds and a tower region's span indexes images, while its staged
    /// `Tensor` is a zero-length seat at the head of its own slot. Reading a
    /// count out of that would be reading whatever the padding holds.
    ///
    /// **THE ADDRESS THIS RETURNS MAY BE BAKED INTO A GRAPH**, and is safe to
    /// bake: a conditional setter is a recorded node and the pointer it takes
    /// is `blob base + slot * stride` ([`Windows::packed`]), where the slot is
    /// a function of the `record::BodyKey` ([`seat`]) and the stride is the
    /// load's. Every fire of one key therefore finds this region's CSR at the
    /// address the capture froze — which is exactly what the H2 fix is for,
    /// and what a tightly packed blob could not promise past slot 0.
    fn count_of(&self, cond: Conditionals<'a>, region: u32) -> (u64, u32, bool) {
        match cond.windows.runs(region) {
            1 if !cond.windows.at(region, 0).indptr_host.is_empty() => {
                let window = cond.windows.at(region, 0);
                let lanes = window.indptr_host.len().saturating_sub(1) as u32;
                (window.indptr.ptr, lanes, false)
            }
            _ => (0, 0, true),
        }
    }

    /// Close whatever body is recording and begin `arm`'s, leaving the walk's
    /// launches pointed at the body stream.
    ///
    /// **THE CELL SAYS `BODY` ONLY WHILE A CAPTURE IS ACTUALLY RUNNING ON THAT
    /// STREAM**, which is why the write is after the begin and the restore is
    /// on the failure path. A cell left at `BODY` over a stream that is not
    /// capturing would send every launch after it somewhere it would EXECUTE,
    /// mid-capture, and the graph would come out empty where the body should
    /// be — silent, and exactly the shape a conditional exists to prevent.
    fn enter(&mut self, arm: u32) {
        let Some((open, was, body)) = self.open else {
            return;
        };
        let Some(cond) = self.cond else {
            return;
        };
        if body && let Err(fault) = crate::device::conditional::end_body(cond.body) {
            self.open = Some((open, was, false));
            cond.at.set(was);
            if self.fault.is_none() {
                self.fault = Some(fault);
            }
            return;
        }
        let Some(graph) = open.body(arm) else {
            self.open = Some((open, was, false));
            cond.at.set(was);
            if self.fault.is_none() {
                self.fault = Some(Fault::Unbound {
                    what: format!(
                        "arm {arm} of a conditional node the driver minted {} bodies for",
                        open.arms,
                    ),
                });
            }
            return;
        };
        match crate::device::conditional::begin_body(cond.body, graph) {
            Ok(()) => {
                self.open = Some((open, was, true));
                cond.at.set(BODY);
            }
            Err(fault) => {
                self.open = Some((open, was, false));
                cond.at.set(was);
                if self.fault.is_none() {
                    self.fault = Some(fault);
                }
            }
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
        // **THE ROTATION'S SEAM, AND THE WALK HAS EXACTLY ONE** (alto
        // streaming §3 item 4). Before anything of this region is dispatched:
        // release the slots whose tenants the previous region finished with,
        // issue the copies coming due, and make the compute stream wait for
        // the planes this region is about to read. Every one of those is an
        // enqueue — nothing here synchronizes, and the fire path never waits
        // on a demand read by design.
        //
        // **THE ORDINAL IS THE TEMPLATE'S AND NOT THE PASS'S**, which is what
        // makes it the index the rotation was planned against: `walk_phases`
        // announces every region of `CompiledModel::regions` in order,
        // including the ones a phase filter dispatches nothing for.
        if let Some(pump) = self.pump
            && self.fault.is_none()
            && let Err(fault) = pump.rotor.at(self.at, pump.compute)
        {
            self.fault = Some(fault);
        }
        self.at += 1;
        // The stream switch, and it is the whole of it: everything the `Run`
        // resolves afterwards fires on whatever this names, until the next
        // region says otherwise.
        //
        // **THE SAME CELL UNDER BOTH BUNDLES, AND ONLY ONE WRITE.** A load
        // that opened side streams carries it on [`Lanes`] and one that only
        // opened a conditional body carries it on [`Conditionals`]; they are
        // the same `Cell` when both are present, so writing through whichever
        // is there writes the one the `Run` reads.
        //
        // **AND NOT AT ALL WHILE A BRACKET IS OPEN**, which is the one line a
        // `SWITCH` needed here. A group's arms are `arms` consecutive REGIONS
        // under ONE conditional node, so the walk crosses this seam `arms - 1`
        // times with a body capture running; a write here would put the next
        // arm's launches back on the main stream, inside a `SWITCH` that is
        // still recording, and the body would be empty while its contents ran
        // unconditionally. Skipping it costs nothing: a conditional region is
        // never forkable (`model_compiler::stream::forkable` has read
        // `lowering == AlwaysLaunch` since D1), so every arm names stream 0 and
        // the value being skipped is the one already there.
        if self.open.is_some() {
            return;
        }
        if let Some(lanes) = self.lanes {
            lanes.at.set(region.stream);
        } else if let Some(cond) = self.cond {
            cond.at.set(region.stream);
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

    /// **THE EAGER CURSOR IGNORES IT AND THE RECORDING ONE RECORDS A NODE.**
    ///
    /// Ignoring is correct for an eager pass and it is not a shortcut: the
    /// walk's zero-row rule decides exactly what a conditional decides, at the
    /// same instant, so a fire that walks a conditional region eagerly runs
    /// the same nodes over the same rows (design §4 — conditionals are the
    /// optimization, zero-row always-launch is the semantics). That is what
    /// `model_exec::fire::EagerSink` says too, and why the two agree.
    ///
    /// A CAPTURE CANNOT IGNORE IT. The graph outlives the fire that recorded
    /// it, so a body recorded outside its conditional node is a body that runs
    /// under every composition the exec is replayed for — and it would
    /// compute. So the recording cursor places a real
    /// `CU_GRAPH_NODE_TYPE_CONDITIONAL` node and captures the region's
    /// launches into its child graph
    /// ([`crate::device::conditional`] holds the four driver calls), with the
    /// predicate stored by a KERNEL reading this region's row count off the
    /// device — `kernels_cuda::graph::set_conditional`, which is design §5's
    /// "the kernel reads the count" and the reason the decision is inside the
    /// graph rather than beside it.
    ///
    /// # A `SWITCH` IS THE SAME NODE ASKED `arms` TIMES OVER `arms` REGIONS
    ///
    /// P3 groups a merge's arms into one `SWITCH` when it can prove at most one
    /// of them is demanded by any admissible composition, and hands them over
    /// as CONSECUTIVE regions each stamped `Lowering::Switch { arm, arms }`.
    /// So the walk opens the bracket at arm 0, announces
    /// [`cond_arm`](Sink::cond_arm) once per arm, and closes at the last — and
    /// the bracket lives ACROSS `arms - 1` region boundaries, which is what
    /// `region_begin`'s one guard above is for. The node is minted with
    /// `size: arms` and `phGraph_out` holds one child graph per arm, indexed by
    /// the same number the walk announces; `cond_arm` is where one body is
    /// closed and the next begun.
    ///
    /// The predicate is `arms` launches of `set_switch` rather than one of
    /// `set_conditional`, because each arm's row count lives in its own
    /// window and there is no vector holding all of them. Each stores its own
    /// index only if its own window has rows; at most one does, which is the
    /// activation P3 proved before it formed the group; and if none does, the
    /// handle keeps the out-of-range default [`Kind::quiescent`] minted it
    /// with and no body runs.
    ///
    /// # What a region that cannot state a count gets, and it differs by kind
    ///
    /// A region P4 could not seat runs once per interval of its class set has
    /// no single row count to read, and a region on the patch axis has no
    /// boundary vector at all. For an `IF` such a region takes its body
    /// unconditionally — a null table with `absent` set, which the setter
    /// stores as 1 — and that is the conservative half of decision #3:
    /// always-launch is the correctness mechanism, and a conditional that
    /// declines to decide has given up an optimization and nothing else. The
    /// per-run zero-row skips inside the body are untouched.
    ///
    /// **A `SWITCH` HAS NO SUCH DIRECTION AND SO IT REFUSES.** Exactly one body
    /// runs; "take it anyway" is not available, and an arm guessed at is
    /// another arm's fire computed wrong. So a group with an unreadable arm is
    /// [`Fault::Unlowered`] naming the region, and the deployment's recourse is
    /// the same as every other conditional's: bake with
    /// `fat_region_us: INFINITY` and every region is always-launch.
    ///
    /// # A shell with no conditional machinery still refuses by name
    ///
    /// A load whose artifact holds no conditional never opens a body stream,
    /// so a cursor here with no [`Conditionals`] is a shell being asked for
    /// something it was not set up for — and [`Fault::Unlowered`] is still the
    /// answer, now naming a load rather than a toolkit.
    fn cond_begin(&mut self, lowering: &Lowering) {
        if !self.recording || self.fault.is_some() {
            return;
        }
        let region = self.at.saturating_sub(1);
        let unlowered = |why: &Lowering| Fault::Unlowered {
            region,
            lowering: format!("{why:?}"),
        };
        let kind = match *lowering {
            Lowering::If => Kind::If,
            Lowering::Switch { arms, .. } => Kind::Switch { arms: u32::from(arms) },
            Lowering::AlwaysLaunch => {
                self.fault = Some(unlowered(lowering));
                return;
            }
        };
        let Some(cond) = self.cond else {
            self.fault = Some(unlowered(lowering));
            return;
        };
        let outcome = (|| {
            let handle = crate::device::conditional::handle(cond.main, kind)?;
            match kind {
                // **AN `IF` ASKS ONE QUESTION ABOUT ONE REGION.** The predicate
                // is the one launch of the whole sequence: a region with
                // exactly one run hands its staged boundary vector and the
                // setter reads `indptr[lanes]`; anything else hands nothing and
                // states that an absent table means "take it".
                Kind::If => {
                    let (indptr, lanes, absent) = self.count_of(cond, region);
                    kernels_cuda::graph::set_conditional(
                        cond.setter,
                        handle,
                        indptr,
                        lanes,
                        absent,
                        kernels_cuda::graph::Arm::Set,
                    )
                    .map_err(|why| Fault::Unbound {
                        what: format!(
                            "the conditional setter for region {region}, which answered {why}"
                        ),
                    })?;
                }
                // **A `SWITCH` ASKS THE SAME QUESTION `arms` TIMES, ONE REGION
                // APART.** The arms are CONSECUTIVE regions — that is what P2
                // hands P3 and what `switch_groups` requires to form a group at
                // all — so this region's index plus the arm number is that
                // arm's index, and its window is already in the table. Each
                // setter stores its own number only if its own window has rows,
                // and at most one of them does: `switch_groups` proves no
                // admissible composition demands two arms, which is the
                // activation proof P3 asks for before it forms a group.
                Kind::Switch { arms } => {
                    for arm in 0..arms {
                        let (indptr, lanes, absent) = self.count_of(cond, region + arm);
                        // **AND AN ARM THAT CANNOT STATE A COUNT REFUSES THE
                        // GROUP.** For an `IF` "cannot tell" resolves to "take
                        // it", which is always-launch and is the correctness
                        // mechanism. A `SWITCH` has no such direction — exactly
                        // one body runs — so an arm with no readable row count
                        // would have to be guessed at, and a guess here picks
                        // somebody else's arm. `Fault::Unlowered` naming the
                        // region is the honest answer.
                        if absent {
                            return Err(unlowered(lowering));
                        }
                        kernels_cuda::graph::set_switch(
                            cond.setter,
                            handle,
                            arm,
                            indptr,
                            lanes,
                            kernels_cuda::graph::Arm::Set,
                        )
                        .map_err(|why| Fault::Unbound {
                            what: format!(
                                "the switch setter for arm {arm} of the group at region \
                                 {region}, which answered {why}"
                            ),
                        })?;
                    }
                }
            }
            crate::device::conditional::open(cond.main, handle, kind)
        })();
        match outcome {
            Ok(open) => {
                let was = cond.at.get();
                self.open = Some((open, was, false));
                // An `IF` has no `cond_arm`, so its one body opens here; a
                // `SWITCH`'s opens at `cond_arm(0)`, which the walk calls
                // before anything of arm 0 is dispatched.
                if kind == Kind::If {
                    self.enter(0);
                }
            }
            Err(fault) => self.fault = Some(fault),
        }
    }

    /// **ONE ARM OF A `SWITCH`, AND THE SEAM BETWEEN TWO BODIES.**
    ///
    /// The walk announces this once per arm, including arm 0, between that
    /// arm's `region_begin` and its first launch. So it is exactly the instant
    /// to close whatever body was recording and open this arm's — `phGraph_out`
    /// is indexed by the same arm number, because the driver mints the array in
    /// the order `size` states and `Def::Merge`'s arm order is what P3 grouped.
    ///
    /// Never called for an `IF` (`walk` passes `None` for its arm), so the
    /// no-bracket case here is a walk that reached an arm without an open node,
    /// which `cond_begin` has already faulted about.
    fn cond_arm(&mut self, arm: u8) {
        if !self.recording || self.open.is_none() {
            return;
        }
        self.enter(u32::from(arm));
    }

    /// Close the body and put the walk back on the stream the region named.
    ///
    /// **THE BODY IS CLOSED EVEN WHEN THE WALK FAULTED INSIDE IT**, for the
    /// reason `Graph::capture` ends the parent capture on every path: a stream
    /// left mid-capture answers every later call with
    /// `cudaErrorStreamCaptureUnjoined` for the rest of the process. The first
    /// fault still wins — a close that also fails is a second sentence about
    /// the same failure.
    fn cond_end(&mut self) {
        let Some((_, was, body)) = self.open.take() else {
            return;
        };
        let Some(cond) = self.cond else {
            return;
        };
        let closed = body
            .then(|| crate::device::conditional::end_body(cond.body))
            .unwrap_or(Ok(()));
        cond.at.set(was);
        if let Err(fault) = closed
            && self.fault.is_none()
        {
            self.fault = Some(fault);
        }
    }
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
    use model_exec::fire::{ClassWindow, WindowTable};
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

    /// A recording cursor with no [`Conditionals`] is a load whose context
    /// opened no body stream — which is every artifact P3 declined — and it
    /// still refuses by name rather than recording the body bare. The arm that
    /// RECORDS needs a capturing stream and a device, so it lives in
    /// `tests/conditional_nodes.rs` and `tests/conditional_lowering.rs`.
    #[test]
    fn a_recording_cursor_with_nowhere_to_put_a_conditional_still_refuses_it() {
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
        assert!(fault.to_string().contains("nowhere"), "{fault}");
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

    /// **THE SECOND READING IS THE FIRST ONE'S SLICE, UN-SUBTRACTED** (bodies
    /// design, chunk 2c-a) — the identity `Run::qo_indptr_absolute_host` is
    /// built on, pinned here because it is what makes the two readings one
    /// fact. A window's absolute boundaries are `indptr[lane_offset ..=
    /// lane_offset + lanes]` with nothing done to them, and subtracting that
    /// slice's own first entry is exactly [`rebase`].
    ///
    /// **THIS IS THE HOST READING AND ONLY THE HOST READING.** The slice is
    /// legal here because it is taken by the fire that made it; the DEVICE
    /// reading is handed over whole, and the test below is why.
    #[test]
    fn the_rebased_reading_is_the_whole_vector_s_lane_slice_minus_its_own_base() {
        let indptr = [0, 7, 10, 11, 12, 13];
        for set in [ClassSet::of([0]), ClassSet::of([1]), ClassSet::of([0, 1])] {
            let span = table()
                .span(&set)
                .expect("consecutive")
                .expect("non-empty");
            let first = span.lane_offset as usize;
            let absolute = &indptr[first..=first + span.lanes as usize];
            let rebased = rebase(&indptr, span);
            assert_eq!(absolute.len(), rebased.len(), "both are `[lanes + 1]`");
            assert!(
                absolute
                    .iter()
                    .zip(&rebased)
                    .all(|(there, here)| there - absolute[0] == *here),
                "{absolute:?} minus its own first entry is not {rebased:?}",
            );
        }
        // And the two readings are NOT the same vector wherever a window does
        // not begin at the fire's lane zero, which is the case the seam
        // exists for.
        let decode = table()
            .span(&ClassSet::of([1]))
            .expect("consecutive")
            .expect("non-empty");
        assert_eq!(&indptr[2..=5], &[10, 11, 12, 13], "absolute");
        assert_eq!(rebase(&indptr, decode), vec![0, 1, 2, 3], "rebased");
    }

    /// **THE DEVICE READING IS HANDED OVER WHOLE, AND THAT IS THE WHOLE OF
    /// WHAT MAKES IT SAFE TO BAKE.** A body records the pointer it is given
    /// and replays it at another fire of the same `record::BodyKey`, and
    /// `lane_offset` is the sum of the lanes of the classes in front of the
    /// window — a number the key does not fix. So `base + lane_offset * 4`
    /// would be a stale address on every replay but the recording one, which
    /// is exactly the staleness the absolute seam exists to remove; the fire
    /// vector's own base is a function of the LOAD and does not move at all.
    ///
    /// Which entries are a launch's requests is the SCHEDULE's business, not
    /// the pointer's — a sliced absolute vector would be the worst of both
    /// readings, a moving pointer AND values counting from somewhere else.
    #[test]
    fn the_absolute_reading_does_not_move_with_the_window() {
        // Two windows of one fire, the second one well off lane zero.
        let mut first = plain(0, 7);
        first.span.lane_offset = 0;
        first.span.lanes = 2;
        let mut second = plain(7, 6);
        second.span.lane_offset = 2;
        second.span.lanes = 4;
        let mut table = windows(&[vec![first], vec![second]]);
        table.qo_absolute_host = vec![0, 7, 10, 11, 12, 13];

        const BASE: u64 = 0x7000;
        table.bind_qo_absolute(Some(BASE));
        let whole = table.qo_absolute().expect("bound");
        assert_eq!(whole.ptr, BASE, "the fire's own base, and no window's");
        assert_eq!(
            whole.rows,
            table.qo_absolute_host().len() as u32,
            "the FIRE's lanes + 1, not any window's",
        );
        // The address a lane slice would have produced for the second window
        // — the one a body would have baked and replayed stale.
        assert_ne!(whole.ptr, BASE + 2 * 4, "the reading is not cut at lane_offset");
        // And there is no per-window reading at all to disagree with it: the
        // table answers one vector for the fire, asked from anywhere.
        assert_eq!(table.qo_absolute().expect("bound").ptr, whole.ptr);
    }

    /// **AN UNBOUND SECOND READING IS READ BY NOBODY.** The vector is staged
    /// only for a fire the shell routed to a body (`inputs::Fire::qo_absolute`
    /// is empty otherwise), so the table answers `None` and every ragged view
    /// keeps the rebased boundaries it always took — which is what makes this
    /// chunk inert on every path but the bodied one.
    #[test]
    fn an_unbound_absolute_qo_vector_answers_none() {
        let mut table = windows(&[vec![plain(0, 13)]]);
        assert!(table.qo_absolute().is_none(), "no vector, nothing bound");

        table.qo_absolute_host = vec![0, 7, 10, 11, 12, 13];
        assert!(table.qo_absolute().is_none(), "a vector, still nothing bound");

        const BASE: u64 = 0x7000;
        table.bind_qo_absolute(Some(BASE));
        let whole = table.qo_absolute().expect("bound");
        assert_eq!(whole.ptr, BASE, "the vector is the FIRE's, so it starts at the base");
        assert_eq!(whole.rows, 6, "lanes + 1");
        assert_eq!(whole.width, 1);
        assert_eq!(table.qo_absolute_host(), &[0, 7, 10, 11, 12, 13]);

        table.bind_qo_absolute(None);
        assert!(table.qo_absolute().is_none(), "unbinding is the off switch");
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

    // ── THE LIVE-ROWS SEAT (bodies design). Pure arithmetic over a table
    //    built by hand: `Windows::of` wants a `Trace` and a `CompiledModel`,
    //    and none of what these three assert is about either.

    /// A window with nothing but a span — every other field at the shape an
    /// ordinary (ungathered, ungrouped) one has.
    fn plain(row_offset: u32, rows: u32) -> Window {
        Window {
            span: MaskSpan {
                row_offset,
                rows,
                lane_offset: 0,
                lanes: 1,
            },
            indptr_host: Vec::new(),
            indptr: Tensor::new(0, 0, 1, Dtype::I32),
            segments_host: Vec::new(),
            segments: Tensor::new(0, 0, 2, Dtype::I32),
            segment_cap: 0,
            gathered: None,
            patch: MaskSpan::default(),
        }
    }

    /// A table of `spans[region][run]`, seated the way `Windows::of` seats
    /// one — one window per run, and the live words filled with the identity.
    fn windows(spans: &[Vec<Window>]) -> Windows {
        let mut table = Windows {
            windows: Vec::new(),
            runs: Vec::new(),
            of_region: Vec::new(),
            // A carve wide enough for anything these tables hold: two classes,
            // eight lanes, one run. `Windows::of` takes the load's.
            slots: Slots::new(2, 8, 1, 1),
            live_words: Vec::new(),
            live_stride: 0,
            live_base: 0,
            // The second reading is a fire-wide vector these span-only tables
            // have no fire for; the tests that want one fill it themselves.
            qo_absolute_host: Vec::new(),
            qo_absolute_base: 0,
            qo_absolute_lanes: 0,
        };
        for region in spans {
            let start = table.runs.len() as u32;
            for window in region {
                table.runs.push(table.windows.len() as u32);
                table.windows.push(window.clone());
            }
            table.of_region.push((start, region.len() as u32));
        }
        let stride = table.max_runs();
        let wide = stride as usize;
        let mut live = vec![0u32; table.of_region.len() * wide * 4];
        for region in 0..table.of_region.len() as u32 {
            for run in 0..table.runs(region) {
                let span = table.at(region, run).span;
                let seat = 4 * (region as usize * wide + run as usize);
                live[seat] = span.rows;
                live[seat + 1] = span.row_offset;
                live[seat + 2] = span.lanes;
                live[seat + 3] = span.lane_offset;
            }
        }
        table.live_words = live;
        table.live_stride = stride;
        table
    }

    /// **THE IDENTITY, WORD FOR WORD.** The seat's contract is `[rows,
    /// row_offset, lanes, lane_offset]` per (region, run) at a fixed stride,
    /// and every word of it is the window's own — which is what makes arming
    /// it change no arithmetic on either axis.
    #[test]
    fn the_live_seat_is_every_windows_own_rows_and_offset_at_a_fixed_stride() {
        let table = windows(&[
            vec![plain(0, 13)],
            vec![plain(0, 10), plain(10, 3)],
            vec![plain(3, 5)],
        ]);
        let stride = table.max_runs() as usize;
        assert_eq!(stride, 2, "the widest region cuts two runs");
        assert_eq!(
            table.live().len(),
            table.of_region.len() * stride * 4,
            "regions x max_runs x 4, and no run's words are missing",
        );
        for region in 0..table.of_region.len() as u32 {
            for run in 0..table.runs(region) {
                let seat = 4 * (region as usize * stride + run as usize);
                let span = table.at(region, run).span;
                assert_eq!(table.live()[seat], span.rows, "row count first");
                assert_eq!(table.live()[seat + 1], span.row_offset, "row start second");
                assert_eq!(table.live()[seat + 2], span.lanes, "lane count third");
                assert_eq!(
                    table.live()[seat + 3],
                    span.lane_offset,
                    "lane start fourth",
                );
            }
        }
        // The words of a run the table did not cut are untouched zeros — the
        // stride is a rectangle and region 0 has one run in a two-run one.
        assert_eq!(&table.live()[4..8], &[0, 0, 0, 0]);
    }

    /// **THE ADDRESS IS A MULTIPLICATION, AND `0` IS THE DISARMED SEAT.**
    #[test]
    fn a_bound_seat_answers_base_plus_sixteen_per_slot_and_an_unbound_one_answers_zero() {
        let mut table = windows(&[vec![plain(0, 10), plain(10, 3)], vec![plain(0, 13)]]);
        let stride = u64::from(table.max_runs());

        assert_eq!(table.live_at(0, 0), 0, "nothing is bound yet");
        assert_eq!(table.live_at(1, 0), 0);

        const BASE: u64 = 0x4000;
        table.bind_live(Some(BASE));
        for (region, run) in [(0, 0), (0, 1), (1, 0)] {
            assert_eq!(
                table.live_at(region, run),
                BASE + 16 * (u64::from(region) * stride + u64::from(run)),
                "region {region} run {run}",
            );
        }
        // Past the table in either index is the disarmed seat and not an
        // address past the carve.
        assert_eq!(table.live_at(2, 0), 0, "no such region");
        assert_eq!(table.live_at(0, 2), 0, "no such run");

        // And unbinding puts it back, without touching what `bind` seated.
        table.bind(0x9000);
        let packed = table.at(0, 1).indptr;
        table.bind_live(None);
        assert_eq!(table.live_at(0, 0), 0, "`None` re-disarms the seat");
        assert_eq!(
            table.at(0, 1).indptr,
            packed,
            "unbinding the seat moved a packed tensor",
        );
    }

    /// **THE BODIES PATH'S ADMISSIBILITY RULE**, which is `Run::whole_fire`
    /// over the whole table: an absent window is not asked, and one windowed
    /// region is enough to refuse.
    #[test]
    fn a_fire_covers_itself_only_when_every_region_with_rows_holds_all_of_them() {
        let whole = windows(&[vec![plain(0, 13)], vec![plain(0, 13)], vec![plain(0, 0)]]);
        assert!(whole.covers_fire(13), "one class, and the absent region is not asked");
        assert!(whole.covers_fire(4), "a shorter fire is a prefix of it");

        let split = windows(&[vec![plain(0, 10)], vec![plain(10, 3)]]);
        assert!(!split.covers_fire(13), "region 1 starts above the fire's zero");

        let short = windows(&[vec![plain(0, 10)]]);
        assert!(!short.covers_fire(13), "ten rows is not thirteen");

        let mut grouped = plain(0, 13);
        grouped.segments_host = vec![0, 3, 8, 5];
        assert!(
            !windows(&[vec![grouped]]).covers_fire(13),
            "a segment list's span is a union and says nothing about the rows it owns",
        );
    }

    /// A window that gathered its rows out of the fire's — every other field
    /// at [`plain`]'s shape, because only `gathered` is under test.
    fn compacted(row_offset: u32, rows: u32) -> Window {
        Window {
            gathered: Some(Gathered {
                runs: vec![MaskSpan {
                    row_offset,
                    rows,
                    lane_offset: 0,
                    lanes: 1,
                }],
                rows_host: (0..rows as i32).collect(),
                rows: Tensor::new(0, 0, 1, Dtype::I32),
                spaces: Vec::new(),
            }),
            ..plain(row_offset, rows)
        }
    }

    /// **THE WIDE READING OF THE SAME RULE**: a region whose every op reads
    /// both seat words may be windowed, and the two shape clauses are waived
    /// for nobody.
    #[test]
    fn a_shifting_region_may_be_windowed_and_a_guard_only_one_may_not() {
        // Region 1 holds rows 10..13 of a 13-row fire. Whether that is
        // admissible is now a question about region 1's OPS.
        let split = windows(&[vec![plain(0, 10)], vec![plain(10, 3)]]);
        assert!(
            split.covers_fire_shifted(13, &[true, true]),
            "both regions move their own base off the seat's start",
        );
        assert!(
            !split.covers_fire_shifted(13, &[true, false]),
            "region 1 is windowed and reads only the seat's count",
        );
        // A short slice is not a licence: region 0 covers the fire on the
        // narrow clauses and region 1 is windowed with no flag to stand on.
        let tail = windows(&[vec![plain(0, 13)], vec![plain(10, 3)]]);
        assert!(tail.covers_fire_shifted(13, &[false, true]));
        assert!(
            !tail.covers_fire_shifted(13, &[]),
            "a region past the end of the slice is not shifting",
        );

        // The narrow reading is untouched by the argument, and the wide one
        // agrees with it wherever nothing is windowed.
        let whole = windows(&[vec![plain(0, 13)], vec![plain(0, 13)], vec![plain(0, 0)]]);
        assert!(!split.covers_fire(13), "the narrow reading still refuses it");
        assert!(whole.covers_fire_shifted(13, &[false, false, false]));

        // A region with no rows is exempt whatever its ops read: there is no
        // window for a seat word to be wrong about.
        let absent = windows(&[vec![plain(0, 13)], vec![plain(0, 0)]]);
        assert!(
            absent.covers_fire_shifted(13, &[false, false]),
            "the absent region is not asked",
        );
    }

    /// **THE TWO SHAPES THE SEAT HAS NO WORD FOR**, refused whether or not
    /// the region's ops read the start.
    #[test]
    fn a_gathered_or_grouped_region_is_refused_however_its_ops_address() {
        let gathered = windows(&[vec![compacted(0, 13)]]);
        assert!(
            !gathered.covers_fire_shifted(13, &[true]),
            "a compacted rectangle is its own plane; no offset into the fire names its rows",
        );
        assert!(!gathered.covers_fire_shifted(13, &[false]));

        let mut segmented = plain(0, 13);
        segmented.segments_host = vec![0, 3, 8, 5];
        let grouped = windows(&[vec![segmented]]);
        assert!(
            !grouped.covers_fire_shifted(13, &[true]),
            "a union of intervals is not an interval, whatever the start says",
        );
        assert!(!grouped.covers_fire_shifted(13, &[false]));
    }

    /// **THE TABLE, AND THE BOOL IS ITS COLLAPSE** (the tier-2 campaign).
    ///
    /// The clause arithmetic is asserted twice over above; what this asserts
    /// is the thing tier 2 actually spends — that the answer is PER REGION, so
    /// one region losing its shift makes ONE island rather than refusing the
    /// composition, and that
    /// [`covers_fire_shifted`](Windows::covers_fire_shifted) is exactly "every
    /// entry is [`Admit::Captured`]" and not a second arithmetic beside it.
    #[test]
    fn one_region_that_cannot_be_captured_makes_one_island_and_not_a_refusal() {
        // Region 0 is the fire's first ten rows, region 1 the last three, and
        // region 2 has none at all.
        let table = windows(&[
            vec![plain(0, 10)],
            vec![plain(10, 3)],
            vec![plain(0, 0)],
        ]);

        // Everything shifting: a body holds the whole composition, and the
        // collapsed reading says so.
        let whole = table.admits(13, &[true, true, true]);
        assert!(whole.iter().all(|admit| *admit == Admit::Captured));
        assert!(table.covers_fire_shifted(13, &[true, true, true]));

        // Take the shift away from the WINDOWED region and exactly one entry
        // moves. The collapsed reading refuses — which is what it is for — and
        // the table is what says the refusal costs one region.
        let crippled = table.admits(13, &[true, false, true]);
        assert_eq!(
            crippled,
            vec![Admit::Captured, Admit::Island, Admit::Captured],
            "one region lost its shift and the table blamed somebody else",
        );
        assert!(!table.covers_fire_shifted(13, &[true, false, true]));

        // And an EMPTY region is capturable whatever its ops read, which is
        // what keeps a region no composition demands from splitting one
        // segment into two.
        assert_eq!(crippled[2], Admit::Captured);

        // A gathered region is an island under every reading of the slice —
        // the shape clauses are waived for nobody.
        let gathered = windows(&[vec![plain(0, 10)], vec![compacted(10, 3)]]);
        assert_eq!(
            gathered.admits(13, &[true, true]),
            vec![Admit::Captured, Admit::Island],
        );
    }

    // ── **THE H2 FIX**: the packed blob's per-window addresses are a function
    //    of the `record::BodyKey` and not of the fire, because a body bakes
    //    them (this file's header, `Windows::packed`, `seat`). Both halves are
    //    host arithmetic and neither needs a `Trace`.

    /// A plain window carrying the one vector the packed blob is made of: a
    /// rebased `[lanes + 1]` boundary list, which is the thing whose LENGTH
    /// moves between fires of one key.
    fn bounded(span: MaskSpan) -> Window {
        Window {
            span,
            indptr_host: (0..=span.lanes as i32).collect(),
            ..plain(0, 0)
        }
    }

    /// The three spans a two-class fire resolves for the masks `{A}`, `{A,B}`
    /// and `{B}`, the way `WindowTable::spans_into` resolves them: a class
    /// with no rows contributes nothing and an empty mask is the zero window.
    /// `a` and `b` are each `(rows, lanes)`.
    fn resolved(a: (u32, u32), b: (u32, u32)) -> [MaskSpan; 3] {
        let one = |(rows, lanes): (u32, u32), row_offset, lane_offset| MaskSpan {
            row_offset,
            rows,
            lane_offset,
            lanes,
        };
        let first = if a.0 == 0 { MaskSpan::default() } else { one(a, 0, 0) };
        let second = if b.0 == 0 { MaskSpan::default() } else { one(b, a.0, a.1) };
        let both = match (a.0 == 0, b.0 == 0) {
            (false, false) => one((a.0 + b.0, a.1 + b.1), 0, 0),
            (false, true) => first,
            (true, false) => second,
            (true, true) => MaskSpan::default(),
        };
        [first, both, second]
    }

    /// **ONE KEY, TWO LANE SPLITS, THE SAME ADDRESSES.** The defect this pins:
    /// a tightly packed blob puts window `i` at `base + Σ_{j<i} words_j`, the
    /// lane counts move between fires of one `BodyKey`, and a graph that baked
    /// window 1's `indptr` pointer would read window 0's tail on the next
    /// fire. At a fixed stride only the slot index is in the address.
    #[test]
    fn a_slot_lands_at_the_same_address_however_the_fire_split_its_lanes() {
        // Three regions over the two-class fire's three masks, so that the
        // table has a slot 0, a slot 1 and a slot 2 to be wrong about.
        let table_of = |a, b| {
            let [first, both, second] = resolved(a, b);
            windows(&[
                vec![bounded(both)],
                vec![bounded(first)],
                vec![bounded(second)],
            ])
        };
        const BASE: u64 = 0x8000;
        let mut wide = table_of((10, 2), (3, 3));
        let mut narrow = table_of((4, 1), (6, 6));
        wide.bind(BASE);
        narrow.bind(BASE);

        let stride = wide.slots().stride() * 4;
        assert_eq!(stride, narrow.slots().stride() * 4, "one load, one carve");
        for region in 0..3 {
            let (here, there) = (wide.at(region, 0).indptr, narrow.at(region, 0).indptr);
            assert_eq!(
                here.ptr, there.ptr,
                "region {region}'s rebased CSR moved with the lane split",
            );
            assert_eq!(
                here.ptr,
                BASE + u64::from(region) * stride,
                "a slot's address is base + slot * stride and nothing else",
            );
            assert_ne!(here.rows, there.rows, "the two fires really are different fires");
        }

        // And the words the shell stages agree with the addresses it bound:
        // every slot's vector is at its own offset, with padding behind it.
        for table in [&wide, &narrow] {
            let blob = table.packed();
            for region in 0..3u32 {
                let window = table.at(region, 0);
                let at = table.slots().at(region as usize) as usize;
                assert_eq!(
                    &blob[at..at + window.indptr_host.len()],
                    window.indptr_host.as_slice(),
                    "region {region}'s boundaries are not at its slot",
                );
            }
            assert!(
                blob.len() as u64 <= table.slots().tail(),
                "the blob outgrew the carve it was reserved in",
            );
        }
    }

    /// **AND THE SLOT ITSELF IS THE KEY'S**, which is `seat`'s half of the
    /// same argument: the span is this fire's encoding of `mask ∩ present`,
    /// so which masks share a slot is fixed by which classes have rows — and
    /// that is exactly what a `BodyKey` carries.
    #[test]
    fn two_masks_share_a_slot_in_every_fire_of_a_key_or_in_none_of_them() {
        // Four regions over three masks — `{A}`, `{A,B}`, `{A}` again, `{B}`
        // — seated the way `Windows::of` seats them.
        let seated = |a, b| {
            let [first, both, second] = resolved(a, b);
            let mut held: Vec<Window> = Vec::new();
            let map: Vec<u32> = [first, both, first, second]
                .into_iter()
                .map(|span| seat(&mut held, bounded(span)))
                .collect();
            (map, held.len())
        };

        let wide = seated((10, 2), (3, 3));
        assert_eq!(wide.0, vec![0, 1, 0, 2], "three distinct masks, three slots");
        assert_eq!(
            seated((4, 1), (6, 6)),
            wide,
            "the lane split moved every span and no slot",
        );
        assert_eq!(
            seated((1, 1), (1, 1)),
            wide,
            "nor does the row split, down to one row a class",
        );

        // **THE ONE SPLIT THAT DOES MOVE THE SHARING MOVES THE KEY WITH IT.**
        // A class with no rows is absent from `BodyKey::classes`, so `{A,B}`
        // collapsing onto `{A}` is a different body — recorded against this
        // table, never replayed against the other one's addresses.
        let absent = seated((10, 2), (0, 0));
        assert_eq!(
            absent.0,
            vec![0, 0, 0, 1],
            "the both-classes mask is the A mask when B has no rows",
        );
        assert_ne!(absent.0, wide.0, "and that is a different key, so a different body");
    }
}
