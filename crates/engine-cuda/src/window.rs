//! Per-region windows (which fire rows/lanes each template region runs over)
//! and the cursor that tells a [`Run`] which window it is in.
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

/// One window: its span (rows/lanes) plus a rebased qo-boundary CSR (`[lanes + 1]`, first entry 0 — a ragged view's boundaries are offsets into the rectangle it cuts, so each window carries its own copy).
#[derive(Debug, Clone)]
pub struct Window {
    /// Rows/lanes this window covers, per row axis (token vs. patch). A token region's two entries are different rectangles; a patch region's are the same interval. Gathered: the token entry is the compacted rectangle.
    pub spans: model_ir::PerAxis<MaskSpan>,
    /// `[lanes + 1]`: the window's qo boundaries, rebased to start at 0.
    pub indptr_host: Vec<i32>,
    /// The same vector, staged. `Tensor::new(0, 0, 0, ..)` until [`Windows::bind`] has been given the staging base.
    pub indptr: Tensor,

    /// `Fallback::Grouped`: which rows of this rectangle are the consumer's — `[segs][2]` as `(row offset within the span, rows)`, ascending. Empty for an ordinary window.
    pub segments_host: Vec<i32>,
    /// The same vector, staged, beside the boundaries.
    pub segments: Tensor,
    /// The artifact's load-time bound on the segment count (`model_exec::fire::max_runs`); sizes the grid's segment axis.
    pub segment_cap: u32,
    /// Present iff this window is a [`Fallback::Copy`](model_compiler::Fallback) — the runs it compacts, and everything a consumer needs to read them as one.
    pub gathered: Option<Gathered>,
}

/// What shape of rows a window is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowShape {
    /// One contiguous run of the fire's own rows — the only shape a `(count, start)` seat can speak for.
    Interval,
    /// `Fallback::Copy`: rows compacted into a scratch rectangle, numbered from its own zero.
    Gathered,
    /// `Fallback::Grouped`: the span is a union of intervals with foreign rows in the gaps.
    Grouped,
}

impl Window {
    /// This window's interval on one row axis.
    #[must_use]
    pub fn on(&self, axis: model_ir::RowAxis) -> MaskSpan {
        self.spans[axis]
    }

    /// The region's own interval (primary entry on either axis).
    #[must_use]
    pub fn span(&self) -> MaskSpan {
        self.spans[model_ir::RowAxis::PRIMARY]
    }

    /// Which of the three shapes this window is, read off `segments`/`gathered`. Grouped is checked first: `Windows::of` replaces a grouped region's spans with their union, so the two cannot both be set.
    #[must_use]
    pub fn shape(&self) -> WindowShape {
        if self.segs() != 0 {
            WindowShape::Grouped
        } else if self.gathered.is_some() {
            WindowShape::Gathered
        } else {
            WindowShape::Interval
        }
    }

    /// Is this window one run of the fire's own rows?
    #[must_use]
    pub fn is_interval(&self) -> bool {
        matches!(self.shape(), WindowShape::Interval)
    }

    /// Is this window one run of all `total` of the fire's rows (starts at 0, covers at least `total`, and is an interval)?
    #[must_use]
    pub fn is_whole(&self, total: u32) -> bool {
        let span = self.span();
        span.row_offset == 0 && span.rows >= total && self.is_interval()
    }

    /// How many segments this window states — `0` for an ordinary one.
    #[must_use]
    pub fn segs(&self) -> u32 {
        self.segments_host.len() as u32 / 2
    }

    /// The longest segment's row count — the grid's row axis for a grouped launch, and `0` when there are none.
    #[must_use]
    pub fn segment_rows(&self) -> u32 {
        self.segments_host
            .chunks_exact(2)
            .map(|pair| pair[1].max(0) as u32)
            .max()
            .unwrap_or(0)
    }
}

/// A `Fallback::Copy`'s window: which fire rows the compacted rectangle is made of, and the per-space tables the gathered lanes address the pool by. Only activations move via device gather; kv tables are recomputed host-side.
#[derive(Debug, Clone)]
pub struct Gathered {
    /// The fire intervals this rectangle compacts, in order.
    pub runs: Vec<MaskSpan>,
    /// `[rows]`: the fire row each compacted row was read from.
    pub rows_host: Vec<i32>,
    /// The same vector, staged.
    pub rows: Tensor,
    /// One entry per kv geometry space, in space order.
    pub spaces: Vec<GatheredSpace>,
}

/// One kv space's geometry, re-cut for a gathered window's lanes. The page-id list is copied, not sliced: gathered lanes own non-contiguous spans, so the list is compacted with a fresh prefix sum.
#[derive(Debug, Clone)]
pub struct GatheredSpace {
    /// `[lanes + 1]`: bounds over [`page_indices_host`](GatheredSpace::page_indices_host), a fresh prefix sum starting at 0.
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
#[derive(Debug, Clone, Copy)]
pub struct Copies<'a> {
    /// Which position of `Budget::buckets` this fire's rows land in; `0` for a deployment that declared no lattice.
    pub bucket: u32,
    /// Does this shell serve `Fallback::Copy` at all? `false` for a masked fire regardless: gathering the mask slab needs the same page-id-list treatment kv gets, which is not implemented, so a masked fire always splits.
    pub enabled: bool,
    /// This fire's host geometry, one per kv space — what the gathered pool tables are re-cut from.
    pub spaces: &'a [Geometry],
}

impl Copies<'_> {
    /// The answer for a shell that does not copy: split everything.
    #[must_use]
    pub fn off() -> Copies<'static> {
        Copies {
            bucket: 0,
            enabled: false,
            spaces: &[],
        }
    }
}

/// The fixed carve the packed window blob is laid out in: one slot per distinct window, all the same width, plus the gathered payloads behind them. Fixed stride (not packed tight) so a slot's address is `base + slot * stride`, surviving replay across fires whose per-window lengths differ.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Slots {
    /// Words per slot: `max_lanes + 1 + 2 * max_segs`.
    stride: u64,
    /// How many slots the carve holds.
    slots: u64,
    /// How many GATHERED payloads ride behind them — `model_exec::fire::fragmentable`, the artifact's bound on distinct masks found in pieces.
    gathered: u64,
    /// Words per gathered payload: the row map plus, per kv space, the page bounds, the compacted page-id list and the two per-lane vectors.
    gathered_stride: u64,
}

impl Slots {
    /// The carve for one load: `classes`/`lanes`/`segs`/`gathered` size the slots; `rows`/`spaces`/`pages` (budget row ceiling, kv space count, page ceiling) bound one gathered payload.
    #[must_use]
    pub fn new(
        classes: usize,
        lanes: u64,
        segs: u32,
        gathered: usize,
        rows: u64,
        spaces: usize,
        pages: u64,
    ) -> Slots {
        Slots {
            stride: lanes + 1 + 2 * u64::from(segs.max(1)),
            slots: (classes * (classes + 1) / 2 + 1 + gathered) as u64,
            gathered: gathered as u64,
            gathered_stride: rows + spaces as u64 * (3 * lanes + 1 + pages),
        }
    }

    /// Words per slot.
    #[must_use]
    pub fn stride(&self) -> u64 {
        self.stride
    }

    /// How many `i32`s the fixed slot region occupies — where the gathered payloads begin.
    #[must_use]
    pub fn tail(&self) -> u64 {
        self.slots * self.stride
    }

    /// The word offset of one slot's vectors, from the blob's base.
    #[must_use]
    pub fn at(&self, slot: usize) -> u64 {
        slot as u64 * self.stride
    }

    /// The word offset of the `which`th gathered payload, at its own fixed stride behind the slots.
    #[must_use]
    pub fn gathered_at(&self, which: u64) -> u64 {
        self.tail() + which * self.gathered_stride
    }

    /// How many `i32`s the whole carve is: the fixed slot region plus every gathered payload behind it.
    #[must_use]
    pub fn words(&self) -> u64 {
        self.gathered_at(self.gathered)
    }

    /// Will one window's vectors (boundary vector + segment list) fit the slot they are about to be seated in?
    #[must_use]
    pub fn fits(&self, words: u64) -> bool {
        words <= self.stride
    }
}

/// The fixed carve the live-geometry seat is laid out in: four `u32` per (region ordinal, run), addressed by multiplication, never lookup — so the device address is a function of the cursor's two `u32`s. `Windows::of` builds it at the fire's own bounds, never wider than `Inputs::reserve`'s.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Seat {
    /// How many region ordinals the rectangle has rows for.
    regions: u64,
    /// How many runs each of them has columns for — the stride.
    runs: u64,
}

impl Seat {
    /// One (region, run) seat's word count and order: `[rows, row_offset, lanes, lane_offset]`.
    pub const WORDS: u64 = 4;

    /// The rectangle for one table: `regions` ordinals by `runs` columns.
    #[must_use]
    pub fn new(regions: u64, runs: u64) -> Seat {
        Seat { regions, runs }
    }

    /// How many `u32`s the whole rectangle occupies — what a reserve carves and what a fill allocates.
    #[must_use]
    pub fn words(&self) -> u64 {
        self.regions * self.runs * Self::WORDS
    }

    /// The word offset of one (region, run)'s four words, or [`None`] for a pair this rectangle does not hold.
    #[must_use]
    pub fn at(&self, region: u32, run: u32) -> Option<u64> {
        let (region, run) = (u64::from(region), u64::from(run));
        if region >= self.regions || run >= self.runs {
            return None;
        }
        Some((region * self.runs + run) * Self::WORDS)
    }

    /// The same offset in bytes — what a device base is added to.
    #[must_use]
    pub fn bytes_at(&self, region: u32, run: u32) -> Option<u64> {
        self.at(region, run).map(|at| at * 4)
    }
}

/// Every region's windows, deduplicated: a region holds a list of runs rather than one window, since P4's fallback is one launch per maximal interval; an empty window is its own entry so a region with no rows resolves.
#[derive(Debug, Clone, Default)]
pub struct Windows {
    windows: Vec<Window>,
    /// Every region's runs end to end, as positions in [`windows`](Windows::windows): region `r`'s are `runs[of_region[r].0 .. of_region[r].0 + of_region[r].1]`.
    runs: Vec<u32>,
    /// Region index → `(where its runs start, how many)`.
    of_region: Vec<(u32, u32)>,
    /// Which row axis each region's own window is a span of, one entry per template region. All `Tokens` on a one-unit artifact.
    axes: Vec<model_ir::RowAxis>,
    /// The carve [`packed`](Windows::packed) lays itself out in — the load's.
    slots: Slots,

    /// The live-rows seat's host side: four `u32` per (region ordinal, run), flat at [`Seat::at`], holding `[rows, row_offset, lanes, lane_offset]`; addressed by region, not dedup slot.
    live_words: Vec<u32>,
    /// The rectangle [`live_words`](Windows::live_words) is addressed in.
    seat: Seat,
    /// Where [`live`](Windows::live) landed on the device, or `0` for a fire that staged none.
    live_base: u64,

    /// The fire's qo boundaries, un-rebased — `[lanes + 1]`. Kept whole because a body bakes the whole-vector pointer and cannot bake a per-window slice of it (see [`qo_absolute`](Windows::qo_absolute)).
    qo_absolute_host: Vec<i32>,
    /// Where [`qo_absolute_host`](Windows::qo_absolute_host) landed on the device, or `0` if unbound.
    qo_absolute_base: u64,
    /// How many lanes the staged reading covers, or `0` for "exactly what [`qo_absolute_host`](Windows::qo_absolute_host) holds" — a bodied fire stages a copy padded to the key's ladder reach.
    qo_absolute_lanes: u32,
}

/// Is this region's work something the copy path can actually serve? A copy re-points every operand at a compacted rectangle, which works for a token-shaped tensor, a cache binding, and the four gathered geometry vectors; anything else takes the split instead.
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
            // Only the paged pool; a recurrent bank addressed by slot would hand a gathered window the wrong lanes' banks.
            Def::Cache(c) => matches!(
                trace.caches.get(*c as usize),
                Some(model_ir::CacheRow::Kv { .. })
            ),
            // The four geometry vectors `GatheredSpace` re-cuts.
            Def::Input(RuntimeInput::Geometry { kind, .. }) => matches!(
                kind,
                GeomKind::Indptr | GeomKind::Indices | GeomKind::LastPageLen | GeomKind::KvLen
            ),
            // The mask slab: bit-addressed, not row-addressed; not gatherable.
            Def::Input(RuntimeInput::Mask { .. }) => false,
            _ => match &decl.ty {
                // A plan payload: host state, not a rectangle.
                Ty::Struct(_) => true,
                Ty::Tensor { shape, .. } => match shape.first() {
                    // Row-shaped: the slab.
                    Some(Dim::Tokens) => true,
                    // `k` rectangle rows per token row: the row map is token-indexed, so this shape mismatches it.
                    Some(Dim::TokensTimes(_)) => false,
                    // Window-free: handed over whole, gathered or not.
                    Some(Dim::Const(_)) | None => true,
                    Some(Dim::Lanes | Dim::LanesPlus(_)) => false,
                    // The patch axis: a different row space than the token map `Gathered::rows_host` describes.
                    Some(Dim::Patches | Dim::Images | Dim::ImagesPlus(_)) => false,
                },
            },
        }
    })
}

impl Windows {
    /// The windows of one fire: every region of the template resolved
    /// against this composition's class table, one per interval its mask
    /// covers. `tables` is one window table per row axis; `slots` is the load's window carve.
    /// # Errors: [`Fault::Fragmented`] for a region whose classes aren't consecutive with no `Fallback` row; [`Fault::Ceiling`] for a window outrunning one slot's stride.
    pub fn of(
        trace: &Trace,
        compiled: &CompiledModel,
        tables: model_ir::PerAxis<&WindowTable>,
        indptr_host: &[i32],
        copies: Copies<'_>,
        slots: Slots,
    ) -> Result<Windows> {
        let mut windows: Vec<Window> = Vec::new();
        let mut runs: Vec<u32> = Vec::with_capacity(compiled.template().len());
        let mut of_region: Vec<(u32, u32)> = Vec::with_capacity(compiled.template().len());
        let mut axes: Vec<model_ir::RowAxis> = Vec::with_capacity(compiled.template().len());
        let mut spans: Vec<MaskSpan> = Vec::new();
        // The grid's segment axis: how many intervals the shipped order breaks any mask into.
        let segment_cap = fallback::max_runs(compiled);

        for (at, region) in compiled.template().iter().enumerate() {
            // Which table this region's own rows come from.
            let axis = compiled.axis_of(at);
            axes.push(axis);
            tables[axis].spans_into(&region.mask, &mut spans);
            // The patch axis's interval, for a token region whose embed merge reads a patch rectangle — resolved before this region's own window, bounding the patch axis to exactly one span.
            let patch = match tables[model_ir::RowAxis::Patches].span(&region.mask) {
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
            if spans.len() > 1 {
                // Was this window promised consecutive, and is the run count within what the shipped order can produce?
                let bound = fallback::bound(compiled, axis, &region.mask);
                if fallback::promised(compiled, axis, region) || spans.len() > bound as usize {
                    return Err(Fault::Fragmented {
                        region: at as u32,
                        runs: spans.len(),
                        promised: fallback::promised(compiled, axis, region).then_some(bound),
                    });
                }
                // `Fallback::Grouped`: one window over the union, carrying the intervals so the kernel can skip foreign rows between them.
                if fallback::grouped(compiled, axis, region.nodes.clone()) {
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
            // An empty mask answers the zero window.
            if spans.is_empty() {
                spans.push(MaskSpan::default());
            }

            // `Fallback::Copy`: a window in pieces whose bucket asks for a copy becomes one window over the compacted rectangle.
            if spans.len() > 1
                && copies.enabled
                && fallback::copies(compiled, axis, &region.mask, copies.bucket)
                && copyable(trace, region)
            {
                let mut gathered = gather_of(&spans, indptr_host, copies.spaces);
                gathered.spans[model_ir::RowAxis::Patches] = patch;
                seats(slots, &gathered)?;
                of_region.push((runs.len() as u32, 1));
                runs.push(insert(&mut windows, gathered));
                continue;
            }

            of_region.push((runs.len() as u32, spans.len() as u32));
            for &span in &spans {
                let window = Window {
                    // The region's own interval at the primary entry, the patch table's at the patch entry.
                    spans: model_ir::PerAxis::new([span, patch]),
                    // A patch region has no rebased qo boundaries; it has its own bounds vector (`RuntimeInput::PatchSegments`).
                    indptr_host: match axis {
                        model_ir::RowAxis::Tokens => rebase(indptr_host, span)?,
                        model_ir::RowAxis::Patches => Vec::new(),
                    },
                    indptr: Tensor::new(0, 0, 1, Dtype::I32),
                    gathered: None,
                    segments_host: segments_host.clone(),
                    segments: Tensor::new(0, 0, 2, Dtype::I32),
                    segment_cap,
                };
                seats(slots, &window)?;
                runs.push(insert(&mut windows, window));
            }
        }

        let mut table = Windows {
            windows,
            runs,
            of_region,
            axes,
            slots,
            live_words: Vec::new(),
            seat: Seat::default(),
            live_base: 0,
            qo_absolute_host: indptr_host.to_vec(),
            qo_absolute_base: 0,
            qo_absolute_lanes: 0,
        };
        table.fill_live();
        Ok(table)
    }

    /// Fill the live-geometry seat with the identity: every (region, run)'s own `[rows, row_offset, lanes, lane_offset]`.
    fn fill_live(&mut self) {
        let seat = Seat::new(self.of_region.len() as u64, u64::from(self.max_runs()));
        let mut live = vec![0u32; seat.words() as usize];
        for region in 0..self.of_region.len() as u32 {
            for run in 0..self.runs(region) {
                // In bounds by construction; skipped rather than clamped if not.
                let Some(at) = seat.at(region, run) else {
                    continue;
                };
                let at = at as usize;
                let span = self.at(region, run).span();
                live[at] = span.rows;
                live[at + 1] = span.row_offset;
                live[at + 2] = span.lanes;
                live[at + 3] = span.lane_offset;
            }
        }
        self.live_words = live;
        self.seat = seat;
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

    /// Every window's `i32` vectors, at their slot's own offset — what the shell stages in one copy. Slot `i` starts at `i * stride`; gathered payloads ride behind every slot at their own fixed offset.
    #[must_use]
    pub fn packed(&self) -> Vec<i32> {
        let mut out: Vec<i32> = Vec::new();
        // Each slot is opened by padding out to its own offset.
        let open = |out: &mut Vec<i32>, at: u64| {
            let at = at as usize;
            assert!(out.len() <= at, "a window's vectors overran the stride they were carved at");
            // Grow only: a truncation would turn a `Fault::Ceiling` into silently wrong bytes.
            if out.len() < at {
                out.resize(at, 0);
            }
        };
        for (slot, window) in self.windows.iter().enumerate() {
            open(&mut out, self.slots.at(slot));
            out.extend_from_slice(&window.indptr_host);
            out.extend_from_slice(&window.segments_host);
        }
        assert!(
            self.windows.iter().enumerate().all(|(slot, window)| {
                let words = (window.indptr_host.len() + window.segments_host.len()) as u64;
                self.slots.fits(words) && self.slots.at(slot) < self.slots.tail()
            }),
            "the packed window layout does not fit the carve it was reserved in",
        );
        open(&mut out, self.slots.tail());
        // Each gathered payload at its own offset, not where the last one ended.
        for (which, gathered) in self
            .windows
            .iter()
            .filter_map(|window| window.gathered.as_ref())
            .enumerate()
        {
            open(&mut out, self.slots.gathered_at(which as u64));
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

    /// Seat the staged vectors: `base` is where [`packed`](Windows::packed) landed on the device. Walks the same offsets `packed` writes.
    pub fn bind(&mut self, base: u64) {
        let slots = self.slots;
        // `cols`: a segment list is `[segs][2]`, every other vector `[n][1]`.
        let take = |at: &mut u64, entries: usize, cols: u32| {
            let here = *at;
            *at += entries as u64 * 4;
            Tensor::new(here, entries as u32 / cols.max(1), cols, Dtype::I32)
        };
        for (slot, window) in self.windows.iter_mut().enumerate() {
            let mut at = base + slots.at(slot) * 4;
            window.indptr = take(&mut at, window.indptr_host.len(), 1);
            window.segments = take(&mut at, window.segments_host.len(), 2);
        }
        for (which, gathered) in self
            .windows
            .iter_mut()
            .filter_map(|window| window.gathered.as_mut())
            .enumerate()
        {
            let mut at = base + slots.gathered_at(which as u64) * 4;
            gathered.rows = take(&mut at, gathered.rows_host.len(), 1);
            for space in &mut gathered.spaces {
                space.page_indptr = take(&mut at, space.page_indptr_host.len(), 1);
                space.page_indices = take(&mut at, space.page_indices_host.len(), 1);
                space.last_page_lens = take(&mut at, space.last_page_lens_host.len(), 1);
                space.kv_len = take(&mut at, space.kv_len_host.len(), 1);
            }
        }
    }

    /// The carve this table lays its packed blob out in.
    #[must_use]
    pub fn slots(&self) -> Slots {
        self.slots
    }

    /// The live-rows seat's words, in [`live_at`](Windows::live_at)'s order — four per (region, run), `[rows, row_offset, lanes, lane_offset]`.
    #[must_use]
    pub fn live(&self) -> &[u32] {
        &self.live_words
    }

    /// Seat the live-rows words: `base` is where [`live`](Windows::live) landed on the device, or `None` for a fire that staged none.
    pub fn bind_live(&mut self, base: Option<u64>) {
        self.live_base = base.unwrap_or(0);
    }

    /// The fire's qo boundaries with nothing subtracted, host side. Stays `[fire lanes + 1]` even when the staged device copy is padded — its length tells `Run::planning` how many lanes the fire actually brought.
    #[must_use]
    pub fn qo_absolute_host(&self) -> &[i32] {
        &self.qo_absolute_host
    }

    /// Seat that vector: `base` is where [`qo_absolute_host`](Windows::qo_absolute_host) landed on the device, or `None` for a fire that staged none.
    pub fn bind_qo_absolute(&mut self, base: Option<u64>) {
        self.qo_absolute_base = base.unwrap_or(0);
    }

    /// How many lanes of the staged device vector went over — the key's ladder reach on the bodies path, `0` elsewhere. Never shrinks.
    pub fn stage_qo_absolute(&mut self, lanes: u32) {
        self.qo_absolute_lanes = self.qo_absolute_lanes.max(lanes);
    }

    /// The whole fire's boundaries, absolute, or `None` if none staged. Handed to a launch uncut since `lane_offset` isn't fixed by a `BodyKey`; `[fire lanes + 1]` rows, or `[ceiling + 1]` for a padded bodied fire.
    #[must_use]
    pub fn qo_absolute(&self) -> Option<Tensor> {
        if self.qo_absolute_base == 0 || self.qo_absolute_host.is_empty() {
            return None;
        }
        Some(Tensor::new(
            self.qo_absolute_base,
            // The table's own length, or the padded ceiling for a bodied fire.
            (self.qo_absolute_lanes + 1).max(self.qo_absolute_host.len() as u32),
            1,
            Dtype::I32,
        ))
    }

    /// The address of one (region, run)'s live-geometry words (`[rows, row_offset, lanes, lane_offset]`), or `0` for a fire that bound no seat or a pair this table does not hold.
    #[must_use]
    pub fn live_at(&self, region: u32, run: u32) -> u64 {
        if self.live_base == 0 {
            return 0;
        }
        self.seat
            .bytes_at(region, run)
            .map_or(0, |at| self.live_base + at)
    }

    /// How many launches a region costs in this fire — `1` for a window P4 seated, `r` for one it could not, and `1` for an empty window.
    #[must_use]
    pub fn runs(&self, region: u32) -> u32 {
        self.of_region.get(region as usize).map_or(0, |held| held.1)
    }

    /// How many launches this fire's walk makes over the whole template.
    #[must_use]
    pub fn launches(&self) -> u32 {
        self.of_region.iter().map(|&(_, runs)| runs.max(1)).sum()
    }

    /// How many regions of this fire are served as a `Fallback::Copy`.
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

    /// The most launches any region of this fire costs — what a per-run table is sized at.
    #[must_use]
    pub fn max_runs(&self) -> u32 {
        self.of_region
            .iter()
            .map(|&(_, runs)| runs)
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// One region's window, for one run of it. Panics for a region or run this table does not hold — an integrity failure of the shell, since the cursor and the walk are both cut from the same template.
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

    /// What one recorded body may be replayed over: per template region, may a captured graph hold this region's launches, or must it re-issue ([`Admit`])? The narrow reading: every present region's window must BE the whole fire. Neither this nor the wide reading waives the shape clause.
    #[must_use]
    pub fn covers_fire(&self, rows: u32) -> bool {
        (0..self.of_region.len() as u32).all(|region| {
            (0..self.runs(region)).all(|run| {
                let window = self.at(region, run);
                window.span().rows == 0 || window.is_whole(rows)
            })
        })
    }

    /// The same question for a table whose regions can move their own base: [`covers_fire`](Windows::covers_fire) with offset and rows waived per region on [`crate::shifted`]; the shape clause is never waived.
    #[must_use]
    pub fn covers_fire_shifted(&self, rows: u32, shifted: &[bool], lane_shifted: &[bool]) -> bool {
        (0..self.of_region.len() as u32)
            .all(|region| {
                self.admit_axes(
                    region,
                    model_ir::PerAxis::new([rows, 0]),
                    shifted,
                    lane_shifted,
                ) == Admit::Captured
            })
    }

    /// Which regions of this fire a body may hold, and which it must re-issue — per-region rather than collapsed to one `bool`. A function of the [`record::BodyKey`](crate::record::BodyKey), except the copy knob, which a differently-armed fire walks eagerly instead of re-deriving.
    #[must_use]
    pub fn admits(&self, rows: u32, shifted: &[bool], lane_shifted: &[bool]) -> Vec<Admit> {
        self.admits_axes(model_ir::PerAxis::new([rows, 0]), shifted, lane_shifted)
    }

    /// The same table for an artifact with two row axes: every region is judged against its own axis's total ([`axis_of`](Windows::axis_of)), since judging a tower region against the token total would misclassify it. A patch region is never gathered, grouped, or in pieces.
    #[must_use]
    pub fn admits_axes(
        &self,
        totals: model_ir::PerAxis<u32>,
        shifted: &[bool],
        lane_shifted: &[bool],
    ) -> Vec<Admit> {
        (0..self.of_region.len() as u32)
            .map(|region| self.admit_axes(region, totals, shifted, lane_shifted))
            .collect()
    }

    /// Which row space this region's window counts, or [`RowAxis::Tokens`](model_ir::RowAxis::Tokens) for a region index past the table.
    #[must_use]
    pub fn axis_of(&self, region: u32) -> model_ir::RowAxis {
        self.axes
            .get(region as usize)
            .copied()
            .unwrap_or(model_ir::RowAxis::Tokens)
    }

    /// One region's entry of [`admits`](Windows::admits), judged against the total of its own axis.
    #[must_use]
    fn admit_axes(
        &self,
        region: u32,
        totals: model_ir::PerAxis<u32>,
        shifted: &[bool],
        lane_shifted: &[bool],
    ) -> Admit {
        let moves = shifted.get(region as usize).copied().unwrap_or(false);
        // The same question one axis over; an unheld index reads `false` (refuses).
        let finds_its_lane = lane_shifted.get(region as usize).copied().unwrap_or(false);
        let total = totals[self.axis_of(region)];
        let held = (0..self.runs(region)).all(|run| {
            let window = self.at(region, run);
            let span = window.span();
            span.rows == 0
                || (window.is_interval()
                    && (moves || (span.row_offset == 0 && span.rows >= total))
                    // The lane axis's own clause: a region reading per-lane tables sliced by `lane_offset` (not fixed by a `BodyKey`) would replay another lane's state.
                    && (span.lane_offset == 0 || finds_its_lane))
        });
        if held { Admit::Captured } else { Admit::Island }
    }
}

/// May a body hold this region's launches, or must it re-issue them? — one entry of [`Windows::admits`], per template region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admit {
    /// A graph may hold it: every run either has no rows, or is a window the staged seat can speak for, so it replays at every split of its key.
    Captured,
    /// Has to be re-issued every fire: some run is not an interval, or is windowed without every op reading the seat's start or lane. Runs eagerly at this fire's own live geometry.
    Island,
}

/// No attention schedule may be built over more classes than the node consuming it runs in. # Errors: [`Fault::Straddled`], naming the value, the consuming node, and the two class sets.
pub fn no_schedule_straddles_its_readers(trace: &Trace, compiled: &CompiledModel) -> Result<()> {
    Ok(check::no_schedule_straddles_its_readers(trace, compiled)?)
}

/// No grouped consumer shares its window with a prepare region: unlike
/// `Fallback::Copy`, `Fallback::Grouped` has no per-mask inheritance, so a
/// prepare builder sharing it would carve `r` schedules while the consumer ran once.
/// # Errors: [`Fault::Straddled`], naming the grouped region's first node and the two class sets.
pub fn no_grouped_window_is_also_a_prepare_window(compiled: &CompiledModel) -> Result<()> {
    for (at, region) in compiled.template().iter().enumerate() {
        if !fallback::grouped(compiled, compiled.axis_of(at), region.nodes.clone()) {
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

/// The one rectangle that contains every one of these intervals — a grouped launch is cut at the union and told which rows are its own. Caller must pass an ascending, non-empty list.
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

/// Refuses a window whose staged words don't fit one slot — taken before dedup, or a window past the stride would silently overwrite the next slot. # Errors: [`Fault::Ceiling`] naming the slot stride.
fn seats(slots: Slots, window: &Window) -> Result<()> {
    let words = (window.indptr_host.len() + window.segments_host.len()) as u64;
    if !slots.fits(words) {
        return Err(Fault::Ceiling {
            what: "one window slot's staged words",
            need: words,
            have: slots.stride(),
        });
    }
    Ok(())
}

/// Give this window a position in the fire's deduplicated list. Deduplicated on every field, not only the span; the position is a function of the `BodyKey` so it cannot drift between fires of one key.
fn insert(windows: &mut Vec<Window>, window: Window) -> u32 {
    let same = |held: &Window| {
        held.spans == window.spans
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

/// Build the gathered window a list of runs compacts to: the row map, the rebased qo boundaries over the union, and per-space pool tables re-cut lane by lane.
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
            // The lane's own row count, added to the running total (the rebase).
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
        // The compacted rectangle at the primary entry; the caller fills the patch entry from the patch table.
        spans: model_ir::PerAxis::new([
            MaskSpan {
                row_offset: 0,
                rows: rows_host.len() as u32,
                lane_offset: 0,
                lanes: lanes.len() as u32,
            },
            MaskSpan::default(),
        ]),
        indptr_host: bounds,
        indptr: Tensor::new(0, 0, 1, Dtype::I32),
        // A gathered rectangle holds only the consumer's own rows, so there is no segment list to write.
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

/// Where the walk is: which region of the template, and which run of that region's window. A `Cell` (not `&mut`) because `walk` takes the sink and the dispatch as two separate borrows.
#[derive(Debug, Default)]
pub struct At {
    /// The region index, in `CompiledModel::template` order.
    pub region: Cell<u32>,
    /// Which run of that region's window: `0` always, and `0..r` for a region P4 could not seat.
    pub run: Cell<u32>,
}

impl At {
    /// A cursor position at the top of the template.
    #[must_use]
    pub fn new() -> At {
        At::default()
    }
}

/// The stream handles and events a [`Cursor`] switches between. Handed in, never owned: the streams/events are the context's, opened once at load.
#[derive(Debug, Clone, Copy)]
pub struct Lanes<'a> {
    /// The side streams, in stream order: `side[0]` is stream 1. The main stream is not here — a region on stream 0 needs no lookup.
    pub side: &'a [*mut core::ffi::c_void],
    /// The main stream, which is what an event on stream 0 is recorded on.
    pub main: *mut core::ffi::c_void,
    /// One event per `EventId`, in id order.
    pub events: &'a [Event],
    /// Which stream the walk is on now.
    pub at: &'a Cell<u32>,
}

/// The sentinel `Lanes::at` carries while a conditional body is open — no artifact can name this stream index legitimately, since streams are numbered from zero.
pub const BODY: u32 = u32::MAX;

/// What a [`Cursor`] needs to put a conditional node in the graph it is recording. Handed in, never owned, like [`Lanes`].
#[derive(Clone, Copy)]
pub struct Conditionals<'a> {
    /// The stream the parent capture is on: where the handle is minted, the setter is launched and the node is placed.
    pub main: *mut core::ffi::c_void,
    /// The stream a body is captured on — opened at load, never enqueued on outside a `cuStreamBeginCaptureToGraph`.
    pub body: *mut core::ffi::c_void,
    /// The kernel context on [`main`](Conditionals::main), where the device-side setter's one launch goes.
    pub setter: &'a kernels_cuda::Ctx,
    /// This fire's windows: the setter reads a region's row count out of its staged boundary vector.
    pub windows: &'a Windows,
    /// Which stream the walk is on.
    pub at: &'a Cell<u32>,
}

/// What a [`Cursor`] needs to rotate a slot's contents at a region boundary. Handed in, never owned; only an eager cursor is given one, since a recording cursor's pump would bake one fire's copies into a graph.
#[derive(Clone, Copy)]
pub struct Pump<'a> {
    /// The load's rotor: the slots, the two event rings, and the copy stream.
    pub rotor: &'a crate::rotate::Rotor,
    /// The stream this fire's launches are on — where `free` is recorded and `ready` is waited.
    pub compute: *mut core::ffi::c_void,
}

/// This shell's [`Sink`]: the region counter a [`Run`](crate::run::Run) reads its window out of, and — when forked — the stream switch and event points. An eager cursor records nothing; a device fault inside a `Sink` method is kept and surfaced at [`Cursor::settle`].
pub struct Cursor<'a> {
    at: u32,
    place: &'a At,
    lanes: Option<Lanes<'a>>,
    /// Is this walk being written down? [`cond_begin`](Sink::cond_begin) reads it to distinguish ignoring a conditional (correct, eager) from recording its body unconditionally (silently wrong).
    recording: bool,
    /// The conditional machinery, when this load opened any. `None` refuses a conditional region by name.
    cond: Option<Conditionals<'a>>,
    /// The bracket currently open, the stream to restore when it closes, and whether a body capture is running right now. `Some` only between [`cond_begin`](Sink::cond_begin) and [`cond_end`](Sink::cond_end).
    open: Option<(crate::device::conditional::Conditional, u32, bool)>,
    /// The rotating dense pump, when this load armed one. `None` for every load whose weights are where the fire expects them.
    pump: Option<Pump<'a>>,
    fault: Option<Fault>,
}

/// By hand because one field is a kernel context (a stream plus opaque handles) and derives nothing.
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
    #[must_use]
    pub fn writing(self) -> Cursor<'a> {
        Cursor {
            recording: true,
            ..self
        }
    }

    /// The same, plus switching streams at every region boundary and putting the baked event points on the device.
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

    /// The same cursor, told where to put a conditional node. Only a recording walk is given it — an eager pass ignores the bracket correctly, since the walk's zero-row rule decides the same thing.
    #[must_use]
    pub fn conditionals(self, cond: Conditionals<'a>) -> Cursor<'a> {
        Cursor {
            cond: Some(cond),
            ..self
        }
    }

    /// The same cursor, told to rotate this load's dense slots. Only an eager pass is given it — see [`Pump`].
    #[must_use]
    pub fn pumping(self, pump: Pump<'a>) -> Cursor<'a> {
        Cursor {
            pump: Some(pump),
            ..self
        }
    }

    /// What the device refused during the walk, if anything. # Errors: [`Fault::Device`] from a `cudaEventRecord` or `cudaStreamWaitEvent`, or [`Fault::Unbound`] for a template naming a stream/event this load never opened.
    pub fn settle(mut self) -> Result<()> {
        // A bracket left open is closed here: a body stream left mid-capture answers every later call with `cudaErrorStreamCaptureUnjoined` for the rest of the process.
        self.cond_end();
        match self.fault {
            Some(fault) => Err(fault),
            None => Ok(()),
        }
    }

    /// What the setter reads for one region: the device address of its window's rebased row CSR, the lane count to index it at, whether this region can state a count, and its live-geometry seat address. The CSR pointer is key-stable; the lane count is not, so the live seat corrects a replay.
    fn count_of(&self, cond: Conditionals<'a>, region: u32) -> (u64, u32, bool, u64) {
        match cond.windows.runs(region) {
            1 if !cond.windows.at(region, 0).indptr_host.is_empty() => {
                let window = cond.windows.at(region, 0);
                let lanes = window.indptr_host.len().saturating_sub(1) as u32;
                (
                    window.indptr.ptr,
                    lanes,
                    false,
                    cond.windows.live_at(region, 0),
                )
            }
            _ => (0, 0, true, 0),
        }
    }

    /// Close whatever body is recording and begin `arm`'s, leaving the walk's launches pointed at the body stream. The cell is set to `BODY` only while a capture is actually running.
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

    /// The stream the current region is on, or a fault for a region naming one this load did not open.
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
        // Before this region dispatches: release slots the previous region freed, issue due copies, and make compute wait on planes this region reads. All enqueues; nothing here synchronizes.
        if let Some(pump) = self.pump
            && self.fault.is_none()
            && let Err(fault) = pump.rotor.at(self.at, pump.compute)
        {
            self.fault = Some(fault);
        }
        self.at += 1;
        // The stream switch: everything the `Run` resolves afterwards fires on whatever this names. Skipped while a bracket is open, so a write between a SWITCH's arms doesn't put the next arm's launches on the main stream mid-capture.
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

    /// A region P4 could not seat runs once per interval of its class set; every operand the `Run` resolves after this call is cut at this interval.
    fn run(&mut self, run: u32, _runs: u32) {
        self.place.run.set(run);
    }

    /// The eager cursor ignores this; the recording one records a node. A
    /// capture cannot ignore it, since the graph outlives the fire, so it
    /// places a real conditional node predicated on a kernel reading the row count off the device.
    /// A `SWITCH` is the same node asked `arms` times over `arms` consecutive regions. A region that cannot state a count takes its body unconditionally for an `IF` but refuses a `SWITCH` ([`Fault::Unlowered`]).
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
                // One question, one region: the sole launch reads `indptr[lanes]`; an absent table means "take it".
                Kind::If => {
                    let (indptr, lanes, absent, win) = self.count_of(cond, region);
                    kernels_cuda::graph::set_conditional(
                        cond.setter,
                        handle,
                        indptr,
                        lanes,
                        absent,
                        kernels_cuda::graph::Arm::Set,
                        win,
                    )
                    .map_err(|why| Fault::Unbound {
                        what: format!(
                            "the conditional setter for region {region}, which answered {why}"
                        ),
                    })?;
                }
                // The arms are consecutive regions, so `region + arm` is that arm's window.
                Kind::Switch { arms } => {
                    for arm in 0..arms {
                        let (indptr, lanes, absent, win) = self.count_of(cond, region + arm);
                        // A `SWITCH` has no "take it anyway" direction.
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
                            win,
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
                // An `IF` has no `cond_arm`; its body opens here.
                if kind == Kind::If {
                    self.enter(0);
                }
            }
            Err(fault) => self.fault = Some(fault),
        }
    }

    /// One arm of a `SWITCH`: closes whatever body was recording and opens this arm's. Never called for an `IF`.
    fn cond_arm(&mut self, arm: u8) {
        if !self.recording || self.open.is_none() {
            return;
        }
        self.enter(u32::from(arm));
    }

    /// Close the body and put the walk back on the stream the region named, even when the walk faulted inside it, so no stream is left mid-capture.
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
    
    use model_ir::ClassSet;

    /// A windowed region in the capture phase, behind a conditional node.
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
            axis: None,
            collective: false,
        }
    }

    #[test]
    fn a_recording_cursor_with_nowhere_to_put_a_conditional_still_refuses_it() {
        let cell = At::new();
        let mut eager = Cursor::new(&cell);
        let region = conditional();
        eager.region_begin(&region);
        eager.cond_begin(&region.lowering);
        eager.cond_end();
        eager.region_end(&region);
        // Correct, not a shortcut: the walk's zero-row rule decides what the conditional decides, so an eager pass runs the same rows anyway.
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

    // The live-rows seat: pure arithmetic over a table built by hand.

    /// A window with nothing but a span — every other field at the shape an ordinary (ungathered, ungrouped) one has.
    fn plain(row_offset: u32, rows: u32) -> Window {
        Window {
            spans: model_ir::PerAxis::new([
                MaskSpan {
                    row_offset,
                    rows,
                    lane_offset: 0,
                    lanes: 1,
                },
                MaskSpan::default(),
            ]),
            indptr_host: Vec::new(),
            indptr: Tensor::new(0, 0, 1, Dtype::I32),
            segments_host: Vec::new(),
            segments: Tensor::new(0, 0, 2, Dtype::I32),
            segment_cap: 0,
            gathered: None,
        }
    }

    /// A table of `spans[region][run]`, seated the way `Windows::of` seats one — one window per run, and the live words filled with the identity.
    fn windows(spans: &[Vec<Window>]) -> Windows {
        let mut table = Windows {
            windows: Vec::new(),
            runs: Vec::new(),
            of_region: Vec::new(),
            axes: Vec::new(),
            // A carve wide enough for anything these tables hold.
            slots: Slots::new(2, 8, 1, 1, 64, 1, 32),
            live_words: Vec::new(),
            seat: Seat::default(),
            live_base: 0,
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
            table.axes.push(model_ir::RowAxis::Tokens);
        }
        table.fill_live();
        table
    }

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
                let span = table.at(region, run).span();
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
        // Untouched zeros for a run the table did not cut.
        assert_eq!(&table.live()[4..8], &[0, 0, 0, 0]);
    }

    // The packed blob's per-window addresses are a function of the `record::BodyKey`, not of the fire, since a body bakes them.

    /// A plain window carrying the one vector the packed blob is made of: a rebased `[lanes + 1]` boundary list, whose length moves between fires of one key.
    fn bounded(span: MaskSpan) -> Window {
        Window {
            spans: model_ir::PerAxis::new([span, MaskSpan::default()]),
            indptr_host: (0..=span.lanes as i32).collect(),
            ..plain(0, 0)
        }
    }

    /// The three spans a two-class fire resolves for the masks `{A}`, `{A,B}` and `{B}`: a class with no rows contributes nothing. `a` and `b` are each `(rows, lanes)`.
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

    /// The slot itself is the key's: the span is this fire's encoding of `mask ∩ present`, so which masks share a slot is fixed by which classes have rows — exactly what a `BodyKey` carries.
    #[test]
    fn two_masks_share_a_slot_in_every_fire_of_a_key_or_in_none_of_them() {
        // Four regions over three masks, seated the way `Windows::of` seats them.
        let seated = |a, b| {
            let [first, both, second] = resolved(a, b);
            let mut held: Vec<Window> = Vec::new();
            let map: Vec<u32> = [first, both, first, second]
                .into_iter()
                .map(|span| insert(&mut held, bounded(span)))
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

        // The one split that moves the sharing moves the key with it: a class with no rows is absent from `BodyKey::classes`.
        let absent = seated((10, 2), (0, 0));
        assert_eq!(
            absent.0,
            vec![0, 0, 0, 1],
            "the both-classes mask is the A mask when B has no rows",
        );
        assert_ne!(absent.0, wide.0, "and that is a different key, so a different body");
    }
}
