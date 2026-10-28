use std::cell::Cell;

use crate::device::conditional::Kind;
use crate::device::graph::Event;
use kernels_cuda::Tensor;
use poem_compiler::{CompiledModel, Lowering, Phase, Region};
use poem_exec::fire::{EventId, MaskSpan, Sink, WindowTable, fallback};
use poem_exec::store::check::{self, rebase};
use poem_ir::{Def, Dim, Dtype, GeomKind, Operands, RuntimeInput, Trace, Ty};

use crate::error::{Fault, Result};
use crate::store::kv::Geometry;

#[derive(Debug, Clone)]
pub struct Window {
    pub spans: poem_ir::PerAxis<MaskSpan>,
    pub indptr_host: Vec<i32>,
    pub indptr: Tensor,

    pub segments_host: Vec<i32>,
    pub segments: Tensor,
    pub segment_cap: u32,
    pub gathered: Option<Gathered>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowShape {
    Interval,
    Gathered,
    Grouped,
}

impl Window {
    /// A slot no region of this fire cuts: no rows, no tables.
    #[must_use]
    pub fn empty() -> Window {
        Window {
            spans: poem_ir::PerAxis::default(),
            indptr_host: Vec::new(),
            indptr: Tensor::new(0, 0, 1, Dtype::I32),
            segments_host: Vec::new(),
            segments: Tensor::new(0, 0, 2, Dtype::I32),
            segment_cap: 0,
            gathered: None,
        }
    }

    #[must_use]
    pub fn on(&self, axis: poem_ir::RowAxis) -> MaskSpan {
        self.spans[axis]
    }

    #[must_use]
    pub fn span(&self) -> MaskSpan {
        self.spans[poem_ir::RowAxis::PRIMARY]
    }

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

    #[must_use]
    pub fn is_interval(&self) -> bool {
        matches!(self.shape(), WindowShape::Interval)
    }

    /// Whether its ops launch over one span of rows: an interval, or a
    /// grouped window's union, which a grouped op walks whole
    /// ([`crate::GROUPED`]), the runs' own rows read off its segments.
    #[must_use]
    pub fn launches_one_span(&self) -> bool {
        matches!(self.shape(), WindowShape::Interval | WindowShape::Grouped)
    }

    #[must_use]
    pub fn is_whole(&self, total: u32) -> bool {
        let span = self.span();
        span.row_offset == 0 && span.rows >= total && self.is_interval()
    }

    #[must_use]
    pub fn segs(&self) -> u32 {
        self.segments_host.len() as u32 / 2
    }

    #[must_use]
    pub fn segment_rows(&self) -> u32 {
        self.segments_host
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| pair[1].max(0) as u32)
            .max()
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct Gathered {
    pub runs: Vec<MaskSpan>,
    pub rows_host: Vec<i32>,
    pub rows: Tensor,
    pub spaces: Vec<GatheredSpace>,
}

#[derive(Debug, Clone)]
pub struct GatheredSpace {
    pub page_indptr_host: Vec<i32>,
    pub page_indices_host: Vec<i32>,
    pub last_page_lens_host: Vec<i32>,
    pub kv_len_host: Vec<i32>,
    pub page_indptr: Tensor,
    pub page_indices: Tensor,
    pub last_page_lens: Tensor,
    pub kv_len: Tensor,
}

#[derive(Debug, Clone, Copy)]
pub struct Copies<'a> {
    pub bucket: u32,
    pub enabled: bool,
    pub spaces: &'a [Geometry],
}

impl Copies<'_> {
    #[must_use]
    pub fn off() -> Copies<'static> {
        Copies {
            bucket: 0,
            enabled: false,
            spaces: &[],
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Slots {
    stride: u64,
    slots: u64,
    gathered: u64,
    gathered_stride: u64,
}

#[must_use]
pub const fn lane_capacity(max_lanes: u32) -> u32 {
    max_lanes.saturating_mul(2)
}

impl Slots {
    #[must_use]
    /// `windows` slots ([`slot_bases`]) of `lanes` lanes and `segs` segments.
    pub fn new(
        windows: usize,
        lanes: u64,
        segs: u32,
        gathered: usize,
        rows: u64,
        spaces: usize,
        pages: u64,
    ) -> Slots {
        Slots {
            stride: lanes + 1 + 2 * u64::from(segs.max(1)),
            slots: (windows + gathered) as u64,
            gathered: gathered as u64,
            gathered_stride: rows + spaces as u64 * (3 * lanes + 1 + pages),
        }
    }

    #[must_use]
    pub fn stride(&self) -> u64 {
        self.stride
    }

    #[must_use]
    pub fn tail(&self) -> u64 {
        self.slots * self.stride
    }

    #[must_use]
    pub fn at(&self, slot: usize) -> u64 {
        slot as u64 * self.stride
    }

    #[must_use]
    pub fn gathered_at(&self, which: u64) -> u64 {
        self.tail() + which * self.gathered_stride
    }

    #[must_use]
    pub fn words(&self) -> u64 {
        self.gathered_at(self.gathered)
    }

    #[must_use]
    pub fn fits(&self, words: u64) -> bool {
        words <= self.stride
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Seat {
    regions: u64,
    runs: u64,
}

impl Seat {
    pub const WORDS: u64 = 4;

    #[must_use]
    pub fn new(regions: u64, runs: u64) -> Seat {
        Seat { regions, runs }
    }

    #[must_use]
    pub fn words(&self) -> u64 {
        self.regions * self.runs * Self::WORDS
    }

    #[must_use]
    pub fn at(&self, region: u32, run: u32) -> Option<u64> {
        let (region, run) = (u64::from(region), u64::from(run));
        if region >= self.regions || run >= self.runs {
            return None;
        }
        Some((region * self.runs + run) * Self::WORDS)
    }

    #[must_use]
    pub fn bytes_at(&self, region: u32, run: u32) -> Option<u64> {
        self.at(region, run).map(|at| at * 4)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Windows {
    windows: Vec<Window>,
    runs: Vec<u32>,
    of_region: Vec<(u32, u32)>,
    axes: Vec<poem_ir::RowAxis>,
    slots: Slots,

    live_words: Vec<u32>,
    seat: Seat,
    live_base: u64,

    qo_absolute_host: Vec<i32>,
    qo_absolute_base: u64,
    qo_absolute_lanes: u32,
}

fn copyable(trace: &Trace, region: &Region) -> bool {
    let mut operands: Vec<poem_ir::ValueId> = Vec::new();
    for node in region.nodes.clone() {
        let Some(node) = trace.nodes.get(node as usize) else {
            return false;
        };
        node.op.inputs(&mut operands);
        node.op.outputs(&mut operands);
    }
    operands.iter().all(|id| {
        let Some(decl) = trace.values.get(id.0 as usize) else {
            return false;
        };
        match &decl.def {
            Def::Cache(c) => matches!(
                trace.caches.get(*c as usize),
                Some(poem_ir::CacheRow::Kv { .. })
            ),
            Def::Input(RuntimeInput::Geometry { kind, .. }) => matches!(
                kind,
                GeomKind::Indptr | GeomKind::Indices | GeomKind::LastPageLen | GeomKind::KvLen
            ),
            Def::Input(RuntimeInput::Mask { .. }) => false,
            _ => match &decl.ty {
                Ty::Struct(_) => true,
                Ty::Tensor { shape, .. } => match shape.first() {
                    Some(Dim::Tokens) => true,
                    Some(Dim::TokensTimes(_)) => false,
                    Some(Dim::Const(_)) | None => true,
                    Some(Dim::Lanes | Dim::LanesPlus(_)) => false,
                    Some(Dim::Readouts) => false,
                    Some(Dim::Patches | Dim::Images | Dim::ImagesPlus(_)) => false,
                    Some(Dim::Voxels | Dim::VoxelsScaled(..) | Dim::Clips | Dim::ClipsPlus(_)) => {
                        false
                    }
                },
            },
        }
    })
}

/// Each template region's first window slot, and how many slots the
/// template takes. A region's slots follow from its class mask alone (one a
/// run it may cut into, regions of one mask sharing them), never from which
/// windows a fire's composition happens to hold: a body reads each window's
/// tables where it was recorded whichever of its key's classes a fire seats.
#[must_use]
pub fn slot_bases(compiled: &CompiledModel) -> (Vec<u32>, usize) {
    bases_of(compiled.template().iter().enumerate().map(|(at, region)| {
        let axis = compiled.axis_of(at);
        (
            axis,
            &region.mask,
            fallback::bound(compiled, axis, &region.mask),
        )
    }))
}

/// [`slot_bases`] over each region's axis, mask and run bound.
fn bases_of<'a>(
    regions: impl Iterator<Item = (poem_ir::RowAxis, &'a poem_ir::ClassSet, u32)>,
) -> (Vec<u32>, usize) {
    let mut held: Vec<(poem_ir::RowAxis, &poem_ir::ClassSet, u32)> = Vec::new();
    let mut next = 0u32;
    let bases = regions
        .map(|(axis, mask, bound)| {
            if let Some(&(_, _, base)) = held.iter().find(|(a, m, _)| *a == axis && *m == mask) {
                return base;
            }
            let base = next;
            next += bound.max(1);
            held.push((axis, mask, base));
            base
        })
        .collect();
    (bases, next as usize)
}

impl Windows {
    pub fn of(
        trace: &Trace,
        compiled: &CompiledModel,
        tables: poem_ir::PerAxis<&WindowTable>,
        indptr_host: &[i32],
        copies: Copies<'_>,
        slots: Slots,
        order: &[u8],
    ) -> Result<Windows> {
        let (bases, count) = slot_bases(compiled);
        let mut windows: Vec<Window> = vec![Window::empty(); count];
        let mut runs: Vec<u32> = Vec::with_capacity(compiled.template().len());
        let mut of_region: Vec<(u32, u32)> = Vec::with_capacity(compiled.template().len());
        let mut axes: Vec<poem_ir::RowAxis> = Vec::with_capacity(compiled.template().len());
        let mut spans: Vec<MaskSpan> = Vec::new();
        let segment_cap = fallback::max_runs(compiled);

        for (at, region) in compiled.template().iter().enumerate() {
            let axis = compiled.axis_of(at);
            axes.push(axis);
            // A wide key cuts each token-axis region into the runs its classes
            // form in the tier's order (`FireDescriptor::order`), as the walk
            // does.
            if axis == poem_ir::RowAxis::Tokens && !order.is_empty() {
                tables[axis].spans_by_order_into(&region.mask, order, &mut spans);
            } else {
                tables[axis].spans_into(&region.mask, &mut spans);
            }
            let patch = match tables[poem_ir::RowAxis::Patches].span(&region.mask) {
                Ok(span) => span.unwrap_or_default(),
                Err(runs) => {
                    return Err(Fault::Fragmented {
                        region: at as u32,
                        runs,
                        promised: None,
                    });
                }
            };
            let voxel = match tables[poem_ir::RowAxis::Voxels].span(&region.mask) {
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
                let bound = fallback::bound(compiled, axis, &region.mask);
                if fallback::promised(compiled, axis, region) || spans.len() > bound as usize {
                    return Err(Fault::Fragmented {
                        region: at as u32,
                        runs: spans.len(),
                        promised: fallback::promised(compiled, axis, region).then_some(bound),
                    });
                }
                if fallback::grouped(compiled, axis, region.nodes.clone()) {
                    // A run a wide key cuts that the fire seats no row of
                    // (`spans_by_order_into`) is no segment of the union.
                    spans.retain(|span| span.rows > 0);
                    let union = union_of(&spans);
                    segments_host = spans
                        .iter()
                        .flat_map(|span| {
                            [
                                (span.row_offset - union.row_offset) as i32,
                                span.rows as i32,
                            ]
                        })
                        .collect();
                    spans.clear();
                    spans.push(union);
                }
            }
            if spans.is_empty() {
                spans.push(MaskSpan::default());
            }

            if spans.len() > 1
                && copies.enabled
                && fallback::copies(compiled, axis, &region.mask, copies.bucket)
                && copyable(trace, region)
            {
                let mut gathered = gather_of(&spans, indptr_host, copies.spaces);
                gathered.spans[poem_ir::RowAxis::Patches] = patch;
                gathered.spans[poem_ir::RowAxis::Voxels] = voxel;
                seats(slots, &gathered)?;
                of_region.push((runs.len() as u32, 1));
                runs.push(place(&mut windows, bases[at], gathered));
                continue;
            }

            of_region.push((runs.len() as u32, spans.len() as u32));
            for (run, &span) in spans.iter().enumerate() {
                let window = Window {
                    spans: poem_ir::PerAxis::new([span, patch, voxel]),
                    indptr_host: match axis {
                        poem_ir::RowAxis::Tokens => rebase(indptr_host, span)?,
                        poem_ir::RowAxis::Patches | poem_ir::RowAxis::Voxels => Vec::new(),
                    },
                    indptr: Tensor::new(0, 0, 1, Dtype::I32),
                    gathered: None,
                    segments_host: segments_host.clone(),
                    segments: Tensor::new(0, 0, 2, Dtype::I32),
                    segment_cap,
                };
                seats(slots, &window)?;
                runs.push(place(&mut windows, bases[at] + run as u32, window));
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

    fn fill_live(&mut self) {
        let seat = Seat::new(self.of_region.len() as u64, u64::from(self.max_runs()));
        let mut live = vec![0u32; seat.words() as usize];
        for region in 0..self.of_region.len() as u32 {
            for run in 0..self.runs(region) {
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

    #[must_use]
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    #[must_use]
    pub fn packed(&self) -> Vec<i32> {
        let mut out: Vec<i32> = Vec::new();
        let open = |out: &mut Vec<i32>, at: u64| {
            let at = at as usize;
            assert!(
                out.len() <= at,
                "a window's vectors overran the stride they were carved at"
            );
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

    pub fn bind(&mut self, base: u64) {
        let slots = self.slots;
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

    #[must_use]
    pub fn slots(&self) -> Slots {
        self.slots
    }

    #[must_use]
    pub fn live(&self) -> &[u32] {
        &self.live_words
    }

    pub fn bind_live(&mut self, base: Option<u64>) {
        self.live_base = base.unwrap_or(0);
    }

    #[must_use]
    pub fn qo_absolute_host(&self) -> &[i32] {
        &self.qo_absolute_host
    }

    pub fn bind_qo_absolute(&mut self, base: Option<u64>) {
        self.qo_absolute_base = base.unwrap_or(0);
    }

    pub fn stage_qo_absolute(&mut self, lanes: u32) {
        self.qo_absolute_lanes = self.qo_absolute_lanes.max(lanes);
    }

    #[must_use]
    pub fn qo_absolute(&self) -> Option<Tensor> {
        if self.qo_absolute_base == 0 || self.qo_absolute_host.is_empty() {
            return None;
        }
        Some(Tensor::new(
            self.qo_absolute_base,
            (self.qo_absolute_lanes + 1).max(self.qo_absolute_host.len() as u32),
            1,
            Dtype::I32,
        ))
    }

    #[must_use]
    pub fn live_at(&self, region: u32, run: u32) -> u64 {
        if self.live_base == 0 {
            return 0;
        }
        self.seat
            .bytes_at(region, run)
            .map_or(0, |at| self.live_base + at)
    }

    #[must_use]
    pub fn runs(&self, region: u32) -> u32 {
        self.of_region.get(region as usize).map_or(0, |held| held.1)
    }

    #[must_use]
    pub fn launches(&self) -> u32 {
        self.of_region.iter().map(|&(_, runs)| runs.max(1)).sum()
    }

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

    #[must_use]
    pub fn max_runs(&self) -> u32 {
        self.of_region
            .iter()
            .map(|&(_, runs)| runs)
            .max()
            .unwrap_or(1)
            .max(1)
    }

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

    #[must_use]
    pub fn covers_fire(&self, rows: u32) -> bool {
        (0..self.of_region.len() as u32).all(|region| {
            (0..self.runs(region)).all(|run| {
                let window = self.at(region, run);
                window.span().rows == 0 || window.is_whole(rows)
            })
        })
    }

    #[must_use]
    pub fn covers_fire_shifted(&self, rows: u32, shifted: &[bool], lane_shifted: &[bool]) -> bool {
        (0..self.of_region.len() as u32).all(|region| {
            self.admit_axes(
                region,
                poem_ir::PerAxis::new([rows, 0, 0]),
                shifted,
                lane_shifted,
                Staging::NONE,
            ) == Admit::Captured
        })
    }

    #[must_use]
    pub fn admits(&self, rows: u32, shifted: &[bool], lane_shifted: &[bool]) -> Vec<Admit> {
        self.admits_axes(
            poem_ir::PerAxis::new([rows, 0, 0]),
            shifted,
            lane_shifted,
            Staging::NONE,
        )
    }

    #[must_use]
    pub fn admits_axes(
        &self,
        totals: poem_ir::PerAxis<u32>,
        shifted: &[bool],
        lane_shifted: &[bool],
        staging: Staging<'_>,
    ) -> Vec<Admit> {
        (0..self.of_region.len() as u32)
            .map(|region| self.admit_axes(region, totals, shifted, lane_shifted, staging))
            .collect()
    }

    #[must_use]
    pub fn axis_of(&self, region: u32) -> poem_ir::RowAxis {
        self.axes
            .get(region as usize)
            .copied()
            .unwrap_or(poem_ir::RowAxis::Tokens)
    }

    #[must_use]
    fn admit_axes(
        &self,
        region: u32,
        totals: poem_ir::PerAxis<u32>,
        shifted: &[bool],
        lane_shifted: &[bool],
        staging: Staging<'_>,
    ) -> Admit {
        if self.axis_of(region) == poem_ir::RowAxis::Voxels {
            return Admit::Island;
        }
        // A staging attention at a prefill's width launches eagerly: what it
        // stages is sized by this fire's page tables (`regions_staging`).
        if staging.from > 0
            && staging
                .regions
                .get(region as usize)
                .copied()
                .unwrap_or(false)
            && (0..self.runs(region)).any(|run| self.at(region, run).span().rows >= staging.from)
        {
            return Admit::Island;
        }
        let moves = shifted.get(region as usize).copied().unwrap_or(false);
        let finds_its_lane = lane_shifted.get(region as usize).copied().unwrap_or(false);
        let total = totals[self.axis_of(region)];
        let held = (0..self.runs(region)).all(|run| {
            let window = self.at(region, run);
            let span = window.span();
            let admitted = span.rows == 0
                || (window.launches_one_span()
                    && (moves || (span.row_offset == 0 && span.rows >= total))
                    && (span.lane_offset == 0 || finds_its_lane));
            if !admitted && crate::serve::diag::on().arm_trace {
                eprintln!(
                    "[admit] region {region} run {run} islands: rows={} row_offset={} lanes={} \
                     lane_offset={} of {total}, interval={}, moves={moves}, \
                     finds_its_lane={finds_its_lane}, segments={}",
                    span.rows,
                    span.row_offset,
                    span.lanes,
                    span.lane_offset,
                    window.is_interval(),
                    window.segs()
                );
            }
            admitted
        });
        if held { Admit::Captured } else { Admit::Island }
    }
}

/// The regions whose DeepSeek attention stages its quantized rows
/// (`exports::regions_staging`), and the rows from which it does
/// (`kernel.sdpa.compressed_prefill_from`; 0: none does).
#[derive(Debug, Clone, Copy)]
pub struct Staging<'a> {
    pub regions: &'a [bool],
    pub from: u32,
}

impl Staging<'_> {
    pub const NONE: Staging<'static> = Staging {
        regions: &[],
        from: 0,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admit {
    Captured,
    Island,
}

pub fn no_schedule_straddles_its_readers(trace: &Trace, compiled: &CompiledModel) -> Result<()> {
    Ok(check::no_schedule_straddles_its_readers(trace, compiled)?)
}

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

/// Seats `window` at `slot` ([`slot_bases`]): regions of one mask cut one
/// window, so a second lands where the first did.
fn place(windows: &mut [Window], slot: u32, window: Window) -> u32 {
    if let Some(held) = windows.get_mut(slot as usize) {
        *held = window;
    }
    slot
}

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
        spans: poem_ir::PerAxis::new([
            MaskSpan {
                row_offset: 0,
                rows: rows_host.len() as u32,
                lane_offset: 0,
                lanes: lanes.len() as u32,
            },
            MaskSpan::default(),
            MaskSpan::default(),
        ]),
        indptr_host: bounds,
        indptr: Tensor::new(0, 0, 1, Dtype::I32),
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

#[derive(Debug, Default)]
pub struct At {
    pub region: Cell<u32>,
    pub run: Cell<u32>,
    pub lane: Cell<u32>,
}

impl At {
    #[must_use]
    pub fn new() -> At {
        At::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Lanes<'a> {
    pub side: &'a [*mut core::ffi::c_void],
    pub main: *mut core::ffi::c_void,
    pub events: &'a [Event],
    pub at: &'a Cell<u32>,
}

pub const BODY: u32 = u32::MAX;

#[derive(Clone, Copy)]
pub struct Conditionals<'a> {
    pub main: *mut core::ffi::c_void,
    pub body: *mut core::ffi::c_void,
    pub setter: &'a kernels_cuda::Ctx,
    pub windows: &'a Windows,
    pub at: &'a Cell<u32>,
}

#[derive(Clone, Copy)]
pub struct Pump<'a> {
    pub rotor: &'a crate::rotate::Rotor,
    pub compute: *mut core::ffi::c_void,
}

pub struct Cursor<'a> {
    at: u32,
    place: &'a At,
    lanes: Option<Lanes<'a>>,
    recording: bool,
    cond: Option<Conditionals<'a>>,
    open: Option<(crate::device::conditional::Conditional, u32, bool)>,
    pump: Option<Pump<'a>>,
    marks: Option<Marks<'a>>,
    fault: Option<Fault>,
}

/// While a body records: the nodes each gated region added to the capture
/// (`FireDescriptor::gated`). The walk adds one region at a time, so what
/// the capture holds after a region and did not before is that region's.
struct Marks<'a> {
    gated: &'a [bool],
    stream: *mut core::ffi::c_void,
    begin: Option<(u32, std::collections::HashSet<usize>)>,
    regions: Vec<RegionMark>,
}

impl Marks<'_> {
    /// Closes the run under way, the capture's nodes since it began its own,
    /// and opens `next` where one follows.
    fn close(&mut self, region: u32, next: Option<u32>) -> Result<()> {
        let Some((run, begin)) = self.begin.take() else {
            return Ok(());
        };
        let after = crate::device::conditional::captured_nodes(self.stream)?;
        let nodes: Vec<usize> = after.iter().copied().filter(|node| !begin.contains(node)).collect();
        self.regions.push(RegionMark { region, run, nodes });
        if let Some(next) = next {
            self.begin = Some((next, after.into_iter().collect()));
        }
        Ok(())
    }
}

/// A run of a gated region and the nodes it added to a capture ([`Marks`]).
pub struct RegionMark {
    pub region: u32,
    pub run: u32,
    pub nodes: Vec<usize>,
}

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
    #[must_use]
    pub fn new(place: &'a At) -> Cursor<'a> {
        place.region.set(0);
        place.run.set(0);
        place.lane.set(0);
        Cursor {
            at: 0,
            place,
            lanes: None,
            recording: false,
            cond: None,
            open: None,
            pump: None,
            marks: None,
            fault: None,
        }
    }

    #[must_use]
    pub fn writing(self) -> Cursor<'a> {
        Cursor {
            recording: true,
            ..self
        }
    }

    #[must_use]
    pub fn across(place: &'a At, lanes: Lanes<'a>) -> Cursor<'a> {
        place.region.set(0);
        place.run.set(0);
        place.lane.set(0);
        lanes.at.set(0);
        Cursor {
            at: 0,
            place,
            lanes: Some(lanes),
            recording: false,
            cond: None,
            open: None,
            pump: None,
            marks: None,
            fault: None,
        }
    }

    #[must_use]
    pub fn conditionals(self, cond: Conditionals<'a>) -> Cursor<'a> {
        Cursor {
            cond: Some(cond),
            ..self
        }
    }

    #[must_use]
    pub fn pumping(self, pump: Pump<'a>) -> Cursor<'a> {
        Cursor {
            pump: Some(pump),
            ..self
        }
    }

    /// Records the nodes each region `gated` holds adds to the capture
    /// under way on `stream` ([`Cursor::settle_marked`]).
    #[must_use]
    pub fn marking(self, stream: *mut core::ffi::c_void, gated: &'a [bool]) -> Cursor<'a> {
        Cursor {
            marks: Some(Marks {
                gated,
                stream,
                begin: None,
                regions: Vec::new(),
            }),
            ..self
        }
    }

    /// [`Cursor::settle`], answering the nodes each gated region added.
    pub fn settle_marked(mut self) -> Result<Vec<RegionMark>> {
        let marks = self.marks.take().map(|marks| marks.regions);
        self.settle()?;
        Ok(marks.unwrap_or_default())
    }

    pub fn settle(mut self) -> Result<()> {
        self.cond_end();
        match self.fault {
            Some(fault) => Err(fault),
            None => Ok(()),
        }
    }

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
        if self.recording
            && let Some(marks) = self.marks.as_mut()
            && marks.gated.get(self.at as usize).copied().unwrap_or(false)
        {
            match crate::device::conditional::captured_nodes(marks.stream) {
                Ok(nodes) => marks.begin = Some((0, nodes.into_iter().collect())),
                Err(fault) => {
                    if self.fault.is_none() {
                        self.fault = Some(fault);
                    }
                }
            }
        }
        self.place.region.set(self.at);
        self.place.run.set(0);
        // The lane keys a region's shared scratch: a fire walked without lanes runs
        // every region on the main stream, so they take the main lane's scratch.
        self.place.lane.set(if self.lanes.is_some() {
            region.stream
        } else {
            0
        });
        if let Some(pump) = self.pump
            && self.fault.is_none()
            && let Err(fault) = pump.rotor.at(self.at, pump.compute)
        {
            self.fault = Some(fault);
        }
        self.at += 1;
        if self.open.is_some() {
            return;
        }
        if let Some(lanes) = self.lanes {
            lanes.at.set(region.stream);
        } else if let Some(cond) = self.cond {
            // No lanes (a serial walk, or a fire wider than the side streams
            // run for): every region on the main stream, its waits moot.
            cond.at.set(0);
        }
    }
    fn region_end(&mut self, _region: &Region) {
        let region = self.at.saturating_sub(1);
        if let Some(marks) = self.marks.as_mut()
            && let Err(fault) = marks.close(region, None)
            && self.fault.is_none()
        {
            self.fault = Some(fault);
        }
    }

    fn run(&mut self, run: u32, _runs: u32) {
        self.place.run.set(run);
        let region = self.at.saturating_sub(1);
        if run > 0
            && let Some(marks) = self.marks.as_mut()
            && marks.begin.is_some()
            && let Err(fault) = marks.close(region, Some(run))
            && self.fault.is_none()
        {
            self.fault = Some(fault);
        }
    }

    fn records(&self) -> bool {
        self.recording
    }

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
            Lowering::Switch { arms, .. } => Kind::Switch {
                arms: u32::from(arms),
            },
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
                Kind::If => {
                    let (indptr, lanes, absent, win) = self.count_of(cond, region);
                    kernels_cuda::runtime::graph::set_conditional(
                        cond.setter,
                        handle,
                        indptr,
                        lanes,
                        absent,
                        kernels_cuda::runtime::graph::Arm::Set,
                        win,
                    )
                    .map_err(|why| Fault::Unbound {
                        what: format!(
                            "the conditional setter for region {region}, which answered {why}"
                        ),
                    })?;
                }
                Kind::Switch { arms } => {
                    for arm in 0..arms {
                        let (indptr, lanes, absent, win) = self.count_of(cond, region + arm);
                        if absent {
                            return Err(unlowered(lowering));
                        }
                        kernels_cuda::runtime::graph::set_switch(
                            cond.setter,
                            handle,
                            arm,
                            indptr,
                            lanes,
                            kernels_cuda::runtime::graph::Arm::Set,
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
                if kind == Kind::If {
                    self.enter(0);
                }
            }
            Err(fault) => self.fault = Some(fault),
        }
    }

    fn cond_arm(&mut self, arm: u8) {
        if !self.recording || self.open.is_none() {
            return;
        }
        self.enter(u32::from(arm));
    }

    fn cond_end(&mut self) {
        let Some((_, was, body)) = self.open.take() else {
            return;
        };
        let Some(cond) = self.cond else {
            return;
        };
        let closed = if body {
            crate::device::conditional::end_body(cond.body)
        } else {
            Ok(())
        };
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

    use poem_ir::ClassSet;

    fn conditional() -> Region {
        Region {
            nodes: 0..26,
            mask: ClassSet::of([0]),
            phase: poem_compiler::Phase::Capture,
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
    fn window_every_case() {
        a_recording_cursor_with_nowhere_to_put_a_conditional_still_refuses_it();
        the_live_seat_is_every_windows_own_rows_and_offset_at_a_fixed_stride();
        a_regions_slot_is_its_masks_whatever_the_fire_seats();
    }

    fn a_recording_cursor_with_nowhere_to_put_a_conditional_still_refuses_it() {
        let cell = At::new();
        let mut eager = Cursor::new(&cell);
        let region = conditional();
        eager.region_begin(&region);
        eager.cond_begin(&region.lowering);
        eager.cond_end();
        eager.region_end(&region);
        eager.settle().expect("an eager walk ignores the bracket");

        let cell = At::new();
        let mut recording = Cursor::new(&cell).writing();
        recording.region_begin(&region);
        recording.cond_begin(&region.lowering);
        let fault = recording
            .settle()
            .expect_err("a capture may not record a body outside its node");
        assert!(
            matches!(fault, Fault::Unlowered { region: 0, .. }),
            "{fault}"
        );
        assert!(fault.to_string().contains("nowhere"), "{fault}");
    }

    fn plain(row_offset: u32, rows: u32) -> Window {
        Window {
            spans: poem_ir::PerAxis::new([
                MaskSpan {
                    row_offset,
                    rows,
                    lane_offset: 0,
                    lanes: 1,
                },
                MaskSpan::default(),
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

    fn windows(spans: &[Vec<Window>]) -> Windows {
        let mut table = Windows {
            windows: Vec::new(),
            runs: Vec::new(),
            of_region: Vec::new(),
            axes: Vec::new(),
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
            table.axes.push(poem_ir::RowAxis::Tokens);
        }
        table.fill_live();
        table
    }

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
        assert_eq!(&table.live()[4..8], &[0, 0, 0, 0]);
    }

    fn a_regions_slot_is_its_masks_whatever_the_fire_seats() {
        use poem_ir::RowAxis::Tokens;
        let (a, both, b) = (
            poem_ir::ClassSet::of([0usize]),
            poem_ir::ClassSet::of([0usize, 1]),
            poem_ir::ClassSet::of([1usize]),
        );
        // Regions over A, A and B (cut into at most two runs), A again, B.
        let (bases, count) = bases_of(
            [
                (Tokens, &a, 1),
                (Tokens, &both, 2),
                (Tokens, &a, 1),
                (Tokens, &b, 1),
            ]
            .into_iter(),
        );
        assert_eq!(bases, vec![0, 1, 0, 3], "a mask's regions share its slots");
        assert_eq!(count, 4, "one slot a run each mask may cut into");
    }
}
