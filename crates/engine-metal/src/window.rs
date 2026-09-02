//! Fire windows: which rows/lanes each region of the baked template runs over, and the cursor that tracks which window a [`Run`] resolves against.

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

/// One window, and its own rebased qo boundaries.
///
/// `indptr` is per-window (not sliced from another's): a ragged view's
/// offsets are relative to its own start.
#[derive(Debug, Clone)]
pub struct Window {
    /// The rows and lanes this window covers, in fire coordinates.
    pub span: MaskSpan,
    /// `[lanes + 1]`: the window's qo boundaries, rebased to start at 0.
    pub indptr_host: Vec<i32>,
    /// The same vector, staged; [`NIL`] until [`Windows::bind`] mints its view.
    pub indptr: Tensor,
    /// Present iff a [`Fallback::Copy`](model_compiler::Fallback) window:
    /// the runs it compacts. When present, `span` is the compacted
    /// rectangle (offsets 0), not a fire interval.
    pub gathered: Option<Gathered>,
    /// Which expert-major pass of its region's run this window is, of how
    /// many (`0` of `1` for a run walked once).
    pub pass: u32,
    pub passes: u32,
    /// The mask's interval on the second row axis: patch rows if `span`
    /// covers tokens, image rows if it covers lanes. All zero when the
    /// fire carries no image.
    pub patch: MaskSpan,
}

/// A `Fallback::Copy` window: which fire rows the rectangle draws from,
/// the ambient row tables re-laid in that order, and per-space pool tables
/// re-cut for the gathered lanes. Only activations move on device; the
/// rest is recomputed on the host.
#[derive(Debug, Clone)]
pub struct Gathered {
    /// The fire intervals this rectangle compacts, in order.
    pub runs: Vec<MaskSpan>,
    /// `[rows]`: fire row each compacted row came from.
    pub rows_host: Vec<i32>,
    /// The same vector, staged.
    pub rows: Tensor,
    /// `[rows]`: `positions`, re-laid in gathered row order (local row `i`
    /// maps to fire row `rows_host[i]`).
    pub positions_host: Vec<i32>,
    /// The same vector, staged.
    pub positions: Tensor,
    /// `[rows]`: `request_of_token`, re-laid in gathered row order; stays
    /// absolute (not renumbered).
    pub request_of_token_host: Vec<i32>,
    /// The same vector, staged.
    pub request_of_token: Tensor,
    /// One entry per kv geometry space, in space order.
    pub spaces: Vec<GatheredSpace>,
}

/// One kv space's geometry, re-cut for a gathered window's lanes. The
/// page-id list is copied (not sliced): gathered lanes aren't contiguous,
/// so bounds are a fresh prefix sum.
#[derive(Debug, Clone)]
pub struct GatheredSpace {
    /// `[lanes + 1]`: bounds over
    /// [`page_indices_host`](GatheredSpace::page_indices_host), fresh
    /// prefix sum from 0.
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
/// `bucket`/`enabled` come from the deployment; the three vectors are the
/// fire's own host state, borrowed for the call.
#[derive(Debug, Clone, Copy)]
pub struct Copies<'a> {
    /// Which `Budget::buckets` position this fire's rows land in; `0` if
    /// the deployment declared no lattice.
    pub bucket: u32,
    /// Does this shell serve `Fallback::Copy` at all? A masked fire always
    /// takes the split (the mask plane isn't permuted for a gathered window).
    pub enabled: bool,
    /// This fire's host geometry, one per kv space.
    pub spaces: &'a [Geometry],
    /// `[rows]`: this fire's absolute positions, in fire row order.
    pub positions: &'a [i32],
    /// `[rows]`: which lane owns each token row, in fire row order.
    pub request_of_token: &'a [i32],
}

impl Copies<'_> {
    /// No copies: split everything.
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

/// Every region's windows, deduplicated: many regions share few distinct
/// windows, each staged once. A region holds a list of windows (one per
/// P4 fallback interval); an empty region gets one empty-window entry.
#[derive(Debug, Clone, Default)]
pub struct Windows {
    windows: Vec<Window>,
    /// Every region's runs end to end, as positions in
    /// [`windows`](Windows::windows).
    runs: Vec<u32>,
    /// Region index → `(where its runs start, how many)`.
    of_region: Vec<(u32, u32)>,
}

/// Every value the region's nodes name: inputs then outputs, flat (not
/// per-node). `None` if the region names a node the plan lacks.
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

/// Is this region's work something the copy path can serve? Only a few
/// operand shapes qualify (token-row tensors, cache bindings, the four
/// geometry vectors, struct operands); Metal's row gather is bf16/f32
/// only. Anything else takes the split, which is always correct.
pub(crate) fn copyable(trace: &Trace, region: &Region) -> bool {
    let Some((ins, outs)) = operands(&trace.nodes, region) else {
        return false;
    };
    ins.iter().chain(outs.iter()).all(|id| {
        let Some(decl) = trace.values.get(id.0 as usize) else {
            return false;
        };
        match &decl.def {
            // Only the paged pool copies; a recurrent bank is slot-addressed.
            Def::Cache(c) => matches!(
                trace.caches.get(*c as usize),
                Some(model_ir::CacheRow::Kv { .. })
            ),
            // Indices is compacted; the other three geometry vectors are per-lane.
            Def::Input(RuntimeInput::Geometry { kind, .. }) => matches!(
                kind,
                GeomKind::Indptr | GeomKind::Indices | GeomKind::LastPageLen | GeomKind::KvLen
            ),
            // Mask plane isn't permuted for a gathered window; masked fires decline copies.
            Def::Input(RuntimeInput::Mask { .. }) => false,
            _ => match &decl.ty {
                // A plan payload: host state, not a rectangle.
                Ty::Struct(_) => true,
                Ty::Tensor { shape, dtype } => match shape.first() {
                    // Row-shaped: stages if the row-move gather is stamped for this dtype.
                    Some(Dim::Tokens) => matches!(dtype, Dtype::Bf16 | Dtype::F32),
                    // k rows per token row; `rows_host` maps one index per k rows.
                    Some(Dim::TokensTimes(_)) => false,
                    // Window-free: handed over whole, gathered or not.
                    Some(Dim::Const(_)) | None => true,
                    Some(Dim::Lanes | Dim::LanesPlus(_)) => false,
                    // Patch/image rows are a different row space; a token-row map can't cut them.
                    Some(Dim::Patches | Dim::Images | Dim::ImagesPlus(_)) => false,
                },
            },
        }
    })
}

/// The same question asked of a mask — a prepare region has no row of its
/// own, so it must agree with its readers. All regions over one mask must
/// admit the copy, or none does.
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

/// How many distinct windows this artifact can ever gather — a count of
/// masks (all regions over one mask share a window).
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
/// Deduplicated on span and gathered runs (same extent, different rows).
fn seat(windows: &mut Vec<Window>, window: Window) -> u32 {
    // An expert-major pass is its own window even over the same rows: the
    // cut reads which pass it is off the window.
    let same = |held: &Window| {
        held.span == window.span
            && held.pass == window.pass
            && held.passes == window.passes
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

/// Build the gathered window a list of runs compacts to: row map, qo
/// boundaries (rebased over the union), ambient row tables re-laid, and
/// per-space pool tables re-cut lane by lane.
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
            // Rebase done once over the union rather than per run.
            let width = indptr_host
                .get(lane + 1)
                .zip(indptr_host.get(lane))
                .map_or(0, |(end, start)| end - start);
            bounds.push(bounds.last().copied().unwrap_or(0) + width);
            lanes.push(lane);
        }
    }

    // Permutes ambient tables to gathered row order; defaults to 0 for a missing row.
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
        // Filled by the caller: patch interval is the region's, not the union's.
        patch: MaskSpan::default(),
        pass: 0,
        passes: 1,
    }
}

impl Windows {
    /// The windows of one fire: every region resolved against this
    /// composition's class table, one per interval its mask covers.
    ///
    /// # Errors
    ///
    /// [`Fault::Fragmented`] for a region whose classes aren't consecutive
    /// and owes no `Fallback` row. Otherwise served as `Fallback::Split { r }`
    /// or, where copies allow it, as a single [`Gathered`] window.
    pub fn of(
        trace: &Trace,
        compiled: &CompiledModel,
        classes: &WindowTable,
        patches: &WindowTable,
        indptr_host: &[i32],
        copies: Copies<'_>,
        run_caps: &[u32],
        run_passes: &[u32],
    ) -> Result<Windows> {
        let mut windows: Vec<Window> = Vec::new();
        let mut runs: Vec<u32> = Vec::with_capacity(compiled.template().len());
        let mut of_region: Vec<(u32, u32)> = Vec::with_capacity(compiled.template().len());
        let mut spans: Vec<MaskSpan> = Vec::new();

        for (at, region) in compiled.template().iter().enumerate() {
            // Which table a region's rows come from is its capture unit's axis.
            let axis = compiled.axis_of(at);
            match axis {
                model_ir::RowAxis::Tokens => classes.spans_into(&region.mask, &mut spans),
                model_ir::RowAxis::Patches => patches.spans_into(&region.mask, &mut spans),
            }
            // The other axis's interval, computed for every region. A
            // fragmented patch window is refused, not resolved to its first piece.
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
            // The rebased qo boundaries below are the token rectangle's alone;
            // a patch region's `indptr_host` stays empty.
            if spans.len() > 1 {
                // Checks: did P4 promise this window consecutive, and is the run count in bounds.
                let bound = fallback::bound(compiled, axis, &region.mask);
                if fallback::promised(compiled, axis, region) || spans.len() > bound as usize {
                    return Err(Fault::Fragmented {
                        region: at as u32,
                        runs: spans.len(),
                        promised: fallback::promised(compiled, axis, region).then_some(bound),
                    });
                }
                // This shell serves Split and Copy only, never Grouped.
            }
            // An empty mask gets the zero window; the walk skips it.
            if spans.is_empty() {
                spans.push(MaskSpan::default());
            }

            // A copy turns a fragmented window into one window over the compacted rectangle.
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

            // A capped region (`FireDescriptor::run_caps`, the same cap the
            // walk cuts its launches by) is seated in pieces of at most `cap`
            // rows. A piece inside a lane has no qo boundary of its own to
            // rebase to — its ops are row-local — so it states `[0, rows]`.
            let cap = run_caps.get(at).copied().unwrap_or(0);
            let max_passes = run_passes.get(at).copied().unwrap_or(0);
            // Expert-major passes walk every span whole (the same row
            // boundaries as an uncapped run) and cut before each pass.
            let (capped, passes) = if cap > 0 && max_passes > 1 {
                (false, model_exec::fire::pass_spans(&mut spans, cap, max_passes))
            } else {
                let capped = cap > 0 && spans.iter().any(|span| span.rows > cap);
                if capped {
                    model_exec::fire::chunk_spans(&mut spans, cap);
                }
                (capped, 1)
            };
            of_region.push((runs.len() as u32, spans.len() as u32));
            for (i, &span) in spans.iter().enumerate() {
                let window = Window {
                    span,
                    indptr_host: match axis {
                        model_ir::RowAxis::Tokens if capped => vec![0, span.rows as i32],
                        model_ir::RowAxis::Tokens => rebase(indptr_host, span)?,
                        model_ir::RowAxis::Patches => Vec::new(),
                    },
                    indptr: Tensor::new(NIL, 0, 1, Dtype::I32),
                    gathered: None,
                    patch,
                    pass: (i as u32) % passes,
                    passes,
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
    /// one copy. [`bind`](Windows::bind) walks the same blob in the same order.
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

    /// Seat the staged boundaries: `base` is where [`packed`](Windows::packed)
    /// landed inside `buffer`. One handle per distinct window, minted in
    /// `packed`'s order.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] when a window's boundaries would leave `buffer`,
    /// or the handle table is full.
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
    #[must_use]
    pub fn runs(&self, region: u32) -> u32 {
        self.of_region.get(region as usize).map_or(0, |held| held.1)
    }

    /// How many encodes this fire's walk makes over the whole template —
    /// one per region, plus `r - 1` per split, minus what a copy takes back off.
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

    /// One region's window, for one run of it. Panics on a region/run this
    /// table doesn't hold (a shell integrity failure).
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

/// Bake-time check: no attention schedule may be built over more classes
/// than the node consuming it runs in.
///
/// # Errors
///
/// [`Fault::Straddled`], naming the value, node, and the two class sets.
pub fn no_schedule_straddles_its_readers(trace: &Trace, compiled: &CompiledModel) -> Result<()> {
    Ok(check::no_schedule_straddles_its_readers(trace, compiled)?)
}

/// Where the walk is: which region, and which run of that region's
/// window. A `Cell` because `walk` holds the sink and dispatch as two
/// separate borrows.
#[derive(Debug, Default)]
pub struct At {
    /// The region index, in `CompiledModel::template` order.
    pub region: Cell<u32>,
    /// Which run of that region's window: `0..r` for a region P4 couldn't seat.
    pub run: Cell<u32>,
}

impl At {
    /// A cursor position at the top of the template.
    #[must_use]
    pub fn new() -> At {
        At::default()
    }
}

/// This shell's [`Sink`]: the region counter [`Run`](crate::run::Run) reads
/// its window out of. Eager encoding means the DAG's topological order is
/// already a legal schedule, so fork/join are no-ops.
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

    /// What the device refused during the walk, if anything. Always `Ok`
    /// (this cursor makes no device call); the `Result` matches the CUDA
    /// sibling's seam.
    ///
    /// # Errors
    ///
    /// None today, by construction.
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

    /// Which class-set interval every operand after this call resolves against.
    fn run(&mut self, run: u32, _runs: u32) {
        self.place.run.set(run);
    }
    fn cond_begin(&mut self, _lowering: &Lowering) {}
    fn cond_arm(&mut self, _arm: u8) {}
    fn cond_end(&mut self) {}
    /// Nothing to record: an eager encode has already ordered this region
    /// against everything before it.
    fn fork(&mut self, _event: EventId) {}
    /// Nothing to wait on, for the same reason `fork` records nothing.
    fn join(&mut self, _event: EventId) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_exec::fire::ClassWindow;
    
    use model_ir::ClassSet;

    // 10 prefill rows over 2 lanes, then 3 decode rows over 3 lanes.
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

}
