//! Seriates a fire's lanes into per-class row/lane windows (one
//! [`WindowTable`] per row axis), using the class order the compiler already
//! baked, so every node's window is a contiguous `[offset, offset + rows)`
//! range. Runs on the host in front of every launch, so it must stay
//! allocation-light.

use model_compiler::{Budget, Budgets, ClassOrder, CompiledModel, PatchLadder};
use model_ir::{ClassSet, PerAxis, RowAxis};

use crate::fire::Fault;
use crate::{Error, Result};

/// One request inside a fire, as the runtime submits it. `word` decides
/// which windows this lane is in; `rows` decides how much of them it
/// occupies. Everything else rides in buffers the engine already binds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lane {
    /// The lane's fact bits. `Guard::Fact(bit)` indexes them.
    pub word: u64,
    /// How many token rows this lane contributes — 1 for decode, prompt length for prefill.
    pub rows: u32,
    /// How many images this lane submitted. Zero for every text lane.
    pub images: u32,
    /// How many patch rows those images total, concatenated.
    pub patches: u32,
}

impl Lane {
    /// A lane of `rows` token rows whose facts are `word`, carrying no image.
    #[must_use]
    pub fn new(word: u64, rows: u32) -> Lane {
        Lane {
            word,
            rows,
            images: 0,
            patches: 0,
        }
    }

    /// The same lane, carrying `images` images of `patches` patch rows total.
    #[must_use]
    pub fn with_images(word: u64, rows: u32, images: u32, patches: u32) -> Lane {
        Lane {
            word,
            rows,
            images,
            patches,
        }
    }

    /// What this request contributes to one row space: `(rows, lanes)`. A
    /// token lane is always one lane; a patch lane is zero or more images.
    #[must_use]
    pub fn on(self, axis: RowAxis) -> (u32, u32) {
        match axis {
            RowAxis::Tokens => (self.rows, 1),
            RowAxis::Patches => (self.patches, self.images),
        }
    }
}

/// One class's place in the seriated fire. `Dim::Tokens` columns are
/// indexed by row offset, `Dim::Lanes` columns by lane offset. A class
/// with no lanes is the zero window (`rows == 0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClassWindow {
    /// First row of this class's interval.
    pub row_offset: u32,
    /// How many rows it has. Zero for a class no lane is in.
    pub rows: u32,
    /// First lane of this class's interval.
    pub lane_offset: u32,
    /// How many lanes it has.
    pub lanes: u32,
}

/// The one row-and-lane interval a class mask covers in a fire — the same
/// shape as [`ClassWindow`], over a mask's classes rather than one.
pub type MaskSpan = ClassWindow;

/// Cut every span of `spans` longer than `cap` rows into consecutive pieces
/// of at most `cap` rows, in place and in order. A piece keeps its span's
/// lane interval: the ops of a capped region are row-local (a routed
/// mixture's matmuls and their combine), and read no lane-shaped value. `0`
/// caps nothing.
/// **EXPERT-MAJOR PASSES** over a routed segment: instead of cutting a run
/// into row pieces (`chunk_spans`), every span is walked whole `passes`
/// times, and at each pass's cut the tier seats ONE GROUP of the distinct
/// experts the run routes to and masks the routing vector to it (`-1`
/// elsewhere), so each expert is copied once per run rather than once per
/// piece it appears in. The pass count is what `cap` would have cut the
/// widest span into, bounded by `max_passes` (the groups the whole expert
/// set fills). Returns the passes; `1` leaves the spans alone.
pub fn pass_spans(spans: &mut Vec<MaskSpan>, cap: u32, max_passes: u32) -> u32 {
    if cap == 0 || max_passes <= 1 {
        return 1;
    }
    let widest = spans.iter().map(|span| span.rows).max().unwrap_or(0);
    let pieces = widest.div_ceil(cap);
    // A run the cap would not have cut stays one segment. Otherwise the
    // tier seats HALF the slab per pass (the other half is being filled for
    // the next pass while this one runs), so the pass count doubles.
    if pieces <= 1 {
        return 1;
    }
    let passes = (2 * pieces).clamp(1, max_passes);
    if passes <= 1 {
        return 1;
    }
    let whole = std::mem::take(spans);
    for span in whole {
        for _ in 0..passes {
            spans.push(span);
        }
    }
    passes
}

pub fn chunk_spans(spans: &mut Vec<MaskSpan>, cap: u32) {
    if cap == 0 || spans.iter().all(|span| span.rows <= cap) {
        return;
    }
    let whole = std::mem::take(spans);
    for span in whole {
        if span.rows <= cap {
            spans.push(span);
            continue;
        }
        let mut done = 0;
        while done < span.rows {
            let take = (span.rows - done).min(cap);
            spans.push(MaskSpan {
                row_offset: span.row_offset + done,
                rows: take,
                lane_offset: span.lane_offset,
                lanes: span.lanes,
            });
            done += take;
        }
    }
}

/// The window table: one [`ClassWindow`] per class, indexed by class
/// position. [`walk()`](fn@crate::fire::walk) checks the width first, or a
/// wrong-width table finds the wrong class.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WindowTable {
    classes: Vec<ClassWindow>,
    /// The classes with rows, in the order their rows stand.
    order: Vec<u32>,
}

impl WindowTable {
    /// A table of one window per class; the seriated order is read back
    /// off the windows.
    #[must_use]
    pub fn new(classes: Vec<ClassWindow>) -> WindowTable {
        let mut order: Vec<u32> = (0..classes.len() as u32)
            .filter(|&class| classes[class as usize].rows > 0)
            .collect();
        order.sort_unstable_by_key(|&class| classes[class as usize].row_offset);
        WindowTable { classes, order }
    }

    /// The same table with the seriated order the composer already had.
    #[must_use]
    pub fn seriated(classes: Vec<ClassWindow>, order: Vec<u32>) -> WindowTable {
        WindowTable { classes, order }
    }

    /// The classes this table has rows in, in order.
    pub fn present_in_order(&self) -> impl Iterator<Item = u32> + '_ {
        self.order.iter().copied()
    }

    /// How many classes it covers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// Does it cover no classes at all? A real plan always has at least one.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// One class's window, or the zero window for a class this table lacks.
    #[must_use]
    pub fn class(&self, class: usize) -> ClassWindow {
        self.classes.get(class).copied().unwrap_or_default()
    }

    /// The windows, in class order.
    #[must_use]
    pub fn as_slice(&self) -> &[ClassWindow] {
        &self.classes
    }

    /// How many token rows a node with this class mask runs over.
    #[must_use]
    pub fn rows_of(&self, mask: &ClassSet) -> u32 {
        mask.iter().map(|c| self.class(c).rows).sum()
    }

    /// How many lanes a node with this class mask runs over.
    #[must_use]
    pub fn lanes_of(&self, mask: &ClassSet) -> u32 {
        mask.iter().map(|c| self.class(c).lanes).sum()
    }

    /// The one interval this mask covers. `Ok(None)` is the empty window;
    /// `Err(runs)` is a mask that needs more than one launch — use
    /// [`spans`](WindowTable::spans) instead.
    ///
    /// # Errors
    ///
    /// The number of runs the mask covers, when that is more than one.
    pub fn span(&self, mask: &ClassSet) -> core::result::Result<Option<MaskSpan>, usize> {
        let runs = self.spans(mask);
        match runs.len() {
            0 => Ok(None),
            1 => Ok(Some(runs[0])),
            more => Err(more),
        }
    }

    /// Every interval this mask covers, ascending — one per launch. Two
    /// classes merge into one run iff the second's rows begin where the
    /// first's end; a zero-row class is invisible and never breaks a run.
    #[must_use]
    pub fn spans(&self, mask: &ClassSet) -> Vec<MaskSpan> {
        let mut out = Vec::new();
        self.spans_into(mask, &mut out);
        out
    }

    /// [`spans`](WindowTable::spans), into a caller-kept, reused buffer.
    pub fn spans_into(&self, mask: &ClassSet, out: &mut Vec<MaskSpan>) {
        out.clear();
        for class in mask.iter() {
            let window = self.class(class);
            if window.rows == 0 {
                continue;
            }
            out.push(window);
        }
        out.sort_unstable_by_key(|span| span.row_offset);

        // `open` is how many runs are settled; the entry under it grows.
        let mut open = 0usize;
        for read in 0..out.len() {
            let span = out[read];
            let grows = open > 0 && {
                let last = out[open - 1];
                last.row_offset + last.rows == span.row_offset
            };
            if grows {
                out[open - 1].rows += span.rows;
                out[open - 1].lanes += span.lanes;
            } else {
                out[open] = span;
                open += 1;
            }
        }
        out.truncate(open);
    }
}

/// One lane, placed. `source` carries the submission-to-fire permutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LaneRow {
    /// This lane's index in the submitted slice.
    pub source: u32,
    /// Its fact word, carried through so the device never re-derives it.
    pub word: u64,
    /// The class the word resolved to.
    pub class: u32,
    /// Its first token row in the seriated fire.
    pub row_offset: u32,
    /// How many rows it contributes.
    pub rows: u32,
    /// Its first patch row in the second seriation.
    pub patch_offset: u32,
    /// How many patch rows it contributes.
    pub patches: u32,
    /// Its first image in the patch seriation, where its `images + 1` indptr run begins.
    pub image_offset: u32,
    /// How many images it contributes.
    pub images: u32,
}

/// One fire's composition: which windows have rows, where, and in what
/// order. Not a schedule — this is the data the baked script reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Composition {
    lanes: Vec<LaneRow>,
    /// One [`AxisComposition`] per row space. A text-only fire's patch
    /// entry is the zero seriation, not an absent table.
    axes: PerAxis<AxisComposition>,
}

/// One row space's whole composition: its window table, two totals, and
/// the rung they round up to. On the token axis, lanes are requests; on
/// the patch axis, lanes are images.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AxisComposition {
    /// One [`ClassWindow`] per class — this rectangle's seriation.
    pub classes: WindowTable,
    /// This rectangle's total rows.
    pub rows: u32,
    /// This rectangle's total lanes.
    pub lanes: u32,
    /// The rung `rows` rounds up to — which recorded graph this fire's unit launches.
    pub bucket: u32,
}

impl Composition {
    /// The lanes, in fire order: grouped by class, submission order inside each class.
    #[must_use]
    pub fn lanes(&self) -> &[LaneRow] {
        &self.lanes
    }

    /// This fire's composition on one row axis.
    #[must_use]
    pub fn axis(&self, axis: RowAxis) -> &AxisComposition {
        &self.axes[axis]
    }

    /// This fire's window table on one axis.
    #[must_use]
    pub fn table(&self, axis: RowAxis) -> &WindowTable {
        &self.axes[axis].classes
    }

    /// How many lanes this fire carries.
    #[must_use]
    pub fn lane_count(&self) -> u32 {
        self.axes[RowAxis::Tokens].lanes
    }

    /// The window table, indexed by class.
    #[must_use]
    pub fn classes(&self) -> &WindowTable {
        self.table(RowAxis::Tokens)
    }

    /// The classes this fire has lanes in, in order.
    #[must_use]
    pub fn present(&self) -> &[u32] {
        &self.axes[RowAxis::Tokens].classes.order
    }

    /// Total token rows.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.axes[RowAxis::Tokens].rows
    }

    /// The shape bucket these rows round up to. Equal to
    /// [`rows`](Composition::rows) when the budget lists no lattice.
    #[must_use]
    pub fn bucket(&self) -> u32 {
        self.axes[RowAxis::Tokens].bucket
    }

    /// How many patch rows this fire carries. Zero for a text-only fire.
    #[must_use]
    pub fn patch_rows(&self) -> u32 {
        self.axes[RowAxis::Patches].rows
    }

    /// How many images this fire carries — the patch axis's lane count.
    #[must_use]
    pub fn images(&self) -> u32 {
        self.axes[RowAxis::Patches].lanes
    }

    /// The patch window table, indexed by class. The class order is the
    /// patch axis's own — a class third in the token rectangle may stand first here.
    #[must_use]
    pub fn patch_classes(&self) -> &WindowTable {
        self.table(RowAxis::Patches)
    }

    /// The patch rung these patch rows round up to. Equal to
    /// [`patch_rows`](Composition::patch_rows) when the ladder lists no rungs.
    #[must_use]
    pub fn patch_bucket(&self) -> u32 {
        self.axes[RowAxis::Patches].bucket
    }

}

/// Compose one fire on the token axis — the door every text-only deployment
/// uses. [`compose_axes`] is this same seriation with a second row axis.
///
/// # Errors
///
/// [`Fault::UnknownWord`], [`Fault::EmptyLane`], [`Fault::TooManyLanes`] /
/// [`Fault::TooManyRows`] past the arena's ceilings, [`Fault::NoBucket`]
/// above the bucket lattice, or [`Fault::Towerless`] for images (this door admits none).
pub fn compose(compiled: &CompiledModel, budget: &Budget, lanes: &[Lane]) -> Result<Composition> {
    seriate(compiled, budget, None, lanes)
}

/// Compose one fire over every row axis the deployment admits. Two
/// seriations, not one widened: the patch pass runs beside the token pass
/// over the artifact's own patch `ClassOrder`, with images where lanes were.
///
/// # Errors
///
/// Everything [`compose`] refuses, plus [`Fault::PatchGeometry`] for a
/// lane whose image count and patch payload disagree, and
/// [`Fault::TooManyPatches`] / [`Fault::TooManyImages`] / [`Fault::NoPatchBucket`].
pub fn compose_axes(
    compiled: &CompiledModel,
    budgets: &Budgets,
    lanes: &[Lane],
) -> Result<Composition> {
    seriate(
        compiled,
        &budgets.tokens,
        budgets.patches.as_ref(),
        lanes,
    )
}

fn seriate(
    compiled: &CompiledModel,
    budget: &Budget,
    ladder: Option<&PatchLadder>,
    lanes: &[Lane],
) -> Result<Composition> {
    if lanes.len() > budget.max_lanes as usize {
        return Err(Fault::TooManyLanes {
            lanes: lanes.len(),
            max: budget.max_lanes,
        }
        .into());
    }

    // Pass one: every lane's class. Linear-scanned since a fire's
    // distinct words are few — cheaper than hashing a `u64`.
    let count = compiled.classes.classes.len();
    let mut memo: Vec<(u64, u32)> = Vec::new();
    let mut of_lane: Vec<u32> = Vec::with_capacity(lanes.len());
    // `(rows, lanes)` per class, per axis; all-zero on the patch entry for
    // a text-only fire.
    let mut tally: PerAxis<Vec<(u64, u64)>> = PerAxis::from_fn(|_| vec![(0, 0); count]);
    // Each rectangle's `(rows, lanes)` total, `u64` until the ceiling checks.
    let mut totals: PerAxis<(u64, u64)> = PerAxis::from_fn(|_| (0, 0));
    // Whether this artifact has anywhere for a patch row to go — read off
    // the bake, not the budget.
    let towered = compiled.order_for(RowAxis::Patches).is_some();

    for (i, lane) in lanes.iter().enumerate() {
        let i = i as u32;
        if lane.rows == 0 {
            return Err(Fault::EmptyLane { lane: i }.into());
        }
        // An image is at least one patch row and vice versa; stating one
        // without the other is inconsistent.
        if (lane.images == 0) != (lane.patches == 0) {
            return Err(Fault::PatchGeometry {
                lane: i,
                images: lane.images,
                patches: lane.patches,
            }
            .into());
        }
        // Images against a text with no patch axis are refused, not dropped.
        if lane.images > 0 && !towered {
            return Err(Fault::Towerless { lane: i }.into());
        }
        if lane.images > 0 && ladder.is_none() {
            return Err(Fault::NoPatchLadder { lane: i }.into());
        }
        // Masked to the bits the sweep ran over, since a model may state a
        // fact it does not split on.
        let word = lane.word & compiled.classes.mask;
        let class = match memo.iter().find(|(seen, _)| *seen == word) {
            Some((_, class)) => *class,
            None => {
                let class = compiled.classes.class_of(word).ok_or(Fault::UnknownWord {
                    lane: i,
                    word: lane.word,
                })? as u32;
                memo.push((word, class));
                class
            }
        };
        // `class_of`'s index is in range by construction; the tally is
        // sized from the same list.
        for axis in RowAxis::ALL {
            let (rows, images) = lane.on(axis);
            tally[axis][class as usize].0 += u64::from(rows);
            tally[axis][class as usize].1 += u64::from(images);
            totals[axis].0 += u64::from(rows);
            totals[axis].1 += u64::from(images);
        }
        of_lane.push(class);
    }

    let rows = totals[RowAxis::Tokens].0;
    if rows > u64::from(budget.max_tokens) {
        return Err(Fault::TooManyRows {
            rows,
            max: budget.max_tokens,
        }
        .into());
    }
    let rows = rows as u32;

    // The patch ceilings, checked even with no ladder (`patches` is then zero).
    let (patches, images) = totals[RowAxis::Patches];
    let (max_patches, max_images) = ladder.map_or((0, 0), |l| (l.max_patches, l.max_images));
    if patches > u64::from(max_patches) {
        return Err(Fault::TooManyPatches {
            patches,
            max: max_patches,
        }
        .into());
    }
    if images > u64::from(max_images) {
        return Err(Fault::TooManyImages {
            images,
            max: max_images,
        }
        .into());
    }
    let patches = patches as u32;

    // Taken before the seriations: the token refusal is owed before the patch one.
    let buckets = PerAxis::new([bucket_of(budget, rows)?, patch_bucket_of(ladder, patches)?]);

    // Where each submitted lane's rows land on each axis, one side table
    // (not two lists) to avoid reading a `source` against the wrong record.
    let mut placed: Vec<PerAxis<(u32, u32)>> = vec![PerAxis::from_fn(|_| (0, 0)); lanes.len()];

    // Both rectangles' totals, narrowed — each just admitted by its
    // ceiling, so the cast is a conversion, not a truncation.
    let checked: PerAxis<(u32, u32)> =
        PerAxis::from_fn(|axis| (totals[axis].0 as u32, totals[axis].1 as u32));

    // The prefix sums and lane placement, once per row axis, over that
    // axis's own baked order and tallies.
    let axes = PerAxis::from_fn(|axis| {
        seriate_axis(
            compiled.order_for(axis),
            axis,
            &tally[axis],
            lanes,
            &of_lane,
            checked[axis],
            buckets[axis],
            &mut placed,
        )
    });

    // Pass three: the lane records, class by class in token fire order,
    // submission order inside a class. Every offset is read out of
    // `placed` rather than re-accumulated.
    let mut seriated: Vec<LaneRow> = Vec::with_capacity(lanes.len());
    for class in axes[RowAxis::Tokens].classes.present_in_order() {
        for (i, lane) in lanes.iter().enumerate() {
            if of_lane[i] != class {
                continue;
            }
            let token = placed[i][RowAxis::Tokens];
            let patch = placed[i][RowAxis::Patches];
            seriated.push(LaneRow {
                source: i as u32,
                word: lane.word,
                class,
                row_offset: token.0,
                rows: lane.rows,
                patch_offset: patch.0,
                patches: lane.patches,
                image_offset: patch.1,
                images: lane.images,
            });
        }
    }

    Ok(Composition {
        lanes: seriated,
        axes,
    })
}

/// One row axis's seriation — the prefix sums that place its classes and
/// the walk that places its lanes inside them, over the axis's baked
/// order. A lane contributing zero requests is not placed. An absent plan
/// is the zero seriation.
#[allow(clippy::too_many_arguments)]
fn seriate_axis(
    plan: Option<&ClassOrder>,
    axis: RowAxis,
    tally: &[(u64, u64)],
    lanes: &[Lane],
    of_lane: &[u32],
    totals: (u32, u32),
    bucket: u32,
    placed: &mut [PerAxis<(u32, u32)>],
) -> AxisComposition {
    let count = tally.len();
    let mut classes = vec![ClassWindow::default(); count];
    let (rows, lane_total) = totals;
    let Some(plan) = plan else {
        return AxisComposition {
            classes: WindowTable::seriated(classes, Vec::new()),
            rows,
            lanes: lane_total,
            bucket,
        };
    };

    // The classes this fire has rows in, in order: the baked order,
    // filtered. The widening to `u32` is never a truncation (a class is a `u8`).
    let present = ClassSet::of((0..count).filter(|&class| tally[class].1 > 0));
    let order: Vec<u32> = plan
        .class_order(&present)
        .into_iter()
        .map(u32::from)
        .collect();

    let (mut row_at, mut lane_at) = (0u32, 0u32);
    for &class in &order {
        // Narrowed here, not at the tally: part of a total already admitted.
        let (class_rows, class_lanes) = tally[class as usize];
        let (class_rows, class_lanes) = (class_rows as u32, class_lanes as u32);
        classes[class as usize] = ClassWindow {
            row_offset: row_at,
            rows: class_rows,
            lane_offset: lane_at,
            lanes: class_lanes,
        };
        // Inside a class, submission order.
        let (mut lane_row_at, mut lane_lane_at) = (row_at, lane_at);
        for (i, lane) in lanes.iter().enumerate() {
            if of_lane[i] != class {
                continue;
            }
            let (lane_rows, lane_lanes) = lane.on(axis);
            // A lane contributing no request here is not placed.
            if lane_lanes == 0 {
                continue;
            }
            placed[i][axis] = (lane_row_at, lane_lane_at);
            lane_row_at += lane_rows;
            lane_lane_at += lane_lanes;
        }
        row_at += class_rows;
        lane_at += class_lanes;
    }

    AxisComposition {
        classes: WindowTable::seriated(classes, order),
        rows,
        lanes: lane_total,
        bucket,
    }
}

/// The smallest patch rung that holds these patch rows — [`bucket_of`]'s
/// question on the second row axis. `0` for no ladder or no patch rows;
/// past the top rung is [`Fault::NoPatchBucket`]; otherwise [`rung_of`].
fn patch_bucket_of(ladder: Option<&PatchLadder>, patches: u32) -> Result<u32> {
    let Some(ladder) = ladder else {
        return Ok(0);
    };
    if patches == 0 {
        return Ok(0);
    }
    match ladder.buckets.last().copied() {
        Some(top) if patches > top => Err(Error::Fire(Fault::NoPatchBucket { patches, top })),
        _ => Ok(rung_of(&ladder.buckets, patches)),
    }
}

/// The smallest bucket that holds these rows. An empty lattice is not an
/// error: the bucket for `rows` rows is then `rows` itself. Past the top
/// rung is [`Fault::NoBucket`].
fn bucket_of(budget: &Budget, rows: u32) -> Result<u32> {
    match budget.buckets.last().copied() {
        Some(top) if rows > top => Err(Error::Fire(Fault::NoBucket { rows, top })),
        _ => Ok(rung_of(&budget.buckets, rows)),
    }
}

/// The smallest rung of this lattice that holds `rows`, for a count
/// already admitted by the ceiling above — [`bucket_of`]'s arithmetic
/// without its refusal.
#[must_use]
pub fn rung_of(buckets: &[u32], rows: u32) -> u32 {
    buckets
        .iter()
        .copied()
        .find(|rung| *rung >= rows)
        .unwrap_or(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::fixture::{Build, fact};
    use crate::{Error, fire::Fault};
    use model_compiler::{DeviceProfile, compile};
    use model_ir::{Guard, ValueId};

    /// A deployment small enough to state in an assert.
    fn budget() -> Budget {
        Budget::new(8, 64)
    }

    /// A shared producer, an attention pair split on `qo_one`, a merge, a
    /// shared consumer. Returns the plan's output value.
    fn diagram() -> (Build, ValueId) {
        let mut b = Build::new();
        let x = b.input(8);
        let plan = b.prepare(Guard::Always); // node 0 — prepare, every class
        let q = b.op(x, 4, Guard::Always); // node 1 — every class
        let d = b.decode(q, plan, fact(0)); // node 2 — the decode window
        let p = b.op(q, 4, Guard::not(fact(0))); // node 3 — the prefill window
        let o = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))], 4);
        let y = b.op(o, 4, Guard::Always); // node 4 — every class again
        b.out(y);
        (b, y)
    }

    /// The nodes of [`diagram`], named.
    const SHARED: u32 = 1;
    const DECODE: u32 = 2;
    const PREFILL: u32 = 3;

    /// The rows one node runs over — its region's mask, against this fire.
    fn rows_of(compiled: &CompiledModel, fire: &Composition, node: u32) -> u32 {
        let region = compiled
            .template()
            .iter()
            .find(|r| r.nodes.contains(&node))
            .expect("the regions tile the node list");
        fire.classes().rows_of(&region.mask)
    }

    #[test]
    fn the_thirteen_row_diagram_windows_the_way_the_design_draws_it() {
        // fire (R=5): lane0 prefill(7) lane1 prefill(3) lane2..4 decode(1 each)
        let (b, _) = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let lanes = [
            Lane::new(0, 7),
            Lane::new(0, 3),
            Lane::new(1, 1),
            Lane::new(1, 1),
            Lane::new(1, 1),
        ];
        let fire = compose(&compiled, &budget(), &lanes).expect("composes");

        assert_eq!(fire.rows(), 13);
        assert_eq!(fire.lane_count(), 5);
        assert_eq!(fire.present().len(), 2);

        let prefill = compiled.classes.class_of(0).expect("word 0 is a class");
        let decode = compiled.classes.class_of(1).expect("word 1 is a class");
        let p = fire.classes().class(prefill);
        let d = fire.classes().class(decode);
        assert_eq!((p.rows, p.lanes), (10, 2), "two prefill lanes, ten rows");
        assert_eq!((d.rows, d.lanes), (3, 3), "three decode lanes, three rows");

        // The two windows tile the fire, in whichever order the compiler seated them.
        let mut spans = [(p.row_offset, p.rows), (d.row_offset, d.rows)];
        spans.sort_unstable();
        assert!(
            spans == [(0, 10), (10, 3)] || spans == [(0, 3), (3, 10)],
            "the windows do not tile [0, 13): {spans:?}",
        );

        // The diagram's actual claim: one kernel over all 13 rows for
        // shared ops, one per window for the split pair.
        assert_eq!(rows_of(&compiled, &fire, SHARED), 13);
        assert_eq!(rows_of(&compiled, &fire, DECODE), 3);
        assert_eq!(rows_of(&compiled, &fire, PREFILL), 10);
    }

    #[test]
    fn a_fire_rounds_up_to_a_bucket_and_one_above_them_all_is_refused() {
        let (b, _) = diagram();
        let mut budget = budget();
        budget.buckets = vec![1, 4, 16];
        let compiled = compile(&b.trace, &budget, &DeviceProfile::default()).expect("bakes");

        let fire = compose(&compiled, &budget, &[Lane::new(0, 5)]).expect("composes");
        assert_eq!((fire.rows(), fire.bucket()), (5, 16));
        let fire = compose(&compiled, &budget, &[Lane::new(1, 1)]).expect("composes");
        assert_eq!((fire.rows(), fire.bucket()), (1, 1));

        // 17 rows round up to nothing, so there is no graph to launch them in.
        assert_eq!(
            compose(&compiled, &budget, &[Lane::new(0, 17)]),
            Err(Error::Fire(Fault::NoBucket { rows: 17, top: 16 })),
        );

        // No lattice: the bucket is the fire's own size. `budget` is
        // shadowed above, so the plain one is named through the module.
        let open = super::tests::budget();
        let compiled = compile(&b.trace, &open, &DeviceProfile::default()).expect("bakes");
        let fire = compose(&compiled, &open, &[Lane::new(0, 5)]).expect("composes");
        assert_eq!(fire.bucket(), 5);
    }

}
