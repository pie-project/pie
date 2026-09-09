use model_compiler::{Budget, Budgets, ClassOrder, CompiledModel, Ladder};
use model_ir::{ClassSet, PerAxis, RowAxis};

use crate::fire::Fault;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lane {
    pub word: u64,
    pub rows: u32,
    pub images: u32,
    pub patches: u32,
    pub clips: u32,
    pub voxels: u32,
}

impl Lane {
    #[must_use]
    pub fn new(word: u64, rows: u32) -> Lane {
        Lane {
            word,
            rows,
            images: 0,
            patches: 0,
            clips: 0,
            voxels: 0,
        }
    }

    #[must_use]
    pub fn with_clips(word: u64, rows: u32, clips: u32, voxels: u32) -> Lane {
        Lane {
            word,
            rows,
            images: 0,
            patches: 0,
            clips,
            voxels,
        }
    }

    #[must_use]
    pub fn with_images(word: u64, rows: u32, images: u32, patches: u32) -> Lane {
        Lane {
            word,
            rows,
            images,
            patches,
            clips: 0,
            voxels: 0,
        }
    }

    #[must_use]
    pub fn on(self, axis: RowAxis) -> (u32, u32) {
        match axis {
            RowAxis::Tokens => (self.rows, 1),
            RowAxis::Patches => (self.patches, self.images),
            RowAxis::Voxels => (self.voxels, self.clips),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClassWindow {
    pub row_offset: u32,
    pub rows: u32,
    pub lane_offset: u32,
    pub lanes: u32,
}

pub type MaskSpan = ClassWindow;

pub fn pass_spans(spans: &mut Vec<MaskSpan>, cap: u32, max_passes: u32) -> u32 {
    if cap == 0 || max_passes <= 1 {
        return 1;
    }
    let widest = spans.iter().map(|span| span.rows).max().unwrap_or(0);
    let pieces = widest.div_ceil(cap);
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WindowTable {
    classes: Vec<ClassWindow>,
    order: Vec<u32>,
}

impl WindowTable {
    #[must_use]
    pub fn new(classes: Vec<ClassWindow>) -> WindowTable {
        let mut order: Vec<u32> = (0..classes.len() as u32)
            .filter(|&class| classes[class as usize].rows > 0)
            .collect();
        order.sort_unstable_by_key(|&class| classes[class as usize].row_offset);
        WindowTable { classes, order }
    }

    #[must_use]
    pub fn seriated(classes: Vec<ClassWindow>, order: Vec<u32>) -> WindowTable {
        WindowTable { classes, order }
    }

    pub fn present_in_order(&self) -> impl Iterator<Item = u32> + '_ {
        self.order.iter().copied()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    #[must_use]
    pub fn class(&self, class: usize) -> ClassWindow {
        self.classes.get(class).copied().unwrap_or_default()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[ClassWindow] {
        &self.classes
    }

    #[must_use]
    pub fn rows_of(&self, mask: &ClassSet) -> u32 {
        mask.iter().map(|c| self.class(c).rows).sum()
    }

    #[must_use]
    pub fn lanes_of(&self, mask: &ClassSet) -> u32 {
        mask.iter().map(|c| self.class(c).lanes).sum()
    }

    pub fn span(&self, mask: &ClassSet) -> core::result::Result<Option<MaskSpan>, usize> {
        let runs = self.spans(mask);
        match runs.len() {
            0 => Ok(None),
            1 => Ok(Some(runs[0])),
            more => Err(more),
        }
    }

    #[must_use]
    pub fn spans(&self, mask: &ClassSet) -> Vec<MaskSpan> {
        let mut out = Vec::new();
        self.spans_into(mask, &mut out);
        out
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LaneRow {
    pub source: u32,
    pub word: u64,
    pub class: u32,
    pub row_offset: u32,
    pub rows: u32,
    pub patch_offset: u32,
    pub patches: u32,
    pub image_offset: u32,
    pub images: u32,
    pub voxel_offset: u32,
    pub voxels: u32,
    pub clip_offset: u32,
    pub clips: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Composition {
    lanes: Vec<LaneRow>,
    axes: PerAxis<AxisComposition>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AxisComposition {
    pub classes: WindowTable,
    pub rows: u32,
    pub lanes: u32,
    pub bucket: u32,
}

impl Composition {
    #[must_use]
    pub fn lanes(&self) -> &[LaneRow] {
        &self.lanes
    }

    #[must_use]
    pub fn axis(&self, axis: RowAxis) -> &AxisComposition {
        &self.axes[axis]
    }

    #[must_use]
    pub fn table(&self, axis: RowAxis) -> &WindowTable {
        &self.axes[axis].classes
    }

    #[must_use]
    pub fn lane_count(&self) -> u32 {
        self.axes[RowAxis::Tokens].lanes
    }

    #[must_use]
    pub fn classes(&self) -> &WindowTable {
        self.table(RowAxis::Tokens)
    }

    #[must_use]
    pub fn present(&self) -> &[u32] {
        &self.axes[RowAxis::Tokens].classes.order
    }

    #[must_use]
    pub fn rows(&self) -> u32 {
        self.axes[RowAxis::Tokens].rows
    }

    #[must_use]
    pub fn bucket(&self) -> u32 {
        self.axes[RowAxis::Tokens].bucket
    }

    #[must_use]
    pub fn patch_rows(&self) -> u32 {
        self.axes[RowAxis::Patches].rows
    }

    #[must_use]
    pub fn images(&self) -> u32 {
        self.axes[RowAxis::Patches].lanes
    }

    #[must_use]
    pub fn patch_classes(&self) -> &WindowTable {
        self.table(RowAxis::Patches)
    }

    #[must_use]
    pub fn patch_bucket(&self) -> u32 {
        self.axes[RowAxis::Patches].bucket
    }

    #[must_use]
    pub fn voxel_rows(&self) -> u32 {
        self.axes[RowAxis::Voxels].rows
    }

    #[must_use]
    pub fn clips(&self) -> u32 {
        self.axes[RowAxis::Voxels].lanes
    }

    #[must_use]
    pub fn voxel_classes(&self) -> &WindowTable {
        self.table(RowAxis::Voxels)
    }

    #[must_use]
    pub fn voxel_bucket(&self) -> u32 {
        self.axes[RowAxis::Voxels].bucket
    }
}

pub fn compose(compiled: &CompiledModel, budget: &Budget, lanes: &[Lane]) -> Result<Composition> {
    seriate(compiled, budget, None, None, lanes)
}

pub fn compose_axes(
    compiled: &CompiledModel,
    budgets: &Budgets,
    lanes: &[Lane],
) -> Result<Composition> {
    seriate(
        compiled,
        &budgets.tokens,
        budgets.ladder(RowAxis::Patches),
        budgets.ladder(RowAxis::Voxels),
        lanes,
    )
}

fn seriate(
    compiled: &CompiledModel,
    budget: &Budget,
    ladder: Option<Ladder<'_>>,
    voxel_ladder: Option<Ladder<'_>>,
    lanes: &[Lane],
) -> Result<Composition> {
    if lanes.len() > budget.max_lanes as usize {
        return Err(Fault::TooManyLanes {
            lanes: lanes.len(),
            max: budget.max_lanes,
        }
        .into());
    }

    let count = compiled.classes.classes.len();
    let mut memo: Vec<(u64, u32)> = Vec::new();
    let mut of_lane: Vec<u32> = Vec::with_capacity(lanes.len());
    let mut tally: PerAxis<Vec<(u64, u64)>> = PerAxis::from_fn(|_| vec![(0, 0); count]);
    let mut totals: PerAxis<(u64, u64)> = PerAxis::from_fn(|_| (0, 0));
    let towered = compiled.order_for(RowAxis::Patches).is_some();
    let voxeled = compiled.order_for(RowAxis::Voxels).is_some();

    for (i, lane) in lanes.iter().enumerate() {
        let i = i as u32;
        if lane.rows == 0 {
            return Err(Fault::EmptyLane { lane: i }.into());
        }
        if (lane.images == 0) != (lane.patches == 0) {
            return Err(Fault::PatchGeometry {
                lane: i,
                images: lane.images,
                patches: lane.patches,
            }
            .into());
        }
        if lane.images > 0 && !towered {
            return Err(Fault::Towerless { lane: i }.into());
        }
        if lane.images > 0 && ladder.is_none() {
            return Err(Fault::NoPatchLadder { lane: i }.into());
        }
        if (lane.clips == 0) != (lane.voxels == 0) {
            return Err(Fault::ClipGeometry {
                lane: i,
                clips: lane.clips,
                voxels: lane.voxels,
            }
            .into());
        }
        if lane.clips > 0 && !voxeled {
            return Err(Fault::Vaeless { lane: i }.into());
        }
        if lane.clips > 0 && voxel_ladder.is_none() {
            return Err(Fault::NoVoxelLadder { lane: i }.into());
        }
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

    let (patches, images) = totals[RowAxis::Patches];
    let (max_patches, max_images) = ladder.map_or((0, 0), |l| (l.max_rows, l.max_lanes));
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

    let (voxels, clips) = totals[RowAxis::Voxels];
    let (max_voxels, max_clips) = voxel_ladder.map_or((0, 0), |l| (l.max_rows, l.max_lanes));
    if voxels > u64::from(max_voxels) {
        return Err(Fault::TooManyVoxels {
            voxels,
            max: max_voxels,
        }
        .into());
    }
    if clips > u64::from(max_clips) {
        return Err(Fault::TooManyClips {
            clips,
            max: max_clips,
        }
        .into());
    }
    let voxels = voxels as u32;

    let buckets = PerAxis::new([
        bucket_of(budget, rows)?,
        patch_bucket_of(ladder, patches)?,
        voxel_bucket_of(voxel_ladder, voxels)?,
    ]);

    let mut placed: Vec<PerAxis<(u32, u32)>> = vec![PerAxis::from_fn(|_| (0, 0)); lanes.len()];

    let checked: PerAxis<(u32, u32)> =
        PerAxis::from_fn(|axis| (totals[axis].0 as u32, totals[axis].1 as u32));

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

    let mut seriated: Vec<LaneRow> = Vec::with_capacity(lanes.len());
    for class in axes[RowAxis::Tokens].classes.present_in_order() {
        for (i, lane) in lanes.iter().enumerate() {
            if of_lane[i] != class {
                continue;
            }
            let token = placed[i][RowAxis::Tokens];
            let patch = placed[i][RowAxis::Patches];
            let voxel = placed[i][RowAxis::Voxels];
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
                voxel_offset: voxel.0,
                voxels: lane.voxels,
                clip_offset: voxel.1,
                clips: lane.clips,
            });
        }
    }

    Ok(Composition {
        lanes: seriated,
        axes,
    })
}

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

    let present = ClassSet::of((0..count).filter(|&class| tally[class].1 > 0));
    let order: Vec<u32> = plan
        .class_order(&present)
        .into_iter()
        .map(u32::from)
        .collect();

    let (mut row_at, mut lane_at) = (0u32, 0u32);
    for &class in &order {
        let (class_rows, class_lanes) = tally[class as usize];
        let (class_rows, class_lanes) = (class_rows as u32, class_lanes as u32);
        classes[class as usize] = ClassWindow {
            row_offset: row_at,
            rows: class_rows,
            lane_offset: lane_at,
            lanes: class_lanes,
        };
        let (mut lane_row_at, mut lane_lane_at) = (row_at, lane_at);
        for (i, lane) in lanes.iter().enumerate() {
            if of_lane[i] != class {
                continue;
            }
            let (lane_rows, lane_lanes) = lane.on(axis);
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

fn patch_bucket_of(ladder: Option<Ladder<'_>>, patches: u32) -> Result<u32> {
    let Some(ladder) = ladder else {
        return Ok(0);
    };
    if patches == 0 {
        return Ok(0);
    }
    match ladder.buckets.last().copied() {
        Some(top) if patches > top => Err(Error::Fire(Fault::NoPatchBucket { patches, top })),
        _ => Ok(rung_of(ladder.buckets, patches)),
    }
}

fn voxel_bucket_of(ladder: Option<Ladder<'_>>, voxels: u32) -> Result<u32> {
    let Some(ladder) = ladder else {
        return Ok(0);
    };
    if voxels == 0 {
        return Ok(0);
    }
    match ladder.buckets.last().copied() {
        Some(top) if voxels > top => Err(Error::Fire(Fault::NoVoxelBucket { voxels, top })),
        _ => Ok(rung_of(ladder.buckets, voxels)),
    }
}

fn bucket_of(budget: &Budget, rows: u32) -> Result<u32> {
    match budget.buckets.last().copied() {
        Some(top) if rows > top => Err(Error::Fire(Fault::NoBucket { rows, top })),
        _ => Ok(rung_of(&budget.buckets, rows)),
    }
}

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

    fn budget() -> Budget {
        Budget::new(8, 64)
    }

    fn diagram() -> (Build, ValueId) {
        let mut b = Build::new();
        let x = b.input(8);
        let plan = b.prepare(Guard::Always);
        let q = b.op(x, 4, Guard::Always);
        let d = b.decode(q, plan, fact(0));
        let p = b.op(q, 4, Guard::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))], 4);
        let y = b.op(o, 4, Guard::Always);
        b.out(y);
        (b, y)
    }

    const SHARED: u32 = 1;
    const DECODE: u32 = 2;
    const PREFILL: u32 = 3;

    fn rows_of(compiled: &CompiledModel, fire: &Composition, node: u32) -> u32 {
        let region = compiled
            .template()
            .iter()
            .find(|r| r.nodes.contains(&node))
            .expect("the regions tile the node list");
        fire.classes().rows_of(&region.mask)
    }

    #[test]
    fn compose_every_case() {
        the_thirteen_row_diagram_windows_the_way_the_design_draws_it();
        a_fire_rounds_up_to_a_bucket_and_one_above_them_all_is_refused();
    }

    fn the_thirteen_row_diagram_windows_the_way_the_design_draws_it() {
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

        let mut spans = [(p.row_offset, p.rows), (d.row_offset, d.rows)];
        spans.sort_unstable();
        assert!(
            spans == [(0, 10), (10, 3)] || spans == [(0, 3), (3, 10)],
            "the windows do not tile [0, 13): {spans:?}",
        );

        assert_eq!(rows_of(&compiled, &fire, SHARED), 13);
        assert_eq!(rows_of(&compiled, &fire, DECODE), 3);
        assert_eq!(rows_of(&compiled, &fire, PREFILL), 10);
    }

    fn a_fire_rounds_up_to_a_bucket_and_one_above_them_all_is_refused() {
        let (b, _) = diagram();
        let mut budget = budget();
        budget.buckets = vec![1, 4, 16];
        let compiled = compile(&b.trace, &budget, &DeviceProfile::default()).expect("bakes");

        let fire = compose(&compiled, &budget, &[Lane::new(0, 5)]).expect("composes");
        assert_eq!((fire.rows(), fire.bucket()), (5, 16));
        let fire = compose(&compiled, &budget, &[Lane::new(1, 1)]).expect("composes");
        assert_eq!((fire.rows(), fire.bucket()), (1, 1));

        assert_eq!(
            compose(&compiled, &budget, &[Lane::new(0, 17)]),
            Err(Error::Fire(Fault::NoBucket { rows: 17, top: 16 })),
        );

        let open = super::tests::budget();
        let compiled = compile(&b.trace, &open, &DeviceProfile::default()).expect("bakes");
        let fire = compose(&compiled, &open, &[Lane::new(0, 5)]).expect("composes");
        assert_eq!(fire.bucket(), 5);
    }
}
