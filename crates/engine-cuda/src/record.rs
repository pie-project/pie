//! Record mode: captures each distinct composition once as a `cudaGraph`
//! ("body") and replays it later instead of walking launch by launch. A
//! body is keyed by [`BodyKey`] — lattice bucket plus present-class set, not
//! exact row/lane counts — because the live row count rides a staged seat
//! the kernels read, so one exec serves every fire rounding to that bucket.
//! Regions a graph cannot hold (gathered/grouped spans, or windowed ones
//! whose ops don't all read the seat's start) are cut out as islands and
//! re-issued eagerly between the captured stretches ([`cuts`], [`Step`]).

use std::collections::{HashMap, HashSet};

use model_compiler::CompiledModel;
use model_exec::fire::{
    FireDescriptor, MaskSpan, Phases, Regions, Units, WindowTable, walk_phases, walk_regions,
};
use model_ir::Trace;

use crate::device::graph::{Graph, GraphExec};
use crate::error::Result;
use crate::run::{Ceilings, Run};
use crate::window::{Admit, At, Cursor, Lanes};

/// Which fire of a key captures it — earlier fires, and this one's own
/// first pass, walk eagerly. Two, because the dense autotuner only tunes a
/// shape on its second sighting, and a capturing stream never benchmarks.
pub const WARM_FIRES: u32 = 2;

/// Everything one fire tells the record mode about itself.
pub struct Fire<'a> {
    /// Walk this body's cuts eagerly instead of launching its execs
    /// (the golden pass's control arm). `false` for ordinary callers.
    pub eager_twin: bool,
    /// The plan the template's node ranges index.
    pub trace: &'a Trace,
    /// The artifact being walked.
    pub compiled: &'a CompiledModel,
    /// This fire's class windows, which the walk reads its counts from.
    pub descriptor: &'a FireDescriptor,
    /// The stream the shell enqueues on.
    pub stream: *mut core::ffi::c_void,
    /// P6's side streams and event handles, given only to the capturing
    /// walk (free inside a capture; a real sync outside one).
    pub lanes: Option<Lanes<'a>>,
    /// Where a conditional node goes, given only to the capturing walk.
    /// `None` for every SKU but the drafting ones.
    pub conditionals: Option<crate::window::Conditionals<'a>>,
    /// Which classes a decode lane lands in ([`Ladder::rung`]). Empty is
    /// a plan with no decode arm.
    pub decoding: &'a model_ir::ClassSet,
    /// Does this artifact state a patch axis? An artifact fact, not a
    /// fire's.
    pub towered: bool,
    /// The most lanes this load can ever seat — must match what
    /// `Shell::prepare` used to build the key.
    pub lane_ceiling: u32,
    /// The pad pair per axis, the admission table, the shifted slice and
    /// the key's carve, matching what the `Run` beside this fire was handed.
    pub ceilings: Ceilings<'a>,
}
/// One load's graph cache: the body map, and the recorder around it.
#[derive(Default)]
pub struct Bodies {
    /// The bodies, their order, their warmth, their refusals and the seal.
    map: BodyMap,
    /// The tally, the last capture, the settlement counter and the probe seam.
    recorder: Recorder,
}

/// The body map: what stands, in what order, what is warming, what is
/// refused, and whether the map is closed.
#[derive(Default)]
pub struct BodyMap {
    bodies: HashMap<BodyKey, Body>,
    /// Eviction order, least recently launched first.
    body_order: Vec<BodyKey>,
    /// How many fires have run eagerly per key — see [`WARM_FIRES`].
    body_warm: HashMap<BodyKey, u32>,
    /// Keys refused for the life of the load (see [`Uncut::Eager`]).
    bodies_refused: HashSet<BodyKey>,
    /// Set once by [`Bodies::seal_bodies`]; past it an unarmed key is
    /// served eagerly instead of warmed toward a capture.
    sealed: bool,
}

/// The recorder's own state: what it has counted, what it last measured,
/// how far ahead of the device it is, and the probe seam.
#[derive(Default)]
pub struct Recorder {
    /// Probe seam: keeps a capture's `cudaGraph_t` instead of dropping it.
    keep: bool,
    kept: Vec<(BodyKey, Graph)>,
    /// Lets eviction tell whether an exec may still be running.
    airborne: crate::settle::Airborne,
    at_seq: u64,
    bstats: BodyTally,
    last_capture: LastCapture,
}

impl Bodies {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Bodies {
        Bodies::default()
    }

    /// Tell this cache how to ask whether an exec is still in flight.
    pub fn watch(&mut self, airborne: crate::settle::Airborne) {
        self.recorder.watch(airborne);
    }

    /// Stamp the step sequence the fire about to be walked will settle at.
    pub fn at_step(&mut self, seq: u64) {
        self.recorder.at_step(seq);
    }

    /// One more fire that walked eagerly without reaching this cache.
    pub fn eager_walk(&mut self, rotating: bool, buffered: bool) {
        self.recorder.eager_walk(rotating, buffered);
    }

    /// One more fire whose copy world is not its key's.
    pub fn eager_copy_world(&mut self) {
        self.recorder.eager_copy_world();
    }

    /// Probe seam: ask captures to keep their graphs.
    pub fn keep_graphs(&mut self, keep: bool) {
        self.recorder.keep_graphs(keep);
    }

    /// The graphs kept by [`Bodies::keep_graphs`], in capture order.
    #[must_use]
    pub fn kept(&self) -> &[(BodyKey, Graph)] {
        self.recorder.kept()
    }
}

impl Recorder {
    pub fn watch(&mut self, airborne: crate::settle::Airborne) {
        self.airborne = airborne;
    }

    pub fn at_step(&mut self, seq: u64) {
        self.at_seq = seq;
    }

    pub fn eager_walk(&mut self, rotating: bool, buffered: bool) {
        if rotating {
            self.bstats.eager_rotating += 1;
        }
        if buffered {
            self.bstats.eager_buffered += 1;
        }
    }

    pub fn eager_copy_world(&mut self) {
        self.bstats.eager_copy_world += 1;
    }

    pub fn keep_graphs(&mut self, keep: bool) {
        self.keep = keep;
        if !keep {
            self.kept.clear();
        }
    }

    #[must_use]
    pub fn kept(&self) -> &[(BodyKey, Graph)] {
        &self.kept
    }
}

/// One contiguous run of the template, on one side of the capture line —
/// the unit of a segmented body. A function of the [`BodyKey`], so every
/// fire of one key cuts the template the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stretch {
    /// The capture unit these regions belong to (`CompiledModel::unit_of`);
    /// a cut opens wherever the unit changes too.
    pub unit: u32,
    /// The first template region in the stretch.
    pub from: u32,
    /// One past the last.
    pub upto: u32,
    /// `false`: a stretch a graph holds, captured once. `true`: one
    /// re-issued eagerly every fire, at that fire's own live geometry.
    pub island: bool,
}

/// The decline a segmented capture answers instead of recording something
/// wrong. Not a fault: every variant describes a composition the eager walk
/// still serves fine, just with nothing captured for that reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uncut {
    /// The widening ate the whole template: every region is an island.
    Eager {
        /// How many regions the template holds — all of them islands.
        regions: u32,
    },
    /// A cut fell inside a fork group — unreachable, [`widen`] covers it.
    Fork {
        /// The region the boundary opened at.
        region: u32,
    },
    /// A cut fell between two arms of a `SWITCH` group — unreachable,
    /// [`widen`] covers it.
    Bracket {
        /// The region the boundary opened at, an arm past the first.
        region: u32,
    },
    /// A plan builder and a schedule reader disagreed about the seat's
    /// start — unreachable, [`widen`] covers the whole mask family.
    Plan {
        /// The region whose admission disagrees with an earlier region of
        /// the same planned mask.
        region: u32,
    },
}

impl core::fmt::Display for Uncut {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Uncut::Eager { regions } => write!(
                f,
                "all {regions} of its regions are islands once they have grown to \
                 their legal boundaries, so there is no stretch left for a graph to \
                 hold"
            ),
            Uncut::Fork { region } => write!(
                f,
                "a segment boundary at region {region} fell inside a fork group the \
                 widening should have closed"
            ),
            Uncut::Bracket { region } => write!(
                f,
                "a segment boundary at region {region} fell between two arms of one \
                 conditional group"
            ),
            Uncut::Plan { region } => write!(
                f,
                "region {region} disagrees with an earlier region of its own planned \
                 mask about whether a graph can hold it"
            ),
        }
    }
}

/// Grows every island until every boundary around it is legal, turning
/// `Captured` into `Island` (never the reverse — an island is always safe).
/// Floods each weld (fork group, `SWITCH` group, or mask family) to a fixpoint.
#[must_use]
pub fn widen(compiled: &CompiledModel, admits: &[Admit]) -> Vec<Admit> {
    let template = compiled.template();
    let mut table: Vec<Admit> = (0..template.len())
        .map(|at| admits.get(at).copied().unwrap_or(Admit::Island))
        .collect();
    // A composition a graph holds whole has nothing to grow, which is every
    // decode, prefill and mixed key of every catalog SKU.
    if !table.iter().any(|admit| *admit == Admit::Island) {
        return table;
    }
    let welded = welds(compiled);
    loop {
        let mut grew = false;
        for weld in &welded {
            if !weld
                .iter()
                .any(|at| table.get(*at as usize) == Some(&Admit::Island))
            {
                continue;
            }
            for at in weld {
                if let Some(held) = table.get_mut(*at as usize)
                    && *held == Admit::Captured
                {
                    *held = Admit::Island;
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }
    table
}

/// The sets of regions that must share one admission — [`widen`]'s rules,
/// read off the template and never off the table, so two callers with the
/// same table get the same widening.
fn welds(compiled: &CompiledModel) -> Vec<Vec<u32>> {
    let template = compiled.template();
    let mut welds: Vec<Vec<u32>> = Vec::new();

    // Rule 1: a fork group, read off the event ledger — opens where a
    // stream records into an empty ledger, closes after the join region
    // that waits it clear. The join itself is inside the span.
    let mut pending: Vec<model_compiler::EventId> = Vec::new();
    let mut opened: Option<u32> = None;
    for (index, region) in template.iter().enumerate() {
        let at = index as u32;
        let settled = pending.is_empty();
        if settled && let Some(from) = opened.take() {
            welds.push((from..at).collect());
        }
        for event in &region.wait {
            pending.retain(|held| held != event);
        }
        pending.extend(region.open);
        pending.extend(region.close);
        if settled && !pending.is_empty() {
            opened = Some(at);
        }
    }
    // An unjoined group is welded anyway; the belt in `cuts` refuses a
    // template the driver would not accept.
    if let Some(from) = opened {
        welds.push((from..template.len() as u32).collect());
    }

    // Rule 2: a `SWITCH` group, named once by its first arm (`arm == 0`).
    for (index, region) in template.iter().enumerate() {
        if let model_compiler::Lowering::Switch { arm: 0, arms, .. } = region.lowering {
            let from = index as u32;
            let upto = from
                .saturating_add(u32::from(arms))
                .min(template.len() as u32);
            welds.push((from..upto).collect());
        }
    }

    // Rule 3: a planned mask's regions — a prepare builder and every region
    // stating the same mask, only for a mask some prepare region states.
    // Keyed by (mask, unit) rather than mask alone, so a trunk region and a
    // tower region that carry byte-identical masks on different axes don't
    // flood each other.
    let mut planned: Vec<(&model_ir::ClassSet, u32)> = Vec::new();
    for (at, region) in template
        .iter()
        .enumerate()
        .filter(|(_, region)| region.phase == model_compiler::Phase::Prepare)
    {
        let unit = compiled.unit_of(at);
        if !planned
            .iter()
            .any(|(mask, held)| **mask == region.mask && *held == unit)
        {
            planned.push((&region.mask, unit));
        }
    }
    for (mask, unit) in planned {
        let family: Vec<u32> = template
            .iter()
            .enumerate()
            .filter(|(at, region)| region.mask == *mask && compiled.unit_of(*at) == unit)
            .map(|(at, _)| at as u32)
            .collect();
        if family.len() > 1 {
            welds.push(family);
        }
    }
    welds
}

/// Cuts one composition's widened template ([`widen`]) into maximal runs
/// of one `(unit, admission)` pair — one exec per captured run. Errors
/// with [`Uncut::Eager`] when nothing is left to capture.
pub fn cuts(compiled: &CompiledModel, admits: &[Admit]) -> core::result::Result<Vec<Stretch>, Uncut> {
    let template = compiled.template();
    let table = widen(compiled, admits);
    // One mask, one side ([`Uncut::Plan`]) — only when there is an island at
    // all, since a graph that holds the whole composition has nothing to
    // disagree about.
    if table.iter().any(|admit| *admit == Admit::Island) {
        let mut seen: Vec<(&model_ir::ClassSet, Admit)> = Vec::new();
        for (index, region) in template.iter().enumerate() {
            let planned = template.iter().any(|other| {
                other.phase == model_compiler::Phase::Prepare && other.mask == region.mask
            });
            if !planned {
                continue;
            }
            let admit = table.get(index).copied().unwrap_or(Admit::Island);
            match seen.iter().find(|(mask, _)| **mask == region.mask) {
                Some((_, held)) if *held != admit => {
                    return Err(Uncut::Plan { region: index as u32 });
                }
                Some(_) => {}
                None => seen.push((&region.mask, admit)),
            }
        }
    }

    let mut cuts: Vec<Stretch> = Vec::new();
    // Events recorded and not yet waited — P6's ledger, in region order.
    let mut pending: Vec<model_compiler::EventId> = Vec::new();
    for (index, region) in template.iter().enumerate() {
        let at = index as u32;
        let unit = compiled.unit_of(index);
        let island = table.get(index).copied().unwrap_or(Admit::Island) == Admit::Island;
        let extends = cuts
            .last()
            .is_some_and(|open| open.unit == unit && open.island == island);
        if extends {
            if let Some(open) = cuts.last_mut() {
                open.upto = at + 1;
            }
        } else {
            // A new stretch opens here, so this is a boundary the two belts
            // below must be legal against (the first stretch is not one).
            if !cuts.is_empty() {
                if !pending.is_empty() {
                    return Err(Uncut::Fork { region: at });
                }
                if matches!(
                    region.lowering,
                    model_compiler::Lowering::Switch { arm, .. } if arm != 0
                ) {
                    return Err(Uncut::Bracket { region: at });
                }
            }
            cuts.push(Stretch { unit, from: at, upto: at + 1, island });
        }
        for event in &region.wait {
            pending.retain(|held| held != event);
        }
        pending.extend(region.open);
        pending.extend(region.close);
    }
    // A script of nothing but islands is the eager walk with a map entry in
    // front of it, so the key is refused instead — once, through the door
    // every other refusal uses.
    if !cuts.iter().any(|cut| !cut.island) {
        return Err(Uncut::Eager { regions: template.len() as u32 });
    }
    Ok(cuts)
}

/// How many bodies one load may keep. The real bound is a byte budget
/// (`[engine] bodies_mem`); this is a belt for a load that cannot weigh
/// its captures.
pub const MAX_BODIES: usize = 512;

/// Which body a fire asks for: the lattice bucket, per-class ceilings
/// ([`Ladder`]), and an optional second axis for a two-unit load. See
/// [`crate::window::Windows::admits`] for what a graph may capture.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BodyKey {
    /// The lattice point (`Composition::bucket`).
    pub bucket: u32,
    /// Which classes this fire has rows in, and the ceiling each one is
    /// carved to — [`Ladder`], in the order the rows stand.
    pub classes: Ladder,
    /// The second capture unit's own pair, or `None` for an artifact with
    /// no patch axis (every text-only SKU).
    pub patch: Option<AxisKey>,
}

impl BodyKey {
    /// The key of one fire's class table at one lattice point, on the token
    /// axis alone — a test door: `of(..) == of_axes(.., None)` is the G4
    /// invariant, so this exists to state that equality byte for byte.
    #[cfg(test)]
    #[must_use]
    pub fn of(
        classes: &WindowTable,
        bucket: u32,
        decoding: &model_ir::ClassSet,
        lane_ceiling: u32,
    ) -> BodyKey {
        BodyKey {
            bucket,
            classes: Ladder::of(classes, bucket, decoding, lane_ceiling),
            patch: None,
        }
    }

    /// The same key, told the fire's patch rectangle as well — the door a
    /// two-unit artifact's shell takes. `patch: None` when the artifact
    /// states no patch axis, making this identical to [`of`](BodyKey::of).
    #[must_use]
    pub fn of_axes(
        classes: &WindowTable,
        bucket: u32,
        decoding: &model_ir::ClassSet,
        lane_ceiling: u32,
        patch: Option<(&WindowTable, u32)>,
    ) -> BodyKey {
        BodyKey {
            bucket,
            classes: Ladder::of(classes, bucket, decoding, lane_ceiling),
            patch: patch.map(|(classes, bucket)| AxisKey::of(classes, bucket)),
        }
    }
}

/// One capture unit's coordinates, mirroring what a [`BodyKey`] carries
/// for the token rectangle. Every class is carved to the patch bucket
/// itself, via [`Ladder::flat`] rather than [`Ladder::rung`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct AxisKey {
    /// The patch lattice point (`Composition::patch_bucket`), or `0` for a
    /// fire of this artifact that carries no image — the rung that launches
    /// no tower exec.
    pub bucket: u32,
    /// Which classes this fire has PATCH rows in, and the ceiling each is
    /// carved to — all of them [`bucket`](AxisKey::bucket).
    pub classes: Ladder,
}

impl AxisKey {
    /// One fire's patch coordinates, off the patch window table and the
    /// patch bucket. The rung is the bucket itself — both the arming pass
    /// and the fire path reach this function, so they never diverge.
    #[must_use]
    pub fn of(classes: &WindowTable, bucket: u32) -> AxisKey {
        AxisKey {
            bucket,
            classes: Ladder::flat(classes, bucket),
        }
    }
}

/// Which classes have rows, and the ceiling each may be carved over —
/// `(class, rung)` per present class, in row order. A function of (present
/// set, bucket) alone — canonical, never of the fire's actual split.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Ladder(Box<[(u32, u32)]>);

impl Ladder {
    /// One fire's ladder: every class with rows, at the ceiling this key
    /// carves it to ([`Ladder::rung`]), in the order the rows stand.
    #[must_use]
    pub fn of(
        classes: &WindowTable,
        bucket: u32,
        decoding: &model_ir::ClassSet,
        lane_ceiling: u32,
    ) -> Ladder {
        Ladder(
            classes
                .present_in_order()
                .map(|class| {
                    (
                        class,
                        Ladder::rung(class as usize, bucket, decoding, lane_ceiling),
                    )
                })
                .collect(),
        )
    }

    /// The ceiling one class is carved to — the load's lane ceiling for a
    /// decode class, the bucket whole for a prefill one. Must be the only
    /// place a rung is computed, so an armed key and a fired key agree.
    #[must_use]
    pub fn rung(
        class: usize,
        bucket: u32,
        decoding: &model_ir::ClassSet,
        lane_ceiling: u32,
    ) -> u32 {
        if decoding.contains(class) {
            lane_ceiling.min(bucket)
        } else {
            bucket
        }
    }

    /// Every present class at one ceiling — the patch axis's form, which
    /// has no decode notion to call [`rung`](Ladder::rung) with.
    #[must_use]
    pub fn flat(classes: &WindowTable, rung: u32) -> Ladder {
        Ladder(classes.present_in_order().map(|class| (class, rung)).collect())
    }

    /// The one-class ladder — `Shell::arm_bodies`'s form. `rung` must come
    /// from [`Ladder::rung`] and nowhere else, or the armed key names a
    /// body traffic will never ask for.
    #[must_use]
    pub fn single(class: usize, rung: u32) -> Ladder {
        Ladder(vec![(class as u32, rung)].into_boxed_slice())
    }

    /// The pairs, in the order the rows stand.
    #[must_use]
    pub fn rungs(&self) -> &[(u32, u32)] {
        &self.0
    }

    /// Does this class have rows in this key?
    #[must_use]
    pub fn contains(&self, class: usize) -> bool {
        self.0.iter().any(|(held, _)| *held as usize == class)
    }

    /// One past the last row any window of this key may be carved to —
    /// the sum of every present class's rung.
    #[must_use]
    pub fn reach(&self) -> u32 {
        self.0.iter().map(|(_, rung)| *rung).sum()
    }

    /// The same sum read as lanes, each rung capped at the lane ceiling
    /// first.
    #[must_use]
    pub fn lane_reach(&self, lane_ceiling: u32) -> u32 {
        self.0.iter().map(|(_, rung)| (*rung).min(lane_ceiling)).sum()
    }
}

impl core::fmt::Display for BodyKey {
    /// `b8[c0:8 c1:4]` for a text key, `b8[c0:8]+p64[c0:64]` for a tower
    /// one. A `None` patch writes nothing (not `+p0[]`).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "b{}[", self.bucket)?;
        let mut first = true;
        for (class, rung) in self.classes.rungs() {
            if !first {
                f.write_str(" ")?;
            }
            first = false;
            write!(f, "c{class}:{rung}")?;
        }
        f.write_str("]")?;
        if let Some(patch) = &self.patch {
            write!(f, "+{patch}")?;
        }
        Ok(())
    }
}

impl core::fmt::Display for AxisKey {
    /// `p64[c0:64]` — the [`BodyKey`] spelling with a `p` where the `b` is.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "p{}[", self.bucket)?;
        let mut first = true;
        for (class, rung) in self.classes.rungs() {
            if !first {
                f.write_str(" ")?;
            }
            first = false;
            write!(f, "c{class}:{rung}")?;
        }
        f.write_str("]")
    }
}

/// The ladders beside the fire's own class tables, one pair per row axis.
/// A region reads exactly one entry; its capture unit picks which
/// ([`Carve::on`]).
#[derive(Clone, Copy)]
pub struct Carve<'a> {
    /// One axis's pair per row space, or `None` (the eager reading: no
    /// ceiling taken, grids at the window's own live span).
    pub per_axis: model_ir::PerAxis<Option<AxisCarve<'a>>>,
}

/// One row axis's half of a [`Carve`] — this fire's class table, the
/// key's ladder over it, and the lane ceiling the axis has one of.
#[derive(Clone, Copy)]
pub struct AxisCarve<'a> {
    /// This fire's class table on this axis — `Composition::classes` on the
    /// token axis, `Composition::patch_classes` on the patch one.
    pub classes: &'a WindowTable,
    /// The key's ladder over it — [`BodyKey::classes`] on the token axis,
    /// [`AxisKey::classes`] on the patch one.
    pub ladder: &'a Ladder,
    /// The load's lane ceiling, on the axis that has one — `min(slots,
    /// Budget::max_lanes, Budget::max_tokens)`. `None` on the patch axis:
    /// its lane is an image, and nothing here carves an image count yet.
    pub lane_ceiling: Option<u32>,
}

impl<'a> Carve<'a> {
    /// This carve read on one row axis — the token pair for a token-unit
    /// region, the patch pair for a patch-unit one, `None` for an axis this
    /// fire has no ladder for. The unit is the only thing that picks.
    #[must_use]
    pub fn on(&self, axis: model_ir::RowAxis) -> Option<AxisCarve<'a>> {
        self.per_axis[axis]
    }
}

impl AxisCarve<'_> {
    /// How many rows stand in front of this span, and how many it may be
    /// carved over. `None` for a gathered or grouped span (an island).
    #[must_use]
    pub fn ceiling(&self, span: MaskSpan) -> Option<(u32, u32)> {
        self.prefix(span, u32::MAX)
    }

    /// The same two numbers read as lanes: rungs capped at
    /// [`AxisCarve::lane_ceiling`] first. `None` wherever
    /// [`ceiling`](AxisCarve::ceiling) is, or on an axis with no lane ceiling.
    #[must_use]
    pub fn lanes(&self, span: MaskSpan) -> Option<(u32, u32)> {
        self.prefix(span, self.lane_ceiling?)
    }

    /// The prefix walk both readings share, with each rung capped at `cap`
    /// — `u32::MAX` for rows, the lane ceiling for lanes.
    fn prefix(&self, span: MaskSpan, cap: u32) -> Option<(u32, u32)> {
        let end = span.row_offset + span.rows;
        let (mut before, mut own) = (0u32, 0u32);
        for (class, rung) in self.ladder.rungs() {
            let rung = (*rung).min(cap);
            let window = self.classes.class(*class as usize);
            let last = window.row_offset + window.rows;
            if last <= span.row_offset {
                before += rung;
            } else if window.row_offset >= end {
                // Wholly behind the span: contributes to neither number.
            } else if window.row_offset >= span.row_offset && last <= end {
                own += rung;
            } else {
                return None;
            }
        }
        Some((before, own))
    }
}

/// One step of a body's replay script — a stretch the graph holds, or a
/// stretch it re-issues eagerly. No second vector: the sequence itself is
/// the representation.
enum Step {
    /// Launch the exec captured for one stretch, or walk that stretch
    /// under the golden's eager twin.
    Exec { exec: GraphExec, cut: Stretch },
    /// Re-issue one contiguous stretch of island regions eagerly, on the
    /// same stream between the execs around it, so its launches are byte
    /// for byte the launches the eager walk makes.
    Island(Stretch),
}

/// One recorded body: the replay script, the plan-payload shape hash it
/// was captured against, and the step it last launched at.
struct Body {
    /// One [`Step`] per [`Stretch`] cut. Empty stretches are dropped.
    script: Box<[Step]>,
    /// `(rows, lanes)` per captured launch, at the ceiling gridded.
    grids: Box<[(u32, u32)]>,
    /// A fire whose shape disagrees is a miss, not a refusal — it re-captures.
    shape: u64,
    /// [`Airborne::NEVER`](crate::settle::Airborne::NEVER) if never
    /// launched; lets eviction avoid a still-running exec.
    launched_at: u64,
    /// `true` for a composition `Shell::arm_bodies` armed; exempts it
    /// from the LRU eviction scan.
    pinned: bool,
    /// Node count at the load's measured per-node price, or `None` when
    /// the probe refused.
    bytes: Option<usize>,
}

/// One fire's per-launch grids, `(rows, lanes)` in walk order
/// ([`Body::grids`]'s layout). Captured regions only.
fn launch_grids(at: &Fire<'_>, run: &Run<'_>) -> Box<[(u32, u32)]> {
    let mut grids = Vec::new();
    for region in 0..at.compiled.template().len() as u32 {
        if at.island(region) {
            continue;
        }
        for at_run in 0..run.windows().runs(region) {
            grids.push(launch_grid(at, run, region, at_run));
        }
    }
    grids.into_boxed_slice()
}

/// The ceiling one (region, run)'s launches were gridded at. An axis
/// with no ladder or armed bucket takes its own live span.
fn launch_grid(at: &Fire<'_>, run: &Run<'_>, region: u32, at_run: u32) -> (u32, u32) {
    let windows = run.windows();
    let span = windows.at(region, at_run).span();
    let axis = windows.axis_of(region);
    let carved = at.ceilings.carve.and_then(|carve| carve.on(axis)).is_some();
    if !carved || at.ceilings.pads[axis].bucket == 0 {
        return (span.rows, span.lanes);
    }
    let standing = run.standing_as(region, at_run, true);
    let rows = standing.rows(span).unwrap_or(span.rows);
    let lanes = standing.lanes(windows, span).unwrap_or(span.lanes);
    (rows.max(span.rows), lanes.max(span.lanes))
}

/// Does this fire ask any launch for a bigger grid than the capture
/// holds? A belt, not a climb — should never fire inside one key.
fn grew_past(held: &[(u32, u32)], at: &Fire<'_>, run: &Run<'_>) -> bool {
    let mut seen = 0usize;
    for region in 0..at.compiled.template().len() as u32 {
        // Islands are skipped on the read exactly as on the write.
        if at.island(region) {
            continue;
        }
        for at_run in 0..run.windows().runs(region) {
            let Some(&(rows, lanes)) = held.get(seen) else {
                return true;
            };
            let (want_rows, want_lanes) = launch_grid(at, run, region, at_run);
            if want_rows > rows || want_lanes > lanes {
                return true;
            }
            seen += 1;
        }
    }
    seen != held.len()
}

/// What this cache has counted since the load. Monotonic totals; counts
/// reasons, not always fires (a fire disqualified twice counts twice).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BodyTally {
    /// Fires that replayed a body. At steady state this is every fire.
    pub hits: u64,
    /// Fires whose key held no body yet — warming or capturing.
    pub misses: u64,
    /// Fires that found their body but demoted it (schedule shape moved)
    /// and re-captured. Expect zero.
    pub reshapes: u64,
    /// Bodies captured and instantiated.
    pub captures: u64,
    /// Fires that ran eagerly because a schedule declined to be
    /// graph-shaped (a bucket that declines once declines every time).
    pub declines: u64,
    /// Compositions no key can name, or [`widen`] left nothing captured
    /// in ([`Uncut::Eager`]). Per composition, not per fire.
    pub refusals: u64,
    /// Bodies dropped to stay under [`MAX_BODIES`], plus captures declined
    /// for want of a droppable seat. Never an armed body ([`Body::pinned`]).
    pub evictions: u64,
    /// Bodies captured before any fire arrived (a subset of
    /// [`captures`](BodyTally::captures)); also the pinned count.
    pub armed_at_load: u64,
    /// Fires the sealed map turned away. Zero is expected.
    pub sealed_declines: u64,
    /// Fires of an armed key whose carve outgrew the body's recorded grid
    /// ([`grew_past`]) and walked instead.
    pub sealed_short: u64,
    /// Eager walks the router took without asking this cache (dense
    /// planes rotate). Nonzero under `Bodies::On` is a warning.
    pub eager_rotating: u64,
    /// A fire that moved buffered RS bytes; not summed with
    /// [`eager_rotating`](BodyTally::eager_rotating).
    pub eager_buffered: u64,
    /// A fire whose copy-fallback policy disagrees with the world its
    /// key was armed under. Zero on a load that states the policy once.
    pub eager_copy_world: u64,
}

/// The most recently captured body, measured. Assigned by each capture
/// rather than accumulated, so these three settle at boot and a reader may
/// not subtract two readings of them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LastCapture {
    /// Nodes, summed across the body's execs.
    pub nodes: usize,
    /// Edges, summed across the body's execs (a P6 fork's only observable).
    pub edges: usize,
    /// Stretches of the body's template no graph holds, re-issued eagerly.
    /// Zero is the common case.
    pub islands: usize,
}

/// What stands in the map right now. Folded at every reading and stored
/// nowhere, since `insert_body` replaces bodies in place.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BodyCensus {
    /// Bodies resident now.
    pub bodies: usize,
    /// How many of them are segmented — hold at least one island, so their
    /// replay is not one submission. Not [`LastCapture::islands`], which
    /// counts stretches inside one body rather than bodies that have any.
    pub segmented: usize,
    /// What the resident bodies took off the device, in bytes — what the
    /// arming pass spends against `[engine] bodies_mem`. Zero on a load
    /// that cannot measure, which is honest: no body was weighed there.
    pub bytes: usize,
    /// Bodies the device would not weigh.
    pub unweighed: usize,
}

/// What this load's graph cache has done. Three groups, three lifetimes:
/// [`BodyTally`] accumulates for the load's life, [`LastCapture`] is
/// assigned by the last capture, [`BodyCensus`] is refolded on every read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BodyStats {
    /// Everything counted since the load — see [`BodyTally`].
    pub tally: BodyTally,
    /// The last capture's three measurements — see [`LastCapture`].
    pub last_capture: LastCapture,
    /// The resident map, refolded for this reading — see [`BodyCensus`].
    pub census: BodyCensus,
}

impl core::fmt::Display for BodyStats {
    /// Three segments, each `name=value` token stable for operators and
    /// tests to grep.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let tally = &self.tally;
        let last = &self.last_capture;
        let census = &self.census;
        write!(
            f,
            "[body-stats] hits={} misses={} reshapes={} captures={} \
             declines={} refusals={} evictions={} armed_at_load={} \
             sealed_declines={} sealed_short={} eager_rotating={} eager_buffered={} \
             eager_copy_world={} | last-capture: nodes={} edges={} islands={} \
             | resident: bodies={} segmented={} bytes={}",
            tally.hits,
            tally.misses,
            tally.reshapes,
            tally.captures,
            tally.declines,
            tally.refusals,
            tally.evictions,
            tally.armed_at_load,
            tally.sealed_declines,
            tally.sealed_short,
            tally.eager_rotating,
            tally.eager_buffered,
            tally.eager_copy_world,
            last.nodes,
            last.edges,
            last.islands,
            census.bodies,
            census.segmented,
            census.bytes,
        )
    }
}

impl Bodies {
    /// What the bodies path has done. See [`BodyStats`].
    #[must_use]
    pub fn body_stats(&self) -> BodyStats {
        BodyStats {
            tally: self.recorder.bstats,
            last_capture: self.recorder.last_capture,
            census: self.map.census(),
        }
    }

    /// No body will ever stand for this composition — counted once per key ([`BodyTally::refusals`]).
    pub fn body_refuse(&mut self, key: BodyKey) {
        if self.map.refuse(key) {
            self.recorder.bstats.refusals += 1;
        }
    }

    /// One more body the load armed: pinned in the map and counted ([`BodyTally::armed_at_load`]).
    /// Answers whether the key held a body to arm.
    pub fn body_armed(&mut self, key: &BodyKey) -> bool {
        let armed = self.map.pin(key);
        if armed {
            self.recorder.bstats.armed_at_load += 1;
        }
        armed
    }

    /// Close the map — `Shell::arm_bodies`'s last line. Idempotent and one-way.
    pub fn seal_bodies(&mut self) {
        self.map.seal();
    }

    /// Is the map closed? See [`BodyMap::sealed`].
    #[must_use]
    pub fn bodies_sealed(&self) -> bool {
        self.map.sealed()
    }

    /// Has this key already been refused admission?
    #[must_use]
    pub fn body_refused(&self, key: &BodyKey) -> bool {
        self.map.refused(key)
    }

    /// Is this key already captured?
    #[must_use]
    pub fn holds_body(&self, key: &BodyKey) -> bool {
        self.map.holds(key)
    }

    /// Run one fire against its body: prepare eagerly, then replay or
    /// record (a miss walks eagerly first). Errors with
    /// [`Fault::Fire`](crate::Fault::Fire) or [`Fault::Device`](crate::Fault::Device).
    pub fn fire_body(&mut self, at: &Fire<'_>, run: &mut Run<'_>, place: &At) -> Result<()> {
        // 1. Prepare: the host half, on the open stream, under every outcome.
        let mut prepare = at.serial(place);
        walk_phases(
            at.trace,
            at.compiled,
            at.descriptor,
            run,
            &mut prepare,
            Phases::Prepare,
        )?;
        prepare.settle()?;
        let shape = run.schedule_shape();
        // The key, on every axis the artifact states. `towered` is a load
        // constant, not read off this fire, so a text lane of a vision SKU
        // keys into the same family as an image lane.
        let key = BodyKey::of_axes(
            at.descriptor.table(model_ir::RowAxis::Tokens),
            at.descriptor.bucket,
            at.decoding,
            at.lane_ceiling,
            at.towered.then_some((
                at.descriptor.table(model_ir::RowAxis::Patches),
                at.descriptor.patch_bucket,
            )),
        );

        // 2. A hit is one submission per unit at the shell's staged
        //    live-rows seat. Not a hit if the body is too short
        //    (`Body::grids`, falls through and re-captures at the larger
        //    count) or its shape hash moved (`BodyTally::reshapes`).
        let at_seq = self.recorder.at_seq;
        // Asked per launch, not off the fire's total.
        let (short, moved, empty) = match self.map.bodies.get(&key) {
            Some(body) => {
                let short = grew_past(&body.grids, at, run);
                // `moved` only when short isn't the reason; `empty` is a
                // belt (the capture no longer seats an empty script).
                (short, !short && body.shape != shape, body.script.is_empty())
            }
            None => (false, false, false),
        };
        if moved {
            self.recorder.bstats.reshapes += 1;
        }
        if short && self.map.sealed() {
            // The lane column depends on the fire's own lane staging, so
            // an armed key can be too short for a wider fire of itself.
            self.recorder.bstats.sealed_short += 1;
        }
        let replays = !short && !moved && !empty;
        if replays && let Some(body) = self.map.bodies.get_mut(&key) {
            // The hit path: one host for-loop over one stream, captured
            // stretches submitted and islands re-issued eagerly between
            // them in stream order. Inputs are already fresh from phase
            // 1's unfiltered prepare walk.
            //
            // Assert: every island this body re-issues is one this fire's
            // own derivation also names, in order (a subsequence, since
            // the capture loop drops prepare-only/empty-node stretches).
            debug_assert!(
                {
                    let asked = cuts(at.compiled, at.ceilings.admits);
                    let asked = asked.as_deref().unwrap_or(&[]);
                    let mut at_cut = asked.iter();
                    body.script
                        .iter()
                        .filter_map(|step| match step {
                            Step::Island(cut) => Some(*cut),
                            Step::Exec { .. } => None,
                        })
                        .all(|mine| at_cut.any(|theirs| *theirs == mine))
                },
                "the resident body for {key} re-issues a stretch this fire does not ask \
                 for. `Windows::admits` is a function of the key, so two fires of one \
                 key cut the template in the same places; if they did not, the \
                 admissibility table has grown an input the key does not carry",
            );
            // A script of nothing never gets here — the capture refuses to
            // seat one, and this says so where the launches are.
            debug_assert!(
                !body.script.is_empty(),
                "the resident body for {key} holds no steps, so replaying it launches \
                 nothing and reports the fire served — the caller would read whatever \
                 the readout rectangle held from the last fire that ran",
            );
            // The golden's control arm ([`Fire::eager_twin`]) walks what it
            // would have launched instead of launching the exec.
            for step in body.script.iter() {
                match step {
                    Step::Exec { exec, .. } if !at.eager_twin => exec.launch(at.stream)?,
                    Step::Exec { cut, .. } | Step::Island(cut) => {
                        walk_capture_cut(at, run, place, Streams::Serial, *cut)?;
                    }
                }
            }
            body.launched_at = at_seq;
            self.map.touch(&key);
            self.recorder.bstats.hits += 1;
            return Ok(());
        }

        // 3. A miss runs for real. The fire's numbers come from here, and so
        //    does every lazily-warmed thing a capture must not do.
        walk_capture(at, run, place, Streams::Serial)?;

        // A sealed map mints nothing (`BodyTally::sealed_declines`, not
        // `misses`).
        if self.sealed_decline() {
            // Named once, so an operator knows which shape to widen by.
            if self.recorder.bstats.sealed_declines == 1 {
                eprintln!(
                    "engine-cuda: the sealed map holds no body for {key} — \
                     this shape walks eagerly for the life of the load \
                     (BodyTally::sealed_declines counts each such fire)"
                );
            }
            return Ok(());
        }
        self.recorder.bstats.misses += 1;

        let warmed = self.map.warm(&key);
        if warmed < WARM_FIRES {
            return Ok(());
        }
        if !run.capturable() {
            // A property of the key (overflowed float grant), so this
            // body declines forever. Printed once per key.
            if warmed == WARM_FIRES {
                eprintln!(
                    "engine-cuda: body {key} declines to capture — a schedule it \
                     built would not fit its workspace grant, so `graph_capturable` \
                     is false and this composition walks eagerly for good. The \
                     prefill float grant is sized at the lattice's top rung in \
                     `inputs::reserve` (`prefill_float_bytes`); a bucket that \
                     outgrows it is this line."
                );
            }
            self.recorder.bstats.declines += 1;
            return Ok(());
        }

        // 4. The same regions again, recorded rather than run, one capture
        //    per cut. `Shell::prepare` already ran [`cuts`] and refused the
        //    key on failure, so the `let ... else` is unreachable in
        //    practice; kept so a failure records nothing rather than half.
        let Ok(script) = cuts(at.compiled, at.ceilings.admits) else {
            return Ok(());
        };
        let mut steps: Vec<Step> = Vec::with_capacity(script.len());
        let mut nodes = 0;
        let mut edges = 0;
        let mut islands = 0usize;
        let mut bytes: Option<usize> = Some(0);
        for cut in script {
            if cut.island {
                islands += 1;
                steps.push(Step::Island(cut));
                continue;
            }
            let graph =
                Graph::capture(at.stream, || walk_capture_cut(at, run, place, Streams::Forked, cut))?;
            // Prepare-only stretches that recorded nothing are dropped
            // (`Some(0)`, not `== 0` — a refused query is not empty).
            // Capture-phase ones become islands instead, re-issued eagerly.
            if graph.nodes() == Some(0) {
                let prepare_only = (cut.from..cut.upto).all(|at_region| {
                    at.compiled
                        .template()
                        .get(at_region as usize)
                        .is_some_and(|region| region.phase == model_compiler::Phase::Prepare)
                });
                if prepare_only {
                    continue;
                }
                islands += 1;
                steps.push(Step::Island(cut));
                continue;
            }
            // Priced by node count at a per-load measured rate, not the
            // free-memory delta (the driver sub-allocates from a reservation).
            let per_node = Recorder::node_price(&graph);
            let exec = graph.instantiate(at.stream)?;
            let took = exec.nodes();
            bytes = match (bytes, per_node) {
                (Some(held), Some(price)) => Some(held + took * price),
                _ => None,
            };
            nodes += took;
            // Report-only: a refused query under-reports here too.
            edges += graph.edges().unwrap_or(0);
            steps.push(Step::Exec { exec, cut });
            if self.recorder.keep {
                self.recorder.kept.push((key.clone(), graph));
            }
        }
        self.recorder.last_capture = LastCapture { nodes, edges, islands };
        // A body whose script is empty is not a body — refused through the
        // same door every other refusal uses.
        if steps.is_empty() {
            self.body_refuse(key);
            return Ok(());
        }
        let grids = launch_grids(at, run);
        let _ = self.insert_body(key, Body {
            script: steps.into_boxed_slice(),
            grids,
            shape,
            // Summed across the body's stretches; what `Shell::arm_bodies`
            // stops the arming pass on ([`Body::bytes`]).
            bytes,
            launched_at: crate::settle::Airborne::NEVER,
            // Not pinned from here: this line is every capture there is
            // (arming rungs included), so [`Bodies::body_armed`] marks it
            // afterwards, from the one caller that knows.
            pinned: false,
        });
        // A capture the map had no room for is not a capture at all:
        // nothing was cached, and the next fire of this composition walks
        // again (`BodyTally::evictions`).
        Ok(())
    }

    /// Does the seal refuse to mint? `true` closes the fire out and counts
    /// it; `false` is an unsealed map and the miss-then-capture ladder,
    /// unchanged. On its own so a host test can ask it without a device.
    fn sealed_decline(&mut self) -> bool {
        if !self.map.sealed() {
            return false;
        }
        self.recorder.bstats.sealed_declines += 1;
        true
    }

    /// Seat a body ([`BodyMap::insert`]) and tally what it cost. Answers whether it was seated.
    fn insert_body(&mut self, key: BodyKey, body: Body) -> bool {
        let seating = self.map.insert(key, body, &self.recorder.airborne);
        self.recorder.bstats.evictions += seating.evictions;
        self.recorder.bstats.captures += u64::from(seating.seated);
        seating.seated
    }
}

impl Recorder {
    /// What one graph node's exec costs the device, in bytes, measured once
    /// and reused for every body this process prices. `None` if the driver
    /// refuses the probe ([`BodyCensus::unweighed`]).
    fn node_price(graph: &Graph) -> Option<usize> {
        /// Enough instantiations that one reservation boundary cannot
        /// dominate the reading, few enough to be transient at load.
        const COPIES: usize = 8;

        static PRICE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        *PRICE.get_or_init(|| {
            let nodes = graph.nodes().filter(|nodes| *nodes > 0)?;
            let (bytes, _) = crate::device::nodes::exec_footprint(graph, COPIES).ok()?;
            // A zero delta is the driver's pool absorbing the execs, not a
            // free body: unweighed, so the count belt is the only bound.
            (bytes > 0.0).then(|| (bytes / nodes as f64).ceil() as usize)
        })
    }
}

/// What seating a body cost: whether it was seated, and how many evictions
/// it took (a seat declined for want of a droppable body counts one).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Seating {
    seated: bool,
    evictions: u64,
}

impl BodyMap {
    /// Is this key already captured?
    fn holds(&self, key: &BodyKey) -> bool {
        self.bodies.contains_key(key)
    }

    /// Has this key already been refused admission?
    fn refused(&self, key: &BodyKey) -> bool {
        self.bodies_refused.contains(key)
    }

    /// No body will ever stand for this composition. Answers whether the
    /// key was newly refused, so the caller counts compositions not traffic.
    fn refuse(&mut self, key: BodyKey) -> bool {
        self.bodies_refused.insert(key)
    }

    /// Is the map closed?
    fn sealed(&self) -> bool {
        self.sealed
    }

    /// Close the map. Called once at least one key seated. Idempotent,
    /// one-way.
    fn seal(&mut self) {
        self.sealed = true;
    }

    /// Mark this key's body the load's own ([`Body::pinned`]). Answers
    /// whether the key held a body to pin.
    fn pin(&mut self, key: &BodyKey) -> bool {
        let Some(body) = self.bodies.get_mut(key) else {
            return false;
        };
        debug_assert!(
            !body.script.is_empty(),
            "the arming pass is about to count {key} armed and its body holds no steps; \
             a load that armed it would report every fire of that composition served \
             while launching nothing",
        );
        body.pinned = true;
        true
    }

    /// Move a body to the back of the eviction order.
    fn touch(&mut self, key: &BodyKey) {
        if let Some(at) = self.body_order.iter().position(|held| held == key) {
            let key = self.body_order.remove(at);
            self.body_order.push(key);
        }
    }

    /// One more eager sighting of this key; answers how many it has had.
    fn warm(&mut self, key: &BodyKey) -> u32 {
        if self.body_warm.len() > MAX_BODIES * 4 {
            let held = &self.bodies;
            self.body_warm.retain(|key, _| held.contains_key(key));
        }
        let seen = self.body_warm.entry(key.clone()).or_insert(0);
        *seen += 1;
        *seen
    }

    /// Folded at every reading and stored nowhere (`insert` replaces
    /// bodies in place, so a maintained counter would drift).
    fn census(&self) -> BodyCensus {
        BodyCensus {
            bodies: self.bodies.len(),
            segmented: self
                .bodies
                .values()
                .filter(|body| body.script.iter().any(|step| matches!(step, Step::Island(_))))
                .count(),
            bytes: self.bodies.values().filter_map(|body| body.bytes).sum(),
            unweighed: self.bodies.values().filter(|body| body.bytes.is_none()).count(),
        }
    }

    /// Seat a body, dropping the least recently launched settled and
    /// unpinned one if full (armed bodies, [`Body::pinned`], are never
    /// candidates). Declines rather than growing the map. Answers what it
    /// cost ([`Seating`]).
    fn insert(&mut self, key: BodyKey, body: Body, airborne: &crate::settle::Airborne) -> Seating {
        let mut seating = Seating::default();
        // The map holds no empty body — the belt under `fire_body`'s own
        // refusal, and the one a host test can hold without a device.
        if body.script.is_empty() {
            return seating;
        }
        // A replacement is not an insert: this key's body is too short for
        // the traffic (`Body::grids`), so this is a swap that must never
        // drop a `cudaGraphExec_t` the device is still running.
        if let Some((launched_at, pinned)) = self
            .bodies
            .get(&key)
            .map(|held| (held.launched_at, held.pinned))
        {
            if !airborne.settled_past(launched_at) {
                seating.evictions += 1;
                return seating;
            }
            // The pin survives the swap: it belongs to the key, not the exec.
            let body = Body { pinned, ..body };
            self.bodies.insert(key.clone(), body);
            self.touch(&key);
            seating.seated = true;
            return seating;
        }
        while self.body_order.len() >= MAX_BODIES {
            // Least recently launched first, settled and unpinned. An
            // order entry whose body is already gone is droppable either way.
            let Some(at) = self.body_order.iter().position(|key| {
                self.bodies.get(key).is_none_or(|body| {
                    !body.pinned && airborne.settled_past(body.launched_at)
                })
            }) else {
                // Every resident body may still be on the device, or is one
                // the load armed. This composition tries again next fire.
                seating.evictions += 1;
                return seating;
            };
            let evicted = self.body_order.remove(at);
            self.bodies.remove(&evicted);
            self.body_warm.remove(&evicted);
            seating.evictions += 1;
        }
        seating.seated = true;
        self.body_order.push(key.clone());
        self.bodies.insert(key, body);
        seating
    }
}

/// Which schedule of P6's DAG this walk is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Streams {
    /// One stream, program order — the serialization, and the golden.
    Serial,
    /// The baked streams and event points, which only a capture wants.
    Forked,
}

impl<'a> Fire<'a> {
    /// Is this region one the body re-issues eagerly? A region the table
    /// does not hold defaults to island — costs a launch, not a bad address.
    fn island(&self, region: u32) -> bool {
        self.ceilings.admits.get(region as usize) != Some(&Admit::Captured)
    }

    /// A cursor that stays on the main stream, and puts the stream cell back
    /// there if a capture pass left it somewhere else.
    fn serial(&self, place: &'a At) -> Cursor<'a> {
        if let Some(lanes) = self.lanes {
            lanes.at.set(0);
        }
        // Same reset for a load that carries the cell on the conditional
        // bundle instead (a baked `If` with no fork group has one, no `Lanes`).
        if let Some(cond) = self.conditionals {
            cond.at.set(0);
        }
        Cursor::new(place)
    }
}

/// The capture-phase regions, dispatched. A fresh [`Cursor`] each time,
/// counting regions from zero (the window index every `Run` resolution reads).
fn walk_capture(
    at: &Fire<'_>,
    run: &mut Run<'_>,
    place: &At,
    streams: Streams,
) -> Result<()> {
    walk_capture_units(at, run, place, streams, Units::All, Regions::All)
}

/// The same capture, restricted to one [`Stretch`]. The cursor still
/// walks the whole template every pass, dispatching only inside
/// `[from, upto)`, so a fork group split across segments states its
/// matched record/wait pair in every segment — keeping each graph
/// independently joinable.
fn walk_capture_cut(
    at: &Fire<'_>,
    run: &mut Run<'_>,
    place: &At,
    streams: Streams,
    cut: Stretch,
) -> Result<()> {
    walk_capture_units(
        at,
        run,
        place,
        streams,
        Units::One(cut.unit),
        Regions::Span { from: cut.from, upto: cut.upto },
    )
}

fn walk_capture_units(
    at: &Fire<'_>,
    run: &mut Run<'_>,
    place: &At,
    streams: Streams,
    units: Units,
    regions: Regions,
) -> Result<()> {
    let mut cursor = match (streams, at.lanes) {
        (Streams::Forked, Some(lanes)) => Cursor::across(place, lanes),
        _ => at.serial(place),
    };
    // Whether this walk is being written down is separate from whether it
    // has side streams; only the conditional bracket reads it.
    if streams == Streams::Forked {
        cursor = cursor.writing();
        if let Some(cond) = at.conditionals {
            cursor = cursor.conditionals(cond);
        }
    }
    let walked = walk_regions(
        at.trace,
        at.compiled,
        at.descriptor,
        run,
        &mut cursor,
        Phases::Capture,
        units,
        regions,
    );
    // Asked inside the capture body so `Graph::capture` still ends it, and
    // even on a refused walk, since `settle` also closes a conditional
    // body — left open, it poisons every later call on the stream.
    let settled = cursor.settle();
    walked?;
    settled?;
    Ok(())
}

impl core::fmt::Debug for Bodies {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Bodies")
            .field("bodies", &self.map.bodies.len())
            .field("sealed", &self.map.sealed)
            .field("stats", &self.body_stats())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_exec::fire::ClassWindow;

    /// The lane ceiling these tests carve a decode class to, chosen so the
    /// `min` in [`Ladder::rung`] is always the interesting half.
    const LANES: u32 = 4;

    /// A bake with no `attention.decode` arm: every present class is
    /// carved to the bucket.
    fn prefill_only() -> model_ir::ClassSet {
        model_ir::ClassSet::default()
    }

    fn table(classes: &[(u32, u32)]) -> WindowTable {
        let mut at = (0, 0);
        WindowTable::new(
            classes
                .iter()
                .map(|(rows, lanes)| {
                    let window = ClassWindow {
                        row_offset: at.0,
                        rows: *rows,
                        lane_offset: at.1,
                        lanes: *lanes,
                    };
                    at = (at.0 + rows, at.1 + lanes);
                    window
                })
                .collect(),
        )
    }

    // ── G4: a text-only load is byte-for-byte unmoved ──────────────────
    //
    // A key with `patch: None` must be the key that existed before the
    // field did — same construction, `Eq`, `Hash`, and `Display`.

    /// A decode class is carved to the lane ceiling, a prefill one to the
    /// bucket, and the arming pass computes the same number traffic will
    /// ([`Ladder::rung`], asserted from both of its callers).
    #[test]
    fn a_rung_is_the_keys_own_ceiling_and_arming_computes_the_same_one() {
        let decoding = model_ir::ClassSet::of([0usize]);
        let fired = BodyKey::of(&table(&[(3, 3)]), 8, &decoding, LANES);
        assert_eq!(
            fired.to_string(),
            "b8[c0:4]",
            "the lane ceiling binds below the bucket, and three rows say nothing",
        );
        assert_eq!(
            BodyKey::of(&table(&[(3, 3)]), 2, &decoding, LANES).to_string(),
            "b2[c0:2]",
            "and the bucket binds below the lane ceiling",
        );
        assert_eq!(
            BodyKey::of(&table(&[(3, 3)]), 8, &prefill_only(), LANES).to_string(),
            "b8[c0:8]",
            "a class the decode arm does not name takes the bucket whole",
        );

        // `Shell::arm_bodies` builds the one-class ladder by hand, through
        // the same rung function.
        let armed = BodyKey {
            bucket: 8,
            classes: Ladder::single(0, Ladder::rung(0, 8, &decoding, LANES)),
            patch: None,
        };
        assert_eq!(armed, fired, "the armed key must be the fired key");
    }

    /// One key per rung. `insert_body`'s eviction arithmetic reads only the
    /// launch stamp and the pin, so this runs on a host with no CUDA at all.
    fn rung(bucket: u32) -> BodyKey {
        BodyKey {
            bucket,
            // A prefill class's canonical ceiling is its bucket
            // ([`Ladder::rung`]).
            classes: Ladder::single(0, bucket),
            patch: None,
        }
    }

    /// The same body, carrying a stated device footprint
    /// ([`Body::bytes`]) for the census to add up.
    fn weighing(bytes: usize) -> Body {
        Body {
            // One island step: a script must not be empty, and an island
            // is the one step a host test can build without a device.
            script: vec![Step::Island(Stretch { unit: 0, from: 0, upto: 1, island: true })]
                .into_boxed_slice(),
            grids: Vec::new().into_boxed_slice(),
            shape: 0,
            // Never launched, so `Airborne::settled_past` answers `true`.
            launched_at: crate::settle::Airborne::NEVER,
            pinned: false,
            bytes: Some(bytes),
        }
    }

    /// The budget's arithmetic is a fold and not a ledger
    /// ([`BodyCensus::bytes`]): what a load has spent is what its resident
    /// bodies weigh, so a replaced-in-place body gives its bytes back at
    /// the same line it gives its seat back.
    #[test]
    fn what_a_load_has_spent_is_what_its_resident_bodies_weigh() {
        let mut graphs = Bodies::new();
        assert_eq!(graphs.body_stats().census.bytes, 0, "an empty map has spent nothing");
        for (bucket, bytes) in [(8u32, 3_000usize), (16, 5_000), (32, 7_000)] {
            assert!(graphs.insert_body(rung(bucket), weighing(bytes)));
        }
        assert_eq!(
            graphs.body_stats().census.bytes,
            15_000,
            "the census is the sum over the residents: {}",
            graphs.body_stats(),
        );
        // A re-capture of one key is a swap, not a second charge: the
        // total moves by the difference, not by the whole.
        assert!(graphs.insert_body(rung(16), weighing(6_000)));
        assert_eq!(
            graphs.body_stats().census.bytes,
            16_000,
            "a replacement charged twice: {}",
            graphs.body_stats(),
        );
        // A load that cannot weigh (no runtime: `free_bytes` answers
        // `None`) spends nothing; the seat count belt is the only bound.
        assert!(graphs.insert_body(rung(64), weighing(0)));
        assert_eq!(
            graphs.body_stats().census.bytes,
            16_000,
            "an unweighable body moved the budget: {}",
            graphs.body_stats(),
        );
        assert_eq!(graphs.body_stats().census.bodies, 4, "and it still took a seat");
    }

    /// A body whose script is empty is never seated and never counted
    /// armed: previously, a captured stretch whose node count the driver
    /// refused to give read as "recorded nothing", so a key could be seated
    /// with zero steps and silently replay stale readout forever.
    #[test]
    fn the_map_never_inserts_a_body_whose_script_is_empty() {
        let mut graphs = Bodies::new();
        let classes = table(&[(8, 1)]);
        let key = BodyKey::of(&classes, 8, &prefill_only(), LANES);
        let empty = Body {
            script: Box::new([]),
            grids: Box::new([]),
            shape: 0,
            launched_at: crate::settle::Airborne::NEVER,
            pinned: false,
            bytes: Some(0),
        };

        assert!(
            !graphs.insert_body(key.clone(), empty),
            "the map seated a body with no steps: a fire of {key} would launch nothing \
             and be reported served",
        );
        assert!(
            !graphs.holds_body(&key),
            "an empty body was refused a seat and the map holds it anyway",
        );
        assert!(
            !graphs.body_armed(&key),
            "the arming pass would count {key} armed, and the boot line's `armed a of b` \
             is the sentence an operator stops looking at",
        );
        assert_eq!(
            graphs.body_stats().tally.captures,
            0,
            "an empty body is not a capture: nothing was recorded",
        );
    }
}
