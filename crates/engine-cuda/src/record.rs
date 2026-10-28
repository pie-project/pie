use std::collections::{HashMap, HashSet};

use poem_compiler::CompiledModel;
use poem_exec::fire::{
    FireDescriptor, MaskSpan, Phases, Regions, Units, WindowTable, walk_phases, walk_regions,
};
use poem_ir::Trace;

use crate::device::graph::{Graph, GraphExec};
use crate::error::Result;
use crate::run::{Ceilings, Run};
use crate::window::{Admit, At, Cursor, Lanes};

pub const WARM_FIRES: u32 = 2;

pub struct Fire<'a> {
    pub eager_twin: bool,
    pub trace: &'a Trace,
    pub compiled: &'a CompiledModel,
    pub descriptor: &'a FireDescriptor,
    pub stream: *mut core::ffi::c_void,
    pub lanes: Option<Lanes<'a>>,
    pub conditionals: Option<crate::window::Conditionals<'a>>,
    pub decoding: &'a poem_ir::ClassSet,
    pub tiers: &'a Tiers,
    pub towered: bool,
    pub lane_ceiling: u32,
    pub ceilings: Ceilings<'a>,
    /// [`BodyKey::islands`] as prepared for this fire.
    pub islands: &'a [u8],
}
#[derive(Default)]
pub struct Bodies {
    map: BodyMap,
    recorder: Recorder,
}

#[derive(Default)]
pub struct BodyMap {
    bodies: HashMap<BodyKey, Body>,
    body_order: Vec<BodyKey>,
    body_warm: HashMap<BodyKey, u32>,
    bodies_refused: HashSet<BodyKey>,
    sealed: bool,
}

#[derive(Default)]
pub struct Recorder {
    keep: bool,
    kept: Vec<(BodyKey, Graph)>,
    airborne: crate::settle::Airborne,
    at_seq: u64,
    bstats: BodyTally,
    last_capture: LastCapture,
    sealed_seen: HashSet<BodyKey>,
}

/// Whether a sealed-map decline is worth a log line: the first fire of each
/// shape says which shape walks eagerly, and every doubling of the count
/// says how many have — a load that declines every decode step for minutes
/// leaves a handful of lines, not one per thousand fires.
fn reports_decline<K: Eq + std::hash::Hash + Clone>(
    seen: &mut HashSet<K>,
    key: &K,
    declines: u64,
) -> bool {
    if seen.len() > MAX_BODIES * 4 {
        seen.clear();
    }
    let first = !seen.contains(key);
    if first {
        seen.insert(key.clone());
    }
    first || declines.is_power_of_two()
}

impl Bodies {
    #[must_use]
    pub fn new() -> Bodies {
        Bodies::default()
    }

    pub fn watch(&mut self, airborne: crate::settle::Airborne) {
        self.recorder.watch(airborne);
    }

    pub fn at_step(&mut self, seq: u64) {
        self.recorder.at_step(seq);
    }

    pub fn eager_walk(&mut self, rotating: bool, buffered: bool) {
        self.recorder.eager_walk(rotating, buffered);
    }

    pub fn eager_copy_world(&mut self) {
        self.recorder.eager_copy_world();
    }

    pub fn keep_graphs(&mut self, keep: bool) {
        self.recorder.keep_graphs(keep);
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stretch {
    pub unit: u32,
    pub from: u32,
    pub upto: u32,
    pub island: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uncut {
    Eager { regions: u32 },
    Fork { region: u32 },
    Bracket { region: u32 },
    Plan { region: u32 },
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

#[must_use]
pub fn widen(compiled: &CompiledModel, admits: &[Admit]) -> Vec<Admit> {
    let template = compiled.template();
    let mut table: Vec<Admit> = (0..template.len())
        .map(|at| admits.get(at).copied().unwrap_or(Admit::Island))
        .collect();
    if !table.contains(&Admit::Island) {
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

fn fork_open(template: &[poem_compiler::Region]) -> Vec<bool> {
    let n = template.len();
    let mut open = vec![false; n];
    for (src, region) in template.iter().enumerate() {
        for event in region.open.into_iter().chain(region.close) {
            if let Some(last) = (src + 1..n).rfind(|&dst| template[dst].wait.contains(&event)) {
                for slot in &mut open[src + 1..=last] {
                    *slot = true;
                }
            }
        }
        if region.stream != 0
            && region.close.is_none()
            && let Some(next) = (src + 1..n).find(|&dst| template[dst].stream == region.stream)
        {
            for slot in &mut open[src + 1..=next] {
                *slot = true;
            }
        }
    }
    open
}

fn welds(compiled: &CompiledModel) -> Vec<Vec<u32>> {
    let template = compiled.template();
    let mut welds: Vec<Vec<u32>> = Vec::new();

    let open_fork = fork_open(template);
    let mut index = 1usize;
    while index < template.len() {
        if !open_fork[index] {
            index += 1;
            continue;
        }
        let from = (index - 1) as u32;
        while index < template.len() && open_fork[index] {
            index += 1;
        }
        welds.push((from..index as u32).collect());
    }

    for (index, region) in template.iter().enumerate() {
        if let poem_compiler::Lowering::Switch { arm: 0, arms, .. } = region.lowering {
            let from = index as u32;
            let upto = from
                .saturating_add(u32::from(arms))
                .min(template.len() as u32);
            welds.push((from..upto).collect());
        }
    }

    let mut planned: Vec<(&poem_ir::ClassSet, u32)> = Vec::new();
    for (at, region) in template
        .iter()
        .enumerate()
        .filter(|(_, region)| region.phase == poem_compiler::Phase::Prepare)
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

pub fn cuts(
    compiled: &CompiledModel,
    admits: &[Admit],
) -> core::result::Result<Vec<Stretch>, Uncut> {
    let template = compiled.template();
    let table = widen(compiled, admits);
    if table.contains(&Admit::Island) {
        let mut seen: Vec<((&poem_ir::ClassSet, u32), Admit)> = Vec::new();
        for (index, region) in template.iter().enumerate() {
            let unit = compiled.unit_of(index);
            let planned = template.iter().enumerate().any(|(other_at, other)| {
                other.phase == poem_compiler::Phase::Prepare
                    && other.mask == region.mask
                    && compiled.unit_of(other_at) == unit
            });
            if !planned {
                continue;
            }
            let admit = table.get(index).copied().unwrap_or(Admit::Island);
            match seen
                .iter()
                .find(|((mask, held_unit), _)| **mask == region.mask && *held_unit == unit)
            {
                Some((_, held)) if *held != admit => {
                    return Err(Uncut::Plan {
                        region: index as u32,
                    });
                }
                Some(_) => {}
                None => seen.push(((&region.mask, unit), admit)),
            }
        }
    }

    let open_fork = fork_open(template);
    let mut cuts: Vec<Stretch> = Vec::new();
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
            if !cuts.is_empty() {
                if open_fork.get(index).copied().unwrap_or(false) {
                    return Err(Uncut::Fork { region: at });
                }
                if matches!(
                    region.lowering,
                    poem_compiler::Lowering::Switch { arm, .. } if arm != 0
                ) {
                    return Err(Uncut::Bracket { region: at });
                }
            }
            cuts.push(Stretch {
                unit,
                from: at,
                upto: at + 1,
                island,
            });
        }
    }
    if !cuts.iter().any(|cut| !cut.island) {
        return Err(Uncut::Eager {
            regions: template.len() as u32,
        });
    }
    Ok(cuts)
}

/// Bodies a load keeps beside `mem_mb`: the count bounds host bookkeeping and
/// graph handles, which bytes do not measure.
pub const MAX_BODIES: usize = 1024;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tiers {
    pub plain: poem_ir::ClassSet,
    /// The classes one body per bucket lays out whole, in the order a fire
    /// lays their rows out: a composition holding a class off the plain
    /// tier keys this list whichever of its classes are present, and its
    /// body launches every class's regions at their rungs, an absent
    /// class's with no live row ([`crate::run::Run::launches_at_ceiling`]).
    /// So one body answers every subset, where a key per subset would
    /// multiply with the classes a mix of programs seats.
    pub whole: Box<[u32]>,
}

impl Tiers {
    /// Whether `ladder` is the whole tier's: a wide key, whichever of its
    /// classes the fire seats.
    #[must_use]
    pub fn keys_whole(&self, ladder: &Ladder) -> bool {
        !self.whole.is_empty()
            && ladder.rungs().len() == self.whole.len()
            && ladder
                .rungs()
                .iter()
                .zip(self.whole.iter())
                .all(|((class, _), held)| class == held)
    }

    /// The regions a fire keyed by `ladder` gates on the device
    /// ([`poem_exec::fire::FireDescriptor::gated`]): under a wide key, each
    /// captured, shifting region that runs some of the tier's classes but
    /// not all, so the one body holds it whichever of them a fire seats,
    /// and each region planning only for gated regions (`readers`: the
    /// regions reading a region's plans), so a plan is made at its reader's
    /// ceiling exactly when its reader is held.
    #[must_use]
    pub fn gates(
        &self,
        ladder: &Ladder,
        template: &[poem_compiler::Region],
        shifted: &[bool],
        admits: &[Admit],
        readers: &[Vec<u32>],
    ) -> Vec<bool> {
        if !self.keys_whole(ladder) {
            return Vec::new();
        }
        let mut gated: Vec<bool> = template
            .iter()
            .enumerate()
            .map(|(at, region)| {
                let held = self
                    .whole
                    .iter()
                    .filter(|class| region.mask.contains(**class as usize))
                    .count();
                region.phase == poem_compiler::Phase::Capture
                    && held > 0
                    && held < self.whole.len()
                    && shifted.get(at).copied().unwrap_or(false)
                    && admits.get(at) == Some(&Admit::Captured)
            })
            .collect();
        for (at, region) in template.iter().enumerate() {
            let reads = readers.get(at).map_or(&[][..], Vec::as_slice);
            if region.phase == poem_compiler::Phase::Prepare && !reads.is_empty() {
                gated[at] = reads
                    .iter()
                    .all(|reader| gated.get(*reader as usize).copied().unwrap_or(false));
            }
        }
        gated
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BodyKey {
    pub bucket: u32,
    pub classes: Ladder,
    pub patch: Option<AxisKey>,
    /// The regions the fire's windows keep out of a graph, one bit each
    /// (empty: none). A sliding or evicted window admits differently from
    /// a whole one at the same bucket and classes, and a body is only
    /// replayable for the admissibility it was recorded under.
    pub islands: Box<[u8]>,
}

impl BodyKey {
    #[cfg(test)]
    #[must_use]
    pub fn of(
        classes: &WindowTable,
        bucket: u32,
        decoding: &poem_ir::ClassSet,
        lane_ceiling: u32,
        tiers: &Tiers,
    ) -> BodyKey {
        BodyKey {
            bucket,
            classes: Ladder::of(classes, bucket, decoding, lane_ceiling, tiers),
            patch: None,
            islands: Box::default(),
        }
    }

    #[must_use]
    pub fn of_axes(
        classes: &WindowTable,
        bucket: u32,
        decoding: &poem_ir::ClassSet,
        lane_ceiling: u32,
        tiers: &Tiers,
        patch: Option<(&WindowTable, u32)>,
    ) -> BodyKey {
        BodyKey {
            bucket,
            classes: Ladder::of(classes, bucket, decoding, lane_ceiling, tiers),
            patch: patch.map(|(classes, bucket)| AxisKey::of(classes, bucket)),
            islands: Box::default(),
        }
    }

    /// The same key, carrying which regions `admits` keeps out of a graph.
    #[must_use]
    pub fn with_islands(mut self, admits: &[Admit]) -> BodyKey {
        let mut bits = vec![0u8; admits.len().div_ceil(8)];
        for (at, admit) in admits.iter().enumerate() {
            if *admit == Admit::Island {
                bits[at / 8] |= 1 << (at % 8);
            }
        }
        while bits.last() == Some(&0) {
            bits.pop();
        }
        self.islands = bits.into();
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct AxisKey {
    pub bucket: u32,
    pub classes: Ladder,
}

impl AxisKey {
    #[must_use]
    pub fn of(classes: &WindowTable, bucket: u32) -> AxisKey {
        AxisKey {
            bucket,
            classes: Ladder::flat(classes, bucket),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Ladder(Box<[(u32, u32)]>);

impl Ladder {
    #[must_use]
    pub fn of(
        classes: &WindowTable,
        bucket: u32,
        decoding: &poem_ir::ClassSet,
        lane_ceiling: u32,
        tiers: &Tiers,
    ) -> Ladder {
        let plain = classes
            .present_in_order()
            .all(|class| tiers.plain.contains(class as usize));
        let whole = classes
            .present_in_order()
            .all(|class| tiers.whole.contains(&class));
        let listed: Vec<u32> = if plain {
            classes.present_in_order().collect()
        } else if whole {
            tiers.whole.to_vec()
        } else {
            Vec::new()
        };
        Ladder(
            listed
                .into_iter()
                .map(|class| {
                    (
                        class,
                        Ladder::rung(class as usize, bucket, decoding, lane_ceiling),
                    )
                })
                .collect(),
        )
    }

    #[must_use]
    pub fn rung(class: usize, bucket: u32, decoding: &poem_ir::ClassSet, lane_ceiling: u32) -> u32 {
        if decoding.contains(class) {
            lane_ceiling.min(bucket)
        } else {
            bucket
        }
    }

    #[must_use]
    pub fn flat(classes: &WindowTable, rung: u32) -> Ladder {
        Ladder(
            classes
                .present_in_order()
                .map(|class| (class, rung))
                .collect(),
        )
    }

    #[must_use]
    pub fn single(class: usize, rung: u32) -> Ladder {
        Ladder(vec![(class as u32, rung)].into_boxed_slice())
    }

    #[must_use]
    pub fn rungs(&self) -> &[(u32, u32)] {
        &self.0
    }

    #[must_use]
    pub fn contains(&self, class: usize) -> bool {
        self.0.iter().any(|(held, _)| *held as usize == class)
    }

    #[must_use]
    pub fn reach(&self) -> u32 {
        self.0.iter().map(|(_, rung)| *rung).sum()
    }

    #[must_use]
    pub fn lane_reach(&self, lane_ceiling: u32) -> u32 {
        self.0
            .iter()
            .map(|(_, rung)| (*rung).min(lane_ceiling))
            .sum()
    }
}

impl core::fmt::Display for BodyKey {
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
        if !self.islands.is_empty() {
            f.write_str("~i")?;
            for byte in self.islands.iter().rev() {
                write!(f, "{byte:02x}")?;
            }
        }
        Ok(())
    }
}

impl core::fmt::Display for AxisKey {
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

#[derive(Clone, Copy)]
pub struct Carve<'a> {
    pub per_axis: poem_ir::PerAxis<Option<AxisCarve<'a>>>,
}

#[derive(Clone, Copy)]
pub struct AxisCarve<'a> {
    pub classes: &'a WindowTable,
    pub ladder: &'a Ladder,
    pub lane_ceiling: Option<u32>,
}

impl<'a> Carve<'a> {
    #[must_use]
    pub fn on(&self, axis: poem_ir::RowAxis) -> Option<AxisCarve<'a>> {
        self.per_axis[axis]
    }
}

impl AxisCarve<'_> {
    /// The rows before a region's classes and its own, by the region's
    /// class mask where that answers ([`AxisCarve::of_mask`]), else by
    /// where the fire's rows sit.
    #[must_use]
    pub fn ceiling(&self, span: MaskSpan, mask: Option<&poem_ir::ClassSet>) -> Option<(u32, u32)> {
        mask.and_then(|mask| self.of_mask(mask, u32::MAX))
            .or_else(|| self.prefix(span, u32::MAX))
    }

    #[must_use]
    pub fn lanes(&self, span: MaskSpan, mask: Option<&poem_ir::ClassSet>) -> Option<(u32, u32)> {
        let cap = self.lane_ceiling?;
        let (_, own) = mask
            .and_then(|mask| self.of_mask(mask, cap))
            .or_else(|| self.prefix(span, cap))?;
        Some((span.lane_offset, own.min(cap)))
    }

    /// The rungs before a region's classes and its own, read off the
    /// ladder (which lists its classes in the order a fire lays them out)
    /// by the region's mask: the same for every fire of the key, whichever
    /// of its classes are present. Its own rungs run from the mask's first
    /// ladder class to its last, the classes between included: a launch
    /// there walks whatever of that hull a fire seats (a grouped op its
    /// union, [`crate::GROUPED`]). `None` where the ladder holds none of
    /// the mask's classes.
    /// The classes of `mask`'s `run`th run on the ladder (consecutive rungs
    /// of its classes): the run a wide key cuts a region into
    /// (`FireDescriptor::order`), whose rows its `run`th window holds.
    #[must_use]
    pub fn run_mask(&self, mask: &poem_ir::ClassSet, run: u32) -> poem_ir::ClassSet {
        let mut at = 0u32;
        let mut open = false;
        let mut held: Vec<usize> = Vec::new();
        for &(class, _) in self.ladder.rungs() {
            let class = class as usize;
            if !mask.contains(class) {
                if open {
                    at += 1;
                }
                open = false;
                continue;
            }
            open = true;
            if at == run {
                held.push(class);
            }
        }
        poem_ir::ClassSet::of(held)
    }

    fn of_mask(&self, mask: &poem_ir::ClassSet, cap: u32) -> Option<(u32, u32)> {
        let rungs = self.ladder.rungs();
        let first = rungs
            .iter()
            .position(|(class, _)| mask.contains(*class as usize))?;
        let last = rungs
            .iter()
            .rposition(|(class, _)| mask.contains(*class as usize))?;
        let sum = |held: &[(u32, u32)]| held.iter().map(|(_, rung)| (*rung).min(cap)).sum();
        Some((sum(&rungs[..first]), sum(&rungs[first..=last])))
    }

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
            } else if window.row_offset >= span.row_offset && last <= end {
                own += rung;
            } else {
                return None;
            }
        }
        Some((before, own))
    }
}

pub static REPLAY_UPTO: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);
pub static REPLAY_FROM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static PTR_TAG: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn ptr_traced(key: &BodyKey) -> bool {
    match crate::serve::diag::on().ptr_trace.as_deref() {
        Some(want) => key.to_string().contains(want),
        None => false,
    }
}

enum Step {
    Exec {
        exec: GraphExec,
        cut: Stretch,
        gated: Option<Gated>,
    },
    Island(Stretch),
}

/// A stretch holding gated regions (`FireDescriptor::gated`): the graph it
/// was captured into, each gated region's nodes, and an instantiation for
/// each set of them a replay has left out, made the first time a fire seats
/// none of those regions' rows. A fire launches the one for the regions it
/// seats no row of, so nothing runs a launch with no row, and no exec in
/// flight is ever changed.
struct Gated {
    #[allow(dead_code)]
    graph: Graph,
    runs: Box<[((u32, u32), Box<[usize]>)]>,
    left_out: Box<[(u32, u32)]>,
}

impl Gated {
    /// The exec for a fire whose windows are `run`'s: updates `whole` in-place
    /// via `cuGraphNodeSetEnabled` whenever the set of zero-row gated runs
    /// changes, avoiding a second `cudaGraphInstantiateWithFlags` on graphs
    /// that hold conditional nodes.
    fn exec<'b>(
        &'b mut self,
        whole: &'b GraphExec,
        run: &Run<'_>,
        stream: *mut core::ffi::c_void,
    ) -> Result<&'b GraphExec> {
        let windows = run.windows();
        let absent: Box<[(u32, u32)]> = self
            .runs
            .iter()
            .map(|(at, _)| *at)
            .filter(|&(region, at)| windows.at(region, at).span().rows == 0)
            .collect();
        if absent != self.left_out {
            #[cfg(feature = "device")]
            crate::device::ctx::sync(stream)?;
            for (at, nodes) in self.runs.iter() {
                let was_absent = self.left_out.contains(at);
                let now_absent = absent.contains(at);
                if was_absent != now_absent {
                    whole.set_enabled(nodes, !now_absent)?;
                }
            }
            self.left_out = absent;
        }
        Ok(whole)
    }
}

struct Body {
    script: Box<[Step]>,
    grids: Box<[(u32, u32)]>,
    shape: u64,
    launched_at: u64,
    pinned: bool,
    bytes: Option<usize>,
}

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

fn grew_past(held: &[(u32, u32)], at: &Fire<'_>, run: &Run<'_>) -> bool {
    let mut seen = 0usize;
    for region in 0..at.compiled.template().len() as u32 {
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BodyTally {
    pub hits: u64,
    pub misses: u64,
    pub reshapes: u64,
    pub captures: u64,
    pub declines: u64,
    pub refusals: u64,
    pub evictions: u64,
    pub armed_at_load: u64,
    pub sealed_declines: u64,
    pub sealed_short: u64,
    pub eager_rotating: u64,
    pub eager_buffered: u64,
    pub eager_copy_world: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LastCapture {
    pub nodes: usize,
    pub edges: usize,
    pub islands: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BodyCensus {
    pub bodies: usize,
    pub segmented: usize,
    pub bytes: usize,
    pub unweighed: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BodyStats {
    pub tally: BodyTally,
    pub last_capture: LastCapture,
    pub census: BodyCensus,
}

impl core::fmt::Display for BodyStats {
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
    #[must_use]
    pub fn body_stats(&self) -> BodyStats {
        BodyStats {
            tally: self.recorder.bstats,
            last_capture: self.recorder.last_capture,
            census: self.map.census(),
        }
    }

    pub fn body_refuse(&mut self, key: BodyKey) {
        if self.map.refuse(key) {
            self.recorder.bstats.refusals += 1;
        }
    }

    pub fn body_armed(&mut self, key: &BodyKey) -> bool {
        let armed = self.map.pin(key);
        if armed {
            self.recorder.bstats.armed_at_load += 1;
        }
        armed
    }

    pub fn body_script(&self, key: &BodyKey) -> Vec<(bool, u32, u32)> {
        self.map
            .bodies
            .get(key)
            .map(|body| {
                body.script
                    .iter()
                    .map(|step| match step {
                        Step::Exec { cut, .. } => (false, cut.from, cut.upto),
                        Step::Island(cut) => (true, cut.from, cut.upto),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn seal_bodies(&mut self) {
        self.map.seal();
    }

    #[must_use]
    pub fn body_refused(&self, key: &BodyKey) -> bool {
        self.map.refused(key)
    }

    #[must_use]
    pub fn holds_body(&self, key: &BodyKey) -> bool {
        self.map.holds(key)
    }

    pub fn fire_body(&mut self, at: &Fire<'_>, run: &mut Run<'_>, place: &At) -> Result<()> {
        let mut prepare = at.serial(place);
        walk_phases(
            at.trace,
            at.compiled,
            at.descriptor,
            run,
            &mut prepare,
            Phases::Prepare,
        )?;
        crate::serve::btrace::mark("walk");
        prepare.settle()?;
        crate::serve::btrace::mark("settle");
        let shape = run.schedule_shape();
        let mut key = BodyKey::of_axes(
            at.descriptor.table(poem_ir::RowAxis::Tokens),
            at.descriptor.bucket,
            at.decoding,
            at.lane_ceiling,
            at.tiers,
            at.towered.then_some((
                at.descriptor.table(poem_ir::RowAxis::Patches),
                at.descriptor.patch_bucket,
            )),
        );
        key.islands = at.islands.into();

        crate::serve::btrace::mark("key");
        let at_seq = self.recorder.at_seq;
        let (short, moved, empty) = match self.map.bodies.get(&key) {
            Some(body) => {
                let short = grew_past(&body.grids, at, run);
                (short, !short && body.shape != shape, body.script.is_empty())
            }
            None => (false, false, false),
        };
        if moved {
            self.recorder.bstats.reshapes += 1;
        }
        if short && self.map.sealed() {
            self.recorder.bstats.sealed_short += 1;
        }
        let replays = !short && !moved && !empty;
        if crate::serve::diag::on().golden_probe {
            let held = self.map.bodies.get(&key).map(|body| body.shape);
            eprintln!(
                "[body-probe] {key} eager_twin={} replays={replays} short={short} moved={moved} empty={empty} shape={shape:#x} held={held:x?}",
                at.eager_twin
            );
        }
        if replays && let Some(body) = self.map.bodies.get_mut(&key) {
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
            debug_assert!(
                !body.script.is_empty(),
                "the resident body for {key} holds no steps, so replaying it launches \
                 nothing and reports the fire served — the caller would read whatever \
                 the readout rectangle held from the last fire that ran",
            );
            crate::serve::btrace::mark("body_lookup");
            let upto = REPLAY_UPTO.load(std::sync::atomic::Ordering::Relaxed);
            let from = REPLAY_FROM.load(std::sync::atomic::Ordering::Relaxed);
            let mut nth = 0usize;
            for step in body.script.iter_mut() {
                match step {
                    Step::Exec { exec, cut, gated } => {
                        let launch = !at.eager_twin && nth >= from && nth < upto;
                        nth += 1;
                        if launch {
                            crate::serve::btrace::mark("pre_launch");
                            match gated {
                                Some(gated) => gated.exec(exec, run, at.stream)?.launch(at.stream)?,
                                None => exec.launch(at.stream)?,
                            }
                            crate::serve::btrace::mark("launch");
                        } else {
                            walk_capture_cut(at, run, place, Streams::Serial, *cut)?;
                        }
                    }
                    Step::Island(cut) => {
                        walk_capture_cut(at, run, place, Streams::Serial, *cut)?;
                    }
                }
            }
            body.launched_at = at_seq;
            self.map.touch(&key);
            self.recorder.bstats.hits += 1;
            return Ok(());
        }

        // The warm fires walk the streams the capture will: a side stream's
        // regions take their own slabs (`scratch_stream` scopes by stream),
        // which a capture cannot grow. Nothing is capturing, so no region
        // opens a conditional node.
        walk_capture(at, run, place, Streams::Warm)?;

        if self.sealed_decline() {
            let declines = self.recorder.bstats.sealed_declines;
            if reports_decline(&mut self.recorder.sealed_seen, &key, declines) {
                let why = if !self.map.bodies.contains_key(&key) {
                    "no body is resident for it"
                } else if short {
                    "its grids grew past the resident body"
                } else if moved {
                    "its schedule shape moved from the resident body's"
                } else {
                    "the resident body is empty"
                };
                eprintln!(
                    "engine-cuda: the sealed map holds no body for {key} ({why}) — \
                     this shape walks eagerly for the life of the load \
                     (BodyTally::sealed_declines counts each such fire; {declines} so far) {}",
                    self.body_stats()
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
            if warmed == WARM_FIRES {
                eprintln!(
                    "engine-cuda: body {key} declines to capture — a schedule it \
                     built refused its graph form ({}), so `graph_capturable` \
                     is false and this composition walks eagerly for good. The \
                     prefill float grant is sized at the lattice's top rung in \
                     `inputs::reserve` (`prefill_float_bytes`); a bucket that \
                     outgrows it is this line.",
                    run.uncapturable_why().join("; ")
                );
            }
            self.recorder.bstats.declines += 1;
            return Ok(());
        }

        let Ok(script) = cuts(at.compiled, at.ceilings.admits) else {
            return Ok(());
        };
        // An arming expert-parallel group captures and instantiates its
        // bodies with nothing in flight (`EpRank::fence`): an instantiation
        // takes the context, and a peer's collective spinning on the same
        // device would wait on the launch that holds back.
        let fence = |run: &Run<'_>| run.ep_rank().map_or(Ok(()), |ep| ep.fence(at.stream));
        fence(run)?;
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
            let mut marked: Vec<crate::window::RegionMark> = Vec::new();
            let graph = {
                if ptr_traced(&key) {
                    PTR_TAG.store(1, std::sync::atomic::Ordering::Relaxed);
                }
                let captured = Graph::capture(at.stream, || {
                    marked = capture_cut(at, run, place, cut)?;
                    Ok(())
                });
                PTR_TAG.store(0, std::sync::atomic::Ordering::Relaxed);
                captured?
            };
            if let Some(dir) = crate::serve::diag::on().graph_dot.as_deref()
                && ptr_traced(&key)
            {
                let nth = steps
                    .iter()
                    .filter(|step| matches!(step, Step::Exec { .. }))
                    .count();
                let path = format!("{}/exec{nth}.dot", dir.display());
                let wrote = graph.debug_dot(&path);
                eprintln!(
                    "[graph-dot] {key} exec {nth} regions {}..{} -> {path} ({})",
                    cut.from,
                    cut.upto,
                    if wrote { "written" } else { "REFUSED" }
                );
            }
            if graph.nodes() == Some(0) {
                let prepare_only = (cut.from..cut.upto).all(|at_region| {
                    at.compiled
                        .template()
                        .get(at_region as usize)
                        .is_some_and(|region| region.phase == poem_compiler::Phase::Prepare)
                });
                if prepare_only {
                    continue;
                }
                islands += 1;
                steps.push(Step::Island(cut));
                continue;
            }
            let per_node = Recorder::node_price(&graph);
            let Ok(exec) = graph.instantiate(at.stream) else {
                islands += 1;
                steps.push(Step::Island(cut));
                continue;
            };
            let took = exec.nodes();
            bytes = match (bytes, per_node) {
                (Some(held), Some(price)) => Some(held + took * price),
                _ => None,
            };
            nodes += took;
            edges += graph.edges().unwrap_or(0);
            let mut regions: Vec<((u32, u32), Box<[usize]>)> = Vec::with_capacity(marked.len());
            for mark in marked {
                let nodes = graph.disableable(&mark.nodes)?;
                if !nodes.is_empty() {
                    regions.push(((mark.region, mark.run), nodes.into_boxed_slice()));
                }
            }
            let regions = regions.into_boxed_slice();
            if regions.is_empty() {
                steps.push(Step::Exec {
                    exec,
                    cut,
                    gated: None,
                });
                if self.recorder.keep {
                    self.recorder.kept.push((key.clone(), graph));
                }
            } else {
                steps.push(Step::Exec {
                    exec,
                    cut,
                    gated: Some(Gated {
                        graph,
                        runs: regions,
                        left_out: Box::new([]),
                    }),
                });
            }
        }
        // Every rank instantiated before any launches what it captured.
        fence(run)?;
        self.recorder.last_capture = LastCapture {
            nodes,
            edges,
            islands,
        };
        if steps.is_empty() {
            self.body_refuse(key);
            return Ok(());
        }
        let grids = launch_grids(at, run);
        if let Some(wanted) = crate::serve::diag::on().grid_trace.as_deref()
            && key.to_string().contains(wanted)
        {
            let windows = run.windows();
            let mut seen = 0usize;
            for region in 0..at.compiled.template().len() as u32 {
                if at.island(region) {
                    continue;
                }
                let template = &at.compiled.template()[region as usize];
                for at_run in 0..windows.runs(region) {
                    let span = windows.at(region, at_run).span();
                    let (rows, lanes) = grids.get(seen).copied().unwrap_or((0, 0));
                    seen += 1;
                    eprintln!(
                        "[grid-trace] {key} r{region} run{at_run} nodes={:?} phase={:?} stream={} \
                         live=({}, {}) grid=({rows}, {lanes})",
                        template.nodes, template.phase, template.stream, span.rows, span.lanes
                    );
                }
            }
        }
        let seated = self.insert_body(
            key.clone(),
            Body {
                script: steps.into_boxed_slice(),
                grids,
                shape,
                bytes,
                launched_at: crate::settle::Airborne::NEVER,
                pinned: false,
            },
        );
        if seated && let Some(body) = self.map.bodies.get_mut(&key) {
            for step in body.script.iter_mut() {
                match step {
                    Step::Exec {
                        exec,
                        gated: Some(gated),
                        ..
                    } => gated.exec(exec, run, at.stream)?.launch(at.stream)?,
                    Step::Exec { exec, .. } => exec.launch(at.stream)?,
                    Step::Island(cut) => {
                        walk_capture_cut(at, run, place, Streams::Serial, *cut)?;
                    }
                }
            }
            body.launched_at = at_seq;
        }
        Ok(())
    }

    fn sealed_decline(&mut self) -> bool {
        if !self.map.sealed() {
            return false;
        }
        self.recorder.bstats.sealed_declines += 1;
        true
    }

    fn insert_body(&mut self, key: BodyKey, body: Body) -> bool {
        let seating = self.map.insert(key, body, &self.recorder.airborne);
        self.recorder.bstats.evictions += seating.evictions;
        self.recorder.bstats.captures += u64::from(seating.seated);
        seating.seated
    }
}

impl Recorder {
    fn node_price(graph: &Graph) -> Option<usize> {
        const COPIES: usize = 8;

        static PRICE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        *PRICE.get_or_init(|| {
            let nodes = graph.nodes().filter(|nodes| *nodes > 0)?;
            let bytes = graph.instantiated_bytes(COPIES).ok()?;
            let price = (bytes > 0.0).then(|| (bytes / nodes as f64).ceil() as usize);
            if crate::serve::diag::on().arm_trace {
                eprintln!(
                    "[arm-trace] node price {price:?} B/node, weighed on a {nodes}-node graph \
                     ({bytes:.0} B per instantiation over {COPIES} copies)"
                );
            }
            price
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Seating {
    seated: bool,
    evictions: u64,
}

impl BodyMap {
    fn holds(&self, key: &BodyKey) -> bool {
        self.bodies.contains_key(key)
    }

    fn refused(&self, key: &BodyKey) -> bool {
        self.bodies_refused.contains(key)
    }

    fn refuse(&mut self, key: BodyKey) -> bool {
        self.bodies_refused.insert(key)
    }

    fn sealed(&self) -> bool {
        self.sealed
    }

    fn seal(&mut self) {
        self.sealed = true;
    }

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

    fn touch(&mut self, key: &BodyKey) {
        if self.body_order.last() == Some(key) {
            return;
        }
        if let Some(at) = self.body_order.iter().position(|held| held == key) {
            let key = self.body_order.remove(at);
            self.body_order.push(key);
        }
    }

    fn warm(&mut self, key: &BodyKey) -> u32 {
        if self.body_warm.len() > MAX_BODIES * 4 {
            let held = &self.bodies;
            self.body_warm.retain(|key, _| held.contains_key(key));
        }
        let seen = self.body_warm.entry(key.clone()).or_insert(0);
        *seen += 1;
        *seen
    }

    fn census(&self) -> BodyCensus {
        BodyCensus {
            bodies: self.bodies.len(),
            segmented: self
                .bodies
                .values()
                .filter(|body| {
                    body.script
                        .iter()
                        .any(|step| matches!(step, Step::Island(_)))
                })
                .count(),
            bytes: self.bodies.values().filter_map(|body| body.bytes).sum(),
            unweighed: self
                .bodies
                .values()
                .filter(|body| body.bytes.is_none())
                .count(),
        }
    }

    fn insert(&mut self, key: BodyKey, body: Body, airborne: &crate::settle::Airborne) -> Seating {
        let mut seating = Seating::default();
        if body.script.is_empty() {
            return seating;
        }
        if let Some((launched_at, pinned)) = self
            .bodies
            .get(&key)
            .map(|held| (held.launched_at, held.pinned))
        {
            if !airborne.settled_past(launched_at) {
                seating.evictions += 1;
                return seating;
            }
            let body = Body { pinned, ..body };
            self.bodies.insert(key.clone(), body);
            self.touch(&key);
            seating.seated = true;
            return seating;
        }
        while self.body_order.len() >= MAX_BODIES {
            let Some(at) = self.body_order.iter().position(|key| {
                self.bodies
                    .get(key)
                    .is_none_or(|body| !body.pinned && airborne.settled_past(body.launched_at))
            }) else {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Streams {
    Serial,
    /// The capture's side streams, eagerly: no conditional node is opened.
    Warm,
    Forked,
}

impl<'a> Fire<'a> {
    fn island(&self, region: u32) -> bool {
        self.ceilings.admits.get(region as usize) != Some(&Admit::Captured)
    }

    fn serial(&self, place: &'a At) -> Cursor<'a> {
        if let Some(lanes) = self.lanes {
            lanes.at.set(0);
        }
        if let Some(cond) = self.conditionals {
            cond.at.set(0);
        }
        Cursor::new(place)
    }
}

fn walk_capture(at: &Fire<'_>, run: &mut Run<'_>, place: &At, streams: Streams) -> Result<()> {
    walk_capture_units(at, run, place, streams, Units::All, Regions::All).map(|_| ())
}

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
        Regions::Span {
            from: cut.from,
            upto: cut.upto,
        },
    )
    .map(|_| ())
}

/// [`walk_capture_cut`] under a capture, answering the nodes each gated
/// region (`FireDescriptor::gated`) added to it.
fn capture_cut(
    at: &Fire<'_>,
    run: &mut Run<'_>,
    place: &At,
    cut: Stretch,
) -> Result<Vec<crate::window::RegionMark>> {
    walk_capture_units(
        at,
        run,
        place,
        Streams::Forked,
        Units::One(cut.unit),
        Regions::Span {
            from: cut.from,
            upto: cut.upto,
        },
    )
}

fn walk_capture_units(
    at: &Fire<'_>,
    run: &mut Run<'_>,
    place: &At,
    streams: Streams,
    units: Units,
    regions: Regions,
) -> Result<Vec<crate::window::RegionMark>> {
    // Each pass launches its own quantizations: a capture that reused the
    // warm pass's rows would replay their codes, not the fire's.
    #[cfg(feature = "rocm")]
    run.forget_fp8_rows();
    let serial_capture = crate::serve::diag::on().capture_serial;
    let mut cursor = match (streams, at.lanes) {
        (Streams::Forked | Streams::Warm, Some(lanes)) if !serial_capture => {
            Cursor::across(place, lanes)
        }
        _ => at.serial(place),
    };
    if streams == Streams::Forked {
        cursor = cursor.writing();
        if let Some(cond) = at.conditionals {
            cursor = cursor.conditionals(cond);
        }
        if at.descriptor.gated.contains(&true) {
            cursor = cursor.marking(at.stream, &at.descriptor.gated);
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
    let settled = cursor.settle_marked();
    walked?;
    settled
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
    use poem_exec::fire::ClassWindow;

    const LANES: u32 = 4;

    fn prefill_only() -> poem_ir::ClassSet {
        poem_ir::ClassSet::default()
    }

    fn plain_tiers() -> Tiers {
        Tiers {
            plain: poem_ir::ClassSet::of(0..8usize),
            whole: Box::new([]),
        }
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

    #[test]
    fn record_every_case() {
        a_rung_is_the_keys_own_ceiling_and_arming_computes_the_same_one();
        what_a_load_has_spent_is_what_its_resident_bodies_weigh();
        the_map_never_inserts_a_body_whose_script_is_empty();
        a_sealed_decline_is_reported_once_a_shape_and_then_at_each_doubling();
        a_carve_reads_a_regions_ceiling_off_its_mask_whatever_is_present();
        a_composition_off_the_plain_tier_keys_the_whole_tier();
        a_wide_key_gates_its_partial_regions_and_no_other_key_gates();
    }

    fn a_carve_reads_a_regions_ceiling_off_its_mask_whatever_is_present() {
        // Three classes of the whole tier, laid out c2, c0, c1; the region runs c0 and c1.
        let ladder = Ladder(vec![(2, 4), (0, 8), (1, 8)].into_boxed_slice());
        let mask = poem_ir::ClassSet::of([0usize, 1]);
        let carve = |classes: &WindowTable| {
            let carve = AxisCarve {
                classes,
                ladder: &ladder,
                lane_ceiling: Some(LANES),
            };
            (
                carve.ceiling(MaskSpan::default(), Some(&mask)),
                carve.lanes(MaskSpan::default(), Some(&mask)),
            )
        };
        let every = table(&[(3, 3), (2, 2), (1, 1)]);
        let none = table(&[(0, 0), (0, 0), (5, 5)]);
        assert_eq!(carve(&every), (Some((4, 16)), Some((0, LANES))));
        assert_eq!(
            carve(&none),
            carve(&every),
            "a fire seating none of the region's classes launches it at the same ceiling"
        );
        let split = poem_ir::ClassSet::of([2usize, 1]);
        let apart = AxisCarve {
            classes: &none,
            ladder: &ladder,
            lane_ceiling: None,
        };
        assert_eq!(
            apart.ceiling(MaskSpan::default(), Some(&split)),
            Some((0, 20)),
            "a mask apart on the ladder launches over its hull, whatever is present"
        );
    }

    fn a_wide_key_gates_its_partial_regions_and_no_other_key_gates() {
        let tiers = Tiers {
            plain: poem_ir::ClassSet::of([0usize]),
            whole: Box::new([0, 1]),
        };
        let region = |mask: &[usize]| poem_compiler::Region {
            nodes: 0..1,
            mask: poem_ir::ClassSet::of(mask.iter().copied()),
            phase: poem_compiler::Phase::Capture,
            axis: None,
            lowering: poem_compiler::Lowering::AlwaysLaunch,
            stream: 0,
            wait: Vec::new(),
            open: None,
            close: None,
            collective: false,
        };
        // Every class, class 1 alone shifting, class 1 alone in place.
        let template = [region(&[0, 1]), region(&[1]), region(&[1])];
        let shifted = [true, true, false];
        let admits = [Admit::Captured; 3];
        let wide = Ladder(vec![(0, 8), (1, 8)].into_boxed_slice());
        assert_eq!(
            tiers.gates(&wide, &template, &shifted, &admits, &[]),
            vec![false, true, false],
            "a wide key gates each shifting region some of its classes run"
        );
        let plain = Ladder(vec![(0, 8)].into_boxed_slice());
        assert!(tiers.gates(&plain, &template, &shifted, &admits, &[]).is_empty());
        assert!(
            tiers
                .gates(&Ladder::default(), &template, &shifted, &admits, &[])
                .is_empty(),
            "an idle step (no lanes, no rungs) gates nothing, so launches nothing"
        );
    }

    fn a_composition_off_the_plain_tier_keys_the_whole_tier() {
        let tiers = Tiers {
            plain: poem_ir::ClassSet::of([0usize]),
            whole: Box::new([1, 0]),
        };
        let decoding = prefill_only();
        let key = |classes: &[(u32, u32)]| {
            BodyKey::of(&table(classes), 8, &decoding, LANES, &tiers).to_string()
        };
        assert_eq!(
            key(&[(8, 1)]),
            "b8[c0:8]",
            "a plain composition keys its own classes"
        );
        assert_eq!(key(&[(0, 0), (8, 1)]), "b8[c1:8 c0:8]");
        assert_eq!(
            key(&[(4, 1), (4, 1)]),
            key(&[(0, 0), (8, 1)]),
            "every subset holding a class off the plain tier keys one body"
        );
        assert_eq!(
            key(&[(4, 1), (2, 1), (2, 1)]),
            "b8[]",
            "a class outside the whole tier keys none"
        );
    }

    fn a_sealed_decline_is_reported_once_a_shape_and_then_at_each_doubling() {
        let mut seen = HashSet::new();
        let reported: Vec<u64> = (1..=40_000u64)
            .filter(|declines| {
                let key = if *declines == 1 {
                    "b256[c0:256]"
                } else {
                    "b1[c1:1]"
                };
                reports_decline(&mut seen, &key, *declines)
            })
            .collect();
        assert_eq!(
            reported,
            vec![
                1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
            ],
            "the first fire of each shape and each doubling of the count, no line per thousand"
        );
        assert!(
            reports_decline(&mut seen, &"b8[c1:8]", 40_001),
            "a shape not seen before is reported whatever the count"
        );
    }

    fn a_rung_is_the_keys_own_ceiling_and_arming_computes_the_same_one() {
        let decoding = poem_ir::ClassSet::of([0usize]);
        let fired = BodyKey::of(&table(&[(3, 3)]), 8, &decoding, LANES, &plain_tiers());
        assert_eq!(
            fired.to_string(),
            "b8[c0:4]",
            "the lane ceiling binds below the bucket, and three rows say nothing",
        );
        assert_eq!(
            BodyKey::of(&table(&[(3, 3)]), 2, &decoding, LANES, &plain_tiers()).to_string(),
            "b2[c0:2]",
            "and the bucket binds below the lane ceiling",
        );
        assert_eq!(
            BodyKey::of(&table(&[(3, 3)]), 8, &prefill_only(), LANES, &plain_tiers()).to_string(),
            "b8[c0:8]",
            "a class the decode arm does not name takes the bucket whole",
        );

        let armed = BodyKey {
            bucket: 8,
            classes: Ladder::single(0, Ladder::rung(0, 8, &decoding, LANES)),
            patch: None,
            islands: Box::default(),
        };
        assert_eq!(armed, fired, "the armed key must be the fired key");
    }

    fn rung(bucket: u32) -> BodyKey {
        BodyKey {
            bucket,
            classes: Ladder::single(0, bucket),
            patch: None,
            islands: Box::default(),
        }
    }

    #[test]
    fn a_key_carries_which_regions_are_islands() {
        let whole = rung(8);
        let same = rung(8).with_islands(&[Admit::Captured, Admit::Captured]);
        assert_eq!(whole, same, "no island is the plain key");
        assert_eq!(same.to_string(), "b8[c0:8]");
        let cut = rung(8).with_islands(&[Admit::Captured, Admit::Island, Admit::Captured]);
        assert_ne!(whole, cut, "a windowed fire is not the whole-window body");
        assert_eq!(cut.to_string(), "b8[c0:8]~i02");
        let mut ten = vec![Admit::Captured; 9];
        ten.push(Admit::Island);
        assert_eq!(rung(8).with_islands(&ten).to_string(), "b8[c0:8]~i0200");
    }

    fn weighing(bytes: usize) -> Body {
        Body {
            script: vec![Step::Island(Stretch {
                unit: 0,
                from: 0,
                upto: 1,
                island: true,
            })]
            .into_boxed_slice(),
            grids: Vec::new().into_boxed_slice(),
            shape: 0,
            launched_at: crate::settle::Airborne::NEVER,
            pinned: false,
            bytes: Some(bytes),
        }
    }

    fn what_a_load_has_spent_is_what_its_resident_bodies_weigh() {
        let mut graphs = Bodies::new();
        assert_eq!(
            graphs.body_stats().census.bytes,
            0,
            "an empty map has spent nothing"
        );
        for (bucket, bytes) in [(8u32, 3_000usize), (16, 5_000), (32, 7_000)] {
            assert!(graphs.insert_body(rung(bucket), weighing(bytes)));
        }
        assert_eq!(
            graphs.body_stats().census.bytes,
            15_000,
            "the census is the sum over the residents: {}",
            graphs.body_stats(),
        );
        assert!(graphs.insert_body(rung(16), weighing(6_000)));
        assert_eq!(
            graphs.body_stats().census.bytes,
            16_000,
            "a replacement charged twice: {}",
            graphs.body_stats(),
        );
        assert!(graphs.insert_body(rung(64), weighing(0)));
        assert_eq!(
            graphs.body_stats().census.bytes,
            16_000,
            "an unweighable body moved the budget: {}",
            graphs.body_stats(),
        );
        assert_eq!(
            graphs.body_stats().census.bodies,
            4,
            "and it still took a seat"
        );
    }

    fn the_map_never_inserts_a_body_whose_script_is_empty() {
        let mut graphs = Bodies::new();
        let classes = table(&[(8, 1)]);
        let key = BodyKey::of(&classes, 8, &prefill_only(), LANES, &plain_tiers());
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
