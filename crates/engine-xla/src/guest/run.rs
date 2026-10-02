//! Running lowered eta stages on the device.
//!
//! [`OnDevice`] is an `eta_exec::StageRunner`: `eta_exec::step_with` drives
//! one instance's pass and hands each stage here, which packs the stage's
//! roots into one upload, runs the stage's executable (the logits read
//! straight out of the fire's device readout) and unpacks the one download
//! of what the stage puts.
//!
//! [`step_group`] is the same pass for several instances of one program at
//! once: `step_with`'s control flow (readiness, the per-pass overlay, the
//! commit) per instance, and every stage one executable over the lanes.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use eta_compiler::codegen::launch::{LaunchPackage, ValueOrigin};
use eta_exec::{ExecPlan, InterpInstance, StagePlan, StageRunner, StepOutcome, Value};
use eta_ir::Dtype;
use eta_ir::op::IntrinsicId;
use eta_ir::registry::Stage;
use eta_ir::validate::Direction;

use super::lower::{self, Batch, Feed, Layout, Slot};
use crate::device::{Device, Program};
use crate::pjrt::{Arg, Buffer};
use crate::readout::{Kept, Seat};

/// A lowered stage, compiled; `program` is `None` for a stage that puts
/// nothing (there is nothing to run).
pub struct Staged {
    pub program: Option<Arc<Program>>,
    pub layout: Layout,
}

/// Compiled stages by (program content, stage, batch shape): a pass that
/// was seen skips the lowering and the compile both.
/// A compiled stage's key: program content, stage, batch shape, carried cells.
type StageKey = ([u8; 32], usize, Batch, Vec<u32>);

#[derive(Default)]
pub struct Stages {
    map: HashMap<StageKey, Arc<Staged>>,
    /// Stage executions and lanes they carried, for the tally.
    pub runs: u64,
    pub lanes: u64,
}

impl Stages {
    pub fn get(
        &mut self,
        device: &Device,
        package: &LaunchPackage,
        key: [u8; 32],
        at: usize,
        batch: Batch,
        carried: &[u32],
    ) -> Result<Arc<Staged>, String> {
        let slot = (key, at, batch, carried.to_vec());
        if let Some(hit) = self.map.get(&slot) {
            return Ok(Arc::clone(hit));
        }
        let lowered = lower::lower(package, at, batch, carried).map_err(|why| why.to_string())?;
        let program = if !lowered.layout.runs() {
            None
        } else {
            let sig = crate::trace::Signature {
                params: Vec::new(),
                results: Vec::new(),
            };
            Some(
                device
                    .program(&lowered.text, sig)
                    .map_err(|e| format!("stage {at} did not compile: {e}"))?,
            )
        };
        let staged = Arc::new(Staged {
            program,
            layout: lowered.layout,
        });
        self.map.insert(slot, Arc::clone(&staged));
        Ok(staged)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// The batch shape stage `at` runs at for `lanes` lanes against `kept`.
#[must_use]
pub fn batch_of(package: &LaunchPackage, at: usize, lanes: u32, kept: Option<&Kept>) -> Batch {
    let (logits, mtp) = lower::reads(package, at);
    Batch {
        lanes,
        logits: kept.filter(|_| logits).map(|k| (k.rows, k.width)),
        mtp: kept
            .filter(|_| mtp)
            .and_then(|k| k.mtp.as_ref().map(|(_, w)| (k.rows, *w))),
    }
}

/// Lanes padded to a power of two, so a group's size changes the
/// executable only at powers of two.
#[must_use]
pub fn padded(lanes: usize) -> u32 {
    u32::try_from(lanes.max(1).next_power_of_two()).unwrap_or(u32::MAX)
}

/// One lane's feed for a stage: its values (by package-global id; only
/// the stage's host-fed roots are read) and where its readout rows sit.
pub struct Feeding<'a> {
    pub vals: &'a [Value],
    pub seat: Option<Seat>,
}

fn encode(value: &Value, dtype: Dtype, n: usize, out: &mut Vec<u32>) -> Result<(), String> {
    if value.len() != n || value.dtype() != dtype {
        return Err(format!(
            "a root holds {} {:?} lane(s) where the stage reads {n} {dtype:?}",
            value.len(),
            value.dtype()
        ));
    }
    match value {
        Value::F32(v) => out.extend(v.iter().map(|x| x.to_bits())),
        Value::I32(v) => out.extend(v.iter().map(|&x| x as u32)),
        Value::U32(v) => out.extend_from_slice(v),
        Value::Bool(v) => {
            let base = out.len();
            out.resize(base + n.div_ceil(32), 0);
            for (j, &b) in v.iter().enumerate() {
                if b != 0 {
                    out[base + j / 32] |= 1 << (j % 32);
                }
            }
        }
    }
    Ok(())
}

fn decode(words: &[u32], dtype: Dtype, n: usize) -> Value {
    match dtype {
        Dtype::F32 => Value::F32(words[..n].iter().map(|&w| f32::from_bits(w)).collect()),
        Dtype::I32 => Value::I32(words[..n].iter().map(|&w| w as i32).collect()),
        Dtype::Bool => Value::Bool(
            (0..n)
                .map(|j| ((words[j / 32] >> (j % 32)) & 1) as u8)
                .collect(),
        ),
        _ => Value::U32(words[..n].to_vec()),
    }
}

/// What one stage run answers: per lane, the values it puts (host side),
/// and per carried channel the device cells of every lane (`u32 [B, words]`).
pub type Ran = (Vec<Vec<(u32, Value)>>, Vec<(u32, Buffer)>);

fn upload_words(
    device: &Device,
    words: &[u32],
    rows: usize,
    width: usize,
) -> Result<Buffer, String> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    device
        .upload(Dtype::U32, rows as u32, width as u32, &bytes)
        .map_err(|e| e.to_string())
}

/// A stage launched on the device and not read back: its packed output
/// (`u32 [batch, out_words]`, `None` when it puts nothing to the host) and
/// the carried cells it returned.
pub struct Launched {
    pub out: Option<Arc<Buffer>>,
    pub carried: Vec<(u32, Buffer)>,
    pub batch: usize,
    pub lanes: usize,
    /// Host words packed and uploaded, for the timing line.
    pub packed: usize,
}

/// Packs `lanes`' roots (padded to the staged batch by repeating lane 0),
/// uploads them and enqueues the stage; nothing waits for the device. A
/// carried channel's cells come from `cells` when it holds them (the
/// previous pass's, same lanes in the same order), else from the lanes'
/// values.
pub fn launch_stage(
    device: &Device,
    staged: &Staged,
    kept: Option<&Kept>,
    lanes: &[Feeding<'_>],
    cells: Option<&BTreeMap<u32, Buffer>>,
) -> Result<Launched, String> {
    let batch = padded(lanes.len()) as usize;
    let Some(program) = &staged.program else {
        return Ok(Launched {
            out: None,
            carried: Vec::new(),
            batch,
            lanes: lanes.len(),
            packed: 0,
        });
    };
    let layout = &staged.layout;
    let mut words: Vec<u32> = Vec::with_capacity(batch * layout.in_words);
    for b in 0..batch {
        let lane = &lanes[if b < lanes.len() { b } else { 0 }];
        for slot in &layout.inputs {
            match slot.feed {
                Feed::Value(dtype, n) => {
                    encode(&lane.vals[slot.value as usize], dtype, n, &mut words)?;
                }
                Feed::Row { intrinsic, rows } => {
                    let seat = lane.seat.ok_or_else(|| {
                        format!(
                            "this pass reads the `{}` intrinsic and its lane offered no \
                             readout rows",
                            intrinsic.name()
                        )
                    })?;
                    if rows > seat.count {
                        return Err(
                            "logits intrinsic row range exceeds the forward's readout rows"
                                .to_owned(),
                        );
                    }
                    words.push(seat.first);
                }
            }
        }
    }
    let pack: Option<Buffer> = if layout.in_words > 0 {
        Some(upload_words(device, &words, batch, layout.in_words)?)
    } else {
        None
    };
    let mut fresh: Vec<Buffer> = Vec::new();
    for (chan, slot) in &layout.carried_in {
        if cells.is_some_and(|c| c.contains_key(chan)) {
            continue;
        }
        let Feed::Value(dtype, n) = slot.feed else {
            unreachable!("carried cells are values");
        };
        let mut cell: Vec<u32> = Vec::with_capacity(batch * slot.words);
        for b in 0..batch {
            let lane = &lanes[if b < lanes.len() { b } else { 0 }];
            encode(&lane.vals[slot.value as usize], dtype, n, &mut cell)?;
        }
        fresh.push(upload_words(device, &cell, batch, slot.words)?);
    }
    let mut args: Vec<Arg<'_>> = Vec::with_capacity(3);
    if let Some(pack) = &pack {
        args.push(Arg::Keep(pack));
    }
    if layout.logits {
        let kept = kept.ok_or("this stage reads the logits and the fire kept no readout")?;
        args.push(Arg::Keep(&kept.logits));
    }
    if layout.mtp {
        let (mtp, _) = kept
            .and_then(|k| k.mtp.as_ref())
            .ok_or("this stage reads the draft head and the fire kept none")?;
        args.push(Arg::Keep(mtp));
    }
    let mut fresh = fresh.iter();
    for (chan, _) in &layout.carried_in {
        match cells.and_then(|c| c.get(chan)) {
            Some(cell) => args.push(Arg::Keep(cell)),
            None => args.push(Arg::Keep(
                fresh.next().expect("a fresh cell per carried root"),
            )),
        }
    }
    let outs = device
        .run(program, args, crate::serve::timing())
        .map_err(|e| e.to_string())?;
    let mut outs = outs.into_iter();
    let out = if layout.out_words > 0 {
        Some(Arc::new(outs.next().ok_or("the stage returned nothing")?))
    } else {
        None
    };
    let carried: Vec<(u32, Buffer)> = layout
        .carried_out
        .iter()
        .map(|(chan, _)| outs.next().map(|b| (*chan, b)))
        .collect::<Option<_>>()
        .ok_or("the stage returned fewer carried cells than it puts")?;
    Ok(Launched {
        out,
        carried,
        batch,
        lanes: lanes.len(),
        packed: words.len(),
    })
}

/// The output words of a stage's packed download.
#[must_use]
pub fn words_of(raw: &[u8]) -> Vec<u32> {
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// Lane `row`'s value in `slot` of a stage's output words.
pub fn cell_of(all: &[u32], out_words: usize, row: usize, slot: &Slot) -> Result<Value, String> {
    let Feed::Value(dtype, n) = slot.feed else {
        return Err("a stage output slot holds no value".to_owned());
    };
    let from = row * out_words + slot.at;
    let words = all
        .get(from..from + slot.words)
        .ok_or_else(|| format!("output row {row} is past the {} words returned", all.len()))?;
    Ok(decode(words, dtype, n))
}

/// Every lane's put values out of a stage's download.
pub fn unpack(
    layout: &Layout,
    raw: &[u8],
    lanes: usize,
    batch: usize,
) -> Result<Vec<Vec<(u32, Value)>>, String> {
    let all = words_of(raw);
    if all.len() != batch * layout.out_words {
        return Err(format!(
            "the stage returned {} words for {batch} lane(s) of {}",
            all.len(),
            layout.out_words
        ));
    }
    (0..lanes)
        .map(|b| {
            layout
                .outputs
                .iter()
                .map(|slot| Ok((slot.value, cell_of(&all, layout.out_words, b, slot)?)))
                .collect()
        })
        .collect()
}

/// Runs one staged stage over `lanes` and reads its outputs back
/// (`launch_stage`, then the download).
pub fn run_stage(
    device: &Device,
    staged: &Staged,
    kept: Option<&Kept>,
    lanes: &[Feeding<'_>],
    cells: Option<&BTreeMap<u32, Buffer>>,
) -> Result<Ran, String> {
    let t0 = std::time::Instant::now();
    let launched = launch_stage(device, staged, kept, lanes, cells)?;
    let t1 = std::time::Instant::now();
    let raw = match &launched.out {
        Some(out) => out.download().map_err(|e| e.to_string())?,
        None => Vec::new(),
    };
    if crate::serve::timing() && staged.program.is_some() {
        eprintln!(
            "  xla guest stage: {} lane(s) in {}: pack+upload+run {:.2}ms ({} KiB) download {:.2}ms ({} KiB)",
            lanes.len(),
            launched.batch,
            (t1 - t0).as_secs_f64() * 1e3,
            launched.packed >> 8,
            t1.elapsed().as_secs_f64() * 1e3,
            raw.len() >> 10
        );
    }
    if staged.program.is_none() {
        return Ok((vec![Vec::new(); lanes.len()], Vec::new()));
    }
    let values = unpack(&staged.layout, &raw, lanes.len(), launched.batch)?;
    Ok((values, launched.carried))
}

/// Whether the device binds `id` for a lane seated at `seat` in `kept`.
fn device_binds(id: Option<IntrinsicId>, kept: Option<&Kept>, seat: Option<Seat>) -> bool {
    match id {
        Some(IntrinsicId::Logits) => kept.is_some_and(|k| k.f32) && seat.is_some(),
        Some(IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts) => {
            kept.is_some_and(|k| k.f32 && k.mtp.is_some()) && seat.is_some()
        }
        _ => false,
    }
}

/// One instance's stages on the device, as `step_with`'s runner.
pub struct OnDevice<'a> {
    device: &'a Device,
    stages: &'a mut Stages,
    key: [u8; 32],
    kept: Option<&'a Kept>,
    seat: Option<Seat>,
    ran: usize,
}

impl<'a> OnDevice<'a> {
    pub fn new(
        device: &'a Device,
        stages: &'a mut Stages,
        key: [u8; 32],
        kept: Option<&'a Kept>,
        seat: Option<Seat>,
    ) -> OnDevice<'a> {
        OnDevice {
            device,
            stages,
            key,
            kept,
            seat,
            ran: 0,
        }
    }

    /// Stages this runner executed.
    #[must_use]
    pub fn ran(&self) -> usize {
        self.ran
    }
}

impl StageRunner for OnDevice<'_> {
    fn run(
        &mut self,
        plan: &ExecPlan,
        sp: &StagePlan,
        _roots: &[u32],
        _wanted: &[u32],
        vals: &mut [Value],
    ) -> eta_exec::Result<()> {
        let at = sp.stage_index;
        let fail = |message: String| eta_exec::Error { message };
        let batch = batch_of(&plan.package, at, 1, self.seat.and(self.kept));
        let staged = self
            .stages
            .get(self.device, &plan.package, self.key, at, batch, &[])
            .map_err(fail)?;
        let (outs, _) = run_stage(
            self.device,
            &staged,
            self.kept,
            &[Feeding {
                vals,
                seat: self.seat,
            }],
            None,
        )
        .map_err(fail)?;
        self.stages.runs += 1;
        self.stages.lanes += 1;
        for (id, value) in outs.into_iter().next().unwrap_or_default() {
            vals[id as usize] = value;
        }
        self.ran += 1;
        Ok(())
    }

    fn binds(&self, id: Option<IntrinsicId>) -> bool {
        device_binds(id, self.kept, self.seat)
    }
}

/// Whether every intrinsic `plan` reads binds on the device for lanes that
/// have readout seats in `kept` (so a group needs no host-side binding).
#[must_use]
pub fn groups(plan: &ExecPlan, kept: Option<&Kept>) -> bool {
    plan.package.values.iter().all(|v| {
        v.source != ValueOrigin::Intrinsic
            || device_binds(v.intrinsic, kept, Some(Seat { first: 0, count: 1 }))
    })
}

/// One instance of a group.
pub struct Member<'a> {
    pub inst: &'a mut InterpInstance,
    pub seat: Option<Seat>,
}

/// A cell a pass puts: known on the host, or lane `row`'s `slot` of the
/// last stage's output, still on its way back from the device.
#[derive(Clone)]
enum Cell {
    Known(Value),
    Out { row: usize, slot: Slot },
}

struct Overlay {
    pending: BTreeMap<u32, Cell>,
    taken: Vec<bool>,
    put: Vec<bool>,
}

impl Overlay {
    fn new(channels: usize) -> Overlay {
        Overlay {
            pending: BTreeMap::new(),
            taken: vec![false; channels],
            put: vec![false; channels],
        }
    }

    fn resolve(&self, inst: &InterpInstance, chan: u32) -> Value {
        if let Some(Cell::Known(v)) = self.pending.get(&chan) {
            return v.clone();
        }
        inst.channels[chan as usize].current()
    }

    fn take(&mut self, inst: &InterpInstance, chan: u32) -> Value {
        self.taken[chan as usize] = true;
        self.resolve(inst, chan)
    }
}

/// A put still on the device: lane `row`'s `slot` of the pass's last stage
/// output, to land in `ring` at sequence `seq`. The ring's tail is stored
/// only once the cell is written, so no reader (the runtime reads a host
/// ring's mirror directly) sees it early.
pub struct DeferredPut {
    pub ring: Arc<eta_exec::ChannelState>,
    pub seq: u64,
    pub row: usize,
    pub slot: Slot,
}

/// What a deferred group pass left on the device: its last stage's packed
/// output and the puts that read it, per member (`members[i]` is the
/// member index of `puts[i]`).
pub struct Deferred {
    pub out: Arc<Buffer>,
    pub out_words: usize,
    pub batch: usize,
    pub puts: Vec<DeferredPut>,
    pub members: Vec<usize>,
}

fn const_root(root: &eta_compiler::codegen::launch::LaunchValue) -> Value {
    match root.dtype {
        Dtype::I32 => Value::I32(vec![root.literal_bits as i32]),
        Dtype::U32 => Value::U32(vec![root.literal_bits]),
        Dtype::Bool => Value::Bool(vec![u8::from(root.literal_bits != 0)]),
        _ => Value::F32(vec![f32::from_bits(root.literal_bits)]),
    }
}

/// Writes carried cells (`u32 [B, words]` per channel, row `r` is
/// `insts[r]`'s) back into the instances' rings, where a pass that kept
/// them on the device left the ring's front stale.
pub fn flush(
    plan: &ExecPlan,
    cells: &BTreeMap<u32, Buffer>,
    insts: &[Option<&InterpInstance>],
) -> Result<(), String> {
    for (&chan, cell) in cells {
        let decl = plan
            .package
            .channels
            .get(chan as usize)
            .ok_or_else(|| format!("carried channel {chan} is past the program's"))?;
        let dtype = eta_exec::concrete_dtype(decl.dtype);
        let n = decl
            .shape
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        let w = lower::words(dtype, n);
        let raw = cell.download().map_err(|e| e.to_string())?;
        let all: Vec<u32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        for (r, inst) in insts.iter().enumerate() {
            let (Some(inst), Some(row)) = (inst, all.get(r * w..(r + 1) * w)) else {
                continue;
            };
            if let Some(ring) = inst.channels.get(chan as usize) {
                ring.encode_sequence(ring.head(), &decode(row, dtype, n));
            }
        }
    }
    Ok(())
}

/// `eta_exec::step_with` for every member at once: the same readiness
/// check, overlay and commit per instance, and each stage one executable
/// over the members. Members must all run `plan` and every intrinsic it
/// reads must bind on the device (`groups`).
///
/// The `carried` channels' cells stay on the device: `cells` are the last
/// pass's (row `r` is member `r`'s) and the pass's come back when every
/// member committed. Otherwise the cells are written into the rings.
#[allow(clippy::too_many_arguments)]
pub fn step_group(
    device: &Device,
    stages: &mut Stages,
    key: [u8; 32],
    plan: &ExecPlan,
    kept: Option<&Kept>,
    members: &mut [Member<'_>],
    carried: &[u32],
    cells: Option<BTreeMap<u32, Buffer>>,
) -> (Vec<StepOutcome>, Option<BTreeMap<u32, Buffer>>) {
    let (outcomes, cells, deferred) = step_group_with(
        device, stages, key, plan, kept, members, carried, cells, false,
    );
    debug_assert!(deferred.is_none(), "a synchronous pass defers nothing");
    (outcomes, cells)
}

/// `step_group` whose last stage is left running: its output comes back
/// later (`Deferred`), and the puts that read it are committed without
/// their cells, whose ring tails wait for them. Every other part of the
/// commit (heads, host-known puts) is done here.
#[allow(clippy::too_many_arguments)]
pub fn step_group_deferred(
    device: &Device,
    stages: &mut Stages,
    key: [u8; 32],
    plan: &ExecPlan,
    kept: Option<&Kept>,
    members: &mut [Member<'_>],
    carried: &[u32],
    cells: Option<BTreeMap<u32, Buffer>>,
) -> (
    Vec<StepOutcome>,
    Option<BTreeMap<u32, Buffer>>,
    Option<Deferred>,
) {
    step_group_with(
        device, stages, key, plan, kept, members, carried, cells, true,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn step_group_with(
    device: &Device,
    stages: &mut Stages,
    key: [u8; 32],
    plan: &ExecPlan,
    kept: Option<&Kept>,
    members: &mut [Member<'_>],
    carried: &[u32],
    cells: Option<BTreeMap<u32, Buffer>>,
    defer: bool,
) -> (
    Vec<StepOutcome>,
    Option<BTreeMap<u32, Buffer>>,
    Option<Deferred>,
) {
    let mut outcome: Vec<Option<StepOutcome>> = vec![None; members.len()];
    for (m, member) in members.iter().enumerate() {
        if member.inst.poisoned {
            outcome[m] = Some(StepOutcome::Faulted("instance is poisoned".to_string()));
            continue;
        }
        for (channel, ring) in member.inst.channels.iter().enumerate() {
            let readiness = plan.package.channels.get(channel).and_then(|c| c.readiness);
            let ready = match readiness {
                Some(Direction::NeedsFull) => !ring.is_empty(),
                Some(Direction::NeedsEmpty) => !ring.is_full(),
                None => true,
            };
            if !ready {
                outcome[m] = Some(StepOutcome::Blocked(channel as u32));
                break;
            }
        }
    }
    let mut cells = cells;
    if cells.is_some() && outcome.iter().any(Option::is_some) {
        // Not every member runs: the rows no longer line up. Back to the rings.
        let insts: Vec<Option<&InterpInstance>> = members.iter().map(|m| Some(&*m.inst)).collect();
        if let Some(c) = cells.take()
            && let Err(why) = flush(plan, &c, &insts)
        {
            for o in &mut outcome {
                o.get_or_insert(StepOutcome::Faulted(why.clone()));
            }
        }
    }
    let mut produced: BTreeMap<u32, Buffer> = BTreeMap::new();
    let mut overlays: Vec<Overlay> = members
        .iter()
        .map(|member| Overlay::new(member.inst.channels.len()))
        .collect();
    let mut vals: Vec<Vec<Value>> = members
        .iter()
        .map(|_| vec![Value::F32(Vec::new()); plan.package.values.len()])
        .collect();

    let kinds = [Stage::Prologue, Stage::Epilogue];
    // The stage that runs last: the one whose output may stay in flight.
    let last = kinds
        .iter()
        .flat_map(|kind| {
            plan.stages
                .iter()
                .filter(move |sp| plan.package.stages[sp.stage_index].stage == *kind)
        })
        .last()
        .map(|sp| sp.stage_index);
    let mut launched_out: Option<(Arc<Buffer>, usize, usize)> = None;
    for (k, kind) in kinds.iter().enumerate() {
        if k == 1 {
            for (m, member) in members.iter().enumerate() {
                if outcome[m].is_some() {
                    continue;
                }
                for port in &plan.package.ports {
                    if !port.is_const && port.port.consumes() {
                        let _ = overlays[m].take(member.inst, port.channel);
                    }
                }
            }
        }
        for sp in &plan.stages {
            if plan.package.stages[sp.stage_index].stage != *kind {
                continue;
            }
            let live: Vec<usize> = (0..members.len())
                .filter(|&m| outcome[m].is_none())
                .collect();
            if live.is_empty() {
                break;
            }
            for &m in &live {
                if let Err(why) =
                    bind_roots(plan, sp, members[m].inst, &mut overlays[m], &mut vals[m])
                {
                    outcome[m] = Some(StepOutcome::Faulted(why));
                }
            }
            let live: Vec<usize> = live.into_iter().filter(|&m| outcome[m].is_none()).collect();
            if live.is_empty() {
                continue;
            }
            let at = sp.stage_index;
            let batch = batch_of(&plan.package, at, padded(live.len()), kept);
            let staged = stages.get(device, &plan.package, key, at, batch, carried);
            let leave = defer
                && last == Some(at)
                && staged
                    .as_ref()
                    .is_ok_and(|s| s.program.is_some() && s.layout.out_words > 0);
            let ran = staged.and_then(|staged| {
                let feeds: Vec<Feeding<'_>> = live
                    .iter()
                    .map(|&m| Feeding {
                        vals: &vals[m],
                        seat: members[m].seat,
                    })
                    .collect();
                if leave {
                    let launched = launch_stage(device, &staged, kept, &feeds, cells.as_ref())?;
                    let out = launched.out.ok_or("a deferred stage returned no output")?;
                    launched_out = Some((out, staged.layout.out_words, launched.batch));
                    Ok((None, launched.carried, Some(staged)))
                } else {
                    run_stage(device, &staged, kept, &feeds, cells.as_ref())
                        .map(|(outs, carry)| (Some(outs), carry, None))
                }
            });
            match ran {
                Ok((outs, carry, left)) => {
                    produced.extend(carry);
                    stages.runs += 1;
                    stages.lanes += live.len() as u64;
                    let stage = &plan.package.stages[at];
                    match outs {
                        Some(outs) => {
                            for (&m, out) in live.iter().zip(outs) {
                                for (id, value) in out {
                                    vals[m][id as usize] = value;
                                }
                                for put in &stage.puts {
                                    if !carried.contains(&put.channel) {
                                        overlays[m].pending.insert(
                                            put.channel,
                                            Cell::Known(vals[m][put.value as usize].clone()),
                                        );
                                    }
                                    overlays[m].put[put.channel as usize] = true;
                                }
                            }
                        }
                        None => {
                            let layout = &left.expect("a deferred stage is kept").layout;
                            for (row, &m) in live.iter().enumerate() {
                                for put in &stage.puts {
                                    if !carried.contains(&put.channel) {
                                        let cell = match layout
                                            .outputs
                                            .iter()
                                            .find(|slot| slot.value == put.value)
                                        {
                                            Some(slot) => Cell::Out { row, slot: *slot },
                                            None => {
                                                Cell::Known(vals[m][put.value as usize].clone())
                                            }
                                        };
                                        overlays[m].pending.insert(put.channel, cell);
                                    }
                                    overlays[m].put[put.channel as usize] = true;
                                }
                            }
                        }
                    }
                }
                Err(why) => {
                    for &m in &live {
                        outcome[m] = Some(StepOutcome::Faulted(why.clone()));
                    }
                }
            }
        }
    }

    let mut puts: Vec<DeferredPut> = Vec::new();
    let mut put_members: Vec<usize> = Vec::new();
    let outcomes: Vec<StepOutcome> = members
        .iter_mut()
        .zip(outcome)
        .zip(&overlays)
        .enumerate()
        .map(|(m, ((member, outcome), overlay))| match outcome {
            Some(StepOutcome::Faulted(why)) => {
                member.inst.poisoned = true;
                StepOutcome::Faulted(why)
            }
            Some(other) => other,
            None => {
                let (outcome, left) = commit(member.inst, overlay);
                for (chan, seq, row, slot) in left {
                    puts.push(DeferredPut {
                        ring: Arc::clone(&member.inst.channels[chan]),
                        seq,
                        row,
                        slot,
                    });
                    put_members.push(m);
                }
                outcome
            }
        })
        .collect();
    let deferred = match launched_out {
        Some((out, out_words, batch)) if !puts.is_empty() => Some(Deferred {
            out,
            out_words,
            batch,
            puts,
            members: put_members,
        }),
        _ => None,
    };
    if produced.is_empty() {
        return (outcomes, None, deferred);
    }
    if outcomes.iter().all(|o| *o == StepOutcome::Committed) {
        return (outcomes, Some(produced), deferred);
    }
    // Rows are live members in order; write the committed ones back.
    let insts: Vec<Option<&InterpInstance>> = members
        .iter()
        .zip(&outcomes)
        .map(|(m, o)| (*o == StepOutcome::Committed).then_some(&*m.inst))
        .collect();
    let _ = flush(plan, &produced, &insts);
    (outcomes, None, deferred)
}

fn bind_roots(
    plan: &ExecPlan,
    sp: &StagePlan,
    inst: &InterpInstance,
    overlay: &mut Overlay,
    vals: &mut [Value],
) -> Result<(), String> {
    for &id in &sp.value_ids {
        if sp.op_by_result.contains_key(&id) {
            continue;
        }
        let root = &plan.package.values[id as usize];
        let cell = match root.source {
            ValueOrigin::Const => const_root(root),
            ValueOrigin::ChannelTake => overlay.take(inst, root.channel),
            ValueOrigin::ChannelRead => overlay.resolve(inst, root.channel),
            ValueOrigin::Intrinsic => Value::F32(Vec::new()),
            ValueOrigin::OpResult => {
                return Err(
                    "unresolved value root (intrinsic/host input) reached execution".to_owned(),
                );
            }
        };
        vals[id as usize] = cell;
    }
    Ok(())
}

/// Commits a pass's overlay into the instance's rings. A put whose cell is
/// still on the device (`Cell::Out`) is left out of its ring's tail and
/// answered as (channel, sequence, row, slot) for whoever lands it.
#[allow(clippy::type_complexity)]
fn commit(
    inst: &mut InterpInstance,
    overlay: &Overlay,
) -> (StepOutcome, Vec<(usize, u64, usize, Slot)>) {
    let n = inst.channels.len();
    let mut old_tails = vec![0u64; n];
    let mut new_heads = vec![0u64; n];
    let mut new_tails = vec![0u64; n];

    for (ci, ring) in inst.channels.iter().enumerate() {
        let head = ring.head();
        let tail = ring.tail();
        if tail < head {
            inst.poisoned = true;
            return (
                StepOutcome::Faulted(format!("channel {ci}: tail precedes head at commit")),
                Vec::new(),
            );
        }
        let mut next_head = head;
        let mut next_tail = tail;
        let mut used = tail - head;
        if overlay.taken[ci] && used != 0 {
            next_head += 1;
            used -= 1;
        }
        if overlay.put[ci] {
            if used >= ring.capacity() as u64 {
                inst.poisoned = true;
                return (
                    StepOutcome::Faulted(format!(
                        "channel {ci}: put overflows capacity {} at commit",
                        ring.capacity()
                    )),
                    Vec::new(),
                );
            }
            next_tail += 1;
        }
        old_tails[ci] = tail;
        new_heads[ci] = next_head;
        new_tails[ci] = next_tail;
    }

    let mut left = Vec::new();
    for (ci, ring) in inst.channels.iter().enumerate() {
        // A carried cell has no host value: it stays on the device.
        if !overlay.put[ci] {
            continue;
        }
        match overlay.pending.get(&(ci as u32)) {
            Some(Cell::Known(value)) => ring.encode_sequence(old_tails[ci], value),
            Some(Cell::Out { row, slot }) => {
                left.push((ci, old_tails[ci], *row, *slot));
                new_tails[ci] = old_tails[ci];
            }
            None => {}
        }
    }
    for (ci, ring) in inst.channels.iter().enumerate() {
        if new_heads[ci] != ring.head() {
            ring.store_head(new_heads[ci]);
        }
        if new_tails[ci] != ring.tail() {
            ring.store_tail(new_tails[ci]);
        }
    }
    (StepOutcome::Committed, left)
}
