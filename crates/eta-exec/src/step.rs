use std::collections::BTreeMap;

use eta_ir::registry::Stage;
use eta_ir::validate::Direction;

use eta_compiler::codegen::launch::{LaunchValue, ValueOrigin};

use super::channel::InterpInstance;
use super::op::eval_op;
use super::plan::{ExecPlan, StagePlan, port_consumes};
use super::value::Value;
use crate::{Error, Result, shape_numel};

#[derive(Clone, Copy, Debug)]
pub struct PassInputs<'a> {
    /// What each output reads as, by its name and the lane reading it.
    pub outputs: &'a [(String, u32, Value)],
}

impl PassInputs<'_> {
    #[must_use]
    pub fn none() -> Self {
        PassInputs { outputs: &[] }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepOutcome {
    Committed,

    Blocked(u32),

    Faulted(String),
}

struct Overlay {
    pending: BTreeMap<u32, Value>,

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
        if let Some(v) = self.pending.get(&chan) {
            return v.clone();
        }
        inst.channels[chan as usize].current()
    }

    fn take(&mut self, inst: &InterpInstance, chan: u32) -> Value {
        self.taken[chan as usize] = true;
        self.resolve(inst, chan)
    }
}

/// The first `numel` values lane `lane` reads of output `name`.
fn bind_output(root: &LaunchValue, inputs: &PassInputs) -> Result<Value> {
    let name = root.output.as_deref().unwrap_or("");
    let unbound = |why: &str| Error {
        message: format!("output `{name}` of lane {}: {why}", root.lane),
    };
    let (_, _, value) = inputs
        .outputs
        .iter()
        .find(|(named, lane, _)| named == name && *lane == root.lane)
        .ok_or_else(|| unbound("unbound"))?;
    let want = shape_numel(&root.shape).max(1) as usize;
    match value {
        Value::F32(values) if values.len() >= want => Ok(Value::F32(values[..want].to_vec())),
        Value::I32(values) if values.len() >= want => Ok(Value::I32(values[..want].to_vec())),
        _ => Err(unbound("holds fewer values than the program reads")),
    }
}

pub trait StageRunner {
    fn run(
        &mut self,
        plan: &ExecPlan,
        sp: &StagePlan,
        roots: &[u32],
        wanted: &[u32],
        vals: &mut [Value],
    ) -> Result<()>;

    /// Whether the runner binds the output `root` reads itself.
    fn binds(&self, _root: &LaunchValue) -> bool {
        false
    }

    /// Whether the runner keeps `channel`'s cell itself between fires: a
    /// take of it hands the runner an empty value rather than the ring's
    /// cell, and a put of an empty value leaves the ring's bytes as they are.
    fn keeps(&self, _channel: u32) -> bool {
        false
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Interpreted;

impl StageRunner for Interpreted {
    fn run(
        &mut self,
        plan: &ExecPlan,
        sp: &StagePlan,
        _roots: &[u32],
        _wanted: &[u32],
        vals: &mut [Value],
    ) -> Result<()> {
        let stage = &plan.package.stages[sp.stage_index];
        for &id in &sp.value_ids {
            if let Some(&op_idx) = sp.op_by_result.get(&id) {
                eval_op(&stage.ops[op_idx], &plan.package, vals)?;
            }
        }
        Ok(())
    }
}

fn exec_stage(
    inst: &InterpInstance,
    plan: &ExecPlan,
    sp: &StagePlan,
    inputs: &PassInputs,
    overlay: &mut Overlay,
    vals: &mut [Value],
    runner: &mut dyn StageRunner,
) -> Result<()> {
    let roots = prepare_stage(
        inst,
        plan,
        sp,
        inputs,
        overlay,
        vals,
        &|root| match root.source {
            ValueOrigin::ChannelTake => runner.keeps(root.channel),
            ValueOrigin::Output => runner.binds(root),
            _ => false,
        },
    )?;
    let wanted = wanted_of(plan, sp);
    runner.run(plan, sp, &roots, &wanted, vals)?;
    finish_stage(plan, sp, overlay, vals);
    Ok(())
}

/// Resolves the stage's roots into `vals` (constants, channel reads and
/// takes through the overlay, intrinsics from `inputs`; a root the runner
/// `held` itself is left empty) and returns their ids.
fn prepare_stage(
    inst: &InterpInstance,
    plan: &ExecPlan,
    sp: &StagePlan,
    inputs: &PassInputs,
    overlay: &mut Overlay,
    vals: &mut [Value],
    held: &dyn Fn(&LaunchValue) -> bool,
) -> Result<Vec<u32>> {
    let mut roots = Vec::new();
    for &id in &sp.value_ids {
        if sp.op_by_result.contains_key(&id) {
            continue;
        }
        let root = &plan.package.values[id as usize];
        let cell = match root.source {
            ValueOrigin::Const => const_root_value(root),
            ValueOrigin::ChannelTake if held(root) => {
                overlay.taken[root.channel as usize] = true;
                Value::F32(Vec::new())
            }
            ValueOrigin::ChannelTake => overlay.take(inst, root.channel),
            ValueOrigin::ChannelRead => overlay.resolve(inst, root.channel),
            ValueOrigin::Output if held(root) => Value::F32(Vec::new()),
            ValueOrigin::Output => bind_output(root, inputs)?,
            ValueOrigin::OpResult => {
                return Err(Error {
                    message: "unresolved value root (output/host input) reached execution"
                        .to_owned(),
                });
            }
        };
        vals[id as usize] = cell;
        roots.push(id);
    }
    Ok(roots)
}

fn wanted_of(plan: &ExecPlan, sp: &StagePlan) -> Vec<u32> {
    plan.package.stages[sp.stage_index]
        .puts
        .iter()
        .map(|put| put.value)
        .collect()
}

/// Records the stage's puts in the overlay (they land at commit).
fn finish_stage(plan: &ExecPlan, sp: &StagePlan, overlay: &mut Overlay, vals: &[Value]) {
    for put in &plan.package.stages[sp.stage_index].puts {
        overlay
            .pending
            .insert(put.channel, vals[put.value as usize].clone());
        overlay.put[put.channel as usize] = true;
    }
}

/// Runs the stages of several instances of one program in lockstep: every
/// stage once, for every lane that is still stepping (a device program
/// carrying the lanes as rows, say).
pub trait BatchRunner {
    /// Computes stage `sp` for `lanes`, the instances still stepping (each
    /// with its index in the batch and its values: roots resolved, puts to
    /// be written). An error faults every lane.
    fn run(&mut self, plan: &ExecPlan, sp: &StagePlan, lanes: &mut [Lane<'_>]) -> Result<()>;

    /// Whether the runner binds the output `root` reads for lane `lane` itself.
    fn binds(&self, _lane: usize, _root: &LaunchValue) -> bool {
        false
    }
}

/// One lane of a `BatchRunner`'s stage: the instance's index in the batch
/// and its values.
pub struct Lane<'v> {
    pub index: usize,
    pub vals: &'v mut [Value],
}

/// One pass of each of `insts` (instances of `plan`) with the stages run in
/// lockstep by `runner`: lane `l` of a stage's batch is `insts[l]` while
/// that instance is still stepping. A blocked or poisoned instance leaves
/// the batch before its first stage, a faulted one when it faults; the
/// others step on. Returns each instance's outcome, in order.
#[must_use]
pub fn step_many(
    insts: &mut [&mut InterpInstance],
    plan: &ExecPlan,
    inputs: &PassInputs,
    runner: &mut dyn BatchRunner,
    at: Stage,
) -> Vec<StepOutcome> {
    let n = insts.len();
    let mut outcomes: Vec<Option<StepOutcome>> = vec![None; n];
    for (l, inst) in insts.iter().enumerate() {
        if inst.poisoned {
            outcomes[l] = Some(StepOutcome::Faulted("instance is poisoned".to_string()));
            continue;
        }
        if let Some(channel) = blocked(inst, plan, at) {
            outcomes[l] = Some(StepOutcome::Blocked(channel));
        }
    }
    let mut overlays: Vec<Overlay> = insts
        .iter()
        .map(|inst| Overlay::new(inst.channels.len()))
        .collect();
    let mut vals: Vec<Vec<Value>> = (0..n)
        .map(|_| vec![Value::F32(vec![]); plan.package.values.len()])
        .collect();
    fn fault(
        outcomes: &mut [Option<StepOutcome>],
        insts: &mut [&mut InterpInstance],
        l: usize,
        why: String,
    ) {
        insts[l].poisoned = true;
        outcomes[l] = Some(StepOutcome::Faulted(why));
    }
    for kind in [Stage::Prologue, Stage::Epilogue] {
        if kind == Stage::Epilogue && at != Stage::Prologue {
            for (l, inst) in insts.iter().enumerate() {
                if outcomes[l].is_some() {
                    continue;
                }
                for port in &plan.package.ports {
                    if port.is_const {
                        continue;
                    }
                    if port_consumes(port.port) {
                        let _ = overlays[l].take(inst, port.channel);
                    }
                }
            }
        }
        for sp in &plan.stages {
            if plan.package.stages[sp.stage_index].stage != kind
                || !plan.runs_at(sp.stage_index, at)
            {
                continue;
            }
            for l in 0..n {
                if outcomes[l].is_some() {
                    continue;
                }
                let prepared = prepare_stage(
                    insts[l],
                    plan,
                    sp,
                    inputs,
                    &mut overlays[l],
                    &mut vals[l],
                    &|root| matches!(root.source, ValueOrigin::Output) && runner.binds(l, root),
                );
                if let Err(why) = prepared {
                    fault(&mut outcomes, insts, l, why.to_string());
                }
            }
            let mut lanes: Vec<Lane<'_>> = vals
                .iter_mut()
                .enumerate()
                .filter(|(l, _)| outcomes[*l].is_none())
                .map(|(index, v)| Lane {
                    index,
                    vals: v.as_mut_slice(),
                })
                .collect();
            if lanes.is_empty() {
                break;
            }
            if let Err(why) = runner.run(plan, sp, &mut lanes) {
                for l in 0..n {
                    if outcomes[l].is_none() {
                        fault(&mut outcomes, insts, l, why.to_string());
                    }
                }
                break;
            }
            for l in 0..n {
                if outcomes[l].is_none() {
                    finish_stage(plan, sp, &mut overlays[l], &vals[l]);
                }
            }
        }
    }
    for l in 0..n {
        if outcomes[l].is_none() {
            outcomes[l] = Some(commit(insts[l], &overlays[l]));
        }
    }
    outcomes
        .into_iter()
        .map(|o| o.unwrap_or(StepOutcome::Committed))
        .collect()
}

fn const_root_value(root: &LaunchValue) -> Value {
    match root.dtype {
        eta_ir::Dtype::I32 => Value::I32(vec![root.literal_bits as i32]),
        eta_ir::Dtype::U32 => Value::U32(vec![root.literal_bits]),
        eta_ir::Dtype::Bool => Value::Bool(vec![u8::from(root.literal_bits != 0)]),
        eta_ir::Dtype::F32 => Value::F32(vec![f32::from_bits(root.literal_bits)]),
        other => crate::value::no_lane(other),
    }
}

#[must_use]
pub fn step(inst: &mut InterpInstance, plan: &ExecPlan, inputs: &PassInputs) -> StepOutcome {
    if plan.prologue_runs() {
        let outcome = step_with(inst, plan, inputs, &mut Interpreted, Stage::Prologue);
        if outcome != StepOutcome::Committed {
            return outcome;
        }
    }
    step_with(inst, plan, inputs, &mut Interpreted, Stage::Epilogue)
}

/// The stages that run at `at`, gated and committed on their own.
#[must_use]
pub fn step_with(
    inst: &mut InterpInstance,
    plan: &ExecPlan,
    inputs: &PassInputs,
    runner: &mut dyn StageRunner,
    at: Stage,
) -> StepOutcome {
    if inst.poisoned {
        return StepOutcome::Faulted("instance is poisoned".to_string());
    }
    if let Some(channel) = blocked(inst, plan, at) {
        return StepOutcome::Blocked(channel);
    }

    let mut overlay = Overlay::new(inst.channels.len());
    let mut vals = vec![Value::F32(vec![]); plan.package.values.len()];

    if let Err(reason) = run_kind(
        inst,
        plan,
        Stage::Prologue,
        at,
        inputs,
        &mut overlay,
        &mut vals,
        runner,
    ) {
        inst.poisoned = true;
        return StepOutcome::Faulted(reason.to_string());
    }

    if at != Stage::Prologue {
        for port in &plan.package.ports {
            if port.is_const {
                continue;
            }
            if port_consumes(port.port) {
                let _ = overlay.take(inst, port.channel);
            }
        }
    }

    if let Err(reason) = run_kind(
        inst,
        plan,
        Stage::Epilogue,
        at,
        inputs,
        &mut overlay,
        &mut vals,
        runner,
    ) {
        inst.poisoned = true;
        return StepOutcome::Faulted(reason.to_string());
    }

    commit(inst, &overlay)
}

fn blocked(inst: &InterpInstance, plan: &ExecPlan, at: Stage) -> Option<u32> {
    inst.channels
        .iter()
        .enumerate()
        .find_map(|(channel, ring)| {
            let ready = match plan.readiness(channel as u32, at) {
                Some(Direction::NeedsFull) => !ring.is_empty(),
                Some(Direction::NeedsEmpty) => !ring.is_full(),
                None => true,
            };
            (!ready).then_some(channel as u32)
        })
}

#[allow(clippy::too_many_arguments)]
fn run_kind(
    inst: &InterpInstance,
    plan: &ExecPlan,
    kind: Stage,
    at: Stage,
    inputs: &PassInputs,
    overlay: &mut Overlay,
    vals: &mut [Value],
    runner: &mut dyn StageRunner,
) -> Result<()> {
    for sp in &plan.stages {
        if plan.package.stages[sp.stage_index].stage != kind || !plan.runs_at(sp.stage_index, at) {
            continue;
        }
        exec_stage(inst, plan, sp, inputs, overlay, vals, runner)?;
    }
    Ok(())
}

fn commit(inst: &mut InterpInstance, overlay: &Overlay) -> StepOutcome {
    let n = inst.channels.len();
    let mut old_tails = vec![0u64; n];
    let mut new_heads = vec![0u64; n];
    let mut new_tails = vec![0u64; n];

    for (ci, ring) in inst.channels.iter().enumerate() {
        let head = ring.head();
        let tail = ring.tail();
        if tail < head {
            inst.poisoned = true;
            return StepOutcome::Faulted(format!("channel {ci}: tail precedes head at commit"));
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
                return StepOutcome::Faulted(format!(
                    "channel {ci}: put overflows capacity {} at commit",
                    ring.capacity()
                ));
            }
            next_tail += 1;
        }
        old_tails[ci] = tail;
        new_heads[ci] = next_head;
        new_tails[ci] = next_tail;
    }

    for (ci, ring) in inst.channels.iter().enumerate() {
        if overlay.put[ci] {
            ring.encode_sequence(old_tails[ci], &overlay.pending[&(ci as u32)]);
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

    StepOutcome::Committed
}
