use std::collections::BTreeMap;

use eta_ir::op::IntrinsicId;
use eta_ir::registry::Stage;
use eta_ir::validate::Direction;

use eta_compiler::codegen::launch::ValueOrigin;

use super::channel::InterpInstance;
use super::op::eval_op;
use super::plan::{ExecPlan, StagePlan, port_consumes};
use super::value::Value;
use crate::{Error, Result, shape_numel};

#[derive(Clone, Copy, Debug)]
pub struct PassInputs<'a> {
    pub logits: Option<&'a [f32]>,

    pub mtp_logits: Option<&'a [f32]>,

    pub rows: u32,

    pub vocab: u32,

    pub mtp_draft_row: Option<u32>,

    pub attn_score: Option<&'a [f32]>,
}

impl PassInputs<'_> {
    #[must_use]
    pub fn none() -> Self {
        PassInputs {
            logits: None,
            mtp_logits: None,
            rows: 0,
            vocab: 0,
            mtp_draft_row: None,
            attn_score: None,
        }
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

fn bind_intrinsic(
    root_intr: Option<IntrinsicId>,
    root_numel: u64,
    inputs: &PassInputs,
) -> Result<Value> {
    if root_intr == Some(IntrinsicId::AttnScore) {
        let Some(scores) = inputs.attn_score else {
            return Err(Error {
                message: "attn_score intrinsic unbound (no lane of this fire captured scores)"
                    .to_owned(),
            });
        };
        let want = root_numel.max(1) as usize;
        if scores.len() < want {
            return Err(Error {
                message: "attn_score intrinsic declares more planes than this load exports"
                    .to_owned(),
            });
        }
        return Ok(Value::F32(scores[..want].to_vec()));
    }
    let bounded = matches!(
        root_intr,
        Some(IntrinsicId::Logits | IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts)
    );
    if !bounded {
        return Err(Error {
            message: "unresolved value root (unsupported intrinsic) reached execution".to_owned(),
        });
    }
    let drafts_column = matches!(
        root_intr,
        Some(IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts)
    );
    let own = drafts_column && inputs.mtp_logits.is_some();
    let Some(logits) = (if own {
        inputs.mtp_logits
    } else {
        inputs.logits
    }) else {
        return Err(Error {
            message: "logits intrinsic unbound (forward did not run before step)".to_owned(),
        });
    };
    if inputs.vocab == 0 {
        return Err(Error {
            message: "logits intrinsic unbound (forward did not run before step)".to_owned(),
        });
    }
    let want = root_numel.max(1);
    let vocab = u64::from(inputs.vocab);
    let drafts = root_intr == Some(IntrinsicId::MtpDrafts);
    let rows_needed = if drafts { want } else { want / vocab };
    if !drafts && !want.is_multiple_of(vocab) {
        return Err(Error {
            message: "logits intrinsic shape mismatch (program vocab != model vocab)".to_owned(),
        });
    }
    let base_row = if drafts_column {
        u64::from(inputs.mtp_draft_row.unwrap_or(0))
    } else {
        0
    };
    let held_rows = if own {
        (logits.len() as u64) / vocab.max(1)
    } else {
        u64::from(inputs.rows)
    };
    if base_row + rows_needed > held_rows {
        return Err(Error {
            message: "logits intrinsic row range exceeds the forward's readout rows".to_owned(),
        });
    }
    if drafts {
        let tokens: Vec<i32> = (0..want)
            .map(|row| {
                let start = ((base_row + row) * vocab) as usize;
                let slice = &logits[start..start + vocab as usize];
                super::op::argmax_row(slice)
            })
            .collect();
        Ok(Value::I32(tokens))
    } else {
        let start = (base_row * vocab) as usize;
        Ok(Value::F32(logits[start..start + want as usize].to_vec()))
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

    fn binds(&self, _id: Option<IntrinsicId>) -> bool {
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
    let roots = prepare_stage(inst, plan, sp, inputs, overlay, vals, &|id| {
        runner.binds(id)
    })?;
    let wanted = wanted_of(plan, sp);
    runner.run(plan, sp, &roots, &wanted, vals)?;
    finish_stage(plan, sp, overlay, vals);
    Ok(())
}

/// Resolves the stage's roots into `vals` (constants, channel reads and
/// takes through the overlay, intrinsics from `inputs` unless `binds` says
/// the runner binds them itself) and returns their ids.
fn prepare_stage(
    inst: &InterpInstance,
    plan: &ExecPlan,
    sp: &StagePlan,
    inputs: &PassInputs,
    overlay: &mut Overlay,
    vals: &mut [Value],
    binds: &dyn Fn(Option<IntrinsicId>) -> bool,
) -> Result<Vec<u32>> {
    let mut roots = Vec::new();
    for &id in &sp.value_ids {
        if sp.op_by_result.contains_key(&id) {
            continue;
        }
        let root = &plan.package.values[id as usize];
        let cell = match root.source {
            ValueOrigin::Const => const_root_value(root),
            ValueOrigin::ChannelTake => overlay.take(inst, root.channel),
            ValueOrigin::ChannelRead => overlay.resolve(inst, root.channel),
            ValueOrigin::Intrinsic if binds(root.intrinsic) => Value::F32(Vec::new()),
            ValueOrigin::Intrinsic => {
                bind_intrinsic(root.intrinsic, shape_numel(&root.shape), inputs)?
            }
            ValueOrigin::OpResult => {
                return Err(Error {
                    message: "unresolved value root (intrinsic/host input) reached execution"
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

    /// Whether the runner binds intrinsic `id` for lane `lane` itself.
    fn binds(&self, _lane: usize, _id: Option<IntrinsicId>) -> bool {
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
) -> Vec<StepOutcome> {
    let n = insts.len();
    let mut outcomes: Vec<Option<StepOutcome>> = vec![None; n];
    for (l, inst) in insts.iter().enumerate() {
        if inst.poisoned {
            outcomes[l] = Some(StepOutcome::Faulted("instance is poisoned".to_string()));
            continue;
        }
        for (channel, ring) in inst.channels.iter().enumerate() {
            let readiness = plan.package.channels.get(channel).and_then(|c| c.readiness);
            let ready = match readiness {
                Some(Direction::NeedsFull) => !ring.is_empty(),
                Some(Direction::NeedsEmpty) => !ring.is_full(),
                None => true,
            };
            if !ready {
                outcomes[l] = Some(StepOutcome::Blocked(channel as u32));
                break;
            }
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
        if kind == Stage::Epilogue {
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
            if plan.package.stages[sp.stage_index].stage != kind {
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
                    &|id| runner.binds(l, id),
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

fn const_root_value(root: &eta_compiler::codegen::launch::LaunchValue) -> Value {
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
    step_with(inst, plan, inputs, &mut Interpreted)
}

#[must_use]
pub fn step_with(
    inst: &mut InterpInstance,
    plan: &ExecPlan,
    inputs: &PassInputs,
    runner: &mut dyn StageRunner,
) -> StepOutcome {
    if inst.poisoned {
        return StepOutcome::Faulted("instance is poisoned".to_string());
    }

    for (channel, ring) in inst.channels.iter().enumerate() {
        let readiness = plan.package.channels.get(channel).and_then(|c| c.readiness);
        let ready = match readiness {
            Some(Direction::NeedsFull) => !ring.is_empty(),
            Some(Direction::NeedsEmpty) => !ring.is_full(),
            None => true,
        };
        if !ready {
            return StepOutcome::Blocked(channel as u32);
        }
    }

    let mut overlay = Overlay::new(inst.channels.len());
    let mut vals = vec![Value::F32(vec![]); plan.package.values.len()];

    if let Err(reason) = run_kind(
        inst,
        plan,
        Stage::Prologue,
        inputs,
        &mut overlay,
        &mut vals,
        runner,
    ) {
        inst.poisoned = true;
        return StepOutcome::Faulted(reason.to_string());
    }

    for port in &plan.package.ports {
        if port.is_const {
            continue;
        }
        if port_consumes(port.port) {
            let _ = overlay.take(inst, port.channel);
        }
    }

    if let Err(reason) = run_kind(
        inst,
        plan,
        Stage::Epilogue,
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

fn run_kind(
    inst: &InterpInstance,
    plan: &ExecPlan,
    kind: Stage,
    inputs: &PassInputs,
    overlay: &mut Overlay,
    vals: &mut [Value],
    runner: &mut dyn StageRunner,
) -> Result<()> {
    for sp in &plan.stages {
        if plan.package.stages[sp.stage_index].stage != kind {
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
