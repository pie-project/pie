use std::collections::BTreeMap;

use tensor_ir::op::IntrinsicId;
use tensor_ir::registry::Stage;
use tensor_ir::validate::Direction;

use driver_api::program::ValueSource;

use super::channel::InterpInstance;
use super::op::eval_op;
use super::plan::{ExecPlan, StagePlan, port_consumes};
use super::value::Value;
use crate::{Error, Result, shape_numel};

#[derive(Clone, Copy, Debug)]
pub struct PassInputs<'a> {
    pub logits: Option<&'a [f32]>,

    pub rows: u32,

    pub vocab: u32,

    pub mtp_draft_row: Option<u32>,
}

impl PassInputs<'_> {
    #[must_use]
    pub fn none() -> Self {
        PassInputs {
            logits: None,
            rows: 0,
            vocab: 0,
            mtp_draft_row: None,
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
    let bounded = matches!(
        root_intr,
        Some(IntrinsicId::Logits | IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts)
    );
    if !bounded {
        return Err(Error::Program {
            message: "unresolved value root (unsupported intrinsic) reached execution".to_owned(),
        });
    }
    let Some(logits) = inputs.logits else {
        return Err(Error::Program {
            message: "logits intrinsic unbound (forward did not run before step)".to_owned(),
        });
    };
    if inputs.vocab == 0 {
        return Err(Error::Program {
            message: "logits intrinsic unbound (forward did not run before step)".to_owned(),
        });
    }
    let want = root_numel.max(1);
    let vocab = u64::from(inputs.vocab);
    let drafts = root_intr == Some(IntrinsicId::MtpDrafts);
    let rows_needed = if drafts { want } else { want / vocab };
    if !drafts && !want.is_multiple_of(vocab) {
        return Err(Error::Program {
            message: "logits intrinsic shape mismatch (program vocab != model vocab)".to_owned(),
        });
    }
    let base_row = if drafts || root_intr == Some(IntrinsicId::MtpLogits) {
        u64::from(inputs.mtp_draft_row.unwrap_or(0))
    } else {
        0
    };
    if base_row + rows_needed > u64::from(inputs.rows) {
        return Err(Error::Program {
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

fn exec_stage(
    inst: &InterpInstance,
    plan: &ExecPlan,
    sp: &StagePlan,
    inputs: &PassInputs,
    overlay: &mut Overlay,
    vals: &mut [Value],
) -> Result<()> {
    let stage = &plan.package.stages[sp.stage_index];
    for &id in &sp.value_ids {
        if let Some(&op_idx) = sp.op_by_result.get(&id) {
            eval_op(&stage.ops[op_idx], &plan.package, vals)?;
            continue;
        }
        let root = &plan.package.values[id as usize];
        let cell = match root.source {
            ValueSource::Const => const_root_value(root),
            ValueSource::ChannelTake => overlay.take(inst, root.channel),
            ValueSource::ChannelRead => overlay.resolve(inst, root.channel),
            ValueSource::Intrinsic => {
                bind_intrinsic(root.intrinsic, shape_numel(&root.shape), inputs)?
            }
            ValueSource::OpResult => {
                return Err(Error::Program {
                    message: "unresolved value root (intrinsic/host input) reached execution"
                        .to_owned(),
                });
            }
        };
        vals[id as usize] = cell;
    }
    for put in &stage.puts {
        overlay
            .pending
            .insert(put.channel, vals[put.value as usize].clone());
        overlay.put[put.channel as usize] = true;
    }
    Ok(())
}

fn const_root_value(root: &driver_api::program::LaunchValue) -> Value {
    match root.dtype {
        tensor_ir::DType::I32 => Value::I32(vec![root.literal_bits as i32]),
        tensor_ir::DType::U32 => Value::U32(vec![root.literal_bits]),
        tensor_ir::DType::Bool => Value::Bool(vec![u8::from(root.literal_bits != 0)]),
        tensor_ir::DType::F32 => Value::F32(vec![f32::from_bits(root.literal_bits)]),
    }
}

#[must_use]
pub fn step(inst: &mut InterpInstance, plan: &ExecPlan, inputs: &PassInputs) -> StepOutcome {
    if inst.poisoned {
        return StepOutcome::Faulted("instance is poisoned".to_string());
    }

    for (channel, ring) in inst.channels.iter().enumerate() {
        let readiness = plan
            .package
            .channels
            .get(channel)
            .and_then(|c| c.readiness);
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

    if let Err(reason) = run_kind(inst, plan, Stage::Prologue, inputs, &mut overlay, &mut vals) {
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

    if let Err(reason) = run_kind(inst, plan, Stage::Epilogue, inputs, &mut overlay, &mut vals) {
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
) -> Result<()> {
    for sp in &plan.stages {
        if plan.package.stages[sp.stage_index].stage != kind {
            continue;
        }
        exec_stage(inst, plan, sp, inputs, overlay, vals)?;
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
