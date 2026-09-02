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

    /// The draft head's own readout, when the fire has one: a rectangle of
    /// its own, bound at [`IntrinsicId::MtpLogits`]. Leaving this `None`
    /// falls back to rows `mtp_draft_row ..` of the trunk buffer (the layout
    /// [`super::plan::bounded_mtp_row_base`] computes).
    pub mtp_logits: Option<&'a [f32]>,

    pub rows: u32,

    pub vocab: u32,

    pub mtp_draft_row: Option<u32>,

    /// The fire's per-key attention rectangle, `[planes, ATTN_SCORE_KV_MAX]`
    /// F32 row-major, read by [`IntrinsicId::AttnScore`] at the epilogue.
    /// Not logits-shaped, so it gets its own field. `None` means no lane
    /// captured anything, and reading the intrinsic against it faults rather
    /// than returning a row of zeros.
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
    // the score rectangle is its own buffer: nothing about the logits family
    // below describes it (no vocabulary, no readout row, no draft base).
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
    // a draft intrinsic reads the draft column when bound, else falls back
    // to rows `mtp_draft_row ..` of the trunk's; both index by the same row.
    let drafts_column = matches!(
        root_intr,
        Some(IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts)
    );
    let own = drafts_column && inputs.mtp_logits.is_some();
    let Some(logits) = (if own { inputs.mtp_logits } else { inputs.logits }) else {
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
    // A column of its own is measured by its own height, and the trunk's rows
    // say nothing about it.
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
            ValueOrigin::Const => const_root_value(root),
            ValueOrigin::ChannelTake => overlay.take(inst, root.channel),
            ValueOrigin::ChannelRead => overlay.resolve(inst, root.channel),
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
    }
    for put in &stage.puts {
        overlay
            .pending
            .insert(put.channel, vals[put.value as usize].clone());
        overlay.put[put.channel as usize] = true;
    }
    Ok(())
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
