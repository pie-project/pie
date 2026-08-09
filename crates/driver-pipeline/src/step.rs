//! The pass: readiness gate, stage execution, and the predicated commit.
//!
//! [`step`] runs one fire of a program instance. It is the only entry point
//! that mutates channel rings, and it does so **pass-atomically**: it validates
//! every resulting ring state before publishing any head/tail word, so a
//! semantic fault leaves the instance's channels exactly as it found them (and
//! poisons the instance) rather than half-applied.
//!
//! # The overlay is what makes a fire a transaction
//!
//! Ops do not touch rings. A `chan_take` reads through the [`Overlay`], a
//! `chan_put` records a pending cell in it, and only the commit phase turns the
//! overlay's take/put marks into head/tail advances. This is what gives a
//! channel its register semantics inside a pass — a take after an in-pass put
//! sees the pending value, and a double put is last-wins — and it is what lets
//! the whole fire roll back on a fault.

use std::collections::BTreeMap;

use tensor_ir::op::{IntrinsicId, intrinsic_tags};
use tensor_ir::registry::Stage;

use driver_abi::local::{
    PIE_READINESS_NEEDS_EMPTY, PIE_READINESS_NEEDS_FULL, PIE_VALUE_CHANNEL_READ,
    PIE_VALUE_CHANNEL_TAKE, PIE_VALUE_CONST, PIE_VALUE_INTRINSIC,
};

use super::channel::InterpInstance;
use super::op::eval_op;
use super::plan::{ExecPlan, StagePlan, port_consumes};
use super::shape_numel;
use super::value::Value;
use crate::{Error, Result};

/// The per-fire forward inputs the interpreter binds into `Intrinsic` roots.
///
/// `logits` is borrowed, not owned: it is the executor's read-out matrix for
/// this fire, and copying it per fire would be pure waste. A channel-plane-only
/// fire passes [`PassInputs::none`] (`logits: None`); `step` never looks at it
/// unless the plan actually roots an `Intrinsic` value, so the absent case is
/// represented by absence rather than a null pointer plus a zero count.
#[derive(Clone, Copy, Debug)]
pub struct PassInputs<'a> {
    /// The `[rows, vocab]` row-major f32 read-out for this fire, or `None` for a
    /// fire that needs no forward.
    pub logits: Option<&'a [f32]>,
    /// The number of read-out rows in `logits`.
    pub rows: u32,
    /// The vocabulary width — the row stride of `logits`.
    pub vocab: u32,
    /// The first draft row an MTP intrinsic reads, or `None` to fall back to
    /// row 0.
    ///
    /// `Option` rather than the C++ `-1` sentinel: "unset, use row 0" is the
    /// absence of a base row, not the row before the first.
    pub mtp_draft_row: Option<u32>,
}

impl PassInputs<'_> {
    /// The inputs for a fire that needs no forward pass.
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

/// The outcome of one [`step`].
///
/// Splits the C++ struct's mixed flags into a status enum plus the one datum a
/// non-commit carries. A hard fault is [`StepOutcome::Faulted`] with the reason;
/// a readiness miss is [`StepOutcome::Blocked`] with the channel that was not
/// ready; a successful fire is [`StepOutcome::Committed`]. There is no "ok but
/// not committed and no reason" state to misread, because the enum does not let
/// one be constructed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepOutcome {
    /// The fire's readiness held and its channel effects were published.
    Committed,
    /// A channel's readiness gate was not satisfied; no effect was applied. The
    /// field is the channel that blocked the fire.
    Blocked(u32),
    /// A semantic fault occurred; the instance is now poisoned. The field is the
    /// fault reason, rendered.
    ///
    /// A rendered message rather than an [`Error`] because an outcome is
    /// compared and cloned -- a run is replayed and the two outcomes checked
    /// against each other -- and [`Error`] carries an `io::Error`, which is
    /// neither `Clone` nor `PartialEq`. The reason a fault is worth carrying
    /// here is so it can be reported, and Display is that.
    Faulted(String),
}

/// The in-pass channel overlay: pending puts and the take/put marks the commit
/// phase reads.
struct Overlay {
    /// Channel → the value a `chan_put` staged this pass (last put wins).
    pending: BTreeMap<u32, Value>,
    /// Per channel, whether a `chan_take` consumed it this pass.
    taken: Vec<bool>,
    /// Per channel, whether a `chan_put` staged into it this pass.
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

    /// Resolve a channel's current value: the pending put if one was staged this
    /// pass, otherwise the ring's current cell.
    fn resolve(&self, inst: &InterpInstance, chan: u32) -> Value {
        if let Some(v) = self.pending.get(&chan) {
            return v.clone();
        }
        inst.channels[chan as usize].current()
    }

    /// Resolve a channel and mark it taken so the commit phase advances its
    /// head.
    fn take(&mut self, inst: &InterpInstance, chan: u32) -> Value {
        self.taken[chan as usize] = true;
        self.resolve(inst, chan)
    }
}

/// Bind an `Intrinsic` value root from the forward's read-out.
///
/// Returns the decoded cell or a fault reason. Kept separate from
/// [`exec_stage`] because the logits/MTP/drafts binding is the one place a fire
/// dereferences [`PassInputs`], and its several shape checks (vocab divides the
/// value, the row range fits the read-out) are each a distinct fault the caller
/// must surface rather than paper over with zeros.
fn bind_intrinsic(root_intr: u8, root_numel: u64, inputs: &PassInputs) -> Result<Value> {
    let intr = IntrinsicId::from_u16(u16::from(root_intr));
    let is_bounded = matches!(
        intr,
        Some(IntrinsicId::Logits | IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts)
    );
    if !is_bounded {
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
    let drafts = u16::from(root_intr) == intrinsic_tags::MTP_DRAFTS;
    let rows_needed = if drafts { want } else { want / vocab };
    if !drafts && !want.is_multiple_of(vocab) {
        return Err(Error::Program {
            message: "logits intrinsic shape mismatch (program vocab != model vocab)".to_owned(),
        });
    }
    let base_row = if drafts || u16::from(root_intr) == intrinsic_tags::MTP_LOGITS {
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
        // Each draft row's token is the argmax of its logits row, NaN-skipping
        // with a lower-index tie-break — the same contract as `reduce_argmax`,
        // computed inline because the source is the raw read-out, not a cell.
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

/// Execute one stage: materialize its value ids, then stage its puts on the
/// overlay.
///
/// Returns `Err(reason)` on a semantic fault. Puts land at stage end (register
/// semantics: the trace model carries no put position within a stage, so a
/// double put resolves to last-wins), which is why they are applied here after
/// every value id is evaluated rather than interleaved.
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
            PIE_VALUE_CONST => const_root_value(root),
            PIE_VALUE_CHANNEL_TAKE => overlay.take(inst, root.channel),
            PIE_VALUE_CHANNEL_READ => overlay.resolve(inst, root.channel),
            PIE_VALUE_INTRINSIC => {
                bind_intrinsic(root.intrinsic, shape_numel(&root.shape), inputs)?
            }
            _ => {
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

/// Decode a `const` value root from its literal bits, per its dtype byte.
fn const_root_value(root: &driver_abi::plan::LaunchValue) -> Value {
    match super::value::concrete_dtype(root.dtype) {
        tensor_ir::DType::I32 => Value::I32(vec![root.literal_bits as i32]),
        tensor_ir::DType::U32 => Value::U32(vec![root.literal_bits]),
        tensor_ir::DType::Bool => Value::Bool(vec![u8::from(root.literal_bits != 0)]),
        tensor_ir::DType::F32 => Value::F32(vec![f32::from_bits(root.literal_bits)]),
    }
}

/// Execute one pass of the instance under `plan`.
///
/// The order is: readiness gate → prologue stages → descriptor-port takes →
/// epilogue stages → validate every ring → publish. A readiness miss returns
/// [`StepOutcome::Blocked`] having changed nothing; a semantic fault returns
/// [`StepOutcome::Faulted`] and poisons the instance; success advances the rings
/// and returns [`StepOutcome::Committed`]. The per-layer tap stages this
/// interpreter rejects at classification never run here.
#[must_use]
pub fn step(inst: &mut InterpInstance, plan: &ExecPlan, inputs: &PassInputs) -> StepOutcome {
    if inst.poisoned {
        return StepOutcome::Faulted("instance is poisoned".to_string());
    }

    // Readiness gate: each channel points one way — the direction its first op
    // in pass order needs — and a channel both taken and put has only its
    // order to say which gate a fire must clear.
    for (channel, ring) in inst.channels.iter().enumerate() {
        let readiness = plan
            .package
            .channels
            .get(channel)
            .map_or(0, |c| c.readiness);
        let ready = match readiness {
            PIE_READINESS_NEEDS_FULL => !ring.is_empty(),
            PIE_READINESS_NEEDS_EMPTY => !ring.is_full(),
            _ => true,
        };
        if !ready {
            return StepOutcome::Blocked(channel as u32);
        }
    }

    let mut overlay = Overlay::new(inst.channels.len());
    let mut vals = vec![Value::F32(vec![]); plan.package.values.len()];

    // Prologue.
    if let Err(reason) = run_kind(inst, plan, Stage::Prologue, inputs, &mut overlay, &mut vals) {
        inst.poisoned = true;
        return StepOutcome::Faulted(reason.to_string());
    }
    // Descriptor ports: a consuming port advances its channel; a peeking port
    // does not. The port values feed the forward, which this increment does
    // not run, so only the ring effect matters here.
    for port in &plan.package.ports {
        if port.is_const {
            continue;
        }
        if port_consumes(port.port) {
            let _ = overlay.take(inst, port.channel);
        }
    }
    // Epilogue.
    if let Err(reason) = run_kind(inst, plan, Stage::Epilogue, inputs, &mut overlay, &mut vals) {
        inst.poisoned = true;
        return StepOutcome::Faulted(reason.to_string());
    }

    commit(inst, &overlay)
}

/// Run every stage of one kind, in package order.
fn run_kind(
    inst: &InterpInstance,
    plan: &ExecPlan,
    kind: Stage,
    inputs: &PassInputs,
    overlay: &mut Overlay,
    vals: &mut [Value],
) -> Result<()> {
    for sp in &plan.stages {
        if plan.package.stages[sp.stage_index].kind != kind as u8 {
            continue;
        }
        exec_stage(inst, plan, sp, inputs, overlay, vals)?;
    }
    Ok(())
}

/// Validate every ring, then publish the pass's head/tail advances atomically.
///
/// Two passes on purpose: the first computes each ring's next head/tail and
/// rejects an overflow or an inverted ring *before* any word is written, so a
/// fault cannot leave some rings advanced and others not. Only once every ring
/// is known good are the pending cells encoded and the tail words released.
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

    // Pending cells become visible only when the tail word is released below.
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use driver_abi::local::{
        PIE_CHANNEL_HOST_VISIBLE, PIE_READINESS_NEEDS_EMPTY, PIE_READINESS_NEEDS_FULL,
    };
    use driver_abi::plan::{
        LaunchChannel, LaunchOp, LaunchPackage, LaunchPut, LaunchStage, LaunchStagePlan,
        LaunchValue,
    };
    use tensor_ir::op::tags;

    use super::super::channel::make_host_instance;
    use super::super::plan::adopt_launch_package;
    use super::*;

    // A program with one host-visible input channel (id 0, needs full) and one
    // output channel (id 1, needs empty). The epilogue takes channel 0, adds 1,
    // and puts the result into channel 1.
    fn counter_package() -> LaunchPackage {
        let channels = vec![
            LaunchChannel {
                id: 0,
                capacity: 1,
                dtype: 0,
                flags: PIE_CHANNEL_HOST_VISIBLE,
                extern_dir: -1,
                readiness: PIE_READINESS_NEEDS_FULL,
                shape: vec![],
                extern_name: vec![],
            },
            LaunchChannel {
                id: 1,
                capacity: 1,
                dtype: 0,
                flags: 0,
                extern_dir: -1,
                readiness: PIE_READINESS_NEEDS_EMPTY,
                shape: vec![],
                extern_name: vec![],
            },
        ];
        // v0: chan_take(0); v1: const 1.0; v2: v0 + v1.
        let values = vec![
            LaunchValue {
                id: 0,
                source: PIE_VALUE_CHANNEL_TAKE,
                dtype: 0,
                intrinsic: 0,
                channel: 0,
                literal_bits: 0,
                shape: vec![],
            },
            LaunchValue {
                id: 1,
                source: PIE_VALUE_CONST,
                dtype: 0,
                intrinsic: 0,
                channel: 0,
                literal_bits: 1.0f32.to_bits(),
                shape: vec![],
            },
            LaunchValue {
                id: 2,
                source: driver_abi::local::PIE_VALUE_OP_RESULT,
                dtype: 0,
                intrinsic: 0,
                channel: 0,
                literal_bits: 0,
                shape: vec![],
            },
        ];
        let stage = LaunchStage {
            kind: Stage::Epilogue as u8,
            ops: vec![LaunchOp {
                code: u16::from(tags::ADD),
                result_count: 1,
                result_id: 2,
                args: vec![0, 1],
                shape: vec![],
                ..Default::default()
            }],
            puts: vec![LaunchPut {
                channel: 1,
                value: 2,
            }],
            takes: vec![0],
            reads: vec![],
        };
        LaunchPackage {
            values,
            channels,
            ports: vec![],
            names: vec![],
            stages: vec![stage.clone()],
            plans: vec![LaunchStagePlan::default()],
        }
    }

    #[test]
    fn a_fire_blocks_when_its_input_channel_is_empty_and_changes_nothing() {
        let plan = adopt_launch_package(counter_package()).expect("well-formed");
        let mut inst = make_host_instance(&plan, &BTreeMap::new(), &BTreeMap::new());
        let outcome = step(&mut inst, &plan, &PassInputs::none());
        assert_eq!(
            outcome,
            StepOutcome::Blocked(0),
            "channel 0 needs a full cell; an empty ring must block the fire on channel 0"
        );
        assert!(!inst.poisoned, "a readiness miss is not a fault");
    }

    #[test]
    fn a_committed_fire_takes_the_input_and_publishes_the_incremented_output() {
        let plan = adopt_launch_package(counter_package()).expect("well-formed");
        let mut seeds = BTreeMap::new();
        seeds.insert(0u32, Value::F32(vec![41.0]));
        let mut inst = make_host_instance(&plan, &BTreeMap::new(), &seeds);
        let outcome = step(&mut inst, &plan, &PassInputs::none());
        assert_eq!(
            outcome,
            StepOutcome::Committed,
            "input full, output empty: the fire commits"
        );
        assert!(
            inst.channels[0].is_empty(),
            "the take must have advanced channel 0's head"
        );
        assert_eq!(
            inst.channels[1].pop(),
            Some(Value::F32(vec![42.0])),
            "channel 1 must hold 41 + 1 = 42 after commit"
        );
    }

    /// The counter program with a second output channel that the commit will
    /// find unusable, so the first one's put has to be held back.
    fn two_output_package() -> LaunchPackage {
        let mut package = counter_package();
        package.channels.push(LaunchChannel {
            id: 2,
            capacity: 1,
            dtype: 0,
            flags: 0,
            extern_dir: -1,
            // No readiness requirement, so the fire is admitted and the
            // trouble is only discovered during the commit -- which is the
            // case the two-pass commit exists for.
            readiness: 0,
            shape: vec![],
            extern_name: vec![],
        });
        package.stages[0].puts.push(LaunchPut {
            channel: 2,
            value: 2,
        });
        package
    }

    #[test]
    fn a_fault_on_the_second_channel_leaves_the_first_channel_untouched() {
        let plan = adopt_launch_package(two_output_package()).expect("well-formed");
        let mut seeds = BTreeMap::new();
        seeds.insert(0u32, Value::F32(vec![41.0]));
        let mut inst = make_host_instance(&plan, &BTreeMap::new(), &seeds);

        // Channel 2 is filled behind the program's back, so its put overflows
        // a capacity of one. Nothing else about the fire is wrong: channel 0
        // is full, channel 1 is empty, and the arithmetic succeeds.
        assert!(inst.channels[2].push(&Value::F32(vec![7.0])));

        let outcome = step(&mut inst, &plan, &PassInputs::none());
        assert!(
            matches!(outcome, StepOutcome::Faulted(_)),
            "a put that overflows its ring must fault, not silently drop"
        );

        // This is the whole claim. The fault was found on channel 2, and
        // channel 1's put was already computed by the time it was found. If
        // the commit wrote as it went, channel 1 now holds 42 and channel 0
        // has been consumed -- a fire that half happened, which a replay
        // cannot reproduce and a retry would double-count.
        assert!(
            inst.channels[1].is_empty(),
            "channel 1 was published even though the pass faulted, so the \
             fire was half applied"
        );
        assert!(
            !inst.channels[0].is_empty(),
            "channel 0's take was released even though the pass faulted, so \
             the input is gone and the output never arrived"
        );
        assert_eq!(
            inst.channels[2].pop(),
            Some(Value::F32(vec![7.0])),
            "the cell already in the overflowing ring was overwritten"
        );
        assert!(
            inst.poisoned,
            "a fault must poison; a half-checked instance cannot be resumed"
        );
    }

    #[test]
    fn a_poisoned_instance_refuses_every_later_fire_rather_than_recovering() {
        let plan = adopt_launch_package(counter_package()).expect("well-formed");
        let mut seeds = BTreeMap::new();
        seeds.insert(0u32, Value::F32(vec![41.0]));
        let mut inst = make_host_instance(&plan, &BTreeMap::new(), &seeds);
        inst.poisoned = true;
        assert!(
            matches!(
                step(&mut inst, &plan, &PassInputs::none()),
                StepOutcome::Faulted(_)
            ),
            "a poisoned instance ran a fire; its rings are in an unknown state \
             and anything it computes from them is unreproducible"
        );
        assert!(
            !inst.channels[0].is_empty(),
            "the refused fire consumed its input anyway"
        );
    }

    #[test]
    fn a_second_fire_blocks_because_the_output_cell_is_still_full() {
        let plan = adopt_launch_package(counter_package()).expect("well-formed");
        let mut seeds = BTreeMap::new();
        seeds.insert(0u32, Value::F32(vec![41.0]));
        let mut inst = make_host_instance(&plan, &BTreeMap::new(), &seeds);
        assert_eq!(
            step(&mut inst, &plan, &PassInputs::none()),
            StepOutcome::Committed
        );
        // Re-seed the input, but leave the output full: the second fire must
        // block on the output's NeedsEmpty gate, not commit into a full ring.
        let _ = inst.channels[0].push(&Value::F32(vec![1.0]));
        assert_eq!(
            step(&mut inst, &plan, &PassInputs::none()),
            StepOutcome::Blocked(1),
            "channel 1 is still full, so its NeedsEmpty gate blocks the fire"
        );
    }
}
