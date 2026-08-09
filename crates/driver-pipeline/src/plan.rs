//! The executable plan: a launch package plus the derived facts the pass reads.
//!
//! A [`driver_abi::plan::LaunchPackage`] is what the host ships — the value
//! table, channels, ports and stage DAGs. It is complete but not yet *indexed*
//! for execution: it does not say which values a given stage must evaluate,
//! which const ports fold to which cells, or whether the program is even
//! runnable on this CPU interpreter. [`ExecPlan`] is the package with those
//! facts computed once, at adoption, so [`crate::step`] never has to
//! re-derive them per fire.
//!
//! # Why adoption can reject
//!
//! This interpreter is the channel-plane fallback: it runs the sampling/
//! control logic that lives between forward passes, plus a bounded logits read.
//! It deliberately does **not** run the model forward, so a program that reads
//! a per-layer intrinsic (hidden state, query, attention scores) or attaches a
//! per-layer tap has no meaning here. [`classify_exec_plan`] marks those plans
//! non-executable with a reason rather than failing mid-fire, so the caller can
//! route them to a backend that does run the forward.

use std::collections::{BTreeSet, HashMap};

use driver_abi::local::{
    PIE_READINESS_NEEDS_FULL, PIE_VALUE_CHANNEL_READ, PIE_VALUE_CHANNEL_TAKE, PIE_VALUE_INTRINSIC,
};
use driver_abi::plan::LaunchPackage;
use tensor_ir::op::{IntrinsicId, intrinsic_tags, tags};
use tensor_ir::registry::{Port, Stage};

use super::shape_numel;
use super::value::{Value, concrete_dtype, decode_wire};
use crate::{Error, Result};

/// True when a descriptor port **consumes** (takes) its channel at the
/// descriptor phase, advancing the ring, rather than peeking it.
///
/// This defers to [`tensor_ir::registry::Port::consumes`] — the source of truth
/// — rather than the C++ mirror's hand-written `port == 0 || 2 || 6 || 7`. The
/// mirror predates the RS ports (`w_slot`/`w_off` for the recurrent state, and
/// `rs_fold_len`) joining the consuming side, so reusing the registry both
/// avoids a second copy of the list and picks up the fix documented on
/// `Port::consumes`: a peeked `rs_fold_len` records a persistent read that made
/// a device-handoff fire wait on its own output forever. An unrecognised port
/// byte peeks, matching the mirror's default.
#[must_use]
pub fn port_consumes(port: u8) -> bool {
    Port::from_u8(port).is_some_and(Port::consumes)
}

/// A const descriptor port pre-folded to the cell it always feeds.
///
/// A const port never consumes a channel, so its value is fixed at adoption.
/// Decoding it once here keeps it off the per-fire path.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstPortValue {
    /// The `PTIR_PORT_*` tag this value binds to.
    pub port: u8,
    /// The decoded cell.
    pub value: Value,
}

/// One stage's execution index: the value ids to evaluate, and which op
/// produces each result.
///
/// Built by [`rebuild_stage_indexes`]. `value_ids` is **sorted ascending**,
/// which is load-bearing: SSA operands always have lower ids than the ops that
/// consume them, so evaluating ids in ascending order guarantees every operand
/// is ready before its op runs, without a topological sort.
#[derive(Clone, Debug, Default)]
pub struct StagePlan {
    /// Index of this stage in the package's `stages`/`plans` arrays.
    pub stage_index: usize,
    /// The value ids this stage must materialize, ascending.
    pub value_ids: Vec<u32>,
    /// First result id → index of the producing op within the stage body.
    ///
    /// Keyed on the *first* result id only. A two-result op (`sort_desc`,
    /// `top_k`) is triggered by its first id and writes both cells, so the
    /// second id is never itself a key — a lookup on it would miss and be
    /// mistaken for a value root.
    pub op_by_result: HashMap<u32, usize>,
}

/// A launch package plus the derived facts the pass needs.
///
/// Owns the package: adoption is a one-way handoff, and the interpreter reads
/// the trace on every fire, so borrowing it would only push a lifetime through
/// every signature for no benefit.
#[derive(Clone, Debug)]
pub struct ExecPlan {
    /// The adopted launch package — the value table, channels, ports, stage
    /// bodies, and per-stage launch plans (`package.plans`).
    pub package: LaunchPackage,
    /// Per-stage execution indexes, parallel to `package.stages`.
    pub stages: Vec<StagePlan>,
    /// Const ports folded to their cells.
    pub const_ports: Vec<ConstPortValue>,
    /// Whether this interpreter can run the program at all.
    ///
    /// A plain flag plus [`ExecPlan::reject_reason`] rather than a `Result`:
    /// executability is a *persistent property* of the plan that the caller
    /// queries and re-queries while routing, not a one-shot outcome of a call.
    /// A `Result` would force a clone or a re-run to ask twice.
    pub executable: bool,
    /// Why the plan is not executable, or `None` when it is.
    pub reject_reason: Option<String>,
    /// The program reads the epilogue logits intrinsic.
    pub needs_logits: bool,
    /// The program reads the multi-token-prediction logits/drafts intrinsics.
    pub needs_mtp_logits: bool,
}

impl ExecPlan {
    /// Whether any stage or non-const consuming port takes channel `c`.
    #[must_use]
    pub fn takes_channel(&self, c: u32) -> bool {
        self.package.stages.iter().any(|s| s.takes.contains(&c))
            || self
                .package
                .ports
                .iter()
                .any(|p| !p.is_const && p.channel == c && port_consumes(p.port))
    }

    /// Whether any stage or non-const peeking port reads channel `c`.
    #[must_use]
    pub fn reads_channel(&self, c: u32) -> bool {
        self.package.stages.iter().any(|s| s.reads.contains(&c))
            || self
                .package
                .ports
                .iter()
                .any(|p| !p.is_const && p.channel == c && !port_consumes(p.port))
    }

    /// Whether any stage puts into channel `c`.
    #[must_use]
    pub fn puts_channel(&self, c: u32) -> bool {
        self.package
            .stages
            .iter()
            .any(|s| s.puts.iter().any(|put| put.channel == c))
    }

    /// Whether channel `c`'s readiness gate requires a full cell before a fire.
    #[must_use]
    pub fn requires_channel_input(&self, c: u32) -> bool {
        (c as usize) < self.package.channels.len()
            && self.package.channels[c as usize].readiness == PIE_READINESS_NEEDS_FULL
    }

    /// Whether this plan needs the model forward run before a fire — either it
    /// reads a logits intrinsic or it binds descriptor ports the forward feeds.
    #[must_use]
    pub fn needs_forward(&self) -> bool {
        self.needs_logits || self.needs_mtp_logits || !self.package.ports.is_empty()
    }
}

/// The largest per-vocab row count any `Logits` intrinsic in the plan reads, or
/// `None` when the plan does not need MTP logits, `vocab` is zero, a logits
/// value is not a whole multiple of `vocab`, or the row count overflows `u32`.
///
/// Returns `Option<u32>` in place of the C++ `-1` sentinel: "no bounded row
/// base" is the absence of an answer, not the number minus one. `Some(0)` is a
/// real answer (the plan needs MTP logits but declares no `Logits` value), kept
/// distinct from `None`.
#[must_use]
pub fn bounded_mtp_row_base(plan: &ExecPlan, vocab: u32) -> Option<u32> {
    if !plan.needs_mtp_logits || vocab == 0 {
        return None;
    }
    let mut rows: u64 = 0;
    for value in &plan.package.values {
        if value.source != PIE_VALUE_INTRINSIC
            || u16::from(value.intrinsic) != intrinsic_tags::LOGITS
        {
            continue;
        }
        let numel = shape_numel(&value.shape);
        if !numel.is_multiple_of(u64::from(vocab)) {
            return None;
        }
        rows = rows.max(numel / u64::from(vocab));
    }
    u32::try_from(rows).ok()
}

/// Decode a const port's payload into the cell it always feeds.
///
/// Bool is handled separately from [`decode_wire`] because a const port ships
/// its bool payload **one byte per lane** (not bit-packed): it is host-authored
/// constant data, not a channel cell. The bytes are clamped to the declared
/// element count, zero-extended if short, and normalized to `0`/`1`. Any other
/// dtype goes through the wire decoder, falling back to zeros if the length is
/// wrong — a wrong-length const is a rejected-upstream case, and zeros keep the
/// fold total.
#[must_use]
pub fn const_port_value(port: &driver_abi::plan::LaunchPort) -> Value {
    let count = shape_numel(&port.const_shape).max(1) as usize;
    let dtype = concrete_dtype(port.const_dtype);
    if port.const_dtype == driver_abi::local::PIE_CHANNEL_DTYPE_BOOL {
        let mut values: Vec<u8> = port
            .const_data
            .iter()
            .take(count)
            .map(|&b| u8::from(b != 0))
            .collect();
        values.resize(count, 0);
        return Value::Bool(values);
    }
    decode_wire(&port.const_data, dtype, count).unwrap_or_else(|| Value::zeros(dtype, count))
}

/// Recompute every stage's execution index from the package's stage bodies.
///
/// See [`StagePlan`] for why `value_ids` is sorted and keyed on first result
/// ids. The final sweep over the value table is not redundant with the op walk:
/// a `chan_take` whose result is dead still has to advance the ring, so a stage
/// that lists a channel in its `takes`/`reads` must evaluate that channel root
/// even though nothing downstream reads it. Missing it would leave the ring un-
/// advanced while the readiness gate kept demanding the channel be non-empty.
fn rebuild_stage_indexes(package: &LaunchPackage) -> Vec<StagePlan> {
    let values_len = package.values.len() as u32;
    let mut plans = Vec::with_capacity(package.stages.len());

    for (stage_index, stage) in package.stages.iter().enumerate() {
        let mut op_by_result = HashMap::new();
        let mut producer: HashMap<u32, usize> = HashMap::new();
        for (op_idx, op) in stage.ops.iter().enumerate() {
            op_by_result.insert(op.result_id, op_idx);
            for r in 0..op.result_count {
                producer.insert(op.result_id + u32::from(r), op_idx);
            }
        }

        let mut ids: BTreeSet<u32> = BTreeSet::new();
        let mark = |value: u32, ids: &mut BTreeSet<u32>| {
            let mut stack = vec![value];
            while let Some(v) = stack.pop() {
                if v >= values_len {
                    continue;
                }
                if let Some(&op_idx) = producer.get(&v) {
                    let op = &stage.ops[op_idx];
                    ids.insert(op.result_id);
                    for &arg in &op.args {
                        stack.push(arg);
                    }
                } else {
                    ids.insert(v);
                }
            }
        };

        for op in &stage.ops {
            ids.insert(op.result_id);
            for &arg in &op.args {
                mark(arg, &mut ids);
            }
        }
        for put in &stage.puts {
            mark(put.value, &mut ids);
        }
        for (v, root) in package.values.iter().enumerate() {
            let taken =
                root.source == PIE_VALUE_CHANNEL_TAKE && stage.takes.contains(&root.channel);
            let peeked =
                root.source == PIE_VALUE_CHANNEL_READ && stage.reads.contains(&root.channel);
            if taken || peeked {
                ids.insert(v as u32);
            }
        }

        plans.push(StagePlan {
            stage_index,
            value_ids: ids.into_iter().collect(),
            op_by_result,
        });
    }
    plans
}

/// Classify the plan: set the logits-need flags, or mark it non-executable with
/// a reason if it reads an unsupported intrinsic or attaches a per-layer tap.
pub fn classify_exec_plan(plan: &mut ExecPlan) {
    plan.executable = true;
    plan.needs_logits = false;
    plan.needs_mtp_logits = false;
    plan.reject_reason = None;

    for value in &plan.package.values {
        if value.source != PIE_VALUE_INTRINSIC {
            continue;
        }
        match IntrinsicId::from_u16(u16::from(value.intrinsic)) {
            Some(IntrinsicId::Logits) => plan.needs_logits = true,
            Some(IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts) => {
                plan.needs_logits = true;
                plan.needs_mtp_logits = true;
            }
            _ => {
                plan.executable = false;
                plan.reject_reason = Some(
                    "program reads an unsupported model intrinsic \
                     (hidden/query/value-head/layer/attn-score; Metal forward not wired)"
                        .to_string(),
                );
            }
        }
    }

    for stage in &plan.package.stages {
        if Stage::from_u8(stage.kind).is_some_and(Stage::per_layer) {
            plan.executable = false;
            plan.reject_reason =
                Some("program attaches per-layer taps (Metal forward not wired)".to_string());
        }
    }

    if !plan.executable {
        plan.needs_logits = false;
        plan.needs_mtp_logits = false;
    }
}

/// Adopt a launch package into an [`ExecPlan`], or reject it structurally.
///
/// The only structural failures are an empty stage list and a plan/stage count
/// mismatch — everything else the host already validated. Returns `Err(reason)`
/// for those, mirroring the C++ `bool` + out-param but without the sentinel.
///
/// Two boundary ops are recognised specially: this backend implements
/// `kernel_call`/`sink_call` only as the identity/discard reshape named
/// `metal.identity`/`metal.discard`; any other named or wrongly-shaped boundary
/// is not something it can launch, so the plan is adopted but marked non-
/// executable. A non-executable plan is still a valid, inspectable value — the
/// caller reads `executable`/`reject_reason` to route it — which is why this is
/// not an `Err`.
///
/// # Errors
///
/// Returns [`Error::Program`] if the package has no stages or its
/// `plans` and `stages` arrays disagree in length.
pub fn adopt_launch_package(package: LaunchPackage) -> Result<ExecPlan> {
    if package.stages.is_empty() {
        return Err(Error::Program {
            message: "launch package has no stages".to_owned(),
        });
    }
    if package.plans.len() != package.stages.len() {
        return Err(Error::Program {
            message: "launch package plan/stage count mismatch".to_owned(),
        });
    }

    let stages = rebuild_stage_indexes(&package);
    let const_ports = package
        .ports
        .iter()
        .filter(|p| p.is_const)
        .map(|p| ConstPortValue {
            port: p.port,
            value: const_port_value(p),
        })
        .collect();

    let mut plan = ExecPlan {
        package,
        stages,
        const_ports,
        executable: false,
        reject_reason: None,
        needs_logits: false,
        needs_mtp_logits: false,
    };
    classify_exec_plan(&mut plan);

    for stage in &plan.package.stages {
        for op in &stage.ops {
            let code = op.code as u8;
            let boundary = code == tags::KERNEL_CALL || code == tags::SINK_CALL;
            if !boundary {
                continue;
            }
            let want = if code == tags::KERNEL_CALL {
                "metal.identity"
            } else {
                "metal.discard"
            };
            let named = (op.name_index as usize) < plan.package.names.len()
                && plan.package.names[op.name_index as usize] == want;
            // `kernel_call` lowers to a reshape of its single operand; any
            // other arity is not the boundary this backend implements.
            let shaped = code != tags::KERNEL_CALL || op.args.len() == 1;
            if !(named && shaped) {
                plan.executable = false;
                plan.needs_logits = false;
                plan.needs_mtp_logits = false;
                plan.reject_reason = Some(
                    "program requests an unsupported Metal semantic library boundary".to_string(),
                );
            }
        }
    }

    Ok(plan)
}

#[cfg(test)]
mod tests {
    use driver_abi::local::{PIE_VALUE_INTRINSIC, PIE_VALUE_OP_RESULT};
    use driver_abi::plan::{LaunchOp, LaunchPackage, LaunchStage, LaunchStagePlan, LaunchValue};

    use super::*;

    fn intrinsic_value(id: u32, intr: u16, dims: &[u32]) -> LaunchValue {
        LaunchValue {
            id,
            source: PIE_VALUE_INTRINSIC,
            dtype: 0,
            intrinsic: intr as u8,
            channel: 0,
            literal_bits: 0,
            shape: dims.to_vec(),
        }
    }

    fn package_with(values: Vec<LaunchValue>, stages: Vec<LaunchStage>) -> LaunchPackage {
        let plans = stages.iter().map(|_| LaunchStagePlan::default()).collect();
        LaunchPackage {
            values,
            channels: vec![],
            ports: vec![],
            names: vec![],
            stages,
            plans,
        }
    }

    #[test]
    fn adoption_rejects_an_empty_stage_list_rather_than_producing_a_plan() {
        let err = adopt_launch_package(LaunchPackage::default())
            .expect_err("a package with no stages cannot be executed");
        assert_eq!(
            err.to_string(),
            "launch program cannot be interpreted: launch package has no stages"
        );
    }

    #[test]
    fn a_logits_reading_program_is_executable_and_needs_the_forward() {
        let values = vec![intrinsic_value(0, intrinsic_tags::LOGITS, &[1, 8])];
        let stage = LaunchStage {
            kind: Stage::Epilogue as u8,
            ..Default::default()
        };
        let plan = adopt_launch_package(package_with(values, vec![stage])).expect("well-formed");
        assert!(
            plan.executable,
            "a bounded logits read is exactly what this interpreter runs"
        );
        assert!(
            plan.needs_logits,
            "the epilogue logits intrinsic must set needs_logits"
        );
        assert!(!plan.needs_mtp_logits, "plain logits is not an MTP read");
    }

    #[test]
    fn a_hidden_state_read_is_rejected_because_the_forward_is_not_wired() {
        let values = vec![intrinsic_value(0, intrinsic_tags::HIDDEN, &[1, 8])];
        let stage = LaunchStage {
            kind: Stage::Epilogue as u8,
            ..Default::default()
        };
        let plan = adopt_launch_package(package_with(values, vec![stage])).expect("well-formed");
        assert!(
            !plan.executable,
            "hidden state has no meaning without a forward pass"
        );
        assert!(
            plan.reject_reason.is_some(),
            "a rejected plan must carry a reason so the caller can route it elsewhere"
        );
        assert!(
            !plan.needs_logits && !plan.needs_mtp_logits,
            "a rejected plan must clear its need flags"
        );
    }

    #[test]
    fn bounded_mtp_row_base_is_none_when_vocab_does_not_divide_the_logits() {
        let values = vec![
            intrinsic_value(0, intrinsic_tags::MTP_LOGITS, &[2, 8]),
            intrinsic_value(1, intrinsic_tags::LOGITS, &[10]),
        ];
        let stage = LaunchStage {
            kind: Stage::Epilogue as u8,
            ..Default::default()
        };
        let plan = adopt_launch_package(package_with(values, vec![stage])).expect("well-formed");
        assert!(plan.needs_mtp_logits);
        assert_eq!(
            bounded_mtp_row_base(&plan, 3),
            None,
            "a 10-element logits value is not a whole number of vocab=3 rows"
        );
        assert_eq!(
            bounded_mtp_row_base(&plan, 5),
            Some(2),
            "vocab=5 divides the 10-element logits into two rows"
        );
    }

    #[test]
    fn stage_value_ids_are_sorted_so_ssa_operands_precede_their_ops() {
        // v0 = intrinsic, v1 = op reading v0. Ids must come out ascending.
        let values = vec![
            intrinsic_value(0, intrinsic_tags::LOGITS, &[4]),
            LaunchValue {
                id: 1,
                source: PIE_VALUE_OP_RESULT,
                dtype: 0,
                intrinsic: 0,
                channel: 0,
                literal_bits: 0,
                shape: vec![4],
            },
        ];
        let stage = LaunchStage {
            kind: Stage::Epilogue as u8,
            ops: vec![LaunchOp {
                code: u16::from(tags::EXP),
                result_count: 1,
                result_id: 1,
                args: vec![0],
                shape: vec![4],
                ..Default::default()
            }],
            ..Default::default()
        };
        let plan = adopt_launch_package(package_with(values, vec![stage])).expect("well-formed");
        assert_eq!(
            plan.stages[0].value_ids,
            vec![0, 1],
            "operand v0 must be listed before its consumer v1 for ascending evaluation"
        );
    }
}
