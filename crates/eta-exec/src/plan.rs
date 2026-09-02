use std::collections::{BTreeSet, HashMap};

use eta_compiler::codegen::launch::{LaunchPackage, ValueOrigin};
use eta_ir::Dtype;
use eta_ir::op::IntrinsicId;
use eta_ir::op::tags;
use eta_ir::registry::Port;
use eta_ir::validate::Direction;

use super::value::{Value, decode_wire};
use crate::{Error, Result, shape_numel};

/// True iff a channel bound to this port is consumed rather than peeked.
#[must_use]
pub fn port_consumes(port: Port) -> bool {
    port.consumes()
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstPortValue {
    pub port: Port,

    pub value: Value,
}

#[derive(Clone, Debug, Default)]
pub struct StagePlan {
    pub stage_index: usize,

    pub value_ids: Vec<u32>,

    pub op_by_result: HashMap<u32, usize>,
}

#[derive(Clone, Debug)]
pub struct ExecPlan {
    pub package: LaunchPackage,

    pub stages: Vec<StagePlan>,

    pub const_ports: Vec<ConstPortValue>,

    pub executable: bool,

    pub reject_reason: Option<String>,

    pub needs_logits: bool,

    pub needs_mtp_logits: bool,

    /// The program reads [`IntrinsicId::AttnScore`]: the shell must bind the
    /// score rectangle before the epilogue runs, since a program reading
    /// scores from a fire nobody captured dereferences the side table's zero
    /// and poisons the CUDA context for the life of the process.
    pub needs_attn_scores: bool,
}

impl ExecPlan {
    #[must_use]
    pub fn takes_channel(&self, c: u32) -> bool {
        self.package.stages.iter().any(|s| s.takes.contains(&c))
            || self
                .package
                .ports
                .iter()
                .any(|p| !p.is_const && p.channel == c && port_consumes(p.port))
    }

    #[must_use]
    pub fn reads_channel(&self, c: u32) -> bool {
        self.package.stages.iter().any(|s| s.reads.contains(&c))
            || self
                .package
                .ports
                .iter()
                .any(|p| !p.is_const && p.channel == c && !port_consumes(p.port))
    }

    #[must_use]
    pub fn puts_channel(&self, c: u32) -> bool {
        self.package
            .stages
            .iter()
            .any(|s| s.puts.iter().any(|put| put.channel == c))
    }

    #[must_use]
    pub fn requires_channel_input(&self, c: u32) -> bool {
        (c as usize) < self.package.channels.len()
            && self.package.channels[c as usize].readiness == Some(Direction::NeedsFull)
    }

    #[must_use]
    pub fn needs_forward(&self) -> bool {
        self.needs_logits
            || self.needs_mtp_logits
            || self.needs_attn_scores
            || !self.package.ports.is_empty()
    }
}

#[must_use]
pub fn const_port_value(port: &eta_compiler::codegen::launch::LaunchPort) -> Value {
    let count = shape_numel(&port.const_shape).max(1) as usize;
    let dtype = port.const_dtype;
    if dtype == Dtype::Bool {
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

            if op.tag == tags::PIVOT_THRESHOLD {
                mark(op.pred_payload, &mut ids);
            }
        }
        for put in &stage.puts {
            mark(put.value, &mut ids);
        }
        for (v, root) in package.values.iter().enumerate() {
            let taken =
                root.source == ValueOrigin::ChannelTake && stage.takes.contains(&root.channel);
            let peeked =
                root.source == ValueOrigin::ChannelRead && stage.reads.contains(&root.channel);
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

pub fn classify_exec_plan(plan: &mut ExecPlan) {
    plan.executable = true;
    plan.needs_logits = false;
    plan.needs_mtp_logits = false;
    plan.needs_attn_scores = false;
    plan.reject_reason = None;

    for value in &plan.package.values {
        // `source == Intrinsic` and `intrinsic == Some(_)` are set together by
        // the lowering, so asking the second is asking both.
        match value.intrinsic {
            None => continue,
            Some(IntrinsicId::Logits) => plan.needs_logits = true,
            Some(IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts) => {
                plan.needs_logits = true;
                plan.needs_mtp_logits = true;
            }
            // The score rectangle is a column of its own, bound at its own
            // base, so reading it does not set `needs_logits`.
            Some(IntrinsicId::AttnScore) => plan.needs_attn_scores = true,
            _ => {
                plan.executable = false;
                plan.reject_reason = Some(
                    "program reads an unsupported model intrinsic \
                     (hidden/query/value-head/layer; Metal forward not wired)"
                        .to_string(),
                );
            }
        }
    }

    for stage in &plan.package.stages {
        if stage.stage.per_layer() {
            plan.executable = false;
            plan.reject_reason =
                Some("program attaches per-layer taps (Metal forward not wired)".to_string());
        }
    }

    if !plan.executable {
        plan.needs_logits = false;
        plan.needs_mtp_logits = false;
        plan.needs_attn_scores = false;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Boundaries {
    pub kernel_calls: &'static [&'static str],

    pub sink_calls: &'static [&'static str],
}

impl Boundaries {
    /// `lora` is admitted here because a sink call is a declaration, not an
    /// op: the interpreter runs nothing for it, and the effect lands on the
    /// host at instance bind. Admitting it is a claim that this backend
    /// consumes it, not that it interprets it — so it stays out of
    /// `kernel_calls` too.
    pub const METAL: Self = Self {
        kernel_calls: &["metal.identity"],
        sink_calls: &["metal.discard", "lora"],
    };

    pub const CUDA: Self = Self {
        kernel_calls: &["envelope_dot"],
        sink_calls: &["lora", "attn_page_mask"],
    };

    fn admits(self, code: u8, name: &str) -> bool {
        let vocabulary = if code == tags::KERNEL_CALL {
            self.kernel_calls
        } else {
            self.sink_calls
        };
        vocabulary.contains(&name)
    }

    fn describe(self) -> String {
        let mut all: Vec<&str> = self.kernel_calls.to_vec();
        all.extend_from_slice(self.sink_calls);
        all.join(", ")
    }
}

pub fn adopt_launch_package(package: LaunchPackage) -> Result<ExecPlan> {
    adopt_launch_package_with(package, Boundaries::METAL)
}

pub fn adopt_launch_package_with(
    package: LaunchPackage,
    boundaries: Boundaries,
) -> Result<ExecPlan> {
    if package.stages.is_empty() {
        return Err(Error {
            message: "launch package has no stages".to_owned(),
        });
    }
    if package.plans.len() != package.stages.len() {
        return Err(Error {
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
        needs_attn_scores: false,
    };
    classify_exec_plan(&mut plan);

    for stage in &plan.package.stages {
        for op in &stage.ops {
            let boundary = op.tag == tags::KERNEL_CALL || op.tag == tags::SINK_CALL;
            if !boundary {
                continue;
            }
            let name = plan
                .package
                .names
                .get(op.name_index as usize)
                .map_or("<unnamed>", String::as_str);
            let named = boundaries.admits(op.tag, name);

            let shaped = op.tag != tags::KERNEL_CALL || op.args.len() == 1;
            if !(named && shaped) {
                plan.executable = false;
                plan.needs_logits = false;
                plan.needs_mtp_logits = false;

                plan.reject_reason = Some(format!(
                    "program requests the semantic library boundary `{name}`, \
                     which this backend does not implement (it has: {})",
                    boundaries.describe()
                ));
            }
        }
    }

    Ok(plan)
}
