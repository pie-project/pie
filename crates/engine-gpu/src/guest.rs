//! Guest passes on the device: each stage of a launch package emitted as
//! WGSL, lowered to what the backend's pipelines take, and the forms a
//! program admits. The session that dispatches a stage over the API's
//! bindings stays in each backend crate; what it shares is here.

use engine::program::ProgramId;
use std::collections::HashMap;
use std::sync::Arc;

use eta_compiler::codegen::launch::{LaunchPackage, LaunchStagePlan, ValueOrigin};
use eta_compiler::codegen::wgsl;

pub mod defer;
pub mod held;
pub mod ring;
pub mod run;
pub mod session;
pub mod state;
pub mod widen;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    Ops {
        stage: usize,

        missing: Vec<(u8, &'static str)>,
    },

    Emitting {
        stage: usize,

        why: String,
    },

    Lowering {
        stage: usize,

        why: String,
    },

    /// A stage whose tables name a value or an op past their own: the
    /// device reads its descriptors, offsets and op params unchecked, so
    /// the host refuses such a plan here.
    Tables {
        stage: usize,

        why: String,
    },

    /// A value the host writes before the stage's launches or reads after
    /// them that the stage's scratch is not laid to hold for it.
    Scratch {
        stage: usize,

        value: u32,

        why: &'static str,
    },
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ops { stage, missing } => {
                let named = missing
                    .iter()
                    .map(|(tag, name)| format!("`{name}` ({tag:#04x})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "stage {stage} runs {named}, which this backend emits no WGSL for, and a \
                     guest pass runs on the device whole or not at all"
                )
            }
            Self::Emitting { stage, why } => {
                write!(f, "stage {stage} has no emitted WGSL: {why}")
            }
            Self::Lowering { stage, why } => {
                write!(f, "step call {stage}'s WGSL did not lower: {why}")
            }
            Self::Tables { stage, why } => {
                write!(f, "stage {stage}'s tables name past themselves: {why}")
            }
            Self::Scratch { stage, value, why } => {
                write!(
                    f,
                    "stage {stage}'s scratch does not hold value {value} for the host: {why}"
                )
            }
        }
    }
}

pub const ENTRY: &str = "main";

/// How a backend lowers a stage's emitted WGSL to what its pipelines take.
pub trait Lower {
    type Code;

    fn lower(stage: usize, source: String) -> Result<Self::Code, Refused>;
}

fn every_op_emits(stage: usize, plan: &LaunchStagePlan) -> Result<(), Refused> {
    let mut missing: Vec<(u8, &'static str)> = Vec::new();
    for op in &plan.ops {
        if wgsl::is_boundary(op.tag) || wgsl::emits(op.tag) {
            continue;
        }
        if !missing.iter().any(|&(tag, _)| tag == op.tag) {
            missing.push((op.tag, eta_ir::op::spec(op.tag).map_or("?", |row| row.name)));
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(Refused::Ops { stage, missing })
    }
}

/// Holds the scratch's lifetimes ([`session::Stepping`]) to what the host
/// does with `package`'s values: every root it writes before a stage's
/// launches is one the stage holds from its first, and every put it reads
/// after them is one the stage holds to its last.
fn scratch_holds_the_hosts_values(package: &LaunchPackage) -> Result<(), Refused> {
    let steppings: Vec<session::Stepping> =
        package.plans.iter().map(session::Stepping::of).collect();
    let local = |global: u32| match package.plan_local(global) {
        Some((stage, Some(local))) => Some((stage, local)),
        _ => None,
    };
    for (global, value) in package.values.iter().enumerate() {
        if value.source == ValueOrigin::OpResult {
            continue;
        }
        let Some((stage, value)) = local(global as u32) else {
            continue;
        };
        if !steppings
            .get(stage)
            .is_some_and(|stepping| stepping.staged.contains(&value))
        {
            return Err(Refused::Scratch {
                stage,
                value,
                why: "the host writes it before the stage runs and no output, channel read, \
                      take or constant of the stage's plan yields it",
            });
        }
    }
    for put in package.stages.iter().flat_map(|stage| &stage.puts) {
        let Some((stage, value)) = local(put.value) else {
            continue;
        };
        if !steppings
            .get(stage)
            .is_some_and(|stepping| stepping.kept.contains(&value))
        {
            return Err(Refused::Scratch {
                stage,
                value,
                why: "the host reads it after the stage runs and no put of the stage's plan \
                      reads it",
            });
        }
    }
    Ok(())
}

/// Refuses a stage whose tables would have the device read past them: an
/// op naming a value at or past the stage's values (an operand, a result,
/// a pivot's payload), a value of more axes than a descriptor holds, or a
/// step naming an op past the stage's. The WGSL runtime reads descriptors,
/// offsets and op params at these indices unchecked: every one is written
/// by the host, which checks them here, once, at registration.
pub fn plan_tables_valid(
    stage: usize,
    plan: &LaunchStagePlan,
    steps: &[wgsl::Step],
) -> Result<(), Refused> {
    let refused = |why: String| Refused::Tables { stage, why };
    let values = plan.value_types.len() as u64;
    if let Some((at, value)) = plan
        .value_types
        .iter()
        .enumerate()
        .find(|(_, value)| value.axes.len() > eta_ir::types::MAX_RANK)
    {
        return Err(refused(format!(
            "value {at} has {} axes, past the {} a descriptor holds",
            value.axes.len(),
            eta_ir::types::MAX_RANK
        )));
    }
    for (at, op) in plan.ops.iter().enumerate() {
        if let Some(&arg) = op.args.iter().find(|&&arg| u64::from(arg) >= values) {
            return Err(refused(format!(
                "op {at} reads value {arg}, past the stage's {values}"
            )));
        }
        let end = u64::from(op.result_id) + u64::from(op.result_count);
        if end > values {
            return Err(refused(format!(
                "op {at} writes values {}..{end}, past the stage's {values}",
                op.result_id
            )));
        }
        if op.tag == eta_ir::op::tags::PIVOT_THRESHOLD && u64::from(op.pred_payload) >= values {
            return Err(refused(format!(
                "op {at}'s pivot reads value {}, past the stage's {values}",
                op.pred_payload
            )));
        }
    }
    if let Some(step) = steps
        .iter()
        .find(|step| step.node as usize >= plan.ops.len())
    {
        return Err(refused(format!(
            "a step names op {}, past the stage's {}",
            step.node,
            plan.ops.len()
        )));
    }
    Ok(())
}

pub fn step_source(stage: usize, plan: &LaunchStagePlan) -> Result<wgsl::Stepwise, Refused> {
    every_op_emits(stage, plan)?;
    wgsl::emit_launch_steps(ENTRY, plan).map_err(|why| Refused::Emitting {
        stage,
        why: why.to_string(),
    })
}

/// A stage's dispatches, in order: each a call of the closed set
/// ([`wgsl::step_calls`]) at its node, rung and selector.
#[derive(Clone, Debug)]
pub struct Lowered {
    pub steps: Vec<wgsl::Step>,

    pub strides_the_grid: bool,
}

/// What a program's stages dispatch. No stage holds code of its own: every
/// call runs on one of the pipelines a backend built at load
/// ([`step_calls`]).
#[derive(Debug)]
pub struct Compiled {
    stages: Vec<Lowered>,
}

impl Compiled {
    pub fn of(package: &LaunchPackage) -> Result<Compiled, Refused> {
        scratch_holds_the_hosts_values(package)?;
        let mut stages = Vec::with_capacity(package.plans.len());
        for (at, plan) in package.plans.iter().enumerate() {
            let steps = step_source(at, plan)?.steps;
            plan_tables_valid(at, plan, &steps)?;
            stages.push(Lowered {
                steps,
                strides_the_grid: true,
            });
        }
        Ok(Compiled { stages })
    }

    #[must_use]
    pub fn stage(&self, at: usize) -> Option<&Lowered> {
        self.stages.get(at)
    }

    #[must_use]
    pub fn all(&self) -> &[Lowered] {
        &self.stages
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

/// How many calls a guest stage can dispatch ([`wgsl::step_calls`]): the
/// pipelines a backend builds before it serves, so a program registered
/// while serving builds none.
#[must_use]
pub fn step_calls() -> usize {
    wgsl::step_calls().len()
}

/// Call `call`'s module, with its one entry point [`ENTRY`].
#[must_use]
pub fn step_module(call: usize) -> Option<String> {
    wgsl::step_calls()
        .get(call)
        .map(|text| wgsl::step_module(ENTRY, text))
}

pub struct Forms {
    compiled: HashMap<ProgramId, Arc<Compiled>>,

    refused: HashMap<ProgramId, Refused>,
}

impl Default for Forms {
    fn default() -> Forms {
        Forms {
            compiled: HashMap::new(),
            refused: HashMap::new(),
        }
    }
}

impl Forms {
    pub fn admit(&mut self, program: ProgramId, package: &LaunchPackage) -> bool {
        match Compiled::of(package) {
            Ok(compiled) => {
                self.compiled.insert(program, Arc::new(compiled));
                true
            }
            Err(why) => {
                self.refused.insert(program, why);
                false
            }
        }
    }

    /// Holds `program`'s form, as its registration's compile lowered it
    /// ([`crate::program::adopt`]).
    pub fn hold(&mut self, program: ProgramId, form: Compiled) {
        self.compiled.insert(program, Arc::new(form));
    }

    #[must_use]
    pub fn get(&self, program: ProgramId) -> Option<&Arc<Compiled>> {
        self.compiled.get(&program)
    }

    #[must_use]
    pub fn refusal(&self, program: ProgramId) -> Option<&Refused> {
        self.refused.get(&program)
    }

    pub fn forget(&mut self, program: ProgramId) {
        self.compiled.remove(&program);
        self.refused.remove(&program);
    }

    #[must_use]
    pub fn tally(&self) -> (usize, usize) {
        (self.compiled.len(), self.refused.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stage of two values and one op, `args` its operands, its result at
    /// `result`.
    fn a_stage(args: Vec<u32>, result: u32) -> LaunchStagePlan {
        use eta_compiler::codegen::launch::{LaunchOp, LaunchPlanValue};
        LaunchStagePlan {
            value_types: vec![LaunchPlanValue::default(), LaunchPlanValue::default()],
            ops: vec![LaunchOp {
                args,
                result_id: result,
                result_count: 1,
                ..LaunchOp::default()
            }],
            ..LaunchStagePlan::default()
        }
    }

    fn a_step(node: u32) -> wgsl::Step {
        wgsl::Step {
            entry: String::new(),
            node,
            call: 0,
            rung: 0,
            selector: 0,
        }
    }

    #[test]
    fn a_stage_whose_tables_stay_in_themselves_is_admitted() {
        assert!(plan_tables_valid(0, &a_stage(vec![0], 1), &[a_step(0)]).is_ok());
    }

    #[test]
    fn an_op_naming_a_value_past_its_stage_is_refused_at_registration() {
        let why = plan_tables_valid(0, &a_stage(vec![5], 1), &[a_step(0)])
            .expect_err("refused")
            .to_string();
        assert!(why.contains("reads value 5"), "{why}");
        // Its results past the stage's values too.
        assert!(plan_tables_valid(0, &a_stage(vec![0], 2), &[a_step(0)]).is_err());
    }

    #[test]
    fn a_step_naming_an_op_past_its_stage_is_refused_at_registration() {
        let why = plan_tables_valid(0, &a_stage(vec![0], 1), &[a_step(0), a_step(1)])
            .expect_err("refused")
            .to_string();
        assert!(why.contains("names op 1"), "{why}");
    }

    mod shipped {
        use eta_compiler::codegen::launch::LaunchPackage;
        use eta_compiler::plan::compile_bound;
        use eta_dsl::Traced;
        use eta_dsl::prelude::*;
        use eta_ir::registry::{Axis, EngineCaps, OutputDecl, Port};

        fn leak(channel: Channel) -> &'static Channel {
            Box::leak(Box::new(channel))
        }

        fn logits(vocab: u32) -> Vec<Vec<OutputDecl>> {
            vec![vec![OutputDecl {
                name: "logits".into(),
                axis: Axis::Row,
                width: vocab,
                dtype: eta_ir::Dtype::F32,
                layers: 0,
                readout: true,
            }]]
        }

        fn builder(vocab: u32) -> (Builder<'static>, &'static Channel) {
            let tok_in = leak(Channel::from([42i32]));
            let mut b = Builder::new(vocab, 16);
            b.outputs(logits(vocab));
            b.bind_port(Port::EmbedTokens, tok_in);
            b.bind_port(Port::EmbedIndptr, leak(Channel::from([0u32, 1])));
            b.bind_port(Port::Fold, leak(Channel::from([u32::MAX])));
            (b, tok_in)
        }

        /// The greedy decode epilogue.
        pub fn greedy(vocab: u32) -> Traced {
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let t = reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]);
                tok_in.put(&t);
                out.put(&t);
            });
            out.note_host_take();
            b.build().expect("the greedy epilogue traces")
        }

        pub fn package(traced: &Traced, vocab: u32) -> LaunchPackage {
            let caps = EngineCaps {
                lora: true,
                attn_page_mask: true,
            };
            let bound = eta_ir::validate::bind(traced.container().clone(), logits(vocab), caps)
                .expect("the program binds");
            let stages = compile_bound(&bound);
            eta_compiler::codegen::launch::build(&bound, &stages)
        }
    }

    /// A value the host stages or reads back that the stage's plan does not
    /// hold for it is refused by name, never laid over.
    #[test]
    fn a_value_the_plan_does_not_hold_for_the_host_is_refused() {
        let vocab = 64;
        let package = shipped::package(&shipped::greedy(vocab), vocab);
        assert!(scratch_holds_the_hosts_values(&package).is_ok());
        let put = package.stages.iter().flat_map(|stage| &stage.puts).next();
        let put = put.expect("greedy puts its token").value;
        let (stage, Some(held)) = package.plan_local(put).expect("a stage's value") else {
            panic!("the put's value is a stage's");
        };
        let stepping = session::Stepping::of(&package.plans[stage]);
        assert!(stepping.kept.contains(&held));
        let unheld = (0..package.plans[stage].value_types.len() as u32)
            .find(|value| !stepping.kept.contains(value) && !stepping.staged.contains(value))
            .expect("greedy's argmax is neither staged nor kept");

        let global_of = |local: u32| {
            (0..package.values.len() as u32)
                .find(|&global| package.plan_local(global) == Some((stage, Some(local))))
                .expect("a stage's value has a global")
        };
        let mut read_back = package.clone();
        for put in read_back
            .stages
            .iter_mut()
            .flat_map(|stage| &mut stage.puts)
        {
            put.value = global_of(unheld);
        }
        assert!(matches!(
            scratch_holds_the_hosts_values(&read_back),
            Err(Refused::Scratch { value, .. }) if value == unheld
        ));
        let mut staged = package.clone();
        staged.values[global_of(unheld) as usize].source = ValueOrigin::Const;
        assert!(matches!(
            scratch_holds_the_hosts_values(&staged),
            Err(Refused::Scratch { value, .. }) if value == unheld
        ));
    }
}
