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

#[derive(Default)]
pub struct Forms {
    compiled: HashMap<ProgramId, Arc<Compiled>>,

    refused: HashMap<ProgramId, Refused>,
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
    use eta_exec::Extents;

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

        pub const QWEN_VOCAB: u32 = 151_936;

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

        /// `text-completion-bench`'s decode epilogue: a temperature-scaled
        /// top-p Gumbel-max draw.
        pub fn top_p(vocab: u32) -> Traced {
            let (temperature, top_p) = (0.6f32, 0.95f32);
            let rng = leak(Channel::from([0x7ce1u32, 0]));
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let logits = intrinsics::output(0, "logits");
                let probs = softmax(cast(&logits, dtype::bf16) * (1.0 / temperature));
                let keep = pivot_threshold(probs, cummass_le(top_p));
                let r = rng.take();
                let r_next = &r + iota(2);
                rng.put(&r_next);
                let t = reshape(gumbel_argmax(&logits, &keep, 1.0 / temperature, &r), [1]);
                tok_in.put(&t);
                out.put(&t);
            });
            out.note_host_take();
            b.build().expect("the top-p epilogue traces")
        }

        /// The `compat` generate step's sampler with a top-k and a top-p
        /// cut: both pivots, their `and`, the mask and a Gumbel-max draw.
        pub fn top_k_top_p(vocab: u32) -> Traced {
            let temperature = 0.8f32;
            let rng = leak(Channel::from([0x7ce1u32, 0]));
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let r = rng.take();
                let logits = reshape(intrinsics::output(0, "logits"), [vocab]);
                let probs = softmax(cast(&logits, dtype::bf16) * (1.0 / temperature));
                let ranked = pivot_threshold(&probs, rank_le(40u32));
                let nucleus = pivot_threshold(&probs, cummass_le(0.9f32));
                let keep = and(&ranked, &nucleus);
                let token = reshape(gumbel_argmax(&logits, keep, 1.0 / temperature, &r), [1]);
                rng.put(&r + iota(2));
                tok_in.put(&token);
                out.put(&token);
            });
            out.note_host_take();
            b.build().expect("the top-k top-p epilogue traces")
        }

        /// A temperature Gumbel-max draw with no cut.
        pub fn gumbel_max_only(vocab: u32) -> Traced {
            let rng = leak(Channel::from([0x7ce1u32, 0]));
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let r = rng.take();
                let logits = intrinsics::output(0, "logits");
                let token = reshape(gumbel_argmax(logits, true, 1.0 / 0.7f32, &r), [1]);
                rng.put(&r + iota(2));
                tok_in.put(&token);
                out.put(&token);
            });
            out.note_host_take();
            b.build().expect("the gumbel-max epilogue traces")
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

        /// A greedy step that lands its top logprobs: a top-k region.
        pub fn top_logprobs(vocab: u32) -> Traced {
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let values_out = leak(Channel::new([5], dtype::f32).capacity(8));
            let ids_out = leak(Channel::new([5], dtype::i32).capacity(8));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let logits = reshape(intrinsics::output(0, "logits"), [vocab]);
                let t = reshape(reduce_argmax(&logits), [1]);
                let (values, ids) = top_k(log_softmax(cast(&logits, dtype::bf16)), 5);
                values_out.put(cast(&values, dtype::f32));
                ids_out.put(cast(&ids, dtype::i32));
                tok_in.put(&t);
                out.put(&t);
            });
            values_out.note_host_take();
            ids_out.note_host_take();
            out.note_host_take();
            b.build().expect("the top-logprobs epilogue traces")
        }

        /// A greedy step that lands the whole row in order: a sort region.
        pub fn sorted(vocab: u32) -> Traced {
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let ids_out = leak(Channel::new([vocab], dtype::i32).capacity(2));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let logits = reshape(intrinsics::output(0, "logits"), [vocab]);
                let t = reshape(reduce_argmax(&logits), [1]);
                let (_, ids) = sort_desc(softmax(cast(&logits, dtype::bf16)));
                ids_out.put(cast(&ids, dtype::i32));
                tok_in.put(&t);
                out.put(&t);
            });
            ids_out.note_host_take();
            out.note_host_take();
            b.build().expect("the sorted epilogue traces")
        }

        /// `beam-search`'s prompt epilogue: the prompt's top `beams` by log
        /// probability, and its greedy pick beside them.
        pub fn beam_prefill(vocab: u32, beams: u32) -> Traced {
            let ids = leak(Channel::new([beams], dtype::i32));
            let scores = leak(Channel::new([beams], dtype::f32));
            let greedy = leak(Channel::new([1], dtype::i32));
            let (mut b, _) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let logits = cast(
                    reshape(intrinsics::output(0, "logits"), [vocab]),
                    dtype::bf16,
                );
                let (s, i) = top_k(log_softmax(&logits), beams);
                ids.put(cast(&i, dtype::i32));
                scores.put(cast(&s, dtype::f32));
                greedy.put(reshape(reduce_argmax(&logits), [1]));
            });
            ids.note_host_take();
            scores.note_host_take();
            greedy.note_host_take();
            b.build().expect("the beam prefill epilogue traces")
        }

        /// `beam-search`'s decode epilogue over `beams` rows: each beam's
        /// score plus its row's log probabilities, the top `beams` of all
        /// of them, their parents and tokens, and each row's greedy pick.
        pub fn beam_decode(vocab: u32, beams: u32) -> Traced {
            let carried = leak(Channel::from(vec![0f32; beams as usize]));
            let out = leak(Channel::new([beams], dtype::i32).capacity(8));
            let parents = leak(Channel::new([beams], dtype::u32).capacity(8));
            let greedy = leak(Channel::new([beams], dtype::i32).capacity(8));
            let mut b = Builder::new(vocab, 16);
            b.outputs(logits(vocab));
            let rows = beams as usize;
            b.bind_port(Port::EmbedTokens, leak(Channel::from(vec![42i32; rows])));
            b.bind_port(
                Port::EmbedIndptr,
                leak(Channel::from((0u32..=beams).collect::<Vec<u32>>())),
            );
            b.bind_port(Port::Fold, leak(Channel::from([u32::MAX])));
            b.stage(Stage::Epilogue, move || {
                let logits = cast(
                    reshape(intrinsics::output(0, "logits"), [beams, vocab]),
                    dtype::bf16,
                );
                let scores = carried.take();
                let best = reduce_max(&scores);
                let relative = cast(&scores - &best, dtype::bf16);
                let cand =
                    broadcast(reshape(relative, [beams, 1]), [beams, vocab]) + log_softmax(&logits);
                let (s, i) = top_k(reshape(cand, [beams * vocab]), beams);
                carried.put(cast(&s, dtype::f32) + &best);
                out.put(cast(&i % vocab, dtype::i32));
                parents.put(&i / vocab);
                greedy.put(reshape(reduce_argmax(&logits), [beams]));
            });
            out.note_host_take();
            parents.note_host_take();
            greedy.note_host_take();
            b.build().expect("the beam decode epilogue traces")
        }

        /// Every program here, by name.
        pub fn all(vocab: u32) -> Vec<(&'static str, Traced)> {
            vec![
                ("top-p", top_p(vocab)),
                ("top-k + top-p", top_k_top_p(vocab)),
                ("gumbel-max", gumbel_max_only(vocab)),
                ("greedy", greedy(vocab)),
                ("top logprobs", top_logprobs(vocab)),
                ("sorted", sorted(vocab)),
            ]
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

    /// Laid over the shipped programs, no two values alive together share a
    /// byte, a reshape read as its source lies at its source's offset and
    /// keeps the source alive through its own readers, and no value alive
    /// with a node's temporary area shares a byte of it or of the reserve
    /// below it: an area takes only bytes of values dead while it is.
    #[test]
    fn a_shipped_program_s_live_items_share_no_bytes() {
        let vocab = shipped::QWEN_VOCAB;
        let four = Extents {
            row_count: 4,
            token_count: 4,
            sampled_rows: 4,
            ..Extents::default()
        };
        let mut programs: Vec<(&str, eta_dsl::Traced, Extents)> = shipped::all(vocab)
            .into_iter()
            .map(|(name, traced)| (name, traced, Extents::default()))
            .collect();
        programs.push((
            "beam prefill",
            shipped::beam_prefill(vocab, 4),
            Extents::default(),
        ));
        programs.push(("beam-4 decode", shipped::beam_decode(vocab, 4), four));
        for (name, traced, extents) in &programs {
            let package = shipped::package(traced, vocab);
            let plan = &package.plans[0];
            let tables = session::stage_tables(plan, extents).expect("static shapes");
            let stepping = session::Stepping::of(plan);
            let launches = stepping.launches(&tables.descriptors);
            let lifetimes = eta_exec::value_lifetimes(plan, &launches).expect("every node runs");
            let laid = eta_exec::lay_out(plan, &tables.descriptors, &launches).expect("fits");
            let span = |value: usize| {
                let at = tables.offsets[value];
                (at, at + tables.descriptors[value].device_bytes())
            };
            let overlap =
                |a: eta_exec::Lifetime, b: eta_exec::Lifetime| a.last >= b.def && b.last >= a.def;
            let aliases = eta_compiler::codegen::wgsl_analysis::analyze_stage(plan).aliases;
            for &(result, source) in &aliases {
                let (result, source) = (result as usize, source as usize);
                assert!(
                    lifetimes[result].dead,
                    "{name}: alias {result} takes no bytes"
                );
                assert_eq!(tables.offsets[result], tables.offsets[source], "{name}");
                assert!(
                    lifetimes[source].last >= lifetimes[result].last,
                    "{name}: value {source} dies before its alias {result} is read"
                );
            }
            let live: Vec<usize> = (0..lifetimes.len())
                .filter(|&v| !lifetimes[v].dead)
                .collect();
            for (at, &a) in live.iter().enumerate() {
                for &b in &live[at + 1..] {
                    let ((sa, ea), (sb, eb)) = (span(a), span(b));
                    if overlap(lifetimes[a], lifetimes[b]) {
                        assert!(
                            ea <= sb || eb <= sa,
                            "{name}: values {a} and {b} alive together"
                        );
                    }
                }
            }
            let areas = eta_exec::launch_areas(&launches);
            assert_eq!(areas.len(), laid.areas.len(), "{name}");
            for (at, (area, &first)) in areas.iter().zip(&laid.areas).enumerate() {
                let low = first - area.reserve;
                let high = first + area.bytes;
                for &value in &live {
                    let (start, end) = span(value);
                    if overlap(lifetimes[value], area.life) {
                        assert!(
                            end <= low || high <= start,
                            "{name}: value {value} in area {at}"
                        );
                    }
                }
                for (other, (them, &their)) in
                    areas.iter().zip(&laid.areas).enumerate().skip(at + 1)
                {
                    if overlap(area.life, them.life) {
                        assert!(
                            high <= their - them.reserve || their + them.bytes <= low,
                            "{name}: areas {at} and {other}"
                        );
                    }
                }
            }
            assert_eq!(laid.temporary_bytes, 0, "{name}: no area is the stage's");
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
