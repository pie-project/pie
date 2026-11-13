//! Guest passes on the device: each stage of a launch package emitted as
//! WGSL, lowered to what the backend's pipelines take, and the forms a
//! program admits. The session that dispatches a stage over the API's
//! bindings stays in each backend crate; what it shares is here.

use engine::program::ProgramId;
use std::collections::HashMap;
use std::sync::Arc;

use eta_compiler::codegen::launch::{LaunchPackage, LaunchStagePlan};
use eta_compiler::codegen::wgsl;

pub mod run;
pub mod session;
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
                write!(f, "stage {stage}'s emitted WGSL did not lower: {why}")
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

pub fn step_source(stage: usize, plan: &LaunchStagePlan) -> Result<wgsl::Stepwise, Refused> {
    every_op_emits(stage, plan)?;
    wgsl::emit_launch_steps(ENTRY, plan).map_err(|why| Refused::Emitting {
        stage,
        why: why.to_string(),
    })
}

pub struct Compiled<L: Lower> {
    stages: Vec<Lowered<L::Code>>,
}

pub struct Lowered<C> {
    pub code: C,

    pub entries: Vec<String>,

    pub strides_the_grid: bool,
}

impl<L: Lower> std::fmt::Debug for Compiled<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compiled")
            .field("stages", &self.stages.len())
            .field(
                "dispatches",
                &self.stages.iter().map(|s| s.entries.len()).sum::<usize>(),
            )
            .finish()
    }
}

impl<L: Lower> Compiled<L> {
    pub fn of(package: &LaunchPackage) -> Result<Compiled<L>, Refused> {
        let mut stages = Vec::with_capacity(package.plans.len());
        for (at, plan) in package.plans.iter().enumerate() {
            let stepwise = step_source(at, plan)?;
            stages.push(Lowered {
                code: L::lower(at, stepwise.source)?,
                entries: stepwise.steps.into_iter().map(|step| step.entry).collect(),
                strides_the_grid: true,
            });
        }
        Ok(Compiled { stages })
    }

    #[must_use]
    pub fn stage(&self, at: usize) -> Option<&Lowered<L::Code>> {
        self.stages.get(at)
    }

    #[must_use]
    pub fn all(&self) -> &[Lowered<L::Code>] {
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

pub struct Forms<L: Lower> {
    compiled: HashMap<ProgramId, Arc<Compiled<L>>>,

    refused: HashMap<ProgramId, Refused>,
}

impl<L: Lower> Default for Forms<L> {
    fn default() -> Forms<L> {
        Forms {
            compiled: HashMap::new(),
            refused: HashMap::new(),
        }
    }
}

impl<L: Lower> Forms<L> {
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

    #[must_use]
    pub fn get(&self, program: ProgramId) -> Option<&Arc<Compiled<L>>> {
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
