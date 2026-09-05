use std::collections::HashMap;
use std::sync::Arc;

use eta_compiler::codegen::launch::{LaunchPackage, LaunchStagePlan};
use eta_compiler::codegen::wgsl;

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
                write!(f, "stage {stage}'s emitted WGSL was refused: {why}")
            }
        }
    }
}

pub const ENTRY: &str = "main";

pub fn entry_source(stage: usize, plan: &LaunchStagePlan) -> Result<String, Refused> {
    every_op_emits(stage, plan)?;
    wgsl::emit_launch_stage(ENTRY, plan).map_err(|why| Refused::Emitting {
        stage,
        why: why.to_string(),
    })
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

pub struct Compiled {
    stages: Vec<Lowered>,
}

pub struct Lowered {
    pub source: String,

    pub entries: Vec<String>,

    pub strides_the_grid: bool,
}

impl std::fmt::Debug for Compiled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compiled")
            .field("stages", &self.stages.len())
            .field(
                "bytes",
                &self.stages.iter().map(|s| s.source.len()).sum::<usize>(),
            )
            .field(
                "dispatches",
                &self.stages.iter().map(|s| s.entries.len()).sum::<usize>(),
            )
            .finish()
    }
}

impl Compiled {
    pub fn of(package: &LaunchPackage) -> Result<Compiled, Refused> {
        let mut stages = Vec::with_capacity(package.plans.len());
        for (at, plan) in package.plans.iter().enumerate() {
            let stepwise = step_source(at, plan)?;
            stages.push(Lowered {
                source: stepwise.source,
                entries: stepwise.steps.into_iter().map(|step| step.entry).collect(),

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

#[derive(Default)]
pub struct Forms {
    compiled: HashMap<u64, Arc<Compiled>>,

    refused: HashMap<u64, Refused>,
}

impl Forms {
    pub fn admit(&mut self, program: u64, package: &LaunchPackage) -> bool {
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
    pub fn get(&self, program: u64) -> Option<&Arc<Compiled>> {
        self.compiled.get(&program)
    }

    #[must_use]
    pub fn refusal(&self, program: u64) -> Option<&Refused> {
        self.refused.get(&program)
    }

    pub fn forget(&mut self, program: u64) {
        self.compiled.remove(&program);
        self.refused.remove(&program);
    }

    #[must_use]
    pub fn tally(&self) -> (usize, usize) {
        (self.compiled.len(), self.refused.len())
    }
}

#[cfg(feature = "wgpu")]
pub mod run;
#[cfg(feature = "wgpu")]
pub mod session;
#[cfg(feature = "wgpu")]
pub mod widen;
