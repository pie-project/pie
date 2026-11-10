//! Program families: what a model's program is made of. A family reads its plan off the prep's
//! meta (until the spatial compiler hands the plan in), lists the blocks the plan composes, and
//! builds each kind's kernel from the plan. Everything else (the floor, the wiring, the fabric,
//! the emitters, the placer, the host) is generic over the blocks.

use super::abi::Kernels;
use super::block::Block;
use super::emit::meta::Meta;
use super::plan::{FamilyId, Plan};
use super::program::Program;

pub mod gemma;
pub mod gemma12;
pub mod gptoss;
pub mod qwen;

type R<T> = Result<T, String>;

/// A program family. Every method is pure: same inputs, same outputs (the parity tests and the
/// `split-geo` tool rely on it).
pub trait Family: Send + Sync {
    fn id(&self) -> FamilyId;

    /// The logical plan from the prep's meta (and its overrides). Later the spatial compiler
    /// hands the plan in directly; nothing physical may be read here.
    fn plan(&self, meta: &Meta) -> R<Plan>;

    /// The blocks the plan composes, in order: the units first, then what is placed against
    /// them. Floor, wiring, palette, overrides and copies run over them in this order.
    fn blocks(&self, plan: &Plan) -> R<Vec<Box<dyn Block>>>;

    /// Each kind's kernel and its params (levers applied by [`super::emit::params::kernels`]).
    fn kernels(&self, prog: &Program) -> R<Kernels>;

    /// Every name the kernels export, with its element type, in export order: the ABI between
    /// the kernels and the host's memcpy.
    fn symbols(&self) -> &'static [(&'static str, &'static str)];
}

/// The family a meta was prepped for.
pub fn of(id: FamilyId) -> Box<dyn Family> {
    match id {
        FamilyId::Qwen => Box::new(qwen::Qwen),
        FamilyId::GptOss => Box::new(gptoss::GptOss),
        FamilyId::Gemma => Box::new(gemma::Gemma),
        FamilyId::Gemma12 => Box::new(gemma12::Gemma12),
    }
}
