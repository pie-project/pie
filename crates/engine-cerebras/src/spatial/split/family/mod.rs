//! Program families: what differs between the models the role-split program runs, as one trait
//! a family. Everything generic (floor building, wiring primitives, the fabric, the emitters,
//! the placer, the host) lives outside; a family composes it.

use std::collections::BTreeMap;

use super::emit::meta::Meta;
use super::emit::params::KernelBuild;
use super::emit::rcfg::RcfgAbi;
use super::fabric::Palette;
use super::floor::Floor;
use super::plan::{FamilyId, Plan};
use super::program::Program;
use super::role::RoleKind;
use super::weights::Binding;
use super::wiring::Wiring;

pub mod abi;
pub mod gemma;
pub mod gemma12;
pub mod gptoss;
pub mod qwen;
pub mod tiles;

/// A program family. Every method is pure: same inputs, same outputs (the parity tests and the
/// `split-geo` tool rely on it).
pub trait Family: RcfgAbi + Send + Sync {
    fn id(&self) -> FamilyId;

    /// The logical plan from the prep's meta (and its overrides). Later the spatial compiler
    /// hands the plan in directly; nothing physical may be read here.
    fn plan(&self, meta: &Meta) -> Result<Plan, String>;

    /// The floorplan: units, gap / relay slots, head block, store blocks, every PE's role.
    fn floor(&self, plan: &Plan) -> Result<Floor, String>;

    /// Every link: the generic unit wiring ([`super::wiring::units`]) plus the family's
    /// segments, head input, token loop variant and extras.
    fn wire(&self, plan: &Plan, floor: &Floor) -> Result<Wiring, String>;

    /// The family's colour map (today's constants, so every word is unchanged).
    fn palette(&self, plan: &Plan) -> Palette;

    /// Each role kind's kernel source, owned symbols and shape params (levers applied by
    /// [`super::emit::params::kernels`]).
    fn kernels(&self, prog: &Program) -> Result<BTreeMap<RoleKind, KernelBuild>, String>;

    /// Every copy onto the wafer, in placement order.
    fn bindings(&self, prog: &Program) -> Result<Vec<Binding>, String>;
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
