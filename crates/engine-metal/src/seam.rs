//! Phase-2 debt: this shell still binds the standard outputs by their
//! conventional names, one plane a kind, until it binds every output by its
//! declaration.

use poem_ir::{Trace, ValueId};

pub const OUT: &str = "logits";
pub const SCORES: &str = "attn.scores";
pub const VELOCITY: &str = "velocity";
pub const HIDDEN: &str = "hidden";
pub const PIXELS: &str = "pixels";

/// The plane kinds this shell reads a lane back through.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Seam {
    #[default]
    Logits,
    Velocity,
    Hidden,
    Pixels,
}

impl Seam {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Seam::Logits => OUT,
            Seam::Velocity => VELOCITY,
            Seam::Hidden => HIDDEN,
            Seam::Pixels => PIXELS,
        }
    }
}

/// The values `trace` reads out as `name`, in the order it wrote them.
pub fn named<'a>(trace: &'a Trace, name: &'a str) -> impl Iterator<Item = ValueId> + 'a {
    trace
        .outputs
        .iter()
        .filter(move |output| output.name == name)
        .map(|output| output.value)
}

/// The outputs this shell binds, each at a slot of its own a lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Intrinsic {
    Logits = 0,
    Hidden = 2,
    Velocity = 8,
    Pixels = 9,
}

impl Intrinsic {
    /// Slots a lane takes in this shell's own table.
    pub const SLOTS: u32 = 10;

    /// The output `name` reads as, or `None` for one this shell does not bind.
    #[must_use]
    pub fn of_name(name: &str) -> Option<Intrinsic> {
        Some(match name {
            OUT => Intrinsic::Logits,
            HIDDEN => Intrinsic::Hidden,
            VELOCITY => Intrinsic::Velocity,
            PIXELS => Intrinsic::Pixels,
            _ => return None,
        })
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Intrinsic::Logits => OUT,
            Intrinsic::Hidden => HIDDEN,
            Intrinsic::Velocity => VELOCITY,
            Intrinsic::Pixels => PIXELS,
        }
    }

    /// A per-lane output, read through the address table.
    #[must_use]
    pub fn addressed(self) -> bool {
        self != Intrinsic::Logits
    }

    /// This shell's slot of it for `lane`.
    #[must_use]
    pub fn slot(self, lane: u32) -> u32 {
        self as u32 + Intrinsic::SLOTS * lane
    }
}

/// What an output op of a stage plan reads as.
#[must_use]
pub fn op_intrinsic(
    plan: &eta_compiler::codegen::launch::LaunchStagePlan,
    op: &eta_compiler::codegen::launch::LaunchOp,
) -> Option<Intrinsic> {
    (op.tag == eta_ir::op::tags::OUTPUT)
        .then(|| plan.names.get(op.name_index as usize))
        .flatten()
        .and_then(|name| Intrinsic::of_name(name))
}

/// What a package value reads as.
#[must_use]
pub fn value_intrinsic(value: &eta_compiler::codegen::launch::LaunchValue) -> Option<Intrinsic> {
    value.output.as_deref().and_then(Intrinsic::of_name)
}

/// The outputs a package reads this shell does not bind, by name.
#[must_use]
pub fn unbound(package: &eta_compiler::codegen::launch::LaunchPackage) -> Vec<&str> {
    let mut names: Vec<&str> = package
        .values
        .iter()
        .filter_map(|value| value.output.as_deref())
        .filter(|name| Intrinsic::of_name(name).is_none())
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}
