//! A built role-split program: the plan, its floor, wiring and laid fabric, and the family that
//! made them. The one entry point the host, the tools and the tests use.
//!
//! Replaces `mod.rs::SplitSpec::geo` and the geometry steps of `SplitModel::open*` (build the
//! family's geometry, then place); `SplitModel` keeps the run-time driving and takes a
//! [`Program`] + [`super::emit::host::HostLayout`].

use super::emit::meta::Meta;
use super::fabric::Fabric;
use super::family::{self, Family};
use super::floor::Floor;
use super::plan::Plan;
use super::wiring::Wiring;

pub struct Program {
    pub plan: Plan,
    pub floor: Floor,
    pub wiring: Wiring,
    pub fabric: Fabric,
    pub family: Box<dyn Family>,
}

impl Program {
    /// From a prep's meta: the family it names, its plan, then [`Program::build`].
    pub fn from_meta(meta: &Meta) -> Result<Program, String> {
        let fam = family::of(meta.family());
        let plan = fam.plan(meta)?;
        Program::build(fam, plan)
    }

    /// The physical build of a plan: validate the plan, floor, wire, lay the fabric through the
    /// family's palette, validate the fabric (its invariants, then every PE's routes against
    /// its kernel's route slots). Pure (no device, no files).
    pub fn build(family: Box<dyn Family>, plan: Plan) -> Result<Program, String> {
        plan.validate()?;
        let floor = family.floor(&plan)?;
        let wiring = family.wire(&plan, &floor)?;
        let fabric = Fabric::lay(&floor, &wiring, &family.palette(&plan))?;
        fabric.validate(&floor)?;
        fabric.check_route_slots(&floor, &|k| family.route_slots(k))?;
        Ok(Program {
            plan,
            floor,
            wiring,
            fabric,
            family,
        })
    }
}
