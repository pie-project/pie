//! A built wafer program: the plan, the blocks it composes, their floor, wiring, palette
//! and laid fabric, every kind's kernel and the per-PE param overrides. The one entry point the
//! host, the tools and the tests use.

use super::abi::{Kernels, Overrides};
use super::block::Block;
use super::fabric::{Fabric, Palette};
use super::family::{self, Family};
use super::floor::{Floor, FloorBuilder};
use super::plan::Plan;
use super::weights::{Binding, Copies};
use super::wiring::Wiring;

pub struct Program {
    pub plan: Plan,
    pub blocks: Vec<Box<dyn Block>>,
    pub floor: Floor,
    pub wiring: Wiring,
    pub palette: Palette,
    pub fabric: Fabric,
    pub kernels: Kernels,
    pub overrides: Overrides,
    pub family: Box<dyn Family>,
}

impl Program {
    /// The program of `plan`, by its family ([`Program::build`]).
    pub fn of(plan: Plan) -> Result<Program, String> {
        Program::build(family::of(plan.family), plan)
    }

    /// The physical build of a plan: validate it; each block places its PEs, then lays its
    /// links, then settles the floor on them; the palette is the blocks' colours; lay the fabric
    /// and validate it (its invariants); the family's kernels, every PE's routes within its
    /// kernel's route table; the blocks' param overrides. Pure (no device, no files).
    pub fn build(family: Box<dyn Family>, plan: Plan) -> Result<Program, String> {
        plan.validate()?;
        let blocks = family.blocks(&plan)?;
        let mut fb = FloorBuilder::default();
        for b in &blocks {
            b.floor(&plan, &mut fb)?;
        }
        let mut floor = fb.build()?;
        let mut wiring = Wiring::default();
        for b in &blocks {
            b.wire(&plan, &floor, &mut wiring)?;
        }
        for b in &blocks {
            b.settle(&plan, &wiring, &mut floor)?;
        }
        let palette = Palette::of(blocks.iter().flat_map(|b| b.colours()));
        let fabric = Fabric::lay(&floor, &wiring, &palette)?;
        fabric.validate(&floor)?;
        let mut prog = Program {
            plan,
            blocks,
            floor,
            wiring,
            palette,
            fabric,
            kernels: Kernels::new(),
            overrides: Overrides::default(),
            family,
        };
        prog.kernels = prog.family.kernels(&prog)?;
        prog.fabric.check_route_slots(&prog.floor, &|k| {
            prog.kernels.get(&k).map(|kb| kb.route_slots)
        })?;
        let mut o = Overrides::default();
        for b in &prog.blocks {
            b.overrides(&prog, &mut o)?;
        }
        prog.overrides = o;
        Ok(prog)
    }

    /// Every copy onto the wafer: the blocks' in order.
    pub fn bindings(&self) -> Result<Vec<Binding>, String> {
        let mut out = Copies::default();
        for b in &self.blocks {
            b.bindings(self, &mut out)?;
        }
        Ok(out.0)
    }
}
