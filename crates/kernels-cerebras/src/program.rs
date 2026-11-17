//! A built wafer program: the plan, the blocks it composes, their floor, wiring, palette
//! and laid fabric, every kind's kernel and the per-PE param overrides. The one entry point the
//! host, the tools and the tests use.

use super::abi::{Kernels, Overrides};
use super::block::Block;
use super::compose::{self, Needs, Schedule};
use super::fabric::{Fabric, Palette};
use super::floor::{Floor, FloorBuilder};
use super::floorplan::{Floorplan, Requests};
use super::kernels::KernelSet;
use super::layers;
use super::plan::Plan;
use super::toolchain::{Arch, memcpy_fabric};
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
    /// The plan's elements: what the blocks place.
    pub floorplan: Floorplan,
    /// The role kernels the units run (by what the plan's layers need).
    pub roles: KernelSet,
    /// The role kernels' layer phases, from what the plan's layers need.
    pub schedule: Schedule,
}

impl Program {
    /// The program of `plan` ([`Program::build`]).
    pub fn of(plan: Plan) -> Result<Program, String> {
        Program::build(plan)
    }

    /// The physical build of a plan: validate it; each block places its PEs, then lays its
    /// links, then settles the floor on them; the palette is the blocks' colours; lay the fabric
    /// and validate it (its invariants); the layout's kernels, every PE's routes within its
    /// kernel's route table; the blocks' param overrides. Pure (no device, no files).
    /// The blocks are the composer's ([`compose`]) from what the layers need, their copies the
    /// layout's kernels'.
    pub fn build(plan: Plan) -> Result<Program, String> {
        plan.validate()?;
        compose::refuse(&plan.shape)?;
        let needs = Needs::of(&layers::of(&plan.shape));
        let floorplan = Floorplan::of(&plan, &Requests::of(&needs))?;
        let blocks = compose::blocks(&plan, KernelSet::of(&needs), &floorplan)?;
        Program::build_with(blocks, floorplan, plan)
    }

    /// The physical build of a plan over `blocks` ([`Program::build`]).
    pub fn build_with(
        blocks: Vec<Box<dyn Block>>,
        floorplan: Floorplan,
        plan: Plan,
    ) -> Result<Program, String> {
        plan.validate()?;
        let mut fb = FloorBuilder::default();
        for b in &blocks {
            b.floor(&plan, &mut fb)?;
        }
        let floor = fb.build()?;
        // The program and its memcpy halo on a system's wafer (the largest fabric there is).
        let ((fw, fh), _) = memcpy_fabric(floor.w, floor.h);
        let (sw, sh) = Arch::Wse3.system_fabric();
        if fw > sw || fh > sh {
            return Err(format!(
                "{} x {} PEs (with the memcpy halo {fw} x {fh}): over the wafer's {sw} x {sh}",
                floor.w, floor.h
            ));
        }
        let mut wiring = Wiring::default();
        for b in &blocks {
            b.wire(&plan, &floor, &mut wiring)?;
        }
        Program::finish(blocks, floorplan, plan, floor, wiring)
    }

    /// [`Program::build`] with its blocks where `place` puts them: the program's floor and
    /// links as a placement problem ([`crate::place::decode::problem`]), `place`'s solution
    /// applied (every block moved with its links, the nets between them with their blocks),
    /// then settled, coloured and laid as `build` does. The problem's own placement
    /// ([`crate::place::Solution::pinned`]) gives `build`'s program.
    pub fn build_placed(
        plan: Plan,
        place: &dyn Fn(&crate::place::Problem) -> Result<crate::place::Solution, String>,
    ) -> Result<Program, String> {
        plan.validate()?;
        compose::refuse(&plan.shape)?;
        let needs = Needs::of(&layers::of(&plan.shape));
        let floorplan = Floorplan::of(&plan, &Requests::of(&needs))?;
        let blocks = compose::blocks(&plan, KernelSet::of(&needs), &floorplan)?;
        let mut fb = FloorBuilder::default();
        for b in &blocks {
            b.floor(&plan, &mut fb)?;
        }
        let floor = fb.build()?;
        let mut wiring = Wiring::default();
        for b in &blocks {
            b.wire(&plan, &floor, &mut wiring)?;
        }
        let problem = crate::place::decode::problem(&floor, &wiring)?;
        let solution = place(&problem)?;
        let floor = crate::place::decode::floor(&problem, &floor, &solution)?;
        let ((fw, fh), _) = memcpy_fabric(floor.w, floor.h);
        let (sw, sh) = Arch::Wse3.system_fabric();
        if fw > sw || fh > sh {
            return Err(format!(
                "{} x {} PEs (with the memcpy halo {fw} x {fh}): over the wafer's {sw} x {sh}",
                floor.w, floor.h
            ));
        }
        let wiring = problem.apply(&wiring, &solution)?;
        Program::finish(blocks, floorplan, plan, floor, wiring)
    }

    /// [`Program::build`] with its units at `origins` (in visit order: a placer's), the head
    /// and every other block placed against them as their blocks place them, the links between
    /// blocks that do not meet routed ([`crate::place::route`]).
    pub fn build_at(plan: Plan, origins: Vec<(u32, u32)>) -> Result<Program, String> {
        plan.validate()?;
        compose::refuse(&plan.shape)?;
        let needs = Needs::of(&layers::of(&plan.shape));
        let floorplan = Floorplan::of_at(
            &plan,
            &Requests::of(&needs),
            Some(crate::floor::UnitLayout::Explicit { origins }),
        )?;
        let blocks = compose::blocks(&plan, KernelSet::of(&needs), &floorplan)?;
        Program::build_with(blocks, floorplan, plan)
    }

    /// A program's build past its links: the blocks settle, the palette, the fabric, the
    /// kernels and their route tables, the overrides.
    fn finish(
        blocks: Vec<Box<dyn Block>>,
        floorplan: Floorplan,
        plan: Plan,
        mut floor: Floor,
        mut wiring: Wiring,
    ) -> Result<Program, String> {
        // the links blocks state by their waypoints, routed ([`crate::place::route`])
        crate::place::route::fill(&mut floor, &mut wiring)?;
        for b in &blocks {
            b.settle(&plan, &wiring, &mut floor)?;
        }
        let prefs: Vec<_> = blocks.iter().flat_map(|b| b.colours()).collect();
        let (palette, fabric) = Fabric::lay_coloured(&floor, &wiring, prefs)?;
        let program = layers::of(&plan.shape);
        let schedule = Schedule::of(&program, &floorplan);
        let roles = KernelSet::of(&Needs::of(&program));
        let mut prog = Program {
            plan,
            blocks,
            floor,
            wiring,
            palette,
            fabric,
            kernels: Kernels::new(),
            overrides: Overrides::default(),
            floorplan,
            roles,
            schedule,
        };
        prog.kernels = super::kernels::of(&prog)?;
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
