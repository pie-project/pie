//! A built wafer program: the plan, the blocks it composes, their floor, wiring, palette
//! and laid fabric, every kind's kernel and the per-PE param overrides. The one entry point the
//! host, the tools and the tests use.

use super::abi::{Kernels, Overrides};
use super::block::Block;
use super::compose::{self, Layout, Needs, Schedule};
use super::fabric::{Fabric, Palette};
use super::floor::{Floor, FloorBuilder};
use super::floorplan::{Floorplan, Requests};
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
    /// How the plan's units lie: the blocks' and the role kernels' one choice.
    pub layout: Layout,
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
        let blocks = compose::blocks(&plan, &needs, &floorplan)?;
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
        let mut floor = fb.build()?;
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
        for b in &blocks {
            b.settle(&plan, &wiring, &mut floor)?;
        }
        let palette = Palette::of(blocks.iter().flat_map(|b| b.colours()));
        let fabric = Fabric::lay(&floor, &wiring, &palette)?;
        fabric.validate(&floor)?;
        let program = layers::of(&plan.shape);
        let schedule = Schedule::of(&program);
        let layout = Layout::of(&Needs::of(&program));
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
            layout,
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
