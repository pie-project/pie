//! Blocks: the unit a role-split program is composed of. A block is one part of the model's
//! dataflow as it sits on the wafer (the units and their chains, a segment kind, the head, the
//! token loop, a storage column ...): it places its PEs, lays its links, declares the colours of
//! the classes it owns, states per-PE param differences and copies its data. A family is a list
//! of blocks ([`super::family::Family::blocks`]); [`super::program::Program::build`] runs each
//! step over the list in order.
//!
//! A block owns its vocabulary: the [`Class`]es, [`super::wiring::Slot`]s and
//! [`super::wiring::NoteKey`]s it lays are consts in its module, and the kernels' rcfg writers
//! read the slots by them. Nothing in the core names a block's.

use super::abi::{Kernels, Overrides, Param, RcfgFn};
use super::fabric::Palette;
use super::floor::{Coord, Floor, FloorBuilder};
use super::plan::{LayerKind, Plan};
use super::program::Program;
use super::role::{Kind, Place};
use super::weights::{Binding, Copies};
use super::wiring::{Class, Wiring};

pub mod band;
pub mod gdn;
pub mod head;
pub mod moe;
pub mod ple;
pub mod segments;
pub mod shared_kv;
pub mod token_loop;
pub mod units;

type R<T> = Result<T, String>;

/// One part of a program. Every step is pure and defaults to nothing.
pub trait Block: Send + Sync {
    /// Places this block's PEs (the units first: later blocks place against them).
    fn floor(&self, _plan: &Plan, _b: &mut FloorBuilder) -> R<()> {
        Ok(())
    }

    /// Lays this block's links.
    fn wire(&self, _plan: &Plan, _floor: &Floor, _w: &mut Wiring) -> R<()> {
        Ok(())
    }

    /// Refines the floor once every link is laid (a block whose PEs are wherever links cross:
    /// Gemma's route-only PEs).
    fn settle(&self, _plan: &Plan, _w: &Wiring, _floor: &mut Floor) -> R<()> {
        Ok(())
    }

    /// The physical colours of the classes this block owns (a later block may recolour one).
    fn colours(&self) -> Vec<(Class, u8)> {
        Vec::new()
    }

    /// Per-PE kernel params over their kind's.
    fn overrides(&self, _prog: &Program, _o: &mut Overrides) -> R<()> {
        Ok(())
    }

    /// This block's copies onto the wafer, in placement order.
    fn bindings(&self, _prog: &Program, _out: &mut Copies) -> R<()> {
        Ok(())
    }
}

/// The colours of every standard block's classes, then `own` (replacing): the palette of a
/// family still laid as one [`Legacy`] block.
pub fn standard_palette(own: &[(Class, u8)]) -> Palette {
    Palette::of(
        [
            units::COLOURS,
            segments::COLOURS,
            gdn::COLOURS,
            head::COLOURS,
            token_loop::COLOURS,
        ]
        .into_iter()
        .flatten()
        .chain(own)
        .copied(),
    )
}

/// The colour of a standard class, if it has one.
pub fn standard_colour(c: Class) -> Option<u8> {
    standard_palette(&[]).map.get(&c).map(|c| c.0)
}

/// A family not yet composed of blocks: its whole floor, wiring, palette, overrides and copies.
/// Phase B ports gpt-oss, Gemma and Gemma 12B onto blocks and removes this.
pub trait Whole: Send + Sync {
    fn floor(&self, plan: &Plan) -> R<Floor>;
    fn wire(&self, plan: &Plan, floor: &Floor) -> R<Wiring>;
    fn palette(&self, plan: &Plan) -> Palette;
    fn overrides(&self, _prog: &Program, _o: &mut Overrides) -> R<()> {
        Ok(())
    }
    fn bindings(&self, prog: &Program) -> R<Vec<Binding>>;
}

/// A [`Whole`] family as the one block of its program.
pub struct Legacy<F: Whole> {
    pub family: F,
    colours: Vec<(Class, u8)>,
}

impl<F: Whole> Legacy<F> {
    pub fn new(family: F, plan: &Plan) -> Legacy<F> {
        let colours = family
            .palette(plan)
            .map
            .into_iter()
            .map(|(k, c)| (k, c.0))
            .collect();
        Legacy { family, colours }
    }
}

impl<F: Whole> Block for Legacy<F> {
    fn floor(&self, plan: &Plan, b: &mut FloorBuilder) -> R<()> {
        *b = FloorBuilder::of(self.family.floor(plan)?);
        Ok(())
    }

    fn wire(&self, plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        *w = self.family.wire(plan, floor)?;
        Ok(())
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        self.colours.clone()
    }

    fn overrides(&self, prog: &Program, o: &mut Overrides) -> R<()> {
        self.family.overrides(prog, o)
    }

    fn bindings(&self, prog: &Program, out: &mut Copies) -> R<()> {
        out.0.extend(self.family.bindings(prog)?);
        Ok(())
    }
}

/// The standard links of a [`Whole`] family's units: the unit chains and handoffs, and for a
/// head with a control column its rows and (under the contract's token loop) the token loop.
pub fn standard_units(plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
    units::unit_chains(plan, floor, w)?;
    if floor.head.ctl_x().is_some() {
        head::head_rows(plan, floor, w)?;
        if plan.contract.token_loop {
            token_loop::token_loop(plan, floor, w)?;
        }
    }
    Ok(())
}

/// A [`Whole`] family's kernels with each kind's route table and the family's rcfg writer.
pub fn legacy_kernels(mut ks: Kernels, slots: fn(Kind) -> usize, rcfg: RcfgFn) -> Kernels {
    for (k, kb) in ks.iter_mut() {
        kb.route_slots = slots(*k);
        if *k != Kind::IDLE {
            kb.rcfg = Some(rcfg);
        }
    }
    ks
}

/// What a [`Whole`] family's roles imply for their kernels' params, as the build stated it
/// before blocks: a row run's dims (`Hc`) on the weight and CX PEs of plans with several runs, a
/// weight PE's `ATT` where `att` says, `SH 1` on the units whose layers all read another layer's
/// KV, a unit's MLP width a column (`Ic`) where the layers' widths differ, `NTW 1` on the store
/// PEs outside every storage block. Only params the PE's kind takes are set.
pub fn legacy_overrides(
    prog: &Program,
    att: &dyn Fn(Coord) -> Option<u32>,
    o: &mut Overrides,
) -> R<()> {
    let plan = &prog.plan;
    let runs = &plan.partition.row_dims;
    let k = plan.partition.layers_per_unit;
    let mlp = &plan.shape.mlp;
    let shared = |u: u32| {
        let ls = &plan.shape.layers;
        let (a, b) = ((u * k) as usize, ((u + 1) * k) as usize);
        b <= ls.len()
            && ls[a..b]
                .iter()
                .all(|l| matches!(l, LayerKind::SharedKv { .. }))
    };
    for (at, role) in prog.floor.roles.iter() {
        let Some(kb) = prog.kernels.get(&role.kind) else {
            continue;
        };
        let has = |n: &str| kb.params.has(n);
        let mut set = |n: &'static str, v: u32| o.set(at, Param(n), v);
        let run = |r: u32| {
            (runs.len() > 1)
                .then(|| runs.iter().find(|x| x.rows.0 <= r && r < x.rows.1))
                .flatten()
        };
        let unit = |u: u32, set: &mut dyn FnMut(&'static str, u32)| {
            if has("SH") && shared(u) {
                set("SH", 1);
            }
            if has("Ic") && !mlp.layer_widths.is_empty() {
                set("Ic", mlp.width(u * k) / plan.partition.cols);
            }
        };
        match role.place {
            Place::Weight { unit: u, row, .. } => {
                if let (Some(x), true) = (run(row), has("Hc")) {
                    set("Hc", x.dims);
                }
                if let (Some(a), true) = (att(at), has("ATT")) {
                    set("ATT", a);
                }
                unit(u, &mut set);
            }
            Place::Cx { row, .. } => {
                if let (Some(x), true) = (run(row), has("Hc")) {
                    set("Hc", x.dims);
                }
            }
            Place::Cy { unit: u, .. } => unit(u, &mut set),
            Place::Store { count: 0, .. }
                if has("NTW") && !prog.floor.stores.iter().any(|b| b.rect.contains(at)) =>
            {
                set("NTW", 1);
            }
            _ => {}
        }
    }
    Ok(())
}
