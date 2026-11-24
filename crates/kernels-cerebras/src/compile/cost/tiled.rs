//! The tiled regime's op-derived time: an op runs over the whole image before the next, every PE
//! in lockstep with the weight stream and the all-reduces, so an op costs its busiest PE's work
//! plus what passes through every PE (stream words, keys). Convolutions, group norms, regions,
//! upsampling and attention are priced from their words, elements and hops; each op adds `tiled_op`.

use serde::Serialize;

use super::stream::RegionFeatures;
use crate::compile::stream::grid::RegIn;
use crate::compile::stream::tiled::{TOp, TileProgram};
use crate::tuning;
use poem_ir::ValueId;

/// One op, priced.
#[derive(Clone, Debug, Serialize)]
pub struct OpCost {
    pub op: usize,
    pub what: &'static str,
    /// Its output's grid (h, w).
    pub at: (u32, u32),
    pub cycles: f64,
}

/// A tiled program's ops and its cycles.
#[derive(Clone, Debug, Serialize)]
pub struct TileCost {
    pub ops: Vec<OpCost>,
    pub cycles: f64,
}

/// An all-reduce of `n` words over the array.
pub(crate) fn all_reduce(p: &TileProgram, n: u32) -> f64 {
    f64::from(p.gx + p.gy + 2) * tuning::get().tiled_hop
        + 3.0 * (f64::from(n) * tuning::get().tiled_word + tuning::get().tiled_call)
}

/// A printed region's cycles over a PE's `px` pixels.
pub(crate) fn region(p: &TileProgram, r: usize, px: u32) -> f64 {
    let g = &p.regs[r];
    let calls = f64::from(px * (g.region.width / g.chunk));
    let copies =
        (g.ins.len() + g.outs.len()) as f64 * f64::from(g.chunk) * tuning::get().stream_dsd_elem;
    let row = g.printed.as_ref().map_or(0.0, |pr| {
        RegionFeatures::of(&pr.csl, g.chunk).cycles(g.chunk)
    });
    calls * (tuning::get().tiled_call + copies + row)
}

/// `p`'s time.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn time(p: &TileProgram) -> TileCost {
    let mut ops = Vec::new();
    let mut cycles = 0.0;
    for (i, op) in p.ops.iter().enumerate() {
        let (what, out, c) = match op {
            TOp::In(v) => (
                "in",
                *v,
                f64::from(p.words(*v)) * tuning::get().stream_dsd_elem,
            ),
            TOp::Out(v) => (
                "out",
                *v,
                f64::from(p.words(*v)) * tuning::get().stream_dsd_elem,
            ),
            TOp::Region(r) => {
                let g = &p.regs[*r];
                let out = g.outs[0];
                let l = p.lay_of(out);
                let res = g.ins.iter().filter(|i| matches!(i, RegIn::Res(_))).count() as f64
                    * f64::from(g.region.width)
                    * tuning::get().tiled_stream_word;
                ("region", out, res + region(p, *r, l.px()))
            }
            // a kernel's op: its price its module's ([`crate::kernel::TiledForm`])
            _ => {
                let out = p.io(i).1.first().copied().unwrap_or(ValueId(0));
                match crate::kernel::tiled_form(op) {
                    Some(f) => (f.what, out, (f.price)(p, i)),
                    None => ("unpriced", out, 0.0),
                }
            }
        };
        let c = if matches!(op, TOp::In(_) | TOp::Out(_)) {
            c
        } else {
            c + tuning::get().tiled_op
        };
        let g = p.vals[&out].grid;
        ops.push(OpCost {
            op: i,
            what,
            at: (g[1], g[2]),
            cycles: c,
        });
        cycles += c;
    }
    TileCost { ops, cycles }
}
