//! The tiled regime's op-derived time (GENERIC.md M2d): an op runs over the whole image before
//! the next, every PE in lockstep with the weight stream and the all-reduces, so an op costs its
//! busiest PE's work plus what passes through every PE (the stream's words, the keys):
//!
//! - **A convolution**: its weight blocks' words received ([`STREAM_WORD`] a word, two weights a
//!   word)
//!   and multiplied ([`super::stream::P1_STREAM`] an element, `P` pixels a PE), not
//!   overlapped (the receive is synchronous); its input staged 16-bit and its halos exchanged
//!   (eight phases, each a border's words and its hops); its bias added.
//! - **A group norm**: its input summed twice ([`DSD_ELEM`] an element), two all-reduces (the
//!   array's rows, its last column, the hub's broadcast: a hop's latency each), its gammas and
//!   betas received, a channel's scale and shift ([`SCALAR`] each), the apply's printed region.
//! - **A region**: its printed `row()` ([`RegionFeatures`]) a chunk of a pixel, its residents
//!   received.
//! - **An upsampling**: the copy, a move's words and hops.
//! - **The attention**: every key past every PE: its 16-bit [k | v] received (and forwarded on
//!   the row and column it comes along), widened, and each of a PE's queries' score (a product,
//!   a halving sum), its exponent ([`SCALAR`]s) and its value's accumulation; and a key's turn
//!   through the hub's loop ([`KEY`], [`KEY_HOP`]).
//!
//! Each op but the host's in and out adds [`OP`], fitted on the simulator.

use std::collections::BTreeMap;

use serde::Serialize;

use super::stream::{DSD_ELEM, RegionFeatures};
use crate::stream::grid::RegIn;
use crate::stream::tiled::{TOp, TileProgram};
use poem_ir::ValueId;

/// A wavelet received or sent.
pub const WORD: f64 = 1.0;
/// A wavelet of the weight stream: from the tape's snake up the spine and along every row to
/// every PE at once, the array reading it in lockstep. On the chip (1565, klein's 128² decode,
/// its tape's stamps: each tape PE's own words gone) the 512-channel convolutions read it at
/// 1.088x their price with [`WORD`] (2.91M against 2.68M cycles, 1,180,160 words each, at every
/// resolution alike), the decode's 24.8M words 5.9M cycles past it: 0.17 a word more, the rest
/// the ops' own tails (a norm's apply, priced with the norm, read before the next op's words).
pub const STREAM_WORD: f64 = 1.17;
/// A key's turn through the hub's loop (its row and the last column forwarded by every PE on
/// the way, the hub, the broadcast tree back) on top of its words and its queries' work: a fixed
/// part and a part an array's row and column long. Fitted on two attentions of klein's decode at
/// 512 channels: a 4 x 4 latent's 16 keys over a 32 x 32 array on the simulator (83.3k cycles,
/// 1.9k a key past the price without it) and the 128² decode's 256 keys over 128 x 128 on the
/// chip (1565: ~1.74M cycles between its neighbours' stream words, 3.5k a key past it).
pub const KEY: f64 = 1_350.0;
pub const KEY_HOP: f64 = 8.3;
/// A DSD op's or a call's fixed cycles.
pub const CALL: f64 = 40.0;
/// A wavelet's hop through a router.
pub const HOP: f64 = 2.0;
/// A scalar statement (a channel's scale and shift, an exponent's step).
pub const SCALAR: f64 = 12.0;
/// An op's fixed cycles (the array's lockstep between ops: the slowest PE's last word, the
/// interpreter's setup), fitted to flux2-mini's decode on the simulator (M2d: its prefixes of
/// 2 to 78 ops, 31k to 905k cycles, within 3% with this term and the rest as priced).
pub const OP: f64 = 5_100.0;

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

/// Cycles by output resolution (h, w), and by kind of op.
pub type By = (BTreeMap<(u32, u32), f64>, BTreeMap<&'static str, f64>);

impl TileCost {
    /// The cycles by output resolution (h, w), and by kind of op.
    #[must_use]
    pub fn by(&self) -> By {
        let mut a = BTreeMap::new();
        let mut b = BTreeMap::new();
        for o in &self.ops {
            *a.entry(o.at).or_insert(0.0) += o.cycles;
            *b.entry(o.what).or_insert(0.0) += o.cycles;
        }
        (a, b)
    }
}

/// An all-reduce of `n` words over the array.
pub(crate) fn all_reduce(p: &TileProgram, n: u32) -> f64 {
    f64::from(p.gx + p.gy + 2) * HOP + 3.0 * (f64::from(n) * WORD + CALL)
}

/// A printed region's cycles over a PE's `px` pixels.
pub(crate) fn region(p: &TileProgram, r: usize, px: u32) -> f64 {
    let g = &p.regs[r];
    let calls = f64::from(px * (g.region.width / g.chunk));
    let copies = (g.ins.len() + g.outs.len()) as f64 * f64::from(g.chunk) * DSD_ELEM;
    let row = g.printed.as_ref().map_or(0.0, |pr| {
        RegionFeatures::of(&pr.csl, g.chunk).cycles(g.chunk)
    });
    calls * (CALL + copies + row)
}

/// `p`'s time.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn time(p: &TileProgram) -> TileCost {
    let mut ops = Vec::new();
    let mut cycles = 0.0;
    for (i, op) in p.ops.iter().enumerate() {
        let (what, out, c) = match op {
            TOp::In(v) => ("in", *v, f64::from(p.words(*v)) * DSD_ELEM),
            TOp::Out(v) => ("out", *v, f64::from(p.words(*v)) * DSD_ELEM),
            TOp::Region(r) => {
                let g = &p.regs[*r];
                let out = g.outs[0];
                let l = p.lay_of(out);
                let res = g.ins.iter().filter(|i| matches!(i, RegIn::Res(_))).count() as f64
                    * f64::from(g.region.width)
                    * STREAM_WORD;
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
            c + OP
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
