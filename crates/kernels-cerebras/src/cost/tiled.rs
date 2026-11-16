//! The tiled regime's op-derived time (GENERIC.md M2d): an op runs over the whole image before
//! the next, every PE in lockstep with the weight stream and the all-reduces, so an op costs its
//! busiest PE's work plus what passes through every PE (the stream's words, the keys):
//!
//! - **A convolution**: its weight blocks' words received ([`WORD`] a word, two weights a word)
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
//!   a halving sum), its exponent ([`SCALAR`]s) and its value's accumulation.
//!
//! Each op but the host's in and out adds [`OP`], fitted on the simulator.

use std::collections::BTreeMap;

use serde::Serialize;

use super::stream::{DSD_ELEM, P1_STREAM, RegionFeatures};
use crate::stream::grid::RegIn;
use crate::stream::tiled::{TOp, TileProgram};

/// A wavelet received or sent.
pub const WORD: f64 = 1.0;
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
fn all_reduce(p: &TileProgram, n: u32) -> f64 {
    f64::from(p.gx + p.gy + 2) * HOP + 3.0 * (f64::from(n) * WORD + CALL)
}

/// A printed region's cycles over a PE's `px` pixels.
fn region(p: &TileProgram, r: usize, px: u32) -> f64 {
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
            TOp::Conv {
                x,
                y,
                k,
                cin,
                cout,
                bias,
                ..
            } => {
                let l = p.lay_of(*y);
                let px = f64::from(l.px());
                let taps = f64::from(k * k);
                let (cin_, cout_) = (f64::from(*cin), f64::from(*cout));
                let weights = taps * cin_ * cout_;
                let kb = f64::from(p.kb[i].max(1));
                let blocks = taps * cin_ / kb;
                let recv = weights / 2.0 * WORD + blocks * CALL;
                let mac = px * weights * P1_STREAM + blocks * px * CALL;
                let lx = p.lay_of(*x);
                // the input staged a chunk of kc channels at a time, its halo exchanged
                let kc = f64::from(p.kc[i].max(1));
                let chunks = cin_ / kc;
                let stage = if *k == 3 {
                    let (th, tw) = (f64::from(lx.th), f64::from(lx.tw));
                    let fill = (th + 2.0) * (tw + 2.0) * kc * DSD_ELEM;
                    // two 16-bit values a wavelet
                    let ew = 4.0 * (th * kc / 2.0 * WORD + th * CALL)
                        + 4.0 * f64::from(1u32 << lx.lsx) * HOP;
                    let ns = 4.0 * ((tw + 2.0) * kc / 2.0 * WORD + CALL)
                        + 4.0 * f64::from(1u32 << lx.lsy) * HOP;
                    chunks * (fill + ew + ns)
                } else {
                    px * cin_ * DSD_ELEM
                };
                let tail = if bias.is_some() {
                    cout_ * WORD + px * cout_ * DSD_ELEM
                } else {
                    0.0
                } + if p.vals[y].round {
                    2.0 * px * cout_ * DSD_ELEM
                } else {
                    0.0
                };
                ("conv", *y, recv + mac + stage + tail)
            }
            TOp::Norm {
                x, y, groups, reg, ..
            } => {
                let l = p.lay_of(*x);
                let c = f64::from(p.vals[x].width);
                let px = f64::from(l.px());
                let local = px * c * 4.0 * DSD_ELEM + 2.0 * c * SCALAR / 4.0;
                let ar = 2.0 * all_reduce(p, *groups);
                let recv = 2.0 * c * WORD + c * SCALAR;
                ("norm", *y, local + ar + recv + region(p, *reg, l.px()))
            }
            TOp::Region(r) => {
                let g = &p.regs[*r];
                let out = g.outs[0];
                let l = p.lay_of(out);
                let res = g.ins.iter().filter(|i| matches!(i, RegIn::Res(_))).count() as f64
                    * f64::from(g.region.width)
                    * WORD;
                ("region", out, res + region(p, *r, l.px()))
            }
            TOp::Up { x, y, axis, .. } => {
                let (lx, ly) = (p.lay_of(*x), p.lay_of(*y));
                let w = f64::from(p.vals[x].width);
                let copy = f64::from(ly.px()) * w * DSD_ELEM;
                let ls = if *axis == 0 { lx.lsy } else { lx.lsx };
                let mv = if ls > 0 {
                    f64::from(lx.px()) * w * WORD + f64::from(1u32 << (ls - 1)) * HOP + CALL
                } else {
                    0.0
                };
                ("up", *y, copy + mv)
            }
            TOp::Attend { q, o, .. } => {
                let v = &p.vals[q];
                let l = p.lay_of(*q);
                let c = f64::from(v.width);
                let keys = f64::from(v.grid[1] * v.grid[2]);
                // a key: forwarded along its row and column, received off the broadcast (two
                // 16-bit values a wavelet), widened; each query's product, halving sum,
                // exponent, accumulation
                let per = c * WORD
                    + c * WORD
                    + 2.0 * c * DSD_ELEM
                    + f64::from(l.px())
                        * (3.0 * c * DSD_ELEM + c.log2().ceil() * CALL + 3.0 * CALL + 4.0 * SCALAR);
                ("attend", *o, keys * per)
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
