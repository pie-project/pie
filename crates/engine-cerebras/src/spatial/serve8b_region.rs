//! The serve8b_region wafer program (`spatial/csl/serve8b_region`): Qwen3-8B
//! as layer units, and its host side. The head and the embedding strip are
//! serve8b's (`serve8b::head`, `serve8b::embed`); requests and logits
//! are serve2d's, the tables written per unit with that unit's rows.
//!
//! Units, in layer order ([`Units::units`]):
//! - grids: `PX` x `PY`, one layer (qwen2d_paged8b), `Hc` dims a row;
//! - regions: `RPX` x `RPY`, `RNL` layers (qwen2d_region), `RHc` dims a row;
//!   x stays in place between a region's layers.
//!
//! Layer 0 is a grid at (XC, 0), XC = `RC` `RPX`, next to the head; the rest
//! of the top band runs west to (0, 0); `RC` x `RR` regions below the band,
//! serpentine by columns; `NCOL` grids down x = XC from (XC, PY), the last
//! one the last layer.
//!
//! Each PE's place, routes and handoff come from the host's table (`rtab`,
//! [`Tables`]); with `HROUTE` its collectives' routes too (`rcfg`,
//! [`route_words`]). Both go before `init`. A page id names row id % P of a
//! unit of P rows, so the pool is `NB` x the smallest P, and page copies are
//! the host's (`KVCP` 0).

use std::collections::{BTreeMap, HashMap};

use super::ckpt::Ckpt;
use super::device::Device;
use super::program::Layers;
use super::program::ProgramSpec;
use super::serve8b::{Kernel, Rope};
use super::wafer::rect;
use crate::sdk::{DataType, Order};

type R<T> = std::result::Result<T, String>;

const NONE: u32 = 0xff;
const FWD: u32 = 0xffff;
const D_RAMP: u32 = 0;
const D_N: u32 = 1;
const D_S: u32 = 2;
const D_E: u32 = 3;
const D_W: u32 = 4;
const C_R: [u32; 2] = [11, 12];
/// Vertical handoffs: (CV, CV2) by the receiving unit's parity.
const C_V: [[u32; 2]; 2] = [[13, 17], [18, 19]];
/// paths.csl: S (last layer -> head), even and odd rows.
const C_S: [u32; 2] = [14, 15];
/// paths.csl: E (embedding strip -> layer 0), row % 4.
const C_E: [u32; 4] = [0, 1, 4, 13];
/// Re-chunk A: legs by dest row / PX, hops alternate.
const C_A_LEG: [u32; 3] = [0, 1, 20];
const C_A_HOP: [u32; 2] = [2, 3];
/// Re-chunk B: one leg color, hops alternate.
const C_B_LEG: u32 = 3;
const C_B_HOP: [u32; 2] = [4, 14];
/// Re-chunk C (lever D: layer 0's grid into the region west of it): legs by
/// dest row parity, the third where a parity's last leg left the same row
/// (the grid's last rows feed several dest rows); hops C_A_HOP.
const C_C_LEG: [u32; 3] = [19, 20, 17];
const RT_I: usize = 3;
const RT_O: usize = 3;

/// serve8b_region's units (its params past the grids').
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Units {
    pub rpx: usize,
    pub rpy: usize,
    pub rhc: usize,
    pub rdc: usize,
    pub rgs: usize,
    pub ric: usize,
    pub rnl: usize,
    pub rc: usize,
    pub rr: usize,
    pub ncol: usize,
    /// Routes a unit PE's table holds.
    pub rtr: usize,
    /// The Y phase-1 groups of the grids' and the regions' collectives.
    pub gy: usize,
    pub rgy: usize,
    /// The host writes the collectives' routes (`rcfg`).
    pub hroute: bool,
    /// RoPE tables of 4 x 8 (else 3 x 16).
    pub rot8: bool,
    /// The units skip their layers (checking the handoffs): no weights.
    pub nolayer: bool,
    /// Grids in the top band, layer 0's included (XC / PX + 1; 1 under `YD`).
    pub band: usize,
    /// `YD` > 0 (lever D): no band; the regions from row 0 (RC 2), entered
    /// from layer 0's grid to their east, and the last NCOL layers' grids
    /// down the column from (XC, YD) beside the last region (in region row
    /// RR - 2, RR odd, or row 1). 0: option A.
    pub yd: usize,
    /// The eta VM as the block east of the strip (`VMX`, `VMREAL`).
    pub vm: Option<VmBlock>,
}

/// The VM block's image: ops, registers, pool and bank words, and
/// one VM lane a head lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmBlock {
    pub max_ops: usize,
    pub nreg: usize,
    pub pool_words: usize,
    pub bank_words: usize,
    pub lanes: usize,
}

/// The VM block's columns (its hub on the first, at the feeder row).
pub const NVC: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Grid,
    Region,
    /// serve_gptoss's units.
    Attn,
    Moe,
}

/// One layer unit: its kind, place and layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    pub kind: Kind,
    pub idx: usize,
    pub x0: usize,
    pub y0: usize,
    pub k: Kernel,
    pub layers: std::ops::Range<usize>,
}

impl Unit {
    fn nl(&self) -> usize {
        self.layers.len()
    }
}

impl Units {
    pub fn layers(&self) -> usize {
        self.band + self.rc * self.rr * self.rnl + self.ncol
    }

    /// The units in layer order.
    pub fn units(&self, lay: &ProgramSpec) -> Vec<Unit> {
        let grid = Kernel {
            px: lay.px(),
            py: lay.py(),
            hc: lay.hc(),
            dc: lay.dc(),
            gs: lay.gs(),
            ic: lay.ic(),
        };
        let region = Kernel {
            px: self.rpx,
            py: self.rpy,
            hc: self.rhc,
            dc: self.rdc,
            gs: self.rgs,
            ic: self.ric,
        };
        let mut out: Vec<Unit> = Vec::new();
        let mut add = |kind: Kind, x0: usize, y0: usize| {
            let (k, nl) = if kind == Kind::Grid {
                (grid, 1)
            } else {
                (region, self.rnl)
            };
            let l = out.last().map_or(0, |u| u.layers.end);
            out.push(Unit {
                kind,
                idx: out.len(),
                x0,
                y0,
                k,
                layers: l..l + nl,
            });
        };
        let xc = self.rc * self.rpx;
        if self.yd > 0 {
            // Lever D: layer 0 at (XC, 0); the regions (RC 2 x RR) entered at
            // (1, 0); the last NCOL grids down the column from (XC, YD), east
            // of the last region. YD in region row RR - 2 (RR odd): from (1,
            // 0) by rows 0..RR-3 (east-west, then west-east, ...), then (0,
            // RR-2), (0, RR-1), (1, RR-1), (1, RR-2). YD in region row 1 (any
            // RR): down column 0 from (0, 0), back up column 1 to (1, 1).
            add(Kind::Grid, xc, 0);
            let mut cells: Vec<(usize, usize)> = Vec::new();
            if self.yd / self.rpy == 1 && self.rr != 3 {
                cells.push((1, 0));
                cells.extend((0..self.rr).map(|r| (0, r)));
                cells.extend((1..self.rr).rev().map(|r| (1, r)));
            } else {
                for r in 0..self.rr.saturating_sub(2) {
                    if r % 2 == 0 {
                        cells.extend([(1, r), (0, r)]);
                    } else {
                        cells.extend([(0, r), (1, r)]);
                    }
                }
                let (a, b) = (self.rr.saturating_sub(2), self.rr.saturating_sub(1));
                cells.extend([(0, a), (0, b), (1, b), (1, a)]);
            }
            for (c, r) in cells {
                add(Kind::Region, c * self.rpx, r * self.rpy);
            }
            for k in 0..self.ncol.max(1) {
                add(Kind::Grid, xc, self.yd + k * lay.py());
            }
            return out;
        }
        for i in 0..self.band {
            add(Kind::Grid, xc - i * lay.px(), 0);
        }
        for c in 0..self.rc {
            for i in 0..self.rr {
                let r = if c % 2 == 0 { i } else { self.rr - 1 - i };
                add(Kind::Region, c * self.rpx, lay.py() + r * self.rpy);
            }
        }
        for k in 0..self.ncol {
            add(Kind::Grid, xc, lay.py() * (1 + k));
        }
        out
    }

    /// A unit's collectives' Y group.
    fn y_group(&self, u: &Unit) -> usize {
        if u.kind == Kind::Grid {
            self.gy
        } else {
            self.rgy
        }
    }
}

/// One PE's handoff state.
struct Pe {
    routes: BTreeMap<u32, (u32, u32, u32)>,
    recv: Vec<[u32; 3]>,
    send: Vec<[u32; 3]>,
    relay: u32,
    fwd_in: u32,
}

/// Every unit PE's routes, pieces in and out, relay color and forward
/// color.
pub struct Tables {
    pub(crate) units: Vec<Unit>,
    pub(crate) lanes: u32,
    rtr: usize,
    /// Pieces a PE receives (RT_I; serve_gptoss's S2D_RTI).
    pub(crate) rti: usize,
    pe: HashMap<(usize, usize), Pe>,
    /// serve8b_region_pipe's ring: the last unit's x to the head carries
    /// the micro-batch word (`nsend` bit 8).
    pub pipe: bool,
    /// `HVC` 1: a vertical handoff sends row py down column (PX - PY) / 2 + py
    /// (rows centred on the unit's columns) instead of column py, so the
    /// next unit's row multicast reaches its far column sooner.
    centred: bool,
}

fn step(d: u32) -> (isize, isize) {
    match d {
        D_N => (0, -1),
        D_S => (0, 1),
        D_E => (1, 0),
        _ => (-1, 0),
    }
}

fn opp(d: u32) -> u32 {
    match d {
        D_N => D_S,
        D_S => D_N,
        D_E => D_W,
        _ => D_E,
    }
}

/// A source piece: (source fabric y, source offset, dest offset, words a lane).
pub(crate) type Piece = (usize, u32, u32, u32);

impl Tables {
    pub fn new(units: Vec<Unit>, lanes: usize, rtr: usize) -> R<Tables> {
        Tables::new_with(units, lanes, rtr, false)
    }

    /// [`Tables::new`], `centred`: `HVC`'s vertical handoff columns.
    pub fn new_with(units: Vec<Unit>, lanes: usize, rtr: usize, centred: bool) -> R<Tables> {
        let mut t = Tables::empty(units, lanes, rtr);
        t.centred = centred;
        t.build()?;
        Ok(t)
    }

    /// Every unit PE with no routes or pieces yet.
    pub(crate) fn empty(units: Vec<Unit>, lanes: usize, rtr: usize) -> Tables {
        let mut pe = HashMap::new();
        for u in &units {
            for y in u.y0..u.y0 + u.k.py {
                for x in u.x0..u.x0 + u.k.px {
                    let e = Pe {
                        routes: BTreeMap::new(),
                        recv: Vec::new(),
                        send: Vec::new(),
                        relay: NONE,
                        fwd_in: NONE,
                    };
                    pe.insert((x, y), e);
                }
            }
        }
        Tables {
            units,
            lanes: lanes as u32,
            rtr,
            rti: RT_I,
            pe,
            pipe: false,
            centred: false,
        }
    }

    pub(crate) fn set_relay(&mut self, x: usize, y: usize, c: u32) -> R<()> {
        self.at(x, y)?.relay = c;
        Ok(())
    }

    pub(crate) fn set_fwd_in(&mut self, x: usize, y: usize, c: u32) -> R<()> {
        self.at(x, y)?.fwd_in = c;
        Ok(())
    }

    fn at(&mut self, x: usize, y: usize) -> R<&mut Pe> {
        self.pe
            .get_mut(&(x, y))
            .ok_or_else(|| format!("a path leaves the units at ({x}, {y})"))
    }

    pub(crate) fn route(
        &mut self,
        x: usize,
        y: usize,
        color: u32,
        rx: u32,
        t1: u32,
        t2: u32,
    ) -> R<()> {
        let e = self.at(x, y)?;
        let r = (rx, t1, t2);
        match e.routes.get(&color) {
            Some(&old) if old != r => Err(format!("({x}, {y}) color {color}: {old:?} vs {r:?}")),
            _ => {
                e.routes.insert(color, r);
                Ok(())
            }
        }
    }

    /// A point-to-point path through corner points (straight segments):
    /// RAMP in at the first, RAMP out at the last.
    pub(crate) fn leg(&mut self, color: u32, pts: &[(usize, usize)]) -> R<()> {
        let mut cells = vec![pts[0]];
        let mut dirs = Vec::new();
        for w in pts.windows(2) {
            let ((ax, ay), (bx, by)) = (w[0], w[1]);
            if ax != bx && ay != by {
                return Err(format!(
                    "a leg from {:?} to {:?} is not straight",
                    w[0], w[1]
                ));
            }
            let d = if ay == by {
                if bx > ax { D_E } else { D_W }
            } else if by > ay {
                D_S
            } else {
                D_N
            };
            while *cells.last().expect("one cell") != (bx, by) {
                let (cx, cy) = *cells.last().expect("one cell");
                let (sx, sy) = step(d);
                cells.push(((cx as isize + sx) as usize, (cy as isize + sy) as usize));
                dirs.push(d);
            }
        }
        if cells.len() == 1 {
            return Err("a leg to itself".into());
        }
        for (i, &(x, y)) in cells.iter().enumerate() {
            let rx = if i == 0 { D_RAMP } else { opp(dirs[i - 1]) };
            let tx = if i == cells.len() - 1 {
                D_RAMP
            } else {
                dirs[i]
            };
            self.route(x, y, color, rx, tx, tx)?;
        }
        Ok(())
    }

    /// Unit `u`'s row `row` takes `color` at column `xin` (from `rxin`, or
    /// RAMP: that PE multicasts its own) and every PE of the row gets it.
    pub(crate) fn row_mcast(
        &mut self,
        color: u32,
        u: &Unit,
        row: usize,
        xin: usize,
        rxin: u32,
    ) -> R<()> {
        let y = u.y0 + row;
        let (xa, xb) = (u.x0, u.x0 + u.k.px - 1);
        for x in xa..=xb {
            if x == xin {
                let mut outs: Vec<u32> = Vec::new();
                if rxin != D_RAMP {
                    outs.push(D_RAMP);
                }
                if x > xa {
                    outs.push(D_W);
                }
                if x < xb {
                    outs.push(D_E);
                }
                if outs.is_empty() || outs.len() > 2 {
                    return Err(format!("({x}, {y}) multicasts to {outs:?}"));
                }
                self.route(x, y, color, rxin, outs[0], outs[outs.len() - 1])?;
            } else if x < xin {
                self.route(x, y, color, D_E, D_RAMP, if x > xa { D_W } else { D_RAMP })?;
            } else {
                self.route(x, y, color, D_W, D_RAMP, if x < xb { D_E } else { D_RAMP })?;
            }
        }
        Ok(())
    }

    pub(crate) fn recv(&mut self, x: usize, y: usize, color: u32, off: u32, n: u32) -> R<()> {
        self.at(x, y)?.recv.push([color, off, n]);
        Ok(())
    }

    pub(crate) fn send(&mut self, x: usize, y: usize, color: u32, off: u32, n: u32) -> R<()> {
        self.at(x, y)?.send.push([color, off, n]);
        Ok(())
    }

    /// Row `row` of `b` received on `color` by every PE but `except`.
    fn recv_row(&mut self, b: &Unit, row: usize, color: u32, except: Option<usize>) -> R<()> {
        for x in b.x0..b.x0 + b.k.px {
            if Some(x) != except {
                self.recv(x, b.y0 + row, color, 0, b.k.hc as u32)?;
            }
        }
        Ok(())
    }

    fn build(&mut self) -> R<()> {
        let units = self.units.clone();
        let (u0, ul) = (&units[0], &units[units.len() - 1]);
        // Layer 0 takes row py from the east on E[py % 4] (the strip's E).
        for py in 0..u0.k.py {
            let c = C_E[py % 4];
            self.row_mcast(c, u0, py, u0.x0 + u0.k.px - 1, D_E)?;
            self.recv_row(u0, py, c, None)?;
        }
        // The last layer's east column sends row py east on S[py % 2].
        for py in 0..ul.k.py {
            let (x, y) = (ul.x0 + ul.k.px - 1, ul.y0 + py);
            self.route(x, y, C_S[py % 2], D_RAMP, D_E, D_E)?;
            self.send(x, y, C_S[py % 2], 0, ul.k.hc as u32)?;
        }
        for w in units.windows(2) {
            self.handoff(&w[0], &w[1])?;
        }
        Ok(())
    }

    fn handoff(&mut self, a: &Unit, b: &Unit) -> R<()> {
        let r = C_R[b.idx % 2];
        let same = (a.k.px, a.k.py, a.k.hc) == (b.k.px, b.k.py, b.k.hc);
        let (ak, bk) = (a.k, b.k);
        if same && a.y0 == b.y0 && a.x0.abs_diff(b.x0) == ak.px {
            let east = b.x0 > a.x0;
            for py in 0..ak.py {
                let xs = if east { a.x0 + ak.px - 1 } else { a.x0 };
                let d = if east { D_E } else { D_W };
                self.route(xs, a.y0 + py, r, D_RAMP, d, d)?;
                self.send(xs, a.y0 + py, r, 0, ak.hc as u32)?;
                let xin = if east { b.x0 } else { b.x0 + bk.px - 1 };
                self.row_mcast(r, b, py, xin, if east { D_W } else { D_E })?;
                self.recv_row(b, py, r, None)?;
            }
        } else if same && a.x0 == b.x0 && a.y0.abs_diff(b.y0) == ak.py {
            // (HVC: the rows' columns centred when the unit is wider than tall)
            let c0 = if self.centred && ak.px >= ak.py {
                (ak.px - ak.py) / 2
            } else {
                0
            };
            for py in 0..ak.py {
                let c = a.x0 + (c0 + py) % ak.px;
                let cv = C_V[b.idx % 2][usize::from(py >= ak.px)];
                self.leg(cv, &[(c, a.y0 + py), (c, b.y0 + py)])?;
                self.send(c, a.y0 + py, cv, 0, ak.hc as u32)?;
                self.recv(c, b.y0 + py, cv, 0, bk.hc as u32)?;
                self.at(c, b.y0 + py)?.relay = r;
                self.row_mcast(r, b, py, c, D_RAMP)?;
                self.recv_row(b, py, r, Some(c))?;
            }
        } else if a.x0 == b.x0 && b.y0 == a.y0 + ak.py && ak.px <= bk.px {
            self.rechunk_a(a, b, r)?;
        } else if a.kind == Kind::Grid
            && b.kind == Kind::Region
            && a.x0 == b.x0 + bk.px
            && a.y0 == b.y0
        {
            self.rechunk_c(a, b, r)?;
        } else if b.x0 == a.x0 + ak.px
            && b.y0 >= a.y0
            && b.y0 + bk.py <= a.y0 + ak.py
            && ak.px >= bk.py
        {
            self.rechunk_b(a, b, r)?;
        } else {
            return Err(format!(
                "no handoff from unit {} at {:?} to unit {} at {:?}",
                a.idx,
                (a.x0, a.y0),
                b.idx,
                (b.x0, b.y0)
            ));
        }
        Ok(())
    }

    /// Dest row g's pieces: (source row, source offset, dest offset, n).
    pub(crate) fn pieces(a: &Unit, b: &Unit, g: usize) -> Vec<(usize, u32, u32, u32)> {
        let (lo, hi) = (g * b.k.hc, ((g + 1) * b.k.hc).min(a.k.py * a.k.hc));
        let mut out = Vec::new();
        let mut d = lo;
        while d < hi {
            let r = d / a.k.hc;
            let e = hi.min((r + 1) * a.k.hc);
            out.push((r, (d - r * a.k.hc) as u32, (d - lo) as u32, (e - d) as u32));
            d = e;
        }
        out
    }

    /// Pieces `srcs` in dims order chained along column `col` toward the
    /// leg's start (the last source carries everything), then the leg to
    /// `dest`. Down (A): top first, each adds its own after what it
    /// forwards; up (B): bottom first, each sends its own first.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn chain(
        &mut self,
        col: usize,
        srcs: &[Piece],
        hop: [u32; 2],
        leg_color: u32,
        leg_pts: &[(usize, usize)],
        dest: (usize, usize),
        down: bool,
    ) -> R<()> {
        let order: Vec<Piece> = if down {
            srcs.to_vec()
        } else {
            srcs.iter().rev().copied().collect()
        };
        for (i, &(y, so, _, n)) in order.iter().enumerate() {
            let last = i == order.len() - 1;
            let color = if last { leg_color } else { hop[i % 2] };
            let fwd = self.lanes * order[..i].iter().map(|p| p.3).sum::<u32>();
            if i > 0 {
                self.at(col, y)?.fwd_in = hop[(i - 1) % 2];
            }
            if down {
                if i > 0 {
                    self.send(col, y, color, FWD, fwd)?;
                }
                self.send(col, y, color, so, n)?;
            } else {
                self.send(col, y, color, so, n)?;
                if i > 0 {
                    self.send(col, y, color, FWD, fwd)?;
                }
            }
            if !last {
                self.leg(color, &[(col, y), (col, order[i + 1].0)])?;
            }
        }
        for &(_, _, d, n) in srcs {
            self.recv(dest.0, dest.1, leg_color, d, n)?;
        }
        self.leg(leg_color, leg_pts)
    }

    /// A grid above a region: dest row g's pieces chained down column
    /// a.x0 + g mod a.PX, then down to (that column, b's row g), which
    /// multicasts the row.
    pub(crate) fn rechunk_a(&mut self, a: &Unit, b: &Unit, r: u32) -> R<()> {
        for g in 0..b.k.py {
            let p = Tables::pieces(a, b, g);
            if p.is_empty() {
                continue;
            }
            let col = a.x0 + g % a.k.px;
            let srcs: Vec<Piece> = p
                .iter()
                .map(|&(row, so, d, n)| (a.y0 + row, so, d, n))
                .collect();
            let dest = (col, b.y0 + g);
            let last = srcs[srcs.len() - 1].0;
            self.chain(
                col,
                &srcs,
                C_A_HOP,
                C_A_LEG[g / a.k.px],
                &[(col, last), dest],
                dest,
                true,
            )?;
            self.at(dest.0, dest.1)?.relay = r;
            self.row_mcast(r, b, g, col, D_RAMP)?;
            self.recv_row(b, g, r, Some(col))?;
        }
        Ok(())
    }

    /// A region west of a grid lying within its rows: dest row g's pieces
    /// chained up column a.x0 + a.PX - b.PY + g, then up to the dest row's
    /// fabric row and east into the grid's west column, which multicasts it.
    fn rechunk_b(&mut self, a: &Unit, b: &Unit, r: u32) -> R<()> {
        for g in 0..b.k.py {
            let p = Tables::pieces(a, b, g);
            if p.is_empty() {
                continue;
            }
            let col = a.x0 + a.k.px - b.k.py + g;
            let yg = b.y0 + g;
            let srcs: Vec<Piece> = p
                .iter()
                .map(|&(row, so, d, n)| (a.y0 + row, so, d, n))
                .collect();
            let top = srcs[0].0;
            if top < yg {
                return Err("re-chunk B: a source row above its dest row".into());
            }
            let mut pts = vec![(col, top)];
            if top != yg {
                pts.push((col, yg));
            }
            pts.push((b.x0, yg));
            self.chain(col, &srcs, C_B_HOP, C_B_LEG, &pts, (b.x0, yg), false)?;
            self.at(b.x0, yg)?.relay = r;
            self.row_mcast(r, b, g, b.x0, D_RAMP)?;
            self.recv_row(b, g, r, Some(b.x0))?;
        }
        Ok(())
    }

    /// Lever D: a grid east of a region, both from the same row. Dest row
    /// g's pieces chained down grid column a.x0 + g mod a.PX (hops C_A_HOP),
    /// then on C_C_LEG[g % 2] (or [2]) west along the last source's row into the
    /// region, to its column b.x0 + b.PX - ceil(b.PY / 2) + g / 2, and down
    /// to row g, whose PE multicasts the row. A leg's columns increase with
    /// g within its color and its source rows strictly, so a row's westward
    /// leg crosses only columns of later rows, whose legs start below it.
    fn rechunk_c(&mut self, a: &Unit, b: &Unit, r: u32) -> R<()> {
        let mut prev = [usize::MAX; 3];
        for g in 0..b.k.py {
            let p = Tables::pieces(a, b, g);
            if p.is_empty() {
                continue;
            }
            let col = a.x0 + g % a.k.px;
            let srcs: Vec<Piece> = p
                .iter()
                .map(|&(row, so, d, n)| (a.y0 + row, so, d, n))
                .collect();
            let yg = b.y0 + g;
            let last = srcs[srcs.len() - 1].0;
            if last > yg {
                return Err("re-chunk C: a source row below its dest row".into());
            }
            let xg = b.x0 + b.k.px - b.k.py.div_ceil(2) + g / 2;
            let mut pts = vec![(col, last), (xg, last)];
            if last != yg {
                pts.push((xg, yg));
            }
            let dest = (xg, yg);
            let k = if prev[g % 2] == last { 2 } else { g % 2 };
            if prev[k] == last {
                return Err("re-chunk C: three dest rows' legs leave one row".into());
            }
            prev[k] = last;
            self.chain(col, &srcs, C_A_HOP, C_C_LEG[k], &pts, dest, true)?;
            self.at(dest.0, dest.1)?.relay = r;
            self.row_mcast(r, b, g, xg, D_RAMP)?;
            self.recv_row(b, g, r, Some(xg))?;
        }
        Ok(())
    }

    /// Words of a PE's table.
    pub fn rtn(&self) -> usize {
        7 + 4 * self.rtr + 3 * self.rti + 3 * RT_O
    }

    /// Unit `u`'s `rtab`, `PY` x `PX` PEs of [`Tables::rtn`] words (16-bit
    /// values): `[nr, nrecv, nsend, relay, fwd_in, px, py, (color, rx, tx1,
    /// tx2) x RTR, (color, off, n) x 3 in, (color, off, n) x 3 out]`.
    pub fn table(&self, u: &Unit) -> R<Vec<u32>> {
        let rtn = self.rtn();
        let (oi, oo) = (7 + 4 * self.rtr, 7 + 4 * self.rtr + 3 * self.rti);
        let mut t = vec![0u32; u.k.py * u.k.px * rtn];
        for py in 0..u.k.py {
            for px in 0..u.k.px {
                let e = &self.pe[&(u.x0 + px, u.y0 + py)];
                let one_color = |v: &[[u32; 3]]| v.iter().all(|p| p[0] == v[0][0]);
                if e.routes.len() > self.rtr
                    || e.recv.len() > self.rti
                    || e.send.len() > RT_O
                    || !one_color(&e.recv)
                    || !one_color(&e.send)
                {
                    return Err(format!(
                        "unit {} PE ({px}, {py}): routes {:?} in {:?} out {:?} do not fit its table",
                        u.idx, e.routes, e.recv, e.send
                    ));
                }
                // The ring: the next pass's pieces may arrive on the recv
                // color while queue 2 is bound to fwd_in, so the two differ;
                // the last unit's PEs sending x to the head (colors 14, 15)
                // append the micro-batch word after their pieces.
                let mut nsend = e.send.len() as u32;
                if self.pipe {
                    if let Some(r) = e.recv.first()
                        && e.fwd_in != NONE
                        && r[0] == e.fwd_in
                    {
                        return Err(format!(
                            "unit {} PE ({px}, {py}): recv color == fwd_in {}",
                            u.idx, e.fwd_in
                        ));
                    }
                    let last = self.units.last().is_some_and(|l| l.idx == u.idx);
                    if last && e.send.first().is_some_and(|s| C_S.contains(&s[0])) {
                        nsend |= 0x100;
                    }
                }
                let at = (py * u.k.px + px) * rtn;
                t[at..at + 7].copy_from_slice(&[
                    e.routes.len() as u32,
                    e.recv.len() as u32,
                    nsend,
                    e.relay,
                    e.fwd_in,
                    px as u32,
                    py as u32,
                ]);
                for (i, (&c, &(rx, t1, t2))) in e.routes.iter().enumerate() {
                    t[at + 7 + 4 * i..at + 11 + 4 * i].copy_from_slice(&[c, rx, t1, t2]);
                }
                for (i, p) in e.recv.iter().enumerate() {
                    t[at + oi + 3 * i..at + oi + 3 * i + 3].copy_from_slice(p);
                }
                for (i, p) in e.send.iter().enumerate() {
                    t[at + oo + 3 * i..at + oo + 3 * i + 3].copy_from_slice(p);
                }
            }
        }
        Ok(t)
    }
}

/// Each PE's collective route words, 11 a PE (the kernels'
/// route_phase1/phase2/bcast/gbcast run on the host): for
/// (c_r1_0, c_r1_1, c_r2_0, c_r2_1, c_bc) on X, then on Y, then c_bc's
/// group broadcast, each `rb[rx] | rb[5 + tx1] | rb[5 + tx2]`.
pub fn route_words(rb: &[u32], px_n: usize, py_n: usize, gs: usize, gy: usize) -> Vec<u32> {
    route_words_gx(rb, px_n, py_n, gs, gy, 0)
}

/// [`route_words`], and with `gx` > 0 (RGX) 6 more a PE: the X axis's
/// five words in phase-1 groups of `gx`, then log2(gx) (17 a PE).
pub fn route_words_gx(
    rb: &[u32],
    px_n: usize,
    py_n: usize,
    gs: usize,
    gy: usize,
    gx: usize,
) -> Vec<u32> {
    let stride = if gx > 0 { 17 } else { 11 };
    const RAMP: usize = 0;
    const N_: usize = 1;
    const S_: usize = 2;
    const E_: usize = 3;
    const W_: usize = 4;
    let word = |(rx, t1, t2): (usize, usize, usize)| rb[rx] | rb[5 + t1] | rb[5 + t2];
    let mut out = vec![0u32; py_n * px_n * stride];
    for py in 0..py_n {
        for px in 0..px_n {
            let at = (py * px_n + px) * stride;
            let axes = [
                (W_, E_, px, px_n, gs),
                (N_, S_, py, py_n, gy),
                (W_, E_, px, px_n, gx),
            ];
            for (ax, (h, t, idv, alen, ga)) in
                axes.into_iter()
                    .enumerate()
                    .take(if gx > 0 { 3 } else { 2 })
            {
                let mut cfg = [(0, 0, 0); 5];
                let r1 = ga / 2;
                let ng = alen / ga;
                let r2 = (ng / 2) * ga + r1;
                let r = idv % ga;
                let (ca, cb) = if r % 2 == 0 { (0, 1) } else { (1, 0) };
                if r == r1 {
                    (cfg[ca], cfg[cb]) = ((h, RAMP, RAMP), (t, RAMP, RAMP));
                } else if r < r1 {
                    (cfg[ca], cfg[cb]) = ((h, RAMP, RAMP), (RAMP, t, t));
                } else {
                    (cfg[cb], cfg[ca]) = ((t, RAMP, RAMP), (RAMP, h, h));
                }
                let qt = idv / ga;
                let (ca, cb) = if qt % 2 == 0 { (2, 3) } else { (3, 2) };
                if idv == r2 {
                    (cfg[ca], cfg[cb]) = ((h, RAMP, RAMP), (t, RAMP, RAMP));
                } else if idv < r2 {
                    if r == r1 {
                        (cfg[ca], cfg[cb]) = ((h, RAMP, RAMP), (RAMP, t, t));
                    } else {
                        cfg[2] = (h, t, t);
                        cfg[3] = (h, t, t);
                    }
                } else if r == r1 {
                    (cfg[cb], cfg[ca]) = ((t, RAMP, RAMP), (RAMP, h, h));
                } else {
                    cfg[2] = (t, h, h);
                    cfg[3] = (t, h, h);
                }
                cfg[4] = if idv == r2 {
                    (RAMP, h, t)
                } else if idv < r2 {
                    if idv == 0 {
                        (t, RAMP, RAMP)
                    } else {
                        (t, h, RAMP)
                    }
                } else if idv == alen - 1 {
                    (h, RAMP, RAMP)
                } else {
                    (h, RAMP, t)
                };
                for (c, &f) in cfg.iter().enumerate() {
                    out[at + if ax == 2 { 11 } else { 5 * ax } + c] = word(f);
                }
            }
            let (r, r1) = (px % gs, gs / 2);
            let g = if r == r1 {
                (RAMP, W_, if r1 == gs - 1 { W_ } else { E_ })
            } else if r < r1 {
                if r == 0 {
                    (E_, RAMP, RAMP)
                } else {
                    (E_, W_, RAMP)
                }
            } else if r == gs - 1 {
                (W_, RAMP, RAMP)
            } else {
                (W_, RAMP, E_)
            };
            out[at + 10] = word(g);
            if gx > 0 {
                out[at + 16] = gx.trailing_zeros();
            }
        }
    }
    out
}

/// ARF's routes and role on one line of `n` PEs (rows of a column, or columns
/// of a row) at place `k`: the five colours' (rx, tx1, tx2) with directions
/// 0 ramp, 1 away from the root, 2 toward it, 3 along the line both ways from
/// the root (the broadcast); and the role word (see [`route_words_arf`]).
#[allow(clippy::type_complexity)]
fn arf_line(n: usize, k: usize, arg: usize) -> ([Option<(usize, usize, usize)>; 5], u32, bool) {
    const RAMP: usize = 0;
    const AWY: usize = 1;
    const TOW: usize = 2;
    let mut cfg = [None; 5];
    let root = n / 2;
    if k == root {
        // (the root's line ends: AWY = the first half's side, TOW = the second's)
        cfg[2] = Some((AWY, RAMP, RAMP));
        cfg[3] = Some((TOW, RAMP, RAMP));
        cfg[4] = Some((RAMP, AWY, TOW));
        return (cfg, 1, true);
    }
    let first = k < root;
    let half = usize::from(!first);
    let d = if first { root - k } else { k - root };
    let dmax = if first { root } else { n - 1 - root };
    let nq = dmax.div_ceil(arg);
    let q = (d - 1) / arg;
    let kk = d - q * arg;
    let far = q + 1 == nq;
    let gsz = if far { dmax - q * arg } else { arg };
    let (r1, s2) = (d % 2, (q + half) % 2);
    let r2 = 1 - s2;
    if kk < gsz {
        cfg[r1] = Some((AWY, RAMP, RAMP));
    }
    if kk > 1 {
        cfg[1 - r1] = Some((RAMP, TOW, TOW));
    }
    if kk == 1 {
        cfg[2 + s2] = Some((RAMP, TOW, TOW));
        if !far {
            cfg[2 + r2] = Some((AWY, RAMP, RAMP));
        }
    } else if !far {
        cfg[2 + r2] = Some((AWY, TOW, TOW));
    }
    let edge = k == 0 || k == n - 1;
    cfg[4] = Some((TOW, RAMP, if edge { RAMP } else { AWY }));
    let role = (u32::from(far) << 1)
        | ((r1 as u32) << 2)
        | ((s2 as u32) << 3)
        | ((kk as u32) << 4)
        | ((gsz as u32) << 12);
    (cfg, role, false)
}

/// ARF's words a region PE: [`route_words`]' 11 with the Y set replaced by
/// the streamed-halves routes (docs/cerebras-8b-speed.md 9.5 / 10), then the
/// PE's Y role: bit 0 the root (row py_n / 2), bit 1 in its half's farthest
/// group, bit 2 its level-1 input colour c_r1[k % 2], bit 3 its level-2 output
/// colour c_r2[(q + half) % 2], bits 4..12 its 1-based place in its group
/// (1: the group's end, next to the root's side), bits 12..20 the group's size
/// (12 words a PE). `arg`: rows a group, a power of two. `argx` > 0 (ARF 3):
/// also the X axis's full all-reduces' set and role, the same streamed halves
/// along the row (root column px_n / 2, groups of `argx` columns): words
/// 12..17 and 17 (18 words a PE). `seg` (ARF bit 4): the X set (words 0..5)
/// as streamed halves on each segment of `gs` columns (root at its column
/// gs / 2, groups of 8) and its role in word 10 (the group broadcast's word).
#[allow(clippy::too_many_arguments)]
pub fn route_words_arf(
    rb: &[u32],
    px_n: usize,
    py_n: usize,
    gs: usize,
    gy: usize,
    arg: usize,
    argx: usize,
    seg: bool,
) -> Vec<u32> {
    const N_: usize = 1;
    const S_: usize = 2;
    const E_: usize = 3;
    const W_: usize = 4;
    let stride = if argx > 0 { 18 } else { 12 };
    let base = route_words(rb, px_n, py_n, gs, gy);
    let mut out = vec![0u32; py_n * px_n * stride];
    // One line's abstract routes as this PE's route words: `side` maps the
    // line's directions (1 away, 2 toward; the root: 1 first half, 2 second).
    let put = |out: &mut [u32],
               at: usize,
               n: usize,
               k: usize,
               a: usize,
               (lo, hi): (usize, usize)|
     -> u32 {
        let (cfg, role, root) = arf_line(n, k, a);
        let first = k < n / 2;
        let dir = |d: usize| match (d, root, first) {
            (0, _, _) => 0,
            (1, true, _) => lo,
            (2, true, _) => hi,
            (1, false, true) | (2, false, false) => lo,
            _ => hi,
        };
        for (c, f) in cfg.iter().enumerate() {
            out[at + c] = f.map_or(0, |(rx, t1, t2)| {
                rb[dir(rx)] | rb[5 + dir(t1)] | rb[5 + dir(t2)]
            });
        }
        if root {
            out[at + 4] |= rb[5]; // the root's loop-back
        }
        role
    };
    for py in 0..py_n {
        for px in 0..px_n {
            let (at, b) = ((py * px_n + px) * stride, (py * px_n + px) * 11);
            out[at..at + 11].copy_from_slice(&base[b..b + 11]);
            out[at + 11] = put(&mut out, at + 5, py_n, py, arg, (N_, S_));
            if argx > 0 {
                out[at + 17] = put(&mut out, at + 12, px_n, px, argx, (W_, E_));
            }
            if seg {
                out[at + 10] = put(&mut out, at, gs, px % gs, 8, (W_, E_));
            }
        }
    }
    out
}

/// Every unit's `rcfg` (with `HROUTE`, from the program's `rbits`) and
/// `rtab`: before `init`.
pub fn put_routes(
    lay: &ProgramSpec,
    u: &Units,
    dev: &mut dyn Device,
    rbits: Option<&[u32]>,
) -> R<()> {
    let units = u.units(lay);
    if u.hroute {
        let rb = rbits
            .filter(|b| b.len() >= 10)
            .ok_or("HROUTE 1 without the program's rbits")?;
        for v in &units {
            // (RGX: the regions' third X route set)
            let gx = if v.kind == Kind::Grid {
                0
            } else {
                lay.model.rgx
            };
            if gx > 0 && (!gx.is_power_of_two() || v.k.px % gx != 0 || gx > v.k.px / 2) {
                return Err(format!(
                    "RGX {gx} does not split the region's {} columns",
                    v.k.px
                ));
            }
            // (ARF: the regions' Y all-reduces as streamed halves, groups of 8)
            // (GRD bit 1: the grids' too)
            let arf = (v.kind != Kind::Grid || lay.model.grid & 1 != 0) && lay.model.arf != 0;
            if arf && gx > 0 {
                return Err("ARF with RGX: not built".into());
            }
            let argx = if lay.model.arf & 2 != 0 { 16 } else { 0 };
            // (ARY: the Y groups' rows, 8 by default; 0495: 16 is ~15 cycles faster at 208 rows)
            let ary = if lay.model.ary > 0 { lay.model.ary } else { 8 };
            // (ARF bit 4: the segment all-reduce as streamed halves on each segment too)
            let seg = lay.model.arf & 4 != 0;
            if seg && v.k.px % v.k.gs != 0 {
                return Err(format!(
                    "ARF 7: segments of {} do not split {} columns",
                    v.k.gs, v.k.px
                ));
            }
            let (words, n) = if arf {
                (
                    route_words_arf(rb, v.k.px, v.k.py, v.k.gs, u.y_group(v), ary, argx, seg),
                    if argx > 0 { 18 } else { 12 },
                )
            } else {
                (
                    route_words_gx(rb, v.k.px, v.k.py, v.k.gs, u.y_group(v), gx),
                    if gx > 0 { 17 } else { 11 },
                )
            };
            dev.put("rcfg", rect(v.x0, v.y0, v.k.px, v.k.py), n, &words)?;
        }
    }
    let mut tables = Tables::new_with(
        units.clone(),
        lay.paged().map_or(1, |s| s.lanes),
        u.rtr,
        lay.model.hvc,
    )?;
    tables.pipe = lay.mmax().is_some();
    for v in &units {
        dev.put16(
            "rtab",
            rect(v.x0, v.y0, v.k.px, v.k.py),
            tables.rtn(),
            &tables.table(v)?,
        )?;
    }
    Ok(())
}

/// Every layer's weights (and zeroed KV), unit by unit (`_unit`: a
/// region's layers stacked in each symbol).
pub(crate) fn place_layers(
    lay: &ProgramSpec,
    u: &Units,
    ckpt: &Ckpt,
    rope: &super::wafer::Rotary,
    dev: &mut dyn Device,
) -> R<()> {
    let inv_freq = rope.inv_freq(lay.gs() * lay.dc())?;
    let rope = if u.rot8 {
        Rope {
            units: &[512, 64, 8, 1],
            len: 8,
            inv_freq: &inv_freq,
        }
    } else {
        Rope {
            units: &[256, 16, 1],
            len: 16,
            inv_freq: &inv_freq,
        }
    };
    if !u.nolayer {
        for v in u.units(lay) {
            let layers: Vec<usize> = v.layers.clone().collect();
            let qkn = (v.kind == Kind::Grid && lay.model.grid & 4 == 0) || lay.model.mem & 2 == 0;
            super::serve8b::unit(
                lay,
                ckpt,
                v.k,
                &layers,
                (v.x0, v.y0),
                &rope,
                false,
                qkn,
                dev,
            )?;
        }
    }
    Ok(())
}

/// `copy_kv`'s whole pages by the host: for every unit, the source and
/// destination rows' `kc`/`vc` read, each of its layers' block patched
/// (block b of layer l at `1 + l NB Dc PS + b Dc PS`), the destination row
/// written back. Only between launches. A run of pairs `(s + i, d + i)`
/// whose rows do not wrap (one block each side: an HV build's PIE page is
/// 16 kernel pages on 16 consecutive rows) moves as one read of its source
/// rows, one of its destination rows and one write: three row-rect memcpys
/// a unit and array instead of three a kernel page. (HV builds, PS 1;
/// other page sizes copy a pair at a time.)
pub fn copy_pages(
    lay: &ProgramSpec,
    u: &Units,
    rt: &mut dyn Device,
    pairs: &[(u32, u32)],
) -> R<()> {
    let shape = lay
        .paged()
        .ok_or("serve8b_region without its paged shape")?;
    if shape.page_size != 1 {
        // (pages of PIE's size: a pair at a time, the pinned protocol)
        return copy_pages_each(lay, u, rt, pairs);
    }
    let e = |e: crate::sdk::Error| e.to_string();
    let units = u.units(lay);
    if let Some(&(src, dst)) = pairs
        .iter()
        .find(|&&(s, d)| s as usize >= shape.pages() || d as usize >= shape.pages())
    {
        return Err(format!(
            "page copy {src} -> {dst} outside a pool of {}",
            shape.pages()
        ));
    }
    let pairs: Vec<(usize, usize)> = pairs
        .iter()
        .filter(|&&(s, d)| s != d)
        .map(|&(s, d)| (s as usize, d as usize))
        .collect();
    for v in &units {
        let p = v.k.py;
        let n = shape.page_size * v.k.dc;
        let kcn = shape.blocks * n;
        let len = v.nl() * kcn + 1;
        let rows = |r: usize, k: usize| rect(v.x0, v.y0 + r, v.k.px, k);
        let mut i = 0;
        while i < pairs.len() {
            let (s0, d0) = pairs[i];
            let (rs, bs, rd, bd) = (s0 % p, s0 / p, d0 % p, d0 / p);
            let mut k = 1;
            while i + k < pairs.len()
                && pairs[i + k] == (s0 + k, d0 + k)
                && rs + k < p
                && rd + k < p
            {
                k += 1;
            }
            // Row ranges that overlap without being the same: a pair at a time.
            if rs != rd && rs.max(rd) < rs.min(rd) + k {
                k = 1;
            }
            for sym in ["kc", "vc"] {
                let mut a = vec![0u32; v.k.px * k * len];
                rt.memcpy_d2h(
                    sym,
                    &mut a,
                    rows(rs, k),
                    len,
                    Order::RowMajor,
                    DataType::Bits16,
                )
                .map_err(e)?;
                let mut d = if rd == rs {
                    a.clone()
                } else {
                    let mut d = vec![0u32; v.k.px * k * len];
                    rt.memcpy_d2h(
                        sym,
                        &mut d,
                        rows(rd, k),
                        len,
                        Order::RowMajor,
                        DataType::Bits16,
                    )
                    .map_err(e)?;
                    d
                };
                for (s, t) in a.chunks_exact(len).zip(d.chunks_exact_mut(len)) {
                    for l in 0..v.nl() {
                        let o = 1 + l * kcn;
                        t[o + bd * n..o + (bd + 1) * n]
                            .copy_from_slice(&s[o + bs * n..o + (bs + 1) * n]);
                    }
                }
                rt.memcpy_h2d(sym, &d, rows(rd, k), len, Order::RowMajor, DataType::Bits16)
                    .map_err(e)?;
            }
            i += k;
        }
    }
    Ok(())
}

/// [`copy_pages`] a pair at a time (every unit's rows per pair).
fn copy_pages_each(
    lay: &ProgramSpec,
    u: &Units,
    rt: &mut dyn Device,
    pairs: &[(u32, u32)],
) -> R<()> {
    let shape = lay
        .paged()
        .ok_or("serve8b_region without its paged shape")?;
    let e = |e: crate::sdk::Error| e.to_string();
    let units = u.units(lay);
    for &(src, dst) in pairs {
        let (src, dst) = (src as usize, dst as usize);
        if src >= shape.pages() || dst >= shape.pages() {
            return Err(format!(
                "page copy {src} -> {dst} outside a pool of {}",
                shape.pages()
            ));
        }
        if src == dst {
            continue;
        }
        for v in &units {
            let p = v.k.py;
            let n = shape.page_size * v.k.dc;
            let kcn = shape.blocks * n;
            let len = v.nl() * kcn + 1;
            let (rs, bs, rd, bd) = (src % p, src / p, dst % p, dst / p);
            let row = |r: usize| rect(v.x0, v.y0 + r, v.k.px, 1);
            for sym in ["kc", "vc"] {
                let mut a = vec![0u32; v.k.px * len];
                rt.memcpy_d2h(sym, &mut a, row(rs), len, Order::RowMajor, DataType::Bits16)
                    .map_err(e)?;
                let mut d = if rd == rs {
                    a.clone()
                } else {
                    let mut d = vec![0u32; v.k.px * len];
                    rt.memcpy_d2h(sym, &mut d, row(rd), len, Order::RowMajor, DataType::Bits16)
                        .map_err(e)?;
                    d
                };
                for (s, t) in a.chunks_exact(len).zip(d.chunks_exact_mut(len)) {
                    for l in 0..v.nl() {
                        let o = 1 + l * kcn;
                        t[o + bd * n..o + (bd + 1) * n]
                            .copy_from_slice(&s[o + bs * n..o + (bs + 1) * n]);
                    }
                }
                rt.memcpy_h2d(sym, &d, row(rd), len, Order::RowMajor, DataType::Bits16)
                    .map_err(e)?;
            }
        }
    }
    Ok(())
}

/// serve8b_region_pipe: every unit's `mtab`, micro-batch m's table in slot
/// m of each of the unit's rows (`table_of(rows)`: `tables` for a unit of
/// `rows` rows, one per micro-batch; unused slots zero), written by row
/// broadcast.
pub fn put_pipe_tables(
    lay: &ProgramSpec,
    rt: &mut dyn Device,
    table_of: &dyn Fn(usize) -> R<Vec<Vec<u32>>>,
) -> R<()> {
    let (shape, slots) = match (lay.paged(), lay.mmax(), &lay.layers) {
        (Some(s), Some(m), Layers::Units { .. }) => (s, m),
        _ => return Err("micro-batch tables on a build without serve8b_region_pipe's ring".into()),
    };
    let w = shape.row_words();
    let mut made: Vec<(usize, Vec<u32>)> = Vec::new();
    for g in lay.grid_areas() {
        let rows = g.h as usize;
        if !made.iter().any(|t| t.0 == rows) {
            let tables = table_of(rows)?;
            if tables.is_empty() || tables.len() > slots {
                return Err(format!(
                    "{} micro-batch(es) for a ring of {slots}",
                    tables.len()
                ));
            }
            made.push((rows, pipe_rows(&tables, rows, slots, w)));
        }
        let mt = &made.iter().find(|t| t.0 == rows).expect("built above").1;
        rt.memcpy_h2d_rowbcast("mtab", mt, g, slots * w, DataType::Bits32)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// A unit's `mtab` rows: row y holds slot m's `w` words, micro-batch m's
/// table row y (`tables[m]`, `rows` rows of `w`), zero past the tables.
fn pipe_rows(tables: &[Vec<u32>], rows: usize, slots: usize, w: usize) -> Vec<u32> {
    let mut mt = Vec::with_capacity(rows * slots * w);
    for y in 0..rows {
        for m in 0..slots {
            match tables.get(m) {
                Some(t) => mt.extend_from_slice(&t[y * w..(y + 1) * w]),
                None => mt.extend(std::iter::repeat_n(0, w)),
            }
        }
    }
    mt
}

/// serve8b_region_pipe's head check after a launch: each feeder PE's
/// `perr` (mismatches, first expected, first got) between the micro-batch
/// it expected and the index word that came with x from the last unit. Any
/// mismatch is an error: tokens of one micro-batch taken as another's.
pub fn check_pipe(lay: &ProgramSpec, rt: &mut dyn Device) -> R<()> {
    let c = lay.chunk_cols();
    let mut pe = vec![0u32; c * 3];
    rt.memcpy_d2h(
        "perr",
        &mut pe,
        rect(lay.head_x(), lay.feeder(), c, 1),
        3,
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(|e| e.to_string())?;
    let bad: Vec<(usize, &[u32; 3])> = pe
        .as_chunks::<3>()
        .0
        .iter()
        .enumerate()
        .filter(|(_, r)| r[0] != 0)
        .collect();
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "micro-batch misalignment at the head's feeder (column: [mismatches, first expected, first got]): {bad:?}"
        ))
    }
}

/// One VM lane's program: its table and its homes' seeds, the logits rows
/// it reads a turn.
pub struct VmLane<'a> {
    pub table: &'a super::guest4::Table,
    pub seeds: &'a [super::guest4::wafer::Seed],
    pub rows: u32,
}

/// The VM block's first column (XV: east of the strip).
pub fn vm_x(lay: &ProgramSpec) -> usize {
    lay.head_x() + lay.chunk_cols() + lay.vmc() + lay.py() / 4
}

/// A lane's pool block and wide rows (region row x column x register x Vt,
/// f32) from its seeds.
fn seeded(
    lay: &ProgramSpec,
    vm: &VmBlock,
    seeds: &[super::guest4::wafer::Seed],
) -> (Vec<u32>, Vec<f32>) {
    let (hh, vr) = (lay.hh(), lay.vr());
    let vt = vr / NVC;
    let mut pool = vec![0u32; vm.pool_words];
    let mut wide = vec![0f32; hh * NVC * vm.nreg * vt];
    for s in seeds {
        let (at, n) = (s.at as usize, s.n as usize);
        if s.kind == "pool" {
            for (k, &w) in s.words.iter().take(n).enumerate() {
                if let Some(d) = pool.get_mut(at + k) {
                    *d = w;
                }
            }
        } else {
            for (v, &w) in s.words.iter().take(hh * vr).enumerate() {
                let x = match s.dtype {
                    eta_ir::Dtype::F32 => f32::from_bits(w),
                    eta_ir::Dtype::I32 => w as i32 as f32,
                    _ => w as f32,
                };
                let (q, t) = (v / vt, v % vt);
                wide[(q * vm.nreg + at) * vt + t] = x;
            }
        }
    }
    (pool, wide)
}

/// Column `j`'s wide rows over the VM rows (the region rows around the
/// feeder row F, which stays zero): (HH + 1) x register x Vt f32 bits.
fn wide_column(lay: &ProgramSpec, vm: &VmBlock, wide: &[f32], j: usize) -> Vec<u32> {
    let (hh, f, vt) = (lay.hh(), lay.feeder(), lay.vr() / NVC);
    let per = vm.nreg * vt;
    let mut out = vec![0u32; (hh + 1) * per];
    for ry in 0..hh {
        let y = if ry < f { ry } else { ry + 1 };
        let q = ry * NVC + j;
        for (k, x) in wide[q * per..(q + 1) * per].iter().enumerate() {
            out[y * per + k] = x.to_bits();
        }
    }
    out
}

/// A lane's bank window: (offset, pool ranges, wide registers).
type BankWindow = (usize, Vec<(u32, u32)>, Vec<u32>);

/// Writes lane b's program `lanes[b]` (`None`: an empty lane, its `glane`
/// zeroed, the VM skips its turns) to every VM PE and sets `gmode` 1 over
/// the VM block and the head: one shared op arena
/// (each distinct table once), lane 0's seeds as the live pool and wide
/// registers, each lane's bank window (its live pool ranges, at most 8,
/// and wide registers) in `gbank` when there are several lanes.
#[allow(clippy::too_many_lines)]
pub fn vm_load(lay: &ProgramSpec, dev: &mut dyn Device, lanes: &[Option<VmLane>]) -> R<()> {
    let Layers::Units {
        units: Units { vm: Some(vm), .. },
        ..
    } = lay.layers
    else {
        return Err("this serve8b_region build has no eta VM (VMX 1, VMREAL 1)".into());
    };
    let (hh, vt) = (lay.hh(), lay.vr() / NVC);
    let nv = hh + 1;
    let n_lanes = lanes.len();
    if n_lanes != vm.lanes {
        return Err(format!("{n_lanes} VM lanes for an image of {}", vm.lanes));
    }
    let live: Vec<&VmLane> = lanes.iter().flatten().collect();
    let first = live.first().ok_or("no live VM lane")?;
    let mut arena: Vec<[u32; 6]> = Vec::new();
    let mut bases: Vec<(&[[u32; 6]], u32)> = Vec::new();
    for l in &live {
        if !bases.iter().any(|b| b.0 == l.table.ins.as_slice()) {
            bases.push((&l.table.ins, arena.len() as u32));
            arena.extend_from_slice(&l.table.ins);
        }
        if l.table.nreg as usize > vm.nreg || l.table.pool_words as usize > vm.pool_words {
            return Err(format!(
                "a program of {} registers and {} pool words for an image of {} and {}",
                l.table.nreg, l.table.pool_words, vm.nreg, vm.pool_words
            ));
        }
    }
    let ops = super::guest4::wafer::pack_ops(&arena, vm.max_ops)?;
    let base_of = |l: &VmLane| {
        bases
            .iter()
            .find(|b| b.0 == l.table.ins.as_slice())
            .map_or(0, |b| b.1)
    };
    let mut glane = vec![0u32; n_lanes * 4];
    let mut gbdesc = vec![0u32; n_lanes * 8];
    // Each lane's bank window: its offset, pool ranges and wide registers.
    let mut windows: Vec<Option<BankWindow>> = Vec::new();
    let mut boff = 0usize;
    for (l, lane) in lanes.iter().enumerate() {
        let Some(lane) = lane else {
            windows.push(None);
            continue;
        };
        let t = lane.table;
        let mut rg: Vec<(u32, u32)> = t.bank_pool.iter().copied().filter(|r| r.1 > 0).collect();
        while rg.len() > 8 {
            let k = (0..rg.len() - 1)
                .min_by_key(|&i| rg[i + 1].0 as i64 - rg[i].0 as i64 - rg[i].1 as i64)
                .unwrap_or(0);
            let merged = (rg[k].0, rg[k + 1].0 + rg[k + 1].1 - rg[k].0);
            rg.splice(k..k + 2, [merged]);
        }
        for (k, &(o, n)) in rg.iter().enumerate() {
            gbdesc[l * 8 + k] = o | n << 16;
        }
        let rows = lane.rows.max(1);
        let mask = t.bank_regs.iter().fold(0u32, |m, &x| m | 1 << x);
        glane[l * 4..l * 4 + 4].copy_from_slice(&[
            base_of(lane) | rows << 16,
            t.tok_home.unwrap_or(0),
            boff as u32,
            mask,
        ]);
        let words = rg.iter().map(|r| r.1 as usize).sum::<usize>() + t.bank_regs.len() * vt;
        windows.push(Some((boff, rg, t.bank_regs.clone())));
        boff += words;
    }
    if n_lanes > 1 && boff > vm.bank_words {
        return Err(format!(
            "the lanes' bank windows take {boff} words of {}",
            vm.bank_words
        ));
    }
    let states: Vec<Option<(Vec<u32>, Vec<f32>)>> = lanes
        .iter()
        .map(|l| l.as_ref().map(|l| seeded(lay, &vm, l.seeds)))
        .collect();
    let (pool0, wide0) = seeded(lay, &vm, first.seeds);
    let th = first.table.tok_home.unwrap_or(0);
    let tile = |w: &[u32]| -> Vec<u32> { (0..nv).flat_map(|_| w.iter().copied()).collect() };
    let xv = vm_x(lay);
    for j in 0..NVC {
        let col = rect(xv + j, 0, 1, nv);
        dev.put("gops", col, vm.max_ops * 4, &tile(&ops))?;
        dev.put("gpool", col, vm.pool_words, &tile(&pool0))?;
        dev.put("gvalid", col, 1, &vec![lay.vocab() as u32; nv])?;
        dev.put("gtok_home", col, 1, &vec![th; nv])?;
        dev.put("gostep", col, 1, &vec![0; nv])?;
        dev.put(
            "gwide",
            col,
            vm.nreg * vt,
            &wide_column(lay, &vm, &wide0, j),
        )?;
        dev.put("glane", col, n_lanes * 4, &tile(&glane))?;
        dev.put("gbdesc", col, n_lanes * 8, &tile(&gbdesc))?;
        if n_lanes > 1 {
            let mut bank = vec![0u32; nv * vm.bank_words];
            for (st, w) in states.iter().zip(&windows) {
                let (Some((pool, wide)), Some((bo, rg, regs))) = (st, w) else {
                    continue;
                };
                let lc = wide_column(lay, &vm, wide, j);
                let per = vm.nreg * vt;
                for y in 0..nv {
                    let row = &mut bank[y * vm.bank_words..(y + 1) * vm.bank_words];
                    let mut at = *bo;
                    for &(o, n) in rg {
                        row[at..at + n as usize]
                            .copy_from_slice(&pool[o as usize..(o + n) as usize]);
                        at += n as usize;
                    }
                    for &x in regs {
                        let from = y * per + x as usize * vt;
                        row[at..at + vt].copy_from_slice(&lc[from..from + vt]);
                        at += vt;
                    }
                }
            }
            dev.put("gbank", col, vm.bank_words, &bank)?;
        }
    }
    dev.put("gmode", rect(xv, 0, NVC, nv), 1, &vec![1; NVC * nv])?;
    let hw = lay.chunk_cols() + lay.vmc();
    dev.put("gmode", rect(lay.head_x(), 0, hw, nv), 1, &vec![1; hw * nv])
}

/// The external VM's per-fire control (`vmctl`, on the root column): bit
/// lane CAP + iteration set where `skip(lane, iteration)` (the VM skips that
/// turn and answers token 0: a forced token, an idle lane).
pub fn put_vmctl(
    lay: &ProgramSpec,
    dev: &mut dyn Device,
    skip: impl Fn(usize, usize) -> bool,
) -> R<()> {
    let b_n = lay.paged().map_or(1, |s| s.lanes);
    let cap = lay.cap();
    let mut words = vec![0u32; (b_n * cap).div_ceil(32)];
    for b in 0..b_n {
        for i in 0..cap {
            if skip(b, i) {
                let k = b * cap + i;
                words[k / 32] |= 1 << (k % 32);
            }
        }
    }
    let nv = lay.hh() + 1;
    let all: Vec<u32> = (0..nv).flat_map(|_| words.iter().copied()).collect();
    dev.put("vmctl", rect(lay.root().0, 0, 1, nv), words.len(), &all)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lay_of(params: &str) -> ProgramSpec {
        let kv: Vec<(&str, usize)> = params
            .split(',')
            .map(|p| {
                p.split_once(':')
                    .map(|(k, v)| (k, v.parse().unwrap()))
                    .unwrap()
            })
            .collect();
        ProgramSpec::of_pairs(&kv).unwrap()
    }

    /// The hardware Qwen3-8B build (option A): 20 units, 36 layers.
    pub(crate) const B1: &str = "PX:128,PY:144,Hc:29,Dc:8,Gs:16,Ic:96,RPX:256,RPY:240,RHc:18,RDc:4,RGs:32,RIc:48,\
        RNL:3,RGY:16,RC:2,RR:4,NCOL:7,RTR:4,HR:4096,Qh:4,PS:16,NB:1,E:1,MC:1,W16:1,KV16:1,KVCP:0,PROF:0,TIME:0,\
        HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,PASSES:3,MODE:0,OVL:0,VMC:1,\
        CAP:64,iters:1,B:1,NP:3";

    /// serve8b_region_pipe's ring: MMAX micro-batches of B (B 1 x 12, as
    /// validated, on the hardware build's 20 units); a ring of every unit, the VM block or staging
    /// make no layout. A unit's mtab row y is the micro-batches' row y in
    /// slot order, zero past them.
    #[test]
    fn the_pipe_ring_and_its_tables() {
        let pipe = B1.replace("CAP:64", "CAP:32").replace("NP:3", "NP:1");
        let lay = lay_of(&format!("{pipe},PIPE:1,MMAX:12"));
        assert_eq!((lay.mmax(), lay.launch_lanes()), (Some(12), 12));
        assert_eq!(lay_of(&pipe).mmax(), None);
        assert_eq!(lay_of(&format!("{pipe},PIPE:0,MMAX:12")).mmax(), None);
        let of = ProgramSpec::of_str;
        assert!(of(&format!("{pipe},PIPE:1,MMAX:19")).is_ok());
        assert!(
            of(&format!("{pipe},PIPE:1,MMAX:20"))
                .unwrap_err()
                .contains("20 units")
        );
        assert!(of(&format!("{pipe},PIPE:1,MMAX:4,KVST:1")).is_err());
        // The ring's tables: the last unit's PEs that send x to the head
        // flag their piece count (bit 8: append the micro-batch word), and
        // nothing else changes; the plain build flags none.
        let ring = lay_of(&format!("{pipe},PIPE:1,MMAX:12"));
        let Layers::Units { units: u, .. } = ring.layers else {
            panic!("units")
        };
        let units = u.units(&ring);
        let mut tables = Tables::new(units.clone(), 1, u.rtr).unwrap();
        let plain: Vec<Vec<u32>> = units.iter().map(|v| tables.table(v).unwrap()).collect();
        tables.pipe = true;
        let rtn = tables.rtn();
        for (v, p) in units.iter().zip(&plain) {
            let r = tables.table(v).unwrap();
            let flagged = r.chunks(rtn).filter(|pe| pe[2] & 0x100 != 0).count();
            assert_eq!(flagged > 0, v.idx == units.len() - 1, "unit {}", v.idx);
            let unflag: Vec<u32> = r
                .chunks(rtn)
                .flat_map(|pe| {
                    let mut pe = pe.to_vec();
                    pe[2] &= 0xff;
                    pe
                })
                .collect();
            assert_eq!(&unflag, p, "unit {}", v.idx);
            assert!(p.chunks(rtn).all(|pe| pe[2] & 0x100 == 0));
        }
        let t = |m: u32| (0..2 * 3).map(|i| 100 * m + i).collect::<Vec<u32>>();
        assert_eq!(
            pipe_rows(&[t(1), t(2)], 2, 3, 3),
            [
                100, 101, 102, 200, 201, 202, 0, 0, 0, 103, 104, 105, 203, 204, 205, 0, 0, 0
            ]
        );
    }

    /// The VM's per-fire control: bit lane CAP + iteration, every VM row.
    /// RGX's third route set is the X axis's at groups of RGX, its last
    /// word log2(RGX); the first 11 words a PE are as without it.
    #[test]
    fn rgx_route_words_add_the_x_axis_at_its_group() {
        let rb: Vec<u32> = (0..10).map(|i| 1u32 << i).collect();
        let (px, py) = (256, 208);
        let base = route_words(&rb, px, py, 32, 16);
        let g16 = route_words(&rb, px, py, 16, 16);
        let gx = route_words_gx(&rb, px, py, 32, 16, 16);
        assert_eq!(gx.len(), px * py * 17);
        for pe in 0..px * py {
            assert_eq!(gx[pe * 17..pe * 17 + 11], base[pe * 11..pe * 11 + 11]);
            assert_eq!(gx[pe * 17 + 11..pe * 17 + 16], g16[pe * 11..pe * 11 + 5]);
            assert_eq!(gx[pe * 17 + 16], 4);
        }
    }

    /// One line of ARF words, run as the kernel's arf_ar runs them: every
    /// stream lands on the queue its PE's role reads, the root's sum counts
    /// every PE once, and the broadcast reaches every PE once. `word(p, c)`:
    /// PE p's colour c (0..5) word; `(lo, hi)`: the line's directions toward
    /// p - 1 and p + 1 (1 north / 2 south, or 4 west / 3 east).
    fn check_arf_line(
        n: usize,
        word: &dyn Fn(usize, usize) -> u32,
        role: &dyn Fn(usize) -> u32,
        (lo, hi): (usize, usize),
    ) {
        // Where a stream sent by p on colour c lands: every PE whose router
        // passes it to its ramp.
        let lands = |p: usize, c: usize| -> Vec<usize> {
            assert_ne!(word(p, c) & 1, 0, "PE {p} colour {c}: not from its ramp");
            let mut out = vec![];
            let mut todo = vec![(p, word(p, c))];
            while let Some((at, wd)) = todo.pop() {
                if wd & (1 << 5) != 0 && at != p || (at == p && wd & (1 << 5) != 0 && c == 4) {
                    out.push(at);
                }
                for (d, from) in [(lo, hi), (hi, lo)] {
                    if wd & (1 << (5 + d)) != 0 {
                        let q = if d == lo { at - 1 } else { at + 1 };
                        assert_ne!(
                            word(q, c) & (1 << from),
                            0,
                            "PE {q} colour {c}: not from {from}"
                        );
                        todo.push((q, word(q, c)));
                    }
                }
            }
            out
        };
        // Each PE's inputs (colour) and output colour, as arf_ar reads them.
        let root = n / 2;
        let mut ins: Vec<Vec<usize>> = vec![vec![]; n];
        let mut outc = vec![0usize; n];
        for p in 0..n {
            let r = role(p);
            let (r1, s2) = (((r >> 2) & 1) as usize, ((r >> 3) & 1) as usize);
            let (kk, gs) = ((r >> 4) & 0xff, (r >> 12) & 0xff);
            if r & 1 != 0 {
                assert_eq!(p, root);
                ins[p] = vec![2, 3];
                outc[p] = 4;
            } else if kk > 1 {
                ins[p] = if kk < gs { vec![r1] } else { vec![] };
                outc[p] = 1 - r1;
            } else if r & 2 != 0 {
                ins[p] = if gs > 1 { vec![r1] } else { vec![] };
                outc[p] = 2 + s2;
            } else {
                ins[p] = vec![r1, 3 - s2];
                outc[p] = 2 + s2;
            }
        }
        // Dataflow: a PE's sum (PEs counted) once all its inputs landed.
        let mut sum: Vec<Option<Vec<u32>>> = vec![None; n];
        let mut got: Vec<Vec<(usize, Vec<u32>)>> = vec![vec![]; n];
        for _ in 0..n + 1 {
            for p in 0..n {
                if sum[p].is_some() || got[p].len() < ins[p].len() {
                    continue;
                }
                let mut v = vec![0u32; n];
                v[p] += 1;
                for (_, g) in &got[p] {
                    for (a, b) in v.iter_mut().zip(g) {
                        *a += b;
                    }
                }
                if p != root {
                    let to = lands(p, outc[p]);
                    assert_eq!(to.len(), 1, "PE {p}'s stream lands on {to:?}");
                    assert!(
                        ins[to[0]].contains(&outc[p]),
                        "PE {p} -> {}: colour {}",
                        to[0],
                        outc[p]
                    );
                    assert!(
                        got[to[0]].iter().all(|(c, _)| *c != outc[p]),
                        "two streams on one queue"
                    );
                    got[to[0]].push((outc[p], v.clone()));
                }
                sum[p] = Some(v);
            }
        }
        assert_eq!(
            sum[root].as_deref(),
            Some(&vec![1u32; n][..]),
            "line of {n}"
        );
        let mut bc = lands(root, 4);
        bc.sort_unstable();
        assert_eq!(bc, (0..n).collect::<Vec<_>>(), "line of {n}: the broadcast");
    }

    /// ARF's Y words (and ARF 3's X words), run on every line as the kernel's
    /// arf_ar runs them; the other words are route_words' own.
    #[test]
    fn arf_route_words_reduce_every_row_once() {
        let rb: Vec<u32> = (0..10).map(|i| 1u32 << i).collect();
        // (ARY: Y groups of 8 or 16 rows)
        for (argx, ary, seg) in [
            (0, 8, false),
            (16, 8, false),
            (16, 16, false),
            (16, 8, true),
        ] {
            for (py_n, px_n) in [
                (208, 2),
                (24, 1),
                (17, 1),
                (9, 1),
                (3, 1),
                (5, 256),
                (4, 33),
                (3, 17),
                (3, 3),
                (152, 128),
            ] {
                if argx == 0 && px_n > 2 {
                    continue;
                }
                let st = if argx > 0 { 18 } else { 12 };
                let (px_n, py_n): (usize, usize) = (px_n, py_n);
                let gsx = 32.min(px_n.next_power_of_two());
                if seg && (px_n % gsx != 0 || gsx < 3) {
                    continue;
                }
                let w = route_words_arf(
                    &rb,
                    px_n,
                    py_n,
                    gsx,
                    16.min(py_n.next_power_of_two()),
                    ary,
                    argx,
                    seg,
                );
                let base = route_words(
                    &rb,
                    px_n,
                    py_n,
                    32.min(px_n.next_power_of_two()),
                    16.min(py_n.next_power_of_two()),
                );
                assert_eq!(w.len(), px_n * py_n * st);
                for pe in 0..px_n * py_n {
                    if !seg {
                        assert_eq!(w[pe * st..pe * st + 5], base[pe * 11..pe * 11 + 5], "X set");
                        assert_eq!(w[pe * st + 10], base[pe * 11 + 10], "group broadcast");
                    }
                }
                // (ARF bit 4: every segment of gsx columns a line of its own)
                if seg {
                    for py in 0..py_n {
                        for s0 in (0..px_n).step_by(gsx) {
                            let word = |p: usize, c: usize| w[(py * px_n + s0 + p) * st + c];
                            let role = |p: usize| w[(py * px_n + s0 + p) * st + 10];
                            check_arf_line(gsx, &word, &role, (4, 3));
                        }
                    }
                }
                for px in 0..px_n {
                    let word = |p: usize, c: usize| w[(p * px_n + px) * st + 5 + c];
                    let role = |p: usize| w[(p * px_n + px) * st + 11];
                    check_arf_line(py_n, &word, &role, (1, 2));
                }
                // (a line of 2 has one half: ARF needs >= 3 PEs; regions are 256 x 208)
                if argx > 0 && px_n >= 3 {
                    for py in 0..py_n {
                        let word = |p: usize, c: usize| w[(py * px_n + p) * st + 12 + c];
                        let role = |p: usize| w[(py * px_n + p) * st + 17];
                        check_arf_line(px_n, &word, &role, (4, 3));
                    }
                }
            }
        }
    }

    #[test]
    fn vmctl_marks_the_skipped_turns() {
        let params = "PX:8,PY:16,Hc:3,Dc:8,Gs:8,Ic:24,RPX:16,RPY:24,RHc:2,RDc:4,RGs:16,RIc:12,RNL:2,RGY:8,RC:2,RR:2,\
            NCOL:2,RTR:5,HR:40,Qh:4,PS:4,NB:2,E:2,B:2,MC:1,W16:1,NP:3,KV16:1,KVCP:0,HROUTE:1,ROT8:1,HEADMODE:2,HH:32,\
            Vr:160,Gx:4,Gy:8,vocab:1000,VMC:0,CAP:16,VMX:1,VMREAL:1,max_ops:104,nreg:12,pool_words:211,\
            bank_words:768,vm_lanes:2";
        let lay = lay_of(params);
        assert_eq!((vm_x(&lay), lay.root()), (52, (46, 16)));
        let mut rec = crate::spatial::device::Recorder::default();
        put_vmctl(&lay, &mut rec, |_, i| i == 0).unwrap();
        let calls = rec.take();
        assert!(calls[0].starts_with("h2d vmctl 46,0,1,33 n1 "), "{calls:?}");
        let w = rec.get("vmctl", rect(46, 0, 1, 33), 1).unwrap();
        assert!(w.iter().all(|&x| x == (1 | 1 << 16)));
        assert!(
            vm_load(&lay, &mut rec, &[None, None])
                .unwrap_err()
                .contains("no live")
        );
        assert!(
            vm_load(&lay, &mut rec, &[None])
                .unwrap_err()
                .contains("lanes")
        );
    }

    /// Lever D (`YD`): Llama 3.1 8B as layer 0's grid, 10 regions of
    /// 256 x 208 (30 layers) and the last grid at (512, 624).
    /// The batched host page copy moves what a pair at a time did: an HV
    /// build's PIE pages (16 kernel pages on consecutive rows each), one
    /// whose rows meet its source's (a pair at a time), on every unit.
    #[test]
    fn page_copies_move_runs_of_rows_as_pairs_did() {
        let p = "PX:128,PY:152,Hc:27,Dc:8,Gs:16,Ic:112,RPX:256,RPY:208,RHc:20,RDc:4,RGs:32,RIc:56,\
        RNL:3,RGY:16,GY:8,RC:2,RR:5,NCOL:1,YD:624,RTR:5,HR:4096,Qh:4,PS:1,NB:16,E:5,MC:1,W16:1,KV16:1,KVCP:0,\
        PROF:0,TIME:1,HROUTE:1,ROT8:1,QKN:0,EPS:925353388,NOLAYER:0,HEADMODE:2,HH:864,Vr:150,Gx:4,Gy:24,\
        vocab:128256,PASSES:3,MODE:2,OVL:0,VMC:1,CAP:128,iters:1,B:1,NP:1,SPD:1,LSE:1";
        let lay = lay_of(p);
        let Layers::Units { units: u, .. } = &lay.layers else {
            panic!("not serve8b_region")
        };
        fn fill(sym: &str, x: i32, y: i32, i: usize) -> u32 {
            (x as u32 * 7919 + y as u32 * 104_729 + i as u32 * 31 + u32::from(sym == "vc") * 17)
                & 0xffff
        }
        let rec = crate::spatial::device::Recorder::default().with_fill(fill);
        let mut dev = rec.clone();
        // PIE pages 3 -> 40 and 12 -> 13 (16 kernel pages each), then kernel
        // pages 5 -> 9 (rows 5 and 9: they meet within a run of 5).
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for (s, d) in [(3u32, 40u32), (12, 13)] {
            pairs.extend((0..16).map(|o| (s * 16 + o, d * 16 + o)));
        }
        pairs.extend((0..5).map(|o| (5 + o, 9 + o)));
        copy_pages(&lay, u, &mut dev, &pairs).unwrap();
        let shape = lay.paged().unwrap();
        for v in u.units(&lay) {
            let pr = v.k.py;
            let n = shape.page_size * v.k.dc;
            let kcn = shape.blocks * n;
            let len = v.nl() * kcn + 1;
            for sym in ["kc", "vc"] {
                for x in [v.x0, v.x0 + v.k.px - 1] {
                    // Expected rows, a pair at a time from the fill.
                    let mut want: std::collections::HashMap<usize, Vec<u32>> = Default::default();
                    let row = |want: &std::collections::HashMap<usize, Vec<u32>>, r: usize| {
                        want.get(&r).cloned().unwrap_or_else(|| {
                            (0..len)
                                .map(|i| fill(sym, x as i32, (v.y0 + r) as i32, i))
                                .collect()
                        })
                    };
                    for &(s, d) in &pairs {
                        let (s, d) = (s as usize, d as usize);
                        let (rs, bs, rd, bd) = (s % pr, s / pr, d % pr, d / pr);
                        let a = row(&want, rs);
                        let mut t = row(&want, rd);
                        for l in 0..v.nl() {
                            let o = 1 + l * kcn;
                            t[o + bd * n..o + (bd + 1) * n]
                                .copy_from_slice(&a[o + bs * n..o + (bs + 1) * n]);
                        }
                        want.insert(rd, t);
                    }
                    for (r, w) in &want {
                        let got = rec
                            .memory(sym, x as i32, (v.y0 + r) as i32)
                            .expect("written");
                        assert_eq!(&got, w, "unit at ({}, {}) {sym} x {x} row {r}", v.x0, v.y0);
                    }
                }
            }
        }
    }

    /// XAC's colours (0, 1, 2, 15) and the park colour 16 route through no
    /// region PE (codegen.rs `xac`), nor do the collectives' 5-10.
    #[test]
    fn xac_colours_are_free_on_regions() {
        let p = "PX:128,PY:152,Hc:27,Dc:8,Gs:16,Ic:112,RPX:256,RPY:208,RHc:20,RDc:4,RGs:32,RIc:56,\
        RNL:3,RGY:16,GY:8,RC:2,RR:5,NCOL:1,YD:624,RTR:5,HR:4096,Qh:4,PS:1,NB:16,E:5,MC:1,W16:1,KV16:1,KVCP:0,\
        PROF:0,TIME:1,HROUTE:1,ROT8:1,QKN:0,EPS:925353388,NOLAYER:0,HEADMODE:2,HH:864,Vr:150,Gx:4,Gy:24,\
        vocab:128256,PASSES:3,MODE:2,OVL:0,VMC:1,CAP:128,iters:1,B:1,NP:1,SPD:1,LSE:1";
        // (and Qwen3-8B's: PY 144, NCOL 5 column grids from YD 208)
        let q = p
            .replace(
                "PY:152,Hc:27,Dc:8,Gs:16,Ic:112",
                "PY:144,Hc:29,Dc:8,Gs:16,Ic:96",
            )
            .replace("RIc:56", "RIc:48")
            .replace("NCOL:1,YD:624", "NCOL:5,YD:208")
            .replace("QKN:0,EPS:925353388,", "")
            .replace("HH:864,Vr:150,Gx:4", "HH:1152,Vr:132,Gx:8")
            .replace("vocab:128256", "vocab:151936")
            .replace("CAP:128", "CAP:64");
        for p in [p.to_string(), q] {
            let lay = lay_of(&p);
            let Layers::Units { units: u, .. } = lay.layers else {
                panic!("not serve8b_region")
            };
            let units = u.units(&lay);
            let t = Tables::new(units.clone(), 1, u.rtr).unwrap();
            for v in units.iter().filter(|v| v.kind == Kind::Region) {
                for py in 0..v.k.py {
                    for px in 0..v.k.px {
                        let e = &t.pe[&(v.x0 + px, v.y0 + py)];
                        let mut cs: Vec<u32> = e.routes.keys().copied().collect();
                        cs.extend(e.recv.iter().chain(&e.send).map(|r| r[0]));
                        cs.extend([e.relay, e.fwd_in]);
                        assert!(
                            cs.iter()
                                .all(|c| ![0, 1, 2, 15, 16].contains(c) && !(5..=10).contains(c)),
                            "{cs:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lever_d_units_and_tables() {
        let p = "PX:128,PY:152,Hc:27,Dc:8,Gs:16,Ic:112,RPX:256,RPY:208,RHc:20,RDc:4,RGs:32,RIc:56,\
        RNL:3,RGY:16,GY:8,RC:2,RR:5,NCOL:1,YD:624,RTR:5,HR:4096,Qh:4,PS:1,NB:16,E:5,MC:1,W16:1,KV16:1,KVCP:0,\
        PROF:0,TIME:1,HROUTE:1,ROT8:1,QKN:0,EPS:925353388,NOLAYER:0,HEADMODE:2,HH:864,Vr:150,Gx:4,Gy:24,\
        vocab:128256,PASSES:3,MODE:2,OVL:0,VMC:1,CAP:128,iters:1,B:1,NP:1,SPD:1,LSE:1";
        let lay = lay_of(p);
        let Layers::Units { units: u, .. } = lay.layers else {
            panic!("not serve8b_region")
        };
        let units = u.units(&lay);
        assert_eq!((units.len(), lay.layers(), lay.head_x()), (12, 32, 640));
        let at: Vec<_> = units.iter().map(|v| (v.x0, v.y0, v.layers.start)).collect();
        assert_eq!(
            at,
            vec![
                (512, 0, 0),
                (256, 0, 1),
                (0, 0, 4),
                (0, 208, 7),
                (256, 208, 10),
                (256, 416, 13),
                (0, 416, 16),
                (0, 624, 19),
                (0, 832, 22),
                (256, 832, 25),
                (256, 624, 28),
                (512, 624, 31)
            ]
        );
        assert_eq!(lay.rectangle().1, 1040);
        let t = Tables::new(units.clone(), 1, u.rtr).unwrap();
        for v in &units {
            assert_eq!(t.table(v).unwrap().len(), v.k.px * v.k.py * t.rtn());
        }
        // (HVC: the vertical handoffs on centred columns build too, every route
        // conflict-free, and differ from the default only there)
        let tc = Tables::new_with(units.clone(), 1, u.rtr, true).unwrap();
        let mut changed = 0;
        for v in &units {
            let (a, b) = (t.table(v).unwrap(), tc.table(v).unwrap());
            assert_eq!(a.len(), b.len());
            changed += usize::from(a != b);
        }
        assert!(changed >= 5, "{changed} units' tables changed");
        let bad = p.replace("YD:624", "YD:432");
        assert!(
            ProgramSpec::of_pairs(
                &bad.split(',')
                    .map(|q| q
                        .split_once(':')
                        .map(|(k, v)| (k, v.parse().unwrap()))
                        .unwrap())
                    .collect::<Vec<_>>()
            )
            .is_err()
        );
    }

    /// Lever D for Qwen3-8B (36 layers): layer 0's grid, 10 regions of
    /// 256 x 208 by the U (down column 0, back up column 1 to (1, 1)) and
    /// the last 5 grids down the column from (512, 208).
    #[test]
    fn lever_d_u_units_and_tables() {
        let p = "PX:128,PY:144,Hc:29,Dc:8,Gs:16,Ic:96,RPX:256,RPY:208,RHc:20,RDc:4,RGs:32,RIc:48,RNL:3,RGY:16,\
        RC:2,RR:5,NCOL:5,YD:208,RTR:5,HR:4096,Qh:4,PS:1,NB:16,E:5,MC:1,W16:1,KV16:1,KVCP:0,PROF:0,TIME:0,\
        HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,PASSES:3,MODE:2,OVL:0,\
        VMC:1,CAP:64,iters:1,B:1,NP:1,SPD:1,LSE:1";
        let lay = lay_of(p);
        let Layers::Units { units: u, .. } = lay.layers else {
            panic!("not serve8b_region")
        };
        let units = u.units(&lay);
        assert_eq!((units.len(), lay.layers()), (16, 36));
        let at: Vec<_> = units.iter().map(|v| (v.x0, v.y0, v.layers.start)).collect();
        assert_eq!(
            at,
            vec![
                (512, 0, 0),
                (256, 0, 1),
                (0, 0, 4),
                (0, 208, 7),
                (0, 416, 10),
                (0, 624, 13),
                (0, 832, 16),
                (256, 832, 19),
                (256, 624, 22),
                (256, 416, 25),
                (256, 208, 28),
                (512, 208, 31),
                (512, 352, 32),
                (512, 496, 33),
                (512, 640, 34),
                (512, 784, 35)
            ]
        );
        assert_eq!(lay.rectangle().1, 1153);
        let t = Tables::new(units.clone(), 1, u.rtr).unwrap();
        for v in &units {
            assert_eq!(t.table(v).unwrap().len(), v.k.px * v.k.py * t.rtn());
        }
    }

    #[test]
    fn qwen3_8b_units_and_tables() {
        let lay = lay_of(B1);
        let Layers::Units { units: u, .. } = lay.layers else {
            panic!("not serve8b_region")
        };
        let units = u.units(&lay);
        assert_eq!(
            (units.len(), lay.layers(), lay.head_x(), lay.root()),
            (20, 36, 640, (676, 576))
        );
        let at: Vec<_> = units.iter().map(|v| (v.x0, v.y0, v.layers.start)).collect();
        assert_eq!(
            &at[..6],
            &[
                (512, 0, 0),
                (384, 0, 1),
                (256, 0, 2),
                (128, 0, 3),
                (0, 0, 4),
                (0, 144, 5)
            ]
        );
        assert_eq!(at[8], (0, 864, 14));
        assert_eq!(at[9], (256, 864, 17));
        assert_eq!(at[12], (256, 144, 26));
        assert_eq!((at[13], at[19]), ((512, 144, 29), (512, 1008, 35)));
        assert_eq!(lay.paged().map(|s| s.pages()), Some(144));
        let t = Tables::new(units.clone(), 1, u.rtr).unwrap();
        for v in &units {
            assert_eq!(t.table(v).unwrap().len(), v.k.px * v.k.py * t.rtn());
        }
    }
}
