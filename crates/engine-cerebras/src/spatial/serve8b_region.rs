//! The serve8b_region wafer program (`spatial/csl/serve8b_region`): Qwen3-8B
//! as layer units. The Rust form of the host side of its `serve_lib8b.py`
//! and `units8r.py`. The head and the embedding strip are serve8b's
//! ([`super::serve8b::head`], [`super::serve8b::embed`]); requests and logits
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
use super::serve2d::Layout2d;
use super::serve8b::{Kernel, Rope, S8b, rect};
use super::wafer::Sink;
use crate::sdk::{DataType, Order, Runtime};

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
    /// Grids in the top band, layer 0's included (XC / PX + 1).
    pub band: usize,
    /// Fork 3's eta VM as the block east of the strip (`VMX`, `VMREAL`).
    pub vm: Option<VmBlock>,
}

/// The VM block's image (VM5.md): ops, registers, pool and bank words, and
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

/// serve8b_region's shape from its params.
pub fn layout(get: &dyn Fn(&str) -> Option<usize>) -> R<Layout2d> {
    let p = |k: &str| -> R<usize> { get(k).ok_or_else(|| format!("no param {k}")) };
    let mode = p("HEADMODE")?;
    if mode != 2 {
        return Err(format!("HEADMODE {mode}: the engine serves serve8b_region with its head (HEADMODE 2)"));
    }
    if p("W16")? != 1 || p("KV16")? != 1 || get("KVCP").unwrap_or(0) != 0 {
        return Err("serve8b_region keeps weights and KV in bf16 and copies pages by the host (W16 1, KV16 1, KVCP 0)".into());
    }
    let (px, py, gs) = (p("PX")?, p("PY")?, p("Gs")?);
    let (rpx, rpy, rgs, rc) = (p("RPX")?, p("RPY")?, p("RGs")?, p("RC")?);
    if px == 0 || rc * rpx % px != 0 {
        return Err(format!("serve8b_region: RC RPX {} is not whole grids of {px}", rc * rpx));
    }
    let u = Units {
        rpx,
        rpy,
        rhc: p("RHc")?,
        rdc: p("RDc")?,
        rgs,
        ric: p("RIc")?,
        rnl: p("RNL")?,
        rc,
        rr: p("RR")?,
        ncol: p("NCOL")?,
        rtr: p("RTR")?,
        gy: get("GY").filter(|&g| g > 0).unwrap_or(gs),
        rgy: get("RGY").filter(|&g| g > 0).unwrap_or(rgs),
        hroute: get("HROUTE").unwrap_or(0) == 1,
        rot8: get("ROT8").unwrap_or(0) == 1,
        nolayer: get("NOLAYER").unwrap_or(0) == 1,
        band: rc * rpx / px + 1,
        vm: match (get("VMX"), get("VMREAL")) {
            (Some(1), Some(1)) => Some(VmBlock {
                max_ops: p("max_ops")?,
                nreg: p("nreg")?,
                pool_words: p("pool_words")?,
                bank_words: p("bank_words")?,
                lanes: p("vm_lanes")?,
            }),
            (Some(1), _) => return Err("VMX 1 with the stand-in VM (VMREAL 0): a check build, not served".into()),
            _ => None,
        },
    };
    let lay = Layout2d {
        px,
        py,
        hc: p("Hc")?,
        hidden: p("HR")?,
        dc: p("Dc")?,
        gs,
        qh: p("Qh")?,
        ic: p("Ic")?,
        tc: 0,
        gw: 0,
        gh: 0,
        hh: p("HH")?,
        vr: p("Vr")?,
        gx: p("Gx")?,
        vmc: p("VMC")?,
        vocab: p("vocab")?,
        cap: p("CAP")?,
        paged: Some(super::paged::Shape {
            lanes: p("B")?,
            rows: py.min(rpy),
            page_size: p("PS")?,
            blocks: p("NB")?,
            entries: p("E")?,
        }),
        // serve8b_region_pipe (`PIPE` 1): MMAX micro-batches of B lanes a
        // launch round the units.
        mmax: if get("PIPE").unwrap_or(0) == 1 { Some(p("MMAX")?) } else { None },
        host_copy: true,
        vm: None,
        hlm: None,
        hlbn: get("HLBN"),
        vmlane_words: 2,
        vm_b: None,
        vm_conc: false,
        vm_defer: false,
        kvst: get("KVST").unwrap_or(0),
        s8b: Some(S8b::Units(u)),
    };
    let c = lay.chunk_cols();
    let ok = py % 4 == 0
        && gs > 0
        && rgs > 0
        && px % gs == 0
        && rpx % rgs == 0
        && px / gs == rpx / rgs
        && rgs * u.rdc == gs * lay.dc
        && px * lay.ic == rpx * u.ric
        && lay.dc % 2 == 0
        && u.rdc % 2 == 0
        && lay.gx > 0
        && c % lay.gx == 0
        && lay.vmc <= 1
        && lay.hh % 2 == 0
        && lay.vr % 2 == 0
        && py * lay.hc >= lay.hidden
        && rpy * u.rhc >= lay.hidden
        && c * 2 * lay.hc >= lay.hidden;
    if !ok {
        return Err(format!("serve8b_region params do not make a layout: {lay:?} {u:?}"));
    }
    if let Some(m) = lay.mmax {
        // A ring with every unit busy can deadlock (the handoffs are
        // synchronous); the VM block is not on the ring.
        let n = u.units(&lay).len();
        if m == 0 || m + 1 > n || u.vm.is_some() || lay.kvst > 0 {
            return Err(format!("serve8b_region_pipe: MMAX {m} on {n} units, VMX {}, KVST {}", u.vm.is_some(), lay.kvst));
        }
    }
    Ok(lay)
}

impl Units {
    pub fn layers(&self) -> usize {
        self.band + self.rc * self.rr * self.rnl + self.ncol
    }

    /// The units in layer order (`units8r.units`).
    pub fn units(&self, lay: &Layout2d) -> Vec<Unit> {
        let grid = Kernel { px: lay.px, py: lay.py, hc: lay.hc, dc: lay.dc, gs: lay.gs, ic: lay.ic };
        let region = Kernel { px: self.rpx, py: self.rpy, hc: self.rhc, dc: self.rdc, gs: self.rgs, ic: self.ric };
        let mut out: Vec<Unit> = Vec::new();
        let mut add = |kind: Kind, x0: usize, y0: usize| {
            let (k, nl) = if kind == Kind::Grid { (grid, 1) } else { (region, self.rnl) };
            let l = out.last().map_or(0, |u| u.layers.end);
            out.push(Unit { kind, idx: out.len(), x0, y0, k, layers: l..l + nl });
        };
        let xc = self.rc * self.rpx;
        for i in 0..self.band {
            add(Kind::Grid, xc - i * lay.px, 0);
        }
        for c in 0..self.rc {
            for i in 0..self.rr {
                let r = if c % 2 == 0 { i } else { self.rr - 1 - i };
                add(Kind::Region, c * self.rpx, lay.py + r * self.rpy);
            }
        }
        for k in 0..self.ncol {
            add(Kind::Grid, xc, lay.py * (1 + k));
        }
        out
    }

    /// A unit's collectives' Y group.
    fn y_group(&self, u: &Unit) -> usize {
        if u.kind == Kind::Grid { self.gy } else { self.rgy }
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
/// color (`units8r.Tables`).
pub struct Tables {
    units: Vec<Unit>,
    lanes: u32,
    rtr: usize,
    pe: HashMap<(usize, usize), Pe>,
    /// serve8b_region_pipe's ring (units8r.py there): the last unit's x to
    /// the head carries the micro-batch word (`nsend` bit 8).
    pub pipe: bool,
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
type Piece = (usize, u32, u32, u32);

impl Tables {
    pub fn new(units: Vec<Unit>, lanes: usize, rtr: usize) -> R<Tables> {
        let mut pe = HashMap::new();
        for u in &units {
            for y in u.y0..u.y0 + u.k.py {
                for x in u.x0..u.x0 + u.k.px {
                    let e = Pe { routes: BTreeMap::new(), recv: Vec::new(), send: Vec::new(), relay: NONE, fwd_in: NONE };
                    pe.insert((x, y), e);
                }
            }
        }
        let mut t = Tables { units, lanes: lanes as u32, rtr, pe, pipe: false };
        t.build()?;
        Ok(t)
    }

    fn at(&mut self, x: usize, y: usize) -> R<&mut Pe> {
        self.pe.get_mut(&(x, y)).ok_or_else(|| format!("a path leaves the units at ({x}, {y})"))
    }

    fn route(&mut self, x: usize, y: usize, color: u32, rx: u32, t1: u32, t2: u32) -> R<()> {
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
    fn leg(&mut self, color: u32, pts: &[(usize, usize)]) -> R<()> {
        let mut cells = vec![pts[0]];
        let mut dirs = Vec::new();
        for w in pts.windows(2) {
            let ((ax, ay), (bx, by)) = (w[0], w[1]);
            if ax != bx && ay != by {
                return Err(format!("a leg from {:?} to {:?} is not straight", w[0], w[1]));
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
            let tx = if i == cells.len() - 1 { D_RAMP } else { dirs[i] };
            self.route(x, y, color, rx, tx, tx)?;
        }
        Ok(())
    }

    /// Unit `u`'s row `row` takes `color` at column `xin` (from `rxin`, or
    /// RAMP: that PE multicasts its own) and every PE of the row gets it.
    fn row_mcast(&mut self, color: u32, u: &Unit, row: usize, xin: usize, rxin: u32) -> R<()> {
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

    fn recv(&mut self, x: usize, y: usize, color: u32, off: u32, n: u32) -> R<()> {
        self.at(x, y)?.recv.push([color, off, n]);
        Ok(())
    }

    fn send(&mut self, x: usize, y: usize, color: u32, off: u32, n: u32) -> R<()> {
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
            for py in 0..ak.py {
                let c = a.x0 + py % ak.px;
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
        } else if b.x0 == a.x0 + ak.px && b.y0 >= a.y0 && b.y0 + bk.py <= a.y0 + ak.py && ak.px >= bk.py {
            self.rechunk_b(a, b, r)?;
        } else {
            return Err(format!("no handoff from unit {} at {:?} to unit {} at {:?}", a.idx, (a.x0, a.y0), b.idx, (b.x0, b.y0)));
        }
        Ok(())
    }

    /// Dest row g's pieces: (source row, source offset, dest offset, n).
    fn pieces(a: &Unit, b: &Unit, g: usize) -> Vec<(usize, u32, u32, u32)> {
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
    fn chain(
        &mut self,
        col: usize,
        srcs: &[Piece],
        hop: [u32; 2],
        leg_color: u32,
        leg_pts: &[(usize, usize)],
        dest: (usize, usize),
        down: bool,
    ) -> R<()> {
        let order: Vec<Piece> = if down { srcs.to_vec() } else { srcs.iter().rev().copied().collect() };
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
    fn rechunk_a(&mut self, a: &Unit, b: &Unit, r: u32) -> R<()> {
        for g in 0..b.k.py {
            let p = Tables::pieces(a, b, g);
            if p.is_empty() {
                continue;
            }
            let col = a.x0 + g % a.k.px;
            let srcs: Vec<Piece> = p.iter().map(|&(row, so, d, n)| (a.y0 + row, so, d, n)).collect();
            let dest = (col, b.y0 + g);
            let last = srcs[srcs.len() - 1].0;
            self.chain(col, &srcs, C_A_HOP, C_A_LEG[g / a.k.px], &[(col, last), dest], dest, true)?;
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
            let srcs: Vec<Piece> = p.iter().map(|&(row, so, d, n)| (a.y0 + row, so, d, n)).collect();
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

    /// Words of a PE's table.
    pub fn rtn(&self) -> usize {
        7 + 4 * self.rtr + 3 * RT_I + 3 * RT_O
    }

    /// Unit `u`'s `rtab`, `PY` x `PX` PEs of [`Tables::rtn`] words (16-bit
    /// values): `[nr, nrecv, nsend, relay, fwd_in, px, py, (color, rx, tx1,
    /// tx2) x RTR, (color, off, n) x 3 in, (color, off, n) x 3 out]`.
    pub fn table(&self, u: &Unit) -> R<Vec<u32>> {
        let rtn = self.rtn();
        let (oi, oo) = (7 + 4 * self.rtr, 7 + 4 * self.rtr + 3 * RT_I);
        let mut t = vec![0u32; u.k.py * u.k.px * rtn];
        for py in 0..u.k.py {
            for px in 0..u.k.px {
                let e = &self.pe[&(u.x0 + px, u.y0 + py)];
                let one_color = |v: &[[u32; 3]]| v.iter().all(|p| p[0] == v[0][0]);
                if e.routes.len() > self.rtr
                    || e.recv.len() > RT_I
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
                        return Err(format!("unit {} PE ({px}, {py}): recv color == fwd_in {}", u.idx, e.fwd_in));
                    }
                    let last = self.units.last().is_some_and(|l| l.idx == u.idx);
                    if last && e.send.first().is_some_and(|s| s[0] == 14 || s[0] == 15) {
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

/// Each PE's collective route words, 11 a PE (`units8r.route_words`: the
/// kernels' route_phase1/phase2/bcast/gbcast run on the host): for
/// (c_r1_0, c_r1_1, c_r2_0, c_r2_1, c_bc) on X, then on Y, then c_bc's
/// group broadcast, each `rb[rx] | rb[5 + tx1] | rb[5 + tx2]`.
pub fn route_words(rb: &[u32], px_n: usize, py_n: usize, gs: usize, gy: usize) -> Vec<u32> {
    const RAMP: usize = 0;
    const N_: usize = 1;
    const S_: usize = 2;
    const E_: usize = 3;
    const W_: usize = 4;
    let word = |(rx, t1, t2): (usize, usize, usize)| rb[rx] | rb[5 + t1] | rb[5 + t2];
    let mut out = vec![0u32; py_n * px_n * 11];
    for py in 0..py_n {
        for px in 0..px_n {
            let at = (py * px_n + px) * 11;
            for (ax, (h, t, idv, alen, ga)) in [(W_, E_, px, px_n, gs), (N_, S_, py, py_n, gy)].into_iter().enumerate() {
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
                    if idv == 0 { (t, RAMP, RAMP) } else { (t, h, RAMP) }
                } else if idv == alen - 1 {
                    (h, RAMP, RAMP)
                } else {
                    (h, RAMP, t)
                };
                for (c, &f) in cfg.iter().enumerate() {
                    out[at + 5 * ax + c] = word(f);
                }
            }
            let (r, r1) = (px % gs, gs / 2);
            let g = if r == r1 {
                (RAMP, W_, if r1 == gs - 1 { W_ } else { E_ })
            } else if r < r1 {
                if r == 0 { (E_, RAMP, RAMP) } else { (E_, W_, RAMP) }
            } else if r == gs - 1 {
                (W_, RAMP, RAMP)
            } else {
                (W_, RAMP, E_)
            };
            out[at + 10] = word(g);
        }
    }
    out
}

/// Every unit's `rcfg` (with `HROUTE`, from the program's `rbits`) and
/// `rtab`, in `serve_lib8b.py`'s order: before `init`.
pub fn put_routes(lay: &Layout2d, u: &Units, sink: &mut dyn Sink, rbits: Option<&[u32]>) -> R<()> {
    let units = u.units(lay);
    if u.hroute {
        let rb = rbits.filter(|b| b.len() >= 10).ok_or("HROUTE 1 without the program's rbits")?;
        for v in &units {
            let words = route_words(rb, v.k.px, v.k.py, v.k.gs, u.y_group(v));
            sink.h2d("rcfg", rect(v.x0, v.y0, v.k.px, v.k.py), 11, &words)?;
        }
    }
    let mut tables = Tables::new(units.clone(), lay.paged.map_or(1, |s| s.lanes), u.rtr)?;
    tables.pipe = lay.mmax.is_some();
    for v in &units {
        sink.h2d16("rtab", rect(v.x0, v.y0, v.k.px, v.k.py), tables.rtn(), &tables.table(v)?)?;
    }
    Ok(())
}

/// Every weight (and zeroed KV), unit by unit (`_unit`: a region's layers
/// stacked in each symbol), then the head and the strip.
pub fn place(lay: &Layout2d, u: &Units, ckpt: &Ckpt, rope_theta: f64, sink: &mut dyn Sink) -> R<()> {
    let inv_freq = super::serve8b::inv_freq(rope_theta, lay.gs * lay.dc);
    let rope = if u.rot8 {
        Rope { units: &[512, 64, 8, 1], len: 8, inv_freq: &inv_freq }
    } else {
        Rope { units: &[256, 16, 1], len: 16, inv_freq: &inv_freq }
    };
    if !u.nolayer {
        for v in u.units(lay) {
            let layers: Vec<usize> = v.layers.clone().collect();
            super::serve8b::unit(lay, ckpt, v.k, &layers, (v.x0, v.y0), &rope, false, sink)?;
        }
    }
    super::serve8b::head(lay, ckpt, sink)?;
    super::serve8b::embed(lay, ckpt, sink)
}

/// `copy_kv`'s whole pages by the host: for every unit, the source and
/// destination rows' `kc`/`vc` read, each of its layers' block patched
/// (block b of layer l at `1 + l NB Dc PS + b Dc PS`), the destination row
/// written back. Only between launches.
pub fn copy_pages(lay: &Layout2d, u: &Units, rt: &mut Runtime, pairs: &[(u32, u32)]) -> R<()> {
    let shape = lay.paged.ok_or("serve8b_region without its paged shape")?;
    let e = |e: crate::sdk::Error| e.to_string();
    let units = u.units(lay);
    for &(src, dst) in pairs {
        let (src, dst) = (src as usize, dst as usize);
        if src >= shape.pages() || dst >= shape.pages() {
            return Err(format!("page copy {src} -> {dst} outside a pool of {}", shape.pages()));
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
                rt.memcpy_d2h(sym, &mut a, row(rs), len, Order::RowMajor, DataType::Bits16).map_err(e)?;
                let mut d = if rd == rs {
                    a.clone()
                } else {
                    let mut d = vec![0u32; v.k.px * len];
                    rt.memcpy_d2h(sym, &mut d, row(rd), len, Order::RowMajor, DataType::Bits16).map_err(e)?;
                    d
                };
                for (s, t) in a.chunks_exact(len).zip(d.chunks_exact_mut(len)) {
                    for l in 0..v.nl() {
                        let o = 1 + l * kcn;
                        t[o + bd * n..o + (bd + 1) * n].copy_from_slice(&s[o + bs * n..o + (bs + 1) * n]);
                    }
                }
                rt.memcpy_h2d(sym, &d, row(rd), len, Order::RowMajor, DataType::Bits16).map_err(e)?;
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
    lay: &Layout2d,
    rt: &mut Runtime,
    table_of: &dyn Fn(usize) -> R<Vec<Vec<u32>>>,
) -> R<()> {
    let (shape, slots) = match (lay.paged, lay.mmax, lay.s8b) {
        (Some(s), Some(m), Some(S8b::Units(_))) => (s, m),
        _ => return Err("micro-batch tables on a build without serve8b_region_pipe's ring".into()),
    };
    let w = shape.row_words();
    let mut made: Vec<(usize, Vec<u32>)> = Vec::new();
    for g in lay.grid_areas() {
        let rows = g.h as usize;
        if !made.iter().any(|t| t.0 == rows) {
            let tables = table_of(rows)?;
            if tables.is_empty() || tables.len() > slots {
                return Err(format!("{} micro-batch(es) for a ring of {slots}", tables.len()));
            }
            made.push((rows, pipe_rows(&tables, rows, slots, w)));
        }
        let mt = &made.iter().find(|t| t.0 == rows).expect("built above").1;
        rt.memcpy_h2d_rowbcast("mtab", mt, g, slots * w, DataType::Bits32).map_err(|e| e.to_string())?;
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
pub fn check_pipe(lay: &Layout2d, rt: &mut Runtime) -> R<()> {
    let c = lay.chunk_cols();
    let mut pe = vec![0u32; c * 3];
    rt.memcpy_d2h("perr", &mut pe, rect(lay.head_x(), lay.feeder(), c, 1), 3, Order::RowMajor, DataType::Bits32)
        .map_err(|e| e.to_string())?;
    let bad: Vec<(usize, &[u32])> = pe.chunks_exact(3).enumerate().filter(|(_, r)| r[0] != 0).collect();
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "micro-batch misalignment at the head's feeder (column: [mismatches, first expected, first got]): {bad:?}"
        ))
    }
}

/// serve8b_region_pipe's queue probe: a launch of no laps (the units run no
/// pass; the head and strip only their bootstrap) that records each unit
/// PE's output-queue status at its start (`prof[14]`). The PEs (unit, px,
/// py, status) whose output queues are not all empty: wavelets left from
/// the last launch.
pub fn queue_probe(lay: &Layout2d, rt: &mut Runtime, nmb: usize) -> R<Vec<(usize, usize, usize, u32)>> {
    let Some(S8b::Units(u)) = lay.s8b else {
        return Err("the queue probe is serve8b_region_pipe's".into());
    };
    let e = |e: crate::sdk::Error| e.to_string();
    rt.launch("compute", &[0, nmb as u32]).map_err(e)?;
    let mut bad = Vec::new();
    for v in u.units(lay) {
        let (px, py) = (v.k.px, v.k.py);
        let mut a = vec![0u32; px * py * 16];
        rt.memcpy_d2h("prof", &mut a, rect(v.x0, v.y0, px, py), 16, Order::RowMajor, DataType::Bits32).map_err(e)?;
        for (i, w) in a.chunks_exact(16).enumerate() {
            if (w[14] >> 8) & 0xff != 0xff {
                bad.push((v.idx, i % px, i / px, w[14]));
            }
        }
    }
    Ok(bad)
}

/// One VM lane's program: its table and its homes' seeds, the logits rows
/// it reads a turn.
pub struct VmLane<'a> {
    pub table: &'a super::guest4::Table,
    pub seeds: &'a [super::guest4::wafer::Seed],
    pub rows: u32,
}

/// The VM block's first column (XV: east of the strip).
pub fn vm_x(lay: &Layout2d) -> usize {
    lay.head_x() + lay.chunk_cols() + lay.vmc + lay.py / 4
}

/// A lane's pool block and wide rows (region row x column x register x Vt,
/// f32) from its seeds (`vmhost.seeded`).
fn seeded(lay: &Layout2d, vm: &VmBlock, seeds: &[super::guest4::wafer::Seed]) -> (Vec<u32>, Vec<f32>) {
    let (hh, vr) = (lay.hh, lay.vr);
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
fn wide_column(lay: &Layout2d, vm: &VmBlock, wide: &[f32], j: usize) -> Vec<u32> {
    let (hh, f, vt) = (lay.hh, lay.feeder(), lay.vr / NVC);
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

/// Writes lane b's program `lanes[b]` (`None`: an empty lane, its `glane`
/// zeroed, the VM skips its turns) to every VM PE and sets `gmode` 1 over
/// the VM block and the head (`vmhost.load_tables`): one shared op arena
/// (each distinct table once), lane 0's seeds as the live pool and wide
/// registers, each lane's bank window (its live pool ranges, at most 8,
/// and wide registers) in `gbank` when there are several lanes.
#[allow(clippy::too_many_lines)]
pub fn vm_load(lay: &Layout2d, sink: &mut dyn Sink, lanes: &[Option<VmLane>]) -> R<()> {
    let Some(S8b::Units(Units { vm: Some(vm), .. })) = lay.s8b else {
        return Err("this serve8b_region build has no eta VM (VMX 1, VMREAL 1)".into());
    };
    let (hh, vt) = (lay.hh, lay.vr / NVC);
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
    let base_of = |l: &VmLane| bases.iter().find(|b| b.0 == l.table.ins.as_slice()).map_or(0, |b| b.1);
    let mut glane = vec![0u32; n_lanes * 4];
    let mut gbdesc = vec![0u32; n_lanes * 8];
    let mut windows: Vec<Option<(usize, Vec<(u32, u32)>, Vec<u32>)>> = Vec::new();
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
        glane[l * 4..l * 4 + 4].copy_from_slice(&[base_of(lane) | rows << 16, t.tok_home.unwrap_or(0), boff as u32, mask]);
        let words = rg.iter().map(|r| r.1 as usize).sum::<usize>() + t.bank_regs.len() * vt;
        windows.push(Some((boff, rg, t.bank_regs.clone())));
        boff += words;
    }
    if n_lanes > 1 && boff > vm.bank_words {
        return Err(format!("the lanes' bank windows take {boff} words of {}", vm.bank_words));
    }
    let states: Vec<Option<(Vec<u32>, Vec<f32>)>> =
        lanes.iter().map(|l| l.as_ref().map(|l| seeded(lay, &vm, l.seeds))).collect();
    let (pool0, wide0) = seeded(lay, &vm, first.seeds);
    let th = first.table.tok_home.unwrap_or(0);
    let tile = |w: &[u32]| -> Vec<u32> { (0..nv).flat_map(|_| w.iter().copied()).collect() };
    let xv = vm_x(lay);
    for j in 0..NVC {
        let col = rect(xv + j, 0, 1, nv);
        sink.h2d("gops", col, vm.max_ops * 4, &tile(&ops))?;
        sink.h2d("gpool", col, vm.pool_words, &tile(&pool0))?;
        sink.h2d("gvalid", col, 1, &vec![lay.vocab as u32; nv])?;
        sink.h2d("gtok_home", col, 1, &vec![th; nv])?;
        sink.h2d("gostep", col, 1, &vec![0; nv])?;
        sink.h2d("gwide", col, vm.nreg * vt, &wide_column(lay, &vm, &wide0, j))?;
        sink.h2d("glane", col, n_lanes * 4, &tile(&glane))?;
        sink.h2d("gbdesc", col, n_lanes * 8, &tile(&gbdesc))?;
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
                        row[at..at + n as usize].copy_from_slice(&pool[o as usize..(o + n) as usize]);
                        at += n as usize;
                    }
                    for &x in regs {
                        let from = y * per + x as usize * vt;
                        row[at..at + vt].copy_from_slice(&lc[from..from + vt]);
                        at += vt;
                    }
                }
            }
            sink.h2d("gbank", col, vm.bank_words, &bank)?;
        }
    }
    sink.h2d("gmode", rect(xv, 0, NVC, nv), 1, &vec![1; NVC * nv])?;
    let hw = lay.chunk_cols() + lay.vmc;
    sink.h2d("gmode", rect(lay.head_x(), 0, hw, nv), 1, &vec![1; hw * nv])
}

/// The external VM's per-fire control (`vmctl`, on the root column): bit
/// lane CAP + iteration set where `skip(lane, iteration)` (the VM skips that
/// turn and answers token 0: a forced token, an idle lane).
pub fn put_vmctl(lay: &Layout2d, sink: &mut dyn Sink, skip: impl Fn(usize, usize) -> bool) -> R<()> {
    let b_n = lay.paged.map_or(1, |s| s.lanes);
    let cap = lay.cap;
    let mut words = vec![0u32; (b_n * cap).div_ceil(32)];
    for b in 0..b_n {
        for i in 0..cap {
            if skip(b, i) {
                let k = b * cap + i;
                words[k / 32] |= 1 << (k % 32);
            }
        }
    }
    let nv = lay.hh + 1;
    let all: Vec<u32> = (0..nv).flat_map(|_| words.iter().copied()).collect();
    sink.h2d("vmctl", rect(lay.root().0, 0, 1, nv), words.len(), &all)
}

/// The VM's out ring (`vmhost.ring`): RING rows of RW words from the block's
/// hub (its first column, the feeder row).
pub fn vm_ring(lay: &Layout2d, rt: &mut Runtime) -> R<Vec<Vec<u32>>> {
    let e = |e: crate::sdk::Error| e.to_string();
    let hub = rect(vm_x(lay), lay.feeder(), 1, 1);
    let mut g = [0u32; 3];
    rt.memcpy_d2h("gring", &mut g, hub, 3, Order::RowMajor, DataType::Bits32).map_err(e)?;
    let (ring, rw) = (g[0] as usize, g[1] as usize);
    let mut out = vec![0u32; ring * rw];
    rt.memcpy_d2h("gout", &mut out, hub, ring * rw, Order::RowMajor, DataType::Bits32).map_err(e)?;
    Ok(out.chunks(rw.max(1)).map(<[u32]>::to_vec).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lay_of(params: &str) -> Layout2d {
        let kv: Vec<(&str, usize)> = params
            .split(',')
            .map(|p| p.split_once(':').map(|(k, v)| (k, v.parse().unwrap())).unwrap())
            .collect();
        Layout2d::of(|k| kv.iter().find(|p| p.0 == k).map(|p| p.1)).unwrap().unwrap()
    }

    /// 0125's Qwen3-8B build (option A): 20 units, 36 layers, as units8r.
    pub(crate) const B1: &str = "PX:128,PY:144,Hc:29,Dc:8,Gs:16,Ic:96,RPX:256,RPY:240,RHc:18,RDc:4,RGs:32,RIc:48,\
        RNL:3,RGY:16,RC:2,RR:4,NCOL:7,RTR:4,HR:4096,Qh:4,PS:16,NB:1,E:1,MC:1,W16:1,KV16:1,KVCP:0,PROF:0,TIME:0,\
        HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,PASSES:3,MODE:0,OVL:0,VMC:1,\
        CAP:64,iters:1,B:1,NP:3";

    /// serve8b_region_pipe's ring: MMAX micro-batches of B (0167's B 1 x 12
    /// on 0125's 20 units); a ring of every unit, the VM block or staging
    /// make no layout. A unit's mtab row y is the micro-batches' row y in
    /// slot order, zero past them.
    #[test]
    fn the_pipe_ring_and_its_tables() {
        let pipe = B1.replace("CAP:64", "CAP:32").replace("NP:3", "NP:1");
        let lay = lay_of(&format!("{pipe},PIPE:1,MMAX:12"));
        assert_eq!((lay.mmax, lay.launch_lanes()), (Some(12), 12));
        assert_eq!(lay_of(&pipe).mmax, None);
        assert_eq!(lay_of(&format!("{pipe},PIPE:0,MMAX:12")).mmax, None);
        let of = |p: &str| {
            let kv: Vec<(&str, usize)> =
                p.split(',').map(|p| p.split_once(':').map(|(k, v)| (k, v.parse().unwrap())).unwrap()).collect();
            Layout2d::of(|k| kv.iter().find(|p| p.0 == k).map(|p| p.1)).unwrap()
        };
        assert!(of(&format!("{pipe},PIPE:1,MMAX:19")).is_ok());
        assert!(of(&format!("{pipe},PIPE:1,MMAX:20")).unwrap_err().contains("20 units"));
        assert!(of(&format!("{pipe},PIPE:1,MMAX:4,KVST:1")).is_err());
        // The ring's tables: the last unit's PEs that send x to the head
        // flag their piece count (bit 8: append the micro-batch word), and
        // nothing else changes; the plain build flags none.
        let ring = lay_of(&format!("{pipe},PIPE:1,MMAX:12"));
        let Some(S8b::Units(u)) = ring.s8b else { panic!("units") };
        let units = u.units(&ring);
        let mut tables = Tables::new(units.clone(), 1, u.rtr).unwrap();
        let plain: Vec<Vec<u32>> = units.iter().map(|v| tables.table(v).unwrap()).collect();
        tables.pipe = true;
        let rtn = tables.rtn();
        for (v, p) in units.iter().zip(&plain) {
            let r = tables.table(v).unwrap();
            let flagged = r.chunks(rtn).filter(|pe| pe[2] & 0x100 != 0).count();
            assert_eq!(flagged > 0, v.idx == units.len() - 1, "unit {}", v.idx);
            let unflag: Vec<u32> = r.chunks(rtn).flat_map(|pe| {
                let mut pe = pe.to_vec();
                pe[2] &= 0xff;
                pe
            }).collect();
            assert_eq!(&unflag, p, "unit {}", v.idx);
            assert!(p.chunks(rtn).all(|pe| pe[2] & 0x100 == 0));
        }
        let t = |m: u32| (0..2 * 3).map(|i| 100 * m + i).collect::<Vec<u32>>();
        assert_eq!(
            pipe_rows(&[t(1), t(2)], 2, 3, 3),
            [100, 101, 102, 200, 201, 202, 0, 0, 0, 103, 104, 105, 203, 204, 205, 0, 0, 0]
        );
    }

    /// The VM's per-fire control: bit lane CAP + iteration, every VM row.
    #[test]
    fn vmctl_marks_the_skipped_turns() {
        use crate::sdk::Rect;
        struct Rec(Vec<(String, Rect, usize, Vec<u32>)>);
        impl Sink for Rec {
            fn h2d(&mut self, s: &str, r: Rect, n: usize, w: &[u32]) -> R<()> {
                self.0.push((s.into(), r, n, w.to_vec()));
                Ok(())
            }
            fn h2d16(&mut self, s: &str, r: Rect, n: usize, w: &[u32]) -> R<()> {
                self.h2d(s, r, n, w)
            }
        }
        let params = "PX:8,PY:16,Hc:3,Dc:8,Gs:8,Ic:24,RPX:16,RPY:24,RHc:2,RDc:4,RGs:16,RIc:12,RNL:2,RGY:8,RC:2,RR:2,\
            NCOL:2,RTR:5,HR:40,Qh:4,PS:4,NB:2,E:2,B:2,MC:1,W16:1,NP:3,KV16:1,KVCP:0,HROUTE:1,ROT8:1,HEADMODE:2,HH:32,\
            Vr:160,Gx:4,Gy:8,vocab:1000,VMC:0,CAP:16,VMX:1,VMREAL:1,max_ops:104,nreg:12,pool_words:211,\
            bank_words:768,vm_lanes:2";
        let lay = lay_of(params);
        assert_eq!((vm_x(&lay), lay.root()), (52, (46, 16)));
        let mut rec = Rec(Vec::new());
        put_vmctl(&lay, &mut rec, |_, i| i == 0).unwrap();
        let (sym, r, n, w) = &rec.0[0];
        assert_eq!((sym.as_str(), r.x, r.h, *n), ("vmctl", 46, 33, 1));
        assert!(w.iter().all(|&x| x == (1 | 1 << 16)));
        assert!(vm_load(&lay, &mut rec, &[None, None]).unwrap_err().contains("no live"));
        assert!(vm_load(&lay, &mut rec, &[None]).unwrap_err().contains("lanes"));
    }

    #[test]
    fn qwen3_8b_units_and_tables() {
        let lay = lay_of(B1);
        let Some(S8b::Units(u)) = lay.s8b else { panic!("not serve8b_region") };
        let units = u.units(&lay);
        assert_eq!((units.len(), lay.layers(), lay.head_x(), lay.root()), (20, 36, 640, (676, 576)));
        let at: Vec<_> = units.iter().map(|v| (v.x0, v.y0, v.layers.start)).collect();
        assert_eq!(&at[..6], &[(512, 0, 0), (384, 0, 1), (256, 0, 2), (128, 0, 3), (0, 0, 4), (0, 144, 5)]);
        assert_eq!(at[8], (0, 864, 14));
        assert_eq!(at[9], (256, 864, 17));
        assert_eq!(at[12], (256, 144, 26));
        assert_eq!((at[13], at[19]), ((512, 144, 29), (512, 1008, 35)));
        assert_eq!(lay.paged.map(|s| s.pages()), Some(144));
        let t = Tables::new(units.clone(), 1, u.rtr).unwrap();
        for v in &units {
            assert_eq!(t.table(v).unwrap().len(), v.k.px * v.k.py * t.rtn());
        }
    }
}
