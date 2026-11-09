//! Gemma 4's role-split geometry (`tools/split/gemma/split_geo_gemma.py`, word for word):
//! explicit unit origins, the PLE storage PEs above each unit column with their token word (TB)
//! and bus (PB), the shared-KV line (KVS) and the Gemma rcfg words of every PE kind.

use super::{Geo, P, Params, R, RX_R, TX_R, rx, tx};
use std::collections::{BTreeSet, HashMap, HashSet};

pub const TB: u32 = 0;
pub const KVS: u32 = 10;

/// A unit column: its x, its units top to bottom, its storage rows, its bus colour, tokens a storage PE.
#[derive(Clone, Debug)]
pub struct Col {
    pub x: u32,
    pub units: Vec<usize>,
    pub srows: (u32, u32),
    pub pb: u32,
    pub nt: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Store {
    pub t0: u32,
    pub nt: u32,
    pub lv: u32,
    pub top: bool,
    pub pb: u32,
}

pub struct GemmaGeo {
    pub g: Geo,
    pub cols: Vec<Col>,
    pub k: u32,
    pub kvsrc: Option<usize>,
    pub kvdst: Vec<usize>,
    pub ple: HashMap<P, Store>,
    pub tbsend: HashSet<u32>,
    pub cyple: HashMap<P, (u32, u32, u32)>,
    pub kvsend: HashSet<P>,
    pub kvrecv: HashSet<P>,
}

/// (PX, PY, HC, HR, Hc, SWA, unit origins, columns, KV source unit, KV sharers, K)
pub type Layout = (
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Vec<(u32, u32)>,
    Vec<Col>,
    usize,
    Vec<usize>,
    u32,
);

/// The layout a geometry name states (`split_geo_gemma.GEOS`).
pub fn layout(name: &str) -> R<Layout> {
    match name {
        "tiny" => Ok((
            8,
            8,
            4,
            32,
            3,
            4,
            vec![(9, 4), (9, 22), (0, 22), (0, 13), (0, 4)],
            vec![
                Col {
                    x: 0,
                    units: vec![4, 3, 2],
                    srows: (0, 4),
                    pb: 17,
                    nt: 128,
                },
                Col {
                    x: 9,
                    units: vec![0, 1],
                    srows: (2, 4),
                    pb: 10,
                    nt: 256,
                },
            ],
            2,
            vec![3, 4],
            5,
        )),
        "e2b" => {
            let uh = 129;
            let oa: Vec<(u32, u32)> = (0..5).map(|i| (0, 1153 - uh * (5 - i))).collect();
            Ok((
                256,
                128,
                64,
                1130,
                12,
                128,
                vec![(257, 508), (257, 1024), oa[4], oa[3], oa[2], oa[1], oa[0]],
                vec![
                    Col {
                        x: 0,
                        units: vec![6, 5, 4, 3, 2],
                        srows: (163, 508),
                        pb: 17,
                        nt: 760,
                    },
                    Col {
                        x: 257,
                        units: vec![0, 1],
                        srows: (370, 508),
                        pb: 10,
                        nt: 1900,
                    },
                ],
                2,
                vec![3, 4, 5, 6],
                5,
            ))
        }
        "e4b" => {
            let uh = 161;
            let oa: Vec<(u32, u32)> = (0..4).map(|i| (0, 1153 - uh * (4 - i))).collect();
            Ok((
                256,
                160,
                80,
                1130,
                16,
                128,
                vec![
                    (257, oa[0].1),
                    (257, oa[1].1),
                    (257, oa[3].1),
                    oa[3],
                    oa[2],
                    oa[1],
                    oa[0],
                ],
                vec![
                    Col {
                        x: 0,
                        units: vec![6, 5, 4, 3],
                        srows: (oa[0].1 - 332, oa[0].1),
                        pb: 17,
                        nt: 791,
                    },
                    Col {
                        x: 257,
                        units: vec![0, 1, 2],
                        srows: (oa[0].1 - 249, oa[0].1),
                        pb: 10,
                        nt: 1055,
                    },
                ],
                3,
                vec![4, 5, 6],
                6,
            ))
        }
        _ => Err(format!("split gemma: no geometry {name}")),
    }
}

impl GemmaGeo {
    pub fn new(name: &str, g2: u32, g2w: u32) -> R<GemmaGeo> {
        let (px, py, hc, hr, hcc, swa, origins, cols, kvsrc, kvdst, k) = layout(name)?;
        let p = Params {
            px,
            py,
            unx: 1,
            uny: 1,
            hc,
            hr,
            hcc,
            swa,
            swg: swa,
            swv: swa,
            dkg: 1,
            dkh: 0,
            g2x: g2,
            g2y: g2,
            g2h: g2,
            wl: true,
            g2w,
        };
        let mut g = Geo::new(p);
        let uw = g.uw;
        g.hx0 = origins.iter().map(|o| o.0).max().unwrap() + uw;
        g.w = g.hx0 + hc + 1;
        g.h = (origins.iter().map(|o| o.1).max().unwrap() + g.uh).max(hr);
        g.routes = vec![Vec::new(); (g.w * g.h) as usize];
        g.origins = Some(origins);
        g.wnr = 18;
        g.hand_n = 2 * hcc + 1;
        Ok(GemmaGeo {
            g,
            cols,
            k,
            kvsrc: Some(kvsrc),
            kvdst,
            ple: HashMap::new(),
            tbsend: HashSet::new(),
            cyple: HashMap::new(),
            kvsend: HashSet::new(),
            kvrecv: HashSet::new(),
        })
    }

    pub fn unit_of(&self, x: u32, y: u32) -> Option<usize> {
        let g = &self.g;
        (0..g.units()).find(|&u| {
            let (x0, y0) = g.unit_origin(u);
            x >= x0 && x < x0 + g.uw && y >= y0 && y < y0 + g.uh
        })
    }

    /// "w", "cx", "cy", "filler", "head", "hctl" or "ple".
    pub fn kind(&self, x: u32, y: u32) -> &'static str {
        let g = &self.g;
        if x >= g.hx0 {
            if x - g.hx0 > g.p.hc || y >= g.p.hr {
                return "filler";
            }
            return if x - g.hx0 == g.hxc { "hctl" } else { "head" };
        }
        let Some(u) = self.unit_of(x, y) else {
            return "ple";
        };
        let (x0, y0) = g.unit_origin(u);
        let (lx, ly) = (x - x0, y - y0);
        if lx == g.xc && ly == g.yc {
            "filler"
        } else if lx == g.xc {
            "cx"
        } else if ly == g.yc {
            "cy"
        } else {
            "w"
        }
    }

    pub fn build(&mut self) -> R<()> {
        self.g.build(true, false)?;
        let (px, xc, yc) = (self.g.p.px, self.g.xc, self.g.yc);
        let (hx0, hxc, uw, uh) = (self.g.hx0, self.g.hxc, self.g.uw, self.g.uh);
        // TB: one stream a storage row, west from the head control row to the westmost block on it
        let mut trows: HashMap<u32, Vec<u32>> = HashMap::new();
        for col in &self.cols {
            for y in col.srows.0..col.srows.1 {
                trows.entry(y).or_default().push(col.x);
            }
        }
        let mut ys: Vec<u32> = trows.keys().copied().collect();
        ys.sort_unstable();
        for y in ys {
            let xs0 = trows[&y].clone();
            let xw = *xs0.iter().min().unwrap();
            let hc = hx0 + hxc;
            let pts: Vec<u32> = (xw..=hc).rev().collect();
            for (i, &x) in pts.iter().enumerate() {
                let rxb = if i == 0 { RX_R } else { rx('E') };
                let mut txb = if i == pts.len() - 1 { 0 } else { tx('W') };
                if x < hx0 && xs0.iter().any(|&x0| x0 <= x && x < x0 + uw && x != x0 + xc) {
                    txb |= TX_R;
                }
                self.g.add(x, y, TB, rxb | txb)?;
            }
            self.tbsend.insert(y);
        }
        let cols = self.cols.clone();
        for col in &cols {
            let (y0, y1) = col.srows;
            let top = self.g.unit_origin(col.units[0]).1;
            if y1 != top {
                return Err(
                    "split gemma: the storage sits directly above the column's top unit".into(),
                );
            }
            let nu = col.units.len() as u32;
            let lv = self.k * nu;
            for c in 0..px {
                let x = col.x + self.g.wl(c, xc);
                for y in y0..y1 {
                    self.g
                        .add(x, y, col.pb, if y == y0 { RX_R } else { rx('N') } | tx('S'))?;
                    self.ple.insert(
                        (x, y),
                        Store {
                            t0: (y - y0) * col.nt,
                            nt: col.nt,
                            lv,
                            top: y == y0,
                            pb: col.pb,
                        },
                    );
                }
                let mut prev_end = y1;
                for (ui, &u) in col.units.iter().enumerate() {
                    let (_, uy) = self.g.unit_origin(u);
                    let lastu = ui + 1 == col.units.len();
                    for y in prev_end..uy {
                        self.g.add(x, y, col.pb, rx('N') | tx('S'))?;
                    }
                    prev_end = uy + uh;
                    for ly in 0..uh {
                        let y = uy + ly;
                        if ly == yc {
                            self.g.add(
                                x,
                                y,
                                col.pb,
                                rx('N') | TX_R | if lastu { 0 } else { tx('S') },
                            )?;
                            self.cyple.insert((x, y), (col.pb, ui as u32, lv));
                            if lastu {
                                break;
                            }
                        } else {
                            self.g.add(x, y, col.pb, rx('N') | tx('S'))?;
                        }
                    }
                }
            }
        }
        if let Some(src) = self.kvsrc {
            let (sx, sy) = self.g.unit_origin(src);
            let ytop = self
                .kvdst
                .iter()
                .map(|&u| self.g.unit_origin(u).1)
                .min()
                .unwrap();
            for &u in &self.kvdst {
                let o = self.g.unit_origin(u);
                if o.0 != sx || o.1 >= sy {
                    return Err(
                        "split gemma: the KVS consumers sit above the source in its column".into(),
                    );
                }
            }
            for c in 0..px {
                let x = sx + self.g.wl(c, xc);
                let ysrc = sy + yc;
                self.g.add(x, ysrc, KVS, RX_R | tx('N'))?;
                self.kvsend.insert((x, ysrc));
                for y in (ytop..ysrc).rev() {
                    let u = self
                        .unit_of(x, y)
                        .ok_or_else(|| format!("KVS crosses a gap at ({x}, {y})"))?;
                    let ly = y - self.g.unit_origin(u).1;
                    let take = self.kvdst.contains(&u) && ly != yc;
                    let txb = if take { TX_R } else { 0 } | if y > ytop { tx('N') } else { 0 };
                    self.g.add(x, y, KVS, rx('S') | txb)?;
                    if take {
                        self.kvrecv.insert((x, y));
                    }
                }
            }
        }
        Ok(())
    }

    /// The storage rectangles (x, y, w, h).
    pub fn ple_rects(&self) -> Vec<(u32, u32, u32, u32)> {
        self.cols
            .iter()
            .map(|c| (c.x, c.srows.0, self.g.uw, c.srows.1 - c.srows.0))
            .collect()
    }

    /// Non-unit PEs west of the head with a route, outside the storage blocks (sorted).
    pub fn pass_pes(&self) -> Vec<P> {
        let mut st = HashSet::new();
        for (x0, y0, w, h) in self.ple_rects() {
            for x in x0..x0 + w {
                for y in y0..y0 + h {
                    st.insert((x, y));
                }
            }
        }
        let mut out = BTreeSet::new();
        for y in 0..self.g.h {
            for x in 0..self.g.hx0 {
                if !self.g.route((x, y)).is_empty()
                    && self.kind(x, y) == "ple"
                    && !st.contains(&(x, y))
                {
                    out.insert((y, x));
                }
            }
        }
        out.into_iter().map(|(y, x)| (x, y)).collect()
    }

    pub fn rcfg_weight(&self, x: u32, y: u32, r: u32, c: u32, flush: bool) -> R<Vec<u32>> {
        let mut w = self.g.rcfg_weight(x, y, r)?;
        let n = 1 + 2 * self.g.wnr;
        let fl = u32::from(c == 0)
            | if self.kvrecv.contains(&(x, y)) { 2 } else { 0 }
            | if flush { 4 } else { 0 };
        w[n + 11] = fl;
        w[n + 12] = KVS;
        Ok(w)
    }

    pub fn rcfg_cx(&self, x: u32, y: u32, first: bool, flush: bool) -> R<Vec<u32>> {
        let mut w = self.g.rcfg_cx(x, y, first)?;
        if flush {
            w[1 + 2 * 12 + 7] |= 8 | 16;
        }
        Ok(w)
    }

    pub fn rcfg_cy(&self, x: u32, y: u32, flush: bool) -> R<Vec<u32>> {
        let mut w = self.g.rcfg_cy(x, y)?;
        let (pb, ui, lv) = match self.cyple.get(&(x, y)) {
            Some(&(pb, ui, lv)) => (Some(pb), ui, lv),
            None => (None, 0, 0),
        };
        let fl = u32::from(pb.is_some())
            | if self.kvsend.contains(&(x, y)) { 2 } else { 0 }
            | if flush { 4 } else { 0 };
        w.extend_from_slice(&[fl, pb.unwrap_or(0), ui, lv, KVS]);
        Ok(w)
    }

    pub fn rcfg_ple(&self, x: u32, y: u32) -> R<Vec<u32>> {
        let p = (x, y);
        let mut w = self.g.words(p, 8)?;
        let st = self.ple.get(&p);
        let tb_in = self.g.get_route(p, TB).is_some_and(|b| b & TX_R != 0);
        let pk = self.g.parks(p, 2)?;
        let fl = u32::from(st.is_some())
            | if tb_in { 2 } else { 0 }
            | if st.is_some_and(|s| s.top) { 4 } else { 0 };
        let tbq = if tb_in { TB } else { pk[0] };
        w.extend_from_slice(&[
            fl,
            st.map_or(0, |s| s.t0),
            st.map_or(0, |s| s.nt),
            st.map_or(0, |s| s.lv),
            st.map_or(pk[1], |s| s.pb),
            tbq,
        ]);
        w.push(self.g.tmpc(p, &[tbq], &Geo::CT6)?);
        Ok(w)
    }

    pub fn rcfg_hctl(&self, x: u32, y: u32, vr: u32) -> R<Vec<u32>> {
        let mut w = self.g.rcfg_hctl(x, y, vr)?;
        w.extend_from_slice(&[u32::from(self.tbsend.contains(&y)), TB]);
        Ok(w)
    }
}

impl GemmaGeo {
    /// Every placed PE's rcfg words as "x y kind w..." lines (the dump `split_geo_gemma.dump_rcfg` writes too).
    pub fn rcfg_dump(&self, vr: u32) -> R<String> {
        let g = &self.g;
        let last = g.units() - 1;
        let pass: HashSet<P> = self.pass_pes().into_iter().collect();
        let mut out = String::new();
        for x in 0..g.w {
            for y in 0..g.h {
                let k = self.kind(x, y);
                let w = match k {
                    "w" | "cx" | "cy" => {
                        let u = self.unit_of(x, y).unwrap();
                        let (x0, y0) = g.unit_origin(u);
                        let (lx, ly) = (x - x0, y - y0);
                        let r = ly - u32::from(ly > g.yc);
                        let c = lx - u32::from(lx > g.xc);
                        match k {
                            "w" => self.rcfg_weight(x, y, r, c, u == last)?,
                            "cx" => self.rcfg_cx(x, y, u == 0, u == last)?,
                            _ => self.rcfg_cy(x, y, u == last)?,
                        }
                    }
                    "head" => g.rcfg_head(x, y)?,
                    "hctl" => self.rcfg_hctl(x, y, vr)?,
                    "ple"
                        if self.ple.contains_key(&(x, y))
                            || pass.contains(&(x, y))
                            || self.ple_rects().iter().any(|&(a, b, w, h)| {
                                x >= a && x < a + w && y >= b && y < b + h
                            }) =>
                    {
                        self.rcfg_ple(x, y)?
                    }
                    _ => continue,
                };
                out.push_str(&format!("{x} {y} {k}"));
                for v in w {
                    out.push_str(&format!(" {v}"));
                }
                out.push('\n');
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GEMMA_RCFG=<geo>:<vr>:<out path>: writes the dump (compare with split_geo_gemma.py's).
    #[test]
    #[ignore]
    fn gemma_rcfg_dump() {
        let v = std::env::var("GEMMA_RCFG").expect("GEMMA_RCFG=<geo>:<vr>:<out>");
        let mut it = v.split(':');
        let (name, vr, out) = (
            it.next().unwrap(),
            it.next().unwrap().parse().unwrap(),
            it.next().unwrap(),
        );
        let (g2, g2w) = if name == "tiny" { (2, 2) } else { (16, 24) };
        let mut g = GemmaGeo::new(name, g2, g2w).unwrap();
        g.build().unwrap();
        std::fs::write(out, g.rcfg_dump(vr).unwrap()).unwrap();
    }
}
