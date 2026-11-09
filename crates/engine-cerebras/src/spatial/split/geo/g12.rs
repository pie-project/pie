//! Gemma 4 12B's role-split geometry (`scratch/split/host/split_geo_g12.py`, word for word): explicit unit
//! origins, the row-role split (attention rows r < RA with Hc A dims and the scores segments; plain rows with Hc W
//! dims; handoffs of 2 Hc(r) + 1 words a row) and the head groups (head column c takes x's dims of the CX rows in
//! its group: the members chain their words to the root on the last unit's CX column, the root sends them east;
//! the wafer loop's embedding comes back to unit 0's root row and down the group's chain).

use super::{Geo, HE, HM, HR3, HS, HX, P, Params, R, RX_R, TX_R, VA, VS, WA, WB, WC2, WM, rx, tx};
use std::collections::HashMap;

/// The head-group chains (CX columns only), alternating a hop.
pub const CH: [u32; 2] = [4, 5];

/// A CX's head-group chain: its input / output colours, the words it forwards (x to the head / the embedding).
#[derive(Clone, Copy, Debug, Default)]
pub struct Chain {
    pub ci: Option<u32>,
    pub co: Option<u32>,
    pub nfx: u32,
    pub nfe: u32,
    pub member: bool,
    pub emember: bool,
}

pub struct G12Geo {
    pub g: Geo,
    pub hca: u32,
    pub hcw: u32,
    pub ra: u32,
    pub k: u32,
    pub groups: Vec<Vec<u32>>,
    pub cxch: HashMap<P, Chain>,
}

/// Head groups: the A rows in groups of `ga`, the W rows above the CY row and those below it in groups of `gw`.
pub fn groups_split(py: u32, yc: u32, ra: u32, ga: u32, gw: u32) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    for (lo, hi, gs) in [(0, ra.min(yc), ga), (ra.min(yc), yc, gw), (yc, py, gw)] {
        let mut r = lo;
        while r < hi {
            out.push((r..(r + gs).min(hi)).collect());
            r += gs;
        }
    }
    out
}

/// (PX, PY, HR, Hc A, Hc W, RA, SWA, origins, groups, K) of a geometry name (`split_geo_g12.GEOS`).
#[allow(clippy::type_complexity)]
pub fn layout(
    name: &str,
) -> R<(
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Vec<(u32, u32)>,
    Vec<Vec<u32>>,
    u32,
)> {
    match name {
        "tiny" => Ok((
            8,
            8,
            18,
            2,
            3,
            3,
            4,
            vec![(0, 0), (0, 9), (9, 9), (9, 0)],
            groups_split(8, 4, 3, 3, 2),
            3,
        )),
        "g12" => {
            let uh = 284;
            let mut o: Vec<(u32, u32)> = (0..4).map(|u| (0, uh * u)).collect();
            o.extend((4..8).map(|u| (345, uh * (7 - u))));
            Ok((
                344,
                283,
                1153,
                12,
                15,
                132,
                43,
                o,
                groups_split(283, 141, 132, 6, 5),
                6,
            ))
        }
        _ => Err(format!("split g12: no geometry {name}")),
    }
}

impl G12Geo {
    pub fn new(name: &str, g2: u32, g2w: u32) -> R<G12Geo> {
        let (px, py, hr, hca, hcw, ra, swa, origins, groups, k) = layout(name)?;
        let hc = groups.len() as u32;
        let p = Params {
            px,
            py,
            unx: 1,
            uny: 1,
            hc,
            hr,
            hcc: hcw,
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
        g.hx0 = origins.iter().map(|o| o.0).max().unwrap() + g.uw;
        g.w = g.hx0 + hc + 1;
        g.h = (origins.iter().map(|o| o.1).max().unwrap() + g.uh).max(hr);
        g.routes = vec![Vec::new(); (g.w * g.h) as usize];
        g.origins = Some(origins);
        g.wnr = 18;
        g.seg_rows = Some(ra);
        Ok(G12Geo {
            g,
            hca,
            hcw,
            ra,
            k,
            groups,
            cxch: HashMap::new(),
        })
    }

    pub fn hc(&self, r: u32) -> u32 {
        if r < self.ra { self.hca } else { self.hcw }
    }

    /// Head column `c`'s dims (its group's).
    pub fn nx(&self, c: usize) -> u32 {
        self.groups[c].iter().map(|&r| self.hc(r)).sum()
    }

    pub fn unit_of(&self, x: u32, y: u32) -> Option<usize> {
        let g = &self.g;
        (0..g.units()).find(|&u| {
            let (x0, y0) = g.unit_origin(u);
            x >= x0 && x < x0 + g.uw && y >= y0 && y < y0 + g.uh
        })
    }

    /// "w", "cx", "cy", "filler", "head" or "hctl".
    pub fn kind(&self, x: u32, y: u32) -> &'static str {
        let g = &self.g;
        if x >= g.hx0 {
            if x - g.hx0 > g.p.hc || y >= g.p.hr {
                return "filler";
            }
            return if x - g.hx0 == g.hxc { "hctl" } else { "head" };
        }
        let Some(u) = self.unit_of(x, y) else {
            return "filler";
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

    fn ry(&self, u: usize, r: u32) -> u32 {
        self.g.unit_origin(u).1 + self.g.wl(r, self.g.yc)
    }

    /// Geo's build without its head paths and loop (`head_in`, `wloop`: the groups' here).
    pub fn build(&mut self) -> R<()> {
        self.g.skip_head = true;
        self.g.build(true, false)?;
        self.head_in()?;
        self.wloop()
    }

    fn head_in(&mut self) -> R<()> {
        let ul = self.g.units() - 1;
        let xcx = self.g.unit_origin(ul).0 + self.g.xc;
        let groups = self.groups.clone();
        for (c, grp) in groups.iter().enumerate() {
            let y = self.ry(ul, grp[0]);
            let hx = self.g.hx(c as u32);
            let p: Vec<P> = (xcx..=hx).map(|x| (x, y)).collect();
            self.g.path(&p, HX)?;
            self.g.cxq.entry((xcx, y)).or_default().out = Some(HX);
            self.g.hrole.insert((hx, y), 2);
            let col: Vec<P> = (0..self.g.p.hr).map(|yy| (hx, yy)).collect();
            self.g.mcast(&col, y as usize, HM)?;
            for (k, &r) in grp.iter().enumerate() {
                let yk = self.ry(ul, r);
                let nfx = grp[k + 1..].iter().map(|&rr| self.hc(rr)).sum();
                if k > 0 {
                    let yp = self.ry(ul, grp[k - 1]);
                    self.g.path(&[(xcx, yk), (xcx, yp)], CH[k % 2])?;
                }
                let ch = self.cxch.entry((xcx, yk)).or_default();
                ch.nfx = nfx;
                if k > 0 {
                    ch.co = Some(CH[k % 2]);
                    ch.member = true;
                }
                if k + 1 < grp.len() {
                    ch.ci = Some(CH[(k + 1) % 2]);
                }
            }
        }
        Ok(())
    }

    fn wloop(&mut self) -> R<()> {
        let hr = self.g.p.hr;
        let hcol = self.g.hx0 + self.g.hxc;
        let pts: Vec<P> = (0..hr).map(|y| (hcol, y)).collect();
        let (o, h) = self
            .g
            .line2(&pts, (hr / 2) as usize, WA, WB, WC2, WM, self.g.p.g2w)?;
        self.g.wrow.extend(o);
        self.g.wroot = Some((hcol, hr / 2, h));
        let u0 = self.g.unit_origin(0);
        if u0.1 != self.g.unit_origin(self.g.units() - 1).1 {
            return Err("split g12: unit 0 shares the last unit's rows".into());
        }
        let xcx = u0.0 + self.g.xc;
        let groups = self.groups.clone();
        for (c, grp) in groups.iter().enumerate() {
            let x = self.g.hx(c as u32);
            let te = self.ry(0, grp[0]);
            self.g.add(x, hr - 1, VA, RX_R | tx('N'))?;
            for y in 1..hr - 1 {
                self.g.add(x, y, VA, rx('S') | tx('N'))?;
            }
            self.g.add(x, 0, VA, rx('S') | TX_R)?;
            if te > 0 {
                let p: Vec<P> = (0..=te).map(|y| (x, y)).collect();
                self.g.path(&p, VS)?;
            }
            let p: Vec<P> = (xcx..=x).rev().map(|xx| (xx, te)).collect();
            self.g.path(&p, HE)?;
            self.g.cxq.entry((xcx, te)).or_default().emb = Some(HE);
            *self.g.hturn.entry((x, 0)).or_insert(0) |= 1;
            *self.g.hturn.entry((x, te)).or_insert(0) |= 2;
            for (k, &r) in grp.iter().enumerate() {
                let yk = self.ry(0, r);
                let nfe = grp[k + 1..].iter().map(|&rr| self.hc(rr)).sum();
                if k > 0 {
                    let yp = self.ry(0, grp[k - 1]);
                    self.g.path(&[(xcx, yp), (xcx, yk)], CH[k % 2])?;
                }
                let ch = self.cxch.entry((xcx, yk)).or_default();
                ch.nfe = nfe;
                if k > 0 {
                    ch.ci = Some(CH[k % 2]);
                    ch.emember = true;
                }
                if k + 1 < grp.len() {
                    ch.co = Some(CH[(k + 1) % 2]);
                }
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------- rcfg words

    pub fn rcfg_weight(&self, x: u32, y: u32, r: u32, c: u32, flush: bool) -> R<Vec<u32>> {
        let mut w = self.g.rcfg_weight(x, y, r)?;
        let n = 1 + 2 * self.g.wnr;
        for o in [18, 21] {
            if w[n + o] != 0 {
                w[n + o] = 2 * self.hc(r) + 1;
            }
        }
        w[n + 11] = u32::from(c == 0) | if flush { 4 } else { 0 };
        w[n + 12] = 0;
        Ok(w)
    }

    pub fn rcfg_cx(&self, x: u32, y: u32, first: bool, flush: bool) -> R<Vec<u32>> {
        let p = (x, y);
        let mut w = self.g.rcfg_cx(x, y, first)?;
        let n = 1 + 2 * 12;
        let ch = self.cxch.get(&p).copied().unwrap_or_default();
        let mut fl = w[n + 7];
        if flush {
            fl |= 8 | 16;
            if ch.member {
                fl |= 64 | 2;
            }
        }
        if first && ch.emember {
            fl |= 4 | 128;
        }
        w[n + 7] = fl;
        let pk = self.g.parks(p, 4)?;
        let ci = ch.ci.unwrap_or(pk[3]);
        let co = ch.co.unwrap_or(pk[3]);
        let fin = [w[n], w[n + 1], w[n + 2], w[n + 3], w[n + 9], ci];
        w[n + 8] = self.g.tmpc(p, &fin, &Geo::CT6)?;
        w.extend_from_slice(&[
            ci,
            co,
            if flush { ch.nfx } else { 0 },
            if first { ch.nfe } else { 0 },
        ]);
        Ok(w)
    }

    pub fn rcfg_cy(&self, x: u32, y: u32, flush: bool) -> R<Vec<u32>> {
        let mut w = self.g.rcfg_cy(x, y)?;
        w.extend_from_slice(&[if flush { 4 } else { 0 }, 0, 0, 0, 0]);
        Ok(w)
    }

    pub fn rcfg_head(&self, x: u32, y: u32) -> R<Vec<u32>> {
        let g = &self.g;
        let p = (x, y);
        let pk = g.parks(p, 7)?;
        let mut w = g.words(p, 12)?;
        let ro = g.hrow[&p];
        let xr = g.hrole.get(&p).copied().unwrap_or(0);
        let mut q = [
            ro.2.unwrap_or(pk[0]),
            ro.3.unwrap_or(pk[1]),
            if xr == 2 { HX } else { pk[2] },
            pk[3],
            if xr == 0 { HM } else { pk[4] },
            ro.1,
            if xr == 2 { HM } else { pk[0] },
        ];
        let tr = g.hturn.get(&p).copied().unwrap_or(0);
        if tr & 1 != 0 && tr & 2 == 0 {
            q[3] = VA;
        }
        if tr & 2 != 0 {
            if xr != 2 {
                return Err(format!("{p:?}: VS turn off a group root"));
            }
            q[4] = if y == 0 { VA } else { VS };
        }
        let q7 = HR3;
        let fl = 1 | (tr << 1);
        let c = (0..g.p.hc).find(|&cc| g.hx(cc) == x).ok_or("head column")?;
        let fin = [q[0], q[1], q[2], q[3], q[4], q7];
        w.extend_from_slice(&q);
        w.push(if ro.0 == 3 { 1 } else { ro.0 });
        w.push(xr);
        w.push(g.tmpc(p, &fin, &Geo::CT6)?);
        w.push(q7);
        w.push(fl);
        w.push(y);
        w.extend_from_slice(&[VA, VS, HE, HS]);
        w.push(self.nx(c as usize));
        Ok(w)
    }

    pub fn rcfg_hctl(&self, x: u32, y: u32, vr: u32) -> R<Vec<u32>> {
        self.g.rcfg_hctl(x, y, vr)
    }

    /// Every placed PE's rcfg words as "x y kind w..." lines (`split_geo_g12`'s dump format).
    pub fn rcfg_dump(&self, vr: u32) -> R<String> {
        let g = &self.g;
        let last = g.units() - 1;
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
                    "head" => self.rcfg_head(x, y)?,
                    "hctl" => self.rcfg_hctl(x, y, vr)?,
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

    /// G12_RCFG=<geo>:<vr>:<out path>: writes the dump (compare with split_geo_g12.py's).
    #[test]
    #[ignore]
    fn g12_rcfg_dump() {
        let v = std::env::var("G12_RCFG").expect("G12_RCFG=<geo>:<vr>:<out>");
        let mut it = v.split(':');
        let (name, vr, out) = (
            it.next().unwrap(),
            it.next().unwrap().parse().unwrap(),
            it.next().unwrap(),
        );
        let (g2, g2w) = if name == "tiny" { (2, 2) } else { (16, 24) };
        let mut g = G12Geo::new(name, g2, g2w).unwrap();
        g.build().unwrap();
        std::fs::write(out, g.rcfg_dump(vr).unwrap()).unwrap();
    }
}
