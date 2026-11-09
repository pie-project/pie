//! gpt-oss's role-split geometry (`tools/split/gptoss/split_geo_gptoss.py`, word for word): a unit is
//! [AX attention columns | CX | MX MoE columns] by PY weight rows (the base unit with its control column at
//! local x = AX), the units a snake from the bottom of unit column 0 up and down column 1, and a head band
//! below them: PY dims-rows (dims-row i = the last unit's CX row i) and a control row in its middle, by HVC
//! vocabulary columns. See the Python module's docs for the routes.

use super::{
    CM, CS, CS2, Geo, HA, HM, HR3, LB, MX, MY, NCOL, P, Params, R, RX_R, Role, RootQ, SA, SB, SM,
    TX_R, WA, WB, WC2, WM, X2, XA, XB, Y2, YA, YB, or, rx, tx,
};
use std::collections::{HashMap, HashSet};

pub const HV2: u32 = 9;
pub const HE: u32 = 10;
pub const V1: u32 = 8;
pub const W2: u32 = 13;

pub struct GptossGeo {
    pub g: Geo,
    pub ax: u32,
    pub mx: u32,
    pub swr: u32,
    pub hvc: u32,
    pub hb0: u32,
    pub hxc: u32,
    pub w: u32,
    pub h: u32,
    rout: HashMap<P, u32>,
    hsrc: HashSet<P>,
    hturn: HashSet<P>,
    embrelay: HashSet<P>,
    hrow: HashMap<P, Role>,
    wrow: HashMap<P, Role>,
    wroot: Option<(u32, u32, RootQ)>,
}

/// (AX, MX, PY, UNX, UNY, Hc, SWA, SWR, HVC) of a geometry name (`split_geo_gptoss.GEOS`).
pub fn shape(name: &str) -> R<[u32; 9]> {
    match name {
        "tiny" => Ok([4, 8, 4, 2, 1, 8, 2, 8, 26]),
        "full" => Ok([64, 288, 160, 2, 6, 18, 8, 32, 748]),
        _ => Err(format!("gpt-oss geometry {name}?")),
    }
}

impl GptossGeo {
    pub fn new(name: &str, g2: u32, g2w: u32, wl: bool) -> R<GptossGeo> {
        let [ax, mx, py, unx, uny, hcc, swa, swr, hvc] = shape(name)?;
        let p = Params {
            px: ax + mx,
            py,
            unx,
            uny,
            hc: 0,
            hr: py,
            hcc,
            swa,
            swg: 1,
            swv: 1,
            dkg: 1,
            dkh: 0,
            g2x: g2,
            g2y: g2,
            g2h: g2,
            wl,
            g2w,
        };
        let mut g = Geo::new(p);
        if unx % 2 != 0 {
            return Err(
                "gpt-oss: an even unit-column count (the last unit next to the band)".into(),
            );
        }
        g.xc = ax;
        g.uw = ax + mx + 1;
        g.uh = py + 1;
        g.order.clear();
        for ux in 0..unx {
            if ux % 2 == 0 {
                g.order.extend((0..uny).rev().map(|uy| (ux, uy)));
            } else {
                g.order.extend((0..uny).map(|uy| (ux, uy)));
            }
        }
        let hb0 = uny * g.uh;
        let w = (unx * g.uw).max(hvc);
        let h = hb0 + py + 1;
        g.w = w;
        g.h = h;
        g.routes = vec![Vec::new(); (w * h) as usize];
        Ok(GptossGeo {
            g,
            ax,
            mx,
            swr,
            hvc,
            hb0,
            hxc: py / 2,
            w,
            h,
            rout: HashMap::new(),
            hsrc: HashSet::new(),
            hturn: HashSet::new(),
            embrelay: HashSet::new(),
            hrow: HashMap::new(),
            wrow: HashMap::new(),
            wroot: None,
        })
    }

    /// Dims-row i's y.
    pub fn by(&self, i: u32) -> u32 {
        self.hb0 + self.g.wl(i, self.hxc)
    }

    pub fn kind(&self, x: u32, y: u32) -> &'static str {
        if y >= self.hb0 {
            if x >= self.hvc {
                return "filler";
            }
            return if y - self.hb0 == self.hxc {
                "hctl"
            } else {
                "head"
            };
        }
        let (ux, uy) = (x / self.g.uw, y / self.g.uh);
        if ux >= self.g.p.unx || uy >= self.g.p.uny {
            return "filler";
        }
        let (lx, ly) = (x % self.g.uw, y % self.g.uh);
        match (lx == self.g.xc, ly == self.g.yc) {
            (true, true) => "filler",
            (true, false) => "cx",
            (false, true) => {
                if lx < self.g.xc {
                    "cya"
                } else {
                    "cym"
                }
            }
            _ => {
                if lx < self.g.xc {
                    "wa"
                } else {
                    "wm"
                }
            }
        }
    }

    pub fn build(&mut self) -> R<()> {
        let (px, py, xc, yc, ax) = (self.g.p.px, self.g.p.py, self.g.xc, self.g.yc, self.ax);
        if !ax.is_multiple_of(self.g.p.swa) || !self.mx.is_multiple_of(self.swr) {
            return Err("gpt-oss: segments must divide their halves".into());
        }
        for u in 0..self.g.units() {
            let (x0, y0) = self.g.unit_origin(u);
            for r in 0..py {
                let y = y0 + self.g.wl(r, yc);
                let pts: Vec<P> = (0..=px).map(|lx| (x0 + lx, y)).collect();
                let (o, h) = self
                    .g
                    .line2(&pts, xc as usize, XA, XB, X2, MX, self.g.p.g2x)?;
                self.g.rootq.insert((x0 + xc, y), h);
                for (p, v) in o {
                    self.g.wrole_x.insert(p, v);
                }
                self.g.add(x0 + xc, y, LB, RX_R | TX_R)?;
                let wpts: Vec<P> = (0..ax).map(|c| (x0 + c, y)).collect();
                for seg in wpts.chunks(self.g.p.swa as usize) {
                    let root = seg.len() / 2;
                    let (o, (lo, hi)) = self.g.line(seg, root, SA, SB, SM, false)?;
                    for (p, (role, snd, rcv)) in o {
                        self.g
                            .wseg
                            .insert(p, (role, [rcv, None, Some(snd), Some(SM)]));
                    }
                    let two = lo.is_some() && hi.is_some();
                    self.g.wseg.insert(
                        seg[root],
                        (
                            if two { 4 } else { 6 },
                            [or(lo, hi), if two { hi } else { None }, Some(SM), None],
                        ),
                    );
                }
            }
            for c in 0..px {
                let x = x0 + self.g.wl(c, xc);
                let pts: Vec<P> = (0..=py).map(|ly| (x, y0 + ly)).collect();
                let (o, h) = self
                    .g
                    .line2(&pts, yc as usize, YA, YB, Y2, MY, self.g.p.g2y)?;
                self.g.rootq.insert((x, y0 + yc), h);
                for (p, v) in o {
                    self.g.wrole_y.insert(p, v);
                }
                self.g.add(x, y0 + yc, LB, RX_R | TX_R)?;
            }
            let y = y0 + yc;
            let cpts: Vec<P> = (0..self.mx).map(|c| (x0 + xc + 1 + c, y)).collect();
            for seg in cpts.chunks(self.swr as usize) {
                for (i, &p) in seg.iter().enumerate() {
                    self.rout.insert(p, i as u32);
                }
                let root = seg.len() / 2;
                let (o, (lo, hi)) = self.g.line(seg, root, CS, CS2, CM, false)?;
                for (p, (role, snd, rcv)) in o {
                    self.g
                        .cyseg
                        .insert(p, (role, rcv, Some(snd), Some(CM), None));
                }
                let two = lo.is_some() && hi.is_some();
                self.g.cyseg.insert(
                    seg[root],
                    (
                        if two { 4 } else { 6 },
                        or(lo, hi),
                        Some(CM),
                        None,
                        if two { hi } else { None },
                    ),
                );
            }
        }
        for u in 0..self.g.units() - 1 {
            self.g.handoff(u, u + 1)?;
        }
        self.head_in(self.g.units() - 1)?;
        for j in 0..self.hvc {
            let pts: Vec<P> = (0..=py).map(|ly| (j, self.hb0 + ly)).collect();
            let (o, h) = self
                .g
                .line2(&pts, self.hxc as usize, XA, XB, X2, HR3, self.g.p.g2h)?;
            self.hrow.extend(o);
            self.g.rootq.insert((j, self.hb0 + self.hxc), h);
            self.g.add(j, self.hb0 + self.hxc, LB, RX_R | TX_R)?;
        }
        if self.g.p.wl {
            self.wloop()?;
        }
        Ok(())
    }

    fn head_in(&mut self, ul: usize) -> R<()> {
        let (x0, y0) = self.g.unit_origin(ul);
        let (xc, yc, py) = (self.g.xc, self.g.yc, self.g.p.py);
        if y0 + self.g.uh != self.hb0 || py > self.mx {
            return Err("gpt-oss: the last unit sits on the band, PY <= MX".into());
        }
        for r in 0..py {
            let y = y0 + self.g.wl(r, yc);
            let xr = x0 + xc + 1 + r;
            let p1: Vec<P> = (x0 + xc..=xr).map(|x| (x, y)).collect();
            self.g.path(&p1, HA)?;
            self.g.cxq.entry((x0 + xc, y)).or_default().out = Some(HA);
            let yb = self.by(r);
            let p2: Vec<P> = (y..=yb).map(|yy| (xr, yy)).collect();
            self.g.path(&p2, HV2)?;
            self.g.relay.entry((xr, y)).or_default().post = Some((HA, HV2));
            if xr >= self.hvc {
                return Err("gpt-oss: the x path's column past the band".into());
            }
            self.hsrc.insert((xr, yb));
            let pts: Vec<P> = (0..self.hvc).map(|j| (j, yb)).collect();
            self.g.mcast(&pts, xr as usize, HM)?;
        }
        Ok(())
    }

    fn wloop(&mut self) -> R<()> {
        let (hb0, hxc, py) = (self.hb0, self.hxc, self.g.p.py);
        let yc = hb0 + hxc;
        let pts: Vec<P> = (0..self.hvc).map(|j| (j, yc)).collect();
        let (o, h) = self
            .g
            .line2(&pts, (self.hvc / 2) as usize, WA, WB, WC2, WM, self.g.p.g2w)?;
        self.wrow.extend(o);
        self.wroot = Some((self.hvc / 2, yc, h));
        let u0 = self.g.unit_origin(0);
        if u0.1 + self.g.uh != hb0 {
            return Err("gpt-oss: unit 0 sits on the band".into());
        }
        for i in 0..py {
            let yb = self.by(i);
            for j in 1..self.hvc {
                let b = if j < self.hvc - 1 {
                    rx('E') | tx('W')
                } else {
                    RX_R | tx('W')
                };
                self.g.add(j, yb, HE, b)?;
            }
            self.g.add(0, yb, HE, rx('E') | TX_R)?;
            self.hturn.insert((0, yb));
            let ci = u0.0 + self.g.xc + 1 + i;
            let yu = u0.1 + self.g.wl(i, self.g.yc);
            if ci >= self.hvc {
                return Err("gpt-oss: the return column past the band".into());
            }
            let mut pts: Vec<P> = (0..=ci).map(|x| (x, yb)).collect();
            pts.extend((yu..yb).rev().map(|yy| (ci, yy)));
            self.g.path(&pts, V1)?;
            let p3: Vec<P> = (u0.0 + self.g.xc..=ci).rev().map(|x| (x, yu)).collect();
            self.g.path(&p3, W2)?;
            self.g.relay.entry((ci, yu)).or_default().pre = Some((V1, W2));
            self.embrelay.insert((ci, yu));
            self.g.cxq.entry((u0.0 + self.g.xc, yu)).or_default().emb = Some(W2);
        }
        Ok(())
    }

    // ---------------------------------------------------------- rcfg words

    /// The base parks, never on the band's bus colour (a winner's own send colour at run time).
    fn parks(&self, p: P, n: usize) -> R<Vec<u32>> {
        if p.1 < self.hb0 {
            return self.g.parks(p, n);
        }
        let mut free: Vec<u32> = (0..NCOL)
            .filter(|&c| c != HE && self.g.get_route(p, c).is_none())
            .collect();
        free.extend((0..NCOL).filter(|&c| {
            c != HE
                && self
                    .g
                    .get_route(p, c)
                    .is_some_and(|b| b & TX_R == 0 && (b & 0xe0) != RX_R)
        }));
        if free.len() < n {
            return Err(format!("{p:?}: {} free colours < {n}", free.len()));
        }
        free.truncate(n);
        Ok(free)
    }

    fn tmpc(&self, p: P, finals: &[u32], comptime: &[u32]) -> R<u32> {
        let mut f = finals.to_vec();
        if p.1 >= self.hb0 {
            f.push(HE);
        }
        self.g.tmpc(p, &f, comptime)
    }

    pub fn rcfg_w(&self, x: u32, y: u32, r: u32, rwc: u32) -> R<Vec<u32>> {
        let p = (x, y);
        let rx_ = *self
            .g
            .wrole_x
            .get(&p)
            .ok_or_else(|| format!("{p:?}: no row role"))?;
        let ry = *self
            .g
            .wrole_y
            .get(&p)
            .ok_or_else(|| format!("{p:?}: no column role"))?;
        let need = [rx_.2, rx_.3, ry.2, ry.3]
            .iter()
            .filter(|v| v.is_none())
            .count();
        let pk = self.parks(p, need)?;
        let mut it = pk.into_iter();
        let mut w = self.g.words(p, 16)?;
        let mut nx = |v: Option<u32>| v.unwrap_or_else(|| it.next().unwrap());
        let q = [
            nx(rx_.2),
            nx(rx_.3),
            nx(ry.2),
            nx(ry.3),
            MX,
            MY,
            rx_.1,
            ry.1,
        ];
        let (role_s, sq) = self.g.wseg.get(&p).copied().unwrap_or((5, [None; 4]));
        let dflt = [q[5], q[4], q[6], q[4]];
        let sq: Vec<u32> = (0..4).map(|i| sq[i].unwrap_or(dflt[i])).collect();
        let rel = self.g.relay.get(&p).cloned().unwrap_or_default();
        let n = self.g.p.hcc;
        let pre = rel.pre.map_or([0, 0, 0], |(a, b)| [n, a, b]);
        let post = rel.post.map_or([0, 0, 0], |(a, b)| [n, a, b]);
        w.extend_from_slice(&q);
        w.extend_from_slice(&[rx_.0, ry.0, role_s, rwc, q[4]]);
        w.extend_from_slice(&sq);
        w.push(r);
        w.extend_from_slice(&pre);
        w.extend_from_slice(&post);
        w.push(self.tmpc(p, &q[..6], &Geo::CT6)?);
        w.push(u32::from(self.embrelay.contains(&p)));
        Ok(w)
    }

    pub fn rcfg_cx(&self, x: u32, y: u32, first: bool) -> R<Vec<u32>> {
        let p = (x, y);
        let pk = self.parks(p, 3)?;
        let mut w = self.g.words(p, 12)?;
        let h = self.g.cxq.get(&p).cloned().unwrap_or_default();
        let (lo, hi) = self.g.rootq[&p];
        let q = [
            lo.unwrap_or(pk[0]),
            hi.unwrap_or(pk[1]),
            pk[2],
            h.inn.unwrap_or(pk[0]),
            LB,
            MX,
            h.out.unwrap_or(pk[1]),
        ];
        let fl = u32::from(first || h.inn.is_none())
            | if h.out.is_none() { 2 } else { 0 }
            | if self.g.p.wl && h.emb.is_some() { 4 } else { 0 };
        let q6 = match h.emb {
            Some(e) => e,
            None => self.parks(p, 4)?[3],
        };
        let fin = [q[0], q[1], q[2], q[3], q6];
        let mut s = fin.to_vec();
        s.sort_unstable();
        s.dedup();
        if s.len() != 5 {
            return Err(format!("{p:?}: CX queues share a colour {fin:?}"));
        }
        w.extend_from_slice(&q);
        w.push(fl);
        w.push(self.tmpc(p, &fin, &[13, 14, 17, 18, 19])?);
        w.push(q6);
        Ok(w)
    }

    pub fn rcfg_cy(&self, x: u32, y: u32) -> R<Vec<u32>> {
        let p = (x, y);
        let pk = self.parks(p, 4)?;
        let mut w = self.g.words(p, 12)?;
        let (role, si, so, sm, s2) = self
            .g
            .cyseg
            .get(&p)
            .copied()
            .unwrap_or((5, None, None, None, None));
        let (lo, hi) = self.g.rootq[&p];
        let q = [
            lo.unwrap(),
            hi.unwrap(),
            LB,
            si.unwrap_or(pk[0]),
            LB,
            MY,
            so.unwrap_or(pk[3]),
        ];
        let tail = [role, sm.unwrap_or(pk[1]), s2.unwrap_or(pk[2])];
        let fin = [q[0], q[1], q[2], q[3], tail[1], tail[2]];
        w.extend_from_slice(&q);
        w.extend_from_slice(&tail);
        w.push(self.tmpc(p, &fin, &Geo::CT6)?);
        w.push(self.rout.get(&p).copied().unwrap_or(0));
        Ok(w)
    }

    pub fn rcfg_head(&self, x: u32, y: u32) -> R<Vec<u32>> {
        let p = (x, y);
        let pk = self.parks(p, 6)?;
        let mut w = self.g.words(p, 12)?;
        let ro = *self
            .hrow
            .get(&p)
            .ok_or_else(|| format!("{p:?}: no head role"))?;
        let src = self.hsrc.contains(&p);
        let bus = self.g.p.wl && self.hturn.contains(&p);
        let q = [
            ro.2.unwrap_or(pk[0]),
            ro.3.unwrap_or(pk[1]),
            if src { HV2 } else { pk[2] },
            if bus { HE } else { pk[3] },
            if src { pk[5] } else { HM },
            if self.g.p.wl { HR3 } else { pk[4] },
        ];
        let mut s = q.to_vec();
        s.sort_unstable();
        s.dedup();
        if s.len() != 6 {
            return Err(format!("{p:?}: head queues share a colour {q:?}"));
        }
        let fl = u32::from(self.g.p.wl) | if bus { 2 } else { 0 };
        w.extend_from_slice(&q);
        w.extend_from_slice(&[
            ro.1,
            HM,
            if ro.0 == 3 { 1 } else { ro.0 },
            if src { 2 } else { 0 },
        ]);
        w.push(self.tmpc(p, &q, &Geo::CT6)?);
        w.extend_from_slice(&[fl, 0]);
        w.push(if self.g.p.wl { HE } else { pk[5] });
        w.push(if self.g.p.wl { V1 } else { pk[5] });
        Ok(w)
    }

    pub fn rcfg_hctl(&self, x: u32, y: u32, vr: u32) -> R<Vec<u32>> {
        let p = (x, y);
        let mut w = self.g.words(p, 12)?;
        let (lo, hi) = self.g.rootq[&p];
        let (lo, hi) = (lo.unwrap(), hi.unwrap());
        let pk = self.parks(p, 4)?;
        let (role, q5, q6, q7, oq3, oq4);
        if self.g.p.wl {
            let wr = self.wroot.unwrap();
            if (x, y) == (wr.0, wr.1) {
                let (wl_, wh_) = wr.2;
                role = 4 | if wl_.is_some() { 8 } else { 0 } | if wh_.is_some() { 16 } else { 0 };
                q5 = or(wl_, Some(pk[0])).unwrap();
                q6 = or(wh_, Some(pk[1])).unwrap();
                q7 = pk[2];
                oq3 = WM;
            } else {
                let (rl, s, r1, r2) = self.wrow[&p];
                role = rl;
                q5 = r1.unwrap_or(pk[0]);
                q6 = r2.unwrap_or(pk[1]);
                q7 = WM;
                oq3 = s;
            }
            oq4 = HR3;
        } else {
            role = 0;
            q5 = pk[0];
            q6 = pk[1];
            q7 = pk[2];
            oq3 = pk[3];
            oq4 = pk[3];
        }
        let fin = [lo, hi, LB, q5, q6, q7];
        w.extend_from_slice(&[lo, hi, LB, LB, x * vr]);
        w.push(self.tmpc(p, &fin, &Geo::CT6)?);
        w.extend_from_slice(&[q5, q6, q7, oq3, oq4, role, u32::from(self.g.p.wl)]);
        Ok(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;

    /// sha256 of every PE's rcfg words by kind, as `tools/split/gptoss/rcfg_hash.py` (json `[x, y, words..]`).
    fn hashes(name: &str) -> BTreeMap<&'static str, String> {
        let tiny = name == "tiny";
        let mut g = GptossGeo::new(
            name,
            if tiny { 2 } else { 16 },
            if tiny { 2 } else { 24 },
            true,
        )
        .unwrap();
        g.build().unwrap();
        let vr = if tiny { 8 } else { 272 };
        let mut h: BTreeMap<&'static str, Sha256> = BTreeMap::new();
        let u0 = g.g.unit_origin(0);
        for y in 0..g.h {
            for x in 0..g.w {
                let k = g.kind(x, y);
                let w = match k {
                    "wa" | "wm" => {
                        let (x0, y0) = (x / g.g.uw * g.g.uw, y / g.g.uh * g.g.uh);
                        let ly = y - y0;
                        let r = ly - u32::from(ly > g.g.yc);
                        g.rcfg_w(x, y, r, u32::from(x - x0 == g.g.xc + 1)).unwrap()
                    }
                    "cx" => g
                        .rcfg_cx(
                            x,
                            y,
                            x >= u0.0 && x < u0.0 + g.g.uw && y >= u0.1 && y < u0.1 + g.g.uh,
                        )
                        .unwrap(),
                    "cya" | "cym" => g.rcfg_cy(x, y).unwrap(),
                    "head" => g.rcfg_head(x, y).unwrap(),
                    "hctl" => g.rcfg_hctl(x, y, vr).unwrap(),
                    _ => continue,
                };
                let mut s = format!("[{x}, {y}");
                for v in w {
                    s += &format!(", {v}");
                }
                s += "]";
                h.entry(k).or_default().update(s.as_bytes());
            }
        }
        h.into_iter()
            .map(|(k, v)| {
                (
                    k,
                    v.finalize()
                        .iter()
                        .take(8)
                        .map(|b| format!("{b:02x}"))
                        .collect(),
                )
            })
            .collect()
    }

    #[test]
    fn the_tiny_rcfg_words_equal_the_python_reference() {
        let h = hashes("tiny");
        let want = [
            ("cx", "628fe03ddc3d017b"),
            ("cya", "eac9ea48268927c2"),
            ("cym", "6a0582f962e0d762"),
            ("hctl", "2e1440be35e7f520"),
            ("head", "34ff5b1b63bc1165"),
            ("wa", "82213837a5130570"),
            ("wm", "e9d7b11e3492d59a"),
        ];
        for (k, v) in want {
            assert_eq!(h[k], v, "{k}");
        }
    }

    #[test]
    #[ignore]
    fn the_full_rcfg_words_equal_the_python_reference() {
        let h = hashes("full");
        let want = [
            ("cx", "efb77748b2fc1808"),
            ("cya", "c674c4d3a0d18b98"),
            ("cym", "2c9fa6ef472fee38"),
            ("hctl", "3791a9526f56bd54"),
            ("head", "c3aa173f893a83b7"),
            ("wa", "4be51a6e41971cdc"),
            ("wm", "f2d779a2812a7b48"),
        ];
        for (k, v) in want {
            assert_eq!(h[k], v, "{k}");
        }
    }
}
