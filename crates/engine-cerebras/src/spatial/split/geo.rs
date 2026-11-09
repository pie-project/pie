//! The role-split wafer's geometry: PE kinds, routes and queue colours
//! (`rcfg`) of every PE. A word-for-word port of the Python host reference
//! the kernels were brought up with; see the module docs of [`super`] and
//! `split/README.md` for the layout.
//!
//! Units of PX x PY weight PEs (+ the CX column at local x = PX / 2, the CY
//! row at local y = PY / 2, a filler corner), unit (ux, uy) at
//! (ux (PX + 1), uy (PY + 1)), UNX x UNY of them visited as a snake (down
//! unit column 0, up column 1, ...). The head (HC weight columns + a control
//! column in the middle, HR rows) east of the units. Collectives are one- or
//! two-level chains into the CX / CY roots; handoffs carry x from unit to
//! unit and into the head; the wafer token loop (WL) max-reduces the head
//! rows' winners down the head control column and sends the winner's
//! embedding back to unit 0's CX column.

use std::collections::HashMap;

pub mod gptoss;

type P = (u32, u32);

const fn rx(d: char) -> u32 {
    match d {
        'W' => 0x00,
        'E' => 0x20,
        'S' => 0x40,
        'N' => 0x60,
        _ => 0x80,
    }
}
const fn tx(d: char) -> u32 {
    match d {
        'W' => 0x01,
        'E' => 0x02,
        'S' => 0x04,
        'N' => 0x08,
        _ => 0x10,
    }
}
const RX_R: u32 = 0x80;
const TX_R: u32 = 0x10;
fn opp(d: char) -> char {
    match d {
        'W' => 'E',
        'E' => 'W',
        'N' => 'S',
        _ => 'N',
    }
}

// Colours. The rows' and columns' reductions (level 1: XA / XB, YA / YB;
// level 2: X2 / Y2) and multicasts (MX, MY), the roots' loop-back (LB); on
// the weight rows the scores segment (SA / SB chain, SM multicast) and GDN's
// (k, q) exchange (SG); on the CY row the o-norm segment (CS / CS2, CM); the
// handoffs (HA, HB, HX; HV the vertical ones leaving even / odd units); the
// head's relay north (HN), column multicast (HM) and row word (HR3); the
// wafer token loop's max-reduce (WA / WB / WC2) and winner multicast (WM),
// and the embedding return (VA up the head column, VS down, HS between the
// CX row pair, HE west into unit 0's CX). A PE's unused input queues park on
// colours nothing routes to its ramp (never its own send colour).
const XA: u32 = 1;
const XB: u32 = 2;
const MX: u32 = 3;
const YA: u32 = 4;
const YB: u32 = 5;
const MY: u32 = 6;
const LB: u32 = 7;
const SA: u32 = 8;
const SB: u32 = 13;
const SM: u32 = 9;
const SG: u32 = 10;
const CS: u32 = 11;
const CS2: u32 = 17;
const CM: u32 = 12;
const HA: u32 = 0;
const HB: u32 = 11;
const HX: u32 = 12;
const HV: [u32; 2] = [15, 14];
const X2: [u32; 2] = [18, 19];
const Y2: [u32; 2] = [20, 16];
const HN: u32 = 4;
const HM: u32 = 6;
const HR3: u32 = 3;
const WA: u32 = 10;
const WB: u32 = 11;
const WC2: [u32; 2] = [13, 14];
const WM: u32 = 15;
const VA: u32 = 5;
const VS: u32 = 8;
const HS: u32 = 9;
const HE: u32 = 17;
const NCOL: u32 = 21;

/// Python's `a or b` over optional colours: colour 0 counts as absent.
fn or(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match a {
        Some(v) if v != 0 => Some(v),
        _ => b,
    }
}

/// A chain member: (role, send colour, level-1 recv, level-2 recv).
pub type Role = (u32, u32, Option<u32>, Option<u32>);

/// A single-level line member: (role, send colour, recv colour).
type LineRole = (u32, u32, Option<u32>);

/// A chain root's receive colours, one a half (`None`: that half is empty).
type RootQ = (Option<u32>, Option<u32>);

/// A CY segment member: (role, recv, send, multicast, level-2 recv).
type CySeg = (u32, Option<u32>, Option<u32>, Option<u32>, Option<u32>);

#[derive(Default, Clone)]
struct Cxq {
    inn: Option<u32>,
    out: Option<u32>,
    emb: Option<u32>,
}

#[derive(Default, Clone)]
struct Relay {
    pre: Option<(u32, u32)>,
    post: Option<(u32, u32)>,
}

/// The geometry's parameters (the prep's meta plus the build's overrides).
#[derive(Clone, Debug)]
pub struct Params {
    pub px: u32,
    pub py: u32,
    pub unx: u32,
    pub uny: u32,
    pub hc: u32,
    pub hr: u32,
    pub hcc: u32,
    pub swa: u32,
    pub swg: u32,
    pub swv: u32,
    pub dkg: u32,
    pub dkh: u32,
    pub g2x: u32,
    pub g2y: u32,
    pub g2h: u32,
    pub wl: bool,
    pub g2w: u32,
}

pub struct Geo {
    pub p: Params,
    pub xc: u32,
    pub yc: u32,
    pub uw: u32,
    pub uh: u32,
    pub hx0: u32,
    pub hxc: u32,
    pub w: u32,
    pub h: u32,
    pub order: Vec<(u32, u32)>,
    routes: Vec<Vec<(u32, u32)>>,
    relay: HashMap<P, Relay>,
    wrole_x: HashMap<P, Role>,
    wrole_y: HashMap<P, Role>,
    cxq: HashMap<P, Cxq>,
    cyseg: HashMap<P, CySeg>,
    wseg: HashMap<P, (u32, [Option<u32>; 4])>,
    gsrc: HashMap<P, (u32, Option<u32>)>,
    hrole: HashMap<P, u32>,
    hrow: HashMap<P, Role>,
    rootq: HashMap<P, RootQ>,
    wrow: HashMap<P, Role>,
    hturn: HashMap<P, u32>,
    wroot: Option<(u32, u32, RootQ)>,
    /// Explicit unit origins (Gemma's layout, `gemma.rs`); `None`: the UNX x UNY snake.
    origins: Option<Vec<(u32, u32)>>,
    /// Route slots in a weight PE's rcfg (16; Gemma 18).
    wnr: usize,
    /// Words a unit handoff relays (Hc; Gemma 2 Hc + 1).
    hand_n: u32,
}

type R<T> = std::result::Result<T, String>;

pub mod gemma;

impl Geo {
    pub fn new(p: Params) -> Geo {
        let (xc, yc) = (p.px / 2, p.py / 2);
        let (uw, uh) = (p.px + 1, p.py + 1);
        let hx0 = p.unx * uw;
        let hxc = p.hc / 2;
        let w = hx0 + p.hc + 1;
        let h = (p.uny * uh).max(p.hr);
        let mut order = Vec::new();
        for ux in 0..p.unx {
            if ux % 2 == 0 {
                order.extend((0..p.uny).map(|uy| (ux, uy)));
            } else {
                order.extend((0..p.uny).rev().map(|uy| (ux, uy)));
            }
        }
        Geo {
            p,
            xc,
            yc,
            uw,
            uh,
            hx0,
            hxc,
            w,
            h,
            order,
            routes: vec![Vec::new(); (w * h) as usize],
            relay: HashMap::new(),
            wrole_x: HashMap::new(),
            wrole_y: HashMap::new(),
            cxq: HashMap::new(),
            cyseg: HashMap::new(),
            wseg: HashMap::new(),
            gsrc: HashMap::new(),
            hrole: HashMap::new(),
            hrow: HashMap::new(),
            rootq: HashMap::new(),
            wrow: HashMap::new(),
            hturn: HashMap::new(),
            wroot: None,
            origins: None,
            wnr: 16,
            hand_n: 0,
        }
    }

    /// Local index `i` of an axis with its control line at `c`, skipping it.
    pub fn wl(&self, i: u32, c: u32) -> u32 {
        i + u32::from(i >= c)
    }

    pub fn unit_origin(&self, u: usize) -> (u32, u32) {
        if let Some(o) = &self.origins {
            return o[u];
        }
        let (ux, uy) = self.order[u];
        (ux * self.uw, uy * self.uh)
    }

    pub fn hx(&self, c: u32) -> u32 {
        self.hx0 + c + u32::from(c >= self.hxc)
    }

    pub fn units(&self) -> usize {
        self.origins.as_ref().map_or(self.order.len(), |o| o.len())
    }

    fn route(&self, p: P) -> &Vec<(u32, u32)> {
        &self.routes[(p.1 * self.w + p.0) as usize]
    }

    fn get_route(&self, p: P, c: u32) -> Option<u32> {
        self.route(p).iter().find(|r| r.0 == c).map(|r| r.1)
    }

    fn add(&mut self, x: u32, y: u32, c: u32, b: u32) -> R<()> {
        let d = &mut self.routes[(y * self.w + x) as usize];
        if let Some(e) = d.iter().find(|r| r.0 == c) {
            if e.1 != b {
                return Err(format!(
                    "route clash at ({x}, {y}) colour {c}: {:#x} vs {b:#x}",
                    e.1
                ));
            }
            return Ok(());
        }
        d.push((c, b));
        Ok(())
    }

    fn dirn(a: P, b: P) -> char {
        if b.0 == a.0 + 1 {
            'E'
        } else if b.0 + 1 == a.0 {
            'W'
        } else if b.1 == a.1 + 1 {
            'S'
        } else {
            'N'
        }
    }

    /// Single-level chains toward pts[root] on colours a / b, the root's
    /// multicast on m.
    fn line(
        &mut self,
        pts: &[P],
        root: usize,
        a: u32,
        b: u32,
        m: u32,
        lp: bool,
    ) -> R<(Vec<(P, LineRole)>, RootQ)> {
        let mut out = Vec::new();
        let n = pts.len();
        for (i, &p) in pts.iter().enumerate() {
            if i == root {
                continue;
            }
            let low = i < root;
            let k = if low { root - i } else { i - root };
            let kmax = if low { root } else { n - 1 - root };
            let send = if low {
                if k % 2 == 1 { a } else { b }
            } else if k % 2 == 1 {
                b
            } else {
                a
            };
            let recv = if low {
                if (k + 1) % 2 == 1 { a } else { b }
            } else if (k + 1) % 2 == 1 {
                b
            } else {
                a
            };
            let tow = Self::dirn(p, if low { pts[i + 1] } else { pts[i - 1] });
            let awy = opp(tow);
            self.add(p.0, p.1, send, RX_R | tx(tow))?;
            if k < kmax {
                self.add(p.0, p.1, recv, rx(awy) | TX_R)?;
                self.add(p.0, p.1, m, rx(tow) | TX_R | tx(awy))?;
            } else {
                self.add(p.0, p.1, m, rx(tow) | TX_R)?;
            }
            out.push((
                p,
                (
                    if k == kmax { 0 } else { 1 },
                    send,
                    if k < kmax { Some(recv) } else { None },
                ),
            ));
        }
        let r = pts[root];
        let mut t = if lp { TX_R } else { 0 };
        let (mut lo, mut hi) = (None, None);
        if root > 0 {
            let d = Self::dirn(r, pts[root - 1]);
            self.add(r.0, r.1, a, rx(d) | TX_R)?;
            lo = Some(a);
            t |= tx(d);
        }
        if root < n - 1 {
            let d = Self::dirn(r, pts[root + 1]);
            self.add(r.0, r.1, b, rx(d) | TX_R)?;
            hi = Some(b);
            t |= tx(d);
        }
        self.add(r.0, r.1, m, RX_R | t)?;
        Ok((out, (lo, hi)))
    }

    /// Two-level streamed halves toward pts[root] (groups of g).
    #[allow(clippy::too_many_arguments)]
    fn line2(
        &mut self,
        pts: &[P],
        root: usize,
        a: u32,
        b: u32,
        c2: [u32; 2],
        m: u32,
        g: u32,
    ) -> R<(Vec<(P, Role)>, RootQ)> {
        if g == 0 {
            let (o, h) = self.line(pts, root, a, b, m, false)?;
            return Ok((
                o.into_iter()
                    .map(|(p, v)| (p, (v.0, v.1, v.2, None)))
                    .collect(),
                h,
            ));
        }
        let g = g as usize;
        let mut out = Vec::new();
        let n = pts.len();
        for (i, &p) in pts.iter().enumerate() {
            if i == root {
                continue;
            }
            let low = i < root;
            let half = if low { 0 } else { 1 };
            let k = if low { root - i } else { i - root };
            let kmax = if low { root } else { n - 1 - root };
            let qn = kmax.div_ceil(g);
            let q = (k - 1) / g;
            let kk = k - q * g;
            let gs = if q == qn - 1 { kmax - q * g } else { g };
            let tow = Self::dirn(p, if low { pts[i + 1] } else { pts[i - 1] });
            let awy = opp(tow);
            self.add(
                p.0,
                p.1,
                m,
                rx(tow) | TX_R | if k < kmax { tx(awy) } else { 0 },
            )?;
            let s2 = c2[(q + half) % 2];
            let r2 = c2[(q + 1 + half) % 2];
            let l1s = if low {
                if k % 2 == 1 { a } else { b }
            } else if k % 2 == 1 {
                b
            } else {
                a
            };
            let l1r = if low {
                if (k + 1) % 2 == 1 { a } else { b }
            } else if (k + 1) % 2 == 1 {
                b
            } else {
                a
            };
            if kk > 1 {
                self.add(p.0, p.1, l1s, RX_R | tx(tow))?;
                if kk < gs {
                    self.add(p.0, p.1, l1r, rx(awy) | TX_R)?;
                }
                if q + 1 < qn {
                    self.add(p.0, p.1, r2, rx(awy) | tx(tow))?;
                }
                out.push((
                    p,
                    (
                        if kk == gs { 0 } else { 1 },
                        l1s,
                        if kk < gs { Some(l1r) } else { None },
                        None,
                    ),
                ));
            } else {
                self.add(p.0, p.1, s2, RX_R | tx(tow))?;
                let has1 = gs > 1;
                if has1 {
                    self.add(p.0, p.1, l1r, rx(awy) | TX_R)?;
                }
                if q + 1 < qn {
                    self.add(p.0, p.1, r2, rx(awy) | TX_R)?;
                    out.push((
                        p,
                        (
                            if has1 { 2 } else { 1 },
                            s2,
                            Some(if has1 { l1r } else { r2 }),
                            if has1 { Some(r2) } else { None },
                        ),
                    ));
                } else {
                    out.push((
                        p,
                        (
                            if has1 { 3 } else { 0 },
                            s2,
                            if has1 { Some(l1r) } else { None },
                            None,
                        ),
                    ));
                }
            }
        }
        let r = pts[root];
        let mut t = 0;
        let (mut lo, mut hi) = (None, None);
        if root > 0 {
            let d = Self::dirn(r, pts[root - 1]);
            self.add(r.0, r.1, c2[0], rx(d) | TX_R)?;
            lo = Some(c2[0]);
            t |= tx(d);
        }
        if root < n - 1 {
            let d = Self::dirn(r, pts[root + 1]);
            self.add(r.0, r.1, c2[1], rx(d) | TX_R)?;
            hi = Some(c2[1]);
            t |= tx(d);
        }
        self.add(r.0, r.1, m, RX_R | t)?;
        Ok((out, (lo, hi)))
    }

    fn path(&mut self, pts: &[P], c: u32) -> R<()> {
        for (i, &p) in pts.iter().enumerate() {
            let r = if i == 0 {
                RX_R
            } else {
                rx(Self::dirn(p, pts[i - 1]))
            };
            let t = if i == pts.len() - 1 {
                TX_R
            } else {
                tx(Self::dirn(p, pts[i + 1]))
            };
            self.add(p.0, p.1, c, r | t)?;
        }
        Ok(())
    }

    fn mcast(&mut self, pts: &[P], src: usize, c: u32) -> R<()> {
        let n = pts.len();
        for (i, &p) in pts.iter().enumerate() {
            if i == src {
                let t = (if i > 0 {
                    tx(Self::dirn(p, pts[i - 1]))
                } else {
                    0
                }) | (if i < n - 1 {
                    tx(Self::dirn(p, pts[i + 1]))
                } else {
                    0
                });
                self.add(p.0, p.1, c, RX_R | t)?;
                continue;
            }
            let (prev, nxt) = if i < src {
                (pts[i + 1], if i > 0 { Some(pts[i - 1]) } else { None })
            } else {
                (pts[i - 1], if i < n - 1 { Some(pts[i + 1]) } else { None })
            };
            self.add(
                p.0,
                p.1,
                c,
                rx(Self::dirn(p, prev)) | TX_R | nxt.map_or(0, |q| tx(Self::dirn(p, q))),
            )?;
        }
        Ok(())
    }

    // ---------------------------------------------------------- build

    pub fn build(&mut self, has_att: bool, has_gdn: bool) -> R<()> {
        let (px, py, xc, yc) = (self.p.px, self.p.py, self.xc, self.yc);
        // A segment (scores, GDN's (k, q), the o-norm) is a line of adjacent PEs: it must not span
        // the control column (the gap at local x = XC).
        for (n, sw) in [
            ("SWA", self.p.swa),
            ("SWG", self.p.swg),
            ("SWV", self.p.swv),
        ] {
            if sw == 0 || xc % sw != 0 {
                return Err(format!(
                    "split geometry: {n} {sw} does not divide XC {xc} (a segment would span the control column)"
                ));
            }
        }
        for u in 0..self.units() {
            let (x0, y0) = self.unit_origin(u);
            for r in 0..py {
                let y = y0 + self.wl(r, yc);
                let pts: Vec<P> = (0..=px).map(|lx| (x0 + lx, y)).collect();
                let (o, h) = self.line2(&pts, xc as usize, XA, XB, X2, MX, self.p.g2x)?;
                self.rootq.insert((x0 + xc, y), h);
                for (p, v) in o {
                    self.wrole_x.insert(p, v);
                }
                self.add(x0 + xc, y, LB, RX_R | TX_R)?;
                let wpts: Vec<P> = (0..px).map(|c| (x0 + self.wl(c, xc), y)).collect();
                if has_att {
                    let mut s0 = 0;
                    while s0 < px as usize {
                        let seg: Vec<P> =
                            wpts[s0..(s0 + self.p.swa as usize).min(wpts.len())].to_vec();
                        let root = seg.len() / 2;
                        let (o, (lo, hi)) = self.line(&seg, root, SA, SB, SM, false)?;
                        for (p, (role, snd, rcv)) in o {
                            self.wseg
                                .insert(p, (role, [rcv, None, Some(snd), Some(SM)]));
                        }
                        let two = lo.is_some() && hi.is_some();
                        self.wseg.insert(
                            seg[root],
                            (
                                if two { 4 } else { 6 },
                                [or(lo, hi), if two { hi } else { None }, Some(SM), None],
                            ),
                        );
                        s0 += self.p.swa as usize;
                    }
                }
                if has_gdn && r < self.p.dkh {
                    let mut s0 = 0;
                    while s0 < px as usize {
                        let seg: Vec<P> =
                            wpts[s0..(s0 + self.p.swg as usize).min(wpts.len())].to_vec();
                        s0 += self.p.swg as usize;
                        if seg.len() == 1 {
                            self.gsrc.insert(seg[0], (0xC000 | (r % self.p.dkg), None));
                            continue;
                        }
                        let own = (r / self.p.dkg) as usize;
                        self.mcast(&seg, own, SG)?;
                        for (i, &p) in seg.iter().enumerate() {
                            self.gsrc.insert(
                                p,
                                (
                                    if i == own {
                                        0x8000 | (r % self.p.dkg)
                                    } else {
                                        0
                                    },
                                    Some(SG),
                                ),
                            );
                        }
                    }
                } else if has_gdn {
                    for &p in &wpts {
                        self.gsrc.insert(p, (0x4000, None));
                    }
                }
            }
            for c in 0..px {
                let x = x0 + self.wl(c, xc);
                let pts: Vec<P> = (0..=py).map(|ly| (x, y0 + ly)).collect();
                let (o, h) = self.line2(&pts, yc as usize, YA, YB, Y2, MY, self.p.g2y)?;
                self.rootq.insert((x, y0 + yc), h);
                for (p, v) in o {
                    self.wrole_y.insert(p, v);
                }
            }
            let y = y0 + yc;
            for c in 0..px {
                let x = x0 + self.wl(c, xc);
                self.add(x, y, LB, RX_R | TX_R)?;
            }
            if has_gdn {
                let cpts: Vec<P> = (0..px).map(|c| (x0 + self.wl(c, xc), y)).collect();
                let mut s0 = 0;
                while s0 < px as usize {
                    let seg: Vec<P> = cpts[s0..(s0 + self.p.swv as usize).min(cpts.len())].to_vec();
                    s0 += self.p.swv as usize;
                    if seg.len() == 1 {
                        self.cyseg.insert(seg[0], (5, None, None, None, None));
                        continue;
                    }
                    let root = seg.len() / 2;
                    let (o, (lo, hi)) = self.line(&seg, root, CS, CS2, CM, false)?;
                    for (p, (role, snd, rcv)) in o {
                        self.cyseg.insert(p, (role, rcv, Some(snd), Some(CM), None));
                    }
                    let two = lo.is_some() && hi.is_some();
                    self.cyseg.insert(
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
        }
        for u in 0..self.units().saturating_sub(1) {
            self.handoff(u, u + 1)?;
        }
        self.head_in(self.units() - 1)?;
        for y in 0..self.p.hr {
            let pts: Vec<P> = (0..=self.p.hc).map(|i| (self.hx0 + i, y)).collect();
            let (o, h) = self.line2(&pts, self.hxc as usize, XA, XB, X2, HR3, self.p.g2h)?;
            self.hrow.extend(o);
            self.rootq.insert((self.hx0 + self.hxc, y), h);
            self.add(self.hx0 + self.hxc, y, LB, RX_R | TX_R)?;
        }
        if self.p.wl {
            self.wloop()?;
        }
        Ok(())
    }

    fn wloop(&mut self) -> R<()> {
        let (hr, yc) = (self.p.hr, self.yc);
        let pts: Vec<P> = (0..hr).map(|y| (self.hx0 + self.hxc, y)).collect();
        let (o, h) = self.line2(&pts, (hr / 2) as usize, WA, WB, WC2, WM, self.p.g2w)?;
        self.wrow.extend(o);
        self.wroot = Some((self.hx0 + self.hxc, hr / 2, h));
        let u0 = self.unit_origin(0);
        // (explicit origins: the paths' clash checks guard the layout; unit 0's CX rows from its origin)
        if self.origins.is_none() && (u0.1 != 0 || self.unit_origin(self.units() - 1).1 != 0) {
            return Err("the wafer loop wants unit 0 and the last unit on row 0".into());
        }
        let ty = if self.origins.is_some() { u0.1 } else { 0 };
        for c in 0..self.p.hc {
            let x = self.hx(c);
            let (te, to) = (ty + self.wl(2 * c, yc), ty + self.wl(2 * c + 1, yc));
            if to != te + 1 || to >= hr {
                return Err(format!("head column {c}: CX rows {te} / {to}"));
            }
            self.add(x, hr - 1, VA, RX_R | tx('N'))?;
            for y in 1..hr - 1 {
                self.add(x, y, VA, rx('S') | tx('N'))?;
            }
            self.add(x, 0, VA, rx('S') | TX_R)?;
            if te > 0 {
                let p: Vec<P> = (0..=te).map(|y| (x, y)).collect();
                self.path(&p, VS)?;
            }
            self.path(&[(x, te), (x, to)], HS)?;
            for y in [te, to] {
                let p: Vec<P> = (u0.0 + self.xc..=x).rev().map(|xx| (xx, y)).collect();
                self.path(&p, HE)?;
                self.cxq.entry((u0.0 + self.xc, y)).or_default().emb = Some(HE);
            }
            *self.hturn.entry((x, 0)).or_insert(0) |= 1;
            *self.hturn.entry((x, te)).or_insert(0) |= 2;
            *self.hturn.entry((x, to)).or_insert(0) |= 4;
        }
        Ok(())
    }

    fn handoff(&mut self, ua: usize, ub: usize) -> R<()> {
        let (a0, b0) = (self.unit_origin(ua), self.unit_origin(ub));
        let (ax, bx) = if self.origins.is_some() {
            (a0.0, b0.0)
        } else {
            (self.order[ua].0, self.order[ub].0)
        };
        if ax != bx && a0.1 != b0.1 {
            return Err(format!(
                "units {ua} / {ub}: a horizontal handoff joins units on the same rows"
            ));
        }
        let (xc, yc, py) = (self.xc as i64, self.yc, self.p.py);
        for r in 0..py {
            let pr = self.wl(r, yc);
            let (xa, ya) = (a0.0 as i64 + xc, a0.1 + pr);
            if ax == bx {
                let hv = HV[ua % 2];
                let pc = self.wl(r, xc as u32) as i64;
                let xcc = a0.0 as i64 + pc;
                let st: i64 = if xcc > xa { 1 } else { -1 };
                let yb = b0.1 + pr;
                let vs: i64 = if yb > ya { 1 } else { -1 };
                let p1: Vec<P> = range_step(xa, xcc + st, st)
                    .map(|x| (x as u32, ya))
                    .collect();
                self.path(&p1, HA)?;
                let p2: Vec<P> = range_step(ya as i64, yb as i64 + vs, vs)
                    .map(|y| (xcc as u32, y as u32))
                    .collect();
                self.path(&p2, hv)?;
                let p3: Vec<P> = range_step(xcc, b0.0 as i64 + xc - st, -st)
                    .map(|x| (x as u32, yb))
                    .collect();
                self.path(&p3, HB)?;
                self.relay.entry((xcc as u32, ya)).or_default().post = Some((HA, hv));
                self.relay.entry((xcc as u32, yb)).or_default().pre = Some((hv, HB));
                self.cxq.entry((xa as u32, ya)).or_default().out = Some(HA);
                self.cxq
                    .entry(((b0.0 as i64 + xc) as u32, yb))
                    .or_default()
                    .inn = Some(HB);
            } else {
                let xb = b0.0 as i64 + xc;
                let st: i64 = if xb > xa { 1 } else { -1 };
                let p: Vec<P> = range_step(xa, xb + st, st)
                    .map(|x| (x as u32, ya))
                    .collect();
                self.path(&p, HX)?;
                self.cxq.entry((xa as u32, ya)).or_default().out = Some(HX);
                self.cxq.entry((xb as u32, ya)).or_default().inn = Some(HX);
            }
        }
        Ok(())
    }

    fn head_in(&mut self, ul: usize) -> R<()> {
        let a0 = self.unit_origin(ul);
        let (xc, yc, py) = (self.xc, self.yc, self.p.py);
        if self.p.hc * 2 != py {
            return Err("head columns = CX rows / 2".into());
        }
        for r in 0..py {
            let y = a0.1 + self.wl(r, yc);
            let xa = a0.0 + xc;
            let c = r / 2;
            let p: Vec<P> = (xa..=self.hx(c)).map(|x| (x, y)).collect();
            self.path(&p, HX)?;
            self.cxq.entry((xa, y)).or_default().out = Some(HX);
            if r % 2 == 1 {
                self.hrole.insert((self.hx(c), y), 1);
                self.path(&[(self.hx(c), y), (self.hx(c), y - 1)], HN)?;
            }
        }
        for c in 0..self.p.hc {
            let src = a0.1 + self.wl(2 * c, yc);
            self.hrole.insert((self.hx(c), src), 2);
            let pts: Vec<P> = (0..self.p.hr).map(|y| (self.hx(c), y)).collect();
            self.mcast(&pts, src as usize, HM)?;
        }
        Ok(())
    }

    // ---------------------------------------------------------- rcfg words

    fn words(&self, p: P, nr: usize) -> R<Vec<u32>> {
        let mut r = self.route(p).clone();
        if r.len() > nr {
            return Err(format!("{p:?} has {} routes > {nr}", r.len()));
        }
        r.sort_unstable();
        let mut w = vec![r.len() as u32];
        for (c, b) in r {
            w.push(c);
            w.push(b);
        }
        w.resize(1 + 2 * nr, 0);
        Ok(w)
    }

    fn parks(&self, p: P, n: usize) -> R<Vec<u32>> {
        let mut free: Vec<u32> = (0..NCOL)
            .filter(|&c| self.get_route(p, c).is_none())
            .collect();
        free.extend((0..NCOL).filter(|&c| {
            self.get_route(p, c)
                .is_some_and(|b| b & TX_R == 0 && (b & 0xe0) != RX_R)
        }));
        if free.len() < n {
            return Err(format!("{p:?}: {} free colours < {n}", free.len()));
        }
        free.truncate(n);
        Ok(free)
    }

    fn tmpc(&self, p: P, finals: &[u32], comptime: &[u32]) -> R<u32> {
        let ramp = |c: u32| self.get_route(p, c).is_some_and(|b| b & TX_R != 0);
        for c in 0..NCOL {
            if finals.contains(&c) || comptime.contains(&c) {
                continue;
            }
            if !ramp(c) {
                return Ok(c);
            }
        }
        for c in 0..NCOL {
            if !finals.contains(&c) && !ramp(c) {
                return Ok(c);
            }
        }
        Err(format!("{p:?}: no spare colour"))
    }

    const CT6: [u32; 6] = [13, 14, 17, 18, 19, 20];

    pub fn rcfg_weight(&self, x: u32, y: u32, r: u32) -> R<Vec<u32>> {
        let p = (x, y);
        let rx_ = *self
            .wrole_x
            .get(&p)
            .ok_or_else(|| format!("{p:?}: no row role"))?;
        let ry_ = *self
            .wrole_y
            .get(&p)
            .ok_or_else(|| format!("{p:?}: no column role"))?;
        let need = [rx_.2, rx_.3, ry_.2, ry_.3]
            .iter()
            .filter(|v| v.is_none())
            .count();
        let pk = self.parks(p, need)?;
        let mut it = pk.into_iter();
        let mut w = self.words(p, self.wnr)?;
        let mut nx = |v: Option<u32>| v.unwrap_or_else(|| it.next().unwrap());
        let q = [
            nx(rx_.2),
            nx(rx_.3),
            nx(ry_.2),
            nx(ry_.3),
            MX,
            MY,
            rx_.1,
            ry_.1,
        ];
        let (role_s, sq) = self.wseg.get(&p).copied().unwrap_or((5, [None; 4]));
        let dflt = [q[5], q[4], q[6], q[4]];
        let sq: Vec<u32> = (0..4).map(|i| sq[i].unwrap_or(dflt[i])).collect();
        let (gs, gc) = self.gsrc.get(&p).copied().unwrap_or((0x4000, None));
        let gc = gc.unwrap_or(q[4]);
        let rel = self.relay.get(&p).cloned().unwrap_or_default();
        let hn = if self.hand_n > 0 {
            self.hand_n
        } else {
            self.p.hcc
        };
        let pre = rel.pre.map_or([0, 0, 0], |(a, b)| [hn, a, b]);
        let post = rel.post.map_or([0, 0, 0], |(a, b)| [hn, a, b]);
        w.extend_from_slice(&q);
        w.extend_from_slice(&[rx_.0, ry_.0, role_s, gs, gc]);
        w.extend(sq);
        w.push(r);
        w.extend_from_slice(&pre);
        w.extend_from_slice(&post);
        w.push(self.tmpc(p, &q[..6], &Self::CT6)?);
        Ok(w)
    }

    pub fn rcfg_cx(&self, x: u32, y: u32, first: bool) -> R<Vec<u32>> {
        let p = (x, y);
        let pk = self.parks(p, 2)?;
        let mut w = self.words(p, 12)?;
        let h = self.cxq.get(&p).cloned().unwrap_or_default();
        let (lo, hi) = self.rootq[&p];
        let q = [
            lo.unwrap(),
            hi.unwrap(),
            LB,
            h.inn.unwrap_or(pk[0]),
            LB,
            MX,
            h.out.unwrap_or(pk[1]),
        ];
        let mut fl = u32::from(first || h.inn.is_none()) | if h.out.is_none() { 2 } else { 0 };
        let q6 = match h.emb {
            Some(e) => e,
            None => self.parks(p, 3)?[2],
        };
        if self.p.wl && h.emb.is_some() {
            fl |= 4;
        }
        let fin = [q[0], q[1], q[2], q[3], q6];
        w.extend_from_slice(&q);
        w.push(fl);
        w.push(self.tmpc(p, &fin, &[13, 14, 17, 18, 19])?);
        w.push(q6);
        Ok(w)
    }

    pub fn rcfg_cy(&self, x: u32, y: u32) -> R<Vec<u32>> {
        let p = (x, y);
        let pk = self.parks(p, 4)?;
        let mut w = self.words(p, 12)?;
        let (role, si, so, sm, s2) = self
            .cyseg
            .get(&p)
            .copied()
            .unwrap_or((5, None, None, None, None));
        let (lo, hi) = self.rootq[&p];
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
        w.push(self.tmpc(p, &fin, &Self::CT6)?);
        Ok(w)
    }

    pub fn rcfg_head(&self, x: u32, y: u32) -> R<Vec<u32>> {
        let p = (x, y);
        let pk = self.parks(p, 7)?;
        let mut w = self.words(p, 12)?;
        let ro = self.hrow[&p];
        let xr = self.hrole.get(&p).copied().unwrap_or(0);
        let mut q = [
            ro.2.unwrap_or(pk[0]),
            ro.3.unwrap_or(pk[1]),
            if xr == 1 || xr == 2 { HX } else { pk[2] },
            if xr == 2 { HN } else { pk[3] },
            if xr == 0 || xr == 1 { HM } else { pk[4] },
            ro.1,
            if xr == 1 {
                HN
            } else if xr == 2 {
                HM
            } else {
                pk[0]
            },
        ];
        let tr = if self.p.wl {
            self.hturn.get(&p).copied().unwrap_or(0)
        } else {
            0
        };
        if tr & 1 != 0 && tr & 2 == 0 {
            q[3] = VA;
        }
        if tr & 4 != 0 {
            if xr != 1 {
                return Err(format!("{p:?}: HS turn off relay A"));
            }
            q[3] = HS;
        }
        if tr & 2 != 0 {
            if xr != 2 {
                return Err(format!("{p:?}: VS turn off relay B"));
            }
            q[4] = if y == 0 { VA } else { VS };
        }
        let q7 = if self.p.wl { HR3 } else { pk[5] };
        let fl = u32::from(self.p.wl) | (tr << 1);
        let oq = if self.p.wl {
            [VA, VS, HE, HS]
        } else {
            [pk[6]; 4]
        };
        let fin = [q[0], q[1], q[2], q[3], q[4], q7];
        w.extend_from_slice(&q);
        w.push(if ro.0 == 3 { 1 } else { ro.0 });
        w.push(xr);
        w.push(self.tmpc(p, &fin, &Self::CT6)?);
        w.push(q7);
        w.push(fl);
        w.push(y);
        w.extend_from_slice(&oq);
        Ok(w)
    }

    pub fn rcfg_hctl(&self, x: u32, y: u32, vr: u32) -> R<Vec<u32>> {
        let p = (x, y);
        let mut w = self.words(p, 12)?;
        let (lo, hi) = self.rootq[&p];
        let (lo, hi) = (lo.unwrap(), hi.unwrap());
        let pk = self.parks(p, 4)?;
        let (role, q5, q6, q7, oq3, oq4);
        if self.p.wl {
            let wr = self.wroot.unwrap();
            let snd;
            if (x, y) == (wr.0, wr.1) {
                let (wl_, wh_) = wr.2;
                let mut rl = 4;
                snd = WM;
                q5 = or(wl_, Some(pk[0])).unwrap();
                q6 = or(wh_, Some(pk[1])).unwrap();
                q7 = pk[2];
                rl |= if wl_.is_some() { 8 } else { 0 } | if wh_.is_some() { 16 } else { 0 };
                role = rl;
            } else {
                let (rl, s, r1, r2) = self.wrow[&p];
                role = rl;
                snd = s;
                q5 = r1.unwrap_or(pk[0]);
                q6 = r2.unwrap_or(pk[1]);
                q7 = WM;
            }
            oq3 = snd;
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
        w.extend_from_slice(&[lo, hi, LB, LB, y * vr]);
        w.push(self.tmpc(p, &fin, &Self::CT6)?);
        w.extend_from_slice(&[q5, q6, q7, oq3, oq4, role, u32::from(self.p.wl)]);
        Ok(w)
    }
}

/// Python's `range(a, b, s)`.
fn range_step(a: i64, b: i64, s: i64) -> impl Iterator<Item = i64> {
    let mut v = a;
    std::iter::from_fn(move || {
        if (s > 0 && v < b) || (s < 0 && v > b) {
            let r = v;
            v += s;
            Some(r)
        } else {
            None
        }
    })
}
