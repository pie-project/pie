//! The fabric: physical colours, every PE's route table and queue bindings, and the hardware
//! invariants checked on them. [`Fabric::lay`] turns a [`Wiring`] into routes through a family
//! [`Palette`]; [`Fabric::validate`] refuses a program the CS-3 would hang or miscompute on.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use super::floor::{Coord, Dir, Floor, Grid};
use super::role::{PeRole, RoleKind};
use super::wiring::{Chain, Class, Half, Link, Multicast, Parity, Path, Port, Slot, Turn, Wiring};

/// A physical colour (0 .. [`COLOURS`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Colour(pub u8);

/// Colours a PE's router has.
pub const COLOURS: u8 = 21;

/// A route's switch word: receive from one port, transmit to a set (`RX_R` / `TX_R` bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct RouteWord(pub u32);

impl RouteWord {
    pub fn rx_ramp() -> u32 {
        0x80
    }
    pub fn tx_ramp() -> u32 {
        0x10
    }
    pub fn rx(d: Dir) -> u32 {
        match d {
            Dir::W => 0x00,
            Dir::E => 0x20,
            Dir::S => 0x40,
            Dir::N => 0x60,
        }
    }
    pub fn tx(d: Dir) -> u32 {
        match d {
            Dir::W => 0x01,
            Dir::E => 0x02,
            Dir::S => 0x04,
            Dir::N => 0x08,
        }
    }
    /// Delivers to the PE's own ramp.
    pub fn to_ramp(self) -> bool {
        self.0 & 0x10 != 0
    }
    /// Takes from the PE's own ramp.
    pub fn from_ramp(self) -> bool {
        self.0 & 0xe0 == 0x80
    }
}

/// A family's logical -> physical colour map. Families keep today's constants so every word is
/// unchanged; a future allocator may assign them instead ([`Palette::allocate`]).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Palette {
    pub map: BTreeMap<Class, Colour>,
}

/// The role-split kernels' colours (`XA`, `XB`, `MX`, ... as the kernels and tools name them):
/// every family's palette is this plus its own classes.
const BASE: [(Class, u8); 35] = {
    use Class::*;
    use Half::{Hi, Lo};
    [
        (RowL1(Lo), 1),
        (RowL1(Hi), 2),
        (RowCast, 3),
        (ColL1(Lo), 4),
        (ColL1(Hi), 5),
        (ColCast, 6),
        (Loopback, 7),
        (ScoreL1(Lo), 8),
        (ScoreL1(Hi), 13),
        (ScoreCast, 9),
        (GdnKq, 10),
        (OnormL1(Lo), 11),
        (OnormL1(Hi), 17),
        (OnormCast, 12),
        (HandoffOut, 0),
        (HandoffIn, 11),
        (HandoffAcross, 12),
        (HandoffVert(Parity::Even), 15),
        (HandoffVert(Parity::Odd), 14),
        (RowL2(Lo), 18),
        (RowL2(Hi), 19),
        (ColL2(Lo), 20),
        (ColL2(Hi), 16),
        (HeadRelay, 4),
        (HeadColCast, 6),
        (HeadRowWord, 3),
        (LoopL1(Lo), 10),
        (LoopL1(Hi), 11),
        (LoopL2(Lo), 13),
        (LoopL2(Hi), 14),
        (LoopCast, 15),
        (EmbUp, 5),
        (EmbDown, 8),
        (EmbSplit, 9),
        (EmbWest, 17),
    ]
};

/// The base colour of `c`, if it has one.
pub fn base_colour(c: Class) -> Option<u8> {
    BASE.iter().find(|(k, _)| *k == c).map(|&(_, v)| v)
}

impl Palette {
    /// The base colours with `own` added or replacing.
    pub fn base(own: &[(Class, u8)]) -> Palette {
        Palette {
            map: BASE
                .iter()
                .chain(own)
                .map(|&(k, c)| (k, Colour(c)))
                .collect(),
        }
    }

    pub fn get(&self, c: Class) -> Result<Colour, String> {
        self.map
            .get(&c)
            .copied()
            .ok_or_else(|| format!("palette: no colour for {c:?}"))
    }

    /// Assigns colours to every class `w` uses so no PE sees two classes on one colour, by
    /// graph colouring over the classes that meet at a PE. Not used by the shipping families
    /// (their palettes are fixed); the seam for the spatial compiler.
    ///
    /// Greedy over the classes in their order: each takes the lowest colour no class it meets
    /// has taken. Every class of a link counts as meeting at every PE of that link (a chain's
    /// level-1, level-2 and cast classes all), which over-approximates and so never clashes.
    pub fn allocate(floor: &Floor, w: &Wiring) -> Result<Palette, String> {
        let mut at: BTreeMap<Coord, BTreeSet<Class>> = BTreeMap::new();
        for link in &w.links {
            let (pts, classes): (Vec<Coord>, Vec<Class>) = match link {
                Link::Chain(c) => {
                    let mut cl = vec![c.l1[0], c.l1[1], c.cast];
                    if let Some(l2) = c.l2 {
                        cl.extend(l2.colours);
                    }
                    (c.pts.clone(), cl)
                }
                Link::Multicast(m) => (m.pts.clone(), vec![m.class]),
                Link::Path(p) => (p.pts.clone(), vec![p.class]),
                Link::Turn(t) => (vec![t.at], vec![t.class]),
            };
            for p in pts {
                if p.x >= floor.w || p.y >= floor.h {
                    return Err(format!("palette: ({}, {}) off the floor", p.x, p.y));
                }
                at.entry(p).or_default().extend(classes.iter().copied());
            }
        }
        let mut meets: BTreeMap<Class, BTreeSet<Class>> = BTreeMap::new();
        for set in at.values() {
            for &a in set {
                let e = meets.entry(a).or_default();
                e.extend(set.iter().copied().filter(|&b| b != a));
            }
        }
        let mut map: BTreeMap<Class, Colour> = BTreeMap::new();
        for (&class, others) in &meets {
            let taken: BTreeSet<u8> = others
                .iter()
                .filter_map(|o| map.get(o).map(|c| c.0))
                .collect();
            let c = (0..COLOURS)
                .find(|c| !taken.contains(c))
                .ok_or_else(|| format!("palette: no colour left for {class:?}"))?;
            map.insert(class, Colour(c));
        }
        Ok(Palette { map })
    }
}

/// A chain member's queues: its role code in the kernel's
/// chain protocol, its send colour, its level-1 / level-2 receive colours.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ChainMember {
    pub role: u32,
    pub send: Colour,
    pub recv1: Option<Colour>,
    pub recv2: Option<Colour>,
}

/// A chain root's receive colours, one a half; `None`: that half is empty.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RootQueues {
    pub lo: Option<Colour>,
    pub hi: Option<Colour>,
}

/// What a link left at one PE for one slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Endpoint {
    Member(ChainMember),
    Root(RootQueues),
    /// A multicast source / receiver, a path's sender / receiver: the colour.
    Send(Colour),
    Recv(Colour),
    /// A relay's turn: from colour -> to colour.
    Turn(Colour, Colour),
}

/// A PE's queue bindings, by slot: what the rcfg emitters read.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PeIo {
    pub slots: BTreeMap<Slot, Endpoint>,
    /// The family's [`super::wiring::Note`]s at this PE.
    pub notes: BTreeMap<&'static str, u32>,
}

/// One route at a PE.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Route {
    pub colour: Colour,
    pub word: RouteWord,
}

/// The laid fabric: routes and queue bindings of every PE.
#[derive(Clone, Debug, Serialize)]
pub struct Fabric {
    /// Every PE's routes, sorted by colour (one route a colour).
    pub routes: Grid<Vec<Route>>,
    pub io: Grid<PeIo>,
    /// The palette's [`Class::Loopback`] colour, for the roots' loop-back check.
    pub loopback: Option<Colour>,
}

/// Slots of the reductions (a segment's queues never share a colour with these at a PE).
const REDUCTIONS: [Slot; 4] = [
    Slot::RowReduce,
    Slot::ColReduce,
    Slot::HeadRowReduce,
    Slot::LoopReduce,
];
const SEGMENTS: [Slot; 3] = [Slot::Segment, Slot::Gdn, Slot::Onorm];

impl Endpoint {
    /// The colours it takes from the ramp's input queues.
    fn receives(&self) -> Vec<Colour> {
        match *self {
            Endpoint::Member(m) => m.recv1.into_iter().chain(m.recv2).collect(),
            Endpoint::Root(r) => r.lo.into_iter().chain(r.hi).collect(),
            Endpoint::Recv(c) | Endpoint::Turn(c, _) => vec![c],
            Endpoint::Send(_) => vec![],
        }
    }

    /// The colours it sends from the ramp.
    fn sends(&self) -> Vec<Colour> {
        match *self {
            Endpoint::Member(m) => vec![m.send],
            Endpoint::Send(c) | Endpoint::Turn(_, c) => vec![c],
            Endpoint::Root(_) | Endpoint::Recv(_) => vec![],
        }
    }
}

/// [`Fabric::lay`]'s working state.
struct Lay<'a> {
    palette: &'a Palette,
    routes: Grid<Vec<Route>>,
    io: Grid<PeIo>,
}

impl Lay<'_> {
    fn colour(&self, c: Class) -> Result<Colour, String> {
        let col = self.palette.get(c)?;
        if col.0 >= COLOURS {
            return Err(format!("palette: {c:?} -> colour {} >= {COLOURS}", col.0));
        }
        Ok(col)
    }

    fn at(&self, p: Coord) -> Result<(), String> {
        if self.routes.contains(p) {
            Ok(())
        } else {
            Err(format!(
                "fabric: ({}, {}) off the {} x {} floor",
                p.x, p.y, self.routes.w, self.routes.h
            ))
        }
    }

    /// One route; the same word again is a no-op, another word for the colour a clash
    ///.
    fn add(&mut self, p: Coord, c: Colour, word: u32) -> Result<(), String> {
        self.at(p)?;
        let d = self.routes.get_mut(p);
        if let Some(e) = d.iter().find(|r| r.colour == c) {
            if e.word.0 != word {
                return Err(format!(
                    "route clash at ({}, {}) colour {}: {:#x} vs {word:#x}",
                    p.x, p.y, c.0, e.word.0
                ));
            }
            return Ok(());
        }
        d.push(Route {
            colour: c,
            word: RouteWord(word),
        });
        Ok(())
    }

    /// One endpoint into a slot. Order-free: the same endpoint again is a no-op, and a receive
    /// and a send in one slot (in either order) make a [`Endpoint::Turn`] (a relay: the path
    /// arriving and the path leaving); anything else is a clash.
    fn bind(&mut self, p: Coord, slot: Slot, ep: Endpoint) -> Result<(), String> {
        if slot == Slot::Pass {
            return Ok(());
        }
        self.at(p)?;
        let io = self.io.get_mut(p);
        let merged = match io.slots.get(&slot) {
            None => ep,
            Some(&old) if old == ep => return Ok(()),
            Some(&Endpoint::Recv(a)) => match ep {
                Endpoint::Send(b) => Endpoint::Turn(a, b),
                _ => return Err(slot_clash(p, slot, Endpoint::Recv(a), ep)),
            },
            Some(&Endpoint::Send(b)) => match ep {
                Endpoint::Recv(a) => Endpoint::Turn(a, b),
                _ => return Err(slot_clash(p, slot, Endpoint::Send(b), ep)),
            },
            Some(&old) => return Err(slot_clash(p, slot, old, ep)),
        };
        io.slots.insert(slot, merged);
        Ok(())
    }

    fn dir(a: Coord, b: Coord) -> Result<Dir, String> {
        a.dir_to(b).ok_or_else(|| {
            format!(
                "fabric: ({}, {}) and ({}, {}) are not adjacent",
                a.x, a.y, b.x, b.y
            )
        })
    }

    /// A reduction chain, single level or two-level (groups of `l2.group`).
    fn chain(&mut self, ch: &Chain) -> Result<(), String> {
        let (pts, root, n) = (&ch.pts, ch.root, ch.pts.len());
        if root >= n {
            return Err(format!("fabric: chain root {root} of {n} points"));
        }
        let a = self.colour(ch.l1[0])?;
        let b = self.colour(ch.l1[1])?;
        let m = self.colour(ch.cast)?;
        let c2 = match ch.l2 {
            Some(l2) if l2.group > 0 => Some((
                [self.colour(l2.colours[0])?, self.colour(l2.colours[1])?],
                l2.group as usize,
            )),
            _ => None,
        };
        // The level-1 colour a member at hop k sends (`odd`: k odd) / receives on: the low
        // half alternates a, b, ... outward from the root, the high half b, a, ...
        let l1 = |low: bool, odd: bool| if low == odd { a } else { b };
        for (i, &p) in pts.iter().enumerate() {
            if i == root {
                continue;
            }
            let low = i < root;
            let k = if low { root - i } else { i - root };
            let kmax = if low { root } else { n - 1 - root };
            let tow = Self::dir(p, if low { pts[i + 1] } else { pts[i - 1] })?;
            let awy = tow.opposite();
            let send = l1(low, k % 2 == 1);
            let recv = l1(low, (k + 1) % 2 == 1);
            let member = match c2 {
                None => {
                    self.add(p, send, RX_R | RouteWord::tx(tow))?;
                    if k < kmax {
                        self.add(p, recv, RouteWord::rx(awy) | TX_R)?;
                        self.add(p, m, RouteWord::rx(tow) | TX_R | RouteWord::tx(awy))?;
                    } else {
                        self.add(p, m, RouteWord::rx(tow) | TX_R)?;
                    }
                    ChainMember {
                        role: u32::from(k != kmax),
                        send,
                        recv1: (k < kmax).then_some(recv),
                        recv2: None,
                    }
                }
                Some((c2, g)) => {
                    let half = usize::from(!low);
                    let qn = kmax.div_ceil(g);
                    let q = (k - 1) / g;
                    let kk = k - q * g;
                    let gs = if q == qn - 1 { kmax - q * g } else { g };
                    let mw =
                        RouteWord::rx(tow) | TX_R | if k < kmax { RouteWord::tx(awy) } else { 0 };
                    self.add(p, m, mw)?;
                    let s2 = c2[(q + half) % 2];
                    let r2 = c2[(q + 1 + half) % 2];
                    if kk > 1 {
                        self.add(p, send, RX_R | RouteWord::tx(tow))?;
                        if kk < gs {
                            self.add(p, recv, RouteWord::rx(awy) | TX_R)?;
                        }
                        if q + 1 < qn {
                            self.add(p, r2, RouteWord::rx(awy) | RouteWord::tx(tow))?;
                        }
                        ChainMember {
                            role: u32::from(kk != gs),
                            send,
                            recv1: (kk < gs).then_some(recv),
                            recv2: None,
                        }
                    } else {
                        self.add(p, s2, RX_R | RouteWord::tx(tow))?;
                        let has1 = gs > 1;
                        if has1 {
                            self.add(p, recv, RouteWord::rx(awy) | TX_R)?;
                        }
                        if q + 1 < qn {
                            self.add(p, r2, RouteWord::rx(awy) | TX_R)?;
                            ChainMember {
                                role: if has1 { 2 } else { 1 },
                                send: s2,
                                recv1: Some(if has1 { recv } else { r2 }),
                                recv2: has1.then_some(r2),
                            }
                        } else {
                            ChainMember {
                                role: if has1 { 3 } else { 0 },
                                send: s2,
                                recv1: has1.then_some(recv),
                                recv2: None,
                            }
                        }
                    }
                }
            };
            self.bind(p, ch.slot, Endpoint::Member(member))?;
        }
        let r = pts[root];
        let (lo_c, hi_c) = match c2 {
            None => (a, b),
            Some((c2, _)) => (c2[0], c2[1]),
        };
        let mut t = if ch.loopback { TX_R } else { 0 };
        let mut rq = RootQueues::default();
        if root > 0 {
            let d = Self::dir(r, pts[root - 1])?;
            self.add(r, lo_c, RouteWord::rx(d) | TX_R)?;
            rq.lo = Some(lo_c);
            t |= RouteWord::tx(d);
        }
        if root < n - 1 {
            let d = Self::dir(r, pts[root + 1])?;
            self.add(r, hi_c, RouteWord::rx(d) | TX_R)?;
            rq.hi = Some(hi_c);
            t |= RouteWord::tx(d);
        }
        self.add(r, m, RX_R | t)?;
        self.bind(r, ch.slot, Endpoint::Root(rq))
    }

    /// A multicast from its source.
    fn multicast(&mut self, mc: &Multicast) -> Result<(), String> {
        let (pts, src, n) = (&mc.pts, mc.src, mc.pts.len());
        if src >= n {
            return Err(format!("fabric: multicast source {src} of {n} points"));
        }
        let c = self.colour(mc.class)?;
        for (i, &p) in pts.iter().enumerate() {
            if i == src {
                let mut t = 0;
                if i > 0 {
                    t |= RouteWord::tx(Self::dir(p, pts[i - 1])?);
                }
                if i < n - 1 {
                    t |= RouteWord::tx(Self::dir(p, pts[i + 1])?);
                }
                self.add(p, c, RX_R | t)?;
                self.bind(p, mc.slot, Endpoint::Send(c))?;
                continue;
            }
            let (prev, next) = if i < src {
                (pts[i + 1], (i > 0).then(|| pts[i - 1]))
            } else {
                (pts[i - 1], (i < n - 1).then(|| pts[i + 1]))
            };
            let mut word = RouteWord::rx(Self::dir(p, prev)?) | TX_R;
            if let Some(q) = next {
                word |= RouteWord::tx(Self::dir(p, q)?);
            }
            self.add(p, c, word)?;
            self.bind(p, mc.slot, Endpoint::Recv(c))?;
        }
        Ok(())
    }

    /// A point-to-point path.
    fn path(&mut self, pa: &Path) -> Result<(), String> {
        let (pts, n) = (&pa.pts, pa.pts.len());
        if n == 0 {
            return Err("fabric: an empty path".into());
        }
        let c = self.colour(pa.class)?;
        for (i, &p) in pts.iter().enumerate() {
            let r = if i == 0 {
                RX_R
            } else {
                RouteWord::rx(Self::dir(p, pts[i - 1])?)
            };
            let t = if i == n - 1 {
                TX_R
            } else {
                RouteWord::tx(Self::dir(p, pts[i + 1])?)
            };
            self.add(p, c, r | t)?;
        }
        self.bind(pts[0], pa.send, Endpoint::Send(c))?;
        self.bind(pts[n - 1], pa.recv, Endpoint::Recv(c))
    }

    fn turn(&mut self, t: &Turn) -> Result<(), String> {
        let c = self.colour(t.class)?;
        let mut word = match t.from {
            Port::Ramp => RX_R,
            Port::Dir(d) => RouteWord::rx(d),
        };
        for &to in &t.to {
            word |= match to {
                Port::Ramp => TX_R,
                Port::Dir(d) => RouteWord::tx(d),
            };
        }
        self.add(t.at, c, word)
    }
}

const RX_R: u32 = 0x80;
const TX_R: u32 = 0x10;

fn slot_clash(p: Coord, slot: Slot, old: Endpoint, new: Endpoint) -> String {
    format!(
        "fabric: ({}, {}) slot {slot:?} bound twice: {old:?} vs {new:?}",
        p.x, p.y
    )
}

impl Fabric {
    /// Lays every link of `w` through `palette`: per link the routes of each PE on it (a clash,
    /// two words for one colour at one PE, is an error), and the queue
    /// bindings of its endpoints into their slots. Chains are laid single
    /// level) and `line2` (two-level, groups of `Level2::group`) exactly: the same member role
    /// codes, alternating colours by hop parity, the root's halves.
    ///
    /// Endpoints bind order-free ([`Slot::Pass`] binds nothing; a receive and a send in one slot
    /// make a [`Endpoint::Turn`]); [`super::wiring::Note`]s with one key at one PE are OR-ed (the
    /// turn bits). Consecutive points of every link must be adjacent.
    pub fn lay(floor: &Floor, w: &Wiring, palette: &Palette) -> Result<Fabric, String> {
        let mut l = Lay {
            palette,
            routes: Grid::filled(floor.w, floor.h, Vec::new()),
            io: Grid::filled(floor.w, floor.h, PeIo::default()),
        };
        for link in &w.links {
            match link {
                Link::Chain(c) => l.chain(c)?,
                Link::Multicast(m) => l.multicast(m)?,
                Link::Path(p) => l.path(p)?,
                Link::Turn(t) => l.turn(t)?,
            }
        }
        for n in &w.notes {
            l.at(n.at)?;
            *l.io.get_mut(n.at).notes.entry(n.key).or_insert(0) |= n.value;
        }
        let mut routes = l.routes;
        for y in 0..floor.h {
            for x in 0..floor.w {
                routes.get_mut(Coord::new(x, y)).sort_unstable();
            }
        }
        Ok(Fabric {
            routes,
            io: l.io,
            loopback: palette.map.get(&Class::Loopback).copied(),
        })
    }

    pub fn routes(&self, at: Coord) -> &[Route] {
        self.routes.get(at)
    }

    pub fn io(&self, at: Coord) -> &PeIo {
        self.io.get(at)
    }

    pub fn route(&self, at: Coord, c: Colour) -> Option<RouteWord> {
        self.routes(at)
            .iter()
            .find(|r| r.colour == c)
            .map(|r| r.word)
    }

    /// The route table's words: count, then (colour, word) sorted by colour, zero-padded to
    /// `slots` routes; errs past `slots`.
    pub fn route_words(&self, at: Coord, slots: usize) -> Result<Vec<u32>, String> {
        let r = self.routes(at);
        if r.len() > slots {
            return Err(format!(
                "({}, {}) has {} routes > {slots}",
                at.x,
                at.y,
                r.len()
            ));
        }
        let mut w = Vec::with_capacity(1 + 2 * slots);
        w.push(r.len() as u32);
        for rt in r {
            w.push(u32::from(rt.colour.0));
            w.push(rt.word.0);
        }
        w.resize(1 + 2 * slots, 0);
        Ok(w)
    }

    /// `n` colours to park unused input queues on: first colours with no route at `at`, then
    /// colours whose route neither delivers to nor takes from the ramp — never the PE's own send
    /// colour.
    pub fn parks(&self, at: Coord, n: usize) -> Result<Vec<Colour>, String> {
        self.parks_avoiding(at, n, &[])
    }

    /// [`Fabric::parks`], never on `avoid` (a colour the kernel sends on at run time).
    pub fn parks_avoiding(
        &self,
        at: Coord,
        n: usize,
        avoid: &[Colour],
    ) -> Result<Vec<Colour>, String> {
        let open = |c: &Colour| !avoid.contains(c);
        let mut free: Vec<Colour> = (0..COLOURS)
            .map(Colour)
            .filter(|&c| open(&c) && self.route(at, c).is_none())
            .collect();
        free.extend((0..COLOURS).map(Colour).filter(|&c| {
            open(&c)
                && self
                    .route(at, c)
                    .is_some_and(|w| !w.to_ramp() && !w.from_ramp())
        }));
        if free.len() < n {
            return Err(format!(
                "({}, {}): {} free colours < {n}",
                at.x,
                at.y,
                free.len()
            ));
        }
        free.truncate(n);
        Ok(free)
    }

    /// A spare colour for a kernel's temporary route: not in `finals`, preferably not in
    /// `comptime`, not delivered to the ramp.
    pub fn spare(
        &self,
        at: Coord,
        finals: &[Colour],
        comptime: &[Colour],
    ) -> Result<Colour, String> {
        let ramp = |c: Colour| self.route(at, c).is_some_and(|w| w.to_ramp());
        if let Some(c) = (0..COLOURS)
            .map(Colour)
            .find(|c| !finals.contains(c) && !comptime.contains(c) && !ramp(*c))
        {
            return Ok(c);
        }
        (0..COLOURS)
            .map(Colour)
            .find(|c| !finals.contains(c) && !ramp(*c))
            .ok_or_else(|| format!("({}, {}): no spare colour", at.x, at.y))
    }

    /// The hardware invariants, all PEs:
    /// - every colour < [`COLOURS`] (route slots a PE: [`Fabric::check_route_slots`], which
    ///   needs the family's ABI);
    /// - every bound input queue's colour delivers to the PE's ramp, and every bound send
    ///   colour takes from it: no queue waits on a colour the PE only sends on (the 0704 hang;
    ///   [`Fabric::parks`] keeps unbound queues off those too);
    /// - a segment's queues are its own (no colour shared with a reduction's queues on the same
    ///   PE: 0733);
    /// - every unit-row, unit-column and head-row root loops back to its own ramp on the
    ///   palette's loop-back colour.
    ///
    /// Path continuity (adjacent points, ramp to ramp) holds by construction in
    /// [`Fabric::lay`]. Route-restore-after-drain and stack room are kernel-side; see
    /// ARCHITECTURE.md.
    pub fn validate(&self, floor: &Floor) -> Result<(), String> {
        let _ = floor;
        for (p, routes) in self.routes.iter() {
            let here = |m: String| Err(format!("fabric: ({}, {}): {m}", p.x, p.y));
            for r in routes {
                if r.colour.0 >= COLOURS {
                    return here(format!("colour {} >= {COLOURS}", r.colour.0));
                }
            }
            let io = self.io(p);
            for (slot, ep) in &io.slots {
                for c in ep.receives() {
                    if !self.route(p, c).is_some_and(|w| w.to_ramp()) {
                        return here(format!(
                            "{slot:?} receives on colour {} that never reaches the ramp",
                            c.0
                        ));
                    }
                }
                for c in ep.sends() {
                    if !self.route(p, c).is_some_and(|w| w.from_ramp()) {
                        return here(format!(
                            "{slot:?} sends on colour {} that takes nothing from the ramp",
                            c.0
                        ));
                    }
                }
            }
            let colours = |slots: &[Slot]| -> BTreeSet<Colour> {
                slots
                    .iter()
                    .filter_map(|s| io.slots.get(s))
                    .flat_map(|ep| ep.receives().into_iter().chain(ep.sends()))
                    .collect()
            };
            let shared: Vec<u8> = colours(&SEGMENTS)
                .intersection(&colours(&REDUCTIONS))
                .map(|c| c.0)
                .collect();
            if !shared.is_empty() {
                return here(format!(
                    "a segment shares colours {shared:?} with a reduction"
                ));
            }
            if let Some(lb) = self.loopback {
                for s in [Slot::RowReduce, Slot::ColReduce, Slot::HeadRowReduce] {
                    if matches!(io.slots.get(&s), Some(Endpoint::Root(_)))
                        && self.route(p, lb) != Some(RouteWord(RX_R | TX_R))
                    {
                        return here(format!("the {s:?} root has no loop-back"));
                    }
                }
            }
        }
        Ok(())
    }

    /// At most `slots(kind)` routes at every PE that runs a kernel (
    /// bound, checked up front): `slots` is the family's
    /// [`super::emit::rcfg::RcfgAbi::route_slots`]. Idle PEs are not checked here (their
    /// words, if any family reads them, are bounded by [`Fabric::route_words`]).
    pub fn check_route_slots(
        &self,
        floor: &Floor,
        slots: &dyn Fn(RoleKind) -> usize,
    ) -> Result<(), String> {
        for (p, role) in floor.roles.iter() {
            if matches!(role, PeRole::Idle) {
                continue;
            }
            let (n, max) = (self.routes(p).len(), slots(role.kind()));
            if n > max {
                return Err(format!(
                    "fabric: ({}, {}) {} has {n} routes > {max}",
                    p.x,
                    p.y,
                    role.kind().name()
                ));
            }
        }
        Ok(())
    }
}
