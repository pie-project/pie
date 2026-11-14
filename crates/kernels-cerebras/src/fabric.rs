//! The fabric: physical colours, every PE's route table and queue bindings, and the hardware
//! invariants checked on them. [`Palette::allocate`] colours the wiring's classes (an
//! interference colouring of their footprints, each class its block's preferred colour where it
//! is free: [`Fabric::lay_coloured`] takes the preferences outright where they lay with no
//! clash); [`Fabric::lay`] turns a [`Wiring`] into routes through that [`Palette`];
//! [`Fabric::validate`] refuses a program the CS-3 would hang or miscompute on.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use super::floor::{Coord, Dir, Floor, Grid};
use super::role::Kind;
use super::wiring::{
    Broadcast, Chain, Class, LOOPBACK, Link, Multicast, NoteKey, Path, Port, Slot, SlotKind, Tree,
    Turn, Wiring,
};

/// A physical colour (0 .. [`COLOURS`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Colour(pub u8);

/// Colours a PE's router has.
pub const COLOURS: u8 = 21;

/// A route's switch word: receive from one port, transmit to a set (`RX_R` / `TX_R` bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct RouteWord(pub u32);

impl RouteWord {
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

/// The logical -> physical colour map ([`Palette::allocate`]: every class the blocks declare or
/// the links lay, coloured apart from every class it meets).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Palette {
    pub map: BTreeMap<Class, Colour>,
}

impl Palette {
    /// `decls` in order, later ones replacing.
    pub fn of(decls: impl IntoIterator<Item = (Class, u8)>) -> Palette {
        Palette {
            map: decls.into_iter().map(|(k, c)| (k, Colour(c))).collect(),
        }
    }

    /// The colours of `w`'s classes on `floor`: an interference colouring of their footprints.
    /// Two classes meet where both route at one PE, unless their words there are one and the same
    /// pass-through (what [`Fabric::lay`] merges); classes that never meet share colours freely.
    /// `prefs` are the blocks' preferred colours ([`super::block::Block::colours`], a later
    /// declaration of a class replacing an earlier one): a class keeps its preferred colour where
    /// no class it meets kept it first (classes in their order), so a program whose preferences
    /// never clash is coloured as its blocks declare. Every other class (a preference taken, or
    /// none: a class the links lay that no block colours) takes the lowest colour no class it
    /// meets holds, the most constrained first. A class that meets every colour is refused by
    /// name, with the classes it meets.
    pub fn allocate(
        prefs: impl IntoIterator<Item = (Class, u8)>,
        floor: &Floor,
        w: &Wiring,
    ) -> Result<Palette, String> {
        let prefs: BTreeMap<Class, u8> = prefs.into_iter().collect();
        let mut classes: BTreeSet<Class> = prefs.keys().copied().collect();
        for l in &w.links {
            classes.extend(l.classes());
        }
        let ids: Vec<Class> = classes.into_iter().collect();
        if ids.len() > usize::from(u8::MAX) {
            return Err(format!("colouring: {} classes", ids.len()));
        }
        // each class a colour of its own, to read its footprint off the laid routes
        let own = Palette {
            map: ids
                .iter()
                .enumerate()
                .map(|(i, &c)| (c, Colour(i as u8)))
                .collect(),
        };
        let routes = Lay::routes(floor, w, &own, u8::MAX)?;
        let n = ids.len();
        let mut meets = vec![BTreeSet::<usize>::new(); n];
        let pass = |r: &Route| !r.word.to_ramp() && r.word.0 & 0x80 == 0;
        for (_, rs) in routes.iter() {
            for (i, a) in rs.iter().enumerate() {
                for b in &rs[i + 1..] {
                    if a.word == b.word && pass(a) {
                        continue;
                    }
                    let (x, y) = (usize::from(a.colour.0), usize::from(b.colour.0));
                    meets[x].insert(y);
                    meets[y].insert(x);
                }
            }
        }
        let mut colour: Vec<Option<u8>> = vec![None; n];
        let held = |colour: &[Option<u8>], i: usize| -> BTreeSet<u8> {
            meets[i].iter().filter_map(|&j| colour[j]).collect()
        };
        for (i, c) in ids.iter().enumerate() {
            if let Some(&p) = prefs.get(c)
                && p < COLOURS
                && !held(&colour, i).contains(&p)
            {
                colour[i] = Some(p);
            }
        }
        while let Some(i) = (0..n)
            .filter(|&i| colour[i].is_none())
            .max_by_key(|&i| (held(&colour, i).len(), meets[i].len(), std::cmp::Reverse(i)))
        {
            let h = held(&colour, i);
            let c = (0..COLOURS).find(|c| !h.contains(c)).ok_or_else(|| {
                format!(
                    "colouring: {:?} meets classes holding all {COLOURS} colours: {:?}",
                    ids[i],
                    meets[i].iter().map(|&j| ids[j]).collect::<Vec<_>>()
                )
            })?;
            colour[i] = Some(c);
        }
        Ok(Palette {
            map: ids
                .into_iter()
                .zip(colour)
                .map(|(k, c)| (k, Colour(c.unwrap_or(0))))
                .collect(),
        })
    }

    pub fn get(&self, c: Class) -> Result<Colour, String> {
        self.map
            .get(&c)
            .copied()
            .ok_or_else(|| format!("palette: no colour for {c:?}"))
    }
}

/// A chain member's role code in the kernels' chain protocol.
pub mod member_role {
    /// The chain's (or its last group's) far end: sends, receives nothing.
    pub const LEAF: u32 = 0;
    /// Receives from the next member out, adds, sends on.
    pub const MID: u32 = 1;
    /// A level-2 group's head with members behind it and a group further out.
    pub const GROUP_HEAD: u32 = 2;
    /// The last group's head with members behind it.
    pub const LAST_GROUP_HEAD: u32 = 3;
}

/// A chain member's queues: its role code ([`member_role`]), its send colour, its level-1 /
/// level-2 receive colours.
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
    /// The blocks' [`super::wiring::Note`]s at this PE.
    pub notes: BTreeMap<NoteKey, u32>,
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
    /// The palette's [`LOOPBACK`] colour, for the roots' loop-back check.
    pub loopback: Option<Colour>,
}

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
    /// The colours the palette may use ([`COLOURS`]; every `u8` when a footprint is read).
    limit: u8,
    routes: Grid<Vec<Route>>,
    io: Grid<PeIo>,
}

impl Lay<'_> {
    /// Every link of `w` laid through `palette` (colours under `limit`): the routes only.
    fn routes(
        floor: &Floor,
        w: &Wiring,
        palette: &Palette,
        limit: u8,
    ) -> Result<Grid<Vec<Route>>, String> {
        let mut l = Lay {
            palette,
            limit,
            routes: Grid::filled(floor.w, floor.h, Vec::new()),
            io: Grid::filled(floor.w, floor.h, PeIo::default()),
        };
        l.links(w)?;
        Ok(l.routes)
    }

    fn links(&mut self, w: &Wiring) -> Result<(), String> {
        for link in &w.links {
            match link {
                Link::Chain(c) => self.chain(c)?,
                Link::Tree(t) => self.tree(t)?,
                Link::Multicast(m) => self.multicast(m)?,
                Link::Path(p) => self.path(p)?,
                Link::Turn(t) => self.turn(t)?,
                Link::Broadcast(b) => self.broadcast(b)?,
            }
        }
        Ok(())
    }

    fn colour(&self, c: Class) -> Result<Colour, String> {
        let col = self.palette.get(c)?;
        if col.0 >= self.limit {
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
        if slot.kind == SlotKind::Pass {
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
                        role: if k != kmax {
                            member_role::MID
                        } else {
                            member_role::LEAF
                        },
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
                            role: if kk != gs {
                                member_role::MID
                            } else {
                                member_role::LEAF
                            },
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
                                role: if has1 {
                                    member_role::GROUP_HEAD
                                } else {
                                    member_role::MID
                                },
                                send: s2,
                                recv1: Some(if has1 { recv } else { r2 }),
                                recv2: has1.then_some(r2),
                            }
                        } else {
                            ChainMember {
                                role: if has1 {
                                    member_role::LAST_GROUP_HEAD
                                } else {
                                    member_role::LEAF
                                },
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

    /// A reduction tree ([`Tree`]): each child's link coloured apart from every link it shares a
    /// PE with, the members' queues by their children (nearest first), the root's by side, the
    /// multicast back out as a chain's.
    fn tree(&mut self, t: &Tree) -> Result<(), String> {
        let (pts, root, n) = (&t.pts, t.root, t.pts.len());
        if root >= n || t.parent.len() != n || t.parent[root].is_some() {
            return Err(format!("fabric: tree root {root} of {n} points"));
        }
        let pool = t
            .colours
            .iter()
            .map(|&c| self.colour(c))
            .collect::<Result<Vec<_>, _>>()?;
        let m = self.colour(t.cast)?;
        // the links (child, parent), each toward the root
        let mut links = Vec::with_capacity(n);
        for (i, p) in t.parent.iter().enumerate() {
            if i == root {
                continue;
            }
            let p = p.ok_or_else(|| format!("fabric: tree point {i} has no parent"))?;
            let toward = if i < root {
                p > i && p <= root
            } else {
                p < i && p >= root
            };
            if !toward {
                return Err(format!(
                    "fabric: tree point {i}'s parent {p} not toward the root {root}"
                ));
            }
            links.push((i, p));
        }
        let span = |(i, p): (usize, usize)| (i.min(p), i.max(p));
        let mut order: Vec<usize> = (0..links.len()).collect();
        order.sort_by_key(|&e| span(links[e]));
        let mut col = vec![usize::MAX; links.len()];
        // left ends first: a link overlaps an earlier one iff that one reaches its left end
        let mut open: Vec<(usize, usize)> = Vec::new(); // (right end, colour) of the links so far
        for &e in &order {
            let (a, b) = span(links[e]);
            open.retain(|&(r, _)| r >= a);
            let c = (0..pool.len())
                .find(|c| open.iter().all(|&(_, oc)| oc != *c))
                .ok_or_else(|| {
                    format!(
                        "fabric: tree over {} PEs at point {a} wants more than {} colours",
                        open.len() + 1,
                        pool.len()
                    )
                })?;
            col[e] = c;
            open.push((b, c));
        }
        let mut kids: Vec<Vec<(usize, Colour)>> = vec![Vec::new(); n];
        let mut send = vec![None; n];
        for (e, &(i, p)) in links.iter().enumerate() {
            let c = pool[col[e]];
            let step = |j: usize| if p < i { j - 1 } else { j + 1 };
            let back = |j: usize| if p < i { j + 1 } else { j - 1 };
            let mut j = i;
            loop {
                let rx = if j == i {
                    RX_R
                } else {
                    RouteWord::rx(Self::dir(pts[j], pts[back(j)])?)
                };
                let tx = if j == p {
                    TX_R
                } else {
                    RouteWord::tx(Self::dir(pts[j], pts[step(j)])?)
                };
                self.add(pts[j], c, rx | tx)?;
                if j == p {
                    break;
                }
                j = step(j);
            }
            kids[p].push((i, c));
            send[i] = Some(c);
        }
        for (i, &p) in pts.iter().enumerate() {
            let mut k = std::mem::take(&mut kids[i]);
            k.sort_by_key(|&(j, _)| j.abs_diff(i));
            if i == root {
                let side = |lo: bool| {
                    let s: Vec<Colour> = k
                        .iter()
                        .filter(|&&(j, _)| (j < i) == lo)
                        .map(|&(_, c)| c)
                        .collect();
                    match s.len() {
                        0 => Ok(None),
                        1 => Ok(Some(s[0])),
                        _ => Err(format!(
                            "fabric: tree root {root} with {} children a side",
                            s.len()
                        )),
                    }
                };
                let rq = RootQueues {
                    lo: side(true)?,
                    hi: side(false)?,
                };
                let mut tw = 0;
                if i > 0 {
                    tw |= RouteWord::tx(Self::dir(p, pts[i - 1])?);
                }
                if i < n - 1 {
                    tw |= RouteWord::tx(Self::dir(p, pts[i + 1])?);
                }
                self.add(p, m, RX_R | tw)?;
                self.bind(p, t.slot, Endpoint::Root(rq))?;
                continue;
            }
            let role = match k.len() {
                0 => member_role::LEAF,
                1 => member_role::MID,
                2 => member_role::GROUP_HEAD,
                c => return Err(format!("fabric: tree point {i} with {c} children")),
            };
            let low = i < root;
            let tow = Self::dir(p, if low { pts[i + 1] } else { pts[i - 1] })?;
            let far = if low { i > 0 } else { i < n - 1 };
            let mw = RouteWord::rx(tow)
                | TX_R
                | if far {
                    RouteWord::tx(tow.opposite())
                } else {
                    0
                };
            self.add(p, m, mw)?;
            let member = ChainMember {
                role,
                send: send[i].ok_or("fabric: a tree member sends nothing")?,
                recv1: k.first().map(|&(_, c)| c),
                recv2: k.get(1).map(|&(_, c)| c),
            };
            self.bind(p, t.slot, Endpoint::Member(member))?;
        }
        Ok(())
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

    /// A multicast tree: each point receives from its parent (the source from its ramp) and
    /// passes on to its children, to its own ramp where it is a receiver.
    fn broadcast(&mut self, b: &Broadcast) -> Result<(), String> {
        let n = b.pts.len();
        if b.src >= n || b.parent.len() != n || b.recv.len() != n || b.parent[b.src].is_some() {
            return Err(format!("fabric: broadcast source {} of {n} points", b.src));
        }
        let c = self.colour(b.class)?;
        for (i, &p) in b.pts.iter().enumerate() {
            let mut word = match b.parent[i] {
                None if i == b.src => RX_R,
                None => return Err(format!("fabric: broadcast point {i} has no parent")),
                Some(q) => RouteWord::rx(Self::dir(p, b.pts[q])?),
            };
            for (j, par) in b.parent.iter().enumerate() {
                if *par == Some(i) {
                    word |= RouteWord::tx(Self::dir(p, b.pts[j])?);
                }
            }
            if i == b.src {
                self.add(p, c, word)?;
                self.bind(p, b.slot, Endpoint::Send(c))?;
            } else {
                if b.recv[i] {
                    word |= TX_R;
                    self.bind(p, b.slot, Endpoint::Recv(c))?;
                }
                self.add(p, c, word)?;
            }
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
    /// Endpoints bind order-free ([`super::wiring::PASS`] binds nothing; a receive and a send in one slot
    /// make a [`Endpoint::Turn`]); [`super::wiring::Note`]s with one key at one PE are OR-ed (the
    /// turn bits). Consecutive points of every link must be adjacent.
    pub fn lay(floor: &Floor, w: &Wiring, palette: &Palette) -> Result<Fabric, String> {
        let mut l = Lay {
            palette,
            limit: COLOURS,
            routes: Grid::filled(floor.w, floor.h, Vec::new()),
            io: Grid::filled(floor.w, floor.h, PeIo::default()),
        };
        l.links(w)?;
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
            loopback: palette.map.get(&LOOPBACK).copied(),
        })
    }

    /// The palette of `w` and the fabric laid through it, validated ([`Fabric::validate`]): the
    /// blocks' preferred colours (`prefs`, later replacing earlier) where they colour every class
    /// the links lay and lay and validate with no clash (the common case: the wiring is laid
    /// once), else [`Palette::allocate`]'s colouring.
    pub fn lay_coloured(
        floor: &Floor,
        w: &Wiring,
        prefs: Vec<(Class, u8)>,
    ) -> Result<(Palette, Fabric), String> {
        let declared = Palette::of(prefs.iter().copied());
        let covers = w
            .links
            .iter()
            .flat_map(Link::classes)
            .all(|c| declared.map.contains_key(&c));
        if covers
            && let Ok(f) = Fabric::lay(floor, w, &declared)
            && f.validate(floor).is_ok()
        {
            return Ok((declared, f));
        }
        let palette = Palette::allocate(prefs, floor, w)?;
        let f = Fabric::lay(floor, w, &palette)?;
        f.validate(floor)?;
        Ok((palette, f))
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
    /// colour, nor one of `avoid` (a colour the kernel sends on at run time).
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
    ///   needs the layout's ABI);
    /// - every bound input queue's colour delivers to the PE's ramp, and every bound send
    ///   colour takes from it: no queue waits on a colour the PE only sends on (the 0704 hang;
    ///   [`Fabric::parks_avoiding`] keeps unbound queues off those too);
    /// - a segment's queues are its own (no colour shared with a reduction's queues on the same
    ///   PE: 0733);
    /// - every root of a looping reduction ([`SlotKind::Reduce`] `loopback`) loops back to its
    ///   own ramp on the palette's loop-back colour.
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
            let colours = |f: fn(SlotKind) -> bool| -> BTreeSet<Colour> {
                io.slots
                    .iter()
                    .filter(|(s, _)| f(s.kind))
                    .flat_map(|(_, ep)| ep.receives().into_iter().chain(ep.sends()))
                    .collect()
            };
            let shared: Vec<u8> = colours(|k| k == SlotKind::Segment)
                .intersection(&colours(|k| matches!(k, SlotKind::Reduce { .. })))
                .map(|c| c.0)
                .collect();
            if !shared.is_empty() {
                return here(format!(
                    "a segment shares colours {shared:?} with a reduction"
                ));
            }
            if let Some(lb) = self.loopback {
                for (s, ep) in &io.slots {
                    if s.kind == (SlotKind::Reduce { loopback: true })
                        && matches!(ep, Endpoint::Root(_))
                        && self.route(p, lb) != Some(RouteWord(RX_R | TX_R))
                    {
                        return here(format!("the {} root has no loop-back", s.name));
                    }
                }
            }
        }
        Ok(())
    }

    /// At most `slots(kind)` routes at every PE that runs a kernel: `slots` is the kind's
    /// kernel's route table ([`super::kernel::KernelBuild::route_slots`]; `None`: the program
    /// builds no kernel for the kind, an error). Idle PEs are not checked here (their words, if
    /// a kernel reads them, are bounded by [`Fabric::route_words`]).
    pub fn check_route_slots(
        &self,
        floor: &Floor,
        slots: &dyn Fn(Kind) -> Option<usize>,
    ) -> Result<(), String> {
        for (p, role) in floor.roles.iter() {
            if role.is_idle() {
                continue;
            }
            let kind = role.kind;
            let max = slots(kind)
                .ok_or_else(|| format!("fabric: ({}, {}) {}: no kernel", p.x, p.y, kind.name))?;
            let n = self.routes(p).len();
            if n > max {
                return Err(format!(
                    "fabric: ({}, {}) {} has {n} routes > {max}",
                    p.x, p.y, kind.name
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::floor::{CtlLine, HeadBlock, Rect};
    use crate::role::PeRole;
    use crate::wiring::PASS;

    fn floor(w: u32, h: u32) -> Floor {
        Floor {
            w,
            h,
            units: Vec::new(),
            relays: Vec::new(),
            head: HeadBlock {
                rect: Rect::new(0, 0, 0, 0),
                ctl: CtlLine::Column(0),
                root: Coord::new(0, 0),
            },
            stores: Vec::new(),
            roles: Grid::filled(w, h, PeRole::IDLE),
        }
    }

    fn path(class: Class, pts: &[(u32, u32)]) -> Link {
        Link::Path(Path {
            pts: pts.iter().map(|&(x, y)| Coord::new(x, y)).collect(),
            class,
            send: PASS,
            recv: PASS,
        })
    }

    /// Classes that meet at a PE take different colours, the first in class order its preferred
    /// one; classes that never meet share colours; a class no block colours takes the lowest one
    /// free; the result lays with no clash.
    #[test]
    fn the_colouring_keeps_preferences_that_do_not_clash() {
        let (a, b, c, d) = (
            Class::new("A"),
            Class::new("B"),
            Class::new("C"),
            Class::new("D"),
        );
        let w = Wiring {
            links: vec![
                // A along row 0, B down column 1: they meet at (1, 0)
                path(a, &[(0, 0), (1, 0), (2, 0), (3, 0)]),
                path(b, &[(1, 0), (1, 1), (1, 2)]),
                // C and D far from both
                path(c, &[(3, 2), (3, 3)]),
                path(d, &[(0, 3), (1, 3)]),
            ],
            notes: Vec::new(),
        };
        let fl = floor(4, 4);
        let p = Palette::allocate([(a, 5), (b, 5), (d, 5)], &fl, &w).unwrap();
        assert_eq!(p.get(a).unwrap(), Colour(5));
        assert_ne!(p.get(b).unwrap(), Colour(5));
        assert_eq!(p.get(b).unwrap(), Colour(0));
        assert_eq!(p.get(c).unwrap(), Colour(0));
        assert_eq!(p.get(d).unwrap(), Colour(5));
        Fabric::lay(&fl, &w, &p).unwrap();
        // the declared palette clashes; laid coloured, the colouring's is taken
        let prefs = vec![(a, 5), (b, 5), (c, 0), (d, 5)];
        assert!(Fabric::lay(&fl, &w, &Palette::of(prefs.iter().copied())).is_err());
        let (q, _) = Fabric::lay_coloured(&fl, &w, prefs).unwrap();
        assert_eq!(q.get(b).unwrap(), Colour(0));
        // preferences that lay as they are, are the palette
        let ok = vec![(a, 5), (b, 6), (c, 0), (d, 5)];
        let (q, _) = Fabric::lay_coloured(&fl, &w, ok.clone()).unwrap();
        assert_eq!(q.map, Palette::of(ok).map);
    }

    /// A class that meets more classes than there are colours is refused by name.
    #[test]
    fn a_class_past_every_colour_is_refused() {
        const NAMES: [&str; 22] = [
            "K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10", "K11", "K12", "K13",
            "K14", "K15", "K16", "K17", "K18", "K19", "K20", "K21",
        ];
        let links = NAMES
            .iter()
            .map(|&n| path(Class::new(n), &[(0, 0), (1, 0)]))
            .collect();
        let w = Wiring {
            links,
            notes: Vec::new(),
        };
        let e = Palette::allocate([], &floor(2, 1), &w).unwrap_err();
        assert!(
            e.contains("colouring:") && e.contains("all 21 colours"),
            "{e}"
        );
    }
}
