//! The program's communication, as typed links over floor coordinates with logical colour
//! classes. A [`Wiring`] says who talks to whom and how (a reduction chain into a root, a
//! multicast from a source, a point-to-point path); the [`super::fabric`] turns it into colours,
//! routes and per-PE queue bindings.

use serde::{Deserialize, Serialize};

use super::floor::{Coord, CtlLine, Floor, Unit};
use super::plan::Plan;

/// A logical colour: what a colour carries. A family's [`super::fabric::Palette`] maps each to a
/// physical colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Class {
    /// Unit rows' reductions into CX: level-1 halves (`XA` / `XB`), level-2 (`X2`), multicast (`MX`).
    RowL1(Half),
    RowL2(Half),
    RowCast,
    /// Unit columns' reductions into CY (`YA` / `YB`, `Y2`, `MY`).
    ColL1(Half),
    ColL2(Half),
    ColCast,
    /// A root's loop-back to its own ramp (`LB`).
    Loopback,
    /// Weight rows' scores segments (`SA` / `SB`, `SM`), GDN's (k, q) exchange (`SG`).
    ScoreL1(Half),
    ScoreCast,
    GdnKq,
    /// The CY row's o-norm segments (`CS` / `CS2`, `CM`).
    OnormL1(Half),
    OnormCast,
    /// The x handoff: along the row (`HA`), vertical (`HV`, by the leaving unit's parity),
    /// into the next unit's CX (`HB`), horizontal (`HX`).
    HandoffOut,
    HandoffVert(Parity),
    HandoffIn,
    HandoffAcross,
    /// The head: relay north (`HN`), column multicast (`HM`), row word / token-loop row (`HR3`).
    HeadRelay,
    HeadColCast,
    HeadRowWord,
    /// The token loop: max-reduce (`WA` / `WB`, `WC2`), winner multicast (`WM`).
    LoopL1(Half),
    LoopL2(Half),
    LoopCast,
    /// The embedding's return: up the head column (`VA`), down (`VS`), the CX row pair split
    /// (`HS`), west into unit 0's CX (`HE`).
    EmbUp,
    EmbDown,
    EmbSplit,
    EmbWest,
    /// Gemma: PLE token word (`TB`), PLE bus by column (`PB`), shared-KV line (`KVS`).
    PleToken,
    PleBus(u8),
    SharedKv,
    /// Gemma 12B: head-group chains (`CH`).
    HeadGroup(Parity),
    /// gpt-oss: its band's extra classes (`HV2`, `V1`, `W2`).
    BandVert,
    BandV1,
    BandW2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Half {
    Lo,
    Hi,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Parity {
    Even,
    Odd,
}

/// Which per-PE queue slot a link's endpoints fill ([`super::fabric::PeIo`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Slot {
    /// The unit-row / unit-column / head-row reduction a PE is a member or root of.
    RowReduce,
    ColReduce,
    HeadRowReduce,
    /// The scores / GDN / o-norm segment.
    Segment,
    Gdn,
    Onorm,
    /// The token loop's max-reduce.
    LoopReduce,
    /// CX's x in / out, embedding in.
    XIn,
    XOut,
    EmbIn,
    /// A relay PE's handoff turn before / after it.
    RelayPre,
    RelayPost,
    /// Head relays and the token loop's turns (`hrole`, `hturn`).
    HeadIn,
    LoopTurn,
    /// The head's relay north arriving at a column's source PE (`HN`), the head column
    /// multicast (`HM`): [`head_input`].
    HeadRelay,
    HeadCast,
    /// No queue binding at that end: the link's routes only (a family whose kernel takes the
    /// colour from a constant, e.g. the token loop's embedding return).
    Pass,
    /// Family slots (PLE store, shared KV, head groups, band).
    Family(&'static str),
}

/// A reduction chain along `pts` into `pts[root]`, then a multicast back out.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Chain {
    pub pts: Vec<Coord>,
    pub root: usize,
    /// Level-1 colours (alternating a hop), and level 2 when the chain streams in groups.
    pub l1: [Class; 2],
    pub l2: Option<Level2>,
    pub cast: Class,
    /// The root multicasts to its own ramp too (`line`'s `lp`).
    pub loopback: bool,
    pub slot: Slot,
}

/// A chain's second level: groups of `group` members stream partial sums on alternating colours
///.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Level2 {
    pub colours: [Class; 2],
    pub group: u32,
}

/// A multicast along `pts` from `pts[src]` both ways.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Multicast {
    pub pts: Vec<Coord>,
    pub src: usize,
    pub class: Class,
    pub slot: Slot,
}

/// A point-to-point path from `pts[0]`'s ramp to the last PE's ramp.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Path {
    pub pts: Vec<Coord>,
    pub class: Class,
    /// The slots the sender and the receiver fill.
    pub send: Slot,
    pub recv: Slot,
}

/// A route a family states directly at one PE (the few that are not chains, multicasts or
/// paths: a root's loop-back, a relay PE's turn).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub at: Coord,
    pub class: Class,
    pub from: Port,
    pub to: Vec<Port>,
}

/// A route endpoint: a fabric direction or the PE's own ramp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Port {
    Ramp,
    Dir(super::floor::Dir),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Link {
    Chain(Chain),
    Multicast(Multicast),
    Path(Path),
    Turn(Turn),
}

/// Per-PE facts a family records for its rcfg words that are not queues (a role code, a flag
/// word, a word count): kept beside the links so the rcfg emitter reads one place.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Note {
    pub at: Coord,
    pub key: &'static str,
    pub value: u32,
}

/// Every link of the program.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Wiring {
    pub links: Vec<Link>,
    pub notes: Vec<Note>,
}

impl Wiring {
    pub fn chain(&mut self, c: Chain) {
        self.links.push(Link::Chain(c));
    }

    pub fn multicast(&mut self, m: Multicast) {
        self.links.push(Link::Multicast(m));
    }

    pub fn path(&mut self, p: Path) {
        self.links.push(Link::Path(p));
    }

    pub fn turn(&mut self, t: Turn) {
        self.links.push(Link::Turn(t));
    }

    pub fn note(&mut self, at: Coord, key: &'static str, value: u32) {
        self.notes.push(Note { at, key, value });
    }
}

/// The generic wiring every family's units share (all but the segments and the head
/// input): [`unit_chains`] (the unit rows' / columns' chains, loop-backs and the x
/// handoffs); for a head with a control column ([`CtlLine::Column`]) also [`head_rows`], and
/// [`token_loop`] when the contract runs the on-wafer loop. Families add their segments
/// ([`segments`]) and head input ([`head_input`] or their own) on top. A family whose head or
/// loop differs (gpt-oss's band, Gemma 12B's head groups) calls the parts it shares instead.
pub fn units(plan: &Plan, floor: &Floor, w: &mut Wiring) -> Result<(), String> {
    unit_chains(plan, floor, w)?;
    if matches!(floor.head.ctl, CtlLine::Column(_)) {
        head_rows(plan, floor, w)?;
        if plan.contract.token_loop {
            token_loop(plan, floor, w)?;
        }
    }
    Ok(())
}

/// Per unit, the row chains into CX (two-level by `reduce.row`; [`Slot::RowReduce`]), the
/// column chains into CY (`reduce.col`; [`Slot::ColReduce`]), every root's loop-back to its own
/// ramp ([`Class::Loopback`]); then the unit-to-unit x handoffs along the visit order
/// ([`handoff`]).
pub fn unit_chains(plan: &Plan, floor: &Floor, w: &mut Wiring) -> Result<(), String> {
    let red = &plan.partition.reduce;
    for u in &floor.units {
        let o = u.origin();
        for r in 0..u.rows() {
            let y = u.row_y(r);
            w.chain(Chain {
                pts: (0..u.rect.w).map(|lx| Coord::new(o.x + lx, y)).collect(),
                root: u.cx as usize,
                l1: [Class::RowL1(Half::Lo), Class::RowL1(Half::Hi)],
                l2: level2(Class::RowL2, red.row),
                cast: Class::RowCast,
                loopback: false,
                slot: Slot::RowReduce,
            });
            w.turn(loopback(Coord::new(u.cx_x(), y)));
        }
        for c in 0..u.cols() {
            let x = u.col_x(c);
            w.chain(Chain {
                pts: (0..u.rect.h).map(|ly| Coord::new(x, o.y + ly)).collect(),
                root: u.cy as usize,
                l1: [Class::ColL1(Half::Lo), Class::ColL1(Half::Hi)],
                l2: level2(Class::ColL2, red.col),
                cast: Class::ColCast,
                loopback: false,
                slot: Slot::ColReduce,
            });
            w.turn(loopback(Coord::new(x, u.cy_y())));
        }
    }
    for i in 1..floor.units.len() {
        handoff(floor, i - 1, i, w)?;
    }
    Ok(())
}

/// A level-2 spec of `group` (`None` for 0: a single-level chain).
/// A chain's level 2 on `class`'s halves, for groups of `group` (0: single level).
pub fn level2(class: fn(Half) -> Class, group: u32) -> Option<Level2> {
    (group > 0).then(|| Level2 {
        colours: [class(Half::Lo), class(Half::Hi)],
        group,
    })
}

/// A root's loop-back to its own ramp.
pub fn loopback(at: Coord) -> Turn {
    Turn {
        at,
        class: Class::Loopback,
        from: Port::Ramp,
        to: vec![Port::Ramp],
    }
}

/// Segment chains over runs of adjacent PEs (scores on weight rows, the CY row's o-norm): `pts`
/// cut into consecutive runs of `width` (the last may be shorter), each a single-level chain
/// rooted at its middle (`len / 2`), level-1 colours `l1`, multicast `cast`, into `slot`. A run
/// of one PE is laid too; a family
/// that skips single-PE runs (the o-norm) passes the runs it wants one call each. Refuses a run
/// whose PEs are not adjacent: a segment spanning a control line.
pub fn segments(
    pts: &[Coord],
    width: u32,
    l1: [Class; 2],
    cast: Class,
    slot: Slot,
    w: &mut Wiring,
) -> Result<(), String> {
    if width == 0 {
        return Err("wiring: a segment width of 0".into());
    }
    for seg in pts.chunks(width as usize) {
        if let Some(i) = (1..seg.len()).find(|&i| seg[i - 1].dir_to(seg[i]).is_none()) {
            return Err(format!(
                "wiring: a segment of {width} spans a control line between ({}, {}) and ({}, {})",
                seg[i - 1].x,
                seg[i - 1].y,
                seg[i].x,
                seg[i].y
            ));
        }
        w.chain(Chain {
            pts: seg.to_vec(),
            root: seg.len() / 2,
            l1,
            l2: None,
            cast,
            loopback: false,
            slot,
        });
    }
    Ok(())
}

/// Python's `range(a, b, s)` over coordinates.
/// `a`, `a + s`, ... short of `b` (either direction).
pub fn steps(a: i64, b: i64, s: i64) -> impl Iterator<Item = u32> {
    let mut v = a;
    std::iter::from_fn(move || {
        if (s > 0 && v < b) || (s < 0 && v > b) {
            let r = v;
            v += s;
            Some(r as u32)
        } else {
            None
        }
    })
}

/// The x handoff between consecutive units `a` -> `b`, one path a
/// weight row `r`. Units in one unit column (equal rect x): along row r from `a`'s CX to the
/// weight PE at column r ([`Class::HandoffOut`]), down / up that column to `b`'s row r
/// ([`Class::HandoffVert`] by `a`'s parity; a gap slot between them only relays), back along the
/// row into `b`'s CX ([`Class::HandoffIn`]); the turning weight PEs relay ([`Slot::RelayPost`]
/// leaving `a`, [`Slot::RelayPre`] arriving at `b`). Units in different columns must share rows:
/// one path along the row ([`Class::HandoffAcross`]). CX ends bind [`Slot::XOut`] /
/// [`Slot::XIn`].
pub fn handoff(floor: &Floor, a: usize, b: usize, w: &mut Wiring) -> Result<(), String> {
    let (ua, ub): (&Unit, &Unit) = match (floor.units.get(a), floor.units.get(b)) {
        (Some(ua), Some(ub)) => (ua, ub),
        _ => return Err(format!("wiring: handoff {a} -> {b} past the units")),
    };
    let (a0, b0) = (ua.origin(), ub.origin());
    let vertical = a0.x == b0.x;
    if !vertical && a0.y != b0.y {
        return Err(format!(
            "units {a} / {b}: a horizontal handoff joins units on the same rows"
        ));
    }
    let parity = if a.is_multiple_of(2) {
        Parity::Even
    } else {
        Parity::Odd
    };
    for r in 0..ua.rows() {
        let (xa, ya) = (ua.cx_x(), ua.row_y(r));
        if vertical {
            let xcc = ua.col_x(r);
            let st: i64 = if xcc > xa { 1 } else { -1 };
            let yb = ub.row_y(r);
            let vs: i64 = if yb > ya { 1 } else { -1 };
            w.path(Path {
                pts: steps(xa as i64, xcc as i64 + st, st)
                    .map(|x| Coord::new(x, ya))
                    .collect(),
                class: Class::HandoffOut,
                send: Slot::XOut,
                recv: Slot::RelayPost,
            });
            w.path(Path {
                pts: steps(ya as i64, yb as i64 + vs, vs)
                    .map(|y| Coord::new(xcc, y))
                    .collect(),
                class: Class::HandoffVert(parity),
                send: Slot::RelayPost,
                recv: Slot::RelayPre,
            });
            w.path(Path {
                pts: steps(xcc as i64, ub.cx_x() as i64 - st, -st)
                    .map(|x| Coord::new(x, yb))
                    .collect(),
                class: Class::HandoffIn,
                send: Slot::RelayPre,
                recv: Slot::XIn,
            });
        } else {
            let xb = ub.cx_x();
            let st: i64 = if xb > xa { 1 } else { -1 };
            w.path(Path {
                pts: steps(xa as i64, xb as i64 + st, st)
                    .map(|x| Coord::new(x, ya))
                    .collect(),
                class: Class::HandoffAcross,
                send: Slot::XOut,
                recv: Slot::XIn,
            });
        }
    }
    Ok(())
}

/// The head's rows: each head row a chain over the head
/// block's columns into the control column (two-level by `reduce.head`; level-1 / level-2
/// colours the unit rows', cast [`Class::HeadRowWord`]; [`Slot::HeadRowReduce`]), the root
/// looping back. For a head with a control column (Qwen, Gemma, Gemma 12B); gpt-oss's band
/// wires its own.
pub fn head_rows(plan: &Plan, floor: &Floor, w: &mut Wiring) -> Result<(), String> {
    let hb = &floor.head;
    let CtlLine::Column(hcol) = hb.ctl else {
        return Err("wiring: head rows want a head control column".into());
    };
    if hcol < hb.rect.x || hcol >= hb.rect.x + hb.rect.w {
        return Err(format!(
            "wiring: head control column {hcol} outside the head"
        ));
    }
    for y in hb.rect.y..hb.rect.y + hb.rect.h {
        w.chain(Chain {
            pts: (0..hb.rect.w)
                .map(|i| Coord::new(hb.rect.x + i, y))
                .collect(),
            root: (hcol - hb.rect.x) as usize,
            l1: [Class::RowL1(Half::Lo), Class::RowL1(Half::Hi)],
            l2: level2(Class::RowL2, plan.partition.reduce.head),
            cast: Class::HeadRowWord,
            loopback: false,
            slot: Slot::HeadRowReduce,
        });
        w.turn(loopback(Coord::new(hcol, y)));
    }
    Ok(())
}

/// The last unit's x into the head: head column `c` takes CX rows
/// `2c` and `2c + 1` of the last unit. Each CX row's path east into its head column
/// ([`Class::HandoffAcross`]; [`Slot::XOut`] at CX, [`Slot::HeadIn`] at the head PE); the odd
/// row's head PE relays north to the even row's ([`Class::HeadRelay`]; [`Slot::HeadIn`] ->
/// [`Slot::HeadRelay`]); the even row's head PE multicasts the column ([`Class::HeadColCast`],
/// [`Slot::HeadCast`]). Notes `head_in` = 1 at the relaying PE, 2 at the column's source
///. Wants head columns = CX rows / 2.
pub fn head_input(plan: &Plan, floor: &Floor, w: &mut Wiring) -> Result<(), String> {
    let _ = plan;
    let hb = &floor.head;
    let ul = floor
        .units
        .last()
        .ok_or_else(|| "wiring: no units".to_string())?;
    if hb.cols() * 2 != ul.rows() {
        return Err("head columns = CX rows / 2".into());
    }
    let xa = ul.cx_x();
    for r in 0..ul.rows() {
        let y = ul.row_y(r);
        let hx = hb.col_x(r / 2);
        w.path(Path {
            pts: (xa..=hx).map(|x| Coord::new(x, y)).collect(),
            class: Class::HandoffAcross,
            send: Slot::XOut,
            recv: Slot::HeadIn,
        });
        if r % 2 == 1 {
            w.note(Coord::new(hx, y), "head_in", 1);
            w.path(Path {
                pts: vec![Coord::new(hx, y), Coord::new(hx, y - 1)],
                class: Class::HeadRelay,
                send: Slot::HeadIn,
                recv: Slot::HeadRelay,
            });
        }
    }
    for c in 0..hb.cols() {
        let hx = hb.col_x(c);
        let src = ul.row_y(2 * c);
        if src < hb.rect.y || src >= hb.rect.y + hb.rect.h {
            return Err(format!("head column {c}: source row {src} off the head"));
        }
        w.note(Coord::new(hx, src), "head_in", 2);
        w.multicast(Multicast {
            pts: (hb.rect.y..hb.rect.y + hb.rect.h)
                .map(|y| Coord::new(hx, y))
                .collect(),
            src: (src - hb.rect.y) as usize,
            class: Class::HeadColCast,
            slot: Slot::HeadCast,
        });
    }
    Ok(())
}

/// The token loop over the head control column and the embedding's return into unit 0
///: the head control PEs max-reduce into the middle row (two-level by
/// `reduce.token_loop`; [`Class::LoopL1`] / [`Class::LoopL2`], cast [`Class::LoopCast`],
/// [`Slot::LoopReduce`]). Then per head column `c`, with `te` / `to` unit 0's CX rows `2c` /
/// `2c + 1` (adjacent rows inside the head's rows): up the column to the head's top row
/// ([`Class::EmbUp`]), down to `te` ([`Class::EmbDown`], when `te` is below the top), across to
/// `to` ([`Class::EmbSplit`]), and west from both rows into unit 0's CX ([`Class::EmbWest`],
/// [`Slot::EmbIn`] there; the head ends bind nothing). Notes `loop_turn` at the head PEs: bit 1
/// on the top row, 2 at `te`, 4 at `to`.
pub fn token_loop(plan: &Plan, floor: &Floor, w: &mut Wiring) -> Result<(), String> {
    let hb = &floor.head;
    let Some(hcol) = hb.ctl_x() else {
        return Err("wiring: the token loop wants a head control column".into());
    };
    let (y0, hr) = (hb.rect.y, hb.rect.h);
    w.chain(Chain {
        pts: (y0..y0 + hr).map(|y| Coord::new(hcol, y)).collect(),
        root: (hr / 2) as usize,
        l1: [Class::LoopL1(Half::Lo), Class::LoopL1(Half::Hi)],
        l2: level2(Class::LoopL2, plan.partition.reduce.token_loop),
        cast: Class::LoopCast,
        loopback: false,
        slot: Slot::LoopReduce,
    });
    let u0 = floor
        .units
        .first()
        .ok_or_else(|| "wiring: no units".to_string())?;
    for c in 0..hb.cols() {
        let x = hb.col_x(c);
        let (te, to) = (u0.row_y(2 * c), u0.row_y(2 * c + 1));
        if to != te + 1 || te < y0 || to >= y0 + hr {
            return Err(format!("head column {c}: CX rows {te} / {to}"));
        }
        w.path(Path {
            pts: (y0..y0 + hr).rev().map(|y| Coord::new(x, y)).collect(),
            class: Class::EmbUp,
            send: Slot::Pass,
            recv: Slot::Pass,
        });
        if te > y0 {
            w.path(Path {
                pts: (y0..=te).map(|y| Coord::new(x, y)).collect(),
                class: Class::EmbDown,
                send: Slot::Pass,
                recv: Slot::Pass,
            });
        }
        w.path(Path {
            pts: vec![Coord::new(x, te), Coord::new(x, to)],
            class: Class::EmbSplit,
            send: Slot::Pass,
            recv: Slot::Pass,
        });
        for y in [te, to] {
            w.path(Path {
                pts: (u0.cx_x()..=x).rev().map(|xx| Coord::new(xx, y)).collect(),
                class: Class::EmbWest,
                send: Slot::Pass,
                recv: Slot::EmbIn,
            });
        }
        w.note(Coord::new(x, y0), "loop_turn", 1);
        w.note(Coord::new(x, te), "loop_turn", 2);
        w.note(Coord::new(x, to), "loop_turn", 4);
    }
    Ok(())
}
