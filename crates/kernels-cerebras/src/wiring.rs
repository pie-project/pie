//! The program's communication, as typed links over floor coordinates with logical colour
//! classes. A [`Wiring`] says who talks to whom and how (a reduction chain into a root, a
//! multicast from a source, a point-to-point path); the [`super::fabric`] turns it into colours,
//! routes and per-PE queue bindings.
//!
//! Classes, slots and note keys are open: the core declares only what the fabric's invariants
//! read ([`LOOPBACK`], [`PASS`]); every block declares its own as consts in its module
//! (`block::units::ROW_L1`, `block::head::HEAD_IN`, ...).

use serde::{Serialize, Serializer};

use super::floor::Coord;

/// A logical colour: what a colour carries. Declared by the block that lays it; the palette
/// ([`super::fabric::Palette`]) maps it to a physical colour. `name` is the kernels' and tools'
/// name for it (`XA`, `MX`, ...); `index` tells apart the members of an indexed family of
/// classes (a level-2 pair, a bus by column).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Class {
    pub name: &'static str,
    pub index: u8,
}

impl Class {
    pub const fn new(name: &'static str) -> Class {
        Class { name, index: 0 }
    }

    pub const fn at(name: &'static str, index: u8) -> Class {
        Class { name, index }
    }

    /// The `[lo, hi]` pair of an indexed class (a chain's level-2 halves).
    pub const fn pair(name: &'static str) -> [Class; 2] {
        [Class::at(name, 0), Class::at(name, 1)]
    }
}

impl Serialize for Class {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.index == 0 {
            s.serialize_str(self.name)
        } else {
            s.serialize_str(&format!("{}{}", self.name, self.index))
        }
    }
}

/// A root's loop-back to its own ramp (`LB`): the fabric checks every looping root has it.
pub const LOOPBACK: Class = Class::new("LB");

/// What a slot's queues are, for the fabric's invariants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum SlotKind {
    /// A reduction chain; `loopback`: its root multicasts back to its own ramp ([`LOOPBACK`]).
    Reduce { loopback: bool },
    /// A segment (scores, o-norm, a multicast within a row): its queues never share a colour
    /// with a reduction's on the same PE.
    Segment,
    /// Any other endpoint (x in / out, relays, head input, a block's own).
    Io,
    /// No queue at that end: the link's routes only.
    Pass,
}

/// Which per-PE queue slot a link's endpoints fill ([`super::fabric::PeIo`]); the kernel's rcfg
/// writer reads its slots by these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Slot {
    pub name: &'static str,
    pub kind: SlotKind,
}

impl Slot {
    pub const fn new(name: &'static str, kind: SlotKind) -> Slot {
        Slot { name, kind }
    }

    pub const fn reduce(name: &'static str, loopback: bool) -> Slot {
        Slot::new(name, SlotKind::Reduce { loopback })
    }

    pub const fn segment(name: &'static str) -> Slot {
        Slot::new(name, SlotKind::Segment)
    }

    pub const fn io(name: &'static str) -> Slot {
        Slot::new(name, SlotKind::Io)
    }
}

/// No queue binding at that end (a kernel that takes the colour from a constant).
pub const PASS: Slot = Slot::new("pass", SlotKind::Pass);

/// A per-PE fact a block records for a kernel's rcfg that is not a queue (a role code, a turn
/// bit). Notes with one key at one PE are OR-ed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct NoteKey(pub &'static str);

/// A reduction chain along `pts` into `pts[root]`, then a multicast back out.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Chain {
    pub pts: Vec<Coord>,
    pub root: usize,
    /// Level-1 colours (alternating a hop), and level 2 when the chain streams in groups.
    pub l1: [Class; 2],
    pub l2: Option<Level2>,
    pub cast: Class,
    /// The root multicasts to its own ramp too.
    pub loopback: bool,
    pub slot: Slot,
}

/// A chain's second level: groups of `group` members stream partial sums on alternating colours.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
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

/// A route stated directly at one PE (the few that are not chains, multicasts or paths: a
/// root's loop-back, a relay PE's turn).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Turn {
    pub at: Coord,
    pub class: Class,
    pub from: Port,
    pub to: Vec<Port>,
}

/// A route endpoint: a fabric direction or the PE's own ramp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Note {
    pub at: Coord,
    pub key: NoteKey,
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

    pub fn note(&mut self, at: Coord, key: NoteKey, value: u32) {
        self.notes.push(Note { at, key, value });
    }
}

/// A chain's level 2 on `colours`, for groups of `group` (0: single level).
pub fn level2(colours: [Class; 2], group: u32) -> Option<Level2> {
    (group > 0).then_some(Level2 { colours, group })
}

/// A root's loop-back to its own ramp.
pub fn loopback(at: Coord) -> Turn {
    Turn {
        at,
        class: LOOPBACK,
        from: Port::Ramp,
        to: vec![Port::Ramp],
    }
}

/// Segment chains over runs of adjacent PEs: `pts` cut into consecutive runs of `width` (the
/// last may be shorter), each a single-level chain rooted at its middle (`len / 2`), level-1
/// colours `l1`, multicast `cast`, into `slot`; runs of one PE only when `singles`. Refuses a
/// run whose PEs are not adjacent: a segment spanning a control line.
pub fn segments(
    pts: &[Coord],
    width: u32,
    singles: bool,
    (l1, cast, slot): ([Class; 2], Class, Slot),
    w: &mut Wiring,
) -> Result<(), String> {
    if width == 0 {
        return Err("wiring: a segment width of 0".into());
    }
    for seg in pts.chunks(width as usize) {
        if seg.len() == 1 && !singles {
            continue;
        }
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

/// `a`, `a + s`, ... short of `b` (either direction): Python's `range(a, b, s)`.
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
