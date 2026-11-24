//! A program as a placement problem. Its **blocks** are rigid rectangles of PEs (a unit, the
//! head, a store column, a relay slot, a K / V column) carrying the links inside them; its
//! **nets** are the links between blocks. A [`Solution`] puts every block at an origin; nets
//! between blocks moved apart are routed afresh. A program's own placement
//! ([`Solution::pinned`]) reproduces it byte for byte.

pub mod anneal;
pub mod bundle;
pub mod cost;
pub mod decode;
pub mod flow;
pub mod pack;
pub mod route;
pub mod stream;

use serde::Serialize;

use crate::compile::floor::{Coord, Rect};
use crate::compile::wiring::{Link, Note, Wiring};

type R<T> = Result<T, String>;

/// What a block is (what its PEs run, as the regime laid it).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum BlockKind {
    /// A decode unit (its index in the visit order).
    Unit(u32),
    /// A gap slot whose PEs only relay.
    Relay,
    /// The head (and its control line).
    Head,
    /// A side input's storage column (its index on the floor).
    Store(usize),
    /// A stream unit (its index in the stream).
    StreamUnit(usize),
    /// A stream's K / V column (its index in the stream).
    KvColumn(usize),
    /// A co-resident reading (a whole program; its index in the load).
    Reading(usize),
}

/// A rigid rectangle of PEs and the links inside it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Block {
    pub kind: BlockKind,
    pub rect: Rect,
    /// The links that lie wholly inside it (indices into the wiring's links).
    pub links: Vec<usize>,
}

/// What a net carries between its blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum NetKind {
    /// The x from one unit to the next.
    Handoff,
    /// The last unit's x into the head.
    HeadInput,
    /// The embedding's return from the head into unit 0.
    TokenReturn,
    /// A shared cache's line through its sharers.
    KvLine,
    /// A side input's token line or bus.
    SideBus,
    /// Any other link that leaves its block.
    Other,
}

/// A link between blocks (or through free PEs).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Net {
    pub kind: NetKind,
    /// Its links (indices into the wiring's links): one, or a class's turns (a bus stated a PE
    /// at a time).
    pub links: Vec<usize>,
    /// The blocks its points fall in, in order of first touch.
    pub blocks: Vec<usize>,
    /// Some of its points lie outside every block.
    pub free: bool,
}

/// A program as blocks and nets on a fabric of `w` x `h` PEs.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Problem {
    pub w: u32,
    pub h: u32,
    pub blocks: Vec<Block>,
    pub nets: Vec<Net>,
    /// The links' notes: the block each one's PE falls in (`None`: a free PE, moved with the
    /// net through it).
    pub notes: Vec<Option<usize>>,
}

/// Every block's origin and the fabric they are placed on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Solution {
    pub w: u32,
    pub h: u32,
    pub at: Vec<Coord>,
}

impl Solution {
    /// The program's own placement.
    #[must_use]
    pub fn pinned(p: &Problem) -> Solution {
        Solution {
            w: p.w,
            h: p.h,
            at: p
                .blocks
                .iter()
                .map(|b| Coord::new(b.rect.x, b.rect.y))
                .collect(),
        }
    }

    /// Block `i`'s move from where the problem has it.
    #[must_use]
    pub fn delta(&self, p: &Problem, i: usize) -> (i64, i64) {
        let r = p.blocks[i].rect;
        (
            i64::from(self.at[i].x) - i64::from(r.x),
            i64::from(self.at[i].y) - i64::from(r.y),
        )
    }

    /// Block `i`'s rectangle where it is placed.
    #[must_use]
    pub fn rect(&self, p: &Problem, i: usize) -> Rect {
        let r = p.blocks[i].rect;
        Rect::new(self.at[i].x, self.at[i].y, r.w, r.h)
    }
}

/// Every PE a link names (its points; a turn's PE).
#[must_use]
pub fn points(l: &Link) -> Vec<Coord> {
    match l {
        Link::Chain(c) => c.pts.clone(),
        Link::Tree(t) => t.pts.clone(),
        Link::Multicast(m) => m.pts.clone(),
        Link::Path(p) => p.pts.clone(),
        Link::Broadcast(b) => b.pts.clone(),
        Link::Turn(t) => vec![t.at],
    }
}

/// `c` moved by `d`, on the fabric.
pub fn shift(c: Coord, d: (i64, i64)) -> R<Coord> {
    let (x, y) = (i64::from(c.x) + d.0, i64::from(c.y) + d.1);
    match (u32::try_from(x), u32::try_from(y)) {
        (Ok(x), Ok(y)) => Ok(Coord::new(x, y)),
        _ => Err(format!("place: ({}, {}) moved off the fabric", c.x, c.y)),
    }
}

/// `l` moved by `d`.
pub fn translate(l: &Link, d: (i64, i64)) -> R<Link> {
    let mv = |pts: &[Coord]| pts.iter().map(|&c| shift(c, d)).collect::<R<Vec<_>>>();
    let mut l = l.clone();
    match &mut l {
        Link::Chain(c) => c.pts = mv(&c.pts)?,
        Link::Tree(t) => t.pts = mv(&t.pts)?,
        Link::Multicast(m) => m.pts = mv(&m.pts)?,
        Link::Path(p) => p.pts = mv(&p.pts)?,
        Link::Broadcast(b) => b.pts = mv(&b.pts)?,
        Link::Turn(t) => t.at = shift(t.at, d)?,
    }
    Ok(l)
}

impl Problem {
    /// The blocks `rects` (with their kinds) and `w`'s links sorted into each block's own and
    /// the nets between them; a net's kind by its link's class (`kind`).
    pub fn of(
        w: u32,
        h: u32,
        rects: Vec<(BlockKind, Rect)>,
        wiring: &Wiring,
        kind: &dyn Fn(&Link) -> NetKind,
    ) -> R<Problem> {
        for (i, (_, a)) in rects.iter().enumerate() {
            for (_, b) in &rects[i + 1..] {
                if a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h {
                    return Err(format!("place: blocks {a:?} and {b:?} overlap"));
                }
            }
        }
        // the block at each PE
        let mut at = vec![u32::MAX; (w * h) as usize];
        for (i, (_, r)) in rects.iter().enumerate() {
            for y in r.y..r.y + r.h {
                for x in r.x..r.x + r.w {
                    at[(y * w + x) as usize] = i as u32;
                }
            }
        }
        let block_at = |c: Coord| -> Option<usize> {
            (c.x < w && c.y < h)
                .then(|| at[(c.y * w + c.x) as usize])
                .filter(|&b| b != u32::MAX)
                .map(|b| b as usize)
        };
        let mut blocks: Vec<Block> = rects
            .into_iter()
            .map(|(kind, rect)| Block {
                kind,
                rect,
                links: Vec::new(),
            })
            .collect();
        let mut nets: Vec<Net> = Vec::new();
        // where links fall: (blocks touched, any point free)
        let touch = |pts: &[Coord]| {
            let mut touched: Vec<usize> = Vec::new();
            let mut free = false;
            for &c in pts {
                match block_at(c) {
                    Some(b) if !touched.contains(&b) => touched.push(b),
                    Some(_) => {}
                    None => free = true,
                }
            }
            (touched, free)
        };
        let mut sort = |links: Vec<usize>, pts: &[Coord], kind: NetKind| {
            let (touched, free) = touch(pts);
            match (touched.as_slice(), free) {
                ([b], false) => blocks[*b].links.extend(links),
                _ => nets.push(Net {
                    kind,
                    links,
                    blocks: touched,
                    free,
                }),
            }
        };
        // a class's turns at adjacent PEs are one link (a bus stated a PE at a time)
        let mut turns: std::collections::BTreeMap<crate::compile::wiring::Class, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (li, l) in wiring.links.iter().enumerate() {
            match l {
                Link::Turn(t) => turns.entry(t.class).or_default().push(li),
                _ => sort(vec![li], &points(l), kind(l)),
            }
        }
        for (_, ls) in turns {
            let at: Vec<Coord> = ls
                .iter()
                .flat_map(|&li| points(&wiring.links[li]))
                .collect();
            let mut by: std::collections::HashMap<Coord, Vec<usize>> =
                std::collections::HashMap::new();
            for (j, c) in at.iter().enumerate() {
                by.entry(*c).or_default().push(j);
            }
            let mut seen = vec![false; ls.len()];
            for i in 0..ls.len() {
                if seen[i] {
                    continue;
                }
                seen[i] = true;
                let mut comp = vec![i];
                let mut k = 0;
                while k < comp.len() {
                    let c = at[comp[k]];
                    let near = [
                        Some(c),
                        c.x.checked_sub(1).map(|x| Coord::new(x, c.y)),
                        Some(Coord::new(c.x + 1, c.y)),
                        c.y.checked_sub(1).map(|y| Coord::new(c.x, y)),
                        Some(Coord::new(c.x, c.y + 1)),
                    ];
                    for n in near.into_iter().flatten() {
                        for &j in by.get(&n).map_or(&[][..], Vec::as_slice) {
                            if !seen[j] {
                                seen[j] = true;
                                comp.push(j);
                            }
                        }
                    }
                    k += 1;
                }
                comp.sort_unstable();
                let pts: Vec<Coord> = comp.iter().map(|&j| at[j]).collect();
                let links: Vec<usize> = comp.iter().map(|&j| ls[j]).collect();
                let kd = kind(&wiring.links[links[0]]);
                sort(links, &pts, kd);
            }
        }
        let notes = wiring.notes.iter().map(|n| block_at(n.at)).collect();
        Ok(Problem {
            w,
            h,
            blocks,
            nets,
            notes,
        })
    }

    /// `wiring` with every block's links and every net moved with its blocks. A net between
    /// blocks moved apart needs the router and is refused here.
    pub fn apply(&self, wiring: &Wiring, s: &Solution) -> R<Wiring> {
        if s.at.len() != self.blocks.len() {
            return Err(format!(
                "place: {} origins for {} blocks",
                s.at.len(),
                self.blocks.len()
            ));
        }
        let mut links: Vec<Option<Link>> = vec![None; wiring.links.len()];
        for (i, b) in self.blocks.iter().enumerate() {
            let d = s.delta(self, i);
            for &li in &b.links {
                links[li] = Some(translate(&wiring.links[li], d)?);
            }
        }
        // the move of each free PE a net crosses (for its notes)
        let mut free_moves: Vec<(Coord, (i64, i64))> = Vec::new();
        for n in &self.nets {
            let d = self.net_delta(n, s)?;
            for &li in &n.links {
                let l = &wiring.links[li];
                if n.free {
                    free_moves.extend(points(l).into_iter().map(|c| (c, d)));
                }
                links[li] = Some(translate(l, d)?);
            }
        }
        let notes = wiring
            .notes
            .iter()
            .zip(&self.notes)
            .map(|(n, b)| {
                let d = match b {
                    Some(b) => s.delta(self, *b),
                    None => free_moves
                        .iter()
                        .find(|(c, _)| *c == n.at)
                        .map(|(_, d)| *d)
                        .ok_or_else(|| format!("place: a note at {:?} on no block or net", n.at))?,
                };
                Ok(Note {
                    at: shift(n.at, d)?,
                    ..*n
                })
            })
            .collect::<R<Vec<_>>>()?;
        Ok(Wiring {
            links: links
                .into_iter()
                .map(|l| l.expect("every link sorted"))
                .collect(),
            notes,
        })
    }

    /// The one move every block of `n` takes (a net on no block: the one every block takes).
    fn net_delta(&self, n: &Net, s: &Solution) -> R<(i64, i64)> {
        let all: Vec<usize> = if n.blocks.is_empty() {
            (0..self.blocks.len()).collect()
        } else {
            n.blocks.clone()
        };
        let mut ds = all.iter().map(|&b| s.delta(self, b));
        let d = ds.next().unwrap_or((0, 0));
        if ds.any(|e| e != d) {
            return Err(format!(
                "place: a {:?} net between blocks moved apart wants the router",
                n.kind
            ));
        }
        Ok(d)
    }
}
