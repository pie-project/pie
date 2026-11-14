//! The stream regime's floorplan: units laid one a layer side along a row stream, written
//! directly in the floorplan's elements (the stream regime never had a hand layout).
//!
//! | feature ([`super::Requests`]) | element |
//! |---|---|
//! | `stream`: rows through units, one a layer side | a [`StreamUnit`] a side (a [`UnitShape`]: head columns, plain columns, `py` rows), in path order left to right, the row stream between consecutive units' CX roots ([`Stream::path`]) |
//! | `joint`: attention over the step's rows | a [`KvColumn`] a kv head under each attending block's first unit, linked to every unit of the block |
//! | `lanes`: lane vectors once a step | the lane prologue's unit (west), its CY roots' gather into the lane root and the bus from it to every root that holds a lane vector ([`LaneBus`]) |
//!
//! [`Stream::wiring`] is the composer: each element its links (a unit: its row and column chains,
//! its CX roots' segment where a row region reduces across them; the path; a K / V column's
//! chain and its links to each unit's head root; the lane gather and bus), every class its own,
//! coloured by [`crate::fabric::Palette::allocate`] with no preferred colours.
//!
//! Unit `u` at x `x0`: CX root `j` at `(x0, 1 + j)`, weight PE `(i, j)` at `(x0 + 1 + i, 1 + j)`,
//! CY root `i` at `(x0 + 1 + i, 1 + py)`; row 0 carries the lane bus.

use serde::Serialize;

use super::{ColRole, ColRun, Control, Requests, UnitShape};
use crate::floor::Coord;
use crate::floor::{CtlLine, Floor, HeadBlock, Rect};
use crate::plan::RowRun;
use crate::role::PeRole;
use crate::wiring::{Broadcast, Chain, Class, Path, Slot, Wiring};

type R<T> = Result<T, String>;

/// A unit of the stream: its shape and where it sits.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StreamUnit {
    pub name: String,
    pub shape: UnitShape,
    /// Its CX column's x.
    pub x: u32,
    /// Its CX roots all-reduce a row's sums (a row region's norm).
    pub segment: bool,
    /// On the row stream (the lane prologue's unit is not).
    pub on_path: bool,
}

impl StreamUnit {
    #[must_use]
    pub fn cx(&self, j: u32) -> Coord {
        Coord::new(self.x, 1 + j)
    }

    #[must_use]
    pub fn w(&self, i: u32, j: u32) -> Coord {
        Coord::new(self.x + 1 + i, 1 + j)
    }

    #[must_use]
    pub fn cy(&self, i: u32) -> Coord {
        Coord::new(self.x + 1 + i, 1 + self.rows())
    }

    #[must_use]
    pub fn rows(&self) -> u32 {
        self.shape.rows()
    }

    #[must_use]
    pub fn cols(&self) -> u32 {
        self.shape.cols()
    }

    /// The head columns (a K / V link each).
    #[must_use]
    pub fn heads(&self) -> u32 {
        self.shape
            .cols_of(ColRole::Attn)
            .map_or(0, |r| r.end - r.start)
    }
}

/// A K / V column a kv head of one attending block: its root under the first unit's head
/// column, `pya` attention PEs below it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KvColumn {
    pub block: u32,
    pub head: u32,
    /// The block's units, the first on the path first.
    pub units: Vec<usize>,
    pub pya: u32,
}

/// The lane bus: from the lane root (the lane unit's corner) up its CX column, along row 0, and
/// down each column to its last root that holds a lane vector.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LaneBus {
    pub unit: usize,
    pub to: Vec<Coord>,
}

/// The stream's elements.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Stream {
    pub w: u32,
    pub h: u32,
    pub py: u32,
    pub units: Vec<StreamUnit>,
    /// The units the row stream visits, in order.
    pub path: Vec<usize>,
    pub kv: Vec<KvColumn>,
    pub lanes: Option<LaneBus>,
}

/// What a stream unit asks the floorplan for (the builder's reading of its steps).
#[derive(Clone, Debug)]
pub struct UnitAsk {
    pub name: String,
    pub heads: u32,
    pub cols: u32,
    pub segment: bool,
    pub on_path: bool,
}

impl KvColumn {
    /// The column's root.
    #[must_use]
    pub fn root(&self, s: &Stream) -> Coord {
        let u = &s.units[self.units[0]];
        Coord::new(u.x + 1 + self.head, s.py + 2 + self.head)
    }

    /// Attention PE `k` of the column.
    #[must_use]
    pub fn pe(&self, s: &Stream, k: u32) -> Coord {
        let r = self.root(s);
        Coord::new(r.x, r.y + 1 + k)
    }

    /// The route from unit `k`'s head root down (and west) to the column's root.
    #[must_use]
    pub fn link(&self, s: &Stream, k: usize) -> Vec<Coord> {
        let root = self.root(s);
        let u = &s.units[self.units[k]];
        let top = u.cy(self.head);
        let mut pts: Vec<Coord> = (top.y..=root.y).map(|y| Coord::new(top.x, y)).collect();
        let mut x = top.x;
        while x > root.x {
            x -= 1;
            pts.push(Coord::new(x, root.y));
        }
        pts
    }
}

impl Stream {
    /// The elements `requests` ask for: `asks` the units (the lane prologue's first where
    /// `requests.lanes`), `blocks` the attending blocks (unit indices), `pya` attention PEs a
    /// K / V column, `lane_to` the roots that hold a lane vector.
    pub fn of(
        requests: &Requests,
        py: u32,
        asks: &[UnitAsk],
        blocks: &[Vec<usize>],
        heads: u32,
        pya: u32,
        lane_to: &[Coord],
    ) -> R<Stream> {
        if !requests.stream {
            return Err("stream floorplan: no row stream asked for".into());
        }
        let mut units = Vec::new();
        let mut x = 0;
        for a in asks {
            let mut col_runs = Vec::new();
            if a.heads > 0 {
                col_runs.push(ColRun {
                    n: a.heads,
                    role: ColRole::Attn,
                });
            }
            if a.cols > a.heads {
                col_runs.push(ColRun {
                    n: a.cols - a.heads,
                    role: ColRole::Plain,
                });
            }
            let shape = UnitShape {
                col_runs,
                row_runs: vec![RowRun {
                    rows: (0, py),
                    dims: 0,
                    attention: false,
                }],
                control: Control { cx: 0, cy: py },
                layers: 1,
                kv_rows: 0,
            };
            units.push(StreamUnit {
                name: a.name.clone(),
                shape,
                x,
                segment: a.segment,
                on_path: a.on_path,
            });
            x += a.cols + 1;
        }
        let path: Vec<usize> = (0..units.len()).filter(|&u| units[u].on_path).collect();
        let mut kv = Vec::new();
        if requests.joint {
            for (b, blk) in blocks.iter().enumerate() {
                for h in 0..heads {
                    if blk.iter().any(|&u| units[u].heads() <= h) {
                        return Err(format!(
                            "stream floorplan: block {b}'s units hold fewer than {heads} head columns"
                        ));
                    }
                    kv.push(KvColumn {
                        block: b as u32,
                        head: h,
                        units: blk.clone(),
                        pya,
                    });
                }
            }
        }
        let lanes = requests.lanes.then(|| LaneBus {
            unit: 0,
            to: lane_to.to_vec(),
        });
        let h = py + 2 + if kv.is_empty() { 0 } else { heads + pya };
        Ok(Stream {
            w: x,
            h,
            py,
            units,
            path,
            kv,
            lanes,
        })
    }

    /// A floor of the stream's size (the fabric's laying reads its size only).
    #[must_use]
    pub fn floor(&self) -> Floor {
        Floor {
            w: self.w,
            h: self.h,
            units: Vec::new(),
            relays: Vec::new(),
            head: HeadBlock {
                rect: Rect::new(0, 0, 0, 0),
                ctl: CtlLine::Column(0),
                root: Coord::new(0, 0),
            },
            stores: Vec::new(),
            roles: crate::floor::Grid::filled(self.w, self.h, PeRole::IDLE),
        }
    }

    /// Every link of the stream (see the module doc).
    #[must_use]
    pub fn wiring(&self) -> Wiring {
        let mut w = Wiring::default();
        for (u, unit) in self.units.iter().enumerate() {
            let py = unit.rows();
            for j in 0..py {
                let mut pts = vec![unit.cx(j)];
                pts.extend((0..unit.cols()).map(|i| unit.w(i, j)));
                w.chain(Chain {
                    pts,
                    root: 0,
                    l1: pair(&format!("UR{u}")),
                    l2: None,
                    cast: class(&format!("XR{u}")),
                    loopback: false,
                    slot: ROW_CHAIN,
                });
            }
            for i in 0..unit.cols() {
                let mut pts: Vec<Coord> = (0..py).map(|j| unit.w(i, j)).collect();
                pts.push(unit.cy(i));
                w.chain(Chain {
                    root: pts.len() - 1,
                    pts,
                    l1: pair(&format!("UC{u}")),
                    l2: None,
                    cast: class(&format!("XC{u}")),
                    loopback: false,
                    slot: COL_CHAIN,
                });
            }
            if unit.segment {
                w.chain(Chain {
                    pts: (0..py).map(|j| unit.cx(j)).collect(),
                    root: py as usize - 1,
                    l1: pair(&format!("SG{u}")),
                    l2: None,
                    cast: class(&format!("SM{u}")),
                    loopback: false,
                    slot: SEGMENT,
                });
            }
        }
        for (k, pair_) in self.path.windows(2).enumerate() {
            let (a, b) = (&self.units[pair_[0]], &self.units[pair_[1]]);
            for j in 0..self.py {
                let pts: Vec<Coord> = (a.x..=b.x).map(|x| Coord::new(x, 1 + j)).collect();
                w.path(Path {
                    pts,
                    class: class(&format!("SP{}", k % 2)),
                    send: PATH_OUT,
                    recv: PATH_IN,
                });
            }
        }
        for kc in &self.kv {
            let tag = format!("{}_{}", kc.block, kc.head);
            let mut pts = vec![kc.root(self)];
            pts.extend((0..kc.pya).map(|k| kc.pe(self, k)));
            w.chain(Chain {
                pts,
                root: 0,
                l1: pair(&format!("AC{tag}")),
                l2: None,
                cast: class(&format!("AM{tag}")),
                loopback: false,
                slot: KV_CHAIN,
            });
            for k in 0..kc.units.len() {
                let pts = kc.link(self, k);
                let mut back = pts.clone();
                back.reverse();
                w.path(Path {
                    pts,
                    class: class(&format!("KI{tag}_{k}")),
                    send: KV_SEND,
                    recv: kv_in(k),
                });
                w.path(Path {
                    pts: back,
                    class: class(&format!("KO{tag}_{k}")),
                    send: kv_out(k),
                    recv: KV_RECV,
                });
            }
        }
        if let Some(lb) = &self.lanes {
            let lu = &self.units[lb.unit];
            let root = Coord::new(lu.x, 1 + self.py);
            let mut pts = vec![root];
            pts.extend((0..lu.cols()).map(|i| lu.cy(i)));
            w.chain(Chain {
                pts,
                root: 0,
                l1: pair("LG"),
                l2: None,
                cast: class("LGX"),
                loopback: false,
                slot: LANE_GATHER,
            });
            w.link(crate::wiring::Link::Broadcast(lane_tree(root, &lb.to)));
        }
        w
    }
}

/// The lane bus as a tree: from `root` north to row 0, east along it, down each column to its
/// deepest receiver.
fn lane_tree(root: Coord, to: &[Coord]) -> Broadcast {
    let mut pts: Vec<Coord> = Vec::new();
    let mut parent: Vec<Option<usize>> = Vec::new();
    let at = |pts: &mut Vec<Coord>, parent: &mut Vec<Option<usize>>, c: Coord, p: Option<usize>| {
        if let Some(i) = pts.iter().position(|&q| q == c) {
            return i;
        }
        pts.push(c);
        parent.push(p);
        pts.len() - 1
    };
    let mut prev = at(&mut pts, &mut parent, root, None);
    for y in (0..root.y).rev() {
        prev = at(&mut pts, &mut parent, Coord::new(root.x, y), Some(prev));
    }
    let xmax = to.iter().map(|c| c.x).max().unwrap_or(root.x);
    let mut row0 = vec![prev];
    for x in root.x + 1..=xmax {
        let p = *row0.last().unwrap_or(&prev);
        row0.push(at(&mut pts, &mut parent, Coord::new(x, 0), Some(p)));
    }
    let mut cols: Vec<u32> = to.iter().map(|c| c.x).collect();
    cols.sort_unstable();
    cols.dedup();
    for x in cols {
        let deepest = to
            .iter()
            .filter(|c| c.x == x)
            .map(|c| c.y)
            .max()
            .unwrap_or(0);
        let mut p = row0[(x - root.x) as usize];
        if x == root.x {
            continue;
        }
        for y in 1..=deepest {
            p = at(&mut pts, &mut parent, Coord::new(x, y), Some(p));
        }
    }
    let recv = pts.iter().map(|c| to.contains(c)).collect();
    Broadcast {
        pts,
        src: 0,
        parent,
        recv,
        class: class("LB"),
        slot: LANE_BUS,
    }
}

/// A class named at run time (each unit's, block's and head's own).
fn class(name: &str) -> Class {
    Class::new(Box::leak(name.to_string().into_boxed_str()))
}

fn pair(name: &str) -> [Class; 2] {
    Class::pair(Box::leak(name.to_string().into_boxed_str()))
}

pub const ROW_CHAIN: Slot = Slot::reduce("srow", false);
pub const COL_CHAIN: Slot = Slot::reduce("scol", false);
pub const SEGMENT: Slot = Slot::segment("sseg");
pub const PATH_IN: Slot = Slot::io("spin");
pub const PATH_OUT: Slot = Slot::io("spout");
pub const KV_CHAIN: Slot = Slot::reduce("skv", false);
pub const KV_SEND: Slot = Slot::io("skvs");
pub const KV_RECV: Slot = Slot::io("skvr");
pub const LANE_GATHER: Slot = Slot::reduce("slg", false);
pub const LANE_BUS: Slot = Slot::io("slb");

/// The K / V root's receive slot from the block's `k`-th unit.
#[must_use]
pub fn kv_in(k: usize) -> Slot {
    [
        Slot::io("skvi0"),
        Slot::io("skvi1"),
        Slot::io("skvi2"),
        Slot::io("skvi3"),
    ][k.min(3)]
}

/// The K / V root's send slot to the block's `k`-th unit.
#[must_use]
pub fn kv_out(k: usize) -> Slot {
    [
        Slot::io("skvo0"),
        Slot::io("skvo1"),
        Slot::io("skvo2"),
        Slot::io("skvo3"),
    ][k.min(3)]
}
