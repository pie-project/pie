//! The stream regime's floorplan: units laid one a layer side along a row stream, filling bands
//! left to right ([`StreamUnit`], [`Stream::path`]), a [`KvColumn`] a kv head under each attending
//! block's first unit, and the lane prologue's bus ([`LaneBus`]). [`Stream::wiring`] lays every
//! element's links, each class its own, coloured with no preferred colours.
//!
//! Unit `u` at `(x0, y0)`: CX root `j` at `(x0, y0 + 1 + j)`, weight PE `(i, j)` at
//! `(x0 + 1 + c(i), y0 + 1 + j)`, CY root `i` at `(x0 + 1 + c(i), y0 + 1 + py)`; row 0 carries the
//! lane bus. The row stream into a new band is a staircase ([`Stream::wrap`]).

use serde::Serialize;

use super::{ColRole, ColRun, Control, Requests, UnitShape};
use crate::compile::floor::Coord;
use crate::compile::floor::{CtlLine, Floor, HeadBlock, Rect};
use crate::compile::plan::RowRun;
use crate::compile::role::PeRole;
use crate::compile::wiring::{Broadcast, Chain, Class, Path, Slot, Wiring};

type R<T> = Result<T, String>;

/// A unit of the stream: its shape and where it sits.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StreamUnit {
    pub name: String,
    pub shape: UnitShape,
    /// Its CX column's x.
    pub x: u32,
    /// Its band's top row (the lane bus row of band 0).
    pub y: u32,
    /// Its CX roots all-reduce a row's sums (a row region's norm).
    pub segment: bool,
    /// Its CY roots `first .. first + roots` all-reduce a row's sums (a column region's norm
    /// across those roots: a norm over all heads at the head roots).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub col_segment: Option<(u32, u32)>,
    /// On the row stream (the lane prologue's unit is not).
    pub on_path: bool,
    /// Helper PEs below each rest root (`cy(i)` for `i >= heads`): [`StreamUnit::helper`].
    pub helpers: u32,
    /// The head columns' pitch (1: the heads first; [`StreamUnit::col`]).
    pub pitch: u32,
}

impl StreamUnit {
    #[must_use]
    pub fn cx(&self, j: u32) -> Coord {
        Coord::new(self.x, self.y + 1 + j)
    }

    /// Column `i`'s offset in the unit: head column `h` at `h pitch`, the rest roots in order in
    /// the columns left between the heads and after them.
    #[must_use]
    pub fn col(&self, i: u32) -> u32 {
        let (p, h) = (self.pitch.max(1), self.heads());
        if p == 1 || h == 0 {
            return i;
        }
        if i < h {
            return i * p;
        }
        // the `k`-th rest column: `p - 1` between each head, then the rest past the last
        let k = i - h;
        let between = h * (p - 1);
        if k < between {
            (k / (p - 1)) * p + 1 + k % (p - 1)
        } else {
            h * p + (k - between)
        }
    }

    /// The unit's columns in their order on the fabric, west to east.
    #[must_use]
    pub fn west_to_east(&self) -> Vec<u32> {
        let mut v: Vec<u32> = (0..self.cols()).collect();
        v.sort_by_key(|&i| self.col(i));
        v
    }

    #[must_use]
    pub fn w(&self, i: u32, j: u32) -> Coord {
        Coord::new(self.x + 1 + self.col(i), self.y + 1 + j)
    }

    #[must_use]
    pub fn cy(&self, i: u32) -> Coord {
        Coord::new(self.x + 1 + self.col(i), self.y + 1 + self.rows())
    }

    #[must_use]
    pub fn rows(&self) -> u32 {
        self.shape.rows()
    }

    /// Helper `k` of column root `i`: straight below it.
    #[must_use]
    pub fn helper(&self, i: u32, k: u32) -> Coord {
        let c = self.cy(i);
        Coord::new(c.x, c.y + 1 + k)
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
/// columns, `pya` attention PEs below it snaked over `sub` columns. At `sub > 1` the first unit's
/// head columns are `sub` apart ([`StreamUnit::col`]) so each column sits straight below its head
/// root and no link crosses a K / V column.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KvColumn {
    pub block: u32,
    pub head: u32,
    /// The block's units, the first on the path first.
    pub units: Vec<usize>,
    pub pya: u32,
    pub sub: u32,
    /// Its root where a placer put it ([`KvColumn::at`]); `None`: under its first unit's head
    /// column, as the module doc lays it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at: Option<Coord>,
}

/// The lane bus: from the lane root up its CX column, along row 0, and down each column to its
/// last root that holds a lane vector.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LaneBus {
    pub unit: usize,
    pub to: Vec<Coord>,
}

/// The fabric a stream may take at most (the program, without the memcpy halo).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Bounds {
    pub w: u32,
    pub h: u32,
}

impl Bounds {
    /// A WSE-3 system's fabric less the memcpy layouts' halo.
    #[must_use]
    pub fn system() -> Bounds {
        let (w, h) = crate::compile::toolchain::Arch::Wse3.system_fabric();
        let ((hw, hh), _) = crate::compile::toolchain::memcpy_fabric(0, 0);
        Bounds {
            w: w - hw,
            h: h - hh,
        }
    }

    /// These bounds at most `w` wide (`None`: as they are).
    #[must_use]
    pub fn narrowed(self, w: Option<u32>) -> Bounds {
        Bounds {
            w: w.map_or(self.w, |w| w.min(self.w)),
            h: self.h,
        }
    }
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
    /// The rows between the units' column roots and the K / V roots where K / V columns take
    /// several columns a head: the later units' turns (one a head) and the helpers above them.
    pub turns: u32,
}

/// What a stream unit asks the floorplan for (the builder's reading of its steps).
#[derive(Clone, Debug)]
pub struct UnitAsk {
    pub name: String,
    pub heads: u32,
    pub cols: u32,
    pub segment: bool,
    /// Its CY roots `first .. first + roots` all-reduce ([`StreamUnit::col_segment`]).
    pub col_segment: Option<(u32, u32)>,
    pub on_path: bool,
    /// It reads lane vectors on its row / column roots.
    pub lanes: (bool, bool),
    /// Helper PEs below each of its rest roots (a column region split over them).
    pub helpers: u32,
}

impl KvColumn {
    /// This column with its root at `root` (a placer's): its PEs below it as before, its links
    /// to the block's units routed.
    #[must_use]
    pub fn at(&self, root: Coord) -> KvColumn {
        KvColumn {
            at: Some(root),
            ..self.clone()
        }
    }

    /// The column's root.
    #[must_use]
    pub fn root(&self, s: &Stream) -> Coord {
        if let Some(at) = self.at {
            return at;
        }
        let u = &s.units[self.units[0]];
        if self.sub > 1 {
            return Coord::new(u.cy(self.head).x, u.y + s.py + 2 + s.turns);
        }
        Coord::new(u.x + 1 + self.head, u.y + s.py + 2 + self.head)
    }

    /// The rows its PEs take below its root.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.pya.div_ceil(self.sub.max(1))
    }

    /// Attention PE `k` of the column: down its first sub-column, up the next, ...
    #[must_use]
    pub fn pe(&self, s: &Stream, k: u32) -> Coord {
        let r = self.root(s);
        let n = self.rows();
        let (c, i) = (k / n, k % n);
        let dy = if c % 2 == 0 { i } else { n - 1 - i };
        Coord::new(r.x + c, r.y + 1 + dy)
    }

    /// The route from unit `k`'s head root to the column's root
    /// ([`crate::compile::place::bundle::kv_link`]).
    #[must_use]
    pub fn link(&self, s: &Stream, k: usize) -> Vec<Coord> {
        // (the bundle router's: a head's link turns on its own row, so a unit's links do not
        // cross; a placed root's too)
        let root = self.root(s);
        let top = s.units[self.units[k]].cy(self.head);
        let turn = s.units[self.units[0]].y + s.py + 2 + self.head;
        crate::compile::place::bundle::kv_link(top, root, turn)
    }
}

impl Stream {
    /// The elements `requests` ask for: `asks` the units (the lane prologue's first), `blocks`
    /// the attending blocks, `pya` attention PEs a K / V column over `sub` columns a head, within
    /// `bounds` (bands of units where one row is too wide).
    #[allow(clippy::too_many_arguments)]
    pub fn of(
        requests: &Requests,
        py: u32,
        asks: &[UnitAsk],
        blocks: &[Vec<usize>],
        heads: u32,
        pya: u32,
        sub: u32,
        bounds: Bounds,
    ) -> R<Stream> {
        let sub = sub.max(1);
        if !requests.stream {
            return Err("stream floorplan: no row stream asked for".into());
        }
        // (K / V columns over several columns a head: their roots on one row below the turns
        // and the helpers)
        let turns = if sub > 1 {
            asks.iter().map(|a| a.helpers).fold(heads, u32::max)
        } else {
            heads
        };
        let band = py
            + 2
            + if requests.joint && !blocks.is_empty() {
                turns + pya.div_ceil(sub) + u32::from(sub > 1)
            } else {
                0
            };
        // the last unit of the block each unit opens (a block's units share a band)
        let block_end: Vec<usize> = (0..asks.len())
            .map(|u| {
                blocks
                    .iter()
                    .find(|b| b.first() == Some(&u))
                    .and_then(|b| b.iter().max().copied())
                    .unwrap_or(u)
            })
            .collect();
        let mut units = Vec::new();
        let (mut x, mut y, mut w) = (0u32, 0u32, 0u32);
        for (u, a) in asks.iter().enumerate() {
            let span: u32 = asks[u..=block_end[u]].iter().map(|b| b.cols + 1).sum();
            if x > 0 && x + span > bounds.w {
                x = 0;
                y += band;
            }
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
            // a block's first unit holds its K / V columns: its heads at their pitch
            let first = blocks.iter().any(|b| b.first() == Some(&u));
            units.push(StreamUnit {
                name: a.name.clone(),
                shape,
                x,
                y,
                segment: a.segment,
                col_segment: a.col_segment,
                on_path: a.on_path,
                helpers: a.helpers,
                pitch: if first && requests.joint && a.heads > 0 {
                    sub
                } else {
                    1
                },
            });
            x += a.cols + 1;
            w = w.max(x);
        }
        // the fabric ends where its last band's blocks do: a band holding no K / V columns keeps
        // none of the K / V rows (idle rows cost the chip time)
        let kv_rows = band - (py + 2);
        let holds_kv = |u: usize| requests.joint && blocks.iter().any(|b| b.first() == Some(&u));
        let h = (0..units.len())
            .map(|u| {
                units[u].y
                    + py
                    + 2
                    + if holds_kv(u) {
                        kv_rows
                    } else {
                        asks[u].helpers.min(kv_rows)
                    }
            })
            .max()
            .unwrap_or(0)
            .max(y + py + 2);
        // (at `sub > 1` the helpers stay above the K / V roots' row)
        let room = if sub > 1 { py + 2 + turns } else { band };
        if let Some(a) = asks.iter().find(|a| py + 2 + a.helpers > room) {
            return Err(format!(
                "stream floorplan: {}'s {} helpers a root outgrow its band's {room} rows",
                a.name, a.helpers
            ));
        }
        let path: Vec<usize> = (0..units.len()).filter(|&u| units[u].on_path).collect();
        for p in path.windows(2) {
            let (a, b) = (&units[p[0]], &units[p[1]]);
            if a.y != b.y && a.x < b.x + py {
                return Err(format!(
                    "stream floorplan: {} ends its band at x {}: the row stream's turn to {} needs {py} columns",
                    a.name, a.x, b.name
                ));
            }
        }
        if w > bounds.w || h > bounds.h {
            return Err(format!(
                "stream floorplan: {w} x {h} PEs ({} bands of {band} rows) over the {} x {} fabric",
                y / band + 1,
                bounds.w,
                bounds.h
            ));
        }
        let mut kv = Vec::new();
        if requests.joint {
            for (b, blk) in blocks.iter().enumerate() {
                // (a block's heads its first unit's: blocks of other attentions apart)
                let heads = blk.first().map_or(heads, |&u| units[u].heads().min(heads));
                for h in 0..heads {
                    if blk.iter().any(|&u| units[u].heads() <= h) {
                        return Err(format!(
                            "stream floorplan: block {b}'s units hold fewer than {heads} head columns"
                        ));
                    }
                    if blk.first().is_some_and(|&u| units[u].cols() < heads * sub) {
                        return Err(format!(
                            "stream floorplan: block {b}'s K / V columns over {sub} columns a \
                             head ({}) outgrow its first unit's {} columns",
                            heads * sub,
                            blk.first().map_or(0, |&u| units[u].cols())
                        ));
                    }
                    kv.push(KvColumn {
                        block: b as u32,
                        head: h,
                        units: blk.clone(),
                        pya,
                        sub,
                        at: None,
                    });
                }
            }
        }
        // the lane receivers: every root of a unit that reads lanes on that side
        let mut lane_to: Vec<Coord> = Vec::new();
        for (a, u) in asks.iter().zip(&units) {
            if a.lanes.0 {
                lane_to.extend((0..py).map(|j| u.cx(j)));
            }
            if a.lanes.1 {
                lane_to.extend((0..u.cols()).map(|i| u.cy(i)));
            }
        }
        let lanes = requests.lanes.then_some(LaneBus {
            unit: 0,
            to: lane_to,
        });
        Ok(Stream {
            w,
            h,
            py,
            units,
            path,
            kv,
            lanes,
            turns,
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
            roles: crate::compile::floor::Grid::filled(self.w, self.h, PeRole::IDLE),
        }
    }

    /// Row `j`'s wire from unit `a`'s CX root to unit `b`'s, `b` the first of a later band: west
    /// along row `j` to column `b.x + 1 + j`, south, then west to `b`'s CX root. The staircase
    /// keeps rows' wires apart.
    #[must_use]
    pub fn wrap(a: &StreamUnit, b: &StreamUnit, j: u32) -> Vec<Coord> {
        // (the bundle router's staircase down and west)
        use crate::compile::place::bundle::{Turn, z};
        z(a.cx(j), b.cx(j), Turn::Column(b.x + 1 + j))
    }

    /// Every link of the stream (see the module doc).
    #[must_use]
    pub fn wiring(&self) -> Wiring {
        let mut w = Wiring::default();
        for (u, unit) in self.units.iter().enumerate() {
            let py = unit.rows();
            let order = unit.west_to_east();
            for j in 0..py {
                let mut pts = vec![unit.cx(j)];
                pts.extend(order.iter().map(|&i| unit.w(i, j)));
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
            if unit.helpers > 0 {
                for i in unit.heads()..unit.cols() {
                    let mut pts = vec![unit.cy(i)];
                    pts.extend((0..unit.helpers).map(|k| unit.helper(i, k)));
                    w.chain(Chain {
                        pts,
                        root: 0,
                        l1: pair(&format!("HL{u}")),
                        l2: None,
                        cast: class(&format!("HM{u}")),
                        loopback: false,
                        slot: HELP_CHAIN,
                    });
                }
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
            // the column side's: its CY roots in a row, rooted at the last (one half, as the
            // CX roots')
            if let Some((first, roots)) = unit.col_segment {
                w.chain(Chain {
                    pts: (first..first + roots).map(|i| unit.cy(i)).collect(),
                    root: roots as usize - 1,
                    l1: pair(&format!("YG{u}")),
                    l2: None,
                    cast: class(&format!("YM{u}")),
                    loopback: false,
                    slot: SEGMENT,
                });
            }
        }
        for (k, pair_) in self.path.windows(2).enumerate() {
            let (a, b) = (&self.units[pair_[0]], &self.units[pair_[1]]);
            let wrap = a.y != b.y;
            for j in 0..self.py {
                // (the bundle router's: a straight row on `b`'s rows, else a staircase)
                let pts = crate::compile::place::bundle::gap(a, b, self.py, j);
                let name = if wrap {
                    format!("SW{k}")
                } else {
                    format!("SP{}", k % 2)
                };
                w.path(Path {
                    pts,
                    class: class(&name),
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
            let root = Coord::new(lu.x, lu.y + 1 + self.py);
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
            w.link(crate::compile::wiring::Link::Broadcast(lane_tree(
                root, &lb.to,
            )));
        }
        w
    }
}

/// The lane bus as a tree: from `root` north to row 0, east along it, down each column to its
/// deepest receiver; in `root`'s own column, south from `root` to the receivers below it.
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
        // the root's own column: its receivers above it are on the trunk north, those below
        // (a later band's units under the lane unit) on a branch south from the root
        let (mut p, from) = if x == root.x {
            (0, root.y + 1)
        } else {
            (row0[(x - root.x) as usize], 1)
        };
        for y in from..=deepest {
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
pub const HELP_CHAIN: Slot = Slot::reduce("shelp", false);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(name: &str, cols: u32, heads: u32) -> UnitAsk {
        UnitAsk {
            name: name.into(),
            heads,
            cols,
            segment: false,
            col_segment: None,
            on_path: true,
            lanes: (false, false),
            helpers: 0,
        }
    }

    /// The fabric ends where its blocks do: a last band of units that hold no K / V columns (a
    /// decode program's exit) keeps none of the band's K / V rows.
    #[test]
    fn a_band_without_k_v_columns_keeps_no_k_v_rows() {
        let mut asks: Vec<UnitAsk> = (0..4).map(|u| ask(&format!("u{u}"), 9, 2)).collect();
        asks[3].heads = 0;
        let blocks = vec![vec![0], vec![1], vec![2]];
        let requests = Requests {
            stream: true,
            joint: true,
            ..Requests::default()
        };
        let (py, pya) = (4, 7);
        let s = Stream::of(
            &requests,
            py,
            &asks,
            &blocks,
            2,
            pya,
            1,
            Bounds { w: 32, h: 60 },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let exit = &s.units[3];
        assert!(exit.y > 0, "the exit opens a second band");
        assert_eq!(s.h, exit.y + py + 2);
        // (the first band keeps its K / V rows; its wiring lays inside the cut fabric)
        assert!(s.kv.iter().all(|k| k.units[0] < 3));
        let mut floor = s.floor();
        let mut wiring = s.wiring();
        crate::compile::place::route::fill_layout(&mut floor, &mut wiring)
            .unwrap_or_else(|e| panic!("{e}"));
    }

    /// The lane bus reaches every root that reads a lane vector, in every band, including a later
    /// band's first unit below the lane root in its column (flux2-mini over three bands).
    #[test]
    fn the_lane_bus_reaches_every_band() {
        let mut asks: Vec<UnitAsk> = (0..7).map(|u| ask(&format!("u{u}"), 9, 2)).collect();
        asks[0].heads = 0;
        asks[0].on_path = false;
        for a in &mut asks[1..] {
            a.lanes = (true, true);
        }
        let blocks = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let requests = Requests {
            stream: true,
            joint: true,
            lanes: true,
            ..Requests::default()
        };
        let s = Stream::of(
            &requests,
            4,
            &asks,
            &blocks,
            2,
            3,
            1,
            Bounds { w: 32, h: 60 },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(s.units.iter().filter(|u| u.x == 0).count() >= 3);
        let lb = s.lanes.as_ref().expect("a lane bus");
        let wiring = s.wiring();
        let bus = wiring
            .links
            .iter()
            .find_map(|l| match l {
                crate::compile::wiring::Link::Broadcast(b) => Some(b),
                _ => None,
            })
            .expect("the lane bus");
        for c in &lb.to {
            let i = bus.pts.iter().position(|p| p == c);
            assert!(i.is_some_and(|i| bus.recv[i]), "the lane bus misses {c:?}");
        }
        for (i, p) in bus.parent.iter().enumerate() {
            if let Some(q) = *p {
                let (a, b) = (bus.pts[i], bus.pts[q]);
                assert_eq!(a.x.abs_diff(b.x) + a.y.abs_diff(b.y), 1);
            }
        }
        assert_eq!(bus.recv.iter().filter(|&&r| r).count(), lb.to.len());
    }

    /// Units too wide for one row of the fabric fill bands, a block never split over two; the
    /// row stream's wires into a new band pass no PE twice and no two rows' wires meet.
    #[test]
    fn units_fill_bands_and_the_wraps_do_not_cross() {
        let asks: Vec<UnitAsk> = (0..7).map(|u| ask(&format!("u{u}"), 9, 2)).collect();
        let blocks = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let requests = Requests {
            stream: true,
            joint: true,
            ..Requests::default()
        };
        let s = Stream::of(
            &requests,
            4,
            &asks,
            &blocks,
            2,
            3,
            1,
            Bounds { w: 32, h: 60 },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let band = 4 + 2 + 2 + 3;
        let ys: Vec<u32> = s.units.iter().map(|u| u.y).collect();
        assert_eq!(ys, [0, 0, 0, band, band, 2 * band, 2 * band]);
        assert_eq!((s.w, s.h), (30, 3 * band));
        let (a, b) = (&s.units[2], &s.units[3]);
        let wires: Vec<Vec<Coord>> = (0..4).map(|j| Stream::wrap(a, b, j)).collect();
        for (j, w) in wires.iter().enumerate() {
            assert_eq!(w[0], a.cx(j as u32));
            assert_eq!(*w.last().unwrap(), b.cx(j as u32));
            for p in w.windows(2) {
                assert_eq!(p[0].x.abs_diff(p[1].x) + p[0].y.abs_diff(p[1].y), 1);
            }
            for v in &wires[j + 1..] {
                assert!(w.iter().all(|c| !v.contains(c)));
            }
        }
        // too narrow for a block: refused by name
        assert!(
            Stream::of(
                &requests,
                4,
                &asks,
                &blocks,
                2,
                3,
                1,
                Bounds { w: 15, h: 60 }
            )
            .is_err()
        );
    }
}
