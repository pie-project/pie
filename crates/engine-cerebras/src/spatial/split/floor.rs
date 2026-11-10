//! The floorplan: where every part of the program sits on the wafer and what each PE is.
//! A [`Floor`] is pure placement: rectangles, the unit visit order and one [`PeRole`] a PE. It
//! holds no colours or routes (those are [`super::fabric`]'s).

use serde::{Deserialize, Serialize};

use super::role::{Cx, Cy, Head, HeadCtl, PeRole, Weight, WeightPart};

/// Position `i` of an axis whose control line sits at `c`, the line skipped.
pub fn skip(i: u32, c: u32) -> u32 {
    i + u32::from(i >= c)
}

/// The inverse of [`skip`] for a position off the line.
fn unskip(i: u32, c: u32) -> u32 {
    i - u32::from(i > c)
}

fn overlap(a: &Rect, b: &Rect) -> bool {
    a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
}

/// A PE position on the wafer (x east, y south).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Coord {
    pub x: u32,
    pub y: u32,
}

impl Coord {
    pub const fn new(x: u32, y: u32) -> Coord {
        Coord { x, y }
    }

    /// The direction of an adjacent `to`; `None` when not adjacent.
    pub fn dir_to(self, to: Coord) -> Option<Dir> {
        if to.y == self.y && to.x == self.x + 1 {
            Some(Dir::E)
        } else if to.y == self.y && to.x + 1 == self.x {
            Some(Dir::W)
        } else if to.x == self.x && to.y == self.y + 1 {
            Some(Dir::S)
        } else if to.x == self.x && to.y + 1 == self.y {
            Some(Dir::N)
        } else {
            None
        }
    }
}

/// A fabric direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dir {
    W,
    E,
    S,
    N,
}

impl Dir {
    pub fn opposite(self) -> Dir {
        match self {
            Dir::W => Dir::E,
            Dir::E => Dir::W,
            Dir::S => Dir::N,
            Dir::N => Dir::S,
        }
    }
}

/// A PE rectangle (inclusive origin, `w` x `h` PEs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub const fn new(x: u32, y: u32, w: u32, h: u32) -> Rect {
        Rect { x, y, w, h }
    }

    pub fn contains(&self, c: Coord) -> bool {
        c.x >= self.x && c.x < self.x + self.w && c.y >= self.y && c.y < self.y + self.h
    }

    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }
}

/// A dense per-PE grid (row-major), the storage of every per-PE fact (roles, routes, queues).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grid<T> {
    pub w: u32,
    pub h: u32,
    cells: Vec<T>,
}

impl<T: Clone> Grid<T> {
    pub fn filled(w: u32, h: u32, v: T) -> Grid<T> {
        Grid {
            w,
            h,
            cells: vec![v; (w * h) as usize],
        }
    }
}

impl<T> Grid<T> {
    pub fn contains(&self, c: Coord) -> bool {
        c.x < self.w && c.y < self.h
    }

    pub fn get(&self, c: Coord) -> &T {
        &self.cells[(c.y * self.w + c.x) as usize]
    }

    pub fn get_mut(&mut self, c: Coord) -> &mut T {
        &mut self.cells[(c.y * self.w + c.x) as usize]
    }

    /// Every cell with its coordinate, row-major.
    pub fn iter(&self) -> impl Iterator<Item = (Coord, &T)> {
        let w = self.w;
        self.cells
            .iter()
            .enumerate()
            .map(move |(i, v)| (Coord::new(i as u32 % w, i as u32 / w), v))
    }
}

/// A unit: `rows` x `cols` weight PEs around a control column (CX, at local x = `cx`) and a
/// control row (CY, at local y = `cy`), running `layers` consecutive layers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unit {
    pub index: u32,
    /// Its rectangle (weight PEs + CX column + CY row + the filler corner).
    pub rect: Rect,
    /// Local x of the CX column, local y of the CY row.
    pub cx: u32,
    pub cy: u32,
    /// The layers it runs (global indices, half-open: `[layers.0, layers.1)`).
    pub layers: (u32, u32),
}

impl Unit {
    /// Wafer y of weight row `r`.
    pub fn row_y(&self, r: u32) -> u32 {
        self.rect.y + skip(r, self.cy)
    }

    /// Wafer x of weight column `c` (the CX column skipped).
    pub fn col_x(&self, c: u32) -> u32 {
        self.rect.x + skip(c, self.cx)
    }

    /// Weight rows / columns (the control row / column not counted).
    pub fn rows(&self) -> u32 {
        self.rect.h - 1
    }

    pub fn cols(&self) -> u32 {
        self.rect.w - 1
    }

    /// The unit's top-left PE.
    pub fn origin(&self) -> Coord {
        Coord::new(self.rect.x, self.rect.y)
    }

    pub fn cx_x(&self) -> u32 {
        self.rect.x + self.cx
    }

    pub fn cy_y(&self) -> u32 {
        self.rect.y + self.cy
    }
}

/// The head block: `cols` weight columns with the head control column in their middle, `rows`
/// rows (gpt-oss: a band below the units with its control row; Gemma 12B: one column a head group).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadBlock {
    pub rect: Rect,
    /// The head control line, in wafer coordinates: a column's x (Qwen, Gemma) or a row's y
    /// (gpt-oss's band).
    pub ctl: CtlLine,
    /// The token loop's root (the head control PE at the middle row): the ring, force, nforce.
    pub root: Coord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CtlLine {
    Column(u32),
    Row(u32),
}

impl HeadBlock {
    /// Wafer x of head weight column `c`.
    pub fn col_x(&self, c: u32) -> u32 {
        match self.ctl {
            CtlLine::Column(x) => self.rect.x + skip(c, x - self.rect.x),
            CtlLine::Row(_) => self.rect.x + c,
        }
    }

    /// Head weight columns (the control column not counted).
    pub fn cols(&self) -> u32 {
        match self.ctl {
            CtlLine::Column(_) => self.rect.w - 1,
            CtlLine::Row(_) => self.rect.w,
        }
    }

    /// Wafer x of the control column (`None` for a control row).
    pub fn ctl_x(&self) -> Option<u32> {
        match self.ctl {
            CtlLine::Column(x) => Some(x),
            CtlLine::Row(_) => None,
        }
    }
}

/// Gemma's PLE storage block above a unit column: its rows, the tokens a storage PE holds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreBlock {
    pub rect: Rect,
    /// The units of the column it serves, top to bottom.
    pub units: Vec<u32>,
    pub tokens_per_pe: u32,
}

/// The whole floorplan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Floor {
    pub w: u32,
    pub h: u32,
    /// Units in visit order (the x handoff follows it; unit 0 takes the embedding).
    pub units: Vec<Unit>,
    /// Gap slots: rectangles whose PEs only relay the vertical handoff through them.
    pub relays: Vec<Rect>,
    pub head: HeadBlock,
    pub stores: Vec<StoreBlock>,
    /// Every PE's role.
    pub roles: Grid<PeRole>,
}

impl Floor {
    /// The unit whose rectangle holds `c`.
    pub fn unit_at(&self, c: Coord) -> Option<&Unit> {
        self.units.iter().find(|u| u.rect.contains(c))
    }

    /// Maximal rectangles of PEs with an equal role kind, row-major: the copies the placer and
    /// the layout emitter iterate.
    /// Greedy and deterministic: from the first uncovered PE (row-major), the widest run of its
    /// kind, then as many rows down as repeat that run uncovered.
    pub fn rects_by_kind(&self) -> Vec<(super::role::RoleKind, Rect)> {
        let (w, h) = (self.w, self.h);
        let mut done = vec![false; (w * h) as usize];
        let kind = |x: u32, y: u32| self.roles.get(Coord::new(x, y)).kind();
        let mut out = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if done[(y * w + x) as usize] {
                    continue;
                }
                let k = kind(x, y);
                let free = |done: &[bool], xx: u32, yy: u32| {
                    !done[(yy * w + xx) as usize] && kind(xx, yy) == k
                };
                let mut rw = 1;
                while x + rw < w && free(&done, x + rw, y) {
                    rw += 1;
                }
                let mut rh = 1;
                while y + rh < h && (x..x + rw).all(|xx| free(&done, xx, y + rh)) {
                    rh += 1;
                }
                for yy in y..y + rh {
                    for xx in x..x + rw {
                        done[(yy * w + xx) as usize] = true;
                    }
                }
                out.push((k, Rect { x, y, w: rw, h: rh }));
            }
        }
        out
    }
}

/// Unit placement strategies the families compose (the snake, Gemma's explicit
/// origins, gpt-oss's snake that starts upward).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitLayout {
    /// `unx` x `uny` unit slots visited as a snake; `up_first`: column 0 runs upward (gpt-oss);
    /// `gaps`: slots left out (relay only).
    Snake {
        unx: u32,
        uny: u32,
        up_first: bool,
        gaps: Vec<(u32, u32)>,
    },
    /// Explicit unit origins in visit order (Gemma, Gemma 12B).
    Explicit { origins: Vec<(u32, u32)> },
}

/// The unit size and placement a family asks for; [`FloorBuilder`] turns it into units.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitGrid {
    /// Weight columns / rows a unit (the control column / row come on top).
    pub cols: u32,
    pub rows: u32,
    /// Local x of the CX column, local y of the CY row (default cols / 2, rows / 2; gpt-oss
    /// puts CX after the attention columns).
    pub cx: u32,
    pub cy: u32,
    pub layout: UnitLayout,
    pub layers_per_unit: u32,
}

/// Generic floor construction the families call; each family adds its own blocks (head,
/// stores) and fills roles through [`FloorBuilder::set_role`].
pub struct FloorBuilder {
    pub w: u32,
    pub h: u32,
    pub units: Vec<Unit>,
    pub relays: Vec<Rect>,
    pub stores: Vec<StoreBlock>,
    roles: Grid<PeRole>,
}

impl FloorBuilder {
    /// Units per `grid`, the wafer sized to hold them plus `extra` (w, h) for the family's
    /// blocks; every unit PE gets its generic role (weight / CX / CY / filler corner) and gap
    /// slots `Relay`.
    ///
    /// The units' bounding box is `unx (cols + 1)` x `uny (rows + 1)` for a snake (gap slots
    /// included), the farthest origin plus a unit for explicit origins. The wafer is that box
    /// plus `extra`: `extra.0` columns east (the head block's `hx0` is the box's width) and
    /// `extra.1` rows below (`w = hx0 + HC + 1`, `h = max(UNY UH, HR)`: a family passes `(HC + 1, HR.saturating_sub(box height))`). A block elsewhere (gpt-oss's
    /// band below) passes `(0, 0)` and calls [`FloorBuilder::grow`].
    ///
    /// Unit `u` runs layers `[u K, (u + 1) K)`. Generic roles: `Weight { part: Dense }` at
    /// local (c, r), `Cx { first: u == 0, last: u == n - 1 }`, `Cy { part: Dense }`, `Idle` at
    /// the corner; a family overwrites them where its kernels differ (gpt-oss's halves).
    pub fn units(grid: &UnitGrid, extra: (u32, u32)) -> Result<FloorBuilder, String> {
        if grid.cx > grid.cols || grid.cy > grid.rows {
            return Err(format!(
                "floor: control lines ({}, {}) outside a {} x {} unit",
                grid.cx, grid.cy, grid.cols, grid.rows
            ));
        }
        let (uw, uh) = (grid.cols + 1, grid.rows + 1);
        let (origins, gaps, bw, bh) = match &grid.layout {
            UnitLayout::Snake {
                unx,
                uny,
                up_first,
                gaps,
            } => {
                let mut order = Vec::new();
                for ux in 0..*unx {
                    let down = ux.is_multiple_of(2) != *up_first;
                    if down {
                        order.extend((0..*uny).map(|uy| (ux, uy)));
                    } else {
                        order.extend((0..*uny).rev().map(|uy| (ux, uy)));
                    }
                }
                for g in gaps {
                    if g.0 >= *unx || g.1 >= *uny {
                        return Err(format!("floor: gap slot {g:?} outside {unx} x {uny}"));
                    }
                }
                order.retain(|s| !gaps.contains(s));
                let origins: Vec<(u32, u32)> =
                    order.iter().map(|&(ux, uy)| (ux * uw, uy * uh)).collect();
                let gaps: Vec<Rect> = gaps
                    .iter()
                    .map(|&(gx, gy)| Rect {
                        x: gx * uw,
                        y: gy * uh,
                        w: uw,
                        h: uh,
                    })
                    .collect();
                (origins, gaps, unx * uw, uny * uh)
            }
            UnitLayout::Explicit { origins } => {
                let bw = origins.iter().map(|o| o.0 + uw).max().unwrap_or(0);
                let bh = origins.iter().map(|o| o.1 + uh).max().unwrap_or(0);
                (origins.clone(), Vec::new(), bw, bh)
            }
        };
        if origins.is_empty() {
            return Err("floor: no units".into());
        }
        let w = bw + extra.0;
        let h = bh + extra.1;
        let mut roles = Grid::filled(w, h, PeRole::Idle);
        let n = origins.len() as u32;
        let k = grid.layers_per_unit;
        let mut units = Vec::with_capacity(origins.len());
        for (i, &(x0, y0)) in origins.iter().enumerate() {
            let u = i as u32;
            let rect = Rect {
                x: x0,
                y: y0,
                w: uw,
                h: uh,
            };
            if units.iter().any(|o: &Unit| overlap(&o.rect, &rect)) {
                return Err(format!("floor: unit {u} at ({x0}, {y0}) overlaps another"));
            }
            for ly in 0..uh {
                for lx in 0..uw {
                    let role = match (lx == grid.cx, ly == grid.cy) {
                        (true, true) => PeRole::Idle,
                        (true, false) => PeRole::Cx(Cx {
                            unit: u,
                            row: unskip(ly, grid.cy),
                            first: u == 0,
                            last: u + 1 == n,
                        }),
                        (false, true) => PeRole::Cy(Cy {
                            unit: u,
                            col: unskip(lx, grid.cx),
                            part: WeightPart::Dense,
                        }),
                        (false, false) => PeRole::Weight(Weight {
                            unit: u,
                            row: unskip(ly, grid.cy),
                            col: unskip(lx, grid.cx),
                            part: WeightPart::Dense,
                        }),
                    };
                    *roles.get_mut(Coord::new(x0 + lx, y0 + ly)) = role;
                }
            }
            units.push(Unit {
                index: u,
                rect,
                cx: grid.cx,
                cy: grid.cy,
                layers: (u * k, (u + 1) * k),
            });
        }
        for g in &gaps {
            for y in g.y..g.y + g.h {
                for x in g.x..g.x + g.w {
                    *roles.get_mut(Coord::new(x, y)) = PeRole::Relay;
                }
            }
        }
        Ok(FloorBuilder {
            w,
            h,
            units,
            relays: gaps,
            stores: Vec::new(),
            roles,
        })
    }

    pub fn set_role(&mut self, c: Coord, role: PeRole) {
        *self.roles.get_mut(c) = role;
    }

    pub fn role(&self, c: Coord) -> &PeRole {
        self.roles.get(c)
    }

    /// Applies `f` to every PE's role (a family's refinement of the generic roles).
    pub fn each_role(&mut self, f: impl FnMut(&mut PeRole)) {
        self.roles.cells.iter_mut().for_each(f);
    }

    /// Grows the wafer to at least `w` x `h` (a family's head or store block past the units);
    /// new PEs are `Idle`.
    pub fn grow(&mut self, w: u32, h: u32) {
        let (nw, nh) = (w.max(self.w), h.max(self.h));
        if (nw, nh) == (self.w, self.h) {
            return;
        }
        let mut roles = Grid::filled(nw, nh, PeRole::Idle);
        for (c, r) in self.roles.iter() {
            *roles.get_mut(c) = r.clone();
        }
        self.roles = roles;
        self.w = nw;
        self.h = nh;
    }

    /// The head east of the units from wafer x `x0`: `cols` weight columns with the control
    /// column in their middle, `rows` rows from row 0 (the wafer grown to hold it). The control
    /// column's middle row roots the token loop when it runs.
    pub fn head_east(&mut self, x0: u32, cols: u32, rows: u32, token_loop: bool) -> HeadBlock {
        self.grow(x0 + cols + 1, rows);
        let hxc = cols / 2;
        let root = Coord::new(x0 + hxc, rows / 2);
        for y in 0..rows {
            for i in 0..=cols {
                let role = if i == hxc {
                    PeRole::HeadCtl(HeadCtl {
                        row: y,
                        root: token_loop && y == root.y,
                    })
                } else {
                    PeRole::Head(Head {
                        row: y,
                        col: unskip(i, hxc),
                    })
                };
                self.set_role(Coord::new(x0 + i, y), role);
            }
        }
        HeadBlock {
            rect: Rect::new(x0, 0, cols + 1, rows),
            ctl: CtlLine::Column(x0 + hxc),
            root,
        }
    }

    pub fn finish(self, head: HeadBlock) -> Floor {
        Floor {
            w: self.w,
            h: self.h,
            units: self.units,
            relays: self.relays,
            head,
            stores: self.stores,
            roles: self.roles,
        }
    }
}
