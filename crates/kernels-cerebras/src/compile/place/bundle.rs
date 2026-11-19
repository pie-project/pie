//! The bundle router: wires that travel together (a row stream's gap, a unit's K / V links) laid
//! as one staircase. Every wire is a Z (leg, turn, leg), the turns ordered so no two wires of the
//! bundle cross or share a PE.

use crate::compile::floor::Coord;
use crate::compile::floorplan::stream::StreamUnit;

/// Where a Z turns: a column (the wire runs along its source's row, down or up that column,
/// along its sink's row) or a row (down or up its source's column, along that row, down or up
/// its sink's column).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Turn {
    Column(u32),
    Row(u32),
}

/// `from ..= to`, either way.
fn line(from: u32, to: u32) -> Box<dyn Iterator<Item = u32>> {
    if from <= to {
        Box::new(from..=to)
    } else {
        Box::new((to..=from).rev())
    }
}

/// The Z from `src` to `dst` turning at `turn`: every PE on its way, its ends included.
#[must_use]
pub fn z(src: Coord, dst: Coord, turn: Turn) -> Vec<Coord> {
    let mut pts = Vec::new();
    match turn {
        Turn::Column(t) => {
            pts.extend(line(src.x, t).map(|x| Coord::new(x, src.y)));
            pts.extend(line(src.y, dst.y).skip(1).map(|y| Coord::new(t, y)));
            pts.extend(line(t, dst.x).skip(1).map(|x| Coord::new(x, dst.y)));
        }
        Turn::Row(t) => {
            pts.extend(line(src.y, t).map(|y| Coord::new(src.x, y)));
            pts.extend(line(src.x, dst.x).skip(1).map(|x| Coord::new(x, t)));
            pts.extend(line(t, dst.y).skip(1).map(|y| Coord::new(dst.x, y)));
        }
    }
    pts
}

/// Row `j`'s turn in the row stream's gap from unit `a` to unit `b` (`py` rows): none on `b`'s
/// rows, else a column beside `b`, ordered so the staircase does not cross itself.
#[must_use]
pub fn gap_turn(a: &StreamUnit, b: &StreamUnit, py: u32, j: u32) -> Turn {
    if a.y == b.y {
        return Turn::Column(b.x);
    }
    let down = b.y > a.y;
    let west = b.x < a.x;
    Turn::Column(match (down, west) {
        (true, true) => b.x + 1 + j,
        (true, false) => b.x.saturating_sub(1 + j),
        (false, true) => b.x + py - j,
        (false, false) => b.x.saturating_sub(py - j),
    })
}

/// Row `j`'s wire in the gap from unit `a` to unit `b`: from `a`'s CX root `j` to `b`'s.
#[must_use]
pub fn gap(a: &StreamUnit, b: &StreamUnit, py: u32, j: u32) -> Vec<Coord> {
    z(a.cx(j), b.cx(j), gap_turn(a, b, py, j))
}

/// A head's link from its unit's head root `top` to its K / V column's root, turning on
/// `turn` (its head's own row, below the block's units), or straight down where the root is
/// below the head root.
#[must_use]
pub fn kv_link(top: Coord, root: Coord, turn: u32) -> Vec<Coord> {
    let t = if top.x == root.x {
        root.y
    } else {
        turn.min(root.y)
    };
    z(top, root, Turn::Row(t))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The band layout's wrap as it was written before the bundle router: the oracle.
    fn old_wrap(ax: u32, ay: u32, bx: u32, by: u32, j: u32) -> Vec<Coord> {
        let turn = bx + 1 + j;
        let (ya, yb) = (ay + 1 + j, by + 1 + j);
        let mut pts: Vec<Coord> = (turn..=ax).rev().map(|x| Coord::new(x, ya)).collect();
        pts.extend((ya + 1..=yb).map(|y| Coord::new(turn, y)));
        pts.extend((bx..turn).rev().map(|x| Coord::new(x, yb)));
        pts
    }

    /// The band layout's K / V link as it was written: the oracle.
    fn old_link(top: Coord, root: Coord, turn: u32) -> Vec<Coord> {
        let turn = if top.x == root.x {
            root.y
        } else {
            turn.min(root.y)
        };
        let mut pts: Vec<Coord> = (top.y..=turn).map(|y| Coord::new(top.x, y)).collect();
        let mut x = top.x;
        while x != root.x {
            x = if x > root.x { x - 1 } else { x + 1 };
            pts.push(Coord::new(x, turn));
        }
        pts.extend((turn + 1..=root.y).map(|y| Coord::new(root.x, y)));
        pts
    }

    /// The bundle router's Zs are the band layout's wraps, straight rows and K / V links point
    /// for point, over every geometry the layout makes.
    #[test]
    fn the_bundle_router_is_the_band_layout_where_it_laid() {
        for (ax, ay, bx, by) in [(40, 0, 0, 30), (100, 5, 7, 60), (33, 0, 1, 9)] {
            for j in 0..(ax - bx).min(16) {
                let turn = Turn::Column(bx + 1 + j);
                let got = z(Coord::new(ax, ay + 1 + j), Coord::new(bx, by + 1 + j), turn);
                assert_eq!(got, old_wrap(ax, ay, bx, by, j), "{ax} {ay} {bx} {by} {j}");
            }
        }
        // a straight row: (a.x ..= b.x) along row j
        let got = z(Coord::new(3, 7), Coord::new(19, 7), Turn::Column(19));
        let want: Vec<Coord> = (3..=19).map(|x| Coord::new(x, 7)).collect();
        assert_eq!(got, want);
        for (top, root, turn) in [
            (Coord::new(5, 10), Coord::new(5, 30), 14),
            (Coord::new(9, 10), Coord::new(5, 30), 14),
            (Coord::new(2, 10), Coord::new(5, 30), 40),
            (Coord::new(12, 3), Coord::new(4, 9), 6),
        ] {
            assert_eq!(kv_link(top, root, turn), old_link(top, root, turn));
        }
    }

    /// A staircase of any direction keeps its wires apart: no PE on two wires.
    #[test]
    fn a_staircase_does_not_cross_itself() {
        use std::collections::BTreeSet;
        let py = 6;
        for (ax, ay, bx, by) in [
            (60, 0, 10, 40),
            (10, 0, 60, 40),
            (60, 40, 10, 0),
            (10, 40, 60, 0),
        ] {
            let mut seen = BTreeSet::new();
            for j in 0..py {
                let src = Coord::new(ax, ay + 1 + j);
                let dst = Coord::new(bx, by + 1 + j);
                let down = by > ay;
                let west = bx < ax;
                let t = match (down, west) {
                    (true, true) => bx + 1 + j,
                    (true, false) => bx - 1 - j,
                    (false, true) => bx + py - j,
                    (false, false) => bx - py + j,
                };
                for c in z(src, dst, Turn::Column(t)) {
                    assert!(seen.insert(c), "{ax} {ay} -> {bx} {by}: {c:?} twice");
                }
            }
        }
    }
}
