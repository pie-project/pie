//! The copies the snake-unit families share: a unit's weight quadrants, CX column and CY row,
//! and the head's halves. Placement order is part of the host's device protocol, so each
//! function keeps the order its families copied in.

use crate::spatial::split::floor::{Rect, Unit, skip};
use crate::spatial::split::weights::{Copies, From, Slice, chunks};

/// A unit's copies, quadrant by quadrant (rows, then columns): its rcfg and `w{u}` in blocks
/// of 16 rows; then each half of its CX column (rcfg, `gam`); then each half of its CY row
/// (rcfg, each of `cy`: (symbol, whether the table is per unit), optional when `cy_optional`).
pub struct UnitCopies<'a> {
    pub cols: u32,
    pub rows: u32,
    pub w_from: From,
    pub gam_b16: bool,
    pub gam_from: From,
    pub cy: &'a [(&'static str, bool)],
    pub cy_optional: bool,
}

impl UnitCopies<'_> {
    pub fn put(&self, out: &mut Copies, unit: &Unit) {
        let (px, py) = (self.cols, self.rows);
        let (xc, yc) = (px / 2, py / 2);
        let (x0u, y0u, u) = (unit.rect.x, unit.rect.y, unit.index);
        for (r0, r1) in [(0, yc), (yc, py)] {
            for (c0, c1) in [(0, xc), (xc, px)] {
                let x0 = x0u + skip(c0, xc);
                out.rcfg(Rect::new(x0, y0u + skip(r0, yc), c1 - c0, r1 - r0));
                for (rb, re) in chunks(r0, r1, 16) {
                    out.slice(
                        "w",
                        Rect::new(x0, y0u + skip(rb, yc), c1 - c0, re - rb),
                        Slice::of(&format!("w{u}"), None, vec![(rb, re), (c0, c1)]),
                        true,
                        self.w_from.clone(),
                    );
                }
            }
        }
        for (r0, r1) in [(0, yc), (yc, py)] {
            let rect = Rect::new(x0u + xc, y0u + skip(r0, yc), 1, r1 - r0);
            out.rcfg(rect);
            let gam = Slice::of("gam", Some(u), vec![(r0, r1)]);
            out.slice("gam", rect, gam, self.gam_b16, self.gam_from.clone());
        }
        for (c0, c1) in [(0, xc), (xc, px)] {
            let rect = Rect::new(x0u + skip(c0, xc), y0u + yc, c1 - c0, 1);
            out.rcfg(rect);
            for &(sym, per_unit) in self.cy {
                let s = Slice::of(sym, per_unit.then_some(u), vec![(c0, c1)]);
                let s = if self.cy_optional { s.optional() } else { s };
                out.slice(sym, rect, s, false, Vec::new());
            }
        }
    }
}

/// Where the head's embedding table goes: nowhere, with each block, or after every `hw` / `hg`.
pub enum Hemb {
    None,
    Inline(From),
    After(From),
}

/// The head's copies: each half of its columns (the control column skipped) in blocks of 64
/// rows: rcfg, `hw` (the family's slice of its block: half, rows, columns), `hg` repeated down
/// the block, `hemb` as [`Hemb`] says; then the control column's rcfg.
pub struct HeadCopies<'a> {
    /// Wafer x of the head's first column, its weight columns, rows.
    pub x0: u32,
    pub cols: u32,
    pub rows: u32,
    pub hw: &'a dyn Fn(usize, (u32, u32), (u32, u32)) -> Slice,
    pub hw_from: From,
    pub hg_from: From,
    pub hemb: Hemb,
}

impl HeadCopies<'_> {
    pub fn put(&self, out: &mut Copies) {
        let hxc = self.cols / 2;
        let halves: Vec<(u32, u32)> = [(0, hxc), (hxc, self.cols)]
            .into_iter()
            .filter(|(a, b)| b > a)
            .collect();
        let rect = |c0: u32, c1: u32, rb: u32, re: u32| {
            Rect::new(self.x0 + skip(c0, hxc), rb, c1 - c0, re - rb)
        };
        let hemb = |out: &mut Copies, r: Rect, c0, c1, rb, re, from: &From| {
            let s = Slice::of("hemb", None, vec![(rb, re), (c0, c1)]);
            out.slice("hemb", r, s, true, from.clone());
        };
        for (i, &(c0, c1)) in halves.iter().enumerate() {
            for (rb, re) in chunks(0, self.rows, 64) {
                let r = rect(c0, c1, rb, re);
                out.rcfg(r);
                let hw = (self.hw)(i, (rb, re), (c0, c1));
                out.slice("hw", r, hw, true, self.hw_from.clone());
                let hg = Slice::of("hg", None, vec![(c0, c1)]).repeat(re - rb);
                out.slice("hg", r, hg, false, self.hg_from.clone());
                if let Hemb::Inline(f) = &self.hemb {
                    hemb(out, r, c0, c1, rb, re, f);
                }
            }
        }
        if let Hemb::After(f) = &self.hemb {
            for &(c0, c1) in &halves {
                for (rb, re) in chunks(0, self.rows, 64) {
                    hemb(out, rect(c0, c1, rb, re), c0, c1, rb, re, f);
                }
            }
        }
        out.rcfg(Rect::new(self.x0 + hxc, 0, 1, self.rows));
    }
}
