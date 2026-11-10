//! Units split into an attention half and a MoE half (gpt-oss): the attention columns west of
//! CX, the MoE columns east of it. The units are the standard ones ([`units`]) with their
//! weight / CY kinds refined by half; the scores segments run on the attention columns only
//! ([`AttnScores`]), the router's segments on the CY row's MoE columns ([`Router`]). Copies
//! each unit's staged blocks by half.

use crate::spatial::floor::{Coord, Floor, FloorBuilder, Rect, Unit, UnitGrid};
use crate::spatial::plan::Plan;
use crate::spatial::program::Program;
use crate::spatial::role::{Kind, Place};
use crate::spatial::weights::{Copies, From, Source, chunks};
use crate::spatial::wiring::{Class, Slot, Wiring, segments};

use super::Block;
use super::segments::{ONORM_CAST, ONORM_L1, SCORE_CAST, SCORE_L1, SEGMENT};
use super::units;

type R<T> = Result<T, String>;

/// The attention / MoE halves' weight and CY kinds.
pub const WA: Kind = Kind::new("wa", "wa");
pub const WM: Kind = Kind::new("wm", "wm");
pub const CYA: Kind = Kind::new("cya", "cya");
pub const CYM: Kind = Kind::new("cym", "cym");

/// The router segment a MoE CY PE is in (the o-norm's colours, on the MoE half).
pub const ROUTER: Slot = Slot::segment("router");

/// Words a PE holds of each staged block: the attention half's `w`, the MoE half's `wx` / `wr`,
/// CX's `gam` / `bo` / `bd`, the CY row's attention half's `qb` / `snk` / `rot` and MoE half's
/// `rb` / `gb` / `ps`.
pub struct MoeCopies {
    pub w: u32,
    pub wx: u32,
    pub wr: u32,
    pub cx: [u32; 3],
    pub cya: [u32; 3],
    pub cym: [u32; 3],
    /// The MoE half's expert weights as the prep transformed them.
    pub wx_from: From,
}

/// The units, by halves: `grid.cx` attention columns, then CX, then the MoE columns.
pub struct MoeUnits {
    pub grid: UnitGrid,
    pub copies: MoeCopies,
}

/// Whether local weight column `col` of a unit with CX at `cx` is an attention column.
fn attn(col: u32, cx: u32) -> bool {
    col < cx
}

impl Block for MoeUnits {
    fn floor(&self, _plan: &Plan, b: &mut FloorBuilder) -> R<()> {
        b.add_units(&self.grid, (0, 0))?;
        let cx = self.grid.cx;
        b.each_role(|r| {
            r.kind = match (r.place, r.kind) {
                (Place::Weight { col, .. }, Kind::WEIGHT) => {
                    if attn(col, cx) {
                        WA
                    } else {
                        WM
                    }
                }
                (Place::Cy { col, .. }, Kind::CY) => {
                    if attn(col, cx) {
                        CYA
                    } else {
                        CYM
                    }
                }
                (_, k) => k,
            }
        });
        Ok(())
    }

    fn wire(&self, plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        units::unit_chains(plan, floor, w)
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        units::COLOURS.to_vec()
    }

    /// Per unit, by row halves in blocks of 16 rows: the attention columns' rcfg and `w`, the
    /// MoE columns' rcfg, `wx`, `wr`; then each half's CX rcfg, `gam`, `bo`, `bd`; the CY row's
    /// attention half (rcfg, `qb`, `snk`, `rot`) and MoE half (rcfg, `rb`, `gb`, `ps`).
    fn bindings(&self, prog: &Program, out: &mut Copies) -> R<()> {
        for u in &prog.floor.units {
            self.put(out, u);
        }
        Ok(())
    }
}

impl MoeUnits {
    fn put(&self, out: &mut Copies, unit: &Unit) {
        let c = &self.copies;
        let (ax, py, cy) = (unit.cx, unit.rows(), unit.cy);
        let mx = unit.cols() - ax;
        let (x0, xm, u) = (unit.rect.x, unit.cx_x() + 1, unit.index);
        let staged =
            |name: String, index: (u32, u32), words: u32| Source::Staged { name, index, words };
        for (r0, r1) in [(0, cy), (cy, py)] {
            for (rb, re) in chunks(r0, r1, 16) {
                let y = unit.row_y(rb);
                let ra = Rect::new(x0, y, ax, re - rb);
                out.rcfg(ra);
                out.push(
                    "w",
                    ra,
                    staged(format!("wa{u}"), (rb, re), c.w),
                    true,
                    vec![],
                );
                let rm = Rect::new(xm, y, mx, re - rb);
                out.rcfg(rm);
                let wx = staged(format!("wx{u}"), (rb, re), c.wx);
                out.push("wx", rm, wx, true, c.wx_from.clone());
                out.push(
                    "wr",
                    rm,
                    staged(format!("wr{u}"), (rb, re), c.wr),
                    false,
                    vec![],
                );
            }
            let rc = Rect::new(unit.cx_x(), unit.row_y(r0), 1, r1 - r0);
            out.rcfg(rc);
            for (s, n) in ["gam", "bo", "bd"].into_iter().zip(c.cx) {
                out.push(s, rc, staged(format!("{s}{u}"), (r0, r1), n), false, vec![]);
            }
        }
        let y = unit.cy_y();
        let ca = Rect::new(x0, y, ax, 1);
        out.rcfg(ca);
        let [qb, snk, rot] = c.cya;
        for (s, name, n) in [
            ("qb", format!("qb{u}"), qb),
            ("snk", format!("snk{u}"), snk),
            ("rot", "rot".to_string(), rot),
        ] {
            out.push(s, ca, staged(name, (0, ax), n), false, vec![]);
        }
        let cm = Rect::new(xm, y, mx, 1);
        out.rcfg(cm);
        for (s, n) in ["rb", "gb", "ps"].into_iter().zip(c.cym) {
            out.push(s, cm, staged(format!("{s}{u}"), (0, mx), n), false, vec![]);
        }
    }
}

/// The scores segments of `width` columns on every weight row's attention columns.
pub struct AttnScores {
    pub width: u32,
}

impl Block for AttnScores {
    fn wire(&self, _plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        for u in &floor.units {
            for r in 0..u.rows() {
                let pts: Vec<Coord> = (0..u.cx)
                    .map(|c| Coord::new(u.col_x(c), u.row_y(r)))
                    .collect();
                segments(&pts, self.width, true, (SCORE_L1, SCORE_CAST, SEGMENT), w)?;
            }
        }
        Ok(())
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        super::segments::COLOURS.to_vec()
    }
}

/// The router's segments of `width` columns on every unit's CY row, MoE columns only (the
/// o-norm's colours; [`ROUTER`]).
pub struct Router {
    pub width: u32,
}

impl Block for Router {
    fn wire(&self, _plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        for u in &floor.units {
            let pts: Vec<Coord> = (u.cx..u.cols())
                .map(|c| Coord::new(u.col_x(c), u.cy_y()))
                .collect();
            segments(&pts, self.width, true, (ONORM_L1, ONORM_CAST, ROUTER), w)?;
        }
        Ok(())
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        super::segments::COLOURS.to_vec()
    }
}
