//! The head east of the units: `cols` weight columns around a control column, `rows` rows from
//! row 0. Each head row reduces into the control column (two-level by `reduce.head`, the unit
//! rows' colours, the row word as its multicast); the last unit's x comes in two CX rows a head
//! column (the odd row's PE relays north, the even row's multicasts the column). Copies the
//! head's weights (`hw`), its norm (`hg`) and, under the token loop, the embedding (`hemb`).

use crate::spatial::floor::{Coord, CtlLine, Floor, FloorBuilder, Rect, skip};
use crate::spatial::plan::Plan;
use crate::spatial::program::Program;
use crate::spatial::weights::{Copies, From, Slice, chunks};
use crate::spatial::wiring::{
    Chain, Class, Multicast, NoteKey, Path, Slot, Wiring, level2, loopback,
};

use super::Block;
use super::units::{HANDOFF_ACROSS, ROW_L1, ROW_L2, X_OUT};

type R<T> = Result<T, String>;

/// The odd CX row's x relayed north, the head column's multicast, the head rows' (and the token
/// loop's) row word.
pub const HEAD_RELAY: Class = Class::new("HN");
pub const HEAD_COL_CAST: Class = Class::new("HM");
pub const HEAD_ROW_WORD: Class = Class::new("HR3");

/// The head row reduction a head PE is a member of, its control PE the root.
pub const HEAD_ROW_REDUCE: Slot = Slot::reduce("head_row", true);
/// A CX row's x arriving at its head PE; the relay arriving at the column's source; the column
/// multicast.
pub const HEAD_IN: Slot = Slot::io("head_in");
pub const HEAD_RELAY_IN: Slot = Slot::io("head_relay");
pub const HEAD_CAST: Slot = Slot::io("head_cast");

/// A head PE's part in the head input ([`head_input`]): [`ROLE_RELAY`] on an odd CX row (it
/// relays its x north), [`ROLE_SOURCE`] on the even row (it multicasts the pair down its column).
pub const ROLE: NoteKey = NoteKey("head_in");
pub const ROLE_RELAY: u32 = 1;
pub const ROLE_SOURCE: u32 = 2;

pub const COLOURS: &[(Class, u8)] = &[(HEAD_RELAY, 4), (HEAD_COL_CAST, 6), (HEAD_ROW_WORD, 3)];

/// The head east of the units, fed two CX rows a column.
pub struct HeadEast {
    pub cols: u32,
    pub rows: u32,
    pub token_loop: bool,
    pub copies: HeadCopies,
}

impl Block for HeadEast {
    fn floor(&self, _plan: &Plan, b: &mut FloorBuilder) -> R<()> {
        let x0 = b.w;
        b.head_east(x0, self.cols, self.rows, self.token_loop);
        Ok(())
    }

    fn wire(&self, plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        head_rows(plan, floor, w)?;
        head_input(plan, floor, w)
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        COLOURS.to_vec()
    }

    fn bindings(&self, prog: &Program, out: &mut Copies) -> R<()> {
        self.copies.put(out, prog.floor.head.rect.x);
        Ok(())
    }
}

/// The head's rows: each head row a chain over the head block's columns into the control column
/// (two-level by `reduce.head`; the unit rows' level-1 / level-2 colours, cast
/// [`HEAD_ROW_WORD`]; [`HEAD_ROW_REDUCE`]), the root looping back. For a head with a control
/// column.
pub fn head_rows(plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
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
            l1: ROW_L1,
            l2: level2(ROW_L2, plan.partition.reduce.head),
            cast: HEAD_ROW_WORD,
            loopback: false,
            slot: HEAD_ROW_REDUCE,
        });
        w.turn(loopback(Coord::new(hcol, y)));
    }
    Ok(())
}

/// The last unit's x into the head: head column `c` takes CX rows `2c` and `2c + 1` of the last
/// unit. Each CX row's path east into its head column ([`HANDOFF_ACROSS`]; [`X_OUT`] at CX,
/// [`HEAD_IN`] at the head PE); the odd row's head PE relays north to the even row's
/// ([`HEAD_RELAY`]; [`HEAD_IN`] -> [`HEAD_RELAY_IN`]); the even row's head PE multicasts the
/// column ([`HEAD_COL_CAST`], [`HEAD_CAST`]). Notes [`ROLE`]. Wants head columns = CX rows / 2.
pub fn head_input(_plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
    let hb = &floor.head;
    let ul = floor.units.last().ok_or("wiring: no units")?;
    if hb.cols() * 2 != ul.rows() {
        return Err("head columns = CX rows / 2".into());
    }
    let xa = ul.cx_x();
    for r in 0..ul.rows() {
        let y = ul.row_y(r);
        let hx = hb.col_x(r / 2);
        w.path(Path {
            pts: (xa..=hx).map(|x| Coord::new(x, y)).collect(),
            class: HANDOFF_ACROSS,
            send: X_OUT,
            recv: HEAD_IN,
        });
        if r % 2 == 1 {
            w.note(Coord::new(hx, y), ROLE, ROLE_RELAY);
            w.path(Path {
                pts: vec![Coord::new(hx, y), Coord::new(hx, y - 1)],
                class: HEAD_RELAY,
                send: HEAD_IN,
                recv: HEAD_RELAY_IN,
            });
        }
    }
    for c in 0..hb.cols() {
        let hx = hb.col_x(c);
        let src = ul.row_y(2 * c);
        if src < hb.rect.y || src >= hb.rect.y + hb.rect.h {
            return Err(format!("head column {c}: source row {src} off the head"));
        }
        w.note(Coord::new(hx, src), ROLE, ROLE_SOURCE);
        w.multicast(Multicast {
            pts: (hb.rect.y..hb.rect.y + hb.rect.h)
                .map(|y| Coord::new(hx, y))
                .collect(),
            src: (src - hb.rect.y) as usize,
            class: HEAD_COL_CAST,
            slot: HEAD_CAST,
        });
    }
    Ok(())
}

/// Where the head's embedding table goes: nowhere, with each block, or after every `hw` / `hg`.
pub enum Hemb {
    None,
    Inline(From),
    After(From),
}

/// The head's copies: each half of its columns (the control column skipped) in blocks of 64
/// rows: rcfg, `hw` (the slice of its block: half, rows, columns), `hg` repeated down the
/// block, `hemb` as [`Hemb`] says; then the control column's rcfg. Placement order is part of
/// the host's device protocol.
pub struct HeadCopies {
    /// The head's weight columns, rows.
    pub cols: u32,
    pub rows: u32,
    pub hw: fn(usize, (u32, u32), (u32, u32)) -> Slice,
    pub hw_from: From,
    pub hg_from: From,
    pub hemb: Hemb,
}

impl HeadCopies {
    /// The copies of a head whose first column is at wafer x `x0`.
    pub fn put(&self, out: &mut Copies, x0: u32) {
        let hxc = self.cols / 2;
        let halves: Vec<(u32, u32)> = [(0, hxc), (hxc, self.cols)]
            .into_iter()
            .filter(|(a, b)| b > a)
            .collect();
        let rect = |c0: u32, c1: u32, rb: u32, re: u32| {
            Rect::new(x0 + skip(c0, hxc), rb, c1 - c0, re - rb)
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
        out.rcfg(Rect::new(x0 + hxc, 0, 1, self.rows));
    }
}
