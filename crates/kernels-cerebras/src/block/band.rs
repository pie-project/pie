//! The head as a band below the units (gpt-oss): `rows` dims-rows (dims-row i takes the last
//! unit's CX row i) and the head control row in their middle, by `cols` vocabulary columns from
//! x 0. The last unit's x leaves each CX row east along the row to MoE column r, down it to
//! dims-row r ([`BAND_VERT`]) and is multicast along the dims-row; each band column reduces into
//! the control row. Under the token loop ([`BandLoop`]) the control row max-reduces the argmax,
//! and the winner's embedding returns west along each dims-row (the band bus, [`BUS_COLOUR`]),
//! up unit 0's MoE column i ([`BAND_V1`]) and west into its CX row i ([`BAND_W2`]).
//!
//! Wants the units in two unit columns visited so that the first and the last unit sit on the
//! band, and the band's rows = the units' rows <= their MoE columns.

use crate::floor::{Coord, CtlLine, Floor, FloorBuilder, HeadBlock, Rect, skip};
use crate::plan::Plan;
use crate::program::Program;
use crate::role::{Kind, PeRole, Place};
use crate::weights::{Copies, Slice, Transform, chunks};
use crate::wiring::{Chain, Class, Multicast, Path, Slot, Wiring, level2, loopback};

use super::Block;
use super::head::{self, HEAD_COL_CAST, HEAD_IN, HEAD_ROW_REDUCE, HEAD_ROW_WORD};
use super::token_loop::{self, EMB_WEST};
use super::units::{HANDOFF_OUT, ROW_L1, ROW_L2, X_OUT};

type R<T> = Result<T, String>;

/// The x's turn down a MoE column into the band; the embedding's return up unit 0's MoE column
/// and west into its CX.
pub const BAND_VERT: Class = Class::new("HV2");
pub const BAND_V1: Class = Class::new("V1");
pub const BAND_W2: Class = Class::new("W2");

/// The band bus's colour (the token loop's `EMB_WEST`, recoloured): the band's head and control
/// PEs never park on it and count it as final.
pub const BUS_COLOUR: u8 = 10;

/// The x leaving its MoE row for the band (the turning weight PE's in / out); the dims-row
/// multicast.
pub const BAND_IN: Slot = Slot::io("band_in");
pub const BAND_OUT: Slot = Slot::io("band_out");
pub const BAND_CAST: Slot = Slot::io("band_cast");
/// The band bus: its east end's send, dims-row 0's take.
pub const BUS_TX: Slot = Slot::io("band_bus_tx");
pub const BUS: Slot = Slot::io("band_bus");
/// The embedding's return: up unit 0's MoE column (sent, taken at the turn) and west into CX.
pub const V1_TX: Slot = Slot::io("band_v1_tx");
pub const V1_RX: Slot = Slot::io("band_v1_rx");
pub const W2_TX: Slot = Slot::io("band_w2_tx");

/// The band head below the units.
pub struct Band {
    pub cols: u32,
    pub rows: u32,
    /// Vocabulary rows a band PE (the lm_head / embedding tables padded to `cols x vocab_rows`).
    pub vocab_rows: u32,
}

/// The band's geometry on a floor: its first row, its control row's local index.
struct At {
    y0: u32,
    hxc: u32,
}

impl At {
    fn of(floor: &Floor) -> At {
        let r = &floor.head.rect;
        At {
            y0: r.y,
            hxc: (r.h - 1) / 2,
        }
    }

    /// Dims-row i's y.
    fn row(&self, i: u32) -> u32 {
        self.y0 + skip(i, self.hxc)
    }

    fn ctl(&self) -> u32 {
        self.y0 + self.hxc
    }
}

impl Block for Band {
    fn floor(&self, _plan: &Plan, b: &mut FloorBuilder) -> R<()> {
        let y0 = b.h;
        let hxc = self.rows / 2;
        b.grow(b.w.max(self.cols), y0 + self.rows + 1);
        for ly in 0..=self.rows {
            for x in 0..self.cols {
                let role = if ly == hxc {
                    PeRole::new(
                        Kind::HEAD_CTL,
                        Place::HeadCtl {
                            row: x,
                            root: x == self.cols / 2,
                        },
                    )
                } else {
                    PeRole::new(
                        Kind::HEAD,
                        Place::Head {
                            row: ly - u32::from(ly > hxc),
                            col: x,
                        },
                    )
                };
                b.set_role(Coord::new(x, y0 + ly), role);
            }
        }
        b.head = Some(HeadBlock {
            rect: Rect::new(0, y0, self.cols, self.rows + 1),
            ctl: CtlLine::Row(y0 + hxc),
            root: Coord::new(self.cols / 2, y0 + hxc),
        });
        Ok(())
    }

    /// The last unit's x into the band: CX row r east to MoE column r ([`HANDOFF_OUT`]), down
    /// it to dims-row r ([`BAND_VERT`]), multicast along the dims-row ([`HEAD_COL_CAST`]); each
    /// band column a chain into the control row (two-level by `reduce.head`).
    fn wire(&self, plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        let at = At::of(floor);
        let ul = floor.units.last().ok_or("band: no units")?;
        if ul.rect.y + ul.rect.h != at.y0 || self.rows != ul.rows() || self.rows > ul.cols() - ul.cx
        {
            return Err("band: the last unit sits on the band, its rows <= its MoE columns".into());
        }
        for r in 0..self.rows {
            let y = ul.row_y(r);
            let xr = ul.cx_x() + 1 + r;
            if xr >= self.cols {
                return Err("band: the x path's column past the band".into());
            }
            w.path(Path {
                pts: (ul.cx_x()..=xr).map(|x| Coord::new(x, y)).collect(),
                class: HANDOFF_OUT,
                send: X_OUT,
                recv: BAND_IN,
            });
            let yb = at.row(r);
            w.path(Path {
                pts: (y..=yb).map(|yy| Coord::new(xr, yy)).collect(),
                class: BAND_VERT,
                send: BAND_OUT,
                recv: HEAD_IN,
            });
            w.multicast(Multicast {
                pts: (0..self.cols).map(|j| Coord::new(j, yb)).collect(),
                src: xr as usize,
                class: HEAD_COL_CAST,
                slot: BAND_CAST,
            });
        }
        for j in 0..self.cols {
            w.chain(Chain {
                pts: (0..=self.rows)
                    .map(|ly| Coord::new(j, at.y0 + ly))
                    .collect(),
                root: at.hxc as usize,
                l1: ROW_L1,
                l2: level2(ROW_L2, plan.partition.reduce.head),
                cast: HEAD_ROW_WORD,
                loopback: false,
                slot: HEAD_ROW_REDUCE,
            });
            w.turn(loopback(Coord::new(j, at.ctl())));
        }
        Ok(())
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        let mut c = head::COLOURS.to_vec();
        c.push((BAND_VERT, 9));
        c
    }

    /// By dims-row halves in blocks of 16 rows and 64 columns: rcfg, `hw`, `hemb`, `hg` a
    /// dims-row; then the control row's rcfg.
    fn bindings(&self, prog: &Program, out: &mut Copies) -> R<()> {
        let at = At::of(&prog.floor);
        let pad = |name: &str| {
            vec![(
                name.to_string(),
                vec![Transform::PadLastRow {
                    to: self.cols * self.vocab_rows,
                }],
            )]
        };
        for (i0, i1) in [(0, at.hxc), (at.hxc, self.rows)] {
            for (ib, ie) in chunks(i0, i1, 16) {
                for (cb, ce) in chunks(0, self.cols, 64) {
                    let r = Rect::new(cb, at.row(ib), ce - cb, ie - ib);
                    let block = |n: &str| Slice::of(n, None, vec![(ib, ie), (cb, ce)]);
                    out.rcfg(r);
                    out.slice("hw", r, block("hw"), true, pad("lm_head.weight"));
                    let emb = pad("model.embed_tokens.weight");
                    out.slice("hemb", r, block("hemb"), true, emb);
                    let hg = Slice {
                        across: r.w,
                        ..Slice::of("hg", None, vec![(ib, ie)])
                    };
                    out.slice("hg", r, hg, false, Vec::new());
                }
            }
        }
        out.rcfg(Rect::new(0, at.ctl(), self.cols, 1));
        Ok(())
    }
}

/// The token loop over the band: the control row's max-reduce into its middle (two-level by
/// `reduce.token_loop`), then per dims-row i the bus west to column 0 ([`EMB_WEST`]: [`BUS_TX`]
/// -> [`BUS`]), east along the dims-row and up unit 0's MoE column i ([`BAND_V1`]), west into
/// its CX row i ([`BAND_W2`]).
pub struct BandLoop;

impl Block for BandLoop {
    fn wire(&self, plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        let at = At::of(floor);
        let cols = floor.head.rect.w;
        w.link(token_loop::loop_reduce(
            plan,
            (0..cols).map(|j| Coord::new(j, at.ctl())).collect(),
            (cols / 2) as usize,
        ));
        let u0 = floor.units.first().ok_or("band: no units")?;
        if u0.rect.y + u0.rect.h != at.y0 {
            return Err("band: unit 0 sits on the band".into());
        }
        if cols < 2 {
            return Err("band: a band of one column".into());
        }
        for i in 0..floor.head.rect.h - 1 {
            let yb = at.row(i);
            w.path(Path {
                pts: (0..cols).rev().map(|j| Coord::new(j, yb)).collect(),
                class: EMB_WEST,
                send: BUS_TX,
                recv: BUS,
            });
            let ci = u0.cx_x() + 1 + i;
            let yu = u0.row_y(i);
            if ci >= cols {
                return Err("band: the return column past the band".into());
            }
            let mut pts: Vec<Coord> = (0..=ci).map(|x| Coord::new(x, yb)).collect();
            pts.extend((yu..yb).rev().map(|yy| Coord::new(ci, yy)));
            w.path(Path {
                pts,
                class: BAND_V1,
                send: V1_TX,
                recv: V1_RX,
            });
            w.path(Path {
                pts: (u0.cx_x()..=ci).rev().map(|x| Coord::new(x, yu)).collect(),
                class: BAND_W2,
                send: W2_TX,
                recv: token_loop::EMB_IN,
            });
        }
        Ok(())
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        let mut c = token_loop::COLOURS.to_vec();
        c.extend([(EMB_WEST, BUS_COLOUR), (BAND_V1, 8), (BAND_W2, 13)]);
        c
    }
}
