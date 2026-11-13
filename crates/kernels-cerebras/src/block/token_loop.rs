//! The on-wafer token loop: the head control column max-reduces the logits' argmax into its
//! middle row (the root: ring, force, nforce), which multicasts the winner; each head column
//! carries the winner's embedding row back up, down onto unit 0's two CX rows of that column
//! and west into unit 0's CX.

use crate::floor::{Coord, Floor};
use crate::plan::Plan;
use crate::wiring::{
    Chain, Class, Link, NoteKey, PASS, Path, Slot, Tree, Wiring, level2, three_level,
};

use super::Block;

type R<T> = Result<T, String>;

/// The max-reduce's level-1 halves, level 2, the winner multicast.
pub const LOOP_L1: [Class; 2] = [Class::new("WA"), Class::new("WB")];
pub const LOOP_L2: [Class; 2] = Class::pair("WC2");
pub const LOOP_CAST: Class = Class::new("WM");
/// The embedding's return: up the head column, down to the even CX row, the split onto the odd
/// row, west into unit 0's CX.
pub const EMB_UP: Class = Class::new("VA");
pub const EMB_DOWN: Class = Class::new("VS");
pub const EMB_SPLIT: Class = Class::new("HS");
pub const EMB_WEST: Class = Class::new("HE");

/// The max-reduce a head control PE is a member or root of.
pub const LOOP_REDUCE: Slot = Slot::reduce("loop", false);
/// The embedding arriving at unit 0's CX.
pub const EMB_IN: Slot = Slot::io("emb_in");

/// A head PE's turns of the embedding's return ([`token_loop`]): [`TURN_TOP`] on the head's top
/// row, [`TURN_EVEN`] at the even CX row, [`TURN_ODD`] at the odd one.
pub const TURN: NoteKey = NoteKey("loop_turn");
pub const TURN_TOP: u32 = 1;
pub const TURN_EVEN: u32 = 2;
pub const TURN_ODD: u32 = 4;

pub const COLOURS: &[(Class, u8)] = &[
    (LOOP_L1[0], 10),
    (LOOP_L1[1], 11),
    (LOOP_L2[0], 13),
    (LOOP_L2[1], 14),
    (LOOP_CAST, 15),
    (EMB_UP, 5),
    (EMB_DOWN, 8),
    (EMB_SPLIT, 9),
    (EMB_WEST, 17),
];

/// The token loop over a head with a control column.
pub struct TokenLoop;

impl Block for TokenLoop {
    fn wire(&self, plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        token_loop(plan, floor, w)
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        COLOURS.to_vec()
    }
}

/// The max-reduce over `pts` into `pts[root]` ([`LOOP_REDUCE`]): a two-level chain in groups of
/// `reduce.token_loop`, or with `reduce.token_loop3` (`G3W`) a three-level tree
/// ([`three_level`]) on the chain's four colours. The merge (the larger value, ties to the
/// lower id) is a total order, so either gives the same winner.
pub fn loop_reduce(plan: &Plan, pts: Vec<Coord>, root: usize) -> Link {
    let r = plan.partition.reduce;
    if r.token_loop3 > 0 && r.token_loop > 0 {
        let parent = three_level(
            pts.len(),
            root,
            r.token_loop as usize,
            r.token_loop3 as usize,
        );
        Link::Tree(Tree {
            pts,
            root,
            parent,
            colours: [LOOP_L1[0], LOOP_L1[1], LOOP_L2[0], LOOP_L2[1]],
            cast: LOOP_CAST,
            slot: LOOP_REDUCE,
        })
    } else {
        Link::Chain(Chain {
            pts,
            root,
            l1: LOOP_L1,
            l2: level2(LOOP_L2, r.token_loop),
            cast: LOOP_CAST,
            loopback: false,
            slot: LOOP_REDUCE,
        })
    }
}

/// The head control PEs max-reduce into the middle row ([`loop_reduce`];
/// [`LOOP_REDUCE`]). Then per head column `c`, with `te` / `to` unit 0's CX rows `2c` / `2c + 1`
/// (adjacent rows inside the head's rows): up the column to the head's top row ([`EMB_UP`]),
/// down to `te` ([`EMB_DOWN`], when `te` is below the top), across to `to` ([`EMB_SPLIT`]), and
/// west from both rows into unit 0's CX ([`EMB_WEST`], [`EMB_IN`] there; the head ends bind
/// nothing). Notes [`TURN`] at the head PEs.
pub fn token_loop(plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
    let hb = &floor.head;
    let Some(hcol) = hb.ctl_x() else {
        return Err("wiring: the token loop wants a head control column".into());
    };
    let (y0, hr) = (hb.rect.y, hb.rect.h);
    w.link(loop_reduce(
        plan,
        (y0..y0 + hr).map(|y| Coord::new(hcol, y)).collect(),
        (hr / 2) as usize,
    ));
    let u0 = floor.units.first().ok_or("wiring: no units")?;
    for c in 0..hb.cols() {
        let x = hb.col_x(c);
        let (te, to) = (u0.row_y(2 * c), u0.row_y(2 * c + 1));
        if to != te + 1 || te < y0 || to >= y0 + hr {
            return Err(format!("head column {c}: CX rows {te} / {to}"));
        }
        let pass = |pts: Vec<Coord>, class| Path {
            pts,
            class,
            send: PASS,
            recv: PASS,
        };
        w.path(pass(
            (y0..y0 + hr).rev().map(|y| Coord::new(x, y)).collect(),
            EMB_UP,
        ));
        if te > y0 {
            w.path(pass(
                (y0..=te).map(|y| Coord::new(x, y)).collect(),
                EMB_DOWN,
            ));
        }
        w.path(pass(vec![Coord::new(x, te), Coord::new(x, to)], EMB_SPLIT));
        for y in [te, to] {
            w.path(Path {
                pts: (u0.cx_x()..=x).rev().map(|xx| Coord::new(xx, y)).collect(),
                class: EMB_WEST,
                send: PASS,
                recv: EMB_IN,
            });
        }
        w.note(Coord::new(x, y0), TURN, TURN_TOP);
        w.note(Coord::new(x, te), TURN, TURN_EVEN);
        w.note(Coord::new(x, to), TURN, TURN_ODD);
    }
    Ok(())
}
