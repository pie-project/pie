//! The rcfg words the wafer program kernels share: every layout's weight, CX, CY, head and head
//! control kernel reads the same leading words (its route table, its reductions' and segment's
//! queues, the spare colour) and appends its own. A kernel's writer calls these with what is its
//! own (the weight kernel's two words, CX's flags and tail, ...).
//!
//! Queue words a PE does not bind are parked ([`Pe::parks`]) on colours that never reach its
//! ramp; a spare colour ([`Pe::spare`]) avoids the kernel's final and compile-time colours.

use crate::abi::{Kernel, params, words};
use crate::block::head::{
    HEAD_COL_CAST, HEAD_RELAY, HEAD_ROW_REDUCE, HEAD_ROW_WORD, ROLE_RELAY, ROLE_SOURCE,
};
use crate::block::segments::{SCORE_CAST, SEGMENT};
use crate::block::token_loop::{
    EMB_DOWN, EMB_SPLIT, EMB_UP, EMB_WEST, LOOP_CAST, LOOP_REDUCE, TURN_EVEN, TURN_ODD, TURN_TOP,
};
use crate::block::units::{COL_CAST, COL_REDUCE, HANDOFF_ACROSS, ROW_CAST, ROW_REDUCE};
use crate::fabric::{Colour, Endpoint, member_role};
use crate::wiring::{Class, LOOPBACK, Slot};

use super::{Pe, or, word};

type R<T> = Result<T, String>;

/// The colours the kernels' compile-time routes take; a spare colour avoids them where it can.
pub const CT6: [u8; 6] = [13, 14, 17, 18, 19, 20];
/// CX's list.
pub const CT5: [u8; 5] = [13, 14, 17, 18, 19];

/// A segment's role code (`seg_role`): a member's chain role, or one of these.
pub const SEG_ROOT_BOTH: u32 = 4;
pub const SEG_NONE: u32 = 5;
pub const SEG_ROOT_ONE: u32 = 6;

/// CX's flag bits: it takes no x (the first unit), sends none (the last), takes the embedding.
pub const CX_NO_INPUT: u32 = 1;
pub const CX_NO_OUTPUT: u32 = 2;
pub const CX_EMB: u32 = 4;

/// The head control's token-loop role: the root, with a low / high half.
pub const LOOP_ROOT: u32 = 4;
pub const LOOP_ROOT_LO: u32 = 8;
pub const LOOP_ROOT_HI: u32 = 16;

/// A segment's queue words at a PE: role code and (in, second in, out, multicast).
pub struct Seg {
    pub role: u32,
    pub recv: Option<u32>,
    pub recv2: Option<u32>,
    pub send: Option<u32>,
    pub cast: Option<u32>,
}

impl Seg {
    pub const NONE: Seg = Seg {
        role: SEG_NONE,
        recv: None,
        recv2: None,
        send: None,
        cast: None,
    };
}

/// The segment in `s`, its multicast on `cast` (looked up only where the PE is in one).
pub fn seg(pe: &Pe, s: Slot, cast: Class) -> R<Seg> {
    Ok(match pe.io.slots.get(&s) {
        Some(Endpoint::Member(m)) => Seg {
            role: m.role,
            recv: m.recv1.map(word),
            recv2: None,
            send: Some(word(m.send)),
            cast: Some(pe.c(cast)?),
        },
        Some(Endpoint::Root(q)) => {
            let (lo, hi) = (q.lo.map(word), q.hi.map(word));
            let two = lo.is_some() && hi.is_some();
            Seg {
                role: if two { SEG_ROOT_BOTH } else { SEG_ROOT_ONE },
                recv: or(lo, hi),
                recv2: if two { hi } else { None },
                send: Some(pe.c(cast)?),
                cast: None,
            }
        }
        _ => Seg::NONE,
    })
}

words! {
    /// A weight PE's shared words.
    pub struct WeightWords {
        /// The row / column reductions' level-1 and level-2 receive queues (parked if absent).
        row_in1, row_in2, col_in1, col_in2,
        /// The multicasts back out, the sends.
        row_cast, col_cast, row_send, col_send,
        /// The reductions' and the scores segment's role codes.
        row_role, col_role, seg_role,
        /// The kernel's own two words.
        own0, own1,
        /// The segment's queues.
        seg_in, seg_in2, seg_send, seg_cast,
        row,
        /// The handoff's turns before / after the PE: words relayed, from, to (0: none).
        pre_n, pre_from, pre_to, post_n, post_from, post_to,
        spare,
    }
}

/// A weight PE's words: the route table (`slots` routes), then [`WeightWords`]: `own` the
/// kernel's two words, `hn` the words a handoff turn relays, `relays` the turns before / after.
pub fn weight(
    pe: &Pe,
    slots: usize,
    row: u32,
    own: [u32; 2],
    hn: u32,
    relays: [Option<(u32, u32)>; 2],
) -> R<Vec<u32>> {
    let rx = pe.member(ROW_REDUCE)?;
    let ry = pe.member(COL_REDUCE)?;
    let need = [rx.recv1, rx.recv2, ry.recv1, ry.recv2]
        .iter()
        .filter(|v| v.is_none())
        .count();
    let mut pk = pe.parks(need)?.into_iter();
    let mut nx = |v: Option<Colour>| v.map_or_else(|| pk.next().unwrap_or(0), word);
    let (row_in1, row_in2, col_in1, col_in2) =
        (nx(rx.recv1), nx(rx.recv2), nx(ry.recv1), nx(ry.recv2));
    let (row_cast, col_cast) = (pe.c(ROW_CAST)?, pe.c(COL_CAST)?);
    let s = seg(pe, SEGMENT, SCORE_CAST)?;
    let turn = |r: Option<(u32, u32)>| r.map_or([0, 0, 0], |(a, b)| [hn, a, b]);
    let ([pre_n, pre_from, pre_to], [post_n, post_from, post_to]) =
        (turn(relays[0]), turn(relays[1]));
    let finals = [row_in1, row_in2, col_in1, col_in2, row_cast, col_cast];
    let ww = WeightWords {
        row_in1,
        row_in2,
        col_in1,
        col_in2,
        row_cast,
        col_cast,
        row_send: word(rx.send),
        col_send: word(ry.send),
        row_role: rx.role,
        col_role: ry.role,
        seg_role: s.role,
        own0: own[0],
        own1: own[1],
        seg_in: s.recv.unwrap_or(col_cast),
        seg_in2: s.recv2.unwrap_or(row_cast),
        seg_send: s.send.unwrap_or(word(rx.send)),
        seg_cast: s.cast.unwrap_or(row_cast),
        row,
        pre_n,
        pre_from,
        pre_to,
        post_n,
        post_from,
        post_to,
        spare: pe.spare(&finals, &CT6)?,
    };
    let mut w = pe.routes(slots)?;
    w.extend(ww.words());
    Ok(w)
}

/// What a CX's words take past its row root: its x in / out and embedding queues, the kernel's
/// own flag bits, extra final colours and words after the shared ones.
#[derive(Default)]
pub struct CxIo {
    pub first: bool,
    pub inn: Option<u32>,
    pub out: Option<u32>,
    pub emb: Option<u32>,
    pub flags: u32,
    pub finals: Vec<u32>,
    pub tail: Vec<u32>,
}

words! {
    /// A CX PE's shared words.
    pub struct CxWords {
        /// The row root's halves, the loop-back.
        row_lo, row_hi, loopback,
        /// x in, the loop-back again (the kernel's second queue on it), the multicast, x out.
        x_in, loopback2, row_cast, x_out,
        /// [`CX_NO_INPUT`] | [`CX_NO_OUTPUT`] | [`CX_EMB`] | the kernel's own bits.
        flags,
        spare,
        emb_in,
    }
}

/// A CX PE's words: the route table, [`CxWords`], the kernel's tail. `parks` queues are parked
/// (x in, x out, the embedding first); `comptime` the kernel's compile-time colours.
pub fn cx(pe: &Pe, x: CxIo, parks: usize, comptime: &[u8]) -> R<Vec<u32>> {
    let pk = pe.parks(parks)?;
    let (row_lo, row_hi) = pe.halves(ROW_REDUCE)?;
    let lb = pe.c(LOOPBACK)?;
    let x_in = x.inn.unwrap_or(pk[0]);
    let emb_in = x.emb.unwrap_or(pk[2]);
    let flags = if x.first || x.inn.is_none() {
        CX_NO_INPUT
    } else {
        0
    } | if x.out.is_none() { CX_NO_OUTPUT } else { 0 }
        | if x.emb.is_some() { CX_EMB } else { 0 }
        | x.flags;
    let mut finals = vec![row_lo, row_hi, lb, x_in, emb_in];
    finals.extend_from_slice(&x.finals);
    let cw = CxWords {
        row_lo,
        row_hi,
        loopback: lb,
        x_in,
        loopback2: lb,
        row_cast: pe.c(ROW_CAST)?,
        x_out: x.out.unwrap_or(pk[1]),
        flags,
        spare: pe.spare(&finals, comptime)?,
        emb_in,
    };
    let mut w = pe.routes(12)?;
    w.extend(cw.words());
    w.extend(x.tail);
    Ok(w)
}

words! {
    /// A CY PE's words.
    pub struct CyWords {
        /// The column root's halves, the loop-back.
        col_lo, col_hi, loopback,
        /// The segment's in, the loop-back again, the column multicast, the segment's out.
        seg_in, loopback2, col_cast, seg_send,
        /// The segment's role, multicast, second in.
        seg_role, seg_cast, seg_in2,
        spare,
    }
}

/// A CY PE's words: the route table, [`CyWords`]; `seg` the segment it may be in (slot,
/// multicast class; none: [`SEG_NONE`], its queues parked).
pub fn cy(pe: &Pe, seg_of: Option<(Slot, Class)>) -> R<Vec<u32>> {
    let pk = pe.parks(4)?;
    let (col_lo, col_hi) = pe.halves(COL_REDUCE)?;
    let lb = pe.c(LOOPBACK)?;
    let s = match seg_of {
        Some((slot, cast)) => seg(pe, slot, cast)?,
        None => Seg::NONE,
    };
    let (seg_in, seg_cast, seg_in2) = (
        s.recv.unwrap_or(pk[0]),
        s.cast.unwrap_or(pk[1]),
        s.recv2.unwrap_or(pk[2]),
    );
    let finals = [col_lo, col_hi, lb, seg_in, seg_cast, seg_in2];
    let cw = CyWords {
        col_lo,
        col_hi,
        loopback: lb,
        seg_in,
        loopback2: lb,
        col_cast: pe.c(COL_CAST)?,
        seg_send: s.send.unwrap_or(pk[3]),
        seg_role: s.role,
        seg_cast,
        seg_in2,
        spare: pe.spare(&finals, &CT6)?,
    };
    let mut w = pe.routes(12)?;
    w.extend(cw.words());
    Ok(w)
}

/// A head PE's place in the head input and the token loop.
pub struct HeadIo {
    /// The head row (the token loop's own row word).
    pub row: u32,
    /// Its head-input role: [`ROLE_RELAY`], [`ROLE_SOURCE`] or 0.
    pub role: u32,
    /// Whether a source takes a relay from the north (`HN`; Gemma 12B's group roots do not).
    pub relay_north: bool,
    /// Its turns of the embedding's return ([`TURN_TOP`] | [`TURN_EVEN`] | [`TURN_ODD`]).
    pub turns: u32,
    pub token_loop: bool,
}

words! {
    /// A head PE's words.
    pub struct HeadWords {
        /// The head row's level-1 / level-2 receive queues.
        row_in1, row_in2,
        /// x in (a CX row's), the relay in (from the north) or the embedding's up / split
        /// turn, the column multicast in or the embedding's down turn.
        x_in, relay_in, cast_in,
        /// The head row's send, the relay's out (north, or the column multicast).
        row_send, relay_out,
        /// The head row's role (a last group's head runs as a mid member), the input role.
        row_role, input_role,
        spare,
        /// The token loop's row word.
        row_word,
        /// token loop (bit 0) | turns << 1.
        flags,
        row,
        /// The embedding's return queues.
        emb_up, emb_down, emb_west, emb_split,
    }
}

/// A head PE's words: the route table, [`HeadWords`].
pub fn head(pe: &Pe, h: HeadIo) -> R<Vec<u32>> {
    let pk = pe.parks(7)?;
    let ro = pe.member(HEAD_ROW_REDUCE)?;
    let hx = pe.c(HANDOFF_ACROSS)?;
    let hn = if h.relay_north { pe.c(HEAD_RELAY)? } else { 0 };
    let hm = pe.c(HEAD_COL_CAST)?;
    let (relay, source) = (h.role == ROLE_RELAY, h.role == ROLE_SOURCE);
    let emb = |k| if h.token_loop { pe.c(k) } else { Ok(pk[6]) };
    let (emb_up, emb_down, emb_west, emb_split) = (
        emb(EMB_UP)?,
        emb(EMB_DOWN)?,
        emb(EMB_WEST)?,
        emb(EMB_SPLIT)?,
    );
    let mut relay_in = if source && h.relay_north { hn } else { pk[3] };
    let mut cast_in = if h.role == 0 || relay { hm } else { pk[4] };
    if h.turns & TURN_TOP != 0 && h.turns & TURN_EVEN == 0 {
        relay_in = emb_up;
    }
    if h.turns & TURN_ODD != 0 {
        if !relay {
            return Err(format!("{:?}: HS turn off a relay", pe.at));
        }
        relay_in = emb_split;
    }
    if h.turns & TURN_EVEN != 0 {
        if !source {
            return Err(format!("{:?}: VS turn off a column source", pe.at));
        }
        cast_in = if h.row == 0 { emb_up } else { emb_down };
    }
    let row_word = if h.token_loop {
        pe.c(HEAD_ROW_WORD)?
    } else {
        pk[5]
    };
    let (row_in1, row_in2) = (ro.recv1.map_or(pk[0], word), ro.recv2.map_or(pk[1], word));
    let x_in = if relay || source { hx } else { pk[2] };
    let finals = [row_in1, row_in2, x_in, relay_in, cast_in, row_word];
    let hw = HeadWords {
        row_in1,
        row_in2,
        x_in,
        relay_in,
        cast_in,
        row_send: word(ro.send),
        relay_out: if relay {
            hn
        } else if source {
            hm
        } else {
            pk[0]
        },
        row_role: if ro.role == member_role::LAST_GROUP_HEAD {
            member_role::MID
        } else {
            ro.role
        },
        input_role: h.role,
        spare: pe.spare(&finals, &CT6)?,
        row_word,
        flags: u32::from(h.token_loop) | (h.turns << 1),
        row: h.row,
        emb_up,
        emb_down,
        emb_west,
        emb_split,
    };
    let mut w = pe.routes(12)?;
    w.extend(hw.words());
    Ok(w)
}

words! {
    /// A head control PE's words.
    pub struct HctlWords {
        /// The head row root's halves, the loop-back (twice).
        row_lo, row_hi, loopback, loopback2,
        /// Its vocabulary offset.
        vocab,
        spare,
        /// The token loop's queues: a member's level-1 / level-2 in, the winner multicast in,
        /// its send; a root's halves (parked if absent), a parked queue, the multicast out.
        loop_a, loop_b, loop_c, loop_send,
        row_word,
        /// A member's chain role, or [`LOOP_ROOT`] | [`LOOP_ROOT_LO`] | [`LOOP_ROOT_HI`].
        loop_role,
        token_loop,
    }
}

/// A head control PE's words: the route table, [`HctlWords`].
pub fn hctl(pe: &Pe, vocab: u32, token_loop: bool) -> R<Vec<u32>> {
    let (row_lo, row_hi) = pe.halves(HEAD_ROW_REDUCE)?;
    let pk = pe.parks(4)?;
    let lb = pe.c(LOOPBACK)?;
    let (loop_role, loop_a, loop_b, loop_c, loop_send, row_word) = if token_loop {
        let (wm, hr3) = (pe.c(LOOP_CAST)?, pe.c(HEAD_ROW_WORD)?);
        match pe.io.slots.get(&LOOP_REDUCE) {
            Some(Endpoint::Root(r)) => {
                let (l, h) = (r.lo.map(word), r.hi.map(word));
                let role = LOOP_ROOT
                    | if l.is_some() { LOOP_ROOT_LO } else { 0 }
                    | if h.is_some() { LOOP_ROOT_HI } else { 0 };
                (
                    role,
                    or(l, Some(pk[0])).unwrap_or(pk[0]),
                    or(h, Some(pk[1])).unwrap_or(pk[1]),
                    pk[2],
                    wm,
                    hr3,
                )
            }
            Some(Endpoint::Member(m)) => (
                m.role,
                m.recv1.map_or(pk[0], word),
                m.recv2.map_or(pk[1], word),
                wm,
                word(m.send),
                hr3,
            ),
            e => return Err(format!("{:?}: token loop slot {e:?}", pe.at)),
        }
    } else {
        (0, pk[0], pk[1], pk[2], pk[3], pk[3])
    };
    let finals = [row_lo, row_hi, lb, loop_a, loop_b, loop_c];
    let hw = HctlWords {
        row_lo,
        row_hi,
        loopback: lb,
        loopback2: lb,
        vocab,
        spare: pe.spare(&finals, &CT6)?,
        loop_a,
        loop_b,
        loop_c,
        loop_send,
        row_word,
        loop_role,
        token_loop: u32::from(token_loop),
    };
    let mut w = pe.routes(12)?;
    w.extend(hw.words());
    Ok(w)
}

params! {
    /// A kernel with no params.
    pub struct NoParams {}
}

/// `relay.csl`: a gap slot PE's routes only (it relays the vertical handoff).
pub const RELAY: Kernel = Kernel {
    source: "relay.csl",
    owns: &["rcfg"],
    params: NoParams::NAMES,
    route_slots: 4,
    rcfg: Some(|pe| pe.routes(4)),
};
