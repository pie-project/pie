//! The rcfg ABI the role-split kernels share: every family's weight, CX, CY, head and head
//! control kernel reads the same leading words (its route table, its reductions' and segment's
//! queues, the spare colour) and appends its own. A family's [`super::super::emit::rcfg::RcfgAbi`]
//! builds a [`Pe`] and calls these with what is its own (role words, flags, tails).

use crate::spatial::split::fabric::{ChainMember, Colour, Endpoint, Palette, PeIo, RootQueues};
use crate::spatial::split::floor::Coord;
use crate::spatial::split::program::Program;
use crate::spatial::split::wiring::{Class, Slot};

type R<T> = Result<T, String>;

/// The colours the kernels' compile-time routes take; a spare colour avoids them where it can.
pub const CT6: [u8; 6] = [13, 14, 17, 18, 19, 20];
/// CX's list.
pub const CT5: [u8; 5] = [13, 14, 17, 18, 19];

pub fn word(c: Colour) -> u32 {
    u32::from(c.0)
}

/// `a`, unless absent or colour 0, else `b` (the tools' `a or b`).
pub fn or(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match a {
        Some(v) if v != 0 => Some(v),
        _ => b,
    }
}

/// One PE as its rcfg sees it: its bindings, the family's palette, and (gpt-oss's band) a colour
/// it never parks on and counts as final.
pub struct Pe<'a> {
    pub prog: &'a Program,
    pub at: Coord,
    pub io: &'a PeIo,
    pub pal: &'a Palette,
    pub band: Option<u8>,
}

/// A segment's queue words at a PE: role code (5: none) and (in, second in, out, multicast).
pub struct Seg {
    pub role: u32,
    pub recv: Option<u32>,
    pub recv2: Option<u32>,
    pub send: Option<u32>,
    pub cast: Option<u32>,
}

impl Seg {
    const NONE: Seg = Seg {
        role: 5,
        recv: None,
        recv2: None,
        send: None,
        cast: None,
    };
}

impl<'a> Pe<'a> {
    pub fn new(prog: &'a Program, pal: &'a Palette, at: Coord) -> Pe<'a> {
        Pe {
            prog,
            at,
            io: prog.fabric.io(at),
            pal,
            band: None,
        }
    }

    /// The palette's colour of `k`.
    pub fn c(&self, k: Class) -> R<u32> {
        self.pal.get(k).map(word)
    }

    pub fn member(&self, s: Slot) -> R<ChainMember> {
        match self.io.slots.get(&s) {
            Some(Endpoint::Member(m)) => Ok(*m),
            e => Err(format!("{:?}: {s:?} is {e:?}, not a chain member", self.at)),
        }
    }

    pub fn root(&self, s: Slot) -> R<RootQueues> {
        match self.io.slots.get(&s) {
            Some(Endpoint::Root(r)) => Ok(*r),
            e => Err(format!("{:?}: {s:?} is {e:?}, not a chain root", self.at)),
        }
    }

    /// A root's two halves, both required.
    pub fn halves(&self, s: Slot) -> R<(u32, u32)> {
        let r = self.root(s)?;
        let half = |h: Option<Colour>, n| {
            h.map(word)
                .ok_or_else(|| format!("{:?}: {s:?} root without a {n} half", self.at))
        };
        Ok((half(r.lo, "low")?, half(r.hi, "high")?))
    }

    /// The colour a path / multicast endpoint left in `s`.
    pub fn colour(&self, s: Slot) -> Option<u32> {
        match self.io.slots.get(&s)? {
            Endpoint::Send(c) | Endpoint::Recv(c) => Some(word(*c)),
            _ => None,
        }
    }

    /// A relay's turn (from, to) in `s`.
    pub fn turn(&self, s: Slot) -> Option<(u32, u32)> {
        match self.io.slots.get(&s)? {
            Endpoint::Turn(a, b) => Some((word(*a), word(*b))),
            _ => None,
        }
    }

    /// The segment in `s`, its multicast on `cast`.
    pub fn seg(&self, s: Slot, cast: u32) -> Seg {
        match self.io.slots.get(&s) {
            Some(Endpoint::Member(m)) => Seg {
                role: m.role,
                recv: m.recv1.map(word),
                recv2: None,
                send: Some(word(m.send)),
                cast: Some(cast),
            },
            Some(Endpoint::Root(q)) => {
                let (lo, hi) = (q.lo.map(word), q.hi.map(word));
                let two = lo.is_some() && hi.is_some();
                Seg {
                    role: if two { 4 } else { 6 },
                    recv: or(lo, hi),
                    recv2: if two { hi } else { None },
                    send: Some(cast),
                    cast: None,
                }
            }
            _ => Seg::NONE,
        }
    }

    /// The route table, `slots` slots.
    pub fn routes(&self, slots: usize) -> R<Vec<u32>> {
        self.prog.fabric.route_words(self.at, slots)
    }

    /// `n` colours to park unused queues on.
    pub fn parks(&self, n: usize) -> R<Vec<u32>> {
        let avoid: Vec<Colour> = self.band.map(Colour).into_iter().collect();
        Ok(self
            .prog
            .fabric
            .parks_avoiding(self.at, n, &avoid)?
            .into_iter()
            .map(word)
            .collect())
    }

    /// A spare colour for the kernel's temporary route.
    pub fn spare(&self, finals: &[u32], comptime: &[u8]) -> R<u32> {
        let mut f: Vec<Colour> = finals.iter().map(|&c| Colour(c as u8)).collect();
        f.extend(self.band.map(Colour));
        let ct: Vec<Colour> = comptime.iter().map(|&c| Colour(c)).collect();
        self.prog.fabric.spare(self.at, &f, &ct).map(word)
    }
}

/// A weight PE's words: the route table, the row / column reductions' queues (parked where
/// absent), the multicasts, the sends, the reductions' and the scores segment's roles, the
/// family's two words `mid`, the segment's queues, the row, the handoff relays' turns (`hn`
/// words each, from, to), a spare colour.
pub fn weight(
    pe: &Pe,
    slots: usize,
    row: u32,
    mid: [u32; 2],
    hn: u32,
    relays: [Option<(u32, u32)>; 2],
) -> R<Vec<u32>> {
    let rx = pe.member(Slot::RowReduce)?;
    let ry = pe.member(Slot::ColReduce)?;
    let need = [rx.recv1, rx.recv2, ry.recv1, ry.recv2]
        .iter()
        .filter(|v| v.is_none())
        .count();
    let mut pk = pe.parks(need)?.into_iter();
    let mut nx = |v: Option<Colour>| v.map_or_else(|| pk.next().unwrap_or(0), word);
    let q = [
        nx(rx.recv1),
        nx(rx.recv2),
        nx(ry.recv1),
        nx(ry.recv2),
        pe.c(Class::RowCast)?,
        pe.c(Class::ColCast)?,
        word(rx.send),
        word(ry.send),
    ];
    let s = pe.seg(Slot::Segment, pe.c(Class::ScoreCast)?);
    let mut w = pe.routes(slots)?;
    w.extend_from_slice(&q);
    w.extend_from_slice(&[rx.role, ry.role, s.role, mid[0], mid[1]]);
    w.extend_from_slice(&[
        s.recv.unwrap_or(q[5]),
        s.recv2.unwrap_or(q[4]),
        s.send.unwrap_or(q[6]),
        s.cast.unwrap_or(q[4]),
    ]);
    w.push(row);
    for r in relays {
        w.extend_from_slice(&r.map_or([0, 0, 0], |(a, b)| [hn, a, b]));
    }
    w.push(pe.spare(&q[..6], &CT6)?);
    Ok(w)
}

/// What a CX's words take past its row root: its x in / out and embedding queues, the family's
/// flag bits, extra final colours and words after the common ones.
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

/// A CX PE's words: the row root's halves, the loop-back, x in, the multicast, x out, the flags
/// (first unit or no input 1, no output 2, the embedding in 4, the family's), a spare colour, the
/// embedding queue, the family's tail.
pub fn cx(pe: &Pe, x: CxIo, parks: usize, comptime: &[u8]) -> R<Vec<u32>> {
    let pk = pe.parks(parks)?;
    let (lo, hi) = pe.halves(Slot::RowReduce)?;
    let lb = pe.c(Class::Loopback)?;
    let q = [
        lo,
        hi,
        lb,
        x.inn.unwrap_or(pk[0]),
        lb,
        pe.c(Class::RowCast)?,
        x.out.unwrap_or(pk[1]),
    ];
    let fl = u32::from(x.first || x.inn.is_none())
        | if x.out.is_none() { 2 } else { 0 }
        | if x.emb.is_some() { 4 } else { 0 }
        | x.flags;
    let q6 = x.emb.unwrap_or(pk[2]);
    let mut fin = vec![q[0], q[1], q[2], q[3], q6];
    fin.extend_from_slice(&x.finals);
    let mut w = pe.routes(12)?;
    w.extend_from_slice(&q);
    w.extend_from_slice(&[fl, pe.spare(&fin, comptime)?, q6]);
    w.extend(x.tail);
    Ok(w)
}

/// A CY PE's words: the column root's halves, the loop-back, the segment `seg` (slot, multicast
/// class; none: role 5, its queues parked), the multicast, a spare colour.
pub fn cy(pe: &Pe, seg: Option<(Slot, Class)>) -> R<Vec<u32>> {
    let pk = pe.parks(4)?;
    let (lo, hi) = pe.halves(Slot::ColReduce)?;
    let lb = pe.c(Class::Loopback)?;
    let s = match seg {
        Some((slot, cast)) => pe.seg(slot, pe.c(cast)?),
        None => Seg::NONE,
    };
    let q = [
        lo,
        hi,
        lb,
        s.recv.unwrap_or(pk[0]),
        lb,
        pe.c(Class::ColCast)?,
        s.send.unwrap_or(pk[3]),
    ];
    let tail = [s.role, s.cast.unwrap_or(pk[1]), s.recv2.unwrap_or(pk[2])];
    let fin = [q[0], q[1], q[2], q[3], tail[1], tail[2]];
    let mut w = pe.routes(12)?;
    w.extend_from_slice(&q);
    w.extend_from_slice(&tail);
    w.push(pe.spare(&fin, &CT6)?);
    Ok(w)
}

/// A head PE's place in the head input and the token loop.
pub struct HeadIo {
    /// The head row (the token loop's own row word).
    pub row: u32,
    /// 1: an odd CX row's x, relayed north; 2: the column's source (multicasts it down).
    pub xr: u32,
    /// Whether a source takes a relay from the north (`HN`; Gemma 12B's group roots do not).
    pub relay_north: bool,
    /// The token loop's turns: 1 row 0 takes `VA`, 2 the source's row turns `VS`, 4 the odd row
    /// splits `HS`.
    pub tr: u32,
    pub token_loop: bool,
}

/// A head PE's words: the head row's queues, x in, the relay, the column multicast, the send,
/// the relay's out, the role, the relay role, a spare colour, the loop's row word, the flags, the
/// row, the embedding's queues.
pub fn head(pe: &Pe, h: HeadIo) -> R<Vec<u32>> {
    let pk = pe.parks(7)?;
    let ro = pe.member(Slot::HeadRowReduce)?;
    let (hx, hn, hm) = (
        pe.c(Class::HandoffAcross)?,
        if h.relay_north {
            pe.c(Class::HeadRelay)?
        } else {
            0
        },
        pe.c(Class::HeadColCast)?,
    );
    let xr = h.xr;
    let mut q = [
        ro.recv1.map_or(pk[0], word),
        ro.recv2.map_or(pk[1], word),
        if xr == 1 || xr == 2 { hx } else { pk[2] },
        if xr == 2 && h.relay_north { hn } else { pk[3] },
        if xr == 0 || xr == 1 { hm } else { pk[4] },
        word(ro.send),
        match xr {
            1 => hn,
            2 => hm,
            _ => pk[0],
        },
    ];
    let emb = |k| if h.token_loop { pe.c(k) } else { Ok(pk[6]) };
    let (va, vs, he, hs) = (
        emb(Class::EmbUp)?,
        emb(Class::EmbDown)?,
        emb(Class::EmbWest)?,
        emb(Class::EmbSplit)?,
    );
    if h.tr & 1 != 0 && h.tr & 2 == 0 {
        q[3] = va;
    }
    if h.tr & 4 != 0 {
        if xr != 1 {
            return Err(format!("{:?}: HS turn off relay A", pe.at));
        }
        q[3] = hs;
    }
    if h.tr & 2 != 0 {
        if xr != 2 {
            return Err(format!("{:?}: VS turn off a column source", pe.at));
        }
        q[4] = if h.row == 0 { va } else { vs };
    }
    let q7 = if h.token_loop {
        pe.c(Class::HeadRowWord)?
    } else {
        pk[5]
    };
    let fin = [q[0], q[1], q[2], q[3], q[4], q7];
    let mut w = pe.routes(12)?;
    w.extend_from_slice(&q);
    w.extend_from_slice(&[
        if ro.role == 3 { 1 } else { ro.role },
        xr,
        pe.spare(&fin, &CT6)?,
        q7,
        u32::from(h.token_loop) | (h.tr << 1),
        h.row,
        va,
        vs,
        he,
        hs,
    ]);
    Ok(w)
}

/// A head control PE's words: the head row root's halves, the loop-back, its vocabulary offset
/// `vocab`, a spare colour, the token loop's queues (its member, or root: role 4 | its halves 8 /
/// 16), the winner multicast, the row word, the role, whether the loop runs.
pub fn hctl(pe: &Pe, vocab: u32, token_loop: bool) -> R<Vec<u32>> {
    let (lo, hi) = pe.halves(Slot::HeadRowReduce)?;
    let pk = pe.parks(4)?;
    let lb = pe.c(Class::Loopback)?;
    let (role, q5, q6, q7, oq3, oq4) = if token_loop {
        let (wm, hr3) = (pe.c(Class::LoopCast)?, pe.c(Class::HeadRowWord)?);
        match pe.io.slots.get(&Slot::LoopReduce) {
            Some(Endpoint::Root(r)) => {
                let (l, h) = (r.lo.map(word), r.hi.map(word));
                let role = 4 | if l.is_some() { 8 } else { 0 } | if h.is_some() { 16 } else { 0 };
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
    let fin = [lo, hi, lb, q5, q6, q7];
    let mut w = pe.routes(12)?;
    w.extend_from_slice(&[lo, hi, lb, lb, vocab, pe.spare(&fin, &CT6)?]);
    w.extend_from_slice(&[q5, q6, q7, oq3, oq4, role, u32::from(token_loop)]);
    Ok(w)
}
