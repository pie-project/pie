//! A static check of a GEMM-regime program's fabric protocol (`emit/gemm_pe.csl`'s attention
//! and `gseg.csl`'s reductions): every PE's sends and reads, colour by colour, run as a process
//! network over the array. **Kahn mode** (unbounded FIFOs) checks counts, orders and queue
//! bindings independent of timing; **bounded mode** (`cap` messages a PE and colour) finds a
//! deterministic schedule's deadlock; **ramp mode** (`hol`) finds a wavelet reaching a ramp on
//! a colour no queue is bound to, which holds the ramp, under slow PEs (`lag`) and `drift`.
//!
//! Colours (`gpe.csl`): the column line `KF` (south) / `KB` (north) on iq6, the reductions'
//! chain `A0` / `A1` on iq4 (a PE sends on its parity's, reads the other), their broadcast `RB`
//! (the row's) / `BB` (the band's) on iq5 (`gseg.setmode` rebinds it), a GEMM's row line `XF`
//! (east) / `XB` (west) on iq2 (rebound to `XB` at the PE's own piece, to `XF` after the op).

use std::collections::VecDeque;

/// The colours a message travels on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Colour {
    Kf,
    Kb,
    A0,
    A1,
    Rb,
    Bb,
    Xf,
    Xb,
}

const NC: usize = 8;
const COLOURS: [Colour; NC] = [
    Colour::Kf,
    Colour::Kb,
    Colour::A0,
    Colour::A1,
    Colour::Rb,
    Colour::Bb,
    Colour::Xf,
    Colour::Xb,
];

fn ci(c: Colour) -> usize {
    c as usize
}

/// An op of the program, as the reductions and the column line see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// The joint attention over the request's rows, `kbm` members a key block.
    Attend,
    /// A region's sums over the row (`seg.all_reduce`), `n` a row of the PE's.
    RowReduce(u32),
    /// A region's sums over the head's band (`seg.head_reduce`), `n` a row of the PE's.
    HeadReduce(u32),
    /// A GEMM over `k` input words split over the row's PEs (`plen`; a PE of no words sends
    /// none, but still rebinds iq2 at its turn), run by the rows of the classes in `mask`.
    Gemm { k: u32, mask: u8 },
    /// A region op: `row` all-reduces and `head` head-reduces, by the rows of `mask`'s classes.
    Region { mask: u8, row: u32, head: u32 },
}

/// The array and the attention's shape.
#[derive(Clone, Debug)]
pub struct Shape {
    pub gx: u32,
    /// The request's rows of PEs (one request).
    pub rq: u32,
    pub band: u32,
    pub kbm: u32,
    pub ops: Vec<Op>,
    /// Bounded mode: a delivery to a ramp whose PE has no queue on the colour waits.
    pub bind_blocks: bool,
    /// A seeded shuffle of the schedule (0: in order).
    pub seed: u64,
    /// A protocol bug planted (the check's own test): 1 a member keeps iq6 on `KF`, 2 the
    /// band's reductions keep iq5 on `RB`.
    pub bug: u8,
    /// Rows below `split` are of class 0 (mask bit 1), the rest of class 1 (bit 2).
    pub split: u32,
    /// Ramp mode: a colour no queue is bound to holds the PE's ramp.
    pub hol: bool,
    /// PEs that run only when no other PE can (the slow ones).
    pub lag: Vec<(u32, u32)>,
    /// Chunks the other PEs may start ahead of the slowest lagging PE (`None`: any).
    pub drift: Option<u64>,
    /// The row line's rule: iq2 rebound right after the last piece on the old colour is read
    /// (before its weights), a PE of no words too; else at its turn and after the op.
    pub v13: bool,
    /// The row line's `XF` and `XB` on queues of their own, bound for good (no rebinding).
    pub x_queues: bool,
    /// A GEMM of fewer input words than PEs gives them to the westmost PEs (none of no words
    /// west of one with), not spread over the row.
    pub packed: bool,
    /// The column line's rule (as emitted): iq6 rebound to `KF` once the last row's piece is
    /// read, not left on `KB` to the next attention.
    pub v17: bool,
    /// The column line as emitted: the attentions alternate their members' order (forward,
    /// then from the last row), each starting on the colour the last ended on.
    pub alt: bool,
    /// Ramp mode, the tape bounded: a column's weight stream moves a word on only once every PE
    /// of the column has room for it (`n` words unread) and none's ramp is held (`None`: the
    /// tape always there).
    pub tape_cap: Option<u64>,
}

impl Shape {
    /// A GEMM-regime program's ops as the check sees them, on a `gx x rq` array: its GEMMs'
    /// input words scaled from the program's row, its classes' rows split as its rows are, its
    /// host ins and outs a tape header each (ramp mode, as emitted).
    #[must_use]
    pub fn of(p: &crate::compile::stream::gemm::GemmProgram, gx: u32, rq: u32) -> Shape {
        use crate::compile::stream::gemm::GOp;
        // (the classes a check of two tells apart: class 0, then every other)
        let bits = |m: u16| u8::from(m & 1 != 0) | (u8::from(m & !1 != 0) << 1);
        let ops =
            p.l.ops
                .iter()
                .map(|o| match o {
                    GOp::Gemm { k, mask, .. } => Op::Gemm {
                        k: (k * gx).div_ceil(p.c),
                        mask: bits(*mask),
                    },
                    GOp::Region(r) => {
                        let g = &p.l.regs[*r];
                        let (row, head) = g
                            .printed
                            .as_ref()
                            .map_or((0, 0), |x| (x.all_reduces, x.head_reduces));
                        Op::Region {
                            mask: bits(g.mask),
                            row,
                            head,
                        }
                    }
                    GOp::Attend { .. } => Op::Attend,
                    GOp::In { .. } | GOp::Out { .. } => Op::Region {
                        mask: 0,
                        row: 0,
                        head: 0,
                    },
                })
                .collect::<Vec<_>>();
        // (the attention's row-wide ordering point at its end, where the PE code has one)
        let ops = if crate::compile::check::hazard::attention_orders_row(
            crate::compile::emit::gemm::GPE_OPS,
        ) {
            ops.into_iter()
                .flat_map(|o| match o {
                    Op::Attend => vec![Op::Attend, Op::RowReduce(1)],
                    o => vec![o],
                })
                .collect()
        } else {
            ops
        };
        let first = p.row_class.iter().take_while(|&&c| c == 0).count() as u32;
        Shape {
            gx,
            rq,
            band: p.band,
            kbm: p.kbm,
            ops,
            bind_blocks: false,
            seed: 0,
            bug: 0,
            split: (first * rq).div_ceil(p.r.max(1)),
            hol: true,
            lag: Vec::new(),
            drift: None,
            v13: false,
            x_queues: true,
            packed: false,
            v17: false,
            alt: true,
            tape_cap: None,
        }
    }
}

impl Shape {
    /// The first of PE `i`'s input words of `k` over the row.
    fn word0(&self, i: u32, k: u32) -> u32 {
        if self.packed && k < self.gx {
            i.min(k)
        } else {
            (u64::from(i) * u64::from(k) / u64::from(self.gx)) as u32
        }
    }
}

/// A message: its sender, and what it is (an op's index and a round of it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Msg {
    from: (u32, u32),
    tag: (u32, u32, u32),
    /// (`cap` 0) its multicast and its destination's place on the path, nearest first.
    cast: u32,
    pos: u32,
}

#[derive(Clone, Copy, Debug)]
enum Act {
    /// Input queue 5 or 6 bound to a colour (iq4's is fixed by parity).
    Bind(u8, Colour),
    /// A message multicast on a colour by its line's routes, tagged.
    Send(Colour, (u32, u32, u32)),
    /// A read off input queue 4, 5 or 6, the message expected (its sender, its tag).
    Recv(u8, (u32, u32), (u32, u32, u32)),
    /// Words off the column's tape (iq3: an op's header, a piece's weights), always coming:
    /// in ramp mode they reach the PE only while its ramp is not held.
    Tape,
}

/// What the check found.
#[derive(Clone, Debug, Default)]
pub struct Found {
    /// Messages delivered and read.
    pub messages: u64,
    /// A read that took another message than the program means: (PE, expected, got).
    pub wrong: Vec<String>,
    /// PEs blocked at the end (none finished): (PE, what it waits on).
    pub blocked: Vec<String>,
    /// Messages delivered that no read took: (PE, colour, how many).
    pub stranded: Vec<String>,
    /// Ramps held at the end by a colour no queue of the PE is bound to.
    pub held: Vec<String>,
    /// The most chunks a PE started ahead of the slowest lagging PE.
    pub lead: u64,
    pub done: bool,
}

/// A PE's program, generated an action at a time.
struct Pe {
    x: u32,
    y: u32,
    /// The op and its chunk running, the chunk's actions and the next of them.
    op: u32,
    chunk: u32,
    acts: Vec<Act>,
    at: usize,
    iq2: Colour,
    iq5: Colour,
    iq6: Colour,
    /// Chunks started.
    started: u64,
    /// iq5's binding as the program's next chunk is generated (`gseg.mode`).
    gen_mode: u32,
    done: bool,
}

impl Shape {
    /// The chunks an op runs in: an attention's key blocks, a reduction's rounds.
    fn chunks(&self, op: Op) -> u32 {
        match op {
            Op::Attend => self.rq.div_ceil(self.kbm),
            Op::RowReduce(n) | Op::HeadReduce(n) => n,
            Op::Gemm { .. } | Op::Region { .. } => 1,
        }
    }

    /// PE `(x, y)`'s actions in chunk `k` of op `oi` (`mode`: iq5's binding, as `gseg` keeps
    /// it).
    fn acts(&self, x: u32, y: u32, oi: u32, k: u32, mode: &mut u32) -> Vec<Act> {
        let mut v = Vec::new();
        if k == 0 && self.hol {
            // (the op's header off the tape)
            v.push(Act::Tape);
        }
        match self.ops[oi as usize] {
            Op::Attend => {
                // (`alt`: every other attention's members in reverse, from the last row; its
                // pieces come on `KB` until a PE's turn, then on `KF`)
                let rev = self.alt
                    && self.ops[..oi as usize]
                        .iter()
                        .filter(|o| matches!(o, Op::Attend))
                        .count()
                        % 2
                        == 1;
                let (pre, post) = if rev {
                    (Colour::Kb, Colour::Kf)
                } else {
                    (Colour::Kf, Colour::Kb)
                };
                if k == 0 {
                    v.push(Act::Bind(6, pre));
                }
                let j = k * self.kbm;
                for i in 0..self.kbm {
                    if j + i >= self.rq {
                        break;
                    }
                    let m = if rev { self.rq - 1 - (j + i) } else { j + i };
                    if m == y {
                        if self.bug != 1 {
                            v.push(Act::Bind(6, post));
                        }
                        if y > 0 {
                            v.push(Act::Send(Colour::Kb, (oi, m, 0)));
                        }
                        if y + 1 < self.rq {
                            v.push(Act::Send(Colour::Kf, (oi, m, 0)));
                        }
                    } else {
                        v.push(Act::Recv(6, (x, m), (oi, m, 0)));
                    }
                    if self.v17 && !self.alt && j + i + 1 == self.rq {
                        v.push(Act::Bind(6, Colour::Kf));
                    }
                }
                for r in 0..2 {
                    self.allred(&mut v, x, y, true, mode, (oi, 1, 2 * k + r));
                }
            }
            Op::RowReduce(_) => self.allred(&mut v, x, y, false, mode, (oi, 2, k)),
            Op::HeadReduce(_) => self.allred(&mut v, x, y, true, mode, (oi, 3, k)),
            Op::Gemm { k: kw, mask } if self.v13 && mask & self.class_bit(y) != 0 => {
                let words = |cp: u32| self.word0(cp + 1, kw) > self.word0(cp, kw);
                let last_f = (0..x).rev().find(|&c| words(c));
                let last_b = (x + 1..self.gx).rev().find(|&c| words(c));
                if last_f.is_none() {
                    v.push(Act::Bind(2, Colour::Xb));
                }
                for cp in 0..self.gx {
                    if cp == x {
                        if words(cp) {
                            if x > 0 {
                                v.push(Act::Send(Colour::Xb, (oi, cp, 0)));
                            }
                            if x + 1 < self.gx {
                                v.push(Act::Send(Colour::Xf, (oi, cp, 0)));
                            }
                        }
                        if last_b.is_none() {
                            v.push(Act::Bind(2, Colour::Xf));
                        }
                        if words(cp) && self.hol {
                            v.push(Act::Tape);
                        }
                        continue;
                    }
                    if !words(cp) {
                        continue;
                    }
                    v.push(Act::Recv(2, (cp, y), (oi, cp, 0)));
                    if Some(cp) == last_f {
                        v.push(Act::Bind(2, Colour::Xb));
                    }
                    if Some(cp) == last_b {
                        v.push(Act::Bind(2, Colour::Xf));
                    }
                    if self.hol {
                        v.push(Act::Tape);
                    }
                }
            }
            Op::Gemm { k: kw, mask } => {
                if mask & self.class_bit(y) != 0 {
                    for cp in 0..self.gx {
                        if cp == x {
                            v.push(Act::Bind(2, Colour::Xb));
                        }
                        if self.word0(cp + 1, kw) == self.word0(cp, kw) {
                            continue;
                        }
                        if cp == x {
                            if x > 0 {
                                v.push(Act::Send(Colour::Xb, (oi, cp, 0)));
                            }
                            if x + 1 < self.gx {
                                v.push(Act::Send(Colour::Xf, (oi, cp, 0)));
                            }
                        } else {
                            v.push(Act::Recv(2, (cp, y), (oi, cp, 0)));
                        }
                        if self.hol {
                            // (the piece's weights)
                            v.push(Act::Tape);
                        }
                    }
                    v.push(Act::Bind(2, Colour::Xf));
                } else if self.hol {
                    // (the op's weights drained: as many words as a member takes, where the
                    // tape is bounded)
                    let n = if self.tape_cap.is_some() {
                        (0..self.gx)
                            .filter(|&cp| self.word0(cp + 1, kw) > self.word0(cp, kw))
                            .count()
                    } else {
                        1
                    };
                    v.extend(std::iter::repeat_n(Act::Tape, n));
                }
            }
            Op::Region { mask, row, head } => {
                if mask & self.class_bit(y) != 0 {
                    for r in 0..row {
                        self.allred(&mut v, x, y, false, mode, (oi, 4, r));
                    }
                    for r in 0..head {
                        self.allred(&mut v, x, y, true, mode, (oi, 5, r));
                    }
                }
            }
        }
        v
    }

    /// Row `y`'s class's mask bit.
    fn class_bit(&self, y: u32) -> u8 {
        if y < self.split { 1 } else { 2 }
    }

    /// `gseg.allred`: the chain east by parity colours, the last PE's broadcast back west.
    fn allred(
        &self,
        v: &mut Vec<Act>,
        x: u32,
        y: u32,
        band: bool,
        mode: &mut u32,
        tag: (u32, u32, u32),
    ) {
        let (lo, hi) = if band {
            let lo = (x / self.band) * self.band;
            (lo, lo + self.band - 1)
        } else {
            (0, self.gx - 1)
        };
        if lo == hi {
            return;
        }
        let m = u32::from(band);
        if *mode != m && self.bug != 2 {
            v.push(Act::Bind(5, if m == 0 { Colour::Rb } else { Colour::Bb }));
            *mode = m;
        }
        let send = if x.is_multiple_of(2) {
            Colour::A0
        } else {
            Colour::A1
        };
        if x == lo {
            v.push(Act::Send(send, tag));
        } else if x < hi {
            v.push(Act::Recv(4, (x - 1, y), tag));
            v.push(Act::Send(send, tag));
        } else {
            v.push(Act::Recv(4, (x - 1, y), tag));
        }
        let bc = if band { Colour::Bb } else { Colour::Rb };
        if x == hi {
            v.push(Act::Send(bc, (tag.0, tag.1 + 10, tag.2)));
        } else {
            v.push(Act::Recv(5, (hi, y), (tag.0, tag.1 + 10, tag.2)));
        }
    }

    /// The PEs a message from `(x, y)` on `c` reaches the ramps of (the layout's routes).
    fn dests(&self, x: u32, y: u32, c: Colour) -> Vec<(u32, u32)> {
        match c {
            Colour::Kf => (y + 1..self.rq).map(|r| (x, r)).collect(),
            Colour::Kb => (0..y).rev().map(|r| (x, r)).collect(),
            // (a chain hop: the next PE east)
            Colour::A0 | Colour::A1 => {
                if x + 1 < self.gx {
                    vec![(x + 1, y)]
                } else {
                    Vec::new()
                }
            }
            Colour::Rb => (0..x).rev().map(|c| (c, y)).collect(),
            Colour::Bb => {
                let lo = (x / self.band) * self.band;
                (lo..x).rev().map(|c| (c, y)).collect()
            }
            Colour::Xf => (x + 1..self.gx).map(|c| (c, y)).collect(),
            Colour::Xb => (0..x).rev().map(|c| (c, y)).collect(),
        }
    }

    /// Runs the network; `cap`: messages a PE's colour holds (`None`: unbounded; `Some(0)`: a
    /// message longer than the fabric's buffers, read by a PE only after the PEs before it on
    /// its path, its sender done once the nearest has read it).
    #[must_use]
    pub fn run(&self, cap: Option<usize>) -> Found {
        let (gx, rq) = (self.gx as usize, self.rq as usize);
        let id = |x: u32, y: u32| (y as usize) * gx + x as usize;
        let n = gx * rq;
        let mut pes: Vec<Pe> = (0..rq as u32)
            .flat_map(|y| (0..gx as u32).map(move |x| (x, y)))
            .map(|(x, y)| Pe {
                x,
                y,
                op: 0,
                chunk: 0,
                acts: Vec::new(),
                at: 0,
                iq2: Colour::Xf,
                iq5: Colour::Rb,
                iq6: Colour::Kf,
                started: 0,
                gen_mode: 0,
                done: false,
            })
            .collect();
        let mut fifo: Vec<[VecDeque<Msg>; NC]> = (0..n).map(|_| Default::default()).collect();
        // (ramp mode: what reached a PE's ramp behind a colour no queue takes, in order)
        let mut ramp: Vec<VecDeque<(Colour, Msg)>> = vec![VecDeque::new(); n];
        let mut found = Found::default();
        let lagging: Vec<bool> = {
            let mut v = vec![false; n];
            for &(x, y) in &self.lag {
                v[id(x, y)] = true;
            }
            v
        };
        let lag_ids: Vec<usize> = (0..n).filter(|&p| lagging[p]).collect();
        // the ready PEs: the others first, the lagging ones when none of the others is
        let mut ready: VecDeque<usize> = (0..n).filter(|&p| !lagging[p]).collect();
        let mut ready_lag: VecDeque<usize> = lag_ids.iter().copied().collect();
        let mut queued = vec![true; n];
        // PEs held by the drift bound: woken when a lagging PE starts a chunk
        let mut parked: Vec<usize> = Vec::new();
        let iq4 = |x: u32| {
            if x.is_multiple_of(2) {
                Colour::A1
            } else {
                Colour::A0
            }
        };
        let bound = |pe: &Pe, c: Colour| match c {
            Colour::A0 | Colour::A1 => iq4(pe.x) == c,
            Colour::Rb | Colour::Bb => pe.iq5 == c,
            Colour::Kf | Colour::Kb => pe.iq6 == c,
            Colour::Xf | Colour::Xb => self.x_queues || pe.iq2 == c,
        };
        // PEs waiting for room at a destination: woken when it reads
        let mut waiting_room: Vec<Vec<usize>> = vec![Vec::new(); n];
        // (a bounded tape: the words a column's stream has moved on, a PE has taken; the PEs
        // waiting on their column's)
        let mut tape_moved = vec![0u64; gx];
        let mut tape_taken = vec![0u64; n];
        let mut tape_wait: Vec<Vec<usize>> = vec![Vec::new(); gx];
        // (`cap` 0: a sender's messages not yet read)
        let mut unread = vec![0usize; n];
        // (`cap` 0: each multicast's destinations, which have read it, and the first that has
        // not: a wavelet reaches a PE only through the ones before it on the path)
        let mut casts: Vec<(Vec<usize>, Vec<bool>, usize)> = Vec::new();
        let mut rng = self.seed;
        macro_rules! wake {
            ($w:expr) => {{
                let w = $w;
                if !queued[w] && !pes[w].done {
                    queued[w] = true;
                    if lagging[w] {
                        ready_lag.push_back(w);
                    } else {
                        ready.push_back(w);
                    }
                }
            }};
        }
        while let Some(p) = {
            if self.seed != 0 && ready.len() > 1 {
                // (xorshift: a random ready PE next)
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let k = (rng % ready.len() as u64) as usize;
                ready.swap(0, k);
            }
            ready.pop_front().or_else(|| ready_lag.pop_front())
        } {
            queued[p] = false;
            loop {
                if pes[p].done {
                    break;
                }
                if pes[p].at >= pes[p].acts.len() {
                    // the next chunk (or op), within the drift bound
                    if pes[p].op as usize >= self.ops.len() {
                        pes[p].done = true;
                        pes[p].acts = Vec::new();
                        break;
                    }
                    if !lagging[p] && !lag_ids.is_empty() {
                        let slow = lag_ids
                            .iter()
                            .filter(|&&q| !pes[q].done)
                            .map(|&q| pes[q].started)
                            .min();
                        if let (Some(d), Some(s)) = (self.drift, slow)
                            && pes[p].started >= s + d
                        {
                            parked.push(p);
                            break;
                        }
                        if let Some(s) = slow {
                            found.lead = found.lead.max((pes[p].started + 1).saturating_sub(s));
                        }
                    }
                    let pe = &mut pes[p];
                    let (x, y, op, chunk) = (pe.x, pe.y, pe.op, pe.chunk);
                    let mut mode = pe.gen_mode;
                    pe.acts = self.acts(x, y, op, chunk, &mut mode);
                    pe.gen_mode = mode;
                    pe.at = 0;
                    pe.started += 1;
                    pe.chunk += 1;
                    if pe.chunk >= self.chunks(self.ops[op as usize]) {
                        pe.op += 1;
                        pe.chunk = 0;
                    }
                    if lagging[p] {
                        for w in std::mem::take(&mut parked) {
                            wake!(w);
                        }
                    }
                    continue;
                }
                if unread[p] > 0 {
                    break;
                }
                let a = pes[p].acts[pes[p].at];
                let (x, y) = (pes[p].x, pes[p].y);
                match a {
                    Act::Bind(q, c) => {
                        match q {
                            2 => pes[p].iq2 = c,
                            5 => pes[p].iq5 = c,
                            _ => pes[p].iq6 = c,
                        }
                        // the ramp's held wavelets, its head's queue now bound
                        while let Some(&(hc, m)) = ramp[p].front() {
                            if !bound(&pes[p], hc) {
                                break;
                            }
                            fifo[p][ci(hc)].push_back(m);
                            ramp[p].pop_front();
                        }
                        for w in std::mem::take(&mut waiting_room[p]) {
                            wake!(w);
                        }
                        if ramp[p].is_empty() {
                            for w in std::mem::take(&mut tape_wait[x as usize]) {
                                wake!(w);
                            }
                        }
                    }
                    Act::Send(c, tag) => {
                        let ds = self.dests(x, y, c);
                        // bounded: every destination's queue on the colour has room, and (not
                        // in ramp mode) a queue is bound to the colour there
                        if let Some(cap) = cap.filter(|&c| c > 0)
                            && let Some(&(dx, dy)) = ds.iter().find(|&&(dx, dy)| {
                                let d = id(dx, dy);
                                fifo[d][ci(c)].len() >= cap
                                    || (!self.hol && self.bind_blocks && !bound(&pes[d], c))
                            })
                        {
                            waiting_room[id(dx, dy)].push(p);
                            break;
                        }
                        let cast = casts.len() as u32;
                        if cap == Some(0) {
                            let ids: Vec<usize> = ds.iter().map(|&(dx, dy)| id(dx, dy)).collect();
                            let k = ids.len();
                            casts.push((ids, vec![false; k], 0));
                        }
                        for (pos, (dx, dy)) in ds.into_iter().enumerate() {
                            let d = id(dx, dy);
                            let m = Msg {
                                from: (x, y),
                                tag,
                                cast,
                                pos: pos as u32,
                            };
                            if self.hol && (!ramp[d].is_empty() || !bound(&pes[d], c)) {
                                ramp[d].push_back((c, m));
                            } else {
                                fifo[d][ci(c)].push_back(m);
                            }
                            wake!(d);
                        }
                        if cap == Some(0) && !self.dests(x, y, c).is_empty() {
                            // (its output queue drained: the nearest PE on the path has it)
                            unread[p] += 1;
                        }
                    }
                    Act::Tape => {
                        if !ramp[p].is_empty() {
                            break;
                        }
                        if let Some(room) = self.tape_cap {
                            let col = x as usize;
                            if tape_taken[p] >= tape_moved[col] {
                                // the stream's next word: every PE of the column has room and
                                // its ramp free (woken when a PE takes a word or frees its ramp)
                                if (0..rq as u32).map(|r| id(x, r)).any(|q| {
                                    tape_moved[col] - tape_taken[q] >= room || !ramp[q].is_empty()
                                }) {
                                    tape_wait[col].push(p);
                                    break;
                                }
                                tape_moved[col] += 1;
                            }
                            tape_taken[p] += 1;
                            for w in std::mem::take(&mut tape_wait[col]) {
                                wake!(w);
                            }
                        }
                    }
                    Act::Recv(q, from, tag) => {
                        let c = match q {
                            2 if self.x_queues => {
                                if from.0 < x {
                                    Colour::Xf
                                } else {
                                    Colour::Xb
                                }
                            }
                            2 => pes[p].iq2,
                            4 => iq4(x),
                            5 => pes[p].iq5,
                            _ => pes[p].iq6,
                        };
                        let Some(&m) = fifo[p][ci(c)].front() else {
                            break;
                        };
                        if cap == Some(0) && m.pos as usize > casts[m.cast as usize].2 {
                            // (not yet through the PEs before it on the path)
                            break;
                        }
                        fifo[p][ci(c)].pop_front();
                        if cap == Some(0) {
                            let cs = &mut casts[m.cast as usize];
                            cs.1[m.pos as usize] = true;
                            while cs.2 < cs.1.len() && cs.1[cs.2] {
                                cs.2 += 1;
                            }
                            if let Some(&next) = cs.0.get(cs.2) {
                                wake!(next);
                            }
                        }
                        found.messages += 1;
                        if cap == Some(0) && m.pos == 0 {
                            let s = id(m.from.0, m.from.1);
                            unread[s] -= 1;
                            if unread[s] == 0 {
                                wake!(s);
                            }
                        }
                        if (m.from != from || m.tag != tag) && found.wrong.len() < 20 {
                            found.wrong.push(format!(
                                "({x}, {y}) on {c:?}: wants {from:?} {tag:?}, takes {:?} {:?}",
                                m.from, m.tag
                            ));
                        }
                        for w in std::mem::take(&mut waiting_room[p]) {
                            wake!(w);
                        }
                    }
                }
                pes[p].at += 1;
            }
        }
        let mut blocked = 0usize;
        for (p, pe) in pes.iter().enumerate() {
            if !pe.done {
                blocked += 1;
                if found.blocked.len() < 3000 {
                    let what = pe.acts.get(pe.at).map(|a| match *a {
                        Act::Recv(q, from, tag) => {
                            let c = match q {
                                2 => pe.iq2,
                                4 => iq4(pe.x),
                                5 => pe.iq5,
                                _ => pe.iq6,
                            };
                            format!("reads iq{q} ({c:?}) from {from:?} {tag:?}")
                        }
                        Act::Send(c, tag) => format!("sends {c:?} {tag:?} (no room)"),
                        Act::Bind(q, c) => format!("binds iq{q} {c:?}"),
                        Act::Tape => {
                            if ramp[p].is_empty() {
                                "reads the tape (iq3), its column's stream stopped".into()
                            } else {
                                "reads the tape (iq3) behind its held ramp".into()
                            }
                        }
                    });
                    found.blocked.push(format!(
                        "({}, {}) {}{}",
                        pe.x,
                        pe.y,
                        what.unwrap_or_else(|| "?".into()),
                        if unread[p] > 0 {
                            format!(" (its {} sends not yet taken)", unread[p])
                        } else {
                            String::new()
                        }
                    ));
                }
            }
            if let Some(&(c, m)) = ramp[p].front()
                && found.held.len() < 30
            {
                found.held.push(format!(
                    "({}, {}) ramp held by {c:?} from {:?} {:?} ({} behind it)",
                    pe.x,
                    pe.y,
                    m.from,
                    m.tag,
                    ramp[p].len() - 1
                ));
            }
            for c in COLOURS {
                let k = fifo[p][ci(c)].len();
                if k > 0 && found.stranded.len() < 30 {
                    found
                        .stranded
                        .push(format!("({}, {}) {c:?}: {k}", pe.x, pe.y));
                }
            }
        }
        found.done = blocked == 0;
        if blocked > 0 {
            found.blocked.push(format!("{blocked} PEs blocked in all"));
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use super::{Op, Shape};

    fn shape(gx: u32, rq: u32, kbm: u32) -> Shape {
        Shape {
            gx,
            rq,
            band: 16,
            kbm,
            bind_blocks: true,
            seed: 0,
            bug: 0,
            split: rq,
            hol: false,
            lag: Vec::new(),
            drift: None,
            v13: false,
            x_queues: false,
            packed: false,
            v17: false,
            alt: false,
            tape_cap: None,
            ops: vec![
                Op::RowReduce(8),
                Op::HeadReduce(8),
                Op::Attend,
                Op::RowReduce(8),
                Op::HeadReduce(8),
                Op::Attend,
            ],
        }
    }

    /// The protocol's counts, orders and bindings agree on small arrays (every read takes the
    /// message the program means, none stranded, every PE done), unbounded and with a message
    /// of room a colour.
    #[test]
    fn the_gemm_protocol_agrees_on_small_arrays() {
        for (gx, rq, kbm) in [(32, 16, 8), (32, 16, 4), (48, 24, 8), (64, 32, 16)] {
            for cap in [None, Some(1)] {
                let f = shape(gx, rq, kbm).run(cap);
                assert!(
                    f.done && f.wrong.is_empty() && f.stranded.is_empty(),
                    "{gx} x {rq} kbm {kbm} cap {cap:?}: {f:?}"
                );
            }
        }
    }

    /// The check finds a planted bug: a member that keeps its column queue on the forward
    /// colour reads nothing of the members after it; band reductions with iq5 left on the row's
    /// colour wait on a broadcast that never comes to it.
    #[test]
    fn the_check_finds_planted_bugs() {
        for bug in [1, 2] {
            let mut s = shape(32, 16, 8);
            s.bug = bug;
            let f = s.run(None);
            assert!(!f.done, "bug {bug} unfound: {f:?}");
            eprintln!("bug {bug}: {:?}", f.blocked.first());
        }
    }

    /// klein's GEMM program (`kbm` the one free parameter), on a `gx x rq` array: its GEMMs'
    /// input words scaled from the 384-wide row.
    pub(super) fn klein(gx: u32, rq: u32, kbm: u32) -> Shape {
        let k = |w: u32| (w * gx).div_ceil(384);
        let g = |w: u32, mask: u8| Op::Gemm { k: k(w), mask };
        let r = |mask: u8, row: u32, head: u32| Op::Region { mask, row, head };
        Shape {
            gx,
            rq,
            band: 16,
            kbm,
            bind_blocks: false,
            seed: 0,
            bug: 0,
            // (the context's rows first: 512 of 4,608 tokens)
            split: (rq * 30).div_ceil(272),
            hol: true,
            lag: Vec::new(),
            drift: None,
            v13: false,
            x_queues: false,
            packed: false,
            v17: false,
            alt: false,
            tape_cap: None,
            ops: vec![
                g(256, 3),
                g(3072, 3),
                g(3072, 3),
                g(3072, 3),
                g(3072, 3),
                g(3072, 3),
                g(128, 2),
                r(1, 8, 0),
                g(3072, 1),
                r(1, 0, 4),
                r(2, 8, 0),
                g(3072, 2),
                r(2, 0, 4),
                Op::Attend,
                g(3072, 1),
                r(1, 8, 0),
                g(3072, 1),
                g(9216, 1),
                g(3072, 2),
                r(2, 8, 0),
                g(3072, 2),
                g(9216, 2),
            ],
        }
    }

    /// klein's attention and the region ops around it (no GEMM) in ramp mode: no colour reaches
    /// a PE before its queue is bound, at either `kbm`, whichever PEs are slow and however far
    /// the others drift.
    #[test]
    fn klein_attention_holds_no_ramp_at_either_kbm() {
        let r = |mask: u8, row: u32, head: u32| Op::Region { mask, row, head };
        for kbm in [8, 16] {
            for (lag, seed) in [
                (vec![], 0),
                (vec![], 7),
                ((0..48).map(|x| (x, 9)).collect(), 0),
            ] {
                let mut s = klein(48, 32, kbm);
                s.ops = vec![r(1, 8, 0), r(2, 0, 4), Op::Attend, r(1, 8, 0), r(2, 8, 0)];
                s.lag = lag;
                s.seed = seed;
                let f = s.run(None);
                assert!(
                    f.done && f.wrong.is_empty() && f.held.is_empty(),
                    "kbm {kbm}: {f:?}"
                );
            }
        }
    }

    /// Two GEMMs in a row in ramp mode: a PE that starts the next one sends `XF` east while the
    /// PE after it still has iq2 on `XB` for the last one's weights; the `XF` holds that ramp
    /// and the row stops, at either `kbm`.
    #[test]
    fn back_to_back_gemms_hold_a_ramp() {
        for kbm in [8, 16] {
            let f = klein(48, 32, kbm).run(None);
            assert!(
                !f.done && f.held.iter().any(|h| h.contains("held by Xf")),
                "kbm {kbm}: {f:?}"
            );
        }
    }

    /// A GEMM of fewer input words than PEs: a slow PE of no words rebinds iq2 to `XB` only at
    /// its own turn, so the next member's `XB` holds its ramp before the last piece's weights
    /// arrive, and the row stops.
    #[test]
    fn a_wordless_pe_slow_to_rebind_holds_its_ramp() {
        let mut s = klein(48, 4, 4);
        s.ops = vec![Op::Gemm { k: 24, mask: 3 }];
        s.split = 4;
        // (x 2: no words, `plen(2, 48, 24)` 0; members 1 and 3 have)
        s.lag = vec![(2, 0)];
        let f = s.run(None);
        assert!(!f.done && !f.held.is_empty(), "{f:?}");
        eprintln!("{:?}", f.held.first());
        s.lag.clear();
        assert!(s.run(None).done);
    }

    /// klein's whole program in ramp mode under a row-line rule, every slow PE set, drift and
    /// seed: the first run that holds a ramp, if any.
    fn ramp_free(rule: (bool, bool, bool), sync: bool, caps: &[Option<usize>]) -> Option<String> {
        let (gx, rq) = (48, 32);
        let lags: Vec<Vec<(u32, u32)>> = vec![
            vec![],
            vec![(2, 0)],
            vec![(1, 5)],
            vec![(20, 17)],
            vec![(47, 31)],
            (0..gx).map(|x| (x, 9)).collect(),
            (0..rq).map(|y| (21, y)).collect(),
            (16..32).map(|x| (x, 3)).collect(),
        ];
        for kbm in [8, 16] {
            for lag in &lags {
                for drift in [Some(0), Some(1), Some(4), None] {
                    for seed in [0, 7, 99] {
                        for &cap in caps {
                            let mut s = klein(gx, rq, kbm);
                            (s.v13, s.packed, s.x_queues) = rule;
                            if sync {
                                s.ops = s
                                    .ops
                                    .iter()
                                    .flat_map(|&o| match o {
                                        Op::Gemm { .. } => vec![o, Op::RowReduce(1)],
                                        _ => vec![o],
                                    })
                                    .collect();
                            }
                            s.lag = lag.clone();
                            s.drift = drift;
                            s.seed = seed;
                            let f = s.run(cap);
                            if !(f.done && f.wrong.is_empty() && f.held.is_empty()) {
                                return Some(format!(
                                    "kbm {kbm} lag {lag:?} drift {drift:?} seed {seed} cap {cap:?}: held {:?} wrong {:?} blocked {:?}",
                                    f.held.first(),
                                    f.wrong.first(),
                                    f.blocked.iter().take(6).collect::<Vec<_>>()
                                ));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// The row line's rules in ramp mode, on klein's whole program at both `kbm`s, every slow
    /// PE set, drift and seed, with pieces longer than the fabric's buffers (`cap` 0) and whole
    /// pieces buffered (`None`): rebinding iq2 anywhere inside the op holds a ramp; a row-wide
    /// reduction after each GEMM with `packed` words holds none, and neither does the row line
    /// on queues of its own.
    #[test]
    fn row_line_rules_in_ramp_mode() {
        let both = [Some(0), None];
        let free = |rule, sync| ramp_free(rule, sync, &both).is_none();
        assert!(!free((false, false, false), false), "as built");
        assert!(!free((true, false, false), false), "v13");
        assert!(!free((true, true, false), false), "v13, packed");
        assert!(!free((true, false, false), true), "v13, a reduction after");
        assert!(
            free((false, true, false), true),
            "packed, a reduction after"
        );
        assert!(
            free((true, true, false), true),
            "v13, packed, a reduction after"
        );
        assert!(free((false, false, true), false), "own queues");
    }

    /// klein's 1024² denoise as compile on load places it (384 x 576 PEs, key blocks of 8), its
    /// first `stages` stages (`None`: all).
    fn klein_1024(stages: Option<usize>) -> crate::compile::stream::gemm::GemmProgram {
        let o = crate::compile::stream::Options {
            reading: "denoise".into(),
            rows: 16,
            round16: true,
            upto: stages,
            taps: Vec::new(),
            anneal: Default::default(),
        };
        let t = poem_compiler::catalog::repository()
            .parse("flux2-klein-4b-bf16-kv-bf16")
            .unwrap()
            .trace(poem_ir::Platform::Cerebras);
        let rows = [("context".to_string(), 512), ("latents".to_string(), 4096)];
        let l =
            crate::compile::stream::gemm::lower(&t, &o, &rows).unwrap_or_else(|e| panic!("{e}"));
        crate::compile::stream::gemm::laid(&l, 384, 576, 8)
            .unwrap_or_else(|e| panic!("{}", e.text()))
    }

    /// The first run of `s` over slow PEs, drifts and seeds that holds a ramp, reads a wrong
    /// message or does not finish.
    fn first_unclean(s: &Shape) -> Option<String> {
        let (gx, rq) = (s.gx, s.rq);
        let lags: Vec<Vec<(u32, u32)>> = vec![
            vec![],
            vec![(1, 5)],
            vec![(gx - 1, rq - 1)],
            (0..gx).map(|x| (x, rq / 3)).collect(),
            (0..rq).map(|y| (gx / 2, y)).collect(),
            (16..32).map(|x| (x, 3)).collect(),
        ];
        for lag in &lags {
            for drift in [Some(0), Some(1), Some(4), None] {
                for seed in [0, 7] {
                    let mut t = s.clone();
                    t.lag = lag.clone();
                    t.drift = drift;
                    t.seed = seed;
                    let f = t.run(None);
                    if !(f.done && f.wrong.is_empty() && f.held.is_empty()) {
                        return Some(format!(
                            "lag {lag:?} drift {drift:?} seed {seed}: held {:?} wrong {:?} blocked {:?}",
                            f.held.first(),
                            f.wrong.first(),
                            f.blocked.iter().take(4).collect::<Vec<_>>()
                        ));
                    }
                }
            }
        }
        None
    }

    /// klein's whole 1024² program, its op list read off the placed program, on a 48 x 64 array
    /// in ramp mode: as emitted (the row line on its own queues, the attentions alternating
    /// their members' order) no colour reaches a PE before its queue is bound. With the order
    /// fixed, a slow row is still on `KB` when the next attention's first `KF` comes and holds
    /// its ramp.
    #[test]
    fn klein_1024s_whole_program_holds_no_ramp_as_emitted() {
        let p = klein_1024(None);
        let s = Shape::of(&p, 48, 64);
        let attends = s.ops.iter().filter(|o| matches!(o, Op::Attend)).count();
        assert_eq!(attends, 25, "{:?}", s.ops);
        assert_eq!(first_unclean(&s), None, "as emitted");
        for v17 in [false, true] {
            let mut fixed = s.clone();
            fixed.alt = false;
            fixed.v17 = v17;
            let bad = first_unclean(&fixed).expect("the order fixed holds a ramp");
            assert!(
                bad.contains("held Some(") && bad.contains("by Kf"),
                "v17 {v17}: {bad}"
            );
        }
    }
}
