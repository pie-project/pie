//! A static check of a GEMM-regime program's fabric protocol (`emit/gemm_pe.csl`'s attention
//! and `gseg.csl`'s reductions): every PE's sends and reads, colour by colour, run as a
//! process network over the array, message by message.
//!
//! - **Kahn mode** (unbounded FIFOs a PE and colour): the outcome does not depend on timing.
//!   A read takes the next message on the colour its queue is bound to *at that moment*; it
//!   must be the message the program means (the sender and the round it is tagged with). A
//!   message delivered to a PE's ramp that no read takes is stranded; a PE waiting on a colour
//!   no message is coming on is a deadlock. Clean here means the protocol's counts, orders
//!   and queue bindings agree: a hang must then come from bounded buffers or timing.
//! - **Bounded mode** (`cap` messages a PE and colour, a multicast sent only once every
//!   destination has room): a deterministic schedule's deadlock, reported as the cycle of
//!   waits it ends in.
//!
//! Colours (`gpe.csl`): the column line `KF` (south) / `KB` (north) on iq6, the reductions'
//! chain `A0` / `A1` on iq4 (a PE sends on its parity's, reads the other), their broadcast `RB`
//! (the row's) / `BB` (the band's) on iq5 (`gseg.setmode` rebinds it).

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
}

const NC: usize = 6;

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
}

/// A message: its sender, and what it is (an op's index and a round of it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Msg {
    from: (u32, u32),
    tag: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug)]
enum Act {
    /// Input queue 5 or 6 bound to a colour (iq4's is fixed by parity).
    Bind(u8, Colour),
    /// A message multicast on a colour by its line's routes, tagged.
    Send(Colour, (u32, u32, u32)),
    /// A read off input queue 4, 5 or 6, the message expected (its sender, its tag).
    Recv(u8, (u32, u32), (u32, u32, u32)),
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
    iq5: Colour,
    iq6: Colour,
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
        }
    }

    /// PE `(x, y)`'s actions in chunk `k` of op `oi` (`mode`: iq5's binding, as `gseg` keeps
    /// it).
    fn acts(&self, x: u32, y: u32, oi: u32, k: u32, mode: &mut u32) -> Vec<Act> {
        let mut v = Vec::new();
        match self.ops[oi as usize] {
            Op::Attend => {
                if k == 0 {
                    v.push(Act::Bind(6, Colour::Kf));
                }
                let j = k * self.kbm;
                for i in 0..self.kbm {
                    let m = j + i;
                    if m >= self.rq {
                        break;
                    }
                    if m == y {
                        if self.bug != 1 {
                            v.push(Act::Bind(6, Colour::Kb));
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
                }
                for r in 0..2 {
                    self.allred(&mut v, x, y, true, mode, (oi, 1, 2 * k + r));
                }
            }
            Op::RowReduce(_) => self.allred(&mut v, x, y, false, mode, (oi, 2, k)),
            Op::HeadReduce(_) => self.allred(&mut v, x, y, true, mode, (oi, 3, k)),
        }
        v
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
            Colour::Kb => (0..y).map(|r| (x, r)).collect(),
            // (a chain hop: the next PE east)
            Colour::A0 | Colour::A1 => {
                if x + 1 < self.gx {
                    vec![(x + 1, y)]
                } else {
                    Vec::new()
                }
            }
            Colour::Rb => (0..x).map(|c| (c, y)).collect(),
            Colour::Bb => {
                let lo = (x / self.band) * self.band;
                (lo..x).map(|c| (c, y)).collect()
            }
        }
    }

    /// Runs the network; `cap`: messages a PE's colour holds (`None`: unbounded).
    #[must_use]
    pub fn run(&self, cap: Option<usize>) -> Found {
        let (gx, rq) = (self.gx as usize, self.rq as usize);
        let id = |x: u32, y: u32| (y as usize) * gx + x as usize;
        let mut pes: Vec<Pe> = (0..rq as u32)
            .flat_map(|y| (0..gx as u32).map(move |x| (x, y)))
            .map(|(x, y)| Pe {
                x,
                y,
                op: 0,
                chunk: 0,
                acts: Vec::new(),
                at: 0,
                iq5: Colour::Rb,
                iq6: Colour::Kf,
                gen_mode: 0,
                done: false,
            })
            .collect();
        let mut fifo: Vec<[VecDeque<Msg>; NC]> = (0..gx * rq).map(|_| Default::default()).collect();
        let mut found = Found::default();
        let mut ready: VecDeque<usize> = (0..gx * rq).collect();
        let mut queued = vec![true; gx * rq];
        let iq4 = |x: u32| {
            if x.is_multiple_of(2) {
                Colour::A1
            } else {
                Colour::A0
            }
        };
        // PEs waiting for room at a destination: woken when it reads
        let mut waiting_room: Vec<Vec<usize>> = vec![Vec::new(); gx * rq];
        let mut rng = self.seed;
        while let Some(p) = {
            if self.seed != 0 && ready.len() > 1 {
                // (xorshift: a random ready PE next)
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let k = (rng % ready.len() as u64) as usize;
                ready.swap(0, k);
            }
            ready.pop_front()
        } {
            queued[p] = false;
            loop {
                if pes[p].done {
                    break;
                }
                if pes[p].at >= pes[p].acts.len() {
                    // the next chunk (or op)
                    let pe = &mut pes[p];
                    if pe.op as usize >= self.ops.len() {
                        pe.done = true;
                        pe.acts = Vec::new();
                        break;
                    }
                    let (x, y, op, chunk) = (pe.x, pe.y, pe.op, pe.chunk);
                    let mut mode = pe.gen_mode;
                    pe.acts = self.acts(x, y, op, chunk, &mut mode);
                    pe.gen_mode = mode;
                    pe.at = 0;
                    pe.chunk += 1;
                    if pe.chunk >= self.chunks(self.ops[op as usize]) {
                        pe.op += 1;
                        pe.chunk = 0;
                    }
                    continue;
                }
                let a = pes[p].acts[pes[p].at];
                let (x, y) = (pes[p].x, pes[p].y);
                match a {
                    Act::Bind(q, c) => {
                        if q == 5 {
                            pes[p].iq5 = c;
                        } else {
                            pes[p].iq6 = c;
                        }
                        for w in std::mem::take(&mut waiting_room[p]) {
                            if !queued[w] {
                                queued[w] = true;
                                ready.push_back(w);
                            }
                        }
                    }
                    Act::Send(c, tag) => {
                        let ds = self.dests(x, y, c);
                        // bounded: every destination's queue on the colour has room, and a
                        // queue is bound to the colour there (a wavelet to a ramp with no queue
                        // on its colour waits in the router, holding the multicast)
                        if let Some(cap) = cap
                            && let Some(&(dx, dy)) = ds.iter().find(|&&(dx, dy)| {
                                let d = &pes[id(dx, dy)];
                                let bound = match c {
                                    Colour::A0 | Colour::A1 => iq4(dx) == c,
                                    Colour::Rb | Colour::Bb => d.iq5 == c,
                                    Colour::Kf | Colour::Kb => d.iq6 == c,
                                };
                                fifo[id(dx, dy)][ci(c)].len() >= cap || (self.bind_blocks && !bound)
                            })
                        {
                            waiting_room[id(dx, dy)].push(p);
                            break;
                        }
                        for (dx, dy) in ds {
                            let d = id(dx, dy);
                            fifo[d][ci(c)].push_back(Msg { from: (x, y), tag });
                            if !queued[d] && !pes[d].done {
                                queued[d] = true;
                                ready.push_back(d);
                            }
                        }
                    }
                    Act::Recv(q, from, tag) => {
                        let c = match q {
                            4 => iq4(x),
                            5 => pes[p].iq5,
                            _ => pes[p].iq6,
                        };
                        let Some(m) = fifo[p][ci(c)].pop_front() else {
                            break;
                        };
                        found.messages += 1;
                        if (m.from != from || m.tag != tag) && found.wrong.len() < 20 {
                            found.wrong.push(format!(
                                "({x}, {y}) on {c:?}: wants {from:?} {tag:?}, takes {:?} {:?}",
                                m.from, m.tag
                            ));
                        }
                        for w in std::mem::take(&mut waiting_room[p]) {
                            if !queued[w] {
                                queued[w] = true;
                                ready.push_back(w);
                            }
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
                if found.blocked.len() < 30 {
                    let what = pe.acts.get(pe.at).map(|a| match *a {
                        Act::Recv(q, from, tag) => {
                            let c = match q {
                                4 => iq4(pe.x),
                                5 => pe.iq5,
                                _ => pe.iq6,
                            };
                            format!("reads iq{q} ({c:?}) from {from:?} {tag:?}")
                        }
                        Act::Send(c, tag) => format!("sends {c:?} {tag:?} (no room)"),
                        Act::Bind(q, c) => format!("binds iq{q} {c:?}"),
                    });
                    found.blocked.push(format!(
                        "({}, {}) {}",
                        pe.x,
                        pe.y,
                        what.unwrap_or_else(|| "?".into())
                    ));
                }
            }
            for c in [
                Colour::Kf,
                Colour::Kb,
                Colour::A0,
                Colour::A1,
                Colour::Rb,
                Colour::Bb,
            ] {
                let n = fifo[p][ci(c)].len();
                if n > 0 && found.stranded.len() < 30 {
                    found
                        .stranded
                        .push(format!("({}, {}) {c:?}: {n}", pe.x, pe.y));
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

    /// klein's cuts at full size (c2, c4, c5): `GEMM_FLOW=1 cargo test -- --ignored`.
    #[test]
    #[ignore = "full-size arrays: minutes"]
    fn klein_cuts() {
        for (name, rq, kbm) in [("c5", 272, 8), ("c4", 272, 16), ("c2", 136, 8)] {
            for (cap, seed) in [
                (None, 0),
                (Some(1), 0),
                (Some(2), 0),
                (Some(1), 7),
                (Some(1), 99),
            ] {
                let t0 = std::time::Instant::now();
                let mut s = shape(384, rq, kbm);
                s.seed = seed;
                let f = s.run(cap);
                eprintln!(
                    "{name} cap {cap:?} seed {seed}: done {} messages {} wrong {:?} blocked {:?} stranded {:?} ({:.0} s)",
                    f.done,
                    f.messages,
                    f.wrong.first(),
                    f.blocked.iter().take(6).collect::<Vec<_>>(),
                    f.stranded.first(),
                    t0.elapsed().as_secs_f64()
                );
            }
        }
    }
}
