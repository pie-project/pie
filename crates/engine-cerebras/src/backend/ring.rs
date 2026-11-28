//! A program's ring of seats: Pie's lanes over the sequences it holds at once. A launch passes
//! every seat, so a seat is never paused: one whose lane asks nothing runs on over its own argmax
//! (rows the lane takes later), one with no lane runs a sequence nothing reads. A seat can't go
//! back: a lane whose rows leave what its seat ran restarts it from position 0, its history forced.

use std::collections::HashMap;

use engine::program::InstanceId;

use super::fire::FIRST_LOOP;
use super::kv::Cached;
use crate::wafer;

/// A lane's rows in a fire: every token to its last row, and the positions whose logits are read
/// (none: the argmax after the last).
pub(super) struct Want {
    pub(super) slot: u32,
    pub(super) instance: Option<InstanceId>,
    pub(super) fed: Vec<u32>,
    pub(super) logits: Vec<usize>,
}

/// What a lane's want was answered with.
#[derive(Default)]
pub(super) struct Got {
    pub(super) argmax: Option<u32>,
    pub(super) logits: Vec<Vec<f32>>,
}

#[derive(Default)]
struct Lane {
    /// The tokens run and the argmax after each, as the host was told.
    known: Cached,
    instance: Option<InstanceId>,
    /// The fire it last asked rows in.
    fired: u64,
    /// The rows it last asked.
    asked: usize,
}

/// One seat's passes of the launch out: its lane, the position and token they start from, and
/// the tokens forced after.
struct Run {
    slot: u32,
    pos: usize,
    first: u32,
    force: Vec<u32>,
}

#[derive(Clone, Copy, Default)]
struct Seat {
    lane: Option<u32>,
    /// The wafer holds the lane's known rows here (no other sequence ran since).
    holds: bool,
}

/// The seats' shared KV pool: its positions and rows, and each seat's window in it (first
/// position, positions); `sent` once the wafer was told them. A seat with no lane has no window
/// (none of its positions: its idle passes write no KV).
struct Pool {
    size: usize,
    rows: usize,
    wins: Vec<(usize, usize)>,
    sent: bool,
}

/// A seat's window when it has no lane: no positions (the wafer writes none of its KV).
pub(super) const IDLE: (usize, usize) = (0, 0);

/// The pool's free stretches (from, to) past the windows of the seats `of` takes.
fn gaps(wins: &[(usize, usize)], size: usize, of: impl Fn(usize) -> bool) -> Vec<(usize, usize)> {
    let mut taken: Vec<(usize, usize)> = (0..wins.len())
        .filter(|&k| of(k))
        .map(|k| wins[k])
        .collect();
    taken.sort_unstable();
    let (mut gaps, mut at) = (Vec::new(), 0);
    for &(b, n) in &taken {
        gaps.push((at, b));
        at = at.max(b + n);
    }
    gaps.push((at, size));
    gaps
}

/// Each seat's window: (first position, positions).
type Windows = Vec<(usize, usize)>;

/// Seat `j`'s window grown to at least `floor` positions (as much of `want` as fits): from `at`
/// (in place), else from the start of the widest stretch, the seats `movable` takes put where
/// room is left wherever they are in its way, each at its size (one holding nothing at `least`).
/// The windows, the rows the moved seats held, and the stretch it grows in.
#[allow(clippy::too_many_arguments)]
fn make_room(
    wins: &[(usize, usize)],
    j: usize,
    movable: &dyn Fn(usize) -> bool,
    held: &dyn Fn(usize) -> usize,
    at: Option<usize>,
    (floor, want, least): (usize, usize, usize),
    size: usize,
    rows: usize,
) -> Option<(Windows, usize, usize)> {
    let up = |n: usize| n.div_ceil(rows) * rows;
    let m = wins.len();
    let open = gaps(wins, size, |k| k != j && !movable(k));
    let (a, b) = match at {
        Some(base) => open
            .iter()
            .find(|&&(a, b)| a <= base && base < b)
            .map(|&(_, b)| (base, b))?,
        None => open
            .iter()
            .map(|&(a, b)| (up(a), b))
            .filter(|&(a, b)| a < b)
            .max_by_key(|&(a, b)| b - a)?,
    };
    // (as much of `want` as leaves the seats in its way room)
    let fit = |n: usize| {
        let mut placed = wins.to_vec();
        placed[j] = (a, n);
        let (mut cost, mut done) = (0, vec![false; m]);
        for k in (0..m).filter(|&k| k != j && movable(k)) {
            let fixed = |o: usize| o != k && (!movable(o) || o == j || done[o]);
            let (c, mut l) = placed[k];
            let over = (0..m)
                .filter(|&o| fixed(o))
                .any(|o| c < placed[o].0 + placed[o].1 && placed[o].0 < c + l);
            if over {
                if held(k) == 0 {
                    l = l.min(least);
                }
                let room = gaps(&placed, size, fixed);
                let &(s, _) = room.iter().find(|&&(s, e)| up(s) + l <= e)?;
                placed[k] = (up(s), l);
                cost += held(k);
            }
            done[k] = true;
        }
        Some((placed, cost, b - a))
    };
    let n = want.min((b - a) / rows * rows);
    let shorter = (floor.div_ceil(rows)..n.div_ceil(rows))
        .rev()
        .map(|r| r * rows);
    std::iter::once(n)
        .chain(shorter)
        .filter(|&n| n >= floor)
        .find_map(fit)
}

pub(super) struct Ring {
    positions: usize,
    pool: Option<Pool>,
    /// The laps a launch may run, and those past a lane's rows it runs at most.
    most: usize,
    ahead: usize,
    /// The next launch's laps past its rows: doubled a launch, short again after a lane starts.
    next: usize,
    seats: Vec<Seat>,
    lanes: HashMap<u32, Lane>,
    fires: u64,
    out: Option<Vec<Option<Run>>>,
}

impl Ring {
    /// `pool`: the seats' shared KV pool, (positions, rows); every seat starts with no window.
    pub(super) fn new(
        seats: usize,
        positions: usize,
        cap: usize,
        ahead: usize,
        pool: Option<(usize, usize)>,
    ) -> Ring {
        let most = (cap / seats).max(1);
        let pool = pool.map(|(size, rows)| Pool {
            size: size / rows * rows,
            rows,
            wins: vec![IDLE; seats],
            sent: false,
        });
        // (a seat holds at most the pool)
        let positions = pool.as_ref().map_or(positions, |p| positions.min(p.size));
        Ring {
            positions,
            pool,
            most,
            ahead: ahead.clamp(1, most),
            next: FIRST_LOOP,
            seats: vec![Seat::default(); seats],
            lanes: HashMap::new(),
            fires: 0,
            out: None,
        }
    }

    /// The positions a sequence may reach on a seat.
    pub(super) fn positions(&self) -> usize {
        self.positions
    }

    pub(super) fn seats(&self) -> u32 {
        u32::try_from(self.seats.len()).unwrap_or(u32::MAX)
    }

    /// Pie's pages on a shared pool: (a page's positions, pages). A page is a pool row, the
    /// unit a window takes, so the rows Pie's lanes hold are their windows' rows (no rounding
    /// slack, and a seat with no lane holds none).
    pub(super) fn pool_pages(&self) -> Option<(u32, u32)> {
        let p = self.pool.as_ref()?;
        Some((
            u32::try_from(p.rows).ok()?,
            u32::try_from(p.size / p.rows).ok()?,
        ))
    }

    /// The positions seat `j` may run: its window's, else a seat's.
    fn room(&self, j: usize) -> usize {
        self.pool
            .as_ref()
            .map_or(self.positions, |p| p.wins[j].1.min(self.positions))
    }

    /// The positions a window holds at least: a launch's laps on an idle seat, whole rows.
    fn least(&self, rows: usize) -> usize {
        self.most.div_ceil(rows) * rows
    }

    /// The pool's windows for the `wants`' seats: a seat whose window can't hold its lane's rows
    /// takes a larger one (twice the rows, at least an even share): in place where the stretch
    /// past it is free, else in the widest free stretch, else every window is laid out again from
    /// the start, each at what its lane holds, the asking seat last; a moved window's seat starts
    /// again from position 0.
    /// `needs`: the positions each lane's seat is to run to, and whether it must (else its
    /// window grows only where room is left, none laid out again for it).
    fn place(&mut self, needs: &[(u32, usize, bool)]) -> Result<(), String> {
        let Some(pool) = self.pool.as_ref() else {
            return Ok(());
        };
        let (size, rows) = (pool.size, pool.rows);
        let m = self.seats.len();
        let least = self.least(rows);
        let up = |n: usize| n.div_ceil(rows) * rows;
        // (a window whole rows)
        let cap = up(self.positions).min(size);
        let even = size / m / rows * rows;
        let mut wins = pool.wins.clone();
        // (a seat that holds nothing and asks for nothing: no window)
        let idle = |k: usize, wins: &Windows| {
            let s = self.seats[k];
            let asks = s.lane.is_some_and(|l| needs.iter().any(|w| w.0 == l));
            wins[k].1 == 0
                || (!asks
                    && s.lane
                        .filter(|_| s.holds)
                        .is_none_or(|l| self.lane(l).is_empty()))
        };
        for k in 0..m {
            if idle(k, &wins) {
                wins[k] = IDLE;
            }
        }
        for &(slot, need, must) in needs {
            let Some(j) = self.seat_of(slot) else {
                continue;
            };
            if wins[j].1 >= need {
                continue;
            }
            if need > cap {
                return Err(format!(
                    "slot {slot} holds {need} positions, a seat at most {cap}"
                ));
            }
            // (a launch's laps past its rows where room is left: Pie's pages hold the rows alone)
            let floor = up(need).min(cap);
            let want = up(2 * need).max(even).max(least).max(floor).min(cap);
            // the rows a seat runs again where its window moves (none for one holding nothing)
            let held = |k: usize| {
                let s = self.seats[k];
                s.lane.filter(|_| s.holds).map_or(0, |l| self.lane(l).len())
            };
            let mine = held(j);
            // windows that may move out of its way: those holding nothing, or fewer rows than it
            let free = |k: usize| held(k) == 0;
            let cheap = |k: usize| held(k) < mine;
            // the least rows run again of: in place (the free, else the cheap, out of its way),
            // or moved to the widest stretch; one in a stretch short of a sequence's most counted
            // as moving again at its end, and of the same, the widest stretch to grow in
            let tries: [(&dyn Fn(usize) -> bool, bool); 3] =
                [(&free, true), (&cheap, true), (&free, false)];
            let best = tries
                .into_iter()
                .filter_map(|(movable, here)| {
                    let at = here.then_some(wins[j].0);
                    let (placed, cost, room) = make_room(
                        &wins,
                        j,
                        movable,
                        &held,
                        at,
                        (floor, want, least),
                        size,
                        rows,
                    )?;
                    let again = if here { 0 } else { mine } + if room < cap { room } else { 0 };
                    Some(((cost + again, std::cmp::Reverse(room)), placed))
                })
                .min_by_key(|t| t.0);
            if let Some((_, placed)) = best {
                wins = placed;
                continue;
            }
            if !must {
                continue;
            }
            // every window again from the start: each at the rows its lane asks, an idle lane's
            // at what it holds while they fit, each at the least while they fit
            let held = |k: usize, (idle, low): (bool, bool)| {
                let n = match self.seats[k].lane {
                    _ if k == j => floor,
                    Some(l) => needs
                        .iter()
                        .find(|w| w.0 == l && w.2)
                        .map_or(if idle { self.lane(l).len() } else { 0 }, |w| w.1),
                    None => 0,
                };
                // (none: no window)
                let low = if low { least } else { rows };
                if n == 0 { 0 } else { up(n).max(low).min(cap) }
            };
            let fits = |t| (0..m).map(|k| held(k, t)).sum::<usize>() <= size;
            let idle = [(true, true), (false, true), (true, false)]
                .into_iter()
                .find(|&t| fits(t))
                .unwrap_or((false, false));
            // (the seat that asked last, the pool's end past it to grow into in place)
            let mut at = 0;
            for k in (0..m).filter(|&k| k != j).chain([j]) {
                let n = held(k, idle);
                wins[k] = (at, n);
                at += n;
            }
            if at > size {
                return Err(format!(
                    "the seats' windows take {at} positions of a pool of {size}"
                ));
            }
            wins[j].1 += want
                .saturating_sub(wins[j].1)
                .min((size - at) / rows * rows);
        }
        for k in 0..m {
            if idle(k, &wins) {
                wins[k] = IDLE;
            }
        }
        let pool = self.pool.as_mut().expect("a pool");
        for (k, (w, was)) in wins.iter().zip(&pool.wins).enumerate() {
            // (one grown in place keeps its rows; moved or cut short, its seat starts again)
            if w.0 != was.0 || w.1 < was.1 {
                self.seats[k].holds = false;
            }
            pool.sent &= w == was;
        }
        pool.wins = wins;
        Ok(())
    }

    fn lane(&self, slot: u32) -> &[u32] {
        self.lanes.get(&slot).map_or(&[], |l| &l.known.tokens)
    }

    fn seat_of(&self, slot: u32) -> Option<usize> {
        self.seats.iter().position(|s| s.lane == Some(slot))
    }

    /// The argmax after `fed`, where the ring ran it for lane `slot`.
    fn known(&self, slot: u32, fed: &[u32]) -> Option<u32> {
        let l = self.lanes.get(&slot)?;
        if fed.is_empty() || !l.known.tokens.starts_with(fed) {
            return None;
        }
        l.known.argmax[fed.len() - 1]
    }

    /// Where lane `slot`'s seat runs `fed` from: the end of what it ran, else position 0.
    fn from(&self, slot: u32, fed: &[u32]) -> usize {
        let ran = self.lane(slot);
        let holds = self.seat_of(slot).is_some_and(|j| self.seats[j].holds);
        if holds && ran.len() < fed.len() && fed.starts_with(ran) {
            ran.len()
        } else {
            0
        }
    }

    /// Seats the lanes `slots` (no more than the seats): those unseated take the first free seat,
    /// else the seat whose lane fired longest ago.
    fn seat(&mut self, slots: &[u32]) {
        for &slot in slots {
            if self.seat_of(slot).is_some() {
                continue;
            }
            let fired = |s: &Seat| {
                s.lane
                    .map_or(0, |l| self.lanes.get(&l).map_or(0, |l| l.fired))
            };
            let free = (0..self.seats.len())
                .filter(|&j| self.seats[j].lane.is_none_or(|l| !slots.contains(&l)))
                .min_by_key(|&j| fired(&self.seats[j]));
            if let Some(j) = free {
                self.seats[j] = Seat {
                    lane: Some(slot),
                    holds: false,
                };
            }
        }
    }

    /// The next launch, running `wants`' rows (lanes told apart, each with rows to run) and
    /// every other seat on: to the end of the first want that reads logits where one launch
    /// reaches it, else past the rows.
    fn launch(&mut self, wants: &[(u32, &[u32], bool)]) -> Result<wafer::Launch, String> {
        let m = self.seats.len();
        let wants = &wants[..wants.len().min(m)];
        let slots: Vec<u32> = wants.iter().map(|w| w.0).collect();
        self.seat(&slots);
        // the windows to hold the wants' rows, and to let a seat running on over its own argmax
        // run its next laps (its window full, it would be let go)
        let laps = self.next.max(1).min(self.ahead);
        let mut needs: Vec<(u32, usize, bool)> =
            wants.iter().map(|w| (w.0, w.1.len(), true)).collect();
        for s in &self.seats {
            if let (Some(slot), true) = (s.lane, s.holds)
                && !slots.contains(&slot)
            {
                // (no further than it runs past the rows it asked)
                let on = self.lane(slot).len() + laps;
                let asked = self.lanes.get(&slot).map_or(0, |l| l.asked);
                if on <= self.positions && on <= asked + self.ahead + laps {
                    needs.push((slot, on, false));
                }
            }
        }
        self.place(&needs)?;
        let from = |s: &Ring, slot| {
            wants
                .iter()
                .find(|w| w.0 == slot)
                .map(|w| (w.1, s.from(slot, w.1)))
        };
        // laps: the rows' and some past them, no seat past its positions
        let need = wants.iter().map(|w| w.1.len() - self.from(w.0, w.1)).max();
        // (a lane joining or restarting: the others' rows wait on its launch, kept short)
        if wants.iter().any(|w| self.from(w.0, w.1) == 0) {
            self.next = FIRST_LOOP;
        }
        let room = wants
            .iter()
            .filter_map(|w| Some(self.room(self.seat_of(w.0)?) - self.from(w.0, w.1)))
            .min();
        let read = wants
            .iter()
            .filter(|w| w.2)
            .filter_map(|w| Some((self.seat_of(w.0)?, w.1.len() - self.from(w.0, w.1))))
            .min();
        let mut passes = 0;
        let mut reads = false;
        if let Some((j, laps)) = read {
            let most = self.most.min(room.unwrap_or(self.positions));
            // (a forced stretch before it stops short of the rows read)
            reads = laps <= most;
            passes = if reads {
                (laps - 1) * m + j + 1
            } else {
                most * m
            };
        }
        if passes == 0 {
            let mut laps = self.next.max(need.unwrap_or(0)).min(self.ahead);
            laps = laps.min(room.unwrap_or(laps));
            for (j, s) in self.seats.iter().enumerate() {
                if let (Some(slot), true) = (s.lane, s.holds)
                    && !slots.contains(&slot)
                {
                    // (one at its last position has no more rows: its seat is let go)
                    match self.room(j).saturating_sub(self.lane(slot).len()) {
                        0 => {}
                        left => laps = laps.min(left),
                    }
                }
            }
            passes = laps * m;
            self.next = (self.next * 2).min(self.ahead);
        }
        let mut launch = wafer::Launch {
            first: vec![0; m],
            pos: vec![0; m],
            force: vec![Vec::new(); m],
            passes,
            logits: reads,
            windows: None,
        };
        let mut out = Vec::with_capacity(m);
        for j in 0..m {
            let laps = (passes + m - 1 - j) / m;
            let seat = self.seats[j];
            let run = seat.lane.filter(|_| laps > 0).and_then(|slot| {
                if let Some((fed, pos)) = from(self, slot) {
                    let end = fed.len().min(pos + laps);
                    return Some(Run {
                        slot,
                        pos,
                        first: fed[pos],
                        force: fed[pos + 1..end].to_vec(),
                    });
                }
                let k = &self.lanes.get(&slot)?.known;
                let on = seat.holds && k.tokens.len() + laps <= self.room(j);
                on.then_some(Run {
                    slot,
                    pos: k.tokens.len(),
                    first: k.argmax.last().copied().flatten()?,
                    force: Vec::new(),
                })
            });
            // (a seat of no pass stays as it is)
            self.seats[j].holds = if laps > 0 { run.is_some() } else { seat.holds };
            if let Some(run) = &run {
                launch.first[j] = run.first;
                launch.pos[j] = run.pos;
                launch.force[j].clone_from(&run.force);
                let known = &mut self.lanes.entry(run.slot).or_default().known;
                // (rows it ran before, run again)
                let again = wants.iter().find(|w| w.0 == run.slot).map_or(0, |w| {
                    known
                        .tokens
                        .iter()
                        .zip(w.1)
                        .take_while(|(a, b)| a == b)
                        .count()
                });
                if run.pos == 0 && again > 0 {
                    tracing::info!(
                        slot = run.slot,
                        seat = j,
                        again,
                        "ring: a lane's seat starts again from 0, its rows forced"
                    );
                }
                known.truncate(run.pos);
            }
            // (a seat running no lane's rows writes no cells: it has no window)
            if let Some(p) = self.pool.as_mut().filter(|_| laps > 0 && run.is_none()) {
                p.sent &= p.wins[j] == IDLE;
                p.wins[j] = IDLE;
            }
            out.push(run);
        }
        launch.windows = self.pool.as_mut().filter(|p| !p.sent).map(|p| {
            p.sent = true;
            p.wins.clone()
        });
        self.out = Some(out);
        Ok(launch)
    }

    /// The launch out landed with `winners`, every pass's argmax.
    fn land(&mut self, winners: &[u32]) {
        let m = self.seats.len();
        for (j, run) in self.out.take().into_iter().flatten().enumerate() {
            let Some(run) = run.filter(|r| self.seats[j].lane == Some(r.slot)) else {
                continue;
            };
            let Some(lane) = self.lanes.get_mut(&run.slot) else {
                continue;
            };
            let mut token = run.first;
            for (lap, &best) in winners.iter().skip(j).step_by(m).enumerate() {
                lane.known.tokens.push(token);
                lane.known.argmax.push(Some(best));
                token = run.force.get(lap).copied().unwrap_or(best);
            }
        }
    }

    /// The launch out failed: what the seats hold is not known.
    fn lost(&mut self) {
        self.out = None;
        self.lanes.clear();
        self.seats.fill(Seat::default());
        if let Some(p) = self.pool.as_mut() {
            p.sent = false;
        }
    }

    /// The lanes `instance` fired last are done: their seats are free.
    pub(super) fn close(&mut self, instance: InstanceId) {
        let done = |l: &Lane| l.instance == Some(instance);
        // (a free seat's window gives back all past an even share, as the ring started)
        let m = self.seats.len();
        let keep = self.pool.as_ref().map(|p| {
            (p.size / m / p.rows * p.rows)
                .min(self.positions)
                .max(self.least(p.rows))
        });
        for (j, s) in self.seats.iter_mut().enumerate() {
            if s.lane.is_some_and(|l| self.lanes.get(&l).is_some_and(done)) {
                *s = Seat::default();
                if let (Some(p), Some(keep)) = (self.pool.as_mut(), keep)
                    && p.wins[j].1 > keep
                {
                    p.wins[j].1 = keep;
                    p.sent = false;
                }
            }
        }
        self.lanes.retain(|_, l| !done(l));
    }

    /// Whether a seat's lane among `wants` is within a launch of its last row run.
    fn runs_low(&self, wants: &[Want]) -> bool {
        wants.iter().any(|w| {
            let ran = self.lane(w.slot).len();
            let seat = self.seat_of(w.slot);
            let holds = seat.is_some_and(|j| self.seats[j].holds);
            holds && seat.is_some_and(|j| ran < self.room(j)) && ran <= w.fed.len() + self.ahead
        })
    }

    /// Runs a fire's `wants` on `dev`, the seats running on after it where `ahead`; each one's
    /// answer.
    pub(super) fn rows(
        &mut self,
        dev: &mut dyn Dev,
        wants: &[Want],
        ahead: bool,
    ) -> Result<Vec<Got>, String> {
        self.fires += 1;
        for w in wants {
            if w.fed.len() > self.positions {
                return Err(format!(
                    "slot {} fires position {} and a seat of the wafer program holds {}",
                    w.slot,
                    w.fed.len() - 1,
                    self.positions
                ));
            }
            let lane = self.lanes.entry(w.slot).or_default();
            lane.fired = self.fires;
            lane.asked = w.fed.len();
            lane.instance = w.instance.or(lane.instance);
        }
        let mut got: Vec<Got> = wants.iter().map(|_| Got::default()).collect();
        self.take(dev, false)?;
        loop {
            // each lane's rows still to run, up to the next whose logits are read
            let mut open: Vec<usize> = Vec::new();
            let mut rows: Vec<(u32, &[u32], bool)> = Vec::new();
            for (i, w) in wants.iter().enumerate() {
                if let Some(&at) = w.logits.get(got[i].logits.len()) {
                    open.push(i);
                    rows.push((w.slot, &w.fed[..=at], true));
                } else if !w.logits.is_empty() || got[i].argmax.is_some() {
                } else if let Some(token) = self.known(w.slot, &w.fed) {
                    got[i].argmax = Some(token);
                } else {
                    open.push(i);
                    rows.push((w.slot, &w.fed, false));
                }
            }
            if rows.is_empty() {
                break;
            }
            if self.out.is_some() {
                self.take(dev, true)?;
                continue;
            }
            rows.truncate(self.seats.len());
            let launch = self.launch(&rows)?;
            // the want its logits are for: the lowest seat's
            let reader = rows
                .iter()
                .zip(&open)
                .filter(|(r, _)| launch.logits && r.2)
                .filter_map(|(r, &i)| Some((self.seat_of(r.0)?, i)))
                .min();
            dev.post(launch)?;
            let logits = self.take(dev, true)?;
            if let (Some((_, i)), Some(logits)) = (reader, logits) {
                got[i].logits.push(logits);
            }
        }
        // the wafer runs on while the rows are answered (a lane that samples must not run on)
        let samples = wants.iter().any(|w| !w.logits.is_empty());
        if ahead && !samples && self.out.is_none() && self.runs_low(wants) {
            let launch = self.launch(&[])?;
            dev.post(launch)?;
        }
        Ok(got)
    }

    /// The launch out landed (waiting for it, or only if it has): its logits where it read them.
    fn take(&mut self, dev: &mut dyn Dev, wait: bool) -> Result<Option<Vec<f32>>, String> {
        if self.out.is_none() {
            return Ok(None);
        }
        match dev.take(wait) {
            None => Ok(None),
            Some(Ok((winners, logits))) => {
                self.land(&winners);
                Ok(logits.into_iter().next())
            }
            Some(Err(e)) => {
                self.lost();
                Err(e)
            }
        }
    }
}

/// What runs a ring's launches, one out at a time.
pub(super) trait Dev {
    fn post(&mut self, launch: wafer::Launch) -> Result<(), String>;
    /// The launch's answer: waiting for it, or `None` if it has not landed.
    fn take(&mut self, wait: bool) -> Option<wafer::Answer>;
}

impl Dev for wafer::Wafer {
    fn post(&mut self, launch: wafer::Launch) -> Result<(), String> {
        self.post_ring(launch)
    }

    fn take(&mut self, wait: bool) -> Option<wafer::Answer> {
        self.take_answer(wait)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tokens' argmax: every token and its place counts.
    fn next(tokens: &[u32]) -> u32 {
        tokens.iter().fold(7u32, |h, &t| {
            h.wrapping_mul(31).wrapping_add(t).rotate_left(5)
        }) % 1000
    }

    /// A ring of sequences that can't go back: a launch starts each at its end or at 0.
    struct Fake {
        seats: Vec<Vec<u32>>,
        positions: usize,
        cap: usize,
        out: Option<wafer::Answer>,
        /// Polls before a posted launch lands.
        slow: usize,
        waits: usize,
        passes: usize,
        launches: usize,
        fail: bool,
        /// The shared pool (positions, rows) and each seat's window; a seat whose cells went
        /// (its window moved or cut short) until it starts again from 0.
        pool: Option<(usize, usize)>,
        wins: Vec<(usize, usize)>,
        lost: Vec<bool>,
    }

    impl Fake {
        fn new(m: usize, positions: usize, cap: usize, slow: usize) -> Fake {
            Fake {
                seats: vec![Vec::new(); m],
                positions,
                cap,
                out: None,
                slow,
                waits: 0,
                passes: 0,
                launches: 0,
                fail: false,
                pool: None,
                wins: Vec::new(),
                lost: vec![false; m],
            }
        }
    }

    impl Dev for Fake {
        fn post(&mut self, l: wafer::Launch) -> Result<(), String> {
            assert!(self.out.is_none() && l.passes > 0 && l.passes <= self.cap);
            (self.waits, self.launches) = (0, self.launches + 1);
            self.passes += l.passes;
            if let (Some((size, rows)), Some(w)) = (self.pool, &l.windows) {
                assert_eq!(w.len(), self.seats.len());
                for (j, &(b, n)) in w.iter().enumerate() {
                    assert!(b % rows == 0 && b + n <= size, "window {j} {w:?}");
                    for (k, &(c, o)) in w.iter().enumerate().skip(j + 1) {
                        assert!(b + n <= c || c + o <= b, "windows {j} and {k} {w:?}");
                    }
                    if let Some(&(a, was)) = self.wins.get(j) {
                        self.lost[j] |= a != b || n < was;
                    }
                }
                self.wins.clone_from(w);
            }
            if std::mem::take(&mut self.fail) {
                self.out = Some(Err("down".into()));
                return Ok(());
            }
            let m = self.seats.len();
            let mut winners = Vec::new();
            for s in 0..l.passes {
                let (j, lap) = (s % m, s / m);
                let p = l.pos[j] + lap;
                assert!(p < self.positions, "seat {j} past its positions");
                // (a seat of no window writes no cells: its winners are no token's)
                let idle = self.pool.is_some() && self.wins[j].1 == 0;
                if idle {
                    self.lost[j] = true;
                    winners.push(1_000_000);
                    continue;
                }
                if self.pool.is_some() {
                    assert!(
                        p < self.wins[j].1,
                        "seat {j} past its window: p {p} wins {:?} pos {:?} passes {}",
                        self.wins,
                        l.pos,
                        l.passes
                    );
                }
                if p == 0 {
                    self.seats[j].clear();
                    self.lost[j] = false;
                }
                assert!(!self.lost[j], "seat {j} ran on over cells it lost");
                assert_eq!(self.seats[j].len(), p, "seat {j} rolled back");
                let token = if lap == 0 {
                    l.first[j]
                } else if let Some(&t) = l.force[j].get(lap - 1) {
                    t
                } else {
                    winners[s - m]
                };
                self.seats[j].push(token);
                winners.push(next(&self.seats[j]));
            }
            let logits = l.logits.then(|| vec![winners[l.passes - 1] as f32]);
            self.out = Some(Ok((winners, logits.into_iter().collect())));
            Ok(())
        }

        fn take(&mut self, wait: bool) -> Option<wafer::Answer> {
            self.waits += 1;
            if wait || self.waits > self.slow {
                self.out.take()
            } else {
                None
            }
        }
    }

    fn want(slot: u32, fed: &[u32], logits: &[usize]) -> Want {
        Want {
            slot,
            instance: Some(InstanceId(u64::from(slot))),
            fed: fed.to_vec(),
            logits: logits.to_vec(),
        }
    }

    /// Every lane's rows are its own tokens' whatever the others do: they join and leave at any
    /// time, fire at their own pace, sample, share slots in turn, and are more than the seats.
    #[test]
    fn a_lanes_rows_are_its_own_tokens_whoever_rides_beside_it() {
        for m in [1usize, 2, 5, 7] {
            for seed in 0..40u32 {
                let mut x = seed * 7919 + 1;
                let mut rng = |n: u32| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    x % n
                };
                let mut positions = 60 + rng(80) as usize;
                // half on a shared pool of whole rows, a row or more
                let pool = (m > 1 && seed % 2 == 1).then(|| {
                    let rows = 8 + rng(24) as usize;
                    ((1 + rng(3 * m as u32) as usize) * rows, rows)
                });
                let mut ring = Ring::new(m, positions, 256, [32, 8, 1][rng(3) as usize], pool);
                positions = ring.positions();
                // (Pie's lanes hold no more of the pool than its pages)
                let pages = ring.pool_pages().map(|(s, p)| (s as usize, p as usize));
                let held = |n: usize| pages.map_or(n, |(s, _)| n.div_ceil(s));
                let room = pages.map_or(usize::MAX, |(_, p)| p);
                let mut dev = Fake::new(m, positions, 256, rng(3) as usize);
                dev.pool = pool;
                // slot -> its tokens, whether it samples, how often it fires
                let mut lanes: HashMap<u32, (Vec<u32>, bool, u32)> = HashMap::new();
                for _ in 0..400 {
                    let slot = rng(m as u32 + 2);
                    let mut joins = false;
                    lanes.entry(slot).or_insert_with(|| {
                        joins = true;
                        let prompt = (0..1 + rng(50)).map(|_| rng(1000)).collect();
                        (prompt, rng(5) == 0, 1 + rng(6))
                    });
                    if !joins && rng(40) == 0 {
                        // cancelled, told to the engine or not
                        lanes.remove(&slot);
                        if rng(2) == 0 {
                            ring.close(InstanceId(u64::from(slot)));
                        }
                    }
                    lanes.retain(|_, l| l.0.len() <= positions);
                    while lanes.values().map(|l| held(l.0.len() + 1)).sum::<usize>() > room {
                        let s = *lanes.iter().max_by_key(|l| (l.1.0.len(), *l.0)).unwrap().0;
                        lanes.remove(&s);
                        ring.close(InstanceId(u64::from(s)));
                    }
                    let mut firing: Vec<u32> = lanes.keys().copied().collect();
                    firing.sort_unstable();
                    firing.retain(|s| rng(lanes[s].2) == 0);
                    let wants: Vec<Want> = firing
                        .iter()
                        .map(|s| {
                            let (fed, samples, _) = &lanes[s];
                            let last = fed.len() - 1;
                            let all: &[usize] = if last > 2 && fed.len() % 9 == 0 {
                                &[1, last]
                            } else {
                                &[last]
                            };
                            want(*s, fed, if *samples { all } else { &[] })
                        })
                        .collect();
                    let got = ring.rows(&mut dev, &wants, true).unwrap();
                    for (w, g) in wants.iter().zip(got) {
                        let best = next(&w.fed);
                        let token = if w.logits.is_empty() {
                            assert_eq!(g.argmax, Some(best), "M {m} seed {seed} slot {}", w.slot);
                            best
                        } else {
                            let rows: Vec<f32> = w
                                .logits
                                .iter()
                                .map(|&p| next(&w.fed[..=p]) as f32)
                                .collect();
                            let read: Vec<f32> = g.logits.iter().flatten().copied().collect();
                            assert_eq!(read, rows, "M {m} seed {seed} slot {}", w.slot);
                            if rng(3) == 0 { rng(1000) } else { best }
                        };
                        lanes.get_mut(&w.slot).unwrap().0.push(token);
                    }
                }
            }
        }
    }

    /// Pie's pages on a pool are its rows, and lanes holding them all decode together, however
    /// they share them, the seats with no lane holding none: six 275-row lanes on Gemma 12B's six
    /// seats (12 rows of 160), one at the most a sequence holds beside short ones or alone there,
    /// and a 2000-row lane beside five short ones on E2B's (32 rows of 128).
    #[test]
    fn lanes_holding_every_page_decode_together() {
        let run = |m: usize, positions: usize, pool: (usize, usize), ends: &[usize]| {
            let mut ring = Ring::new(m, positions, 256, 32, Some(pool));
            let (page, pages) = ring.pool_pages().unwrap();
            let held: usize = ends.iter().map(|e| e.div_ceil(page as usize)).sum();
            assert!(held <= pages as usize, "{ends:?}: {held} pages of {pages}");
            let mut dev = Fake::new(m, ring.positions(), 256, 0);
            dev.pool = Some(pool);
            let mut fed: Vec<Vec<u32>> = (0..ends.len()).map(|s| vec![s as u32 + 1; 19]).collect();
            while fed.iter().zip(ends).any(|(f, &e)| f.len() < e) {
                let wants: Vec<Want> = (0..ends.len())
                    .filter(|&s| fed[s].len() < ends[s])
                    .map(|s| want(s as u32, &fed[s], &[]))
                    .collect();
                for (w, g) in wants.iter().zip(ring.rows(&mut dev, &wants, true).unwrap()) {
                    assert_eq!(g.argmax, Some(next(&w.fed)), "{ends:?} slot {}", w.slot);
                    fed[w.slot as usize].push(g.argmax.unwrap());
                }
            }
            (page, pages, ring.positions())
        };
        let g12 = (1920, 160);
        let six = [275; 6];
        assert_eq!(run(6, 1120, g12, &six), (160, 12, 1120));
        assert_eq!(
            run(6, 1120, g12, &[1120, 100, 100, 100, 100]),
            (160, 12, 1120)
        );
        assert_eq!(run(6, 4096, g12, &[1600, 100, 100]), (160, 12, 1920));
        assert_eq!(run(6, 4096, g12, &[1920]), (160, 12, 1920));
        let e2b = (4096, 128);
        let long = [2000, 100, 100, 100, 100, 100];
        assert_eq!(run(6, 3456, e2b, &long), (128, 32, 3456));
    }

    /// `n` greedy lanes decode together at a lap a row; one that stops asking costs the others
    /// nothing, and its rows are there when it asks again.
    #[test]
    fn a_slow_lane_neither_stalls_the_others_nor_replays() {
        let mut dev = Fake::new(7, 488, 256, 0);
        let mut ring = Ring::new(7, 488, 256, 32, None);
        let mut fed: Vec<Vec<u32>> = (0..7).map(|s| vec![s, 2, 3]).collect();
        for step in 0..300 {
            let asks = |s: usize| s != 3 || step % 50 == 0;
            let wants: Vec<Want> = (0..7)
                .filter(|&s| asks(s))
                .map(|s| want(s as u32, &fed[s], &[]))
                .collect();
            for (w, g) in wants.iter().zip(ring.rows(&mut dev, &wants, true).unwrap()) {
                assert_eq!(g.argmax, Some(next(&w.fed)));
                fed[w.slot as usize].push(g.argmax.unwrap());
            }
        }
        // 300 rows a lane, under two launches' laps run past them
        assert!(dev.passes / 7 <= 302 + 64, "{} laps", dev.passes / 7);
        assert!(dev.launches <= 16, "{} launches", dev.launches);
    }

    /// A lane that samples alone has launches of one pass, with no replay.
    #[test]
    fn a_lane_that_samples_alone_runs_a_pass_a_row() {
        let mut dev = Fake::new(5, 100, 256, 0);
        let mut ring = Ring::new(5, 100, 256, 32, None);
        let mut fed = vec![4, 5, 6];
        for i in 0..50 {
            let last = fed.len() - 1;
            let g = ring
                .rows(&mut dev, &[want(0, &fed, &[last])], true)
                .unwrap();
            assert_eq!(g[0].logits, [[next(&fed) as f32]]);
            fed.push(i);
        }
        assert_eq!(dev.passes, 2 * 5 + 1 + 49);
    }

    /// A cancelled lane's seat is the next lane's; a lane past the positions is refused and the
    /// others go on; a failed launch loses what the seats held, and the lanes' rows replay.
    #[test]
    fn a_cancelled_a_refused_and_a_failed_lane_leave_the_rest() {
        let mut dev = Fake::new(2, 40, 64, 1);
        let mut ring = Ring::new(2, 40, 64, 8, None);
        let (a, b, c) = (vec![1, 2], vec![3], vec![9, 9, 9]);
        ring.rows(&mut dev, &[want(0, &a, &[]), want(1, &b, &[])], true)
            .unwrap();
        let seat = ring.seat_of(1);
        ring.close(InstanceId(1));
        let go = |ring: &mut Ring, dev: &mut Fake| {
            let wants = [want(0, &a, &[]), want(5, &c, &[])];
            ring.rows(dev, &wants, true)
                .map(|g| g.iter().map(|g| g.argmax).collect::<Vec<_>>())
        };
        let right = Ok(vec![Some(next(&a)), Some(next(&c))]);
        assert_eq!(go(&mut ring, &mut dev), right);
        assert_eq!(ring.seat_of(5), seat);
        assert!(
            ring.rows(&mut dev, &[want(0, &[0; 41], &[])], true)
                .is_err()
        );
        assert_eq!(go(&mut ring, &mut dev), right);
        dev.out = None;
        ring.lost();
        dev.fail = true;
        assert!(go(&mut ring, &mut dev).is_err());
        assert_eq!(go(&mut ring, &mut dev), right);
    }
}
