//! Where a MoE layer's experts sit across an expert-parallel group: which
//! logical expert each physical slot holds, and the moves that take one
//! placement to the next. The decision is a pure function of the loads, so
//! every engine that grows expert parallelism shares it, and the engine's
//! expert tier only carries it out (writes the table, copies the planes).
//!
//! A group of `ranks` ranks has `slots` physical slots on each; slot `p` of
//! the group is rank `p / slots`'s local slot `p % slots`. Slots past the
//! logical experts hold replicas of the most loaded ones, so a hot expert's
//! tokens split across the ranks that hold it.

use core::fmt;

/// One layer's placement: the logical expert in each physical slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    pub experts: u32,
    pub ranks: u32,
    pub slots: u32,
    /// `ranks * slots` entries, the logical expert of each physical slot.
    pub held: Vec<u32>,
}

/// Why a group cannot hold a layer's experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unplaceable {
    /// Fewer physical slots than logical experts: some expert would have no
    /// place.
    Short { experts: u32, places: u32 },
    /// A group of no ranks, or ranks of no slots.
    Empty,
}

impl fmt::Display for Unplaceable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Unplaceable::Short { experts, places } => write!(
                f,
                "{experts} experts do not fit {places} expert slots across the group; \
                 raise `expert_replicas` or the expert-parallel size"
            ),
            Unplaceable::Empty => write!(f, "an expert-parallel group holds no expert slots"),
        }
    }
}

/// One plane group's copy: logical `expert` into physical slot `to`, from
/// physical slot `from`, which held it under the placement being left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub expert: u32,
    pub from: u32,
    pub to: u32,
}

impl Placement {
    /// Slots per rank for `experts` over `ranks`, with `replicas` extra slots
    /// on each rank: the logical experts rounded up to a multiple of `ranks`.
    #[must_use]
    pub const fn slots_for(experts: u32, ranks: u32, replicas: u32) -> u32 {
        if ranks == 0 {
            return 0;
        }
        experts.div_ceil(ranks) + replicas
    }

    /// The placement at load: expert `p % experts` in physical slot `p`, so
    /// with no extra slots each rank holds a contiguous run of experts, and
    /// the extra slots hold the first experts again.
    pub fn uniform(experts: u32, ranks: u32, slots: u32) -> Result<Placement, Unplaceable> {
        let places = check(experts, ranks, slots)?;
        Ok(Placement {
            experts,
            ranks,
            slots,
            held: (0..places).map(|p| p % experts).collect(),
        })
    }

    /// The placement `loads` (tokens each logical expert took over the load
    /// window) asks for, reached from `self` by a few edits. An edit either
    /// turns a replica slot (one of an expert held more than once) over to
    /// another expert, one slot rewritten, or trades two experts between
    /// ranks, two. Each edit is the one that leaves the ranks' loads, busiest
    /// first, lowest; edits stop when none lowers them or `limits.moves`
    /// slots are spent, and those past the lowest peak are dropped. Unless
    /// the peak ([`Placement::balance`]) falls by more than `limits.gain`,
    /// `self` is kept: a replica stays where it is until moving it pays.
    #[must_use]
    pub fn rebalance(&self, loads: &[u64], limits: Limits) -> Placement {
        let experts = self.experts as usize;
        let ranks = self.ranks as usize;
        let slots = self.slots as usize;
        let load = |e: usize| loads.get(e).copied().unwrap_or(0) as f64;
        let total: f64 = (0..experts).map(load).sum();
        if ranks < 2 || total == 0.0 {
            return self.clone();
        }
        let mean = total / ranks as f64;
        let near = mean * 1e-9;

        let mut next = self.clone();
        let mut replicas = next.replicas();
        // How many slots of each rank hold each expert, `experts * ranks`.
        let mut count = vec![0u32; experts * ranks];
        for (p, &e) in next.held.iter().enumerate() {
            count[e as usize * ranks + p / slots] += 1;
        }
        let mut weight = next.weights(loads);
        let peak = |w: &[f64]| w.iter().copied().fold(0.0, f64::max);
        let start = peak(&weight);
        // No peak falls below the mean: one within the gain of it stays.
        if start - mean <= limits.gain * mean {
            return self.clone();
        }
        let (mut best, mut kept) = (start, None);

        // Whether ranks' loads `a`, busiest first, sit lower than `b`'s.
        let lower = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .find(|(x, y)| (*x - *y).abs() > near)
                .is_some_and(|(x, y)| x < y)
        };
        let sorted = |w: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.extend_from_slice(w);
            out.sort_by(|a, b| b.total_cmp(a));
        };

        let mut budget = limits.moves;
        let (mut here, mut trial, mut pick) = (Vec::new(), Vec::new(), Vec::new());
        let mut scratch = vec![0.0; ranks];
        while budget > 0 {
            sorted(&weight, &mut here);
            pick.clone_from(&here);
            let mut edit = None;

            // A replica slot of `a` given to `b`, which its rank lacks.
            for p in 0..ranks * slots {
                let (a, r) = (next.held[p] as usize, p / slots);
                if replicas[a] < 2 {
                    continue;
                }
                let (ka, la) = (f64::from(replicas[a]), load(a));
                for b in (0..experts).filter(|&b| count[b * ranks + r] == 0) {
                    let (kb, lb) = (f64::from(replicas[b]), load(b));
                    for (t, w) in scratch.iter_mut().enumerate() {
                        let (ca, cb) = (count[a * ranks + t], count[b * ranks + t]);
                        let (na, nb) = (ca - u32::from(t == r), cb + u32::from(t == r));
                        *w = weight[t]
                            + la * (f64::from(na) / (ka - 1.0) - f64::from(ca) / ka)
                            + lb * (f64::from(nb) / (kb + 1.0) - f64::from(cb) / kb);
                    }
                    if peak(&scratch) > pick[0] + near {
                        continue;
                    }
                    sorted(&scratch, &mut trial);
                    if lower(&trial, &pick) {
                        core::mem::swap(&mut pick, &mut trial);
                        edit = Some((p, b as u32, None));
                    }
                }
            }

            // Two experts traded between ranks, neither landing beside a
            // replica of itself: only where it lowers the loads further than
            // any retarget, which rewrites one slot to its two. Trading `a`
            // on `r` for `b` on `s` shifts `share(b) - share(a)` from `s` to
            // `r`, and the pair's busier rank is lowest for the shift nearest
            // half their gap: for each `a`, the nearest `b` either side.
            if budget >= 2 {
                let share = |e: usize| load(e) / f64::from(replicas[e].max(1));
                let mut theirs: Vec<(f64, usize)> = Vec::with_capacity(slots);
                for r in 0..ranks {
                    for s in r + 1..ranks {
                        theirs.clear();
                        theirs.extend(
                            (s * slots..(s + 1) * slots).map(|q| (share(next.held[q] as usize), q)),
                        );
                        theirs.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
                        let half = (weight[s] - weight[r]) / 2.0;
                        for p in r * slots..(r + 1) * slots {
                            let a = next.held[p] as usize;
                            if count[a * ranks + s] > 0 {
                                continue;
                            }
                            let fits = |&&(_, q): &&(f64, usize)| {
                                count[next.held[q] as usize * ranks + r] == 0
                            };
                            let at = theirs.partition_point(|x| x.0 < share(a) + half);
                            let below = theirs[..at].iter().rev().find(fits);
                            let above = theirs[at..].iter().find(fits);
                            for &(theirs_share, q) in below.into_iter().chain(above) {
                                scratch.copy_from_slice(&weight);
                                let shift = theirs_share - share(a);
                                scratch[r] += shift;
                                scratch[s] -= shift;
                                if peak(&scratch) > pick[0] + near {
                                    continue;
                                }
                                sorted(&scratch, &mut trial);
                                if lower(&trial, &pick) {
                                    core::mem::swap(&mut pick, &mut trial);
                                    edit = Some((p, next.held[q], Some(q)));
                                }
                            }
                        }
                    }
                }
            }

            let Some((p, b, traded)) = edit else {
                break;
            };
            let a = next.held[p];
            let mut set = |slot: usize, e: u32| {
                let old = next.held[slot] as usize;
                count[old * ranks + slot / slots] -= 1;
                count[e as usize * ranks + slot / slots] += 1;
                replicas[old] -= 1;
                replicas[e as usize] += 1;
                next.held[slot] = e;
            };
            match traded {
                Some(q) => {
                    set(p, b);
                    set(q, a);
                    budget -= 2;
                }
                None => {
                    set(p, b);
                    budget -= 1;
                }
            }
            weight = next.weights(loads);
            if peak(&weight) < best - near {
                best = peak(&weight);
                kept = Some(next.clone());
            }
        }

        match kept {
            Some(placement) if start - best > limits.gain * mean => placement,
            _ => self.clone(),
        }
    }

    /// Each rank's load under `loads`: its slots' experts' tokens, an
    /// expert's split evenly between its replicas.
    #[must_use]
    pub fn weights(&self, loads: &[u64]) -> Vec<f64> {
        let replicas = self.replicas();
        let slots = self.slots as usize;
        (0..self.ranks as usize)
            .map(|r| {
                self.held[r * slots..(r + 1) * slots]
                    .iter()
                    .map(|&e| {
                        let load = loads.get(e as usize).copied().unwrap_or(0) as f64;
                        load / f64::from(replicas[e as usize])
                    })
                    .sum()
            })
            .collect()
    }

    /// How far `loads` leave the ranks from even: the busiest rank's load
    /// over the mean, 1 when even (or when there is no load).
    #[must_use]
    pub fn balance(&self, loads: &[u64]) -> f64 {
        let weight = self.weights(loads);
        let total: f64 = weight.iter().sum();
        if total == 0.0 {
            return 1.0;
        }
        weight.iter().copied().fold(0.0, f64::max) * weight.len() as f64 / total
    }

    /// Whether rank `rank` holds logical expert `expert`.
    #[must_use]
    pub fn on(&self, rank: u32, expert: u32) -> bool {
        let base = (rank * self.slots) as usize;
        self.held[base..base + self.slots as usize].contains(&expert)
    }

    /// The physical slots holding logical expert `expert`, ascending.
    #[must_use]
    pub fn places(&self, expert: u32) -> Vec<u32> {
        (0..self.held.len() as u32)
            .filter(|&p| self.held[p as usize] == expert)
            .collect()
    }

    /// The table a dispatch reads: `experts` rows of `stride` i32s, row `e`
    /// its replica count then its physical slots ascending, the rest zero.
    /// `stride` is fixed at load ([`Placement::stride_for`]), so a rebalance
    /// rewrites the table's contents in place.
    #[must_use]
    pub fn table(&self, stride: u32) -> Vec<i32> {
        let stride = stride as usize;
        let mut out = vec![0i32; self.experts as usize * stride];
        for (p, &e) in self.held.iter().enumerate() {
            let row = &mut out[e as usize * stride..(e as usize + 1) * stride];
            let count = row[0] as usize;
            if 1 + count < stride {
                row[1 + count] = p as i32;
                row[0] += 1;
            }
        }
        out
    }

    /// A table row's width for a group where an expert holds at most
    /// `places` slots: the count and that many slots, at least two. The most
    /// any expert can hold is every slot past the other experts' one each,
    /// `ranks * slots - experts + 1`.
    #[must_use]
    pub const fn stride_for(places: u32) -> u32 {
        if places + 1 < 2 { 2 } else { places + 1 }
    }

    /// How many slots hold each logical expert.
    #[must_use]
    pub fn replicas(&self) -> Vec<u32> {
        let mut out = vec![0u32; self.experts as usize];
        for &e in &self.held {
            out[e as usize] += 1;
        }
        out
    }

    /// The copies that take `self` to `next`, in an order that runs as
    /// written: a slot is overwritten only once every copy reading it is
    /// done. A copy reads a slot that holds its expert under `self`, one that
    /// keeps it under `next` and sits on the destination's rank first. Where
    /// the copies form a cycle (two slots trading experts), one slot's old
    /// expert is first copied `to` [`SPARE`], a slot outside the placement,
    /// and the copies that read that slot read the spare instead; a cycle is
    /// broken only once the spare's readers of the one before are done.
    #[must_use]
    pub fn moves(&self, next: &Placement) -> Vec<Move> {
        let rank_of = |p: u32| p / self.slots;
        let mut pending: Vec<Move> = (0..self.held.len() as u32)
            .filter(|&p| self.held[p as usize] != next.held[p as usize])
            .map(|to| {
                let expert = next.held[to as usize];
                let places = self.places(expert);
                let stays = |p: &&u32| next.held[**p as usize] == expert;
                let from = places
                    .iter()
                    .filter(stays)
                    .find(|&&p| rank_of(p) == rank_of(to))
                    .or_else(|| places.iter().find(stays))
                    .or_else(|| places.iter().find(|&&p| rank_of(p) == rank_of(to)))
                    .or(places.first())
                    .copied()
                    .unwrap_or(to);
                Move { expert, from, to }
            })
            .collect();

        let mut out = Vec::with_capacity(pending.len() + 1);
        while !pending.is_empty() {
            let free = pending
                .iter()
                .position(|m| !pending.iter().any(|o| o.from == m.to));
            if let Some(at) = free {
                out.push(pending.remove(at));
                continue;
            }
            let to = pending[0].to;
            out.push(Move {
                expert: self.held[to as usize],
                from: to,
                to: SPARE,
            });
            for m in &mut pending {
                if m.from == to {
                    m.from = SPARE;
                }
            }
        }
        out
    }
}

/// How far one [`Placement::rebalance`] may go.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// The most slots it rewrites in the layer.
    pub moves: u32,
    /// How far it must lower the layer's peak, the busiest rank's load over
    /// the mean ([`Placement::balance`]), before it moves anything.
    pub gain: f64,
}

impl Limits {
    /// Two trades a layer, as the expert offload tier makes at most two
    /// swaps a fire, and only for a peak lowered by a twentieth of the mean.
    pub const DEFAULT: Limits = Limits {
        moves: 4,
        gain: 0.05,
    };
}

impl Default for Limits {
    fn default() -> Limits {
        Limits::DEFAULT
    }
}

/// The `from` of a [`Move`] that reads the spare slot (see
/// [`Placement::moves`]).
pub const SPARE: u32 = u32::MAX;

fn check(experts: u32, ranks: u32, slots: u32) -> Result<u32, Unplaceable> {
    if ranks == 0 || slots == 0 || experts == 0 {
        return Err(Unplaceable::Empty);
    }
    let places = ranks * slots;
    if places < experts {
        return Err(Unplaceable::Short { experts, places });
    }
    Ok(places)
}

/// The loads a rebalance reads: each logical expert's received tokens summed
/// over the last `steps` steps.
#[derive(Debug, Clone)]
pub struct Window {
    steps: usize,
    ring: Vec<Vec<u32>>,
    next: usize,
    sum: Vec<u64>,
}

impl Window {
    #[must_use]
    pub fn new(experts: u32, steps: usize) -> Window {
        Window {
            steps: steps.max(1),
            ring: Vec::new(),
            next: 0,
            sum: vec![0; experts as usize],
        }
    }

    /// One step's tokens per logical expert.
    pub fn push(&mut self, loads: &[u32]) {
        let step: Vec<u32> = (0..self.sum.len())
            .map(|e| loads.get(e).copied().unwrap_or(0))
            .collect();
        for (s, &l) in self.sum.iter_mut().zip(&step) {
            *s += u64::from(l);
        }
        if self.ring.len() < self.steps {
            self.ring.push(step);
            return;
        }
        let old = std::mem::replace(&mut self.ring[self.next], step);
        for (s, l) in self.sum.iter_mut().zip(old) {
            *s -= u64::from(l);
        }
        self.next = (self.next + 1) % self.steps;
    }

    #[must_use]
    pub fn loads(&self) -> &[u64] {
        &self.sum
    }
}

#[cfg(test)]
mod tests {
    use super::{Limits, Move, Placement, SPARE, Unplaceable, Window};

    const LIMITS: Limits = Limits::DEFAULT;

    fn apply(from: &Placement, moves: &[Move]) -> Vec<u32> {
        let mut held = from.held.clone();
        let mut spare = None;
        for m in moves {
            let read = if m.from == SPARE {
                spare
            } else {
                Some(held[m.from as usize])
            };
            assert_eq!(
                read,
                Some(m.expert),
                "{m:?} reads a slot already overwritten"
            );
            if m.to == SPARE {
                spare = Some(m.expert);
            } else {
                held[m.to as usize] = m.expert;
            }
        }
        held
    }

    #[test]
    fn the_load_placement_is_contiguous_and_replicates_the_first() {
        let p = Placement::uniform(8, 4, 2).unwrap();
        assert_eq!(p.held, [0, 1, 2, 3, 4, 5, 6, 7]);
        let p = Placement::uniform(6, 2, 4).unwrap();
        assert_eq!(p.held, [0, 1, 2, 3, 4, 5, 0, 1]);
        assert_eq!(Placement::slots_for(6, 4, 1), 3);
        assert_eq!(
            Placement::uniform(9, 4, 2),
            Err(Unplaceable::Short {
                experts: 9,
                places: 8
            })
        );
        assert_eq!(Placement::uniform(4, 0, 2), Err(Unplaceable::Empty));
    }

    /// A skewed layer's loads: expert `e` of `experts` a Zipf weight,
    /// shuffled by `seed` so the hot ones are not the first.
    fn skewed(experts: u32, seed: u64) -> Vec<u64> {
        let mut order: Vec<u64> = (0..u64::from(experts)).collect();
        let mut x = seed;
        for i in (1..order.len()).rev() {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            order.swap(i, (x >> 33) as usize % (i + 1));
        }
        order.iter().map(|&rank| 100_000 / (rank + 1)).collect()
    }

    /// The slots `next` holds a different expert in than `from`.
    fn rewritten(from: &Placement, next: &Placement) -> u32 {
        (from.held.iter().zip(&next.held))
            .filter(|(a, b)| a != b)
            .count() as u32
    }

    /// Rebalances on `loads` until a placement holds, at most `rounds`
    /// times; every placement on the way.
    fn settle(p: &Placement, loads: &[u64], rounds: usize) -> Vec<Placement> {
        let mut out = vec![p.clone()];
        for _ in 0..rounds {
            let next = out.last().unwrap().rebalance(loads, LIMITS);
            if &next == out.last().unwrap() {
                return out;
            }
            out.push(next);
        }
        panic!("no placement held after {rounds} rebalances");
    }

    #[test]
    fn a_hot_expert_gets_the_replicas_and_spreads_across_ranks() {
        let p = Placement::uniform(4, 4, 2).unwrap();
        let loads = [300, 10, 10, 10];
        let last = settle(&p, &loads, 8).pop().unwrap();
        assert!(last.replicas()[0] >= 4, "{:?}", last.replicas());
        assert!((0..4).all(|r| last.on(r, 0)));
    }

    #[test]
    fn a_second_rebalance_on_the_same_loads_moves_nothing() {
        let p = Placement::uniform(8, 4, 3).unwrap();
        for loads in [
            vec![90, 1, 1, 1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1, 1, 1, 90],
            vec![40, 30, 20, 10, 10, 20, 30, 40],
        ] {
            let next = p.rebalance(&loads, LIMITS);
            assert_ne!(next, p, "{loads:?} moves something");
            assert_eq!(next.rebalance(&loads, LIMITS), next, "{loads:?}");
        }
        // And wherever the rebalances settle, on a layer too skewed for one.
        let p = Placement::uniform(64, 2, 34).unwrap();
        for seed in 0..8 {
            let loads = skewed(64, seed);
            let last = settle(&p, &loads, 64).pop().unwrap();
            assert_eq!(last.rebalance(&loads, LIMITS), last, "seed {seed}");
        }
    }

    #[test]
    fn a_rebalance_moves_only_what_lowers_the_peak_past_the_gain() {
        for (experts, ranks, slots) in [(64, 2, 34), (64, 4, 18), (16, 4, 5)] {
            let p = Placement::uniform(experts, ranks, slots).unwrap();
            for seed in 0..8 {
                let loads = skewed(experts, seed);
                for pair in settle(&p, &loads, 64).windows(2) {
                    let (was, now) = (pair[0].balance(&loads), pair[1].balance(&loads));
                    assert!(
                        was - now > LIMITS.gain,
                        "{experts}/{ranks}x{slots} seed {seed}: {was} to {now}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_rebalance_rewrites_at_most_its_budget_of_slots() {
        for (experts, ranks, slots) in [(256, 2, 130), (64, 4, 18), (8, 4, 3)] {
            let p = Placement::uniform(experts, ranks, slots).unwrap();
            for seed in 0..8 {
                let loads = skewed(experts, seed);
                for pair in settle(&p, &loads, 256).windows(2) {
                    assert!(rewritten(&pair[0], &pair[1]) <= LIMITS.moves);
                    let moves = pair[0].moves(&pair[1]);
                    assert_eq!(apply(&pair[0], &moves), pair[1].held);
                }
            }
        }
    }

    #[test]
    fn a_skewed_load_converges_over_several_rebalances() {
        // Every hot expert on the first rank.
        let p = Placement::uniform(128, 2, 66).unwrap();
        let loads: Vec<u64> = (0..128).map(|e| if e < 64 { 100 } else { 10 }).collect();
        let path = settle(&p, &loads, 64);
        let (first, last) = (p.balance(&loads), path.last().unwrap().balance(&loads));
        assert!(path.len() > 2, "one rebalance is capped short of balance");
        assert!(first > 1.5 && last < 1.0 + LIMITS.gain, "{first} to {last}");
        // The window's noise about the settled loads moves nothing.
        let mut x = 1u64;
        let noisy: Vec<u64> = loads
            .iter()
            .map(|&l| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                l + l * ((x >> 33) % 3) / 100
            })
            .collect();
        let last = path.last().unwrap();
        assert_eq!(&last.rebalance(&noisy, LIMITS), last);
    }

    #[test]
    fn a_placement_no_packing_improves_on_is_kept() {
        // Two ranks of three: the hot expert's two replicas already split it
        // evenly; a third would leave one rank holding two of them.
        let p = Placement::uniform(4, 2, 3).unwrap();
        assert_eq!(p.rebalance(&[100, 10, 10, 10], LIMITS), p);
    }

    #[test]
    fn hot_experts_on_one_rank_are_parted() {
        let p = Placement::uniform(4, 2, 2).unwrap();
        let next = p.rebalance(&[100, 90, 10, 10], LIMITS);
        assert!(next.on(0, 0) != next.on(0, 1));
        // Expert 1 trades places with one of rank 1's: two copies and the
        // spare between them.
        let moves = p.moves(&next);
        assert_eq!(moves.len(), 3);
        assert_eq!(apply(&p, &moves), next.held);
    }

    #[test]
    fn balanced_loads_move_nothing() {
        let p = Placement::uniform(8, 4, 2).unwrap();
        let next = p.rebalance(&[5; 8], LIMITS);
        assert_eq!(next, p);
        assert!(p.moves(&next).is_empty());
    }

    #[test]
    fn the_ranks_end_near_even() {
        let p = Placement::uniform(16, 4, 5).unwrap();
        let loads: Vec<u64> = (0..16).map(|e| 1 + e * e * 7).collect();
        let next = p.rebalance(&loads, LIMITS);
        let replicas = next.replicas();
        let per_rank: Vec<u64> = (0..4)
            .map(|r| {
                next.held[r * 5..r * 5 + 5]
                    .iter()
                    .map(|&e| loads[e as usize] / u64::from(replicas[e as usize]))
                    .sum()
            })
            .collect();
        let (lo, hi) = (
            *per_rank.iter().min().unwrap(),
            *per_rank.iter().max().unwrap(),
        );
        assert!(
            hi - lo <= loads[15] / u64::from(replicas[15]),
            "{per_rank:?}"
        );
    }

    #[test]
    fn the_moves_take_one_placement_to_the_next() {
        let p = Placement::uniform(8, 4, 3).unwrap();
        for loads in [
            vec![90, 1, 1, 1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1, 1, 1, 90],
            vec![40, 30, 20, 10, 10, 20, 30, 40],
        ] {
            let next = p.rebalance(&loads, LIMITS);
            let moves = p.moves(&next);
            assert_eq!(apply(&p, &moves), next.held, "{loads:?}");
            let again = next.rebalance(&loads, LIMITS);
            assert_eq!(again, next, "a second rebalance on the same loads holds");
        }
    }

    #[test]
    fn a_trade_reads_the_spare_once() {
        let p = Placement {
            experts: 2,
            ranks: 2,
            slots: 1,
            held: vec![0, 1],
        };
        let next = Placement {
            held: vec![1, 0],
            ..p.clone()
        };
        let moves = p.moves(&next);
        assert_eq!(moves.iter().filter(|m| m.to == SPARE).count(), 1);
        assert_eq!(apply(&p, &moves), next.held);
    }

    #[test]
    fn the_table_lists_each_experts_slots() {
        let p = Placement::uniform(3, 2, 2).unwrap();
        assert_eq!(p.held, [0, 1, 2, 0]);
        assert_eq!(
            p.table(Placement::stride_for(2)),
            [2, 0, 3, 1, 1, 0, 1, 2, 0]
        );
        let p = Placement::uniform(4, 2, 2).unwrap();
        assert_eq!(p.table(2), [1, 0, 1, 1, 1, 2, 1, 3]);
    }

    #[test]
    fn the_window_forgets_old_steps() {
        let mut w = Window::new(2, 2);
        w.push(&[1, 10]);
        w.push(&[2, 20]);
        assert_eq!(w.loads(), [3, 30]);
        w.push(&[4, 40]);
        assert_eq!(w.loads(), [6, 60]);
    }
}
