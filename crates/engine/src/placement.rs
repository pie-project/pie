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
    /// window) asks for, kept as close to `self` as it allows. Each expert
    /// keeps one slot; each extra slot goes, one at a time, to the expert
    /// with the most load per replica; the replicas are then packed onto
    /// ranks heaviest first, each onto the least loaded rank with a free slot
    /// that holds no replica of it yet. A packing whose busiest rank is no
    /// lighter than `self`'s keeps `self`. Each packed group goes to the rank
    /// that holds most of it, and on a rank an expert keeps the slot it held,
    /// so only what changes rank moves.
    #[must_use]
    pub fn rebalance(&self, loads: &[u64]) -> Placement {
        let experts = self.experts as usize;
        let ranks = self.ranks as usize;
        let slots = self.slots as usize;
        let load = |e: usize| loads.get(e).copied().unwrap_or(0);

        let mut replicas = vec![1u32; experts];
        for _ in experts..ranks * slots {
            let hottest = (0..experts)
                .max_by(|&a, &b| {
                    let (la, lb) = (
                        load(a) * u64::from(replicas[b]),
                        load(b) * u64::from(replicas[a]),
                    );
                    la.cmp(&lb).then(b.cmp(&a))
                })
                .unwrap_or(0);
            replicas[hottest] += 1;
        }

        let mut pieces: Vec<(u64, usize)> = (0..experts)
            .flat_map(|e| {
                let share = load(e) / u64::from(replicas[e]);
                (0..replicas[e]).map(move |_| (share, e))
            })
            .collect();
        pieces.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        let mut on_rank: Vec<Vec<u32>> = vec![Vec::new(); ranks];
        let mut weight = vec![0u64; ranks];
        for (share, e) in pieces {
            let fits = |r: usize, fresh: bool| {
                on_rank[r].len() < slots && (!fresh || !on_rank[r].contains(&(e as u32)))
            };
            let pick = |fresh: bool| {
                (0..ranks)
                    .filter(|&r| fits(r, fresh))
                    .min_by_key(|&r| (weight[r], usize::from(!self.on(r as u32, e as u32)), r))
            };
            let r = pick(true).or_else(|| pick(false)).unwrap_or(0);
            on_rank[r].push(e as u32);
            weight[r] += share;
        }

        let peak = |held: &[u32], replicas: &[u32]| {
            (0..ranks)
                .map(|r| {
                    held[r * slots..(r + 1) * slots]
                        .iter()
                        .map(|&e| load(e as usize) / u64::from(replicas[e as usize].max(1)))
                        .sum::<u64>()
                })
                .max()
                .unwrap_or(0)
        };
        if weight.iter().copied().max().unwrap_or(0) >= peak(&self.held, &self.replicas()) {
            return self.clone();
        }

        // The ranks are interchangeable: each packed group goes to the rank
        // that already holds most of it.
        let overlap =
            |g: usize, r: usize| on_rank[g].iter().filter(|&&e| self.on(r as u32, e)).count();
        let mut rank_of = vec![usize::MAX; ranks];
        let mut taken = vec![false; ranks];
        for _ in 0..ranks {
            let best = (0..ranks)
                .filter(|&g| rank_of[g] == usize::MAX)
                .flat_map(|g| (0..ranks).filter(|&r| !taken[r]).map(move |r| (g, r)))
                .max_by_key(|&(g, r)| (overlap(g, r), core::cmp::Reverse((g, r))));
            if let Some((g, r)) = best {
                rank_of[g] = r;
                taken[r] = true;
            }
        }

        let mut held = vec![u32::MAX; ranks * slots];
        for (g, mine) in on_rank.into_iter().enumerate() {
            let base = rank_of[g] * slots;
            let mut rest = Vec::new();
            for e in mine {
                let kept =
                    (0..slots).find(|&s| self.held[base + s] == e && held[base + s] == u32::MAX);
                match kept {
                    Some(s) => held[base + s] = e,
                    None => rest.push(e),
                }
            }
            let free = (0..slots)
                .filter(|&s| held[base + s] == u32::MAX)
                .collect::<Vec<_>>();
            for (s, e) in free.into_iter().zip(rest) {
                held[base + s] = e;
            }
        }
        Placement {
            experts: self.experts,
            ranks: self.ranks,
            slots: self.slots,
            held,
        }
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
    use super::{Move, Placement, SPARE, Unplaceable, Window};

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

    #[test]
    fn a_hot_expert_gets_the_replicas_and_spreads_across_ranks() {
        let p = Placement::uniform(4, 4, 2).unwrap();
        let next = p.rebalance(&[300, 10, 10, 10]);
        assert_eq!(next.replicas(), [5, 1, 1, 1]);
        assert!((0..4).all(|r| next.on(r, 0)));
    }

    #[test]
    fn a_placement_no_packing_improves_on_is_kept() {
        // Two ranks of three: the hot expert's two replicas already split it
        // evenly; a third would leave one rank holding two of them.
        let p = Placement::uniform(4, 2, 3).unwrap();
        assert_eq!(p.rebalance(&[100, 10, 10, 10]), p);
    }

    #[test]
    fn hot_experts_on_one_rank_are_parted() {
        let p = Placement::uniform(4, 2, 2).unwrap();
        let next = p.rebalance(&[100, 90, 10, 10]);
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
        let next = p.rebalance(&[5; 8]);
        assert_eq!(next, p);
        assert!(p.moves(&next).is_empty());
    }

    #[test]
    fn the_ranks_end_near_even() {
        let p = Placement::uniform(16, 4, 5).unwrap();
        let loads: Vec<u64> = (0..16).map(|e| 1 + e * e * 7).collect();
        let next = p.rebalance(&loads);
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
            let next = p.rebalance(&loads);
            let moves = p.moves(&next);
            assert_eq!(apply(&p, &moves), next.held, "{loads:?}");
            let again = next.rebalance(&loads);
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
    fn the_window_forgets_old_steps() {
        let mut w = Window::new(2, 2);
        w.push(&[1, 10]);
        w.push(&[2, 20]);
        assert_eq!(w.loads(), [3, 30]);
        w.push(&[4, 40]);
        assert_eq!(w.loads(), [6, 60]);
    }
}
