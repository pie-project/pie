//! The host replay of a beam group's step (`new-memory/designs/BEAM-PAGE-LISTS.md`
//! §5–6): the same algorithm the engines' head kernels run, as a pure
//! function, written from the spec. Off the serving path: each backend's
//! test checks its device step against it fire by fire.

/// What `BEAM-PAGE-LISTS.md` §7 calls each lane's fault.
pub mod fault {
    pub const PAST_RESERVE: u32 = 1;
    pub const SHORT: u32 = 2;
    pub const BELOW_FLOOR: u32 = 3;
    pub const COPY_BOUNDS: u32 = 4;
    pub const PARENT: u32 = 5;
    pub const RESERVE_EXHAUSTED: u32 = 6;
}

/// A beam group's resident state (§4), laid for `cap` columns and pool
/// pages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    pub lanes: u32,
    pub cap: u32,
    pub frames: u32,
    /// Fires run: 0 until the first primes.
    pub fired: u32,
    /// The pool's pages at the last fire: it grows a column at a time.
    pub pool: u32,
    /// `map[b * cap + j]`: the pool page of beam `b`'s column `j`.
    pub map: Vec<u32>,
    pub free: Vec<bool>,
    /// `pending[k * cap + q]`: page `q` freed by a fire `t` with `t % frames == k`.
    pub pending: Vec<bool>,
}

impl State {
    /// A group of `lanes` beams not yet primed.
    #[must_use]
    pub fn new(lanes: u32, cap: u32, frames: u32) -> State {
        let frames = frames.max(1);
        State {
            lanes,
            cap,
            frames,
            fired: 0,
            pool: 0,
            map: vec![0; (lanes * cap) as usize],
            free: vec![false; cap as usize],
            pending: vec![false; (frames * cap) as usize],
        }
    }
}

/// One fire's numbers (§3): the page size, the set's frozen slots, its pool
/// pages, and the slot every beam writes.
#[derive(Clone, Copy, Debug)]
pub struct Fire {
    pub page: u32,
    pub folded: u32,
    pub reserve: u32,
    pub slot: u32,
}

/// What one fire's step does: each beam's copy `[src, dst, tokens]`
/// (`tokens` 0: none), its list's live columns (`0..=c`), and its fault (0:
/// none).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub copies: Vec<[u32; 3]>,
    pub lists: Vec<Vec<u32>>,
    pub faults: Vec<u32>,
}

/// Runs one fire of the group over `state`: primes it on its first fire,
/// else reorders it by `parents` (§6.1–6.3); returns the copies (§6.4) and
/// the lists the fire's tables resolve over (§6.5).
#[must_use]
pub fn step(state: &mut State, parents: &[u32], at: Fire) -> Step {
    let (b_n, cap) = (state.lanes as usize, state.cap as usize);
    let s = at.page.max(1);
    let f0 = at.folded / s;
    let tail = at.folded % s;
    let c = (at.slot / s).saturating_sub(f0);
    let o = at.slot % s;
    let cp = if at.slot == 0 {
        0
    } else {
        ((at.slot - 1) / s).saturating_sub(f0)
    };
    let q_n = at.reserve.min(state.cap);
    let mut faults = vec![0u32; b_n];
    let mut copies = vec![[0u32; 3]; b_n];
    let raise = |faults: &mut Vec<u32>, b: usize, code: u32| faults[b] = faults[b].max(code);

    if state.fired == 0 {
        // §5.
        for b in 0..b_n {
            for j in 0..cap {
                state.map[b * cap + j] = if j == 0 { b as u32 } else { 0 };
            }
        }
        for q in 0..cap {
            let q32 = q as u32;
            state.free[q] = q32 < q_n && q32 >= state.lanes && !(q32 == 0 && tail > 0);
        }
        state.pending.iter_mut().for_each(|p| *p = false);
        for (b, copy) in copies.iter_mut().enumerate() {
            if b > 0 && tail > 0 && o > 0 {
                *copy = [0, b as u32, tail];
            }
            if c != 0 {
                raise(&mut faults, b, fault::SHORT);
            }
            if state.lanes > q_n {
                raise(&mut faults, b, fault::RESERVE_EXHAUSTED);
            }
        }
    } else {
        // §3: pages that joined the pool since the last fire are free.
        for q in state.pool.min(q_n)..q_n {
            state.free[q as usize] = !(q == 0 && tail > 0);
        }
        // §6.1.
        let parent: Vec<usize> = (0..b_n)
            .map(|b| match parents.get(b) {
                Some(&p) if (p as usize) < b_n => p as usize,
                _ => {
                    raise(&mut faults, b, fault::PARENT);
                    b
                }
            })
            .collect();
        // §6.2.
        let mut need = vec![false; b_n];
        let mut rank = vec![0usize; b_n];
        let mut needed = 0usize;
        for b in 0..b_n {
            let first = (0..b).all(|e| parent[e] != parent[b]);
            need[b] = o == 0 || !first;
            rank[b] = needed;
            needed += usize::from(need[b]);
        }
        let hand: Vec<u32> = (0..cap)
            .filter(|&q| state.free[q])
            .take(needed)
            .map(|q| q as u32)
            .collect();
        let fresh: Vec<Option<u32>> = (0..b_n)
            .map(|b| {
                if !need[b] {
                    return None;
                }
                let got = hand.get(rank[b]).copied();
                if got.is_none() {
                    raise(&mut faults, b, fault::RESERVE_EXHAUSTED);
                }
                got
            })
            .collect();
        let old = state.map.clone();
        let live = (c as usize + 1).min(cap);
        for b in 0..b_n {
            for j in 0..live {
                let inherited = old[parent[b] * cap + j];
                state.map[b * cap + j] = match fresh[b] {
                    Some(page) if j == c as usize => page,
                    _ => inherited,
                };
            }
        }
        for b in 0..b_n {
            if let Some(page) = fresh[b]
                && o > 0
            {
                copies[b] = [old[parent[b] * cap + c as usize], page, o];
            }
        }
        // §6.3.
        let k = (state.fired % state.frames) as usize;
        for q in 0..cap {
            let taken = fresh.contains(&Some(q as u32));
            let mut f = (state.free[q] && !taken) || state.pending[k * cap + q];
            if (q == 0 && tail > 0) || q as u32 >= q_n {
                f = false;
            }
            state.free[q] = f;
        }
        for q in 0..cap {
            state.pending[k * cap + q] = false;
        }
        let held_cols = (cp as usize + 1).min(cap);
        for p in 0..b_n {
            for j in 0..held_cols {
                let page = old[p * cap + j];
                let held = (0..b_n).any(|b| state.map[b * cap + j] == page);
                if !held && (page as usize) < cap && !(page == 0 && tail > 0) {
                    state.pending[k * cap + page as usize] = true;
                }
            }
        }
    }
    // §6.5's bounds, over each beam's live columns and its write.
    let live = (c as usize + 1).min(cap);
    let lists: Vec<Vec<u32>> = (0..b_n)
        .map(|b| state.map[b * cap..b * cap + live].to_vec())
        .collect();
    for (b, list) in lists.iter().enumerate() {
        if c as usize >= cap {
            raise(&mut faults, b, fault::SHORT);
            continue;
        }
        if list.iter().any(|&q| q >= q_n) {
            raise(&mut faults, b, fault::PAST_RESERVE);
        }
        let q = list[c as usize];
        let slot = u64::from(f0 + q) * u64::from(s) + u64::from(o);
        if q >= q_n || slot < u64::from(at.folded) || slot >= u64::from(f0 + q_n) * u64::from(s) {
            raise(&mut faults, b, fault::BELOW_FLOOR);
        }
    }
    // §6.4's bounds.
    for (b, copy) in copies.iter().enumerate() {
        let [src, dst, tokens] = *copy;
        if tokens == 0 || src == dst {
            continue;
        }
        if src >= q_n || dst >= q_n || tokens > s || u64::from(f0 + dst) * u64::from(s) < u64::from(at.folded)
        {
            raise(&mut faults, b, fault::COPY_BOUNDS);
        }
    }
    state.fired += 1;
    state.pool = q_n;
    Step {
        copies,
        lists,
        faults,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator for the parents of each fire.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    fn run(lanes: u32, folded: u32, steps: u32, seed: u64) {
        let page = 16;
        let frames = 2;
        let f0 = folded / page;
        let columns = (folded + steps - 1) / page - f0 + 1;
        let reserve = lanes * columns + lanes + lanes * frames + u32::from(folded % page > 0);
        let mut state = State::new(lanes, reserve + 8, frames);
        let mut rng = Lcg(seed);
        for t in 0..steps {
            let parents: Vec<u32> = if rng.next() % 3 == 0 {
                (0..lanes).collect()
            } else {
                let mut p: Vec<u32> = (0..lanes).map(|_| rng.next() % lanes).collect();
                p.sort_unstable();
                p
            };
            let step = step(
                &mut state,
                &parents,
                Fire {
                    page,
                    folded,
                    reserve,
                    slot: folded + t,
                },
            );
            assert!(
                step.faults.iter().all(|&f| f == 0),
                "fire {t}: faults {:?} (lanes {lanes}, folded {folded})",
                step.faults
            );
            // No list names a free page, and no two beams write one page
            // in this fire's column.
            let c = ((folded + t) / page - f0) as usize;
            for (b, list) in step.lists.iter().enumerate() {
                for &q in list {
                    assert!(!state.free[q as usize], "fire {t}: beam {b} names free page {q}");
                }
                for (e, other) in step.lists.iter().enumerate().take(b) {
                    assert_ne!(list[c], other[c], "fire {t}: beams {e} and {b} write one page");
                }
            }
        }
    }

    #[test]
    fn the_replay_holds_its_invariants() {
        for lanes in [1, 2, 4, 8] {
            for folded in [32, 37] {
                for seed in 0..4 {
                    run(lanes, folded, 3 * 16 + 9, seed);
                }
            }
        }
    }
}
