//! The guest term (REDESIGN §6): what a fire takes of the span for the
//! guest programs it carries, counted for the fire from each open pair's
//! [`FireFacts`]. The runtime sets it aside at every rung and an engine
//! that backs its own carve counts the same, both from this one function.

use super::{FireFacts, LaidLanes};

/// One open (program, extents) pair of a rung's guest term: the facts the
/// engine stated for it and the instances of it that can be in a fire
/// (bound, or binding).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FirePair {
    pub facts: FireFacts,
    pub live: u32,
}

impl FirePair {
    /// The most members of the pair that lay a lane more: its live
    /// instances, under its own cap where it states one.
    fn members(&self) -> u64 {
        let live = u64::from(self.live);
        if self.facts.max_lanes == 0 {
            live
        } else {
            live.min(u64::from(self.facts.max_lanes))
        }
    }

    /// The pair as a table of fires of up to `most` lanes reads it: the
    /// fewest live members that give the same table, `None` where it takes
    /// nothing. Laid member by member, its members up to `most`; laid by
    /// powers of two, the last breakpoint at or under them (1, 2, then one
    /// past each power of two): members between two breakpoints lay what
    /// the lower one does.
    fn reduced(&self, most: u32) -> Option<FirePair> {
        let members = self.members().min(u64::from(most));
        if members == 0 || (self.facts.fixed == 0 && self.facts.a_lane == 0) {
            return None;
        }
        let members = match self.facts.laid {
            LaidLanes::Exact => members,
            LaidLanes::PowerOfTwo if members <= 2 => members,
            // One past the largest power of two under `members`.
            LaidLanes::PowerOfTwo => (1u64 << (members - 1).ilog2()) + 1,
        };
        Some(FirePair {
            facts: self.facts,
            live: u32::try_from(members).unwrap_or(u32::MAX),
        })
    }
}

/// The guest term (REDESIGN §6): what a fire of up to `lanes` lanes takes
/// of the span beside the model's activation, whatever members it carries.
/// It is counted for a fire, not for a pair: a fire holds `lanes` members
/// in all, so with the open pairs `i`
///
/// ```text
/// term(lanes) = max Σ fixed_i + a_lane_i × min(laid_i.of(m_i), max_lanes_i)
///               over the pairs given a member, 1 ≤ m_i ≤ live_i,
///               with Σ m_i ≤ lanes
/// ```
///
/// A pair given no member pays nothing, its `fixed` neither: the span's
/// guest region holds what a fire lays and nothing that outlives it (bytes
/// an engine keeps for a pair from its bind to its close are state, and
/// live in slots, counted there). `max_lanes_i` of 0 is no cap of its own;
/// a cap is on the lanes a pair lays, not on its members: its flights fly
/// one after another in the same bytes.
///
/// The figure is exact for either form an engine lays by ([`LaidLanes`]).
/// A pair laid member by member pays `fixed + a_lane` for its first member
/// and `a_lane` for each one more, never more for a later member than for
/// an earlier, so the worst fire takes the dearest members first. A pair
/// laid by powers of two does pay more later (two pairs of one `a_lane`
/// and 256 lanes: 129 and 127 members lay 256 and 128, where one pair's
/// 256 lay 256), so those are a multiple-choice knapsack over each pair's
/// breakpoints (1, 2, and one past each power of two up to its members),
/// the fire's lanes the budget. Both forms together take the best split of
/// the lanes between the two.
///
/// Monotone in `lanes` and in every pair's `live`: a count of the binds
/// under way beside the bound (the runtime's) is never under the bound
/// alone (the engine's).
///
/// The table is built when the pairs change (a bind, a close), never in a
/// fire, and only when the change shows in it ([`GuestTerm::fit`]): it
/// keeps the pairs it was built from, each reduced to the fewest members
/// that give the same table, so a bind or close that crosses no breakpoint
/// of a pair laid by powers of two, or moves a pair laid member by member
/// that already has the fire's lanes, costs the reduction alone. What a
/// build costs, measured (one core, release; `a_rebuild_is_timed`):
///
/// ```text
/// pairs x lanes   member by member   by powers of two
///   30 x 256           1.5 us           39 - 53 us
/// 1000 x 1024         17 - 19 us       8.8 - 12.3 ms
/// ```
///
/// Thirty pairs is a wave of samplers at their extents; a thousand open
/// distinct pairs laid by powers of two is where a build stops being
/// negligible beside a bind, on whichever thread it runs (FOLLOWUPS).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GuestTerm {
    /// The worst bytes of a fire of `n` lanes, at `n`; past its end, its
    /// last.
    lanes: Vec<u64>,
    /// The pairs it was built from, reduced ([`FirePair::reduced`]) and in
    /// order, and the most lanes it was built for.
    built: Vec<FirePair>,
    most: u32,
}

impl GuestTerm {
    /// The term of `pairs` for every fire of up to `most` lanes. Built when
    /// the pairs change (a bind, a close), never in a fire. With `B =
    /// min(most, Σ live)`: a sort of the pairs and `B` steps where every
    /// pair is laid member by member; `pairs × B × log B` steps for those
    /// laid by powers of two, and `B²` more where both forms are open.
    #[must_use]
    pub fn new(pairs: &[FirePair], most: u32) -> GuestTerm {
        let built = GuestTerm::reduced(pairs, most);
        GuestTerm {
            lanes: GuestTerm::table(&built, most),
            built,
            most,
        }
    }

    /// Makes this the term of `pairs` for fires of up to `most` lanes,
    /// building the table again only where it differs: whether it did.
    /// What a caller holds across binds and closes in place of
    /// [`GuestTerm::new`]; the two give the same term.
    pub fn fit(&mut self, pairs: &[FirePair], most: u32) -> bool {
        let built = GuestTerm::reduced(pairs, most);
        if self.most == most && self.built == built {
            return false;
        }
        *self = GuestTerm {
            lanes: GuestTerm::table(&built, most),
            built,
            most,
        };
        true
    }

    /// `pairs` as the table reads them, in one order whatever theirs.
    fn reduced(pairs: &[FirePair], most: u32) -> Vec<FirePair> {
        let mut built: Vec<FirePair> = pairs.iter().filter_map(|pair| pair.reduced(most)).collect();
        built.sort_unstable_by_key(|pair| {
            (
                pair.facts.a_lane,
                pair.facts.fixed,
                pair.facts.max_lanes,
                pair.facts.laid,
                pair.live,
            )
        });
        built
    }

    /// The table of `pairs`, reduced, for fires of up to `most` lanes.
    fn table(pairs: &[FirePair], most: u32) -> Vec<u64> {
        let paying = || pairs.iter();
        let budget = paying()
            .fold(0u64, |sum, pair| sum.saturating_add(pair.members()))
            .min(u64::from(most));
        let budget = usize::try_from(budget).unwrap_or(usize::MAX);

        // Member by member: each pair's first member at `fixed + a_lane`,
        // its others at `a_lane`, the dearest first.
        let mut steps: Vec<(u64, u64)> = Vec::new();
        for pair in paying().filter(|pair| pair.facts.laid == LaidLanes::Exact) {
            let facts = &pair.facts;
            steps.push((facts.fixed.saturating_add(facts.a_lane), 1));
            if facts.a_lane > 0 && pair.members() > 1 {
                steps.push((facts.a_lane, pair.members() - 1));
            }
        }
        steps.sort_unstable_by_key(|&(bytes, _)| std::cmp::Reverse(bytes));
        let mut linear = Vec::with_capacity(budget + 1);
        linear.push(0u64);
        'steps: for &(bytes, members) in &steps {
            for _ in 0..members {
                if linear.len() > budget {
                    break 'steps;
                }
                let last = linear[linear.len() - 1];
                linear.push(last.saturating_add(bytes));
            }
        }

        // By powers of two: a knapsack over each pair's breakpoints.
        let mut stepped = vec![0u64; budget + 1];
        let mut before = Vec::with_capacity(budget + 1);
        let mut any_stepped = false;
        for pair in paying().filter(|pair| pair.facts.laid != LaidLanes::Exact) {
            any_stepped = true;
            before.clone_from(&stepped);
            let members = pair.members();
            // The fewest members that lay each count of lanes the pair can.
            let mut at = 1u64;
            while at <= members {
                let Ok(taken) = usize::try_from(at) else {
                    break;
                };
                if taken > budget {
                    break;
                }
                let bytes = pair.facts.bytes(at);
                for (best, under) in stepped[taken..].iter_mut().zip(&before) {
                    *best = (*best).max(under.saturating_add(bytes));
                }
                // One member past the lanes laid now lays the next count,
                // unless the pair's own cap is reached.
                let laid = pair.facts.laid.of(at);
                if pair.facts.lanes(at) < laid {
                    break;
                }
                at = laid.saturating_add(1);
            }
        }

        let at = |table: &[u64], lanes: usize| table[lanes.min(table.len() - 1)];
        if !any_stepped {
            linear
        } else if steps.is_empty() {
            stepped
        } else {
            (0..=budget)
                .map(|lanes| {
                    (0..=lanes)
                        .map(|given| at(&stepped, given).saturating_add(at(&linear, lanes - given)))
                        .max()
                        .unwrap_or(0)
                })
                .collect()
        }
    }

    /// The bytes a fire of up to `lanes` lanes takes; lanes past the most
    /// it was built for take what the most do.
    #[must_use]
    pub fn bytes(&self, lanes: u32) -> u64 {
        let at = usize::try_from(lanes)
            .unwrap_or(usize::MAX)
            .min(self.lanes.len().saturating_sub(1));
        self.lanes.get(at).copied().unwrap_or(0)
    }

    /// No open pair takes anything of the span.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.lanes.last().is_none_or(|&bytes| bytes == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every split of `lanes` members over `pairs`, the dearest taken.
    fn brute(pairs: &[FirePair], lanes: u64) -> u64 {
        let Some((pair, rest)) = pairs.split_first() else {
            return 0;
        };
        (0..=u64::from(pair.live).min(lanes))
            .map(|members| {
                let own = if members == 0 {
                    0
                } else {
                    let laid = match pair.facts.laid {
                        LaidLanes::Exact => members,
                        LaidLanes::PowerOfTwo => members.next_power_of_two(),
                    };
                    let laid = match pair.facts.max_lanes {
                        0 => laid,
                        cap => laid.min(u64::from(cap)),
                    };
                    pair.facts.fixed + laid * pair.facts.a_lane
                };
                own + brute(rest, lanes - members)
            })
            .max()
            .unwrap_or(0)
    }

    fn pair(a_lane: u64, fixed: u64, max_lanes: u32, laid: LaidLanes, live: u32) -> FirePair {
        FirePair {
            facts: FireFacts {
                a_lane,
                fixed,
                max_lanes,
                laid,
            },
            live,
        }
    }

    /// The table is the brute force over every split, for each form laid
    /// and for both together, with unequal `fixed`, over small cases.
    #[test]
    fn the_term_is_the_worst_split_of_a_fires_lanes() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |below: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % below
        };
        for forms in [
            &[LaidLanes::Exact][..],
            &[LaidLanes::PowerOfTwo][..],
            &[LaidLanes::Exact, LaidLanes::PowerOfTwo][..],
        ] {
            for case in 0..2000 {
                let count = 1 + next(4);
                let pairs: Vec<FirePair> = (0..count)
                    .map(|_| {
                        pair(
                            next(6) * 100,
                            [0, 0, 7, 150, 420, 1000][next(6) as usize],
                            [0, 0, 1, 2, 3, 4, 8][next(7) as usize],
                            forms[next(forms.len() as u64) as usize],
                            next(10) as u32,
                        )
                    })
                    .collect();
                let most = 1 + next(14) as u32;
                let term = GuestTerm::new(&pairs, most);
                for lanes in 0..=most {
                    assert_eq!(
                        term.bytes(lanes),
                        brute(&pairs, u64::from(lanes)),
                        "case {case}, {lanes} lanes of {most}: {pairs:?}"
                    );
                }
                // Past the most it was built for: what the most take.
                assert_eq!(term.bytes(most + 5), term.bytes(most));
            }
        }
    }

    /// A term fitted through any run of binds and closes is the term built
    /// afresh from the pairs then, and the brute force's; fits that cross
    /// no breakpoint build nothing.
    #[test]
    #[allow(clippy::print_stdout, reason = "the test reports what it built, under --nocapture")]
    fn a_fitted_term_is_the_term_built_afresh() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |below: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % below
        };
        let (mut built, mut stood) = (0u32, 0u32);
        for run in 0..200 {
            let most = 1 + next(12) as u32;
            let mut pairs: Vec<FirePair> = (0..1 + next(4))
                .map(|_| {
                    pair(
                        next(6) * 100,
                        [0, 0, 7, 150, 420, 1000][next(6) as usize],
                        [0, 0, 1, 2, 3, 4, 8][next(7) as usize],
                        [LaidLanes::Exact, LaidLanes::PowerOfTwo][next(2) as usize],
                        0,
                    )
                })
                .collect();
            let mut term = GuestTerm::default();
            for step in 0..60 {
                let at = next(pairs.len() as u64) as usize;
                // A bind, or a close where one is bound; binds twice as
                // often, up to more members than the fire has lanes.
                if next(3) > 0 && pairs[at].live < 14 {
                    pairs[at].live += 1;
                } else {
                    pairs[at].live = pairs[at].live.saturating_sub(1);
                }
                let before = term.clone();
                if term.fit(&pairs, most) {
                    built += 1;
                } else {
                    stood += 1;
                    assert_eq!(term, before);
                }
                assert_eq!(
                    term,
                    GuestTerm::new(&pairs, most),
                    "run {run}, step {step}: {pairs:?}"
                );
                for lanes in 0..=most {
                    assert_eq!(
                        term.bytes(lanes),
                        brute(&pairs, u64::from(lanes)),
                        "run {run}, step {step}, {lanes} lanes of {most}: {pairs:?}"
                    );
                }
            }
            // The same pairs in another order are the same term.
            pairs.reverse();
            assert!(!term.fit(&pairs, most));
        }
        assert!(built > 0 && stood > 0, "{built} built, {stood} stood");
        println!("{built} fits built the table, {stood} left it standing");
    }

    /// Powers of two are not taken greedily: 129 and 127 members of two
    /// pairs lay 256 and 128 lanes, more than one pair's 256.
    #[test]
    fn two_pairs_laid_by_powers_of_two_lay_more_than_the_fires_lanes() {
        let pairs = [
            pair(1, 0, 0, LaidLanes::PowerOfTwo, 256),
            pair(1, 0, 0, LaidLanes::PowerOfTwo, 256),
        ];
        assert_eq!(GuestTerm::new(&pairs, 256).bytes(256), 384);
        let exact = [
            pair(1, 0, 0, LaidLanes::Exact, 256),
            pair(1, 0, 0, LaidLanes::Exact, 256),
        ];
        assert_eq!(GuestTerm::new(&exact, 256).bytes(256), 256);
    }

    /// Fourteen pairs of one sampler are one fire's lanes, not fourteen
    /// times them; `fixed` is the pairs' a fire carries, one a member at
    /// the most.
    #[test]
    fn many_pairs_of_one_program_take_one_fires_lanes() {
        let lives = [6, 256, 30, 32, 2, 45, 28, 46, 12, 30, 14, 5, 4, 2];
        let pairs: Vec<FirePair> = lives
            .iter()
            .map(|&live| pair(9_312_512, 4_718_592, 0, LaidLanes::Exact, live))
            .collect();
        let term = GuestTerm::new(&pairs, 256);
        assert_eq!(term.bytes(256), 14 * 4_718_592 + 256 * 9_312_512);
        assert_eq!(term.bytes(64), 14 * 4_718_592 + 64 * 9_312_512);
        assert_eq!(term.bytes(3), 3 * 4_718_592 + 3 * 9_312_512);
        assert_eq!(term.bytes(0), 0);
        assert!(!term.is_zero());
        assert!(GuestTerm::new(&[], 256).is_zero());
        assert!(GuestTerm::new(&[pair(5, 5, 0, LaidLanes::Exact, 0)], 256).is_zero());
    }

    /// Monotone in the lanes and in each pair's live count.
    #[test]
    fn the_term_grows_with_lanes_and_live_members() {
        let at = |live: u32, lanes: u32| {
            GuestTerm::new(
                &[
                    pair(3, 10, 8, LaidLanes::PowerOfTwo, live),
                    pair(2, 5, 0, LaidLanes::Exact, 4),
                ],
                64,
            )
            .bytes(lanes)
        };
        for live in 1..20 {
            for lanes in 0..30 {
                assert!(at(live, lanes) <= at(live, lanes + 1));
                assert!(at(live, lanes) <= at(live + 1, lanes));
            }
        }
    }

    /// What a rebuild costs, printed (`--nocapture`): every bind and close
    /// builds the table again.
    #[test]
    #[allow(clippy::print_stdout, reason = "the test reports its timing, under --nocapture")]
    fn a_rebuild_is_timed() {
        let mut seed = 7u64;
        let mut next = |below: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % below
        };
        for (count, most) in [(14usize, 256u32), (30, 256), (1000, 1024)] {
            for laid in [LaidLanes::Exact, LaidLanes::PowerOfTwo] {
                for wide in [false, true] {
                    // One pair holds the wave; the others a few members
                    // each (`wide`: every pair as many as the fire).
                    let pairs: Vec<FirePair> = (0..count)
                        .map(|at| {
                            let live = if at == 0 || wide {
                                most
                            } else {
                                1 + next(46) as u32
                            };
                            pair(9_312_512, 4_718_592 + next(3) * 4096, 0, laid, live)
                        })
                        .collect();
                    let rounds = if count > 100 { 20 } else { 2000 };
                    let begun = std::time::Instant::now();
                    let mut sum = 0u64;
                    for _ in 0..rounds {
                        sum = sum.wrapping_add(
                            GuestTerm::new(std::hint::black_box(&pairs), most).bytes(most),
                        );
                    }
                    let each = begun.elapsed().as_nanos() / rounds;
                    println!(
                        "{count} pairs x {most} lanes, {laid:?}, wide {wide}: {each} ns a rebuild ({sum})"
                    );
                }
            }
        }
    }
}
