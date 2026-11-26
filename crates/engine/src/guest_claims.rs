//! The device's account of where each live guest instance's state lies: the
//! byte runs its tables, rings, status words and scratch take, by segment
//! (a pool slab, or the one device address space where an engine addresses
//! its pool by device address). The runtime hands each `Instance` and
//! `Channel` slot ref to one instance; a fire writes the instance's state
//! and reads tables the host wrote, so were one ref ever handed out twice,
//! one guest's writes would land in another's tables. Every engine claims
//! an instance's runs here at its bind, claims them again wherever a rebind
//! or an evacuation moved them, before it writes a table or a descriptor
//! there, and releases them at its close: a claim overlapping a run another
//! live holder holds is refused by name ([`Overlap`]), and nothing changes.
//!
//! A guest batch's (program, extents) pair state is one holder of its own,
//! claimed by the group of instances bound on it, not by any one of them
//! ([`Claims::claim_with_pair`]): each member joins it at its bind or
//! rebind and leaves at its close, and its runs go with the last member. It
//! moves only as the whole group's move: an evacuation
//! ([`Claims::claim_all`]), or a new taking of its slots, which the runtime
//! makes only for the pair item as a whole (every holder parked; its
//! generation, `engine::InstanceBinding::pair_generation`, is new each time
//! the item gets fresh slots). A member's rebind naming the same taking at
//! other bytes while other members are bound would move or clear the
//! group's state alone, and is refused by name ([`Pinned`]).
//!
//! A park is said to the engine before its slots are handed to anyone
//! else ([`crate::Rebind::Parked`]): the parked instance keeps only the
//! runs that stayed where they lie (its tables and rings where no tier
//! below took them), leaves its pair's group, and claims again where its
//! unpark's rebind names it ([`Claims::park`]). Held through a park, its
//! scratch's runs refused the next owner the runtime gave them to.
//!
//! The runs live in one ordered map keyed by `(segment, start)`, disjoint by
//! construction (another holder's overlap is refused; a holder's own
//! overlapping runs are merged), so a claim or a release is `O(log n)` a
//! run.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// The tracing target every claim, release, park and evacuation's claim is
/// said on at debug, with the holder, its runs before and after, and the
/// outcome: `RUST_LOG=engine::claims=debug` names whose claim stayed where.
pub const TARGET: &str = "engine::claims";

/// Whether claims lines are said ([`TARGET`] at debug): an engine reads it
/// before it builds a line of its own for [`say`].
#[must_use]
pub fn saying() -> bool {
    tracing::enabled!(target: TARGET, tracing::Level::DEBUG)
}

/// An engine's own claims line, on [`TARGET`] at debug: why it claims,
/// releases or parks (a bind, a rebind, an evacuation, a park, a close),
/// beside the claims' own lines. For an engine with no tracing of its own.
pub fn say(why: &str, fields: fmt::Arguments<'_>) {
    tracing::debug!(target: TARGET, %fields, "{why}");
}

/// `runs` as `segment:start..end` pieces, for a claims line.
fn show(runs: &[Run]) -> String {
    let pieces: Vec<String> = runs
        .iter()
        .map(|run| format!("{}:{}..{}", run.segment, run.start, run.end))
        .collect();
    format!("[{}]", pieces.join(", "))
}

/// Bytes `[start, end)` of segment `segment`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Run {
    pub segment: u64,
    pub start: u64,
    pub end: u64,
}

impl Run {
    #[must_use]
    pub const fn overlaps(&self, other: &Run) -> bool {
        self.segment == other.segment && self.start < other.end && other.start < self.end
    }

    const fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

impl fmt::Display for Run {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bytes {}..{} of segment {}",
            self.start, self.end, self.segment
        )
    }
}

/// Who holds a run: one instance's own state, or a pair's, which the
/// group of instances bound on it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Holder {
    Instance(u64),
    Pair(u64),
}

impl fmt::Display for Holder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Holder::Instance(id) => write!(f, "instance {id}"),
            Holder::Pair(id) => write!(f, "pair {id}"),
        }
    }
}

/// A claim refused: `claimant`'s run `run` overlaps `held`, which live
/// `holder` holds. Nothing was claimed or released.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overlap {
    pub claimant: Holder,
    pub run: Run,
    pub holder: Holder,
    pub held: Run,
}

impl fmt::Display for Overlap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Instance slots at bytes {}..{} of segment {} already hold {}'s run at {}..{}: \
             {}'s would overlap it",
            self.run.start,
            self.run.end,
            self.run.segment,
            self.holder,
            self.held.start,
            self.held.end,
            self.claimant
        )
    }
}

/// A member's rebind refused: it names pair `pair`'s state, at the taking
/// the group holds, at `wanted`, while `members` (other than `mover`) are
/// still bound on it at `held`. Nothing was claimed or released.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pinned {
    pub pair: u64,
    pub mover: u64,
    pub members: Vec<u64>,
    pub held: Vec<Run>,
    pub wanted: Vec<Run>,
}

impl fmt::Display for Pinned {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let runs = |runs: &[Run]| {
            runs.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        write!(
            f,
            "pair {}'s state at {} is held by its still-bound instance(s) {:?}: instance {}'s \
             rebind would move or clear it alone, to {} at the same taking; the pair moves only \
             as the whole group's move or once its last instance lets it go",
            self.pair,
            runs(&self.held),
            self.members,
            self.mover,
            runs(&self.wanted)
        )
    }
}

/// Why a claim was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    Overlap(Overlap),
    Pinned(Pinned),
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::Overlap(overlap) => overlap.fmt(f),
            Refusal::Pinned(pinned) => pinned.fmt(f),
        }
    }
}

impl std::error::Error for Refusal {}

impl From<Overlap> for Refusal {
    fn from(overlap: Overlap) -> Refusal {
        Refusal::Overlap(overlap)
    }
}

impl From<Pinned> for Refusal {
    fn from(pinned: Pinned) -> Refusal {
        Refusal::Pinned(pinned)
    }
}

impl From<Refusal> for crate::Error {
    fn from(refusal: Refusal) -> crate::Error {
        crate::Error::Claimed(refusal)
    }
}

impl From<Overlap> for crate::Error {
    fn from(overlap: Overlap) -> crate::Error {
        crate::Error::Claimed(overlap.into())
    }
}

/// A pair's state as a member's bind or rebind names it: the pair (the
/// engine's name for the group), the taking of its slots
/// (`engine::InstanceBinding::pair_generation`) and its runs there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairClaim {
    pub pair: u64,
    pub taking: u64,
    pub runs: Vec<Run>,
}

/// What a member's join found its pair's state to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Joined {
    /// Laid anew where named (the first member, a new taking, or a move
    /// no other member holds against): another owner's bytes until the
    /// engine clears and lays them.
    Fresh,
    /// As the group holds it already: nothing to clear.
    Held,
}

#[derive(Debug)]
struct Group {
    taking: u64,
    members: BTreeSet<u64>,
}

#[derive(Debug, Default)]
struct Held {
    /// `(segment, start)` to `(end, holder)`: every live run, disjoint.
    runs: BTreeMap<(u64, u64), (u64, Holder)>,
    /// Each holder's runs, merged, as they lie in `runs`.
    by_holder: HashMap<Holder, Vec<Run>>,
    /// Each pair's group.
    groups: HashMap<u64, Group>,
    /// The pair each member is bound on.
    member_of: HashMap<u64, u64>,
}

impl Held {
    /// A run another holder than `claimant` holds that overlaps `run`.
    fn overlap(&self, claimant: Holder, run: &Run) -> Option<(Holder, Run)> {
        let from = self
            .runs
            .range(..=(run.segment, run.start))
            .next_back()
            .filter(|&(&(segment, _), &(end, _))| segment == run.segment && end > run.start)
            .map_or((run.segment, run.start), |(&key, _)| key);
        // Disjoint runs: from the one `run` starts in (if any), each one
        // starting before `run` ends overlaps it.
        self.runs
            .range(from..(run.segment, run.end))
            .find(|&(_, &(end, holder))| holder != claimant && end > run.start)
            .map(|(&(segment, start), &(end, holder))| {
                (
                    holder,
                    Run {
                        segment,
                        start,
                        end,
                    },
                )
            })
    }

    /// `runs` for `claimant` in place of what it held, or the first
    /// overlap with another's and nothing changed.
    fn lay(&mut self, claimant: Holder, runs: &[Run]) -> Result<(), Overlap> {
        for run in runs.iter().filter(|run| !run.is_empty()) {
            if let Some((holder, held)) = self.overlap(claimant, run) {
                return Err(Overlap {
                    claimant,
                    run: *run,
                    holder,
                    held,
                });
            }
        }
        self.remove(claimant);
        self.insert(claimant, merge(runs));
        Ok(())
    }

    /// Merged, disjoint runs no other holder's overlaps, for `holder`,
    /// which holds none.
    fn insert(&mut self, holder: Holder, merged: Vec<Run>) {
        for run in &merged {
            self.runs
                .insert((run.segment, run.start), (run.end, holder));
        }
        if !merged.is_empty() {
            self.by_holder.insert(holder, merged);
        }
    }

    fn remove(&mut self, holder: Holder) -> Vec<Run> {
        let runs = self.by_holder.remove(&holder).unwrap_or_default();
        for run in &runs {
            self.runs.remove(&(run.segment, run.start));
        }
        runs
    }

    fn held(&self, holder: Holder) -> Vec<Run> {
        self.by_holder.get(&holder).cloned().unwrap_or_default()
    }

    /// `member` joins `claim.pair` as its bind or rebind names it.
    fn join(&mut self, member: u64, claim: &PairClaim) -> Result<Joined, Refusal> {
        let pair = Holder::Pair(claim.pair);
        let wanted = merge(&claim.runs);
        let joined = match self.groups.get(&claim.pair) {
            None => {
                self.lay(pair, &wanted)?;
                Joined::Fresh
            }
            Some(group) if group.taking == claim.taking => {
                let held = self.held(pair);
                if held == wanted {
                    Joined::Held
                } else {
                    let members: Vec<u64> = group
                        .members
                        .iter()
                        .copied()
                        .filter(|&held| held != member)
                        .collect();
                    if !members.is_empty() {
                        return Err(Pinned {
                            pair: claim.pair,
                            mover: member,
                            members,
                            held,
                            wanted,
                        }
                        .into());
                    }
                    self.lay(pair, &wanted)?;
                    Joined::Fresh
                }
            }
            // A new taking: the runtime gave the pair item fresh slots, which
            // it does for the item as a whole.
            Some(_) => {
                self.lay(pair, &wanted)?;
                Joined::Fresh
            }
        };
        if let Some(was) = self.member_of.get(&member).copied()
            && was != claim.pair
        {
            self.leave(member);
        }
        let group = self.groups.entry(claim.pair).or_insert_with(|| Group {
            taking: claim.taking,
            members: BTreeSet::new(),
        });
        group.taking = claim.taking;
        group.members.insert(member);
        self.member_of.insert(member, claim.pair);
        Ok(joined)
    }

    /// `member` leaves its pair, whose runs go with its last member.
    fn leave(&mut self, member: u64) {
        let Some(pair) = self.member_of.remove(&member) else {
            return;
        };
        if let Some(group) = self.groups.get_mut(&pair) {
            group.members.remove(&member);
            if group.members.is_empty() {
                self.groups.remove(&pair);
                self.remove(Holder::Pair(pair));
            }
        }
    }
}

/// What the holders a [`Claims::claim_all_undoable`] listed held before it.
#[must_use = "a claim undone puts back what it took the place of"]
pub struct Before(Vec<(Holder, Vec<Run>)>);

/// Every live holder's runs.
#[derive(Debug, Default)]
pub struct Claims {
    held: Mutex<Held>,
    next: AtomicU64,
}

impl Claims {
    /// A name for a new instance's or pair's claims, for an engine whose
    /// instances or pairs carry no id of their own here.
    pub fn instance(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Held> {
        self.held.lock().expect("claims lock poisoned")
    }

    /// Claims `runs` for `instance` in place of what it held, at a bind or
    /// a rebind: refused by name, and nothing changed (what it held it
    /// still holds), where one overlaps a run another holder holds. Its own
    /// runs may overlap each other, and what it held: they are merged.
    /// Empty runs claim nothing.
    pub fn claim(&self, instance: u64, runs: &[Run]) -> Result<(), Overlap> {
        let mut held = self.lock();
        let before = held.held(Holder::Instance(instance));
        let laid = held.lay(Holder::Instance(instance), runs);
        tracing::debug!(
            target: TARGET,
            instance,
            before = %show(&before),
            runs = %show(runs),
            refused = ?laid.as_ref().err().map(ToString::to_string),
            "claim"
        );
        laid
    }

    /// [`Claims::claim`], and `instance` joins the pair `pair` names, in one
    /// step: the pair's runs laid where it is the first member, where the
    /// taking is new, or where no other member is bound on it; the group's
    /// as they stand where they are the runs named. Refused by name, and
    /// nothing changed, where a run overlaps another holder's, or where the
    /// rebind names the group's taking at other bytes while other members
    /// are bound ([`Pinned`]).
    pub fn claim_with_pair(
        &self,
        instance: u64,
        runs: &[Run],
        pair: &PairClaim,
    ) -> Result<Joined, Refusal> {
        let mut held = self.lock();
        let own = Holder::Instance(instance);
        let before = held.held(own);
        let pair_before = held.held(Holder::Pair(pair.pair));
        let joined = match held.lay(own, runs) {
            Err(overlap) => Err(overlap.into()),
            Ok(()) => {
                let joined = held.join(instance, pair);
                if joined.is_err() {
                    held.remove(own);
                    held.insert(own, before.clone());
                }
                joined
            }
        };
        tracing::debug!(
            target: TARGET,
            instance,
            before = %show(&before),
            runs = %show(runs),
            pair = pair.pair,
            taking = pair.taking,
            pair_before = %show(&pair_before),
            pair_runs = %show(&pair.runs),
            joined = ?joined.as_ref().ok(),
            refused = ?joined.as_ref().err().map(ToString::to_string),
            "claim with pair"
        );
        joined
    }

    /// Claims each `(holder, runs)` in place of what that holder held, all
    /// in one step: the whole of an evacuation's moves, where one may land
    /// on bytes another listed holder moved off. A pair's members stay its
    /// members. Refused by name, and nothing changed, where a run overlaps
    /// one an unlisted holder holds or another listed holder's new one.
    pub fn claim_all(&self, claims: &[(Holder, Vec<Run>)]) -> Result<(), Overlap> {
        self.claim_all_undoable(claims).map(drop)
    }

    /// [`Claims::claim_all`], answering what each listed holder held
    /// before: what [`Claims::restore`] puts back, where a step after the
    /// claim fails and the whole is undone.
    pub fn claim_all_undoable(&self, claims: &[(Holder, Vec<Run>)]) -> Result<Before, Overlap> {
        let mut held = self.lock();
        let before: Vec<(Holder, Vec<Run>)> = claims
            .iter()
            .map(|&(holder, _)| (holder, held.remove(holder)))
            .collect();
        let mut refused = None;
        for (at, (holder, runs)) in claims.iter().enumerate() {
            if let Err(overlap) = held.lay(*holder, runs) {
                for &(holder, _) in &claims[..at] {
                    held.remove(holder);
                }
                for (holder, runs) in &before {
                    held.insert(*holder, runs.clone());
                }
                refused = Some(overlap);
                break;
            }
        }
        // Each listed holder's runs before and after, and every holder the
        // evacuation left unlisted where it was: a claim a move left
        // behind is seen beside the moves.
        if saying() {
            for ((holder, runs), (_, was)) in claims.iter().zip(&before) {
                tracing::debug!(
                    target: TARGET,
                    %holder,
                    before = %show(was),
                    runs = %show(runs),
                    "evacuation claim"
                );
            }
            let listed: std::collections::HashSet<Holder> =
                claims.iter().map(|&(holder, _)| holder).collect();
            for (holder, runs) in held
                .by_holder
                .iter()
                .filter(|(holder, _)| !listed.contains(holder))
            {
                tracing::debug!(target: TARGET, %holder, runs = %show(runs), "evacuation leaves unlisted");
            }
            tracing::debug!(
                target: TARGET,
                listed = claims.len(),
                refused = ?refused.as_ref().map(ToString::to_string),
                "evacuation claims"
            );
        }
        refused.map_or(Ok(Before(before)), Err)
    }

    /// Puts back what [`Claims::claim_all_undoable`] took the place of: each
    /// listed holder holds what it held before the claim.
    pub fn restore(&self, before: Before) {
        let mut held = self.lock();
        for (holder, runs) in before.0 {
            held.remove(holder);
            held.insert(holder, runs);
        }
    }

    /// Releases what `instance` holds, at its close, and leaves its pair:
    /// the pair's runs go with its last member.
    pub fn release(&self, instance: u64) {
        let mut held = self.lock();
        let was = held.remove(Holder::Instance(instance));
        let pair = held.member_of.get(&instance).copied();
        held.leave(instance);
        tracing::debug!(
            target: TARGET,
            instance,
            released = %show(&was),
            left_pair = ?pair,
            pair_runs_now = %pair.map_or_else(String::new, |pair| show(&held.held(Holder::Pair(pair)))),
            "release"
        );
    }

    /// `instance`'s process parked (`crate::Rebind::Parked`): it keeps
    /// `kept` of what it held (its tables and rings where they stayed in
    /// place; none where they went down a tier), gives back the rest, and
    /// leaves its pair, whose runs go with its last member. Its rebind
    /// claims where it lies again.
    pub fn park(&self, instance: u64, kept: &[Run]) {
        let asked = kept;
        let mut held = self.lock();
        let own = Holder::Instance(instance);
        let was = held.remove(own);
        let kept: Vec<Run> = merge(kept)
            .into_iter()
            .filter(|run| {
                was.iter().any(|had| {
                    had.segment == run.segment && had.start <= run.start && run.end <= had.end
                })
            })
            .collect();
        tracing::debug!(
            target: TARGET,
            instance,
            held = %show(&was),
            kept = %show(&kept),
            asked = %show(asked),
            pair = ?held.member_of.get(&instance),
            "park"
        );
        held.insert(own, kept);
        held.leave(instance);
    }

    /// What `holder` holds now, merged, in order.
    #[must_use]
    pub fn held(&self, holder: Holder) -> Vec<Run> {
        self.lock().held(holder)
    }

    /// The members bound on pair `pair`, in order.
    #[must_use]
    pub fn members(&self, pair: u64) -> Vec<u64> {
        self.lock()
            .groups
            .get(&pair)
            .map(|group| group.members.iter().copied().collect())
            .unwrap_or_default()
    }
}

/// `runs` less the empty ones, sorted, the overlapping or touching ones of
/// a segment merged into one.
fn merge(runs: &[Run]) -> Vec<Run> {
    let mut sorted: Vec<Run> = runs.iter().copied().filter(|run| !run.is_empty()).collect();
    sorted.sort_unstable();
    let mut merged: Vec<Run> = Vec::with_capacity(sorted.len());
    for run in sorted {
        match merged.last_mut() {
            Some(last) if last.segment == run.segment && run.start <= last.end => {
                last.end = last.end.max(run.end);
            }
            _ => merged.push(run),
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::{Claims, Holder, Joined, PairClaim, Refusal, Run};

    const fn run(segment: u64, start: u64, end: u64) -> Run {
        Run {
            segment,
            start,
            end,
        }
    }

    /// Two instances whose runs a runtime laid over each other: the second
    /// bind is refused by name and claims nothing; the same bytes in another
    /// segment, or after the first is released, are taken.
    #[test]
    fn a_run_another_instance_holds_refuses_the_bind() {
        let claims = Claims::default();
        let (a, b) = (claims.instance(), claims.instance());
        claims
            .claim(a, &[run(1, 0, 4096), run(2, 8192, 16384)])
            .expect("the first instance claims its runs");
        let why = claims
            .claim(b, &[run(3, 0, 64), run(2, 12288, 12544)])
            .expect_err("a heap over the first instance's run");
        assert!(why.to_string().contains("already hold instance"), "{why}");
        claims
            .claim(b, &[run(3, 0, 64)])
            .expect("the refused claim took nothing, and another segment is free");
        claims
            .claim(a, &[run(1, 0, 4096), run(2, 0, 4096)])
            .expect("an instance's own runs move without refusing it");
        claims
            .claim(b, &[run(2, 12288, 12544), run(3, 0, 64)])
            .expect("what the first moved off is free");
        claims.release(a);
        claims
            .claim(b, &[run(1, 0, 4096)])
            .expect("a released instance's runs are free");
        assert!(
            !run(1, 0, 4096).overlaps(&run(1, 4096, 8192)),
            "touching runs do not overlap"
        );
    }

    /// Every way a run can overlap a held one is refused, naming both
    /// instances and both runs, as an `engine::Error`.
    #[test]
    fn an_overlap_is_refused_by_name() {
        let claims = Claims::default();
        claims
            .claim(7, &[run(0, 1000, 2000)])
            .expect("the first claim");
        for over in [
            run(0, 500, 1001),
            run(0, 1999, 2500),
            run(0, 1200, 1300),
            run(0, 0, 9000),
            run(0, 1000, 2000),
        ] {
            let overlap = claims.claim(9, &[over]).expect_err("an overlap");
            assert_eq!(
                (overlap.claimant, overlap.holder),
                (Holder::Instance(9), Holder::Instance(7))
            );
            assert_eq!(overlap.run, over);
            assert_eq!(overlap.held, run(0, 1000, 2000));
            let error = crate::Error::from(overlap);
            assert!(matches!(error, crate::Error::Claimed(_)), "{error}");
            assert!(
                error.to_string().contains("instance 7's run at 1000..2000"),
                "{error}"
            );
        }
        assert!(
            claims.held(Holder::Instance(9)).is_empty(),
            "a refused claim holds nothing"
        );
    }

    /// A rebind replaces what the instance held in one step: it may move
    /// over its own old runs, frees what it moved off, and a refused one
    /// leaves the old claim whole.
    #[test]
    fn a_rebind_moves_the_claim_whole_or_not_at_all() {
        let claims = Claims::default();
        claims
            .claim(1, &[run(0, 0, 4096), run(1, 0, 64)])
            .expect("the first claim");
        claims.claim(2, &[run(0, 8192, 12288)]).expect("another");
        // Over its own old run and partly past it.
        claims
            .claim(1, &[run(0, 2048, 6144), run(1, 0, 64)])
            .expect("a move over its own run");
        assert_eq!(
            claims.held(Holder::Instance(1)),
            vec![run(0, 2048, 6144), run(1, 0, 64)]
        );
        claims
            .claim(3, &[run(0, 0, 2048)])
            .expect("what it moved off is free");
        // Onto another's run: refused, the old claim kept whole.
        claims
            .claim(1, &[run(0, 10000, 10100), run(1, 4096, 8192)])
            .expect_err("onto the second's run");
        assert_eq!(
            claims.held(Holder::Instance(1)),
            vec![run(0, 2048, 6144), run(1, 0, 64)]
        );
        claims
            .claim(4, &[run(1, 0, 1)])
            .expect_err("still held by the first");
        claims
            .claim(4, &[run(1, 4096, 8192)])
            .expect("the refused target took nothing");
    }

    fn pair(pair: u64, taking: u64, runs: &[Run]) -> PairClaim {
        PairClaim {
            pair,
            taking,
            runs: runs.to_vec(),
        }
    }

    /// A member's rebind naming the group's taking at other bytes, while
    /// another member is bound, is refused by name, naming the pair, the
    /// mover and the members still bound; nothing changes, its own runs
    /// included. Alone on it, the same rebind moves it.
    #[test]
    fn a_solo_move_of_a_held_pair_is_refused() {
        let claims = Claims::default();
        let at = [run(0, 1000, 2000)];
        claims
            .claim_with_pair(1, &[run(0, 0, 100)], &pair(50, 7, &at))
            .expect("first");
        claims
            .claim_with_pair(2, &[run(0, 100, 200)], &pair(50, 7, &at))
            .expect("second");
        let refused = claims
            .claim_with_pair(1, &[run(0, 300, 400)], &pair(50, 7, &[run(0, 5000, 6000)]))
            .expect_err("a solo move with member 2 bound");
        let Refusal::Pinned(pinned) = &refused else {
            panic!("{refused}")
        };
        assert_eq!(
            (pinned.pair, pinned.mover, pinned.members.clone()),
            (50, 1, vec![2])
        );
        assert_eq!(pinned.held, at.to_vec());
        assert_eq!(pinned.wanted, vec![run(0, 5000, 6000)]);
        let message = crate::Error::from(refused).to_string();
        assert!(
            message.contains("pair 50")
                && message.contains("[2]")
                && message.contains("instance 1's"),
            "{message}"
        );
        assert_eq!(
            claims.held(Holder::Instance(1)),
            vec![run(0, 0, 100)],
            "its own runs as they stood"
        );
        assert_eq!(claims.held(Holder::Pair(50)), at.to_vec());
        claims.release(2);
        assert_eq!(
            claims.claim_with_pair(1, &[run(0, 300, 400)], &pair(50, 7, &[run(0, 5000, 6000)])),
            Ok(Joined::Fresh),
            "alone on it, the member moves it"
        );
        claims
            .claim(3, &[run(0, 1000, 2000)])
            .expect("what it moved off is free");
    }
}
