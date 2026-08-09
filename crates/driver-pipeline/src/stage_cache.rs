//! The stage cache, and the collision guard that decides whether a hit is real.
//!
//! A stage is cached under `(device_cache_id, signature_hash)`. That key is a
//! hash and hashes collide, so the cache also stores the stage's `identity` —
//! a second, independent number — and compares it after a hit. This is the
//! right design and the C++ had it. What follows is about what it did with the
//! answer.
//!
//! ## A collision is not the program's fault
//!
//! The C++ answered a detected collision with `reject_deterministic("Metal M1
//! stage signature hash collision")`. Two things follow from that, and both are
//! wrong.
//!
//! A deterministic rejection is one that says: this program cannot compile, and
//! will not compile later, so stop asking. That is a claim about the program.
//! But a collision is not a property of the program being compiled at all — it
//! is a property of which *other* program happens to occupy the slot right now.
//! The same program, submitted to a fresh process, compiles. Submitted after
//! the squatter is evicted, it compiles. Nothing about it is broken.
//!
//! And [`Failure::Deterministic`] is the classification the negative cache
//! remembers. So the C++ took a perfectly good program, blamed it for a
//! collision it did not cause, and then wrote that verdict down so it would not
//! have to re-derive it — permanently refusing, for the life of the process, a
//! program that was never at fault and whose obstacle may already be gone.
//!
//! The answer here is that the incumbent loses. A hit whose identity does not
//! match is a miss, the stale entry is evicted, and the caller compiles. That
//! is what a cache is allowed to do: it is an optimisation, and the one thing
//! it must never do is turn a cache state into a program error.
//!
//! [`Failure::Deterministic`]: crate::Failure

use crate::cache::{Bounded, MAX_STAGE_ENTRIES};

/// One cached stage: what was compiled, and the identity that proves the key
/// found the right thing.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry<V> {
    /// `LaunchStagePlan::identity` — the graph-cache identity, independent of
    /// the signature hash the entry is keyed on.
    ///
    /// The C++ held this as a `std::vector<std::uint8_t>` of the `u64`'s eight
    /// bytes, so every entry heap-allocated eight bytes in order to compare a
    /// number with `!=` on two vectors. It is a `u64`.
    identity: u64,
    value: V,
}

/// What a lookup found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// The key hit and the identity matched. Reuse the entry.
    Hit,
    /// The key did not hit. Compile.
    Miss,
    /// The key hit and the identity did not: two different stages share a
    /// signature hash on this device.
    ///
    /// The stale entry has been evicted, so the caller compiles and the next
    /// lookup for the same key is a plain [`Miss`](Lookup::Miss). Worth
    /// counting — [`Stages::collisions`] — because a collision rate above
    /// roughly zero means the signature hash is not doing its job, which is a
    /// real finding and one the C++'s permanent rejection would have buried
    /// under a program-shaped error message.
    Collided,
}

/// The stage cache: committed entries, plus the stages of a compile that has
/// not finished yet.
///
/// The C++ kept these as two separate maps and checked them one after the
/// other, with the collision guard — five lines, including the rejection —
/// written out identically in both arms. Two copies of a guard is two things to
/// fix when the guard is wrong, and the guard was wrong.
///
/// The pending half exists because a program's stages must land together: a
/// compile that fails partway must leave the cache exactly as it found it, or a
/// later program keyed the same way reuses half a stage. [`commit`](Self::commit)
/// and [`abandon`](Self::abandon) are the two ends of that.
#[derive(Debug)]
pub struct Stages<V> {
    committed: Bounded<u64, Entry<V>>,
    pending: Vec<(u64, Entry<V>)>,
    hits: u64,
    collisions: u64,
}

impl<V: Clone> Default for Stages<V> {
    /// A cache of [`MAX_STAGE_ENTRIES`], which is `kMaxStageCacheEntries`.
    fn default() -> Stages<V> {
        Stages::new(MAX_STAGE_ENTRIES)
    }
}

impl<V: Clone> Stages<V> {
    /// A cache holding `capacity` committed stages.
    #[must_use]
    pub fn new(capacity: usize) -> Stages<V> {
        Stages {
            committed: Bounded::new(capacity),
            pending: Vec::new(),
            hits: 0,
            collisions: 0,
        }
    }

    /// Look `key` up and check the identity of whatever it found.
    ///
    /// The pending stages of the compile in flight are searched too, because a
    /// program whose stages repeat — the same stage appearing twice in one
    /// program is ordinary — must reuse the one it just built rather than
    /// build it again.
    ///
    /// On [`Lookup::Collided`] the offending entry is dropped from whichever
    /// half held it. See the module docs for why that, and not a rejection.
    pub fn lookup(&mut self, key: u64, identity: u64) -> (Lookup, Option<V>) {
        if let Some(position) = self.pending.iter().position(|(at, _)| *at == key) {
            if self.pending[position].1.identity == identity {
                self.hits += 1;
                return (Lookup::Hit, Some(self.pending[position].1.value.clone()));
            }
            self.pending.remove(position);
            self.collisions += 1;
            return (Lookup::Collided, None);
        }
        match self.committed.get(&key) {
            Some(entry) if entry.identity == identity => {
                let value = entry.value.clone();
                self.hits += 1;
                (Lookup::Hit, Some(value))
            }
            Some(_) => {
                self.committed.remove(&key);
                self.collisions += 1;
                (Lookup::Collided, None)
            }
            None => (Lookup::Miss, None),
        }
    }

    /// Add a freshly compiled stage to the compile in flight.
    ///
    /// Nothing is refused here. The C++ checked `committed.size() +
    /// pending.size() >= max` before compiling and returned a *retryable*
    /// failure when it was over — the same "a full cache refuses" mistake the
    /// program cache made, with the same consequence: a caller that retries
    /// forever against the one condition retrying cannot change. A full cache
    /// evicts; see [`Bounded`].
    pub fn stage(&mut self, key: u64, identity: u64, value: V) {
        self.pending.push((key, Entry { identity, value }));
    }

    /// Move the compile in flight into the cache.
    ///
    /// Returns how many entries were evicted to make room, which is worth
    /// knowing: a program whose own stages evict each other is a program that
    /// will never see a warm cache, and the number says so.
    pub fn commit(&mut self) -> usize {
        let mut evicted = 0;
        for (key, entry) in self.pending.drain(..) {
            if self.committed.insert(key, entry).is_some() {
                evicted += 1;
            }
        }
        evicted
    }

    /// Discard the compile in flight, leaving the cache as it was.
    pub fn abandon(&mut self) {
        self.pending.clear();
    }

    /// How many lookups found a usable entry.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// How many lookups found the wrong entry under the right key.
    #[must_use]
    pub fn collisions(&self) -> u64 {
        self.collisions
    }

    /// How many stages are committed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.committed.len()
    }

    /// Whether nothing is committed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.committed.is_empty()
    }

    /// How many stages are staged but not yet committed.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// How many committed stages the cache holds at once.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.committed.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> Stages<&'static str> {
        Stages::new(4)
    }

    #[test]
    fn a_miss_is_a_miss() {
        let mut stages = cache();
        assert_eq!(stages.lookup(1, 10), (Lookup::Miss, None));
    }

    #[test]
    fn a_committed_stage_is_found_by_its_key_and_its_identity() {
        let mut stages = cache();
        stages.stage(1, 10, "compiled");
        stages.commit();
        assert_eq!(stages.lookup(1, 10), (Lookup::Hit, Some("compiled")));
        assert_eq!(stages.hits(), 1);
    }

    #[test]
    fn a_stage_of_the_compile_in_flight_is_reused_rather_than_rebuilt() {
        let mut stages = cache();
        stages.stage(1, 10, "compiled");
        assert_eq!(stages.lookup(1, 10), (Lookup::Hit, Some("compiled")));
        assert_eq!(stages.len(), 0, "still uncommitted");
    }

    #[test]
    fn the_same_key_with_a_different_identity_is_a_collision_and_not_a_hit() {
        let mut stages = cache();
        stages.stage(1, 10, "someone elses stage");
        stages.commit();
        assert_eq!(stages.lookup(1, 999), (Lookup::Collided, None));
        assert_eq!(stages.collisions(), 1);
        assert_eq!(stages.hits(), 0);
    }

    #[test]
    fn a_collision_evicts_the_incumbent_so_the_next_try_is_an_ordinary_miss() {
        // This is the whole argument. The C++ answered the first lookup with a
        // permanent, remembered rejection of a program that did nothing wrong,
        // so the second lookup -- and every lookup after it, forever -- got the
        // same rejection.
        let mut stages = cache();
        stages.stage(1, 10, "someone elses stage");
        stages.commit();
        assert_eq!(stages.lookup(1, 999).0, Lookup::Collided);
        assert_eq!(stages.lookup(1, 999).0, Lookup::Miss);
    }

    #[test]
    fn a_program_recovers_from_a_collision_by_compiling_and_taking_the_slot() {
        let mut stages = cache();
        stages.stage(1, 10, "incumbent");
        stages.commit();
        assert_eq!(stages.lookup(1, 999).0, Lookup::Collided);
        stages.stage(1, 999, "mine");
        stages.commit();
        assert_eq!(stages.lookup(1, 999), (Lookup::Hit, Some("mine")));
    }

    #[test]
    fn a_collision_inside_the_compile_in_flight_is_caught_too() {
        let mut stages = cache();
        stages.stage(1, 10, "first");
        assert_eq!(stages.lookup(1, 11), (Lookup::Collided, None));
    }

    #[test]
    fn an_abandoned_compile_leaves_the_cache_exactly_as_it_was() {
        let mut stages = cache();
        stages.stage(1, 10, "committed");
        stages.commit();
        stages.stage(2, 20, "half built");
        stages.abandon();
        assert_eq!(stages.pending(), 0);
        assert_eq!(stages.len(), 1);
        assert_eq!(stages.lookup(2, 20), (Lookup::Miss, None));
        assert_eq!(stages.lookup(1, 10).0, Lookup::Hit);
    }

    #[test]
    fn a_full_cache_takes_the_new_stage_rather_than_refusing_it() {
        let mut stages = Stages::new(2);
        for key in 0..5u64 {
            stages.stage(key, key, "stage");
            stages.commit();
        }
        assert_eq!(stages.len(), 2);
        assert_eq!(stages.lookup(4, 4).0, Lookup::Hit);
    }

    #[test]
    fn commit_reports_how_many_stages_the_program_evicted() {
        let mut stages = Stages::new(2);
        for key in 0..4u64 {
            stages.stage(key, key, "stage");
        }
        assert_eq!(stages.commit(), 2);
    }

    #[test]
    fn a_program_whose_stages_repeat_compiles_each_one_once() {
        let mut stages = cache();
        assert_eq!(stages.lookup(7, 70).0, Lookup::Miss);
        stages.stage(7, 70, "shared");
        // The second occurrence of the same stage in the same program.
        assert_eq!(stages.lookup(7, 70), (Lookup::Hit, Some("shared")));
        stages.commit();
        assert_eq!(stages.len(), 1);
    }

    #[test]
    fn the_capacity_is_the_one_cache_rs_already_names() {
        assert_eq!(Stages::<()>::default().capacity(), 64);
    }
}
