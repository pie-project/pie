//! The bounded caches a compile goes through, and what a full one does.
//!
//! Compiling a pipeline state is expensive enough to cache three ways: in
//! memory by program and by stage, on disk as an `mtl4archive`, and negatively
//! for the failures that will fail again. This module owns the memory tiers and
//! the one decision the C++ got wrong in all of them.
//!
//! # A bounded cache must evict, not refuse
//!
//! The C++ reached its bound and stopped compiling:
//!
//! ```text
//! if (impl_->programs.size() >= impl_->max_program_cache_entries) {
//!     return reject_retryable("Metal M1 program executable cache is full");
//! }
//! ```
//!
//! Nothing removes an entry from `programs` or from `stage_cache` anywhere in
//! the file. So the sixty-fifth distinct program a process sees is refused, and
//! refused again on every retry, forever -- the failure is classified
//! *retryable*, so the caller keeps trying, and the condition that makes it fail
//! is the one thing retrying cannot change. A cache that is full is supposed to
//! be slower. This one is a wedge, and it is a wedge that only appears on a
//! long-lived process with a varied workload, which is exactly the deployment
//! nobody tests.
//!
//! [`Bounded`] evicts the least recently used entry instead. Sixty-five
//! programs on a sixty-four entry cache means one recompile, which is what a
//! cache bound is for.
//!
//! # Why least-recently-used, and why the obvious implementation
//!
//! The negative cache did evict, with `negative.erase(negative.begin())` -- the
//! entry the container happens to order first, which is neither the oldest nor
//! the coldest. A program that fails on every fire can be evicted while one
//! that failed once and was never seen again survives, and then the expensive
//! compile is re-attempted every fire forever. LRU is the eviction order that
//! matches how these are used: a program in the active batch is touched
//! constantly, and one that has left is not touched at all.
//!
//! The implementation is a hash map of `(value, last-used tick)` and a linear
//! scan for the minimum on eviction. An intrusive list would make that O(1), at
//! the cost of either `unsafe` or a hand-rolled index arena. At a bound of
//! sixty-four entries, on a path that already involves a shader compiler, the
//! scan is not measurable and the machinery would be the more expensive thing.
//!
//! # Two kinds of failure
//!
//! A compile can fail because the program is not compilable -- a plan that is
//! not executable, a stage wanting more channel slots than a lane has, a
//! signature-hash collision -- or because the driver could not compile it *now*.
//! Only the first kind is worth remembering: retrying it burns a shader compile
//! to reach the same answer. The second must not be remembered, because
//! remembering it makes a transient condition permanent. This is the same
//! distinction [`Readiness`](super::readiness::Readiness) draws for fires, and
//! it is drawn here for the same reason.

use std::collections::HashMap;
use std::hash::Hash;

/// How many compiled programs to keep.
pub const MAX_PROGRAM_ENTRIES: usize = 64;
/// How many compiled stages to keep.
pub const MAX_STAGE_ENTRIES: usize = 64;
/// How many remembered failures to keep.
pub const MAX_NEGATIVE_ENTRIES: usize = 64;

/// Why a compile did not produce an executable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Failure {
    /// The program cannot be compiled by this driver. Retrying reaches the
    /// same answer, so it is worth remembering.
    Deterministic {
        /// What was wrong.
        reason: String,
    },
    /// The driver could not compile it now. Remembering this would make a
    /// transient condition permanent.
    Retryable {
        /// What was wrong.
        reason: String,
    },
}

impl Failure {
    /// What went wrong, whichever kind this is.
    #[must_use]
    pub fn reason(&self) -> &str {
        match self {
            Failure::Deterministic { reason } | Failure::Retryable { reason } => reason,
        }
    }

    /// Whether this failure belongs in the negative cache.
    #[must_use]
    pub fn is_remembered(&self) -> bool {
        matches!(self, Failure::Deterministic { .. })
    }
}

/// What the caches did, for the counters a caller reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Answered from a memory tier.
    pub memory_hits: u64,
    /// Answered from an on-disk archive.
    pub persistent_hits: u64,
    /// Actually compiled.
    pub compilations: u64,
    /// Answered from the negative cache.
    pub negative_hits: u64,
    /// Entries dropped to make room.
    ///
    /// Not in the C++'s stats, because the C++ never evicted from the two
    /// caches that mattered. It is the number that says whether a bound is too
    /// small: a hit rate can look fine while every insert costs an eviction.
    pub evictions: u64,
}

/// One entry and when it was last touched.
#[derive(Clone, Debug)]
struct Entry<V> {
    value: V,
    used: u64,
}

/// A bounded map that evicts its least recently used entry.
#[derive(Clone, Debug)]
pub struct Bounded<K, V> {
    entries: HashMap<K, Entry<V>>,
    capacity: usize,
    tick: u64,
}

impl<K: Eq + Hash + Clone, V> Bounded<K, V> {
    /// A cache holding at most `capacity` entries.
    ///
    /// A capacity of zero is raised to one. A cache that can hold nothing is
    /// not a smaller cache, it is a cache whose every insert immediately
    /// evicts itself, and the code that reads from it then has a branch that
    /// is never taken -- so the degenerate configuration is turned into the
    /// nearest working one rather than silently disabling a tier.
    #[must_use]
    pub fn new(capacity: usize) -> Bounded<K, V> {
        Bounded {
            entries: HashMap::new(),
            capacity: capacity.max(1),
            tick: 0,
        }
    }

    /// Look up `key`, marking it as used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.tick += 1;
        let used = self.tick;
        let entry = self.entries.get_mut(key)?;
        entry.used = used;
        Some(&entry.value)
    }

    /// Look up `key` without marking it as used.
    ///
    /// For counters and diagnostics. A `peek` that touched would make merely
    /// reporting on a cache change what it evicts.
    #[must_use]
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|entry| &entry.value)
    }

    /// Whether `key` is present, without marking it as used.
    #[must_use]
    pub fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    /// Insert `key`, evicting the least recently used entry if that is what it
    /// takes, and returning whatever was evicted.
    ///
    /// Replacing an existing key evicts nothing: the entry count does not grow.
    pub fn insert(&mut self, key: K, value: V) -> Option<(K, V)> {
        self.tick += 1;
        let used = self.tick;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.value = value;
            entry.used = used;
            return None;
        }
        let evicted = if self.entries.len() >= self.capacity {
            self.evict()
        } else {
            None
        };
        self.entries.insert(key, Entry { value, used });
        evicted
    }

    /// Drop one named entry, wherever it sits in the order.
    ///
    /// This is not what a cache does on its own — an LRU map chooses its own
    /// victim. It exists for the caller that has learned the entry is *wrong*,
    /// which is a different fact from the entry being cold: the stage cache's
    /// collision guard finds the right key holding the wrong stage, and the
    /// entry has to go regardless of how recently it was used.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.entries.remove(key).map(|entry| entry.value)
    }

    /// Drop the least recently used entry.
    fn evict(&mut self) -> Option<(K, V)> {
        let coldest = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.used)
            .map(|(key, _)| key.clone())?;
        self.entries
            .remove_entry(&coldest)
            .map(|(key, entry)| (key, entry.value))
    }

    /// How many entries are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many entries may be held.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the C++ did not have: a full cache costs a recompile, not
    /// a request that can never run.
    #[test]
    fn a_full_cache_evicts_rather_than_refusing_to_accept_anything_more() {
        let mut cache: Bounded<u32, u32> = Bounded::new(2);
        assert_eq!(cache.insert(1, 10), None);
        assert_eq!(cache.insert(2, 20), None);
        let evicted = cache.insert(3, 30);
        assert_eq!(
            evicted,
            Some((1, 10)),
            "the third entry must be accepted; refusing it wedges every \
             subsequent program forever"
        );
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.peek(&3), Some(&30));
    }

    /// The eviction order is use, not insertion and not whatever the container
    /// happens to order first.
    #[test]
    fn the_least_recently_used_entry_is_the_one_dropped() {
        let mut cache: Bounded<u32, u32> = Bounded::new(2);
        cache.insert(1, 10);
        cache.insert(2, 20);
        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(
            cache.insert(3, 30),
            Some((2, 20)),
            "key 1 was inserted first but used last, so key 2 is the cold one"
        );
        assert!(cache.contains(&1));
    }

    /// A program that fails on every fire is touched on every fire. Evicting it
    /// in favour of one seen once is how `erase(begin())` re-attempts the same
    /// expensive compile forever.
    #[test]
    fn a_repeatedly_used_entry_survives_a_stream_of_one_off_ones() {
        let mut cache: Bounded<u32, u32> = Bounded::new(4);
        cache.insert(0, 0);
        for key in 1..32 {
            assert_eq!(
                cache.get(&0),
                Some(&0),
                "the hot entry was evicted at {key}"
            );
            cache.insert(key, key);
        }
        assert!(cache.contains(&0));
    }

    /// Reporting on a cache must not change what it evicts.
    #[test]
    fn peeking_does_not_count_as_use() {
        let mut cache: Bounded<u32, u32> = Bounded::new(2);
        cache.insert(1, 10);
        cache.insert(2, 20);
        assert_eq!(cache.peek(&1), Some(&10));
        assert!(cache.contains(&1));
        assert_eq!(
            cache.insert(3, 30),
            Some((1, 10)),
            "a peek left key 1 the coldest, which is the point of having one"
        );
    }

    /// Replacing a key is not growth, so it must not cost an unrelated entry.
    #[test]
    fn replacing_an_existing_key_evicts_nothing() {
        let mut cache: Bounded<u32, u32> = Bounded::new(2);
        cache.insert(1, 10);
        cache.insert(2, 20);
        assert_eq!(cache.insert(2, 21), None);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.peek(&1), Some(&10));
        assert_eq!(cache.peek(&2), Some(&21));
    }

    /// A cache that can hold nothing is not a smaller cache; it is a tier that
    /// silently does not exist.
    #[test]
    fn a_capacity_of_zero_becomes_one() {
        let mut cache: Bounded<u32, u32> = Bounded::new(0);
        assert_eq!(cache.capacity(), 1);
        cache.insert(1, 10);
        assert_eq!(cache.peek(&1), Some(&10));
    }

    /// Only failures that will recur are worth a cache entry. Remembering a
    /// transient one converts it into a permanent one.
    #[test]
    fn only_deterministic_failures_are_remembered() {
        let permanent = Failure::Deterministic {
            reason: "plan is not executable".into(),
        };
        let transient = Failure::Retryable {
            reason: "stage executable cache is full".into(),
        };
        assert!(permanent.is_remembered());
        assert!(
            !transient.is_remembered(),
            "a remembered transient failure never gets a second chance to succeed"
        );
        assert_eq!(transient.reason(), "stage executable cache is full");
    }

    /// The negative cache is the same structure, so it inherits the eviction
    /// order the C++'s `erase(begin())` did not have.
    #[test]
    fn the_negative_cache_keeps_the_failure_it_keeps_hitting() {
        let mut negative: Bounded<u64, Failure> = Bounded::new(MAX_NEGATIVE_ENTRIES);
        let hot = Failure::Deterministic {
            reason: "unsupported op".into(),
        };
        negative.insert(0, hot.clone());
        for key in 1..(MAX_NEGATIVE_ENTRIES as u64 * 2) {
            assert_eq!(negative.get(&0), Some(&hot));
            negative.insert(
                key,
                Failure::Deterministic {
                    reason: "seen once".into(),
                },
            );
        }
        assert!(
            negative.contains(&0),
            "evicting the failure being hit every fire re-attempts its compile \
             every fire"
        );
        assert_eq!(negative.len(), MAX_NEGATIVE_ENTRIES);
    }
}
