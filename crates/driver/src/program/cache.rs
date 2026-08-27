use std::collections::HashMap;
use std::hash::Hash;

pub const MAX_PROGRAM_ENTRIES: usize = 64;

pub const MAX_STAGE_ENTRIES: usize = 64;

pub const MAX_NEGATIVE_ENTRIES: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Failure {
    Deterministic { reason: String },

    Retryable { reason: String },
}

impl Failure {
    #[must_use]
    pub fn reason(&self) -> &str {
        match self {
            Failure::Deterministic { reason } | Failure::Retryable { reason } => reason,
        }
    }

    #[must_use]
    pub fn is_remembered(&self) -> bool {
        matches!(self, Failure::Deterministic { .. })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub memory_hits: u64,

    pub persistent_hits: u64,

    pub compilations: u64,

    pub negative_hits: u64,

    pub evictions: u64,
}

#[derive(Clone, Debug)]
struct Entry<V> {
    value: V,
    used: u64,
}

#[derive(Clone, Debug)]
pub struct Bounded<K, V> {
    entries: HashMap<K, Entry<V>>,
    capacity: usize,
    tick: u64,
}

impl<K: Eq + Hash + Clone, V> Bounded<K, V> {
    #[must_use]
    pub fn new(capacity: usize) -> Bounded<K, V> {
        Bounded {
            entries: HashMap::new(),
            capacity: capacity.max(1),
            tick: 0,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.tick += 1;
        let used = self.tick;
        let entry = self.entries.get_mut(key)?;
        entry.used = used;
        Some(&entry.value)
    }

    #[must_use]
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|entry| &entry.value)
    }

    #[must_use]
    pub fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

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

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.entries.remove(key).map(|entry| entry.value)
    }

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

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
