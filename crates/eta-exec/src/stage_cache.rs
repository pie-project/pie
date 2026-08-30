use super::cache::{Bounded, MAX_STAGE_ENTRIES};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry<V> {
    identity: u64,
    value: V,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    Hit,

    Miss,

    Collided,
}

#[derive(Debug)]
pub struct Stages<V> {
    committed: Bounded<u64, Entry<V>>,
    pending: Vec<(u64, Entry<V>)>,
    hits: u64,
    collisions: u64,
}

impl<V: Clone> Default for Stages<V> {
    fn default() -> Stages<V> {
        Stages::new(MAX_STAGE_ENTRIES)
    }
}

impl<V: Clone> Stages<V> {
    #[must_use]
    pub fn new(capacity: usize) -> Stages<V> {
        Stages {
            committed: Bounded::new(capacity),
            pending: Vec::new(),
            hits: 0,
            collisions: 0,
        }
    }

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

    pub fn stage(&mut self, key: u64, identity: u64, value: V) {
        self.pending.push((key, Entry { identity, value }));
    }

    pub fn commit(&mut self) -> usize {
        let mut evicted = 0;
        for (key, entry) in self.pending.drain(..) {
            if self.committed.insert(key, entry).is_some() {
                evicted += 1;
            }
        }
        evicted
    }

    pub fn abandon(&mut self) {
        self.pending.clear();
    }

    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    #[must_use]
    pub fn collisions(&self) -> u64 {
        self.collisions
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.committed.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.committed.is_empty()
    }

    #[must_use]
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.committed.capacity()
    }
}
