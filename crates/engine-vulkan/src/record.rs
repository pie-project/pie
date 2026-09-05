use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::device::Kept;

pub(crate) struct Recording {
    pub(crate) kept: Kept,

    used: std::cell::Cell<u64>,

    pub(crate) layout: Vec<(u32, u32)>,
    pub(crate) launches: u32,
}

impl Recording {
    pub(crate) fn new(kept: Kept, layout: Vec<(u32, u32)>, launches: u32) -> Recording {
        Recording {
            kept,
            used: std::cell::Cell::new(0),
            layout,
            launches,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Key(u64);

pub(crate) struct KeyBuilder(std::collections::hash_map::DefaultHasher);

impl KeyBuilder {
    pub(crate) fn new() -> KeyBuilder {
        KeyBuilder(std::collections::hash_map::DefaultHasher::new())
    }

    pub(crate) fn bytes(&mut self, bytes: &[u8]) {
        bytes.hash(&mut self.0);
    }

    pub(crate) fn shown(&mut self, what: &dyn std::fmt::Debug) {
        use std::fmt::Write;

        struct Sink<'a>(&'a mut std::collections::hash_map::DefaultHasher);
        impl Write for Sink<'_> {
            fn write_str(&mut self, s: &str) -> std::fmt::Result {
                s.as_bytes().hash(self.0);
                Ok(())
            }
        }

        if write!(Sink(&mut self.0), "{what:?}").is_err() {
            "the debug rendering refused".hash(&mut self.0);
        }
    }

    pub(crate) fn finish(self) -> Key {
        Key(self.0.finish())
    }
}

const SHAPES_KEPT: usize = 64;

pub(crate) struct Recorder {
    by_key: HashMap<Key, Recording>,

    replayed: std::cell::Cell<u64>,
    recorded: std::cell::Cell<u64>,
    evicted: std::cell::Cell<u64>,

    clock: std::cell::Cell<u64>,

    refused: Option<&'static str>,
}

impl Recorder {
    pub(crate) fn new() -> Recorder {
        Recorder {
            by_key: HashMap::new(),
            replayed: std::cell::Cell::new(0),
            recorded: std::cell::Cell::new(0),
            evicted: std::cell::Cell::new(0),
            clock: std::cell::Cell::new(0),
            refused: None,
        }
    }

    pub(crate) fn refusing(why: &'static str) -> Recorder {
        Recorder {
            by_key: HashMap::new(),
            replayed: std::cell::Cell::new(0),
            recorded: std::cell::Cell::new(0),
            evicted: std::cell::Cell::new(0),
            clock: std::cell::Cell::new(0),
            refused: Some(why),
        }
    }

    pub(crate) fn refused(&self) -> Option<&'static str> {
        self.refused
    }

    pub(crate) fn get(&self, key: Key) -> Option<&Recording> {
        if self.refused.is_some() {
            return None;
        }
        let found = self.by_key.get(&key);
        if let Some(recording) = found {
            self.replayed.set(self.replayed.get() + 1);
            recording.used.set(self.tick());
        }
        found
    }

    fn tick(&self) -> u64 {
        let next = self.clock.get() + 1;
        self.clock.set(next);
        next
    }

    pub(crate) fn counts(&self) -> (u64, u64) {
        (self.replayed.get(), self.recorded.get())
    }

    pub(crate) fn records(&self) -> bool {
        self.refused.is_none()
    }

    pub(crate) fn has_room(&self) -> bool {
        self.refused.is_none()
    }

    pub(crate) fn insert(&mut self, key: Key, recording: Recording) {
        self.recorded.set(self.recorded.get() + 1);
        recording.used.set(self.tick());
        if self.by_key.len() >= SHAPES_KEPT
            && !self.by_key.contains_key(&key)
            && let Some(stalest) = self
                .by_key
                .iter()
                .min_by_key(|(_, held)| held.used.get())
                .map(|(key, _)| *key)
        {
            self.by_key.remove(&stalest);
            self.evicted.set(self.evicted.get() + 1);
        }
        self.by_key.insert(key, recording);
    }

    pub(crate) fn evicted(&self) -> u64 {
        self.evicted.get()
    }

    pub(crate) fn len(&self) -> usize {
        self.by_key.len()
    }
}
