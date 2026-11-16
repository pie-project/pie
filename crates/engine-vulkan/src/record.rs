use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::device::Kept;

pub(crate) struct Recording {
    pub(crate) kept: Kept,

    used: std::cell::Cell<u64>,

    pub(crate) layout: Vec<engine_gpu::serve::ReadSpan>,
    pub(crate) launches: u32,

    /// The host step after each segment but the last, in order.
    pub(crate) cuts: Vec<engine_gpu::store::experts::Cut<crate::device::Api>>,
}

impl Recording {
    pub(crate) fn new(
        kept: Kept,
        layout: Vec<engine_gpu::serve::ReadSpan>,
        launches: u32,
        cuts: Vec<engine_gpu::store::experts::Cut<crate::device::Api>>,
    ) -> Recording {
        Recording {
            kept,
            cuts,
            used: std::cell::Cell::new(0),
            layout,
            launches,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Key(u64);

pub(crate) struct KeyBuilder(std::collections::hash_map::DefaultHasher);

impl KeyBuilder {
    pub(crate) fn new() -> KeyBuilder {
        KeyBuilder(std::collections::hash_map::DefaultHasher::new())
    }

    pub(crate) fn bytes(&mut self, bytes: &[u8]) {
        bytes.hash(&mut self.0);
    }

    pub(crate) fn hashed(&mut self, what: &impl Hash) {
        what.hash(&mut self.0);
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

/// The arms a composition was kept at, as of which forget.
struct KeptAt {
    arms: Vec<usize>,
    forgets: u64,
}

pub(crate) struct Recorder {
    by_key: HashMap<Key, Recording>,

    /// What each composition (the key less its arm and its tables) was kept
    /// at, and since which forget: why a fire keeps a recording.
    kept: HashMap<Key, KeptAt>,
    forgets: u64,

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
            kept: HashMap::new(),
            forgets: 0,
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
            kept: HashMap::new(),
            forgets: 0,
            replayed: std::cell::Cell::new(0),
            recorded: std::cell::Cell::new(0),
            evicted: std::cell::Cell::new(0),
            clock: std::cell::Cell::new(0),
            refused: Some(why),
        }
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

    pub(crate) fn records(&self) -> bool {
        self.refused.is_none()
    }

    pub(crate) fn has_room(&self) -> bool {
        self.refused.is_none()
    }

    /// Drops every recording: each names the slabs it was recorded over, and
    /// a replay after those slabs change would land in the old ones.
    pub(crate) fn forget(&mut self) {
        self.by_key.clear();
        self.forgets += 1;
    }

    /// Why a fire of `composition` at `arm` keeps a recording, and notes it
    /// kept one: no recording of its composition before; one only at
    /// another arm; one at this arm over other tables (kv pages, a mask's
    /// width, a gathered window); or one a forget since dropped.
    pub(crate) fn why_kept(&mut self, composition: Key, arm: usize) -> &'static str {
        let forgets = self.forgets;
        let seen = self.kept.entry(composition).or_insert_with(|| KeptAt {
            arms: Vec::new(),
            forgets,
        });
        let why = if seen.arms.is_empty() {
            "a composition no recording held"
        } else if seen.forgets != forgets {
            "a composition whose recordings a commit, release or residency forgot"
        } else if !seen.arms.contains(&arm) {
            "a composition recorded at another arm"
        } else {
            "a composition recorded at this arm over other tables (kv pages, a mask, a window)"
        };
        if !seen.arms.contains(&arm) {
            seen.arms.push(arm);
        }
        seen.forgets = forgets;
        why
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
}

#[cfg(test)]
mod tests {
    use super::{KeyBuilder, Recorder};

    fn key(n: u32) -> super::Key {
        let mut key = KeyBuilder::new();
        key.hashed(&n);
        key.finish()
    }

    /// Why a fire keeps a recording: a composition no recording held, then
    /// the same at another arm, then at an arm it was kept at (its tables
    /// changed), then after a forget.
    #[test]
    fn a_kept_recording_says_why() {
        let mut recorder = Recorder::new();
        assert_eq!(
            recorder.why_kept(key(1), 0),
            "a composition no recording held"
        );
        assert_eq!(
            recorder.why_kept(key(1), 1),
            "a composition recorded at another arm"
        );
        assert!(recorder.why_kept(key(1), 0).contains("over other tables"));
        recorder.forget();
        assert!(recorder.why_kept(key(1), 0).contains("forgot"));
        assert_eq!(
            recorder.why_kept(key(2), 0),
            "a composition no recording held"
        );
    }
}
