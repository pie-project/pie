use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::device::Kept;

pub(crate) struct Recording {
    pub(crate) kept: Kept,

    pub(crate) launches: u32,

    /// The host step after each segment but the last, in order.
    pub(crate) cuts: Vec<engine_gpu::store::experts::Cut<crate::device::Api>>,

    /// What the fire it was walked from bound beside its composition: a
    /// fire replays it only over the same.
    pub(crate) binds: Key,
}

impl Recording {
    pub(crate) fn new(
        kept: Kept,
        launches: u32,
        cuts: Vec<engine_gpu::store::experts::Cut<crate::device::Api>>,
        binds: Key,
    ) -> Recording {
        Recording {
            kept,
            launches,
            cuts,
            binds,
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

/// The recordings of the decode lattice the load armed, each under its
/// composition and arm: held for the load's life, none evicted.
pub(crate) struct Recorder {
    by_key: HashMap<Key, Recording>,

    refused: Option<&'static str>,
}

impl Recorder {
    pub(crate) fn new() -> Recorder {
        Recorder {
            by_key: HashMap::new(),
            refused: None,
        }
    }

    pub(crate) fn refusing(why: &'static str) -> Recorder {
        Recorder {
            by_key: HashMap::new(),
            refused: Some(why),
        }
    }

    pub(crate) fn get(&self, key: Key) -> Option<&Recording> {
        if self.refused.is_some() {
            return None;
        }
        self.by_key.get(&key)
    }

    pub(crate) fn records(&self) -> bool {
        self.refused.is_none()
    }

    /// Why it keeps no recording, where it keeps none.
    pub(crate) fn refused(&self) -> Option<&'static str> {
        self.refused
    }

    /// Drops every recording: each names the slabs it was recorded over, and
    /// a replay after those slabs change would land in the old ones.
    pub(crate) fn forget(&mut self) {
        if !self.by_key.is_empty() {
            tracing::warn!(
                recordings = self.by_key.len(),
                "the slabs the armed recordings bind changed: they are dropped"
            );
        }
        self.by_key.clear();
    }

    /// Every recording's kept frame.
    pub(crate) fn kept(&self) -> impl Iterator<Item = &Kept> {
        self.by_key.values().map(|recording| &recording.kept)
    }

    pub(crate) fn insert(&mut self, key: Key, recording: Recording) {
        self.by_key.insert(key, recording);
    }
}
