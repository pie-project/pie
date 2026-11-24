use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::device::{Image, Kept};

/// A kept frame, replayed for every fire of its family: the fires whose
/// walks record the same commands but for what lies in the fire block (the
/// scalars a kernel reads there) and the group counts. A dispatch whose
/// counts differ between the family's shapes reads them from the block
/// (indirect); every other carries the counts it was recorded with.
pub(crate) struct Recording {
    pub(crate) kept: Kept,

    used: std::cell::Cell<u64>,

    /// The host step after each segment but the last, in order.
    pub(crate) cuts: Vec<engine_gpu::store::experts::Cut<crate::device::Api>>,
}

impl Recording {
    pub(crate) fn new(
        kept: Kept,
        cuts: Vec<engine_gpu::store::experts::Cut<crate::device::Api>>,
    ) -> Recording {
        Recording {
            kept,
            cuts,
            used: std::cell::Cell::new(0),
        }
    }

    /// Whether a fire whose dispatches take `groups` replays this: every
    /// dispatch whose counts differ from the recording's reads them from
    /// the block.
    pub(crate) fn serves(&self, groups: &[[u32; 3]]) -> bool {
        let held = self.kept.groups();
        held.len() == groups.len()
            && held
                .iter()
                .zip(groups)
                .enumerate()
                .all(|(at, (held, wanted))| held == wanted || self.kept.indirect(at))
    }
}

/// A family: its recording, and the dispatches whose group counts differ
/// between the shapes seen of it, which its next recording makes indirect.
pub(crate) struct Family {
    pub(crate) recording: Recording,

    pub(crate) varying: Arc<Vec<bool>>,
}

impl Family {
    /// Widens the varying set by every dispatch whose counts in `groups`
    /// differ from the recording's.
    pub(crate) fn varies(&mut self, groups: &[[u32; 3]]) {
        let held = self.recording.kept.groups();
        let mut varying = self.varying.as_ref().clone();
        varying.resize(groups.len().max(varying.len()), false);
        for (at, wanted) in groups.iter().enumerate() {
            if held.get(at) != Some(wanted) {
                varying[at] = true;
            }
        }
        self.varying = Arc::new(varying);
    }

    /// Whether `kept`, walked with some dispatches indirect, makes every one
    /// of the family's varying dispatches so.
    pub(crate) fn covered_by(&self, kept: &Kept) -> bool {
        self.varying
            .iter()
            .enumerate()
            .all(|(at, &varies)| !varies || kept.indirect(at))
    }
}

/// What a fire of one shape writes in its family's fire block, the group
/// counts its dispatches take, and what it reads out: laid by the walk of
/// the first fire of that shape.
pub(crate) struct Fill {
    /// The family whose commands that walk recorded.
    pub(crate) family: Key,

    pub(crate) groups: Arc<Vec<[u32; 3]>>,

    pub(crate) image: Arc<Image>,

    pub(crate) layout: Vec<engine_gpu::serve::ReadSpan>,
    pub(crate) launches: u32,

    used: std::cell::Cell<u64>,
}

impl Fill {
    pub(crate) fn new(
        family: Key,
        groups: Arc<Vec<[u32; 3]>>,
        image: Arc<Image>,
        layout: Vec<engine_gpu::serve::ReadSpan>,
        launches: u32,
    ) -> Fill {
        Fill {
            family,
            groups,
            image,
            layout,
            launches,
            used: std::cell::Cell::new(0),
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


/// Fills kept: one a fire shape, each a few bytes a dispatch.
const FILLS_KEPT: usize = 256;

/// Entries under keys, the stalest dropped past a count.
struct Held<T> {
    by_key: HashMap<Key, T>,
    most: usize,
}

impl<T> Held<T> {
    fn new(most: usize) -> Held<T> {
        Held {
            by_key: HashMap::new(),
            most,
        }
    }

    /// Inserts `held` under `key`, the stalest entry leaving past the count.
    fn insert(&mut self, key: Key, held: T, used: impl Fn(&T) -> u64) {
        if self.by_key.len() >= self.most
            && !self.by_key.contains_key(&key)
            && let Some(stalest) = self
                .by_key
                .iter()
                .min_by_key(|(_, held)| used(held))
                .map(|(key, _)| *key)
        {
            self.by_key.remove(&stalest);
        }
        self.by_key.insert(key, held);
    }
}

pub(crate) struct Recorder {
    families: Held<Family>,

    /// Each fire shape's fill, under the shape's key.
    fills: Held<Fill>,


    recorded: std::cell::Cell<u64>,
    filled: std::cell::Cell<u64>,

    clock: std::cell::Cell<u64>,

    refused: Option<&'static str>,
}

/// What a fire shape's next fire does: replays its family, or walks, with
/// the dispatches its family's shapes vary in made indirect.
pub(crate) enum Next<'a> {
    Replay(&'a Fill, &'a Recording),
    Walk(Option<Arc<Vec<bool>>>),
}

impl Recorder {
    pub(crate) fn new() -> Recorder {
        Recorder {
            // Every family stands: the load records its bodies before it
            // serves, a fixed set, and none leaves for a later one.
            families: Held::new(usize::MAX),
            fills: Held::new(FILLS_KEPT),
            recorded: std::cell::Cell::new(0),
            filled: std::cell::Cell::new(0),
            clock: std::cell::Cell::new(0),
            refused: None,
        }
    }

    pub(crate) fn refusing(why: &'static str) -> Recorder {
        Recorder {
            refused: Some(why),
            ..Recorder::new()
        }
    }

    /// The family under `key`.
    pub(crate) fn family(&self, key: Key) -> Option<&Family> {
        if self.refused.is_some() {
            return None;
        }
        let found = self.families.by_key.get(&key);
        if let Some(family) = found {
            family.recording.used.set(self.tick());
        }
        found
    }

    pub(crate) fn family_mut(&mut self, key: Key) -> Option<&mut Family> {
        self.families.by_key.get_mut(&key)
    }

    /// What the fire shape `shape`, of the (arm, rung, class subset)
    /// `kind`, does next.
    pub(crate) fn next(&self, shape: Key) -> Next<'_> {
        // A shape without a fill walks direct: the load recorded every
        // family it serves, so its walk bakes one of them and fills it.
        let Some(fill) = self.fills.by_key.get(&shape) else {
            return Next::Walk(None);
        };
        let Some(family) = self.family(fill.family) else {
            return Next::Walk(None);
        };
        fill.used.set(self.tick());
        if family.recording.serves(&fill.groups) {
            Next::Replay(fill, &family.recording)
        } else {
            Next::Walk(Some(Arc::clone(&family.varying)))
        }
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
        self.families.by_key.clear();
        self.fills.by_key.clear();
    }

    /// Every family held: its key and recording.
    pub(crate) fn families(&self) -> impl Iterator<Item = (&Key, &Recording)> {
        self.families
            .by_key
            .iter()
            .map(|(key, family)| (key, &family.recording))
    }

    /// Drops every family whose recording binds a pool segment where no
    /// rebind writes it ([`crate::device::Kept::pinned`]), and the fills and
    /// bodies naming it: the segments it binds changed. What pinned each.
    pub(crate) fn forget_pinned(&mut self) -> Vec<String> {
        let pinned: Vec<(Key, String)> = self
            .families
            .by_key
            .iter()
            .filter_map(|(key, family)| family.recording.kept.pinned().map(|why| (*key, why)))
            .collect();
        for (key, _) in &pinned {
            self.families.by_key.remove(key);
        }
        if !pinned.is_empty() {
            let gone = |family: &Key| pinned.iter().any(|(key, _)| key == family);
            self.fills.by_key.retain(|_, fill| !gone(&fill.family));
        }
        pinned.into_iter().map(|(_, why)| why).collect()
    }

    /// Makes `recording` its family's, varying in `varying`.
    pub(crate) fn insert(&mut self, key: Key, recording: Recording, varying: Arc<Vec<bool>>) {
        self.recorded.set(self.recorded.get() + 1);
        recording.used.set(self.tick());
        self.families
            .insert(key, Family { recording, varying }, |held| {
                held.recording.used.get()
            });
    }

    pub(crate) fn insert_fill(&mut self, shape: Key, fill: Fill) {
        self.filled.set(self.filled.get() + 1);
        fill.used.set(self.tick());
        self.fills.insert(shape, fill, |held| held.used.get());
    }

    /// Recordings made, and fire shapes filled, so far.
    pub(crate) fn counts(&self) -> (u64, u64) {
        (self.recorded.get(), self.filled.get())
    }
}

