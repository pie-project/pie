//! How many more fires a lane's pipeline will send after a given one, when
//! that is known: once the inferlet has closed the pipeline, every fire it
//! will post is posted. An engine that runs ahead of the fires it is given
//! (a device token loop) asks so it stops where the pipeline will.
//!
//! The runtime tells this table which frames each pipeline posted and when
//! it closed, and notes each fire it hands an engine by the lane's KV slot
//! and the rows it held before the fire (which name one fire of one
//! sequence). The answer is worked out when the engine asks, not when the
//! fire was handed over: a fire is often handed over before its pipeline
//! has closed. Notes are taken when read; the tables are bounded (the oldest
//! go first).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};

const CAP: usize = 4096;

/// A fire's place: its pipeline, its frame's sequence number, its slot in
/// the frame and the frame's fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Place {
    pub pipeline: u128,
    pub seq: u64,
    pub slot: u32,
    pub fires: u32,
}

#[derive(Default)]
struct Posted {
    /// Fires of each frame, by its sequence number.
    frames: BTreeMap<u64, u32>,
    closed: bool,
}

#[derive(Default)]
struct Notes {
    pipelines: HashMap<u128, Posted>,
    fires: HashMap<(u32, u32), Place>,
    order: VecDeque<(u32, u32)>,
}

static NOTES: LazyLock<Mutex<Notes>> = LazyLock::new(Default::default);

fn with<T>(f: impl FnOnce(&mut Notes) -> T) -> T {
    f(&mut NOTES.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

/// `pipeline` posted frame `seq` of `fires` fires.
pub fn posted(pipeline: u128, seq: u64, fires: u32) {
    with(|n| {
        n.pipelines.entry(pipeline).or_default().frames.insert(seq, fires);
    });
}

/// `pipeline` posts no more.
pub fn closed(pipeline: u128) {
    with(|n| {
        if let Some(p) = n.pipelines.get_mut(&pipeline) {
            p.closed = true;
        }
    });
}

/// `pipeline` is gone.
pub fn gone(pipeline: u128) {
    with(|n| {
        n.pipelines.remove(&pipeline);
    });
}

/// The fire of `slot` at `held` rows is the one at `place`.
pub fn note(slot: u32, held: u32, place: Place) {
    with(|n| {
        if n.fires.insert((slot, held), place).is_none() {
            n.order.push_back((slot, held));
        }
        while n.order.len() > CAP {
            if let Some(old) = n.order.pop_front() {
                n.fires.remove(&old);
            }
        }
    });
}

/// The fires known to follow the fire of `slot` at `held` rows (its
/// pipeline closed), if it was noted.
#[must_use]
pub fn take(slot: u32, held: u32) -> Option<u32> {
    with(|n| {
        let place = n.fires.remove(&(slot, held))?;
        n.order.retain(|key| *key != (slot, held));
        let posted = n.pipelines.get_mut(&place.pipeline)?;
        // Frames before this one have been run.
        posted.frames = posted.frames.split_off(&place.seq);
        if !posted.closed {
            return None;
        }
        let later: u32 = posted
            .frames
            .range(place.seq + 1..)
            .map(|(_, &fires)| fires)
            .sum();
        Some(later + place.fires.saturating_sub(place.slot + 1))
    })
}
