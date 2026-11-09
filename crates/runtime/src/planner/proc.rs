use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Weak};

use tokio::sync::Notify;

use super::departure::Departure;
use super::grant::Demand;
use super::rank::Rank;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Residency {
    Resident,
    Evicting,
    Evicted,
    Restoring,
}

pub(super) struct Proc {
    pub(super) seq: u64,
    pub(super) key: i64,
    pub(super) held: bool,
    pub(super) state: Residency,
    pub(super) restore_retried: bool,
    pub(super) progressed: bool,
    pub(super) admitted: bool,
    pub(super) respawn: bool,
    pub(super) signal: Arc<Notify>,
    pub(super) resident: Arc<AtomicBool>,
    /// A handle's stand-in: never ran, so its leaving is no completion.
    pub(super) holder: bool,
    /// Waiting on its client: a victim for anyone, however old it is.
    pub(super) idle: bool,
    /// What an idle eviction freed; it asks for it back only once it wakes.
    pub(super) parked: Option<Demand>,
    /// Parked below the floor, it gave its execution seat back: it takes
    /// one again before its restore is queued.
    pub(super) seatless: bool,
    pub(super) reseating: bool,
    /// It has been granted pages: a seated lane whose next claim is growth.
    pub(super) seated: bool,
    /// A holder's retention key: among those at rest, the lowest goes first.
    pub(super) kept: i64,
    /// The most device kv pages and rs units it has held at a grant, for
    /// [`super::to_end::ToEnd`].
    pub(super) peak: (u32, crate::store::rs::RsUnits),
    /// What it held at its first grant: its prompt.
    pub(super) first: (u32, crate::store::rs::RsUnits),
    /// Its holds past its exit; see [`Departure`].
    pub(super) departure: Weak<Departure>,
    /// Admitted against the state pool: it counts for the fixed sets a
    /// running working set takes until it leaves.
    pub(super) charged: bool,
}

impl Proc {
    pub(super) fn rank(&self) -> Rank {
        Rank::of(self.key, self.seq).kept(self.kept)
    }

    /// A holder at rest: past every process, whose rank is what goes first,
    /// and the longest parked past all. `order` counts up as holders park.
    pub(super) fn rest(&mut self, order: u64) {
        self.key = i64::MIN;
        self.seq = u64::MAX - order;
    }

    pub(super) fn new(seq: u64) -> Self {
        Self {
            seq,
            key: 0,
            kept: 0,
            held: false,
            state: Residency::Resident,
            restore_retried: false,
            progressed: true,
            admitted: false,
            respawn: false,
            signal: Arc::new(Notify::new()),
            resident: Arc::new(AtomicBool::new(true)),
            holder: false,
            idle: false,
            parked: None,
            seatless: false,
            reseating: false,
            seated: false,
            peak: Default::default(),
            first: Default::default(),
            departure: Weak::new(),
            charged: false,
        }
    }
}
