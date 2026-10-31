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
    /// Its holds past its exit; see [`Departure`].
    pub(super) departure: Weak<Departure>,
}

impl Proc {
    pub(super) fn rank(&self) -> Rank {
        Rank::of(self.key, self.seq)
    }

    pub(super) fn new(seq: u64) -> Self {
        Self {
            seq,
            key: 0,
            held: false,
            state: Residency::Resident,
            restore_retried: false,
            progressed: true,
            admitted: false,
            respawn: false,
            signal: Arc::new(Notify::new()),
            resident: Arc::new(AtomicBool::new(true)),
            idle: false,
            parked: None,
            seatless: false,
            reseating: false,
            seated: false,
            departure: Weak::new(),
        }
    }
}
