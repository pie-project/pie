use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Weak};

use tokio::sync::Notify;

use super::departure::Departure;
use super::grant::Demand;
use super::rank::{Rank, VictimOrder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Residency {
    Resident,
    Evicting,
    Evicted,
    Restoring,
}

pub(super) struct Proc {
    pub(super) seq: u64,
    /// When it last became resident (admitted or restored), on the same
    /// count as `seq`: what [`VictimOrder`] reads past the policy's keys.
    pub(super) came: u64,
    pub(super) key: i64,
    pub(super) held: bool,
    pub(super) state: Residency,
    pub(super) progressed: bool,
    pub(super) admitted: bool,
    pub(super) respawn: bool,
    pub(super) signal: Arc<Notify>,
    pub(super) resident: Arc<AtomicBool>,
    /// A handle's stand-in: never ran, so its leaving is no completion.
    pub(super) holder: bool,
    /// Waiting on its client: a victim for anyone, however old it is.
    pub(super) idle: bool,
    /// Waiting at a bind for room (a commit, room given back, the planner's
    /// serve, or its turn behind a guest bind's head): it makes no progress
    /// until room comes, its refused bind holds nothing, and parked it binds
    /// once restored, so it yields as an idle process does.
    pub(super) bind_waiting: bool,
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
    /// Its fires on their way through the engine ([`super::FireInFlight`]).
    pub(super) fires: u32,
    /// When it last began or settled a fire.
    pub(super) fired_at: crate::rt::Instant,
    /// Its fires stood unsettled past `FIRE_PROGRESS`, and the planner was
    /// told.
    pub(super) fire_lapsed: bool,
    /// Fires settled since it last became resident.
    pub(super) landed: u32,
    /// Its last restore found no device room for its parked role slots
    /// (instance tables, rings, guest state): the runs of whole chunks it
    /// is short, and the pages that frees. It waits, queued, for a
    /// completion, a commit or an eviction to free them, never failed for it.
    pub(super) roles_short: Option<PendingBind>,
    /// The need its restore last went again on because its runs stood free
    /// ([`Step::EvictForRoles`](super::plan)): came back short with the
    /// same need, the need understates what its unpark takes.
    pub(super) roles_free_retry: Option<(u32, u32)>,
    /// A need known to understate its unpark: its runs stood free and the
    /// unpark was short all the same. Free runs of it are no reason to go
    /// again; room is made for it (idle cache, a grow, the chunks'
    /// occupants) until its restore lands.
    pub(super) roles_free_short: Option<(u32, u32)>,
    /// Suspended by a bind short of room, and parked: the device pool's
    /// count of room given back when its park ended. Its restore takes its
    /// own items back, so the room its park gave back is no room for the
    /// bind; it is served only once room has come free since (the M1
    /// freeze: suspended, restored straight back into the room it freed,
    /// short again, 750 times a second).
    pub(super) bind_room_after: Option<u64>,
    /// Suspended by a bind of its short of guest-state room
    /// ([`super::ResidencyPlanner::suspend_for_room`]): the room that bind
    /// asks. Its restore is served only once that room stands free beside
    /// its kv pages, as a bind short of room is
    /// ([`super::wait::WaitKind::BindRoom`]), so it never comes back to be
    /// short again.
    pub(super) bind_room: Option<PendingBind>,
    /// Waiting to be admitted: the pages its first fire grows its working
    /// sets by, past those it reserved ([`super::ResidencyPlanner::admits`]).
    pub(super) next_step: u64,
}

/// The guest-state room a suspended process's pending bind asks:
/// `pieces` free runs of `run` adjacent chunks, `pages` the kv pages that,
/// moved out whole, free them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PendingBind {
    pub(super) run: u32,
    pub(super) pieces: u32,
    pub(super) pages: u32,
}

impl Proc {
    /// A fire of its own is on its way through the engine and has been
    /// begun or settled lately: it is making progress, whatever of it is
    /// parked (see [`super::in_flight`]).
    pub(super) fn firing(&self) -> bool {
        self.fires > 0 && self.fired_at.elapsed() < super::in_flight::FIRE_PROGRESS
    }

    pub(super) fn rank(&self) -> Rank {
        Rank::of(self.key, self.seq).kept(self.kept)
    }

    /// Its place among victims ([`VictimOrder`]).
    pub(super) fn victim_order(&self) -> VictimOrder {
        VictimOrder::of(self.key, self.kept, self.came)
    }

    /// A holder at rest: past every process, whose rank is what goes first,
    /// and the longest parked past all. `order` counts up as holders park.
    pub(super) fn rest(&mut self, order: u64) {
        self.key = i64::MIN;
        self.seq = u64::MAX - order;
        self.came = self.seq;
    }

    pub(super) fn new(seq: u64) -> Self {
        Self {
            seq,
            came: seq,
            key: 0,
            kept: 0,
            held: false,
            state: Residency::Resident,
            progressed: true,
            admitted: false,
            respawn: false,
            signal: Arc::new(Notify::new()),
            resident: Arc::new(AtomicBool::new(true)),
            holder: false,
            idle: false,
            bind_waiting: false,
            parked: None,
            seatless: false,
            reseating: false,
            seated: false,
            peak: Default::default(),
            first: Default::default(),
            departure: Weak::new(),
            charged: false,
            fires: 0,
            fired_at: crate::rt::Instant::now(),
            fire_lapsed: false,
            landed: 0,
            roles_short: None,
            roles_free_retry: None,
            roles_free_short: None,
            bind_room_after: None,
            bind_room: None,
            next_step: 0,
        }
    }
}
