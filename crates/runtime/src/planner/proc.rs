use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Weak};

use tokio::sync::Notify;

use super::departure::Departure;
use super::grant::Demand;
use super::ProcessId;
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
    /// Restoring, its state back and committed, it waits for instances its
    /// rebinds cannot name yet (their pair parked with another process, or
    /// their room not given back): the instances it waits on, said by the
    /// stall dump. It moves no state and runs no fire meanwhile: nothing in
    /// flight on its account ([`super::inner::Inner::nothing_in_flight`]),
    /// or the restore that brings what it waits on, or the room, is never
    /// planned (Mac Pro at 2 GiB on 1e87ab171: a restore waiting on
    /// InstanceId(5) for good, the queue's head held behind it as a move
    /// in flight).
    pub(super) rebind_waiting: Option<String>,
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
    /// Its last restore found no device room for its parked role slots
    /// (instance tables, rings, guest state): the runs of whole chunks it
    /// is short, and the pages that frees. It waits, queued, for a
    /// completion, a commit or an eviction to free them, never failed for it.
    pub(super) roles_short: Option<PendingBind>,
    /// Why its last unpark came back short, as the roles said it (the
    /// shape and room short), for a stall's census.
    pub(super) roles_short_why: Option<String>,
    /// The need its restore last went again on because its runs stood free
    /// ([`Step::EvictForRoles`](super::plan)): came back short with the
    /// same need, the need understates what its unpark takes.
    pub(super) roles_free_retry: Option<(u32, u32)>,
    /// A need known to understate its unpark: its runs stood free and the
    /// unpark was short all the same. Free runs of it are no reason to go
    /// again; room is made for it (idle cache, a grow, the chunks'
    /// occupants) until its restore lands.
    pub(super) roles_free_short: Option<(u32, u32)>,
    /// The free runs of its need's length its refusal saw when its need
    /// was marked understated ([`Self::roles_free_short`]): once the pool
    /// holds more, it goes again (a change a quiescent pool can show, as
    /// room moves or comes), never twice on the same runs.
    pub(super) roles_free_short_at: u32,
    /// The pool's count of room given back (the map's epoch) when its last
    /// unpark was refused: a plain short head goes again only once the map
    /// has moved past it, never on the same map twice (a refusal the map
    /// disagrees with re-sent every plan pass was a hot loop).
    pub(super) roles_short_gives: Option<u64>,
    /// The length its last unpark refusal named, and the free runs of that
    /// length (within the state budget) then: it goes again on runs only
    /// once more of them stand. A count of room given back moved with every refused
    /// restore's own return, and refused restores sent each other again,
    /// round after round (Mac Pro's 2 GiB spin: 139,315 refusals, the free
    /// room no different).
    pub(super) roles_short_runs: Option<(u32, u32)>,
    /// The free chunks a guest's take reads (within the state budget) as
    /// its last unpark refusal left them: it goes again only once more
    /// stand, as well as more runs of the refused length.
    pub(super) roles_short_free: Option<u32>,
    /// Its last refusal was taken with nothing else in flight: no take of
    /// another's raced its restore, so a placement standing in that room
    /// is the placement's own mismatch, not a race.
    pub(super) roles_short_quiet: bool,
    /// Its last refusal and the room it left (free chunks, free runs of its
    /// length): the same refusal again on the same room is a settled stall
    /// ([`super::ResidencyPlanner::requeue_restore_roles`]).
    pub(super) last_refusal: Option<((u32, u32), u32, u32)>,
    /// The free chunks it was last sent on as a placement mismatch with no
    /// move to make: sent once a room, never re-sent on the same one.
    pub(super) mismatch_sent_at: Option<u32>,
    /// Sent once on that room whatever its placement reads: the takes
    /// decide ([`Self::mismatch_sent_at`]).
    pub(super) send_regardless: bool,
    /// Whether the store's own placement ([`super::pool::PoolPort::guest_need_free`],
    /// its serve's state laid first) stood its need as its last unpark
    /// refusal left the pool. Refused where it did not stand, it goes again
    /// once it does: the placement decides, not a count of free chunks
    /// beside it (Mac Pro at 5 GiB on 1e87ab171: 4 chunks free, the
    /// refused restore held on "no more chunks free than it left", 18,973
    /// no-step lines). Refused while it stood it, the placement read a
    /// room the unpark did not, and it goes again only once more chunks
    /// stand free than the refusal left ([`Self::roles_short_free`]): a
    /// decoding fleet's takes and gives sent it again ~350 times a second.
    pub(super) roles_short_stood: Option<bool>,
    /// The run its last unpark was refused, `(run, pieces)`, kept across a
    /// re-send until a restore serves it: its need always holds what the
    /// unpark asked. Cleared with the refusal at a re-send, the need fell
    /// back to its bind's runs and its own, and the run of nine its unpark
    /// was refused was never asked of the map (Mac Pro on 22b0a3e83).
    pub(super) roles_refused: Option<(u32, u32)>,
    /// Evicted while it waited in the queue: its restore goes only once no
    /// process ahead of it (older) has an entry in the queue. Restored for
    /// its own queued bind while the binds ahead of it still waited, it was
    /// evicted for them again, 1,730 times (the L40S on 46296784b).
    pub(super) evicted_queued: bool,
    /// Refused while the map showed runs of its need standing that its
    /// unpark did not see: `(map's free runs, the refusal's)`, for the
    /// census, so the need is fixed rather than the retry.
    pub(super) roles_short_mismatch: Option<(u32, Option<u32>)>,
    /// Suspended by a bind of its short of guest-state room
    /// ([`super::ResidencyPlanner::suspend_for_room`]): the room that bind
    /// asks. Its restore is served only once that room stands free beside
    /// its kv pages, as a bind short of room is
    /// ([`super::wait::WaitKind::BindRoom`]), so it never comes back to be
    /// short again.
    pub(super) bind_room: Option<PendingBind>,
    /// Restored, or its bind served, for a waiting head, and not fired
    /// since: no eviction candidate until it fires once (the anti-thrash
    /// rule's first half). Evicted back before it ran, a process was moved
    /// 443 times with the head it was moved for never served.
    pub(super) unfired: bool,
    /// The head (its process) this one was evicted for: its restore is not
    /// served while that head waits unserved (the rule's second half),
    /// whatever key the head waits under now.
    pub(super) evicted_for: Option<ProcessId>,
    /// The head this one was evicted for, its restore served ahead of it
    /// while that head stood passed over with no move to make
    /// ([`super::ResidencyPlanner::stuck_head`]): no victim for that
    /// head until it has fired. Moved back first, it went round with the
    /// head, restored past it and evicted for it again, its waits never
    /// served (seed 3392: queued for a page behind the head, so the mark
    /// alone did not hold it).
    pub(super) restored_past: Option<ProcessId>,
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
    /// The runs its restore waits on, `(run, pieces)`: its roles' need (or
    /// the bind that suspended it), and, marked understated, one run more
    /// than its refusal saw, the count it goes again on
    /// ([`Self::roles_free_short_at`]). What room is made for is what lets
    /// it go: funded for fewer, the map's cheapest runs cost nothing, no
    /// one moved, and the head waited for runs past the ones standing with
    /// nothing in flight (the planner silent).
    pub(super) fn restore_need(&self) -> Option<(u32, u32)> {
        let need = self.roles_short.or(self.bind_room)?;
        let pieces = match self.roles_free_short {
            Some((run, _)) if run == need.run => need.pieces.max(self.roles_free_short_at.saturating_add(1)),
            _ => need.pieces,
        };
        Some((need.run, pieces))
    }

    /// A fire of its own is on its way through the engine and has been
    /// begun or settled lately: it is making progress, whatever of it is
    /// parked (see [`super::in_flight`]).
    pub(super) fn firing(&self) -> bool {
        self.fires > 0 && self.fired_at.elapsed() < super::in_flight::FIRE_PROGRESS
    }

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
            progressed: true,
            admitted: false,
            respawn: false,
            signal: Arc::new(Notify::new()),
            resident: Arc::new(AtomicBool::new(true)),
            holder: false,
            idle: false,
            bind_waiting: false,
            rebind_waiting: None,
            roles_short_why: None,
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
            roles_short: None,
            roles_free_retry: None,
            roles_free_short: None,
            roles_free_short_at: 0,
            roles_short_gives: None,
            roles_short_runs: None,
            roles_short_free: None,
            roles_short_quiet: false,
            last_refusal: None,
            mismatch_sent_at: None,
            send_regardless: false,
            roles_short_stood: None,
            roles_refused: None,
            evicted_queued: false,
            roles_short_mismatch: None,
            bind_room: None,
            unfired: false,
            evicted_for: None,
            restored_past: None,
        }
    }
}
