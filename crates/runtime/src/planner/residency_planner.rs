use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::departure::Departure;
use super::grant::{AllocationGrant, Demand, DevicePageReservation, RsSlotReservation};
use super::holdings::{held_page_count, held_rs_units};
use super::inner::Inner;
use super::pool::PoolPort;
use super::proc::{PendingBind, Proc, Residency};
use super::rank::VictimOrder;
use super::reclaim::Reclaim;
use super::short::{Shortfall, Sink, WeightShed};
use super::stats::PlannerStats;
use super::wait::{EntryKey, WaitKind, Waiter};
use super::{ProcessId, exec};

pub(super) const HOST_RESERVE_DIVISOR: u32 = 8;

/// The fixed state sets a working set holds, resting or running
/// (`state-chunk` decision 4). The copy-on-write set a lane's write takes
/// while a fork or snapshot still reads the first is no part of it: the
/// fire that writes it asks for it with the rest of its demand, and once
/// held it counts against the seats ([`ResidencyPlanner::state_seats`]).
pub(super) const FIXED_SETS_PER_WORKING_SET: u32 = 1;

/// It suspends no process of its own.
const SWEEP: ProcessId = ProcessId::nil();
/// The key a demote's pages are expected under while it moves: they come
/// back once its copy has landed, and a shortfall read meanwhile counts them
/// arriving instead of evicting another process for room on its way.
const DEMOTE: ProcessId = ProcessId::max();

pub struct ResidencyPlanner {
    pub(super) inner: parking_lot::Mutex<Inner>,
    pub(super) pause_due: AtomicBool,
    pub(super) waiters: AtomicUsize,
    pub(super) nonresident: AtomicUsize,
    /// [`Inner::parked_digest`] as the last `with_inner` left it: a change
    /// re-runs the admission gates' grants, which claim nothing for a
    /// parked process.
    parked: std::sync::atomic::AtomicU64,
    drain: OnceLock<Arc<Notify>>,
    /// The runtime the planner was made on: where a tiers sweep asked from
    /// off it (the policy port's thread) runs.
    runtime: Option<tokio::runtime::Handle>,
    idle_reclaim_exhausted: std::sync::atomic::AtomicBool,
    pub(super) sweeping: AtomicBool,
    sweep_exhausted: AtomicBool,
    pub(super) demoting: AtomicBool,
    /// A host -> disk spill failed: the engine moves no slots between
    /// tiers below the device, and none is tried again.
    spill_failed: AtomicBool,
    /// The segment walk made no grow it promised: the next pass goes on
    /// past its grow steps ([`Self::no_grow`]).
    pub(super) no_grow: AtomicBool,
    pub(super) port: Arc<dyn PoolPort>,
    pub(super) stats: PlannerStats,
    /// `PoolFacts::seated_first`: a seated lane's growth before a prompt
    /// waiting to be seated, which waits for a lane to finish.
    pub(super) seated_first: bool,
    /// Held while an uncontended claim reads the rings' free room, weighs it
    /// against the seated lanes' growth and takes its grant, so prompts seated
    /// at once each see the others' pages gone.
    pub(super) seating: parking_lot::Mutex<()>,
    /// `PoolFacts::seat_to_end`: see [`super::to_end::ToEnd`].
    seat_to_end: bool,
    to_end_notify: Notify,
    /// Told when the state pool may seat more: a charged process left, a
    /// set was let go, or the pool backed more slots.
    seat_room: Notify,
    /// What its waiters lack that only the policy can free.
    pub(super) short: parking_lot::Mutex<Shortfall>,
    /// Where that is said (`pressure::raise`, for this planner's engine).
    sink: Sink,
    /// The engine's weight cache, shed before the device's shortfall is
    /// said; `None` where no cache stands in front of the pool.
    weight_shed: Option<WeightShed>,
}

/// How long a process waits at admission before it says why, and again
/// after each such span: what a stall at the gates is read from.
const ADMIT_SAYS_AFTER: std::time::Duration = std::time::Duration::from_secs(5);

/// Waits on `wait` for [`ADMIT_SAYS_AFTER`] at most: `false` once that
/// passed, for the caller to say why it waits and look again. With no
/// timer to wait on (a future polled outside a runtime), waits it out.
async fn waits_said(wait: impl std::future::Future<Output = ()>) -> bool {
    if tokio::runtime::Handle::try_current().is_err() {
        wait.await;
        return true;
    }
    crate::rt::time::timeout(ADMIT_SAYS_AFTER, wait)
        .await
        .is_ok()
}

impl ResidencyPlanner {
    pub fn new(port: Arc<dyn PoolPort>) -> Self {
        let sink = super::short::sink(port.locus().1);
        Self {
            inner: parking_lot::Mutex::new(Inner::default()),
            pause_due: AtomicBool::new(false),
            waiters: AtomicUsize::new(0),
            nonresident: AtomicUsize::new(0),
            parked: std::sync::atomic::AtomicU64::new(0),
            drain: OnceLock::new(),
            runtime: tokio::runtime::Handle::try_current().ok(),
            idle_reclaim_exhausted: std::sync::atomic::AtomicBool::new(false),
            sweeping: AtomicBool::new(false),
            sweep_exhausted: AtomicBool::new(false),
            demoting: AtomicBool::new(false),
            spill_failed: AtomicBool::new(false),
            no_grow: AtomicBool::new(false),
            port,
            stats: PlannerStats::default(),
            seated_first: false,
            seating: parking_lot::Mutex::new(()),
            seat_to_end: false,
            to_end_notify: Notify::new(),
            seat_room: Notify::new(),
            short: parking_lot::Mutex::new(Shortfall::default()),
            sink,
            weight_shed: None,
        }
    }

    /// Sheds `shed` (the engine's weight cache) before a device shortfall
    /// is said, so the policy lets no kv or state go while streamed weights
    /// above their floor could go instead.
    #[must_use]
    pub fn weight_shed(mut self, shed: WeightShed) -> Self {
        self.weight_shed = Some(shed);
        self
    }

    /// What `units` take, in bytes, at this engine's scale; a unit of
    /// unknown size counts as a byte, so a shortfall never reads as none.
    pub(super) fn bytes_of(&self, units: Reclaim) -> u64 {
        let scale = crate::policy::pressure::scale_of(self.port.locus().1);
        let page = scale.kv_page_bytes.max(1);
        let slot = scale.state_slot_bytes.max(1);
        u64::from(units.pages)
            .saturating_add(u64::from(units.rs.pages))
            .saturating_mul(page)
            .saturating_add(u64::from(units.rs.slots).saturating_mul(slot))
    }

    /// The host rows one set's state takes, which a new ask leaves free for
    /// a restore: a set parked on the host comes back by parking another in
    /// its place, which needs that much room while both stand.
    pub(super) fn restore_reserve(&self) -> RsUnits {
        let (_, total) = self.port.rs_host_stats();
        let mut windows = total.windows;
        for list in &mut windows {
            *list = (*list).min(1);
        }
        RsUnits {
            slots: total.slots.min(1),
            pages: total.pages.min(1),
            windows,
        }
    }

    /// Waiter `key` lacks `device` bytes on the device and `host` on the
    /// host that the planner's own moves cannot give it; both 0 once it
    /// lacks nothing the policy must free.
    pub(super) fn note_short(&self, key: EntryKey, device: u64, host: u64) {
        self.short.lock().note(key.1, [device, host]);
    }

    /// The head is short of room only the policy can make, and there is
    /// something the policy could let go: idle cache, a checkpoint, or a
    /// handle at rest on the device. A starved fleet then waits on the
    /// policy, up to the head's deadline, instead of restarting anyone.
    pub(super) fn waits_on_policy(&self) -> bool {
        let Some((head, resting)) = self.with_inner(|inner| {
            let (head, _) = inner.unmet_head()?;
            let resting = inner
                .procs
                .values()
                .any(|proc| proc.holder && proc.state == Residency::Resident);
            Some((head, resting))
        }) else {
            return false;
        };
        if self.short.lock().since(head.1).is_none() {
            return false;
        }
        // The host is what is short: the policy frees it by letting a set at
        // rest go, and the head fails at its deadline if none does.
        if self.short.lock().host_short(head.1) {
            return true;
        }
        let (model, engine) = self.port.locus();
        resting
            || self.port.reclaimable_idle() > 0
            || crate::store::registry::try_get(model, engine)
                .is_some_and(|stores| stores.rs.lock().unwrap().holds_checkpoints())
    }

    /// A suspend for the head found no host rows for the state it would
    /// move: it moved and dropped nothing, and the host is short of `rows`.
    pub(super) fn note_host_short(&self, rows: RsUnits) {
        let Some(head) = self.with_inner(|inner| inner.unmet_head().map(|(key, _)| key)) else {
            return;
        };
        let bytes = self.bytes_of(Reclaim { pages: 0, rs: rows });
        self.short.lock().note_host(head.1, bytes);
        self.report_short();
    }

    /// Says how short each tier is now, for the waiters still waiting:
    /// again while a shortfall stands (the gauge tells the policy again
    /// only when it has stood a while), and once as it clears.
    pub fn report_short(&self) {
        if self.short.lock().quiet() {
            return;
        }
        let waiting: HashSet<u64> = self.with_inner(|inner| {
            inner
                .queue
                .iter()
                .filter(|(_, waiter)| waiter.is_unmet())
                .map(|(key, _)| key.1)
                .collect()
        });
        let said = {
            let mut short = self.short.lock();
            short.retain(|id| waiting.contains(&id));
            short.take_said()
        };
        // Whose demand it is: the head's key bounds what the responder takes
        // (the head ranks first, so its key is the highest the waiters
        // carry), and a fleet standing still lets higher-keyed sets move down.
        let demand = crate::policy::pressure::Demand {
            up_to: self.head_retention(),
            stalled: self.is_wedged(),
        };
        for (tier, bytes) in said {
            // What the weight cache gave back comes free as the fires
            // reading it retire; the next report says what is still short.
            if tier == crate::policy::pressure::Tier::Device
                && bytes > 0
                && self.weight_shed.as_ref().is_some_and(|shed| shed() > 0)
            {
                tracing::debug!(
                    bytes,
                    "planner: the weight cache sheds before tier pressure"
                );
                continue;
            }
            (self.sink)(tier, bytes, demand);
        }
    }

    /// The error waiter `key` fails with when it has stood short of what
    /// only the policy can free past the seat deadline, rather than wait on.
    pub(super) fn short_expired(&self, key: EntryKey) -> Option<super::PlannerError> {
        let (waited, host, answered) = {
            let short = self.short.lock();
            (
                short.since(key.1)?.elapsed(),
                short.host_short(key.1),
                short.answered_nothing(key.1),
            )
        };
        // The event, not a clock: the policy answered its pressure and
        // freed nothing.
        if !answered {
            return None;
        }
        // Short of device room while processes still run, it is behind
        // them, not starved: their completions give the room back, so its
        // deadline runs again from now, and only a fleet standing still (or
        // a host short of room) fails it. Failed so, a gpt-oss prompt behind
        // 24 running 31k-token passes was refused room their completions
        // would have given it.
        if !host && !self.is_wedged() {
            self.short.lock().restart(key.1);
            return None;
        }
        let short = self.short.lock();
        let (tier, bytes) = short.worst(key.1)?;
        Some(super::PlannerError::Short {
            tier,
            bytes,
            waited,
        })
    }

    /// With `PoolFacts::seat_to_end`, a process starts only while the pools
    /// hold the started ones to the end (see [`super::to_end::ToEnd`]).
    #[must_use]
    pub fn seat_to_end(mut self, on: bool) -> Self {
        self.seat_to_end = on;
        self
    }

    /// With `PoolFacts::seat_spread`, the starts the pools do not hold to the
    /// end at once are spread over a life (see [`super::to_end::ToEnd`]).
    #[must_use]
    pub fn seat_spread(mut self, on: bool) -> Self {
        self.inner.get_mut().to_end.spread = on;
        self
    }

    /// With `PoolFacts::seated_first`, seated growth doesn't park behind a waiting
    /// prompt, which is seated only within the rings' growth room
    /// (`PoolPort::rs_growth`) and suspends only an idle lane.
    #[must_use]
    pub fn seated_first(mut self, on: bool) -> Self {
        self.seated_first = on;
        self
    }

    /// The rs units a seated lane may still claim, when the planner keeps
    /// them for it.
    pub(super) fn rs_growth(&self) -> RsUnits {
        if self.seated_first {
            self.port.rs_growth()
        } else {
            RsUnits::default()
        }
    }

    pub fn arm_drain_task(self: &Arc<Self>) {
        if !crate::rt::has_runtime() {
            return;
        };
        let notify = Arc::new(Notify::new());
        if self.drain.set(notify.clone()).is_err() {
            return;
        }
        let planner = self.clone();
        crate::rt::spawn(async move {
            loop {
                notify.notified().await;
                planner.plan();
            }
        });
    }

    /// The retention table changed: a sweep is wanted at the next plan.
    pub fn sweep_wanted(self: &Arc<Self>) {
        self.poke();
    }

    pub(super) fn poke(self: &Arc<Self>) {
        match self.drain.get() {
            Some(notify) => notify.notify_one(),
            None => self.plan(),
        }
    }

    pub(super) fn reclaim_idle_once(&self) -> bool {
        if self.idle_reclaim_exhausted.swap(true, Ordering::AcqRel) || self.port.reclaim_idle() == 0
        {
            return false;
        }
        self.idle_reclaim_exhausted.store(false, Ordering::Release);
        true
    }

    /// Idle cache the device is short of room for moves down a tier, host
    /// first then disk, rather than being dropped, where a policy keys it
    /// above the retention floor (the stores gate it: floor-keyed cache is
    /// dropped by [`Self::reclaim_idle_once`] instead, never copied): whether
    /// a move is on its way (its landing frees the pages and pokes the
    /// planner). Only under
    /// a waiter's want, never while pages stand free; what keeps back a
    /// share of the host for suspends ([`HOST_RESERVE_DIVISOR`]) is not
    /// spent on cache.
    ///
    /// `wanted`: the pages the head lacks. No more than that moves: a head
    /// short of one prompt's pages waited behind the copy of every idle page
    /// on the device.
    pub(super) fn demote_idle(self: &Arc<Self>, wanted: u32) -> bool {
        if !self.port.suspend_capable() {
            return false;
        }
        if self.demoting.swap(true, Ordering::AcqRel) {
            return true;
        }
        let (mut host_free, host_total) = self.port.host_stats();
        let (disk_free, _) = self.port.disk_stats();
        let reserve = host_total / HOST_RESERVE_DIVISOR;
        // A host with no room past its reserve is under pressure too: its
        // own idle cache, lowest key first, goes on down to disk before the
        // device's lands there, so the tiers keep the newer above the older
        // and nothing on the host is dropped for want of room.
        if host_total > 0
            && host_free <= reserve
            && disk_free > 0
            && !self.spill_failed.load(Ordering::Acquire)
        {
            if let Some(txn) = self.port.prepare_spill(reserve.max(1) as usize) {
                exec::spawn_spill(self.clone(), txn, Self::spill_done);
                return true;
            }
            // Settling pages on their disk copies may have freed some.
            host_free = self.port.host_stats().0;
        }
        let (disk_free, _) = self.port.disk_stats();
        let room = (host_free.saturating_sub(reserve) + disk_free).min(wanted.max(1));
        // The demand is the head's: it takes cache keyed at or below its own
        // key, and its own, never above (PLEX's keys say which survives;
        // equal keys take from each other, so a run with no policy demotes
        // as it did). Where nothing it may take is left and nothing running
        // will free room (the fleet stands still), higher-keyed cache moves
        // down a tier for it, lowest key first: kept, never dropped, so
        // over-demand runs slower and is never refused.
        let bound = self.head_bound();
        let mut txn = (room > 0)
            .then(|| self.port.prepare_demote(room as usize, bound))
            .flatten();
        if txn.is_none() && bound.is_some() && room > 0 && self.is_wedged() {
            txn = self.port.prepare_demote(room as usize, None);
        }
        let Some(txn) = txn else {
            self.demoting.store(false, Ordering::Release);
            return false;
        };
        let pages = txn.page_count() as u32;
        ptrace!("demote begins pages={}", pages);
        self.with_inner(|inner| {
            inner.evicting.insert(
                DEMOTE,
                Reclaim {
                    pages,
                    rs: RsUnits::default(),
                },
            )
        });
        exec::spawn_move_out(self.clone(), txn, "demote", Self::demote_done);
        true
    }

    /// Moves `owner`'s idle cache on this device down a tier, whatever its
    /// key (the policy layer chose it), as far as the tiers below have room
    /// past the host's reserve: whether a move is on its way.
    pub fn demote_owner(self: &Arc<Self>, owner: ProcessId) -> bool {
        // Its own move, beside the planner's (it takes the `demoting` turn
        // of neither): A6000's conform answered false for every set but the
        // first while that turn was held, and freed nothing.
        let why = if !self.port.suspend_capable() {
            Some("the device moves nothing below it")
        } else {
            None
        };
        let (host_free, host_total) = self.port.host_stats();
        let (disk_free, _) = self.port.disk_stats();
        let room = host_free.saturating_sub(host_total / HOST_RESERVE_DIVISOR) + disk_free;
        let why = why.or((room == 0).then_some("no room below the device past the host's reserve"));
        let bound = crate::store::kv::DemoteBound {
            up_to: i64::MIN,
            own: Some(owner),
        };
        let txn = if why.is_none() {
            self.port.prepare_demote(room as usize, Some(bound))
        } else {
            None
        };
        let Some(txn) = txn else {
            tracing::debug!(
                %owner,
                room,
                why = why.unwrap_or(
                    "none of its cache is idle on the device above the floor (in use, moving, \
                     already below, or floor-keyed: dropped, not moved)"
                ),
                "planner: a set the policy chose does not move down"
            );
            return false;
        };
        exec::spawn_move_out(self.clone(), txn, "demote", Self::owner_demote_done);
        true
    }

    fn owner_demote_done(self: &Arc<Self>, moved: bool) {
        if moved {
            self.re_arm_idle_reclaim();
        }
        self.poke();
    }

    /// A spill that failed is not tried again: the host's cache stays where
    /// it is and the device's goes to disk past it, as before spilling.
    fn spill_done(self: &Arc<Self>, moved: bool) {
        if !moved {
            self.spill_failed.store(true, Ordering::Release);
        }
        self.demote_done(moved);
    }

    /// The retention table changed: idle cache its tiers rows allow
    /// nowhere it is goes now, and what they keep off the device moves
    /// down now, with no waiter's want.
    pub(super) fn sweep_tiers(self: &Arc<Self>) {
        // A move runs on the runtime: called off it, the sweep goes there,
        // or (a planner made off any runtime) waits for the next plan that
        // runs on one.
        if !crate::rt::has_runtime() {
            match &self.runtime {
                Some(runtime) => {
                    let planner = self.clone();
                    runtime.spawn(async move { planner.sweep_tiers() });
                }
                None => crate::policy::retention::idle_cache_appeared(),
            }
            return;
        }
        tracing::debug!(target: "pie::retention", busy = self.demoting.load(Ordering::Acquire), "tiers sweep");
        // A process waiting on its client whose row keeps its KV off the
        // device is paused, its KV moved down for the wait.
        let off: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .filter(|(pid, proc)| proc.idle && Self::kept_off_device(**pid))
                .map(|(pid, _)| *pid)
                .collect()
        });
        let idle = self.with_inner(|inner| inner.procs.values().filter(|proc| proc.idle).count());
        tracing::debug!(target: "pie::retention", idle, off = off.len(), "tiers sweep: idle processes");
        if !off.is_empty() {
            self.pause(off, |proc| proc.idle);
        }
        if self.port.drop_disallowed() > 0 {
            self.re_arm_idle_reclaim();
            self.poke();
        }
        if self.demoting.swap(true, Ordering::AcqRel) {
            return;
        }
        let txn = self.port.prepare_demote_now();
        tracing::debug!(target: "pie::retention", moves = txn.is_some(), "tiers sweep: demote now");
        match txn {
            Some(txn) => exec::spawn_move_out(self.clone(), txn, "demote", Self::tiers_done),
            None => self.demoting.store(false, Ordering::Release),
        }
    }

    /// Whether `pid`'s own working set may not sit on the device.
    fn kept_off_device(pid: ProcessId) -> bool {
        use crate::policy::retention::{DEVICE, Installed, Keys};
        Installed.tiers_at(Some(pid), 0) & DEVICE == 0
    }

    /// A move a tiers row asked for landed: the next group goes, until none
    /// is left.
    fn tiers_done(self: &Arc<Self>, moved: bool) {
        tracing::debug!(target: "pie::retention", moved, "tiers sweep: demote landed");
        self.demote_done(moved);
        if moved {
            self.sweep_tiers();
        }
    }

    fn demote_done(self: &Arc<Self>, moved: bool) {
        ptrace!("demote done moved={}", moved);
        self.with_inner(|inner| inner.settle_eviction(DEMOTE));
        self.demoting.store(false, Ordering::Release);
        if moved {
            self.re_arm_idle_reclaim();
        }
        self.poke();
    }

    /// What the demand heading the queue may demote: idle cache keyed at
    /// or below its process's key, and its own.
    pub(super) fn head_bound(&self) -> Option<crate::store::kv::DemoteBound> {
        let up_to = self.head_retention()?;
        let pid = self.with_counts(|inner| inner.unmet_head().map(|(_, w)| w.pid))?;
        Some(crate::store::kv::DemoteBound {
            up_to,
            own: Some(crate::policy::schedule::plex_id(pid)),
        })
    }

    /// The retention key of the head's own sets (what the policy keeps its
    /// working set and cache at, the policy's default where it states none):
    /// what bounds the cache and resting sets its demand may take, like with
    /// like. Schedule priority orders who is served first, not what is kept.
    pub(super) fn head_retention(&self) -> Option<i64> {
        let pid = self.with_counts(|inner| inner.unmet_head().map(|(_, w)| w.pid))?;
        Some(holder_key(crate::policy::schedule::plex_id(pid)))
    }

    /// The schedule key of the process whose demand heads the queue, as the
    /// victim choice reads it (the processes it may park are keyed at or
    /// below it, like with like): one no row names yet carries the table's
    /// default; `None` where no table is installed.
    #[cfg(test)]
    pub(super) fn head_key(&self) -> Option<i64> {
        let key = self.with_counts(|inner| {
            let (_, waiter) = inner.unmet_head()?;
            inner.procs.get(&waiter.pid).map(|proc| proc.key)
        })?;
        crate::policy::schedule::current().map(|_| key)
    }

    pub(super) fn re_arm_idle_reclaim(&self) {
        self.idle_reclaim_exhausted.store(false, Ordering::Release);
        self.sweep_exhausted.store(false, Ordering::Release);
    }

    /// Not what the head and restores under way read: they would bring it
    /// straight back.
    /// With `tails_only`, only the tails past a head a key keeps go: a set
    /// keyed whole stays as it is (what a head short of state reaches for,
    /// as a whole set moves with its state instead).
    pub(super) fn sweep_stranded(self: &Arc<Self>, wanted: u32, tails_only: bool) -> bool {
        if self.sweep_exhausted.load(Ordering::Acquire)
            || self.sweeping.swap(true, Ordering::AcqRel)
        {
            return false;
        }
        let (swap_free, swap_total) = self.swap_stats();
        let room = swap_free.saturating_sub(swap_total / HOST_RESERVE_DIVISOR) as usize;
        let (model, engine) = self.port.locus();
        // A set at rest keyed above the head's demand stays for it, as idle
        // cache keyed so does ([`Self::demote_idle`]), while running work
        // will free room; with the fleet standing still, it moves down.
        let up_to = if self.is_wedged() { None } else { self.head_retention() };
        let mut above: Vec<ProcessId> = Vec::new();
        // What rests goes in the order it would be evicted, and no more of
        // it than is wanted: a policy's keys hold here too, the head of a set
        // they keep going last.
        let ordered: Vec<(crate::store::kv::page_table::WorkingSetId, u64)> = self
            .with_inner(|inner| {
                let mut holders: Vec<_> = inner
                    .procs
                    .iter()
                    .filter(|(_, proc)| proc.holder)
                    .filter(|(pid, proc)| {
                        let under = up_to.is_none_or(|up_to| proc.kept <= up_to);
                        if !under {
                            above.push(**pid);
                        }
                        under
                    })
                    .map(|(pid, proc)| (proc.victim_order(), *pid))
                    .collect();
                holders.sort_unstable();
                holders
            })
            .into_iter()
            .flat_map(|(_, pid)| {
                let head = holder_head(pid);
                crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine)
                    .into_iter()
                    .map(move |ws| (ws, head))
            })
            .collect();
        // Tails only: a set keyed whole stays where it is (a sweep moves
        // every suspended set it is not told to keep).
        let (order, whole): (Vec<_>, Vec<_>) = ordered
            .into_iter()
            .partition(|&(_, head)| !tails_only || head > 0);
        if order.is_empty() && tails_only {
            self.sweeping.store(false, Ordering::Release);
            return false;
        }
        let mut keep: HashSet<_> = self
            .with_inner(|inner| {
                let mut pids: Vec<ProcessId> = inner
                    .procs
                    .iter()
                    .filter(|(_, proc)| proc.state == Residency::Restoring)
                    .map(|(pid, _)| *pid)
                    .collect();
                pids.extend(inner.unmet_head().map(|(_, waiter)| waiter.pid));
                pids
            })
            .into_iter()
            .flat_map(|pid| {
                crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine)
            })
            .collect();
        keep.extend(whole.into_iter().map(|(ws, _)| ws));
        keep.extend(above.into_iter().flat_map(|pid| {
            crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine)
        }));
        let txn = (room > 0)
            .then(|| {
                if tails_only {
                    self.port
                        .prepare_tail_sweep(&keep, room, &order, wanted as usize)
                } else {
                    self.port
                        .prepare_sweep(&keep, room, &order, wanted as usize)
                }
            })
            .flatten();
        let Some(txn) = txn else {
            self.sweeping.store(false, Ordering::Release);
            self.sweep_exhausted.store(true, Ordering::Release);
            return false;
        };
        let pages = txn.page_count() as u32;
        self.with_inner(|inner| {
            inner.evicting.insert(
                SWEEP,
                Reclaim {
                    pages,
                    rs: RsUnits::default(),
                },
            )
        });
        exec::spawn_move_out(self.clone(), txn, "sweep", Self::sweep_done);
        true
    }

    fn sweep_done(self: &Arc<Self>, moved: bool) {
        self.with_inner(|inner| inner.settle_eviction(SWEEP));
        self.sweeping.store(false, Ordering::Release);
        if moved {
            self.re_arm_idle_reclaim();
        } else {
            // Not again until something changes: a failed copy would fail
            // the same way.
            self.sweep_exhausted.store(true, Ordering::Release);
        }
        self.poke();
    }

    pub fn locus(&self) -> (usize, usize) {
        self.port.locus()
    }

    pub(super) fn with_inner<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        let mut inner = self.lock_inner();
        let result = f(&mut inner);
        self.waiters.store(inner.queue.len(), Ordering::Release);
        self.nonresident
            .store(inner.nonresident_count(), Ordering::Release);
        inner.admit_unpreempted();
        let parked = inner.parked_digest();
        drop(inner);
        // A process parked or un-parked: the gates' bars move with it.
        if self.parked.swap(parked, Ordering::AcqRel) != parked {
            crate::inferlet::process::regrant_admission_soon();
        }
        result
    }

    /// The processes that cannot fire now ([`Inner::parked`]).
    pub fn parked_pids(&self) -> HashSet<ProcessId> {
        self.lock_inner().parked().collect()
    }

    /// [`Self::with_inner`] for `f` that touches no process's residency,
    /// rank or parking and no queue: what `with_inner` refreshes after it
    /// (the waiters, the nonresident count, the admissions a preemption
    /// held back) reads all the processes, and a fire's own count moves
    /// twice a lane a step.
    pub(super) fn with_counts<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        f(&mut self.lock_inner())
    }

    pub fn register(&self, pid: ProcessId) {
        self.with_inner(|inner| {
            let seq = inner.next_seq;
            inner.next_seq += 1;
            let mut proc = Proc::new(seq);
            proc.key = crate::policy::schedule::key(pid);
            proc.held = crate::policy::schedule::held(pid);
            inner.procs.insert(pid, proc);
        });
    }

    /// Registers what holds a handle's working set between processes: idle
    /// from the start, so the first to want its pages takes them. It ranks
    /// past every process, the longest parked last of all: a turn resumes
    /// what it just left, so the handle left longest goes first, where the
    /// youngest process would. `parked` is what it owes the device, for a set
    /// that comes already evicted.
    pub fn register_holder(self: &Arc<Self>, pid: ProcessId, parked: Option<Demand>) {
        self.with_inner(|inner| {
            let seq = inner.next_seq;
            inner.next_seq += 1;
            let mut proc = Proc::new(0);
            proc.rest(seq);
            proc.kept = holder_key(pid);
            proc.holder = true;
            proc.admitted = true;
            proc.idle = true;
            if let Some(demand) = parked {
                proc.state = Residency::Evicted;
                proc.resident.store(false, Ordering::Release);
                proc.parked = Some(demand);
            }
            inner.procs.insert(pid, proc);
        });
        self.re_arm_idle_reclaim();
        self.poke();
    }

    /// Reads each holder's retention key again: the table has changed.
    pub fn rekey_holders(&self) {
        self.with_inner(|inner| {
            for (pid, proc) in inner.procs.iter_mut().filter(|(_, proc)| proc.holder) {
                proc.kept = holder_key(*pid);
            }
        });
    }

    /// Whether a holder's state is on the device, if it is one here.
    pub fn holder_resident(&self, pid: ProcessId) -> Option<bool> {
        self.with_inner(|inner| {
            let proc = inner.procs.get(&pid).filter(|proc| proc.holder)?;
            Some(proc.state == Residency::Resident)
        })
    }

    /// A launch is taking this holder's set: no one's victim from here.
    pub fn claim_holder(&self, pid: ProcessId) {
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.idle = false;
            }
        });
    }

    /// The launch did not happen.
    pub fn unclaim_holder(self: &Arc<Self>, pid: ProcessId) {
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.idle = true;
            }
        });
        self.poke();
    }

    /// Whether an eviction has this holder's set in hand.
    /// Waits until `pid` is not being moved off the device: its eviction
    /// ends (or rolls back) and tells its signal. At once where it is not.
    pub async fn wait_not_moving(&self, pid: ProcessId) {
        loop {
            let Some(signal) = self
                .lock_inner()
                .procs
                .get(&pid)
                .map(|proc| proc.signal.clone())
            else {
                return;
            };
            let notified = signal.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !self.holder_moving(pid) {
                return;
            }
            notified.await;
        }
    }

    pub fn holder_moving(&self, pid: ProcessId) -> bool {
        self.lock_inner()
            .procs
            .get(&pid)
            .is_some_and(|proc| proc.state == Residency::Evicting)
    }

    /// `pid` took sets that are off the device: it restores them before it
    /// runs, as if it had been evicted with them.
    pub fn owe(self: &Arc<Self>, pid: ProcessId, demand: Demand) {
        self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return;
            };
            proc.state = Residency::Evicted;
            inner.enqueue(pid, WaitKind::Restore { demand });
        });
        self.poke();
    }

    pub fn register_with_seq(&self, pid: ProcessId, seq: u64) {
        self.with_inner(|inner| {
            let mut proc = Proc::new(seq);
            proc.key = crate::policy::schedule::key(pid);
            proc.held = crate::policy::schedule::held(pid);
            proc.respawn = inner
                .killing
                .iter()
                .any(|killed| inner.procs.get(killed).is_some_and(|p| p.seq == seq));
            inner.procs.insert(pid, proc);
            // A respawn is resident anew, whatever seq it keeps.
            inner.came_back(pid);
        });
    }

    pub fn spawn_seq(&self, pid: ProcessId) -> Option<u64> {
        self.lock_inner().procs.get(&pid).map(|proc| proc.seq)
    }

    /// Holds `pid` back from starting while an older process is preempted,
    /// so new work does not take the pages that process needs to come back.
    pub async fn admit(&self, pid: ProcessId) {
        loop {
            let notify = self.with_inner(|inner| {
                let rank = inner.procs.get(&pid)?.rank();
                if inner.preempted_floor().is_none_or(|floor| rank <= floor) {
                    return None;
                }
                let notify = Arc::new(Notify::new());
                inner.admitting.insert((rank, pid), notify.clone());
                Some(notify)
            });
            match notify {
                Some(notify) => {
                    if !waits_said(notify.notified()).await {
                        let holders: Vec<String> = self.with_inner(|inner| {
                            inner
                                .procs
                                .iter()
                                .filter(|(holder, proc)| {
                                    (proc.state != Residency::Resident && proc.parked.is_none())
                                        || proc.respawn
                                        || inner.killing.contains(holder)
                                })
                                .map(|(holder, proc)| format!("{holder}: {:?}", proc.state))
                                .collect()
                        });
                        tracing::warn!(
                            %pid,
                            ?holders,
                            "a process waits at admission behind a preempted one"
                        );
                    }
                }
                None => break,
            }
        }
        self.admit_state(pid).await;
        if self.seat_to_end {
            self.admit_to_end(pid).await;
        }
    }

    /// The processes the state pool runs at once: each holds
    /// [`FIXED_SETS_PER_WORKING_SET`] fixed sets, and the sets an admitted
    /// process holds past those (a fork's copy-on-write sets, a second
    /// context's) seat no one else while held. At least one runs. `None`
    /// where the model keeps no fixed state.
    pub(super) fn state_seats(&self) -> Option<usize> {
        let (_, total) = self.port.rs_stats();
        if total.slots == 0 {
            return None;
        }
        let (model, engine) = self.port.locus();
        let charged: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .filter(|(_, proc)| proc.charged)
                .map(|(&pid, _)| pid)
                .collect()
        });
        // Read off the planner's lock: the holdings take the store's.
        let past: u32 = charged
            .into_iter()
            .map(|pid| {
                held_rs_units(pid, model, engine)
                    .slots
                    .saturating_sub(FIXED_SETS_PER_WORKING_SET)
            })
            .sum();
        Some((total.slots.saturating_sub(past) / FIXED_SETS_PER_WORKING_SET).max(1) as usize)
    }

    /// Holds `pid` back until the state pool has the fixed set it holds
    /// free of those the admitted hold or are owed, so a pool of fixed state
    /// never runs more working sets than it can write: the runtime frees no
    /// state on its own to make room for one admitted past it. A fork's
    /// copy-on-write set is no seat's: its fire asks for it as any demand
    /// is asked, and waits or is refused by name there.
    async fn admit_state(&self, pid: ProcessId) {
        loop {
            let notified = self.seat_room.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            // Read again on each wake: the pool may have backed more, or a
            // fork let its copies go.
            let (_, total) = self.port.rs_stats();
            if total.slots == 0 {
                return;
            }
            // The holdings are read only where the whole pool would seat it.
            let most = (total.slots / FIXED_SETS_PER_WORKING_SET).max(1) as usize;
            let room = self.with_inner(|inner| {
                let charged = inner.procs.values().filter(|proc| proc.charged).count();
                inner
                    .procs
                    .get(&pid)
                    .is_none_or(|proc| proc.charged || charged < most)
            });
            let seats = if room { self.state_seats() } else { None };
            let passed = seats.is_some_and(|seats| {
                self.with_inner(|inner| {
                    let charged = inner.procs.values().filter(|proc| proc.charged).count();
                    let Some(proc) = inner.procs.get_mut(&pid) else {
                        return true;
                    };
                    if proc.charged || charged < seats {
                        proc.charged = true;
                        return true;
                    }
                    false
                })
            });
            if passed {
                return;
            }
            if !waits_said(notified).await {
                let charged = self
                    .with_inner(|inner| inner.procs.values().filter(|proc| proc.charged).count());
                let seats = seats.unwrap_or(most);
                tracing::warn!(
                    %pid,
                    seats,
                    charged,
                    "a process waits at admission for a state seat"
                );
            }
        }
    }

    /// What `pid` holds once `demand` lands, kept as its peak.
    pub fn note_peak(&self, pid: ProcessId, demand: Demand) {
        if !self.seat_to_end {
            return;
        }
        let (model, engine) = self.port.locus();
        let kv = held_page_count(pid, model, engine) + demand.kv_pages;
        let rs = super::holdings::held_rs_units(pid, model, engine).add(demand.rs);
        if let Some(proc) = self.lock_inner().procs.get_mut(&pid) {
            if proc.peak.0 == 0 && proc.peak.1.is_zero() {
                proc.first = (kv, rs);
            }
            proc.peak = (
                proc.peak.0.max(kv),
                RsUnits {
                    slots: proc.peak.1.slots.max(rs.slots),
                    pages: proc.peak.1.pages.max(rs.pages),
                    ..RsUnits::default()
                },
            );
        }
    }

    /// Holds `pid` back, oldest first, until the pools hold it to the end
    /// beside the processes already started.
    async fn admit_to_end(&self, pid: ProcessId) {
        loop {
            let notified = self.to_end_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let (_, kv_total) = self.port.device_stats();
            let (_, rs_total) = self.port.rs_stats();
            let now = std::time::Instant::now();
            let passed = self.with_inner(|inner| {
                let Some(rank) = inner.procs.get(&pid).map(Proc::rank) else {
                    return Ok(());
                };
                inner.to_end.waiting.insert((rank, pid));
                // By pid: a reschedule may have moved its rank since it queued.
                if inner
                    .to_end
                    .waiting
                    .first()
                    .is_none_or(|(_, first)| *first != pid)
                {
                    return Err(None);
                }
                inner.to_end.room(&inner.procs, kv_total, rs_total, now)?;
                inner.to_end.waiting.retain(|(_, waiting)| *waiting != pid);
                inner.to_end.live.insert(pid, now);
                Ok(())
            });
            match passed {
                Ok(()) => {
                    self.to_end_notify.notify_waiters();
                    return;
                }
                Err(None) => notified.await,
                Err(Some(spread)) => {
                    let _ = crate::rt::time::timeout(spread, notified).await;
                }
            }
        }
    }

    /// Queues the restore of a parked `pid` because a message for it is on
    /// its way. It stays idle until it reads the message, so a process
    /// restored early remains the first victim and never pins pages.
    pub fn wake(self: &Arc<Self>, pid: ProcessId) {
        let queued = self.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid)?;
            if proc.held || proc.seatless {
                return None;
            }
            let demand = proc.parked.take()?;
            inner.enqueue(pid, WaitKind::Restore { demand });
            Some(())
        });
        if queued.is_some() {
            self.poke();
        }
    }

    /// Marks `pid` as waiting at a bind for room, or past it
    /// ([`super::proc::Proc::bind_waiting`]): waiting, it is a victim as an
    /// idle process is, and the planner looks again for a head it may now
    /// fund.
    pub fn set_bind_waiting(self: &Arc<Self>, pid: ProcessId, waiting: bool) {
        let (changed, released) = self.with_inner(|inner| {
            let changed = inner.procs.get_mut(&pid).is_some_and(|proc| {
                let was = proc.bind_waiting;
                proc.bind_waiting = waiting;
                was != waiting
            });
            // The suspended bind it was restored for waits for room again:
            // its restore is over, and the next bind-suspended restore may
            // go. Held for it, every other one waited on a bind that waits
            // behind them (Mac Pro's beam mix: a restored process short at
            // its bind again, the queue's head a bind-suspended restore,
            // flat with every thread parked).
            let released = waiting && inner.bind_restoring == Some(pid);
            if released {
                inner.bind_restoring = None;
            }
            (changed, released)
        });
        if (changed && waiting) || released {
            self.poke();
        }
    }

    /// Marks `pid` as waiting on its client, or as awake again. Waking a
    /// parked process queues its restore, ahead of its next fire.
    pub fn set_idle(self: &Arc<Self>, pid: ProcessId, idle: bool) {
        self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return;
            };
            proc.idle = idle;
            if idle {
                // A tiers row may keep its working set off the device while
                // it waits.
                crate::policy::retention::idle_cache_appeared();
            }
            if self.seat_to_end {
                self.to_end_notify.notify_waiters();
            }
            // Waiting on its client, it asks for no room: admission weighs
            // the others again.
            if idle && proc.admitted {
                crate::inferlet::process::regrant_admission_soon();
            }
            let parked = if idle || proc.held || proc.seatless {
                None
            } else {
                proc.parked.take()
            };
            if let Some(demand) = parked {
                inner.enqueue(pid, WaitKind::Restore { demand });
            }
        });
        self.poke();
    }

    pub fn reschedule(self: &Arc<Self>) {
        let schedule = crate::policy::schedule::current();
        self.with_inner(|inner| {
            inner.reschedule(|pid| {
                schedule
                    .as_ref()
                    .map_or((0, false), |s| (s.key(*pid), s.held(*pid)))
            });
        });
        self.vacate_held_seats();
        self.pause_due.store(true, Ordering::Release);
        self.poke();
    }

    /// A paused process must not keep out one the schedule chose.
    pub(super) fn vacate_held_seats(&self) {
        if !crate::inferlet::process::seats_capped() {
            return;
        }
        let pids: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter_mut()
                .filter(|(_, p)| {
                    p.held && !p.seatless && p.state == Residency::Evicted && p.parked.is_some()
                })
                .map(|(pid, p)| {
                    p.seatless = true;
                    *pid
                })
                .collect()
        });
        for pid in pids {
            crate::inferlet::process::vacate_seat(pid);
        }
    }

    pub(super) fn reseat_released(self: &Arc<Self>) {
        let pids: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter_mut()
                .filter(|(_, p)| {
                    p.seatless
                        && !p.reseating
                        && !p.held
                        && p.state == Residency::Evicted
                        && p.parked.is_some()
                })
                .map(|(pid, p)| {
                    p.reseating = true;
                    *pid
                })
                .collect()
        });
        for pid in pids {
            exec::spawn_reseat(self.clone(), pid);
        }
    }

    pub(super) fn reseated(self: &Arc<Self>, pid: ProcessId, seat: crate::inferlet::process::Seat) {
        let queued = self.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid)?;
            proc.reseating = false;
            if proc.held || proc.state != Residency::Evicted {
                return None;
            }
            let demand = proc.parked.take()?;
            proc.seatless = false;
            inner.enqueue(pid, WaitKind::Restore { demand });
            Some(())
        });
        if queued.is_some() {
            crate::inferlet::process::reseat(pid, seat);
            self.poke();
        }
    }

    /// The next plan tries again; poking here would recurse off the runtime.
    pub(super) fn reseat_failed(&self, pid: ProcessId) {
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.reseating = false;
            }
        });
    }

    pub fn can_pause(&self) -> bool {
        self.port.suspend_capable()
    }

    /// Suspends `pid` to give back the room its parked items hold on the
    /// device: a bind of its short of room it holds part of itself would
    /// wait on itself ([`crate::store::roles::bind_when_resident`]).
    /// Whether a suspension began. The bind asks `pieces` runs of `run`
    /// adjacent chunks: its restore waits for them beside its kv pages
    /// ([`super::proc::Proc::bind_room`]), never served into the same
    /// shortfall.
    pub fn suspend_for_room(self: &Arc<Self>, pid: ProcessId, run: u32, pieces: u32) -> bool {
        let need = PendingBind {
            run: run.max(1),
            pieces,
            pages: self.port.chunks_as_pages(run.max(1).saturating_mul(pieces)),
        };

        let before = self.stats.evictions_started.load(Ordering::Relaxed);
        ptrace!(
            "suspend-for-room pid={} run={} pieces={}",
            pid,
            need.run,
            need.pieces
        );
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.bind_room = Some(need);
            }
        });
        self.pause(vec![pid], |_| true);
        // Not suspended, it waits as any bind short of room: no restore
        // reads the room.
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid)
                && proc.state == Residency::Resident
            {
                proc.bind_room = None;
            }
        });
        self.stats.evictions_started.load(Ordering::Relaxed) > before
    }

    /// A worker with no host or disk tier leaves it running.
    pub(super) fn pause_held(self: &Arc<Self>) {
        let held: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .filter(|(_, proc)| proc.held)
                .map(|(pid, _)| *pid)
                .collect()
        });
        self.pause(held, |proc| proc.held);
    }

    /// Suspends those of `pids` that are running and a lower tier can hold.
    fn pause(self: &Arc<Self>, pids: Vec<ProcessId>, still: fn(&Proc) -> bool) {
        if !self.port.suspend_capable() {
            ptrace!("pause: cannot suspend");
            return;
        }
        let pausable: Vec<(ProcessId, VictimOrder)> = self.with_inner(|inner| {
            pids.iter()
                .filter_map(|pid| {
                    let proc = inner.procs.get(pid)?;
                    (proc.admitted
                        && proc.state == Residency::Resident
                        && !inner.host_swap_blocked.contains(pid)
                        && !inner.prepare_blocked.contains(pid))
                    .then(|| (*pid, proc.victim_order()))
                })
                .collect()
        });
        ptrace!("pause candidates={}", pausable.len());
        if pausable.is_empty() {
            return;
        }
        let (model, engine) = self.port.locus();
        let everything = Reclaim {
            pages: u32::MAX,
            rs: RsUnits {
                slots: u32::MAX,
                pages: u32::MAX,
                ..RsUnits::default()
            },
        };
        let (picks, unhostable) =
            self.quote_and_pick(pausable, everything, self.swap_room(), 0, model, engine);
        // No tier takes them: they keep running.
        ptrace!(
            "pause picks={} unhostable={} room={:?}",
            picks.len(),
            unhostable.len(),
            self.swap_room()
        );
        let (wake, spawned) = self.with_inner(|inner| {
            let (mut wake, mut spawned) = (Vec::new(), Vec::new());
            for (pid, expected) in picks {
                if !inner
                    .procs
                    .get(&pid)
                    .is_some_and(|proc| still(proc) && proc.state == Residency::Resident)
                {
                    continue;
                }
                wake.extend(inner.begin_eviction(pid, expected));
                spawned.push(pid);
            }
            (wake, spawned)
        });
        for pid in &spawned {
            ptrace!(
                "pause-evict pid={} landed={}",
                pid,
                self.with_inner(|inner| inner.procs.get(pid).map_or(0, |p| p.landed))
            );
        }
        self.start_evictions(wake, spawned);
    }

    pub(super) fn start_evictions(self: &Arc<Self>, wake: Vec<Arc<Notify>>, pids: Vec<ProcessId>) {
        for notify in wake {
            notify.notify_waiters();
        }
        for pid in pids {
            self.stats.evictions_started.fetch_add(1, Ordering::Relaxed);
            exec::spawn_evict(self.clone(), pid);
        }
    }

    pub fn note_admitted(&self, pid: ProcessId) {
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.admitted = true;
                proc.respawn = false;
                proc.next_step = 0;
            }
        });
    }

    /// `pid`, waiting to be admitted, grows its working sets by `pages`
    /// with its first fire, past what it reserved.
    pub fn note_next_step(&self, pid: ProcessId, pages: u64) {
        self.with_counts(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.next_step = pages;
            }
        });
    }

    /// Whether `pid` is admitted beside `admitted` (the processes holding
    /// execution seats; those of other devices are not weighed here): the
    /// working sets of those running, `pid`'s own with its next step, fit
    /// the device room the kv budget states. A process waiting on its
    /// client, or a handle's stand-in, asks for no room until it runs
    /// again. With none running, it is admitted: one runs, whatever it
    /// asks (a set that can never fit is refused by name where it asks).
    /// Whether the running among `admitted` fill every tier: no next
    /// process, its first step a page at the least, fits beside them on
    /// the device, even once all they hold there and all idle cache moved
    /// below it as far as the swap has room.
    pub fn full(&self, admitted: &[ProcessId]) -> bool {
        let running: Vec<ProcessId> = self.with_counts(|inner| {
            admitted
                .iter()
                .copied()
                .filter(|pid| {
                    inner
                        .procs
                        .get(pid)
                        .is_some_and(|proc| !proc.holder && !proc.idle)
                })
                .collect()
        });
        if running.is_empty() {
            return false;
        }
        let parts = self.port.working_set_pages(&running);
        let movable = parts
            .resident
            .saturating_add(parts.roles)
            .saturating_add(self.port.idle_pages_up_to(i64::MAX));
        let room = u64::from(self.port.device_stats().1)
            .saturating_add(movable.min(u64::from(self.swap_stats().0)));
        parts.total() >= room
    }

    /// Whether `pid` is admitted beside `admitted` (the processes holding
    /// execution seats; those of other devices are not weighed here): the
    /// working sets of those running and `pid`'s own with its next step fit
    /// the device, or what they lack there the planner may move below it
    /// for `pid` under its key (idle cache, and the running sets it may
    /// park: keyed below it, or alike and younger), as far as the swap has
    /// room. Never past the device and the swap together. A process waiting
    /// on its client, or a handle's stand-in, asks for no room until it
    /// runs again. With none running, it is admitted: one runs, whatever it
    /// asks (a set that can never fit is refused by name where it asks).
    pub fn admits(&self, pid: ProcessId, admitted: &[ProcessId]) -> bool {
        let (mut running, parkable, next_step, key) = self.with_counts(|inner| {
            let rank = inner.procs.get(&pid).map(Proc::rank);
            let mut parkable = Vec::new();
            let running: Vec<ProcessId> = admitted
                .iter()
                .copied()
                .filter(|other| *other != pid)
                .filter(|other| {
                    let Some(proc) = inner.procs.get(other) else {
                        return false;
                    };
                    let runs = !proc.holder && !proc.idle;
                    if runs && rank.is_some_and(|rank| proc.rank() > rank) {
                        parkable.push(*other);
                    }
                    runs
                })
                .collect();
            let proc = inner.procs.get(&pid);
            (
                running,
                parkable,
                proc.map_or(0, |proc| proc.next_step),
                proc.map_or(0, |proc| proc.key),
            )
        });
        if running.is_empty() {
            return true;
        }
        running.push(pid);
        // Read off the planner's lock: the working sets take the stores'.
        let parts = self.port.working_set_pages(&running);
        let need = parts.total().saturating_add(next_step);
        let (free, total) = self.port.device_stats();
        if need <= u64::from(total) {
            return true;
        }
        let short = need - u64::from(total);
        let parked = self.port.working_set_pages(&parkable);
        // Cache is weighed by the waiter's retention key, like with like.
        let idle = self
            .port
            .idle_pages_up_to(holder_key(crate::policy::schedule::plex_id(pid)));
        let swap_free = u64::from(self.swap_stats().0);
        let movable = parked
            .resident
            .saturating_add(parked.roles)
            .saturating_add(idle)
            .min(swap_free);
        let fits = short <= movable;
        if !fits && tracing::enabled!(target: "pie::admission", tracing::Level::DEBUG) {
            let own = self.port.working_set_pages(&[pid]);
            tracing::debug!(
                target: "pie::admission",
                %pid,
                key,
                running = running.len() - 1,
                total,
                need,
                resident = parts.resident,
                reserved = parts.reserved,
                state = parts.state,
                roles = parts.roles,
                next_step,
                own_resident = own.resident,
                own_reserved = own.reserved,
                own_roles = own.roles,
                free,
                parkable = parkable.len(),
                parkable_resident = parked.resident,
                idle,
                swap_free,
                "refused for room"
            );
        }
        fits
    }

    /// The hold `pid` keeps on its place here past its exit (see
    /// [`Departure`]): the same one for every holder while any lives, none
    /// once `pid` has left.
    pub fn departure(self: &Arc<Self>, pid: ProcessId) -> Option<Arc<Departure>> {
        self.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid)?;
            if let Some(departure) = proc.departure.upgrade() {
                return Some(departure);
            }
            let departure = Arc::new(Departure::new(self, pid));
            proc.departure = Arc::downgrade(&departure);
            Some(departure)
        })
    }

    /// `pid`'s last hold dropped: what its working sets held is back.
    pub(super) fn departed(self: &Arc<Self>, pid: ProcessId) {
        let settled = self.with_inner(|inner| {
            !inner.procs.contains_key(&pid) && inner.evicting.remove(&pid).is_some()
        });
        if settled {
            self.re_arm_idle_reclaim();
            self.poke();
        }
    }

    /// `pid` leaves: its client has heard, its seat and asks go now. What
    /// its working sets hold stays on its way back, as an eviction's would,
    /// until its last [`Departure`] hold drops (each set's release drops its
    /// own, however long a fire leases it), so the fleet is not read as
    /// wedged for pages no one registered holds.
    pub fn unregister(self: &Arc<Self>, pid: ProcessId) {
        crate::store::shared::forget_process(pid);
        let (model, engine) = self.port.locus();
        let held = held_page_count(pid, model, engine);
        let held_rs = held_rs_units(pid, model, engine);
        // A seat taken again after the process ended goes back.
        crate::inferlet::process::vacate_seat(pid);
        let (signal, removed, charged) = self.with_inner(|inner| {
            let departed = inner.procs.remove(&pid);
            if inner.bind_restoring == Some(pid) {
                inner.bind_restoring = None;
            }
            let charged = departed.as_ref().is_some_and(|proc| proc.charged);
            if departed.is_some() {
                inner.admitting.retain(|(_, waiting), _| *waiting != pid);
            }
            if self.seat_to_end {
                if let Some(passed) = inner.to_end.live.remove(&pid)
                    && let Some(proc) = departed.as_ref().filter(|proc| {
                        proc.admitted
                            && proc.state == Residency::Resident
                            && !proc.respawn
                            && proc.peak.0 > 0
                    })
                {
                    inner.to_end.learn(proc.peak, proc.first, passed.elapsed());
                }
                inner.to_end.waiting.retain(|(_, waiting)| *waiting != pid);
            }
            if departed.as_ref().is_some_and(|proc| {
                proc.admitted && !proc.holder && proc.state == Residency::Resident
            }) {
                inner.completion_credit = inner.completion_credit.saturating_add(1);
            }
            inner.settle_eviction(pid);
            // Its holds outlive it only while one lives; the last one's drop
            // settles this, and cannot have run before it is put here.
            // Its state is on its way back as its pages are: a pool dry of
            // the slots its sets still hold is not wedged.
            if (held > 0 || !held_rs.is_zero())
                && departed
                    .as_ref()
                    .is_some_and(|proc| proc.departure.strong_count() > 0)
            {
                inner.evicting.insert(
                    pid,
                    Reclaim {
                        pages: held,
                        rs: held_rs,
                    },
                );
            }
            let signal = departed.map(|proc| proc.signal);
            inner.killing.remove(&pid);
            inner.host_swap_blocked.clear();
            inner.prepare_blocked.clear();
            let keys: Vec<EntryKey> = inner
                .queue
                .iter()
                .filter(|(_, waiter)| waiter.pid == pid)
                .map(|(key, _)| *key)
                .collect();
            let removed: Vec<Waiter> = keys
                .iter()
                .filter_map(|key| inner.queue.remove(key))
                .collect();
            (signal, removed, charged)
        });
        if charged {
            self.seat_room.notify_waiters();
        }
        for waiter in &removed {
            if let WaitKind::Allocation { notify, .. } = &waiter.kind {
                notify.notify_waiters();
            }
        }
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        drop(removed);
        if self.seat_to_end {
            self.to_end_notify.notify_waiters();
        }
        self.re_arm_idle_reclaim();
        self.poke();
    }

    /// Whether a process's demand waits for room: a kv allocation, a
    /// restore or a guest bind, unmet. A rise is ordered among them by its
    /// own key and is not counted: it is what the room it waits on goes to
    /// in turn, not what a rung walk gives room back for.
    #[must_use]
    pub fn demand_waits(&self) -> bool {
        self.with_inner(|inner| {
            inner.queue.values().any(|waiter| {
                waiter.is_unmet()
                    && !matches!(
                        waiter.kind,
                        super::wait::WaitKind::Allocation { rise: true, .. }
                    )
            })
        })
    }

    pub fn pages_freed(self: &Arc<Self>) {
        self.port.freed();
        // A set let go or a segment backed: admission reads its seats again.
        self.seat_room.notify_waiters();
        self.re_arm_idle_reclaim();
        if self.waiters.load(Ordering::Acquire) != 0 {
            self.poke();
        }
    }

    pub(super) fn try_reserve(&self, demand: Demand) -> Option<AllocationGrant> {
        let kv = if demand.kv_pages > 0 {
            let pages = self.port.reserve_device(demand.kv_pages)?;
            DevicePageReservation::new(pages, self.port.clone())
        } else {
            DevicePageReservation::empty()
        };
        let rs = self.reserve_rs_slots(demand.rs)?;
        Some(AllocationGrant::new(kv, rs))
    }

    pub(super) fn reserve_rs_slots(&self, need: RsUnits) -> Option<RsSlotReservation> {
        if need.is_zero() {
            return Some(RsSlotReservation::empty());
        }
        let grant = self.port.reserve_rs(need)?;
        Some(RsSlotReservation::new(grant, self.port.clone()))
    }

    pub(super) fn absorb_free(&self, count: u32) -> bool {
        let pages = self.port.reserve_device_up_to(count);
        if pages.is_empty() {
            return false;
        }
        let reservation = DevicePageReservation::new(pages, self.port.clone());
        self.with_inner(|inner| inner.accum.absorb(reservation));
        true
    }
}

/// What a holder ranks by as a victim. Moving it whole takes its state
/// too (on a shared pool its record, rs list and window slots), which every
/// token it keeps reads: so it ranks by the highest key on its set, the
/// key on its head where a policy keeps one (`Retain::upto`). A tail the
/// head's key does not cover goes first by itself ([`holder_head`], the
/// sweep), its pages at the whole set's key.
pub(super) fn holder_key(pid: ProcessId) -> i64 {
    use crate::policy::retention::{Installed, Keys};
    Installed.key_at(Some(pid), 0)
}

/// The tokens a holder's keys keep above the whole of it.
fn holder_head(pid: ProcessId) -> u64 {
    use crate::policy::retention::{Installed, Keys};
    let whole = Installed.key_at(Some(pid), u64::MAX);
    Installed
        .steps(pid)
        .into_iter()
        .filter(|&upto| upto > 0 && Installed.key_at(Some(pid), upto - 1) > whole)
        .max()
        .unwrap_or(0)
}
