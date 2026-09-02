//! Residency planner: FCFS-by-spawn membership admission over the KV pool.

mod exec;
mod grant;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use tokio::sync::Notify;

pub use grant::{AllocationGrant, Demand};
use grant::{DevicePageReservation, RsSlotReservation};

use crate::store::kv::page_table::ReclaimQuote;

/// Opt-in event markers (`PIE_CONTENTION_TRACE_EVENTS=1`), not `tracing` —
/// the embedded server installs no subscriber.
pub(crate) fn trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PIE_CONTENTION_TRACE_EVENTS")
            .map(|raw| !matches!(raw.trim(), "" | "0" | "false" | "off"))
            .unwrap_or(false)
    })
}

macro_rules! ptrace {
    ($($arg:tt)*) => {
        if crate::planner::trace_enabled() {
            println!(
                "[planner t_us={}] {}",
                crate::scheduler::fire_timing_now_us(),
                format_args!($($arg)*)
            );
        }
    };
}

/// Process identity the planner tracks (FCFS clock key, residency key).
pub type ProcessId = uuid::Uuid;

/// Minimal physical port the planner drives — pool stats and reservations.
/// The planner owns all policy; this port owns only physics.
pub trait PoolPort: Send + Sync + 'static {
    /// `(free, total)` device pages of the planned pool.
    fn device_stats(&self) -> (u32, u32);
    /// `(free, total)` host swap slots.
    fn host_stats(&self) -> (u32, u32);
    /// `(free, total)` RS folded slots.
    fn rs_stats(&self) -> (u32, u32);
    /// Rung 0: reclaim whatever costs no work — cache-root leases nothing
    /// live reaches. Returns pages recycled NOW.
    fn reclaim_idle(&self) -> u32;
    /// Whether the engine can physically move KV bytes to/from host swap.
    /// Arms eviction; without it the planner is pool-only, not a mode change.
    fn suspend_capable(&self) -> bool;
    /// The `(model, engine)` pair this pool belongs to.
    fn locus(&self) -> (usize, usize);
    fn reserve_device(
        &self,
        count: u32,
    ) -> Option<Vec<crate::store::kv::page_table::PhysicalKvPageId>>;
    /// Pop up to `count` free device pages under one store-lock hold — the
    /// drain's absorb step.
    fn reserve_device_up_to(
        &self,
        count: u32,
    ) -> Vec<crate::store::kv::page_table::PhysicalKvPageId>;
    fn release_device(&self, pages: Vec<crate::store::kv::page_table::PhysicalKvPageId>);
    fn reserve_rs(&self, count: u32) -> Option<Vec<crate::store::rs::RsSlotId>>;
    fn release_rs(&self, slots: Vec<crate::store::rs::RsSlotId>);
}

/// Production [`PoolPort`] over the typed per-(model, engine) stores.
pub struct RegistryPool {
    model: usize,
    engine: usize,
    suspend_capable: bool,
}

impl RegistryPool {
    pub fn new(model: usize, engine: usize, suspend_capable: bool) -> Self {
        Self {
            model,
            engine,
            suspend_capable,
        }
    }

    fn with_kv_tagged<R>(
        &self,
        tag: &'static str,
        operation: impl FnOnce(&mut crate::store::kv::KvStore) -> R,
    ) -> R {
        let stores = crate::store::registry::get(self.model, self.engine);
        crate::store::registry::with_kv_lock(&stores.kv, tag, operation)
    }

    fn with_rs<R>(&self, operation: impl FnOnce(&mut crate::store::rs::RsStore) -> R) -> R {
        let stores = crate::store::registry::get(self.model, self.engine);
        let mut store = stores.rs.lock().unwrap();
        operation(&mut store)
    }
}

impl PoolPort for RegistryPool {
    fn device_stats(&self) -> (u32, u32) {
        self.with_kv_tagged("planner-device-stats", |kv| {
            (kv.available_pages() as u32, kv.capacity_pages())
        })
    }

    fn host_stats(&self) -> (u32, u32) {
        self.with_kv_tagged("planner-host-stats", |kv| {
            (kv.host_swap_available() as u32, kv.host_swap_capacity())
        })
    }

    fn rs_stats(&self) -> (u32, u32) {
        self.with_rs(|rs| (rs.available_slots() as u32, rs.capacity_slots()))
    }

    fn reclaim_idle(&self) -> u32 {
        self.with_kv_tagged("planner-reclaim-idle", |kv| {
            let epoch = kv.current_epoch();
            let freed = kv.drop_unused_cache_leases(epoch);
            if freed > 0 {
                kv.retire_idle();
            }
            freed as u32
        })
    }

    fn suspend_capable(&self) -> bool {
        self.suspend_capable
    }

    fn locus(&self) -> (usize, usize) {
        (self.model, self.engine)
    }

    fn reserve_device(
        &self,
        count: u32,
    ) -> Option<Vec<crate::store::kv::page_table::PhysicalKvPageId>> {
        self.with_kv_tagged("planner-reserve", |kv| {
            kv.reserve_device_pages(count as usize)
        })
    }

    fn reserve_device_up_to(
        &self,
        count: u32,
    ) -> Vec<crate::store::kv::page_table::PhysicalKvPageId> {
        self.with_kv_tagged("planner-reserve-up-to", |kv| {
            let take = (kv.available_pages() as u32).min(count);
            if take == 0 {
                return Vec::new();
            }
            kv.reserve_device_pages(take as usize).unwrap_or_default()
        })
    }

    fn release_device(&self, pages: Vec<crate::store::kv::page_table::PhysicalKvPageId>) {
        self.with_kv_tagged("planner-release", |kv| kv.release_device_reservation(pages));
    }

    fn reserve_rs(&self, count: u32) -> Option<Vec<crate::store::rs::RsSlotId>> {
        self.with_rs(|rs| rs.reserve_slots(count as usize))
    }

    fn release_rs(&self, slots: Vec<crate::store::rs::RsSlotId>) {
        self.with_rs(|rs| rs.release_slot_reservation(slots));
    }
}

/// What [`ResidencyPlanner::acquire`] resolved to.
pub enum Acquired {
    Granted(AllocationGrant),
    /// The asking process was chosen for eviction (or is already out of the
    /// set); the fire path settles its own tail and re-asks.
    Yield,
}

/// Why eviction could not fund the head, carried into [`PlannerError::Starved`]
/// so the message names the real wedge instead of reading as a generic exhaustion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarveCause {
    /// No KV swap transport, or the host swap pool has no free slot.
    NoSwapRoom,
    /// Swap room exists, but no candidate was eligible. Eligibility is FCFS
    /// anti-thrash: only processes younger than the head may be evicted.
    NoEligibleVictim,
    /// Every RS folded slot is held and no admitted process is left
    /// running to return one. Eviction frees KV pages, never folded slots.
    NoRsSlots,
}

impl StarveCause {
    fn describe(self) -> &'static str {
        match self {
            StarveCause::NoSwapRoom => "no host swap room to evict into (or no swap transport)",
            StarveCause::NoEligibleVictim => {
                "no evictable victim — every page is held by a process OLDER \
                 than the head, which FCFS anti-thrash forbids evicting"
            }
            StarveCause::NoRsSlots => {
                "every RS folded slot is held and no admitted process is still \
                 running to return one — asking for more concurrent state slots \
                 than the pool has cannot be waited out"
            }
        }
    }
}

/// A hard failure out of [`ResidencyPlanner::acquire`]. Every variant is a
/// computed predicate — none is a timer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannerError {
    /// The request can never fit: `need` exceeds the pool's total capacity.
    Impossible { need: u32, total: u32 },
    /// The hog endgame: the head's own holdings plus its ask exceed the
    /// pool, so no amount of evicting others can ever cover it.
    Hog { need: u32, held: u32, total: u32 },
    /// The starvation endgame, computed — not timed: eviction cannot fund the
    /// unmet head. The youngest parked ask is failed loud (see [`StarveCause`]).
    Starved {
        need: u32,
        free: u32,
        total: u32,
        cause: StarveCause,
    },
    /// The process was unregistered while waiting.
    Cancelled,
}

impl std::fmt::Display for PlannerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlannerError::Impossible { need, total } => write!(
                f,
                "allocation of {need} units can never fit (pool total {total})"
            ),
            PlannerError::Hog { need, held, total } => write!(
                f,
                "KV pool exhausted by this process alone: it holds {held} pages and asks \
                 {need} more of a {total}-page pool — no eviction of others can cover it"
            ),
            PlannerError::Starved {
                need,
                free,
                total,
                cause,
            } => write!(
                f,
                "KV pool starved: {need} pages asked, {free} free of {total}, {}, and no \
                 fire in flight anywhere to complete and free pages",
                cause.describe()
            ),
            PlannerError::Cancelled => f.write_str("planner request cancelled"),
        }
    }
}

impl std::error::Error for PlannerError {}

/// Opt-in planner-mutex census (`PIE_PLANNER_LOCK_TRACE=1`).
pub(crate) struct LockCensus {
    pub n: AtomicU64,
    pub wait_ns: AtomicU64,
    pub hold_ns: AtomicU64,
    pub wait_max_ns: AtomicU64,
    pub hold_max_ns: AtomicU64,
}
pub(crate) static LOCK_CENSUS: LockCensus = LockCensus {
    n: AtomicU64::new(0),
    wait_ns: AtomicU64::new(0),
    hold_ns: AtomicU64::new(0),
    wait_max_ns: AtomicU64::new(0),
    hold_max_ns: AtomicU64::new(0),
};
pub(crate) fn lock_trace_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("PIE_PLANNER_LOCK_TRACE").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}
/// Guard that charges its hold span on drop.
struct TimedGuard<'a> {
    guard: parking_lot::MutexGuard<'a, Inner>,
    held_from: Option<std::time::Instant>,
}
impl<'a> std::ops::Deref for TimedGuard<'a> {
    type Target = Inner;
    fn deref(&self) -> &Inner {
        &self.guard
    }
}
impl<'a> std::ops::DerefMut for TimedGuard<'a> {
    fn deref_mut(&mut self) -> &mut Inner {
        &mut self.guard
    }
}
impl<'a> Drop for TimedGuard<'a> {
    fn drop(&mut self) {
        if let Some(t) = self.held_from {
            let ns = t.elapsed().as_nanos() as u64;
            LOCK_CENSUS.hold_ns.fetch_add(ns, Ordering::Relaxed);
            LOCK_CENSUS.hold_max_ns.fetch_max(ns, Ordering::Relaxed);
        }
    }
}

/// Engagement counters (lock-free reads).
#[derive(Debug, Default)]
pub struct PlannerStats {
    /// `acquire` calls that parked (fast-path reserve refused or short).
    pub parks: AtomicU64,
    /// Parked asks served out of the accumulation.
    pub serves: AtomicU64,
    /// Eviction attempts spawned.
    pub evictions_started: AtomicU64,
    /// Shortages that did not trigger an eviction because the fleet was
    /// still making progress — the load-control rung.
    pub eviction_deferrals: AtomicU64,
    /// Eviction rounds the supply runway motivated. Zero unless
    /// `PIE_SUPPLY_RUNWAY` is set.
    pub runway_rounds: AtomicU64,
    /// Pages those rounds ordered on the runway's account (the shortfall
    /// component, not the queued-demand component).
    pub runway_pages: AtomicU64,
    /// Evictions that committed (working sets moved to host swap).
    pub evictions: AtomicU64,
    /// Evictions abandoned before commit (nothing reclaimable, non-detachable
    /// tail, transport failure, host swap full).
    pub eviction_rollbacks: AtomicU64,
    /// Evicted processes restored to residency.
    pub restores: AtomicU64,
    /// Restore attempts that failed and re-queued.
    pub restore_failures: AtomicU64,
    /// Fires parked at the residency gate or the working-set fence.
    pub gate_parks: AtomicU64,
    pub cancelled_waits: AtomicU64,
    /// Heads failed loud by the hog predicate.
    pub hog_failures: AtomicU64,
    /// Youngest asks failed loud by the starvation predicate.
    pub starvations: AtomicU64,
    /// Subset of `starvations` where the victim had declared itself
    /// restartable, so its work was re-queued instead of lost.
    pub starvation_restarts: AtomicU64,
    /// Times the starvation rung found the pool refilled underneath it and
    /// handed the head back to the drain instead of destroying a request.
    pub salvages: AtomicU64,
    /// Asks served out of FCFS order by the last rung before destruction,
    /// because the head's own hoard covered them and the head was uncoverable.
    pub hoard_bypasses: AtomicU64,
    /// Evictions that had to relax the post-restore hysteresis because no
    /// normally-eligible victim could fund the head.
    pub e6_relaxations: AtomicU64,
    /// Restore heads that found the pool dry and declined to evict for
    /// themselves; the readmission waits for a completion instead.
    pub restore_absorb_short: AtomicU64,
    pub host_swap_exhaustions: AtomicU64,
    /// How many times host room returned and re-armed the blocked victims.
    pub host_swap_unblocks: AtomicU64,
    pub d2h_pages: AtomicU64,
    pub h2d_pages: AtomicU64,
    pub d2h_copy_us: AtomicU64,
    pub h2d_copy_us: AtomicU64,
}

/// One diagnostics row (queue entry or evicted process).
#[derive(Debug, Clone)]
pub struct PlannerQueueEntry {
    pub process_id: String,
    pub spawn_seq: u64,
    pub kind: &'static str,
    pub pages: u32,
    /// RS folded slots this ask needs. A head parked with pages available
    /// can still be blocked on this instead.
    pub rs_slots: u32,
}

/// How many unparked admitted processes [`PlannerDiagnostics::runners`]
/// reports. A wedge needs the first few; a healthy fleet needs none.
const RUNNER_DUMP_CAP: usize = 24;

/// The host-pool slice the eviction rung may not spend on a fleet still
/// running — large enough to avoid the kill arm, small enough to never bind when roomy.
const HOST_RESERVE_DIVISOR: u32 = 8;

#[derive(Debug, Clone)]
pub struct PlannerDiagnostics {
    pub device_pages_free: u32,
    pub device_pages_total: u32,
    pub host_slots_free: u32,
    pub host_slots_total: u32,
    /// `(free, total)` RS folded slots — the other resource an allocation ask
    /// can park on, and the one a KV-only trace cannot see.
    pub rs_slots_free: u32,
    pub rs_slots_total: u32,
    pub accumulation: u32,
    pub queue: Vec<PlannerQueueEntry>,
    /// Registered processes by state: Resident, Evicting, Evicted, Restoring.
    pub proc_states: [u32; 4],
    /// Registered processes that have claimed an execution slot. The
    /// difference against the total is queued for a permit, page-less.
    pub admitted_procs: u32,
    /// The admitted processes NOT parked in the queue — the only cohort that
    /// can still make progress. Capped: a wedge is diagnosed from a handful.
    pub runners: Vec<(u64, u32, bool)>,
    pub parks_total: u64,
    pub serves_total: u64,
    pub evictions_total: u64,
    /// Shortages that did not trigger an eviction because the fleet could
    /// still make progress on its own (load control).
    pub eviction_deferrals_total: u64,
    /// The entry the drain is blocked on, and how many entries still
    /// compete for pages at all (a served-but-uncollected grant does not).
    pub unmet_head_pages: u32,
    pub unmet_head_kind: &'static str,
    pub unmet_queued: u32,
    /// Entries behind the blocked head that the currently-free stock could
    /// cover on its own — the head-of-line cost.
    pub bypassable_entries: u32,
    pub bypassable_pages: u32,
    pub eviction_rollbacks_total: u64,
    pub restores_total: u64,
    pub restore_failures_total: u64,
    pub gate_parks_total: u64,
    pub cancelled_waits_total: u64,
    pub hog_failures_total: u64,
    pub starvations_total: u64,
    pub starvation_restarts_total: u64,
    pub salvages_total: u64,
    pub hoard_bypasses_total: u64,
    pub e6_relaxations_total: u64,
    pub restore_absorb_short_total: u64,
    pub host_swap_exhaustions_total: u64,
    pub host_swap_unblocks_total: u64,
    pub d2h_pages_total: u64,
    pub h2d_pages_total: u64,
    pub d2h_copy_us_total: u64,
    pub h2d_copy_us_total: u64,
    /// Supply-runway probes — zero unless `PIE_SUPPLY_RUNWAY`.
    pub runway_rounds_total: u64,
    pub runway_pages_total: u64,
}

/// One process the FCFS anti-thrash rule permits evicting for the current
/// head. `e6_fresh` is POLICY (see [`VictimSet`]), never eligibility.
#[derive(Clone, Copy)]
struct Victim {
    pid: ProcessId,
    seq: u64,
    /// False during the post-restore E6 hysteresis window — a reason to
    /// *prefer* someone else, never a reason to consider this un-evictable.
    e6_fresh: bool,
}

/// The legal victims for one head, from one consistent snapshot: younger
/// than the head, resident. E6 hysteresis only narrows [`Self::preferred`].
struct VictimSet {
    head: EntryKey,
    head_pid: ProcessId,
    deficit: u32,
    members: Vec<Victim>,
    /// How many of `deficit`'s pages are the supply runway's shortfall rather
    /// than queued demand. Nonzero only with `PIE_SUPPLY_RUNWAY` set.
    runway_grab: u32,
}

impl VictimSet {
    /// Members E6 hysteresis permits evicting right now. May be empty while
    /// the set is not; never read this as "no victim exists".
    fn preferred(&self) -> Vec<(ProcessId, u64)> {
        self.members
            .iter()
            .filter(|v| v.e6_fresh)
            .map(|v| (v.pid, v.seq))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Residency {
    Resident,
    /// Planner-driven D2H in progress (`exec::evict`). The residency gate is
    /// already closed and the working-set fences are up.
    Evicting,
    /// Working sets on host swap; a restore entry sits in the queue at this
    /// process's spawn position.
    Evicted,
    /// Planner-driven H2D in progress (`exec::restore`).
    Restoring,
}

struct Proc {
    /// Spawn-order clock — the single, authoritative FCFS key.
    seq: u64,
    state: Residency,
    /// Whether this process has already spent its one restore retry. Cleared
    /// on every successful restore — the allowance is per episode, not lifetime.
    restore_retried: bool,
    /// E6 — progress before re-eviction. `false` from restore commit until the
    /// next `acquire`; prevents re-evicting the same pages before a fire runs.
    progressed: bool,
    /// Whether this process has claimed an execution slot. An unadmitted
    /// process holds zero pooled pages and cannot unwedge the pool.
    admitted: bool,
    /// Wakes residency-gate waiters and out-of-set acquire loops on any
    /// state change.
    signal: Arc<Notify>,
    /// Lock-free mirror of `state == Resident`, refreshed by `with_inner`. Lets
    /// the residency gate do one relaxed load instead of taking the planner mutex.
    resident: Arc<AtomicBool>,
}

impl Proc {
    fn new(seq: u64) -> Self {
        Self {
            seq,
            state: Residency::Resident,
            restore_retried: false,
            progressed: true,
            admitted: false,
            signal: Arc::new(Notify::new()),
            resident: Arc::new(AtomicBool::new(true)),
        }
    }
}

/// What kind of service a queue entry waits for.
enum WaitKind {
    /// A parked fire; its `acquire` future waits on `notify`.
    Allocation {
        demand: Demand,
        notify: Arc<Notify>,
        /// The served grant (or a computed failure), parked until collected.
        outcome: Option<Result<AllocationGrant, PlannerError>>,
        /// Set when the owner was chosen for eviction while parked: the collector
        /// yields back to the fire path instead of waiting for pages.
        yielded: bool,
    },
    /// An evicted process awaiting readmission. `demand` is refreshed from
    /// the store's swapped count when the entry reaches the head.
    Restore { demand: u32 },
}

struct Waiter {
    pid: ProcessId,
    kind: WaitKind,
}

impl Waiter {
    /// Whether this entry still waits for service (a served, uncollected
    /// grant no longer competes).
    fn is_unmet(&self) -> bool {
        match &self.kind {
            WaitKind::Allocation {
                outcome, yielded, ..
            } => outcome.is_none() && !yielded,
            WaitKind::Restore { .. } => true,
        }
    }

    /// The device-page component of this entry's ask.
    fn kv_need(&self) -> u32 {
        match &self.kind {
            WaitKind::Allocation { demand, .. } => demand.kv_pages,
            WaitKind::Restore { demand } => *demand,
        }
    }
}

/// Key: spawn clock first, insertion order second.
type EntryKey = (u64, u64);

/// One eviction in flight, as the planner accounts for it.
struct EvictionMark {
    /// Pages this victim is expected to free when it lands.
    pages: u32,
}

#[derive(Default)]
struct Inner {
    next_seq: u64,
    next_id: u64,
    procs: HashMap<ProcessId, Proc>,
    queue: BTreeMap<EntryKey, Waiter>,
    /// Head-first accumulation: pages pulled OUT of the free list toward the
    /// head's demand. Planner-level, so a head change strands nothing.
    accum: DevicePageReservation,
    /// Evictions in flight → what each is expected to free. Subtracted from
    /// the deficit so concurrent plans never over-evict.
    evicting: HashMap<ProcessId, EvictionMark>,
    /// Rotation damping (Rule A as a stock): resident completions banked,
    /// debited per served restore — bounds rotation by the completion rate.
    completion_credit: u32,
    /// Victims whose last eviction rolled back on `HostSwapFull`. Cleared
    /// wholesale once host slots return — see `clear_host_swap_blocks`.
    host_swap_blocked: HashSet<ProcessId>,
    /// Victims whose last eviction rolled back because `prepare_suspend`
    /// deferred. Same deterministic-re-pick spin as `host_swap_blocked`.
    prepare_blocked: HashSet<ProcessId>,
    /// Destructions ordered but not yet paid out — see [`Inner::kill_in_flight`].
    killing: HashSet<ProcessId>,
    /// Runway hysteresis: at most one runway-motivated round in flight. Cleared
    /// by [`Inner::settle_eviction`]; always `false` while the runway is unset.
    runway_round_in_flight: bool,
}

impl Inner {
    /// A starvation kill has been ordered whose pages have not come back yet.
    /// Self-clearing on both exits; `unregister` prunes the set so it stays bounded.
    fn kill_in_flight(&self) -> bool {
        self.killing.iter().any(|pid| {
            self.procs.contains_key(pid)
                && !self
                    .queue
                    .values()
                    .any(|waiter| waiter.pid == *pid && waiter.is_unmet())
        })
    }

    /// Retire one eviction from the in-flight set and release the runway
    /// latch once nothing at all is in flight.
    fn settle_eviction(&mut self, pid: ProcessId) {
        self.evicting.remove(&pid);
        if self.evicting.is_empty() {
            self.runway_round_in_flight = false;
        }
    }

    /// The seq floor for a runway round's victims: the oldest admitted
    /// resident, raised to the FCFS head's seq when one is queued.
    fn runway_floor(&self) -> Option<u64> {
        let oldest = self
            .procs
            .values()
            .filter(|proc| proc.admitted && proc.state == Residency::Resident)
            .map(|proc| proc.seq)
            .min()?;
        let head = self.unmet_head().map_or(0, |(key, _)| key.0);
        Some(oldest.max(head))
    }

    /// Count non-resident processes and refresh each process's lock-free
    /// residency mirror in one pass, since both are derived facts.
    fn nonresident_count(&self) -> usize {
        let mut n = 0;
        for proc in self.procs.values() {
            let resident = proc.state == Residency::Resident;
            if !resident {
                n += 1;
            }
            proc.resident.store(resident, Ordering::Release);
        }
        n
    }

    /// The FCFS head: the oldest unmet entry. A restore yields to any unmet
    /// allocation unless [`Inner::fleet_stalled`] or [`Inner::eviction_unfundable`].
    fn unmet_head(&self) -> Option<(EntryKey, &Waiter)> {
        let mut oldest_restore: Option<EntryKey> = None;
        let mut allocation: Option<EntryKey> = None;
        for (&key, waiter) in &self.queue {
            if !waiter.is_unmet() {
                continue;
            }
            match &waiter.kind {
                // Keys ascend, so the first unmet allocation is the oldest,
                // and every restore that could outrank it has been seen.
                WaitKind::Allocation { .. } => {
                    allocation = Some(key);
                    break;
                }
                WaitKind::Restore { .. } => {
                    oldest_restore.get_or_insert(key);
                }
            }
        }
        let head = match (allocation, oldest_restore) {
            (Some(allocation), Some(restore)) => {
                if restore < allocation && (self.fleet_stalled() || self.eviction_unfundable()) {
                    restore
                } else {
                    allocation
                }
            }
            (Some(allocation), None) => allocation,
            (None, restore) => restore?,
        };
        self.queue
            .get_key_value(&head)
            .map(|(&key, waiter)| (key, waiter))
    }

    /// Eviction has run out of the resource that funds it: a victim parked in
    /// `host_swap_blocked` is proof. Live, not a latch — clears once room returns.
    fn eviction_unfundable(&self) -> bool {
        !self.host_swap_blocked.is_empty()
    }

    /// The pages the burst behind the head still needs beyond the accumulation.
    /// The run is the maximal consecutive stretch of unmet, RS-free waiters.
    fn burst_shortfall(&self) -> u32 {
        let mut supply = self.accum.len() as u64;
        let mut extra = 0u64;
        for waiter in self.queue.values() {
            if !waiter.is_unmet() {
                continue;
            }
            let WaitKind::Allocation { demand, .. } = &waiter.kind else {
                continue;
            };
            if demand.rs_slots > 0 {
                break;
            }
            let need = u64::from(demand.kv_pages);
            let covered = need.min(supply);
            supply -= covered;
            extra += need - covered;
        }
        extra.try_into().unwrap_or(u32::MAX)
    }

    /// The burst's single lock hold: fund consecutive waiters head-first,
    /// parking each outcome. Stops at the first waiter it cannot cover in full.
    fn serve_burst(&mut self) -> Vec<(EntryKey, u32, Arc<Notify>)> {
        let mut wake = Vec::new();
        let Some((head, waiter)) = self.unmet_head() else {
            return wake;
        };
        // The head is re-derived here after the pull released the lock; a restore
        // that took it meanwhile outranks everything, so let the drain re-decide.
        if !matches!(waiter.kind, WaitKind::Allocation { .. }) {
            return wake;
        }
        // Disjoint field borrows: the accumulation is donated from while the
        // queue entry that receives it is held mutably.
        let Inner { queue, accum, .. } = self;
        for (&key, waiter) in queue.range_mut(head..) {
            if !waiter.is_unmet() {
                continue;
            }
            let WaitKind::Allocation {
                demand,
                notify,
                outcome,
                ..
            } = &mut waiter.kind
            else {
                continue; // a restore never ends the allocation run
            };
            let demand = *demand;
            if demand.rs_slots > 0 {
                break;
            }
            if (accum.len() as u32) < demand.kv_pages {
                break;
            }
            let kv = accum.donate(demand.kv_pages as usize);
            debug_assert!(outcome.is_none(), "an unmet waiter carries no outcome");
            *outcome = Some(Ok(AllocationGrant::new(
                demand,
                kv,
                RsSlotReservation::empty(),
            )));
            wake.push((key, demand.kv_pages, notify.clone()));
        }
        wake
    }

    /// No completion can ever arrive on its own: no eviction in flight, no
    /// grant awaiting collection, every admitted resident parked.
    fn fleet_stalled(&self) -> bool {
        if !self.evicting.is_empty() {
            return false;
        }
        let mut parked = HashSet::new();
        for waiter in self.queue.values() {
            match &waiter.kind {
                WaitKind::Allocation {
                    outcome: Some(_), ..
                } => return false,
                _ => {
                    parked.insert(waiter.pid);
                }
            }
        }
        self.procs.iter().all(|(pid, proc)| match proc.state {
            Residency::Resident => !proc.admitted || parked.contains(pid),
            Residency::Evicting | Residency::Restoring => false,
            Residency::Evicted => true,
        })
    }
}

/// One step of the drain: computed under the lock, executed against the
/// port/store outside it.
enum Step {
    /// The head still misses `count` device pages; pull them from the pool.
    /// `fund_by_eviction` is false for a restore the fleet is not stalled behind.
    Absorb {
        count: u32,
        fund_by_eviction: bool,
    },
    /// The head's KV side is covered; finish an allocation with `rs` slots.
    /// RS-free allocations take [`Step::ServeAllocationBurst`] instead.
    ServeAllocation {
        key: EntryKey,
        demand: Demand,
    },
    /// The head is a covered, RS-free allocation: serve the maximal consecutive
    /// run of fundable RS-free waiters in one pass. FCFS is untouched.
    ServeAllocationBurst {
        extra: u32,
    },
    /// The head is a restore whose recorded demand is covered; re-validate
    /// its swapped count and board it.
    ServeRestore {
        key: EntryKey,
        pid: ProcessId,
    },
    /// Nobody waits: return the stranded accumulation to the pool.
    Release(DevicePageReservation),
    Done,
}

pub struct ResidencyPlanner {
    // parking_lot (like the KV store lock): every contended acquire touches
    // this; adaptive spinning keeps the wake-herd from becoming a futex storm.
    inner: parking_lot::Mutex<Inner>,
    /// Lock-free mirror of `queue.len()` — the acquire fast path's only planner
    /// touch. Readers may observe one transition late; the slow path re-checks.
    waiters: AtomicUsize,
    /// Lock-free mirror of the not-Resident process count — the residency
    /// gate's only planner touch.
    nonresident: AtomicUsize,
    /// The single-owner drain task; when armed every call site pokes this
    /// instead of running `plan()` inline (used pre-boot).
    drain: OnceLock<Arc<Notify>>,
    /// Rung 0 exhaustion latch: set when `reclaim_idle` returns 0, cleared by
    /// real free events. Prevents a fruitless scan while the pool sits at free=0.
    idle_reclaim_exhausted: std::sync::atomic::AtomicBool,
    port: Arc<dyn PoolPort>,
    stats: PlannerStats,
}

/// Removes a parked allocation entry when its `acquire` future is dropped:
/// the entry vanishes and any parked reservation returns to the pool.
struct WaitRegistration<'a> {
    planner: &'a Arc<ResidencyPlanner>,
    key: EntryKey,
    active: bool,
}

impl WaitRegistration<'_> {
    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for WaitRegistration<'_> {
    fn drop(&mut self) {
        if self.active {
            self.planner.cancel_waiter(self.key);
        }
    }
}

/// What [`ResidencyPlanner::acquire_or_enqueue`] resolved to. `Ticket` is the
/// parked case handed back instead of awaited.
#[allow(
    dead_code,
    reason = "same unwired `PIE_DEFER_ALLOC` path as \
              `ResidencyPlanner::acquire_or_enqueue`, which constructs these \
              variants; nothing destructures them because the consumer's handle \
              was not carried by this merge (see `pipeline.rs`)"
)]
pub(crate) enum Enqueued {
    Granted(AllocationGrant),
    Ticket(AllocationTicket),
    /// The asking process is out of the resident set: the fire path must settle
    /// its own tail — same contract as [`Acquired::Yield`].
    NotResident,
}

/// A parked allocation ask, owned: inserted at its FCFS position. Dropping
/// an uncollected ticket deregisters the ask like [`WaitRegistration`].
pub(crate) struct AllocationTicket {
    planner: Arc<ResidencyPlanner>,
    key: EntryKey,
    notify: Arc<Notify>,
    collected: bool,
}

impl AllocationTicket {
    /// The park-collect half of [`ResidencyPlanner::acquire`]: arm the notify,
    /// sleep until served. Returns [`Acquired::Yield`] if chosen for eviction.
    #[allow(
        dead_code,
        reason = "the collect half of the unwired `PIE_DEFER_ALLOC` path; reached \
                  only once `acquire_or_enqueue` has a caller again (see \
                  `pipeline.rs`). This allow also covers the `notify` field, which \
                  only this loop reads"
    )]
    pub(crate) async fn collect(mut self) -> Result<Acquired, PlannerError> {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            match self.planner.collect_outcome(self.key) {
                Collect::Ready(outcome) => {
                    self.collected = true;
                    ptrace!("collect key={:?} ok={}", self.key, outcome.is_ok());
                    if matches!(outcome, Err(PlannerError::Cancelled)) {
                        self.planner
                            .stats
                            .cancelled_waits
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    // The collection unblocked the next head.
                    self.planner.poke();
                    return outcome.map(Acquired::Granted);
                }
                Collect::Yield => {
                    self.collected = true;
                    self.planner.poke();
                    return Ok(Acquired::Yield);
                }
                Collect::Wait => {}
            }
            notified.await;
        }
    }
}

impl Drop for AllocationTicket {
    fn drop(&mut self) {
        if !self.collected {
            self.planner.cancel_waiter(self.key);
        }
    }
}

impl ResidencyPlanner {
    pub fn new(port: Arc<dyn PoolPort>) -> Self {
        Self {
            inner: parking_lot::Mutex::new(Inner::default()),
            waiters: AtomicUsize::new(0),
            nonresident: AtomicUsize::new(0),
            drain: OnceLock::new(),
            idle_reclaim_exhausted: std::sync::atomic::AtomicBool::new(false),
            port,
            stats: PlannerStats::default(),
        }
    }

    /// Arm the single-owner drain task, once at bootstrap. Idempotent; a planner
    /// constructed outside a runtime keeps inline planning.
    pub fn arm_drain_task(self: &Arc<Self>) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let notify = Arc::new(Notify::new());
        if self.drain.set(notify.clone()).is_err() {
            return; // already armed
        }
        let planner = self.clone();
        runtime.spawn(async move {
            loop {
                notify.notified().await;
                planner.plan();
            }
        });
    }

    /// Request a planning pass. With the drain armed this is one atomic notify;
    /// coalescing is free since `Notify` holds one permit.
    fn poke(self: &Arc<Self>) {
        match self.drain.get() {
            Some(notify) => notify.notify_one(),
            None => self.plan(),
        }
    }

    /// Poke the drain when the free list has fallen below the configured
    /// supply runway; the uncontended paths never wake it otherwise.
    fn poke_if_runway_short(self: &Arc<Self>) {
        let runway = supply_runway_pages();
        if runway == 0 {
            return;
        }
        let (free, _) = self.port.device_stats();
        if free < runway {
            self.poke();
        }
    }

    /// The single semantic site for "device pages actually became free": re-arms
    /// rung 0. Every caller pairs it with a poke; only the condition differs.
    fn re_arm_idle_reclaim(&self) {
        self.idle_reclaim_exhausted.store(false, Ordering::Release);
    }

    /// The `(model, engine)` pair this planner manages.
    pub fn locus(&self) -> (usize, usize) {
        self.port.locus()
    }

    /// The single door to the planner mutex, so the census sees every
    /// taker.
    fn lock_inner(&self) -> TimedGuard<'_> {
        if !lock_trace_enabled() {
            return TimedGuard {
                guard: self.inner.lock(),
                held_from: None,
            };
        }
        let t0 = std::time::Instant::now();
        let guard = self.inner.lock();
        let waited = t0.elapsed().as_nanos() as u64;
        LOCK_CENSUS.n.fetch_add(1, Ordering::Relaxed);
        LOCK_CENSUS.wait_ns.fetch_add(waited, Ordering::Relaxed);
        LOCK_CENSUS.wait_max_ns.fetch_max(waited, Ordering::Relaxed);
        TimedGuard {
            guard,
            held_from: Some(std::time::Instant::now()),
        }
    }

    fn with_inner<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        let mut inner = self.lock_inner();
        let result = f(&mut inner);
        self.waiters.store(inner.queue.len(), Ordering::Release);
        self.nonresident
            .store(inner.nonresident_count(), Ordering::Release);
        result
    }

    /// Register a process at spawn — its registration order is the FCFS
    /// clock, the single authoritative service order.
    pub fn register(&self, pid: ProcessId) {
        self.with_inner(|inner| {
            let seq = inner.next_seq;
            inner.next_seq += 1;
            inner.procs.insert(pid, Proc::new(seq));
        });
    }

    /// Register `pid` at an existing FCFS position — the restart path. The
    /// clock is not rewound, so a fresh registration still sorts after it.
    pub fn register_with_seq(&self, pid: ProcessId, seq: u64) {
        self.with_inner(|inner| {
            inner.procs.insert(pid, Proc::new(seq));
        });
    }

    /// This process's position in the FCFS clock, if it is still registered.
    pub fn spawn_seq(&self, pid: ProcessId) -> Option<u64> {
        self.lock_inner().procs.get(&pid).map(|proc| proc.seq)
    }

    /// `pid` has claimed an execution slot and may now hold pooled pages. Until
    /// this lands it is registered but provably page-less. Idempotent.
    pub fn note_admitted(&self, pid: ProcessId) {
        let mut inner = self.lock_inner();
        if let Some(proc) = inner.procs.get_mut(&pid) {
            proc.admitted = true;
        }
    }

    /// Unregister at process exit/terminate: queue entries are removed, gate
    /// waiters woken for teardown, and freed capacity drains to the queue.
    pub fn unregister(self: &Arc<Self>, pid: ProcessId) {
        let (signal, removed) = self.with_inner(|inner| {
            let departed = inner.procs.remove(&pid);
            // Rotation damping: a resident departure funds one readmission. An evicted
            // process completing held no pooled pages, so it funds nothing.
            if departed
                .as_ref()
                .is_some_and(|proc| proc.admitted && proc.state == Residency::Resident)
            {
                inner.completion_credit = inner.completion_credit.saturating_add(1);
            }
            let signal = departed.map(|proc| proc.signal);
            inner.settle_eviction(pid);
            inner.killing.remove(&pid);
            // Teardown returns this process's host slots (and drops it from
            // the candidate pool either way), so re-arm the parked victims.
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
            (signal, removed)
        });
        for waiter in &removed {
            if let WaitKind::Allocation { notify, .. } = &waiter.kind {
                notify.notify_waiters();
            }
        }
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        // Parked outcomes' reservations return to the pool here, outside the
        // lock.
        drop(removed);
        self.re_arm_idle_reclaim();
        self.poke();
    }

    /// Pages or slots freed somewhere (a fire finalized, a process exited, a
    /// working set released). Drains if anyone is waiting.
    pub fn pages_freed(self: &Arc<Self>) {
        self.re_arm_idle_reclaim();
        if self.waiters.load(Ordering::Acquire) != 0 {
            self.poke();
        } else {
            // With nobody waiting the poke is normally skipped, but a free event is
            // also the runway's chance to top up before the next ask parks.
            self.poke_if_runway_short();
        }
    }

    // === The acquisition path ===

    /// All-or-nothing direct reservation off the free lists — the hot path.
    fn try_reserve(&self, demand: Demand) -> Option<AllocationGrant> {
        let kv = if demand.kv_pages > 0 {
            let pages = self.port.reserve_device(demand.kv_pages)?;
            DevicePageReservation::new(pages, self.port.clone())
        } else {
            DevicePageReservation::empty()
        };
        let rs = if demand.rs_slots > 0 {
            {
                let slots = self.port.reserve_rs(demand.rs_slots)?;
                RsSlotReservation::new(slots, self.port.clone())
            }
        } else {
            RsSlotReservation::empty()
        };
        Some(AllocationGrant::new(demand, kv, rs))
    }

    /// `(acquisitions, wait_us, hold_us, wait_max_us, hold_max_us)` since the
    /// last read — zero unless `PIE_PLANNER_LOCK_TRACE=1`.
    pub fn lock_census(&self) -> (u64, u64, u64, u64, u64) {
        let c = &LOCK_CENSUS;
        (
            c.n.swap(0, Ordering::Relaxed),
            c.wait_ns.swap(0, Ordering::Relaxed) / 1000,
            c.hold_ns.swap(0, Ordering::Relaxed) / 1000,
            c.wait_max_ns.swap(0, Ordering::Relaxed) / 1000,
            c.hold_max_ns.swap(0, Ordering::Relaxed) / 1000,
        )
    }

    pub fn park_census(&self) -> (u64, usize, usize) {
        (
            self.stats.parks.load(Ordering::Relaxed),
            self.waiters.load(Ordering::Acquire),
            self.nonresident.load(Ordering::Acquire),
        )
    }

    /// Acquire one grant. Uncontended, two free-list pops with no planner lock;
    /// otherwise the ask parks FCFS. [`Acquired::Yield`] means settle the tail.
    pub async fn acquire(
        self: &Arc<Self>,
        pid: ProcessId,
        quorum_id: ProcessId,
        demand: Demand,
    ) -> Result<Acquired, PlannerError> {
        if demand.is_zero() {
            // E6 progress event: a zero-demand fire is progress too. Protection only
            // matters under contention, so the uncontended hot path skips the lock.
            if self.waiters.load(Ordering::Acquire) != 0
                || self.nonresident.load(Ordering::Acquire) != 0
            {
                self.note_progress(pid);
            }
            return Ok(Acquired::Granted(AllocationGrant::empty()));
        }
        // Fast path. Uncontended: two free-list pops, no planner lock. Once
        // anything is queued the fast path closes.
        let uncontended = self.waiters.load(Ordering::Acquire) == 0
            && self.nonresident.load(Ordering::Acquire) == 0;
        // Opt-in (`PIE_ALLOC_FAST_SMALL=1`): serve straight from the free lists
        // when free pages cover this ask and the FCFS head's remaining shortfall.
        if !uncontended && fast_small_bypass_enabled() && demand.rs_slots == 0 {
            let (free_now, _) = self.port.device_stats();
            let head_covered = self.with_inner(|inner| {
                let head_shortfall = inner
                    .unmet_head()
                    .map(|(_, waiter)| waiter.kv_need().saturating_sub(inner.accum.len() as u32))
                    .unwrap_or(0);
                free_now >= demand.kv_pages.saturating_add(head_shortfall)
            });
            if head_covered && let Some(grant) = self.try_reserve(demand) {
                self.note_progress(pid);
                return Ok(Acquired::Granted(grant));
            }
        }
        if !uncontended {
            // E6 progress: this process is asking to fire again.
            self.note_progress(pid);
        }
        if uncontended && let Some(grant) = self.try_reserve(demand) {
            self.poke_if_runway_short();
            return Ok(Acquired::Granted(grant));
        }
        // Reserve failed (or fast path closed): run the exact-arithmetic
        // exhaustion check before parking — slow path only.
        let (_, kv_total) = self.port.device_stats();
        if demand.kv_pages > kv_total {
            return Err(PlannerError::Impossible {
                need: demand.kv_pages,
                total: kv_total,
            });
        }
        if demand.rs_slots > 0 {
            let (_, rs_total) = self.port.rs_stats();
            if demand.rs_slots > rs_total {
                return Err(PlannerError::Impossible {
                    need: demand.rs_slots,
                    total: rs_total,
                });
            }
        }
        enum Parked {
            Entry(EntryKey, Arc<Notify>),
            NotResident,
            Gone,
        }
        let parked = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get(&pid) else {
                return Parked::Gone;
            };
            if proc.state != Residency::Resident {
                return Parked::NotResident;
            }
            let key = (proc.seq, inner.next_id);
            inner.next_id += 1;
            let notify = Arc::new(Notify::new());
            inner.queue.insert(
                key,
                Waiter {
                    pid,
                    kind: WaitKind::Allocation {
                        demand,
                        notify: notify.clone(),
                        outcome: None,
                        yielded: false,
                    },
                },
            );
            Parked::Entry(key, notify)
        });
        match parked {
            Parked::Gone => Err(PlannerError::Cancelled),
            Parked::NotResident => {
                // Out of the set (or in transfer): the fire path settles
                // the process's tail and waits out the eviction.
                Ok(Acquired::Yield)
            }
            Parked::Entry(key, notify) => {
                self.stats.parks.fetch_add(1, Ordering::Relaxed);
                ptrace!("park key={:?} pid={} kv={}", key, pid, demand.kv_pages);
                // The park empties this lane's seat in the wait-all quorum so frames seal
                // without it; rejoin is implicit on the lane's next accepted fire.
                crate::scheduler::worker::notify_lane_close(quorum_id, Some(pid));
                let mut registration = WaitRegistration {
                    planner: self,
                    key,
                    active: true,
                };
                self.poke();
                loop {
                    let notified = notify.notified();
                    tokio::pin!(notified);
                    notified.as_mut().enable();
                    match self.collect_outcome(key) {
                        Collect::Ready(outcome) => {
                            registration.disarm();
                            ptrace!("collect key={:?} ok={}", key, outcome.is_ok());
                            if matches!(outcome, Err(PlannerError::Cancelled)) {
                                self.stats.cancelled_waits.fetch_add(1, Ordering::Relaxed);
                            }
                            // The collection unblocked the next head.
                            self.poke();
                            return outcome.map(Acquired::Granted);
                        }
                        Collect::Yield => {
                            registration.disarm();
                            self.poke();
                            return Ok(Acquired::Yield);
                        }
                        Collect::Wait => {}
                    }
                    notified.await;
                }
            }
        }
    }

    /// [`Self::acquire`]'s decision half, non-blocking: reproduces `acquire`'s
    /// logic, but returns an owned [`AllocationTicket`] instead of parking.
    #[allow(
        dead_code,
        reason = "the deferred-allocation park/collect half of `acquire`. Its \
                  consumer is upstream's `PIE_DEFER_ALLOC` fire path, whose handle \
                  `pipeline.rs` records as deliberately NOT carried by this merge \
                  (see the comment on `Pipeline`: reconciling it with this \
                  branch's kv-contention rewrite is named, visible work rather \
                  than a conflict resolution). Kept so that work has something to \
                  reconnect to. This allow also covers `Enqueued` and \
                  `AllocationTicket::collect`, which are reachable only from here"
    )]
    pub(crate) fn acquire_or_enqueue(
        self: &Arc<Self>,
        pid: ProcessId,
        quorum_id: ProcessId,
        demand: Demand,
    ) -> Result<Enqueued, PlannerError> {
        if demand.is_zero() {
            // E6 progress event — see `acquire`'s zero-demand path.
            if self.waiters.load(Ordering::Acquire) != 0
                || self.nonresident.load(Ordering::Acquire) != 0
            {
                self.note_progress(pid);
            }
            return Ok(Enqueued::Granted(AllocationGrant::empty()));
        }
        // Fast path — see `acquire`.
        let uncontended = self.waiters.load(Ordering::Acquire) == 0
            && self.nonresident.load(Ordering::Acquire) == 0;
        if !uncontended && fast_small_bypass_enabled() && demand.rs_slots == 0 {
            let (free_now, _) = self.port.device_stats();
            let head_covered = self.with_inner(|inner| {
                let head_shortfall = inner
                    .unmet_head()
                    .map(|(_, waiter)| waiter.kv_need().saturating_sub(inner.accum.len() as u32))
                    .unwrap_or(0);
                free_now >= demand.kv_pages.saturating_add(head_shortfall)
            });
            if head_covered && let Some(grant) = self.try_reserve(demand) {
                self.note_progress(pid);
                return Ok(Enqueued::Granted(grant));
            }
        }
        if !uncontended {
            // E6 progress: this process is asking to fire again.
            self.note_progress(pid);
        }
        if uncontended && let Some(grant) = self.try_reserve(demand) {
            self.poke_if_runway_short();
            return Ok(Enqueued::Granted(grant));
        }
        // Exact-arithmetic exhaustion check before parking — slow path only,
        // as in `acquire`.
        let (_, kv_total) = self.port.device_stats();
        if demand.kv_pages > kv_total {
            return Err(PlannerError::Impossible {
                need: demand.kv_pages,
                total: kv_total,
            });
        }
        if demand.rs_slots > 0 {
            let (_, rs_total) = self.port.rs_stats();
            if demand.rs_slots > rs_total {
                return Err(PlannerError::Impossible {
                    need: demand.rs_slots,
                    total: rs_total,
                });
            }
        }
        enum Parked {
            Entry(EntryKey, Arc<Notify>),
            NotResident,
            Gone,
        }
        let parked = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get(&pid) else {
                return Parked::Gone;
            };
            if proc.state != Residency::Resident {
                return Parked::NotResident;
            }
            let key = (proc.seq, inner.next_id);
            inner.next_id += 1;
            let notify = Arc::new(Notify::new());
            inner.queue.insert(
                key,
                Waiter {
                    pid,
                    kind: WaitKind::Allocation {
                        demand,
                        notify: notify.clone(),
                        outcome: None,
                        yielded: false,
                    },
                },
            );
            Parked::Entry(key, notify)
        });
        match parked {
            Parked::Gone => Err(PlannerError::Cancelled),
            Parked::NotResident => Ok(Enqueued::NotResident),
            Parked::Entry(key, notify) => {
                self.stats.parks.fetch_add(1, Ordering::Relaxed);
                ptrace!("park key={:?} pid={} kv={}", key, pid, demand.kv_pages);
                // Empties this lane's seat in the wait-all quorum so frames seal without
                // it; rejoin is implicit on the lane's next accepted fire.
                crate::scheduler::worker::notify_lane_close(quorum_id, Some(pid));
                let ticket = AllocationTicket {
                    planner: Arc::clone(self),
                    key,
                    notify,
                    collected: false,
                };
                self.poke();
                Ok(Enqueued::Ticket(ticket))
            }
        }
    }

    /// E6: mark `pid` as having progressed since its last restore, making it an
    /// eviction candidate again. Does not feed the lock-free mirrors.
    fn note_progress(&self, pid: ProcessId) {
        if let Some(proc) = self.lock_inner().procs.get_mut(&pid) {
            proc.progressed = true;
        }
    }

    /// One arming of a signal wait with the lost-wakeup guard: enable the
    /// waiter, re-check, then sleep until woken.
    async fn await_signal(&self, signal: &Notify, settled: impl Fn() -> bool) {
        let notified = signal.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if settled() {
            return;
        }
        notified.await;
    }

    fn collect_outcome(&self, key: EntryKey) -> Collect {
        self.with_inner(|inner| {
            let Some(waiter) = inner.queue.get_mut(&key) else {
                // Unregister purged the entry.
                return Collect::Ready(Err(PlannerError::Cancelled));
            };
            let WaitKind::Allocation {
                outcome, yielded, ..
            } = &mut waiter.kind
            else {
                unreachable!("allocation keys never collide with restore entries");
            };
            match outcome.take() {
                Some(outcome) => {
                    inner.queue.remove(&key);
                    Collect::Ready(outcome)
                }
                None if *yielded => {
                    inner.queue.remove(&key);
                    Collect::Yield
                }
                None => Collect::Wait,
            }
        })
    }

    fn cancel_waiter(self: &Arc<Self>, key: EntryKey) {
        let removed = self.with_inner(|inner| inner.queue.remove(&key));
        if let Some(waiter) = removed {
            // A parked grant (served but never collected) returns to the
            // pool here, outside the lock.
            drop(waiter);
            self.poke();
        }
    }

    // === The residency gate ===

    pub fn is_resident(&self, pid: ProcessId) -> bool {
        self.inner
            .lock()
            .procs
            .get(&pid)
            .is_none_or(|proc| proc.state == Residency::Resident)
    }

    /// This process's lock-free residency flag, read on every host-method
    /// prologue. `None` means unregistered, treated as resident.
    pub fn residency_flag(&self, pid: ProcessId) -> Option<Arc<AtomicBool>> {
        self.lock_inner()
            .procs
            .get(&pid)
            .map(|proc| proc.resident.clone())
    }

    /// Park until `pid` is resident again (or gone). Before the first park, a
    /// process-wide `Suspend` leave is re-posted here — idempotent.
    pub async fn wait_resident(&self, pid: ProcessId) -> Result<(), PlannerError> {
        let mut left_pipeline = false;
        loop {
            let signal = {
                let inner = self.lock_inner();
                let Some(proc) = inner.procs.get(&pid) else {
                    return Err(PlannerError::Cancelled);
                };
                if proc.state == Residency::Resident {
                    return Ok(());
                }
                proc.signal.clone()
            };
            if !left_pipeline {
                left_pipeline = true;
                crate::scheduler::worker::notify_process_suspend(pid);
            }
            self.stats.gate_parks.fetch_add(1, Ordering::Relaxed);
            self.await_signal(&signal, || self.is_resident(pid)).await;
        }
    }

    // === The drain — head-first accumulation ===

    /// Serve the queue: pull free pages into the accumulation until the head
    /// is covered, serve it, repeat. Falls through to rung 0 then eviction.
    pub fn plan(self: &Arc<Self>) {
        loop {
            let step = self.with_inner(|inner| {
                let Some((key, waiter)) = inner.unmet_head() else {
                    if inner.accum.len() > 0 && inner.queue.is_empty() {
                        let stranded = std::mem::take(&mut inner.accum);
                        return Step::Release(stranded);
                    }
                    return Step::Done;
                };
                let is_restore = matches!(waiter.kind, WaitKind::Restore { .. });
                let pid = waiter.pid;
                let demand = match &waiter.kind {
                    WaitKind::Allocation { demand, .. } => Some(*demand),
                    WaitKind::Restore { .. } => None,
                };
                let missing = waiter.kv_need().saturating_sub(inner.accum.len() as u32);
                if missing > 0 {
                    // Two currencies: an eviction may fund an allocation but never a restore
                    // — otherwise eviction and readmission are the same act and the pool oscillates.
                    let fund_by_eviction = !is_restore || inner.fleet_stalled();
                    return Step::Absorb {
                        count: missing,
                        fund_by_eviction,
                    };
                }
                match demand {
                    // An RS-carrying ask keeps the per-step path: its RS reservation is a port
                    // call, never touched under the planner lock.
                    Some(demand) if demand.rs_slots > 0 => Step::ServeAllocation { key, demand },
                    Some(_) => Step::ServeAllocationBurst {
                        extra: inner.burst_shortfall(),
                    },
                    None => {
                        // Rotation damping: a covered restore still waits for a completion credit
                        // — surplus must flow to allocations. `fleet_stalled` is the backstop.
                        if rotation_damping_enabled()
                            && inner.completion_credit == 0
                            && !inner.fleet_stalled()
                        {
                            return Step::Done;
                        }
                        inner.completion_credit = inner.completion_credit.saturating_sub(1);
                        Step::ServeRestore { key, pid }
                    }
                }
            });
            match step {
                Step::Done => {
                    // The drain has nothing to serve — exactly the idle phase in which the
                    // runway is rebuilt.
                    self.plan_runway();
                    return;
                }
                Step::Release(stranded) => {
                    drop(stranded);
                    // As above; the stranded accumulation just returned to the free lists, so
                    // the shortfall is measured against the honest free count.
                    self.plan_runway();
                    return;
                }
                Step::Absorb {
                    count,
                    fund_by_eviction,
                } => {
                    let pages = self.port.reserve_device_up_to(count);
                    if !pages.is_empty() {
                        let reservation = DevicePageReservation::new(pages, self.port.clone());
                        self.with_inner(|inner| inner.accum.absorb(reservation));
                        continue;
                    }
                    // Rung 0, latched: the cache-lease scan runs at most once per free event,
                    // or every poke at free=0 would re-scan fruitlessly.
                    if !self
                        .idle_reclaim_exhausted
                        .swap(true, std::sync::atomic::Ordering::AcqRel)
                        && self.port.reclaim_idle() > 0
                    {
                        self.idle_reclaim_exhausted.store(false, Ordering::Release);
                        continue;
                    }
                    if !fund_by_eviction {
                        // A restore head with a dry pool: nothing to do but wait for a completion.
                        // Safe because the head is only a restore when no allocation is unmet.
                        self.stats
                            .restore_absorb_short
                            .fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                    self.plan_eviction();
                    return;
                }
                Step::ServeAllocation { key, demand } => {
                    let rs = if demand.rs_slots > 0 {
                        match self.port.reserve_rs(demand.rs_slots) {
                            Some(slots) => RsSlotReservation::new(slots, self.port.clone()),
                            // RS pool short; the next RS free re-plans, but only a running process
                            // frees one — if every admitted process is parked the head waits forever.
                            None => {
                                self.check_rs_starvation(key, demand);
                                return;
                            }
                        }
                    } else {
                        RsSlotReservation::empty()
                    };
                    let outcome = self.with_inner(|inner| {
                        // Re-validate under the lock: still the oldest UNMET
                        // entry, still covered.
                        match inner.unmet_head() {
                            Some((head, _)) if head == key => {}
                            _ => return ServeOutcome::Stale(rs),
                        }
                        if (inner.accum.len() as u32) < demand.kv_pages {
                            return ServeOutcome::Stale(rs);
                        }
                        let kv = inner.accum.donate(demand.kv_pages as usize);
                        let waiter = inner.queue.get_mut(&key).expect("head exists");
                        let WaitKind::Allocation {
                            outcome, notify, ..
                        } = &mut waiter.kind
                        else {
                            unreachable!("ServeAllocation only targets allocation entries");
                        };
                        debug_assert!(outcome.is_none(), "unserved head carries no outcome");
                        *outcome = Some(Ok(AllocationGrant::new(demand, kv, rs)));
                        ServeOutcome::Served(notify.clone())
                    });
                    match outcome {
                        ServeOutcome::Served(notify) => {
                            self.stats.serves.fetch_add(1, Ordering::Relaxed);
                            notify.notify_waiters();
                            // Cascade: the served head no longer competes, so keep draining —
                            // the next-oldest unmet ask absorbs whatever free pages remain.
                            continue;
                        }
                        ServeOutcome::Stale(rs) => {
                            drop(rs);
                            continue;
                        }
                    }
                }
                Step::ServeAllocationBurst { extra } => {
                    // One free-list pull for the whole burst, outside the
                    // planner lock like every other port call.
                    if extra > 0 {
                        let pages = self.port.reserve_device_up_to(extra);
                        if !pages.is_empty() {
                            let reservation = DevicePageReservation::new(pages, self.port.clone());
                            self.with_inner(|inner| inner.accum.absorb(reservation));
                        }
                        // An empty pull is not a shortage to escalate: the head is already
                        // covered by the accumulation, which is why this step was chosen.
                    }
                    let wake = self.with_inner(|inner| inner.serve_burst());
                    if !wake.is_empty() {
                        self.stats
                            .serves
                            .fetch_add(wake.len() as u64, Ordering::Relaxed);
                    }
                    // Outside the lock: the woken collectors take it.
                    for (key, pages, notify) in wake {
                        ptrace!("serve key={:?} kv={}", key, pages);
                        notify.notify_waiters();
                    }
                    // Cascade, as on the per-step path: the served entries no longer compete,
                    // so keep draining.
                    continue;
                }
                Step::ServeRestore { key, pid } => {
                    // Re-validate the ask against the store: the demand is whatever is swapped
                    // NOW (teardown or discards may have shrunk it while it waited).
                    let (model, engine) = self.port.locus();
                    let swapped = swapped_page_count(pid, model, engine);
                    let boarded = self.with_inner(|inner| {
                        match inner.unmet_head() {
                            Some((head, _)) if head == key => {}
                            _ => return Board::Replan,
                        }
                        let Some(waiter) = inner.queue.get_mut(&key) else {
                            return Board::Replan;
                        };
                        let WaitKind::Restore { demand } = &mut waiter.kind else {
                            unreachable!("ServeRestore only targets restore entries");
                        };
                        if *demand != swapped {
                            *demand = swapped;
                            return Board::Replan; // coverage re-evaluates
                        }
                        if (inner.accum.len() as u32) < swapped {
                            return Board::Replan;
                        }
                        let Some(proc) = inner.procs.get_mut(&pid) else {
                            inner.queue.remove(&key);
                            return Board::Replan;
                        };
                        if proc.state != Residency::Evicted {
                            return Board::Replan;
                        }
                        proc.state = Residency::Restoring;
                        inner.queue.remove(&key);
                        Board::Go(inner.accum.donate(swapped as usize))
                    });
                    match boarded {
                        Board::Go(pages) => {
                            exec::spawn_restore(self.clone(), pid, pages);
                            continue;
                        }
                        Board::Replan => continue,
                    }
                }
            }
        }
    }

    // === Eviction planning — deficit-sized, youngest-first, younger than head ===

    /// Quote `candidates`, stopping once `deficit` is covered; return victims
    /// to evict plus ones host swap cannot currently hold.
    fn quote_and_pick(
        &self,
        candidates: Vec<(ProcessId, u64)>,
        deficit: u32,
        host_room: u32,
        model: usize,
        engine: usize,
    ) -> (Vec<(ProcessId, u32)>, Vec<ProcessId>) {
        let mut ordered: Vec<(ProcessId, u64, bool)> = candidates
            .into_iter()
            .map(|(pid, seq)| {
                let quiescent =
                    crate::inferlet::process::residency::kv_lease_quiescent(pid, model, engine);
                (pid, seq, quiescent)
            })
            .collect();
        ordered.sort_by_key(|&(_, seq, quiescent)| (!quiescent, std::cmp::Reverse(seq)));
        let pids: Vec<ProcessId> = ordered.iter().map(|(pid, ..)| *pid).collect();
        Self::pick_with_budget_escalation(&pids, deficit, host_room, |pids, budget| {
            crate::inferlet::process::residency::kv_reclaim_quotes(pids, model, engine, budget)
        })
    }

    /// The budgeted pick, plus a re-ask that keeps the budget from hiding a
    /// victim skipped for room, indistinguishable from unknown.
    fn pick_with_budget_escalation(
        pids: &[ProcessId],
        deficit: u32,
        host_room: u32,
        quote: impl Fn(&[ProcessId], u32) -> Vec<Option<ReclaimQuote>>,
    ) -> (Vec<(ProcessId, u32)>, Vec<ProcessId>) {
        let (picks, unhostable) =
            Self::pick_from_quotes(pids, quote(pids, deficit), deficit, host_room);
        if picks.iter().map(|&(_, pages)| pages).sum::<u32>() < deficit && !unhostable.is_empty() {
            return Self::pick_from_quotes(pids, quote(pids, u32::MAX), deficit, host_room);
        }
        (picks, unhostable)
    }

    /// Walk quotes in preference order, banking those the host pool can hold
    /// until `deficit` is covered and reporting the refused ones separately.
    fn pick_from_quotes(
        pids: &[ProcessId],
        quotes: Vec<Option<ReclaimQuote>>,
        deficit: u32,
        host_room: u32,
    ) -> (Vec<(ProcessId, u32)>, Vec<ProcessId>) {
        let mut picks = Vec::new();
        let mut unhostable = Vec::new();
        let mut covered = 0u32;
        let mut room = host_room;
        for (&pid, quote) in pids.iter().zip(quotes) {
            if covered >= deficit {
                break;
            }
            if let Some(ReclaimQuote::Pages(pages)) = quote
                && pages > 0
            {
                if pages > room {
                    unhostable.push(pid);
                    continue;
                }
                room -= pages;
                covered += pages;
                picks.push((pid, pages));
            }
        }
        (picks, unhostable)
    }

    fn plan_eviction(self: &Arc<Self>) {
        // Eviction funds the head only when KV bytes can physically move out: a
        // swap-incapable engine or exhausted host pool leaves starvation as the last rung.
        let (host_free, host_total) = self.port.host_stats();
        if !self.port.suspend_capable() || host_free == 0 {
            self.check_starvation(StarveCause::NoSwapRoom);
            return;
        }
        // Host reserve: the rung may not spend the last host pool on a fleet still
        // running. [`Self::is_wedged`] spends it; anything less defers.
        if host_free.saturating_mul(HOST_RESERVE_DIVISOR) <= host_total && !self.is_wedged() {
            self.stats
                .eviction_deferrals
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        // Load control: eviction is a supply rung, not a demand rung — the gate is
        // the arithmetic of supply, not a reaction to a shortage alone.
        if !self.supply_stalled() {
            self.stats
                .eviction_deferrals
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        let Some(victims) = self.victim_set() else {
            return;
        };
        let (model, engine) = self.port.locus();
        // Routine path: honour E6 hysteresis. `preferred()` may be empty while the
        // set is not — the endgame below must consult the set, never this subset.
        let (picks, unhostable) = self.quote_and_pick(
            victims.preferred(),
            victims.deficit,
            host_free,
            model,
            engine,
        );
        // A candidate whose reclaim does not fit the host pool is parked on the
        // same set `HostSwapFull` would park it on.
        if !unhostable.is_empty() {
            self.with_inner(|inner| {
                for pid in &unhostable {
                    inner.host_swap_blocked.insert(*pid);
                }
            });
            self.record_host_swap_exhaustion();
        }
        if picks.is_empty() {
            if !unhostable.is_empty() {
                // Same terminal answer as the `host_free == 0` gate above:
                // the pool, not victim availability, is what is missing.
                self.check_starvation(StarveCause::NoSwapRoom);
                return;
            }
            self.check_hog(victims.head, victims.head_pid);
            // The last-resort rung in `check_starvation` builds its own `VictimSet` at
            // the instant the kill is decided — reusing this one would be stale.
            self.check_starvation(StarveCause::NoEligibleVictim);
            return;
        }
        self.commit_evictions(victims.head, picks, false, victims.runway_grab);
    }

    /// Supply-phase runway, drain-idle entry (off = one flag load). Opportunistic
    /// unlike `plan_eviction`: an unquotable fleet simply means "no round".
    fn plan_runway(self: &Arc<Self>) {
        let runway = supply_runway_pages();
        if runway == 0 {
            return;
        }
        // Same physical precondition as `plan_eviction`, minus its
        // starvation rung: no transport or no host room is "no round".
        let (host_free, _) = self.port.host_stats();
        if !self.port.suspend_capable() || host_free == 0 {
            return;
        }
        let (free, _) = self.port.device_stats();
        let Some((deficit, members)) = self.runway_shortfall_set(runway, free) else {
            return;
        };
        let (model, engine) = self.port.locus();
        let (picks, _unhostable) = self.quote_and_pick(members, deficit, host_free, model, engine);
        if picks.is_empty() {
            return;
        }
        self.commit_runway(picks);
    }

    /// The detection + candidate half of [`Self::plan_runway`]: `Some` when the
    /// free list is short of `runway` and the rotation is credit-starved.
    fn runway_shortfall_set(&self, runway: u32, free: u32) -> Option<(u32, Vec<(ProcessId, u64)>)> {
        self.with_inner(|inner| {
            // No credit gate here: FCFS already orders queued boundary asks against
            // the victim's restore.
            if inner.runway_round_in_flight || inner.kill_in_flight() || !rotation_damping_enabled()
            {
                return None;
            }
            let expected: u32 = inner.evicting.values().map(|mark| mark.pages).sum();
            let deficit = runway
                .saturating_sub(free)
                .saturating_sub(inner.accum.len() as u32 + expected);
            if deficit == 0 {
                return None;
            }
            let floor = inner.runway_floor()?;
            let members: Vec<(ProcessId, u64)> = inner
                .procs
                .iter()
                .filter(|(pid, proc)| {
                    proc.admitted
                        && proc.seq > floor
                        && proc.state == Residency::Resident
                        && proc.progressed
                        && !inner.host_swap_blocked.contains(*pid)
                        && !inner.prepare_blocked.contains(*pid)
                })
                .map(|(pid, proc)| (*pid, proc.seq))
                .collect();
            if members.is_empty() {
                return None;
            }
            Some((deficit, members))
        })
    }

    /// Commit a runway round: [`Self::commit_evictions`]'s re-validation with
    /// [`Inner::runway_floor`] standing in for the head.
    fn commit_runway(self: &Arc<Self>, picks: Vec<(ProcessId, u32)>) {
        let mut spawned = Vec::new();
        let mut wake = Vec::new();
        let mut pages = 0u64;
        self.with_inner(|inner| {
            if inner.runway_round_in_flight || inner.kill_in_flight() {
                return; // a concurrent plan won; its round is the one
            }
            let Some(floor) = inner.runway_floor() else {
                return;
            };
            for (pid, expected) in picks {
                let Some(proc) = inner.procs.get_mut(&pid) else {
                    continue;
                };
                if proc.state != Residency::Resident || proc.seq <= floor || !proc.progressed {
                    continue;
                }
                proc.state = Residency::Evicting;
                inner.evicting.insert(pid, EvictionMark { pages: expected });
                pages += u64::from(expected);
                spawned.push(pid);
                // The victim's parked allocations yield exactly as on the routine path:
                // their fire tasks settle the tail the eviction quiesces on.
                for waiter in inner.queue.values_mut().filter(|w| w.pid == pid) {
                    if let WaitKind::Allocation {
                        notify, yielded, ..
                    } = &mut waiter.kind
                    {
                        *yielded = true;
                        wake.push(notify.clone());
                    }
                }
            }
            if !spawned.is_empty() {
                inner.runway_round_in_flight = true;
            }
        });
        for notify in wake {
            notify.notify_waiters();
        }
        if spawned.is_empty() {
            return;
        }
        self.stats.runway_rounds.fetch_add(1, Ordering::Relaxed);
        self.stats.runway_pages.fetch_add(pages, Ordering::Relaxed);
        for pid in spawned {
            self.stats.evictions_started.fetch_add(1, Ordering::Relaxed);
            exec::spawn_evict(self.clone(), pid);
        }
    }

    /// Snapshot every process FCFS permits evicting for the current head. A
    /// heuristic that can empty this set can destroy a request.
    fn victim_set(&self) -> Option<VictimSet> {
        // Read the pool before the planner lock (`inner` is innermost). One flag
        // load and no read at all with the runway off.
        let runway = supply_runway_pages();
        let runway_free = if runway > 0 {
            self.port.device_stats().0
        } else {
            0
        };
        self.with_inner(|inner| {
            let (head, waiter) = inner.unmet_head()?;
            // The round covers the head's shortfall plus the queued quantum-ask
            // stream — sizing by the head alone kept rounds at ~one victim.
            let queued_quantum: u32 = inner
                .queue
                .values()
                .filter(|queued| queued.is_unmet())
                .filter_map(|queued| match &queued.kind {
                    WaitKind::Allocation { demand, .. } if demand.kv_pages == 1 => Some(1),
                    _ => None,
                })
                .sum();
            // The round also restores the free-page runway, on the same brakes
            // as the `supply_stalled` clause that orders it.
            let runway_grab = if runway > 0 && !inner.runway_round_in_flight {
                runway.saturating_sub(runway_free)
            } else {
                0
            };
            let missing = waiter
                .kv_need()
                .max(queued_quantum.saturating_add(runway_grab))
                .saturating_sub(inner.accum.len() as u32);
            // Evictions already in flight fund the head when they land;
            // never over-evict for pages that are already on their way.
            let expected: u32 = inner.evicting.values().map(|mark| mark.pages).sum();
            let deficit = missing.saturating_sub(expected);
            if deficit == 0 {
                return None;
            }
            let head_seq = head.0;
            let members = inner
                .procs
                .iter()
                // Unadmitted processes hold no pooled pages by construction, so they can
                // never cover a deficit.
                .filter(|(pid, proc)| {
                    proc.admitted
                        && proc.seq > head_seq
                        && proc.state == Residency::Resident
                        && !inner.host_swap_blocked.contains(*pid)
                        && !inner.prepare_blocked.contains(*pid)
                })
                .map(|(pid, proc)| Victim {
                    pid: *pid,
                    seq: proc.seq,
                    e6_fresh: proc.progressed,
                })
                .collect();
            Some(VictimSet {
                head,
                head_pid: waiter.pid,
                deficit,
                members,
                runway_grab,
            })
        })
    }

    /// Mark the picked victims and spawn their eviction executors. Re-validates
    /// under the lock and yields each victim's parked allocations.
    fn commit_evictions(
        self: &Arc<Self>,
        head_key: EntryKey,
        picks: Vec<(ProcessId, u32)>,
        e6_relaxed: bool,
        runway_grab: u32,
    ) -> bool {
        let mut spawned = Vec::new();
        let mut wake = Vec::new();
        self.with_inner(|inner| {
            let head_seq = match inner.unmet_head() {
                Some((key, _)) if key == head_key => key.0,
                _ => return, // the head changed under us; the next plan re-runs
            };
            for (pid, expected) in picks {
                let Some(proc) = inner.procs.get_mut(&pid) else {
                    continue;
                };
                if proc.state != Residency::Resident
                    || proc.seq <= head_seq
                    || (!proc.progressed && !e6_relaxed)
                {
                    continue;
                }
                proc.state = Residency::Evicting;
                inner.evicting.insert(pid, EvictionMark { pages: expected });
                spawned.push(pid);
                for waiter in inner.queue.values_mut().filter(|w| w.pid == pid) {
                    if let WaitKind::Allocation {
                        notify, yielded, ..
                    } = &mut waiter.kind
                    {
                        *yielded = true;
                        wake.push(notify.clone());
                    }
                }
            }
            if runway_grab > 0 && !spawned.is_empty() {
                inner.runway_round_in_flight = true;
            }
        });
        for notify in wake {
            notify.notify_waiters();
        }
        let any = !spawned.is_empty();
        if runway_grab > 0 && any {
            self.stats.runway_rounds.fetch_add(1, Ordering::Relaxed);
            self.stats
                .runway_pages
                .fetch_add(u64::from(runway_grab), Ordering::Relaxed);
        }
        for pid in spawned {
            self.stats.evictions_started.fetch_add(1, Ordering::Relaxed);
            exec::spawn_evict(self.clone(), pid);
        }
        any
    }

    /// The starvation endgame, as a computed predicate: fires when the head is
    /// unmet and unfundable and nobody registered and admitted is running.
    fn supply_stalled(&self) -> bool {
        let (free, _) = self.port.device_stats();
        self.with_inner(|inner| {
            if inner.kill_in_flight() {
                return false;
            }
            if inner.unmet_head().is_none() {
                return false;
            }
            // Supply-phase runway (default off): the free list itself is a demand, so
            // a shortfall against the target opens the rung before asks even park.
            let runway = supply_runway_pages();
            if runway > 0 && !inner.runway_round_in_flight {
                let expected: u32 = inner.evicting.values().map(|mark| mark.pages).sum();
                if runway.saturating_sub(free) > inner.accum.len() as u32 + expected {
                    return true;
                }
            }
            // Demand-exact over the whole queued allocation demand, not just the
            // head's ask, so the free list runs slightly ahead of the boundary-ask stream.
            let queued_quantum: u32 = inner
                .queue
                .values()
                .filter(|waiter| waiter.is_unmet())
                .filter_map(|waiter| match &waiter.kind {
                    WaitKind::Allocation { demand, .. } if demand.kv_pages == 1 => Some(1),
                    _ => None,
                })
                .sum();
            let expected: u32 = inner.evicting.values().map(|mark| mark.pages).sum();
            let supply = free + inner.accum.len() as u32 + expected;
            if queued_quantum > supply {
                return true;
            }
            // The classic liveness form: the head's own shortfall with the
            // pool empty.
            free == 0
                && inner
                    .unmet_head()
                    .is_some_and(|(_, head)| head.kv_need() > inner.accum.len() as u32 + expected)
        })
    }

    /// No completion can ever arrive on its own — see [`Inner::fleet_stalled`].
    /// Shared by the starvation rung and the eviction rung's last resort.
    fn is_wedged(&self) -> bool {
        self.with_inner(|inner| inner.fleet_stalled())
    }

    /// Last-resort victim search, immediately before the starvation rung would
    /// destroy a request — decided from one atomic snapshot.
    fn last_resort_evict(self: &Arc<Self>) -> bool {
        let (model, engine) = self.port.locus();
        let Some(stores) = crate::store::registry::try_get(model, engine) else {
            return false;
        };
        // Outside every lock: which processes might we quote?
        let pids: Vec<ProcessId> = self.with_inner(|inner| inner.procs.keys().copied().collect());
        if pids.is_empty() {
            return false;
        }
        let working_sets =
            crate::inferlet::process::residency::kv_working_sets_for(&pids, model, engine);

        // One atomic decision: store lock outside, planner lock inside.
        let (head, _deficit, picks) =
            crate::store::registry::with_kv_lock(&stores.kv, "planner-endgame", |kv| {
                self.with_inner(|inner| {
                    let Some((head, waiter)) = inner.unmet_head() else {
                        return (None, 0, Vec::new());
                    };
                    let missing = waiter.kv_need().saturating_sub(inner.accum.len() as u32);
                    let expected: u32 = inner.evicting.values().map(|mark| mark.pages).sum();
                    let deficit = missing.saturating_sub(expected);
                    if deficit == 0 {
                        return (None, 0, Vec::new());
                    }
                    let head_seq = head.0;
                    // Every pid, not a deficit-bounded prefix: the legality filter below
                    // is planner state the quote order knows nothing about.
                    let quotes = crate::inferlet::process::residency::quote_locked(
                        kv,
                        working_sets,
                        u32::MAX,
                    );
                    // Legal victims: younger than the head and resident. E6 hysteresis is
                    // waived here only, as a preference never a survival reason.
                    let mut legal: Vec<(ProcessId, u64, bool, u32)> = pids
                        .iter()
                        .zip(quotes)
                        .filter_map(|(pid, quote)| {
                            let proc = inner.procs.get(pid)?;
                            if proc.seq <= head_seq
                                || proc.state != Residency::Resident
                                || inner.host_swap_blocked.contains(pid)
                                || inner.prepare_blocked.contains(pid)
                            {
                                return None;
                            }
                            match quote? {
                                ReclaimQuote::Pages(pages) if pages > 0 => {
                                    Some((*pid, proc.seq, proc.progressed, pages))
                                }
                                _ => None,
                            }
                        })
                        .collect();
                    legal.sort_by_key(|&(_, seq, fresh, _)| (!fresh, std::cmp::Reverse(seq)));
                    let mut picks = Vec::new();
                    let mut covered = 0u32;
                    for (pid, _, _, pages) in legal {
                        if covered >= deficit {
                            break;
                        }
                        covered += pages;
                        picks.push((pid, pages));
                    }
                    (Some(head), deficit, picks)
                })
            });
        let (Some(head), false) = (head, picks.is_empty()) else {
            return false;
        };
        // The last-resort rung is liveness, never runway: `runway_grab` 0.
        if !self.commit_evictions(head, picks, true, 0) {
            return false;
        }
        self.stats.e6_relaxations.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Pull pages that returned to the pool since the drain's failed absorb
    /// into the head's accumulation. Re-runs the drain's own primitive.
    fn salvage_free_pages(&self) -> bool {
        let missing = self.with_inner(|inner| {
            let (_, head) = inner.unmet_head()?;
            let missing = head.kv_need().saturating_sub(inner.accum.len() as u32);
            (missing > 0).then_some(missing)
        });
        let Some(missing) = missing else {
            return false;
        };
        let pages = self.port.reserve_device_up_to(missing);
        if pages.is_empty() {
            return false;
        }
        let reservation = DevicePageReservation::new(pages, self.port.clone());
        self.with_inner(|inner| inner.accum.absorb(reservation));
        self.stats.salvages.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Last rung before destruction: serve the oldest queued ask the head's own
    /// hoard already covers — only inside the wedge, only once uncoverable.
    fn serve_from_hoard(self: &Arc<Self>) -> bool {
        let Some((key, demand)) = self.with_inner(|inner| {
            let (head_key, _) = inner.unmet_head()?;
            let hoard = inner.accum.len() as u32;
            inner.queue.iter().find_map(|(&key, waiter)| {
                if key == head_key || !waiter.is_unmet() {
                    return None;
                }
                match &waiter.kind {
                    WaitKind::Allocation { demand, .. } if demand.kv_pages <= hoard => {
                        Some((key, *demand))
                    }
                    _ => None,
                }
            })
        }) else {
            return false;
        };
        let rs = if demand.rs_slots > 0 {
            match self.port.reserve_rs(demand.rs_slots) {
                Some(slots) => RsSlotReservation::new(slots, self.port.clone()),
                None => return false,
            }
        } else {
            RsSlotReservation::empty()
        };
        // Mirrors `Step::ServeAllocation`: re-validate under the lock, and hand a
        // rejected RS reservation back OUT to drop outside the planner lock.
        let outcome = self.with_inner(|inner| {
            if (inner.accum.len() as u32) < demand.kv_pages {
                return ServeOutcome::Stale(rs);
            }
            let Some(waiter) = inner.queue.get_mut(&key) else {
                return ServeOutcome::Stale(rs);
            };
            let WaitKind::Allocation {
                demand: queued,
                outcome,
                notify,
                yielded,
            } = &mut waiter.kind
            else {
                return ServeOutcome::Stale(rs);
            };
            if outcome.is_some() || *yielded || *queued != demand {
                return ServeOutcome::Stale(rs);
            }
            let kv = inner.accum.donate(demand.kv_pages as usize);
            *outcome = Some(Ok(AllocationGrant::new(demand, kv, rs)));
            ServeOutcome::Served(notify.clone())
        });
        match outcome {
            ServeOutcome::Served(notify) => {
                self.stats.serves.fetch_add(1, Ordering::Relaxed);
                self.stats.hoard_bypasses.fetch_add(1, Ordering::Relaxed);
                notify.notify_waiters();
                self.poke();
                true
            }
            ServeOutcome::Stale(rs) => {
                drop(rs);
                false
            }
        }
    }

    /// The RS analogue of [`Self::check_starvation`], for the resource the
    /// eviction ladder cannot fund. The head is failed, not the youngest.
    fn check_rs_starvation(self: &Arc<Self>, key: EntryKey, demand: Demand) {
        if !self.is_wedged() {
            return;
        }
        // The predicate was evaluated after a reservation that already failed, so
        // re-read the pool: a slot freed in between makes the question moot.
        let (free, total) = self.port.rs_stats();
        if free >= demand.rs_slots {
            self.poke();
            return;
        }
        if !self.is_wedged() {
            return;
        }
        let notify = self.with_inner(|inner| {
            // Still the same unmet head, still unserved.
            match inner.unmet_head() {
                Some((head, _)) if head == key => {}
                _ => return None,
            }
            let waiter = inner.queue.get_mut(&key)?;
            let WaitKind::Allocation {
                notify, outcome, ..
            } = &mut waiter.kind
            else {
                return None;
            };
            if outcome.is_some() {
                return None;
            }
            *outcome = Some(Err(PlannerError::Starved {
                need: demand.rs_slots,
                free,
                total,
                cause: StarveCause::NoRsSlots,
            }));
            Some(notify.clone())
        });
        if let Some(notify) = notify {
            self.stats.starvations.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                need = demand.rs_slots,
                free,
                total,
                "planner: RS slot pool wedged — failing the head"
            );
            notify.notify_waiters();
        }
    }

    fn check_starvation(self: &Arc<Self>, cause: StarveCause) {
        if !self.is_wedged() {
            return;
        }
        // The wedge predicate can be stale in the one direction that matters:
        // pages back in the pool. Salvage before evicting anyone.
        if self.salvage_free_pages() {
            self.poke();
            return;
        }
        // Last rung before destruction: re-scan for any evictable victim younger
        // than the head. Only for `NoEligibleVictim`.
        if cause == StarveCause::NoEligibleVictim && self.last_resort_evict() {
            return;
        }
        // The relaxed scan may itself have raced; re-verify before killing.
        if !self.is_wedged() {
            return;
        }
        // Final salvage, immediately before destruction: `last_resort_evict` walks
        // every proc under the global KV lock, ample time for teardown to land.
        if self.salvage_free_pages() {
            self.poke();
            return;
        }
        // Nothing can be assembled for the head. Before destroying anyone,
        // check whether the head's own hoard already covers a younger ask.
        if self.serve_from_hoard() {
            return;
        }
        let (free, total) = self.port.device_stats();
        // A destruction already ordered has not been paid out yet: gate the rung
        // that destroys, and only that one.
        if self.with_inner(|inner| inner.kill_in_flight()) {
            return;
        }
        // Pick the victim outside the lock: the choice needs reclaim
        // quotes, and quoting takes store locks.
        let Some(victim_key) = self.pick_starvation_victim() else {
            return;
        };
        let notify = self.with_inner(|inner| {
            // Re-validate: still an unmet head, still no transfers.
            let (_, head) = inner.unmet_head()?;
            let head_need = head.kv_need();
            if head_need <= inner.accum.len() as u32 || !inner.evicting.is_empty() {
                return None;
            }
            let victim = inner.queue.get_mut(&victim_key)?;
            let victim_pid = victim.pid;
            let WaitKind::Allocation {
                demand,
                notify,
                outcome,
                ..
            } = &mut victim.kind
            else {
                return None; // re-typed under the lock: re-plan instead
            };
            if outcome.is_some() {
                return None; // served while we were quoting
            }
            *outcome = Some(Err(PlannerError::Starved {
                need: demand.kv_pages,
                free,
                total,
                cause,
            }));
            let notify = notify.clone();
            inner.killing.insert(victim_pid);
            Some((notify, victim_pid))
        });
        if let Some((notify, victim_pid)) = notify {
            self.stats.starvations.fetch_add(1, Ordering::Relaxed);
            // Reclaiming a process's pages is not the same as failing its request: a
            // restartable guest is re-queued at the same FCFS position.
            let restarting = crate::inferlet::process::request_restart(victim_pid);
            if restarting {
                self.stats
                    .starvation_restarts
                    .fetch_add(1, Ordering::Relaxed);
                tracing::info!(
                    cause = ?cause,
                    pid = %victim_pid,
                    "planner: pool starved — reclaiming a restartable process, its work is re-queued"
                );
            } else {
                tracing::warn!(
                    cause = ?cause,
                    pid = %victim_pid,
                    "planner: pool starved with no reclaim path — failing the youngest parked ask that holds pages"
                );
            }
            notify.notify_waiters();
        }
    }

    /// Choose which parked ask to destroy, youngest-first, restricted to asks
    /// whose destruction actually returns pages.
    fn pick_starvation_victim(&self) -> Option<EntryKey> {
        // Youngest-first parked allocations. Restores are never victims, and
        // an already-served entry is not parked.
        let candidates: Vec<(EntryKey, ProcessId)> = self.with_inner(|inner| {
            inner
                .queue
                .iter()
                .rev()
                .filter(|(_, waiter)| {
                    matches!(&waiter.kind, WaitKind::Allocation { outcome: None, .. })
                })
                .map(|(key, waiter)| (*key, waiter.pid))
                .collect()
        });
        let fallback = candidates.first().map(|(key, _)| *key);
        let (model, engine) = self.port.locus();
        // Two passes over the same youngest-first order: spend restartable
        // holders first, since they only lose time.
        let mut first_holder = None;
        for restartable_only in [true, false] {
            let selected: Vec<(EntryKey, ProcessId)> = candidates
                .iter()
                .filter(|(_, pid)| {
                    !restartable_only || crate::inferlet::process::is_restartable(*pid)
                })
                .copied()
                .collect();
            if selected.is_empty() {
                continue;
            }
            let pids: Vec<ProcessId> = selected.iter().map(|(_, pid)| *pid).collect();
            // The loop below stops at the first candidate holding anything,
            // so one page of budget quotes exactly the prefix it reads.
            let quotes =
                crate::inferlet::process::residency::kv_reclaim_quotes(&pids, model, engine, 1);
            for ((key, _), quote) in selected.iter().zip(quotes) {
                if let Some(ReclaimQuote::Pages(pages)) = quote
                    && pages > 0
                {
                    if restartable_only {
                        return Some(*key);
                    }
                    first_holder.get_or_insert(*key);
                    break;
                }
            }
        }
        first_holder.or(fallback)
    }

    /// The hog endgame, as a computed predicate: no transfer in flight and the
    /// head's own holdings plus its ask exceed the pool — fail the head loud.
    fn check_hog(self: &Arc<Self>, head_key: EntryKey, head_pid: ProcessId) {
        let transfers_in_flight = self.with_inner(|inner| {
            !inner.evicting.is_empty()
                || inner
                    .procs
                    .values()
                    .any(|proc| proc.state == Residency::Restoring)
        });
        if transfers_in_flight {
            return;
        }
        let (model, engine) = self.port.locus();
        // The head's own footprint must be the durable `held_page_count`,
        // never a `ReclaimQuote` (reports 0 for `Nothing`, undercounting a hog).
        let held = held_page_count(head_pid, model, engine);
        let (_, total) = self.port.device_stats();
        let notify = self.with_inner(|inner| {
            let waiter = inner.queue.get_mut(&head_key)?;
            let WaitKind::Allocation {
                demand,
                notify,
                outcome,
                ..
            } = &mut waiter.kind
            else {
                // A restore head can always wait: its demand once fit (it
                // was resident) and shrinks monotonically with discards.
                return None;
            };
            if outcome.is_some() || demand.kv_pages.saturating_add(held) <= total {
                return None;
            }
            *outcome = Some(Err(PlannerError::Hog {
                need: demand.kv_pages,
                held,
                total,
            }));
            Some(notify.clone())
        });
        if let Some(notify) = notify {
            self.stats.hog_failures.fetch_add(1, Ordering::Relaxed);
            notify.notify_waiters();
        }
    }

    // === Executor callbacks (planner::exec) ===

    /// D2H committed: `pid` is out of the set, its restore entry queues at
    /// its spawn position, and its freed pages drain to the head.
    fn report_evicted(self: &Arc<Self>, pid: ProcessId, freed: u32) {
        self.with_inner(|inner| {
            inner.settle_eviction(pid);
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return;
            };
            debug_assert_eq!(proc.state, Residency::Evicting);
            proc.state = Residency::Evicted;
            let key = (proc.seq, inner.next_id);
            inner.next_id += 1;
            inner.queue.insert(
                key,
                Waiter {
                    pid,
                    kind: WaitKind::Restore { demand: freed },
                },
            );
        });
        self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        self.stats
            .d2h_pages
            .fetch_add(u64::from(freed), Ordering::Relaxed);
        self.re_arm_idle_reclaim();
        self.poke();
    }

    /// An eviction attempt was abandoned before commit; the process stays
    /// resident and the next poke re-plans (possibly picking someone else).
    fn eviction_failed(self: &Arc<Self>, pid: ProcessId) {
        self.eviction_failed_inner(pid, false, false);
    }

    /// `prepare_suspend` deferred: nothing movable at this instant. Parked for
    /// the same deterministic-re-pick reason as `HostSwapFull`.
    fn eviction_failed_prepare_deferred(self: &Arc<Self>, pid: ProcessId) {
        self.eviction_failed_inner(pid, false, true);
    }

    /// `HostSwapFull`: this victim's bytes have nowhere to go. Park until host
    /// room returns rather than re-running the whole cycle to fail identically.
    fn eviction_failed_host_swap_full(self: &Arc<Self>, pid: ProcessId) {
        self.record_host_swap_exhaustion();
        self.eviction_failed_inner(pid, true, false);
    }

    fn eviction_failed_inner(
        self: &Arc<Self>,
        pid: ProcessId,
        host_swap_full: bool,
        prepare_deferred: bool,
    ) {
        let signal = self.with_inner(|inner| {
            inner.settle_eviction(pid);
            if host_swap_full {
                inner.host_swap_blocked.insert(pid);
            }
            if prepare_deferred {
                inner.prepare_blocked.insert(pid);
            }
            let proc = inner.procs.get_mut(&pid)?;
            if proc.state == Residency::Evicting {
                proc.state = Residency::Resident;
            }
            Some(proc.signal.clone())
        });
        self.stats
            .eviction_rollbacks
            .fetch_add(1, Ordering::Relaxed);
        // The victim never left, so undo the suspend's wait-set effects: without
        // this it would stay marked suspended with frames cut to single slots.
        crate::scheduler::worker::notify_process_resume(pid);
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        self.poke();
    }

    /// Host slots came back, so every host-swap-blocked victim is a candidate
    /// again. Cleared wholesale so none leaks and stays parked forever.
    fn clear_host_swap_blocks(&self) {
        let cleared = self.with_inner(|inner| {
            if inner.host_swap_blocked.is_empty() && inner.prepare_blocked.is_empty() {
                return false;
            }
            inner.host_swap_blocked.clear();
            inner.prepare_blocked.clear();
            true
        });
        if cleared {
            self.stats
                .host_swap_unblocks
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// H2D committed: `pid` is resident again, so its working-set fences drop.
    /// Owning the unfence here makes "restored ⇒ unfenced" structural.
    fn report_restored(self: &Arc<Self>, pid: ProcessId, restored: u32) {
        // The restored pages' host slots are back in the swap pool: every
        // victim parked on `HostSwapFull` may now fit.
        self.clear_host_swap_blocks();
        let (model, engine) = self.port.locus();
        for handle in crate::inferlet::process::residency::kv_suspend_handles(pid, model, engine) {
            handle.unfence();
        }
        let signal = self.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid)?;
            debug_assert_eq!(proc.state, Residency::Restoring);
            proc.state = Residency::Resident;
            proc.restore_retried = false;
            // E6: not re-evictable until its next `acquire`.
            proc.progressed = false;
            Some(proc.signal.clone())
        });
        // Runnable again: its lanes rejoin the wait-set and batch full
        // frames (the mirror of the eviction's `notify_process_suspend`).
        crate::scheduler::worker::notify_process_resume(pid);
        self.stats.restores.fetch_add(1, Ordering::Relaxed);
        self.stats
            .h2d_pages
            .fetch_add(u64::from(restored), Ordering::Relaxed);
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        self.poke();
    }

    /// A restore broke somewhere that says nothing about what the transfer left
    /// behind, so there is no clean evicted state to return to — fail loud.
    fn restore_failed(self: &Arc<Self>, pid: ProcessId, reason: &str) {
        let live = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return false;
            };
            if proc.state != Residency::Restoring {
                return false;
            }
            proc.state = Residency::Evicted;
            true
        });
        self.stats.restore_failures.fetch_add(1, Ordering::Relaxed);
        if live {
            self.fail_restore_loud(pid, reason);
        } else {
            self.poke(); // stale report: the process already moved on.
        }
    }

    /// A restore broke before anything was committed, so the process re-enters
    /// the queue at its spawn position. Once per episode.
    fn restore_deferred(self: &Arc<Self>, pid: ProcessId, reason: &str) {
        enum Outcome {
            /// The report no longer applies — the process terminated, or
            /// another path already moved it out of `Restoring`.
            Stale,
            Requeued,
            /// The one re-queue was already spent this episode.
            Spent,
        }
        let outcome = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return Outcome::Stale;
            };
            if proc.state != Residency::Restoring {
                return Outcome::Stale;
            }
            proc.state = Residency::Evicted;
            if proc.restore_retried {
                return Outcome::Spent;
            }
            proc.restore_retried = true;
            let key = (proc.seq, inner.next_id);
            inner.next_id += 1;
            inner.queue.insert(
                key,
                Waiter {
                    pid,
                    kind: WaitKind::Restore { demand: 0 },
                },
            );
            Outcome::Requeued
        });
        self.stats.restore_failures.fetch_add(1, Ordering::Relaxed);
        match outcome {
            Outcome::Stale | Outcome::Requeued => self.poke(),
            Outcome::Spent => self.fail_restore_loud(pid, reason),
        }
    }

    fn fail_restore_loud(self: &Arc<Self>, pid: ProcessId, reason: &str) {
        let reason = format!("KV restore failed: {reason}");
        tracing::error!(pid = %pid, %reason, "planner: failing evicted process loud");
        crate::inferlet::process::terminate(pid, Err(reason));
    }

    pub fn record_host_swap_exhaustion(&self) {
        self.stats
            .host_swap_exhaustions
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_d2h_copy(&self, elapsed: Duration) {
        self.stats
            .d2h_copy_us
            .fetch_add(micros(elapsed), Ordering::Relaxed);
    }

    pub fn record_h2d_copy(&self, elapsed: Duration) {
        self.stats
            .h2d_copy_us
            .fetch_add(micros(elapsed), Ordering::Relaxed);
    }

    // === Telemetry ===

    pub fn diagnostics(&self) -> PlannerDiagnostics {
        let (device_pages_free, device_pages_total) = self.port.device_stats();
        let (host_slots_free, host_slots_total) = self.port.host_stats();
        let (rs_slots_free, rs_slots_total) = self.port.rs_stats();
        let inner = self.lock_inner();
        let queue = inner
            .queue
            .iter()
            .map(|(key, waiter)| PlannerQueueEntry {
                process_id: waiter.pid.to_string(),
                spawn_seq: key.0,
                kind: match &waiter.kind {
                    WaitKind::Allocation { .. } => "allocation",
                    WaitKind::Restore { .. } => "restore",
                },
                pages: waiter.kv_need(),
                rs_slots: match &waiter.kind {
                    WaitKind::Allocation { demand, .. } => demand.rs_slots,
                    WaitKind::Restore { .. } => 0,
                },
            })
            .collect();
        // The entry the drain is actually blocked on. `queue.first()` is merely
        // the oldest entry, met or not.
        let (unmet_head_pages, unmet_head_kind) = match inner.unmet_head() {
            Some((_, waiter)) => (
                waiter.kv_need(),
                match &waiter.kind {
                    WaitKind::Allocation { .. } => "allocation",
                    WaitKind::Restore { .. } => "restore",
                },
            ),
            None => (0, "-"),
        };
        let unmet_queued = inner
            .queue
            .values()
            .filter(|waiter| waiter.is_unmet())
            .count() as u32;
        // What the head-first rule costs right now: walk PAST the blocked head and
        // see how much the free stock alone would already cover.
        let (bypassable_entries, bypassable_pages) = {
            let head_seq = inner.unmet_head().map(|(key, _)| key.0);
            match head_seq {
                None => (0, 0),
                Some(head) => {
                    let mut budget = device_pages_free;
                    let (mut entries, mut pages) = (0u32, 0u32);
                    for (key, waiter) in inner.queue.iter() {
                        if key.0 <= head || !waiter.is_unmet() {
                            continue;
                        }
                        let need = waiter.kv_need();
                        if need <= budget {
                            budget -= need;
                            entries += 1;
                            pages += need;
                        }
                    }
                    (entries, pages)
                }
            }
        };
        let parked: std::collections::HashSet<ProcessId> =
            inner.queue.values().map(|waiter| waiter.pid).collect();
        let mut runner_ids: Vec<(ProcessId, u64, bool)> = inner
            .procs
            .iter()
            .filter(|(pid, proc)| proc.admitted && !parked.contains(pid))
            .map(|(pid, proc)| (*pid, proc.seq, proc.progressed))
            .collect();
        runner_ids.sort_by_key(|&(_, seq, _)| seq);
        runner_ids.truncate(RUNNER_DUMP_CAP);
        let mut proc_states = [0u32; 4];
        let mut admitted_procs = 0u32;
        for proc in inner.procs.values() {
            let slot = match proc.state {
                Residency::Resident => 0,
                Residency::Evicting => 1,
                Residency::Evicted => 2,
                Residency::Restoring => 3,
            };
            proc_states[slot] += 1;
            admitted_procs += u32::from(proc.admitted);
        }
        let accumulation = inner.accum.len() as u32;
        drop(inner);
        // Outside the planner lock: `held_page_count` takes RESIDENCIES and
        // the KV store, both of which order before `inner`.
        let (model, engine) = self.port.locus();
        let runners = runner_ids
            .into_iter()
            .map(|(pid, seq, progressed)| (seq, held_page_count(pid, model, engine), progressed))
            .collect();
        use Ordering::Relaxed;
        PlannerDiagnostics {
            device_pages_free,
            device_pages_total,
            host_slots_free,
            host_slots_total,
            rs_slots_free,
            rs_slots_total,
            accumulation,
            queue,
            proc_states,
            admitted_procs,
            runners,
            parks_total: self.stats.parks.load(Relaxed),
            serves_total: self.stats.serves.load(Relaxed),
            evictions_total: self.stats.evictions.load(Relaxed),
            eviction_deferrals_total: self.stats.eviction_deferrals.load(Relaxed),
            unmet_head_pages,
            unmet_head_kind,
            unmet_queued,
            bypassable_entries,
            bypassable_pages,
            eviction_rollbacks_total: self.stats.eviction_rollbacks.load(Relaxed),
            restores_total: self.stats.restores.load(Relaxed),
            restore_failures_total: self.stats.restore_failures.load(Relaxed),
            gate_parks_total: self.stats.gate_parks.load(Relaxed),
            cancelled_waits_total: self.stats.cancelled_waits.load(Relaxed),
            hog_failures_total: self.stats.hog_failures.load(Relaxed),
            starvations_total: self.stats.starvations.load(Relaxed),
            starvation_restarts_total: self.stats.starvation_restarts.load(Relaxed),
            salvages_total: self.stats.salvages.load(Relaxed),
            hoard_bypasses_total: self.stats.hoard_bypasses.load(Relaxed),
            e6_relaxations_total: self.stats.e6_relaxations.load(Relaxed),
            restore_absorb_short_total: self.stats.restore_absorb_short.load(Relaxed),
            host_swap_exhaustions_total: self.stats.host_swap_exhaustions.load(Relaxed),
            host_swap_unblocks_total: self.stats.host_swap_unblocks.load(Relaxed),
            d2h_pages_total: self.stats.d2h_pages.load(Relaxed),
            h2d_pages_total: self.stats.h2d_pages.load(Relaxed),
            d2h_copy_us_total: self.stats.d2h_copy_us.load(Relaxed),
            h2d_copy_us_total: self.stats.h2d_copy_us.load(Relaxed),
            runway_rounds_total: self.stats.runway_rounds.load(Relaxed),
            runway_pages_total: self.stats.runway_pages.load(Relaxed),
        }
    }

    /// Coarse pressure signal for the worker heartbeat. Pool stats plus two
    /// lock-free flags — never the full [`Self::diagnostics`] snapshot.
    pub fn kv_pressure_bucket(&self) -> u8 {
        let (device_free, device_total) = self.port.device_stats();
        let (host_free, host_total) = self.port.host_stats();
        let ratio = |free: u32, total: u32| {
            if total == 0 {
                0.0
            } else {
                f64::from(total.saturating_sub(free)) / f64::from(total)
            }
        };
        let mut bucket = (ratio(device_free, device_total).max(ratio(host_free, host_total))
            * 255.0)
            .round() as u8;
        if self.nonresident.load(Ordering::Acquire) != 0 {
            bucket = bucket.max(224);
        }
        if self.waiters.load(Ordering::Acquire) != 0 {
            bucket = bucket.max(240);
        }
        bucket
    }
}

enum Collect {
    Ready(Result<AllocationGrant, PlannerError>),
    Yield,
    Wait,
}

enum ServeOutcome {
    Served(Arc<Notify>),
    Stale(RsSlotReservation),
}

enum Board {
    Go(DevicePageReservation),
    Replan,
}

/// `PIE_ALLOC_FAST_SMALL=1`: allow the head-harmless free-list bypass in
/// `acquire`. Default off.
fn fast_small_bypass_enabled() -> bool {
    static CONFIGURED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CONFIGURED.get_or_init(|| {
        std::env::var("PIE_ALLOC_FAST_SMALL").is_ok_and(|value| !value.is_empty() && value != "0")
    })
}

/// Probe-only: `PIE_ROTATION_DAMPING=0` disables the completion-credit gate
/// on covered restores. Default on.
fn rotation_damping_enabled() -> bool {
    static CONFIGURED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CONFIGURED
        .get_or_init(|| !std::env::var("PIE_ROTATION_DAMPING").is_ok_and(|value| value == "0"))
}

/// `PIE_SUPPLY_RUNWAY=<pages>`: the supply-phase runway target. Sizes
/// eviction rounds toward this many free device pages. Unset/`0` = off.
fn supply_runway_pages() -> u32 {
    static CONFIGURED: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CONFIGURED.get_or_init(|| {
        std::env::var("PIE_SUPPLY_RUNWAY")
            .ok()
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(0)
    })
}

fn micros(elapsed: Duration) -> u64 {
    elapsed.as_micros().min(u128::from(u64::MAX)) as u64
}

/// The process's swapped page count on `(model, engine)` — the exact restore
/// ask, computed fresh at serve time.
fn swapped_page_count(pid: ProcessId, model: usize, engine: usize) -> u32 {
    kv_page_count(pid, model, engine, |kv, ws| {
        kv.swapped_page_count(ws).unwrap_or(0)
    })
}

/// Every device+host page this process holds — the durable fact the
/// liveness predicates need, never a [`ReclaimQuote`].
fn held_page_count(pid: ProcessId, model: usize, engine: usize) -> u32 {
    kv_page_count(pid, model, engine, |kv, ws| {
        kv.held_page_count(ws).unwrap_or(0)
    })
}

fn kv_page_count(
    pid: ProcessId,
    model: usize,
    engine: usize,
    count: impl FnOnce(
        &crate::store::kv::KvStore,
        &std::collections::HashSet<crate::store::kv::page_table::WorkingSetId>,
    ) -> usize,
) -> u32 {
    let working_sets = crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine);
    if working_sets.is_empty() {
        return 0;
    }
    let Some(stores) = crate::store::registry::try_get(model, engine) else {
        return 0;
    };
    crate::store::registry::with_kv_lock(&stores.kv, "planner-held-pages", |kv| {
        count(kv, &working_sets) as u32
    })
}

// === Registry-owned instances, per (model, engine) ===

type PlannerMap = HashMap<(usize, usize), Arc<ResidencyPlanner>>;

static PRIMARY: OnceLock<Arc<ResidencyPlanner>> = OnceLock::new();
static PLANNERS: OnceLock<RwLock<PlannerMap>> = OnceLock::new();

/// Install the planner for `(model, engine)` (bootstrap, once per pair).
pub fn init_planner(model: usize, engine: usize, planner: ResidencyPlanner) {
    let planner = Arc::new(planner);
    // The drain runs on one dedicated task from here on; a planner
    // constructed outside a runtime keeps inline planning.
    planner.arm_drain_task();
    if model == 0 && engine == 0 {
        let _ = PRIMARY.set(planner.clone());
    }
    PLANNERS
        .get_or_init(Default::default)
        .write()
        .unwrap()
        .insert((model, engine), planner);
}

/// The planner for `(model, engine)`.
pub fn planner_for(model: usize, engine: usize) -> Option<Arc<ResidencyPlanner>> {
    if model == 0 && engine == 0 {
        return PRIMARY.get().cloned();
    }
    PLANNERS
        .get()?
        .read()
        .unwrap()
        .get(&(model, engine))
        .cloned()
}

/// The (0, 0) planner — the hot-path shorthand while process-side call sites
/// are still hardwired to engine 0. `None` only before bootstrap has run.
pub fn planner() -> Option<&'static Arc<ResidencyPlanner>> {
    PRIMARY.get()
}

// Service order across the allocation/restore boundary.
#[cfg(test)]
mod service_order_tests {
    use super::*;
    use crate::planner::grant::Demand;

    /// Build an `Inner` holding one process per `(seq, residency, admitted)`
    /// triple, returning the pids in the order given.
    fn fleet(spec: &[(u64, Residency, bool)]) -> (Inner, Vec<ProcessId>) {
        let mut inner = Inner::default();
        let mut pids = Vec::new();
        for &(seq, state, admitted) in spec {
            let pid = ProcessId::new_v4();
            let mut proc = Proc::new(seq);
            proc.state = state;
            proc.admitted = admitted;
            inner.procs.insert(pid, proc);
            pids.push(pid);
        }
        (inner, pids)
    }

    fn park_allocation(inner: &mut Inner, pid: ProcessId, seq: u64, kv_pages: u32) {
        inner.queue.insert(
            (seq, seq),
            Waiter {
                pid,
                kind: WaitKind::Allocation {
                    demand: Demand {
                        kv_pages,
                        rs_slots: 0,
                    },
                    notify: Arc::new(Notify::new()),
                    outcome: None,
                    yielded: false,
                },
            },
        );
    }

    fn park_restore(inner: &mut Inner, pid: ProcessId, seq: u64, demand: u32) {
        inner.queue.insert(
            (seq, seq),
            Waiter {
                pid,
                kind: WaitKind::Restore { demand },
            },
        );
    }

    /// The ping-pong regression: an evictee carries an old spawn seq back into
    /// the queue, ahead of the residents whose eviction paid for those pages.
    #[test]
    fn a_restore_yields_the_head_to_a_younger_allocation() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Evicted, true),
            // Still running, so the fleet has not stalled.
            (2, Residency::Resident, true),
            (3, Residency::Resident, true),
        ]);
        park_restore(&mut inner, pids[0], 1, 18);
        park_allocation(&mut inner, pids[2], 3, 1);

        let (key, waiter) = inner.unmet_head().expect("a head");
        assert_eq!(key.0, 3, "the younger allocation must take the head");
        assert!(matches!(waiter.kind, WaitKind::Allocation { .. }));
    }

    /// The safety valve. Yielding forever would deadlock a fleet whose residents
    /// are all parked on something an evictee owns.
    #[test]
    fn a_restore_takes_the_head_once_the_fleet_has_stalled() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Evicted, true),
            (3, Residency::Resident, true),
        ]);
        park_restore(&mut inner, pids[0], 1, 18);
        park_allocation(&mut inner, pids[1], 3, 1);

        assert!(inner.fleet_stalled(), "every admitted resident is parked");
        let (key, waiter) = inner.unmet_head().expect("a head");
        assert_eq!(key.0, 1, "the older restore must take the head");
        assert!(matches!(waiter.kind, WaitKind::Restore { .. }));
    }

    /// Nothing to yield TO: a served-but-uncollected grant has stopped
    /// competing for pages.
    #[test]
    fn a_restore_holds_the_head_when_no_allocation_is_unmet() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Evicted, true),
            (3, Residency::Resident, true),
        ]);
        park_restore(&mut inner, pids[0], 1, 18);
        park_allocation(&mut inner, pids[1], 3, 1);
        let Some(Waiter {
            kind: WaitKind::Allocation { outcome, .. },
            ..
        }) = inner.queue.get_mut(&(3, 3))
        else {
            unreachable!("just parked an allocation");
        };
        *outcome = Some(Err(PlannerError::Cancelled));

        assert!(!inner.fleet_stalled(), "a served grant is uncollected");
        let (key, waiter) = inner.unmet_head().expect("a head");
        assert_eq!(key.0, 1, "the restore is the only unmet entry");
        assert!(matches!(waiter.kind, WaitKind::Restore { .. }));
    }

    /// The yield is one-directional: a restore that is genuinely younger
    /// than the head keeps queueing behind it, exactly as FCFS says.
    #[test]
    fn a_younger_restore_still_queues_behind_an_older_allocation() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Resident, true),
            (2, Residency::Resident, true),
            (3, Residency::Evicted, true),
        ]);
        park_allocation(&mut inner, pids[0], 1, 4);
        park_restore(&mut inner, pids[2], 3, 18);

        let (key, waiter) = inner.unmet_head().expect("a head");
        assert_eq!(key.0, 1);
        assert!(matches!(waiter.kind, WaitKind::Allocation { .. }));
    }

    /// A live signal, not a latch: once host room returns the blocked set is
    /// cleared and the yield resumes.
    #[test]
    fn returning_host_room_resumes_the_yield() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Evicted, true),
            (2, Residency::Resident, true),
            (3, Residency::Resident, true),
        ]);
        park_restore(&mut inner, pids[0], 1, 18);
        park_allocation(&mut inner, pids[2], 3, 1);
        inner.host_swap_blocked.insert(pids[1]);
        assert_eq!(inner.unmet_head().expect("a head").0.0, 1);

        inner.host_swap_blocked.clear();

        assert!(!inner.eviction_unfundable());
        assert_eq!(
            inner.unmet_head().expect("a head").0.0,
            3,
            "the allocation takes the head again"
        );
    }
}

#[cfg(test)]
mod starvation_race_tests {
    use super::*;
    use crate::planner::grant::Demand;
    use crate::store::kv::page_table::PhysicalKvPageId;
    use std::sync::atomic::AtomicU32;

    /// A pool whose free list refills after the drain's absorb already came
    /// back empty — the production race, made deterministic.
    struct RacePool {
        free: parking_lot::Mutex<Vec<PhysicalKvPageId>>,
        total: u32,
        stall: AtomicU32,
        /// Host swap room still free. Zero keeps `plan_eviction` on its `NoSwapRoom`
        /// short-circuit; non-zero lets a test reach the load-control gate.
        host: u32,
        /// Every `reserve_device_up_to` request, in order — tells one pull for a
        /// whole burst apart from one pull per waiter.
        pulls: parking_lot::Mutex<Vec<u32>>,
        /// Host swap capacity. Kept apart from `host` so a test can put the
        /// pool under the reserve without emptying it.
        host_total: u32,
    }

    impl RacePool {
        fn new(total: u32) -> Self {
            Self {
                free: parking_lot::Mutex::new(Vec::new()),
                total,
                stall: AtomicU32::new(0),
                host: 0,
                pulls: parking_lot::Mutex::new(Vec::new()),
                host_total: 0,
            }
        }

        /// A pool that CAN swap, so `plan_eviction` runs past its
        /// no-transport short-circuit and reaches the load-control gate.
        fn with_swap(total: u32, host: u32) -> Self {
            Self {
                host,
                host_total: host,
                ..Self::new(total)
            }
        }

        /// Hand `n` pages back to the pool, but make the next `stall`
        /// reservations still see it empty.
        fn refill_after_stall(&self, n: u32, stall: u32) {
            self.stall.store(stall, Ordering::SeqCst);
            let mut free = self.free.lock();
            for i in 0..n {
                free.push(PhysicalKvPageId(i));
            }
        }
    }

    impl PoolPort for RacePool {
        fn device_stats(&self) -> (u32, u32) {
            (self.free.lock().len() as u32, self.total)
        }
        fn host_stats(&self) -> (u32, u32) {
            (self.host, self.host_total)
        }
        fn rs_stats(&self) -> (u32, u32) {
            (0, 0)
        }
        fn reclaim_idle(&self) -> u32 {
            0
        }
        fn suspend_capable(&self) -> bool {
            // Without host room: no swap transport, so `plan_eviction` goes straight
            // to `check_starvation(NoSwapRoom)`, skipping `last_resort_evict`.
            self.host_total > 0
        }
        fn locus(&self) -> (usize, usize) {
            (0, 0)
        }
        fn reserve_device(&self, count: u32) -> Option<Vec<PhysicalKvPageId>> {
            let mut free = self.free.lock();
            if (free.len() as u32) < count {
                return None;
            }
            let at = free.len() - count as usize;
            Some(free.split_off(at))
        }
        fn reserve_device_up_to(&self, count: u32) -> Vec<PhysicalKvPageId> {
            self.pulls.lock().push(count);
            if self
                .stall
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
            {
                return Vec::new();
            }
            let mut free = self.free.lock();
            let take = (free.len() as u32).min(count) as usize;
            let at = free.len() - take;
            free.split_off(at)
        }
        fn release_device(&self, pages: Vec<PhysicalKvPageId>) {
            self.free.lock().extend(pages);
        }
        fn reserve_rs(&self, _count: u32) -> Option<Vec<crate::store::rs::RsSlotId>> {
            Some(Vec::new())
        }
        fn release_rs(&self, _slots: Vec<crate::store::rs::RsSlotId>) {}
    }

    /// A completion's unpark burst is served in one pass: the drain funds the
    /// whole fundable FCFS prefix under a single lock hold.
    #[test]
    fn a_burst_serves_exactly_the_fundable_prefix_in_one_pass() {
        let pool = Arc::new(RacePool::new(64));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));

        // A running holder keeps the wedge predicate false, so the rungs
        // below the drain stay out of this while the pool is empty.
        let holder = ProcessId::new_v4();
        planner.register(holder);
        planner.note_admitted(holder);

        // Six one-page asks parked in FCFS order. Parked directly: the point
        // here is the drain's service order, not the acquire path.
        let demand = Demand {
            kv_pages: 1,
            rs_slots: 0,
        };
        let pids: Vec<ProcessId> = (0..6)
            .map(|_| {
                let pid = ProcessId::new_v4();
                planner.register(pid);
                planner.note_admitted(pid);
                pid
            })
            .collect();
        let keys: Vec<EntryKey> = planner.with_inner(|inner| {
            pids.iter()
                .map(|&pid| {
                    let key = (inner.procs[&pid].seq, inner.next_id);
                    inner.next_id += 1;
                    inner.queue.insert(
                        key,
                        Waiter {
                            pid,
                            kind: WaitKind::Allocation {
                                demand,
                                notify: Arc::new(Notify::new()),
                                outcome: None,
                                yielded: false,
                            },
                        },
                    );
                    key
                })
                .collect()
        });

        // The third ask's owner was picked for eviction while parked: it is
        // no longer a claimant, and the burst must hand it no pages.
        planner.with_inner(|inner| {
            let waiter = inner.queue.get_mut(&keys[2]).expect("parked");
            let WaitKind::Allocation { yielded, .. } = &mut waiter.kind else {
                unreachable!()
            };
            *yielded = true;
        });

        // Four pages come back — a completion that covers four of the five
        // live asks.
        pool.refill_after_stall(4, 0);
        pool.pulls.lock().clear();
        planner.plan();

        let served: Vec<bool> = planner.with_inner(|inner| {
            keys.iter()
                .map(
                    |key| match &inner.queue.get(key).expect("still queued").kind {
                        WaitKind::Allocation { outcome, .. } => outcome.is_some(),
                        WaitKind::Restore { .. } => unreachable!(),
                    },
                )
                .collect()
        });
        assert_eq!(
            served,
            vec![true, true, false, true, true, false],
            "exactly the fundable FCFS prefix is served, the yielded entry is \
             passed over rather than funded, and the ask past the supply waits"
        );
        assert_eq!(
            planner.diagnostics().serves_total,
            4,
            "one grant per served waiter"
        );
        // The served set above is what the two paths agree on — the burst is a
        // pure latency fix.
        let pulls = pool.pulls.lock().clone();
        assert_eq!(
            pulls,
            vec![1, 4, 1],
            "the head is covered by the ordinary absorb rung (1), the REST of \
             the run is asked for in a SINGLE pull (4 — the four live asks \
             behind the head), and the one ask the supply did not reach \
             re-enters the absorb rung (1). The pre-burst drain took one \
             absorb round trip per served waiter (vec![1, 1, 1, 1, 1]), which \
             is the serialization this removes"
        );
        assert_eq!(
            planner.diagnostics().device_pages_free,
            0,
            "the burst consumed what it pulled"
        );
        assert_eq!(
            planner.diagnostics().accumulation,
            0,
            "and stranded nothing in the accumulation"
        );
    }

    /// Regression: the host-swap eviction livelock. `HostSwapFull` leaves the
    /// victim resident, and the deterministic scan would re-pick it forever.
    #[tokio::test(flavor = "current_thread")]
    async fn a_host_swap_full_victim_is_parked_until_host_room_returns() {
        // Total 4, free 0: the victim owns the pool, so the head's ask is
        // fundable in principle (not `Impossible`) but parks.
        let pool = Arc::new(RacePool::new(4));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone()));

        // A younger admitted resident is the only legal victim for the head.
        let head = ProcessId::new_v4();
        let victim = ProcessId::new_v4();
        for pid in [head, victim] {
            planner.register(pid);
            planner.note_admitted(pid);
        }

        let p = planner.clone();
        let head_task = tokio::spawn(async move {
            p.acquire(
                head,
                head,
                Demand {
                    kv_pages: 1,
                    rs_slots: 0,
                },
            )
            .await
        });
        for _ in 0..200 {
            tokio::task::yield_now().await;
            if planner.diagnostics().queue.len() == 1 {
                break;
            }
        }
        assert_eq!(planner.diagnostics().queue.len(), 1, "the head must park");

        let members = |planner: &Arc<ResidencyPlanner>| {
            planner
                .victim_set()
                .map(|set| set.members.iter().map(|v| v.pid).collect::<Vec<_>>())
                .unwrap_or_default()
        };
        assert_eq!(
            members(&planner),
            vec![victim],
            "the younger resident is the head's victim"
        );

        // The eviction rolls back on host swap: re-picking it is the spin.
        planner.eviction_failed_host_swap_full(victim);
        assert!(
            members(&planner).is_empty(),
            "a host-swap-blocked victim must not be re-picked"
        );
        assert_eq!(planner.diagnostics().host_swap_exhaustions_total, 1);

        // Host room returns with a restore: the victim is a candidate again,
        // so the block can never strand a legal eviction.
        planner.clear_host_swap_blocks();
        assert_eq!(
            members(&planner),
            vec![victim],
            "returned host room must re-arm the parked victim"
        );
        assert_eq!(planner.diagnostics().host_swap_unblocks_total, 1);

        head_task.abort();
    }

    /// Load control: an exhausted pool with no transfer in flight evicts even
    /// though admitted processes are still running.
    #[tokio::test(flavor = "current_thread")]
    async fn an_exhausted_pool_evicts_even_while_the_fleet_still_runs() {
        // Non-zero capacity with an empty free list: an ask bigger than the
        // pool is failed loud instead of parking.
        let pool = Arc::new(RacePool::with_swap(4, 64));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone()));
        // Pin the runway latch closed so the demand-exact arithmetic stays
        // under test regardless of whether PIE_SUPPLY_RUNWAY is exported.
        planner.with_inner(|inner| inner.runway_round_in_flight = true);

        // An unparked admitted runner: relief is still on its way, so the
        // fleet is not wedged.
        let holder = ProcessId::new_v4();
        planner.register(holder);
        planner.note_admitted(holder);

        let head = ProcessId::new_v4();
        planner.register(head);
        planner.note_admitted(head);
        // Younger, resident, admitted: legal victims under the anti-thrash
        // rule, so a set exists and only the gate can hold the round back.
        for _ in 0..3 {
            let pid = ProcessId::new_v4();
            planner.register(pid);
            planner.note_admitted(pid);
        }

        let demand = Demand {
            kv_pages: 1,
            rs_slots: 0,
        };
        let p = planner.clone();
        let parked = tokio::spawn(async move { p.acquire(head, head, demand).await });
        for _ in 0..200 {
            tokio::task::yield_now().await;
            if planner.diagnostics().queue.len() == 1 {
                break;
            }
        }
        let d = planner.diagnostics();
        assert_eq!(
            d.queue.len(),
            1,
            "the ask must park (starved={} salvaged={} free={}/{})",
            d.starvations_total,
            d.salvages_total,
            d.device_pages_free,
            d.device_pages_total
        );

        // The victim set is non-empty and the deficit is exactly the head's
        // need: nothing about the shortage has changed.
        let set = planner.victim_set().expect("an unmet head yields a set");
        assert_eq!(set.deficit, 1, "the round is demand-exact");
        assert!(
            !set.members.is_empty(),
            "younger residents are legal victims"
        );

        // `holder` is unparked, so the fleet is not wedged, yet the rung must run
        // anyway: nothing in flight cannot produce a page by waiting.
        assert!(!planner.is_wedged(), "a running holder is not a wedge");
        assert!(planner.supply_stalled(), "an empty pool with an unmet head");
        assert_eq!(
            planner.diagnostics().eviction_deferrals_total,
            0,
            "the rung must not have deferred"
        );

        // The two ways relief IS already on its way, each shutting the gate
        // on its own: a victim in flight, and a kill ordered.
        let mark = ProcessId::new_v4();
        planner.with_inner(|inner| {
            inner.evicting.insert(mark, EvictionMark { pages: 1 });
        });
        assert!(
            !planner.supply_stalled(),
            "a victim already in flight must hold the rung shut"
        );
        planner.with_inner(|inner| {
            inner.evicting.remove(&mark);
        });
        assert!(planner.supply_stalled(), "and open again once it lands");

        parked.abort();
    }

}

