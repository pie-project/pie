//! Project Rainer — the residency planner (design: `rainer.md`).
//!
//! One decision replaces the retired preempt/restore negotiation (the
//! `store::reclaim` ladder): **membership**. A process is either RESIDENT —
//! its fires reserve straight off the pool free lists with no planner
//! involvement at all — or it is out of the resident set: structurally
//! quiescent, its KV working sets on host swap, its guest task parked at the
//! residency gate. Membership changes are planner-owned and planner-executed;
//! nothing is negotiated with the victim.
//!
//! - **FCFS by spawn.** Registration order is the single priority key. The
//!   oldest unmet ask — a parked allocation or an evictee's restore — is the
//!   HEAD; every freed page is physically pulled into the planner's
//!   accumulation until the head's demand is covered. Younger asks wait
//!   behind it, so drainage is monotone and victim/restore thrash cannot
//!   oscillate: a victim is younger than the head it funded, and its restore
//!   entry queues behind every older ask.
//! - **The hot path pays nothing.** With no waiters and everyone resident
//!   (two relaxed atomic loads), `acquire` is two free-list pops. There is
//!   no global grant mutex, no queue bookkeeping, no notify arming — the
//!   per-fire toll that cost −17% roomy throughput (CONTENTION_FOLLOWUP.md
//!   §13) is deleted structurally.
//! - **Eviction is sized and aimed, never negotiated** (`planner::exec`):
//!   fence the victim's working sets (an atomic paired with the fire lease),
//!   drain its detachable tail, wait for lease quiescence, then
//!   prepare → D2H → commit on the planner's own task. No safe points, no
//!   park requests, no victim cooperation, no per-victim state machine.
//! - **Restores are accumulation-gated.** An evicted process re-queues at
//!   its spawn position; the oldest evictee boards alone when the
//!   accumulation covers its swapped set. A thundering herd is
//!   unrepresentable: evictions are deficit-sized from the youngest edge and
//!   restores are funded one head at a time.
//! - **No timers, no knobs.** Exhaustion is arithmetic — an ask larger than
//!   the pool fails loud ([`PlannerError::Impossible`]), a head whose own
//!   holdings plus its ask exceed the pool is the hog endgame
//!   ([`PlannerError::Hog`]). There is no progress deadline and no kill rung.
//!
//! Lock discipline: `inner` is a plain mutex ordered INSIDE everything else —
//! no port/store call is made and no reservation is dropped while holding
//! it. Every mutation funnels through `with_inner`, which recomputes the two
//! lock-free hot-path counters (`waiters`, `nonresident`) on the way out.

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

/// Opt-in event markers (`PIE_CONTENTION_TRACE_EVENTS=1`): `println!`, not
/// `tracing` — the embedded (pyo3) server installs no subscriber. The
/// timestamp shares the fire-timing monotonic clock, which is what lets these
/// markers be read against the scheduler's own timing records.
///
/// **This is a separate switch from the periodic stall sampler**
/// (`PIE_CONTENTION_TRACE_MS`) on purpose. The markers fire per planner
/// EVENT — one line per park, serve, restore and eviction step — and a
/// contended run emits tens of thousands of them. Tying them to the
/// sampler's variable made `PIE_CONTENTION_TRACE_MS=0`, the natural
/// spelling of "off", the single most expensive setting there is: the
/// sampler thread never starts (it needs `ms > 0`) but every marker prints.
/// Runs taken that way read 25-30% under the same build untraced, which is
/// larger than most effects this planner is tuned against.
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



/// Minimal physical port the planner drives — pool stats, rung-0 idle
/// reclaim, and concrete reservations. The planner owns ALL policy; this
/// port owns only physics. One impl ([`RegistryPool`]); the contention
/// integration tests drive it through the real store.
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
    /// Whether the driver can physically move KV bytes to/from host swap.
    /// Arms eviction; without it the planner is pool-only (a capability
    /// degradation, not a mode).
    fn suspend_capable(&self) -> bool;
    /// The `(model, driver)` pair this pool belongs to.
    fn locus(&self) -> (usize, usize);
    fn reserve_device(
        &self,
        count: u32,
    ) -> Option<Vec<crate::store::kv::page_table::PhysicalKvPageId>>;
    /// Pop up to `count` free device pages under ONE store-lock hold — the
    /// drain's absorb step. (A stats read followed by a sized reserve would
    /// be two holds of the contended lock per absorb, per poke.)
    fn reserve_device_up_to(
        &self,
        count: u32,
    ) -> Vec<crate::store::kv::page_table::PhysicalKvPageId>;
    fn release_device(&self, pages: Vec<crate::store::kv::page_table::PhysicalKvPageId>);
    fn reserve_rs(&self, count: u32) -> Option<Vec<crate::store::rs::RsSlotId>>;
    fn release_rs(&self, slots: Vec<crate::store::rs::RsSlotId>);
}

/// Production [`PoolPort`] over the typed per-(model, driver) stores.
pub struct RegistryPool {
    model: usize,
    driver: usize,
    suspend_capable: bool,
}

impl RegistryPool {
    pub fn new(model: usize, driver: usize, suspend_capable: bool) -> Self {
        Self {
            model,
            driver,
            suspend_capable,
        }
    }

    fn with_kv<R>(&self, operation: impl FnOnce(&mut crate::store::kv::KvStore) -> R) -> R {
        self.with_kv_tagged("planner", operation)
    }

    fn with_kv_tagged<R>(
        &self,
        tag: &'static str,
        operation: impl FnOnce(&mut crate::store::kv::KvStore) -> R,
    ) -> R {
        let stores = crate::store::registry::get(self.model, self.driver);
        crate::store::registry::with_kv_lock(&stores.kv, tag, operation)
    }

    fn with_rs<R>(&self, operation: impl FnOnce(&mut crate::store::rs::RsStore) -> R) -> R {
        let stores = crate::store::registry::get(self.model, self.driver);
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
        (self.model, self.driver)
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
    /// set). The fire path settles its own pipeline tail — the ONLY task
    /// that can finalize device-geometry ops is the guest's own, since they
    /// need its ResourceTable — then waits out the eviction and re-asks.
    /// This is the one safe point Rainer keeps: the ask holds no lease, no
    /// pins, and no open transaction here.
    Yield,
}

/// Why eviction could not fund the head at the instant the starvation
/// predicate was evaluated. Carried into [`PlannerError::Starved`] so the
/// message names the real wedge: the two callers fail for unrelated
/// reasons, and reporting resource exhaustion for a policy wedge sends the
/// reader hunting a full swap pool that is in fact ~99% free
/// (CONTENTION_FOLLOWUP.md §18.7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarveCause {
    /// The driver advertises no KV swap transport, or the host swap pool has
    /// no free slot to evict into.
    NoSwapRoom,
    /// Swap room exists, but no candidate was eligible or had pages to give.
    /// Eligibility is FCFS-anti-thrash: only processes YOUNGER than the head
    /// may be evicted for it. When the pages are all held by processes OLDER
    /// than the head, the planner has no legal move.
    NoEligibleVictim,
    /// Every RS folded slot is held and no admitted process is left running to
    /// return one. Eviction is not a rung here: it frees KV pages, never
    /// folded slots, so nothing the planner can do will fund this ask.
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
    /// pool, so no amount of evicting OTHERS can ever cover it. Fail loud.
    Hog { need: u32, held: u32, total: u32 },
    /// The starvation endgame, computed — not timed: the head is unmet,
    /// eviction cannot fund it (see [`StarveCause`]), no transfer is in
    /// flight, and not one fire lease is held anywhere — no completion can
    /// ever arrive. The youngest parked ask is failed loud so its frees
    /// restart the fleet (destruction youngest-first, never the head).
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

/// Opt-in planner-mutex census (`PIE_PLANNER_LOCK_TRACE=1`). The planner lock
/// is global and the contended hot path takes it on EVERY fire (`note_progress`
/// once `waiters != 0`), so "is this a convoy?" has to be answered by
/// measurement, not by reading the call graph.
struct LockCensus {
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
    /// Shortages that did NOT trigger an eviction because the fleet was
    /// still making progress — the load-control rung. Every one of these is
    /// a fence/quiesce/D2H/H2D round trip not paid.
    pub eviction_deferrals: AtomicU64,
    /// §10.14: eviction rounds the supply runway motivated — started with
    /// an empty ask queue, or enlarged past the queued demand. Zero unless
    /// `PIE_SUPPLY_RUNWAY` is set.
    pub runway_rounds: AtomicU64,
    /// Pages those rounds ordered on the runway's account (the shortfall
    /// component, not the queued-demand component).
    pub runway_pages: AtomicU64,
    /// Evictions that committed (working sets moved to host swap).
    pub evictions: AtomicU64,
    /// Evictions abandoned before commit (nothing reclaimable, non-detachable
    /// tail, transport failure, host swap full). The planner re-plans.
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
    /// Nonzero means the fleet reached the state that used to lose requests
    /// at high oversubscription.
    pub salvages: AtomicU64,
    /// Asks served out of FCFS order by the last rung before destruction,
    /// because the head's own hoard covered them and the head itself was
    /// uncoverable. Nonzero means the fleet reached the head-of-line wedge
    /// that used to destroy a request over a fundable pool (§20.17).
    pub hoard_bypasses: AtomicU64,
    /// Evictions that had to relax the E6 post-restore hysteresis because
    /// no normally-eligible victim could fund the head. Nonzero means the
    /// fleet reached the state that used to destroy a request (§18.7).
    pub e6_relaxations: AtomicU64,
    /// Restore heads that found the pool dry and declined to evict for
    /// themselves. Every one of these is an evict→restore→evict ping-pong
    /// round trip not paid; the readmission waits for a completion instead.
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
    /// RS folded slots this ask needs. A head parked with pages available is
    /// blocked on this instead, and reading only `pages` hid that.
    pub rs_slots: u32,
}

/// How many unparked admitted processes [`PlannerDiagnostics::runners`]
/// reports. A wedge needs the first few; a healthy fleet needs none.
const RUNNER_DUMP_CAP: usize = 24;

/// The reciprocal of the host-pool slice the eviction rung may not spend on
/// a fleet that is still running — see the HOST RESERVE note in
/// [`ResidencyPlanner::plan_eviction`]. It only has to be large enough that
/// a live fleet's evictees cannot squeeze the pool to the `host_free == 0`
/// kill arm, and small enough that it never binds where host room is
/// plentiful relative to the device pool.
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
    /// difference against the registered total is the cohort that is queued
    /// for a permit: page-less, unable to run, and therefore irrelevant to
    /// the wedge predicate. Reading `resident` without this was what made
    /// the admission-cap wedge invisible in a trace.
    pub admitted_procs: u32,
    /// The admitted processes that are NOT parked in the queue — the only
    /// cohort that can still make progress, and therefore the only cohort
    /// that can silently disarm the starvation rung. `(seq, held_pages,
    /// progressed)`, capped: a wedge is diagnosed from a handful of these,
    /// and the aggregate counters above cannot show them at all.
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
    /// Entries BEHIND the blocked head that the currently-free stock could
    /// cover on its own. The drain is head-first, so these wait even though
    /// the resource to serve them exists — the head-of-line cost.
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
    /// §10.14 supply-runway probes — zero unless `PIE_SUPPLY_RUNWAY`.
    pub runway_rounds_total: u64,
    pub runway_pages_total: u64,
}

/// One process the FCFS anti-thrash rule permits evicting for the current
/// head. `e6_fresh` is POLICY (see [`VictimSet`]), never eligibility.
#[derive(Clone, Copy)]
struct Victim {
    pid: ProcessId,
    seq: u64,
    /// False while the process is inside its post-restore E6 hysteresis
    /// window. A reason to *prefer* someone else, never a reason to
    /// consider this process un-evictable.
    e6_fresh: bool,
}

/// The legal victims for one head on the ROUTINE eviction path, from one
/// consistent snapshot.
///
/// The split this type exists to enforce (`rainer_v3.md` §3.2): membership
/// is the FCFS anti-thrash rule and nothing else — younger than the head,
/// resident. E6 hysteresis rides as a per-member tag and narrows only
/// [`Self::preferred`], the *policy* view. It can never remove a member,
/// so it can never make the fleet look wedged when it is not; that is the
/// property whose absence let a hysteresis rule destroy requests (§18.9).
///
/// The liveness question — "does a legal victim exist at all?" — is NOT
/// answered here. [`ResidencyPlanner::last_resort_evict`] recomputes the
/// same membership rule fused with quoting inside one atomic
/// store+planner snapshot, because a set built here and quoted after the
/// lock is released is exactly the stale snapshot that destroyed requests
/// twice (§18.10, `rainer_v3.md` §8.3).
struct VictimSet {
    head: EntryKey,
    head_pid: ProcessId,
    deficit: u32,
    members: Vec<Victim>,
    /// §10.14: how many of `deficit`'s pages are the supply runway's
    /// shortfall rather than queued demand. Nonzero only with
    /// `PIE_SUPPLY_RUNWAY` set and no runway round already in flight;
    /// committing such a round latches the hysteresis flag.
    runway_grab: u32,
}

/// Where a registered process stands. Two real states plus the two
/// transfer-in-flight transients — membership, not negotiation.

impl VictimSet {
    /// Members E6 hysteresis permits evicting right now — the routine
    /// path. May be empty while the set is not; callers must therefore
    /// never read this as "no victim exists".
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
    /// on every successful restore, so the allowance is per episode and not
    /// per lifetime.
    restore_retried: bool,
    /// E6 — progress before re-eviction. `false` from restore commit until
    /// this process's next `acquire` call (any outcome — every fire passes
    /// through `acquire`, and a restored guest always re-asks: it was parked
    /// at `wait_resident` or the residency gate). While `false` the process
    /// is not an eviction candidate: without this, an elder's ask landing
    /// right after a restore re-evicts the same pages before a single fire
    /// runs — the measured evict⇄restore ping-pong that burned 3.9 s of H2D
    /// in one 40 s window (CONTENTION_FOLLOWUP.md §14.4). Event-lifted,
    /// never timed: a parked process has already cleared it, so the
    /// starvation predicate is unaffected.
    progressed: bool,
    /// Whether this process has claimed an EXECUTION slot
    /// (`inferlet::process::ensure_execution_admitted`).
    ///
    /// Registration happens at spawn, because registration order IS the FCFS
    /// clock. But a process only reaches the pooled resources after the
    /// execution gate: "everything that creates per-instance driver state or
    /// claims pooled KV/RS resources waits for the execution permit"
    /// (`inferlet/process.rs`). So a registered-but-unadmitted process holds
    /// exactly zero device pages, and its existence can never be the reason
    /// a wedged pool will unwedge — see [`Planner::is_wedged`].
    admitted: bool,
    /// Wakes residency-gate waiters and out-of-set acquire loops on any
    /// state change.
    signal: Arc<Notify>,
    /// Lock-free mirror of `state == Resident`, refreshed by `with_inner`
    /// on the same pass that recomputes `nonresident`.
    ///
    /// The residency gate runs in the prologue of EVERY WIT host method
    /// that can touch pooled state. Its published fast path is
    /// `gate_open()` — "nobody at all is evicted" — which is a
    /// true statement about an idle fleet and a false one about every
    /// contended fleet: with any process out of the set the gate fell
    /// through to `is_resident`, which took the planner MUTEX once per
    /// host call. That made the global planner lock a per-call
    /// serialization point exactly when contention was highest. Handing
    /// each process its own flag makes the gate one relaxed load with no
    /// lookup and no lock, whatever the rest of the fleet is doing.
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
        /// Set when the owner was chosen for eviction while parked: the
        /// collector yields back to the fire path (which settles the tail),
        /// instead of waiting for pages that would bounce off the fence.
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
    /// grant no longer competes for pages).
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
    /// Head-first accumulation: pages physically pulled OUT of the free list
    /// toward the current head's demand. Planner-level (not per-entry), so a
    /// head change strands nothing; released only when the queue empties.
    accum: DevicePageReservation,
    /// Evictions in flight → what each is expected to free. Subtracted from
    /// the deficit so concurrent plans never over-evict.
    evicting: HashMap<ProcessId, EvictionMark>,
    /// Rotation damping (Rule A as a stock): resident completions banked,
    /// one per departure, debited one per served restore. Restores wait for
    /// this credit unless the fleet is stalled, so the rotation rate is
    /// bounded by the completion rate instead of riding eviction surplus.
    completion_credit: u32,
    /// Victims whose last eviction rolled back on `HostSwapFull`. Re-picking
    /// one before host room changes is pure spin: victim selection is
    /// deterministic, so the retry re-runs the whole fence → suspend →
    /// drain → quiesce → prepare cycle and fails identically, forever
    /// (§20.6). Cleared wholesale the moment host slots are actually
    /// returned — see `clear_host_swap_blocks`.
    host_swap_blocked: HashSet<ProcessId>,
    /// Victims whose last eviction rolled back because `prepare_suspend`
    /// deferred (nothing of theirs is movable right now: pinned leases, an
    /// in-flight fire, shared-only pages). Exactly the same spin as
    /// `host_swap_blocked` — deterministic re-pick, identical failure — and
    /// it is a hot one: a wedged fleet logged 674k rollbacks in 60 s, and
    /// because `evicting` is never empty while that runs, EVERY rung above
    /// the kill is gated out and the pool can never be unwedged. Cleared
    /// whenever a process retires or host room returns, i.e. whenever the
    /// answer could actually have changed.
    prepare_blocked: HashSet<ProcessId>,
    /// Destructions ordered but not yet paid out — see [`Inner::kill_in_flight`].
    killing: HashSet<ProcessId>,
    /// §10.14 runway hysteresis: a runway-motivated eviction round is in
    /// flight. At most one at a time — the runway's contribution must never
    /// pyramid the way the first whole-queue supply cut did in tiny-pool
    /// regimes (onefits: 58 starvation kills). Set when a round the runway
    /// started or enlarged commits; cleared by [`Inner::settle_eviction`]
    /// once nothing is in flight at all. Always `false` while
    /// `PIE_SUPPLY_RUNWAY` is unset, and never read then.
    runway_round_in_flight: bool,
}

impl Inner {
    /// A starvation kill has been ordered whose pages have not come back yet.
    ///
    /// `evicting` covers reclaim by *transfer*; this covers reclaim by
    /// *destruction*, and the two must gate identically — neither is a reason
    /// to start another. The window is not the one the `outcome: Some(_)`
    /// check already covers: setting the victim's outcome only wakes it. Its
    /// pages return when its store drops, which is a WASM unwind later —
    /// milliseconds, against the microseconds its waiter takes to collect the
    /// error and vacate the queue. In between, `free` is still 0 and nothing
    /// marks a reclaim as under way, so every planner poke ordered another
    /// kill. Measured on D/512 under the supply-stall rule: **250 kills/s
    /// sustained, 3590 total for 768 requests, and a resident set that ended
    /// exactly where it started (541)** — the destructions raced each other
    /// instead of the deficit.
    ///
    /// Self-clearing on both exits, so it cannot wedge the deadlock breaker
    /// it guards: the victim either dies (leaves `procs`) or survives the
    /// error and re-contends (a fresh unmet ask), and either way stops
    /// counting. `unregister` prunes the set so it stays bounded.
    fn kill_in_flight(&self) -> bool {
        self.killing.iter().any(|pid| {
            self.procs.contains_key(pid)
                && !self
                    .queue
                    .values()
                    .any(|waiter| waiter.pid == *pid && waiter.is_unmet())
        })
    }

    /// Retire one eviction from the in-flight set (landed, rolled back, or
    /// the victim unregistered) and release the §10.14 runway latch once
    /// nothing at all is in flight: the runway may motivate its next round
    /// only after the previous round's victims have all settled. Every
    /// `evicting` removal goes through here so the latch can never wedge.
    fn settle_eviction(&mut self, pid: ProcessId) {
        self.evicting.remove(&pid);
        if self.evicting.is_empty() {
            self.runway_round_in_flight = false;
        }
    }

    /// §10.14: the seq floor for a runway round's victims. The OLDEST
    /// admitted resident — the runway must never evict the last resident,
    /// because with nobody resident nothing can complete and the pool loses
    /// its only organic supply — raised to the FCFS head's seq when one is
    /// queued (the anti-thrash rule: victims are younger than the head).
    /// `None` when nobody admitted is resident, which is also "no round".
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

    /// Count non-resident processes AND refresh each process's lock-free
    /// residency mirror, in one pass. Both are derived facts recomputed at
    /// every lock release rather than incrementally mirrored, so no
    /// `state` write site can forget to maintain them.
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

    /// The FCFS head: the oldest UNMET entry — the single service-order
    /// invariant, looked up here and nowhere else. A served-but-uncollected
    /// grant no longer competes for pages (its collector removes it
    /// promptly), so the next ask behind it drains meanwhile.
    ///
    /// A restore YIELDS to every unmet allocation. Spawn order alone is the
    /// wrong key across that boundary: an allocation comes from a resident
    /// process that already holds pages and needs one more step, and only a
    /// resident can complete — a completion is the single event that
    /// returns a whole working set to the pool. A restore instead adds a
    /// claimant to a pool the resident set cannot already cover, and it
    /// arrives holding an OLD spawn seq, so pure FCFS hands it the head
    /// ahead of the residents whose eviction paid for the pages.
    ///
    /// The module header argued thrash was unrepresentable because "a
    /// victim is younger than the head it funded, and its restore entry
    /// queues behind every older ask". That holds for one step and not for
    /// the loop: the older asks each want ONE page, they drain in
    /// microseconds, and then the evictee is the oldest entry left and
    /// takes its whole working set back. Measured on D/512 with the rung
    /// open: 993 evictions against 929 restores for a net of 64 processes
    /// out, `free` pinned at 0, and 314 residents holding 26 pages each
    /// against the 35 they each need to finish — 234 residents would have
    /// fit. The pool oscillated instead of draining.
    ///
    /// Yielding cannot deadlock on its own: an unmet allocation is funded
    /// by evicting residents, and evicting every resident leaves nothing
    /// but restores in the queue, which then board in spawn order. It is
    /// the fleet stalling for some OTHER reason — a resident parked on
    /// something an evictee owns — that the yield must not outlive, so
    /// [`Inner::fleet_stalled`] hands the head straight back to the oldest
    /// entry of any kind.
    ///
    /// That argument has a second precondition it did not name: eviction
    /// must be able to KEEP evicting. It is funded by the host pool, which
    /// is finite, and the only event that returns a host slot is a restore
    /// — the very thing being yielded. Once the pool is out, "evict every
    /// resident" is not reachable, the unmet allocations stay unmet, and
    /// the evictees holding the host pool can never come back to release
    /// it. `fleet_stalled` does not catch it, because residents ARE
    /// completing; the fleet is live and starving a resource at the same
    /// time. Measured on `soak` (160 device pages, 1024 host pages, 4096
    /// requests at 256-way): the rung drained host_free 1024 -> 0 in the
    /// first seconds and it never recovered — restores pinned at 1-8 and
    /// `evicted` at ~210 for the remaining 60 s while every further
    /// attempt rolled back on `HostSwapFull`, 21895 of them, and the
    /// starvation rung killed 138 of the stranded processes. Before the
    /// yield the same fleet ran 123 evictions, 0 rollbacks and 0 kills.
    /// So [`Inner::eviction_unfundable`] is the second valve.
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

    /// Eviction has run out of the resource that funds it. A victim parked
    /// in `host_swap_blocked` is proof: its bytes were refused by the host
    /// pool, and victim selection is deterministic, so nothing about that
    /// answer changes until host slots are actually returned. Only a
    /// restore returns them, which is why this gates the restore yield in
    /// [`Inner::unmet_head`] — preferring an allocation here cannot serve
    /// it, since the rung that would fund it is blocked on the same pool.
    ///
    /// Deliberately a live signal rather than a latch: `clear_host_swap_blocks`
    /// empties the set the moment host room returns, so the yield resumes as
    /// soon as eviction can pay for itself again.
    fn eviction_unfundable(&self) -> bool {
        !self.host_swap_blocked.is_empty()
    }

    /// The pages the burst behind the head still needs beyond the
    /// accumulation — the size of the ONE free-list pull
    /// [`Step::ServeAllocationBurst`] makes for the whole pass.
    ///
    /// The run is the maximal consecutive stretch of unmet, RS-free
    /// allocation waiters in FCFS order. Restores are transparent to it:
    /// a restore younger than the head allocation never outranks it, and an
    /// older one can only outrank it while [`Inner::fleet_stalled`] holds —
    /// which the burst's first serve falsifies (a served, uncollected grant
    /// is exactly what that predicate rules out). An RS-carrying ask ENDS
    /// the run: funding RS is a port call, and the port must never be
    /// touched under this lock.
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

    /// The burst's single lock hold: fund consecutive waiters head-first out
    /// of the accumulation, parking each outcome in its own entry, and hand
    /// back the notifies for the caller to fire AFTER the lock is released.
    ///
    /// This is [`Step::ServeAllocation`]'s body, run in a loop under one
    /// hold. Equivalent to re-deriving [`Inner::unmet_head`] per serve — the
    /// head is derived once here and the walk then follows queue order,
    /// which is the same sequence, because serving an entry is precisely
    /// what makes `unmet_head` move on to the next one. Nothing else can
    /// interleave: the whole pass happens under this hold.
    ///
    /// Stops at the first waiter the accumulation cannot cover in full (and
    /// at the first RS-carrying ask, which the per-step path serves).
    /// Strict FCFS — no skipping, no bypass.
    fn serve_burst(&mut self) -> Vec<(EntryKey, u32, Arc<Notify>)> {
        let mut wake = Vec::new();
        let Some((head, waiter)) = self.unmet_head() else {
            return wake;
        };
        // The head is re-derived here, after the pull released the lock, and
        // it must still be an allocation. A restore that took the head in
        // that window (the fleet went stalled) outranks everything behind
        // it — serving past it would be the one FCFS inversion this pass
        // could introduce. Serve nothing and let the drain re-decide.
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
    /// grant served and awaiting collection, and every admitted resident
    /// parked in the queue.
    ///
    /// The eviction rung's last-resort path and the starvation rung share
    /// this predicate through [`ResidencyPlanner::is_wedged`], so the two
    /// can never disagree about when waiting is still an option.
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
    ///
    /// `fund_by_eviction` is false when the head is a restore that the fleet
    /// is not stalled behind: an eviction must never pay for a readmission
    /// (see the eviction ladder in [`ResidencyPlanner::plan`]).
    Absorb {
        count: u32,
        fund_by_eviction: bool,
    },
    /// The head's KV side is covered; finish an allocation with `rs` slots.
    /// Only RS-carrying asks take this step — the RS reservation is a port
    /// call that must happen outside the planner lock, so they keep the
    /// per-step path. RS-free allocations take [`Step::ServeAllocationBurst`].
    ServeAllocation {
        key: EntryKey,
        demand: Demand,
    },
    /// The head is a covered, RS-free allocation: serve the maximal
    /// consecutive run of fundable RS-free allocation waiters in ONE pass —
    /// one free-list pull sized `extra` (the run's total ask beyond the
    /// accumulation), then one planner-lock hold that pops-and-funds the run
    /// head-first, parking each outcome; the notifies fire after release.
    ///
    /// This exists because the sequential drain served a completion's
    /// unpark burst nearly one waiter at a time: every serve paid three to
    /// four lock cycles (step decision, absorb, serve re-validation — each
    /// recomputing the lock-free mirrors over every proc) plus a port pull,
    /// interleaved against the just-woken collectors contending on the same
    /// mutex — measured intra-burst serve spacing p50 54µs / p90 321µs,
    /// ~4.3 ms wall for a ~32-waiter burst, and the wave seal idled the
    /// device for all of it. FCFS is untouched: the run is re-derived under
    /// the lock via `unmet_head` exactly as the sequential drain would, and
    /// it stops at the first waiter it cannot fund — no skipping, no bypass.
    ///
    /// One clarification, because the walk does step over entries: a RESTORE
    /// inside the run is passed rather than served, and that is not an
    /// inversion — [`Inner::unmet_head`] already ranks an unmet allocation
    /// ahead of any older restore unless [`Inner::fleet_stalled`] holds, and
    /// the burst's first serve (a served, uncollected grant) falsifies that
    /// predicate for the rest of the pass. The sequential drain reaches the
    /// same entries in the same order.
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
    // parking_lot (like the KV store lock): every contended acquire and
    // every gate check touches this; adaptive spinning keeps the wake-herd
    // from degenerating into a futex storm (§15).
    inner: parking_lot::Mutex<Inner>,
    /// Lock-free mirror of `queue.len()` — the acquire fast path's only
    /// planner touch. Readers may observe one transition late; the slow path
    /// re-checks under the lock.
    waiters: AtomicUsize,
    /// Lock-free mirror of the not-Resident process count — the residency
    /// gate's only planner touch.
    nonresident: AtomicUsize,
    /// E5 — the single-owner drain. When armed (bootstrap, inside the
    /// runtime), every production call site *pokes* this and ONE dedicated
    /// task runs [`Self::plan`]. Without it (pre-boot), pokes fall back to
    /// an inline `plan()`. Running the drain on guest tasks
    /// was the measured contended-throughput killer (§15): every finalize
    /// poke ran a full planning pass — idle-reclaim scans, per-candidate
    /// eviction quotes — under the global KV lock, serializing the whole
    /// fleet's turnaround at ~0.4 ms per guest.
    drain: OnceLock<Arc<Notify>>,
    /// Rung 0 exhaustion latch: set when `reclaim_idle` returns 0, cleared
    /// by real free events. Prevents a fruitless cache-lease scan on every
    /// poke while the pool sits at free=0 (the common contended state).
    idle_reclaim_exhausted: std::sync::atomic::AtomicBool,
    port: Arc<dyn PoolPort>,
    stats: PlannerStats,
}

/// Removes a parked allocation entry when its `acquire` future is dropped
/// (cancellation): the entry vanishes and any parked outcome's reservation
/// returns to the pool outside the lock.
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

/// What [`ResidencyPlanner::acquire_or_enqueue`] resolved to. `Granted` and
/// `NotResident` are [`Acquired::Granted`] / [`Acquired::Yield`] one-to-one;
/// `Ticket` is the parked case handed back instead of awaited, so the caller
/// (the deferred-allocation fire path, `PIE_DEFER_ALLOC=1`) can collect the
/// grant from its own task while the guest keeps running.
pub(crate) enum Enqueued {
    Granted(AllocationGrant),
    Ticket(AllocationTicket),
    /// The asking process is out of the resident set (or in transfer): the
    /// fire path must settle its own tail and wait out the eviction — same
    /// contract as [`Acquired::Yield`].
    NotResident,
}

/// A parked allocation ask, owned: the queue entry was inserted at its FCFS
/// position by [`ResidencyPlanner::acquire_or_enqueue`] (the park side
/// effects — census, `notify_lane_close`, poke — already happened), and
/// [`AllocationTicket::collect`] runs the exact notified/collect loop
/// [`ResidencyPlanner::acquire`] would have run inline. Dropping an
/// uncollected ticket deregisters the ask exactly like [`WaitRegistration`]
/// (cancellation on process death); a parked outcome's reservation returns
/// to the pool through the removed entry's drop.
pub(crate) struct AllocationTicket {
    planner: Arc<ResidencyPlanner>,
    key: EntryKey,
    notify: Arc<Notify>,
    collected: bool,
}

impl AllocationTicket {
    /// The park-collect half of [`ResidencyPlanner::acquire`]: arm the
    /// notify, probe the outcome, sleep until served. Returns
    /// [`Acquired::Yield`] when the owner was chosen for eviction while
    /// parked (the caller settles the tail and waits out the eviction, as
    /// the inline path does).
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

    /// E5: arm the single-owner drain task. Called once at bootstrap, inside
    /// the runtime. Idempotent; a planner constructed outside a runtime
    /// keeps inline planning (pre-boot callers).
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

    /// Request a planning pass. With the drain armed this is one atomic
    /// notify — the caller NEVER runs the drain itself (guest paths must
    /// not carry idle-reclaim scans or eviction quoting). Coalescing is
    /// free: `Notify` holds one permit, and `plan()` drains everything.
    fn poke(self: &Arc<Self>) {
        match self.drain.get() {
            Some(notify) => notify.notify_one(),
            None => self.plan(),
        }
    }

    /// §10.14: poke the drain when the free list has fallen below the
    /// configured supply runway. The uncontended paths neither park nor
    /// poke — nothing would ever wake the planner to rebuild the runway
    /// while the fast path quietly drains the pool toward the phase
    /// mismatch the runway exists to pre-empt. One flag load (and nothing
    /// else) with the runway off.
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

    /// The single semantic site for "device pages actually became free":
    /// re-arms rung 0 (the idle-reclaim scan may find work again). Every
    /// caller pairs it with a poke; only the poke's condition differs.
    fn re_arm_idle_reclaim(&self) {
        self.idle_reclaim_exhausted.store(false, Ordering::Release);
    }

    /// The `(model, driver)` pair this planner manages.
    pub fn locus(&self) -> (usize, usize) {
        self.port.locus()
    }

    /// The mutation chokepoint for everything the lock-free mirrors track:
    /// queue and residency changes go through here, and both counters are
    /// recomputed on the way out — computed, never incrementally mirrored.
    /// (`note_progress` / `note_ask_and_check_elder` lock `inner` directly:
    /// they touch only `progressed`, which no mirror reads.)
    /// The single door to the planner mutex, so the census sees every taker.
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

    /// Register `pid` at an *existing* FCFS position rather than the newest
    /// one — the restart path (see [`crate::inferlet::process::spawn`]).
    ///
    /// The clock is not rewound: `next_seq` keeps advancing, so a subsequent
    /// fresh registration still sorts after everything. Two live processes
    /// briefly sharing a seq is harmless because queue entries are ordered by
    /// `EntryKey = (spawn_seq, insertion_order)` and keyed per-process.
    pub fn register_with_seq(&self, pid: ProcessId, seq: u64) {
        self.with_inner(|inner| {
            inner.procs.insert(pid, Proc::new(seq));
        });
    }

    /// This process's position in the FCFS clock, if it is still registered.
    pub fn spawn_seq(&self, pid: ProcessId) -> Option<u64> {
        self.lock_inner().procs.get(&pid).map(|proc| proc.seq)
    }

    /// `pid` has claimed an execution slot and may now hold pooled pages.
    ///
    /// Until this lands the process is registered (it owns its FCFS seq) but
    /// provably page-less, so [`Self::is_wedged`] must not count it as a
    /// process that could still free something. Idempotent; called on every
    /// execution admit, including the uncapped case.
    pub fn note_admitted(&self, pid: ProcessId) {
        let mut inner = self.lock_inner();
        if let Some(proc) = inner.procs.get_mut(&pid) {
            proc.admitted = true;
        }
    }

    /// Unregister at process exit/terminate. Its queue entries are removed,
    /// gate waiters are woken for teardown, and freed capacity drains to the
    /// queue.
    pub fn unregister(self: &Arc<Self>, pid: ProcessId) {
        let (signal, removed) = self.with_inner(|inner| {
            let departed = inner.procs.remove(&pid);
            // Rotation damping: a RESIDENT departure is the completion
            // event that funds one readmission (Rule A as a stock, not
            // just a flow — measured: 33% of restores were landing within
            // 3 ms of an evict commit, because demand-sized rounds leave
            // surplus in the shared pool and that surplus was driving the
            // rotation to 2× the completion rate). One completed working
            // set funds one restored working set; no page counts, no
            // constants. An evicted process completing while out held no
            // pooled pages, so it funds nothing.
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
            // §10.14: with nobody waiting the poke is normally skipped, but
            // a free event is also the runway's chance to notice it is
            // still short and top up before the next ask parks.
            self.poke_if_runway_short();
        }
    }

    // =========================================================================
    // The acquisition path
    // =========================================================================

    /// All-or-nothing direct reservation off the free lists — the hot path.
    fn try_reserve(&self, demand: Demand) -> Option<AllocationGrant> {
        let kv = if demand.kv_pages > 0 {
            let pages = self.port.reserve_device(demand.kv_pages)?;
            DevicePageReservation::new(pages, self.port.clone())
        } else {
            DevicePageReservation::empty()
        };
        let rs = if demand.rs_slots > 0 {
            match self.port.reserve_rs(demand.rs_slots) {
                Some(slots) => RsSlotReservation::new(slots, self.port.clone()),
                None => return None, // kv reservation returns via Drop
            }
        } else {
            RsSlotReservation::empty()
        };
        Some(AllocationGrant::new(demand, kv, rs))
    }

    /// Acquire one grant. Uncontended (no waiters, everyone resident), this
    /// is two free-list pops — no planner lock is taken. Otherwise the ask
    /// parks FCFS at the process's spawn position and is served out of the
    /// head-first accumulation. [`Acquired::Yield`] hands control back to
    /// the fire path when this process must settle its own tail (eviction).
    /// Instrumentation: cumulative parks and the current parked width.
    /// Sampling the width alone misses parks shorter than the sample
    /// interval, which is what made an oversubscribed pool look idle.
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

    pub async fn acquire(
        self: &Arc<Self>,
        pid: ProcessId,
        quorum_id: ProcessId,
        demand: Demand,
    ) -> Result<Acquired, PlannerError> {
        if demand.is_zero() {
            // E6 progress event: a zero-demand fire is progress too (the
            // nonzero path notes via `note_ask_and_check_elder`, same
            // condition). Protection only matters under contention, so the
            // uncontended hot path never takes the lock.
            if self.waiters.load(Ordering::Acquire) != 0
                || self.nonresident.load(Ordering::Acquire) != 0
            {
                self.note_progress(pid);
            }
            return Ok(Acquired::Granted(AllocationGrant::empty()));
        }
        // Fast path. Uncontended (no waiters, everyone resident): two
        // free-list pops, no planner lock. Once anything is queued the
        // fast path closes and EVERY ask parks FCFS at its process's
        // spawn position.
        //
        // The elder bypass that used to live here — a process older
        // than the queue head could still reserve directly — was
        // deleted 2026-07-26 (`rainer_v3.md` §8.5). It was justified by
        // a 47% inter-batch gap measured 2026-07-25, before the §17
        // mechanism fixes; re-measured after them it earns nothing:
        // A4/A6/E3 all land inside their bands with it gone, F2 stays
        // 12/12, and one ordering rule disappears with it. v2 lists it
        // under "dies ... derivatives of implicit membership".
        let uncontended = self.waiters.load(Ordering::Acquire) == 0
            && self.nonresident.load(Ordering::Acquire) == 0;
        // Opt-in (`PIE_ALLOC_FAST_SMALL=1`) head-harmless bypass: even
        // with the fast path closed, serve THIS ask straight from the
        // free lists when free pages cover the ask AND the FCFS head's
        // remaining shortfall — the head could not have used what this
        // ask takes, so no page the head is entitled to is diverted.
        // (12% of contended parks happen with free >= demand; under
        // strict FCFS they wait for the queue anyway.) Exact
        // arithmetic, no constants; racing drains are safe because
        // `try_reserve` simply fails and the ask parks as before.
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
            // E6 progress: this process is asking to fire again. Under
            // contention only — the same condition the zero-demand path
            // uses, and the elder check used to fold this in.
            self.note_progress(pid);
        }
        if uncontended && let Some(grant) = self.try_reserve(demand) {
            self.poke_if_runway_short();
            return Ok(Acquired::Granted(grant));
        }
        // The reserve failed (or the fast path is closed): run the
        // exact-arithmetic exhaustion check before parking. Slow path
        // only — a demand that reserves off the free lists trivially
        // fits, and reading totals is a store-lock hold the per-fire
        // hot path must not pay (§16).
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
            Parked::Gone => return Err(PlannerError::Cancelled),
            Parked::NotResident => {
                // Out of the set (or in transfer): the fire path settles
                // the process's tail and waits out the eviction.
                return Ok(Acquired::Yield);
            }
            Parked::Entry(key, notify) => {
                self.stats.parks.fetch_add(1, Ordering::Relaxed);
                ptrace!("park key={:?} pid={} kv={}", key, pid, demand.kv_pages);
                // The park empties this lane's seat in the wait-all
                // quorum so frames seal without it; rejoin is implicit
                // on the lane's next accepted fire.
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

    /// [`Self::acquire`]'s decision half, non-blocking (the deferred-
    /// allocation path, `PIE_DEFER_ALLOC=1`; `acquire` itself is untouched).
    /// Fast paths, E6 progress notes, `Impossible` sync checks, park census
    /// probes and the queue insert at the SAME FCFS position (including
    /// `notify_lane_close`) are reproduced from `acquire` exactly; where
    /// `acquire` would enter the notified/collect loop this returns the
    /// owned [`AllocationTicket`] instead, to be awaited via
    /// [`AllocationTicket::collect`] from the caller's own task.
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
        // Fast path — see `acquire`: uncontended is two free-list pops; the
        // opt-in head-harmless bypass serves small asks that cannot divert
        // pages the FCFS head is entitled to.
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
                // The park empties this lane's seat in the wait-all quorum so
                // frames seal without it; rejoin is implicit on the lane's
                // next accepted fire — for a deferred frame, the engine
                // completion task's own submission.
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

    /// E6: mark `pid` as having progressed since its last restore (it is
    /// asking to fire), making it an eviction candidate again. Zero-demand
    /// asks note through here; nonzero contended asks note through
    /// [`Self::note_ask_and_check_elder`] — the same event, one lock hit
    /// either way. `progressed` does not feed the lock-free mirrors, so a
    /// plain lock suffices.
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

    // =========================================================================
    // The residency gate
    // =========================================================================

    pub fn is_resident(&self, pid: ProcessId) -> bool {
        self.inner
            .lock()
            .procs
            .get(&pid)
            .is_none_or(|proc| proc.state == Residency::Resident)
    }

    /// This process's lock-free residency flag, taken ONCE and then read
    /// on every host-method prologue without touching the planner lock.
    ///
    /// `None` means the process is not registered, which the gate treats
    /// as resident for the same reason [`Self::is_resident`] does: an
    /// unregistered process owns no pooled pages, so there is nothing to
    /// wait for. A process is registered before it can run any guest code,
    /// so the caller's first call already sees its own flag.
    pub fn residency_flag(&self, pid: ProcessId) -> Option<Arc<AtomicBool>> {
        self.lock_inner()
            .procs
            .get(&pid)
            .map(|proc| proc.resident.clone())
    }

    /// Park until `pid` is resident again (or gone). The caller drained its
    /// own pending fires first, so the parked task holds no pins.
    ///
    /// This is the choke point for the frame-policy contract ("any
    /// guest-side wait that stops a lane's next fire MUST post a leave"):
    /// before the first actual park, a process-wide `Suspend` leave is
    /// re-posted here — idempotent, and ordered AFTER any fire this guest
    /// submitted before it observed the eviction. Without it, a pre-fence
    /// fire arriving after the evictor's leave resurrects the lane
    /// `awaited: true` and, once drained, wedges the fleet's seal (the
    /// p4-h2h-3 300 s freeze, CONTENTION_FOLLOWUP.md §15.2). Posting here
    /// covers every current and future park site by construction.
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

    // =========================================================================
    // The drain — head-first accumulation
    // =========================================================================

    /// Serve the queue: pull free pages into the accumulation until the head
    /// is covered, serve it, repeat. Falls through to rung 0 (idle reclaim)
    /// and then eviction planning when the pool runs dry.
    pub fn plan(self: &Arc<Self>) {
        loop {
            let step = self.with_inner(|inner| {
                let Some((key, waiter)) = inner.unmet_head() else {
                    if inner.accum.len() > 0 && inner.queue.is_empty() {
                        let stranded =
                            std::mem::replace(&mut inner.accum, DevicePageReservation::empty());
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
                    // TWO CURRENCIES. An eviction may fund an ALLOCATION —
                    // that is forward progress: the payer's pages let ~20
                    // running lanes each take their next page. It must never
                    // fund a RESTORE, because then eviction and readmission
                    // are the same act and the pool oscillates: measured on
                    // D/128 @2048, 1290 evictions against 1259 restores, a
                    // standing 28 of 128 processes out of the fleet, and 98%
                    // of their 850 ms absence spent queued rather than
                    // copying (the H2D itself is 15 ms).
                    //
                    // Readmission is funded by COMPLETION instead (and idle
                    // reclaim), which is the only sustainable rate: as the
                    // evicted population grows the resident set shrinks until
                    // its members can actually reach their full working set
                    // and finish, and each completion then pays for one
                    // readmission. No cap, no score, no constant — the
                    // equilibrium is whatever the pool can carry.
                    //
                    // `fleet_stalled` keeps the liveness backstop exactly as
                    // it was: when nothing can complete, a restore may evict.
                    let fund_by_eviction = !is_restore || inner.fleet_stalled();
                    return Step::Absorb {
                        count: missing,
                        fund_by_eviction,
                    };
                }
                match demand {
                    // An RS-carrying ask keeps the per-step path: its RS
                    // reservation is a port call, and the port is never
                    // touched under the planner lock.
                    Some(demand) if demand.rs_slots > 0 => {
                        Step::ServeAllocation { key, demand }
                    }
                    Some(_) => Step::ServeAllocationBurst {
                        extra: inner.burst_shortfall(),
                    },
                    None => {
                        // Rotation damping: a covered restore still waits
                        // for a completion credit — accum surplus (often
                        // eviction yield) must flow to allocations, not
                        // fund readmission. `fleet_stalled` keeps the
                        // liveness backstop: when nothing can complete, a
                        // restore proceeds regardless.
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
                    // §10.14: the drain has nothing to serve (empty queue,
                    // or a covered-but-damped restore head) — exactly the
                    // idle phase in which the runway is rebuilt. One flag
                    // load with the runway off.
                    self.plan_runway();
                    return;
                }
                Step::Release(stranded) => {
                    drop(stranded);
                    // As above; the release has just returned the stranded
                    // accumulation to the free lists, so the shortfall is
                    // measured against the honest free count.
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
                    // Rung 0, latched: the cache-lease scan runs at most once
                    // per free event. Without the latch every poke at free=0
                    // re-scanned fruitlessly under the global KV lock (§15).
                    if !self
                        .idle_reclaim_exhausted
                        .swap(true, std::sync::atomic::Ordering::AcqRel)
                        && self.port.reclaim_idle() > 0
                    {
                        self.idle_reclaim_exhausted.store(false, Ordering::Release);
                        continue;
                    }
                    if !fund_by_eviction {
                        // A restore head with a dry pool. Nothing to do but
                        // wait for a completion — and waiting is safe: the
                        // head is only a restore when NO allocation is unmet,
                        // so returning strands no other ask.
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
                            // RS pool short; the next RS free re-plans — but
                            // only a RUNNING process can free one, so if every
                            // admitted process is parked there is no next free
                            // and the head waits forever. The page ladder
                            // cannot catch this: it is entered from a failed
                            // absorb, and this head's pages are already
                            // covered, so nothing below reaches the rung.
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
                            // CASCADE: the served head no longer competes
                            // (`is_unmet` skips it), so keep draining — the
                            // next-oldest unmet ask absorbs whatever free
                            // pages remain. Without this, surplus pages sit
                            // idle while the whole fleet is parked (the
                            // fast path is closed whenever waiters exist),
                            // which is exactly the old drain's
                            // younger-entries-from-surplus rule lost.
                            continue;
                        }
                        ServeOutcome::Stale(rs) => {
                            drop(rs);
                            continue;
                        }
                    }
                }
                Step::ServeAllocationBurst { extra } => {
                    // ONE free-list pull for the whole burst, outside the
                    // planner lock like every other port call — and the only
                    // supply read this pass makes (§16: the store lock is
                    // never taken inside the planner lock, and never more
                    // often than the sequential path did per serve).
                    if extra > 0 {
                        let pages = self.port.reserve_device_up_to(extra);
                        if !pages.is_empty() {
                            let reservation = DevicePageReservation::new(pages, self.port.clone());
                            self.with_inner(|inner| inner.accum.absorb(reservation));
                        }
                        // An empty pull is NOT a shortage to escalate: the
                        // head is covered by the accumulation already (that
                        // is why this step was chosen), so the burst still
                        // serves what it can and the next iteration takes
                        // the ordinary `Absorb` rung for the rest.
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
                    // CASCADE, as on the per-step path: the served entries no
                    // longer compete, so keep draining — whatever is left
                    // (an RS-carrying ask, a restore, a short head) is
                    // decided by the ordinary rungs on the next iteration.
                    // A pass that served nothing behaves exactly like a
                    // `Stale` serve: re-decide.
                    continue;
                }
                Step::ServeRestore { key, pid } => {
                    // Re-validate the ask against the store: the demand is
                    // whatever is swapped NOW (teardown or discards may have
                    // shrunk it while the entry waited).
                    let (model, driver) = self.port.locus();
                    let swapped = swapped_page_count(pid, model, driver);
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

    // =========================================================================
    // Eviction planning — deficit-sized, youngest-first, younger than head
    // =========================================================================

    /// Quote `candidates`, stopping as soon as `deficit` is covered, and
    /// return the victims to evict plus the ones the host swap pool cannot
    /// currently hold.
    ///
    /// Quoting is done OUTSIDE the planner lock (the probe takes store
    /// locks). It is bounded instead by the `deficit` budget the quoter
    /// itself honours — with one re-ask when a candidate refused for host
    /// room makes that budget stop short of what this loop can bank.
    ///
    /// `host_room` is the free host-swap slot count. `prepare_suspend`
    /// allocates one host slot per page it moves, ALL OR NOTHING, over
    /// exactly the page set the quote counted — so a quote larger than the
    /// room left is a proof, available here for free, that the eviction
    /// would roll back on `HostSwapFull` after paying the whole fence →
    /// lane-leave → drain → lease-quiesce cycle. Such candidates are
    /// reported separately rather than picked; the caller parks them, which
    /// is what the rollback path did anyway, minus the cycle.
    ///
    /// Candidates are ordered lease-QUIESCENT first (a racy preference
    /// snapshot), youngest-first within each class.
    fn quote_and_pick(
        &self,
        candidates: Vec<(ProcessId, u64)>,
        deficit: u32,
        host_room: u32,
        model: usize,
        driver: usize,
    ) -> (Vec<(ProcessId, u32)>, Vec<ProcessId>) {
        let mut ordered: Vec<(ProcessId, u64, bool)> = candidates
            .into_iter()
            .map(|(pid, seq)| {
                let quiescent =
                    crate::inferlet::process::residency::kv_lease_quiescent(pid, model, driver);
                (pid, seq, quiescent)
            })
            .collect();
        ordered.sort_by_key(|&(_, seq, quiescent)| (!quiescent, std::cmp::Reverse(seq)));
        let pids: Vec<ProcessId> = ordered.iter().map(|(pid, ..)| *pid).collect();
        Self::pick_with_budget_escalation(&pids, deficit, host_room, |pids, budget| {
            crate::inferlet::process::residency::kv_reclaim_quotes(pids, model, driver, budget)
        })
    }

    /// The budgeted pick, plus the re-ask that keeps the budget from hiding a
    /// victim. Takes the quoter as a parameter so the escalation itself is
    /// testable without a live KV store.
    ///
    /// The budget and the picker do NOT charge the same things. The quoter
    /// stops once the answers it EMITTED reach `deficit`; the picker only
    /// banks the ones that fit the host room left, skipping the rest. When a
    /// candidate is skipped the two accountings diverge, and the tail the
    /// budget cut off comes back as `None` — indistinguishable from "process
    /// unknown". Reading that as "nothing to give" parks every candidate and
    /// hands the head to the starvation rung, which destroys a live
    /// allocation while a victim that DOES fit sat one position past the cut
    /// (`check_starvation` skips `last_resort_evict` on `NoSwapRoom`, so
    /// nothing downstream recovers it either).
    ///
    /// So re-ask for an opinion on everyone. Bounded to one extra call, and
    /// only on the path where the first pass came up short with at least one
    /// candidate refused for room — never when the host pool is roomy enough
    /// to fund the deficit outright, which is the contended case the budget
    /// exists for. The trigger is exact, not conservative: the quoter never
    /// emits `Pages(0)`, so it truncates only once the pages it emitted reach
    /// `deficit`, and the picker routes every emitted page into `picks` or
    /// `unhostable` unless it already covered `deficit` — hence coming up
    /// short after a truncation implies `unhostable` is non-empty.
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
        // Eviction funds the head only when KV bytes can physically move
        // out: a swap-incapable driver or an exhausted host pool leaves the
        // starvation predicate as the last rung.
        let (host_free, host_total) = self.port.host_stats();
        if !self.port.suspend_capable() || host_free == 0 {
            self.check_starvation(StarveCause::NoSwapRoom);
            return;
        }
        // HOST RESERVE. The rung may not spend the last of the host pool on
        // a fleet that is still running.
        //
        // The gate below asks whether waiting can produce the next page. It
        // does not ask what the eviction is being paid for with. Eviction is
        // funded by the host pool, the pool is finite, and the only event
        // that returns a host slot is a restore — so a saturated fleet,
        // whose `free` is 0 by definition and whose head is therefore
        // permanently unmet, drains the host pool to nothing and then meets
        // the `host_free == 0` arm above, which is a KILL. The fleet was
        // never in trouble; it was recycling its pages as fast as it could.
        //
        // Measured on `soak` (160 device pages, 1024 host pages, 4096
        // requests at 256-way): the rung took host_free 1024 -> 0 in the
        // first seconds and held it there for the remaining 60 s, running
        // 631 evictions against 123 before the rung was opened and handing
        // 99 processes to the starvation rung — every one of them destroyed
        // and restarted, i.e. work thrown away by a fleet that was
        // completing normally.
        //
        // So the last slice of the host pool is reserved for the case the
        // rung actually exists for. [`Self::is_wedged`] — nothing admitted
        // can progress on its own — spends it; anything less defers and
        // waits for the completion that a live fleet will deliver. This
        // does not close the rung at high oversubscription, which is the
        // defect the wedge predicate had: the reserve is a fraction of the
        // HOST pool, so it binds only where host room is scarce relative to
        // demand. On the D/512 shape the rung exists for (8192 device pages
        // against 16384 host pages) the reserve is 2048 host pages and the
        // device pool cannot put enough out to reach it.
        if host_free.saturating_mul(HOST_RESERVE_DIVISOR) <= host_total && !self.is_wedged() {
            self.stats
                .eviction_deferrals
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        // LOAD CONTROL. Eviction is a SUPPLY rung, not a demand rung.
        //
        // A shortage is not by itself a reason to move KV bytes: a process
        // that is still running will retire, free, and poke the drain, and
        // evicting to serve the head sooner only trades that free for a
        // fence → lane-leave → lease-quiesce → D2H round trip and an H2D
        // owed to put the victim back.
        //
        // What that argument needs, and what the wedge predicate assumed,
        // is that a running process's pages are ALREADY on their way back.
        // At high oversubscription they are not. Measured on D/512 (768
        // requests, 512-way, 8192 pages, 35 pages needed per request): 512
        // admitted processes each held ~16 pages of the 35 their last token
        // needs, `free` was 0 for 98.75% of samples, and the FCFS head's
        // ask was 1 page. Nobody could finish, so nobody freed, so the
        // wedge predicate — which is false while ANY admitted process is
        // unparked — deferred 22826 of 22859 attempts (99.86%) and moved
        // 595 of 8192 pages. Host swap was dead code in the one regime it
        // exists for, and the shape ran at 0.71x of vLLM.
        //
        // So the gate is the arithmetic of supply instead: the pool is
        // empty, nothing is already on its way (no eviction in flight, no
        // kill ordered), and the head's ask exceeds what the accumulation
        // has managed to pull. Those four together are "waiting cannot
        // produce the next page", which is the honest form of what the
        // wedge predicate was reaching for.
        //
        // This IS more eviction than before, and the X-shape number above
        // is the standing warning about paying for it. Three things bound
        // it. Host swap is opt-in — with no host room the rung returns
        // `NoSwapRoom` above and never reaches this gate at all, so every
        // no-swap deployment is unaffected. The HOST RESERVE above caps
        // how much of the host pool a live fleet may spend. And the gate is
        // demand-exact: pages already in flight count as supply, so a round
        // is ordered only for the shortfall the in-flight ones cannot cover,
        // and the rung shuts itself the instant they do.
        //
        // It used to be `evicting.is_empty()` instead, which capped the
        // rung at ONE victim in flight — a rate of one victim-landing
        // (~36 ms) per round regardless of how deep the shortage was.
        // Measured on D/512 at 128-way with 2048 pages: that ceiling is
        // ~605 pages/s against a fleet demand of ~625 pages/s, so the
        // parked population never drained. Wave width sat at 172 of 253
        // (26 lanes parked at all times) while the wave PERIOD was flat in
        // width from 128 to 287 — i.e. the lanes were free to carry and
        // the only thing missing was pages. Demand-exactness bounds the
        // over-eviction the serialization was really there to prevent,
        // without capping the rate.
        if !self.supply_stalled() {
            self.stats
                .eviction_deferrals
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        let Some(victims) = self.victim_set() else {
            return;
        };
        let (model, driver) = self.port.locus();
        // Routine path: honour E6 hysteresis. `preferred()` may be empty
        // while the set is not — that is a policy outcome, and it is why
        // the endgame below must consult the SET, never this subset.
        let (picks, unhostable) = self.quote_and_pick(
            victims.preferred(),
            victims.deficit,
            host_free,
            model,
            driver,
        );
        // A candidate whose reclaim does not fit the host pool is parked on
        // exactly the set the `HostSwapFull` rollback would have parked it
        // on, so the deterministic re-pick still walks down to a victim that
        // DOES fit, and the set still empties into the starvation rung once
        // none does. What it no longer costs is the fence → suspend-notify →
        // drain → lease-quiesce → prepare cycle that produced the answer the
        // quote already had: soak (160 pages / 1024 host slots / 256-way)
        // spent 96% of its steady state at host_free <= 3 against victims
        // holding 4-9 pages, and rolled back 10864 evictions to learn it.
        // Nothing was ordered here, so nothing is rolled back — the rollback
        // counter measures abandoned WORK, and there is none.
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
            // The last-resort rung lives in `check_starvation`, which
            // builds its own `VictimSet` at the instant the kill is
            // decided. Re-using this one would decide against a snapshot
            // that is already stale — the defect that destroyed a request
            // twice (§18.9, §18.10).
            self.check_starvation(StarveCause::NoEligibleVictim);
            return;
        }
        self.commit_evictions(victims.head, picks, false, victims.runway_grab);
    }

    /// §10.14 SUPPLY-PHASE RUNWAY, drain-idle entry (`PIE_SUPPLY_RUNWAY`;
    /// off = this returns after one flag load). `plan_eviction` is only
    /// reachable behind an unmet head, so a shortfall against the runway
    /// target with an EMPTY ask queue — the phase §10.14 exists for: demand
    /// is continuous, supply is lumpy, and the reactive gate opens only
    /// after the boundary asks have already parked — needs its own entry
    /// from the drain's idle exits.
    ///
    /// This rung is OPPORTUNISTIC, which is the whole difference from
    /// `plan_eviction`: no head is starving, so there is no starvation or
    /// hog rung on any of its exits — an unquotable fleet, an exhausted
    /// host pool, or a missing transport simply mean "no round". The
    /// victims it does order are ordinary evictions: same executor, same
    /// landing, their yield returns to the free lists / accumulation like
    /// any eviction yield, and grants stay strictly FCFS.
    ///
    /// Rotation damping is not routed around: a runway victim's restore
    /// entry queues at its spawn position exactly as today, and its SERVE
    /// still waits for a completion credit. The round itself funds
    /// ALLOCATIONS — which is why `runway_shortfall_set` refuses to run
    /// with a banked credit, see there.
    fn plan_runway(self: &Arc<Self>) {
        let runway = supply_runway_pages();
        if runway == 0 {
            return;
        }
        // The same physical precondition as `plan_eviction`, minus its
        // starvation rung: no transport or no host room is "no round".
        let (host_free, _) = self.port.host_stats();
        if !self.port.suspend_capable() || host_free == 0 {
            return;
        }
        let (free, _) = self.port.device_stats();
        let Some((deficit, members)) = self.runway_shortfall_set(runway, free) else {
            return;
        };
        let (model, driver) = self.port.locus();
        let (picks, _unhostable) = self.quote_and_pick(members, deficit, host_free, model, driver);
        if picks.is_empty() {
            return;
        }
        self.commit_runway(picks);
    }

    /// The detection + candidate half of [`Self::plan_runway`], under one
    /// planner-lock snapshot. `Some((deficit, members))` when a runway
    /// round should be ordered: the free list (plus accumulation and
    /// in-flight eviction yield — the same supply arithmetic as
    /// `supply_stalled`) is short of `runway`, no runway round is already
    /// in flight (hysteresis), no kill is pending, and the readmission
    /// rotation is credit-starved.
    ///
    /// The credit check is what keeps this rung physical: with a banked
    /// completion the damping gate is open, so the round's yield would be
    /// absorbed by a serveable restore — including the restore the round
    /// itself creates — and the free list ends where it started, one
    /// D2H+H2D round trip poorer. Credit-starved is precisely the regime
    /// where evicted pages stay free for the boundary-ask stream.
    ///
    /// Membership is the anti-thrash rule against [`Inner::runway_floor`]:
    /// younger than the oldest admitted resident (the round must never
    /// evict the LAST resident — that clamp is what keeps a onefits-sized
    /// pool safe from an over-large runway) and younger than any queued
    /// head. E6 hysteresis is honored and never relaxed: an opportunistic
    /// round has no liveness claim on a just-restored victim.
    ///
    /// Takes `runway` and `free` as parameters so the arithmetic is
    /// testable without the process-global env latch.
    fn runway_shortfall_set(&self, runway: u32, free: u32) -> Option<(u32, Vec<(ProcessId, u64)>)> {
        self.with_inner(|inner| {
            // No credit gate here: measured on the D cell, completions bank
            // credits continuously, so gating on credit == 0 kept the runway
            // at 0 rounds for a whole run (runway=0/0). FCFS itself already
            // orders queued boundary asks against the victim's restore; the
            // idle-churn case the gate aimed at is bounded by the in-flight
            // latch and the shortfall arithmetic (a restored victim's pages
            // re-enter `free` and close the deficit).
            if inner.runway_round_in_flight
                || inner.kill_in_flight()
                || !rotation_damping_enabled()
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

    /// Commit a runway round: [`Self::commit_evictions`]'s re-validation
    /// with [`Inner::runway_floor`] standing in for the head (there may be
    /// none), and the §10.14 hysteresis latched under the same lock hold
    /// that marks the victims — a racing plan can never order a second
    /// runway round in between.
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
                // The victim's parked allocations yield exactly as on the
                // routine path: their fire tasks settle the pipeline tail
                // the eviction quiesces on.
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

    /// Snapshot every process FCFS permits evicting for the current head.
    ///
    /// **This is the liveness-bearing fact.** Membership is exactly the
    /// anti-thrash rule — younger than the head, resident — and nothing
    /// else. E6 hysteresis rides along as a per-member *policy tag*
    /// ([`Victim::e6_fresh`]) and can never remove a member, because a
    /// heuristic that can empty this set is a heuristic that can destroy a
    /// request: that is precisely what E6 did before §18.9.
    ///
    /// Read under one lock so the set is internally consistent, and cheap
    /// enough to re-take at every decision point — callers are expected to
    /// build a FRESH one rather than carry one across a lock release.
    fn victim_set(&self) -> Option<VictimSet> {
        // §10.14: read the pool BEFORE the planner lock (`inner` is
        // innermost; the pool must never be taken inside it). One flag load
        // and no read at all with the runway off.
        let runway = supply_runway_pages();
        let runway_free = if runway > 0 {
            self.port.device_stats().0
        } else {
            0
        };
        self.with_inner(|inner| {
            let (head, waiter) = inner.unmet_head()?;
            // The round covers the head's shortfall PLUS the queued
            // quantum-ask stream (the same demand `supply_stalled` orders
            // the round for): sizing by the head alone kept rounds at ~one
            // victim, so the supply RATE saturated at the round latency and
            // the early gate moved nothing (free 2.0 pages, parks
            // unchanged — measured). Rule A untouched: restores still do
            // not count.
            let queued_quantum: u32 = inner
                .queue
                .values()
                .filter(|queued| queued.is_unmet())
                .filter_map(|queued| match &queued.kind {
                    WaitKind::Allocation { demand, .. } if demand.kv_pages == 1 => Some(1),
                    _ => None,
                })
                .sum();
            // §10.14: the round ALSO restores the free-page runway, on the
            // same one-round-in-flight and completion-credit brakes as the
            // `supply_stalled` clause that orders it (see there). With the
            // runway off (or latched) `runway_grab` is 0 and the target is
            // exactly the shipped demand-exact arithmetic.
            let runway_grab =
                if runway > 0 && !inner.runway_round_in_flight {
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
                // Unadmitted processes hold no pooled pages by construction,
                // so they can never cover a deficit — quoting them is pure
                // cost on the contended path. A host-swap-blocked victim
                // cannot physically move until host room returns, and
                // re-picking it is what livelocked §20.6.
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

    /// Mark the picked victims and spawn their eviction executors.
    ///
    /// Re-validates under the lock (the quotes were taken without it) and
    /// YIELDS each victim's parked allocations: their fire tasks wake,
    /// settle the victim's pipeline tail (the only tasks that can finalize
    /// device-geometry ops), and wait out the eviction — that settling is
    /// what drains the leases the eviction quiesces on.
    ///
    /// `e6_relaxed` waives the post-restore hysteresis check here too;
    /// without it a relaxed pick is silently dropped at this re-validation
    /// and the planner spins (the livelock in §18.9).
    ///
    /// `runway_grab` is the §10.14 shortfall component of the round's
    /// deficit ([`VictimSet::runway_grab`], 0 with the runway off):
    /// committing a round it enlarged latches the one-round hysteresis.
    ///
    /// Returns whether anything was actually spawned.
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

    /// The starvation endgame, as a computed predicate — the timer-free
    /// replacement for the retired deadline-kill rung. Fires only when ALL
    /// of the following hold at once:
    ///
    /// - the head is unmet and eviction cannot fund it (checked by the
    ///   caller, which names the reason via [`StarveCause`]: no swap
    ///   transport / exhausted host pool, or no eligible victim);
    /// - no eviction is in flight (its landing pokes);
    /// - every OTHER registered process that has claimed an execution slot
    ///   is either parked in this queue or out of the resident set — nobody
    ///   is running, so no completion can ever arrive to free pages. (A
    ///   lease snapshot is NOT enough: a guest between two fires holds no
    ///   lease for an instant but is very much alive.)
    ///
    /// Then the YOUNGEST parked allocation is failed loud (never the head:
    /// destruction youngest-first), and its rollback frees restart the
    /// fleet. Every quantity is read at one instant; any new fire or free
    /// re-plans and re-evaluates.
    /// The wedge predicate: nothing is running, nothing is in transit, and
    /// every resident process is parked — so no completion can arrive to
    /// free pages on its own. Shared by the E6 relaxation rung and
    /// [`Self::check_starvation`] so the two can never disagree about what
    /// "wedged" means.
    ///
    /// A SERVED-but-uncollected entry means its process is about to run
    /// (and eventually complete, freeing pages): that is relief on its way,
    /// not a wedge. Only genuinely unmet parks count. A lease snapshot is
    /// NOT enough either: a guest between two fires holds no lease for an
    /// instant but is very much alive.
    ///
    /// A registered process that has NOT passed execution admission is
    /// likewise not relief: it holds no pooled pages (the execution gate is
    /// what stands between a process and every pooled KV/RS resource), so it
    /// can never free any, and it cannot even start until an admitted
    /// process retires — which is exactly what the wedge prevents. Counting
    /// it as "someone is still running" is what let a fleet of
    /// `num_requests > max_concurrent_processes` deadlock silently and
    /// forever: the whole admitted cohort parked on an empty pool, the
    /// unadmitted remainder queued behind the very permits those parked
    /// processes hold, and the starvation rung disarmed by their presence.
    /// Waiting cannot produce the next page: the pool is empty, no kill is
    /// ordered, and the head still asks for more than the accumulation
    /// holds PLUS the pages already in flight from evictions under way.
    ///
    /// This is the eviction rung's load control. It is strictly weaker than
    /// [`Self::is_wedged`], which additionally requires that no admitted
    /// process is running — see the measurement in `plan_eviction` for why
    /// that extra clause made the rung unreachable at high
    /// oversubscription.
    fn supply_stalled(&self) -> bool {
        let (free, _) = self.port.device_stats();
        self.with_inner(|inner| {
            if inner.kill_in_flight() {
                return false;
            }
            if inner.unmet_head().is_none() {
                return false;
            }
            // §10.14 supply-phase runway (`PIE_SUPPLY_RUNWAY`, default off):
            // the free list itself is a demand. Demand at D is continuous
            // (~32 pages/wave of 1-page boundary asks) while supply is lumpy
            // (~35 pages per completion), so at free≈0 the demand-exact gate
            // below is still REACTIVE — it opens only once asks have queued.
            // A shortfall against the runway target opens the rung by
            // itself, so the round starts before the boundary asks park.
            // Same supply arithmetic as the gate below (accumulation and
            // in-flight eviction yield count as pages on their way, so an
            // in-flight round shuts this clause), plus two brakes of its
            // own: the hysteresis latch (one runway-motivated round in
            // flight at a time) and the completion-credit check — with a
            // banked credit the damping gate is open, meaning the yield
            // would be absorbed by a serveable restore rather than staying
            // free, and the runway cannot be defended by evicting.
            let runway = supply_runway_pages();
            if runway > 0 && !inner.runway_round_in_flight {
                let expected: u32 = inner.evicting.values().map(|mark| mark.pages).sum();
                if runway.saturating_sub(free) > inner.accum.len() as u32 + expected {
                    return true;
                }
            }
            // Demand-exact over the WHOLE queued allocation demand, not just
            // the head's ask. The old `free == 0 && head short` form started
            // supply only after demand had already parked: free hovered at
            // ~1 page, every 1-page boundary ask parked first (measured
            // 614 parks/s against 39 evictions/s with 716 deferrals/s — the
            // rung was evaluated constantly and deferred 18:1), and ~5 of
            // the ~95 resident lanes were absent from every wave. Summing
            // queued allocation demand orders the round as soon as the
            // queue outgrows what free + accumulation + in-flight evictions
            // can serve, so the free list runs slightly ahead of the
            // boundary-ask stream and asks are served without parking.
            // Restore demand is deliberately EXCLUDED: an eviction may not
            // fund a restore (Rule A) — restores are funded by completions.
            // And only SINGLE-PAGE asks (the allocation quantum — decode
            // page-boundary asks, 87% of parks) are counted: the runway
            // exists for the small-ask stream. Summing larger asks ordered
            // overlapping rounds in tiny-pool regimes where the demand can
            // never be co-resident (onefits: 58 starvation kills). With no
            // quantum asks queued this reduces exactly to the previous
            // head-shortfall predicate.
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
            // The classic liveness form: the head's own shortfall, with the
            // pool empty.
            free == 0
                && inner
                    .unmet_head()
                    .is_some_and(|(_, head)| head.kv_need() > inner.accum.len() as u32 + expected)
        })
    }

    /// No completion can ever arrive on its own — see
    /// [`Inner::fleet_stalled`]. The starvation rung's precondition, and
    /// the eviction rung's last resort, so the two can never disagree
    /// about when waiting is still an option.
    fn is_wedged(&self) -> bool {
        self.with_inner(|inner| inner.fleet_stalled())
    }

    /// Last-resort victim search, immediately before the starvation rung
    /// would destroy a request — decided from ONE atomic snapshot.
    ///
    /// This is the shape `rainer_v3.md` §8.3 argues the whole planner
    /// should have. The stale-snapshot defects (§18.9, §18.10) all trace to
    /// one cause: `plan_eviction` must release the planner lock to quote,
    /// because reclaimability lives behind the KV store lock. Candidate set
    /// at T0, quotes at T1, decision at T2 — and a process that changed
    /// state in between is invisible or misjudged.
    ///
    /// Here the store lock is taken FIRST and the planner lock inside it,
    /// which is the tree's documented order (`inner` is innermost), so the
    /// procs, the queue and the quotes are all read at one instant and the
    /// decision cannot be overtaken. Working sets are gathered before
    /// either lock: `RESIDENCIES` is acquired before the KV lock
    /// everywhere, and inverting that would risk deadlock.
    ///
    /// A process registered after the gather cannot invalidate the result:
    /// it starts with no pages, and it cannot acquire any while this holds
    /// the planner lock.
    ///
    /// Runs only in the wedge, so the (rare) KV-lock hold does not touch
    /// the convoy §16 measured on the per-fire path.
    ///
    /// Returns whether an eviction was spawned.
    fn last_resort_evict(self: &Arc<Self>) -> bool {
        let (model, driver) = self.port.locus();
        let Some(stores) = crate::store::registry::try_get(model, driver) else {
            return false;
        };
        // Outside every lock: which processes might we quote?
        let pids: Vec<ProcessId> = self.with_inner(|inner| inner.procs.keys().copied().collect());
        if pids.is_empty() {
            return false;
        }
        let working_sets =
            crate::inferlet::process::residency::kv_working_sets_for(&pids, model, driver);

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
                    // Every pid, not a deficit-bounded prefix: the legality
                    // filter below is planner state the quote order knows
                    // nothing about, so a truncated list would silently drop
                    // the only legal victim.
                    let quotes = crate::inferlet::process::residency::quote_locked(
                        kv,
                        working_sets,
                        u32::MAX,
                    );
                    // Legal victims: younger than the head and resident.
                    // E6 hysteresis is waived here and ONLY here — it is a
                    // preference, never a reason to let a request die
                    // (§18.9). Order: E6-fresh first, then youngest-first.
                    //
                    // A host-swap-blocked victim is NOT waived: unlike E6
                    // that is a physical impossibility, not a preference.
                    // Re-picking one here is what kept this rung "succeeding"
                    // at dispatching an eviction that instantly rolled back —
                    // so `starved` never advanced and the kill that would
                    // have broken the jam never fired (§20.6). When the block
                    // empties this set, the rung falls through to the kill,
                    // which is the correct terminal behaviour.
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
    /// into the head's accumulation. Returns whether anything was salvaged.
    ///
    /// The starvation rung runs on a state that is one `plan()` iteration
    /// old: the drain only reaches it because `reserve_device_up_to` came
    /// back empty, and everything after that — `is_wedged`, the exhaustive
    /// `last_resort_evict` scan under the global KV lock, the trace dump —
    /// is a window in which a retiring process can hand its pages back. A
    /// process leaves `procs` when it unregisters, but its page leases drop
    /// afterwards, so the wedge predicate goes TRUE strictly before the pool
    /// refills: the rung is at its most trigger-happy exactly when relief is
    /// arriving. Observed in the parity matrix at 32x oversubscription,
    /// where a 4-page ask was destroyed with 35 of 69 pages free
    /// (`PlannerError::Starved` even prints the contradiction).
    ///
    /// Re-running the drain's own primitive rather than comparing counters
    /// is what makes this terminating: `reserve_device_up_to` cannot
    /// disagree with the absorb step, so a salvage always strictly shrinks
    /// the free list, and a genuinely empty pool still falls through to
    /// destruction on this same pass.
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

    /// Last rung before destruction: serve the OLDEST queued ask that the
    /// head's own hoard already covers, head-of-line order be damned.
    ///
    /// `plan` accumulates head-first, so an uncoverable head pulls every
    /// free page into `accum` and sits on it. When the pool can only ever
    /// be completed by the processes currently holding it, those hoarded
    /// pages are exactly what those holders need to finish and release.
    /// Measured (§20.17): four processes holding 61 pages each of a
    /// 256-page pool parked asking for ONE page apiece, while a 61-page
    /// head hoarded the 12 pages that would have served all four — and the
    /// starvation rung then destroyed a request over a pool that was
    /// fundable all along. The queue was wedged by FCFS alone, not by
    /// exhaustion.
    ///
    /// Strict FCFS is worth an unbounded wait; it is not worth a destroyed
    /// request. So the inversion is admitted HERE and nowhere else: only
    /// inside the wedge, only once the head is proven uncoverable by
    /// salvage and eviction both, and only for an ask the hoard already
    /// covers in full. Every earlier rung keeps its exact behaviour, and
    /// the only path this can divert is one that would otherwise return a
    /// `Starved` error.
    ///
    /// It cannot spin: each bypass moves one waiter to SERVED (which also
    /// un-wedges the predicate until it is collected) and strictly shrinks
    /// the hoard, so at most `accum.len()` asks can be diverted before the
    /// rung stops finding one and destruction proceeds as before. A served
    /// waiter runs, completes and returns at least what it took.
    ///
    /// Restores are deliberately NOT bypassable: boarding one is a
    /// multi-step handoff through `spawn_restore`, and the measured wedge
    /// is an allocation wedge. A queue of pure restores falls through to
    /// destruction exactly as it does today.
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
        // Mirrors `Step::ServeAllocation`: re-validate under the lock, and
        // hand a rejected RS reservation back OUT so it is dropped (which
        // takes the store lock) outside the planner lock.
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

    /// The RS analogue of [`Self::check_starvation`], for the one resource the
    /// eviction ladder cannot fund. A folded slot is released by a process
    /// finishing work, so the wedge predicate — every admitted process parked —
    /// is exactly "no slot can ever come back". Reached only from the drain's
    /// failed `reserve_rs`, so the common case (a transiently empty pool with
    /// somebody still running) still just waits.
    ///
    /// The head is failed, not the youngest: destroying a younger waiter frees
    /// no slot, because slots are held by RUNNING requests rather than by
    /// queued ones. This is the deployment asking for more concurrent state
    /// than the pool has, and saying so is the only honest move.
    fn check_rs_starvation(self: &Arc<Self>, key: EntryKey, demand: Demand) {
        if !self.is_wedged() {
            return;
        }
        // The predicate was evaluated after a reservation that already failed,
        // so re-read the pool before destroying anything: a slot freed in
        // between makes the whole question moot.
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
        // The wedge predicate is evaluated after the drain's absorb already
        // failed, so it can be stale in the one direction that matters:
        // pages back in the pool. Salvage before evicting anyone.
        if self.salvage_free_pages() {
            self.poke();
            return;
        }
        // Last rung before destruction: re-scan for ANY evictable victim
        // younger than the head, at THIS instant rather than at the
        // caller's phase-1 sample (§18.10).
        //
        // Only for `NoEligibleVictim`. Under `NoSwapRoom` there is nowhere
        // to evict INTO, so a pick would be marked `Evicting`, fail in the
        // executor and roll back, forever — the rung must not run when the
        // swap channel is the thing that is gone.
        if cause == StarveCause::NoEligibleVictim && self.last_resort_evict() {
            return;
        }
        // The relaxed scan may itself have raced; re-verify before killing.
        if !self.is_wedged() {
            return;
        }
        // Final salvage, immediately before destruction. `last_resort_evict`
        // walks every proc under the global KV lock, which is ample time for
        // a teardown to land.
        if self.salvage_free_pages() {
            self.poke();
            return;
        }
        // Nothing can be assembled for the head. Before destroying anyone,
        // check whether the head's own hoard already covers a younger ask —
        // the head-of-line wedge (§20.17). Only reachable on a path that
        // would otherwise return `Starved`.
        if self.serve_from_hoard() {
            return;
        }
        let (free, total) = self.port.device_stats();
        // A destruction already ordered has not been paid out yet: gate the
        // rung that DESTROYS, and only that one. The cheaper rungs above
        // (salvage, last-resort eviction) stay open — they are what we would
        // rather have happen — and eviction's own load control keeps using
        // the wedge predicate unchanged.
        if self.with_inner(|inner| inner.kill_in_flight()) {
            return;
        }
        // Pick the victim OUTSIDE the lock: the choice needs reclaim quotes,
        // and quoting takes store locks (`quote_and_pick` has the same rule).
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
            // Reclaiming this process's pages is not the same as failing its
            // request. If the guest declared itself restartable, the work is
            // re-queued at the same FCFS position instead of being lost; the
            // caller sees one slower reply rather than an error. A guest that
            // made no such declaration keeps today's fail-loud behaviour,
            // because re-running it could duplicate its side effects.
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

    /// Choose which parked ask to destroy, youngest-first, **restricted to
    /// asks whose destruction actually returns pages to the pool**.
    ///
    /// A parked process that holds nothing is not a victim, it is a queue
    /// entry: it is parked on its FIRST allocation, so it occupies no pool
    /// capacity and killing it frees exactly zero pages. The wedge therefore
    /// survives its death and the rung fires again on the next-youngest —
    /// the destruction cascade. Measured on a 3x-oversubscribed pool
    /// (§20.40): 752 wedge kills, **737 of them (98%) on
    /// `NoReclaim::HoldsNothing`**, while the ~190 processes that actually
    /// held the whole pool were never touched. 752 of 1024 requests were
    /// destroyed to reclaim nothing.
    ///
    /// This is the same lesson [`NoReclaim::HoldsNothing`] already carries
    /// ("cannot become reclaimable by waiting — only by ALLOCATING; this is
    /// the case that livelocked the ladder") and that [`Self::quote_and_pick`]
    /// already applies to EVICTION victims. It was simply never applied at
    /// the destruction site.
    ///
    /// Quoting takes store locks, so the pick happens outside the planner
    /// lock and the caller re-validates the key under it.
    ///
    /// Fallback: if no parked ask holds anything, the wedge cannot be broken
    /// by any choice, so keep the original rule (youngest) rather than lose
    /// the deadlock breaker.
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
        let (model, driver) = self.port.locus();
        // Two passes over the same youngest-first order. A restartable
        // process loses only time when it is reclaimed — its work is
        // re-queued — while a non-restartable one loses the request itself.
        // So spend the restartable ones first and only reach for a real
        // failure when no restartable holder can break the wedge. Within
        // each pass the youngest-first rule is unchanged.
        let mut first_holder = None;
        for restartable_only in [true, false] {
            // One call for the whole fleet, not chunks: see
            // `quote_and_pick` for why chunking multiplied the cost of the
            // group-independent census instead of bounding it.
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
                crate::inferlet::process::residency::kv_reclaim_quotes(&pids, model, driver, 1);
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

    /// The hog endgame, as a computed predicate: nothing younger can fund
    /// the head, no transfer is in flight, and the head's own holdings plus
    /// its ask exceed the pool — no eviction of OTHERS can ever cover it.
    /// Fail the head loud. Anything short of that is just "wait for the
    /// next completion": elders' frees arrive as `pages_freed` pokes.
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
        let (model, driver) = self.port.locus();
        // The head's OWN footprint: resident + swapped, pinned or shared
        // alike. This must be the durable `held_page_count`, never a
        // `ReclaimQuote` — `pages()` reports 0 for every `Nothing` variant,
        // so reading it here silently under-counted a head holding the whole
        // pool to 0 the moment one in-flight pin overlapped, and the hog
        // endgame never fired (`rainer_v3.md` §3.3).
        let held = held_page_count(head_pid, model, driver);
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

    // =========================================================================
    // Executor callbacks (planner::exec)
    // =========================================================================

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

    /// `prepare_suspend` deferred: this victim has nothing movable at this
    /// instant. Park it for the same reason `HostSwapFull` is parked — the
    /// re-pick is deterministic — and additionally because a rollback loop
    /// keeps `evicting` non-empty, which gates out the starvation rung that
    /// is the designed terminal answer to an unfundable fleet.
    fn eviction_failed_prepare_deferred(self: &Arc<Self>, pid: ProcessId) {
        self.eviction_failed_inner(pid, false, true);
    }

    /// `HostSwapFull`: this victim's bytes have nowhere to go. Victim
    /// selection is deterministic, so an immediate re-plan re-picks the same
    /// process and re-runs the entire fence → suspend-notify → drain →
    /// quiesce → prepare cycle only to fail identically — a hot livelock at
    /// ~7.5k iterations/s that also churns the frame policy through
    /// `notify_process_suspend` on every turn (§20.6). Park the victim until
    /// host room actually returns; the poke still fires so a SMALLER victim
    /// that does fit can be tried, and once every candidate is parked the
    /// set empties and the starvation rung — the designed last resort —
    /// takes over.
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
        // The victim never left, so undo the suspend's wait-set effects:
        // without this it would stay marked suspended and keep running with
        // its frames cut to single slots (§20.8).
        crate::scheduler::worker::notify_process_resume(pid);
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        self.poke();
    }

    /// Host slots came back, so every host-swap-blocked victim is a
    /// candidate again. Clearing wholesale (rather than per-pid accounting)
    /// keeps the set from leaking a victim that would otherwise stay parked
    /// forever, and costs at most one wasted eviction attempt per real host
    /// release.
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

    /// H2D committed (or nothing left to move): `pid` is resident again —
    /// its working-set fences drop, then gate waiters and out-of-set
    /// acquires wake. Owning the unfence HERE, not at each restore-executor
    /// exit path, makes "restored ⇒ unfenced" structural: a forgotten
    /// unfence is a silent per-process wedge (every fire bounces off
    /// `Fenced` forever, with no rung to clear it). Failure paths keep the
    /// fences up by simply never reporting a restore.
    fn report_restored(self: &Arc<Self>, pid: ProcessId, restored: u32) {
        // The restored pages' host slots are back in the swap pool: every
        // victim parked on `HostSwapFull` may now fit.
        self.clear_host_swap_blocks();
        let (model, driver) = self.port.locus();
        for handle in crate::inferlet::process::residency::kv_suspend_handles(pid, model, driver) {
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

    /// A restore broke somewhere that says nothing about what the transfer
    /// left behind — a committed page table, or an executor that died
    /// mid-sequence. There is no clean evicted state to return to, so the
    /// process is failed loud through the runtime abort path.
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

    /// A restore broke before anything was committed: it either never
    /// started, or `abort_now` rolled it back. Either way the process is
    /// exactly where it was — evicted, holding nothing — so it goes back in
    /// line at its spawn position rather than dying for it.
    ///
    /// Once per episode, and deliberately not a tunable. The re-serve
    /// rebuilds the ask from what is swapped *now* (`Step::ServeRestore`), so
    /// a short grant cannot be short twice; anything that breaks again breaks
    /// for a reason a third attempt would not fix.
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

    // =========================================================================
    // Telemetry
    // =========================================================================

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
        // The entry the DRAIN is actually blocked on. `queue.first()` is
        // merely the oldest entry, met or not, so reporting it made a
        // served-but-uncollected grant look like a stalled head.
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
        // What the head-first rule costs right now: walk PAST the blocked
        // head and see how much of the queue the free stock alone would
        // already cover. Nothing is served here — this only measures.
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
        // the KV store, both of which order BEFORE `inner`.
        let (model, driver) = self.port.locus();
        let runners = runner_ids
            .into_iter()
            .map(|(pid, seq, progressed)| (seq, held_page_count(pid, model, driver), progressed))
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

/// Saturating microseconds for the stat counters.
/// `PIE_ALLOC_FAST_SMALL=1`: allow the head-harmless free-list bypass in
/// `acquire` (see the comment at its use site). Default off.
fn fast_small_bypass_enabled() -> bool {
    static CONFIGURED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CONFIGURED.get_or_init(|| {
        std::env::var("PIE_ALLOC_FAST_SMALL").is_ok_and(|value| !value.is_empty() && value != "0")
    })
}

/// Probe-only: `PIE_ROTATION_DAMPING=0` disables the completion-credit gate
/// on covered restores (rotation damping, commit fa7f3adf5). Default ON —
/// the damping is a shipped win at heavy oversubscription; the toggle
/// exists to measure its sign at mild oversubscription (8192-page D cell),
/// where parked boundary asks wait ~a full completion interval for supply
/// an eviction-funded restore could have covered sooner.
fn rotation_damping_enabled() -> bool {
    static CONFIGURED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CONFIGURED.get_or_init(|| {
        !std::env::var("PIE_ROTATION_DAMPING").is_ok_and(|value| value == "0")
    })
}

/// `PIE_SUPPLY_RUNWAY=<pages>`: the supply-phase runway target (§10.14).
/// When nonzero the planner treats the free list itself as a demand: it
/// starts and sizes eviction rounds so `free` is pulled back toward this
/// many device pages, even with an empty ask queue — supply runs ahead of
/// the decode boundary-ask stream instead of reacting to it. Absent,
/// empty, `0`, or unparseable = off (behavior identical to before the
/// runway existed).
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

/// The process's swapped page count on `(model, driver)` — the exact restore
/// ask, computed fresh at serve time.
fn swapped_page_count(pid: ProcessId, model: usize, driver: usize) -> u32 {
    kv_page_count(pid, model, driver, |kv, ws| {
        kv.swapped_page_count(ws).unwrap_or(0)
    })
}

/// Every device+host page this process holds — the DURABLE fact the
/// liveness predicates need, never a [`ReclaimQuote`] (`rainer_v3.md` §3.3).
fn held_page_count(pid: ProcessId, model: usize, driver: usize) -> u32 {
    kv_page_count(pid, model, driver, |kv, ws| {
        kv.held_page_count(ws).unwrap_or(0)
    })
}

fn kv_page_count(
    pid: ProcessId,
    model: usize,
    driver: usize,
    count: impl FnOnce(
        &crate::store::kv::KvStore,
        &std::collections::HashSet<crate::store::kv::page_table::WorkingSetId>,
    ) -> usize,
) -> u32 {
    let working_sets = crate::inferlet::process::residency::kv_working_set_ids(pid, model, driver);
    if working_sets.is_empty() {
        return 0;
    }
    let Some(stores) = crate::store::registry::try_get(model, driver) else {
        return 0;
    };
    crate::store::registry::with_kv_lock(&stores.kv, "planner-held-pages", |kv| {
        count(kv, &working_sets) as u32
    })
}

// =============================================================================
// Registry-owned instances, per (model, driver)
// =============================================================================

type PlannerMap = HashMap<(usize, usize), Arc<ResidencyPlanner>>;

static PRIMARY: OnceLock<Arc<ResidencyPlanner>> = OnceLock::new();
static PLANNERS: OnceLock<RwLock<PlannerMap>> = OnceLock::new();

/// Install the planner for `(model, driver)` (bootstrap, once per pair).
pub fn init_planner(model: usize, driver: usize, planner: ResidencyPlanner) {
    let planner = Arc::new(planner);
    // E5: the drain runs on one dedicated task from here on (bootstrap is
    // inside the runtime); a planner constructed outside a runtime keeps
    // inline planning.
    planner.arm_drain_task();
    if model == 0 && driver == 0 {
        let _ = PRIMARY.set(planner.clone());
    }
    PLANNERS
        .get_or_init(Default::default)
        .write()
        .unwrap()
        .insert((model, driver), planner);
}

/// The planner for `(model, driver)`.
pub fn planner_for(model: usize, driver: usize) -> Option<Arc<ResidencyPlanner>> {
    if model == 0 && driver == 0 {
        return PRIMARY.get().cloned();
    }
    PLANNERS
        .get()?
        .read()
        .unwrap()
        .get(&(model, driver))
        .cloned()
}

/// The (0, 0) planner — the hot-path shorthand while process-side call sites
/// are still hardwired to driver 0. `None` only before bootstrap has run.
pub fn planner() -> Option<&'static Arc<ResidencyPlanner>> {
    PRIMARY.get()
}

/// Service order across the allocation/restore boundary — [`Inner::unmet_head`]
/// is the single place that decides it, so these drive it directly.
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

    /// The ping-pong regression. An evictee carries an OLD spawn seq back
    /// into the queue, so pure FCFS hands it the head ahead of the residents
    /// whose eviction paid for those pages — measured on D/512 as 993
    /// evictions against 929 restores, net 64 processes out, `free` pinned
    /// at 0. Only a resident can complete, so the resident's ask goes first.
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

    /// The safety valve. Yielding forever would deadlock a fleet whose
    /// residents are all parked on something an evictee owns, so once no
    /// completion can arrive on its own the head goes back to spawn order.
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

    /// An eviction in flight is a completion on its way, so the fleet has
    /// not stalled and the valve stays shut.
    #[test]
    fn an_eviction_in_flight_keeps_the_restore_yielding() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Evicted, true),
            (3, Residency::Resident, true),
        ]);
        park_restore(&mut inner, pids[0], 1, 18);
        park_allocation(&mut inner, pids[1], 3, 1);
        inner
            .evicting
            .insert(ProcessId::new_v4(), EvictionMark { pages: 26 });

        assert!(!inner.fleet_stalled(), "a transfer is already in flight");
        assert_eq!(inner.unmet_head().expect("a head").0.0, 3);
    }

    /// Nothing to yield TO: a served-but-uncollected grant has stopped
    /// competing for pages, so the restore boards on its own order without
    /// needing the stall valve at all.
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

    /// The second valve. The fleet is live — a resident is running, so
    /// `fleet_stalled` is false and the first valve stays shut — but the
    /// host pool that funds eviction is out, and only a restore returns a
    /// host slot. Yielding here serves nobody: the allocation it yields to
    /// can only be funded by an eviction that cannot be paid for. On `soak`
    /// this ran for 60 s and cost 21895 rollbacks and 138 starvation kills.
    #[test]
    fn a_restore_takes_the_head_once_eviction_cannot_be_funded() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Evicted, true),
            (2, Residency::Resident, true),
            (3, Residency::Resident, true),
        ]);
        park_restore(&mut inner, pids[0], 1, 18);
        park_allocation(&mut inner, pids[2], 3, 1);
        assert!(!inner.fleet_stalled(), "a resident is still running");
        assert_eq!(
            inner.unmet_head().expect("a head").0.0,
            3,
            "with host room the yield stands"
        );

        inner.host_swap_blocked.insert(pids[1]);

        assert!(inner.eviction_unfundable());
        let (key, waiter) = inner.unmet_head().expect("a head");
        assert_eq!(key.0, 1, "the older restore must take the head");
        assert!(matches!(waiter.kind, WaitKind::Restore { .. }));
    }

    /// And it is a live signal, not a latch: once host room returns the
    /// blocked set is cleared and the yield resumes, so the fix cannot
    /// re-introduce the evict/restore ping-pong it is bounding.
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

    /// A pool whose free list refills *after* the drain's absorb has already
    /// come back empty — the production race, made deterministic.
    ///
    /// `stall` is the number of leading `reserve_device_up_to` calls that
    /// report an empty pool regardless of `free`. One stall reproduces the
    /// real ordering exactly: a retiring process leaves `procs` (so the
    /// wedge predicate goes true) before its page leases actually drop.
    struct RacePool {
        free: parking_lot::Mutex<Vec<PhysicalKvPageId>>,
        total: u32,
        stall: AtomicU32,
        /// Host swap room still free. Zero (the default) keeps
        /// `plan_eviction` on its `NoSwapRoom` short-circuit, which is what
        /// every starvation-rung test wants; non-zero lets a test reach the
        /// load-control gate.
        host: u32,
        /// Every `reserve_device_up_to` request, in order — how the drain
        /// ASKED for its supply, which is what tells one pull for a whole
        /// burst apart from one pull per waiter.
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

        /// A swap-capable pool whose host room is already down to `host` of
        /// `host_total`, so the HOST RESERVE gate can be exercised without
        /// emptying the pool (which is the separate `NoSwapRoom` arm).
        fn with_host_pressure(total: u32, host: u32, host_total: u32) -> Self {
            Self {
                host,
                host_total,
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
            // Without host room: no swap transport, so `plan_eviction` goes
            // straight to `check_starvation(NoSwapRoom)` — the shortest path
            // to the starvation rung, and it skips `last_resort_evict`
            // (which needs the real residency tables).
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

    /// A completion's unpark burst is served in ONE pass: the drain funds
    /// the whole fundable FCFS prefix under a single lock hold off a single
    /// free-list pull, and stops exactly where the supply runs out.
    ///
    /// The serialization this pins down was the contended decode cell's
    /// remaining throughput gap: each completion freed ~35 pages and
    /// un-parked ~32 one-page asks, and the drain served them nearly one at
    /// a time (p50 54 µs between serves, ~4.3 ms for the burst) because
    /// every waiter cost its own step decision, its own free-list round
    /// trip and its own serve re-validation — three planner-lock cycles,
    /// each recomputing the lock-free mirrors over every process, all while
    /// the just-woken collectors contended for the same mutex. The wave seal
    /// waits for the last unparked lane, so the device idled through all of
    /// it.
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
                .map(|key| match &inner.queue.get(key).expect("still queued").kind {
                    WaitKind::Allocation { outcome, .. } => outcome.is_some(),
                    WaitKind::Restore { .. } => unreachable!(),
                })
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
        // The served set above is what the two paths AGREE on — the burst is
        // a pure latency fix. What changes is the shape of the pass: how many
        // times the drain had to leave the lock and go back to the pool.
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

    /// The starvation rung must not destroy a request while the pool holds
    /// enough free pages to fund the head.
    ///
    /// Before the salvage step this failed the youngest ask with
    /// `Starved { need: 1, free: 4, total: 4 }` — the parity matrix hit the
    /// same contradiction on real hardware at 32x oversubscription
    /// ("4 pages asked, 35 free of 69").
    #[tokio::test(flavor = "current_thread")]
    async fn starvation_rung_does_not_kill_while_the_pool_can_fund_the_head() {
        let pool = Arc::new(RacePool::new(4));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone()));

        // A running process owns the whole pool. While it is registered and
        // unparked the wedge predicate is false, so the two asks below park
        // instead of being destroyed on arrival.
        let holder = ProcessId::new_v4();
        planner.register(holder);
        planner.note_admitted(holder);

        // Two admitted processes park on an empty pool: an older head and a
        // younger ask (the one the rung destroys).
        let head = ProcessId::new_v4();
        let young = ProcessId::new_v4();
        for pid in [head, young] {
            planner.register(pid);
            planner.note_admitted(pid);
        }

        let demand = Demand {
            kv_pages: 1,
            rs_slots: 0,
        };
        let (p1, p2) = (planner.clone(), planner.clone());
        let head_task = tokio::spawn(async move { p1.acquire(head, head, demand).await });
        let young_task = tokio::spawn(async move { p2.acquire(young, young, demand).await });

        // Let both park before the pool refills.
        for _ in 0..200 {
            tokio::task::yield_now().await;
            if planner.diagnostics().queue.len() == 2 {
                break;
            }
        }
        assert_eq!(planner.diagnostics().queue.len(), 2, "both asks must park");

        // The holder retires. Its pages are back in the pool, but the very
        // next reservation still reports empty — the production ordering,
        // where `unregister` publishes before the page leases drop. That is
        // enough for `unregister`'s own poke to reach the starvation rung
        // with a fundable pool underneath it.
        pool.refill_after_stall(4, 1);
        planner.unregister(holder);

        let one = std::time::Duration::from_secs(5);
        let a = tokio::time::timeout(one, head_task).await;
        let b = tokio::time::timeout(one, young_task).await;

        let diagnostics = planner.diagnostics();
        assert_eq!(
            diagnostics.starvations_total, 0,
            "no ask may be destroyed while the pool has free pages \
             (free={} total={})",
            diagnostics.device_pages_free, diagnostics.device_pages_total
        );
        assert!(
            diagnostics.salvages_total > 0,
            "the rung should have salvaged the refilled pool"
        );
        for (name, result) in [("head", a), ("young", b)] {
            let granted = result
                .unwrap_or_else(|_| panic!("{name} timed out"))
                .expect("task panicked");
            assert!(
                matches!(granted, Ok(Acquired::Granted(_))),
                "{name} should have been granted, got {:?}",
                granted.err()
            );
        }
    }

    /// REGRESSION (the host-swap eviction livelock, §20.6).
    ///
    /// `HostSwapFull` leaves the victim resident, and victim selection is a
    /// deterministic FCFS scan — so the poke that follows the rollback
    /// re-picks the *same* process and re-runs the whole fence →
    /// suspend-notify → drain → quiesce → prepare cycle only to fail
    /// identically. Observed at ~7.5k iterations/s on one pid for 150 s:
    /// 4.8M `planner-exec` step lines, `evict_rollbacks` past 1.2M, and
    /// `serves` frozen — a hot livelock that ALSO disarms the starvation
    /// rung, because a perpetually in-flight eviction keeps `is_wedged`
    /// false.
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

    /// A restart must re-enter the FCFS clock at the position it lost, not
    /// as the newest process.
    ///
    /// This is the liveness argument for restart-instead-of-kill in one
    /// assertion. Starvation victims are chosen youngest-first, so a restart
    /// that took a fresh seq would be the immediate next victim and the
    /// request would be reclaimed forever. Inheriting the seq means the
    /// re-run ages exactly as the original would have, reaches the head, and
    /// the head is never a victim.
    #[test]
    fn a_restart_inherits_its_fcfs_position() {
        let pool = Arc::new(RacePool::new(4));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));

        let victim = ProcessId::new_v4();
        let younger = ProcessId::new_v4();
        planner.register(victim);
        planner.register(younger);
        let victim_seq = planner.spawn_seq(victim).expect("registered");
        assert!(
            victim_seq < planner.spawn_seq(younger).expect("registered"),
            "registration order must be the seq order"
        );

        // The victim is reclaimed and re-spawned under a new pid.
        planner.unregister(victim);
        assert_eq!(planner.spawn_seq(victim), None);
        let restarted = ProcessId::new_v4();
        planner.register_with_seq(restarted, victim_seq);

        assert_eq!(
            planner.spawn_seq(restarted),
            Some(victim_seq),
            "the re-run keeps the position it lost"
        );
        assert!(
            planner.spawn_seq(restarted) < planner.spawn_seq(younger),
            "the re-run must still outrank a process that arrived after it"
        );

        // The clock itself is not rewound: anything registering later still
        // sorts behind both, so seqs stay a total arrival order.
        let newest = ProcessId::new_v4();
        planner.register(newest);
        assert!(
            planner.spawn_seq(newest) > planner.spawn_seq(younger),
            "a fresh registration must remain the youngest"
        );
    }

    /// Load control: an EXHAUSTED pool with no transfer in flight evicts,
    /// even though admitted processes are still running.
    ///
    /// This inverts what the rung used to do. Gating on the wedge predicate
    /// — false while ANY admitted process is unparked — made the rung
    /// unreachable exactly when it is needed: at 512-way contention every
    /// resident holds a partial working set, so nobody can finish, nobody
    /// frees, and "its pages are already on their way back" is false for
    /// every one of them. See `plan_eviction`'s LOAD CONTROL note.
    #[tokio::test(flavor = "current_thread")]
    async fn an_exhausted_pool_evicts_even_while_the_fleet_still_runs() {
        // Non-zero capacity with an empty free list: an ask bigger than the
        // pool is failed loud instead of parking.
        let pool = Arc::new(RacePool::with_swap(4, 64));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone()));
        // §10.14: pin the runway latch closed so the DEMAND-EXACT arithmetic
        // stays under test even when the verification environment exports
        // PIE_SUPPLY_RUNWAY (the env read is a process-global OnceLock, so a
        // test cannot scope it). With the runway off the latch is never
        // read; nothing in this test settles an eviction, so it stays
        // pinned. The runway's own arithmetic is covered separately by
        // `a_runway_shortfall_starts_a_round_with_an_empty_queue`.
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

        // `holder` is unparked, so the fleet is NOT wedged — and the rung
        // must run anyway, because an exhausted pool with nothing in flight
        // cannot produce the head's page by waiting. (It gets no further
        // than the victim quotes here: `kv_reclaim_quotes` reads the real
        // residency registry, which a planner-only fixture never fills.)
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

    /// §10.14 supply-phase runway: a free-list shortfall ALONE — empty ask
    /// queue, nobody parked — must be able to open an eviction round, and
    /// the round must respect its three brakes: the one-round hysteresis
    /// latch, the completion-credit check, and the oldest-resident floor
    /// (the clamp that keeps a onefits-sized pool from being evicted down
    /// to nobody).
    ///
    /// Exercised through `runway_shortfall_set` with an explicit target:
    /// the env read is a process-global OnceLock, so a test cannot set and
    /// unset `PIE_SUPPLY_RUNWAY` per-case; `plan_runway` is that read plus
    /// the transport gate in front of this exact arithmetic. (The commit
    /// half gets no further than the victim quotes in this fixture — the
    /// real residency registry is empty — same limit as the load-control
    /// test above.)
    #[test]
    fn a_runway_shortfall_starts_a_round_with_an_empty_queue() {
        let pool = Arc::new(RacePool::with_swap(4, 64));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));

        // Two admitted residents; the free list and the queue are empty.
        let elder = ProcessId::new_v4();
        let younger = ProcessId::new_v4();
        for pid in [elder, younger] {
            planner.register(pid);
            planner.note_admitted(pid);
        }

        let (deficit, members) = planner
            .runway_shortfall_set(32, 0)
            .expect("a shortfall with an empty queue must open a round");
        assert_eq!(deficit, 32, "the round is sized to the whole shortfall");
        assert_eq!(
            members.iter().map(|&(pid, _)| pid).collect::<Vec<_>>(),
            vec![younger],
            "the oldest resident is never a runway victim"
        );

        // In-flight eviction yield is supply on its way — same arithmetic
        // as `supply_stalled` — so it shrinks the deficit page for page.
        let mark = ProcessId::new_v4();
        planner.with_inner(|inner| {
            inner.evicting.insert(mark, EvictionMark { pages: 30 });
        });
        let (deficit, _) = planner
            .runway_shortfall_set(32, 0)
            .expect("a two-page residue still opens a round");
        assert_eq!(deficit, 2, "in-flight yield funds the runway first");
        planner.with_inner(|inner| {
            inner.evicting.remove(&mark);
        });

        // Hysteresis: one runway-motivated round in flight at a time.
        planner.with_inner(|inner| inner.runway_round_in_flight = true);
        assert!(
            planner.runway_shortfall_set(32, 0).is_none(),
            "no second runway round until the first settles"
        );

        // A banked completion credit does NOT stand the runway down:
        // measured on the D cell, completions bank credits continuously and
        // a credit gate held the runway at zero rounds for whole runs.
        // FCFS already arbitrates the yield between queued boundary asks
        // and the victim's (still credit-gated) restore.
        planner.with_inner(|inner| {
            inner.runway_round_in_flight = false;
            inner.completion_credit = 1;
        });
        assert!(
            planner.runway_shortfall_set(32, 0).is_some(),
            "a banked credit must not veto the runway (D-cell measurement)"
        );
    }

    /// HOST RESERVE: the rung may not spend the last of the host pool on a
    /// fleet that is still running.
    ///
    /// Load control asks whether waiting can produce the next page. It does
    /// not ask what the eviction is PAID for. A saturated fleet has `free`
    /// at 0 by definition, so its head is permanently unmet and the gate is
    /// permanently open — which drained the host pool to nothing and then
    /// handed the stranded processes to the starvation rung, one rung below
    /// the `host_free == 0` arm. See `plan_eviction`'s HOST RESERVE note
    /// for the `soak` measurement.
    ///
    /// Two fixtures identical but for host room, so the reserve is the only
    /// thing that can differ in the outcome. That the reserve is still
    /// SPENT by a wedge is the `!is_wedged()` clause in the gate, and the
    /// suite's wedge scenarios (`tinyswap`, `tinyswap_thrash`, `onefits`)
    /// are what hold it: all of them still reach eviction.
    #[tokio::test(flavor = "current_thread")]
    async fn a_running_fleet_may_not_spend_the_last_of_the_host_pool() {
        for host_free in [8u32, 64] {
            let under_reserve = host_free * HOST_RESERVE_DIVISOR <= 128;
            assert_eq!(under_reserve, host_free == 8, "the fixtures straddle it");

            // Host room down to `host_free` of 128, and never zero: the
            // `NoSwapRoom` arm is a different gate and must not be what
            // fires here.
            let pool = Arc::new(RacePool::with_host_pressure(4, host_free, 128));
            let planner = Arc::new(ResidencyPlanner::new(pool.clone()));

            // An unparked admitted runner: the fleet is live, so the head's
            // ask parks instead of being destroyed on arrival.
            let holder = ProcessId::new_v4();
            planner.register(holder);
            planner.note_admitted(holder);

            let head = ProcessId::new_v4();
            planner.register(head);
            planner.note_admitted(head);
            // Younger, resident, admitted: legal victims, so a set exists
            // and only the gate can hold the round back.
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
            assert_eq!(planner.diagnostics().queue.len(), 1, "the ask must park");

            // Everything load control looks at says "evict": empty pool,
            // unmet head, nothing in flight, fleet not wedged. Only the
            // reserve is left to tell the two rounds apart.
            assert!(planner.supply_stalled(), "an empty pool with an unmet head");
            assert!(!planner.is_wedged(), "a running holder is not a wedge");

            let before = planner.diagnostics().eviction_deferrals_total;
            planner.plan_eviction();
            let deferred = planner.diagnostics().eviction_deferrals_total - before;
            assert_eq!(
                deferred,
                u64::from(under_reserve),
                "a live fleet under the reserve waits for the completion it \
                 will deliver, and evicts as usual above it (host_free={host_free})"
            );
            assert_eq!(
                planner.diagnostics().starvations_total,
                0,
                "and deferring is never starving"
            );

            parked.abort();
        }
    }

    /// The destruction guard clears on BOTH of its exits, so it can never
    /// wedge the deadlock breaker it rate-limits.
    ///
    /// Ordering a kill only wakes the victim; its pages return when its
    /// store drops, a WASM unwind later. Without a marker for that window
    /// every planner poke ordered another destruction — 250 kills/s on
    /// D/512, 3590 for 768 requests, resident set unchanged. The marker is
    /// only safe if it lifts by itself, which is what this pins down.
    #[tokio::test(flavor = "current_thread")]
    async fn an_ordered_kill_stops_counting_once_the_victim_dies_or_re_contends() {
        let pool = Arc::new(RacePool::new(4));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone()));

        // A running holder keeps the wedge predicate false throughout, so
        // the rung under test is never entered for real.
        let holder = ProcessId::new_v4();
        planner.register(holder);
        planner.note_admitted(holder);

        let victim = ProcessId::new_v4();
        planner.register(victim);
        planner.note_admitted(victim);
        planner.with_inner(|inner| inner.killing.insert(victim));
        assert!(
            planner.with_inner(|inner| inner.kill_in_flight()),
            "an ordered destruction counts until it is paid out"
        );

        // Exit 1 — the victim dies: its pages are back, nothing is owed.
        planner.unregister(victim);
        assert!(
            !planner.with_inner(|inner| inner.kill_in_flight()),
            "a retired victim cannot still owe pages"
        );

        // Exit 2 — the victim survives the error and asks again. It is a
        // legal target once more, so the marker must lift for it too.
        let survivor = ProcessId::new_v4();
        planner.register(survivor);
        planner.note_admitted(survivor);
        planner.with_inner(|inner| inner.killing.insert(survivor));
        assert!(planner.with_inner(|inner| inner.kill_in_flight()));

        let demand = Demand {
            kv_pages: 1, // the pool starts empty, so even one page must park
            rs_slots: 0,
        };
        let p = planner.clone();
        let parked = tokio::spawn(async move { p.acquire(survivor, survivor, demand).await });
        for _ in 0..200 {
            tokio::task::yield_now().await;
            if planner.diagnostics().queue.len() == 1 {
                break;
            }
        }
        assert_eq!(planner.diagnostics().queue.len(), 1, "the ask must park");
        assert!(
            !planner.with_inner(|inner| inner.kill_in_flight()),
            "a fresh unmet ask means the victim survived and is contending again"
        );

        parked.abort();
    }

    /// The deferred-allocation contract's cancellation half: dropping an
    /// uncollected [`AllocationTicket`] must deregister the parked ask,
    /// exactly like dropping an inline `acquire` future (process-death
    /// cleanup — a dead process's deferred frame must not hold the FCFS
    /// head hostage).
    #[tokio::test(flavor = "current_thread")]
    async fn dropping_an_uncollected_ticket_deregisters_the_ask() {
        let pool = Arc::new(RacePool::new(4));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone()));

        // A running holder keeps the wedge predicate false so the parked ask
        // is not destroyed on arrival by the starvation rung.
        let holder = ProcessId::new_v4();
        planner.register(holder);
        planner.note_admitted(holder);

        let pid = ProcessId::new_v4();
        planner.register(pid);
        planner.note_admitted(pid);
        let demand = Demand {
            kv_pages: 1, // the pool starts empty, so the ask must park
            rs_slots: 0,
        };
        let ticket = match planner.acquire_or_enqueue(pid, pid, demand) {
            Ok(Enqueued::Ticket(ticket)) => ticket,
            Ok(Enqueued::Granted(_)) => panic!("an empty pool cannot grant"),
            Ok(Enqueued::NotResident) => panic!("the process is resident"),
            Err(error) => panic!("enqueue failed: {error}"),
        };
        assert_eq!(planner.diagnostics().queue.len(), 1, "the ask parked FCFS");

        drop(ticket);
        assert_eq!(
            planner.diagnostics().queue.len(),
            0,
            "dropping an uncollected ticket must remove the queue entry"
        );
    }

    /// The deferred-allocation contract's service half: a ticket collects the
    /// grant the drain serves — same outcome the inline collect loop would
    /// have returned, from a task that is not the enqueuer's.
    #[tokio::test(flavor = "current_thread")]
    async fn a_ticket_collects_the_grant_the_drain_serves() {
        let pool = Arc::new(RacePool::new(4));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone()));

        let holder = ProcessId::new_v4();
        planner.register(holder);
        planner.note_admitted(holder);

        let pid = ProcessId::new_v4();
        planner.register(pid);
        planner.note_admitted(pid);
        let demand = Demand {
            kv_pages: 2,
            rs_slots: 0,
        };
        let ticket = match planner.acquire_or_enqueue(pid, pid, demand) {
            Ok(Enqueued::Ticket(ticket)) => ticket,
            _ => panic!("an empty pool must park the ask"),
        };

        // Pages come back; the freed poke drains the queue head into a
        // parked outcome the ticket then collects.
        pool.refill_after_stall(4, 0);
        planner.pages_freed();

        let collected = tokio::time::timeout(std::time::Duration::from_secs(5), ticket.collect())
            .await
            .expect("collect must resolve once the pool refills");
        match collected {
            Ok(Acquired::Granted(grant)) => drop(grant),
            Ok(Acquired::Yield) => panic!("nothing evicted this process"),
            Err(error) => panic!("collect failed: {error}"),
        }
        assert_eq!(
            planner.diagnostics().queue.len(),
            0,
            "the collected entry left the queue"
        );
    }
}

#[cfg(test)]
mod quote_budget_tests {
    use super::*;
    use std::cell::RefCell;

    /// A stand-in for `PageTable::reclaim_quotes` that honours a page budget
    /// exactly as the real quoter does: emit answers in order, stop once the
    /// pages EMITTED reach the budget, and report the positions past the cut
    /// as `None` — the same value an unknown process produces.
    struct BudgetedQuoter {
        pages: Vec<u32>,
        budgets: RefCell<Vec<u32>>,
    }

    impl BudgetedQuoter {
        fn new(pages: &[u32]) -> Self {
            Self {
                pages: pages.to_vec(),
                budgets: RefCell::new(Vec::new()),
            }
        }

        fn quote(&self, pids: &[ProcessId], budget: u32) -> Vec<Option<ReclaimQuote>> {
            self.budgets.borrow_mut().push(budget);
            let mut covered = 0u32;
            pids.iter()
                .enumerate()
                .map(|(i, _)| {
                    if covered >= budget {
                        return None;
                    }
                    let pages = self.pages[i];
                    covered = covered.saturating_add(pages);
                    Some(ReclaimQuote::Pages(pages))
                })
                .collect()
        }
    }

    /// A victim too large for the host pool must not hide the smaller one
    /// behind it.
    ///
    /// The budget charges the big victim's 5 pages and truncates; the picker
    /// refuses it for room and banks nothing, so without the re-ask the small
    /// victim is never quoted, `plan_eviction` reports `NoSwapRoom`, and
    /// `check_starvation` skips `last_resort_evict` on that cause — a live
    /// allocation is destroyed while a fitting victim sat one position past
    /// the cut. Deleting the escalation fails this test.
    #[test]
    fn a_victim_refused_for_room_does_not_hide_the_one_that_fits() {
        let big = ProcessId::new_v4();
        let small = ProcessId::new_v4();
        let quoter = BudgetedQuoter::new(&[5, 2]);

        let (picks, unhostable) =
            ResidencyPlanner::pick_with_budget_escalation(&[big, small], 5, 2, |pids, budget| {
                quoter.quote(pids, budget)
            });

        assert_eq!(picks, vec![(small, 2)]);
        assert_eq!(unhostable, vec![big]);
        assert_eq!(
            *quoter.budgets.borrow(),
            vec![5, u32::MAX],
            "the short pass must be followed by exactly one unbudgeted re-ask"
        );
    }

    /// The re-ask costs a full-fleet quote under the global KV mutex, which
    /// is what the budget exists to avoid: it must stay off the contended
    /// path where the host pool can fund the deficit outright.
    #[test]
    fn a_covered_deficit_never_re_asks() {
        let first = ProcessId::new_v4();
        let second = ProcessId::new_v4();
        let quoter = BudgetedQuoter::new(&[5, 4]);

        let (picks, unhostable) = ResidencyPlanner::pick_with_budget_escalation(
            &[first, second],
            5,
            16,
            |pids, budget| quoter.quote(pids, budget),
        );

        assert_eq!(picks, vec![(first, 5)]);
        assert!(unhostable.is_empty());
        assert_eq!(*quoter.budgets.borrow(), vec![5], "no re-ask when covered");
    }

    /// Coming up short with nothing refused is a genuine "nobody has
    /// anything", not a truncation artifact — the quoter charges zero for
    /// every `Nothing`, so it never truncates and a re-ask would return the
    /// identical list.
    #[test]
    fn a_fleet_holding_nothing_does_not_re_ask() {
        let a = ProcessId::new_v4();
        let b = ProcessId::new_v4();
        let quoter = BudgetedQuoter::new(&[0, 0]);

        let (picks, unhostable) =
            ResidencyPlanner::pick_with_budget_escalation(&[a, b], 5, 16, |pids, budget| {
                quoter.quote(pids, budget)
            });

        assert!(picks.is_empty());
        assert!(unhostable.is_empty());
        assert_eq!(
            *quoter.budgets.borrow(),
            vec![5],
            "no re-ask when nothing is refused"
        );
    }
}
