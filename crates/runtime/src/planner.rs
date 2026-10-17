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
use crate::store::rs::RsUnits;

pub fn trace_enabled() -> bool {
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

pub type ProcessId = uuid::Uuid;

pub trait PoolPort: Send + Sync + 'static {
    fn device_stats(&self) -> (u32, u32);
    fn host_stats(&self) -> (u32, u32);
    fn disk_stats(&self) -> (u32, u32);
    fn rs_stats(&self) -> (RsUnits, RsUnits);
    fn rs_host_stats(&self) -> (RsUnits, RsUnits);
    fn reclaim_idle(&self) -> u32;
    /// Called until it frees nothing, without freeing them.
    fn reclaimable_idle(&self) -> u32;
    fn suspend_capable(&self) -> bool;
    fn locus(&self) -> (usize, usize);
    fn reserve_device(
        &self,
        count: u32,
    ) -> Option<Vec<crate::store::kv::page_table::PhysicalKvPageId>>;
    fn reserve_device_up_to(
        &self,
        count: u32,
    ) -> Vec<crate::store::kv::page_table::PhysicalKvPageId>;
    fn release_device(&self, pages: Vec<crate::store::kv::page_table::PhysicalKvPageId>);
    fn reserve_rs(&self, need: RsUnits) -> Option<crate::store::rs::RsGrant>;
    fn release_rs(&self, grant: crate::store::rs::RsGrant);
    /// Drops the checkpoints sharing state with `working_sets`, so
    /// their next write lands in place; returns how many were dropped.
    fn yield_checkpoints(&self, working_sets: &[crate::store::rs::RsWorkingSetId]) -> usize;
    fn prepare_sweep(
        &self,
        keep: &HashSet<crate::store::kv::page_table::WorkingSetId>,
        room: usize,
    ) -> Option<crate::store::kv::KvResidencyTxn>;
    fn prepare_demote(&self, room: usize) -> Option<crate::store::kv::KvResidencyTxn>;
}

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

    fn disk_stats(&self) -> (u32, u32) {
        self.with_kv_tagged("planner-disk-stats", |kv| {
            (kv.disk_swap_available() as u32, kv.disk_swap_capacity())
        })
    }

    fn rs_stats(&self) -> (RsUnits, RsUnits) {
        self.with_rs(|rs| (rs.available(), rs.capacity()))
    }

    fn rs_host_stats(&self) -> (RsUnits, RsUnits) {
        self.with_rs(|rs| (rs.host_available(), rs.host_capacity()))
    }

    fn reclaim_idle(&self) -> u32 {
        let pages = self.with_kv_tagged("planner-reclaim-idle", |kv| {
            let epoch = kv.current_epoch();
            kv.reclaim_cache(epoch, crate::policy::retention::Installed)
        });
        let slots = self.with_rs(|rs| rs.reclaim_cache(crate::policy::retention::Installed));
        (pages + slots) as u32
    }

    fn reclaimable_idle(&self) -> u32 {
        self.with_kv_tagged("planner-reclaimable-idle", |kv| {
            kv.reclaimable_cache_pages()
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
        self.with_kv_tagged("planner-release", |kv| {
            kv.release_device_reservation(pages);
        });
    }

    fn reserve_rs(&self, need: RsUnits) -> Option<crate::store::rs::RsGrant> {
        self.with_rs(|rs| rs.reserve(need))
    }

    fn release_rs(&self, grant: crate::store::rs::RsGrant) {
        self.with_rs(|rs| rs.release_reservation(grant));
    }

    fn yield_checkpoints(&self, working_sets: &[crate::store::rs::RsWorkingSetId]) -> usize {
        self.with_rs(|rs| rs.yield_checkpoints(working_sets))
    }

    fn prepare_sweep(
        &self,
        keep: &HashSet<crate::store::kv::page_table::WorkingSetId>,
        room: usize,
    ) -> Option<crate::store::kv::KvResidencyTxn> {
        if !self.suspend_capable {
            return None;
        }
        self.with_kv_tagged("planner-sweep", |kv| kv.prepare_sweep(keep, room))
            .inspect_err(|error| tracing::warn!(%error, "planner: sweep prepare failed"))
            .ok()
            .flatten()
    }

    fn prepare_demote(&self, room: usize) -> Option<crate::store::kv::KvResidencyTxn> {
        if !self.suspend_capable {
            return None;
        }
        self.with_kv_tagged("planner-demote", |kv| {
            kv.prepare_demote(room, crate::policy::retention::Installed)
        })
        .inspect_err(|error| tracing::warn!(%error, "planner: demote prepare failed"))
        .ok()
        .flatten()
    }
}

pub enum Acquired {
    Granted(AllocationGrant),
    Yield,
    /// The snapshots the demand counted were dropped; quote it again.
    Requote,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarveCause {
    NoSwapRoom,
    NoEligibleVictim,
    NoRsSlots,
}

impl StarveCause {
    fn describe(self) -> &'static str {
        match self {
            StarveCause::NoSwapRoom => {
                "no host or disk swap room to evict into (or no swap transport)"
            }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannerError {
    Impossible {
        need: u32,
        total: u32,
    },
    Hog {
        need: u32,
        held: u32,
        total: u32,
    },
    Starved {
        need: u32,
        free: u32,
        total: u32,
        cause: StarveCause,
    },
    RsHog {
        need: RsUnits,
        held: RsUnits,
        total: RsUnits,
    },
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
            } => {
                let (pool, unit) = if matches!(cause, StarveCause::NoRsSlots) {
                    ("state", "slots")
                } else {
                    ("KV", "pages")
                };
                write!(
                    f,
                    "{pool} pool starved: {need} {unit} asked, {free} free of {total}, {}, and no \
                     fire in flight anywhere to complete and free {unit}",
                    cause.describe(),
                )
            }
            PlannerError::RsHog { need, held, total } => write!(
                f,
                "recurrent pools exhausted by this process alone: it holds {held} and asks \
                 {need} more of {total} — raise `[engine] max_state_slots`"
            ),
            PlannerError::Cancelled => f.write_str("planner request cancelled"),
        }
    }
}

impl std::error::Error for PlannerError {}

struct LockCensus {
    n: AtomicU64,
    wait_ns: AtomicU64,
    hold_ns: AtomicU64,
    wait_max_ns: AtomicU64,
    hold_max_ns: AtomicU64,
}
static LOCK_CENSUS: LockCensus = LockCensus {
    n: AtomicU64::new(0),
    wait_ns: AtomicU64::new(0),
    hold_ns: AtomicU64::new(0),
    wait_max_ns: AtomicU64::new(0),
    hold_max_ns: AtomicU64::new(0),
};
fn lock_trace_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("PIE_PLANNER_LOCK_TRACE").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}
struct TimedGuard<'a> {
    guard: parking_lot::MutexGuard<'a, Inner>,
    held_from: Option<crate::rt::Instant>,
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

#[derive(Debug, Default)]
pub struct PlannerStats {
    pub parks: AtomicU64,
    pub serves: AtomicU64,
    pub evictions_started: AtomicU64,
    pub eviction_deferrals: AtomicU64,
    pub evictions: AtomicU64,
    pub eviction_rollbacks: AtomicU64,
    pub restores: AtomicU64,
    pub restore_failures: AtomicU64,
    pub gate_parks: AtomicU64,
    pub cancelled_waits: AtomicU64,
    pub hog_failures: AtomicU64,
    pub starvations: AtomicU64,
    pub starvation_restarts: AtomicU64,
    pub salvages: AtomicU64,
    pub hoard_bypasses: AtomicU64,
    pub e6_relaxations: AtomicU64,
    pub restore_absorb_short: AtomicU64,
    pub host_swap_exhaustions: AtomicU64,
    pub host_swap_unblocks: AtomicU64,
    pub d2h_pages: AtomicU64,
    pub h2d_pages: AtomicU64,
    pub disk_write_pages: AtomicU64,
    pub disk_read_pages: AtomicU64,
    pub disk_write_us: AtomicU64,
    pub disk_read_us: AtomicU64,
    pub d2h_rs_slots: AtomicU64,
    pub h2d_rs_slots: AtomicU64,
    pub d2h_copy_us: AtomicU64,
    pub h2d_copy_us: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct PlannerQueueEntry {
    pub process_id: String,
    pub spawn_seq: u64,
    pub kind: &'static str,
    pub pages: u32,
    pub rs: RsUnits,
}

const RUNNER_DUMP_CAP: usize = 24;

const HOST_RESERVE_DIVISOR: u32 = 8;

/// It suspends no process of its own.
const SWEEP: ProcessId = ProcessId::nil();

#[derive(Debug, Clone)]
pub struct PlannerDiagnostics {
    pub device_pages_free: u32,
    pub device_pages_total: u32,
    pub host_slots_free: u32,
    pub host_slots_total: u32,
    pub disk_slots_free: u32,
    pub disk_slots_total: u32,
    pub rs_free: RsUnits,
    pub rs_total: RsUnits,
    pub accumulation: u32,
    pub queue: Vec<PlannerQueueEntry>,
    pub proc_states: [u32; 4],
    pub admitted_procs: u32,
    pub runners: Vec<(u64, u32, bool)>,
    pub parks_total: u64,
    pub serves_total: u64,
    pub evictions_total: u64,
    pub eviction_deferrals_total: u64,
    pub unmet_head_pages: u32,
    pub unmet_head_kind: &'static str,
    pub unmet_queued: u32,
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
    pub disk_write_pages_total: u64,
    pub disk_read_pages_total: u64,
    pub disk_write_us_total: u64,
    pub disk_read_us_total: u64,
    pub d2h_rs_slots_total: u64,
    pub h2d_rs_slots_total: u64,
    pub d2h_copy_us_total: u64,
    pub h2d_copy_us_total: u64,
}

#[derive(Clone, Copy)]
struct Victim {
    pid: ProcessId,
    rank: Rank,
    e6_fresh: bool,
}

struct VictimSet {
    head: EntryKey,
    head_pid: ProcessId,
    deficit: Reclaim,
    members: Vec<Victim>,
}

impl VictimSet {
    fn preferred(&self) -> Vec<(ProcessId, Rank)> {
        self.members
            .iter()
            .filter(|v| v.e6_fresh)
            .map(|v| (v.pid, v.rank))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Residency {
    Resident,
    Evicting,
    Evicted,
    Restoring,
}

struct Proc {
    seq: u64,
    key: i64,
    held: bool,
    state: Residency,
    restore_retried: bool,
    progressed: bool,
    admitted: bool,
    respawn: bool,
    signal: Arc<Notify>,
    resident: Arc<AtomicBool>,
    /// Waiting on its client: a victim for anyone, however old it is.
    idle: bool,
    /// What an idle eviction freed; it asks for it back only once it wakes.
    parked: Option<Demand>,
    /// Parked below the floor, it gave its execution seat back: it takes
    /// one again before its restore is queued.
    seatless: bool,
    reseating: bool,
}

impl Proc {
    fn rank(&self) -> Rank {
        Rank::of(self.key, self.seq)
    }

    fn new(seq: u64) -> Self {
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
        }
    }
}

enum WaitKind {
    Allocation {
        demand: Demand,
        notify: Arc<Notify>,
        outcome: Option<Result<AllocationGrant, PlannerError>>,
        yielded: bool,
    },
    Restore {
        demand: Demand,
    },
}

struct Waiter {
    pid: ProcessId,
    kind: WaitKind,
}

impl Waiter {
    fn is_unmet(&self) -> bool {
        match &self.kind {
            WaitKind::Allocation {
                outcome, yielded, ..
            } => outcome.is_none() && !yielded,
            WaitKind::Restore { .. } => true,
        }
    }

    fn kv_need(&self) -> u32 {
        match &self.kind {
            WaitKind::Allocation { demand, .. } | WaitKind::Restore { demand } => demand.kv_pages,
        }
    }

    fn rs_need(&self) -> RsUnits {
        match &self.kind {
            WaitKind::Allocation { demand, .. } | WaitKind::Restore { demand } => demand.rs,
        }
    }

    fn kind_name(&self) -> &'static str {
        match self.kind {
            WaitKind::Allocation { .. } => "allocation",
            WaitKind::Restore { .. } => "restore",
        }
    }
}

/// The policy key first, larger earlier, then age, older earlier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Rank {
    key: std::cmp::Reverse<i64>,
    seq: u64,
}

impl Rank {
    fn of(key: i64, seq: u64) -> Self {
        Self {
            key: std::cmp::Reverse(key),
            seq,
        }
    }
}

/// Kept current by [`Inner::reschedule`]; the id alone names the entry.
type EntryKey = (Rank, u64);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Reclaim {
    pages: u32,
    rs: RsUnits,
}

impl Reclaim {
    fn covers(self, need: Reclaim) -> bool {
        self.pages >= need.pages && self.rs.covers(need.rs)
    }

    fn fits(self, room: Reclaim) -> bool {
        self.pages <= room.pages && room.rs.covers(self.rs)
    }

    fn add(self, other: Reclaim) -> Reclaim {
        Reclaim {
            pages: self.pages.saturating_add(other.pages),
            rs: self.rs.add(other.rs),
        }
    }

    fn sub(self, other: Reclaim) -> Reclaim {
        Reclaim {
            pages: self.pages.saturating_sub(other.pages),
            rs: self.rs.sub(other.rs),
        }
    }
}

#[derive(Default)]
struct Inner {
    next_seq: u64,
    next_id: u64,
    procs: HashMap<ProcessId, Proc>,
    queue: BTreeMap<EntryKey, Waiter>,
    accum: DevicePageReservation,
    evicting: HashMap<ProcessId, Reclaim>,
    completion_credit: u32,
    host_swap_blocked: HashSet<ProcessId>,
    prepare_blocked: HashSet<ProcessId>,
    killing: HashSet<ProcessId>,
    admitting: BTreeMap<(Rank, ProcessId), Arc<Notify>>,
}

impl Inner {
    fn kill_in_flight(&self) -> bool {
        self.killing.iter().any(|pid| {
            self.procs.contains_key(pid)
                && !self
                    .queue
                    .values()
                    .any(|waiter| waiter.pid == *pid && waiter.is_unmet())
        })
    }

    fn settle_eviction(&mut self, pid: ProcessId) {
        self.evicting.remove(&pid);
    }

    /// What the evictions in flight are expected to free.
    fn expected_reclaim(&self) -> Reclaim {
        self.evicting
            .values()
            .fold(Reclaim::default(), |sum, &mark| sum.add(mark))
    }

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

    fn preempted_floor(&self) -> Option<Rank> {
        self.procs
            .iter()
            .filter(|(pid, proc)| {
                (proc.state != Residency::Resident && proc.parked.is_none())
                    || proc.respawn
                    || self.killing.contains(pid)
            })
            .map(|(_, proc)| proc.rank())
            .min()
    }

    fn admit_unpreempted(&mut self) {
        if self.admitting.is_empty() {
            return;
        }
        let floor = self.preempted_floor();
        let (go, blocked): (BTreeMap<_, _>, BTreeMap<_, _>) = std::mem::take(&mut self.admitting)
            .into_iter()
            .partition(|((rank, _), _)| floor.is_none_or(|floor| *rank <= floor));
        self.admitting = blocked;
        for notify in go.into_values() {
            notify.notify_one();
        }
    }

    fn enqueue(&mut self, pid: ProcessId, kind: WaitKind) -> EntryKey {
        let key = (self.procs[&pid].rank(), self.next_id);
        self.next_id += 1;
        self.queue.insert(key, Waiter { pid, kind });
        key
    }

    /// Wherever a reschedule has since moved it.
    fn locate(&self, key: EntryKey) -> Option<EntryKey> {
        if self.queue.contains_key(&key) {
            return Some(key);
        }
        self.queue.keys().find(|(_, id)| *id == key.1).copied()
    }

    fn reschedule(&mut self, order_of: impl Fn(&ProcessId) -> (i64, bool)) {
        for (pid, proc) in &mut self.procs {
            (proc.key, proc.held) = order_of(pid);
        }
        let held_restores: Vec<EntryKey> = self
            .queue
            .iter()
            .filter(|(_, w)| matches!(w.kind, WaitKind::Restore { .. }))
            .filter(|(_, w)| self.procs.get(&w.pid).is_some_and(|p| p.held))
            .map(|(key, _)| *key)
            .collect();
        for key in held_restores {
            if let Some(Waiter {
                pid,
                kind: WaitKind::Restore { demand },
            }) = self.queue.remove(&key)
                && let Some(proc) = self.procs.get_mut(&pid)
            {
                proc.parked = Some(demand);
            }
        }
        let released: Vec<(ProcessId, Demand)> = self
            .procs
            .iter_mut()
            .filter(|(_, p)| !p.held && !p.idle && !p.seatless && p.state == Residency::Evicted)
            .filter_map(|(pid, p)| Some((*pid, p.parked.take()?)))
            .collect();
        for (pid, demand) in released {
            self.enqueue(pid, WaitKind::Restore { demand });
        }
        let procs = &self.procs;
        let rank = |pid: &ProcessId, was: Rank| procs.get(pid).map_or(was, Proc::rank);
        self.queue = std::mem::take(&mut self.queue)
            .into_iter()
            .map(|((was, id), waiter)| ((rank(&waiter.pid, was), id), waiter))
            .collect();
        self.admitting = std::mem::take(&mut self.admitting)
            .into_iter()
            .map(|((was, pid), notify)| ((rank(&pid, was), pid), notify))
            .collect();
    }

    fn unmet_head(&self) -> Option<(EntryKey, &Waiter)> {
        let mut oldest_restore: Option<EntryKey> = None;
        let mut allocation: Option<EntryKey> = None;
        for (&key, waiter) in &self.queue {
            if !waiter.is_unmet() {
                continue;
            }
            match &waiter.kind {
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

    fn queued_quantum(&self) -> u32 {
        self.queue
            .values()
            .filter(|waiter| waiter.is_unmet())
            .filter(|waiter| {
                matches!(&waiter.kind, WaitKind::Allocation { demand, .. } if demand.kv_pages == 1)
            })
            .count() as u32
    }

    fn begin_eviction(&mut self, pid: ProcessId, expected: Reclaim) -> Vec<Arc<Notify>> {
        if let Some(proc) = self.procs.get_mut(&pid) {
            proc.state = Residency::Evicting;
        }
        self.evicting.insert(pid, expected);
        let mut wake = Vec::new();
        for waiter in self.queue.values_mut().filter(|w| w.pid == pid) {
            if let WaitKind::Allocation {
                notify, yielded, ..
            } = &mut waiter.kind
            {
                *yielded = true;
                wake.push(notify.clone());
            }
        }
        wake
    }

    fn eviction_unfundable(&self) -> bool {
        !self.host_swap_blocked.is_empty()
    }

    fn burst_shortfall(&self) -> u32 {
        let mut supply = self.accum.len() as u32;
        let mut extra = 0u32;
        for waiter in self.queue.values() {
            if !waiter.is_unmet() {
                continue;
            }
            let WaitKind::Allocation { demand, .. } = &waiter.kind else {
                continue;
            };
            if !demand.rs.is_zero() {
                break;
            }
            let need = demand.kv_pages;
            let covered = need.min(supply);
            supply -= covered;
            extra = extra.saturating_add(need - covered);
        }
        extra
    }

    fn serve_burst(&mut self) -> Vec<(EntryKey, u32, Arc<Notify>)> {
        let mut wake = Vec::new();
        let Some((head, waiter)) = self.unmet_head() else {
            return wake;
        };
        if !matches!(waiter.kind, WaitKind::Allocation { .. }) {
            return wake;
        }
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
                continue;
            };
            let demand = *demand;
            if !demand.rs.is_zero() {
                break;
            }
            if (accum.len() as u32) < demand.kv_pages {
                break;
            }
            let kv = accum.donate(demand.kv_pages);
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
            Residency::Resident => !proc.admitted || proc.idle || parked.contains(pid),
            Residency::Evicting | Residency::Restoring => false,
            Residency::Evicted => true,
        })
    }
}

enum Step {
    Absorb { count: u32, fund_by_eviction: bool },
    ShortRs { fund_by_eviction: bool },
    ServeAllocation { key: EntryKey, demand: Demand },
    ServeAllocationBurst { extra: u32 },
    ServeRestore { key: EntryKey, pid: ProcessId },
    Release(DevicePageReservation),
    Done,
}

pub struct ResidencyPlanner {
    inner: parking_lot::Mutex<Inner>,
    pause_due: AtomicBool,
    waiters: AtomicUsize,
    nonresident: AtomicUsize,
    drain: OnceLock<Arc<Notify>>,
    idle_reclaim_exhausted: std::sync::atomic::AtomicBool,
    sweeping: AtomicBool,
    sweep_exhausted: AtomicBool,
    demoting: AtomicBool,
    demote_after: AtomicU64,
    port: Arc<dyn PoolPort>,
    stats: PlannerStats,
}

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

impl ResidencyPlanner {
    pub fn new(port: Arc<dyn PoolPort>) -> Self {
        Self {
            inner: parking_lot::Mutex::new(Inner::default()),
            pause_due: AtomicBool::new(false),
            waiters: AtomicUsize::new(0),
            nonresident: AtomicUsize::new(0),
            drain: OnceLock::new(),
            idle_reclaim_exhausted: std::sync::atomic::AtomicBool::new(false),
            sweeping: AtomicBool::new(false),
            sweep_exhausted: AtomicBool::new(false),
            demoting: AtomicBool::new(false),
            demote_after: AtomicU64::new(0),
            port,
            stats: PlannerStats::default(),
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

    fn poke(self: &Arc<Self>) {
        match self.drain.get() {
            Some(notify) => notify.notify_one(),
            None => self.plan(),
        }
    }

    fn reclaim_idle_once(&self) -> bool {
        if self.idle_reclaim_exhausted.swap(true, Ordering::AcqRel) || self.port.reclaim_idle() == 0
        {
            return false;
        }
        self.idle_reclaim_exhausted.store(false, Ordering::Release);
        true
    }

    fn re_arm_idle_reclaim(&self) {
        self.idle_reclaim_exhausted.store(false, Ordering::Release);
        self.sweep_exhausted.store(false, Ordering::Release);
    }

    /// Not what the head and restores under way read: they would bring it
    /// straight back.
    fn sweep_stranded(self: &Arc<Self>) -> bool {
        if self.sweep_exhausted.load(Ordering::Acquire)
            || self.sweeping.swap(true, Ordering::AcqRel)
        {
            return false;
        }
        let (swap_free, swap_total) = self.swap_stats();
        let room = swap_free.saturating_sub(swap_total / HOST_RESERVE_DIVISOR) as usize;
        let (model, engine) = self.port.locus();
        let keep: HashSet<_> = self
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
        let txn = (room > 0)
            .then(|| self.port.prepare_sweep(&keep, room))
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

    fn lock_inner(&self) -> TimedGuard<'_> {
        if !lock_trace_enabled() {
            return TimedGuard {
                guard: self.inner.lock(),
                held_from: None,
            };
        }
        let t0 = crate::rt::Instant::now();
        let guard = self.inner.lock();
        let waited = t0.elapsed().as_nanos() as u64;
        LOCK_CENSUS.n.fetch_add(1, Ordering::Relaxed);
        LOCK_CENSUS.wait_ns.fetch_add(waited, Ordering::Relaxed);
        LOCK_CENSUS.wait_max_ns.fetch_max(waited, Ordering::Relaxed);
        TimedGuard {
            guard,
            held_from: Some(crate::rt::Instant::now()),
        }
    }

    fn with_inner<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        let mut inner = self.lock_inner();
        let result = f(&mut inner);
        self.waiters.store(inner.queue.len(), Ordering::Release);
        self.nonresident
            .store(inner.nonresident_count(), Ordering::Release);
        inner.admit_unpreempted();
        result
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
                Some(notify) => notify.notified().await,
                None => return,
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

    /// Marks `pid` as waiting on its client, or as awake again. Waking a
    /// parked process queues its restore, ahead of its next fire.
    pub fn set_idle(self: &Arc<Self>, pid: ProcessId, idle: bool) {
        self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return;
            };
            proc.idle = idle;
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
    fn vacate_held_seats(&self) {
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

    fn reseat_released(self: &Arc<Self>) {
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

    fn reseated(self: &Arc<Self>, pid: ProcessId, seat: crate::inferlet::process::Seat) {
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
    fn reseat_failed(&self, pid: ProcessId) {
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.reseating = false;
            }
        });
    }

    pub fn can_pause(&self) -> bool {
        self.port.suspend_capable()
    }

    /// A worker with no host or disk tier leaves it running.
    fn pause_held(self: &Arc<Self>) {
        if !self.port.suspend_capable() {
            ptrace!("pause_held: cannot suspend");
            return;
        }
        let held: Vec<(ProcessId, Rank)> = self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .filter(|(pid, proc)| {
                    proc.held
                        && proc.admitted
                        && proc.state == Residency::Resident
                        && !inner.host_swap_blocked.contains(*pid)
                        && !inner.prepare_blocked.contains(*pid)
                })
                .map(|(pid, proc)| (*pid, proc.rank()))
                .collect()
        });
        ptrace!("pause_held candidates={}", held.len());
        if held.is_empty() {
            return;
        }
        let (model, engine) = self.port.locus();
        let everything = Reclaim {
            pages: u32::MAX,
            rs: RsUnits {
                slots: u32::MAX,
                pages: u32::MAX,
            },
        };
        let (picks, unhostable) =
            self.quote_and_pick(held, everything, self.swap_room(), model, engine);
        // No tier takes them: they keep running.
        ptrace!(
            "pause_held picks={} unhostable={} room={:?}",
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
                    .is_some_and(|proc| proc.held && proc.state == Residency::Resident)
                {
                    continue;
                }
                wake.extend(inner.begin_eviction(pid, expected));
                spawned.push(pid);
            }
            (wake, spawned)
        });
        self.start_evictions(wake, spawned);
    }

    fn start_evictions(self: &Arc<Self>, wake: Vec<Arc<Notify>>, pids: Vec<ProcessId>) {
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
            }
        });
    }

    pub fn unregister(self: &Arc<Self>, pid: ProcessId) {
        // A seat taken again after the process ended goes back.
        crate::inferlet::process::vacate_seat(pid);
        let (signal, removed) = self.with_inner(|inner| {
            let departed = inner.procs.remove(&pid);
            if departed.is_some() {
                inner.admitting.retain(|(_, waiting), _| *waiting != pid);
            }
            if departed
                .as_ref()
                .is_some_and(|proc| proc.admitted && proc.state == Residency::Resident)
            {
                inner.completion_credit = inner.completion_credit.saturating_add(1);
            }
            let signal = departed.map(|proc| proc.signal);
            inner.settle_eviction(pid);
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
        drop(removed);
        self.re_arm_idle_reclaim();
        self.poke();
    }

    pub fn pages_freed(self: &Arc<Self>) {
        self.re_arm_idle_reclaim();
        if self.waiters.load(Ordering::Acquire) != 0 {
            self.poke();
        } else {
            self.demote_idle();
        }
    }

    fn demote_idle(self: &Arc<Self>) {
        const BACKOFF_MS: u64 = 100;
        static START: std::sync::OnceLock<crate::rt::Instant> = std::sync::OnceLock::new();
        let now = START
            .get_or_init(crate::rt::Instant::now)
            .elapsed()
            .as_millis() as u64;
        if now < self.demote_after.load(Ordering::Acquire)
            || self.demoting.swap(true, Ordering::AcqRel)
        {
            return;
        }
        let (device_free, device_total) = self.port.device_stats();
        let (host_free, host_total) = self.port.host_stats();
        let room = host_free.saturating_sub(host_total / HOST_RESERVE_DIVISOR) as usize;
        let txn = (device_free.saturating_mul(HOST_RESERVE_DIVISOR) < device_total && room > 0)
            .then(|| self.port.prepare_demote(room))
            .flatten();
        let Some(txn) = txn else {
            self.demoting.store(false, Ordering::Release);
            self.demote_after.store(now + BACKOFF_MS, Ordering::Release);
            return;
        };
        exec::spawn_move_out(self.clone(), txn, "demote", Self::demote_done);
    }

    fn demote_done(self: &Arc<Self>, moved: bool) {
        self.demoting.store(false, Ordering::Release);
        if moved {
            self.re_arm_idle_reclaim();
            self.poke();
        }
    }

    fn try_reserve(&self, demand: Demand) -> Option<AllocationGrant> {
        let kv = if demand.kv_pages > 0 {
            let pages = self.port.reserve_device(demand.kv_pages)?;
            DevicePageReservation::new(pages, self.port.clone())
        } else {
            DevicePageReservation::empty()
        };
        let rs = self.reserve_rs_slots(demand.rs)?;
        Some(AllocationGrant::new(demand, kv, rs))
    }

    fn reserve_rs_slots(&self, need: RsUnits) -> Option<RsSlotReservation> {
        if need.is_zero() {
            return Some(RsSlotReservation::empty());
        }
        let grant = self.port.reserve_rs(need)?;
        Some(RsSlotReservation::new(grant, self.port.clone()))
    }

    fn absorb_free(&self, count: u32) -> bool {
        let pages = self.port.reserve_device_up_to(count);
        if pages.is_empty() {
            return false;
        }
        let reservation = DevicePageReservation::new(pages, self.port.clone());
        self.with_inner(|inner| inner.accum.absorb(reservation));
        true
    }

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

    /// The grant `demand` takes without parking, if the pools hand it out
    /// now; see `acquire`.
    pub fn grant_now(self: &Arc<Self>, pid: ProcessId, demand: Demand) -> Option<AllocationGrant> {
        if demand.is_zero() {
            if self.waiters.load(Ordering::Acquire) != 0
                || self.nonresident.load(Ordering::Acquire) != 0
            {
                self.note_progress(pid);
            }
            return Some(AllocationGrant::empty());
        }
        let uncontended = self.waiters.load(Ordering::Acquire) == 0
            && self.nonresident.load(Ordering::Acquire) == 0;
        if !uncontended {
            self.note_progress(pid);
        }
        if uncontended && let Some(grant) = self.try_reserve(demand) {
            return Some(grant);
        }
        None
    }

    pub async fn acquire(
        self: &Arc<Self>,
        pid: ProcessId,
        quorum_id: ProcessId,
        demand: Demand,
    ) -> Result<Acquired, PlannerError> {
        if let Some(grant) = self.grant_now(pid, demand) {
            return Ok(Acquired::Granted(grant));
        }
        let (_, kv_total) = self.port.device_stats();
        if demand.kv_pages > kv_total {
            return Err(PlannerError::Impossible {
                need: demand.kv_pages,
                total: kv_total,
            });
        }
        if !demand.rs.is_zero() {
            let (rs_free, rs_total) = self.port.rs_stats();
            if !rs_total.covers(demand.rs) {
                let (need, total) = if demand.rs.slots > rs_total.slots {
                    (demand.rs.slots, rs_total.slots)
                } else {
                    (demand.rs.pages, rs_total.pages)
                };
                return Err(PlannerError::Impossible { need, total });
            }
            // A cache entry never makes live work wait: short of state, a
            // resident process quotes again once the idle cache is dropped,
            // or else the snapshots sharing its own state, instead of parking.
            let resident = self.with_inner(|inner| {
                inner.procs.get(&pid).is_some_and(|proc| {
                    proc.state == Residency::Resident && !inner.killing.contains(&pid)
                })
            });
            if !rs_free.covers(demand.rs) && resident {
                // What this frees may be what an older waiter parked on.
                if self.reclaim_idle_once() {
                    self.pages_freed();
                    return Ok(Acquired::Requote);
                }
                if self.yield_own_checkpoints(pid) {
                    self.pages_freed();
                    return Ok(Acquired::Requote);
                }
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
            let notify = Arc::new(Notify::new());
            let key = inner.enqueue(
                pid,
                WaitKind::Allocation {
                    demand,
                    notify: notify.clone(),
                    outcome: None,
                    yielded: false,
                },
            );
            Parked::Entry(key, notify)
        });
        match parked {
            Parked::Gone => Err(PlannerError::Cancelled),
            Parked::NotResident => Ok(Acquired::Yield),
            Parked::Entry(key, notify) => {
                self.stats.parks.fetch_add(1, Ordering::Relaxed);
                ptrace!("park key={:?} pid={} kv={:?}", key, pid, demand.kv_pages);
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
                    let acquired = match self.collect_outcome(key) {
                        Collect::Wait => {
                            notified.await;
                            continue;
                        }
                        Collect::Ready(outcome) => {
                            ptrace!("collect key={:?} ok={}", key, outcome.is_ok());
                            if matches!(outcome, Err(PlannerError::Cancelled)) {
                                self.stats.cancelled_waits.fetch_add(1, Ordering::Relaxed);
                            }
                            outcome.map(Acquired::Granted)
                        }
                        Collect::Yield => Ok(Acquired::Yield),
                        Collect::Requote => Ok(Acquired::Requote),
                    };
                    registration.disarm();
                    self.poke();
                    return acquired;
                }
            }
        }
    }

    fn note_progress(&self, pid: ProcessId) {
        if let Some(proc) = self.lock_inner().procs.get_mut(&pid) {
            proc.progressed = true;
        }
    }

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
            let Some(key) = inner.locate(key) else {
                return Collect::Ready(Err(PlannerError::Cancelled));
            };
            let Some(waiter) = inner.queue.get_mut(&key) else {
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
                    let pid = waiter.pid;
                    inner.queue.remove(&key);
                    // A resident waiter yielded only its cache: it quotes
                    // again, with no residency to wait for.
                    match inner.procs.get(&pid) {
                        Some(proc) if proc.state == Residency::Resident => Collect::Requote,
                        _ => Collect::Yield,
                    }
                }
                None => Collect::Wait,
            }
        })
    }

    fn cancel_waiter(self: &Arc<Self>, key: EntryKey) {
        let removed = self.with_inner(|inner| {
            let key = inner.locate(key)?;
            inner.queue.remove(&key)
        });
        if let Some(waiter) = removed {
            drop(waiter);
            self.poke();
        }
    }

    pub fn is_resident(&self, pid: ProcessId) -> bool {
        self.inner
            .lock()
            .procs
            .get(&pid)
            .is_none_or(|proc| proc.state == Residency::Resident)
    }

    pub fn residency_flag(&self, pid: ProcessId) -> Option<Arc<AtomicBool>> {
        self.lock_inner()
            .procs
            .get(&pid)
            .map(|proc| proc.resident.clone())
    }

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

    pub fn plan(self: &Arc<Self>) {
        if self.pause_due.swap(false, Ordering::AcqRel) {
            self.pause_held();
        }
        self.reseat_released();
        loop {
            let (rs_free, _) = self.port.rs_stats();
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
                    return Step::Absorb {
                        count: missing,
                        fund_by_eviction: !is_restore || inner.fleet_stalled(),
                    };
                }
                // rs slots are taken whole when the head is served; a short
                // pool is funded as paged pages are, by suspending holders.
                if !rs_free.covers(waiter.rs_need()) {
                    return Step::ShortRs {
                        fund_by_eviction: !is_restore || inner.fleet_stalled(),
                    };
                }
                match demand {
                    Some(demand) if !demand.rs.is_zero() => Step::ServeAllocation { key, demand },
                    Some(_) => Step::ServeAllocationBurst {
                        extra: inner.burst_shortfall(),
                    },
                    None => {
                        if inner.completion_credit == 0 && !inner.fleet_stalled() {
                            return Step::Done;
                        }
                        inner.completion_credit = inner.completion_credit.saturating_sub(1);
                        Step::ServeRestore { key, pid }
                    }
                }
            });
            let fund_by_eviction = match step {
                Step::Done => return,
                Step::Release(stranded) => {
                    drop(stranded);
                    return;
                }
                Step::Absorb {
                    count,
                    fund_by_eviction,
                } => {
                    if self.absorb_free(count) || self.reclaim_idle_once() {
                        continue;
                    }
                    if self.sweep_stranded() {
                        return;
                    }
                    fund_by_eviction
                }
                Step::ShortRs { fund_by_eviction } => {
                    if self.reclaim_idle_once() || self.yield_head_checkpoints() {
                        continue;
                    }
                    fund_by_eviction
                }
                Step::ServeAllocation { key, demand } => {
                    let Some(rs) = self.reserve_rs_slots(demand.rs) else {
                        continue;
                    };
                    let outcome = self.with_inner(|inner| {
                        match inner.unmet_head() {
                            Some((head, _)) if head == key => {}
                            _ => return ServeOutcome::Stale(rs),
                        }
                        if (inner.accum.len() as u32) < demand.kv_pages {
                            return ServeOutcome::Stale(rs);
                        }
                        let kv = inner.accum.donate(demand.kv_pages);
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
                            continue;
                        }
                        ServeOutcome::Stale(rs) => {
                            drop(rs);
                            continue;
                        }
                    }
                }
                Step::ServeAllocationBurst { extra } => {
                    if extra > 0 {
                        self.absorb_free(extra);
                    }
                    let wake = self.with_inner(|inner| inner.serve_burst());
                    if !wake.is_empty() {
                        self.stats
                            .serves
                            .fetch_add(wake.len() as u64, Ordering::Relaxed);
                    }
                    for (key, pages, notify) in wake {
                        ptrace!("serve key={:?} kv={:?}", key, pages);
                        notify.notify_waiters();
                    }
                    continue;
                }
                Step::ServeRestore { key, pid } => {
                    let (model, engine) = self.port.locus();
                    let need = Demand {
                        kv_pages: swapped_page_count(pid, model, engine),
                        rs: swapped_rs_units(pid, model, engine),
                    };
                    let stale = self.with_inner(|inner| match inner.queue.get_mut(&key) {
                        Some(Waiter {
                            kind: WaitKind::Restore { demand },
                            ..
                        }) if *demand != need => {
                            *demand = need;
                            true
                        }
                        _ => false,
                    });
                    if stale {
                        continue;
                    }
                    let Some(slots) = self.reserve_rs_slots(need.rs) else {
                        continue;
                    };
                    let boarded = self.with_inner(|inner| {
                        match inner.unmet_head() {
                            Some((head, _)) if head == key => {}
                            _ => return Board::Replan,
                        }
                        if (inner.accum.len() as u32) < need.kv_pages {
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
                        Board::Go(inner.accum.donate(need.kv_pages))
                    });
                    match boarded {
                        Board::Go(pages) => {
                            exec::spawn_restore(self.clone(), pid, pages, slots);
                            continue;
                        }
                        Board::Replan => {
                            drop(slots);
                            continue;
                        }
                    }
                }
            };
            if fund_by_eviction {
                self.plan_eviction();
            } else {
                self.stats
                    .restore_absorb_short
                    .fetch_add(1, Ordering::Relaxed);
            }
            return;
        }
    }

    fn quote_and_pick(
        &self,
        candidates: Vec<(ProcessId, Rank)>,
        deficit: Reclaim,
        room: Reclaim,
        model: usize,
        engine: usize,
    ) -> (Vec<(ProcessId, Reclaim)>, Vec<ProcessId>) {
        let mut ordered: Vec<(ProcessId, Rank, bool)> = candidates
            .into_iter()
            .map(|(pid, rank)| {
                let quiescent =
                    crate::inferlet::process::residency::kv_lease_quiescent(pid, model, engine);
                (pid, rank, quiescent)
            })
            .collect();
        ordered.sort_by_key(|&(_, rank, quiescent)| (!quiescent, std::cmp::Reverse(rank)));
        let pids: Vec<ProcessId> = ordered.iter().map(|(pid, ..)| *pid).collect();
        let rs_quotes: Vec<RsUnits> = pids
            .iter()
            .map(|&pid| suspendable_rs_units(pid, model, engine))
            .collect();
        Self::pick_with_budget_escalation(&pids, &rs_quotes, deficit, room, |pids, budget| {
            crate::inferlet::process::residency::kv_reclaim_quotes(pids, model, engine, budget)
        })
    }

    fn pick_with_budget_escalation(
        pids: &[ProcessId],
        rs_quotes: &[RsUnits],
        deficit: Reclaim,
        room: Reclaim,
        quote: impl Fn(&[ProcessId], u32) -> Vec<Option<ReclaimQuote>>,
    ) -> (Vec<(ProcessId, Reclaim)>, Vec<ProcessId>) {
        // A pick moves its pages whole, so every candidate is quoted when
        // slots, not pages, are what the head lacks.
        let budget = if !deficit.rs.is_zero() {
            u32::MAX
        } else {
            deficit.pages
        };
        let (picks, unhostable) =
            Self::pick_from_quotes(pids, quote(pids, budget), rs_quotes, deficit, room);
        let covered = picks
            .iter()
            .fold(Reclaim::default(), |sum, &(_, pick)| sum.add(pick));
        if !covered.covers(deficit) && !unhostable.is_empty() {
            return Self::pick_from_quotes(pids, quote(pids, u32::MAX), rs_quotes, deficit, room);
        }
        (picks, unhostable)
    }

    /// Picks candidates in order until what they free covers the deficit.
    /// A pick's pages move whole, so one whose pages exceed the room left
    /// is unhostable; its rs slots park only when host rows hold them all,
    /// and stay on the device otherwise. One that adds nothing the head
    /// still lacks stays.
    fn pick_from_quotes(
        pids: &[ProcessId],
        quotes: Vec<Option<ReclaimQuote>>,
        rs_quotes: &[RsUnits],
        deficit: Reclaim,
        room: Reclaim,
    ) -> (Vec<(ProcessId, Reclaim)>, Vec<ProcessId>) {
        let mut picks = Vec::new();
        let mut unhostable = Vec::new();
        let (mut covered, mut room) = (Reclaim::default(), room);
        for ((&pid, quote), &rs) in pids.iter().zip(quotes).zip(rs_quotes) {
            if covered.covers(deficit) {
                break;
            }
            let pages = match quote {
                Some(ReclaimQuote::Pages(pages)) => pages,
                _ => 0,
            };
            let rs = if room.rs.covers(rs) {
                rs
            } else {
                RsUnits::default()
            };
            let frees = Reclaim { pages, rs };
            let helps = (pages > 0 && covered.pages < deficit.pages)
                || (rs.slots > 0 && covered.rs.slots < deficit.rs.slots)
                || (rs.pages > 0 && covered.rs.pages < deficit.rs.pages);
            if !helps {
                continue;
            }
            if !frees.fits(room) {
                unhostable.push(pid);
                continue;
            }
            room = room.sub(frees);
            covered = covered.add(frees);
            picks.push((pid, frees));
        }
        (picks, unhostable)
    }

    /// Free and total slots suspended kv can go to: host, then disk.
    fn swap_stats(&self) -> (u32, u32) {
        let (host_free, host_total) = self.port.host_stats();
        let (disk_free, disk_total) = self.port.disk_stats();
        (host_free + disk_free, host_total + disk_total)
    }

    /// The room below the device: swap slots for pages, host rows for rs.
    fn swap_room(&self) -> Reclaim {
        Reclaim {
            pages: self.swap_stats().0,
            rs: self.port.rs_host_stats().0,
        }
    }

    fn plan_eviction(self: &Arc<Self>) {
        let Some(victims) = self.victim_set() else {
            return;
        };
        let (swap_free, swap_total) = self.swap_stats();
        if victims.deficit.pages > 0 {
            if !self.port.suspend_capable() || swap_free == 0 {
                self.check_starvation(StarveCause::NoSwapRoom);
                return;
            }
            if (swap_free.saturating_mul(HOST_RESERVE_DIVISOR) <= swap_total && !self.is_wedged())
                || !self.supply_stalled()
            {
                self.stats
                    .eviction_deferrals
                    .fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        let (model, engine) = self.port.locus();
        let (picks, unhostable) = self.quote_and_pick(
            victims.preferred(),
            victims.deficit,
            self.swap_room(),
            model,
            engine,
        );
        if !unhostable.is_empty() {
            self.with_inner(|inner| {
                for pid in &unhostable {
                    inner.host_swap_blocked.insert(*pid);
                }
            });
            self.record_host_swap_exhaustion();
        }
        if picks.is_empty() {
            if victims.deficit.pages == 0 {
                self.check_rs_starvation(victims.head);
            } else if !unhostable.is_empty() {
                self.check_starvation(StarveCause::NoSwapRoom);
            } else if !self.check_hog(victims.head, victims.head_pid) {
                self.check_starvation(StarveCause::NoEligibleVictim);
            }
            return;
        }
        self.commit_evictions(victims.head, picks, false);
    }

    fn victim_set(&self) -> Option<VictimSet> {
        let (free, _) = self.port.device_stats();
        let (rs_free, _) = self.port.rs_stats();
        self.with_inner(|inner| {
            let (head, waiter) = inner.unmet_head()?;
            // What the device still has free and an eviction in flight
            // returns is supply, not deficit, as `supply_stalled` counts it.
            let missing = waiter
                .kv_need()
                .max(inner.queued_quantum())
                .saturating_sub(inner.accum.len() as u32);
            let expected = inner.expected_reclaim();
            let deficit = Reclaim {
                pages: missing.saturating_sub(expected.pages.saturating_add(free)),
                rs: waiter.rs_need().sub(rs_free.add(expected.rs)),
            };
            if deficit == Reclaim::default() {
                return None;
            }
            let head_rank = head.0;
            let members = inner
                .procs
                .iter()
                .filter(|(pid, proc)| {
                    proc.admitted
                        && (proc.rank() > head_rank || proc.idle || proc.held)
                        && proc.state == Residency::Resident
                        && !inner.host_swap_blocked.contains(*pid)
                        && !inner.prepare_blocked.contains(*pid)
                })
                .map(|(pid, proc)| Victim {
                    pid: *pid,
                    rank: proc.rank(),
                    e6_fresh: proc.progressed || proc.idle || proc.held,
                })
                .collect();
            Some(VictimSet {
                head,
                head_pid: waiter.pid,
                deficit,
                members,
            })
        })
    }

    fn commit_evictions(
        self: &Arc<Self>,
        head_key: EntryKey,
        picks: Vec<(ProcessId, Reclaim)>,
        e6_relaxed: bool,
    ) -> bool {
        let (wake, spawned) = self.with_inner(|inner| {
            let (mut wake, mut spawned) = (Vec::new(), Vec::new());
            let head_rank = match inner.unmet_head() {
                Some((key, _)) if key == head_key => key.0,
                _ => return (wake, spawned),
            };
            for (pid, expected) in picks {
                let Some(proc) = inner.procs.get(&pid) else {
                    continue;
                };
                if proc.state != Residency::Resident
                    || (proc.rank() <= head_rank && !proc.idle && !proc.held)
                    || (!proc.progressed && !proc.idle && !proc.held && !e6_relaxed)
                {
                    continue;
                }
                wake.extend(inner.begin_eviction(pid, expected));
                spawned.push(pid);
            }
            (wake, spawned)
        });
        let any = !spawned.is_empty();
        self.start_evictions(wake, spawned);
        any
    }

    fn yield_own_checkpoints(&self, pid: ProcessId) -> bool {
        let (model, engine) = self.port.locus();
        let working_sets: Vec<_> =
            crate::inferlet::process::residency::rs_working_set_ids(pid, model, engine)
                .into_iter()
                .collect();
        !working_sets.is_empty() && self.port.yield_checkpoints(&working_sets) > 0
    }

    /// A cache entry never makes live work wait: when the head's ask is
    /// short of state, the snapshots sharing its state are dropped so its
    /// write lands in place, and the head re-quotes as a yielded waiter.
    fn yield_head_checkpoints(self: &Arc<Self>) -> bool {
        let Some((key, pid)) = self.with_inner(|inner| {
            inner
                .unmet_head()
                .and_then(|(key, waiter)| match waiter.kind {
                    WaitKind::Allocation { .. } => Some((key, waiter.pid)),
                    WaitKind::Restore { .. } => None,
                })
        }) else {
            return false;
        };
        if !self.yield_own_checkpoints(pid) {
            return false;
        }
        let wake = self.with_inner(|inner| {
            let waiter = inner.queue.get_mut(&key)?;
            let WaitKind::Allocation {
                notify, yielded, ..
            } = &mut waiter.kind
            else {
                return None;
            };
            *yielded = true;
            Some(notify.clone())
        });
        match wake {
            Some(notify) => {
                notify.notify_waiters();
                true
            }
            None => false,
        }
    }

    fn supply_stalled(&self) -> bool {
        let (free, _) = self.port.device_stats();
        self.with_inner(|inner| {
            let accum = inner.accum.len() as u32;
            if inner.kill_in_flight() {
                return false;
            }
            if inner.unmet_head().is_none() {
                return false;
            }
            let expected = inner.expected_reclaim().pages;
            let supply = free + accum + expected;
            if inner.queued_quantum() > supply {
                return true;
            }
            free == 0
                && inner
                    .unmet_head()
                    .is_some_and(|(_, head)| head.kv_need() > accum + expected)
        })
    }

    fn is_wedged(&self) -> bool {
        self.with_inner(|inner| inner.fleet_stalled())
    }

    fn last_resort_evict(self: &Arc<Self>) -> bool {
        let (model, engine) = self.port.locus();
        let Some(stores) = crate::store::registry::try_get(model, engine) else {
            return false;
        };
        let pids: Vec<ProcessId> = self.with_inner(|inner| inner.procs.keys().copied().collect());
        if pids.is_empty() {
            return false;
        }
        let working_sets =
            crate::inferlet::process::residency::kv_working_sets_for(&pids, model, engine);
        let rs_quotes: Vec<RsUnits> = pids
            .iter()
            .map(|&pid| suspendable_rs_units(pid, model, engine))
            .collect();

        let (head, picks) =
            crate::store::registry::with_kv_lock(&stores.kv, "planner-endgame", |kv| {
                self.with_inner(|inner| {
                    let Some((head, waiter)) = inner.unmet_head() else {
                        return (None, Vec::new());
                    };
                    let missing = waiter.kv_need().saturating_sub(inner.accum.len() as u32);
                    let deficit = missing.saturating_sub(inner.expected_reclaim().pages);
                    if deficit == 0 {
                        return (None, Vec::new());
                    }
                    let head_rank = head.0;
                    let quotes = crate::inferlet::process::residency::quote_locked(
                        kv,
                        working_sets,
                        u32::MAX,
                    );
                    let mut legal: Vec<(ProcessId, Rank, bool, Reclaim)> = pids
                        .iter()
                        .zip(quotes)
                        .zip(&rs_quotes)
                        .filter_map(|((pid, quote), &rs)| {
                            let proc = inner.procs.get(pid)?;
                            if (proc.rank() <= head_rank && !proc.idle && !proc.held)
                                || proc.state != Residency::Resident
                                || inner.host_swap_blocked.contains(pid)
                                || inner.prepare_blocked.contains(pid)
                            {
                                return None;
                            }
                            match quote? {
                                ReclaimQuote::Pages(pages) if pages > 0 => Some((
                                    *pid,
                                    proc.rank(),
                                    proc.progressed,
                                    Reclaim { pages, rs },
                                )),
                                _ => None,
                            }
                        })
                        .collect();
                    legal.sort_by_key(|&(_, rank, fresh, _)| (!fresh, std::cmp::Reverse(rank)));
                    let mut picks = Vec::new();
                    let mut covered = 0u32;
                    for (pid, _, _, frees) in legal {
                        if covered >= deficit {
                            break;
                        }
                        covered += frees.pages;
                        picks.push((pid, frees));
                    }
                    (Some(head), picks)
                })
            });
        let (Some(head), false) = (head, picks.is_empty()) else {
            return false;
        };
        if !self.commit_evictions(head, picks, true) {
            return false;
        }
        self.stats.e6_relaxations.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn salvage_free_pages(&self) -> bool {
        let missing = self.with_inner(|inner| {
            let (_, head) = inner.unmet_head()?;
            Some(head.kv_need().saturating_sub(inner.accum.len() as u32))
        });
        let salvaged = missing.is_some_and(|missing| missing > 0 && self.absorb_free(missing));
        if salvaged {
            self.stats.salvages.fetch_add(1, Ordering::Relaxed);
        }
        salvaged
    }

    fn serve_from_hoard(self: &Arc<Self>) -> bool {
        let Some((key, demand)) = self.with_inner(|inner| {
            let (head_key, _) = inner.unmet_head()?;
            let hoard = inner.accum.len() as u32;
            inner.queue.iter().find_map(|(&key, waiter)| {
                if key == head_key || !waiter.is_unmet() {
                    return None;
                }
                match &waiter.kind {
                    WaitKind::Allocation { demand, .. } if hoard >= demand.kv_pages => {
                        Some((key, *demand))
                    }
                    _ => None,
                }
            })
        }) else {
            return false;
        };
        let Some(rs) = self.reserve_rs_slots(demand.rs) else {
            return false;
        };
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
            let kv = inner.accum.donate(demand.kv_pages);
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

    fn check_rs_starvation(self: &Arc<Self>, key: EntryKey) {
        if !self.is_wedged() {
            return;
        }
        let Some((pid, need)) = self.with_inner(|inner| {
            inner
                .queue
                .get(&key)
                .map(|waiter| (waiter.pid, waiter.rs_need()))
        }) else {
            return;
        };
        let (free, total) = self.port.rs_stats();
        if free.covers(need) {
            self.poke();
            return;
        }
        if !self.is_wedged() {
            return;
        }
        // What the head itself holds is not coming back while it waits, so
        // past that the pool is too small for it; short of that, the slots
        // are held by processes no suspend can move, and a restart frees them.
        let (model, engine) = self.port.locus();
        let held = held_rs_units(pid, model, engine);
        let restore = self.with_inner(|inner| {
            inner
                .queue
                .get(&key)
                .is_some_and(|waiter| matches!(waiter.kind, WaitKind::Restore { .. }))
        });
        if restore || total.covers(need.add(held)) {
            self.check_starvation(StarveCause::NoRsSlots);
            return;
        }
        let notify = self.with_inner(|inner| {
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
            *outcome = Some(Err(PlannerError::RsHog { need, held, total }));
            Some(notify.clone())
        });
        if let Some(notify) = notify {
            self.stats.hog_failures.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                %need,
                %free,
                %held,
                %total,
                "planner: RS ask past the pool — failing the head"
            );
            notify.notify_waiters();
        }
    }

    fn check_starvation(self: &Arc<Self>, cause: StarveCause) {
        if !self.is_wedged() {
            return;
        }
        if self.salvage_free_pages() {
            self.poke();
            return;
        }
        if cause == StarveCause::NoEligibleVictim && self.last_resort_evict() {
            return;
        }
        if !self.is_wedged() {
            return;
        }
        if self.salvage_free_pages() {
            self.poke();
            return;
        }
        if self.serve_from_hoard() {
            return;
        }
        let (free, total) = self.port.device_stats();
        if self.with_inner(|inner| inner.kill_in_flight()) {
            return;
        }
        // A head past the pool with what it holds itself fails: restarting
        // it would only replay it into the same wall.
        if let Some((head, pid)) =
            self.with_inner(|inner| inner.unmet_head().map(|(k, w)| (k, w.pid)))
            && self.check_hog(head, pid)
        {
            return;
        }
        let Some(victim_key) = self.pick_starvation_victim(cause) else {
            return;
        };
        let (rs_free, _) = self.port.rs_stats();
        let notify = self.with_inner(|inner| {
            let (_, head) = inner.unmet_head()?;
            // A head the pools already cover is served next, so a kill would
            // free nothing it lacks. One short of state alone is not covered
            // by the pages it has.
            let servable =
                (inner.accum.len() as u32) >= head.kv_need() && rs_free.covers(head.rs_need());
            if servable || !inner.evicting.is_empty() {
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
                return None;
            };
            if outcome.is_some() {
                return None;
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

    fn pick_starvation_victim(&self, cause: StarveCause) -> Option<EntryKey> {
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
            let quotes =
                crate::inferlet::process::residency::kv_reclaim_quotes(&pids, model, engine, 1);
            for ((key, pid), quote) in selected.iter().zip(quotes) {
                // A restart frees what the stall lacks: pages, or the state
                // slots (windows included) a stuck restore waits on.
                let holds = match cause {
                    StarveCause::NoRsSlots => !held_rs_units(*pid, model, engine).is_zero(),
                    _ => matches!(quote, Some(ReclaimQuote::Pages(pages)) if pages > 0),
                };
                if holds {
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

    fn check_hog(self: &Arc<Self>, head_key: EntryKey, head_pid: ProcessId) -> bool {
        let transfers_in_flight = self.with_inner(|inner| {
            !inner.evicting.is_empty()
                || inner
                    .procs
                    .values()
                    .any(|proc| proc.state == Residency::Restoring)
        });
        if transfers_in_flight {
            return false;
        }
        let (model, engine) = self.port.locus();
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
                return None;
            };
            let need = demand.kv_pages;
            if outcome.is_some() || need.saturating_add(held) <= total {
                return None;
            }
            *outcome = Some(Err(PlannerError::Hog { need, held, total }));
            Some(notify.clone())
        });
        let Some(notify) = notify else {
            return false;
        };
        self.stats.hog_failures.fetch_add(1, Ordering::Relaxed);
        notify.notify_waiters();
        true
    }

    fn report_evicted(self: &Arc<Self>, pid: ProcessId, freed: u32, rs_freed: RsUnits) {
        self.with_inner(|inner| {
            inner.settle_eviction(pid);
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return;
            };
            debug_assert_eq!(proc.state, Residency::Evicting);
            proc.state = Residency::Evicted;
            let demand = Demand {
                kv_pages: freed,
                rs: rs_freed,
            };
            if proc.idle || proc.held {
                proc.parked = Some(demand);
                return;
            }
            inner.enqueue(pid, WaitKind::Restore { demand });
        });
        crate::policy::report(crate::policy::Event::Paused(pid));
        self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(target: "pie::kv", kv_pages = freed, "paused");
        self.vacate_held_seats();
        self.re_arm_idle_reclaim();
        self.poke();
    }

    fn eviction_failed(self: &Arc<Self>, pid: ProcessId) {
        self.eviction_failed_inner(pid, false, false);
    }

    fn eviction_found_nothing(self: &Arc<Self>, pid: ProcessId) {
        self.eviction_failed_inner(pid, false, true);
    }

    fn eviction_failed_host_swap_full(self: &Arc<Self>, pid: ProcessId) {
        self.record_host_swap_exhaustion();
        self.eviction_failed_inner(pid, true, false);
    }

    fn eviction_failed_inner(
        self: &Arc<Self>,
        pid: ProcessId,
        host_swap_full: bool,
        found_nothing: bool,
    ) {
        let signal = self.with_inner(|inner| {
            inner.settle_eviction(pid);
            if host_swap_full {
                inner.host_swap_blocked.insert(pid);
            }
            if found_nothing {
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
        crate::scheduler::worker::notify_process_resume(pid);
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        self.poke();
    }

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

    fn report_restored(self: &Arc<Self>, pid: ProcessId) {
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
            proc.progressed = false;
            Some(proc.signal.clone())
        });
        crate::scheduler::worker::notify_process_resume(pid);
        crate::policy::report(crate::policy::Event::Resumed(pid));
        self.stats.restores.fetch_add(1, Ordering::Relaxed);
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        self.poke();
    }

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
            self.poke();
        }
    }

    /// A `failure` spends its one retry; none (pages it shares moved since its
    /// grant) is no failure.
    fn requeue_restore(self: &Arc<Self>, pid: ProcessId, failure: Option<&str>) {
        let spent = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return false;
            };
            if proc.state != Residency::Restoring {
                return false;
            }
            proc.state = Residency::Evicted;
            if failure.is_some() {
                if proc.restore_retried {
                    return true;
                }
                proc.restore_retried = true;
            }
            inner.enqueue(
                pid,
                WaitKind::Restore {
                    demand: Demand::default(),
                },
            );
            false
        });
        if failure.is_some() {
            self.stats.restore_failures.fetch_add(1, Ordering::Relaxed);
        }
        match failure {
            Some(reason) if spent => self.fail_restore_loud(pid, reason),
            _ => self.poke(),
        }
    }

    fn fail_restore_loud(self: &Arc<Self>, pid: ProcessId, reason: &str) {
        let reason = format!("KV restore failed: {reason}");
        tracing::error!(pid = %pid, %reason, "planner: failing evicted process loud");
        crate::inferlet::process::terminate_by(pid, Err(reason), crate::policy::By::Host);
    }

    fn record_host_swap_exhaustion(&self) {
        self.stats
            .host_swap_exhaustions
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Counts one engine copy of a residency move and, for kv, its time.
    pub(crate) fn record_tier_copy(&self, copy: &crate::scheduler::TierMove, elapsed: Duration) {
        use crate::scheduler::TierKind::{Kv, Rs, RsPages, Window};
        use ::engine::transfer::MemoryDomain::LocalDisk;
        let stats = &self.stats;
        let (count, us) = match (copy.kind, copy.tier, copy.out) {
            (Kv, LocalDisk, true) => (&stats.disk_write_pages, Some(&stats.disk_write_us)),
            (Kv, LocalDisk, false) => (&stats.disk_read_pages, Some(&stats.disk_read_us)),
            (Kv, _, true) => (&stats.d2h_pages, Some(&stats.d2h_copy_us)),
            (Kv, _, false) => (&stats.h2d_pages, Some(&stats.h2d_copy_us)),
            (Rs | RsPages | Window, _, true) => (&stats.d2h_rs_slots, None),
            (Rs | RsPages | Window, _, false) => (&stats.h2d_rs_slots, None),
        };
        count.fetch_add(copy.device.len() as u64, Ordering::Relaxed);
        if let Some(us) = us {
            let micros = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
            us.fetch_add(micros, Ordering::Relaxed);
        }
    }

    pub fn diagnostics(&self) -> PlannerDiagnostics {
        let (device_pages_free, device_pages_total) = self.port.device_stats();
        let (host_slots_free, host_slots_total) = self.port.host_stats();
        let (disk_slots_free, disk_slots_total) = self.port.disk_stats();
        let (rs_free, rs_total) = self.port.rs_stats();
        let inner = self.lock_inner();
        let queue = inner
            .queue
            .iter()
            .map(|(key, waiter)| PlannerQueueEntry {
                process_id: waiter.pid.to_string(),
                spawn_seq: key.0.seq,
                kind: waiter.kind_name(),
                pages: waiter.kv_need(),
                rs: waiter.rs_need(),
            })
            .collect();
        let head = inner.unmet_head();
        let (unmet_head_pages, unmet_head_kind) = head.map_or((0, "-"), |(_, waiter)| {
            (waiter.kv_need(), waiter.kind_name())
        });
        let unmet_queued = inner
            .queue
            .values()
            .filter(|waiter| waiter.is_unmet())
            .count() as u32;
        let (bypassable_entries, bypassable_pages) = {
            match head.map(|(key, _)| key.0) {
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
            disk_slots_free,
            disk_slots_total,
            rs_free,
            rs_total,
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
            disk_write_pages_total: self.stats.disk_write_pages.load(Relaxed),
            disk_read_pages_total: self.stats.disk_read_pages.load(Relaxed),
            disk_write_us_total: self.stats.disk_write_us.load(Relaxed),
            disk_read_us_total: self.stats.disk_read_us.load(Relaxed),
            d2h_rs_slots_total: self.stats.d2h_rs_slots.load(Relaxed),
            h2d_rs_slots_total: self.stats.h2d_rs_slots.load(Relaxed),
            d2h_copy_us_total: self.stats.d2h_copy_us.load(Relaxed),
            h2d_copy_us_total: self.stats.h2d_copy_us.load(Relaxed),
        }
    }

    pub fn debug_queue(&self) -> String {
        self.with_inner(|inner| {
            inner
                .queue
                .iter()
                .map(|(key, waiter)| {
                    let kind = match &waiter.kind {
                        WaitKind::Allocation {
                            demand,
                            outcome,
                            yielded,
                            ..
                        } => format!(
                            "alloc kv={:?} rs={} outcome={} yielded={yielded}",
                            demand.kv_pages,
                            demand.rs,
                            match outcome {
                                None => "none",
                                Some(Ok(_)) => "granted",
                                Some(Err(_)) => "err",
                            }
                        ),
                        WaitKind::Restore { demand } => {
                            format!("restore kv={} rs={}", demand.kv_pages, demand.rs)
                        }
                    };
                    format!("{key:?} pid={} {kind}", waiter.pid)
                })
                .collect::<Vec<_>>()
                .join("; ")
        })
    }

    pub fn kv_pressure_bucket(&self) -> u8 {
        let (swap_free, swap_total) = self.swap_stats();
        let ratio = |(free, total): (u32, u32)| {
            if total == 0 {
                0.0
            } else {
                f64::from(total.saturating_sub(free)) / f64::from(total)
            }
        };
        // Pages an idle process holds are headroom: the next demand evicts
        // them first. Counted as pressure, they would hold a launch at the
        // gateway that no fire is left to free them for. So are the pages
        // only an unused cache lease keeps (a finished process's indexed
        // prefix): the next reservation drops the lease, but a launch the
        // gateway holds back on their account never makes one.
        let idle: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .filter(|(_, proc)| proc.idle && proc.state == Residency::Resident)
                .map(|(pid, _)| *pid)
                .collect()
        });
        let (model, engine) = self.port.locus();
        let reclaimable: u32 = idle
            .into_iter()
            .map(|pid| held_page_count(pid, model, engine))
            .sum::<u32>()
            .saturating_add(self.port.reclaimable_idle());
        let (free, total) = self.port.device_stats();
        let device = ratio((free.saturating_add(reclaimable).min(total), total));
        let mut bucket = (device.max(ratio((swap_free, swap_total))) * 255.0).round() as u8;
        if self.nonresident.load(Ordering::Acquire) != 0
            || self.waiters.load(Ordering::Acquire) != 0
        {
            bucket = bucket.max(224);
        }
        bucket
    }
}

enum Collect {
    Ready(Result<AllocationGrant, PlannerError>),
    Yield,
    Requote,
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

fn swapped_page_count(pid: ProcessId, model: usize, engine: usize) -> u32 {
    kv_page_count(pid, model, engine, |kv, ws| {
        kv.swapped_page_count(ws).unwrap_or(0)
    })
}

fn held_page_count(pid: ProcessId, model: usize, engine: usize) -> u32 {
    kv_page_count(pid, model, engine, |kv, ws| {
        kv.held_page_count(ws).unwrap_or(0)
    })
}

fn held_rs_units(pid: ProcessId, model: usize, engine: usize) -> RsUnits {
    rs_units(pid, model, engine, |rs, ws| rs.held(ws.iter().copied()))
}

fn swapped_rs_units(pid: ProcessId, model: usize, engine: usize) -> RsUnits {
    rs_units(pid, model, engine, |rs, ws| rs.swapped_units(ws))
}

fn suspendable_rs_units(pid: ProcessId, model: usize, engine: usize) -> RsUnits {
    rs_units(pid, model, engine, |rs, ws| rs.suspendable(ws))
}

fn rs_units(
    pid: ProcessId,
    model: usize,
    engine: usize,
    count: impl FnOnce(
        &crate::store::rs::RsStore,
        &HashSet<crate::store::rs::RsWorkingSetId>,
    ) -> RsUnits,
) -> RsUnits {
    let working_sets = crate::inferlet::process::residency::rs_working_set_ids(pid, model, engine);
    if working_sets.is_empty() {
        return RsUnits::default();
    }
    let Some(stores) = crate::store::registry::try_get(model, engine) else {
        return RsUnits::default();
    };
    count(&stores.rs.lock().unwrap(), &working_sets)
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

type PlannerMap = HashMap<(usize, usize), Arc<ResidencyPlanner>>;

static PRIMARY: OnceLock<Arc<ResidencyPlanner>> = OnceLock::new();
static PLANNERS: OnceLock<RwLock<PlannerMap>> = OnceLock::new();

pub fn init_planner(model: usize, engine: usize, planner: ResidencyPlanner) {
    let planner = Arc::new(planner);
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

/// A message for `pid` is on its way: if it was parked, its restore starts
/// now rather than when it reads the message.
pub fn wake(pid: ProcessId) {
    if let Some(planner) = planner() {
        planner.wake(pid);
    }
}

pub fn planner() -> Option<&'static Arc<ResidencyPlanner>> {
    PRIMARY.get()
}

#[cfg(test)]
mod service_order_tests {
    use super::*;
    use crate::planner::grant::Demand;

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

    fn park_allocation(inner: &mut Inner, pid: ProcessId, kv_pages: u32) -> EntryKey {
        inner.enqueue(
            pid,
            WaitKind::Allocation {
                demand: Demand {
                    kv_pages,
                    rs: RsUnits::default(),
                },
                notify: Arc::new(Notify::new()),
                outcome: None,
                yielded: false,
            },
        )
    }

    fn park_restore(inner: &mut Inner, pid: ProcessId, kv_pages: u32) -> EntryKey {
        inner.enqueue(
            pid,
            WaitKind::Restore {
                demand: Demand {
                    kv_pages,
                    rs: RsUnits::default(),
                },
            },
        )
    }

    #[test]
    fn planner_every_case() {
        a_restore_yields_the_head_to_a_younger_allocation();
        a_restore_takes_the_head_once_the_fleet_has_stalled();
        a_restore_holds_the_head_when_no_allocation_is_unmet();
        a_younger_restore_still_queues_behind_an_older_allocation();
        returning_host_room_resumes_the_yield();
    }

    fn a_restore_yields_the_head_to_a_younger_allocation() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Evicted, true),
            (2, Residency::Resident, true),
            (3, Residency::Resident, true),
        ]);
        park_restore(&mut inner, pids[0], 18);
        park_allocation(&mut inner, pids[2], 1);

        let (key, waiter) = inner.unmet_head().expect("a head");
        assert_eq!(key.0.seq, 3, "the younger allocation must take the head");
        assert!(matches!(waiter.kind, WaitKind::Allocation { .. }));
    }

    fn a_restore_takes_the_head_once_the_fleet_has_stalled() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Evicted, true),
            (3, Residency::Resident, true),
        ]);
        park_restore(&mut inner, pids[0], 18);
        park_allocation(&mut inner, pids[1], 1);

        assert!(inner.fleet_stalled(), "every admitted resident is parked");
        let (key, waiter) = inner.unmet_head().expect("a head");
        assert_eq!(key.0.seq, 1, "the older restore must take the head");
        assert!(matches!(waiter.kind, WaitKind::Restore { .. }));
    }

    fn a_restore_holds_the_head_when_no_allocation_is_unmet() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Evicted, true),
            (3, Residency::Resident, true),
        ]);
        park_restore(&mut inner, pids[0], 18);
        let parked = park_allocation(&mut inner, pids[1], 1);
        let Some(Waiter {
            kind: WaitKind::Allocation { outcome, .. },
            ..
        }) = inner.queue.get_mut(&parked)
        else {
            unreachable!("just parked an allocation");
        };
        *outcome = Some(Err(PlannerError::Cancelled));

        assert!(!inner.fleet_stalled(), "a served grant is uncollected");
        let (key, waiter) = inner.unmet_head().expect("a head");
        assert_eq!(key.0.seq, 1, "the restore is the only unmet entry");
        assert!(matches!(waiter.kind, WaitKind::Restore { .. }));
    }

    fn a_younger_restore_still_queues_behind_an_older_allocation() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Resident, true),
            (2, Residency::Resident, true),
            (3, Residency::Evicted, true),
        ]);
        park_allocation(&mut inner, pids[0], 4);
        park_restore(&mut inner, pids[2], 18);

        let (key, waiter) = inner.unmet_head().expect("a head");
        assert_eq!(key.0.seq, 1);
        assert!(matches!(waiter.kind, WaitKind::Allocation { .. }));
    }

    fn returning_host_room_resumes_the_yield() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Evicted, true),
            (2, Residency::Resident, true),
            (3, Residency::Resident, true),
        ]);
        park_restore(&mut inner, pids[0], 18);
        park_allocation(&mut inner, pids[2], 1);
        inner.host_swap_blocked.insert(pids[1]);
        assert_eq!(inner.unmet_head().expect("a head").0.0.seq, 1);

        inner.host_swap_blocked.clear();

        assert!(!inner.eviction_unfundable());
        assert_eq!(
            inner.unmet_head().expect("a head").0.0.seq,
            3,
            "the allocation takes the head again"
        );
    }

    #[test]
    fn a_key_outranks_age_in_the_queue_and_among_victims() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Resident, true),
            (2, Residency::Resident, true),
        ]);
        let (old, young) = (pids[0], pids[1]);
        let old_key = park_allocation(&mut inner, old, 4);
        let young_key = park_allocation(&mut inner, young, 4);
        assert_eq!(inner.unmet_head().unwrap().1.pid, old, "no schedule: age");

        inner.reschedule(|pid| (if *pid == young { 5 } else { 0 }, false));
        let (head, waiter) = inner.unmet_head().unwrap();
        assert_eq!(waiter.pid, young, "the higher key takes the head");
        assert!(
            inner.procs[&old].rank() > head.0,
            "the older, lower-keyed process may now be preempted for it"
        );
        assert!(inner.procs[&young].rank() <= head.0);
        for handed_out in [old_key, young_key] {
            assert_eq!(
                inner.locate(handed_out).map(|(_, id)| id),
                Some(handed_out.1),
                "a key handed out before the re-sort still finds its entry"
            );
        }

        inner.reschedule(|_| (0, false));
        assert_eq!(
            inner.unmet_head().unwrap().1.pid,
            old,
            "an empty schedule is age again"
        );
    }

    #[test]
    fn the_floor_parks_a_restore_and_lifting_it_queues_the_restore_again() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Evicted, true),
            (2, Residency::Resident, true),
        ]);
        let (paused, running) = (pids[0], pids[1]);
        park_restore(&mut inner, paused, 3);
        let running_key = park_allocation(&mut inner, running, 2);

        inner.reschedule(|pid| (0, *pid == paused || *pid == running));
        assert!(
            inner.queue.values().all(|w| w.pid != paused),
            "a held process is not restored"
        );
        assert_eq!(inner.procs[&paused].parked.map(|d| d.kv_pages), Some(3));
        assert!(
            inner.preempted_floor().is_none(),
            "nor does it hold others back"
        );
        let head = inner.unmet_head().unwrap().0;
        assert_eq!(head, running_key);
        assert!(inner.procs[&running].held, "held, so a victim at any rank");

        inner.reschedule(|_| (0, false));
        assert!(inner.procs[&paused].parked.is_none());
        assert!(
            inner
                .queue
                .values()
                .any(|w| w.pid == paused && matches!(w.kind, WaitKind::Restore { .. })),
            "the lifted floor queues its restore"
        );
    }

    #[test]
    fn a_preempted_process_holds_back_only_what_it_outranks() {
        let (mut inner, pids) = fleet(&[
            (1, Residency::Evicted, true),
            (2, Residency::Resident, true),
        ]);
        let newcomer = pids[1];
        let floor = inner.preempted_floor().unwrap();
        assert!(
            inner.procs[&newcomer].rank() > floor,
            "no schedule: the newcomer waits for the older preempted process"
        );

        inner.reschedule(|pid| (if *pid == newcomer { 5 } else { 0 }, false));
        let floor = inner.preempted_floor().unwrap();
        assert!(
            inner.procs[&newcomer].rank() <= floor,
            "a higher key starts ahead of a lower-keyed preempted process"
        );
    }
}

#[cfg(test)]
mod starvation_race_tests {
    use super::*;
    use crate::planner::grant::Demand;
    use crate::store::kv::page_table::PhysicalKvPageId;
    use std::sync::atomic::AtomicU32;

    fn allocation(demand: Demand) -> WaitKind {
        WaitKind::Allocation {
            demand,
            notify: Arc::new(Notify::new()),
            outcome: None,
            yielded: false,
        }
    }

    struct RacePool {
        free: parking_lot::Mutex<Vec<PhysicalKvPageId>>,
        total: u32,
        stall: AtomicU32,
        host: u32,
        pulls: parking_lot::Mutex<Vec<u32>>,
        host_total: u32,
        // A registered model whose RS store backs the slot calls.
        rs_model: Option<usize>,
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
                rs_model: None,
            }
        }

        fn with_rs<R>(&self, f: impl FnOnce(&mut crate::store::rs::RsStore) -> R) -> Option<R> {
            let stores = crate::store::registry::get(self.rs_model?, 0);
            let mut rs = stores.rs.lock().unwrap();
            Some(f(&mut rs))
        }

        fn with_swap(total: u32, host: u32) -> Self {
            Self {
                host,
                host_total: host,
                ..Self::new(total)
            }
        }

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
        fn disk_stats(&self) -> (u32, u32) {
            (0, 0)
        }
        fn rs_stats(&self) -> (RsUnits, RsUnits) {
            self.with_rs(|rs| (rs.available(), rs.capacity()))
                .unwrap_or_default()
        }
        fn rs_host_stats(&self) -> (RsUnits, RsUnits) {
            Default::default()
        }
        fn reclaim_idle(&self) -> u32 {
            0
        }
        fn reclaimable_idle(&self) -> u32 {
            0
        }
        fn suspend_capable(&self) -> bool {
            self.host_total > 0
        }
        fn locus(&self) -> (usize, usize) {
            (self.rs_model.unwrap_or(0), 0)
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
        fn reserve_rs(&self, need: RsUnits) -> Option<crate::store::rs::RsGrant> {
            self.with_rs(|rs| rs.reserve(need))
                .unwrap_or_else(|| Some(Default::default()))
        }
        fn release_rs(&self, grant: crate::store::rs::RsGrant) {
            self.with_rs(|rs| rs.release_reservation(grant));
        }
        fn yield_checkpoints(&self, working_sets: &[crate::store::rs::RsWorkingSetId]) -> usize {
            self.with_rs(|rs| rs.yield_checkpoints(working_sets))
                .unwrap_or(0)
        }
        fn prepare_sweep(
            &self,
            _keep: &HashSet<crate::store::kv::page_table::WorkingSetId>,
            _room: usize,
        ) -> Option<crate::store::kv::KvResidencyTxn> {
            None
        }
        fn prepare_demote(&self, _room: usize) -> Option<crate::store::kv::KvResidencyTxn> {
            None
        }
    }

    #[test]
    fn a_younger_process_is_not_admitted_while_an_older_one_is_preempted() {
        use futures::FutureExt;
        let planner = Arc::new(ResidencyPlanner::new(
            Arc::new(RacePool::new(8)) as Arc<dyn PoolPort>
        ));
        let [evicted, killed, younger] = [(); 3].map(|_| ProcessId::new_v4());
        for pid in [evicted, killed, younger] {
            planner.register(pid);
        }
        planner
            .with_inner(|inner| inner.procs.get_mut(&evicted).unwrap().state = Residency::Evicted);
        let mut admit = Box::pin(planner.admit(younger));
        assert!(admit.as_mut().now_or_never().is_none());

        planner.with_inner(|inner| {
            inner.procs.get_mut(&evicted).unwrap().state = Residency::Resident;
            inner.killing.insert(killed);
        });
        assert!(admit.as_mut().now_or_never().is_none());

        let respawn = ProcessId::new_v4();
        planner.register_with_seq(respawn, planner.spawn_seq(killed).unwrap());
        planner.unregister(killed);
        assert!(planner.admit(respawn).now_or_never().is_some());
        assert!(admit.as_mut().now_or_never().is_none());
        planner.note_admitted(respawn);
        assert!(admit.as_mut().now_or_never().is_some());
    }

    #[test]
    fn a_head_short_of_state_beside_free_pages_is_funded_by_eviction() {
        // The head lacks a state slot while the device has kv pages to
        // spare: the pages a queued decode step asks for are supply, not a
        // deficit the eviction planner should defer on.
        let model = crate::store::registry::register_model(16, &[64], &[2]);
        let stores = crate::store::registry::get(model, 0);
        {
            let mut rs = stores.rs.lock().unwrap();
            let ws = rs.create_working_set(crate::store::rs::RsGeometry {
                state_size: 64,
                buffer_page_tokens: 4,
            });
            let folded = rs.prepare_group(ws, true, false, None, None, None).unwrap();
            let published = rs.publish_prepared(folded).unwrap();
            rs.settle(published);
        }
        let pool = Arc::new(RacePool {
            rs_model: Some(model),
            ..RacePool::new(64)
        });
        pool.refill_after_stall(64, 0);
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
        let (head, younger) = (ProcessId::new_v4(), ProcessId::new_v4());
        for pid in [head, younger] {
            planner.register(pid);
            planner.note_admitted(pid);
        }
        planner.with_inner(|inner| {
            for (pid, kv_pages, slots) in [(head, 0, 2), (younger, 1, 0)] {
                let rs = RsUnits { slots, pages: 0 };
                inner.enqueue(pid, allocation(Demand { kv_pages, rs }));
            }
        });
        let victims = planner.victim_set().expect("the head is short");
        assert_eq!(
            victims.deficit,
            Reclaim {
                pages: 0,
                rs: RsUnits { slots: 1, pages: 0 }
            }
        );
        assert_eq!(victims.members.len(), 1);
        assert_eq!(victims.members[0].pid, younger);
    }

    #[test]
    fn a_wedged_head_short_of_state_alone_still_reclaims_by_restart() {
        // The head asks for a state slot and no pages: the pages it has
        // do not cover it, so a stalled fleet with nothing to evict reclaims
        // the youngest ask rather than waiting on a serve that cannot come.
        let model = crate::store::registry::register_model(16, &[64], &[2]);
        let stores = crate::store::registry::get(model, 0);
        let held = stores
            .rs
            .lock()
            .unwrap()
            .reserve(RsUnits { slots: 2, pages: 0 })
            .unwrap();
        let pool = Arc::new(RacePool {
            rs_model: Some(model),
            ..RacePool::new(8)
        });
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
        let (head, younger) = (ProcessId::new_v4(), ProcessId::new_v4());
        let mut keys = Vec::new();
        planner.with_inner(|inner| {
            for (pid, kv_pages, slots) in [(head, 0, 1), (younger, 1, 0)] {
                let rs = RsUnits { slots, pages: 0 };
                let seq = inner.next_seq;
                inner.next_seq += 1;
                let mut proc = Proc::new(seq);
                proc.admitted = true;
                inner.procs.insert(pid, proc);
                keys.push(inner.enqueue(pid, allocation(Demand { kv_pages, rs })));
            }
        });
        assert!(planner.is_wedged());
        planner.check_starvation(StarveCause::NoEligibleVictim);
        planner.with_inner(|inner| {
            assert!(matches!(
                inner.queue[&keys[0]].kind,
                WaitKind::Allocation { outcome: None, .. }
            ));
            assert!(matches!(
                inner.queue[&keys[1]].kind,
                WaitKind::Allocation {
                    outcome: Some(Err(PlannerError::Starved { .. })),
                    ..
                }
            ));
        });
        stores.rs.lock().unwrap().release_reservation(held);
    }

    #[test]
    fn a_burst_serves_exactly_the_fundable_prefix_in_one_pass() {
        let pool = Arc::new(RacePool::new(64));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));

        let holder = ProcessId::new_v4();
        planner.register(holder);
        planner.note_admitted(holder);

        let demand = Demand {
            kv_pages: 1,
            rs: RsUnits::default(),
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
                .map(|&pid| inner.enqueue(pid, allocation(demand)))
                .collect()
        });

        planner.with_inner(|inner| {
            let waiter = inner.queue.get_mut(&keys[2]).expect("parked");
            let WaitKind::Allocation { yielded, .. } = &mut waiter.kind else {
                unreachable!()
            };
            *yielded = true;
        });

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

    #[tokio::test(flavor = "current_thread")]
    async fn a_host_swap_full_victim_is_parked_until_host_room_returns() {
        let pool = Arc::new(RacePool::new(4));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone()));

        let head = ProcessId::new_v4();
        let victim = ProcessId::new_v4();
        for pid in [head, victim] {
            planner.register(pid);
            planner.note_admitted(pid);
        }

        let p = planner.clone();
        let head_task = crate::rt::spawn(async move {
            p.acquire(
                head,
                head,
                Demand {
                    kv_pages: 1,
                    rs: RsUnits::default(),
                },
            )
            .await
        });
        for _ in 0..200 {
            crate::rt::yield_now().await;
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

        planner.eviction_failed_host_swap_full(victim);
        assert!(
            members(&planner).is_empty(),
            "a host-swap-blocked victim must not be re-picked"
        );
        assert_eq!(planner.diagnostics().host_swap_exhaustions_total, 1);

        planner.clear_host_swap_blocks();
        assert_eq!(
            members(&planner),
            vec![victim],
            "returned host room must re-arm the parked victim"
        );
        assert_eq!(planner.diagnostics().host_swap_unblocks_total, 1);

        head_task.abort();
    }

    #[test]
    fn a_wake_restores_a_parked_process_that_stays_idle() {
        let planner = Arc::new(ResidencyPlanner::new(
            Arc::new(RacePool::new(4)) as Arc<dyn PoolPort>
        ));
        let agent = ProcessId::new_v4();
        planner.register(agent);
        planner.note_admitted(agent);
        planner.set_idle(agent, true);
        planner
            .with_inner(|inner| inner.procs.get_mut(&agent).unwrap().state = Residency::Evicting);
        planner.report_evicted(agent, 3, RsUnits::default());
        assert!(planner.diagnostics().queue.is_empty());

        planner.wake(agent);
        let (idle, restores) = planner.with_inner(|inner| {
            let restores = inner
                .queue
                .values()
                .map(|w| match w.kind {
                    WaitKind::Restore { demand } if w.pid == agent => demand.kv_pages,
                    _ => 0,
                })
                .collect::<Vec<_>>();
            (inner.procs[&agent].idle, restores)
        });
        assert_eq!(
            restores,
            vec![3],
            "a message on its way restores what it parked"
        );
        assert!(
            idle,
            "and it stays idle, the first victim, until it reads the message"
        );

        planner.wake(agent);
        assert_eq!(
            planner.diagnostics().queue.len(),
            1,
            "a second wake queues nothing more"
        );
    }

    #[test]
    fn an_idle_eviction_asks_for_its_pages_back_only_on_waking() {
        let planner = Arc::new(ResidencyPlanner::new(
            Arc::new(RacePool::new(4)) as Arc<dyn PoolPort>
        ));
        let agent = ProcessId::new_v4();
        planner.register(agent);
        planner.note_admitted(agent);
        assert!(!planner.with_inner(|inner| inner.fleet_stalled()));

        planner.set_idle(agent, true);
        assert!(
            planner.with_inner(|inner| inner.fleet_stalled()),
            "a resident waiting on its client does not keep the fleet running"
        );

        planner
            .with_inner(|inner| inner.procs.get_mut(&agent).unwrap().state = Residency::Evicting);
        planner.report_evicted(agent, 3, RsUnits::default());
        assert!(
            planner.diagnostics().queue.is_empty(),
            "an idle eviction queues no restore"
        );

        planner.set_idle(agent, false);
        let restores = planner.with_inner(|inner| {
            inner
                .queue
                .values()
                .map(|w| match w.kind {
                    WaitKind::Restore { demand } if w.pid == agent => demand.kv_pages,
                    _ => 0,
                })
                .collect::<Vec<_>>()
        });
        assert_eq!(restores, vec![3], "waking asks for what the eviction freed");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn an_exhausted_pool_evicts_even_while_the_fleet_still_runs() {
        let pool = Arc::new(RacePool::with_swap(4, 64));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone()));

        let holder = ProcessId::new_v4();
        planner.register(holder);
        planner.note_admitted(holder);

        let head = ProcessId::new_v4();
        planner.register(head);
        planner.note_admitted(head);
        for _ in 0..3 {
            let pid = ProcessId::new_v4();
            planner.register(pid);
            planner.note_admitted(pid);
        }

        let demand = Demand {
            kv_pages: 1,
            rs: RsUnits::default(),
        };
        let p = planner.clone();
        let parked = crate::rt::spawn(async move { p.acquire(head, head, demand).await });
        for _ in 0..200 {
            crate::rt::yield_now().await;
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

        let set = planner.victim_set().expect("an unmet head yields a set");
        assert_eq!(set.deficit.pages, 1, "the round is demand-exact");
        assert!(
            !set.members.is_empty(),
            "younger residents are legal victims"
        );

        assert!(!planner.is_wedged(), "a running holder is not a wedge");
        assert!(planner.supply_stalled(), "an empty pool with an unmet head");
        assert_eq!(
            planner.diagnostics().eviction_deferrals_total,
            0,
            "the rung must not have deferred"
        );

        let mark = ProcessId::new_v4();
        planner.with_inner(|inner| {
            inner.evicting.insert(
                mark,
                Reclaim {
                    pages: 1,
                    rs: RsUnits::default(),
                },
            );
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

    #[tokio::test(flavor = "current_thread")]
    async fn a_lane_holding_the_whole_state_pool_is_told_so() {
        use crate::inferlet::process::residency::{ProcessResidency, register_residency};
        use crate::store::rs::RsGeometry;

        // One process holds both slots of a two-slot pool and asks for a
        // third: nobody else's to come back, so the refusal names it (#686).
        let model = crate::store::registry::register_model(16, &[4], &[2]);
        let stores = crate::store::registry::get(model, 0);
        let held: Vec<_> = (0..2)
            .map(|_| {
                let mut rs = stores.rs.lock().unwrap();
                let ws = rs.create_working_set(RsGeometry {
                    state_size: 64,
                    buffer_page_tokens: 4,
                });
                let folded = rs.prepare_group(ws, true, false, None, None, None).unwrap();
                let published = rs.publish_prepared(folded).unwrap();
                rs.settle(published);
                ws
            })
            .collect();
        let pid = ProcessId::new_v4();
        let residency = Arc::new(std::sync::Mutex::new(ProcessResidency::default()));
        residency
            .lock()
            .unwrap()
            .rs_working_sets
            .extend(held.into_iter().map(|ws| (model, 0, ws)));
        register_residency(pid, Arc::downgrade(&residency));

        let pool = RacePool {
            rs_model: Some(model),
            ..RacePool::new(0)
        };
        let planner = Arc::new(ResidencyPlanner::new(Arc::new(pool)));
        planner.register(pid);
        planner.note_admitted(pid);
        let error = planner
            .acquire(
                pid,
                pid,
                Demand {
                    kv_pages: 0,
                    rs: RsUnits { slots: 1, pages: 0 },
                },
            )
            .await
            .err()
            .expect("a pool the lane already holds cannot serve it");
        assert!(
            matches!(
                error,
                PlannerError::RsHog { need, held, total }
                    if need.slots == 1 && held.slots == 2 && total.slots == 2
            ),
            "{error}"
        );
    }

    /// A ring model's store: a window of 32 tokens over 16-token pages,
    /// 8 windowed page ids of which 0 is the null page.
    fn ring_model() -> usize {
        crate::store::registry::register_model_with_swap(
            16,
            &[64],
            &[0],
            &[0],
            &[crate::store::registry::StatePool {
                window_tokens: 32,
                window_pages: 8,
                ..Default::default()
            }],
            &[0],
        )
    }

    /// The pages a fire of `rows` tokens from `held` writes, and how it
    /// moves its ring when it commits them all.
    fn ring_fire(held: u32, rows: u32) -> (Vec<u32>, crate::store::rs::RingAdvance) {
        let end = held + rows;
        (
            (held / 16..=(end - 1) / 16).collect(),
            crate::store::rs::RingAdvance {
                positioned: true,
                settled: held,
                fold: Some(u32::MAX),
                commit_end: end,
                head: Some(end),
            },
        )
    }

    fn ring_demand(
        stores: &crate::store::registry::Stores,
        ws: crate::store::rs::RsWorkingSetId,
        held: u32,
        rows: u32,
    ) -> Demand {
        let (pages, _) = ring_fire(held, rows);
        Demand {
            kv_pages: 0,
            rs: stores.rs.lock().unwrap().ring_demand(ws, &pages).unwrap(),
        }
    }

    /// Fires from the grant; the published write is settled unless `hold`.
    fn ring_write(
        stores: &crate::store::registry::Stores,
        ws: crate::store::rs::RsWorkingSetId,
        held: u32,
        rows: u32,
        mut grant: AllocationGrant,
        hold: bool,
    ) -> Option<crate::store::rs::write::RsPublished> {
        let (pages, advance) = ring_fire(held, rows);
        let mut rs = stores.rs.lock().unwrap();
        let prepared = rs
            .prepare_ring(ws, &pages, Some(grant.lend_rs()))
            .expect("the grant covers the claim");
        let published = rs.publish_prepared(prepared).unwrap();
        rs.ring_advance(ws, advance, &(0..64).collect::<Vec<u32>>());
        if hold {
            return Some(published);
        }
        rs.settle(published);
        None
    }

    // #699: a fire short of windowed pages parks in the planner like a
    // kv-short one, and is served once another sequence's window moves on
    // and the pages behind it retire, which the fire that read them settles.
    #[tokio::test(flavor = "current_thread")]
    async fn a_fire_short_of_windowed_pages_waits_for_a_window_to_move_on() {
        let model = ring_model();
        let stores = crate::store::registry::get(model, 0);
        let geom = crate::store::rs::RsGeometry {
            state_size: 0,
            buffer_page_tokens: 16,
        };
        let (a, b) = {
            let mut rs = stores.rs.lock().unwrap();
            (rs.create_working_set(geom), rs.create_working_set(geom))
        };
        let planner = Arc::new(ResidencyPlanner::new(Arc::new(RegistryPool::new(
            model, 0, false,
        ))));
        let (pa, pb) = (ProcessId::new_v4(), ProcessId::new_v4());
        for pid in [pa, pb] {
            planner.register(pid);
            planner.note_admitted(pid);
        }
        let granted = |acquired: Result<Acquired, PlannerError>| match acquired {
            Ok(Acquired::Granted(grant)) => grant,
            _ => panic!("a grant"),
        };

        let prefill = ring_demand(&stores, a, 0, 96);
        assert_eq!(prefill.rs.pages, 6);
        let in_flight = ring_write(
            &stores,
            a,
            0,
            96,
            granted(planner.acquire(pa, pa, prefill).await),
            true,
        );

        let short = ring_demand(&stores, b, 0, 64);
        assert_eq!(short.rs.pages, 4);
        let p = planner.clone();
        let parked = crate::rt::spawn(async move { p.acquire(pb, pb, short).await });
        for _ in 0..50 {
            crate::rt::yield_now().await;
        }
        assert!(
            !parked.is_finished(),
            "four pages asked, one free: the fire parks"
        );

        // The prefill committed 96 tokens: the four pages behind its window
        // retire once it settles.
        stores.rs.lock().unwrap().settle(in_flight.unwrap());
        planner.pages_freed();
        for _ in 0..50 {
            if parked.is_finished() {
                break;
            }
            crate::rt::yield_now().await;
        }
        assert!(
            parked.is_finished(),
            "the retired pages serve the parked fire"
        );
        let grant = granted(parked.await.expect("the parked task"));
        assert_eq!(grant.remaining_rs().pages, 4);
        ring_write(&stores, b, 0, 64, grant, false);
    }

    // A lone sequence is served by the pages its own window moved past,
    // with no other fire left to retire them, and a window that cannot
    // fit beside what it holds is refused rather than restarted forever.
    #[tokio::test(flavor = "current_thread")]
    async fn a_lone_window_is_served_by_what_it_moved_past_or_told_it_never_fits() {
        use crate::inferlet::process::residency::{ProcessResidency, register_residency};

        let model = ring_model();
        let stores = crate::store::registry::get(model, 0);
        let ws = stores
            .rs
            .lock()
            .unwrap()
            .create_working_set(crate::store::rs::RsGeometry {
                state_size: 0,
                buffer_page_tokens: 16,
            });
        let pid = ProcessId::new_v4();
        let residency = Arc::new(std::sync::Mutex::new(ProcessResidency::default()));
        residency
            .lock()
            .unwrap()
            .rs_working_sets
            .insert((model, 0, ws));
        register_residency(pid, Arc::downgrade(&residency));
        let planner = Arc::new(ResidencyPlanner::new(Arc::new(RegistryPool::new(
            model, 0, false,
        ))));
        planner.register(pid);
        planner.note_admitted(pid);
        let fire = |held: u32, rows: u32| {
            let planner = planner.clone();
            let stores = stores.clone();
            async move {
                let demand = ring_demand(&stores, ws, held, rows);
                let Acquired::Granted(grant) = planner.acquire(pid, pid, demand).await? else {
                    panic!("a resident process is never yielded");
                };
                ring_write(&stores, ws, held, rows, grant, false);
                Ok::<(), PlannerError>(())
            }
        };

        fire(0, 96).await.expect("six of seven pages");
        fire(96, 64)
            .await
            .expect("four fresh pages, four retired behind the window");
        let error = fire(160, 96)
            .await
            .expect_err("six fresh beside the two it keeps");
        assert!(
            matches!(
                error,
                PlannerError::RsHog { need, held, total }
                    if need.pages == 6 && held.pages == 2 && total.pages == 7
            ),
            "{error}"
        );
    }
}
