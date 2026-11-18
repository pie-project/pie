//! Admission by room: who runs is the policy's order, how many is what
//! their working sets take of the device ([`ResidencyPlanner::admits`]).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::pool::{ChunkStats, PoolPort};
use super::{ProcessId, ResidencyPlanner};
use crate::inferlet::process::ordered::{OrderedPermit, OrderedSemaphore, Room};
use crate::policy::schedule::{self, Table};
use crate::store::kv::page_table::PhysicalKvPageId;
use crate::store::rs::RsUnits;

/// A device of `total` pages, each process's working set the pages
/// `sizes` names.
struct SizedPool {
    total: u32,
    /// Free host slots: room below the device.
    host: u32,
    /// The bounds each demote was asked under (`None`: unbounded).
    demotes: parking_lot::Mutex<Vec<Option<i64>>>,
    sizes: parking_lot::Mutex<HashMap<ProcessId, u64>>,
}

impl PoolPort for SizedPool {
    fn chunk_stats(&self) -> ChunkStats {
        ChunkStats {
            pages: (self.total, self.total),
            rs: Default::default(),
        }
    }
    fn working_set_pages(&self, pids: &[ProcessId]) -> super::pool::WorkingSetPages {
        let sizes = self.sizes.lock();
        super::pool::WorkingSetPages {
            resident: pids
                .iter()
                .map(|pid| sizes.get(pid).copied().unwrap_or(0))
                .sum(),
            ..Default::default()
        }
    }
    fn host_stats(&self) -> (u32, u32) {
        (self.host, self.host)
    }
    fn disk_stats(&self) -> (u32, u32) {
        (0, 0)
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
    fn idle_below_device(&self) -> (u32, u32) {
        (0, 0)
    }
    fn suspend_capable(&self) -> bool {
        self.host > 0
    }
    fn locus(&self) -> (usize, usize) {
        (0, 0)
    }
    fn reserve_device(&self, _count: u32) -> Option<Vec<PhysicalKvPageId>> {
        None
    }
    fn reserve_device_up_to(&self, _count: u32) -> Vec<PhysicalKvPageId> {
        Vec::new()
    }
    fn release_device(&self, _pages: Vec<PhysicalKvPageId>) {}
    fn reserve_rs(&self, _need: RsUnits) -> Option<crate::store::rs::RsGrant> {
        None
    }
    fn release_rs(&self, _grant: crate::store::rs::RsGrant) {}
    fn reclaim_sharing(&self, _working_sets: &[crate::store::rs::RsWorkingSetId]) -> usize {
        0
    }
    fn prepare_sweep(
        &self,
        _keep: &HashSet<crate::store::kv::page_table::WorkingSetId>,
        _room: usize,
        _order: &[(crate::store::kv::page_table::WorkingSetId, u64)],
        _wanted: usize,
    ) -> Option<crate::store::kv::KvResidencyTxn> {
        None
    }
    fn prepare_demote(
        &self,
        _room: usize,
        bound: Option<crate::store::kv::DemoteBound>,
    ) -> Option<crate::store::kv::KvResidencyTxn> {
        self.demotes.lock().push(bound.map(|bound| bound.up_to));
        None
    }
}

/// The room of one planner's device, as execution admits by it.
struct On(Arc<ResidencyPlanner>);

impl Room for On {
    fn device(&self, _pid: ProcessId) -> usize {
        0
    }
    fn fits(&self, pid: ProcessId, holders: &[ProcessId]) -> bool {
        self.0.admits(pid, holders)
    }
    fn full(&self, holders: &[ProcessId]) -> bool {
        self.0.full(holders)
    }
}

/// A planner over a device of `total` pages, each of `sized` registered
/// with the working set it names, and an execution gate of `seats` that
/// admits by its room.
fn device(
    total: u32,
    sized: &[(ProcessId, u64)],
    seats: usize,
) -> (Arc<ResidencyPlanner>, Arc<OrderedSemaphore>) {
    device_with_host(total, 0, sized, seats)
}

/// [`device`] with `host` free slots below the device.
fn device_with_host(
    total: u32,
    host: u32,
    sized: &[(ProcessId, u64)],
    seats: usize,
) -> (Arc<ResidencyPlanner>, Arc<OrderedSemaphore>) {
    let pool = SizedPool {
        total,
        host,
        demotes: parking_lot::Mutex::new(Vec::new()),
        sizes: parking_lot::Mutex::new(sized.iter().copied().collect()),
    };
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(pool)));
    for (pid, _) in sized {
        planner.register(*pid);
    }
    let gate = OrderedSemaphore::admitting(
        "execution",
        seats,
        true,
        Vec::new(),
        Some(Box::new(On(planner.clone()))),
    );
    (planner, gate)
}

async fn settle() {
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
}

/// Each of `pids` asks `gate` for a seat; what each was granted, as it
/// was, through `granted`.
fn ask(
    gate: &Arc<OrderedSemaphore>,
    pids: &[ProcessId],
    granted: &tokio::sync::mpsc::UnboundedSender<(ProcessId, OrderedPermit)>,
) {
    for &pid in pids {
        let (gate, granted) = (gate.clone(), granted.clone());
        crate::rt::spawn(async move {
            let permit = gate.acquire_owned(pid).await;
            let _ = granted.send((pid, permit));
        });
    }
}

fn drain(
    granted: &mut tokio::sync::mpsc::UnboundedReceiver<(ProcessId, OrderedPermit)>,
) -> Vec<(ProcessId, OrderedPermit)> {
    std::iter::from_fn(|| granted.try_recv().ok()).collect()
}

/// More processes than fit: the admitted are the policy's top keys, as
/// many as their working sets fit the device; the rest wait, and once a
/// seat frees, the next is the top key then, as a re-key left it.
#[tokio::test(flavor = "current_thread")]
async fn the_policy_s_order_decides_who_is_admitted() {
    let _schedule = schedule::TEST_LOCK.write().await;
    let pids: Vec<ProcessId> = (0..6).map(|_| ProcessId::new_v4()).collect();
    // Keys as Continuum ranks its programs: the one holding most cache
    // first. Arrival is the reverse.
    schedule::install(Some(Table::new(
        pids.iter()
            .enumerate()
            .map(|(i, pid)| (*pid, i as i64))
            .collect(),
        None,
    )));
    // Three sets of 3 pages fit a device of 10; a fourth does not.
    let sized: Vec<(ProcessId, u64)> = pids.iter().map(|pid| (*pid, 3)).collect();
    let (_planner, gate) = device(10, &sized, 0);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    ask(&gate, &pids, &tx);
    settle().await;
    gate.add_permits(16);
    settle().await;
    let mut running = drain(&mut rx);
    let admitted: HashSet<ProcessId> = running.iter().map(|(pid, _)| *pid).collect();
    assert_eq!(
        admitted,
        HashSet::from([pids[5], pids[4], pids[3]]),
        "the top three keys, as many as fit, with seats to spare"
    );
    // The lowest key is ranked first now.
    schedule::install(Some(Table::new(
        pids.iter()
            .enumerate()
            .map(|(i, pid)| (*pid, if i == 0 { 9 } else { i as i64 }))
            .collect(),
        None,
    )));
    gate.regrant();
    settle().await;
    assert!(drain(&mut rx).is_empty(), "a re-key makes no room");
    let (done, permit) = running.remove(0);
    drop(permit);
    settle().await;
    let next = drain(&mut rx);
    assert_eq!(
        next.iter().map(|(pid, _)| *pid).collect::<Vec<_>>(),
        vec![pids[0]],
        "{done} left: the re-keyed one is next, not the next by the old order"
    );
    drop(next);
    drop(running);
    schedule::install(None);
}

/// Over-demand waits, never refused: a process that does not fit beside
/// the running waits until one leaves, or waits on its client, and is
/// admitted then. With none running, one is admitted whatever it asks.
#[tokio::test(flavor = "current_thread")]
async fn over_demand_is_admitted_later_never_refused() {
    let _schedule = schedule::TEST_LOCK.write().await;
    schedule::install(None);
    let (first, second, third, huge) = (
        ProcessId::new_v4(),
        ProcessId::new_v4(),
        ProcessId::new_v4(),
        ProcessId::new_v4(),
    );
    let (planner, gate) = device(4, &[(first, 3), (second, 3), (third, 3), (huge, 9)], 8);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    ask(&gate, &[first], &tx);
    settle().await;
    let mut running = drain(&mut rx);
    assert_eq!(running.len(), 1, "the first is admitted");
    ask(&gate, &[second], &tx);
    settle().await;
    assert!(drain(&mut rx).is_empty(), "the second waits: 6 pages of 4");
    // Gone to wait on its client, the first asks for no room.
    planner.set_idle(first, true);
    gate.regrant();
    settle().await;
    running.extend(drain(&mut rx));
    assert_eq!(
        running.len(),
        2,
        "the second is admitted beside an idle one"
    );
    ask(&gate, &[third], &tx);
    settle().await;
    assert!(drain(&mut rx).is_empty(), "the third waits");
    // The second leaves.
    running.retain(|(pid, _)| *pid != second);
    settle().await;
    running.extend(drain(&mut rx));
    assert!(
        running.iter().any(|(pid, _)| *pid == third),
        "the third is admitted once the second left"
    );
    // A set past the whole device, with none running, runs.
    running.clear();
    planner.unregister(first);
    settle().await;
    ask(&gate, &[huge], &tx);
    settle().await;
    let alone = drain(&mut rx);
    assert_eq!(
        alone.iter().map(|(pid, _)| *pid).collect::<Vec<_>>(),
        vec![huge],
        "alone, it runs, whatever it asks"
    );
}

/// Strict priority at 4 seats: the seats are taken; six lower processes
/// arrive and queue, past prewarm and bind or at them, then a higher one.
/// As seats free, the higher one is seated first, passing every lower one
/// that went through the earlier gates before it came, and with each
/// running alike long, all six lower finish after it. (Bind holds a cohort
/// more than execution's seats, its staged permits being open while every
/// seat is taken.)
#[tokio::test(flavor = "current_thread")]
async fn at_four_seats_all_six_lower_finish_after_the_higher_one() {
    let _schedule = schedule::TEST_LOCK.write().await;
    const SEATS: usize = 4;
    let filling: Vec<ProcessId> = (0..SEATS).map(|_| ProcessId::new_v4()).collect();
    let lower: Vec<ProcessId> = (0..6).map(|_| ProcessId::new_v4()).collect();
    let high = ProcessId::new_v4();
    let mut keys: HashMap<ProcessId, i64> = lower.iter().map(|pid| (*pid, 1)).collect();
    keys.insert(high, 9);
    schedule::install(Some(Table::new(keys, None)));
    let prewarm = OrderedSemaphore::new("prewarm", SEATS, true);
    let bind = OrderedSemaphore::after("bind", 2 * SEATS, false, vec![prewarm.clone()]);
    let execution = OrderedSemaphore::after("execution", SEATS, true, vec![bind.clone()]);
    let mut seated = Vec::new();
    for &pid in &filling {
        let bound = bind.clone().acquire_owned(pid).await;
        let seat = execution.clone().acquire_owned(pid).await;
        seated.push((bound, seat));
    }
    let (tx, mut finished) = tokio::sync::mpsc::unbounded_channel();
    for &pid in lower.iter().chain([&high]) {
        let (prewarm, bind, execution, tx) =
            (prewarm.clone(), bind.clone(), execution.clone(), tx.clone());
        crate::rt::spawn(async move {
            let warm = prewarm.acquire_owned(pid).await;
            let bound = bind.acquire_owned(pid).await;
            drop(warm);
            // Setting up: registering its programs, its pipelines.
            settle().await;
            let seat = execution.acquire_owned(pid).await;
            for _ in 0..20 {
                crate::rt::yield_now().await;
            }
            let _ = tx.send(pid);
            drop(bound);
            drop(seat);
        });
        settle().await;
    }
    drop(tx);
    // The seats free, one by one.
    for seat in seated {
        drop(seat);
        settle().await;
    }
    let mut order = Vec::new();
    while let Some(pid) = finished.recv().await {
        order.push(pid);
    }
    assert_eq!(order.len(), 7);
    assert_eq!(order[0], high, "all six lower finish after the higher one");
    schedule::install(None);
}

/// Saturated, what a worker reports and the gateway admits by, is this
/// admission's own rule: a device whose admitted processes all wait on
/// their clients (their sets idle cache, room to reclaim) is not; one whose
/// running sets fill it is, until one of them leaves.
#[tokio::test(flavor = "current_thread")]
async fn saturated_is_the_admission_rule_idle_cache_is_room() {
    let _schedule = schedule::TEST_LOCK.write().await;
    schedule::install(None);
    let (a, b) = (ProcessId::new_v4(), ProcessId::new_v4());
    let (planner, gate) = device(6, &[(a, 3), (b, 3)], 8);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    ask(&gate, &[a, b], &tx);
    settle().await;
    let mut running = drain(&mut rx);
    assert_eq!(running.len(), 2);
    assert!(gate.saturated(), "two running sets of 3 fill a device of 6");
    for pid in [a, b] {
        planner.set_idle(pid, true);
    }
    assert!(!gate.saturated(), "sets of processes waiting on clients are room");
    planner.set_idle(a, false);
    planner.set_idle(b, false);
    assert!(gate.saturated());
    running.pop();
    settle().await;
    assert!(!gate.saturated(), "one left: a next one fits");
}

/// A stated seat count stays a cap: room is weighed only past it. Four
/// stated seats, seven processes whose working sets all fit: four hold
/// execution, the higher key among them, and the three others wait for a
/// seat however much room stands free.
#[tokio::test(flavor = "current_thread")]
async fn stated_seats_cap_admission_whatever_room_stands() {
    let _schedule = schedule::TEST_LOCK.write().await;
    let lower: Vec<ProcessId> = (0..6).map(|_| ProcessId::new_v4()).collect();
    let high = ProcessId::new_v4();
    let mut keys: HashMap<ProcessId, i64> = lower.iter().map(|pid| (*pid, 1)).collect();
    keys.insert(high, 9);
    schedule::install(Some(Table::new(keys, None)));
    let sized: Vec<(ProcessId, u64)> = lower.iter().chain([&high]).map(|pid| (*pid, 1)).collect();
    let (_planner, gate) = device(1_000, &sized, 0);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    ask(&gate, &lower, &tx);
    ask(&gate, &[high], &tx);
    settle().await;
    gate.add_permits(4);
    settle().await;
    let mut running = drain(&mut rx);
    assert_eq!(running.len(), 4, "four stated seats, room for all seven");
    assert!(running.iter().any(|(pid, _)| *pid == high), "the higher key holds one");
    running.pop();
    settle().await;
    assert_eq!(drain(&mut rx).len(), 1, "a freed seat lets one more in");
    drop(running);
    schedule::install(None);
}

/// Admission counts the tiers: a waiter whose set does not fit the device
/// beside the running is admitted where what it lacks there the planner may
/// move below for it (sets keyed below it, as far as the host has room);
/// not where only higher-keyed sets could move; never past the device and
/// the host together.
#[tokio::test(flavor = "current_thread")]
async fn admission_counts_the_room_below_the_device_under_the_key_bound() {
    let _schedule = schedule::TEST_LOCK.write().await;
    let (a, b, c) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    let sized = [(a, 3), (b, 3), (c, 3)];
    let case = |keys: [i64; 3], host: u32| {
        schedule::install(Some(Table::new(
            HashMap::from([(a, keys[0]), (b, keys[1]), (c, keys[2])]),
            None,
        )));
        let (planner, _gate) = device_with_host(6, host, &sized, 8);
        planner.admits(c, &[a, b])
    };
    assert!(case([1, 1, 5], 8), "lower-keyed sets can move to the host for it");
    assert!(!case([9, 9, 5], 8), "only higher-keyed sets could move: it waits");
    assert!(!case([1, 1, 5], 2), "never past the device and the host together");
    schedule::install(None);
}

/// The liveness fallback of the planner's demote follows the bound: while a
/// running request will free room, the head's demand demotes only within
/// its retention key; set aside (lowest key first, moved down, never
/// dropped) only where nothing running or admitted can free room and the
/// head's own row has landed.
#[tokio::test(flavor = "current_thread")]
async fn the_demote_fallback_sets_the_bound_aside_only_when_nothing_frees_room() {
    use crate::policy::retention::{Table, install};
    let _schedule = schedule::TEST_LOCK.write().await;
    let (head, filler) = (ProcessId::new_v4(), ProcessId::new_v4());
    let pool = Arc::new(SizedPool {
        total: 4,
        host: 8,
        demotes: parking_lot::Mutex::new(Vec::new()),
        sizes: parking_lot::Mutex::new(HashMap::new()),
    });
    let planner = Arc::new(ResidencyPlanner::new(pool.clone()));
    for pid in [head, filler] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    install(Some(Table::new(HashMap::from([(head, vec![(None, 15)])]))));
    planner.with_inner(|inner| {
        inner.enqueue(
            head,
            super::wait::WaitKind::Allocation {
                demand: super::Demand {
                    kv_pages: 100,
                    rs: RsUnits::default(),
                },
                notify: Arc::new(tokio::sync::Notify::new()),
                outcome: None,
                yielded: false,
            },
        );
    });
    planner.demote_idle(8);
    assert_eq!(*pool.demotes.lock(), vec![Some(15)], "the filler will run: bounded");
    pool.demotes.lock().clear();
    planner.with_inner(|inner| inner.procs.get_mut(&filler).unwrap().idle = true);
    planner.demote_idle(8);
    assert_eq!(
        *pool.demotes.lock(),
        vec![Some(15), None],
        "nothing frees room: the bound is set aside, after the bounded ask"
    );
    install(None);
}
