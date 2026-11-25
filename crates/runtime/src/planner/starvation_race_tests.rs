use super::*;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::proc::{Proc, Residency};
use super::reclaim::Reclaim;
use super::wait::{EntryKey, WaitKind};

use crate::planner::grant::Demand;
use crate::store::kv::page_table::PhysicalKvPageId;
use std::sync::atomic::AtomicU32;

fn allocation(demand: Demand) -> WaitKind {
    WaitKind::Allocation {
        demand,
        notify: Arc::new(Notify::new()),
        outcome: None,
        yielded: false,
        rise: false,
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
    /// A shared pool's segment walk the planner's grow asks, as
    /// `RegistryPool`'s does: the waiter registered short, then promised a
    /// grow only where the walk's next step maps a segment.
    grow: Option<crate::store::shared::SharedGrow>,
    /// The room signal the planner set (`PoolPort::set_room_signal`).
    signal: parking_lot::Mutex<Option<crate::store::slots::RoomSignal>>,
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
            grow: None,
            signal: parking_lot::Mutex::new(None),
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
    fn set_room_signal(&self, signal: crate::store::slots::RoomSignal) {
        *self.signal.lock() = Some(signal);
    }
    fn grow(&self, need: Demand) -> bool {
        let Some(grow) = &self.grow else {
            return false;
        };
        let short = if need.kv_pages > 0 {
            crate::store::shared::tests::Short::Kv
        } else {
            crate::store::shared::tests::Short::State
        };
        crate::store::shared::tests::wait_short(&mut grow.shared().lock().unwrap(), short);
        grow.promise_grow()
    }
    fn chunk_stats(&self) -> ChunkStats {
        ChunkStats {
            pages: (self.free.lock().len() as u32, self.total),
            rs: self
                .with_rs(|rs| (rs.available(), rs.capacity()))
                .unwrap_or_default(),
        }
    }
    fn host_stats(&self) -> (u32, u32) {
        (self.host, self.host_total)
    }
    fn disk_stats(&self) -> (u32, u32) {
        (0, 0)
    }
    fn rs_host_stats(&self) -> (RsUnits, RsUnits) {
        Default::default()
    }
    fn reclaim_idle(&self, _: super::pool::IdleShort) -> super::pool::IdleReclaim {
        super::pool::IdleReclaim::default()
    }
    fn reclaimable_idle(&self) -> u32 {
        0
    }
    fn idle_below_device(&self) -> (u32, u32) {
        (0, 0)
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
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
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
    fn reclaim_sharing(&self, working_sets: &[crate::store::rs::RsWorkingSetId]) -> usize {
        self.with_rs(|rs| rs.reclaim_sharing(working_sets))
            .unwrap_or(0)
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
    fn prepare_demote_now(&self) -> Option<crate::store::kv::KvResidencyTxn> {
        None
    }
    fn drop_disallowed(&self) -> usize {
        0
    }
    fn prepare_demote(&self, _room: usize) -> Option<crate::store::kv::KvResidencyTxn> {
        None
    }
}

#[test]
fn a_younger_process_is_not_admitted_while_an_older_one_is_preempted() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    use futures::FutureExt;
    let planner = Arc::new(ResidencyPlanner::new(
        Arc::new(RacePool::new(8)) as Arc<dyn PoolPort>
    ));
    let [evicted, killed, younger] = [(); 3].map(|_| ProcessId::new_v4());
    for pid in [evicted, killed, younger] {
        planner.register(pid);
    }
    planner.with_inner(|inner| inner.procs.get_mut(&evicted).unwrap().state = Residency::Evicted);
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
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
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
            let rs = RsUnits {
                slots,
                pages: 0,
                ..RsUnits::default()
            };
            inner.enqueue(pid, allocation(Demand { kv_pages, rs }));
        }
    });
    let victims = planner.victim_set().expect("the head is short");
    assert_eq!(
        victims.deficit,
        Reclaim {
            pages: 0,
            rs: RsUnits {
                slots: 1,
                pages: 0,
                ..RsUnits::default()
            }
        }
    );
    assert_eq!(victims.members.len(), 1);
    assert_eq!(victims.members[0].pid, younger);
}

#[test]
fn a_wedged_head_short_of_state_alone_still_reclaims_by_restart() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    // The head asks for a state slot and no pages: the pages it has
    // do not cover it, so a stalled fleet with nothing to evict reclaims
    // the youngest ask rather than waiting on a serve that cannot come.
    let model = crate::store::registry::register_model(16, &[64], &[2]);
    let stores = crate::store::registry::get(model, 0);
    let held = stores
        .rs
        .lock()
        .unwrap()
        .reserve(RsUnits {
            slots: 2,
            pages: 0,
            ..RsUnits::default()
        })
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
            let rs = RsUnits {
                slots,
                pages: 0,
                ..RsUnits::default()
            };
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

/// A dry pool of eight pages, an older lane's growth and a younger prompt
/// waiting on it, neither holding a page a kill could free: the off run's
/// moment, when a third process had just finished.
fn dry_pool_two_asks(planner: &ResidencyPlanner) -> Vec<EntryKey> {
    planner.with_inner(|inner| {
        [(ProcessId::new_v4(), 1), (ProcessId::new_v4(), 3)]
            .into_iter()
            .map(|(pid, kv_pages)| {
                let seq = inner.next_seq;
                inner.next_seq += 1;
                let mut proc = Proc::new(seq);
                proc.admitted = true;
                inner.procs.insert(pid, proc);
                inner.enqueue(
                    pid,
                    allocation(Demand {
                        kv_pages,
                        rs: RsUnits::default(),
                    }),
                )
            })
            .collect()
    })
}

fn unfailed(planner: &ResidencyPlanner, keys: &[EntryKey]) -> usize {
    planner.with_inner(|inner| {
        keys.iter()
            .filter(|key| {
                matches!(
                    inner.queue[key].kind,
                    WaitKind::Allocation { outcome: None, .. }
                )
            })
            .count()
    })
}

type HeldResidency = Arc<std::sync::Mutex<crate::inferlet::process::residency::ProcessResidency>>;

/// A process about to leave: registered and admitted, its residency naming
/// one KV working set that holds two device pages. With `hold`, the set
/// holds the process's [`crate::planner::Departure`] as its context gives
/// it one.
struct Leaving {
    pid: ProcessId,
    ws: crate::store::kv::working_set::KvWorkingSet,
    departure: Arc<crate::planner::Departure>,
    residency: HeldResidency,
}

fn leaving(planner: &Arc<ResidencyPlanner>, model: usize, hold: bool) -> Leaving {
    use crate::inferlet::process::residency::{ProcessResidency, register_residency};
    let stores = crate::store::registry::get(model, 0);
    let id = crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
        let id = kv.create_working_set();
        kv.reserve(id, 2).unwrap();
        let mut granted = kv.reserve_device_pages(2).unwrap();
        kv.ensure_backed_reserved(id, 2, &mut granted).unwrap();
        id
    });
    let ws = crate::store::kv::working_set::KvWorkingSet::new(model, 0, id, 16);
    let pid = ProcessId::new_v4();
    let residency = Arc::new(std::sync::Mutex::new(ProcessResidency::default()));
    residency
        .lock()
        .unwrap()
        .kv_working_sets
        .insert((model, 0, id), ws.suspend_handle());
    register_residency(pid, Arc::downgrade(&residency));
    planner.register(pid);
    planner.note_admitted(pid);
    let departure = planner.departure(pid).expect("a registered process");
    assert!(Arc::ptr_eq(&departure, &planner.departure(pid).unwrap()));
    if hold {
        ws.hold_departure(departure.clone());
    }
    Leaving {
        pid,
        ws,
        departure,
        residency,
    }
}

/// `Process::terminate`'s planner steps: the planner forgets it, it gives
/// its own hold up, its residency goes. Its working set, still in the
/// resource table the teardown holds, is what is left.
fn terminate(
    planner: &Arc<ResidencyPlanner>,
    leaving: Leaving,
) -> (ProcessId, crate::store::kv::working_set::KvWorkingSet) {
    planner.unregister(leaving.pid);
    drop(leaving.departure);
    crate::inferlet::process::residency::unregister_residency(leaving.pid);
    drop(leaving.residency);
    assert!(planner.departure(leaving.pid).is_none(), "it has left");
    (leaving.pid, leaving.ws)
}

/// A dry pool of eight pages at `model`'s locus, where the departing
/// processes' working sets live.
fn departure_pool(model: usize) -> Arc<dyn PoolPort> {
    Arc::new(RacePool {
        rs_model: Some(model),
        ..RacePool::new(8)
    })
}

fn on_its_way_back(planner: &ResidencyPlanner, pid: ProcessId) -> Option<u32> {
    planner.with_inner(|inner| inner.evicting.get(&pid).map(|mark| mark.pages))
}

fn youngest_starved(planner: &ResidencyPlanner, keys: &[EntryKey]) {
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
}

#[test]
fn a_departure_s_pages_are_waited_for_before_an_ask_holding_nothing_fails() {
    // The finished process has left the planner, but its working set still
    // holds its pages until the teardown drops its resources: nothing fails
    // while it does. Once the set is released, a stall that remains is a
    // wedge, and the youngest ask fails as before.
    let model = crate::store::registry::register_model(16, &[64], &[2]);
    let planner = Arc::new(ResidencyPlanner::new(departure_pool(model)));
    let keys = dry_pool_two_asks(&planner);
    let (pid, ws) = terminate(&planner, leaving(&planner, model, true));
    assert_eq!(on_its_way_back(&planner, pid), Some(2));
    assert!(!planner.is_wedged(), "its pages are on their way back");
    planner.check_starvation(StarveCause::NoSwapRoom);
    assert_eq!(
        unfailed(&planner, &keys),
        2,
        "an ask fails while a departure's pages come back"
    );
    // The release itself pokes the planner, which finds the wedge: no
    // later event is needed to fail the youngest ask.
    drop(ws);
    assert_eq!(on_its_way_back(&planner, pid), None, "its set is released");
    youngest_starved(&planner, &keys);
}

#[test]
fn a_departure_held_behind_a_long_fire_is_waited_for_however_long() {
    // The departed process's working set is released while a fire still
    // leases it (a launch can take minutes): its pages come back only when
    // the lease drops. No clock is involved: the planner waits for the
    // release itself, so the empty asker does not fail before it.
    let model = crate::store::registry::register_model(16, &[64], &[2]);
    crate::planner::init_planner(model, 0, ResidencyPlanner::new(departure_pool(model)));
    let planner = crate::planner::planner_for(model, 0).expect("the test model's planner");
    let keys = dry_pool_two_asks(&planner);
    let gone = leaving(&planner, model, true);
    let lease = gone.ws.fire_lease().expect("a fire leases the working set");
    let (pid, ws) = terminate(&planner, gone);
    ws.release();
    drop(ws);
    for _ in 0..100 {
        planner.check_starvation(StarveCause::NoSwapRoom);
    }
    assert_eq!(
        unfailed(&planner, &keys),
        2,
        "an ask fails while the departure's release waits on a fire"
    );
    assert_eq!(on_its_way_back(&planner, pid), Some(2));
    drop(lease);
    assert_eq!(
        on_its_way_back(&planner, pid),
        None,
        "the fire's end releases the working set"
    );
    planner.check_starvation(StarveCause::NoSwapRoom);
    assert_eq!(
        unfailed(&planner, &keys),
        1,
        "a stall that outlasts the release is a wedge"
    );
}

#[test]
fn a_departure_never_waits_on_its_pages_nor_outlives_its_holds() {
    let model = crate::store::registry::register_model(16, &[64], &[2]);
    let planner = Arc::new(ResidencyPlanner::new(departure_pool(model)));

    // Its exit is not held up by a set a fire leases: its asks and its
    // place go at once, and the completion is credited.
    let gone = leaving(&planner, model, true);
    let ask = planner.with_inner(|inner| {
        inner.enqueue(
            gone.pid,
            allocation(Demand {
                kv_pages: 1,
                rs: RsUnits::default(),
            }),
        )
    });
    let lease = gone.ws.fire_lease().unwrap();
    let credit = planner.with_inner(|inner| inner.completion_credit);
    let (pid, ws) = terminate(&planner, gone);
    planner.with_inner(|inner| {
        assert!(!inner.procs.contains_key(&pid));
        assert!(!inner.queue.contains_key(&ask));
        assert_eq!(inner.completion_credit, credit + 1);
    });
    assert_eq!(on_its_way_back(&planner, pid), Some(2));
    drop((ws, lease));
    assert_eq!(on_its_way_back(&planner, pid), None);

    // A set released before its process leaves leaves nothing on the way.
    let gone = leaving(&planner, model, true);
    gone.ws.release();
    let (pid, _ws) = terminate(&planner, gone);
    assert_eq!(on_its_way_back(&planner, pid), None);

    // A suspend in flight as it leaves: the suspend's end does not settle
    // the departure, its set's release does.
    let gone = leaving(&planner, model, true);
    let wake = planner.with_inner(|inner| {
        inner.begin_eviction(
            gone.pid,
            Reclaim {
                pages: 2,
                rs: RsUnits::default(),
            },
        )
    });
    drop(wake);
    let (pid, ws) = terminate(&planner, gone);
    planner.report_evicted(pid, 2, RsUnits::default());
    assert_eq!(on_its_way_back(&planner, pid), Some(2));
    planner.eviction_failed(pid);
    assert_eq!(on_its_way_back(&planner, pid), Some(2));
    drop(ws);
    assert_eq!(on_its_way_back(&planner, pid), None);
}

#[test]
fn a_state_set_a_handle_takes_gives_its_owner_s_hold_back() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    // A process's sets go to a handle (`Held::park`): its kv set gives the
    // departed owner's hold back, and so must its state set. Kept, the
    // owner's pages and slot stood "on their way back" for as long as the
    // handle lived, and a head short of a slot planned no eviction again
    // (t_evict hung on Qwen3.5 at 176 MiB).
    let model = crate::store::registry::register_model(16, &[64], &[2]);
    let planner = Arc::new(ResidencyPlanner::new(departure_pool(model)));
    let gone = leaving(&planner, model, true);
    let geom = crate::store::rs::RsGeometry {
        state_size: 64,
        buffer_page_tokens: 4,
    };
    let id = crate::store::registry::get(model, 0)
        .rs
        .lock()
        .unwrap()
        .create_working_set(geom);
    let rs = crate::store::rs::working_set::RsWorkingSet::new(model, 0, id, geom);
    rs.hold_departure(gone.departure.clone());
    let (pid, ws) = terminate(&planner, gone);
    assert_eq!(on_its_way_back(&planner, pid), Some(2));
    ws.leave_owner();
    assert_eq!(
        on_its_way_back(&planner, pid),
        Some(2),
        "the state set still holds the owner"
    );
    rs.leave_owner();
    assert_eq!(
        on_its_way_back(&planner, pid),
        None,
        "both sets gave it back"
    );
    drop((ws, rs));
}

#[test]
fn pages_no_hold_watches_are_not_waited_for() {
    // A set its context never saw holds no departure: no release would
    // clear a mark for it, so none is made, and the stall is a wedge.
    let model = crate::store::registry::register_model(16, &[64], &[2]);
    let planner = Arc::new(ResidencyPlanner::new(departure_pool(model)));
    let keys = dry_pool_two_asks(&planner);
    let (pid, _ws) = terminate(&planner, leaving(&planner, model, false));
    assert_eq!(on_its_way_back(&planner, pid), None);
    youngest_starved(&planner, &keys);
}

#[tokio::test(flavor = "current_thread")]
async fn a_host_swap_full_victim_is_parked_until_host_room_returns() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(RacePool::new(4));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone()));

    let head = ProcessId::new_v4();
    let victim = ProcessId::new_v4();
    for pid in [head, victim] {
        planner.register(pid);
        planner.note_admitted(pid);
    }

    // A newcomer whose first ask does not fit now is refused at
    // admission; asked again as an admitted lane's growth, it waits.
    refused_at_admission_then_seated(&planner, head, 1);
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
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let planner = Arc::new(ResidencyPlanner::new(
        Arc::new(RacePool::new(4)) as Arc<dyn PoolPort>
    ));
    let agent = ProcessId::new_v4();
    planner.register(agent);
    planner.note_admitted(agent);
    planner.set_idle(agent, true);
    planner.with_inner(|inner| inner.procs.get_mut(&agent).unwrap().state = Residency::Evicting);
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
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
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

    planner.with_inner(|inner| inner.procs.get_mut(&agent).unwrap().state = Residency::Evicting);
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
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
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
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
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
                rs: RsUnits {
                    slots: 1,
                    pages: 0,
                    ..RsUnits::default()
                },
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

#[test]
fn a_holder_goes_before_every_process_and_oldest_first() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(8))));
    let process = ProcessId::new_v4();
    planner.register(process);
    let (older, newer) = (ProcessId::new_v4(), ProcessId::new_v4());
    planner.register_holder(older, None);
    planner.register_holder(newer, None);
    planner.with_inner(|inner| {
        let rank = |pid: &ProcessId| inner.procs[pid].rank();
        assert!(rank(&newer) > rank(&process), "a process stays");
        assert!(rank(&older) > rank(&newer), "the larger rank goes first");
    });
}

#[test]
fn a_claimed_holder_is_no_one_s_victim_and_owes_nothing_of_its_own() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(8))));
    let holder = ProcessId::new_v4();
    planner.register_holder(holder, None);
    planner.claim_holder(holder);
    planner.with_inner(|inner| {
        assert!(!inner.procs[&holder].idle);
        inner.procs.get_mut(&holder).unwrap().state = Residency::Evicting;
    });
    assert!(planner.holder_moving(holder), "the claim waits for this");
    planner.report_evicted(holder, 3, Default::default());
    assert!(!planner.holder_moving(holder));
    planner.with_inner(|inner| {
        assert!(
            !inner.queue.values().any(|waiter| waiter.pid == holder),
            "a holder restores nothing: the process that takes the set does"
        );
    });
    planner.unclaim_holder(holder);
    planner.with_inner(|inner| assert!(inner.procs[&holder].idle));
}

#[test]
fn a_process_that_took_sets_off_the_device_owes_them_as_one_ask() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(8))));
    let process = ProcessId::new_v4();
    planner.register(process);
    let demand = Demand {
        kv_pages: 3,
        rs: Default::default(),
    };
    planner.owe(process, demand);
    assert!(!planner.is_resident(process), "it waits before it runs");
    planner.with_inner(|inner| {
        let asks = inner.queue.values().filter(|waiter| waiter.pid == process);
        assert_eq!(asks.count(), 1);
    });
}

#[test]
fn a_handle_at_rest_off_the_device_is_no_contention() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(8))));
    planner.register_holder(ProcessId::new_v4(), Some(Demand::default()));
    planner.with_inner(|inner| assert_eq!(inner.nonresident_count(), 0));
}

fn park_asks(planner: &ResidencyPlanner, pid: ProcessId, asks: &[u32]) -> Vec<EntryKey> {
    planner.with_inner(|inner| {
        asks.iter()
            .map(|&kv_pages| {
                inner.enqueue(
                    pid,
                    allocation(Demand {
                        kv_pages,
                        rs: RsUnits::default(),
                    }),
                )
            })
            .collect()
    })
}

/// text-completion-bench, 96 MiB, a 5,000-token lane beside seven short
/// ones: once nothing of the process ran, its sixth short lane parked for a
/// state slot its own five held sets and scattered pages left no chunk for.
/// It held all the pool held, so the restart freed only its own and the
/// replay stalled in the same place, hundreds of times a minute. A stalled
/// victim that alone holds anything fails instead, saying so.
#[test]
fn a_stalled_process_that_alone_holds_the_pool_fails_instead_of_restarting() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let model = crate::store::registry::register_model(16, &[64], &[2]);
    let planner = Arc::new(ResidencyPlanner::new(departure_pool(model)));
    let process = leaving(&planner, model, false);
    let pid = process.pid;
    crate::inferlet::process::declare_restartable(pid);
    let keys = park_asks(&planner, pid, &[4]);
    assert!(planner.sole_holder(pid));
    planner.check_starvation(StarveCause::NoSwapRoom);
    planner.with_inner(|inner| {
        let WaitKind::Allocation {
            outcome: Some(Err(error)),
            ..
        } = &inner.queue[&keys[0]].kind
        else {
            panic!("the parked ask fails");
        };
        assert!(
            matches!(
                error,
                PlannerError::Starved {
                    cause: StarveCause::SoleHolder,
                    ..
                }
            ),
            "{error}"
        );
        assert!(
            error.to_string().contains("this process alone holds"),
            "{error}"
        );
    });
    assert_eq!(
        planner.stats.starvation_restarts.load(Ordering::Relaxed),
        0,
        "no replay into the same wall"
    );
    drop(terminate(&planner, process));
}

/// An eviction whose copy to the host fails is never tried again: the
/// process is failed by name (a kill in flight, counted as an eviction
/// fault, not a rollback) and leaves the victims, so the next round moves
/// another. An eviction that found nothing to move faults nobody.
#[test]
fn an_eviction_whose_copy_fails_faults_its_process_and_the_next_is_moved() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    // A full pool: the head's page is a deficit only an eviction funds.
    let pool = Arc::new(RacePool::new(64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, first, second) = (
        ProcessId::new_v4(),
        ProcessId::new_v4(),
        ProcessId::new_v4(),
    );
    for pid in [head, first, second] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        inner.enqueue(
            head,
            allocation(Demand {
                kv_pages: 1,
                rs: RsUnits::default(),
            }),
        );
    });
    let members = |planner: &Arc<ResidencyPlanner>| -> HashSet<ProcessId> {
        planner
            .victim_set()
            .expect("the head is short")
            .members
            .iter()
            .map(|victim| victim.pid)
            .collect()
    };
    assert_eq!(members(&planner), HashSet::from([first, second]));

    // Negative control: an eviction that found nothing to move keeps its
    // process out of this round but fails nobody.
    planner.eviction_found_nothing(second);
    assert_eq!(planner.stats.eviction_faults.load(Ordering::Relaxed), 0);
    assert!(!planner.with_inner(|inner| inner.killing.contains(&second)));
    planner.clear_host_swap_blocks();
    assert_eq!(members(&planner), HashSet::from([first, second]));

    // The copy fails: the process is failed once, by name, and never
    // picked again; the other is the round's victim.
    let rollbacks = planner.stats.eviction_rollbacks.load(Ordering::Relaxed);
    planner.eviction_copy_failed(
        first,
        "role-slots to HostPinned copy: a slot copy names shape 131072",
    );
    assert_eq!(planner.stats.eviction_faults.load(Ordering::Relaxed), 1);
    assert_eq!(
        planner.stats.eviction_rollbacks.load(Ordering::Relaxed),
        rollbacks,
        "a fault is no rollback"
    );
    assert!(planner.with_inner(|inner| inner.killing.contains(&first)));
    for _ in 0..3 {
        assert_eq!(
            members(&planner),
            HashSet::from([second]),
            "the failed process is not tried again"
        );
    }
    assert_eq!(planner.stats.eviction_faults.load(Ordering::Relaxed), 1);
}

/// A restore that fails at a step of its own, its copy back or a prepare,
/// is never queued again: the process is failed by name, once, counted as
/// a restore fault. A restore queued again because more of its pages were
/// swapped out since its quote is no failure and no fault.
#[test]
fn a_restore_that_fails_a_step_is_failed_once_not_queued_again() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let pool = Arc::new(RacePool::new(64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (copying, preparing, moved) = (
        ProcessId::new_v4(),
        ProcessId::new_v4(),
        ProcessId::new_v4(),
    );
    for pid in [copying, preparing, moved] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    // A fire in flight elsewhere.
    let decoding = ProcessId::new_v4();
    planner.register(decoding);
    planner.note_admitted(decoding);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&decoding).unwrap();
        proc.fires = 1;
        proc.fired_at = crate::rt::Instant::now();
    });
    let restoring = |planner: &Arc<ResidencyPlanner>, pid: ProcessId| {
        planner.with_inner(|inner| {
            inner.procs.get_mut(&pid).expect("registered").state = Residency::Restoring;
        });
    };
    let queued = |planner: &Arc<ResidencyPlanner>, pid: ProcessId| {
        planner.with_inner(|inner| {
            inner
                .queue
                .values()
                .any(|waiter| waiter.pid == pid && matches!(waiter.kind, WaitKind::Restore { .. }))
        })
    };
    let faults =
        |planner: &Arc<ResidencyPlanner>| planner.stats.restore_faults.load(Ordering::Relaxed);

    // Negative control: more of its pages were swapped out since its quote;
    // queued again, to be quoted anew.
    // (Its parked state moving down a tier holds it queued, so the pass
    // does not serve it at once here, with no executor to run it.)
    restoring(&planner, moved);
    planner.with_inner(|inner| {
        inner.parked_moving.insert(moved);
    });
    planner.requeue_restore(moved);
    assert!(queued(&planner, moved), "queued again, quoted anew");
    assert_eq!(faults(&planner), 0);

    // Its copy back fails: failed once by name, never queued again.
    restoring(&planner, copying);
    planner.restore_faulted(
        copying,
        "its state could not be copied back to the device: role-slots from HostPinned copy",
    );
    assert!(
        !queued(&planner, copying),
        "a failed copy is not tried again"
    );
    assert_eq!(faults(&planner), 1);

    // A prepare fails (here its state's): the same, at once, no retry.
    restoring(&planner, preparing);
    planner.restore_faulted(preparing, "rs prepare: no device rows");
    assert!(
        !queued(&planner, preparing),
        "a failed prepare is not tried again"
    );
    assert_eq!(faults(&planner), 2);
    for pid in [copying, preparing] {
        assert_eq!(
            planner.with_inner(|inner| inner.procs[&pid].state),
            Residency::Evicted
        );
    }
}

/// A shared pool's chunks for a bind short of room: free, held by idle
/// cache (dropped by `reclaim_idle`), or held by a resident's guest state.
/// Chunk ids stand for kv pages one a chunk.
struct BindRoomPool {
    free: parking_lot::Mutex<Vec<u32>>,
    cache: parking_lot::Mutex<Vec<u32>>,
    total: u32,
    host: u32,
    /// The registered model whose kv store its victims' pages lie in.
    model: usize,
    /// The process whose state holds the chunks, if one does
    /// ([`PoolPort::chunk_occupants`]): each chunk of the pool, whole.
    occupant: parking_lot::Mutex<Option<ProcessId>>,
    /// The pool's count of room given back (`room_gives`).
    gives: std::sync::atomic::AtomicU64,
    /// The free chunks a restore's own parked guest state retakes first
    /// ([`PoolPort::room_beside_own`]).
    own: parking_lot::Mutex<Vec<u32>>,
    /// Parked state lying on the device: each chunk it holds whole, and
    /// the process it parked with ([`PoolPort::parked_occupants`]).
    parked: parking_lot::Mutex<Vec<(u32, ProcessId)>>,
    /// Host slots held (of `host`), and disk slots free.
    host_used: AtomicU32,
    disk_free: AtomicU32,
    /// The moves of parked state down a tier, in order: `"spill"` (host kv
    /// on down to disk) or `"park"` (role slots to the host).
    moved: parking_lot::Mutex<Vec<&'static str>>,
    /// Scattered free ids a compaction would join into the runs asked
    /// ([`PoolPort::compaction_comes`]), and how often it was asked.
    compactable: std::sync::atomic::AtomicBool,
    compactions: AtomicU32,
    /// Chunks residents' kv pages hold, one resident's whole: a shape the
    /// head's runs are not, emptied (free for any shape) once their
    /// resident moves off the device.
    kv: parking_lot::Mutex<Vec<(u32, ProcessId)>>,
    /// Its free chunks are handed out as device pages (a chunk a page), as
    /// an activation rise's grant takes them; else it hands out none.
    pages: std::sync::atomic::AtomicBool,
    /// Moves of parked state down a tier that begin and then fail (the
    /// engine refused the copy), counted down.
    park_down_fails: AtomicU32,
    /// Disk slots only idle cache holds (`PoolPort::idle_below_device`):
    /// room a suspend makes by dropping it.
    disk_idle: AtomicU32,
    /// The state budget's headroom: the chunks that may be opened, free
    /// ones beyond it standing open but unusable (`None`: no budget).
    headroom: parking_lot::Mutex<Option<u32>>,
    /// The pages a unit of state (an rs slot) stands for (`rs_as_pages`).
    rs_pages: AtomicU32,
    /// The state slots free (`rs_stats`).
    rs_free: AtomicU32,
    /// The store's placement refuses a restore its map stands
    /// (`PoolPort::restore_stands`).
    store_refuses: std::sync::atomic::AtomicBool,
    /// Chunks several processes' slots share (a forked prompt's prefix
    /// pages, a shared guest item): each chunk, the slots it holds, and
    /// each holder's share ([`PoolPort::chunk_occupants`]).
    shared: parking_lot::Mutex<Vec<super::pool::ChunkHeld>>,
    /// Whether a move down of parked state takes the shared slots every
    /// holder of which it moves (the store's sweep of evicted processes'
    /// kv left on the device), or leaves them on the device.
    sweeps_shared: std::sync::atomic::AtomicBool,
    /// Where set, what a reclaim frees waits on a fire's retirement
    /// here (`returning`), as the kv store's pages read by a fire in
    /// flight do: a reclaim frees only what is short past them.
    retiring: parking_lot::Mutex<Option<Vec<u32>>>,
    /// The cache chunks reclaims let go, in all.
    reclaimed: AtomicU32,
}

impl BindRoomPool {
    fn new(total: u32, cache: std::ops::Range<u32>, host: u32) -> Self {
        Self {
            free: parking_lot::Mutex::new(Vec::new()),
            cache: parking_lot::Mutex::new(cache.collect()),
            total,
            host,
            model: 0,
            occupant: parking_lot::Mutex::new(None),
            gives: std::sync::atomic::AtomicU64::new(0),
            own: parking_lot::Mutex::new(Vec::new()),
            parked: parking_lot::Mutex::new(Vec::new()),
            host_used: AtomicU32::new(0),
            disk_free: AtomicU32::new(0),
            moved: parking_lot::Mutex::new(Vec::new()),
            compactable: std::sync::atomic::AtomicBool::new(false),
            compactions: AtomicU32::new(0),
            kv: parking_lot::Mutex::new(Vec::new()),
            pages: std::sync::atomic::AtomicBool::new(false),
            park_down_fails: AtomicU32::new(0),
            disk_idle: AtomicU32::new(0),
            headroom: parking_lot::Mutex::new(None),
            rs_pages: AtomicU32::new(0),
            rs_free: AtomicU32::new(0),
            store_refuses: std::sync::atomic::AtomicBool::new(false),
            shared: parking_lot::Mutex::new(Vec::new()),
            sweeps_shared: std::sync::atomic::AtomicBool::new(true),
            retiring: parking_lot::Mutex::new(None),
            reclaimed: AtomicU32::new(0),
        }
    }

    /// Whether `pieces` disjoint runs of `run` adjacent free ids stand.
    fn runs_free(&self, run: u32, pieces: u32) -> bool {
        let mut ids = self.free.lock().clone();
        ids.sort_unstable();
        let (mut runs, mut stretch) = (0u32, 0u32);
        for (at, &id) in ids.iter().enumerate() {
            stretch = if at > 0 && ids[at - 1] + 1 == id {
                stretch + 1
            } else {
                1
            };
            if stretch == run {
                runs += 1;
                stretch = 0;
            }
        }
        // As the pool's own count: no more runs than the budget opens.
        if let Some(headroom) = *self.headroom.lock() {
            runs = runs.min(headroom / run.max(1));
        }
        runs >= pieces
    }
}

impl PoolPort for BindRoomPool {
    fn prepare_demote_now(&self) -> Option<crate::store::kv::KvResidencyTxn> {
        None
    }
    fn drop_disallowed(&self) -> usize {
        0
    }
    /// Its own retake (`own`, single chunks) takes its chunks first; what
    /// is short is the rest of `need` (the head's need as the planner reads
    /// it, its own retake's single chunks among them), laid in what that
    /// leaves.
    fn restore_placement(
        &self,
        pid: ProcessId,
        need: &super::pool::RunNeed,
        state: RsUnits,
        _kv_pages: u32,
        bind: &[(u32, u32)],
    ) -> super::pool::RestorePlacement {
        let _ = (pid, state, bind);
        let refuses = self.store_refuses.load(Ordering::Relaxed);
        let free = self.free.lock().clone();
        let opened: Vec<u32> = self.own.lock().iter().copied().filter(|c| free.contains(c)).collect();
        let own = opened.len() as u32;
        let rest = super::pool::RunNeed::of(need.entries().iter().map(|&(run, pieces)| {
            (run, if run == 1 { pieces.saturating_sub(own) } else { pieces })
        }));
        let beside = super::pool::ChunkMap {
            free: free.iter().copied().filter(|c| !opened.contains(c)).collect(),
            headroom: self.headroom.lock().map(|room| room.saturating_sub(own)),
            ..super::pool::ChunkMap::default()
        };
        let stands = !refuses && beside.cheapest_need(&rest).is_some_and(|moves| moves.is_empty());
        super::pool::RestorePlacement {
            opened,
            room: None,
            short: if stands { super::pool::RunNeed::default() } else { rest },
            bind_free_runs: None,
        }
    }
    fn chunk_stats(&self) -> ChunkStats {
        let rs = self.rs_free.load(Ordering::Relaxed);
        ChunkStats {
            pages: (self.free.lock().len() as u32, self.total),
            rs: (
                RsUnits {
                    slots: rs,
                    ..RsUnits::default()
                },
                RsUnits {
                    slots: rs.max(8),
                    ..RsUnits::default()
                },
            ),
        }
    }
    fn host_stats(&self) -> (u32, u32) {
        (self.host - self.host_used.load(Ordering::Relaxed), self.host)
    }
    fn disk_stats(&self) -> (u32, u32) {
        let free = self.disk_free.load(Ordering::Relaxed);
        (free, free + self.disk_idle.load(Ordering::Relaxed))
    }
    fn rs_host_stats(&self) -> (RsUnits, RsUnits) {
        Default::default()
    }
    fn parked_occupants(&self, except: ProcessId) -> super::pool::ChunkMap {
        super::pool::ChunkMap {
            held: self
                .parked
                .lock()
                .iter()
                .filter(|&&(_, pid)| pid != except)
                .map(|&(chunk, pid)| super::pool::ChunkHeld {
                    chunk,
                    used: 1,
                    by: vec![(pid, 1)],
                })
                .chain(self.shared.lock().iter().filter_map(|held| {
                    let by: Vec<_> = held.by.iter().copied().filter(|&(pid, _)| pid != except).collect();
                    (!by.is_empty()).then(|| super::pool::ChunkHeld { by, ..held.clone() })
                }))
                .collect(),
            free: self.free.lock().clone(),
            ..Default::default()
        }
    }
    /// One host slot a chunk of parked state: the host full, host kv goes
    /// to disk for it first, a slot a chunk; with no host tier, the parked
    /// state goes to disk itself.
    fn move_parked_down(
        &self,
        planner: &Arc<ResidencyPlanner>,
        pids: &[ProcessId],
        _order: &[ProcessId],
    ) -> bool {
        // Shared slots every holder of which moves down go with them, as
        // the store's sweep takes evicted processes' kv left on the device.
        if self.sweeps_shared.load(Ordering::Relaxed) {
            let swept: Vec<u32> = self
                .shared
                .lock()
                .iter()
                .filter(|held| held.by.iter().all(|(pid, _)| pids.contains(pid)))
                .map(|held| held.chunk)
                .collect();
            if !swept.is_empty() {
                self.shared.lock().retain(|held| !swept.contains(&held.chunk));
                self.free.lock().extend(&swept);
                self.gives.fetch_add(swept.len() as u64, Ordering::Relaxed);
                self.moved.lock().push("sweep");
                let planner = planner.clone();
                crate::rt::spawn(async move { planner.parked_down_done(true) });
                return true;
            }
        }
        let chunks: Vec<u32> = self
            .parked
            .lock()
            .iter()
            .filter(|(_, pid)| pids.contains(pid))
            .map(|&(chunk, _)| chunk)
            .collect();
        // An evicted process's state left on the device (kept with the
        // residents' here, `kv`): it goes down to disk at once.
        let state: Vec<u32> = self
            .kv
            .lock()
            .iter()
            .filter(|(_, pid)| pids.contains(pid))
            .map(|&(chunk, _)| chunk)
            .collect();
        if !state.is_empty() && chunks.is_empty() {
            self.kv.lock().retain(|(_, pid)| !pids.contains(pid));
            self.free.lock().extend(state);
            self.moved.lock().push("state");
            let planner = planner.clone();
            crate::rt::spawn(async move { planner.parked_down_done(true) });
            return true;
        }
        if chunks.is_empty() {
            return false;
        }
        if self
            .park_down_fails
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
        {
            let planner = planner.clone();
            crate::rt::spawn(async move { planner.parked_down_done(false) });
            return true;
        }
        let want = chunks.len() as u32;
        let host_free = self.host - self.host_used.load(Ordering::Relaxed);
        if self.host == 0 && self.disk_free.load(Ordering::Relaxed) >= want {
            // No host tier (unified memory): to disk at once.
            self.disk_free.fetch_sub(want, Ordering::Relaxed);
            self.parked.lock().retain(|(_, pid)| !pids.contains(pid));
            self.free.lock().extend(chunks);
            self.moved.lock().push("disk");
        } else if self.host > 0 && host_free >= want {
            self.host_used.fetch_add(want, Ordering::Relaxed);
            self.parked.lock().retain(|(_, pid)| !pids.contains(pid));
            self.free.lock().extend(chunks);
            self.moved.lock().push("park");
        } else if self.disk_free.load(Ordering::Relaxed) >= want {
            self.disk_free.fetch_sub(want, Ordering::Relaxed);
            self.host_used.fetch_sub(want, Ordering::Relaxed);
            self.moved.lock().push("spill");
        } else {
            return false;
        }
        let planner = planner.clone();
        crate::rt::spawn(async move { planner.parked_down_done(true) });
        true
    }
    fn reclaim_idle(&self, short: super::pool::IdleShort) -> super::pool::IdleReclaim {
        if let Some(returning) = self.retiring.lock().as_mut() {
            let want = (short.kv_pages as usize).saturating_sub(returning.len());
            let mut cache = self.cache.lock();
            let take = want.min(cache.len());
            let dropped: Vec<u32> = cache.drain(..take).collect();
            self.reclaimed.fetch_add(dropped.len() as u32, Ordering::Relaxed);
            let freed = dropped.len() as u32;
            returning.extend(dropped);
            return super::pool::IdleReclaim {
                freed,
                returning: returning.len() as u32,
            };
        }
        let dropped: Vec<u32> = std::mem::take(&mut *self.cache.lock());
        let count = dropped.len() as u32;
        self.reclaimed.fetch_add(count, Ordering::Relaxed);
        self.free.lock().extend(dropped);
        super::pool::IdleReclaim {
            freed: count,
            returning: 0,
        }
    }
    fn reclaimable_idle(&self) -> u32 {
        self.cache.lock().len() as u32
    }
    fn idle_below_device(&self) -> (u32, u32) {
        (0, self.disk_idle.load(Ordering::Relaxed))
    }
    fn suspend_capable(&self) -> bool {
        self.host > 0
            || self.disk_free.load(Ordering::Relaxed) > 0
            || self.disk_idle.load(Ordering::Relaxed) > 0
    }
    fn locus(&self) -> (usize, usize) {
        (self.model, 0)
    }
    fn guest_room_free(&self, run: u32, pieces: u32) -> bool {
        self.runs_free(run, pieces)
    }
    fn guest_free_runs(&self, run: u32) -> u32 {
        (0..).take_while(|&k| self.runs_free(run, k + 1)).count() as u32
    }
    /// A rung here is the free chunks its raise takes, anywhere.
    fn rise_room(&self, _pid: ProcessId, rung: u32) -> super::pool::RiseRoom {
        if rung > self.total {
            return super::pool::RiseRoom::Past {
                need: rung,
                own: 0,
                total: self.total,
            };
        }
        if self.free.lock().len() as u32 >= rung {
            super::pool::RiseRoom::Fits
        } else {
            super::pool::RiseRoom::Short { pieces: rung }
        }
    }
    fn compaction_comes(&self, run: u32, pieces: u32) -> bool {
        let comes = !self.runs_free(run, pieces) && self.compactable.load(Ordering::Acquire);
        if comes {
            self.compactions.fetch_add(1, Ordering::Relaxed);
        }
        comes
    }
    fn rs_as_pages(&self, rs: RsUnits) -> u32 {
        rs.slots * self.rs_pages.load(Ordering::Relaxed)
    }
    fn chunks_as_pages(&self, chunks: u32) -> u32 {
        chunks
    }
    /// The chunks its own restore retakes first, single chunks.
    fn unpark_need(&self, _pid: ProcessId) -> super::pool::RunNeed {
        super::pool::RunNeed::one(1, self.own.lock().len() as u32)
    }
    fn state_chunks(&self) -> u32 {
        self.total
    }
    fn state_chunks_most(&self) -> u32 {
        self.total
    }
    fn room_gives(&self) -> u64 {
        self.gives.load(Ordering::Relaxed)
    }
    /// A resident's kv chunks and parked state lying on the device are
    /// room it holds, as the registry would count them for a real one.
    fn holds_device_room(&self, pid: ProcessId) -> bool {
        self.kv.lock().iter().any(|&(_, holder)| holder == pid)
            || self.parked.lock().iter().any(|&(_, holder)| holder == pid)
            || *self.occupant.lock() == Some(pid)
            || self.shared.lock().iter().any(|held| held.by.iter().any(|&(holder, _)| holder == pid))
    }
    fn room_beside_own(&self, _pid: ProcessId, run: u32, pieces: u32) -> bool {
        let own = self.own.lock().clone();
        let kept: Vec<u32> = self.free.lock().iter().copied().filter(|c| !own.contains(c)).collect();
        let saved = std::mem::replace(&mut *self.free.lock(), kept);
        let fits = self.runs_free(run, pieces);
        *self.free.lock() = saved;
        fits
    }
    fn chunk_occupants(&self, pids: &[ProcessId]) -> super::pool::ChunkMap {
        let kv: Vec<super::pool::ChunkHeld> = self
            .kv
            .lock()
            .iter()
            .filter(|(_, pid)| pids.contains(pid))
            .map(|&(chunk, pid)| super::pool::ChunkHeld {
                chunk,
                used: 1,
                by: vec![(pid, 1)],
            })
            .chain(self.shared.lock().iter().filter_map(|held| {
                let by: Vec<_> = held.by.iter().copied().filter(|(pid, _)| pids.contains(pid)).collect();
                (!by.is_empty()).then(|| super::pool::ChunkHeld { by, ..held.clone() })
            }))
            .collect();
        if !kv.is_empty() {
            return super::pool::ChunkMap {
                held: kv,
                free: self.free.lock().clone(),
                headroom: *self.headroom.lock(),
                ..Default::default()
            };
        }
        let Some(occupant) = *self.occupant.lock() else {
            return super::pool::ChunkMap::default();
        };
        if !pids.contains(&occupant) {
            return super::pool::ChunkMap::default();
        }
        super::pool::ChunkMap {
            held: (0..self.total)
                .map(|chunk| super::pool::ChunkHeld {
                    chunk,
                    used: 1,
                    by: vec![(occupant, 1)],
                })
                .collect(),
            free: Vec::new(),
            headroom: None,
            segment_chunks: 0,
        }
    }
    fn reserve_device(&self, count: u32) -> Option<Vec<PhysicalKvPageId>> {
        if !self.pages.load(Ordering::Acquire) {
            return None;
        }
        let mut free = self.free.lock();
        if (free.len() as u32) < count {
            return None;
        }
        let at = free.len() - count as usize;
        Some(free.split_off(at).into_iter().map(PhysicalKvPageId).collect())
    }
    fn reserve_device_up_to(&self, count: u32) -> Vec<PhysicalKvPageId> {
        if !self.pages.load(Ordering::Acquire) {
            return Vec::new();
        }
        let mut free = self.free.lock();
        let take = (free.len() as u32).min(count) as usize;
        let at = free.len() - take;
        free.split_off(at).into_iter().map(PhysicalKvPageId).collect()
    }
    fn release_device(&self, pages: Vec<PhysicalKvPageId>) {
        if self.pages.load(Ordering::Acquire) {
            self.free.lock().extend(pages.into_iter().map(|page| page.0));
            self.gives.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn reserve_rs(&self, _need: RsUnits) -> Option<crate::store::rs::RsGrant> {
        Some(Default::default())
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
    fn prepare_demote(&self, _room: usize) -> Option<crate::store::kv::KvResidencyTxn> {
        None
    }
}

impl BindRoomPool {
}

/// A raise whose rung could never fit the pool, even with every resident
/// moved off it, is refused by name at once: never queued, nothing moved,
/// its take never run.
#[tokio::test(flavor = "current_thread")]
async fn a_raise_past_the_whole_pool_is_refused_by_name_at_once() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (owner, holder) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [owner, holder] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    *pool.occupant.lock() = Some(holder);
    let refused = planner
        .fund_rise(owner, 9, || async { unreachable!("a raise past the pool is never taken") })
        .await;
    let error = refused.expect_err("refused");
    assert_eq!(
        error,
        PlannerError::PastRung {
            rung: 9,
            need: 9,
            own: 0,
            total: 8
        }
    );
    assert!(error.to_string().contains("can never fit"), "{error}");
    assert!(planner.diagnostics().queue.is_empty(), "never queued");
    assert_eq!(planner.stats.evictions_started.load(Ordering::Relaxed), 0);
}

/// AMD's 500-concurrent wedge: a bind short of room waited for the fleet
/// to stall before the planner moved anyone for it, as a restore does, and
/// with the other processes decoding it never stalled: no eviction was
/// planned for the bind, ever. A bind is new demand: with a process still
/// decoding (the fleet not stalled), the planner evicts the occupants of
/// the chunks the bind could take, here the idle process holding them.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_short_of_room_is_funded_while_the_fleet_decodes() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (binder, holder, decoding) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [binder, holder, decoding] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| inner.procs.get_mut(&holder).unwrap().idle = true);
    *pool.occupant.lock() = Some(holder);
    let p = planner.clone();
    let waited = crate::rt::spawn(async move { p.wait_bind_room(binder, 3, 1, 0).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        !planner.with_inner(|inner| inner.fleet_stalled()) || planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "a process decodes beside the bind"
    );
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the planner moves the chunks' occupant for the bind, the fleet decoding"
    );
    waited.abort();
}

/// Twelve requests whose first frames together ask three times the pool,
/// all at once, beside a policy that answers every shortfall with nothing
/// freed (Mac's budgeted mix). Those with no room now are refused at
/// admission by name (the user's ruling) and retried once one ahead of
/// them leaves; every one is served in the end.
#[tokio::test(flavor = "current_thread")]
async fn requests_past_the_pool_together_are_all_served_none_refused() {
    // Alone: it raises the policy's global answer.
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    const REQUESTS: usize = 12;
    const PAGES: u32 = 4;
    let pool = Arc::new(RacePool::new(16));
    pool.free.lock().extend((0..16).map(PhysicalKvPageId));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let mut tasks = Vec::new();
    // Requests that have left: a refused one asks again after the next.
    let departed = Arc::new(AtomicU32::new(0));
    for _ in 0..REQUESTS {
        let pid = ProcessId::new_v4();
        planner.register(pid);
        planner.note_admitted(pid);
        let p = planner.clone();
        let departed = departed.clone();
        tasks.push(crate::rt::spawn(async move {
            let demand = Demand {
                kv_pages: PAGES,
                rs: RsUnits::default(),
            };
            // Refused at admission while the pool holds no room for it
            // now, it is retried as the harness's client does, once a
            // request ahead of it has left.
            let mut refusals = 0u32;
            let served = loop {
                let departures = departed.load(Ordering::Acquire);
                match p.acquire(pid, pid, demand).await {
                    Ok(Acquired::Granted(grant)) => {
                        // Its fires run, then it completes: its pages go back.
                        for _ in 0..3 {
                            crate::rt::yield_now().await;
                        }
                        drop(grant);
                        break Ok(refusals);
                    }
                    Ok(_) => break Err("a resident request was not granted".to_string()),
                    Err(PlannerError::AdmissionRefused { .. }) => {
                        refusals += 1;
                        while departed.load(Ordering::Acquire) == departures {
                            crate::rt::yield_now().await;
                        }
                    }
                    Err(error) => break Err(error.to_string()),
                }
            };
            p.unregister(pid);
            p.pages_freed();
            departed.fetch_add(1, Ordering::Release);
            served
        }));
    }
    for _ in 0..5000 {
        if tasks.iter().all(|task| task.is_finished()) {
            break;
        }
        crate::rt::yield_now().await;
    }
    let done = tasks.iter().filter(|task| task.is_finished()).count();
    assert_eq!(done, REQUESTS, "{done} of {REQUESTS} ended");
    let mut refused = 0;
    for task in tasks {
        refused += task.await.expect("the task").expect("served in the end, refused only at admission");
    }
    assert!(refused > 0, "asks past the room now are refused at admission and retried");
    assert_eq!(pool.free.lock().len(), 16, "every page back");
}



/// `pid`, a newcomer, asks `pages` with no room for them now (not on the
/// device, nor there by evictions landing below): refused at admission by
/// name, at once (the user's ruling: the client retries). Its retry stands
/// for an admitted lane's growth (seated), which is contention: it waits.
fn refused_at_admission_then_seated(planner: &Arc<ResidencyPlanner>, pid: ProcessId, pages: u32) {
    use futures::FutureExt;
    let refused = planner
        .acquire(
            pid,
            pid,
            Demand {
                kv_pages: pages,
                rs: RsUnits::default(),
            },
        )
        .now_or_never()
        .expect("refused at once, no wait");
    match refused {
        Err(error @ PlannerError::AdmissionRefused { .. }) => {
            assert_eq!(error.code(), Some(super::ADMISSION_REFUSED));
        }
        Err(other) => panic!("refused at admission, not {other}"),
        Ok(_) => panic!("a newcomer with no room now is refused at admission"),
    }
    planner.with_inner(|inner| inner.procs.get_mut(&pid).expect("registered").seated = true);
}

/// A head parked for two pages behind a holder's grant of the whole pool;
/// `drain` arms the planner's drain task (as `init_planner` does) or
/// leaves it unarmed (each parked waiter plans for itself).
async fn head_behind_a_held_grant(
    drain: bool,
) -> (
    Arc<ResidencyPlanner>,
    AllocationGrant,
    crate::rt::JoinHandle<Result<super::acquire::Acquired, PlannerError>>,
) {
    let pool = Arc::new(RacePool::new(4));
    pool.refill_after_stall(4, 0);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone()));
    if drain {
        planner.arm_drain_task();
    }
    let (holder, head) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [holder, head] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let grant = planner
        .grant_now(
            holder,
            Demand {
                kv_pages: 4,
                rs: RsUnits::default(),
            },
        )
        .expect("the pool is free: the holder's grant is taken at once");
    // A newcomer whose first ask does not fit now is refused at
    // admission; asked again as an admitted lane's growth, it waits.
    refused_at_admission_then_seated(&planner, head, 2);
    let p = planner.clone();
    let head_task = crate::rt::spawn(async move {
        p.acquire(
            head,
            head,
            Demand {
                kv_pages: 2,
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
    assert_eq!(planner.diagnostics().queue.len(), 1, "the head parks");
    // Quiet: the head stays parked with nothing else happening.
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    assert!(!head_task.is_finished(), "nothing frees room: the head waits");
    (planner, grant, head_task)
}

/// Whether `head_task` ends granted within a few scheduler turns: well
/// inside one planner step, never by the short-wait re-report
/// (`pressure::AGAIN`), the only other wake a quiet pool gives it.
async fn served_at_once(
    head_task: crate::rt::JoinHandle<Result<super::acquire::Acquired, PlannerError>>,
    since: std::time::Instant,
) {
    for _ in 0..50 {
        if head_task.is_finished() {
            break;
        }
        crate::rt::yield_now().await;
    }
    assert!(
        head_task.is_finished(),
        "room a dropped grant gave back serves the head within one step"
    );
    assert!(since.elapsed() < crate::policy::pressure::AGAIN, "served before any re-report");
    match head_task.await.expect("the head's task ends") {
        Ok(super::acquire::Acquired::Granted(grant)) => assert_eq!(grant.remaining_kv(), 2),
        Ok(_) => panic!("the head yielded or requoted instead of being served"),
        Err(error) => panic!("the head was refused: {error}"),
    }
}

/// The drop may run under the planner's lock (a waiter's entry, its served
/// grant in it, removed under the lock): it takes no planner lock and plans
/// nothing itself, so it never deadlocks, and the head its room serves is
/// still served once the lock is let go.
#[tokio::test(flavor = "current_thread")]
async fn a_grant_dropped_under_the_planner_lock_does_not_deadlock() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    for drain in [true, false] {
        let (planner, grant, head_task) = head_behind_a_held_grant(drain).await;
        let p = planner.clone();
        let (done, dropped) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let guard = p.lock_inner();
            drop(grant);
            drop(guard);
            // And inside the planner's own critical section, a grant
            // holding pages (two of the four now free; the head asks two).
            let grant = p
                .try_reserve(Demand {
                    kv_pages: 2,
                    rs: RsUnits::default(),
                })
                .expect("four pages stand free");
            p.with_inner(|_| drop(grant));
            let _ = done.send(());
        });
        dropped
            .recv_timeout(Duration::from_secs(10))
            .expect("a grant dropped under the planner's lock must not deadlock");
        served_at_once(head_task, std::time::Instant::now()).await;
    }
}


/// A restore whose guest state and runs pass what the state limit leaves
/// would wait with nothing evictable. A head whose whole step takes more chunks than state
/// may ever take is refused by name as never fitting alone; one that fits
/// waits.
#[test]
fn a_restore_past_every_state_chunk_is_refused_as_never_fitting_alone() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| {
        inner.procs.get_mut(&head).unwrap().state = Residency::Restoring;
    });
    // Its unpark lays four chunks beside a run of one: it fits.
    pool.own.lock().extend(0..4);
    planner.requeue_restore_roles(head, (1, 1), Some(0));
    let key = planner
        .with_inner(|inner| inner.queue.iter().find(|(_, w)| w.pid == head).map(|(k, _)| *k))
        .expect("queued again");
    assert!(planner.never_fits(key).is_none(), "five chunks of eight: it waits");
    // Its unpark lays ten: past the eight chunks state may take.
    pool.own.lock().extend(4..10);
    assert!(
        matches!(planner.never_fits(key), Some(super::PlannerError::PastDevice { .. })),
        "past every state chunk: refused by name"
    );
}

/// ENOSPC on an eviction's write (the disk tier's slot file full): nothing
/// on the device changed. The eviction is taken back, the process kept
/// resident and said by name, and it is no victim until the pool moves;
/// once it has, it is a candidate again.
#[test]
fn an_eviction_to_a_full_tier_is_taken_back_and_its_process_kept() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::with_swap(8, 64))));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    planner.with_inner(|inner| {
        let _ = inner.begin_eviction(pid, Reclaim::default());
    });
    planner.eviction_tier_full(pid, "kv to LocalDisk copy: the tier below is full: ...");
    assert_eq!(planner.stats.eviction_tier_full.load(Ordering::Relaxed), 1, "said by name");
    planner.with_inner(|inner| {
        assert_eq!(inner.procs[&pid].state, Residency::Resident, "kept, resident");
        assert!(!inner.killing.contains(&pid), "not failed");
        assert!(inner.host_swap_blocked.contains(&pid), "no victim until the pool moves");
        assert!(inner.evicting.is_empty(), "its eviction is settled");
    });
    // The pool moves (a fire lands): the block lapses.
    planner.stats.fires_landed.fetch_add(1, Ordering::Relaxed);
    planner.lapse_blocks();
    planner.with_inner(|inner| {
        assert!(!inner.host_swap_blocked.contains(&pid), "a candidate again once the pool moved");
    });
}


// ---------------------------------------------------------------------------
// Replays of field stalls.
//
// A stall's state as its stall line records it (the head and its need, its
// refusal, the free chunks, which processes' slots hold which chunks, who is
// resident and who evicted, the queue) is rebuilt here as a pool and a
// planner, and the planner's passes are run to a fixpoint. The progress
// invariant: once the passes settle, the head is served or something is in
// flight (an eviction, a move, a restore) that will change the generation. A
// settled state with the head waiting and nothing in flight fails, naming
// the head and its need. A new field stall is added as one more snapshot.
// ---------------------------------------------------------------------------

/// Every tier full, the head with no move left: the victim (here named,
/// as an eviction whose copy found the tier below full names it) fails by
/// name with every tier's room ([`PlannerError::TiersFull`]), its waits
/// ending with it (its next ask and its wait for residency both), and no
/// one else fails.
#[tokio::test(flavor = "current_thread")]
async fn every_tier_full_fails_the_lowest_resident_by_name() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, older, younger) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, older, younger] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
        inner.procs.get_mut(&head).unwrap().state = Residency::Evicted;
    });
    assert!(planner.fail_tiers_full(Some(younger)), "one fails");
    let Err(error) = planner
        .acquire(younger, younger, Demand { kv_pages: 1, ..Demand::default() })
        .await
    else {
        panic!("the lowest in the order fails");
    };
    assert!(matches!(error, PlannerError::TiersFull { .. }), "{error}");
    assert!(error.to_string().contains("every tier is full"), "{error}");
    // Its stable code leads its text, read by machine past the worker and
    // the gateway (`gateway::ingress::compat`'s `coded`).
    assert_eq!(error.code(), Some(super::TIERS_FULL));
    assert!(error.to_string().starts_with("[tiers_full] "), "{error}");
    let refused = PlannerError::AdmissionRefused {
        need: 9,
        device: (0, 8),
        host: (0, 0),
        disk: (0, 0),
    };
    assert_eq!(refused.code(), Some(super::ADMISSION_REFUSED));
    assert!(refused.to_string().starts_with("[admission_refused] "), "{refused}");
    // Wrapped in context (a pipeline's, a guest's), the code still leads:
    // found by type in an error chain, or from the planner's record of
    // what it handed the process, never by reading the text.
    let wrapped = anyhow::Error::new(refused).context("fire: acquire 9 pages");
    let found = wrapped
        .chain()
        .find_map(|cause| cause.downcast_ref::<PlannerError>())
        .expect("found by type");
    let led = found.lead(format!("{wrapped:#}"));
    assert!(led.starts_with("[admission_refused] fire: acquire 9 pages: "), "{led}");
    assert_eq!(found.lead(led.clone()), led, "led once");
    assert_eq!(planner.handed(younger), Some(error), "the planner keeps what it handed");
    let led = planner.handed(younger).unwrap().lead(format!("fire: {error}"));
    assert!(led.starts_with("[tiers_full] fire: [tiers_full] "), "{led}");
    assert_eq!(planner.wait_resident(younger).await, Err(error), "its waits end with it");
    // The head and the older resident go on.
    assert!(planner.with_inner(|inner| !inner.failed.contains_key(&head) && !inner.failed.contains_key(&older)));
}
