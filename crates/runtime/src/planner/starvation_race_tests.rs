use super::*;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::inner::{RESERVE_SHARE, RestoreGate};
use super::proc::{PendingBind, Proc, Residency};
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
    fn reclaim_sharing(&self, working_sets: &[crate::store::rs::RsWorkingSetId]) -> usize {
        self.with_rs(|rs| rs.reclaim_sharing(working_sets, crate::policy::retention::Installed))
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

#[test]
fn a_restore_leaves_a_page_for_each_running_resident_process() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let pool = RacePool::new(64);
    pool.refill_after_stall(10, 0);
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(pool) as Arc<dyn PoolPort>));
    let running: Vec<ProcessId> = (0..4).map(|_| ProcessId::new_v4()).collect();
    for &pid in &running {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    // Four lanes resident and running: four pages held back for their next
    // steps, whatever the restore takes.
    planner.with_inner(|inner| {
        assert_eq!(inner.growth_reserve(), 4);
        inner.procs.get_mut(&running[0]).unwrap().idle = true;
        assert_eq!(inner.growth_reserve(), 3, "an idle process grows nothing");
        inner.procs.get_mut(&running[1]).unwrap().state = Residency::Evicted;
        assert_eq!(inner.growth_reserve(), 2, "nor one off the device");
    });
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

/// A rise asks for every free page beside the chunks it lacks, and gives
/// them back at once: past the pool's kv total (the lent span in none of
/// it), it is no hog, where the same ask of kv is.
#[test]
fn a_rise_past_the_kv_total_is_no_hog() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(4))));
    let (kv, rise) = planner.with_inner(|inner| {
        let mut park = |rise: bool| {
            let pid = ProcessId::new_v4();
            let seq = inner.next_seq;
            inner.next_seq += 1;
            let mut proc = Proc::new(seq);
            proc.admitted = true;
            inner.procs.insert(pid, proc);
            let WaitKind::Allocation {
                demand,
                notify,
                outcome,
                yielded,
                ..
            } = allocation(Demand {
                kv_pages: 8,
                rs: RsUnits::default(),
            })
            else {
                unreachable!()
            };
            let key = inner.enqueue(
                pid,
                WaitKind::Allocation {
                    demand,
                    notify,
                    outcome,
                    yielded,
                    rise,
                },
            );
            (key, pid)
        };
        (park(false), park(true))
    });
    assert!(!planner.check_hog(rise.0, rise.1), "a rise is no hog");
    assert!(planner.check_hog(kv.0, kv.1), "the same ask of kv is");
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

#[test]
fn a_burst_serves_exactly_the_fundable_prefix_in_one_pass() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
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
                    WaitKind::Restore { .. } | WaitKind::BindRoom { .. } => unreachable!(),
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
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
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

/// An eviction moves its victim's recurrent state off the device, where
/// the engine's own count of how far a record is folded does not follow:
/// it waits for a fold the device counts to settle, as it waits for the
/// fires, and goes on when it has, whichever way the fire ended. One whose
/// folds the host counted waits for nothing.
#[tokio::test(flavor = "current_thread")]
async fn an_eviction_waits_for_a_fold_the_device_counts() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    use crate::inferlet::process::residency::{ProcessResidency, register_residency};
    use crate::store::rs::RsGeometry;
    use futures::FutureExt;

    for (device_counted, failure) in [
        (false, None),
        (true, None),
        (true, Some("pipeline: forward failed: device lost")),
    ] {
        // No host rows: an eviction that gets as far as asking the store
        // finds the host short and ends there, with no engine to copy.
        let model = crate::store::registry::register_model(16, &[4], &[2]);
        let stores = crate::store::registry::get(model, 0);
        let ws = {
            let mut rs = stores.rs.lock().unwrap();
            let ws = rs.create_working_set(RsGeometry {
                state_size: 64,
                buffer_page_tokens: 4,
            });
            let folded = rs.prepare_group(ws, true, false, None, None, None).unwrap();
            let published = rs.publish_prepared(folded).unwrap();
            rs.settle(published);
            rs.fold(ws, 4);
            if device_counted {
                rs.begin_device_fold(ws);
            }
            ws
        };
        let pid = ProcessId::new_v4();
        let residency = Arc::new(std::sync::Mutex::new(ProcessResidency::default()));
        residency
            .lock()
            .unwrap()
            .rs_working_sets
            .insert((model, 0, ws));
        register_residency(pid, Arc::downgrade(&residency));
        let pool = RacePool {
            rs_model: Some(model),
            ..RacePool::new(0)
        };
        let planner = Arc::new(ResidencyPlanner::new(Arc::new(pool)));
        planner.register(pid);
        planner.note_admitted(pid);

        let mut evicting = Box::pin(super::exec::evict(planner.clone(), pid));
        let asked = evicting.as_mut().now_or_never().is_some();
        assert_eq!(
            asked, !device_counted,
            "only a fold the device counts keeps the suspend waiting"
        );
        if device_counted {
            stores
                .rs
                .lock()
                .unwrap()
                .end_device_fold(ws, failure.is_none().then_some(3), failure);
            assert!(
                evicting.as_mut().now_or_never().is_some(),
                "the fold's end is what it waited for"
            );
        }
        assert_eq!(
            stores.rs.lock().unwrap().frozen(ws),
            Ok(if device_counted && failure.is_none() {
                7
            } else {
                4
            })
        );
    }
}

/// A windowed model's store: a shared pool whose one window list keeps 32
/// tokens over 16-token slots, a slot a chunk, 7 slots beside the null
/// chunk.
fn window_model() -> usize {
    use ::engine::chunk::{
        ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId, SlotKind, SlotRole,
        SlotShape,
    };
    let kind = SlotKind::List {
        tokens: 16,
        rate: 1,
        window: Some(32),
    };
    let facts = ChunkFacts {
        chunk_bytes: 1 << 20,
        total_chunks: 8,
        segment_chunks: 8,
        segments: SegmentFacts {
            committed: 1,
            max: 1,
            growth: Growth::Fixed,
        },
        reserved: vec![ReservedRun {
            what: Reserved::Null,
            first: 0,
            chunks: 1,
        }],
        shapes: vec![SlotShape {
            id: ShapeId(1),
            kind,
            slot_bytes: 1 << 20,
            slots_per_chunk: 1,
            sets_per_working_set: kind.sets_per_working_set(),
            rows: Vec::new(),
            pinned: None,
            role: SlotRole::State,
        }],
        ..ChunkFacts::default()
    };
    let pool = crate::store::shared::SharedPool::new(&facts, 1024).expect("the window pool");
    crate::store::registry::register_model_with_chunks(
        16,
        &[crate::store::registry::EngineStores {
            pools: crate::store::chunks::StorePools::of(&facts).expect("shared facts"),
            max_context: 1024,
            shared: Some(Arc::new(std::sync::Mutex::new(pool))),
            tiers: crate::store::shared::Tiers::default(),
        }],
    )
}

/// The rows a fire of `rows` tokens from `held` lands, and how it moves
/// its window when it commits them all.
fn window_fire(held: u32, rows: u32) -> (std::ops::Range<u32>, crate::store::rs::WindowAdvance) {
    let end = held + rows;
    (
        held..end,
        crate::store::rs::WindowAdvance {
            positioned: true,
            settled: held,
            reach: held,
            fold: Some(u32::MAX),
            commit_end: end,
            head: Some(end),
            keep: None,
        },
    )
}

fn window_demand(
    stores: &crate::store::registry::Stores,
    ws: crate::store::rs::RsWorkingSetId,
    held: u32,
    rows: u32,
) -> Demand {
    let (rows, _) = window_fire(held, rows);
    Demand {
        kv_pages: 0,
        rs: stores.rs.lock().unwrap().window_demand(ws, rows).unwrap(),
    }
}

/// Fires from the grant; the published write is settled unless `hold`.
fn window_write(
    stores: &crate::store::registry::Stores,
    ws: crate::store::rs::RsWorkingSetId,
    held: u32,
    rows: u32,
    mut grant: AllocationGrant,
    hold: bool,
) -> Option<crate::store::rs::write::RsPublished> {
    let (rows, advance) = window_fire(held, rows);
    let mut rs = stores.rs.lock().unwrap();
    let prepared = rs
        .prepare_window(ws, rows, Some(grant.lend_rs()))
        .expect("the grant covers the claim");
    let published = rs.publish_prepared(prepared).unwrap();
    rs.window_advance(ws, advance);
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
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = window_model();
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

    let prefill = window_demand(&stores, a, 0, 96);
    assert_eq!(prefill.rs.pages, 6);
    let in_flight = window_write(
        &stores,
        a,
        0,
        96,
        granted(planner.acquire(pa, pa, prefill).await),
        true,
    );

    let short = window_demand(&stores, b, 0, 64);
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
    window_write(&stores, b, 0, 64, grant, false);
}

// A lone sequence is served by the pages its own window moved past,
// with no other fire left to retire them, and a window that cannot
// fit beside what it holds is refused rather than restarted forever.
#[tokio::test(flavor = "current_thread")]
async fn a_lone_window_is_served_by_what_it_moved_past_or_told_it_never_fits() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    use crate::inferlet::process::residency::{ProcessResidency, register_residency};

    let model = window_model();
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
            let demand = window_demand(&stores, ws, held, rows);
            let Acquired::Granted(grant) = planner.acquire(pid, pid, demand).await? else {
                panic!("a resident process is never yielded");
            };
            window_write(&stores, ws, held, rows, grant, false);
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

#[test]
fn a_seated_lane_grows_before_a_prompt_waiting_to_be_seated() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let planner = Arc::new(
        ResidencyPlanner::new(Arc::new(RacePool::new(8)) as Arc<dyn PoolPort>).seated_first(true),
    );
    let [older, seated, younger] = [(); 3].map(|_| ProcessId::new_v4());
    for pid in [older, seated, younger] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let ring = RsUnits {
        slots: 0,
        pages: 2,
        ..RsUnits::default()
    };
    let prompt = RsUnits {
        slots: 0,
        pages: 3,
        ..RsUnits::default()
    };
    planner.with_inner(|inner| {
        inner.procs.get_mut(&seated).unwrap().seated = true;
        let roomy = RsUnits {
            slots: 0,
            pages: 320,
            ..RsUnits::default()
        };
        assert_eq!(
            inner.seating_reserve(older, ring, prompt, roomy),
            ring,
            "a prompt is seated only past the decoding lane's growth"
        );
        assert_eq!(
            inner.seating_reserve(
                older,
                ring,
                prompt,
                RsUnits {
                    slots: 0,
                    pages: 32,
                    ..RsUnits::default()
                }
            ),
            RsUnits {
                slots: 0,
                pages: 1,
                ..RsUnits::default()
            },
            "and the growth holds back at most a thirty-second of the ring"
        );
        assert!(inner.seating_reserve(seated, ring, prompt, roomy).is_zero());
        assert!(
            inner
                .seating_reserve(older, RsUnits::default(), prompt, roomy)
                .is_zero()
        );

        // A higher-ranked prompt waiting to be seated does not hold the
        // lane's growth back; a higher-ranked seated waiter does.
        inner.enqueue(
            older,
            allocation(Demand {
                kv_pages: 4,
                rs: prompt,
            }),
        );
        assert!(inner.growth_first(seated));
        inner.procs.get_mut(&older).unwrap().seated = true;
        assert!(!inner.growth_first(seated));
        inner.procs.get_mut(&older).unwrap().seated = false;
    });

    // The prompt is short of pages: it suspends no decoding lane, only an
    // idle one.
    let victims = planner.victim_set().expect("the prompt is short");
    assert!(
        victims.members.is_empty(),
        "no lane is suspended mid-decode"
    );
    planner.with_inner(|inner| inner.procs.get_mut(&younger).unwrap().idle = true);
    let victims = planner.victim_set().expect("the prompt is short");
    assert_eq!(
        victims.members.iter().map(|v| v.pid).collect::<Vec<_>>(),
        [younger]
    );
}

/// A window ring alone, its room read a moment before it is handed back,
/// as a busy pool's is.
struct RingPool {
    free: AtomicU32,
    total: u32,
}

impl PoolPort for RingPool {
    fn chunk_stats(&self) -> ChunkStats {
        ChunkStats {
            pages: (0, 0),
            rs: self.rs_stats(),
        }
    }
    fn host_stats(&self) -> (u32, u32) {
        (0, 0)
    }
    fn disk_stats(&self) -> (u32, u32) {
        (0, 0)
    }
    fn rs_stats(&self) -> (RsUnits, RsUnits) {
        let free = self.free.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_micros(200));
        (
            RsUnits {
                slots: 0,
                pages: free,
                ..RsUnits::default()
            },
            RsUnits {
                slots: 0,
                pages: self.total,
                ..RsUnits::default()
            },
        )
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
        false
    }
    fn locus(&self) -> (usize, usize) {
        (0, 0)
    }
    fn rs_growth(&self) -> RsUnits {
        RsUnits {
            slots: 0,
            pages: 2,
            ..RsUnits::default()
        }
    }
    fn reserve_device(&self, _count: u32) -> Option<Vec<PhysicalKvPageId>> {
        Some(Vec::new())
    }
    fn reserve_device_up_to(&self, _count: u32) -> Vec<PhysicalKvPageId> {
        Vec::new()
    }
    fn release_device(&self, _pages: Vec<PhysicalKvPageId>) {}
    fn reserve_rs(&self, need: RsUnits) -> Option<crate::store::rs::RsGrant> {
        self.free
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |free| {
                free.checked_sub(need.pages)
            })
            .ok()
            .map(|_| Default::default())
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

/// Prompts seated at once each see the others' pages gone: the ring keeps
/// the seated lanes' growth however many arrive together.
#[test]
fn prompts_seated_at_once_leave_the_seated_lanes_their_growth() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    const PROMPT: u32 = 64;
    const RING: u32 = 6400;
    let pool = Arc::new(RingPool {
        free: AtomicU32::new(RING),
        total: RING,
    });
    let planner =
        Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>).seated_first(true));
    let pids: Vec<ProcessId> = (0..120).map(|_| ProcessId::new_v4()).collect();
    for &pid in &pids {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let demand = Demand {
        kv_pages: 0,
        rs: RsUnits {
            slots: 0,
            pages: PROMPT,
            ..RsUnits::default()
        },
    };
    let start = std::sync::Barrier::new(pids.len());
    let grants = parking_lot::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for &pid in &pids {
            let (planner, start, grants) = (&planner, &start, &grants);
            scope.spawn(move || {
                start.wait();
                if let Some(grant) = planner.grant_now(pid, demand) {
                    grants.lock().push(grant);
                }
            });
        }
    });
    let seated = u32::try_from(grants.lock().len()).expect("a few lanes");
    let free = pool.free.load(Ordering::SeqCst);
    let held_back = (2 * seated.saturating_sub(1)).min(RING / RESERVE_SHARE);
    assert!(seated > 0, "the ring seats prompts");
    assert!(
        free >= held_back,
        "{seated} prompts seated left {free} pages for the lanes' growth, short of {held_back}"
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

/// Mac's gemma-4-e4b c64 stall: a windowed model on a pool that states no
/// activation rungs and grows (one segment of eight committed at load).
fn growing_window_model() -> usize {
    use ::engine::chunk::{
        ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId, SlotKind, SlotRole,
        SlotShape,
    };
    let kind = SlotKind::List {
        tokens: 16,
        rate: 1,
        window: Some(32),
    };
    let facts = ChunkFacts {
        chunk_bytes: 1 << 20,
        total_chunks: 256,
        segment_chunks: 32,
        segments: SegmentFacts {
            committed: 1,
            max: 8,
            growth: Growth::Bottom,
        },
        reserved: vec![ReservedRun {
            what: Reserved::Null,
            first: 0,
            chunks: 1,
        }],
        shapes: vec![SlotShape {
            id: ShapeId(1),
            kind,
            slot_bytes: 1 << 20,
            slots_per_chunk: 1,
            sets_per_working_set: kind.sets_per_working_set(),
            rows: Vec::new(),
            pinned: None,
            role: SlotRole::State,
        }],
        ..ChunkFacts::default()
    };
    let pool = crate::store::shared::SharedPool::new(&facts, 1024).expect("the window pool");
    crate::store::registry::register_model_with_chunks(
        16,
        &[crate::store::registry::EngineStores {
            pools: crate::store::chunks::StorePools::of(&facts).expect("shared facts"),
            max_context: 1024,
            shared: Some(Arc::new(std::sync::Mutex::new(pool))),
            tiers: crate::store::shared::Tiers::default(),
        }],
    )
}

/// 64 processes each fire 48 prompt rows over a windowed model whose pool
/// commits one segment at load and states no rungs: the waiters' window
/// slots reach the governor's demand, the walk commits ahead of them, and
/// every one is served. The governor and the segment walk are played here
/// a turn at a time, the engine committing what it is told.
#[tokio::test(flavor = "current_thread")]
async fn sixty_four_windowed_prompts_grow_a_pool_that_starts_at_one_segment() {
    use crate::governor::run::Steer;
    use crate::governor::Yield;
    use crate::store::shared::SegmentMove;

    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    const PROCESSES: usize = 64;
    let model = growing_window_model();
    let stores = crate::store::registry::get(model, 0);
    let shared = stores.shared.clone().expect("a shared pool");
    assert_eq!(shared.lock().unwrap().mapped_segments(), 1);
    assert!(shared.lock().unwrap().rungs().is_empty());
    let mut grow = crate::store::shared::SharedGrow::new(model, 0, shared.clone());
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(
        RegistryPool::new(model, 0, false).with_shared_grow(grow.clone()),
    )));
    let mut steer = Steer::new(&grow, Yield::new(0.5).unwrap());
    let geom = crate::store::rs::RsGeometry {
        state_size: 0,
        buffer_page_tokens: 16,
    };
    let mut tasks = Vec::new();
    for _ in 0..PROCESSES {
        let pid = ProcessId::new_v4();
        planner.register(pid);
        planner.note_admitted(pid);
        let ws = stores.rs.lock().unwrap().create_working_set(geom);
        let (planner, stores) = (planner.clone(), stores.clone());
        tasks.push(crate::rt::spawn(async move {
            let demand = window_demand(&stores, ws, 0, 48);
            match planner.acquire(pid, pid, demand).await {
                Ok(Acquired::Granted(grant)) => {
                    window_write(&stores, ws, 0, 48, grant, false);
                    Ok(())
                }
                Ok(other) => Err(format!("not granted: {}", matches!(other, Acquired::Yield))),
                Err(error) => Err(error.to_string()),
            }
        }));
    }
    let mut epoch = 0u64;
    for _ in 0..2000 {
        if tasks.iter().all(|task| task.is_finished()) {
            break;
        }
        // The governor's tick, then the walk toward its target.
        steer.turn(&mut grow, 0.0, &mut |_| {});
        loop {
            let goal = grow.goal_for_test();
            epoch += 1;
            let step = shared.lock().unwrap().next_move(goal, epoch);
            match step {
                Some(SegmentMove::Commit { prefix }) => {
                    shared.lock().unwrap().committed(prefix);
                    planner.pages_freed();
                }
                _ => break,
            }
        }
        shared.lock().unwrap().pool_mut().retire_through(epoch);
        // The fires the served prompts ran retire: what their windows moved
        // past is free, as a retiring fire tells the planner.
        planner.pages_freed();
        for _ in 0..4 {
            crate::rt::yield_now().await;
        }
    }
    let done = tasks.iter().filter(|task| task.is_finished()).count();
    let mapped = shared.lock().unwrap().mapped_segments();
    let demand = shared.lock().unwrap().demand();
    assert_eq!(
        done, PROCESSES,
        "{done} of {PROCESSES} served; mapped {mapped} segments, demand {demand:?}"
    );
    for task in tasks {
        task.await.expect("the task").expect("granted");
    }
}

/// A head short of window slots tells the segment walker its whole ask, the
/// free slots it will take included: a shortfall alone under-counts the
/// pool's demand by the free slots, so a pool whose free slots fall just
/// short of the head is never asked to grow.
#[tokio::test(flavor = "current_thread")]
async fn a_window_head_short_of_slots_asks_the_walker_for_its_whole_ask() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = growing_window_model();
    let stores = crate::store::registry::get(model, 0);
    let shared = stores.shared.clone().expect("a shared pool");
    let grow = crate::store::shared::SharedGrow::new(model, 0, shared.clone());
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(
        RegistryPool::new(model, 0, false).with_shared_grow(grow.clone()),
    )));
    let geom = crate::store::rs::RsGeometry {
        state_size: 0,
        buffer_page_tokens: 16,
    };
    let start = |stores: &crate::store::registry::Stores| {
        let pid = ProcessId::new_v4();
        planner.register(pid);
        planner.note_admitted(pid);
        (pid, stores.rs.lock().unwrap().create_working_set(geom))
    };
    // Prompts of 48 rows, three slots each, until fewer than three stand
    // free.
    while stores.rs.lock().unwrap().free().pages >= 3 {
        let (pid, ws) = start(&stores);
        let demand = window_demand(&stores, ws, 0, 48);
        let Ok(Acquired::Granted(grant)) = planner.acquire(pid, pid, demand).await else {
            panic!("a grant while three stand free");
        };
        window_write(&stores, ws, 0, 48, grant, false);
    }
    let free = stores.rs.lock().unwrap().free().pages;
    assert!((1..3).contains(&free), "{free} free");
    let before = shared.lock().unwrap().demand();
    let (pid, ws) = start(&stores);
    let demand = window_demand(&stores, ws, 0, 48);
    let p = planner.clone();
    let parked = crate::rt::spawn(async move { p.acquire(pid, pid, demand).await });
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    assert!(!parked.is_finished(), "{free} free, three asked: it parks");
    let after = shared.lock().unwrap().demand();
    assert_eq!(
        after.wanted - before.wanted,
        3 << 20,
        "the walker is asked the head's three slots, not the {} the free ones lack",
        3 - free
    );
    parked.abort();
}

/// A window model's list 0 is its pages: a seating reserve stated in
/// `pages` reaches the walker as an ask stated in `windows[0]` does.
#[tokio::test(flavor = "current_thread")]
async fn a_window_store_tells_the_walker_a_reserve_stated_in_pages() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = growing_window_model();
    let stores = crate::store::registry::get(model, 0);
    let shared = stores.shared.clone().expect("a shared pool");
    let wanted = || shared.lock().unwrap().demand().wanted;
    let before = wanted();
    stores.rs.lock().unwrap().note_short_shared(RsUnits {
        pages: 2,
        ..RsUnits::default()
    });
    assert_eq!(wanted() - before, 2 << 20);
    stores
        .rs
        .lock()
        .unwrap()
        .note_short_shared(RsUnits::default());
    assert_eq!(wanted(), before);
}

/// What waits for a process to stop being moved off the device (a handle
/// claiming its sets) wakes when its eviction ends, on no clock.
#[tokio::test(flavor = "current_thread")]
async fn a_wait_for_a_move_to_end_wakes_when_the_eviction_ends() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let planner = Arc::new(ResidencyPlanner::new(
        Arc::new(RacePool::new(4)) as Arc<dyn PoolPort>
    ));
    let agent = ProcessId::new_v4();
    planner.register(agent);
    planner.note_admitted(agent);
    planner.set_idle(agent, true);
    planner.with_inner(|inner| inner.procs.get_mut(&agent).unwrap().state = Residency::Evicting);
    let waiting = {
        let planner = planner.clone();
        crate::rt::spawn(async move { planner.wait_not_moving(agent).await })
    };
    for _ in 0..20 {
        crate::rt::yield_now().await;
    }
    assert!(!waiting.is_finished(), "still being moved: it waits");
    planner.report_evicted(agent, 3, RsUnits::default());
    for _ in 0..20 {
        crate::rt::yield_now().await;
    }
    assert!(waiting.is_finished(), "the eviction ended: it wakes");
}

/// What waits on a process's full departure (a served offload prefill
/// waiting for its pages to be kept) wakes when its last hold drops.
#[tokio::test(flavor = "current_thread")]
async fn a_departure_watch_wakes_when_the_last_hold_drops() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let planner = Arc::new(ResidencyPlanner::new(
        Arc::new(RacePool::new(4)) as Arc<dyn PoolPort>
    ));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    let hold = super::departure::Departure::new(&planner, pid);
    let watch = crate::planner::DepartureWatch::new(pid);
    let waiting = crate::rt::spawn(async move {
        loop {
            let next = crate::planner::DepartureWatch::next();
            tokio::pin!(next);
            next.as_mut().enable();
            if watch.gone() {
                return;
            }
            next.await;
        }
    });
    for _ in 0..20 {
        crate::rt::yield_now().await;
    }
    assert!(!waiting.is_finished(), "still held: it waits");
    drop(hold);
    for _ in 0..20 {
        crate::rt::yield_now().await;
    }
    assert!(waiting.is_finished(), "the last hold dropped: it wakes");
}

/// A watch follows a restart: the old run departing is not the process's
/// departure; the new run's is.
#[tokio::test(flavor = "current_thread")]
async fn a_departure_watch_follows_a_restart() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let planner = Arc::new(ResidencyPlanner::new(
        Arc::new(RacePool::new(4)) as Arc<dyn PoolPort>
    ));
    let (old, new) = (ProcessId::new_v4(), ProcessId::new_v4());
    planner.register(old);
    planner.register(new);
    let watch = crate::planner::DepartureWatch::new(old);
    let old_hold = super::departure::Departure::new(&planner, old);
    let new_hold = super::departure::Departure::new(&planner, new);
    crate::planner::follow_restart(old, new);
    drop(old_hold);
    assert!(!watch.gone(), "the old run left; the process runs on");
    drop(new_hold);
    assert!(watch.gone(), "its newest run departed");
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
    restoring(&planner, moved);
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

/// A process suspended between its prefill bind and its decode bind, its
/// channel parked with it (a registered channel holds `Channel` slots):
/// the decode bind waits for it to be resident again, by the wait a fire
/// takes, and lands once it is restored. Negative control: the bind
/// alone, without the wait, is refused, `RolesError::Parked`.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_by_a_suspended_process_waits_for_its_restore() {
    use crate::store::roles::tests::{FakeEngine, PROGRAM, Pools, holder, pools, registered};
    use crate::store::roles::{Owner, RolesError, bind_instance, bind_when_resident};
    use std::cell::{Cell, RefCell};

    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let planner = Arc::new(ResidencyPlanner::new(
        Arc::new(RacePool::new(8)) as Arc<dyn PoolPort>
    ));
    let pid = holder(1);
    planner.register(pid);
    planner.note_admitted(pid);
    let Pools { roles, .. } = pools();
    let engine = RefCell::new(FakeEngine::default());
    let channel = registered(&roles, &mut engine.borrow_mut(), 1137, 1);
    let binding = ::engine::InstanceBinding {
        program: PROGRAM,
        channels: vec![channel],
        ..::engine::InstanceBinding::default()
    };
    let attempts = Cell::new(0);
    let bind = || {
        attempts.set(attempts.get() + 1);
        bind_instance(
            Some(&roles),
            &mut *engine.borrow_mut(),
            &binding,
            Some(pid),
            |id| id == channel,
        )
    };

    bind().expect("resident, the prefill binds");

    // Suspended: its items, the channel's ring among them, park with it.
    planner.with_inner(|inner| inner.procs.get_mut(&pid).unwrap().state = Residency::Evicting);
    let park = roles
        .lock()
        .unwrap()
        .prepare_park(pid)
        .expect("its items park");
    roles.lock().unwrap().commit(park);
    planner.with_inner(|inner| inner.procs.get_mut(&pid).unwrap().state = Residency::Evicted);

    // Negative control: the bind alone is refused while it is parked.
    let refused = bind().expect_err("a parked channel refuses the bind");
    assert_eq!(
        refused.downcast_ref::<RolesError>(),
        Some(&RolesError::Parked {
            channel,
            holder: Some(pid),
        })
    );
    assert_eq!(engine.borrow().bindings.len(), 1, "nothing bound");

    // The decode bind waits for the restore, not refused.
    attempts.set(0);
    let decode = bind_when_resident(
        pid,
        || std::future::ready(bind()),
        || planner.wait_resident(pid),
        |_, _| std::future::ready(Err::<(), String>("no guest state here".into())),
        || None,
        |_, _| false,
        |_, _| false,
        |_, _| std::future::ready(Err::<(), String>("no guest state here".into())),
        || {},
        |_| {},
    );
    tokio::pin!(decode);
    assert!(futures::poll!(decode.as_mut()).is_pending(), "it waits");
    assert_eq!(attempts.get(), 1, "one bind, refused, then the wait");
    assert_eq!(engine.borrow().bindings.len(), 1);

    // Restored: its items come back to the device, rebound, and it is
    // resident again.
    planner.with_inner(|inner| inner.procs.get_mut(&pid).unwrap().state = Residency::Restoring);
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(pid)
        .unwrap()
        .expect("its items unpark");
    roles.lock().unwrap().commit(unpark);
    let stale = roles.lock().unwrap().stale(pid);
    for rebind in stale {
        roles.lock().unwrap().rebound(Owner::of(&rebind));
    }
    planner.report_restored(pid);

    decode
        .await
        .expect("the decode bind lands once it is restored");
    assert_eq!(attempts.get(), 2, "bound again once, after the restore");
    assert_eq!(engine.borrow().bindings.len(), 2);
}

/// A pool whose segment walk is `grow` over `pool`, the planner's grow
/// asking it; the device dry, a host to evict to, and a head short one page
/// with younger residents behind it. The head's entry key.
fn short_of_a_grow(
    pool: crate::store::shared::SharedPool,
    target: u32,
) -> (
    Arc<ResidencyPlanner>,
    crate::store::shared::SharedGrow,
    EntryKey,
) {
    use crate::governor::PoolTarget;
    let shared: crate::store::shared::Shared = Arc::new(std::sync::Mutex::new(pool));
    let planner_cell: Arc<std::sync::OnceLock<std::sync::Weak<ResidencyPlanner>>> = Arc::default();
    let told = planner_cell.clone();
    let mut grow = crate::store::shared::SharedGrow::new(0, 44, shared).with_no_grow(move || {
        if let Some(planner) = told.get().and_then(std::sync::Weak::upgrade) {
            planner.no_grow();
        }
    });
    let _ = grow.retarget(u64::from(target));
    let port = RacePool {
        grow: Some(grow.clone()),
        ..RacePool::with_swap(4, 64)
    };
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(port)));
    planner_cell.set(Arc::downgrade(&planner)).unwrap();
    let key = planner.with_inner(|inner| {
        let mut keys = Vec::new();
        for _ in 0..3 {
            let pid = ProcessId::new_v4();
            let seq = inner.next_seq;
            inner.next_seq += 1;
            let mut proc = Proc::new(seq);
            proc.admitted = true;
            inner.procs.insert(pid, proc);
            keys.push(pid);
        }
        inner.enqueue(
            keys[0],
            allocation(Demand {
                kv_pages: 1,
                rs: RsUnits::default(),
            }),
        )
    });
    (planner, grow, key)
}

/// Whether the plan went on to `plan_eviction` for the head `key`: the
/// eviction rung deferred, a move started, or the head noted short for the
/// policy (nothing the planner may move covers it).
fn reached_eviction(planner: &ResidencyPlanner, key: EntryKey) -> bool {
    planner.stats.eviction_deferrals.load(Ordering::Relaxed) > 0
        || planner.stats.evictions_started.load(Ordering::Relaxed) > 0
        || planner.short.lock().is_short(key.1)
}

/// Mac's over-demand stall reworked (Metal, M1, Qwen3-0.6B, 64 single-lane
/// requests, 0 of 539 pages free, 0 evictions for good): the prefix whole
/// and a lowered rung lending span no segment backs, a head short of room
/// gets an eviction plan, and nothing waits on a span commit. Before, the
/// planner was told the pool grows (the lent span) while the walk's goal
/// stopped at the prefix, so the plan returned on the grow and never
/// reached `plan_eviction`.
#[tokio::test(flavor = "current_thread")]
async fn a_head_short_beside_an_unbacked_lent_span_gets_an_eviction_plan() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = crate::store::shared::tests::grow_pool(true, true);
    assert_eq!(
        pool.pool().span_to_commit(),
        Some(1),
        "a span lent, unbacked"
    );
    let (planner, grow, key) = short_of_a_grow(pool, u32::MAX);
    planner.plan();
    assert_eq!(
        planner.stats.grow_waits.load(Ordering::Relaxed),
        0,
        "no plan waits on a grow"
    );
    assert!(!grow.grow_promised(), "nothing promised");
    assert!(
        reached_eviction(&planner, key),
        "the head is funded by eviction"
    );
    let shared = grow.shared().clone();
    let goal = grow.goal_for_test();
    assert!(
        !matches!(
            shared.lock().unwrap().next_move(goal, 1),
            Some(crate::store::shared::SegmentMove::CommitSpan { .. })
        ),
        "and the walk backs no span"
    );
}

/// A grow the plan waits on that the walk does not make (the governor
/// lowered its target since the promise) wakes the planner with no grow,
/// and that plan goes on past the grow to eviction, on the event alone.
#[tokio::test(flavor = "current_thread")]
async fn a_grow_the_walk_does_not_make_wakes_the_plan_into_eviction() {
    use crate::governor::PoolTarget;
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = crate::store::shared::tests::grow_pool(false, false);
    let mapped = pool.mapped_segments();
    let (planner, mut grow, key) = short_of_a_grow(pool, u32::MAX);
    planner.plan();
    assert_eq!(planner.stats.grow_waits.load(Ordering::Relaxed), 1);
    assert_eq!(
        planner.diagnostics().grow_parked,
        1,
        "the head parks behind the grow"
    );
    assert!(grow.grow_promised());
    assert!(!reached_eviction(&planner, key), "waiting on the grow");
    let _ = grow.retarget(u64::from(mapped));
    grow.walk().await;
    let d = planner.diagnostics();
    assert_eq!(d.no_grows_total, 1, "the walk told no grow");
    assert_eq!(
        d.grow_waits_total, 1,
        "and the plan it woke did not wait again"
    );
    assert_eq!(d.grow_parked, 0);
    assert!(reached_eviction(&planner, key), "it went on to eviction");
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
        runs >= pieces
    }
}

impl PoolPort for BindRoomPool {
    fn chunk_stats(&self) -> ChunkStats {
        ChunkStats {
            pages: (self.free.lock().len() as u32, self.total),
            rs: Default::default(),
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
        let dropped: Vec<u32> = std::mem::take(&mut *self.cache.lock());
        let count = dropped.len() as u32;
        self.free.lock().extend(dropped);
        count
    }
    fn reclaimable_idle(&self) -> u32 {
        self.cache.lock().len() as u32
    }
    fn idle_below_device(&self) -> (u32, u32) {
        (0, 0)
    }
    fn suspend_capable(&self) -> bool {
        self.host > 0
    }
    fn locus(&self) -> (usize, usize) {
        (self.model, 0)
    }
    fn guest_room_free(&self, run: u32, pieces: u32) -> bool {
        self.runs_free(run, pieces)
    }
    fn chunks_as_pages(&self, chunks: u32) -> u32 {
        chunks
    }
    fn room_gives(&self) -> u64 {
        self.gives.load(Ordering::Relaxed)
    }
    fn chunk_occupants(&self, pids: &[ProcessId]) -> super::pool::ChunkMap {
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
    fn reserve_device(&self, _count: u32) -> Option<Vec<PhysicalKvPageId>> {
        None
    }
    fn reserve_device_up_to(&self, _count: u32) -> Vec<PhysicalKvPageId> {
        Vec::new()
    }
    fn release_device(&self, _pages: Vec<PhysicalKvPageId>) {}
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

/// Idle cache fills the pool: a bind short of a run of 3 chunks waits in
/// the planner's queue, the plan drops the cache (as for a restore short
/// of room for its roles) and serves it in what that left. No close, no
/// park: nothing is evicted.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_short_of_room_lands_once_idle_cache_is_dropped() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..8, 0));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    assert!(!pool.runs_free(3, 1), "the cache holds every chunk");
    let p = planner.clone();
    let waited = crate::rt::spawn(async move { p.wait_bind_room(pid, 3, 1).await });
    waited.await.expect("the wait ends").expect("served");
    assert!(pool.cache.lock().is_empty(), "the cache was dropped for it");
    assert!(pool.runs_free(3, 1));
    let d = planner.diagnostics();
    assert_eq!(d.queue.len(), 0, "off the queue");
    assert_eq!(
        planner.stats.evictions_started.load(Ordering::Relaxed),
        0,
        "no process was parked"
    );
}

/// Resident guest state fills the pool and no cache is left: the bind
/// waits in the queue, and with the fleet stalled the plan names a
/// resident for the planner's victim choice to park, the pages its run
/// stands for its deficit. Once that process's chunks come back (its park
/// landed), the bind is served. A bind of a process that is gone ends.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_short_of_room_lands_once_a_resident_is_parked() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    // The binder is the older: the resident holding the pool is a victim.
    let (binder, holder) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [binder, holder] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| inner.procs.get_mut(&holder).unwrap().idle = true);
    let p = planner.clone();
    let waited = crate::rt::spawn(async move { p.wait_bind_room(binder, 3, 1).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.diagnostics().queue.len() == 1 {
            break;
        }
    }
    assert_eq!(
        planner.diagnostics().queue.len(),
        1,
        "it waits in the queue"
    );
    assert!(!waited.is_finished());
    assert!(planner.with_inner(|inner| inner.fleet_stalled()));
    let set = planner.victim_set().expect("the bind is short");
    assert_eq!(set.head_pid, binder);
    assert_eq!(set.deficit.pages, 3, "its run of 3 chunks, as pages");
    assert_eq!(
        set.members
            .iter()
            .map(|victim| victim.pid)
            .collect::<Vec<_>>(),
        [holder],
        "the planner's victim choice names the resident"
    );
    // The holder's park lands: its chunks are back.
    pool.free.lock().extend(0..8);
    planner.pages_freed();
    waited.await.expect("the wait ends").expect("served");

    let gone = ProcessId::new_v4();
    assert!(matches!(
        planner.wait_bind_room(gone, 3, 1).await,
        Err(PlannerError::Cancelled)
    ));
}

impl BindRoomPool {
    /// Takes a free run of `run` adjacent chunks, as a bind does.
    fn take_run(&self, run: u32) -> Option<Vec<u32>> {
        let mut free = self.free.lock();
        let mut ids = free.clone();
        ids.sort_unstable();
        let first = ids
            .windows(run as usize)
            .find(|window| window.windows(2).all(|pair| pair[1] == pair[0] + 1))?[0];
        let taken: Vec<u32> = (first..first + run).collect();
        free.retain(|id| !taken.contains(id));
        Some(taken)
    }
}

/// Shaped like the capped soak: many short-lived guest binds over a pool
/// idle cache keeps filling. Each bind finds no free run, waits in the
/// planner's queue, is served once the cache is dropped for it, takes its
/// run, fires, and gives it back, where cache takes the chunks again.
/// Every bind lands; none fails for room nobody was asked to free.
#[tokio::test(flavor = "current_thread")]
async fn short_lived_binds_over_a_cache_filled_pool_all_land() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(16, 0..16, 0));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let mut binds = Vec::new();
    for _ in 0..96 {
        let pid = ProcessId::new_v4();
        planner.register(pid);
        planner.note_admitted(pid);
        let (planner, pool) = (planner.clone(), pool.clone());
        binds.push(crate::rt::spawn(async move {
            let run = loop {
                if let Some(run) = pool.take_run(2) {
                    break run;
                }
                planner
                    .wait_bind_room(pid, 2, 1)
                    .await
                    .map_err(|why| format!("{pid}: {why}"))?;
            };
            for _ in 0..2 {
                crate::rt::yield_now().await;
            }
            // Its process ends; idle cache takes its chunks again.
            pool.cache.lock().extend(run);
            planner.unregister(pid);
            planner.pages_freed();
            Ok::<(), String>(())
        }));
    }
    for bind in binds {
        bind.await.expect("the task ends").expect("it landed");
    }
    assert_eq!(
        planner.stats.evictions_started.load(Ordering::Relaxed),
        0,
        "cache alone was dropped for them"
    );
}

/// A process a bind short of guest-state room suspended comes back only
/// with its whole need: its kv pages and the room its pending bind asks.
/// The device has free pages (kv room, a plain restore would go) but no
/// run of 3 free chunks: its restore waits, queued, and nothing evicts for
/// it while the fleet runs. Once a run frees, the room event alone serves
/// it.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_suspended_for_a_bind_waits_for_the_bind_s_room_too() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    // Free pages, but no 3 of them adjacent.
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (binder, runner) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [binder, runner] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    // Its bind asked one run of 3 chunks, and it was parked for it.
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&binder).unwrap();
        proc.state = Residency::Evicted;
        proc.bind_room = Some(PendingBind {
            run: 3,
            pieces: 1,
            pages: 3,
        });
        inner.enqueue(
            binder,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    let state = |planner: &Arc<ResidencyPlanner>| {
        planner.with_inner(|inner| inner.procs.get(&binder).unwrap().state)
    };
    // Control: by kv pages alone it would go now.
    let spare = pool.chunk_stats().pages.0;
    assert_eq!(
        planner.with_inner(|inner| inner.restore_gate(binder, spare)),
        RestoreGate::Serve,
        "its kv room stands"
    );
    planner.pages_freed();
    assert_eq!(state(&planner), Residency::Evicted, "not restored");
    assert_eq!(planner.diagnostics().queue.len(), 1, "it waits, queued");
    assert_eq!(
        planner.stats.evictions_started.load(Ordering::Relaxed),
        0,
        "the fleet runs: nothing parked for it"
    );
    // A run of 3 frees: the room event serves it.
    pool.free.lock().extend([1, 3, 5]);
    planner.pages_freed();
    assert_eq!(state(&planner), Residency::Restoring, "restored now");
    assert_eq!(planner.diagnostics().queue.len(), 0);
}

/// The over-demand stall on the RTX PRO 4500 (Qwen3-0.6B, 6 GiB, 256 x
/// 1536 top-p): every live process waited in the queue, the head a bind
/// short of guest-state room, and no resident had made progress since it
/// came back, so the victim choice found none fresh. The last-resort pick
/// counted only the head's missing kv, none for a bind, and picked nobody:
/// nothing moved for 20 minutes. It counts the pages that free the bind's
/// runs as the victim set does, and a younger resident is parked for it.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_room_head_with_no_fresh_victim_gets_a_last_resort_pick() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model_with_swap(16, &[16], &[64], &[0], &[], &[0]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    // The binder is the oldest; the younger residents hold kv pages.
    let binder = ProcessId::new_v4();
    planner.register(binder);
    planner.note_admitted(binder);
    let victims: Vec<Leaving> = (0..2).map(|_| leaving(&planner, model, false)).collect();
    // None has progressed since it came back: none is fresh.
    planner.with_inner(|inner| {
        for proc in inner.procs.values_mut() {
            proc.progressed = false;
        }
    });
    // Every one of them waits for room for a bind.
    let waits: Vec<_> = std::iter::once(binder)
        .chain(victims.iter().map(|victim| victim.pid))
        .map(|pid| {
            let planner = planner.clone();
            crate::rt::spawn(async move { planner.wait_bind_room(pid, 3, 1).await })
        })
        .collect();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.diagnostics().queue.len() == 3 {
            break;
        }
    }
    assert_eq!(planner.diagnostics().queue.len(), 3, "all of them wait");
    // Every process is queued and none is fresh: the victim choice names
    // none, and the pass falls to the last resort.
    for _ in 0..50 {
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
        planner.poke();
        crate::rt::yield_now().await;
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the last resort parks a younger resident for the bind"
    );
    assert!(planner.stats.e6_relaxations.load(Ordering::Relaxed) > 0);
    for wait in waits {
        wait.abort();
    }
}

/// A span walk funding a bind asks the planner for pages; its process
/// suspended meanwhile, the planner answers at once (`Yield`). The ask is
/// made again only once the process is resident again: asked again at once,
/// the walk spun on its tokio worker with no await between asks, and the
/// planner's own task, woken on that worker, never ran again (the L40S
/// over-demand livelock: every waiter parked with pages free).
#[tokio::test(flavor = "current_thread")]
async fn a_suspended_process_funding_a_span_waits_to_be_resident_before_asking_again() {
    use futures::FutureExt;
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let planner = Arc::new(ResidencyPlanner::new(
        Arc::new(RacePool::with_swap(4, 64)) as Arc<dyn PoolPort>
    ));
    let (pid, peer) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [pid, peer] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    // A peer waits, so the ask is contended and parks rather than taking.
    planner.with_inner(|inner| {
        inner.enqueue(
            peer,
            allocation(Demand {
                kv_pages: 1,
                rs: RsUnits::default(),
            }),
        );
        inner.procs.get_mut(&pid).unwrap().state = Residency::Evicted;
    });
    let mut asked = Box::pin(crate::store::shared::programs::fund_with(
        &planner, pid, pid, 2,
    ));
    assert!(
        asked.as_mut().now_or_never().is_none(),
        "a suspended process's ask came back at once, to be asked again at once"
    );
    let signal = planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&pid).unwrap();
        proc.state = Residency::Resident;
        proc.signal.clone()
    });
    signal.notify_waiters();
    assert_eq!(asked.await, Ok(false), "resident again, it asks again");
}

/// AMD's 500-concurrent wedge, run 2: the planner's head the kv
/// allocation of a prompt not yet seated, the device full, and every other
/// process waiting at a bind for room. Seated first, only an idle or held
/// process yields to such a prompt, and a process waiting at a bind was
/// neither, so the head found no victim and nothing moved. Waiting at a
/// bind it makes no progress until room comes, so it yields as an idle one
/// does; past its wait it no longer does.
#[test]
fn a_process_waiting_at_a_bind_yields_to_a_prompt_not_yet_seated() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let planner = Arc::new(
        ResidencyPlanner::new(Arc::new(RacePool::new(0)) as Arc<dyn PoolPort>).seated_first(true),
    );
    let (waiting, prompt) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [waiting, prompt] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        inner.procs.get_mut(&waiting).unwrap().seated = true;
        inner.enqueue(
            prompt,
            allocation(Demand {
                kv_pages: 16,
                rs: RsUnits::default(),
            }),
        );
    });
    let members = |planner: &Arc<ResidencyPlanner>| {
        planner
            .victim_set()
            .map(|set| set.members.iter().map(|victim| victim.pid).collect::<Vec<_>>())
            .unwrap_or_default()
    };
    assert!(members(&planner).is_empty(), "a seated lane mid-decode does not yield");
    planner.set_bind_waiting(waiting, true);
    assert_eq!(members(&planner), vec![waiting], "waiting at a bind, it yields");
    planner.set_bind_waiting(waiting, false);
    assert!(members(&planner).is_empty(), "past its wait, it no longer does");
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
    let waited = crate::rt::spawn(async move { p.wait_bind_room(binder, 3, 1).await });
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

/// AMD's wedge, its second half: once the rung was lowered (its segments
/// given back) no bind was quoted again for two minutes, the room there
/// and the head waiting on its own serve. Room that moves now has the
/// planner quote the waiting head again: a rung change alone, nothing told
/// the planner, serves the bind in the room it left.
#[tokio::test(flavor = "current_thread")]
async fn a_waiting_bind_is_quoted_again_when_the_rung_moves() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 0));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let binder = ProcessId::new_v4();
    planner.register(binder);
    planner.note_admitted(binder);
    let p = planner.clone();
    let waited = crate::rt::spawn(async move { p.wait_bind_room(binder, 3, 1).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
    }
    assert!(!waited.is_finished(), "no room, no victim: it waits");
    // The rung comes down: its chunks are free, and only the rung says so.
    pool.free.lock().extend(0..8);
    crate::store::shared::rung_changed(pool.locus().1);
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if waited.is_finished() {
            break;
        }
    }
    assert!(waited.is_finished(), "quoted again on the rung's move, and served");
    waited.await.expect("the wait ends").expect("served");
}

/// AMD's wedge on c75d39a0b: the chunk branch chose the same process on
/// every quote, an older one waiting at a bind of its own, and nothing
/// moved: the victim set took a bind waiter as a victim, but the start of
/// the moves declined it (ranked ahead of the head, neither idle nor
/// held). A bind waiter the victim set chose now starts its move.
#[tokio::test(flavor = "current_thread")]
async fn an_older_process_waiting_at_a_bind_is_moved_for_the_head_s_chunks() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (holder, binder, decoding) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    // The holder first: older, ranked ahead of the binder.
    for pid in [holder, binder, decoding] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&holder).unwrap();
        proc.bind_waiting = true;
        proc.progressed = false;
    });
    *pool.occupant.lock() = Some(holder);
    let p = planner.clone();
    let waited = crate::rt::spawn(async move { p.wait_bind_room(binder, 3, 1).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the bind waiter holding the chunks is moved"
    );
    waited.abort();
}

/// An evicted process's gate sleeps until its own residency changes: the
/// planner poked for others and room freed wake it not once (it holds no
/// clock: on a 100 ms re-report of a standing shortfall it woke each
/// waiter ten times a second). Its restore wakes it, once. Every event is
/// driven here in order on one thread, each followed by yields that would
/// run a woken gate to its next park before the count is read: no sleep.
#[tokio::test(flavor = "current_thread")]
async fn a_gate_sleeps_through_pokes_for_others_until_its_restore() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (pid, other) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [pid, other] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| inner.procs.get_mut(&pid).unwrap().state = Residency::Evicted);
    let parks = |planner: &Arc<ResidencyPlanner>| planner.stats.gate_parks.load(Ordering::Relaxed);
    let p = planner.clone();
    let gate = crate::rt::spawn(async move { p.wait_resident(pid).await });
    for _ in 0..50 {
        crate::rt::yield_now().await;
        if parks(&planner) == 1 {
            break;
        }
    }
    assert_eq!(parks(&planner), 1, "parked");
    // No drain task armed: each poke plans now, on this thread.
    for _ in 0..50 {
        planner.poke();
        planner.pages_freed();
        crate::rt::yield_now().await;
        crate::rt::yield_now().await;
    }
    assert!(!gate.is_finished(), "still evicted");
    assert_eq!(parks(&planner), 1, "parked once, woken by nothing of its own");
    let signal = planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&pid).unwrap();
        proc.state = Residency::Resident;
        proc.signal.clone()
    });
    signal.notify_waiters();
    gate.await.expect("the gate ends").expect("resident");
    assert_eq!(parks(&planner), 1, "its restore ends the wait, no park after");
}

/// The policy's answer of nothing is process-wide (any planner's, any
/// tier's), and an evicted gate woke on every one, a restore of its own
/// queued or not, to find nothing changed and park again. A gate sleeps
/// through such answers, its restore short or not: no answer refuses a
/// need that fits. Only a restore past the whole device pool is refused,
/// its gate told as that need is learned.
#[tokio::test(flavor = "current_thread")]
async fn a_gate_sleeps_through_policy_answers_and_is_refused_only_past_the_pool() {
    use crate::policy::pressure::{Tier, answered_nothing};
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (idle, short) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [idle, short] {
        planner.register(pid);
        planner.note_admitted(pid);
        planner.with_inner(|inner| inner.procs.get_mut(&pid).unwrap().state = Residency::Evicted);
    }
    let parks = |planner: &Arc<ResidencyPlanner>| planner.stats.gate_parks.load(Ordering::Relaxed);

    // No restore queued: the policy answering anyone's shortfall is not its.
    let p = planner.clone();
    let idle_gate = crate::rt::spawn(async move { p.wait_resident(idle).await });
    for _ in 0..50 {
        crate::rt::yield_now().await;
        if parks(&planner) == 1 {
            break;
        }
    }
    assert_eq!(parks(&planner), 1, "parked");
    for _ in 0..10 {
        answered_nothing(Tier::Device);
        answered_nothing(Tier::Host);
        crate::rt::yield_now().await;
        crate::rt::yield_now().await;
    }
    assert!(!idle_gate.is_finished(), "still evicted");
    assert_eq!(parks(&planner), 1, "answers that are not its wake it not once");

    // A restore queued short of host room: the answer of nothing is no
    // refusal either; it waits for room, however long.
    let key = planner.with_inner(|inner| {
        inner.enqueue(
            short,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        )
    });
    planner.note_short(key, 0, 1 << 20);
    let p = planner.clone();
    let short_gate = crate::rt::spawn(async move { p.wait_resident(short).await });
    for _ in 0..50 {
        crate::rt::yield_now().await;
        if parks(&planner) == 2 {
            break;
        }
    }
    assert_eq!(parks(&planner), 2, "the short gate parked, once");
    answered_nothing(Tier::Host);
    answered_nothing(Tier::Device);
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    assert!(!short_gate.is_finished(), "short and fitting: it waits on");
    assert!(!idle_gate.is_finished(), "the other gate sleeps through it");
    assert_eq!(parks(&planner), 2, "neither parked again");

    // Its unpark comes back short of runs past the whole pool (three runs
    // of three in eight chunks): its gate is told and refuses it by name.
    planner.with_inner(|inner| inner.procs.get_mut(&short).unwrap().state = Residency::Restoring);
    planner.with_inner(|inner| inner.queue.remove(&key));
    planner.requeue_restore_roles(short, (3, 3));
    for _ in 0..50 {
        crate::rt::yield_now().await;
        if short_gate.is_finished() {
            break;
        }
    }
    let ended = short_gate.await.expect("the gate ends");
    assert!(
        matches!(ended, Err(PlannerError::PastDevice { runs: Some((3, 3)), .. })),
        "past the pool: refused by name: {ended:?}"
    );
    assert!(!idle_gate.is_finished(), "the other gate sleeps through it");
    idle_gate.abort();
}

/// Reuse after release, the eviction window (the 1200-beam divergence:
/// beam_search's verify rescore registers a fresh working set per
/// hypothesis mid-process and prefills on it). P's set S0 stands when the
/// planner moves P off Resident; the eviction fences and quiesces it (its
/// snapshot). A set S1 that P registers after that, before the park, was
/// born unfenced: its prefill took its lease, got its grant, and ran on
/// P's instance while the park copied and freed its slots, which another
/// owner then took and wrote. Now S1 is registered under the lock P left
/// Resident under: born fenced, its prefill does not fire, and no grant is
/// given to P, until the restore. The park's copy is P's last completed
/// state, and no slot goes to another owner under a fire of P's.
#[tokio::test(flavor = "current_thread")]
async fn a_set_registered_mid_eviction_prefills_only_after_the_restore() {
    use crate::inferlet::process::residency::{
        ProcessResidency, kv_lease_quiescent, kv_suspend_handles, register_residency,
    };
    use crate::store::kv::working_set::{FireLeaseError, KvWorkingSet};
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let stores = crate::store::registry::get(model, 0);
    let fresh =
        || crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| kv.create_working_set());
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    let residency = Arc::new(std::sync::Mutex::new(ProcessResidency::default()));
    register_residency(pid, Arc::downgrade(&residency));
    let s0 = KvWorkingSet::new(model, 0, fresh(), 16);
    planner.register_kv_working_set(pid, &residency, &s0);
    drop(s0.fire_lease().expect("resident: S0 fires"));

    // The eviction: off Resident under the planner's lock, then its
    // snapshot fenced and quiesced.
    planner.with_inner(|inner| inner.procs.get_mut(&pid).unwrap().state = Residency::Evicting);
    let snapshot = kv_suspend_handles(pid, model, 0);
    assert_eq!(snapshot.len(), 1);
    for handle in &snapshot {
        handle.fence();
        handle.quiesce().await;
    }

    // Between the snapshot and the park, P registers a fresh set for a
    // rescore, and its prefill asks to fire.
    let s1 = KvWorkingSet::new(model, 0, fresh(), 16);
    planner.register_kv_working_set(pid, &residency, &s1);
    assert!(
        matches!(s1.fire_lease(), Err(FireLeaseError::Fenced)),
        "a set registered mid-eviction is born fenced: its prefill does not fire before the restore"
    );
    let prefill = Demand {
        kv_pages: 2,
        rs: RsUnits::default(),
    };
    assert!(
        planner.grant_now(pid, prefill).is_none(),
        "no grant for a fire of a process not resident"
    );
    assert!(
        planner.grant_now(pid, Demand::default()).is_none(),
        "not even an empty one"
    );
    // The park runs with no fire of P's in flight on any set it holds:
    // what it copies is P's last completed state, and the slots it frees
    // go to another owner under no fire of P's.
    assert!(kv_lease_quiescent(pid, model, 0));

    // The restore: every set P holds fires again, the rescore's included.
    planner.with_inner(|inner| inner.procs.get_mut(&pid).unwrap().state = Residency::Restoring);
    planner.report_restored(pid);
    for set in [&s0, &s1] {
        drop(set.fire_lease().expect("resident again: it fires"));
    }
    assert!(planner.grant_now(pid, Demand::default()).is_some());
}

/// AMD's stall on adc4c1506 (7+ minutes, `head_kind=restore`,
/// `restore_roles_short` climbing, 143 processes at a bind): the head
/// restore had its kv pages and was short only of an Instance run, and it
/// waited for the fleet to stall, which a fleet of processes waiting at a
/// bind never did (they were neither idle nor parked in the planner). A
/// bind waiter makes no progress: such a fleet is stalled, and the restore's
/// runs are funded as a bind's are, the chunks' occupant (younger, at a
/// bind) moved for it. (Funding a restore while the fleet decodes thrashed:
/// 3x the wall on the M1's beam stress, 15,578 role waits.)
#[tokio::test(flavor = "current_thread")]
async fn a_restore_short_only_of_an_instance_run_is_funded_once_the_fleet_waits_at_binds() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (restoring, decoding) = (ProcessId::new_v4(), ProcessId::new_v4());
    // The restoring process first: older, ranked ahead.
    for pid in [restoring, decoding] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        let other = inner.procs.get_mut(&decoding).unwrap();
        other.progressed = true;
        other.bind_waiting = true;
        let proc = inner.procs.get_mut(&restoring).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(super::proc::PendingBind {
            run: 3,
            pieces: 1,
            pages: 3,
        });
        inner.enqueue(
            restoring,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    *pool.occupant.lock() = Some(decoding);
    assert!(
        planner.with_inner(|inner| inner.fleet_waits_at_binds()),
        "the fleet waits at binds"
    );
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the chunks' occupant is moved for the restore's run"
    );
}

/// Mac Pro's M5 livelock (7929eb792, 500 at once): nothing resident, the
/// head restore short of role slots, and free runs of its need standing,
/// so the planner sent it again, its unpark came back short (the need
/// understated what the unpark takes), and so on 20 million times, never
/// reclaiming the idle cache that held the device. A restore sent again
/// on free runs of a need and short of that need once more marks it
/// understated; a different need does not.
#[test]
fn a_restore_short_again_on_free_runs_of_its_need_marks_the_need_understated() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    let restoring = |planner: &Arc<ResidencyPlanner>, retried: Option<(u32, u32)>| {
        planner.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid).unwrap();
            proc.state = Residency::Restoring;
            proc.roles_free_retry = retried;
        });
    };
    restoring(&planner, Some((2, 1)));
    planner.requeue_restore_roles(pid, (1, 1));
    assert_eq!(
        planner.with_inner(|inner| inner.procs[&pid].roles_free_short),
        None,
        "a different need: not understated"
    );
    restoring(&planner, Some((1, 1)));
    planner.requeue_restore_roles(pid, (1, 1));
    assert_eq!(
        planner.with_inner(|inner| inner.procs[&pid].roles_free_short),
        Some((1, 1)),
        "short again on free runs of the same need: understated"
    );
    assert_eq!(planner.with_inner(|inner| inner.procs[&pid].roles_free_retry), None);
}

/// The same livelock, the planner's side: the head restore's need is known
/// to understate its unpark, and free runs of it stand; with nothing
/// resident, it is no reason to send the restore again. Room is made for
/// it: the idle cache holding chunks is reclaimed.
#[tokio::test(flavor = "current_thread")]
async fn an_understated_need_s_free_runs_send_no_restore_and_the_idle_cache_goes() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..4, 64));
    pool.free.lock().extend(4..8);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let restoring = ProcessId::new_v4();
    planner.register(restoring);
    planner.note_admitted(restoring);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&restoring).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(super::proc::PendingBind {
            run: 1,
            pieces: 1,
            pages: 1,
        });
        proc.roles_free_short = Some((1, 1));
        inner.enqueue(
            restoring,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    assert!(pool.guest_room_free(1, 1), "free runs of its need stand");
    assert!(planner.with_inner(|inner| inner.fleet_stalled()), "nothing resident");
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if pool.cache.lock().is_empty() {
            break;
        }
    }
    assert!(pool.cache.lock().is_empty(), "the idle cache is reclaimed for it");
    assert_eq!(
        planner.stats.restore_roles_short.load(Ordering::Relaxed),
        0,
        "and no restore was sent on the understated need"
    );
}

/// The M1 beam-stress freeze: a bind short of room suspended its own
/// process, whose restore was served straight back into the room its park
/// gave back, retook its items, and the bind came up short again: suspend,
/// restore, short, 750 times a second. Runs standing free only by what its
/// own park gave back are no room for its bind: its restore waits until
/// room has come free since (the pool's count of room given back moved
/// past the park's end), reclaiming idle cache meanwhile; once it has, it
/// is sent.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_suspended_for_a_bind_waits_for_room_given_back_past_its_own_park() {
    for moved in [false, true] {
        let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
        let pool = Arc::new(BindRoomPool::new(8, 0..2, 64));
        pool.free.lock().extend(4..8);
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
        let pid = ProcessId::new_v4();
        planner.register(pid);
        planner.note_admitted(pid);
        planner.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid).unwrap();
            proc.state = Residency::Evicted;
            proc.bind_room = Some(super::proc::PendingBind {
                run: 1,
                pieces: 1,
                pages: 1,
            });
            proc.bind_room_after = Some(0);
            inner.enqueue(
                pid,
                WaitKind::Restore {
                    demand: Demand::default(),
                },
            );
        });
        pool.gives.store(u64::from(moved), Ordering::Relaxed);
        assert!(pool.guest_room_free(1, 1), "runs stand free");
        planner.poke();
        for _ in 0..200 {
            crate::rt::yield_now().await;
        }
        if moved {
            assert!(!pool.cache.lock().is_empty(), "room came free since: it is sent, nothing reclaimed");
        } else {
            assert!(
                pool.cache.lock().is_empty(),
                "only its own park's room stands: not sent, the idle cache goes for it"
            );
        }
    }
}

/// The M1 stress on restore-4: ten processes suspended for their binds were
/// restored together, each took its items back, and every bind came up
/// short against the others' (suspend, restore, short, 190 times each).
/// One such restore at a time: with two waiting and room for either, the
/// planner serves one; the other waits until the first fires (its bind went
/// through), is parked again, or leaves.
#[tokio::test(flavor = "current_thread")]
async fn bind_suspended_restores_come_back_one_at_a_time() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    pool.free.lock().extend(0..8);
    pool.gives.store(1, Ordering::Relaxed);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (a, b) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [a, b] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        for pid in [a, b] {
            let proc = inner.procs.get_mut(&pid).unwrap();
            proc.state = Residency::Evicted;
            proc.bind_room = Some(super::proc::PendingBind {
                run: 1,
                pieces: 1,
                pages: 1,
            });
            proc.bind_room_after = Some(0);
            inner.enqueue(
                pid,
                WaitKind::Restore {
                    demand: Demand::default(),
                },
            );
        }
    });
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
    }
    let (restoring, queued) = planner.with_inner(|inner| {
        (
            inner.bind_restoring,
            inner
                .queue
                .values()
                .filter(|waiter| matches!(waiter.kind, WaitKind::Restore { .. }))
                .map(|waiter| waiter.pid)
                .collect::<Vec<_>>(),
        )
    });
    let first = restoring.expect("one is restored");
    let other = if first == a { b } else { a };
    assert_eq!(queued, vec![other], "the other waits");
    // The first fires: its bind went through, and the other may come back.
    let fire = planner.fire_in_flight(first);
    assert_eq!(planner.with_inner(|inner| inner.bind_restoring), None);
    drop(fire);
}

/// A short restore that gives up parks its process out of the queue and
/// answers the wait with the error; the process lives on. The M1 / M5
/// stress: its next wait to be resident found it parked, queued nothing,
/// and never ended once the rest drained (evicted = evictions - restores,
/// queue empty). A process waiting to be resident asks for its restore again.
#[tokio::test(flavor = "current_thread")]
async fn a_parked_process_waiting_to_be_resident_is_restored() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    pool.free.lock().extend(0..8);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&pid).unwrap();
        proc.state = Residency::Evicted;
        proc.parked = Some(Demand::default());
    });
    let waited = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        planner.wait_resident(pid),
    )
    .await;
    assert!(
        matches!(waited, Ok(Ok(()))),
        "the parked process is restored: {waited:?}"
    );
}

/// Mac Pro's M5 shape C: a restore short of role room left the queue with
/// its need marked understated, was queued again as its process waited to
/// be resident, and, the mark still set, skipped the free runs that stood
/// once every resident had gone. A restore short of room that fits waits on
/// (no answer of nothing refuses it); one past the whole device pool is
/// refused and parked with what it learned of the room cleared.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_refused_past_the_pool_forgets_what_it_learned_of_the_room() {
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    let need = |pieces| super::proc::PendingBind {
        run: 3,
        pieces,
        pages: 3 * pieces,
    };
    let key = planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&pid).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(need(2));
        proc.roles_free_retry = Some((3, 2));
        proc.roles_free_short = Some((3, 2));
        inner.enqueue(
            pid,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        )
    });
    // Alone, short, and the policy freed nothing: two runs of three fit
    // eight chunks, so it waits on.
    planner.note_short(key, 6 << 20, 0);
    crate::policy::pressure::answered_nothing(crate::policy::pressure::Tier::Device);
    assert!(planner.restore_never_fits(pid).is_none(), "it fits alone: it waits");
    assert!(planner.with_inner(|inner| inner.queue.contains_key(&key)), "still queued");
    // Three runs of three never fit eight chunks: refused, parked.
    planner.with_inner(|inner| inner.procs.get_mut(&pid).unwrap().roles_short = Some(need(3)));
    let refused = planner.restore_never_fits(pid).expect("past the pool: refused");
    assert!(matches!(refused, PlannerError::PastDevice { need: 9, total: 8, .. }), "{refused}");
    planner.with_inner(|inner| {
        let proc = &inner.procs[&pid];
        assert!(proc.parked.is_some(), "parked");
        assert_eq!(proc.roles_short, None, "its role need is the room's then");
        assert_eq!(proc.roles_free_retry, None);
        assert_eq!(proc.roles_free_short, None, "no longer marked understated");
    });
}

/// Mac Pro's beam mix (no refusals): the head restore was short of two
/// runs of four chunks for its roles, 440 such runs stood free, and the
/// serves went flat for 191 s with every thread parked: while the fleet
/// decoded the restore waited for a completion, never looking at the runs
/// that stood free, and no event came to look again. Free runs of its need
/// send it again whatever the fleet does: one plan serves it, with no
/// further event.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_short_of_runs_goes_again_on_free_runs_while_the_fleet_decodes() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    pool.free.lock().extend(0..8);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (restoring, decoding) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [restoring, decoding] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        inner.procs.get_mut(&decoding).unwrap().progressed = true;
        let proc = inner.procs.get_mut(&restoring).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(super::proc::PendingBind {
            run: 4,
            pieces: 2,
            pages: 8,
        });
        inner.enqueue(
            restoring,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    assert!(!planner.with_inner(|inner| inner.fleet_stalled()), "the fleet decodes");
    assert!(pool.guest_room_free(4, 2), "its runs stand free");
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.restores.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert_eq!(
        planner.stats.restores.load(Ordering::Relaxed),
        1,
        "sent again on the free runs, no other event"
    );
    assert_eq!(
        planner.stats.evictions_started.load(Ordering::Relaxed),
        0,
        "no one moved for it"
    );
}

/// Mac Pro's beam mix, stall 1: a bind-suspended process was restored
/// (the one bind-suspended restore at a time), and its bind came up short
/// again and waited for room. The next bind-suspended restore, the queue's
/// head, waited on that one's turn, which waited on a bind behind the
/// head: flat, every thread parked. A restored process whose bind waits
/// again ends its turn, and the head goes.
#[tokio::test(flavor = "current_thread")]
async fn a_restored_bind_that_waits_again_lets_the_next_bind_suspended_restore_go() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (restored, head) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [restored, head] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        inner.bind_restoring = Some(restored);
        inner.procs.get_mut(&head).unwrap().bind_room = Some(super::proc::PendingBind {
            run: 1,
            pieces: 1,
            pages: 1,
        });
    });
    planner.set_bind_waiting(restored, true);
    assert_eq!(
        planner.with_inner(|inner| inner.bind_restoring),
        None,
        "its turn ends as its bind waits again"
    );
    planner.set_bind_waiting(restored, false);
    planner.with_inner(|inner| inner.bind_restoring = Some(restored));
    planner.set_bind_waiting(head, true);
    assert_eq!(
        planner.with_inner(|inner| inner.bind_restoring),
        Some(restored),
        "another's bind waiting leaves its turn"
    );
}

/// The ruling "wait with no deadline": twelve requests whose first frames
/// together ask three times the pool, all at once, beside a policy that
/// answers every shortfall with nothing freed (Mac's budgeted mix: 64
/// requests each refused at its first frame, "device memory short ... for
/// 246µs"). Every one is served as those before it complete; none is
/// refused.
#[tokio::test(flavor = "current_thread")]
async fn requests_past_the_pool_together_are_all_served_none_refused() {
    use crate::policy::pressure::{Tier, answered_nothing};
    // Alone: it raises the policy's global answer.
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    const REQUESTS: usize = 12;
    const PAGES: u32 = 4;
    let pool = Arc::new(RacePool::new(16));
    pool.free.lock().extend((0..16).map(PhysicalKvPageId));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let mut tasks = Vec::new();
    for _ in 0..REQUESTS {
        let pid = ProcessId::new_v4();
        planner.register(pid);
        planner.note_admitted(pid);
        let p = planner.clone();
        tasks.push(crate::rt::spawn(async move {
            let demand = Demand {
                kv_pages: PAGES,
                rs: RsUnits::default(),
            };
            let served = match p.acquire(pid, pid, demand).await {
                Ok(Acquired::Granted(grant)) => {
                    // Its fires run, then it completes: its pages go back.
                    for _ in 0..3 {
                        crate::rt::yield_now().await;
                    }
                    drop(grant);
                    Ok(())
                }
                Ok(_) => Err("a resident request was not granted".to_string()),
                Err(error) => Err(error.to_string()),
            };
            p.unregister(pid);
            p.pages_freed();
            served
        }));
    }
    for _ in 0..5000 {
        if tasks.iter().all(|task| task.is_finished()) {
            break;
        }
        answered_nothing(Tier::Device);
        crate::rt::yield_now().await;
    }
    let done = tasks.iter().filter(|task| task.is_finished()).count();
    assert_eq!(done, REQUESTS, "{done} of {REQUESTS} ended");
    for task in tasks {
        task.await.expect("the task").expect("served, never refused");
    }
    assert_eq!(pool.free.lock().len(), 16, "every page back");
}

/// Only a need that could never fit alone is refused, by name and at once:
/// a first frame past the whole pool, a bind whose runs are past it, and a
/// restore whose kv and runs together are past it. A need of the whole
/// pool is no refusal.
#[tokio::test(flavor = "current_thread")]
async fn a_need_past_the_whole_pool_is_refused_by_name_at_once() {
    use futures::FutureExt;
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(RacePool::new(16));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    let demand = |kv_pages| Demand {
        kv_pages,
        rs: RsUnits::default(),
    };
    let refused = planner
        .acquire(pid, pid, demand(17))
        .now_or_never()
        .expect("refused at once, no wait");
    assert!(
        matches!(refused, Err(PlannerError::Impossible { need: 17, total: 16 })),
        "{:?}",
        refused.err()
    );
    pool.free.lock().extend((0..16).map(PhysicalKvPageId));
    assert!(
        matches!(
            planner.acquire(pid, pid, demand(16)).now_or_never(),
            Some(Ok(Acquired::Granted(_)))
        ),
        "the whole pool fits alone: it is served"
    );

    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    planner.register(pid);
    planner.note_admitted(pid);
    let refused = planner
        .wait_bind_room(pid, 3, 3)
        .now_or_never()
        .expect("refused at once, no wait");
    match refused {
        Err(error @ PlannerError::PastDevice { need: 9, total: 8, runs: Some((3, 3)) }) => {
            assert!(error.to_string().contains("3 runs of 3 adjacent chunks"), "{error}");
        }
        other => panic!("refused by name: {other:?}"),
    }
    assert!(
        planner.wait_bind_room(pid, 4, 2).now_or_never().is_none(),
        "two runs of four fit eight chunks: it waits"
    );
    let other = ProcessId::new_v4();
    planner.register(other);
    planner.note_admitted(other);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&other).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(super::proc::PendingBind {
            run: 3,
            pieces: 2,
            pages: 6,
        });
        inner.enqueue(
            other,
            WaitKind::Restore {
                demand: demand(3),
            },
        );
    });
    let refused = planner
        .wait_resident(other)
        .now_or_never()
        .expect("refused at once, no wait");
    assert!(
        matches!(refused, Err(PlannerError::PastDevice { need: 9, total: 8, .. })),
        "its kv beside its runs is past the pool: {refused:?}"
    );
}

/// A head short only by fragmentation (free chunks enough, no run of its
/// need among them: "0 such runs were free (16 free chunks)") waits however
/// the policy answers; the chunk map names the resident whose chunks make
/// its run, the planner moves it, and once its park lands the head is
/// served.
#[tokio::test(flavor = "current_thread")]
async fn a_head_short_only_by_fragmentation_waits_and_is_served_once_eviction_makes_a_run() {
    use crate::policy::pressure::{Tier, answered_nothing};
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    // Four chunks free, no two adjacent; the resident holds the rest.
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (binder, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [binder, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    *pool.occupant.lock() = Some(resident);
    let p = planner.clone();
    let bind = crate::rt::spawn(async move { p.wait_bind_room(binder, 3, 1).await });
    for _ in 0..400 {
        answered_nothing(Tier::Device);
        crate::rt::yield_now().await;
    }
    assert!(!bind.is_finished(), "short by fragmentation only: it waits");
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the resident whose chunks make its run is moved for it"
    );
    // Its park lands: its chunks are back, a run with them.
    *pool.occupant.lock() = None;
    pool.free.lock().extend([1, 3, 5, 7]);
    pool.gives.fetch_add(1, Ordering::Relaxed);
    planner.pages_freed();
    let served = tokio::time::timeout(Duration::from_secs(5), bind)
        .await
        .expect("the wait ends")
        .expect("the task");
    assert!(matches!(served, Ok(())), "{served:?}");
}
