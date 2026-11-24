use super::*;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::inner::RestoreGate;
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

    /// `n` pages come free by a path that tells the planner nothing but
    /// the pool's room signal (a sequence ending, a fire's epoch retiring).
    fn free_quietly(&self, n: u32) {
        let mut free = self.free.lock();
        let base = 1000 + free.len() as u32;
        free.extend((0..n).map(|i| PhysicalKvPageId(base + i)));
        drop(free);
        if let Some(signal) = self.signal.lock().clone() {
            let free = self.free.lock().len() as u64;
            (signal.0)(crate::store::slots::RoomCount {
                tier: crate::store::slots::RoomTier::Device,
                chunks: free,
                slots: free,
            });
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
        ResidencyPlanner::new(Arc::new(RacePool::new(8)) as Arc<dyn PoolPort>),
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
    // The seated lane decodes: a fire of its in flight.
    let decoding = planner.fire_in_flight(seated);
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
            ring,
            "and a small ring holds back the lane's growth too, no fraction of it in its place"
        );
        assert_eq!(
            inner.seating_reserve(
                older,
                ring,
                prompt,
                RsUnits {
                    slots: 0,
                    pages: 1,
                    ..RsUnits::default()
                }
            ),
            RsUnits {
                slots: 0,
                pages: 1,
                ..RsUnits::default()
            },
            "never more than the ring"
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

    // The prompt is short of pages: the lane with a fire in flight is
    // spared; one between fires is a candidate (the one rule: any resident
    // not in a fire in flight), idle or not.
    let victims = planner.victim_set().expect("the prompt is short");
    assert_eq!(
        victims.members.iter().map(|v| v.pid).collect::<Vec<_>>(),
        [younger],
        "no lane is suspended mid-fire"
    );
    planner.with_inner(|inner| inner.procs.get_mut(&younger).unwrap().idle = true);
    let victims = planner.victim_set().expect("the prompt is short");
    assert_eq!(
        victims.members.iter().map(|v| v.pid).collect::<Vec<_>>(),
        [younger]
    );
    drop(decoding);
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
        Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
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
    // One lane seated and decoding before the rest arrive together.
    let first = planner.grant_now(pids[0], demand).expect("the ring seats one");
    let decoding = planner.fire_in_flight(pids[0]);
    let start = std::sync::Barrier::new(pids.len() - 1);
    let grants = parking_lot::Mutex::new(vec![first]);
    std::thread::scope(|scope| {
        for &pid in &pids[1..] {
            let (planner, start, grants) = (&planner, &start, &grants);
            scope.spawn(move || {
                start.wait();
                if let Some(grant) = planner.grant_now(pid, demand) {
                    grants.lock().push(grant);
                }
            });
        }
    });
    drop(decoding);
    let seated = u32::try_from(grants.lock().len()).expect("a few lanes");
    let free = pool.free.load(Ordering::SeqCst);
    let held_back = (2 * seated.saturating_sub(1)).min(RING);
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
        |_, _, _| false,
        |_, _, _| false,
        |_, _, _| {
            std::future::ready(Err::<crate::planner::BindRoomWait, String>(
                "no guest state here".into(),
            ))
        },
        || {},
        |_| {},
        || std::future::ready(()),
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
/// requests, 0 of 539 pages free): the prefix whole and a lowered rung
/// lending span no segment backs. Span lending: the head waits on the
/// walk backing the lent span (a grow the walk makes, its next move), not
/// on evictions; the grow promised is the walk's own step. Before, a live
/// waiter's grow never backed the span, and the head was funded by
/// evicting residents while the span's chunks stood idle (Mac Pro at
/// 2 GiB on 5120cbc3c: 126 chunks lent, committable_one_chunk=0).
#[tokio::test(flavor = "current_thread")]
async fn a_head_short_beside_an_unbacked_lent_span_waits_on_lending_it() {
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
        1,
        "the plan waits on the span's commit"
    );
    assert!(grow.grow_promised(), "a grow promised");
    assert!(!reached_eviction(&planner, key), "no one is moved for it");
    let shared = grow.shared().clone();
    let goal = grow.goal_for_test();
    assert!(
        matches!(
            shared.lock().unwrap().next_move(goal, 1),
            Some(crate::store::shared::SegmentMove::CommitSpan { .. })
        ),
        "and the walk backs the lent span"
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

/// A head short of room the governor's target holds back (the pool at
/// its target, no grow to promise) plans again when the target rises past
/// what the pool maps, and waits on the grow it can now be promised: the
/// rise is the only event that can serve it, and nothing else plans then
/// (Mac's verified beam + greedy mix beside another suite: the target swung
/// 693 to 779 MiB for 25 minutes, processes parked as "plan: resident
/// again" and nothing planned).
#[tokio::test(flavor = "current_thread")]
async fn a_target_risen_past_what_is_mapped_plans_the_waiting_head_into_a_grow() {
    use crate::governor::PoolTarget;
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = crate::store::shared::tests::grow_pool(false, false);
    let mapped = pool.mapped_segments();
    let (planner, mut grow, key) = short_of_a_grow(pool, mapped);
    super::global::set_test_planner(0, 44, Some(planner.clone()));
    planner.plan();
    assert_eq!(
        planner.stats.grow_waits.load(Ordering::Relaxed),
        0,
        "at its target the pool promises no grow"
    );
    assert!(!grow.grow_promised());
    assert!(planner.with_inner(|inner| inner.queue.contains_key(&key)), "the head waits");
    let passes = planner.stats.plan_passes.load(Ordering::Relaxed);
    let _ = grow.retarget(u64::from(u32::MAX));
    super::global::set_test_planner(0, 44, None);
    assert!(
        planner.stats.plan_passes.load(Ordering::Relaxed) > passes,
        "the rise planned again"
    );
    assert_eq!(
        planner.stats.grow_waits.load(Ordering::Relaxed),
        1,
        "and the head waits on the grow it is promised now"
    );
    assert!(grow.grow_promised());
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

    /// The fire reading what reclaims let go retires: it comes free, and
    /// the planner hears of it as a settle tells it.
    fn retire(&self, planner: &Arc<ResidencyPlanner>) {
        let back = self.retiring.lock().as_mut().map(std::mem::take).unwrap_or_default();
        self.free.lock().extend(back);
        planner.pages_freed();
    }

    /// The kv chunks of residents no longer on the device go back free,
    /// each a room given back: their moves landed.
    fn land_moves(&self, planner: &ResidencyPlanner) -> u32 {
        let gone: Vec<u32> = self
            .kv
            .lock()
            .iter()
            .filter(|&&(_, pid)| {
                // Here a move finds no working set to copy (none is
                // registered): one tried (blocked as found nothing) lands
                // as one that copied would.
                planner.with_inner(|inner| {
                    inner.prepare_blocked.contains(&pid)
                        || inner.procs.get(&pid).is_none_or(|proc| proc.state != Residency::Resident)
                })
            })
            .map(|&(chunk, _)| chunk)
            .collect();
        self.kv.lock().retain(|(chunk, _)| !gone.contains(chunk));
        self.free.lock().extend(&gone);
        self.gives.fetch_add(gone.len() as u64, Ordering::Relaxed);
        gone.len() as u32
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
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
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
    let waited = crate::rt::spawn(async move { p.wait_bind_room(pid, 3, 1, 0).await });
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

/// Idle cache fills the pool and what a reclaim lets go waits on a fire's
/// retirement: the bind short of a run of 3 has 3 chunks reclaimed, and
/// the next pass for the same shortfall counts them as returning, reclaims
/// nothing more and waits, never on a clock. Their retirement wakes it,
/// and it is served in them; the rest of the cache stays.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_waits_for_what_its_reclaim_freed_to_retire_and_reclaims_no_more() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..8, 0));
    *pool.retiring.lock() = Some(Vec::new());
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    let p = planner.clone();
    let mut waited = crate::rt::spawn(async move { p.wait_bind_room(pid, 3, 1, 0).await });
    for _ in 0..3 {
        planner.plan();
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
    }
    let early = tokio::time::timeout(std::time::Duration::from_millis(300), &mut waited).await;
    assert!(early.is_err(), "it waits for the retirement");
    assert_eq!(pool.reclaimed.load(Ordering::Relaxed), 3, "the shortfall, once");
    assert_eq!(pool.cache.lock().len(), 5, "no more of the cache went");
    assert_eq!(
        planner.stats.evictions_started.load(Ordering::Relaxed),
        0,
        "and no one moved for it"
    );
    pool.retire(&planner);
    waited.await.expect("the wait ends").expect("served");
    assert_eq!(pool.reclaimed.load(Ordering::Relaxed), 3);
    assert_eq!(pool.cache.lock().len(), 5);
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
    let waited = crate::rt::spawn(async move { p.wait_bind_room(binder, 3, 1, 0).await });
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
        planner.wait_bind_room(gone, 3, 1, 0).await,
        Err(PlannerError::Cancelled)
    ));
}

impl BindRoomPool {
    /// Takes `count` free chunks anywhere, as a rung raise does; whether
    /// they stood.
    fn take_chunks(&self, count: u32) -> bool {
        let mut free = self.free.lock();
        if (free.len() as u32) < count {
            return false;
        }
        let at = free.len() - count as usize;
        free.truncate(at);
        true
    }

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
                    .wait_bind_room(pid, 2, 1, 0)
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

/// A walk's raise to frames held out under a lowered cap, short of room,
/// beside many binds also waiting for room, nothing else in flight: every
/// chunk is held by idle residents' kv. The raise is a planner head keyed
/// by its frame's process, funded exactly as a bind's room (the chunk map's
/// occupants moved for it), and once its room stands it takes it with the
/// queue held behind it, so no bind is served into that room first. Every
/// bind lands and the raise is taken: none stalls, none is refused.
#[tokio::test(flavor = "current_thread")]
async fn a_raise_short_of_room_among_waiting_binds_is_funded_and_all_are_served() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(16, 0..0, 64)
    });
    pool.disk_free.store(64, Ordering::Relaxed);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    // Idle residents hold every chunk, two each: the oldest, the victims.
    let holders: Vec<ProcessId> = (0..8).map(|_| ProcessId::new_v4()).collect();
    for (at, &pid) in holders.iter().enumerate() {
        planner.register(pid);
        planner.note_admitted(pid);
        planner.with_inner(|inner| inner.procs.get_mut(&pid).unwrap().idle = true);
        let at = at as u32;
        pool.kv.lock().extend([(2 * at, pid), (2 * at + 1, pid)]);
    }
    // Binds of two chunks each, and among them the raise's process (its
    // frame owed 6 chunks of rung): binds before it and after it.
    let mut binds = Vec::new();
    let mut rise = None;
    for at in 0..6 {
        let pid = ProcessId::new_v4();
        planner.register(pid);
        planner.note_admitted(pid);
        let (planner, pool) = (planner.clone(), pool.clone());
        if at == 3 {
            rise = Some(crate::rt::spawn(async move {
                let taker = pool.clone();
                planner
                    .fund_rise(pid, 6, || async move { taker.take_chunks(6) })
                    .await
            }));
            continue;
        }
        binds.push(crate::rt::spawn(async move {
            let run = loop {
                if let Some(run) = pool.take_run(2) {
                    break run;
                }
                planner
                    .wait_bind_room(pid, 2, 1, 0)
                    .await
                    .map_err(|why| format!("{pid}: {why}"))?;
            };
            for _ in 0..2 {
                crate::rt::yield_now().await;
            }
            // Its fire done, its process ends: its run comes back.
            pool.free.lock().extend(run);
            planner.unregister(pid);
            planner.pages_freed();
            Ok::<(), String>(())
        }));
    }
    let rise = rise.unwrap();
    // The moves land as the engine would land them; nothing else happens.
    for _ in 0..64 {
        for _ in 0..200 {
            crate::rt::yield_now().await;
        }
        if rise.is_finished() && binds.iter().all(|bind| bind.is_finished()) {
            break;
        }
        if pool.land_moves(&planner) > 0 {
            planner.pages_freed();
        }
        planner.poke();
    }
    assert!(
        rise.is_finished(),
        "the raise stalled: queue {}",
        planner.debug_queue()
    );
    assert_eq!(
        rise.await.expect("the task ends").expect("not refused"),
        Some(true),
        "served with its six chunks free, and took them"
    );
    for bind in binds {
        assert!(bind.is_finished(), "a bind stalled: queue {}", planner.debug_queue());
        bind.await.expect("the task ends").expect("it landed");
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "residents were moved for them"
    );
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
            singles: 0,
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
    assert_eq!(
        planner.with_inner(|inner| inner.restore_gate(binder)),
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
            crate::rt::spawn(async move { planner.wait_bind_room(pid, 3, 1, 0).await })
        })
        .collect();
    // Each waits: queued, or, its own process evicted meanwhile, withdrawn
    // to wait for its restore.
    let waiting = |planner: &Arc<ResidencyPlanner>| {
        planner.diagnostics().queue.len() + waits.iter().filter(|wait| wait.is_finished()).count()
    };
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if waiting(&planner) == 3 {
            break;
        }
    }
    assert_eq!(waiting(&planner), 3, "all of them wait");
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
        if wait.is_finished() {
            assert_eq!(
                wait.await.expect("the wait ends"),
                Ok(crate::planner::BindRoomWait::Withdrawn),
                "a wait that ended was withdrawn by its own eviction"
            );
        } else {
            wait.abort();
        }
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
        ResidencyPlanner::new(Arc::new(RacePool::new(0)) as Arc<dyn PoolPort>),
    );
    let (waiting, prompt) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [waiting, prompt] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&waiting).unwrap();
        proc.seated = true;
        // Mid-decode: a fire of its in flight (with none anywhere, every
        // resident is a victim).
        proc.fires = 1;
        proc.fired_at = crate::rt::Instant::now();
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
    let waited = crate::rt::spawn(async move { p.wait_bind_room(binder, 3, 1, 0).await });
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
    let waited = crate::rt::spawn(async move { p.wait_bind_room(binder, 3, 1, 0).await });
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
        crate::rt::yield_now().await;
        crate::rt::yield_now().await;
    }
    assert!(!idle_gate.is_finished(), "still evicted");
    assert_eq!(parks(&planner), 1, "answers that are not its wake it not once");

    // A restore queued short of host room (and of a run for its roles, none
    // standing): the answer of nothing is no refusal either; it plans again
    // and waits for room, however long.
    let key = planner.with_inner(|inner| {
        inner.procs.get_mut(&short).unwrap().roles_short = Some(super::proc::PendingBind {
            singles: 0,
            run: 3,
            pieces: 1,
            pages: 3,
        });
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
    planner.requeue_restore_roles(short, (3, 3), None);
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
            singles: 0,
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
        planner.with_inner(|inner| inner.nothing_in_flight()),
        "nothing in flight frees room"
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
/// reclaiming the idle cache that held the device. A restore refused while
/// runs of its need stood (the refusal's own view) marks it understated;
/// one refused with too few standing is a plain short head (the
/// 500-at-once census: refused with free_runs 0, then held behind room
/// given back on a quiescent pool).
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
    // Refused with no run of its need standing (its own view): a plain
    // short head, whatever it was sent on.
    restoring(&planner, Some((1, 1)));
    planner.requeue_restore_roles(pid, (1, 1), Some(0));
    assert_eq!(
        planner.with_inner(|inner| inner.procs[&pid].roles_free_short),
        None,
        "refused short of runs: not understated"
    );
    // Refused while runs of its need stood: understated, at what it saw,
    // its need one run more than stood.
    restoring(&planner, Some((1, 1)));
    planner.requeue_restore_roles(pid, (1, 1), Some(2));
    assert_eq!(
        planner.with_inner(|inner| inner.procs[&pid].roles_free_short),
        Some((1, 3)),
        "refused with runs of its need standing: understated, the need updated"
    );
    assert_eq!(
        planner.with_inner(|inner| inner.procs[&pid].roles_short.map(|need| (need.run, need.pieces))),
        Some((1, 3))
    );
    assert_eq!(planner.with_inner(|inner| inner.procs[&pid].roles_free_short_at), 2);
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
            singles: 0,
            run: 1,
            pieces: 1,
            pages: 1,
        });
        proc.roles_free_short = Some((1, 1));
        // Refused while the four free runs stood.
        proc.roles_free_short_at = 4;
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

/// Mac Pro's M5 stall (sc/np-stack-restore-5, 500 at once), shape C: the
/// host full (its kv the parked processes'), every process evicted, the
/// device nearly free but no run of the head restore's need standing: the
/// role slots parked processes left on the device (no host room at their
/// park) sit between the free chunks, and nothing moved them, nor was
/// anyone resident to evict. Parked state on the device is room a move
/// makes: host kv goes on down to disk to make host room, the parked role
/// slots go to the host, and the restore lands in the run that leaves.
#[tokio::test(flavor = "current_thread")]
async fn parked_state_on_the_device_moves_down_for_the_head_restore_s_run() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 4)
    });
    // Chunks 2 and 5 hold parked role slots; no three adjacent free.
    pool.free.lock().extend([0, 1, 3, 4, 6, 7]);
    pool.host_used.store(4, Ordering::Relaxed);
    pool.disk_free.store(64, Ordering::Relaxed);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, a, b) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, a, b] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    pool.parked.lock().extend([(2, a), (5, b)]);
    planner.with_inner(|inner| {
        for pid in [head, a, b] {
            inner.procs.get_mut(&pid).unwrap().state = Residency::Evicted;
        }
        inner.procs.get_mut(&head).unwrap().roles_short = Some(super::proc::PendingBind {
            singles: 0,
            run: 3,
            pieces: 1,
            pages: 3,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    assert!(!pool.guest_room_free(3, 1), "no run of its need stands");
    assert!(planner.with_inner(|inner| inner.fleet_stalled()), "nothing resident");
    planner.poke();
    for _ in 0..400 {
        crate::rt::yield_now().await;
        if planner.stats.restores.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert_eq!(
        *pool.moved.lock(),
        vec!["spill", "park"],
        "host kv to disk first, then the parked role slots to the host"
    );
    assert!(pool.guest_room_free(3, 1), "a run of its need stands");
    assert_eq!(
        planner.stats.restores.load(Ordering::Relaxed),
        1,
        "and the head restore lands"
    );
    assert_eq!(
        planner.with_inner(|inner| inner.procs[&head].state),
        Residency::Resident
    );
}

/// The M1 stress (unified memory: no host tier): evictions moved only kv
/// to disk, so every Instance chunk stayed held by parked processes' role
/// slots, and the bind-suspended head needing two runs of three chunks
/// found no victim, as no occupant was resident; nothing was in flight and
/// the fleet stopped. The parked role chunks are movable room: they go
/// down to disk at once (no host tier to go through), and the head is
/// restored into the runs that leaves.
#[tokio::test(flavor = "current_thread")]
async fn with_no_host_tier_parked_role_chunks_go_to_disk_for_a_bind_suspended_head() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 0)
    });
    pool.disk_free.store(64, Ordering::Relaxed);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, a, b) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, a, b] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    // Every chunk held by parked role slots, none free.
    pool.parked
        .lock()
        .extend((0..8).map(|chunk| (chunk, if chunk < 4 { a } else { b })));
    planner.with_inner(|inner| {
        for pid in [head, a, b] {
            inner.procs.get_mut(&pid).unwrap().state = Residency::Evicted;
        }
        inner.procs.get_mut(&head).unwrap().bind_room = Some(super::proc::PendingBind {
            singles: 0,
            run: 3,
            pieces: 2,
            pages: 6,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    assert_eq!(pool.host_stats(), (0, 0), "no host tier");
    planner.poke();
    for _ in 0..400 {
        crate::rt::yield_now().await;
        if planner.stats.restores.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert_eq!(*pool.moved.lock(), vec!["disk"], "to disk at once, never a host tier");
    assert!(pool.guest_room_free(3, 2), "two runs of three stand");
    assert_eq!(
        planner.with_inner(|inner| inner.procs[&head].state),
        Residency::Resident,
        "and the bind-suspended head is restored"
    );
}

/// The M1 beam-stress freeze: a bind short of room suspended its own
/// process, whose restore was served straight back into the room its park
/// gave back, retook its items, and the bind came up short again: suspend,
/// restore, short, 750 times a second. Runs standing free only by what its
/// own restore retakes are no room for its bind: it waits, reclaiming idle
/// cache meanwhile; runs beside its own are, read from the pool as it is
/// (not from room given back since its park, which never comes once the
/// fleet waits on it), and it is sent.
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
                singles: 0,
                run: 1,
                pieces: 1,
                pages: 1,
            });
            inner.enqueue(
                pid,
                WaitKind::Restore {
                    demand: Demand::default(),
                },
            );
        });
        // Not moved: the free runs are only what its own restore retakes.
        if !moved {
            pool.own.lock().extend(4..8);
        }
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
    let (a, b, running) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [a, b, running] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    // The fleet runs (a fire in flight): the first restored can fire, so
    // the other waits for it (one at a time holds only while it can).
    let in_flight = planner.fire_in_flight(running);
    planner.with_inner(|inner| {
        for pid in [a, b] {
            let proc = inner.procs.get_mut(&pid).unwrap();
            proc.state = Residency::Evicted;
            proc.bind_room = Some(super::proc::PendingBind {
                singles: 0,
                run: 1,
                pieces: 1,
                pages: 1,
            });
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
    // Its first fire plans at once: the other is restored now.
    let fire = planner.fire_in_flight(first);
    assert_ne!(planner.with_inner(|inner| inner.bind_restoring), Some(first));
    drop(fire);
    // Asking for pages also says its bind went through: the hold must not
    // wait on a grant queued behind the held restore.
    planner.with_inner(|inner| {
        inner.bind_restoring = Some(other);
        inner.enqueue(other, allocation(Demand::default()));
        assert_eq!(inner.bind_restoring, None, "released as it asks for pages");
    });
    drop(in_flight);
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
        singles: 0,
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
    // Alone and short: two runs of three fit
    // eight chunks, so it waits on.
    planner.note_short(key, 6 << 20, 0);
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
            singles: 0,
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
            singles: 0,
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

/// Mac Pro's beam mix (two of two): a bind short of room suspended its own
/// process, the eviction completed, and the serves went flat with every
/// thread parked: the restore waited for room given back since its park,
/// and with the fleet waiting on it as the head nothing gave any, though
/// runs stood free beside what it retakes. The runs beside its own, read as
/// the pool stands, send it with no further event.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_suspended_restore_goes_on_runs_beside_its_own_with_no_room_given_since() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    // Its own retakes chunks 0 and 1; 4..8 stand beside them. No room has
    // been given back since its park.
    pool.free.lock().extend([0, 1, 4, 5, 6, 7]);
    pool.own.lock().extend([0, 1]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&pid).unwrap();
        proc.state = Residency::Evicted;
        proc.bind_room = Some(super::proc::PendingBind {
            singles: 0,
            run: 2,
            pieces: 2,
            pages: 4,
        });
        inner.enqueue(
            pid,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
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
        "sent on the runs beside its own, no further event"
    );
}

/// A wedged fleet, a restore short of room and the policy freeing nothing:
/// over-demand is served, slower, never failed. While another process holds
/// room that can move, the restore keeps waiting and the planner moves a
/// resident for it (the M1 stress: refused at the first wedge, a beam
/// failed with room the next eviction gave it). With nothing else on the
/// device to move it still waits (room given back, an arrival's completion
/// or the policy may come); only a need past the whole pool is refused, by
/// name.
#[tokio::test(flavor = "current_thread")]
async fn a_short_restore_on_a_wedged_fleet_is_funded_and_refused_only_past_the_pool() {
    // Alone: it raises the policy's global answer, which wakes other tests'
    // gates.
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (restoring, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [restoring, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let key = planner.with_inner(|inner| {
        inner.procs.get_mut(&resident).unwrap().idle = true;
        let proc = inner.procs.get_mut(&restoring).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(super::proc::PendingBind {
            singles: 0,
            run: 3,
            pieces: 1,
            pages: 3,
        });
        inner.enqueue(
            restoring,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        )
    });
    *pool.occupant.lock() = Some(resident);
    assert!(planner.is_wedged(), "the fleet stands still");
    planner.note_short(key, 3 << 20, 0);
    assert!(
        planner.never_fits(key).is_none(),
        "a resident can move: the restore waits on"
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
        "the resident is moved for it"
    );

    // Alone on the device, nothing else to move: a need that fits waits on.
    // The resident leaves while being moved for it (a departure mid-move
    // is no completion that sends the restore on its credit).
    planner.with_inner(|inner| inner.procs.get_mut(&resident).unwrap().state = Residency::Evicting);
    planner.unregister(resident);
    planner.with_inner(|inner| {
        inner.evicting.clear();
    });
    assert!(planner.never_fits(key).is_none(), "alone and fitting: it waits");
    // Past the whole pool (three runs of three in eight chunks): refused.
    planner.with_inner(|inner| {
        inner.procs.get_mut(&restoring).unwrap().roles_short = Some(super::proc::PendingBind {
            singles: 0,
            run: 3,
            pieces: 3,
            pages: 9,
        });
    });
    let refused = planner.never_fits(key).expect("past the pool: refused");
    assert!(refused.to_string().contains("past the whole 8-page device pool"), "{refused}");
}

/// The M1 stress on the funding commit: a restore head held while another
/// process restored for its bind had not fired, and that process's bind
/// waited behind the head for good (and the planner, funding the head,
/// moved every resident for a restore it would not serve). The restored
/// one's bind goes first.
#[tokio::test(flavor = "current_thread")]
async fn the_bind_of_the_one_restored_for_it_goes_ahead_of_the_held_restore() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    pool.free.lock().extend(0..8);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (held, restored) = (ProcessId::new_v4(), ProcessId::new_v4());
    // The held restore first: older, ranked ahead.
    for pid in [held, restored] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        inner.bind_restoring = Some(restored);
        let proc = inner.procs.get_mut(&held).unwrap();
        proc.state = Residency::Evicted;
        proc.bind_room = Some(super::proc::PendingBind {
            singles: 0,
            run: 1,
            pieces: 1,
            pages: 1,
        });
        inner.enqueue(
            held,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    let p = planner.clone();
    let bind = crate::rt::spawn(async move { p.wait_bind_room(restored, 1, 1, 0).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if bind.is_finished() {
            break;
        }
    }
    assert!(bind.is_finished(), "the restored one's bind is served");
    assert_eq!(
        planner.stats.evictions_started.load(Ordering::Relaxed),
        0,
        "no one moved for the held restore"
    );
}

/// A short head with nothing resident and parked state on the device that a
/// move down a tier would clear for its runs is no refusal: it waits while
/// the move funds it, and so does one no move funds now. Only a need past
/// the whole pool is refused by name.
#[tokio::test(flavor = "current_thread")]
async fn a_short_head_beside_movable_parked_state_waits_and_is_refused_only_past_the_pool() {
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 4)
    });
    pool.free.lock().extend([0, 1, 3, 4, 6, 7]);
    pool.host_used.store(4, Ordering::Relaxed);
    pool.disk_free.store(64, Ordering::Relaxed);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, a, b) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, a, b] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    pool.parked.lock().extend([(2, a), (5, b)]);
    let need = |pieces| super::proc::PendingBind {
        singles: 0,
        run: 3,
        pieces,
        pages: 3 * pieces,
    };
    let key = planner.with_inner(|inner| {
        for pid in [head, a, b] {
            inner.procs.get_mut(&pid).unwrap().state = Residency::Evicted;
        }
        inner.procs.get_mut(&head).unwrap().roles_short = Some(need(1));
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        )
    });
    planner.note_short(key, 3 << 20, 0);
    assert!(
        planner.never_fits(key).is_none(),
        "parked state moving down stands its run: it waits"
    );
    // Nothing below to take the parked state: no move funds it now, and it
    // waits all the same.
    pool.disk_free.store(0, Ordering::Relaxed);
    assert!(planner.never_fits(key).is_none(), "fits alone: it waits");
    planner.with_inner(|inner| inner.procs.get_mut(&head).unwrap().roles_short = Some(need(3)));
    let refused = planner
        .never_fits(key)
        .expect("three runs of three fit no eight chunks: refused");
    assert!(
        refused.to_string().contains("even with every other process moved"),
        "{refused}"
    );
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
        .wait_bind_room(pid, 3, 3, 0)
        .now_or_never()
        .expect("refused at once, no wait");
    match refused {
        Err(error @ PlannerError::PastDevice { need: 9, total: 8, runs: Some((3, 3)) }) => {
            assert!(error.to_string().contains("3 runs of 3 adjacent chunks"), "{error}");
        }
        other => panic!("refused by name: {other:?}"),
    }
    assert!(
        planner.wait_bind_room(pid, 4, 2, 0).now_or_never().is_none(),
        "two runs of four fit eight chunks: it waits"
    );
    let other = ProcessId::new_v4();
    planner.register(other);
    planner.note_admitted(other);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&other).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(super::proc::PendingBind {
            singles: 0,
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
    let bind = crate::rt::spawn(async move { p.wait_bind_room(binder, 3, 1, 0).await });
    for _ in 0..400 {
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
    assert!(matches!(served, Ok(BindRoomWait::Served)), "{served:?}");
}

/// Mac Pro's M5 soak on f9afad90a (4 of 5 over-demand runs stalled,
/// evicting=1 for good, pool free 0): a process restored into room that
/// just fit its parked state binds a program that needs more, its bind
/// waits in the planner, and the planner, short, evicts that process. Its
/// eviction waited for its fires' leases, whose finalizer was its guest,
/// in the bind; the bind waited on the eviction. Now the bind settles its
/// process's fires before it waits, and its own eviction withdraws its
/// wait for room: it waits for its restore instead, nothing funded for it
/// meanwhile, and binds once restored (no wake lost that way either).
#[tokio::test(flavor = "current_thread")]
async fn a_bind_waiting_for_room_yields_to_its_own_eviction_and_binds_after_the_restore() {
    use crate::store::roles::{RolesError, bind_when_resident};
    use std::cell::Cell;

    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);

    // Short of one chunk until its restore; bound after it.
    let attempts = Cell::new(0u32);
    let restored = Cell::new(false);
    let bind = || {
        attempts.set(attempts.get() + 1);
        std::future::ready(if restored.get() {
            Ok(7u32)
        } else {
            Err(anyhow::Error::new(RolesError::BindShort(Box::new(
                RolesError::Short {
                    role: ::engine::SlotRole::Instance,
                    slots: 1,
                },
            ))))
        })
    };
    let settled = Cell::new(0u32);
    let left = Cell::new(0u32);
    let p = planner.clone();
    let binding = bind_when_resident(
        pid,
        bind,
        || planner.wait_resident(pid),
        |_, _| std::future::ready(Err::<(), String>("no grow here".into())),
        || None,
        |_, _, _| false,
        |_, _, _| false,
        move |run, pieces, singles| {
            let p = p.clone();
            async move {
                p.wait_bind_room(pid, run, pieces, singles)
                    .await
                    .map_err(|why| why.to_string())
            }
        },
        || left.set(left.get() + 1),
        |waits| planner.set_bind_waiting(pid, waits),
        || {
            settled.set(settled.get() + 1);
            std::future::ready(())
        },
    );
    tokio::pin!(binding);
    let bind_rooms = |planner: &Arc<ResidencyPlanner>| {
        planner.with_inner(|inner| {
            inner
                .queue
                .values()
                .filter(|waiter| {
                    waiter.pid == pid && matches!(waiter.kind, WaitKind::BindRoom { .. })
                })
                .count()
        })
    };
    assert!(futures::poll!(binding.as_mut()).is_pending(), "it waits for room");
    assert_eq!(attempts.get(), 1);
    assert_eq!(settled.get(), 1, "what it has in flight settles before it waits");
    assert_eq!(bind_rooms(&planner), 1, "queued in the planner");
    assert!(planner.with_inner(|inner| inner.procs[&pid].bind_waiting));

    // The planner, short, evicts it: its wait for room is withdrawn.
    let wake = planner.with_inner(|inner| inner.begin_eviction(pid, Reclaim::default()));
    for notify in wake {
        notify.notify_waiters();
    }
    for _ in 0..20 {
        assert!(futures::poll!(binding.as_mut()).is_pending(), "it waits for its restore");
        crate::rt::yield_now().await;
    }
    assert_eq!(bind_rooms(&planner), 0, "its wait for room is off the queue");
    assert_eq!(attempts.get(), 1, "nothing sent while it is not resident");
    assert!(left.get() >= 1, "its need is no longer counted");
    assert!(
        !planner.with_inner(|inner| inner.procs[&pid].bind_waiting),
        "waiting for its restore, it is no bind's victim"
    );
    // A bind that asks while its process is evicting waits for the restore
    // at once, queued nowhere.
    assert_eq!(
        planner.wait_bind_room(pid, 1, 1, 0).await,
        Ok(crate::planner::BindRoomWait::Withdrawn)
    );
    assert_eq!(bind_rooms(&planner), 0);

    // Parked, then restored: the bind goes again, and lands.
    planner.with_inner(|inner| inner.procs.get_mut(&pid).unwrap().state = Residency::Evicted);
    planner.with_inner(|inner| inner.procs.get_mut(&pid).unwrap().state = Residency::Restoring);
    restored.set(true);
    planner.report_restored(pid);
    let bound = crate::rt::time::timeout(Duration::from_secs(10), binding)
        .await
        .expect("the bind is woken by its restore")
        .expect("it binds once restored");
    assert_eq!(bound, 7);
    assert_eq!(attempts.get(), 2, "sent again once, after the restore");
    assert_eq!(bind_rooms(&planner), 0, "the queue holds nothing of it");
}

/// The same head with a decoder that never stops firing: funded only once
/// nothing was in flight, it starved. Another process waiting at a bind
/// behind it funds it: the chunks it needs are emptied (their occupant,
/// the waiting binder, parked) while the decoder keeps firing, resident.
/// Without a bind waiting behind it nothing is moved for it.
#[tokio::test(flavor = "current_thread")]
async fn a_roles_short_restore_is_funded_beside_a_steady_decoder_once_a_bind_waits_behind_it() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    for binder_waits in [false, true] {
        let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
        let (restoring, decoder, binder) =
            (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
        for pid in [restoring, decoder, binder] {
            planner.register(pid);
            planner.note_admitted(pid);
        }
        planner.with_inner(|inner| {
            let firing = inner.procs.get_mut(&decoder).unwrap();
            firing.progressed = true;
            firing.fires = 1;
            firing.fired_at = crate::rt::Instant::now();
            inner.procs.get_mut(&binder).unwrap().bind_waiting = binder_waits;
            let proc = inner.procs.get_mut(&restoring).unwrap();
            proc.state = Residency::Evicted;
            proc.roles_short = Some(super::proc::PendingBind {
                singles: 0,
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
        *pool.occupant.lock() = Some(binder);
        assert!(
            !planner.with_inner(|inner| inner.nothing_in_flight()),
            "the decoder fires"
        );
        planner.poke();
        for _ in 0..200 {
            crate::rt::yield_now().await;
            if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
                break;
            }
        }
        let started = planner.stats.evictions_started.load(Ordering::Relaxed);
        if binder_waits {
            assert!(started > 0, "the chunks' occupant is moved for the restore's run");
        } else {
            assert_eq!(started, 0, "control: nothing moved while the fleet decodes");
        }
        assert_eq!(
            planner.with_inner(|inner| inner.procs[&decoder].state),
            Residency::Resident,
            "the decoder keeps firing"
        );
    }
}



/// Mac Pro's disk-only home on 45cf89b8c (500 at once, stalled in
/// seconds): the head a restore with no kv to ask (head_pages=0) and a
/// roles need of a run of 3, 70 free runs of 3 standing (the map's
/// cheapest cost 0), every resident waiting on a grant, nothing firing.
/// The census showed it refused with free_runs 0: a plain short head,
/// sent at once now its runs stand, with no wait on room given back (a
/// quiescent pool gives none). One truly understated (refused while as
/// many runs stood) is not sent again on the same runs (the 20M-round
/// livelock), but is once the pool holds more of them than it saw.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_head_whose_run_stands_is_sent_and_an_understated_one_once_room_is_given() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    for understated in [false, true] {
        let model = crate::store::registry::register_model(16, &[8], &[4]);
        let pool = Arc::new(BindRoomPool {
            model,
            ..BindRoomPool::new(12, 0..0, 0)
        });
        pool.free.lock().extend([0, 1, 2, 6, 7, 8]);
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
        let head = ProcessId::new_v4();
        planner.register(head);
        planner.note_admitted(head);
        let residents: Vec<ProcessId> = (0..3).map(|_| ProcessId::new_v4()).collect();
        for &pid in &residents {
            planner.register(pid);
            planner.note_admitted(pid);
        }
        planner.with_inner(|inner| {
            let proc = inner.procs.get_mut(&head).unwrap();
            proc.state = Residency::Evicted;
            proc.roles_short = Some(super::proc::PendingBind {
                singles: 0,
                run: 3,
                pieces: 1,
                pages: 3,
            });
            // Understated: refused while as many runs stood as stand now.
            if understated {
                proc.roles_free_short = Some((3, 1));
                proc.roles_free_short_at = 2;
            }
            inner.enqueue(
                head,
                WaitKind::Restore {
                    demand: Demand::default(),
                },
            );
            for &pid in &residents {
                inner.enqueue(
                    pid,
                    WaitKind::Allocation {
                        demand: Demand {
                            kv_pages: 1,
                            rs: RsUnits::default(),
                        },
                        notify: Arc::new(Notify::new()),
                        outcome: None,
                        yielded: false,
                        rise: false,
                    },
                );
            }
        });
        assert!(pool.guest_room_free(3, 1), "runs of its need stand");
        let sent = |planner: &Arc<ResidencyPlanner>| {
            planner.with_inner(|inner| inner.procs[&head].roles_free_retry.is_some())
                || planner.stats.restores.load(Ordering::Relaxed) > 0
        };
        planner.poke();
        for _ in 0..200 {
            crate::rt::yield_now().await;
            if sent(&planner) {
                break;
            }
        }
        if !understated {
            assert!(sent(&planner), "its runs stand: it is sent at once");
            assert_eq!(planner.stats.evictions_started.load(Ordering::Relaxed), 0);
            continue;
        }
        assert!(!sent(&planner), "understated: not sent again on the same runs");
        assert_eq!(planner.stats.evictions_started.load(Ordering::Relaxed), 0, "no one moved");
        // More runs of its need stand than its refusal saw (room moved, no
        // completion): it goes once more.
        pool.free.lock().extend([3, 4, 5]);
        planner.pages_freed();
        for _ in 0..200 {
            crate::rt::yield_now().await;
            if sent(&planner) {
                break;
            }
        }
        assert!(sent(&planner), "room given since: sent once more");
    }
}

/// Yields until the planner and the pool stand still: no plan pass, move,
/// restore, serve, park, queue or pool change, and no task ending, for a
/// run of consecutive turns. A verdict read there holds however loaded the
/// machine is (a fixed count of turns could end before the planner had
/// done its work). False where it never stands still within the bound: a
/// planner that keeps stepping with nothing coming.
async fn quiesce(planner: &ResidencyPlanner, pool: &BindRoomPool, finished: impl Fn() -> u64) -> bool {
    let snapshot = || {
        let stats = &planner.stats;
        let (queue, unmet) = planner.with_inner(|inner| {
            (
                inner.queue.len() as u64,
                inner.queue.values().filter(|waiter| waiter.is_unmet()).count() as u64,
            )
        });
        vec![
            stats.plan_passes.load(Ordering::Relaxed),
            stats.evictions_started.load(Ordering::Relaxed),
            stats.restores.load(Ordering::Relaxed),
            stats.serves.load(Ordering::Relaxed),
            stats.parks.load(Ordering::Relaxed),
            stats.compaction_waits.load(Ordering::Relaxed),
            stats.grow_waits.load(Ordering::Relaxed),
            queue,
            unmet,
            pool.free.lock().len() as u64,
            pool.gives.load(Ordering::Relaxed),
            pool.kv.lock().len() as u64,
            pool.parked.lock().len() as u64,
            pool.moved.lock().len() as u64,
            finished(),
        ]
    };
    let mut last = snapshot();
    let mut still = 0;
    for _ in 0..200_000 {
        crate::rt::yield_now().await;
        let now = snapshot();
        if now == last {
            still += 1;
            if still >= 64 {
                return true;
            }
        } else {
            still = 0;
            last = now;
        }
    }
    false
}

/// The planner's liveness, as a property over seeded random states (the
/// soaks' "free chunks and no run" stalls): a pool of random size and
/// fragmentation, disk-only or with a host tier; a head that is a restore
/// short of runs (plain, or understated at some count of runs) or a bind
/// waiting for room; residents waiting on grants or idle; a compaction
/// possible or not. At quiescence: where the head could be served from the
/// map as it stands (its runs stand, and an understated restore's stand
/// beyond what its refusal saw), or a compaction could stand them, some
/// plan step made progress (the head sent or served, a compaction asked,
/// a resident moved); and a quiescent planner stays still (no step
/// repeats with nothing coming). And at every step, beside policy answers
/// of nothing freed, no waiter is refused unless its need could never fit
/// the pool alone (none here can't): no allocation's outcome is an error,
/// and the head's bind or restore gate ends only served. Failing seeds are
/// printed to replay. In a rise seed fires take grants of free chunks and
/// drop them unused, the drop alone telling the planner (no `pages_freed`,
/// no poke): room it gives back that serves the head is the head's within
/// that step.
#[tokio::test(flavor = "current_thread")]
async fn the_planner_makes_progress_wherever_the_map_serves_the_head() {
    // Alone: it raises the policy's global answer.
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            if n == 0 { 0 } else { self.next() % n }
        }
    }
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let mut failures = Vec::new();
    let only: Option<u64> = std::env::var("PIE_TEST_SEED").ok().and_then(|s| s.parse().ok());
    for seed in 0..2000u64 {
        if only.is_some_and(|only| only != seed) {
            continue;
        }
        let mut rng = Rng(seed);
        let total = 8 + rng.below(17) as u32;
        let host = if rng.below(2) == 0 { 0 } else { 64 };
        let pool = Arc::new(BindRoomPool {
            model,
            ..BindRoomPool::new(total, 0..0, host)
        });
        pool.disk_free.store(64, Ordering::Relaxed);
        let free: Vec<u32> = (0..total).filter(|_| rng.below(100) < 45).collect();
        pool.free.lock().extend(free);
        // The pool could compact or not; compaction is out of the
        // planner's paths, so no head ever waits on one (shape 1).
        pool.compactable.store(rng.below(2) == 0, Ordering::Release);
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
        let head = ProcessId::new_v4();
        planner.register(head);
        planner.note_admitted(head);
        let run = 1 + rng.below(3) as u32;
        let pieces = 1 + rng.below(2) as u32;
        let runs_now = pool.guest_free_runs(run);
        let is_restore = rng.below(3) != 0;
        let is_rise = !is_restore && rng.below(2) == 0;
        // Half the rise seeds are a walk's raise to frames held out under
        // the cap (`fund_rise`): funded as a bind's room is, by the chunk
        // map, its chunks any free ones; the rest a pass's (`acquire_rise`).
        let walk_rise = is_rise && seed % 2 == 1;
        // An understated restore: refused while some count of runs stood,
        // at, below or past what stands now.
        // Never at the most runs the pool stands or past (the planner marks
        // no need understated there, `requeue_restore_roles`).
        let understated_at = (is_restore && rng.below(2) == 0)
            .then(|| (runs_now + rng.below(3) as u32).saturating_sub(1).max(pieces))
            .filter(|&at| at < total / run);
        let residents: Vec<ProcessId> = (0..rng.below(4)).map(|_| ProcessId::new_v4()).collect();
        for &pid in &residents {
            planner.register(pid);
            planner.note_admitted(pid);
        }
        // Mixed shapes: in some pools the chunks not free hold residents'
        // kv pages, each resident's whole, emptied for any shape once it
        // moves off the device.
        // A rise seed's chunks not free are all residents' (a real pool's
        // are someone's, which the registry counts): none lies held by no
        // one, as a chunk taken here otherwise does.
        let mixed = !residents.is_empty() && (is_rise || rng.below(2) == 0);
        if is_rise && residents.is_empty() {
            // No one to hold a chunk: every chunk is free.
            let mut free = pool.free.lock();
            free.clear();
            free.extend(0..total);
        }
        if mixed {
            let held: Vec<u32> = (0..total).filter(|id| !pool.free.lock().contains(id)).collect();
            for chunk in held {
                if is_rise || rng.below(10) < 7 {
                    let pid = residents[rng.below(residents.len() as u64) as usize];
                    pool.kv.lock().push((chunk, pid));
                }
            }
        }
        // Fragmented: free chunks and residents' chunks alternate, so the
        // free count covers the head's need but no run of two stands. The
        // head fits the pool alone; only moving residents stands its runs.
        let fragmented = !is_rise && !residents.is_empty() && rng.below(4) == 0;
        if fragmented {
            let mut free = pool.free.lock();
            free.clear();
            free.extend((0..total).step_by(2));
            let mut kv = pool.kv.lock();
            kv.clear();
            for (k, chunk) in (1..total).step_by(2).enumerate() {
                kv.push((chunk, residents[k % residents.len()]));
            }
        }
        // Half the fragmented pools also hold an evicted process's state
        // parked in place (no host tier took its live parts) in every other
        // held chunk: no one runs on it, and only moving it down a tier
        // frees those chunks.
        if fragmented && rng.below(2) == 0 {
            let sleeper = ProcessId::new_v4();
            planner.register(sleeper);
            planner.note_admitted(sleeper);
            planner.with_inner(|inner| {
                inner.procs.get_mut(&sleeper).unwrap().state = Residency::Evicted;
            });
            let mut kv = pool.kv.lock();
            let mut parked = pool.parked.lock();
            let mut k = 0;
            kv.retain(|&(chunk, _)| {
                k += 1;
                if k % 2 == 0 {
                    parked.push((chunk, sleeper));
                    false
                } else {
                    true
                }
            });
        }
        let mixed = mixed || fragmented;
        // A rise's residents are idle: a grant here is never taken by a
        // task, and the fleet it leaves never stalls (a granted waiter is
        // no stalled one), so a rise head would wait on completions this
        // pool never makes.
        let grants = !is_rise && rng.below(2) == 0;
        planner.with_inner(|inner| {
            if is_restore {
                let proc = inner.procs.get_mut(&head).unwrap();
                proc.state = Residency::Evicted;
                proc.roles_short = Some(super::proc::PendingBind {
                    singles: 0,
                    run,
                    pieces,
                    pages: run * pieces,
                });
                if let Some(at) = understated_at {
                    proc.roles_free_short = Some((run, pieces));
                    proc.roles_free_short_at = at;
                }
                inner.enqueue(
                    head,
                    WaitKind::Restore {
                        demand: Demand::default(),
                    },
                );
            }
            for &pid in &residents {
                if grants {
                    inner.enqueue(
                        pid,
                        WaitKind::Allocation {
                            demand: Demand {
                                kv_pages: 1,
                                rs: RsUnits::default(),
                            },
                            notify: Arc::new(Notify::new()),
                            outcome: None,
                            yielded: false,
                            rise: false,
                        },
                    );
                } else {
                    inner.procs.get_mut(&pid).unwrap().idle = true;
                }
            }
        });
        // Not a restore: a bind waiting for room, or an activation rise
        // (a cut fire's raise funded as a planner head, its chunks taken as
        // pages, any chunks), each served or waiting, never refused.
        if is_rise && !walk_rise {
            pool.pages.store(true, Ordering::Release);
        }
        // Grants a fire took and holds unspent (a rise seed's pool hands
        // its free chunks out as pages): one dropped unused gives them back
        // with nothing else said, no `pages_freed`, no poke. Half the rise
        // seeds start with one, taken before the head asks, so the head
        // may wait on the room it holds. Only beside a resident holding
        // chunks: a grant is no process's, and with no one else holding any
        // the head would read as the pool's sole holder (the starvation
        // endgame), not as waiting on room that comes back.
        let mut held: Vec<AllocationGrant> = Vec::new();
        if is_rise && !walk_rise && !pool.kv.lock().is_empty() && rng.below(2) == 0 {
            let free = pool.free.lock().len() as u64;
            if free > 0 {
                let pages = 1 + rng.below(free) as u32;
                if let Some(grant) = planner.try_reserve(Demand {
                    kv_pages: pages,
                    rs: RsUnits::default(),
                }) {
                    held.push(grant);
                }
            }
        }
        let bind = (!is_restore).then(|| {
            let p = planner.clone();
            let taker = pool.clone();
            crate::rt::spawn(async move {
                if walk_rise {
                    match p
                        .fund_rise(head, run * pieces, || async { taker.take_chunks(run * pieces) })
                        .await
                    {
                        Ok(Some(false)) => Err("served with its chunks not free".to_string()),
                        Ok(_) => Ok(()),
                        Err(error) => Err(error.to_string()),
                    }
                } else if is_rise {
                    p.acquire_rise(head, head, run * pieces)
                        .await
                        .map(drop)
                        .map_err(|error| error.to_string())
                } else {
                    p.wait_bind_room(head, run, pieces, 0)
                        .await
                        .map(drop)
                        .map_err(|error| error.to_string())
                }
            })
        });
        // The restore's gate, as its process waits to be resident.
        let gate = is_restore.then(|| {
            let p = planner.clone();
            crate::rt::spawn(async move { p.wait_resident(head).await })
        });
        // No waiter is refused unless its need could never fit the pool
        // alone (none here: every need is at most six chunks of eight or
        // more): no bind or gate ends in an error, and no allocation's
        // outcome is one.
        let refused = |planner: &Arc<ResidencyPlanner>| -> Option<String> {
            planner.with_inner(|inner| {
                inner.queue.values().find_map(|waiter| match &waiter.kind {
                    // The starvation endgame (a stuck holder restarted or
                    // failed, no wait's give-up) meets a pool this one
                    // cannot stand for: it counts free chunks as free pages
                    // it never hands out (`reserve_device` is `None`).
                    WaitKind::Allocation {
                        outcome: Some(Err(PlannerError::Starved { .. })),
                        ..
                    } => None,
                    WaitKind::Allocation {
                        outcome: Some(Err(error)),
                        ..
                    } => Some(format!("an allocation refused: {error}")),
                    _ => None,
                })
            })
        };
        // Progress is the head served (its restore sent, sent again, or its
        // bind's runs given), never a move alone: an eviction and restore
        // cycle with no serve is no progress.
        let progress = |planner: &Arc<ResidencyPlanner>, pool: &BindRoomPool| {
            planner.stats.restores.load(Ordering::Relaxed)
                + u64::from(pool.compactions.load(Ordering::Relaxed))
                + u64::from(
                    planner.with_inner(|inner| inner.procs[&head].roles_free_retry.is_some()),
                )
        };
        // Time steps: at each, an event (or none), then quiescence, then
        // the property; the schedule is kept to print.
        let mut schedule = vec![format!(
            "total={total} host={host} run={run} pieces={pieces} restore={is_restore} \
             understated_at={understated_at:?} runs_now={runs_now} residents={} grants={grants} \
             compactable={} mixed={mixed} fragmented={fragmented} rise={is_rise} \
             walk_rise={walk_rise} kv={:?} \
             held={:?}",
            residents.len(),
            pool.compactable.load(Ordering::Acquire),
            pool.kv.lock(),
            held.iter().map(AllocationGrant::remaining_kv).collect::<Vec<_>>()
        )];
        let mut applied = 0u32;
        let mut alive = residents.clone();
        for step in 0..8 {
            // This step's event was a grant dropped unused: only the drop
            // tells the planner.
            let mut dropped_unused = false;
            if step > 0 && !held.is_empty() && rng.below(2) == 0 {
                let grant = held.remove(rng.below(held.len() as u64) as usize);
                let pages = grant.remaining_kv();
                drop(grant);
                dropped_unused = true;
                schedule.push(format!("a grant of {pages} dropped unused"));
            } else if step > 0 {
                match rng.below(6) {
                    // A chunk taken: fragmentation.
                    0 => {
                        let mut free = pool.free.lock();
                        if !free.is_empty() {
                            let at = rng.below(free.len() as u64) as usize;
                            // In a rise seed a live resident takes it, or
                            // no one does.
                            let by = alive.iter().copied().find(|pid| {
                                planner.with_inner(|inner| {
                                    inner.procs.get(pid).is_some_and(|proc| proc.state == Residency::Resident)
                                })
                            });
                            match (is_rise, by) {
                                (true, None) => {}
                                (rise, by) => {
                                    let id = free.remove(at);
                                    if let (true, Some(pid)) = (rise, by) {
                                        pool.kv.lock().push((id, pid));
                                    }
                                    schedule.push(format!("take {id}"));
                                }
                            }
                        }
                    }
                    // A chunk given back.
                    1 => {
                        let held: Vec<u32> =
                            (0..total).filter(|id| !pool.free.lock().contains(id)).collect();
                        if !held.is_empty() {
                            let id = held[rng.below(held.len() as u64) as usize];
                            pool.free.lock().push(id);
                            pool.gives.fetch_add(1, Ordering::Relaxed);
                            schedule.push(format!("give {id}"));
                        }
                    }
                    // An asked compaction carried out: the free chunks moved
                    // to the lowest ids, a run standing where they suffice.
                    2 if pool.compactions.load(Ordering::Relaxed) > applied => {
                        applied = pool.compactions.load(Ordering::Relaxed);
                        let mut free = pool.free.lock();
                        let count = free.len() as u32;
                        free.clear();
                        free.extend(0..count);
                        pool.gives.fetch_add(1, Ordering::Relaxed);
                        schedule.push("compaction carried out".into());
                    }
                    // An arrival: a new process admitted, waiting on a grant.
                    4 => {
                        let pid = ProcessId::new_v4();
                        planner.register(pid);
                        planner.note_admitted(pid);
                        planner.with_inner(|inner| {
                            inner.enqueue(
                                pid,
                                WaitKind::Allocation {
                                    demand: Demand {
                                        kv_pages: 1,
                                        rs: RsUnits::default(),
                                    },
                                    notify: Arc::new(Notify::new()),
                                    outcome: None,
                                    yielded: false,
                                    rise: false,
                                },
                            );
                        });
                        alive.push(pid);
                        schedule.push("an arrival".into());
                    }
                    // A resident completes.
                    3 if !alive.is_empty() => {
                        let pid = alive.remove(rng.below(alive.len() as u64) as usize);
                        // Its sets go with it: their chunks come back as it
                        // leaves, not after (a real departure's pages are
                        // counted on their way back).
                        let gone: Vec<u32> = pool
                            .kv
                            .lock()
                            .iter()
                            .filter(|&&(_, holder)| holder == pid)
                            .map(|&(chunk, _)| chunk)
                            .collect();
                        pool.kv.lock().retain(|&(_, holder)| holder != pid);
                        pool.free.lock().extend(&gone);
                        planner.unregister(pid);
                        schedule.push("a resident completes".into());
                    }
                    // A fire takes a grant of free chunks, or drops one
                    // it holds unused.
                    _ if is_rise && !walk_rise => {
                        let free = pool.free.lock().len() as u64;
                        if !held.is_empty() && (free == 0 || rng.below(2) == 0) {
                            let grant = held.remove(rng.below(held.len() as u64) as usize);
                            let pages = grant.remaining_kv();
                            drop(grant);
                            dropped_unused = true;
                            schedule.push(format!("a grant of {pages} dropped unused"));
                        } else if free > 0 {
                            let pages = 1 + rng.below(free.min(3)) as u32;
                            if let Some(grant) = planner.try_reserve(Demand {
                                kv_pages: pages,
                                rs: RsUnits::default(),
                            }) {
                                held.push(grant);
                                schedule.push(format!("a grant of {pages} taken"));
                            }
                        }
                    }
                    _ => schedule.push("nothing".into()),
                }
            }
            let before = progress(&planner, &pool)
                + u64::from(bind.as_ref().is_some_and(|bind| bind.is_finished()));
            let compactions_before = pool.compactions.load(Ordering::Relaxed);
            if !dropped_unused {
                planner.pages_freed();
                planner.poke();
            }
            let finished = || {
                u64::from(bind.as_ref().is_some_and(|bind| bind.is_finished())) * 2
                    + u64::from(gate.as_ref().is_some_and(|gate| gate.is_finished()))
            };
            if !quiesce(&planner, &pool, finished).await {
                failures.push(format!(
                    "seed {seed} step {step}: the planner never stood still\n  schedule: {}",
                    schedule.join(" -> ")
                ));
                break;
            }
            // The residents' moves land: their kv chunks come back free,
            // and the plan looks again, until no more land.
            for _ in 0..8 {
                let landed = pool.land_moves(&planner);
                if landed == 0 {
                    break;
                }
                schedule.push(format!("{landed} kv chunks land free"));
                planner.pages_freed();
                planner.poke();
                quiesce(&planner, &pool, finished).await;
            }
            if let Some(why) = refused(&planner) {
                failures.push(format!(
                    "seed {seed} step {step}: {why}\n  schedule: {}",
                    schedule.join(" -> ")
                ));
                break;
            }
            quiesce(&planner, &pool, finished).await;
            if gate.as_ref().is_some_and(|gate| gate.is_finished())
                && planner.stats.restores.load(Ordering::Relaxed) == 0
            {
                failures.push(format!(
                    "seed {seed} step {step}: the restore's gate ended unserved\n  schedule: {}",
                    schedule.join(" -> ")
                ));
                break;
            }
            let served_bind = bind.as_ref().is_some_and(|bind| bind.is_finished());
            let made = progress(&planner, &pool) + u64::from(served_bind);
            let head_served = planner.stats.restores.load(Ordering::Relaxed) > 0 || served_bind;
            if head_served {
                break;
            }
            let stands = if is_rise {
                // Before its take: a walk's raise takes them as it is served.
                pool.free.lock().len() >= (run * pieces) as usize
            } else {
                pool.guest_room_free(run, pieces)
            };
            let retried = planner.with_inner(|inner| inner.procs[&head].roles_free_retry.is_some());
            let at = planner.with_inner(|inner| {
                let proc = &inner.procs[&head];
                proc.roles_free_short.map(|_| proc.roles_free_short_at)
            });
            let servable = stands && at.is_none_or(|at| pool.guest_free_runs(run) > at);
            let compactable = false;
            // Mixed shapes: where the map serves the head by moving the
            // residents whose kv holds its chunks (or with no one moved),
            // and nothing is in flight, the planner is never silent: the
            // head was served, or a move is on its way.
            let need = planner
                .with_inner(|inner| inner.procs[&head].restore_need())
                .unwrap_or((run, pieces));
            let alive_now: Vec<ProcessId> = planner.with_inner(|inner| inner.procs.keys().copied().collect());
            let map_serves = if walk_rise {
                // Any chunks, free or emptied by moving their occupants:
                // the chunk map serves it as a bind's runs of one.
                pool.chunk_occupants(&alive_now).cheapest_runs(1, run * pieces).is_some()
                    || pool.free.lock().len() >= (run * pieces) as usize
            } else if is_rise {
                // Any free chunks. A rise is an allocation: its victims are
                // chosen by the registry's page quotes, which this pool's
                // residents' kv is not in (it is in the chunk map only), so
                // their chunks count as held here, as room the head waits
                // for with no deadline, never as room it is refused for.
                pool.free.lock().len() >= (run * pieces) as usize
            } else {
                pool.chunk_occupants(&alive_now).cheapest_runs(need.0, need.1).is_some()
                    || pool.runs_free(need.0, need.1)
            };
            let quiet = planner.with_inner(|inner| inner.nothing_in_flight());
            // A compaction is asked only where no eviction forms the head's
            // runs: moving chunks is then the only way; asked while moving
            // residents off the device stood them, it waited on what an
            // eviction would have made.
            let asked = pool.compactions.load(Ordering::Relaxed) > compactions_before;
            if asked && mixed && map_serves && !is_rise {
                failures.push(format!(
                    "seed {seed} step {step}: a compaction was asked while evictions stood the \
                     head's runs\n  schedule: {}",
                    schedule.join(" -> ")
                ));
                break;
            }
            if mixed && map_serves && quiet && !retried {
                failures.push(format!(
                    "seed {seed} step {step}: the map serves the head's {need:?} and the planner \
                     is silent\n  schedule: {}",
                    schedule.join(" -> ")
                ));
                break;
            }
            // Every chunk free or a live resident's (none held by no one, as
            // in a real pool): the head fits the pool alone, so unserved
            // with nothing in flight is a planner bug, never a wait.
            // Each chunk counted once: a give here may free a chunk a
            // resident's kv still names.
            let owned = {
                let mut chunks: HashSet<u32> = pool.free.lock().iter().copied().collect();
                chunks.extend(
                    pool.kv
                        .lock()
                        .iter()
                        .filter(|(_, pid)| alive_now.contains(pid))
                        .map(|&(chunk, _)| chunk),
                );
                chunks.extend(pool.parked.lock().iter().map(|&(chunk, _)| chunk));
                chunks.len() == total as usize
            };
            if owned && quiet && !retried && !is_rise {
                failures.push(format!(
                    "seed {seed} step {step}: every chunk is free or a resident's, the head \
                     fits alone, and nothing is in flight\n  schedule: {}",
                    schedule.join(" -> ")
                ));
                break;
            }
            // Unserved with nothing in flight: the stall is named.
            if quiet && planner.stats.stall_dumps.load(Ordering::Relaxed) == 0 {
                failures.push(format!(
                    "seed {seed} step {step}: the head is unserved with nothing in flight and \
                     no stall was named\n  schedule: {}",
                    schedule.join(" -> ")
                ));
                break;
            }
            // Wake latency: room that stands at a step's quiescence is the
            // head's within that step (the event that made it, a give, a
            // landing, a completion, wakes the plan at once): one step from
            // room appearing to the head served, never a later one.
            if dropped_unused && servable && !retried && !head_served && !is_restore {
                failures.push(format!(
                    "seed {seed} step {step}: a grant dropped unused gave back the head's room \
                     and the head was not reconsidered within the step\n  schedule: {}",
                    schedule.join(" -> ")
                ));
                break;
            }
            if servable && !retried && !head_served && !is_restore {
                failures.push(format!(
                    "seed {seed} step {step}: the head's room stood and it was not served \
                     within the step\n  schedule: {}",
                    schedule.join(" -> ")
                ));
                break;
            }
            if (servable || compactable) && made == before && !retried && made == 0 {
                failures.push(format!(
                    "seed {seed} step {step}: servable={servable} compactable={compactable}: no \
                     progress\n  schedule: {}",
                    schedule.join(" -> ")
                ));
                break;
            }
            // Quiescent: nothing more happens without an event.
            let still = (
                progress(&planner, &pool),
                planner.stats.compaction_waits.load(Ordering::Relaxed),
                planner.stats.grow_waits.load(Ordering::Relaxed),
            );
            let stood_still = quiesce(&planner, &pool, finished).await;
            let after = (
                progress(&planner, &pool),
                planner.stats.compaction_waits.load(Ordering::Relaxed),
                planner.stats.grow_waits.load(Ordering::Relaxed),
            );
            if still != after || !stood_still {
                failures.push(format!(
                    "seed {seed} step {step}: a quiescent planner kept stepping: {still:?} -> \
                     {after:?}\n  schedule: {}",
                    schedule.join(" -> ")
                ));
                break;
            }
        }
        // No ping-pong: each process is moved at most once for the head
        // (none here comes back to be moved again).
        let processes = residents.len() as u64 + alive.len() as u64 + 1;
        let moves = planner.stats.evictions_started.load(Ordering::Relaxed);
        if moves > processes + schedule.iter().filter(|e| e.as_str() == "an arrival").count() as u64 {
            failures.push(format!(
                "seed {seed}: {moves} moves for {processes} processes: a move cycle\n  schedule: {}",
                schedule.join(" -> ")
            ));
        }
        // A bind or gate that ended did so served, never refused.
        if let Some(bind) = bind {
            if bind.is_finished() {
                match bind.await {
                    Ok(Err(error)) => {
                        failures.push(format!(
                            "seed {seed}: the head's bind was refused: {error}\n  schedule: {}",
                            schedule.join(" -> ")
                        ));
                    }
                    _ => {}
                }
            } else {
                bind.abort();
            }
        }
        if let Some(gate) = gate {
            if gate.is_finished() {
                if let Ok(Err(error)) = gate.await {
                    failures.push(format!("seed {seed}: the head's restore was refused: {error}"));
                }
            } else {
                gate.abort();
            }
        }
        if failures.len() > 20 {
            break;
        }
    }
    assert!(failures.is_empty(), "{} failing seeds:\n{}", failures.len(), failures.join("\n"));
}

/// A compaction is asked only where no eviction forms the head's runs: a
/// bind short of a run among scattered free chunks, their gaps held by a
/// resident it may move, is funded by the map, the occupant moved, and no
/// compaction is asked first, however the pool could compact.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_short_of_a_run_is_funded_by_the_map_never_waiting_on_a_compaction() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    pool.free.lock().extend([0, 2, 4, 6]);
    pool.compactable.store(true, Ordering::Release);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (binder, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [binder, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    // Idle: a candidate whatever its rank against the binder, so its
    // chunks are emptiable by evicting it, and no compaction is the way.
    planner.with_inner(|inner| inner.procs.get_mut(&resident).unwrap().idle = true);
    *pool.occupant.lock() = Some(resident);
    let p = planner.clone();
    let bind = crate::rt::spawn(async move { p.wait_bind_room(binder, 2, 1, 0).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the chunks' occupant is moved for it"
    );
    // Here the move finds nothing to copy (no store backs the occupant):
    // its chunks stay, and only then is a compaction the way. Never before
    // the eviction was tried.
    let compacted = pool.compactions.load(Ordering::Relaxed) > 0;
    let found_nothing = planner.with_inner(|inner| inner.prepare_blocked.contains(&resident));
    assert!(
        !compacted || found_nothing,
        "a compaction only once evicting the occupant moved nothing"
    );
    bind.abort();
}

/// A restore whose unpark keeps refusing while the map shows runs of its
/// need standing (the planner's view and the unpark's disagree): re-sent on
/// every plan pass it was a hot loop. Its refusal, seen against the map,
/// marks it understated and the census names the mismatch (map's runs
/// against the refusal's), so the need gets fixed rather than the retry;
/// it is re-sent at most once per change of the map. A plain short head
/// (refused with the map short too) goes again once more runs of its need's
/// length stand than its refusal saw, and only then.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_refused_against_the_map_goes_again_at_most_once_per_map_change() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(12, 0..0, 0)
    });
    // The map shows two runs of 3 standing.
    pool.free.lock().extend([0, 1, 2, 6, 7, 8]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    let refuse = |planner: &Arc<ResidencyPlanner>, refused: Option<u32>| {
        planner.with_inner(|inner| {
            inner.procs.get_mut(&head).unwrap().state = Residency::Restoring;
        });
        planner.requeue_restore_roles(head, (3, 1), refused);
    };
    // Refused with its own view short (free_runs 0) while the map shows 2:
    // a room the map does not read. Its need is not grown on it (Mac Pro on
    // 3b45462d4: (7, 6) to (7, 22)); the census names the mismatch.
    refuse(&planner, Some(0));
    let (understated, need, mismatch) = planner.with_inner(|inner| {
        let proc = &inner.procs[&head];
        (
            proc.roles_free_short,
            proc.roles_short.map(|need| (need.run, need.pieces)),
            proc.roles_short_mismatch,
        )
    });
    assert_eq!(understated, None, "not understated on runs only the map shows");
    assert_eq!(need, Some((3, 1)), "its need as refused, not grown");
    assert_eq!(mismatch, Some((2, Some(0))), "the census names the mismatch");
    let retried = |planner: &Arc<ResidencyPlanner>| {
        planner.with_inner(|inner| inner.procs[&head].roles_free_retry.is_some())
            || planner.stats.restores.load(Ordering::Relaxed) > 0
    };
    // Plan passes on the same map: not re-sent.
    for _ in 0..5 {
        planner.poke();
        for _ in 0..50 {
            crate::rt::yield_now().await;
        }
    }
    // Refused with nothing else in flight, its placement standing and no
    // move to make: sent once on this room (the progress rule), never
    // again on it.
    assert!(retried(&planner), "one re-send on the same map, nothing to move");
    // The map changes (more runs stand): one re-send.
    pool.free.lock().extend([3, 4, 5]);
    pool.gives.fetch_add(1, Ordering::Relaxed);
    planner.pages_freed();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if retried(&planner) {
            break;
        }
    }
    assert!(retried(&planner), "the map moved: sent once more");

    // A plain short head (the map short too at its refusal) waits until
    // more runs of its need's length stand than its refusal saw: room given
    // back that leaves them as they were sends nothing; a run come free
    // sends it.
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(12, 0..0, 0)
    });
    pool.free.lock().extend([0, 2, 4]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    planner.register(head);
    planner.note_admitted(head);
    refuse(&planner, Some(0));
    assert_eq!(planner.with_inner(|inner| inner.procs[&head].roles_free_short), None);
    // Room given back elsewhere, no run of three: not re-sent.
    pool.gives.fetch_add(3, Ordering::Relaxed);
    planner.pages_freed();
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    assert!(!retried(&planner), "no more usable room: not re-sent");
    // A run of three stands now: sent.
    pool.free.lock().extend([1]);
    planner.pages_freed();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if retried(&planner) {
            break;
        }
    }
    assert!(retried(&planner), "a run come free: sent");
}

/// mx-2's cycle at the planner: a process suspended for its bind's room is
/// restored only once that room stands (the map shows its run beside its
/// own), not on room for its kv alone; while no run of its need stands it
/// stays parked, no one moved and no restore sent, however often the plan
/// runs; the run appears and it is restored, once.
#[tokio::test(flavor = "current_thread")]
async fn a_process_suspended_for_its_bind_s_room_is_restored_only_once_the_room_stands() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    // Its own stores: its restore runs against them, whatever another test
    // registered.
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    // Four chunks free, no three adjacent.
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&pid).unwrap();
        proc.state = Residency::Evicted;
        proc.bind_room = Some(super::proc::PendingBind {
            singles: 0,
            run: 3,
            pieces: 1,
            pages: 3,
        });
        inner.enqueue(
            pid,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    for _ in 0..50 {
        planner.pages_freed();
        planner.poke();
        for _ in 0..10 {
            crate::rt::yield_now().await;
        }
    }
    assert_eq!(planner.stats.restores.load(Ordering::Relaxed), 0, "no run of its need: not restored");
    assert_eq!(planner.stats.evictions_started.load(Ordering::Relaxed), 0, "no one moved, no cycle");
    // The run appears.
    pool.free.lock().push(1);
    pool.gives.fetch_add(1, Ordering::Relaxed);
    planner.pages_freed();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.restores.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert_eq!(planner.stats.restores.load(Ordering::Relaxed), 1, "its room stands: restored once");
}

/// mx-7's state (a restore waiting for room for its roles, recorded
/// bind_room (3,1) beside a refused roles_short (4,1), free 4-runs standing
/// in the map, every other chunk emptiable by residents, and nothing ever
/// moved, evictions frozen): the refusal's need becomes the head's need
/// beside the bind's, each length apart, and, refused while runs of it
/// stood, one run more than stood; the planner plans against that and
/// moves the occupants of the cheapest such runs. No wait for a map change that never comes.
#[tokio::test(flavor = "current_thread")]
async fn a_refused_restore_s_need_is_its_refusal_s_and_the_planner_moves_for_it() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(16, 0..0, 64)
    });
    // Two free runs of 4 (0..4 and 8..12): the map shows them.
    pool.free.lock().extend([0, 1, 2, 3, 8, 9, 10, 11]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    *pool.occupant.lock() = Some(resident);
    planner.with_inner(|inner| {
        inner.procs.get_mut(&resident).unwrap().idle = true;
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Restoring;
        proc.bind_room = Some(super::proc::PendingBind {
            singles: 0,
            run: 3,
            pieces: 1,
            pages: 3,
        });
    });
    // Its unpark is refused short of (4,1), having seen two runs of 4.
    planner.requeue_restore_roles(head, (4, 1), Some(2));
    let (roles_short, bind_room) = planner.with_inner(|inner| {
        let proc = &inner.procs[&head];
        (
            proc.roles_short.map(|need| (need.run, need.pieces)),
            proc.bind_room.map(|need| (need.run, need.pieces)),
        )
    });
    assert_eq!(roles_short, Some((4, 3)), "its refusal's run, one more piece than stood");
    assert_eq!(bind_room, Some((3, 1)), "the bind's runs keep their own length");
    let key = planner
        .with_inner(|inner| inner.queue.iter().find(|(_, w)| w.pid == head).map(|(k, _)| *k))
        .expect("queued again");
    assert_eq!(
        planner.head_room(key),
        Some(super::pool::RunNeed::of([(4, 3), (3, 1)])),
        "the head's need: each length apart, never runs of three read as runs of four"
    );
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the occupants of the cheapest runs of the updated need are moved"
    );
}

/// A rise (a cut fire's raise funded as a head) whose room an older
/// resident holds that no move frees here (its pages are in no quote this
/// pool makes, so no victim stands): the starvation endgame never refuses
/// it, its need fitting the pool alone; and the resident, holding device
/// room the registry's counts do not see (`holds_device_room`), is no
/// sole holder's "nobody". It waits, named as a stall, and is served once
/// the resident leaves and its chunks come back.
#[tokio::test(flavor = "current_thread")]
async fn a_rise_whose_room_no_move_frees_waits_and_is_served_once_the_room_returns() {
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 0)
    });
    pool.pages.store(true, Ordering::Release);
    let (resident, head) = (ProcessId::new_v4(), ProcessId::new_v4());
    pool.free.lock().extend([6, 7]);
    pool.kv.lock().extend((0..6).map(|chunk| (chunk, resident)));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    for pid in [resident, head] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| inner.procs.get_mut(&resident).unwrap().idle = true);
    assert!(pool.holds_device_room(resident));
    let p = planner.clone();
    let rise = crate::rt::spawn(async move { p.acquire_rise(head, head, 4).await.map(drop) });
    for _ in 0..3 {
        planner.pages_freed();
        planner.poke();
        for _ in 0..200 {
            crate::rt::yield_now().await;
        }
    }
    assert!(!rise.is_finished(), "it waits, never refused: it fits the pool alone");
    assert!(planner.stats.stall_dumps.load(Ordering::Relaxed) > 0, "the stall is named");
    // The resident leaves: its chunks come back, and the rise is served.
    pool.kv.lock().clear();
    pool.free.lock().extend(0..6);
    planner.unregister(resident);
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if rise.is_finished() {
            break;
        }
    }
    assert!(rise.is_finished(), "served once the room returns");
    assert!(matches!(rise.await, Ok(Ok(()))));
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

/// An allocation's grant dropped unused gives its pages back to the pool,
/// and the planner hears of it from the drop alone: a head that room can
/// serve is served within one planner step, with no completion, no map
/// change and no `pages_freed` (the gap the no-deadline wait exposed, a
/// head asleep until an unrelated event). With the drain task armed and
/// without it.
#[tokio::test(flavor = "current_thread")]
async fn a_grant_dropped_unused_wakes_the_head_its_room_serves() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    for drain in [true, false] {
        let (planner, grant, head_task) = head_behind_a_held_grant(drain).await;
        let epoch = planner.returns.epoch();
        let since = std::time::Instant::now();
        drop(grant);
        assert_eq!(planner.returns.epoch(), epoch + 1, "the drop moves the epoch");
        served_at_once(head_task, since).await;
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

/// A move of parked-in-place state down a tier that failed is tried again
/// once the pool's room has moved past the failure. Latched for good, one
/// failure left every such set on the device: the head's runs held by an
/// evicted process's parked state, the chunk map naming it each pass, and
/// nothing issued (no move, no eviction), with room the head fits alone in.
#[tokio::test(flavor = "current_thread")]
async fn a_failed_park_down_is_tried_again_once_room_moves() {
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 0)
    });
    pool.disk_free.store(64, Ordering::Relaxed);
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, sleeper) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, sleeper] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        inner.procs.get_mut(&sleeper).unwrap().state = Residency::Evicted;
    });
    // The evicted process's state parked in place between the free chunks:
    // no run of two stands until it moves down.
    pool.parked.lock().extend([(1, sleeper), (3, sleeper), (5, sleeper), (7, sleeper)]);
    pool.park_down_fails.store(1, Ordering::SeqCst);
    let p = planner.clone();
    let bind = crate::rt::spawn(async move { p.wait_bind_room(head, 2, 1, 0).await.map(drop) });
    for _ in 0..200 {
        crate::rt::yield_now().await;
    }
    assert_eq!(pool.park_down_fails.load(Ordering::SeqCst), 0, "the first move failed");
    assert!(!bind.is_finished(), "its room is still held");
    // Room moves (a chunk given back elsewhere): the move goes again.
    pool.gives.fetch_add(1, Ordering::Relaxed);
    planner.pages_freed();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if bind.is_finished() {
            break;
        }
    }
    assert!(bind.is_finished(), "the parked state moved down and the head was served");
    assert!(bind.await.unwrap().is_ok());
    assert_eq!(*pool.moved.lock(), vec!["disk"]);
}

/// A disk tier with no slot free but idle cache in it is room for a
/// suspend (it drops the cache to make its room, `make_swap_room`): the
/// head short of a run its occupant holds is funded, the occupant moved.
/// Counted as full, the planner returned with nothing issued (no swap room)
/// while the head fit the pool alone.
#[tokio::test(flavor = "current_thread")]
async fn a_disk_tier_full_of_idle_cache_is_room_for_a_victim() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 0));
    pool.free.lock().extend([0, 2, 4, 6]);
    pool.disk_idle.store(64, Ordering::Relaxed);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (binder, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [binder, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    *pool.occupant.lock() = Some(resident);
    let p = planner.clone();
    let bind = crate::rt::spawn(async move { p.wait_bind_room(binder, 2, 1, 0).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the occupant is moved, the disk's idle cache its room"
    );
    bind.abort();
}

/// Parked state lying on the device whose process the planner no longer
/// knows (it left while parked) moves down a tier for a head as an evicted
/// process's does: passed over (only evicted processes' moved), it held the
/// head's runs for good with the chunk map naming it each pass.
#[tokio::test(flavor = "current_thread")]
async fn parked_state_of_a_process_gone_while_parked_moves_down_for_the_head() {
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 0)
    });
    pool.disk_free.store(64, Ordering::Relaxed);
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    // Never registered here: a process gone while parked.
    let gone = ProcessId::new_v4();
    pool.parked.lock().extend([(1, gone), (3, gone), (5, gone), (7, gone)]);
    let p = planner.clone();
    let bind = crate::rt::spawn(async move { p.wait_bind_room(head, 2, 1, 0).await.map(drop) });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if bind.is_finished() {
            break;
        }
    }
    assert!(bind.is_finished(), "its parked state moved down and the head was served");
    assert!(bind.await.unwrap().is_ok());
    assert_eq!(*pool.moved.lock(), vec!["disk"]);
}

/// A swap tier nearly full (a tight disk budget) defers the eviction rung
/// only while something in flight can bring the head its pages. With no
/// fire, move or kill under way, and residents neither firing nor parked
/// in the queue (blocked elsewhere: on another process, a sibling lane),
/// the rung goes on: before, the fleet read as not stalled by its waiters'
/// kinds and the head was deferred for good (the disk-only M1 hang).
#[tokio::test(flavor = "current_thread")]
async fn a_nearly_full_swap_defers_eviction_only_while_something_is_in_flight() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(RacePool {
        host: 4,
        host_total: 64,
        ..RacePool::new(4)
    });
    let planner = Arc::new(ResidencyPlanner::new(pool.clone()));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    // Admitted, not idle, not queued, not firing: blocked outside the planner.
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
    assert_eq!(planner.diagnostics().queue.len(), 1, "the ask parks");
    assert!(!planner.is_wedged(), "its waiters' kinds read no stall");
    assert!(planner.with_inner(|inner| inner.nothing_in_flight()), "nothing is in flight");
    planner.plan();
    assert_eq!(
        planner.stats.eviction_deferrals.load(Ordering::Relaxed),
        0,
        "nothing in flight brings supply: the rung goes on"
    );
    parked.abort();
}

/// A restore a bind suspended is served by the chunk map as its census
/// reads it: the bind's runs beside those its own unpark takes back first.
/// Free runs at 0-1 and 4-5, its own retake taking 0-1, the bind asking two
/// runs of two: the census finds one beside its own, short; the map asked
/// for the bind's two alone saw both free, found no one to move, and the
/// restore waited with no step (restore heads with runs open, no step).
/// Asked for three, the map moves the resident holding 2-3 or 6-7.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_suspended_restore_is_funded_beside_its_own_retake() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    pool.free.lock().extend([0, 1, 4, 5]);
    pool.own.lock().extend([0, 1]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    pool.kv.lock().extend([(2, resident), (3, resident), (6, resident), (7, resident)]);
    planner.with_inner(|inner| {
        inner.procs.get_mut(&resident).unwrap().idle = true;
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.bind_room = Some(super::proc::PendingBind {
            singles: 0,
            run: 2,
            pieces: 2,
            pages: 4,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    assert!(pool.guest_room_free(2, 2), "the map sees two runs free");
    assert!(!pool.room_beside_own(head, 2, 2), "the census: one beside its own retake");
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the resident holding a run beside them is moved for it"
    );
    // Its move lands, its chunks free: the restore is served on the step's
    // real need, its bind's runs standing beside its own retake.
    for _ in 0..8 {
        if pool.land_moves(&planner) == 0 && planner.stats.restores.load(Ordering::Relaxed) > 0 {
            break;
        }
        planner.pages_freed();
        planner.poke();
        for _ in 0..200 {
            crate::rt::yield_now().await;
        }
    }
    assert!(pool.room_beside_own(head, 2, 2), "its runs stand beside its own retake");
    assert!(
        planner.stats.restores.load(Ordering::Relaxed) > 0,
        "the restore is served once they stand"
    );
}

/// A victim blocked when its move found no swap room (or nothing to move)
/// goes back among the candidates once the pool has moved past the block:
/// never on the same pool twice, and never held out for good. Cleared only
/// by a restore or a departure, the block outlived the pool it was set on,
/// and a stall in which neither came kept the resident out of every victim
/// set, the head waiting with no one to move.
#[tokio::test(flavor = "current_thread")]
async fn a_blocked_victim_is_a_candidate_again_once_the_pool_moves() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 0));
    pool.disk_free.store(4, Ordering::Relaxed);
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let pool_now = planner.pool_now();
    planner.with_inner(|inner| {
        inner.procs.get_mut(&resident).unwrap().idle = true;
        inner.enqueue(
            head,
            WaitKind::BindRoom {
                run: 2,
                pieces: 1,
                singles: 0,
                pages: 2,
                notify: Arc::new(Notify::new()),
                served: false,
                withdrawn: false,
                rise: None,
            },
        );
        inner.host_swap_blocked.insert(resident);
        inner.blocked_at = Some(pool_now);
    });
    let members = |planner: &ResidencyPlanner| {
        planner
            .victim_set()
            .map(|set| set.members.iter().map(|v| v.pid).collect::<Vec<_>>())
            .unwrap_or_default()
    };
    assert!(members(&planner).is_empty(), "blocked on the pool as it stands");
    // Room given back elsewhere: the pool has moved past the block.
    pool.gives.fetch_add(1, Ordering::Relaxed);
    assert_eq!(members(&planner), vec![resident], "a candidate again");
    // Blocked again on the new pool, it stays out until that moves too.
    let pool_now = planner.pool_now();
    planner.with_inner(|inner| {
        inner.host_swap_blocked.insert(resident);
        inner.blocked_at = Some(pool_now);
    });
    assert!(members(&planner).is_empty(), "never on the same pool twice");
    pool.disk_free.fetch_add(1, Ordering::Relaxed);
    assert_eq!(members(&planner), vec![resident], "swap slots freed: a candidate again");
}

/// Where no eviction can form the head's runs (scattered free chunks, their
/// gaps held by what no candidate moves) and nothing is in flight, moving
/// chunks is the only way: a compaction is asked and the head waits on it.
/// Out of every head's path, such a head waited for good with the pool able
/// to stand its runs.
#[tokio::test(flavor = "current_thread")]
async fn a_head_no_eviction_can_fund_asks_a_compaction() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    pool.free.lock().extend([0, 2, 4, 6]);
    pool.compactable.store(true, Ordering::Release);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let binder = ProcessId::new_v4();
    planner.register(binder);
    planner.note_admitted(binder);
    // The gaps are held by no candidate (no occupant the planner may move).
    let p = planner.clone();
    let bind = crate::rt::spawn(async move { p.wait_bind_room(binder, 2, 1, 0).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if pool.compactions.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(pool.compactions.load(Ordering::Relaxed) > 0, "a compaction is asked");
    assert!(planner.stats.compaction_waits.load(Ordering::Relaxed) > 0, "and waited on");
    assert_eq!(planner.stats.evictions_started.load(Ordering::Relaxed), 0, "no one to evict");
    bind.abort();
}

/// The anti-thrash rule, on the L40S loop (a process evicted for one head,
/// restored ~43 ms later, re-evicted ~4 ms after that, every ~65 ms, the
/// head never served; one pid evicted 443 times). A victim evicted for
/// head H is not restored while H waits unserved: its restore, ranked
/// ahead of H, is passed over and H is the head the planner funds. Once H
/// is served the victim's restore goes; restored for a waiting head, it is
/// no eviction candidate for the next head until it has fired once, and is
/// one again after.
#[test]
fn a_victim_is_not_restored_before_its_head_nor_evicted_again_before_it_fires() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::with_swap(8, 64))));
    let (victim, head, next) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    // The victim older than the head: its restore ranks ahead of the
    // head's ask, as in the loop.
    for pid in [victim, head, next] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let demand = Demand {
        kv_pages: 2,
        rs: RsUnits::default(),
    };
    let head_key = planner.with_inner(|inner| {
        let head_key = inner.enqueue(head, allocation(demand));
        // Evicted for the head (as `commit_evictions` records it), its
        // restore queued at once.
        let proc = inner.procs.get_mut(&victim).unwrap();
        proc.state = Residency::Evicted;
        proc.evicted_for = Some(head);
        let restore_key = inner.enqueue(
            victim,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
        assert!(restore_key < head_key, "the victim's restore ranks ahead of the head");
        head_key
    });
    let unmet = |planner: &ResidencyPlanner| {
        planner.with_inner(|inner| inner.unmet_head().map(|(_, waiter)| waiter.pid))
    };
    assert_eq!(unmet(&planner), Some(head), "the head is funded, its victim's restore passed over");

    // The head is served: its victim's restore goes.
    planner.with_inner(|inner| {
        inner.queue.remove(&head_key);
    });
    assert_eq!(unmet(&planner), Some(victim), "served, the head no longer holds it");

    // Restored for a waiting head: no candidate for the next head until it
    // fires once.
    planner.with_inner(|inner| {
        inner.queue.retain(|_, waiter| waiter.pid != victim);
        inner.procs.get_mut(&victim).unwrap().state = Residency::Restoring;
    });
    planner.report_restored(victim);
    planner.with_inner(|inner| {
        inner.enqueue(next, allocation(demand));
        let proc = inner.procs.get_mut(&victim).unwrap();
        proc.idle = true;
        proc.progressed = true;
    });
    let members = |planner: &ResidencyPlanner| {
        planner
            .victim_set()
            .map(|set| set.members.iter().map(|v| v.pid).collect::<Vec<_>>())
            .unwrap_or_default()
    };
    // Something in flight (another's fire on its way): the victim can run.
    let running = planner.fire_in_flight(head);
    assert!(
        !members(&planner).contains(&victim),
        "restored for a waiting head and able to run: not moved back before it runs"
    );
    // Queued for its first fire's room while something is in flight: the
    // ask is part of that fire, so the mark still holds it (else it went
    // back down before it ran: the L40S at 3 GiB, evictions 4017, restores
    // 4012).
    let own = planner.with_inner(|inner| inner.enqueue(victim, allocation(demand)));
    assert!(
        planner.with_inner(|inner| inner.unfired_protects(victim)),
        "queued for its first fire's room beside a fire in flight: still held"
    );
    planner.with_inner(|inner| {
        inner.queue.remove(&own);
    });
    // Nothing in flight anywhere: it still fires as soon as its lanes
    // wake, so the mark still holds it (moved back first, it went round
    // with the head it made room for: seed 4139); a head no move serves
    // meanwhile is passed over, no stall.
    drop(running);
    assert!(
        !members(&planner).contains(&victim),
        "nothing in flight: still able to run, not moved back before it runs"
    );
    // Waiting itself with nothing in flight (its lanes blocked on the
    // planner): it runs nothing, so the mark does not hold it, and no stall
    // comes of it.
    let own = planner.with_inner(|inner| inner.enqueue(victim, allocation(demand)));
    assert!(
        planner.with_inner(|inner| !inner.unfired_protects(victim)),
        "blocked on the head's progress with nothing in flight: the mark does not hold it"
    );
    planner.with_inner(|inner| {
        inner.queue.remove(&own);
    });
    // Able to run again, then fired once: a candidate for good.
    let running = planner.fire_in_flight(head);
    assert!(!members(&planner).contains(&victim));
    drop(planner.fire_in_flight(victim));
    assert!(members(&planner).contains(&victim), "fired once: a candidate again");
    drop(running);
}

/// Mac Pro's restore heads (need (9, 2), five free runs, nothing issued):
/// a restore refused short of its roles names the part it came up short
/// on, and its whole unpark takes more. Its need is the whole unpark: here
/// one run of two recorded, its own parts two such runs, one free run
/// standing, refused on this very map. Asked the recorded run, the map saw
/// it free, moved no one, and the restore never went again; asked the
/// whole, the resident holding a run beside it is moved.
#[tokio::test(flavor = "current_thread")]
async fn a_roles_short_restore_is_funded_for_its_whole_unpark() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    pool.free.lock().extend([0, 1, 4]);
    // Its own parts: two runs of two.
    pool.own.lock().extend([20, 21, 22, 23]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    pool.kv.lock().extend([(2, resident), (3, resident), (5, resident), (6, resident), (7, resident)]);
    let gives = pool.room_gives();
    planner.with_inner(|inner| {
        inner.procs.get_mut(&resident).unwrap().idle = true;
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(super::proc::PendingBind {
            singles: 0,
            run: 2,
            pieces: 1,
            pages: 2,
        });
        proc.roles_short_gives = Some(gives);
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    assert!(pool.guest_room_free(2, 1), "the recorded run stands");
    assert!(!pool.guest_room_free(2, 2), "the whole unpark's two do not");
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the resident is moved for the whole unpark"
    );
}

/// The M1's disk-only stall (a plan in the map, nothing issued, no evict,
/// restore or compaction line): with the swap tiers full of parked kv, a
/// head short of chunk runs is still funded, its occupant moved, for the
/// chunks empty as guest scratch is given back and state moves, which a
/// full swap does not stop. Only kv the head itself lacks waits on swap.
#[tokio::test(flavor = "current_thread")]
async fn a_head_short_of_runs_is_funded_with_the_swap_full() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    // No host tier and no free disk slot: the swap is full.
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 0));
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (binder, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [binder, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| inner.procs.get_mut(&resident).unwrap().idle = true);
    *pool.occupant.lock() = Some(resident);
    assert_eq!(planner.swap_stats().0, 0, "the swap is full");
    let p = planner.clone();
    let bind = crate::rt::spawn(async move { p.wait_bind_room(binder, 2, 1, 0).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the occupant is moved though no swap slot is free"
    );
    bind.abort();
}

/// The M1's first stall (a restore with no recorded need, free room,
/// nothing issued) and the L40S's guest-only restore (kv 0, rs 0): a
/// restore asking nothing past what stands is served from it at once.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_asking_nothing_is_served_from_free_room() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    pool.free.lock().extend(0..8);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| {
        inner.procs.get_mut(&head).unwrap().state = Residency::Evicted;
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.restores.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(planner.stats.restores.load(Ordering::Relaxed) > 0, "served at once");
}

/// The M1's self-eviction (a bind short of room picked its own process
/// first, evicted it, and did again on its return, three times over): a
/// head's own process is never among its victims, though waiting at its
/// bind it yields as its victims do.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_never_picks_its_own_process() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let binder = ProcessId::new_v4();
    planner.register(binder);
    planner.note_admitted(binder);
    planner.with_inner(|inner| inner.procs.get_mut(&binder).unwrap().bind_waiting = true);
    // Its own state holds the chunks between the free ones.
    *pool.occupant.lock() = Some(binder);
    let p = planner.clone();
    let bind = crate::rt::spawn(async move { p.wait_bind_room(binder, 2, 1, 0).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
    }
    assert_eq!(planner.stats.evictions_started.load(Ordering::Relaxed), 0, "itself no victim");
    assert!(
        planner.with_inner(|inner| inner.procs[&binder].state == Residency::Resident),
        "its process stays"
    );
    bind.abort();
}

/// The M1's 4-run restore that never found a window (3,710 passes, picks
/// empty, a map of two stuck chunks): with nothing in flight, residents
/// that made no progress since their last move are candidates too, nothing
/// coming to move them on. Read as the fresh alone, the map held none of
/// their chunks and the head waited with 371 of them emptiable.
#[tokio::test(flavor = "current_thread")]
async fn with_nothing_in_flight_a_head_is_funded_by_residents_that_made_no_progress() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (binder, a, b) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [binder, a, b] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    // Neither idle nor progressed: not "fresh".
    planner.with_inner(|inner| {
        for pid in [a, b] {
            inner.procs.get_mut(&pid).unwrap().progressed = false;
        }
    });
    pool.kv.lock().extend([(1, a), (3, a), (5, b), (7, b)]);
    assert!(planner.with_inner(|inner| inner.nothing_in_flight()));
    let p = planner.clone();
    let bind = crate::rt::spawn(async move { p.wait_bind_room(binder, 2, 1, 0).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "a resident holding a run beside the free chunks is moved"
    );
    bind.abort();
}

/// Room that comes back by any path wakes a parked waiter it serves within
/// a turn: the pool's room signal, told wherever device slots come free (a
/// sequence ending, a fire's epoch retiring, an eviction's pages let go),
/// with no pages_freed and no timer. Without it the head slept until an
/// unrelated event (the L40S at 4 GiB: silent waits to 793 ms once the
/// 100 ms re-plan was gone). With the drain task armed and without.
#[tokio::test(flavor = "current_thread")]
async fn room_that_comes_free_on_any_path_wakes_the_waiter_it_serves() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    for drain in [true, false] {
        let pool = Arc::new(RacePool::new(4));
        pool.refill_after_stall(4, 0);
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
        if drain {
            planner.arm_drain_task();
        }
        let (holder, head) = (ProcessId::new_v4(), ProcessId::new_v4());
        for pid in [holder, head] {
            planner.register(pid);
            planner.note_admitted(pid);
        }
        // A holder takes the pool; its pages never come back by the grant
        // (the grant is leaked), only by the pool's own return below.
        let held = planner
            .grant_now(
                holder,
                Demand {
                    kv_pages: 4,
                    rs: RsUnits::default(),
                },
            )
            .expect("the pool is free");
        std::mem::forget(held);
        // A newcomer whose first ask does not fit now is refused at
        // admission; asked again as an admitted lane's growth, it waits.
        refused_at_admission_then_seated(&planner, head, 2);
        let p = planner.clone();
        let task = crate::rt::spawn(async move {
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
        let since = std::time::Instant::now();
        pool.free_quietly(2);
        for _ in 0..50 {
            if task.is_finished() {
                break;
            }
            crate::rt::yield_now().await;
        }
        assert!(task.is_finished(), "woken by the room coming free (drain {drain})");
        assert!(since.elapsed() < crate::policy::pressure::AGAIN, "within a turn, no tick");
        assert!(matches!(task.await.unwrap(), Ok(super::acquire::Acquired::Granted(_))));
    }
}


/// The M1 hang (head d293aade: 398 chunks held, two of them the map's, its
/// own 126 instances' tables and its channels parked in place on the
/// device): an evicted head's own parked state is moved down a tier for
/// its runs. Nothing runs on it, and its restore lays it back compactly.
/// Left out of the map and of park-down ("never its own state"), it held
/// the chunks between the free ones, no run of its bind's length stood,
/// and nothing moved it. Here its own state lies at 1, 3, 5 and 7 between
/// free chunks; it goes to disk, a run stands, and the restore is served.
#[tokio::test(flavor = "current_thread")]
async fn an_evicted_head_s_own_parked_state_moves_down_for_its_runs() {
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 0)
    });
    pool.disk_free.store(64, Ordering::Relaxed);
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    pool.parked.lock().extend([(1, head), (3, head), (5, head), (7, head)]);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.bind_room = Some(super::proc::PendingBind {
            singles: 0,
            run: 2,
            pieces: 1,
            pages: 2,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    assert!(!pool.guest_room_free(2, 1), "no run of two stands");
    planner.poke();
    for _ in 0..8 {
        for _ in 0..200 {
            crate::rt::yield_now().await;
        }
        if planner.stats.restores.load(Ordering::Relaxed) > 0 {
            break;
        }
        planner.pages_freed();
        planner.poke();
    }
    assert_eq!(*pool.moved.lock(), vec!["disk"], "its own parked state went down");
    assert!(pool.guest_room_free(2, 1), "a run stands");
    assert!(planner.stats.restores.load(Ordering::Relaxed) > 0, "the restore is served");
}

/// The L40S livelock on 3d177c213 (one victim evicted 5,894 times for one
/// restore head, the head never served): a restore head leaves the queue
/// while its restore runs and is queued again when its unpark comes up
/// short. Its victim's restore is held through that gap too: the head is
/// served only once it is resident, not once its waiter has left.
#[test]
fn a_victim_s_restore_is_held_while_its_restore_head_is_away_from_the_queue() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::with_swap(8, 64))));
    let (head, victim) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, victim] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        // The head's restore was served: off the queue, restoring.
        inner.procs.get_mut(&head).unwrap().state = Residency::Restoring;
        let proc = inner.procs.get_mut(&victim).unwrap();
        proc.state = Residency::Evicted;
        proc.evicted_for = Some(head);
        inner.enqueue(
            victim,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    let unmet = |planner: &ResidencyPlanner| {
        planner.with_inner(|inner| inner.unmet_head().map(|(_, waiter)| waiter.pid))
    };
    assert_eq!(unmet(&planner), None, "held while its head's restore runs");
    // Short again: the head queued again; still held.
    planner.with_inner(|inner| {
        inner.procs.get_mut(&head).unwrap().state = Residency::Evicted;
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    assert_eq!(unmet(&planner), Some(head), "the head is funded first");
    // The head is back: served, the victim's restore goes.
    planner.with_inner(|inner| {
        inner.queue.retain(|_, waiter| waiter.pid != head);
        inner.procs.get_mut(&head).unwrap().state = Residency::Resident;
    });
    assert_eq!(unmet(&planner), Some(victim), "its head served, it goes");
}

/// Mac Pro's "free" stalls (a restore of (3, 3) and 41 kv pages, 8 runs
/// open): a restore's step gathers its kv pages, then lays its runs, both
/// from the same free chunks. Its need is both: here one run of two and
/// two kv pages, one run and one more chunk standing. Asked the run alone,
/// it stood, no one moved, and gathering the pages broke it; asked both,
/// the resident holding a chunk beside them is moved.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_is_funded_for_its_runs_and_its_kv_pages_together() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    pool.free.lock().extend([0, 1, 4]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    pool.kv.lock().extend([(2, resident), (3, resident), (5, resident), (6, resident), (7, resident)]);
    let key = planner.with_inner(|inner| {
        inner.procs.get_mut(&resident).unwrap().idle = true;
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(super::proc::PendingBind {
            singles: 0,
            run: 2,
            pieces: 1,
            pages: 2,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand {
                    kv_pages: 2,
                    rs: RsUnits::default(),
                },
            },
        )
    });
    // Its run, and its two pages' chunks beside it as single chunks (not
    // a run of their own): four chunks against three free.
    let mut need = super::pool::RunNeed::one(2, 1);
    need.add(1, 2);
    assert_eq!(planner.head_room(key), Some(need), "its run and its pages' chunks");
    assert!(pool.guest_room_free(2, 1), "the run alone stands");
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the resident is moved for the run and the pages together"
    );
}

/// With its runs and its kv pages all free, a restore is granted at once.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_with_its_runs_and_kv_pages_free_is_granted() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    pool.pages.store(true, Ordering::Release);
    pool.free.lock().extend(0..8);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand {
                    kv_pages: 2,
                    rs: RsUnits::default(),
                },
            },
        );
    });
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.restores.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(planner.stats.restores.load(Ordering::Relaxed) > 0, "granted at once");
}

/// The L40S at 4 GiB on 3d177c213 (+23%: rows per dispatch 14 to 10.67):
/// woken on every free, the planner admitted what fit at that instant and
/// split fires into smaller batches. Room coming free wakes it only once
/// what the head lacks is met: frees trickling in short of the head's need
/// plan nothing; the one that meets it wakes the planner within a turn.
#[tokio::test(flavor = "current_thread")]
async fn trickling_frees_short_of_the_head_s_need_plan_nothing() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(RacePool::new(4));
    pool.refill_after_stall(4, 0);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    planner.arm_drain_task();
    let (holder, head) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [holder, head] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let held = planner
        .grant_now(
            holder,
            Demand {
                kv_pages: 4,
                rs: RsUnits::default(),
            },
        )
        .expect("the pool is free");
    std::mem::forget(held);
    // A newcomer whose first ask does not fit now is refused at
    // admission; asked again as an admitted lane's growth, it waits.
    refused_at_admission_then_seated(&planner, head, 3);
    let p = planner.clone();
    let task = crate::rt::spawn(async move {
        p.acquire(
            head,
            head,
            Demand {
                kv_pages: 3,
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
    assert_eq!(planner.diagnostics().queue.len(), 1, "the head parks, short of 3");
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    let passes = planner.stats.plan_passes.load(Ordering::Relaxed);
    // Two pages, one at a time: short of the head's three.
    pool.free_quietly(1);
    pool.free_quietly(1);
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    assert_eq!(
        planner.stats.plan_passes.load(Ordering::Relaxed),
        passes,
        "frees short of its need plan nothing"
    );
    assert!(!task.is_finished());
    // The third meets it.
    pool.free_quietly(1);
    for _ in 0..50 {
        if task.is_finished() {
            break;
        }
        crate::rt::yield_now().await;
    }
    assert!(task.is_finished(), "the free that meets its need wakes it within a turn");
    assert!(matches!(task.await.unwrap(), Ok(super::acquire::Acquired::Granted(_))));
}

/// Mac Pro's 5 GiB stall (a restore needing runs of four, 19 open runs in
/// the map, the unpark refused with no free chunk and nothing committable):
/// the state budget's headroom was spent, so no free chunk could be opened
/// however many stood open. A head short of runs there is funded by
/// emptying chunks elsewhere, giving budget back, not by the open runs.
/// Here two runs of four stand free, the headroom is none, and a resident
/// holds four chunks elsewhere: it is moved.
#[tokio::test(flavor = "current_thread")]
async fn a_head_short_of_runs_with_no_headroom_frees_budget_elsewhere() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(12, 0..0, 64));
    pool.free.lock().extend(0..8);
    *pool.headroom.lock() = Some(0);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (binder, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [binder, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| inner.procs.get_mut(&resident).unwrap().idle = true);
    pool.kv.lock().extend([(8, resident), (9, resident), (10, resident), (11, resident)]);
    assert!(!pool.guest_room_free(4, 1), "runs stand open, the budget opens none");
    let p = planner.clone();
    let bind = crate::rt::spawn(async move { p.wait_bind_room(binder, 4, 1, 0).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the resident is moved to give budget back"
    );
    bind.abort();
}

/// The L40S on 81aa5f879 (8 of 9 opened chunks held by no candidate and
/// nothing parked, the head a restore of no kv, nothing in flight): pages
/// gathered for another head stay gathered while a restore that needs none
/// of them waits, holding the chunks (and the state budget) its runs need.
/// They go back: here four of eight chunks gathered, a restore of no kv
/// needing six single chunks; released, its runs stand and it is served.
#[tokio::test(flavor = "current_thread")]
async fn pages_gathered_for_another_head_go_back_for_a_restore_that_needs_none() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    pool.pages.store(true, Ordering::Release);
    pool.free.lock().extend(0..8);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    // Gathered for a head that went: four chunks held as pages.
    assert!(planner.absorb_free(4));
    assert_eq!(pool.free.lock().len(), 4);
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(super::proc::PendingBind {
            singles: 0,
            run: 1,
            pieces: 6,
            pages: 6,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.restores.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert_eq!(planner.with_inner(|inner| inner.accum.len()), 0, "the pages went back");
    assert!(planner.stats.restores.load(Ordering::Relaxed) > 0, "the restore is served");
}

/// Mac Pro's no-step restores with no refusal on record: one bind-suspended
/// restore at a time, the next waiting until the first fires. A first that
/// cannot fire (nothing in flight) held every other for good, their runs
/// standing. By state, the gate lets the next go: here the first never
/// fires, and the second, its room standing, is restored.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_suspended_restore_that_cannot_fire_holds_the_next_no_longer() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    pool.free.lock().extend(0..8);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (first, second) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [first, second] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        // The first was restored for its bind and never fired.
        inner.bind_restoring = Some(first);
        let proc = inner.procs.get_mut(&second).unwrap();
        proc.state = Residency::Evicted;
        proc.bind_room = Some(super::proc::PendingBind {
            singles: 0,
            run: 1,
            pieces: 1,
            pages: 1,
        });
        inner.enqueue(
            second,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    assert!(planner.with_inner(|inner| inner.nothing_in_flight()), "the first cannot fire");
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.restores.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(planner.stats.restores.load(Ordering::Relaxed) > 0, "the second is restored");
}

/// A real shared store for [`bench_a_plan_pass_on_a_shared_store`]: 480
/// chunks of 1 MiB, a kv list (16 slots a chunk), a one-piece state record
/// and the role shapes (instance, channel, adapter), 64 host chunks.
fn bench_store_model() -> (usize, crate::store::roles::RolesHandle) {
    use ::engine::chunk::{
        ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId, SlotKind, SlotRole,
        SlotShape,
    };
    // `PIE_BENCH_FCGATE=1`: the L40S's fcgate pair, a 6 GiB pool of CUDA's
    // ~17 MiB chunks, nine kv pages a chunk.
    let fcgate = std::env::var_os("PIE_BENCH_FCGATE").is_some();
    let (chunk, total, page) = if fcgate {
        (9 * (1980u64 << 10), 361u32, 1980u64 << 10)
    } else {
        (1u64 << 20, 480, 64 << 10)
    };
    let shape = |id: u32, kind: SlotKind, slot_bytes: u64, role: SlotRole| SlotShape {
        id: ShapeId(id),
        kind,
        slot_bytes,
        slots_per_chunk: u32::try_from(chunk / slot_bytes).unwrap(),
        sets_per_working_set: 1,
        rows: Vec::new(),
        pinned: None,
        role,
    };
    let list = SlotKind::List {
        tokens: 16,
        rate: 1,
        window: None,
    };
    let fixed = SlotKind::Fixed { pieces: 1 };
    let facts = ChunkFacts {
        chunk_bytes: chunk,
        total_chunks: total,
        segment_chunks: total,
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
        shapes: vec![
            shape(0, list, page, SlotRole::State),
            shape(1, fixed, 256 << 10, SlotRole::State),
            shape(2, fixed, 4 << 10, SlotRole::Instance),
            shape(3, fixed, 1 << 10, SlotRole::Channel),
            shape(4, fixed, 64 << 10, SlotRole::Adapter),
        ],
        act_chunks: vec![(16, 1), (256, 4)],
        host_chunks: 64,
        ..ChunkFacts::default()
    };
    facts.validate().unwrap();
    let device = Arc::new(std::sync::Mutex::new(
        crate::store::shared::SharedPool::new(&facts, 256).unwrap(),
    ));
    let host = Arc::new(std::sync::Mutex::new(
        crate::store::shared::SharedPool::tier(&facts, facts.host_chunks)
            .unwrap()
            .unwrap(),
    ));
    let model = crate::store::registry::register_model_with_chunks(
        16,
        &[crate::store::registry::EngineStores {
            pools: crate::store::chunks::StorePools::of(&facts).expect("shared facts"),
            max_context: 256,
            shared: Some(device),
            tiers: crate::store::shared::Tiers {
                host: Some(host),
                disk_slots: 0,
                disk_chunk_slots: 0,
            },
        }],
    );
    let roles = crate::store::registry::get(model, 0).roles.expect("roles");
    (model, roles)
}

/// A plan pass against a real shared store (the `RegistryPool` through the
/// shared pool, the roles and the kv store) in the state of
/// `bench_a_plan_pass_with_nothing_to_do`: 250 residents each bound to an
/// instance and holding kv pages interleaved over every chunk, the device
/// near full; an evicted head with parked roles short of runs (two of
/// four), and 100 allocation waiters behind it. Run with `--ignored
/// --nocapture`. `PIE_BENCH_STALLED=1` leaves nothing in flight (the head
/// funded: the stalled path, dump_stall's whole census); by default one
/// resident is firing, as a decoding fleet holds the head unfunded.
#[test]
#[ignore]
fn bench_a_plan_pass_on_a_shared_store() {
    use crate::inferlet::process::residency::{ProcessResidency, register_residency};
    use crate::store::roles::tests::{FakeEngine, PROGRAM};
    use crate::store::roles::{bind_instance, learn_program};

    // Moves a stalled pass issues are spawned and never polled.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let _entered = runtime.enter();
    let stalled = std::env::var_os("PIE_BENCH_STALLED").is_some();
    let rounds: u32 = std::env::var("PIE_BENCH_ROUNDS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(2_000);
    let (model, roles) = bench_store_model();
    let stores = crate::store::registry::get(model, 0);
    let pool = RegistryPool::new(model, 0, false);
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RegistryPool::new(
        model, 0, false,
    ))));

    // 250 residents and the head, each bound to an instance (the fcgate
    // pair: 8 beam-4 processes and 64 decodes, 72).
    let count = if std::env::var_os("PIE_BENCH_FCGATE").is_some() { 72 } else { 250 };
    let residents: Vec<ProcessId> = (0..count).map(|_| ProcessId::new_v4()).collect();
    let head = ProcessId::new_v4();
    let mut engine = FakeEngine::default();
    learn_program(&roles, &engine, PROGRAM).unwrap();
    for &pid in residents.iter().chain([&head]) {
        let binding = ::engine::InstanceBinding {
            program: PROGRAM,
            ..::engine::InstanceBinding::default()
        };
        bind_instance(Some(&roles), &mut engine, &binding, Some(pid), |_| false).unwrap();
    }
    // The head's roles parked (on the host): its restore unparks them.
    let park = roles.lock().unwrap().prepare_park(head).expect("it parks");
    roles.lock().unwrap().commit(park);

    // Each resident's kv working set, one page a round across every
    // resident, so each chunk holds sixteen residents' pages; filled until
    // a few chunks stand free.
    const MOST: u64 = 64;
    let mut residencies = Vec::new();
    let mut sets = Vec::new();
    for &pid in &residents {
        let ws = crate::store::registry::with_kv_lock(&stores.kv, "bench", |kv| {
            let ws = kv.create_working_set();
            kv.reserve(ws, MOST).unwrap();
            ws
        });
        let handle = crate::store::kv::working_set::KvWorkingSet::new(model, 0, ws, 16);
        let residency = Arc::new(std::sync::Mutex::new(ProcessResidency::default()));
        residency
            .lock()
            .unwrap()
            .kv_working_sets
            .insert((model, 0, ws), handle.suspend_handle());
        register_residency(pid, Arc::downgrade(&residency));
        residencies.push(residency);
        sets.push((ws, handle));
    }
    let leave_free: u32 = std::env::var("PIE_BENCH_FREE")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(1);
    let mut written = 0u64;
    'fill: for round in 0..MOST {
        for (ws, _) in &sets {
            if pool.free_chunks() <= leave_free {
                break 'fill;
            }
            let done = crate::store::registry::with_kv_lock(&stores.kv, "bench", |kv| {
                let need = kv.write_demand(*ws, &[round]).unwrap();
                let Some(mut granted) = kv.reserve_device_pages(need) else {
                    return false;
                };
                let prepared = kv.prepare_write_reserved(*ws, &[round], &mut granted).unwrap();
                let seq = kv
                    .publish_prepared(
                        prepared,
                        &[crate::store::kv::write::PageCommit {
                            token_hashes: Vec::new(),
                            page_hash: None,
                        }],
                    )
                    .unwrap();
                kv.settle(seq);
                true
            });
            if !done {
                break 'fill;
            }
            written += 1;
        }
    }

    for &pid in residents.iter().chain([&head]) {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let key = planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(PendingBind {
            singles: 0,
            run: 4,
            pieces: 2,
            pages: 8,
        });
        // `PIE_BENCH_BIND=1`: a bind of two runs of four suspended it too,
        // so its placement (the bind's runs beside its unpark) does not
        // stand on the one free chunk.
        if std::env::var_os("PIE_BENCH_BIND").is_some() {
            proc.bind_room = proc.roles_short;
        }
        let key = inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
        for &pid in residents.iter().take(100) {
            inner.enqueue(
                pid,
                allocation(Demand {
                    kv_pages: 1,
                    rs: RsUnits::default(),
                }),
            );
        }
        if !stalled {
            // A decoder's fire in flight, as long as the bench runs.
            let proc = inner.procs.get_mut(&residents[residents.len() - 1]).unwrap();
            proc.fires = 1;
            proc.fired_at = crate::rt::Instant::now() + Duration::from_secs(3600);
        }
        key
    });
    let (free, total) = pool.device_stats();
    eprintln!(
        "BENCH store: pages written={written} device pages free/total={free}/{total} \
         free_chunks={} free runs of 4={} restore_stands={} stalled={stalled}",
        pool.free_chunks(),
        pool.guest_free_runs(4),
        pool.restore_stands(head, &super::pool::RunNeed::one(4, 2), RsUnits::default(), 0, &[]),
    );

    let first = std::time::Instant::now();
    planner.plan();
    eprintln!(
        "BENCH first plan pass: {:.1} us; head unmet after={:?}",
        first.elapsed().as_secs_f64() * 1e6,
        planner.with_inner(|inner| inner.unmet_head().map(|(key, _)| key)),
    );
    let passes_then = planner.pass_census();
    let phases_then = planner.phase_census();
    // Each round a pass, as on an event: the settled generation cleared.
    let began = std::time::Instant::now();
    for _ in 0..rounds {
        *planner.settled_at.lock() = None;
        planner.plan();
    }
    let per_pass = began.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    let woke = std::time::Instant::now();
    for _ in 0..rounds {
        planner.plan();
    }
    eprintln!(
        "BENCH wake on a settled generation: {:.1} us; skipped={}",
        woke.elapsed().as_secs_f64() * 1e6 / f64::from(rounds),
        planner.stats.skipped_wakes.load(Ordering::Relaxed),
    );
    let passes = planner.pass_census();
    let phases = planner.phase_census();
    eprintln!(
        "BENCH plan pass (shared store): {per_pass:.1} us; pass census {:?} -> {:?}; \
         phases us/pass steps={:.1} wants={:.1} rest={:.1}",
        passes_then,
        passes,
        (phases[0] - phases_then[0]) as f64 / f64::from(rounds),
        (phases[1] - phases_then[1]) as f64 / f64::from(rounds),
        (phases[2] - phases_then[2]) as f64 / f64::from(rounds),
    );
    eprintln!(
        "BENCH queue after: head unmet={:?} queued={} head state={:?} evicting={} diag={:?}",
        planner.with_inner(|inner| inner.unmet_head().map(|(key, _)| key)),
        planner.with_inner(|inner| inner.queue.len()),
        planner.with_inner(|inner| inner.procs[&head].state),
        planner.with_inner(|inner| inner.evicting.len()),
        planner.stats.restores.load(Ordering::Relaxed),
    );

    let time = |name: &str, f: &dyn Fn()| {
        let began = std::time::Instant::now();
        for _ in 0..rounds {
            f();
        }
        eprintln!(
            "BENCH {name}: {:.1} us",
            began.elapsed().as_secs_f64() * 1e6 / f64::from(rounds)
        );
    };
    time("unmet_head", &|| {
        planner.with_inner(|inner| {
            let _ = inner.unmet_head().map(|(key, _)| key);
        })
    });
    time("head_room", &|| {
        *planner.need_cache.lock() = None;
        let _ = planner.head_room(key);
    });
    time("publish_wants", &|| planner.publish_wants());
    time("wants_met_now", &|| {
        let _ = planner.wants_met_now();
    });
    time("dump_stall (as a pass finds it)", &|| planner.dump_stall());
    time("dump_stall (forced: said again each call)", &|| {
        planner.with_inner(|inner| inner.stall_dumped = None);
        planner.dump_stall();
    });
    time("report_short", &|| planner.report_short());
    time("plan_once", &|| {
        let _ = planner.plan_once();
    });
    time("victim_set", &|| {
        let _ = planner.victim_set();
    });
    time("held_by_state", &|| {
        let _ = planner.held_by_state();
    });
    time("store_holders", &|| {
        let _ = planner.store_holders();
    });
    // The port's calls, on the same store.
    time("port.chunk_occupants(&[])", &|| {
        let _ = pool.chunk_occupants(&[]);
    });
    time("port.chunk_occupants(residents)", &|| {
        let _ = pool.chunk_occupants(&residents);
    });
    // Inside chunk_occupants, part by part.
    time("  kv_working_sets_for(residents)", &|| {
        let _ = crate::inferlet::process::residency::kv_working_sets_for(&residents, model, 0);
    });
    let groups: Vec<HashSet<_>> = crate::inferlet::process::residency::kv_working_sets_for(&residents, model, 0)
        .into_iter()
        .flatten()
        .collect();
    time("  kv.movable_resident_pages_each(residents)", &|| {
        crate::store::registry::with_kv_lock(&stores.kv, "bench", |kv| {
            let _ = kv.movable_resident_pages_each(&groups);
        });
    });
    time("  roles.guest_chunk_slots(residents)", &|| {
        let _ = roles.lock().unwrap().guest_chunk_slots(&residents);
    });
    time("  rs private refs (residents)", &|| {
        for &pid in &residents {
            let sets = crate::inferlet::process::residency::rs_working_set_ids(pid, model, 0);
            if !sets.is_empty() {
                let _ = stores.rs.lock().unwrap().private_slot_refs(&sets);
            }
        }
    });
    time("planner.occupants(residents) (cached by generation)", &|| {
        let _ = planner.occupants(&residents);
    });
    time("planner.generation", &|| {
        let _ = planner.generation();
    });
    time("port.restore_stands", &|| {
        let _ = pool.restore_stands(head, &super::pool::RunNeed::one(4, 2), RsUnits::default(), 0, &[]);
    });
    time("port.restore_stands(bind 4x2)", &|| {
        let _ = pool.restore_stands(head, &super::pool::RunNeed::one(4, 2), RsUnits::default(), 0, &[(4, 2)]);
    });
    time("port.chunk_census(4)", &|| {
        let _ = pool.chunk_census(4);
    });
    time("port.unpark_census", &|| {
        let _ = pool.unpark_census(head, Some((4, 2)));
    });
    time("port.unpark_need", &|| {
        let _ = pool.unpark_need(head);
    });
    time("port.device_stats", &|| {
        let _ = pool.device_stats();
    });
    time("port.free_chunks", &|| {
        let _ = pool.free_chunks();
    });
    time("port.guest_free_runs(4)", &|| {
        let _ = pool.guest_free_runs(4);
    });
    time("port.room_gives", &|| {
        let _ = pool.room_gives();
    });
    time("port.rs_stats", &|| {
        let _ = pool.rs_stats();
    });
    time("port.state_chunks", &|| {
        let _ = pool.state_chunks();
    });
    // Inside chunk_occupants: one resident's movable pages (its own sets'
    // locations, then every other running set's pages scanned for sharing).
    let one: HashSet<_> = [sets[0].0].into_iter().collect();
    time("kv.movable_resident_pages(one resident)", &|| {
        crate::store::registry::with_kv_lock(&stores.kv, "bench", |kv| {
            let _ = kv.movable_resident_pages_each(std::slice::from_ref(&one));
        });
    });
    time("kv_working_sets_for(residents)", &|| {
        let _ = crate::inferlet::process::residency::kv_working_sets_for(&residents, model, 0);
    });
    drop(residencies);
}

/// Mac Pro's 2 GiB spin and the L40S's (139,315 and 132,792 refusals, the
/// free room no different): refused restores sent one another again, each
/// refusal's own return of what its unpark took counted as room moved. A
/// refused restore goes again on runs only once more runs of its need's
/// length stand than its refusal saw: give-backs that leave those runs as
/// they were send no one; a run come free does.
#[tokio::test(flavor = "current_thread")]
async fn a_refused_restore_goes_again_only_on_more_usable_runs() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    pool.free.lock().extend([0, 4]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| {
        inner.procs.get_mut(&head).unwrap().state = Residency::Evicted;
    });
    // Refused short of a run of two, none standing; queued again.
    planner.requeue_restore_roles(head, (2, 1), Some(0));
    planner.with_inner(|inner| {
        if !inner.queue.values().any(|waiter| waiter.pid == head) {
            inner.enqueue(
                head,
                WaitKind::Restore {
                    demand: Demand::default(),
                },
            );
        }
    });
    let sent = |planner: &ResidencyPlanner| {
        planner.stats.restores.load(Ordering::Relaxed) > 0
            || planner.with_inner(|inner| inner.procs[&head].roles_free_retry.is_some())
    };
    // Other refusals give back what their unparks took: the count of room
    // given back moves, the runs do not.
    pool.gives.fetch_add(5, Ordering::Relaxed);
    planner.plan();
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    assert!(!sent(&planner), "no more usable room: not sent again");
    // A run of two comes free: it goes again.
    pool.free.lock().push(1);
    pool.gives.fetch_add(1, Ordering::Relaxed);
    planner.plan();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if sent(&planner) {
            break;
        }
    }
    assert!(sent(&planner), "a run come free: sent again");
}

/// The L40S's spin on 5f41b43e5 (a stuck head, the planner passing sixty
/// times a second with nothing to issue): the policy's answers of nothing
/// freed woke a pass each, the pass said the shortfall again and drew the
/// next answer. A head with nothing to do now costs a bounded number of
/// passes however often the policy answers nothing.
#[tokio::test(flavor = "current_thread")]
async fn a_stuck_head_costs_a_bounded_number_of_passes() {
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    let pool = Arc::new(RacePool::new(4));
    pool.refill_after_stall(4, 0);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    planner.arm_drain_task();
    let (holder, head) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [holder, head] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let held = planner
        .grant_now(
            holder,
            Demand {
                kv_pages: 4,
                rs: RsUnits::default(),
            },
        )
        .expect("the pool is free");
    std::mem::forget(held);
    // A newcomer whose first ask does not fit now is refused at
    // admission; asked again as an admitted lane's growth, it waits.
    refused_at_admission_then_seated(&planner, head, 2);
    let p = planner.clone();
    let task = crate::rt::spawn(async move {
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
    }
    let passes = planner.stats.plan_passes.load(Ordering::Relaxed);
    for _ in 0..100 {
        for _ in 0..10 {
            crate::rt::yield_now().await;
        }
    }
    let more = planner.stats.plan_passes.load(Ordering::Relaxed) - passes;
    assert!(more <= 2, "a hundred answers of nothing freed planned {more} passes");
    assert!(!task.is_finished());
    task.abort();
}

/// Mac Pro's g2 spin (56,671 no-step lines, ~199 s of map passes in one
/// run): a stuck head re-ran every pass on every wake. Edge-triggered, a
/// stuck state is said once and costs no pass until an event moves the
/// generation; its wakes are counted, and the next event plans again.
#[tokio::test(flavor = "current_thread")]
async fn a_stuck_state_is_said_once_and_costs_no_pass_until_an_event() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 0));
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| {
        inner.enqueue(
            head,
            WaitKind::BindRoom {
                run: 2,
                pieces: 1,
                singles: 0,
                pages: 2,
                notify: Arc::new(Notify::new()),
                served: false,
                withdrawn: false,
                rise: None,
            },
        );
    });
    // Scattered free chunks, no one to move: stuck.
    planner.plan();
    planner.plan();
    let passes = planner.stats.plan_passes.load(Ordering::Relaxed);
    let dumps = planner.stats.stall_dumps.load(Ordering::Relaxed);
    let maps = planner.stats.map_passes.load(Ordering::Relaxed);
    for _ in 0..100 {
        planner.plan();
    }
    assert_eq!(planner.stats.plan_passes.load(Ordering::Relaxed), passes, "no pass on no event");
    assert_eq!(planner.stats.stall_dumps.load(Ordering::Relaxed), dumps, "said once");
    assert_eq!(planner.stats.map_passes.load(Ordering::Relaxed), maps, "no map built");
    assert!(planner.stats.skipped_wakes.load(Ordering::Relaxed) >= 100, "the wakes counted");
    // An event: room given back. The next wake plans, and serves.
    pool.free.lock().push(1);
    pool.gives.fetch_add(1, Ordering::Relaxed);
    planner.plan();
    assert_eq!(planner.stats.plan_passes.load(Ordering::Relaxed), passes + 1, "planned on the event");
}

/// The M1's allocation head with 38 free runs not granted: only the seated
/// lanes whose next step needs room now are held back for. A seated lane
/// waiting in the queue (blocked, asking for its room there) holds back
/// nothing, and with nothing in flight no lane grows before the claim.
#[test]
fn the_seated_lanes_reserve_counts_only_lanes_that_grow_now() {
    let planner = Arc::new(
        ResidencyPlanner::new(Arc::new(RacePool::new(8)) as Arc<dyn PoolPort>),
    );
    let claimant = ProcessId::new_v4();
    let lanes: Vec<ProcessId> = (0..10).map(|_| ProcessId::new_v4()).collect();
    for &pid in lanes.iter().chain([&claimant]) {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let units = |pages: u32| RsUnits {
        slots: 0,
        pages,
        ..RsUnits::default()
    };
    let decoding = planner.fire_in_flight(lanes[0]);
    planner.with_inner(|inner| {
        for pid in &lanes {
            inner.procs.get_mut(pid).unwrap().seated = true;
        }
        // Ten lanes growing two pages each, a fire in flight.
        let reserve = inner.seating_reserve(claimant, units(2), units(3), units(64));
        assert_eq!(reserve, units(20), "every running lane grows: {reserve:?}");
        // Eight of them wait in the queue: two grow before the claim.
        for &pid in &lanes[2..] {
            inner.enqueue(pid, allocation(units_demand(1)));
        }
        let reserve = inner.seating_reserve(claimant, units(2), units(3), units(64));
        assert_eq!(reserve, units(4), "the waiting lanes hold back nothing: {reserve:?}");
        // The M1: 38 free, a claim of 3, the waiting lanes' growth not kept.
        assert!(units(38).covers(units(3).add(reserve)), "the claim is granted");
    });
    drop(decoding);
    planner.with_inner(|inner| {
        assert!(inner.nothing_in_flight());
        assert!(
            inner
                .seating_reserve(claimant, units(2), units(3), units(64))
                .is_zero(),
            "nothing in flight: nothing held back"
        );
    });
}

fn units_demand(pages: u32) -> Demand {
    Demand {
        kv_pages: pages,
        rs: RsUnits::default(),
    }
}

/// Mac Pro's 5 GiB mx-3 spin (85,319 no-step lines, ~470 a second, nothing
/// in flight): a refused restore head short of a run while chunks enough
/// stood free. Its want of room was its need's chunk count, which the free
/// chunks already met, so every chunk given back anywhere (a resident's,
/// or the pass's own) woke a pass that found no step. Room coming free
/// wakes such a head only past what stood free when its pass ended.
#[tokio::test(flavor = "current_thread")]
async fn a_refused_restore_head_short_of_a_run_is_not_woken_by_chunks_it_already_had() {
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    // Four free chunks, no two adjacent.
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    planner.arm_drain_task();
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| {
        inner.procs.get_mut(&head).unwrap().state = Residency::Evicted;
    });
    planner.requeue_restore_roles(head, (2, 1), Some(0));
    planner.with_inner(|inner| {
        if !inner.queue.values().any(|waiter| waiter.pid == head) {
            inner.enqueue(
                head,
                WaitKind::Restore {
                    demand: Demand::default(),
                },
            );
        }
    });
    planner.returns.waiters.fetch_add(1, Ordering::AcqRel);
    planner.plan();
    for _ in 0..200 {
        crate::rt::yield_now().await;
    }
    let passes = planner.stats.plan_passes.load(Ordering::Relaxed);
    // Chunks given back and taken again elsewhere: four free each time.
    for _ in 0..100 {
        pool.gives.fetch_add(1, Ordering::Relaxed);
        planner.returns.slots_freed(crate::store::slots::RoomCount {
            tier: crate::store::slots::RoomTier::Device,
            chunks: 4,
            slots: 0,
        });
        for _ in 0..10 {
            crate::rt::yield_now().await;
        }
    }
    let more = planner.stats.plan_passes.load(Ordering::Relaxed) - passes;
    planner.returns.waiters.fetch_sub(1, Ordering::AcqRel);
    assert!(more <= 2, "a hundred give-backs of chunks it had planned {more} passes");
}

/// Mac Pro's 2 GiB stall: a restore head left unfunded while a fire was in
/// flight, then the fleet went quiet with no store telling the planner (one
/// stall line in three minutes, no eviction issued, no pass after). The
/// beat plans a quiet unserved head once on the room it finds, and never
/// again on the same room.
#[tokio::test(flavor = "current_thread")]
async fn the_fleet_going_quiet_plans_an_unserved_head_once() {
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    let pool = Arc::new(RacePool::new(4));
    pool.refill_after_stall(4, 0);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (resident, head) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [resident, head] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let held = planner
        .grant_now(
            resident,
            Demand {
                kv_pages: 4,
                rs: RsUnits::default(),
            },
        )
        .expect("the pool is free");
    planner.with_inner(|inner| {
        inner.procs.get_mut(&head).unwrap().state = Residency::Evicted;
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand {
                    kv_pages: 2,
                    rs: RsUnits::default(),
                },
            },
        );
        // A fire of the resident's in flight.
        let proc = inner.procs.get_mut(&resident).unwrap();
        proc.fires = 1;
        proc.fired_at = crate::rt::Instant::now();
    });
    planner.tick();
    let passes = planner.stats.plan_passes.load(Ordering::Relaxed);
    planner.tick();
    assert_eq!(
        planner.stats.plan_passes.load(Ordering::Relaxed),
        passes,
        "a fire in flight: the beat plans nothing"
    );
    // The fire lands with no one told: the fleet is quiet.
    planner.with_inner(|inner| inner.procs.get_mut(&resident).unwrap().fires = 0);
    planner.tick();
    let quiet = planner.stats.plan_passes.load(Ordering::Relaxed);
    assert_eq!(quiet, passes + 1, "quiet with the head unserved: planned once");
    planner.tick();
    planner.tick();
    assert_eq!(
        planner.stats.plan_passes.load(Ordering::Relaxed),
        quiet,
        "the same room plans nothing again"
    );
    drop(held);
}

/// Mac Pro's 2 GiB mix on c27717c48: a restore head suspended for a bind,
/// its need [(7, 1), (4, 1)] standing (open stretches of 19 and 11), 33 kv
/// pages to gather, never refused (why=None), and never attempted while a
/// fleet of 248 decoded. A head that arrives with its room standing is
/// attempted in its first pass, fires in flight or none.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_head_whose_need_stands_is_attempted_in_its_first_pass() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    for decoding in [true, false] {
        let pool = Arc::new(BindRoomPool::new(200, 0..0, 64));
        pool.pages.store(true, Ordering::Release);
        // Stretches of 19 and 11, and pages enough for its kv past them.
        pool.free.lock().extend((0..19).chain(30..41).chain(100..140));
        // Its own unpark lays four chunks.
        pool.own.lock().extend(150..154);
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
        let head = ProcessId::new_v4();
        planner.register(head);
        planner.note_admitted(head);
        // A decoding fleet larger than the pages free.
        let fleet: Vec<ProcessId> = (0..248).map(|_| ProcessId::new_v4()).collect();
        for &pid in &fleet {
            planner.register(pid);
            planner.note_admitted(pid);
        }
        let fires: Vec<_> = if decoding {
            fleet.iter().take(8).map(|&pid| planner.fire_in_flight(pid)).collect()
        } else {
            Vec::new()
        };
        planner.with_inner(|inner| {
            let proc = inner.procs.get_mut(&head).unwrap();
            proc.state = Residency::Evicted;
            proc.bind_room = Some(super::proc::PendingBind {
                singles: 0,
                run: 7,
                pieces: 1,
                pages: 7,
            });
            inner.enqueue(
                head,
                WaitKind::Restore {
                    demand: Demand {
                        kv_pages: 33,
                        rs: RsUnits::default(),
                    },
                },
            );
        });
        // One pass (no drain task: the poke plans at once).
        planner.poke();
        let state = planner.with_inner(|inner| inner.procs[&head].state);
        assert_ne!(
            state,
            Residency::Evicted,
            "decoding={decoding}: its need stands, it is attempted"
        );
        drop(fires);
    }
}

/// Mac Pro's 2 GiB mix on 0d96fb179: an allocation head, 0 chunks free,
/// one stall line, then silence while chunks came free (35 later). A
/// quiet unserved head is planned by the beat on the room it finds, and
/// again whenever the free counts differ from its last plan: room come
/// free with no one told serves it on the next beat.
#[test]
fn a_quiet_allocation_head_is_planned_again_when_room_comes_free() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let pool = Arc::new(RacePool::new(8));
    pool.refill_after_stall(4, 0);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (holder, head) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [holder, head] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    // A resident holds every page.
    let held = planner
        .grant_now(
            holder,
            Demand {
                kv_pages: 4,
                rs: RsUnits::default(),
            },
        )
        .expect("the pool is free");
    std::mem::forget(held);
    let key = planner.with_inner(|inner| {
        inner.enqueue(
            head,
            allocation(Demand {
                kv_pages: 2,
                rs: RsUnits::default(),
            }),
        )
    });
    let served = |planner: &ResidencyPlanner| {
        planner.with_inner(|inner| {
            inner.queue.get(&key).is_some_and(|waiter| {
                matches!(&waiter.kind, WaitKind::Allocation { outcome: Some(Ok(_)), .. })
            })
        })
    };
    planner.tick();
    assert!(!served(&planner), "nothing free");
    let passes = planner.stats.plan_passes.load(Ordering::Relaxed);
    planner.tick();
    assert_eq!(
        planner.stats.plan_passes.load(Ordering::Relaxed),
        passes,
        "the same room: no pass"
    );
    // Pages come free by a path that tells no one.
    pool.refill_after_stall(2, 0);
    planner.with_inner(|inner| {
        let waiter = inner.queue.get_mut(&key).unwrap();
        waiter.since -= std::time::Duration::from_millis(200);
    });
    planner.tick();
    assert!(served(&planner), "the free counts moved: planned again, served");
    assert_eq!(
        planner.stats.long_waits[usize::from(super::grant::WAKE_BEAT)].load(Ordering::Relaxed),
        1,
        "a wait past 100 ms served, attributed to the beat"
    );
}

/// The M1 on 3b45462d4: 138 processes queued, 364 opened chunks, none held
/// by a candidate. A prompt not yet seated takes only idle or held
/// processes' room, and an older process waits ahead of it untouched; but
/// with nothing in flight a process waiting in the queue runs nothing, and
/// its resident slots are a victim's like any other's.
#[test]
fn with_nothing_in_flight_a_queued_process_s_slots_are_a_victim_s() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let pool = Arc::new(RacePool::new(8));
    pool.refill_after_stall(4, 0);
    let planner = Arc::new(
        ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>),
    );
    let (waiting, prompt) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [prompt, waiting] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let held = planner
        .grant_now(
            waiting,
            Demand {
                kv_pages: 4,
                rs: RsUnits::default(),
            },
        )
        .expect("the pool is free");
    planner.with_inner(|inner| {
        inner.procs.get_mut(&waiting).unwrap().seated = true;
        // The prompt, not yet seated, heads the queue; the seated process
        // waits behind it for more.
        inner.enqueue(
            prompt,
            allocation(Demand {
                kv_pages: 2,
                rs: RsUnits::default(),
            }),
        );
        inner.enqueue(
            waiting,
            allocation(Demand {
                kv_pages: 1,
                rs: RsUnits::default(),
            }),
        );
    });
    let members = |planner: &ResidencyPlanner| {
        planner
            .victim_set()
            .map(|set| set.members.iter().map(|v| v.pid).collect::<Vec<_>>())
            .unwrap_or_default()
    };
    // A fire in flight elsewhere: the seating rule holds.
    let other = ProcessId::new_v4();
    planner.register(other);
    planner.note_admitted(other);
    // One rule: not in a fire in flight, it is a candidate, whatever is
    // in flight elsewhere.
    let fire = planner.fire_in_flight(other);
    assert!(members(&planner).contains(&waiting), "a fire in flight elsewhere: a candidate");
    drop(fire);
    planner.with_inner(|inner| inner.procs.get_mut(&other).unwrap().idle = true);
    assert!(
        members(&planner).contains(&waiting),
        "nothing in flight: the queued process's slots are a victim's"
    );
    drop(held);
}

/// Mac Pro's 2 GiB on 3b45462d4: a restore refused a run of four (no chunk
/// free in its unpark's room) while the map showed its need standing,
/// spinning ~350 lines a second, its need growing. Its need counts the
/// state its serve opens chunks for first (records, rs list slots); it is
/// not grown on runs only the map shows; and it goes again only once more
/// chunks stand free than its refusal left, not on chunks given back and
/// taken again.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_refused_in_its_unpark_s_room_goes_again_only_on_more_room() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(16, 0..0, 64)
    });
    pool.rs_pages.store(4, Ordering::Relaxed);
    // Two runs of four stand in the map.
    pool.free.lock().extend(0..8);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| {
        inner.procs.get_mut(&head).unwrap().state = Residency::Evicted;
    });
    planner.requeue_restore_roles(head, (4, 1), Some(0));
    let key = planner.with_inner(|inner| {
        let at = inner.queue.iter().find(|(_, w)| w.pid == head).map(|(k, _)| *k);
        let key = at.unwrap_or_else(|| {
            inner.enqueue(
                head,
                WaitKind::Restore {
                    demand: Demand::default(),
                },
            )
        });
        if let Some(super::wait::Waiter {
            kind: WaitKind::Restore { demand },
            ..
        }) = inner.queue.get_mut(&key)
        {
            demand.rs = RsUnits {
                slots: 2,
                pages: 0,
                ..RsUnits::default()
            };
        }
        key
    });
    assert_eq!(
        planner.with_inner(|inner| inner.procs[&head].roles_short.map(|n| (n.run, n.pieces))),
        Some((4, 1)),
        "not grown on runs only the map shows"
    );
    // Its state's chunks are single chunks beside its run, not runs of
    // four (asked so, a restore with much to gather asked runs no pool
    // held: Mac Pro at 2 GiB on c10d7de5d).
    let mut need = super::pool::RunNeed::one(4, 1);
    need.add(1, 8);
    assert_eq!(
        planner.head_room(key),
        Some(need),
        "its run of four, and its state's eight chunks beside it"
    );
    // (The pool here holds no state slots: the rest asks none.)
    planner.with_inner(|inner| {
        if let Some(super::wait::Waiter {
            kind: WaitKind::Restore { demand },
            ..
        }) = inner.queue.get_mut(&key)
        {
            demand.rs = RsUnits::default();
        }
    });
    let sent = |planner: &ResidencyPlanner| {
        planner.stats.restores.load(Ordering::Relaxed) > 0
            || planner.with_inner(|inner| inner.procs[&head].roles_free_retry.is_some())
    };
    // A decoding fleet takes a chunk and gives it back: no more room.
    for _ in 0..50 {
        let taken = pool.free.lock().pop();
        if let Some(chunk) = taken {
            pool.free.lock().push(chunk);
        }
        pool.gives.fetch_add(1, Ordering::Relaxed);
        planner.pages_freed();
        for _ in 0..5 {
            crate::rt::yield_now().await;
        }
    }
    // Nothing to move and its placement standing: sent once on this room
    // (the progress rule), whatever the fleet's takes and gives.
    assert!(sent(&planner), "sent once, nothing to move");
    // More chunks stand free, a run of four among them: sent.
    pool.free.lock().extend(8..12);
    pool.gives.fetch_add(1, Ordering::Relaxed);
    planner.pages_freed();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if sent(&planner) {
            break;
        }
    }
    assert!(sent(&planner), "more room: sent");
}

/// Mac Pro and the L40S on 22b0a3e83: the opened chunks all held by queued
/// processes waiting to be restored (85 chunks of one live slot each), no
/// candidate among them. An evicted process's state left on the device at
/// its eviction is parked state: the map counts it, and it moves down a
/// tier for the head, its chunks freed.
#[tokio::test(flavor = "current_thread")]
async fn an_evicted_process_s_state_left_on_the_device_moves_down_for_the_head() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(4, 0..0, 0));
    pool.disk_free.store(64, Ordering::Relaxed);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, evicted) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, evicted] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    // Every chunk holds the evicted process's state; it waits to be
    // restored behind the head.
    pool.kv.lock().extend((0..4).map(|chunk| (chunk, evicted)));
    planner.with_inner(|inner| {
        for pid in [head, evicted] {
            inner.procs.get_mut(&pid).unwrap().state = Residency::Evicted;
        }
        inner.procs.get_mut(&head).unwrap().bind_room = Some(PendingBind {
            singles: 0,
            run: 2,
            pieces: 1,
            pages: 2,
        });
        for pid in [head, evicted] {
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
        if !pool.moved.lock().is_empty() {
            break;
        }
    }
    assert_eq!(*pool.moved.lock(), vec!["state"], "its state moves down for the head");
    assert_eq!(pool.free.lock().len(), 4, "its chunks free");
}

/// A victim whose move found nothing to move (its state still settling) is
/// blocked until the pool moves on; a fire landing (state settled) is such
/// a move too, so a stall with no room given back keeps it out for good no
/// longer (the L40S on 22b0a3e83: a queued process's chunk, no candidate).
#[test]
fn a_victim_that_found_nothing_is_a_candidate_again_once_a_fire_lands() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let pool = Arc::new(RacePool::new(8));
    pool.refill_after_stall(4, 0);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, held) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [held, head] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let grant = planner
        .grant_now(
            held,
            Demand {
                kv_pages: 4,
                rs: RsUnits::default(),
            },
        )
        .expect("the pool is free");
    planner.with_inner(|inner| {
        inner.enqueue(
            head,
            allocation(Demand {
                kv_pages: 2,
                rs: RsUnits::default(),
            }),
        );
    });
    let fire = planner.fire_in_flight(held);
    let now = planner.pool_now();
    planner.with_inner(|inner| {
        // Idle: a member, but for the block.
        inner.procs.get_mut(&held).unwrap().idle = true;
        inner.prepare_blocked.insert(held);
        inner.blocked_at = Some(now);
    });
    let member = |planner: &ResidencyPlanner| {
        planner
            .victim_set()
            .is_some_and(|set| set.members.iter().any(|v| v.pid == held))
    };
    let blocked = |planner: &ResidencyPlanner| {
        planner.lapse_blocks();
        planner.with_inner(|inner| inner.prepare_blocked.contains(&held))
    };
    assert!(blocked(&planner), "blocked on this pool");
    assert!(!member(&planner), "no candidate while blocked");
    drop(fire);
    assert!(!blocked(&planner), "a fire landed: a candidate again");
    drop(grant);
}

/// Mac Pro on 22b0a3e83: a restore refused a run of nine whose need read
/// [(7, 8), (4, 1)]. A re-send cleared its refusal, and its need fell back
/// to its bind's runs and its own. The refused run stays in its need
/// across a re-send, until a restore serves it.
#[test]
fn a_refused_run_stays_in_the_need_across_a_re_send() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let pool = Arc::new(BindRoomPool::new(64, 0..0, 64));
    pool.free.lock().extend(0..64);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Restoring;
        proc.bind_room = Some(PendingBind {
            singles: 0,
            run: 7,
            pieces: 1,
            pages: 7,
        });
    });
    planner.requeue_restore_roles(head, (9, 1), Some(0));
    let key = planner
        .with_inner(|inner| inner.queue.iter().find(|(_, w)| w.pid == head).map(|(k, _)| *k))
        .expect("queued again");
    let has_nine = |planner: &ResidencyPlanner| {
        planner.head_room(key).is_some_and(|need| need.pieces_at(9) >= 1)
    };
    assert!(has_nine(&planner), "refused a run of nine: its need asks it");
    // Re-sent: its refusal cleared for the try.
    planner.with_inner(|inner| inner.roles_retry_head());
    assert!(has_nine(&planner), "re-sent: its need still asks the run of nine");
}

/// The L40S at 6 GiB on 22b0a3e83 (+12% against its base): restores waited
/// for a fire landing's poke, up to 15 s, the room signal waking about one
/// a run. A restore short of its state slots publishes no count the signal
/// measures; room given back wakes it within a turn, never left for a poke.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_short_of_state_is_served_by_the_room_signal() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    planner.arm_drain_task();
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| {
        inner.procs.get_mut(&head).unwrap().state = Residency::Evicted;
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand {
                    kv_pages: 0,
                    rs: RsUnits {
                        slots: 1,
                        ..RsUnits::default()
                    },
                },
            },
        );
    });
    planner.poke();
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    let state = |planner: &ResidencyPlanner| planner.with_inner(|inner| inner.procs[&head].state);
    assert_eq!(state(&planner), Residency::Evicted, "no state slot free");
    // A state slot is given back: the room signal, no fire landing.
    pool.rs_free.store(1, Ordering::Relaxed);
    planner.returns.slots_freed(crate::store::slots::RoomCount {
        tier: crate::store::slots::RoomTier::Device,
        chunks: 0,
        slots: 0,
    });
    for _ in 0..50 {
        crate::rt::yield_now().await;
        if planner.stats.restores.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.restores.load(Ordering::Relaxed) > 0 || state(&planner) != Residency::Evicted,
        "the room signal serves it"
    );
}

/// The L40S on 3b45462d4: a restore whose guest state (1,058 slots, about
/// sixteen chunks) and runs past what the state limit leaves, waiting with
/// nothing evictable. A head whose whole step takes more chunks than state
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

/// Mac Pro on 22b0a3e83: a refused restore head re-planned by every fire
/// landing (838 no-step lines a second at peak, nothing in its room
/// moved). A fire landing plans no pass for a head refused on room that
/// has not moved since; room that moves does.
#[tokio::test(flavor = "current_thread")]
async fn fire_landings_plan_no_pass_for_a_head_refused_on_room_unmoved() {
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(16, 0..0, 64)
    });
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    planner.arm_drain_task();
    let (head, decoding) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, decoding] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        inner.procs.get_mut(&head).unwrap().state = Residency::Evicted;
    });
    planner.requeue_restore_roles(head, (2, 1), Some(0));
    planner.with_inner(|inner| {
        if !inner.queue.values().any(|waiter| waiter.pid == head) {
            inner.enqueue(
                head,
                WaitKind::Restore {
                    demand: Demand::default(),
                },
            );
        }
    });
    for _ in 0..200 {
        crate::rt::yield_now().await;
    }
    let passes = planner.stats.plan_passes.load(Ordering::Relaxed);
    for _ in 0..100 {
        drop(planner.fire_in_flight(decoding));
        for _ in 0..5 {
            crate::rt::yield_now().await;
        }
    }
    let more = planner.stats.plan_passes.load(Ordering::Relaxed) - passes;
    assert!(more <= 2, "a hundred fire landings planned {more} passes");
    // Room moves: a fire landing plans again.
    pool.free.lock().extend([1, 3]);
    drop(planner.fire_in_flight(decoding));
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    assert!(
        planner.stats.plan_passes.load(Ordering::Relaxed) > passes + more,
        "room moved: planned"
    );
}

/// The L40S on 46296784b: four binds queued (seq 10 to 13), the youngest
/// evicted while it waited, restored for its own bind while the binds of 11
/// and 12 ahead of it still waited, and evicted for them again 1,730 times
/// (rs disk churn: 256,939 copies out). A process evicted while it waited
/// in the queue is restored only once no process ahead of it has an entry
/// there.
#[test]
fn a_process_evicted_while_queued_is_restored_only_behind_the_queue_ahead_of_it() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(8))));
    let pids: Vec<ProcessId> = (0..4).map(|_| ProcessId::new_v4()).collect();
    for &pid in &pids {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let bind = || WaitKind::BindRoom {
        run: 1,
        pieces: 1,
        singles: 0,
        pages: 1,
        notify: Arc::new(Notify::new()),
        served: false,
        withdrawn: false,
        rise: None,
    };
    let youngest = pids[3];
    let keys = planner.with_inner(|inner| {
        let keys: Vec<EntryKey> = pids[..3].iter().map(|&pid| inner.enqueue(pid, bind())).collect();
        let proc = inner.procs.get_mut(&youngest).unwrap();
        proc.state = Residency::Evicted;
        proc.evicted_queued = true;
        inner.enqueue(
            youngest,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
        keys
    });
    let head_pid = |planner: &ResidencyPlanner| {
        planner.with_inner(|inner| inner.unmet_head().map(|(_, waiter)| waiter.pid))
    };
    // The oldest's bind served and not yet taken, the next two waiting.
    planner.with_inner(|inner| {
        if let Some(super::wait::Waiter {
            kind: WaitKind::BindRoom { served, .. },
            ..
        }) = inner.queue.get_mut(&keys[0])
        {
            *served = true;
        }
    });
    assert_eq!(head_pid(&planner), Some(pids[1]), "the binds ahead of it go first");
    planner.with_inner(|inner| {
        for key in &keys[1..] {
            inner.queue.remove(key);
        }
    });
    assert_ne!(
        head_pid(&planner),
        Some(youngest),
        "the oldest's served bind not yet taken: still held"
    );
    planner.with_inner(|inner| {
        inner.queue.remove(&keys[0]);
    });
    assert_eq!(head_pid(&planner), Some(youngest), "nothing ahead of it: restored");
}

/// Mac Pro's M5 at 2 GiB on 5120cbc3c, stuck at 77 serves: a bind head
/// short of a run of chunks (need [(9, 1)]), one queued resident's chunk
/// in the cheapest window (need_moves Some(Some(1))), 300 more queued
/// behind, nothing in flight, and no eviction ever issued. Here: a run of
/// three over chunks 0..3, chunk 1 a queued resident's, the disk with no
/// slot free, and more one-page allocations queued than chunks stand
/// free. The one move is issued.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_head_one_queued_resident_away_is_funded_with_nothing_in_flight() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(3, 0..0, 0));
    pool.free.lock().extend([0, 2]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let binder = ProcessId::new_v4();
    let occupant = ProcessId::new_v4();
    let queued: Vec<ProcessId> = (0..4).map(|_| ProcessId::new_v4()).collect();
    for &pid in [binder, occupant].iter().chain(&queued) {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    pool.kv.lock().push((1, occupant));
    planner.with_inner(|inner| {
        for &pid in std::iter::once(&occupant).chain(&queued) {
            inner.enqueue(
                pid,
                allocation(Demand {
                    kv_pages: 1,
                    rs: RsUnits::default(),
                }),
            );
        }
    });
    let p = planner.clone();
    let _bind = crate::rt::spawn(async move { p.wait_bind_room(binder, 3, 1, 0).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the queued occupant is moved for the head"
    );
    assert!(
        planner.with_inner(|inner| inner.last_plan.is_some()),
        "the stall dump's plan line says where the plan stopped"
    );
}

/// Mac Pro's M5 at 5 GiB on 5120cbc3c, stalled at 2,637 serves: a restore
/// head refused in its unpark (UnparkShort, a run of four short, no free
/// run), its need two runs of seven and one of four, one queued resident
/// away (need_moves Some(Some(1))), 1,600 queued, nothing in flight, no
/// eviction ever issued. Here: a run of two over chunks 0..2, chunk 1 a
/// queued resident's, more one-page allocations queued than chunks stand
/// free, the disk with no slot free. The one move is issued.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_head_one_queued_resident_away_is_funded_with_nothing_in_flight() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(3, 0..0, 0));
    pool.free.lock().extend([0, 2]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    let occupant = ProcessId::new_v4();
    let queued: Vec<ProcessId> = (0..4).map(|_| ProcessId::new_v4()).collect();
    for &pid in [head, occupant].iter().chain(&queued) {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    pool.kv.lock().push((1, occupant));
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(PendingBind {
            singles: 0,
            run: 2,
            pieces: 1,
            pages: 2,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
        for &pid in std::iter::once(&occupant).chain(&queued) {
            inner.enqueue(
                pid,
                allocation(Demand {
                    kv_pages: 1,
                    rs: RsUnits::default(),
                }),
            );
        }
    });
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the queued occupant is moved for the head"
    );
}

/// The L40S at 3 GiB disk-only on 5120cbc3c: the last six beams waited
/// silently, every one evicted and queued for its restore, none resident,
/// nothing in flight, and no head chosen (head_kind=-): each restore was
/// held for the head it had been evicted for, itself an evicted restore
/// held the same way. With nothing else to serve, the oldest such restore
/// heads the queue.
#[test]
fn with_every_waiter_an_evicted_restore_held_for_another_one_heads_the_queue() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(8))));
    let pids: Vec<ProcessId> = (0..3).map(|_| ProcessId::new_v4()).collect();
    for &pid in &pids {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let keys = planner.with_inner(|inner| {
        pids.iter()
            .enumerate()
            .map(|(at, &pid)| {
                let proc = inner.procs.get_mut(&pid).unwrap();
                proc.state = Residency::Evicted;
                proc.evicted_for = Some(pids[(at + 1) % pids.len()]);
                inner.enqueue(
                    pid,
                    WaitKind::Restore {
                        demand: Demand::default(),
                    },
                )
            })
            .collect::<Vec<_>>()
    });
    assert_eq!(
        planner.with_inner(|inner| inner.unmet_head().map(|(key, _)| key)),
        Some(keys[0]),
        "the oldest held restore heads the queue"
    );
}

/// The L40S at 3 GiB disk-only on 5120cbc3c, its last six beams: every
/// waiter an evicted restore held for another's, no one resident, nothing
/// in flight, and the device still held by their state (17 chunks, 150
/// pages). The oldest heads the queue, and the state the others left on
/// the device moves down for its runs, nobody resident.
#[tokio::test(flavor = "current_thread")]
async fn with_no_one_resident_evicted_state_moves_down_for_a_held_restore() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(4, 0..0, 0));
    pool.disk_free.store(64, Ordering::Relaxed);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (first, second) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [first, second] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    // The second's state holds every chunk; each was evicted for the other.
    pool.kv.lock().extend((0..4).map(|chunk| (chunk, second)));
    planner.with_inner(|inner| {
        for (pid, other) in [(first, second), (second, first)] {
            let proc = inner.procs.get_mut(&pid).unwrap();
            proc.state = Residency::Evicted;
            proc.evicted_for = Some(other);
        }
        inner.procs.get_mut(&first).unwrap().roles_short = Some(PendingBind {
            singles: 0,
            run: 2,
            pieces: 1,
            pages: 2,
        });
        for pid in [first, second] {
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
        if !pool.moved.lock().is_empty() {
            break;
        }
    }
    assert_eq!(*pool.moved.lock(), vec!["state"], "the second's state moves down for the first");
}

/// The L40S at 6 GiB on 5120cbc3c: restore waits served by pokes, up to
/// 16.5 s, a few by the room signal. A restore refused in an earlier
/// unpark and short of its state slots now: its step is the state slots
/// (read ahead of its runs), but its want was the refusal's (more free
/// chunks than stood), which state given back inside open chunks never
/// moves. Any room given back wakes it, as for one never refused; its
/// wait says what it lacked.
#[tokio::test(flavor = "current_thread")]
async fn a_refused_restore_short_of_state_is_served_by_the_room_signal() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    planner.arm_drain_task();
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    let key = planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(PendingBind {
            singles: 0,
            run: 1,
            pieces: 1,
            pages: 1,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand {
                    kv_pages: 0,
                    rs: RsUnits {
                        slots: 1,
                        ..RsUnits::default()
                    },
                },
            },
        )
    });
    planner.poke();
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    let state = |planner: &ResidencyPlanner| planner.with_inner(|inner| inner.procs[&head].state);
    assert_eq!(state(&planner), Residency::Evicted, "no state slot free");
    assert_eq!(
        planner.with_inner(|inner| inner.queue.get(&key).map(|waiter| waiter.lacked)),
        Some("state slots, refused in its unpark before"),
        "its lack is recorded for its long wait"
    );
    let passes = planner.stats.plan_passes.load(Ordering::Relaxed);
    pool.rs_free.store(1, Ordering::Relaxed);
    planner.returns.slots_freed(crate::store::slots::RoomCount {
        tier: crate::store::slots::RoomTier::Device,
        chunks: 0,
        slots: 0,
    });
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    assert!(
        planner.stats.plan_passes.load(Ordering::Relaxed) > passes,
        "the room signal plans it, no fire landing"
    );
}

/// A policy's hold (the test-pause policy of sc/amd-beam-pages
/// holds a process for a while, then lifts it): a process held
/// and parked stays parked while room stands free and its
/// own lanes wait to be resident, and, taking sets that are off the
/// device, owes them parked; its restore is queued once the policy lifts
/// it, never before.
#[tokio::test(flavor = "current_thread")]
async fn a_process_held_below_the_floor_stays_parked_until_the_policy_lifts_it() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(64))));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    let demand = Demand {
        kv_pages: 2,
        rs: RsUnits::default(),
    };
    planner.with_inner(|inner| {
        inner.reschedule(None, |_| (0, true));
        let proc = inner.procs.get_mut(&pid).unwrap();
        proc.state = Residency::Evicted;
        proc.parked = Some(demand);
    });
    let restore_queued = |planner: &ResidencyPlanner| {
        planner.with_inner(|inner| {
            inner
                .queue
                .values()
                .any(|waiter| waiter.pid == pid && matches!(waiter.kind, WaitKind::Restore { .. }))
        })
    };
    // Its lanes wait to be resident, 64 pages free.
    let p = planner.clone();
    let waiting = crate::rt::spawn(async move { p.wait_resident(pid).await });
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    assert!(!restore_queued(&planner), "held: no restore queued for its waiting lanes");
    assert_eq!(planner.with_inner(|inner| inner.procs[&pid].state), Residency::Evicted);
    // It takes sets off the device: owed, still parked.
    planner.owe(
        pid,
        Demand {
            kv_pages: 1,
            rs: RsUnits::default(),
        },
    );
    assert!(!restore_queued(&planner), "held: what it owes stays parked");
    assert_eq!(
        planner.with_inner(|inner| inner.procs[&pid].parked.map(|d| d.kv_pages)),
        Some(3),
        "parked with what it owes"
    );
    // The policy lifts it: its restore goes.
    planner.with_inner(|inner| inner.reschedule(None, |_| (1, false)));
    assert!(restore_queued(&planner), "lifted: its restore is queued");
    waiting.abort();
}

/// The test-pause policy (`crates/builtins/policies/test-pause`), built and
/// run as PLEX 1.1 runs it: a process that states `pause-at-ms: 200` and
/// `pause-for-ms: 300` is held from 200 ms after its arrival to 500 ms, by
/// its `schedule/hold` row, and lifted at the alarm it set for 500 ms. The
/// rows go to the schedule table as the port enacts them, and the planner
/// reads them: before its window it runs, inside it its set stays parked
/// (no restore, even with its lanes asking), and once lifted its restore is
/// queued. Another process, stating none, is never held; one stating the
/// prop as text is terminated with a reason naming it, and never held.
#[tokio::test(flavor = "current_thread")]
async fn the_test_pause_policy_parks_a_set_in_its_window_and_resumes_it_after() {
    use plex_host::host::{Config, Host};
    use plex_host::tables::Value;
    use std::str::FromStr;

    const MS: u64 = 1_000_000;
    let _schedule = crate::policy::schedule::TEST_LOCK.write().await;
    crate::policy::schedule::install(None);
    let host = Host::new(Config {
        supported: vec![plex_host::tables::SCHEDULE_HOLD.to_string()],
        manual: true,
        ..Config::default()
    });
    let component = builtins::policy("test-pause").expect("built with test-policies");
    let imports = host.load("test-pause", component.component).unwrap();
    assert!(
        imports
            .iter()
            .any(|i| i == plex_host::tables::SCHEDULE_HOLD),
        "it writes schedule/hold: {imports:?}"
    );
    // Each round's committed hold rows, enacted as the port enacts them.
    let round = |at_ms: u64| {
        host.set_now(at_ms * MS);
        host.run(false);
        let delta = host.poll_delta();
        let holds = delta
            .schedule_hold
            .rows
            .into_iter()
            .map(|(id, row)| (ProcessId::from_str(&id).unwrap(), row))
            .collect();
        crate::policy::schedule::apply(crate::policy::schedule::Change {
            holds,
            hold_default: delta.schedule_hold.default,
            ..Default::default()
        });
    };

    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(64))));
    let (paused, other) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [paused, other] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    host.set_now(10 * MS);
    let hints = |at: Value| {
        vec![
            ("pause-at-ms".to_string(), at),
            ("pause-for-ms".to_string(), Value::Int(300)),
        ]
    };
    host.arrived(
        &paused.to_string(),
        None,
        None,
        Some("self".into()),
        None,
        None,
        hints(Value::Int(200)),
        0,
    );
    host.arrived(
        &other.to_string(),
        None,
        None,
        Some("self".into()),
        None,
        None,
        vec![],
        0,
    );
    host.started(&paused.to_string());
    host.started(&other.to_string());

    let restore_queued = |planner: &ResidencyPlanner| {
        planner.with_inner(|inner| {
            inner.queue.values().any(|waiter| {
                waiter.pid == paused && matches!(waiter.kind, WaitKind::Restore { .. })
            })
        })
    };
    let held = |pid| crate::policy::schedule::held(pid);

    // Before its window: it runs, and the policy asks to be woken at 210.
    round(10);
    assert!(
        !held(paused) && !held(other),
        "before its window nothing is held"
    );
    assert_eq!(host.alarm(), Some(210 * MS), "woken where its window opens");
    round(100);
    assert!(!held(paused), "still before its window");

    // Inside it: held, and the planner pauses it (its set parked below the
    // device, as the pause path leaves it).
    round(210);
    assert!(held(paused), "held inside its window");
    assert!(!held(other), "one stating no pause is never held");
    assert_eq!(
        host.alarm(),
        Some(510 * MS),
        "woken where its window closes"
    );
    planner.reschedule();
    planner.with_inner(|inner| {
        assert!(inner.procs[&paused].held, "the planner reads the hold");
        assert!(!inner.procs[&other].held);
        let proc = inner.procs.get_mut(&paused).unwrap();
        proc.state = Residency::Evicted;
        proc.parked = Some(Demand {
            kv_pages: 2,
            rs: RsUnits::default(),
        });
    });
    let p = planner.clone();
    let waiting = crate::rt::spawn(async move { p.wait_resident(paused).await });
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    round(400);
    planner.reschedule();
    assert!(held(paused), "still inside its window");
    assert!(
        !restore_queued(&planner),
        "held: its set stays parked though its lanes ask"
    );
    assert_eq!(
        planner.with_inner(|inner| inner.procs[&paused].state),
        Residency::Evicted
    );

    // Its window closes at the alarm: lifted, its restore queued.
    round(510);
    assert!(!held(paused), "lifted once its window closed");
    assert_eq!(host.alarm(), None, "no later edge: no alarm");
    planner.reschedule();
    assert!(restore_queued(&planner), "lifted: its restore is queued");
    waiting.abort();

    // A prop stated as text ("300") is refused by name: the process is
    // terminated with a reason naming it, and nothing is held.
    let text = ProcessId::new_v4();
    host.set_now(600 * MS);
    host.arrived(
        &text.to_string(),
        None,
        None,
        Some("self".into()),
        None,
        None,
        hints(Value::Text("300".into())),
        0,
    );
    host.run(false);
    let delta = host.poll_delta();
    assert!(
        delta
            .terminate
            .iter()
            .any(|(id, why)| *id == text.to_string() && why.contains("pause-at-ms")),
        "refused by name: {:?}",
        delta.terminate
    );
    assert!(
        !delta
            .schedule_hold
            .rows
            .iter()
            .any(|(id, row)| *id == text.to_string() && *row == Some(true)),
        "and never held"
    );
    assert_eq!(host.stats().traps, 0, "no call traps");
    crate::policy::schedule::install(None);
}

/// Mac Pro at 2 GiB disk-only on 1e87ab171, stalled at 86 serves: a
/// process suspended for its bind (two runs of nine Instance chunks
/// short), evicted to disk, restored ~95 ms later, then waiting on
/// "its instances' rebinds" (InstanceId(5)) until teardown, Restoring the
/// whole while; the queue's head, a restore, held behind it as a move in
/// flight, no stall dump, nothing evicted or compacted. Its state back
/// and committed, a restore waiting on its rebinds moves nothing: it is
/// no move in flight, the bind-suspended restore it was no longer holds
/// the next, the stall dump names what it waits on, and its restore
/// landing (or failing) clears it.
#[test]
fn a_restore_waiting_on_its_rebinds_is_nothing_in_flight() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(8))));
    let (rebinding, head) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [rebinding, head] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&rebinding).unwrap();
        proc.state = Residency::Restoring;
        inner.bind_restoring = Some(rebinding);
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.bind_room = Some(PendingBind {
            singles: 0,
            run: 9,
            pieces: 2,
            pages: 18,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    let quiet = |planner: &ResidencyPlanner| planner.with_inner(|inner| inner.nothing_in_flight());
    let released =
        |planner: &ResidencyPlanner| planner.with_inner(|inner| inner.bind_restoring_released());
    assert!(!quiet(&planner), "its copy back on its way: in flight");
    assert!(!released(&planner), "and it holds the next bind-suspended restore");
    planner.rebind_waiting(rebinding, Some("[InstanceId(5)]".to_owned()));
    assert!(quiet(&planner), "waiting on its rebinds: nothing in flight");
    assert!(released(&planner), "the next bind-suspended restore goes");
    // Its rebinds land: restored, cleared.
    planner.report_restored(rebinding);
    assert!(planner.with_inner(|inner| inner.procs[&rebinding].rebind_waiting.is_none()));
}

/// Four free chunks scattered and a restore head refused a run of four:
/// the store stands no run, so the head is not sent again on the free
/// count; the map reads moves for it, and with nothing in flight the
/// residents between the free chunks are moved for it.
#[tokio::test(flavor = "current_thread")]
async fn four_scattered_free_chunks_and_a_run_of_four_asked_issue_a_plan() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    pool.kv.lock().extend([1, 3, 5, 7].map(|chunk| (chunk, resident)));
    let key = planner.with_inner(|inner| {
        inner.procs.get_mut(&resident).unwrap().idle = true;
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(PendingBind {
            singles: 0,
            run: 4,
            pieces: 1,
            pages: 4,
        });
        proc.roles_short_stood = Some(false);
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        )
    });
    let mut map = pool.chunk_occupants(&[resident]);
    map.merge(planner.parked_map(head));
    let need = planner.head_room(key).expect("a need of runs");
    assert!(
        map.cheapest_need(&need).is_some_and(|moves| !moves.is_empty()),
        "need_moves > 0"
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
        "a plan is issued"
    );
}

/// Mac Pro at 5 GiB on 1e87ab171: the no-step lines carried "no eviction
/// plan for this head" and none of ended= / host_short / unhostable, the
/// pass having stopped short of the eviction plan. Every place a pass
/// stops for its head says where, and every unserved-head line carries the
/// same fields.
#[tokio::test(flavor = "current_thread")]
async fn a_pass_that_stops_short_of_the_eviction_plan_says_where() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, other) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, other] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let key = planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(PendingBind {
            singles: 0,
            run: 4,
            pieces: 1,
            pages: 4,
        });
        proc.roles_short_stood = Some(false);
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        )
    });
    let fire = planner.fire_in_flight(other);
    planner.plan();
    let (plan, _) = planner.with_inner(|inner| inner.plan_fields(key));
    assert!(
        plan.contains("ended=\"restore short of runs: something in flight, not funded\"")
            && plan.contains("host_short=")
            && plan.contains("unhostable="),
        "{plan}"
    );
    drop(fire);
}

/// Mac Pro at 2 GiB on 4cef0c7d3: a bind-short eviction suspended
/// 236c3046 and waited on its set's fires, one of them in a complete frame
/// wider than the rung in hand (capped), which goes only once raised. Its
/// eviction withdrew that raise's wait, and the eviction waited for good.
/// A rung raise for a capped frame is queued for an owner being evicted,
/// stays queued through its eviction, and is served once room frees: the
/// frame goes, and the eviction's wait on it ends.
#[tokio::test(flavor = "current_thread")]
async fn a_capped_frame_s_raise_waits_through_its_owner_s_eviction_and_is_served() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let owner = ProcessId::new_v4();
    planner.register(owner);
    planner.note_admitted(owner);
    let p = planner.clone();
    let raised = crate::rt::spawn(async move {
        p.fund_rise(owner, 3, || std::future::ready("raised")).await
    });
    for _ in 0..20 {
        crate::rt::yield_now().await;
    }
    let raise_queued = |planner: &ResidencyPlanner| {
        planner.with_inner(|inner| {
            inner.queue.values().any(|waiter| {
                matches!(
                    waiter.kind,
                    WaitKind::BindRoom {
                        rise: Some(3),
                        withdrawn: false,
                        ..
                    }
                )
            })
        })
    };
    assert!(raise_queued(&planner), "short of room: the raise waits");
    // Its owner's eviction begins (a bind elsewhere picked it).
    let wake = planner.with_inner(|inner| inner.begin_eviction(owner, Reclaim::default()));
    for notify in wake {
        notify.notify_waiters();
    }
    for _ in 0..20 {
        crate::rt::yield_now().await;
    }
    assert!(raise_queued(&planner), "its eviction waits on the frame: the raise stays");
    assert!(!raised.is_finished());
    // Room frees: the raise is served and taken.
    pool.free.lock().extend(0..3);
    pool.gives.fetch_add(1, Ordering::Relaxed);
    planner.pages_freed();
    let taken = raised.await.expect("the wait ends");
    assert_eq!(taken.expect("not refused"), Some("raised"), "served: the frame goes");
}

/// The planner's census says each pass's cost: passes, their total time,
/// mean, and the bound 99 of each 100 fell under.
#[test]
fn the_census_counts_each_plan_pass_and_its_time() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(8))));
    for _ in 0..10 {
        // Each on an event: unmoved, a wake runs no pass.
        planner.events.fetch_add(1, Ordering::Relaxed);
        planner.plan();
    }
    let (passes, total, mean, p99) = planner.pass_census();
    assert_eq!(passes, 10);
    assert_eq!(mean, total / 10);
    assert!(p99 >= mean, "the tail bound covers the mean: {p99} {mean}");
    planner.stats.pass_took(5);
    assert_eq!(planner.pass_census().0, 11);
}

/// Mac Pro at 2 GiB on 5aa45805b: 151k no-step lines, each with
/// ended=none and need_moves Some(Some(1)): the pass sent the refused
/// restore again (its placement stood) and recorded nowhere that it had.
/// A pass that sends a refused restore again says so.
#[tokio::test(flavor = "current_thread")]
async fn a_pass_that_sends_a_refused_restore_again_says_so() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    pool.free.lock().extend(0..8);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, other) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, other] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let key = planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(PendingBind {
            singles: 0,
            run: 2,
            pieces: 1,
            pages: 2,
        });
        proc.roles_short_stood = Some(false);
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        )
    });
    // Something in flight, so the restore it sends lands nowhere here.
    let fire = planner.fire_in_flight(other);
    planner.plan();
    let (plan, _) = planner.with_inner(|inner| inner.plan_fields(key));
    // Boarded, the pass's last end for it is the send itself.
    assert!(plan.contains("restore sent"), "{plan}");
    drop(fire);
}

/// The L40S at 3 GiB on 4cef0c7d3: 613 evictions within 200 ms of their
/// process's restore, about one eviction a serve. A victim was restored as
/// soon as the head it was evicted for was served, took the room back
/// before the head ran on it, and the head's next ask evicted it again.
/// Restored once the head had fired, it was evicted again for the head's
/// next page (the field mix: a lane evicted for the same head with no
/// restore of the head between). A victim's restore waits until its head
/// is done with its request (here, waits on its client).
#[test]
fn a_victim_is_restored_only_once_its_head_is_done_with_its_request() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::with_swap(8, 64))));
    let (head, victim, other) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, victim, other] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        // The head was restored for its wait and has not fired since.
        inner.procs.get_mut(&head).unwrap().unfired = true;
        let proc = inner.procs.get_mut(&victim).unwrap();
        proc.state = Residency::Evicted;
        proc.evicted_for = Some(head);
        inner.enqueue(
            victim,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    let head_of = |planner: &ResidencyPlanner| {
        planner.with_inner(|inner| inner.unmet_head().map(|(_, waiter)| waiter.pid))
    };
    // Something else in flight, so no quiet fallback applies.
    let fire = planner.fire_in_flight(other);
    assert_eq!(head_of(&planner), None, "its head has not run on the room yet");
    // The head fires: it runs on in its request, and grows into the room.
    let head_fire = planner.fire_in_flight(head);
    assert_eq!(head_of(&planner), None, "its head runs on in its request");
    // The head waits on its client: its victim may come back.
    planner.set_idle(head, true);
    let sent = planner.with_inner(|inner| {
        inner.unmet_head().map(|(_, waiter)| waiter.pid) == Some(victim)
            || !inner.queue.values().any(|waiter| waiter.pid == victim)
    });
    assert!(sent, "the head's request is done: the victim is restored");
    drop(head_fire);
    drop(fire);
}

/// Mac Pro at 5 GiB on 5aa45805b: an eviction's copy refused untouched by
/// the engine (state the runtime placed inside its carve) failed the
/// process: a beam lost. A copy refused untouched takes the eviction back
/// and keeps the process, no victim again until the pool moves.
#[test]
fn an_eviction_copy_refused_untouched_keeps_its_process() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::with_swap(8, 64))));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    planner.with_inner(|inner| {
        let _ = inner.begin_eviction(pid, Reclaim::default());
    });
    planner.eviction_copy_refused(pid, "kv to LocalDisk copy: impossible submission: ...");
    planner.with_inner(|inner| {
        assert_eq!(inner.procs[&pid].state, Residency::Resident, "kept, resident");
        assert!(!inner.killing.contains(&pid), "not failed");
        assert!(inner.prepare_blocked.contains(&pid), "no victim until the pool moves");
        assert!(inner.evicting.is_empty(), "its eviction is settled");
    });
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

/// Mac Pro at 2 GiB on 7e75af541: a 52k-line no-step spin, head dc373c1e
/// a bind-suspended restore, free runs (7, 3) beside its need, refused each
/// time by its own check (PendingBindShort, free runs after its own unpark
/// 1). The pass read the bind's runs beside its dead parts alone and sent
/// it; the restore, after its unpark took the rest, refused. The pass now
/// reads the one placement the restore is checked by, and where that does
/// not stand it is not sent: room is made for it instead.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_suspended_restore_is_sent_only_where_its_whole_placement_stands() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(16, 0..0, 64)
    });
    pool.free.lock().extend(0..16);
    pool.store_refuses.store(true, Ordering::Relaxed);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    let key = planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&pid).unwrap();
        proc.state = Residency::Evicted;
        proc.bind_room = Some(PendingBind {
            singles: 0,
            run: 7,
            pieces: 2,
            pages: 14,
        });
        inner.enqueue(
            pid,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        )
    });
    for _ in 0..20 {
        planner.pages_freed();
        planner.poke();
        for _ in 0..5 {
            crate::rt::yield_now().await;
        }
    }
    // Its placement does not stand, the map stands it, and nothing moves:
    // sent once on this room by the progress rule (the takes decide),
    // never re-sent on it.
    let (plan, _) = planner.with_inner(|inner| inner.plan_fields(key));
    assert!(plan.contains("sent"), "{plan}");
    assert_eq!(
        planner.with_inner(|inner| inner.procs[&pid].mismatch_sent_at),
        Some(16),
        "sent once on this room"
    );
    // Its placement stands: sent.
    pool.store_refuses.store(false, Ordering::Relaxed);
    planner.pages_freed();
    planner.poke();
    for _ in 0..50 {
        crate::rt::yield_now().await;
        if planner.with_inner(|inner| inner.procs[&pid].state) != Residency::Evicted {
            break;
        }
    }
    let (plan, _) = planner.with_inner(|inner| inner.plan_fields(key));
    assert_ne!(
        planner.with_inner(|inner| inner.procs[&pid].state),
        Residency::Evicted,
        "sent once its placement stands: {plan} {}",
        planner.debug_queue()
    );
}

/// The M1 on 7e75af541 hung 9 s after ready: a bind head short of two runs
/// of nine, 342 processes resident between fires holding 356 chunks, and
/// the map's candidates 0, "no pick: the policy is asked", nothing in
/// flight. Those processes had bound (their guest state on the device)
/// and waited for an execution seat: not admitted, they were never
/// victims. With nothing in flight such a process runs nothing: it is a
/// victim like any other; with something in flight it is still spared.
#[test]
fn with_nothing_in_flight_a_process_not_yet_admitted_is_a_victim() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(8))));
    let (head, seated_later, other) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, seated_later, other] {
        planner.register(pid);
    }
    planner.note_admitted(head);
    planner.note_admitted(other);
    planner.with_inner(|inner| {
        inner.enqueue(
            head,
            WaitKind::BindRoom {
                run: 9,
                pieces: 2,
                singles: 0,
                pages: 18,
                notify: Arc::new(Notify::new()),
                served: false,
                withdrawn: false,
                rise: None,
            },
        );
    });
    let members = |planner: &ResidencyPlanner| {
        planner
            .victim_set()
            .map(|set| set.members.iter().map(|v| v.pid).collect::<Vec<_>>())
            .unwrap_or_default()
    };
    // One rule: a resident not in a fire in flight is a candidate, with
    // something in flight elsewhere or not, admitted or not.
    let fire = planner.fire_in_flight(other);
    assert!(members(&planner).contains(&seated_later), "something in flight elsewhere: a candidate");
    assert!(!members(&planner).contains(&other), "the one firing is not");
    drop(fire);
    assert!(members(&planner).contains(&seated_later), "nothing in flight: a candidate");
}

/// Mac Pro at 2 GiB on dc1414ae8: "an eviction waits for its sets' fires
/// pid=8d961f69 ... capped=rows=112 owner=Some(8d961f69)": the owner of
/// a frame cut wider than the cap was evicted, and its eviction waited on
/// that frame, which goes only once raised. Such a process is no victim
/// while its frame waits.
#[test]
fn the_owner_of_a_capped_frame_is_no_victim() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_write();
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(8))));
    let (head, capped, other) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, capped, other] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        inner.enqueue(
            head,
            WaitKind::BindRoom {
                run: 2,
                pieces: 1,
                singles: 0,
                pages: 2,
                notify: Arc::new(Notify::new()),
                served: false,
                withdrawn: false,
                rise: None,
            },
        );
    });
    let members = |planner: &ResidencyPlanner| {
        planner
            .victim_set()
            .map(|set| set.members.iter().map(|v| v.pid).collect::<Vec<_>>())
            .unwrap_or_default()
    };
    assert!(members(&planner).contains(&capped), "a victim like any other");
    crate::store::shared::owe_cut_rows(0, 112, Some(capped));
    let with_capped = members(&planner);
    crate::store::shared::owe_cut_rows(0, 0, None);
    assert!(!with_capped.contains(&capped), "its frame waits on a raise: no victim");
    assert!(with_capped.contains(&other));
}

/// The L40S at 3 GiB on f623134fc: kv resident 0, the stall's map naming 7
/// moves for a restore head, the pass ending "no pick: the policy is
/// asked", nothing in flight, 185 s with no serve. Such an end is a quiet
/// wait (a planner bug: no step found) only where the moves named are moves
/// a step makes: here the map names a holder on its way into its process,
/// which no rule lets the pass move, and the wait is no bug and not said as
/// one (the M1 at beam x40: 53 such lines over residents no step moves).
#[tokio::test(flavor = "current_thread")]
async fn a_wait_on_moves_no_step_makes_is_no_quiet_wait() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(6, 0..0, 64));
    pool.free.lock().extend([0, 1, 2, 3]);
    pool.store_refuses.store(true, Ordering::Relaxed);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, holder) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, holder] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    pool.kv.lock().extend([(4, holder), (5, holder)]);
    planner.with_inner(|inner| {
        inner.procs.get_mut(&holder).unwrap().holder = true;
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(PendingBind {
            singles: 0,
            run: 1,
            pieces: 5,
            pages: 5,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.quiet_waits.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    let end = planner.with_inner(|inner| inner.last_plan.as_ref().map(|end| end.ended));
    assert!(
        end.is_some_and(|end| end.contains("the policy is asked")),
        "the policy asked ({end:?})"
    );
    assert_eq!(
        planner.stats.quiet_waits.load(Ordering::Relaxed),
        0,
        "a holder no step moves: no quiet wait ({end:?})"
    );
}

/// Mac Pro at 2 GiB on 3e0f0f18f: 4,435 no-step lines for a restore head,
/// need_moves 1 and "ended=none": a pass that found nothing to do returned
/// without saying where it stopped. Every such pass records its end.
#[tokio::test(flavor = "current_thread")]
async fn a_pass_with_nothing_to_do_says_where_it_stopped() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(6, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| {
        inner.procs.get_mut(&head).unwrap().state = Residency::Evicted;
        inner.parked_moving.insert(head);
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    planner.plan();
    let end = planner.with_inner(|inner| inner.last_plan.as_ref().map(|end| end.ended));
    assert_eq!(end, Some("its parked state moves down a tier: restored once it lands"));
}

/// The L40S at 3 GiB on f623134fc: a restore head needing 13 single chunks
/// with 12 free, one chunk holding residents whose state can park. No
/// chunk-level compaction makes a free chunk; the eviction of that chunk's
/// occupants does, and it is made, never a quiet wait.
#[tokio::test(flavor = "current_thread")]
async fn one_chunk_short_of_singles_evicts_its_parkable_occupants() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(14, 0..0, 64));
    pool.free.lock().extend(0..12);
    pool.store_refuses.store(true, Ordering::Relaxed);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, a, b) = (ProcessId::new_v4(), ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, a, b] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    pool.kv.lock().extend([(12, a), (12, b), (13, a)]);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(PendingBind {
            singles: 0,
            run: 1,
            pieces: 13,
            pages: 13,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    planner.poke();
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if planner.stats.evictions_started.load(Ordering::Relaxed) > 0 {
            break;
        }
    }
    let end = planner.with_inner(|inner| inner.last_plan.as_ref().map(|end| end.ended));
    assert!(
        planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
        "the one chunk's occupants are evicted ({end:?})"
    );
    assert_eq!(planner.stats.quiet_waits.load(Ordering::Relaxed), 0);
}

/// Mac Pro at 2 GiB on c10d7de5d: a restore with a bind's run of 7 and
/// many kv pages to gather asked them as runs of 7 ((7, 1) grew to (7,
/// 31), past the 30 free runs on a device 310 of 365 pages free), and the
/// pass ended "no pick". Its pages are single chunks beside its run.
#[test]
fn a_restore_s_pages_to_gather_are_single_chunks_beside_its_runs() {
    let pool = Arc::new(BindRoomPool::new(64, 0..0, 64));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    let key = planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.roles_short = Some(PendingBind {
            singles: 0,
            run: 7,
            pieces: 1,
            pages: 7,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand {
                    kv_pages: 30,
                    rs: RsUnits::default(),
                },
            },
        )
    });
    let mut need = super::pool::RunNeed::one(7, 1);
    need.add(1, 30);
    assert_eq!(planner.head_room(key), Some(need));
}


/// Mac Pro at 2 GiB (g2): free runs of three at 0-2 and 4-6, the
/// restore's own take opening chunk 0, its pending bind asking two runs of
/// three. Read as a mirror of the takes, the map laid the bind in both
/// free runs, need_moves 0, while the restore's own take left one
/// (PendingBindShort { run: 3, pieces: 2, free_runs_after_own_unpark: 1 }),
/// and the head spun. Its own take placed first by the one placement, the
/// map's need is the bind beside it: one move (the resident on chunk 3),
/// and the move is issued.
#[tokio::test(flavor = "current_thread")]
async fn a_restore_s_own_take_consumes_a_run_and_its_move_is_issued() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(8, 0..0, 64)
    });
    pool.free.lock().extend([0, 1, 2, 4, 5, 6]);
    pool.own.lock().push(0);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, resident] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    pool.kv.lock().extend([(3, resident), (7, resident)]);
    let key = planner.with_inner(|inner| {
        inner.procs.get_mut(&resident).unwrap().idle = true;
        let proc = inner.procs.get_mut(&head).unwrap();
        proc.state = Residency::Evicted;
        proc.bind_room = Some(super::proc::PendingBind {
            singles: 0,
            run: 3,
            pieces: 2,
            pages: 6,
        });
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        )
    });
    assert!(pool.guest_room_free(3, 2), "two free runs of three before its own take");
    let need = planner.head_room(key).expect("a need of runs");
    let placement = planner.head_placement(key, &need).expect("a restore head");
    assert!(!placement.stands(), "its own take leaves one run of three");
    let mut map = pool.chunk_occupants(&[resident]);
    placement.lay(&mut map);
    assert_eq!(
        map.cheapest_need(&placement.short).map(|pids| pids.len()),
        Some(1),
        "need_moves 1: the resident on chunk 3"
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
        "the move is issued"
    );
}

/// A stall line names its own head's last end: another entry's end is not
/// its, and its own end, recorded under the entry it was queued as before
/// (a refused restore queued again), is still its own, named so.
#[test]
fn a_stall_line_names_its_own_head_s_last_end() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(8))));
    let (a, b) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [a, b] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let restore = || WaitKind::Restore {
        demand: Demand::default(),
    };
    let (key_a, key_b) = planner.with_inner(|inner| (inner.enqueue(a, restore()), inner.enqueue(b, restore())));
    planner.plan_ended(key_a, "no pick: the policy is asked", false, &[]);
    let (plan_b, _) = planner.with_inner(|inner| inner.plan_fields(key_b));
    assert!(plan_b.starts_with("ended=none"), "another entry's end is not its: {plan_b}");
    let (plan_a, _) = planner.with_inner(|inner| inner.plan_fields(key_a));
    assert!(plan_a.contains("no pick"), "{plan_a}");
    // Queued again under a new entry: its own end, named as its old entry's.
    let again = planner.with_inner(|inner| {
        inner.queue.remove(&key_a);
        inner.enqueue(a, restore())
    });
    let (plan_again, _) = planner.with_inner(|inner| inner.plan_fields(again));
    assert!(plan_again.contains("no pick") && plan_again.contains("as entry"), "{plan_again}");
}

/// The L40S at 3 GiB on c201812ea: a restore refused (PendingBindShort)
/// while others' takes raced it, its placement standing in the room its
/// refusal left, held "until more chunks stand free" with nothing in
/// flight: 186 s, no pass, nothing to free a chunk. With nothing in flight
/// no race comes again: it is sent. Refused with nothing else in flight,
/// a placement standing there is the placement's own mismatch: said as a
/// planner bug, and the head never waits on it: its need is planned as if
/// the placement did not stand, and the resident the map names is moved.
#[tokio::test(flavor = "current_thread")]
async fn a_raced_refusal_standing_with_nothing_in_flight_is_sent_again() {
    for quiet in [false, true] {
        let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
        let model = crate::store::registry::register_model(16, &[8], &[4]);
        let pool = Arc::new(BindRoomPool {
            model,
            ..BindRoomPool::new(8, 0..0, 64)
        });
        pool.free.lock().extend(0..6);
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
        let (head, resident) = (ProcessId::new_v4(), ProcessId::new_v4());
        for pid in [head, resident] {
            planner.register(pid);
            planner.note_admitted(pid);
        }
        pool.kv.lock().extend([(6, resident), (7, resident)]);
        let key = planner.with_inner(|inner| {
            inner.procs.get_mut(&resident).unwrap().idle = true;
            let proc = inner.procs.get_mut(&head).unwrap();
            proc.state = Residency::Evicted;
            proc.roles_short = Some(PendingBind {
                singles: 0,
                run: 1,
                pieces: 1,
                pages: 1,
            });
            proc.bind_room = proc.roles_short;
            proc.roles_short_free = Some(6);
            // Refused in the room as it stands: none given back since.
            proc.roles_short_gives = Some(PoolPort::room_gives(&*pool));
            proc.roles_short_quiet = quiet;
            inner.enqueue(
                head,
                WaitKind::Restore {
                    demand: Demand::default(),
                },
            )
        });
        assert!(planner.with_inner(|inner| inner.nothing_in_flight()));
        assert_eq!(planner.no_more_room_than_refused(head), quiet);
        planner.plan();
        let (plan, _) = planner.with_inner(|inner| inner.plan_fields(key));
        if quiet {
            assert!(!plan.contains("restore sent"), "not sent on the mismatch: {plan}");
            assert!(
                planner.stats.evictions_started.load(Ordering::Relaxed) > 0,
                "never a wait: the map's candidate is moved ({plan})"
            );
        } else {
            assert!(plan.contains("restore sent"), "nothing in flight: sent again: {plan}");
        }
    }
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

/// What a process was doing in the stall's state.
#[derive(Clone, Copy)]
enum SnapWait {
    /// Queued as a restore.
    Restore,
}

/// A process of the stall's state, by its place in [`StallSnapshot::procs`].
#[derive(Clone)]
struct SnapProc {
    state: Residency,
    queued: Option<SnapWait>,
    /// Its last refusal (`roles_short` / `bind_room`): run and pieces.
    refused: Option<(u32, u32)>,
    /// The free runs of the refusal's length its refusal left
    /// (`roles_short_free`), and whether it was refused with nothing else
    /// in flight (`roles_short_quiet`).
    refused_free: Option<u32>,
    quiet: bool,
}

impl SnapProc {
    fn resident() -> Self {
        Self {
            state: Residency::Resident,
            queued: None,
            refused: None,
            refused_free: None,
            quiet: false,
        }
    }

    fn evicted_restore() -> Self {
        Self {
            state: Residency::Evicted,
            queued: Some(SnapWait::Restore),
            ..Self::resident()
        }
    }
}

/// A stall's state, as its stall line records it.
struct StallSnapshot {
    /// The run that stalled, for the failure message.
    name: &'static str,
    /// The state chunks the pool holds (`state_limit`).
    chunks: u32,
    /// The chunks free now, and the budget's headroom.
    free: Vec<u32>,
    headroom: Option<u32>,
    /// Host slots (0: no host tier) and disk slots free.
    host: u32,
    disk_free: u32,
    procs: Vec<SnapProc>,
    /// Chunks held: the chunk, the slots it holds, and each holder's share
    /// (by its place in `procs`). One resident holding a chunk whole is its
    /// kv; anything else is shared slots.
    held: Vec<(u32, u32, Vec<(usize, u32)>)>,
    /// The queue's head, by its place in `procs`.
    head: usize,
    /// Whether a move down of evicted processes' state takes the kv they
    /// left on the device (the store's sweep).
    sweeps_shared: bool,
    /// The chunks the head's own restore retakes, single chunks (its
    /// unpark need, `RunNeed([(1, own.len())])`): those free it takes first.
    own: Vec<u32>,
}

/// Rebuilds `snap` as a pool and a planner and runs the planner's passes to
/// a fixpoint; `Err` names the head and its need where the state settles
/// with the head waiting and nothing in flight.
async fn replay(snap: &StallSnapshot) -> Result<(), String> {
    let model = crate::store::registry::register_model(16, &[8], &[4]);
    let pool = Arc::new(BindRoomPool {
        model,
        ..BindRoomPool::new(snap.chunks, 0..0, snap.host)
    });
    pool.free.lock().extend(snap.free.iter().copied());
    *pool.headroom.lock() = snap.headroom;
    pool.disk_free.store(snap.disk_free, Ordering::Relaxed);
    pool.sweeps_shared.store(snap.sweeps_shared, Ordering::Relaxed);
    pool.own.lock().extend(snap.own.iter().copied());
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let pids: Vec<ProcessId> = snap.procs.iter().map(|_| ProcessId::new_v4()).collect();
    for &pid in &pids {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    for (chunk, used, holders) in &snap.held {
        let whole_resident = matches!(holders.as_slice(), [(at, slots)]
            if *slots >= *used && snap.procs[*at].state == Residency::Resident);
        let by: Vec<(ProcessId, u32)> = holders.iter().map(|&(at, slots)| (pids[at], slots)).collect();
        if whole_resident {
            pool.kv.lock().push((*chunk, by[0].0));
        } else {
            pool.shared.lock().push(super::pool::ChunkHeld {
                chunk: *chunk,
                used: *used,
                by,
            });
        }
    }
    let keys: Vec<Option<EntryKey>> = planner.with_inner(|inner| {
        snap.procs
            .iter()
            .zip(&pids)
            .map(|(snap_proc, &pid)| {
                let proc = inner.procs.get_mut(&pid).unwrap();
                proc.state = snap_proc.state;
                proc.roles_short = snap_proc.refused.map(|(run, pieces)| PendingBind {
                    singles: 0,
                    run,
                    pieces,
                    pages: run * pieces,
                });
                proc.bind_room = proc.roles_short;
                proc.roles_short_free = snap_proc.refused_free;
                proc.roles_short_quiet = snap_proc.quiet;
                snap_proc.queued.map(|wait| {
                    inner.enqueue(
                        pid,
                        match wait {
                            SnapWait::Restore => WaitKind::Restore {
                                demand: Demand::default(),
                            },
                        },
                    )
                })
            })
            .collect()
    });
    let head = pids[snap.head];
    let head_key = keys[snap.head];
    // The passes, to a fixpoint: an unmoved generation for a while, with
    // whatever moves the pool's moves land landed.
    let (mut last, mut steady) = (0u64, 0u32);
    for _ in 0..5000 {
        planner.plan();
        crate::rt::yield_now().await;
        pool.land_moves(&planner);
        let generation = planner.generation();
        if generation == last {
            steady += 1;
            if steady >= 200 {
                break;
            }
        } else {
            (last, steady) = (generation, 0);
        }
    }
    planner.with_inner(|inner| {
        let served = inner.procs.get(&head).is_none_or(|proc| proc.state == Residency::Resident)
            || head_key.is_some_and(|key| !inner.queue.contains_key(&key));
        if served || !inner.nothing_in_flight() {
            return Ok(());
        }
        let proc = &inner.procs[&head];
        let (plan, _) = head_key.map_or_else(|| (String::new(), String::new()), |key| inner.plan_fields(key));
        Err(format!(
            "{}: the head {head} waits with nothing in flight: state {:?}, refused {:?} \
             (free runs it left {:?}), plan {plan}; free chunks {:?}, moved {:?}",
            snap.name,
            proc.state,
            proc.roles_short.map(|need| (need.run, need.pieces)),
            proc.roles_short_free,
            pool.free.lock(),
            pool.moved.lock(),
        ))
    })
}

/// The L40S at 3 GiB disk-only on d41e15f24 (serve-Y3_3s-1): a prompt-prefix
/// page seven evicted forks shared stayed on the device (chunk 4, one slot,
/// two refs each), the restore head among them short of a thirteenth
/// single chunk with twelve free. The map named the forks for the chunk,
/// none a candidate; the policy was asked with nothing in flight.
fn stall_d41e15f24() -> StallSnapshot {
    let mut procs = vec![SnapProc::evicted_restore(); 7];
    procs[0].refused = Some((1, 1));
    procs[0].refused_free = Some(12);
    StallSnapshot {
        name: "d41e15f24 serve-Y3_3s-1",
        chunks: 22,
        free: (5..17).collect(),
        headroom: Some(12),
        host: 0,
        disk_free: 4014,
        held: vec![(4, 1, (0..7).map(|at| (at, 2)).collect())],
        procs,
        head: 0,
        sweeps_shared: true,
        own: (5..17).chain([4]).collect(),
    }
}

/// The L40S at 3 GiB disk-only on c201812ea (serve-Z3_3s-1): the restore
/// head (26e5834e) refused PendingBindShort (run 1) while others' takes raced
/// it, one free run left and its placement standing in that room; held
/// "until more chunks stand free than its refusal left" with nothing in
/// flight, no event coming to end it. Beside it a resident candidate
/// (37b23987) holding slots across fifteen chunks, ten processes' prompt
/// prefix sharing chunk 1.
fn stall_c201812ea() -> StallSnapshot {
    // 0: 37b23987 (resident, the candidate); 1: 26e5834e (the head);
    // 2: 66751556; 3..=9: the prefix's other sharers.
    let mut procs = vec![SnapProc::resident()];
    procs.push(SnapProc {
        refused: Some((1, 1)),
        refused_free: Some(1),
        ..SnapProc::evicted_restore()
    });
    procs.extend(std::iter::repeat_n(SnapProc::evicted_restore(), 8));
    let all: Vec<(usize, u32)> = (0..10).map(|at| (at, 168)).collect();
    let mut chunk3: Vec<(usize, u32)> = (0..10).map(|at| (at, 1)).collect();
    chunk3[1].1 = 2;
    StallSnapshot {
        name: "c201812ea serve-Z3_3s-1",
        chunks: 22,
        free: vec![19],
        headroom: Some(1),
        host: 0,
        disk_free: 4014,
        held: vec![
            (1, 1680, all),
            (3, 11, chunk3),
            (6, 9, vec![(0, 1), (2, 8)]),
            (7, 599, vec![(0, 96), (1, 262), (2, 241)]),
            (8, 6, vec![(0, 2), (1, 2), (2, 2)]),
            (10, 9, vec![(0, 2), (1, 4)]),
            (11, 8, vec![(0, 3)]),
            (12, 9, vec![(0, 5), (2, 4)]),
            (13, 9, vec![(0, 2), (2, 4)]),
            (15, 3, vec![(0, 3)]),
            (16, 9, vec![(0, 1), (1, 4), (2, 1)]),
            (17, 9, vec![(0, 1), (2, 8)]),
            (18, 9, vec![(0, 1), (2, 8)]),
            (20, 9, vec![(0, 9)]),
            (21, 9, vec![(0, 5), (2, 4)]),
        ],
        procs,
        head: 1,
        sweeps_shared: true,
        own: Vec::new(),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn replay_d41e15f24_shared_prefix_page_left_by_evicted_forks() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    replay(&stall_d41e15f24()).await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn replay_c201812ea_head_beside_a_resident_candidate() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    replay(&stall_c201812ea()).await.unwrap();
}

/// The store as it was on d41e15f24 (a move down of evicted processes'
/// state left their kv on the device): the same state settles with the head
/// waiting and nothing in flight, which the invariant names.
#[tokio::test(flavor = "current_thread")]
async fn replay_names_a_head_left_waiting_when_shared_kv_is_not_swept() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let snap = StallSnapshot {
        sweeps_shared: false,
        ..stall_d41e15f24()
    };
    let err = replay(&snap).await.expect_err("the head waits with nothing in flight");
    assert!(err.contains("waits with nothing in flight"), "{err}");
}

/// The progress invariant, enforced: a settled pass that leaves a head
/// waiting with nothing in flight fails a strict test by name (here a bind
/// short of a run of two on scattered free chunks, no one to move and no
/// compaction to ask).
#[test]
#[should_panic(expected = "progress invariant")]
fn a_settled_head_with_nothing_in_flight_fails_a_strict_test() {
    super::plan::PROGRESS_STRICT.with(|strict| strict.set(true));
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 0));
    pool.free.lock().extend([0, 2, 4, 6]);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| {
        inner.enqueue(
            head,
            WaitKind::BindRoom {
                run: 2,
                pieces: 1,
                singles: 0,
                pages: 2,
                notify: Arc::new(Notify::new()),
                served: false,
                withdrawn: false,
                rise: None,
            },
        );
    });
    planner.plan();
}

/// A process the policy holds below its floor is never restored, nor any
/// room held for it, on any path: a forced pass (the progress invariant's)
/// plans moves only, through the victim set in the policy's order.
#[tokio::test(flavor = "current_thread")]
async fn a_forced_pass_never_restores_or_reserves_for_a_held_process() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    pool.free.lock().extend(0..8);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let held = ProcessId::new_v4();
    planner.register(held);
    planner.note_admitted(held);
    planner.with_inner(|inner| {
        let proc = inner.procs.get_mut(&held).unwrap();
        proc.state = Residency::Evicted;
        proc.held = true;
        inner.enqueue(
            held,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
    });
    planner.force_moves.store(true, Ordering::Release);
    planner.plan();
    assert_eq!(planner.stats.restores.load(Ordering::Relaxed), 0, "never restored");
    assert_eq!(
        planner.with_inner(|inner| inner.procs[&held].state),
        Residency::Evicted,
        "still held, evicted"
    );
}

/// The L40S at 3 GiB on c8077be0f (scratchpad
/// stalls/l40s-c807-forced.txt, lines 1 and 2): forced passes said with
/// evictions and a restore in flight, and after "kv on its way". A force
/// spans the one pass after its stall: something in flight by then is
/// progress, the pass plans as any other, and the force is gone after it.
#[tokio::test(flavor = "current_thread")]
async fn a_force_with_something_in_flight_is_no_forced_pass_and_lasts_one_pass() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(BindRoomPool::new(8, 0..0, 64));
    pool.free.lock().extend(0..8);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (head, moving) = (ProcessId::new_v4(), ProcessId::new_v4());
    for pid in [head, moving] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    planner.with_inner(|inner| {
        inner.procs.get_mut(&head).unwrap().state = Residency::Evicted;
        inner.enqueue(
            head,
            WaitKind::Restore {
                demand: Demand::default(),
            },
        );
        // Another process's eviction copy in flight.
        inner.procs.get_mut(&moving).unwrap().state = Residency::Evicting;
        inner.evicting.insert(moving, Reclaim::default());
    });
    planner.force_moves.store(true, Ordering::Release);
    planner.plan();
    assert_eq!(
        planner.stats.forced_passes.load(Ordering::Relaxed),
        0,
        "an eviction in flight: no forced pass"
    );
    assert!(
        !planner.force_moves.load(Ordering::Acquire),
        "the force lasts the one pass"
    );
}

/// Mac Pro at 2 GiB (g2) on c201812ea, serves 10,594 (scratchpad
/// stalls/macpro-c2018-g2.txt): a restore head, need [(7, 2), (2, 2)], its
/// own takes opening four chunks (its two runs of two), the placement
/// standing two runs of seven after them while its refusal found one
/// (PendingBindShort { run: 7, pieces: 2, free_runs_after_own_unpark: 1 }),
/// held "until more chunks stand free" with nothing in flight; eighteen
/// residents' chunks emptiable around. The line carries no per-chunk
/// layout: three free runs of seven, the head's own four in the first, the
/// rest held whole by residents. `quiet`: refused with nothing else in
/// flight (the placement's own mismatch) or raced (another head's take
/// between its serve and its unpark, which the reservation now prevents).
fn stall_macpro_c2018_g2(quiet: bool) -> StallSnapshot {
    let residents = 18usize;
    let mut procs = vec![SnapProc {
        refused: Some((7, 2)),
        refused_free: Some(21),
        quiet,
        ..SnapProc::evicted_restore()
    }];
    procs.extend(std::iter::repeat_n(SnapProc::resident(), residents));
    let free: Vec<u32> = (0..7).chain(10..17).chain(20..27).collect();
    let held: Vec<(u32, u32, Vec<(usize, u32)>)> = (0..54u32)
        .filter(|chunk| !free.contains(chunk))
        .enumerate()
        .map(|(at, chunk)| (chunk, 9, vec![(1 + at % residents, 9)]))
        .collect();
    StallSnapshot {
        name: "c201812ea Mac Pro g2",
        chunks: 54,
        free,
        headroom: Some(21),
        host: 0,
        disk_free: 4000,
        held,
        procs,
        head: 0,
        sweeps_shared: true,
        own: vec![0, 1, 2, 3],
    }
}

#[tokio::test(flavor = "current_thread")]
async fn replay_macpro_c2018_g2_raced_refusal() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    replay(&stall_macpro_c2018_g2(false)).await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn replay_macpro_c2018_g2_quiet_mismatch() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    replay(&stall_macpro_c2018_g2(true)).await.unwrap();
}

/// Mac Pro at 5 GiB (mx) on c201812ea, serves 1,307 (scratchpad
/// stalls/macpro-c2018-mx.txt): a restore head, need [(3, 1), (2, 2)], its
/// own two runs of two in the lowest two-stretches (segments 2 and 4), the
/// two runs of three (a four-stretch, a three-stretch) left standing, and
/// its refusal's PendingBindShort { run: 3, pieces: 1, free_runs: 0 } taken
/// on another pool (refused_on_this_pool=false): held with nothing in
/// flight. As the line lays it out, scaled to sixty chunks.
fn stall_macpro_c2018_mx(quiet: bool) -> StallSnapshot {
    let residents = 12usize;
    let mut procs = vec![SnapProc {
        refused: Some((3, 1)),
        refused_free: Some(11),
        quiet,
        ..SnapProc::evicted_restore()
    }];
    procs.extend(std::iter::repeat_n(SnapProc::resident(), residents));
    let free: Vec<u32> = vec![4, 5, 8, 9, 20, 21, 22, 23, 40, 41, 42];
    let held: Vec<(u32, u32, Vec<(usize, u32)>)> = (0..60u32)
        .filter(|chunk| !free.contains(chunk))
        .enumerate()
        .map(|(at, chunk)| (chunk, 9, vec![(1 + at % residents, 9)]))
        .collect();
    StallSnapshot {
        name: "c201812ea Mac Pro mx",
        chunks: 60,
        free,
        headroom: Some(11),
        host: 0,
        disk_free: 4000,
        held,
        procs,
        head: 0,
        sweeps_shared: true,
        own: vec![4, 5, 8, 9],
    }
}

#[tokio::test(flavor = "current_thread")]
async fn replay_macpro_c2018_mx_raced_refusal() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    replay(&stall_macpro_c2018_mx(false)).await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn replay_macpro_c2018_mx_quiet_mismatch() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    replay(&stall_macpro_c2018_mx(true)).await.unwrap();
}

/// Mac Pro g2 on bd4f4ffe7 (scratchpad stalls/macpro-bd4f-g2.txt): a restore
/// refused (PendingBindShort) on every serve with the room unchanged, its
/// own re-send moving the generation, so no pass settled: 394k passes, no
/// forced pass, no strict failure. The same refusal again on the same room
/// is a settled stall: a strict test fails by name; in service it is read
/// as the placement's mismatch, not sent again until the room moves, its
/// need planned on chunks only moves empty.
#[test]
fn a_restore_refused_again_on_the_same_room_is_a_settled_stall() {
    let pool = Arc::new(BindRoomPool::new(16, 0..0, 64));
    pool.free.lock().extend(0..10);
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    planner.with_inner(|inner| inner.procs.get_mut(&head).unwrap().state = Residency::Restoring);
    planner.requeue_restore_roles(head, (7, 2), Some(0));
    assert_eq!(planner.stats.refusal_repeats.load(Ordering::Relaxed), 0, "a first refusal is no stall");
    planner.with_inner(|inner| inner.procs.get_mut(&head).unwrap().state = Residency::Restoring);
    planner.requeue_restore_roles(head, (7, 2), Some(0));
    assert_eq!(planner.stats.refusal_repeats.load(Ordering::Relaxed), 1, "the same again on the same room is");
    assert!(
        planner.with_inner(|inner| inner.procs[&head].roles_short_quiet),
        "read as the placement's mismatch: held from a send until the room moves"
    );
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::plan::PROGRESS_STRICT.with(|strict| strict.set(true));
        planner.with_inner(|inner| inner.procs.get_mut(&head).unwrap().state = Residency::Restoring);
        planner.requeue_restore_roles(head, (7, 2), Some(0));
    }));
    super::plan::PROGRESS_STRICT.with(|strict| strict.set(false));
    assert!(caught.is_err(), "a strict test fails on it");
}

/// A fire held at a bind the store has not served is no fire in flight:
/// a head waiting behind it with nothing else under way is a stall the
/// invariant sees, never "waits for the restored one's bind to fire".
#[test]
fn a_fire_held_at_its_bind_is_not_in_flight() {
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RacePool::new(8))));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    let _fire = planner.fire_in_flight(pid);
    assert!(!planner.with_inner(|inner| inner.nothing_in_flight()), "a fire submitted is");
    planner.with_inner(|inner| inner.procs.get_mut(&pid).unwrap().bind_waiting = true);
    assert!(planner.with_inner(|inner| inner.nothing_in_flight()), "one held at its bind is not");
}

/// The L40S at 3 GiB disk-only on bd4f4ffe7 (serve-B4_3s-1, 998 lines of
/// it): a restore head (6ae0bc15) refused PendingBindShort (run 1) with
/// nothing else in flight, one free run left, its placement standing in the
/// same room ("a planner bug: the placement and the takes differ"): held,
/// with nothing in flight.
fn stall_bd4f4ffe7_quiet_mismatch() -> StallSnapshot {
    StallSnapshot {
        name: "bd4f4ffe7 serve-B4_3s-1",
        chunks: 22,
        free: vec![9],
        headroom: Some(1),
        host: 0,
        disk_free: 4014,
        procs: vec![SnapProc {
            refused: Some((1, 1)),
            refused_free: Some(1),
            quiet: true,
            ..SnapProc::evicted_restore()
        }],
        held: Vec::new(),
        head: 0,
        sweeps_shared: true,
        own: Vec::new(),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn replay_bd4f4ffe7_quiet_refusal_standing_in_its_placement() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    replay(&stall_bd4f4ffe7_quiet_mismatch()).await.unwrap();
}

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
