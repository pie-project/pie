use super::*;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::inner::RESERVE_SHARE;
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
                inner.enqueue(pid, allocation(Demand { kv_pages, rs: RsUnits::default() }))
            })
            .collect()
    })
}

fn unfailed(planner: &ResidencyPlanner, keys: &[EntryKey]) -> usize {
    planner.with_inner(|inner| {
        keys.iter()
            .filter(|key| matches!(inner.queue[key].kind, WaitKind::Allocation { outcome: None, .. }))
            .count()
    })
}

#[test]
fn a_departure_s_pages_are_waited_for_before_an_ask_holding_nothing_fails() {
    // The finished process's working set still holds its pages: nothing
    // fails while it does. Once it is released, a stall that remains is a
    // wedge, and the youngest ask fails as before.
    let pool = Arc::new(RacePool::new(8));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let keys = dry_pool_two_asks(&planner);
    let model = crate::store::registry::register_model(16, &[64], &[2]);
    let gone = crate::store::registry::get(model, 0).kv.lock().create_working_set();
    planner.with_inner(|inner| inner.releasing.insert(gone));
    assert!(planner.is_wedged());
    planner.check_starvation(StarveCause::NoSwapRoom);
    assert_eq!(unfailed(&planner, &keys), 2, "an ask fails while a departure's pages come back");
    planner.working_set_released(gone);
    planner.check_starvation(StarveCause::NoSwapRoom);
    planner.with_inner(|inner| {
        assert!(matches!(inner.queue[&keys[0]].kind, WaitKind::Allocation { outcome: None, .. }));
        assert!(matches!(
            inner.queue[&keys[1]].kind,
            WaitKind::Allocation { outcome: Some(Err(PlannerError::Starved { .. })), .. }
        ));
    });
}

#[test]
fn a_departure_held_behind_a_long_fire_is_waited_for_however_long() {
    // The departed process's working set is released while a fire still
    // leases it (a launch can take minutes): its pages come back only when
    // the lease drops. No clock is involved: the planner waits for the
    // release itself, so the empty asker does not fail before it.
    let model = crate::store::registry::register_model(16, &[64], &[2]);
    let stores = crate::store::registry::get(model, 0);
    let id = stores.kv.lock().create_working_set();
    crate::planner::init_planner(model, 0, ResidencyPlanner::new(Arc::new(RacePool::new(8)) as Arc<dyn PoolPort>));
    let planner = crate::planner::planner_for(model, 0).expect("the test model's planner");
    let keys = dry_pool_two_asks(&planner);
    let ws = crate::store::kv::working_set::KvWorkingSet::new(model, 0, id, 16);
    let lease = ws.fire_lease().expect("a fire leases the working set");
    planner.with_inner(|inner| inner.releasing.insert(id));
    ws.release();
    for _ in 0..100 {
        planner.check_starvation(StarveCause::NoSwapRoom);
    }
    assert_eq!(unfailed(&planner, &keys), 2, "an ask fails while the departure's release waits on a fire");
    assert!(planner.with_inner(|inner| inner.releasing.contains(&id)));
    drop(lease);
    assert!(planner.with_inner(|inner| inner.releasing.is_empty()), "the fire's end releases the working set");
    planner.check_starvation(StarveCause::NoSwapRoom);
    assert_eq!(unfailed(&planner, &keys), 1, "a stall that outlasts the release is a wedge");
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
            reach: held,
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

#[test]
fn a_seated_lane_grows_before_a_prompt_waiting_to_be_seated() {
    let planner = Arc::new(
        ResidencyPlanner::new(Arc::new(RacePool::new(8)) as Arc<dyn PoolPort>).seated_first(true),
    );
    let [older, seated, younger] = [(); 3].map(|_| ProcessId::new_v4());
    for pid in [older, seated, younger] {
        planner.register(pid);
        planner.note_admitted(pid);
    }
    let ring = RsUnits { slots: 0, pages: 2 };
    let prompt = RsUnits { slots: 0, pages: 3 };
    planner.with_inner(|inner| {
        inner.procs.get_mut(&seated).unwrap().seated = true;
        let roomy = RsUnits {
            slots: 0,
            pages: 320,
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
                    pages: 32
                }
            ),
            RsUnits { slots: 0, pages: 1 },
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
    fn device_stats(&self) -> (u32, u32) {
        (0, 0)
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
            },
            RsUnits {
                slots: 0,
                pages: self.total,
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
    fn suspend_capable(&self) -> bool {
        false
    }
    fn locus(&self) -> (usize, usize) {
        (0, 0)
    }
    fn rs_growth(&self) -> RsUnits {
        RsUnits { slots: 0, pages: 2 }
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
    fn yield_checkpoints(&self, _working_sets: &[crate::store::rs::RsWorkingSetId]) -> usize {
        0
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

/// Prompts seated at once each see the others' pages gone: the ring keeps
/// the seated lanes' growth however many arrive together.
#[test]
fn prompts_seated_at_once_leave_the_seated_lanes_their_growth() {
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
