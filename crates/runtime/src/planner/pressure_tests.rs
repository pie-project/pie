//! A waiter short of room only the policy can make: it parks, says so as tier
//! pressure while it waits, and is served once the policy lets something go.
//! And admission charges the state pool, so no cap is needed to keep it.

use super::*;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use futures::FutureExt;

use crate::policy::pressure::{Band, Tier};
use crate::store::kv::page_table::PhysicalKvPageId;
use crate::store::rs::RsUnits;

use super::grant::Demand;
use super::short::Sink;

/// A pool whose pages idle cache keeps under keys above 0: the runtime may
/// not drop them, only the policy (`discard`).
struct KeptPool {
    free: parking_lot::Mutex<Vec<PhysicalKvPageId>>,
    kept: parking_lot::Mutex<Vec<PhysicalKvPageId>>,
    total: u32,
    model: usize,
    /// A shared pool's state shapes: `(free, total)` of them, and the kv
    /// pages whose chunks hold one unit of state (`rs_as_pages`).
    shared_state: Option<((RsUnits, RsUnits), u32)>,
    /// A shared pool's division: weights and activation hold all but
    /// `room.list_slots` pages for good.
    room: Option<crate::store::shared::PoolRoom>,
}

impl KeptPool {
    fn new(free: u32, kept: u32, model: usize) -> Self {
        Self {
            free: parking_lot::Mutex::new((0..free).map(PhysicalKvPageId).collect()),
            kept: parking_lot::Mutex::new((free..free + kept).map(PhysicalKvPageId).collect()),
            total: free + kept,
            model,
            shared_state: None,
            room: None,
        }
    }

    /// A shared pool whose weights and activation leave `room` for kv.
    fn dividing(mut self, room: crate::store::shared::PoolRoom) -> Self {
        self.room = Some(room);
        self
    }

    /// State drawn from the kv pages' chunks: `stats` of it, each unit
    /// worth `pages` kv pages parked.
    fn sharing_state(mut self, stats: (RsUnits, RsUnits), pages: u32) -> Self {
        self.shared_state = Some((stats, pages));
        self
    }

    /// The policy's `discard`: what the cache kept is free.
    fn discard(&self) {
        let pages = std::mem::take(&mut *self.kept.lock());
        self.free.lock().extend(pages);
    }

    fn with_rs<R>(&self, f: impl FnOnce(&mut crate::store::rs::RsStore) -> R) -> Option<R> {
        let stores = crate::store::registry::try_get(self.model, 0)?;
        let mut rs = stores.rs.lock().unwrap();
        Some(f(&mut rs))
    }
}

impl PoolPort for KeptPool {
    fn chunk_stats(&self) -> ChunkStats {
        ChunkStats {
            pages: self.device_stats(),
            rs: self.rs_stats(),
        }
    }
    fn device_stats(&self) -> (u32, u32) {
        let total = self
            .room
            .map_or(self.total, |room| self.total.min(room.list_slots));
        (self.free.lock().len() as u32, total)
    }
    fn room(&self) -> Option<crate::store::shared::PoolRoom> {
        self.room
    }
    fn host_stats(&self) -> (u32, u32) {
        (0, 0)
    }
    fn disk_stats(&self) -> (u32, u32) {
        (0, 0)
    }
    fn rs_stats(&self) -> (RsUnits, RsUnits) {
        if let Some((stats, _)) = self.shared_state {
            return stats;
        }
        self.with_rs(|rs| (rs.free(), rs.capacity()))
            .unwrap_or_default()
    }
    fn rs_as_pages(&self, rs: RsUnits) -> u32 {
        self.shared_state
            .map_or(0, |(_, pages)| (rs.slots + rs.pages) * pages)
    }
    fn rs_host_stats(&self) -> (RsUnits, RsUnits) {
        Default::default()
    }
    /// Every key is above 0: nothing goes unasked.
    fn reclaim_idle(&self) -> u32 {
        0
    }
    fn reclaimable_idle(&self) -> u32 {
        self.kept.lock().len() as u32
    }
    fn suspend_capable(&self) -> bool {
        false
    }
    fn locus(&self) -> (usize, usize) {
        (self.model, 0)
    }
    fn reserve_device(&self, count: u32) -> Option<Vec<PhysicalKvPageId>> {
        let mut free = self.free.lock();
        let at = free.len().checked_sub(count as usize)?;
        Some(free.split_off(at))
    }
    fn reserve_device_up_to(&self, count: u32) -> Vec<PhysicalKvPageId> {
        let mut free = self.free.lock();
        let at = free.len().saturating_sub(count as usize);
        free.split_off(at)
    }
    fn release_device(&self, pages: Vec<PhysicalKvPageId>) {
        self.free.lock().extend(pages);
    }
    fn reserve_rs(&self, need: RsUnits) -> Option<crate::store::rs::RsGrant> {
        if let Some(((free, _), _)) = self.shared_state
            && !free.covers(need)
        {
            return None;
        }
        self.with_rs(|rs| rs.reserve(need))
            .unwrap_or_else(|| Some(Default::default()))
    }
    fn release_rs(&self, grant: crate::store::rs::RsGrant) {
        self.with_rs(|rs| rs.release_reservation(grant));
    }
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

type Said = Arc<parking_lot::Mutex<Vec<(Tier, u64)>>>;

fn recording() -> (Sink, Said) {
    let said: Said = Arc::default();
    let sink = {
        let said = said.clone();
        Arc::new(move |tier, bytes| said.lock().push((tier, bytes))) as Sink
    };
    (sink, said)
}

/// The band crossings a policy hears for what was said on `tier`: one
/// `tier-pressure` edge each.
fn crossings(said: &[(Tier, u64)], tier: Tier) -> Vec<Band> {
    let mut band = Band::Clear;
    let mut edges = Vec::new();
    for &(_, short) in said.iter().filter(|(at, _)| *at == tier) {
        let next = Band::of(band, short, 0);
        if next != band {
            edges.push(next);
        }
        band = next;
    }
    edges
}

#[tokio::test(flavor = "current_thread")]
async fn a_short_waiter_raises_once_per_band_crossing_and_is_served_after_a_discard() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(KeptPool::new(0, 4, usize::MAX));
    let (sink, said) = recording();
    let planner =
        Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>).pressure_sink(sink));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);

    let demand = Demand {
        kv_pages: 2,
        rs: RsUnits::default(),
    };
    let p = planner.clone();
    let parked = crate::rt::spawn(async move { p.acquire(head, head, demand).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if !said.lock().is_empty() {
            break;
        }
    }
    assert_eq!(planner.diagnostics().queue.len(), 1, "it parks");
    assert_eq!(
        planner.diagnostics().starvations_total,
        0,
        "the policy could free room: nobody is restarted for it"
    );
    // It says so again while it waits: the gauge keeps a band only while told.
    tokio::time::sleep(Duration::from_millis(350)).await;
    let raised = said.lock().clone();
    assert!(
        raised.len() >= 3,
        "raised again while parked, not once: {raised:?}"
    );
    let short = raised[0].1;
    assert!(short > 0);
    assert!(
        raised.iter().all(|&said| said == (Tier::Device, short)),
        "the device alone, short of the same: {raised:?}"
    );
    assert_eq!(crossings(&raised, Tier::Device), [Band::Short]);

    // The policy's answer: a discard.
    pool.discard();
    planner.pages_freed();
    let acquired = tokio::time::timeout(Duration::from_secs(5), parked)
        .await
        .expect("served")
        .unwrap();
    assert!(matches!(acquired, Ok(Acquired::Granted(_))));
    let told = said.lock().clone();
    assert_eq!(
        told.last(),
        Some(&(Tier::Device, 0)),
        "relieved once served"
    );
    assert_eq!(
        crossings(&told, Tier::Device),
        [Band::Short, Band::Clear],
        "one tier-pressure per band crossing: {told:?}"
    );
    planner.report_short();
    assert_eq!(said.lock().len(), told.len(), "a clear tier is told once");
    assert_eq!(planner.diagnostics().queue.len(), 0);
}

/// What a parked waiter's planner did, in order: a shed of the weight cache
/// (and what it gave back), or a tier-pressure said.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Shed(u64),
    Said(Tier, u64),
}

#[tokio::test(flavor = "current_thread")]
async fn a_short_waiter_sheds_the_weight_cache_before_it_raises_tier_pressure() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let pool = Arc::new(KeptPool::new(0, 4, usize::MAX));
    let steps: Arc<parking_lot::Mutex<Vec<Step>>> = Arc::default();
    let sink = {
        let steps = steps.clone();
        Arc::new(move |tier, bytes| steps.lock().push(Step::Said(tier, bytes))) as Sink
    };
    // Three atoms above the floor: the first shed gives them all back.
    let cached = Arc::new(std::sync::atomic::AtomicU64::new(3));
    let shed: super::WeightShed = {
        let (steps, cached) = (steps.clone(), cached.clone());
        Arc::new(move || {
            let shed = cached.swap(0, std::sync::atomic::Ordering::AcqRel);
            steps.lock().push(Step::Shed(shed));
            shed
        })
    };
    let planner = Arc::new(
        ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>)
            .pressure_sink(sink)
            .weight_shed(shed),
    );
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    let demand = Demand {
        kv_pages: 2,
        rs: RsUnits::default(),
    };
    let p = planner.clone();
    let parked = crate::rt::spawn(async move { p.acquire(head, head, demand).await });
    tokio::time::sleep(Duration::from_millis(350)).await;
    let steps = steps.lock().clone();
    let first_said = steps
        .iter()
        .position(|step| matches!(step, Step::Said(Tier::Device, bytes) if *bytes > 0))
        .expect("the policy hears of it once the cache is at its floor");
    assert_eq!(
        steps[0],
        Step::Shed(3),
        "the cache goes first, before any tier-pressure: {steps:?}"
    );
    assert!(
        steps[..first_said].contains(&Step::Shed(0)),
        "tier-pressure only after a shed found nothing above the floor: {steps:?}"
    );
    assert!(
        steps[first_said..]
            .iter()
            .all(|step| !matches!(step, Step::Shed(n) if *n > 0)),
        "nothing above the floor was left once pressure was said: {steps:?}"
    );
    parked.abort();
}

#[test]
fn a_waiter_the_planner_funds_itself_raises_nothing() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let pool = Arc::new(KeptPool::new(4, 0, usize::MAX));
    let (sink, said) = recording();
    let planner =
        Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>).pressure_sink(sink));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    let demand = Demand {
        kv_pages: 2,
        rs: RsUnits::default(),
    };
    let acquired = planner
        .acquire(head, head, demand)
        .now_or_never()
        .expect("free pages serve it at once");
    assert!(matches!(acquired, Ok(Acquired::Granted(_))));
    planner.plan();
    assert!(said.lock().is_empty(), "nothing short, nothing said");
}

#[test]
fn admission_charges_one_fixed_set_a_process_beside_the_lanes_copy_reserve() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    // Five state slots and one lane a fire: one slot stays back for the
    // lane's copy-on-write, and four processes each hold the set they run.
    let model = crate::store::registry::register_model(16, &[8], &[5]);
    let planner = Arc::new(
        ResidencyPlanner::new(Arc::new(KeptPool::new(8, 0, model)) as Arc<dyn PoolPort>)
            .cow_lanes(1),
    );
    assert_eq!(planner.state_seats(), Some(4));
    let pids = [(); 6].map(|_| ProcessId::new_v4());
    for pid in pids {
        planner.register(pid);
    }
    for &pid in &pids[..4] {
        assert!(planner.admit(pid).now_or_never().is_some());
    }
    let mut fifth = Box::pin(planner.admit(pids[4]));
    assert!(
        fifth.as_mut().now_or_never().is_none(),
        "a fifth would take the lane's copy-on-write set"
    );
    assert!(
        planner.admit(pids[0]).now_or_never().is_some(),
        "one charged already is not charged again"
    );
    planner.unregister(pids[0]);
    assert!(
        fifth.as_mut().now_or_never().is_some(),
        "one leaves: the next runs"
    );
    let mut sixth = Box::pin(planner.admit(pids[5]));
    assert!(sixth.as_mut().now_or_never().is_none());
    planner.unregister(pids[1]);
    assert!(sixth.as_mut().now_or_never().is_some());

    // A fire's lanes all copying: what is left after their reserve, and
    // never under one process.
    let wide = ResidencyPlanner::new(Arc::new(KeptPool::new(8, 0, model)) as Arc<dyn PoolPort>);
    assert_eq!(wide.cow_lanes(3).state_seats(), Some(2));
    let wider = ResidencyPlanner::new(Arc::new(KeptPool::new(8, 0, model)) as Arc<dyn PoolPort>);
    assert_eq!(wider.cow_lanes(64).state_seats(), Some(1));
}

#[test]
fn a_model_without_fixed_state_is_not_charged() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let planner = Arc::new(ResidencyPlanner::new(
        Arc::new(KeptPool::new(8, 0, usize::MAX)) as Arc<dyn PoolPort>,
    ));
    assert_eq!(planner.state_seats(), None);
    for _ in 0..8 {
        let pid = ProcessId::new_v4();
        planner.register(pid);
        assert!(planner.admit(pid).now_or_never().is_some());
    }
}

/// On a shared pool the state a waiter lacks comes from chunks kv holds:
/// the planner counts it as kv pages to park, so it moves idle holders out
/// rather than leave the policy to discard them.
#[tokio::test(flavor = "current_thread")]
async fn state_short_on_a_shared_pool_is_kv_for_the_planner_to_park() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    // No state free of 23 rs list pages; each takes 2 kv pages parked.
    let stats = (RsUnits::default(), RsUnits::of(0, 23));
    // Idle cache the policy may let go: the head waits on it, not on a restart.
    let pool = Arc::new(KeptPool::new(4, 2, usize::MAX).sharing_state(stats, 2));
    let (sink, _said) = recording();
    let planner =
        Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>).pressure_sink(sink));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    let demand = Demand {
        kv_pages: 1,
        rs: RsUnits::of(0, 5),
    };
    let p = planner.clone();
    let parked = crate::rt::spawn(async move { p.acquire(head, head, demand).await });
    for _ in 0..200 {
        crate::rt::yield_now().await;
        if !planner.diagnostics().queue.is_empty() {
            break;
        }
    }
    let victims = planner.victim_set().expect("the head is short");
    assert_eq!(victims.deficit.rs, RsUnits::default(), "no state left owed");
    assert_eq!(
        victims.rs_pages, 10,
        "five rs pages as ten kv pages to park"
    );
    assert_eq!(victims.deficit.pages, 10, "the head's own kv is free");

    // Without a shared pool the state is its own: no kv move frees it.
    let own = Arc::new(KeptPool::new(4, 0, usize::MAX));
    assert_eq!(own.rs_as_pages(RsUnits::of(0, 5)), 0);
    parked.abort();
}

/// A device and host of idle cache, and what the planner asked to move of
/// it: `("spill" | "demote", room)` in order.
struct TieredPool {
    host: (u32, u32),
    disk: (u32, u32),
    asked: parking_lot::Mutex<Vec<(&'static str, usize)>>,
}

impl PoolPort for TieredPool {
    fn chunk_stats(&self) -> ChunkStats {
        ChunkStats {
            pages: (0, 64),
            rs: Default::default(),
        }
    }
    fn host_stats(&self) -> (u32, u32) {
        self.host
    }
    fn disk_stats(&self) -> (u32, u32) {
        self.disk
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
        true
    }
    fn locus(&self) -> (usize, usize) {
        (usize::MAX, 0)
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
    fn prepare_demote(&self, room: usize) -> Option<crate::store::kv::KvResidencyTxn> {
        self.asked.lock().push(("demote", room));
        None
    }
    fn prepare_spill(&self, room: usize) -> Option<crate::store::kv::KvSpillTxn> {
        self.asked.lock().push(("spill", room));
        None
    }
}

/// A host with no room past its reserve is under pressure: its own idle
/// cache is asked to go on down to disk (a reserve's worth, at least)
/// before the device's is demoted past it. A host with room is not.
#[test]
fn a_host_short_of_room_spills_its_cache_to_disk_before_the_device_demotes() {
    let asked = |host: (u32, u32), disk: (u32, u32)| {
        let pool = Arc::new(TieredPool {
            host,
            disk,
            asked: Default::default(),
        });
        let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
        assert!(!planner.demote_idle(), "nothing moves: none was offered");
        std::mem::take(&mut *pool.asked.lock())
    };
    // 80 host slots keep 10 back: 8 free is short of room.
    assert_eq!(
        asked((8, 80), (20, 64)),
        [("spill", 10), ("demote", 20)],
        "the host's cache first; the device's then past the host"
    );
    assert_eq!(
        asked((50, 80), (20, 64)),
        [("demote", 60)],
        "a host with room takes the device's"
    );
    assert_eq!(
        asked((8, 80), (0, 64)),
        [],
        "nothing goes where neither tier has room"
    );
}

/// IT round 9: weights held resident and the activation left one chunk of a
/// shared pool for kv, and a session's prompt needed two. The list shape's
/// capacity counted the weights' chunks as kv, so the ask parked, found no
/// one to move (the pool was empty) and failed as a starved pool with "no
/// evictable victim". It fails up front, saying how the pool divides.
#[test]
fn an_ask_past_what_weights_and_activation_leave_fails_saying_so() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    const MIB: u64 = 1 << 20;
    // 67 chunks of 18 MiB: 40 of pinned weights, 25 of activation, the
    // null chunk; one chunk (144 pages of 16 tokens) left.
    let room = crate::store::shared::PoolRoom {
        chunk_bytes: 18 * MIB,
        total_chunks: 67,
        reserved_chunks: 1,
        act_chunks: 25,
        weight_chunks: 40,
        state_chunks: 1,
        list_slots: 144,
        list_tokens: 16,
    };
    // The list shape's capacity: every chunk but the null one and the
    // activation, the weights' included.
    let pool = Arc::new(KeptPool::new(144, 41 * 144 - 144, usize::MAX).dividing(room));
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let head = ProcessId::new_v4();
    planner.register(head);
    planner.note_admitted(head);
    let demand = Demand {
        kv_pages: 188,
        rs: RsUnits::default(),
    };
    let acquired = planner
        .acquire(head, head, demand)
        .now_or_never()
        .expect("refused at once, not parked");
    let Err(error) = acquired else {
        panic!("an ask past the pool is refused");
    };
    assert_eq!(error, PlannerError::PastPool { need: 188, room });
    let said = error.to_string();
    for part in [
        "188 kv pages (3008 tokens)",
        "weights pinned on it take 720 MiB",
        "activation 450 MiB",
        "leaves 18 MiB (144 kv pages, 2304 tokens)",
        "raise device_kv_budget by at least 18 MiB",
    ] {
        assert!(said.contains(part), "{part:?} missing from {said:?}");
    }
    assert_eq!(planner.diagnostics().queue.len(), 0, "nothing parked");
    assert_eq!(planner.diagnostics().starvations_total, 0);

    // What fits the room still waits for it as before.
    let fits = Demand {
        kv_pages: 100,
        rs: RsUnits::default(),
    };
    let acquired = planner.acquire(head, head, fits).now_or_never();
    assert!(
        matches!(acquired, Some(Ok(Acquired::Granted(_)))),
        "served from the room"
    );
}
