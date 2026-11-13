//! The lane arms an engine's rungs one composition at a time: idle, and
//! between fires for a raise waiting on a rung not armed, each step lent
//! free slots no sequence holds and given them back.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ::engine::live::{ArmingLease, ArmingRoom, Live};

use super::OnPlanner;
use crate::planner::{ChunkStats, Demand, PoolPort, ResidencyPlanner};
use crate::scheduler::lane::{Lane, LaunchReplyRx};
use crate::scheduler::stats::SchedulerStats;
use crate::store::kv::page_table::PhysicalKvPageId;
use crate::store::rs::{RsGrant, RsSlotId, RsUnits};

const RUNGS: [u32; 4] = [256, 512, 1024, 2048];
const COMPOSITIONS: u32 = 3;
const ROOM: ArmingRoom = ArmingRoom {
    pages: 4,
    records: 1,
};

/// Free kv pages and record slots, as a shared pool's lists hand them out.
struct Slots {
    pages: Mutex<Vec<PhysicalKvPageId>>,
    records: Mutex<Vec<RsSlotId>>,
}

impl Slots {
    fn new(pages: u32, records: u32) -> Arc<Self> {
        Arc::new(Self {
            pages: Mutex::new((0..pages).map(PhysicalKvPageId).collect()),
            records: Mutex::new((0..records).map(RsSlotId).collect()),
        })
    }

    fn free(&self) -> (usize, usize) {
        (
            self.pages.lock().unwrap().len(),
            self.records.lock().unwrap().len(),
        )
    }
}

struct Port(Arc<Slots>);

impl PoolPort for Port {
    fn chunk_stats(&self) -> ChunkStats {
        let (pages, records) = self.0.free();
        ChunkStats {
            pages: (pages as u32, 16),
            rs: (RsUnits::of(records as u32, 0), RsUnits::of(4, 0)),
        }
    }
    fn host_stats(&self) -> (u32, u32) {
        (0, 0)
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
    fn suspend_capable(&self) -> bool {
        false
    }
    fn locus(&self) -> (usize, usize) {
        (0, 0)
    }
    fn reserve_device(&self, count: u32) -> Option<Vec<PhysicalKvPageId>> {
        let mut free = self.0.pages.lock().unwrap();
        let at = free.len().checked_sub(count as usize)?;
        Some(free.split_off(at))
    }
    fn reserve_device_up_to(&self, count: u32) -> Vec<PhysicalKvPageId> {
        let mut free = self.0.pages.lock().unwrap();
        let at = free.len().saturating_sub(count as usize);
        free.split_off(at)
    }
    fn release_device(&self, pages: Vec<PhysicalKvPageId>) {
        self.0.pages.lock().unwrap().extend(pages);
    }
    fn reserve_rs(&self, need: RsUnits) -> Option<RsGrant> {
        let mut free = self.0.records.lock().unwrap();
        let at = free.len().checked_sub(need.slots as usize)?;
        Some(RsGrant {
            slots: free.split_off(at),
            ..RsGrant::default()
        })
    }
    fn release_rs(&self, grant: RsGrant) {
        self.0.records.lock().unwrap().extend(grant.slots);
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

/// What the fake engine saw.
#[derive(Default)]
struct Seen {
    steps: AtomicUsize,
    /// Calls after `arm_step` answered `false`.
    after_over: AtomicUsize,
    over: AtomicBool,
    /// Fires wider than the widest rung armed when they were submitted.
    unarmed_fires: AtomicUsize,
    /// Leases that named a slot a sequence holds, or not the whole room.
    bad_leases: AtomicUsize,
    /// Free slots seen during a step, its lease out.
    free_in_step: Mutex<Vec<(usize, usize)>>,
}

/// Arms `RUNGS` past the first, `COMPOSITIONS` steps a rung.
struct Arms {
    live: Live,
    next: usize,
    done: u32,
    seen: Arc<Seen>,
    slots: Arc<Slots>,
    /// The pages and records sequences hold, never to be lent.
    held: Arc<Mutex<(HashSet<u32>, HashSet<u32>)>>,
    fire_sleep: Duration,
    step_sleep: Duration,
    fail: bool,
}

impl Arms {
    fn new(seen: &Arc<Seen>, slots: &Arc<Slots>) -> Self {
        Self {
            live: Live::arming(RUNGS[0]),
            next: 1,
            done: 0,
            seen: seen.clone(),
            slots: slots.clone(),
            held: Arc::default(),
            fire_sleep: Duration::ZERO,
            step_sleep: Duration::ZERO,
            fail: false,
        }
    }
}

impl ::engine::Engine for Arms {
    fn kind(&self) -> &'static str {
        "arms"
    }

    fn live(&self) -> Live {
        self.live.clone()
    }

    fn arming_room(&self) -> ArmingRoom {
        ROOM
    }

    fn arm_step(&mut self, lent: &ArmingLease) -> ::engine::Result<bool> {
        if self.seen.over.load(Ordering::SeqCst) {
            self.seen.after_over.fetch_add(1, Ordering::SeqCst);
        }
        if self.fail {
            // Any later call counts as one after the last.
            self.seen.over.store(true, Ordering::SeqCst);
            return Err(::engine::Error::Load("capture refused".into()));
        }
        self.seen.steps.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(self.step_sleep);
        {
            let held = self.held.lock().unwrap();
            let whole = lent.pages.len() == ROOM.pages as usize
                && lent.records.len() == ROOM.records as usize;
            let foreign = lent.pages.iter().any(|page| held.0.contains(page))
                || lent.records.iter().any(|record| held.1.contains(record));
            if !whole || foreign {
                self.seen.bad_leases.fetch_add(1, Ordering::SeqCst);
            }
        }
        self.seen
            .free_in_step
            .lock()
            .unwrap()
            .push(self.slots.free());
        if self.next >= RUNGS.len() {
            self.seen.over.store(true, Ordering::SeqCst);
            return Ok(false);
        }
        self.done += 1;
        if self.done == COMPOSITIONS {
            self.done = 0;
            let rung = RUNGS[self.next];
            self.next += 1;
            self.live.arm(if self.next == RUNGS.len() {
                u32::MAX
            } else {
                rung
            });
        }
        Ok(true)
    }

    fn load(&mut self, _request: ::engine::LoadRequest) -> ::engine::Result<::engine::Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        frame: &::engine::FrameSubmission,
    ) -> ::engine::Result<::engine::FrameTicket> {
        let armed = self.live.armed_through();
        for step in &frame.steps {
            let tokens: usize = step.lanes.iter().map(|lane| lane.tokens.len()).sum();
            if tokens > armed as usize {
                self.seen.unarmed_fires.fetch_add(1, Ordering::SeqCst);
            }
        }
        if !self.fire_sleep.is_zero() {
            std::thread::sleep(self.fire_sleep);
        }
        Ok(::engine::FrameTicket {
            id: ::engine::FrameId(0),
            steps: Vec::new(),
        })
    }
}

fn planner(slots: &Arc<Slots>) -> Arc<ResidencyPlanner> {
    Arc::new(ResidencyPlanner::new(
        Arc::new(Port(slots.clone())) as Arc<dyn PoolPort>
    ))
}

fn spawn(engine_idx: usize, engine: Arms, planner: &Arc<ResidencyPlanner>) -> Lane {
    Lane::spawn_lending(
        engine_idx,
        Some(Box::new(engine)),
        Arc::new(SchedulerStats::default()),
        Box::new(OnPlanner(planner.clone())),
    )
}

fn fire(lane: &Lane, tokens: u32) -> LaunchReplyRx {
    lane.fire(
        crate::engine::FrameFire {
            steps: vec![crate::engine::StepFire {
                submission: ::engine::Step {
                    lanes: vec![::engine::Lane {
                        tokens: vec![0; tokens as usize],
                        ..::engine::Lane::default()
                    }],
                    attachments: Vec::new(),
                    media: Vec::new(),
                    voxels: Vec::new(),
                },
                terminal_cells: Vec::new(),
                completions: Vec::new(),
                instances: Vec::new(),
            }],
        },
        false,
    )
}

fn answer(reply: LaunchReplyRx) -> Result<(), String> {
    crate::rt::block_on(async { crate::rt::time::timeout(Duration::from_secs(10), reply).await })
        .expect("a fire is answered")
        .expect("a fire is answered")
        .map(|_| ())
}

fn wait_until(what: &str, done: impl Fn() -> bool) {
    let until = std::time::Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(std::time::Instant::now() < until, "timed out: {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The widest rung armed that holds no more than `rows`: the rung a pass
/// fires at ([`crate::store::shared::SharedPool::armed_rung_for`]).
fn armed_rung(live: &Live, rows: u32) -> u32 {
    let armed = live.armed_through();
    RUNGS
        .iter()
        .copied()
        .filter(|&rung| rung <= armed && rung <= rows)
        .max()
        .unwrap_or(RUNGS[0])
}

#[test]
fn an_idle_lane_arms_every_rung_then_never_asks_again() {
    let slots = Slots::new(16, 4);
    let planner = planner(&slots);
    let seen = Arc::new(Seen::default());
    let engine = Arms::new(&seen, &slots);
    let live = engine.live.clone();
    let mut lane = spawn(9101, engine, &planner);
    wait_until("every rung armed", || seen.over.load(Ordering::SeqCst));
    assert_eq!(live.armed_through(), u32::MAX);
    assert_eq!(
        seen.steps.load(Ordering::SeqCst),
        (RUNGS.len() - 1) * COMPOSITIONS as usize + 1,
        "one composition a call, and the call that answers false"
    );
    std::thread::sleep(Duration::from_millis(50));
    answer(fire(&lane, 2048)).expect("the lane serves once armed");
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(
        seen.after_over.load(Ordering::SeqCst),
        0,
        "arm_step is never called once it answered false"
    );
    assert_eq!(seen.bad_leases.load(Ordering::SeqCst), 0);
    assert_eq!(slots.free(), (16, 4), "every lease is given back");
    lane.shutdown();
}

#[test]
fn a_lease_is_free_slots_no_sequence_holds_and_goes_back_after_the_step() {
    let slots = Slots::new(16, 4);
    let planner = planner(&slots);
    // A sequence holds most of the pool: what is left cannot lend a step.
    let mut sequence = planner
        .try_free(Demand {
            kv_pages: 14,
            rs: RsUnits::of(2, 0),
        })
        .expect("the sequence's slots");
    let seen = Arc::new(Seen::default());
    let engine = Arms::new(&seen, &slots);
    {
        let mut held = engine.held.lock().unwrap();
        held.0 = sequence.kv_pages().iter().map(|page| page.0).collect();
        held.1 = sequence.rs_slots().iter().map(|slot| slot.0).collect();
    }
    let held = engine.held.clone();
    let live = engine.live.clone();
    let mut lane = spawn(9102, engine, &planner);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        seen.steps.load(Ordering::SeqCst),
        0,
        "no room free, no step"
    );
    assert_eq!(live.armed_through(), RUNGS[0]);

    // Part of it goes: a step's room stands free beside what still holds.
    let gone: Vec<_> = sequence.lend_kv().drain(..6).collect();
    {
        let mut held = held.lock().unwrap();
        for page in &gone {
            held.0.remove(&page.0);
        }
    }
    slots.pages.lock().unwrap().extend(gone);
    wait_until("every rung armed", || seen.over.load(Ordering::SeqCst));
    assert_eq!(
        seen.bad_leases.load(Ordering::SeqCst),
        0,
        "no lease names a sequence's slot"
    );
    for &free in seen.free_in_step.lock().unwrap().iter() {
        assert_eq!(
            free,
            (8 - ROOM.pages as usize, 2 - ROOM.records as usize),
            "a step holds its room and nothing else"
        );
    }
    std::thread::sleep(Duration::from_millis(10));
    assert_eq!(slots.free(), (8, 2), "every lease is back to free");
    lane.shutdown();
    drop(sequence);
}

#[test]
fn a_raise_under_continuous_fires_is_armed_between_them() {
    const ENGINE: usize = 9103;
    let slots = Slots::new(16, 4);
    let planner = planner(&slots);
    // No room until fires are queued, so the lane is never idle with room.
    let blocker = planner
        .try_free(Demand {
            kv_pages: 16,
            rs: RsUnits::of(4, 0),
        })
        .expect("all of it");
    let seen = Arc::new(Seen::default());
    let mut engine = Arms::new(&seen, &slots);
    engine.fire_sleep = Duration::from_millis(2);
    let live = engine.live.clone();
    let mut lane = spawn(ENGINE, engine, &planner);
    let mut in_flight = std::collections::VecDeque::new();
    for _ in 0..16 {
        in_flight.push_back(fire(&lane, 256));
    }
    drop(blocker);
    // Continuous fires, no raise: the lane is never idle, so nothing arms.
    for _ in 0..40 {
        answer(in_flight.pop_front().unwrap()).expect("served");
        in_flight.push_back(fire(&lane, armed_rung(&live, 2048)));
    }
    assert_eq!(
        live.armed_through(),
        RUNGS[0],
        "the queue never ran dry, and no raise waited"
    );
    assert_eq!(seen.steps.load(Ordering::SeqCst), 0);

    // A pass asks 2048 rows: steps run between fires, as many as a gap's
    // budget holds (all of them, steps here costing nothing), and the pass
    // then fires at its rung.
    let want = crate::store::shared::want_rows(ENGINE, 2048);
    let mut fires = 0;
    let mut widths = Vec::new();
    while armed_rung(&live, 2048) < 2048 {
        answer(in_flight.pop_front().unwrap()).expect("served");
        let width = armed_rung(&live, 2048);
        widths.push(width);
        in_flight.push_back(fire(&lane, width));
        fires += 1;
        assert!(fires < 1000, "the raise starved under load");
    }
    in_flight.push_back(fire(&lane, 2048));
    for reply in in_flight {
        answer(reply).expect("served");
    }
    drop(want);
    assert!(
        fires <= 2,
        "steps this cheap arm the raise's rung within one gap: {widths:?}"
    );
    let steps = seen.steps.load(Ordering::SeqCst);
    let rungs = (RUNGS.len() - 1) * COMPOSITIONS as usize;
    assert!(
        (rungs..=rungs + 1).contains(&steps),
        "every step of the raise's rungs (and the last, answering false, once \
         the lane idles): {steps}"
    );
    assert_eq!(
        seen.unarmed_fires.load(Ordering::SeqCst),
        0,
        "no fire is wider than the widest rung armed"
    );
    assert_eq!(seen.bad_leases.load(Ordering::SeqCst), 0);
    lane.shutdown();
}

#[test]
fn a_failed_step_faults_every_fire_after_it() {
    let slots = Slots::new(16, 4);
    let planner = planner(&slots);
    let seen = Arc::new(Seen::default());
    let mut engine = Arms::new(&seen, &slots);
    engine.fail = true;
    let mut lane = spawn(9104, engine, &planner);
    std::thread::sleep(Duration::from_millis(30));
    let error = answer(fire(&lane, 256)).expect_err("a faulted load serves nothing");
    assert!(error.contains("capture refused"), "{error}");
    let error = answer(fire(&lane, 256)).expect_err("nor after");
    assert!(error.contains("arming"), "{error}");
    assert_eq!(seen.after_over.load(Ordering::SeqCst), 0);
    assert_eq!(slots.free(), (16, 4), "the failed step's lease is back");
    lane.shutdown();
}

/// Steps longer than a gap's budget: one a gap, the raise's pass firing at
/// each rung as it arms, so no request waits on the whole of its arming.
#[test]
fn a_raise_whose_steps_outlast_a_gap_fires_at_each_rung_as_it_arms() {
    const ENGINE: usize = 9105;
    let slots = Slots::new(16, 4);
    let planner = planner(&slots);
    let blocker = planner
        .try_free(Demand {
            kv_pages: 16,
            rs: RsUnits::of(4, 0),
        })
        .expect("all of it");
    let seen = Arc::new(Seen::default());
    let mut engine = Arms::new(&seen, &slots);
    engine.fire_sleep = Duration::from_millis(1);
    engine.step_sleep = super::GAP_BUDGET + Duration::from_millis(5);
    let live = engine.live.clone();
    let mut lane = spawn(ENGINE, engine, &planner);
    let mut in_flight = std::collections::VecDeque::new();
    for _ in 0..8 {
        in_flight.push_back(fire(&lane, 256));
    }
    let want = crate::store::shared::want_rows(ENGINE, 2048);
    drop(blocker);
    let mut widths = Vec::new();
    while armed_rung(&live, 2048) < 2048 {
        answer(in_flight.pop_front().unwrap()).expect("served");
        let width = armed_rung(&live, 2048);
        widths.push(width);
        in_flight.push_back(fire(&lane, width));
        assert!(widths.len() < 1000, "the raise starved under load");
    }
    for reply in in_flight {
        answer(reply).expect("served");
    }
    drop(want);
    assert!(
        widths.contains(&512) && widths.contains(&1024),
        "the pass fires at each rung as it arms: {widths:?}"
    );
    assert!(
        widths.windows(2).all(|w| w[0] <= w[1]),
        "and only ever wider: {widths:?}"
    );
    assert_eq!(seen.unarmed_fires.load(Ordering::SeqCst), 0);
    lane.shutdown();
}
