use std::sync::{Arc, Mutex};

use ::engine::chunk::{ShapeId as EngineShape, WeightKind};

use super::*;
use crate::governor::run::Steer;
use crate::governor::{Pressure, Signal, Tolerance};
use crate::store::slots::SlotPoolFacts;

const MIB: u64 = 1 << 20;
/// Chunks a segment.
const SEGMENT: u32 = 4;
const KV: ShapeId = 0;
const WEIGHTS: ShapeId = 1;
const EXPERTS: EngineShape = EngineShape(7);

fn experts(count: u32, floor: u32) -> WeightShape {
    WeightShape {
        id: EXPERTS,
        kind: WeightKind::Expert,
        slot_bytes: MIB,
        slots_per_chunk: 1,
        count,
        floor_slots: floor,
        name: "layers.moe.experts".into(),
    }
}

/// A pool of `chunks` 1 MiB chunks in segments of four, all committed, chunk
/// 0 reserved: a kv shape, then the weight shape.
fn pool(chunks: u32, weights: &WeightShape) -> SlotPool {
    let mut shapes = vec![SlotShape {
        kind: ShapeKind::List,
        slot_bytes: MIB,
        slots_per_chunk: 1,
    }];
    shapes.extend(pool_shapes(std::slice::from_ref(weights)));
    SlotPool::new(SlotPoolFacts {
        chunk_bytes: MIB,
        total_chunks: chunks,
        segment_chunks: SEGMENT,
        committed_segments: chunks / SEGMENT,
        reserved_chunks: 1,
        shapes,
        act_chunks: Vec::new(),
    })
}

fn atom(atom: u32) -> WeightAtom {
    WeightAtom {
        shape: EXPERTS,
        atom,
    }
}

fn uses(reads: &[(u32, u32)]) -> WeightUses {
    WeightUses {
        uses: reads.iter().map(|&(a, n)| (atom(a), n)).collect(),
    }
}

/// Ticks to the end of the window being recorded.
fn close_window(cache: &mut WeightCache, pool: &mut SlotPool, refill: bool) {
    for _ in 0..WINDOW_TICKS {
        cache.tick(pool, 0, refill);
    }
}

fn resident(cache: &WeightCache, of: u32) -> Vec<u32> {
    (0..of)
        .filter(|&a| cache.resident(atom(a)).is_some())
        .collect()
}

#[test]
fn auto_streams_only_weights_that_do_not_fit() {
    assert!(!WeightStreaming::Auto.streams(true));
    assert!(WeightStreaming::Auto.streams(false));
    assert!(WeightStreaming::On.streams(true));
    assert!(!WeightStreaming::Off.streams(false));
    assert_eq!(WeightStreaming::default(), WeightStreaming::Auto);
    #[derive(Deserialize)]
    struct Model {
        weight_streaming: WeightStreaming,
    }
    for (word, mode) in [
        ("auto", WeightStreaming::Auto),
        ("on", WeightStreaming::On),
        ("off", WeightStreaming::Off),
    ] {
        let model: Model = toml::from_str(&format!("weight_streaming = {word:?}")).unwrap();
        assert_eq!(model.weight_streaming, mode);
    }
    assert!(toml::from_str::<Model>("weight_streaming = \"sometimes\"").is_err());
}

#[test]
fn the_mode_resolved_at_load_is_kept_per_engine() {
    set_streaming(9_001, true);
    set_streaming(9_002, false);
    assert!(streaming(9_001));
    assert!(!streaming(9_002));
    assert!(!streaming(9_003), "an engine never told does not stream");
}

#[test]
fn the_floor_is_live_and_the_cache_above_it_only_wanted() {
    let shape = experts(16, 2);
    let mut slots = pool(16, &shape);
    let mut cache = WeightCache::new(&[shape], WEIGHTS, true, &mut slots).unwrap();
    assert_eq!(slots.demand().live, 2 * MIB);
    cache.record(&uses(&[(3, 1), (4, 1), (5, 1)]));
    close_window(&mut cache, &mut slots, true);
    assert_eq!(resident(&cache, 16), [3, 4, 5]);
    let demand = slots.demand();
    assert_eq!(demand.live, 2 * MIB, "the cache is not live");
    assert_eq!(demand.wanted, 5 * MIB, "but it is wanted");
}

#[test]
fn a_miss_never_inserts_and_membership_changes_only_as_a_window_closes() {
    let shape = experts(16, 2);
    let mut slots = pool(16, &shape);
    let mut cache = WeightCache::new(&[shape], WEIGHTS, true, &mut slots).unwrap();
    let _ = cache.take_changed(&slots);
    for _ in 0..100 {
        cache.record(&uses(&[(1, 8), (2, 8)]));
    }
    assert!(resident(&cache, 16).is_empty(), "a miss streams through");
    assert!(cache.take_changed(&slots).is_none());
    assert_eq!(cache.unmet_bytes(), 2 * MIB, "it is wanted, not taken");
    for _ in 1..WINDOW_TICKS {
        cache.tick(&mut slots, 0, true);
        assert!(resident(&cache, 16).is_empty(), "no tick but the window's");
    }
    cache.tick(&mut slots, 0, true);
    assert_eq!(resident(&cache, 16), [1, 2]);
    let residency = cache.take_changed(&slots).unwrap();
    assert_eq!(residency.resident.len(), 2);
    assert_eq!(residency.staging, [(EXPERTS, vec![1, 2])]);
}

#[test]
fn the_least_recently_used_atom_makes_way_for_one_read_in_the_last_window() {
    // Room for the floor and three atoms.
    let shape = experts(16, 2);
    let mut slots = pool(8, &shape);
    let kv = slots.acquire(Holder::WorkingSet(1), KV, 2).unwrap();
    let mut cache = WeightCache::new(&[shape], WEIGHTS, true, &mut slots).unwrap();
    cache.record(&uses(&[(1, 5), (2, 5), (3, 5)]));
    close_window(&mut cache, &mut slots, true);
    assert_eq!(resident(&cache, 16), [1, 2, 3]);
    assert_eq!(slots.available(WEIGHTS), 0);

    // 2 and 3 are read again, 4 for the first time and the most; 1 not.
    let slot_of_1 = cache.resident(atom(1)).unwrap();
    cache.record(&uses(&[(2, 1), (3, 1), (4, 9)]));
    close_window(&mut cache, &mut slots, true);
    assert_eq!(resident(&cache, 16), [2, 3, 4]);
    assert_eq!(
        cache.resident(atom(4)),
        Some(slot_of_1),
        "the displaced atom's slot goes to the one displacing it"
    );

    // A window in which only 5 is read: of the rest, the most read in the
    // window they were last read in stay, the lower atom on a tie.
    cache.record(&uses(&[(5, 1)]));
    close_window(&mut cache, &mut slots, true);
    assert_eq!(resident(&cache, 16), [2, 4, 5]);
    slots.release(Holder::WorkingSet(1), &kv, 0);
}

#[test]
fn kv_and_state_waiters_come_first_and_the_cache_refills_after() {
    let shape = experts(16, 2);
    let mut slots = pool(8, &shape);
    let mut cache = WeightCache::new(&[shape], WEIGHTS, true, &mut slots).unwrap();
    slots.wait(1, KV, 4);
    assert!(kv_short(&slots, &cache));
    cache.record(&uses(&[(1, 1), (2, 1)]));
    close_window(&mut cache, &mut slots, false);
    assert!(resident(&cache, 16).is_empty(), "no refill while kv waits");

    let kv = slots.acquire(Holder::WorkingSet(1), KV, 4).unwrap();
    slots.unwait(1);
    assert!(!kv_short(&slots, &cache));
    cache.record(&uses(&[(1, 1), (2, 1)]));
    close_window(&mut cache, &mut slots, true);
    assert_eq!(resident(&cache, 16), [1], "the cache takes the room left");
    slots.release(Holder::WorkingSet(1), &kv, 0);
}

#[test]
fn not_streaming_every_atom_is_pinned_and_none_is_ever_shed() {
    let shape = experts(8, 2);
    let mut slots = pool(16, &shape);
    let mut cache = WeightCache::new(&[shape], WEIGHTS, false, &mut slots).unwrap();
    assert_eq!(resident(&cache, 8), (0..8).collect::<Vec<_>>());
    assert_eq!(cache.pinned_bytes(), 10 * MIB);
    assert_eq!(slots.demand().live, 10 * MIB);
    assert_eq!(cache.shed(&mut slots, 0), 0);
    cache.record(&uses(&[(1, 1)]));
    close_window(&mut cache, &mut slots, true);
    assert_eq!(resident(&cache, 8).len(), 8);
}

// ---- the governor's view ----

/// An engine that streams: it reports what it was told to and what fires
/// read.
#[derive(Default)]
struct Seen {
    reads: Vec<WeightUses>,
    told: Vec<WeightResidency>,
}

#[derive(Clone, Default)]
struct FakeEngine(Arc<Mutex<Seen>>);

impl ::engine::Engine for FakeEngine {
    fn kind(&self) -> &'static str {
        "fake"
    }

    fn load(&mut self, _request: ::engine::LoadRequest) -> ::engine::Result<::engine::Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        _frame: &::engine::FrameSubmission,
    ) -> ::engine::Result<::engine::FrameTicket> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn weight_residency(&mut self, residency: &WeightResidency) -> ::engine::Result<()> {
        self.0.lock().unwrap().told.push(residency.clone());
        Ok(())
    }

    fn weight_uses(&mut self) -> ::engine::Result<WeightUses> {
        let mut seen = self.0.lock().unwrap();
        Ok(WeightUses {
            uses: std::mem::take(&mut seen.reads)
                .into_iter()
                .flat_map(|uses| uses.uses)
                .collect(),
        })
    }
}

/// An engine that does not stream.
struct Plain;

impl ::engine::Engine for Plain {
    fn kind(&self) -> &'static str {
        "plain"
    }

    fn load(&mut self, _request: ::engine::LoadRequest) -> ::engine::Result<::engine::Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        _frame: &::engine::FrameSubmission,
    ) -> ::engine::Result<::engine::FrameTicket> {
        Err(::engine::Error::Load("no model".into()))
    }
}

/// The inner pool, as the stores would steer it: a shrink empties the top
/// segment by evacuation, and stops where there is no room. It never drops
/// a slot.
struct Inner {
    slots: SlotPool,
    floor: u64,
}

impl Inner {
    fn committed(&self) -> u64 {
        u64::from(self.slots.committed_chunks() / SEGMENT)
    }
}

impl SlotAccess for Inner {
    fn with_slots(&mut self, f: &mut dyn FnMut(&mut SlotPool, u64)) {
        f(&mut self.slots, 0);
    }
}

impl PoolTarget for Inner {
    fn demand(&mut self) -> Demand {
        let demand = self.slots.demand();
        Demand {
            wanted: demand.wanted,
            live: demand.live,
        }
    }

    fn segment_bytes(&self) -> u64 {
        u64::from(SEGMENT) * MIB
    }

    fn floor(&self) -> u64 {
        self.floor
    }

    fn hard_limit(&self) -> u64 {
        u64::from(self.slots.total_chunks()) * MIB
    }

    fn held(&self) -> u64 {
        self.committed() * self.segment_bytes()
    }

    fn growth(&self) -> Growth {
        Growth::Elastic { least_cut: 0 }
    }

    fn retarget(&mut self, segments: u64) -> Retarget {
        while self.committed() > segments {
            let top = u32::try_from(self.committed() - 1).unwrap();
            if self.slots.segments().used(top) > 0 {
                let Some(evacuation) = self.slots.plan_evacuation(top) else {
                    return Retarget::NoRoom {
                        held: self.committed(),
                    };
                };
                self.slots.apply_evacuation(&evacuation, 0).unwrap();
            }
            self.slots.release_segment(top).unwrap();
        }
        while self.committed() < segments {
            let next = u32::try_from(self.committed()).unwrap();
            self.slots.commit_segment(next).unwrap();
        }
        Retarget::Reached
    }

    fn shrink_rung(&mut self) {}

    fn restore_rung(&mut self) {}
}

/// 16 chunks: chunk 0 reserved, the floor's two, `kv` chunks a working set
/// holds, and the cache filled with every other chunk (streaming) or
/// `atoms` pinned (not).
fn steered(streaming: bool, atoms: u32, kv: u64) -> (WeightFirst<Inner, FakeEngine>, FakeEngine) {
    let shape = experts(atoms, 2);
    let mut slots = pool(16, &shape);
    let mut cache = WeightCache::new(&[shape], WEIGHTS, streaming, &mut slots).unwrap();
    slots.acquire(Holder::WorkingSet(1), KV, kv).unwrap();
    let engine = FakeEngine::default();
    if streaming {
        let room = u32::try_from(slots.available(WEIGHTS)).unwrap();
        cache.record(&uses(&(0..room).map(|a| (a, 1)).collect::<Vec<_>>()));
        close_window(&mut cache, &mut slots, true);
        assert_eq!(slots.available(WEIGHTS), 0, "the pool is full");
    }
    let inner = Inner {
        slots,
        floor: 4 * MIB,
    };
    (WeightFirst::new(inner, cache, engine.clone()), engine)
}

fn critical(steer: &mut Steer, pool: &mut WeightFirst<Inner, FakeEngine>) -> Vec<Signal> {
    let mut signals = Vec::new();
    steer.turn(pool, Some(Pressure::Critical), &mut |signal| {
        signals.push(signal);
    });
    signals
}

#[test]
fn streaming_sheds_the_cache_to_its_floor_before_any_tier_pressure() {
    // 1 reserved + 2 floor + 10 kv = 13 chunks: past three segments even
    // with the cache gone.
    let (mut pool, engine) = steered(true, 32, 10);
    assert_eq!(pool.cache().cached(), 3);
    let mut steer = Steer::new(&pool, Tolerance::Balanced, 64 * MIB);
    let floor_segments = steer.governor().limits().segments(0);
    assert_eq!(floor_segments, 2);

    let first = critical(&mut steer, &mut pool);
    assert!(first.is_empty(), "the cache went first: {first:?}");
    assert_eq!(pool.cache().cached(), 0, "shed to the floor");
    assert_eq!(
        pool.inner().slots.held_by(&Holder::Weights {
            shape: WEIGHTS,
            pinned: true
        }),
        2,
        "the floor stays"
    );
    assert_eq!(pool.inner().slots.held_by(&Holder::WorkingSet(1)), 10);
    let told = engine.0.lock().unwrap().told.clone();
    assert!(told.last().unwrap().resident.is_empty(), "the engine heard");

    let second = critical(&mut steer, &mut pool);
    assert!(
        matches!(second[..], [Signal::TierPressure { .. }]),
        "with the cache at its floor, kv and state go through PLEX: {second:?}"
    );
    assert_eq!(pool.inner().slots.held_by(&Holder::WorkingSet(1)), 10);
}

#[test]
fn a_shed_that_makes_room_raises_no_pressure_at_all() {
    // 1 + 2 + 5 = 8 chunks: two segments once the cache is gone.
    let (mut pool, _) = steered(true, 32, 5);
    let mut steer = Steer::new(&pool, Tolerance::Balanced, 64 * MIB);
    for _ in 0..4 {
        assert!(critical(&mut steer, &mut pool).is_empty());
    }
    assert_eq!(pool.inner().committed(), 2, "the floor: 4 MiB and 2 pinned");
    assert_eq!(pool.cache().cached(), 0);
}

#[test]
fn not_streaming_the_weights_stay_and_pressure_goes_to_kv_and_state() {
    // 1 + 2 floor + 8 pinned + 5 kv = 16 chunks.
    let (mut pool, _) = steered(false, 8, 5);
    pool.inner_mut().floor = MIB;
    let mut steer = Steer::new(&pool, Tolerance::Balanced, 64 * MIB);
    let signals = critical(&mut steer, &mut pool);
    assert!(
        matches!(signals[..], [Signal::TierPressure { .. }]),
        "{signals:?}"
    );
    for _ in 0..8 {
        critical(&mut steer, &mut pool);
    }
    assert_eq!(
        pool.inner().slots.held_by(&Holder::Weights {
            shape: WEIGHTS,
            pinned: true
        }),
        10,
        "no weight was evicted"
    );
}

#[test]
fn a_waiter_turn_never_changes_membership() {
    let (mut pool, engine) = steered(true, 32, 2);
    let before = (0..32)
        .filter(|&a| pool.cache().resident(atom(a)).is_some())
        .count();
    let told = engine.0.lock().unwrap().told.len();
    let mut steer = Steer::new(&pool, Tolerance::Balanced, 64 * MIB);
    engine
        .0
        .lock()
        .unwrap()
        .reads
        .push(uses(&[(30, 50), (31, 50)]));
    for _ in 0..(2 * WINDOW_TICKS) {
        steer.turn(&mut pool, None, &mut |_| {});
    }
    let after = (0..32)
        .filter(|&a| pool.cache().resident(atom(a)).is_some())
        .count();
    assert_eq!(before, after);
    assert!(pool.cache().resident(atom(30)).is_none());
    assert_eq!(engine.0.lock().unwrap().told.len(), told);

    // The ticks of one window take them in, displacing the least used.
    for _ in 0..WINDOW_TICKS {
        steer.turn(&mut pool, Some(Pressure::Calm), &mut |_| {});
    }
    assert!(pool.cache().resident(atom(30)).is_some());
    assert!(pool.cache().resident(atom(31)).is_some());
    assert!(engine.0.lock().unwrap().told.len() > told);
}

#[test]
fn a_kv_waiter_sheds_the_cache_at_the_next_tick_even_on_a_dedicated_pool() {
    let (mut pool, _) = steered(true, 32, 2);
    assert!(pool.cache().cached() > 0);
    let mut steer = Steer::new(&pool, Tolerance::Dedicated, 64 * MIB);
    pool.inner_mut().slots.wait(7, KV, 3);
    steer.turn(&mut pool, None, &mut |_| {});
    assert!(pool.cache().cached() > 0, "not between ticks");
    steer.turn(&mut pool, Some(Pressure::Calm), &mut |_| {});
    assert_eq!(pool.cache().cached(), 0);
    assert!(pool.inner().slots.available(KV) >= 3, "the waiter has room");
}

#[test]
fn an_engine_that_does_not_stream_reads_as_nothing_used() {
    let shape = experts(16, 2);
    let mut slots = pool(16, &shape);
    let cache = WeightCache::new(&[shape], WEIGHTS, true, &mut slots).unwrap();
    let inner = Inner {
        slots,
        floor: 4 * MIB,
    };
    let mut pool = WeightFirst::new(inner, cache, Plain);
    for _ in 0..(2 * WINDOW_TICKS) {
        pool.ticked();
    }
    assert_eq!(pool.cache().cached(), 0);
}

#[test]
fn unmet_reads_are_wanted_demand() {
    let (mut pool, engine) = steered(true, 32, 2);
    let wanted = pool.demand().wanted;
    engine.0.lock().unwrap().reads.push(uses(&[(30, 1)]));
    pool.ticked();
    assert_eq!(pool.demand().wanted, wanted + MIB);
}
