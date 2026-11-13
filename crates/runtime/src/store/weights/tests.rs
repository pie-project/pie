use ::engine::chunk::{ShapeId as EngineShape, WeightKind};

use super::*;
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
        seated: None,
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
        held: Vec::new(),
    }
}

/// One governor tick.
fn tick(cache: &mut WeightCache, pool: &mut SlotPool, refill: bool) {
    cache.tick(pool, 0, refill);
}

fn resident(cache: &WeightCache, of: u32) -> Vec<u32> {
    (0..of)
        .filter(|&a| cache.resident(atom(a)).is_some())
        .collect()
}

#[test]
fn a_missed_atom_comes_back_at_break_even_in_place_of_one_no_fire_read() {
    // Room for the floor and three atoms.
    let shape = experts(16, 2);
    let mut slots = pool(8, &shape);
    let kv = slots.acquire(Holder::WorkingSet(1), KV, 2).unwrap();
    let mut cache = WeightCache::new(&[shape], WEIGHTS, true, &mut slots).unwrap();
    assert_eq!(cache.unmet_bytes(), 0, "nothing missed yet");
    cache.record(&uses(&[(1, 5), (2, 5), (3, 1)]));
    assert_eq!(
        cache.unmet_bytes(),
        3 * MIB,
        "a miss streams the whole atom: each is past break-even"
    );
    tick(&mut cache, &mut slots, true);
    assert_eq!(resident(&cache, 16), [1, 2, 3]);
    assert_eq!(slots.available(WEIGHTS), 0);
    assert_eq!(cache.unmet_bytes(), 0, "what came back owes nothing");

    // With no room, a missed atom takes the place of one no fire read in
    // the tick: 2 and 3 are read again, 1 is not.
    let slot_of_1 = cache.resident(atom(1)).unwrap();
    cache.record(&uses(&[(2, 1), (3, 1), (4, 9)]));
    tick(&mut cache, &mut slots, true);
    assert_eq!(resident(&cache, 16), [2, 3, 4]);
    assert_eq!(
        cache.resident(atom(4)),
        Some(slot_of_1),
        "the displaced atom's slot goes to the one displacing it"
    );

    // Every resident atom read in the tick: a miss displaces none.
    cache.record(&uses(&[(2, 1), (3, 1), (4, 1), (5, 1)]));
    tick(&mut cache, &mut slots, true);
    assert_eq!(resident(&cache, 16), [2, 3, 4]);
    assert_eq!(cache.unmet_bytes(), MIB, "5 waits, wanted");

    // A tick in which only 5 and 6 are read, 6 the most: what has cost
    // most comes back first, each in place of the longest unread.
    cache.record(&uses(&[(5, 1), (6, 3)]));
    tick(&mut cache, &mut slots, true);
    assert_eq!(resident(&cache, 16), [4, 5, 6]);
    slots.release(Holder::WorkingSet(1), &kv, 0);
}

#[test]
fn a_held_set_leaves_for_the_floor_only_once_the_engine_took_it() {
    // Room for every atom.
    let shape = experts(4, 2);
    let mut slots = pool(12, &shape);
    let mut cache = WeightCache::new(&[shape], WEIGHTS, false, &mut slots).unwrap();
    assert!(!cache.streaming());
    assert_eq!(resident(&cache, 4), [0, 1, 2, 3]);
    let (staging, residency) = cache.leaving_set(&slots).expect("held, it may leave");
    assert!(!residency.held);
    assert_eq!(residency.staging[0].1.len(), 2, "the floor stages");
    assert_eq!(residency.resident.len(), 2, "the rest stays resident");
    assert_eq!(resident(&cache, 4), [0, 1, 2, 3], "nothing moved yet");

    cache.left_held(&staging, &mut slots);
    assert!(cache.streaming());
    assert_eq!(
        resident(&cache, 4),
        [0, 1],
        "the floor took the highest atoms"
    );
    assert_eq!(cache.cache_bytes(), 2 * MIB);
    assert_eq!(
        cache.shed(&mut slots, 0),
        2,
        "the rest is cache above the floor"
    );
    assert_eq!(cache.cache_bytes(), 0);
    assert!(
        cache.leaving_set(&slots).is_none(),
        "streaming, it has left"
    );
}

// ---- the governor's view ----

// ---- the load ----

// ---- adopting the load's own seating ----
