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

// ---- the governor's view ----

// ---- the load ----

// ---- adopting the load's own seating ----
