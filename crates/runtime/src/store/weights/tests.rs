use ::engine::chunk::{ShapeId as EngineShape, WeightKind};

use super::*;
use crate::store::slots::SlotPoolFacts;

const MIB: u64 = 1 << 20;
/// Chunks a segment.
const SEGMENT: u32 = 4;
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
        span_on_demand: false,
        act_span: None,
    })
}

fn atom(atom: u32) -> WeightAtom {
    WeightAtom {
        shape: EXPERTS,
        atom,
    }
}

fn resident(cache: &WeightCache, of: u32) -> Vec<u32> {
    (0..of)
        .filter(|&a| cache.resident(atom(a)).is_some())
        .collect()
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
