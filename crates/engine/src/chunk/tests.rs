use super::*;
use crate::fire::{Lane, Serves};

fn dense() -> PoolFacts {
    // Qwen3-8B on a 32 GB card: one kv pool, host and disk tiers.
    PoolFacts {
        kv_pages: 14_336,
        kv_page_size: 16,
        host_kv_pages: 65_536,
        disk_kv_pages: 262_144,
        disk_kv_persistent: true,
        ..PoolFacts::default()
    }
}

fn hybrid() -> PoolFacts {
    // A GDN hybrid: kv pages for its attention rows, state slots and rs
    // buffer pages beside them, each with a host tier.
    PoolFacts {
        kv_pages: 4_096,
        kv_page_size: 16,
        state_slots: 66,
        state_slot_bytes: 37 << 20,
        buffer_pages: 66,
        host_kv_pages: 8_192,
        host_state_slots: 128,
        host_buffer_pages: 128,
        elastic_state_slot_bytes: 38 << 20,
        ..PoolFacts::default()
    }
}

fn windowed() -> PoolFacts {
    // Sliding-window rows in a pool of their own, page 0 null.
    PoolFacts {
        kv_pages: 2_048,
        kv_page_size: 32,
        window_pages: 513,
        window_tokens: 4_096,
        host_kv_pages: 1_024,
        host_window_pages: 256,
        ..PoolFacts::default()
    }
}

fn ring_state() -> PoolFacts {
    // A model whose state is a window and nothing else: no full kv pages.
    PoolFacts {
        kv_page_size: 16,
        window_pages: 129,
        window_tokens: 1_024,
        host_window_pages: 64,
        ..PoolFacts::default()
    }
}

const BYTES: DegenerateBytes = DegenerateBytes {
    kv_page: 2 << 20,
    window_page: 4 << 20,
    buffer_page: 1 << 20,
};

#[test]
fn degenerate_facts_reproduce_every_pool_count() {
    for pools in [
        dense(),
        hybrid(),
        windowed(),
        ring_state(),
        PoolFacts::default(),
    ] {
        let facts = ChunkFacts::degenerate(&pools, &BYTES);
        facts.validate().unwrap();
        assert_eq!(
            facts.degenerate_counts(),
            DegenerateCounts::of(&pools),
            "{pools:?}"
        );
        let device = 1
            + pools.kv_pages
            + pools.window_pages.saturating_sub(1)
            + pools.state_slots
            + pools.buffer_pages;
        assert_eq!(facts.total_chunks, device);
        assert_eq!(
            facts.host_chunks,
            pools.host_kv_pages
                + pools.host_window_pages
                + pools.host_state_slots
                + pools.host_buffer_pages
        );
        assert_eq!(facts.disk_persistent, pools.disk_kv_persistent);
        assert_eq!(
            facts.state_chunks(),
            device - 1,
            "only the null chunk is reserved"
        );
        assert!(facts.is_reserved(0));
    }
}

#[test]
fn a_degenerate_slot_keeps_the_id_the_engine_knows_it_by() {
    let facts = ChunkFacts::degenerate(&windowed(), &BYTES);
    let kv = facts.shape(DEGENERATE_KV).unwrap();
    let ring = facts.shape(DEGENERATE_WINDOW).unwrap();
    assert_eq!(kv.legacy_id(1), Some(0));
    assert_eq!(kv.legacy_id(2_048), Some(2_047));
    assert_eq!(kv.legacy_id(2_049), None, "past the kv run is the ring's");
    assert_eq!(ring.legacy_id(2_049), Some(1), "window page 0 is chunk 0");
    assert_eq!(ring.legacy_id(2_049 + 511), Some(512));
    assert_eq!(ring.legacy_id(0), None);
    assert_eq!(kv.device_slots(&facts), 2_048);
    assert_eq!(facts.chunk_bytes, 4 << 20, "a chunk is the largest slot");

    let facts = ChunkFacts::degenerate(&hybrid(), &BYTES);
    let state = facts.shape(DEGENERATE_STATE).unwrap();
    let buffer = facts.shape(DEGENERATE_BUFFER).unwrap();
    assert_eq!(state.sets_per_working_set, 2);
    assert_eq!(state.legacy_id(4_097), Some(0));
    assert_eq!(buffer.legacy_id(4_097 + 66), Some(0));
    assert_eq!(buffer.pinned.unwrap().host.first, 8_192 + 128);
    assert!(facts.shape(DEGENERATE_WINDOW).is_none());
}

#[test]
fn chunk_facts_round_trip_through_serde() {
    let mut facts = ChunkFacts::degenerate(&hybrid(), &BYTES);
    facts.reserved.push(ReservedRun {
        what: Reserved::Runtimeless { lanes: 4 },
        first: facts.total_chunks,
        chunks: 4,
    });
    facts.total_chunks += 4;
    facts.segment_chunks = facts.total_chunks;
    facts.act_chunks = vec![(1, 1), (512, 3)];
    facts.shapes[0].rows = vec![3, 7, 11];
    facts.validate().unwrap();
    let json = serde_json::to_string(&facts).unwrap();
    assert_eq!(serde_json::from_str::<ChunkFacts>(&json).unwrap(), facts);

    let copy = SlotCopy {
        shape: ShapeId(2),
        src: MemoryDomain::CudaDevice(0),
        dst: MemoryDomain::HostPinned,
        slots: vec![(5, 9)],
        runs: vec![SlotRun {
            src: 1,
            src_token: 3,
            dst: 2,
            dst_token: 0,
            tokens: 13,
        }],
    };
    let json = serde_json::to_string(&copy).unwrap();
    assert_eq!(serde_json::from_str::<SlotCopy>(&json).unwrap(), copy);
    let evacuation = Evacuation {
        moves: vec![(4, 1), (9, 2)],
    };
    let json = serde_json::to_string(&evacuation).unwrap();
    assert_eq!(
        serde_json::from_str::<Evacuation>(&json).unwrap(),
        evacuation
    );
}

#[test]
fn a_lane_round_trips_with_and_without_its_slot_tables() {
    let lane = Lane {
        slot: 2,
        tokens: vec![1, 2, 3],
        tables: vec![(ShapeId(0), vec![4, 5]), (ShapeId(1), vec![0, 9])],
        fixed: vec![(ShapeId(2), vec![[11, 12], [13, 13]])],
        ..Lane::default()
    };
    let json = serde_json::to_string(&lane).unwrap();
    assert_eq!(serde_json::from_str::<Lane>(&json).unwrap(), lane);

    // A lane written before the tables existed still reads.
    let mut value = serde_json::to_value(Lane::decode(1, 0, 7, 3)).unwrap();
    let object = value.as_object_mut().unwrap();
    object.remove("tables");
    object.remove("fixed");
    let old: Lane = serde_json::from_value(value).unwrap();
    assert_eq!(old, Lane::decode(1, 0, 7, 3));
}

#[test]
fn a_lane_names_its_slots_only_to_an_engine_that_states_chunks() {
    let chunks = Serves {
        chunks: true,
        ..Serves::NONE
    };
    let lane = Lane {
        tokens: vec![1],
        tables: vec![(DEGENERATE_KV, vec![1, 2])],
        fixed: vec![(DEGENERATE_STATE, vec![[4_097, 4_098]])],
        ..Lane::default()
    };
    lane.validate_for(chunks).unwrap();
    assert!(lane.validate_for(Serves::NONE).is_err());
    Lane::decode(0, 0, 1, 0).validate_for(Serves::NONE).unwrap();

    let twice = Lane {
        fixed: vec![(DEGENERATE_KV, vec![[3, 4]])],
        ..lane.clone()
    };
    assert!(twice.validate_for(chunks).is_err(), "a shape is named once");
    let in_place = Lane {
        fixed: vec![(DEGENERATE_STATE, vec![[4_097, 4_097]])],
        ..lane.clone()
    };
    in_place
        .validate_for(chunks)
        .expect("cow == run writes the record in place");
    let no_piece = Lane {
        fixed: vec![(DEGENERATE_STATE, Vec::new())],
        ..lane.clone()
    };
    assert!(
        no_piece.validate_for(chunks).is_err(),
        "a fixed shape names its pieces"
    );

    let facts = ChunkFacts::degenerate(&hybrid(), &BYTES);
    lane.validate_chunks(&facts).unwrap();
    for bad in [
        // The null chunk is no kv page.
        Lane {
            tables: vec![(DEGENERATE_KV, vec![0])],
            ..lane.clone()
        },
        // Past the kv run.
        Lane {
            tables: vec![(DEGENERATE_KV, vec![4_097])],
            ..lane.clone()
        },
        // A fixed shape as a table, and a list shape as fixed.
        Lane {
            tables: vec![(DEGENERATE_STATE, vec![4_097])],
            fixed: Vec::new(),
            ..lane.clone()
        },
        Lane {
            fixed: vec![(DEGENERATE_KV, vec![[1, 2]])],
            tables: Vec::new(),
            ..lane.clone()
        },
        // A shape the engine does not state.
        Lane {
            tables: vec![(DEGENERATE_WINDOW, vec![0])],
            ..lane.clone()
        },
        // One piece more than the record has.
        Lane {
            fixed: vec![(DEGENERATE_STATE, vec![[4_097, 4_097], [4_098, 4_098]])],
            ..lane.clone()
        },
    ] {
        assert!(bad.validate_chunks(&facts).is_err(), "{bad:?}");
    }

    let facts = ChunkFacts::degenerate(&windowed(), &BYTES);
    Lane {
        tables: vec![(DEGENERATE_WINDOW, vec![0, 0, 2_049, 2_560])],
        ..Lane::default()
    }
    .validate_chunks(&facts)
    .unwrap();
}

#[test]
fn facts_that_do_not_hang_together_are_refused() {
    let good = ChunkFacts::degenerate(&hybrid(), &BYTES);
    let mut no_null = good.clone();
    no_null.reserved.clear();
    let mut overlap = good.clone();
    overlap.shapes[1].pinned.as_mut().unwrap().device.first -= 1;
    let mut too_big = good.clone();
    too_big.shapes[0].slots_per_chunk = 64;
    let mut rungs = good.clone();
    rungs.act_chunks = vec![(512, 2), (64, 1)];
    let mut segments = good.clone();
    segments.segments.committed = 0;
    let mut twice = good.clone();
    twice.shapes[1].id = twice.shapes[0].id;
    for bad in [no_null, overlap, too_big, rungs, segments, twice] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }
}

#[test]
fn a_rung_takes_the_smallest_activation_that_holds_it() {
    let facts = ChunkFacts {
        act_chunks: vec![(1, 1), (64, 2), (2_048, 8)],
        total_chunks: 100,
        reserved: vec![ReservedRun {
            what: Reserved::Null,
            first: 0,
            chunks: 1,
        }],
        ..ChunkFacts::default()
    };
    assert_eq!(facts.act_chunks_at(1), 1);
    assert_eq!(facts.act_chunks_at(2), 2);
    assert_eq!(facts.act_chunks_at(2_048), 8);
    assert_eq!(facts.act_chunks_at(9_999), 8);
    assert_eq!(facts.state_chunks(), 100 - 1 - 8);
}

#[test]
fn slot_refs_name_a_chunk_and_an_index() {
    let shape = SlotShape {
        id: ShapeId(0),
        kind: SlotKind::List {
            tokens: 16,
            rate: 4,
        },
        slot_bytes: 1 << 20,
        slots_per_chunk: 16,
        sets_per_working_set: 1,
        rows: Vec::new(),
        pinned: None,
    };
    assert_eq!(shape.slot_ref(3, 5), 53);
    assert_eq!(shape.chunk_of(53), 3);
    assert_eq!(shape.legacy_id(53), None);
}

#[test]
fn evacuation_moves_a_chunk_and_keeps_the_table_a_permutation() {
    let mut table = ChunkTable::identity(8);
    table
        .apply(&Evacuation {
            moves: vec![(6, 2), (7, 3)],
        })
        .unwrap();
    assert_eq!(table.physical(6), Some(2));
    assert_eq!(table.physical(7), Some(3));
    assert_eq!(table.physical(2), Some(6), "the free chunk took its place");
    assert_eq!(table.logical(2), Some(6));
    let mut seen = table.as_slice().to_vec();
    seen.sort_unstable();
    assert_eq!(seen, (0..8).collect::<Vec<_>>());
    for physical in 0..8 {
        let logical = table.logical(physical).unwrap();
        assert_eq!(table.physical(logical), Some(physical));
    }

    let before = table.clone();
    assert!(
        table
            .apply(&Evacuation {
                moves: vec![(1, 8)]
            })
            .is_err()
    );
    assert!(
        table
            .apply(&Evacuation {
                moves: vec![(1, 4), (1, 5)]
            })
            .is_err()
    );
    assert_eq!(table, before, "a refused evacuation changes nothing");
}

#[test]
fn a_pool_fits_whole_chunks_beside_the_largest_rung() {
    const C: u64 = 16 << 20;
    assert_eq!(fit(100 * C + 5, C, 8, 40), Ok(100));
    assert_eq!(fit(48 * C, C, 8, 40), Ok(48));
    assert_eq!(
        fit(47 * C, C, 8, 40),
        Err(FitShort {
            room: 47 * C,
            least: 48 * C
        })
    );
    assert!(fit(C, 0, 0, 0).is_err());
    assert_eq!(fit(u64::MAX, 1, 0, 1), Ok(u32::MAX));
    assert!(
        fit(47 * C, C, 8, 40)
            .unwrap_err()
            .to_string()
            .contains("MiB")
    );
}

#[test]
fn weight_shapes_round_trip_and_are_checked() {
    let mut facts = ChunkFacts::degenerate(&hybrid(), &BYTES);
    let experts = WeightShape {
        id: ShapeId(10),
        kind: WeightKind::Expert,
        slot_bytes: facts.chunk_bytes / 2,
        slots_per_chunk: 2,
        count: 48 * 128,
        floor_slots: 2 * 8,
        name: "layers.moe.experts".into(),
    };
    facts.weights.push(experts.clone());
    facts.validate().unwrap();
    assert_eq!(facts.weight(ShapeId(10)), Some(&experts));
    assert_eq!(experts.floor_bytes(), 16 * experts.slot_bytes);
    let text = serde_json::to_string(&facts).unwrap();
    assert_eq!(serde_json::from_str::<ChunkFacts>(&text).unwrap(), facts);

    let mut clash = facts.clone();
    clash.weights[0].id = facts.shapes[0].id;
    let mut too_big = facts.clone();
    too_big.weights[0].slots_per_chunk = 3;
    let mut floor = facts.clone();
    floor.weights[0].floor_slots = floor.weights[0].count + 1;
    let mut empty = facts.clone();
    empty.weights[0].count = 0;
    for bad in [clash, too_big, floor, empty] {
        assert!(bad.validate().is_err(), "{:?}", bad.weights);
    }
}

#[test]
fn chunk_facts_without_weights_read_as_none() {
    let facts = ChunkFacts::degenerate(&dense(), &BYTES);
    let mut value = serde_json::to_value(&facts).unwrap();
    value.as_object_mut().unwrap().remove("weights");
    let read: ChunkFacts = serde_json::from_value(value).unwrap();
    assert!(read.weights.is_empty());
}
