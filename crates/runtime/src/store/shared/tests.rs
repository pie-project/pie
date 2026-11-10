//! The shared pool against a fake engine that states real, unpinned facts:
//! a list, a window list, a fixed record of three pieces and one of one, a weight
//! shape, two-ended growth.

use ::engine::chunk::{
    ChunkTable, Growth as SegmentGrowth, Reserved, ReservedRun, SegmentFacts, SlotShape as Facts,
    WeightKind, WeightShape,
};
use ::engine::fire::Lane;

use super::*;

const MIB: u64 = 1 << 20;
const CHUNK: u64 = 16 * MIB;
const TOTAL: u32 = 64;
const SEGMENT: u32 = 8;
const LIST: EngineShape = EngineShape(0);
const RING: EngineShape = EngineShape(1);
const GDN: EngineShape = EngineShape(2);
const CONV: EngineShape = EngineShape(3);
/// The rs buffer's list: 4 rows a slot.
const RS: EngineShape = EngineShape(4);
const RS_TOKENS: u32 = 4;
/// Pool ids, in the facts' order.
const LIST_POOL: ShapeId = 0;
const RING_POOL: ShapeId = 1;

fn shape(id: EngineShape, kind: SlotKind, slot_bytes: u64) -> Facts {
    Facts {
        id,
        kind,
        slot_bytes,
        slots_per_chunk: u32::try_from(CHUNK / slot_bytes).unwrap(),
        sets_per_working_set: kind.sets_per_working_set(),
        rows: Vec::new(),
        pinned: None,
        role: SlotRole::State,
    }
}

/// 64 chunks of 16 MiB in segments of 8, the lowest two committed; the
/// largest rung's 12 chunks put segments 6 and 7 under activation.
fn facts() -> ChunkFacts {
    let facts = ChunkFacts {
        chunk_bytes: CHUNK,
        total_chunks: TOTAL,
        segment_chunks: SEGMENT,
        segments: SegmentFacts {
            committed: 2,
            max: TOTAL / SEGMENT,
            growth: SegmentGrowth::TwoEnded,
        },
        reserved: vec![ReservedRun {
            what: Reserved::Null,
            first: 0,
            chunks: 1,
        }],
        shapes: vec![
            shape(
                LIST,
                SlotKind::List {
                    tokens: 16,
                    rate: 1,
                    window: None,
                },
                MIB,
            ),
            shape(
                RING,
                SlotKind::List {
                    tokens: 16,
                    rate: 1,
                    window: Some(128),
                },
                MIB / 2,
            ),
            shape(GDN, SlotKind::Fixed { pieces: 3 }, 4 * MIB),
            shape(CONV, SlotKind::Fixed { pieces: 1 }, 64 << 10),
        ],
        act_chunks: vec![(16, 1), (256, 4), (2048, 12)],
        weights: vec![WeightShape {
            id: EngineShape(9),
            kind: WeightKind::Expert,
            slot_bytes: 8 * MIB,
            slots_per_chunk: 2,
            count: 8,
            floor_slots: 2,
            name: "layers.moe.experts".into(),
            seated: None,
        }],
        ..ChunkFacts::default()
    };
    facts.validate().unwrap();
    facts
}

fn pool() -> SharedPool {
    SharedPool::new(&facts(), 2, 256).unwrap()
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Call {
    Commit(u32),
    Release(u32),
    Evacuate(Vec<(u32, u32)>),
}

/// The engine half: a chunk table, a committed prefix and the top span
/// backed from the load. It checks what it is told against what it backs.
struct FakeEngine {
    table: ChunkTable,
    prefix: u32,
    top: u32,
    calls: Vec<Call>,
    refuse: bool,
}

impl FakeEngine {
    fn new() -> Self {
        Self {
            table: ChunkTable::identity(TOTAL),
            prefix: 2,
            top: 6,
            calls: Vec::new(),
            refuse: false,
        }
    }

    fn backed(&self, physical: u32) -> bool {
        let segment = physical / SEGMENT;
        segment < self.prefix || segment >= self.top
    }

    /// A fire at `act` activation chunks reading `lane`'s slots: every slot
    /// it names sits on a backed chunk below the activation.
    fn fire(&self, facts: &ChunkFacts, lane: &Lane, act: u32) {
        lane.validate_chunks(facts).unwrap();
        let refs =
            lane.tables
                .iter()
                .flat_map(|(shape, refs)| refs.iter().map(move |&r| (*shape, r)))
                .chain(lane.fixed.iter().flat_map(|(shape, pieces)| {
                    pieces.iter().flatten().map(move |&r| (*shape, r))
                }));
        for (shape, slot) in refs {
            if shape == RING && slot == 0 {
                continue;
            }
            let logical = facts.shape(shape).unwrap().chunk_of(slot);
            let physical = self.table.physical(logical).unwrap();
            assert!(self.backed(physical), "slot {slot} on unbacked {physical}");
            assert!(
                physical < TOTAL - act,
                "slot {slot} of shape {shape} on {physical}, in the activation"
            );
        }
    }
}

impl SegmentPort for FakeEngine {
    fn commit_segments(&mut self, segments: u32) -> ::engine::Result<()> {
        if self.refuse {
            return Err(::engine::Error::invalid("refused"));
        }
        assert!(segments > self.prefix && segments <= self.top);
        self.prefix = segments;
        self.calls.push(Call::Commit(segments));
        Ok(())
    }

    fn release_segments(&mut self, segments: u32) -> ::engine::Result<()> {
        assert!(segments < self.prefix);
        self.prefix = segments;
        self.calls.push(Call::Release(segments));
        Ok(())
    }

    fn evacuate(&mut self, evacuation: &EngineEvacuation) -> ::engine::Result<()> {
        if self.refuse {
            return Err(::engine::Error::invalid("refused"));
        }
        for &(_, physical) in &evacuation.moves {
            assert!(
                self.backed(physical),
                "an evacuation onto unbacked {physical}"
            );
        }
        self.table.apply(evacuation)?;
        self.calls.push(Call::Evacuate(evacuation.moves.clone()));
        Ok(())
    }
}

fn physical(pool: &SharedPool, shape: ShapeId, slot_ref: u32) -> u32 {
    pool.pool()
        .physical(pool.slot(shape, slot_ref).chunk)
        .unwrap()
}

#[test]
fn every_shape_opens_its_own_chunks_of_the_one_pool_lowest_first() {
    let mut pool = pool();
    assert_eq!(pool.pool().state_limit(), TOTAL - 12, "the largest rung");
    // 20 list slots of 16 a chunk, 5 ring slots of 32, two records of three
    // 4 MiB pieces (4 a chunk) and one 64 KiB piece (256 a chunk).
    let list = pool.take_refs(LIST_POOL, 20).unwrap();
    let ring = pool.take_refs(RING_POOL, 5).unwrap();
    let records = pool.take_records(2).unwrap();
    assert_eq!(records, [0, 1]);
    let mut chunks: Vec<(ShapeId, u32)> = list
        .iter()
        .map(|&r| (LIST_POOL, physical(&pool, LIST_POOL, r)))
        .chain(
            ring.iter()
                .map(|&r| (RING_POOL, physical(&pool, RING_POOL, r))),
        )
        .collect();
    for (at, shape) in pool.fixed().to_vec().into_iter().enumerate() {
        for record in &records {
            for &r in &pool.record_refs(*record)[at].1 {
                chunks.push((shape.pool, physical(&pool, shape.pool, r)));
            }
        }
    }
    chunks.sort_unstable();
    chunks.dedup();
    let opened: BTreeSet<u32> = chunks.iter().map(|&(_, c)| c).collect();
    assert_eq!(opened, (1..=6).collect(), "lowest first, none shared");
    assert_eq!(opened.len(), chunks.len(), "a chunk holds one shape");
    for (shape, chunk) in &chunks {
        let logical = pool
            .pool()
            .chunk_table()
            .iter()
            .position(|&p| p == *chunk)
            .unwrap();
        assert_eq!(pool.pool().chunk_shape(logical as u32), Some(*shape));
    }
    // A record's pieces: three of the 4 MiB shape, one of the conv's.
    let refs = pool.record_refs(records[1]);
    assert_eq!(refs[0].0, GDN);
    assert_eq!(refs[0].1.len(), 3);
    assert_eq!(refs[1].0, CONV);
    assert_eq!(refs[1].1.len(), 1);
    // A chunk let go is free for any shape: the list's second chunk goes to
    // a record next.
    pool.give_refs(LIST_POOL, &list[16..]);
    let next = pool.take_records(1).unwrap();
    assert_eq!(
        physical(
            &pool,
            pool.fixed()[0].pool,
            pool.record_refs(next[0])[0].1[2]
        ),
        2,
        "the list's chunk, now the 4 MiB shape's"
    );
}

#[test]
fn records_count_what_the_chunks_hold_and_come_back_whole() {
    let mut pool = pool();
    // 62 chunks state may take under the smallest rung: 3 pieces of 4 a
    // chunk and 1 of 256 a record.
    assert_eq!(pool.records_capacity(), 81);
    let free = pool.records_available();
    // 19 chunks open now: 15 below, 4 of the top span under the largest
    // rung; 18 for 24 records' pieces of 4 MiB, 1 for their conv pieces.
    assert_eq!(free, 24);
    let records = pool.take_records(free as usize).unwrap();
    assert!(pool.take_records(1).is_none(), "all or nothing");
    pool.give_records(&records);
    assert_eq!(pool.records_available(), free);
    assert_eq!(pool.pool().state_chunks(), 0);
}

#[test]
fn a_lane_names_each_piece_run_and_cow_in_place_unless_it_forks() {
    let facts = facts();
    let mut pool = pool();
    let engine = FakeEngine::new();
    let list = pool.take_refs(LIST_POOL, 3).unwrap();
    let ring = pool.take_refs(RING_POOL, 2).unwrap();
    assert!(!ring.contains(&0), "ref 0 is the null chunk's");
    let [run, cow] = pool.take_records(2).unwrap()[..] else {
        panic!("two records");
    };
    let act = pool.pool().act_reserved();
    // Decode: the record written in place.
    let decode = Lane {
        tokens: vec![1],
        tables: pool.lane_tables(&list, &[vec![0, ring[0], ring[1]]]),
        fixed: pool.lane_fixed(run, run),
        ..Lane::default()
    };
    assert_eq!(decode.tables[0], (LIST, list.clone()));
    assert_eq!(
        decode.tables[1],
        (RING, vec![0, ring[0], ring[1]]),
        "a full-length table, ref 0 the null chunk where the window trimmed"
    );
    assert_eq!(decode.fixed.len(), 2);
    assert_eq!(decode.fixed[0].1.len(), 3, "one [run, cow] a piece");
    assert!(
        decode
            .fixed
            .iter()
            .flat_map(|(_, pieces)| pieces)
            .all(|[run, cow]| run == cow)
    );
    decode
        .validate_for(::engine::fire::Serves {
            chunks: true,
            ..::engine::fire::Serves::NONE
        })
        .unwrap();
    engine.fire(&facts, &decode, act);
    // A fork in the fire: the shared record is the run, a fresh one the cow.
    let fork = Lane {
        fixed: pool.lane_fixed(run, cow),
        ..decode.clone()
    };
    for ((_, forked), (_, own)) in fork.fixed.iter().zip(&decode.fixed) {
        for ([run, cow], [kept, _]) in forked.iter().zip(own) {
            assert_eq!(run, kept, "the run is the record forked from");
            assert_ne!(run, cow);
        }
    }
    engine.fire(&facts, &fork, act);
    assert_eq!(pool.snapshot_ref(cow), fork.fixed[0].1[0][1]);
}

#[test]
fn raising_the_rung_moves_state_out_of_the_activation_before_the_next_fire() {
    let facts = facts();
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    // Back everything below the top span, then let state into the top span
    // by lowering the rung.
    assert_eq!(pool.steer(8, 0, &mut engine), Retarget::Reached);
    assert_eq!(engine.prefix, 6);
    pool.lower_rung(16).unwrap();
    assert_eq!(pool.pool().state_limit(), TOTAL - 1);
    // Fill every state chunk with records, then let the low ones go.
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    let kept: Vec<u32> = all[all.len() - 6..].to_vec();
    pool.give_records(&all[..all.len() - 6]);
    let refs: Vec<_> = kept.iter().map(|&r| pool.record_refs(r)).collect();
    let in_way = |pool: &SharedPool| {
        kept.iter()
            .flat_map(|&r| pool.record_slots(r))
            .filter(|slot| pool.pool().physical(slot.chunk).unwrap() >= TOTAL - 12)
            .count()
    };
    assert!(in_way(&pool) > 0, "state sits in the largest rung's span");
    // The engine refuses: nothing moves, the rung stays.
    engine.refuse = true;
    assert!(matches!(
        pool.raise_rung(2048, 3, &mut engine),
        Err(RungError::Engine(_))
    ));
    assert_eq!(pool.pool().act_reserved(), 1);
    engine.refuse = false;
    pool.raise_rung(2048, 3, &mut engine).unwrap();
    assert!(matches!(engine.calls.last(), Some(Call::Evacuate(_))));
    assert_eq!(pool.pool().act_reserved(), 12);
    assert_eq!(in_way(&pool), 0);
    assert_eq!(engine.table.as_slice(), pool.pool().chunk_table());
    for (&record, before) in kept.iter().zip(&refs) {
        assert_eq!(&pool.record_refs(record), before, "slot refs are logical");
        let lane = Lane {
            tokens: vec![1],
            fixed: pool.lane_fixed(record, record),
            ..Lane::default()
        };
        engine.fire(&facts, &lane, 12);
    }
    // Raising to a rung already set aside moves nothing more.
    let calls = engine.calls.len();
    pool.raise_rung(256, 4, &mut engine).unwrap();
    assert_eq!(engine.calls.len(), calls);
}

#[test]
fn segments_commit_ahead_as_a_prefix_and_shrink_by_evacuating_the_highest() {
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    assert_eq!(pool.mapped_segments(), 4, "two below, two under activation");
    // A waiter short of 30 chunks: commit ahead of it and the step.
    let target = pool.ahead_target(30 * CHUNK);
    assert_eq!(
        target, 6,
        "19 free, 30 and the 4 ahead wanted: two segments more"
    );
    assert_eq!(pool.steer(target, 0, &mut engine), Retarget::Reached);
    assert_eq!(engine.calls, [Call::Commit(3), Call::Commit(4)]);
    assert_eq!(pool.steer(8, 0, &mut engine), Retarget::Reached);
    assert_eq!(
        engine.calls,
        [
            Call::Commit(3),
            Call::Commit(4),
            Call::Commit(5),
            Call::Commit(6)
        ]
    );
    assert_eq!(pool.pool().prefix_segments(), 6);
    assert!(pool.ahead_target(0) <= 8, "never past the pool");
    // Hold chunks across the prefix, then free most of them.
    let held = pool.take_refs(LIST_POOL, 16 * 44).unwrap();
    let keep: Vec<u32> = held
        .chunks(16)
        .enumerate()
        .filter(|(at, _)| at % 8 == 0)
        .flat_map(|(_, refs)| refs.to_vec())
        .collect();
    let gone: Vec<u32> = held.iter().copied().filter(|r| !keep.contains(r)).collect();
    pool.give_refs(LIST_POOL, &gone);
    assert!(pool.pool().segments().used(5) > 0);
    // Shrink by one segment: the highest of the prefix evacuates first.
    engine.calls.clear();
    assert_eq!(pool.steer(7, 9, &mut engine), Retarget::Reached);
    let [Call::Evacuate(moves)] = &engine.calls[..] else {
        panic!("{:?}", engine.calls);
    };
    assert!(moves.iter().all(|&(_, to)| to < 5 * SEGMENT));
    assert_eq!(pool.pool().segments().used(5), 0);
    assert_eq!(engine.table.as_slice(), pool.pool().chunk_table());
    // Released only once the fire carrying the copy retires.
    assert_eq!(pool.steer(7, 9, &mut engine), Retarget::Reached);
    assert_eq!(engine.prefix, 6, "still draining");
    pool.pool_mut().retire_through(9);
    assert_eq!(pool.steer(7, 9, &mut engine), Retarget::Reached);
    assert_eq!(engine.prefix, 5);
    assert_eq!(pool.mapped_segments(), 7);
    // A shrink with no room below says so and drops nothing.
    let rest = pool
        .take_refs(LIST_POOL, pool.pool().available(LIST_POOL))
        .unwrap();
    assert!(!rest.is_empty());
    assert!(matches!(
        pool.steer(4, 10, &mut engine),
        Retarget::NoRoom { held: 7 }
    ));
    // The top span never goes.
    assert_eq!(pool.pool().segments().state(6), SegmentState::Committed);
    assert_eq!(pool.pool().segments().state(7), SegmentState::Committed);
}

#[test]
fn a_refused_commit_hands_out_nothing_of_the_segment() {
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    engine.refuse = true;
    let before = pool.pool().available(LIST_POOL);
    assert_eq!(pool.steer(5, 0, &mut engine), Retarget::Reached);
    assert_eq!(pool.pool().available(LIST_POOL), before);
    assert_eq!(pool.pool().segments().state(2), SegmentState::Uncommitted);
}

#[test]
fn governor_bytes_count_the_top_span_as_committed_from_the_load() {
    let mut pool = pool();
    // Two segments below and the two under activation.
    assert_eq!(pool.held(), 4 * u64::from(SEGMENT) * CHUNK);
    assert_eq!(pool.hard_limit(), u64::from(TOTAL) * CHUNK);
    assert_eq!(pool.segment_bytes(), u64::from(SEGMENT) * CHUNK);
    let empty = pool.demand();
    assert_eq!(
        empty.live,
        (1 + 12) * CHUNK,
        "the null chunk and the top span"
    );
    assert_eq!(empty.wanted, empty.live);
    // A working set at 256 tokens: 16 list slots (1 chunk), 9 window slots
    // (ceil(128 / 16) + 1, 1 chunk), a record (1 + 1 chunks).
    assert_eq!(pool.floor(), 4 * CHUNK + 13 * CHUNK);
    assert!(matches!(pool.growth(), Growth::Elastic { .. }));
    let list = pool.take_refs(LIST_POOL, 4).unwrap();
    assert_eq!(
        pool.demand().live,
        13 * CHUNK + 4 * MIB,
        "slots at their own size"
    );
    pool.give_refs(LIST_POOL, &list);
    // A fixed pool reads as fixed.
    let mut fixed = facts();
    fixed.segments = SegmentFacts {
        committed: 8,
        max: 8,
        growth: SegmentGrowth::Fixed,
    };
    let fixed = SharedPool::new(&fixed, 2, 256).unwrap();
    assert_eq!(fixed.growth(), Growth::Fixed);
    assert_eq!(fixed.held(), u64::from(TOTAL) * CHUNK);
}

#[test]
fn pinned_or_unready_facts_are_refused() {
    let degenerate = ChunkFacts::degenerate(
        &::engine::PoolFacts {
            kv_pages: 8,
            kv_page_size: 16,
            ..Default::default()
        },
        &::engine::chunk::DegenerateBytes {
            kv_page: 64,
            ..Default::default()
        },
    );
    assert!(!SharedPool::states_shared(&degenerate));
    assert!(matches!(
        SharedPool::new(&degenerate, 1, 16),
        Err(SharedError::Pinned(0))
    ));
    assert!(SharedPool::states_shared(&facts()));
    let mut two_lists = facts();
    two_lists.shapes.push(shape(
        EngineShape(5),
        SlotKind::List {
            tokens: 16,
            rate: 4,
            window: None,
        },
        MIB,
    ));
    assert!(matches!(
        SharedPool::new(&two_lists, 1, 16),
        Err(SharedError::Shapes {
            kind: "kv list",
            ..
        })
    ));
    // CONTRACT-BC 18: one kv list and one rs list stand together; a second
    // rs list does not.
    let rs_list = |id| Facts {
        role: SlotRole::RsBuffer,
        ..shape(
            id,
            SlotKind::List {
                tokens: RS_TOKENS,
                rate: 1,
                window: None,
            },
            MIB / 4,
        )
    };
    let mut kv_and_rs = facts();
    kv_and_rs.shapes.push(rs_list(EngineShape(5)));
    let pool = SharedPool::new(&kv_and_rs, 1, 16).unwrap();
    assert_eq!(pool.list().map(|list| list.id), Some(LIST));
    assert_eq!(pool.rs_list().map(|list| list.id), Some(EngineShape(5)));
    let mut two_rs = kv_and_rs;
    two_rs.shapes.push(rs_list(EngineShape(6)));
    assert!(matches!(
        SharedPool::new(&two_rs, 1, 16),
        Err(SharedError::Shapes {
            kind: "rs list",
            ..
        })
    ));
}

/// The facts without the ring: a model whose state is fixed records, its
/// unfolded rows in an rs list (CONTRACT-BC 18), the shapes' last.
fn fixed_facts() -> ChunkFacts {
    let mut facts = facts();
    facts.shapes.retain(|shape| shape.id != RING);
    facts.shapes.push(Facts {
        role: SlotRole::RsBuffer,
        ..shape(
            RS,
            SlotKind::List {
                tokens: RS_TOKENS,
                rate: 1,
                window: None,
            },
            MIB / 4,
        )
    });
    facts.validate().unwrap();
    facts
}

/// `fixed_facts`' rs list's pool id: the shapes' last.
const RS_POOL: ShapeId = 3;

/// State a waiter lacks is kv to park: the kv list slots whose chunks, let go
/// whole, open for the rs list's slots and for whole records.
#[test]
fn state_short_counts_as_the_kv_list_slots_whose_chunks_it_takes() {
    use crate::store::rs::RsUnits;
    let pool = SharedPool::new(&fixed_facts(), 2, 256).unwrap();
    let spc = |shape: ShapeId| u64::from(pool.pool().shape(shape).unwrap().slots_per_chunk);
    let list = spc(LIST_POOL);
    let rs = spc(RS_POOL);
    assert_eq!(pool.list_slots_for(RsUnits::default()), 0);
    // One rs list slot past a chunk's takes two chunks.
    assert_eq!(
        pool.list_slots_for(RsUnits::of(0, u32::try_from(rs).unwrap() + 1)),
        2 * list
    );
    // A record: each fixed shape's pieces, chunk by chunk.
    let record: u64 = pool
        .fixed()
        .iter()
        .map(|shape| u64::from(shape.pieces).div_ceil(spc(shape.pool)))
        .sum();
    assert!(record > 0);
    assert_eq!(pool.list_slots_for(RsUnits::of(1, 0)), record * list);

    // A model whose only state is its windows (gemma): window list slots,
    // each list by its own chunks, narrowest first.
    let windowed = SharedPool::new(&window_facts(), 2, 256).unwrap();
    let spc = |pool: &SharedPool, shape: ShapeId| {
        u64::from(pool.pool().shape(shape).unwrap().slots_per_chunk)
    };
    let list = spc(&windowed, LIST_POOL);
    let narrow = windowed.windows()[0];
    let wide = windowed.windows()[1];
    let mut lists = [0; crate::store::rs::MAX_WINDOW_LISTS];
    lists[0] = u32::try_from(spc(&windowed, narrow.pool)).unwrap();
    lists[1] = u32::try_from(spc(&windowed, wide.pool)).unwrap() + 1;
    assert_eq!(
        windowed.list_slots_for(RsUnits::of_windows(lists)),
        3 * list,
        "one chunk of the narrow list, two of the wide"
    );
}

#[test]
fn the_stores_take_list_refs_and_records_and_a_fork_names_its_cow() {
    use crate::pipeline::fire::rs::{PreparedRs, apply_shared, prepare, settle};
    use crate::store::rs::RsGeometry;

    let facts = fixed_facts();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 2, 256).unwrap()));
    let stores = crate::store::registry::shared_stores(&shared, &Tiers::default(), 16, 256);
    assert!(stores.shared.is_some());
    // kv pages are the list shape's slot refs, from the one pool.
    let pages = crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
        kv.reserve_device_pages(3).unwrap()
    });
    let refs: Vec<u32> = pages.iter().map(|page| page.0).collect();
    {
        let pool = shared.lock().unwrap();
        for &r in &refs {
            let chunk = pool.slot(LIST_POOL, r).chunk;
            assert_eq!(pool.pool().chunk_shape(chunk), Some(LIST_POOL));
        }
    }
    let engine = FakeEngine::new();
    let geom = RsGeometry {
        state_size: 1024,
        buffer_page_tokens: 4,
    };
    let mut rs = stores.rs.lock().unwrap();
    assert!(rs.is_shared());
    let ws = rs.create_working_set(geom);
    let first = prepare(&mut rs, ws, 4).unwrap();
    assert!(
        first.state_copy().is_none(),
        "no StateCopy: records copy by [run, cow]"
    );
    let lane = |out: &PreparedRs| {
        let mut req = crate::engine::FireRequest {
            lanes: vec![crate::engine::fire::lane_of(0, vec![1], 0, refs.clone())],
            ..crate::engine::FireRequest::default()
        };
        out.apply_to(&mut req);
        apply_shared(&mut req, Some(&shared), Some(out));
        req.lanes.remove(0)
    };
    let in_place = |lane: &Lane| {
        !lane.fixed.is_empty()
            && lane
                .fixed
                .iter()
                .flat_map(|(_, pieces)| pieces)
                .all(|[run, cow]| run == cow)
    };
    let fresh = lane(&first);
    assert_eq!(fresh.tables, [(LIST, refs.clone())]);
    assert!(in_place(&fresh), "a fresh record is written in place");
    engine.fire(&facts, &fresh, 12);
    settle(&mut rs, first.txn);
    let record = rs.folded_slot(ws).unwrap().unwrap().0;
    // A decode on its own record: cow == run.
    let decode = prepare(&mut rs, ws, 1).unwrap();
    let decoded = lane(&decode);
    assert!(in_place(&decoded));
    assert_eq!(
        decoded.fixed,
        shared.lock().unwrap().lane_fixed(record, record)
    );
    settle(&mut rs, decode.txn);
    // The fork shares the record; its first fire writes a fresh one, the
    // engine copying the shared run onto it.
    let fork = rs.fork(ws).unwrap();
    let forked = prepare(&mut rs, fork, 1).unwrap();
    let forked_lane = lane(&forked);
    let cow = forked.slot_ids[0];
    assert_ne!(cow, record);
    assert_eq!(
        forked_lane.fixed,
        shared.lock().unwrap().lane_fixed(record, cow)
    );
    assert!(!in_place(&forked_lane));
    engine.fire(&facts, &forked_lane, 12);
    settle(&mut rs, forked.txn);
    // Back to its own record, the original writes in place again.
    let again = prepare(&mut rs, ws, 1).unwrap();
    assert!(in_place(&lane(&again)));
    settle(&mut rs, again.txn);
}

#[test]
fn degenerate_facts_keep_their_own_pools() {
    let pools = ::engine::PoolFacts {
        kv_pages: 10,
        kv_page_size: 16,
        state_slots: 4,
        state_slot_bytes: 4096,
        buffer_pages: 6,
        ..Default::default()
    };
    let facts = ChunkFacts::degenerate(
        &pools,
        &::engine::chunk::DegenerateBytes {
            kv_page: 100,
            window_page: 50,
            buffer_page: 25,
        },
    );
    let stores = crate::store::chunks::StorePools::of(&facts).unwrap();
    assert!(!stores.shared);
    assert_eq!(stores.kv.device, 10);
    let shared = crate::store::chunks::StorePools::of(&fixed_facts()).unwrap();
    assert!(shared.shared);
    assert_eq!(shared.page_size, 16);
    let mut mixed = fixed_facts();
    mixed.shapes[0].pinned = Some(::engine::chunk::Pinned::default());
    assert!(crate::store::chunks::StorePools::of(&mixed).is_err());
}

/// A lane's [`Lane::fixed`].
type LaneFixed = Vec<(EngineShape, Vec<[u32; 2]>)>;

#[test]
fn every_lane_of_a_group_names_the_record_the_group_writes() {
    use crate::pipeline::fire::rs::{
        GroupFold, GroupPlan, PreparedRs, RsPlan, apply_shared, prepare, prepare_many, settle,
    };
    use crate::store::rs::RsGeometry;

    let facts = fixed_facts();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 2, 256).unwrap()));
    let stores = crate::store::registry::shared_stores(&shared, &Tiers::default(), 16, 256);
    let geom = RsGeometry {
        state_size: 1024,
        buffer_page_tokens: 4,
    };
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(geom);
    let first = prepare(&mut rs, ws, 4).unwrap();
    settle(&mut rs, first.txn);
    let group = |writes: std::ops::Range<u32>, fold: GroupFold| {
        RsPlan::Groups(vec![GroupPlan {
            lanes: vec![0, 1, 2],
            writes: Some(writes),
            fold,
            snapshots: Vec::new(),
        }])
    };
    let fixed = |out: &PreparedRs| -> Vec<_> {
        let mut req = crate::engine::FireRequest {
            lanes: (0..3)
                .map(|row| crate::engine::fire::lane_of(row, vec![1], 0, Vec::new()))
                .collect(),
            ..crate::engine::FireRequest::default()
        };
        out.apply_to(&mut req);
        apply_shared(&mut req, Some(&shared), Some(out));
        req.lanes.into_iter().map(|lane| lane.fixed).collect()
    };

    // CONTRACT-BC 15: the engine runs a fire's lanes at once, a lane that
    // folds nothing reading `run` while a folding one writes `cow`; so with
    // any lane folding, no lane of the group names its record in place.
    let copied_on_every_lane = |lanes: &[LaneFixed]| {
        lanes.iter().all(|fixed| {
            !fixed.is_empty()
                && fixed
                    .iter()
                    .flat_map(|(_, pieces)| pieces)
                    .all(|[run, cow]| run != cow)
        })
    };

    // One lane folds: the group writes a fresh record, and every lane's
    // buffered rows land in it.
    let held = rs.folded_slot(ws).unwrap().unwrap().0;
    let out = prepare_many(&mut rs, &[ws; 3], &group(4..7, GroupFold::Host(1)), None).unwrap();
    let cow = out.slot_ids[0];
    assert_ne!(cow, held);
    let want = shared.lock().unwrap().lane_fixed(held, cow);
    let lanes = fixed(&out);
    assert!(copied_on_every_lane(&lanes), "{lanes:?}");
    assert_eq!(lanes, vec![want.clone(), want.clone(), want]);
    settle(&mut rs, out.txn);
    assert_eq!(rs.folded_slot(ws).unwrap().map(|slot| slot.0), Some(cow));

    // Nothing folds on its own record: every lane writes it in place.
    let out = prepare_many(&mut rs, &[ws; 3], &group(5..8, GroupFold::None), None).unwrap();
    let want = shared.lock().unwrap().lane_fixed(cow, cow);
    assert_eq!(fixed(&out), vec![want.clone(), want.clone(), want]);
    settle(&mut rs, out.txn);

    // A fork shares the record: buffering onto it with nothing folding
    // leaves the record alone (the rows are rs list slots), so every lane
    // reads it in place.
    let fork = rs.fork(ws).unwrap();
    let out = prepare_many(&mut rs, &[fork; 3], &group(5..9, GroupFold::None), None).unwrap();
    assert_eq!(out.slot_ids[0], cow);
    let want = shared.lock().unwrap().lane_fixed(cow, cow);
    assert_eq!(fixed(&out), vec![want.clone(), want.clone(), want]);
    settle(&mut rs, out.txn);
    rs.release_working_set(fork);

    // A lone lane that folds writes its own record in place.
    let solo = rs.create_working_set(geom);
    let first = prepare(&mut rs, solo, 4).unwrap();
    settle(&mut rs, first.txn);
    let own = rs.folded_slot(solo).unwrap().unwrap().0;
    let out = prepare(&mut rs, solo, 1).unwrap();
    let mut req = crate::engine::FireRequest {
        lanes: vec![crate::engine::fire::lane_of(0, vec![1], 0, Vec::new())],
        ..crate::engine::FireRequest::default()
    };
    out.apply_to(&mut req);
    apply_shared(&mut req, Some(&shared), Some(&out));
    assert_eq!(
        req.lanes[0].fixed,
        shared.lock().unwrap().lane_fixed(own, own)
    );
    settle(&mut rs, out.txn);

    // A fold the device makes is a fold: every lane takes the copy too.
    let frozen = rs.frozen(solo).unwrap();
    let out = prepare_many(
        &mut rs,
        &[solo; 3],
        &group(frozen..frozen + 3, GroupFold::Device),
        None,
    )
    .unwrap();
    let device = out.slot_ids[0];
    assert_ne!(device, own);
    let lanes = fixed(&out);
    assert!(copied_on_every_lane(&lanes), "{lanes:?}");
    let want = shared.lock().unwrap().lane_fixed(own, device);
    assert_eq!(lanes, vec![want.clone(), want.clone(), want]);
    settle(&mut rs, out.txn);
}

#[test]
fn a_snapshot_names_every_piece_of_its_record() {
    use crate::pipeline::fire::rs::{
        GroupFold, GroupPlan, RsPlan, apply_shared, prepare, prepare_many, settle,
    };
    use crate::store::rs::RsGeometry;

    let facts = fixed_facts();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 2, 256).unwrap()));
    let stores = crate::store::registry::shared_stores(&shared, &Tiers::default(), 16, 256);
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(RsGeometry {
        state_size: 1024,
        buffer_page_tokens: 4,
    });
    let first = prepare(&mut rs, ws, 4).unwrap();
    settle(&mut rs, first.txn);
    let plan = RsPlan::Groups(vec![GroupPlan {
        lanes: vec![0],
        writes: None,
        fold: GroupFold::Host(4),
        snapshots: vec![2],
    }]);
    let out = prepare_many(&mut rs, &[ws], &plan, None).unwrap();
    let mut req = crate::engine::FireRequest {
        lanes: vec![crate::engine::fire::lane_of(0, vec![1; 4], 0, Vec::new())],
        ..crate::engine::FireRequest::default()
    };
    out.apply_to(&mut req);
    apply_shared(&mut req, Some(&shared), Some(&out));
    let [snap] = &req.lanes[0].rs_snapshots[..] else {
        panic!("one snapshot");
    };
    let record = out.snapshots[0].slot.0;
    let refs = shared.lock().unwrap().record_refs(record);
    assert_eq!(refs[0].0, GDN);
    assert_eq!(snap.row, 2);
    assert_eq!(snap.pieces, refs[0].1, "every piece, in piece order");
    assert_eq!(snap.pieces.len(), 3);
    assert_eq!(snap.slot, snap.pieces[0], "the slot stays the first piece");
    settle(&mut rs, out.txn);

    // A record of one piece is named by its slot alone.
    let mut one = fixed_facts();
    one.shapes.retain(|shape| shape.id != GDN);
    let mut pool = SharedPool::new(&one, 2, 256).unwrap();
    let [record] = pool.take_records(1).unwrap()[..] else {
        panic!("one record");
    };
    assert!(pool.snapshot_pieces(record).is_empty());
    assert_eq!(pool.snapshot_ref(record), pool.record_refs(record)[0].1[0]);
}

/// CONTRACT-BC 17: every fire takes an epoch of the kv store, so a drained
/// segment goes once the first fire after the engine took its copy retires
/// (before, the epoch moved only on a copy-on-write, and a drain waited
/// on a fire that never came).
#[test]
fn a_drained_segment_goes_once_the_next_fire_retires() {
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    let mut kv = crate::store::kv::KvStore::new(8, [7u8; 32]);
    assert_eq!(pool.steer(8, 0, &mut engine), Retarget::Reached);
    let held = pool.take_refs(LIST_POOL, 16 * 44).unwrap();
    let keep: Vec<u32> = held
        .chunks(16)
        .enumerate()
        .filter(|(at, _)| at % 8 == 0)
        .flat_map(|(_, refs)| refs.to_vec())
        .collect();
    let gone: Vec<u32> = held.iter().copied().filter(|r| !keep.contains(r)).collect();
    pool.give_refs(LIST_POOL, &gone);
    // A decode in flight that copies nothing still holds an epoch.
    let decode = kv.begin_fire();
    let walk = |pool: &mut SharedPool, kv: &crate::store::kv::KvStore, engine: &mut FakeEngine| {
        pool.pool_mut().retire_through(kv.retired_through());
        pool.steer(7, kv.current_epoch() + 1, engine)
    };
    engine.calls.clear();
    assert_eq!(walk(&mut pool, &kv, &mut engine), Retarget::Reached);
    assert!(matches!(engine.calls[..], [Call::Evacuate(_)]));
    // A second fire is handed out before the engine answers the copy: it
    // may run ahead of the copy, and the drain waits on it too.
    let racing = kv.begin_fire();
    pool.evacuated(kv.current_epoch());
    let draining = || SegmentState::Draining { until: racing + 1 };
    assert_eq!(pool.pool().segments().state(5), draining());
    kv.settle(decode);
    kv.settle(racing);
    assert_eq!(walk(&mut pool, &kv, &mut engine), Retarget::Reached);
    assert_eq!(engine.prefix, 6, "no fire after the copy has retired yet");
    // The next fire, a decode that copies nothing, runs after the copy.
    let next = kv.begin_fire();
    assert_eq!(next, racing + 1);
    assert_eq!(walk(&mut pool, &kv, &mut engine), Retarget::Reached);
    assert_eq!(engine.prefix, 6, "it is still in flight");
    kv.settle(next);
    assert_eq!(walk(&mut pool, &kv, &mut engine), Retarget::Reached);
    assert_eq!(engine.prefix, 5, "released");
    assert_eq!(engine.calls.last(), Some(&Call::Release(5)));
}

/// CONTRACT-BC 17: a page let go while a fire that reads it runs goes back
/// to the shared pool only once that fire retires, though the fire copies
/// nothing (before, it went back at once).
#[test]
fn a_freed_slot_waits_for_the_fire_that_read_it() {
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&fixed_facts(), 2, 256).unwrap()));
    let stores = crate::store::registry::shared_stores(&shared, &Tiers::default(), 16, 256);
    let available = || shared.lock().unwrap().pool().available(LIST_POOL);
    crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
        let ws = kv.create_working_set();
        kv.reserve(ws, 1).unwrap();
        let mut granted = kv.reserve_device_pages(1).unwrap();
        let (_, txn) =
            crate::pipeline::fire::kv::prepare_explicit_reserved(kv, ws, &[0], &mut granted)
                .unwrap();
        crate::pipeline::fire::kv::settle(kv, txn);
        let page = kv.lookup(ws, 0).unwrap();
        let before = available();
        // A decode reads the page; its working set is let go meanwhile.
        let fire = crate::pipeline::fire::kv::begin_fire(kv);
        let epoch = kv.current_epoch();
        kv.release_working_set(ws, epoch);
        kv.retire_idle();
        assert_eq!(available(), before, "{page:?} waits on the fire");
        crate::pipeline::fire::kv::settle(kv, fire);
        assert_eq!(available(), before + 1, "{page:?} back once it retired");
    });
}

/// What the Metal engine states for a dense model on its sparse pool: one
/// list shape a page a chunk, bottom growth with one segment of many
/// committed at load, and no activation span (activation keeps its arena).
fn metal_like_facts() -> ChunkFacts {
    let facts = ChunkFacts {
        chunk_bytes: 2 * MIB,
        total_chunks: 1024,
        segment_chunks: 32,
        segments: SegmentFacts {
            committed: 1,
            max: 1024 / 32,
            growth: SegmentGrowth::Bottom,
        },
        reserved: vec![ReservedRun {
            what: Reserved::Null,
            first: 0,
            chunks: 1,
        }],
        shapes: vec![Facts {
            slots_per_chunk: 1,
            ..shape(
                LIST,
                SlotKind::List {
                    tokens: 16,
                    rate: 1,
                    window: None,
                },
                2 * MIB,
            )
        }],
        act_chunks: Vec::new(),
        ..ChunkFacts::default()
    };
    facts.validate().unwrap();
    facts
}

/// Pages out are the pages handed out, not the segments still uncommitted:
/// an idle pool with most of its segments uncommitted read as kv pressure
/// near 255, and the gateway held every launch for headroom no fire would
/// ever come to commit.
#[test]
fn uncommitted_segments_are_not_pages_out() {
    let facts = metal_like_facts();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 1, 4096).unwrap()));
    let stores = crate::store::registry::shared_stores(&shared, &Tiers::default(), 16, 4096);
    let counts = || {
        crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
            (kv.capacity_pages(), kv.pages_out())
        })
    };
    let (capacity, out) = counts();
    assert_eq!(capacity, 1023, "every chunk but the null one");
    assert_eq!(out, 0, "an idle pool hands out nothing");
    crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
        kv.reserve_device_pages(3).unwrap()
    });
    assert_eq!(counts(), (1023, 3));
}

/// The rs list's slots the store holds now (CONTRACT-BC 18), checking that
/// each of `refs` is one of them.
fn rs_live(shared: &Shared, refs: &[u32]) -> u64 {
    let pool = shared.lock().unwrap();
    for &r in refs {
        let chunk = pool.slot(RS_POOL, r).chunk;
        assert_eq!(pool.pool().chunk_shape(chunk), Some(RS_POOL), "ref {r}");
    }
    pool.pool().shape_demand(RS_POOL).live
}

/// The lanes of `out` as a shared pool's engine gets them.
fn shared_lanes(
    out: &crate::pipeline::fire::rs::PreparedRs,
    shared: &Shared,
    lanes: u32,
) -> Vec<Lane> {
    let mut req = crate::engine::FireRequest {
        lanes: (0..lanes)
            .map(|row| crate::engine::fire::lane_of(row, vec![1], 0, Vec::new()))
            .collect(),
        ..crate::engine::FireRequest::default()
    };
    out.apply_to(&mut req);
    crate::pipeline::fire::rs::apply_shared(&mut req, Some(shared), Some(out));
    req.lanes
}

fn buffer_of(lane: &Lane) -> (Vec<u32>, u32, u32) {
    match &lane.rs {
        ::engine::fire::RsVerb::Buffer {
            pages,
            base,
            folded,
            ..
        } => (pages.clone(), *base, *folded),
        ::engine::fire::RsVerb::Fold => panic!("a buffering lane"),
    }
}

/// CONTRACT-BC 18: beam lanes on one working set leave 36 rows unfolded
/// (past the 16 a record once buffered), in rs list slots taken as the rows
/// accumulate, `base` aligned to the slot; folding gives the slots wholly
/// below the state back, and releasing the set the rest.
#[test]
fn beam_lanes_leave_their_rows_unfolded_in_rs_list_slots() {
    use crate::pipeline::fire::rs::{
        GroupFold, GroupPlan, RsPlan, demand, prepare, prepare_many, settle,
    };
    use crate::store::rs::{RsGeometry, RsUnits};

    let facts = fixed_facts();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 3, 256).unwrap()));
    let stores = crate::store::registry::shared_stores(&shared, &Tiers::default(), 16, 256);
    let mut rs = stores.rs.lock().unwrap();
    // The geometry's page is not the rs list's: the list's slot holds 4.
    let ws = rs.create_working_set(RsGeometry {
        state_size: 1024,
        buffer_page_tokens: 16,
    });
    let first = prepare(&mut rs, ws, 5).unwrap();
    settle(&mut rs, first.txn);
    assert_eq!(rs_live(&shared, &[]), 0, "a folded state buffers nothing");

    // Three lanes of twelve rows each past the state's five.
    let plan = RsPlan::Groups(vec![GroupPlan {
        lanes: vec![0, 1, 2],
        writes: Some(5..41),
        fold: GroupFold::None,
        snapshots: Vec::new(),
    }]);
    assert_eq!(
        demand(&rs, &[ws; 3], &plan),
        Ok(RsUnits {
            slots: 0,
            pages: 10,
            ..RsUnits::default()
        }),
        "rows 4..44 in slots of 4"
    );
    let out = prepare_many(&mut rs, &[ws; 3], &plan, None).unwrap();
    let lanes = shared_lanes(&out, &shared, 3);
    let (pages, base, folded) = buffer_of(&lanes[0]);
    assert_eq!(
        (base, folded),
        (4, 5),
        "base aligned to T_rs below the state"
    );
    assert_eq!(pages.len(), 10);
    let distinct: BTreeSet<u32> = pages.iter().copied().collect();
    assert_eq!(distinct.len(), 10);
    for lane in &lanes {
        assert_eq!(buffer_of(lane).0, pages, "every lane reads one buffer");
        assert!(lane.slot_copies.is_empty(), "fresh slots copy nothing");
        lane.validate_chunks(&facts).unwrap();
    }
    assert!(out.state_copy().is_none());
    settle(&mut rs, out.txn);
    assert_eq!(rs_live(&shared, &pages), 10);

    // A fold of 30 of them: the slots wholly below row 35 go.
    let fold = RsPlan::Groups(vec![GroupPlan {
        lanes: vec![0],
        writes: Some(41..42),
        fold: GroupFold::Host(30),
        snapshots: Vec::new(),
    }]);
    let out = prepare_many(&mut rs, &[ws], &fold, None).unwrap();
    settle(&mut rs, out.txn);
    assert_eq!(rs.frozen(ws), Ok(35));
    assert_eq!(rs.base(ws), Ok(32));
    assert_eq!(rs_live(&shared, &pages[7..]), 3);

    rs.release_working_set(ws);
    rs.retire_idle();
    assert_eq!(rs_live(&shared, &[]), 0, "every slot back");
}

/// CONTRACT-BC 18: a fork shares its parent's rs list slots by refcount; a
/// lane writing into a shared, partly filled tail slot names a fresh one
/// the engine copies the shared one onto first (`Lane::slot_copies`), and
/// one it holds alone it writes in place. Every reference comes back.
#[test]
fn a_fork_shares_rs_slots_and_copies_a_shared_tail_before_writing() {
    use crate::pipeline::fire::rs::{
        GroupFold, GroupPlan, RsPlan, demand, prepare, prepare_many, settle,
    };
    use crate::store::rs::{RsGeometry, RsUnits};

    let facts = fixed_facts();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 2, 256).unwrap()));
    let stores = crate::store::registry::shared_stores(&shared, &Tiers::default(), 16, 256);
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(RsGeometry {
        state_size: 1024,
        buffer_page_tokens: 4,
    });
    let first = prepare(&mut rs, ws, 4).unwrap();
    settle(&mut rs, first.txn);
    let write = |writes: std::ops::Range<u32>| {
        RsPlan::Groups(vec![GroupPlan {
            lanes: vec![0],
            writes: Some(writes),
            fold: GroupFold::None,
            snapshots: Vec::new(),
        }])
    };
    // Rows 4..10: a full slot and a partly filled tail.
    let out = prepare_many(&mut rs, &[ws], &write(4..10), None).unwrap();
    let parent = buffer_of(&shared_lanes(&out, &shared, 1)[0]).0;
    settle(&mut rs, out.txn);
    let [head, tail] = parent[..] else {
        panic!("two slots");
    };
    assert_eq!(rs_live(&shared, &parent), 2);

    let fork = rs.fork(ws).unwrap();
    assert_eq!(rs_live(&shared, &[]), 2, "the fork shares them");
    assert_eq!(
        demand(&rs, &[fork], &write(10..11)),
        Ok(RsUnits {
            slots: 0,
            pages: 1,
            ..RsUnits::default()
        }),
        "a copy of the tail"
    );
    let out = prepare_many(&mut rs, &[fork], &write(10..11), None).unwrap();
    let lane = shared_lanes(&out, &shared, 1).remove(0);
    let (pages, base, _) = buffer_of(&lane);
    let [shared_head, copy] = pages[..] else {
        panic!("two slots");
    };
    assert_eq!(base, 4);
    assert_eq!(shared_head, head, "the full slot stays shared");
    assert_ne!(copy, tail);
    assert_eq!(lane.slot_copies, vec![(RS, vec![[tail, copy]])]);
    assert!(
        lane.fixed
            .iter()
            .flat_map(|(_, pieces)| pieces)
            .all(|[run, cow]| run == cow),
        "the record is not copied for a buffered write"
    );
    lane.validate_chunks(&facts).unwrap();
    settle(&mut rs, out.txn);
    assert_eq!(rs_live(&shared, &[head, tail, copy]), 3);
    assert_eq!(
        rs.buffer_translation(ws),
        Ok(parent.clone()),
        "the parent keeps its own"
    );

    // The parent holds its tail alone now: it writes it in place.
    assert_eq!(demand(&rs, &[ws], &write(10..11)), Ok(RsUnits::default()));
    let out = prepare_many(&mut rs, &[ws], &write(10..11), None).unwrap();
    let lane = shared_lanes(&out, &shared, 1).remove(0);
    assert_eq!(buffer_of(&lane).0, parent);
    assert!(lane.slot_copies.is_empty());
    settle(&mut rs, out.txn);

    // The fork grows a slot of its own past its copy.
    let out = prepare_many(&mut rs, &[fork], &write(11..13), None).unwrap();
    let lane = shared_lanes(&out, &shared, 1).remove(0);
    let grown = buffer_of(&lane).0;
    assert_eq!(grown[..2], [head, copy]);
    assert!(lane.slot_copies.is_empty());
    settle(&mut rs, out.txn);
    assert_eq!(rs_live(&shared, &grown), 4);

    // The parent folds past its head slot, which the fork still holds.
    rs.fold(ws, 4);
    rs.retire_idle();
    assert_eq!(rs.buffer_translation(ws), Ok(vec![tail]));
    assert_eq!(
        rs_live(&shared, &[]),
        4,
        "the head stays: the fork holds it"
    );
    rs.release_working_set(fork);
    rs.retire_idle();
    assert_eq!(rs_live(&shared, &[tail]), 1, "the fork's three go");
    rs.release_working_set(ws);
    rs.retire_idle();
    assert_eq!(rs_live(&shared, &[]), 0, "every reference back");
}

/// A second window list beside the facts' wide one: 24 rows, 8 a slot.
const NARROW: EngineShape = EngineShape(5);
/// The window lists' pool ids, in the facts' order.
const WIDE_POOL: ShapeId = 1;
const NARROW_POOL: ShapeId = 2;

/// A windowed model on a shared pool (CONTRACT-BC 19, 21): the kv list of
/// 16 tokens a slot and two window lists, each of its own window and slot:
/// 64 rows in 16-token slots, 24 rows in 8-token slots.
fn window_facts() -> ChunkFacts {
    let mut facts = facts();
    facts
        .shapes
        .retain(|shape| shape.id == LIST || shape.id == RING);
    facts.shapes[1].kind = SlotKind::List {
        tokens: 16,
        rate: 1,
        window: Some(64),
    };
    facts.shapes.push(shape(
        NARROW,
        SlotKind::List {
            tokens: 8,
            rate: 1,
            window: Some(24),
        },
        MIB / 4,
    ));
    facts.validate().unwrap();
    facts
}

/// A windowed working set's rows `rows` landed in one host-positioned fire
/// that commits `fold` of them (`None`: every row), its tables read through
/// them.
fn window_fire(
    rs: &mut crate::store::rs::RsStore,
    ws: crate::store::rs::RsWorkingSetId,
    rows: std::ops::Range<u32>,
    fold: Option<u32>,
) -> crate::pipeline::fire::rs::PreparedRs {
    use crate::pipeline::fire::rs::{RsPlan, WindowRow, prepare_many};
    let plan = RsPlan::Window(vec![WindowRow {
        writes: rows.clone(),
        reads: rows.end,
        advance: crate::store::rs::WindowAdvance {
            positioned: true,
            settled: rows.start,
            reach: rows.start,
            fold: Some(fold.unwrap_or(rows.end - rows.start)),
            commit_end: rows.end,
            head: Some(rows.end),
        },
    }]);
    prepare_many(rs, &[ws], &plan, None).unwrap()
}

/// Each window list's slots the pool holds now: `(narrow, wide)`.
fn window_live(shared: &Shared) -> (u64, u64) {
    let pool = shared.lock().unwrap();
    (
        pool.pool().shape_demand(NARROW_POOL).live,
        pool.pool().shape_demand(WIDE_POOL).live,
    )
}

fn windowed_stores() -> (ChunkFacts, Shared, crate::store::registry::Stores) {
    let facts = window_facts();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 2, 256).unwrap()));
    let stores = crate::store::registry::shared_stores(&shared, &Tiers::default(), 16, 256);
    (facts, shared, stores)
}

/// The window lists' tables a lane of `out` names: `(narrow, wide)`.
fn window_tables(
    out: &crate::pipeline::fire::rs::PreparedRs,
    shared: &Shared,
) -> (Vec<u32>, Vec<u32>) {
    let lane = shared_lanes(out, shared, 1).remove(0);
    let table = |id: EngineShape| {
        lane.tables
            .iter()
            .find(|(shape, _)| *shape == id)
            .map(|(_, refs)| refs.clone())
            .unwrap_or_default()
    };
    (table(NARROW), table(RING))
}

/// The leading null refs of `table`: the slots trimmed behind its window.
fn nulls(table: &[u32]) -> usize {
    table.iter().take_while(|&&r| r == 0).count()
}

fn window_geometry() -> crate::store::rs::RsGeometry {
    crate::store::rs::RsGeometry {
        state_size: 0,
        buffer_page_tokens: 16,
    }
}

/// (a) Each window list keeps the slots its own window reaches and drops
/// one once its last row lies below `committed_end - window`: its table
/// keeps its own token-order length (rows `[i * T_w, (i + 1) * T_w)` at
/// entry `i`), a dropped slot ref 0.
#[test]
fn a_window_list_trims_a_slot_wholly_behind_its_window() {
    use crate::pipeline::fire::rs::settle;
    let (facts, shared, stores) = windowed_stores();
    {
        let pool = shared.lock().unwrap();
        let windows: Vec<(EngineShape, u32, u32)> = pool
            .windows()
            .iter()
            .map(|w| (w.id, w.window, w.tokens))
            .collect();
        assert_eq!(
            windows,
            vec![(NARROW, 24, 8), (RING, 64, 16)],
            "narrowest first"
        );
    }
    let mut rs = stores.rs.lock().unwrap();
    assert!(rs.is_window() && rs.is_shared());
    let ws = rs.create_working_set(window_geometry());
    let out = window_fire(&mut rs, ws, 0..64, None);
    let (narrow, wide) = window_tables(&out, &shared);
    assert_eq!((narrow.len(), wide.len()), (8, 4), "each its own slots");
    assert!(!narrow.contains(&0) && !wide.contains(&0));
    settle(&mut rs, out.txn);
    rs.retire_idle();
    // 64 - 24 = 40: the narrow list's slots 0..5 lie wholly behind it;
    // nothing lies behind 64 - 64.
    assert_eq!(window_live(&shared), (3, 4));

    // Row 79 committed: the wide list's slot 0 (last row 15) is not below
    // 79 - 64; the narrow list's slot 5 (last row 47) is below 79 - 24.
    let out = window_fire(&mut rs, ws, 64..79, None);
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert_eq!(window_live(&shared), (4, 5));
    // Row 80: the wide list's slot 0 goes too, once the fire that read it
    // retired.
    let out = window_fire(&mut rs, ws, 79..80, None);
    let (narrow, wide) = window_tables(&out, &shared);
    assert_ne!(wide[0], 0, "the fire reads it still");
    assert_ne!(narrow[6], 0);
    assert_eq!(window_live(&shared), (4, 5), "held until that fire retires");
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert_eq!(window_live(&shared), (3, 4));
    let out = window_fire(&mut rs, ws, 80..81, None);
    let (narrow, wide) = window_tables(&out, &shared);
    assert_eq!((narrow.len(), wide.len()), (11, 6));
    assert_eq!(nulls(&wide), 1, "the trimmed slot reads the null chunk");
    assert_eq!(nulls(&narrow), 7);
    assert!(wide[1..].iter().chain(&narrow[7..]).all(|&r| r != 0));
    shared_lanes(&out, &shared, 1)
        .remove(0)
        .validate_chunks(&facts)
        .unwrap();
    settle(&mut rs, out.txn);
    rs.release_working_set(ws);
    rs.retire_idle();
    assert_eq!(window_live(&shared), (0, 0));
}

/// (b) A prefill longer than a window holds a slot for every one it writes
/// while it runs, and the trim after it drops all but those each window
/// reaches.
#[test]
fn a_prefill_past_the_window_holds_its_slots_until_the_trim() {
    use crate::pipeline::fire::rs::settle;
    let (_, shared, stores) = windowed_stores();
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(window_geometry());
    let out = window_fire(&mut rs, ws, 0..160, None);
    let (narrow, wide) = window_tables(&out, &shared);
    assert_eq!((narrow.len(), wide.len()), (20, 10));
    assert!(
        !narrow.contains(&0) && !wide.contains(&0),
        "the prefill reads every slot it writes"
    );
    assert_eq!(window_live(&shared), (20, 10));
    settle(&mut rs, out.txn);
    rs.retire_idle();
    // 160 - 64 = 96: wide slots 0..6; 160 - 24 = 136: narrow slots 0..17.
    assert_eq!(window_live(&shared), (3, 4));
    let out = window_fire(&mut rs, ws, 160..161, None);
    let (narrow, wide) = window_tables(&out, &shared);
    assert_eq!((nulls(&narrow), nulls(&wide)), (17, 6));
    settle(&mut rs, out.txn);
    rs.release_working_set(ws);
    rs.retire_idle();
    assert_eq!(window_live(&shared), (0, 0));
}

/// (c) A fork and a prefix-cache checkpoint share a working set's window
/// slots by refcount; a write into a shared, partly filled tail slot of
/// each list copies it first (that list's `Lane::slot_copies`), and every
/// slot comes back once each holder lets go.
#[test]
fn a_fork_and_a_checkpoint_share_window_slots_and_all_come_back() {
    use crate::pipeline::fire::rs::settle;
    let (facts, shared, stores) = windowed_stores();
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(window_geometry());
    let out = window_fire(&mut rs, ws, 0..60, None);
    let (narrow, wide) = window_tables(&out, &shared);
    settle(&mut rs, out.txn);
    rs.retire_idle();
    // Wide: 0..4, its tail 3 (rows 48..60); narrow: 4..8, its tail 7.
    assert_eq!(window_live(&shared), (4, 4));

    let fork = rs.fork(ws).unwrap();
    assert!(
        rs.checkpoint(ws, [7; 32], None).unwrap(),
        "windows checkpoint"
    );
    assert_eq!(window_live(&shared), (4, 4), "all three share them");
    let out = window_fire(&mut rs, fork, 60..62, None);
    let lane = shared_lanes(&out, &shared, 1).remove(0);
    lane.validate_chunks(&facts).unwrap();
    let (forked_narrow, forked_wide) = window_tables(&out, &shared);
    assert_eq!(forked_wide[..3], wide[..3], "full slots stay shared");
    assert_ne!(forked_wide[3], wide[3], "the tail is the fork's copy");
    assert_eq!(forked_narrow[4..7], narrow[4..7]);
    assert_ne!(forked_narrow[7], narrow[7]);
    let copies = |id: EngineShape| {
        lane.slot_copies
            .iter()
            .find(|(shape, _)| *shape == id)
            .map(|(_, pairs)| pairs.clone())
    };
    assert_eq!(copies(RING), Some(vec![[wide[3], forked_wide[3]]]));
    assert_eq!(copies(NARROW), Some(vec![[narrow[7], forked_narrow[7]]]));
    settle(&mut rs, out.txn);
    assert_eq!(window_live(&shared), (5, 5));

    // A fresh set takes the checkpoint's slots, shared.
    let resumed = rs.create_working_set(window_geometry());
    assert!(rs.adopt_checkpoint(resumed, &[7; 32]).unwrap());
    let out = window_fire(&mut rs, resumed, 60..61, None);
    let (adopted_narrow, adopted_wide) = window_tables(&out, &shared);
    assert_eq!(adopted_wide[..3], wide[..3]);
    assert_eq!(adopted_narrow[4..7], narrow[4..7]);
    settle(&mut rs, out.txn);

    for held in [fork, ws, resumed] {
        rs.release_working_set(held);
    }
    rs.retire_idle();
    assert_eq!(window_live(&shared), (4, 4), "the checkpoint holds its own");
    assert_eq!(rs.reclaim_cache(|_| 0), 8, "its eight slots go");
    rs.retire_idle();
    assert_eq!(window_live(&shared), (0, 0), "every reference back");
}

/// (d) Rows of a fire still being decided (spec drafts) never count as
/// committed: each window trims behind what the fires commit only.
#[test]
fn spec_draft_rows_are_not_committed_rows() {
    use crate::pipeline::fire::rs::settle;
    let (_, shared, stores) = windowed_stores();
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(window_geometry());
    let out = window_fire(&mut rs, ws, 0..64, None);
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert_eq!(window_live(&shared), (3, 4));
    // Drafts to 96, none accepted: committed stays 64.
    let out = window_fire(&mut rs, ws, 64..96, Some(0));
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert_eq!(rs.window_positions(ws), Ok(Some((64, 96))));
    assert_eq!(window_live(&shared), (7, 6), "no draft row trims a slot");
    // Five accepted: committed 69; the narrow list's slot 5 (rows 40..48)
    // still reaches 69 - 24, and nothing lies behind 69 - 64.
    let out = window_fire(&mut rs, ws, 64..96, Some(5));
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert_eq!(rs.window_positions(ws), Ok(Some((69, 96))));
    assert_eq!(window_live(&shared), (7, 6));
    // Through 96 accepted.
    let out = window_fire(&mut rs, ws, 69..96, None);
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert_eq!(rs.window_positions(ws), Ok(Some((96, 96))));
    assert_eq!(window_live(&shared), (3, 4));
    rs.release_working_set(ws);
    rs.retire_idle();
    assert_eq!(window_live(&shared), (0, 0));
}

/// (e) Two window lists of different windows and slots trim on their own:
/// decoding from 64, the narrow list lets a slot go at 72 while the wide
/// one keeps all of its, and the wide one lets its first go only at 80.
#[test]
fn two_window_lists_trim_each_by_its_own_window() {
    use crate::pipeline::fire::rs::settle;
    let (_, shared, stores) = windowed_stores();
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(window_geometry());
    let out = window_fire(&mut rs, ws, 0..64, None);
    settle(&mut rs, out.txn);
    rs.retire_idle();
    let mut seen = Vec::new();
    for row in 64..81 {
        let out = window_fire(&mut rs, ws, row..row + 1, None);
        settle(&mut rs, out.txn);
        rs.retire_idle();
        seen.push((row + 1, window_live(&shared)));
    }
    // Narrow (24 rows, 8 a slot): slot i goes once 8(i + 1) <= end - 24.
    // Wide (64 rows, 16 a slot): slot i goes once 16(i + 1) <= end - 64.
    for (end, (narrow, wide)) in seen {
        let narrow_held = end.div_ceil(8) - (end - 24) / 8;
        let wide_held = end.div_ceil(16) - (end - 64) / 16;
        assert_eq!(
            (narrow, wide),
            (u64::from(narrow_held), u64::from(wide_held)),
            "at {end}"
        );
    }
    rs.release_working_set(ws);
    rs.retire_idle();
    assert_eq!(window_live(&shared), (0, 0));
}

mod tiers;

/// What is left for kv and state is the pool less the null chunk, the
/// smallest rung's activation and the chunks the weights pin: held, every
/// atom and the staging floor; streaming, the floor alone.
#[test]
fn the_room_left_for_kv_takes_out_what_the_weights_pin() {
    let facts = facts();
    for (streaming, weight_chunks) in [(false, 5), (true, 1)] {
        let mut shared = pool();
        let first = shared.weights_first();
        crate::store::weights::WeightCache::new(
            &facts.weights,
            first,
            streaming,
            shared.pool_mut(),
        )
        .unwrap();
        let room = shared.room();
        assert_eq!(room.weight_chunks, weight_chunks, "streaming {streaming}");
        assert_eq!((room.reserved_chunks, room.act_chunks), (1, 1));
        assert_eq!(room.state_chunks, TOTAL - 2 - weight_chunks);
        assert_eq!(room.list_slots, room.state_chunks * 16);
        assert_eq!(room.list_tokens, 16);
    }
}
