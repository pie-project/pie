//! The shared pool against a fake engine that states real, unpinned facts:
//! a list, a window list, a fixed record of three pieces and one of one, a weight
//! shape, two-ended growth.

use ::engine::chunk::{
    ChunkTable, Growth as SegmentGrowth, Reserved, ReservedRun, SegmentFacts, SlotShape as Facts,
    WeightKind, WeightShape,
};
use ::engine::fire::Lane;

use super::*;
use crate::governor::Retarget;

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

pub(super) fn pool() -> SharedPool {
    SharedPool::new(&facts(), 256).unwrap()
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

/// The engine's side of a pool move, as the scheduler's [`PoolOp`]s reach
/// it from [`SharedGrow`].
///
/// [`PoolOp`]: crate::scheduler::worker::PoolOp
impl FakeEngine {
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

/// The segment walk of [`SharedGrow`] (target.rs) against `engine`: steps
/// toward `target` mapped segments until there, or until a move has to wait;
/// the fire at `epoch` is the last that may read what an evacuation moves.
fn walk_segments(
    pool: &mut SharedPool,
    target: u32,
    epoch: u64,
    engine: &mut FakeEngine,
) -> Retarget {
    loop {
        match pool.next_move(target, epoch) {
            None | Some(SegmentMove::Wait) => return Retarget::Reached,
            Some(SegmentMove::NoRoom) => {
                return Retarget::NoRoom {
                    held: u64::from(pool.mapped_segments()),
                };
            }
            Some(SegmentMove::Commit { prefix }) => {
                if engine.commit_segments(prefix).is_err() {
                    return Retarget::Reached;
                }
                pool.committed(prefix);
            }
            Some(SegmentMove::Evacuate { evacuation }) => {
                let _ = engine.evacuate(&evacuation);
                return Retarget::Reached;
            }
            Some(SegmentMove::Release { prefix }) => {
                engine.release_segments(prefix).unwrap();
            }
        }
    }
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
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
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
    window_fire_keeping(rs, ws, rows, fold, None)
}

/// [`window_fire`] landing a checkpoint at `keep` inside it.
fn window_fire_keeping(
    rs: &mut crate::store::rs::RsStore,
    ws: crate::store::rs::RsWorkingSetId,
    rows: std::ops::Range<u32>,
    fold: Option<u32>,
    keep: Option<u32>,
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
            keep,
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
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
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

/// (c3) A window pass ending inside a page keeps its checkpoint at the page
/// boundary before its end: the fire trims nothing that boundary's windows
/// reach, so the checkpoint holds the slots `[at - window, at)` touches. Had
/// the fire trimmed behind its end, one would be gone and the checkpoint is
/// refused. A checkpoint inside a slot shares that slot, rows past it too; a
/// resume writing into it copies it first (each list's `slot_copies`), so
/// the checkpoint's rows stay as they were.
#[test]
fn a_window_checkpoint_behind_the_fire_end_holds_what_its_windows_reach() {
    use crate::pipeline::fire::rs::settle;
    let (facts, shared, stores) = windowed_stores();
    let mut rs = stores.rs.lock().unwrap();

    // Trimmed behind its end 72, the narrow list (24 / 8) gives back slot 5
    // (rows 40..48), which 64's window (from 40) reaches: refused.
    let trimmed = rs.create_working_set(window_geometry());
    let out = window_fire(&mut rs, trimmed, 0..72, None);
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert!(!rs.checkpoint_at(trimmed, [1; 32], None, 64).unwrap());
    assert!(!rs.has_checkpoint(&[1; 32]));
    rs.release_working_set(trimmed);
    rs.retire_idle();
    assert_eq!(window_live(&shared), (0, 0));

    // Kept at 64, the fire leaves that slot; the checkpoint at 64 takes
    // narrow 5..8 and wide 0..4, as a set that stopped at 64 would hold.
    let ws = rs.create_working_set(window_geometry());
    let out = window_fire_keeping(&mut rs, ws, 0..72, None, Some(64));
    let (narrow, wide) = window_tables(&out, &shared);
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert_eq!(window_live(&shared), (4, 5), "narrow 5..9, wide 0..5");
    assert!(rs.checkpoint_at(ws, [2; 32], None, 64).unwrap());
    assert!(!rs.checkpoint_at(ws, [3; 32], None, 80).unwrap(), "ahead");
    assert!(
        !rs.checkpoint_at(ws, [3; 32], None, 60).unwrap(),
        "60 reaches narrow 4, which went with the trim at 64"
    );
    // The set goes on; its next fire trims what it no longer reaches.
    let out = window_fire(&mut rs, ws, 72..73, None);
    settle(&mut rs, out.txn);
    rs.release_working_set(ws);
    rs.retire_idle();

    // A resume at 64 stands there and reads the slots it took; 64 is a slot
    // boundary of each list, so it shares no tail and copies nothing.
    let at64 = rs.create_working_set(window_geometry());
    assert!(rs.adopt_checkpoint(at64, &[2; 32]).unwrap());
    assert_eq!(rs.window_positions(at64), Ok(Some((64, 64))));
    let out = window_fire(&mut rs, at64, 64..65, None);
    let lane = shared_lanes(&out, &shared, 1).remove(0);
    assert!(lane.slot_copies.iter().all(|(_, pairs)| pairs.is_empty()));
    let (n64, w64) = window_tables(&out, &shared);
    assert_eq!(n64[5..8], narrow[5..8]);
    assert_eq!(w64[..4], wide[..4]);
    assert_eq!(nulls(&n64), 5, "behind the window");
    settle(&mut rs, out.txn);

    // Inside a slot: 60 lies in narrow 7 (56..64) and wide 3 (48..64), which
    // hold rows past 60 too. A resume at 60 writes into them: copied first.
    let ws = rs.create_working_set(window_geometry());
    let out = window_fire_keeping(&mut rs, ws, 0..72, None, Some(60));
    let (narrow, wide) = window_tables(&out, &shared);
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert!(rs.checkpoint_at(ws, [4; 32], None, 60).unwrap());
    rs.release_working_set(ws);
    rs.retire_idle();
    let at60 = rs.create_working_set(window_geometry());
    assert!(rs.adopt_checkpoint(at60, &[4; 32]).unwrap());
    assert_eq!(rs.window_positions(at60), Ok(Some((60, 60))));
    let out = window_fire(&mut rs, at60, 60..62, None);
    let lane = shared_lanes(&out, &shared, 1).remove(0);
    lane.validate_chunks(&facts).unwrap();
    let (n60, w60) = window_tables(&out, &shared);
    assert_eq!(n60[4..7], narrow[4..7], "full slots stay shared");
    assert_eq!(w60[..3], wide[..3]);
    assert_ne!(n60[7], narrow[7], "the tail is the resume's copy");
    assert_ne!(w60[3], wide[3]);
    let copies = |id: EngineShape| {
        lane.slot_copies
            .iter()
            .find(|(shape, _)| *shape == id)
            .map(|(_, pairs)| pairs.clone())
    };
    assert_eq!(copies(NARROW), Some(vec![[narrow[7], n60[7]]]));
    assert_eq!(copies(RING), Some(vec![[wide[3], w60[3]]]));
    settle(&mut rs, out.txn);

    for held in [at64, at60] {
        rs.release_working_set(held);
    }
    rs.retire_idle();
    assert_eq!(rs.reclaim_cache(|_| 0), 7 + 8, "both checkpoints' slots go");
    rs.retire_idle();
    assert_eq!(window_live(&shared), (0, 0), "every reference back");
}

mod programs;
mod tiers;

/// SPEC decision 6, the governor's rung shrink: under critical pressure the
/// walk caps fires to the smaller rung first, lowers the pool's rung only
/// once the scheduler has no wider step in flight and every fire handed out
/// by then has retired, and state may then take the span above it. Settled,
/// it moves that state back out (the engine copying it before the next
/// fire), takes the largest rung and lifts the cap.
#[test]
fn the_rung_walk_caps_fires_then_lowers_and_raises_back_with_an_evacuation() {
    let facts = facts();
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    let mut walk = RungWalk::default();
    assert_eq!(pool.rungs(), [16, 256, 2048]);
    assert_eq!(pool.rung(), 2048);
    // What may be out counts the room under the rung the pool is at: a
    // smaller rung's only once it is lowered.
    let at_largest = (pool.state_chunks_now(), pool.records_capacity());
    assert_eq!(
        at_largest.0,
        TOTAL - pool.pool().reserved_chunks() - pool.pool().act_reserved()
    );
    assert!(at_largest.1 < pool.records_most());
    // Down to 256: the cap comes first.
    assert_eq!(
        walk.step(&pool, 256, u32::MAX, u32::MAX, 5, 5).unwrap(),
        RungStep::Cap(256)
    );
    // A wider step may still be in flight: wait for the scheduler's grant.
    assert_eq!(
        walk.step(&pool, 256, 256, u32::MAX, 5, 5).unwrap(),
        RungStep::Wait
    );
    // Granted at fire 7: wait until fire 7 retires.
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 7, 5).unwrap(),
        RungStep::Wait
    );
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 8, 6).unwrap(),
        RungStep::Wait
    );
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 9, 7).unwrap(),
        RungStep::Lower(256)
    );
    pool.lower_rung(256).unwrap();
    assert_eq!(pool.rung(), 256);
    assert!(pool.state_chunks_now() > at_largest.0, "lowered: more room");
    assert!(pool.records_capacity() > at_largest.1);
    assert_eq!(
        pool.pool().state_limit(),
        TOTAL - 4,
        "state may take the span above"
    );
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 9, 9).unwrap(),
        RungStep::Wait
    );
    // State takes the span the larger rung's fires would touch.
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    let kept: Vec<u32> = all[all.len() - 6..].to_vec();
    pool.give_records(&all[..all.len() - 6]);
    let in_way = |pool: &SharedPool| {
        kept.iter()
            .flat_map(|&r| pool.record_slots(r))
            .filter(|slot| pool.pool().physical(slot.chunk).unwrap() >= TOTAL - 12)
            .count()
    };
    assert!(in_way(&pool) > 0);
    // Settled: raise, the evacuation first, then lift the cap.
    let RungStep::Raise { rung, evacuation } = walk.step(&pool, u32::MAX, 256, 256, 9, 9).unwrap()
    else {
        panic!("a raise");
    };
    assert_eq!(rung, 2048);
    assert!(!evacuation.moves.is_empty(), "state in the span moves out");
    engine
        .evacuate(&SharedPool::rung_moves(&evacuation))
        .unwrap();
    pool.apply_rung(&evacuation, 10).unwrap();
    assert_eq!(pool.rung(), 2048);
    assert_eq!(
        (pool.state_chunks_now(), pool.records_capacity()),
        at_largest,
        "raised: the room it had"
    );
    assert_eq!(in_way(&pool), 0);
    assert_eq!(engine.table.as_slice(), pool.pool().chunk_table());
    for &record in &kept {
        let lane = Lane {
            tokens: vec![1],
            fixed: pool.lane_fixed(record, record),
            ..Lane::default()
        };
        engine.fire(&facts, &lane, 12);
    }
    assert_eq!(
        walk.step(&pool, u32::MAX, 256, 256, 10, 9).unwrap(),
        RungStep::Cap(u32::MAX),
        "the largest rung: no cap"
    );
    assert_eq!(
        walk.step(&pool, u32::MAX, u32::MAX, u32::MAX, 10, 10)
            .unwrap(),
        RungStep::Wait
    );
    // A goal between rungs takes the rung under it; below the smallest, the
    // smallest.
    assert_eq!(
        walk.step(&pool, 300, u32::MAX, u32::MAX, 10, 10).unwrap(),
        RungStep::Cap(256)
    );
    assert_eq!(
        walk.step(&pool, 1, u32::MAX, u32::MAX, 10, 10).unwrap(),
        RungStep::Cap(16)
    );
}

/// Rungs by demand: the rung a pass's rows ask for, and what a raise lacks
/// while state fills the span the larger rung's fires take; once some of it
/// goes, the raise moves what is left in the span down.
#[test]
fn a_raise_by_demand_says_what_it_lacks_and_moves_the_rest() {
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.rung_for(0), 16, "no ask: the smallest");
    assert_eq!(pool.rung_for(16), 16);
    assert_eq!(pool.rung_for(17), 256);
    assert_eq!(pool.rung_for(5000), 2048, "past every rung: the largest");
    assert_eq!(
        pool.armed_rung_for(5000, 256),
        256,
        "a rung still being captured is not offered"
    );
    assert_eq!(pool.armed_rung_for(17, 256), 256);
    assert_eq!(
        pool.armed_rung_for(17, 0),
        16,
        "the smallest is always served"
    );
    assert_eq!(pool.armed_rung_for(5000, u32::MAX), 2048);
    pool.lower_rung(16).unwrap();
    assert_eq!(
        pool.raise_short(2048),
        (0, 0),
        "an empty pool raises at once"
    );
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    let (chunks, pages) = pool.raise_short(2048);
    assert!(chunks > 0, "state fills the span: the raise lacks room");
    assert!(
        pages >= chunks * 16,
        "16 kv pages a chunk, and the partly filled chunks' free slots on top"
    );
    assert!(pool.plan_raise(2048).is_err());
    pool.give_records(&all[..all.len() / 2]);
    pool.pool_mut().retire_through(u64::MAX);
    assert_eq!(pool.raise_short(2048), (0, 0));
    let evacuation = pool.plan_raise(2048).unwrap();
    assert!(!evacuation.moves.is_empty(), "the span's state moves down");
    engine
        .evacuate(&SharedPool::rung_moves(&evacuation))
        .unwrap();
    pool.apply_rung(&evacuation, 1).unwrap();
    assert_eq!(pool.rung(), 2048);
    assert_eq!(engine.table.as_slice(), pool.pool().chunk_table());
}

/// Twelve rungs, every power of two from 1, the small ones sharing their
/// activation's chunk count: an idle pool lowered to the smallest stays
/// there, the walk never lowering again within the class it is in.
#[test]
fn rungs_sharing_their_chunks_are_one_class_the_walk_never_moves_within() {
    let facts = ChunkFacts {
        act_chunks: vec![
            (1, 1),
            (2, 1),
            (4, 1),
            (8, 1),
            (16, 1),
            (32, 2),
            (64, 2),
            (128, 2),
            (256, 4),
            (512, 4),
            (1024, 8),
            (2048, 12),
        ],
        ..facts()
    };
    facts.validate().unwrap();
    let mut pool = SharedPool::new(&facts, 256).unwrap();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    let mut walk = RungWalk::default();
    assert_eq!(pool.rung(), 2048);
    assert_eq!(
        walk.step(&pool, 1, u32::MAX, u32::MAX, 5, 5).unwrap(),
        RungStep::Cap(1)
    );
    assert_eq!(walk.step(&pool, 1, 1, 1, 7, 5).unwrap(), RungStep::Wait);
    assert_eq!(walk.step(&pool, 1, 1, 1, 9, 7).unwrap(), RungStep::Lower(1));
    pool.lower_rung(1).unwrap();
    assert_eq!(pool.rung(), 16, "the pool reports its class's widest rung");
    // Idle at the smallest rung: no repeated lower, ever.
    for epoch in 10..40 {
        assert_eq!(
            walk.step(&pool, 1, 1, 1, epoch, epoch).unwrap(),
            RungStep::Wait,
            "a lower within the class the pool is in"
        );
    }
    // A goal inside the class only caps fires; none raises the rung.
    assert_eq!(
        walk.step(&pool, 16, 1, 1, 40, 40).unwrap(),
        RungStep::Cap(16)
    );
    assert_eq!(
        walk.step(&pool, 8, 16, 16, 41, 41).unwrap(),
        RungStep::Cap(8)
    );
    // The next class raises.
    assert!(matches!(
        walk.step(&pool, 32, 8, 8, 42, 42).unwrap(),
        RungStep::Raise { rung: 32, .. }
    ));
}

/// What an arming step is lent comes from the allocators sequences take
/// from, named as a lane names them: a record by its pieces' slot refs, a
/// kv page by its list slot ref; none lies in the null chunk, which the
/// engine refuses a ref into.
#[test]
fn a_lent_record_and_lent_pages_name_no_slot_in_the_null_chunk() {
    let mut facts = facts();
    facts.shapes.retain(|shape| shape.id != RING);
    facts.validate().unwrap();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let mut rs = crate::store::rs::RsStore::shared(&shared, None);
    let grant = rs
        .reserve(crate::store::rs::RsUnits::of(1, 0))
        .expect("a record");
    assert_eq!(grant.slots, [crate::store::rs::RsSlotId(0)], "store id 0");
    let refs = rs.device_refs(&grant.slots);
    let pool = shared.lock().unwrap();
    let reserved = pool.pool().reserved_chunks();
    assert!(reserved >= 1, "chunk 0 is the null chunk");
    let mut at = 0;
    for fixed in pool.fixed() {
        let spc = pool.pool().shape(fixed.pool).unwrap().slots_per_chunk;
        for _ in 0..fixed.pieces {
            let chunk = refs[at] / spc;
            assert!(
                chunk >= reserved,
                "piece ref {} lies in chunk {chunk}",
                refs[at]
            );
            at += 1;
        }
    }
    assert_eq!(at, refs.len(), "one record, every piece");
    let list = pool.list().expect("a kv list");
    drop(pool);
    let mut pages =
        crate::store::chunks::ShapePool::<crate::store::kv::page_table::PhysicalKvPageId>::shared(
            &shared, list,
        );
    let spc = shared
        .lock()
        .unwrap()
        .pool()
        .shape(list.pool)
        .unwrap()
        .slots_per_chunk;
    for page in pages.try_alloc_n(64).expect("pages") {
        assert!(
            page.0 / spc >= reserved,
            "page {} lies in the null chunk",
            page.0
        );
    }
    rs.release_reservation(grant);
}

/// While the engine holds back part of the range it stated, no slot it
/// binds none of is handed out, to a sequence or an arming lease alike:
/// every ref lies in a committed segment, past the null chunk and below
/// the span held back; once it holds none, the rest opens.
#[test]
fn no_slot_is_handed_out_past_what_the_engine_binds() {
    let facts = facts();
    let shared: Shared = Arc::new(Mutex::new(pool()));
    let list = shared.lock().unwrap().list().expect("a kv list");
    let spc = facts.shapes[0].slots_per_chunk;
    let chunk_of = |pool: &SharedPool, r: u32| pool.pool().physical(r / spc).unwrap();
    // Two segments committed: refs stay in them.
    {
        let mut pool = shared.lock().unwrap();
        let refs = pool.take_refs(list.pool, 64).expect("refs");
        for &r in &refs {
            let chunk = chunk_of(&pool, r);
            assert!(
                chunk >= 1 && chunk < 2 * SEGMENT,
                "ref {r} in chunk {chunk}"
            );
        }
        pool.give_refs(list.pool, &refs);
    }
    // Every segment committed, the engine holding back all but 8 chunks
    // below the activation: refs stay under chunk 8.
    let mut engine = FakeEngine::new();
    {
        let mut pool = shared.lock().unwrap();
        assert_eq!(
            walk_segments(&mut pool, 8, 0, &mut engine),
            Retarget::Reached
        );
        let top = TOTAL - pool.pool().act_max();
        pool.hold_back(u64::from(top - 8) * CHUNK);
        assert_eq!(pool.pool().ceiling(), 8);
    }
    let mut pages =
        crate::store::chunks::ShapePool::<crate::store::kv::page_table::PhysicalKvPageId>::shared(
            &shared, list,
        );
    let mut taken = Vec::new();
    while let Some(ids) = pages.try_alloc_n(1) {
        taken.extend(ids);
    }
    {
        let pool = shared.lock().unwrap();
        assert!(!taken.is_empty());
        for page in &taken {
            let chunk = chunk_of(&pool, page.0);
            assert!(
                (1..8).contains(&chunk),
                "page {} lies in chunk {chunk}, past what the engine binds",
                page.0
            );
        }
    }
    // It holds none back: the rest of the span opens.
    shared.lock().unwrap().hold_back(0);
    let more = pages.try_alloc_n(1).expect("the span held back opens");
    let pool = shared.lock().unwrap();
    assert!(chunk_of(&pool, more[0].0) >= 8);
}

/// Cold start: arming steps lease pages while decoders take theirs, the
/// engine holding part of the range back; arming settles (the hold drops,
/// the ceiling lifts); then the pool runs dry and a holder fails with a fire
/// in flight. Through all of it no page is ever had by two owners (a lane,
/// a lease, a fire's pending recycle), and nothing handed out is handed out
/// again before it comes back.
#[test]
fn leases_grants_a_settle_and_a_failed_holder_never_give_a_page_two_owners() {
    use crate::store::kv::page_table::PhysicalKvPageId;
    use std::collections::{HashMap, HashSet};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Owner {
        Lane(u32),
        Lease,
        /// Let go under a fire in flight until that epoch retires.
        Pending(u64),
    }

    let shared: Shared = Arc::new(Mutex::new(pool()));
    let list = shared.lock().unwrap().list().expect("a kv list");
    let mut engine = FakeEngine::new();
    {
        let mut pool = shared.lock().unwrap();
        assert_eq!(
            walk_segments(&mut pool, 8, 0, &mut engine),
            Retarget::Reached
        );
        // Arming: the engine holds back all but 16 chunks below activation.
        let top = TOTAL - pool.pool().act_max();
        pool.hold_back(u64::from(top - 16) * CHUNK);
    }
    let mut pages = crate::store::chunks::ShapePool::<PhysicalKvPageId>::shared(&shared, list);
    let mut owners: HashMap<u32, Owner> = HashMap::new();
    let take = |owners: &mut HashMap<u32, Owner>,
                pages: &mut crate::store::chunks::ShapePool<PhysicalKvPageId>,
                n: usize,
                owner: Owner|
     -> Option<Vec<PhysicalKvPageId>> {
        let got = pages.try_alloc_n(n)?;
        let mut seen = HashSet::new();
        for page in &got {
            assert!(
                seen.insert(page.0),
                "page {} handed out twice at once",
                page.0
            );
            if let Some(had) = owners.insert(page.0, owner) {
                panic!("page {} given to {owner:?} while {had:?} has it", page.0);
            }
        }
        Some(got)
    };
    let mut rng = 0x9e37_79b9_u64;
    let mut next = |below: u64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng % below.max(1)
    };
    let mut epoch = 1u64;
    let mut retired = 0u64;
    let mut lanes: HashMap<u32, Vec<PhysicalKvPageId>> = HashMap::new();
    for step in 0..4000 {
        if step == 1500 {
            // Every rung armed: the hold drops and the ceiling lifts.
            shared.lock().unwrap().hold_back(0);
        }
        match next(10) {
            // A decoder lane grows.
            0..=3 => {
                let lane = next(32) as u32;
                if let Some(got) = take(
                    &mut owners,
                    &mut pages,
                    1 + next(3) as usize,
                    Owner::Lane(lane),
                ) {
                    lanes.entry(lane).or_default().extend(got);
                }
            }
            // An arming step (before the settle) is lent room and gives it
            // back once it returns.
            4..=5 if step < 1500 => {
                if let Some(lent) = take(&mut owners, &mut pages, 4, Owner::Lease) {
                    for page in &lent {
                        assert_eq!(owners.remove(&page.0), Some(Owner::Lease));
                    }
                    pages.release_reserved(lent);
                }
            }
            // A lane fails (the pool's sole holder, a fire in flight): its
            // pages come back once that fire retires.
            6 => {
                let lane = next(32) as u32;
                if let Some(held) = lanes.remove(&lane) {
                    for page in &held {
                        owners.insert(page.0, Owner::Pending(epoch));
                    }
                    pages.recycle_after_epoch(held, epoch);
                    epoch += 1;
                }
            }
            // Fires retire.
            _ => {
                retired = (retired + 1).min(epoch - 1);
                pages.retire_through(retired);
                owners.retain(|_, owner| !matches!(owner, Owner::Pending(at) if *at <= retired));
            }
        }
    }
    let held: usize = lanes.values().map(Vec::len).sum();
    assert!(held > 0, "the lanes hold pages at the end");
}

/// A raise that moves state is taken in the hold it was planned in: no
/// slot can be handed out between plan and apply (the engine copies after,
/// never against a pool that moved on), its moved-off chunks stay fenced and
/// no cap is lifted past the rung held before until the engine has the copy.
#[test]
fn a_raise_is_taken_where_it_is_planned_and_held_until_its_copy() {
    const ENGINE: usize = 60;
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(16).unwrap();
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    pool.give_records(&all[..all.len() / 2]);
    pool.pool_mut().retire_through(u64::MAX);
    let evacuation = pool.plan_raise(2048).unwrap();
    assert!(!evacuation.moves.is_empty(), "state in the span moves down");
    pool.raise_applied(&evacuation).unwrap();
    assert_eq!(pool.rung(), 2048, "taken at once");
    assert_eq!(pool.copying(), Some(16), "fires held at the rung before");
    // The engine copies against the table the pool now states.
    engine
        .evacuate(&SharedPool::rung_moves(&evacuation))
        .unwrap();
    assert_eq!(engine.table.as_slice(), pool.pool().chunk_table());
    // No fire retires the fence while the copy is the engine's.
    pool.pool_mut().retire_through(1_000_000);
    let moved_off: Vec<u32> = evacuation.moves.iter().map(|step| step.from).collect();
    let shared: Shared = Arc::new(Mutex::new(pool));
    crate::store::shared::set_fire_cap_for_test(ENGINE, 16);
    crate::store::shared::raise_now(&shared, ENGINE, 2048);
    assert_eq!(
        crate::store::shared::fire_cap(ENGINE),
        16,
        "no cap lifted before the copy"
    );
    let mut pool = shared.lock().unwrap();
    pool.raise_copied(1_000_001);
    assert_eq!(pool.copying(), None);
    pool.pool_mut().retire_through(1_000_001);
    let _ = moved_off;
}

/// The submit check (`PIE_CHECK_SLOT_REFS`): a step wider than the rung
/// applied, or naming a slot its carve would cover, is named; a step within
/// the rung, naming slots below the span, is not.
#[test]
fn a_step_wider_than_the_rung_applied_or_carving_over_a_slot_is_named() {
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(16).unwrap();
    let list = pool.list().expect("a kv list");
    let low = pool.take_refs(list.pool, 1).expect("a page");
    let tables = vec![(LIST, low.clone())];
    assert_eq!(pool.over_carve(16, &tables), None);
    let wide = pool
        .over_carve(2048, &[])
        .expect("a step past the rung is named");
    assert!(wide.contains("wider than the rung applied"), "{wide}");
    // State lent the span the lowered rung gave up: a step within the rung
    // carves past it, and one of the rung that would cover it is named.
    let available = pool.pool().available(list.pool);
    let lent = pool.take_refs(list.pool, available).expect("every page");
    let spc = facts().shapes[0].slots_per_chunk;
    let span = TOTAL - pool.pool().act_max();
    let in_span: Vec<u32> = lent
        .iter()
        .copied()
        .filter(|&r| pool.pool().physical(r / spc).is_some_and(|p| p >= span))
        .collect();
    assert!(
        !in_span.is_empty(),
        "the lowered rung lent the span to state"
    );
    assert_eq!(pool.over_carve(16, &[(LIST, lent.clone())]), None);
    assert!(pool.over_carve(2048, &[(LIST, in_span)]).is_some());
    pool.give_refs(list.pool, &lent);
    pool.give_refs(list.pool, &low);
}

/// While the engine holds part of its range back, a committed chunk past
/// the ceiling is no room: the walker sees the shortfall and commits ahead
/// of a waiter, and the ceiling's lift says so, for the planner's waiters
/// to be woken.
#[test]
fn room_past_the_ceiling_is_not_headroom_and_its_lift_is_said() {
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    let all = pool.pool().headroom();
    let top = TOTAL - pool.pool().act_max();
    assert!(!pool.hold_back(u64::from(top - 8) * CHUNK), "lowered");
    let held_back = pool.pool().headroom();
    assert!(
        held_back <= 8 - pool.pool().reserved_chunks(),
        "only the chunks under the ceiling count: {held_back} of {all}"
    );
    assert!(pool.hold_back(0), "lifted: the caller wakes the waiters");
    assert_eq!(pool.pool().headroom(), all);
}

/// The first commit (the weights' at boot, the walker's ahead of a
/// waiter) never maps a segment into what the engine holds back: however
/// much it is asked for, it stops at the ceiling, so the range less what the
/// engine holds is all that is ever backed below the activation.
#[test]
fn no_commit_reaches_into_what_the_engine_holds_back() {
    let mut pool = pool();
    let top = TOTAL - pool.pool().act_max();
    // All but the lowest two segments below the activation held back.
    let ceiling = 2 * SEGMENT;
    pool.hold_back(u64::from(top - ceiling) * CHUNK);
    assert_eq!(pool.pool().ceiling(), ceiling);
    let list = pool.list().expect("a kv list").pool;
    let committed = pool.pool_mut().commit_for(&[(list, u64::MAX / 2)], 0);
    assert!(
        committed.iter().all(|&segment| segment * SEGMENT < ceiling),
        "committed {committed:?} past the ceiling at chunk {ceiling}"
    );
    assert!(
        pool.pool().commit_ahead(u64::MAX / 2).is_empty(),
        "nothing ahead past the ceiling"
    );
    assert!(pool.mapped_segments() <= 2 + (8 - pool.pool().top_segment()));
}

/// A segment that straddles the ceiling is not committed while the engine
/// holds anything back, by the weights' commit, the walker's ahead or its
/// grow: what is mapped never passes the range less what is held, not for
/// a moment.
#[test]
fn a_segment_straddling_the_ceiling_is_never_mapped() {
    let mut pool = pool();
    let top = TOTAL - pool.pool().act_max();
    // Held back: 19 chunks under the activation, so the ceiling (at 33)
    // falls inside segment 4.
    let held = 19;
    assert!(!pool.hold_back(u64::from(held) * CHUNK));
    let ceiling = pool.pool().ceiling();
    assert_eq!(ceiling, top - held);
    assert_ne!(ceiling % SEGMENT, 0, "the ceiling straddles a segment");
    let list = pool.list().expect("a kv list").pool;
    let committed = pool.pool_mut().commit_for(&[(list, u64::MAX / 2)], 0);
    assert!(
        committed
            .iter()
            .all(|&segment| (segment + 1) * SEGMENT <= ceiling),
        "committed {committed:?} past the ceiling at chunk {ceiling}"
    );
    assert!(pool.pool().commit_ahead(u64::MAX / 2).is_empty());
    let mut grows = 0;
    while let Some(SegmentMove::Commit { prefix }) = pool.next_move(u32::MAX, 0) {
        pool.pool_mut().commit_segment(prefix - 1).unwrap();
        grows += 1;
        assert!(grows < 16, "the walk keeps committing");
    }
    let bound = pool.bound_bytes(u64::from(held) * CHUNK);
    assert!(
        pool.held() <= bound,
        "mapped {} past what the engine binds {bound}",
        pool.held()
    );
}
