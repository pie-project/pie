//! The shared pool against a fake engine that states real, unpinned facts:
//! a list, a fixed record of three pieces and one of one, a weight
//! shape, two-ended growth.

use ::engine::chunk::{
    ChunkTable, Growth as SegmentGrowth, Reserved, ReservedRun, SegmentFacts, SlotShape as Facts,
    WeightKind, WeightShape,
};
use ::engine::fire::Lane;

use super::*;
use crate::governor::Retarget;
use crate::scheduler::worker::PoolOp;

const MIB: u64 = 1 << 20;
const CHUNK: u64 = 16 * MIB;
const TOTAL: u32 = 64;
const SEGMENT: u32 = 8;
const LIST: EngineShape = EngineShape(0);
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
pub(super) fn facts() -> ChunkFacts {
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
    CommitSpan(u32),
    ReleaseSpan(u32),
    Rung(u32, u32),
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

    /// An engine whose span is backed on demand: none of it from the load.
    fn on_demand() -> Self {
        Self {
            top: TOTAL / SEGMENT,
            ..Self::new()
        }
    }

    /// Backs the top `n` span segments: monotone, from the top. Refused
    /// as a placement heap the device has no room for is.
    fn commit_span(&mut self, n: u32) -> ::engine::Result<()> {
        if self.refuse {
            return Err(::engine::Error::Exhausted {
                resource: "placement heap",
                wanted: u64::from(n),
                available: 0,
            });
        }
        assert!(TOTAL / SEGMENT - n <= self.top, "commit_span only widens");
        self.top = TOTAL / SEGMENT - n;
        self.calls.push(Call::CommitSpan(n));
        Ok(())
    }

    /// The rung moves (`Pool::set_rung`): a raise backs the carve, and is
    /// refused as a placement heap the device has no room for is.
    fn set_rung(&mut self, from: u32, to: u32) -> ::engine::Result<()> {
        if self.refuse && to > from {
            return Err(::engine::Error::Exhausted {
                resource: "placement heap",
                wanted: u64::from(to),
                available: 0,
            });
        }
        self.calls.push(Call::Rung(from, to));
        Ok(())
    }

    /// Keeps the top `n` span segments, the rest given back.
    fn release_span(&mut self, n: u32) {
        assert!(TOTAL / SEGMENT - n >= self.top, "release_span only narrows");
        self.top = TOTAL / SEGMENT - n;
        self.calls.push(Call::ReleaseSpan(n));
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
                .chain(lane.record.pieces().iter().flat_map(|(shape, pieces)| {
                    pieces.iter().flatten().map(move |&r| (*shape, r))
                }));
        for (shape, slot) in refs {
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
            Some(SegmentMove::Evacuate { evacuation, .. }) => {
                let _ = engine.evacuate(&evacuation);
                return Retarget::Reached;
            }
            Some(SegmentMove::Compact {
                evacuation, epoch, ..
            }) => {
                let _ = engine.evacuate(&evacuation);
                pool.compaction_copied(epoch, epoch);
            }
            Some(SegmentMove::Release { prefix }) => {
                engine.release_segments(prefix).unwrap();
            }
            Some(SegmentMove::CommitSpan { n }) => {
                if engine.commit_span(n).is_err() {
                    return Retarget::Reached;
                }
                pool.span_committed(n);
            }
            Some(SegmentMove::ReleaseSpan { n }) => {
                engine.release_span(n);
            }
        }
    }
}

/// A model whose state is fixed records, its unfolded rows in an rs list
/// (CONTRACT-BC 18), the shapes' last.
fn fixed_facts() -> ChunkFacts {
    let mut facts = facts();
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
    // Each rung's rs scratch: its rows only, nothing past them laid.
    facts.act_replay = facts
        .act_chunks
        .iter()
        .map(|&(tokens, _)| super::rows_only(tokens))
        .collect();
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
    req.lanes.into_iter().map(|lane| lane.lane).collect()
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
        beams: false,
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
        assert!(
            matches!(&lane.record, ::engine::LaneRecord::Pieces(pieces)
                if !pieces.is_empty() && pieces.iter().all(|(_, p)| !p.is_empty())),
            "a lane bound to a record names it"
        );
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
        beams: false,
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

mod programs;
mod span;
mod tiers;
mod windowed;

/// What a grow waiter is short of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Short {
    /// Kv pages: the list shape's slots.
    Kv,
    /// State: a record's pieces.
    State,
}

/// For the walk's and the planner's tests: an on-demand pool at the largest
/// rung, the engine told, its prefix committed whole (`prefix_whole`) or as
/// loaded, every record held. With `lent`, the 256 rung lends span no
/// segment backs; without, what the rung lends is backed already.
pub(crate) fn grow_pool(prefix_whole: bool, lent: bool) -> SharedPool {
    let (mut pool, mut engine) = span::on_demand_pool();
    if prefix_whole {
        assert_eq!(
            walk_segments(&mut pool, 6, 0, &mut engine),
            Retarget::Reached
        );
        assert_eq!(pool.pool().prefix_segments(), 6, "the prefix whole");
    }
    if lent {
        pool.lower_rung(256).unwrap();
    } else {
        while let Some(n) = pool.pool().span_to_commit() {
            pool.span_committed(n);
        }
    }
    let records = pool.records_available() as usize;
    pool.take_records(records).unwrap();
    pool
}

/// Registers a waiter on `pool` short of `short`, well past what commits
/// ahead.
pub(crate) fn wait_short(pool: &mut SharedPool, short: Short) {
    match short {
        Short::Kv => pool.pool_mut().wait(1, 0, 1024),
        Short::State => {
            let shape = pool.fixed()[0];
            pool.pool_mut()
                .wait(1, shape.pool, 64 * u64::from(shape.pieces));
        }
    }
}

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
    pool.raise_applied(&evacuation).unwrap();
    pool.raise_copied(10);
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
            record: ::engine::LaneRecord::Pieces(pool.lane_fixed(record, record)),
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
/// activation's chunk count (1 to 16 one class of one chunk).
fn classed() -> SharedPool {
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
    SharedPool::new(&facts, 256).unwrap()
}

/// The stack wedge (500 processes at once): the rung rose to the largest
/// while nothing waited, and took the committable room; guest binds then
/// waited behind it, and the walk held the throughput rung (and asked to
/// rise again) while they did. A throughput rung is ordered after every
/// process demand: while one waits the walk comes down at once to what the
/// live lanes need, its give-back tells the room signal, a throughput raise
/// asked meanwhile takes nothing, and the rung rises again only once no
/// demand waits.
#[tokio::test(flavor = "current_thread")]
async fn a_throughput_rung_gives_room_back_to_waiting_binds_and_rises_once_none_waits() {
    const ENGINE: usize = 9306;
    let mut pool = classed();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.rung(), 2048);
    let shared: Shared = Arc::new(Mutex::new(pool));
    let grow = SharedGrow::new(0, ENGINE, shared.clone()).with_live_lanes(|| 1);
    // A cuttable pass asks the largest rung, fundable: nothing waits.
    let want = want_rows(ENGINE, 2048);
    grow.walk_rung().await;
    assert_eq!(fire_cap(ENGINE), u32::MAX);

    // A guest bind arrives short of room: the walk gives the span back at
    // once, past the pass's ask and with no patience.
    let room = shared.lock().unwrap().pool().room_watch();
    shared.lock().unwrap().set_guest_head(Some((1, 4)));
    grow.walk_rung().await;
    assert_eq!(fire_cap(ENGINE), 1, "cut to the live lane's rung");
    grant_rung(ENGINE, 1);
    // Behind the retire fence, as every lower.
    grow.walk_rung().await;
    grow.walk_rung().await;
    assert_eq!(
        shared.lock().unwrap().pool().act_reserved(),
        1,
        "lowered: the span given back"
    );
    tokio::time::timeout(std::time::Duration::from_secs(1), room.given())
        .await
        .expect("the give-back tells the room signal");

    // While the bind waits, a throughput raise takes nothing.
    raise_now(&shared, ENGINE, 2048);
    grow.walk_rung().await;
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 1);
    assert_eq!(fire_cap(ENGINE), 1);

    // The bind lands: the rung rises again for the pass.
    shared.lock().unwrap().set_guest_head(None);
    grow.walk_rung().await;
    assert_eq!(shared.lock().unwrap().rung(), 2048);
    assert_eq!(fire_cap(ENGINE), u32::MAX);
    drop(want);
}

/// While the engine holds back part of the range it stated, no slot it
/// binds none of is handed out: every ref lies in a committed segment,
/// past the null chunk and below the span held back; once it holds none,
/// the rest opens.
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
    let bound = pool.bound_bytes();
    assert!(
        pool.held() <= bound,
        "mapped {} past what the engine binds {bound}",
        pool.held()
    );
}

/// A load that holds recurrent state states every lane's record: the one
/// its working set binds, or none, never left unsaid; a load without
/// state states none.
#[test]
fn every_lane_of_a_state_holding_load_states_its_record() {
    use crate::pipeline::fire::rs::{GroupFold, GroupPlan, RsPlan, prepare_many, settle};
    use crate::store::rs::RsGeometry;

    let facts = fixed_facts();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let stores = crate::store::registry::shared_stores(&shared, &Tiers::default(), 16, 256);
    let mut rs = stores.rs.lock().unwrap();
    let geometry = RsGeometry {
        state_size: 1024,
        buffer_page_tokens: 16,
    };
    let bound = rs.create_working_set(geometry.clone());
    let unbound = rs.create_working_set(geometry);
    // Lane 0's working set writes its state; lane 1's is in no group.
    let plan = RsPlan::Groups(vec![GroupPlan {
        lanes: vec![0],
        writes: None,
        fold: GroupFold::Host(3),
        snapshots: Vec::new(),
        beams: false,
    }]);
    let out = prepare_many(&mut rs, &[bound, unbound], &plan, None).unwrap();
    let lanes = shared_lanes(&out, &shared, 2);
    assert!(
        matches!(&lanes[0].record, ::engine::LaneRecord::Pieces(pieces) if !pieces.is_empty()),
        "{:?}",
        lanes[0].record
    );
    assert_eq!(
        lanes[1].record,
        ::engine::LaneRecord::Null,
        "stated: no state"
    );
    settle(&mut rs, out.txn);

    // The negative control: a pool that lays no record states none.
    let mut stateless = facts.clone();
    stateless.shapes.retain(|shape| {
        shape.role != SlotRole::State || !matches!(shape.kind, SlotKind::Fixed { .. })
    });
    let stateless: Shared = Arc::new(Mutex::new(SharedPool::new(&stateless, 256).unwrap()));
    let mut req = crate::engine::FireRequest {
        lanes: vec![crate::engine::fire::lane_of(0, vec![1], 0, vec![3])],
        ..crate::engine::FireRequest::default()
    };
    crate::pipeline::fire::rs::apply_shared(&mut req, Some(&stateless), None);
    assert_eq!(req.lanes[0].record, ::engine::LaneRecord::Unstated);
}

/// An evacuation the engine refuses copied nothing, so the pool takes it
/// back whole: every chunk lies where it lay, in the pool's table as in the
/// engine's, the segment stays committed and mapped, nothing is released,
/// and the answer is short of the target, never reached. An evacuation the
/// engine takes later goes through.
#[tokio::test(flavor = "current_thread")]
async fn a_refused_evacuation_is_taken_back_whole_and_a_later_one_goes_through() {
    use crate::governor::PoolTarget;

    let mut pool = pool();
    pool.committed(6);
    let records = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    pool.give_records(&records[..16]);
    let kept = &records[16..];
    let placed = |pool: &SharedPool| -> Vec<Vec<u32>> {
        kept.iter()
            .map(|&id| {
                pool.record_slots(id)
                    .iter()
                    .map(|slot| pool.pool().physical(slot.chunk).unwrap())
                    .collect()
            })
            .collect()
    };
    let before = placed(&pool);
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(FakeEngine {
        prefix: 6,
        refuse: true,
        ..FakeEngine::new()
    }));
    let lane = engine.clone();
    let mut grow = SharedGrow::new(0, 9401, shared.clone()).with_lane(move |op| {
        let mut engine = lane.lock().unwrap();
        match op {
            PoolOp::Evacuate(evacuation) => engine.evacuate(&evacuation),
            PoolOp::ReleaseSegments(prefix) => engine.release_segments(prefix),
            PoolOp::CommitSegments(prefix) => engine.commit_segments(prefix),
            PoolOp::CheckTable(runtime) => {
                match super::programs::table_apart(engine.table.as_slice(), &runtime) {
                    Some(apart) => Err(::engine::Error::Program(apart)),
                    None => Ok(()),
                }
            }
            other => panic!("no {other:?} in this walk"),
        }
        .map_err(Into::into)
    });
    // Every logical chunk lies where the engine has it.
    let agree = |shared: &Shared, engine: &FakeEngine| {
        let pool = shared.lock().unwrap();
        for (logical, &physical) in pool.pool().chunk_table().iter().enumerate() {
            assert_eq!(
                engine.table.physical(logical as u32),
                Some(physical),
                "logical chunk {logical}"
            );
        }
        pool.pool().check();
    };

    assert_eq!(grow.retarget(6), Retarget::Moving { held: 8 });
    grow.walk().await;
    {
        let engine = engine.lock().unwrap();
        assert!(engine.calls.is_empty(), "nothing copied, nothing released");
        agree(&shared, &engine);
    }
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.mapped_segments(), 8);
        assert_eq!(pool.draining_segments(), 0);
        assert_eq!(pool.pool().segments().state(5), SegmentState::Committed);
        assert_eq!(placed(&pool), before, "every record where it was");
    }
    assert_eq!(grow.retarget(6), Retarget::NoRoom { held: 8 });

    // The engine takes the next: segment 5 goes, and segment 4 finds no
    // room left.
    engine.lock().unwrap().refuse = false;
    grow.walk().await;
    {
        let engine = engine.lock().unwrap();
        assert!(
            matches!(engine.calls[0], Call::Evacuate(_)),
            "{:?}",
            engine.calls
        );
        assert_eq!(engine.calls[1..], [Call::Release(5)]);
        agree(&shared, &engine);
    }
    assert_eq!(shared.lock().unwrap().mapped_segments(), 7);
    assert_eq!(grow.retarget(6), Retarget::NoRoom { held: 7 });
}
