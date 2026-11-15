//! The shared pool against a fake engine that states real, unpinned facts:
//! a list, a window list, a fixed record of three pieces and one of one, a weight
//! shape, two-ended growth.

use ::engine::chunk::{
    ChunkTable, Growth as SegmentGrowth, Reserved, ReservedRun, SegmentFacts, SlotShape as Facts,
    WeightKind, WeightShape,
};
use ::engine::fire::Lane;

use super::*;
use crate::governor::{PATIENCE, Retarget};

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

    /// The rung moves (`Engine::set_rung`): a raise backs the carve, and is
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
mod span;
mod tiers;

/// What a grow waiter is short of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Short {
    /// Kv pages: the list shape's slots.
    Kv,
    /// State: a record's pieces.
    State,
}

/// The weight cache's resident atoms above its floor, `n` of them, as it
/// takes them back into the span a lowered rung leaves: idle cache, the one
/// holder that opens a chunk there.
pub(crate) fn span_cache(pool: &mut SharedPool, n: u64) -> Vec<Slot> {
    let shape = pool.weights_first();
    let holder = Holder::Weights {
        shape,
        pinned: false,
    };
    let slots = pool.pool_mut().acquire(holder, shape, n).unwrap();
    let base = pool.pool().live_limit();
    for slot in &slots {
        assert!(
            pool.pool().physical(slot.chunk).unwrap() >= base,
            "cache opens in the top rung's span first"
        );
    }
    slots
}

/// What a raise has the weight cache do with `cached` ([`span_cache`]):
/// let it go behind the fire at `epoch`, which then retires.
pub(crate) fn clear_span_cache(pool: &mut SharedPool, cached: &[Slot], epoch: u64) {
    let shape = pool.weights_first();
    let holder = Holder::Weights {
        shape,
        pinned: false,
    };
    pool.pool_mut().release(holder, cached, epoch);
    pool.pool_mut().retire_through(epoch);
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
/// by then has retired, and the span above it is then idle cache's alone
/// (live state never opens in the top rung's span). Settled, the raise
/// finds the cache there and has it cleared, moves no state, takes the
/// largest rung and lifts the cap.
#[test]
fn the_rung_walk_caps_fires_then_lowers_and_raises_back_clearing_the_span_s_cache() {
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
    // What may be out counts the room below the top rung's span, at every
    // rung.
    let at_largest = (pool.state_chunks_now(), pool.records_capacity());
    assert_eq!(
        at_largest.0,
        TOTAL - pool.pool().reserved_chunks() - pool.pool().act_max()
    );
    assert_eq!(at_largest.1, pool.records_most());
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
    assert_eq!(
        (pool.state_chunks_now(), pool.records_capacity()),
        at_largest,
        "lowered: no more live room"
    );
    assert_eq!(
        pool.pool().state_limit(),
        TOTAL - 4,
        "the span above is idle cache's"
    );
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 9, 9).unwrap(),
        RungStep::Wait
    );
    // Every record live state may take lies below the span the larger
    // rung's fires would touch; the weight cache's atoms above its floor
    // take the span.
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
    assert_eq!(in_way(&pool), 0, "no live state in the top rung's span");
    let cached = span_cache(&mut pool, 4);
    // Settled: the raise finds the cache in its span; its owner lets it go
    // and the fire reading it retires; then the raise, nothing moved, then
    // lift the cap.
    assert_eq!(
        walk.step(&pool, u32::MAX, 256, 256, 9, 9).unwrap_err(),
        RungError::Pool(SlotError::SpanCached {
            rung: 2048,
            chunks: 2
        })
    );
    clear_span_cache(&mut pool, &cached, 9);
    let table = pool.pool().chunk_table().to_vec();
    let RungStep::Raise { rung, raise } = walk.step(&pool, u32::MAX, 256, 256, 9, 9).unwrap()
    else {
        panic!("a raise");
    };
    assert_eq!(rung, 2048);
    pool.apply_rung(&raise).unwrap();
    assert_eq!(
        pool.pool().chunk_table(),
        table.as_slice(),
        "no state moves"
    );
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

/// Rungs by demand: the rung a pass's rows ask for, and what a raise lacks:
/// nothing, however full live state is, since it never fills the span the
/// larger rung's fires take; idle cache there is no room to fund either,
/// but is cleared, and the raise then moves nothing.
#[test]
fn a_raise_by_demand_lacks_nothing_live_state_fills_and_clears_the_span_s_cache() {
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
    assert!(!all.is_empty());
    assert_eq!(
        pool.raise_short(2048),
        (0, 0),
        "live state never fills the span: the raise lacks nothing"
    );
    let cached = span_cache(&mut pool, 6);
    assert_eq!(pool.raise_short(2048), (0, 0), "cache is no room to fund");
    assert!(
        pool.plan_raise(2048)
            .is_err_and(|error| error.span_cached()),
        "cache in the span: cleared first"
    );
    clear_span_cache(&mut pool, &cached, 1);
    let raise = pool.plan_raise(2048).unwrap();
    pool.apply_rung(&raise).unwrap();
    assert_eq!(pool.rung(), 2048);
    assert_eq!(
        engine.table.as_slice(),
        pool.pool().chunk_table(),
        "no state moves"
    );
    pool.pool().check();
}

/// The walk's raise over the span's cache: it asks the cache's owner (the
/// weight cache, through the engine's [`SpanClear`]) to let go what lies
/// there and waits for room freed, the fires reading it retired; then it
/// raises, nothing moved.
#[tokio::test(flavor = "current_thread")]
async fn the_walk_has_the_span_s_cache_cleared_and_then_raises_moving_nothing() {
    const ENGINE: usize = 9304;
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(256).unwrap();
    let records = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    let cached = span_cache(&mut pool, 4);
    let table = pool.pool().chunk_table().to_vec();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let asked = Arc::new(Mutex::new(Vec::new()));
    crate::store::shared::set_span_clear(ENGINE, {
        let (shared, asked) = (shared.clone(), asked.clone());
        Arc::new(move |from| {
            asked.lock().unwrap().push(from);
            let mut pool = shared.lock().unwrap();
            assert_eq!(pool.pool().clearing(), from, "asked before the owner");
            // The owner lets go behind the fire at epoch 1; not yet retired.
            let holder = Holder::Weights {
                shape: pool.weights_first(),
                pinned: false,
            };
            pool.pool_mut().release(holder, &cached, 1);
            cached.len() as u64
        })
    });
    let grow = SharedGrow::new(0, ENGINE, shared.clone()).with_live_lanes(|| 1);
    let _want = want_rows(ENGINE, 2048);
    grow.walk_rung().await;
    assert_eq!(
        *asked.lock().unwrap(),
        [TOTAL - 12],
        "the span from 52 asked clear"
    );
    assert_eq!(shared.lock().unwrap().rung(), 256, "not raised over cache");
    // The fire reading it retires and room is freed: the walk raises.
    shared.lock().unwrap().pool_mut().retire_through(1);
    grow.room_freed();
    grow.walk_rung().await;
    let pool = shared.lock().unwrap();
    assert_eq!(pool.rung(), 2048);
    assert_eq!(pool.pool().chunk_table(), table.as_slice(), "nothing moved");
    assert_eq!(asked.lock().unwrap().len(), 1);
    assert!(
        records
            .iter()
            .flat_map(|&record| pool.record_slots(record))
            .all(|slot| pool.pool().physical(slot.chunk).unwrap() < TOTAL - 12)
    );
    pool.pool().check();
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

/// An idle pool lowered to the smallest rung stays there, the walk never
/// lowering again within the class it is in.
#[test]
fn rungs_sharing_their_chunks_are_one_class_the_walk_never_moves_within() {
    let mut pool = classed();
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

/// The cap thrash of a host-paced decode: each two-row pass lifted the cap
/// to 2 and its settle had the walker cut it back to the live lane's one
/// row, a cut a fire, the pool's span the same before and after. A cut
/// within a class frees no chunk, so the walker takes none: the pass gone
/// and nobody short, the cap stays where the pass left it, for any time.
#[tokio::test(flavor = "current_thread")]
async fn a_cap_down_that_frees_no_chunk_is_not_taken_however_long_it_stands() {
    const ENGINE: usize = 9301;
    let mut pool = classed();
    pool.lower_rung(1).unwrap();
    assert_eq!(pool.pool().act_reserved(), 1);
    let shared: Shared = Arc::new(Mutex::new(pool));
    let grow = SharedGrow::new(0, ENGINE, shared.clone()).with_live_lanes(|| 1);
    set_fire_cap_for_test(ENGINE, 1);
    grow.walk_rung().await;
    assert_eq!(fire_cap(ENGINE), 1, "one live lane, nothing asked");

    // A two-row pass asks, and is cut at its width at once.
    let want = want_rows(ENGINE, 2);
    raise_now(&shared, ENGINE, 2);
    assert_eq!(fire_cap(ENGINE), 2);
    grow.walk_rung().await;
    assert_eq!(fire_cap(ENGINE), 2);

    // Its last fire settled: the ask is gone, one lane lives, none is short.
    drop(want);
    assert_eq!(wanted_rows(ENGINE), 0);
    for tick in 0..32 * PATIENCE {
        grow.walk_rung().await;
        assert_eq!(fire_cap(ENGINE), 2, "no cut at tick {tick}");
        grow.ticked_for_test();
    }
    assert_eq!(
        shared.lock().unwrap().pool().act_reserved(),
        1,
        "the span set aside never moved"
    );
    // The next pass finds its width standing: nothing to lift.
    let _want = want_rows(ENGINE, 2);
    raise_now(&shared, ENGINE, 2);
    assert_eq!(fire_cap(ENGINE), 2);
}

/// The lower that frees chunks is still taken once the pool has stood past
/// what is asked for the governor's patience, the cap down before it; a
/// taker short of records while the largest rung holds its span stays
/// short (live state never opens in the top rung's span), and the span the
/// lower leaves is idle cache's.
#[tokio::test(flavor = "current_thread")]
async fn a_lower_past_the_patience_leaves_the_span_to_cache_not_to_a_taker_short_of_records() {
    const ENGINE: usize = 9302;
    let mut pool = classed();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.rung(), 2048);
    // Every record that fits under the largest rung is out: one more is
    // short.
    let n = pool.records_available() as usize;
    let _held = pool.take_records(n).unwrap();
    assert!(pool.take_records(1).is_none(), "short of a record");
    let shared: Shared = Arc::new(Mutex::new(pool));
    let grow = SharedGrow::new(0, ENGINE, shared.clone()).with_live_lanes(|| 1);
    assert_eq!(fire_cap(ENGINE), u32::MAX);
    for _ in 0..PATIENCE {
        grow.ticked_for_test();
        grow.walk_rung().await;
        assert_eq!(fire_cap(ENGINE), u32::MAX, "not before the patience");
    }
    grow.ticked_for_test();
    grow.walk_rung().await;
    assert_eq!(fire_cap(ENGINE), 1, "past the patience: fires cut");
    grant_rung(ENGINE, 1);
    grow.walk_rung().await;
    grow.walk_rung().await;
    let mut pool = shared.lock().unwrap();
    assert_eq!(pool.pool().act_reserved(), 1, "lowered");
    assert!(pool.take_records(1).is_none(), "the taker stays short");
    let weights = pool.weights_first();
    assert!(
        pool.pool().available_cache(weights) > 0,
        "the span the lower leaves is idle cache's"
    );
}

/// A fire term appearing that the rung in hand cannot seat beside cuts
/// fires at once, under a pass still asking the width and before any tick:
/// the fit, not the patience, decides it.
#[tokio::test(flavor = "current_thread")]
async fn a_fire_term_appearing_under_a_standing_ask_cuts_fires_at_once() {
    const ENGINE: usize = 9303;
    let shared: Shared = Arc::new(Mutex::new(classed()));
    let grow = SharedGrow::new(0, ENGINE, shared.clone()).with_live_lanes(|| 1);
    let _want = want_rows(ENGINE, 2048);
    grow.walk_rung().await;
    assert_eq!(fire_cap(ENGINE), u32::MAX, "asked and held");
    let fit = {
        let mut pool = shared.lock().unwrap();
        let key = crate::store::shared::programs::FireKey {
            program: ::engine::ProgramId(1),
            extents: ::engine::BindExtents::default(),
        };
        let fixed = 55 * pool.pool().chunk_bytes();
        pool.fire_facts_learned(
            key,
            &::engine::FireFacts {
                fixed,
                ..::engine::FireFacts::default()
            },
        );
        pool.instance_bound(::engine::InstanceId(1), key);
        pool.fit_cap()
    };
    assert!(fit < u32::MAX, "the largest rung no longer fits");
    grow.walk_rung().await;
    assert_eq!(fire_cap(ENGINE), fit, "cut to the fit before any tick");
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

/// A guest term past the top rung's span is cut as the readout term is:
/// no rung whose activation, the term's included, reaches past the span is
/// set aside, so fires are cut to the widest rung that fits it, and the
/// raise there moves nothing (live state lies below the span at every
/// rung).
#[test]
fn a_guest_term_past_the_top_rung_s_span_cuts_fires_and_moves_nothing() {
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    // Three chunks the guests keep standing at every rung: the largest
    // rung's activation would take 15 of the span's 12.
    pool.pool_mut().set_standing(3 * CHUNK);
    pool.lower_rung(16).unwrap();
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    assert_eq!(pool.pool().act_chunks(2048), 15);
    assert_eq!(pool.widest_fit(), Some(256), "256 takes 7");
    assert_eq!(pool.fit_cap(), 256);
    assert_eq!(pool.rung_for(5000), 256, "fires cut to it");
    assert_eq!(
        pool.plan_raise(2048).unwrap_err(),
        RungError::Pool(SlotError::PastSpan {
            rung: 2048,
            chunks: 15,
            span: 12
        })
    );
    let table = pool.pool().chunk_table().to_vec();
    let raise = pool.plan_raise(256).unwrap();
    assert_eq!(pool.raise_applied(&raise).unwrap(), None);
    assert_eq!(pool.rung(), 256);
    assert_eq!(pool.pool().chunk_table(), table.as_slice(), "nothing moved");
    assert!(
        all.iter()
            .flat_map(|&record| pool.record_slots(record))
            .all(|slot| pool.pool().physical(slot.chunk).unwrap() < TOTAL - 12)
    );
    pool.pool().check();
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
    let bound = pool.bound_bytes();
    assert!(
        pool.held() <= bound,
        "mapped {} past what the engine binds {bound}",
        pool.held()
    );
}

/// The room the planner counts is what can be backed: while the engine holds
/// part of the range back, the list's slots are the chunks under the
/// ceiling in segments the pool may map, as `bound_bytes` bounds them,
/// plus what the smallest rung leaves of the span; a segment straddling the
/// ceiling counts for nothing. Held on a segment boundary, exactly the held
/// chunks go; held none, the room is the range's, as before.
#[test]
fn the_room_counts_only_what_can_be_backed_under_what_the_engine_holds() {
    let mut pool = pool();
    let act_start = TOTAL - pool.pool().act_max();
    let top_start = pool.pool().top_segment() * SEGMENT;
    let whole = pool.room();
    let spc = pool.list().expect("a kv list").slots_per_chunk;
    assert_eq!(whole.list_slots, whole.state_chunks * spc);

    // Held: 19 chunks under the activation, the ceiling inside a segment.
    let held: u32 = 19;
    pool.hold_back(u64::from(held) * CHUNK);
    let ceiling = pool.pool().ceiling();
    assert_ne!(ceiling % SEGMENT, 0, "the ceiling straddles a segment");
    assert!(ceiling < top_start);
    let backable = ceiling / SEGMENT * SEGMENT;
    // The same chunks the pool may map below the span.
    let bound = pool.bound_bytes() / CHUNK;
    assert_eq!(u64::from(backable), bound - u64::from(TOTAL - top_start));
    let room = pool.room();
    assert_eq!(
        room.state_chunks,
        whole.state_chunks - (act_start - backable),
        "the held chunks and the straddled segment are no room"
    );
    assert_eq!(room.list_slots, room.state_chunks * spc);
    assert!(room.list_slots < whole.list_slots);

    // Negative control: held on a segment boundary, only the held chunks go.
    pool.hold_back(u64::from(act_start - 2 * SEGMENT) * CHUNK);
    assert_eq!(pool.pool().ceiling(), 2 * SEGMENT);
    assert_eq!(
        pool.room().state_chunks,
        whole.state_chunks - (act_start - 2 * SEGMENT)
    );
    // Held none: the range's room, as before.
    pool.hold_back(0);
    assert_eq!(pool.room(), whole);
}

/// An engine's rung gate is its own whatever its id: the 65th engine a
/// process registers is cut, granted and woken by its own gate, never by
/// another engine's and never left uncapped.
#[tokio::test]
async fn an_engine_past_the_sixty_fourth_has_its_own_rung_gate() {
    const NEAR: usize = 63;
    const PAST: usize = 64;
    const FAR: usize = 4_000_000;
    crate::store::shared::set_fire_cap_for_test(NEAR, 16);
    crate::store::shared::set_fire_cap_for_test(PAST, 256);
    crate::store::shared::set_fire_cap_for_test(FAR, 2048);
    assert_eq!(crate::store::shared::fire_cap(NEAR), 16);
    assert_eq!(crate::store::shared::fire_cap(PAST), 256, "its own cap");
    assert_eq!(crate::store::shared::fire_cap(FAR), 2048, "its own cap");
    grant_rung(PAST, 256);
    grant_rung(FAR, 2048);
    assert_eq!(rung_granted(PAST), 256);
    assert_eq!(rung_granted(FAR), 2048);
    // A wait on the 65th engine is woken by its own change, not by the
    // 64th's.
    let watch = rung_watch(PAST);
    tokio::pin!(watch);
    watch.as_mut().enable();
    rung_changed(NEAR);
    assert!(
        futures::poll!(watch.as_mut()).is_pending(),
        "another engine's change does not wake it"
    );
    rung_changed(PAST);
    assert!(futures::poll!(watch.as_mut()).is_ready(), "its own does");
}
