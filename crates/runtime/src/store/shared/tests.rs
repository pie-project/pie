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
use crate::scheduler::worker::PoolOp;

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

/// A pass cut so its window checkpoint is a piece's own end (64, the end of
/// 0..64), with the next piece (64..72) on its way before the first settles:
/// each piece keeps what 64's windows reach, so the checkpoint is taken at the
/// first piece's settle though the next one writes, and a later pass adopts
/// it at 64 by its key. Without the keep across the cut the next piece's
/// advance trims narrow 5 (rows 40..48), which 64 reaches; without taking it
/// beside the piece in flight it is refused outright.
#[test]
fn a_window_checkpoint_on_a_cut_is_kept_and_found() {
    use crate::pipeline::fire::rs::settle;
    let (facts, shared, stores) = windowed_stores();
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(window_geometry());
    let first = window_fire_keeping(&mut rs, ws, 0..64, None, Some(64));
    let (narrow, wide) = window_tables(&first, &shared);
    // The next piece is prepared (its writes in flight) before the first
    // settles, as the pass's pieces go out.
    let next = window_fire_keeping(&mut rs, ws, 64..72, None, Some(64));
    settle(&mut rs, first.txn);
    rs.retire_idle();
    assert!(
        rs.checkpoint_at(ws, [7; 32], None, 64).unwrap(),
        "the checkpoint on the cut is taken beside the piece in flight"
    );
    settle(&mut rs, next.txn);
    rs.release_working_set(ws);
    rs.retire_idle();

    // The next pass finds it by its key and stands at 64 on the slots it took.
    let warm = rs.create_working_set(window_geometry());
    assert!(rs.adopt_checkpoint(warm, &[7; 32]).unwrap());
    assert_eq!(rs.window_positions(warm), Ok(Some((64, 64))));
    let out = window_fire(&mut rs, warm, 64..65, None);
    let lane = shared_lanes(&out, &shared, 1).remove(0);
    lane.validate_chunks(&facts).unwrap();
    let (n64, w64) = window_tables(&out, &shared);
    assert_eq!(n64[5..8], narrow[5..8]);
    assert_eq!(w64[..4], wide[..4]);
    settle(&mut rs, out.txn);
}

/// The window lists' rows as a device holds them: each slot (by shape and
/// engine id) a row tag per position, written as fires' lanes name them.
#[derive(Default)]
struct WindowDevice(std::collections::BTreeMap<(EngineShape, u32), Vec<Option<(u32, u32)>>>);

impl WindowDevice {
    /// Each window list's `(shape, tokens a slot)`.
    const LISTS: [(EngineShape, u32); 2] = [(RING, 16), (NARROW, 8)];

    /// A fire of `lane` landing rows `rows`, tagged `pass`: its slot copies
    /// first, then each row at its slot's position.
    fn fire(&mut self, lane: &Lane, rows: std::ops::Range<u32>, pass: u32) {
        for (shape, pairs) in &lane.slot_copies {
            for &[src, dst] in pairs {
                let held = self.0.get(&(*shape, src)).cloned().unwrap_or_default();
                self.0.insert((*shape, dst), held);
            }
        }
        for (shape, tokens) in Self::LISTS {
            let table = Self::table(lane, shape);
            for row in rows.clone() {
                let id = table[(row / tokens) as usize];
                assert_ne!(id, 0, "row {row} of {shape:?} lands on a slot");
                let cells = self.0.entry((shape, id)).or_default();
                cells.resize(tokens as usize, None);
                cells[(row % tokens) as usize] = Some((pass, row));
            }
        }
    }

    /// What `lane` reads of rows `rows` of list `shape`.
    fn read(&self, lane: &Lane, shape: EngineShape, tokens: u32, rows: std::ops::Range<u32>) -> Vec<Option<(u32, u32)>> {
        let table = Self::table(lane, shape);
        rows.map(|row| {
            self.0
                .get(&(shape, table[(row / tokens) as usize]))
                .and_then(|cells| cells.get((row % tokens) as usize).copied().flatten())
        })
        .collect()
    }

    fn table(lane: &Lane, shape: EngineShape) -> Vec<u32> {
        lane.tables
            .iter()
            .find(|(id, _)| *id == shape)
            .map(|(_, refs)| refs.clone())
            .unwrap_or_default()
    }
}

/// A checkpoint mid-slot (60: 12 rows into wide slot 3, 4 into narrow 7),
/// taken at its piece's settle beside the next piece (60..100) already on
/// its way: that piece writes rows 60.. into the slots the checkpoint shares,
/// and goes far enough that its windows move past every slot the checkpoint
/// reads. Settled before the adopt, it leaves the checkpoint's rows as a cold
/// fold to 60 has them: a slot's rows past 60 are another position of the
/// slot (a list's slot `i` holds rows `[i T, (i + 1) T)`, never wrapping), and
/// the slots the pass retires the checkpoint holds by refcount. The adopter's
/// first write into a shared slot copies it first (`RsBufferTarget::Cow`).
/// A cold fold after the pass, and sets live beside it, write none of the
/// checkpoint's slots; two resumes from it each write their own copy.
#[test]
fn a_mid_slot_window_checkpoint_reads_what_a_cold_fold_does() {
    use crate::pipeline::fire::rs::settle;
    const AT: u32 = 60;
    let (_facts, shared, stores) = windowed_stores();
    let mut rs = stores.rs.lock().unwrap();
    let mut device = WindowDevice::default();
    let ws = rs.create_working_set(window_geometry());
    let first = window_fire_keeping(&mut rs, ws, 0..AT, None, Some(AT));
    let first_lane = shared_lanes(&first, &shared, 1).remove(0);
    let next = window_fire_keeping(&mut rs, ws, AT..100, None, Some(AT));
    let next_lane = shared_lanes(&next, &shared, 1).remove(0);
    device.fire(&first_lane, 0..AT, 1);
    settle(&mut rs, first.txn);
    rs.retire_idle();
    assert!(rs.checkpoint_at(ws, [9; 32], None, AT).unwrap());
    // The next piece was laid before the checkpoint shared its slots: it
    // writes them in place, rows 60.. of them.
    device.fire(&next_lane, AT..100, 1);
    settle(&mut rs, next.txn);
    rs.release_working_set(ws);
    rs.retire_idle();

    // A cold fold to 60 on a set of its own.
    let cold_ws = rs.create_working_set(window_geometry());
    let cold = window_fire(&mut rs, cold_ws, 0..AT, None);
    let cold_lane = shared_lanes(&cold, &shared, 1).remove(0);
    // Its own tag: a slot it takes that the checkpoint still reads shows.
    device.fire(&cold_lane, 0..AT, 3);
    settle(&mut rs, cold.txn);

    // Other sets live beside it take the slots the pass let go, each
    // written with a tag of its own: a slot the checkpoint reads that went
    // back to the pool shows.
    let fillers: Vec<_> = (0..4)
        .map(|_| {
            let filler = rs.create_working_set(window_geometry());
            let out = window_fire(&mut rs, filler, 0..AT, None);
            device.fire(&shared_lanes(&out, &shared, 1).remove(0), 0..AT, 4);
            settle(&mut rs, out.txn);
            filler
        })
        .collect();

    let warm = rs.create_working_set(window_geometry());
    assert!(rs.adopt_checkpoint(warm, &[9; 32]).unwrap());
    assert_eq!(rs.window_positions(warm), Ok(Some((AT, AT))));
    let resume = window_fire(&mut rs, warm, AT..AT + 1, None);
    let lane = shared_lanes(&resume, &shared, 1).remove(0);
    for (shape, tokens) in WindowDevice::LISTS {
        let whole = WindowDevice::table(&lane, shape)[(AT / tokens) as usize];
        let shared_slot = WindowDevice::table(&first_lane, shape)[(AT / tokens) as usize];
        assert!(
            lane.slot_copies
                .iter()
                .any(|(id, pairs)| *id == shape && pairs.contains(&[shared_slot, whole])),
            "{shape:?}: the resume copies the shared slot {shared_slot} before writing row {AT}"
        );
    }
    device.fire(&lane, AT..AT + 1, 2);
    for (shape, tokens) in WindowDevice::LISTS {
        let window = if shape == RING { 64 } else { 24 };
        let reach = AT.saturating_sub(window)..AT;
        let warm_rows = device.read(&lane, shape, tokens, reach.clone());
        let cold_rows = device.read(&cold_lane, shape, tokens, reach.clone());
        let row_of = |cells: &[Option<(u32, u32)>]| cells.iter().map(|cell| cell.map(|(_, row)| row)).collect::<Vec<_>>();
        assert_eq!(row_of(&warm_rows), row_of(&cold_rows), "{shape:?}: the rows a cold fold reads");
        assert!(cold_rows.iter().all(|cell| cell.is_some_and(|(pass, _)| pass == 3)));
        assert!(
            warm_rows.iter().zip(reach).all(|(cell, row)| *cell == Some((1, row))),
            "{shape:?}: the checkpoint's rows are the pass's own, {warm_rows:?}"
        );
        assert_eq!(device.read(&lane, shape, tokens, AT..AT + 1), [Some((2, AT))]);
    }
    settle(&mut rs, resume.txn);

    // A second resume from the checkpoint copies the shared slot again: its
    // row 60 is its own, the first resume's is the first's.
    let again = rs.create_working_set(window_geometry());
    assert!(rs.adopt_checkpoint(again, &[9; 32]).unwrap());
    let out = window_fire(&mut rs, again, AT..AT + 1, None);
    let second = shared_lanes(&out, &shared, 1).remove(0);
    device.fire(&second, AT..AT + 1, 5);
    for (shape, tokens) in WindowDevice::LISTS {
        assert_eq!(device.read(&second, shape, tokens, AT..AT + 1), [Some((5, AT))]);
        assert_eq!(device.read(&lane, shape, tokens, AT..AT + 1), [Some((2, AT))], "{shape:?}");
        let window = if shape == RING { 64 } else { 24 };
        let reach = AT.saturating_sub(window)..AT;
        assert!(device.read(&second, shape, tokens, reach.clone()).iter().zip(reach).all(|(cell, row)| *cell == Some((1, row))));
    }
    settle(&mut rs, out.txn);
    drop(fillers);
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
    assert_eq!(
        rs.checkpoint_at(trimmed, [1; 32], None, 64),
        Err(crate::store::rs::RsError::CheckpointRetired { at: 64 })
    );
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
    assert_eq!(
        rs.checkpoint_at(ws, [3; 32], None, 80),
        Err(crate::store::rs::RsError::CheckpointAhead { at: 80, committed: 72 }),
        "ahead"
    );
    assert_eq!(
        rs.checkpoint_at(ws, [3; 32], None, 60),
        Err(crate::store::rs::RsError::CheckpointRetired { at: 60 }),
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

/// An idle pool lowered to the smallest rung stays there, the walk never
/// lowering again within the class it is in; a move up within it is told
/// the engine, moving nothing.
#[test]
fn rungs_sharing_their_chunks_are_one_class_the_walk_never_lowers_within() {
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
    // A goal inside the class sets aside nothing more: past the rung the
    // engine was told it is a raise that moves nothing (the engine hears
    // the wider rung before any fire takes it), then the cap lifts.
    let RungStep::Raise { rung, evacuation } = walk.step(&pool, 16, 1, 1, 40, 40).unwrap() else {
        panic!("past the rung told: a raise");
    };
    assert_eq!(rung, 16);
    assert!(
        evacuation.moves.is_empty(),
        "nothing moves within the class"
    );
    assert_eq!(
        pool.raise_applied(&evacuation).unwrap(),
        None,
        "no lane trip"
    );
    assert_eq!(pool.told(), 16);
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
    for tick in 0..64 {
        grow.walk_rung().await;
        assert_eq!(fire_cap(ENGINE), 2, "no cut at tick {tick}");
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

/// The lower that frees chunks is taken once a taker is short of them, the
/// cap down before it: a taker short of records while the largest rung
/// holds its span is seated from the span the lower lends. Until one is,
/// the rung stands however long nothing asks its width.
#[tokio::test(flavor = "current_thread")]
async fn a_lower_seats_a_taker_short_of_records() {
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
    for _ in 0..64 {
        grow.walk_rung().await;
        assert_eq!(fire_cap(ENGINE), u32::MAX, "nothing is short yet");
    }
    // The taker parks short of its record, as the planner notes it.
    {
        let mut pool = shared.lock().unwrap();
        let shape = pool.fixed()[0];
        pool.pool_mut()
            .wait(1, shape.pool, u64::from(shape.pieces));
    }
    grow.walk_rung().await;
    assert_eq!(fire_cap(ENGINE), 1, "a taker is short: fires cut");
    grant_rung(ENGINE, 1);
    grow.walk_rung().await;
    grow.walk_rung().await;
    let mut pool = shared.lock().unwrap();
    assert_eq!(pool.pool().act_reserved(), 1, "lowered: the span lent");
    assert!(pool.take_records(1).is_some(), "the taker is seated");
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

/// A fire term appearing that the rung in hand cannot seat beside cuts
/// fires at once, under a pass still asking the width and before any tick:
/// the fit decides it, not a waiter short of chunks.
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

/// A lane's kv list table says what it holds: a translation where the
/// runtime stamped one (a device-geometry instance's lane), else the
/// lane's pages, though the lane beside it is a device-geometry one; a
/// lane with neither states no list table, and pages as its kind.
#[test]
fn a_lane_s_list_kind_is_stated_beside_its_table() {
    let facts = fixed_facts();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let mut placed = crate::engine::fire::lane_of(0, vec![1], 0, Vec::new());
    placed.pages = vec![5];
    placed.translation = vec![7, 8];
    let mut req = crate::engine::FireRequest {
        lanes: vec![
            placed,
            crate::engine::fire::lane_of(1, vec![1], 0, vec![9]),
            crate::engine::fire::lane_of(2, vec![1], 0, Vec::new()),
        ],
        ..crate::engine::FireRequest::default()
    };
    crate::pipeline::fire::rs::apply_shared(&mut req, Some(&shared), None);
    let list = |lane: &Lane| {
        lane.tables
            .iter()
            .find(|(shape, _)| *shape == LIST)
            .map(|(_, refs)| refs.clone())
    };
    let [placed, unattached, bare] = req.lanes.as_slice() else {
        panic!("three lanes");
    };
    assert_eq!(placed.list, ::engine::ListKind::Translation);
    assert_eq!(
        list(placed),
        Some(vec![7, 8]),
        "the translation, not the pages"
    );
    assert_eq!(
        unattached.list,
        ::engine::ListKind::Pages,
        "beside it, pages"
    );
    assert_eq!(list(unattached), Some(vec![9]));
    assert_eq!((bare.list, list(bare)), (::engine::ListKind::Pages, None));
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
        assert!(matches!(engine.calls[0], Call::Evacuate(_)), "{:?}", engine.calls);
        assert_eq!(engine.calls[1..], [Call::Release(5)]);
        agree(&shared, &engine);
    }
    assert_eq!(shared.lock().unwrap().mapped_segments(), 7);
    assert_eq!(grow.retarget(6), Retarget::NoRoom { held: 7 });
}

/// The weights' pinned slots never hold state: the room a need is held
/// against (the never-fits check, a guest bind's short-or-refused read)
/// leaves them out. Counted whole, a Qwen3-30B-A3B ROCm boot whose pool
/// was almost all weights left its first binds waiting 45 minutes for
/// room no move could make, nothing running.
#[test]
fn the_state_room_leaves_out_the_weights_pinned_slots() {
    let mut pool = pool();
    let (most, now) = (pool.state_chunks_most(), pool.state_chunks_now());
    let shape = pool.weights_first();
    let holder = crate::store::slots::Holder::Weights {
        shape,
        pinned: true,
    };
    pool.pool_mut()
        .acquire(holder, shape, 8)
        .expect("the weights' slots");
    // Eight atoms, two a chunk: four chunks.
    assert_eq!(pool.room().weight_chunks, 4);
    assert_eq!(pool.state_chunks_most(), most - 4);
    assert_eq!(pool.state_chunks_now(), now - 4);
}

/// A compaction the planner asked for that no window clears by the walk's
/// turn is answered, never left pending (else a restore's plan ends
/// "compaction asked" every pass, the walk sending none, nothing in
/// flight): the ask is dropped, and the walk tells the
/// planner, whose next pass reads the pool as it stands.
#[test]
fn a_compaction_asked_that_no_window_clears_is_answered() {
    let mut p = pool();
    p.walked_to(Arc::new(std::sync::atomic::AtomicU32::new(8)));
    let list = p.pool_shape(LIST).expect("the kv list's pool shape");
    let holder = crate::store::slots::Holder::WorkingSet(1);
    let spc = p.pool().shape(list).expect("its shape").slots_per_chunk;
    let free = p.pool().free_chunks();
    let held = p
        .pool_mut()
        .acquire(holder, list, u64::from(free * spc))
        .expect("every committed chunk");
    // One free run of three (1..=3), and free chunks 5, 12 and 14 scattered:
    // a window over 11..=13 clears a second run by moving 11 and 13 to 5
    // and 14.
    let physical = |p: &SharedPool, slot: &crate::store::slots::Slot| p.pool().physical(slot.chunk);
    let back: Vec<_> = held
        .iter()
        .copied()
        .filter(|slot| physical(&p, slot).is_some_and(|at| [1, 2, 3, 5, 12, 14].contains(&at)))
        .collect();
    p.pool_mut().release(holder, &back, 0);
    p.pool_mut().retire_through(u64::MAX);
    assert_eq!(p.pool().free_chunk_runs(3), 1);
    assert!(p.compaction_comes(3, 2), "a window clears a second run now");
    // Before the walk's turn, every free chunk is taken: no window clears.
    let left = p.pool().free_chunks();
    let taken = p
        .pool_mut()
        .acquire(holder, list, u64::from(left * spc))
        .expect("the free chunks");
    assert!(p.pool().plan_compaction_for(3, 2).is_none(), "no window clears now");
    // At the segments it has: no grow comes first.
    let at = p.mapped_segments();
    let step = p.next_move(at, 1);
    assert!(
        !matches!(step, Some(SegmentMove::Compact { .. })),
        "nothing to send: {step:?}"
    );
    assert!(p.take_compaction_answered(), "the ask is answered");
    assert!(!p.compaction_comes(3, 2), "asked again on the pool as it stands, none comes");
    drop(taken);
}
