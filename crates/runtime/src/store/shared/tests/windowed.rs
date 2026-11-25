//! CONTRACT-BC 19, 21: a load whose engine serves windowed layers as ring +
//! buffer, against a fake engine stating windowed facts: the kv list, one
//! fixed record holding every windowed layer's ring, and
//! the window buffer. The rs store keeps each set's folded window as its
//! record and its unfolded rows in window-buffer slots; each lane names
//! them as `Lane::window` and its window-buffer table.

use std::collections::HashSet;

use ::engine::chunk::WindowedLayer;
use ::engine::fire::{FoldLen, Lineage, Serves, WindowFold};

use super::*;
use crate::pipeline::fire::rs::{
    GroupFold, GroupPlan, PreparedRs, RsPlan, demand, prepare, prepare_many, settle,
};
use crate::store::rs::{RsGeometry, RsStore, RsUnits, RsWorkingSetId};

/// The window buffer: 4 rows a slot.
const WB: EngineShape = EngineShape(6);
const WB_TOKENS: u32 = 4;
/// The window buffer's pool id: the shapes' last (kv list, record, buffer).
const WB_POOL: ShapeId = 2;

/// The facts' kv list, the one-piece record (`CONV`) holding two windowed
/// layers' rings (8 rows each), and the window buffer.
fn windowed_facts() -> ChunkFacts {
    let mut facts = facts();
    facts
        .shapes
        .retain(|shape| shape.id == LIST || shape.id == CONV);
    facts.shapes.push(Facts {
        role: SlotRole::WindowBuffer,
        ..shape(
            WB,
            SlotKind::List {
                tokens: WB_TOKENS,
                rate: 1,
                window: None,
            },
            MIB / 4,
        )
    });
    facts.windowed = vec![
        WindowedLayer {
            layer: 0,
            window: 8,
            bytes: 8 * 256,
            ..WindowedLayer::default()
        },
        WindowedLayer {
            layer: 2,
            window: 8,
            bytes: 8 * 256,
            ..WindowedLayer::default()
        },
    ];
    facts.record_bytes = facts.ring_bytes();
    facts.validate().unwrap();
    facts
}

fn geom() -> RsGeometry {
    RsGeometry {
        state_size: windowed_facts().record_bytes,
        buffer_page_tokens: 16,
    }
}

fn stores_of(facts: &ChunkFacts) -> (Shared, crate::store::registry::Stores) {
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(facts, 256).unwrap()));
    let stores = crate::store::registry::shared_stores(&shared, &Tiers::default(), 16, 256);
    (shared, stores)
}

/// The window buffer's slots the pool holds now.
fn wb_live(shared: &Shared) -> u64 {
    shared.lock().unwrap().pool().shape_demand(WB_POOL).live
}

/// The lanes of `out` as the engine gets them: lane `i` writes `rows[i] =
/// (held, count)`, its set folded through `folded` as the fire found it.
fn lanes_of(
    out: &PreparedRs,
    shared: &Shared,
    rows: &[(u32, u32)],
    folded: u32,
) -> Vec<::engine::Lane> {
    let mut req = crate::engine::FireRequest {
        lanes: rows
            .iter()
            .enumerate()
            .map(|(at, &(held, count))| {
                let mut lane = crate::engine::fire::lane_of(
                    at as u32,
                    vec![1; count as usize],
                    held,
                    Vec::new(),
                );
                lane.lane.kv.folded = folded;
                lane
            })
            .collect(),
        ..crate::engine::FireRequest::default()
    };
    out.apply_to(&mut req);
    crate::pipeline::fire::rs::apply_shared(&mut req, Some(shared), Some(out));
    req.lanes.into_iter().map(|lane| lane.lane).collect()
}

/// A lane the engine takes: stated to an engine serving rings, and its
/// refs inside the pool.
fn valid(lane: &::engine::Lane, facts: &ChunkFacts) {
    let serves = Serves {
        windowed_state: true,
        chunks: true,
        ..Serves::NONE
    };
    lane.validate_for(serves).unwrap();
    lane.validate_chunks(facts).unwrap();
    assert!(
        matches!(lane.rs, ::engine::fire::RsVerb::Fold),
        "no recurrent verb on a load of windowed layers alone"
    );
}

/// The lane's window-buffer table (empty where it names none).
fn table(lane: &::engine::Lane) -> Vec<u32> {
    lane.tables
        .iter()
        .find(|(shape, _)| *shape == WB)
        .map(|(_, refs)| refs.clone())
        .unwrap_or_default()
}

/// The lane's `[run, cow]` record pairs.
fn record(lane: &::engine::Lane) -> Vec<[u32; 2]> {
    match &lane.record {
        ::engine::LaneRecord::Pieces(pieces) => {
            pieces.iter().flat_map(|(_, pairs)| pairs.clone()).collect()
        }
        other => panic!("a windowed lane names its record, not {other:?}"),
    }
}

fn one(lanes: Vec<usize>, writes: std::ops::Range<u32>, fold: GroupFold) -> RsPlan {
    RsPlan::Groups(vec![GroupPlan {
        lanes,
        writes: Some(writes),
        fold,
        snapshots: Vec::new(),
        beams: false,
    }])
}

/// A set that has folded `rows` from nothing in one fire.
fn warm(rs: &mut RsStore, rows: u32) -> RsWorkingSetId {
    let ws = rs.create_working_set(geom());
    let out = prepare(rs, ws, rows).unwrap();
    settle(rs, out.txn);
    ws
}

/// Beam lanes on one set share its folded window: every lane names one
/// record and one buffer table, each lane's divergent rows in slots of its
/// own and read by its lineage (`Lineage::Admitted`); the group's lagged
/// common-ancestor fold folds once (its first lane), every lane naming the
/// fresh record it folds into (CONTRACT-BC 15). No lane is refused for
/// sharing the set.
#[test]
fn a_beam_group_shares_one_folded_window() {
    let facts = windowed_facts();
    let (shared, stores) = stores_of(&facts);
    let mut rs = stores.rs.lock().unwrap();
    let ws = warm(&mut rs, 5);
    // Three lanes of four rows each past the five folded.
    let plan = one(vec![0, 1, 2], 5..17, GroupFold::None);
    let out = prepare_many(&mut rs, &[ws; 3], &plan, None).unwrap();
    let lanes = lanes_of(&out, &shared, &[(5, 4), (9, 4), (13, 4)], 5);
    let shared_table = table(&lanes[0]);
    assert_eq!(shared_table.len(), 4, "rows 4..20 in slots of 4");
    for lane in &lanes {
        valid(lane, &facts);
        assert_eq!(table(lane), shared_table, "one buffer");
        assert_eq!(record(lane), record(&lanes[0]), "one folded window");
        assert_eq!(
            lane.window,
            Some(WindowFold {
                base: 4,
                fold: FoldLen::Host(0),
                lineage: Lineage::Admitted,
            })
        );
    }
    settle(&mut rs, out.txn);
    assert_eq!(wb_live(&shared), 4);

    // Each lane one row more; the common ancestor folds through 9.
    let plan = one(vec![0, 1, 2], 17..20, GroupFold::Host(4));
    let out = prepare_many(&mut rs, &[ws; 3], &plan, None).unwrap();
    let lanes = lanes_of(&out, &shared, &[(17, 1), (18, 1), (19, 1)], 5);
    let folds: Vec<FoldLen> = lanes
        .iter()
        .map(|lane| lane.window.unwrap().fold)
        .collect();
    assert_eq!(
        folds,
        vec![FoldLen::Host(4), FoldLen::Host(0), FoldLen::Host(0)],
        "the group folds once"
    );
    let [[run, cow]] = record(&lanes[0])[..] else {
        panic!("one piece")
    };
    assert_ne!(run, cow, "a group's fold writes a fresh record");
    for lane in &lanes {
        valid(lane, &facts);
        assert_eq!(record(lane), vec![[run, cow]], "every lane names it");
    }
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert_eq!((rs.frozen(ws), rs.base(ws)), (Ok(9), Ok(8)));
    assert_eq!(wb_live(&shared), 3, "the slot of rows 4..8 went");
    rs.release_working_set(ws);
    rs.retire_idle();
}

/// A fire wider than the window buffer's free slots is never refused for
/// them: one that folds every row it writes takes no buffer slot however
/// wide; one leaving its rows unfolded asks more than stands free, which is
/// counted (for the planner to wait for), not refused, and cut as a prompt
/// is, each piece asks its own rows' slots, those its pass folds coming
/// back before the next.
#[test]
fn a_fire_too_wide_for_the_free_buffer_slots_is_cut_not_refused() {
    let facts = windowed_facts();
    let (_shared, stores) = stores_of(&facts);
    let mut rs = stores.rs.lock().unwrap();
    let ws = warm(&mut rs, 4);
    let free = rs.free().pages;

    // Folding all it writes: no buffer slot, whatever the width.
    let wide = 64 * free;
    let folds_all = RsPlan::Groups(vec![GroupPlan {
        lanes: vec![0],
        writes: None,
        fold: GroupFold::Host(wide),
        snapshots: Vec::new(),
        beams: false,
    }]);
    assert_eq!(demand(&rs, &[ws], &folds_all).map(|units| units.pages), Ok(0));

    // Left unfolded: the whole asks more than stands free, and is counted.
    let rows = (free + 8) * WB_TOKENS;
    let whole = demand(&rs, &[ws], &one(vec![0], 4..4 + rows, GroupFold::None)).unwrap();
    assert!(whole.pages > free, "{} past {free}", whole.pages);
    // Cut into pieces of 16 rows, each buffering its own rows (four slots)
    // and folding the piece before it, as its pass accepts them: what a
    // piece took comes back once the next folds it.
    let mut asked = 0;
    let mut at = 4;
    while at < 4 + rows {
        let end = (at + 16).min(4 + rows);
        let behind = at - rs.frozen(ws).unwrap();
        let fold = if behind == 0 {
            GroupFold::None
        } else {
            GroupFold::Host(behind)
        };
        let plan = one(vec![0], at..end, fold);
        let piece = demand(&rs, &[ws], &plan).unwrap();
        assert!(piece.pages <= 16 / WB_TOKENS, "a piece asks its own rows' slots");
        asked += piece.pages;
        let out = prepare_many(&mut rs, &[ws], &plan, None).unwrap();
        settle(&mut rs, out.txn);
        rs.retire_idle();
        assert!(rs.held([ws]).pages <= 2 * 16 / WB_TOKENS, "two pieces' slots at most");
        at = end;
    }
    assert_eq!(asked, whole.pages, "the pieces asked what the whole would, in turn");
    assert!(asked > free, "more than stood free at once");
}

/// A ring at chunk starts (CONTRACT-BC 21, `SlotRole::Ring`): one slot a
/// chunk.
const RINGS: EngineShape = EngineShape(7);
/// A hybrid's window buffer holds 8 rows a slot, its rs list 4.
const HY_WB_TOKENS: u32 = 8;

/// A hybrid of recurrent and windowed layers: the kv list, the ring shape
/// (before the state's), the recurrent state (`CONV`), the rs list (4 rows
/// a slot) and the window buffer (8 rows a slot).
fn hybrid_facts() -> ChunkFacts {
    let mut facts = facts();
    facts.shapes.retain(|shape| shape.id == LIST || shape.id == CONV);
    facts.shapes.insert(
        1,
        Facts {
            role: SlotRole::Ring,
            ..shape(RINGS, SlotKind::Fixed { pieces: 1 }, CHUNK)
        },
    );
    facts.shapes.push(Facts {
        role: SlotRole::RsBuffer,
        ..shape(
            RS,
            SlotKind::List {
                tokens: RS_TOKENS,
                rate: 1,
                window: None,
            },
            MIB / 8,
        )
    });
    facts.shapes.push(Facts {
        role: SlotRole::WindowBuffer,
        ..shape(
            WB,
            SlotKind::List {
                tokens: HY_WB_TOKENS,
                rate: 1,
                window: None,
            },
            MIB / 4,
        )
    });
    facts.windowed = vec![WindowedLayer {
        layer: 1,
        window: 8,
        bytes: 8 * 256,
        ..WindowedLayer::default()
    }];
    facts.record_bytes = facts.ring_bytes() + 4096;
    // Each rung's rs scratch: its rows only, nothing past them laid.
    facts.act_replay = facts
        .act_chunks
        .iter()
        .map(|&(tokens, _)| super::rows_only(tokens))
        .collect();
    facts.validate().unwrap();
    facts
}

/// A hybrid parks whole: its record (ring and state pieces), its rs list
/// slots and its window-buffer slots each to host rows of their own shape,
/// and back; an export names both buffers.
#[test]
fn a_hybrid_parks_both_buffers_and_its_whole_record() {
    let mut facts = hybrid_facts();
    facts.host_chunks = 8;
    let tiers = Tiers::of(&facts).unwrap();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let mut rs = RsStore::shared(&shared, tiers.host.as_ref());
    let ws = rs.create_working_set(RsGeometry {
        state_size: facts.record_bytes,
        buffer_page_tokens: 16,
    });
    let out = prepare(&mut rs, ws, 6).unwrap();
    settle(&mut rs, out.txn);
    let out = prepare_many(&mut rs, &[ws], &one(vec![0], 6..10, GroupFold::None), None).unwrap();
    settle(&mut rs, out.txn);
    let export = rs.export(ws).unwrap();
    assert_eq!(export.buffer.iter().flatten().count(), 2, "rs 4..8, 8..12");
    assert_eq!(export.wbuffer.iter().flatten().count(), 2, "window 0..8, 8..16");
    let sets = HashSet::from([ws]);
    let held = rs.suspendable(&sets);
    assert_eq!(
        held,
        RsUnits {
            slots: 1,
            pages: 4,
            wbuf: 2,
            ..RsUnits::default()
        }
    );
    let txn = rs.prepare_suspend(&sets).expect("the host takes it");
    let moved: Vec<EngineShape> = txn
        .copy_plan()
        .shared
        .iter()
        .map(|(shape, _)| *shape)
        .collect();
    for shape in [RINGS, CONV, RS, WB] {
        assert!(moved.contains(&shape), "{shape:?} in {moved:?}");
    }
    assert_eq!(rs.commit_suspend(txn), held);
    let mut grant = rs.reserve(rs.swapped_units(&sets)).expect("room to come back");
    let txn = rs.prepare_restore(&sets, &mut grant).expect("it comes back");
    rs.commit_restore(txn);
    assert_eq!(rs.suspendable(&sets), held, "on the device again");
}

/// Two frames in flight: lane A's fire folds its buffered rows (frame N,
/// still running), and every slot then free is handed to the next frame.
/// The slots A's fold frees are not among them: its fire still reads them
/// (buffer rows into the ring), so they stay held until it settles, and only
/// then come free.
#[test]
fn a_fold_s_freed_slots_stay_held_until_its_fire_settles() {
    let facts = windowed_facts();
    let (shared, stores) = stores_of(&facts);
    let mut rs = stores.rs.lock().unwrap();
    let a = warm(&mut rs, 6);
    let b = warm(&mut rs, 6);
    let out = prepare_many(&mut rs, &[a], &one(vec![0], 6..13, GroupFold::None), None).unwrap();
    let held: Vec<u32> = table(&lanes_of(&out, &shared, &[(6, 7)], 6)[0]);
    settle(&mut rs, out.txn);

    // Frame N: A folds through 12, freeing the slots of rows 4..12.
    let folding = prepare_many(&mut rs, &[a], &one(vec![0], 13..14, GroupFold::Host(6)), None)
        .unwrap();
    rs.retire_idle();
    let freed: HashSet<u32> = held[..2].iter().copied().collect();
    let free_before = rs.free().pages;

    // Frame N+1, staged while N runs: B takes every slot that stands free.
    let grant = rs.reserve(RsUnits::of(0, free_before)).expect("what stands free");
    let taken: HashSet<u32> = grant.pages.iter().map(|&id| rs.engine_id(id)).collect();
    assert!(
        taken.is_disjoint(&freed),
        "a slot frame N's fold still reads was handed to frame N+1"
    );
    rs.release_reservation(grant);
    let out = prepare_many(&mut rs, &[b], &one(vec![0], 6..10, GroupFold::None), None).unwrap();
    let theirs: HashSet<u32> = table(&lanes_of(&out, &shared, &[(6, 4)], 6)[0])
        .into_iter()
        .collect();
    assert!(theirs.is_disjoint(&freed), "B's rows land in slots of their own");

    // Frame N lands: its freed slots come back.
    settle(&mut rs, folding.txn);
    rs.retire_idle();
    assert!(rs.free().pages >= free_before - theirs.len() as u32 + freed.len() as u32);
    settle(&mut rs, out.txn);
}
