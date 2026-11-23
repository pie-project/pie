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

/// The window-buffer copies the lane states.
fn copies(lane: &::engine::Lane) -> Vec<[u32; 2]> {
    lane.slot_copies
        .iter()
        .find(|(shape, _)| *shape == WB)
        .map(|(_, pairs)| pairs.clone())
        .unwrap_or_default()
}

fn one(lanes: Vec<usize>, writes: std::ops::Range<u32>, fold: GroupFold) -> RsPlan {
    RsPlan::Groups(vec![GroupPlan {
        lanes,
        writes: Some(writes),
        fold,
        snapshots: Vec::new(),
    }])
}

/// A set that has folded `rows` from nothing in one fire.
fn warm(rs: &mut RsStore, rows: u32) -> RsWorkingSetId {
    let ws = rs.create_working_set(geom());
    let out = prepare(rs, ws, rows).unwrap();
    settle(rs, out.txn);
    ws
}

/// A lane that folds every row it writes takes no buffer slot (its rows'
/// K/V live in the fire) and states its window folding them all; rows left
/// unfolded take window-buffer slots as they accrue, the table from `base`
/// aligned to the slot; a fold gives back every slot whose rows are all
/// folded.
#[test]
fn window_buffer_slots_are_taken_on_demand_and_freed_by_the_fold() {
    let facts = windowed_facts();
    let (shared, stores) = stores_of(&facts);
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.window_buffer().map(|shape| shape.id), Some(WB));
        assert_eq!(pool.list().map(|shape| shape.id), Some(LIST), "one kv list");
    }
    let mut rs = stores.rs.lock().unwrap();
    assert!(rs.is_shared() && !rs.is_window(), "a recurrent store, no window list");

    let ws = rs.create_working_set(geom());
    let out = prepare(&mut rs, ws, 6).unwrap();
    let lane = lanes_of(&out, &shared, &[(0, 6)], 0).remove(0);
    valid(&lane, &facts);
    assert_eq!(
        lane.window,
        Some(WindowFold {
            base: 4,
            fold: FoldLen::Host(6),
            lineage: Lineage::Set,
        })
    );
    assert!(table(&lane).is_empty(), "rows a fire folds take no buffer slot");
    assert_eq!(record(&lane).len(), 1, "its record, the rings");
    settle(&mut rs, out.txn);
    assert_eq!(wb_live(&shared), 0);
    assert_eq!(rs.frozen(ws), Ok(6));

    // Rows 6..13 left unfolded: slots of rows 4..8, 8..12, 12..16.
    let plan = one(vec![0], 6..13, GroupFold::None);
    assert_eq!(
        demand(&rs, &[ws], &plan),
        Ok(RsUnits::of(0, 3)),
        "three slots, the record untouched"
    );
    let out = prepare_many(&mut rs, &[ws], &plan, None).unwrap();
    let lane = lanes_of(&out, &shared, &[(6, 7)], 6).remove(0);
    valid(&lane, &facts);
    assert_eq!(
        lane.window,
        Some(WindowFold {
            base: 4,
            fold: FoldLen::Host(0),
            lineage: Lineage::Set,
        })
    );
    let held = table(&lane);
    assert_eq!(held.len(), 3);
    assert_eq!(
        held.iter().copied().collect::<HashSet<u32>>().len(),
        3,
        "every entry a slot of its own"
    );
    let [[run, cow]] = record(&lane)[..] else {
        panic!("one piece")
    };
    assert_eq!(run, cow, "a fire folding nothing writes no record");
    settle(&mut rs, out.txn);
    assert_eq!(wb_live(&shared), 3);

    // A decode row folding six: the fire reads the buffer as it found it,
    // and the fold through 12 gives back the slots of rows 4..12.
    let out = prepare_many(&mut rs, &[ws], &one(vec![0], 13..14, GroupFold::Host(6)), None)
        .unwrap();
    let lane = lanes_of(&out, &shared, &[(13, 1)], 6).remove(0);
    valid(&lane, &facts);
    assert_eq!(table(&lane), held, "the buffer as the fire found it");
    assert_eq!(
        lane.window.map(|window| (window.base, window.fold)),
        Some((4, FoldLen::Host(6)))
    );
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert_eq!((rs.frozen(ws), rs.base(ws)), (Ok(12), Ok(12)));
    assert_eq!(wb_live(&shared), 1, "the slots wholly folded went");

    rs.release_working_set(ws);
    rs.retire_idle();
    assert_eq!(wb_live(&shared), 0, "every slot back");
}

/// A record a fork shares is copied once, at its first fold, never by a
/// fire that only buffers (copy-on-first-fold: `[run, cow]`, `cow != run`);
/// a write into a partly filled buffer slot the fork shares copies that
/// slot first (`Lane::slot_copies` under the buffer's shape), the full ones
/// staying shared.
#[test]
fn a_shared_record_is_copied_at_its_first_fold() {
    let facts = windowed_facts();
    let (shared, stores) = stores_of(&facts);
    let mut rs = stores.rs.lock().unwrap();
    let parent = warm(&mut rs, 6);
    let out = prepare_many(&mut rs, &[parent], &one(vec![0], 6..10, GroupFold::None), None)
        .unwrap();
    let theirs = table(&lanes_of(&out, &shared, &[(6, 4)], 6)[0]);
    settle(&mut rs, out.txn);
    assert_eq!(theirs.len(), 2, "rows 4..8 and 8..12 (8..10 written)");
    let held = rs.folded_slot(parent).unwrap();

    let child = rs.fork(parent).unwrap();
    let out = prepare_many(&mut rs, &[child], &one(vec![0], 10..11, GroupFold::None), None)
        .unwrap();
    let lane = lanes_of(&out, &shared, &[(10, 1)], 6).remove(0);
    valid(&lane, &facts);
    let [[run, cow]] = record(&lane)[..] else {
        panic!("one piece")
    };
    assert_eq!(run, cow, "buffering copies no ring");
    let mine = table(&lane);
    assert_eq!(mine[0], theirs[0], "a full slot stays shared");
    assert_ne!(mine[1], theirs[1], "the tail is the child's copy");
    assert_eq!(copies(&lane), vec![[theirs[1], mine[1]]]);
    settle(&mut rs, out.txn);

    let out = prepare_many(&mut rs, &[child], &one(vec![0], 11..12, GroupFold::Host(5)), None)
        .unwrap();
    let lane = lanes_of(&out, &shared, &[(11, 1)], 6).remove(0);
    valid(&lane, &facts);
    let [[first_run, first_cow]] = record(&lane)[..] else {
        panic!("one piece")
    };
    assert_eq!(first_run, run, "it folds from the shared ring");
    assert_ne!(first_cow, first_run, "into a copy of its own");
    settle(&mut rs, out.txn);
    assert_eq!(rs.folded_slot(parent).unwrap(), held, "the parent's ring as it was");
    assert_eq!(rs.frozen(parent), Ok(6));
    assert_eq!(rs.frozen(child), Ok(11));

    // Its own now: the next fold writes in place.
    let out = prepare_many(&mut rs, &[child], &one(vec![0], 12..13, GroupFold::Host(2)), None)
        .unwrap();
    let lane = lanes_of(&out, &shared, &[(12, 1)], 11).remove(0);
    let [[again_run, again_cow]] = record(&lane)[..] else {
        panic!("one piece")
    };
    assert_eq!((again_run, again_cow), (first_cow, first_cow));
    settle(&mut rs, out.txn);
    for ws in [parent, child] {
        rs.release_working_set(ws);
    }
    rs.retire_idle();
    assert_eq!(wb_live(&shared), 0);
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

/// A snapshot of a windowed set is its record and its unfolded buffer
/// slots, as an rs set's (no window table): a checkpoint shares both, a
/// resume reads them as they were, writes into the partly filled one after
/// copying it, and copies the record at its first fold; an export names
/// the record and the buffer, no window list.
#[test]
fn a_windowed_snapshot_is_its_record_and_unfolded_buffer_slots() {
    let facts = windowed_facts();
    let (shared, stores) = stores_of(&facts);
    let mut rs = stores.rs.lock().unwrap();
    let ws = warm(&mut rs, 6);
    let out = prepare_many(&mut rs, &[ws], &one(vec![0], 6..10, GroupFold::None), None).unwrap();
    let theirs = table(&lanes_of(&out, &shared, &[(6, 4)], 6)[0]);
    settle(&mut rs, out.txn);
    let ring = rs.folded_slot(ws).unwrap();

    let export = rs.export(ws).unwrap();
    assert_eq!(export.folded, ring);
    assert_eq!(export.buffer.iter().flatten().count(), 2);
    assert_eq!((export.base, export.frozen), (4, 6));
    assert!(export.windows.is_empty() && export.window.is_none(), "no window table");

    assert!(rs.checkpoint(ws, [7; 32], None).unwrap());
    rs.release_working_set(ws);
    rs.retire_idle();
    assert_eq!(wb_live(&shared), 2, "the checkpoint holds the buffer");

    let resumed = rs.create_working_set(geom());
    assert!(rs.adopt_checkpoint(resumed, &[7; 32]).unwrap());
    assert_eq!(rs.frozen(resumed), Ok(6));
    assert_eq!(rs.folded_slot(resumed).unwrap(), ring, "the ring shared");
    let out = prepare_many(&mut rs, &[resumed], &one(vec![0], 10..12, GroupFold::Host(4)), None)
        .unwrap();
    let lane = lanes_of(&out, &shared, &[(10, 2)], 6).remove(0);
    valid(&lane, &facts);
    let mine = table(&lane);
    assert_eq!(mine[0], theirs[0], "the checkpoint's rows read as they were");
    assert_eq!(copies(&lane), vec![[theirs[1], mine[1]]], "its tail copied first");
    let [[run, cow]] = record(&lane)[..] else {
        panic!("one piece")
    };
    assert_ne!(run, cow, "the shared ring copied at its first fold");
    settle(&mut rs, out.txn);
    assert_eq!(rs.frozen(resumed), Ok(10));

    rs.release_working_set(resumed);
    rs.retire_idle();
    assert!(rs.reclaim_cache(|_| 0) > 0, "the checkpoint's slots go");
    rs.retire_idle();
    assert_eq!(wb_live(&shared), 0);
}

/// A windowed set parks whole and comes back: its record (the rings) and
/// its buffer slots each to a host slot of its own shape, and back.
#[test]
fn a_windowed_set_parks_its_record_and_buffer_on_the_host() {
    let mut facts = windowed_facts();
    facts.host_chunks = 8;
    let tiers = Tiers::of(&facts).unwrap();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let mut rs = RsStore::shared(&shared, tiers.host.as_ref());
    let ws = warm(&mut rs, 6);
    let out = prepare_many(&mut rs, &[ws], &one(vec![0], 6..10, GroupFold::None), None).unwrap();
    settle(&mut rs, out.txn);
    let sets = HashSet::from([ws]);
    let held = rs.suspendable(&sets);
    assert_eq!(held, RsUnits::of(1, 2), "the record and two buffer slots");
    let txn = rs.prepare_suspend(&sets).expect("the host takes it");
    let moved: Vec<EngineShape> = txn
        .copy_plan()
        .shared
        .iter()
        .map(|(shape, _)| *shape)
        .collect();
    assert!(moved.contains(&CONV) && moved.contains(&WB), "{moved:?}");
    assert_eq!(rs.commit_suspend(txn), held);
    assert_eq!(wb_live(&shared), 0, "its buffer slots came free");
    let mut grant = rs.reserve(rs.swapped_units(&sets)).expect("room to come back");
    let txn = rs.prepare_restore(&sets, &mut grant).expect("it comes back");
    rs.commit_restore(txn);
    assert_eq!(rs.suspendable(&sets), held, "on the device again");
    assert_eq!(rs.frozen(ws), Ok(6));
}

/// A stop that drops a lane's unfolded rows (the next fire writes from
/// inside them) leaves the ring untouched: no record is written, the set
/// stays folded where it was, and the dropped rows' slots are written over
/// in place.
#[test]
fn a_stop_dropping_unfolded_rows_leaves_the_ring_untouched() {
    let facts = windowed_facts();
    let (shared, stores) = stores_of(&facts);
    let mut rs = stores.rs.lock().unwrap();
    let ws = warm(&mut rs, 6);
    let out = prepare_many(&mut rs, &[ws], &one(vec![0], 6..13, GroupFold::None), None).unwrap();
    let before = table(&lanes_of(&out, &shared, &[(6, 7)], 6)[0]);
    settle(&mut rs, out.txn);
    let ring = rs.folded_slot(ws).unwrap();

    // Stopped at 9: rows 9..13 are dropped, the next row lands at 9.
    let out = prepare_many(&mut rs, &[ws], &one(vec![0], 9..10, GroupFold::None), None).unwrap();
    let lane = lanes_of(&out, &shared, &[(9, 1)], 6).remove(0);
    valid(&lane, &facts);
    let [[run, cow]] = record(&lane)[..] else {
        panic!("one piece")
    };
    assert_eq!(run, cow, "no ring written");
    assert_eq!(table(&lane), before, "written over in place");
    assert!(copies(&lane).is_empty());
    settle(&mut rs, out.txn);
    assert_eq!(rs.folded_slot(ws).unwrap(), ring, "the ring as it was");
    assert_eq!(rs.frozen(ws), Ok(6));

    // Accepted through 10: the fold reads rows 6..10 and the ring moves.
    let out = prepare_many(&mut rs, &[ws], &one(vec![0], 10..11, GroupFold::Host(4)), None)
        .unwrap();
    settle(&mut rs, out.txn);
    assert_eq!(rs.frozen(ws), Ok(10));
    rs.release_working_set(ws);
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

/// The stores read a window buffer as the rs store's buffer, never as a
/// second kv list, and state no window list for it; beside an rs list (a
/// hybrid) the rs list stays the buffer and the window buffer is a pool of
/// its own.
#[test]
fn the_window_buffer_is_the_buffer_or_a_pool_beside_the_rs_list() {
    let facts = windowed_facts();
    let pools = crate::store::chunks::StorePools::of(&facts).unwrap();
    assert!(pools.shared);
    assert_eq!(pools.page_size, 16, "the kv list's tokens, not the buffer's");
    assert_eq!(pools.buffer.slot_bytes, MIB / 4, "the window buffer's slots");
    assert_eq!(pools.window_tokens, 0, "no window list");

    let facts = hybrid_facts();
    let pools = crate::store::chunks::StorePools::of(&facts).unwrap();
    assert_eq!(pools.buffer.slot_bytes, MIB / 8, "the rs list's slots");
    let pool = SharedPool::new(&facts, 256).unwrap();
    assert_eq!(pool.rs_list().map(|list| list.id), Some(RS));
    assert_eq!(pool.window_buffer().map(|list| list.id), Some(WB));
    assert_eq!(pool.buffer_list().map(|list| list.id), Some(RS));
}

/// A lone lane carrying no buffered row that folds part of what it writes
/// (a prefill accepted short of its end) buffers only the rows past its
/// fold: the rows it folds itself take no slot, and its table starts at the
/// folded index after the fire, rounded down to the slot.
#[test]
fn a_fire_s_own_folded_rows_take_no_buffer_slot() {
    let facts = windowed_facts();
    let (shared, stores) = stores_of(&facts);
    let mut rs = stores.rs.lock().unwrap();
    let ws = warm(&mut rs, 6);
    let plan = one(vec![0], 6..30, GroupFold::Host(20));
    assert_eq!(
        demand(&rs, &[ws], &plan).map(|units| units.pages),
        Ok(2),
        "rows 26..30 in the slots of 24..28 and 28..32"
    );
    let out = prepare_many(&mut rs, &[ws], &plan, None).unwrap();
    let lane = lanes_of(&out, &shared, &[(6, 24)], 6).remove(0);
    valid(&lane, &facts);
    assert_eq!(
        lane.window,
        Some(WindowFold {
            base: 24,
            fold: FoldLen::Host(20),
            lineage: Lineage::Set,
        })
    );
    assert_eq!(table(&lane).len(), 2);
    settle(&mut rs, out.txn);
    assert_eq!((rs.frozen(ws), rs.base(ws)), (Ok(26), Ok(24)));
    assert_eq!(wb_live(&shared), 2);

    // The next fire carries rows 26..30: its table starts below them.
    let out = prepare_many(&mut rs, &[ws], &one(vec![0], 30..31, GroupFold::Host(5)), None)
        .unwrap();
    let lane = lanes_of(&out, &shared, &[(30, 1)], 26).remove(0);
    valid(&lane, &facts);
    assert_eq!(lane.window.map(|window| window.base), Some(24));
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert_eq!((rs.frozen(ws), rs.base(ws)), (Ok(31), Ok(28)));
    rs.release_working_set(ws);
}

/// A record is one unit over every shape it spans (the rings in one fixed
/// shape, other state in another, in pieces): a lane names each shape's
/// pieces, a fold copies them all, and a park moves them all.
#[test]
fn a_windowed_record_is_one_unit_over_its_shapes() {
    let mut facts = windowed_facts();
    facts.shapes.insert(
        1,
        shape(GDN, SlotKind::Fixed { pieces: 3 }, 4 * MIB),
    );
    facts.host_chunks = 8;
    facts.validate().unwrap();
    let tiers = Tiers::of(&facts).unwrap();
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let mut rs = RsStore::shared(&shared, tiers.host.as_ref());
    let ws = warm(&mut rs, 6);
    let child = rs.fork(ws).unwrap();
    let out = prepare_many(&mut rs, &[child], &one(vec![0], 6..7, GroupFold::Host(1)), None)
        .unwrap();
    let lane = lanes_of(&out, &shared, &[(6, 1)], 6).remove(0);
    let ::engine::LaneRecord::Pieces(pieces) = &lane.record else {
        panic!("a record")
    };
    let shapes: Vec<(EngineShape, usize)> =
        pieces.iter().map(|(shape, pairs)| (*shape, pairs.len())).collect();
    assert_eq!(shapes, vec![(GDN, 3), (CONV, 1)], "every shape's pieces");
    assert!(
        pieces.iter().flat_map(|(_, pairs)| pairs).all(|[run, cow]| run != cow),
        "the first fold copies the whole record"
    );
    settle(&mut rs, out.txn);
    let sets = HashSet::from([ws]);
    let txn = rs.prepare_suspend(&sets).expect("the host takes it");
    let moved: Vec<(EngineShape, usize)> = txn
        .copy_plan()
        .shared
        .iter()
        .map(|(shape, pairs)| (*shape, pairs.len()))
        .collect();
    assert!(moved.contains(&(GDN, 3)) && moved.contains(&(CONV, 1)), "{moved:?}");
    rs.abort_suspend(txn);
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
    facts.validate().unwrap();
    facts
}

/// The slots of engine shape `id` the pool holds now.
fn live_of(shared: &Shared, id: EngineShape) -> u64 {
    let pool = shared.lock().unwrap();
    let shape = pool.pool_shape(id).expect("a shape of the pool");
    pool.pool().shape_demand(shape).live
}

/// A hybrid's lane the engine takes: its recurrent verb and its window
/// fold both stated, one fold.
fn hybrid_valid(lane: &::engine::Lane, facts: &ChunkFacts) {
    let serves = Serves {
        windowed_state: true,
        rs_verbs: true,
        chunks: true,
        ..Serves::NONE
    };
    lane.validate_for(serves).unwrap();
    lane.validate_chunks(facts).unwrap();
}

/// The lane's rs list pages, base and folded index.
fn rs_buffer(lane: &::engine::Lane) -> (Vec<u32>, u32, u32) {
    match &lane.rs {
        ::engine::fire::RsVerb::Buffer {
            pages,
            base,
            folded,
            ..
        } => (pages.clone(), *base, *folded),
        ::engine::fire::RsVerb::Fold => (Vec::new(), 0, 0),
    }
}

/// A hybrid buffers its recurrent rows in its rs list and its windowed
/// layers' rows in its window buffer, two slot lists of their own sizes and
/// bases sharing one folded index and one fold: a fold frees what each
/// holds wholly below it; the rows a lone lane folds itself take no window
/// slot (the rs list buffers them all, as an rs load does); its record is
/// the ring shape's pieces first, then the state's.
#[test]
fn a_hybrid_buffers_rs_and_window_rows_apart_and_folds_them_once() {
    let facts = hybrid_facts();
    let (shared, stores) = stores_of(&facts);
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(RsGeometry {
        state_size: facts.record_bytes,
        buffer_page_tokens: 16,
    });
    let out = prepare(&mut rs, ws, 6).unwrap();
    let lane = lanes_of(&out, &shared, &[(0, 6)], 0).remove(0);
    hybrid_valid(&lane, &facts);
    let ::engine::LaneRecord::Pieces(pieces) = &lane.record else {
        panic!("a record")
    };
    let shapes: Vec<EngineShape> = pieces.iter().map(|(shape, _)| *shape).collect();
    assert_eq!(shapes, vec![RINGS, CONV], "the ring's pieces first");
    assert_eq!(lane.window.map(|window| window.base), Some(0));
    settle(&mut rs, out.txn);

    // Rows 6..13 unfolded: rs slots of 4..8, 8..12, 12..16; window slots of
    // 0..8, 8..16.
    let plan = one(vec![0], 6..13, GroupFold::None);
    assert_eq!(
        demand(&rs, &[ws], &plan),
        Ok(RsUnits {
            pages: 5,
            wbuf: 2,
            ..RsUnits::default()
        })
    );
    let out = prepare_many(&mut rs, &[ws], &plan, None).unwrap();
    let lane = lanes_of(&out, &shared, &[(6, 7)], 6).remove(0);
    hybrid_valid(&lane, &facts);
    let (pages, base, folded) = rs_buffer(&lane);
    assert_eq!((pages.len(), base, folded), (3, 4, 6));
    assert_eq!(
        lane.window,
        Some(WindowFold {
            base: 0,
            fold: FoldLen::Host(0),
            lineage: Lineage::Set,
        })
    );
    assert_eq!(table(&lane).len(), 2, "the window rows in their own list");
    settle(&mut rs, out.txn);
    assert_eq!((live_of(&shared, RS), live_of(&shared, WB)), (3, 2));

    // One fold through 12: each list gives back what lies wholly below.
    let out = prepare_many(&mut rs, &[ws], &one(vec![0], 13..14, GroupFold::Host(6)), None)
        .unwrap();
    let lane = lanes_of(&out, &shared, &[(13, 1)], 6).remove(0);
    hybrid_valid(&lane, &facts);
    assert_eq!(lane.window.map(|window| window.fold), Some(FoldLen::Host(6)));
    settle(&mut rs, out.txn);
    rs.retire_idle();
    assert_eq!(rs.frozen(ws), Ok(12));
    assert_eq!((live_of(&shared, RS), live_of(&shared, WB)), (1, 1));

    // A lone lane folding part of what it writes: the rs list buffers it
    // all; the window buffer only the rows past the fold.
    let fresh = rs.create_working_set(RsGeometry {
        state_size: facts.record_bytes,
        buffer_page_tokens: 16,
    });
    let out = prepare(&mut rs, fresh, 6).unwrap();
    settle(&mut rs, out.txn);
    let plan = one(vec![0], 6..30, GroupFold::Host(20));
    assert_eq!(
        demand(&rs, &[fresh], &plan).map(|units| (units.pages - units.wbuf, units.wbuf)),
        Ok((7, 1)),
        "rs 4..32; window 24..32"
    );
    let out = prepare_many(&mut rs, &[fresh], &plan, None).unwrap();
    let lane = lanes_of(&out, &shared, &[(6, 24)], 6).remove(0);
    hybrid_valid(&lane, &facts);
    assert_eq!(lane.window.map(|window| window.base), Some(24));
    assert_eq!(table(&lane).len(), 1);
    settle(&mut rs, out.txn);

    for set in [ws, fresh] {
        rs.release_working_set(set);
    }
    rs.retire_idle();
    assert_eq!((live_of(&shared, RS), live_of(&shared, WB)), (0, 0));
}

/// A hybrid's fork shares both lists: a write into a shared partly filled
/// slot of either copies it first, each under its own shape; a beam group
/// on one set names one record, one rs buffer and one window table.
#[test]
fn a_hybrid_fork_copies_each_list_s_shared_tail_and_a_beam_shares_both() {
    let facts = hybrid_facts();
    let (shared, stores) = stores_of(&facts);
    let mut rs = stores.rs.lock().unwrap();
    let geom = RsGeometry {
        state_size: facts.record_bytes,
        buffer_page_tokens: 16,
    };
    let parent = rs.create_working_set(geom);
    let out = prepare(&mut rs, parent, 6).unwrap();
    settle(&mut rs, out.txn);
    let out = prepare_many(&mut rs, &[parent], &one(vec![0], 6..10, GroupFold::None), None)
        .unwrap();
    let theirs = lanes_of(&out, &shared, &[(6, 4)], 6).remove(0);
    settle(&mut rs, out.txn);
    let child = rs.fork(parent).unwrap();
    let out = prepare_many(&mut rs, &[child], &one(vec![0], 10..11, GroupFold::None), None)
        .unwrap();
    let lane = lanes_of(&out, &shared, &[(10, 1)], 6).remove(0);
    hybrid_valid(&lane, &facts);
    let pairs = |id: EngineShape| -> Vec<[u32; 2]> {
        lane.slot_copies
            .iter()
            .find(|(shape, _)| *shape == id)
            .map(|(_, pairs)| pairs.clone())
            .unwrap_or_default()
    };
    // rs: 8..12 holds 8..10, shared; window: 8..16 holds 8..10, shared.
    assert_eq!(pairs(RS), vec![[rs_buffer(&theirs).0[1], rs_buffer(&lane).0[1]]]);
    assert_eq!(pairs(WB), vec![[table(&theirs)[1], table(&lane)[1]]]);
    assert_eq!(table(&lane)[0], table(&theirs)[0], "a full slot stays shared");
    settle(&mut rs, out.txn);

    // Beams on the parent.
    let plan = one(vec![0, 1], 10..14, GroupFold::None);
    let out = prepare_many(&mut rs, &[parent; 2], &plan, None).unwrap();
    let lanes = lanes_of(&out, &shared, &[(10, 2), (12, 2)], 6);
    for lane in &lanes {
        hybrid_valid(lane, &facts);
        assert_eq!(table(lane), table(&lanes[0]));
        assert_eq!(rs_buffer(lane), rs_buffer(&lanes[0]));
        assert_eq!(record(lane), record(&lanes[0]));
        assert_eq!(lane.window.map(|window| window.lineage), Some(Lineage::Admitted));
    }
    settle(&mut rs, out.txn);
    for set in [parent, child] {
        rs.release_working_set(set);
    }
    rs.retire_idle();
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

/// A model whose state is its windowed layers alone states only the ring
/// shape: its record is the ring's pieces.
#[test]
fn a_kv_only_window_model_s_record_is_its_ring() {
    let mut facts = windowed_facts();
    facts.shapes.retain(|shape| shape.id != CONV);
    facts.shapes.insert(
        1,
        Facts {
            role: SlotRole::Ring,
            ..shape(RINGS, SlotKind::Fixed { pieces: 1 }, CHUNK)
        },
    );
    facts.validate().unwrap();
    let (shared, stores) = stores_of(&facts);
    let mut rs = stores.rs.lock().unwrap();
    let ws = warm(&mut rs, 6);
    let out = prepare_many(&mut rs, &[ws], &one(vec![0], 6..7, GroupFold::Host(1)), None)
        .unwrap();
    let lane = lanes_of(&out, &shared, &[(6, 1)], 6).remove(0);
    valid(&lane, &facts);
    let ::engine::LaneRecord::Pieces(pieces) = &lane.record else {
        panic!("a record")
    };
    assert_eq!(pieces.len(), 1);
    assert_eq!(pieces[0].0, RINGS);
    settle(&mut rs, out.txn);
}

/// A beam group's long prompt on a windowed-only load (four beams of 792
/// rows sharing one record, its lagged fold stated, wider together than the 2897 a fire holds) is cut as kv lanes
/// are: its lanes replay nothing, so several lanes are cuttable
/// ([`crate::pipeline::fire::window_only`]); a hybrid's are not, its
/// recurrent layers keeping their bound. The cut run buffers the same rows
/// in the same slots' worth, folds to the same index and leaves the record
/// where the uncut run does: what the engine reads is the same rows either
/// way.
#[test]
fn a_beam_group_s_long_prompt_on_a_windowed_load_is_cut_not_refused() {
    const BEAMS: usize = 4;
    const ROWS: u32 = 792;
    const HOLDS: usize = 2897;
    let facts = windowed_facts();
    let (_shared, stores) = stores_of(&facts);
    assert!(crate::pipeline::fire::window_only(&stores, true));
    let (_, hybrid) = stores_of(&hybrid_facts());
    assert!(!crate::pipeline::fire::window_only(&hybrid, true), "a hybrid keeps the rs bound");
    // An engine stating an rs list beside a record of rings alone: no
    // recurrent layer replays, so its lanes are cut too.
    let mut listed = windowed_facts();
    listed.shapes.push(Facts {
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
    listed.validate().unwrap();
    let (_, listed) = stores_of(&listed);
    assert!(crate::pipeline::fire::window_only(&listed, true));

    // The pass, cut as the host pass cuts a window-only load's lanes.
    let req = crate::engine::FireRequest {
        lanes: (0..BEAMS)
            .map(|at| {
                let mut lane = crate::engine::fire::lane_of(
                    at as u32,
                    vec![7; ROWS as usize],
                    5 + at as u32 * ROWS,
                    Vec::new(),
                );
                lane.lane.readout = ::engine::Readout::Last;
                lane
            })
            .collect(),
        boundary_program: true,
        ..crate::engine::FireRequest::default()
    };
    let pieces = crate::pipeline::fire::pass_pieces(&req, true, true, 1024, 16, &[], &[]);
    assert!(pieces.len() > 1, "the pass is cut");
    for piece in &pieces {
        let rows: u32 = piece.iter().flatten().map(|(from, to)| to - from).sum();
        assert!(rows as usize <= HOLDS, "a piece of {rows} rows fits a fire");
    }

    // Uncut and cut, the set ends holding the same rows.
    let run = |cut: bool| {
        let (shared, stores) = stores_of(&facts);
        let mut rs = stores.rs.lock().unwrap();
        let ws = warm(&mut rs, 5);
        let spans: Vec<Vec<(u32, u32)>> = if cut {
            pieces
                .iter()
                .map(|piece| {
                    piece
                        .iter()
                        .enumerate()
                        .filter_map(|(at, cut)| {
                            cut.map(|(from, to)| {
                                let held = 5 + at as u32 * ROWS;
                                (held + from, held + to)
                            })
                        })
                        .collect()
                })
                .collect()
        } else {
            vec![(0..BEAMS as u32).map(|at| (5 + at * ROWS, 5 + (at + 1) * ROWS)).collect()]
        };
        let count = spans.len();
        for (at, spans) in spans.into_iter().enumerate() {
            // The group's lagged common-ancestor fold lands with the last
            // piece; the pieces before it buffer their rows.
            let fold = if at + 1 == count {
                GroupFold::Host(400)
            } else {
                GroupFold::None
            };
            let low = spans.iter().map(|&(from, _)| from).min().unwrap();
            let high = spans.iter().map(|&(_, to)| to).max().unwrap();
            let lanes: Vec<usize> = (0..spans.len()).collect();
            let ids = vec![ws; spans.len()];
            let plan = one(lanes, low..high, fold);
            demand(&rs, &ids, &plan).expect("counted, never refused");
            let out = prepare_many(&mut rs, &ids, &plan, None).unwrap();
            let rows: Vec<(u32, u32)> = spans.iter().map(|&(from, to)| (from, to - from)).collect();
            let lanes = lanes_of(&out, &shared, &rows, 5);
            for lane in &lanes {
                valid(lane, &facts);
                assert_eq!(record(lane), record(&lanes[0]), "the group's one record");
            }
            settle(&mut rs, out.txn);
        }
        let held = (
            rs.frozen(ws),
            rs.base(ws),
            rs.buffer_size(ws),
            rs.folded_slot(ws).map(|slot| slot.is_some()),
            wb_live(&shared),
        );
        rs.release_working_set(ws);
        held
    };
    assert_eq!(run(true), run(false), "the same rows, folded the same");
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

/// A masked windowed lane folding nothing (a beam inferlet's Fold::None):
/// the lowering to ring + buffer leaves the lane's folded index and its
/// mask as staged, so the mask's column 0 is the folded index the engine
/// reads from (`kv.folded`, the kv set's, which the inferlet's
/// `WorkingSet::folded` reads too). The window buffer's base is a table
/// origin at or below it, never the mask's origin; the buffered rows from
/// the folded index on are the mask's columns.
#[test]
fn a_masked_lane_folding_nothing_keeps_its_mask_origin_at_the_folded_index() {
    let facts = windowed_facts();
    let (shared, stores) = stores_of(&facts);
    let mut rs = stores.rs.lock().unwrap();
    let ws = warm(&mut rs, 6);
    let out = prepare_many(&mut rs, &[ws], &one(vec![0], 6..13, GroupFold::None), None).unwrap();
    settle(&mut rs, out.txn);
    // Rows 13.. on, the lane carrying 6..13 unfolded: a mask of the 8
    // columns from the folded index (6) through its own row.
    let out = prepare_many(&mut rs, &[ws], &one(vec![0], 13..14, GroupFold::None), None).unwrap();
    let mask = ::engine::Masking::Extent(::engine::Mask::new(vec![8], 8));
    let mut req = crate::engine::FireRequest {
        lanes: vec![{
            let mut lane = crate::engine::fire::lane_of(0, vec![1], 13, Vec::new());
            lane.lane.kv.folded = 6;
            lane.lane.mask = Some(mask.clone());
            lane
        }],
        ..crate::engine::FireRequest::default()
    };
    out.apply_to(&mut req);
    crate::pipeline::fire::rs::apply_shared(&mut req, Some(&shared), Some(&out));
    let lane = &req.lanes[0].lane;
    assert_eq!(lane.kv.folded, 6, "the folded index as staged");
    assert_eq!(lane.mask, Some(mask), "the mask as staged, column 0 at the folded index");
    let window = lane.window.unwrap();
    assert_eq!(window.fold, FoldLen::Host(0));
    assert_eq!(window.base, 4, "the table's origin, at or below the mask's");
    assert_eq!(rs.frozen(ws), Ok(6), "the store folds nothing either");
    valid(lane, &facts);
    settle(&mut rs, out.txn);
}
