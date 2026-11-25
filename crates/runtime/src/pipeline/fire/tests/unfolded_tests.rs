use super::*;
use crate::inferlet::host::working_set::Cursor;

/// At the contract's floor (a rung stating nothing holds its tokens of
/// own rows and as many carried) no rung's replay scratch holds more
/// carried rows than the largest rung's tokens: a fire carrying more is
/// refused by name, naming the bound, before anything is submitted;
/// own rows past it are another refusal's (the capacity of a fire).
#[test]
fn a_fire_carrying_past_the_largest_rung_is_refused_by_name() {
    let (_, stores) = pool_with_rungs_of(vec![(256, 2), (2048, 8)]);
    let carrying = |carried| FireRows { own: 5, carried };
    assert_eq!(replay_rung(&stores, 56, carrying(0), 0, 1, false), Ok(5));
    assert_eq!(replay_rung(&stores, 56, carrying(2048), 2048, 1, false), Ok(2048));
    let host = replay_rung(&stores, 56, carrying(2049), 2049, 1, false).unwrap_err();
    assert!(
        host.contains("of 2048 tokens")
            && host.contains("2049 carried recurrent rows")
            && host.contains("fold what its lanes have written before they write more"),
        "{host}"
    );
    assert!(replay_rung(&stores, 56, FireRows { own: 4096, carried: 0 }, 0, 1, false).is_ok());
}

/// A host pass whose uncut own rows and carried rows together pass the
/// widest rung's replay scratch, but whose pieces each fit, is served in
/// pieces: the whole-pass check counts no own row, each piece its own
/// as it is cut. A piece that no cut brings under a rung is refused by
/// name.
#[test]
fn a_pass_too_wide_uncut_is_served_in_pieces_that_fit() {
    let metal = |tokens: u32| ::engine::RungReplay {
        own_max: Some(tokens),
        carried_max: Some(tokens),
        scratch: rows_bytes(2 * tokens),
    };
    let (_, stores) = pool_stating(vec![(256, 2), (1024, 8)], vec![metal(256), metal(1024)], false);
    // Folded through 100, 300 rows buffered since, 2000 more to write
    // folding all: uncut, 2000 own and 300 carried pass 2048.
    let full = lane(400, 2000);
    let frozen = [Some(100)];
    let whole = vec![Some((0, 2000))];
    assert!(
        replay_rung(
            &stores,
            56,
            piece_rows(&full, &whole, &frozen, false),
            piece_replayed(&full, &whole, &frozen, false),
            1,
            false
        )
        .is_ok(),
        "own rows past every rung are the cut's, not a refusal"
    );
    let cuts: Vec<Vec<Option<(u32, u32)>>> = host_cuts(&full, true, 512, 16, &[], 0)
        .into_iter()
        .map(|cut| vec![Some(cut)])
        .collect();
    assert!(cuts.len() > 1, "the pass is cut");
    assert_eq!(
        pieces_rung(&stores, 56, &full, &cuts, &frozen, true, &no_scratch()),
        Ok(512),
        "every piece fits: the widest asks its 512 rows, the 1024 rung, the first \
         carrying the 300 rows beside them"
    );
    // A piece of eleven lanes, 929 own rows, replaying 1144: refused by
    // name, whatever its lane count.
    let mut eleven = lane(1244, 84);
    for _ in 1..11 {
        let more = eleven.lanes[0].clone();
        eleven.lanes.push(more);
    }
    eleven.lanes[0].tokens = vec![0; 89];
    let frozen = [Some(1140); 11];
    let one = vec![(0..11).map(|at| Some((0, if at == 0 { 89 } else { 84 }))).collect()];
    let refused =
        pieces_rung(&stores, 56, &eleven, &one, &frozen, false, &no_scratch()).unwrap_err();
    assert!(refused.contains("of 1024 tokens"), "{refused}");
}

/// A pass of lanes alone on their sets, folding nothing, landing no
/// snapshot.
fn no_scratch() -> PassScratch<'static> {
    PassScratch {
        shared: Vec::new(),
        checkpoints: &[],
        snapshots: 0,
    }
}

pub(super) fn pool_of_one_row_rs_pages() -> crate::store::registry::Stores {
    pool_with_rungs(Vec::new())
}

/// That pool with activation rungs `act_chunks` (tokens, chunks), its
/// span backed from the load: it starts at the largest.
pub(super) fn pool_with_rungs(act_chunks: Vec<(u32, u32)>) -> crate::store::registry::Stores {
    pool_with_rungs_of(act_chunks).1
}

/// [`pool_with_rungs`], with the model it is registered as.
pub(super) fn pool_with_rungs_of(
    act_chunks: Vec<(u32, u32)>,
) -> (usize, crate::store::registry::Stores) {
    pool_stating(act_chunks, Vec::new(), false)
}

/// [`pool_with_rungs_of`], each rung's replay scratch stated
/// (`act_replay`) and the load's rs scratch `rs_compact` or not.
/// A Metal-shaped test scratch: one byte a row, no lane term past its
/// 4-byte floor (`ScratchBytes::bytes`, Metal's formula), holding `rows`
/// scratch rows.
pub(super) fn rows_bytes(rows: u32) -> ::engine::ScratchBytes {
    ::engine::ScratchBytes {
        slot_bytes: vec![1],
        lane_bytes: 0,
        room_bytes: u64::from(rows) + 4,
        pad: 1,
    }
}

pub(super) fn pool_stating(
    act_chunks: Vec<(u32, u32)>,
    act_replay: Vec<::engine::RungReplay>,
    rs_compact: bool,
) -> (usize, crate::store::registry::Stores) {
    let act_replay = if act_replay.is_empty() {
        act_chunks
            .iter()
            .map(|&(tokens, _)| crate::store::shared::rows_only(tokens))
            .collect()
    } else {
        act_replay
    };
    use ::engine::chunk::{
        ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId, SlotKind, SlotRole,
        SlotShape,
    };
    let shape = |id, kind: SlotKind, role| SlotShape {
        id: ShapeId(id),
        kind,
        slot_bytes: 1 << 20,
        slots_per_chunk: 1,
        sets_per_working_set: kind.sets_per_working_set(),
        rows: Vec::new(),
        pinned: None,
        role,
    };
    let list = |tokens| SlotKind::List {
        tokens,
        rate: 1,
        window: None,
    };
    let facts = ChunkFacts {
        chunk_bytes: 1 << 20,
        total_chunks: 16,
        segment_chunks: 16,
        segments: SegmentFacts {
            committed: 1,
            max: 1,
            growth: Growth::Fixed,
        },
        reserved: vec![ReservedRun {
            what: Reserved::Null,
            first: 0,
            chunks: 1,
        }],
        shapes: vec![
            shape(0, list(16), SlotRole::State),
            // One recurrent row an rs page, as Qwen3.5's.
            shape(1, list(1), SlotRole::RsBuffer),
            shape(2, SlotKind::Fixed { pieces: 1 }, SlotRole::State),
        ],
        act_chunks,
        act_replay,
        rs_compact,
        rs_scratch_per_fire: true,
        ..ChunkFacts::default()
    };
    let pool = crate::store::shared::SharedPool::new(&facts, 1024).unwrap();
    let model = crate::store::registry::register_model_with_chunks(
        16,
        &[crate::store::registry::EngineStores {
            pools: crate::store::chunks::StorePools::of(&facts).unwrap(),
            max_context: 1024,
            shared: Some(Arc::new(Mutex::new(pool))),
            tiers: crate::store::shared::Tiers::default(),
        }],
    );
    (model, crate::store::registry::get(model, 0))
}

fn lane(held: u32, rows: u32) -> crate::engine::FireRequest {
    crate::engine::FireRequest {
        lanes: vec![
            ::engine::Lane {
                tokens: vec![0; rows as usize],
                kv: ::engine::KvDelta {
                    held,
                    ..Default::default()
                },
                ..::engine::Lane::default()
            }
            .into(),
        ],
        ..crate::engine::FireRequest::default()
    }
}

fn groups(plan: rs::RsPlan) -> Vec<rs::GroupPlan> {
    match plan {
        rs::RsPlan::Groups(groups) => groups,
        rs::RsPlan::Window(_) => panic!("a plain recurrent model plans groups"),
    }
}

/// A decode run ahead four fires on two recurrent sets, the host's read
/// of every output still to come: one pass stated its rows final (a
/// default pass: the SDK states it, CONTRACT-BC 25), the other did not
/// (a pass the program declared retractable). The final
/// lane folds its row at each fire and carries none into the next: its
/// verb is `Fold` and its step alone is classed folded. The lagged
/// lane folds nothing it has not read: it carries one more row each
/// fire and buffers. A step of both is buffered, each lane by its own
/// verb: the final lane's still `Fold`, carrying nothing.
#[test]
fn a_final_lane_folds_at_fire_and_a_lagged_one_beside_it_carries() {
    use ::engine::fire::RsVerb;
    use ::engine::shapes::{StepComposition, Treatment};
    let stores = pool_of_one_row_rs_pages();
    let ring = Arc::new(Mutex::new(crate::pipeline::channel::ChannelCell::new(
        vec![1],
        eta_ir::types::Dtype::I32,
        8,
    )));
    let lane_of = |final_rows: bool| {
        let kv = stores.kv.lock().create_working_set();
        let ws = stores
            .rs
            .lock()
            .unwrap()
            .create_working_set(crate::store::rs::RsGeometry {
                state_size: 1 << 10,
                buffer_page_tokens: 1,
            });
        // The prompt is folded through 10.
        stores.rs.lock().unwrap().fold(ws, 10);
        (final_rows, kv, ws, Arc::new(Mutex::new(Cursor::default())))
    };
    let lanes = [lane_of(true), lane_of(false)];
    for step in 0..4u32 {
        let start = 10 + step;
        let mut fired = Vec::new();
        for (final_rows, kv, ws, cursor) in &lanes {
            let ids = [*ws];
            let groups_of = lane_groups(&[*kv]);
            let frozen = stores.rs.lock().unwrap().frozen(*ws).unwrap();
            let pass = PassLanes {
                places: vec![geometry::Place {
                    folded: frozen,
                    folded_exact: true,
                    tail: None,
                    shared: false,
                    spans: Vec::new(),
                    readable: start,
                    readable_known: true,
                    set: 1,
                }],
                cursors: vec![cursor.clone()],
                kv: Vec::new(),
            };
            let landings = [geometry::Landing { start, rows: 1 }];
            // Every decode states the folding geometry: all it writes.
            let fold = pass_fold(
                geometry::Fold::Host(vec![u32::MAX]),
                true,
                true,
                *final_rows,
                &pass,
                &groups_of,
                &landings,
            );
            let mut req = lane(start, 1);
            let plan = host_rs_plan(&stores, &ids, &groups_of, &req, &fold, &[], None, None)
                .unwrap();
            let frozen_at = lanes_frozen(&stores, &ids, &groups_of, 1, false).unwrap();
            let counted = piece_rows(&req, &[Some((0, 1))], &frozen_at, false);
            let mut store = stores.rs.lock().unwrap();
            let prepared = rs::prepare_many(&mut store, &ids, &plan, None).unwrap();
            prepared.apply_to(&mut req);
            rs::settle(&mut store, prepared.txn);
            drop(store);
            // What the fire's submit records: a lagged run-ahead fire's
            // rows wait for the host to take its output.
            if !*final_rows {
                cursor.lock().unwrap().run_ahead(
                    start,
                    start + 1,
                    vec![crate::inferlet::host::working_set::HostRead {
                        cell: Arc::downgrade(&ring),
                        index: u64::from(step),
                    }],
                );
            }
            let replayed = req.lanes[0]
                .lane
                .replay_stated
                .map_or(0, |stated| stated.replayed);
            if *final_rows {
                assert_eq!(frozen_at, [Some(start)], "step {step}: folded through its first row");
                assert_eq!(counted.carried, 0, "step {step}");
                assert_eq!(req.lanes[0].lane.rs, RsVerb::Fold, "step {step}");
                assert_eq!(replayed, 0, "step {step}");
                assert_eq!(
                    stores.rs.lock().unwrap().frozen(*ws).unwrap(),
                    start + 1,
                    "step {step}: its row folded at fire"
                );
                let alone = StepComposition::of([&req.lanes[0].lane]);
                assert_eq!((alone.treatment, alone.carried), (Treatment::Folds, 0));
            } else {
                assert_eq!(frozen_at, [Some(10)], "step {step}: nothing read, nothing folded");
                assert_eq!(replayed, step, "step {step}: every row before it carried");
                assert!(matches!(req.lanes[0].lane.rs, RsVerb::Buffer { .. }), "step {step}");
                let alone = StepComposition::of([&req.lanes[0].lane]);
                assert_eq!(alone.treatment, Treatment::Buffers);
            }
            fired.push(req.lanes[0].lane.clone());
        }
        let mixed = StepComposition::of(&fired);
        assert_eq!(mixed.treatment, Treatment::Buffers, "step {step}: a mixed step buffers");
        assert_eq!(mixed.decode_lanes, 2);
    }
}

/// MTP with k = 4 on a pool whose rs pages hold one row: the draft
/// writes a window of five rows unfolded, the next window carries those
/// five in and folds the accepted rows. Not one rs page: a lane's buffer
/// names several. No count of rows stops a lane that never folds: its
/// fire asks the rung for what it carries.
#[test]
fn mtp_k4_rides_one_row_rs_pages() {
    let stores = pool_of_one_row_rs_pages();
    let kv = stores.kv.lock().create_working_set();
    let ws = stores
        .rs
        .lock()
        .unwrap()
        .create_working_set(crate::store::rs::RsGeometry {
            state_size: 1 << 10,
            buffer_page_tokens: 1,
        });
    // The prompt is folded through 10.
    stores.rs.lock().unwrap().fold(ws, 10);
    let ids = [ws];
    let groups_of = lane_groups(&[kv]);
    let host = |held, rows, fold: Vec<u32>| {
        host_rs_plan(
            &stores,
            &ids,
            &groups_of,
            &lane(held, rows),
            &geometry::Fold::Host(fold),
            &[],
            None,
            None,
        )
    };

    // Draft: five rows (the truth and k = 4 drafts), nothing folded.
    let draft = groups(host(10, 5, vec![0]).expect("a draft of five rows"));
    assert_eq!(draft[0].writes, Some(10..15));
    assert_eq!(draft[0].fold, rs::GroupFold::None);

    // Verify: carries the draft's five in and folds the accepted three.
    let verify = groups(host(15, 5, vec![3]).expect("carries five in"));
    assert_eq!(verify[0].writes, Some(15..20));
    assert_eq!(verify[0].fold, rs::GroupFold::Host(3));
    stores.rs.lock().unwrap().fold(ws, 3);
    assert_eq!(stores.rs.lock().unwrap().frozen(ws).unwrap(), 13);

    // A lane that never folds piles up: past 13, carrying 100 in. Its
    // plan stands; the fire asks the rung for the hundred.
    let piled = groups(host(113, 5, vec![0]).expect("planned whatever it carries"));
    assert_eq!(piled[0].writes, Some(113..118));
    let frozen = lanes_frozen(&stores, &ids, &groups_of, 1, false).unwrap();
    assert_eq!(frozen, [Some(13)]);
    let counted = piece_rows(&lane(113, 5), &[Some((0, 5))], &frozen, false);
    assert_eq!(
        counted,
        FireRows {
            own: 5,
            carried: 100
        }
    );
    assert_eq!(counted.ask(), 100);
    // A host fold may leave any count behind it: the next fire asks.
    let left = groups(host(13, 100, vec![4]).expect("a fold of four of a hundred"));
    assert_eq!(left[0].fold, rs::GroupFold::Host(4));
    // A model with window lists, or a pass binding no state, carries none.
    assert_eq!(
        lanes_frozen(&stores, &ids, &groups_of, 1, true).unwrap(),
        [None]
    );
    assert_eq!(
        lanes_frozen(&stores, &[], &groups_of, 1, false).unwrap(),
        [None]
    );

    // The device plan: the device picks where the rows start, inside
    // what the lane reserved.
    let device = |reserved: u32| {
        let cell = Arc::new(Mutex::new(Cursor {
            reserved,
            tail: None,
            ..Cursor::default()
        }));
        let lanes = PassLanes {
            places: vec![geometry::Place {
                folded: 13,
                folded_exact: true,
                tail: None,
                shared: false,
                spans: Vec::new(),
                // The device places the rows: its engine holds the lane
                // to what the set wrote.
                readable: 0,
                readable_known: false,
                set: 1,
            }],
            cursors: vec![cell],
            kv: Vec::new(),
        };
        device_rs_plan(
            &stores,
            &ids,
            &groups_of,
            &lanes,
            &[0, 5],
            &geometry::Fold::Device,
            false,
        )
    };
    let planned = groups(device(13 + 5 + 5).unwrap());
    assert_eq!(planned[0].writes, Some(13..23));
    assert_eq!(planned[0].fold, rs::GroupFold::Device);
    // The decode reserved its whole extent up front, as the MTP example:
    // what it reserved is buffered, and is no count of what it carries.
    let ahead = groups(device(13 + 64).expect("planned whatever it reserved"));
    assert_eq!(ahead[0].writes, Some(13..13 + 64));
}
