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
    assert_eq!(
        replay_rung(&stores, 56, carrying(2048), 2048, 1, false),
        Ok(2048)
    );
    let host = replay_rung(&stores, 56, carrying(2049), 2049, 1, false).unwrap_err();
    assert!(
        host.contains("of 2048 tokens")
            && host.contains("2049 carried recurrent rows")
            && host.contains("fold what its lanes have written before they write more"),
        "{host}"
    );
    assert!(
        replay_rung(
            &stores,
            56,
            FireRows {
                own: 4096,
                carried: 0
            },
            0,
            1,
            false
        )
        .is_ok()
    );
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
            let plan =
                host_rs_plan(&stores, &ids, &groups_of, &req, &fold, &[], None, None).unwrap();
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
                assert_eq!(
                    frozen_at,
                    [Some(start)],
                    "step {step}: folded through its first row"
                );
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
                assert_eq!(
                    frozen_at,
                    [Some(10)],
                    "step {step}: nothing read, nothing folded"
                );
                assert_eq!(replayed, step, "step {step}: every row before it carried");
                assert!(
                    matches!(req.lanes[0].lane.rs, RsVerb::Buffer { .. }),
                    "step {step}"
                );
                let alone = StepComposition::of([&req.lanes[0].lane]);
                assert_eq!(alone.treatment, Treatment::Buffers);
            }
            fired.push(req.lanes[0].lane.clone());
        }
        let mixed = StepComposition::of(&fired);
        assert_eq!(
            mixed.treatment,
            Treatment::Buffers,
            "step {step}: a mixed step buffers"
        );
        assert_eq!(mixed.decode_lanes, 2);
    }
}
