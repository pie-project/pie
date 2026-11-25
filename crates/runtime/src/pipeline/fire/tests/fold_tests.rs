use super::*;
use crate::inferlet::host::working_set::Cursor;
use crate::store::kv::KvStoreError;
use crate::store::kv::page_table::{KvTableError, WorkingSetId};
use crate::store::registry::Stores;

fn cursor(reserved: u32) -> CursorCell {
    Arc::new(Mutex::new(Cursor {
        reserved,
        ..Cursor::default()
    }))
}

/// The lanes of a pass, each on a working set folded through `folded`,
/// `exact` when no fold on its way or untold may have moved it further.
fn lanes(folded: u32, exact: bool, cursors: &[&CursorCell]) -> PassLanes {
    PassLanes {
        places: cursors
            .iter()
            .map(|cell| {
                let now = cell.lock().unwrap().clone();
                geometry::Place {
                    folded,
                    folded_exact: exact,
                    tail: now.tail,
                    shared: false,
                    readable: now.readable(),
                    readable_known: now.readable_known(),
                    set: 1,
                    spans: now.spans,
                }
            })
            .collect(),
        cursors: cursors.iter().map(|&cell| cell.clone()).collect(),
        kv: Vec::new(),
    }
}

/// Those lanes as a pass reads them when it starts ([`pass_lanes`]):
/// what their set has folded is its KV working set's.
fn lanes_on(stores: &Stores, ws: WorkingSetId, cursors: &[&CursorCell]) -> PassLanes {
    let (folded, exact) = folded(stores, ws);
    lanes(folded as u32, exact, cursors)
}

/// The set's folded length, and whether that is all it has folded.
fn folded(stores: &Stores, ws: WorkingSetId) -> (u64, bool) {
    let kv = stores.kv.lock();
    (kv.folded(ws).unwrap(), kv.folded_exact(ws).unwrap())
}

fn landing(start: u32, rows: u32) -> geometry::Landing {
    geometry::Landing { start, rows }
}

fn all() -> geometry::Fold {
    geometry::Fold::Host(vec![u32::MAX])
}

#[test]
fn a_beam_pool_that_could_never_fit_alone_is_refused_by_name() {
    assert_eq!(super::beam_pool_fits(0, 40, 37, 40), Ok(()));
    let refusal = super::beam_pool_fits(2, 41, 38, 40).unwrap_err();
    assert!(refusal.contains(super::BEAM_POOL_OUTGROWN), "{refusal}");
    assert!(refusal.contains("lane 2"), "{refusal}");
}

fn request(lanes: usize) -> crate::engine::FireRequest {
    crate::engine::FireRequest {
        lanes: (0..lanes)
            .map(|_| crate::engine::FireLane::default())
            .collect(),
        ..crate::engine::FireRequest::default()
    }
}

fn rows(held: u32, tokens: &[u32]) -> crate::engine::FireRequest {
    crate::engine::FireRequest {
        lanes: vec![
            ::engine::Lane {
                tokens: tokens.to_vec(),
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

/// A model of 4-row pages, and a working set of it folded through
/// `folded`.
fn frozen_at(folded: u64) -> (usize, Stores, WorkingSetId) {
    let model = crate::store::registry::register_model(4, &[8], &[1]);
    let stores = crate::store::registry::get(model, 0);
    let ws = {
        let mut kv = stores.kv.lock();
        let ws = kv.create_working_set();
        kv.fold(ws, folded, false).unwrap();
        ws
    };
    (model, stores, ws)
}

fn state_at(stores: &Stores, frozen: u32) -> crate::store::rs::RsWorkingSetId {
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(crate::store::rs::RsGeometry {
        state_size: 8,
        buffer_page_tokens: 4,
    });
    rs.fold(ws, frozen);
    ws
}

/// The fire's part on one working set, folding it by `by`.
fn part(
    model: usize,
    id: WorkingSetId,
    by: FoldBy,
    tags: Vec<ReportTag>,
) -> (KvWorkingSet, FireWs) {
    let kv = KvWorkingSet::new(model, 0, id, 8);
    let part = FireWs {
        ws: id,
        kvtxn: KvTxnGuard::new(model, 0, None),
        log: None,
        lease: kv.fire_lease().expect("lease the submitted working set"),
        fold: Some(FoldAt { by, decoded: false }),
        reports: tags,
        staged: None,
    };
    (kv, part)
}

/// What a fold placed as its fire is prepared freezes.
fn frozen_as_prepared(by: Option<&FoldBy>) -> Option<u64> {
    match by {
        Some(FoldBy::To(to)) => Some(*to),
        _ => None,
    }
}

/// A fold placed when its fire settles, the fire submitted on `ws`.
fn settled(
    stores: &Stores,
    ws: WorkingSetId,
    count: Option<u32>,
    tags: Vec<u64>,
    reach: u64,
    written: bool,
) -> FoldAt {
    stores.kv.lock().unsettle(ws).unwrap();
    let completion = crate::engine::WorkItemCompletion::deferred_with_guard(None);
    FoldAt {
        by: FoldBy::Settled {
            count,
            tags,
            reach,
            written,
            rs: None,
            unplaced: Some(Unplaced {
                ws,
                submitted: Some((stores.kv.clone(), completion)),
            }),
        },
        decoded: false,
    }
}

fn taken(tags: &[ReportTag]) -> std::collections::HashMap<u64, ::engine::KvWrites> {
    tags.iter()
        .filter_map(|tag| tag.take().map(|written| (tag.0, written)))
        .collect()
}

fn report(tag: u64, fold: u32) -> ::engine::KvWrites {
    ::engine::KvWrites {
        tag,
        fold: Some(fold),
        ..::engine::KvWrites::default()
    }
}

fn none() -> std::collections::HashMap<u64, ::engine::KvWrites> {
    std::collections::HashMap::new()
}

const UNSETTLED: KvStoreError = KvStoreError::Table(KvTableError::FoldUnsettled);
const UNPLACED: KvStoreError = KvStoreError::Table(KvTableError::FoldUnplaced);

#[test]
fn fold_every_case() {
    a_fold_the_host_knows_is_frozen_as_its_fire_is_prepared();
    a_fold_the_device_counts_leaves_the_folded_length_until_it_settles();
    the_last_fold_to_settle_makes_the_folded_length_exact();
    a_fold_never_reported_leaves_the_folded_length_a_bound();
    a_fire_carrying_a_span_leaves_its_lag();
    lanes_of_one_set_fold_to_where_the_first_ends();
    a_count_on_rows_the_device_places_is_frozen_as_the_fire_is_prepared();
    a_count_past_rows_the_device_places_is_placed_where_they_ended();
    a_cut_pass_folds_from_where_each_piece_finds_it();
    a_cut_pass_that_stops_short_leaves_what_its_pieces_folded();
    a_piece_inside_the_frozen_prefix_begins_nothing();
    a_pass_supplied_from_the_cache_folds_what_it_states();
    a_state_resumes_no_further_than_its_pass_folds();
    a_record_checkpoint_on_a_cut_is_kept_with_the_next_piece_on_its_way();
    a_checkpoint_taken_ahead_follows_the_fold_the_device_counts();
}

fn a_fire_carrying_a_span_leaves_its_lag() {
    let cell = cursor(0);
    cell.lock().unwrap().spans = vec![(10, 14, 2), (50, 60, 4)];
    let span = Arc::new(media::front::EncodedSpan {
        token_count: 20,
        position_span: 5,
        grid: media::front::Grid::still(4, 5),
        patch_grid: media::front::Grid::still(8, 10),
        uses_mrope: true,
        payload: Vec::new(),
        rows: 80,
        positions: Vec::new(),
        prefix: Vec::new(),
        placeholder: 0,
        suffix: Vec::new(),
    });
    let run = crate::pipeline::media::MatchedRun {
        lane: 0,
        anchor: 3,
        rows: 20,
        span,
    };
    let landings = [landing(40, 30)];
    let spans = lagging_spans(&landings, &[run], &[0], &(40..70));
    advance_cursor(&cell, Some(70), Some((40, spans)));
    let now = cell.lock().unwrap().clone();
    assert_eq!((now.tail, now.reserved, now.written), (Some(70), 70, 70));
    assert_eq!(
        now.spans,
        [(10, 14, 2), (43, 63, 15)],
        "the write from slot 40 replaces what lay past it"
    );
    assert_eq!(crate::inferlet::host::working_set::lag(&now.spans, 70), 17);
}

/// A count the host knows, over a folded length it knows, is a slot:
/// frozen in the step that begins the fire's log, read by whatever is
/// staged next, and only sealed when the fire settles.
fn a_fold_the_host_knows_is_frozen_as_its_fire_is_prepared() {
    let (_, stores, ws) = frozen_at(4);
    let cell = cursor(0);
    let pass = lanes_on(&stores, ws, &[&cell]);
    let mut req = request(1);
    let mut tags = Vec::new();
    let landings = [landing(4, 6)];
    let mut at = |fold: &geometry::Fold, end: u64, last: bool| {
        fold_at(
            &pass,
            &[0],
            &landings,
            fold,
            end,
            last,
            None,
            &mut req,
            &mut tags,
        )
    };
    let three = geometry::Fold::Host(vec![3]);
    assert!(matches!(at(&three, 10, false), Some(FoldBy::To(7))));
    assert!(
        matches!(at(&three, 6, true), Some(FoldBy::To(6))),
        "short of what this fire writes"
    );
    assert!(
        matches!(at(&all(), 20, true), Some(FoldBy::To(10))),
        "no further than its rows land"
    );
    assert!(at(&geometry::Fold::Host(vec![0]), 10, true).is_none());
    assert!(tags.is_empty() && req.lanes[0].reports == 0);

    let seq = begin_piece(&mut stores.kv.lock(), ws, 4, Some(7)).unwrap();
    assert!(seq.is_some());
    assert_eq!(folded(&stores, ws), (7, true));
    assert_eq!(stores.kv.lock().unsettled(ws).unwrap(), 0);
    let next = lanes_on(&stores, ws, &[&cell]);
    assert!(next.places[0].folded == 7 && next.places[0].folded_exact);
    // Nothing waits on the fire: a fork takes the length it froze.
    let child = stores.kv.lock().fork(ws).unwrap();
    assert_eq!(folded(&stores, child), (7, true));
    let to = |to| FoldAt {
        by: FoldBy::To(to),
        decoded: false,
    };
    settle_fold(&stores, ws, to(7), &none(), None, None);
    assert_eq!(folded(&stores, ws), (7, true));
    // A fire that failed leaves what it froze frozen: its pipeline
    // failed with it, and its sets fire no more.
    begin_piece(&mut stores.kv.lock(), ws, 7, Some(9)).unwrap();
    settle_fold(&stores, ws, to(9), &none(), None, Some("the fire failed"));
    assert_eq!(folded(&stores, ws), (9, true));
}

/// A fold the device counts is not known when its fire is prepared:
/// the folded length stays where it was, marked a lower bound from the
/// fire's submit, and moves by the reported count when the fire
/// settles.
fn a_fold_the_device_counts_leaves_the_folded_length_until_it_settles() {
    let (model, stores, ws) = frozen_at(4);
    let cell = cursor(12);
    let pass = lanes_on(&stores, ws, &[&cell, &cell]);
    let mut req = request(2);
    let mut tags = Vec::new();
    let by = fold_at(
        &pass,
        &[0, 1],
        &[landing(4, 8), landing(4, 8)],
        &geometry::Fold::Device,
        u64::MAX,
        true,
        None,
        &mut req,
        &mut tags,
    )
    .expect("a device fold is placed when it settles");
    assert!(
        req.lanes[0].reports != 0 && req.lanes[1].reports == 0,
        "the first lane carries the fold"
    );
    assert!(matches!(
        &by,
        FoldBy::Settled {
            count: None,
            reach: 12,
            written: false,
            ..
        }
    ));
    // Its prepare freezes nothing.
    let freeze = frozen_as_prepared(Some(&by));
    assert_eq!(freeze, None);
    begin_piece(&mut stores.kv.lock(), ws, 4, freeze)
        .unwrap()
        .expect("it writes past what is frozen");
    assert_eq!(folded(&stores, ws), (4, true));
    // Its submit.
    let (_set, mut fire) = part(model, ws, by, tags);
    let completion = crate::engine::WorkItemCompletion::deferred_with_guard(None);
    unsettle(&stores, &completion, std::slice::from_mut(&mut fire));
    assert_eq!(folded(&stores, ws), (4, false));
    assert_eq!(stores.kv.lock().unsettled(ws).unwrap(), 1);
    assert_eq!(stores.kv.lock().fork(ws), Err(UNSETTLED));
    // A pass staged meanwhile reads a lower bound, and says so.
    let next = lanes_on(&stores, ws, &[&cell]);
    assert!(next.places[0].folded == 4 && !next.places[0].folded_exact);
    assert!(!appends_folded(&next, &[0]));

    record_kv_writes(vec![report(req.lanes[0].reports, 3)]);
    let reported = taken(&fire.reports);
    let fold = fire.fold.take().unwrap();
    settle_fold(&stores, ws, fold, &reported, None, None);
    assert_eq!(folded(&stores, ws), (7, true));
    assert_eq!(stores.kv.lock().unsettled(ws).unwrap(), 0);
    let child = stores.kv.lock().fork(ws).unwrap();
    assert_eq!(folded(&stores, child), (7, true));
}

fn the_last_fold_to_settle_makes_the_folded_length_exact() {
    let (_, stores, ws) = frozen_at(4);
    let counted =
        |count: u32, reach: u64| settled(&stores, ws, Some(count), Vec::new(), reach, false);
    let (first, second) = (counted(2, 20), counted(9, 10));
    assert_eq!(stores.kv.lock().unsettled(ws).unwrap(), 2);
    settle_fold(&stores, ws, first, &none(), None, None);
    assert_eq!(folded(&stores, ws), (6, false), "another is on its way");
    // Counted from where the first left it, short of what the fire wrote.
    settle_fold(&stores, ws, second, &none(), None, None);
    assert_eq!(folded(&stores, ws), (10, true));
    // One never submitted lifts nothing of another's.
    let other = counted(1, 20);
    let unsubmitted = FoldAt {
        by: FoldBy::Settled {
            count: Some(1),
            tags: Vec::new(),
            reach: 20,
            written: false,
            rs: None,
            unplaced: None,
        },
        decoded: false,
    };
    settle_fold(
        &stores,
        ws,
        unsubmitted,
        &none(),
        None,
        Some("never submitted"),
    );
    assert_eq!(folded(&stores, ws), (10, false));
    assert_eq!(stores.kv.lock().unsettled(ws).unwrap(), 1);
    settle_fold(&stores, ws, other, &none(), None, None);
    assert_eq!(folded(&stores, ws), (11, true));
}

/// A fold whose count never came (the engine reported none, or the
/// pass failed) settles unplaced: the folded length is a lower bound
/// from then on, in the set, so the set is forked no more, and nothing
/// waits for a settle that has happened.
fn a_fold_never_reported_leaves_the_folded_length_a_bound() {
    for failure in [None, Some("the fire failed")] {
        let (_, stores, ws) = frozen_at(4);
        let (count, tags) = match failure {
            None => (None, vec![u64::MAX]),
            Some(_) => (Some(3), Vec::new()),
        };
        let fold = settled(&stores, ws, count, tags, 20, false);
        assert_eq!(stores.kv.lock().fork(ws), Err(UNSETTLED));
        settle_fold(&stores, ws, fold, &none(), None, failure);
        assert_eq!(folded(&stores, ws), (4, false));
        assert_eq!(stores.kv.lock().unsettled(ws).unwrap(), 0);
        // No fire is left to place it: a copy would state the bound
        // as its length, and is refused by that name.
        assert_eq!(stores.kv.lock().fork(ws), Err(UNPLACED));
    }
}

/// Several lanes on one set ending at different slots: the set folds
/// through where its first lane ends, in the KV and in its recurrent
/// state alike.
fn lanes_of_one_set_fold_to_where_the_first_ends() {
    let (_, stores, ws) = frozen_at(4);
    let rs = state_at(&stores, 4);
    let cell = cursor(0);
    let pass = lanes_on(&stores, ws, &[&cell, &cell]);
    // Three rows and four, from slot 4.
    let mut req = crate::engine::FireRequest {
        lanes: [3, 4]
            .into_iter()
            .map(|rows| rows_of(4, rows).into())
            .collect(),
        ..crate::engine::FireRequest::default()
    };
    let landings = [landing(4, 3), landing(4, 4)];
    let ids = [rs, rs];
    let groups = lane_groups(&[ws, ws]);
    assert_eq!(groups.len(), 1);
    let plan = host_rs_plan(&stores, &ids, &groups, &req, &all(), &[], None, None).unwrap();
    let rs::RsPlan::Groups(plan) = plan;
    assert_eq!(plan[0].fold, rs::GroupFold::Host(3));
    let mut tags = Vec::new();
    let by = fold_at(
        &pass,
        &[0, 1],
        &landings,
        &all(),
        8,
        true,
        Some(rs),
        &mut req,
        &mut tags,
    );
    assert!(
        matches!(by, Some(FoldBy::To(7))),
        "4 + 3, not the 8 written"
    );
}

fn rows_of(held: u32, rows: usize) -> ::engine::Lane {
    ::engine::Lane {
        tokens: vec![0; rows],
        kv: ::engine::KvDelta {
            held,
            ..Default::default()
        },
        ..::engine::Lane::default()
    }
}

/// Rows the device places, folded by a count the host knows. The
/// engine folds through `min(folded + count, first + rows)`: a count
/// within the lane's rows is exact wherever they land, and one past
/// them is exact when they start at the folded length. Either is
/// frozen as the fire is prepared, never past the rows: a set that
/// reserved seventeen slots and fires one row folding all it holds
/// folds one.
fn a_count_on_rows_the_device_places_is_frozen_as_the_fire_is_prepared() {
    use DeviceFold::{Count, Device, None as Nothing, PastRows};
    assert_eq!(device_fold(None, 2, 8, false), Device);
    assert_eq!(device_fold(Some(0), 2, 8, true), Nothing);
    assert_eq!(device_fold(Some(1), 2, 8, false), Count(1));
    assert_eq!(device_fold(Some(2), 2, 8, false), Count(2));
    assert_eq!(device_fold(Some(5), 2, 8, false), PastRows(5));
    assert_eq!(device_fold(Some(5), 2, 8, true), Count(2));
    assert_eq!(device_fold(Some(u32::MAX), 1, 17, true), Count(1));
    assert_eq!(device_fold(Some(u32::MAX), 2, 1, true), Count(1));
    assert_eq!(device_fold(Some(2), 2, 0, true), Nothing);

    let (_, stores, ws) = frozen_at(4);
    let rs = state_at(&stores, 4);
    let groups = lane_groups(&[ws]);
    let cell = cursor(21);
    cell.lock().unwrap().tail = Some(4);
    // Two fires of one row, the second staged before the first settles.
    for at in [4u64, 5] {
        let pass = lanes_on(&stores, ws, &[&cell]);
        assert!(appends_folded(&pass, &[0]));
        let stated = device_fold(all().of(0), 1, 21 - at as u32, true);
        assert_eq!(stated, Count(1));
        // Its recurrent state folds the same count, as the plan is
        // reserved.
        let plan = device_rs_plan(&stores, &[rs], &groups, &pass, &[0, 1], &all(), false).unwrap();
        let rs::RsPlan::Groups(plan) = plan;
        assert_eq!(plan[0].fold, rs::GroupFold::Host(1));
        stores.rs.lock().unwrap().fold(rs, 1);
        let mut req = request(1);
        let mut tags = Vec::new();
        let by = device_fold_at(&pass, 0, stated, false, Some(rs), &mut req, &mut tags);
        assert!(matches!(by, Some(FoldBy::To(to)) if to == at + 1));
        assert!(tags.is_empty(), "no report is asked for");
        begin_piece(
            &mut stores.kv.lock(),
            ws,
            at,
            frozen_as_prepared(by.as_ref()),
        )
        .unwrap()
        .expect("it writes past what is frozen");
        advance_cursor(&cell, Some(at as u32 + 1), None);
        assert_eq!(folded(&stores, ws), (at + 1, true));
        assert_eq!(stores.rs.lock().unwrap().frozen(rs).unwrap(), at as u32 + 1);
    }
    // They settle: the pages seal, the length stays.
    for to in [5, 6] {
        let fold = FoldAt {
            by: FoldBy::To(to),
            decoded: false,
        };
        settle_fold(&stores, ws, fold, &none(), None, None);
        assert_eq!(folded(&stores, ws), (6, true));
    }

    // A lane that carries unfolded rows in: a count within its rows is
    // still a slot.
    cell.lock().unwrap().tail = None;
    let pass = lanes_on(&stores, ws, &[&cell]);
    assert!(!appends_folded(&pass, &[0]));
    let stated = device_fold(Some(2), 3, 21 - 6, false);
    let mut req = request(1);
    let mut tags = Vec::new();
    let by = device_fold_at(&pass, 0, stated, false, None, &mut req, &mut tags);
    assert!(matches!(by, Some(FoldBy::To(8))));
    // Behind a fold on its way unplaced, the same count is placed at
    // settle: the length it counts from is not known yet.
    let behind = lanes(6, false, &[&cell]);
    let by = device_fold_at(&behind, 0, stated, false, None, &mut req, &mut tags);
    assert!(matches!(
        by,
        Some(FoldBy::Settled {
            count: Some(2),
            written: false,
            ..
        })
    ));
    assert!(tags.is_empty());
}

/// A lane that carries unfolded rows in and folds past its own rows
/// folds through where those rows end, which the engine alone knows:
/// the fold is placed when the fire settles, from the end it reports,
/// and left unplaced without one. A fire that reports no rows is
/// refused it by name, as is a recurrent state.
fn a_count_past_rows_the_device_places_is_placed_where_they_ended() {
    let (_, stores, ws) = frozen_at(4);
    let cell = cursor(40);
    cell.lock().unwrap().tail = None;
    let pass = lanes_on(&stores, ws, &[&cell]);
    let stated = device_fold(Some(9), 2, 36, appends_folded(&pass, &[0]));
    assert_eq!(stated, DeviceFold::PastRows(9));
    let mut req = request(1);
    let mut tags = Vec::new();
    assert!(device_fold_at(&pass, 0, stated, false, None, &mut req, &mut tags).is_none());
    let refusal = past_rows_refusal(0, 2);
    assert!(
        refusal.contains("lane 0 folds past the 2 row(s) it writes")
            && refusal.contains("Fold no more than the lane writes"),
        "{refusal}"
    );
    let by = device_fold_at(&pass, 0, stated, true, None, &mut req, &mut tags);
    assert!(matches!(
        by,
        Some(FoldBy::Settled {
            count: Some(9),
            reach: 40,
            written: true,
            ..
        })
    ));
    // The rows landed at 8 and 9, behind four carried in: the engine
    // folded through min(4 + 9, 10).
    let fold = settled(&stores, ws, Some(9), Vec::new(), 40, true);
    settle_fold(&stores, ws, fold, &none(), Some(10), None);
    assert_eq!(folded(&stores, ws), (10, true));
    // The rows landed further on and the count ran out first.
    let fold = settled(&stores, ws, Some(9), Vec::new(), 40, true);
    settle_fold(&stores, ws, fold, &none(), Some(30), None);
    assert_eq!(folded(&stores, ws), (19, true));
    // No end reported: not placed, and the length is a bound.
    let fold = settled(&stores, ws, Some(9), Vec::new(), 40, true);
    settle_fold(&stores, ws, fold, &none(), None, None);
    assert_eq!(folded(&stores, ws), (19, false));

    let rs = state_at(&stores, 19);
    let groups = lane_groups(&[ws]);
    let pass = lanes(19, true, &[&cell]);
    let over = geometry::Fold::Host(vec![9]);
    let refused = device_rs_plan(&stores, &[rs], &groups, &pass, &[0, 2], &over, false)
        .err()
        .expect("a recurrent state's fold is final as stated");
    assert_eq!(refused, past_rows_refusal(0, 2));
}

/// One piece of a cut pass on a set, prepared and fired as the host
/// loop does: its recurrent state's fold committed as the piece is
/// reserved, the KV's frozen with the piece's log begun, the cursor
/// moved once the piece is on its way. Returns the folded length the
/// piece found, in the KV and in the recurrent state.
fn fire_piece(
    stores: &Stores,
    ws: WorkingSetId,
    rs: Option<crate::store::rs::RsWorkingSetId>,
    pass: &PassLanes,
    landings: &[geometry::Landing],
    fold: &geometry::Fold,
    slots: std::ops::Range<u64>,
    last: bool,
) -> (u64, Option<u32>) {
    let found = stores.kv.lock().folded(ws).unwrap();
    let state = rs.map(|rs| stores.rs.lock().unwrap().frozen(rs).unwrap());
    let mut req = crate::engine::FireRequest {
        lanes: vec![rows_of(slots.start as u32, (slots.end - slots.start) as usize).into()],
        ..crate::engine::FireRequest::default()
    };
    if let Some(rs) = rs {
        let groups = lane_groups(&[ws]);
        let plan = host_rs_plan(stores, &[rs], &groups, &req, fold, &[], None, None).unwrap();
        let rs::RsPlan::Groups(plan) = plan;
        if let rs::GroupFold::Host(count) = plan[0].fold {
            stores.rs.lock().unwrap().fold(rs, count);
        }
    }
    let mut tags = Vec::new();
    let by = fold_at(
        pass,
        &[0],
        landings,
        fold,
        slots.end,
        last,
        rs,
        &mut req,
        &mut tags,
    );
    begin_piece(
        &mut stores.kv.lock(),
        ws,
        slots.start,
        frozen_as_prepared(by.as_ref()),
    )
    .unwrap()
    .expect("it writes past what is frozen");
    advance_cursor(
        &pass.cursors[0],
        Some(slots.end as u32),
        Some((slots.start as u32, Vec::new())),
    );
    (found, state)
}

/// A pass of 128 rows from slot 10 in two pieces. Each piece folds
/// from the folded length its prepare finds, to the same absolute slot
/// the whole pass states: folding all, the second piece starts where
/// the first froze; folding twenty, both end at 30; folding nothing,
/// neither moves it. A recurrent state, cut only when it folds all or
/// nothing, stands at the same length before every piece.
fn a_cut_pass_folds_from_where_each_piece_finds_it() {
    let pieces = [(10..74, false), (74..138, true)];
    let cases = [
        (all(), [10, 74], 138, true),
        (geometry::Fold::Host(vec![20]), [10, 30], 30, false),
        (geometry::Fold::Host(vec![0]), [10, 10], 10, true),
    ];
    for (fold, found, after, with_state) in cases {
        for state in [false, true] {
            if state && !with_state {
                continue;
            }
            let (_, stores, ws) = frozen_at(10);
            let rs = state.then(|| state_at(&stores, 10));
            let cell = cursor(0);
            cell.lock().unwrap().tail = Some(10);
            // Read once, as the pass starts.
            let pass = lanes_on(&stores, ws, &[&cell]);
            let landings = [landing(10, 128)];
            for ((slots, last), found) in pieces.clone().into_iter().zip(found) {
                let at = fire_piece(&stores, ws, rs, &pass, &landings, &fold, slots, last);
                assert_eq!(at, (found, rs.map(|_| found as u32)));
            }
            assert_eq!(folded(&stores, ws), (after, true));
            if let Some(rs) = rs {
                assert_eq!(stores.rs.lock().unwrap().frozen(rs).unwrap(), after as u32);
            }
            assert_eq!(cell.lock().unwrap().tail, Some(138));
        }
    }
}

/// A cut pass whose later piece never fires (its reserve is refused)
/// leaves the folded length and the cursor at the pieces that went.
fn a_cut_pass_that_stops_short_leaves_what_its_pieces_folded() {
    let (_, stores, ws) = frozen_at(10);
    let rs = state_at(&stores, 10);
    let cell = cursor(0);
    cell.lock().unwrap().tail = Some(10);
    let pass = lanes_on(&stores, ws, &[&cell]);
    let landings = [landing(10, 192)];
    for (slots, went) in [(10..74, 74), (74..138, 138)] {
        fire_piece(
            &stores,
            ws,
            Some(rs),
            &pass,
            &landings,
            &all(),
            slots,
            false,
        );
        assert_eq!(folded(&stores, ws), (went, true));
        assert_eq!(stores.rs.lock().unwrap().frozen(rs).unwrap(), went as u32);
        let now = cell.lock().unwrap().clone();
        assert_eq!(
            (now.tail, now.reserved, now.written),
            (Some(went as u32), went as u32, went as u32)
        );
    }
}

/// A piece that would write inside what its set froze begins no log
/// and freezes nothing; it and any other piece prepared and not fired
/// fail the pipeline, and a set of a failed pipeline, or a fork of
/// one, joins no other.
fn a_piece_inside_the_frozen_prefix_begins_nothing() {
    use crate::store::{PipelineScope, ScopeRefusal};
    let (model, stores, ws) = frozen_at(4);
    let began = begin_piece(&mut stores.kv.lock(), ws, 3, Some(9)).unwrap();
    assert_eq!(began, None);
    assert_eq!(folded(&stores, ws), (4, true));
    let child = stores.kv.lock().fork(ws).unwrap();

    let failure = PipelineFailure::default();
    let scope = PipelineScope::failing_as(failure.clone(), || true);
    let set = KvWorkingSet::new(model, 0, ws, 4);
    set.claim_pipeline_scope(&scope).unwrap();
    let forked = set.forked(child);
    let mut ctx = Flight::context();
    let reason = "pipeline: a fire writes slot 3, inside its working set's frozen prefix";
    let gone: Resource<ForwardPass> = Resource::new_borrow(u32::MAX);
    record_submit_failure(&mut ctx, &gone, &failure, reason);
    record_submit_failure(&mut ctx, &gone, &failure, "a later one");
    assert_eq!(set.pipeline_failure().as_deref(), Some(reason));
    scope.close();
    let next = PipelineScope::failing_as(Default::default(), || true);
    let refused = Err(ScopeRefusal::Failed {
        owner: scope.id(),
        reason: reason.to_string(),
    });
    assert_eq!(set.claim_pipeline_scope(&next), refused);
    assert_eq!(forked.claim_pipeline_scope(&next), refused);
    assert_eq!(set.forked(child).claim_pipeline_scope(&next), refused);
}

/// What a fire froze as it was prepared stands when the fire does not
/// follow, so the pipeline fails however the call that prepared it
/// ends: here by a trap between the freeze and the submit, on a pass
/// gone, and dropped where it waits. What [`reserve`] returns says so
/// until the engine takes the fire.
#[tokio::test(flavor = "current_thread")]
async fn a_fire_prepared_and_never_submitted_fails_its_pipeline() {
    use crate::store::{PipelineScope, ScopeRefusal};

    async fn prepared(
        ctx: &mut TestContext,
        model: usize,
        stores: &Stores,
        failure: &PipelineFailure,
    ) -> Unfired {
        let none = || Ok(rs::RsPlan::Groups(Vec::new()));
        let id = uuid::Uuid::new_v4();
        let reserved = reserve(ctx, (model, 0), stores, id, failure, &[], &[], &[], none);
        reserved.await.unwrap().unfired
    }
    /// A piece prepared as a pass prepares one, whose call then asks
    /// for a pass gone, as each path does before its submit.
    async fn trapped(
        ctx: &mut TestContext,
        model: usize,
        stores: &Stores,
        ws: WorkingSetId,
        failure: &PipelineFailure,
    ) -> Anyhow<()> {
        let unfired = prepared(ctx, model, stores, failure).await;
        begin_piece(&mut stores.kv.lock(), ws, 4, Some(7))?;
        let gone: Resource<ForwardPass> = Resource::new_borrow(u32::MAX);
        ctx.resources().get_mut(&gone)?;
        drop(unfired.submitted());
        Ok(())
    }

    let (model, stores, ws) = frozen_at(4);
    let mut ctx = Flight::context();
    let failure = PipelineFailure::default();
    let scope = PipelineScope::failing_as(failure.clone(), || true);
    let set = KvWorkingSet::new(model, 0, ws, 4);
    set.claim_pipeline_scope(&scope).unwrap();
    assert!(
        trapped(&mut ctx, model, &stores, ws, &failure)
            .await
            .is_err()
    );
    assert_eq!(folded(&stores, ws), (7, true), "the freeze stands");
    assert_eq!(set.pipeline_failure().as_deref(), Some(UNFIRED));
    let child = stores.kv.lock().fork(ws).unwrap();
    scope.close();
    let next = PipelineScope::failing_as(Default::default(), || true);
    let refused = Err(ScopeRefusal::Failed {
        owner: scope.id(),
        reason: UNFIRED.to_string(),
    });
    assert_eq!(set.claim_pipeline_scope(&next), refused);
    assert_eq!(set.forked(child).claim_pipeline_scope(&next), refused);

    // A refusal that said why keeps its reason.
    let failure = PipelineFailure::default();
    let unfired = prepared(&mut ctx, model, &stores, &failure).await;
    let gone: Resource<ForwardPass> = Resource::new_borrow(u32::MAX);
    record_submit_failure(&mut ctx, &gone, &failure, "pipeline: submit failed");
    drop(unfired);
    assert_eq!(
        failure.lock().unwrap().as_deref(),
        Some("pipeline: submit failed")
    );

    // A call dropped where it waits, its fire prepared.
    let failure = PipelineFailure::default();
    let unfired = prepared(&mut ctx, model, &stores, &failure).await;
    let mut waiting = Box::pin(async move {
        let _unfired = unfired;
        std::future::pending::<()>().await;
    });
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    assert!(waiting.as_mut().poll(&mut cx).is_pending());
    assert_eq!(*failure.lock().unwrap(), None);
    drop(waiting);
    assert_eq!(failure.lock().unwrap().as_deref(), Some(UNFIRED));

    // A fire the engine took leaves this one no more (what it leaves
    // until it is queued is its own), and a reserve refused leaves none.
    let failure = PipelineFailure::default();
    let unqueued = prepared(&mut ctx, model, &stores, &failure)
        .await
        .submitted();
    assert_eq!(*failure.lock().unwrap(), None);
    drop(unqueued);
    assert_eq!(failure.lock().unwrap().as_deref(), Some(UNQUEUED));
    let failure = PipelineFailure::default();
    let refused = || Err("no mode".to_string());
    let id = uuid::Uuid::new_v4();
    let reserved = reserve(
        &mut ctx,
        (model, 0),
        &stores,
        id,
        &failure,
        &[],
        &[],
        &[],
        refused,
    );
    assert!(reserved.await.is_err());
    assert_eq!(*failure.lock().unwrap(), None);
}

/// A prompt of `tokens` sealed in the cache of `stores` by a set since
/// gone, 16 rows a page; its rows' hashes, and its pages' keys.
fn sealed_prompt(
    stores: &Stores,
    tokens: &[u32],
) -> (
    Vec<crate::store::kv::hash::Hash256>,
    Vec<crate::store::kv::hash::Hash256>,
) {
    let ws = stores.kv.lock().create_shared_working_set();
    let slots = written(stores, ws, tokens);
    let mut kv = stores.kv.lock();
    kv.seal_rows(ws, 0, &slots, 16).unwrap();
    let keys = kv.page_keys(ws, 0, &slots, 16).unwrap().unwrap();
    let epoch = kv.current_epoch();
    kv.release_working_set(ws, epoch);
    kv.retire_idle();
    (slots, keys)
}

/// `tokens` written into `ws` from row 0, 16 rows a page, by a pass
/// whose fires have not landed yet; their rows' hashes.
fn written(
    stores: &Stores,
    ws: WorkingSetId,
    tokens: &[u32],
) -> Vec<crate::store::kv::hash::Hash256> {
    let mut kv = stores.kv.lock();
    kv.set_keeps(true);
    let pages = (tokens.len() as u64).div_ceil(16);
    kv.reserve(ws, pages).unwrap();
    let indexes: Vec<u64> = (0..pages).collect();
    let need = kv.write_demand(ws, &indexes).unwrap();
    let mut granted = kv.reserve_device_pages(need).unwrap();
    let prepared = kv
        .prepare_write_reserved(ws, &indexes, &mut granted)
        .unwrap();
    let commits: Vec<_> = (0..pages)
        .map(|_| crate::store::kv::write::PageCommit {
            token_hashes: Vec::new(),
            page_hash: None,
        })
        .collect();
    let seq = kv.publish_prepared(prepared, &commits).unwrap();
    kv.settle(seq);
    kv.slot_hashes(ws, 0, tokens, 16).unwrap().unwrap()
}

/// A keyed pass over `tokens` on a fresh set of `stores`, shared within
/// the test tenant: the set, and its rows' hashes.
fn keyed_pass(
    stores: &Stores,
    tokens: &[u32],
) -> (WorkingSetId, Vec<crate::store::kv::hash::Hash256>) {
    keyed_pass_as(stores, tokens, Some(crate::policy::share::test_tenant()))
}

/// [`keyed_pass`] on a set that shares as `ask` says (`None`: private).
fn keyed_pass_as(
    stores: &Stores,
    tokens: &[u32],
    ask: Option<crate::policy::share::ShareAs>,
) -> (WorkingSetId, Vec<crate::store::kv::hash::Hash256>) {
    let mut kv = stores.kv.lock();
    let ws = kv.create_working_set();
    if let Some(ask) = ask {
        kv.share(ws, ask).unwrap();
    }
    let slots = kv.slot_hashes(ws, 0, tokens, 16).unwrap().unwrap();
    (ws, slots)
}

/// Two tenants' identical prompts never share a page: not by a lookup,
/// not by a coalesced claim, not by what a released set leaves kept.
/// One tenant's tenant-scoped sets share all three ways, and a private
/// set with the same prompt shares with no one.
#[tokio::test(flavor = "current_thread")]
async fn tenants_never_share_pages_and_one_tenants_sets_do() {
    use crate::policy::share::tenant_of;
    let model = crate::store::registry::register_model(16, &[64], &[4]);
    let stores = crate::store::registry::get(model, 0);
    stores.kv.lock().set_keeps(true);
    let tokens: Vec<u32> = (0..65).map(|t| 2700 + t).collect();
    let of = |tenant: &str| keyed_pass_as(&stores, &tokens, Some(tenant_of(tenant)));
    let looked = |ws, slots: &[_]| {
        supply(&stores, ws, &rows(0, &tokens), slots, Attach::Kv, 65, 16).rows
    };
    let (a1, a1_slots) = of("a");
    let claims = crate::pipeline::coalesce::before_pass(&stores, a1, 0, &a1_slots, || {})
        .await
        .expect("a's first claims what it computes");
    // Coalesce: b's identical prompt waits on nothing of a's; a's own
    // second waits on its first.
    let (b1, b1_slots) = of("b");
    assert_ne!(a1_slots, b1_slots, "another tenant keys another chain");
    let b_claims =
        crate::pipeline::coalesce::before_pass(&stores, b1, 0, &b1_slots, || {}).await;
    assert!(b_claims.is_some(), "b claims its own pages");
    let (a2, a2_slots) = of("a");
    assert_eq!(a1_slots, a2_slots, "one tenant keys one chain");
    let a_waits = {
        let (stores, slots) = (stores.clone(), a2_slots.clone());
        tokio::spawn(async move {
            crate::pipeline::coalesce::before_pass(&stores, a2, 0, &slots, || {}).await
        })
    };
    settle_tasks().await;
    assert!(!a_waits.is_finished(), "a's second waits on a's first");
    written(&stores, a1, &tokens);
    stores.kv.lock().freeze(a1, 65).unwrap();
    SealOnLanding {
        claims,
        stores: stores.clone(),
        ws: a1,
        start: 0,
        slots: a1_slots,
        page_size: 16,
        failure: PipelineFailure::default(),
    }
    .landed(true);
    assert!(a_waits.await.unwrap().is_none(), "sealed: nothing left to claim");
    // Lookup, while a's first is live: a's second maps it in, b nothing.
    assert_eq!(looked(a2, &a2_slots), 64, "a maps a's four sealed pages");
    let (b2, b2_slots) = of("b");
    assert_eq!(looked(b2, &b2_slots), 0, "b finds none of a's");
    // Keep: a's sets gone, their prefix kept for a alone.
    drop(b_claims);
    {
        let mut kv = stores.kv.lock();
        let epoch = kv.current_epoch();
        for ws in [a1, a2, b1, b2] {
            kv.release_working_set(ws, epoch);
        }
        kv.retire_idle();
        assert!(!kv.cached_runs().is_empty(), "a's prefix is kept");
    }
    let (b3, b3_slots) = of("b");
    assert_eq!(looked(b3, &b3_slots), 0, "b finds nothing a kept");
    let (a3, a3_slots) = of("a");
    assert_eq!(looked(a3, &a3_slots), 64, "a finds what a kept");
    // A private set with a prompt of its own: no one maps it while it
    // lives, it looks nothing up, and it leaves nothing kept.
    let mine: Vec<u32> = (0..65).map(|t| 2900 + t).collect();
    let (p1, p1_slots) = keyed_pass_as(&stores, &mine, None);
    assert_eq!(
        supply(&stores, p1, &rows(0, &mine), &p1_slots, Attach::No, 65, 16).rows,
        0,
        "a private set looks nothing up"
    );
    written(&stores, p1, &mine);
    {
        let mut kv = stores.kv.lock();
        kv.freeze(p1, 65).unwrap();
        kv.seal_rows(p1, 0, &p1_slots, 16).unwrap();
        assert!(kv.share(p1, tenant_of("a")).is_err(), "too late to ask");
    }
    let (a4, a4_slots) = keyed_pass_as(&stores, &mine, Some(tenant_of("a")));
    assert_ne!(a4_slots, p1_slots);
    assert_eq!(
        supply(&stores, a4, &rows(0, &mine), &a4_slots, Attach::Kv, 65, 16).rows,
        0,
        "a live private set is no one's to map"
    );
    let kept = {
        let mut kv = stores.kv.lock();
        let before = kv.cached_runs().len();
        let epoch = kv.current_epoch();
        kv.release_working_set(p1, epoch);
        kv.retire_idle();
        kv.cached_runs().len() - before
    };
    assert_eq!(kept, 0, "a private set keeps nothing at release");
}

/// A group scope crosses tenants only under a grant: two granted
/// tenants' sets asking `group(g)` share pages; an ungranted tenant's
/// `group(g)` is its own. Once the grant changes (revoked for one
/// tenant, a new id for the rest), a set with no rows takes the
/// tenant-local domain on its next pass, and a set with rows shares no
/// more.
#[tokio::test(flavor = "current_thread")]
async fn a_group_shares_across_tenants_only_while_granted() {
    use crate::policy::grants;
    use crate::policy::share::{
        Asker, SessionId, ShareAs, ShareScope, TenantId, scope_domain,
    };
    let _grants = grants::TEST_LOCK.lock();
    let grant = |version, tenants: &[&str]| {
        let tenants = tenants.iter().map(|t| t.to_string()).collect();
        grants::install(version, vec![("g".into(), tenants, version)], None);
    };
    grants::withdraw();
    grant(1, &["a", "b"]);
    let asker = |tenant: &str| Asker {
        tenant: TenantId(tenant.into()),
        session: SessionId::Named("s".into()),
    };
    let group = |tenant: &str| ShareAs::Scope(ShareScope::Group("g".into()), asker(tenant));
    let model = crate::store::registry::register_model(16, &[64], &[4]);
    let stores = crate::store::registry::get(model, 0);
    stores.kv.lock().set_keeps(true);
    let tokens: Vec<u32> = (0..65).map(|t| 3100 + t).collect();
    let of = |tenant: &str| keyed_pass_as(&stores, &tokens, Some(group(tenant)));
    let looked = |ws, slots: &[_]| {
        supply(&stores, ws, &rows(0, &tokens), slots, Attach::Kv, 65, 16).rows
    };
    // a computes and seals the prompt under group(g).
    let (a1, a1_slots) = of("a");
    written(&stores, a1, &tokens);
    {
        let mut kv = stores.kv.lock();
        kv.freeze(a1, 65).unwrap();
        kv.seal_rows(a1, 0, &a1_slots, 16).unwrap();
    }
    // b, granted too, keys a's chain and maps a's pages in.
    let (b1, b1_slots) = of("b");
    assert_eq!(a1_slots, b1_slots, "granted tenants key one chain");
    assert_eq!(looked(b1, &b1_slots), 64, "b maps a's four sealed pages");
    // c, not granted: its group(g) is its own.
    let (c1, c1_slots) = of("c");
    assert_ne!(c1_slots, a1_slots, "an ungranted tenant keys its own chain");
    assert_eq!(looked(c1, &c1_slots), 0, "c finds none of a's");
    // A set of b's that asked but keyed nothing yet; a1 holds rows.
    let fresh = {
        let mut kv = stores.kv.lock();
        let ws = kv.create_working_set();
        kv.share(ws, group("b")).unwrap();
        ws
    };
    // b's grant is revoked; a's row is reissued under a new id.
    grant(2, &["a"]);
    let mut kv = stores.kv.lock();
    let local = scope_domain(&kv.domain(), &ShareScope::Group("g".into()), &asker("b"), None);
    assert_eq!(
        kv.pass_domain(fresh).unwrap(),
        (local, true),
        "a set with no rows takes the tenant-local domain"
    );
    assert!(kv.shares(fresh));
    let (_, shared) = kv.pass_domain(a1).unwrap();
    assert!(!shared, "a set with rows keyed under the old grant shares no more");
    assert!(!kv.shares(a1));
    let fresh_slots = kv.slot_hashes(fresh, 0, &tokens, 16).unwrap().unwrap();
    drop(kv);
    assert_ne!(fresh_slots, a1_slots, "b no longer keys the group's chain");
    assert_eq!(looked(fresh, &fresh_slots), 0, "b finds none of a's now");
    grants::withdraw();
}

/// Passes over `prompt(2..5)` that arrive while a first pass over
/// `prompt(1)` claims the pages they share, each waiting.
fn waiting_on(
    stores: &Stores,
    prompt: impl Fn(u32) -> Vec<u32>,
) -> Vec<(
    WorkingSetId,
    Vec<u32>,
    tokio::task::JoinHandle<Option<crate::pipeline::coalesce::ClaimGuard>>,
)> {
    (2..5)
        .map(|tail| {
            let tokens = prompt(tail);
            let (ws, slots) = keyed_pass(stores, &tokens);
            let stores = stores.clone();
            let waiting = tokio::spawn(async move {
                crate::pipeline::coalesce::before_pass(&stores, ws, 0, &slots, || {}).await
            });
            (ws, tokens, waiting)
        })
        .collect()
}

async fn settle_tasks() {
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
}

/// Passes over one prompt that arrive while the first computes it wait
/// on its fires' landing, which seals what it froze: they map in all of
/// it, short only of their last row's page. One over another prompt
/// waits on nothing.
#[tokio::test(flavor = "current_thread")]
async fn passes_of_one_prefix_map_what_the_first_lands() {
    let model = crate::store::registry::register_model(16, &[64], &[4]);
    let stores = crate::store::registry::get(model, 0);
    stores.kv.lock().set_keeps(true);
    let prompt = |tail: u32| -> Vec<u32> { (0..64).map(|t| 1300 + t).chain([tail]).collect() };
    let (first, slots) = keyed_pass(&stores, &prompt(1));
    let claims = crate::pipeline::coalesce::before_pass(&stores, first, 0, &slots, || {})
        .await
        .expect("the first claims the pages it computes");
    let waiters = waiting_on(&stores, prompt);
    let other: Vec<u32> = (0..65).map(|t| 1500 + t).collect();
    let (ws, other_slots) = keyed_pass(&stores, &other);
    let own = crate::pipeline::coalesce::before_pass(&stores, ws, 0, &other_slots, || {}).await;
    assert!(
        own.is_some(),
        "another prompt claims its own, waiting on nothing"
    );
    settle_tasks().await;
    assert!(waiters.iter().all(|(_, _, waiting)| !waiting.is_finished()));
    // The first's fires are written and frozen; then they land.
    written(&stores, first, &prompt(1));
    stores.kv.lock().freeze(first, 65).unwrap();
    settle_tasks().await;
    assert!(waiters.iter().all(|(_, _, waiting)| !waiting.is_finished()));
    let sealing = SealOnLanding {
        claims,
        stores: stores.clone(),
        ws: first,
        start: 0,
        slots,
        page_size: 16,
        failure: PipelineFailure::default(),
    };
    sealing.landed(true);
    for (ws, tokens, waiting) in waiters {
        assert!(waiting.await.unwrap().is_none(), "all sealed: none claimed");
        let slots = stores
            .kv
            .lock()
            .slot_hashes(ws, 0, &tokens, 16)
            .unwrap()
            .unwrap();
        let supplied = supply(&stores, ws, &rows(0, &tokens), &slots, Attach::Kv, 65, 16);
        assert_eq!(supplied.rows, 64, "all four sealed pages mapped in");
    }
}

/// A pass cut into fires that land out of order (the last first) seals
/// once all of them have landed, and its waiters map it all in; one
/// whose fire failed seals nothing, however the others landed, and a
/// waiter computes the pages.
#[tokio::test(flavor = "current_thread")]
async fn a_pass_whose_fires_land_out_of_order_seals_once_all_have() {
    let model = crate::store::registry::register_model(16, &[64], &[4]);
    let stores = crate::store::registry::get(model, 0);
    stores.kv.lock().set_keeps(true);
    for (case, failed) in [(0u32, None), (1, Some(0usize))] {
        let prompt = move |tail: u32| -> Vec<u32> {
            (0..64)
                .map(|t| 2100 + case * 100 + t)
                .chain([tail])
                .collect()
        };
        let (first, slots) = keyed_pass(&stores, &prompt(1));
        let claims = crate::pipeline::coalesce::before_pass(&stores, first, 0, &slots, || {})
            .await
            .unwrap();
        let waiters = waiting_on(&stores, prompt);
        written(&stores, first, &prompt(1));
        stores.kv.lock().freeze(first, 65).unwrap();
        let fires: Vec<_> = (0..3)
            .map(|at| crate::engine::WorkItemCompletion::new(900 + at, 0))
            .collect();
        SealOnLanding {
            claims,
            stores: stores.clone(),
            ws: first,
            start: 0,
            slots,
            page_size: 16,
            failure: PipelineFailure::default(),
        }
        .on_landing(&fires);
        let cached = || {
            let (probe, probe_slots) = keyed_pass(&stores, &prompt(9));
            let keys = crate::store::kv::KvStore::fresh_page_keys(&probe_slots, 16);
            let found = stores.kv.lock().cached_run(&keys).len();
            stores.kv.lock().release_working_set(probe, 0);
            found
        };
        for at in [2, 0] {
            resolve_test_completion(&fires[at], failed == Some(at));
            settle_tasks().await;
            assert!(waiters.iter().all(|(_, _, waiting)| !waiting.is_finished()));
            assert_eq!(cached(), 0, "case {case}: not all landed, nothing sealed");
        }
        resolve_test_completion(&fires[1], false);
        settle_tasks().await;
        if failed.is_none() {
            for (ws, tokens, waiting) in waiters {
                assert!(waiting.await.unwrap().is_none());
                let slots = stores
                    .kv
                    .lock()
                    .slot_hashes(ws, 0, &tokens, 16)
                    .unwrap()
                    .unwrap();
                let supplied =
                    supply(&stores, ws, &rows(0, &tokens), &slots, Attach::Kv, 65, 16);
                assert_eq!(supplied.rows, 64, "all four sealed pages mapped in");
            }
        } else {
            assert_eq!(cached(), 0, "a fire failed: nothing sealed");
            let finished: Vec<_> = waiters
                .into_iter()
                .filter(|(_, _, waiting)| waiting.is_finished())
                .collect();
            assert_eq!(finished.len(), 1, "one waiter claims the pages");
            for (_, _, waiting) in finished {
                assert!(waiting.await.unwrap().is_some());
            }
        }
    }
}

/// A pass that does not land what it claimed (it failed, was refused or
/// dropped), or landed it unfrozen, seals none of it and lets its
/// waiters go: one claims the pages and computes them.
#[tokio::test(flavor = "current_thread")]
async fn waiters_on_a_pass_that_seals_nothing_compute_its_pages() {
    let model = crate::store::registry::register_model(16, &[64], &[4]);
    let stores = crate::store::registry::get(model, 0);
    stores.kv.lock().set_keeps(true);
    for (case, landed, frozen) in [(0, false, 65), (1, true, 0), (2, true, 10)] {
        let prompt = move |tail: u32| -> Vec<u32> {
            (0..64)
                .map(|t| 1700 + case * 100 + t)
                .chain([tail])
                .collect()
        };
        let (first, slots) = keyed_pass(&stores, &prompt(1));
        let claims = crate::pipeline::coalesce::before_pass(&stores, first, 0, &slots, || {})
            .await
            .unwrap();
        let waiters = waiting_on(&stores, prompt);
        written(&stores, first, &prompt(1));
        stores.kv.lock().freeze(first, frozen).unwrap();
        settle_tasks().await;
        assert!(waiters.iter().all(|(_, _, waiting)| !waiting.is_finished()));
        let sealing = SealOnLanding {
            claims,
            stores: stores.clone(),
            ws: first,
            start: 0,
            slots,
            page_size: 16,
            failure: PipelineFailure::default(),
        };
        sealing.landed(landed);
        // One claims the pages and the rest wait on it; here it seals
        // nothing either, and the next goes on.
        let (mut pending, mut computing) = (waiters, 0);
        while !pending.is_empty() {
            settle_tasks().await;
            let (done, rest): (Vec<_>, Vec<_>) = pending
                .into_iter()
                .partition(|(_, _, waiting)| waiting.is_finished());
            assert!(!done.is_empty(), "case {case}: a waiter goes on");
            for (_, _, waiting) in done {
                computing += usize::from(waiting.await.unwrap().is_some());
            }
            pending = rest;
        }
        assert!(computing >= 1, "case {case}: a waiter computes the pages");
        let (probe, probe_slots) = keyed_pass(&stores, &prompt(9));
        let keys = crate::store::kv::KvStore::fresh_page_keys(&probe_slots, 16);
        assert_eq!(
            stores.kv.lock().cached_run(&keys).len(),
            0,
            "case {case}: nothing sealed"
        );
        stores.kv.lock().release_working_set(probe, 0);
    }
}

/// A keyed pass of 64 rows whose first 48 the cache holds maps them in
/// and freezes nothing: the set has folded what its passes state,
/// whatever the cache held. A later row written into a mapped page
/// copies it and every mapped page after it.
fn a_pass_supplied_from_the_cache_folds_what_it_states() {
    let model = crate::store::registry::register_model(16, &[16], &[4]);
    let stores = crate::store::registry::get(model, 0);
    let tokens: Vec<u32> = (0..64).map(|t| 400 + t).collect();
    let (slots, _) = sealed_prompt(&stores, &tokens);
    let full = rows(0, &tokens);
    for (fold, states) in [
        (geometry::Fold::Host(vec![0]), 0),
        (geometry::Fold::Host(vec![20]), 20),
        (all(), 64),
    ] {
        let ws = stores.kv.lock().create_working_set();
        let folds = u64::from(fold.of(0).unwrap()).min(64);
        let supplied = supply(&stores, ws, &full, &slots, Attach::Kv, folds, 16);
        assert_eq!(supplied.rows, 48, "short of its last row's page");
        assert_eq!(stores.kv.lock().mapped_len(ws).unwrap(), 3);
        assert_eq!(folded(&stores, ws), (0, true), "mapping freezes nothing");
        // The pass runs from row 48, and folds what it states.
        let cell = cursor(0);
        let pass = lanes_on(&stores, ws, &[&cell]);
        let landings = [landing(0, 64)];
        fire_piece(&stores, ws, None, &pass, &landings, &fold, 48..64, true);
        assert_eq!(folded(&stores, ws), (states, true));
        if states < 48 {
            let mut kv = stores.kv.lock();
            kv.reserve(ws, 4).unwrap();
            assert_eq!(kv.write_demand(ws, &[2]).unwrap(), 1);
            assert_eq!(kv.write_demand(ws, &[1]).unwrap(), 2);
        }
    }
}

/// A recurrent state resumes from a checkpoint, which both halves are
/// then folded through: no further than the rows its pass folds. A
/// checkpoint whose pages the cache no longer holds resumes nothing,
/// and neither half is left folded.
fn a_state_resumes_no_further_than_its_pass_folds() {
    let model = crate::store::registry::register_model(16, &[16], &[4]);
    let stores = crate::store::registry::get(model, 0);
    let geometry = crate::store::rs::RsGeometry {
        state_size: 8,
        buffer_page_tokens: 4,
    };
    let tokens: Vec<u32> = (0..64).map(|t| 800 + t).collect();
    let (slots, keys) = sealed_prompt(&stores, &tokens);
    let checkpoint = |key, frozen: u32| {
        let mut rs = stores.rs.lock().unwrap();
        let ws = rs.create_working_set(geometry);
        let prepared = rs.prepare_group(ws, true, false, None, None, None).unwrap();
        let published = rs.publish_prepared(prepared).unwrap();
        rs.settle(published);
        rs.fold(ws, frozen);
        assert!(rs.checkpoint(ws, key, None).unwrap());
        rs.release_working_set(ws);
    };
    checkpoint(keys[1], 32);
    checkpoint(keys[2], 48);
    let fresh = || {
        (
            stores.kv.lock().create_working_set(),
            stores.rs.lock().unwrap().create_working_set(geometry),
        )
    };
    let both = |ws, rs| {
        (
            stores.kv.lock().folded(ws).unwrap(),
            stores.rs.lock().unwrap().frozen(rs).unwrap(),
        )
    };
    let full = rows(0, &tokens);
    for (folds, resumed) in [(64, 48), (40, 32), (20, 0), (0, 0)] {
        let (ws, rs) = fresh();
        let supplied = supply(
            &stores,
            ws,
            &full,
            &slots,
            Attach::Checkpoint(rs),
            folds,
            16,
        );
        assert_eq!(supplied.rows, resumed, "a pass folding {folds} rows");
        assert_eq!(both(ws, rs), (u64::from(resumed), resumed));
    }
    // The state is kept under a key whose pages are gone.
    checkpoint([9; 32], 32);
    let (ws, rs) = fresh();
    assert_eq!(resume(&stores, ws, rs, &[[8; 32], [9; 32]], 2, 16), 0);
    assert_eq!(both(ws, rs), (0, 0));
    assert_eq!(stores.kv.lock().mapped_len(ws).unwrap(), 0);
}

/// An engine that lands no state inside a fire (`rs_snapshots` 0) has a
/// pass over a record state (a folded state, or a ring-only window) cut
/// at its asked checkpoint row, 32 here: the first piece (0..32) takes it
/// as it settles, and the next (32..64) is laid before then. Taken at
/// the settle, the record is already the next piece's to write in
/// place, so such a take is refused by name.
/// Taken ahead, as the first piece is submitted, the next piece finds
/// the record shared and copies it on the device behind the first: the
/// checkpoint is kept while the next piece is on its way, and a later
/// request maps it.
fn a_record_checkpoint_on_a_cut_is_kept_with_the_next_piece_on_its_way() {
    let model = crate::store::registry::register_model(16, &[16], &[4]);
    let stores = crate::store::registry::get(model, 0);
    let geometry = crate::store::rs::RsGeometry {
        state_size: 8,
        buffer_page_tokens: 4,
    };
    let tokens: Vec<u32> = (0..64).map(|t| 1900 + t).collect();
    let (slots, keys) = sealed_prompt(&stores, &tokens);
    // A piece of 32 rows, folding them, laid and on its way.
    let piece = |rs| {
        let mut store = stores.rs.lock().unwrap();
        let prepared = store
            .prepare_group(rs, true, false, Some(32), None, None)
            .unwrap();
        let state = *prepared.state().unwrap();
        let (published, folds) = store.publish_batch(vec![prepared]).unwrap();
        store.commit_folds(folds);
        (published, state)
    };
    let pass = |ahead: bool| {
        let kv = stores.kv.lock().create_working_set();
        let rs = stores.rs.lock().unwrap().create_working_set(geometry);
        let (first, written) = piece(rs);
        let mut checkpoints = vec![TakeCheckpoint {
            kv,
            rs,
            snapshot: false,
            pages: 2,
            key: keys[1],
        }];
        if ahead {
            checkpoints_ahead(&stores, &mut checkpoints, &mut []);
            assert!(checkpoints[0].snapshot, "forked as the piece is submitted");
        }
        let (next, copied) = piece(rs);
        let mut store = stores.rs.lock().unwrap();
        store.settle(first);
        drop(store);
        (rs, written, next, copied, checkpoints)
    };

    // Taken at the settle: the next piece writes the record in place.
    let (rs, written, next, copied, checkpoints) = pass(false);
    assert_eq!(copied.slot, written.slot, "nothing shared it: written in place");
    let refused = take_checkpoint(&stores, &checkpoints[0], true).unwrap_err();
    assert!(
        refused.contains("a fire on its way writes the state it would keep"),
        "{refused}"
    );
    assert!(!stores.rs.lock().unwrap().has_checkpoint(&keys[1]));
    let mut store = stores.rs.lock().unwrap();
    store.settle(next);
    store.release_working_set(rs);
    drop(store);

    // Taken ahead: the next piece copies the record first.
    let (rs, written, next, copied, checkpoints) = pass(true);
    assert_ne!(copied.slot, written.slot);
    assert_eq!(copied.copy_from, Some(written.slot), "copied on the device");
    assert_eq!(take_checkpoint(&stores, &checkpoints[0], true), Ok(()));
    assert!(
        stores.rs.lock().unwrap().has_checkpoint(&keys[1]),
        "kept while the next piece is on its way"
    );
    let kept = stores.rs.lock().unwrap().frozen(checkpoints[0].rs);
    assert_eq!(kept, Ok(32), "the state as of the cut");
    let ws = stores.kv.lock().create_working_set();
    let resumed = stores.rs.lock().unwrap().create_working_set(geometry);
    let supplied = supply(
        &stores,
        ws,
        &rows(0, &tokens),
        &slots,
        Attach::Checkpoint(resumed),
        64,
        16,
    );
    assert_eq!(supplied.rows, 32, "a later request maps it");
    let mut store = stores.rs.lock().unwrap();
    assert_eq!(store.frozen(resumed), Ok(32));
    assert_eq!(store.folded_slot(resumed), Ok(Some(written.slot)));
    assert_eq!(store.folded_slot(rs), Ok(Some(copied.slot)));
    store.settle(next);
    assert_eq!(store.frozen(rs), Ok(64), "the set went on");
}

/// A checkpoint taken ahead of a fire whose fold the device counts is
/// forked before that fold: it follows the fold, refused as a
/// checkpoint until the count lands, then standing where it left the
/// set. One whose fold went unreported is refused by name.
fn a_checkpoint_taken_ahead_follows_the_fold_the_device_counts() {
    let model = crate::store::registry::register_model(16, &[16], &[4]);
    let stores = crate::store::registry::get(model, 0);
    for reported in [Some(3), None] {
        let kv = stores.kv.lock().create_working_set();
        let rs = state_at(&stores, 4);
        {
            let mut store = stores.rs.lock().unwrap();
            let prepared = store
                .prepare_group(rs, true, false, None, None, None)
                .unwrap();
            let published = store.publish_prepared(prepared).unwrap();
            store.settle(published);
        }
        let key = [0x5a + u8::from(reported.is_none()); 32];
        let mut checkpoints = vec![TakeCheckpoint {
            kv,
            rs,
            snapshot: false,
            pages: 1,
            key,
        }];
        checkpoints_ahead(&stores, &mut checkpoints, &mut []);
        let snapshot = checkpoints[0].rs;
        let mut fold = RsDeviceFold::planned(rs);
        fold.followers.push(snapshot);
        let completion = crate::engine::WorkItemCompletion::deferred_with_guard(None);
        fold.submitted(&stores, &completion);
        assert_eq!(
            stores.rs.lock().unwrap().device_fold_gate(snapshot),
            Err(RsError::DeviceFoldUnsettled)
        );
        let failure = reported.is_none().then_some(RsDeviceFold::UNREPORTED);
        fold.settle(reported, failure);
        let taken = take_checkpoint(&stores, &checkpoints[0], true);
        let store = stores.rs.lock().unwrap();
        match reported {
            Some(count) => {
                assert_eq!(taken, Ok(()));
                assert!(store.has_checkpoint(&key));
                assert_eq!(store.frozen(snapshot), Ok(4 + count));
                assert_eq!(store.frozen(rs), Ok(4 + count));
            }
            None => {
                assert!(taken.unwrap_err().contains("did not report"));
                assert!(!store.has_checkpoint(&key));
            }
        }
    }
}

// A fold the device counts, between its fire's submit and its settle:
// the set is not forked, kept or moved, and what waits for it is woken
// by the fire's end, however it ends.

use super::lifecycle_tests::{TestContext, resolve_test_completion};
use crate::store::rs::{RsError, RsWorkingSetId, device_folds_settled};
use futures::FutureExt;

/// One fire in flight on a pipeline, on a working set frozen at 4 in
/// both stores with 12 slots reserved.
struct Flight {
    stores: Stores,
    fires: PendingFires,
    completion: crate::engine::WorkItemCompletion,
    failure: PipelineFailure,
    kv: KvWorkingSet,
    rs: RsWorkingSetId,
    /// Where the pass's fires are staged from.
    cursor: CursorCell,
    /// What its first lane reports under, where the device counts.
    tag: u64,
}

impl Flight {
    /// The pass's next fire, submitted behind the one in flight: its
    /// completion, and what its first lane reports under.
    fn next(&self, fold: &geometry::Fold) -> (crate::engine::WorkItemCompletion, u64) {
        submit(
            &self.kv,
            Some(self.rs),
            &self.cursor,
            &self.fires,
            &self.failure,
            fold,
        )
    }

    fn fork(&self) -> Result<u32, RsError> {
        let mut rs = self.stores.rs.lock().unwrap();
        let child = rs.fork(self.rs)?;
        rs.frozen(child)
    }

    /// The folded length a KV fork made now takes, and whether it is
    /// all the child has folded.
    fn kv_fork(&self) -> Result<(u64, bool), KvStoreError> {
        let mut kv = self.stores.kv.lock();
        let child = kv.fork(self.kv.id)?;
        Ok((kv.folded(child)?, kv.folded_exact(child)?))
    }

    fn exported(&self) -> Result<u32, RsError> {
        let rs = self.stores.rs.lock().unwrap();
        rs.export(self.rs).map(|_| rs.frozen(self.rs).unwrap())
    }

    /// The set's one folded length, whether it is all the set has
    /// folded, and the fires on their way that fold it by a count
    /// placed when they settle.
    fn length(&self) -> (u64, bool, u32) {
        let kv = self.stores.kv.lock();
        (
            kv.folded(self.kv.id).unwrap(),
            kv.folded_exact(self.kv.id).unwrap(),
            kv.unsettled(self.kv.id).unwrap(),
        )
    }

    fn frozen(&self) -> (u64, u32) {
        (
            self.stores.kv.lock().folded(self.kv.id).unwrap(),
            self.stores.rs.lock().unwrap().frozen(self.rs).unwrap(),
        )
    }

    fn context() -> TestContext {
        TestContext {
            id: uuid::Uuid::new_v4(),
            resources: wasmtime::component::ResourceTable::new(),
        }
    }
}

/// `exact` is whether the host knew the folded length when it planned
/// the fold: not while an earlier fire's fold is on its way unplaced.
/// The fire's eight rows land at slot 4.
fn in_flight(fold: &geometry::Fold, exact: bool) -> Flight {
    in_flight_on(fold, exact, true)
}

/// [`in_flight`]'s fire as the engine took it, its pipeline still to
/// queue it.
fn submitted(fold: &geometry::Fold, exact: bool) -> (Flight, PendingFire, Unqueued) {
    let (model, stores, id) = frozen_at(4);
    let rs = state_at(&stores, 4);
    if !exact {
        stores.kv.lock().unsettle(id).unwrap();
    }
    let cell = cursor(12);
    let kv = KvWorkingSet::new(model, 0, id, 8);
    let failure: PipelineFailure = Arc::new(Mutex::new(None));
    let (completion, tag, fire, unqueued) = stage(&kv, Some(rs), &cell, &failure, fold);
    let flight = Flight {
        stores,
        fires: Arc::new(PendingFireQueue::new()),
        completion,
        failure,
        kv,
        rs,
        cursor: cell,
        tag,
    };
    (flight, fire, unqueued)
}

/// `recurrent` is whether the pass keeps recurrent state: a pass that
/// keeps none folds its set with the recurrent store left out.
fn in_flight_on(fold: &geometry::Fold, exact: bool, recurrent: bool) -> Flight {
    let (model, stores, id) = frozen_at(4);
    let rs = state_at(&stores, 4);
    if !exact {
        stores.kv.lock().unsettle(id).unwrap();
    }
    let cell = cursor(12);
    let kv = KvWorkingSet::new(model, 0, id, 8);
    let failure: PipelineFailure = Arc::new(Mutex::new(None));
    let fires = Arc::new(PendingFireQueue::new());
    let part = recurrent.then_some(rs);
    let (completion, tag) = submit(&kv, part, &cell, &fires, &failure, fold);
    Flight {
        stores,
        fires,
        completion,
        failure,
        kv,
        rs,
        cursor: cell,
        tag,
    }
}

/// A fire of a pass on `kv` and `rs`, submitted: its fold placed on the
/// folded length the set states now (frozen as it is prepared, where
/// the host knows the count), its set marked, queued on `fires` behind
/// the pass's fires in flight. Its eight rows land at slot 4. Its
/// completion, and what its first lane reports under, where the device
/// counts.
fn submit(
    kv: &KvWorkingSet,
    rs: Option<RsWorkingSetId>,
    cursor: &CursorCell,
    fires: &PendingFires,
    failure: &PipelineFailure,
    fold: &geometry::Fold,
) -> (crate::engine::WorkItemCompletion, u64) {
    let (completion, tag, fire, unqueued) = stage(kv, rs, cursor, failure, fold);
    unqueued.queue(fires, fire);
    (completion, tag)
}

/// [`submit`] up to the queue: the fire as the engine took it, its
/// pipeline still to queue it.
fn stage(
    kv: &KvWorkingSet,
    rs: Option<RsWorkingSetId>,
    cursor: &CursorCell,
    failure: &PipelineFailure,
    fold: &geometry::Fold,
) -> (
    crate::engine::WorkItemCompletion,
    u64,
    PendingFire,
    Unqueued,
) {
    let stores = crate::store::registry::get(kv.model, kv.engine);
    let pass = lanes_on(&stores, kv.id, &[cursor]);
    let mut req = request(1);
    let mut tags = Vec::new();
    let by = fold_at(
        &pass,
        &[0],
        &[landing(4, 8)],
        fold,
        u64::MAX,
        true,
        rs,
        &mut req,
        &mut tags,
    )
    .expect("the fire folds");
    // Its prepare: a count the host knows is frozen now, in the
    // recurrent state as the fire is reserved and in the KV with its
    // log begun.
    let freeze = frozen_as_prepared(Some(&by));
    if let (Some(to), Some(rs)) = (freeze, rs) {
        let mut store = stores.rs.lock().unwrap();
        let frozen = u64::from(store.frozen(rs).unwrap());
        store.fold(rs, to.saturating_sub(frozen) as u32);
    }
    let start = stores.kv.lock().folded(kv.id).unwrap();
    begin_piece(&mut stores.kv.lock(), kv.id, start, freeze)
        .unwrap()
        .expect("it writes past what is frozen");
    let completion = crate::engine::WorkItemCompletion::deferred_with_guard(None);
    let mut parts = vec![FireWs {
        ws: kv.id,
        kvtxn: KvTxnGuard::new(kv.model, kv.engine, None),
        log: None,
        lease: kv.fire_lease().expect("lease the submitted working set"),
        fold: Some(FoldAt { by, decoded: false }),
        reports: tags,
        staged: None,
    }];
    // The submit.
    mark_device_folds(&stores, &completion, &mut parts);
    unsettle(&stores, &completion, &mut parts);
    let unqueued = Unfired::new(failure).submitted();
    let fire = PendingFire {
        completion: completion.clone(),
        checkpoints: Vec::new(),
        kv: parts,
        device_geometry: true,
        rstxn: RsTxnsGuard::new(kv.model, kv.engine, None),
        model: kv.model,
        engine: kv.engine,
        fwd_rep: u32::MAX,
        cells: Vec::new(),
        failure: failure.clone(),
        retired: None,
        lane: None,
    };
    // Its rows land where the device says: the cursor no longer knows
    // where they end.
    advance_cursor(cursor, None, None);
    (completion, req.lanes[0].reports, fire, unqueued)
}

/// A process at its end, as the SDK's `run_ahead` leaves it: its
/// pipeline closed with its last fire still in flight, a fire that folds
/// by a count the device computes a set that keeps no recurrent state,
/// frozen at 4 and still in the process's table.
pub(crate) struct Ending {
    resources: wasmtime::component::ResourceTable,
    pub(crate) residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
    /// The set's rep in the table.
    pub(crate) ws: u32,
    /// Its KV working set, whose folded length is the set's.
    kv_id: WorkingSetId,
    pub(crate) stores: crate::store::registry::Stores,
    completion: crate::engine::WorkItemCompletion,
    tag: u64,
    /// Stands where the pass does. A pass dropped with a fire on its way
    /// is handed to its last pending fire ([`retire_pass`]) and keeps
    /// its pipeline's queue until that fire is finalized; no pass can
    /// be bound without an engine.
    _pass: PendingFires,
}

impl Ending {
    /// Its KV working set.
    pub(crate) fn kv_id(&self) -> WorkingSetId {
        self.kv_id
    }

    /// The process's table, as its end hands it to the teardown.
    pub(crate) fn table(&mut self) -> wasmtime::component::ResourceTable {
        std::mem::replace(
            &mut self.resources,
            wasmtime::component::ResourceTable::new(),
        )
    }

    /// The device ran the fire to its end and counted `fold` more
    /// slots frozen.
    pub(crate) fn settle(&self, fold: u32) {
        record_kv_writes(vec![report(self.tag, fold)]);
        resolve_test_completion(&self.completion, false);
    }
}

/// `dropped` is whether the guest dropped its pipeline too, as the end
/// of its `main` does, or only closed it.
pub(crate) async fn ending(dropped: bool) -> Anyhow<Ending> {
    use crate::inferlet::process::residency::ResidentPipeline;
    let model = crate::store::registry::register_model(4, &[8], &[1]);
    let stores = crate::store::registry::get(model, 0);
    let id = {
        let mut kv = stores.kv.lock();
        let id = kv.create_working_set();
        kv.fold(id, 4, false).unwrap();
        id
    };
    let kv = KvWorkingSet::new(model, 0, id, 8);
    let cell = cursor(12);
    let mut ctx = Flight::context();
    let pipeline = ctx.resources.push(Pipeline::new())?;
    let rep = pipeline.rep();
    let (fires, failure, scope) = {
        let pipeline = ctx.resources.get(&pipeline)?;
        (
            pipeline.fires.clone(),
            pipeline.failure.clone(),
            pipeline.scope.clone(),
        )
    };
    // As the host registers a pipeline its guest makes.
    let residency = Arc::new(Mutex::new(
        crate::inferlet::process::ProcessResidency::default(),
    ));
    residency.lock().unwrap().pipelines.push(ResidentPipeline {
        scope: scope.clone(),
        fires: Arc::downgrade(&fires),
    });
    assert_eq!(kv.claim_pipeline_scope(&scope), Ok(()));
    let fold = geometry::Fold::Device;
    let (completion, tag) = submit(&kv, None, &cell, &fires, &failure, &fold);
    pipeline_close(&mut ctx, Resource::new_borrow(rep)).await?;
    if dropped {
        pipeline_drop(&mut ctx, pipeline).await?;
    }
    assert!(
        !fires.lock().unwrap().is_empty() && stores.kv.lock().unsettled(id).unwrap() == 1,
        "a close does not wait for a fire in flight"
    );
    let kv = ctx.resources.push(kv)?.rep();
    let ws = crate::inferlet::host::working_set::WorkingSet {
        kv,
        rs: None,
        cursor: cell.clone(),
    };
    let ws = ctx.resources.push(ws)?.rep();
    Ok(Ending {
        resources: ctx.resources,
        residency,
        ws,
        kv_id: id,
        stores,
        completion,
        tag,
        _pass: fires,
    })
}

/// The in-pass fork: the guest forks a set between its own pipelined
/// fires. The host fork runs the pipeline's fires to their end
/// ([`drain_pipeline_fires`], which only its own task can); a handle's
/// fork, a save and the planner's suspend wait on the store and on the
/// fire's lease. None ends before the fire settles, and each then reads
/// the folded length the device's count left, one length in both
/// stores.
/// A soak's stall: a process fires, then binds a
/// program short of room, and the planner evicts it while its bind
/// waits. The fire's lease goes back only as the fire is finalized,
/// and its finalizer is the guest, in the bind; the eviction's drain
/// finalizes only fires settled as it starts, and none of a pass whose
/// geometry the device keeps, so the eviction would wait on the lease
/// for good.
/// A bind about to wait settles its process's fires first: the lease
/// goes back as the fire lands, and the eviction's wait ends.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_about_to_wait_settles_its_fires_and_an_eviction_s_wait_for_their_leases_ends()
-> Anyhow<()> {
    use crate::inferlet::process::residency::{ResidentPipeline, register_residency};
    use futures::FutureExt;
    let model = crate::store::registry::register_model(4, &[8], &[1]);
    let stores = crate::store::registry::get(model, 0);
    let id = {
        let mut kv = stores.kv.lock();
        let id = kv.create_working_set();
        kv.fold(id, 4, false).unwrap();
        id
    };
    let kv = KvWorkingSet::new(model, 0, id, 8);
    let cell = cursor(12);
    let pipeline = Pipeline::new();
    let (fires, failure, scope) = (
        pipeline.fires.clone(),
        pipeline.failure.clone(),
        pipeline.scope.clone(),
    );
    let pid = uuid::Uuid::new_v4();
    let residency = Arc::new(Mutex::new(
        crate::inferlet::process::ProcessResidency::default(),
    ));
    residency.lock().unwrap().pipelines.push(ResidentPipeline {
        scope: scope.clone(),
        fires: Arc::downgrade(&fires),
    });
    register_residency(pid, Arc::downgrade(&residency));
    assert_eq!(kv.claim_pipeline_scope(&scope), Ok(()));
    // A fire of a pass whose geometry the device keeps, on its way.
    let (completion, tag) = submit(&kv, None, &cell, &fires, &failure, &geometry::Fold::Device);
    assert!(!fires.lock().unwrap().front().unwrap().is_preemption_detachable());

    // The eviction: its sets fenced, then it waits for their leases.
    let handle = kv.suspend_handle();
    handle.fence();
    assert_eq!(handle.active_leases(), 1, "the fire holds its lease");
    let mut quiesced = Box::pin(handle.quiesce());
    assert!(quiesced.as_mut().now_or_never().is_none());

    // The bind, about to wait, settles what its process has in flight.
    let mut settled = Box::pin(settle_process_fires(pid));
    assert!(
        settled.as_mut().now_or_never().is_none(),
        "it waits for the fire to land, not for anything the eviction holds"
    );
    assert_eq!(handle.active_leases(), 1);
    // The fire lands.
    record_kv_writes(vec![report(tag, 2)]);
    resolve_test_completion(&completion, false);
    settled.await;
    assert!(fires.lock().unwrap().is_empty(), "finalized");
    assert_eq!(handle.active_leases(), 0, "its lease is back");
    quiesced.await;
    assert_eq!(
        stores.kv.lock().folded(id).unwrap(),
        6,
        "its fold is placed as its own finalize places it"
    );
    assert_eq!(*failure.lock().unwrap(), None, "a fire that ran fails nothing");
    drop(residency);
    Ok(())
}

/// A device-geometry fire that fails, finalized by a bind's settle
/// ([`settle_process_fires`]) rather than by its guest, latches its
/// failure as the eviction's drain does ([`finalize_op_detached`]): its
/// pipeline is failed by name, so the guest's next submit on it (which
/// reads the pipeline's failure before its pass's, and a pass never
/// submits on another pipeline) is refused by that name, and its
/// readers are poisoned with it. Never success, never a silent drop.
#[tokio::test(flavor = "current_thread")]
async fn a_failed_fire_settled_by_a_bind_latches_its_failure_as_the_eviction_s_drain_does()
-> Anyhow<()> {
    use crate::inferlet::process::residency::{ResidentPipeline, register_residency};
    use crate::pipeline::channel::ChannelCell;
    // `by_bind`: the bind's settle finalizes it; else the eviction's drain.
    let mut outcomes = Vec::new();
    for by_bind in [true, false] {
        let model = crate::store::registry::register_model(4, &[8], &[1]);
        let stores = crate::store::registry::get(model, 0);
        let id = {
            let mut kv = stores.kv.lock();
            let id = kv.create_working_set();
            kv.fold(id, 4, false).unwrap();
            id
        };
        let kv = KvWorkingSet::new(model, 0, id, 8);
        let mut ctx = Flight::context();
        let pipeline = ctx.resources.push(Pipeline::new())?;
        let (fires, failure, scope) = {
            let pipeline = ctx.resources.get(&pipeline)?;
            (
                pipeline.fires.clone(),
                pipeline.failure.clone(),
                pipeline.scope.clone(),
            )
        };
        let pid = uuid::Uuid::new_v4();
        let residency = Arc::new(Mutex::new(
            crate::inferlet::process::ProcessResidency::default(),
        ));
        residency.lock().unwrap().pipelines.push(ResidentPipeline {
            scope: scope.clone(),
            fires: Arc::downgrade(&fires),
        });
        register_residency(pid, Arc::downgrade(&residency));
        assert_eq!(kv.claim_pipeline_scope(&scope), Ok(()));
        let reader = Arc::new(Mutex::new(ChannelCell::new(vec![1], eta_ir::types::Dtype::U32, 2)));
        reader.lock().unwrap().role = Some(HostRole::Reader);
        let (completion, _, mut fire, unqueued) =
            stage(&kv, None, &cursor(12), &failure, &geometry::Fold::Device);
        fire.cells = vec![reader.clone()];
        assert!(fire.device_geometry);
        // The bind settles a device-geometry fire; the eviction drains only
        // a host one.
        fire.device_geometry = by_bind;
        unqueued.queue(&fires, fire);
        let handle = kv.suspend_handle();
        // The engine fails the fire.
        resolve_test_completion(&completion, true);
        if by_bind {
            settle_process_fires(pid).await;
        } else {
            let op = fires.lock().unwrap().pop_front().expect("queued");
            assert!(op.is_preemption_detachable());
            finalize_op_detached(op).await?;
        }
        assert!(fires.lock().unwrap().is_empty(), "finalized, not dropped unread");
        assert_eq!(handle.active_leases(), 0, "its lease is back");
        let latched = failure.lock().unwrap().clone().expect("the failure is latched");
        assert!(latched.contains("test callback failure"), "{latched}");
        // The guest's next submit on the pipeline: refused by name.
        let fwd: Resource<ForwardPass> = Resource::new_borrow(u32::MAX);
        let next = open_pipeline(&mut ctx, &Resource::new_borrow(pipeline.rep()), &fwd).await?;
        let Err(refused) = next else {
            panic!("a failed pipeline takes no submit");
        };
        assert!(refused.contains(&latched), "{refused}");
        // Its reader: poisoned by the same name.
        let read = reader.lock().unwrap().take();
        assert_eq!(read, Err(ChannelError::Poisoned(latched.clone())));
        outcomes.push((latched, refused));
        drop(residency);
    }
    assert_eq!(outcomes[0], outcomes[1], "the bind's settle and the eviction's drain alike");
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn what_copies_or_moves_a_set_waits_for_the_fold_the_device_counts() -> Anyhow<()> {
    let flight = in_flight(&geometry::Fold::Device, true);
    let mut ctx = Flight::context();
    assert_eq!(flight.fork(), Err(RsError::DeviceFoldUnsettled));
    assert_eq!(flight.kv_fork(), Err(UNSETTLED));
    assert_eq!(flight.exported(), Err(RsError::DeviceFoldUnsettled));
    let sets = [flight.rs];
    let mut waiting = Box::pin(device_folds_settled(&flight.stores.rs, &sets));
    assert!(waiting.as_mut().now_or_never().is_none());
    let suspend = flight.kv.suspend_handle();
    let mut quiesce = Box::pin(suspend.quiesce());
    assert!(quiesce.as_mut().now_or_never().is_none());
    {
        let mut drain = Box::pin(drain_pipeline_fires(&mut ctx, &flight.fires));
        assert!(
            drain.as_mut().now_or_never().is_none(),
            "the fork goes no further than the fire in flight"
        );
        assert_eq!(flight.frozen(), (4, 4), "what a copy made now would state");
        record_kv_writes(vec![report(flight.tag, 3)]);
        resolve_test_completion(&flight.completion, false);
        drain.await?;
    }
    assert!(waiting.as_mut().now_or_never().is_some());
    assert!(quiesce.as_mut().now_or_never().is_some());
    assert_eq!(flight.frozen(), (7, 7));
    assert_eq!(flight.fork(), Ok(7), "the child takes what the fold left");
    assert_eq!(flight.kv_fork(), Ok((7, true)));
    assert_eq!(flight.exported(), Ok(7));
    assert!(flight.failure.lock().unwrap().is_none());
    Ok(())
}

/// A set that keeps no recurrent state is folded by the device's count
/// too, and nothing in the recurrent store marks it: its cursor does.
/// The fork reads the cursor only at rest ([`cursor_at_rest`]): it goes
/// no further than the fire in flight, nor than one another finalizer
/// holds off the queue, and the child then starts from the frozen
/// length the count left, the one its pages are frozen to.
#[tokio::test(flavor = "current_thread")]
async fn a_set_with_no_recurrent_state_forks_at_the_length_its_fold_leaves() -> Anyhow<()> {
    // Who settles the fire: the fork's own drain, or another finalizer
    // that took it off the queue under the gate before the fork began.
    for other_finalizer in [false, true] {
        let flight = in_flight_on(&geometry::Fold::Device, true, false);
        let (mut ctx, mut other) = (Flight::context(), Flight::context());
        assert_eq!(
            flight.stores.rs.lock().unwrap().device_fold_gate(flight.rs),
            Ok(()),
            "the recurrent store has nothing of this pass to mark"
        );
        let staged = flight.cursor.lock().unwrap().clone();
        assert!(
            flight.length() == (4, false, 1),
            "what a child made now would state for good: {staged:?}"
        );
        let window = if other_finalizer {
            record_kv_writes(vec![report(flight.tag, 3)]);
            resolve_test_completion(&flight.completion, false);
            let gate = flight.fires.finalize_guard().await;
            let op = pop_settled(Some(&flight.fires)).expect("the fire settled");
            let finalized = finalize_op_await(op).await;
            assert!(flight.fires.lock().unwrap().is_empty());
            Some((gate, finalized))
        } else {
            None
        };
        let child = {
            let mut fork = Box::pin(cursor_at_rest(
                &mut ctx,
                &flight.fires,
                &flight.failure,
                &flight.kv,
                None,
                &flight.cursor,
            ));
            assert!(
                fork.as_mut().now_or_never().is_none(),
                "the fork goes no further than the fire: {other_finalizer}"
            );
            match window {
                Some((gate, finalized)) => {
                    complete_finalize(&mut other, finalized);
                    drop(gate);
                }
                None => {
                    record_kv_writes(vec![report(flight.tag, 3)]);
                    resolve_test_completion(&flight.completion, false);
                }
            }
            fork.await?.expect("the fold was placed")
        };
        assert!(flight.length() == (7, true, 0), "{child:?}");
        let pages = {
            let mut kv = flight.stores.kv.lock();
            let forked = kv.fork(flight.kv.id).unwrap();
            kv.folded(forked).unwrap()
        };
        assert_eq!(
            flight.length().0,
            pages,
            "the child states the length its pages are frozen to"
        );
        // At rest there is nothing to wait for.
        let again = cursor_at_rest(
            &mut ctx,
            &flight.fires,
            &flight.failure,
            &flight.kv,
            None,
            &flight.cursor,
        );
        let again = again.now_or_never().expect("no fire is on its way")?;
        assert_eq!(again.map(|_| flight.length().0), Ok(7));
    }
    Ok(())
}

/// A fold no fire is left to place never lets the cursor come to rest:
/// the fork does not hang on it, and does not hand out the bound. It is
/// refused by name where the fire was dropped unsettled or ran and
/// brought no count (the recurrent store's name for it, where the set
/// keeps recurrent state), and by the pipeline's failure where the fire
/// failed: also one another finalizer holds off the queue, its failure
/// not latched yet.
#[tokio::test(flavor = "current_thread")]
async fn a_fold_that_is_never_placed_refuses_the_fork_by_name() -> Anyhow<()> {
    for ended in ["dropped", "no count", "failed", "failed, off the queue"] {
        let flight = in_flight_on(&geometry::Fold::Device, true, false);
        let (mut ctx, mut other) = (Flight::context(), Flight::context());
        let mut window = None;
        match ended {
            "dropped" => flight.fires.lock().unwrap().clear(),
            "no count" => resolve_test_completion(&flight.completion, false),
            "failed" => resolve_test_completion(&flight.completion, true),
            _ => {
                resolve_test_completion(&flight.completion, true);
                let gate = flight.fires.finalize_guard().await;
                let op = pop_settled(Some(&flight.fires)).expect("the fire settled");
                let finalized = finalize_op_await(op).await;
                assert_eq!(flight.failure.lock().unwrap().as_ref(), None);
                window = Some((gate, finalized));
            }
        }
        let mut fork = Box::pin(cursor_at_rest(
            &mut ctx,
            &flight.fires,
            &flight.failure,
            &flight.kv,
            None,
            &flight.cursor,
        ));
        if let Some((gate, finalized)) = window {
            assert!(
                fork.as_mut().now_or_never().is_none(),
                "its failure is not written yet"
            );
            complete_finalize(&mut other, finalized);
            drop(gate);
        }
        let refused = fork
            .as_mut()
            .now_or_never()
            .unwrap_or_else(|| panic!("{ended}: it never hangs"))?
            .expect_err("the frozen length is not known");
        let staged = flight.cursor.lock().unwrap().clone();
        assert!(
            flight.length().0 == 4 && !flight.length().1,
            "{ended}: {staged:?}"
        );
        match ended {
            "dropped" | "no count" => {
                assert_eq!(
                    refused,
                    format!("working set: {}", crate::store::kv::page_table::UNPLACED)
                );
                assert!(flight.failure.lock().unwrap().is_none(), "{ended}");
            }
            _ => assert!(
                refused.starts_with("pipeline failed: pipeline: forward failed"),
                "{ended}: {refused}"
            ),
        }
    }
    // A set that keeps recurrent state is refused as its store names
    // how the fire ended.
    let flight = in_flight(&geometry::Fold::Device, true);
    let mut ctx = Flight::context();
    resolve_test_completion(&flight.completion, false);
    let fork = cursor_at_rest(
        &mut ctx,
        &flight.fires,
        &flight.failure,
        &flight.kv,
        Some(flight.rs),
        &flight.cursor,
    );
    let refused = fork.now_or_never().expect("it never hangs")?;
    let unreported = RsError::DeviceFoldFailed(RsDeviceFold::UNREPORTED.to_string());
    assert_eq!(refused.map(|_| ()), Err(unreported.to_string()));
    Ok(())
}

/// A pass that fails releases whoever waited, and what they asked for
/// is refused by the pass's own failure: the engine's, or that of the
/// pipeline's earlier fire.
#[tokio::test(flavor = "current_thread")]
async fn a_failed_pass_ends_the_wait_with_its_failure() -> Anyhow<()> {
    for earlier in [None, Some("pipeline: an earlier fire failed")] {
        let flight = in_flight(&geometry::Fold::Device, true);
        let mut ctx = Flight::context();
        let sets = [flight.rs];
        let mut waiting = Box::pin(device_folds_settled(&flight.stores.rs, &sets));
        assert!(waiting.as_mut().now_or_never().is_none());
        *flight.failure.lock().unwrap() = earlier.map(str::to_string);
        record_kv_writes(vec![report(flight.tag, 3)]);
        resolve_test_completion(&flight.completion, earlier.is_none());
        drain_pipeline_fires(&mut ctx, &flight.fires).await?;
        assert!(waiting.as_mut().now_or_never().is_some(), "it never hangs");
        let reason = flight
            .failure
            .lock()
            .unwrap()
            .clone()
            .expect("the pipeline failed: the host fork refuses by this");
        match earlier {
            Some(earlier) => assert_eq!(reason, earlier),
            None => assert!(
                reason.starts_with("pipeline: forward failed") && reason.contains("test"),
                "{reason}"
            ),
        }
        assert_eq!(
            flight.fork(),
            Err(RsError::DeviceFoldFailed(reason.clone()))
        );
        assert_eq!(flight.exported(), Err(RsError::DeviceFoldFailed(reason)));
        assert_eq!(
            flight.frozen(),
            (4, 4),
            "a count from a failed pass is not taken"
        );
        assert_eq!(
            flight.kv_fork(),
            Err(KvStoreError::Table(KvTableError::FoldUnplaced)),
            "and the length is a bound, which no copy states"
        );
    }
    Ok(())
}

/// A fire that fails ends its pipeline and the sets the pipeline owns:
/// closed and drained, it passes none of them to another pipeline, and
/// the refusal names it and its first failure. The cursor the failed
/// fire moved when it was staged is not moved back, which is why: here
/// its fold stays counted as on its way and its tail unknown. A set
/// whose pipeline's fires all served passes on as it did. Neither
/// passes on while its fire, out of the queue, is still being
/// finalized.
#[tokio::test(flavor = "current_thread")]
async fn a_set_of_a_failed_pipeline_fires_on_no_other() -> Anyhow<()> {
    use crate::store::{PipelineScope, ScopeRefusal};
    for failed in [true, false] {
        let flight = in_flight(&geometry::Fold::Device, true);
        let mut ctx = Flight::context();
        let fires = Arc::downgrade(&flight.fires);
        let own = PipelineScope::failing_as(flight.failure.clone(), move || {
            fires.upgrade().is_none_or(|fires| fires.drained())
        });
        let rs = RsWorkingSet::new(
            flight.kv.model,
            0,
            flight.rs,
            crate::store::rs::RsGeometry {
                state_size: 8,
                buffer_page_tokens: 4,
            },
        );
        assert_eq!(flight.kv.claim_pipeline_scope(&own), Ok(()));
        assert_eq!(rs.claim_pipeline_scope(&own), Ok(()));

        // A fire that failed before it ran reports nothing.
        if !failed {
            record_kv_writes(vec![report(flight.tag, 3)]);
        }
        resolve_test_completion(&flight.completion, failed);
        own.close();
        let next = PipelineScope::new(|| true);

        // The fire leaves its pipeline's queue ahead of its failure
        // being written, as every finalizer takes it: under the gate.
        // Until the gate lifts the set is still its pipeline's, however
        // the fire ended.
        {
            let _gate = flight.fires.finalize_guard().await;
            let op = pop_settled(Some(&flight.fires)).expect("the fire settled");
            let finalized = finalize_op_await(op).await;
            assert!(flight.fires.lock().unwrap().is_empty());
            assert_eq!(flight.failure.lock().unwrap().as_ref(), None);
            let held = Err(ScopeRefusal::Held(own.id()));
            assert_eq!(flight.kv.claim_pipeline_scope(&next), held);
            assert_eq!(rs.claim_pipeline_scope(&next), held);
            complete_finalize(&mut ctx, finalized);
        }

        let staged = flight.cursor.lock().unwrap().clone();
        if !failed {
            assert_eq!(flight.length(), (7, true, 0));
            assert_eq!(flight.kv.claim_pipeline_scope(&next), Ok(()));
            assert_eq!(rs.claim_pipeline_scope(&next), Ok(()));
            continue;
        }
        assert!(
            !flight.length().1 && staged.tail.is_none(),
            "the set stands as the failed fire staged it: {staged:?}, {:?}",
            flight.length()
        );
        let reason = flight.failure.lock().unwrap().clone().expect("it failed");
        let refused = ScopeRefusal::Failed {
            owner: own.id(),
            reason,
        };
        assert_eq!(flight.kv.claim_pipeline_scope(&next), Err(refused.clone()));
        assert_eq!(rs.claim_pipeline_scope(&next), Err(refused.clone()));
        let text = refused.to_string();
        assert!(
            text.contains("which failed (pipeline: forward failed") && text.contains("test"),
            "{text}"
        );
    }
    Ok(())
}

/// A fire that ran to its end on a set whose fold the device counts
/// owes the count: the engine's record went on by it. One that brings
/// none, or its rows' end without it, lifts the mark and leaves the set
/// refused by name, where a count of nothing is a count. A fire that
/// failed keeps its own failure.
#[tokio::test(flavor = "current_thread")]
async fn a_fire_that_ran_and_reported_no_fold_fails_its_set_by_name() -> Anyhow<()> {
    type Brought = fn(u64) -> Option<::engine::KvWrites>;
    let unreported = || RsError::DeviceFoldFailed(RsDeviceFold::UNREPORTED.to_string());
    // What the engine brought under the lane's tag, whether its fire
    // ran, and where the fold left the set frozen, where it is known.
    let cases: [(&str, Brought, bool, Option<u32>); 5] = [
        ("no report", |_| None, true, None),
        (
            "the rows' end alone",
            |tag| {
                Some(::engine::KvWrites {
                    tag,
                    end: Some(9),
                    ..::engine::KvWrites::default()
                })
            },
            true,
            None,
        ),
        (
            "a fold of nothing",
            |tag| Some(report(tag, 0)),
            true,
            Some(4),
        ),
        ("a fold of three", |tag| Some(report(tag, 3)), true, Some(7)),
        ("no report from a fire that failed", |_| None, false, None),
    ];
    for (what, brought, ran, left) in cases {
        let flight = in_flight(&geometry::Fold::Device, true);
        let mut ctx = Flight::context();
        let sets = [flight.rs];
        let mut waiting = Box::pin(device_folds_settled(&flight.stores.rs, &sets));
        assert!(waiting.as_mut().now_or_never().is_none(), "{what}");
        if let Some(written) = brought(flight.tag) {
            record_kv_writes(vec![written]);
        }
        resolve_test_completion(&flight.completion, !ran);
        drain_pipeline_fires(&mut ctx, &flight.fires).await?;
        assert!(
            waiting.as_mut().now_or_never().is_some(),
            "{what}: the mark lifts however the fire ended"
        );
        let failed = flight.failure.lock().unwrap().clone();
        assert_eq!(
            failed.is_none(),
            ran,
            "{what}: the pass fails only by its fire"
        );
        match (left, failed) {
            (Some(left), _) => {
                assert_eq!(flight.frozen(), (u64::from(left), left), "{what}");
                assert_eq!(flight.fork(), Ok(left), "{what}");
                assert_eq!(flight.exported(), Ok(left), "{what}");
                assert!(flight.length().1, "{what}");
            }
            (None, None) => {
                assert_eq!(flight.frozen(), (4, 4), "{what}: nothing is frozen");
                assert_eq!(flight.fork(), Err(unreported()), "{what}");
                assert_eq!(flight.exported(), Err(unreported()), "{what}");
                assert!(!flight.length().1, "{what}: its fires go on from the bound");
            }
            (None, Some(reason)) => {
                assert_ne!(reason, RsDeviceFold::UNREPORTED, "{what}");
                assert_eq!(
                    flight.fork(),
                    Err(RsError::DeviceFoldFailed(reason)),
                    "{what}: the fire's own failure names it"
                );
            }
        }
    }
    Ok(())
}

/// A fold the host counts over a folded length it knows is frozen as
/// its fire is prepared, in both stores: nothing waits on the fire,
/// and a copy made while it is in flight takes the length it froze.
/// Behind a fold on its way unplaced it is placed when its own fire
/// settles.
#[tokio::test(flavor = "current_thread")]
async fn a_fold_the_host_counts_delays_nothing() -> Anyhow<()> {
    let host = geometry::Fold::Host(vec![2]);
    for exact in [true, false] {
        let flight = in_flight(&host, exact);
        let mut ctx = Flight::context();
        let sets = [flight.rs];
        assert_eq!(flight.tag, 0, "the host needs no report");
        assert!(
            device_folds_settled(&flight.stores.rs, &sets)
                .now_or_never()
                .is_some()
        );
        if exact {
            assert_eq!(flight.frozen(), (6, 6));
            assert_eq!(flight.fork(), Ok(6));
            assert_eq!(flight.kv_fork(), Ok((6, true)));
            assert_eq!(flight.exported(), Ok(6));
        } else {
            assert_eq!(flight.frozen().0, 4);
            assert_eq!(flight.kv_fork(), Err(UNSETTLED));
        }
        let mut drain = Box::pin(drain_pipeline_fires(&mut ctx, &flight.fires));
        assert!(
            drain.as_mut().now_or_never().is_none(),
            "while its fire is still in flight"
        );
        resolve_test_completion(&flight.completion, false);
        drain.await?;
        assert_eq!(flight.frozen().0, 6);
    }
    Ok(())
}

/// A fire dropped before anything settled it (its process torn down)
/// still lifts the mark: when the engine is done with it, or when the
/// engine itself is gone and its completion with it. What it folded is
/// then not known, and the set says so.
#[test]
fn a_fire_dropped_unsettled_lifts_the_mark_when_the_engine_is_done_with_it() {
    let dropped = || RsError::DeviceFoldFailed(RsDeviceFold::DROPPED.to_string());
    for lost in [false, true] {
        let Flight {
            stores,
            fires,
            completion,
            kv,
            rs,
            ..
        } = in_flight(&geometry::Fold::Device, true);
        let sets = [rs];
        let mut waiting = Box::pin(device_folds_settled(&stores.rs, &sets));
        drop(fires);
        assert_eq!(
            stores.rs.lock().unwrap().device_fold_gate(rs),
            Err(RsError::DeviceFoldUnsettled),
            "the engine may still run it"
        );
        assert_eq!(stores.kv.lock().fork(kv.id), Err(UNSETTLED));
        assert!(waiting.as_mut().now_or_never().is_none());
        if lost {
            drop(completion);
        } else {
            resolve_test_completion(&completion, false);
        }
        assert!(waiting.as_mut().now_or_never().is_some());
        assert_eq!(
            stores.rs.lock().unwrap().device_fold_gate(rs),
            Err(dropped())
        );
        // The KV's folded length is a bound from then on, and waits
        // on nothing: a copy is refused by that name.
        assert_eq!(folded(&stores, kv.id), (4, false));
        assert_eq!(stores.kv.lock().unsettled(kv.id).unwrap(), 0);
        assert_eq!(stores.kv.lock().fork(kv.id), Err(UNPLACED));
    }
}

// A pass that runs ahead submits a frame's fires one after another, and
// the scheduler dispatches none of them before the frame's last has
// arrived. A fire behind one that folds by a count the device computes
// is therefore staged before that count exists.

use crate::scheduler::frame::{FramePlan, FramePolicy, FrameStamp, QueuedFireIds};

/// The scheduler's frames, two fires each, as one lane fills them.
struct Frames {
    policy: FramePolicy,
    lane: uuid::Uuid,
    arrived: Vec<u64>,
}

impl Frames {
    fn of_two(lane: uuid::Uuid) -> Self {
        Self {
            policy: FramePolicy::new(2, 64, 4096, None),
            lane,
            arrived: Vec::new(),
        }
    }

    /// The fire in `slot` of the lane's first frame reaches the
    /// scheduler, under its slot as its id.
    fn arrive(&mut self, slot: u32) {
        let stamp = FrameStamp {
            lane: self.lane,
            seq: 0,
            slot,
            fires: 2,
        };
        let fire = u64::from(slot);
        self.policy
            .on_fire_enqueued(stamp, Some(self.lane), fire, 1, 1);
        self.arrived.push(fire);
    }

    /// The fires the scheduler dispatches now, wave by wave: none while
    /// it holds the frame for a fire still to arrive.
    fn dispatched(&mut self) -> Vec<Vec<u64>> {
        let queued: QueuedFireIds = self.arrived.iter().copied().collect();
        let none_blocked = std::collections::HashSet::new();
        match self.policy.plan_dispatch(
            &queued,
            &none_blocked,
            false,
            crate::rt::Instant::now(),
        ) {
            FramePlan::Dispatch(waves) => waves,
            FramePlan::Hold(_) | FramePlan::Park => Vec::new(),
        }
    }
}

/// The first fire of a frame of two is submitted, folding by a count
/// the device computes: the scheduler holds it for the second, and the
/// host no longer knows where the set is frozen.
fn first_of_a_frame() -> (Flight, Frames, TestContext) {
    let ctx = Flight::context();
    let mut frames = Frames::of_two(ctx.id);
    let flight = in_flight(&geometry::Fold::Device, true);
    frames.arrive(0);
    assert!(
        frames.dispatched().is_empty(),
        "a frame is dispatched whole"
    );
    assert_eq!(flight.length(), (4, false, 1));
    (flight, frames, ctx)
}

/// The second fire is staged on the frozen length as a lower bound,
/// behind the first: the frame seals and dispatches both, and the
/// counts they report, taken in order, leave the set frozen where the
/// device left it, in both stores and on the cursor.
#[tokio::test(flavor = "current_thread")]
async fn a_frame_of_fires_that_fold_by_a_device_count_seals_and_dispatches() -> Anyhow<()> {
    let device = geometry::Fold::Device;
    let (flight, mut frames, mut ctx) = first_of_a_frame();
    let (second, tag) = flight.next(&device);
    assert!(tag != 0 && tag != flight.tag, "each fire reports its own");
    assert_eq!(flight.length(), (4, false, 2));
    frames.arrive(1);
    assert_eq!(
        frames.dispatched(),
        [[0], [1]],
        "the frame is whole: both fires go, in slot order"
    );
    assert_eq!(flight.fork(), Err(RsError::DeviceFoldUnsettled));
    for (completion, tag) in [(&flight.completion, flight.tag), (&second, tag)] {
        record_kv_writes(vec![report(tag, 3)]);
        resolve_test_completion(completion, false);
    }
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        drain_pipeline_fires(&mut ctx, &flight.fires),
    )
    .await
    .expect("both fires settle")?;
    assert_eq!(flight.frozen(), (10, 10), "the first fold, then the second");
    assert_eq!(flight.length(), (10, true, 0));
    assert_eq!(flight.fork(), Ok(10));
    assert!(flight.failure.lock().unwrap().is_none());
    Ok(())
}

/// The control, for staging the second fire on a lower bound: a submit
/// of a recurrent fire that first runs the pass's fires in flight to
/// their end wherever a frozen length is not exact waits, in a frame,
/// for a fire the scheduler dispatches only once the fire that waits
/// has arrived: it does not end, and the frame never fills.
#[tokio::test(flavor = "current_thread")]
async fn a_fire_that_waited_for_the_fold_before_it_would_hold_its_own_frame() {
    let (flight, mut frames, mut ctx) = first_of_a_frame();
    let waited = tokio::time::timeout(
        std::time::Duration::from_millis(250),
        drain_pipeline_fires(&mut ctx, &flight.fires),
    )
    .await;
    assert!(waited.is_err(), "nothing ends the wait");
    assert!(
        frames.dispatched().is_empty(),
        "the fire it waits for is held for the fire that waits"
    );
    assert_eq!(flight.frozen(), (4, 4));
}

/// A fire the engine took whose call then traps, here on a pass gone
/// where each path advances its shadow, is never queued: nothing
/// finalizes it. Its pipeline fails by that name, and its sets with it.
/// The marks its submit left stand while the engine may still run it
/// and lift when the engine is done with it, the pipeline and its queue
/// long gone, or when the engine itself is gone: nothing waits on them
/// for good. Its folded length is a bound from then on, which no fork
/// takes; no other pipeline takes the set, and its recurrent state
/// stays refused, since what the fire folded was never read.
#[test]
fn a_fire_submitted_and_never_queued_fails_its_pipeline_and_its_settle_lifts_the_marks() {
    use crate::store::{PipelineScope, ScopeRefusal};

    /// What each path does between its submit and its queueing, on a
    /// pass since gone.
    fn trapped(
        ctx: &mut TestContext,
        fires: &PendingFires,
        fire: PendingFire,
        unqueued: Unqueued,
    ) -> Anyhow<()> {
        let gone: Resource<ForwardPass> = Resource::new_borrow(u32::MAX);
        advance_shadow(ctx, &gone)?;
        unqueued.queue(fires, fire);
        Ok(())
    }

    let dropped = || RsError::DeviceFoldFailed(RsDeviceFold::DROPPED.to_string());
    let unqueued = |owner| {
        Err(ScopeRefusal::Failed {
            owner,
            reason: UNQUEUED.to_string(),
        })
    };
    for lost in [false, true] {
        let (flight, fire, guard) = submitted(&geometry::Fold::Device, true);
        let Flight {
            stores,
            fires,
            completion,
            failure,
            kv,
            rs,
            ..
        } = flight;
        let scope = PipelineScope::failing_as(failure.clone(), || true);
        kv.claim_pipeline_scope(&scope).unwrap();
        let mut ctx = Flight::context();
        assert!(trapped(&mut ctx, &fires, fire, guard).is_err());
        assert!(fires.lock().unwrap().is_empty(), "nothing finalizes it");
        assert_eq!(failure.lock().unwrap().as_deref(), Some(UNQUEUED));
        assert_eq!(kv.pipeline_failure().as_deref(), Some(UNQUEUED));

        // The engine may still run it: the marks stand.
        let sets = [rs];
        let mut waiting = Box::pin(device_folds_settled(&stores.rs, &sets));
        assert_eq!(
            stores.rs.lock().unwrap().device_fold_gate(rs),
            Err(RsError::DeviceFoldUnsettled)
        );
        assert_eq!(stores.kv.lock().fork(kv.id), Err(UNSETTLED));
        assert!(waiting.as_mut().now_or_never().is_none());

        // The pipeline goes, and the engine settles the fire after it.
        scope.close();
        drop((fires, failure, ctx));
        if lost {
            drop(completion);
        } else {
            resolve_test_completion(&completion, false);
        }
        assert!(waiting.as_mut().now_or_never().is_some());
        assert_eq!(stores.kv.lock().unsettled(kv.id).unwrap(), 0);
        assert_eq!(folded(&stores, kv.id), (4, false));
        assert_eq!(stores.kv.lock().fork(kv.id), Err(UNPLACED));
        assert_eq!(
            stores.rs.lock().unwrap().device_fold_gate(rs),
            Err(dropped())
        );
        let next = PipelineScope::failing_as(Default::default(), || true);
        assert_eq!(kv.claim_pipeline_scope(&next), unqueued(scope.id()));
    }

    // A failure that said why first keeps its reason.
    let (flight, fire, guard) = submitted(&geometry::Fold::Device, true);
    *flight.failure.lock().unwrap() = Some("pipeline: submit failed".to_string());
    drop((guard, fire));
    assert_eq!(
        flight.failure.lock().unwrap().as_deref(),
        Some("pipeline: submit failed")
    );
    resolve_test_completion(&flight.completion, false);

    // A fire queued leaves none: its pipeline finalizes it.
    let flight = in_flight(&geometry::Fold::Device, true);
    assert_eq!(flight.fires.lock().unwrap().len(), 1);
    assert_eq!(*flight.failure.lock().unwrap(), None);
    resolve_test_completion(&flight.completion, false);
}
