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

#[test]
fn fold_every_case() {
    a_fold_the_host_knows_is_frozen_as_its_fire_is_prepared();
    a_fold_the_device_counts_leaves_the_folded_length_until_it_settles();
    a_piece_inside_the_frozen_prefix_begins_nothing();
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

async fn settle_tasks() {
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
}

// A fold the device counts, between its fire's submit and its settle:
// the set is not forked, kept or moved, and what waits for it is woken
// by the fire's end, however it ends.

use super::lifecycle_tests::{TestContext, resolve_test_completion};
use crate::store::rs::{RsError, RsWorkingSetId};
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
}

impl Flight {
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

    fn context() -> TestContext {
        TestContext {
            id: uuid::Uuid::new_v4(),
            resources: wasmtime::component::ResourceTable::new(),
            tails: Vec::new(),
        }
    }
}

/// `exact` is whether the host knew the folded length when it planned
/// the fold: not while an earlier fire's fold is on its way unplaced.
/// The fire's eight rows land at slot 4.
fn in_flight(fold: &geometry::Fold, exact: bool) -> Flight {
    in_flight_on(fold, exact, true)
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
    let (completion, _) = submit(&kv, part, &cell, &fires, &failure, fold);
    Flight {
        stores,
        fires,
        completion,
        failure,
        kv,
        rs,
        cursor: cell,
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
