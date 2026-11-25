use super::*;
use crate::inferlet::host::working_set::Cursor;
use crate::store::kv::page_table::WorkingSetId;
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

/// What a fold placed as its fire is prepared freezes.
fn frozen_as_prepared(by: Option<&FoldBy>) -> Option<u64> {
    match by {
        Some(FoldBy::To(to)) => Some(*to),
        _ => None,
    }
}

fn report(tag: u64, fold: u32) -> ::engine::KvWrites {
    ::engine::KvWrites {
        tag,
        fold: Some(fold),
        ..::engine::KvWrites::default()
    }
}

#[test]
fn fold_every_case() {
    a_fire_carrying_a_span_leaves_its_lag();
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

// A pass that runs ahead submits a frame's fires one after another, and
// the scheduler dispatches none of them before the frame's last has
// arrived. A fire behind one that folds by a count the device computes
// is therefore staged before that count exists.

