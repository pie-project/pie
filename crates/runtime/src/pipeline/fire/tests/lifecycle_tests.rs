use super::*;
use wasmtime::component::ResourceTable;

pub(super) struct TestContext {
    pub(super) id: uuid::Uuid,
    pub(super) resources: ResourceTable,
    pub(super) tails: Vec<PendingFires>,
}

impl FireContext for TestContext {
    fn resources(&mut self) -> &mut ResourceTable {
        &mut self.resources
    }

    fn process_id(&self) -> uuid::Uuid {
        self.id
    }

    fn pipeline_tails(&self) -> Vec<PendingFires> {
        self.tails.clone()
    }
}

/// A beam's finish while a frame the process submitted is held unsealed
/// under the rung's cap (a frame of the set staged before its finish and
/// fired after a raise, its lanes' reserve naming the old tail the
/// finish would let go). The finish lets
/// nothing go until that frame lands: the set's table stands as the
/// frame named it, and once it lands the finish relists.
#[tokio::test(flavor = "current_thread")]
async fn a_beams_finish_waits_for_a_frame_held_under_the_cap() -> anyhow::Result<()> {
    struct HeldFrame {
        id: uuid::Uuid,
        resources: ResourceTable,
        landed: Option<tokio::sync::oneshot::Receiver<()>>,
    }
    impl FireContext for HeldFrame {
        fn resources(&mut self) -> &mut ResourceTable {
            &mut self.resources
        }
        fn process_id(&self) -> uuid::Uuid {
            self.id
        }
        async fn settle_pipeline_tail(&mut self) -> anyhow::Result<()> {
            if let Some(landed) = self.landed.take() {
                let _ = landed.await;
            }
            Ok(())
        }
    }
    let model = crate::store::registry::register_model(32, &[8], &[4]);
    let stores = crate::store::registry::get(model, 0);
    let ws = {
        let mut store = stores.kv.lock();
        let ws = store.create_working_set();
        store.reserve(ws, 4).expect("reserve four pages");
        let mut granted = store.reserve_device_pages(4).expect("four free pages");
        let (_, txn) = kv::prepare_explicit_reserved(&mut store, ws, &[0, 1, 2, 3], &mut granted)
            .expect("write four pages");
        kv::settle(&mut store, txn);
        ws
    };
    let table = |stores: &crate::store::registry::Stores| {
        crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
            kv.flat_table(ws).map(<[_]>::to_vec).unwrap_or_default()
        })
    };
    let before = table(&stores);
    assert_eq!(before.len(), 4);
    let (land, landed) = tokio::sync::oneshot::channel();
    let mut context = HeldFrame {
        id: uuid::Uuid::new_v4(),
        resources: ResourceTable::new(),
        landed: Some(landed),
    };
    let finish = relist_beam(&mut context, model, 0, ws, None, 1, &[2, 0]);
    tokio::pin!(finish);
    for _ in 0..20 {
        tokio::select! {
            biased;
            _ = &mut finish => panic!("the finish ran before the held frame landed"),
            () = tokio::task::yield_now() => {}
        }
    }
    assert_eq!(table(&stores), before, "nothing let go while the frame is held");
    let _ = land.send(());
    tokio::time::timeout(std::time::Duration::from_secs(5), finish)
        .await
        .expect("the finish goes once the frame landed")
        .expect("relisted");
    assert_eq!(table(&stores), vec![before[0], before[3], before[1]]);
    // The next fire's translation and reserve are stamped from this
    // table: they name the listed pages alone.
    let stamped = working_set_flat_table(&stores, ws).expect("the set's table");
    assert_eq!(stamped, [before[0].0, before[3].0, before[1].0]);
    Ok(())
}

/// Mac's gemma-4-e4b c64 stall: a windowed ask that settled its tail
/// whole first waited on its pass's earlier piece, whose frame only the
/// piece asking completes. The ask now returns once the planner grants
/// (growth), the tail still in flight and untouched.
#[tokio::test(flavor = "current_thread")]
async fn a_windowed_ask_never_waits_on_its_own_unlanded_tail() -> anyhow::Result<()> {
    let (fires, completion, ws, _stores, _failure) = pending_fire_fixture(Vec::new());
    let mut context = TestContext {
        id: uuid::Uuid::new_v4(),
        resources: ResourceTable::new(),
        tails: vec![fires.clone()],
    };
    context.resources.push(ws)?;
    let (grow, grown) = tokio::sync::oneshot::channel::<()>();
    let ask = async move {
        grown.await.expect("the pool grows");
        "granted"
    };
    let task = tokio::spawn(async move {
        // The pool grows a moment later; the tail never lands meanwhile.
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        let _ = grow.send(());
    });
    let asked = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        ask_while_tail_lands(&mut context, ask, || {}),
    )
    .await
    .expect("the ask returns without its tail landing");
    assert_eq!(asked.map_err(anyhow::Error::msg)?, "granted");
    task.await?;
    assert_eq!(fires.lock().expect("test FIFO").len(), 1, "the tail stays");
    resolve_test_completion(&completion, false);
    finalize_all(&mut context, &fires, true).await?;
    Ok(())
}

/// Two windowed processes, a pool that cannot grant at once: one ask is
/// served only by the window slots the other's tail gives back, and the
/// other's tail lands while it asks too. Each finalizes its tail as it
/// lands, so the slots come back and both asks are served.
#[tokio::test(flavor = "current_thread")]
async fn two_windowed_asks_are_served_by_the_tails_they_finalize_as_they_land()
-> anyhow::Result<()> {
    let mut processes = Vec::new();
    for _ in 0..2 {
        let (fires, completion, ws, _stores, _failure) = pending_fire_fixture(Vec::new());
        let mut context = TestContext {
            id: uuid::Uuid::new_v4(),
            resources: ResourceTable::new(),
            tails: vec![fires.clone()],
        };
        context.resources.push(ws)?;
        processes.push((context, fires, completion));
    }
    // The planner's room: slots come back as a tail is finalized.
    let freed = Arc::new(tokio::sync::Semaphore::new(0));
    let mut tasks = Vec::new();
    for (mut context, fires, completion) in processes {
        let freed = freed.clone();
        let released = freed.clone();
        tasks.push(tokio::spawn(async move {
            let ask = async move {
                // Served once the other process's tail gave its slots back.
                freed.acquire().await.expect("room").forget();
            };
            let lands = tokio::spawn(async move {
                for _ in 0..5 {
                    tokio::task::yield_now().await;
                }
                resolve_test_completion(&completion, false);
            });
            ask_while_tail_lands(&mut context, ask, || released.add_permits(1))
                .await
                .map_err(anyhow::Error::msg)?;
            lands.await?;
            // Whatever landed after the grant is finalized on the fire's
            // own path, and its slots go back to the planner as well.
            finalize_all(&mut context, &fires, true).await?;
            released.add_permits(1);
            anyhow::Ok(fires.lock().expect("test FIFO").len())
        }));
    }
    for task in tasks {
        let left = tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .expect("both asks are served")??;
        assert_eq!(left, 0);
    }
    Ok(())
}

/// A stale grant is quoted again on drift (the demand changed), not by
/// count, and refused when stale against the demand it was quoted for.
#[test]
fn a_stale_grant_is_quoted_again_only_on_drift() {
    let demand = |kv_pages| crate::planner::Demand {
        kv_pages,
        rs: crate::store::rs::RsUnits::default(),
    };
    let mut stale_at = None;
    assert!(
        requote_on_drift(&mut stale_at, demand(4)),
        "first stale: quoted again"
    );
    assert!(
        requote_on_drift(&mut stale_at, demand(6)),
        "drifted: quoted again"
    );
    assert!(
        requote_on_drift(&mut stale_at, demand(7)),
        "drifted again: however often"
    );
    assert!(
        !requote_on_drift(&mut stale_at, demand(7)),
        "stale against the demand it was quoted for: refused"
    );
}

pub(crate) fn pending_fire_fixture(
    cells: BoundCells,
) -> (
    PendingFires,
    crate::engine::WorkItemCompletion,
    KvWorkingSet,
    crate::store::registry::Stores,
    PipelineFailure,
) {
    let model = crate::store::registry::register_model(32, &[1], &[1]);
    let stores = crate::store::registry::get(model, 0);
    let (id, txn) = {
        let mut store = stores.kv.lock();
        let id = store.create_working_set();
        store.reserve(id, 1).expect("reserve the test sequence");
        let mut granted = store
            .reserve_device_pages(1)
            .expect("one free physical page");
        let (_, txn) = kv::prepare_explicit_reserved(&mut store, id, &[0], &mut granted)
            .expect("publish a real pending KV transaction");
        (id, txn)
    };
    let ws = KvWorkingSet::new(model, 0, id, 32);
    let completion = crate::engine::WorkItemCompletion::deferred_with_guard(None);
    let failure = Arc::new(Mutex::new(None));
    let fires = Arc::new(PendingFireQueue::new());
    fires
        .lock()
        .expect("test FIFO")
        .push_back(PendingOp::Fire(PendingFire {
            completion: completion.clone(),
            checkpoints: Vec::new(),
            kv: vec![FireWs {
                ws: id,
                kvtxn: KvTxnGuard::new(model, 0, Some(txn)),
                log: None,
                lease: ws.fire_lease().expect("lease the submitted working set"),
                fold: None,
                reports: Vec::new(),
                staged: None,
            }],
            device_geometry: false,
            rstxn: RsTxnsGuard::new(model, 0, None),
            model,
            engine: 0,
            fwd_rep: u32::MAX,
            cells,
            failure: failure.clone(),
            retired: None,
            lane: None,
        }));
    (fires, completion, ws, stores, failure)
}

pub(crate) fn resolve_test_completion(
    completion: &crate::engine::WorkItemCompletion,
    failed: bool,
) {
    completion.mark_native_retired();
    if failed {
        completion.reject("test callback failure");
    } else {
        // SAFETY: `completion` is borrowed for the whole call and owns this
        // terminal cell, so the pointer is live; `publish` is an atomic store.
        unsafe {
            (*completion.terminal_cell_ptr().0)
                .publish(crate::engine::completion::TERMINAL_OUTCOME_SUCCESS);
        }
        completion
            .resolve_from_terminal()
            .expect("publish successful completion");
    }
}

/// Without run-ahead (`frame_dispatch_depth = 1`) a fire's last handle
/// goes as soon as it lands, before its settled hooks run off the
/// resolving stack. The rows it placed stay written: the next fire,
/// which reads them, is not refused as reading slots never written.
#[tokio::test(flavor = "current_thread")]
async fn a_landed_fire_dropped_before_its_hooks_run_keeps_its_rows_written() {
    use crate::inferlet::host::working_set::Cursor;
    let cell: CursorCell = std::sync::Arc::new(std::sync::Mutex::new(Cursor {
        reserved: 64,
        ..Cursor::default()
    }));
    // The prompt, 31 rows at 0..31, host-placed.
    let fire = crate::engine::WorkItemCompletion::deferred_with_guard(None);
    advance_cursor(&cell, Some(31), None);
    host_wrote(&cell, &fire, 0, 31);
    resolve_test_completion(&fire, false);
    drop(fire);
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    let now = cell.lock().unwrap().clone();
    assert_eq!((now.readable(), now.readable_known()), (31, true));
    // The next fire appends its one row at 31.
    let place = geometry::Place {
        folded: 0,
        folded_exact: true,
        tail: now.tail,
        shared: false,
        spans: Vec::new(),
        readable: now.readable(),
        readable_known: now.readable_known(),
        set: 1,
    };
    let next = geometry::ReqGeometry {
        token_ids: vec![7],
        qo_indptr: vec![0, 1],
        ..geometry::ReqGeometry::default()
    };
    let (_, landings) = next.lanes(&[place], 16).expect("reads only what was written");
    assert_eq!(landings, vec![geometry::Landing { start: 31, rows: 1 }]);

    // A fire that failed still unwrites its rows, dropped or not.
    let failed = crate::engine::WorkItemCompletion::deferred_with_guard(None);
    host_wrote(&cell, &failed, 31, 32);
    resolve_test_completion(&failed, true);
    drop(failed);
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    assert_eq!(cell.lock().unwrap().readable(), 31);
    // One never resolved, its last handle gone, landed nothing.
    let unresolved = crate::engine::WorkItemCompletion::deferred_with_guard(None);
    host_wrote(&cell, &unresolved, 31, 32);
    drop(unresolved);
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    assert_eq!(cell.lock().unwrap().readable(), 31);
}

#[tokio::test(flavor = "current_thread")]
async fn cancelled_full_drain_keeps_pending_kv_for_teardown() -> anyhow::Result<()> {
    use std::future::Future;
    for failed in [false, true] {
        let (fires, completion, ws, stores, failure) = pending_fire_fixture(Vec::new());
        let mut context = TestContext {
            id: uuid::Uuid::new_v4(),
            resources: ResourceTable::new(),
            tails: Vec::new(),
        };
        context.resources.push(ws)?;
        {
            let mut drain = Box::pin(finalize_all(&mut context, &fires, false));
            let mut task = std::task::Context::from_waker(std::task::Waker::noop());
            assert!(drain.as_mut().poll(&mut task).is_pending());
            // Dropping the real host future models guest-task cancellation.
        }
        assert_eq!(
            fires.lock().expect("test FIFO").len(),
            1,
            "cancelled drain must leave its unresolved operation in the FIFO"
        );
        context.resources = ResourceTable::new();
        assert_eq!(
            stores.kv.lock().available_pages(),
            0,
            "resource destruction must not recycle pages before completion"
        );
        resolve_test_completion(&completion, failed);
        finalize_all(&mut context, &fires, true).await?;
        assert!(fires.lock().expect("test FIFO").is_empty());
        assert_eq!(
            stores.kv.lock().available_pages(),
            1,
            "teardown must settle the original transaction and release its page"
        );
        assert_eq!(failure.lock().expect("test failure").is_some(), failed);
    }
    Ok(())
}

struct Flag(std::sync::atomic::AtomicBool);

impl std::task::Wake for Flag {
    fn wake(self: Arc<Self>) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// Lanes that all wait on one front fire register their wakers on its
/// completion before it settles: settling it wakes every one of them,
/// not only the last to register (else all but one lane sleep for good
/// on a settled fire while the last decodes alone).
#[tokio::test(flavor = "current_thread")]
async fn every_lane_waiting_on_the_front_fire_wakes_when_it_settles() -> anyhow::Result<()> {
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, Ordering};
    let (fires, completion, ws, _stores, _failure) = pending_fire_fixture(Vec::new());
    let mut context = TestContext {
        id: uuid::Uuid::new_v4(),
        resources: ResourceTable::new(),
        tails: Vec::new(),
    };
    context.resources.push(ws)?;
    let flag = || Arc::new(Flag(AtomicBool::new(false)));
    // Three lanes at the front fire, and a drain awaiting its completion.
    let flags: Vec<Arc<Flag>> = (0..4).map(|_| flag()).collect();
    let mut woken = Vec::new();
    {
        let mut lanes: Vec<_> = flags[..3]
            .iter()
            .map(|flag| {
                (
                    Box::pin(fires.await_front_completion()),
                    std::task::Waker::from(flag.clone()),
                )
            })
            .collect();
        let mut drain = completion.clone();
        let drain_waker = std::task::Waker::from(flags[3].clone());
        for (wait, waker) in &mut lanes {
            let mut cx = std::task::Context::from_waker(waker);
            assert!(
                wait.as_mut().poll(&mut cx).is_pending(),
                "nothing settled yet"
            );
        }
        let mut cx = std::task::Context::from_waker(&drain_waker);
        assert!(std::pin::Pin::new(&mut drain).poll(&mut cx).is_pending());

        resolve_test_completion(&completion, false);
        woken.extend(flags.iter().map(|flag| flag.0.load(Ordering::Acquire)));
        for (wait, waker) in &mut lanes {
            let mut cx = std::task::Context::from_waker(waker);
            assert!(wait.as_mut().poll(&mut cx).is_ready(), "the fire settled");
        }
    }
    finalize_all(&mut context, &fires, true).await?;
    assert_eq!(
        woken,
        vec![true; 4],
        "every waiter registered before the settle is woken by it (three lanes, then the \
         drain)"
    );
    Ok(())
}

/// A decode run ahead on a closed pipeline: `fires` fires after a
/// 20-token prompt, each writing one row and folding it as it is
/// prepared (a decode's fold, frozen at once, not sealed), their
/// completions pending. The set, its cursor, and the fires.
struct RunAhead {
    context: TestContext,
    pipeline: u32,
    ws: KvWorkingSet,
    cursor: CursorCell,
    stores: crate::store::registry::Stores,
    completions: Vec<crate::engine::WorkItemCompletion>,
}

/// `state`: the set keeps a recurrent state frozen through that row,
/// and its decodes fold nothing (the lagged fold: the program accepted
/// none of them yet).
fn run_ahead(fires: u32, state: Option<u32>) -> anyhow::Result<RunAhead> {
    run_ahead_stating(fires, state, &[])
}

/// [`run_ahead`], the fires `finals` names (from 0) a final pass's
/// (CONTRACT-BC 25).
fn run_ahead_stating(
    fires: u32,
    state: Option<u32>,
    finals: &[u32],
) -> anyhow::Result<RunAhead> {
    let model = crate::store::registry::register_model_with_swap(
        16,
        &[64],
        &[0],
        &[0],
        &[crate::store::registry::StatePool { slots: 1, pages: 4 }],
        &[0],
    );
    let stores = crate::store::registry::get(model, 0);
    let rs = state.map(|frozen| {
        let mut rs = stores.rs.lock().unwrap();
        let ws = rs.create_working_set(crate::store::rs::RsGeometry {
            state_size: 8,
            buffer_page_tokens: 4,
        });
        rs.fold(ws, frozen);
        ws
    });
    let prompt: Vec<u32> = (0..20).map(|t| 3000 + t).collect();
    let id = {
        let mut kv = stores.kv.lock();
        kv.set_keeps(true);
        let id = kv.create_working_set();
        kv.reserve(id, 4).expect("reserve the test sequence");
        let mut granted = kv.reserve_device_pages(4).expect("four free pages");
        kv::prepare_explicit_reserved(&mut kv, id, &[0, 1, 2, 3], &mut granted)
            .expect("publish the pages");
        kv.log_rows(id, 0, &prompt).unwrap();
        kv.fold(id, 20, false).unwrap();
        id
    };
    let ws = KvWorkingSet::new(model, 0, id, 16);
    let cursor: CursorCell = Arc::new(Mutex::new(Default::default()));
    let mut context = TestContext {
        id: uuid::Uuid::new_v4(),
        resources: ResourceTable::new(),
        tails: Vec::new(),
    };
    let pipeline = context.resources.push(Pipeline::new())?.rep();
    let borrowed: Resource<Pipeline> = Resource::new_borrow(pipeline);
    let (queue, failure) = {
        let pipeline = context.resources.get(&borrowed)?;
        ws.claim_pipeline_scope(&pipeline.scope)
            .expect("the pipeline takes the set");
        (pipeline.fires.clone(), pipeline.failure.clone())
    };
    let mut completions = Vec::new();
    for at in 0..fires {
        let start = 20 + u64::from(at);
        let seq = {
            let mut kv = stores.kv.lock();
            let seq = kv.log_begin(id, start).unwrap();
            if rs.is_none() {
                kv.freeze(id, start + 1).unwrap();
            }
            seq
        };
        let completion = crate::engine::WorkItemCompletion::deferred_with_guard(None);
        queue
            .lock()
            .unwrap()
            .push_back(PendingOp::Fire(PendingFire {
                completion: completion.clone(),
                checkpoints: Vec::new(),
                kv: vec![FireWs {
                    ws: id,
                    kvtxn: KvTxnGuard::new(model, 0, None),
                    log: Some(LogFire {
                        seq,
                        rows: LogRows::Known {
                            start,
                            tokens: vec![5000 + at],
                        },
                    }),
                    lease: ws.fire_lease().expect("lease the set"),
                    fold: rs.is_none().then_some(FoldAt {
                        by: FoldBy::To(start + 1),
                        decoded: true,
                    }),
                    reports: Vec::new(),
                    staged: None,
                }],
                device_geometry: false,
                rstxn: RsTxnsGuard::new(model, 0, None),
                model,
                engine: 0,
                fwd_rep: u32::MAX,
                cells: Vec::new(),
                failure: failure.clone(),
                retired: None,
                lane: None,
            }));
        advance_cursor(&cursor, Some(start as u32 + 1), None);
        record_writes(
            &mut context,
            pipeline,
            Some(vec![SetStart {
                model,
                engine: 0,
                kv: id,
                cursor: cursor.clone(),
                start,
                rs,
            }]),
            finals.contains(&at),
        )?;
        completions.push(completion);
    }
    context.resources.get(&borrowed)?.scope.close();
    Ok(RunAhead {
        context,
        pipeline,
        ws,
        cursor,
        stores,
        completions,
    })
}

impl RunAhead {
    /// Folded length, log length and pages held.
    fn length(&self) -> (u64, usize, u64) {
        let kv = self.stores.kv.lock();
        (
            kv.folded(self.ws.id).unwrap(),
            kv.logged(self.ws.id).unwrap().len(),
            kv.mapped_len(self.ws.id).unwrap(),
        )
    }

    fn failure(&mut self) -> Option<String> {
        let borrowed: Resource<Pipeline> = Resource::new_borrow(self.pipeline);
        let failure = self
            .context
            .resources
            .get(&borrowed)
            .unwrap()
            .failure
            .clone();
        failure.lock().unwrap().clone()
    }
}

/// A program stopped at the third token of a decode run ahead six fires
/// (EOS or a stop string alike: the loop breaks on the token the third
/// fire wrote out). The take-back waits for the fires still in flight:
/// nothing is cut under one. Once they land the set holds the prompt
/// and the three rows kept, its cursor ends there, a fork taken then
/// holds nothing past them, and the rows taken back are never sealed.
/// A run that stopped at its budget takes nothing back.
#[tokio::test(flavor = "current_thread")]
async fn a_stop_takes_back_what_ran_ahead_once_it_lands() -> anyhow::Result<()> {
    use std::future::Future;
    let mut run = run_ahead(6, None)?;
    for completion in &run.completions[..2] {
        resolve_test_completion(completion, false);
    }
    assert_eq!(run.length().0, 26, "every fire folded as it was prepared");
    let pipeline = run.pipeline;
    let retract = {
        let context = &mut run.context;
        let mut retract =
            Box::pin(pipeline_retract(context, Resource::new_borrow(pipeline), 3));
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        for _ in 0..8 {
            assert!(
                retract.as_mut().poll(&mut cx).is_pending(),
                "it waits on the fires in flight"
            );
            tokio::task::yield_now().await;
        }
        retract
    };
    drop(retract);
    assert_eq!(
        run.length(),
        (26, 22, 4),
        "nothing cut under a fire in flight"
    );
    for completion in &run.completions[2..] {
        resolve_test_completion(completion, false);
    }
    pipeline_retract(&mut run.context, Resource::new_borrow(pipeline), 3)
        .await?
        .map_err(anyhow::Error::msg)?;
    assert_eq!(
        run.length(),
        (23, 23, 2),
        "the prompt and the three rows kept"
    );
    assert_eq!(run.cursor.lock().unwrap().tail, Some(23));
    assert_eq!(run.failure(), None);
    let kept: Vec<u32> = (0..20).map(|t| 3000 + t).chain(5000..5003).collect();
    {
        let mut kv = run.stores.kv.lock();
        assert_eq!(kv.logged(run.ws.id).unwrap(), &kept[..]);
        let child = kv.fork(run.ws.id).unwrap();
        assert_eq!(
            kv.folded(child).unwrap(),
            23,
            "a fork holds nothing past the stop"
        );
        assert_eq!(kv.mapped_len(child).unwrap(), 2);
        assert_eq!(
            kv.sealed_in(run.ws.id, 0..2).unwrap(),
            [0],
            "and nothing past it sealed"
        );
        let epoch = kv.current_epoch();
        kv.release_working_set(child, epoch);
    }

    let mut whole = run_ahead(6, None)?;
    for completion in &whole.completions {
        resolve_test_completion(completion, false);
    }
    let pipeline = whole.pipeline;
    pipeline_retract(&mut whole.context, Resource::new_borrow(pipeline), 0)
        .await?
        .map_err(anyhow::Error::msg)?;
    let fires = whole
        .context
        .resources
        .get(&Resource::<Pipeline>::new_borrow(pipeline))?
        .fires
        .clone();
    finalize_all(&mut whole.context, &fires, false).await?;
    assert_eq!(
        whole.length(),
        (26, 26, 4),
        "its budget's rows, no stray one"
    );
    Ok(())
}

/// Under the lagged fold a run-ahead past a stop on a set with a
/// recurrent state folded none of its rows: the take-back is a buffer
/// truncate, the state untouched, the set usable for the next turn.
#[tokio::test(flavor = "current_thread")]
async fn a_stop_on_a_recurrent_set_takes_back_its_buffered_rows() -> anyhow::Result<()> {
    let mut run = run_ahead(6, Some(20))?;
    for completion in &run.completions {
        resolve_test_completion(completion, false);
    }
    let pipeline = run.pipeline;
    pipeline_retract(&mut run.context, Resource::new_borrow(pipeline), 3)
        .await?
        .map_err(anyhow::Error::msg)?;
    assert_eq!(run.failure(), None);
    let (folded, logged, _) = run.length();
    assert_eq!((folded, logged), (20, 23), "the three rows kept stay buffered");
    assert_eq!(run.cursor.lock().unwrap().tail, Some(23));
    Ok(())
}

/// A retract reaching a fire whose pass stated its rows final (a
/// default pass: the SDK states it unless the program declared the pass
/// retractable) is refused by name before anything is cut: the set
/// keeps every row, the pipeline is not failed. This is the loud failure
/// of an inferlet that retracts without opting out. One stopping short
/// of it takes back the lagged fires after it as before.
#[tokio::test(flavor = "current_thread")]
async fn a_retract_over_a_final_fire_is_refused_by_name() -> anyhow::Result<()> {
    let mut run = run_ahead_stating(5, Some(20), &[2])?;
    for completion in &run.completions {
        resolve_test_completion(completion, false);
    }
    let pipeline = run.pipeline;
    let before = run.length();
    let refused = pipeline_retract(&mut run.context, Resource::new_borrow(pipeline), 3)
        .await?
        .unwrap_err();
    assert!(refused.starts_with("final rows:"), "{refused}");
    assert!(refused.contains("the 3 fire from the end"), "{refused}");
    assert_eq!(run.length(), before, "nothing cut");
    assert_eq!(run.failure(), None, "the pipeline is not failed");
    pipeline_retract(&mut run.context, Resource::new_borrow(pipeline), 2)
        .await?
        .map_err(anyhow::Error::msg)?;
    assert_eq!(run.failure(), None);
    assert_eq!(run.cursor.lock().unwrap().tail, Some(23), "the final row kept");
    Ok(())
}

/// Rows a recurrent state folded past the cut, or the run-ahead bound
/// accepted, are not taken back: the pipeline fails by that name, and
/// its set is refused where it is used next. A
/// retract on an open pipeline, or past what it submitted, is refused.
#[tokio::test(flavor = "current_thread")]
async fn what_cannot_be_taken_back_is_refused_by_name() -> anyhow::Result<()> {
    let mut run = run_ahead(4, Some(20))?;
    for completion in &run.completions {
        resolve_test_completion(completion, false);
    }
    let pipeline = run.pipeline;
    let past = pipeline_retract(&mut run.context, Resource::new_borrow(pipeline), 5).await?;
    assert!(past.unwrap_err().contains("past the 4 it can take back"));
    // No more than the run-ahead window is kept, however long the
    // pipeline runs.
    let window = crate::scheduler::run_ahead_window();
    let long = run_ahead(u32::try_from(window).unwrap() + 3, None)?;
    let borrowed: Resource<Pipeline> = Resource::new_borrow(long.pipeline);
    assert_eq!(
        long.context
            .resources
            .get(&borrowed)?
            .written
            .lock()
            .unwrap()
            .len(),
        window
    );
    // A state folded past the cut is not taken back: the pipeline fails
    // by that name, and its set is refused where it is used next.
    let mut run = run_ahead(4, Some(23))?;
    for completion in &run.completions {
        resolve_test_completion(completion, false);
    }
    let pipeline = run.pipeline;
    pipeline_retract(&mut run.context, Resource::new_borrow(pipeline), 2)
        .await?
        .map_err(anyhow::Error::msg)?;
    let failure = run.failure().expect("the pipeline fails");
    assert!(failure.contains("a fold is not taken back"), "{failure}");
    assert_eq!(run.length().1, 24, "nothing taken back");
    let next = crate::store::PipelineScope::failing_as(Default::default(), || true);
    let refused = run.ws.claim_pipeline_scope(&next).unwrap_err().to_string();
    assert!(refused.contains("could not be taken back"), "{refused}");
    // Nor are rows the run-ahead bound accepted unread.
    let mut run = run_ahead(4, Some(20))?;
    run.cursor.lock().unwrap().bound_accepted = 23;
    for completion in &run.completions {
        resolve_test_completion(completion, false);
    }
    let pipeline = run.pipeline;
    pipeline_retract(&mut run.context, Resource::new_borrow(pipeline), 2)
        .await?
        .map_err(anyhow::Error::msg)?;
    let failure = run.failure().expect("the pipeline fails");
    assert!(failure.contains("run-ahead bound"), "{failure}");

    let mut context = TestContext {
        id: uuid::Uuid::new_v4(),
        resources: ResourceTable::new(),
        tails: Vec::new(),
    };
    let open = context.resources.push(Pipeline::new())?.rep();
    let refused = pipeline_retract(&mut context, Resource::new_borrow(open), 1).await?;
    assert!(refused.unwrap_err().contains("open pipeline"));
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn close_and_drop_share_graceful_fifo_drain_semantics() -> anyhow::Result<()> {
    let mut context = TestContext {
        id: uuid::Uuid::new_v4(),
        resources: ResourceTable::new(),
        tails: Vec::new(),
    };
    let pipeline = context.resources.push(Pipeline::new())?;
    let rep = pipeline.rep();
    let borrowed: Resource<Pipeline> = Resource::new_borrow(rep);
    let fires = context.resources.get(&borrowed)?.fires.clone();
    fires
        .lock()
        .unwrap()
        .extend([test_pending_op_stub(), test_pending_op_stub()]);

    pipeline_close(&mut context, Resource::new_borrow(rep)).await?;
    assert!(context.resources.get(&borrowed)?.scope.is_closed());
    assert!(fires.lock().unwrap().is_empty());
    pipeline_close(&mut context, Resource::new_borrow(rep)).await?;
    assert!(fires.lock().unwrap().is_empty());

    pipeline_drop(&mut context, pipeline).await?;
    assert!(context.resources.get(&borrowed).is_err());
    Ok(())
}
