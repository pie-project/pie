use super::*;
use wasmtime::component::ResourceTable;

pub(super) struct TestContext {
    pub(super) id: uuid::Uuid,
    pub(super) resources: ResourceTable,
}

impl FireContext for TestContext {
    fn resources(&mut self) -> &mut ResourceTable {
        &mut self.resources
    }

    fn process_id(&self) -> uuid::Uuid {
        self.id
    }
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
        completion
            .terminal_cell_ptr()
            .0
            .publish(crate::engine::completion::TERMINAL_OUTCOME_SUCCESS);
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
