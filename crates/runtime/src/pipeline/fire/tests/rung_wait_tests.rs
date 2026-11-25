/// The MTP k=4 stall: a device-carried decode's fires of two rows sat
/// queued behind a rung the walker had lowered to one row, since the
/// scheduler's lane count asks one row a lane and nothing asked for
/// theirs; the scheduler keeps a fire past the rung queued, so its frame
/// never ran. Such a fire now asks the rung for its rows, enqueues once
/// the rung takes them, and holds the ask while it is queued.
#[tokio::test(flavor = "current_thread")]
async fn a_device_fire_asks_the_rung_for_its_rows_and_holds_it() {
    const ENGINE: usize = 58;
    let stores = super::unfolded_tests::pool_of_one_row_rs_pages();
    let pid = uuid::Uuid::new_v4();
    crate::store::shared::set_fire_cap_for_test(ENGINE, 1);
    // One row a lane: the lane count holds it; it asks nothing.
    assert!(super::ask_rung(&stores, ENGINE, pid, pid, 1, 1).is_none());
    assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);

    let raise = tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        // As the walker lifts the cap toward what is asked.
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 2);
        crate::store::shared::set_fire_cap_for_test(ENGINE, 2);
    });
    let ask = super::ask_rung(&stores, ENGINE, pid, pid, 2, 2).expect("two rows ask");
    tokio::time::timeout(std::time::Duration::from_secs(5), ask.wait(2))
        .await
        .expect("no indefinite wait")
        .expect("the rung takes it");
    raise.await.unwrap();
    assert!(
        crate::store::shared::fire_cap(ENGINE) >= 2,
        "enqueued once the rung takes it"
    );
    assert_eq!(
        crate::store::shared::wanted_rows(ENGINE),
        2,
        "the ask stands while the fire is queued"
    );
    drop(ask);
    assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
}

/// The stack wedge through the planner (500 processes at once): the rung
/// rose for a pass while nothing waited; a guest bind short of a run of
/// chunks then waits in the planner's queue. The walk comes down at once
/// to the live lane's rung, its give-back tells the planner, which
/// serves the bind from the span given back; the rung rises for the
/// pass again only once no process demand waits.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_waiting_in_the_planner_takes_the_span_back_and_the_rung_rises_after() {
    const ENGINE: usize = 73;
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    // 16 chunks, the null chunk, the largest rung's activation 8, the
    // smallest's 2: the pool starts at the largest.
    let (model, stores) = super::unfolded_tests::pool_with_rungs_of(vec![(1, 2), (512, 8)]);
    let shared = stores.shared.clone().expect("a shared pool");
    let list = shared.lock().unwrap().list().expect("a kv list").pool;
    let planner = crate::planner::ResidencyPlanner::new(std::sync::Arc::new(
        crate::planner::RegistryPool::new(model, 0, false),
    ));
    crate::planner::init_planner(model, ENGINE, planner);
    let planner = crate::planner::planner_for(model, ENGINE).expect("installed");
    let grow = crate::store::shared::SharedGrow::new(model, ENGINE, shared.clone())
        .with_live_lanes(|| 1);
    // kv takes every chunk below the span; a pass asks the largest rung.
    let _kv = {
        let mut pool = shared.lock().unwrap();
        let free = pool.pool().available(list);
        pool.take_refs(list, free).expect("free slots")
    };
    let want = crate::store::shared::want_rows(ENGINE, 512);
    grow.walk_rung().await;
    assert_eq!(crate::store::shared::fire_cap(ENGINE), u32::MAX);

    // A bind short of a run of two chunks waits in the planner's queue.
    let pid = uuid::Uuid::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    let bind = {
        let planner = planner.clone();
        crate::rt::spawn(async move { planner.wait_bind_room(pid, 2, 1, 0).await })
    };
    while !planner.demand_waits() {
        tokio::task::yield_now().await;
    }

    // The walk comes down to the live lane's rung at once, past the
    // pass's ask, and the planner serves the bind from the span.
    grow.walk_rung().await;
    assert_eq!(crate::store::shared::fire_cap(ENGINE), 1);
    crate::store::shared::grant_rung(ENGINE, 1);
    grow.walk_rung().await;
    grow.walk_rung().await;
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 2);
    tokio::time::timeout(std::time::Duration::from_secs(5), bind)
        .await
        .expect("the bind is served from the span given back")
        .expect("its task ends")
        .expect("served");
    assert!(!planner.demand_waits());

    // Nothing waits: the rung rises for the pass again.
    grow.walk_rung().await;
    assert_eq!(shared.lock().unwrap().rung(), 512);
    assert_eq!(crate::store::shared::fire_cap(ENGINE), u32::MAX);
    drop(want);
}
