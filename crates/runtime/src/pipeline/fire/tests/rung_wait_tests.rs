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

/// The MTP `--k 0 --pad 1` stall: a decode's pass of two rows was
/// staged while the cap stood wide (a bind had raised the rung), so it
/// asked nothing; the walker, seeing one live lane and no ask, then cut
/// fires to one row, and the scheduler kept the pass's two-row fires
/// queued behind a rung nothing raised. The pass now asks whatever the
/// cap is, and the walker cuts nothing under it until its last fire has
/// settled.
#[tokio::test(flavor = "current_thread")]
async fn a_few_rows_staged_under_a_wide_cap_hold_the_rung_until_they_settle() {
    const ENGINE: usize = 49;
    let stores = super::unfolded_tests::pool_with_rungs(vec![(1, 1), (2, 2), (4, 3)]);
    let shared = stores.shared.clone().expect("a shared pool");
    // The walker of a pool with one live lane, as the governor ticks it.
    let grow =
        crate::store::shared::SharedGrow::new(0, ENGINE, shared.clone()).with_live_lanes(|| 1);
    let pid = uuid::Uuid::new_v4();
    // The rung is up and fires may take any width: a cap that covers
    // the pass.
    assert_eq!(shared.lock().unwrap().rung(), 4);
    assert_eq!(crate::store::shared::fire_cap(ENGINE), u32::MAX);

    // A two-row pass is cut and its fire staged under that cap.
    let ask =
        super::ask_rung(&stores, ENGINE, pid, pid, 2, 2).expect("two rows ask whatever the cap");
    assert_eq!(ask.wait(2).await, Ok(()));
    let fire = crate::engine::completion::WorkItemCompletion::new(1, 1);
    ask.until_settled(&fire);

    // A waiter is short of a slot with the fire queued: fires are cut
    // to what is asked, never under the staged fire.
    shared.lock().unwrap().pool_mut().wait(1, 0, 1);
    for _ in 0..6 {
        grow.walk_rung().await;
        assert!(
            crate::store::shared::fire_cap(ENGINE) >= 2,
            "never cut under the staged fire"
        );
    }
    assert_eq!(
        crate::store::shared::fire_cap(ENGINE),
        2,
        "cut to the rung the staged fire asks, not to the live lane's row"
    );
    assert!(!fire.is_settled());

    // Settled, the ask goes, and the walker takes the cap down to the
    // live lane's row.
    fire.reject("settled");
    // A settlement runs its hooks off the caller's stack.
    tokio::task::yield_now().await;
    assert!(fire.is_settled());
    assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
    for _ in 0..6 {
        grow.walk_rung().await;
    }
    assert_eq!(crate::store::shared::fire_cap(ENGINE), 1);
}

/// A wedge under 500 concurrent processes with an empty host tier: the
/// walk lowered the rung to one row and the span it lent filled
/// with kv, admitted against a device total that counted the lent
/// chunks as kv's; a wide pass then asked the rung back, and its raise
/// waited in the planner's queue as a kv demand among the waiters it
/// would starve, with nothing left to discard. The total counts no
/// lent chunk now, and a rise short of free room asks no planner: the
/// pass fires at the rung held, decided before it is submitted, and
/// the rung rises on the pool's room signal once kv gives chunks back.
#[tokio::test(flavor = "current_thread")]
async fn a_rise_short_of_lent_room_fires_at_the_held_rung_and_rises_once_kv_frees() {
    const ENGINE: usize = 71;
    // 16 chunks, the null chunk, the largest rung's activation 8 (8..16),
    // the smallest's 2 (14..16): a lowered rung lends 8..14.
    let stores = super::unfolded_tests::pool_with_rungs(vec![(1, 2), (512, 8)]);
    let shared = stores.shared.clone().expect("a shared pool");
    let pid = uuid::Uuid::new_v4();
    let list = {
        let mut pool = shared.lock().unwrap();
        pool.steers(ENGINE);
        pool.lower_rung(1).expect("nothing lies in the span");
        pool.list().expect("a kv list").pool
    };
    crate::store::shared::set_fire_cap_for_test(ENGINE, 1);

    // One count of the kv pages, the lent chunks in none: what the
    // facts state (16 chunks less the null one and the largest rung's
    // 8, a page a chunk) is what the planner's total reports.
    let room = shared.lock().unwrap().room();
    assert_eq!(room.list_slots, 16 - 1 - 8, "{room:?}");
    assert_eq!(room.lent_chunks, 6, "{room:?}");

    // kv takes every free slot, the lent chunks among them.
    let held = {
        let mut pool = shared.lock().unwrap();
        let free = pool.pool().available(list);
        assert_eq!(free, 16 - 1 - 2, "a grant may use the lent chunks");
        pool.take_refs(list, free).expect("free slots")
    };

    // A wide pass asks the rung back. The room is all kv's, so its
    // rise waits for room, and the pass is cut at the rung held before
    // anything is submitted: a piece of the held width fires at once.
    let ask = super::ask_rung(&stores, ENGINE, pid, pid, 512, 512).expect("a wide ask");
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        crate::store::shared::rung_wait_until(ENGINE, || ask.fires_at_held_rung()),
    )
    .await
    .expect("the rise says it waits for room");
    assert_eq!(crate::store::shared::fire_cap(ENGINE), 1);
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(5), ask.wait(1)).await,
        Ok(Ok(())),
        "a piece of the held width never waits on the rise"
    );

    // kv gives its chunks back: the rung rises on the room signal.
    shared.lock().unwrap().give_refs(list, &held);
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        crate::store::shared::rung_wait_until(ENGINE, || {
            crate::store::shared::fire_cap(ENGINE) >= 512
        }),
    )
    .await
    .expect("the rung rises once kv frees");
    assert!(!ask.fires_at_held_rung());
    drop(ask);
}

/// The walk looks again as the last waiter is served, not only at the
/// landing that woke the pass: a raise waits behind every demand, and the
/// walk woken by the landing finds it waiting.
#[tokio::test(flavor = "current_thread")]
async fn the_walk_is_told_as_the_last_waiter_is_served() {
    const ENGINE: usize = 78;
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let (model, stores) = super::unfolded_tests::pool_with_rungs_of(vec![(1, 2), (512, 8)]);
    let shared = stores.shared.clone().expect("a shared pool");
    let list = shared.lock().unwrap().list().expect("a kv list").pool;
    let grow = crate::store::shared::SharedGrow::new(model, ENGINE, shared.clone());
    let planner = std::sync::Arc::new(crate::planner::ResidencyPlanner::new(std::sync::Arc::new(
        crate::planner::RegistryPool::new(model, 0, false).with_shared_grow(grow.clone()),
    )));
    planner.arm_drain_task();
    let held = {
        let mut pool = shared.lock().unwrap();
        let free = pool.pool().available(list);
        pool.take_refs(list, free).expect("free slots")
    };
    let pid = uuid::Uuid::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    let bind = {
        let planner = planner.clone();
        crate::rt::spawn(async move { planner.wait_bind_room(pid, 1, 1, 0).await })
    };
    while !planner.demand_waits() {
        tokio::task::yield_now().await;
    }
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    shared.lock().unwrap().give_refs(list, &held);
    planner.pages_freed();
    assert!(grow.told(), "the landing's");
    assert!(planner.demand_waits(), "its pass comes after");
    bind.await.expect("its task ends").expect("served");
    assert!(grow.told(), "the last waiter's");
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
    let grow =
        crate::store::shared::SharedGrow::new(model, ENGINE, shared.clone()).with_live_lanes(|| 1);
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

/// A bind's room queued for while idle cache holds the room: the rung stays where it
/// stands until a pass has asked idle cache and found it out, and comes down
/// then. A bind short of a run outside the planner's queue, which no pass
/// asks idle cache for, has the rung down at once.
#[tokio::test(flavor = "current_thread")]
async fn the_rung_comes_down_for_a_waiter_only_once_idle_cache_is_out() {
    use crate::store::kv::hash::{Hash256, chain_slots, page_hash};
    use crate::store::kv::page_table::KvPageBacking;
    const ENGINE: usize = 75;
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let (model, stores) = super::unfolded_tests::pool_with_rungs_of(vec![(1, 2), (512, 8)]);
    let shared = stores.shared.clone().expect("a shared pool");
    let grow =
        crate::store::shared::SharedGrow::new(model, ENGINE, shared.clone()).with_live_lanes(|| 1);
    let planner = crate::planner::ResidencyPlanner::new(std::sync::Arc::new(
        crate::planner::RegistryPool::new(model, 0, false).with_shared_grow(grow.clone()),
    ));
    crate::planner::init_planner(model, ENGINE, planner);
    let planner = crate::planner::planner_for(model, ENGINE).expect("installed");
    // Its passes run on the drain task, once this one yields.
    planner.arm_drain_task();
    let cap = || crate::store::shared::fire_cap(ENGINE);
    // Idle cache, every page there is, a reader on it.
    let list = shared.lock().unwrap().list().expect("a kv list").pool;
    let filed = || {
        crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
            let pages = kv.reserve_device_pages(kv.available_pages()).unwrap();
            let rows: Vec<u32> = (0..pages.len() as u32 * 16).collect();
            let tokens: Vec<Vec<Option<Hash256>>> = chain_slots(&[3; 32], None, 0, &rows)
                .chunks(16)
                .map(|page| page.iter().map(|hash| Some(*hash)).collect())
                .collect();
            let keys: Vec<Hash256> = tokens.iter().map(|page| page_hash(page)).collect();
            let backings = pages.into_iter().map(KvPageBacking::Resident).collect();
            assert!(kv.import_cached_at(&[], tokens, keys.clone(), backings));
            let run = kv.cached_run(&keys);
            kv.pin_resident(&run).expect("on the device").0
        })
    };
    let pid = uuid::Uuid::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    let passes = || async {
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
    };
    let mut pin = filed();
    // Asked for, not granted, and not queued yet: nothing is told short.
    let demand = crate::planner::Demand {
        kv_pages: 2,
        rs: crate::store::rs::RsUnits::default(),
    };
    assert!(
        planner.grant_now(pid, demand).is_none(),
        "idle cache is pinned"
    );
    grow.walk_rung().await;
    assert_eq!(cap(), u32::MAX, "no waiter is queued");
    {
        let ask = planner.wait_bind_room(pid, 1, 2, 0);
        tokio::pin!(ask);
        assert!(
            futures::poll!(ask.as_mut()).is_pending(),
            "idle cache is pinned"
        );
        passes().await;
        assert!(futures::poll!(ask.as_mut()).is_pending());
        // The reader's end has idle cache found out asked again.
        crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| kv.release_reader(pin));
        let served = tokio::time::timeout(std::time::Duration::from_secs(5), ask)
            .await
            .expect("woken by the reader's end");
        assert!(matches!(served, Ok(crate::planner::BindRoomWait::Served)));
    }
    passes().await;
    pin = filed();
    {
        let ask = planner.wait_bind_room(pid, 1, 2, 0);
        tokio::pin!(ask);
        assert!(
            futures::poll!(ask.as_mut()).is_pending(),
            "idle cache is pinned"
        );
        grow.walk_rung().await;
        assert_eq!(cap(), u32::MAX, "a pass asks idle cache first");
        passes().await;
        grow.walk_rung().await;
        assert_eq!(cap(), 1, "asked and found out: the rung comes down");
    }
    passes().await;
    crate::store::shared::set_fire_cap_for_test(ENGINE, u32::MAX);
    {
        let mut pool = shared.lock().unwrap();
        pool.pool_mut()
            .wait(crate::store::roles::GUEST_WAITER, list, 2);
        pool.set_guest_head(Some((2, 1)));
    }
    planner.pages_freed();
    grow.walk_rung().await;
    assert_eq!(cap(), 1, "a bind outside the queue, idle cache pinned");
    crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| kv.release_reader(pin));
}
