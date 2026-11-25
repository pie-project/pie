//! A fire's units against a small device pool, through [`reserve`] and
//! the planner's grant as a pass asks it: a unit is one KV working set's
//! lanes, with every page they read and write (what they share counted).
//! Over-demand is served slower; only a unit past the whole device is
//! refused, by name and at once.
use super::lifecycle_tests::TestContext;
use super::*;
use crate::store::kv::page_table::WorkingSetId;
use crate::store::registry::Stores;

const PAGE: u32 = 4;

/// A device of `pages` pages, a planner the test's process asks (on an
/// engine slot of its own, `engine`), and that process.
fn device(engine: usize, pages: usize) -> (usize, Stores, TestContext) {
    let model = crate::store::registry::register_model(PAGE, &[pages], &[1]);
    let stores = crate::store::registry::get(model, 0);
    let planner = crate::planner::ResidencyPlanner::new(Arc::new(
        crate::planner::RegistryPool::new(model, 0, false),
    ));
    crate::planner::init_planner(0, engine, planner);
    let pid = uuid::Uuid::new_v4();
    crate::planner::place(pid, engine);
    let planner = crate::planner::planner_of(pid).expect("installed");
    planner.register(pid);
    planner.note_admitted(pid);
    let ctx = TestContext {
        id: pid,
        resources: wasmtime::component::ResourceTable::new(),
        tails: Vec::new(),
    };
    (model, stores, ctx)
}

/// A fresh set of `stores` reserving `pages` pages, the first `held`
/// of them written and mapped on the device.
fn set(model: usize, stores: &Stores, pages: u64, held: u64) -> KvWorkingSet {
    let mut kv = stores.kv.lock();
    let ws: WorkingSetId = kv.create_working_set();
    kv.reserve(ws, pages).unwrap();
    if held > 0 {
        let indexes: Vec<u64> = (0..held).collect();
        let need = kv.write_demand(ws, &indexes).unwrap();
        let mut granted = kv.reserve_device_pages(need).unwrap();
        let prepared = kv
            .prepare_write_reserved(ws, &indexes, &mut granted)
            .unwrap();
        let commits: Vec<_> = (0..held)
            .map(|_| crate::store::kv::write::PageCommit {
                token_hashes: Vec::new(),
                page_hash: None,
            })
            .collect();
        let seq = kv.publish_prepared(prepared, &commits).unwrap();
        kv.settle(seq);
    }
    KvWorkingSet::new(model, 0, ws, PAGE)
}

/// One fire writing `spans` (in pages), a set each, as a pass's lanes.
async fn fire(
    ctx: &mut TestContext,
    model: usize,
    stores: &Stores,
    sets: &[(&KvWorkingSet, std::ops::Range<u64>)],
) -> Result<Reserved, String> {
    let sets: Vec<(&KvWorkingSet, Written)> = sets
        .iter()
        .map(|(set, span)| (*set, Written::Span(span.clone())))
        .collect();
    let failure = PipelineFailure::default();
    let none = || Ok(rs::RsPlan::Groups(Vec::new()));
    let id = uuid::Uuid::new_v4();
    let asked = reserve(ctx, (model, 0), stores, id, &failure, &sets, &[], &[], none, false);
    tokio::time::timeout(std::time::Duration::from_secs(5), asked)
        .await
        .expect("answered at once, never parked")
}

/// A lane whose rows alone are past the whole device is refused at
/// its submit, by name, and the planner is never asked to hold it.
#[tokio::test(flavor = "current_thread")]
async fn a_lane_past_the_device_alone_is_refused_at_submit() {
    let (model, stores, mut ctx) = device(92, 8);
    let lane = set(model, &stores, 9, 0);
    let refused = fire(&mut ctx, model, &stores, &[(&lane, 0..9)])
        .await
        .err()
        .expect("nine pages of eight");
    assert!(
        refused.contains("device memory can never hold this need: 9 pages")
            && refused.contains("past the whole 8-page device pool"),
        "{refused}"
    );
    let planner = crate::planner::planner_of(ctx.id).unwrap();
    assert!(!planner.demand_waits(), "nothing parked");
}

/// Lanes that each fit but pass the device together are no lane past
/// it: none is refused as one.
#[tokio::test(flavor = "current_thread")]
async fn lanes_past_the_device_only_together_are_no_lane_past_it() {
    let (model, stores, mut ctx) = device(94, 8);
    let lanes: Vec<KvWorkingSet> = (0..2).map(|_| set(model, &stores, 5, 0)).collect();
    let sets: Vec<(&KvWorkingSet, Written)> = lanes
        .iter()
        .map(|set| (set, Written::Span(0..5)))
        .collect();
    let plan = rs::RsPlan::Groups(Vec::new());
    let alone = alone_demands(&stores, &sets, &[], &[], &plan).unwrap();
    let planner = crate::planner::planner_of(ctx.id).unwrap();
    assert_eq!(alone.iter().map(|unit| unit.kv_pages).collect::<Vec<_>>(), [5, 5]);
    assert!(alone.iter().all(|&unit| planner.past_device(unit).is_none()));
    let whole = crate::planner::Demand {
        kv_pages: 10,
        rs: Default::default(),
    };
    assert!(planner.past_device(whole).is_some(), "the whole would be");
    let _ = &mut ctx;
}
