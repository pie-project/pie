//! **Host CONTROL: the runtime arena assigns DISJOINT KV pages under
//! forced-overlap concurrency** (bravo, concurrent-decode hunt).
//!
//! Motivation: charlie's 4090 repro shows concurrent decodes' KV collapsing onto
//! one physical page (`9783`, top-of-free-list) → cross-request clobbering. The
//! open question this pins down: does the collapse originate in the **runtime
//! arena / working-set page assignment** (host, mock-observable via the
//! `kv_page_indices` descriptor the driver receives) or only in the **device**?
//!
//! [`PageProbe`] records every fire's per-request page runs. A per-fire
//! `thread::sleep` keeps the WHOLE fleet outstanding so the `FLEET` decodes are
//! provably concurrently live (they co-batch, R>1). The invariant: concurrently-
//! live requests must hold **disjoint** physical pages — a correct engine gives
//! `FLEET` distinct pages, a collapse gives ~1.
//!
//! **Result (this branch): PASSES — `distinct_write_pages == FLEET`.** So the
//! runtime arena/working-set **retains + isolates correctly under concurrency**;
//! it does NOT reproduce the collapse. (A no-latency run serializes the fleet
//! into R=1 fires that legitimately reuse ~3 top-of-pool pages — correct reuse,
//! not a bug; the latency is what makes the concurrency real.) This localizes
//! charlie's collapse AWAY from the host page-assignment path — consistent with
//! the R>1 batch-merge disjointness proof (`request.rs`) + alpha's Mutex'd-arena
//! no-race analysis.
//!
//! SCOPE: this exercises the co-batched (R>1) concurrent path. It does NOT prove
//! the uncobatched **R=1-overlapping** path charlie observes on-device is host-
//! clean — that needs a scheduler condition the mock can't force here. Kept as a
//! green regression guard on host page isolation under concurrency.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

mod common;
use common::{MockEnv, create_mock_env, inferlets, mock_device::Behavior};

use pie_engine::process;
use pie_engine::program::ProgramName;

const FLEET: usize = 8;
/// Enough physical pages that a correct engine hands each concurrent request its
/// own private KV page(s); a collapse would ignore the headroom.
const NUM_PAGES: usize = 64;
const PROCESS_TIMEOUT: Duration = Duration::from_secs(20);
/// Per-fire simulated device latency — long enough that the WHOLE fleet stays
/// outstanding at once, so the requests are provably concurrently live (not
/// serialized). Under forced overlap, a distinct-page count below `FLEET` would
/// mean a host page-assignment collapse; `== FLEET` proves correct isolation.
const FIRE_LATENCY: Duration = Duration::from_millis(15);

/// Per-fire log of per-request `kv_page_indices` runs (split by `kv_page_indptr`).
type FireLog = Arc<Mutex<Vec<Vec<Vec<u32>>>>>;

/// Records each observed launch's per-request page runs.
struct PageProbe {
    latency: Duration,
    log: FireLog,
}

impl Behavior for PageProbe {
    fn observe_launch(&self, req: &pie_engine::driver::LaunchPlan) {
        let indptr = &req.kv_page_indptr;
        let pages = &req.kv_page_indices;
        let n = indptr.len().saturating_sub(1);
        let mut per_req = Vec::with_capacity(n);
        for r in 0..n {
            let lo = indptr[r] as usize;
            let hi = indptr[r + 1] as usize;
            per_req.push(pages.get(lo..hi).map(<[u32]>::to_vec).unwrap_or_default());
        }
        self.log.lock().unwrap().push(per_req);

        // Stay outstanding so the fleet is provably concurrently live.
        std::thread::sleep(self.latency);
    }
}

struct TestState {
    #[allow(dead_code)]
    env: MockEnv,
    rt: tokio::runtime::Runtime,
    log: FireLog,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        inferlets::build_inferlets();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let log: FireLog = Arc::new(Mutex::new(Vec::new()));
        // Latency overridable (PIE_KVR_LATENCY_MS) to A/B the fire-batching
        // regime: high latency co-batches (R>1); a small non-zero latency keeps
        // fires R=1 yet still overlapping (charlie's on-device condition).
        let latency = std::env::var("PIE_KVR_LATENCY_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_millis)
            .unwrap_or(FIRE_LATENCY);
        let behavior = Arc::new(PageProbe {
            latency,
            log: log.clone(),
        });
        let env = create_mock_env("test-model", 1, NUM_PAGES, behavior);
        let config = env.config();
        rt.block_on(async {
            pie_engine::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt, log }
    })
}

fn program_name(name: &str) -> ProgramName {
    ProgramName::parse(&format!("{name}@0.1.0")).unwrap()
}

/// Launch `FLEET` concurrent decodes (all in flight before any is awaited so the
/// requests are concurrently live) and return once every one has completed.
fn run_concurrent_fleet(s: &TestState) {
    s.rt.block_on(async {
        inferlets::add_and_install("generate").await;
        let rxs: Vec<_> = (0..FLEET)
            .map(|i| {
                let (tx, rx) = tokio::sync::oneshot::channel();
                process::spawn(
                    "retention-user".into(),
                    program_name("generate"),
                    format!(r#"{{"lane":{i}}}"#),
                    None,
                    false,
                    Some(tx),
                )
                .unwrap_or_else(|e| panic!("spawn {i}: {e}"));
                rx
            })
            .collect();
        for rx in rxs {
            let _ = tokio::time::timeout(PROCESS_TIMEOUT, rx).await;
        }
    });
}

/// **Host isolation control:** across the forced-overlap fleet, the per-request
/// WRITE pages (the last page of each `kv_page_indices` run — the new KV token
/// slot) span at least `FLEET` DISTINCT physical pages. A host page-assignment
/// collapse (the runtime analogue of charlie's `9783` domination) would crash
/// this to ~1; `== FLEET` proves the arena retains + isolates correctly under
/// real concurrency, localizing the collapse away from the host descriptor.
#[test]
fn concurrent_decode_retains_disjoint_kv_pages() {
    let s = state();
    run_concurrent_fleet(s);

    let fires = s.log.lock().unwrap();
    let mut write_pages: Vec<u32> = Vec::new();
    let mut all_pages: HashSet<u32> = HashSet::new();
    for fire in fires.iter() {
        for run in fire {
            if let Some(&last) = run.last() {
                write_pages.push(last);
            }
            all_pages.extend(run.iter().copied());
        }
    }
    let distinct_write: HashSet<u32> = write_pages.iter().copied().collect();

    // Usage histogram of the top write pages (a collapse shows as one page
    // carrying almost every fire — charlie's `9783` domination).
    let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for &p in &write_pages {
        *counts.entry(p).or_default() += 1;
    }
    let mut top: Vec<(u32, usize)> = counts.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!(
        "[kv-retention] fires={} write_page_samples={} distinct_write_pages={} all_distinct_pages={}",
        fires.len(),
        write_pages.len(),
        distinct_write.len(),
        all_pages.len(),
    );
    eprintln!(
        "[kv-retention] top write pages (page → #fires): {:?}",
        &top[..top.len().min(6)]
    );

    assert!(
        distinct_write.len() >= FLEET,
        "host KV isolation violated: {} concurrently-live decodes but only {} distinct \
         write pages (correct == {FLEET}) — the runtime arena collapsed concurrent requests \
         onto shared pages. top: {:?}",
        FLEET,
        distinct_write.len(),
        &top[..top.len().min(6)],
    );
}
