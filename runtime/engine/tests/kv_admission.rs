//! **Admission prices actual KV use, not the declared output budget.**
//!
//! The live defect this pins down (tts-bench emergency #2017, H200): a
//! ~20k-token prompt with `max_completion_tokens=32768` was rejected at frame
//! admission with `frame commit impossible` even though the actual generation
//! would have fit — the frame's KV demand was sized by the DECLARED budget
//! (prompt + max output), vLLM-style incremental paging never reserves that.
//!
//! The guest (`declared-budget`) is the tts-bench decoder inferlet's exact
//! fire shape: a pages channel spanning the declared pool, a logical
//! `reserve` that grows one page at a time, and a device loop-carried decode.
//! The physical pool here holds the prompt plus the ACTUAL generation with
//! one page of headroom — an engine that admits on actual use runs it to
//! completion; an engine that prices the declaration rejects it.
//!
//! The probe records every launch's admission-relevant figures straight off
//! the driver submission: `required_kv_pages` (the frame-union figure the
//! CUDA driver commits its arena against) and the pages the launch actually
//! references.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

mod common;
use common::{MockEnv, create_mock_env, inferlets};

use common::mock_device::Behavior;
use pie_engine::inferlet::process;
use pie_engine::inferlet::program::ProgramName;

/// Physical pool: 8 pages x 16-token pages = 128 tokens. Holds the prompt
/// (a few tokens) + 4 generated tokens many times over, but NOT the declared
/// 1900-token budget (~120 pages).
const NUM_PAGES: usize = 8;
const DECLARED_BUDGET_TOKENS: u32 = 1900;
const ACTUAL_TOKENS: usize = 4;
const PROCESS_TIMEOUT: Duration = Duration::from_secs(20);

/// Per-launch admission figures, straight off the driver submission.
#[derive(Debug, Clone, Copy, Default)]
struct AdmissionFigures {
    required_kv_pages: u32,
    max_page_index_plus_one: u32,
    translation_high_water: u32,
}

type FigureLog = Arc<Mutex<Vec<AdmissionFigures>>>;

struct AdmissionProbe {
    log: FigureLog,
}

impl Behavior for AdmissionProbe {
    fn observe_launch(&self, req: &pie_engine::driver::LaunchPlan) {
        let max_page = req
            .kv_page_indices
            .iter()
            .copied()
            .max()
            .map_or(0, |p| p + 1);
        let translation = req
            .kv_translation
            .iter()
            .copied()
            .max()
            .map_or(0, |p| p + 1);
        self.log.lock().unwrap().push(AdmissionFigures {
            required_kv_pages: req.required_kv_pages,
            max_page_index_plus_one: max_page,
            translation_high_water: translation,
        });
    }
}

struct TestState {
    #[allow(dead_code)]
    env: MockEnv,
    rt: tokio::runtime::Runtime,
    log: FigureLog,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        inferlets::build_inferlets();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let log: FigureLog = Arc::new(Mutex::new(Vec::new()));
        let behavior = Arc::new(AdmissionProbe { log: log.clone() });
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

/// A guest declaring a 1900-token output budget but generating 4 tokens runs
/// to completion on an 8-page pool, and no launch's admission figures ever
/// price the declared budget.
#[test]
fn declared_budget_is_not_priced_at_admission() {
    let s = state();
    let result = s.rt.block_on(async {
        inferlets::add_and_install("declared-budget").await;
        let (tx, rx) = tokio::sync::oneshot::channel();
        process::spawn(
            "admission-user".into(),
            program_name("declared-budget"),
            format!("{DECLARED_BUDGET_TOKENS},{ACTUAL_TOKENS}"),
            None,
            false,
            Some(tx),
        )
        .expect("spawn declared-budget");
        tokio::time::timeout(PROCESS_TIMEOUT, rx).await
    });

    let fires = s.log.lock().unwrap().clone();
    eprintln!("[kv-admission] observed {} launches:", fires.len());
    for (i, f) in fires.iter().enumerate() {
        eprintln!(
            "  launch {i}: required_kv_pages={} max_page_index+1={} translation_high_water={}",
            f.required_kv_pages, f.max_page_index_plus_one, f.translation_high_water
        );
    }

    let outcome = match result {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(_)) => panic!("declared-budget result channel dropped"),
        Err(_) => panic!("declared-budget timed out"),
    };
    match outcome {
        Ok(text) => {
            assert!(
                text.contains(&format!("generated {ACTUAL_TOKENS} tokens")),
                "unexpected completion: {text}"
            );
        }
        Err(error) => panic!(
            "declared-budget was rejected on a pool that holds its actual use \
             (the admission-prices-the-declaration defect): {error}"
        ),
    }

    // No launch may demand more physical pages than the pool holds — that is
    // the figure the CUDA driver's folded admission compares against its
    // elastic budget, and the declared budget (~120 pages) dwarfs the pool.
    let peak = fires
        .iter()
        .map(|f| {
            f.required_kv_pages
                .max(f.max_page_index_plus_one)
                .max(f.translation_high_water)
        })
        .max()
        .unwrap_or(0);
    assert!(
        peak <= NUM_PAGES as u32,
        "a launch priced beyond the physical pool (peak={peak}, pool={NUM_PAGES}): \
         admission is charging the declared budget"
    );
}
