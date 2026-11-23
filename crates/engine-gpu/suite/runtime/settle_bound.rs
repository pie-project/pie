//! Under a stated bound on one GPU submission (`engine_gpu::submit`), the
//! dense settle (`Shell::settle_dense`) submits no fire past it: each
//! range's fires are admitted as serving admits its step, the range checked
//! only where serving submits that step whole and each option only where
//! its step, priced by the curves, fits; a range serving splits keeps its
//! kernel from the curves, unchecked. Here the bound is simulated between a
//! one-row step's estimate and the load's widest step's.

use std::time::Duration;

use engine::Engine;
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine_gpu::submit::StepRows;

use super::outputs::contract_for;
use crate::backend;
use crate::model_logits::fixture;

const MODEL: &str = "qwen3-micro-text";
const TOKENS: u32 = 256;

#[test]
fn the_settle_submits_no_fire_past_the_bound() {
    // Alone on the device: its fresh profile directory measures the rates
    // again, and a rates step timed beside other tests' fires drops points.
    crate::gpu::sole();
    // A profile directory of this test's own: the process's shared one
    // keeps each model's settle, so a load after another test's of the
    // same model takes that settle and walks no range.
    let profiles = std::env::temp_dir().join(format!(
        "pie-{}-settle-bound-profiles-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&profiles);
    std::fs::create_dir_all(&profiles).expect("a profile directory");
    let boot = backend::DeviceBoot {
        submit_bound: Some(Duration::from_secs(2)),
        profiles: Some(profiles.clone()),
        ..crate::gpu::device_boot()
    };
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return;
    }
    let fixture = fixture(MODEL, None);
    let mut engine = backend::Engine::new(boot, contract_for);
    engine
        .load(LoadRequest {
            trace: fixture.trace.clone(),
            checkpoint: Checkpoint::Path(fixture.container.clone()),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: TOKENS,
                buckets: vec![TOKENS],
                max_adapters: 0,
                page_size: 16,
                max_context: 2 * TOKENS,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency: Residency {
                device: crate::gpu::share(),
                ..Residency::default()
            },
            ordinal: 0,
            frames_in_flight: 1,
            steps: Default::default(),
        })
        .expect("the model loads");
    let shell = engine.loaded_mut().expect("loaded");
    // Every estimate read at the ratio the rates step measured alone, as
    // the settle reads it once held (`Shell::hold_to`).
    shell.hold_to(f64::MAX);
    let held = shell
        .submit_bound()
        .expect("the bound the boot states")
        .clone();
    let step = |rows: u32| StepRows {
        rows,
        lanes: 1,
        keys: rows,
        prompt: rows > 1,
    };
    // Between a one-row step's estimate and the widest's: the decode
    // ranges' steps fit whole, the widest's does not.
    let at = |rows: u32| held.step_ns(step(rows)).0;
    let tight = (at(1) * at(TOKENS)).sqrt();
    shell.hold_to(tight);
    shell
        .settle_dense()
        .expect("the settle runs under the bound");
    let settled = shell.settled().to_vec();
    assert!(!settled.is_empty(), "the settle walked no range");
    for range in &settled {
        assert_eq!(range.bound_ns, Some(tight), "{range:?}");
        for &estimate in &range.estimates {
            assert!(
                estimate <= tight,
                "{}-{} rows: a fire admitted at {:.3} ms past the bound of {:.3} ms",
                range.bottom,
                range.top,
                estimate / 1e6,
                tight / 1e6
            );
        }
    }
    // The widest range, past the bound whole, fires no whole step: it is
    // weighed on its widest plane alone, within the bound.
    let widest = settled.last().expect("a range");
    assert!(
        widest.estimates.is_empty(),
        "the widest range fired a whole step: {widest:?}"
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&profiles);
}
