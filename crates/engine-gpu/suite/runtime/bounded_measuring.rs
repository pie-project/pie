//! On a device whose platform states a bound on one submission, no load runs
//! a submission past it: a load with no stored or committed measurement of
//! the model (a fresh profile directory) times what it lacks (the dense
//! curves at the model's planes, the settle; the roofline where no layer
//! states it) and every frame it waits on, measuring or serving, stays under
//! the bound by the device's timestamps (where the backend reads them).

use engine::Engine;
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};

use super::outputs::contract_for;
use crate::backend;
use crate::model_logits::fixture;

const MODEL: &str = "qwen3-micro-text";
const TOKENS: u32 = 256;

#[test]
fn a_load_with_no_measurement_under_a_stated_bound_runs_no_frame_past_it() {
    let bound = std::time::Duration::from_millis(25);
    let profiles = std::env::temp_dir().join(format!(
        "pie-{}-bounded-measuring-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&profiles);
    let boot = backend::DeviceBoot {
        submit_bound: Some(bound),
        profiles: Some(profiles.clone()),
        ..crate::gpu::device_boot()
    };
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return;
    }
    let fixture = fixture(MODEL, None);
    let mut engine = backend::Engine::new(boot, contract_for);
    backend::watch_frames(true);
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
        })
        .expect("the model loads under the bound");
    let longest = backend::longest_frame_ns();
    backend::watch_frames(false);
    let shell = engine.loaded_mut().expect("loaded");
    assert!(shell.submit_bound().is_some(), "every serving step is held to the bound");
    assert!(
        shell.settled().iter().any(|range| range.checked),
        "the load measured its settle"
    );
    let stored = std::fs::read_dir(&profiles).is_ok();
    let _ = std::fs::remove_dir_all(&profiles);
    assert!(stored, "what it measured is stored");
    if let Some(longest) = longest {
        assert!(longest > 0, "the device timed no frame");
        assert!(
            longest <= bound.as_nanos() as u64,
            "a frame ran {:.2} ms on the device, past the {:.0} ms bound",
            longest as f64 / 1e6,
            bound.as_secs_f64() * 1e3
        );
        eprintln!(
            "the longest frame the load waited on: {:.2} ms against a {:.0} ms bound",
            longest as f64 / 1e6,
            bound.as_secs_f64() * 1e3
        );
    }
}
