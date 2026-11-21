//! What a step split under the submit bound costs on the device against the
//! same step whole: Qwen3-0.6B (synthetic weights), a 360-row prompt step
//! whole (replayed from its recording), whole but walked eagerly, and in
//! runs of layers. A measurement: run on its own.

use std::path::Path;
use std::time::{Duration, Instant};

use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step};
use engine::load::{Budgets, Checkpoint, LoadRequest};

use crate::backend;
use crate::model_logits::{fixture, word};

fn contract_for(
    trace: &poem_ir::Trace,
    path: &Path,
) -> Result<checkpoint::contract::ModelContract, String> {
    let source = ztensor::Source::open(path).map_err(|why| why.to_string())?;
    poem::import::own_contract(&source, &trace.params, 1, backend::PLATFORM)
        .map_err(|why| why.to_string())
}

#[test]
#[ignore = "a measurement: run on its own"]
fn a_split_step_costs_its_submits_alone() {
    const ROWS: u32 = 360;
    let fixture = fixture("qwen3-0.6b", None);
    let device = backend::DeviceBoot {
        submit_bound: Some(Duration::from_secs(2)),
        ..crate::gpu::device_boot()
    };
    let mut engine = backend::Engine::new(device, contract_for);
    engine
        .load(LoadRequest {
            trace: fixture.trace.clone(),
            checkpoint: Checkpoint::Path(fixture.container.clone()),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: 512,
                buckets: vec![512],
                max_adapters: 0,
                page_size: 16,
                max_context: 4096,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency: engine::load::Residency::default(),
            ordinal: 0,
            frames_in_flight: 1,
        })
        .expect("qwen3-0.6b loads");
    let prompt: Vec<u32> = (0..ROWS).map(|at| (at * 37 + 11) % 1000).collect();
    let step_on = |slot: u32| Step {
        lanes: vec![Lane {
            slot,
            word: word(&fixture.trace, ROWS),
            tokens: prompt.clone(),
            readout: Readout::Last,
            ..Lane::default()
        }],
        attachments: Vec::new(),
        media: Vec::new(),
        voxels: Vec::new(),
    };
    let rows = engine_gpu::submit::StepRows::of(&step_on(0));
    let held = engine
        .loaded()
        .expect("loaded")
        .submit_bound()
        .expect("bound")
        .clone();
    let (whole, _) = held.step_ns(rows);
    let tight = whole / 4.6;
    eprintln!(
        "whole {:?}; under the tight bound {:?}",
        held.admit(rows),
        engine_gpu::submit::SubmitBound {
            bound_ns: tight,
            ..held.clone()
        }
        .admit(rows)
    );
    // Each mode on a slot of its own, so each fire of every mode attends
    // the same keys (each slot grows by the prompt a round).
    let fire = |engine: &mut backend::Engine, slot: u32, bound: f64, profile: bool| {
        engine.loaded_mut().expect("loaded").hold_to(bound);
        if profile {
            engine_vulkan::encode::reset_kernel_profile();
            engine_vulkan::encode::profile_kernels(true);
        }
        let started = Instant::now();
        let mut ticket = engine
            .submit(&FrameSubmission::of(step_on(slot)))
            .expect("the step fires");
        engine.settle_frame(&mut ticket).expect("settles");
        let took = started.elapsed().as_secs_f64() * 1e3;
        if profile {
            engine_vulkan::encode::profile_kernels(false);
            let mut rows = engine_vulkan::encode::kernel_profile();
            rows.sort_by_key(|r| std::cmp::Reverse(r.1));
            let total: u64 = rows.iter().map(|r| r.1).sum();
            eprintln!("slot {slot}: kernels {:.3} ms over {} kinds", total as f64 / 1e6, rows.len());
            for (name, ns, count) in rows.iter().take(10) {
                eprintln!("  {name}: {count} x, {:.3} ms", *ns as f64 / 1e6);
            }
        }
        took
    };
    const ROUNDS: usize = 8;
    let (mut whole_ms, mut split_ms, mut walked_ms) = (Vec::new(), Vec::new(), Vec::new());
    for round in 0..ROUNDS {
        let profile = round == ROUNDS - 1;
        whole_ms.push(fire(&mut engine, 0, 2e9, profile));
        split_ms.push(fire(&mut engine, 1, tight, profile));
    }
    engine.loaded_mut().expect("loaded").walk_every_fire();
    for round in 0..ROUNDS {
        walked_ms.push(fire(&mut engine, 2, 2e9, round == ROUNDS - 1));
    }
    for (what, ms) in [("whole, replayed", &whole_ms), ("split", &split_ms), ("whole, walked", &walked_ms)] {
        eprintln!(
            "{what}: {}",
            ms.iter().map(|t| format!("{t:.1}")).collect::<Vec<_>>().join(" ")
        );
    }
}
