//! What a step split under the submit bound costs on the device against the
//! same step whole: Qwen3-0.6B (synthetic weights), one prompt step whole
//! (replayed from its recording), whole but walked eagerly, and split in
//! runs of layers, each mode on a seat of its own so every round's fire of
//! each attends the same keys. A measurement: run on its own.
//!
//! The device's profile is read as every suite load reads it,
//! `$PIE_HOME/profiles` where `PIE_HOME` is set (point it at the serving
//! home so the rates, and the kernels they pick, are serving's), else the
//! suite's own directory in the temp dir.
//!
//! Under the device's bound on one submission (the platform's, else the 2 s
//! the test states) nothing is fired that the bound does not admit: the
//! step's rows are the most whose whole step is admitted whole at the
//! floor of the device's clock table, at the keys of the test's last round,
//! so the whole modes fit the bound even at the least clock; a device on
//! which no row count does is refused by name, and nothing fires.

use std::path::Path;
use std::time::{Duration, Instant};

use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step};
use engine::load::{Budgets, Checkpoint, LoadRequest};
use engine_gpu::submit::{Admitted, StepRows};

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

/// The rounds each mode fires: each grows its seat by the step's rows.
const ROUNDS: u32 = 6;
/// The most rows the step is tried at.
const MOST: u32 = 360;

#[test]
#[ignore = "a measurement: run on its own"]
fn a_split_step_costs_its_submits_alone() {
    let fixture = fixture("qwen3-0.6b", None);
    let boot = crate::gpu::device_boot();
    eprintln!("profiles: {:?}", boot.profiles);
    let bound = engine_gpu::rates::stated_submit_bound().unwrap_or(Duration::from_secs(2));
    let device = backend::DeviceBoot {
        submit_bound: Some(bound),
        ..boot
    };
    let context = MOST * (ROUNDS + 1);
    let mut engine = backend::Engine::new(device, contract_for);
    let loaded = engine
        .load(LoadRequest {
            trace: fixture.trace.clone(),
            checkpoint: Checkpoint::Path(fixture.container.clone()),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: 512,
                buckets: vec![512],
                max_adapters: 0,
                page_size: 16,
                max_context: context.next_power_of_two(),
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
    // Every segment the pool's range holds, backed now: three seats grow by
    // the step's rows a round, past what the load backs at its start.
    let segments = loaded.caps.chunks.segments;
    if segments.growth != engine::chunk::Growth::Fixed {
        engine
            .commit_segments(segments.max)
            .expect("the pool's segments commit");
    }
    let held = engine
        .loaded()
        .expect("loaded")
        .submit_bound()
        .expect("the bound the boot states")
        .clone();
    eprintln!(
        "bound {} ms; clock table {}; floor factor {:.2}; stored ratio {:.3}",
        bound.as_millis(),
        held.clock.as_ref().map_or_else(|| "none read".into(), ToString::to_string),
        held.floor_factor(),
        held.stored_ratio(),
    );
    // The most rows whose step, at the last round's keys, is admitted whole
    // at the stored ratio and the floor (what `hold_to` admits at).
    let whole_at = |rows: u32| StepRows {
        rows,
        lanes: 1,
        keys: rows * ROUNDS,
        prompt: true,
    };
    let rows = (1..=MOST / 8)
        .rev()
        .map(|eighths| eighths * 8)
        .find(|&rows| matches!(held.admit(whole_at(rows)), Admitted::Whole { .. }))
        .unwrap_or_else(|| {
            panic!(
                "no prompt step of 8 to {MOST} rows is admitted whole under the {} ms bound at \
                 the floor of the clock table (factor {:.2}): the whole modes are not fired",
                bound.as_millis(),
                held.floor_factor()
            )
        });
    let (whole_ns, _) = held.step_ns(whole_at(rows));
    let bound_ns = bound.as_nanos() as f64;
    // Split into runs of about a fifth of the step each, under the bound.
    let tight = (whole_ns / 4.6).min(bound_ns);
    eprintln!(
        "a {rows}-row step: whole {:?}; split {:?}",
        held.admit(whole_at(rows)),
        engine_gpu::submit::SubmitBound {
            bound_ns: tight,
            ..held.clone()
        }
        .admit(whole_at(rows))
    );
    let prompt: Vec<u32> = (0..rows).map(|at| (at * 37 + 11) % 1000).collect();
    let step_on = |slot: u32| Step {
        lanes: vec![Lane {
            slot,
            word: word(&fixture.trace, rows),
            tokens: prompt.clone(),
            readout: Readout::Last,
            ..Lane::default()
        }],
        attachments: Vec::new(),
        media: Vec::new(),
        voxels: Vec::new(),
    };
    // One fire of `mode` on `slot` held to `bound`: its wall, ms, and its
    // device time as the bound reads it (its timestamps), ms.
    let fire = |engine: &mut backend::Engine, slot: u32, bound: f64, profile: Option<&str>| {
        engine.loaded_mut().expect("loaded").hold_to(bound);
        if profile.is_some() {
            engine_vulkan::encode::reset_kernel_profile();
            engine_vulkan::encode::profile_kernels(true);
        }
        let started = Instant::now();
        let mut ticket = engine
            .submit(&FrameSubmission::of(step_on(slot)))
            .expect("the step fires");
        engine.settle_frame(&mut ticket).expect("the step settles");
        let wall = started.elapsed().as_secs_f64() * 1e3;
        let shell = engine.loaded_mut().expect("loaded");
        shell.track();
        let device = shell.last_landed_ns().map_or(f64::NAN, |ns| ns as f64 / 1e6);
        let picks = shell.last_picks().to_vec();
        if let Some(mode) = profile {
            engine_vulkan::encode::profile_kernels(false);
            let mut kinds = engine_vulkan::encode::kernel_profile();
            kinds.sort_by_key(|k| std::cmp::Reverse(k.1));
            let total: u64 = kinds.iter().map(|k| k.1).sum();
            if kinds.is_empty() {
                eprintln!(
                    "{mode}: per-kernel times none (a replayed recording writes no per-kernel \
                     timestamps); device {device:.3} ms"
                );
            } else {
                eprintln!(
                    "{mode}: per-kernel {:.3} ms over {} kinds; device {device:.3} ms",
                    total as f64 / 1e6,
                    kinds.len()
                );
                for (name, ns, count) in kinds.iter().take(10) {
                    eprintln!("  {mode}: {name}: {count} x, {:.3} ms", *ns as f64 / 1e6);
                }
            }
        }
        (wall, device, picks)
    };
    let modes = ["whole-replayed", "split", "whole-walked"];
    let mut times: Vec<Vec<(f64, f64)>> = vec![Vec::new(); modes.len()];
    // Each mode's kernel picks as its first round walked them (the
    // replayed mode's recording is walked then).
    let mut picks: Vec<Vec<String>> = vec![Vec::new(); modes.len()];
    let mut keep = |at: usize, round: u32, (wall, device, walked): (f64, f64, Vec<String>)| {
        times[at].push((wall, device));
        if round == 0 {
            picks[at] = walked;
        }
    };
    for round in 0..ROUNDS {
        let last = round == ROUNDS - 1;
        keep(0, round, fire(&mut engine, 0, bound_ns, last.then_some(modes[0])));
        keep(1, round, fire(&mut engine, 1, tight, last.then_some(modes[1])));
    }
    engine.loaded_mut().expect("loaded").walk_every_fire();
    for round in 0..ROUNDS {
        let last = round == ROUNDS - 1;
        keep(2, round, fire(&mut engine, 2, bound_ns, last.then_some(modes[2])));
    }
    for (mode, picks) in modes.iter().zip(&picks) {
        eprintln!("{mode}: kernel picks {picks:?}");
    }
    for (mode, times) in modes.iter().zip(&times) {
        eprintln!(
            "{mode}: wall / device ms by round: {}",
            times
                .iter()
                .map(|(wall, device)| format!("{wall:.1}/{device:.1}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}
