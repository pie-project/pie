//! A speed probe, run by hand on a device: `PIE_XLA_SPEED=<model>` (a
//! catalog name, random weights at the model's own size), optionally
//! `PIE_XLA_SPEED_RUNS=decode:1,decode:64,prefill:512` (the default runs
//! decode at 1, 64 and 256 lanes and prefill of 512 and 2048 tokens). Each
//! run loads the model with the budgets it needs, warms its
//! programs, and reports medians: of windows of decode steps fired back to back, of prompts each timed to its readout. The same file
//! builds against the base build, so two builds compare on one device.

use std::time::Instant;

use engine::fire::Readout;

use crate::common_llm::{Model, Refs, Seq, tokens};

fn set(name: &str, v: u32) {
    crate::common_llm::set_knob(name, v);
}

/// Loads with the warm ladder run (as a server with `prewarm` does), unless
/// `PIE_XLA_SPEED_WARM=0`; with `PIE_XLA_WARM_STRICT=1` a fire outside the
/// ladder fails the probe.
fn warm() {
    let on = std::env::var("PIE_XLA_SPEED_WARM").as_deref() != Ok("0");
    set("PREWARM", u32::from(on));
}

fn quiet(lane: &mut engine::fire::Lane) {
    lane.readout = Readout::None;
}

/// Decode tok/s at `lanes` lanes, each holding `held` prompt rows.
fn decode(name: &str, lanes: u32, held: u32, steps: u32) -> f64 {
    // a lane holds `held` + 103 rows at most
    let context = 512;
    warm();
    set("SLOTS", lanes.max(4));
    set("LANES", lanes.max(8));
    set("TOKENS", (2 * lanes).max(held).max(128));
    let mut model = load(name, context);
    let vocab = model.vocab;
    let mut refs = Refs::of(&model);
    let mut seqs: Vec<Seq> = (0..lanes).map(Seq::new).collect();
    // Prompts, one a fire.
    for (i, seq) in seqs.iter_mut().enumerate() {
        let mut lane = seq.lane(&model, &mut refs, tokens(i as u64, held as usize, vocab));
        quiet(&mut lane);
        model.fire(vec![lane]);
        seq.held += held;
    }
    let step = |model: &mut Model, refs: &mut Refs, seqs: &mut [Seq], last: bool| {
        let lanes: Vec<_> = seqs
            .iter_mut()
            .enumerate()
            .map(|(i, seq)| {
                let mut lane = seq.lane(model, refs, vec![(i as u32 * 7 + seq.held) % vocab]);
                if !(last && i == 0) {
                    quiet(&mut lane);
                }
                lane
            })
            .collect();
        model.fire(lanes);
        for seq in seqs.iter_mut() {
            seq.held += 1;
        }
    };
    for _ in 0..3 {
        step(&mut model, &mut refs, &mut seqs, true);
    }
    // Windows of `steps` steps fired back to back (a readout only at the
    // end, so the device stays fed): the median window's rate.
    let mut rates = Vec::with_capacity(5);
    for _ in 0..5 {
        let t = Instant::now();
        for s in 0..steps {
            step(&mut model, &mut refs, &mut seqs, s + 1 == steps);
        }
        rates.push(f64::from(lanes * steps) / t.elapsed().as_secs_f64());
    }
    median(&mut rates)
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

/// Prefill tok/s of one `len`-token prompt.
fn prefill(name: &str, len: u32, runs: u32) -> f64 {
    let context = (len + 64).next_power_of_two();
    warm();
    set("SLOTS", runs + 3);
    // `runs` prompts after two warm ones, each a working set of its own
    set("LANES", 8);
    set("TOKENS", len);
    let mut model = load(name, context);
    let vocab = model.vocab;
    let mut refs = Refs::of(&model);
    let fire = |model: &mut Model, refs: &mut Refs, slot: u32| {
        let mut seq = Seq::new(slot);
        let lane = seq.lane(model, refs, tokens(u64::from(slot), len as usize, vocab));
        model.fire(vec![lane]);
    };
    for slot in 0..2 {
        fire(&mut model, &mut refs, slot);
    }
    let mut times = Vec::with_capacity(runs as usize);
    for r in 0..runs {
        let t = Instant::now();
        fire(&mut model, &mut refs, 2 + r);
        times.push(t.elapsed().as_secs_f64());
    }
    f64::from(len) / median(&mut times)
}

/// The model as the gate loads it: streaming its weights, or held with
/// `PIE_XLA_WEIGHTS_HELD=1` (what the worker loads unless streaming is on);
/// cut to `PIE_XLA_SPEED_LAYERS` layers where set (a whole model's widths
/// at a depth one chip holds).
fn load(name: &str, context: u32) -> Model {
    let layers = std::env::var("PIE_XLA_SPEED_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok());
    if std::env::var("PIE_XLA_WEIGHTS_HELD").is_ok_and(|v| v == "1") {
        Model::load_held(name, layers, context, 7)
    } else {
        Model::load(name, layers, context, 7)
    }
}

#[test]
fn speed() {
    let Ok(name) = std::env::var("PIE_XLA_SPEED") else {
        return;
    };
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let runs = std::env::var("PIE_XLA_SPEED_RUNS")
        .unwrap_or_else(|_| "decode:1,decode:64,decode:256,prefill:512,prefill:2048".into());
    for run in runs.split(',') {
        let (kind, n) = run.split_once(':').expect("kind:n");
        let n: u32 = n.parse().expect("a count");
        let tps = match kind {
            "decode" => decode(&name, n, 64, 20),
            "prefill" => prefill(&name, n, 7),
            other => panic!("unknown run {other}"),
        };
        eprintln!("SPEED {name} {kind} {n}: {tps:.1} tok/s");
    }
}
