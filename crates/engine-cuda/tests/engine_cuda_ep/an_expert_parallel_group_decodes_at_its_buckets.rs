#![cfg(feature = "cuda")]

//! How long an expert-parallel group's decode step takes beside one engine
//! holding every expert: each decode bucket (one lane, two), every rank
//! busy or rank 0 alone with the others idle, and a group that rebalances
//! before every step. Ignored: it measures, and asserts nothing of the
//! numbers (`--ignored --nocapture` prints them).

use std::time::{Duration, Instant};

use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step};
use engine_cuda::lockstep::Rank;
use poem::{Dtype, Platform, Request};

use crate::micro::{MICRO, boot, checkpoint, contract_for, ordinals, prompt, ranks, request, step};

/// Decode steps a measure warms up with, then times: with the prompt,
/// inside the context.
const WARM: usize = 50;
const STEPS: usize = 150;
const PROMPT: u32 = 37;

/// A decode step of one token for each of `lanes` seats.
fn decode(lanes: u32, word: u64) -> Step {
    Step {
        lanes: (0..lanes)
            .map(|slot| Lane {
                slot,
                word,
                tokens: vec![5 + slot],
                readout: Readout::Last,
                ..Lane::default()
            })
            .collect(),
        attachments: Vec::new(),
        media: Vec::new(),
        voxels: Vec::new(),
    }
}

fn fire(engine: &mut dyn Engine, step: &Step) {
    let mut ticket = engine
        .submit(&FrameSubmission::of(step.clone()))
        .expect("the frame fires");
    engine.settle_frame(&mut ticket).expect("the frame settles");
}

struct Took(Vec<Duration>);

impl std::fmt::Display for Took {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut sorted = self.0.clone();
        sorted.sort();
        let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize].as_secs_f64() * 1e6;
        write!(
            f,
            "p50 {:7.1} us  p90 {:7.1} us  p99 {:7.1} us",
            at(0.5),
            at(0.9),
            at(0.99)
        )
    }
}

fn timed(mut once: impl FnMut()) -> Took {
    (0..WARM).for_each(|_| once());
    Took(
        (0..STEPS)
            .map(|_| {
                let began = Instant::now();
                once();
                began.elapsed()
            })
            .collect(),
    )
}

/// Every rank's `lanes` seats opened afresh and prefilled.
fn prefill(ranks: &mut [Rank], lanes: u32, word: u64) {
    for rank in ranks {
        for slot in 0..lanes {
            rank.open(slot).expect("the seat opens");
            fire(rank, &step(slot, word, &prompt(PROMPT, slot + 1)));
        }
    }
}

#[test]
#[ignore = "measures; run with --ignored --nocapture"]
fn an_expert_parallel_group_decodes_at_its_buckets() {
    let _alone = crate::alone();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let size = ranks();
    let ordinals = ordinals(size);
    // `PIE_EP_MODEL` names another MoE fixture to measure (a wider one: `qwen3-a3b-uncached-bank`).
    let model = std::env::var("PIE_EP_MODEL").unwrap_or_else(|_| MICRO.to_owned());
    let whole = poem_compiler::catalog::trace_of(&model, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let lowered =
        poem_compiler::expert::lower(whole.clone(), size, 0).expect("the micro's MoE lowers");
    let word = whole.facts.word(&Request::new(PROMPT, false));

    for &ordinal in ordinals.iter().take(engine_cuda::device::ctx::count()) {
        let (_dir, path) = checkpoint(&whole, &format!("decode-one-{ordinal}"));
        let boot = boot(ordinal);
        let mut one = engine_cuda::open(boot, contract_for).expect("the engine opens");
        one.load(request(&whole, &path)).expect("the micro loads");
        for lanes in [1, 2] {
            for slot in 0..lanes {
                one.open(slot).expect("the seat opens");
                fire(&mut one, &step(slot, word, &prompt(PROMPT, slot + 1)));
            }
            let took = timed(|| fire(&mut one, &decode(lanes, word)));
            eprintln!("ep 1 on cuda:{ordinal}, {lanes} lane(s):          {took}");
        }
    }

    for every in [None, Some(1)] {
        let boots = (ordinals.iter())
            .map(|&ordinal| {
                let mut boot = boot(ordinal);
                boot.knobs.expert_rebalance_steps = every;
                boot.knobs.expert_load_window = every;
                boot
            })
            .collect();
        let mut ranks =
            engine_cuda::lockstep::open_lockstep(boots, contract_for).expect("the group opens");
        let (_dir, path) = checkpoint(&whole, "decode-group");
        std::thread::scope(|scope| {
            for rank in &mut ranks {
                let request = request(&lowered, &path);
                scope.spawn(move || rank.load(request).expect("the rank loads"));
            }
        });
        let label = if every.is_some() { ", rebalancing" } else { "" };
        for lanes in [1, 2] {
            let step = decode(lanes, word);
            // Every rank busy, each bringing its steps from a thread of its
            // own as a worker's lanes do: announcing each before it brings
            // it, or (a decode's next step is not known before its last
            // settles, so a lane has none queued to announce) not.
            for announced in [true, false] {
                prefill(&mut ranks, lanes, word);
                let took: Vec<Took> = std::thread::scope(|scope| {
                    let busy: Vec<_> = (ranks.iter_mut())
                        .map(|rank| {
                            let step = &step;
                            scope.spawn(move || {
                                timed(|| {
                                    if announced {
                                        rank.expect_fire(step);
                                    }
                                    fire(rank, step);
                                })
                            })
                        })
                        .collect();
                    busy.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let how = if announced {
                    "announced"
                } else {
                    "unannounced"
                };
                for (r, took) in took.iter().enumerate() {
                    eprintln!(
                        "ep {size}, {lanes} lane(s), rank {r} of all busy, {how}{label}: {took}"
                    );
                }
            }
            prefill(&mut ranks, lanes, word);
            let alone = timed(|| fire(&mut ranks[0], &step));
            eprintln!("ep {size}, {lanes} lane(s), rank 0 alone{label}:    {alone}");
        }
        if every.is_some() {
            eprintln!(
                "{} slot copies across the group's rebalances",
                ranks[0].moved_experts()
            );
        }
    }
}
