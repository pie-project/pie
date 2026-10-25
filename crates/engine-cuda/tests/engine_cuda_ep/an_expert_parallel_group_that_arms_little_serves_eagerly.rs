#![cfg(feature = "cuda")]

//! An expert-parallel group whose bodies' allowance arms nothing (or only
//! some of its compositions) still serves: its ranks seal what they armed
//! however much that was, so past arming no rank captures, and every
//! composition not armed walks eagerly. A capture past arming would
//! instantiate while a peer, firing a step of its own beside it, spins in a
//! collective. Each rank brings its prompt and a few decode steps, every
//! rank its own at its own pace, then rank 0 a prompt alone; every step
//! settles with its logits written. (What those logits are is the other
//! tests': steps the ranks bring at their own pace meet at buckets no one
//! engine's fire reproduces.)

use engine::Engine;
use engine_cuda::{Graphs, Recording};
use poem::{Dtype, Platform, Request};

use crate::micro::{
    MICRO, answer, boot, checkpoint, contract_for, ordinals, prompt, ranks, request, step,
};

/// Steps of one decode token each, after the prompts.
const DECODES: u32 = 4;

#[test]
fn an_expert_parallel_group_that_arms_little_serves_eagerly() {
    let _alone = crate::alone();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    // Nothing armed, then the first few bodies' worth.
    for mem_megabytes in [0, 2] {
        check(mem_megabytes);
    }
}

fn check(mem_megabytes: u32) {
    let size = ranks();
    let ordinals = ordinals(size);
    let whole = poem_compiler::catalog::trace_of(MICRO, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let lowered =
        poem_compiler::expert::lower(whole.clone(), size, 0).expect("the micro's MoE lowers");
    let word = |len: usize| whole.facts.word(&Request::new(len as u32, false));
    let prompts: Vec<Vec<u32>> = (0..size).map(|r| prompt(37 + 15 * r, r + 1)).collect();
    let profile =
        || engine_cuda::chip::pinned(Graphs::On, Some(Recording::Bodies { mem_megabytes }));

    let boots = (ordinals.iter())
        .map(|&ordinal| {
            let mut boot = boot(ordinal);
            boot.knobs.chip = Some(profile());
            boot
        })
        .collect();
    let mut ranks =
        engine_cuda::lockstep::open_lockstep(boots, contract_for).expect("the group opens");
    let (dir, path) = checkpoint(&whole, &format!("little-group-{mem_megabytes}"));
    std::thread::scope(|scope| {
        for rank in &mut ranks {
            let request = request(&lowered, &path);
            scope.spawn(move || rank.load(request).expect("the rank loads"));
        }
    });

    // Every rank its prompt and decodes, each on a thread of its own as a
    // worker's lanes bring them: steps the ranks bring at different times
    // and buckets, an idle rank serving the others' routes.
    let got: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
        let fired: Vec<_> = (ranks.iter_mut().zip(&prompts).enumerate())
            .map(|(r, (rank, p))| {
                let word = &word;
                scope.spawn(move || {
                    rank.open(0).expect("the seat opens");
                    let mut answers = vec![answer(rank, &step(0, word(p.len()), p))];
                    for d in 0..DECODES {
                        let len = p.len() + 1 + d as usize;
                        answers.push(answer(rank, &step(0, word(len), &[5 + r as u32 + d])));
                    }
                    answers
                })
            })
            .collect();
        fired.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (r, got) in got.iter().enumerate() {
        for (at, got) in got.iter().enumerate() {
            written(
                got,
                &format!("rank {r}'s step {at} ({mem_megabytes} MiB of bodies)"),
            );
        }
    }

    // Rank 0 alone: the others idle through its step.
    let p = &prompts[0];
    ranks[0].open(1).expect("the seat opens");
    let got = answer(&mut ranks[0], &step(1, word(p.len()), p));
    written(
        &got,
        &format!("rank 0's prompt with the others idle ({mem_megabytes} MiB of bodies)"),
    );
    drop(dir);
}

/// Logits a fire wrote: a vocabulary of finite values with a spread.
fn written(got: &[f32], what: &str) {
    assert!(!got.is_empty(), "{what}: no logits");
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{what}: a logit is not finite"
    );
    let spread = got.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - got.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what}: the logits are a rectangle nothing wrote"
    );
}
