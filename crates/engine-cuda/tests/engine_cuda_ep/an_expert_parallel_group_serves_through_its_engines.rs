#![cfg(feature = "cuda")]

//! An expert-parallel group as a worker drives one: `open_lockstep` hands
//! back one engine a rank, each loaded and submitted to on a thread of its
//! own, every rank reading its slots of each bank from the one checkpoint.
//! A step every rank announced fires every rank's prompt at once; a step
//! only one rank brings fires the others idle, serving the routes the
//! first sends them. Every answer is the one a single engine holding every
//! expert gives.

use engine::Engine;
use engine_cuda::DeviceBoot;
use poem::{Dtype, Platform, Request};

use crate::micro::{
    MICRO, answer, boot, checkpoint, close, contract_for, ordinals, prompt, ranks, request, step,
};

#[test]
fn an_expert_parallel_group_serves_through_its_engines() {
    let _alone = crate::alone();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let size = ranks();
    let ordinals = ordinals(size);
    let whole = poem_compiler::catalog::trace_of(MICRO, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let lowered =
        poem_compiler::expert::lower(whole.clone(), size, 0).expect("the micro's MoE lowers");
    let word = |len: usize| whole.facts.word(&Request::new(len as u32, false));
    let prompts: Vec<Vec<u32>> = (0..size).map(|r| prompt(37 + 15 * r, r + 1)).collect();

    // One engine holding every expert, on each rank's device: the group's
    // kernels are loaded and tuned on every device before its ranks spin.
    let mut want = Vec::new();
    for &ordinal in ordinals.iter().take(engine_cuda::device::ctx::count()) {
        let (_dir, path) = checkpoint(&whole, &format!("one-{ordinal}"));
        let boot = DeviceBoot {
            ordinal,
            ..DeviceBoot::default()
        };
        let mut one = engine_cuda::open(boot, contract_for).expect("the engine opens");
        one.load(request(&whole, &path)).expect("the micro loads");
        want = (prompts.iter())
            .map(|p| {
                one.open(0).expect("the seat opens");
                answer(&mut one, &step(0, word(p.len()), p))
            })
            .collect();
    }
    let steps: Vec<_> = (prompts.iter())
        .map(|p| step(0, word(p.len()), p))
        .collect();

    // A rank per boot, every rank loading at once: their loads meet at arming.
    let boots = ordinals.iter().map(|&ordinal| boot(ordinal)).collect();
    let mut ranks =
        engine_cuda::lockstep::open_lockstep(boots, contract_for).expect("the group opens");
    // Every rank reads its own slots of each bank from the one checkpoint.
    let (dir, path) = checkpoint(&whole, "group");
    std::thread::scope(|scope| {
        for rank in &mut ranks {
            let request = request(&lowered, &path);
            scope.spawn(move || rank.load(request).expect("the rank loads"));
        }
    });

    // Every rank announces its prompt, so the step waits for them all.
    for (rank, step) in ranks.iter_mut().zip(&steps) {
        rank.expect_fire(step);
    }
    let got: Vec<Vec<f32>> = std::thread::scope(|scope| {
        let fired: Vec<_> = (ranks.iter_mut().zip(&steps))
            .map(|(rank, step)| scope.spawn(move || answer(rank, step)))
            .collect();
        fired.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (r, (got, want)) in got.iter().zip(&want).enumerate() {
        close(got, want, &format!("rank {r}'s prompt beside the others'"));
    }

    // Only rank 0 brings a prompt: the others idle through its step.
    let got = answer(&mut ranks[0], &step(1, word(prompts[0].len()), &prompts[0]));
    close(&got, &want[0], "rank 0's prompt with the others idle");
    drop(dir);
}
