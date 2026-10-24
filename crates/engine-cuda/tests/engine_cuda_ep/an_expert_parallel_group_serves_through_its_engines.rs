#![cfg(feature = "cuda")]

//! An expert-parallel group as a worker drives it: `open_lockstep` hands
//! back one engine a rank, each loaded and submitted to on a thread of its
//! own, every rank reading its slots of each bank from the one checkpoint.
//! A step both ranks announced fires both their prompts at once; a
//! step only one rank brings fires the other idle, serving the routes the
//! first sends it. Every answer is the one a single engine holding every
//! expert gives.

use engine::Engine;
use engine_cuda::DeviceBoot;
use poem::{Dtype, Platform, Request};

use crate::micro::{MICRO, RANKS, answer, checkpoint, close, contract_for, prompt, request, step};

#[test]
fn an_expert_parallel_group_serves_through_its_engines() {
    let _alone = crate::alone();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let whole = poem_compiler::catalog::trace_of(MICRO, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let lowered =
        poem_compiler::expert::lower(whole.clone(), RANKS, 0).expect("the micro's MoE lowers");
    let word = |len: usize| whole.facts.word(&Request::new(len as u32, false));
    let (a, b) = (prompt(37, 1), prompt(52, 2));
    let boot = || DeviceBoot {
        ordinal: 0,
        ..DeviceBoot::default()
    };

    // One engine holding every expert.
    let (want_a, want_b) = {
        let (_dir, path) = checkpoint(&whole, "one");
        let mut one = engine_cuda::open(boot(), contract_for).expect("the engine opens");
        one.load(request(&whole, &path)).expect("the micro loads");
        let want_a = answer(&mut one, &step(0, word(a.len()), &a));
        let want_b = answer(&mut one, &step(1, word(b.len()), &b));
        (want_a, want_b)
    };

    // A rank per boot, every rank loading at once: their loads meet at arming.
    let mut ranks = engine_cuda::lockstep::open_lockstep(vec![boot(), boot()], contract_for)
        .expect("the group opens");
    // Every rank reads its own slots of each bank from the one checkpoint.
    let (dir, path) = checkpoint(&whole, "group");
    std::thread::scope(|scope| {
        for rank in &mut ranks {
            let request = request(&lowered, &path);
            scope.spawn(move || rank.load(request).expect("the rank loads"));
        }
    });

    // Both ranks announce their prompts, so the step waits for both.
    let (first, second) = (step(0, word(a.len()), &a), step(0, word(b.len()), &b));
    ranks[0].expect_fire(&first);
    ranks[1].expect_fire(&second);
    let [r0, r1] = &mut ranks[..] else {
        unreachable!()
    };
    let (got_a, got_b) = std::thread::scope(|scope| {
        let one = scope.spawn(|| answer(r0, &first));
        let two = scope.spawn(|| answer(r1, &second));
        (one.join().unwrap(), two.join().unwrap())
    });
    close(&got_a, &want_a, "rank 0's prompt beside rank 1's");
    close(&got_b, &want_b, "rank 1's prompt beside rank 0's");

    // Rank 1 brings nothing: it idles through rank 0's step.
    let got = answer(r0, &step(1, word(a.len()), &a));
    close(&got, &want_a, "rank 0's prompt with rank 1 idle");
    drop(dir);
}
