#![cfg(feature = "cuda")]

//! An expert-parallel group that rebalances before every step but the
//! first, over the last step's loads: the experts its random router favours
//! move between the ranks, their planes copied slot to slot and every
//! rank's table rewritten, and each rank still answers what one engine
//! holding every expert answers, with no slot past the experts and with a
//! replica slot a rank. The first step's boundary drops the loads arming
//! left instead, so no expert moves on rows the group never served.

use engine::Engine;
use engine_cuda::DeviceBoot;
use poem::{Dtype, Platform, Request};

use crate::micro::{MICRO, RANKS, answer, checkpoint, close, contract_for, prompt, request, step};

#[test]
fn an_expert_parallel_group_rebalances_its_experts() {
    let _alone = crate::alone();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    check(0);
    check(1);
}

/// A group of `replicas` extra slots a rank: replicas of the hottest
/// experts, split between the ranks that hold them.
fn check(replicas: u32) {
    let whole = poem_compiler::catalog::trace_of(MICRO, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let lowered = poem_compiler::expert::lower(whole.clone(), RANKS, replicas)
        .expect("the micro's MoE lowers");
    let word = |len: usize| whole.facts.word(&Request::new(len as u32, false));
    let prompts = [prompt(37, 1), prompt(52, 2)];

    let want: Vec<Vec<f32>> = {
        let (_dir, path) = checkpoint(&whole, &format!("rebalance-one-{replicas}"));
        let boot = DeviceBoot {
            ordinal: 0,
            ..DeviceBoot::default()
        };
        let mut one = engine_cuda::open(boot, contract_for).expect("the engine opens");
        one.load(request(&whole, &path)).expect("the micro loads");
        (prompts.iter().enumerate())
            .map(|(slot, p)| answer(&mut one, &step(slot as u32, word(p.len()), p)))
            .collect()
    };

    // A rank a device where there are enough, so the rebalance reaches
    // each rank's loads and table on the device that holds them.
    let apart = engine_cuda::device::count() >= RANKS as usize;
    let boot = |rank: i32| {
        let mut boot = DeviceBoot {
            ordinal: if apart { rank } else { 0 },
            ..DeviceBoot::default()
        };
        boot.knobs.expert_rebalance_steps = Some(1);
        boot.knobs.expert_load_window = Some(1);
        boot
    };
    let mut ranks = engine_cuda::lockstep::open_lockstep(vec![boot(0), boot(1)], contract_for)
        .expect("the group opens");
    let (dir, path) = checkpoint(&whole, &format!("rebalance-group-{replicas}"));
    std::thread::scope(|scope| {
        for rank in &mut ranks {
            let request = request(&lowered, &path);
            scope.spawn(move || rank.load(request).expect("the rank loads"));
        }
    });

    // Each round both ranks' prompts, one rank at a time: every step reads
    // the placement the last one's loads chose.
    for round in 0..4 {
        for (r, p) in prompts.iter().enumerate() {
            let rank = &mut ranks[r];
            rank.open(0).expect("the seat opens");
            let got = answer(rank, &step(0, word(p.len()), p));
            close(
                &got,
                &want[r],
                &format!("rank {r}'s prompt, round {round}, {replicas} replicas"),
            );
            if (round, r) == (0, 0) {
                assert_eq!(
                    rank.moved_experts(),
                    0,
                    "the first step rebalanced on arming's rows ({replicas} replicas)"
                );
            }
        }
    }
    let moved = ranks[0].moved_experts();
    assert!(
        moved > 0,
        "the router's skew moved no expert ({replicas} replicas)"
    );
    eprintln!("{moved} slot copies across the rounds, {replicas} replicas");
    drop(dir);
}
