#![cfg(feature = "cuda")]

//! Under `fire-timing` each rank of an expert-parallel group times every
//! fire it serves on its device, its own steps and its idle ones, by the
//! step's token bucket; arming's fires are not counted.

use engine::Engine;
use engine_cuda::DeviceBoot;
use poem::{Dtype, Platform, Request};

use crate::micro::{MICRO, RANKS, answer, checkpoint, contract_for, prompt, request, step};

#[test]
fn an_expert_parallel_rank_times_its_fires_on_the_device() {
    let _alone = crate::alone();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let whole = poem_compiler::catalog::trace_of(MICRO, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let lowered =
        poem_compiler::expert::lower(whole.clone(), RANKS, 0).expect("the micro's MoE lowers");
    let word = |len: usize| whole.facts.word(&Request::new(len as u32, false));

    let boot = || {
        let mut boot = DeviceBoot {
            ordinal: 0,
            ..DeviceBoot::default()
        };
        boot.knobs.diagnostics.fire_timing = true;
        boot
    };
    let mut ranks = engine_cuda::lockstep::open_lockstep(vec![boot(), boot()], contract_for)
        .expect("the group opens");
    let (dir, path) = checkpoint(&whole, "fire-timing");
    std::thread::scope(|scope| {
        for rank in &mut ranks {
            let request = request(&lowered, &path);
            scope.spawn(move || rank.load(request).expect("the rank loads"));
        }
    });
    for rank in &ranks {
        let times = rank.fire_times().expect("the times read");
        assert_eq!(
            times.map(|t| t.len()),
            Some(0),
            "arming's fires are not served"
        );
    }

    // Buckets 64, 64 and 16: one rank's frame each, the other idle.
    let prompts = [prompt(37, 1), prompt(52, 2), prompt(9, 3)];
    for (i, p) in prompts.iter().enumerate() {
        let rank = &mut ranks[i % 2];
        rank.open(0).expect("the seat opens");
        answer(rank, &step(0, word(p.len()), p));
    }
    for (r, rank) in ranks.iter().enumerate() {
        let times = (rank.fire_times())
            .expect("the times read")
            .expect("fire-timing is on");
        let fires: Vec<(u32, u64)> = times.iter().map(|(&b, t)| (b, t.fires)).collect();
        assert_eq!(fires, [(16, 1), (64, 2)], "rank {r}'s fires by bucket");
        for (bucket, time) in &times {
            assert!(
                time.mean_us() > 0.0 && time.max_us >= time.mean_us(),
                "rank {r}, bucket {bucket}: {time:?}"
            );
        }
        eprintln!("rank {r}: {times:?}");
    }
    drop(dir);
}
