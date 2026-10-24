#![cfg(feature = "cuda")]

//! An expert-parallel group with no rebalance still counts the rows each
//! rank's slots receive: read between two steps, each MoE layer's balance
//! holds every row the served steps routed (each token's `top_k`), split
//! by rank and by logical expert, and none of the rows its arming fired.
//! A read closes its window: the next, with no step between, is empty.

use engine::Engine;
use engine_cuda::DeviceBoot;
use poem::{Dtype, Platform, Request};
use poem_ir::{Collective, Operation};

use crate::micro::{MICRO, RANKS, answer, checkpoint, contract_for, prompt, request, step};

#[test]
fn an_expert_parallel_group_reads_its_expert_balance() {
    let _alone = crate::alone();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let whole = poem_compiler::catalog::trace_of(MICRO, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let lowered =
        poem_compiler::expert::lower(whole.clone(), RANKS, 0).expect("the micro's MoE lowers");
    let word = |len: usize| whole.facts.word(&Request::new(len as u32, false));
    let mut layers = std::collections::BTreeMap::new();
    for node in &lowered.nodes {
        if let Operation::Collective(Collective::Dispatch {
            layer,
            top_k,
            experts,
            ..
        }) = &node.op
        {
            layers.insert(*layer, (*top_k, *experts));
        }
    }

    let boot = || DeviceBoot {
        ordinal: 0,
        ..DeviceBoot::default()
    };
    let mut ranks = engine_cuda::lockstep::open_lockstep(vec![boot(), boot()], contract_for)
        .expect("the group opens");
    let (dir, path) = checkpoint(&whole, "balance");
    std::thread::scope(|scope| {
        for rank in &mut ranks {
            let request = request(&lowered, &path);
            scope.spawn(move || rank.load(request).expect("the rank loads"));
        }
    });
    assert!(
        ranks[0].expert_balance().is_empty(),
        "nothing is read before a step"
    );

    let prompts = [prompt(37, 1), prompt(52, 2), prompt(9, 3)];
    for (i, p) in prompts.iter().enumerate() {
        let rank = &mut ranks[i % 2];
        rank.open(0).expect("the seat opens");
        answer(rank, &step(0, word(p.len()), p));
    }
    let tokens: u64 = prompts.iter().map(|p| p.len() as u64).sum();

    let balance = ranks[1]
        .read_expert_balance()
        .expect("the loads read between steps");
    assert_eq!(
        balance.iter().map(|b| b.layer).collect::<Vec<_>>(),
        layers.keys().copied().collect::<Vec<_>>(),
        "a balance a dispatched layer"
    );
    for b in &balance {
        let what = format!("layer {}", b.layer);
        assert_eq!(b.steps, prompts.len() as u64, "{what}: the steps served");
        assert_eq!(
            b.rows(),
            tokens * u64::from(layers[&b.layer].0),
            "{what}: every served token's routes, and no arming row"
        );
        assert_eq!(b.ranks.len(), RANKS as usize, "{what}");
        assert_eq!(b.experts.len(), layers[&b.layer].1 as usize, "{what}");
        assert_eq!(b.experts.iter().sum::<u64>(), b.rows(), "{what}");
        assert!(
            (1.0..=f64::from(RANKS)).contains(&b.rank_skew()),
            "{what}: {}",
            b.rank_skew()
        );
        eprintln!("{b}");
    }
    assert_eq!(
        ranks[0].expert_balance(),
        balance,
        "every rank sees the group's last read"
    );

    let again = ranks[0].read_expert_balance().expect("a second read");
    assert!(
        again.iter().all(|b| b.steps == 0 && b.rows() == 0),
        "a read closes its window: {again:?}"
    );
    drop(dir);
}
