#![cfg(feature = "device")]

//! `inputs.next_tokens()`, the table a draft head's ids are read from: row
//! i of a lane is fed token i + 1 of the same lane, as vLLM's speculator
//! shifts its ids, and a lane's last row its own (the forward puts the
//! sampled token there). Two lanes in one fire, each row read out in turn:
//! each row's embedding is the next token's row of the table, never across the
//! lane boundary, eagerly and recorded.

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::fire::Readout;
use engine_cuda::{Boot, Graphs, Lane, Recording, Seated, Shell};
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Weight, ops};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const LANES: u32 = 2;

struct Shifted;

impl ForwardHybrid for Shifted {
    fn states(&self) -> StateRows {
        StateRows::new()
    }

    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(VOCAB), WIDTH], Dtype::Bf16);
        let fed = ops::gather::embed(&inputs.next_tokens(), &table, VOCAB);
        let fed = ops::gather::gather_rows(&fed, &inputs.readout_rows());
        poem::output("logits", &fed, None);
    }
}

fn shell(trace: poem::Trace, path: &Path, graphs: Graphs) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(LANES, 64),
        patches: None,
        page_size: 16,
        context: 256,
        slots: LANES,
        kv_budget: Some(crate::POOL),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::pinned(graphs, Some(Recording::Shaped))),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    for slot in 0..LANES {
        shell.open(slot).expect("the slot opens");
    }
    shell
}

/// Row i's token i + 1 within its lane, its own at the lane's last row.
fn shifted(tokens: &[u32]) -> Vec<u32> {
    tokens.iter().skip(1).chain(tokens.last()).copied().collect()
}

fn fire_and_compare(graphs: Graphs) {
    let trace = poem::trace_hybrid("next-token-micro", &Shifted, Platform::Cuda);
    assert!(trace.reads_next_tokens());
    let dir = tempfile::tempdir().expect("a scratch directory");
    let weights = Weights::random(&trace, 7);
    let path = weights.write(dir.path());
    let table = weights.get("embed");
    let lanes: [Vec<u32>; 2] = [vec![3, 41, 7, 0, 63, 2, 19], vec![1, 54, 8, 5]];
    let mut shell = shell(trace.clone(), &path, graphs);
    let w = WIDTH as usize;
    // A plain lane reads one row out a fire: each row in turn, both lanes
    // in every fire (they hold no state, so each fire is the same).
    let longest = lanes.iter().map(Vec::len).max().unwrap_or(0);
    for row in 0..longest {
        let seated: Vec<Seated<'_>> = lanes
            .iter()
            .enumerate()
            .map(|(slot, tokens)| Seated {
                kv_less: true,
                ..Seated::of(Lane {
                    slot: slot as u32,
                    word: trace.facts.word(&Request::new(tokens.len() as u32, false)),
                    tokens,
                })
            })
            .collect();
        let want: Vec<Readout> = lanes
            .iter()
            .map(|tokens| Readout::Rows(vec![row.min(tokens.len() - 1) as u32]))
            .collect();
        let readouts = shell.fire_rows(&seated, &want).expect("the fire runs");
        for (lane, (tokens, got)) in lanes.iter().zip(&readouts).enumerate() {
            let at = row.min(tokens.len() - 1);
            let next = shifted(tokens)[at];
            assert_eq!(got.len(), w, "lane {lane}: the one row read out");
            assert_eq!(
                got.as_slice(),
                &table[next as usize * w..][..w],
                "{graphs:?} lane {lane} row {at}: fed token {next}, the lane's next"
            );
        }
    }
}

#[test]
fn a_row_takes_its_lanes_next_token() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    fire_and_compare(Graphs::Off);
    fire_and_compare(Graphs::Shaped);
}
