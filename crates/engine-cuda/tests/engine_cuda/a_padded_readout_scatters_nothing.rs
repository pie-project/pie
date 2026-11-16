#![cfg(feature = "device")]

//! A draft head scatters the trunk's samples over its lane's next tokens at
//! the readout rows (`gather.scatter_rows`, Qwen3.5's MTP head). A fire
//! pads its readout rows past the live ones with row 0 (to a body's
//! ceiling, or an eager launch's extent), and a pad is no sample of the
//! fire's: row 0 keeps its next token. Qwen3.5-9B's head read a pad's
//! stale sample at row 0 and, once that was non-finite, drafted nothing.
//!
//! Each row's sample here is the table row its own index names, so a pad
//! would land table row 0 at row 0; the readout row reads row 0 back
//! through the lane's tokens (token 5 is 0).

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::fire::Readout;
use engine_cuda::{Boot, Graphs, Lane, Recording, Seated, Shell};
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Weight, ops};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
/// Two slots, one fired: a body's readout ceiling is a row a lane, so the
/// fire's one readout row is padded with a second.
const SLOTS: u32 = 2;

struct Placed;

impl ForwardHybrid for Placed {
    fn states(&self) -> StateRows {
        StateRows::new()
    }

    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(VOCAB), WIDTH], Dtype::Bf16);
        let nexts = ops::gather::embed(&inputs.next_tokens(), &table, VOCAB);
        let sampled = ops::gather::embed(&inputs.readout_rows(), &table, VOCAB);
        let placed = ops::gather::scatter_rows(&sampled, &inputs.readout_rows(), &nexts);
        let by_token = ops::gather::gather_rows(&placed, &inputs.tokens());
        let read = ops::gather::gather_rows(&by_token, &inputs.readout_rows());
        poem::output("logits", &read, None);
    }
}

fn shell(trace: poem::Trace, path: &Path, graphs: Graphs, recording: Recording) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(SLOTS, 64),
        patches: None,
        page_size: 16,
        context: 256,
        slots: SLOTS,
        kv_budget: Some(crate::POOL),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::pinned(graphs, Some(recording))),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    for slot in 0..SLOTS {
        shell.open(slot).expect("the slot opens");
    }
    shell
}

fn fire_and_compare(graphs: Graphs, recording: Recording) {
    let trace = poem::trace_hybrid("padded-readout-micro", &Placed, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let weights = Weights::random(&trace, 11);
    let path = weights.write(dir.path());
    let table = weights.get("embed");
    let w = WIDTH as usize;
    // Row 5 is read out; its token, 0, reads row 0 back, whose next token
    // is the lane's second, 1.
    let tokens: Vec<u32> = vec![3, 1, 4, 2, 6, 0, 5];
    let read = [5u32];
    let mut shell = shell(trace.clone(), &path, graphs, recording);
    // Twice: a recorded body replays on the second.
    for fire in 0..2 {
        let seated = [Seated {
            kv_less: true,
            readout: Some(read.as_slice()),
            ..Seated::of(Lane {
                slot: 0,
                word: trace.facts.word(&Request::new(tokens.len() as u32, false)),
                tokens: &tokens,
            })
        }];
        let got = shell
            .fire_rows(&seated, &[Readout::Rows(vec![5])])
            .expect("the fire runs");
        let got = &got[0];
        assert_eq!(got.len(), w, "the one row read out");
        let fed = (0..VOCAB as usize).find(|&id| &table[id * w..][..w] == got.as_slice());
        assert_eq!(
            fed,
            Some(1),
            "{graphs:?} {recording:?} fire {fire}: row 0 holds its next token, 1; a padded \
             readout's sample is table row 0"
        );
    }
}

#[test]
fn a_padded_readout_scatters_nothing() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    fire_and_compare(Graphs::Shaped, Recording::Shaped);
    fire_and_compare(Graphs::On, Recording::Bodies { mem_megabytes: 64 });
}
