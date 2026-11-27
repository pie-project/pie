#![cfg(feature = "device")]
//! An output written over every row of a fire (not gathered at the readout
//! rows) is read back at each lane's own token rows: in a fire mixing a
//! prompt between two decodes, each lane's last row is the row it answers
//! alone, not the row at its index among the fire's readouts.

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::fire::Readout;
use engine_cuda::{Boot, Graphs, Lane, Seated, Shell};
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Weight, ops, trace_hybrid};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const KV_ROW: &str = "kv";
const PAGE: u32 = 16;
const PROMPT: u32 = 40;

/// One attention layer, its logits over every row.
struct Micro;

impl ForwardHybrid for Micro {
    fn states(&self) -> StateRows {
        let mut spec = StateRows::new();
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(KV_ROW, Dtype::Bf16, [plane, plane], HEAD_DIM);
        spec
    }

    fn forward(&self, inputs: Input) {
        let w = |name: &str, out: u64, inner: u64| Weight::sym(name, [out, inner], Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        let table = w("embed", u64::from(VOCAB), WIDTH);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let q = ops::linear::matmul(&x, &w("q", plane, WIDTH));
        let k = ops::linear::matmul(&x, &w("k", plane, WIDTH));
        let v = ops::linear::matmul(&x, &w("v", plane, WIDTH));
        ops::attn::kv_append(
            &k,
            Some(&v),
            inputs.kv(KV_ROW),
            &inputs.write_page(KV_ROW),
            &inputs.write_offset(KV_ROW),
        );
        let a = ops::attn::attend(
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, 0.125, None, None, None, false, None,
            None, false, false,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

fn shell(trace: poem::Trace, path: &Path, graphs: Graphs) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    Shell::load(Boot {
        knobs: engine_cuda::Knobs {
            pin: Some((graphs, None)),
            ..engine_cuda::Knobs::default()
        },
        ..crate::common::boot(trace, &contract, path, Budget::new(4, 64), 256, 6)
    })
    .expect("the shell loads")
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn serve(graphs: Graphs) {
    let trace = trace_hybrid("row-readout-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 2209).write(dir.path());
    let mut shell = shell(trace.clone(), &path, graphs);
    let pages: Vec<Vec<u32>> = (0..6u32)
        .map(|slot| {
            let per = 256 / PAGE;
            (slot * per..(slot + 1) * per).collect()
        })
        .collect();
    let word = |rows: u32| trace.facts.word(&Request::new(rows, false));
    let prompt = |salt: u32| -> Vec<u32> {
        (0..PROMPT)
            .map(|t| (t * 7 + 11 + salt * 13) % VOCAB)
            .collect()
    };
    // `(slot, tokens, held)` a lane.
    let fire = |shell: &mut Shell, lanes: &[(u32, &[u32], u32)]| -> Vec<Vec<f32>> {
        let seated: Vec<Seated<'_>> = lanes
            .iter()
            .map(|&(slot, tokens, held)| Seated {
                pages: &pages[slot as usize],
                held: Some(held),
                ..Seated::of(Lane {
                    slot,
                    word: word(tokens.len() as u32),
                    tokens,
                })
            })
            .collect();
        let want: Vec<Readout> = lanes.iter().map(|_| Readout::Last).collect();
        shell.fire_rows(&seated, &want).expect("the fire serves")
    };
    // Decoding lanes in slots 0 and 2, their twins in 3 and 5; the prompt in
    // 1, its twin in 4.
    for slot in 0..6 {
        shell.open(slot).expect("a slot opens");
    }
    for (slot, salt) in [(0, 0), (3, 0), (2, 2), (5, 2)] {
        fire(&mut shell, &[(slot, &prompt(salt), 0)]);
    }
    let (one, fresh, other) = ([5u32], prompt(1)[..24].to_vec(), [9u32]);
    let mixed = fire(
        &mut shell,
        &[(0, &one, PROMPT), (1, &fresh, 0), (2, &other, PROMPT)],
    );
    let alone = [
        fire(&mut shell, &[(3, &one, PROMPT)]).swap_remove(0),
        fire(&mut shell, &[(4, &fresh, 0)]).swap_remove(0),
        fire(&mut shell, &[(5, &other, PROMPT)]).swap_remove(0),
    ];
    for (lane, (a, b)) in alone.iter().zip(&mixed).enumerate() {
        let apart = worst(a, b);
        eprintln!("{graphs:?} lane {lane}: worst {apart}");
        assert!(
            apart < 1e-3,
            "{graphs:?}: lane {lane}'s last row in the mixed fire is not its own ({apart})"
        );
    }
}

#[test]
fn a_mixed_fires_last_rows_are_each_lanes_own() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    serve(Graphs::Off);
    serve(Graphs::On);
}
