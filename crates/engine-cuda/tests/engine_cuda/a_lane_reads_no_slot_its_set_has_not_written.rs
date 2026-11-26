#![cfg(feature = "device")]
//! A page a working set takes holds what its last owner left until the set
//! writes it. A lane whose rows would read a slot its set never wrote, on a
//! page another set wrote before, is refused by name, and nothing of its
//! fire runs; a lane reads what its set wrote, as far as the runtime knows
//! it or as the fires the engine admitted since carried it.

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

fn serve(graphs: Graphs) {
    let trace = trace_hybrid("extent-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 2209).write(dir.path());
    let mut shell = shell(trace.clone(), &path, graphs);
    // One run of pages, which set 1 writes first and set 2 takes after.
    let pages: Vec<u32> = (0..256 / PAGE).collect();
    let word = |rows: u32| trace.facts.word(&Request::new(rows, false));
    let prompt: Vec<u32> = (0..PROMPT).map(|t| (t * 7 + 11) % VOCAB).collect();
    // `(slot, set, readable, tokens, held)`.
    let fire =
        |shell: &mut Shell, slot: u32, set: u64, readable: u32, tokens: &[u32], held: u32| {
            let seated = [Seated {
                pages: &pages,
                held: Some(held),
                set,
                readable,
                ..Seated::of(Lane {
                    slot,
                    word: word(tokens.len() as u32),
                    tokens,
                })
            }];
            shell.fire_rows(&seated, &[Readout::Last])
        };
    for slot in 0..2 {
        shell.open(slot).expect("a slot opens");
    }
    fire(&mut shell, 0, 1, 0, &prompt, 0).expect("set 1 writes its prompt");
    // Set 2 takes the pages set 1 wrote: rows 0..PROMPT hold set 1's keys.
    let refused = fire(&mut shell, 1, 2, 0, &[5], PROMPT)
        .expect_err("a lane reading slots its set never wrote");
    let why = refused.to_string();
    assert!(why.contains("never written"), "{graphs:?}: {why}");
    // What its runtime states written (a prefix it shares) it reads.
    fire(&mut shell, 1, 2, PROMPT, &[5], PROMPT).expect("a shared prefix is read");
    // A fire staged behind one still in flight: the runtime still states
    // what it knew, and the engine carries what it admitted since.
    fire(&mut shell, 1, 2, PROMPT, &[6], PROMPT + 1).expect("the next row, in flight");
    fire(&mut shell, 1, 2, PROMPT, &[7], PROMPT + 2).expect("and the next");
    let gap = fire(&mut shell, 1, 2, PROMPT, &[8], PROMPT + 4)
        .expect_err("a row past a gap")
        .to_string();
    assert!(gap.contains("never written"), "{graphs:?}: {gap}");
    // The seat taken again by another set starts from what its runtime states.
    let other = fire(&mut shell, 1, 3, 0, &[5], PROMPT + 3)
        .expect_err("another set on the seat")
        .to_string();
    assert!(other.contains("never written"), "{graphs:?}: {other}");
    fire(&mut shell, 1, 3, 0, &prompt[..8], 0).expect("it writes from its start");
}

#[test]
fn a_lane_reads_no_slot_its_set_has_not_written() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    serve(Graphs::Off);
    serve(Graphs::On);
}
