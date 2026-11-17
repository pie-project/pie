#![cfg(feature = "device")]

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine_cuda::{Boot, Graphs, Shell};
use poem::{Dtype, ForwardHybrid, Input, Platform, StateRows, Trace, Weight, ops, trace_hybrid};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const LANES: u32 = 4;
const KV_ROW: &str = "kv";

/// One attention layer, its scores captured or not.
struct Micro {
    scores: bool,
}

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
        let pages = inputs.kv(KV_ROW);
        ops::attn::kv_append(
            &k,
            Some(&v),
            pages,
            &inputs.write_page(KV_ROW),
            &inputs.write_offset(KV_ROW),
        );
        let a = ops::attn::attend(
            &q,
            &inputs,
            KV_ROW,
            HEADS,
            HEADS,
            HEAD_DIM,
            0.125,
            None,
            None,
            None,
            false,
            None,
            None,
            false,
            self.scores,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

/// The activation the load backs, as its ledger enters it.
fn activation(trace: Trace, path: &Path) -> u64 {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let shell = Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(LANES, 64),
        patches: None,
        page_size: 16,
        context: 256,
        slots: 1,
        kv_budget: Some(crate::POOL),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::pinned(Graphs::Off, None)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    shell.footprint().1
}

// A load whose fires capture attention scores backs their planes at load,
// in the ledger's activation: a program that reads them binds with nothing
// left to back.
#[test]
fn a_score_capturing_load_backs_its_score_planes_at_load() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let plain = trace_hybrid("score-micro", &Micro { scores: false }, Platform::Cuda);
    let scored = trace_hybrid("score-micro", &Micro { scores: true }, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&plain, 1103).write(dir.path());

    let plain = activation(plain, &path);
    let scored = activation(scored, &path);
    // A plane a layer, a row a lane: `[lanes, heads * 2048]` floats.
    let planes = u64::from(LANES) * u64::from(HEADS) * 2048 * 4;
    assert!(
        scored >= plain + planes,
        "the capturing load backs {scored} bytes of activation at load and the plain one \
         {plain}: the {planes} bytes of score planes are not backed at load"
    );
}
