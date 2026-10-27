#![cfg(feature = "device")]

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::fire::Readout;
use engine_cuda::{Boot, Graphs, Lane, Recording, Seated, Shell};
use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Weight, ops};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const SM_SCALE: f32 = 0.125;
const KV_ROW: &str = "kv";
const PAGE: u32 = 16;
const LANES: u32 = 4;
const PROMPT: u32 = 31;

/// One attention layer.
struct Micro;

impl ForwardHybrid for Micro {
    fn caches(&self) -> HybridSpec {
        let mut spec = HybridSpec::new();
        let space = spec.kv_space(Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(space, KV_ROW, [plane, plane], HEAD_DIM);
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
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, SM_SCALE, None, None, None, false, None,
            None, false, false,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

/// A load whose every fire launches at its bucket and runs eagerly: no
/// fire is bodied, so none is carved for the bucket by being recorded.
fn shell(trace: poem::Trace, path: &Path) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(LANES, 64),
        patches: None,
        page_size: PAGE,
        context: 256,
        slots: LANES,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::pinned(Graphs::Off, Some(Recording::Shaped))),
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

fn fire(shell: &mut Shell, trace: &poem::Trace, lanes: &[(u32, &[u32], &[u32], u32)]) -> Vec<Vec<f32>> {
    let seated: Vec<Seated<'_>> = lanes
        .iter()
        .map(|&(slot, pages, tokens, held)| Seated {
            pages,
            held: Some(held),
            ..Seated::of(Lane {
                slot,
                word: trace.facts.word(&Request::new(tokens.len() as u32, false)),
                tokens,
            })
        })
        .collect();
    let want: Vec<Readout> = lanes.iter().map(|_| Readout::Last).collect();
    shell.fire_rows(&seated, &want).expect("the fire runs")
}

/// A prompt prefilled into pages 1 and 0 stages write descriptors for its
/// rows; a three-lane decode on page 2 then launches at its bucket of four.
/// Its fourth, padded row writes nothing: a decode over the prompt answers
/// what it answers in a load that never ran the three lanes.
#[test]
fn a_padded_row_writes_nothing() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("padded-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4409).write(dir.path());
    let prompt: Vec<u32> = (0..PROMPT).map(|t| (t * 7 + 11) % VOCAB).collect();
    let prompt_pages = [1u32, 0, 3];
    let other = [2u32];
    let step = [5u32];

    let mut fresh = shell(trace.clone(), &path);
    fire(&mut fresh, &trace, &[(0, &prompt_pages, &prompt, 0)]);
    let want = fire(&mut fresh, &trace, &[(0, &prompt_pages, &step, PROMPT)]);
    drop(fresh);

    let mut shell = shell(trace.clone(), &path);
    fire(&mut shell, &trace, &[(0, &prompt_pages, &prompt, 0)]);
    let (a, b, c) = ([17u32], [23u32], [29u32]);
    fire(
        &mut shell,
        &trace,
        &[(1, &other, &a, 0), (2, &other, &b, 1), (3, &other, &c, 2)],
    );
    let got = fire(&mut shell, &trace, &[(0, &prompt_pages, &step, PROMPT)]);
    let off = got[0]
        .iter()
        .zip(&want[0])
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        off <= 1e-5,
        "a padded row of the three-lane fire wrote into the prompt's pages ({off} away)"
    );
}
