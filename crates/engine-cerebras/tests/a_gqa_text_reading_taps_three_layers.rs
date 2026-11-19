//! klein's text reading in miniature: `flux2-mini-gqa`'s encoder is a three-layer Qwen3 with
//! grouped-query attention, tapped at every layer and the taps' projections summed into the
//! context. engine-cpu's text rows are within the bf16 tolerance of transformers'; the ignored
//! `the_gqa_miniatures_text_files` writes what a fabric run takes.

mod common;

use engine_cpu::Forward;
use engine_cpu::forward::{Output, StreamLane, StreamStep, Values};
use poem_ir::{Request, Trace};

const GQA: &str = "flux2-mini-gqa-bf16-kv-bf16";

/// transformers' rows of the prompt below, [16, 256].
const TEXT_GOLDEN: &[u8] = include_bytes!("flux_2_mini_gqa_text16.f32");

/// "a red fox sleeping" in the chat template, no thinking: 16 tokens of Qwen3's vocabulary.
const PROMPT: [u32; 16] = [
    151_644, 872, 198, 64, 2518, 38835, 21127, 151_645, 198, 151_644, 77091, 198, 151_667, 271,
    151_668, 271,
];

fn trace() -> Trace {
    poem_compiler::catalog::repository()
        .parse(GQA)
        .expect("the catalog holds flux2-mini-gqa")
        .trace(poem_ir::Platform::Cerebras)
}

fn forward(trace: &Trace) -> Forward {
    Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        common::plane(trace, n, c)
    })
    .expect("flux2-mini-gqa loads")
}

fn step(trace: &Trace, tokens: &[u32]) -> StreamStep<'static> {
    let word = trace
        .facts
        .word(&Request::new(tokens.len() as u32, false).switched("reading", "text"));
    StreamStep {
        lanes: vec![StreamLane {
            word,
            rows: tokens.len(),
            clip: None,
        }],
        inputs: Vec::new(),
        tokens: tokens.to_vec(),
    }
}

fn hidden(outs: Vec<Output>) -> Vec<f32> {
    match outs.into_iter().find(|o| o.name == "hidden") {
        Some(Output {
            values: Values::F32(v),
            ..
        }) => v,
        _ => panic!("no hidden rows"),
    }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The text rows of engine-cpu's forward against transformers'.
#[test]
fn the_gqa_miniatures_text_answers_transformers() {
    let t = trace();
    let got = hidden(
        forward(&t)
            .stream(&step(&t, &PROMPT))
            .expect("the text step runs"),
    );
    let want = f32s(TEXT_GOLDEN);
    assert_eq!(got.len(), want.len());
    let (rel, cos) = common::rel_cos(&got, &want);
    eprintln!("flux2-mini-gqa's text rows vs transformers: rel L2 {rel:.3e}, cosine {cos:.8}");
    assert!(rel <= 2e-2 && cos >= 0.999, "rel L2 {rel}, cosine {cos}");
}
