//! klein's text reading in miniature (GENERIC.md M2c): `flux2-mini-gqa`'s encoder is a Qwen3 of
//! three layers with grouped-query attention (4 q heads over 2 kv heads of 128), tapped at every
//! layer, each tap projected into the context by its column block of the context embedder and
//! the projections summed, as klein's three taps are. Its weights are drawn by name as the other
//! miniature's are.
//!
//! - engine-cpu's text rows (the trace's own forward) are within the bf16 tolerance of
//!   transformers' `Qwen3ForCausalLM` with the same weights, its three hidden states
//!   concatenated through the context embedder (`mk_tiny_flux2_pipe.py --gqa --text`).
//! - `the_gqa_miniatures_text_files` (ignored) writes what a fabric run takes: the step's every
//!   node's values (`gold/`), the text reading compiled with them (`prog/`: the stream compiler's
//!   grouped-query attention and the row stream carrying the earlier taps' projections), and
//!   engine-cpu's rows (`cpu_text.f32`), for `tools/wafer/stream/sim.sh`.

mod common;

use std::path::Path;

use engine_cpu::Forward;
use engine_cpu::forward::{Output, StreamLane, StreamStep, Values};
use poem_ir::{Request, Trace};

const GQA: &str = "flux2-mini-gqa-bf16-kv-bf16";

/// transformers' rows of the prompt below (`mk_tiny_flux2_pipe.py --gqa --text`), [16, 256].
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

/// Under `READING_OUT`: the step's capture (`gold/`), the text reading compiled from it at 16
/// rows (`prog/`), engine-cpu's rows (`cpu_text.f32`).
#[test]
#[ignore = "writes a fabric run's files; run with READING_OUT"]
fn the_gqa_miniatures_text_files() {
    let out = std::path::PathBuf::from(std::env::var("READING_OUT").expect("set READING_OUT"));
    let t = trace();
    let mut f = forward(&t);
    let s = step(&t, &PROMPT);
    let gold = out.join("gold");
    common::capture(&mut f, &t, &s, &gold);
    let o = kernels_cerebras::stream::Options {
        reading: "text".into(),
        rows: PROMPT.len() as u32,
        round16: true,
        upto: None,
        taps: Vec::new(),
    };
    kernels_cerebras::stream::compile(&t, &o, &out.join("prog"), Some(&gold))
        .expect("the text reading compiles");
    let rows = hidden(f.stream(&s).expect("the text step runs"));
    write(&out.join("cpu_text.f32"), &rows);
}

fn write(p: &Path, v: &[f32]) {
    std::fs::write(
        p,
        v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .expect("a file");
}
