//! **THE QWEN4 PARITY GATE**: `Model::flash_micro` against
//! `modular_qwen4_exp.py`, over one set of weights.
//!
//! Every layer of the new family's math runs here — the gated residual's
//! mix and inject, the grouped plus-one norms, the sigmoid-gated delta net,
//! the n-gram hasher with an eos mid-stream, the concatenating gather, the
//! PLE gate and its dilated convolution — and the reference implementation
//! computed the same numbers first (`tests/qwen4-parity/make_reference.py`,
//! transformers' own `qwen4_exp` at bf16, seeded). The fixture's indexer
//! budget exceeds every sequence below, so QSA selection is the full causal
//! mask and the comparison owes nothing to the cut this text documents.
//!
//! Three claims per prompt:
//! 1. **the last row agrees** — the prefill's logits, elementwise, inside a
//!    bf16-accumulation tolerance;
//! 2. **every decode row agrees** — sixteen steps teacher-forced along the
//!    reference's own greedy path, each row inside the same tolerance. Rows
//!    and not argmaxes, deliberately: this fixture is a seeded random model
//!    over a 256-token vocabulary, its near-ties sit inside bf16
//!    accumulation noise, and one flipped tie cascades a token comparison
//!    into a comparison of two attractor orbits;
//! 3. **twice is once** — the same fire twice produces identical bits.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --test qwen4_micro_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use engine_cuda::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Dtype, Platform, Request};

/// How many greedy steps follow the prefill — the reference's own count.
const STEPS: usize = 16;

/// The elementwise ceiling on any row's disagreement. Observed, then pinned:
/// bf16 accumulation order differs between torch's eager kernels and this
/// shell's, the worst row over three prompts and sixteen positions each
/// measured 0.056, and the ceiling stands off it far enough to hold and
/// near enough to catch a broken op (an early mis-hash measured 0.36).
const LOGIT_TOLERANCE: f32 = 0.15;

fn fixture() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_QWEN4_FIXTURE") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/qwen4-parity/fixture");
    path.is_dir().then_some(path)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn word(query_len: u32) -> u64 {
    models::qwen_4::forward::Facts::of(&Request::new(query_len, false)).word()
}

/// One prompt's run: prefill, then `STEPS` decodes teacher-forced along the
/// reference's greedy `path`, in slot 0. Answers the prefill row and every
/// decode row.
fn run(shell: &mut Shell, prompt: &[u32], path: &[u32]) -> (Vec<f32>, Vec<Vec<f32>>) {
    shell.open(0).expect("slot 0 opens");
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    assert_eq!(prefill.len(), 1, "one lane in, one row of logits out");
    let last = prefill[0].clone();

    let mut rows = Vec::with_capacity(STEPS - 1);
    for (step, fed) in path[..STEPS - 1].iter().enumerate() {
        let decode = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &[*fed],
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        rows.push(decode[0].clone());
    }
    (last, rows)
}

fn ready() -> Option<(Shell, serde_json::Value)> {
    if !engine_cuda::device::present() {
        eprintln!("skipping the qwen4 parity gate: no CUDA device on this machine");
        return None;
    }
    let Some(fixture) = fixture() else {
        eprintln!(
            "skipping the qwen4 parity gate: no fixture (run \
             tests/qwen4-parity/make_reference.py, or set PIE_QWEN4_FIXTURE)"
        );
        return None;
    };

    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture.join("reference.json"))
            .expect("the fixture ships its reference"),
    )
    .expect("the reference parses");

    let micro = models::qwen_4::model::Model::flash_micro(Dtype::Bf16, Dtype::Bf16, 1);
    let trace = model_dsl::trace_hybrid("qwen4-micro", &micro, Platform::Cuda);
    let source = ztensor_compat::index(&fixture.join("model.safetensors"))
        .expect("the fixture's checkpoint opens");
    let contract = micro
        .import(&source)
        .expect("the micro text's import fits the fixture it was generated for");
    drop(source);

    let shell = Shell::load(Boot {
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &fixture,
        budget: Budget::new(4, 64),
        patches: None,
        profile: None,
        page_size: 16,
        context: 128,
        slots: 2,
        ordinal: 0,
        graphs: engine_cuda::Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
    })
    .expect("the micro shell loads");
    Some((shell, reference))
}

#[test]
#[ignore = "real-hardware: needs a CUDA device; run with `-- --ignored`"]
fn the_micro_text_computes_the_references_own_logits() {
    let Some((mut shell, reference)) = ready() else {
        return;
    };

    let mut faults = Vec::new();
    for (name, case) in reference.as_object().expect("prompts") {
        let tokens: Vec<u32> = case["tokens"]
            .as_array()
            .expect("tokens")
            .iter()
            .map(|v| v.as_u64().expect("a token id") as u32)
            .collect();
        let expect_logits: Vec<f32> = case["last_logits"]
            .as_array()
            .expect("logits")
            .iter()
            .map(|v| v.as_f64().expect("a logit") as f32)
            .collect();
        let expect_greedy: Vec<u32> = case["greedy"]
            .as_array()
            .expect("greedy")
            .iter()
            .map(|v| v.as_u64().expect("a token id") as u32)
            .collect();
        let expect_rows: Vec<Vec<f32>> = case["step_logits"]
            .as_array()
            .expect("step logits")
            .iter()
            .map(|row| {
                row.as_array()
                    .expect("a row")
                    .iter()
                    .map(|v| v.as_f64().expect("a logit") as f32)
                    .collect()
            })
            .collect();

        let (last, rows) = run(&mut shell, &tokens, &expect_greedy);
        let (again, rows_again) = run(&mut shell, &tokens, &expect_greedy);
        assert_eq!(
            last.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "`{name}`: the same prefill answered different bits"
        );
        assert_eq!(rows, rows_again, "`{name}`: twice is not once");

        let spread = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let mut worst = spread(&last, &expect_logits);
        eprintln!("{name}: prefill row spread {:.4}", spread(&last, &expect_logits));
        // The reference's step loop records its own prefill as row zero
        // (its first iteration feeds the whole prompt), so the decode rows
        // start one over.
        for (step, (row, expect)) in rows.iter().zip(&expect_rows[1..]).enumerate() {
            let s = spread(row, expect);
            eprintln!("{name}: step {step} spread {s:.4}");
            worst = worst.max(s);
        }
        // And every position INSIDE the prompt, through prefix prefills —
        // the bisection that found the off-by-one, kept as coverage: a row
        // per prompt position, each against the reference's own.
        let all_rows: Vec<Vec<f32>> = case["all_logits"]
            .as_array()
            .expect("all logits")
            .iter()
            .map(|row| {
                row.as_array()
                    .expect("a row")
                    .iter()
                    .map(|v| v.as_f64().expect("a logit") as f32)
                    .collect()
            })
            .collect();
        for len in 1..=tokens.len() {
            shell.open(1).expect("slot 1 opens");
            let re = shell
                .fire(&[Lane {
                    slot: 1,
                    word: word(len as u32),
                    tokens: &tokens[..len],
                }])
                .expect("the diagnostic prefill fires");
            worst = worst.max(spread(&re[0], &all_rows[len - 1]));
        }
        eprintln!("{name}: worst row disagreement {worst:.4} over the walk");
        if worst > LOGIT_TOLERANCE {
            faults.push(format!(
                "`{name}`: a teacher-forced row disagrees with the reference by {worst}"
            ));
        }
    }
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
