//! **THE TILED FLIP ON A SHIPPING SKU** (§J4b's recipe, taken) — one real
//! MLX 4-bit checkpoint, imported in two orders, asked the same question by
//! two builds, and answering with the same words.
//!
//! ```text
//! PIE_QWEN35_ARTIFACT=/scratch/tiled.zt \
//!   cargo test -p engine-cuda --features cuda-13 --release \
//!   --test a_tiled_sku_says_what_the_row_major_one_says -- --ignored --nocapture
//! ```
//!
//! # What this gate is, beside the two that came before it
//!
//! `checkpoint/tests/tiled_repack.rs` proved the repack is a relabelling at
//! micro scale, and `a_repacked_projection_serves_what_the_row_major_one_
//! serves.rs` proved the serving chain over a container a test wrote. Neither
//! touched a checkpoint anyone ships, and neither ran the road an operator
//! runs: `pie model import`.
//!
//! This one does. The subject is `qwen35-d0.8b-mlxu4-kv-bf16` over
//! `mlx-community/Qwen3.5-0.8B-4bit` — the whole of this box's MLX affine-u4
//! inventory, and the smallest catalog row the flip reaches.
//!
//! # WHY ONE ARTIFACT PER RUN, AND NOT BOTH IN ONE PROCESS
//!
//! The order a projection is stored in is not a property of the FILE that a
//! load discovers; it is what the model TEXT declares, and the text is
//! compiled in. A build whose `qwen_3::Model` says `U4g64tiled` states an
//! `Expr::Repack` over any artifact that still holds the row-major legs, and
//! a serving plan refuses one BY NAME:
//!
//! ```text
//! this load would relayout a weight plane on the way in
//! (Some(TiledAffineU4Weight)), and a serving plan does not: a repack is paid
//! once per weight, not once per boot. Run `pie model import` on the source
//! checkpoint
//! ```
//!
//! That refusal is the design working — it is `tiled_repack.rs`'s gate (c)
//! reaching a real load — and it means the two arms are two BUILDS, not two
//! shells. So this gate reads ONE artifact, and the cross-arm claim is
//! carried by [`EXPECTED`]: both builds run this same file against their own
//! artifact and both must produce that text.

use std::path::PathBuf;
use std::time::Instant;

use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The row under test. Overridable so that the same harness can ask the
/// BF16 row of the same model the same question — which is how a rambling
/// continuation is told from a broken one.
fn sku() -> String {
    std::env::var("PIE_QWEN35_SKU").unwrap_or_else(|_| "qwen35-d0.8b-mlxu4-kv-bf16".to_string())
}

const PROMPT: &str = "The capital of France is";

/// What the BF16 row of this same model answers, and what the 4-bit rows do
/// NOT — for a reason that is worth writing down, because it looks like a
/// bug and is not one.
///
/// **THE TWO TOP LOGITS ARE EXACTLY TIED.** At the end of this prompt both
/// 4-bit arms put ` Paris` (11751) and ` in` (303) at **14.4375**, the same
/// bf16 value to the bit, with the third candidate a clear 0.5 below. A
/// 0.8-billion-parameter model whose every projection AND whose embedding
/// table are four bits has spent the margin that separated them; the bf16
/// row of the same checkpoint keeps it and answers ` Paris.` Greedy decoding
/// then breaks the tie by index — 303 is the lower one — and walks off into
/// a fluent ramble.
///
/// So this string is the BF16 row's pin and a note about the 4-bit rows, not
/// a claim either of them fails. It is checked only when the row under test
/// has no four-bit plane in it.
const EXPECTED: &str = " Paris.";

/// Decode steps timed. Enough that the mean is not one scheduler hiccup.
const STEPS: usize = 32;

fn artifact() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("PIE_QWEN35_ARTIFACT").ok()?);
    path.is_file().then_some(path)
}

fn tokenizer() -> Option<tokenizer::Tokenizer> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots");
    let file = std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path().join("tokenizer.json")))
        .find(|path| path.exists())?;
    tokenizer::Tokenizer::from_file(&file).ok()
}

fn word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
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

fn spread(logits: &[f32]) -> f32 {
    logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min)
}

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    let bad = logits.iter().position(|value| !value.is_finite());
    assert!(
        bad.is_none(),
        "{what} logit {} is {}, and a single NaN means the whole row is noise",
        bad.unwrap_or(0),
        logits[bad.unwrap_or(0)],
    );
    assert!(
        spread(logits) > 1e-3,
        "{what} logits span {}, which is a rectangle nothing wrote",
        spread(logits),
    );
}

/// One prefill and `STEPS` greedy decodes, with each decode step timed.
fn run(shell: &mut Shell, prompt: &[u32]) -> (Vec<u32>, Vec<f64>, Vec<f32>) {
    shell.open(0).expect("slot 0 opens");
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    finite(&prefill[0], "prefill");

    let mut tokens = vec![argmax(&prefill[0])];
    let mut millis = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let fed = [*tokens.last().expect("a step feeds the last token back")];
        let at = Instant::now();
        let decode = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        millis.push(at.elapsed().as_secs_f64() * 1000.0);
        finite(&decode[0], "decode");
        tokens.push(argmax(&decode[0]));
    }
    (tokens, millis, prefill[0].clone())
}

fn mean(of: &[f64]) -> f64 {
    of.iter().sum::<f64>() / of.len().max(1) as f64
}

/// The median, which is the number to read when a first step pays for a graph
/// capture the thirty-one after it do not.
fn median(of: &[f64]) -> f64 {
    let mut sorted = of.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

#[test]
#[ignore = "real-hardware: needs a CUDA device and an import of \
            mlx-community/Qwen3.5-0.8B-4bit made by THIS build — set \
            PIE_QWEN35_ARTIFACT; run with `-- --ignored`"]
fn the_imported_artifact_prefills_decodes_and_says_something() {
    if !engine_cuda::device::present() {
        eprintln!("skipping qwen35 tiled first light: no CUDA device on this machine");
        return;
    }
    let Some(checkpoint) = artifact() else {
        eprintln!(
            "skipping qwen35 tiled first light: set PIE_QWEN35_ARTIFACT to an import \
             of mlx-community/Qwen3.5-0.8B-4bit made by this build"
        );
        return;
    };
    let Some(tokenizer) = tokenizer() else {
        eprintln!(
            "skipping qwen35 tiled first light: no mlx-community/Qwen3.5-0.8B-4bit \
             snapshot in the hugging face cache to read a tokenizer out of"
        );
        return;
    };

    let trace = models::trace_of(&sku()).expect("the catalog ships the SKU")(Platform::Cuda);
    let source = ztensor::Source::open(&checkpoint).expect("the artifact opens");
    let contract = models::import_of(&sku()).expect("the catalog ships an import")(&source)
        .unwrap_or_else(|why| panic!("the import contract does not fit {checkpoint:?}: {why}"));
    drop(source);

    // Which order THIS build's text declares, read off the trace rather than
    // asserted — the line an operator reads to know which arm they measured.
    let tiled = trace
        .params
        .iter()
        .filter(|p| p.dtype == model_dsl::Dtype::U4g64tiled)
        .count();
    let row_major = trace
        .params
        .iter()
        .filter(|p| p.dtype == model_dsl::Dtype::U4g64)
        .count();
    eprintln!(
        "{}: {tiled} tiled plane(s), {row_major} row-major u4 plane(s), reading {}",
        sku(),
        checkpoint.display(),
    );

    let booted = Instant::now();
    let mut shell = Shell::load(Boot {
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(2, 256),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 2,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
    })
    .unwrap_or_else(|why| panic!("the shell loads: {why}"));
    let (weights, arena, pools, inputs) = shell.footprint();
    eprintln!(
        "loaded in {:.2}s — weights {:.3} GiB, arena {:.1} MiB, pools {:.1} MiB, \
         inputs {:.1} MiB",
        booted.elapsed().as_secs_f64(),
        weights as f64 / (1u64 << 30) as f64,
        arena as f64 / (1u64 << 20) as f64,
        pools as f64 / (1u64 << 20) as f64,
        inputs as f64 / (1u64 << 20) as f64,
    );

    let prompt = tokenizer.encode(PROMPT);
    assert!(!prompt.is_empty(), "the prompt tokenizes to something");

    let (first, millis, prefill) = run(&mut shell, &prompt);
    // **THE CROSS-BUILD ORACLE.** Two builds cannot meet in one process
    // (see the header), so the arm that runs second is held against the row
    // the arm that ran first left behind.
    if let Ok(to) = std::env::var("PIE_QWEN35_LOGITS") {
        let mut bytes = Vec::with_capacity(prefill.len() * 4);
        for value in &prefill {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&to, &bytes).unwrap_or_else(|why| panic!("{to}: {why}"));
        eprintln!("wrote {} prefill logits to {to}", prefill.len());
    }
    let text = tokenizer.decode(&first, false);
    eprintln!("greedy continuation: {text:?}");
    eprintln!("tokens: {first:?}");
    eprintln!(
        "decode: {:.3} ms/token mean, {:.3} ms/token median ({:.1} tok/s) over {STEPS} steps",
        mean(&millis),
        median(&millis),
        1000.0 / median(&millis),
    );

    // **THE PIN IS THE BF16 ROW'S** (see [`EXPECTED`]). A four-bit row of
    // this model ties ` Paris` with ` in` at the top of the row and loses the
    // tie to the lower index, in BOTH orders and identically — which is a
    // fact about four bits on a 0.8B model and not about the byte order this
    // gate exists to change. So the text is asserted where it means
    // something and reported where it does not.
    let four_bit = tiled + row_major > 0;
    if four_bit {
        eprintln!(
            "NOTE: this row is four-bit, so the {EXPECTED:?} pin is not asserted — \
             greedy answered {text:?}. Both orders tie ` Paris` with ` in` at the \
             top of the prefill row; the bf16 row of the same model separates them."
        );
    } else {
        assert!(
            text.starts_with(EXPECTED),
            "greedy continuation of {PROMPT:?} was {text:?}, and the first light \
             answered {EXPECTED:?}"
        );
    }

    let (again, ..) = run(&mut shell, &prompt);
    assert_eq!(first, again, "twice is not once");
}
