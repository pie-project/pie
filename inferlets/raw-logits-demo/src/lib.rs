//! Raw-logits correctness + overhead test.
//!
//! For a fixed prompt, runs N decode steps on a fresh forked context per
//! iteration, alternating between two slot modes:
//!
//!   * `Sampler::Argmax`            — greedy, returns one token id.
//!   * `Probe::Logits` (`Logits`)   — full pre-softmax logit vector packed
//!                                    as native-endian f32 bytes.
//!
//! Asserts that argmax(logits) == greedy token (per iteration), then prints
//! structured `KEY=VALUE` lines that the host-side test parses:
//!
//!   VOCAB_SIZE=<n>
//!   ITERS=<n>
//!   GREEDY_AVG_MS=<f>            mean across iters (sensitive to first-call JIT)
//!   GREEDY_MIN_MS=<f>            steady-state best — most informative
//!   RAW_LOGITS_AVG_MS=<f>
//!   RAW_LOGITS_MIN_MS=<f>
//!   OVERHEAD_AVG_MS=<f>          (raw_logits_avg - greedy_avg)
//!   OVERHEAD_MIN_MS=<f>          (raw_logits_min - greedy_min) — steady-state cost
//!   PAYLOAD_BYTES=<n>            vocab_size * 4
//!   ARGMAX_MATCHES_GREEDY=<n>/<n>

use inferlet::{
    Context, Result,
    forward::{Forward, Probe},
    sample::Sampler,
};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize, Default)]
struct Input {
    /// Number of paired (greedy + raw-logits) decode steps to time.
    #[serde(default = "default_iters")]
    iters: usize,
}
fn default_iters() -> usize { 50 }

fn decode_logits(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as u32;
        }
    }
    best_i
}

/// What we want to measure on each iteration.
enum Mode {
    Greedy,
    RawLogits,
}

/// Run a single decode step on a forked context; return (elapsed_ms,
/// greedy_token_or_logits). The forked context already has the prompt's KV
/// committed; we append a single placeholder token to drive one decode step.
async fn timed_step(base: &Context, mode: Mode) -> Result<(f64, GreedyOrLogits)> {
    let mut ctx = base.fork()?;
    let mut pass: Forward = ctx.forward();
    pass.input(&[0u32]);

    match mode {
        Mode::Greedy => {
            let h = pass.sample(Sampler::Argmax)?[0];
            let t0 = Instant::now();
            let out = pass.execute().await?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let token = out.token(h).await?;
            Ok((elapsed_ms, GreedyOrLogits::Greedy(token)))
        }
        Mode::RawLogits => {
            let h = pass.probe(Probe::Logits)?[0];
            let t0 = Instant::now();
            let out = pass.execute().await?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let bytes = out.read_bytes(h).await?;
            Ok((elapsed_ms, GreedyOrLogits::Logits(bytes)))
        }
    }
}

enum GreedyOrLogits {
    Greedy(u32),
    Logits(Vec<u8>),
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let iters = input.iters.max(1);

    let mut base = Context::new()?;

    // Build a small fixed prompt and prefill it once (one Pass — page math
    // and commit happen inside).
    base.system("You complete the user's sentence with a single word.")
        .user("The capital of France is")
        .cue();
    base.flush().await?;

    // Warmup. Covers JIT compile of attention / sampling / softmax kernels
    // plus first-call allocator paths. Five paired rounds is enough on
    // flashinfer/triton in our experience.
    for _ in 0..5 {
        let _ = timed_step(&base, Mode::Greedy).await?;
        let _ = timed_step(&base, Mode::RawLogits).await?;
    }

    // Timed loop.
    let mut greedy_total = 0.0_f64;
    let mut raw_total = 0.0_f64;
    let mut greedy_min = f64::INFINITY;
    let mut raw_min = f64::INFINITY;
    let mut matches = 0usize;
    let mut vocab_size = 0usize;

    for i in 0..iters {
        let (g_ms, g) = timed_step(&base, Mode::Greedy).await?;
        let GreedyOrLogits::Greedy(greedy_token) = g else { unreachable!() };
        greedy_total += g_ms;
        if g_ms < greedy_min { greedy_min = g_ms; }

        let (r_ms, r) = timed_step(&base, Mode::RawLogits).await?;
        let GreedyOrLogits::Logits(bytes) = r else { unreachable!() };
        let logits = decode_logits(&bytes);
        if vocab_size == 0 {
            vocab_size = logits.len();
        }
        let argmax_token = argmax(&logits);
        if argmax_token == greedy_token {
            matches += 1;
        } else {
            println!(
                "iter {i}: argmax({}) != greedy({}) — model may be applying a sampling mask",
                argmax_token, greedy_token
            );
        }
        raw_total += r_ms;
        if r_ms < raw_min { raw_min = r_ms; }
    }

    let greedy_avg = greedy_total / iters as f64;
    let raw_avg = raw_total / iters as f64;
    let overhead_avg = raw_avg - greedy_avg;
    let overhead_min = raw_min - greedy_min;
    let payload_bytes = vocab_size * 4;

    println!("VOCAB_SIZE={}", vocab_size);
    println!("ITERS={}", iters);
    println!("GREEDY_AVG_MS={:.3}", greedy_avg);
    println!("GREEDY_MIN_MS={:.3}", greedy_min);
    println!("RAW_LOGITS_AVG_MS={:.3}", raw_avg);
    println!("RAW_LOGITS_MIN_MS={:.3}", raw_min);
    println!("OVERHEAD_AVG_MS={:.3}", overhead_avg);
    println!("OVERHEAD_MIN_MS={:.3}", overhead_min);
    println!("PAYLOAD_BYTES={}", payload_bytes);
    println!("ARGMAX_MATCHES_GREEDY={}/{}", matches, iters);

    Ok(String::new())
}
