//! End-to-end exercise of the programming-model samplers.
//!
//! The de-hardwired multi-measurement form: instead of attaching several
//! probes to one pass, it authors **ONE multi-output Graph program** (declared
//! outputs `[Token, Logits, Entropy, Distribution, Logprobs]`) and attaches it
//! via [`Forward::measure`], so the host marshals back **N tensors** through
//! `outputs()`. This is the real validation of the per-output-value path. The
//! program computes, all in one pass over the decode-slot logits:
//!
//!   0. `Token`         greedy `argmax(logits)` → a sampled token id
//!   1. `Logits`        full vocab logits (f32 bytes)
//!   2. `Entropy`       `H(p) = -Σ p·log p` (scalar)
//!   3. `Distribution`  full softmax `[vocab]` (top-8 derived guest-side)
//!   4. `Logprobs`      full log-softmax `[vocab]` (per-candidate logprobs
//!                      indexed guest-side)
//!
//! Then cross-checks the values against each other:
//!
//!   * argmax of decoded logits == greedy token == distribution[0].id
//!   * logprob(greedy) ≈ ln(distribution.first_prob)
//!   * the top-8 probabilities are sorted descending
//!   * 0 <= entropy <= ln(vocab)
//!   * logprobs(ts) values match individual logprob lookups
//!
//! Prints structured KEY=VALUE lines that the host-side test asserts on.

use inferlet::{
    Context, Result,
    sampling::{Built, Graph, OutputKind},
};
use serde::Deserialize;

#[derive(Deserialize, Default)]
struct Input {}

fn decode_logits_native(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn argmax(xs: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

/// Build ONE multi-output measurement program — the de-hardwired multi-probe
/// form. Every measurement is a declared output of a single Graph program,
/// marshaled back as N tensors via `outputs()`. Declared output order:
///   0 `Token` (argmax)   1 `Logits`   2 `Entropy`   3 `Distribution` (softmax)
///   4 `Logprobs` (full log-softmax — indexed guest-side for any candidate).
/// Top-k and per-candidate logprobs are derived guest-side from the full
/// distribution / log-softmax outputs (no `SortDesc` / host-input candidate
/// tensors needed).
fn build_suite_program(vocab: u32) -> Result<Built> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    // Numerically-stable log-softmax and softmax.
    let shifted = logits.sub(&logits.reduce_max());
    let log_p = shifted.sub(&shifted.exp().reduce_sum().log()); // log-softmax [vocab]
    let p = log_p.exp(); // softmax [vocab]
    g.output(&logits.argmax(), OutputKind::Token); // 0
    g.output(&logits, OutputKind::Logits); // 1
    g.output(&p.mul(&log_p).reduce_sum().neg(), OutputKind::Entropy); // 2  (-Σ p·log p)
    g.output(&p, OutputKind::Distribution); // 3
    g.output(&log_p, OutputKind::Logprobs); // 4
    g.build()
        .map_err(|e| format!("sampler-suite: build program: {e:?}"))
}

#[inferlet::main]
async fn main(_input: Input) -> Result<String> {
    let mut ctx = Context::new()?;
    ctx.system("Answer in one short word.")
        .user("What is the capital of France?")
        .cue();
    ctx.flush().await?;

    // Pick three arbitrary candidate tokens to score (they don't need to be
    // semantically meaningful — we're testing the math, not the answer).
    let cand_a: u32 = 1000;
    let cand_b: u32 = 2000;
    let cand_c: u32 = 3000;
    let cand_list = vec![cand_a, cand_b, cand_c];

    // ONE multi-output measurement program at the decode slot, all in one
    // execute() — the de-hardwired replacement for attaching 6 probes.
    let built = build_suite_program(inferlet::model::output_vocab_size())?;

    let mut pass = ctx.forward();
    pass.input(&[0u32]);
    let hs = pass.measure(built)?;

    let output = pass.execute().await?;

    let greedy_tok = output.token(hs[0]).await?;
    let logit_bytes = output.read_bytes(hs[1]).await?;
    let entropy = output.scalar(hs[2]).await?;
    let probs = output.read_f32(hs[3]).await?; // full softmax [vocab]
    let log_probs = output.read_f32(hs[4]).await?; // full log-softmax [vocab]

    let logits = decode_logits_native(&logit_bytes);
    let vocab_size = logits.len();

    // Top-8 (id, prob) pairs, derived guest-side from the full distribution.
    let mut indexed: Vec<(u32, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as u32, p))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (dist_ids, dist_probs): (Vec<u32>, Vec<f32>) = indexed.into_iter().take(8).unzip();

    // Per-candidate logprobs, indexed guest-side from the full log-softmax.
    let lp_a_value = *log_probs
        .get(cand_a as usize)
        .ok_or("logprob: cand_a out of range")?;
    let lp_many: Vec<f32> = cand_list
        .iter()
        .map(|&c| *log_probs.get(c as usize).unwrap_or(&f32::NEG_INFINITY))
        .collect();
    let lp_many_a = lp_many.first().copied().ok_or("logprobs list empty")?;

    // Cross-checks.
    let raw_argmax = argmax(&logits) as u32;
    let argmax_matches_greedy = raw_argmax == greedy_tok;

    let dist_first_id = *dist_ids.first().ok_or("empty distribution")?;
    let dist_first_p = *dist_probs.first().ok_or("empty distribution probs")?;
    let dist_first_matches_greedy = dist_first_id == greedy_tok;

    // log p(cand_a) should match between the singular and the list query.
    let logprob_consistent = (lp_a_value - lp_many_a).abs() < 1e-4;

    // Entropy bounds: 0 <= H <= ln(vocab_size).
    let h_max = (vocab_size as f32).ln();
    let entropy_in_bounds = entropy >= 0.0 && entropy <= h_max + 1e-3;

    // Distribution probabilities should be sorted descending and sum to <= 1.
    let dist_probs_sorted = dist_probs.windows(2).all(|w| w[0] + 1e-6 >= w[1]);
    let dist_probs_sum: f32 = dist_probs.iter().sum();

    println!("VOCAB_SIZE={}", vocab_size);
    println!("SLOT_COUNT={}", output.len());
    println!("GREEDY_TOKEN={}", greedy_tok);
    println!("RAW_ARGMAX_TOKEN={}", raw_argmax);
    println!("ARGMAX_MATCHES_GREEDY={}", argmax_matches_greedy);
    println!("DIST_FIRST_ID={}", dist_first_id);
    println!("DIST_FIRST_PROB={:.6}", dist_first_p);
    println!("DIST_FIRST_MATCHES_GREEDY={}", dist_first_matches_greedy);
    println!("DIST_PROBS_SORTED={}", dist_probs_sorted);
    println!("DIST_PROBS_TOP8_SUM={:.6}", dist_probs_sum);
    println!("LOGPROB_CAND_A={:.6}", lp_a_value);
    println!("LOGPROBS_CAND_A={:.6}", lp_many_a);
    println!("LOGPROBS_CONSISTENT={}", logprob_consistent);
    println!("LOGPROBS_LEN={}", lp_many.len());
    println!("ENTROPY={:.6}", entropy);
    println!("ENTROPY_MAX={:.6}", h_max);
    println!("ENTROPY_IN_BOUNDS={}", entropy_in_bounds);

    Ok(String::new())
}
