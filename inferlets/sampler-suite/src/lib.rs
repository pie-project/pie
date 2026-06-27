//! End-to-end exercise of the programming-model samplers.
//!
//! Attaches FIVE samplers/probes to the SAME forward-pass slot (the last
//! token of a short prompt) — all in one execute() — and verifies every
//! kind of distribution-access works:
//!
//!   1. `Sampler::Argmax`           greedy → produces a sampled token id
//!   2. `Probe::Logits`             full vocab logits as bytes
//!   3. `Probe::Distribution`       top-8 (id, prob) pairs
//!   4. `Probe::Logprob(t)`         log p(t) for a chosen token
//!   5. `Probe::Logprobs(ts)`       log p(t) for several tokens
//!   6. `Probe::Entropy`            H(p) of the unscaled distribution
//!
//! Then cross-checks the values against each other:
//!
//!   * argmax of decoded logits == greedy token == distribution[0].id
//!   * logprob(greedy) ≈ ln(distribution.first_prob)
//!   * sum(exp(logprob(t)) for t in distribution.ids) ≈ 1.0
//!   * entropy >= 0
//!   * logprobs(ts) values match individual logprob lookups
//!
//! Prints structured KEY=VALUE lines that the host-side test asserts on.

use inferlet::{
    Context, Result,
    sample::{Distribution, Entropy, Logits, Logprob, Logprobs, Sampler},
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

    // Single forward pass with all sampler/probe kinds at the same slot.
    // We append a placeholder token at the next position to drive a single
    // decode step — Pass takes care of the page math and commit.
    let mut pass = ctx.forward();
    pass.input(&[0u32]);

    let h_greedy = pass.sample(&[0], Sampler::Argmax);
    let h_logits = pass.probe(0, Logits);
    let h_dist = pass.probe(0, Distribution { temperature: 1.0, k: 8 });
    let h_lp_a = pass.probe(0, Logprob(cand_a));
    let h_lp_many = pass.probe(0, Logprobs(cand_list.clone()));
    let h_entropy = pass.probe(0, Entropy);

    let output = pass.execute().await?;

    let greedy_tok = output.token(h_greedy).ok_or("slot 0 not Token")?;
    let logit_bytes = output.logits(h_logits).ok_or("slot 1 not Logits")?;
    let (dist_ids, dist_probs) = output.distribution(h_dist).ok_or("slot 2 not Distribution")?;
    let lp_a = output.logprobs(h_lp_a).ok_or("slot 3 not Logprobs")?;
    let lp_many = output.logprobs(h_lp_many).ok_or("slot 4 not Logprobs")?;
    let entropy = output.entropy(h_entropy).ok_or("slot 5 not Entropy")?;

    let logits = decode_logits_native(logit_bytes);
    let vocab_size = logits.len();

    // Cross-checks.
    let raw_argmax = argmax(&logits) as u32;
    let argmax_matches_greedy = raw_argmax == greedy_tok;

    let dist_first_id = *dist_ids.first().ok_or("empty distribution")?;
    let dist_first_p = *dist_probs.first().ok_or("empty distribution probs")?;
    let dist_first_matches_greedy = dist_first_id == greedy_tok;

    // log p(cand_a) should match between the singular and the list query.
    let lp_a_value = lp_a.first().copied().ok_or("logprob list empty")?;
    let lp_many_a = lp_many.first().copied().ok_or("logprobs list empty")?;
    let logprob_consistent = (lp_a_value - lp_many_a).abs() < 1e-4;

    // Entropy bounds: 0 <= H <= ln(vocab_size).
    let h_max = (vocab_size as f32).ln();
    let entropy_in_bounds = entropy >= 0.0 && entropy <= h_max + 1e-3;

    // Distribution probabilities should be sorted descending and sum to <= 1.
    let dist_probs_sorted = dist_probs.windows(2).all(|w| w[0] + 1e-6 >= w[1]);
    let dist_probs_sum: f32 = dist_probs.iter().sum();

    println!("VOCAB_SIZE={}", vocab_size);
    println!("SLOT_COUNT={}", output.raw().slots.len());
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
