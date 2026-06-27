//! Demonstrates text watermarking for LLM generation.
//!
//! Uses a green/red list approach where tokens are partitioned based on the
//! hash of the previous token, and green-listed tokens receive a probability
//! boost during sampling. The watermark is applied via a manual decode loop:
//! `Probe::Distribution` returns the host distribution per step, and the
//! green-list bias + sampling are applied in user code.

use inferlet::{
    Context, Result,
    sample::Distribution,
};
use serde::Deserialize;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_prompt() -> String { "Explain the LLM decoding process ELI5.".to_string() }
fn default_max_tokens() -> usize { 256 }

/// Watermarking state.
struct WatermarkState {
    /// Proportion of vocabulary in the green list (e.g., 0.5 = 50%).
    gamma: f32,
    /// Bias added to logits of green-listed tokens.
    delta: f32,
    /// The previously generated token ID for green list seeding.
    previous_token: Option<u32>,
}

impl WatermarkState {
    fn new(gamma: f32, delta: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&gamma),
            "gamma must be between 0.0 and 1.0."
        );
        Self { gamma, delta, previous_token: None }
    }

    fn get_seed(&self) -> u64 {
        match self.previous_token {
            Some(token) => {
                let mut hasher = DefaultHasher::new();
                token.hash(&mut hasher);
                hasher.finish()
            }
            None => 0,
        }
    }

    /// Apply watermark bias and select a token from the distribution.
    fn sample(&mut self, ids: &[u32], probs: &[f32]) -> u32 {
        if ids.is_empty() {
            self.previous_token = Some(0);
            return 0;
        }

        let seed = self.get_seed();
        let green_list_size = (ids.len() as f32 * self.gamma).round() as usize;

        let mut indices: Vec<usize> = (0..ids.len()).collect();
        deterministic_shuffle(&mut indices, seed);

        let mut is_green = vec![false; ids.len()];
        for &idx in &indices[..green_list_size.min(indices.len())] {
            is_green[idx] = true;
        }

        let exp_delta = self.delta.exp();
        let mut watermarked: Vec<f32> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| if is_green[i] { p * exp_delta } else { p })
            .collect();

        let prob_sum: f32 = watermarked.iter().sum();
        if prob_sum > 0.0 {
            for p in &mut watermarked {
                *p /= prob_sum;
            }
        }

        let chosen_idx = weighted_sample(&watermarked, seed.wrapping_add(1));
        let chosen_id = ids[chosen_idx];
        self.previous_token = Some(chosen_id);
        chosen_id
    }
}

/// Simple deterministic Fisher-Yates shuffle using xorshift.
fn deterministic_shuffle(indices: &mut [usize], mut seed: u64) {
    for i in (1..indices.len()).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let j = (seed as usize) % (i + 1);
        indices.swap(i, j);
    }
}

/// Simple weighted sampling without external RNG.
fn weighted_sample(probs: &[f32], seed: u64) -> usize {
    let r = (seed as f64 % 1_000_000.0) / 1_000_000.0;
    let r = r as f32;
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let start = Instant::now();
    let stop_tokens = inferlet::chat::stop_tokens();

    let mut ctx = Context::new()?;
    let mut watermark = WatermarkState::new(0.5, 2.0);
    let mut generated_tokens = Vec::new();

    // Build the prompt ourselves so the first Pass feeds it through the
    // forward pass and lands the cue tokens in KV exactly once (auto-
    // commit handles the rest). Subsequent iterations feed one chosen
    // token at a time.
    let mut pending: Vec<u32> = Vec::new();
    pending.extend(inferlet::chat::system(
        "You are a helpful, respectful and honest assistant.",
    ));
    pending.extend(inferlet::chat::user(&input.prompt));
    pending.extend(inferlet::chat::cue());

    for _ in 0..input.max_tokens {
        let mut pass = ctx.forward();
        pass.input(&pending);
        let last_idx = (pending.len() - 1) as u32;
        // `k = 0` returns the full vocabulary so the watermark can bias
        // every token, not just the top-k.
        let h = pass.probe(last_idx, Distribution { temperature: 0.0, k: 0 });
        let out = pass.execute().await?;

        let (ids, probs) = match out.distribution(h) {
            Some(d) => d,
            None => break,
        };

        let chosen = watermark.sample(ids, probs);
        if stop_tokens.contains(&chosen) {
            break;
        }

        generated_tokens.push(chosen);
        pending = vec![chosen];
    }

    let text = inferlet::model::decode(&generated_tokens)?;
    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());
    if !generated_tokens.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (generated_tokens.len() as u32)
        );
    }
    Ok(String::new())
}
