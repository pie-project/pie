//! Demonstrates text watermarking for LLM generation.
//!
//! Uses a green/red list approach where tokens are partitioned based on the
//! hash of the previous token, and green-listed tokens receive a probability
//! boost during sampling. The watermark is applied via a manual decode loop:
//! `Probe::Distribution` returns the host distribution per step, and the
//! green-list bias + sampling are applied in user code.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{probe_program, LoweredProbe, ProbeKind};
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, Result};
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

/// Raw keep-core decode context (no `Context` facade): one KV working set + a
/// sequence cursor. The first sampling fire clears the run-ahead carrier
/// (`fresh_generate`).
struct Ctx {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
}
impl Ctx {
    fn new() -> Self {
        Self { kv: KvWorkingSet::new(), seq_len: 0, fresh: true }
    }
    /// One measurement fire: geometry + input + probe sampler + execute.
    fn probe_fire(&mut self, probe: &LoweredProbe, tokens: &[u32]) -> Result<ForwardPass> {
        let n = tokens.len() as u32;
        let pass = ForwardPass::new();
        if self.fresh {
            pass.fresh_generate();
            self.fresh = false;
        }
        let geom = geometry::ensure_pages(
            &self.kv,
            geometry::kv_write_geometry(self.seq_len, n, self.kv.page_size()),
        )?;
        geometry::attach_kv_write(&pass, &self.kv, &geom);
        let positions: Vec<u32> = (self.seq_len..self.seq_len + n).collect();
        pass.input_tokens(tokens, &positions);
        pass.sampler(&probe.program, probe.bindings(self.seq_len + n - 1)?);
        pass.execute();
        self.seq_len += n;
        Ok(pass)
    }
}

/// Read a dense `[vocab]` f32 distribution off the raw output tensor. `ids` are
/// the implicit identity `0..vocab` (matching the facade `distribution()`).
async fn read_distribution(pass: ForwardPass) -> Result<(Vec<u32>, Vec<f32>)> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    let probs: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ids: Vec<u32> = (0..probs.len() as u32).collect();
    Ok((ids, probs))
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let start = Instant::now();
    let stop_tokens = chat::stop_tokens();
    let vocab = model::output_vocab_size();
    let probe = probe_program(ProbeKind::Distribution, vocab)
        .map_err(|e| format!("probe(Distribution) build: {e}"))?;

    let mut ctx = Ctx::new();
    let mut watermark = WatermarkState::new(0.5, 2.0);
    let mut generated_tokens = Vec::new();

    // Build the prompt ourselves so the first probe fire feeds it through the
    // forward pass and lands the cue tokens in KV exactly once (auto-commit
    // handles the rest). Subsequent iterations feed one chosen token at a time.
    let mut pending: Vec<u32> = Vec::new();
    pending.extend(chat::system(
        "You are a helpful, respectful and honest assistant.",
    ));
    pending.extend(chat::user(&input.prompt));
    pending.extend(chat::cue());

    for _ in 0..input.max_tokens {
        // The full softmax distribution (over all vocab) so the watermark can
        // bias every token, not just the top-k.
        let pass = ctx.probe_fire(&probe, &pending)?;
        let (ids, probs) = read_distribution(pass).await?;

        let chosen = watermark.sample(&ids, &probs);
        if stop_tokens.contains(&chosen) {
            break;
        }

        generated_tokens.push(chosen);
        pending = vec![chosen];
    }

    let text = model::decode(&generated_tokens)?;
    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());
    if !generated_tokens.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (generated_tokens.len() as u32)
        );
    }
    Ok(String::new())
}
