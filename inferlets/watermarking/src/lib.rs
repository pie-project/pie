//! Demonstrates text watermarking for text generation.
//!
//! Uses a green/red list approach where tokens are partitioned based on the
//! hash of the previous token, and green-listed tokens receive a probability
//! boost during sampling. The watermark is applied via manual ForwardPass
//! decoding with Sampler::Dist to get distributions, then custom sampling.

use inferlet::{
    context::Context, model::Model, runtime,
    InstructExt, ForwardPassExt, Result,
    inference::{ForwardPass, Output, Sampler},
};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;

const HELP: &str = "\
Usage: watermarking [OPTIONS]

A program to demonstrate watermarked text generation.

Options:
  -p, --prompt <PROMPT>      The prompt text [default: \"Explain the LLM decoding process ELI5.\"]
  -n, --max-tokens <TOKENS>  Maximum number of tokens to generate [default: 256]
  -h, --help                 Prints this help message";

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
        Self {
            gamma,
            delta,
            previous_token: None,
        }
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

        // Deterministic green list generation using Fisher-Yates with seeded state
        let green_list_size = (ids.len() as f32 * self.gamma).round() as usize;

        // Create deterministic index permutation seeded by previous token hash
        let mut indices: Vec<usize> = (0..ids.len()).collect();
        deterministic_shuffle(&mut indices, seed);

        let mut is_green = vec![false; ids.len()];
        for &idx in &indices[..green_list_size.min(indices.len())] {
            is_green[idx] = true;
        }

        // Apply bias to green-listed tokens
        let exp_delta = self.delta.exp();
        let mut watermarked_probs: Vec<f32> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| if is_green[i] { p * exp_delta } else { p })
            .collect();

        // Normalize
        let prob_sum: f32 = watermarked_probs.iter().sum();
        if prob_sum > 0.0 {
            for p in &mut watermarked_probs {
                *p /= prob_sum;
            }
        }

        // Deterministic-weighted sampling using accumulated probabilities + hash
        let chosen_idx = weighted_sample(&watermarked_probs, seed.wrapping_add(1));
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
    // Generate a pseudo-random value in [0, 1) from the seed
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
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Explain the LLM decoding process ELI5.".to_string());

    let max_num_outputs: usize = args
        .value_from_str(["-n", "--max-tokens"])
        .unwrap_or(256);

    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;
    let tokenizer = model.tokenizer();
    let stop_tokens = Context::stop_tokens(&model);

    let ctx = Context::create(&model)?;

    ctx.system("You are a helpful, respectful and honest assistant.");
    ctx.user(&prompt);
    ctx.cue();

    // Manual decode loop with watermark sampling
    let mut watermark = WatermarkState::new(0.5, 2.0);
    let mut generated_tokens = Vec::new();

    for step in 0..max_num_outputs {
        let seq_len = ctx.last_position().map(|p| p + 1).unwrap_or(0);
        let buffered = ctx.buffered_tokens();
        if buffered.is_empty() {
            break;
        }

        let page_size = ctx.tokens_per_page();
        let cursor = ctx.cursor();
        let total_tokens_after = cursor + buffered.len() as u32;
        let total_pages_needed = (total_tokens_after + page_size - 1) / page_size;
        if total_pages_needed > 0 {
            ctx.reserve_pages(total_pages_needed)
                .map_err(|e| format!("Failed to reserve pages at step {}: {}", step, e))?;
        }

        let pass = ForwardPass::new(&model);
        pass.context(&ctx);
        let positions: Vec<u32> = (seq_len..seq_len + buffered.len() as u32).collect();
        pass.input_tokens(&buffered, &positions);

        // Use Dist sampler to get the probability distribution
        let last_idx = (buffered.len() - 1) as u32;
        pass.sampler(&[last_idx], Sampler::Dist((0.0, 0)));

        let output = pass.execute_async().await
            .map_err(|e| format!("Forward pass failed at step {}: {}", step, e))?;

        // Commit pages and update cursor
        let new_cursor_abs = cursor + buffered.len() as u32;
        let pages_to_commit = new_cursor_abs / page_size;
        if pages_to_commit > 0 {
            let page_indices: Vec<u32> = (0..pages_to_commit).collect();
            ctx.commit_pages(&page_indices)
                .map_err(|e| format!("Failed to commit pages: {}", e))?;
        }
        ctx.set_cursor(new_cursor_abs % page_size);

        // Extract distribution and apply watermark sampling
        let chosen_token = match output {
            Output::Distributions(dists) => {
                if let Some((ids, probs)) = dists.first() {
                    watermark.sample(ids, probs)
                } else {
                    break;
                }
            }
            _ => break,
        };

        // Check for stop tokens
        if stop_tokens.contains(&chosen_token) {
            break;
        }

        generated_tokens.push(chosen_token);
        ctx.set_buffered_tokens(&[chosen_token]);
    }

    let text = tokenizer.decode(&generated_tokens)?;
    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());

    if !generated_tokens.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (generated_tokens.len() as u32)
        );
    }

    Ok(String::new())
}
