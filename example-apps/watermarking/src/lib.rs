//! Demonstrates text watermarking for text generation.
//!
//! Uses a green/red list approach where tokens are partitioned based on the
//! hash of the previous token, and green-listed tokens receive a probability
//! boost during sampling.

use inferlet::sampler::{Sample, Sampler};
use inferlet::stop_condition::StopCondition;
use inferlet::{Args, Result, anyhow, stop_condition};
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use std::cell::RefCell;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;

/// Injects a watermark by partitioning the vocabulary into a "green list" and a "red list"
/// based on the hash of the previous token. It then boosts the probabilities of tokens
/// in the green list.
struct WatermarkSampler {
    inner: RefCell<Inner>,
}

struct Inner {
    /// The proportion of the vocabulary to be included in the green list (e.g., 0.5 for 50%).
    gamma: f32,
    /// The bias added to the logits of the green-listed tokens to increase their probability.
    delta: f32,
    /// The previously generated token ID, used to seed the green/red list generation.
    /// `None` indicates the start of a sequence.
    previous_token: Option<u32>,
    /// Random number generator for sampling.
    rng: ThreadRng,
}

impl WatermarkSampler {
    /// Creates a new `WatermarkSampler`.
    ///
    /// # Arguments
    /// * `gamma` - The proportion of the vocabulary to be in the green list (0.0 to 1.0).
    /// * `delta` - The bias to add to the logits of green-listed tokens.
    pub fn new(gamma: f32, delta: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&gamma),
            "gamma must be between 0.0 and 1.0."
        );
        Self {
            inner: RefCell::new(Inner {
                gamma,
                delta,
                previous_token: None, // No previous token at the beginning
                rng: rand::rng(),
            }),
        }
    }
}

impl Inner {
    /// Hashes the previous token to create a seed for the RNG.
    /// This ensures the green/red list is deterministic based on the context.
    fn get_seed(&self) -> u64 {
        match self.previous_token {
            Some(token) => {
                let mut hasher = DefaultHasher::new();
                token.hash(&mut hasher);
                hasher.finish()
            }
            // Use a default seed for the very first token in a sequence.
            None => 0,
        }
    }
}

impl Sample for WatermarkSampler {
    fn sample(&self, ids: &[u32], probs: &[f32]) -> u32 {
        let mut inner = self.inner.borrow_mut();

        if ids.is_empty() {
            // Handle the edge case of an empty input.
            // This could be an error or a special token like EOS.
            // Here, we'll arbitrarily return 0.
            inner.previous_token = Some(0);
            return 0;
        }

        let seed = inner.get_seed();
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Create a new set of probabilities by applying the watermark bias.
        let mut watermarked_probs = probs.to_vec();

        // The number of tokens to be considered "green" from the given `ids`.
        let green_list_size = (ids.len() as f32 * inner.gamma).round() as usize;

        // Generate green list indices based on the provided `ids`.
        // We do this by creating a shuffled list of indices from 0 to ids.len() - 1
        // and taking the first `green_list_size` elements.
        let mut indices: Vec<usize> = (0..ids.len()).collect();
        indices.shuffle(&mut seeded_rng);

        let green_indices: std::collections::HashSet<usize> =
            indices.into_iter().take(green_list_size).collect();

        // Apply the bias `delta` to the log-probabilities of the green-listed tokens.
        // Since we have probabilities, we convert to logits, add delta, then convert back.
        // logit = log(p)
        // new_logit = log(p) + delta
        // new_p = exp(log(p) + delta) = p * exp(delta)
        let exp_delta = inner.delta.exp();
        for i in 0..watermarked_probs.len() {
            if green_indices.contains(&i) {
                watermarked_probs[i] *= exp_delta;
            }
        }

        // Normalize the probabilities so they sum to 1 again.
        let prob_sum: f32 = watermarked_probs.iter().sum();
        if prob_sum > 0.0 {
            for p in &mut watermarked_probs {
                *p /= prob_sum;
            }
        }

        // Sample from the new, watermarked distribution.
        let dist = WeightedIndex::new(&watermarked_probs)
            .expect("Failed to create watermarked distribution.");
        let chosen_idx = dist.sample(&mut inner.rng);

        let chosen_id = ids[chosen_idx];

        // Update the previous token for the next sampling step.
        inner.previous_token = Some(chosen_id);

        chosen_id
    }
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    let prompt = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Explain the LLM decoding process ELI5.".to_string());

    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);

    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(anyhow!(
            "Unknown arguments found: {:?}. Use --help for usage.",
            remaining
        ));
    }

    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();
    let stop_condition = stop_condition::max_len(max_num_outputs)
        .or(stop_condition::ends_with_any(model.eos_tokens()));

    let mut ctx = model.create_context();

    let sampler = Sampler::Custom {
        temperature: 0.0,
        sampler: Box::new(WatermarkSampler::new(0.5, 0.0)),
    };

    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    let text = ctx.generate(sampler, stop_condition).await;
    let token_ids = tokenizer.tokenize(&text);
    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());

    // Compute per-token latency, avoiding division by zero.
    if !token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (token_ids.len() as u32)
        );
    }

    Ok(())
}
