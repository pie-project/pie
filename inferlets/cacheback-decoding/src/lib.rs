//! Demonstrates CacheBack speculative decoding — a manual loop with a
//! cached draft model.
//!
//! A separate "drafter" context generates candidate tokens; the main
//! context verifies them in a single forward pass. Accepted tokens commit;
//! rejected tokens are rolled back from both contexts via
//! [`Context::truncate`]. Draft tokens are sent as regular `input` (not
//! speculative) so working-token bookkeeping stays consistent.

use inferlet::{
    Context, Result,
    model::Model,
    runtime,
    sample::Sampler,
};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_draft_length")]
    draft_length: usize,
}

fn default_prompt() -> String { "Explain quantum computing.".to_string() }
fn default_max_tokens() -> usize { 256 }
fn default_draft_length() -> usize { 4 }

/// Simple greedy drafter on its own context.
struct GreedyDrafter {
    ctx: Context,
}

impl GreedyDrafter {
    fn new(model: &Model) -> Result<Self> {
        Ok(Self { ctx: Context::new(model)? })
    }

    /// Generate `draft_length` greedy tokens starting from `seed`.
    /// Returns whatever fraction completes before any forward pass fails.
    async fn draft(&mut self, seed: u32, draft_length: usize) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(draft_length);
        let mut current = seed;

        for _ in 0..draft_length {
            let mut pass = self.ctx.forward();
            pass.input(&[current]);
            let h = pass.sample(&[0], Sampler::Argmax);
            let out = match pass.execute().await {
                Ok(o) => o,
                Err(_) => break,
            };
            match out.token(h) {
                Some(t) => {
                    current = t;
                    tokens.push(t);
                }
                None => break,
            }
        }
        tokens
    }

    fn rollback(&mut self, n: u32) {
        self.ctx.truncate(n);
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let prompt = input.prompt;
    let max_tokens = input.max_tokens;
    let draft_length = input.draft_length;

    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;
    let tokenizer = model.tokenizer();
    let stop_tokens = inferlet::chat::stop_tokens(&model);

    let mut ctx = Context::new(&model)?;

    // Fill prompt and flush to populate the KV cache.
    ctx.system("You are a helpful assistant.")
        .user(&prompt)
        .cue();
    ctx.flush().await?;

    // Bootstrap: append the last cue token to drive a single forward pass
    // and read its next-token prediction.
    let first_token = {
        let cue = inferlet::chat::cue(&model);
        let trigger = *cue.last().unwrap_or(&0);
        let mut pass = ctx.forward();
        pass.input(&[trigger]);
        let h = pass.sample(&[0], Sampler::Argmax);
        pass.execute().await?
            .token(h)
            .ok_or("Bootstrap produced no token")?
    };

    let mut drafter = GreedyDrafter::new(&model)?;

    let mut all_generated: Vec<u32> = vec![first_token];
    let mut anchor = first_token;
    let mut total_accepted = 1usize;
    let mut total_steps = 0usize;

    while total_accepted < max_tokens {
        // Step 1: draft tokens off the secondary context.
        let draft_tokens = drafter.draft(anchor, draft_length).await;
        if draft_tokens.is_empty() {
            break;
        }

        // Step 2: verification pass. Sample at every position so we can
        // compare draft predictions vs. anchor-conditioned predictions.
        let mut verify_input = vec![anchor];
        verify_input.extend_from_slice(&draft_tokens);
        let input_count = verify_input.len();

        let mut pass = ctx.forward();
        pass.input(&verify_input);
        let sample_indices: Vec<u32> = (0..input_count as u32).collect();
        let h = pass.sample(&sample_indices, Sampler::Argmax);
        let out = pass.execute().await?;
        let verified = out.tokens_at(h);

        if verified.is_empty() {
            break;
        }

        // Step 3: count how many drafts the verifier accepts. The first
        // verified token is always accepted (anchor's own next prediction);
        // each subsequent draft accepted iff verified[i-1] == draft[i-1].
        let mut accepted_count = 1;
        for i in 1..verified.len().min(draft_tokens.len() + 1) {
            let draft_idx = i - 1;
            if draft_idx < draft_tokens.len() && verified[i - 1] == draft_tokens[draft_idx] {
                accepted_count += 1;
            } else {
                break;
            }
        }

        let newly_accepted: Vec<u32> = verified[..accepted_count.min(verified.len())].to_vec();

        // Step 4: truncate rejected tokens off the verifier context.
        // Pass committed all `input_count` tokens; only the accepted prefix
        // should remain. The rest of the page state re-syncs in `truncate`.
        let n_rejected = (input_count as u32) - (accepted_count as u32);
        ctx.truncate(n_rejected);

        // Roll back rejected drafts from the drafter too. The drafter
        // wrote `draft_length` tokens; `accepted_count - 1` of them were
        // accepted (the rest is the anchor's own first prediction).
        let drafter_rejected = draft_length as u32 - (accepted_count.saturating_sub(1) as u32);
        drafter.rollback(drafter_rejected);

        // Stop on the first stop token.
        let mut hit_stop = false;
        for &t in &newly_accepted {
            if stop_tokens.contains(&t) {
                hit_stop = true;
                break;
            }
            all_generated.push(t);
            total_accepted += 1;
        }
        if hit_stop || total_accepted >= max_tokens {
            break;
        }

        anchor = *newly_accepted.last().unwrap_or(&anchor);
        total_steps += 1;
    }

    let text = tokenizer.decode(&all_generated)?;
    println!(
        "--- CacheBack Decoding (draft_length={}, steps={}) ---",
        draft_length, total_steps
    );
    println!("Generated in {:?}", start.elapsed());
    println!("Output:\n{}", text);

    Ok(String::new())
}
