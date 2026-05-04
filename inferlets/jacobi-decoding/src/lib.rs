//! Demonstrates Jacobi decoding — parallel speculation via fixed-point iteration.
//!
//! Instead of autoregressive decoding, Jacobi decoding initializes N positions
//! with guessed tokens, then iteratively runs forward passes until the
//! predictions converge (reach a fixed point). This can verify multiple
//! tokens per forward pass.
//!
//! All tokens (verified anchor + speculative guesses) are sent as regular
//! `input` so the runtime's working-token bookkeeping stays consistent.
//! After each forward pass, only the accepted prefix is committed to the
//! KV cache; the rejected suffix is truncated via [`Context::truncate`].

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
    #[serde(default = "default_window_size")]
    window_size: usize,
}

fn default_prompt() -> String { "Write a poem about the ocean.".to_string() }
fn default_max_tokens() -> usize { 256 }
fn default_window_size() -> usize { 5 }

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_tokens = input.max_tokens;
    let window_size = input.window_size;

    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;
    let tokenizer = model.tokenizer();
    let stop_tokens = inferlet::chat::stop_tokens(&model);

    let mut ctx = Context::new(&model)?;
    ctx.system("You are a helpful assistant.")
        .user(&input.prompt)
        .cue();
    ctx.flush().await?;

    println!(
        "--- Jacobi Decoding (window_size={}, page_size={}) ---",
        window_size,
        ctx.page_size()
    );

    let mut all_generated: Vec<u32> = Vec::new();

    // Bootstrap: append the last cue token to drive a single forward pass
    // and read its next-token prediction.
    let cue = inferlet::chat::cue(&model);
    let trigger = *cue.last().unwrap_or(&0);
    let first_token = {
        let mut pass = ctx.forward();
        pass.input(&[trigger]);
        let h = pass.sample(&[0], Sampler::Argmax);
        pass.execute()
            .await?
            .token(h)
            .ok_or("Bootstrap produced no token")?
    };
    all_generated.push(first_token);

    let mut anchor = first_token;
    let mut window: Vec<u32> = vec![anchor; window_size];
    let mut total_accepted = 1; // anchor

    while total_accepted < max_tokens {
        // Verifier pass: feed [anchor] + [window guesses] and sample at every
        // position. Pass auto-commits the entire input window — we'll
        // truncate the rejected suffix after seeing the accepted count.
        let mut input_all = vec![anchor];
        input_all.extend_from_slice(&window);
        let input_count = input_all.len();

        let mut pass = ctx.forward();
        pass.input(&input_all);
        let sample_indices: Vec<u32> = (0..input_count as u32).collect();
        let h = pass.sample(&sample_indices, Sampler::Argmax);
        let out = pass.execute().await?;
        let predicted = out.tokens_at(h);

        if predicted.is_empty() {
            break;
        }

        // Jacobi verification: longest converged prefix.
        let mut accepted_count = 1;
        for i in 1..predicted.len().min(window.len() + 1) {
            let i_window = i - 1;
            if i_window < window.len() && predicted[i - 1] == window[i_window] {
                accepted_count += 1;
            } else {
                break;
            }
        }

        let newly_accepted: Vec<u32> =
            predicted[..accepted_count.min(predicted.len())].to_vec();

        // Stop token check.
        let mut stop_at = newly_accepted.len();
        for (i, &t) in newly_accepted.iter().enumerate() {
            if stop_tokens.contains(&t) {
                stop_at = i;
                break;
            }
        }
        let final_accepted = &newly_accepted[..stop_at];
        all_generated.extend_from_slice(final_accepted);
        total_accepted += final_accepted.len();

        // Truncate the unaccepted suffix from KV.
        let speculative_count = (input_count as u32) - (1 + final_accepted.len() as u32);
        if speculative_count > 0 {
            ctx.truncate(speculative_count);
        }

        if stop_at < newly_accepted.len() || total_accepted >= max_tokens {
            break;
        }

        let last_accepted = *final_accepted.last().unwrap();
        anchor = last_accepted;

        // Next window: take fresh predictions past the accepted prefix,
        // padding with the anchor where short.
        window = if accepted_count < predicted.len() {
            let mut w: Vec<u32> = predicted[accepted_count..].to_vec();
            w.truncate(window_size);
            while w.len() < window_size {
                w.push(last_accepted);
            }
            w
        } else {
            vec![last_accepted; window_size]
        };
    }

    let text = tokenizer.decode(&all_generated)?;
    println!(
        "Generated {} tokens in {:?} ({:.1} tokens/s)",
        all_generated.len(),
        start.elapsed(),
        all_generated.len() as f64 / start.elapsed().as_secs_f64()
    );
    println!("Output:\n{}", text);

    Ok(String::new())
}
