//! Demonstrates windowed attention — sliding-window KV management.
//!
//! After filling the prompt, applies a sliding window attention mask during
//! generation to limit the model's attention to the most recent
//! `window_size` tokens. This simulates bounded-memory generation.
//!
//! NOTE: full KV cache eviction is not yet supported by the runtime — the
//! mask only prevents the model from *attending* to old tokens; the KV
//! pages stay in memory.

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
    window_size: u32,
}

fn default_prompt() -> String { "Tell me a long story about a cat.".to_string() }
fn default_max_tokens() -> usize { 512 }
fn default_window_size() -> u32 { 64 }

/// BRLE attention mask for a sliding window: the most recent `window_size`
/// positions attend, everything before is masked.
fn build_window_mask(seq_len: u32, window_size: u32) -> Vec<u32> {
    if seq_len <= window_size {
        vec![0, seq_len]
    } else {
        vec![seq_len - window_size, window_size]
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    let mut ctx = Context::new(&model)?;
    let stop_tokens = inferlet::chat::stop_tokens(&model);

    let mut prompt: Vec<u32> = Vec::new();
    prompt.extend(inferlet::chat::system(&model, "You are a helpful assistant."));
    prompt.extend(inferlet::chat::user(&model, &input.prompt));
    prompt.extend(inferlet::chat::cue(&model));

    println!(
        "--- Windowed Attention (window={} tokens, page_size={}) ---",
        input.window_size,
        ctx.page_size()
    );

    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut pending: Vec<u32> = prompt;

    for _ in 0..input.max_tokens {
        if pending.is_empty() {
            break;
        }

        let mut pass = ctx.forward();
        let total_seq_after = pass.start_position() + pending.len() as u32;
        pass.input(&pending);

        if total_seq_after > input.window_size {
            let mask = build_window_mask(total_seq_after, input.window_size);
            let masks: Vec<Vec<u32>> = (0..pending.len()).map(|_| mask.clone()).collect();
            pass.attention_mask(&masks);
        }

        let last_idx = (pending.len() - 1) as u32;
        let h = pass.sample(&[last_idx], Sampler::Argmax);
        let out = pass.execute().await?;

        let token = match out.token(h) {
            Some(t) => t,
            None => break,
        };
        if stop_tokens.contains(&token) {
            break;
        }
        generated_tokens.push(token);
        pending = vec![token];
    }

    let tokenizer = model.tokenizer();
    let text = tokenizer.decode(&generated_tokens)?;
    println!("Generated {} tokens in {:?}", generated_tokens.len(), start.elapsed());
    println!("Output:\n{}", text);

    Ok(String::new())
}
