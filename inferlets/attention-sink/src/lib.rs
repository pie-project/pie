//! Demonstrates attention sink — bounded KV cache with preserved initial tokens.
//!
//! Maintains an "attention sink" of initial tokens plus a sliding window of
//! the most recent tokens. Tokens between the sink and the window are
//! masked via a per-step attention mask, preventing the model from
//! attending to them.
//!
//! NOTE: full KV cache eviction is not yet supported by the runtime — the
//! mask only prevents the model from *attending* to masked tokens; the KV
//! pages stay in memory.

use inferlet::{
    Context, Result,
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
    #[serde(default = "default_sink_size")]
    sink_size: u32,
    #[serde(default = "default_window_size")]
    window_size: u32,
}

fn default_prompt() -> String { "Tell me a long story about a cat.".to_string() }
fn default_max_tokens() -> usize { 512 }
fn default_sink_size() -> u32 { 4 }
fn default_window_size() -> u32 { 64 }

/// BRLE attention mask for `[sink True, gap False, window True]` over
/// `seq_len` positions. Returns all-true when the sequence fits inside
/// `sink + window`.
fn build_sink_mask(seq_len: u32, sink: u32, window: u32) -> Vec<u32> {
    let total_kept = sink + window;
    if seq_len <= total_kept {
        vec![0, seq_len]
    } else {
        let gap = seq_len - total_kept;
        vec![0, sink, gap, window]
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let start = Instant::now();

    let mut ctx = Context::new()?;
    let page_size = ctx.page_size();
    let stop_tokens = inferlet::chat::stop_tokens();

    // Build the prompt and prefill in one Pass.
    let mut prompt: Vec<u32> = Vec::new();
    prompt.extend(inferlet::chat::system("You are a helpful assistant."));
    prompt.extend(inferlet::chat::user(&input.prompt));
    prompt.extend(inferlet::chat::cue());

    println!(
        "--- Attention Sink (sink={}, window={}, page_size={}) ---",
        input.sink_size, input.window_size, page_size
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

        if total_seq_after > input.sink_size + input.window_size {
            let mask = build_sink_mask(total_seq_after, input.sink_size, input.window_size);
            // One mask per query position (each token in `pending` is a query).
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

    let text = inferlet::model::decode(&generated_tokens)?;
    println!("Generated {} tokens in {:?}", generated_tokens.len(), start.elapsed());
    println!("Output:\n{}", text);

    Ok(String::new())
}
