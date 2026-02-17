//! Demonstrates windowed attention — a sliding window approach to KV cache management.
//!
//! After filling the context, this inferlet applies a sliding window attention mask
//! during generation to limit the model's attention to the most recent `window_size`
//! tokens. This simulates bounded-memory generation.
//!
//! NOTE: Full KV cache eviction is not yet supported by the runtime. The runtime's
//! `release_pages` API only frees *uncommitted* pages; committed pages persist.
//! This implementation uses `ForwardPass::attention_mask` to apply per-step masking,
//! so the model ignores tokens outside the window, but the KV memory is not freed.

use inferlet::{
    context::Context, model::Model, runtime,
    ContextExt, ForwardPassExt, InstructExt, Result,
    inference::{ForwardPass, Output, Sampler},
};
use std::time::Instant;

const HELP: &str = "\
Usage: windowed-attention [OPTIONS]

A program to demonstrate windowed attention with KV cache management.

Options:
  -p, --prompt <PROMPT>      The prompt text [default: \"Tell me a long story about a cat.\"]
  -n, --max-tokens <TOKENS>  Maximum number of tokens to generate [default: 512]
  -w, --window-size <SIZE>   Window size in tokens [default: 64]
  -h, --help                 Prints this help message";

/// Build a BRLE attention mask for windowed attention.
///
/// The mask has shape [seq_len] where 1 = attend, 0 = masked.
/// With windowed attention, we attend to all tokens from position
/// `max(0, seq_len - window_size)` onward.
///
/// BRLE (binary run-length encoding):
///   [count_of_0s, count_of_1s, count_of_0s, count_of_1s, ...]
fn build_window_mask(seq_len: u32, window_size: u32) -> Vec<u32> {
    if seq_len <= window_size {
        // Entire sequence fits in window, attend to everything: BRLE = [0, seq_len]
        vec![0, seq_len]
    } else {
        // Mask the oldest tokens, attend to the window
        let masked = seq_len - window_size;
        // BRLE: [masked_count, window_count]
        vec![masked, window_size]
    }
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
        .unwrap_or_else(|_| "Tell me a long story about a cat.".to_string());
    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(512);
    let window_size: u32 = args.value_from_str(["-w", "--window-size"]).unwrap_or(64);

    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    let ctx = Context::create(&model)?;
    let page_size = ctx.tokens_per_page();
    let stop_tokens = Context::stop_tokens(&model);

    // Fill the prompt
    ctx.system("You are a helpful assistant.");
    ctx.user(&prompt);
    ctx.cue();
    ctx.flush().await?;

    println!(
        "--- Windowed Attention (window={} tokens, page_size={}) ---",
        window_size, page_size
    );

    // Manual generation loop with per-step attention masking.
    // This follows the same page management pattern as the SDK's
    // TokenStream::step(), but adds a windowed attention_mask on
    // each ForwardPass.
    let mut generated_tokens: Vec<u32> = Vec::new();
    let sampler = Sampler::TopP((0.0, 1.0));

    for _step in 0..max_tokens {
        let buffered = ctx.buffered_tokens();
        if buffered.is_empty() {
            break;
        }

        let seq_len = ctx.last_position().map(|p| p + 1).unwrap_or(0);
        let cursor = ctx.cursor();

        // Reserve pages for the new token(s)
        let total_tokens_after = cursor + buffered.len() as u32;
        let total_pages_needed = (total_tokens_after + page_size - 1) / page_size;
        if total_pages_needed > 0 {
            ctx.reserve_pages(total_pages_needed)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
        }

        let pass = ForwardPass::new(&model);
        pass.context(&ctx);

        let positions: Vec<u32> = (seq_len..seq_len + buffered.len() as u32).collect();
        pass.input_tokens(&buffered, &positions);

        // Apply windowed attention mask
        let total_seq_len = seq_len + buffered.len() as u32;
        if total_seq_len > window_size {
            let mask = build_window_mask(total_seq_len, window_size);
            // The mask is per query position; we need one row per input token
            let masks: Vec<Vec<u32>> = (0..buffered.len())
                .map(|_| mask.clone())
                .collect();
            pass.attention_mask(&masks);
        }

        let last_token_idx = (buffered.len() - 1) as u32;
        pass.sampler(&[last_token_idx], sampler.clone());

        let output = pass.execute_async().await?;

        let new_tokens = match output {
            Output::Tokens(tokens) => tokens,
            _ => break,
        };

        if new_tokens.is_empty() {
            break;
        }

        // Page management: commit pages and update cursor
        let new_cursor = cursor + buffered.len() as u32;
        let pages_to_commit = new_cursor / page_size;

        if pages_to_commit > 0 {
            let page_indices: Vec<u32> = (0..pages_to_commit).collect();
            ctx.commit_pages(&page_indices)
                .map_err(|e| format!("Failed to commit pages: {}", e))?;
        }

        ctx.set_cursor(new_cursor % page_size);

        // Check for stop tokens
        let token = new_tokens[0];
        if stop_tokens.contains(&token) {
            break;
        }

        generated_tokens.push(token);

        // Buffer the accepted token for the next step
        ctx.set_buffered_tokens(&[token]);
    }

    let tokenizer = model.tokenizer();
    let text = tokenizer.decode(&generated_tokens)?;
    println!("Generated {} tokens in {:?}", generated_tokens.len(), start.elapsed());
    println!("Output:\n{}", text);

    Ok(String::new())
}
