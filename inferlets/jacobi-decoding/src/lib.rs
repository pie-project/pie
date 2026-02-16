//! Demonstrates Jacobi decoding — parallel speculation via fixed-point iteration.
//!
//! Instead of autoregressive decoding, Jacobi decoding initializes N positions with
//! guessed tokens, then iteratively runs forward passes until the predictions converge
//! (reach a fixed point). This can verify multiple tokens per forward pass.
//!
//! All tokens (verified anchor + speculative guesses) are sent as regular `input_tokens`
//! and buffered beforehand so the runtime's fill() check passes. After each forward pass,
//! only the accepted prefix is committed to the KV cache.

use inferlet::{
    context::Context, model::Model, runtime,
    ContextExt, ForwardPassExt, InstructExt, Result,
    inference::{ForwardPass, Output, Sampler},
};
use std::time::Instant;

const HELP: &str = "\
Usage: jacobi-decoding [OPTIONS]

A program to demonstrate Jacobi decoding (parallel speculation via fixed-point iteration).

Options:
  -p, --prompt <PROMPT>      The prompt text [default: \"Write a poem about the ocean.\"]
  -n, --max-tokens <TOKENS>  Maximum number of tokens to generate [default: 256]
  -w, --window-size <N>      Number of speculative positions per iteration [default: 5]
  -h, --help                 Prints this help message";

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Write a poem about the ocean.".to_string());
    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);
    let window_size: usize = args.value_from_str(["-w", "--window-size"]).unwrap_or(5);

    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;
    let tokenizer = model.tokenizer();
    let stop_tokens = Context::stop_tokens(&model);

    let ctx = Context::create(&model, "jacobi", None)?;
    let page_size = ctx.tokens_per_page();

    ctx.system("You are a helpful assistant.");
    ctx.user(&prompt);
    ctx.cue();
    ctx.flush().await?;

    println!(
        "--- Jacobi Decoding (window_size={}, page_size={}) ---",
        window_size, page_size
    );

    let mut all_generated: Vec<u32> = Vec::new();
    let sampler = Sampler::TopP((0.0, 1.0));

    // Initialize: the anchor is the last buffered token after flush()
    let initial_buffered = ctx.buffered_tokens();
    if initial_buffered.is_empty() {
        return Err("No initial buffered tokens".to_string());
    }
    let anchor = initial_buffered[initial_buffered.len() - 1];

    // Speculative guesses initialized to copies of the anchor
    let mut window: Vec<u32> = vec![anchor; window_size];
    let mut total_accepted = 0;

    while total_accepted < max_tokens {
        let seq_len = ctx.last_position().map(|p| p + 1).unwrap_or(0);
        let cursor = ctx.cursor();

        // Build the full token list: [anchor] + [window guesses]
        // We need to make sure set_buffered_tokens includes all tokens
        // that we plan to send as input_tokens, because fill() checks:
        //   n <= tokens_buffered.len()

        // Set all tokens as buffered BEFORE the forward pass
        let mut buffered_all = vec![anchor];
        buffered_all.extend_from_slice(&window);
        ctx.set_buffered_tokens(&buffered_all);

        let input_count = buffered_all.len();

        // Reserve pages
        let total_tokens_after = cursor + input_count as u32;
        let total_pages_needed = (total_tokens_after + page_size - 1) / page_size;
        if total_pages_needed > 0 {
            ctx.reserve_pages(total_pages_needed)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
        }

        let pass = ForwardPass::new(&model);
        pass.context(&ctx);

        // All tokens go as regular input (so they're all embedded and processed)
        let positions: Vec<u32> = (seq_len..seq_len + input_count as u32).collect();
        pass.input_tokens(&buffered_all, &positions);

        // Request sampling at all positions
        let sample_indices: Vec<u32> = (0..input_count as u32).collect();
        pass.sampler(&sample_indices, sampler.clone());

        let output = pass.execute_async().await
            .map_err(|e| format!("Forward pass failed: {}", e))?;

        let predicted_tokens = match output {
            Output::Tokens(tokens) => tokens,
            _ => break,
        };

        if predicted_tokens.is_empty() {
            break;
        }

        // Jacobi verification: find the longest converged prefix.
        //
        // predicted_tokens[0] = prediction after anchor = the "next token"
        // predicted_tokens[i] = prediction after all_input_tokens[i]
        //
        // We accept predicted_tokens[0] unconditionally.
        // For i >= 1: if predicted_tokens[i-1] == window[i-1], it means
        // the model's prediction at position i-1 matches our guess at
        // position i, so position i was correctly speculated.

        let mut accepted_count = 1; // Always accept the first prediction
        for i in 1..predicted_tokens.len().min(window.len() + 1) {
            let i_window = i - 1;
            if i_window < window.len() && predicted_tokens[i - 1] == window[i_window] {
                accepted_count += 1;
            } else {
                break;
            }
        }

        let newly_accepted: Vec<u32> = predicted_tokens[..accepted_count.min(predicted_tokens.len())]
            .to_vec();

        // Check for stop tokens
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

        // Commit pages for accepted tokens only
        // The fill() already moved all input_count tokens to tokens_filled.
        // We commit pages covering anchor + accepted tokens.
        let commit_count = 1 + final_accepted.len(); // anchor + accepted predictions
        let new_cursor_abs = cursor + commit_count as u32;
        let pages_to_commit = new_cursor_abs / page_size;
        if pages_to_commit > 0 {
            let page_indices: Vec<u32> = (0..pages_to_commit).collect();
            ctx.commit_pages(&page_indices)
                .map_err(|e| format!("Failed to commit pages: {}", e))?;
        }
        ctx.set_cursor(new_cursor_abs % page_size);

        if stop_at < newly_accepted.len() || total_accepted >= max_tokens {
            break;
        }

        // Prepare for next iteration
        let last_accepted = *final_accepted.last().unwrap();

        // The last accepted token becomes the anchor for the next iteration
        ctx.set_buffered_tokens(&[last_accepted]);

        // Build next window of guesses
        window = if accepted_count < predicted_tokens.len() {
            let mut w: Vec<u32> = predicted_tokens[accepted_count..].to_vec();
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
