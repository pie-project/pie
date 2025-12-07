//! Generates text using a simple sliding window for KV cache management.
//!
//! This method keeps only the most recent `window_size` tokens in the KV cache.
//! As new tokens are generated, the oldest tokens beyond the window size are
//! masked and eventually evicted from the cache. This is the simplest form of
//! windowed attention, suitable for tasks where only recent context is relevant.

use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Context, Result, Sampler};
use std::time::Instant;

pub async fn generate_with_sliding_window<C: StopCondition>(
    ctx: &mut Context,
    sampler: Sampler,
    stop_condition: C,
    window_size: usize,
) -> String {
    let mut generated_token_ids = Vec::new();

    // The autoregressive generation loop
    loop {
        // 1. Decode the next token, sample, and add it to the pending buffer.
        let next_token_id = ctx.decode_step(&sampler).await;
        ctx.fill_token(next_token_id);
        generated_token_ids.push(next_token_id);

        // 2. Check for the stop condition.
        if stop_condition.check(&generated_token_ids) {
            break;
        }

        // 3. Apply sliding window logic.
        let committed_len = ctx.token_ids.len();
        if committed_len > window_size {
            // Mask all tokens from the beginning that are now outside the window.
            let evict_end = committed_len - window_size;
            ctx.mask_token_range(1, evict_end, true);
            ctx.drop_masked_kv_pages();
        }
    }

    ctx.tokenizer.detokenize(&generated_token_ids)
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or("Explain LLM decoding process in ELI5.".to_string());
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(512);
    let window_size: usize = args.value_from_str(["-w", "--window-size"]).unwrap_or(32);

    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();

    let mut ctx = Context::new(&model);
    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    let sampler = Sampler::greedy();

    let stop_cond = stop_condition::max_len(max_num_outputs)
        .or(stop_condition::ends_with_any(model.eos_tokens()));

    println!("Starting generation with Windowed Attention (window_size={window_size})",);

    let output = generate_with_sliding_window(&mut ctx, sampler, stop_cond, window_size).await;

    let elapsed = start.elapsed();
    let output_token_ids = tokenizer.tokenize(&output);

    println!("\n--- Output ---\n{}\n--------------", output);

    println!(
        "Total elapsed: {:?}, Tokens generated: {}",
        elapsed,
        output_token_ids.len()
    );

    // compute per token latency
    if !output_token_ids.is_empty() {
        println!(
            "Per-token latency: {:?}",
            elapsed / output_token_ids.len() as u32
        );
    }

    Ok(())
}
