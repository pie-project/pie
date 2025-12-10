//! Demonstrates attention sink for managing long sequence generation.
//!
//! This example implements a sliding window attention mechanism (attention sink)
//! to keep the KV cache size bounded during generation. It maintains an initial
//! "sink" of tokens plus a sliding window of recent tokens, masking out tokens
//! in between to free memory.

use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Context, Result, Sampler, anyhow};
use std::time::Instant;

const HELP: &str = "\
Usage: attention-sink [OPTIONS]

A program to run text generation with a configurable attention sink.

Options:
  -p, --prompt <STRING>        The prompt to send to the model
                               [default: Explain LLM decoding process in ELI5.]
  -n, --max-tokens <INT>       The maximum number of new tokens to generate [default: 512]
  -s, --sink-size <INT>        The initial size of the attention sink [default: 64]
  -w, --sink-window <INT>      The sliding window size for the attention sink [default: 32]
  -h, --help                   Print help information";

/// Generates text autoregressively using an attention sink to manage a rolling KV cache.
///
/// This function implements a sliding window attention mechanism to keep the KV cache
/// size bounded. It maintains an initial "sink" of tokens plus a sliding window of
/// recent tokens, masking out tokens in between.
pub async fn generate_with_attention_sink<C: StopCondition>(
    ctx: &mut Context,
    sampler: &Sampler,
    stop_condition: &C,
    attention_sink_initial_size: usize,
    attention_sink_window_size: usize,
) -> String {
    let mut generated_token_ids = Vec::new();
    let max_cache_size = attention_sink_initial_size + attention_sink_window_size;

    loop {
        // 1. Decode the next token
        let next_token_id = ctx.decode_step(sampler).await;
        ctx.fill_token(next_token_id);
        generated_token_ids.push(next_token_id);

        // 2. Check if the stop condition is met
        if stop_condition.check(&generated_token_ids) {
            break;
        }

        // 3. Apply attention sink logic
        let committed_len = ctx.token_ids.len();

        if committed_len > max_cache_size {
            // Determine the range of tokens to "evict" from the attention window.
            let num_to_evict = committed_len - max_cache_size;
            let evict_start = attention_sink_initial_size;
            let evict_end = attention_sink_initial_size + num_to_evict;

            // Mask this range so the model ignores it in future forward passes.
            ctx.mask_token_range(evict_start, evict_end, true);

            // Drop any full KV cache pages that are now fully masked.
            ctx.drop_masked_kv_pages();
        }
    }

    ctx.tokenizer.detokenize(&generated_token_ids)
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(512);
    let attention_sink_initial_size: usize =
        args.value_from_str(["-s", "--sink-size"]).unwrap_or(64);
    let attention_sink_window_size: usize =
        args.value_from_str(["-w", "--sink-window"]).unwrap_or(32);
    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Explain LLM decoding process in ELI5.".to_string());

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
    let eos_tokens = model.eos_tokens();

    let mut ctx = model.create_context();

    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    let sampler = Sampler::greedy();
    let stop_condition =
        stop_condition::max_len(max_num_outputs).or(stop_condition::ends_with_any(eos_tokens));

    let output = generate_with_attention_sink(
        &mut ctx,
        &sampler,
        &stop_condition,
        attention_sink_initial_size,
        attention_sink_window_size,
    )
    .await;

    let elapsed = start.elapsed();
    let output_token_ids = tokenizer.tokenize(&output);

    println!("\n--- Output ---\n{}\n--------------", output);
    println!(
        "Total elapsed: {:?}, Tokens generated: {}",
        elapsed,
        output_token_ids.len()
    );

    if !output_token_ids.is_empty() {
        println!(
            "Per-token latency: {:?}",
            elapsed / output_token_ids.len() as u32
        );
    }

    Ok(())
}
