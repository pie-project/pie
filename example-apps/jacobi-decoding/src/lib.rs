//! Demonstrates Parallel Jacobi Decoding (PJD) for speculative generation.
//!
//! This example implements the PJD algorithm for speculative decoding. It
//! speculates multiple tokens in parallel, verifies them against the model's
//! actual predictions, accepts correct speculations, and refines incorrect
//! ones in subsequent iterations.

use inferlet::brle::Brle;
use inferlet::forward::Forward;
use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Context, Result};
use std::time::Instant;

const HELP: &str = "\
Usage: jacobi-decoding [OPTIONS]

Demonstrates Parallel Jacobi Decoding for speculative generation.

Options:
  -m, --max-tokens <N>          Maximum number of tokens to generate [default: 512]
  -g, --speculation-length <N>  Number of tokens to speculate in parallel [default: 5]
  -p, --prompt <PROMPT>         Prompt to generate
                                [default: Explain the LLM decoding process ELI5.]
  -h, --help                    Prints help information";

/// Generates text using Parallel Jacobi Decoding.
async fn generate_with_pjd<C: StopCondition>(
    ctx: &mut Context,
    stop_condition: &C,
    gamma: usize,
    unk_token_id: u32,
) -> (String, usize) {
    let mut all_generated_tokens = Vec::new();
    let mut num_steps = 0;

    // The initial batch tokens are the pending tokens followed by `gamma` unknown tokens.
    let mut batch_tokens = std::mem::take(&mut ctx.token_ids_pending);
    batch_tokens.extend(std::iter::repeat(unk_token_id).take(gamma));

    loop {
        // Check stop condition before generating more tokens.
        if stop_condition.check(&all_generated_tokens) {
            break;
        }

        // Example shape of the batch tokens when gamma = 3:
        // [truth_0, truth_1, spec_0, spec_1, spec_2]
        //
        // The truth tokens are the tokens that are input by the user at the beginning of
        // the generation or are the accepted tokens from the previous loop iteration.
        // The number of truth tokens is variable in each loop iteration.
        //
        // The speculative tokens are the ones that will be validated in the upcoming
        // forward pass. There will always be `gamma` speculative tokens.

        let batch_len = batch_tokens.len();

        // Calculate the positions for the tokens in the batch.
        let batch_positions = {
            let pos_offset = ctx.position_ids.last().map(|&p| p + 1).unwrap_or(0);
            (pos_offset..pos_offset + batch_len as u32).collect::<Vec<u32>>()
        };

        // Adjust the KV cache pages to accommodate the tokens in the batch.
        ctx.grow_kv_pages(batch_len);

        // The total length of the context for the next forward pass is the sum of the
        // number of tokens already in the KV cache and the number of tokens in the batch
        // waiting to be filled into the KV cache.
        let total_ctx_len = ctx.token_ids.len() + batch_len;

        // Each of the tokens in the batch will attend to the tokens before it in the context.
        let masks_for_batch = causal_mask(total_ctx_len as u32, batch_len as u32);

        // We sample at the position of the last truth token and the `gamma` speculative tokens.
        let sample_indices: Vec<u32> =
            ((batch_tokens.len() - gamma - 1) as u32..batch_tokens.len() as u32).collect();

        // Run a single forward pass to get sampled tokens for all `gamma + 1` positions.
        // We use temperature=0 for greedy sampling.
        let p = ctx.queue.create_forward_pass();
        p.input_tokens(&batch_tokens, &batch_positions);
        p.kv_cache(&ctx.kv_pages, ctx.kv_page_last_len);
        p.attention_mask(&masks_for_batch);
        p.output_tokens(&sample_indices, 0.0);
        let pass_result = p.execute().await;
        let sampled_tokens = pass_result.tokens.unwrap_or_default();

        // Create a slice over the speculative tokens that were fed into the forward pass.
        let speculated_tokens = &batch_tokens[batch_tokens.len() - gamma..];

        // Input batch:    [truth_0, truth_1, spec_0, spec_1, spec_2]
        //                              ↓       ↓       ↓       ↓
        // Sampled tokens: [N/A,      samp_0, samp_1, samp_2, samp_3]
        //
        // samp_0 is the ground truth next token in the generation process.
        // If the speculated token spec_0 is the same as the sampled token samp_0,
        // our speculation is correct, and the samp_1 token will also be correct.
        // Iteratively, if spec_1 is the same as samp_1, then samp_2 will also be correct,
        // and so on.
        //
        // The correct tokens go into the accepted tokens vector. Upon detecting the first
        // mismatch, all following sampled tokens go into the rejected tokens vector.
        let mut accepted_tokens = vec![sampled_tokens[0]];
        let mut rejected_tokens = vec![];
        for i in 0..gamma {
            if sampled_tokens[i] == speculated_tokens[i] {
                accepted_tokens.push(sampled_tokens[i + 1]);
            } else {
                rejected_tokens.extend_from_slice(&sampled_tokens[i + 1..]);
                break;
            }
        }

        // During the forward pass, all tokens in the batch are stored in the KV cache.
        // We need to shrink the KV cache to remove the wrong speculative tokens. The
        // number of wrong speculative tokens is the length of the rejected tokens vector.
        ctx.shrink_kv_pages(rejected_tokens.len());

        // Update the context's internal state. The token_ids and position_ids field reflect
        // the tokens that have already been stored in the KV cache. We extend these fields
        // with the truth tokens and then the accepted tokens in this loop iteration.
        ctx.token_ids
            .extend_from_slice(&batch_tokens[..batch_len - gamma]);
        ctx.token_ids
            .extend_from_slice(&accepted_tokens[..accepted_tokens.len() - 1]);
        ctx.position_ids
            .extend_from_slice(&batch_positions[..batch_len - gamma]);
        ctx.position_ids.extend_from_slice(
            &batch_positions[batch_len - gamma..batch_len - rejected_tokens.len()],
        );

        // Add the accepted tokens to the generated tokens vector.
        all_generated_tokens.extend_from_slice(&accepted_tokens);

        // The next batch will include the accepted tokens at the beginning.
        batch_tokens = accepted_tokens;

        // The next batch will also contain the sampled tokens from the previously
        // incorrect speculation. They are being "refined" in each loop iteration,
        // and hopefully in the next loop iteration, they will be correct. We pad
        // additional unknown tokens to the end of the batch to keep the speculation
        // length `gamma`.
        let add_unk_token_num = gamma - rejected_tokens.len();
        batch_tokens.extend(rejected_tokens.into_iter());
        batch_tokens.extend(std::iter::repeat(unk_token_id).take(add_unk_token_num));

        num_steps += 1;
    }

    // Clean up the context's internal state.
    ctx.token_ids_pending.clear();
    ctx.token_mask_pending.clear();
    ctx.token_mask_current = Brle::new(0);

    // Return the generated tokens and the number of steps taken.
    (ctx.tokenizer.detokenize(&all_generated_tokens), num_steps)
}

/// Creates a causal attention mask for the given context.
fn causal_mask(num_total_tokens: u32, num_input_tokens: u32) -> Vec<Vec<u32>> {
    let mut mask = Vec::new();
    let offset = num_total_tokens - num_input_tokens;
    for i in 0..num_input_tokens {
        mask.push(Brle::new((offset + i + 1) as usize).buffer);
    }
    mask
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let max_num_outputs: usize = args.value_from_str(["-m", "--max-tokens"]).unwrap_or(512);
    let speculation_length: usize = args
        .value_from_str(["-s", "--speculation-length"])
        .unwrap_or(5);
    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Explain the LLM decoding process ELI5.".to_string());

    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();

    // Use the first EOS token as the UNK token ID.
    let unk_token_id = eos_tokens[0][0];

    let mut ctx = model.create_context();
    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    let stop_condition =
        stop_condition::max_len(max_num_outputs).or(stop_condition::ends_with_any(eos_tokens));

    println!(
        "Starting generation with Parallel Jacobi Decoding (speculation length = {})...",
        speculation_length
    );

    let (output, num_steps) =
        generate_with_pjd(&mut ctx, &stop_condition, speculation_length, unk_token_id).await;

    let elapsed = start.elapsed();
    let output_token_ids = ctx.tokenizer.tokenize(&output);

    println!("\n--- Output ---\n{}\n--------------", output);

    println!(
        "Total elapsed: {:?}, Tokens generated: {}, Mean accepted tokens per step: {:.4}",
        elapsed,
        output_token_ids.len(),
        output_token_ids.len() as f64 / num_steps as f64
    );

    // Compute per-token latency
    if !output_token_ids.is_empty() {
        println!(
            "Per-token latency: {:?}",
            elapsed / output_token_ids.len() as u32
        );
    }

    Ok(())
}
