//! Simple text completion example using chat-style prompting.
//! This example demonstrates basic text generation with a system prompt and
//! user message.

use inferlet::stop_condition::{StopCondition, ends_with_any, max_len};
use inferlet::{Args, Result, Sampler};
use std::time::Instant;

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);

    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();
    let mut ctx = model.create_context();

    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    let sampler = Sampler::top_p(0.6, 0.95);
    let stop_cond = max_len(max_num_outputs).or(ends_with_any(model.eos_tokens()));

    let final_text = ctx.generate(sampler, stop_cond).await;

    let token_ids = tokenizer.tokenize(&final_text);
    println!(
        "Output: {:?} (total elapsed: {:?})",
        final_text,
        start.elapsed()
    );

    // Compute per-token latency, avoiding division by zero.
    if !token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (token_ids.len() as u32)
        );
    }

    Ok(final_text)
}
