//! Demonstrates beam search decoding for text generation.
//!
//! Beam search is a decoding strategy that maintains multiple candidate sequences
//! (beams) at each step, selecting the most likely overall sequences rather than
//! greedily choosing the best token at each position.

use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Result, anyhow};
use std::time::Instant;

const HELP: &str = "\
Usage: beam-search [OPTIONS]

A program to run text generation with beam search decoding.

Options:
  -p, --prompt <STRING>    The prompt to send to the model
                           [default: Explain the LLM decoding process ELI5.]
  -n, --max-tokens <INT>   The maximum number of new tokens to generate [default: 128]
  -b, --beam-size <INT>    The beam size for decoding [default: 1]
  -h, --help               Print help information";

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(128);
    let beam_size: usize = args.value_from_str(["-b", "--beam-size"]).unwrap_or(1);
    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Explain the LLM decoding process ELI5.".to_string());

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

    let mut stop_condition =
        stop_condition::max_len(max_num_outputs).or(stop_condition::ends_with_any(eos_tokens));

    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    let text = ctx.generate_with_beam(&mut stop_condition, beam_size).await;
    let token_ids = tokenizer.tokenize(&text);
    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());

    if !token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (token_ids.len() as u32)
        );
    }

    Ok(())
}
