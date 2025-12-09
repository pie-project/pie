//! Demonstrates parallel text generation from forked contexts.
//!
//! This example creates a shared system prompt context, then forks it into
//! two independent contexts that generate responses concurrently. Both
//! generations share the KV cache from the common prefix.

use futures::future;
use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Result, Sampler};
use std::time::Instant;

const HELP: &str = "\
Usage: parallel-generation [OPTIONS]

A program to demonstrate parallel text generation from forked contexts.

Options:
  -n, --max-tokens <TOKENS>  Max tokens to generate for each prompt [default: 128]
  -h, --help                 Prints this help message";

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(128);

    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();
    let mut common = model.create_context();

    common.fill_system("You are a helpful, respectful and honest assistant.");
    common.flush().await;

    let mut ctx1 = common.fork();
    let eos_tokens1 = eos_tokens.clone();
    let handle1 = async move {
        ctx1.fill_user("Explain Pulmonary Embolism");

        let stop_condition =
            stop_condition::max_len(max_num_outputs).or(stop_condition::ends_with_any(eos_tokens1));
        let output = ctx1.generate(Sampler::greedy(), stop_condition).await;

        println!("Output 1: {:?} (elapsed: {:?})", output, start.elapsed());
    };

    let mut ctx2 = common.fork();
    let eos_tokens2 = eos_tokens.clone();
    let handle2 = async move {
        ctx2.fill_user("Explain the Espresso making process ELI5.");

        let stop_condition =
            stop_condition::max_len(max_num_outputs).or(stop_condition::ends_with_any(eos_tokens2));
        let output = ctx2.generate(Sampler::greedy(), stop_condition).await;

        println!("Output 2: {:?} (elapsed: {:?})", output, start.elapsed());
    };

    future::join(handle1, handle2).await;

    Ok(())
}
