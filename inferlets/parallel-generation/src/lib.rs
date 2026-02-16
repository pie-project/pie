//! Demonstrates parallel text generation from forked contexts.
//!
//! This example creates a shared system prompt context, then forks it into
//! two independent contexts that generate responses concurrently. Both
//! generations share the KV cache from the common prefix.

use futures::future;
use inferlet::{
    context::Context, inference::Sampler, model::Model,
    runtime, ContextExt, InstructExt, Result,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

const HELP: &str = "\
Usage: parallel-generation [OPTIONS]

A program to demonstrate parallel text generation from forked contexts.

Options:
  -n, --max-tokens <TOKENS>  Max tokens to generate for each prompt [default: 128]
  -h, --help                 Prints this help message";

static FORK_COUNTER: AtomicU32 = AtomicU32::new(0);

fn next_fork_name() -> String {
    format!("fork-{}", FORK_COUNTER.fetch_add(1, Ordering::Relaxed))
}

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(128);

    let start = Instant::now();

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let ctx = Context::new(&model)?;
    ctx.system("You are a helpful, respectful and honest assistant.");
    ctx.flush().await?;

    let ctx1 = ctx.fork(&next_fork_name())?;
    let handle1 = async move {
        ctx1.user("Explain Pulmonary Embolism");
        ctx1.cue();

        let output = ctx1
            .generate(Sampler::TopP((0.0, 1.0)))
            .with_max_tokens(max_num_outputs)
            .collect_text()
            .await;

        println!("Output 1: {:?} (elapsed: {:?})", output, start.elapsed());
    };

    let ctx2 = ctx.fork(&next_fork_name())?;
    let handle2 = async move {
        ctx2.user("Explain the Espresso making process ELI5.");
        ctx2.cue();

        let output = ctx2
            .generate(Sampler::TopP((0.0, 1.0)))
            .with_max_tokens(max_num_outputs)
            .collect_text()
            .await;

        println!("Output 2: {:?} (elapsed: {:?})", output, start.elapsed());
    };

    future::join(handle1, handle2).await;

    Ok(String::new())
}
