//! Demonstrates parallel text generation from forked contexts.
//!
//! This example creates a shared system prompt context, then forks it into
//! two independent contexts that generate responses concurrently. Both
//! generations share the KV cache from the common prefix.

use futures::future;
use inferlet::{
    Context, sample::Sampler,
    Result,
};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_max_tokens() -> usize { 128 }

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_num_outputs = input.max_tokens;

    let start = Instant::now();

    let mut ctx = Context::new()?;
    ctx.system("You are a helpful, respectful and honest assistant.");
    ctx.flush().await?;

    let mut ctx1 = ctx.fork()?;
    let handle1 = async move {
        ctx1.user("Explain Pulmonary Embolism");
        ctx1.cue();

        let output = ctx1
            .generate(Sampler::Argmax)
            .max_tokens(max_num_outputs)
            .collect_text()
            .await;

        println!("Output 1: {:?} (elapsed: {:?})", output, start.elapsed());
    };

    let mut ctx2 = ctx.fork()?;
    let handle2 = async move {
        ctx2.user("Explain the Espresso making process ELI5.");
        ctx2.cue();

        let output = ctx2
            .generate(Sampler::Argmax)
            .max_tokens(max_num_outputs)
            .collect_text()
            .await;

        println!("Output 2: {:?} (elapsed: {:?})", output, start.elapsed());
    };

    future::join(handle1, handle2).await;

    Ok(String::new())
}
