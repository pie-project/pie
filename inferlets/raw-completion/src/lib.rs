//! Raw text-completion inferlet (no chat templating).
//!
//! Tokenizes the prompt directly and appends to the context, so a base
//! model sees the prompt verbatim instead of the chat-template wrap.
//! Useful for testing base/pretrained models where the chat template
//! mismatch causes degenerate looping.

use inferlet::{
    Context, Result,
    sample::Sampler,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    /// Raw prompt (NO chat template applied — model sees it verbatim).
    prompt: String,

    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    temperature: f32,

    /// Top-p (nucleus) sampling threshold.
    #[serde(default = "default_top_p")]
    top_p: f32,
}

fn default_max_tokens() -> usize { 64 }
fn default_temperature() -> f32 { 0.6 }
fn default_top_p() -> f32 { 0.95 }

#[derive(Serialize)]
struct Output {
    /// The generated continuation (decoded once at end).
    text: String,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let prompt_tokens = inferlet::model::encode(&input.prompt);

    let mut ctx = Context::new()?;
    ctx.append(&prompt_tokens);

    let mut all_generated: Vec<u32> = Vec::with_capacity(input.max_tokens);

    let mut g = ctx
        .generate(Sampler::TopP {
            temperature: input.temperature,
            p: input.top_p,
        })
        .max_tokens(input.max_tokens);

    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        for &t in &out.tokens {
            all_generated.push(t);
        }
        if all_generated.len() >= input.max_tokens {
            break;
        }
    }

    let text = inferlet::model::decode(&all_generated)
        .unwrap_or_else(|_| String::from("[decode error]"));
    print!("{}", text);

    Ok(Output { text })
}
