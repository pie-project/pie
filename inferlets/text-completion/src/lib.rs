//! Simple text completion inferlet.
//!
//! Demonstrates chat-style generation with the explicit per-step Generator
//! loop, fanning each token batch through both `chat::Decoder` (for the
//! visible response) and `reasoning::Decoder` (for the thinking trace).
//! This composes the two decoders by hand rather than leaning on a
//! framework-provided unified event stream.

use inferlet::{
    Context, Result,
    chat, reasoning,
    model::Model,
    runtime,
    sample::Sampler,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    /// The user prompt to complete.
    prompt: String,

    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    /// System message for the assistant.
    #[serde(default = "default_system")]
    system: String,

    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    temperature: f32,

    /// Top-p (nucleus) sampling threshold.
    #[serde(default = "default_top_p")]
    top_p: f32,
}

fn default_max_tokens() -> usize { 256 }
fn default_system() -> String { "You are a helpful, respectful and honest assistant.".into() }
fn default_temperature() -> f32 { 0.6 }
fn default_top_p() -> f32 { 0.95 }

#[derive(Serialize)]
struct Output {
    /// The thinking/reasoning trace.
    thinking: String,
    /// The generated text.
    text: String,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user(&input.prompt).cue();

    let mut think = reasoning::Decoder::new(&model);
    let mut chat = chat::Decoder::new(&model);
    let mut thinking = String::new();
    let mut text = String::new();
    let mut in_reasoning = false;

    let mut g = ctx
        .generate(Sampler::TopP {
            temperature: input.temperature,
            p: input.top_p,
        })
        .max_tokens(input.max_tokens)
        .stop(&chat::stop_tokens(&model));

    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }

        // Reasoning side: Start / Delta / End. Idle is the no-op signal
        // (tokens outside any reasoning block, or empty visible chunks).
        match think.feed(&out.tokens)? {
            reasoning::Event::Start => in_reasoning = true,
            reasoning::Event::Delta(s) => {
                eprint!("{}", s);
                thinking.push_str(&s);
            }
            reasoning::Event::End(_) => {
                eprintln!();
                in_reasoning = false;
            }
            reasoning::Event::Idle => {}
        }

        // Chat side: Delta is visible text. The `in_reasoning` guard is
        // a defensive filter — the host should already exclude reasoning
        // tokens from chat::Delta, but we keep it until the contract is
        // confirmed.
        match chat.feed(&out.tokens)? {
            chat::Event::Delta(s) if !in_reasoning => {
                print!("{}", s);
                text.push_str(&s);
            }
            chat::Event::Done(s) => {
                text = s;
                break;
            }
            _ => {}
        }
    }

    Ok(Output { thinking, text })
}
