//! Simple text completion inferlet.
//!
//! Demonstrates chat-style generation with the explicit per-step Generator
//! loop and `chat::Decoder`.

use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::Deserialize;

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

fn default_max_tokens() -> usize {
    256
}
fn default_system() -> String {
    "You are a helpful, respectful and honest assistant.".into()
}
fn default_temperature() -> f32 {
    0.6
}
fn default_top_p() -> f32 {
    0.95
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user(&input.prompt).cue();

    let mut chat = chat::Decoder::new(&model);
    let mut text = String::new();

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

        match chat.feed(&out.tokens)? {
            chat::Event::Delta(s) => {
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

    Ok(text)
}
