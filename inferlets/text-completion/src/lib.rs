//! Simple text completion inferlet.
//!
//! Demonstrates chat-style generation using typed JSON input/output
//! with the `InstructExt` + `EventStream` high-level API.

use inferlet::{
    context::Context,
    inference::Sampler,
    model::Model,
    runtime,
    ContextExt, Event, InstructExt,
    Result,
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
    /// The generated text.
    text: String,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    // Load model
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    // Create context and fill with instruct messages
    let ctx = Context::new(&model)?;
    ctx.system(&input.system);
    ctx.user(&input.prompt);
    ctx.cue();

    // Generate
    let mut events = ctx
        .generate(Sampler::TopP((input.temperature, input.top_p)))
        .with_max_tokens(input.max_tokens)
        .decode()
        .with_reasoning();

    let mut output = String::new();

    while let Some(event) = events.next().await? {
        match event {
            Event::Thinking(s) => {
                print!("{}", s);
            }
            Event::ThinkingDone(_) => {
            }
            Event::Text(s) => {
                print!("{}", s);
            }
            Event::Done(s) => {
                output = s;
                break;
            }
            _ => {}
        }
    }

    Ok(Output { text: output })
}
