//! Base prompt-to-completion inferlet for reasoning benchmark experiments.
//!
//! This inferlet intentionally does not implement Direct, Best-of-N, ToT, GoT,
//! answer extraction, scoring, or aggregation. It is the smallest reusable
//! surface that user-submitted reasoning inferlets can build around.

use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    thinking: bool,
}

fn default_system() -> String {
    "You are a helpful, careful assistant.".into()
}

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.95
}

#[derive(Serialize)]
struct Output {
    completion: String,
    stats: GenerationStats,
}

#[derive(Default, Serialize)]
struct GenerationStats {
    generated_tokens: usize,
    generator_steps: usize,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    validate(&input)?;

    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user(&input.prompt).cue();
    if !input.thinking {
        ctx.append(&model.tokenizer().encode("<think>\n\n</think>\n\n"));
    }

    let sampler = if input.temperature <= 0.0 {
        Sampler::Argmax
    } else {
        Sampler::TopP {
            temperature: input.temperature,
            p: input.top_p,
        }
    };
    let stops = chat::stop_tokens(&model);
    let mut generator = ctx
        .generate(sampler)
        .max_tokens(input.max_tokens)
        .stop(&stops);
    let mut decoder = chat::Decoder::new(&model);
    let mut completion = String::new();
    let mut stats = GenerationStats::default();

    while let Some(step) = generator.next()? {
        let output = step.execute().await?;
        stats.generated_tokens += output.tokens.len();
        stats.generator_steps += 1;
        match decoder.feed(&output.tokens)? {
            chat::Event::Delta(delta) => completion.push_str(&delta),
            chat::Event::Done(done) => {
                completion = done;
                break;
            }
            chat::Event::Idle | chat::Event::Interrupt(_) => {}
        }
    }

    Ok(Output { completion, stats })
}

fn validate(input: &Input) -> Result<()> {
    if input.prompt.trim().is_empty() {
        return Err("prompt must not be empty".into());
    }
    if input.max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }
    if !(input.temperature.is_finite() && (0.0..=2.0).contains(&input.temperature)) {
        return Err("temperature must be in [0.0, 2.0]".into());
    }
    if !(input.top_p.is_finite() && input.top_p > 0.0 && input.top_p <= 1.0) {
        return Err("top_p must be in (0.0, 1.0]".into());
    }
    Ok(())
}
