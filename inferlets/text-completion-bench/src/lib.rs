//! Text completion inferlet for benchmarking.
//!
//! Sibling of `text-completion`. The differences are:
//!   * No per-token streaming via stdout — output is delivered only via
//!     the final `Return` event. The harness counted tokens twice
//!     before because the streamed text was *also* present in the
//!     final JSON envelope.
//!   * Returns `num_prompt_tokens` and `num_output_tokens` directly from
//!     the model's tokenizer / generator — no harness-side
//!     re-tokenisation, which is the only way to match what vllm/sglang
//!     report as `Total Out Tokens`.
//!   * `ignore_eos` lets the harness force every request to consume the
//!     full `max_tokens` budget, so cross-engine comparisons aren't
//!     biased by chat-template stop-token handling.
//!
//! Sampler matches `text-completion` (TopP, default temperature 0.6 /
//! top_p 0.95) so workload shape is otherwise identical.

use inferlet::{
    Context, Result,
    chat,
    model::Model,
    runtime,
    sample::Sampler,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    /// When true, drop the chat-template stop tokens so the generator
    /// runs to `max_tokens` regardless of model emit. Lets the bench
    /// guarantee identical token counts across runs / engines.
    #[serde(default)]
    ignore_eos: bool,
}

fn default_max_tokens() -> usize { 256 }
fn default_system() -> String { "You are a helpful, respectful and honest assistant.".into() }
fn default_temperature() -> f32 { 0.6 }
fn default_top_p() -> f32 { 0.95 }

#[derive(Serialize)]
struct Output {
    /// Number of tokens in the chat-templated prompt that fed the
    /// prefill. Lets the harness compute prefill throughput separately.
    num_prompt_tokens: usize,
    /// Number of tokens the sampler actually emitted. Authoritative —
    /// not derived from char/4 nor from re-tokenisation of the text.
    num_output_tokens: usize,
    /// Decoded text — for spot-checking output quality.
    text: String,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user(&input.prompt).cue();

    // Snapshot the prompt token count (chat-templated prefill length)
    // before the generation loop mutates the context. `seq_len()`
    // returns the number of tokens currently committed to the
    // context — at this point that's exactly the chat-templated prompt.
    let num_prompt_tokens = ctx.seq_len() as usize;

    let stop_tokens: Vec<u32> = if input.ignore_eos {
        Vec::new()
    } else {
        chat::stop_tokens(&model)
    };

    let mut all_output_tokens: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut g = ctx
        .generate(Sampler::TopP {
            temperature: input.temperature,
            p: input.top_p,
        })
        .max_tokens(input.max_tokens)
        .stop(&stop_tokens);

    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }
        all_output_tokens.extend(out.tokens.iter().copied());
    }

    let num_output_tokens = all_output_tokens.len();
    let text = model
        .tokenizer()
        .decode(&all_output_tokens)
        .unwrap_or_default();

    Ok(Output {
        num_prompt_tokens,
        num_output_tokens,
        text,
    })
}
