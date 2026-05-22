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

use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
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
    /// Busy-spin inside WASM for this many microseconds between
    /// every `step.execute()` call. Simulates per-token inferlet
    /// work — agent reasoning, tool-call deserialization, etc. —
    /// without needing to wire a real workload. Used by
    /// SPECULATIVE_EXECUTION_DESIGN.md phase B4b.3 to measure
    /// chain-firing overlap with WASM time.
    #[serde(default)]
    wasm_delay_us: u64,
    /// Decode and return full output text. The benchmark hot path only
    /// needs token counts; text is reserved for explicit correctness dumps.
    #[serde(default)]
    return_text: bool,
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

#[derive(Serialize)]
struct Output {
    /// Number of tokens in the chat-templated prompt that fed the
    /// prefill. Lets the harness compute prefill throughput separately.
    num_prompt_tokens: usize,
    /// Number of tokens the sampler actually emitted. Authoritative —
    /// not derived from char/4 nor from re-tokenisation of the text.
    num_output_tokens: usize,
    /// Decoded text — for spot-checking output quality.
    #[serde(skip_serializing_if = "String::is_empty")]
    text: String,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;
    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user(&input.prompt).cue();

    // Snapshot the prompt token count BEFORE the generation loop
    // fires its first forward pass. `seq_len()` only counts
    // tokens already pushed through a forward — the chat fillers
    // above only buffer locally — so the right number lives in
    // `buffer()` at this point.
    let num_prompt_tokens = ctx.buffer().len();

    let stop_tokens: Vec<u32> = if input.ignore_eos {
        Vec::new()
    } else {
        chat::stop_tokens(&model)
    };

    let mut output_tokens: Vec<u32> = if input.return_text {
        Vec::with_capacity(input.max_tokens)
    } else {
        Vec::new()
    };
    let mut num_output_tokens = 0usize;
    let mut g = ctx
        .generate(Sampler::TopP {
            temperature: input.temperature,
            p: input.top_p,
        })
        .max_tokens(input.max_tokens)
        .stop(&stop_tokens);

    let wasm_delay = std::time::Duration::from_micros(input.wasm_delay_us);
    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }
        num_output_tokens += out.tokens.len();
        if input.return_text {
            output_tokens.extend(out.tokens.iter().copied());
        }
        // Sleep to simulate per-token inferlet WASM work. Yields the
        // CPU so chain-firing happening concurrently in the driver's
        // C++ thread can overlap. Skipped when wasm_delay_us == 0 (default).
        if input.wasm_delay_us > 0 {
            std::thread::sleep(wasm_delay);
        }
    }

    let text = if input.return_text {
        model.tokenizer().decode(&output_tokens).unwrap_or_default()
    } else {
        String::new()
    };

    Ok(Output {
        num_prompt_tokens,
        num_output_tokens,
        text,
    })
}
