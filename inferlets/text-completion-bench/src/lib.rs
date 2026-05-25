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
    Context, FutureStringExt, Result, chat, model::Model, pie::core::session, runtime,
    sample::Sampler,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default)]
    prompts: Vec<String>,
    #[serde(default)]
    prompt_tokens_batch: Vec<Vec<u32>>,
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
    /// Decode and return generated text. The throughput benchmark only
    /// needs token counts unless it is dumping a sample.
    #[serde(default = "default_return_text")]
    return_text: bool,
    #[serde(default)]
    wait_for_start: bool,
    #[serde(default)]
    system_speculation: bool,
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
fn default_return_text() -> bool {
    true
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
    text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    request_prompt_tokens: Vec<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    request_output_tokens: Vec<usize>,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;
    let stop_tokens: Vec<u32> = if input.ignore_eos {
        Vec::new()
    } else {
        chat::stop_tokens(&model)
    };

    let batch_len = input.prompt_tokens_batch.len().max(input.prompts.len());
    if batch_len > 0 {
        let mut prepared_prompt_tokens: Vec<Vec<u32>> = Vec::new();
        if input.wait_for_start {
            prepared_prompt_tokens.reserve(batch_len);
            if input.prompt_tokens_batch.is_empty() {
                for i in 0..batch_len {
                    let prompt = input
                        .prompts
                        .get(i)
                        .map(String::as_str)
                        .unwrap_or(input.prompt.as_str());
                    let mut ctx = Context::new(&model)?;
                    ctx.system(&input.system).user(prompt).cue();
                    prepared_prompt_tokens.push(ctx.buffer().to_vec());
                }
            }
            session::send("ready");
            let _ = session::receive().wait_async().await;
        }
        let futures = (0..batch_len).map(|i| {
            let prompt = input
                .prompts
                .get(i)
                .map(String::as_str)
                .unwrap_or(input.prompt.as_str());
            let prompt_tokens = if !prepared_prompt_tokens.is_empty() {
                prepared_prompt_tokens.get(i).map(Vec::as_slice)
            } else {
                input.prompt_tokens_batch.get(i).map(Vec::as_slice)
            };
            run_one(&model, &input, prompt, prompt_tokens, &stop_tokens, false)
        });
        let mut request_prompt_tokens = Vec::with_capacity(batch_len);
        let mut request_output_tokens = Vec::with_capacity(batch_len);
        let mut first_tokens: Vec<u32> = Vec::new();
        for (i, result) in futures::future::join_all(futures)
            .await
            .into_iter()
            .enumerate()
        {
            let result = result?;
            if i == 0 {
                first_tokens = result.tokens.clone();
            }
            request_prompt_tokens.push(result.num_prompt_tokens);
            request_output_tokens.push(result.num_output_tokens);
        }
        let text = if input.return_text {
            model.tokenizer().decode(&first_tokens).unwrap_or_default()
        } else {
            String::new()
        };
        return Ok(Output {
            num_prompt_tokens: request_prompt_tokens.iter().sum(),
            num_output_tokens: request_output_tokens.iter().sum(),
            text,
            token_ids: first_tokens,
            request_prompt_tokens,
            request_output_tokens,
        });
    }

    let result = run_one(
        &model,
        &input,
        &input.prompt,
        input.prompt_tokens.as_deref(),
        &stop_tokens,
        true,
    )
    .await?;
    let text = if input.return_text {
        model.tokenizer().decode(&result.tokens).unwrap_or_default()
    } else {
        String::new()
    };

    Ok(Output {
        num_prompt_tokens: result.num_prompt_tokens,
        num_output_tokens: result.num_output_tokens,
        text,
        token_ids: result.tokens,
        request_prompt_tokens: Vec::new(),
        request_output_tokens: Vec::new(),
    })
}

struct RunResult {
    num_prompt_tokens: usize,
    num_output_tokens: usize,
    tokens: Vec<u32>,
}

async fn run_one(
    model: &Model,
    input: &Input,
    prompt: &str,
    prompt_tokens: Option<&[u32]>,
    stop_tokens: &[u32],
    honor_wait_for_start: bool,
) -> Result<RunResult> {
    let mut ctx = Context::new(model)?;
    let num_prompt_tokens = if let Some(tokens) = prompt_tokens {
        ctx.append(tokens);
        tokens.len()
    } else {
        ctx.system(&input.system).user(prompt).cue();
        ctx.buffer().len()
    };

    if honor_wait_for_start && input.wait_for_start {
        session::send("ready");
        let _ = session::receive().wait_async().await;
    }

    let mut all_output_tokens: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut g = ctx
        .generate(Sampler::TopP {
            temperature: input.temperature,
            p: input.top_p,
        })
        .rebid_each_step(false);
    if input.system_speculation && input.temperature <= 1e-5 {
        g = g.system_speculation();
    }
    let mut g = g.max_tokens(input.max_tokens).stop(&stop_tokens);

    if !input.return_text && stop_tokens.is_empty() && input.wasm_delay_us == 0 {
        let mut num_output_tokens = 0usize;
        while let Some(step) = g.next()? {
            let out = step.execute().await?;
            num_output_tokens += out.tokens.len();
        }
        return Ok(RunResult {
            num_prompt_tokens,
            num_output_tokens,
            tokens: Vec::new(),
        });
    }

    let wasm_delay = std::time::Duration::from_micros(input.wasm_delay_us);
    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }
        all_output_tokens.extend(out.tokens.iter().copied());
        // Sleep to simulate per-token inferlet WASM work. Yields the
        // CPU so chain-firing happening concurrently in the driver's
        // C++ thread can overlap. Skipped when wasm_delay_us == 0 (default).
        if input.wasm_delay_us > 0 {
            std::thread::sleep(wasm_delay);
        }
    }

    let num_output_tokens = all_output_tokens.len();
    Ok(RunResult {
        num_prompt_tokens,
        num_output_tokens,
        tokens: all_output_tokens,
    })
}
