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

use inferlet::inference::ForwardPass;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, pie::core::session, sampler, Result};
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
    system_speculation: Option<bool>,
    #[serde(default)]
    batch_concurrency: Option<usize>,
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
    let stop_tokens: Vec<u32> = if input.ignore_eos {
        Vec::new()
    } else {
        chat::stop_tokens()
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
                    let mut pt = chat::system_user(&input.system, prompt);
                    pt.extend(chat::cue());
                    prepared_prompt_tokens.push(pt);
                }
            }
            session::send("ready");
            let _ = session::receive().await;
        }
        let mut request_prompt_tokens = Vec::with_capacity(batch_len);
        let mut request_output_tokens = Vec::with_capacity(batch_len);
        let mut first_tokens: Vec<u32> = Vec::new();
        let batch_concurrency = input
            .batch_concurrency
            .unwrap_or(batch_len)
            .clamp(1, batch_len);
        let mut offset = 0usize;
        while offset < batch_len {
            let end = (offset + batch_concurrency).min(batch_len);
            let futures = (offset..end).map(|i| {
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
                run_one(&input, prompt, prompt_tokens, &stop_tokens, false)
            });
            for (j, result) in futures::future::join_all(futures)
                .await
                .into_iter()
                .enumerate()
            {
                let result = result?;
                if offset + j == 0 {
                    first_tokens = result.tokens.clone();
                }
                request_prompt_tokens.push(result.num_prompt_tokens);
                request_output_tokens.push(result.num_output_tokens);
            }
            offset = end;
        }
        let text = if input.return_text {
            inferlet::model::decode(&first_tokens).unwrap_or_default()
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
        &input,
        &input.prompt,
        input.prompt_tokens.as_deref(),
        &stop_tokens,
        true,
    )
    .await?;
    let text = if input.return_text {
        inferlet::model::decode(&result.tokens).unwrap_or_default()
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

/// One sequential decode fire over `tokens` at the cursor; returns the sampled
/// token. (`system_speculation` was a shipped no-op — no `output-speculative`
/// emit — so every step is a single token; the raw loop preserves that.)
async fn bench_fire(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    s: &sampler::LoweredSampler,
    tokens: &[u32],
) -> Result<u32> {
    let n = tokens.len() as u32;
    let pass = ForwardPass::new();
    if *fresh {
        pass.fresh_generate();
        *fresh = false;
    }
    let geom = geometry::ensure_pages(kv, geometry::kv_write_geometry(*seq_len, n, kv.page_size()))?;
    geometry::attach_kv_write(&pass, kv, &geom);
    let positions: Vec<u32> = (*seq_len..*seq_len + n).collect();
    pass.input_tokens(tokens, &positions);
    pass.sampler(&s.program, s.bindings(*seq_len + n - 1)?);
    pass.execute();
    *seq_len += n;
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("read: {e:?}"))?;
    if bytes.len() < 4 {
        return Err("empty token output".into());
    }
    Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32)
}

async fn run_one(
    input: &Input,
    prompt: &str,
    prompt_tokens: Option<&[u32]>,
    stop_tokens: &[u32],
    honor_wait_for_start: bool,
) -> Result<RunResult> {
    // Prompt: explicit pre-tokenized tokens (the batch path) or chat-template
    // system+user+cue — the raw token vec that seeds the first fire.
    let prompt_vec: Vec<u32> = if let Some(tokens) = prompt_tokens {
        tokens.to_vec()
    } else {
        let mut p = chat::system_user(&input.system, prompt);
        p.extend(chat::cue());
        p
    };
    let num_prompt_tokens = prompt_vec.len();

    if honor_wait_for_start && input.wait_for_start {
        session::send("ready");
        let _ = session::receive().await;
    }

    let vocab = model::output_vocab_size();
    let spec = if input.temperature <= 0.0 {
        sampler::SamplerSpec::Argmax
    } else {
        sampler::SamplerSpec::TopP {
            temperature: input.temperature,
            p: input.top_p,
        }
    };
    let s = sampler::sampler_program(spec, vocab)?;

    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    // Sequential raw decode: first fire over the whole prompt, then one token per
    // fire off the last token, until a stop token or `max_tokens`. `.stop()` +
    // `system_speculation` no-op behavior preserved (stop excluded from output).
    let count_only = !input.return_text && stop_tokens.is_empty() && input.wasm_delay_us == 0;
    let wasm_delay = std::time::Duration::from_micros(input.wasm_delay_us);
    let mut all_output_tokens: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut num_output_tokens = 0usize;

    let mut pending: Vec<u32> = prompt_vec;
    let mut generated = 0usize;
    while generated < input.max_tokens {
        let tok = bench_fire(&kv, &mut seq_len, &mut fresh, &s, &pending).await?;
        generated += 1;
        if stop_tokens.contains(&tok) {
            break;
        }
        num_output_tokens += 1;
        if !count_only {
            all_output_tokens.push(tok);
            // Simulate per-token WASM work; yields so driver-side chain-firing
            // can overlap. Skipped when wasm_delay_us == 0 (default).
            if input.wasm_delay_us > 0 {
                std::thread::sleep(wasm_delay);
            }
        }
        pending = vec![tok];
    }

    if count_only {
        return Ok(RunResult {
            num_prompt_tokens,
            num_output_tokens,
            tokens: Vec::new(),
        });
    }

    Ok(RunResult {
        num_prompt_tokens,
        num_output_tokens: all_output_tokens.len(),
        tokens: all_output_tokens,
    })
}
