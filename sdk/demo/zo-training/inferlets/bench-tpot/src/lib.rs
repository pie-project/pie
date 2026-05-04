//! Single-decode TPOT/TTFT benchmark with optional adapter.
//!
//! Sends one chat-style prompt, generates `max_tokens` tokens via the
//! same per-step API the runtime uses (`TokenStream::next()`), and
//! records wall time per step. Reports prefill (TTFT) and steady-state
//! decode TPOT, optionally toggling an adapter on or off.

use inferlet::{
    Context,
    adapter::Adapter,
    inference::Sampler,
    model::Model,
    runtime,
    Result,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_sys")]
    system_prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_warmup")]
    warmup_tokens: usize,
    /// Empty string = no adapter.
    #[serde(default)]
    adapter_name: String,
    #[serde(default)]
    zo_seed: i64,
}

fn default_sys() -> String { "You are a helpful assistant.".to_string() }
fn default_max_tokens() -> usize { 64 }
fn default_warmup() -> usize { 4 }

#[derive(Serialize)]
struct Output {
    tokens_total: usize,
    ttft_us: u128,
    decode_total_us: u128,
    decode_steps: usize,
    tpot_mean_us: f64,
    tpot_p50_us: u128,
    tpot_p90_us: u128,
    tpot_min_us: u128,
    tpot_max_us: u128,
    per_step_us: Vec<u128>,
    adapter_active: bool,
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model_name = runtime::models().into_iter().next()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system_prompt);
    ctx.user(&input.prompt);
    ctx.cue();

    let adapter_active = !input.adapter_name.is_empty();
    let adapter = if adapter_active {
        Some(Adapter::open(&model, &input.adapter_name)
            .ok_or_else(|| format!("Adapter '{}' not found", input.adapter_name))?)
    } else {
        None
    };

    let sampler = Sampler::TopP((0.6, 0.95));
    let mut stream = ctx.generate(sampler).with_max_tokens(input.max_tokens);
    if let Some(ref a) = adapter {
        stream = stream.with_adapter(a).with_zo_seed(input.zo_seed);
    }

    // First call to next() includes prefill — that's the TTFT.
    let mut per_step_us: Vec<u128> = Vec::with_capacity(input.max_tokens);
    let t_ttft = Instant::now();
    let first = stream.next().await?;
    let ttft_us = t_ttft.elapsed().as_micros();
    let mut tokens_total = first.map(|v| v.len()).unwrap_or(0);

    // Subsequent calls = pure decode steps.
    loop {
        let t = Instant::now();
        let chunk = stream.next().await?;
        let dt = t.elapsed().as_micros();
        match chunk {
            Some(toks) => {
                tokens_total += toks.len();
                per_step_us.push(dt);
            }
            None => break,
        }
    }

    let warmup = input.warmup_tokens.min(per_step_us.len().saturating_sub(1));
    let stable: Vec<u128> = per_step_us.iter().skip(warmup).copied().collect();
    let decode_total_us: u128 = stable.iter().sum();
    let decode_steps = stable.len();

    let mut sorted = stable.clone();
    sorted.sort();
    let pct = |s: &[u128], p: f64| -> u128 {
        if s.is_empty() { 0 } else {
            let idx = ((s.len() - 1) as f64 * p).round() as usize;
            s[idx]
        }
    };
    let tpot_p50_us = pct(&sorted, 0.50);
    let tpot_p90_us = pct(&sorted, 0.90);
    let tpot_min_us = sorted.first().copied().unwrap_or(0);
    let tpot_max_us = sorted.last().copied().unwrap_or(0);
    let tpot_mean_us = if decode_steps > 0 {
        decode_total_us as f64 / decode_steps as f64
    } else { 0.0 };

    let out = Output {
        tokens_total,
        ttft_us,
        decode_total_us,
        decode_steps,
        tpot_mean_us,
        tpot_p50_us,
        tpot_p90_us,
        tpot_min_us,
        tpot_max_us,
        per_step_us,
        adapter_active,
    };
    Ok(serde_json::to_string(&out).unwrap())
}
