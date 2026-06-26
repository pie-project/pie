//! Text completion that exercises the speculative-decoding interface.
//!
//! Uses the SDK's low-level [`TokenStream`] (via `ctx.generate(...).next()`).
//! `Speculation::Default` is set automatically by the SDK, which means every
//! forward pass requests `output_speculative_tokens(true)`. With the sglang
//! backend's NGRAM drafter enabled, the runtime starts returning draft chains
//! after the trie warms up; the `next()` call then yields >1 token per step
//! when drafts are accepted.
//!
//! Emits one structured line on stdout when generation ends so the test
//! harness can compare runs:
//!
//!     SPEC_STATS prompt_tokens=N generated_tokens=M elapsed_ms=T tokens_per_sec=R steps=S avg_tokens_per_step=A

use inferlet::{Context, Result, sample::Sampler, session};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_system")]
    system: String,
    /// Default 0.0 (greedy). Speculative decoding gets the highest acceptance
    /// rate at temp=0 because the model's predictions are deterministic, so
    /// any matching draft is guaranteed to be accepted.
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    decode_output: bool,
    #[serde(default)]
    start_signal: bool,
    #[serde(default)]
    emit_stats: bool,
    #[serde(default)]
    compact_output: bool,
}

fn default_max_tokens() -> usize {
    128
}
fn default_system() -> String {
    "You are a helpful, respectful and honest assistant.".into()
}
fn default_temperature() -> f32 {
    0.0
}
fn default_top_p() -> f32 {
    1.0
}

#[derive(Serialize)]
struct Output {
    text: String,
    generated_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    elapsed_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_per_sec: Option<f64>,
    /// Decode-loop tokens/sec, excluding the first `next()` call (which
    /// bundles prefill + the first decode step). This is the right number
    /// for NGRAM speedup comparison.
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_tokens_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    steps: Option<usize>,
    /// Average tokens accepted per `next()` call. >1.0 indicates the
    /// backend is returning useful drafts (NGRAM accepting more than the
    /// single bonus token per step).
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_tokens_per_step: Option<f64>,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let mut ctx = Context::new()?;
    ctx.system(&input.system);
    ctx.user(&input.prompt);
    ctx.cue();

    let sampler = if input.temperature <= 0.0 {
        Sampler::Argmax
    } else {
        Sampler::TopP {
            temperature: input.temperature,
            p: input.top_p,
        }
    };

    let mut stream = ctx
        .generate(sampler)
        .max_tokens(input.max_tokens)
        .system_speculation();

    if input.start_signal {
        session::send("ready");
        let _ = session::receive().await;
    }

    // The first step does prefill + the first decode step in one shot —
    // its latency is dominated by prompt processing, not by decode, so
    // we time it separately. NGRAM only affects decode (drafts can't
    // accept prompt tokens), so the speedup comparison should focus on
    // the decode loop.
    let mut all_tokens: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut steps: usize = 0;
    let prefill_start = Instant::now();
    if let Some(step) = stream.next()? {
        let out = step.execute().await?;
        all_tokens.extend_from_slice(&out.tokens);
        steps += 1;
    }
    let prefill_elapsed = prefill_start.elapsed();

    let decode_start = Instant::now();
    while let Some(step) = stream.next()? {
        let out = step.execute().await?;
        all_tokens.extend_from_slice(&out.tokens);
        steps += 1;
    }
    let decode_elapsed = decode_start.elapsed();
    let elapsed = prefill_elapsed + decode_elapsed;

    let text = if input.decode_output {
        inferlet::model::decode(&all_tokens)?
    } else {
        String::new()
    };

    let elapsed_ms = elapsed.as_millis();
    let prefill_ms = prefill_elapsed.as_millis();
    let decode_ms = decode_elapsed.as_millis();
    let secs = elapsed.as_secs_f64();
    let decode_secs = decode_elapsed.as_secs_f64();
    let tps = if secs > 0.0 {
        all_tokens.len() as f64 / secs
    } else {
        0.0
    };
    // Decode-only tokens/sec — this is what the speedup comparison should
    // use. The first step's bonus token is counted toward "decode tokens"
    // because it WAS a decode-side sample (just bundled with prefill).
    let decode_tokens = all_tokens.len().saturating_sub(1);
    let decode_tps = if decode_secs > 0.0 && decode_tokens > 0 {
        decode_tokens as f64 / decode_secs
    } else {
        0.0
    };
    let avg_per_step = if steps > 0 {
        all_tokens.len() as f64 / steps as f64
    } else {
        0.0
    };

    if input.emit_stats {
        // Single-line, parseable. Tests opt in and read this from stdout.
        println!(
            "SPEC_STATS prompt_tokens={} generated_tokens={} elapsed_ms={} \
             prefill_ms={} decode_ms={} \
             tokens_per_sec={:.2} decode_tokens_per_sec={:.2} \
             steps={} avg_tokens_per_step={:.3}",
            // Coarse prompt token proxy (whitespace count). The exact tokenizer
            // count would require an extra tokenization round-trip; not needed
            // for speedup comparison since prefill is excluded from decode_tps.
            input.prompt.split_whitespace().count(),
            all_tokens.len(),
            elapsed_ms,
            prefill_ms,
            decode_ms,
            tps,
            decode_tps,
            steps,
            avg_per_step,
        );
    }

    Ok(Output {
        text,
        generated_tokens: all_tokens.len(),
        elapsed_ms: (!input.compact_output).then_some(elapsed_ms),
        prefill_ms: (!input.compact_output).then_some(prefill_ms),
        decode_ms: (!input.compact_output).then_some(decode_ms),
        tokens_per_sec: (!input.compact_output).then_some(tps),
        decode_tokens_per_sec: (!input.compact_output).then_some(decode_tps),
        steps: (!input.compact_output).then_some(steps),
        avg_tokens_per_step: (!input.compact_output).then_some(avg_per_step),
    })
}
