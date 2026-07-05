//! Text completion that exercised the speculative-decoding interface, on the raw
//! low-level WIT (keep-core), off the `Context`/`Generator` facade.
//!
//! NOTE: `Generator::system_speculation()` was a **no-op** in the shipped code —
//! it only forced the sequential (non-pipelined) decode path; no
//! `output-speculative-tokens` was ever emitted (no SDK call site), so every
//! step yielded exactly one token. The raw conversion preserves that behavior
//! exactly: a sequential 1-token-per-step decode loop
//! (`avg_tokens_per_step == 1.0`). Real device-resident MTP spec-decode lives in
//! the separate `mtp-specdecode` inferlet (the drafts channel).
//!
//! Emits one structured line on stdout when generation ends so the test
//! harness can compare runs:
//!
//!     SPEC_STATS prompt_tokens=N generated_tokens=M elapsed_ms=T tokens_per_sec=R steps=S avg_tokens_per_step=A

use inferlet::inference::ForwardPass;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, sampler, session, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_system")]
    system: String,
    /// Default 0.0 (greedy).
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
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_tokens_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    steps: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_tokens_per_step: Option<f64>,
}

/// One sequential decode fire over `tokens` at the cursor; returns the sampled
/// token (single-token step — the shipped no-op-spec behavior).
async fn fire(
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

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
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

    let mut prompt = chat::system_user(&input.system, &input.prompt);
    prompt.extend(chat::cue());

    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    if input.start_signal {
        session::send("ready");
        let _ = session::receive().await;
    }

    // The first step does prefill + the first decode step in one shot — timed
    // separately (prompt-dominated). Then the sequential decode loop.
    let mut all_tokens: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut steps: usize = 0;

    let prefill_start = Instant::now();
    if input.max_tokens > 0 {
        let first = fire(&kv, &mut seq_len, &mut fresh, &s, &prompt).await?;
        all_tokens.push(first);
        steps += 1;
    }
    let prefill_elapsed = prefill_start.elapsed();

    let decode_start = Instant::now();
    while all_tokens.len() < input.max_tokens {
        let last = *all_tokens.last().unwrap();
        let tok = fire(&kv, &mut seq_len, &mut fresh, &s, &[last]).await?;
        all_tokens.push(tok);
        steps += 1;
    }
    let decode_elapsed = decode_start.elapsed();
    let elapsed = prefill_elapsed + decode_elapsed;

    let text = if input.decode_output {
        model::decode(&all_tokens)?
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
        println!(
            "SPEC_STATS prompt_tokens={} generated_tokens={} elapsed_ms={} \
             prefill_ms={} decode_ms={} \
             tokens_per_sec={:.2} decode_tokens_per_sec={:.2} \
             steps={} avg_tokens_per_step={:.3}",
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
