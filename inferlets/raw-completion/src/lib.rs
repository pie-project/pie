//! Raw text-completion inferlet (no chat templating) — **low-level ① rewrite,
//! pipelined**.
//!
//! Tokenizes the prompt directly and generates a continuation so a base model
//! sees the prompt verbatim (no chat-template wrap). Per In Gim's directive
//! (minimize the Rust helper SDK; rewrite on the raw WIT API), the decode loop is
//! written by hand on the keep-core primitives — NO `Context` / `Generator` /
//! `.generate()` / `Sampler` facade:
//!   - `sampler::sampler_program(spec, vocab)` — the parametric sampler lowering
//!     (top-p rides as host-submit tensors, per the #17 de-hardwiring contract);
//!   - `carrier::submit_pass` — the run-ahead carrier threading the FULL
//!     parametric binding list (Logits + T/p) + the device carrier
//!     (`next_inputs`) so fires flow back-to-back with no host round-trip;
//!   - `geometry::*` (composed inside `carrier`).
//!
//! No-stop / fixed-budget (`max_tokens`), so the plain run-ahead path applies
//! (no EOS rollback). Pipelined-only production shape — the pipelined==sequential
//! token-identity is the GPU value gate (`ptir-inferlet-pipelining-audit §0.5`);
//! mock-clean is asserted by the inferlet-canary.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, model, Result};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    /// Raw prompt (NO chat template applied — model sees it verbatim).
    prompt: String,
    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Sampling temperature (<= 0 ⇒ greedy/argmax).
    #[serde(default = "default_temperature")]
    temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    #[serde(default = "default_top_p")]
    top_p: f32,
}

fn default_max_tokens() -> usize { 64 }
fn default_temperature() -> f32 { 0.6 }
fn default_top_p() -> f32 { 0.95 }

#[derive(Serialize)]
struct Output {
    /// The generated continuation (decoded once at end).
    text: String,
}

async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let vocab = model::output_vocab_size();

    // Keep-core sampler lowering: greedy (T<=0) → argmax; else top-p with T/p as
    // per-fire submit tensors. Built once, reused across steps.
    let spec = if input.temperature <= 0.0 {
        SamplerSpec::Argmax
    } else {
        SamplerSpec::TopP { temperature: input.temperature, p: input.top_p }
    };
    let s = sampler::sampler_program(spec, vocab)?;

    let prompt = model::encode(&input.prompt);
    let prompt = if prompt.is_empty() { vec![0u32] } else { prompt };
    let max_tokens = input.max_tokens;

    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
    if max_tokens == 0 {
        return Ok(Output { text: String::new() });
    }

    // ── run-ahead loop (no-stop): prime → speculate-before-await → promote ──
    // Carry declines only on the terminal pass (count-predictable at max_tokens).
    let prime_carry = max_tokens > 1;
    let mut producer =
        carrier::submit_pass(&kv, &mut seq_len, &mut fresh, &s, &prompt, prime_carry)?;

    let mut generated = 0usize;
    loop {
        let speculate = generated + 1 < max_tokens;
        let consumer = if speculate {
            let carry = generated + 2 < max_tokens;
            // placeholder [0u32]; the device carrier injects the producer's sample
            Some(carrier::submit_pass(&kv, &mut seq_len, &mut fresh, &s, &[0u32], carry)?)
        } else {
            None
        };

        let token = read_token(producer).await?;
        out.push(token);
        generated += 1;

        match consumer {
            Some(c) => producer = c,
            None => break, // hit max_tokens
        }
    }

    let text = model::decode(&out).unwrap_or_else(|_| String::from("[decode error]"));
    print!("{}", text);
    Ok(Output { text })
}
