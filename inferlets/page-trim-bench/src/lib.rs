//! Benchmark inferlet for the page-trim optimization.
//!
//! Runs a controlled prefill + decode loop against synthetic token IDs (no
//! tokenizer dependence) so the measurement isolates forward-pass cost. The
//! attention mask is a sink+window pattern when `use_mask` is true, which
//! triggers the runtime's page-trim optimization (entire pages of the gap
//! region are excluded from the kernel's KV input). Setting `use_mask=false`
//! falls back to the runtime's synthesized causal mask — every page stays in
//! play, giving an apples-to-apples comparison for the optimization's effect.
//!
//! Output is a key=value summary consumed by tests and ad hoc benchmark
//! harnesses. Token IDs
//! produced by the model are discarded; we only care about timing.

use inferlet::{
    Context, Result,
    sample::Sampler,
};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt_tokens")]
    prompt_tokens: u32,
    #[serde(default = "default_decode_steps")]
    decode_steps: u32,
    #[serde(default = "default_sink_size")]
    sink_size: u32,
    #[serde(default = "default_window_size")]
    window_size: u32,
    #[serde(default = "default_use_mask")]
    use_mask: bool,
}

fn default_prompt_tokens() -> u32 { 2048 }
fn default_decode_steps() -> u32 { 256 }
fn default_sink_size() -> u32 { 4 }
fn default_window_size() -> u32 { 64 }
fn default_use_mask() -> bool { true }

/// BRLE for [sink True, gap False, window True] over `seq_len` positions.
/// If the sequence already fits inside `sink + window`, returns all-true.
fn build_sink_mask(seq_len: u32, sink: u32, window: u32) -> Vec<u32> {
    let total_kept = sink + window;
    if seq_len <= total_kept {
        vec![0, seq_len]
    } else {
        let gap = seq_len - total_kept;
        vec![0, sink, gap, window]
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mut ctx = Context::new()?;

    // Synthetic prompt tokens. We avoid tokens 0..999 to dodge any reserved
    // special-token range; modulo by 30000 keeps every id well inside any
    // realistic vocab (Qwen3, Llama, Mistral all >=128k vocab).
    let prompt: Vec<u32> = (0..input.prompt_tokens)
        .map(|i| 1000 + (i % 30000))
        .collect();

    // ── Prefill ──────────────────────────────────────────────────────
    let prefill_start = Instant::now();
    let mut pass = ctx.forward();
    pass.input(&prompt);
    if input.use_mask {
        let masks: Vec<Vec<u32>> = (0..input.prompt_tokens as usize)
            .map(|i| build_sink_mask((i + 1) as u32, input.sink_size, input.window_size))
            .collect();
        pass.attention_mask(&masks);
    }
    let h = pass.sample(&[input.prompt_tokens - 1], Sampler::Argmax);
    let out = pass.execute().await?;
    let mut next_token = out.token(h).ok_or("empty prefill output")?;
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // ── Decode loop ──────────────────────────────────────────────────
    let decode_start = Instant::now();
    for _ in 0..input.decode_steps {
        let mut pass = ctx.forward();
        pass.input(&[next_token]);
        if input.use_mask {
            // The new token lands at `start_position()`; the attention
            // mask covers everything up to and including that slot.
            let total_seq = pass.start_position() + 1;
            let mask = build_sink_mask(total_seq, input.sink_size, input.window_size);
            pass.attention_mask(&[mask]);
        }
        let h = pass.sample(&[0], Sampler::Argmax);
        let out = pass.execute().await?;
        next_token = out.token(h).ok_or("empty decode output")?;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let dec_per_step_ms = decode_ms / input.decode_steps as f64;
    let dec_tps = if dec_per_step_ms > 0.0 {
        1000.0 / dec_per_step_ms
    } else {
        f64::INFINITY
    };

    let page_size = ctx.page_size();
    println!("=== page-trim-bench ===");
    println!("prompt_tokens={}", input.prompt_tokens);
    println!("decode_steps={}", input.decode_steps);
    println!("sink_size={}", input.sink_size);
    println!("window_size={}", input.window_size);
    println!("use_mask={}", input.use_mask);
    println!("page_size={}", page_size);
    println!("prefill_ms={:.3}", prefill_ms);
    println!("decode_ms={:.3}", decode_ms);
    println!("decode_per_step_ms={:.3}", dec_per_step_ms);
    println!("decode_tokens_per_sec={:.2}", dec_tps);

    Ok(String::new())
}
