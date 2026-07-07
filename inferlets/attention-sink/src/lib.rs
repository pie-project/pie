//! Demonstrates attention sink — bounded KV cache with preserved initial tokens.
//!
//! Maintains an "attention sink" of initial tokens plus a sliding window of
//! the most recent tokens. Tokens between the sink and the window are
//! masked via a per-step attention mask, preventing the model from
//! attending to them.
//!
//! NOTE: full KV cache eviction is not yet supported by the runtime — the
//! mask only prevents the model from *attending* to masked tokens; the KV
//! pages stay in memory.
//!
//! **Raw-WIT / keep-core rewrite** (echo, SDK-minimization ①): no `Context` /
//! `Sampler` / `Forward` facade. The decode loop is hand-written on the raw WIT
//! surface, using the kept keep-core primitives:
//!   - [`carrier::submit_pass_with`] — one forward pass (geometry + input +
//!     bind hook + argmax sampler + execute + cursor advance), with the
//!     position-deterministic sink+window mask attached in the **bind seam**
//!     (`ptir-carrier-bind-seam-spec` §5). `carry = false` → plain sequential
//!     decode (no run-ahead carrier).
//!   - [`sampler::sampler_program`] — the greedy (`Argmax`) sampler lowering.
//!   - [`geometry`](inferlet::geometry) — folded inside `submit_pass_with`.
//! `chat` (templating) and `model` (encode/decode/vocab) are kept thin WIT
//! bindings, used directly.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, model, Result};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_sink_size")]
    sink_size: u32,
    #[serde(default = "default_window_size")]
    window_size: u32,
}

fn default_prompt() -> String { "Tell me a long story about a cat.".to_string() }
fn default_max_tokens() -> usize { 512 }
fn default_sink_size() -> u32 { 4 }
fn default_window_size() -> u32 { 64 }

/// BRLE attention mask for `[sink True, gap False, window True]` over
/// `seq_len` positions. Returns all-true when the sequence fits inside
/// `sink + window`.
fn build_sink_mask(seq_len: u32, sink: u32, window: u32) -> Vec<u32> {
    let total_kept = sink + window;
    if seq_len <= total_kept {
        vec![0, seq_len]
    } else {
        let gap = seq_len - total_kept;
        vec![0, sink, gap, window]
    }
}

/// Finalize a pass and read its sampled token (the low 4 bytes of the output
/// tensor, LE). `None` on a short/empty tensor.
async fn read_token(pass: ForwardPass) -> Result<Option<u32>> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32)
    } else {
        None
    })
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let start = Instant::now();

    let vocab = model::output_vocab_size();
    let sampler = sampler::sampler_program(SamplerSpec::Argmax, vocab)?;

    let kv = KvWorkingSet::new();
    let page_size = kv.page_size();
    let stop_tokens = inferlet::chat::stop_tokens();

    // Build the prompt (chat templating is a kept thin WIT binding).
    let mut prompt: Vec<u32> = Vec::new();
    prompt.extend(inferlet::chat::system("You are a helpful assistant."));
    prompt.extend(inferlet::chat::user(&input.prompt));
    prompt.extend(inferlet::chat::cue());

    println!(
        "--- Attention Sink (sink={}, window={}, page_size={}) ---",
        input.sink_size, input.window_size, page_size
    );

    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut pending: Vec<u32> = prompt;
    // Raw decode state (was owned by `Context`): the KV cursor + the #26
    // fresh-generate arm for the first pass of this generation.
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    for _ in 0..input.max_tokens {
        if pending.is_empty() {
            break;
        }

        let n = pending.len() as u32;
        // The token(s) in `pending` land at [seq_len, seq_len + n); the mask
        // covers everything up to and including the last of them.
        let total_seq_after = seq_len + n;
        let masks: Option<Vec<Vec<u32>>> =
            if total_seq_after > input.sink_size + input.window_size {
                let mask = build_sink_mask(total_seq_after, input.sink_size, input.window_size);
                // One mask per query position (each token in `pending` is a query).
                Some((0..pending.len()).map(|_| mask.clone()).collect())
            } else {
                None
            };

        // One sequential forward pass; the sink+window mask is attached in the
        // bind seam (after `input_tokens`, before the sampler/execute tail).
        let pass = carrier::submit_pass_with(
            &kv,
            &mut seq_len,
            &mut fresh,
            &sampler,
            &pending,
            false, // sequential decode — no run-ahead carrier
            |pass| {
                if let Some(masks) = &masks {
                    pass.attention_mask(masks);
                }
            },
        )?;

        let token = match read_token(pass).await? {
            Some(t) => t,
            None => break,
        };
        if stop_tokens.contains(&token) {
            break;
        }
        generated_tokens.push(token);
        pending = vec![token];
    }

    let text = model::decode(&generated_tokens)?;
    println!("Generated {} tokens in {:?}", generated_tokens.len(), start.elapsed());
    println!("Output:\n{}", text);

    Ok(String::new())
}
