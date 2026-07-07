//! Demonstrates windowed attention — sliding-window KV management.
//!
//! After filling the prompt, applies a sliding window attention mask during
//! generation to limit the model's attention to the most recent
//! `window_size` tokens. This simulates bounded-memory generation.
//!
//! NOTE: full KV cache eviction is not yet supported by the runtime — the
//! mask only prevents the model from *attending* to old tokens; the KV
//! pages stay in memory.
//!
//! **Raw-WIT / keep-core rewrite** (echo, SDK-minimization ①): no `Context` /
//! `Sampler` / `Forward` facade. The decode loop is hand-written on the raw WIT
//! surface using the kept keep-core primitives — [`carrier::submit_pass_with`]
//! (one forward pass: geometry + input + bind hook + argmax sampler + execute +
//! advance; `carry = false` → plain sequential decode) with the
//! position-deterministic sliding-window mask attached in the **bind seam**
//! (`ptir-carrier-bind-seam-spec` §5), plus [`sampler::sampler_program`] for the
//! greedy sampler. `chat` / `model` are kept thin WIT bindings used directly.

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
    #[serde(default = "default_window_size")]
    window_size: u32,
}

fn default_prompt() -> String { "Tell me a long story about a cat.".to_string() }
fn default_max_tokens() -> usize { 512 }
fn default_window_size() -> u32 { 64 }

/// BRLE attention mask for a sliding window: the most recent `window_size`
/// positions attend, everything before is masked.
fn build_window_mask(seq_len: u32, window_size: u32) -> Vec<u32> {
    if seq_len <= window_size {
        vec![0, seq_len]
    } else {
        vec![seq_len - window_size, window_size]
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

    let mut prompt: Vec<u32> = Vec::new();
    prompt.extend(inferlet::chat::system("You are a helpful assistant."));
    prompt.extend(inferlet::chat::user(&input.prompt));
    prompt.extend(inferlet::chat::cue());

    println!(
        "--- Windowed Attention (window={} tokens, page_size={}) ---",
        input.window_size, page_size
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
        let total_seq_after = seq_len + n;
        let masks: Option<Vec<Vec<u32>>> = if total_seq_after > input.window_size {
            let mask = build_window_mask(total_seq_after, input.window_size);
            Some((0..pending.len()).map(|_| mask.clone()).collect())
        } else {
            None
        };

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
