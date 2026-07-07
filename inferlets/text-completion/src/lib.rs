//! Simple text completion inferlet — **raw-WIT keep-core** rewrite.
//!
//! Demonstrates chat-style pipelined (run-ahead) generation written directly on
//! the low-level WIT surface (In Gim's SDK-minimize directive): no
//! `Context`/`Generator`/`Sampler` facade. The decode LOOP + stop logic are
//! hand-written and visible here; only the correctness-critical carrier
//! mechanics are the thin keep-core primitives it calls
//! (`carrier::submit_pass` / `carrier::discard_pass`), plus the kept thin
//! bindings `chat` (templating + streaming detok), `model`, `sampler`.
//!
//! Pattern = **depth-1 EOS rollback** (`ptir-pipelined-eos-rollback-spec §4`):
//! chat generation always configures a stop set, so — to still pipeline — each
//! step eagerly speculates the next forward BEFORE the producer's token is known
//! to be a stop; when the producer IS a stop, the speculated consumer is rolled
//! back with `carrier::discard_pass` (finalize-drain + cursor −1). This is the
//! meaningful chat pipelining (the sequential-on-stop path gives zero overlap).

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, LoweredSampler, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, model, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// The user prompt to complete.
    prompt: String,

    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    /// System message for the assistant.
    #[serde(default = "default_system")]
    system: String,

    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    temperature: f32,

    /// Top-p (nucleus) sampling threshold.
    #[serde(default = "default_top_p")]
    top_p: f32,
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

/// Read the sampled token off a finalized pass's single-`Token` output tensor.
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
async fn main(input: Input) -> Result<String> {
    let vocab = model::output_vocab_size();

    // Sampler lowering (keep-core primitive) — parametric top-p, built once.
    let s: LoweredSampler = sampler::sampler_program(
        SamplerSpec::TopP {
            temperature: input.temperature,
            p: input.top_p,
        },
        vocab,
    )?;

    // Chat-templated prompt tokens (kept thin bindings). A DEFERRED system folds
    // into the first user turn via `system_user` — mirrors `Context::user`'s
    // pending-system path exactly (some templates require the combined form).
    let mut prompt = chat::system_user(&input.system, &input.prompt);
    prompt.extend(chat::cue());
    let prompt = if prompt.is_empty() { vec![0u32] } else { prompt };

    let stop = chat::stop_tokens();
    let max_tokens = input.max_tokens;

    let mut chat_dec = chat::Decoder::new();
    let mut text = String::new();

    // One decode context on the raw WIT surface: its own KV working set + cursor.
    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    if max_tokens == 0 {
        return Ok(text);
    }

    // Prime producer (prefill over the prompt tail; samples gen-token-1). With a
    // stop set the terminal pass isn't predictable at submit, so it carries; a
    // dangling carrier is cleared by the next generate's `fresh_generate` (#26).
    let mut producer = carrier::submit_pass(&kv, &mut seq_len, &mut fresh, &s, &prompt, true)?;
    let mut generated = 0usize;

    loop {
        // Speculate the next consumer eagerly UNLESS this step is the last by
        // count (max_tokens). Placeholder `[0]`; the device carrier injects the
        // producer's sampled token into input row 0.
        let speculate = generated + 1 < max_tokens;
        let consumer = if speculate {
            Some(carrier::submit_pass(
                &kv, &mut seq_len, &mut fresh, &s, &[0u32], true,
            )?)
        } else {
            None
        };

        let token = read_token(producer).await?;
        generated += 1;

        // Stop token → drop it (never emitted, never fed to the decoder), roll
        // back the speculated consumer, and finish (matches the facade's
        // `accept` stop-truncation semantics).
        let mut done = stop.contains(&token);
        if !done {
            match chat_dec.feed(&[token])? {
                chat::Event::Delta(s) => {
                    print!("{}", s);
                    text.push_str(&s);
                }
                chat::Event::Done(s) => {
                    text = s;
                    done = true;
                }
                _ => {}
            }
        }
        if generated >= max_tokens {
            done = true;
        }

        if done {
            if let Some(c) = consumer {
                carrier::discard_pass(c, &mut seq_len).await;
            }
            break;
        }

        // Not done: the consumer was speculated (generated < max_tokens ⇒
        // speculate was true), so it becomes the next producer.
        producer = consumer.expect("consumer speculated when not terminal");
    }

    Ok(text)
}
