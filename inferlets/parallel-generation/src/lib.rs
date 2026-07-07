//! Demonstrates parallel text generation from forked contexts — **low-level ①
//! rewrite (chat-EOS, pipelined, FORK)**.
//!
//! A shared system-prompt prefix is prefilled once (`prefill::tokens`), forked
//! into two independent contexts (`kv.fork()` — keep-core), and each decodes
//! concurrently on the run-ahead carrier (`sampler_program(Argmax)` +
//! `submit_pass`/`discard_pass` depth-1 EOS rollback, `chat::` templating). NO
//! `Context`/`Generator`/`Sampler` facade. Both share the common-prefix KV.

use futures::future;
use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, model, prefill, Result};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_max_tokens() -> usize { 128 }

async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

fn pass_carries(stop_empty: bool, max_tokens: usize, produced_token_index: usize) -> bool {
    !(stop_empty && max_tokens == produced_token_index)
}

/// Run-ahead (pipelined) chat-EOS decode with depth-1 rollback, continuing from
/// the current `*seq_len` (so a forked/prefilled prefix on `kv` is reused).
async fn decode_pipelined(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    s: &sampler::LoweredSampler,
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: &[u32],
) -> Result<Vec<u32>> {
    let pending = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
    if max_tokens == 0 {
        return Ok(out);
    }
    let prime_carry = pass_carries(stop.is_empty(), max_tokens, 1);
    let mut producer = carrier::submit_pass(kv, seq_len, fresh, s, &pending, prime_carry)?;
    let mut generated = 0usize;
    loop {
        let speculate = generated + 1 < max_tokens;
        let consumer = if speculate {
            let carry = pass_carries(stop.is_empty(), max_tokens, generated + 2);
            Some(carrier::submit_pass(kv, seq_len, fresh, s, &[0u32], carry)?)
        } else {
            None
        };
        let token = read_token(producer).await?;
        if stop.contains(&token) {
            if let Some(c) = consumer {
                carrier::discard_pass(c, seq_len).await;
            }
            break;
        }
        out.push(token);
        generated += 1;
        match consumer {
            Some(c) => producer = c,
            None => break,
        }
    }
    Ok(out)
}

fn decode_text(tokens: &[u32]) -> Result<String> {
    let mut dec = chat::Decoder::new();
    let mut text = String::new();
    match dec.feed(tokens)? {
        chat::Event::Delta(s) | chat::Event::Done(s) => text.push_str(&s),
        _ => {}
    }
    Ok(text)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_num_outputs = input.max_tokens;
    let start = Instant::now();
    let vocab = model::output_vocab_size();
    let s = sampler::sampler_program(SamplerSpec::Argmax, vocab)?;
    let stop = chat::stop_tokens();

    // Prefill the shared system-prompt prefix once; the forks inherit it
    // (in-flight / stream-ordered, matching the original flush-then-fork).
    let base_kv = KvWorkingSet::new();
    let mut base_seq = 0u32;
    let sys = chat::system("You are a helpful, respectful and honest assistant.");
    prefill::tokens(&base_kv, &mut base_seq, &sys)?;

    let fk1 = base_kv.fork().map_err(|e| format!("fork1: {e}"))?;
    let fk2 = base_kv.fork().map_err(|e| format!("fork2: {e}"))?;
    let s_ref = &s;
    let stop_ref: &[u32] = &stop;

    let handle1 = async move {
        let mut prompt = chat::user("Explain Pulmonary Embolism");
        prompt.extend(chat::cue());
        let mut seq = base_seq;
        let mut fresh = false; // inherited prefix, not a fresh generate
        let output = decode_pipelined(&fk1, &mut seq, &mut fresh, s_ref, prompt, max_num_outputs, stop_ref)
            .await
            .and_then(|t| decode_text(&t));
        println!("Output 1: {:?} (elapsed: {:?})", output, start.elapsed());
    };

    let handle2 = async move {
        let mut prompt = chat::user("Explain the Espresso making process ELI5.");
        prompt.extend(chat::cue());
        let mut seq = base_seq;
        let mut fresh = false;
        let output = decode_pipelined(&fk2, &mut seq, &mut fresh, s_ref, prompt, max_num_outputs, stop_ref)
            .await
            .and_then(|t| decode_text(&t));
        println!("Output 2: {:?} (elapsed: {:?})", output, start.elapsed());
    };

    future::join(handle1, handle2).await;

    Ok(String::new())
}
