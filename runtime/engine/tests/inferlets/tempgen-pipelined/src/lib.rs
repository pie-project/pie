//! **Pipelined ① copy of `tempgen`** — temperature (`Multinomial`) decode on the
//! low-level run-ahead carrier (`sampler::sampler_program` + `carrier::submit_pass`),
//! validating the pure-temperature sampler kind threads through the carrier. NO
//! `Context`/`Generator`/`Sampler` facade.
//!
//! A COPY (the original `tempgen` is a load-bearing Temperature-dispatch baseline).
//! On the mock `EchoBehavior(42)` an 8-token decode → `{"tokens": [42×8]}`.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, model, Result};

async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

async fn decode_pipelined(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    s: &sampler::LoweredSampler,
    prompt: Vec<u32>,
    max_tokens: usize,
) -> Result<Vec<u32>> {
    let prompt = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out = Vec::with_capacity(max_tokens);
    if max_tokens == 0 {
        return Ok(out);
    }
    let prime_carry = max_tokens > 1;
    let mut producer = carrier::submit_pass(kv, seq_len, fresh, s, &prompt, prime_carry)?;
    let mut generated = 0usize;
    loop {
        let speculate = generated + 1 < max_tokens;
        let consumer = if speculate {
            let carry = generated + 2 < max_tokens;
            Some(carrier::submit_pass(kv, seq_len, fresh, s, &[0u32], carry)?)
        } else {
            None
        };
        let token = read_token(producer).await?;
        out.push(token);
        generated += 1;
        match consumer {
            Some(c) => producer = c,
            None => break,
        }
    }
    Ok(out)
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = model::output_vocab_size();
    let s = sampler::sampler_program(SamplerSpec::Multinomial { temperature: 0.8 }, vocab)?;

    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;
    let prompt = model::encode("hello world");

    let got = decode_pipelined(&kv, &mut seq_len, &mut fresh, &s, prompt, 8).await?;
    eprintln!("[TEMPGEN_PIPELINED] tokens: {got:?}");
    Ok(format!("{{\"tokens\": {got:?}}}"))
}
