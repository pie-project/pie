//! **Pipelined ① copy of `multisamp`** — exercises the run-ahead carrier across
//! EVERY parametric standard-sampler kind (TopK / TopP / MinP / TopKTopP) through
//! `carrier::submit_pass`, proving the carrier↔sampler bridge threads the
//! param submit-tensors for all kinds (not just top-p). Keep-core primitives only
//! (`sampler::sampler_program` + `carrier::submit_pass`); NO facade.
//!
//! A COPY, not a rewrite-in-place (the original `multisamp` is a load-bearing #7
//! dispatch-parity baseline). It differs deliberately in ONE way: each kind runs
//! on its OWN fresh context (independent pipelined decode) rather than the
//! original's shared-context continuation — the copy's purpose is carrier×kind
//! validation, not the dispatch-pollution probe. On `EchoBehavior(42)` each of
//! the 4 kinds echoes 4×42 → `{"tokens": [42×16]}`.

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

    let specs: [(&str, SamplerSpec); 4] = [
        ("topk", SamplerSpec::TopK { temperature: 0.8, k: 40 }),
        ("topp", SamplerSpec::TopP { temperature: 0.8, p: 0.9 }),
        ("minp", SamplerSpec::MinP { temperature: 0.8, p: 0.05 }),
        ("joint", SamplerSpec::TopKTopP { temperature: 0.8, k: 40, p: 0.9 }),
    ];

    let mut all = Vec::new();
    for (name, spec) in specs {
        let s = sampler::sampler_program(spec, vocab)?;
        // Independent fresh context per kind (the copy's carrier×kind check).
        let kv = KvWorkingSet::new();
        let mut seq_len: u32 = 0;
        let mut fresh = true;
        let prompt = model::encode("hello world");
        let got = decode_pipelined(&kv, &mut seq_len, &mut fresh, &s, prompt, 4).await?;
        eprintln!("[MULTISAMP_PIPELINED] {name} tokens: {got:?}");
        all.extend(got);
    }

    Ok(format!("{{\"tokens\": {all:?}}}"))
}
