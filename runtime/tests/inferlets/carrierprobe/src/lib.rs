//! **Carrier × parametric-sampler keep-core exercise** (echo). Proves the
//! capability that did NOT exist until the unified `carrier::submit_pass`
//! signature: a **parametric** sampler (top-p, with temperature / top-p riding
//! as host-submit tensors) driven through the **run-ahead carrier** on the raw
//! WIT surface — no `Context`/`Generator`/`collect_*` facade.
//!
//! The gap (pipe-audit, `ptir-carrier-bind-seam-spec §9`): the old
//! `submit_pass` hardwired `pass.sampler(program, [Logits])` — only the logits
//! row, dropping a parametric sampler's param submit-tensors → the driver
//! recognizer couldn't hash-match → `CustomJIT` + wrong sampling. Taking a
//! [`sampler::LoweredSampler`](inferlet::sampler::LoweredSampler) and attaching
//! its FULL binding list (`bindings(decode_pos)` = logits + T/p/k) closes it, so
//! greedy and parametric share ONE carrier path (greedy = `Argmax`).
//!
//! The loop is the faithful run-ahead pipeline (mirrors `runahead`'s
//! `decode_pipelined`): eager-submit the consumer BEFORE awaiting the producer,
//! the device carrier injecting the producer's sample into the consumer's
//! placeholder input row 0. Only the carrier MECHANICS are factored into
//! [`carrier::submit_pass`]; the decode loop + stop logic stay hand-written and
//! visible here (In Gim's SDK-minimize thesis).
//!
//! On the mock `EchoBehavior(42)` every fire echoes token 42, so a 3-step
//! pipelined decode returns `[42, 42, 42]` — the clean signal that the full
//! lowering + parametric binding + carrier path drives e2e.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, LoweredSampler, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, model, Result};

const MAX_TOKENS: usize = 3;

/// Whether a pass producing the `produced_index`-th token (1-based) should
/// declare a `next-inputs` carry (mirrors `runahead::pass_carries` with no
/// stop): decline ONLY on the terminal max-tokens boundary.
fn pass_carries(produced_index: usize) -> bool {
    produced_index != MAX_TOKENS
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

/// Run-ahead (pipelined) parametric decode over the keep-core carrier: each
/// pass is a `carrier::submit_pass` (geometry + input + FULL sampler bindings +
/// carrier decl + execute + advance-on-submit, all inside the primitive).
async fn decode_pipelined(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    sampler: &LoweredSampler,
    prompt: Vec<u32>,
) -> Result<Vec<u32>> {
    let pending = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out: Vec<u32> = Vec::with_capacity(MAX_TOKENS);

    // Prime producer (the prompt tail): carries unless it is itself terminal.
    let mut producer = carrier::submit_pass(kv, seq_len, fresh, sampler, &pending, pass_carries(1))?;

    let mut generated = 0usize;
    loop {
        // Speculate the next consumer eagerly UNLESS this step terminates.
        let speculate = generated + 1 < MAX_TOKENS;
        let consumer = if speculate {
            // placeholder `[0]`; the device carrier injects the producer's sample.
            Some(carrier::submit_pass(
                kv,
                seq_len,
                fresh,
                sampler,
                &[0u32],
                pass_carries(generated + 2),
            )?)
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

    // PARAMETRIC sampler (top-p) — the kind the OLD carrier dropped the params
    // for. Built once, reused across the pipelined decode.
    let s = sampler::sampler_program(SamplerSpec::TopP { temperature: 0.8, p: 0.9 }, vocab)?;

    let prompt = model::encode("hello world");

    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    let tokens = decode_pipelined(&kv, &mut seq_len, &mut fresh, &s, prompt).await?;

    let ok = tokens.len() == MAX_TOKENS && tokens.iter().all(|&t| t != 0);
    let result = format!("CARRIER_PARAM_OK={ok} tokens={tokens:?}");
    eprintln!("[CARRIERPROBE] {result}");
    Ok(result)
}
