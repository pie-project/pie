//! De-hardwired speculation (MTP-logits) — token-exact-vs-non-spec verify, on the
//! raw low-level WIT (keep-core), off the `Context`/`Forward` facade.
//!
//! A sampler **program** ([`edsl::mtp_argmax`]) reads the on-device MTP **draft
//! logits** through the `Binding::MtpLogits` intrinsic (the driver source-selects
//! the speculator's draft row of `ws.logits`) — the draft token is
//! `argmax(draft_logits)`. Lossless speculation ⟺ the draft token equals the
//! non-spec greedy token at the same slot. This inferlet forks the prompt KV
//! twice and samples both, then compares:
//!   * `draft`   = argmax over the MTP draft logits (`edsl::mtp_argmax`, `MtpLogits`).
//!   * `nonspec` = argmax over the target logits (`sampler_program(Argmax)`, `Logits`).
//!
//! Emits: `MTP_SPEC draft=<id> nonspec=<id> token_exact=<bool>`.
//!
//! NOTE (post-land): the draft-row source (`ctx.mtp_draft_row`) is wired post-fold
//! — until then it falls back to `sample_row`, so `draft == nonspec` trivially;
//! the assertion becomes load-bearing once the draft row is live (phase-2 e2e).
//! GPU-gated on that MTP driver wiring (like mtp-native-verify); build-verifies now.

use inferlet::inference::ForwardPass;
use inferlet::sampling::program as edsl;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, prefill, program, sampler, Result};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_system")]
    system: String,
}
fn default_prompt() -> String { "What is the capital of France?".into() }
fn default_system() -> String { "Answer in one short word.".into() }

#[derive(Serialize)]
struct Output {
    draft: u32,
    nonspec: u32,
    token_exact: bool,
}

/// Read a single-Token output tensor as a `u32` id.
async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("read: {e:?}"))?;
    if bytes.len() < 4 {
        return Err("empty token output".into());
    }
    Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32)
}

/// One decode step on a FORK of the prompt KV: the MTP **draft** token, sampled
/// from the on-device draft logits via the de-hardwired `edsl::mtp_argmax`
/// (`Binding::MtpLogits`).
async fn draft_token(root: &KvWorkingSet, seq: u32, vocab: u32) -> Result<u32> {
    let built = edsl::mtp_argmax(vocab).map_err(|e| format!("mtp_argmax build: {e:?}"))?;
    let prog = inferlet::emit::emit_program(&built.program).map_err(|e| format!("emit: {e}"))?;
    let kv = root.fork().map_err(|e| format!("fork: {e}"))?;
    let pass = ForwardPass::new();
    let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq, 1, kv.page_size()))?;
    geometry::attach_kv_write(&pass, &kv, &geom);
    pass.input_tokens(&[0u32], &[seq]);
    let bindings = program::resolve_bindings(&built.bindings, &built.host_inputs, &[seq], &[])?;
    pass.sampler(&prog, bindings);
    pass.execute();
    read_token(pass).await
}

/// One decode step on a FORK of the prompt KV: the non-spec greedy token from the
/// target logits (`sampler_program(Argmax)`) — the lossless reference.
async fn nonspec_token(root: &KvWorkingSet, seq: u32, greedy: &sampler::LoweredSampler) -> Result<u32> {
    let kv = root.fork().map_err(|e| format!("fork: {e}"))?;
    let pass = ForwardPass::new();
    let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq, 1, kv.page_size()))?;
    geometry::attach_kv_write(&pass, &kv, &geom);
    pass.input_tokens(&[0u32], &[seq]);
    pass.sampler(&greedy.program, greedy.bindings(seq)?);
    pass.execute();
    read_token(pass).await
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let vocab = model::output_vocab_size();

    // Materialize the prompt KV (system + user + cue) once; both samples fork it.
    let mut prompt = chat::system_user(&input.system, &input.prompt);
    prompt.extend(chat::cue());
    let root = KvWorkingSet::new();
    let mut seq: u32 = 0;
    prefill::tokens(&root, &mut seq, &prompt)?;

    let greedy = sampler::sampler_program(sampler::SamplerSpec::Argmax, vocab)?;

    let draft = draft_token(&root, seq, vocab).await?;
    let nonspec = nonspec_token(&root, seq, &greedy).await?;
    let token_exact = draft == nonspec;

    println!("MTP_SPEC draft={draft} nonspec={nonspec} token_exact={token_exact}");

    Ok(Output { draft, nonspec, token_exact })
}
