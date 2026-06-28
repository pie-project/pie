//! De-hardwired speculation (MTP-logits) — token-exact-vs-non-spec verify.
//!
//! Demonstrates the de-hardwired speculation path: a sampler **program**
//! ([`program::mtp_argmax`]) reads the on-device MTP **draft logits** through the
//! `Binding::MtpLogits` intrinsic — the driver source-selects the speculator's
//! draft row of `ws.logits` — instead of a hardwired speculation kernel. The
//! draft token is just `argmax(draft_logits)`; the bytecode is byte-identical to
//! a plain logits argmax (only the manifest binding differs).
//!
//! Lossless speculation ⟺ the draft token equals the non-spec greedy token at
//! the same position. This inferlet samples both at the decode slot and compares:
//!   * `draft`   = argmax over the MTP draft logits (`program::mtp_argmax`, `MtpLogits`).
//!   * `nonspec` = argmax over the target logits (`Sampler::Argmax`, `Logits`).
//!
//! Emits a single parseable line for the host test harness:
//!
//!     MTP_SPEC draft=<id> nonspec=<id> token_exact=<bool>
//!
//! NOTE (post-land): the draft-row source (`ctx.mtp_draft_row`, echo's
//! contract-#2) is wired post-fold — until then `mtp_draft_row = -1` falls back
//! to `sample_row`, so `draft == nonspec` trivially. The `token_exact`
//! assertion becomes load-bearing once the draft row is live (phase-2 e2e).

use inferlet::{Context, Result, forward::Forward, sample::Sampler, sampling::program};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_system")]
    system: String,
}
fn default_prompt() -> String {
    "What is the capital of France?".into()
}
fn default_system() -> String {
    "Answer in one short word.".into()
}

#[derive(Serialize)]
struct Output {
    draft: u32,
    nonspec: u32,
    token_exact: bool,
}

/// One decode step on a fork of `base`: the MTP **draft** token, sampled from
/// the on-device draft logits via the de-hardwired [`program::mtp_argmax`]
/// (binds `Binding::MtpLogits`, which the driver source-selects to the
/// speculator's draft row of `ws.logits`).
async fn draft_token(base: &Context, vocab: u32) -> Result<u32> {
    let built = program::mtp_argmax(vocab).map_err(|e| format!("mtp_argmax build: {e:?}"))?;
    let mut ctx = base.fork()?;
    let mut pass: Forward = ctx.forward();
    pass.input(&[0u32]);
    let h = pass.measure(built)?[0];
    let out = pass.execute().await?;
    out.token(h).await
}

/// One decode step on a fork of `base`: the non-spec greedy token, sampled from
/// the target logits (`Sampler::Argmax`) — the lossless reference.
async fn nonspec_token(base: &Context) -> Result<u32> {
    let mut ctx = base.fork()?;
    let mut pass: Forward = ctx.forward();
    pass.input(&[0u32]);
    let h = pass.sample(Sampler::Argmax)?[0];
    let out = pass.execute().await?;
    out.token(h).await
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let mut ctx = Context::new()?;
    ctx.system(&input.system).user(&input.prompt).cue();
    ctx.flush().await?;

    let vocab = inferlet::model::output_vocab_size();

    // Sample the draft (MTP-logits program) and the non-spec reference at the
    // decode slot, then compare — lossless speculation ⟺ token_exact.
    let draft = draft_token(&ctx, vocab).await?;
    let nonspec = nonspec_token(&ctx).await?;
    let token_exact = draft == nonspec;

    println!("MTP_SPEC draft={draft} nonspec={nonspec} token_exact={token_exact}");

    Ok(Output { draft, nonspec, token_exact })
}
