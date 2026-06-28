//! Isolated top-p inferlet — a SINGLE TopP fire on a fresh context.
//!
//! Phase-1 #12 token gate (un-confounded): the `multisamp` harness shares one
//! context across 4 kinds, so top-p fires after top-k's tokens — and in phase-1
//! top-k stays CustomJIT, polluting top-p's input context (the full-sequence
//! `[…]×4` parity is therefore the phase-1+phase-2 done-bar). This inferlet
//! fires top-p ALONE on bare `"hello world"`, so its tokens depend only on the
//! prompt — a clean token-identity check: recognize TopP → extract(T=0.8,
//! p=0.9) → FlashInfer, vs the slot-surface baseline captured off `70e8082d`.
//!
//! Run with `PIE_FIXED_SAMPLING_SEED=12345` for reproducibility (ambient seed),
//! and `PIE_SAMPLING_IR_TRACE=1` to confirm the FlashInfer dispatch flip.

use inferlet::{Context, Result, model, sample::Sampler};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    // P3 single-model: the engine serves exactly one model — no handle to
    // load; tokenizer is the global `model::*` API.
    let mut context = Context::new()?;
    let prompt_tokens = model::encode("hello world");
    context.append(&prompt_tokens);

    let mut g = context
        .generate(Sampler::TopP { temperature: 0.8, p: 0.9 })
        .max_tokens(4);

    let mut got = Vec::new();
    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        got.extend(out.tokens.iter().copied());
    }
    eprintln!("[ISOLATED_TOPP] tokens: {got:?}");
    Ok(format!("{{\"tokens\": {got:?}}}"))
}
