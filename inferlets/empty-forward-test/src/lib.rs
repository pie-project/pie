//! Verifies the runtime API rejects empty-input forward passes.
//!
//! This is a regression test for fix #6 in BRIDGE.md: the API-layer
//! check in `runtime/src/api/inference.rs::execute()` that returns
//! `Err("ForwardPass::execute: empty input ...")` when both
//! `input_tokens` and `speculative_tokens` are empty.

use inferlet::{Context, ForwardPassExt, Result, model::Model, runtime};
use inferlet::pie::core::inference::ForwardPass;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Default)]
struct Input {}

#[derive(Serialize)]
struct Output {
    rejected: bool,
    error_message: String,
}

#[inferlet::main]
async fn main(_input: Input) -> Result<Output> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let ctx = Context::new(&model)?;

    // Build a ForwardPass with NO input_tokens and NO speculative_tokens.
    // Per fix #6, the runtime API should reject this with a descriptive
    // error before any driver-level work is done.
    let pass = ForwardPass::new(&model);
    pass.context(ctx.inner());
    // Intentionally NOT calling pass.input_tokens(...) or
    // pass.input_speculative_tokens(...). pass.sampler(...) is also
    // skipped — even without a sampler attached, the empty input
    // alone should be enough to reject.

    match pass.execute_async().await {
        Ok(_) => {
            // If we reach here, fix #6 is missing — the API let an
            // empty-input pass through.
            Err("FAIL: empty-input forward pass was NOT rejected by the runtime API".into())
        }
        Err(e) => {
            // Expected path. The error message should mention "empty
            // input" or similar.
            let msg = format!("{e}");
            println!("Got expected rejection: {msg}");
            let mentions_empty = msg.contains("empty input")
                || msg.contains("must supply at least one token");
            if !mentions_empty {
                return Err(format!(
                    "REJECTED but error message doesn't match expected fix #6 \
                     wording. Got: {msg}"
                )
                .into());
            }
            Ok(Output {
                rejected: true,
                error_message: msg,
            })
        }
    }
}
