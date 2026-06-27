//! Mirostat v2 test inferlet (Phase 2, WS2).
//!
//! Demonstrates the **sequential late-bind** loop for a sampler whose control
//! value depends on the previous step's output. Each fire runs a Sampling-IR
//! `mirostat` program that truncates to tokens with surprise `-log p ≤ μ`,
//! Gumbel-samples among them, and returns BOTH the sampled `token` and the
//! observed surprise `S = -log p(token)` (nats). The inferlet then runs the
//! mirostat v2 control update on the CPU — `μ ← μ − lr·(S − τ)` — and rebinds
//! the new μ as a **submit-bound** input on the next fire. μ converges so the
//! running mean surprise tracks the target τ, all through the IR.
//!
//! Note: the program emits a Scalar `S` output (the entropy/scalar channel).
//! Backends that only marshal the Token channel will read `S` as absent; this
//! inferlet falls back to leaving μ unchanged in that case so it still runs to
//! completion (the real-driver e2e exercises the full μ-update path).

use inferlet::program::{encode_f32, resolve_bindings};
use inferlet::sampling::program as edsl;
use inferlet::serde_json;
use inferlet::{Context, Result, model};

/// Default target surprise τ (nats); override via `_input` `"tau"`.
const TAU: f32 = 3.0;
/// Default control learning rate; override via `_input` `"lr"`.
const LR: f32 = 0.6;
/// Default number of tokens to generate; override via `_input` `"max_tokens"`.
const MAX_TOKENS: usize = 16;

/// Optional `f32` field from the `_input` JSON (defaults if absent / unparseable).
fn json_f32(v: &serde_json::Value, key: &str, default: f32) -> f32 {
    v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(default)
}

/// Optional `usize` field from the `_input` JSON.
fn json_usize(v: &serde_json::Value, key: &str, default: usize) -> usize {
    v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(default)
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    // Optional run params via JSON input (`{"tau":3.0,"lr":0.6,"max_tokens":32,
    // "mu0":6.0}`); each falls back to its default so `"{}"` reproduces the
    // built-in config. Lets a harness sweep without rebuilding the wasm.
    let params: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let tau = json_f32(&params, "tau", TAU);
    let lr = json_f32(&params, "lr", LR);
    let max_tokens = json_usize(&params, "max_tokens", MAX_TOKENS);

    // Logits/output vocab (= hf_config.vocab_size), not the tokenizer token
    // count: the sampler program is lowered + sampled over the logits dim.
    let vocab = model::output_vocab_size();

    let mut context = Context::new()?;
    let mut prompt = model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }

    // Build the mirostat program once (binding-free; reusable across fires).
    // `keys.mu` is a submit-bound scalar rebound every fire. RNG is ambient
    // (model B) — no seed input. Emit the WIT `tensor::program` once.
    let (built, keys) =
        edsl::mirostat(vocab).map_err(|e| format!("mirostat program build: {e:?}"))?;
    let program =
        inferlet::emit::emit_program(&built.program).map_err(|e| format!("mirostat emit: {e}"))?;
    let n_out = built.outputs.len() as u32;

    // μ starts at 2τ (standard mirostat v2 initialization), overridable.
    let mut mu: f32 = json_f32(&params, "mu0", 2.0 * tau);
    let mut surprises: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut pending: Vec<u32> = prompt;

    for _ in 0..max_tokens {
        let mut pass = context.forward();
        let start = pass.start_position();
        pass.input(&pending);
        // The program's `Logits` input is bound to the decode position (the last
        // pending token); μ is the submit-bound scalar, rebound each fire.
        let decode_pos = start + pending.len() as u32 - 1;
        let bindings = resolve_bindings(
            &built.bindings,
            &built.host_inputs,
            &[decode_pos],
            &[(keys.mu, encode_f32(&[mu]))],
        )?;
        let handles = pass.sampler(&program, bindings, n_out);
        let out = pass.execute().await?;

        let token = out
            .token(handles[0])
            .await
            .map_err(|e| format!("mirostat: read token: {e}"))?;
        generated.push(token);

        // CPU control update: μ ← μ − lr·(S − τ). The scalar S is the program's
        // second output; if it is absent (single-output backend) skip the update
        // for this step (still runs e2e).
        if let Some(h_s) = handles.get(1) {
            if let Ok(s) = out.scalar(*h_s).await {
                surprises.push(s);
                mu -= lr * (s - tau);
            }
        }

        pending = vec![token];
    }

    let mean_s = if surprises.is_empty() {
        f32::NAN
    } else {
        surprises.iter().sum::<f32>() / surprises.len() as f32
    };
    // Late-window mean surprise (second half) — what a convergence assertion
    // should check, since μ needs a few steps to settle toward τ.
    let tail_mean_s = if surprises.len() >= 2 {
        let tail = &surprises[surprises.len() / 2..];
        tail.iter().sum::<f32>() / tail.len() as f32
    } else {
        mean_s
    };
    // `s_flowed` tells the harness whether the Scalar S channel was marshaled
    // (true on the real driver / multi-output mock; false drops the μ-update).
    let s_flowed = !surprises.is_empty();
    let tokens_json = serde_json::to_string(&generated).unwrap_or_else(|_| "[]".to_string());
    let result = format!(
        "{{\"sampler\":\"mirostat\",\"count\":{},\"tau\":{tau},\"final_mu\":{mu:.4},\
         \"mean_surprise\":{mean_s:.4},\"tail_mean_surprise\":{tail_mean_s:.4},\
         \"s_flowed\":{s_flowed},\"tokens\":{tokens_json}}}",
        generated.len(),
    );
    eprintln!("[MIROSTAT] {result}");
    Ok(result)
}
