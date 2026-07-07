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

use inferlet::inference::ForwardPass;
use inferlet::program::{encode_f32, resolve_bindings};
use inferlet::sampling::program as edsl;
use inferlet::serde_json;
use inferlet::working_set::KvWorkingSet;
use inferlet::{geometry, model, Result};

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
    // Min-kept-set floor knob for the RankLe `{"floor":"rank"}` path (k_min top-by-
    // logit). On the real spread vocab the surprise FLOOR (~10 nats, charlie's NVRTC)
    // is far above μ0=2τ → plain mirostat's surprise gate empties → argmax-of-all-
    // (-inf) → token-0 → μ runaway; the rank floor keeps ≥ k_min tokens regardless of
    // μ. `{"k_min":0}` builds plain `mirostat` (the degenerate control).
    let k_min = json_usize(&params, "k_min", 8) as u32;

    // Logits/output vocab (= hf_config.vocab_size), not the tokenizer token
    // count: the sampler program is lowered + sampled over the logits dim.
    let vocab = model::output_vocab_size();

    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;
    let mut prompt = model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }

    // Build the mirostat program once (binding-free; reusable across fires).
    // `keys.mu` is a submit-bound scalar rebound every fire. RNG is ambient
    // (model B) — no seed input. Emit the WIT `tensor::program` once.
    // Floor selection. DEFAULT = `argmax` (#19 honest close): the proven-ops floor
    // (Ge/ReduceMax/Broadcast, no RankLe) keeps the argmax token → never empty-keep,
    // and is robustly non-degenerate (4/4 in delta's e2e runs). The RankLe rank-floor
    // `{"floor":"rank"}` (k_min top-by-logit) gives higher diversity when stable but is
    // RNG-fragile (3/4 — the custom-JIT RankLe residual); `{"k_min":0}` → plain mirostat.
    let floor = params.get("floor").and_then(|v| v.as_str()).unwrap_or("argmax");
    let (built, keys) = match floor {
        "argmax" => edsl::mirostat_argmax_floor(vocab)
            .map_err(|e| format!("mirostat_argmax_floor build: {e:?}"))?,
        _ if k_min > 0 => {
            edsl::mirostat_floor(vocab, k_min).map_err(|e| format!("mirostat_floor build: {e:?}"))?
        }
        _ => edsl::mirostat(vocab).map_err(|e| format!("mirostat program build: {e:?}"))?,
    };
    let program =
        inferlet::emit::emit_program(&built.program).map_err(|e| format!("mirostat emit: {e}"))?;
    let n_out = built.outputs.len() as u32;

    // μ-init re-tune (#19): the standard 2τ init is BELOW the spread-vocab surprise
    // floor (~10 nats on 151936), which starves the gate. `ln(vocab)` is an upper
    // bound on any distribution's min-surprise (= −log(1/vocab)), so μ0 = ln(vocab)+1
    // guarantees a non-empty INITIAL keep regardless of the (unknown) floor; the rank
    // floor (mirostat_floor) keeps it non-empty on every subsequent step too.
    let mu0_default = (vocab as f32).ln() + 1.0;
    let mut mu: f32 = json_f32(&params, "mu0", mu0_default);
    let mut surprises: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut pending: Vec<u32> = prompt;

    for _ in 0..max_tokens {
        let n = pending.len() as u32;
        let pass = ForwardPass::new();
        if fresh {
            pass.fresh_generate();
            fresh = false;
        }
        let geom = geometry::ensure_pages(
            &kv,
            geometry::kv_write_geometry(seq_len, n, kv.page_size()),
        )?;
        geometry::attach_kv_write(&pass, &kv, &geom);
        let positions: Vec<u32> = (seq_len..seq_len + n).collect();
        pass.input_tokens(&pending, &positions);
        // The program's `Logits` input is bound to the decode position (the last
        // pending token); μ is the submit-bound scalar, rebound each fire.
        let decode_pos = seq_len + n - 1;
        let bindings = resolve_bindings(
            &built.bindings,
            &built.host_inputs,
            &[decode_pos],
            &[(keys.mu, encode_f32(&[mu]))],
        )?;
        // Raw keep-core sampler attach (2-arg): output count is inferred from the
        // program's declared outputs; read them below via `outputs()`.
        pass.sampler(&program, bindings);
        pass.execute();
        seq_len += n;

        // The sampler's declared outputs in program order: [token] or
        // [token, surprise]. `n_out` is carried only for the doc/degenerate note.
        let _ = n_out;
        let tensors = pass
            .outputs()
            .await
            .map_err(|e| format!("mirostat: outputs: {e}"))?;
        let tok_bytes = tensors
            .first()
            .ok_or_else(|| "mirostat: no token output".to_string())?
            .read()
            .map_err(|e| format!("mirostat: read token: {e:?}"))?;
        let token = u32::from_le_bytes([tok_bytes[0], tok_bytes[1], tok_bytes[2], tok_bytes[3]]);
        generated.push(token);

        // CPU control update: μ ← μ − lr·(S − τ). The scalar S is the program's
        // second output; if it is absent (single-output backend) skip the update
        // for this step (still runs e2e).
        if let Some(t_s) = tensors.get(1) {
            if let Ok(sb) = t_s.read() {
                if sb.len() >= 4 {
                    let s = f32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);
                    surprises.push(s);
                    mu -= lr * (s - tau);
                }
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
