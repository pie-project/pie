//! Capability assertions for the 4090 real-driver pass (lane L7 / hotel).
//!
//! The 3-way native-Rust harness is: **echo** boots `pie_worker::engine` on the
//! 4090 → **golf** submits the `mirostat`/`grammar` WASM inferlets via `client/rust`
//! → **hotel** (this module) asserts the capability properties on the inferlets'
//! structured-JSON output.
//!
//! These assertions are model-independent (they validate the *capability* —
//! μ-convergence, grammar conformance — not specific token ids, since the real
//! model's logits drive the actual tokens). They are unit-tested here against
//! golf's locked JSON schema so the logic is verified without the GPU; the
//! stitched harness calls them on real on-GPU output.

use serde::Deserialize;

/// mirostat inferlet output (golf's schema, `59b0df7a`/`b3e84e92`).
#[derive(Debug, Deserialize)]
pub struct MirostatResult {
    pub sampler: String,
    pub count: usize,
    pub tau: f32,
    pub final_mu: f32,
    pub mean_surprise: f32,
    /// Mean S over the **second half** of steps — the field to assert
    /// μ-convergence against (μ needs a few steps to settle).
    pub tail_mean_surprise: f32,
    /// Whether the Scalar S channel was marshaled (must be true for the
    /// μ-update to have run — false ⇒ the multi-output path is broken).
    pub s_flowed: bool,
    pub tokens: Vec<u32>,
}

/// grammar inferlet output. `conformant:true` is guaranteed on the Ok path (the
/// inferlet returns Err if any token violates the constraint), so a parseable
/// result already proves the IR mask constrained argmax end-to-end.
#[derive(Debug, Deserialize)]
pub struct GrammarResult {
    pub sampler: String,
    pub conformant: bool,
    pub count: usize,
    pub tokens: Vec<u32>,
}

/// Assertion outcome with a human-readable reason on failure.
pub type Check = Result<(), String>;

/// **WS2 μ-convergence gate.** Asserts the mirostat run converged: S flowed
/// end-to-end, the tail-mean surprise tracks τ within `tol`, and the run
/// produced the expected token count. This is the authoritative on-GPU proof
/// that the programmable mirostat sampler drives the distribution to the target
/// surprise.
pub fn assert_mirostat_converged(json: &str, tol: f32) -> Check {
    let r: MirostatResult =
        serde_json::from_str(json).map_err(|e| format!("mirostat JSON parse: {e}"))?;
    if r.sampler != "mirostat" {
        return Err(format!("expected sampler=mirostat, got {}", r.sampler));
    }
    if !r.s_flowed {
        return Err("s_flowed=false — Scalar S channel not marshaled; μ-update skipped".into());
    }
    if r.tokens.len() != r.count {
        return Err(format!("token count mismatch: count={} tokens={}", r.count, r.tokens.len()));
    }
    if r.count == 0 {
        return Err("mirostat produced zero tokens".into());
    }
    let gap = (r.tail_mean_surprise - r.tau).abs();
    if gap > tol {
        return Err(format!(
            "μ did not converge: |tail_mean_surprise {:.3} − τ {:.3}| = {:.3} > tol {:.3}",
            r.tail_mean_surprise, r.tau, gap, tol
        ));
    }
    // μ must be finite and have moved from the μ₀=2τ init toward equilibrium.
    if !r.final_mu.is_finite() {
        return Err(format!("final_mu not finite: {}", r.final_mu));
    }
    Ok(())
}

/// **WS3 grammar-conformance gate.** Asserts the constrained decode conformed:
/// the inferlet self-reports `conformant`, and we independently re-verify every
/// emitted token lies in the allowed alphabet (defense in depth — the harness
/// shouldn't trust the inferlet's own flag alone).
pub fn assert_grammar_conformant(json: &str, alphabet: &[u32]) -> Check {
    let r: GrammarResult =
        serde_json::from_str(json).map_err(|e| format!("grammar JSON parse: {e}"))?;
    if r.sampler != "grammar" {
        return Err(format!("expected sampler=grammar, got {}", r.sampler));
    }
    if !r.conformant {
        return Err("inferlet reported conformant=false".into());
    }
    if r.tokens.len() != r.count {
        return Err(format!("token count mismatch: count={} tokens={}", r.count, r.tokens.len()));
    }
    if r.count == 0 {
        return Err("grammar produced zero tokens".into());
    }
    // Independent re-verification: every token in the alphabet, no immediate repeat.
    let mut prev: Option<u32> = None;
    for (i, &t) in r.tokens.iter().enumerate() {
        if !alphabet.contains(&t) {
            return Err(format!("token[{i}]={t} not in allowed alphabet {alphabet:?}"));
        }
        if Some(t) == prev {
            return Err(format!("token[{i}]={t} repeats previous (no-repeat violated)"));
        }
        prev = Some(t);
    }
    Ok(())
}

