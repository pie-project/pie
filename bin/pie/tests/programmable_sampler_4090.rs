//! Programmable-sampler **4090 real-driver capability pass** (Phase-2 WS7, lane L7 / hotel).
//!
//! The authoritative end-to-end proof that custom Sampling-IR samplers — authored
//! by an inferlet, lowered to bytecode, JIT-compiled to fused CUDA kernels — drive
//! a real decode on the **RTX 4090** against **qwen-3-0.6b**. The 3-way stitch:
//!   * **echo** — `common::boot_4090()` boots the embedded controller+gateway+worker
//!     with the real `cuda_native` driver (engine-boot gate, `cuda_boot_smoke.rs`).
//!   * **golf** — `common::run_inferlet()` builds the wasm inferlet + submits it
//!     over the `pie-client` edge (connect → add → launch → wait_for_return).
//!   * **hotel** (this file) — asserts the *capability property* on the inferlet's
//!     structured-JSON result (μ→τ convergence, grammar conformance). These are
//!     model-independent: they validate the sampler behaviour, not specific token
//!     ids (the real model's logits drive the actual tokens).
//!
//! Each test boots its own engine (the runtime owns process-global singletons —
//! a 2nd boot in-process panics), so every body is a separate `#[ignore]` test.
//! Compiles **Rust-only** (no `driver-cuda` needed to type-check — `boot_4090`
//! only *runs* under the cuda driver). Run live on the 4090 (GPU-free, one CUDA
//! job at a time) with:
//!   PIE_COMPILER_LAUNCHER=env CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm CARGO_BUILD_JOBS=2 \
//!   cargo test -j2 -p pie-bin --features driver-cuda \
//!     --test programmable_sampler_4090 -- --ignored --nocapture --test-threads=1

mod common;

use anyhow::Result;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Capability assertions (hotel — unit-tested standalone in
// `runtime/tests/sampler_assert.rs` against golf's locked JSON schema; here they
// run on real on-GPU output). Model-independent: they check the *property*, not
// token ids.
// ---------------------------------------------------------------------------

/// mirostat inferlet output (golf's locked schema).
#[derive(Debug, Deserialize)]
struct MirostatResult {
    sampler: String,
    count: usize,
    tau: f32,
    final_mu: f32,
    #[allow(dead_code)]
    mean_surprise: f32,
    /// Mean surprise over the **second half** of steps — the field to assert
    /// μ-convergence against (μ needs a few steps to settle).
    tail_mean_surprise: f32,
    /// Whether the Scalar S channel was marshaled (must be true for the μ-update
    /// to have run — false ⇒ the multi-output path is broken).
    s_flowed: bool,
    tokens: Vec<u32>,
}

/// grammar inferlet output. `conformant:true` is guaranteed on the Ok path (the
/// inferlet returns Err on any constraint violation), so a parseable result
/// already proves the IR mask constrained argmax end-to-end.
#[derive(Debug, Deserialize)]
struct GrammarResult {
    sampler: String,
    conformant: bool,
    count: usize,
    tokens: Vec<u32>,
}

/// **WS2 μ-convergence gate.** Asserts the mirostat run converged: S flowed
/// end-to-end, the tail-mean surprise tracks τ within `tol`, and the run
/// produced the expected token count.
///
/// **τ must be model-achievable.** mirostat truncates to tokens with surprise
/// `≤ μ` and samples the kept (high-prob) set, so it can only drive surprise
/// *up to* the model's natural full-dist sampling surprise (the loosest
/// truncation). A τ above that ceiling (e.g. 3.0 vs qwen3-0.6b's ~2.1–2.6) is
/// structurally unreachable → μ runs away and this gate only passes when the
/// natural surprise coincidentally lands within `tol` of τ. Pass an achievable
/// τ below the ceiling. (hotel diagnosis, 2026-06-27.)
fn assert_mirostat_converged(json: &str, tol: f32) -> Result<(), String> {
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
    if !r.final_mu.is_finite() {
        return Err(format!("final_mu not finite: {}", r.final_mu));
    }
    Ok(())
}

/// **WS3 grammar-conformance gate.** Asserts the constrained decode conformed:
/// the inferlet self-reports `conformant`, and we independently re-verify every
/// emitted token lies in the allowed alphabet with no immediate repeat (defense
/// in depth — don't trust the inferlet's own flag alone).
fn assert_grammar_conformant(json: &str, alphabet: &[u32]) -> Result<(), String> {
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

// ---------------------------------------------------------------------------
// 4090 capability tests (one boot per process → one `#[ignore]` test each).
// ---------------------------------------------------------------------------

/// **WS2 — mirostat μ-control convergence on real qwen-3-0.6b logits.** Boots the
/// real cuda driver, runs the mirostat inferlet at an **achievable** target
/// `{"tau":1.5,"max_tokens":64}`, and asserts the adaptive sampler drove the
/// decode's tail-mean surprise to τ within tolerance — the on-GPU proof that the
/// programmable mirostat μ-control loop genuinely regulates perplexity (Token+Scalar
/// multi-output path, the S channel feeding the μ-update). τ is chosen below the
/// model's natural sampling-surprise ceiling so mirostat's downward truncation can
/// actually reach it (an unreachable τ>ceiling makes μ run away and only "passes"
/// on coincidence — see the test body).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the 4090 + HF-cached qwen-3-0.6b + a driver-cuda build"]
async fn mirostat_converges_on_4090() -> Result<()> {
    let pie = common::boot_4090().await?;

    // τ=1.5 is BELOW qwen3-0.6b's natural sampling-surprise ceiling (measured
    // ≈2.2 nats — see `mirostat_tau_sweep_on_4090`), so mirostat truncates
    // *downward* to reach it → μ settles (no runaway) → genuine convergence.
    // τ=3.0 (the inferlet default) is ABOVE that ceiling and structurally
    // unreachable (mirostat can only drive surprise up to the full-dist surprise:
    // at τ=3.0 the 4090 gives tail≈2.2, μ runs away to ≈23.8, |tail−τ|≈0.80), so
    // the gate would only pass by coincidence when the natural surprise happens to
    // land within tol of τ. 64 steps gives a stable tail. (Diagnosis: hotel,
    // 2026-06-27; verified on the 4090 at dev 00db408d.)
    let json = common::run_inferlet(
        &pie.listen_addr,
        "mirostat",
        "mirostat@0.1.0",
        r#"{"tau":1.5,"max_tokens":64}"#,
    )
    .await?;

    pie.shutdown().await;

    // Genuine μ-control: tail-mean surprise tracks the ACHIEVABLE τ=1.5 within tol.
    // Measured on the 4090: |tail−τ| ≈ 0.04 over two independent runs (μ settles
    // at ≈3.5–6.8, not the τ=3.0 runaway) — tol 0.6 leaves >10× margin while still
    // failing the unreachable-τ runaway (|tail−τ|≈0.80 at τ=3.0).
    assert_mirostat_converged(&json, 0.6).map_err(|e| anyhow::anyhow!("mirostat: {e}\njson={json}"))?;
    Ok(())
}

/// **WS2 — mirostat τ-sweep diagnostic (one boot).** Runs the mirostat inferlet
/// at several targets τ ∈ {1.0, 1.5, 2.0, 3.0} against real qwen-3-0.6b logits and
/// prints `final_mu` / `tail_mean_surprise` / `|tail−τ|` per τ. Empirically locates
/// the model's natural sampling-surprise ceiling: at τ **below** the ceiling μ
/// settles and `tail≈τ` (genuine downward control); at τ **above** it μ runs away
/// (`final_mu` ≫ τ) and `tail` saturates at the ceiling (target unreachable). Not a
/// gate — it's the data that justifies the achievable τ in `mirostat_converges_on_4090`.
/// Run with `--nocapture` to read the table.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the 4090 + HF-cached qwen-3-0.6b + a driver-cuda build"]
async fn mirostat_tau_sweep_on_4090() -> Result<()> {
    let pie = common::boot_4090().await?;

    eprintln!("[MIROSTAT-SWEEP] tau | final_mu | tail_surprise | mean_surprise | |tail-tau| | settled?");
    let mut rows: Vec<(f32, f32, f32, f32)> = Vec::new();
    for tau in [1.0_f32, 1.5, 2.0, 3.0] {
        let input = format!(r#"{{"tau":{tau},"max_tokens":64}}"#);
        let json = common::run_inferlet(&pie.listen_addr, "mirostat", "mirostat@0.1.0", &input).await?;
        let r: MirostatResult =
            serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("sweep parse: {e}\njson={json}"))?;
        // "settled" ⇒ μ did not run away (stays within a few nats of τ, not ≫ τ).
        let settled = r.final_mu <= tau + 3.0;
        eprintln!(
            "[MIROSTAT-SWEEP] {:>3.1} | {:>8.3} | {:>13.4} | {:>13.4} | {:>9.4} | {}",
            tau,
            r.final_mu,
            r.tail_mean_surprise,
            r.mean_surprise,
            (r.tail_mean_surprise - tau).abs(),
            if settled { "yes" } else { "RUNAWAY" },
        );
        assert!(r.s_flowed, "sweep τ={tau}: S channel did not flow (μ-update path broken)");
        rows.push((tau, r.final_mu, r.tail_mean_surprise, r.mean_surprise));
    }
    pie.shutdown().await;

    // The ceiling is ≈ the max tail-surprise reached across the sweep (the loosest
    // truncation — what μ saturates toward when τ is unreachable).
    let ceiling = rows.iter().map(|&(_, _, t, _)| t).fold(f32::MIN, f32::max);
    eprintln!("[MIROSTAT-SWEEP] natural sampling-surprise ceiling ≈ {ceiling:.4} nats");
    Ok(())
}

/// **WS3 — grammar conformance on real qwen-3-0.6b logits.** Boots the real cuda
/// driver, runs the grammar inferlet, and asserts every emitted token obeys the
/// IR mask constraint (alphabet {10,11,12,13}, no immediate repeat) — the on-GPU
/// proof that the programmable mask sampler constrains argmax end-to-end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the 4090 + HF-cached qwen-3-0.6b + a driver-cuda build"]
async fn grammar_conforms_on_4090() -> Result<()> {
    let pie = common::boot_4090().await?;

    let json = common::run_inferlet(
        &pie.listen_addr,
        "grammar",
        "grammar@0.1.0",
        "{}",
    )
    .await?;

    pie.shutdown().await;

    assert_grammar_conformant(&json, &[10, 11, 12, 13])
        .map_err(|e| anyhow::anyhow!("grammar: {e}\njson={json}"))?;
    Ok(())
}

/// Greedy minimal slice (bravo's isolator): boots the 4090 with the real cuda
/// driver, runs the `generate` inferlet (`Sampler::TopK { temperature: 0.0,
/// k: 1 }` ⇒ argmax — single-output, no submit-input), and asserts it produced
/// real greedy tokens. Isolates the core carrier→argmax→`pi.sampled` path from
/// grammar's mask-apply + mirostat's multi-output (the two failing sub-paths):
/// if greedy tokens come back non-degenerate, the core custom-IR decode path is
/// healthy on HW and the remaining bugs are sub-path-specific.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the 4090 + HF-cached qwen-3-0.6b + a driver-cuda build"]
async fn generate_greedy_on_4090() -> Result<()> {
    let pie = common::boot_4090().await?;

    let json = common::run_inferlet(
        &pie.listen_addr,
        "generate",
        "generate@0.1.0",
        "\"\"",
    )
    .await?;

    pie.shutdown().await;

    // The inferlet returns "generated N tokens: [t0, t1, ...]". The core
    // carrier→argmax→pi.sampled path is healthy iff it generated the tokens
    // and they are not the all-zero degenerate (the failure mode under test).
    assert!(
        json.contains("generated 5 tokens"),
        "expected 5 greedy tokens; got: {json}"
    );
    assert!(
        !json.contains("[0, 0, 0, 0, 0]"),
        "greedy argmax produced all-zero tokens — core carrier→argmax→pi.sampled broken: {json}"
    );
    Ok(())
}

// NOTE — spec-verify (WS4) is **not** in this 4090 e2e pass: there is no
// spec-verify inferlet yet (it needs a draft model + target model wired through
// the stack, heavier than the single-model mirostat/grammar capabilities).
// Greedy + lossless spec-verify are already four-way-locked at the eval/codegen
// level (foxtrot's `spec_verify_lossless` on hotel's `GatherCols` eval arm +
// charlie's GPU verify). Add a 3rd body here when a spec-verify inferlet lands.
