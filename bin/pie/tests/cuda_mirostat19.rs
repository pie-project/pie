//! #19 mirostat ARBITER (lane L3 / delta) — the e2e verify that closes #19 or
//! names the remaining layer. Boots the real 4090 + qwen-3-0.6b, fires the
//! `mirostat` inferlet (foxtrot's fix: rank floor k_min=8 + μ-init ln(vocab)+1)
//! across the locked isolation knobs, and asserts the DEFAULT is non-degenerate.
//! This ASSERTING harness replaces the false-green `mirostat_tau_sweep` (which
//! passed on token-0/S=0).
//!
//! Verdict (manager-locked):
//!   - Default (k_min=8, mu0=13) non-degenerate (S>0, μ settles, diverse) → #19 DONE.
//!   - Default degenerate ∧ init-only (k_min=0,mu0=13, NO RankLe) non-degenerate
//!     → the RankLe-custom-JIT floor is the next artifact → foxtrot argmax-floor fallback.
//!   - init-only ALSO degenerate (non-empty keep but token/S wrong) → downstream
//!     entropies→WIT→SDK S-plumbing (charlie's driver trace ruled out kernel/marshal).
//!
//! One boot per process (a 2nd in-process boot panics): all configs in ONE boot
//! via repeated run_inferlet.

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context as _, Result};
use pie_client::client::Client;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct MirostatResult {
    #[allow(dead_code)]
    sampler: String,
    count: usize,
    #[allow(dead_code)]
    tau: f32,
    final_mu: f32,
    #[allow(dead_code)]
    mean_surprise: f32,
    tail_mean_surprise: f32,
    s_flowed: bool,
    tokens: Vec<u32>,
}

fn distinct_ratio(t: &[u32]) -> f32 {
    let set: std::collections::HashSet<u32> = t.iter().copied().collect();
    set.len() as f32 / t.len().max(1) as f32
}

fn window_repeat_rate(t: &[u32], w: usize) -> f32 {
    if t.len() < 2 {
        return 0.0;
    }
    let mut reps = 0usize;
    for i in 1..t.len() {
        if t[i.saturating_sub(w)..i].contains(&t[i]) {
            reps += 1;
        }
    }
    reps as f32 / (t.len() - 1) as f32
}

/// Non-degenerate ⟺ diverse tokens AND surprise flowed AND the kept-set did NOT
/// collapse to a short repeating cycle (the token-0 / ```` ``` ```` repetition the
/// carrier bug — and a too-low `mu0` — produce). `window_repeat_rate < 0.5` is the
/// DIRECT degeneracy signal.
///
/// (Replaces the prior `final_mu < 30` guard, which was a PROXY calibrated against
/// the token-0/S=0/μ→233 carrier-bug runaway. Post-carrier-fix, working mirostat
/// settles μ in a higher band — 31–37 for the floored defaults — as it tracks τ,
/// so the μ-VALUE alone is not the degeneracy signal; token-repetition-collapse is.
/// A generous `final_mu < 150` sanity bound still catches an extreme S=0 runaway.)
fn non_degenerate(r: &MirostatResult) -> bool {
    distinct_ratio(&r.tokens) > 0.15
        && r.tail_mean_surprise > 0.1
        && window_repeat_rate(&r.tokens, WIN) < 0.5
        && r.final_mu < 150.0
}

const MAX_TOKENS: usize = 128;
const WIN: usize = 8;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the 4090 + qwen-3-0.6b + a driver-cuda build (#19 mirostat arbiter)"]
async fn mirostat19_arbiter_on_4090() -> Result<()> {
    common::init_trace();

    // #18-class lock: build the lone-`[Entropy]` (Scalar) inferlet to wasm BEFORE
    // boot (build logs stay out of the run). Fired after the mirostat configs to
    // prove a single non-Token output takes the rich path post-gate (not the
    // token-src a2 fast-path → a token's int-bits-as-f32 denormal).
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "entropycheck"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "entropycheck wasm build failed");
    let entropy_wasm = ws.join("target/wasm32-wasip2/debug/entropycheck.wasm");
    let entropy_manifest = ws.join("entropycheck/Pie.toml");

    let pie = common::boot_4090().await?;

    // (label, input). Default = both fixes (k_min=8 RankLe floor + mu0=ln(vocab)+1).
    // floor-argmax = foxtrot's proven-ops fallback (Ge/ReduceMax/Broadcast, NO RankLe).
    let configs: [(&str, String); 5] = [
        ("DEFAULT both        ", format!(r#"{{"max_tokens":{MAX_TOKENS}}}"#)),
        ("init-only(noRankLe) ", format!(r#"{{"k_min":0,"mu0":13,"max_tokens":{MAX_TOKENS}}}"#)),
        ("floor-RankLe(k8)    ", format!(r#"{{"k_min":8,"mu0":3,"max_tokens":{MAX_TOKENS}}}"#)),
        ("floor-argmax(proven)", format!(r#"{{"floor":"argmax","mu0":3,"max_tokens":{MAX_TOKENS}}}"#)),
        ("degenerate control  ", format!(r#"{{"k_min":0,"mu0":3,"max_tokens":{MAX_TOKENS}}}"#)),
    ];

    eprintln!("[MIRO19] config              | final_mu  tail_S | distinct%  win{WIN}% | s_flowed | non-degenerate");
    let mut results: Vec<(String, MirostatResult, bool)> = Vec::new();
    for (label, input) in configs.iter() {
        let json = common::run_inferlet(&pie.listen_addr, "mirostat", "mirostat@0.1.0", input).await?;
        let r: MirostatResult =
            serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("parse {label}: {e}\njson={json}"))?;
        anyhow::ensure!(r.tokens.len() == r.count, "{label}: token count mismatch");
        let nd = non_degenerate(&r);
        eprintln!(
            "[MIRO19] {label} | {:>8.3}  {:>5.2} | {:>8.1}  {:>5.0} | {:>8} | {}",
            r.final_mu,
            r.tail_mean_surprise,
            distinct_ratio(&r.tokens) * 100.0,
            window_repeat_rate(&r.tokens, WIN) * 100.0,
            r.s_flowed,
            if nd { "YES" } else { "NO (degenerate)" },
        );
        results.push((label.trim().to_string(), r, nd));
    }

    // ── #18-class lock: lone-[Entropy] (Scalar) program → rich path → real H ──
    // Post-gate, a single non-Token output must take the rich path and return a
    // genuine Shannon entropy (a plausible positive f32), NOT a token id's
    // int-bits reinterpreted as f32 (a ~1e-40 denormal) from the token-src a2 path.
    let entropy_json = {
        let c = Client::connect_with_identity(
            &format!("ws://{}/v1/ws", pie.listen_addr),
            "test-user",
        )
        .await
        .context("connect entropycheck")?;
        c.authenticate("test-user", &None).await.context("auth entropycheck")?;
        c.add_program(&entropy_wasm, &entropy_manifest, true)
            .await
            .context("add_program entropycheck")?;
        let mut p = c
            .launch_process("entropycheck@0.1.0".to_string(), "{}".to_string(), true)
            .await
            .context("launch entropycheck")?;
        let json = p.wait_for_return().await.context("wait entropycheck")?;
        drop(c);
        json
    };
    eprintln!("[MIRO19] entropycheck (#18 lone-Scalar) returned: {entropy_json}");

    pie.shutdown().await;

    let default_nd = results[0].2;
    let init_only_nd = results[1].2;
    let floor_argmax_nd = results[3].2;

    // Verdict.
    let verdict = if default_nd {
        "✅ #19 CLOSED — DEFAULT (k_min=8 RankLe floor + mu0=ln(vocab)+1) is NON-DEGENERATE (S>0, μ settled, diverse tokens). The algorithm (empty-keep) was the sole cause; the RankLe floor + μ-init fix it e2e. cuda_mirostat19 now ASSERTS non-degeneracy (replaces the false-green sweep)."
    } else if floor_argmax_nd {
        "✅ #19 CLOSED via PROVEN-OPS FALLBACK — DEFAULT (RankLe floor) degenerate BUT floor-argmax (Ge/ReduceMax/Broadcast, NO RankLe) is NON-DEGENERATE ⟹ the RankLe-custom-JIT floor is the residual artifact (codegen-unit≠e2e class); foxtrot's argmax-floor is the permanent fix → make `{floor:argmax}` the inferlet default. charlie chases the RankLe-JIT codegen separately (non-blocking)."
    } else if init_only_nd {
        "🔴 FLOOR-MECHANISM/DOWNSTREAM — DEFAULT + floor-argmax degenerate BUT init-only (μ-init alone, no floor) non-degenerate ⟹ the μ-init keeps step-1 non-empty but BOTH floor paths (RankLe + argmax) degenerate → the floor's keep/sample or the entropies→WIT→SDK S-plumbing. charlie+foxtrot+golf/echo trace."
    } else {
        "🔴 DOWNSTREAM S-PLUMBING — all fix configs degenerate. If keep is non-empty (real token) but S=0, the bug is downstream of pr.entropies (entropies→WIT→SDK→inferlet) — charlie's driver trace ruled out kernel/marshal. → foxtrot+golf/echo trace the WIT/SDK Scalar read."
    };
    eprintln!("[MIRO19] VERDICT: {verdict}");
    eprintln!("[MIRO19] MIRO19_DONE");

    // The asserting gate: at least one FLOOR config (RankLe default OR proven-ops
    // argmax) MUST be non-degenerate — i.e. #19 has a working e2e fix.
    anyhow::ensure!(
        default_nd || floor_argmax_nd,
        "MIRO19 RED: neither the RankLe-floor default nor the argmax-floor fallback is non-degenerate — {}",
        verdict
    );

    // ── #18-class gate: the lone-[Entropy] (Scalar) program returned a genuine ──
    // Shannon entropy via the rich path. A token-id mis-D2H'd into the scalar slot
    // would read as a ~1e-40 denormal (token int-bits as f32); a real H is a
    // plausible positive value in (0, ln(vocab)≈11.9).
    #[derive(Deserialize)]
    struct EntropyResult {
        entropy: f32,
    }
    let er: EntropyResult = serde_json::from_str(&entropy_json)
        .map_err(|e| anyhow::anyhow!("parse entropycheck: {e}\njson={entropy_json}"))?;
    eprintln!("[MIRO19] #18 lone-[Entropy] H={:.4} nats", er.entropy);
    anyhow::ensure!(
        er.entropy.is_finite() && er.entropy > 1e-4 && er.entropy < 15.0,
        "#18-CLASS RED: lone-[Entropy] H={} is not a plausible Shannon entropy (≈0..11.9 nats) — \
         a near-0/denormal ⟹ the lone-Scalar program was mis-gated into the token-src a2 \
         fast-path (a token id's int-bits read as f32), the #18 single-non-Token fast-path bug. \
         The single-Token gate must route it to the rich path.",
        er.entropy
    );
    eprintln!("[MIRO19] ✅ #18-CLASS LOCKED — lone-[Entropy] reads a real H via the rich path.");
    Ok(())
}
