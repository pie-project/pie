//! **M3 NORTH-STAR — the integrated acceptance run** (delta). The design's FINISH
//! LINE (masterplan §5 M3): overview §6.1 (MTP + grammar [+ Quest], one forward,
//! three attached programs) AND §6.2 (beam: freeze / designated-child / compact)
//! running as ONE integrated pass under the QUORUM scheduler, gated on the four M3
//! quantitative bars:
//!
//!   G1  §6.1 accepted-tokens/s  ≥ the current speculative path (MTP)
//!   G2  beam step time          within 10% of a hand-rolled baseline
//!   G3  bubble p50              < 100 µs sustained
//!   G4  dummy-run (readiness-miss) rate < 1% on the steady-state decode fleet
//!
//! This is a SCAFFOLD that ASSEMBLES the existing harnesses' mechanisms so the
//! remaining pieces plug in as they land, and the moment the GPU frees it is the
//! ONE definitive acceptance run. Gate coverage TODAY (see the #ptir table):
//!   * G3 bubble p50   — READY: `bubble_p50()` prefers the accurate driver
//!                       device-idle stamp (`probe_device_idle_us`), falls back
//!                       to the host proxy; HARD gate opt-in (`PIE_M3_BUBBLE_HARD`),
//!                       valid only on the driver stamp.
//!   * G4 dummy-run    — READY + HARD NOW: `fire.quorum.readiness_miss /
//!                       total_batches < 1%` (the quorum fire rule's structural-
//!                       readiness misses; a clean counter, not a proxy).
//!   * G1 accepted-tok/s — PLUG-IN: reads `system_spec_draft_tokens_accepted`; the
//!                       gate arms once MTP Stage-2 lands (charlie) + the §6.1
//!                       combined inferlet drafts. Provide the spec baseline via
//!                       `PIE_M3_MTP_BASELINE_TOK_S`.
//!   * G2 beam ±10%    — PLUG-IN: needs the §6.2 beam inferlet (charlie's [B,P]
//!                       geometry). The hand-rolled baseline is the
//!                       `beam-baseline` inferlet, auto-measured on the 4090
//!                       (or pin `PIE_M3_BEAM_BASELINE_US`).
//!
//! Workloads are env-selected so the real §6.1/§6.2 inferlets slot in without a
//! harness change:
//!   PIE_M3_S61_INFERLET  (default `mtpverify`) — the §6.1 MTP-verify pass
//!   PIE_M3_S62_INFERLET  (default `multisamp`) — the §6.2 beam-ish pass
//!
//! `#[ignore]` + needs `--features driver-cuda,run-ahead,profile-fire,ptir`.
//! SCAFFOLD run (defaults — G3/G4 hard, G1/G2 plug-in):
//!   PIE_COMPILER_LAUNCHER=env cargo test -p pie-bin \
//!     --features driver-cuda,run-ahead,profile-fire,ptir \
//!     --test cuda_m3_acceptance -- --ignored --nocapture
//!
//! FULL INTEGRATED ACCEPTANCE run (all 4 gates armed — the M3 finish line; G2
//! rides delta's `fire_beam` [B,P] replay + charlie's SEAM-1 gate; G1 auto-measures
//! the plain-MTP spec baseline via `mtp-native-verify` on the same boot):
//!   PIE_M3_S62_INFERLET=beam PIE_M3_S61_INFERLET=mtp-grammar PIE_M3_BUBBLE_HARD=1 \
//!     PIE_COMPILER_LAUNCHER=env cargo test -p pie-bin \
//!     --features driver-cuda,run-ahead,profile-fire,ptir \
//!     --test cuda_m3_acceptance -- --ignored --nocapture
//! → one PASS/FAIL verdict over G1 (§6.1 accepted-tok/s ≥ auto-measured plain-MTP
//!   spec baseline), G2 (beam within 10%), G3 (bubble-p50 <100µs), G4 (dummy-run
//!   <1%). (Pin `PIE_M3_MTP_BASELINE_TOK_S` to skip the baseline fleet; both §6.1
//!   and the baseline need charlie's MTP Stage-2 native draft head on-device.)
//!
//! Host-side only — does NOT touch the executor.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};
use pie_client::client::Client;

/// Steady-state decode fleet width (the M3 gate's "8 concurrent" pipelines).
const FLEET: usize = 8;
/// Per-pipeline token budget (larger ⇒ more steady-state fires past warm-up).
const BUDGET: &str = "32";

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok().and_then(|v| v.trim().parse::<f64>().ok())
}

fn inferlets_ws() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets")
}

/// Estimate the `p`-percentile bubble (µs) from a histogram delta: the UPPER bound
/// of the bucket the p-th count falls in (a conservative "< upper" bound).
fn bubble_pct(hist: &[u64], p: f64) -> u64 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0;
    }
    let target = (total as f64 * p).ceil() as u64;
    let mut cum = 0u64;
    for (i, &c) in hist.iter().enumerate() {
        cum += c;
        if cum >= target {
            return pie_engine::inference::BUBBLE_HIST_UPPER_US[i];
        }
    }
    *pie_engine::inference::BUBBLE_HIST_UPPER_US.last().unwrap()
}

/// Build an inferlet to wasm; return `(wasm, manifest)`.
fn build_inferlet(crate_name: &str) -> Result<(PathBuf, PathBuf)> {
    let ws = inferlets_ws();
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", crate_name])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "{crate_name} wasm build failed");
    let wasm = ws.join(format!("target/wasm32-wasip2/debug/{}.wasm", crate_name.replace('-', "_")));
    let manifest = ws.join(format!("{crate_name}/Pie.toml"));
    anyhow::ensure!(wasm.exists(), "missing wasm: {}", wasm.display());
    Ok((wasm, manifest))
}

async fn run_pipeline(addr: &str, program: &str, input: &str) -> Result<String> {
    let c = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "test-user").await?;
    c.authenticate("test-user", &None).await?;
    let mut proc = c.launch_process(program.to_string(), input.to_string(), true).await?;
    proc.wait_for_return().await
}

/// Spawn a FLEET-wide concurrent decode of `program` under the quorum scheduler;
/// return the count that decoded cleanly. Shared by the G1 baseline (plain-MTP)
/// and §6.1 (mtp-grammar) fleets so both measure accepted-tok/s identically.
async fn run_fleet(addr: &str, program: &str) -> usize {
    let mut handles = Vec::with_capacity(FLEET);
    for _ in 0..FLEET {
        let addr = addr.to_string();
        let prog = program.to_string();
        handles.push(tokio::spawn(async move {
            run_pipeline(&addr, &prog, BUDGET).await.is_ok()
        }));
    }
    let mut n_ok = 0usize;
    for h in handles {
        if h.await.unwrap_or(false) {
            n_ok += 1;
        }
    }
    n_ok
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "M3 north-star integrated acceptance run: needs the 4090 + cuda + run-ahead + profile-fire + ptir"]
async fn m3_integrated_acceptance_run() -> Result<()> {
    // The quorum fire rule (G4's dummy-run rate is the quorum rule's metric) is
    // selected pre-boot (OnceLock read at loop init).
    // SAFETY: set before any engine threads spawn.
    unsafe {
        if std::env::var_os("PIE_SCHED_POLICY").is_none() {
            std::env::set_var("PIE_SCHED_POLICY", "quorum");
        }
    }
    common::init_trace();
    let policy = env_str("PIE_SCHED_POLICY", "quorum");

    // §6.1 / §6.2 workloads — env-selected plug-in points. Defaults are the closest
    // existing inferlets; the true combined §6.1 (MTP+grammar+Quest, one forward)
    // and the real §6.2 beam slot in via env once they land.
    let s61 = env_str("PIE_M3_S61_INFERLET", "mtpverify");
    let s62 = env_str("PIE_M3_S62_INFERLET", "multisamp");
    let s61_prog = format!("{s61}@0.1.0");

    let (s61_wasm, s61_manifest) = build_inferlet(&s61)?;

    let pie = common::boot_4090().await?;
    let addr = pie.listen_addr.to_string();
    eprintln!("[m3] booted addr={addr} policy={policy}  §6.1={s61}  §6.2={s62}  fleet={FLEET}");

    let setup = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "test-user").await?;
    setup.authenticate("test-user", &None).await?;
    setup.add_program(&s61_wasm, &s61_manifest, true).await.context("add §6.1")?;

    // ═══ G1 BASELINE — the plain-MTP spec path's accepted-tok/s (the "current
    //     speculative path" the §6.1 combined pass must not regress). Auto-measured
    //     (like G2's beam-baseline) by running the plain-MTP fleet on the same boot,
    //     unless PIE_M3_MTP_BASELINE_TOK_S pins it. Armed when the real §6.1 inferlet
    //     (mtp-grammar) is selected or a baseline is pinned; both need charlie's MTP
    //     Stage-2 (the mtp_logits native draft head) to RUN on-device. ═══
    let g1_armed = s61 == "mtp-grammar" || env_f64("PIE_M3_MTP_BASELINE_TOK_S").is_some();
    let g1_baseline: Option<f64> = if !g1_armed {
        None
    } else if let Some(b) = env_f64("PIE_M3_MTP_BASELINE_TOK_S") {
        eprintln!("[m3] G1 baseline pinned via PIE_M3_MTP_BASELINE_TOK_S = {b:.1} tok/s");
        Some(b)
    } else {
        let mtp = env_str("PIE_M3_MTP_BASELINE_INFERLET", "mtp-native-verify");
        let (mw, mm) = build_inferlet(&mtp)?;
        setup.add_program(&mw, &mm, true).await.context("add MTP baseline")?;
        let b0 = pie_engine::inference::get_stats().await;
        let bstart = Instant::now();
        let bn_ok = run_fleet(&addr, &format!("{mtp}@0.1.0")).await;
        let bwall = bstart.elapsed();
        let b1 = pie_engine::inference::get_stats().await;
        let bd = b1
            .system_spec_draft_tokens_accepted
            .saturating_sub(b0.system_spec_draft_tokens_accepted);
        let btps = bd as f64 / bwall.as_secs_f64().max(1e-9);
        eprintln!(
            "[m3] G1 baseline (plain-MTP {mtp}) = {btps:.1} tok/s (accepted={bd}, {bn_ok}/{FLEET} ok, {:.2}s)",
            bwall.as_secs_f64()
        );
        Some(btps)
    };

    // ═══ PHASE A — steady-state §6.1 decode fleet under the quorum scheduler ═══
    // (G1 accepted-tok/s, G3 bubble p50, G4 dummy-run rate all read from this run.)
    let base = pie_engine::inference::get_stats().await;
    let phase_a_start = Instant::now();
    let n_ok = run_fleet(&addr, &s61_prog).await;
    let phase_a_wall = phase_a_start.elapsed();
    let after = pie_engine::inference::get_stats().await;

    // ── Gate metrics (deltas over Phase A) ──
    let d_batches = after.total_batches.saturating_sub(base.total_batches);
    let d_readiness_miss = after
        .fire
        .quorum
        .readiness_miss
        .saturating_sub(base.fire.quorum.readiness_miss);
    let d_accepted = after
        .system_spec_draft_tokens_accepted
        .saturating_sub(base.system_spec_draft_tokens_accepted);
    let bubble_delta: Vec<u64> = after
        .bubble_us_hist
        .iter()
        .zip(base.bubble_us_hist.iter())
        .map(|(a, b)| a.saturating_sub(*b))
        .collect();
    // The accurate driver device-idle stamp (probe_device_idle_us), phase-windowed.
    // Prefer it for G3 when the driver profiled it; else fall back to the host
    // proxy (which over-counts) — REPORT-only in that case.
    let bubble_delta_driver: Vec<u64> = after
        .bubble_us_hist_driver
        .iter()
        .zip(base.bubble_us_hist_driver.iter())
        .map(|(a, b)| a.saturating_sub(*b))
        .collect();
    let g3_from_driver: bool = bubble_delta_driver.iter().sum::<u64>() > 0;
    let g3_hist = if g3_from_driver { &bubble_delta_driver } else { &bubble_delta };

    let dummy_run_rate = if d_batches > 0 {
        d_readiness_miss as f64 / d_batches as f64
    } else {
        0.0
    };
    let bubble_p50 = bubble_pct(g3_hist, 0.50);
    let bubble_p99 = bubble_pct(g3_hist, 0.99);
    let bubble_p50_proxy = bubble_pct(&bubble_delta, 0.50);
    let accepted_tok_s = d_accepted as f64 / phase_a_wall.as_secs_f64().max(1e-9);

    // Per-gate verdict for the ONE consolidated M3 acceptance report. Each entry
    // is (gate, "PASS"/"PLUG-IN", detail). HARD gates `ensure!` (panic) on FAIL,
    // so reaching the final report means every armed gate passed.
    let mut gate_results: Vec<(&str, &str, String)> = Vec::new();

    // ═══ M3 GATE TABLE ═══
    eprintln!("\n══════════════ M3 north-star acceptance — gate table ══════════════");
    eprintln!("  fleet={FLEET}  ok={n_ok}/{FLEET}  Δfires={d_batches}  wall={:.2}s", phase_a_wall.as_secs_f64());
    eprintln!("  G1 accepted-tok/s        = {accepted_tok_s:>9.1}  (spec drafts accepted Δ={d_accepted})");
    eprintln!("  G2 beam step time        = (phase B — plug-in)");
    eprintln!("  G3 bubble p50 / p99      < {bubble_p50} µs / {bubble_p99} µs   (target p50 < 100 µs)  [{}]",
        if g3_from_driver { "driver stamp (accurate)" } else { "host proxy — over-counts" });
    if g3_from_driver {
        eprintln!("     cross-check: driver p50={bubble_p50} µs  vs  host-proxy p50={bubble_p50_proxy} µs");
    }
    eprintln!("  G4 dummy-run rate        = {:.4}%  ({d_readiness_miss}/{d_batches} readiness-miss)   (target < 1%)", dummy_run_rate * 100.0);
    eprintln!("═══════════════════════════════════════════════════════════════════");

    // ── Non-degeneracy: the fleet must have fired real batches ──
    anyhow::ensure!(d_batches > 0, "no batches fired — scheduler plumbing broken (§6.1 fleet)");
    if n_ok != FLEET {
        eprintln!("[m3] ⚠ only {n_ok}/{FLEET} §6.1 pipelines decoded cleanly — metrics reported for plumbing.");
    }

    // ── G4 dummy-run rate < 1% (READY + HARD): the quorum rule's structural-
    //    readiness miss rate. A clean counter — assert it now. ──
    anyhow::ensure!(
        dummy_run_rate < 0.01,
        "G4 FAILED: dummy-run (readiness-miss) rate {:.4}% ≥ 1% on the steady-state decode fleet",
        dummy_run_rate * 100.0
    );
    eprintln!("[m3] G4 PASS: dummy-run rate {:.4}% < 1%", dummy_run_rate * 100.0);
    gate_results.push(("G4 dummy-run <1%", "PASS", format!("{:.4}% ({d_readiness_miss}/{d_batches})", dummy_run_rate * 100.0)));

    // ── G3 bubble p50 < 100 µs (READY; HARD is opt-in and ONLY valid on the
    //    accurate driver device-idle stamp — the host proxy over-counts). ──
    if std::env::var_os("PIE_M3_BUBBLE_HARD").is_some() {
        anyhow::ensure!(
            g3_from_driver,
            "PIE_M3_BUBBLE_HARD set but no driver device-idle stamp this phase — the host \
             proxy over-counts, so the hard G3 gate needs profile-driver-cuda. Refusing to \
             gate on the proxy."
        );
        anyhow::ensure!(bubble_p50 <= 100, "G3 FAILED: driver-stamp bubble p50 {bubble_p50} µs exceeds 100 µs");
        eprintln!("[m3] G3 PASS: driver-stamp bubble p50 {bubble_p50} µs ≤ 100 µs");
        gate_results.push(("G3 bubble-p50 <100µs", "PASS", format!("{bubble_p50} µs (driver stamp)")));
    } else {
        eprintln!("[m3] G3 REPORT-only: bubble p50 {bubble_p50} µs vs 100 µs [{}] (set PIE_M3_BUBBLE_HARD to hard-gate the driver stamp).",
            if g3_from_driver { "driver stamp" } else { "host proxy" });
        gate_results.push(("G3 bubble-p50 <100µs", "PLUG-IN", format!("{bubble_p50} µs [{}] — set PIE_M3_BUBBLE_HARD", if g3_from_driver { "driver" } else { "proxy" })));
    }

    // ── G1 §6.1 accepted-tok/s ≥ the plain-MTP spec baseline (auto-measured
    //    above when PIE_M3_S61_INFERLET=mtp-grammar, else pinned). ──
    match g1_baseline {
        Some(baseline) => {
            anyhow::ensure!(
                accepted_tok_s >= baseline,
                "G1 FAILED: §6.1 accepted-tok/s {accepted_tok_s:.1} < plain-MTP spec baseline {baseline:.1}"
            );
            eprintln!("[m3] G1 PASS: §6.1 accepted-tok/s {accepted_tok_s:.1} ≥ plain-MTP baseline {baseline:.1}");
            gate_results.push(("G1 accepted-tok/s ≥ spec", "PASS", format!("{accepted_tok_s:.1} ≥ {baseline:.1}")));
        }
        None => {
            eprintln!(
                "[m3] G1 PLUG-IN: accepted-tok/s={accepted_tok_s:.1} (drafts accepted={d_accepted}). \
                 Set PIE_M3_S61_INFERLET=mtp-grammar to arm (auto-measures the plain-MTP baseline; \
                 needs charlie's MTP Stage-2 on-device) — or pin PIE_M3_MTP_BASELINE_TOK_S."
            );
            gate_results.push(("G1 accepted-tok/s ≥ spec", "PLUG-IN", format!("{accepted_tok_s:.1} tok/s — set PIE_M3_S61_INFERLET=mtp-grammar")));
        }
    }

    // ═══ PHASE B — §6.2 beam step time within 10% of hand-rolled baseline ═══
    // The hand-rolled baseline is the `beam-baseline` inferlet: the SAME beam
    // math host-orchestrated (per-beam forward → read-back logits → host top-B →
    // merge → re-fork). We MEASURE it here on the same 4090 (no hard-coded env
    // needed) unless `PIE_M3_BEAM_BASELINE_US` pins a fixed reference. The fused
    // §6.2 beam (`PIE_M3_S62_INFERLET`) must be within 10% of it.
    //
    // Gate: runs once a REAL §6.2 beam is plugged in (`PIE_M3_S62_INFERLET` !=
    // the `multisamp` placeholder) or a baseline is pinned; otherwise stays a
    // plug-in note so the scaffold is green before charlie's [B,P] geometry lands.
    let run_g2 = s62 != "multisamp" || env_f64("PIE_M3_BEAM_BASELINE_US").is_some();
    if run_g2 {
        let baseline_us = match env_f64("PIE_M3_BEAM_BASELINE_US") {
            Some(us) => {
                eprintln!("[m3] G2 baseline pinned via PIE_M3_BEAM_BASELINE_US = {us:.0} µs");
                us
            }
            None => {
                // Measure the hand-rolled baseline by running it on the 4090.
                let bl = env_str("PIE_M3_BEAM_BASELINE_INFERLET", "beam-baseline");
                let (bl_wasm, bl_manifest) = build_inferlet(&bl)?;
                setup.add_program(&bl_wasm, &bl_manifest, true).await.context("add beam baseline")?;
                let bl_start = Instant::now();
                let bl_out = run_pipeline(&addr, &format!("{bl}@0.1.0"), "{}")
                    .await
                    .context("beam baseline run")?;
                let bl_us = bl_start.elapsed().as_micros() as f64;
                eprintln!("[m3] G2 baseline (hand-rolled {bl}) wall = {bl_us:.0} µs  out={bl_out:?}");
                bl_us
            }
        };

        let (b_wasm, b_manifest) = build_inferlet(&s62)?;
        setup.add_program(&b_wasm, &b_manifest, true).await.context("add §6.2 beam")?;
        let beam_start = Instant::now();
        let out = run_pipeline(&addr, &format!("{s62}@0.1.0"), "{}").await.context("beam run")?;
        let beam_wall_us = beam_start.elapsed().as_micros() as f64;
        eprintln!("[m3] G2 beam ({s62}): wall={beam_wall_us:.0} µs  baseline={baseline_us:.0} µs  out={out:?}");
        anyhow::ensure!(
            beam_wall_us <= baseline_us * 1.10,
            "G2 FAILED: fused beam {beam_wall_us:.0} µs > 110% of hand-rolled baseline {baseline_us:.0} µs"
        );
        eprintln!("[m3] G2 PASS: fused beam within 10% of hand-rolled ({beam_wall_us:.0} ≤ {:.0} µs)", baseline_us * 1.10);
        gate_results.push(("G2 beam within 10%", "PASS", format!("{beam_wall_us:.0} ≤ {:.0} µs", baseline_us * 1.10)));
    } else {
        eprintln!(
            "[m3] G2 PLUG-IN: §6.2 beam step-time gate — set PIE_M3_S62_INFERLET=beam (charlie's \
             [B,P] geometry) to arm; the `beam-baseline` inferlet is auto-measured as the ±10% \
             reference (or pin PIE_M3_BEAM_BASELINE_US)."
        );
        gate_results.push(("G2 beam within 10%", "PLUG-IN", "set PIE_M3_S62_INFERLET=beam".into()));
    }

    pie.shutdown().await;

    // ═══ ONE consolidated M3 acceptance verdict ═══
    let n_pass = gate_results.iter().filter(|(_, v, _)| *v == "PASS").count();
    let n_plug = gate_results.iter().filter(|(_, v, _)| *v == "PLUG-IN").count();
    eprintln!("\n═══════════════ M3 NORTH-STAR ACCEPTANCE — FINAL ═══════════════");
    for (gate, verdict, detail) in &gate_results {
        eprintln!("  [{verdict:>7}] {gate:<26} {detail}");
    }
    eprintln!("  ────────────────────────────────────────────────────────────");
    if n_plug == 0 {
        eprintln!("  ✅ M3 ACCEPTED — all 4 gates hard-passed in one integrated run.");
    } else {
        eprintln!("  ⏳ M3 partial — {n_pass}/4 gates hard-passed, {n_plug} plug-in \
                   (arm: PIE_M3_S62_INFERLET=beam, PIE_M3_S61_INFERLET=mtp-grammar + \
                   PIE_M3_MTP_BASELINE_TOK_S, PIE_M3_BUBBLE_HARD=1).");
    }
    eprintln!("═══════════════════════════════════════════════════════════════");
    Ok(())
}
