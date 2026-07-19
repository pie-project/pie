//! **MTP Stage 1 — native-drafter de-risk** (bravo), on the REAL 4090 +
//! Qwen3.5-0.8B (GDN backbone + a 1-layer MTP head). Validates that the
//! driver's NATIVE system drafter (`qwen3_5_mtp_forward` + `wire_system_drafter`,
//! auto-active on MTP-weight presence) produces GENUINE, LOSSLESS t+2 drafts.
//! This is FUNCTIONAL cross-backend parity + an HONEST throughput number — NOT a
//! claimed perf win (mac-master: MTP is a perf LOSS at K=1, tied 248K lm_head).
//!
//! K (native draft tokens) is `PIE_MTP_DRAFT_TOKENS` (clamp 0..32, read once at
//! driver init — entry.cpp:106; K=0 disables the drafter = the non-spec
//! baseline). Because it is a per-PROCESS static, spec vs non-spec is a
//! CROSS-INVOCATION comparison (run twice, K=0 then K=2), NOT two boots in one
//! process. Each run records `{k, tokens, elapsed}` to a temp file; the second
//! run reads the first and cross-checks.
//!
//! De-risk battery (see the session MTP-Stage-1 plan):
//!   * T1 GREEDY PARITY (host, hard) — spec output == non-spec output,
//!     token-for-token. Spec-verify only accepts what the target would greedily
//!     emit, so the sequences MUST be identical (losslessness). Validates the
//!     VERIFY; note it does NOT distinguish draft quality (a bad MTP head is
//!     still lossless — it just has low acceptance).
//!   * T4 THROUGHPUT (host, measure) — wall-clock tok/s, K=0 vs K=2. Honest
//!     curve; EXPECT K small ≤ K=0. Reported, not gated.
//!   * T2 ACCEPTANCE > 0 (driver signal, hard) — the draft-quality de-risk. A
//!     genuine t+2 head is accepted on varied text; a t+1 ECHO drafts the WRONG
//!     (previous) token → ~never accepted → rate ≈ 0. Requires a driver-emitted
//!     accept trace (see `CHARLIE` note below); the harness reads it from
//!     `PIE_MTP_TRACE_FILE` when present.
//!   * T3 NO-ECHO (driver signal, optional-stronger) — the direct CUDA analog of
//!     mac's attn=V bug: assert the MTP draft distribution differs from the
//!     backbone's t+1 distribution. Needs a cheap driver debug hook; deferred to
//!     charlie if the hook is heavy (T2 acceptance is the black-box detector).
//!
//! CHARLIE (GPU execution, post-dev-land): the native drafter's ACCEPTANCE is
//! internal to the executor (run_step_chained_system_drafter / the verify path,
//! executor.cpp:758-829 / 4379-4492). Two open items to wire the full battery:
//!   1. Confirm the plain single-token `generate` decode loop EXERCISES the
//!      native drafter (executor drafts+verifies per forward when max_drafts>0),
//!      and whether the accepted prefix advances the sequence (multi-token/
//!      forward) or is transparent. If a spec-aware decode loop is needed, adapt.
//!   2. Emit an accept trace — e.g. one line per fire to `PIE_MTP_TRACE_FILE`
//!      (raw writeln!, like the scheduler's `sched_trace_write`):
//!      `[mtp] emitted=<n> drafted=<d> accepted=<a>` — so T2 reads a real rate.
//!   Then run K=0 and K=2 and report T1/T2/T4.
//!
//! `#[ignore]`, driver-cuda. Run (both K, second run cross-checks):
//!   PIE_MTP_DRAFT_TOKENS=0 cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_mtp_stage1 -- --ignored --nocapture
//!   PIE_MTP_DRAFT_TOKENS=2 cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_mtp_stage1 -- --ignored --nocapture

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};
use pie_client::client::Client;

/// Decode length for the de-risk (enough varied text to give the drafter real
/// t+2 opportunities without a long GPU run).
const DECODE_TOKENS: usize = 24;

/// Known-correct GDN decode prefix (token ids) for the `"hello world"` prompt on
/// **Qwen/Qwen3.5-0.8B** (the model the harness loads — NOT `-Base`), greedy,
/// from the HF transformers reference (pure-torch GDN CPU fallback, bf16==fp32).
/// Generated 2026-07 via driver/cuda/tests/gdn_value_golden.py --model
/// Qwen/Qwen3.5-0.8B. Full 24: [271,71093,1497,198,13151,15004,5104,29,198,
/// 13350,8423,420,268,731,198,15863,29,198,262,361,5317,11284,420,8299]
/// ("\n\n```html\n<!DOCTYPE html>..."). NOTE: alpha's committed golden b183291b was
/// for `Qwen3.5-0.8B-Base` ([271,2,220,16,15..], a digit loop) — the WRONG model
/// variant for this harness (same tokenizer, different weights). The discriminating
/// lock is the leading run; T0 asserts the driver reproduces it.
const GDN_GOLDEN: &[u32] = &[271, 71093, 1497, 198, 13151, 15004, 5104, 29];

/// Detect a degenerate decode (the garbage-state signature): a recurrent state
/// that never advances makes the model emit a fixed argmax → too few distinct
/// tokens or one id in a long run. A real GDN decode advances per token → varied.
fn degenerate_reason(tokens: &[u32]) -> Option<String> {
    if tokens.len() < 4 {
        return None;
    }
    let distinct = tokens
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    if distinct < 3 {
        return Some(format!(
            "only {distinct} distinct token id(s) across {} — the recurrent state is \
             likely not advancing (garbage / un-reset slot)",
            tokens.len()
        ));
    }
    let (mut max_run, mut run) = (1usize, 1usize);
    for w in tokens.windows(2) {
        if w[0] == w[1] {
            run += 1;
            max_run = max_run.max(run);
        } else {
            run = 1;
        }
    }
    if max_run > tokens.len() / 2 {
        return Some(format!(
            "a {max_run}-token run of a single id across {} — degenerate output",
            tokens.len()
        ));
    }
    None
}

/// Per-K cross-invocation record path (`{k, elapsed_ms, tokens}`).
fn record_path(k: u32) -> PathBuf {
    std::env::temp_dir().join(format!("pie_mtp_stage1_k{k}.txt"))
}

/// Persist a run's `(elapsed_ms, tokens)` in a trivial line format so the
/// counterpart-K invocation can read it back (no serde dep).
fn write_record(k: u32, elapsed_ms: u128, tokens: &[u32]) -> Result<()> {
    let toks = tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",");
    std::fs::write(
        record_path(k),
        format!("elapsed_ms={elapsed_ms}\ntokens={toks}\n"),
    )
    .context("write mtp-stage1 record")
}

/// Read back a prior run's `(elapsed_ms, tokens)` if it exists.
fn read_record(k: u32) -> Option<(u128, Vec<u32>)> {
    let text = std::fs::read_to_string(record_path(k)).ok()?;
    let mut elapsed_ms = None;
    let mut tokens = None;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("elapsed_ms=") {
            elapsed_ms = v.trim().parse::<u128>().ok();
        } else if let Some(v) = line.strip_prefix("tokens=") {
            tokens = Some(
                v.split(',')
                    .filter_map(|s| s.trim().parse::<u32>().ok())
                    .collect::<Vec<u32>>(),
            );
        }
    }
    Some((elapsed_ms?, tokens?))
}

/// Parse the `generate` inferlet's `generated N tokens: [t0, t1, ...]` result.
fn parse_generated_tokens(result: &str) -> Option<Vec<u32>> {
    let lb = result.find('[')?;
    let rb = result.find(']')?;
    let toks: Vec<u32> = result[lb + 1..rb]
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect();
    (!toks.is_empty()).then_some(toks)
}

/// Read the native drafter's accept trace (`PIE_MTP_TRACE_FILE`, emitted by the
/// driver — see the CHARLIE note) and sum `emitted`/`accepted` for a rate. None
/// when the driver did not emit it (the host cannot see internal drafts).
fn read_acceptance() -> Option<(u64, u64)> {
    let path = std::env::var_os("PIE_MTP_TRACE_FILE")?;
    let text = std::fs::read_to_string(path).ok()?;
    let (mut emitted, mut accepted) = (0u64, 0u64);
    let mut saw = false;
    for line in text.lines().filter(|l| l.contains("[mtp]")) {
        let field = |key: &str| -> Option<u64> {
            let s = line.find(key)? + key.len();
            let rest = &line[s..];
            let e = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            rest[..e].parse::<u64>().ok()
        };
        if let (Some(e), Some(a)) = (field("emitted="), field("accepted=")) {
            emitted += e;
            accepted += a;
            saw = true;
        }
    }
    saw.then_some((emitted, accepted))
}

/// Run the `generate` inferlet once for `DECODE_TOKENS` and return
/// `(tokens, elapsed)`.
async fn run_generate(
    listen_addr: &std::net::SocketAddr,
) -> Result<(Vec<u32>, std::time::Duration)> {
    let c = Client::connect_with_identity(&format!("ws://{listen_addr}/v1/ws"), "test-user")
        .await
        .context("connect generate session")?;
    c.authenticate("test-user", &None)
        .await
        .context("auth generate session")?;
    let start = Instant::now();
    let mut proc = c
        .launch_process(
            format!(
                "{}@0.1.0",
                std::env::var("PIE_GDN_INFERLET").unwrap_or_else(|_| "generate-gdn".to_string())
            ),
            DECODE_TOKENS.to_string(),
            true,
        )
        .await
        .context("launch generate")?;
    let json = proc
        .wait_for_return()
        .await
        .context("wait_for_return generate")?;
    let elapsed = start.elapsed();
    drop(c);
    let tokens = parse_generated_tokens(&json)
        .with_context(|| format!("parse tokens from generate result: {json:?}"))?;
    Ok((tokens, elapsed))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "MTP Stage 1 native-drafter de-risk: needs the 4090 + cuda + Qwen3.5-0.8B (MTP head). \
            Run twice: PIE_MTP_DRAFT_TOKENS=0 then =2."]
async fn mtp_native_drafter_de_risk() -> Result<()> {
    common::init_trace();

    let k: u32 = std::env::var("PIE_MTP_DRAFT_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    eprintln!("[mtp-stage1] K (PIE_MTP_DRAFT_TOKENS) = {k}");

    // Build the greedy-decode inferlet. Qwen3.5-0.8B is a GDN/hybrid model whose
    // linear-attention layers need runtime-assigned recurrent-state (rs_cache)
    // slots — so we use `generate-gdn` (binds BOTH the KvWorkingSet and an
    // RsWorkingSet), NOT the dense `generate` (KV-only, which the GDN forward
    // rejects with "rs_cache forward missing runtime-assigned slot ids").
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let gdn_pkg = std::env::var("PIE_GDN_INFERLET").unwrap_or_else(|_| "generate-gdn".to_string());
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", &gdn_pkg])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "wasm build failed for {gdn_pkg}");

    let pie = common::boot_4090_mtp().await?;
    eprintln!(
        "[mtp-stage1] booted Qwen3.5-0.8B, listen_addr={}",
        pie.listen_addr
    );

    // Install the program once.
    let setup =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect setup")?;
    setup
        .authenticate("test-user", &None)
        .await
        .context("auth setup")?;
    let wasm = ws.join(format!(
        "target/wasm32-wasip2/debug/{}.wasm",
        gdn_pkg.replace('-', "_")
    ));
    let man = ws.join(format!("{gdn_pkg}/Pie.toml"));
    setup
        .add_program(&wasm, &man, true)
        .await
        .context("add_program generate-gdn")?;
    drop(setup);

    // Decode + time.
    let (tokens, elapsed) = run_generate(&pie.listen_addr).await?;
    eprintln!(
        "[mtp-stage1] K={k}: {} tokens in {:?} ({:.1} tok/s): {tokens:?}",
        tokens.len(),
        elapsed,
        tokens.len() as f64 / elapsed.as_secs_f64().max(1e-9),
    );
    write_record(k, elapsed.as_millis(), &tokens)?;

    // Sanity: a real decode produced the full budget of tokens.
    anyhow::ensure!(
        tokens.len() == DECODE_TOKENS,
        "expected {DECODE_TOKENS} decoded tokens, got {} ({tokens:?})",
        tokens.len()
    );

    // T0 VALUE — verify the GDN output is CORRECT, not merely self-consistent.
    // This is the gate that defeats the identical-garbage T1 false-pass I warned
    // about: if the recurrent state never advances (un-reset / wrong RS wiring),
    // K=2 and K=0 both emit the SAME garbage → T1 parity passes on nonsense. So:
    //  (a) NON-DEGENERACY (hard, always): the sequence must be varied — a fixed
    //      argmax from a stuck state is degenerate;
    //  (b) GOLDEN (hard when set): the decode must match the known-correct prefix
    //      captured on the 4090 (the definitive correctness check).
    if let Some(reason) = degenerate_reason(&tokens) {
        anyhow::bail!(
            "T0 VALUE FAILED — GDN output is degenerate: {reason}. A correct GDN decode \
             advances the recurrent state per token → varied, coherent output. Tokens: {tokens:?}"
        );
    }
    if !GDN_GOLDEN.is_empty() {
        let n = GDN_GOLDEN.len().min(tokens.len());
        anyhow::ensure!(
            &tokens[..n] == GDN_GOLDEN,
            "T0 GOLDEN MISMATCH — the GDN decode diverged from the captured golden prefix \
             (a real correctness regression — the recurrent state / RS wiring is wrong).\n \
             golden: {GDN_GOLDEN:?}\n got:    {:?}",
            &tokens[..n]
        );
        eprintln!("[mtp-stage1] T0 GOLDEN OK: first {n} tokens match the captured golden.");
    } else {
        eprintln!(
            "[mtp-stage1] T0: non-degeneracy OK ({} distinct tokens). No golden set yet — \
             EYEBALL the decoded tokens above for coherent output, then paste the leading ids \
             into GDN_GOLDEN to lock correctness (T1 parity alone false-passes on garbage).",
            tokens
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
        );
    }

    // T2 ACCEPTANCE (driver signal) — the draft-quality de-risk. Only when the
    // driver emitted the accept trace (K>0 spec run). A genuine t+2 head has
    // acceptance > 0 on varied text; a t+1 echo ≈ 0.
    if k > 0 {
        match read_acceptance() {
            Some((emitted, accepted)) if emitted > 0 => {
                let rate = accepted as f64 / emitted as f64;
                eprintln!(
                    "[mtp-stage1] T2 ACCEPTANCE: accepted={accepted} emitted={emitted} rate={rate:.3}"
                );
                anyhow::ensure!(
                    accepted > 0,
                    "T2 FAILED — native MTP acceptance is 0 (drafts never accepted). Either the \
                     MTP head echoes t+1 (attn-over-history degenerated → mac's attn=V bug on \
                     CUDA) or the drafter is not engaged. emitted={emitted}."
                );
            }
            _ => eprintln!(
                "[mtp-stage1] T2 SKIPPED — no PIE_MTP_TRACE_FILE accept trace (driver did not \
                 emit `[mtp] emitted=.. accepted=..`; see the CHARLIE note). Wire the emission to \
                 gate T2 on real acceptance."
            ),
        }
    }

    // T1 GREEDY PARITY + T4 THROUGHPUT (cross-invocation). T0 (the GDN_GOLDEN
    // value-lock above) proves the recurrent-STATE is value-correct; T1 proves the
    // spec-verify is LOSSLESS: K=2 (native MTP draft→verify→accept) must produce
    // the byte-identical token sequence to K=0 (plain decode). HARD gate.
    //
    // Fixed (was xfail): the commit-advance/FLA kernel double-rounded the bf16 state
    // (bf16(bf16(state*g)+k·δ)) vs the decode-step kernel's single round, so K=2's
    // accepted-prefix replay diverged from K=0's decode → T1 failed. The FLA kernel
    // is now commit_len-GATED (gated_delta_net.cu): single-round for the
    // commit-advance replay (commit_len!=null → bit-matches the decode step → T1
    // lossless) while the plain prefill (commit_len==null) keeps double-round (the
    // HF-exact T0 trajectory). Both K=0 and K=2 now == the HF golden. bravo's
    // single-round + charlie's commit_len gating.
    let counterpart = if k == 0 { 2 } else { 0 };
    if let Some((other_ms, other_tokens)) = read_record(counterpart) {
        anyhow::ensure!(
            tokens == other_tokens,
            "T1 GREEDY PARITY FAILED — K={k} and K={counterpart} produced DIFFERENT token \
             sequences; the native-MTP spec-verify must be lossless (accept only the target's \
             greedy token). This is a recurrent-state fold regression (see the commit_len gating \
             in gated_delta_net.cu).\n K={k}: {tokens:?}\n K={counterpart}: {other_tokens:?}"
        );
        let (spec_ms, base_ms) = if k > 0 {
            (elapsed.as_millis(), other_ms)
        } else {
            (other_ms, elapsed.as_millis())
        };
        eprintln!(
            "[mtp-stage1] T1 PARITY OK (lossless). T4 THROUGHPUT: spec(K>0)={spec_ms}ms \
             base(K=0)={base_ms}ms ratio={:.2}x (honest — EXPECT ~1x-or-slower at small K; \
             the perf story is the concurrency capstone, not this single-stream number).",
            base_ms as f64 / spec_ms.max(1) as f64
        );
    } else {
        eprintln!(
            "[mtp-stage1] counterpart K={counterpart} record absent — run the other K to \
             cross-check T1 parity + T4 throughput."
        );
    }

    pie.shutdown().await;
    Ok(())
}
