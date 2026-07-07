//! **North-star e2e harness (skeleton)** — the scaffold that ties thrusts 1, 2,
//! and 3 together toward overview §6.1 (MTP speculation + grammar constraint +
//! Quest attention, one pass, three programs) and the masterplan M3 gates.
//!
//! This is bravo's T2-side integration seam. It stands up the structure and
//! proves the slices that are runnable on the mock **today**; the full §6.1
//! composition lands incrementally as the other thrusts converge (see the
//! extension points below). Everything here is mock/host-only — no GPU. The
//! numeric north-star gates (accepted-tokens/s, bubble p50 < 100 µs sustained)
//! are the real-driver (4090) harness's job; this proves *structural*
//! correctness of the composition.
//!
//! ## §6.1 composition — who supplies what
//!
//! | piece                          | thrust | status here |
//! |--------------------------------|--------|-------------|
//! | constrained decode loop (scheduler + working set + program carrier) | **1 + 2 + 3** | proven below (`grammar` §6.1 slice, eval-mock) |
//! | grammar constraint (`argmax(mask_apply(logits, mask))`) | **3** (echo/charlie/delta) | proven below (device mask-apply == host, forced-out) |
//! | run-ahead decode (`submit(t+1)` before `out.take(t)`) | **2** (bravo) | real-driver gate (mock lacks the device carrier); scheduler side proven on mock |
//! | quorum fire rule (F1–F6), depth-1 enqueue | **2** (bravo) | `QuorumPolicy` (unit + fleet-sim tested); wired behind `run-ahead` |
//! | in-flight-safe `alloc` headroom top-up | **2 + 1** | proven in `runahead_alloc.rs` |
//! | MTP speculation (K+1 window, match-verify) | **3** | extension point — building blocks present (`mtp_argmax`/`spec_verify_greedy`/MtpLogits mock support); pending the composed decode-loop inferlet |
//! | Quest page-mask attention sink | **3 + 1** | extension point — pending `attn_page_mask` backend + `envelope_dot` (overview §4, direction-only) |
//! | §6.2 beam search (B lanes in ONE forward) | **1 + 3** | pending — single batched instance ⇒ dodges the multi-pipeline arena bug; likely first full composition to run |
//!
//! ## Extension points (as the thrusts land)
//!
//! 1. **M2 3+2 (C2):** swap the eval-mock's submit-bound grammar mask for a
//!    channel-bit late input — the sampler parks on the channel word, the forward
//!    overlaps. The C2 wait mechanism (device dirty-flag word → channel bit) is
//!    already in place; this harness re-points the grammar slice at it.
//! 2. **M2 3+1 (C1):** `fwd.attn_working_set(&ws, &cursor, P_MAX)` with the
//!    descriptor's geometry contents from a device channel the host never reads.
//! 3. **M3 north star:** compose all three programs on one forward — the MTP
//!    epilogue emits drafts, the grammar walk builds the K+1 mask rows, Quest
//!    selects pages per layer — under the quorum scheduler. Add the accepted-
//!    tokens/s + sustained-bubble gates on the real-driver harness.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

mod common;
use common::{MockEnv, create_mock_env, inferlets, mock_device::SamplingProgramBehavior};

use pie::process;
use pie::program::ProgramName;

const PROCESS_TIMEOUT: Duration = Duration::from_secs(15);

/// Serializes process spawns sharing the one mock env.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

struct TestState {
    #[allow(dead_code)]
    env: MockEnv,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        inferlets::build_inferlets();
        let rt = tokio::runtime::Runtime::new().unwrap();
        // The eval-mock program executor: it runs an attached Sampling-IR program
        // (grammar's `argmax(mask_apply(logits, mask))`, a mirostat epilogue, …)
        // via the CPU `eval` over deterministic synthetic logits — the §6.1
        // epilogue-program path proven without a GPU. `fallback: 0` for the plain
        // (no-program) passes a decode loop's prefill emits.
        let env = create_mock_env(
            "test-model",
            1,
            16,
            Arc::new(SamplingProgramBehavior { fallback: 0 }),
        );
        let config = env.config();
        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

fn program_name(name: &str) -> ProgramName {
    ProgramName::parse(&format!("{name}@0.1.0")).unwrap()
}

fn spawn_and_capture(s: &TestState, name: &str, input: String) -> Result<String, String> {
    let rx = s.rt.block_on(async {
        inferlets::add_and_install(name).await;
        let (tx, rx) = tokio::sync::oneshot::channel();
        process::spawn(
            "test-user".into(),
            program_name(name),
            input,
            None,
            false,
            Some(tx),
        )
        .expect("spawn");
        rx
    });
    s.rt.block_on(async {
        match tokio::time::timeout(PROCESS_TIMEOUT, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("process result channel dropped before completion".to_string()),
            Err(_) => Err("process did not complete within timeout (hang)".to_string()),
        }
    })
}

/// **§6.1 grammar-constrained decode slice (the assembly nucleus, runnable).**
/// The core §6.1 composition achievable on the mock today: a full constrained
/// decode loop where each step binds a per-position grammar mask and the attached
/// Sampling-IR epilogue enforces it via `argmax(mask_apply(logits, mask))`. It
/// drives the whole host stack the north star assembles on — the batch scheduler,
/// the KV working set (slot ids + per-forward write txns), the forward
/// descriptors, the program carrier, and the eval-mock's per-program execution —
/// and the inferlet self-asserts the two grammar invariants:
///   - **CONFORM** — every device token equals the host `apply_mask_argmax`
///     recomputed from the raw logits + the same mask (device == host semantics);
///   - **FORCED-OUT** — the unconstrained natural argmax is disallowed by the mask
///     and the constrained pick diverges from it (the mask positively fired, not
///     a passthrough).
///
/// This is §6.1's "constraint ⟂ spec" leg proven end to end. MTP speculation and
/// Quest attention compose onto this exact loop as T3 lands (see below).
#[test]
fn north_star_grammar_constrained_decode() {
    let _serial = serial_guard();
    let s = state();
    let out = spawn_and_capture(s, "grammar", "{}".into())
        .expect("§6.1 grammar-constrained decode should run to completion");
    eprintln!("[north-star: grammar §6.1] {out}");
    // The §6.1 grammar-constraint verdict — strong, NON-DEGENERATE (delta's
    // catch): `conform` = device token == host apply_mask_argmax (device mask-
    // apply == host); `forced_out` = the mask forced a divergence from the
    // natural argmax (a no-op mask fails); `conformant` = the composite.
    assert!(
        out.contains("\"conform\":true"),
        "device mask-apply must equal host apply_mask_argmax: {out}"
    );
    assert!(
        out.contains("\"forced_out\":true"),
        "the mask must positively fire (force a divergence), not pass through: {out}"
    );
    assert!(
        out.contains("\"conformant\":true"),
        "grammar constraint must hold end-to-end (conform ∧ forced_out): {out}"
    );
    // The decode loop ran the full budget through the scheduler + working set.
    assert!(
        out.contains("\"count\":12"),
        "the constrained decode loop must run its full token budget: {out}"
    );
}

/// **Run-ahead decode slice — real-driver gate.** The `runahead` inferlet
/// (`collect_tokens_pipelined`: `submit(t+1)` before `out.take(t)`, MATCH vs the
/// synchronous stream + the #26 dangling-carry CLEAR) proves the run-ahead
/// carrier end-to-end, but requires the **device-resident producer-link carrier**
/// (retain t's sampled token → inject into t+1's input → drain-gated free) — which
/// the mock device does not implement (it has no device memory to retain into).
/// The scheduler-side quorum rule is proven on the mock (`QuorumPolicy` unit +
/// `scheduler/quorum.rs` fleet-sim); the *decode* run-ahead is validated on the
/// 4090 (its ANCHOR is a real-model token stream). `#[ignore]` until the mock
/// grows a carrier or this moves to the real-driver harness.
#[test]
#[ignore = "run-ahead carrier retain/inject/free is device-resident; validated on the 4090 harness"]
fn north_star_runahead_decode_slice() {
    let _serial = serial_guard();
    let s = state();
    let out = spawn_and_capture(s, "runahead", "8".into())
        .expect("runahead inferlet should run to completion");
    eprintln!("[north-star: runahead] {out}");
    assert!(out.contains("MATCH=true"), "run-ahead == synchronous stream: {out}");
    // depth-2 FIFO submit-ahead loop at the shipped WAR bound — verifiable now.
    // DEEP4_MATCH (depth-4) is observed, not asserted: byte-identity holds at any
    // cap (the carrier injects correctly), but to EXERCISE true 4-in-flight (the
    // depth-k WAR test that empirically confirms the single `last_eager_d2h_done`
    // scalar is depth-k-correct — no WAR ring needed, spec §3.3), run this with
    // `PIE_SCHED_MAX_IN_FLIGHT=4` (else it queues at the default cap-2).
    assert!(
        out.contains("DEEP_MATCH=true"),
        "deep-pre-submission carrier chain (depth-2 FIFO) == synchronous stream \
         (device-resident reduce-R cut): {out}"
    );
    assert!(out.contains("CLEAR_OK=true"), "#26 dangling-carry clear: {out}");
    assert!(
        out.contains("DEEP_STOP_MATCH=true"),
        "depth-k EOS-rollback (over-shoot + discard) == synchronous stop stream: {out}"
    );
    assert!(
        out.contains("DEEP_STOP_CLEAR=true"),
        "depth-k rollback free-all leaves a clean context for the next generate: {out}"
    );
}

/// **§6.1 MTP prerequisite — the spec-verify `[k]`-Token marshal runs on the
/// eval-mock.** Before composing MTP+grammar, prove the building block the
/// composition rides: the `specverify` inferlet emits a `[k,vocab]` matrix
/// intrinsic → per-row argmax → spec-verify DAG (`eq`/`cumprod`/`select`) → a
/// `[k]`-Token, marshaled OFF the system `spec_tokens` channel into the
/// per-(request,output) `program_tokens` CSR (#32/#33). `MARSHAL_EMITS_K` is the
/// value-INDEPENDENT plumbing headline (all k rows emit) — it holds on the mock
/// regardless of the eval-mock's single-row-logits limitation (which makes the
/// matrix argmax VALUES degenerate, `MATRIX_ARGMAX_OK` real-driver-only). This
/// gates the MTP composition below: the K-draft epilogue + match-verify marshal
/// the same `[k]`-Token.
#[test]
fn north_star_specverify_marshal_runs_on_mock() {
    let _serial = serial_guard();
    let s = state();
    let out = spawn_and_capture(s, "specverify", r#"{"k":4}"#.into())
        .expect("specverify inferlet should run to completion on the eval-mock");
    eprintln!("[north-star: specverify] {out}");
    assert!(
        out.contains("MARSHAL_EMITS_K=true"),
        "the spec-verify [k]-Token marshal must emit all k off program_tokens \
         (the #32 plumbing the MTP composition rides): {out}"
    );
}

/// **§6.1 MTP ⟂ grammar composition — runnable (per-position mask via `select`).**
/// The core §6.1 fusion: multi-token spec-verify composed with the grammar
/// constraint in ONE `[k, vocab]` program. The `mtpverify` inferlet (DIRECT WIT
/// bindings, In Gim's directive) fires the fused program — target-logits matrix →
/// PER-POSITION grammar mask via `select(allow[k,vocab], logits, −∞)` → per-row
/// `argmax` → verify the `[k]` draft (`eq → cumprod` prefix-AND) → sentinel-coded
/// `[k]`-Token. Per-position (not packed) masking is echo's design constraint:
/// each speculative position carries its OWN allowed-token set (the grammar state
/// advances per token), which the single-mask packed `mask_apply` can't express.
///   - **`GRAMMAR_FORCES_ACCEPT`** — an allow-only-`T` mask forces every row's
///     masked argmax to `T`, so a draft `[T; k]` is accepted in full (`[T; k]`),
///     independent of the model's natural argmax;
///   - **`COMPOSITION_FIRES`** — an all-allow mask yields a different accept-prefix
///     for the SAME draft (the mask changed the verify result — constraint ⟂
///     speculation, not a passthrough).
///
/// §6.1's "MTP + grammar on one forward" leg, proven e2e on the eval-mock (its
/// `[k, vocab]` matrix intrinsic + `[k]`-Token `program_tokens` marshal are real —
/// see `north_star_specverify_marshal_runs_on_mock`). The remaining §6.1 MTP piece
/// is the multi-STEP accept-prefix decode LOOP (commit lanes `0..=n_acc`, re-draft
/// via `mtp_logits`); its per-step verify program is exactly this fused
/// composition, and its real-model speedup validates on the driver once the R>1
/// attention compute lands.
#[test]
fn north_star_mtp_grammar_composition() {
    let _serial = serial_guard();
    let s = state();
    let out = spawn_and_capture(s, "mtpverify", "4".into())
        .expect("mtpverify inferlet should run the fused spec-verify ⟂ grammar program");
    eprintln!("[north-star: mtp-verify] {out}");
    assert!(
        out.contains("GRAMMAR_FORCES_ACCEPT=true"),
        "the per-position grammar mask must force the spec-verify to accept the \
         constrained draft in full (every row's masked argmax == the allow-only token): {out}"
    );
    assert!(
        out.contains("COMPOSITION_FIRES=true"),
        "the grammar mask must change the accept-prefix vs all-allow — the constraint \
         composes with speculation, not a passthrough: {out}"
    );
    assert!(
        out.contains("LOOP_CTRL_OK=true"),
        "the multi-step accept-prefix decode loop must advance the committed sequence by \
         each step's accepted-prefix length (accept-all → k, reject-mid → n_acc) and \
         terminate — the §6.1 spec-decode control flow: {out}"
    );
}

/// **Quest page-mask attention sink (extension point).** The §6.1 attention leg:
/// per-layer page importance → `pivot_threshold(score, rank_le(budget))` →
/// `attn_page_mask` sink, produced and consumed inside one forward (no channel,
/// no decode-chain edge). Gated on backend `attn_page_mask` availability
/// (overview §4 bind-time rule; direction-only under the no-attention-kernel
/// constraint) + thrust-1's `envelope_dot` kernel. `#[ignore]` until the backend
/// exposes it.
#[test]
#[ignore = "pending backend attn_page_mask availability + thrust-1 envelope_dot kernel (overview §4/§6.1, direction-only)"]
fn north_star_quest_attention_sink() {
    // TODO(M3, overview §6.1): on_attn_proj(|| {
    //   let score = envelope_dot(query());                       // [P_MAX]
    //   attn_page_mask(pivot_threshold(score, rank_le(budget))); // sink, [P_MAX] bool
    // }); — composes onto the MTP+grammar loop with zero added decode edges.
    unreachable!("scaffold only");
}

/// **§6.2 beam-search composition (extension point).** Beam search runs **B lanes
/// in ONE forward** — a *single* batched instance ([B, ·] rows), NOT B concurrent
/// pipelines — so it does **not** exercise the multi-pipeline concurrent-decode
/// arena bug the fleet gate is blocked on (`cuda_bubble.rs`). That makes it the
/// likely FIRST full PTIR composition to run end to end once the thrust-3 beam
/// epilogue lands: reorder is an index gather over `pages [B, P]`, divergence is a
/// **freeze** (not advancing the inherited `lens` entry), and each parent's
/// designated child keeps filling the tail — no KV copy (overview §6.2, §5.2).
///
/// Needs: thrust-1's fork/freeze working-set geometry (`pages`/`lens`/`klen`/`kvm`
/// derivatives, designated-child tail-fill, mark-sweep `free` + token-space
/// `compact`) and thrust-3's beam epilogue program (`top_k(reshape(cand,[B*V]),B)`
/// → parent/heir scatters). `#[ignore]` until both land.
#[test]
#[ignore = "pending thrust-1 fork/freeze/compact geometry + thrust-3 beam epilogue (top_k reorder)"]
fn north_star_beam_search_composition() {
    // TODO(M3, overview §6.2): drive a single B-lane batched forward beam decode —
    //   - epilogue: cand = scores ⊕ log_softmax(logits); (s, i) = top_k([B*V], B);
    //     parent = i / V; heir = designated-child scatter; reorder = gathers;
    //   - divergence = NOT advancing the inherited `lens` entry (a freeze, §5.2);
    //     the designated child fills the parent's tail, siblings open fresh slots;
    //   - mark-sweep `free` on dead branches; one token-space `compact` under a
    //     waste threshold; `klen`/`kvm` derived from `lens` in the same epilogue.
    // Assert: beam step-time within 10% of a hand-rolled baseline (overview §5
    //   gate) + token-for-token vs echo's tier-0 reference interpreter. As a
    //   single batched instance this should run on the real driver ahead of the
    //   multi-pipeline fleet gate — re-point it here once the beam epilogue lands.
    unreachable!("scaffold only");
}

/// **Concurrent fleet — crash-safety under co-batching (mock).** Launches an
/// 8-pipeline decode fleet (`generate`: append "hello world" → greedy decode of 5
/// tokens) with ALL processes in flight before awaiting any, so the scheduler
/// co-batches them. The mock's `synthetic_logits` is seeded by BATCH-ROW index
/// (not content), so pipelines legitimately diverge as co-batching varies — token
/// IDENTITY across pipelines is a real-driver check (bravo's `cuda_bubble.rs`).
/// What the mock CAN prove, and what this guards: concurrent co-batching runs
/// every pipeline to completion with a well-formed budget and NO `arena: unknown
/// object` crash / cross-request drop — the regression guard for the prefill-flush
/// finalize bug (a no-output flush that dropped unfinalized aborted its KV while
/// the working-set slots still referenced the freed pages).
#[test]
fn north_star_fleet_8_concurrent_decode_completes() {
    let _serial = serial_guard();
    let s = state();
    const FLEET: usize = 8;
    let outs: Vec<Result<String, String>> = s.rt.block_on(async {
        inferlets::add_and_install("generate").await;
        // Launch the whole fleet BEFORE awaiting any — all 8 decodes in flight so
        // the scheduler forms concurrent batches (the path that crashed).
        let mut rxs = Vec::with_capacity(FLEET);
        for _ in 0..FLEET {
            let (tx, rx) = tokio::sync::oneshot::channel();
            process::spawn(
                "test-user".into(),
                program_name("generate"),
                "{}".into(),
                None,
                false,
                Some(tx),
            )
            .expect("spawn");
            rxs.push(rx);
        }
        let mut outs = Vec::with_capacity(FLEET);
        for rx in rxs {
            outs.push(match tokio::time::timeout(PROCESS_TIMEOUT, rx).await {
                Ok(Ok(result)) => result,
                Ok(Err(_)) => Err("process result channel dropped".to_string()),
                Err(_) => Err("process did not complete within timeout (hang)".to_string()),
            });
        }
        outs
    });

    // Every pipeline must complete (no `arena: unknown object` crash, no dropped
    // channel) with its full 5-token budget. Token identity is NOT asserted: the
    // mock's row-seeded logits make it a real-driver-only property.
    for (i, o) in outs.iter().enumerate() {
        let o = o
            .as_ref()
            .unwrap_or_else(|e| panic!("fleet pipeline {i} errored under co-batching: {e}"));
        assert!(
            o.contains("generated 5 tokens"),
            "fleet pipeline {i} did not complete its budget: {o}"
        );
    }
    eprintln!("[north-star: fleet] {FLEET}/{FLEET} pipelines completed under co-batching");
}

/// **Composition placeholder — full §6.1 (MTP + grammar + Quest).** Pending the
/// thrust-3 PTIR program model (trace/channels/tiers) + the Quest `attn_page_mask`
/// backend. When those land, this drives a single forward carrying all three
/// stage programs under the quorum scheduler and asserts token-for-token parity
/// with the tier-0 reference interpreter (echo's golden model). Until then it is
/// `#[ignore]`d so the harness compiles and documents the target.
#[test]
#[ignore = "pending thrust-3 PTIR program model (MTP epilogue) + Quest attn_page_mask backend"]
fn north_star_mtp_grammar_quest_composition() {
    // TODO(M3): compose overview §6.1 —
    //   - MTP epilogue emits K drafts + match-verify (n_acc, sentinel -1);
    //   - the host grammar walk builds the [K+1, vocab] mask rows along the
    //     published drafts (`draft_out`), bound via the C2 late channel;
    //   - Quest selects `budget` pages per layer via the attn_page_mask sink;
    //   - the quorum scheduler fires the fleet depth-1, bubble → 0.
    // Assert: token-for-token parity vs echo's tier-0 reference interpreter, and
    // (on the real-driver harness) accepted-tokens/s ≥ the current speculative
    // path + sustained bubble p50 < 100 µs.
    unreachable!("scaffold only");
}
