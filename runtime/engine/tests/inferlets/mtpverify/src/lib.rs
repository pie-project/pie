//! **§6.1 MTP assembly — composed spec-verify ⟂ PER-POSITION grammar mask**,
//! **PTIR keep-core** rewrite: `inferlet::ptir` (`Pipeline`/`ForwardPass`/
//! `Channel`), no classic `ForwardPass::sampler`/`resolve_bindings` machinery.
//! Each verify window is ONE seeded fire — a fresh `ForwardPass` +
//! `WorkingSet` per window, exactly the old guest's
//! fresh-`KvWorkingSet`-per-window shape (and the host geometry prefill
//! resolves embed/mask/draft from SEEDS, the §3 single-fire contract).
//!
//! §6.1 composes MTP speculation with the grammar constraint on ONE forward:
//! draft K tokens, verify them against the target's per-row argmax, and apply a
//! grammar mask **per speculative position** (the grammar state advances per
//! token, so each of the K rows has its OWN allowed-token set). The packed
//! `mask_apply` op broadcasts ONE bitmask row over every query row and so
//! CANNOT express per-position masking (echo's design constraint); the
//! per-position mask is a `[k, vocab]` bool channel applied by `select`:
//!   `masked = select(allow[k,vocab], logits[k,vocab], −∞)` → per-row `argmax`.
//!
//! The fused epilogue: per-position `select`-mask → per-row `argmax` → verify
//! the `[k]` draft (`eq → cumprod` cross-row prefix-AND) → sentinel-coded
//! `[k]`-Token (accepted prefix, then `-1`; the host truncates at the first
//! sentinel — the old marshal's accept-prefix semantics). The k logits rows
//! come from a k-row `Readout` port over `prompt + (k-1)` fillers (read-out
//! positions `l-1 .. l-1+k`), so `intrinsics::logits()` declares `[k, vocab]`.
//! Deterministic verdicts (no prior greedy decode needed):
//!   - `GRAMMAR_FORCES_ACCEPT` — an allow-only-`T` mask (at every position)
//!     forces each row's masked argmax to `T`, so a draft `[T; k]` is accepted
//!     in FULL (`[T; k]`), independent of the model's natural argmax;
//!   - `COMPOSITION_FIRES` — an all-allow mask yields a DIFFERENT accept-prefix
//!     for the same draft (the mask changed the verify result — constraint ⟂
//!     speculation, not a passthrough);
//!   - `LOOP_CTRL_OK` — the multi-step accept-prefix decode LOOP control flow:
//!     per step, read the sentinel `[k]`-Token, advance the committed sequence
//!     by the accepted length, re-draft, terminate at budget (accept-all → k,
//!     reject-mid → n_acc).
//!
//! Plain input: an optional draft-window size `k` (default 4, min 2), e.g. `"6"`.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const PROMPT: &str = "The quick brown fox jumps over";
/// The grammar-forced token: the allow-only-`T` per-position mask pins each
/// row's masked argmax here.
const FORCE_TOKEN: u32 = 1;
const PAGE_T: u32 = 16; // tokens per KV page (mock env page size)
const NUM_LAYERS: u32 = 1;

/// A `[k, vocab]` bool mask (flat, row-major) with EVERY token allowed at every
/// position.
fn allow_all(k: u32, vocab: u32) -> Vec<bool> {
    vec![true; (k * vocab) as usize]
}

/// A `[k, vocab]` bool mask allowing ONLY token `t` at every position (all
/// other logits → −∞ under the `select`, so each row's masked argmax is forced
/// to `t`).
fn allow_only(k: u32, vocab: u32, t: u32) -> Vec<bool> {
    let mut m = vec![false; (k * vocab) as usize];
    for r in 0..k {
        m[(r * vocab + t) as usize] = true;
    }
    m
}

/// The old marshal's truncation semantics: the accepted prefix is everything
/// before the first `-1` sentinel.
fn accepted_prefix(raw: &[i32]) -> Vec<i32> {
    raw.iter().take_while(|&&x| x != -1).copied().collect()
}

/// Fire ONE `[k, vocab]` verify window (fresh pass + working set) over
/// `PROMPT + (k-1)` fillers, seeding the `[k, vocab]` bool allow-mask + the
/// `[k]` draft, and read the sentinel-coded `[k]`-Token back (truncated at the
/// first `-1` — the accepted prefix).
async fn verify_window(
    prompt: &[u32],
    k: u32,
    vocab: u32,
    mask: Vec<bool>,
    draft: &[i32],
) -> Result<Vec<i32>> {
    let l = prompt.len() as u32;
    let n = l + k - 1;
    let input_toks: Vec<i32> = prompt
        .iter()
        .map(|&t| t as i32)
        .chain(std::iter::repeat(0).take((k - 1) as usize))
        .collect();

    let ws = WorkingSet::new();
    let max_pages = n.div_ceil(PAGE_T);
    ws.reserve(max_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;

    // Seeded inputs (single fire: the host geometry prefill reads seeds) + the
    // terminal [k]-Token reader output.
    let toks = Channel::from(input_toks).named("toks");
    let kv_len = Channel::from(vec![n]).named("kv_len");
    let allow = Channel::from_shaped([k, vocab], mask).named("allow");
    let draft_ch = Channel::from(draft.to_vec()).named("draft");
    let out = Channel::new([k], dtype::i32).named("out");

    // k read-out rows ⇒ intrinsics::logits() declares [k, vocab].
    let readout: Vec<u32> = (0..k).map(|i| l - 1 + i).collect();

    let fwd = ForwardPass::new();
    fwd.embed(&toks, Tensor::constant(vec![0u32, n]));
    fwd.port_channel(Port::KvLen, &kv_len);
    fwd.attn_working_set(&ws, .., ..)?;
    fwd.derive_dense_geometry();
    fwd.readout(&Tensor::constant(readout));
    fwd.epilogue(move || {
        // Takes + compute first, PUTS last (value-id discipline).
        let a = allow.take().tensor(); // [k, vocab] bool per-position mask
        let d = draft_ch.take().tensor(); // [k] i32 submit draft
        let logits = intrinsics::logits(); // [k, vocab] f32 target
        let neg_inf = broadcast(Tensor::constant(f32::NEG_INFINITY), [k, vocab]);
        let masked = select(&a, &logits, &neg_inf); // per-position mask
        let tgt = reduce_argmax(masked); // [k] grammar-constrained per-row argmax
        let hit = eq(&tgt, &d); // [k] bool
        let ones = broadcast(Tensor::constant(1.0f32), [k]);
        let zeros = broadcast(Tensor::constant(0.0f32), [k]);
        // Cross-row prefix-AND: the first reject zeroes the whole suffix.
        let keep = gt(cumprod(select(&hit, &ones, &zeros)), 0.5f32);
        let neg1 = broadcast(Tensor::constant(-1i32), [k]);
        let ver = select(&keep, &d, &neg1); // accepted prefix, then -1
        out.put(&ver);
    });

    let pipeline = Pipeline::new();
    fwd.submit(&pipeline).map_err(|e| format!("submit: {e}"))?;
    let raw = out
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("out take: {e}"))?;
    pipeline.close();
    Ok(accepted_prefix(&raw))
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let k: u32 = input.trim().parse().unwrap_or(4).max(2);
    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    let mut prompt = wit_model::encode(PROMPT);
    if prompt.is_empty() {
        prompt.push(0);
    }

    let draft_t: Vec<i32> = vec![FORCE_TOKEN as i32; k as usize];

    // Arm 1 — grammar-FORCED: allow only FORCE_TOKEN at every position ⇒ every
    // row's masked argmax == FORCE_TOKEN ⇒ the draft [FORCE_TOKEN; k] accepts in
    // FULL.
    let forced = verify_window(
        &prompt,
        k,
        vocab,
        allow_only(k, vocab, FORCE_TOKEN),
        &draft_t,
    )
    .await?;

    // Arm 2 — grammar PASSTHROUGH: all-allow ⇒ each row's masked argmax == the
    // model's natural argmax ⇒ the same draft yields a different accept-prefix.
    let passthrough = verify_window(&prompt, k, vocab, allow_all(k, vocab), &draft_t).await?;

    let grammar_forces_accept = forced == draft_t;
    let composition_fires = forced != passthrough;

    // ── Scenario B: multi-step accept-prefix decode LOOP control flow ──
    // Proves the §6.1 spec-decode loop's CONTROL FLOW deterministically (mock):
    // per step, read the sentinel [k]-Token accept-prefix, advance the committed
    // sequence by `n_acc` (the accepted length), re-draft, terminate at budget.
    // The allow-only-`T` mask forces every row's target argmax to `T`, so
    // acceptance depends solely on the supplied drafts (deterministic verdicts).
    // Real acceptance/speedup is a driver gate; the loop wiring is proven here.
    let t = FORCE_TOKEN as i32;
    let w = if FORCE_TOKEN == 2 { 3 } else { 2 } as i32; // a wrong (rejected) draft token
    // A known draft schedule: accept-all, then reject-at-1, then accept-all.
    let schedule: Vec<Vec<i32>> = vec![
        vec![t; k as usize], // → accept k
        {
            let mut d = vec![t; k as usize];
            d[1] = w;
            d
        }, // → accept 1 (reject at row 1)
        vec![t; k as usize], // → accept k
    ];
    let expected_adv: Vec<usize> = vec![k as usize, 1, k as usize];

    let mut committed: Vec<i32> = Vec::new();
    let mut advances: Vec<usize> = Vec::new();
    for drafts in &schedule {
        let accepted =
            verify_window(&prompt, k, vocab, allow_only(k, vocab, FORCE_TOKEN), drafts).await?;
        advances.push(accepted.len());
        committed.extend_from_slice(&accepted);
    }
    // Control-flow verdict: each step advanced by exactly its accepted-prefix
    // length, and the committed stream is the concatenation of accepted tokens
    // (all == FORCE_TOKEN, since the forced mask pins every accepted token to T).
    let loop_advances_ok = advances == expected_adv;
    let loop_commit_ok =
        committed.len() == expected_adv.iter().sum::<usize>() && committed.iter().all(|&x| x == t);
    let loop_ctrl_ok = loop_advances_ok && loop_commit_ok;

    let result = format!(
        "GRAMMAR_FORCES_ACCEPT={grammar_forces_accept} COMPOSITION_FIRES={composition_fires} \
         LOOP_CTRL_OK={loop_ctrl_ok} k={k} force_token={FORCE_TOKEN} forced={forced:?} \
         passthrough={passthrough:?} advances={advances:?} expected_adv={expected_adv:?} \
         committed_len={}",
        committed.len()
    );
    eprintln!("[MTP-VERIFY] {result}");
    Ok(result)
}
