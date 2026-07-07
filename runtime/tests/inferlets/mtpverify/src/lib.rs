//! **§6.1 MTP assembly — composed spec-verify ⟂ PER-POSITION grammar mask** (bravo),
//! on **DIRECT WIT bindings** (In Gim's directive): raw `inference::ForwardPass`
//! + `working_set::KvWorkingSet` + an authored `tensor::Program`, no
//! `Context`/`Generator`/`collect_tokens` sugar.
//!
//! §6.1 composes MTP speculation with the grammar constraint on ONE forward:
//! draft K tokens, verify them against the target's per-row argmax, and apply a
//! grammar mask **per speculative position** (the grammar state advances per
//! token, so each of the K rows has its OWN allowed-token set). The packed
//! `mask_apply` op broadcasts ONE mask over every row and so CANNOT express
//! per-position masking (echo's design constraint); the per-position mask is
//! instead a `[k, vocab]` bool matrix applied by `select`:
//!   `masked = select(allow[k,vocab], logits[k,vocab], −∞)` → per-row `argmax`.
//!
//! This inferlet fires that fused program (per-position `select`-mask + spec-verify
//! DAG `argmax → eq → cumprod → select` → sentinel-coded `[k]`-Token) and proves
//! the grammar constraint drives the spec-verify outcome, deterministically (no
//! prior greedy decode needed):
//!   - `GRAMMAR_FORCES_ACCEPT` — an allow-only-`T` mask (at every position) forces
//!     each row's masked argmax to `T`, so a draft `[T; k]` is accepted in FULL
//!     (`[T; k]`), independent of the model's natural argmax;
//!   - `COMPOSITION_FIRES` — an all-allow mask yields a DIFFERENT accept-prefix for
//!     the same draft (the mask changed the verify result — constraint ⟂
//!     speculation, not a passthrough).
//!
//! JSON/plain input: an optional draft-window size `k` (default 4), e.g. `"6"`.

use inferlet::inference::ForwardPass;
use inferlet::program::{encode_i32, resolve_bindings};
use inferlet::sampling::{DType, Graph, OutputKind, Readiness, dselect};
use inferlet::working_set::KvWorkingSet;
use inferlet::{Result, model, tensor};

const PROMPT: &str = "The quick brown fox jumps over";
/// The grammar-forced token: the allow-only-`T` per-position mask pins each row's
/// masked argmax here.
const FORCE_TOKEN: u32 = 1;

/// A `[k, vocab]` bool mask (1 byte / element, per the eval's `Value::Bool`) with
/// EVERY token allowed at every position.
fn allow_all(k: u32, vocab: u32) -> Vec<u8> {
    vec![1u8; (k * vocab) as usize]
}

/// A `[k, vocab]` bool mask allowing ONLY token `t` at every position (all other
/// logits → −∞ under the `select`, so each row's masked argmax is forced to `t`).
fn allow_only(k: u32, vocab: u32, t: u32) -> Vec<u8> {
    let mut m = vec![0u8; (k * vocab) as usize];
    for r in 0..k {
        m[(r * vocab + t) as usize] = 1;
    }
    m
}

/// The composed program + the host-input keys the caller binds per fire.
struct Composed {
    program: tensor::Program,
    bindings: Vec<inferlet::sampling::ir::Binding>,
    host_inputs: Vec<inferlet::program::HostInputDecl>,
    mask_key: u32,
    draft_key: u32,
}

/// Build the composed **spec-verify ⟂ per-position grammar** program: `[k, vocab]`
/// target logits, a `[k, vocab]` bool allow-mask (one row per speculative
/// position) applied by `select`, per-row `argmax`, verify the `[k]` submit draft
/// (`eq → cumprod` prefix-AND) → sentinel-coded `[k]`-Token (accepted prefix, then
/// `-1`). Per-position masking via `select` (NOT the packed `mask_apply`, which
/// broadcasts a single mask over all rows — echo's design constraint).
fn spec_verify_grammar(vocab: u32, k: u32) -> Result<Composed> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_matrix_dyn(k); // [k, vocab] target
    let allow = g.host_matrix_dyn(DType::Bool, k, vocab, Readiness::Submit); // [k, vocab] per-position
    let draft = g.host_vector_dyn(DType::I32, k, Readiness::Submit); // [k] drafts

    let neg_inf = g.constant_f32_dyn(f32::NEG_INFINITY).broadcast_matrix(k, vocab);
    let masked = dselect(&allow, &logits, &neg_inf); // per-position mask [k, vocab]
    let target = masked.argmax(); // [k] grammar-constrained per-row argmax
    let matched = target.eq(&draft); // [k] bool

    let ones = g.constant_f32_dyn(1.0).broadcast_vec(k);
    let zeros = g.constant_f32_dyn(0.0).broadcast_vec(k);
    let keep = dselect(&matched, &ones, &zeros).cumprod().gt(&g.constant_f32_dyn(0.5));
    let neg1 = g.constant_i32_dyn(-1).broadcast_vec(k);
    let out = dselect(&keep, &draft, &neg1); // accepted prefix, then -1
    g.output(&out, OutputKind::Token);

    let mask_key = allow.input_key().ok_or("allow-mask host input key")?;
    let draft_key = draft.input_key().ok_or("draft host input key")?;
    let built = g.build().map_err(|e| format!("compose build: {e:?}"))?;
    let program =
        inferlet::emit::emit_program(&built.program).map_err(|e| format!("emit: {e}"))?;
    Ok(Composed {
        program,
        bindings: built.bindings,
        host_inputs: built.host_inputs,
        mask_key,
        draft_key,
    })
}

/// Fire ONE `[k, vocab]` verify window (fresh KV) over `PROMPT` + `k-1` fillers,
/// binding the `[k, vocab]` bool allow-mask + `[k]` draft, and read the
/// sentinel-coded `[k]`-Token accept-prefix (truncated at the first `-1`).
async fn verify_window(
    prog: &Composed,
    prompt: &[u32],
    k: u32,
    mask_bytes: &[u8],
    draft: &[i32],
) -> Result<Vec<i32>> {
    // input = prompt + (k-1) fillers ⇒ the last k positions carry the k logit rows.
    let l = prompt.len() as u32;
    let mut input = prompt.to_vec();
    input.extend(std::iter::repeat(0u32).take((k - 1) as usize));
    let n = input.len() as u32;
    let positions: Vec<u32> = (0..n).collect();
    let logits_positions: Vec<u32> = (0..k).map(|i| l - 1 + i).collect();

    let kv = KvWorkingSet::new();
    let page = kv.page_size();
    let total_pages = n.div_ceil(page);
    kv.alloc(total_pages).map_err(|e| format!("alloc: {e}"))?;

    let pass = ForwardPass::new();
    // Fresh prefill: read no prior context, write all `total_pages` pages.
    pass.kv_working_set(&kv, 0, 0, 0, 0, total_pages, 0);
    pass.input_tokens(&input, &positions);

    let submit = vec![
        (prog.mask_key, mask_bytes.to_vec()),
        (prog.draft_key, encode_i32(draft)),
    ];
    let bindings = resolve_bindings(&prog.bindings, &prog.host_inputs, &logits_positions, &submit)?;
    pass.sampler(&prog.program, bindings);
    pass.execute();

    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let k: u32 = input.trim().parse().unwrap_or(4).max(2);
    let vocab = model::output_vocab_size();
    let mut prompt = model::encode(PROMPT);
    if prompt.is_empty() {
        prompt.push(0);
    }

    let prog = spec_verify_grammar(vocab, k)?;
    let draft: Vec<i32> = vec![FORCE_TOKEN as i32; k as usize];

    // Arm 1 — grammar-FORCED: allow only FORCE_TOKEN at every position ⇒ every
    // row's masked argmax == FORCE_TOKEN ⇒ the draft [FORCE_TOKEN; k] accepts in FULL.
    let forced =
        verify_window(&prog, &prompt, k, &allow_only(k, vocab, FORCE_TOKEN), &draft).await?;

    // Arm 2 — grammar PASSTHROUGH: all-allow ⇒ each row's masked argmax == the
    // model's natural argmax ⇒ the same draft yields a different accept-prefix.
    let passthrough = verify_window(&prog, &prompt, k, &allow_all(k, vocab), &draft).await?;

    let grammar_forces_accept = forced == draft;
    let composition_fires = forced != passthrough;

    // ── Scenario B: multi-step accept-prefix decode LOOP control flow ──
    // Proves the §6.1 spec-decode loop's CONTROL FLOW deterministically (mock):
    // per step, read the sentinel `[k]`-Token accept-prefix, advance the committed
    // sequence by `n_acc` (the accepted length), re-draft, terminate at budget. The
    // allow-only-`T` mask forces every row's target argmax to `T`, so acceptance
    // depends solely on the supplied drafts (deterministic verdicts). Real
    // acceptance/speedup is a driver gate; the loop wiring is proven here.
    let forced_mask = allow_only(k, vocab, FORCE_TOKEN);
    let t = FORCE_TOKEN as i32;
    let w = if FORCE_TOKEN == 2 { 3 } else { 2 } as i32; // a wrong (rejected) draft token
    // A known draft schedule: accept-all, then reject-at-1, then accept-all.
    let schedule: Vec<Vec<i32>> = vec![
        vec![t; k as usize],                                  // → accept k
        { let mut d = vec![t; k as usize]; d[1] = w; d },     // → accept 1 (reject at row 1)
        vec![t; k as usize],                                  // → accept k
    ];
    let expected_adv: Vec<usize> = vec![k as usize, 1, k as usize];

    let mut committed: Vec<i32> = Vec::new();
    let mut advances: Vec<usize> = Vec::new();
    for drafts in &schedule {
        let accepted = verify_window(&prog, &prompt, k, &forced_mask, drafts).await?;
        advances.push(accepted.len());
        committed.extend_from_slice(&accepted);
    }
    // Control-flow verdict: each step advanced by exactly its accepted prefix
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
