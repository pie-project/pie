//! **MTP Stage-2 — spec-decode, drafts-channel swap of `mtp-native-verify`**
//! (`inferlet::ptir` bridge rewrite). The ORIGINAL classic-WIT design read this
//! window's drafts DEVICE-RESIDENT via a `Binding::MtpDrafts` intrinsic + a
//! driver-side `carrier::next_inputs_drafts` retain/inject command (zero host
//! round-trip on the `[k]` drafts). Both that intrinsic's author-facing eDSL
//! wrapper and the retain/inject WIT command have been REMOVED in the ptir
//! refactor (there is no `intrinsics::mtp_drafts()` in `ptir-dsl`, and no
//! `pipeline_source_kind`/retain surface anywhere in the current `forward`/
//! `pipeline` WIT interfaces) — this is a genuine capability gap, not a stale
//! rename, so it cannot be surgically restored without extending the WIT/driver
//! surface (out of this migration's scope).
//!
//! This inferlet is therefore, for now, IMPLEMENTATION-IDENTICAL to
//! `mtp-native-verify`'s host-round-trip baseline (see that inferlet's module
//! docs for the full dataflow): the `[k+1]` window is a host-writer channel
//! re-`put` each iteration from the host's running commit/draft bookkeeping,
//! and the draft is device-alias read (`toks.read()`, non-consuming peek) off
//! the SAME embedded window tokens. `cuda_mtp_specdecode_ab.rs`'s hard gate is
//! only that both A and B return a finite `mean_accept` and their own name in
//! the result string — both hold here. The PERF/VALUE A-vs-B commentary in that
//! harness (device-resident should avoid B's host round-trip and be ≥ as fast)
//! is no longer a meaningful distinction until `MtpDrafts`/retain functionality
//! is reintroduced on the `inferlet::ptir` surface; that harness is `#[ignore]`d
//! and only soft-prints the delta, so this does not regress any hard assertion.
//!
//! JSON/plain input: optional draft window `k` (default 4).

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const PROMPT: &str = "The quick brown fox jumps over";
const MAX_TOKENS: u32 = 16;
const PAGE_T: u32 = 16;

async fn get_i32(t: inferlet::ptir::Taken) -> Result<Vec<i32>> {
    t.get::<i32>()
        .await
        .map_err(|e| format!("tensor take: {e}"))
}

/// Committed length of a sentinel `[k+1]` tail = the count before the first
/// `-1` (accepted prefix + the bonus at lane `n_acc`), always ≥ 1.
fn committed_len(tail: &[i32]) -> usize {
    tail.iter().take_while(|&&t| t >= 0).count()
}

fn bind_single_sequence(
    pass: &ForwardPass,
    ws: &WorkingSet,
    toks: &Channel,
    kv_len: &Channel,
    token_count: u32,
    pool_pages: u32,
    readout: &[u32],
) -> Result<()> {
    let embed_indptr = Channel::from(vec![0u32, token_count]).named("embed_indptr");
    let positions = Channel::from((0..token_count).collect::<Vec<_>>()).named("positions");
    let pages = Channel::from((0..pool_pages).collect::<Vec<_>>()).named("pages");
    let page_indptr = Channel::from(vec![0u32, token_count.div_ceil(PAGE_T)]).named("page_indptr");
    let w_slot =
        Channel::from((0..token_count).map(|p| p / PAGE_T).collect::<Vec<_>>()).named("w_slot");
    let w_off =
        Channel::from((0..token_count).map(|p| p % PAGE_T).collect::<Vec<_>>()).named("w_off");
    let readout = Channel::from(readout.to_vec()).named("readout");
    pass.embed(toks, &embed_indptr)?;
    pass.readout(&readout)?;
    pass.attention(
        ws,
        ..,
        ..,
        kv_len,
        &pages,
        &page_indptr,
        &w_slot,
        &w_off,
        &positions,
        None,
    )
}

fn bind_verify_rows(
    pass: &ForwardPass,
    ws: &WorkingSet,
    toks: &Channel,
    kv_len: &Channel,
    seq_len: u32,
    rows: u32,
    pool_pages: u32,
) -> Result<()> {
    let embed_indptr = Channel::from((0u32..=rows).collect::<Vec<_>>()).named("embed_indptr");
    let positions = Channel::from((seq_len..seq_len + rows).collect::<Vec<_>>()).named("positions");
    let lengths: Vec<u32> = (1..=rows).map(|row| seq_len + row).collect();
    let mut page_values = Vec::new();
    let mut page_indptr_values = vec![0u32];
    for length in &lengths {
        page_values.extend(0..length.div_ceil(PAGE_T).min(pool_pages));
        page_indptr_values.push(page_values.len() as u32);
    }
    let pages = Channel::from(page_values).named("pages");
    let page_indptr = Channel::from(page_indptr_values).named("page_indptr");
    let w_slot = Channel::from(
        (seq_len..seq_len + rows)
            .map(|p| p / PAGE_T)
            .collect::<Vec<_>>(),
    )
    .named("w_slot");
    let w_off = Channel::from(
        (seq_len..seq_len + rows)
            .map(|p| p % PAGE_T)
            .collect::<Vec<_>>(),
    )
    .named("w_off");
    pass.embed(toks, &embed_indptr)?;
    pass.attention(
        ws,
        ..,
        (seq_len / PAGE_T)..,
        kv_len,
        &pages,
        &page_indptr,
        &w_slot,
        &w_off,
        &positions,
        None,
    )
}

/// Bootstrap fire over `prompt + (k-1)` fillers: yields the seed (row-0 target
/// argmax at the prompt's REAL last position) + the first REAL `[k]` drafts
/// (native MTP argmax) for window 1.
async fn bootstrap(
    ws: &WorkingSet,
    pipeline: &Pipeline,
    prompt: &[u32],
    k: u32,
    max_pages: u32,
) -> Result<(i32, Vec<i32>)> {
    let l = prompt.len() as u32;
    let mut window: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    window.extend(std::iter::repeat(0i32).take((k - 1) as usize));
    let n = l + k - 1;

    let toks = Channel::from(window).named("b_toks");
    let seed_out = Channel::new([1], dtype::i32).named("b_seed");
    let drafts_out = Channel::new([k], dtype::i32).named("b_drafts");

    let readout: Vec<u32> = (0..k).map(|i| l - 1 + i).collect();

    let fwd = ForwardPass::new();
    let kv_len = Channel::from(vec![n]).named("b_kv_len");
    bind_single_sequence(&fwd, ws, &toks, &kv_len, n, max_pages, &readout)?;
    fwd.epilogue(move || {
        let picked = reduce_argmax(intrinsics::logits());
        let seed = gather(&picked, Tensor::constant(vec![0u32]));
        let mtp = intrinsics::mtp_logits(k);
        let drafts = reduce_argmax(mtp);
        seed_out.put(&seed);
        drafts_out.put(&drafts);
    });

    fwd.submit(pipeline)
        .map_err(|e| format!("bootstrap submit: {e}"))?;
    let seed = get_i32(seed_out.take())
        .await?
        .first()
        .copied()
        .ok_or_else(|| "bootstrap: empty seed".to_string())?;
    let drafts = get_i32(drafts_out.take()).await?;
    Ok((seed, drafts))
}

/// One `[k+1]`-wide verify window: embed `[seed, draft]` at positions derived
/// from the pre-envelope `seq_len` cursor, verify `draft`
/// (device-alias peeked off the SAME embedded tokens) against the target's
/// per-row argmax, and draft the NEXT window natively off `mtp_logits`.
/// Returns `(commit [k+1], next_drafts [k])`.
async fn verify_window(
    ws: &WorkingSet,
    pipeline: &Pipeline,
    k: u32,
    seed: i32,
    draft: &[i32],
    seq_len: u32,
    max_pages: u32,
) -> Result<(Vec<i32>, Vec<i32>)> {
    let kp1 = k + 1;
    let mut window: Vec<i32> = vec![seed];
    window.extend_from_slice(draft);

    let toks = Channel::new([kp1], dtype::i32).named("v_toks");
    let commit_out = Channel::new([kp1], dtype::i32).named("v_commit");
    let drafts_out = Channel::new([k], dtype::i32).named("v_drafts");

    let fwd = ForwardPass::new();
    let kv_len =
        Channel::from((1..=kp1).map(|row| seq_len + row).collect::<Vec<_>>()).named("v_kv_len");
    bind_verify_rows(&fwd, ws, &toks, &kv_len, seq_len, kp1, max_pages)?;
    fwd.epilogue(move || {
        let win = toks.read().tensor(); // [k+1] i32 device-alias peek
        let draft_v = gather(&win, Tensor::constant((1..=k).collect::<Vec<u32>>()));
        let picked = reduce_argmax(intrinsics::logits()); // [k+1]
        let head = gather(&picked, iota(k));
        let hit = eq(&head, &draft_v);
        let ones = broadcast(Tensor::constant(1.0f32), [k]);
        let zeros = broadcast(Tensor::constant(0.0f32), [k]);
        let run = cumprod(select(&hit, &ones, &zeros));
        let n_acc = cast(reduce_sum(run), DType::U32);
        let keep = ge(broadcast(&n_acc, [kp1]), iota(kp1));
        let neg1 = broadcast(Tensor::constant(-1i32), [kp1]);
        let commit = select(&keep, &picked, &neg1);

        let mtp = intrinsics::mtp_logits(k);
        let next_drafts = reduce_argmax(mtp);

        commit_out.put(&commit);
        drafts_out.put(&next_drafts);
    });

    toks.put(window);
    fwd.submit(pipeline)
        .map_err(|e| format!("verify submit: {e}"))?;
    let commit = get_i32(commit_out.take()).await?;
    let drafts = get_i32(drafts_out.take()).await?;
    Ok((commit, drafts))
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let k: u32 = input.trim().parse().unwrap_or(4).max(2);
    let ws = WorkingSet::new();

    let mut prompt = wit_model::encode(PROMPT);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let max_pages = (prompt.len() as u32 + MAX_TOKENS + k + 1).div_ceil(PAGE_T);
    ws.reserve(max_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;

    // ONE pipeline for the whole stream (R4-4): the bootstrap and every
    // verify window continue the same sequential decode, so all their fires
    // submit here. The loop is acceptance-driven (the last submit is not
    // knowable at submit time), so the stream ends with a close after the
    // final drain instead of a final-submit marker.
    let pipeline = Pipeline::new();

    let (seed0, draft0) = bootstrap(&ws, &pipeline, &prompt, k, max_pages).await?;
    let mut seq_len: u32 = prompt.len() as u32 + k - 1;

    let mut committed: Vec<u32> = prompt.clone();
    committed.push(seed0 as u32);
    let mut seed = seed0;
    let mut draft = draft0;
    let mut accepted_lengths: Vec<usize> = Vec::new();
    let mut generated: u32 = 1;

    while generated < MAX_TOKENS {
        let (commit, drafts) =
            verify_window(&ws, &pipeline, k, seed, &draft, seq_len, max_pages).await?;
        let clen = committed_len(&commit);
        seq_len += clen as u32;
        let n_acc = clen.saturating_sub(1);
        accepted_lengths.push(n_acc);
        let commit_toks: Vec<u32> = commit.iter().take(clen).map(|&t| t as u32).collect();
        committed.extend(&commit_toks);
        generated += clen.max(1) as u32;

        draft = drafts;
        seed = *committed.last().unwrap_or(&0) as i32;
    }
    // Every window's takes have drained: this close cancels nothing.
    pipeline.close();

    let total_acc: usize = accepted_lengths.iter().sum();
    let steps = accepted_lengths.len();
    let mean_acc = if steps > 0 {
        total_acc as f64 / steps as f64
    } else {
        0.0
    };
    let result = format!(
        "mtp-specdecode: k={k} steps={steps} accepted_lengths={accepted_lengths:?} \
         mean_accept={mean_acc:.2} committed={} (host round-trip drafts — the device-resident \
         MtpDrafts/carrier-retain path is unavailable on the current inferlet::ptir surface; \
         see the module docs)",
        committed.len()
    );
    eprintln!("{result}");
    eprintln!(
        "[mtp-specdecode] committed[{}]={committed:?}",
        committed.len()
    );
    Ok(result)
}
