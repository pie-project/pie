//! **MTP Stage 2 — PTIR-native explicit draft→verify→accept** (bravo), on the
//! `inferlet::ptir` bridge. The speculative decode is a single traced eDSL
//! epilogue per window — the target VERIFY (match + bonus tail) AND the next
//! window's DRAFTING (native MTP argmax) in ONE lowered graph:
//!
//!   `picked = logits[k+1, vocab].argmax()`         // [k+1] target greedy (k verify + bonus)
//!   `head   = gather(picked, lanes_k)`              // picked[0..k]
//!   `hit    = head.eq(draft)`                       // [k] bool — verify vs the EMBEDDED drafts
//!   `n_acc  = reduce_sum(cumprod(hit))`              // scalar accepted count 0..k
//!   `keep   = broadcast(n_acc).ge(lanes_k1)`          // [k+1] i <= n_acc
//!   commit  = select(keep, picked, -1)                // accepted prefix + BONUS@n_acc, then -1
//!   drafts' = argmax(mtp_logits(k))                   // [k] FRESH drafts — NEXT window's proposals
//!
//! `draft` is read DEVICE-ALIAS off the SAME embedded window tokens (`toks.read()`,
//! non-consuming peek — rows `1..=k`, the previous step's MTP proposals fed as
//! this window's input), not a separately submitted draft channel; each fresh
//! pass seeds its `toks` channel from the host's running commit/draft bookkeeping
//! (the ptir-bridge equivalent of the deleted
//! `sampling::program::mtp_native_verify` + `resolve_bindings` blob-submit
//! surface — there is no lower-level "device-resident retain" primitive on the
//! current `inferlet::ptir` surface, so drafts round-trip through host state
//! each iteration, exactly mirroring this inferlet's ORIGINAL host-round-trip
//! dataflow, contrasted with `mtp-specdecode`'s attempted device residency).
//!
//! Bootstrap fire: `prompt + (k-1)` fillers ⇒ the last k positions carry k
//! logit rows ⇒ both `logits` (target) and `mtp_logits` (drafts) get REAL k-row
//! data (no anchor-row collapse) — yields the seed (row-0 target argmax) + the
//! first REAL drafts (mtp argmax) for window 1.
//!
//! **The accept step is fold-commit, because this is a LINEAR model.** The
//! verify fire cannot fold: how many of its k drafts are real is decided by
//! its own logits, and a fold is irreversible — there is no recurrent-state
//! equivalent of dropping a KV slot. So each window runs in two fires:
//!
//!   1. verify — `fold_len = 0`: the window's pre-recurrence activations land
//!      in the RS buffer and the folded boundary does not move, leaving the
//!      whole window abandonable while its logits are inspected.
//!   2. commit — `fold_len = clen`: with the accepted length now known, the
//!      boundary advances through exactly the accepted prefix plus its bonus
//!      token, replaying the buffered activations instead of recomputing the
//!      in-projection. Freeing the remaining slabs abandons the rejected tail,
//!      which never touched the recurrent state.
//!
//! The window is also ONE request row of `k+1` causal tokens, not the `k+1`
//! -request staircase this inferlet used to build. The two are equivalent for
//! attention, but a linear model carries one recurrent state per REQUEST, so
//! the staircase would demand `k+1` working sets holding divergent copies of a
//! single sequence's state.
//!
//! JSON/plain input: optional draft window `k` (default 4).

use inferlet::ptir::hybrid::prelude::*;
use inferlet::{Result, model as wit_model};

const PROMPT: &str = "The quick brown fox jumps over";
const MAX_TOKENS: u32 = 16;
const PAGE_T: u32 = 16;

/// Decode a `[k]`/`[k+1]` i32 host vector.
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

/// One request row of `count` tokens starting at absolute position
/// `first_pos`, reading out the rows named by `readout` (row indices within
/// the window).
///
/// This replaces the old `k+1`-REQUEST staircase. On a pure-attention model
/// the staircase and a single causal row are equivalent — row `i` saw
/// `seq_len + i` keys either way — but a linear model cannot express the
/// staircase at all: it carries one recurrent state per REQUEST, so `k+1`
/// rows would need `k+1` working sets holding `k+1` divergent copies of one
/// sequence's state. One row of `k+1` causal tokens is the shape that has a
/// single state to advance, which is the shape the buffer is built for.
fn bind_window(
    pass: &ForwardPass,
    ws: &WorkingSet,
    toks: &Channel,
    kv_len: &Channel,
    first_pos: u32,
    count: u32,
    pool_pages: u32,
    readout: &[u32],
) -> Result<()> {
    let embed_indptr = Channel::from(vec![0u32, count]).named("w_embed_indptr");
    let positions =
        Channel::from((first_pos..first_pos + count).collect::<Vec<_>>()).named("w_positions");
    let pages = Channel::from((0..pool_pages).collect::<Vec<_>>()).named("w_pages");
    let page_indptr =
        Channel::from(vec![0u32, (first_pos + count).div_ceil(PAGE_T).min(pool_pages)])
            .named("w_page_indptr");
    let w_slot = Channel::from(
        (first_pos..first_pos + count)
            .map(|p| p / PAGE_T)
            .collect::<Vec<_>>(),
    )
    .named("w_w_slot");
    let w_off = Channel::from(
        (first_pos..first_pos + count)
            .map(|p| p % PAGE_T)
            .collect::<Vec<_>>(),
    )
    .named("w_w_off");
    // A fold fire reads nothing, and `Channel::from(vec![])` cannot express
    // that (an empty shape has no vector length). Omitting `readout` entirely
    // is the honest encoding of "this fire samples no rows" — and the driver
    // requires it, since a fold returns before the output projection.
    if !readout.is_empty() {
        let readout = Channel::from(readout.to_vec()).named("w_readout");
        pass.readout(&readout)?;
    }
    pass.embed(toks, &embed_indptr)?;
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

/// The five `rs-geometry` buffer-addressing channels for a span of `count`
/// buffered tokens starting at buffer token 0, on one request row.
fn rs_span(rs_page: u32, count: u32, tag: &str) -> (Channel, Channel, Channel, Channel, Channel) {
    let pages_n = count.div_ceil(rs_page);
    (
        Channel::from(vec![count]).named(&format!("{tag}_rs_len")),
        Channel::from((0..pages_n).collect::<Vec<_>>()).named(&format!("{tag}_rs_pages")),
        Channel::from(vec![0u32, pages_n]).named(&format!("{tag}_rs_indptr")),
        Channel::from((0..count).map(|t| t / rs_page).collect::<Vec<_>>())
            .named(&format!("{tag}_rs_w_slot")),
        Channel::from((0..count).map(|t| t % rs_page).collect::<Vec<_>>())
            .named(&format!("{tag}_rs_w_off")),
    )
}

/// Bootstrap fire over `prompt + (k-1)` fillers: yields the seed (row-0 target
/// argmax at the prompt's REAL last position) + the first REAL `[k]` drafts
/// (native MTP argmax) for window 1. No verify (nothing to verify yet).
async fn bootstrap(
    ws: &WorkingSet,
    rs: &RsWorkingSet,
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

    // k read-out rows (the last k positions) ⇒ intrinsics::logits() AND
    // intrinsics::mtp_logits(k) both declare [k, vocab] — real k-row data.
    let readout: Vec<u32> = (0..k).map(|i| l - 1 + i).collect();

    let fwd = ForwardPass::new();
    let kv_len = Channel::from(vec![n]).named("b_kv_len");
    bind_single_sequence(&fwd, ws, &toks, &kv_len, n, max_pages, &readout)?;
    fwd.recurrent(std::slice::from_ref(rs))?;
    fwd.epilogue(move || {
        let picked = reduce_argmax(intrinsics::logits()); // [k] target argmax
        let seed = gather(&picked, Tensor::constant(vec![0u32])); // [1] row-0
        let mtp = intrinsics::mtp_logits(k); // [k, vocab]
        let drafts = reduce_argmax(mtp); // [k] fresh drafts
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

/// One `[k+1]`-wide verify window: embed `[seed, draft]` (a freshly seeded
/// channel each iteration) at positions derived from the pre-envelope
/// `seq_len` cursor, verify `draft` (device-alias
/// peeked off the SAME embedded tokens) against the target's per-row argmax,
/// and draft the NEXT window natively off `mtp_logits`. Returns `(commit
/// [k+1], next_drafts [k])`.
#[allow(clippy::too_many_arguments)]
async fn verify_window(
    ws: &WorkingSet,
    rs: &RsWorkingSet,
    pipeline: &Pipeline,
    k: u32,
    rs_page: u32,
    seed: i32,
    draft: &[i32],
    seq_len: u32,
    max_pages: u32,
) -> Result<(Vec<i32>, Vec<i32>)> {
    let kp1 = k + 1;
    let mut window: Vec<i32> = vec![seed];
    window.extend_from_slice(draft);

    let toks = Channel::from(window).named("v_toks");
    let commit_out = Channel::new([kp1], dtype::i32).named("v_commit");
    let drafts_out = Channel::new([k], dtype::i32).named("v_drafts");

    let fwd = ForwardPass::new();
    let kv_len = Channel::from(vec![seq_len + kp1]).named("v_kv_len");
    let readout: Vec<u32> = (0..kp1).collect();
    bind_window(&fwd, ws, &toks, &kv_len, seq_len, kp1, max_pages, &readout)?;

    // BUFFER, do not fold. The verify window is `seed + k drafts`, and how
    // many of those k are real is not known until this fire's own logits come
    // back. Folding them would advance the recurrent state through a tail that
    // may be rejected, and a fold is irreversible — unlike KV, there are no
    // slots to discard. So the window's pre-recurrence activations are parked
    // in the buffer with the folded boundary held still, and the NEXT fire
    // moves it through exactly the accepted prefix.
    let (rs_len, rs_pages, rs_indptr, rs_w_slot, rs_w_off) = rs_span(rs_page, kp1, "v");
    let fold_len = Channel::from(vec![0u32]).named("v_fold_len");
    fwd.recurrent_with(
        std::slice::from_ref(rs),
        &fold_len,
        ..,
        ..,
        &rs_len,
        &rs_pages,
        &rs_indptr,
        &rs_w_slot,
        &rs_w_off,
    )
    .map_err(|e| format!("verify recurrent binding: {e}"))?;
    fwd.epilogue(move || {
        // Device-alias read: peek the embedded window (NOT a resubmitted draft
        // channel) and gather rows 1..=k as the verify operand.
        let win = toks.read().tensor(); // [k+1] i32
        let draft_v = gather(&win, Tensor::constant((1..=k).collect::<Vec<u32>>())); // [k]
        let picked = reduce_argmax(intrinsics::logits()); // [k+1] target (k verify + bonus)
        let head = gather(&picked, iota(k)); // [k] picked[0..k]
        let hit = eq(&head, &draft_v); // [k] bool
        let ones = broadcast(Tensor::constant(1.0f32), [k]);
        let zeros = broadcast(Tensor::constant(0.0f32), [k]);
        let run = cumprod(select(&hit, &ones, &zeros)); // [k]
        let n_acc = cast(reduce_sum(run), DType::U32); // accepted-prefix length
        let keep = ge(broadcast(&n_acc, [kp1]), iota(kp1)); // [k+1] i <= n_acc
        let neg1 = broadcast(Tensor::constant(-1i32), [kp1]);
        let commit = select(&keep, &picked, &neg1); // accepted prefix + bonus + -1s

        let mtp = intrinsics::mtp_logits(k); // [k, vocab]
        let next_drafts = reduce_argmax(mtp); // [k] fresh drafts — NEXT window

        commit_out.put(&commit);
        drafts_out.put(&next_drafts);
    });

    fwd.submit(pipeline)
        .map_err(|e| format!("verify submit: {e}"))?;
    let commit = get_i32(commit_out.take()).await?;
    let drafts = get_i32(drafts_out.take()).await?;
    Ok((commit, drafts))
}

/// Move the folded boundary through the `clen` accepted tokens the verify
/// window buffered, then drop whatever it did not reach.
///
/// `window_prefix` is the first `clen` tokens of the VERIFY WINDOW — not the
/// tokens the verify predicted. The recurrent side ignores them entirely and
/// replays the buffered activations instead, which is the whole point: the
/// expensive in-projection is not recomputed. They matter only to the KV side,
/// which re-runs the same span this window already wrote and must therefore
/// write back the same values. Feeding the predicted tokens here would shift
/// the sequence by one and corrupt the accepted prefix's KV.
///
/// Nothing here is read, and nothing here CAN be read: with a fold boundary
/// set the linear layers stop after the recurrence and never reach the output
/// projection, so the driver refuses a fold fire that declares sample rows
/// ("buffered RS fold is state-only and cannot sample logits"). The fire is
/// therefore submitted with an empty readout and no epilogue, and it is not
/// awaited — fires on one pipeline are ordered, so the next window already
/// observes the advanced boundary, and any driver failure surfaces as channel
/// poison on that window's first read.
#[allow(clippy::too_many_arguments)]
async fn commit_window(
    ws: &WorkingSet,
    rs: &RsWorkingSet,
    pipeline: &Pipeline,
    rs_page: u32,
    window_prefix: &[i32],
    seq_len: u32,
    max_pages: u32,
) -> Result<()> {
    let clen = window_prefix.len() as u32;
    if clen == 0 {
        return Ok(());
    }
    let toks = Channel::from(window_prefix.to_vec()).named("c_toks");

    let fwd = ForwardPass::new();
    let kv_len = Channel::from(vec![seq_len + clen]).named("c_kv_len");
    bind_window(&fwd, ws, &toks, &kv_len, seq_len, clen, max_pages, &[])?;

    let (rs_len, rs_pages, rs_indptr, rs_w_slot, rs_w_off) = rs_span(rs_page, clen, "c");
    let fold_len = Channel::from(vec![clen]).named("c_fold_len");
    fwd.recurrent_with(
        std::slice::from_ref(rs),
        &fold_len,
        ..,
        ..,
        &rs_len,
        &rs_pages,
        &rs_indptr,
        &rs_w_slot,
        &rs_w_off,
    )
    .map_err(|e| format!("commit recurrent binding: {e}"))?;
    // A stage-less pass has no PTIR program to register at all, which the
    // driver rejects. An EMPTY epilogue is the minimal well-formed program: it
    // samples nothing, so the fold fire declares no readout rows.
    fwd.epilogue(|| {});
    fwd.submit(pipeline)
        .map_err(|e| format!("commit submit: {e}"))?;
    Ok(())
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let k: u32 = input.trim().parse().unwrap_or(4).max(2);
    if !wit_model::is_linear() {
        // The buffer this inferlet's accept step depends on only exists on a
        // linear model. On a pure-attention model a rejected tail is discarded
        // by dropping KV slots and none of the fold-commit machinery applies.
        return Ok("skipped: mtp-native-verify needs a linear model".to_string());
    }
    let ws = WorkingSet::new();
    let rs = RsWorkingSet::new();
    let rs_page = rs.buffer_page_size();
    if rs_page == 0 {
        return Err("linear model reports no RS buffer page size".to_string());
    }
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

    // Bootstrap: real seed + real first drafts off the prompt's REAL last position.
    let (seed0, draft0) = bootstrap(&ws, &rs, &pipeline, &prompt, k, max_pages).await?;
    let mut seq_len: u32 = prompt.len() as u32 + k - 1;

    let mut committed: Vec<u32> = prompt.clone();
    committed.push(seed0 as u32);
    let mut seed = seed0;
    let mut draft = draft0;
    let mut accepted_lengths: Vec<usize> = Vec::new();
    let mut generated: u32 = 1;

    // North-star spec-decode loop: verify the embedded drafts against the
    // target with the fold SUPPRESSED → fold exactly the accepted prefix and
    // abandon the rest → take the fresh native-MTP drafts as the NEXT
    // window's proposals → repeat.
    let window_slabs = (k + 1).div_ceil(rs_page);
    while generated < MAX_TOKENS {
        // Each window buffers k+1 tokens and ends with an empty buffer, so
        // the slabs are reserved per window rather than held across the loop.
        // They must be reserved fresh: appending onto a buffer that still held
        // the previous window's tail would ask the recurrence to read what is
        // already buffered, which it cannot do.
        rs.alloc_buffer(window_slabs)
            .map_err(|e| format!("rs.alloc_buffer: {e}"))?;

        let (commit, drafts) = verify_window(
            &ws, &rs, &pipeline, k, rs_page, seed, &draft, seq_len, max_pages,
        )
        .await?;
        let clen = committed_len(&commit); // n_acc accepted + 1 bonus (≥ 1)
        let n_acc = clen.saturating_sub(1);
        accepted_lengths.push(n_acc);
        let commit_toks: Vec<u32> = commit.iter().take(clen).map(|&t| t as u32).collect();

        // Only NOW is the accepted length known, so only now can the folded
        // boundary move. It advances through exactly `clen` of the k+1
        // buffered tokens; the rejected tail is abandoned by freeing its
        // slabs, having never touched the recurrent state.
        //
        // The commit re-runs the WINDOW's first `clen` tokens, which is the
        // span the verify fire already wrote KV for. `commit_toks` is the
        // window shifted by one (each entry is what the target predicted
        // AFTER the corresponding window token), so feeding it here would
        // corrupt exactly the prefix being committed.
        let window_prefix: Vec<i32> = std::iter::once(seed)
            .chain(draft.iter().copied())
            .take(clen)
            .collect();
        commit_window(&ws, &rs, &pipeline, rs_page, &window_prefix, seq_len, max_pages).await?;
        let remaining = rs.buffer_size();
        if remaining > 0 {
            rs.free_buffer(&(0..remaining).collect::<Vec<_>>())
                .map_err(|e| format!("free_buffer: {e}"))?;
        }

        seq_len += clen as u32;
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
        "mtp-native-verify: k={k} steps={steps} accepted_lengths={accepted_lengths:?} \
         mean_accept={mean_acc:.2} committed={} (PTIR-native verify+draft: verify vs embedded \
         drafts, next-drafts from mtp_logits argmax, [k+1] bonus tail, all traced)",
        committed.len()
    );
    eprintln!("{result}");
    Ok(result)
}
