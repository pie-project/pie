//! **FOLD-COMMIT on a linear model** — the speculative-commit shape
//! `model.wit`'s `is-linear()` exists to select.
//!
//! A linear/SSM model folds tokens into its recurrent state IRREVERSIBLY, so
//! the KV trick of discarding rejected slots does not exist for it. The
//! answer the runtime declares is fold-commit: run the uncertain tokens with
//! the fold SUPPRESSED, holding their pre-recurrence activations in the RS
//! working set's buffered slots, and afterwards fold only the accepted
//! prefix. A rejected tail is simply never folded — abandoning it costs one
//! `free-buffer`.
//!
//! This inferlet drives all three modes end to end:
//!
//!  1. **prefill — `fold`** (the default): materializes the folded state.
//!     Buffering and folding both read it, so this must come first.
//!  2. **speculate — `fold_len = 0`**: one chunk of `SPEC_TOKENS` tokens
//!     appended to the buffer; the folded boundary does not move, so the
//!     chunk stays abandonable.
//!  3. **commit — `fold_len = accepted`**: advances the boundary through the
//!     accepted prefix and drops the fully covered head slots.
//!
//! Input: the number of speculative tokens to accept (default all of them),
//! e.g. `"2"`. Output names the three phases so a harness can assert them.

use inferlet::ptir::hybrid::prelude::*;
use inferlet::{Result, model as wit_model};

/// Tokens written in the buffered (speculative) chunk.
const SPEC_TOKENS: u32 = 4;

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let accepted: u32 = input.trim().parse().unwrap_or(SPEC_TOKENS).min(SPEC_TOKENS);

    if !wit_model::is_linear() {
        return Ok("skipped: fold-commit needs a linear model".to_string());
    }

    let ws = WorkingSet::new();
    let page_size = ws.page_size();
    let rs = RsWorkingSet::new();
    let buffer_page = rs.buffer_page_size();
    if buffer_page == 0 {
        return Err("linear model reports no RS buffer page size".to_string());
    }

    let prompt = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let n = prompt.len() as u32;
    let max_pages = (n + SPEC_TOKENS + 1).div_ceil(page_size);
    ws.reserve(max_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;

    // Buffered slots for the speculative chunk. Reserved logically here;
    // the buffering fire materializes them on first write.
    let slabs = SPEC_TOKENS.div_ceil(buffer_page);
    rs.alloc_buffer(slabs)
        .map_err(|e| format!("rs.alloc_buffer: {e}"))?;

    // ───────────────── 1. PREFILL — mode `fold` (default) ─────────────────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Channel::from(vec![0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from((0..n).collect::<Vec<_>>()).named("positions_p");
    let pages_p = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages_p");
    let page_indptr_p = Channel::from(vec![0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p =
        Channel::from((0..n).map(|p| p / page_size).collect::<Vec<_>>()).named("w_slot_p");
    let w_off_p = Channel::from((0..n).map(|p| p % page_size).collect::<Vec<_>>()).named("w_off_p");
    let g0_ch = Channel::new([1], dtype::i32).named("g0");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    let kv_len_p = Channel::from(vec![n]).named("kv_len_p");
    fwd_p.attention(
        &ws,
        ..,
        ..,
        &kv_len_p,
        &pages_p,
        &page_indptr_p,
        &w_slot_p,
        &w_off_p,
        &positions_p,
        None,
    )?;
    fwd_p.recurrent(std::slice::from_ref(&rs))?;
    fwd_p.epilogue(move || {
        let t = reduce_argmax(intrinsics::logits());
        g0_ch.put(&t);
    });

    let pipe = Pipeline::new();
    fwd_p
        .submit(&pipe)
        .map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("g0 take: {e}"))?[0];

    // ────────────── 2. SPECULATE — `fold_len = 0`, nothing folds ──────────
    // One SPEC_TOKENS-wide fire. Its activations land in the buffered slots;
    // the folded state does not move, so nothing here is committed yet.
    let spec_toks = Channel::from(vec![g0; SPEC_TOKENS as usize]).named("spec_toks");
    let spec_indptr = Channel::from(vec![0u32, SPEC_TOKENS]).named("spec_indptr");
    let spec_positions =
        Channel::from((n..n + SPEC_TOKENS).collect::<Vec<_>>()).named("spec_positions");
    let spec_pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("spec_pages");
    let spec_page_indptr =
        Channel::from(vec![0u32, (n + SPEC_TOKENS).div_ceil(page_size)]).named("spec_page_indptr");
    let spec_w_slot = Channel::from(
        (n..n + SPEC_TOKENS)
            .map(|p| p / page_size)
            .collect::<Vec<_>>(),
    )
    .named("spec_w_slot");
    let spec_w_off = Channel::from(
        (n..n + SPEC_TOKENS)
            .map(|p| p % page_size)
            .collect::<Vec<_>>(),
    )
    .named("spec_w_off");
    let spec_out = Channel::new([1], dtype::i32).named("spec_out");

    let fwd_s = ForwardPass::new();
    fwd_s.embed(&spec_toks, &spec_indptr)?;
    let spec_kv_len = Channel::from(vec![n + SPEC_TOKENS]).named("spec_kv_len");
    fwd_s.attention(
        &ws,
        ..,
        ..,
        &spec_kv_len,
        &spec_pages,
        &spec_page_indptr,
        &spec_w_slot,
        &spec_w_off,
        &spec_positions,
        None,
    )?;
    // The buffered geometry: `SPEC_TOKENS` tokens appended at the buffer tail,
    // page-major from slab zero. `fold_len = 0` holds the folded boundary
    // still, so nothing here is committed -- that is what makes this fire
    // abandonable.
    let rs_page = inferlet::model::rs_buffer_page_size().max(1);
    let rs_pages = SPEC_TOKENS.div_ceil(rs_page);
    let spec_rs_len = Channel::from(vec![SPEC_TOKENS]).named("spec_rs_len");
    let spec_rs_pages = Channel::from((0..rs_pages).collect::<Vec<_>>()).named("spec_rs_pages");
    let spec_rs_indptr = Channel::from(vec![0u32, rs_pages]).named("spec_rs_indptr");
    let spec_rs_w_slot =
        Channel::from((0..SPEC_TOKENS).map(|t| t / rs_page).collect::<Vec<_>>())
            .named("spec_rs_w_slot");
    let spec_rs_w_off = Channel::from((0..SPEC_TOKENS).map(|t| t % rs_page).collect::<Vec<_>>())
        .named("spec_rs_w_off");
    let spec_fold_len = Channel::from(vec![0u32]).named("spec_fold_len");
    fwd_s
        .recurrent_with(
            std::slice::from_ref(&rs),
            &spec_fold_len,
            ..,
            ..,
            &spec_rs_len,
            &spec_rs_pages,
            &spec_rs_indptr,
            &spec_rs_w_slot,
            &spec_rs_w_off,
        )
        .map_err(|e| format!("speculative recurrent binding: {e}"))?;
    fwd_s.epilogue(move || {
        let t = reduce_argmax(intrinsics::logits());
        spec_out.put(&t);
    });
    fwd_s
        .submit(&pipe)
        .map_err(|e| format!("speculative submit: {e}"))?;
    let drafted = spec_out
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("spec_out take: {e}"))?[0];

    // ─────────────── 3. COMMIT — `fold_len = accepted` ────────────────────
    // Replays only the accepted prefix into the folded state. No logits: the
    // driver runs the recurrent layers alone.
    let mut committed = 0u32;
    if accepted > 0 {
        let commit_toks = Channel::from(vec![g0; accepted as usize]).named("commit_toks");
        let commit_indptr = Channel::from(vec![0u32, accepted]).named("commit_indptr");
        let commit_positions =
            Channel::from((n..n + accepted).collect::<Vec<_>>()).named("commit_positions");
        let commit_pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("commit_pages");
        let commit_page_indptr = Channel::from(vec![0u32, (n + accepted).div_ceil(page_size)])
            .named("commit_page_indptr");
        let commit_w_slot =
            Channel::from((n..n + accepted).map(|p| p / page_size).collect::<Vec<_>>())
                .named("commit_w_slot");
        let commit_w_off =
            Channel::from((n..n + accepted).map(|p| p % page_size).collect::<Vec<_>>())
                .named("commit_w_off");
        let commit_done = Channel::new([1], dtype::i32).named("commit_done");

        let fwd_c = ForwardPass::new();
        fwd_c.embed(&commit_toks, &commit_indptr)?;
        let commit_kv_len = Channel::from(vec![n + accepted]).named("commit_kv_len");
        fwd_c.attention(
            &ws,
            ..,
            ..,
            &commit_kv_len,
            &commit_pages,
            &commit_page_indptr,
            &commit_w_slot,
            &commit_w_off,
            &commit_positions,
            None,
        )?;
        // The commit replays `accepted` buffered tokens, so it reads exactly
        // the span the speculate fire wrote.
        let commit_rs_pages_n = accepted.div_ceil(rs_page);
        let commit_rs_len = Channel::from(vec![accepted]).named("commit_rs_len");
        let commit_rs_pages =
            Channel::from((0..commit_rs_pages_n).collect::<Vec<_>>()).named("commit_rs_pages");
        let commit_rs_indptr =
            Channel::from(vec![0u32, commit_rs_pages_n]).named("commit_rs_indptr");
        let commit_rs_w_slot =
            Channel::from((0..accepted).map(|t| t / rs_page).collect::<Vec<_>>())
                .named("commit_rs_w_slot");
        let commit_rs_w_off =
            Channel::from((0..accepted).map(|t| t % rs_page).collect::<Vec<_>>())
                .named("commit_rs_w_off");
        let commit_fold_len = Channel::from(vec![accepted]).named("commit_fold_len");
        fwd_c
            .recurrent_with(
                std::slice::from_ref(&rs),
                &commit_fold_len,
                ..,
                ..,
                &commit_rs_len,
                &commit_rs_pages,
                &commit_rs_indptr,
                &commit_rs_w_slot,
                &commit_rs_w_off,
            )
            .map_err(|e| format!("commit recurrent binding: {e}"))?;
        fwd_c.epilogue(move || {
            let t = reduce_argmax(intrinsics::logits());
            commit_done.put(&t);
        });
        fwd_c
            .submit(&pipe)
            .map_err(|e| format!("commit submit: {e}"))?;
        commit_done
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("commit_done take: {e}"))?;
        committed = accepted;
    }

    // Whatever was buffered but not folded is abandoned by dropping its
    // slots — the reject half of fold-commit, and the whole point of the
    // buffer: no folded state was ever perturbed by the rejected tail.
    let remaining = rs.buffer_size();
    if remaining > 0 {
        rs.free_buffer(&(0..remaining).collect::<Vec<_>>())
            .map_err(|e| format!("free_buffer: {e}"))?;
    }

    pipe.close();

    let result = format!(
        "foldcommit prefill=1 buffered={SPEC_TOKENS} committed={committed} \
         abandoned={} g0={g0} drafted={drafted}",
        SPEC_TOKENS - committed
    );
    eprintln!("[GDN_FOLDCOMMIT] {result}");
    Ok(result)
}
