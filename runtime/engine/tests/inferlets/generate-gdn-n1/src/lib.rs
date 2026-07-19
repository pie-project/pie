//! **rs_cache T0 — E1 discriminator: SINGLE-token (N=1) prefill.** A
//! byte-for-byte copy of the migrated `generate-gdn` (same greedy epilogue,
//! same KV geometry sugar, same in-forward RS write) with EXACTLY ONE change:
//! the prefill fire carries a SINGLE token (N=1), not the 2-token
//! "hello world" prompt (N=2). Its sole purpose is to isolate the rs_cache T0
//! boundary bug:
//!
//!   generate-gdn (N=2 prefill) GLITCHES ~4 decode steps at the prefill->decode
//!   boundary, then recovers to HF-exact. Run THIS (N=1 prefill) on the same
//!   4090 + a matching HF golden:
//!     * glitch VANISHES ⇒ the bug is the MULTI-TOKEN prefill state-commit
//!       (the N=2 in-forward fold doesn't commit both tokens' state).
//!     * glitch PERSISTS ⇒ the bug is the GENERAL prefill->decode slab handoff
//!       (CoW-move timing / folded slab toggle — the runtime RS wiring), which
//!       is independent of the prefill token count.
//!
//! The single prefill token = `encode("hello world")[0]` (the FIRST token of
//! `generate-gdn`'s repro prompt), so the HF golden for E1 is just
//! `HF-generate([tok0])` — reproducible from the existing tokenization, no new
//! prompt. Everything downstream (decode loop, RS binding, greedy epilogue) is
//! identical to `generate-gdn` so the ONLY independent variable is prefill N.
//!
//! Input: an optional token budget (default 5), e.g. `"24"` — same as
//! generate-gdn.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const DEFAULT_MAX_TOKENS: usize = 5;

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let max_tokens: usize = input.trim().parse().unwrap_or(DEFAULT_MAX_TOKENS);

    let ws = WorkingSet::new();
    let page_size = ws.page_size();

    // Recurrent-state working set for the model's linear-attention layers.
    // `state_size() == 0` ⇒ the model has no recurrent state (pure attention)
    // → never bind it (this inferlet then behaves like a dense N=1 generate).
    let rs = RsWorkingSet::new();
    let has_rs = rs.state_size() > 0;
    eprintln!(
        "[GENERATE_GDN_N1] rs_state_size={} has_rs={has_rs}",
        rs.state_size()
    );

    if max_tokens == 0 {
        let result = "generated 0 tokens: []".to_string();
        eprintln!("[GENERATE_GDN_N1] {result}");
        return Ok(result);
    }

    // E1 DISCRIMINATOR: force a SINGLE-token (N=1) prefill. Take only the
    // FIRST token of the repro prompt so there is NO multi-token prefill fold
    // — the one and only difference from generate-gdn.
    let full_prompt = wit_model::encode("hello world");
    let first_tok = full_prompt.first().copied().unwrap_or(0);
    let prompt: Vec<u32> = vec![first_tok];
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size);
    ws.reserve(max_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;

    // ───────────────────────── 1. PREFILL FIRE (N=1) ────────────────────────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Channel::from(vec![0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from((0..n).collect::<Vec<_>>()).named("positions_p");
    let pages_p = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages_p");
    let page_indptr_p = Channel::from(vec![0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p = Channel::from(
        (0..n)
            .map(|position| position / page_size)
            .collect::<Vec<_>>(),
    )
    .named("w_slot_p");
    let w_off_p = Channel::from(
        (0..n)
            .map(|position| position % page_size)
            .collect::<Vec<_>>(),
    )
    .named("w_off_p");
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
    if has_rs {
        fwd_p.rs_working_sets(std::slice::from_ref(&rs))?;
    }
    fwd_p.epilogue(move || {
        let t = reduce_argmax(intrinsics::logits()); // [1] i32 greedy token
        g0_ch.put(&t);
    });

    // ONE pipeline for the whole prefill→decode stream (R4-4): the decode
    // fires below are submitted on this same pipeline. The stream is finished
    // (F7) right after the prefill submit only in the degenerate case where
    // zero decode fires follow.
    let pipe = Pipeline::new();
    fwd_p
        .submit(&pipe)
        .map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("g0 take: {e}"))?[0];

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    generated.push(g0 as u32);

    // ───────────────────────── 2. DECODE LOOP (1-wide) ──────────────────────
    if generated.len() < max_tokens {
        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let out = Channel::new([1], dtype::i32).named("out");
        let lane1 = Channel::from(vec![0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from(vec![n]).named("positions");
        let pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages");
        let page_indptr =
            Channel::from(vec![0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from(vec![n / page_size]).named("w_slot");
        let w_off = Channel::from(vec![n % page_size]).named("w_off");

        // ONE bound ForwardPass, resubmitted every fire: each channel may
        // attach to only one pass for its lifetime, so `tok_in` is wired here
        // ONCE, not rebuilt per step. `rs_working_set` is attached to this
        // SAME pass, and the RS store's own reset-once/continue-in-place
        // tracking (keyed on the working set, not the pass) is correct across
        // all of this pass's resubmits.
        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        let kv_len = Channel::from(vec![n + 1]).named("kv_len");
        fwd.attention(
            &ws,
            ..,
            (n / page_size)..,
            &kv_len,
            &pages,
            &page_indptr,
            &w_slot,
            &w_off,
            &positions,
            None,
        )?;
        if has_rs {
            fwd.rs_working_sets(std::slice::from_ref(&rs))?;
        }
        fwd.epilogue(move || {
            let length = kv_len.take().tensor();
            let t = reduce_argmax(intrinsics::logits()); // [1] i32 greedy token
            let next_length = add(&length, 1u32);
            let page_count = div(add(&next_length, page_size - 1), page_size);
            tok_in.put(&t);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(div(&length, page_size));
            w_off.put(rem(&length, page_size));
            page_indptr.take();
            page_indptr.put(mul(iota(2), broadcast(&page_count, [2])));
            out.put(&t);
        });

        for step in 1..max_tokens {
            fwd.submit(&pipe)
                .map_err(|e| format!("decode submit @{step}: {e}"))?;
            let t = out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("out.take @{step}: {e}"))?;
            let Some(&t0) = t.first() else {
                return Err(format!("out.take @{step}: empty tensor"));
            };
            generated.push(t0 as u32);
        }
    }
    pipe.close();

    let result = format!("generated {} tokens: {:?}", generated.len(), generated);
    eprintln!("[GENERATE_GDN_N1] {result}");
    Ok(result)
}
