//! **Greedy decode on a GDN/hybrid model (KV + RECURRENT-STATE working sets)**
//! — `inferlet::ptir` bridge rewrite (bravo) of the MTP Stage-1 harness's model
//! driver for Qwen3.5-0.8B. It is the `generate` inferlet extended for models
//! with linear-attention layers: those layers need a RUNTIME-assigned
//! recurrent-state (rs_cache) slot per request, exactly as the attention
//! layers need KV page slots. The dense `generate` binds only a
//! [`WorkingSet`]; on a GDN model the driver's forward also requires an
//! [`RsWorkingSet`] bound into `forward-pass.new`'s rs-working-sets list (the
//! "this forward writes recurrent state" signal — `execute_impl`'s
//! `prepare_write` allocates the folded slot, +`RS_FLAG_RESET` on the fresh
//! fire, and the GDN forward writes it in-forward).
//!
//! Same greedy-argmax epilogue + KvLen-root dense geometry as `generate`; the
//! only addition is the RS working set binding, gated on
//! `rs.state_size() > 0` (pure-attention models skip it entirely and behave
//! like `generate`). Input: an optional token budget (default 5), e.g. `"24"`.
//!
//! RS wiring notes: the `RsWorkingSet` is attached on the prefill pass AND the
//! decode pass (both forward passes that exist). The decode pass is bound
//! ONCE and resubmitted every fire (a `Channel` may attach to only ONE pass
//! for its lifetime — `forward-pass.new` errs "may attach to only one pass"
//! on a second bind attempt, so `tok_in` is wired into a
//! single `ForwardPass` and that SAME pass is submitted `max_tokens-1` times,
//! never rebuilt). `RsStore::prepare_write` (`runtime/engine/src/store/rs.rs`,
//! `pipeline/fire.rs`) already keys reset-vs-continue-in-place off the
//! `RsWorkingSet`'s OWN `folded` slot + ref-count (bumped only by an explicit
//! `fork()`, which this inferlet never calls) — NOT off ForwardPass identity —
//! so reusing one bound decode pass still resets exactly once (the prefill's
//! first RS write) and continues in-place on every subsequent resubmit.
//! The decode epilogue advances its author-bound `KvLen` by each fire's live
//! token count.

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
    // → never bind it (this inferlet then behaves like `generate`).
    let rs = RsWorkingSet::new();
    let has_rs = rs.state_size() > 0;
    eprintln!(
        "[GENERATE_GDN] rs_state_size={} has_rs={has_rs}",
        rs.state_size()
    );

    if max_tokens == 0 {
        let result = "generated 0 tokens: []".to_string();
        eprintln!("[GENERATE_GDN] {result}");
        return Ok(result);
    }

    let prompt = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size);
    ws.reserve(max_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;

    // ───────────────────────── 1. PREFILL FIRE (N-wide) ─────────────────────
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
        // ONCE, not rebuilt per step. `rs_working_set`
        // is attached to this SAME pass, and the RS store's own reset-once/
        // continue-in-place tracking (keyed on the working set, not the pass)
        // is correct across all of this pass's resubmits.
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
    eprintln!("[GENERATE_GDN] {result}");
    Ok(result)
}
