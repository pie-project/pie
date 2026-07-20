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

    let vocab = wit_model::output_vocab_size();
    let ws = WorkingSet::new();
    model::configure(vocab, ws.page_size(), 1);

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
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(ws.page_size());
    ws.reserve(max_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;

    // ───────────────────────── 1. PREFILL FIRE (N=1) ────────────────────────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Tensor::constant(vec![0u32, n]);
    let g0_ch = Channel::new([1], dtype::i32).named("g0");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, embed_indptr_p);
    let kv_len_p = Channel::from(vec![n]).named("kv_len_p");
    fwd_p.port_channel(Port::KvLen, &kv_len_p);
    fwd_p.attn_working_set(&ws, .., ..)?;
    fwd_p.derive_dense_geometry();
    if has_rs {
        fwd_p.rs_working_set(&rs);
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
    if max_tokens == 1 {
        pipe.finish();
    }
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
        let lane1 = Tensor::constant(vec![0u32, 1u32]);

        // ONE bound ForwardPass, resubmitted every fire: each channel may
        // attach to only one pass for its lifetime, so `tok_in` is wired here
        // ONCE, not rebuilt per step. `rs_working_set` is attached to this
        // SAME pass, and the RS store's own reset-once/continue-in-place
        // tracking (keyed on the working set, not the pass) is correct across
        // all of this pass's resubmits.
        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, lane1);
        let kv_len = Channel::from(vec![n + 1]).named("kv_len");
        fwd.port_channel(Port::KvLen, &kv_len);
        fwd.attn_working_set(&ws, .., (n / ws.page_size())..)?;
        fwd.derive_dense_geometry();
        if has_rs {
            fwd.rs_working_set(&rs);
        }
        fwd.epilogue(move || {
            let length = kv_len.take().tensor();
            let t = reduce_argmax(intrinsics::logits()); // [1] i32 greedy token
            tok_in.put(&t);
            kv_len.put(add(&length, 1u32));
            out.put(&t);
        });

        for step in 1..max_tokens {
            // Fixed budget: the last submit is knowable at submit time →
            // finish() right after it (F7: end of stream; no close needed
            // after the drain).
            fwd.submit(&pipe)
                .map_err(|e| format!("decode submit @{step}: {e}"))?;
            if step + 1 == max_tokens {
                pipe.finish();
            }
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

    let result = format!("generated {} tokens: {:?}", generated.len(), generated);
    eprintln!("[GENERATE_GDN_N1] {result}");
    Ok(result)
}
