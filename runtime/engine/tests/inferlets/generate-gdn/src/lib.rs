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
//! Same greedy-argmax epilogue + KV/attn geometry sugar as `generate`
//! (`attn_working_set(&ws, &klen)` — the host derives + projects the KV
//! pages); the only addition is the RS working set binding, gated on
//! `rs.state_size() > 0` (pure-attention models skip it entirely and behave
//! like `generate`). Input: an optional token budget (default 5), e.g. `"24"`.
//!
//! RS wiring notes: the `RsWorkingSet` is attached on the prefill pass AND the
//! decode pass (both forward passes that exist). The decode pass is bound
//! ONCE and resubmitted every fire (a `Channel` may attach to only ONE pass
//! for its lifetime — `forward-pass.new` errs "may attach to only one pass"
//! on a second bind attempt, so `tok_in`/`pos`/`klen`/`fill` are wired into a
//! single `ForwardPass` and that SAME pass is submitted `max_tokens-1` times,
//! never rebuilt). `RsStore::prepare_write` (`runtime/engine/src/store/rs.rs`,
//! `pipeline/fire.rs`) already keys reset-vs-continue-in-place off the
//! `RsWorkingSet`'s OWN `folded` slot + ref-count (bumped only by an explicit
//! `fork()`, which this inferlet never calls) — NOT off ForwardPass identity —
//! so reusing one bound decode pass still resets exactly once (the prefill's
//! first RS write) and continues in-place on every subsequent resubmit.
//! `pos`/`klen`/`fill` device-loop-carry the growing absolute RoPE position +
//! attended KV length across every fire (the `isolatedtopp`/`mirostat` split).

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const DEFAULT_MAX_TOKENS: usize = 5;

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let max_tokens: usize = input.trim().parse().unwrap_or(DEFAULT_MAX_TOKENS);

    let vocab = wit_model::output_vocab_size();
    let ws: &'static WorkingSet = bx(WorkingSet::new());
    model::configure(vocab, ws.page_size(), 1);

    // Recurrent-state working set for the model's linear-attention layers.
    // `state_size() == 0` ⇒ the model has no recurrent state (pure attention)
    // → never bind it (this inferlet then behaves like `generate`).
    let rs: &'static RsWorkingSet = bx(RsWorkingSet::new());
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

    // ───────────────────────── 1. PREFILL FIRE (N-wide) ─────────────────────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = bx(Channel::from(prompt_i32).named("toks_p"));
    let embed_indptr_p = Tensor::constant(vec![0u32, n]);
    let klen_p = bx(Channel::from(vec![n; 1]).named("klen_p"));
    let g0_ch = bx(Channel::new([1], dtype::i32).named("g0"));

    let fwd_p: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd_p.embed(toks_p, embed_indptr_p);
    fwd_p.attn_working_set(ws, klen_p);
    if has_rs {
        fwd_p.rs_working_set(rs);
    }
    fwd_p.epilogue(move || {
        let t = reduce_argmax(intrinsics::logits()); // [1] i32 greedy token
        g0_ch.put(&t);
    });

    let prefill = Pipeline::new();
    fwd_p
        .submit(&prefill)
        .map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("g0 take: {e}"))?[0];
    prefill.close();

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    generated.push(g0 as u32);

    // ───────────────────────── 2. DECODE LOOP (1-wide) ──────────────────────
    if generated.len() < max_tokens {
        let tok_in = bx(Channel::from(vec![g0; 1]).named("tok_in"));
        let pos = bx(Channel::from(vec![n; 1]).named("pos"));
        let klen = bx(Channel::from(vec![n + 1; 1]).named("klen"));
        let fill = bx(Channel::from(vec![n + 1; 1]).named("fill"));
        let out = bx(Channel::new([1], dtype::i32).named("out"));
        let lane1 = Tensor::constant(vec![0u32, 1u32]);

        // ONE bound ForwardPass, resubmitted every fire: each channel may
        // attach to only one pass for its lifetime, so `tok_in`/`pos`/`klen`/
        // `fill` are wired here ONCE, not rebuilt per step. `rs_working_set`
        // is attached to this SAME pass, and the RS store's own reset-once/
        // continue-in-place tracking (keyed on the working set, not the pass)
        // is correct across all of this pass's resubmits.
        let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
        fwd.embed(tok_in, lane1);
        fwd.positions(pos);
        fwd.attn_working_set(ws, klen);
        if has_rs {
            fwd.rs_working_set(rs);
        }
        fwd.epilogue(move || {
            // Takes + compute first, PUTS last (value-id discipline).
            let base = fill.take().tensor(); // [1] u32 — position the NEXT fire writes
            let t = reduce_argmax(intrinsics::logits()); // [1] i32 greedy token

            let klen_v = add(&base, 1u32);
            let next_free = add(&base, 1u32);

            tok_in.put(&t);
            out.put(&t);
            pos.put(&base);
            klen.put(&klen_v);
            fill.put(&next_free);
        });

        let decode = Pipeline::new();
        for step in 1..max_tokens {
            fwd.submit(&decode)
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
        decode.close();
    }

    let result = format!("generated {} tokens: {:?}", generated.len(), generated);
    eprintln!("[GENERATE_GDN] {result}");
    Ok(result)
}
