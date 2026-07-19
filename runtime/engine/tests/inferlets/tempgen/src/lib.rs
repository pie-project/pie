//! Temperature-sampling test inferlet (Task #4 verify) — `inferlet::ptir`
//! bridge rewrite. Pure temperature sampling (no top-k/top-p/min-p
//! truncation): each fire's epilogue Gumbel-max samples
//! `argmax(logits/T + gumbel_noise)` directly over the FULL vocab, which is
//! exactly a categorical draw from `softmax(logits/T)` (the temperature-only
//! `Spec::Multinomial` shape the old baked-IR `standard_sampler_program`
//! recognized). Proves temperature fires produce valid tokens end-to-end
//! through the `inferlet::ptir` decode loop.
//!
//! RNG state mirrors `text-completion`'s `sample_token`: `gumbel`/`rng`'s
//! `state` operand is validated as an EXACT `[2]` u32 `[key, ctr]` pair (not a
//! scalar/`[1]` value) — a `[2]` channel is taken each fire and the ctr lane
//! advanced (`add(r, iota(2))`) and put back for the next fire.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const TEMPERATURE: f32 = 0.8;
const MAX_TOKENS: usize = 8;

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    let ws = WorkingSet::new();
    let page_size = ws.page_size();

    let prompt_tokens = wit_model::encode("hello world");
    eprintln!("[TEMPGEN] encoded prompt: {} tokens", prompt_tokens.len());
    let prompt: Vec<u32> = if prompt_tokens.is_empty() {
        vec![0]
    } else {
        prompt_tokens
    };
    let n = prompt.len() as u32;
    let max_pages = (n + MAX_TOKENS as u32 + 1).div_ceil(page_size);
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
    let rng_p = Channel::from(vec![0x9e37_u32, 0]).named("rng_p");
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
    fwd_p.epilogue(move || {
        let r = rng_p.take(); // [2] u32 rng state (key, ctr)
        let logits = intrinsics::logits(); // [vocab] f32
        let scaled = div(logits, TEMPERATURE);
        let g = gumbel(&r, [vocab]);
        let t = reduce_argmax(add(scaled, g)); // [1] i32 categorical draw
        let r_next = add(&r, iota(2)); // advance ctr: [key, ctr+1]
        g0_ch.put(&t);
        rng_p.put(&r_next);
    });

    // ONE pipeline for the whole prefill→decode stream (R4-4): the decode
    // fires below are submitted on this same pipeline (MAX_TOKENS > 1, so
    // finish() (F7) lands after the last decode submit, not here).
    let pipe = Pipeline::new();
    fwd_p
        .submit(&pipe)
        .map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("g0 take: {e}"))?[0];

    let mut generated: Vec<u32> = Vec::with_capacity(MAX_TOKENS);
    generated.push(g0 as u32);
    eprintln!("[TEMPGEN] got token: {g0}");

    // ───────────────────────── 2. DECODE LOOP (1-wide) ──────────────────────
    if generated.len() < MAX_TOKENS {
        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let rng = Channel::from(vec![0x51ed_u32, 0]).named("rng");
        let out = Channel::new([1], dtype::i32).named("out");
        let lane1 = Channel::from(vec![0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from(vec![n]).named("positions");
        let pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages");
        let page_indptr =
            Channel::from(vec![0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from(vec![n / page_size]).named("w_slot");
        let w_off = Channel::from(vec![n % page_size]).named("w_off");

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
        fwd.epilogue(move || {
            let length = kv_len.take().tensor();
            let r = rng.take(); // [2] u32 rng state
            let logits = intrinsics::logits(); // [vocab] f32 (single read-out row)
            let scaled = div(logits, TEMPERATURE);
            let g = gumbel(&r, [vocab]);
            let t = reduce_argmax(add(scaled, g)); // [1] i32 categorical draw

            let r_next = add(&r, iota(2));
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
            rng.put(&r_next);
        });

        for step in 1..MAX_TOKENS {
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
            eprintln!("[TEMPGEN] got token: {t0}");
            generated.push(t0 as u32);
        }
    }
    pipe.close();

    let text = wit_model::decode(&generated);
    eprintln!("[TEMPGEN] generated {} tokens: {:?}", generated.len(), text);
    Ok(format!("{{\"tokens\": {generated:?}, \"text\": {text:?}}}"))
}
