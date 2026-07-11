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

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    let ws: &'static WorkingSet = bx(WorkingSet::new());
    model::configure(vocab, ws.page_size(), 1);

    let prompt_tokens = wit_model::encode("hello world");
    eprintln!("[TEMPGEN] encoded prompt: {} tokens", prompt_tokens.len());
    let prompt: Vec<u32> = if prompt_tokens.is_empty() {
        vec![0]
    } else {
        prompt_tokens
    };
    let n = prompt.len() as u32;

    // ───────────────────────── 1. PREFILL FIRE (N-wide) ─────────────────────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = bx(Channel::from(prompt_i32).named("toks_p"));
    let klen_p = bx(Channel::from(vec![n; 1]).named("klen_p"));
    let rng_p = bx(Channel::from(vec![0x9e37_u32, 0]).named("rng_p"));
    let g0_ch = bx(Channel::new([1], dtype::i32).named("g0"));

    let fwd_p: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd_p.embed(toks_p, Tensor::constant(vec![0u32, n]));
    fwd_p.attn_working_set(ws, klen_p);
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

    let prefill = Pipeline::new();
    fwd_p
        .submit(&prefill)
        .map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch
        .take()
        .get::<i32>()
        .map_err(|e| format!("g0 take: {e}"))?[0];
    prefill.close();

    let mut generated: Vec<u32> = Vec::with_capacity(MAX_TOKENS);
    generated.push(g0 as u32);
    eprintln!("[TEMPGEN] got token: {g0}");

    // ───────────────────────── 2. DECODE LOOP (1-wide) ──────────────────────
    if generated.len() < MAX_TOKENS {
        let tok_in = bx(Channel::from(vec![g0; 1]).named("tok_in"));
        let pos = bx(Channel::from(vec![n; 1]).named("pos"));
        let klen = bx(Channel::from(vec![n + 1; 1]).named("klen"));
        let fill = bx(Channel::from(vec![n + 1; 1]).named("fill"));
        let rng = bx(Channel::from(vec![0x51ed_u32, 0]).named("rng"));
        let out = bx(Channel::new([1], dtype::i32).named("out"));
        let lane1 = Tensor::constant(vec![0u32, 1u32]);

        let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
        fwd.embed(tok_in, lane1);
        fwd.positions(pos);
        fwd.attn_working_set(ws, klen);
        fwd.epilogue(move || {
            // Takes + compute first, PUTS last (value-id discipline).
            let base = fill.take().tensor(); // [1] u32 — position the NEXT fire writes
            let r = rng.take(); // [2] u32 rng state
            let logits = intrinsics::logits(); // [vocab] f32 (single read-out row)
            let scaled = div(logits, TEMPERATURE);
            let g = gumbel(&r, [vocab]);
            let t = reduce_argmax(add(scaled, g)); // [1] i32 categorical draw

            let klen_v = add(&base, 1u32);
            let next_free = add(&base, 1u32);
            let r_next = add(&r, iota(2));

            tok_in.put(&t);
            out.put(&t);
            pos.put(&base);
            klen.put(&klen_v);
            fill.put(&next_free);
            rng.put(&r_next);
        });

        let decode = Pipeline::new();
        for step in 1..MAX_TOKENS {
            fwd.submit(&decode)
                .map_err(|e| format!("decode submit @{step}: {e}"))?;
            let t = out
                .take()
                .get::<i32>()
                .map_err(|e| format!("out.take @{step}: {e}"))?;
            let Some(&t0) = t.first() else {
                return Err(format!("out.take @{step}: empty tensor"));
            };
            eprintln!("[TEMPGEN] got token: {t0}");
            generated.push(t0 as u32);
        }
        decode.close();
    }

    let text = wit_model::decode(&generated);
    eprintln!("[TEMPGEN] generated {} tokens: {:?}", generated.len(), text);
    Ok(format!("{{\"tokens\": {generated:?}, \"text\": {text:?}}}"))
}
