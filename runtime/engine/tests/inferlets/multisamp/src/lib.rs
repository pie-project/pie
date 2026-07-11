//! Multi-sampler #7 per-kind coverage inferlet — **`inferlet::ptir` bridge
//! rewrite**. Generates a few tokens with each standard truncated-sampling
//! kind in sequence — TopK, TopP, MinP, TopK+TopP joint — so one run exercises
//! the device eDSL mask primitives across every kind that used to route to a
//! dedicated FlashInfer kernel. Each kind continues the SAME growing KV
//! context (one `WorkingSet` threaded across all 16 fires), so the kinds
//! appear as distinct decode fires over one conversation, matching the old
//! sequential `Ctx`-threaded shape.
//!
//! A `Channel` may attach to only ONE `ForwardPass` for its lifetime
//! (`forward-pass.new` errs "may attach to only one pass" on a second bind),
//! so each kind gets its OWN fresh `tok_in`/`pos`/`klen`/`fill`/`out`/`rng`
//! channels and its own fresh `ForwardPass` — never shared with another
//! kind's pass. The handoff between kinds rides the host (the same seam as
//! the prefill→decode handoff): read the previous kind's last token + the
//! running absolute position off the host, then seed the next kind's fresh
//! channels from those values (`tok_in = [last_tok]`, `pos = [n+count-1]`,
//! `klen = fill = [n+count]`) — exactly the values the ORIGINAL prefill→decode
//! seed used with `count = 1`.
//!
//! Each kind's device mask + Gumbel-max sample is built directly from the
//! eDSL primitives — the author-facing equivalent of the deleted
//! `sampler::sampler_program`/`SamplerSpec` shapes:
//!   - TopK:  `keep = pivot_threshold(scaled, rank_le(k))` (rank is invariant
//!            to any monotonic transform, so it runs directly on scaled logits).
//!   - TopP:  `keep = pivot_threshold(softmax(scaled), cummass_le(p))`.
//!   - MinP:  `keep = pivot_threshold(softmax(scaled), prob_ge(p · max_prob))`.
//!   - Joint: the AND of the TopK and TopP masks.
//! The final sample is always `argmax(select(keep, scaled, -inf) + gumbel(r))`
//! — the Gumbel-max trick, which samples from `softmax(scaled)` restricted to
//! the kept set without an explicit normalize. `r` is a `[2]` u32 rng state
//! (`gumbel`'s `state` operand is validated as EXACTLY `[2]` u32).

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const TEMPERATURE: f32 = 0.8;
const STEPS_PER_KIND: usize = 4;

#[derive(Clone, Copy)]
enum Kind {
    TopK { k: u32 },
    TopP { p: f32 },
    MinP { p: f32 },
    TopKTopP { k: u32, p: f32 },
}

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// This kind's keep-mask over `scaled` (temperature-scaled logits), `[vocab]` bool.
fn keep_mask(kind: Kind, scaled: &Tensor) -> Tensor {
    match kind {
        Kind::TopK { k } => pivot_threshold(scaled, rank_le(k)),
        Kind::TopP { p } => {
            let probs = softmax(scaled);
            pivot_threshold(probs, cummass_le(p))
        }
        Kind::MinP { p } => {
            let probs = softmax(scaled);
            let thr = mul(reduce_max(&probs), p); // scalar threshold (NOT broadcast to [vocab])
            pivot_threshold(probs, prob_ge(thr))
        }
        Kind::TopKTopP { k, p } => {
            let probs = softmax(scaled);
            let m1 = pivot_threshold(scaled, rank_le(k));
            let m2 = pivot_threshold(probs, cummass_le(p));
            and(m1, m2)
        }
    }
}

/// Masked Gumbel-max sample: `argmax(select(keep, scaled, -inf) + gumbel(r))`.
/// `r` is the taken `[2]` u32 rng state (`[key, ctr]`).
fn sample(kind: Kind, scaled: Tensor, vocab: u32, r: impl AsTensor) -> Tensor {
    let keep = keep_mask(kind, &scaled);
    let neg_inf = broadcast(Tensor::constant(f32::NEG_INFINITY), [vocab]);
    let masked = select(&keep, &scaled, &neg_inf);
    let g = gumbel(r, [vocab]);
    reduce_argmax(add(masked, g))
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    let ws: &'static WorkingSet = bx(WorkingSet::new());
    model::configure(vocab, ws.page_size(), 1);

    let samplers: [(&str, Kind); 4] = [
        ("topk", Kind::TopK { k: 40 }),
        ("topp", Kind::TopP { p: 0.9 }),
        ("minp", Kind::MinP { p: 0.05 }),
        ("joint", Kind::TopKTopP { k: 40, p: 0.9 }),
    ];

    // The prompt feeds the FIRST fire (first kind's prefill); every subsequent
    // fire (across all kinds) continues the SAME growing context.
    let prompt = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let n = prompt.len() as u32;

    // ── PREFILL FIRE (N-wide) — the first kind's first token. ──
    let (_, kind0) = samplers[0];
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
        let scaled = div(intrinsics::logits(), TEMPERATURE);
        let t = sample(kind0, scaled, vocab, &r);
        let r_next = add(&r, iota(2));
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

    // `count` = total tokens generated so far (across every kind), INCLUDING
    // `last_tok`; the next fire embeds `last_tok` at absolute position
    // `n+count-1` and writes `klen = n+count` — the same seed the original
    // prefill→decode handoff uses with `count == 1`.
    let mut last_tok = g0;
    let mut count: u32 = 1;

    let mut all = Vec::new();
    for (i, (name, kind)) in samplers.into_iter().enumerate() {
        let mut got = if i == 0 { vec![g0 as u32] } else { Vec::new() };
        let steps = if i == 0 {
            STEPS_PER_KIND - 1
        } else {
            STEPS_PER_KIND
        };

        if steps > 0 {
            // Fresh channels + a fresh ForwardPass for EVERY kind: a Channel
            // may attach to only one pass for its lifetime, so the previous
            // kind's `tok_in`/`pos`/`klen`/`fill`/`rng`/`out` can never be
            // reused here. Seeded from the running host-tracked
            // `(last_tok, count)` — the handoff rides the host, exactly like
            // the prefill→decode seam.
            let tok_in = bx(Channel::from(vec![last_tok; 1]).named("tok_in"));
            let pos = bx(Channel::from(vec![n + count - 1; 1]).named("pos"));
            let klen = bx(Channel::from(vec![n + count; 1]).named("klen"));
            let fill = bx(Channel::from(vec![n + count; 1]).named("fill"));
            let rng = bx(Channel::from(vec![0x51ed_u32 ^ (i as u32), 0]).named("rng"));
            let out = bx(Channel::new([1], dtype::i32).named("out"));
            let lane1 = Tensor::constant(vec![0u32, 1u32]);

            let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
            fwd.embed(tok_in, lane1);
            fwd.positions(pos);
            fwd.attn_working_set(ws, klen);
            fwd.epilogue(move || {
                let base = fill.take().tensor(); // [1] u32
                let r = rng.take(); // [2] u32 rng state
                let scaled = div(intrinsics::logits(), TEMPERATURE);
                let t = sample(kind, scaled, vocab, &r);

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

            let pipeline = Pipeline::new();
            for step in 0..steps {
                fwd.submit(&pipeline)
                    .map_err(|e| format!("{name} submit @{step}: {e}"))?;
                let t = out
                    .take()
                    .get::<i32>()
                    .map_err(|e| format!("{name} out.take @{step}: {e}"))?;
                let Some(&t0) = t.first() else {
                    return Err(format!("{name} out.take @{step}: empty tensor"));
                };
                got.push(t0 as u32);
                last_tok = t0;
                count += 1;
            }
            pipeline.close();
        }

        eprintln!("[MULTISAMP] {name} tokens: {got:?}");
        all.extend(got);
    }

    Ok(format!("{{\"tokens\": {all:?}}}"))
}
