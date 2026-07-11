//! **Prefix-cache e2e — CANONICAL prefill × 2.**
//!
//! Runs the SAME single-lane canonical prompt prefill twice, each over a fresh
//! `WorkingSet` that is dropped after its fire:
//!
//!   1. Round 1 prefills N tokens; on the WorkingSet drop the runtime retains
//!      the canonical path (cache-root lease) and indexes its full pages in
//!      the CAS.
//!   2. Round 2 builds the identical pass over a fresh WorkingSet. At fire
//!      time the runtime recognizes the canonical evidence, probes the CAS,
//!      GRAFTS the cached full-page prefix, and TRIMS the launch to the
//!      uncached suffix — completely invisible from in here: this inferlet
//!      does nothing but run the same pass twice.
//!
//! The engine test asserts the trim on the driver operation log
//! (`launch-shape tokens=24` then `tokens=8` for N=24, page 16). On a real
//! model `g0 == g1` additionally proves the grafted KV is byte-equivalent to
//! recomputation (pass input "strict" to enforce; the dummy driver's
//! synthetic logits are launch-seeded, so equality only holds on real
//! hardware).
//!
//! CANONICAL means: token embed with a single trace-const lane, KvLen-only
//! attention (`attn_working_set(&ws, klen)` — no Pages/WSlot/WOff/AttnMask/
//! Positions ports), token values host-known (seeded channel).

use inferlet::ptir::prelude::*;
use inferlet::{model as wit_model, Result};

const N: u32 = 24; // prompt length: one full cached page (16) + 8-token suffix
const PAGE_T: u32 = 16; // tokens per KV page (matches the engine store)
const NUM_LAYERS: u32 = 2; // irrelevant to this test; required by the DSL config

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// One canonical N-token prefill over a FRESH working set; returns the greedy
/// read-out of the last row. Everything (pass, pipeline, working set) drops
/// before returning, so the runtime's drop-time retention runs.
fn round(tokens: &[i32], tag: &str) -> std::result::Result<i32, String> {
    let ws = WorkingSet::new();
    let toks = bx(Channel::from(tokens.to_vec()).named("toks")); // [N] i32, seeded
    let klen = bx(Channel::from(vec![N; 1]).named("klen")); // [1] total kv len
    let out = bx(Channel::new([1], dtype::i32).named("out"));

    let fwd: ForwardPass<'static> = ForwardPass::new();
    fwd.embed(toks, Tensor::constant(vec![0u32, N])); // single const lane [0, N]
    fwd.attn_working_set(&ws, klen); // KvLen only — the canonical sugar arity
    fwd.epilogue(move || {
        let tok = reduce_argmax(intrinsics::logits()); // read-out row N-1
        out.put(&tok);
    });

    let pipe = Pipeline::new();
    fwd.submit(&pipe)
        .map_err(|e| format!("{tag} submit: {e}"))?;
    let g = out
        .take()
        .get::<i32>()
        .map_err(|e| format!("{tag} out.take: {e}"))?[0];
    pipe.close();
    Ok(g)
}

/// Honest CHUNKED continuation over one working set (no cache): a first
/// canonical pass commits the 16-token prefix, a second appends the 8-token
/// suffix. Its read-out row is the same absolute position as `round`'s, so
/// it isolates graft correctness from full-vs-chunked kernel numerics.
fn round_chunked(tokens: &[i32], tag: &str) -> std::result::Result<i32, String> {
    let ws = WorkingSet::new();
    let k = PAGE_T as usize;

    let toks_a = bx(Channel::from(tokens[..k].to_vec()).named("toks_a"));
    let klen_a = bx(Channel::from(vec![PAGE_T; 1]).named("klen_a"));
    let sink = bx(Channel::new([1], dtype::i32).named("sink"));
    let fwd_a: ForwardPass<'static> = ForwardPass::new();
    fwd_a.embed(toks_a, Tensor::constant(vec![0u32, PAGE_T]));
    fwd_a.attn_working_set(&ws, klen_a);
    fwd_a.epilogue(move || {
        let tok = reduce_argmax(intrinsics::logits());
        sink.put(&tok);
    });
    let pipe = Pipeline::new();
    fwd_a
        .submit(&pipe)
        .map_err(|e| format!("{tag} chunk-a submit: {e}"))?;
    let _ = sink
        .take()
        .get::<i32>()
        .map_err(|e| format!("{tag} sink.take: {e}"))?;

    let toks_b = bx(Channel::from(tokens[k..].to_vec()).named("toks_b"));
    let klen_b = bx(Channel::from(vec![N; 1]).named("klen_b"));
    let out = bx(Channel::new([1], dtype::i32).named("out_b"));
    let fwd_b: ForwardPass<'static> = ForwardPass::new();
    fwd_b.embed(toks_b, Tensor::constant(vec![0u32, N - PAGE_T]));
    fwd_b.attn_working_set(&ws, klen_b);
    fwd_b.epilogue(move || {
        let tok = reduce_argmax(intrinsics::logits());
        out.put(&tok);
    });
    fwd_b
        .submit(&pipe)
        .map_err(|e| format!("{tag} chunk-b submit: {e}"))?;
    let g = out
        .take()
        .get::<i32>()
        .map_err(|e| format!("{tag} out.take: {e}"))?[0];
    pipe.close();
    Ok(g)
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);
    let tokens: Vec<i32> = (0..N as i32).map(|i| (i % 31) + 1).collect();

    let g0 = round(&tokens, "round0")?;
    println!("[prefix-cache] round 0 (cold) done: g0={g0}");
    let g1 = round(&tokens, "round1")?;
    println!("[prefix-cache] round 1 (cached prefix) done: g1={g1}");
    let gc = if input.contains("chunked") {
        // Distinct tokens so the chunked rounds never hit the cache — unless
        // "same" is passed (with retention disabled server-side), which
        // probes chunked-vs-cold on the PRIMARY tokens.
        let tokens_c: Vec<i32> = if input.contains("same") {
            tokens.clone()
        } else {
            (0..N as i32).map(|i| ((i * 7) % 31) + 1).collect()
        };
        let cold = round(&tokens_c, "round-c-cold")?;
        let chunked = round_chunked(&tokens_c, "round-c-chunked")?;
        println!("[prefix-cache] chunked probe: cold={cold} chunked={chunked}");
        Some((cold, chunked))
    } else {
        None
    };

    if input.contains("strict") {
        match gc {
            // The honest chunked continuation is the graft's NUMERICAL
            // equivalent (identical kernel shapes: 8 query rows over 24 kv).
            // A full 24-row prefill may legitimately flip a near-tie argmax
            // against either (device-verified 2026-07-11: graft == chunked
            // exactly; full differs on a synthetic near-tie prompt, matches
            // on others).
            Some((_, chunked)) if input.contains("same") => {
                if g1 != chunked {
                    return Err(format!(
                        "grafted fire diverged from the honest chunked \
                         continuation: g1={g1} chunked={chunked}"
                    ));
                }
            }
            _ => {
                if g0 != g1 {
                    return Err(format!(
                        "grafted prefill diverged from recomputation: g0={g0} g1={g1}"
                    ));
                }
            }
        }
    }
    let result = match gc {
        Some((cold, chunked)) => format!(
            "PREFIX_CACHE_E2E n={N} g0={g0} g1={g1} cold={cold} chunked={chunked}"
        ),
        None => format!("PREFIX_CACHE_E2E n={N} g0={g0} g1={g1}"),
    };
    println!("{result}");
    Ok(result)
}
