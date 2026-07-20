//! **Explicit prefix-index e2e — cold publish then exact lookup.**
//!
//! Round 0 prefills N tokens, slices its completed full-page prefix, and
//! publishes that mapping under an opaque inferlet-owned key. Round 1 looks up
//! the indexed WorkingSet, reserves one suffix page, and explicitly submits
//! only the uncached suffix.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const N: u32 = 24; // prompt length: one full cached page (16) + 8-token suffix
const PAGE_T: u32 = 16; // tokens per KV page (matches the engine store)
const NUM_LAYERS: u32 = 2; // irrelevant to this test; required by the DSL config
const PREFIX_KEY: &[u8] = b"prefix-cache-e2e/full-page-0";

async fn round(tokens: &[i32], tag: &str, cached: bool) -> std::result::Result<i32, String> {
    let ws = if cached {
        WorkingSet::from_index(PREFIX_KEY)?
            .ok_or_else(|| format!("{tag}: explicit prefix index miss"))?
    } else {
        WorkingSet::new()
    };
    let max_pages = N.div_ceil(PAGE_T);
    let suffix_start = if cached { PAGE_T } else { 0 };
    ws.reserve(max_pages - ws.page_len())
        .map_err(|e| format!("{tag} ws.reserve: {e}"))?;
    let input = &tokens[suffix_start as usize..];
    let toks = Channel::from(input.to_vec()).named("toks");
    let out = Channel::new([1], dtype::i32).named("out");

    let fwd: ForwardPass = ForwardPass::new();
    fwd.embed(&toks, Tensor::constant(vec![0u32, input.len() as u32]));
    let kv_len = Channel::from(vec![N]).named("kv_len");
    fwd.port_channel(Port::KvLen, &kv_len);
    fwd.attn_working_set(&ws, .., (suffix_start / PAGE_T)..)?;
    fwd.derive_dense_geometry();
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
        .await
        .map_err(|e| format!("{tag} out.take: {e}"))?[0];
    if !cached {
        let prefix = ws.slice(&pipe, 0, 1)?;
        prefix.update_index(PREFIX_KEY)?;
    }
    pipe.close();
    Ok(g)
}

/// Honest CHUNKED continuation over one working set (no cache): a first
/// canonical pass commits the 16-token prefix, a second appends the 8-token
/// suffix. Its read-out row is the same absolute position as `round`'s, so
/// it isolates graft correctness from full-vs-chunked kernel numerics.
async fn round_chunked(tokens: &[i32], tag: &str) -> std::result::Result<i32, String> {
    let ws = WorkingSet::new();
    let max_pages = N.div_ceil(PAGE_T);
    ws.reserve(max_pages)
        .map_err(|e| format!("{tag} ws.reserve: {e}"))?;
    let k = PAGE_T as usize;

    let toks_a = Channel::from(tokens[..k].to_vec()).named("toks_a");
    let sink = Channel::new([1], dtype::i32).named("sink");
    let fwd_a: ForwardPass = ForwardPass::new();
    fwd_a.embed(&toks_a, Tensor::constant(vec![0u32, PAGE_T]));
    let kv_len_a = Channel::from(vec![PAGE_T]).named("kv_len_a");
    fwd_a.port_channel(Port::KvLen, &kv_len_a);
    fwd_a.attn_working_set(&ws, .., ..)?;
    fwd_a.derive_dense_geometry();
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
        .await
        .map_err(|e| format!("{tag} sink.take: {e}"))?;

    let toks_b = Channel::from(tokens[k..].to_vec()).named("toks_b");
    let out = Channel::new([1], dtype::i32).named("out_b");
    let fwd_b: ForwardPass = ForwardPass::new();
    fwd_b.embed(&toks_b, Tensor::constant(vec![0u32, N - PAGE_T]));
    let kv_len_b = Channel::from(vec![N]).named("kv_len_b");
    fwd_b.port_channel(Port::KvLen, &kv_len_b);
    fwd_b.attn_working_set(&ws, .., (PAGE_T / ws.page_size())..)?;
    fwd_b.derive_dense_geometry();
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
        .await
        .map_err(|e| format!("{tag} out.take: {e}"))?[0];
    pipe.close();
    Ok(g)
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);
    let tokens: Vec<i32> = (0..N as i32).map(|i| (i % 31) + 1).collect();

    let g0 = round(&tokens, "round0", false).await?;
    println!("[prefix-cache] round 0 (cold) done: g0={g0}");
    let g1 = round(&tokens, "round1", true).await?;
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
        let cold = round(&tokens_c, "round-c-cold", false).await?;
        let chunked = round_chunked(&tokens_c, "round-c-chunked").await?;
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
        Some((cold, chunked)) => {
            format!("PREFIX_CACHE_E2E n={N} g0={g0} g1={g1} cold={cold} chunked={chunked}")
        }
        None => format!("PREFIX_CACHE_E2E n={N} g0={g0} g1={g1}"),
    };
    println!("{result}");
    Ok(result)
}
