//! #18-class lock: a **single Shannon-entropy (Scalar)** measurement, on the
//! `inferlet::ptir` bridge (the direct-channel-e2e / generate wire form).
//!
//! One seeded prefill fire's epilogue computes `H = -Σ p·log p` (Shannon
//! entropy of the softmax over the LM-head logits) directly from
//! `intrinsics::logits()` via the eDSL ops and publishes it on a Scalar
//! reader channel. Before the #19 fast-path gate fix, a lone Scalar output
//! could be wrongly routed onto a TOKEN eager-D2H path (a token id's
//! int-bits-as-f32 ≈ a ~1e-40 denormal); a plausible positive entropy here
//! proves the #18-class stays locked.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let ws = WorkingSet::new();
    let page_size = ws.page_size();

    let mut prompt = wit_model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = n.div_ceil(page_size).max(1);
    ws.reserve(max_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

    let toks = Channel::from(prompt_i32).named("toks");
    let embed_indptr = Channel::from(vec![0u32, n]).named("embed_indptr");
    let positions = Channel::from((0..n).collect::<Vec<_>>()).named("positions");
    let pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages");
    let page_indptr = Channel::from(vec![0u32, max_pages]).named("page_indptr");
    let w_slot = Channel::from(
        (0..n)
            .map(|position| position / page_size)
            .collect::<Vec<_>>(),
    )
    .named("w_slot");
    let w_off = Channel::from(
        (0..n)
            .map(|position| position % page_size)
            .collect::<Vec<_>>(),
    )
    .named("w_off");
    let kv_len = Channel::from(vec![n]).named("kv_len");
    let entropy_out = Channel::new([1], dtype::f32).named("entropy_out");

    let fwd = ForwardPass::new();
    fwd.embed(&toks, &embed_indptr)?;
    fwd.attention(
        &ws,
        ..,
        ..,
        &kv_len,
        &pages,
        &page_indptr,
        &w_slot,
        &w_off,
        &positions,
        None,
    )?;
    fwd.epilogue(move || {
        // Shannon entropy H = -Σ p·log p of the softmax over the vocab.
        let logits = intrinsics::logits(); // [vocab] f32 (single read-out row)
        let p = softmax(logits);
        let h = entropy(&p);
        entropy_out.put(&h);
    });

    let pipeline = Pipeline::new();
    fwd.submit(&pipeline).map_err(|e| format!("submit: {e}"))?;
    let entropy = entropy_out
        .take()
        .get::<f32>()
        .await
        .map_err(|e| format!("entropy take: {e}"))?[0];
    pipeline.close();

    eprintln!("[ENTROPYCHECK] entropy={entropy}");
    Ok(format!("{{\"entropy\":{entropy}}}"))
}
