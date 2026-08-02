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

use inferlet::ptir::attention::prelude::*;
use inferlet::{Result, model as wit_model};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    let mut prompt = wit_model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = n.div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("ws.reserve")?;
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

    let toks = Channel::from(prompt_i32).named("toks");
    let embed_indptr = Channel::from([0u32, n]).named("embed_indptr");
    let positions = Channel::from_iter(0..n).named("positions");
    let pages = Channel::from_iter(0..max_pages).named("pages");
    let page_indptr = Channel::from([0u32, max_pages]).named("page_indptr");
    let w_slot = Channel::from_iter((0..n).map(|position| position / page_size)).named("w_slot");
    let w_off = Channel::from_iter((0..n).map(|position| position % page_size)).named("w_off");
    let kv_len = Channel::from([n]).named("kv_len");
    let entropy_out = Channel::new([1], dtype::f32).named("entropy_out");

    let fwd = ForwardPass::new();
    fwd.embed(&toks, &embed_indptr)?;
    fwd.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: None,
        },
    )?;
    fwd.epilogue(move || {
        // Shannon entropy H = -Σ p·log p of the softmax over the vocab.
        let logits = intrinsics::logits(); // [vocab] f32 (single read-out row)
        let p = softmax(logits);
        let h = entropy(&p);
        entropy_out.put(&h);
    });

    let pipeline = Pipeline::new();
    fwd.submit(&pipeline).context("submit")?;
    let entropy = entropy_out.take_host::<f32>().await?;
    pipeline.close();

    eprintln!("[ENTROPYCHECK] entropy={entropy}");
    Ok(format!("{{\"entropy\":{entropy}}}"))
}
