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

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    let ws: &'static WorkingSet = bx(WorkingSet::new());
    model::configure(vocab, ws.page_size(), 1);

    let mut prompt = wit_model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

    let toks = bx(Channel::from(prompt_i32).named("toks"));
    let klen = bx(Channel::from(vec![n; 1]).named("klen"));
    let entropy_out = bx(Channel::new([1], dtype::f32).named("entropy_out"));

    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd.embed(toks, Tensor::constant(vec![0u32, n]));
    fwd.attn_working_set(ws, klen);
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
