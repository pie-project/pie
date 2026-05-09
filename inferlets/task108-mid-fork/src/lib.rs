//! Phase 5b twins-fork smoke (ticket pie-agents#108).
//!
//! Mirrors `parallel-generation`'s known-good pattern but uses IDENTICAL
//! prompts on both forked children so we can assert byte-equal outputs
//! under Argmax. With Phase 5b's `mamba_fork_shmem` working, both
//! children inherit a byte-identical copy of parent's mamba state at
//! fork time → identical Argmax decode → byte-identical token streams.
//! Without copy-on-fork, child #2's slot is zero/stale → diverges.

use futures::future;
use inferlet::{
    Context, sample::Sampler, model::Model,
    runtime, Result,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_post_tokens")]
    post_tokens: usize,
}

fn default_post_tokens() -> usize { 4 }

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let post_tokens = input.post_tokens;

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    // Parent: system message gets prefilled & committed as KV pages.
    // Phase 5b's mamba_fork_shmem op runs at each child fork below;
    // those copies establish each child's mamba slot as a clone of the
    // parent's at this exact (post-prefill) point.
    let mut parent = Context::new(&model)?;
    parent.system("You are a helpful assistant.");
    parent.flush().await?;

    // Fork twice. Each child inherits parent's committed KV pages by
    // refcount AND parent's mamba slot contents by phase 5b's copy-on-
    // fork.
    let mut ctx1 = parent.fork()?;
    let mut ctx2 = parent.fork()?;
    println!("post_tokens: {post_tokens}");

    // Identical user message + cue on both children. Generate runs
    // prefill+decode for each. Argmax → identical token streams iff
    // mamba state at fork was identical.
    let h1 = async move {
        ctx1.user("Count from one to ten:");
        ctx1.cue();
        let out = ctx1
            .generate(Sampler::Argmax)
            .max_tokens(post_tokens)
            .collect_text()
            .await;
        ("twin_a_post", out)
    };
    let h2 = async move {
        ctx2.user("Count from one to ten:");
        ctx2.cue();
        let out = ctx2
            .generate(Sampler::Argmax)
            .max_tokens(post_tokens)
            .collect_text()
            .await;
        ("twin_b_post", out)
    };
    let (a_result, b_result) = future::join(h1, h2).await;
    println!("{}: {:?}", a_result.0, a_result.1);
    println!("{}: {:?}", b_result.0, b_result.1);

    let a_text = a_result.1.unwrap_or_default();
    let b_text = b_result.1.unwrap_or_default();
    let twins_match = a_text == b_text;
    // Smoke harness keys on this name; parent_post / child_post for
    // back-compat with the regex matchers in smoke_phase5b.py.
    println!("parent_post: {:?}", a_text);
    println!("child_post: {:?}", b_text);
    println!("twins_match: {}", twins_match);

    Ok(String::new())
}
