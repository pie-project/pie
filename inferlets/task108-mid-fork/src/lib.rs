//! Phase 5b mid-conversation fork smoke (ticket pie-agents#108).
//!
//! The Phase 5a smoke (parallel-generation) forked **immediately after
//! prefill** — children landed on fresh mamba slots whose contents were
//! never read because the model overwrites recurrent state during the
//! child's own user-prompt prefill before any decode reads it. That
//! workload couldn't distinguish "state was copied" from "state was
//! zeroed and never mattered".
//!
//! This inferlet forks **mid-decode**: parent prefills system + user
//! prompt, then auto-regressively decodes `warmup_tokens` (default 4)
//! Argmax tokens. After those tokens, the parent's mamba slot holds
//! genuine post-prefill recurrent state that the model produced step by
//! step. We fork at that point and have **both** parent and child
//! continue decoding `post_tokens` more Argmax tokens with no further
//! prompt edits.
//!
//! Twins property:
//!   With Phase 5b's `mamba_fork_shmem` working, child's slot is an
//!   exact copy of parent's at the fork point. Both twins read the
//!   same KV cache (committed via refcount + working pages copied via
//!   `copy_d2d_shmem`) and the same recurrent state, sampling Argmax
//!   from the same logits. Therefore `parent_post == child_post`
//!   byte-identically.
//!
//!   Without Phase 5b (or with a broken copy), child's slot is either
//!   zeroed or stale. The model's mamba forward consumes that wrong
//!   state and produces a different next-token distribution than the
//!   parent. Argmax breaks the tie → different tokens →
//!   `parent_post != child_post`.
//!
//! Output JSON (printed to stdout) is parsed by the host-side smoke
//! harness; the verdict is `parent_post == child_post`.

use futures::future;
use inferlet::{
    Context, sample::Sampler, model::Model,
    runtime, Result,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_warmup_tokens")]
    warmup_tokens: usize,
    #[serde(default = "default_post_tokens")]
    post_tokens: usize,
}

fn default_warmup_tokens() -> usize { 4 }
fn default_post_tokens() -> usize { 4 }

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let warmup_tokens = input.warmup_tokens;
    let post_tokens = input.post_tokens;

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut parent = Context::new(&model)?;
    parent.system("You are a helpful assistant.");
    parent.user("Count from one to ten:");
    parent.cue();
    parent.flush().await?;

    // Phase 1: parent decodes warmup_tokens, accumulating mamba state
    // post-prefill. Argmax so the result is deterministic given the
    // KV/mamba state.
    let parent_warmup = parent
        .generate(Sampler::Argmax)
        .max_tokens(warmup_tokens)
        .collect_text()
        .await;
    println!("warmup_tokens: {warmup_tokens}");
    println!("post_tokens: {post_tokens}");
    println!("parent_warmup: {:?}", parent_warmup);

    // Phase 2: fork the parent. With Phase 5b correct, child's mamba
    // slot is now a byte-identical copy of parent's at this exact
    // point — same KV, same conv/ssm state. Both twins enter Phase 3
    // with identical state.
    let mut child = parent.fork()?;

    // Phase 3: each twin independently decodes post_tokens more.
    // Concurrent join() encourages the runtime to land both twins'
    // first decode step in the same fire_batch — exercises the
    // ContextId-keyed slot allocation under multi-row conditions.
    let parent_handle = async move {
        let out = parent
            .generate(Sampler::Argmax)
            .max_tokens(post_tokens)
            .collect_text()
            .await;
        ("parent_post", out)
    };
    let child_handle = async move {
        let out = child
            .generate(Sampler::Argmax)
            .max_tokens(post_tokens)
            .collect_text()
            .await;
        ("child_post", out)
    };
    let (parent_result, child_result) =
        future::join(parent_handle, child_handle).await;
    println!("{}: {:?}", parent_result.0, parent_result.1);
    println!("{}: {:?}", child_result.0, child_result.1);

    // Verdict — twins must match.
    let parent_text = parent_result.1.unwrap_or_default();
    let child_text = child_result.1.unwrap_or_default();
    let twins_match = parent_text == child_text;
    println!("twins_match: {}", twins_match);

    Ok(String::new())
}
