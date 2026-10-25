//! Builds a two-level prompt tree whose branches share their common prefixes.
//!
//! The common prompt is prefilled once. Two first-level branches fork that
//! working set, append distinct text, and are each forked again into two leaves.
//! Generation then continues independently from all four shared-prefix leaves.

use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_num_tokens")]
    num_tokens: usize,
}

fn default_num_tokens() -> usize {
    32
}

/// Freezes what it appends, so a later fork shares it instead of copying.
async fn append_tokens(ws: &WorkingSet, pipeline: &Pipeline, tokens: &[u32]) -> Result<i32> {
    if tokens.is_empty() {
        return Err("cannot append an empty token sequence".into());
    }
    let n = tokens.len() as u32;
    let token_input = Channel::from_iter(tokens.iter().map(|&token| token as i32));
    let embed_indptr = Channel::from([0u32, n]).named("embed_indptr");
    let next_token = Channel::new([1], dtype::i32).named("next_token");

    let fwd = ForwardPass::new();
    fwd.embed(&token_input, &embed_indptr)?;
    fwd.state(&[ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        next_token.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
    });
    fwd.submit(pipeline).context("append shared prefix")?;
    next_token.take_host::<i32>().await
}

async fn generate(
    ws: &WorkingSet,
    pipeline: &Pipeline,
    first_token: i32,
    max_tokens: usize,
) -> Result<Vec<u32>> {
    if max_tokens == 0 {
        return Ok(Vec::new());
    }

    let stop_tokens = chat::stop_tokens();
    let mut generated = Vec::with_capacity(max_tokens);
    if !stop_tokens.contains(&(first_token as u32)) {
        generated.push(first_token as u32);
    }
    if generated.len() >= max_tokens || stop_tokens.contains(&(first_token as u32)) {
        return Ok(generated);
    }

    let token_in = Channel::from([first_token]).named("token_in");
    let embed_indptr = Channel::from([0u32, 1]).named("embed_indptr");
    let token_out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("token_out");

    let fwd = ForwardPass::new();
    fwd.embed(&token_in, &embed_indptr)?;
    fwd.state(&[ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        let token = reshape(reduce_argmax(intrinsics::logits()), [1]);
        token_in.put(&token);
        token_out.put(&token);
    });

    let budget = max_tokens.saturating_sub(generated.len());
    run_ahead(pipeline, &fwd, budget, async || {
        let token = token_out.take_host::<i32>().await? as u32;
        if stop_tokens.contains(&token) {
            return Ok(ControlFlow::Break(()));
        }
        generated.push(token);
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    Ok(generated)
}

struct Branch {
    label: String,
    ws: WorkingSet,
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let root = WorkingSet::new();

    let root_tokens = model::encode("Write a short scene set");
    if root_tokens.is_empty() {
        return Err("tokenizer produced an empty root prompt".into());
    }

    let tree_pipeline = Pipeline::new();
    append_tokens(&root, &tree_pipeline, &root_tokens).await?;

    let mut first_level = Vec::new();
    for suffix in [" in a city", " in a forest"] {
        let child = root.fork(&tree_pipeline)?;
        let tokens = model::encode(suffix);
        append_tokens(&child, &tree_pipeline, &tokens).await?;
        first_level.push(Branch {
            label: suffix.trim().into(),
            ws: child,
        });
    }

    let mut leaves = Vec::new();
    for parent in first_level {
        for suffix in [" at dawn", " at night"] {
            let leaf = parent.ws.fork(&tree_pipeline)?;
            let tokens = model::encode(suffix);
            let first = append_tokens(&leaf, &tree_pipeline, &tokens).await?;
            leaves.push((format!("{} {}", parent.label, suffix.trim()), leaf, first));
        }
    }
    // The BUILD stream ends here: every append is submitted and nothing else
    // will go on this pipeline.
    tree_pipeline.close();

    // One pipeline per leaf: `run_ahead` closes its pipeline when the budget is spent.
    let mut outputs = Vec::with_capacity(leaves.len());
    for (label, ws, first) in leaves {
        let leaf_pipeline = Pipeline::new();
        let generated = generate(&ws, &leaf_pipeline, first, input.num_tokens).await?;
        outputs.push(format!("{label}: {}", model::decode(&generated)?));
    }
    Ok(outputs.join("\n"))
}
