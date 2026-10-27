//! Builds a two-level prompt tree whose branches share their common prefixes.
//!
//! The common prompt is prefilled once. Two first-level branches fork that
//! working set, append distinct text, and are each forked again into two leaves.
//! Generation then continues independently from all four shared-prefix leaves,
//! one decode lane per leaf.

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
fn append_tokens(ws: &WorkingSet, pipeline: &Pipeline, tokens: &[u32]) -> Result<()> {
    if tokens.is_empty() {
        return Err("cannot append an empty token sequence".into());
    }
    let fwd = ForwardPass::new();
    fwd.embed(
        &Channel::from_iter(tokens.iter().map(|&token| token as i32)),
        &Channel::from([0u32, tokens.len() as u32]),
    )?;
    fwd.state(&[ws], StateGeometry::folding())?;
    fwd.submit(pipeline).context("append shared prefix")
}

/// Greedy decode, one lane per leaf, each seeded with its suffix's last token.
async fn generate(
    leaves: &[&WorkingSet],
    pipeline: &Pipeline,
    last: Vec<i32>,
    max_tokens: usize,
) -> Result<Vec<Vec<u32>>> {
    let b = leaves.len() as u32;
    let vocab = model::output_vocab_size();
    let stop_tokens = chat::stop_tokens();
    let token_in = Channel::from(last);
    let token_out = Channel::new([b], dtype::i32).capacity(channel_capacity() as u32);

    let fwd = ForwardPass::new();
    fwd.embed(&token_in, &Channel::from_iter(0..=b))?;
    fwd.state(leaves, StateGeometry::folding())?;
    fwd.epilogue(move || {
        let logits = reshape(intrinsics::output(0, "logits"), [b, vocab]);
        let tokens = reshape(reduce_argmax(logits), [b]);
        token_in.put(&tokens);
        token_out.put(&tokens);
    });

    let mut generated = vec![Vec::new(); leaves.len()];
    let mut done = vec![false; leaves.len()];
    run_ahead(pipeline, &fwd, max_tokens, async || {
        let tokens = token_out.take_host::<Vec<i32>>().await?;
        for (lane, token) in tokens.into_iter().enumerate() {
            let token = token as u32;
            if done[lane] {
                continue;
            }
            if stop_tokens.contains(&token) {
                done[lane] = true;
            } else {
                generated[lane].push(token);
            }
        }
        if done.iter().all(|&d| d) {
            return Ok(ControlFlow::Break(()));
        }
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
    let pipeline = Pipeline::new();
    append_tokens(&root, &pipeline, &model::encode("Write a short scene set"))?;

    let mut first_level = Vec::new();
    for suffix in [" in a city", " in a forest"] {
        let child = root.fork(&pipeline)?;
        append_tokens(&child, &pipeline, &model::encode(suffix))?;
        first_level.push(Branch {
            label: suffix.trim().into(),
            ws: child,
        });
    }

    let mut labels = Vec::new();
    let mut leaves = Vec::new();
    let mut last = Vec::new();
    for parent in &first_level {
        for suffix in [" at dawn", " at night"] {
            let leaf = parent.ws.fork(&pipeline)?;
            let tokens = model::encode(suffix);
            let (&tail, head) = tokens.split_last().ok_or("empty leaf suffix")?;
            if !head.is_empty() {
                append_tokens(&leaf, &pipeline, head)?;
            }
            labels.push(format!("{} {}", parent.label, suffix.trim()));
            leaves.push(leaf);
            last.push(tail as i32);
        }
    }

    let leaves: Vec<&WorkingSet> = leaves.iter().collect();
    let generated = generate(&leaves, &pipeline, last, input.num_tokens).await?;
    pipeline.close();

    let mut outputs = Vec::with_capacity(labels.len());
    for (label, tokens) in labels.iter().zip(&generated) {
        outputs.push(format!("{label}: {}", model::decode(tokens)?));
    }
    Ok(outputs.join("\n"))
}
