//! Greedy text completion (`reduce_argmax`), so the same prompt continues the same way every run.

use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_prompt() -> String {
    "The capital of France is".into()
}

fn default_max_tokens() -> usize {
    8
}

#[derive(Serialize)]
struct Output {
    /// The continuation, decoded.
    text: String,
    /// How many tokens it is.
    count: usize,
    tokens: Vec<u32>,
}

/// The greedy pick over a logits row, as a one-lane `[1]` i32 cell.
fn greedy(logits: Tensor) -> Tensor {
    reshape(reduce_argmax(&logits), [1])
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    // The model's opening (`<bos>` where it has one) before the raw text: a
    // gemma without it answers noise.
    let prompt: Vec<i32> = inferlet::chat::prefix()
        .into_iter()
        .chain(model::encode(&input.prompt))
        .map(|t| t as i32)
        .collect();
    let (&last, head) = prompt.split_last().ok_or("empty prompt")?;
    if !head.is_empty() {
        let fill = ForwardPass::new();
        fill.embed(
            &Channel::from(head),
            &Channel::from([0u32, head.len() as u32]),
        )?;
        fill.state(&[&ws], StateGeometry::folding())?;
        fill.submit(&pipe).context("prefill submit")?;
    }

    let tok_in = Channel::from([last]);
    let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &Channel::from([0u32, 1]))?;
    fwd.state(&[&ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        let token = greedy(intrinsics::output(0, "logits"));
        tok_in.put(&token);
        tok_out.put(&token);
    });

    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    run_ahead(&pipe, &fwd, input.max_tokens, async || {
        let token = tok_out
            .take_host::<i32>()
            .await
            .with_context(|| format!("decode drain @{}", generated.len()))?;
        generated.push(token as u32);
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    pipe.close();

    Ok(Output {
        count: generated.len(),
        text: model::decode(&generated)?,
        tokens: generated,
    })
}
