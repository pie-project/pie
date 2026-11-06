//! One step of a conversation kept by its client.
//!
//! Takes the working set the client moved in, writes `tokens` to it, draws
//! `max_tokens` more, and gives it back. Nothing outlives the process but
//! the set: a client resumes it, forks it to branch, or closes it.

use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    /// What to write first: the prompt, or a resumed turn's last token.
    tokens: Vec<u32>,
    max_tokens: usize,
    seed: u32,
    /// How many tokens this conversation has drawn: a draw depends only on
    /// its place, so a run split into turns draws what it would in one.
    #[serde(default)]
    drawn: u32,
}

#[derive(Serialize)]
struct Output {
    tokens: Vec<u32>,
}

/// Writes `tokens` to `ws` and draws the token after them.
async fn draw(ws: &WorkingSet, pipe: &Pipeline, tokens: &[i32], seed: u32, at: u32) -> Result<i32> {
    let toks = Channel::from(tokens);
    let indptr = Channel::from([0u32, tokens.len() as u32]);
    let rng = Channel::from([seed, at]);
    let out = Channel::new([1], dtype::i32);
    let fwd = ForwardPass::new();
    fwd.embed(&toks, &indptr)?;
    fwd.state(&[ws], StateGeometry::folding())?;
    let sampled = out.clone();
    fwd.epilogue(move || {
        let r = rng.take();
        sampled.put(&gumbel_max(intrinsics::output(0, "logits"), &r));
    });
    fwd.submit(pipe).context("submit")?;
    Ok(out.take_host::<i32>().await.context("drain")?)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let ws = WorkingSet::take("ws").ok_or("the launch moved in no `ws`")?;
    let pipe = Pipeline::new();
    let mut tokens: Vec<i32> = input.tokens.iter().map(|&t| t as i32).collect();
    let mut generated = Vec::with_capacity(input.max_tokens);
    for i in 0..input.max_tokens as u32 {
        let next = draw(&ws, &pipe, &tokens, input.seed, input.drawn + i).await?;
        generated.push(next as u32);
        tokens = vec![next];
    }
    pipe.close();
    ws.give("ws")?;
    Ok(Output { tokens: generated })
}
