//! One step of a conversation kept by its client.
//!
//! Takes the working set the client moved in, writes `tokens` to it, draws
//! `max_tokens` more, and gives it back. Nothing outlives the process but
//! the set: a client resumes it, forks it to branch, or closes it.

use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    /// The handle the client moved in.
    ws: WorkingSet,
    /// What to write first: the prompt, or a resumed turn's last token.
    tokens: Vec<u32>,
    max_tokens: usize,
    seed: u32,
    /// How many tokens this conversation has drawn: a draw depends only on
    /// its place, so a run split into turns draws what it would in one.
    #[serde(default)]
    drawn: u32,
    /// Draw ahead of reading (`run_ahead`), and stop after this many
    /// tokens, as a stop string would: the set given back holds what one
    /// drawing a token at a time would have written, whatever ran ahead.
    #[serde(default)]
    stop_after: Option<usize>,
    /// Draw the most likely token, not a Gumbel draw: a run whose tokens
    /// part from another's then parts by its logits alone.
    #[serde(default)]
    greedy: bool,
}

/// The token a row's logits pick: the most likely one when `greedy`, else
/// a Gumbel draw keyed by `r`.
fn pick(logits: Tensor, r: &Tensor, greedy: bool) -> Tensor {
    if greedy { reduce_argmax(logits) } else { gumbel_max(logits, r) }
}

#[derive(Serialize)]
struct Output {
    tokens: Vec<u32>,
    /// A token drawn one at a time: the gap between its row's two largest
    /// logits over the largest's magnitude (a near tie is a small one).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    gaps: Vec<f32>,
}

/// Writes `tokens` to `ws` and draws the token after them.
async fn draw(
    ws: &WorkingSet,
    pipe: &Pipeline,
    tokens: &[i32],
    seed: u32,
    at: u32,
    greedy: bool,
) -> Result<(i32, f32)> {
    let toks = Channel::from(tokens);
    let indptr = Channel::from([0u32, tokens.len() as u32]);
    let rng = Channel::from([seed, at]);
    let out = Channel::new([1], dtype::i32);
    let gap = Channel::new([2], dtype::f32);
    let fwd = ForwardPass::new();
    fwd.embed(&toks, &indptr)?;
    fwd.state(&[ws], StateGeometry::folding())?;
    let (sampled, top) = (out.clone(), gap.clone());
    fwd.epilogue(move || {
        let r = rng.take();
        let logits = intrinsics::output(0, "logits");
        sampled.put(&pick(logits.clone(), &r, greedy));
        let (values, _) = top_k(logits, 2);
        top.put(&values);
    });
    fwd.submit(pipe).context("submit")?;
    let token = out.take_host::<i32>().await.context("drain")?;
    let two = gap.take_host::<Vec<f32>>().await.context("drain gap")?;
    let g = match two.as_slice() {
        [a, b, ..] => (a - b) / a.abs().max(f32::MIN_POSITIVE),
        _ => 0.0,
    };
    Ok((token, g))
}

/// Writes `tokens` to `ws`, then draws up to `max_tokens` ahead of reading
/// them, stopping after `stop_after`.
async fn draw_ahead(
    ws: &WorkingSet,
    pipe: &Pipeline,
    input: &Input,
    stop_after: usize,
) -> Result<Vec<u32>> {
    let tokens: Vec<i32> = input.tokens.iter().map(|&t| t as i32).collect();
    let (first, _) = draw(ws, pipe, &tokens, input.seed, input.drawn, input.greedy).await?;
    let mut drawn = vec![first as u32];
    if input.max_tokens <= 1 || stop_after <= 1 {
        return Ok(drawn);
    }
    let tok = Channel::from([first]);
    let indptr = Channel::from([0u32, 1u32]);
    let rng = Channel::from([input.seed, input.drawn + 1]);
    let out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
    let fwd = ForwardPass::new();
    fwd.embed(&tok, &indptr)?;
    fwd.state(&[ws], StateGeometry::folding())?;
    let (tok_in, sampled) = (tok.clone(), out.clone());
    let greedy = input.greedy;
    fwd.epilogue(move || {
        let r = rng.take();
        let next = pick(intrinsics::output(0, "logits"), &r, greedy);
        tok_in.put(&next);
        sampled.put(&next);
        rng.put(&(&r + iota(2)));
    });
    run_ahead(pipe, &fwd, input.max_tokens - 1, async || {
        let t = out.take_host::<Vec<i32>>().await?;
        drawn.push(*t.first().unwrap_or(&0) as u32);
        Ok(if drawn.len() >= stop_after {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        })
    })
    .await?;
    Ok(drawn)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if let Some(stop_after) = input.stop_after {
        let pipe = Pipeline::new();
        let generated = draw_ahead(&input.ws, &pipe, &input, stop_after).await?;
        pipe.close();
        input.ws.give("ws")?;
        return Ok(Output {
            tokens: generated,
            gaps: Vec::new(),
        });
    }
    let ws = input.ws;
    let pipe = Pipeline::new();
    let mut tokens: Vec<i32> = input.tokens.iter().map(|&t| t as i32).collect();
    let mut generated = Vec::with_capacity(input.max_tokens);
    let mut gaps = Vec::with_capacity(input.max_tokens);
    for i in 0..input.max_tokens as u32 {
        let (next, gap) =
            draw(&ws, &pipe, &tokens, input.seed, input.drawn + i, input.greedy).await?;
        generated.push(next as u32);
        gaps.push(gap);
        tokens = vec![next];
    }
    pipe.close();
    ws.give("ws")?;
    Ok(Output {
        tokens: generated,
        gaps: if input.greedy { gaps } else { Vec::new() },
    })
}
