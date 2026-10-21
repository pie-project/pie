//! A pass that embeds the tokens the pass ahead of it in its frame picked,
//! with the host never reading them in between.
//!
//! ```text
//! pick     [a]               one row, the host's anchor   -> t = argmax
//! embed    [t, t, .., t]     `width` rows                 -> u[i] = argmax(row i)
//! next     a = u[width - 1]
//! ```
//!
//! `chain` submits the two as one frame, `pick`'s epilogue putting `[t; width]`
//! on the channel `embed` takes its tokens from; off, the host takes `t` and
//! hands `embed` the same row from a channel of its own. The two must write
//! the same tokens: the rows, the slots and the arithmetic are one, and only
//! who carried `t` differs. The rows are what they are because they are cheap
//! to lay out on the device, not because a model would write them.

use inferlet::eta::prelude::*;
use inferlet::{Result, chat};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_rounds")]
    rounds: u32,
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default)]
    chain: bool,
}

fn default_prompt() -> String {
    "Write a short poem about the sea.".into()
}

fn default_rounds() -> u32 {
    8
}

fn default_width() -> u32 {
    3
}

#[derive(Serialize)]
struct Output {
    /// Per round, `t` and then `u[0..width]`.
    tokens: Vec<i32>,
    /// Frames submitted after the prefill.
    frames: u32,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let width = input.width.max(1);
    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue_thinking(false));
    let prompt: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    let n = prompt.len() as u32;
    let toks = Channel::from(prompt.as_slice()).named("toks_p");
    let indptr = Channel::from([0u32, n]).named("embed_indptr_p");
    let readout = Channel::from([n - 1]).named("readout_p");
    let next = Channel::new([1], dtype::i32).named("next_p");
    let fwd = ForwardPass::new();
    fwd.embed(&toks, &indptr)?;
    fwd.readout(&readout)?;
    fwd.state(&[&ws], StateGeometry::folding())?;
    {
        let next = next.clone();
        fwd.epilogue(move || {
            next.put(&reshape(reduce_argmax(intrinsics::logits()), [1]));
        });
    }
    fwd.submit(&pipe)?;
    let mut anchor = next.take_host::<Vec<i32>>().await?[0];

    let mut tokens = Vec::new();
    let mut frames = 0;
    for _ in 0..input.rounds {
        // ── pick: one row from the host's anchor ─────────────────────────
        let toks = Channel::from([anchor]).named("toks_a");
        let indptr = Channel::from([0u32, 1]).named("embed_indptr_a");
        let picked = Channel::new([1], dtype::i32).named("picked_a");
        let window = Channel::new([width], dtype::i32).named("window_b");
        let pick = ForwardPass::new();
        pick.embed(&toks, &indptr)?;
        pick.state(&[&ws], StateGeometry::folding())?;
        {
            let (picked, chain) = (picked.clone(), input.chain);
            pick.epilogue(move || {
                let t = reshape(reduce_argmax(intrinsics::logits()), [1]);
                if chain {
                    window.put(&broadcast(&t, [width]));
                } else {
                    picked.put(&t);
                }
            });
        }

        // ── embed: `width` rows of what pick picked ──────────────────────
        let indptr = Channel::from([0u32, width]).named("embed_indptr_b");
        let readout = Channel::from_iter(0..width).named("readout_b");
        let answer = Channel::new([width + 1], dtype::i32).named("answer_b");
        let embed = ForwardPass::new();
        let fed = if input.chain {
            window
        } else {
            submit_frame(&pipe, &[Some(&pick)])?;
            frames += 1;
            let t = picked.take_host::<Vec<i32>>().await?[0];
            Channel::from(vec![t; width as usize].as_slice()).named("fed_b")
        };
        embed.embed(&fed, &indptr)?;
        embed.readout(&readout)?;
        embed.state(&[&ws], StateGeometry::folding())?;
        {
            let answer = answer.clone();
            embed.epilogue(move || {
                let u = reshape(reduce_argmax(intrinsics::logits()), [width]);
                let i = iota(width + 1);
                let row = min_elem(
                    max_elem(&i, Tensor::constant(1u32)) - Tensor::constant(1u32),
                    Tensor::constant(width - 1),
                );
                // `[t, u[0..width]]`: the embedded row's token leads.
                let t = gather(&fed.take(), iota(1));
                answer.put(&select(
                    eq(&i, Tensor::constant(0u32)),
                    broadcast(&t, [width + 1]),
                    gather(&u, &row),
                ));
            });
        }
        if input.chain {
            submit_frame(&pipe, &[Some(&pick), Some(&embed)])?;
        } else {
            submit_frame(&pipe, &[Some(&embed)])?;
        }
        frames += 1;
        let answer = answer.take_host::<Vec<i32>>().await?;
        anchor = answer[width as usize];
        tokens.extend(answer);
    }
    pipe.close();
    Ok(Output { tokens, frames })
}
