//! The ordinary end of a guest that folds by a count the device computes:
//! it takes the working set its client moved in, writes the prompt, runs the
//! SDK's `run_ahead` to its end (which closes the pipeline as soon as its
//! budget is submitted), and gives the set back. Its client then stats,
//! saves and reopens what was given.

use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

#[derive(Deserialize)]
struct Input {
    /// The handle the client moved in.
    ws: WorkingSet,
    prompt: String,
    max_tokens: usize,
}

#[derive(Serialize)]
struct Output {
    tokens: Vec<u32>,
    prompt_tokens: u32,
    frame_size: usize,
    window: u32,
    /// The set's frozen length as the runtime states it when `run_ahead`
    /// returns: the pipeline is closed, the guest read every answer.
    bound: u32,
}

fn greedy(logits: Tensor) -> Tensor {
    reshape(reduce_argmax(&logits), [1])
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if input.max_tokens < 2 {
        return Err("max_tokens must be at least 2".into());
    }
    let ws = input.ws;
    let mut prompt = inferlet::chat::prefix();
    prompt.extend(model::encode(&input.prompt));
    if prompt.is_empty() {
        return Err("empty prompt".into());
    }
    let n = prompt.len() as u32;
    let pipe = Pipeline::new();
    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);

    // The prompt, folded whole by the host's count.
    let first = {
        let prompt: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let toks = Channel::from(&prompt[..]);
        let indptr = Channel::from([0u32, n]);
        let out = Channel::new([1], dtype::i32);
        let fwd = ForwardPass::new();
        fwd.embed(&toks, &indptr)?;
        fwd.state(&[&ws], StateGeometry::folding())?;
        let drawn = out.clone();
        fwd.epilogue(move || {
            drawn.put(greedy(intrinsics::output(0, "logits")));
        });
        fwd.submit(&pipe).context("prefill submit")?;
        out.take_host::<i32>().await.context("prefill drain")?
    };
    generated.push(first as u32);

    // Every decode fire folds by the count its own epilogue put: one slot,
    // drawn from the fire's answer, so no host can derive it.
    {
        let tok_in = Channel::from([first]);
        let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        let indptr = Channel::from([0u32, 1]);
        let fold = Channel::from([1u32]);
        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &indptr)?;
        fwd.state(
            &[&ws],
            StateGeometry {
                fold: Fold::Count(&fold),
                ..StateGeometry::default()
            },
        )?;
        let answers = tok_out.clone();
        fwd.epilogue(move || {
            let token = greedy(intrinsics::output(0, "logits"));
            tok_in.put(&token);
            answers.put(&token);
            fold.put(cast(ge(&token, Tensor::constant(0i32)), dtype::u32));
        });
        let read = inferlet::eta::run_ahead(&pipe, &fwd, input.max_tokens - 1, async || {
            let token = tok_out.take_host::<i32>().await.context("decode drain")?;
            generated.push(token as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
        if read != input.max_tokens - 1 {
            return Err(format!("run_ahead read {read} answers"));
        }
    }
    let bound = ws.folded();
    drop(pipe);
    ws.give("ws")?;

    Ok(Output {
        tokens: generated,
        prompt_tokens: n,
        frame_size: frame_size(),
        window: inferlet::model::run_ahead_window(),
        bound,
    })
}
