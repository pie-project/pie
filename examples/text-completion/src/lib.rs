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
    /// Last on purpose: `pie::sweep::fleet` reads a lane's answer from the last `[`.
    tokens: Vec<u32>,
}

/// The greedy pick over a logits row, as a one-lane `[1]` i32 cell.
fn greedy(logits: Tensor) -> Tensor {
    reshape(reduce_argmax(&logits), [1])
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let max_tokens = input.max_tokens;
    let ws = WorkingSet::new();

    if max_tokens == 0 {
        return Ok(Output {
            text: String::new(),
            count: 0,
            tokens: Vec::new(),
        });
    }

    // The model's opening (`<bos>` where it has one) before the raw text: a
    // gemma without it answers noise.
    let mut prompt = inferlet::chat::prefix();
    prompt.extend(model::encode(&input.prompt));
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;

    let pipe = Pipeline::new();
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks = Channel::from(&prompt_i32[..]).named("toks_p");
    let embed_indptr = Channel::from([0u32, n]).named("embed_indptr_p");
    let tok_out = Channel::new([1], dtype::i32).named("tok_out_p");

    let fwd = ForwardPass::new();
    fwd.embed(&toks, &embed_indptr)?;
    fwd.state(&[&ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        tok_out.put(greedy(intrinsics::logits()));
    });
    fwd.submit(&pipe).context("prefill submit")?;
    let first = tok_out.take_host::<i32>().await.context("prefill drain")?;
    generated.push(first as u32);

    // ── DECODE (1-wide, device loop-carried token) ─────────────────────────
    if generated.len() < max_tokens {
        let tok_in = Channel::from([first]).named("tok_in");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let embed_indptr = Channel::from([0u32, 1]).named("embed_indptr");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &embed_indptr)?;
        fwd.state(&[&ws], StateGeometry::folding())?;
        fwd.epilogue(move || {
            let token = greedy(intrinsics::logits());
            tok_in.put(&token);
            tok_out.put(&token);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget, async || {
            let token = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("decode drain @{}", generated.len()))?;
            generated.push(token as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    Ok(Output {
        count: generated.len(),
        text: model::decode(&generated)?,
        tokens: generated,
    })
}
