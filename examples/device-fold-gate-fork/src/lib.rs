//! FORK variant of device-fold-gate: `run_ahead` to its end as the SDK runs it
//! (it closes the pipeline), then `bound` read and the set forked on the closed
//! pipeline. Today that fork is refused ("pipeline is closed"): this variant
//! shows the refusal, the park variant the length.
//!
//! Greedy completion whose decode fires fold their working set by a count the
//! device computes (`fold = "device"`), or by a count the host states
//! (`fold = "host"`, one slot a fire). The gate reads the two runs' tokens:
//! every decode fire after the first is another pass of a set the fire before
//! it folded, so under `device` it goes on only once that fold's count has
//! reached the runtime.
//!
//! `every = n` (device only) folds nothing for `n - 1` fires and `n` slots at
//! the `n`th: the count the runtime hears is then not the rows the fire wrote.
//! The count is drawn from the fire's own answer (`token >= 0`, always one),
//! so no host can derive it.
//!
//! `ahead = false` puts one decode fire in a frame and reads its answer
//! before the next; by default the fires run ahead, `frame_size()` to a frame.

use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// `device` or `host`: who counts each decode fire's fold.
    #[serde(default = "default_fold")]
    fold: String,
    #[serde(default = "default_every")]
    every: u32,
    #[serde(default = "default_ahead")]
    ahead: bool,
}

fn default_prompt() -> String {
    "The capital of France is".into()
}

fn default_max_tokens() -> usize {
    32
}

fn default_fold() -> String {
    "device".into()
}

fn default_every() -> u32 {
    1
}

fn default_ahead() -> bool {
    true
}

#[derive(Serialize)]
struct Output {
    fold: String,
    every: u32,
    text: String,
    count: usize,
    tokens: Vec<u32>,
    prompt_tokens: u32,
    frame_size: usize,
    decode_ms: u64,
    /// The set's frozen length as the runtime states it once every fire's
    /// answer was read. An answer can be read ahead of its fire's fold
    /// settling: this is then the length before that fold.
    bound: u32,
    /// The same length after a fork of the set, which waits for the set's
    /// folds to settle: the exact one.
    folded: u32,
    /// The length the fork's child states.
    forked: u32,
}

fn greedy(logits: Tensor) -> Tensor {
    reshape(reduce_argmax(&logits), [1])
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let device = match input.fold.as_str() {
        "device" => true,
        "host" => false,
        other => return Err(format!("fold is `device` or `host`, not `{other}`")),
    };
    let every = input.every.max(1);
    if !device && every != 1 {
        return Err("a host fold here is one slot a fire: `every` is the device's".into());
    }
    if input.max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }
    let max_tokens = input.max_tokens;
    let ws = WorkingSet::new();

    let mut prompt = inferlet::chat::prefix();
    prompt.extend(model::encode(&input.prompt));
    if prompt.is_empty() {
        return Err("empty prompt".into());
    }
    let n = prompt.len() as u32;

    let pipe = Pipeline::new();
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut decode_ms = 0;

    // The prompt, folded whole by the host's count.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks = Channel::from(&prompt_i32[..]);
    let embed_indptr = Channel::from([0u32, n]);
    let tok_out = Channel::new([1], dtype::i32);

    let fwd = ForwardPass::new();
    fwd.embed(&toks, &embed_indptr)?;
    fwd.state(&[&ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        tok_out.put(greedy(intrinsics::output(0, "logits")));
    });
    fwd.submit(&pipe).context("prefill submit")?;
    let first = tok_out.take_host::<i32>().await.context("prefill drain")?;
    generated.push(first as u32);

    if generated.len() < max_tokens {
        let tok_in = Channel::from([first]);
        let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        let embed_indptr = Channel::from([0u32, 1]);
        // Fire 0's count; under `device` each fire's epilogue puts the next.
        let fold = Channel::from([if every == 1 { 1u32 } else { 0 }]);
        let turn = Channel::from([0u32]);

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &embed_indptr)?;
        fwd.state(
            &[&ws],
            StateGeometry {
                fold: Fold::Count(&fold),
                ..StateGeometry::default()
            },
        )?;
        fwd.epilogue(move || {
            let token = greedy(intrinsics::output(0, "logits"));
            tok_in.put(&token);
            tok_out.put(&token);
            if device {
                let next = turn.take() + Tensor::constant(1u32);
                turn.put(&next);
                let due = eq(
                    &next % Tensor::constant(every),
                    Tensor::constant(every - 1),
                );
                let drawn = cast(ge(&token, Tensor::constant(0i32)), dtype::u32);
                let count = select(
                    &due,
                    broadcast(Tensor::constant(every), [1]),
                    broadcast(Tensor::constant(0u32), [1]),
                );
                fold.put(count * &drawn);
            }
        });

        let budget = max_tokens - 1;
        let began = std::time::Instant::now();
        if input.ahead {
            run_ahead(&pipe, &fwd, budget, async || {
                let token = tok_out
                    .take_host::<i32>()
                    .await
                    .with_context(|| format!("decode drain @{}", generated.len()))?;
                generated.push(token as u32);
                Ok(ControlFlow::Continue(()))
            })
            .await?;
        } else {
            for fire in 0..budget {
                fwd.submit(&pipe)
                    .with_context(|| format!("decode submit @{fire}"))?;
                let token = tok_out
                    .take_host::<i32>()
                    .await
                    .with_context(|| format!("decode drain @{fire}"))?;
                generated.push(token as u32);
            }
        }
        decode_ms = began.elapsed().as_millis() as u64;
    }
    let bound = ws.folded();
    let child = ws.fork(&pipe).context("settling fork")?;
    let folded = ws.folded();
    let forked = child.folded();
    pipe.close();

    Ok(Output {
        fold: input.fold,
        every,
        count: generated.len(),
        text: model::decode(&generated)?,
        tokens: generated,
        prompt_tokens: n,
        frame_size: frame_size(),
        decode_ms,
        bound,
        folded,
        forked,
    })
}
