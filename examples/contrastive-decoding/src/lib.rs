//! Same-model contrastive decoding.
//!
//! Each forward pass runs the bound model over two lanes with independent
//! states: the expert sees the complete causal context; the amateur sees only a
//! short attention window. Tokens are selected on-device by
//! `log p_expert - lambda * log p_amateur`, restricted to tokens whose expert
//! probability is at least `alpha` times the expert's maximum probability.
//!
//! Every `keep_every` decodes one decode keeps (`Fold::Keep`, a row a lane):
//! the amateur drops the rows that left its window and leaves the window
//! buffered, the expert commits all it holds. Between keeps the mask speaks
//! of each lane's buffer (column `c` is slot `folded + c`): the amateur's
//! window, the expert's rows since its keep. So the amateur holds its window
//! and the rows since, and on a model with recurrent layers neither lane
//! replays its history.

use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_amateur_window")]
    amateur_window: u32,
    #[serde(default = "default_lambda")]
    lambda: f32,
    #[serde(default = "default_alpha")]
    alpha: f32,
    /// Decodes between keeps.
    #[serde(default = "default_keep_every")]
    keep_every: u32,
}

fn default_keep_every() -> u32 {
    256
}

/// Each lane's mask row over `width` columns for queries at buffer columns
/// `query` (amateur, expert): the amateur's window, the expert's every row.
fn lane_masks(width: u32, query: [u32; 2], window: u32) -> Vec<bool> {
    (0..width)
        .map(|key| key <= query[0] && key.saturating_add(window) > query[0])
        .chain((0..width).map(|key| key <= query[1]))
        .collect()
}

fn default_prompt() -> String {
    "Explain why the sky appears blue.".into()
}

fn default_max_tokens() -> usize {
    256
}

fn default_amateur_window() -> u32 {
    8
}

fn default_lambda() -> f32 {
    0.5
}

fn default_alpha() -> f32 {
    0.1
}

fn contrastive_pick(
    expert_logits: &Tensor,
    amateur_logits: &Tensor,
    lambda: f32,
    alpha: f32,
    vocab: u32,
) -> Tensor {
    let expert = log_softmax(expert_logits);
    let amateur = log_softmax(amateur_logits);
    let score = &expert - &amateur * lambda;

    let best = reduce_max(&expert);
    let threshold = best + alpha.ln();
    let plausible = ge(&expert, broadcast(threshold, [vocab]));
    let neg_inf = broadcast(f32::NEG_INFINITY, [vocab]);
    reshape(
        cast(reduce_argmax(select(plausible, score, neg_inf)), dtype::i32),
        [1],
    )
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if input.amateur_window == 0 {
        return Err("amateur_window must be at least 1".into());
    }
    if !input.lambda.is_finite() || input.lambda < 0.0 {
        return Err("lambda must be finite and non-negative".into());
    }
    if !input.alpha.is_finite() || !(0.0 < input.alpha && input.alpha <= 1.0) {
        return Err("alpha must be finite, greater than 0, and at most 1".into());
    }
    if input.max_tokens == 0 {
        return Ok(String::new());
    }

    let max_tokens =
        u32::try_from(input.max_tokens).map_err(|_| "max_tokens exceeds the u32 range")?;
    let vocab = model::output_vocab_size();

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = u32::try_from(prompt.len()).map_err(|_| "prompt is too long")?;
    let span = n
        .checked_add(max_tokens)
        .and_then(|value| value.checked_add(1))
        .ok_or("prompt and generation length exceed the u32 range")?;
    let window = input.amateur_window.min(span);
    span.checked_add(window)
        .ok_or("amateur attention geometry exceeds the u32 range")?;
    let stop_tokens = chat::stop_tokens();
    let lambda = input.lambda;
    let alpha = input.alpha;

    let amateur_ws = WorkingSet::new();
    let expert_ws = WorkingSet::new();
    let streams = [&amateur_ws, &expert_ws];
    let pipeline = Pipeline::new();

    let prompt_i32 = prompt.iter().map(|&token| token as i32).collect::<Vec<_>>();
    let prefill_tokens = prompt_i32
        .iter()
        .chain(&prompt_i32)
        .copied()
        .collect::<Vec<_>>();
    let prefill_prompt = Channel::from(prefill_tokens);
    let prefill_embed_indptr = Channel::from([0u32, n, 2 * n]);
    let prefill_mask_data = (0..n)
        .flat_map(|query| {
            (0..span).map(move |key| key <= query && key.saturating_add(window) > query)
        })
        .chain((0..n).flat_map(|query| (0..span).map(move |key| key <= query)))
        .collect::<Vec<_>>();
    let prefill_mask = Channel::from_shaped([2 * n, span], prefill_mask_data);
    let first_out = Channel::new([1], dtype::i32);

    let prefill = ForwardPass::new();
    prefill.embed(&prefill_prompt, &prefill_embed_indptr)?;
    prefill.mask(&prefill_mask)?;
    prefill.state(&streams, StateGeometry::default())?;
    prefill.epilogue(move || {
        let logits = reshape(intrinsics::output(0, "logits"), [2, vocab]);
        let amateur = reshape(gather(&logits, iota(1)), [vocab]);
        let expert = reshape(gather(&logits, iota(1) + 1u32), [vocab]);
        let token = contrastive_pick(&expert, &amateur, lambda, alpha, vocab);
        first_out.put(token);
    });
    prefill.submit(&pipeline).context("prefill")?;
    let first = read_expert_token(&first_out).await?;

    let mut generated = Vec::with_capacity(input.max_tokens);
    if !stop_tokens.contains(&first) {
        generated.push(first);
    }
    if generated.len() >= input.max_tokens || stop_tokens.contains(&first) {
        return model::decode(&generated);
    }

    pipeline.close();
    // Each lane's set: rows `0..end`, committed below `folded`; its next
    // query is slot `end`. Amateur first, expert second.
    let every = input.keep_every.max(1);
    let mut folded = [0u32; 2];
    let mut end = [n; 2];
    let mut token = first as i32;
    let mut stopped = false;
    while !stopped && generated.len() < input.max_tokens {
        let left = (input.max_tokens - generated.len()) as u32;
        let steps = every.min(left).saturating_sub(1);
        let query = [end[0] - folded[0], end[1] - folded[1]];
        let width = query[0].max(query[1]) + steps + 1;
        if steps > 0 {
            let toks = Channel::from([token, token]);
            let embed_indptr = Channel::from([0u32, 1, 2]);
            // After the first decode the expert has folded everything: its
            // next query is column 0 from then on.
            let fill = Channel::from([query[0] + 1, 0]);
            let mask = Channel::from_shaped([2, width], lane_masks(width, query, window));
            let out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
            // The expert hides nothing: it folds every row as it decodes, so
            // its buffer stays empty; the amateur buffers until its keep.
            let folds = Channel::from([0u32, u32::MAX]);
            let decode = ForwardPass::new();
            decode.embed(&toks, &embed_indptr)?;
            decode.mask(&mask)?;
            decode.state(
                &streams,
                StateGeometry {
                    fold: Fold::Count(&folds),
                    ..StateGeometry::default()
                },
            )?;
            decode.epilogue(move || {
                let logits = reshape(intrinsics::output(0, "logits"), [2, vocab]);
                let amateur = reshape(gather(&logits, iota(1)), [vocab]);
                let expert = reshape(gather(&logits, iota(1) + 1u32), [vocab]);
                let token = contrastive_pick(&expert, &amateur, lambda, alpha, vocab);

                // Each lane's next query column: [amateur, expert].
                let base = fill.take();
                let flat = iota(2 * width);
                let row = &flat / width;
                let col = &flat - &row * width;
                let base_cols = reshape(broadcast(reshape(&base, [2, 1]), [2, width]), [2 * width]);
                let causal = le(&col, &base_cols);
                let in_window = gt(&col + window, &base_cols);
                let is_expert = eq(&row, 1u32);
                let next_mask = reshape(and(&causal, or(&is_expert, &in_window)), [2, width]);

                toks.put(broadcast(&token, [2]));
                // The amateur's next column moves on; the expert's stays 0.
                fill.put(&base + cast(eq(iota(2), 0u32), dtype::u32));
                mask.put(next_mask);
                out.put(token);
            });
            // `run_ahead` closes its pipeline once its budget is posted.
            let segment = Pipeline::new();
            let mut taken = 0u32;
            run_ahead(&segment, &decode, steps as usize, async || {
                let next = read_expert_token(&out).await?;
                taken += 1;
                token = next as i32;
                if stop_tokens.contains(&next) {
                    stopped = true;
                    return Ok(ControlFlow::Break(()));
                }
                generated.push(next);
                Ok(ControlFlow::Continue(()))
            })
            .await?;
            end = [end[0] + taken, end[1] + taken];
            folded[1] = end[1];
            if stopped || generated.len() >= input.max_tokens {
                break;
            }
        }
        // The keeping decode: the amateur lets go of what left its window,
        // the expert commits all it holds.
        use inferlet::eta::keep::{BUFFER, COMMIT, DROP};
        let query = [end[0] - folded[0], end[1] - folded[1]];
        let width = query[0].max(query[1]) + 1;
        let amateur: Vec<u32> = (0..width)
            .map(|c| if c + window > query[0] { BUFFER } else { DROP })
            .collect();
        let rows: Vec<u32> = amateur.iter().copied().chain((0..width).map(|_| COMMIT)).collect();
        let keep = Channel::from_shaped([2, width], rows);
        let mask = Channel::from_shaped([2, width], lane_masks(width, query, window));
        let toks = Channel::from([token, token]);
        let out = Channel::new([1], dtype::i32);
        let decode = ForwardPass::new();
        decode.embed(&toks, &Channel::from([0u32, 1, 2]))?;
        decode.mask(&mask)?;
        decode.state(
            &streams,
            StateGeometry {
                fold: Fold::Keep(&keep),
                ..StateGeometry::default()
            },
        )?;
        decode.epilogue(move || {
            let logits = reshape(intrinsics::output(0, "logits"), [2, vocab]);
            let amateur = reshape(gather(&logits, iota(1)), [vocab]);
            let expert = reshape(gather(&logits, iota(1) + 1u32), [vocab]);
            out.put(contrastive_pick(&expert, &amateur, lambda, alpha, vocab));
        });
        let keeping = Pipeline::new();
        decode.submit(&keeping).context("contrastive keep")?;
        let next = read_expert_token(&out).await?;
        keeping.close();
        let dropped = amateur[..=query[0] as usize].iter().filter(|&&at| at == DROP).count() as u32;
        end = [end[0] + 1 - dropped, end[1] + 1];
        folded = [folded[0], end[1]];
        token = next as i32;
        if stop_tokens.contains(&next) {
            stopped = true;
        } else {
            generated.push(next);
        }
    }

    model::decode(&generated)
}

async fn read_expert_token(channel: &Channel) -> Result<u32> {
    channel
        .take_host::<Vec<i32>>()
        .await?
        .first()
        .copied()
        .map(|token| token as u32)
        .context("expert returned no token")
}
