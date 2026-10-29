//! Same-model contrastive decoding.
//!
//! Each forward pass runs the bound model over two lanes with independent
//! states: the expert sees the complete causal context; the amateur sees only a
//! short attention window. Tokens are selected on-device by
//! `log p_expert - lambda * log p_amateur`, restricted to tokens whose expert
//! probability is at least `alpha` times the expert's maximum probability.

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
    prefill.state(&streams, StateGeometry::folding())?;
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

    let toks = Channel::from([first as i32, first as i32]);
    let embed_indptr = Channel::from([0u32, 1, 2]);
    let fill = Channel::from([n + 1]);
    let decode_mask_init = (0..span)
        .map(|key| key <= n && key.saturating_add(window) > n)
        .chain((0..span).map(|key| key <= n))
        .collect::<Vec<_>>();
    let mask = Channel::from_shaped([2, span], decode_mask_init);
    let out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);

    let decode = ForwardPass::new();
    decode.embed(&toks, &embed_indptr)?;
    decode.mask(&mask)?;
    decode.state(&streams, StateGeometry::folding())?;
    decode.epilogue(move || {
        let logits = reshape(intrinsics::output(0, "logits"), [2, vocab]);
        let amateur = reshape(gather(&logits, iota(1)), [vocab]);
        let expert = reshape(gather(&logits, iota(1) + 1u32), [vocab]);
        let token = contrastive_pick(&expert, &amateur, lambda, alpha, vocab);

        let base = fill.take();
        let flat = iota(2 * span);
        let row = &flat / span;
        let col = &flat - &row * span;
        let base_cols = broadcast(reshape(&base, [1]), [2 * span]);
        let causal = le(&col, &base_cols);
        let in_window = gt(&col + window, &base_cols);
        let is_expert = eq(&row, 1u32);
        let next_mask = reshape(and(&causal, or(&is_expert, &in_window)), [2, span]);

        toks.put(broadcast(&token, [2]));
        fill.put(&base + 1u32);
        mask.put(next_mask);
        out.put(token);
    });

    let budget = input.max_tokens.saturating_sub(generated.len());
    run_ahead(&pipeline, &decode, budget, async || {
        let token = read_expert_token(&out).await?;
        if stop_tokens.contains(&token) {
            return Ok(ControlFlow::Break(()));
        }
        generated.push(token);
        Ok(ControlFlow::Continue(()))
    })
    .await?;

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
