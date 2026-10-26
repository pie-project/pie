//! Same-model contrastive decoding.
//!
//! Two forward passes use the bound model with independent states. The
//! expert sees the complete causal context; the amateur sees only a short
//! attention window. Tokens are selected by
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

fn contrastive_pick(amateur_logits: &Tensor, lambda: f32, alpha: f32, vocab: u32) -> Tensor {
    let expert = log_softmax(intrinsics::output(0, "logits"));
    let amateur = log_softmax(amateur_logits);
    let score = &expert - &amateur * lambda;

    let best = reduce_max(&expert);
    let threshold = best + alpha.ln();
    let plausible = ge(&expert, broadcast(threshold, [vocab]));
    let neg_inf = broadcast(f32::NEG_INFINITY, [vocab]);
    reshape(reduce_argmax(select(plausible, score, neg_inf)), [1])
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

    // Not shared with the expert: the amateur's windowed state must not leak into it.
    let amateur_ws = WorkingSet::new();

    let prompt_i32 = prompt.iter().map(|&token| token as i32).collect::<Vec<_>>();
    let amateur_prompt = Channel::from(prompt_i32.clone()).named("amateur_prompt");
    let amateur_prefill_embed_indptr =
        Channel::from([0u32, n]).named("amateur_prefill_embed_indptr");
    let amateur_prefill_mask = Channel::from_shaped(
        [n, span],
        (0..n)
            .flat_map(|query| {
                (0..span).map(move |key| key <= query && key.saturating_add(window) > query)
            })
            .collect::<Vec<_>>(),
    );
    let amateur_prefill_out = Channel::new([vocab], dtype::f32).named("amateur_prefill_logits");

    let amateur_prefill = ForwardPass::new();
    amateur_prefill.embed(&amateur_prompt, &amateur_prefill_embed_indptr)?;
    amateur_prefill.mask(&amateur_prefill_mask)?;
    amateur_prefill.state(&[&amateur_ws], StateGeometry::folding())?;
    amateur_prefill.epilogue(move || {
        amateur_prefill_out.put(intrinsics::output(0, "logits"));
    });

    let pipeline = Pipeline::new();
    amateur_prefill
        .submit(&pipeline)
        .context("amateur prefill")?;
    let first_amateur_logits = amateur_prefill_out.take_host::<Vec<f32>>().await?;

    // The expert keeps the complete context and consumes the amateur logits in
    // its epilogue to perform the contrastive token selection on-device.
    let expert_ws = WorkingSet::new();
    let expert_prompt = Channel::from(prompt_i32).named("expert_prompt");
    let expert_prefill_embed_indptr = Channel::from([0u32, n]).named("expert_prefill_embed_indptr");
    let expert_prefill_amateur = Channel::new([vocab], dtype::f32).named("expert_prefill_amateur");
    let first_out = Channel::new([1], dtype::i32).named("first_token");
    let lambda = input.lambda;
    let alpha = input.alpha;

    let expert_prefill = ForwardPass::new();
    expert_prefill.embed(&expert_prompt, &expert_prefill_embed_indptr)?;
    expert_prefill.state(&[&expert_ws], StateGeometry::folding())?;
    expert_prefill.epilogue(move || {
        let token = contrastive_pick(&expert_prefill_amateur.take(), lambda, alpha, vocab);
        first_out.put(token);
    });

    expert_prefill_amateur.put(first_amateur_logits);
    expert_prefill.submit(&pipeline).context("expert prefill")?;
    let first = read_expert_token(&first_out).await?;

    let mut generated = Vec::with_capacity(input.max_tokens);
    if !stop_tokens.contains(&first) {
        generated.push(first);
    }
    if generated.len() >= input.max_tokens || stop_tokens.contains(&first) {
        return model::decode(&generated);
    }

    let amateur_token = Channel::new([1], dtype::i32).named("amateur_token");
    let amateur_fill = Channel::from([n + 1]).named("amateur_fill");
    let amateur_mask = Channel::from_shaped(
        [1, span],
        (0..span)
            .map(|key| key <= n && key.saturating_add(window) > n)
            .collect::<Vec<_>>(),
    );
    let amateur_logits_out = Channel::new([vocab], dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("amateur_logits");

    let amateur_decode = ForwardPass::new();
    let amateur_embed_indptr = Channel::from([0u32, 1]).named("amateur_embed_indptr");
    amateur_decode.embed(&amateur_token, &amateur_embed_indptr)?;
    amateur_decode.mask(&amateur_mask)?;
    amateur_decode.state(&[&amateur_ws], StateGeometry::folding())?;
    amateur_decode.epilogue(move || {
        let base = amateur_fill.take();
        let columns = iota(span);
        let base_columns = broadcast(reshape(&base, [1]), [span]);
        let next_mask = reshape(
            and(
                le(&columns, &base_columns),
                gt(&columns + window, &base_columns),
            ),
            [1, span],
        );

        amateur_logits_out.put(intrinsics::output(0, "logits"));
        amateur_fill.put(&base + 1u32);
        amateur_mask.put(next_mask);
    });

    let expert_token = Channel::from([first as i32]).named("expert_token");
    let expert_embed_indptr = Channel::from([0u32, 1]).named("expert_embed_indptr");
    let expert_amateur = Channel::writer([vocab], dtype::f32).named("expert_amateur_logits");
    let expert_token_out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("expert_token_out");

    let expert_decode = ForwardPass::new();
    expert_decode.embed(&expert_token, &expert_embed_indptr)?;
    expert_decode.state(&[&expert_ws], StateGeometry::folding())?;
    expert_decode.epilogue(move || {
        let token = contrastive_pick(&expert_amateur.take(), lambda, alpha, vocab);
        expert_token.put(&token);
        expert_token_out.put(&token);
    });

    // No run-ahead is available here: the expert's pick for step k needs the
    // amateur's logits for step k, and the amateur's input for step k+1 is
    // that very pick. Every submit is therefore preceded by the host put of
    // the Writer channel the fire takes.
    let budget = input.max_tokens.saturating_sub(generated.len());
    amateur_token.put(vec![first as i32]);
    amateur_decode.submit(&pipeline).context("amateur decode")?;

    for step in 0..budget {
        let amateur_logits = amateur_logits_out.take_host::<Vec<f32>>().await?;

        expert_amateur.put(amateur_logits);
        expert_decode.submit(&pipeline).context("expert decode")?;
        let token = read_expert_token(&expert_token_out).await?;
        if stop_tokens.contains(&token) {
            break;
        }
        generated.push(token);
        if step + 1 < budget {
            amateur_token.put(vec![token as i32]);
            amateur_decode.submit(&pipeline).context("amateur decode")?;
        }
    }
    // Every submitted fire was drained above.
    pipeline.close();

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
