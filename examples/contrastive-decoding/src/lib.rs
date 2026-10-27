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
    let max_tokens =
        u32::try_from(input.max_tokens).map_err(|_| "max_tokens exceeds the u32 range")?;
    let vocab = model::output_vocab_size();

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());
    let n = u32::try_from(prompt.len()).map_err(|_| "prompt is too long")?;
    let span = n
        .checked_add(max_tokens)
        .and_then(|value| value.checked_add(1))
        .ok_or("prompt and generation length exceed the u32 range")?;
    let window = input.amateur_window.min(span);
    span.checked_add(window)
        .ok_or("amateur attention geometry exceeds the u32 range")?;
    let stop_tokens = chat::stop_tokens();

    let prompt: Vec<i32> = prompt.into_iter().map(|t| t as i32).collect();
    let (&last, head) = prompt.split_last().ok_or("empty prompt")?;
    let h = head.len() as u32;
    let visible = move |query: u32, key: u32| key <= query && key.saturating_add(window) > query;

    // Not shared with the expert: the amateur's windowed state must not leak into it.
    let amateur_ws = WorkingSet::new();
    let expert_ws = WorkingSet::new();
    let pipeline = Pipeline::new();
    if !head.is_empty() {
        let amateur_prefill = ForwardPass::new();
        amateur_prefill.embed(&Channel::from(head), &Channel::from([0u32, h]))?;
        amateur_prefill.mask(&Channel::from_shaped(
            [h, span],
            (0..h)
                .flat_map(|query| (0..span).map(move |key| visible(query, key)))
                .collect::<Vec<_>>(),
        ))?;
        amateur_prefill.state(&[&amateur_ws], StateGeometry::folding())?;
        amateur_prefill
            .submit(&pipeline)
            .context("amateur prefill")?;

        let expert_prefill = ForwardPass::new();
        expert_prefill.embed(&Channel::from(head), &Channel::from([0u32, h]))?;
        expert_prefill.state(&[&expert_ws], StateGeometry::folding())?;
        expert_prefill.submit(&pipeline).context("expert prefill")?;
    }

    let amateur_token = Channel::new([1], dtype::i32);
    let amateur_fill = Channel::from([h + 1]);
    let amateur_mask = Channel::from_shaped(
        [1, span],
        (0..span).map(|key| visible(h, key)).collect::<Vec<_>>(),
    );
    let amateur_logits_out = Channel::new([vocab], dtype::f32).capacity(channel_capacity() as u32);

    let amateur_decode = ForwardPass::new();
    let amateur_embed_indptr = Channel::from([0u32, 1]);
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

    // The expert keeps the complete context and consumes the amateur logits in
    // its epilogue to perform the contrastive token selection on-device.
    let lambda = input.lambda;
    let alpha = input.alpha;
    let expert_token = Channel::from([last]);
    let expert_embed_indptr = Channel::from([0u32, 1]);
    let expert_amateur = Channel::writer([vocab], dtype::f32);
    let expert_token_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);

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
    let mut generated = Vec::with_capacity(input.max_tokens);
    let mut token = last;
    for _ in 0..input.max_tokens {
        amateur_token.put(vec![token]);
        amateur_decode.submit(&pipeline).context("amateur decode")?;
        let amateur_logits = amateur_logits_out.take_host::<Vec<f32>>().await?;

        expert_amateur.put(amateur_logits);
        expert_decode.submit(&pipeline).context("expert decode")?;
        let picked = read_expert_token(&expert_token_out).await?;
        if stop_tokens.contains(&picked) {
            break;
        }
        generated.push(picked);
        token = picked as i32;
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
