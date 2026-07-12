//! Same-model contrastive decoding.
//!
//! Two forward passes use the bound model with independent KV states. The
//! expert sees the complete causal context; the amateur sees only a short
//! attention window. Tokens are selected by
//! `log p_expert - lambda * log p_amateur`, restricted to tokens whose expert
//! probability is at least `alpha` times the expert's maximum probability.

use inferlet::ptir::prelude::*;
use inferlet::{Result, chat, model as wit_model};
use serde::Deserialize;

const PAGE_T: u32 = 16;
const NUM_LAYERS: u32 = 28;

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

fn bx<T>(value: T) -> &'static T {
    Box::leak(Box::new(value))
}

fn contrastive_pick(amateur_logits: impl AsTensor, lambda: f32, alpha: f32, vocab: u32) -> Tensor {
    let expert = log_softmax(intrinsics::logits());
    let amateur = log_softmax(amateur_logits);
    let score = sub(&expert, mul(&amateur, lambda));

    let best = reduce_max(&expert);
    let threshold = add(best, alpha.ln());
    let plausible = ge(&expert, broadcast(threshold, [vocab]));
    let neg_inf = broadcast(Tensor::constant(f32::NEG_INFINITY), [vocab]);
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
    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = u32::try_from(prompt.len()).map_err(|_| "prompt is too long")?;
    let required_tokens = n
        .checked_add(max_tokens)
        .and_then(|value| value.checked_add(1))
        .ok_or("prompt and generation length exceed the u32 range")?;
    let pool_pages = required_tokens.div_ceil(PAGE_T);
    let pool_len = pool_pages
        .checked_mul(PAGE_T)
        .ok_or("page-rounded KV capacity exceeds the u32 range")?;
    let window = input.amateur_window.min(pool_len);
    pool_len
        .checked_add(window)
        .ok_or("amateur attention geometry exceeds the u32 range")?;
    let stop_tokens = chat::stop_tokens();

    // The amateur uses a dedicated KV pool so its bounded-context hidden states
    // never contaminate the expert's full-context state.
    let amateur_ws: &'static WorkingSet = bx(WorkingSet::new());
    let amateur_slots = amateur_ws
        .reserve(pool_pages)
        .map_err(|error| format!("reserve amateur KV: {error}"))?;
    let amateur_ids: &'static Vec<u32> = bx(amateur_slots.ids().to_vec());

    let prompt_i32 = prompt.iter().map(|&token| token as i32).collect::<Vec<_>>();
    let amateur_prompt = bx(Channel::from(prompt_i32.clone()).named("amateur_prompt"));
    let amateur_prefill_slots = bx(Channel::from(
        (0..n)
            .map(|position| amateur_ids[(position / PAGE_T) as usize])
            .collect::<Vec<_>>(),
    ));
    let amateur_prefill_offsets = bx(Channel::from(
        (0..n).map(|position| position % PAGE_T).collect::<Vec<_>>(),
    ));
    let amateur_prefill_klen = bx(Channel::from(vec![n]));
    let amateur_prefill_pages = bx(Channel::from(amateur_ids.clone()));
    let amateur_prefill_indptr = bx(Channel::from_shaped([2], vec![0u32, pool_pages]));
    let amateur_prefill_mask = bx(Channel::from_shaped(
        [n, pool_len],
        (0..n)
            .flat_map(|query| {
                (0..pool_len).map(move |key| key <= query && key.saturating_add(window) > query)
            })
            .collect::<Vec<_>>(),
    ));
    let amateur_prefill_out = bx(Channel::new([vocab], dtype::f32).named("amateur_prefill_logits"));

    let amateur_prefill: ForwardPass<'static> = ForwardPass::new();
    amateur_prefill.embed(amateur_prompt, Tensor::constant(vec![0u32, n]));
    amateur_prefill.attn_working_set(amateur_ws, amateur_prefill_klen);
    amateur_prefill.port_channel(Port::Pages, amateur_prefill_pages);
    amateur_prefill.port_channel(Port::PageIndptr, amateur_prefill_indptr);
    amateur_prefill.port_channel(Port::WSlot, amateur_prefill_slots);
    amateur_prefill.port_channel(Port::WOff, amateur_prefill_offsets);
    amateur_prefill.attn_mask(amateur_prefill_mask);
    amateur_prefill.epilogue(move || {
        amateur_prefill_out.put(intrinsics::logits());
    });

    let amateur_prefill_pipeline = Pipeline::new();
    amateur_prefill
        .submit(&amateur_prefill_pipeline)
        .map_err(|error| format!("amateur prefill: {error}"))?;
    let first_amateur_logits = amateur_prefill_out
        .take()
        .get::<f32>()
        .map_err(|error| format!("read amateur prefill logits: {error}"))?;
    amateur_prefill_pipeline.close();

    // The expert keeps the complete context and consumes the amateur logits in
    // its epilogue to perform the contrastive token selection on-device.
    let expert_ws: &'static WorkingSet = bx(WorkingSet::new());
    let expert_prompt = bx(Channel::from(prompt_i32).named("expert_prompt"));
    let expert_prefill_klen = bx(Channel::from(vec![n]));
    let expert_prefill_amateur =
        bx(Channel::new([vocab], dtype::f32).named("expert_prefill_amateur"));
    let first_out = bx(Channel::new([1], dtype::i32).named("first_token"));
    let lambda = input.lambda;
    let alpha = input.alpha;

    let expert_prefill: ForwardPass<'static> = ForwardPass::new();
    expert_prefill.embed(expert_prompt, Tensor::constant(vec![0u32, n]));
    expert_prefill.attn_working_set(expert_ws, expert_prefill_klen);
    expert_prefill.epilogue(move || {
        let token = contrastive_pick(expert_prefill_amateur.take(), lambda, alpha, vocab);
        first_out.put(token);
    });

    expert_prefill_amateur.put(first_amateur_logits);
    let expert_prefill_pipeline = Pipeline::new();
    expert_prefill
        .submit(&expert_prefill_pipeline)
        .map_err(|error| format!("expert prefill: {error}"))?;
    let first = read_expert_token(first_out)?;
    expert_prefill_pipeline.close();

    let mut generated = Vec::with_capacity(input.max_tokens);
    if !stop_tokens.contains(&first) {
        generated.push(first);
    }
    if generated.len() >= input.max_tokens || stop_tokens.contains(&first) {
        return wit_model::decode(&generated);
    }

    let amateur_token = bx(Channel::new([1], dtype::i32).named("amateur_token"));
    let amateur_position = bx(Channel::from(vec![n]).named("amateur_position"));
    let amateur_fill = bx(Channel::from(vec![n + 1]).named("amateur_fill"));
    let amateur_klen = bx(Channel::from(vec![n + 1]).named("amateur_klen"));
    let amateur_write_slot = bx(Channel::from(vec![amateur_ids[(n / PAGE_T) as usize]]));
    let amateur_write_offset = bx(Channel::from(vec![n % PAGE_T]));
    let amateur_mask = bx(Channel::from_shaped(
        [1, pool_len],
        (0..pool_len)
            .map(|key| key <= n && key.saturating_add(window) > n)
            .collect::<Vec<_>>(),
    ));
    let amateur_pages = bx(Channel::from(amateur_ids.clone()));
    let amateur_page_indptr = bx(Channel::from_shaped([2], vec![0u32, pool_pages]));
    let amateur_ids_input = bx(Channel::new([pool_pages], dtype::u32).named("amateur_pool_ids"));
    let amateur_logits_out = bx(Channel::new([vocab], dtype::f32).named("amateur_logits"));

    let amateur_decode: ForwardPass<'static> = ForwardPass::new();
    amateur_decode.embed(amateur_token, Tensor::constant(vec![0u32, 1]));
    amateur_decode.positions(amateur_position);
    amateur_decode.attn_working_set(amateur_ws, amateur_klen);
    amateur_decode.port_channel(Port::Pages, amateur_pages);
    amateur_decode.port_channel(Port::PageIndptr, amateur_page_indptr);
    amateur_decode.port_channel(Port::WSlot, amateur_write_slot);
    amateur_decode.port_channel(Port::WOff, amateur_write_offset);
    amateur_decode.attn_mask(amateur_mask);
    amateur_decode.epilogue(move || {
        let base = amateur_fill.take().tensor();
        let ids = amateur_ids_input.take();
        let columns = iota(pool_len);
        let base_columns = broadcast(reshape(&base, [1]), [pool_len]);
        let next_mask = reshape(
            and(
                le(&columns, &base_columns),
                gt(add(&columns, window), &base_columns),
            ),
            [1, pool_len],
        );
        let next = add(&base, 1u32);

        amateur_logits_out.put(intrinsics::logits());
        amateur_position.put(&base);
        amateur_fill.put(&next);
        amateur_klen.put(&next);
        amateur_write_slot.put(gather(&ids, div(&base, PAGE_T)));
        amateur_write_offset.put(rem(&base, PAGE_T));
        amateur_mask.put(next_mask);
        amateur_pages.put(reshape(&ids, [pool_pages]));
        amateur_page_indptr.put(mul(iota(2), pool_pages));
    });

    let expert_token = bx(Channel::from(vec![first as i32]).named("expert_token"));
    let expert_position = bx(Channel::from(vec![n]).named("expert_position"));
    let expert_fill = bx(Channel::from(vec![n + 1]).named("expert_fill"));
    let expert_klen = bx(Channel::from(vec![n + 1]).named("expert_klen"));
    let expert_amateur = bx(Channel::new([vocab], dtype::f32).named("expert_amateur_logits"));
    let expert_token_out = bx(Channel::new([1], dtype::i32).named("expert_token_out"));

    let expert_decode: ForwardPass<'static> = ForwardPass::new();
    expert_decode.embed(expert_token, Tensor::constant(vec![0u32, 1]));
    expert_decode.positions(expert_position);
    expert_decode.attn_working_set(expert_ws, expert_klen);
    expert_decode.epilogue(move || {
        let base = expert_fill.take().tensor();
        let token = contrastive_pick(expert_amateur.take(), lambda, alpha, vocab);
        let next = add(&base, 1u32);

        expert_token.put(&token);
        expert_token_out.put(&token);
        expert_position.put(&base);
        expert_fill.put(&next);
        expert_klen.put(&next);
    });

    let amateur_pipeline = Pipeline::new();
    let expert_pipeline = Pipeline::new();
    let mut previous = first;
    while generated.len() < input.max_tokens {
        amateur_token.put(vec![previous as i32]);
        amateur_ids_input.put(amateur_ids.clone());
        amateur_decode
            .submit(&amateur_pipeline)
            .map_err(|error| format!("amateur decode: {error}"))?;
        let amateur_logits = amateur_logits_out
            .take()
            .get::<f32>()
            .map_err(|error| format!("read amateur logits: {error}"))?;

        expert_amateur.put(amateur_logits);
        expert_decode
            .submit(&expert_pipeline)
            .map_err(|error| format!("expert decode: {error}"))?;
        let token = read_expert_token(expert_token_out)?;
        if stop_tokens.contains(&token) {
            break;
        }
        generated.push(token);
        previous = token;
    }
    amateur_pipeline.close();
    expert_pipeline.close();

    wit_model::decode(&generated)
}

fn read_expert_token(channel: &Channel) -> Result<u32> {
    channel
        .take()
        .get::<i32>()
        .map_err(|error| format!("read expert token: {error}"))?
        .first()
        .copied()
        .map(|token| token as u32)
        .ok_or_else(|| "expert returned no token".into())
}
