//! Same-model contrastive decoding.
//!
//! Two forward passes use the bound model with independent KV states. The
//! expert sees the complete causal context; the amateur sees only a short
//! attention window. Tokens are selected by
//! `log p_expert - lambda * log p_amateur`, restricted to tokens whose expert
//! probability is at least `alpha` times the expert's maximum probability.

use inferlet::chat;
use inferlet::ptir::attention::prelude::*;
use serde::Deserialize;

const PAGE_T: u32 = 16;

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
    let expert = log_softmax(intrinsics::logits());
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
    let amateur_ws = WorkingSet::new();
    let amateur_slots = amateur_ws
        .reserve(pool_pages)
        .context("reserve amateur KV")?;
    let amateur_ids = amateur_slots.ids().to_vec();

    let prompt_i32 = prompt.iter().map(|&token| token as i32).collect::<Vec<_>>();
    let amateur_prompt = Channel::from(prompt_i32.clone()).named("amateur_prompt");
    let amateur_prefill_embed_indptr =
        Channel::from([0u32, n]).named("amateur_prefill_embed_indptr");
    let amateur_prefill_positions = Channel::from_iter(0..n).named("amateur_prefill_positions");
    let amateur_prefill_slots =
        Channel::from_iter((0..n).map(|position| amateur_ids[(position / PAGE_T) as usize]));
    let amateur_prefill_offsets = Channel::from_iter((0..n).map(|position| position % PAGE_T));
    let amateur_prefill_klen = Channel::from([n]);
    let amateur_prefill_pages = Channel::from(amateur_ids.clone());
    // The page CSR is the wire's source of truth for kv_len: the driver derives
    // `kv_len = (page_count-1)*PAGE_T + last_page_len`. A pool-wide constant page
    // count claims a kv length the pass does not have and silently corrupts
    // attention, so the count must track `kv_len` exactly (as the expert pass does).
    let amateur_prefill_indptr = Channel::from([0u32, n.div_ceil(PAGE_T)]);
    let amateur_prefill_mask = Channel::from_shaped(
        [n, pool_len],
        (0..n)
            .flat_map(|query| {
                (0..pool_len).map(move |key| key <= query && key.saturating_add(window) > query)
            })
            .collect::<Vec<_>>(),
    );
    let amateur_prefill_out = Channel::new([vocab], dtype::f32).named("amateur_prefill_logits");

    let amateur_prefill = ForwardPass::new();
    amateur_prefill.embed(&amateur_prompt, &amateur_prefill_embed_indptr)?;
    amateur_prefill.attention(
        &amateur_ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &amateur_prefill_klen,
            pages: &amateur_prefill_pages,
            page_indptr: &amateur_prefill_indptr,
            w_slot: &amateur_prefill_slots,
            w_off: &amateur_prefill_offsets,
            positions: &amateur_prefill_positions,
            mask: Some(&amateur_prefill_mask),
        },
    )?;
    amateur_prefill.epilogue(move || {
        amateur_prefill_out.put(intrinsics::logits());
    });

    // ONE PIPELINE for the whole inferlet: a KV working set is scoped to the
    // first pipeline that fires it, so the amateur's prefill and its decode
    // must share a FIFO (and likewise the expert's). The two streams are
    // sequential anyway — the expert prefill consumes the amateur's logits.
    let pipeline = Pipeline::new();
    amateur_prefill
        .submit(&pipeline)
        .context("amateur prefill")?;
    let first_amateur_logits = amateur_prefill_out.take_host::<Vec<f32>>().await?;

    // The expert keeps the complete context and consumes the amateur logits in
    // its epilogue to perform the contrastive token selection on-device.
    let expert_ws = WorkingSet::new();
    expert_ws.reserve(pool_pages).context("reserve expert KV")?;
    let expert_prompt = Channel::from(prompt_i32).named("expert_prompt");
    let expert_prefill_embed_indptr = Channel::from([0u32, n]).named("expert_prefill_embed_indptr");
    let expert_prefill_positions = Channel::from_iter(0..n).named("expert_prefill_positions");
    let expert_prefill_pages = Channel::from_iter(0..pool_pages).named("expert_prefill_pages");
    let expert_prefill_indptr =
        Channel::from([0u32, n.div_ceil(PAGE_T)]).named("expert_prefill_indptr");
    let expert_prefill_slots =
        Channel::from_iter((0..n).map(|p| p / PAGE_T)).named("expert_prefill_slots");
    let expert_prefill_offsets =
        Channel::from_iter((0..n).map(|p| p % PAGE_T)).named("expert_prefill_offsets");
    let expert_prefill_amateur = Channel::new([vocab], dtype::f32).named("expert_prefill_amateur");
    let first_out = Channel::new([1], dtype::i32).named("first_token");
    let lambda = input.lambda;
    let alpha = input.alpha;

    let expert_prefill = ForwardPass::new();
    expert_prefill.embed(&expert_prompt, &expert_prefill_embed_indptr)?;
    let expert_prefill_kv_len = Channel::from([n]).named("expert_prefill_kv_len");
    expert_prefill.attention(
        &expert_ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &expert_prefill_kv_len,
            pages: &expert_prefill_pages,
            page_indptr: &expert_prefill_indptr,
            w_slot: &expert_prefill_slots,
            w_off: &expert_prefill_offsets,
            positions: &expert_prefill_positions,
            mask: None,
        },
    )?;
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
    let amateur_position = Channel::from([n]).named("amateur_position");
    let amateur_fill = Channel::from([n + 1]).named("amateur_fill");
    let amateur_klen = Channel::from([n + 1]).named("amateur_klen");
    let amateur_write_slot = Channel::from([amateur_ids[(n / PAGE_T) as usize]]);
    let amateur_write_offset = Channel::from([n % PAGE_T]);
    let amateur_mask = Channel::from_shaped(
        [1, pool_len],
        (0..pool_len)
            .map(|key| key <= n && key.saturating_add(window) > n)
            .collect::<Vec<_>>(),
    );
    let amateur_pages = Channel::from(amateur_ids.clone());
    let amateur_page_indptr = Channel::from([0u32, (n + 1).div_ceil(PAGE_T)]);
    let amateur_ids_input = Channel::from(amateur_ids.clone()).named("amateur_pool_ids");
    let amateur_logits_out = Channel::new([vocab], dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("amateur_logits");

    let amateur_decode = ForwardPass::new();
    let amateur_embed_indptr = Channel::from([0u32, 1]).named("amateur_embed_indptr");
    amateur_decode.embed(&amateur_token, &amateur_embed_indptr)?;
    amateur_decode.attention(
        &amateur_ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: (n / kv_page_size())..,
            kv_len: &amateur_klen,
            pages: &amateur_pages,
            page_indptr: &amateur_page_indptr,
            w_slot: &amateur_write_slot,
            w_off: &amateur_write_offset,
            positions: &amateur_position,
            mask: Some(&amateur_mask),
        },
    )?;
    amateur_decode.epilogue(move || {
        let base = amateur_fill.take();
        let ids = amateur_ids_input.take();
        let columns = iota(pool_len);
        let base_columns = broadcast(reshape(&base, [1]), [pool_len]);
        let next_mask = reshape(
            and(
                le(&columns, &base_columns),
                gt(&columns + window, &base_columns),
            ),
            [1, pool_len],
        );
        let next = &base + 1u32;

        amateur_logits_out.put(intrinsics::logits());
        amateur_position.put(&base);
        amateur_fill.put(&next);
        amateur_klen.put(&next);
        amateur_write_slot.put(gather(&ids, &base / PAGE_T));
        amateur_write_offset.put(&base % PAGE_T);
        amateur_mask.put(next_mask);
        amateur_pages.put(reshape(&ids, [pool_pages]));
        let amateur_page_count = next.div_ceil(PAGE_T);
        amateur_page_indptr.put(indptr(1, &amateur_page_count));
        amateur_ids_input.put(&ids);
    });

    let expert_token = Channel::from([first as i32]).named("expert_token");
    let expert_embed_indptr = Channel::from([0u32, 1]).named("expert_embed_indptr");
    let expert_position = Channel::from([n]).named("expert_position");
    let expert_pages = Channel::from_iter(0..pool_pages).named("expert_pages");
    let expert_page_indptr =
        Channel::from([0u32, (n + 1).div_ceil(PAGE_T)]).named("expert_page_indptr");
    let expert_write_slot = Channel::from([n / PAGE_T]).named("expert_write_slot");
    let expert_write_offset = Channel::from([n % PAGE_T]).named("expert_write_offset");
    let expert_amateur = Channel::writer([vocab], dtype::f32).named("expert_amateur_logits");
    let expert_token_out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("expert_token_out");

    let expert_decode = ForwardPass::new();
    expert_decode.embed(&expert_token, &expert_embed_indptr)?;
    let expert_kv_len = Channel::from([n + 1]).named("expert_kv_len");
    expert_decode.attention(
        &expert_ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: (n / kv_page_size())..,
            kv_len: &expert_kv_len,
            pages: &expert_pages,
            page_indptr: &expert_page_indptr,
            w_slot: &expert_write_slot,
            w_off: &expert_write_offset,
            positions: &expert_position,
            mask: None,
        },
    )?;
    expert_decode.epilogue(move || {
        let length = expert_kv_len.take();
        let token = contrastive_pick(&expert_amateur.take(), lambda, alpha, vocab);
        let next_length = &length + 1u32;
        let page_count = next_length.div_ceil(PAGE_T);

        expert_token.put(&token);
        expert_kv_len.put(&next_length);
        expert_position.put(&length);
        expert_write_slot.put(&length / PAGE_T);
        expert_write_offset.put(&length % PAGE_T);
        expert_page_indptr.put(indptr(1, &page_count));
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
