//! Generates with a sliding-window attention mask.
//!
//! The prompt is prefilled once with normal causal attention. During decoding,
//! each query can attend only to the most recent `window_size` positions. The
//! example masks old KV cells but does not evict their backing pages.

use inferlet::ptir::prelude::*;
use inferlet::{Result, chat, model as wit_model};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_window_size")]
    window_size: u32,
}

fn default_prompt() -> String {
    "Tell me a long story about a cat.".into()
}

fn default_max_tokens() -> usize {
    512
}

fn default_window_size() -> u32 {
    64
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if input.max_tokens == 0 {
        return Ok(String::new());
    }

    let page_t = wit_model::kv_page_size();
    let window = input.window_size.max(1);

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stop_tokens = chat::stop_tokens();
    let pool_pages = (n + input.max_tokens as u32 + 2).div_ceil(page_t);
    let pool_len = pool_pages * page_t;

    let ws = WorkingSet::new();
    let slots = ws
        .reserve(pool_pages)
        .map_err(|e| format!("reserve sliding-window KV: {e}"))?;
    let pool_ids = slots.ids().to_vec();

    let prompt_tokens = Channel::from(prompt.iter().map(|&token| token as i32).collect::<Vec<_>>());
    let prefill_slots = Channel::from(
        (0..n)
            .map(|position| pool_ids[(position / page_t) as usize])
            .collect::<Vec<_>>(),
    );
    let prefill_offsets =
        Channel::from((0..n).map(|position| position % page_t).collect::<Vec<_>>());
    let prefill_klen = Channel::from(vec![n]);
    let prefill_pages = Channel::from(pool_ids.clone());
    let prefill_indptr = Channel::from_shaped([2], vec![0u32, pool_pages]);
    let causal = Channel::from_shaped(
        [n, pool_len],
        (0..n)
            .flat_map(|query| (0..pool_len).map(move |key| key <= query))
            .collect::<Vec<_>>(),
    );
    let prefill_positions = Channel::from((0..n).collect::<Vec<_>>());
    let prefill_embed_indptr = Channel::from(vec![0u32, n]);
    let first_out = Channel::new([1], dtype::i32).named("first_token");

    let prefill = ForwardPass::new();
    prefill.embed(&prompt_tokens, &prefill_embed_indptr)?;
    prefill.attention(
        &ws,
        ..,
        ..,
        &prefill_klen,
        &prefill_pages,
        &prefill_indptr,
        &prefill_slots,
        &prefill_offsets,
        &prefill_positions,
        Some(&causal),
    )?;
    prefill.epilogue(move || {
        first_out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
    });

    // ONE pipeline for the whole stream (R4-4): prefill and decode are one
    // sequential stream. The host round-trip on `first` stays — its take
    // seeds the decode channels below.
    let pipeline = Pipeline::new();
    prefill
        .submit(&pipeline)
        .map_err(|e| format!("sliding-window prefill: {e}"))?;
    if input.max_tokens == 1 {
        pipeline.close();
    }
    let first = first_out
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("read first token: {e}"))?[0] as u32;

    let mut generated = Vec::with_capacity(input.max_tokens);
    if !stop_tokens.contains(&first) {
        generated.push(first);
    }
    if generated.len() >= input.max_tokens || stop_tokens.contains(&first) {
        // The only fire has settled (its take succeeded), so dropping the
        // pipeline here (drop == close) cancels nothing.
        return wit_model::decode(&generated);
    }

    let token_in = Channel::from(vec![first as i32]).named("token_in");
    let position = Channel::from(vec![n]).named("position");
    let fill = Channel::from(vec![n + 1]).named("fill");
    let klen = Channel::from(vec![n + 1]).named("klen");
    let write_slot = Channel::from(vec![pool_ids[(n / page_t) as usize]]);
    let write_offset = Channel::from(vec![n % page_t]);
    let mask = Channel::from_shaped(
        [1, pool_len],
        (0..pool_len)
            .map(|key| key <= n && key.saturating_add(window) > n)
            .collect::<Vec<_>>(),
    );
    let pages = Channel::from(pool_ids.clone());
    let page_indptr = Channel::from_shaped([2], vec![0u32, pool_pages]);
    let decode_embed_indptr = Channel::from(vec![0u32, 1]);
    let pool_ids_input = Channel::from(pool_ids.clone()).named("pool_ids");
    let token_out = Channel::new([1], dtype::i32)
        .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
        .named("token_out");

    let decode = ForwardPass::new();
    decode.embed(&token_in, &decode_embed_indptr)?;
    decode.attention(
        &ws,
        ..,
        ..,
        &klen,
        &pages,
        &page_indptr,
        &write_slot,
        &write_offset,
        &position,
        Some(&mask),
    )?;
    decode.epilogue(move || {
        let base = fill.take().tensor();
        let ids = pool_ids_input.take().tensor();
        let token = reshape(reduce_argmax(intrinsics::logits()), [1]);
        let next_mask = reshape(sliding_window_mask(&base, pool_len, window), [1, pool_len]);
        let logical_slot = div(&base, page_t);
        let next = add(&base, 1u32);

        token_in.put(&token);
        token_out.put(&token);
        position.put(&base);
        fill.put(&next);
        klen.take();
        klen.put(&next);
        write_slot.put(gather(&ids, &logical_slot));
        write_offset.put(rem(&base, page_t));
        mask.take();
        mask.put(&next_mask);
        pages.take();
        pages.put(reshape(&ids, [pool_pages]));
        page_indptr.take();
        page_indptr.put(mul(iota(2), pool_pages));
        pool_ids_input.put(&ids);
    });

    let budget = input.max_tokens.saturating_sub(generated.len());
    let mut submitted = 0usize;
    let mut in_flight = 0usize;
    while in_flight < DEFAULT_RUNAHEAD_DEPTH && submitted < budget {
        decode
            .submit(&pipeline)
            .map_err(|e| format!("sliding-window decode: {e}"))?;
        submitted += 1;
        in_flight += 1;
    }
    if submitted == budget {
        pipeline.close();
    }
    while in_flight > 0 {
        let token = token_out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("read generated token: {e}"))?[0] as u32;
        in_flight -= 1;
        if stop_tokens.contains(&token) {
            pipeline.close();
            break;
        }
        generated.push(token);
        if submitted < budget {
            decode
                .submit(&pipeline)
                .map_err(|e| format!("sliding-window decode: {e}"))?;
            submitted += 1;
            in_flight += 1;
            if submitted == budget {
                pipeline.close();
            }
        }
    }
    while in_flight > 0 {
        token_out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("drain run-ahead token: {e}"))?;
        in_flight -= 1;
    }
    pipeline.close();
    wit_model::decode(&generated)
}
