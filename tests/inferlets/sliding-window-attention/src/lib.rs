//! Generates with a sliding-window attention mask.
//!
//! The prompt is prefilled once with normal causal attention. During decoding,
//! each query can attend only to the most recent `window_size` positions. The
//! example masks old KV cells but does not evict their backing pages.

use inferlet::chat;
use inferlet::ptir::attention::prelude::*;
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

    let page_t = model::kv_page_size();
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
        .context("reserve sliding-window KV")?;
    let pool_ids = slots.ids().to_vec();

    let prompt_tokens = Channel::from_iter(prompt.iter().map(|&token| token as i32));
    let prefill_slots =
        Channel::from_iter((0..n).map(|position| pool_ids[(position / page_t) as usize]));
    let prefill_offsets = Channel::from_iter((0..n).map(|position| position % page_t));
    let prefill_klen = Channel::from([n]);
    let prefill_pages = Channel::from(pool_ids.clone());
    // The page CSR is the wire's source of truth for kv_len: the driver derives
    // `kv_len = (page_count-1)*page_t + last_page_len`. A pool-wide constant page
    // count claims a kv length the pass does not have and silently corrupts
    // attention, so the count must track `kv_len` exactly.
    let prefill_indptr = Channel::from([0u32, n.div_ceil(page_t)]);
    let causal = Channel::from_shaped(
        [n, pool_len],
        (0..n)
            .flat_map(|query| (0..pool_len).map(move |key| key <= query))
            .collect::<Vec<_>>(),
    );
    let prefill_positions = Channel::from_iter(0..n);
    let prefill_embed_indptr = Channel::from([0u32, n]);
    let first_out = Channel::new([1], dtype::i32).named("first_token");

    let prefill = ForwardPass::new();
    prefill.embed(&prompt_tokens, &prefill_embed_indptr)?;
    prefill.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &prefill_klen,
            pages: &prefill_pages,
            page_indptr: &prefill_indptr,
            w_slot: &prefill_slots,
            w_off: &prefill_offsets,
            positions: &prefill_positions,
            mask: Some(&causal),
        },
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
        .context("sliding-window prefill")?;
    if input.max_tokens == 1 {
        pipeline.close();
    }
    let first = first_out.take_host::<i32>().await? as u32;

    let mut generated = Vec::with_capacity(input.max_tokens);
    if !stop_tokens.contains(&first) {
        generated.push(first);
    }
    if generated.len() >= input.max_tokens || stop_tokens.contains(&first) {
        // The only fire has settled (its take succeeded), so dropping the
        // pipeline here (drop == close) cancels nothing.
        return model::decode(&generated);
    }

    let token_in = Channel::from([first as i32]).named("token_in");
    let position = Channel::from([n]).named("position");
    let fill = Channel::from([n + 1]).named("fill");
    let klen = Channel::from([n + 1]).named("klen");
    let write_slot = Channel::from([pool_ids[(n / page_t) as usize]]);
    let write_offset = Channel::from([n % page_t]);
    let mask = Channel::from_shaped(
        [1, pool_len],
        (0..pool_len)
            .map(|key| key <= n && key.saturating_add(window) > n)
            .collect::<Vec<_>>(),
    );
    let pages = Channel::from(pool_ids.clone());
    let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_t)]);
    let decode_embed_indptr = Channel::from([0u32, 1]);
    let pool_ids_input = Channel::from(pool_ids.clone()).named("pool_ids");
    let token_out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("token_out");

    let decode = ForwardPass::new();
    decode.embed(&token_in, &decode_embed_indptr)?;
    decode.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &klen,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &write_slot,
            w_off: &write_offset,
            positions: &position,
            mask: Some(&mask),
        },
    )?;
    decode.epilogue(move || {
        let base = fill.take();
        let ids = pool_ids_input.take();
        let token = reshape(reduce_argmax(intrinsics::logits()), [1]);
        let next_mask = reshape(sliding_window_mask(&base, pool_len, window), [1, pool_len]);
        let logical_slot = &base / page_t;
        let next = &base + 1u32;

        // Device-resolved geometry is loop-carried: the host never drains
        // these rings, so every fire's values are re-put here.
        token_in.put(&token);
        token_out.put(&token);
        position.put(&base);
        fill.put(&next);
        klen.put(&next);
        write_slot.put(gather(&ids, &logical_slot));
        write_offset.put(&base % page_t);
        mask.put(&next_mask);
        pages.put(reshape(&ids, [pool_pages]));
        let page_count = next.div_ceil(page_t);
        page_indptr.put(indptr(1, &page_count));
        pool_ids_input.put(&ids);
    });

    let budget = input.max_tokens.saturating_sub(generated.len());
    run_ahead(&pipeline, &decode, budget as usize, async || {
        let token = token_out.take_host::<i32>().await? as u32;
        if stop_tokens.contains(&token) {
            return Ok(ControlFlow::Break(()));
        }
        generated.push(token);
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    pipeline.close();
    model::decode(&generated)
}
