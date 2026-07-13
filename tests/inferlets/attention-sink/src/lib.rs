//! Generates with attention sinks and a sliding recent-token window.
//!
//! The prompt is prefilled once with normal causal attention. During decoding,
//! each query attends to the first `sink_size` positions and the most recent
//! `window_size` positions. Masked KV pages remain allocated in this example.

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
    #[serde(default = "default_sink_size")]
    sink_size: u32,
    #[serde(default = "default_window_size")]
    window_size: u32,
}

fn default_prompt() -> String {
    "Tell me a long story about a cat.".into()
}

fn default_max_tokens() -> usize {
    512
}

fn default_sink_size() -> u32 {
    4
}

fn default_window_size() -> u32 {
    64
}

fn host_sink_window_mask(pool_len: u32, query: u32, sink: u32, window: u32) -> Vec<bool> {
    (0..pool_len)
        .map(|key| key <= query && (key < sink || key.saturating_add(window) > query))
        .collect()
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if input.max_tokens == 0 {
        return Ok(String::new());
    }

    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);
    let sink = input.sink_size;
    let window = input.window_size.max(1);

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stop_tokens = chat::stop_tokens();
    let pool_pages = (n + input.max_tokens as u32 + 2).div_ceil(PAGE_T);
    let pool_len = pool_pages * PAGE_T;

    let ws = WorkingSet::new();
    let slots = ws
        .reserve(pool_pages)
        .map_err(|e| format!("reserve attention-sink KV: {e}"))?;
    let pool_ids = slots.ids().to_vec();

    let prompt_tokens = Channel::from(prompt.iter().map(|&token| token as i32).collect::<Vec<_>>());
    let prefill_slots = Channel::from(
        (0..n)
            .map(|position| pool_ids[(position / PAGE_T) as usize])
            .collect::<Vec<_>>(),
    );
    let prefill_offsets =
        Channel::from((0..n).map(|position| position % PAGE_T).collect::<Vec<_>>());
    let prefill_klen = Channel::from(vec![n]);
    let prefill_pages = Channel::from(pool_ids.clone());
    let prefill_indptr = Channel::from_shaped([2], vec![0u32, pool_pages]);
    let causal = Channel::from_shaped(
        [n, pool_len],
        (0..n)
            .flat_map(|query| (0..pool_len).map(move |key| key <= query))
            .collect::<Vec<_>>(),
    );
    let first_out = Channel::new([1], dtype::i32).named("first_token");

    let prefill = ForwardPass::new();
    prefill.embed(&prompt_tokens, Tensor::constant(vec![0u32, n]));
    prefill.attn_working_set(&ws, &prefill_klen);
    prefill.port_channel(Port::Pages, &prefill_pages);
    prefill.port_channel(Port::PageIndptr, &prefill_indptr);
    prefill.port_channel(Port::WSlot, &prefill_slots);
    prefill.port_channel(Port::WOff, &prefill_offsets);
    prefill.attn_mask(&causal);
    prefill.epilogue(|| {
        first_out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
    });

    let prefill_pipeline = Pipeline::new();
    prefill
        .submit(&prefill_pipeline)
        .map_err(|e| format!("attention-sink prefill: {e}"))?;
    let first = first_out
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("read first token: {e}"))?[0] as u32;
    prefill_pipeline.close();

    let mut generated = Vec::with_capacity(input.max_tokens);
    if !stop_tokens.contains(&first) {
        generated.push(first);
    }
    if generated.len() >= input.max_tokens || stop_tokens.contains(&first) {
        return wit_model::decode(&generated);
    }

    let token_in = Channel::from(vec![first as i32]).named("token_in");
    let position = Channel::from(vec![n]).named("position");
    let fill = Channel::from(vec![n + 1]).named("fill");
    let klen = Channel::from(vec![n + 1]).named("klen");
    let write_slot = Channel::from(vec![pool_ids[(n / PAGE_T) as usize]]);
    let write_offset = Channel::from(vec![n % PAGE_T]);
    let mask = Channel::from_shaped(
        [1, pool_len],
        host_sink_window_mask(pool_len, n, sink, window),
    );
    let pages = Channel::from(pool_ids.clone());
    let page_indptr = Channel::from_shaped([2], vec![0u32, pool_pages]);
    let pool_ids_input = Channel::from(pool_ids.clone()).named("pool_ids");
    let token_out = Channel::new([1], dtype::i32)
        .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
        .named("token_out");

    let decode = ForwardPass::new();
    decode.embed(&token_in, Tensor::constant(vec![0u32, 1]));
    decode.positions(&position);
    decode.attn_working_set(&ws, &klen);
    decode.port_channel(Port::Pages, &pages);
    decode.port_channel(Port::PageIndptr, &page_indptr);
    decode.port_channel(Port::WSlot, &write_slot);
    decode.port_channel(Port::WOff, &write_offset);
    decode.attn_mask(&mask);
    decode.epilogue(|| {
        let base = fill.take().tensor();
        let ids = pool_ids_input.take().tensor();
        let token = reshape(reduce_argmax(intrinsics::logits()), [1]);
        let next_mask = reshape(
            sink_window_mask(&base, pool_len, sink, window),
            [1, pool_len],
        );
        let logical_slot = div(&base, PAGE_T);
        let next = add(&base, 1u32);

        token_in.put(&token);
        token_out.put(&token);
        position.put(&base);
        fill.put(&next);
        klen.put(&next);
        write_slot.put(gather(&ids, &logical_slot));
        write_offset.put(rem(&base, PAGE_T));
        mask.put(&next_mask);
        pages.put(reshape(&ids, [pool_pages]));
        page_indptr.put(mul(iota(2), pool_pages));
        pool_ids_input.put(&ids);
    });

    let pipeline = Pipeline::new();
    let budget = input.max_tokens.saturating_sub(generated.len());
    let mut submitted = 0usize;
    let mut in_flight = 0usize;
    while in_flight < DEFAULT_RUNAHEAD_DEPTH && submitted < budget {
        decode
            .submit(&pipeline)
            .map_err(|e| format!("attention-sink decode: {e}"))?;
        submitted += 1;
        in_flight += 1;
    }
    while in_flight > 0 {
        let token = token_out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("read generated token: {e}"))?[0] as u32;
        in_flight -= 1;
        if stop_tokens.contains(&token) {
            break;
        }
        generated.push(token);
        if submitted < budget {
            decode
                .submit(&pipeline)
                .map_err(|e| format!("attention-sink decode: {e}"))?;
            submitted += 1;
            in_flight += 1;
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
