//! Applies a greenlist watermark while generating text.
//!
//! Each previously generated token deterministically partitions the vocabulary
//! into a green and red list. Green-token logits receive a configurable bias
//! before Gumbel-max sampling.

use inferlet::chat;
use inferlet::ptir::attention::prelude::*;
use serde::Deserialize;
use std::hash::{DefaultHasher, Hash, Hasher};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_gamma")]
    gamma: f32,
    #[serde(default = "default_delta")]
    delta: f32,
}

fn default_prompt() -> String {
    "Explain language-model decoding in simple terms.".into()
}

fn default_max_tokens() -> usize {
    256
}

fn default_gamma() -> f32 {
    0.5
}

fn default_delta() -> f32 {
    2.0
}

fn green_mask(vocab: u32, previous_token: u32, gamma: f32) -> Vec<bool> {
    let threshold = (gamma * u64::MAX as f32) as u64;
    (0..vocab)
        .map(|token| {
            let mut hasher = DefaultHasher::new();
            previous_token.hash(&mut hasher);
            token.hash(&mut hasher);
            hasher.finish() <= threshold
        })
        .collect()
}

fn watermarked_sample(
    logits: &Tensor,
    green: &Tensor,
    rng_state: &Tensor,
    delta: f32,
    vocab: u32,
) -> Tensor {
    let bias = select(green, broadcast(delta, [vocab]), broadcast(0.0f32, [vocab]));
    gumbel_max(logits + bias, rng_state)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if !(0.0..=1.0).contains(&input.gamma) {
        return Err("gamma must be between 0 and 1".into());
    }
    if !input.delta.is_finite() {
        return Err("delta must be finite".into());
    }
    if input.max_tokens == 0 {
        return Ok(String::new());
    }

    let vocab = model::output_vocab_size();
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stop_tokens = chat::stop_tokens();
    let delta = input.delta;
    let max_pages = (n + input.max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;

    let prompt_tokens = Channel::from_iter(prompt.iter().map(|&token| token as i32));
    let prefill_indptr = Channel::from([0u32, n]).named("prefill_indptr");
    let prefill_positions = Channel::from_iter(0..n).named("prefill_positions");
    let prefill_pages = Channel::from_iter(0..max_pages).named("prefill_pages");
    let prefill_page_indptr =
        Channel::from([0u32, n.div_ceil(page_size)]).named("prefill_page_indptr");
    let prefill_w_slot = Channel::from_iter((0..n).map(|p| p / page_size)).named("prefill_w_slot");
    let prefill_w_off = Channel::from_iter((0..n).map(|p| p % page_size)).named("prefill_w_off");
    let prefill_green = Channel::new([vocab], dtype::bool).named("prefill_green");
    let prefill_rng = Channel::from([0x51ed_u32, 0]).named("prefill_rng");
    let first_out = Channel::new([1], dtype::i32).named("first_token");

    let prefill = ForwardPass::new();
    prefill.embed(&prompt_tokens, &prefill_indptr)?;
    let prefill_kv_len = Channel::from([n]).named("prefill_kv_len");
    prefill.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &prefill_kv_len,
            pages: &prefill_pages,
            page_indptr: &prefill_page_indptr,
            w_slot: &prefill_w_slot,
            w_off: &prefill_w_off,
            positions: &prefill_positions,
            mask: None,
        },
    )?;
    prefill.epilogue(move || {
        let green = prefill_green.take();
        let rng = prefill_rng.take();
        let token = watermarked_sample(&intrinsics::logits(), &green, &rng, delta, vocab);
        first_out.put(&token);
        prefill_rng.put(&rng + iota(2));
    });

    prefill_green.put(green_mask(vocab, *prompt.last().unwrap_or(&0), input.gamma));
    // ONE pipeline for the whole stream (R4-4): prefill and decode are one
    // sequential stream. The host round-trip on `first` stays — it seeds the
    // decode channels and the first green mask below.
    let pipeline = Pipeline::new();
    prefill.submit(&pipeline).context("watermark prefill")?;
    // max_tokens == 1: the prefill spends the whole budget, so it was the
    // stream's last submit — finish() right after it (F7).
    let first = first_out.take_host::<i32>().await? as u32;

    let mut generated = Vec::with_capacity(input.max_tokens);
    if !stop_tokens.contains(&first) {
        generated.push(first);
    }

    if generated.len() < input.max_tokens && !stop_tokens.contains(&first) {
        let token_in = Channel::from([first as i32]).named("token_in");
        let green = Channel::new([vocab], dtype::bool).named("green");
        let rng = Channel::from([0x9e37_u32, 0]).named("rng");
        let embed_indptr = Channel::from([0u32, 1]).named("embed_indptr");
        let positions = Channel::from([n]).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([n / page_size]).named("w_slot");
        let w_off = Channel::from([n % page_size]).named("w_off");
        let token_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("token_out");

        let decode = ForwardPass::new();
        decode.embed(&token_in, &embed_indptr)?;
        let kv_len = Channel::from([n + 1]).named("kv_len");
        decode.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: (n / page_size)..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        )?;
        decode.epilogue(move || {
            let length = kv_len.take();
            let green_value = green.take();
            let rng_value = rng.take();
            let token = watermarked_sample(
                &intrinsics::logits(),
                &green_value,
                &rng_value,
                delta,
                vocab,
            );
            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);

            token_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            token_out.put(&token);
            rng.put(&rng_value + iota(2));
        });

        // The greenlist is keyed on the PREVIOUS OUTPUT token, so the host
        // cannot supply fire k+1's mask until fire k has settled: this loop is
        // inherently depth-1. Running ahead would submit fires whose Writer
        // channel has no published value and fail the fire outright.
        let mut previous = first;
        let budget = input.max_tokens.saturating_sub(generated.len());
        let mut submitted = 0usize;

        while submitted < budget {
            green.put(green_mask(vocab, previous, input.gamma));
            decode.submit(&pipeline).context("watermark decode")?;
            submitted += 1;
            let token = token_out.take_host::<i32>().await? as u32;
            previous = token;
            if stop_tokens.contains(&token) {
                break;
            }
            generated.push(token);
        }
    }
    pipeline.close();

    model::decode(&generated)
}
