//! tart: the 0.3 naive-masked — a CUSTOM (dense-packed) attention mask
//! whose numerics are exactly causal. The prefill mask is a host bool
//! tensor (a literal the structured recognizer cannot match), and the
//! decode evolution is `and(causal, causal)` — same trick, device-side.
//! This is the mask axis's parity probe AND the spatial-split trigger:
//! co-fired with plain lanes it must produce a MASK region in the
//! scheduler's region table and the driver's planned mask split.

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
    #[serde(default)]
    max_layers: Option<u32>,
    /// BISECT: 0 = full (mask everywhere), 1 = no decode mask,
    /// 2 = no masks at all (prefill causal channel still bound? no — none).
    #[serde(default)]
    bisect: u32,
}

fn default_prompt() -> String {
    "Tell me a story about a clockmaker.".into()
}

fn default_max_tokens() -> usize {
    32
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if input.max_tokens == 0 {
        return Ok(String::new());
    }

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
    let slots = ws.reserve(pool_pages).context("reserve tart-masked KV")?;
    let pool_ids = slots.ids().to_vec();

    let prompt_tokens = Channel::from_iter(prompt.iter().map(|&token| token as i32));
    let prefill_embed_indptr = Channel::from([0u32, n]).named("prefill_embed_indptr");
    let prefill_positions = Channel::from_iter(0..n).named("prefill_positions");
    let prefill_slots =
        Channel::from_iter((0..n).map(|position| pool_ids[(position / PAGE_T) as usize]));
    let prefill_offsets = Channel::from_iter((0..n).map(|position| position % PAGE_T));
    let prefill_klen = Channel::from([n]);
    let prefill_pages = Channel::from(pool_ids.clone());
    let prefill_indptr = Channel::from([0u32, n.div_ceil(PAGE_T)]);
    // The DENSE causal literal: byte-for-byte causal numerics, packed as
    // a custom mask because a host literal has no structured form.
    let causal = Channel::from_shaped(
        [n, pool_len],
        (0..n)
            .flat_map(|query| (0..pool_len).map(move |key| key <= query))
            .collect::<Vec<_>>(),
    );
    let first_out = Channel::new([1], dtype::i32).named("first_token");

    let prefill = ForwardPass::new();
    if let Some(k) = input.max_layers {
        prefill.set_max_layers(k)?;
    }
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
            mask: if input.bisect >= 2 { None } else { Some(&causal) },
        },
    )?;
    prefill.epilogue(move || {
        first_out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
    });

    let pipeline = Pipeline::new();
    prefill.submit(&pipeline).context("tart-masked prefill")?;
    let first = first_out.take_host::<i32>().await? as u32;

    let mut generated = Vec::with_capacity(input.max_tokens);
    if !stop_tokens.contains(&first) {
        generated.push(first);
    }
    if generated.len() >= input.max_tokens || stop_tokens.contains(&first) {
        pipeline.close();
        return model::decode(&generated);
    }

    // HOST-DRIVEN decode (the 0.2 naive-masked posture): every geometry
    // channel is put from the host each step, because a HOST wire mask
    // (dense BRLE — the tart spatial path) cannot mix with
    // device-evolved geometry. Sequential, one fire per submit.
    let token_in = Channel::from([first as i32]).named("token_in");
    let decode_indptr = Channel::from([0u32, 1]).named("decode_indptr");
    let position = Channel::from([n]).named("position");
    let klen = Channel::from([n + 1]).named("klen");
    let write_slot = Channel::from([pool_ids[(n / PAGE_T) as usize]]);
    let write_offset = Channel::from([n % PAGE_T]);
    let mask = Channel::from_shaped(
        [1, pool_len],
        (0..pool_len).map(|key| key <= n).collect::<Vec<_>>(),
    );
    let pages = Channel::from(pool_ids.clone());
    let page_indptr = Channel::from([0u32, (n + 1).div_ceil(PAGE_T)]);
    let token_out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("token_out");

    let decode = ForwardPass::new();
    if let Some(k) = input.max_layers {
        decode.set_max_layers(k)?;
    }
    decode.embed(&token_in, &decode_indptr)?;
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
            mask: if input.bisect >= 1 { None } else { Some(&mask) },
        },
    )?;
    decode.epilogue(move || {
        token_out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
    });

    let budget = input.max_tokens.saturating_sub(generated.len());
    let mut filled = n + 1; // tokens in KV after the prefill+first write
    for _ in 0..budget {
        decode.submit(&pipeline).context("tart-masked decode")?;
        let token = token_out.take_host::<i32>().await? as u32;
        if stop_tokens.contains(&token) {
            break;
        }
        generated.push(token);
        if generated.len() >= input.max_tokens {
            break;
        }
        // Host-advance the geometry for the next fire.
        let pos = filled;
        token_in.put([token as i32]);
        position.put([pos]);
        klen.put([pos + 1]);
        write_slot.put([pool_ids[(pos / PAGE_T) as usize]]);
        write_offset.put([pos % PAGE_T]);
        if input.bisect == 0 {
            mask.put((0..pool_len).map(|key| key <= pos).collect::<Vec<bool>>());
        }
        page_indptr.put([0u32, (pos + 1).div_ceil(PAGE_T)]);
        filled += 1;
    }
    pipeline.close();
    model::decode(&generated)
}
