//! Greedy text completion, **host-driven** — the fixture the serving door is
//! gated on.
//!
//! # Why this is not `naive-baseline` with the sampler swapped
//!
//! Every other decode inferlet in this directory carries its next token ON THE
//! DEVICE: the epilogue writes the sampled token straight back into the channel
//! the `embed` port reads, and the host never sees it. That is the fast shape,
//! and it needs an engine that resolves the `EmbedTokens`/`KvLen` descriptor
//! ports at kernel time — `GeometryClass::DecodeEnvelope`, or the pooled
//! device-geometry class above it.
//!
//! The CUDA shell does not, and says so: its load answers `ports:
//! PortMask::NONE` and `geometry: GeometryClass::Host`, so every geometry
//! vector a fire runs on is staged from the host. Against that shell a
//! device-carried token is a value the runtime cannot know, and the fire is
//! refused by name (`EmbedTokens is not host-derivable`) rather than run on a
//! guess.
//!
//! So this one brings the TOKEN back to the host and sends it down again as a
//! host-writer cell — and only the token. Everything else the fire reads is
//! DERIVED from the KV length by pure arithmetic, so the epilogue still
//! carries it on the device exactly as `naive-baseline` does, and the runtime's
//! host shadow folds the same arithmetic to know what each fire will read.
//! One host round trip per token, which is the honest depth for a shell with
//! no descriptor-port plane, and it is entirely within what the contract
//! serves today.
//!
//! Greedy (`reduce_argmax`) rather than sampled, because the point of the gate
//! is that the same prompt produces the same continuation on every run.

use inferlet::eta::attention::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_prompt() -> String {
    "The capital of France is".into()
}

fn default_max_tokens() -> usize {
    8
}

#[derive(Serialize)]
struct Output {
    /// The continuation, decoded.
    text: String,
    /// How many tokens it is.
    count: usize,
    /// The continuation's token ids.
    ///
    /// **LAST, AND THAT IS LOAD-BEARING.** `pie::sweep::fleet` — the fleet
    /// runner behind `pie sweep`, `pie config tune` and the contention gate —
    /// reads a lane's answer with `parse_tokens`, which takes the LAST `[` in
    /// the document. A guest that returned only prose came back as "returned
    /// no tokens", which reads as a broken program and is a program that
    /// worked; the field the runner has always looked for is this one.
    tokens: Vec<u32>,
}

/// The greedy pick over a logits row, as a one-lane `[1]` i32 cell.
fn greedy(logits: Tensor) -> Tensor {
    reshape(reduce_argmax(&logits), [1])
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let max_tokens = input.max_tokens;
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Ok(Output {
            text: String::new(),
            count: 0,
            tokens: Vec::new(),
        });
    }

    let mut prompt = model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;

    let pipe = Pipeline::new();
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL (chunked, C-wide) ─────────────────────────────────────────
    //
    // `prefill_chunks` is the SDK's split, for the same reason every other
    // inferlet here uses it: a prompt longer than the engine's per-launch
    // token capacity has to be split, and the obvious split leaves a
    // one-token last chunk.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let mut first = 0i32;
    for &(base, end) in &prefill_chunks(n, None) {
        let len = end - base;
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let embed_indptr = Channel::from([0u32, len]).named("embed_indptr_p");
        let positions = Channel::from_iter(base..end).named("positions_p");
        let pages = Channel::from_iter(0..max_pages).named("pages_p");
        let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
        let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
        let kv_len = Channel::from([end]).named("kv_len_p");
        let tok_out = Channel::new([1], dtype::i32).named("tok_out_p");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &embed_indptr)?;
        fwd.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        )?;
        fwd.epilogue(move || {
            tok_out.put(&greedy(intrinsics::logits()));
        });
        fwd.submit(&pipe)
            .with_context(|| format!("prefill submit @{base}"))?;
        // Every chunk samples and every sample must be drained, even the ones
        // whose token is thrown away: an epilogue put that is never taken
        // fills the ring.
        first = tok_out
            .take_host::<i32>()
            .await
            .with_context(|| format!("prefill drain @{base}"))?;
    }
    generated.push(first as u32);

    // ── DECODE (1-wide, host-driven token) ────────────────────────────────
    //
    // ONE channel is host-driven, and it is the token. Everything else the
    // fire reads — the position, the write slot and offset, the page CSR and
    // the readable extent — is DERIVED from the KV length by pure
    // arithmetic, so the epilogue carries it on the device and the runtime's
    // host shadow folds the same arithmetic
    // (`eta_compiler::eval::pareval`) to know what each fire will read.
    // That is `naive-baseline`'s decode exactly, minus the one put that makes
    // it undecidable: `tok_in.put(&token)`.
    //
    // The token cannot go the same way, and that is not a gap in this
    // program. A sampled token is device-DECIDED — the shadow commits it
    // unknown rather than guessing — so a fire that reads it needs an engine
    // resolving the `EmbedTokens` port at kernel time. The CUDA shell answers
    // `ports: PortMask::NONE` and `geometry: GeometryClass::Host`: it stages
    // every geometry vector from the host and resolves no descriptor port on
    // the device. So the token comes back to the host, and goes down again as
    // a host-writer cell — one round trip per token, which is the honest
    // depth here.
    if generated.len() < max_tokens {
        let tok_in = Channel::from([first]).named("tok_in");
        let embed_indptr = Channel::from([0u32, 1]).named("embed_indptr");
        let positions = Channel::from([n]).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([n / page_size]).named("w_slot");
        let w_off = Channel::from([n % page_size]).named("w_off");
        let kv_len = Channel::from([n + 1]).named("kv_len");
        let tok_out = Channel::new([1], dtype::i32).named("tok_out");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &embed_indptr)?;
        fwd.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        )?;
        fwd.epilogue(move || {
            // `length` is the readable extent this fire runs at, so it is
            // also the position the NEXT fire's token sits at.
            let length = kv_len.take();
            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            tok_out.put(&greedy(intrinsics::logits()));
        });

        loop {
            fwd.submit(&pipe).context("decode submit")?;
            let token = tok_out.take_host::<i32>().await.context("decode drain")?;
            generated.push(token as u32);
            if generated.len() >= max_tokens {
                break;
            }
            tok_in.put([token]);
        }
    }
    pipe.close();

    Ok(Output {
        count: generated.len(),
        text: model::decode(&generated)?,
        tokens: generated,
    })
}
