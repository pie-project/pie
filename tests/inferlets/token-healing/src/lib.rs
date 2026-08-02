//! Token healing — repairing the boundary artefact at the end of a prompt.
//!
//! A tokenizer is greedy and left-to-right, so a prompt that stops mid-word
//! ends in a token the model would never have produced in that position. Given
//! `"The URL is http:"`, the tokenizer emits `…, "http", ":"` — but in real
//! text the model overwhelmingly emits `"://"` as a single token, and `":"`
//! followed by `"//"` is a sequence it has almost never seen. Conditioning on
//! that split token biases, and often destroys, the continuation.
//!
//! Token healing (Guidance, and llama.cpp's `--token-healing`) fixes this
//! without touching the model:
//!
//! 1. drop the last `backoff` tokens of the prompt, remembering the bytes they
//!    covered as a fragment `f`;
//! 2. prefill the shortened prompt;
//! 3. constrain the first generated token to the set `{v : bytes(v) starts with f}`;
//! 4. decode unconstrained from there.
//!
//! Step 3 is what makes it *healing* rather than truncation: the emitted token
//! must still reproduce every byte the caller wrote, so the visible prompt is
//! unchanged, but the model is free to pick the tokenisation it actually
//! expects. The mask is exactly the constrained-decoding mask this engine
//! already applies for grammars, so healing costs one `mask_apply` on one step.
//!
//! `heal = false` runs the identical geometry with an all-ones mask, which is
//! the unhealed baseline; the two runs differ only in that first mask.
//!
//! ## Source
//!
//! No paper. Reference implementations: guidance-ai/guidance —
//! <https://github.com/guidance-ai/guidance> — its `llguidance` backend —
//! <https://github.com/guidance-ai/llguidance> — and llama.cpp's
//! `--token-healing`.
//!
//! Faithfulness: **Exact**. See
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::ptir::attention::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_heal")]
    heal: bool,
    #[serde(default = "default_backoff")]
    backoff: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    healed: bool,
    /// Bytes rolled off the prompt and re-expanded, as text.
    fragment: String,
    /// Tokens whose byte sequence starts with the fragment.
    prefix_candidates: usize,
    /// The token the healed step chose, and its byte length. A healed run picks
    /// a token strictly longer than the fragment whenever the model prefers to
    /// carry the boundary itself.
    healed_token: u32,
    healed_token_bytes: usize,
    /// Whether the healed step reproduced the caller's bytes exactly.
    prompt_preserved: bool,
    text: String,
    count: usize,
}

fn default_prompt() -> String {
    "The documentation link is http:".into()
}

fn default_heal() -> bool {
    true
}

fn default_backoff() -> usize {
    1
}

fn default_max_tokens() -> usize {
    32
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if input.max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }
    if input.backoff == 0 || input.backoff > 4 {
        return Err("backoff must satisfy 1 <= backoff <= 4".into());
    }

    let vocab = model::output_vocab_size();
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    let full = model::encode(&input.prompt);
    if full.len() <= input.backoff {
        return Err("prompt is too short to roll back that many tokens".into());
    }

    // The fragment is the *bytes* the rolled-back tokens covered, not their
    // text: a prefix test on decoded strings would break on tokens whose bytes
    // are not valid UTF-8 on their own, which is most multi-byte tokens.
    let vocabs = model::vocabs();
    let mut token_bytes: Vec<&[u8]> = vec![&[]; vocab as usize];
    for token in &vocabs {
        if (token.id as usize) < token_bytes.len() {
            token_bytes[token.id as usize] = token.bytes.as_slice();
        }
    }

    let split = full.len() - input.backoff;
    let mut fragment: Vec<u8> = Vec::new();
    for &token in &full[split..] {
        fragment.extend_from_slice(token_bytes[token as usize]);
    }
    if fragment.is_empty() {
        return Err("the rolled-back tokens carry no bytes to heal".into());
    }

    // The healing mask: every token that reproduces the fragment as a prefix.
    // A token equal to the fragment is included, so healing can always fall
    // back to the tokenizer's own choice and never removes a legal completion.
    let mut prefix_mask = vec![false; vocab as usize];
    let mut prefix_candidates = 0usize;
    for (token, bytes) in token_bytes.iter().enumerate() {
        if bytes.starts_with(&fragment) {
            prefix_mask[token] = true;
            prefix_candidates += 1;
        }
    }
    if prefix_candidates == 0 {
        return Err("no vocabulary token starts with the fragment".into());
    }

    // The unhealed baseline keeps the tokenizer's greedy split verbatim; the
    // healed run rolls it back and re-expands it under the prefix mask. Anything
    // else would compare healing against a *truncated* prompt rather than
    // against the artefact it exists to repair.
    let (mut prompt, first_mask) = if input.heal {
        (full[..split].to_vec(), prefix_mask)
    } else {
        (full.clone(), vec![true; vocab as usize])
    };
    if prompt.is_empty() {
        prompt.push(0);
    }

    let n = prompt.len() as u32;
    let max_pages = (n + input.max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;

    let prompt_tokens = Channel::from_iter(prompt.iter().map(|&t| t as i32));
    let prefill_indptr = Channel::from([0u32, n]).named("prefill_indptr");
    let prefill_positions = Channel::from_iter(0..n).named("prefill_positions");
    let prefill_pages = Channel::from_iter(0..max_pages).named("prefill_pages");
    let prefill_page_indptr =
        Channel::from([0u32, n.div_ceil(page_size)]).named("prefill_page_indptr");
    let prefill_w_slot = Channel::from_iter((0..n).map(|p| p / page_size)).named("prefill_w_slot");
    let prefill_w_off = Channel::from_iter((0..n).map(|p| p % page_size)).named("prefill_w_off");
    let prefill_kv_len = Channel::from([n]).named("prefill_kv_len");
    let heal_mask = Channel::new([vocab], dtype::bool).named("heal_mask");
    let first_out = Channel::new([1], dtype::i32).named("first_token");

    let prefill = ForwardPass::new();
    prefill.embed(&prompt_tokens, &prefill_indptr)?;
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
        let allowed = heal_mask.take();
        first_out.put(reshape(masked_argmax(intrinsics::logits(), &allowed), [1]));
    });

    heal_mask.put(first_mask);
    let pipeline = Pipeline::new();
    prefill.submit(&pipeline).context("token-healing prefill")?;
    let first = first_out.take_host::<i32>().await? as u32;

    let mut generated = vec![first];

    if generated.len() < input.max_tokens {
        let token_in = Channel::from([first as i32]).named("token_in");
        let embed_indptr = Channel::from([0u32, 1]).named("embed_indptr");
        let positions = Channel::from([n]).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([n / page_size]).named("w_slot");
        let w_off = Channel::from([n % page_size]).named("w_off");
        let kv_len = Channel::from([n + 1]).named("kv_len");
        let token_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("token_out");

        let decode = ForwardPass::new();
        decode.embed(&token_in, &embed_indptr)?;
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
            let token = reshape(reduce_argmax(intrinsics::logits()), [1]);
            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);

            token_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            token_out.put(&token);
        });

        let budget = input.max_tokens - 1;
        run_ahead(&pipeline, &decode, budget as usize, async || {
            let token = token_out.take_host::<i32>().await? as u32;
            generated.push(token);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipeline.close();

    let healed_bytes = token_bytes[first as usize];
    // The whole point of the mask: whatever the model chose, the caller's
    // prompt must still be reproduced byte for byte.
    let prompt_preserved = !input.heal || healed_bytes.starts_with(&fragment);
    if input.heal && !prompt_preserved {
        return Err(format!(
            "healing broke the prompt: token {first} does not start with the fragment"
        ));
    }

    Ok(Output {
        sampler: "token-healing",
        healed: input.heal,
        fragment: String::from_utf8_lossy(&fragment).into_owned(),
        prefix_candidates,
        healed_token: first,
        healed_token_bytes: healed_bytes.len(),
        prompt_preserved,
        text: model::decode(&generated)?,
        count: generated.len(),
    })
}
