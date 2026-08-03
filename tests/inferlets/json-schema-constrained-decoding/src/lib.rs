//! Generates JSON while enforcing a caller-supplied JSON Schema.
//!
//! The host grammar matcher advances after every accepted token and supplies
//! the next allowed-token mask to a PTIR `mask_apply` + argmax epilogue.

use inferlet::mask::bit_allowed;
use inferlet::ptir::attention::prelude::*;
use inferlet::{Constrain, JsonSchema, Schema, chat};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_schema")]
    schema: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_prompt() -> String {
    "Generate a profile for a fictional software engineer named Alice.".into()
}

fn default_schema() -> String {
    r#"{
        "type": "object",
        "properties": {
            "name": { "type": "string", "minLength": 1 },
            "age": { "type": "integer", "minimum": 0, "maximum": 150 },
            "skills": {
                "type": "array",
                "items": { "type": "string" },
                "minItems": 1
            }
        },
        "required": ["name", "age", "skills"],
        "additionalProperties": false
    }"#
    .into()
}

fn default_max_tokens() -> usize {
    512
}

fn unpack_mask(packed: &[u32], vocab: u32) -> Vec<bool> {
    if packed.is_empty() {
        return vec![true; vocab as usize];
    }
    (0..vocab as usize)
        .map(|token| bit_allowed(packed, token))
        .collect()
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if input.max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }

    let vocab = model::output_vocab_size();
    let ws = WorkingSet::new();
    let page_size = ws.page_size();
    let mut constraint = JsonSchema(&input.schema).build_constraint()?;

    let mut prompt = chat::system_user(
        "Generate only the requested JSON value, with no markdown or explanation.",
        &input.prompt,
    );
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + input.max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;

    let prompt_tokens = Channel::from(prompt.iter().map(|&token| token as i32).collect::<Vec<_>>());
    let prefill_indptr = Channel::from(vec![0u32, n]).named("prefill_indptr");
    let prefill_positions = Channel::from((0..n).collect::<Vec<_>>()).named("prefill_positions");
    let prefill_pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("prefill_pages");
    let prefill_page_indptr =
        Channel::from(vec![0u32, n.div_ceil(page_size)]).named("prefill_page_indptr");
    let prefill_w_slot =
        Channel::from((0..n).map(|p| p / page_size).collect::<Vec<_>>()).named("prefill_w_slot");
    let prefill_w_off =
        Channel::from((0..n).map(|p| p % page_size).collect::<Vec<_>>()).named("prefill_w_off");
    let prefill_mask = Channel::new([vocab], dtype::bool).named("prefill_mask");
    let first_out = Channel::new([1], dtype::i32).named("first_token");

    let prefill = ForwardPass::new();
    prefill.embed(&prompt_tokens, &prefill_indptr)?;
    let prefill_kv_len = Channel::from(vec![n]).named("prefill_kv_len");
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
        let allowed = prefill_mask.take();
        let token = reshape(masked_argmax(intrinsics::logits(), &allowed), [1]);
        first_out.put(&token);
    });

    prefill_mask.put(unpack_mask(&constraint.mask(), vocab));
    // ONE pipeline for the whole stream (R4-4): prefill and decode are one
    // sequential stream. The host round-trip on `first` stays — the grammar
    // matcher advances on it before decode is built.
    let pipeline = Pipeline::new();
    prefill.submit(&pipeline).context("JSON-schema prefill")?;
    // max_tokens == 1: the prefill spends the whole budget, so it was the
    // stream's last submit — finish() right after it (F7).
    let first = first_out
        .take()
        .get::<i32>()
        .await
        .context("read first constrained token")?[0] as u32;

    let mut generated = vec![first];
    constraint.advance(&[first]);

    if !constraint.is_terminated() && generated.len() < input.max_tokens {
        let token_in = Channel::from(vec![first as i32]).named("token_in");
        let grammar_mask = Channel::new([vocab], dtype::bool).named("grammar_mask");
        let embed_indptr = Channel::from(vec![0u32, 1]).named("embed_indptr");
        let positions = Channel::from(vec![n]).named("positions");
        let pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages");
        let page_indptr =
            Channel::from(vec![0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from(vec![n / page_size]).named("w_slot");
        let w_off = Channel::from(vec![n % page_size]).named("w_off");
        let token_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("token_out");

        let decode = ForwardPass::new();
        decode.embed(&token_in, &embed_indptr)?;
        let kv_len = Channel::from(vec![n + 1]).named("kv_len");
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
            let length = kv_len.take().tensor();
            let allowed = grammar_mask.take();
            let token = reshape(masked_argmax(intrinsics::logits(), &allowed), [1]);
            let next_length = add(&length, 1u32);
            let page_count = div(add(&next_length, page_size - 1), page_size);

            token_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(div(&length, page_size));
            w_off.put(rem(&length, page_size));
            page_indptr.take();
            page_indptr.put(mul(iota(2), broadcast(&page_count, [2])));
            token_out.put(&token);
        });

        // The grammar mask for fire k+1 is only known once fire k's token has
        // advanced the matcher, so this loop is inherently depth-1. Running
        // ahead would reuse a stale mask and silently drop the constraint.
        let budget = input.max_tokens.saturating_sub(generated.len());
        let mut submitted = 0usize;

        while submitted < budget {
            grammar_mask.put(unpack_mask(&constraint.mask(), vocab));
            decode.submit(&pipeline).context("JSON-schema decode")?;
            submitted += 1;
            let token = token_out
                .take()
                .get::<i32>()
                .await
                .context("read constrained token")?[0] as u32;
            generated.push(token);
            constraint.advance(&[token]);
            if constraint.is_terminated() || generated.len() == input.max_tokens {
                break;
            }
        }
    }
    pipeline.close();

    if !constraint.is_terminated() {
        return Err(format!(
            "JSON generation did not terminate within {} tokens",
            input.max_tokens
        ));
    }

    let text = model::decode(&generated)?;
    serde_json::from_str::<Value>(&text)
        .map_err(|e| format!("constraint terminated with invalid JSON: {e}; output={text:?}"))?;
    Ok(text)
}
