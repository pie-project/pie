//! Generates JSON while enforcing a caller-supplied JSON Schema.
//!
//! The host grammar matcher advances after every accepted token and supplies
//! the next allowed-token mask to a PTIR `mask_apply` + argmax epilogue.

use inferlet::mask::bit_allowed;
use inferlet::ptir::prelude::*;
use inferlet::{Constrain, JsonSchema, Result, Schema, chat, model as wit_model};
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

fn bx<T>(value: T) -> &'static T {
    Box::leak(Box::new(value))
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

    let vocab = wit_model::output_vocab_size();
    let ws: &'static WorkingSet = bx(WorkingSet::new());
    model::configure(vocab, ws.page_size(), 1);
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

    let prompt_tokens = bx(Channel::from(
        prompt.iter().map(|&token| token as i32).collect::<Vec<_>>(),
    ));
    let prefill_klen = bx(Channel::from(vec![n]));
    let prefill_mask = bx(Channel::new([vocab], dtype::bool).named("prefill_mask"));
    let first_out = bx(Channel::new([1], dtype::i32).named("first_token"));

    let prefill: ForwardPass<'static> = ForwardPass::new();
    prefill.embed(prompt_tokens, Tensor::constant(vec![0u32, n]));
    prefill.attn_working_set(ws, prefill_klen);
    prefill.epilogue(move || {
        let allowed = prefill_mask.take();
        let token = reshape(
            reduce_argmax(mask_apply(intrinsics::logits(), &allowed)),
            [1],
        );
        first_out.put(&token);
    });

    prefill_mask.put(unpack_mask(&constraint.mask(), vocab));
    let prefill_pipeline = Pipeline::new();
    prefill
        .submit(&prefill_pipeline)
        .map_err(|e| format!("JSON-schema prefill: {e}"))?;
    let first = first_out
        .take()
        .get::<i32>()
        .map_err(|e| format!("read first constrained token: {e}"))?[0] as u32;
    prefill_pipeline.close();

    let mut generated = vec![first];
    constraint.advance(&[first]);

    if !constraint.is_terminated() && generated.len() < input.max_tokens {
        let token_in = bx(Channel::from(vec![first as i32]).named("token_in"));
        let position = bx(Channel::from(vec![n]).named("position"));
        let klen = bx(Channel::from(vec![n + 1]).named("klen"));
        let fill = bx(Channel::from(vec![n + 1]).named("fill"));
        let grammar_mask = bx(Channel::new([vocab], dtype::bool).named("grammar_mask"));
        let token_out = bx(Channel::new([1], dtype::i32).named("token_out"));

        let decode: ForwardPass<'static> = ForwardPass::new();
        decode.embed(token_in, Tensor::constant(vec![0u32, 1]));
        decode.positions(position);
        decode.attn_working_set(ws, klen);
        decode.epilogue(move || {
            let base = fill.take().tensor();
            let allowed = grammar_mask.take();
            let token = reshape(
                reduce_argmax(mask_apply(intrinsics::logits(), &allowed)),
                [1],
            );
            let next = add(&base, 1u32);

            token_in.put(&token);
            token_out.put(&token);
            position.put(&base);
            klen.put(&next);
            fill.put(&next);
        });

        let pipeline = Pipeline::new();
        while generated.len() < input.max_tokens && !constraint.is_terminated() {
            grammar_mask.put(unpack_mask(&constraint.mask(), vocab));
            decode
                .submit(&pipeline)
                .map_err(|e| format!("JSON-schema decode: {e}"))?;
            let token = token_out
                .take()
                .get::<i32>()
                .map_err(|e| format!("read constrained token: {e}"))?[0]
                as u32;
            generated.push(token);
            constraint.advance(&[token]);
        }
        pipeline.close();
    }

    if !constraint.is_terminated() {
        return Err(format!(
            "JSON generation did not terminate within {} tokens",
            input.max_tokens
        ));
    }

    let text = wit_model::decode(&generated)?;
    serde_json::from_str::<Value>(&text)
        .map_err(|e| format!("constraint terminated with invalid JSON: {e}; output={text:?}"))?;
    Ok(text)
}
