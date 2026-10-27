//! Generates JSON while enforcing a caller-supplied JSON Schema.
//!
//! The host grammar matcher advances after every accepted token and supplies
//! the next allowed-token mask to an ETA `mask_apply` + argmax epilogue.

use inferlet::chat;
use inferlet::eta::prelude::*;
use inferlet::grammar::{Grammar, Matcher};
use inferlet::mask::unpack_mask;
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

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let vocab = model::output_vocab_size();
    let ws = WorkingSet::new();
    let constraint = Matcher::new(&Grammar::from_json_schema(&input.schema)?);

    let mut prompt = chat::system_user(
        "Generate only the requested JSON value, with no markdown or explanation.",
        &input.prompt,
    );
    prompt.extend(chat::cue());
    let prompt: Vec<i32> = prompt.into_iter().map(|t| t as i32).collect();
    let (&last, head) = prompt.split_last().ok_or("empty prompt")?;

    let pipeline = Pipeline::new();
    if !head.is_empty() {
        let fill = ForwardPass::new();
        fill.embed(
            &Channel::from(head),
            &Channel::from([0u32, head.len() as u32]),
        )?;
        fill.state(&[&ws], StateGeometry::folding())?;
        fill.submit(&pipeline).context("JSON-schema prefill")?;
    }

    let token_in = Channel::from([last]);
    let grammar_mask = Channel::new([vocab], dtype::bool);
    let embed_indptr = Channel::from([0u32, 1]);
    let token_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);

    let decode = ForwardPass::new();
    decode.embed(&token_in, &embed_indptr)?;
    decode.state(&[&ws], StateGeometry::folding())?;
    decode.epilogue(move || {
        let allowed = grammar_mask.take();
        let token = reshape(
            masked_argmax(intrinsics::output(0, "logits"), &allowed),
            [1],
        );

        token_in.put(&token);
        token_out.put(&token);
    });

    // The grammar mask for fire k+1 is only known once fire k's token has
    // advanced the matcher, so this loop is inherently depth-1. Running
    // ahead would reuse a stale mask and silently drop the constraint.
    let mut generated = Vec::with_capacity(input.max_tokens);
    while !constraint.is_terminated() && generated.len() < input.max_tokens {
        grammar_mask.put(unpack_mask(&constraint.mask(), vocab));
        decode.submit(&pipeline).context("JSON-schema decode")?;
        let token = token_out.take_host::<i32>().await? as u32;
        generated.push(token);
        constraint
            .accept_tokens(&[token])
            .context("advance grammar")?;
    }
    pipeline.close();

    // A budget too small for the schema is the common way here: say so
    // WITH the partial value, so a sweep reading this error does not mistake
    // a bounded, finished loop for a hang.
    if !constraint.is_terminated() {
        let partial = model::decode(&generated).unwrap_or_default();
        return Err(format!(
            "JSON generation did not terminate within {} tokens (schema not yet satisfied); partial output={partial:?}",
            input.max_tokens
        ));
    }

    let text = model::decode(&generated)?;
    serde_json::from_str::<Value>(&text)
        .map_err(|e| format!("constraint terminated with invalid JSON: {e}; output={text:?}"))?;
    Ok(text)
}
