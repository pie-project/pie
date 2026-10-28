//! Generates JSON while enforcing a caller-supplied JSON Schema.
//!
//! The host grammar matcher advances after every accepted token and supplies
//! the next allowed-token mask to an ETA `mask_apply` + argmax epilogue.

use inferlet::chat;
use inferlet::eta::prelude::*;
use inferlet::grammar::{Grammar, Matcher};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default = "default_schema")]
    schema: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
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
    if input.max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }

    let vocab = model::output_vocab_size();
    let words = vocab.div_ceil(32);
    let pack_mask = |mut m: Vec<u32>| -> Vec<u32> {
        let fill = if m.is_empty() { u32::MAX } else { 0 };
        m.resize(words as usize, fill);
        m
    };
    let ws = WorkingSet::new();
    let constraint = Matcher::new(&Grammar::from_json_schema(&input.schema)?);

    let mut prompt = if let Some(toks) = input.prompt_tokens {
        toks
    } else {
        let mut p = chat::system_user(
            "Generate only the requested JSON value, with no markdown or explanation.",
            &input.prompt,
        );
        p.extend(chat::cue_without_thinking());
        p
    };
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stream = input.stream;

    let prompt_tokens = Channel::from_iter(prompt.iter().map(|&token| token as i32));
    let prefill_indptr = Channel::from([0u32, n]);
    let prefill_mask = Channel::new([words], dtype::u32);
    let pow2_p = Channel::from_iter((0..32u32).map(|b| 1u32 << b));
    let first_out = Channel::new([1], dtype::i32);

    let prefill = ForwardPass::new();
    prefill.embed(&prompt_tokens, &prefill_indptr)?;
    prefill.state(&[&ws], StateGeometry::folding())?;
    prefill.epilogue(move || {
        let packed = prefill_mask.take();
        let j = iota(vocab);
        let word = gather(&packed, &j / Tensor::constant(32u32));
        let p2 = gather(pow2_p.read(), &j % Tensor::constant(32u32));
        let bit = (&word / &p2) % Tensor::constant(2u32);
        let allowed = ne(&bit, Tensor::constant(0u32));
        let token = reshape(
            masked_argmax(intrinsics::output(0, "logits"), &allowed),
            [1],
        );
        first_out.put(&token);
    });

    prefill_mask.put(pack_mask(constraint.mask()));
    let pipeline = Pipeline::new();
    prefill.submit(&pipeline).context("JSON-schema prefill")?;
    let first = first_out.take_host::<i32>().await? as u32;
    if stream {
        inferlet::session::send("t:1");
    }

    let mut generated = vec![first];
    constraint
        .accept_tokens(&[first])
        .context("advance grammar")?;

    if !constraint.is_terminated() && generated.len() < input.max_tokens {
        let token_in = Channel::from([first as i32]);
        let grammar_mask = Channel::new([words], dtype::u32);
        let pow2_d = Channel::from_iter((0..32u32).map(|b| 1u32 << b));
        let embed_indptr = Channel::from([0u32, 1]);
        let token_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);

        let decode = ForwardPass::new();
        decode.embed(&token_in, &embed_indptr)?;
        decode.state(&[&ws], StateGeometry::folding())?;
        decode.epilogue(move || {
            let packed = grammar_mask.take();
            let j = iota(vocab);
            let word = gather(&packed, &j / Tensor::constant(32u32));
            let p2 = gather(pow2_d.read(), &j % Tensor::constant(32u32));
            let bit = (&word / &p2) % Tensor::constant(2u32);
            let allowed = ne(&bit, Tensor::constant(0u32));
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
        let budget = input.max_tokens.saturating_sub(generated.len());
        let mut submitted = 0usize;

        while submitted < budget {
            grammar_mask.put(pack_mask(constraint.mask()));
            decode.submit(&pipeline).context("JSON-schema decode")?;
            submitted += 1;
            let token = token_out.take_host::<i32>().await? as u32;
            if stream {
                inferlet::session::send("t:1");
            }
            generated.push(token);
            constraint
                .accept_tokens(&[token])
                .context("advance grammar")?;
            if constraint.is_terminated() || generated.len() == input.max_tokens {
                break;
            }
        }
    }
    pipeline.close();

    if input.stream {
        return Ok(model::decode(&generated).unwrap_or_default());
    }

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
