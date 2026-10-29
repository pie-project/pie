//! Generates JSON while enforcing a caller-supplied JSON Schema.
//!
//! The host grammar matcher advances after every accepted token and supplies
//! the next allowed-token mask to an ETA `mask_apply` + argmax epilogue.

use inferlet::chat;
use inferlet::eta::prelude::*;
use inferlet::grammar::{Grammar, Matcher};
use inferlet::mask::bit_allowed;
use serde::Deserialize;
use serde_json::Value;

fn is_deterministic_schema_state(mask: &[u32], vocab: u32, digit_tokens: &[u32]) -> bool {
    if !digit_tokens.is_empty()
        && digit_tokens
            .iter()
            .all(|&t| bit_allowed(mask, t as usize))
    {
        return false;
    }
    let mut count = 0u32;
    for (w_idx, &word) in mask.iter().enumerate() {
        let base = (w_idx as u32) * 32;
        if base >= vocab {
            break;
        }
        let valid_bits = (vocab - base).min(32);
        let valid_mask = if valid_bits == 32 {
            u32::MAX
        } else {
            (1u32 << valid_bits) - 1
        };
        count += (word & valid_mask).count_ones();
        if count > 256 {
            return false;
        }
    }
    count > 0
}

fn match_deterministic_token(
    history: &[u32],
    mask: &[u32],
    vocab: u32,
    digit_tokens: &[u32],
) -> Option<u32> {
    let mut single: Option<u32> = None;
    let mut count = 0u32;
    for (w_idx, &word) in mask.iter().enumerate() {
        let base = (w_idx as u32) * 32;
        if base >= vocab {
            break;
        }
        let valid_bits = (vocab - base).min(32);
        let valid_mask = if valid_bits == 32 {
            u32::MAX
        } else {
            (1u32 << valid_bits) - 1
        };
        let w = word & valid_mask;
        if w == 0 {
            continue;
        }
        count += w.count_ones();
        if count == 1 {
            single = Some(base + w.trailing_zeros());
        } else if count > 256 {
            return None;
        }
    }
    if count == 1 {
        return single;
    }
    if !is_deterministic_schema_state(mask, vocab, digit_tokens) || history.len() < 2 {
        return None;
    }
    let max_match = 8.min(history.len() - 1);
    for width in (1..=max_match).rev() {
        let suffix = &history[history.len() - width..];
        for start in (0..history.len() - width).rev() {
            if &history[start..start + width] != suffix {
                continue;
            }
            let cand = history[start + width];
            if bit_allowed(mask, cand as usize) {
                return Some(cand);
            }
        }
    }
    None
}

struct DecodePass {
    decode: ForwardPass,
    token_in: Channel,
    grammar_mask: Channel,
    token_out: Channel,
}

impl DecodePass {
    fn new(
        len: u32,
        vocab: u32,
        words: u32,
        ws: &WorkingSet,
        pow2_d: &Channel,
    ) -> Result<Self> {
        let token_in = Channel::new([len], dtype::i32);
        let grammar_mask = Channel::new([words], dtype::u32);
        let embed_indptr = Channel::from([0u32, len]);
        let token_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        let pow2 = pow2_d.clone();

        let decode = ForwardPass::new();
        decode.embed(&token_in, &embed_indptr)?;
        decode.state(&[ws], StateGeometry::folding())?;
        let token_out_ep = token_out.clone();
        let grammar_mask_ep = grammar_mask.clone();
        decode.epilogue(move || {
            let packed = grammar_mask_ep.take();
            let j = iota(vocab);
            let word = gather(&packed, &j / Tensor::constant(32u32));
            let p2 = gather(pow2.read(), &j % Tensor::constant(32u32));
            let bit = (&word / &p2) % Tensor::constant(2u32);
            let allowed = ne(&bit, Tensor::constant(0u32));
            let token = reshape(
                masked_argmax(intrinsics::output(0, "logits"), &allowed),
                [1],
            );
            token_out_ep.put(&token);
        });

        Ok(Self {
            decode,
            token_in,
            grammar_mask,
            token_out,
        })
    }
}

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
    let digit_tokens: Vec<u32> = (1..=9)
        .filter_map(|d| model::encode(&d.to_string()).into_iter().next())
        .collect();
    let stream = input.stream;
    let mut generated = Vec::with_capacity(input.max_tokens);
    let mut mask = pack_mask(constraint.mask());
    while prompt.len() < 1024 && generated.len() + 1 < input.max_tokens {
        let Some(forced) = match_deterministic_token(&generated, &mask, vocab, &digit_tokens) else {
            break;
        };
        if stream {
            inferlet::session::send("t:1");
        }
        prompt.push(forced);
        generated.push(forced);
        constraint
            .accept_tokens(&[forced])
            .context("advance grammar")?;
        if constraint.is_terminated() {
            break;
        }
        mask = pack_mask(constraint.mask());
    }

    let pipeline = Pipeline::new();
    if !constraint.is_terminated() && generated.len() < input.max_tokens {
        let n = prompt.len() as u32;
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

        prefill_mask.put(mask);
        prefill.submit(&pipeline).context("JSON-schema prefill")?;
        let first = first_out.take_host::<i32>().await? as u32;
        if stream {
            inferlet::session::send("t:1");
        }

        generated.push(first);
        constraint
            .accept_tokens(&[first])
            .context("advance grammar")?;
    }

    if !constraint.is_terminated() && generated.len() < input.max_tokens {
        let pow2_d = Channel::from_iter((0..32u32).map(|b| 1u32 << b));
        let mut pass_1: Option<DecodePass> = None;

        while !constraint.is_terminated() && generated.len() < input.max_tokens {
            let last = *generated.last().unwrap() as i32;
            let mut chunk = vec![last];
            let mut mask = pack_mask(constraint.mask());

            while chunk.len() < 16 && generated.len() < input.max_tokens {
                let Some(forced) =
                    match_deterministic_token(&generated, &mask, vocab, &digit_tokens)
                else {
                    break;
                };
                if stream {
                    inferlet::session::send("t:1");
                }
                generated.push(forced);
                constraint
                    .accept_tokens(&[forced])
                    .context("advance grammar")?;
                if constraint.is_terminated() || generated.len() == input.max_tokens {
                    break;
                }
                chunk.push(forced as i32);
                mask = pack_mask(constraint.mask());
            }

            if constraint.is_terminated() || generated.len() == input.max_tokens {
                break;
            }

            let len = chunk.len() as u32;
            let token = if len == 1 {
                if pass_1.is_none() {
                    pass_1 = Some(DecodePass::new(1, vocab, words, &ws, &pow2_d)?);
                }
                let pass = pass_1.as_ref().unwrap();
                pass.token_in.put(chunk);
                pass.grammar_mask.put(mask);
                pass.decode.submit(&pipeline).context("JSON-schema decode")?;
                pass.token_out.take_host::<i32>().await? as u32
            } else {
                let pass = DecodePass::new(len, vocab, words, &ws, &pow2_d)?;
                pass.token_in.put(chunk);
                pass.grammar_mask.put(mask);
                pass.decode.submit(&pipeline).context("JSON-schema decode")?;
                pass.token_out.take_host::<i32>().await? as u32
            };
            if stream {
                inferlet::session::send("t:1");
            }
            generated.push(token);
            constraint
                .accept_tokens(&[token])
                .context("advance grammar")?;
        }
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
