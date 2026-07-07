//! JSON-Schema-constrained generation — **low-level ① rewrite (grammar, SEQUENTIAL)**.
//!
//! Off the `Context`/`Generator`/`Sampler`/`constrain_with` facade onto the
//! keep-core (`ptir-grammar-tranche-conversion-spec`): `JsonSchema(schema)
//! .build_constraint()` → a kept `constraint::GrammarConstraint` (host Matcher,
//! `advance`/`mask`) + `sampler::grammar_program` (`argmax(mask_apply(logits,
//! mask))`) + a sequential per-step fire. Grammar is content-dependent (the mask
//! for token N+1 depends on token N) → NO run-ahead carrier; facade-removal, not
//! pipelining. The grammar enforces a structurally valid prefix every step, so
//! the decoded text parses.

use inferlet::inference::ForwardPass;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, sampler, Constrain, JsonSchema, Result, Schema};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_prompt() -> String {
    "Generate a profile for a fictional software engineer named Alice.".to_string()
}
fn default_max_tokens() -> usize { 512 }

const PERSON_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "name":    { "type": "string", "minLength": 1 },
        "age":     { "type": "integer", "minimum": 0, "maximum": 150 },
        "email":   { "type": "string" },
        "skills":  { "type": "array", "items": { "type": "string" }, "minItems": 1 },
        "address": {
            "type": "object",
            "properties": {
                "city":    { "type": "string" },
                "country": { "type": "string" }
            },
            "required": ["city", "country"]
        }
    },
    "required": ["name", "age", "email", "skills", "address"]
}"#;

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant that generates structured data. Output ONLY a \
raw JSON object — no markdown, no explanation, no whitespace beyond what \
JSON requires.";

async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

/// One SEQUENTIAL grammar fire (geometry + input + masked grammar sampler +
/// execute). No carrier — the next mask depends on this token.
fn grammar_fire(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    g: &sampler::LoweredGrammar,
    tokens: &[u32],
    packed_mask: &[u32],
) -> Result<ForwardPass> {
    let n = tokens.len() as u32;
    let pass = ForwardPass::new();
    if *fresh {
        pass.fresh_generate();
        *fresh = false;
    }
    let geom = geometry::ensure_pages(kv, geometry::kv_write_geometry(*seq_len, n, kv.page_size()))?;
    geometry::attach_kv_write(&pass, kv, &geom);
    let positions: Vec<u32> = (*seq_len..*seq_len + n).collect();
    pass.input_tokens(tokens, &positions);
    let decode_pos = *seq_len + n - 1;
    pass.sampler(&g.program, g.bindings(decode_pos, packed_mask)?);
    pass.execute();
    *seq_len += n;
    Ok(pass)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let vocab = model::output_vocab_size();
    let g = sampler::grammar_program(vocab)?;
    let mut matcher = JsonSchema(PERSON_SCHEMA).build_constraint()?;
    let stop = chat::stop_tokens();

    let mut prompt = chat::system_user(SYSTEM_PROMPT, &input.prompt);
    prompt.extend(chat::cue());

    let kv = KvWorkingSet::new();
    let mut seq = 0u32;
    let mut fresh = true;
    let mut pending = prompt;
    let mut tokens: Vec<u32> = Vec::new();

    for _ in 0..input.max_tokens {
        let m = matcher.mask();
        let packed: Vec<u32> = if m.is_empty() { vec![u32::MAX; g.mask_words] } else { m };
        let pass = grammar_fire(&kv, &mut seq, &mut fresh, &g, &pending, &packed)?;
        let token = read_token(pass).await?;
        if stop.contains(&token) {
            break;
        }
        tokens.push(token);
        matcher.advance(&[token]);
        pending = vec![token];
    }

    let text = model::decode(&tokens).unwrap_or_default();

    // The grammar guarantees a structurally valid prefix; a mid-grammar cut at
    // max_tokens is a normal terminal condition, not a violation.
    match serde_json::from_str::<Value>(&text) {
        Ok(parsed) => {
            println!(
                "Generated:\n{}",
                serde_json::to_string_pretty(&parsed).unwrap_or_else(|_| text.clone()),
            );
            Ok(serde_json::to_string(&parsed).unwrap_or(text))
        }
        Err(e) => {
            println!("Generated (truncated at max_tokens={}):\n{}", input.max_tokens, text);
            eprintln!(
                "Note: output is a structurally valid grammar prefix but \
                 max_tokens cut before the JSON closed ({e})."
            );
            Ok(text)
        }
    }
}
