//! Demonstrates JSON-Schema-constrained generation against a *string* schema.
//!
//! Use this pattern when the JSON schema is defined externally (loaded at
//! runtime, supplied by a user, etc.) and you don't have a matching Rust
//! struct. For the typed path, see `collect_json::<T>` where `T` derives
//! `JsonSchema` — that's a one-liner.
//!
//! The grammar enforces structural validity *during* generation, so the
//! decoded text is guaranteed to parse and conform to the schema. Parsing
//! into `serde_json::Value` is the only post-processing required.

use inferlet::{Context, Result, sample::Sampler};
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

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mut ctx = Context::new()?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&input.prompt);
    ctx.cue();

    let text = ctx
        .generate(Sampler::Argmax)
        .max_tokens(input.max_tokens)
        .constrain_with(inferlet::JsonSchema(PERSON_SCHEMA))?
        .collect_text()
        .await?;

    // The grammar enforces a structurally valid prefix at every step, so
    // the happy path is `parsed = serde_json::from_str(&text)`. The parse
    // can still fail when `max_tokens` cuts mid-grammar — e.g. while a
    // long string field is still open. That is a normal terminal
    // condition, not a grammar violation, so report the partial output
    // instead of erroring.
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
                 max_tokens cut before the JSON closed ({e}). Increase \
                 max_tokens or run on a model that converges sooner."
            );
            Ok(text)
        }
    }
}
