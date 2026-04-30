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

use inferlet::{
    Context, Result, Schema, sample::Sampler, model::Model, runtime,
};
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
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    let mut ctx = Context::new(&model)?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&input.prompt);
    ctx.cue();

    let text = ctx
        .generate(Sampler::Argmax)
        .max_tokens(input.max_tokens)
        .constrain_with(inferlet::JsonSchema(PERSON_SCHEMA))?
        .collect_text()
        .await?;

    // Grammar guarantees parse + schema conformance; this is just a sanity
    // step to pretty-print the result.
    let parsed: Value = serde_json::from_str(&text)
        .map_err(|e| format!("JSON parse error (should not happen under grammar constraint): {e}"))?;

    println!(
        "Generated:\n{}",
        serde_json::to_string_pretty(&parsed).unwrap_or(text.clone()),
    );

    Ok(serde_json::to_string(&parsed).unwrap_or(text))
}
