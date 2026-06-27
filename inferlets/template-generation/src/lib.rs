//! Demonstrates template-driven generation: a JSON-Schema-constrained model
//! response feeds a `minijinja` template.
//!
//! `Schema::JsonSchema` guarantees the decoded text parses and conforms to
//! the schema, so `serde_json::from_str` is the only step between the model
//! output and the renderer.

use inferlet::{
    Context, Result, Schema, sample::Sampler,
};
use minijinja::Environment;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_prompt() -> String { "an AI-powered code editor".to_string() }
fn default_max_tokens() -> usize { 1024 }

const TEMPLATE: &str = r#"
========================================
  PRODUCT ANNOUNCEMENT
========================================

{{ product_name | upper }}
"{{ tagline }}"

OVERVIEW
--------
{{ description }}

KEY FEATURES
------------
{% for feature in features %}
  * {{ feature }}
{% endfor %}

PRICING & AVAILABILITY
----------------------
  Price: ${{ price }}
  Release Date: {{ release_date }}
{% if discount_percent %}
  Launch Discount: {{ discount_percent }}% off!
{% endif %}

========================================
"#;

const PRODUCT_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "product_name":     { "type": "string", "minLength": 1 },
        "tagline":          { "type": "string", "minLength": 1 },
        "description":      { "type": "string", "minLength": 1 },
        "features":         { "type": "array", "items": { "type": "string" }, "minItems": 1 },
        "price":            { "type": "string" },
        "release_date":     { "type": "string" },
        "discount_percent": { "type": ["integer", "null"] }
    },
    "required": ["product_name", "tagline", "description", "features", "price", "release_date"],
    "additionalProperties": false
}"#;

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant that generates structured product data. Output \
ONLY a raw JSON object — no markdown, no explanation.";

fn render(data: &Value) -> std::result::Result<String, String> {
    let mut env = Environment::new();
    env.add_template("announcement", TEMPLATE)
        .map_err(|e| format!("Template compile error: {e}"))?;
    let tmpl = env
        .get_template("announcement")
        .map_err(|e| format!("Template lookup error: {e}"))?;
    tmpl.render(data)
        .map_err(|e| format!("Template render error: {e}"))
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mut ctx = Context::new()?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&format!(
        "Generate product announcement data for: {}.",
        input.prompt,
    ));
    ctx.cue();

    let text = ctx
        .generate(Sampler::Argmax)
        .max_tokens(input.max_tokens)
        .constrain_with(inferlet::JsonSchema(PRODUCT_SCHEMA))?
        .collect_text()
        .await?;

    // Grammar enforces a structurally valid prefix at every step;
    // the parse can still fail when `max_tokens` cuts mid-string. On
    // a real model that rarely happens because the model converges;
    // on a backend with no convergence pressure (the dummy driver) it
    // does. Surface the partial text instead of erroring.
    let data: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            println!("(grammar truncated at max_tokens={})", input.max_tokens);
            println!("{}", text);
            eprintln!("Note: structurally valid prefix only ({e}).");
            return Ok(text);
        }
    };

    let rendered = render(&data)?;
    println!("{}", rendered);
    Ok(rendered)
}
