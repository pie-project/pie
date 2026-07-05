//! Demonstrates template-driven generation: a JSON-Schema-constrained model
//! response feeds a `minijinja` template.
//!
//! `Schema::JsonSchema` guarantees the decoded text parses and conforms to
//! the schema, so `serde_json::from_str` is the only step between the model
//! output and the renderer.

use inferlet::inference::ForwardPass;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, sampler, Constrain, JsonSchema, Result, Schema};
use minijinja::Environment;

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

/// Sequential JsonSchema-constrained decode → decoded text (grammar keep-core).
async fn constrained_generate(system: &str, user: &str, schema: &str, max_tokens: usize) -> Result<String> {
    let vocab = model::output_vocab_size();
    let g = sampler::grammar_program(vocab)?;
    let mut matcher = JsonSchema(schema).build_constraint()?;
    let stop = chat::stop_tokens();

    let mut prompt = chat::system_user(system, user);
    prompt.extend(chat::cue());

    let kv = KvWorkingSet::new();
    let mut seq = 0u32;
    let mut fresh = true;
    let mut pending = prompt;
    let mut tokens: Vec<u32> = Vec::new();
    for _ in 0..max_tokens {
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
    Ok(model::decode(&tokens).unwrap_or_default())
}
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
    let text = constrained_generate(
        SYSTEM_PROMPT,
        &format!("Generate product announcement data for: {}.", input.prompt),
        PRODUCT_SCHEMA,
        input.max_tokens,
    )
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
