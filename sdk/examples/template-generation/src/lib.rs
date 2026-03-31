//! Demonstrates template-driven generation with grammar-constrained decoding.
//!
//! This example combines three layers:
//! 1. A Lark grammar via `llguidance` ensures every generated token produces
//!    syntactically valid JSON (no parse errors possible).
//! 2. The `jsonschema` library validates the JSON against a product schema.
//! 3. The `minijinja` library renders the validated data through a Jinja2-style
//!    template. When validation or rendering fails, the error is fed back and
//!    the model regenerates -- still grammar-constrained.

mod sampler;
mod schema;
mod template;

use inferlet::{Args, Result, anyhow};
use sampler::ConstrainedSampler;
use schema::SchemaValidator;
use template::TemplateRenderer;

const HELP: &str = "\
Usage: template-generation [OPTIONS]

Generates structured JSON data from an LLM with grammar-constrained decoding,
validates it against a JSON Schema, then renders it through a Jinja2-style
template using minijinja.

Options:
  -p, --prompt <STRING>      The product/topic to generate content for
                             [default: AI-powered code editor]
  -r, --max-retries <N>      Maximum generation/render retry cycles [default: 3]
  -t, --max-tokens <N>       Max tokens per generation attempt [default: 1024]
  -h, --help                 Prints help information";

const JSON_GRAMMAR: &str = r##"
?start: value
?value: object
        | array
        | string
        | NUMBER             -> number
        | "true"             -> true
        | "false"            -> false
        | "null"             -> null
array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : string ":" value
string : ESCAPED_STRING
NUMBER : /-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?/
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"##;

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
        "product_name": {
            "type": "string",
            "minLength": 1
        },
        "tagline": {
            "type": "string",
            "minLength": 1
        },
        "description": {
            "type": "string",
            "minLength": 1
        },
        "features": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 1
        },
        "price": {
            "type": "string"
        },
        "release_date": {
            "type": "string"
        },
        "discount_percent": {
            "type": ["integer", "null"]
        }
    },
    "required": ["product_name", "tagline", "description", "features", "price", "release_date"]
}"#;

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant that generates structured product data. \
Output ONLY a raw JSON object with no additional text, markdown fences, or explanation. \
The JSON must conform to the JSON Schema provided in the user message. \
If you receive validation or rendering errors, fix the JSON to address the issues \
and output only the corrected JSON object.";

fn build_initial_prompt(user_prompt: &str, schema: &str) -> String {
    format!(
        "Generate product announcement data for: {}.\n\n\
        The output must conform to this JSON Schema:\n{}\n\n\
        Output only the JSON object, nothing else.",
        user_prompt, schema
    )
}

fn build_retry_prompt(errors: &str) -> String {
    format!(
        "The JSON you produced has validation/rendering errors:\n{}\n\n\
        Please fix these errors and output only the corrected JSON object, nothing else.",
        errors
    )
}

/// Runs a grammar-constrained decode loop that produces syntactically valid JSON.
async fn generate_json(
    ctx: &mut inferlet::Context,
    sampler: &ConstrainedSampler,
    tokenizer: &inferlet::Tokenizer,
    eos_sequences: &[Vec<u32>],
    max_tokens: usize,
) -> String {
    let mut generated_token_ids = Vec::new();
    loop {
        let dist = ctx.decode_step_dist().await;
        let token = sampler.sample(&dist.ids, &dist.probs);
        ctx.fill_token(token);
        generated_token_ids.push(token);

        if generated_token_ids.len() >= max_tokens {
            break;
        }
        if let Some(seq) = eos_sequences
            .iter()
            .find(|seq| generated_token_ids.ends_with(seq))
        {
            generated_token_ids.truncate(generated_token_ids.len() - seq.len());
            break;
        }
    }
    tokenizer.detokenize(&generated_token_ids)
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "an AI-powered code editor".to_string());
    let max_retries: u32 = args.value_from_str(["-r", "--max-retries"]).unwrap_or(3);
    let max_tokens: usize = args.value_from_str(["-t", "--max-tokens"]).unwrap_or(1024);

    let validator = SchemaValidator::new(PRODUCT_SCHEMA);
    let renderer = TemplateRenderer::new("announcement", TEMPLATE);

    let model = inferlet::get_auto_model();
    let model_name = model.get_name();
    let eos_sequences = model.eos_tokens();
    let tokenizer = model.get_tokenizer();

    let escape_non_printable = model_name.starts_with("Qwen/Qwen3")
        || model_name.starts_with("deepseek-r1-distill-qwen-2");

    let mut eot_token_id = None;
    for eot_tokens in &eos_sequences {
        if eot_tokens.len() == 1 {
            eot_token_id = Some(eot_tokens[0]);
            break;
        }
    }
    let eot_token_id =
        eot_token_id.ok_or_else(|| anyhow!("No single EOS token found for model"))?;

    let mut ctx = model.create_context();

    ctx.fill_system(SYSTEM_PROMPT);
    ctx.fill_user(&build_initial_prompt(&prompt, PRODUCT_SCHEMA));

    if model_name.starts_with("llama-3") {
        ctx.fill("\n\n");
    } else if model_name.starts_with("Qwen/Qwen3") {
        ctx.fill("\n\n<think></think>\n\n");
    } else if model_name.starts_with("deepseek-r1-distill-qwen-2") {
        ctx.fill("\n</think>\n\n");
    }

    let mut rendered_result = None;

    for attempt in 1..=max_retries {
        println!("--- Attempt {}/{} ---", attempt, max_retries);

        let constrained_sampler = ConstrainedSampler::new(
            tokenizer.get_vocabs(),
            tokenizer.get_special_tokens(),
            tokenizer.get_split_regex(),
            JSON_GRAMMAR.to_string(),
            eot_token_id,
            escape_non_printable,
        );

        let output = generate_json(
            &mut ctx,
            &constrained_sampler,
            &tokenizer,
            &eos_sequences,
            max_tokens,
        )
        .await;

        println!("Raw JSON output: {}", output);

        match validator.validate(&output) {
            Ok(parsed) => {
                println!("Schema validation passed!");
                match renderer.render(&parsed) {
                    Ok(rendered) => {
                        println!("Template rendered successfully!");
                        rendered_result = Some(rendered);
                        break;
                    }
                    Err(e) => {
                        println!("{}", e);
                        ctx.fill_user(&build_retry_prompt(&e));
                    }
                }
            }
            Err(error_report) => {
                println!("Validation errors:\n{}", error_report);
                ctx.fill_user(&build_retry_prompt(&error_report));
            }
        }

        if model_name.starts_with("llama-3") {
            ctx.fill("\n\n");
        } else if model_name.starts_with("Qwen/Qwen3") {
            ctx.fill("\n\n<think></think>\n\n");
        } else if model_name.starts_with("deepseek-r1-distill-qwen-2") {
            ctx.fill("\n</think>\n\n");
        }
    }

    println!("\n--- Result ---");
    if let Some(rendered) = rendered_result {
        println!("{}", rendered);
    } else {
        println!(
            "Failed to produce valid renderable JSON after {} attempts.",
            max_retries
        );
    }

    Ok(())
}
