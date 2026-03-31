//! Demonstrates JSON Schema-validated generation with grammar-constrained decoding.
//!
//! This example combines two layers of validation:
//! 1. A Lark grammar via `llguidance` ensures every generated token produces
//!    syntactically valid JSON (no parse errors possible).
//! 2. The `jsonschema` library validates the JSON against a specific schema.
//!    When schema validation fails, the errors are fed back to the model and
//!    it regenerates -- still grammar-constrained.

mod sampler;
mod schema;

use inferlet::{Args, Result, anyhow};
use sampler::ConstrainedSampler;
use schema::SchemaValidator;

const HELP: &str = "\
Usage: json-schema-validation [OPTIONS]

Generates JSON from an LLM with grammar-constrained decoding, then
validates it against a JSON Schema. Retries with error feedback
until the output satisfies the schema.

Options:
  -p, --prompt <STRING>      The prompt describing what to generate
                             [default: a person profile]
  -r, --max-retries <N>      Maximum validation/retry cycles [default: 3]
  -t, --max-tokens <N>       Max tokens per generation attempt [default: 512]
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

const PERSON_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1
        },
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 150
        },
        "email": {
            "type": "string"
        },
        "skills": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 1
        },
        "address": {
            "type": "object",
            "properties": {
                "city": { "type": "string" },
                "country": { "type": "string" }
            },
            "required": ["city", "country"]
        }
    },
    "required": ["name", "age", "email", "skills", "address"]
}"#;

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant that generates structured data. When asked to produce \
JSON, you must output ONLY the raw JSON object with no additional text, markdown \
fences, or explanation. The JSON must be compact (no unnecessary whitespace). \
If you receive validation errors, fix exactly those issues and output the corrected \
JSON object.";

fn build_initial_prompt(user_prompt: &str, schema: &str) -> String {
    format!(
        "{}\n\nThe output must conform to this JSON Schema:\n{}\n\n\
        Output only the JSON object, nothing else.",
        user_prompt, schema
    )
}

fn build_retry_prompt(errors: &str) -> String {
    format!(
        "The JSON you produced has schema validation errors:\n{}\n\n\
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

    let prompt: String = args.value_from_str(["-p", "--prompt"]).unwrap_or_else(|_| {
        "Generate a profile for a fictional software engineer named Alice.".to_string()
    });
    let max_retries: u32 = args.value_from_str(["-r", "--max-retries"]).unwrap_or(3);
    let max_tokens: usize = args.value_from_str(["-t", "--max-tokens"]).unwrap_or(512);

    let validator = SchemaValidator::new(PERSON_SCHEMA);

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
    ctx.fill_user(&build_initial_prompt(&prompt, PERSON_SCHEMA));

    // Skip thinking tokens for reasoning models so the constrained sampler
    // sees only the JSON output.
    if model_name.starts_with("llama-3") {
        ctx.fill("\n\n");
    } else if model_name.starts_with("Qwen/Qwen3") {
        ctx.fill("\n\n<think></think>\n\n");
    } else if model_name.starts_with("deepseek-r1-distill-qwen-2") {
        ctx.fill("\n</think>\n\n");
    }

    let mut valid_result = None;

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

        println!("Output: {}", output);

        match validator.validate(&output) {
            Ok(parsed) => {
                println!("Schema validation passed!");
                valid_result = Some(parsed);
                break;
            }
            Err(error_report) => {
                println!("Validation errors:\n{}", error_report);
                ctx.fill_user(&build_retry_prompt(&error_report));
            }
        }

        // Skip thinking tokens again for the retry generation.
        if model_name.starts_with("llama-3") {
            ctx.fill("\n\n");
        } else if model_name.starts_with("Qwen/Qwen3") {
            ctx.fill("\n\n<think></think>\n\n");
        } else if model_name.starts_with("deepseek-r1-distill-qwen-2") {
            ctx.fill("\n</think>\n\n");
        }
    }

    println!("\n--- Result ---");
    if let Some(result) = valid_result {
        println!(
            "Valid JSON:\n{}",
            serde_json::to_string_pretty(&result).unwrap()
        );
    } else {
        println!(
            "Failed to produce valid JSON after {} attempts.",
            max_retries
        );
    }

    Ok(())
}
