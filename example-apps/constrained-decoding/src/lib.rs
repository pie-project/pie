//! Demonstrates grammar-constrained decoding using a Lark grammar.
//!
//! This example uses the `llguidance` library to constrain model outputs to
//! valid structured formats (e.g., JSON) by masking invalid tokens during
//! sampling.

mod sampler;

use inferlet::sampler::Sampler;
use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Result, anyhow};
use sampler::ConstrainedSampler;
use std::time::Instant;

/// Defines the command-line interface and help message.
const HELP: &str = r#"
Usage: program [OPTIONS]

A constrained sampler inferlet using a user-provided grammar.

Options:
  -p, --prompt <STRING>    The prompt to send to the model.
                           (default: "what is the capital of France? [...]")
  -g, --grammar <STRING>   The Lark grammar to constrain the output.
                           (default: A JSON grammar)
  -n, --max-tokens <INT>   The maximum number of new tokens to generate.
                           (default: 128)
  -h, --help               Print help information.
"#;

/// Default grammar to use if none is provided via command-line arguments.
const JSON_GRAMMAR: &str = r##"
?start: value
?value: object
        | array
        | string
        | SIGNED_NUMBER      -> number
        | "true"             -> true
        | "false"            -> false
        | "null"             -> null
array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : string ":" value
string : ESCAPED_STRING
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
"##;

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let prompt = args.value_from_str(["-p", "--prompt"]).unwrap_or_else(|_| {
        "Where is the capital of France? \
        Output in compact JSON text format and only the JSON object. \
        Don't include any additional whitespace, newlines, quotes, or comments."
            .to_string()
    });

    let grammar = args
        .value_from_str(["-g", "--grammar"])
        .unwrap_or_else(|_| JSON_GRAMMAR.to_string());

    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(128);

    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(anyhow!(
            "Unknown arguments found: {:?}. Use --help for usage.",
            remaining
        ));
    }

    let start = Instant::now();
    let model = inferlet::get_auto_model();

    // Restrict the example to only run on Llama 3, Qwen 3, and DeepSeek R1 Distill Qwen 2 models.
    // This is because we need to add some special prompts for these models to output the JSON
    // directly. See the `ctx.fill` calls below for more details.
    let model_name = model.get_name();
    if !(model_name.starts_with("llama-3")
        || model_name.starts_with("qwen-3")
        || model_name.starts_with("deepseek-r1-distill-qwen-2"))
    {
        return Err(anyhow!(
            "Constrained decoding example is only implemented for Llama 3 and Qwen 3. Got: {}",
            model_name
        ));
    }

    // Determine if we need to escape non-printable characters.
    // For example, Qwen 3 and DeepSeek R1 Distill Qwen 2 models will output "Ġ" for space, while
    // Llama 3 models will output " " for space.
    let escape_non_printable =
        model_name.starts_with("qwen-3") || model_name.starts_with("deepseek-r1-distill-qwen-2");

    // Find the EOS token ID for the model. This is used as a fallback when the grammar constraint
    // is not met. We need to find a single EOS token ID because the sampler outputs a single token
    // at a time.
    let mut eot_token_id = None;
    for eot_tokens in model.eos_tokens() {
        if eot_tokens.len() == 1 {
            eot_token_id = Some(eot_tokens[0]);
            break;
        }
    }
    let eot_token_id = eot_token_id
        .ok_or_else(|| anyhow!("No single EOS token found for model: {}", model_name))?;

    let tokenizer = model.get_tokenizer();
    let mut ctx = model.create_context();

    let sampler = Box::new(ConstrainedSampler::new(
        tokenizer.get_vocabs(),
        tokenizer.get_special_tokens(),
        tokenizer.get_split_regex(),
        grammar,
        eot_token_id,
        escape_non_printable,
    ));

    let sampler = Sampler::Custom {
        temperature: 0.0,
        sampler,
    };

    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    // Llama 3 models strongly prefer to output two newlines first. We put the newlines here
    // so that the model can output the JSON directly.
    if model_name.starts_with("llama-3") {
        ctx.fill("\n\n");
    // Qwen 3 models are thinking models. We put the <think> and </think> tags here so that
    // the model can output the JSON directly.
    } else if model_name.starts_with("qwen-3") {
        ctx.fill("\n\n<think></think>\n\n");
    // DeepSeek R1 Distill Qwen 2 models are thinking models. We put the </think> tag here so that
    // the model can output the JSON directly.
    } else if model_name.starts_with("deepseek-r1-distill-qwen-2") {
        ctx.fill("\n</think>\n\n");
    }

    let stop_cond =
        stop_condition::max_len(max_tokens).or(stop_condition::ends_with_any(model.eos_tokens()));

    let output = ctx.generate(sampler, stop_cond).await;
    let output_token_ids = tokenizer.tokenize(&output);

    println!("Output: {} (total elapsed: {:?})", output, start.elapsed());

    // Compute per-token latency, avoiding division by zero.
    if !output_token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (output_token_ids.len() as u32)
        );
    }

    Ok(())
}
