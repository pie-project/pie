use inferlet::traits::Tokenize;
use inferlet::{Sampler, set_return};
use pico_args::Arguments;
use std::ffi::OsString;
use std::time::Instant;

/// Defines the command-line interface and help message.
const HELP: &str = r#"
Usage: program [OPTIONS]

A simple inferlet to run a chat model.

Options:
  -p, --prompt <STRING>    The prompt to send to the model
                           (default: "Explain the LLM decoding process ELI5.")
  -n, --max-tokens <INT>   The maximum number of new tokens to generate
                           (default: 256)
  -h, --help               Print help information
"#;

#[inferlet::main]
async fn main() -> Result<(), String> {
    // 1. Get arguments from the inferlet environment and prepare the parser.
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    // 2. Handle the --help flag.
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    // 3. Parse arguments, falling back to defaults if they are not provided.
    let prompt = args
        .opt_value_from_str(["-p", "--prompt"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "Explain the LLM decoding process ELI5.".to_string());

    let max_num_outputs: u32 = args
        .opt_value_from_str(["-n", "--max-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(256);

    // Ensure no unknown arguments were passed.
    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(format!(
            "Unknown arguments found: {:?}. Use --help for usage.",
            remaining
        ));
    }

    // --- Main logic starts here ---
    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();
    let mut ctx = model.create_context();

    // ctx.fill_system("You are a helpful, respectful and honest assistant.");
    // ctx.fill_user(&prompt);

    ctx.fill("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nReasoning: medium\n<|end|><|start|>user<|message|>Where is the capital of France?<|end|><|start|>assistant");
    
    let final_text = ctx
        .generate_until(Sampler::top_p(0.6, 0.95), max_num_outputs as usize)
        .await;

    let token_ids = tokenizer.tokenize(&final_text);
    println!(
        "Output: {:?} (total elapsed: {:?})",
        final_text,
        start.elapsed()
    );

    // Compute per-token latency, avoiding division by zero.
    if !token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (token_ids.len() as u32)
        );
    }
    set_return(&final_text);

    Ok(())
}
