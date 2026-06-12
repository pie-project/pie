//! Generate test inferlet — exercises the full forward pass pipeline.
//!
//! This inferlet tests: fill_tokens → flush → generate (step loop).
//! Skips chat template rendering to work with mock backends.

use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    // Load the first available model
    let models = runtime::models();
    let model = Model::load(&models[0])?;
    let tokenizer = model.tokenizer();

    // Create a context — skip chat template, fill tokens directly
    let mut context = Context::new(&model)?;

    // Encode a test prompt and fill it directly (no chat template needed)
    let prompt_tokens = tokenizer.encode("hello world");
    eprintln!("[GENERATE] encoded prompt: {} tokens", prompt_tokens.len());

    context.append(&prompt_tokens);

    // Flush the prompt tokens into the KV cache
    context.flush().await?;
    eprintln!("[GENERATE] flush done");

    // Generate tokens with a small limit
    let max_tokens: usize = 5;
    let generated = context
        .generate(Sampler::top_k(0.0, 1))
        .max_tokens(max_tokens)
        .collect_tokens()
        .await?;

    let result = format!("generated {} tokens: {:?}", generated.len(), generated);
    eprintln!("[GENERATE] {}", result);
    Ok(result)
}
