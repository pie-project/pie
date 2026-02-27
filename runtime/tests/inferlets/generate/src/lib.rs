//! Generate test inferlet — exercises the full forward pass pipeline.
//!
//! This inferlet tests: fill_tokens → flush → generate (step loop).
//! Skips chat template rendering to work with mock backends.

use inferlet::{
    context::Context,
    inference::Sampler,
    model::Model,
    runtime,
    ContextExt,
    Result,
};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    // Load the first available model
    let models = runtime::models();
    let model = Model::load(&models[0])?;
    let tokenizer = model.tokenizer();

    // Create a context — skip chat template, fill tokens directly
    let context = Context::new(&model)?;

    // Encode a test prompt and fill it directly (no chat template needed)
    let prompt_tokens = tokenizer.encode("hello world");
    eprintln!("[GENERATE] encoded prompt: {} tokens", prompt_tokens.len());

    context.fill_tokens(&prompt_tokens);

    // Flush the prompt tokens into the KV cache
    context.flush().await?;
    eprintln!("[GENERATE] flush done");

    // Generate tokens with a small limit
    let max_tokens: usize = 5;
    let mut stream = context
        .generate(Sampler::TopK((0.0, 1)))
        .with_max_tokens(max_tokens);

    let mut generated = Vec::new();
    while let Some(tokens) = stream.next().await? {
        eprintln!("[GENERATE] got tokens: {:?}", tokens);
        generated.extend(tokens);
    }

    let result = format!("generated {} tokens: {:?}", generated.len(), generated);
    eprintln!("[GENERATE] {}", result);
    Ok(result)
}
