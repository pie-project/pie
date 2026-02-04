//! Hello World Inferlet - Autoregressive Text Generation
//!
//! A simple example that demonstrates autoregressive decoding using the ContextExt API.

use inferlet::{
    context::Context,
    inference::Sampler,
    model::Model,
    runtime,
    ContextExt,
    Result,
};

/// Maximum number of tokens to generate
const MAX_TOKENS: usize = 100;

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    // Get the prompt from args, default to a simple greeting
    let prompt = args.get(0).cloned().unwrap_or_else(|| "Hello, world!".to_string());
    
    // Get the first available model
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    
    // Create the model
    let model = Model::load(model_name)?;
    
    // Create a context, fill with prompt, and generate
    let context = Context::new(&model)?;
    context.fill_tokens(&model.tokenizer().encode(&prompt));
    
    // Generate up to MAX_TOKENS and decode to string
    let generated = context.generate(Sampler::Multinomial((0.7, 42)))
        .collect_text()
        .await?;
    
    // Combine prompt with generated text
    let output = format!("{}{}", prompt, generated);
    
    // Clean up
    let _ = context.destroy();
    
    Ok(output)
}
