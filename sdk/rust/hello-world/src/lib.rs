//! Hello World Inferlet - Autoregressive Text Generation
//!
//! A simple example that demonstrates autoregressive decoding using the Pie APIs.

use anyhow::Result;
use inferlet::{
    context::Context,
    inference::{self, ForwardPass, Output, Sampler},
    model::Model,
    runtime,
};

/// Maximum number of tokens to generate
const MAX_TOKENS: usize = 100;

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    // Get the prompt from args, default to a simple greeting
    let prompt = args.get(0).cloned().unwrap_or_else(|| "Hello, world!".to_string());
    
    // Get the first available model
    let models = runtime::models();
    let model_name = models.first().ok_or_else(|| anyhow::anyhow!("No models available"))?;
    
    // Create the model and get its tokenizer
    let model = Model::new(model_name);
    let tokenizer = model.tokenizer();
    
    // Encode the prompt
    let mut tokens: Vec<u32> = tokenizer.encode(&prompt);
    let prompt_len = tokens.len();
    
    // Create a context to hold the KV cache
    let context = Context::create(&model, "hello-world-ctx", Some(&tokens))
        .map_err(|e| anyhow::anyhow!("Failed to create context: {}", e))?;
    
    // Get stop tokens
    let stop_tokens: Vec<String> = model.stop_tokens();
    let stop_token_ids: Vec<u32> = stop_tokens
        .iter()
        .flat_map(|s| tokenizer.encode(s))
        .collect();
    
    // Autoregressive decoding loop
    for i in 0..MAX_TOKENS {
        let current_pos = (prompt_len + i) as u32;
        
        // Create a forward pass
        let pass = ForwardPass::new(&model);
        
        // Set the context for KV cache
        pass.context(&context);
        
        // For the first iteration, input all prompt tokens
        // For subsequent iterations, only input the last generated token
        if i == 0 {
            let positions: Vec<u32> = (0..tokens.len() as u32).collect();
            pass.input_tokens(&tokens, &positions);
        } else {
            let last_token = *tokens.last().unwrap();
            pass.input_tokens(&[last_token], &[current_pos - 1]);
        }
        
        // Configure sampling: multinomial with temperature 0.7
        pass.sampler(&[0], Sampler::Multinomial((0.7, 42)));
        
        // Execute the forward pass and wait for result asynchronously
        let output = inference::execute(&pass).await
            .map_err(|e| anyhow::anyhow!("Forward pass failed: {}", e))?;
        
        match output {
            Output::Tokens(new_tokens) => {
                if let Some(&new_token) = new_tokens.first() {
                    // Check for stop token
                    if stop_token_ids.contains(&new_token) {
                        break;
                    }
                    tokens.push(new_token);
                }
            }
            Output::None => break,
            _ => {}
        }
        
        // Check if we hit a stop token
        if let Some(&last) = tokens.last() {
            if stop_token_ids.contains(&last) {
                // Remove the stop token from output
                tokens.pop();
                break;
            }
        }
    }
    
    // Decode the generated tokens back to text
    let output_text = tokenizer.decode(&tokens)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
    
    // Clean up
    let _ = context.destroy();
    
    Ok(output_text)
}
