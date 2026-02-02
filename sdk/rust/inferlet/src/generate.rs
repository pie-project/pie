//! Generate trait - extension for autoregressive text generation.

use crate::context::Context;
use crate::inference::{ForwardPass, Output, Sampler};
use crate::ForwardPassExt;

/// Extension trait for autoregressive text generation.
pub trait Generate {
    /// Generates tokens autoregressively until a stop condition is met.
    fn generate(&self, max_tokens: usize) -> impl std::future::Future<Output = Result<String, String>>;
}

impl Generate for Context {
    async fn generate(&self, max_tokens: usize) -> Result<String, String> {
        let model = self.model();
        let tokenizer = model.tokenizer();
        
        let stop_tokens: Vec<String> = model.stop_tokens();
        let stop_token_ids: Vec<u32> = stop_tokens
            .iter()
            .flat_map(|s| tokenizer.encode(s))
            .collect();
        
        let mut generated_tokens: Vec<u32> = Vec::new();
        let start_pos = self.cursor();
        
        for i in 0..max_tokens {
            let current_pos = start_pos + i as u32;
            
            let pass = ForwardPass::new(&model);
            pass.context(self);
            
            // Input the last generated token (or nothing for first iteration)
            if let Some(&last_token) = generated_tokens.last() {
                pass.input_tokens(&[last_token], &[current_pos - 1]);
            }
            
            // Configure sampling
            pass.sampler(&[0], Sampler::Multinomial((0.7, 42)));
            
            let output = pass.execute_async().await?;
            
            match output {
                Output::Tokens(new_tokens) => {
                    if let Some(&new_token) = new_tokens.first() {
                        if stop_token_ids.contains(&new_token) {
                            break;
                        }
                        generated_tokens.push(new_token);
                    }
                }
                Output::None => break,
                _ => {}
            }
        }
        
        tokenizer.decode(&generated_tokens)
            .map_err(|e| format!("Decode failed: {}", e))
    }
}
