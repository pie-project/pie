//! Fill trait - extension for filling context with prompt tokens.

use crate::context::Context;
use crate::inference::ForwardPass;
use crate::ForwardPassExt;
use anyhow::{anyhow, Result};
use serde::Serialize;
use serde_json::Value;

/// Represents a single tool call.
#[derive(Serialize, Clone, Debug)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Represents a message in the conversation for template rendering.
#[derive(Serialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Renders messages using a minijinja chat template.
pub fn render_template(
    template: &str,
    messages: &[Message],
    add_generation_prompt: bool,
    begin_of_sequence: bool,
) -> String {
    minijinja::render!(
        template,
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        begin_of_sequence => begin_of_sequence,
    )
}

/// Extension trait for filling a context with tokens.
/// 
/// Provides a fluent API for building up context with chat messages.
/// Tokens are accumulated until `flush()` is called.
pub trait Fill {
    /// Adds a message with the specified role to the context.
    fn fill_template(&self, role: &str, message: &str) -> &Self;
    
    /// Adds a user message to the context.
    fn user(&self, message: &str) -> &Self {
        self.fill_template("user", message)
    }
    
    /// Adds a system message to the context.
    fn system(&self, message: &str) -> &Self {
        self.fill_template("system", message)
    }
    
    /// Adds an assistant message to the context.
    fn assistant(&self, message: &str) -> &Self {
        self.fill_template("assistant", message)
    }
    
    /// Adds a tool message to the context.
    fn tool(&self, message: &str) -> &Self {
        self.fill_template("tool", message)
    }
    
    /// Fills the context with raw token IDs (appends to buffered tokens).
    fn fill_tokens(&self, tokens: &[u32]) -> &Self;
    
    /// Flushes buffered tokens: executes forward pass and commits pages.
    fn flush(&self) -> impl std::future::Future<Output = Result<()>>;
}

impl Fill for Context {
    fn fill_template(&self, role: &str, message: &str) -> &Self {
        let model = self.model();
        let template = model.prompt_template();
        let tokenizer = model.tokenizer();
        
        // Create a message with the specified role
        let msg = Message {
            role: role.to_string(),
            content: message.to_string(),
            reasoning_content: None,
            tool_calls: None,
        };
        
        // Render using the model's chat template
        let formatted = render_template(&template, &[msg], false, true);
        let tokens = tokenizer.encode(&formatted);
        
        // Use fill_tokens to append
        self.fill_tokens(&tokens)
    }
    
    fn fill_tokens(&self, tokens: &[u32]) -> &Self {
        // Use append_buffered_tokens to add to existing buffered tokens
        self.append_buffered_tokens(tokens);
        self
    }
    
    async fn flush(&self) -> Result<()> {
        let model = self.model();
        
        // Get buffered tokens
        let buffered = self.buffered_tokens();
        
        // Flush only effective when buffered token length > 1
        // (need at least one token left for generate())
        if buffered.len() <= 1 {
            return Ok(());
        }
        
        // Take all but the last token for the forward pass
        let tokens_to_flush = &buffered[..buffered.len() - 1];
        let last_token = buffered[buffered.len() - 1];
        let num_tokens = tokens_to_flush.len() as u32;
        
        // Check available capacity
        let page_size = self.tokens_per_page();
        let uncommitted_pages = self.uncommitted_page_count();
        let cursor = self.cursor();
        let available_capacity = page_size * uncommitted_pages - cursor;
        
        // Reserve more pages if needed
        if available_capacity < num_tokens {
            let tokens_needed = num_tokens - available_capacity;
            let pages_needed = (tokens_needed + page_size - 1) / page_size;
            self.reserve_pages(pages_needed)
                .map_err(|e| anyhow!("Failed to reserve pages: {}", e))?;
        }
        
        // Create forward pass with N-1 tokens
        let pass = ForwardPass::new(&model);
        pass.context(self);
        
        // Calculate positions for each token (starting from cursor)
        let positions: Vec<u32> = (cursor..cursor + num_tokens).collect();
        pass.input_tokens(tokens_to_flush, &positions);
        
        // Execute the forward pass (fills KV cache)
        pass.execute_async().await
            .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
        
        // Calculate new cursor position after flush
        let new_cursor_abs = cursor + num_tokens;
        
        // Commit fully populated pages (pages fully covered by flushed tokens)
        // A page is fully populated if all its tokens are flushed
        let pages_to_commit = new_cursor_abs / page_size;
        if pages_to_commit > 0 {
            let page_indices: Vec<u32> = (0..pages_to_commit).collect();
            self.commit_pages(&page_indices)
                .map_err(|e| anyhow!("Failed to commit pages: {}", e))?;
        }
        
        // Adjust cursor - after committing, cursor is within the remaining uncommitted page
        // cursor should be: new_cursor_abs % page_size
        let new_cursor = new_cursor_abs % page_size;
        self.set_cursor(new_cursor);
        
        // Keep only the last token in buffered tokens
        self.set_buffered_tokens(&[last_token]);
        
        Ok(())
    }
}
