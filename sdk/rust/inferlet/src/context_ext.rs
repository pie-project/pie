//! Context extension trait for filling and generating.
//!
//! Provides high-level APIs for:
//! - Acquiring context lock asynchronously
//! - Filling context with prompt tokens
//! - Generating tokens with speculative decoding

use crate::context::Context;
use crate::model::Model;
use crate::inference::{ForwardPass, Output, Sampler};
use crate::ForwardPassExt;
use crate::Result;
use serde::Serialize;
use serde_json::Value;
use wstd::io::AsyncPollable;

/// Simple counter for generating unique context names.
static CONTEXT_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// =============================================================================
// Supporting Types
// =============================================================================

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

// =============================================================================
// Speculation Types
// =============================================================================

/// Trait for custom speculative decoding.
pub trait Speculate {
    /// Generates draft tokens and their positions based on current context.
    fn draft(&self) -> (Vec<u32>, Vec<u32>);
    
    /// Called with the accepted tokens from the model.
    fn accept(&mut self, tokens: &[u32]);
    
    /// Resets the speculator to its initial state.
    fn reset(&mut self);
    
    /// Rolls back the last `num_tokens` accepted tokens.
    fn rollback(&mut self, num_tokens: usize);
}



/// Trait for token sampling constraints (e.g., grammar-based token masking).
pub trait Constrain {
    /// Returns the logit mask as BRLE-encoded data.
    fn mask(&self) -> Vec<u32>;
    
    /// Called with the accepted tokens to update constraint state.
    fn accept(&mut self, tokens: &[u32]);
    
    /// Resets the constraint to its initial state.
    fn reset(&mut self);
    
    /// Rolls back the last `num_tokens` accepted tokens.
    fn rollback(&mut self, num_tokens: usize);
}

/// Speculation enum - either system-provided or custom.
pub enum Speculation {
    /// Default speculation that uses runtime-provided speculative tokens.
    Default {
        spec_tokens: Vec<u32>,
        spec_positions: Vec<u32>,
    },

    /// Custom speculation that implements the [Speculate] trait.
    Custom(Box<dyn Speculate>),
}

impl Default for Speculation {
    fn default() -> Self {
        Speculation::Default {
            spec_tokens: Vec::new(),
            spec_positions: Vec::new(),
        }
    }
}

impl Speculation {
    /// Creates a new system speculation.
    pub fn system() -> Self {
        Self::default()
    }

    /// Creates a custom speculation from a Speculate implementation.
    pub fn custom<S: Speculate + 'static>(speculator: S) -> Self {
        Speculation::Custom(Box::new(speculator))
    }

    fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
        match self {
            Speculation::Default { spec_tokens, spec_positions } => {
                let tokens = std::mem::take(spec_tokens);
                let positions = std::mem::take(spec_positions);
                (tokens, positions)
            }
            Speculation::Custom(s) => s.draft(),
        }
    }

    fn accept(&mut self, output: Output) -> Vec<u32> {
        match self {
            Speculation::Default { spec_tokens, spec_positions } => {
                match output {
                    Output::TokensWithSpeculation((accepted, next_spec, next_pos)) => {
                        *spec_tokens = next_spec;
                        *spec_positions = next_pos;
                        accepted
                    }
                    Output::Tokens(tokens) => tokens,
                    _ => vec![],
                }
            }
            Speculation::Custom(s) => {
                let tokens = match output {
                    Output::Tokens(tokens) => tokens,
                    Output::TokensWithSpeculation((accepted, _, _)) => accepted,
                    _ => vec![],
                };
                s.accept(&tokens);
                tokens
            }
        }
    }
}

// =============================================================================
// TokenStream
// =============================================================================

/// Async stream of generated tokens.
pub struct TokenStream<'a> {
    ctx: &'a Context,
    model: crate::model::Model,
    page_size: u32,
    stop_tokens: Vec<u32>,
    sampler: Sampler,
    speculation: Speculation,
    constraint: Option<Box<dyn Constrain>>,
    pending_tokens: Vec<u32>,
    done: bool,
    max_tokens: Option<usize>,
    tokens_generated: usize,
}

impl<'a> TokenStream<'a> {
    
    /// Creates a new token stream with default (system) speculation.
    pub fn new(ctx: &'a Context, sampler: Sampler) -> Self {
        let model = ctx.model();
        let page_size = ctx.tokens_per_page();
        let stop_tokens = model.tokenizer().stop_tokens();
        Self {
            ctx,
            model,
            page_size,
            stop_tokens,
            sampler,
            speculation: Speculation::system(),
            constraint: None,
            pending_tokens: Vec::new(),
            done: false,
            max_tokens: None,
            tokens_generated: 0,
        }
    }

    /// Sets a custom speculation strategy for speculative decoding.
    pub fn with_speculation(mut self, speculation: Speculation) -> Self {
        self.speculation = speculation;
        self
    }

    /// Sets a sampling constraint for logit masking.
    pub fn with_constraint<C: Constrain + 'static>(mut self, constraint: C) -> Self {
        self.constraint = Some(Box::new(constraint));
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
    /// Gets the next token from the stream.
    pub async fn next(&mut self) -> Result<Option<u32>> {
        // Check max tokens limit
        if let Some(max) = self.max_tokens {
            if self.tokens_generated >= max {
                return Ok(None);
            }
        }

        if let Some(token) = self.pending_tokens.pop() {
            self.tokens_generated += 1;
            return Ok(Some(token));
        }

        if self.done {
            return Ok(None);
        }

        let tokens = self.step().await?;
        let first = tokens[0];
        
        // Check if first token is a stop token
        if self.stop_tokens.contains(&first) {
            self.done = true;
            return Ok(None);
        }

        if tokens.len() > 1 {
            self.pending_tokens = tokens[1..].iter().copied().rev().collect();
        }

        Ok(Some(tokens[0]))
    }

    /// Collects all tokens from the stream (until stop token or max_tokens limit).
    pub async fn collect_tokens(mut self) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        while let Some(token) = self.next().await? {
            tokens.push(token);
        }
        Ok(tokens)
    }

    /// Collects all tokens and decodes them to text.
    pub async fn collect_text(mut self) -> Result<String> {
        let mut tokens = Vec::new();
        while let Some(token) = self.next().await? {
            tokens.push(token);
        }
        let tokenizer = self.model.tokenizer();
        tokenizer.decode(&tokens)
    }

    async fn step(&mut self) -> Result<Vec<u32>> {
        let buffered = self.ctx.buffered_tokens();
        if buffered.is_empty() {
            return Err("generate requires at least one buffered token".to_string());
        }
        
        let cursor = self.ctx.cursor();
        
        let pass = ForwardPass::new(&self.model);
        pass.context(self.ctx);
        
        let positions: Vec<u32> = (cursor..cursor + buffered.len() as u32).collect();
        pass.input_tokens(&buffered, &positions);
        
        let (draft_tokens, draft_positions) = self.speculation.draft();
        if !draft_tokens.is_empty() {
            pass.input_speculative_tokens(&draft_tokens, &draft_positions);
        }
        
        if matches!(self.speculation, Speculation::Default { .. }) {
            pass.output_speculative_tokens(true);
        }

        let last_token_idx = (buffered.len() - 1) as u32;
        pass.sampler(&[last_token_idx], self.sampler.clone());
        
        // Apply logit mask if constraint is available
        if let Some(ref constraint) = self.constraint {
            pass.logit_mask(&constraint.mask());
        }
        
        let output = pass.execute_async().await?;
        let new_tokens = self.speculation.accept(output);
        
        // Update constraint with accepted tokens
        if let Some(ref mut constraint) = self.constraint {
            constraint.accept(&new_tokens);
        }
        
        if new_tokens.is_empty() {
            return Ok(vec![]);
        }

        let new_cursor = cursor + buffered.len() as u32;
        let pages_to_commit = new_cursor / self.page_size;
        
        if pages_to_commit > 0 {
            let page_indices: Vec<u32> = (0..pages_to_commit).collect();
            self.ctx.commit_pages(&page_indices)
                .map_err(|e| format!("Failed to commit pages: {}", e))?;
        }
        
        self.ctx.set_cursor(new_cursor % self.page_size);
        
        if let Some(&last_token) = new_tokens.last() {
            self.ctx.set_buffered_tokens(&[last_token]);
        }
        
        Ok(new_tokens)
    }
}

// =============================================================================
// ContextExt Trait (Consolidated)
// =============================================================================

/// Extension trait for Context - provides async operations, filling, and generation.
pub trait ContextExt {
    // --- Creation ---
    
    /// Creates a new context with an auto-generated globally unique name.
    fn new(model: &Model) -> Result<Context> {
        // Combine counter with address-based entropy for global uniqueness
        let counter = CONTEXT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Use stack address as additional entropy source
        let stack_addr = &counter as *const _ as u64;
        // Simple mixing function
        let mixed = counter.wrapping_mul(0x517cc1b727220a95) ^ stack_addr;
        let name = format!("ctx-{:016x}", mixed);
        Context::create(model, &name, None)
    }
    
    // --- Async Operations ---
    
    /// Acquires a lock on the context asynchronously.
    fn acquire_lock_async(&self) -> impl std::future::Future<Output = bool>;
    
    // --- Fill Operations ---
    
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
    
    // --- Generate Operations ---
    
    /// Creates a token stream for generation.
    fn generate(&self, sampler: Sampler) -> TokenStream<'_>;
}

// =============================================================================
// Implementation for Context
// =============================================================================

impl ContextExt for Context {
    // --- Async Operations ---
    
    async fn acquire_lock_async(&self) -> bool {
        let future = self.acquire_lock();
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().unwrap_or(false)
    }
    
    // --- Fill Operations ---
    
    fn fill_template(&self, role: &str, message: &str) -> &Self {
        let model = self.model();
        let template = model.chat_template();
        let tokenizer = model.tokenizer();
        
        let msg = Message {
            role: role.to_string(),
            content: message.to_string(),
            reasoning_content: None,
            tool_calls: None,
        };
        
        let formatted = render_template(&template, &[msg], false, true);
        let tokens = tokenizer.encode(&formatted);
        
        self.fill_tokens(&tokens)
    }
    
    fn fill_tokens(&self, tokens: &[u32]) -> &Self {
        self.append_buffered_tokens(tokens);
        self
    }
    
    async fn flush(&self) -> Result<()> {
        let model = self.model();
        let buffered = self.buffered_tokens();
        
        if buffered.len() <= 1 {
            return Ok(());
        }
        
        let tokens_to_flush = &buffered[..buffered.len() - 1];
        let last_token = buffered[buffered.len() - 1];
        let num_tokens = tokens_to_flush.len() as u32;
        
        let page_size = self.tokens_per_page();
        let uncommitted_pages = self.uncommitted_page_count();
        let cursor = self.cursor();
        let available_capacity = page_size * uncommitted_pages - cursor;
        
        if available_capacity < num_tokens {
            let tokens_needed = num_tokens - available_capacity;
            let pages_needed = (tokens_needed + page_size - 1) / page_size;
            self.reserve_pages(pages_needed)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
        }
        
        let pass = ForwardPass::new(&model);
        pass.context(self);
        
        let positions: Vec<u32> = (cursor..cursor + num_tokens).collect();
        pass.input_tokens(tokens_to_flush, &positions);
        
        pass.execute_async().await
            .map_err(|e| format!("Forward pass failed: {}", e))?;
        
        let new_cursor_abs = cursor + num_tokens;
        let pages_to_commit = new_cursor_abs / page_size;
        if pages_to_commit > 0 {
            let page_indices: Vec<u32> = (0..pages_to_commit).collect();
            self.commit_pages(&page_indices)
                .map_err(|e| format!("Failed to commit pages: {}", e))?;
        }
        
        let new_cursor = new_cursor_abs % page_size;
        self.set_cursor(new_cursor);
        self.set_buffered_tokens(&[last_token]);
        
        Ok(())
    }
    
    // --- Generate Operations ---
    
    fn generate(&self, sampler: Sampler) -> TokenStream<'_> {
        TokenStream::new(self, sampler)
    }
}
