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
use std::cell::RefCell;

/// Simple counter for generating unique context names.
static CONTEXT_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Pending system message buffered by `fill_sys`.
/// Drained by the next `fill_role` or `fill_user` call so that
/// system + first message are rendered together in a single template call.
thread_local! {
    static PENDING_SYSTEM: RefCell<Option<Message>> = RefCell::new(None);
}


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

/// Re-export the WIT ChatTemplate and SystemHandling types.
pub use crate::pie::core::model::{ChatTemplate, SystemHandling};

/// Look up the role prefix for a given role in a ChatTemplate.
fn prefix_for<'a>(ct: &'a ChatTemplate, role: &str) -> &'a str {
    ct.role_prefixes
        .iter()
        .find(|(r, _)| r == role)
        .map(|(_, p)| p.as_str())
        .unwrap_or("")
}

/// Look up the role suffix for a given role in a ChatTemplate.
fn suffix_for<'a>(ct: &'a ChatTemplate, role: &str) -> &'a str {
    ct.role_suffixes
        .iter()
        .find(|(r, _)| r == role)
        .map(|(_, s)| s.as_str())
        .unwrap_or("")
}

/// Renders a conversation using a structured ChatTemplate.
///
/// Each message is rendered as: prefix(role) + content + suffix(role).
/// System messages are handled according to `system_handling`:
///   - Standalone: rendered as a regular turn with prefix/suffix.
///   - MergeWithUser: content prepended to first user message.
///   - BarePrepend: content placed raw before all turns.
pub fn render_template(
    ct: &ChatTemplate,
    messages: &[Message],
    add_generation_prompt: bool,
    begin_of_sequence: bool,
) -> String {
    let mut out = String::new();

    if begin_of_sequence {
        out.push_str(&ct.start_token);
    }

    // Collect system content (for merge/bare-prepend modes).
    let system_content: Option<&str> = match ct.system_handling {
        SystemHandling::Standalone => None,
        _ => messages.iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.as_str()),
    };

    // For bare-prepend, output system content immediately.
    if ct.system_handling == SystemHandling::BarePrepend {
        if let Some(sys) = system_content {
            out.push_str(sys);
        }
    }

    let mut first_user = true;

    for msg in messages {
        if msg.role == "system" {
            match ct.system_handling {
                SystemHandling::Standalone => {
                    out.push_str(prefix_for(ct, "system"));
                    out.push_str(&msg.content);
                    out.push_str(suffix_for(ct, "system"));
                }
                _ => { /* handled via merge or bare-prepend */ }
            }
            continue;
        }

        let role = msg.role.as_str();
        out.push_str(prefix_for(ct, role));

        // Merge system content into first user message if needed.
        if role == "user" && first_user && ct.system_handling == SystemHandling::MergeWithUser {
            if let Some(sys) = system_content {
                out.push_str(sys);
                out.push_str(&ct.system_separator);
            }
            first_user = false;
        }

        // Render reasoning content if present.
        if let Some(ref reasoning) = msg.reasoning_content {
            if !ct.thinking_prefix.is_empty() {
                out.push_str(&ct.thinking_prefix);
                out.push_str(reasoning.trim());
                out.push_str(&ct.thinking_suffix);
            }
        }

        // Render tool calls if present.
        if let Some(ref tool_calls) = msg.tool_calls {
            if !ct.tool_calls_prefix.is_empty() {
                out.push_str(&ct.tool_calls_prefix);
            }
            for tc in tool_calls {
                let rendered = ct.tool_call_template
                    .replace("{name}", &tc.name)
                    .replace("{arguments}", &tc.arguments.to_string());
                out.push_str(&rendered);
            }
            if !ct.tool_calls_suffix.is_empty() {
                out.push_str(&ct.tool_calls_suffix);
            }
        } else {
            out.push_str(&msg.content);
        }

        out.push_str(suffix_for(ct, role));
    }

    if add_generation_prompt {
        out.push_str(&ct.generation_header);
    }

    out
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
        let ct = model.chat_template();
        let tokenizer = model.tokenizer();
        let stop_tokens: Vec<u32> = ct.stop_tokens
            .iter()
            .filter_map(|s| tokenizer.encode(s).into_iter().next())
            .collect();
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

        self.tokens_generated += 1;
        Ok(Some(first))
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
        
        let seq_len = self.ctx.last_position().map(|p| p + 1).unwrap_or(0);
        let cursor = self.ctx.cursor();
        
        // Reserve pages for the new token(s)
        let total_tokens_after = cursor + buffered.len() as u32;
        let total_pages_needed = (total_tokens_after + self.page_size - 1) / self.page_size;
        if total_pages_needed > 0 {
            self.ctx.reserve_pages(total_pages_needed)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
        }
        
        let pass = ForwardPass::new(&self.model);
        pass.context(self.ctx);
        
        let positions: Vec<u32> = (seq_len..seq_len + buffered.len() as u32).collect();
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

        // Use the cursor saved BEFORE execute_async, since fill() already
        // moved buffered tokens to tokens_filled (incrementing cursor).
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
    /// Renders with `add_generation_prompt=false` — use for conversation history replay.
    fn fill_role(&self, role: &str, message: &str) -> &Self;
    
    /// Adds a user message and the generation prompt (e.g. `<|im_start|>assistant\n`).
    /// Use this as the **last** fill call before `generate()`.
    fn fill_user(&self, message: &str) -> &Self;
    
    /// Buffers a system message to be rendered together with the next message.
    /// This ensures correct output for templates that extract system content
    /// (e.g. Gemma2 prepends system to first user via `loop.first`).
    fn fill_sys(&self, message: &str) -> &Self;
    
    /// Adds an assistant message to the context (no generation prompt).
    fn fill_assistant(&self, message: &str) -> &Self {
        self.fill_role("assistant", message)
    }
    
    /// Adds a tool message to the context (no generation prompt).
    fn fill_tool(&self, message: &str) -> &Self {
        self.fill_role("tool", message)
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
    
    fn fill_sys(&self, message: &str) -> &Self {
        let msg = Message {
            role: "system".to_string(),
            content: message.to_string(),
            reasoning_content: None,
            tool_calls: None,
        };
        PENDING_SYSTEM.with(|cell| {
            *cell.borrow_mut() = Some(msg);
        });
        self
    }
    
    fn fill_role(&self, role: &str, message: &str) -> &Self {
        let model = self.model();
        let ct = model.chat_template();
        let tokenizer = model.tokenizer();
        
        let msg = Message {
            role: role.to_string(),
            content: message.to_string(),
            reasoning_content: None,
            tool_calls: None,
        };
        let bos = self.buffered_tokens().is_empty();
        // If there's a pending system message, batch it with this message.
        let pending_sys = PENDING_SYSTEM.with(|cell| cell.borrow_mut().take());
        let messages: Vec<Message> = match pending_sys {
            Some(sys) => vec![sys, msg],
            None => vec![msg],
        };
        let formatted = render_template(&ct, &messages, false, bos);
        let tokens = tokenizer.encode(&formatted);
        self.append_buffered_tokens(&tokens);
        self
    }
    
    fn fill_user(&self, message: &str) -> &Self {
        let model = self.model();
        let ct = model.chat_template();
        let tokenizer = model.tokenizer();
        
        let msg = Message {
            role: "user".to_string(),
            content: message.to_string(),
            reasoning_content: None,
            tool_calls: None,
        };
        let bos = self.buffered_tokens().is_empty();
        // If there's a pending system message, batch it with this message.
        let pending_sys = PENDING_SYSTEM.with(|cell| cell.borrow_mut().take());
        let messages: Vec<Message> = match pending_sys {
            Some(sys) => vec![sys, msg],
            None => vec![msg],
        };
        let formatted = render_template(&ct, &messages, true, bos);
        let tokens = tokenizer.encode(&formatted);
        self.append_buffered_tokens(&tokens);
        self
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
        let seq_len = self.last_position().map(|p| p + 1).unwrap_or(0);
        
        // Calculate total pages needed to hold cursor + tokens_to_flush
        let cursor = self.cursor();
        let total_tokens_after = cursor + num_tokens;
        let total_pages_needed = (total_tokens_after + page_size - 1) / page_size;
        
        // Always reserve the pages we need — the host will handle dedup
        if total_pages_needed > 0 {
            self.reserve_pages(total_pages_needed)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
        }
        
        let pass = ForwardPass::new(&model);
        pass.context(self);
        
        let positions: Vec<u32> = (seq_len..seq_len + num_tokens).collect();
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> Message {
        Message {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
        }
    }

    fn qwen3() -> ChatTemplate {
        ChatTemplate {
            start_token: "".into(),
            stop_tokens: vec!["<|im_end|>".into(), "<|im_start|>".into(), "<|endoftext|>".into()],
            role_prefixes: vec![("system".into(), "<|im_start|>system
".into()), ("user".into(), "<|im_start|>user
".into()), ("assistant".into(), "<|im_start|>assistant
".into())],
            role_suffixes: vec![("system".into(), "<|im_end|>
".into()), ("user".into(), "<|im_end|>
".into()), ("assistant".into(), "<|im_end|>
".into())],
            system_handling: SystemHandling::Standalone,
            system_separator: "".into(),
            generation_header: "<|im_start|>assistant
".into(),
            thinking_prefix: "<think>
".into(),
            thinking_suffix: "</think>
".into(),
            tool_call_template: "<tool_call>\\n{\"name\":\"{name}\",\"arguments\": {arguments}}\\n</tool_call>".into(),
            tool_calls_prefix: "".into(),
            tool_calls_suffix: "".into(),
            tool_response_role: "user".into(),
            tool_response_prefix: "<tool_response>
".into(),
            tool_response_suffix: "
</tool_response>".into(),
        }
    }

    fn llama3() -> ChatTemplate {
        ChatTemplate {
            start_token: "".into(),
            stop_tokens: vec!["<|eot_id|>".into(), "<|end_of_text|>".into()],
            role_prefixes: vec![("system".into(), "<|start_header_id|>system<|end_header_id|>
".into()), ("user".into(), "<|start_header_id|>user<|end_header_id|>
".into()), ("assistant".into(), "<|start_header_id|>assistant<|end_header_id|>
".into()), ("ipython".into(), "<|start_header_id|>ipython<|end_header_id|>
".into())],
            role_suffixes: vec![("system".into(), "<|eot_id|>
".into()), ("user".into(), "<|eot_id|>
".into()), ("assistant".into(), "<|eot_id|>
".into()), ("ipython".into(), "<|eot_id|>
".into())],
            system_handling: SystemHandling::Standalone,
            system_separator: "".into(),
            generation_header: "<|start_header_id|>assistant<|end_header_id|>
".into(),
            thinking_prefix: "<think>
".into(),
            thinking_suffix: "</think>
".into(),
            tool_call_template: "".into(),
            tool_calls_prefix: "".into(),
            tool_calls_suffix: "".into(),
            tool_response_role: "".into(),
            tool_response_prefix: "".into(),
            tool_response_suffix: "".into(),
        }
    }

    fn r1() -> ChatTemplate {
        ChatTemplate {
            start_token: "<｜begin▁of▁sentence｜>".into(),
            stop_tokens: vec!["<｜end▁of▁sentence｜>".into()],
            role_prefixes: vec![("user".into(), "<｜User｜>".into()), ("assistant".into(), "<｜Assistant｜>".into())],
            role_suffixes: vec![("user".into(), "".into()), ("assistant".into(), "<｜end▁of▁sentence｜>".into())],
            system_handling: SystemHandling::BarePrepend,
            system_separator: "".into(),
            generation_header: "<｜Assistant｜><think>
".into(),
            thinking_prefix: "".into(),
            thinking_suffix: "".into(),
            tool_call_template: "".into(),
            tool_calls_prefix: "".into(),
            tool_calls_suffix: "".into(),
            tool_response_role: "".into(),
            tool_response_prefix: "".into(),
            tool_response_suffix: "".into(),
        }
    }

    fn gemma2() -> ChatTemplate {
        ChatTemplate {
            start_token: "<bos>
".into(),
            stop_tokens: vec!["<end_of_turn>".into(), "<eos>".into()],
            role_prefixes: vec![("user".into(), "<start_of_turn>user
".into()), ("assistant".into(), "<start_of_turn>model
".into())],
            role_suffixes: vec![("user".into(), "<end_of_turn>
".into()), ("assistant".into(), "<end_of_turn>
".into())],
            system_handling: SystemHandling::MergeWithUser,
            system_separator: "

".into(),
            generation_header: "<start_of_turn>model
".into(),
            thinking_prefix: "".into(),
            thinking_suffix: "".into(),
            tool_call_template: "".into(),
            tool_calls_prefix: "".into(),
            tool_calls_suffix: "".into(),
            tool_response_role: "".into(),
            tool_response_prefix: "".into(),
            tool_response_suffix: "".into(),
        }
    }

    fn mistral3() -> ChatTemplate {
        ChatTemplate {
            start_token: "<s>
".into(),
            stop_tokens: vec!["</s>".into()],
            role_prefixes: vec![("user".into(), "[INST] ".into()), ("assistant".into(), "".into())],
            role_suffixes: vec![("user".into(), " [/INST]".into()), ("assistant".into(), "</s>".into())],
            system_handling: SystemHandling::MergeWithUser,
            system_separator: "

".into(),
            generation_header: "".into(),
            thinking_prefix: "".into(),
            thinking_suffix: "".into(),
            tool_call_template: "".into(),
            tool_calls_prefix: "".into(),
            tool_calls_suffix: "".into(),
            tool_response_role: "".into(),
            tool_response_prefix: "".into(),
            tool_response_suffix: "".into(),
        }
    }

    fn olmo3() -> ChatTemplate {
        ChatTemplate {
            start_token: "".into(),
            stop_tokens: vec!["<|im_end|>".into()],
            role_prefixes: vec![("system".into(), "<|im_start|>system
".into()), ("user".into(), "<|im_start|>user
".into()), ("assistant".into(), "<|im_start|>assistant
".into())],
            role_suffixes: vec![("system".into(), "<|im_end|>
".into()), ("user".into(), "<|im_end|>
".into()), ("assistant".into(), "<|im_end|>
".into())],
            system_handling: SystemHandling::Standalone,
            system_separator: "".into(),
            generation_header: "<|im_start|>assistant
".into(),
            thinking_prefix: "<think>
".into(),
            thinking_suffix: "</think>
".into(),
            tool_call_template: "".into(),
            tool_calls_prefix: "".into(),
            tool_calls_suffix: "".into(),
            tool_response_role: "".into(),
            tool_response_prefix: "".into(),
            tool_response_suffix: "".into(),
        }
    }

    // ── Qwen 3 ──────────────────────────────────────────────────────

    #[test]
    fn qwen3_system_user() {
        let ct = qwen3();
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello")];
        let out = render_template(&ct, &messages, true, false);
        assert!(out.ends_with("<|im_start|>assistant\n"), "got: {:?}", out);
        assert!(out.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nHello<|im_end|>"));
    }

    #[test]
    fn qwen3_multi_turn() {
        let ct = qwen3();
        let messages = vec![msg("system", "You are helpful."), msg("user", "What is 2+2?"), msg("assistant", "4"), msg("user", "And 3+3?")];
        let out = render_template(&ct, &messages, true, false);
        assert!(out.ends_with("<|im_start|>assistant\n"), "got: {:?}", out);
        assert!(out.contains("<|im_start|>assistant\n4<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nAnd 3+3?<|im_end|>"));
    }

    #[test]
    fn qwen3_no_generation_prompt() {
        let ct = qwen3();
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello")];
        let out = render_template(&ct, &messages, false, false);
        assert!(!out.ends_with("<|im_start|>assistant\n"), "should not end with gen prompt: {:?}", out);
    }

    #[test]
    fn qwen3_incremental_matches_full() {
        let ct = qwen3();
        let sys = render_template(&ct, &[msg("system", "Be concise.")], false, false);
        let usr = render_template(&ct, &[msg("user", "Hi")], true, false);
        let incremental = format!("{}{}", sys, usr);
        let full = render_template(&ct, &[msg("system", "Be concise."), msg("user", "Hi")], true, false);
        assert_eq!(incremental, full, "\nincremental:\n{}\nfull:\n{}", incremental, full);
    }

    #[test]
    fn qwen3_incremental_multi_turn() {
        let ct = qwen3();
        let sys = render_template(&ct, &[msg("system", "Be concise.")], false, false);
        let u1 = render_template(&ct, &[msg("user", "What is 2+2?")], false, false);
        let a1 = render_template(&ct, &[msg("assistant", "4")], false, false);
        let u2 = render_template(&ct, &[msg("user", "And 3+3?")], true, false);
        let incremental = format!("{}{}{}{}", sys, u1, a1, u2);
        let full = render_template(&ct, &[msg("system", "Be concise."), msg("user", "What is 2+2?"), msg("assistant", "4"), msg("user", "And 3+3?")], true, false);
        assert_eq!(incremental, full);
    }

    // ── Llama 3 ─────────────────────────────────────────────────────

    #[test]
    fn llama3_system_user() {
        let ct = llama3();
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello")];
        let out = render_template(&ct, &messages, true, false);
        assert!(out.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(out.contains("<|start_header_id|>assistant<|end_header_id|>"), "got: {:?}", out);
    }

    #[test]
    fn llama3_multi_turn() {
        let ct = llama3();
        let messages = vec![msg("system", "Be brief."), msg("user", "Hi"), msg("assistant", "Hello!"), msg("user", "Bye")];
        let out = render_template(&ct, &messages, true, false);
        assert!(out.contains("Hello!"));
        assert!(out.contains("<|eot_id|>"));
        assert!(out.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn llama3_incremental_matches_full() {
        let ct = llama3();
        let sys = render_template(&ct, &[msg("system", "Be brief.")], false, false);
        let usr = render_template(&ct, &[msg("user", "Hi")], true, false);
        let incremental = format!("{}{}", sys, usr);
        let full = render_template(&ct, &[msg("system", "Be brief."), msg("user", "Hi")], true, false);
        assert_eq!(incremental, full);
    }

    // ── DeepSeek R1 ─────────────────────────────────────────────────

    #[test]
    fn r1_system_user() {
        let ct = r1();
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello")];
        let out = render_template(&ct, &messages, true, false);
        assert!(out.contains("You are helpful."));
        assert!(out.contains("<｜User｜>Hello"));
        assert!(out.ends_with("<｜Assistant｜><think>\n"), "got: {:?}", out);
    }

    #[test]
    fn r1_with_bos() {
        let ct = r1();
        let messages = vec![msg("system", "Sys"), msg("user", "Hi")];
        let out = render_template(&ct, &messages, true, true);
        assert!(out.starts_with("<｜begin▁of▁sentence｜>"), "got: {:?}", out);
    }

    #[test]
    fn r1_multi_turn() {
        let ct = r1();
        let messages = vec![msg("system", "Sys"), msg("user", "Q1"), msg("assistant", "A1"), msg("user", "Q2")];
        let out = render_template(&ct, &messages, true, false);
        assert!(out.contains("<｜User｜>Q1"));
        assert!(out.contains("<｜Assistant｜>A1<｜end▁of▁sentence｜>"));
        assert!(out.contains("<｜User｜>Q2"));
        assert!(out.ends_with("<｜Assistant｜><think>\n"));
    }

    // ── Gemma 2/3 ───────────────────────────────────────────────────

    #[test]
    fn gemma2_system_user() {
        let ct = gemma2();
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello")];
        let out = render_template(&ct, &messages, true, false);
        assert!(out.contains("<start_of_turn>user"), "got: {:?}", out);
        assert!(out.contains("Hello"));
        assert!(out.contains("<start_of_turn>model"));
    }

    #[test]
    fn gemma2_multi_turn() {
        let ct = gemma2();
        let messages = vec![msg("system", "Sys"), msg("user", "Q1"), msg("assistant", "A1"), msg("user", "Q2")];
        let out = render_template(&ct, &messages, true, false);
        assert!(out.contains("<start_of_turn>model\nA1<end_of_turn>"));
        assert!(out.contains("Q2<end_of_turn>"));
        assert!(out.ends_with("<start_of_turn>model\n"), "got: {:?}", out);
    }

    // ── Mistral 3 ───────────────────────────────────────────────────

    #[test]
    fn mistral3_system_user() {
        let ct = mistral3();
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello")];
        let out = render_template(&ct, &messages, true, false);
        assert!(out.contains("[INST]"), "got: {:?}", out);
        assert!(out.contains("Hello"));
    }

    #[test]
    fn mistral3_multi_turn() {
        let ct = mistral3();
        let messages = vec![msg("system", "Sys"), msg("user", "Q1"), msg("assistant", "A1"), msg("user", "Q2")];
        let out = render_template(&ct, &messages, true, false);
        assert!(out.contains("A1</s>"));
        assert!(out.contains("Q2 [/INST]"));
    }

    // ── OLMo 3 ──────────────────────────────────────────────────────

    #[test]
    fn olmo3_system_user() {
        let ct = olmo3();
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello")];
        let out = render_template(&ct, &messages, true, false);
        assert!(out.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(out.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn olmo3_multi_turn() {
        let ct = olmo3();
        let messages = vec![msg("system", "Sys"), msg("user", "Q1"), msg("assistant", "A1"), msg("user", "Q2")];
        let out = render_template(&ct, &messages, true, false);
        assert!(out.contains("<|im_start|>assistant\nA1<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nQ2<|im_end|>"));
    }

}
