//! Inferlet SDK for Pie
//!
//! This crate provides the core types and traits for building inferlets
//! that run on the Pie inference engine.

/// Result type for inferlet operations (compatible with WIT bindings).
pub type Result<T> = std::result::Result<T, String>;

// Re-export wstd for the macro to use
pub use wstd;

// Re-export the main attribute macro
pub use inferlet_macros::main;

// Generate WIT bindings directly in lib.rs
wit_bindgen::generate!({
    path: "wit",
    world: "inferlet",
    pub_export_macro: true,
    with: {
         "wasi:io/poll@0.2.4": wasi::io::poll,
    },
    generate_all,
});

// Re-export types that don't need async wrappers directly
pub use pie::core::types;
pub use pie::mcp;
pub use pie::zo;

// =============================================================================
// Re-exports from raw bindings
// =============================================================================

pub mod context {
    pub use crate::pie::core::context::Context;
}

pub mod adapter {
    pub use crate::pie::core::adapter::Adapter;
}

pub mod model {
    pub use crate::pie::core::model::{Model, Tokenizer};
}

pub mod runtime {
    pub use crate::pie::core::runtime::*;
}

pub mod messaging {
    pub use crate::pie::core::messaging::*;
}

pub mod inference {
    pub use crate::pie::core::inference::*;
}

// =============================================================================
// Async Extension Traits
// =============================================================================

use wstd::io::AsyncPollable;

/// Extension trait for async adapter operations.
pub trait AdapterExt {
    /// Acquires a lock on the adapter asynchronously.
    fn acquire_lock_async(&self) -> impl std::future::Future<Output = bool>;
}

impl AdapterExt for adapter::Adapter {
    async fn acquire_lock_async(&self) -> bool {
        let future = self.acquire_lock();
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().unwrap_or(false)
    }
}

/// Extension trait for async forward pass operations.
pub trait ForwardPassExt {
    /// Executes the forward pass and waits for the result asynchronously.
    fn execute_async(&self) -> impl std::future::Future<Output = Result<inference::Output>>;
}

impl ForwardPassExt for inference::ForwardPass {
    async fn execute_async(&self) -> Result<inference::Output> {
        let future_output = self.execute()?;
        let pollable = future_output.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future_output.get().ok_or_else(|| "No output available".to_string())
    }
}

/// Extension trait for async messaging subscription operations.
pub trait SubscriptionExt {
    /// Gets the next message from a subscription asynchronously.
    fn get_async(&self) -> impl std::future::Future<Output = Option<String>>;
}

impl SubscriptionExt for messaging::Subscription {
    async fn get_async(&self) -> Option<String> {
        let pollable = self.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        self.get()
    }
}

/// Extension trait for FutureString (used by receive and spawn).
pub trait FutureStringExt {
    /// Waits for the result asynchronously.
    fn wait_async(&self) -> impl std::future::Future<Output = Option<String>>;
}

impl FutureStringExt for types::FutureString {
    async fn wait_async(&self) -> Option<String> {
        let pollable = self.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        self.get()
    }
}

// =============================================================================
// Context Extension Trait (Consolidated)
// =============================================================================

mod context_ext;

pub use context_ext::{
    // ContextExt trait (has Fill + Generate + async operations)
    ContextExt,
    // Supporting types
    Message, ToolCall, render_template,
    TokenStream, Speculate, Speculation, Constrain,
};

// =============================================================================
// Argument Parsing (re-exported from pico_args)
// =============================================================================

/// Re-export of `pico_args::Arguments` for ergonomic CLI argument parsing.
pub use pico_args::Arguments;

/// Parses a `Vec<String>` (as received from the WIT entry point) into
/// a `pico_args::Arguments` for flag/option extraction.
pub fn parse_args(args: Vec<String>) -> Arguments {
    Arguments::from_vec(args.into_iter().map(std::ffi::OsString::from).collect())
}

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::main;
    pub use crate::context::Context;
    pub use crate::inference::{ForwardPass, Output, Sampler};
    pub use crate::model::Model;
    pub use crate::runtime;
    pub use crate::messaging;
    pub use crate::adapter::Adapter;
    
    // Extension traits
    pub use crate::ContextExt;
    pub use crate::AdapterExt;
    pub use crate::ForwardPassExt;
    pub use crate::SubscriptionExt;
    pub use crate::FutureStringExt;
}