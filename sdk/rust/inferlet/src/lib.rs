//! Inferlet SDK for Pie
//!
//! This crate provides the core types and traits for building inferlets
//! that run on the Pie inference engine.

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
// Async Wrappers
// =============================================================================

use wstd::io::AsyncPollable;

/// Async-friendly context operations.
pub mod context {
    use super::*;
    use crate::pie::core::context as raw;
    
    pub use raw::Context;
    
    /// Acquires a lock on the context asynchronously.
    pub async fn lock(ctx: &raw::Context) -> bool {
        let future = ctx.lock();
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().unwrap_or(false)
    }
}

/// Async-friendly adapter operations.
pub mod adapter {
    use super::*;
    use crate::pie::core::adapter as raw;
    
    pub use raw::Adapter;
    
    /// Acquires a lock on the adapter asynchronously.
    pub async fn lock(adapter: &raw::Adapter) -> bool {
        let future = adapter.lock();
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().unwrap_or(false)
    }
}

/// Async-friendly model operations.
pub mod model {
    pub use crate::pie::core::model::{Model, Tokenizer};
}

/// Async-friendly runtime operations.
pub mod runtime {
    use super::*;
    use crate::pie::core::runtime as raw;
    
    // Re-export sync functions
    pub use raw::{version, instance_id, username, models};
    
    /// Spawns a new inferlet and waits for its result asynchronously.
    pub async fn spawn(package_name: &str, args: &[String]) -> Result<String, String> {
        let future = raw::spawn(package_name, args)?;
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().ok_or_else(|| "Spawn result not available".to_string())
    }
}

/// Async-friendly messaging operations.
pub mod messaging {
    use super::*;
    use crate::pie::core::messaging as raw;
    
    // Re-export sync functions
    pub use raw::{send, broadcast, subscribe, Subscription};
    
    /// Receives an incoming message from the remote user client asynchronously.
    pub async fn receive() -> Option<String> {
        let future = raw::receive();
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get()
    }
    
    /// Gets next message from a subscription asynchronously.
    pub async fn subscription_get(sub: &raw::Subscription) -> Option<String> {
        let pollable = sub.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        sub.get()
    }
}

/// Async-friendly inference operations.
pub mod inference {
    use super::*;
    use crate::pie::core::inference as raw;
    
    // Re-export types that don't need wrapping
    pub use raw::{Output, Sampler, ForwardPass, FutureOutput};
    
    /// Executes a forward pass and waits for the result asynchronously.
    ///
    /// This is the async wrapper around `ForwardPass::execute()` that
    /// uses AsyncPollable to await the result.
    pub async fn execute(pass: &ForwardPass) -> Result<Output, String> {
        let future_output = pass.execute()?;
        
        // Wait for the result using AsyncPollable
        let pollable = future_output.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        
        // Get the result
        future_output.get().ok_or_else(|| "No output available".to_string())
    }
}

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::main;
    pub use crate::context::Context;
    pub use crate::inference::{ForwardPass, Output, Sampler, execute};
    pub use crate::model::Model;
    pub use crate::runtime;
    pub use crate::messaging;
    pub use crate::adapter::Adapter;
}
