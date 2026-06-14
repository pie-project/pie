//! Inferlet SDK for Pie
//!
//! This crate provides the core types and traits for building inferlets
//! that run on the Pie inference engine.

/// Result type for inferlet operations (compatible with WIT bindings).
pub type Result<T> = std::result::Result<T, String>;

// Re-export wstd for the macro to use
pub use wstd;

// Re-export wit_bindgen so the macro-generated inline WIT can reference it
pub use wit_bindgen;

// Re-export serde and serde_json so the macro-generated JSON bridge can use them
pub use serde;
pub use serde_json;
pub use schemars;

// Re-export the attribute macros
pub use inferlet_macros::{main, tool};

// Generate WIT bindings directly in lib.rs
wit_bindgen::generate!({
    path: "wit",
    world: "inferlet",
    pub_export_macro: true,
    with: {
         "wasi:io/poll@0.2.4": ::wasi::io::poll,
    },
    generate_all,
});

// Re-export types that don't need async wrappers directly
pub use pie::core::types;
pub use pie::mcp;
pub use pie::zo;

// =============================================================================
// Context
// =============================================================================

mod context;

pub use context::{
    Context, RawContext,
    RetentionBudget, RetentionReason, RetentionReport,
    Constrain, GrammarConstraint, Schema,
    AnyJson, Ebnf, JsonSchema, Regex,
};

// =============================================================================
// Sampler / Probe + Forward primitive
// =============================================================================

pub mod sample;
pub mod forward;

// =============================================================================
// Generation state machine + decoders + speculation
// =============================================================================

pub mod generation;
pub mod chat;
pub mod reasoning;
pub mod spec;
pub mod tools;

pub use generation::{Generator, GenStep};
pub use spec::Speculator;
pub use tools::Tool;

// =============================================================================
// Adapter
// =============================================================================

pub mod adapter {
    pub use crate::pie::core::adapter::Adapter;
}

// =============================================================================
// Model
// =============================================================================

pub mod model {
    pub use crate::pie::core::model::{Model, Tokenizer};
}

// =============================================================================
// Other re-exports
// =============================================================================

pub mod runtime {
    pub use crate::pie::core::runtime::*;
}

pub mod blob_store {
    pub use crate::pie::core::blob_store::*;
}

pub mod scheduling {
    pub use crate::pie::core::scheduling::*;
}

pub mod messaging {
    pub use crate::pie::core::messaging::*;
}

pub mod inference {
    pub use crate::pie::core::inference::*;
}

/// Grammar matcher — re-export for callers that build their own
/// constraints around it. Most users should reach for [`Schema`] or
/// [`GrammarConstraint`] instead.
pub use crate::pie::core::inference::Matcher;


// =============================================================================
// Async Extension Traits
// =============================================================================

use wstd::io::AsyncPollable;

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

/// Extension trait for FutureBlob — mirror of [`FutureStringExt`] for
/// binary payloads (e.g. files arriving via `session::receive_file`).
pub trait FutureBlobExt {
    /// Waits for the blob asynchronously. `None` when the producer
    /// closes the channel before sending a payload.
    fn wait_async(&self) -> impl std::future::Future<Output = Option<Vec<u8>>>;
}

impl FutureBlobExt for types::FutureBlob {
    async fn wait_async(&self) -> Option<Vec<u8>> {
        let pollable = self.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        self.get()
    }
}

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
///
/// `use inferlet::prelude::*;` covers the common case so inferlets don't
/// have to maintain a hand-rolled import grocery list.
pub mod prelude {
    pub use crate::{main, tool};
    pub use crate::{Context, Result, Schema, Tool};
    pub use crate::model::Model;
    pub use crate::runtime;
    pub use crate::messaging;
    pub use crate::adapter::Adapter;

    pub use crate::forward::{Forward, Output, SampleHandle, ProbeHandle};
    pub use crate::sample::{Sampler, Probe};
    pub use crate::generation::{Generator, GenStep};
    pub use crate::{chat, reasoning, tools};
    pub use crate::spec::Speculator;

    // Extension traits
    pub use crate::ForwardPassExt;
    pub use crate::SubscriptionExt;
    pub use crate::FutureStringExt;
    pub use crate::FutureBlobExt;
}
