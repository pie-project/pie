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
pub use schemars;
pub use serde;
pub use serde_json;

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
    AnyJson, Constrain, Context, Ebnf, GrammarConstraint, JsonSchema, RawContext, Regex, Schema,
};

// =============================================================================
// Sampler / Probe + Forward primitive
// =============================================================================

pub mod audio;
pub mod forward;
pub mod http;
pub mod sample;

// =============================================================================
// Generation state machine + decoders + speculation
// =============================================================================

pub mod chat;
pub mod generation;
pub mod reasoning;
pub mod spec;
pub mod tools;

pub use generation::{GenStep, Generator};
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

pub mod scheduling {
    pub use crate::pie::core::scheduling::*;
}

pub mod messaging {
    pub use crate::pie::core::messaging::*;
}

pub mod session {
    pub use crate::pie::core::session::*;
}

pub mod inference {
    pub use crate::pie::core::inference::*;
}

/// Multimodal input. The inferlet hands the host raw encoded bytes —
/// [`Image::from_bytes`](media::Image) (PNG/JPEG), [`Video::from_bytes`](media::Video)
/// (animated GIF), [`Audio::from_bytes`](media::Audio) (WAV) — and the host
/// decodes + preprocesses per the bound model. The returned handle's
/// `token-count` / `position-span` / `grid` describe how it occupies the
/// context. No model-specific code lives in the inferlet. See MULTIMODAL.md.
pub mod media {
    pub use crate::pie::core::media::{Audio, Image, Video};
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
        if let Some(output) = future_output.get() {
            return Ok(output);
        }
        let pollable = future_output.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        for _ in 0..256 {
            if let Some(output) = future_output.get() {
                return Ok(output);
            }
            wstd::task::sleep(wstd::time::Duration::from_micros(1)).await;
        }
        Err("No output available".to_string())
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
// Inferlet-to-inferlet launch
// =============================================================================

/// Handle to a launched child inferlet. See [`launch`].
///
/// Three modes:
/// - **fire-and-forget**: drop the handle. The child runs to completion on its
///   own; its return value is discarded.
/// - **await the result**: `child.await` (via [`IntoFuture`]).
/// - **timeout / cancel**: keep the handle, call `child.wait()` for a
///   borrowing future, and call `child.cancel()` if the timeout fires.
pub struct Child(pie::core::runtime::Child);

impl Child {
    /// Process id (UUID) of the child, useful for logs.
    pub fn pid(&self) -> String {
        self.0.pid()
    }

    /// Hard-kill the child if still running. Idempotent.
    pub fn cancel(&self) {
        self.0.cancel()
    }

    /// Wait for the child's result without consuming the handle. Use this
    /// when you may need to call `cancel()` later (e.g. after a timeout).
    pub async fn wait(&mut self) -> Result<String> {
        let pollable = self.0.pollable();
        wstd::io::AsyncPollable::new(pollable).wait_for().await;
        self.0
            .get()
            .unwrap_or_else(|| Err("pollable signaled but get() returned None".to_string()))
    }
}

impl std::future::IntoFuture for Child {
    type Output = Result<String>;
    // Box the future to keep the SDK on stable Rust without TAIT. The cost
    // is one allocation per `child.await`, which is negligible next to the
    // wasmtime store creation a launch already pays for.
    type IntoFuture = std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>>>>;

    fn into_future(mut self) -> Self::IntoFuture {
        Box::pin(async move { self.wait().await })
    }
}

/// Launch a child inferlet identified by `name@version`. Returns a
/// [`Child`] handle. See the handle docs for the three usage modes.
pub fn launch(program: &str, input: &str) -> Result<Child> {
    pie::core::runtime::launch(program, input)
        .map(Child)
        .map_err(|e| e.to_string())
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
    pub use crate::adapter::Adapter;
    pub use crate::messaging;
    pub use crate::model::Model;
    pub use crate::runtime;
    pub use crate::{Child, Context, Result, Schema, Tool, launch};
    pub use crate::{main, tool};

    pub use crate::forward::{Forward, Output, ProbeHandle, SampleHandle};
    pub use crate::generation::{GenStep, Generator};
    pub use crate::sample::{Probe, Sampler};
    pub use crate::spec::Speculator;
    pub use crate::{chat, reasoning, tools};

    // Extension traits
    pub use crate::ForwardPassExt;
    pub use crate::FutureBlobExt;
    pub use crate::FutureStringExt;
    pub use crate::SubscriptionExt;
}
