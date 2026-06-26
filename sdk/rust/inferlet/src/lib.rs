//! Inferlet SDK for Pie
//!
//! This crate provides the core types and traits for building inferlets
//! that run on the Pie inference engine.

/// Result type for inferlet operations (compatible with WIT bindings).
pub type Result<T> = std::result::Result<T, String>;

// Re-export wit_bindgen so the macro-generated inline WIT can reference it
pub use wit_bindgen;

// Re-export serde and serde_json so the macro-generated JSON bridge can use them
pub use schemars;
pub use serde;
pub use serde_json;

// Re-export the attribute macros
pub use inferlet_macros::{main, tool};

// Generate WIT bindings directly in lib.rs. With no `async:` option, the
// WIT's own `async func` annotations drive async generation: only
// run/execute/receive/pull become `async fn` (component-model-async) and
// `stream<T>` (messaging.subscribe) becomes a StreamReader; sync funcs
// (model::encode, chat::*, …) stay sync. wit-bindgen generates the wasi:io
// bindings itself (0.58-suffixed cabi_realloc) so it doesn't collide with
// std's 0.57.1 copy.
wit_bindgen::generate!({
    path: "wit",
    world: "inferlet",
    pub_export_macro: true,
    generate_all,
});

// Re-export types that don't need async wrappers directly
pub use pie::core::types;
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

/// The engine serves exactly one model; these are global functions over
/// that single bound model. There is no `Model`/`Tokenizer` handle to pass
/// around — call `model::encode`, `model::name`, etc. directly.
pub mod model {
    pub use crate::pie::core::model::{
        architecture, decode, default_system_speculation, encode, name, special_tokens,
        split_regex, vocabs,
    };
}

// =============================================================================
// Other re-exports
// =============================================================================

pub mod runtime {
    pub use crate::pie::core::runtime::*;
}

/// Suspend the current inferlet for `duration` without blocking the host
/// event loop. Backed by the engine's async timer (host-provided under
/// component-model-async — wasi:clocks 0.2 pollables have no guest-side
/// future bridge). Use for streaming pacing, retry backoff, etc.
///
/// ```ignore
/// inferlet::sleep(std::time::Duration::from_millis(50)).await;
/// ```
pub async fn sleep(duration: std::time::Duration) {
    let nanos = duration.as_nanos().min(u64::MAX as u128) as u64;
    crate::pie::core::runtime::sleep(nanos).await;
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

// Under component-model-async, the WIT `async func`s are generated as native
// `async fn`s directly on the bindings — `forward-pass.execute().await`,
// `session::receive().await`, `messaging::pull().await` — and
// `messaging::subscribe()` returns a `StreamReader<String>`. No SDK-side
// pollable/future polling shim is needed (the old `ForwardPassExt`,
// `SubscriptionExt`, `FutureStringExt`, `FutureBlobExt` and the `wstd`
// executor have been removed); the host event loop drives all of it.

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
    pub use crate::model;
    pub use crate::runtime;
    pub use crate::{Context, Result, Schema, Tool};
    pub use crate::{main, tool};

    pub use crate::forward::{Forward, Output, ProbeHandle, SampleHandle};
    pub use crate::generation::{GenStep, Generator};
    pub use crate::sample::{Probe, Sampler};
    pub use crate::spec::Speculator;
    pub use crate::{chat, reasoning, tools};
}
