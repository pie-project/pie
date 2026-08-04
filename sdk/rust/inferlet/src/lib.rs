//! Inferlet SDK for Pie
//!
//! This crate provides the core types and traits for building inferlets
//! that run on the Pie inference engine.

/// Result type for inferlet operations (compatible with WIT bindings).
pub type Result<T> = std::result::Result<T, String>;

/// Add context to a failure on its way up.
///
/// `pie:inferlet` declares `type error = string`, so every fallible call
/// across the ABI already fails with a message and there is no error type to
/// introduce. What is missing is the standard way of saying WHERE the message
/// came from, which is why the inferlets were writing
/// `map_err(|e| format!("reserve KV: {e}"))` by hand.
///
/// ```ignore
/// ws.reserve(pages).context("reserve KV")?;
/// tok_out
///     .take()
///     .to_host::<i32>()
///     .await
///     .with_context(|| format!("tok_out.take @{}", generated.len()))?;
/// ```
pub trait Context<T> {
    /// Prefix the error with `what`, as `"{what}: {error}"`.
    fn context(self, what: &str) -> Result<T>;

    /// Same, but the prefix is only built if there is an error to prefix.
    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, what: F) -> Result<T>;
}

impl<T, E: std::fmt::Display> Context<T> for std::result::Result<T, E> {
    fn context(self, what: &str) -> Result<T> {
        self.map_err(|error| format!("{what}: {error}"))
    }

    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, what: F) -> Result<T> {
        self.map_err(|error| format!("{}: {error}", what()))
    }
}

/// The same, for an absent value: `Option` has no error to prefix, so `what`
/// becomes the whole message.
impl<T> Context<T> for Option<T> {
    fn context(self, what: &str) -> Result<T> {
        self.ok_or_else(|| what.to_string())
    }

    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, what: F) -> Result<T> {
        self.ok_or_else(|| what().to_string())
    }
}

// Re-export wit_bindgen so the macro-generated inline WIT can reference it
pub use wit_bindgen;

// Re-export serde and serde_json so the macro-generated JSON bridge can use them
pub use serde;
pub use serde_json;

// Re-export the attribute macro
pub use inferlet_macros::main;

// Generate WIT bindings directly in lib.rs. With no `async:` option, the
// WIT's own `async func` annotations drive async generation: only
// run/execute/receive become `async fn` (component-model-async); sync funcs
// (model::encode, chat::*, …) stay sync. wit-bindgen generates the wasi:io
// bindings itself with versioned cabi_realloc symbols so it doesn't collide
// with std's copy.
wit_bindgen::generate!({
    path: "wit",
    world: "inferlet",
    pub_export_macro: true,
    generate_all,
});

// Re-export types that don't need async wrappers directly
pub use pie::inferlet::types;

// =============================================================================
// Context
// =============================================================================

/// The runtime working-set resources (KV page-slot array + recurrent state).
/// The generated WIT resources, unwrapped; [`ptir::WorkingSet`] is the
/// pass-facing handle built over them.
pub mod working_set {
    pub use crate::pie::inferlet::working_set::*;
}

// =============================================================================
// Forward primitive
// =============================================================================

pub mod mask;
/// The author-facing PTIR bridge (overview §3/§5): `ForwardPass`/`Pipeline`/
/// `WorkingSet`/`Channel` over the WIT `ptir` resources, driving the `pie-dsl`
/// trace `Builder`. The single home of the PTIR authoring surface.
pub mod ptir;

// =============================================================================
// Generation state machine + decoders
// =============================================================================

pub mod chat;

// =============================================================================
// Model
// =============================================================================

/// The engine serves exactly one model; these are global functions over
/// that single bound model. There is no `Model`/`Tokenizer` handle to pass
/// around — call `model::encode`, `model::name`, etc. directly.
pub mod model {
    pub use crate::pie::inferlet::model::{
        ForwardKind, architecture, arena_block_size, channel_capacity, default_system_speculation,
        frame_size, kv_page_size, max_embed_length, name, output_vocab_size, pass_kind,
        rs_buffer_page_size, rs_fold_granularity, rs_state_size, submit_deadline_us,
    };
    // Tokenizer functions split into the `tokenizer` interface (§2.2); re-exported
    // here so `model::encode`/`model::decode`/… keep working for inferlet source.
    pub use crate::pie::inferlet::tokenizer::{
        Token, decode, encode, special_tokens, split_regex, vocabs,
    };
}

// =============================================================================
// Other re-exports
// =============================================================================

pub mod runtime {
    pub use crate::pie::inferlet::system::*;
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
    crate::wasi::clocks::monotonic_clock::wait_for(nanos).await;
}

/// Current `wasi:clocks/monotonic-clock` mark in nanoseconds.
pub fn monotonic_now_ns() -> u64 {
    crate::wasi::clocks::monotonic_clock::now()
}

pub mod session {
    pub use crate::pie::inferlet::session::*;
}

/// Grammar compilation + incremental matching (the WIT `grammar` interface).
/// [`Grammar`](grammar::Grammar) compiles a JSON Schema / regex / EBNF source
/// once for the bound model's vocabulary; [`Matcher`](grammar::Matcher) walks
/// one generation against it, exposing the packed allowed-token bitmask that
/// [`mask`] interprets.
pub mod grammar {
    pub use crate::pie::inferlet::grammar::*;
}

/// Multimodal input. The inferlet hands the host raw encoded bytes —
/// [`Image::from_bytes`](media::Image) (PNG/JPEG), [`Video::from_bytes`](media::Video)
/// (animated GIF), [`Audio::from_bytes`](media::Audio) (WAV) — and the host
/// decodes + preprocesses per the bound model. The returned handle's
/// `token-count` / `position-span` / `grid` describe how it occupies the
/// context. No model-specific code lives in the inferlet. See MULTIMODAL.md.
pub mod media {
    pub use crate::pie::inferlet::media::{Audio, Image, Video};
}

// Under component-model-async, the WIT `async func`s are generated as native
// `async fn`s directly on the bindings — `forward-pass.execute().await`,
// `session::receive().await` — so no SDK-side pollable/future polling shim
// is needed (the old `ForwardPassExt`, `FutureStringExt`, `FutureBlobExt` and the `wstd`
// executor have been removed); the host event loop drives all of it.
