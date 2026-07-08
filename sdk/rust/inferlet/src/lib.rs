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

// =============================================================================
// Context
// =============================================================================

mod constraint;

pub use constraint::{AnyJson, Constrain, Ebnf, GrammarConstraint, JsonSchema, Regex, Schema};

/// The runtime working-set resources (KV page-slot array + recurrent state).
/// Most inferlets use the [`Context`] facade; reach here for direct control.
pub mod working_set {
    pub use crate::pie::core::working_set::*;
}

// =============================================================================
// Sampler / Probe + Forward primitive
// =============================================================================

pub mod audio;
/// Run-ahead carrier mechanics (`submit_pass` / `discard_pass`) — the two
/// correctness-critical primitives for pipelined decode on the raw WIT; the
/// decode LOOP stays hand-written in the inferlet. See `ptir-sdk-minimization-audit`
/// (keep-core) + `ptir-lowlevel-runahead-mechanics` (bravo, mechanics).
pub mod carrier;
pub mod emit;
/// KV page-geometry primitives (token→page math) as free functions over the
/// raw WIT — the minimal-core geometry helper a low-level inferlet calls
/// directly instead of re-inlining `first_write_page/total_pages/offset`.
/// See `ptir-sdk-minimization-audit` §4.
pub mod geometry;
pub mod http;
pub mod mask;
/// Non-sampling prefill mechanics (text / image / audio KV materialization) —
/// the keep-core twin of [`carrier`]; the raw-WIT form of `Context::flush` /
/// `append_image` / `append_audio`. See `ptir-sdk-minimization-audit`.
pub mod prefill;
pub mod program;

/// Standard-sampler lowering (keep-core): turn a sampler spec into an
/// attachable `tensor::Program` + its per-fire input bindings, so a low-level
/// inferlet gets top-p/top-k/min-p/temperature sampling without hand-building
/// the per-kind Sampling-IR. The sampler analog of `geometry`/`carrier`; the
/// `Sampler` enum + `.generate()` facade are the sugar that gets deleted. See
/// `ptir-sdk-minimization-audit`.
pub mod sampler;

/// Snapshot manifests (keep-core): the thin `SnapshotData` + serde +
/// wasi:filesystem I/O (`save`/`open`/`snapshot`/`take`/`delete`). The token-log
/// REPLAY factors out to the inferlet's carrier prefill; the `Context::save/open`
/// facade is the sugar that gets deleted. See `ptir-snapshot-keepcore-spec`.
pub mod snapshot;

/// Sampling-IR EDSL — re-export of the `sampling-edsl` crate so inferlets can
/// author programmable samplers (`Graph`, helpers like `softmax` / `top_p_mask`
/// / `mirostat_mask` / `grammar_mask`, typed `Value` handles) with a single
/// `inferlet` dependency. Build a program, then (Stage 2, foxtrot's guest emit)
/// lower it to a [`tensor::Program`](crate::tensor::Program) and attach it via
/// [`Forward::sampler`](crate::forward::Forward::sampler).
pub mod sampling {
    pub use sampling_edsl::*;
}

/// Device tensor + tensor-program substrate (the WIT `tensor` interface).
///
/// Exposes the generated `tensor::{Tensor, Program, Op, OpKind, Value, Input,
/// Dtype, Literal, Predicate, RngKind}` bindings — the **front door** the guest
/// emit (`SamplingProgram` → `op-kind`) and program-authoring inferlets build
/// against. A [`Program`](tensor::Program) is binding-free and reusable;
/// attach it (with attach-time input bindings) via
/// [`Forward::sampler`](crate::forward::Forward::sampler).
pub mod tensor {
    pub use crate::pie::core::tensor::*;
}

// =============================================================================
// Generation state machine + decoders + speculation
// =============================================================================

pub mod chat;
pub mod reasoning;
pub mod tools;

pub use tools::Tool;

// =============================================================================
// Model
// =============================================================================

/// The engine serves exactly one model; these are global functions over
/// that single bound model. There is no `Model`/`Tokenizer` handle to pass
/// around — call `model::encode`, `model::name`, etc. directly.
pub mod model {
    pub use crate::pie::core::model::{
        architecture, arena_block_size, decode, default_system_speculation, encode, is_linear,
        name, output_vocab_size, rs_buffer_page_size, rs_fold_granularity, rs_state_size,
        special_tokens, split_regex, vocabs,
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
    pub use crate::messaging;
    pub use crate::model;
    pub use crate::runtime;
    pub use crate::{Result, Schema, Tool};
    pub use crate::{main, tool};

    pub use crate::program::{LoweredProgramExt, ProgramHandle};
    pub use crate::tensor;
    pub use crate::{chat, reasoning, tools};
}
