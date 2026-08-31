//! Inferlet program management, instantiation, and process execution.
//!
//! - [`process`]: spawn/list/attach/terminate a running guest instance.
//! - [`program`]: install/add a guest program (WASM component + manifest).
//! - `host`: the `pie:inferlet` WIT boundary (bindgen! + `Host*` impls) —
//!   internal wiring only, never named by external callers.
//! - `linker`/`python`/`sandbox`: component linking, Python guest support,
//!   filesystem/network policy — internal.

pub(crate) mod host;
/// **THE MEDIA CODEC AND THE SPAN DIGEST, RE-EXPORTED** — the two pieces of
/// the media pipe that are the HOST'S half (`model::media`'s dependency rule:
/// the catalog does arithmetic, the host decodes and hashes), surfaced so the
/// whole-pipe gate in `tests/media_pipe_is_the_pinned_preprocessing` can
/// compose them exactly as `image.from-bytes` does. The guest boundary
/// (`host`) itself stays private.
pub use host::media::{decode as media_codec, span_digest};
pub(crate) mod linker;
pub mod process;
pub mod program;
pub(crate) mod python;
pub(crate) mod sandbox;

pub use process::ProcessId;
pub(crate) use process::{ProcessCtx, ProcessEvent};
pub(crate) use program::Manifest;
pub use program::ProgramName;
pub(crate) use sandbox::InstancePolicy;
