//! Pure CUDA kernel definitions: jit unit names, argument marshalling, and
//! launch geometry over a stream; no IR types, no execution state.

pub mod attn;

// Vision towers' dense attention; belongs beside the attention family but
// declared via `#[path]` since `src/attn.rs` is closed. Retire by moving this
// to `pub mod dense;` inside `attn.rs`.
#[path = "attn/dense.rs"]
pub mod attn_dense;

// Patch axis row folds; same `#[path]` detour as `attn_dense`, one family
// over. Retire via `pub mod fold;` inside `layout.rs`.
#[path = "layout/fold.rs"]
pub mod layout_fold;

// Scatter honoring a drop sentinel; a second entry beside `layout::scatter_rows`
// rather than a widening of it. Retire via `pub mod scatter_live;` inside `layout.rs`.
#[path = "layout/scatter_live.rs"]
pub mod layout_scatter_live;

// Weighted-gather interpolation; a second entry beside `layout::embed` rather
// than a widening of it. Retire via `pub mod embed_weighted;` inside `layout.rs`.
#[path = "layout/embed_weighted.rs"]
pub mod layout_embed_weighted;

// Observability score capture, re-homed for the same reason as `attn_dense`.
// Retire via `pub mod score;` inside `attn.rs`.
#[path = "attn/score.rs"]
pub mod attn_score;

// qwen4's PLE n-gram hasher, beside the recurrent mixers whose state
// discipline it follows. Retire via `pub mod ple;` inside `attn.rs`.
#[path = "attn/ple.rs"]
pub mod attn_ple;

// qwen4's PLE n-gram embedding gather-concat; a second entry beside
// `layout::embed`. Retire via `pub mod embed_concat;` inside `layout.rs`.
#[path = "layout/embed_concat.rs"]
pub mod layout_embed_concat;

pub mod channel;
pub mod collective;
pub mod custom;
pub mod disk;
pub mod elemwise;
pub mod error;
pub mod graph;
pub mod jit;
pub mod layout;
pub mod linear;
pub mod seat;
pub mod source;
pub mod tensor;

pub use error::Error;
pub use jit::{Arg, ArgValue, Ctx, Fire, Launch, Pad, Slabs};
pub use seat::{ENTRIES, EntryInfo, Lanes, Reads, Routes, Rows};

// Re-exported so a `tests/` target (a separate crate) can reach the exact
// `cudarc` this crate's `cuda` feature resolved, rather than pulling in its
// own copy via a dev-dependency.
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use cudarc;

pub use tensor::{KvPool, RaggedTensor, RecurrentPool, Tensor};
