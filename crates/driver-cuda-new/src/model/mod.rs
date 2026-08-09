//! Model-level driver objects: the per-fire state the forward path writes
//! through, as distinct from the kernels it launches.
//!
//! The generated forward bodies in `driver-cuda/csrc/src/model/*/generated`
//! are only 14% kernel launches; the other 86% are calls on objects like these.
//! Porting them is what lets the emitter target Rust.

pub mod attention_workspace;
pub mod attn_score;
pub mod config;
pub mod descriptor;
pub mod executor;
pub mod llama_like;
pub mod lora;
pub mod page_mask;
pub mod qwen3_5;
pub mod sideband_arena;
pub mod stage_hooks;
pub mod weight_bind;
pub mod weight_view;
pub mod workspace;
