//! Model-level driver objects: the per-fire state the forward path writes
//! through, as distinct from the kernels it launches.
//!
//! # No family names here, and how that happened
//!
//! This module held `llama_like` and `qwen3_5` — 1,529 lines of faithful
//! ports of the C++ driver's per-family host objects, including a
//! twenty-seven-buffer linear-attention workspace whose ALLOCATION ORDER
//! its own doc comment called part of the contract.
//!
//! Nothing in the driver ever called them. Not one symbol: not the
//! workspace, not the plan states, not the graph-layout functions, not the
//! fused-decode-post env gate. The executor had already stopped asking,
//! because `lower()` assigns buffers for every value in the graph and a
//! workspace in a stated order is a hand-maintained instance of exactly
//! that. Their only consumers were the parity tests written to prove the
//! ports matched.
//!
//! So the plan's §5.E1/E2 turned out to be a deletion rather than a
//! migration, and the thing that made it one was not this module — it was
//! the lowering. That is the evidence the declaration approach
//! generalises: the family module did not have to be replaced, it had to
//! be noticed.

pub mod attention_workspace;
pub mod attn_score;
pub mod config;
pub mod descriptor;
pub mod executor;
pub mod lora;
pub mod page_mask;
pub mod sideband_arena;
pub mod stage_hooks;
pub mod supergraph;
pub mod weight_bind;
pub mod weight_view;
pub mod workspace;
