//! One forward pass: its scratch, its tables, its recordings.
//!
//! A fire is the unit this shell exists to serve — a batch of rows
//! through a lowered program — and everything here has that lifetime or
//! is pooled across it. [`launch`] is the pass itself; the rest is what
//! it needs standing before it can run.
//!
//! The pooling is not an optimisation. A recorded graph BAKES an
//! address, so a buffer that moved between fires would be replayed
//! against memory that is no longer there; [`scratch::Scratch`] exists
//! so that a fire's addresses are the same as the last fire's.

pub mod attention_workspace;
pub mod attn_score;
pub mod launch;
pub mod lora;
pub mod page_mask;
pub mod recordings;
pub mod scratch;
pub mod sideband_arena;
pub mod stage_hooks;
