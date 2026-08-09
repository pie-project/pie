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
// GATED ON `abi`, and that is a finding rather than a tidy-up. The
// forward pass takes `driver_api::PieFrameDesc` and reads
// `serve::state::Shell`, so `fire/` — which the tree calls "one forward
// pass: its scratch, its tables, its recordings, its retirement" —
// cannot be built without the door. §6's middle build spelling
// (`--features cuda-13`, cudarc only, no toolkit) did not work before
// this line, and it is the build a CI without a card would run.
//
// The right fix is that a fire is described by a value this crate owns
// rather than by the ABI's struct, which is §3.2's move applied one
// layer further in. Until then the gate says where the seam actually
// is, instead of a build error saying it three ways.
#[cfg(feature = "abi")]
pub mod launch;
pub mod lora;
pub mod page_mask;
pub mod recordings;
pub mod scratch;
pub mod sideband_arena;
pub mod stage_hooks;
