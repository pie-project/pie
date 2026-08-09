//! Checkpoint to device.
//!
//! The plan half — which tensor lands where, and how many bytes that is —
//! is `loader::plan` and needs no card. What is here is the part
//! that does: the heap slots, the staging copies, and the registry the
//! executor resolves a name against.
//!
//! * [`load`] — the call between `loader/` and [`stage`].
//! * [`stage`] — the decode step's resident storage: weights, KV, GDN state,
//!   IO and the scratch pool, allocated and staged.

pub mod load;
pub mod stage;

pub use stage::stage_plan_weights;
