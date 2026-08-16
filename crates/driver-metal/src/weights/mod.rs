//! Checkpoint to device. The plan half — which tensor lands where, how many
//! bytes — is `loader::plan` and needs no card. [`stage`] is the decode step's
//! resident storage: weights, KV, GDN state, IO and the scratch pool.

pub mod load;
pub mod stage;

pub use stage::stage_plan_weights;
