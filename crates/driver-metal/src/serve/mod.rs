//! The transfers the engine asks for.
//!
//! `driver-cuda` spells this `serve/transfer.rs` and this crate spelled it
//! `store/control.rs`; they are one module under two names, which is what
//! `.wiki/driver/real-metal-north-star.md` §5's dictionary exists to stop.

pub mod control;
pub mod launch;
pub mod load;
pub mod register;
pub mod state;
pub mod transfer;

pub use launch::Launched;
pub use state::Shell;
pub use transfer::{
    Capabilities, KvCopyWork, Pool, Refusal, Resize, plan_kv_copy, plan_pool_resize,
    plan_state_copy,
};
