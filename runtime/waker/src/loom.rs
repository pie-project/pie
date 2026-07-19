//! Sync-shim — the single `cfg(loom)` swap point (B9/B11).
//!
//! Under `--cfg loom` the model checker's primitives explore every
//! interleaving of the register/commit race; in normal builds these are `std`.
//! Every other module imports its sync types from here, so the crate carries
//! the loom swap in exactly ONE place.

#[cfg(loom)]
pub(crate) use ::loom::sync::{
    Mutex, RwLock,
    atomic::{AtomicU64, Ordering},
};
#[cfg(not(loom))]
pub(crate) use std::sync::{
    Mutex, OnceLock, RwLock,
    atomic::{AtomicU64, Ordering},
};
