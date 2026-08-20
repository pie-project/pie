//! The Metal execution shell, in Rust.
//!
//! The crate splits into a portable half ([`layout`], [`lowering`], [`batch`],
//! [`model`]) that compiles and tests on any host, and a device half
//! (`device`, `weights`, `pools`, `bind`, `fire`, `program`, `serve`) gated
//! behind the `metal-4` feature, which requires an Apple target. The gate is
//! a Cargo feature rather than `cfg(target_vendor = "apple")` because a
//! platform cfg cannot be exercised in CI on a non-Apple runner, while a
//! feature can be built either way from the same job.
//!
//! Metal objects are held as `Retained<ProtocolObject<dyn _>>`, so retain and
//! release are automatic rather than the caller's responsibility.

#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]

pub mod batch;
pub mod channel;
mod error;
pub mod envelope;
pub mod layout;
pub mod loader;
pub mod lowering;
pub mod model;

pub use error::{Error, Result};
pub use layout::{Batch, Region, Request};

#[cfg(all(feature = "metal-4", not(target_vendor = "apple")))]
compile_error!(
    "`metal-4` needs an Apple target: Metal has no implementation elsewhere. \
     Build without it for the half of this crate that answers questions no \
     GPU changes."
);

#[cfg(feature = "metal-4")]
pub mod bind;
#[cfg(feature = "metal-4")]
pub mod device;
#[cfg(feature = "metal-4")]
pub mod fire;
#[cfg(feature = "metal-4")]
pub mod pools;
#[cfg(feature = "metal-4")]
pub mod program;
#[cfg(feature = "metal-4")]
pub mod serve;
#[cfg(feature = "metal-4")]
pub mod weights;

// A module belongs in the device half only if answering its question
// requires a device (`layout::tuning` is GPU-related but device-free, so it
// stays portable). These modules are declared flat rather than nested under
// a wrapping `gpu` module, which would add a redundant path segment and let
// re-exports duplicate across `gpu::X` and `device::X`. `tests/layering.rs`
// enforces the one case the compiler cannot: a self-contained device module
// declared outside this gated set.
//
// # `unsafe`
//
// Every objc2 message send is `unsafe`, so this half cannot carry the
// workspace's `unsafe_code = "forbid"`. Each `unsafe` block must instead
// state the invariant it is relying on.
