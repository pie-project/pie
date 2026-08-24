//! The Metal execution shell, in Rust.
//!
//! The crate splits into a portable half ([`baker`], [`layout`], [`model`],
//! [`envelope`]) that compiles and tests on any host, and a device half
//! (`device`, `pools`, `bind`, `fire`, `program`, `serve`) gated behind the
//! `metal-4` feature, which requires an Apple target. The gate is a Cargo
//! feature rather than `cfg(target_vendor = "apple")` because a platform cfg
//! cannot be exercised in CI on a non-Apple runner, while a feature can be
//! built either way from the same job.
//!
//! Metal objects are held as `Retained<ProtocolObject<dyn _>>`, so retain and
//! release are automatic rather than the caller's responsibility.
//!
//! # The executor is in the portable half, and that is the P5 design
//!
//! [`baker`] is this driver's whole answer to a fire: it takes a lane's
//! `model_compiler::program::Program`, walks its steps, binds each
//! statement's columns and hands them to
//! `kernels_metal::points_dispatch::dispatch`. Not one line of it names a
//! Metal type, because what a claim body talks to is
//! `kernels_metal::routine::Ctx` — which IS `dyn Encode`, a trait the driver
//! implements. So the device is behind that `dyn`, and the walk's order, the
//! points it asks for and the handles it binds are all checkable with no GPU
//! in the process.
//!
//! What is genuinely device-shaped is what sits BEHIND the door: turning a
//! [`baker::dispatch::Dispatch`] into `dispatchThreads:`, and owning the
//! buffers a [`baker::Slice`] addresses.
//!
//! `lowering` STOOD BESIDE `layout` and was the legacy walk: a `Lowered` list
//! of launches naming kernel SYMBOLS, matched against a by-name registry of
//! routine stems, with a grid planner reading each row's launch rule. All of
//! it went at P5 — `model_compiler::lower` is deleted, `#[routine]` is
//! deleted, and a grid is computed inside the claim body that fires it.
//! `batch` went with it (a `DecodeGeometry` projected from a catalog row is
//! `model::deployment::Deployment` now, read off the same plan the program is
//! built from), and so did `loader` and `weights` (the legacy load contract's
//! plan author and its stager; weights arrive through `model::produce`).

#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]

pub mod baker;
pub mod channel;
pub mod envelope;
mod error;
pub mod layout;
pub mod model;
pub mod skip;
pub mod walk;

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
