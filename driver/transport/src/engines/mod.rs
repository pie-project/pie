//! Pluggable data-plane engines.
//!
//! Backends are asymmetric, so the mover is too:
//!   * co-located prefill+decode → [`local`] (device-to-device copy, zero
//!     network) — the minimal baseline,
//!   * cuda/rocm cross-node → `nixl` (RDMA/TCP/NVMe via NIXL), deferred behind
//!     `feature = "nixl"`,
//!   * metal/vulkan never participate (single-node; NIXL is Linux-only).
//!
//! All engines satisfy [`crate::core::Engine`]; the [`crate::registry`] binds a
//! handle to one and dispatches.

pub mod local;

#[cfg(feature = "nixl")]
pub mod nixl;
