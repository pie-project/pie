//! Pluggable data-plane backends: [`local`] for co-located prefill+decode
//! (device-to-device copy), `nixl` for cuda/rocm cross-node behind
//! `feature = "nixl"`. All satisfy [`crate::core::Backend`].

pub mod local;

#[cfg(feature = "nixl")]
pub mod nixl;
