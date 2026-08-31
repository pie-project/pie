//! `Linear`: everything that multiplies by a weight — dense projections,
//! gated activations over one, the routed fan-out, the low-rank correction
//! that rides on a materialised output, and the quantizers that stage the
//! banks they read. One submodule per member of the family; the entries
//! inside keep one entry per IR variant.

#[cfg(feature = "_cuda")]
mod dense;

pub mod gemm;

#[cfg(feature = "_cuda")]
mod gemv;

/// The correction class: `y += B[a]·(A[a]·x)` over a routed adapter bank.
pub mod lora;

pub mod mlp;

pub mod moe;

pub mod fp8;

pub mod kquant;

pub mod nvfp4;

pub mod quant;
