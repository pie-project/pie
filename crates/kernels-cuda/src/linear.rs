//! `Linear`: everything that multiplies by a weight — dense projections,
//! gated activations over one, the routed fan-out, and the quantizers that
//! stage the banks they read. One submodule per member of the family; the
//! entries inside keep one entry per IR variant.

#[cfg(feature = "_cuda")]
mod dense;

pub mod gemm;

#[cfg(feature = "_cuda")]
mod gemv;

pub mod mlp;

pub mod moe;

pub mod quant;
