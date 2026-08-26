//! `Linear`: the family that multiplies by a weight — dense projections,
//! gated mlp activations over a packed row, the moe routers and their routed
//! matmuls, and the jit-stamped affine quant points. One module per shape of
//! weight; the entries inside are one per IR variant.

pub mod gemm;
pub mod mlp;
pub mod moe;
pub mod quant;
