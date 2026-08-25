#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::missing_safety_doc)]

#[cfg(all(feature = "_cuda", not(any(feature = "cuda-12", feature = "cuda-13"))))]
compile_error!(
    "kernels-cuda's runtime needs exactly one CUDA runtime version: \
     enable `cuda-12` or `cuda-13`, matching the libcudart this binary will load"
);

#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "kernels-cuda: `cuda-12` and `cuda-13` are mutually exclusive -- a binary \
     loads one libcudart, and the two disagree on `cudaGraphAddNode`'s arity"
);

pub use kernels::Refusal;

pub use kernels::plane::{In, InOut, Out};

pub mod jit;

pub mod source;

pub mod raises;

pub mod driver_internal;

pub mod dist;

pub mod comm;

pub mod attn;
pub mod gemm;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;

pub mod points_dispatch;
pub mod quant;
pub mod rope;
pub mod ssm;

#[cfg(all(test, feature = "_cuda"))]
mod devtest;

#[cfg(feature = "_cuda")]
pub mod tower;
pub mod views;
pub mod vision;

pub const CANON: &[(&str, &str)] = &[
    ("hc.collapse", "norm::hc_head_postprocess"),
    ("norm.res_blend", "attn::attn_res_blend"),
];

#[cfg(feature = "_cuda")]
#[cfg_attr(docsrs, doc(cfg(any(feature = "cuda-12", feature = "cuda-13"))))]
pub use jit::Error;

pub use jit::ArgValue;

pub use crate::jit::abi::Pointee as RoutineElem;
