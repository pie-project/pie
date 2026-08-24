#![cfg_attr(docsrs, feature(doc_cfg))]

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

pub use kernels::{Cap, LaunchRule, Lit, Refusal, Source, Ty};

pub use kernels::routine::{In, InOut, Out};

pub mod jit;

pub mod routine;

pub mod source;

pub mod raises;

pub mod driver_internal;

pub mod dist;

pub mod comm;

pub mod tile;

pub mod attn;
pub mod gemm;
pub mod graph;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
/// GENERATED from `kernels::points` × this plane's `*_CLAIMS`; see the
/// file's own header and `tests/points_dispatch_is_current.rs`.
pub mod points_dispatch;
pub mod quant;
pub mod rope;
pub mod ssm;

/// The device-side checks for the kernels this crate declares AND launches
/// itself — the two `#[claims]` bodies that are launchers rather than
/// delegations. Behind `_cuda` because it fires, and a unit module rather
/// than a `tests/` file because `cudarc` is optional and an optional
/// dev-dependency is not a thing Cargo has.
#[cfg(all(test, feature = "_cuda"))]
mod devtest;

#[cfg(feature = "_cuda")]
pub mod tower;
pub mod views;
pub mod vision;

pub type Plane = crate::jit::Cuda;

/// The claims this plane answers by SYMBOL rather than by point.
///
/// TWO ROWS, AND EACH IS A MEASURED BACKLOG WITH ITS REASON WRITTEN DOWN.
/// `model_compiler::sweep::resolve` asks every kernel a plan names, in this
/// order: a `cuda::` prefix is tier-2, a name in a family's `*_CLAIMS` is a
/// point, and anything left is asked of this table. Across all sixteen
/// catalog rows exactly two claims get here:
///
/// * `norm.res_blend` — kimi's variadic ledger item. The text states one
///   value per earlier block and the count grows with the layer, so the
///   statement's arity is a function of where it stands; the floor has no
///   `Vararg` mark to declare that with.
/// * `hc.collapse` — dsv4's head-gate collapse. The kernel reads an
///   `[N, streams]` f32 gate plane beside the residual stack, NO TEXT
///   PRODUCES ONE, and the import ships no bank it could come from, so both
///   honest readings need a checkpoint dsv4 does not have.
///
/// Each names the launcher that would fire it, spelled as the plane's own
/// `module::fn`. Nothing FIRES through this table — a `Call::Symbol` refuses
/// at load, because the staging its fire would need does not exist — so what
/// a row buys is that the claim reports RESOLVED rather than unclaimed, and
/// deleting one silently loses a resolution per lane.
///
/// A TABLE AND NOT A COLUMN ON A ROUTINE ROW. This was the `canon` field of a
/// `#[routine]`, one of thirteen columns on a linkme-collected registry that
/// four crates carried so that this one question could be asked of it. The
/// routine layer is folded — every launch is a `#[claims]` body or a function
/// beside one — and two `(claim, symbol)` pairs are what was left of it.
pub const CANON: &[(&str, &str)] = &[
    ("hc.collapse", "norm::hc_head_postprocess"),
    ("norm.res_blend", "attn::attn_res_blend"),
];

#[cfg(feature = "_cuda")]
#[cfg_attr(docsrs, doc(cfg(any(feature = "cuda-12", feature = "cuda-13"))))]
pub use jit::Error;

pub use jit::ArgValue;

pub use crate::jit::abi::Pointee as RoutineElem;
