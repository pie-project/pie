//! The contract between the IR and the drivers: one `Dispatch*` trait per op
//! family, the `Dispatch` aggregate over a [`Node`](new_model_ir::Node), and
//! [`KernelError`]. No kernels and no execution state live here.

pub mod dispatch;
pub mod error;

pub use dispatch::{
    Dispatch, DispatchAttention, DispatchCuda, DispatchDist, DispatchGate, DispatchGemm,
    DispatchHc, DispatchIndex, DispatchLayout, DispatchMla, DispatchMlp, DispatchMoe, DispatchNorm,
    DispatchPool, DispatchRope, DispatchSsm,
};
pub use error::KernelError;
