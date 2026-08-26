//! The contract between the IR and the drivers, and the one walk written
//! over it: a `Dispatch*` trait per op family, the aggregate over a
//! [`Node`](model_ir::Node), [`KernelError`], and the prepare/capture walk
//! in [`exec`]. No kernels and no execution state live here.

pub mod dispatch;
pub mod error;
pub mod exec;

pub use dispatch::{
    Dispatch, DispatchAttention, DispatchCollective, DispatchCustomCuda, DispatchElementwise,
    DispatchLayout, DispatchLinear,
};
pub use error::KernelError;
pub use exec::{Phases, fire, phases, walk};
