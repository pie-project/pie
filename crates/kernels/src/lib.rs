//! The contract between the IR and the engines: a `Dispatch*` trait per op
//! family, the aggregate over a [`Node`](model_ir::Node), and the
//! [`KernelError`] a backend may answer with. No kernels, no execution
//! state, and no walk — the one loop written over this contract lives in the
//! engine substrate (`engine::fire::walk`), and the prepare/capture split is
//! the model compiler's (`.wiki/palo/design.md`, decisions #11–#12).

pub mod dispatch;
pub mod error;

pub use dispatch::{
    Dispatch, DispatchAttention, DispatchCollective, DispatchCustomCuda, DispatchElementwise,
    DispatchLayout, DispatchLinear,
};
pub use error::KernelError;
