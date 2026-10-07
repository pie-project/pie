#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![allow(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]

mod error;

pub mod dispatch;
pub mod fire;
pub mod law;
pub mod store;

pub use dispatch::{
    Dispatch, DispatchAttention, DispatchCollective, DispatchCustomCuda, DispatchElementwise,
    DispatchLayout, DispatchLinear, DispatchProbe, DispatchSpatial,
};
pub use error::{Error, KernelError, Result};
