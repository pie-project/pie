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
pub mod feeds;
pub mod fire;
pub mod law;
pub mod store;

pub use dispatch::{
    Dispatch, DispatchAttn, DispatchCollective, DispatchConv, DispatchElemwise, DispatchFused,
    DispatchGather, DispatchLinear, DispatchProbe, DispatchSampling, DispatchSsm,
};
pub use error::{Error, KernelError, Result};
