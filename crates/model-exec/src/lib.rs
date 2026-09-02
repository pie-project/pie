//! The model forward substrate: what the host does about one fire that is
//! not a device call, plus the model-state geometry and the seat law under
//! it.
//!
//! Three planes, each navigated by path rather than flattened into the
//! crate root: `fire::compose`, `fire::walk`, `store::kv`, `law::fit`. A
//! reader who sees `fire::walk` at a call site knows where to go and what
//! it is about.
//!
//! This crate does not know what an `Engine` is: the runtime<->engine
//! contract is a separate crate, nor does it name the ETA toolchain
//! (`eta-exec`). What is left is the model toolchain — `model-ir` in,
//! `model-compiler`'s artifact walked, and the dispatch contract in
//! [`dispatch`] called — and nothing else. The six `Dispatch*` traits and
//! `KernelError` live here, where their one caller is; the two kernel
//! libraries keep an `Error` of their own.
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
// overrides the workspace's deny(missing_docs): modules here document
// themselves by convention, not by lint.
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
    DispatchLayout, DispatchLinear,
};
pub use error::{Error, KernelError, Result};
