//! The model forward substrate: what the host does about one fire that is not
//! a device call, plus the model-state geometry and the seat law under it.
//!
//! Three planes, and each is navigated by path rather than flattened into the
//! crate root: `fire::compose`, `fire::walk`, `store::kv`, `law::fit`. A
//! reader who sees `fire::walk` at a call site knows where to go and what it
//! is about. That was a deliberate choice made when this crate still shared a
//! roof with the guest-program plane — 22 files of one subsystem that HAD
//! flattened itself, which is the shape not worth extending — and it survives
//! the parting unchanged.
//!
//! **It does not know what an `Engine` is.** The runtime↔engine contract is a
//! separate crate and this one does not name it; nor does it name the ETA
//! toolchain, whose execution substrate is `eta-exec`. What is left is the
//! model toolchain — `model-ir` in, `model-compiler`'s artifact walked, and
//! the dispatch contract in [`dispatch`] called — and nothing else.
//!
//! **THE CONTRACT IS THIS CRATE'S NOW.** The six `Dispatch*` traits and
//! `KernelError` were `crates/kernels`, a 141-line crate that was two
//! unrelated things sharing a manifest; the traits came here, where their one
//! caller is, and the two kernel libraries kept an `Error` of their own.
//! [`dispatch`]'s header states why that is not the layering inversion palo
//! decisions #11–#12 ruled out.
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
// `deny(missing_docs)` stood here, and the workspace lints table still says
// deny; the allow below overrides it. It was the guest-program plane's
// stripped prose that forced a RUSTFLAGS override onto every consumer build,
// and that plane is a different crate now — but the override is still cheaper
// to keep than to re-litigate module by module, and the modules here document
// themselves by convention, not by threat.
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
