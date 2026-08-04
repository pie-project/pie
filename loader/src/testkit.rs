//! The parts of this crate that exist to *check* the loader rather than to be
//! the loader.
//!
//! Two modules, one feature gate, one reason: neither is on the load path, so
//! a driver that links this crate has no use for them. `worker/Cargo.toml`
//! turns `testkit` off, which is also the check — if the load path ever grows
//! a route into the oracle, the worker stops compiling.
//!
//! * [`host_executor`] runs a finished plan on the CPU against real bytes;
//! * [`mod@reference`] is the differential oracle it is compared against.
//!
//! The POD contract fixture builder that used to sit beside them followed the
//! ABI into `pie-loader-capi` and then out of existence: contracts stopped
//! crossing the ABI when the last C++ author was harvested
//! (`plan/model-in-rust.md` §8-5), so there is no POD graph left to write.

pub mod reference;

/// The executor moved to [`crate::executor::host`] when `pie model convert`
/// became its production caller; this alias keeps the old path reading while
/// callers migrate.
pub use crate::executor::host as host_executor;
