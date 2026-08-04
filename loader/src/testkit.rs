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
//! ABI into `pie-loader-capi` (`capi/src/contract_writer.rs`): the graph it
//! writes is the ABI crate's vocabulary, not the compiler's.

pub mod host_executor;
pub mod reference;
