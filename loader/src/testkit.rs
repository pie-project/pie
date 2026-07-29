//! The parts of this crate that exist to *check* the loader rather than to be
//! the loader.
//!
//! Three modules, one feature gate, one reason: none of them is reachable from
//! [`crate::ffi`], so a driver that links this crate has no use for them.
//! `worker/Cargo.toml` turns `testkit` off, which is also the check — if
//! `ffi/` ever grows a path into the oracle, the worker stops compiling.
//!
//! * [`host_executor`] runs a finished plan on the CPU against real bytes;
//! * [`mod@reference`] is the differential oracle it is compared against;
//! * [`contract_writer`] builds the POD contracts the FFI tests feed in.
//!
//! They were three separate `#[cfg(feature = "testkit")]` lines at the crate
//! root, which put the gate in three places and said nothing about why the
//! three belonged together.

pub mod contract_writer;
pub mod host_executor;
pub mod reference;
