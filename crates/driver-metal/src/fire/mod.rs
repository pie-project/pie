//! What one fire keeps between steps.
//!
//! * [`scratch`] — reusable scratch regions, so a fire's addresses are the
//!   same as the last fire's. `driver-cuda` calls the same thing
//!   `FireArrays`; `.wiki/driver/real-metal-north-star.md` §5 asks both
//!   crates to call it this.
//! * [`recordings`] — the recorded fires, kept by what they are valid for.
//!   `driver-cuda` calls the same thing `SupergraphCache`.
//! * [`run`] — the four calls in one place: allocate the arena the lowering
//!   asked for, plan, compile, encode.

pub mod recordings;
pub mod run;
pub mod scratch;

pub use recordings::{Recordings, fingerprint};
pub use scratch::{Lease, Scratch};
