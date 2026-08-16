//! What one fire keeps between steps: [`scratch`], reusable scratch regions so
//! a fire's addresses are the same as the last fire's; [`recordings`], the
//! recorded fires kept by what they are valid for; and [`run`], the four calls
//! in one place — allocate the arena the lowering asked for, plan, compile,
//! encode.

pub mod recordings;
pub mod run;
pub mod scratch;

pub use recordings::{Recordings, fingerprint};
pub use scratch::{Lease, Scratch};
