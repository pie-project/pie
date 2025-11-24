//! Test application that returns the Pie engine version.
//!
//! This program calls `inferlet::get_version()` to retrieve and return the Pie engine's
//! version string.
//!
//! Used for testing Pie runtime API access and instrumentation.

use inferlet::{Args, Result};

#[inferlet::main]
async fn main(_: Args) -> Result<String> {
    Ok(inferlet::get_version())
}
