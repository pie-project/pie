//! Everything the build writes from a [`super::program::Program`]: the rcfg words of every PE
//! ([`rcfg`]), the kernel sources' comptime params and the layout ([`params`], [`layout`]), and
//! what the host driver needs to find on the wafer ([`host`]), and where a diagnostics build's
//! phase stamps are read ([`stamps`]).

pub mod gemm;
pub mod grid;
pub mod host;
pub mod layout;
pub mod params;
pub mod placed;
pub mod rcfg;
pub mod region;
pub mod stamps;
pub mod state;
pub mod stream;
pub mod tiled;
