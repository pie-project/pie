//! Everything the build writes from a [`super::program::Program`]: the rcfg words of every PE
//! ([`rcfg`]), the kernel sources' comptime params and the layout ([`params`], [`layout`]), and
//! what the host driver needs to find on the wafer ([`host`]).

pub mod host;
pub mod layout;
pub mod params;
pub mod rcfg;
