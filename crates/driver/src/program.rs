//! The guest-program plane: the PTIR host half.
//!
//! Everything a device shell needs to adopt a launch package, run the channel
//! ring, and derive at bind time — the launch-package adoption, the channel
//! ring, the reference pass, and the op interpreter — with no device API call
//! anywhere below this line. The crate's other planes (`fire`, `store`) sit
//! beside it, not under it.

pub(crate) mod cache;
pub(crate) mod channel;
pub(crate) mod emitted;
pub(crate) mod extent;
pub(crate) mod group;
pub(crate) mod identity;
pub(crate) mod lane;
pub(crate) mod meta;
pub(crate) mod op;
pub(crate) mod params;
pub(crate) mod plan;
pub(crate) mod readiness;
pub(crate) mod registry;
pub(crate) mod resolve;
pub(crate) mod scratch;
pub(crate) mod stage_cache;
pub(crate) mod status;
pub(crate) mod step;
pub(crate) mod value;
