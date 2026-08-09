//! Compatibility re-exports for the sealed frame, now owned by `driver-api`.
//!
//! Same shape as [`command`](super::command): the type states what a driver's
//! `launch` verb takes, so it belongs beside the other driver plans rather
//! than in the crate that calls them.

pub use ::driver_api::submission::{FrameSubmission, StepSubmission};
