//! L0: driver selection and the registry — the `DriverSpec`/`DriverBackend`
//! store (`backend`), the concrete seams behind it, channel endpoint
//! lifecycle (`channel`), and the launch-side re-exports the rest of the
//! engine reads.
//!
//! # What this layer is now
//!
//! It is the ENGINE's half of the driver boundary and nothing else. The
//! contract — [`Driver`](driver_api::Driver) and the fourteen verbs, the
//! completion a driver mints, the bind plan, the registration a driver
//! answers — is `driver-api`'s, because both sides say it. What is left here
//! is what only the engine does: pick a backend, keep it in a registry under
//! a `DriverId`, and hold the channel endpoints applications wait on.
//!
//! Four modules went in that split, and it is worth saying which:
//!
//! * `completion` and `instance` moved to `driver-api`. A driver MINTS a
//!   completion — five of the trait's verbs return one — and answers a
//!   `BoundInstance`, so neither could stay in a crate the contract cannot
//!   name.
//! * `command` and `submission` were seven-line `pub use` shims for types
//!   `driver-api` already owned. They are deleted; consumers name the owner.
//! * `abi` marshalled runtime-owned plans into `#[repr(C)]` descriptors so a
//!   driver could take them apart again. Its own header had already deleted
//!   the frame for that reason ("the round trip bought nothing but the loss
//!   of the types"); the other two descriptors went the same way when the
//!   CUDA seam stopped being a C ABI.
//!
//! Strictly leaf: no `crate::{store,scheduler,pipeline,inferlet,server}`
//! imports. That claim was here before and was false — `backend.rs` called
//! `crate::pipeline::program::lookup` to splice host-generated kernels into a
//! registration. The splice is
//! [`crate::pipeline::program::with_host_codegen`]'s now, called by the
//! scheduler that owns the driver handle, which is also the layer that knows
//! which driver a plan is bound for.
//!
//! The per-`driver_id` dispatch verbs (`register_program`, `bind_instance`,
//! the `copy_*` family, ...) live in the scheduler dispatch facade, because
//! they need its driver-id -> handle registry to reach the `BatchScheduler`
//! that owns a given driver instance.

pub mod backend;
pub mod boot;
pub mod channel;

pub use waker;

pub use backend::{
    DriverBackend, DriverSpec, RemoteDisconnectHandle, RemoteDriver, SchedulerLimits, get_spec,
    open, register_driver_backend, take_driver_backend, unregister_driver,
};
pub use boot::BootConfig;
pub use channel::{ChannelCloser, ChannelEndpoint, ChannelValue, RegisteredChannel};

// The contract, re-exported at the path the engine already reads it from.
//
// Not a convenience: `crate::driver::FrameLaunchOutcome` and
// `crate::driver::SubmissionCompletion` are named across the scheduler, the
// pipeline and the tests, and every one of those names a thing the DRIVER
// decides. Re-exporting is what let the contract move out of this crate
// without touching them.
pub use ::driver_api::completion::WorkItemAttemptOutcome;
// `CompletionLease` and `CompletionTarget` are NOT among these, and the
// difference is the point of the paragraph above. Both are read — a lease by
// `BoundInstance`, a target by the four seams — but neither is read THROUGH
// this path: `driver-api` names them itself, seven times for the target and
// twice for the lease, and `crate::driver::CompletionTarget` appears nowhere.
// A re-export nobody reads is a wall in front of a door nobody opens.
pub use ::driver_api::completion::{CompletionBroker, SubmissionCompletion, WorkItemCompletion};
pub use ::driver_api::instance::{BoundInstance, InstanceBindingPlan, InstanceId, ProgramId};
pub use ::driver_api::plan::{
    ChannelRegistrationPlan, KvCopyPlan, LaunchPlan, PoolResizePlan, ProgramRegistration,
    RS_FLAG_BUFFER_WRITE, RS_FLAG_FOLD, RS_FLAG_FOLD_LEN_DEVICE, RS_FLAG_RESET, StateCopyPlan,
};
pub use ::driver_api::submission::{FrameSubmission, StepSubmission};
pub use ::driver_api::{Driver, FrameLaunchOutcome, Unsupported};

/// Which driver, as the registry addresses it.
pub type DriverId = usize;

/// Not wired to any backend.
///
/// Stays a named refusal rather than an absence for the reason the seams
/// give: a verb that cannot be reached teaches nothing, and one that says
/// exactly what is missing is a working door with a stated hole.
///
/// # Errors
///
/// Always.
pub async fn generate_audio(
    _driver_idx: DriverId,
    _prompt: &[u32],
    _max_frames: u32,
) -> anyhow::Result<Vec<f32>> {
    Err(anyhow::anyhow!(
        "generate_audio is not wired to driver backends yet"
    ))
}
