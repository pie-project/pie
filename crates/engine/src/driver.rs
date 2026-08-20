//! L0: driver selection and the registry — the `DriverSpec`/`DriverBackend`
//! store (`backend`), the concrete seams behind it, channel endpoint lifecycle
//! (`channel`), and the launch-side re-exports the rest of the engine reads.
//!
//! This is the ENGINE's half of the driver boundary and nothing else. The
//! contract — [`Driver`](driver_api::Driver) and the fourteen verbs, the
//! completion a driver mints, the bind plan, the registration a driver answers —
//! is `driver-api`'s, because both sides say it. What only the engine does is
//! pick a backend, keep it in a registry under a `DriverId`, and hold the
//! channel endpoints applications wait on.
//!
//! **Strictly leaf**: no `crate::{store,scheduler,pipeline,inferlet,server}`
//! imports. Splicing host-generated kernels into a registration is
//! [`crate::pipeline::program::with_host_codegen`]'s, called by the scheduler
//! that owns the driver handle and knows which driver a plan is bound for. The
//! per-`driver_id` dispatch verbs (`register_program`, `bind_instance`, the
//! `copy_*` family) live in the scheduler dispatch facade for the same reason:
//! they need its driver-id -> handle registry to reach the `BatchScheduler`.

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
pub use ::driver_api::completion::WorkItemAttemptOutcome;
pub use ::driver_api::completion::{
    CompletionBroker, CompletionLease, CompletionTarget, SubmissionCompletion, WorkItemCompletion,
};
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
/// A named refusal rather than an absence: a verb that cannot be reached teaches
/// nothing, and one that says what is missing is a door with a stated hole.
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
