//! L0: the driver ABI surface — concrete backend dispatch plus the
//! `DriverSpec`/`DriverBackend` registry (`backend`), completion delivery
//! (`completion`), runtime-owned command/plan types (`command`), the
//! launch wire request (`submission`), borrowed ABI marshalling (`abi`),
//! channel lifecycle (`channel`), and instance binding (`instance`).
//! Strictly leaf: no `crate::{store,scheduler,pipeline,inferlet,server}`
//! imports. The per-`driver_id` dispatch verbs (`register_program`,
//! `bind_instance`, the `copy_*` family, ...) live in
//! the scheduler dispatch facade instead of here, because they need its
//! driver-id -> handle registry to reach the `BatchScheduler` that owns a
//! given driver instance.

pub mod abi;
pub mod backend;
pub mod channel;
pub mod command;
pub mod completion;
pub mod instance;
pub mod submission;

pub use pie_waker as waker;

pub use backend::{
    DriverBackend, DriverSpec, DummyDriver, SchedulerLimits, get_spec, register_driver,
    register_driver_backend, take_driver_backend,
};
pub use channel::{ChannelCloser, ChannelEndpoint, ChannelValue, RegisteredChannel};
pub use command::{
    ChannelRegistrationPlan, KvCopyPlan, LaunchPlan, PoolResizePlan, ProgramRegistration,
    RS_FLAG_FOLD, RS_FLAG_RESET, StateCopyPlan,
};
pub(crate) use completion::WorkItemAttemptOutcome;
pub use completion::{CompletionBroker, SubmissionCompletion, WorkItemCompletion};
pub use instance::{BoundInstance, InstanceBindingPlan, InstanceId, ProgramId};
pub use submission::LaunchSubmission;

pub type DriverId = usize;

pub async fn generate_audio(
    _driver_idx: DriverId,
    _prompt: &[u32],
    _max_frames: u32,
) -> anyhow::Result<Vec<f32>> {
    Err(anyhow::anyhow!(
        "generate_audio is not wired to driver backends yet"
    ))
}
