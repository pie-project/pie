//! L0: the driver ABI surface — native-backend dispatch (`backend`), FFI
//! shims (`ffi`), completion delivery (`completion`), wire-frame types
//! (`frame`), and the `DriverSpec`/`NativeDriver` registry (`registry`).
//! Strictly leaf: no `crate::{store,scheduler,pipeline,inferlet,server}`
//! imports. The per-`driver_id` dispatch verbs (`register_program`,
//! `bind_instance`, the `copy_*` family, ...) live in
//! the scheduler dispatch facade instead of here, because they need its
//! driver-id -> handle registry to reach the `BatchScheduler` that owns a
//! given driver instance.

pub mod backend;
mod binding_validation;
pub mod completion;
pub mod ffi;
pub mod frame;
pub mod registry;

pub use pie_waker as waker;

pub use backend::{LocalDriver, NativeDriver};
pub use completion::{Completion, CompletionBroker, InstanceCompletion};
pub use frame::{
    BoundInstance, ChannelCloser, ChannelEndpoint, ChannelRegistrationPlan, ChannelValue,
    InstanceBindingPlan, InstanceId, KvCopyPlan, LaunchPlan, LaunchSubmission, PoolResizePlan,
    ProgramId, ProgramRegistration, RS_FLAG_FOLD, RS_FLAG_RESET, RegisteredChannel, StateCopyPlan,
};
pub use registry::{
    DriverSpec, DummyLocalDriver, SchedulerLimits, get_spec, register_driver,
    register_native_driver, take_native_driver,
};

pub type DriverId = usize;

pub async fn generate_audio(
    _driver_idx: DriverId,
    _prompt: &[u32],
    _max_frames: u32,
) -> anyhow::Result<Vec<f32>> {
    Err(anyhow::anyhow!(
        "generate_audio is not wired to direct local drivers yet"
    ))
}
