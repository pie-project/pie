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
pub mod channel;
pub mod completion;
pub mod fire;
pub mod instance;
pub mod load;

pub use waker;

pub use backend::{
    DriverBackend, DriverSpec, RemoteDisconnectHandle, RemoteDriver, SchedulerLimits, get_spec,
    open, register_driver_backend, take_driver_backend, unregister_driver,
};
#[cfg(feature = "_driver-cuda")]
pub use backend::envelopes_resolved;
#[cfg(feature = "_driver-cuda")]
pub use backend::fold_observed;
pub use channel::{
    ChannelBinding, ChannelCloser, ChannelEndpoint, ChannelJoin, ChannelValue, RegisteredChannel,
};

// THE BROKER CAME HOME (palo design §7, decision 19). `CompletionBroker`,
// `SubmissionCompletion`, `WorkItemCompletion` and the terminal cell were
// 807 lines inside `driver-api`, describing how the ENGINE runs ahead of a
// device. They are `driver::completion` and `driver::instance` now, and the
// contract keeps only the receipt — `FireTicket`.
pub use completion::{
    CompletionBroker, CompletionLease, CompletionTarget, SubmissionCompletion, TerminalCell,
    WorkItemAttemptOutcome, WorkItemCompletion,
};
pub use instance::{BoundInstance, BoundWaitSlots, InstanceBindingPlan, InstanceId, ProgramId};

// The engine's own submission vocabulary. `LaunchPlan` and its sixty-two
// parallel CSR arms are gone; what a request IS lives in `fire`, and what
// crosses the boundary is the contract's own `Lane`.
pub use fire::{
    ChannelTicket, FireRequest, FrameFire, MaskWords, Media, RsPlan, StepFire, bitmask_words,
};

// The contract, re-exported at the path the engine already reads it from.
pub use ::driver_api::adapter::{AdapterPlane, AdapterRegistration};
pub use ::driver_api::caps::Capabilities;
pub use ::driver_api::channel::ChannelRegistration;
pub use ::driver_api::error::{DriverError, Result as DriverResult};
pub use ::driver_api::fire::{
    Attachment, Boundary, FireSubmission, FireTicket, KvDelta, Lane, LaneReadout, Mask, MediaEncode,
    Readout,
};
pub use ::driver_api::load::{Budgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
pub use ::driver_api::program::ProgramRegistration;
pub use ::driver_api::transfer::{KvCopy, KvMove, MemoryDomain, Pool, PoolResize, StateCopy, StateMove};
pub use ::driver_api::Driver;

/// The four recurrent-state verbs, as a slot's flag byte spells them.
///
/// **`palo B-rs`**: these were `driver_api::plan::RS_FLAG_*` and the contract
/// has no recurrent-state field left (see [`fire::RsPlan`]). The engine still
/// computes them for its own store, so the numbering lives here — one place,
/// and the byte no longer travels.
pub mod rs_flag {
    /// Clear the slot before the fire writes it.
    pub const RESET: u8 = 1 << 0;
    /// Fold the slot's history into this fire.
    pub const FOLD: u8 = 1 << 1;
    /// Write the slot's buffer as well as its state.
    pub const BUFFER_WRITE: u8 = 1 << 2;
    /// The fold length is resolved on the device, not stated here.
    pub const FOLD_LEN_DEVICE: u8 = 1 << 3;
}

pub use rs_flag::{
    BUFFER_WRITE as RS_FLAG_BUFFER_WRITE, FOLD as RS_FLAG_FOLD,
    FOLD_LEN_DEVICE as RS_FLAG_FOLD_LEN_DEVICE, RESET as RS_FLAG_RESET,
};

/// Which driver, as the registry addresses it.
pub type DriverId = usize;

/// The three adaptations the scheduler lane makes between the contract's
/// verbs and the run-ahead machinery around them.
///
/// One module rather than three inline `match`es at eleven sites: each of
/// these is a place the palo rewrite moved a responsibility across the
/// boundary, and each deserves the argument written once.
pub mod verbs {
    use anyhow::Result;

    use super::{
        ChannelRegistration, DriverBackend, DriverId, RegisteredChannel, SubmissionCompletion,
    };

    /// Which backend a driver's guest-program codegen emits for.
    ///
    /// Was `Driver::codegen_backend()`, a trait method; it is a field of
    /// [`DeviceFacts`](driver_api::DeviceFacts) now, because it is a fact
    /// about the machine and the contract already has a record for those.
    #[must_use]
    pub fn codegen_backend(driver: &DriverBackend) -> Option<&str> {
        driver
            .device_facts()
            .and_then(|facts| facts.codegen_backend.as_deref())
    }

    /// Write one adapter's planes into a loaded driver's banks (palo design
    /// §8, decision 17).
    ///
    /// **THE SMALLEST HONEST DOOR, AND IT IS DELIBERATELY THE SMALLEST.** A
    /// deployment that serves adapters wants an upload path, a registry, an
    /// id space shared with the control plane and a way for a request to name
    /// one — none of which is this. What the axis needed to EXIST is that the
    /// bytes reach the bank and a lane can say which row it wants, and this
    /// is the first half: one call, one id, one plane per bank, forwarded.
    ///
    /// The second half is [`Lane::adapter`](driver_api::fire::Lane::adapter),
    /// which the contract has carried since the rewrite, which the CUDA shell
    /// now honours end to end, and which
    /// [`stamp_lane_words`](crate::pipeline::fire) reads to compute the lane's
    /// fact word — so any caller that sets it gets the axis. What no path in
    /// this crate SETS it from yet is a per-request adapter id, because a
    /// request has nowhere to state one: the PTIR port vocabulary the fire
    /// path is assembled from names no such port, and adding one is the
    /// client-facing half this wave deliberately did not build.
    ///
    /// # Errors
    ///
    /// Whatever the driver refused — a bank it does not declare, an id past
    /// its capacity, a plane that is not one slot's bytes, or
    /// [`Unsupported`](driver_api::DriverError::Unsupported) from a shell
    /// whose loads seat no bank.
    pub fn register_adapter(
        driver: &mut DriverBackend,
        registration: &driver_api::adapter::AdapterRegistration,
    ) -> Result<()> {
        driver
            .register_adapter(registration)
            .map_err(anyhow::Error::from)
    }

    /// A control verb's answer, as the run-ahead broker wants it.
    ///
    /// **THE SHELLS ARE SYNCHRONOUS AND THE CONTRACT SAYS SO.** `copy_kv`,
    /// `copy_state`, `resize_pool` and `encode` used to answer a
    /// `SubmissionCompletion` the driver would settle later; they answer
    /// `Result<()>` now, and the work is done when they return. So the
    /// completion the engine hands its waiters is one that is already
    /// settled — [`SubmissionCompletion::ready`] — rather than a live wait
    /// slot nobody will ever publish into.
    ///
    /// # Errors
    ///
    /// Whatever the driver refused, widened to `anyhow` for the scheduler's
    /// mailbox.
    pub fn settled(result: driver_api::Result<()>) -> Result<SubmissionCompletion> {
        result
            .map(|()| SubmissionCompletion::ready())
            .map_err(anyhow::Error::from)
    }

    /// Register one channel: the engine's host ring, and the driver's device
    /// one if it has a plane for it.
    ///
    /// **BINDING IS REGISTRATION, FOR A SHELL WHOSE RINGS ARE ITS
    /// INSTANCES'.** The engine owns the ring the host puts into and takes out
    /// of ([`crate::driver::channel`], whose header argues it), and a shell
    /// owns the ring a guest program's STAGES read —
    /// `driver_cuda::program::launch` carves one per bound instance, from the
    /// package's own declarations, so there is nothing for a standalone
    /// `register_channel` to allocate over there and the CUDA shell answers
    /// [`Unsupported`](driver_api::DriverError::Unsupported), which is what
    /// the verb's own contract doc now says such a shell should answer.
    ///
    /// That refusal is TOLERATED here, and only that one; any other is a real
    /// one and is returned. What joins the two rings is not this verb but
    /// [`ChannelJoin`], pumping cells across at the fire's boundary.
    ///
    /// # Errors
    ///
    /// Whatever the driver refused, except [`Unsupported`].
    pub fn register_channel(
        driver: &mut DriverBackend,
        driver_id: DriverId,
        registration: &ChannelRegistration,
    ) -> Result<RegisteredChannel> {
        let table = waker::WakerTable::global();
        let (reader_wait_id, writer_wait_id) = match driver.register_channel(registration) {
            Ok(answer) => (answer.reader_wait_id, answer.writer_wait_id),
            Err(driver_api::DriverError::Unsupported { .. }) => (table.alloc(), table.alloc()),
            Err(error) => return Err(anyhow::Error::from(error)),
        };
        let cells: usize = registration
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>()
            .max(1);
        let cell_bytes = super::channel::HostRing::wire_cell_bytes(
            registration.dtype.program_dtype(),
            cells,
        );
        Ok(RegisteredChannel::new(
            driver_id,
            registration.id,
            u32::try_from(cell_bytes).unwrap_or(u32::MAX),
            registration.capacity,
            reader_wait_id,
            writer_wait_id,
        ))
    }
}

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
