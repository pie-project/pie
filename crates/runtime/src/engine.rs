//! L0: engine selection and the registry — the `EngineSpec`/`EngineBox`
//! store (`backend`), the concrete seams behind it, channel endpoint lifecycle
//! (`channel`), and the launch-side re-exports the rest of the runtime reads.
//!
//! This is the RUNTIME's half of the engine boundary and nothing else. The
//! contract — [`Engine`](engine::Engine) and the fourteen verbs, the
//! completion an engine mints, the bind plan, the registration an engine answers —
//! is `engine`'s, because both sides say it. What only the runtime does is
//! pick a backend, keep it in a registry under an `EngineId`, and hold the
//! channel endpoints applications wait on.
//!
//! **Strictly leaf**: no `crate::{store,scheduler,pipeline,inferlet,server}`
//! imports. Splicing host-generated kernels into a registration is
//! [`crate::pipeline::program::with_host_codegen`]'s, called by the scheduler
//! that owns the engine handle and knows which engine a plan is bound for. The
//! per-`engine_id` dispatch verbs (`register_program`, `bind_instance`, the
//! `copy_*` family) live in the scheduler dispatch facade for the same reason:
//! they need its engine-id -> handle registry to reach the `BatchScheduler`.

pub mod backend;
pub mod channel;
pub mod completion;
pub mod fire;
pub mod instance;
pub mod load;

pub use waker;

pub use backend::{
    EngineBox, EngineSpec, RemoteDisconnectHandle, RemoteEngine, SchedulerLimits, get_spec,
    open, register_engine_backend, take_engine_backend, unregister_engine,
};
#[cfg(feature = "_engine-cuda")]
pub use backend::envelopes_resolved;
#[cfg(feature = "_engine-cuda")]
pub use backend::fold_observed;
pub use channel::{
    ChannelBinding, ChannelCloser, ChannelEndpoint, ChannelJoin, ChannelValue, RegisteredChannel,
};

// THE BROKER CAME HOME (palo design §7, decision 19). `CompletionBroker`,
// `SubmissionCompletion`, `WorkItemCompletion` and the terminal cell were
// 807 lines inside `engine`, describing how the RUNTIME runs ahead of a
// device. They are `engine::completion` and `engine::instance` now, and the
// contract keeps only the receipt — `FireTicket`.
pub use completion::{
    CompletionBroker, CompletionLease, CompletionTarget, SubmissionCompletion, TerminalCell,
    WorkItemAttemptOutcome, WorkItemCompletion,
};
pub use instance::{BoundInstance, BoundWaitSlots, InstanceBindingPlan, InstanceId, ProgramId};

// The runtime's own submission vocabulary. `LaunchPlan` and its sixty-two
// parallel CSR arms are gone; what a request IS lives in `fire`, and what
// crosses the boundary is the contract's own `Lane`.
pub use fire::{
    FireRequest, FrameFire, MaskWords, StepFire, bitmask_words,
};

// The contract, re-exported at the path the runtime already reads it from.
pub use ::engine::adapter::{AdapterPlane, AdapterRegistration};
pub use ::engine::caps::Capabilities;
pub use ::engine::channel::ChannelRegistration;
pub use ::engine::error::{Error, Result as EngineResult};
pub use ::engine::fire::{
    Attachment, Boundary, FireTicket, FrameSubmission, FrameTicket, KvDelta, Lane, LaneReadout,
    Mask, Masking, MediaEncode, Readout, RsReset, RsVerb, Step,
};
pub use ::engine::load::{Budgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
pub use ::engine::program::ProgramRegistration;
pub use ::engine::transfer::{KvCopy, KvMove, MemoryDomain, StateCopy, StateMove};
pub use ::engine::Engine;

/// The four recurrent-state verbs, as a slot's flag byte spells them.
///
/// **`palo B-rs`**: these were `engine::plan::RS_FLAG_*`. The byte no
/// longer travels — `engine::RsVerb` and `engine::RsReset` are what
/// crosses the boundary since wave F3-tail, and `PreparedRs::apply_to` reads
/// `RESET` here to state which of the two a lane's slot is. The numbering
/// stays because the runtime's own recurrent store is built on it.
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

/// Which engine, as the registry addresses it.
pub type EngineId = usize;

/// The three adaptations the scheduler lane makes between the contract's
/// verbs and the run-ahead machinery around them.
///
/// One module rather than three inline `match`es at eleven sites: each of
/// these is a place the palo rewrite moved a responsibility across the
/// boundary, and each deserves the argument written once.
pub mod verbs {
    use anyhow::Result;

    use super::{
        ChannelRegistration, EngineBox, EngineId, RegisteredChannel, SubmissionCompletion,
    };

    /// Which backend an engine's guest-program codegen emits for.
    ///
    /// Was `Engine::codegen_backend()`, a trait method; it is a field of
    /// [`DeviceFacts`](engine::DeviceFacts) now, because it is a fact
    /// about the machine and the contract already has a record for those.
    #[must_use]
    pub fn codegen_backend(engine: &EngineBox) -> Option<&str> {
        engine
            .device_facts()
            .and_then(|facts| facts.codegen_backend.as_deref())
    }

    /// Write one adapter's planes into a loaded engine's banks (palo design
    /// §8, decision 17).
    ///
    /// **THE SMALLEST HONEST DOOR, AND IT IS DELIBERATELY THE SMALLEST.** A
    /// deployment that serves adapters wants an upload path, a registry, an
    /// id space shared with the control plane and a way for a request to name
    /// one — none of which is this. What the axis needed to EXIST is that the
    /// bytes reach the bank and a lane can say which row it wants, and this
    /// is the first half: one call, one id, one plane per bank, forwarded.
    ///
    /// The second half is [`Lane::adapter`](engine::fire::Lane::adapter),
    /// which the contract has carried since the rewrite, which the CUDA shell
    /// now honours end to end, and which
    /// [`stamp_lane_words`](crate::pipeline::fire) reads to compute the lane's
    /// fact word — so any caller that sets it gets the axis. What no path in
    /// this crate SETS it from yet is a per-request adapter id, because a
    /// request has nowhere to state one: the ETA port vocabulary the fire
    /// path is assembled from names no such port, and adding one is the
    /// client-facing half this wave deliberately did not build.
    ///
    /// # Errors
    ///
    /// Whatever the engine refused — a bank it does not declare, an id past
    /// its capacity, a plane that is not one slot's bytes, or
    /// [`Unsupported`](engine::Error::Unsupported) from a shell
    /// whose loads seat no bank.
    pub fn register_adapter(
        engine: &mut EngineBox,
        registration: &engine::adapter::AdapterRegistration,
    ) -> Result<()> {
        engine
            .register_adapter(registration)
            .map_err(anyhow::Error::from)
    }

    /// A control verb's answer, as the run-ahead broker wants it.
    ///
    /// **THE SHELLS ARE SYNCHRONOUS AND THE CONTRACT SAYS SO.** `copy_kv`,
    /// `copy_state` and `encode` used to answer a
    /// `SubmissionCompletion` the engine would settle later; they answer
    /// `Result<()>` now, and the work is done when they return. So the
    /// completion the runtime hands its waiters is one that is already
    /// settled — [`SubmissionCompletion::ready`] — rather than a live wait
    /// slot nobody will ever publish into.
    ///
    /// # Errors
    ///
    /// Whatever the engine refused, widened to `anyhow` for the scheduler's
    /// mailbox.
    pub fn settled(result: engine::Result<()>) -> Result<SubmissionCompletion> {
        result
            .map(|()| SubmissionCompletion::ready())
            .map_err(anyhow::Error::from)
    }

    /// Register one channel: the runtime's host ring, and the engine's device
    /// one if it has a plane for it.
    ///
    /// **THE HOST RING IS THE ENGINE'S WHEN THE ENGINE SAYS SO** (alto design
    /// §5, wave F2a).
    ///
    /// Two shapes, and the engine's answer picks between them:
    ///
    /// ```text
    /// mirror published   the engine allocated this channel's host half in
    ///                    mapped pinned memory, its control kernels read and
    ///                    write it, and the runtime's ring becomes a VIEW of
    ///                    those bytes — no pump, no copy, no device call on
    ///                    the guest's thread
    /// no mirror          the pre-F2a shape: the runtime allocates its own
    ///                    ring and `ChannelJoin` pumps cells across at the
    ///                    fire's boundary
    /// ```
    ///
    /// [`Unsupported`](engine::Error::Unsupported) is the third: a shell
    /// with no standalone channel to register at all. It is TOLERATED — and
    /// only it — and falls into the second shape; any other refusal is real
    /// and is returned.
    ///
    /// **WAIT SLOTS ARE THE RUNTIME'S WHEN THE ENGINE MINTS NONE.** A zero id
    /// from an engine means it keeps no waker table (the contract says so),
    /// so the slot is allocated here rather than inventing one nobody
    /// signals.
    ///
    /// # Errors
    ///
    /// Whatever the engine refused, except [`Unsupported`].
    pub fn register_channel(
        engine: &mut EngineBox,
        engine_id: EngineId,
        registration: &ChannelRegistration,
    ) -> Result<RegisteredChannel> {
        let table = waker::WakerTable::global();
        let answered = match engine.register_channel(registration) {
            Ok(answer) => Some(answer),
            Err(engine::Error::Unsupported { .. }) => None,
            Err(error) => return Err(anyhow::Error::from(error)),
        };
        let mint = |id: u64| if id == 0 { table.alloc() } else { id };
        let (reader_wait_id, writer_wait_id) = answered.as_ref().map_or_else(
            || (table.alloc(), table.alloc()),
            |answer| (mint(answer.reader_wait_id), mint(answer.writer_wait_id)),
        );
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
        let cell_bytes = u32::try_from(cell_bytes).unwrap_or(u32::MAX);
        match answered.and_then(|answer| answer.mirror) {
            Some(published) => {
                if published.cell_bytes != cell_bytes || published.capacity != registration.capacity
                {
                    // The engine cut the mirror for a different ring than the
                    // one this registration declares. Adopting it would have
                    // the guest addressing cells at one stride and the pull
                    // reading them at another — a wrong token, never a fault.
                    return Err(anyhow::anyhow!(
                        "channel {} is declared with a {cell_bytes}-byte cell and a capacity \
                         of {}, and the engine published a mirror of {}-byte cells and a \
                         capacity of {}",
                        registration.id,
                        registration.capacity,
                        published.cell_bytes,
                        published.capacity
                    ));
                }
                // SAFETY: the engine published these addresses for this
                // registration and holds them until `close_channel`, which
                // the runtime calls only after dropping the record below.
                let ring = std::sync::Arc::new(unsafe {
                    super::channel::HostRing::adopt(
                        published.mirror,
                        published.words,
                        published.cell_bytes,
                        published.capacity,
                    )
                });
                Ok(RegisteredChannel::over(
                    engine_id,
                    registration.id,
                    ring,
                    reader_wait_id,
                    writer_wait_id,
                ))
            }
            None => Ok(RegisteredChannel::new(
                engine_id,
                registration.id,
                cell_bytes,
                registration.capacity,
                reader_wait_id,
                writer_wait_id,
            )),
        }
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
    _engine_idx: EngineId,
    _prompt: &[u32],
    _max_frames: u32,
) -> anyhow::Result<Vec<f32>> {
    Err(anyhow::anyhow!(
        "generate_audio is not wired to engine backends yet"
    ))
}
