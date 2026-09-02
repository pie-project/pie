//! Engine selection and the registry: the `EngineSpec`/`EngineBox` store
//! (`backend`), channel endpoint lifecycle (`channel`), and the launch-side
//! machinery the rest of the runtime reads. The contract itself
//! ([`Engine`](engine::Engine) and its verbs) is `engine`'s; this crate
//! picks a backend, keeps it in a registry under an `EngineId`, and holds
//! the channel endpoints applications wait on. Strictly leaf: no
//! `crate::{store,scheduler,pipeline,inferlet,server}` imports.

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
#[cfg(feature = "cuda")]
pub use backend::envelopes_resolved;
pub use channel::{
    ChannelBinding, ChannelCloser, ChannelEndpoint, ChannelJoin, ChannelValue, RegisteredChannel,
};

pub use completion::{
    CompletionBroker, CompletionLease, CompletionTarget, SubmissionCompletion, TerminalCell,
    WorkItemAttemptOutcome, WorkItemCompletion,
};
pub use instance::{BoundInstance, BoundWaitSlots, InstanceBindingPlan, InstanceId, ProgramId};

pub use fire::{
    FireRequest, FrameFire, MaskWords, StepFire, bitmask_words,
};

/// The four recurrent-state verbs, as a slot's flag byte spells them. The
/// numbering stays because the runtime's own recurrent store is built on it.
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
pub mod verbs {
    use anyhow::Result;

    use ::engine::ChannelRegistration;

    use super::{EngineBox, EngineId, RegisteredChannel, SubmissionCompletion};

    /// Which backend an engine's guest-program codegen emits for.
    #[must_use]
    pub fn codegen_backend(engine: &EngineBox) -> Option<&str> {
        engine
            .device_facts()
            .and_then(|facts| facts.codegen_backend.as_deref())
    }

    /// Write one adapter's planes into a loaded engine's banks: one call,
    /// one id, one plane per bank, forwarded. The read side is
    /// [`Lane::adapter`](engine::fire::Lane::adapter); no path in this crate
    /// sets a per-request adapter id yet, since the fire path's port
    /// vocabulary has no such port.
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

    /// A control verb's answer, as the run-ahead broker wants it. The
    /// shells are synchronous: `copy_kv`, `copy_state` and `encode` answer
    /// `Result<()>`, so the completion handed to waiters is already settled
    /// ([`SubmissionCompletion::ready`]) rather than a live wait slot.
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

    /// Register one channel: the runtime's host ring, and the engine's
    /// device one if it has a plane for it. Two shapes, picked by the
    /// engine's answer: a published mirror makes the runtime's ring a view
    /// of engine-allocated pinned memory (no pump, no copy); no mirror
    /// falls back to the runtime allocating its own ring and `ChannelJoin`
    /// pumping cells at the fire boundary.
    /// [`Unsupported`](engine::Error::Unsupported) is tolerated and treated
    /// as the no-mirror case; any other refusal is returned. A zero wait id
    /// from the engine means it keeps no waker table, so the slot is
    /// allocated here instead.
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
