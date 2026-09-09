pub mod backend;
pub mod channel;
pub mod completion;
pub mod fire;
pub mod instance;
pub mod load;

pub use waker;

#[cfg(feature = "cuda")]
pub use backend::envelopes_resolved;
pub use backend::{
    EngineBox, EngineSpec, RemoteDisconnectHandle, RemoteEngine, SchedulerLimits, get_spec, open,
    register_engine_backend, take_engine_backend, unregister_engine,
};
pub use channel::{
    ChannelBinding, ChannelCloser, ChannelEndpoint, ChannelJoin, ChannelValue, RegisteredChannel,
};

pub use completion::{
    CompletionBroker, CompletionLease, CompletionTarget, SubmissionCompletion, TerminalCell,
    WorkItemAttemptOutcome, WorkItemCompletion,
};
pub use instance::{BoundInstance, BoundWaitSlots, InstanceBindingPlan, InstanceId, ProgramId};

pub use fire::{FireRequest, FrameFire, MaskWords, StepFire, bitmask_words};

pub mod rs_flag {
    pub const RESET: u8 = 1 << 0;
    pub const FOLD: u8 = 1 << 1;
    pub const BUFFER_WRITE: u8 = 1 << 2;
    pub const FOLD_LEN_DEVICE: u8 = 1 << 3;
}

pub use rs_flag::{
    BUFFER_WRITE as RS_FLAG_BUFFER_WRITE, FOLD as RS_FLAG_FOLD,
    FOLD_LEN_DEVICE as RS_FLAG_FOLD_LEN_DEVICE, RESET as RS_FLAG_RESET,
};

pub type EngineId = usize;

pub mod verbs {
    use anyhow::Result;

    use ::engine::ChannelRegistration;

    use super::{EngineBox, EngineId, RegisteredChannel, SubmissionCompletion};

    #[must_use]
    pub fn codegen_backend(engine: &EngineBox) -> Option<&str> {
        engine
            .device_facts()
            .and_then(|facts| facts.codegen_backend.as_deref())
    }

    pub fn register_adapter(
        engine: &mut EngineBox,
        registration: &engine::adapter::AdapterRegistration,
    ) -> Result<()> {
        engine
            .register_adapter(registration)
            .map_err(anyhow::Error::from)
    }

    pub fn settled(result: engine::Result<()>) -> Result<SubmissionCompletion> {
        result
            .map(|()| SubmissionCompletion::ready())
            .map_err(anyhow::Error::from)
    }

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
        let cell_bytes =
            super::channel::HostRing::wire_cell_bytes(registration.dtype.program_dtype(), cells);
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

pub async fn generate_audio(
    _engine_idx: EngineId,
    _prompt: &[u32],
    _max_frames: u32,
) -> anyhow::Result<Vec<f32>> {
    Err(anyhow::anyhow!(
        "generate_audio is not wired to engine backends yet"
    ))
}
