pub mod backend;
pub mod channel;
pub mod completion;
pub mod fire;
pub mod instance;
pub mod load;
pub mod profiles;

pub use waker;

pub use backend::{
    EngineBox, EngineSpec, SchedulerLimits, fire_limits, get_spec, open, register_engine_backend,
    take_engine_backend, unregister_engine,
};
pub use channel::{
    ChannelBinding, ChannelCloser, ChannelEndpoint, ChannelJoin, ChannelValue, RegisteredChannel,
};

pub use completion::{
    CompletionBroker, CompletionLease, SubmissionCompletion, TerminalCell, WorkItemAttemptOutcome,
    WorkItemCompletion,
};
pub use instance::{BoundInstance, BoundWaitSlots, InstanceBindingPlan, InstanceId, ProgramId};

pub use fire::{FireLane, FireRequest, FrameFire, MaskWords, StepFire, bitmask_words};

pub mod rs_flag {
    pub const RESET: u8 = 1 << 0;
}

pub use rs_flag::RESET as RS_FLAG_RESET;

pub type EngineId = usize;

pub mod verbs {
    use anyhow::Result;

    use ::engine::{ChannelRegistration, Engine};

    use super::{EngineBox, EngineId, RegisteredChannel, SubmissionCompletion};

    #[must_use]
    pub fn codegen_backend(engine: &EngineBox) -> Option<&str> {
        engine
            .device_facts()
            .and_then(|facts| facts.codegen_backend.as_deref())
    }

    /// An engine as the runtime holds it: boxed on its lane, or borrowed.
    pub trait EngineMut {
        fn engine_mut(&mut self) -> &mut dyn Engine;
    }

    impl EngineMut for EngineBox {
        fn engine_mut(&mut self) -> &mut dyn Engine {
            self.as_mut()
        }
    }

    impl<'x> EngineMut for dyn Engine + 'x {
        fn engine_mut(&mut self) -> &mut dyn Engine {
            self
        }
    }

    /// `engine`'s family `family`, or `Unsupported` naming `verb` where
    /// the engine serves none.
    pub fn serve<'a, E: EngineMut + ?Sized, F: ?Sized>(
        engine: &'a mut E,
        family: fn(&'a mut dyn Engine) -> Option<&'a mut F>,
        verb: &'static str,
    ) -> engine::Result<&'a mut F> {
        let engine = engine.engine_mut();
        let kind = engine.kind();
        family(engine).ok_or(engine::Error::unsupported(kind, verb))
    }

    /// Whether `engine` lands its steps after `submit` returns
    /// ([`::engine::Completion::asynchronous`]); one that tells no sink
    /// settles inside `submit`.
    pub fn settles_asynchronously(engine: &mut dyn Engine) -> bool {
        engine
            .completion()
            .is_some_and(|completion| completion.asynchronous())
    }

    pub fn register_adapter(
        engine: &mut EngineBox,
        registration: &engine::adapter::AdapterRegistration,
    ) -> Result<()> {
        serve(engine, Engine::programs, "register_adapter")
            .and_then(|programs| programs.adapter(registration))
            .map_err(anyhow::Error::from)
    }

    pub fn settled(result: engine::Result<()>) -> Result<SubmissionCompletion> {
        result
            .map(|()| SubmissionCompletion::ready())
            .map_err(anyhow::Error::from)
    }

    pub fn register_channel(
        engine: &mut dyn ::engine::Engine,
        engine_id: EngineId,
        registration: &ChannelRegistration,
    ) -> Result<RegisteredChannel> {
        let table = waker::WakerTable::global();
        // What the engine stated it holds is what the bind took rings by
        // (`Channels::holds`): an answer past it would leave a ring with no
        // slots, or slots taken twice.
        let holds = engine
            .channels()
            .is_some_and(|channels| channels.holds(registration));
        let answered = match serve(engine, Engine::channels, "register_channel")
            .and_then(|channels| channels.open(registration))
        {
            Ok(answer) if !holds => {
                if let Some(channels) = engine.channels() {
                    let _ = channels.close(registration.id);
                }
                return Err(anyhow::anyhow!(
                    "channel {}: the engine states it leaves the registration to the runtime \
                     and took it",
                    answer.id
                ));
            }
            Ok(answer) => Some(answer),
            Err(engine::Error::Unsupported { .. }) if holds => {
                return Err(anyhow::anyhow!(
                    "channel {}: the engine states it holds the registration and answered it \
                     unsupported",
                    registration.id
                ));
            }
            Err(engine::Error::Unsupported { .. }) => None,
            Err(error) => return Err(anyhow::Error::from(error)),
        };
        let engine_registered = answered.is_some();
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
                let ring = super::channel::HostRing::adopt(published).ok_or_else(|| {
                    anyhow::anyhow!(
                        "channel {}'s engine mirror is shorter than its ring",
                        registration.id
                    )
                })?;
                Ok(RegisteredChannel::over(
                    engine_id,
                    registration.id,
                    std::sync::Arc::new(ring),
                    reader_wait_id,
                    writer_wait_id,
                ))
            }
            None => Ok(RegisteredChannel {
                engine_registered,
                ..RegisteredChannel::new(
                    engine_id,
                    registration.id,
                    cell_bytes,
                    registration.capacity,
                    reader_wait_id,
                    writer_wait_id,
                )
            }),
        }
    }
}
