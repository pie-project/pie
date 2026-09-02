//! `trait Engine` — the verb set. Object-safe: no generics, no `Self: Sized`. Six verbs default to [`Error::Unsupported`].

use serde::{Deserialize, Serialize};

use crate::adapter::AdapterRegistration;
use crate::caps::DeviceFacts;
use crate::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use crate::error::{Error, Result};
use crate::fire::{FrameId, FrameSubmission, FrameTicket, MediaEncode, Step};
use crate::load::{LoadRequest, Loaded};
use crate::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use crate::transfer::{KvCopy, KvHandle, StateCopy};

/// Which step of which frame a completion is about, for the runtime's
/// broker to correlate on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StepDone {
    /// The frame this step belongs to.
    pub frame: FrameId,
    /// Its position in the frame's steps, in submission order.
    pub step: u32,
}

/// How a step ended. Two verdicts only: it ran, or it didn't (a fault, not
/// a retry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOutcome {
    /// The step's work completed and its effects are durable.
    Committed,
    /// It did not; the frame's remaining steps are poisoned.
    Faulted(String),
}

/// Where an async engine publishes a step's completion; called from
/// whatever thread settles (e.g. CUDA's host-function thread), so a sink
/// does only atomics, a brief lock, and a waker publish.
pub type CompletionSink = std::sync::Arc<dyn Fn(StepDone, StepOutcome) + Send + Sync>;

/// What the runtime calls a device through.
pub trait Engine: Send + Sync {
    /// Which shell this is — `"cuda"`, `"metal"`, `"remote"`.
    fn kind(&self) -> &'static str;

    /// The device's facts, once bound. `None` before load or with no device.
    fn device_facts(&self) -> Option<&DeviceFacts> {
        None
    }

    /// This engine's KV pool, addressable by a peer; `None` if not exportable.
    fn export_kv_handle(&self) -> Option<KvHandle> {
        None
    }

    /// States the calling thread will drive this device from now on; call
    /// once, before any other verb. Default: no-op (no thread-affine state).
    ///
    /// # Errors
    ///
    /// [`Error::Device`] when the thread cannot be bound.
    fn bind_thread(&mut self) -> Result<()> {
        Ok(())
    }

    // ── load ────────────────────────────────────────────────────────────

    /// Bake the plan, land the checkpoint, reserve the pools.
    ///
    /// # Errors
    ///
    /// [`Error::Load`] for a plan the budgets don't admit or a checkpoint
    /// that doesn't fit; [`Error::Device`] for the residency.
    fn load(&mut self, request: LoadRequest) -> Result<Loaded>;

    // ── guest programs ──────────────────────────────────────────────────

    /// Compile and register a guest program, answering its id.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] for a package that doesn't adopt,
    /// [`Error::Unsupported`] from a shell with no guest-program plane.
    fn register_program(&mut self, registration: &ProgramRegistration) -> Result<ProgramId> {
        let _ = registration;
        Err(self.unsupported("register_program"))
    }

    /// Allocate a channel's ring and its wait slots.
    ///
    /// A shell that carves rings inside [`Engine::bind_instance`] may refuse
    /// this instead. A shell declaring
    /// [`device_channel_commit`](crate::Capabilities::device_channel_commit)
    /// allocates the host half as pinned memory
    /// ([`RegisteredChannel::mirror`](crate::RegisteredChannel::mirror));
    /// a zero wait id means the shell keeps no waker table.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] for a declaration the shell can't allocate,
    /// [`Error::Unsupported`] from a shell with no guest-program plane or
    /// whose rings are its instances'.
    fn register_channel(&mut self, registration: &ChannelRegistration) -> Result<RegisteredChannel> {
        let _ = registration;
        Err(self.unsupported("register_channel"))
    }

    /// Bind an instance of a registered program to a set of channels.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] for an unknown program, an undeclared channel, or
    /// a seed that doesn't fit its cell.
    fn bind_instance(&mut self, binding: &InstanceBinding) -> Result<BoundInstance> {
        let _ = binding;
        Err(self.unsupported("bind_instance"))
    }

    /// Tear down an instance and free its wait slots.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] for an instance already gone.
    fn close_instance(&mut self, id: InstanceId) -> Result<()> {
        let _ = id;
        Err(self.unsupported("close_instance"))
    }

    /// Tear down a channel and free its ring; refused by the same shells as
    /// [`Engine::register_channel`] (a bind-carved ring is freed by
    /// [`Engine::close_instance`] instead).
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] for a channel already gone, [`Error::Unsupported`]
    /// from a shell whose rings are its instances'.
    fn close_channel(&mut self, id: ChannelId) -> Result<()> {
        let _ = id;
        Err(self.unsupported("close_channel"))
    }

    /// Push one wire cell into a bound instance's channel; `false` means
    /// back-pressure, not a drop. `channel` is the package's
    /// declaration-order index.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] for an unknown instance, an uncarried channel, or
    /// wrong cell width; [`Error::Unsupported`] from a shell with no
    /// guest-program plane.
    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> Result<bool> {
        let _ = (instance, channel, cell);
        Err(self.unsupported("publish_channel"))
    }

    /// Take one wire cell out of a bound instance's channel, advancing its
    /// head; `None` when the ring is empty.
    ///
    /// # Errors
    ///
    /// As [`Engine::publish_channel`].
    fn take_channel(&mut self, instance: InstanceId, channel: u32) -> Result<Option<Vec<u8>>> {
        let _ = (instance, channel);
        Err(self.unsupported("take_channel"))
    }

    // ── adapter banks ───────────────────────────────────────────────────

    /// Write one adapter's planes into this load's device banks.
    ///
    /// No recapture needed: a bank's contents aren't part of an engine's
    /// graph key, so a lane just selects one by id
    /// ([`Lane::adapter`](crate::fire::Lane::adapter)).
    ///
    /// # Errors
    ///
    /// [`Error::Load`] for an undeclared bank, an id past capacity, or a
    /// plane that isn't one slot's bytes; [`Error::Device`] for the
    /// residency; [`Error::Unsupported`] from a shell whose loads seat no bank.
    fn register_adapter(&mut self, registration: &AdapterRegistration) -> Result<()> {
        let _ = registration;
        Err(self.unsupported("register_adapter"))
    }

    // ── the fire path ───────────────────────────────────────────────────

    /// The one forward verb. Admit a frame — 1..=k steps, sealed in order —
    /// and run it.
    ///
    /// * Static admission: validated and committed once before any stream
    ///   work; past that, stream work is success-only.
    /// * Saturation: all k steps enqueue before this returns.
    /// * Untouched transition: no host read/decision/sync/memcpy between steps.
    /// * Guests are isolated: `k` never appears in a guest ABI.
    ///
    /// Synchronous up to admission; the device may still be running after.
    /// Taken by reference since [`Error::Exhausted`] means retry this frame.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for an undescribed frame, [`Error::Unsupported`]
    /// for an unserved shape, [`Error::Exhausted`] for one that doesn't fit
    /// now, [`Error::Impossible`] for one past a baked ceiling,
    /// [`Error::Device`] for a refused launch.
    fn submit(&mut self, frame: &FrameSubmission) -> Result<FrameTicket>;

    /// Does this engine answer `submit` before the device is done?
    ///
    /// `false` (default): every step's readouts are filled by return.
    /// `true`: outcomes arrive via [`Engine::on_complete`]'s sink, and
    /// numbers come from [`Engine::settle_frame`].
    fn settles_asynchronously(&self) -> bool {
        false
    }

    /// Install where this engine publishes step completions. Called once,
    /// before the first `submit`.
    fn on_complete(&mut self, sink: CompletionSink) {
        let _ = sink;
    }

    /// Fill in a receipt's readouts, waiting for the device if needed. Not
    /// the settlement path — for a smoke test, bench, or tool that wants
    /// numbers. No-op by default (a synchronous engine already filled them).
    ///
    /// # Errors
    ///
    /// [`Error::Device`] for the frame's work, [`Error::Invalid`] for a
    /// ticket this engine did not mint.
    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> Result<()> {
        let _ = ticket;
        Ok(())
    }

    /// State the fire the caller expects to submit next. Advisory: an
    /// engine may warm state for it; doing nothing is exactly as correct.
    /// Reads only composition (word, row count), never token values.
    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
    }

    // ── state movement ──────────────────────────────────────────────────

    /// Move KV pages, within this device or across a domain boundary.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for an unserved direction, [`Error::Invalid`]
    /// for a malformed plan, [`Error::Device`] for the transfer.
    fn copy_kv(&mut self, copy: &KvCopy) -> Result<()> {
        let _ = copy;
        Err(self.unsupported("copy_kv"))
    }

    /// Move recurrent state between slots.
    ///
    /// # Errors
    ///
    /// As [`Engine::copy_kv`].
    fn copy_state(&mut self, copy: &StateCopy) -> Result<()> {
        let _ = copy;
        Err(self.unsupported("copy_state"))
    }

    /// Encode non-text modalities into embedding rows; `&mut` because rows
    /// write back into the caller-sized buffer.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] from a load with no encoder, [`Error::Invalid`]
    /// for a payload with no anchor.
    fn encode(&mut self, plan: &mut MediaEncode) -> Result<()> {
        let _ = plan;
        Err(self.unsupported("encode"))
    }

    // ── lifetime ────────────────────────────────────────────────────────

    /// Tell the engine its caller is going away, with a reason for the log.
    fn disconnect(&self, message: &str) {
        let _ = message;
    }

    /// The refusal this engine answers `verb` with.
    fn unsupported(&self, verb: &'static str) -> Error {
        Error::unsupported(self.kind(), verb)
    }
}

/// Object safety, checked here rather than at the first `Box<dyn Engine>`.
const _: () = {
    #[allow(dead_code)]
    fn object_safe(engine: &dyn Engine) -> &'static str {
        engine.kind()
    }
};
