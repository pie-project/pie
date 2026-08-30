//! `trait Engine` — the verb set.
//!
//! **THE VERBS SURVIVED THE REWRITE; THE ENCODING DID NOT.** Every method
//! below was a method on the trait this replaces, and it means what it meant.
//! What changed is what they take and what they answer: a `model_ir::Trace`
//! instead of a `Vec<ModelLoadDesc>`, a `Step` of lanes instead of a
//! 62-field `LaunchPlan` of parallel CSRs, `Result<_, Error>` instead of
//! `Result<_, anyhow::Error>` with an `i32` status hiding inside it.
//!
//! # Object-safe, and it is checked below
//!
//! The runtime holds `Vec<Box<dyn Engine>>` — a CUDA shell, a Metal shell, a
//! remote one — and dispatches on the same trait for all of them. So: no
//! generic methods, no `Self: Sized`, no `impl Trait` in return position, and
//! a `const` block at the bottom of this file that coerces a `&dyn Engine` so
//! that violating any of those is a compile error in the crate that caused it
//! rather than in the crate that tried to use it.
//!
//! # Remote is a property, not an encoding (decision 19)
//!
//! There is no wire version here, no `ExecutorRequest` enum, no tarpc service.
//! A remote engine is a type in the transport that implements this trait and
//! whose method bodies happen to be round trips; every noun it needs to send
//! is `Serialize + DeserializeOwned` because every noun in this crate is.
//! Which framing, which envelope and which version negotiation it uses are
//! *the transport's* decisions, and a contract that made them for it would be
//! wrong for every transport but one.
//!
//! # Refusal is a value
//!
//! Six of the verbs have default bodies that answer
//! [`Error::Unsupported`]. A Metal shell has no `copy_kv`; a shell with
//! no encoder has no `encode`. Answering "I do not serve this" is a normal
//! thing for an engine to do, and it is cheaper to write it once here than in
//! every shell that does not serve it.

use serde::{Deserialize, Serialize};

use crate::adapter::AdapterRegistration;
use crate::caps::DeviceFacts;
use crate::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use crate::error::{Error, Result};
use crate::fire::{FrameId, FrameSubmission, FrameTicket, MediaEncode, Step};
use crate::load::{LoadRequest, Loaded};
use crate::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use crate::transfer::{KvCopy, KvHandle, StateCopy};

/// **Which step of which frame a completion is about.**
///
/// The correlation the receipt already promised: `FrameTicket::id` plus a
/// position in `FrameTicket::steps`. An engine that answers before the device
/// is done hands this back when the device IS done, and the runtime's broker
/// is what it correlates on (survey §7, I7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StepDone {
    /// The frame this step belongs to.
    pub frame: FrameId,
    /// Its position in the frame's steps, in submission order.
    pub step: u32,
}

/// **How a step ended**, as the settlement path classifies it.
///
/// Two verdicts and no third, because article 4 already spent the third: past
/// admission the stream work is success-only, so what is left is "it ran" and
/// "the device or a guest pass said no", and the second is a fault with a
/// sentence rather than a retry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOutcome {
    /// The step's work completed and its effects are durable.
    Committed,
    /// It did not, and this is why. The frame's remaining steps are poisoned.
    Faulted(String),
}

/// **Where an asynchronous engine publishes a step's completion.**
///
/// Installed once, by the thread that owns the engine
/// ([`Engine::on_complete`]), and called from whatever thread the backend
/// settles on — for CUDA that is the driver's host-function thread, which may
/// make no CUDA call of its own. So a sink does atomics, a lock it holds
/// briefly, and a waker publish; it does not call back into the engine.
pub type CompletionSink = std::sync::Arc<dyn Fn(StepDone, StepOutcome) + Send + Sync>;

/// What the runtime calls a device through.
pub trait Engine: Send + Sync {
    /// Which shell this is — `"cuda"`, `"metal"`, `"remote"`. Used in
    /// diagnostics and in [`Error::Unsupported`].
    fn kind(&self) -> &'static str;

    /// What the machine underneath is, once it is bound. `None` before a load,
    /// and from an engine with no device of its own.
    fn device_facts(&self) -> Option<&DeviceFacts> {
        None
    }

    /// This engine's KV pool, addressable by a peer. `None` when it is not
    /// exportable.
    fn export_kv_handle(&self) -> Option<KvHandle> {
        None
    }

    /// **THE CALLING THREAD IS THE ONE THAT WILL DRIVE THIS DEVICE FROM NOW
    /// ON.** Said once, by whoever takes ownership of the engine, before the
    /// first verb it calls.
    ///
    /// An engine may hold per-THREAD state that no value can carry across a
    /// hand-off: `engine-cuda`'s `Context` says so at the top of its own
    /// module — "`cudaSetDevice` is per-thread state, so binding somewhere
    /// other than where the fires happen strands every later call on device
    /// 0". That binding happens inside [`Engine::load`], on whichever thread
    /// booted the worker; the runtime then moves the engine onto its own lane
    /// thread and every verb after that runs there.
    ///
    /// The CUDA runtime API forgives this — an unbound thread defaults to device 0
    /// and the primary context is created lazily — so a single-device
    /// deployment fires correctly by accident. The CUDA DRIVER api does not: with
    /// no current context `cuModuleLoadData` answers
    /// `CUDA_ERROR_INVALID_CONTEXT`, which is what a guest program's first
    /// registration met, and what this verb exists to prevent.
    ///
    /// Default: nothing. An engine with no thread-affine state — a remote one,
    /// a shell whose device handle is a value — needs no announcement.
    ///
    /// # Errors
    ///
    /// [`Error::Device`] when the thread cannot be bound at all.
    fn bind_thread(&mut self) -> Result<()> {
        Ok(())
    }

    // ── load ────────────────────────────────────────────────────────────

    /// Bake the plan, land the checkpoint, reserve the pools.
    ///
    /// The one door a model comes through. The `Trace` crosses here and
    /// `CompiledModel` never does (decision 18): the compile happens on this side of
    /// the boundary because it is an answer about a device.
    ///
    /// # Errors
    ///
    /// [`Error::Load`] for a plan these budgets do not admit or a
    /// checkpoint that does not fit it, [`Error::Device`] for the
    /// residency.
    fn load(&mut self, request: LoadRequest) -> Result<Loaded>;

    // ── guest programs ──────────────────────────────────────────────────

    /// Compile and register a guest program, answering its id.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] for a package that does not adopt,
    /// [`Error::Unsupported`] from a shell with no guest-program plane.
    fn register_program(&mut self, registration: &ProgramRegistration) -> Result<ProgramId> {
        let _ = registration;
        Err(self.unsupported("register_program"))
    }

    /// Allocate a channel's ring and its wait slots.
    ///
    /// # A shell whose rings are an instance's may refuse this, and that is
    /// not a hole
    ///
    /// The verb states a channel whose life is LONGER than any one instance's
    /// — the shape a channel shared by several passes needs, and the reason
    /// this is its own verb and not a field of [`InstanceBinding`]. A shell
    /// that carves every ring inside [`Engine::bind_instance`], from the
    /// package's own declarations, has nothing to allocate here: for it,
    /// **binding IS registration**, and answering
    /// [`Error::Unsupported`] is the honest report rather than a
    /// pretend allocation the bind would then replace. The runtime tolerates
    /// exactly that refusal and keeps its own host ring
    /// (`runtime::engine::verbs::register_channel`).
    ///
    /// What such a shell still owes the host is a DOOR into the rings it
    /// carved, and that is [`Engine::publish_channel`] /
    /// [`Engine::take_channel`] — not this.
    ///
    /// # A shell that commits its rings on the DEVICE answers with MEMORY
    ///
    /// An engine that declares
    /// [`device_channel_commit`](crate::Capabilities::device_channel_commit)
    /// does not want the caller's cells handed to it a copy at a time; it
    /// wants the caller to write them where its own control kernels will read
    /// them. So it allocates this channel's host half — mapped pinned memory —
    /// and publishes the addresses as
    /// [`RegisteredChannel::mirror`](crate::RegisteredChannel::mirror). The
    /// caller's ring becomes a view of those bytes, the two doors above become
    /// a convenience rather than the path, and a guest round trip makes no
    /// device call at all (alto design §5, survey §7 invariant I5).
    ///
    /// Such a shell may still keep no waker table, and says so by answering
    /// zero for the two wait ids — the caller then mints its own.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] for a declaration the shell cannot allocate,
    /// [`Error::Unsupported`] from a shell with no guest-program plane,
    /// and from one whose rings are its instances'.
    fn register_channel(&mut self, registration: &ChannelRegistration) -> Result<RegisteredChannel> {
        let _ = registration;
        Err(self.unsupported("register_channel"))
    }

    /// Bind an instance of a registered program to a set of channels.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] for an unknown program, a channel the package
    /// does not declare, or a seed that does not fit its cell.
    fn bind_instance(&mut self, binding: &InstanceBinding) -> Result<BoundInstance> {
        let _ = binding;
        Err(self.unsupported("bind_instance"))
    }

    /// Tear down an instance and free its wait slots.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] for an instance that is already gone.
    fn close_instance(&mut self, id: InstanceId) -> Result<()> {
        let _ = id;
        Err(self.unsupported("close_instance"))
    }

    /// Tear down a channel and free its ring.
    ///
    /// The counterpart of [`Engine::register_channel`], and refused by the
    /// same shells for the same reason: a ring that was carved by a bind is
    /// freed by [`Engine::close_instance`].
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] for a channel that is already gone,
    /// [`Error::Unsupported`] from a shell whose rings are its
    /// instances'.
    fn close_channel(&mut self, id: ChannelId) -> Result<()> {
        let _ = id;
        Err(self.unsupported("close_channel"))
    }

    /// Push one wire cell into a bound instance's channel, answering `false`
    /// when the ring has no room — back-pressure, not a drop.
    ///
    /// **THIS IS WHAT `ChannelBinding` USED TO BE.** The contract once
    /// published an engine's private ring layout — `mirror_base`, `word_base`,
    /// `head_word_index`, … — so the host could write a cell into device
    /// memory itself; `channel.rs`'s header records why that died. A host
    /// that no longer poked the ring was left with no way to hand a guest
    /// program its input at all, and this verb is the door that replaces the
    /// pointer: the runtime's own host ring on one side, the shell's device
    /// ring on the other, wire bytes between them.
    ///
    /// `channel` is the index in the package's DECLARATION order — the same
    /// numbering [`ChannelSeed::channel`](crate::channel::ChannelSeed) uses
    /// and [`InstanceBinding::channels`] maps to global ids — because that is
    /// the numbering an instance's rings are carved in.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] for an unknown instance, a channel the
    /// instance does not carry, or a cell of the wrong width;
    /// [`Error::Unsupported`] from a shell with no guest-program plane.
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
    /// The other half of [`Engine::publish_channel`], and the door a guest's
    /// output comes back through.
    ///
    /// # Errors
    ///
    /// As [`Engine::publish_channel`].
    fn take_channel(&mut self, instance: InstanceId, channel: u32) -> Result<Option<Vec<u8>>> {
        let _ = (instance, channel);
        Err(self.unsupported("take_channel"))
    }

    // ── adapter banks ───────────────────────────────────────────────────

    /// Write one adapter's planes into this load's device banks (design §8).
    ///
    /// **ADDITIVE, DEFAULTED, AND A RESIDENCY VERB** — the same shape
    /// [`Engine::publish_channel`] and [`Engine::bind_thread`] joined by. It
    /// is not one of `copy_kv`'s kin: those move state the engine already
    /// holds between places it already owns, and this lands host bytes into
    /// device residency under a name the plan declared, which is what
    /// [`Engine::load`] does. [`crate::adapter`] argues both halves.
    ///
    /// **NO RECAPTURE, AND THAT IS THE WHOLE OF DECISION 17.** An engine's
    /// graph key is a fire's composition; a bank's CONTENTS are not in it, and
    /// its addresses were reserved at load from a capacity the model text
    /// declared. So a deployment adds its two-hundredth adapter without
    /// re-recording anything, and a lane selects one with an integer in a
    /// submission ([`Lane::adapter`](crate::fire::Lane::adapter)).
    ///
    /// # Errors
    ///
    /// [`Error::Load`] for a bank this plan does not declare, an id past
    /// the declared capacity, or a plane that is not one slot's bytes;
    /// [`Error::Device`] for the residency;
    /// [`Error::Unsupported`] from a shell whose loads seat no bank.
    fn register_adapter(&mut self, registration: &AdapterRegistration) -> Result<()> {
        let _ = registration;
        Err(self.unsupported("register_adapter"))
    }

    // ── the fire path ───────────────────────────────────────────────────

    /// **The one forward verb.** Admit a frame — 1..=k steps, sealed in order
    /// — and run it.
    ///
    /// This replaces `fire`, and the replacement is the whole of alto's
    /// execution plane in one signature. `fire` was single-step and
    /// synchronous, so a runtime that wanted to be ahead of the device had to
    /// call it k times and stand between the calls; the four articles below
    /// are what that made unenforceable.
    ///
    /// * **Article 4 — static admission.** Every step is validated, the union
    ///   of their demands is taken, and it is committed once, before any
    ///   stream work. [`Error::Exhausted`] and [`Error::Impossible`] return
    ///   with **zero side effects** — the caller retries the same frame in
    ///   place, or drops it — and past the commit the stream work is
    ///   success-only. RETRY is not an outcome of a launch.
    /// * **Article 1 — saturation.** All k steps are enqueued before this
    ///   returns. Step W+1 goes onto the stream without knowing W's outcome.
    /// * **Article 2 — untouched transition.** No host read, decision,
    ///   synchronize or memcpy stands between consecutive steps. Handing the
    ///   engine the whole frame is what makes that structural rather than
    ///   conventional.
    /// * **Article 11 — guests are isolated.** `k` never appears in a guest
    ///   ABI; it is the runtime's frame policy and the engine's business.
    ///
    /// **SYNCHRONOUS UP TO ADMISSION; THE DEVICE MAY STILL BE RUNNING WHEN IT
    /// RETURNS.** A shell that settles inside the call fills every step's
    /// [`FireTicket::readouts`](crate::FireTicket) — which is what every shell
    /// in this tree does today, and what wave F2 changes. One that does not
    /// answers the id with empty readouts and settles through the runtime's
    /// broker.
    ///
    /// The frame is taken by reference and not by value because
    /// [`Error::Exhausted`] means *retry this same frame*: a by-value verb
    /// would make the caller clone every lane's tokens, pages and mask on
    /// every attempt, and put that clone on the hot path of the frame that
    /// succeeds first time.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a frame the contract does not describe,
    /// [`Error::Unsupported`] for one naming a shape this engine does not
    /// serve, [`Error::Exhausted`] for one that does not fit right now,
    /// [`Error::Impossible`] for one past a baked ceiling, [`Error::Device`]
    /// for a launch the backend refused.
    fn submit(&mut self, frame: &FrameSubmission) -> Result<FrameTicket>;

    /// **Does this engine answer `submit` before the device is done?**
    ///
    /// `false` — the default, and Metal's — means every step's readouts are
    /// filled by the time `submit` returns and the caller may settle its own
    /// bookkeeping on the spot. `true` means the receipt is a correlation id
    /// and nothing more: the outcomes arrive on the sink installed by
    /// [`Engine::on_complete`], and the numbers (if the caller wants any) are
    /// asked for by [`Engine::settle_frame`].
    ///
    /// A predicate rather than an inference from empty readouts, because a
    /// frame every lane of which asked for `Readout::None` has empty readouts
    /// too and settled synchronously — the two are different facts and a
    /// caller that conflated them would park forever on the second.
    fn settles_asynchronously(&self) -> bool {
        false
    }

    /// **Install where this engine publishes step completions.** Called once,
    /// by the thread that owns the engine, before the first `submit`.
    ///
    /// Meaningless for a synchronous engine, so the default body drops it:
    /// there is no instant at which such an engine would call it that is not
    /// simply "inside `submit`", and a caller reading `settles_asynchronously`
    /// already knows not to wait.
    fn on_complete(&mut self, sink: CompletionSink) {
        let _ = sink;
    }

    /// **Fill in a receipt's readouts, waiting for the device if it must.**
    ///
    /// The numbers door, and it is deliberately NOT the settlement path. An
    /// asynchronous engine's serving path never calls this: the guest reads
    /// its logits on the device through the epilogue's `Logits` intrinsic, and
    /// the runtime discards `LaneReadout` entirely. What calls it is a caller
    /// that came for numbers — a smoke test, a bench, a tool — and the price
    /// of numbers is a wait, stated in the verb's name instead of hidden
    /// inside `submit`.
    ///
    /// Idempotent, and a no-op by default: a synchronous engine already filled
    /// them.
    ///
    /// # Errors
    ///
    /// [`Error::Device`] for whatever the frame's work said, carrying that
    /// frame's name; [`Error::Invalid`] for a ticket this engine did not mint.
    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> Result<()> {
        let _ = ticket;
        Ok(())
    }

    /// State the fire the caller expects to submit NEXT. Advisory: an engine
    /// MAY warm state for the stated composition — bind a cached CUDA-graph
    /// binding to an exec that is not in flight, prime a descriptor table —
    /// and an engine that does nothing is exactly as correct.
    ///
    /// **A HINT IS A COMPOSITION, NOT CONTENTS.** What an engine reads off
    /// this submission is each lane's `word` and row count — the shape of
    /// the next fire — because that is what warm state is a function of.
    /// The token VALUES may be anything: the runtime's frame scheduler holds
    /// the next frame sealed before the tokens that will ride in it are
    /// sampled, and this verb is shaped so that sealed-not-yet-sampled
    /// frame can be stated as it stands. (`engine-cuda` says the same from
    /// its side: `Shell::expect` reads `word` and `tokens.len()` and
    /// nothing else.)
    ///
    /// **CORRECTNESS NEVER DEPENDS ON IT, AND THAT IS THE WHOLE CONTRACT.**
    /// A wrong hint — a fire that never comes, a composition the next fire
    /// does not have — costs the engine only the warm-up work it hid off
    /// the critical path; the fire that actually arrives keys its own state
    /// exactly as if nothing had been said (`engine-cuda`'s prebind
    /// consumes a hint per fire and a mis-stated one leaves the next fire
    /// on the rebind path it was already on). That is why this verb answers
    /// `()` and not `Result`: there is no way to serve it wrongly, so there
    /// is nothing to refuse. It is also why the default body is an explicit
    /// nothing rather than [`Error::Unsupported`] — six of this
    /// trait's verbs refuse by default because a caller must HEAR "I do not
    /// serve this";
    /// an advisory an engine ignores is indistinguishable from one it
    /// honours, and a refusal would force every caller to branch on an
    /// answer that cannot matter.
    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
    }

    // ── state movement ──────────────────────────────────────────────────

    /// Move KV pages, within this device or across a domain boundary.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] from an engine that serves no copy
    /// direction, [`Error::Invalid`] for a malformed plan,
    /// [`Error::Device`] for the transfer.
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

    // **THERE IS NO `resize_pool`** (alto design §8, wave C). An elastic
    // pool grows as a side effect of frame admission — the union demand,
    // committed atomically before any of the frame runs — and shrinks through
    // the engine's own `Supply::trim`. A verb that let a caller state a
    // mapping plan would be a second path to the same commit, which is the
    // double allocator article 8 exists to forbid. See `transfer.rs` for what
    // went with it.

    /// Encode non-text modalities into embedding rows.
    ///
    /// Takes `&mut` because the output rows are written back into the plan —
    /// the caller sized the buffer, so the encoder fills it rather than
    /// allocating a second one the caller then copies out of.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] from a load with no encoder,
    /// [`Error::Invalid`] for a payload with no anchor.
    fn encode(&mut self, plan: &mut MediaEncode) -> Result<()> {
        let _ = plan;
        Err(self.unsupported("encode"))
    }

    // ── lifetime ────────────────────────────────────────────────────────

    /// Tell the engine its caller is going away, with a reason for the log.
    ///
    /// Not a teardown — `Drop` is — but the hint a remote engine needs to stop
    /// reconnecting.
    fn disconnect(&self, message: &str) {
        let _ = message;
    }

    /// The refusal this engine answers `verb` with. A helper, not a verb.
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
