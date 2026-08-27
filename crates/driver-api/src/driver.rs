//! `trait Driver` — the verb set.
//!
//! **THE VERBS SURVIVED THE REWRITE; THE ENCODING DID NOT.** Every method
//! below was a method on the trait this replaces, and it means what it meant.
//! What changed is what they take and what they answer: a `model_ir::Plan`
//! instead of a `Vec<ModelLoadDesc>`, a `FireSubmission` of lanes instead of a
//! 62-field `LaunchPlan` of parallel CSRs, `Result<_, DriverError>` instead of
//! `Result<_, anyhow::Error>` with an `i32` status hiding inside it.
//!
//! # Object-safe, and it is checked below
//!
//! The engine holds `Vec<Box<dyn Driver>>` — a CUDA shell, a Metal shell, a
//! remote one — and dispatches on the same trait for all of them. So: no
//! generic methods, no `Self: Sized`, no `impl Trait` in return position, and
//! a `const` block at the bottom of this file that coerces a `&dyn Driver` so
//! that violating any of those is a compile error in the crate that caused it
//! rather than in the crate that tried to use it.
//!
//! # Remote is a property, not an encoding (decision 19)
//!
//! There is no wire version here, no `ExecutorRequest` enum, no tarpc service.
//! A remote driver is a type in the transport that implements this trait and
//! whose method bodies happen to be round trips; every noun it needs to send
//! is `Serialize + DeserializeOwned` because every noun in this crate is.
//! Which framing, which envelope and which version negotiation it uses are
//! *the transport's* decisions, and a contract that made them for it would be
//! wrong for every transport but one.
//!
//! # Refusal is a value
//!
//! Six of the verbs have default bodies that answer
//! [`DriverError::Unsupported`]. A Metal shell has no `copy_kv`; a shell with
//! no encoder has no `encode`. Answering "I do not serve this" is a normal
//! thing for a driver to do, and it is cheaper to write it once here than in
//! every shell that does not serve it.

use crate::caps::DeviceFacts;
use crate::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use crate::error::{DriverError, Result};
use crate::fire::{FireSubmission, FireTicket, MediaEncode};
use crate::load::{LoadRequest, Loaded};
use crate::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use crate::transfer::{KvCopy, KvHandle, PoolResize, StateCopy};

/// What the engine calls a device through.
pub trait Driver: Send + Sync {
    /// Which shell this is — `"cuda"`, `"metal"`, `"remote"`. Used in
    /// diagnostics and in [`DriverError::Unsupported`].
    fn kind(&self) -> &'static str;

    /// What the machine underneath is, once it is bound. `None` before a load,
    /// and from a driver with no device of its own.
    fn device_facts(&self) -> Option<&DeviceFacts> {
        None
    }

    /// This driver's KV pool, addressable by a peer. `None` when it is not
    /// exportable.
    fn export_kv_handle(&self) -> Option<KvHandle> {
        None
    }

    /// **THE CALLING THREAD IS THE ONE THAT WILL DRIVE THIS DEVICE FROM NOW
    /// ON.** Said once, by whoever takes ownership of the driver, before the
    /// first verb it calls.
    ///
    /// A driver may hold per-THREAD state that no value can carry across a
    /// hand-off: `driver-cuda`'s `Context` says so at the top of its own
    /// module — "`cudaSetDevice` is per-thread state, so binding somewhere
    /// other than where the fires happen strands every later call on device
    /// 0". That binding happens inside [`Driver::load`], on whichever thread
    /// booted the worker; the engine then moves the driver onto its own lane
    /// thread and every verb after that runs there.
    ///
    /// The runtime API forgives this — an unbound thread defaults to device 0
    /// and the primary context is created lazily — so a single-device
    /// deployment fires correctly by accident. The DRIVER api does not: with
    /// no current context `cuModuleLoadData` answers
    /// `CUDA_ERROR_INVALID_CONTEXT`, which is what a guest program's first
    /// registration met, and what this verb exists to prevent.
    ///
    /// Default: nothing. A driver with no thread-affine state — a remote one,
    /// a shell whose device handle is a value — needs no announcement.
    ///
    /// # Errors
    ///
    /// [`DriverError::Device`] when the thread cannot be bound at all.
    fn bind_thread(&mut self) -> Result<()> {
        Ok(())
    }

    // ── load ────────────────────────────────────────────────────────────

    /// Bake the plan, land the checkpoint, reserve the pools.
    ///
    /// The one door a model comes through. The `Plan` crosses here and
    /// `Baked` never does (decision 18): the compile happens on this side of
    /// the boundary because it is an answer about a device.
    ///
    /// # Errors
    ///
    /// [`DriverError::Load`] for a plan these budgets do not admit or a
    /// checkpoint that does not fit it, [`DriverError::Device`] for the
    /// residency.
    fn load(&mut self, request: LoadRequest) -> Result<Loaded>;

    // ── guest programs ──────────────────────────────────────────────────

    /// Compile and register a guest program, answering its id.
    ///
    /// # Errors
    ///
    /// [`DriverError::Program`] for a package that does not adopt,
    /// [`DriverError::Unsupported`] from a shell with no guest-program plane.
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
    /// that carves every ring inside [`Driver::bind_instance`], from the
    /// package's own declarations, has nothing to allocate here: for it,
    /// **binding IS registration**, and answering
    /// [`DriverError::Unsupported`] is the honest report rather than a
    /// pretend allocation the bind would then replace. The engine tolerates
    /// exactly that refusal and keeps its own host ring
    /// (`engine::driver::verbs::register_channel`).
    ///
    /// What such a shell still owes the host is a DOOR into the rings it
    /// carved, and that is [`Driver::publish_channel`] /
    /// [`Driver::take_channel`] — not this.
    ///
    /// # Errors
    ///
    /// [`DriverError::Program`] for a declaration the shell cannot allocate,
    /// [`DriverError::Unsupported`] from a shell with no guest-program plane,
    /// and from one whose rings are its instances'.
    fn register_channel(&mut self, registration: &ChannelRegistration) -> Result<RegisteredChannel> {
        let _ = registration;
        Err(self.unsupported("register_channel"))
    }

    /// Bind an instance of a registered program to a set of channels.
    ///
    /// # Errors
    ///
    /// [`DriverError::Program`] for an unknown program, a channel the package
    /// does not declare, or a seed that does not fit its cell.
    fn bind_instance(&mut self, binding: &InstanceBinding) -> Result<BoundInstance> {
        let _ = binding;
        Err(self.unsupported("bind_instance"))
    }

    /// Tear down an instance and free its wait slots.
    ///
    /// # Errors
    ///
    /// [`DriverError::Closed`] for an instance that is already gone.
    fn close_instance(&mut self, id: InstanceId) -> Result<()> {
        let _ = id;
        Err(self.unsupported("close_instance"))
    }

    /// Tear down a channel and free its ring.
    ///
    /// The counterpart of [`Driver::register_channel`], and refused by the
    /// same shells for the same reason: a ring that was carved by a bind is
    /// freed by [`Driver::close_instance`].
    ///
    /// # Errors
    ///
    /// [`DriverError::Closed`] for a channel that is already gone,
    /// [`DriverError::Unsupported`] from a shell whose rings are its
    /// instances'.
    fn close_channel(&mut self, id: ChannelId) -> Result<()> {
        let _ = id;
        Err(self.unsupported("close_channel"))
    }

    /// Push one wire cell into a bound instance's channel, answering `false`
    /// when the ring has no room — back-pressure, not a drop.
    ///
    /// **THIS IS WHAT `ChannelBinding` USED TO BE.** The contract once
    /// published a driver's private ring layout — `mirror_base`, `word_base`,
    /// `head_word_index`, … — so the host could write a cell into device
    /// memory itself; `channel.rs`'s header records why that died. A host
    /// that no longer poked the ring was left with no way to hand a guest
    /// program its input at all, and this verb is the door that replaces the
    /// pointer: the engine's own host ring on one side, the shell's device
    /// ring on the other, wire bytes between them.
    ///
    /// `channel` is the index in the package's DECLARATION order — the same
    /// numbering [`ChannelSeed::channel`](crate::channel::ChannelSeed) uses
    /// and [`InstanceBinding::channels`] maps to global ids — because that is
    /// the numbering an instance's rings are carved in.
    ///
    /// # Errors
    ///
    /// [`DriverError::Program`] for an unknown instance, a channel the
    /// instance does not carry, or a cell of the wrong width;
    /// [`DriverError::Unsupported`] from a shell with no guest-program plane.
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
    /// The other half of [`Driver::publish_channel`], and the door a guest's
    /// output comes back through.
    ///
    /// # Errors
    ///
    /// As [`Driver::publish_channel`].
    fn take_channel(&mut self, instance: InstanceId, channel: u32) -> Result<Option<Vec<u8>>> {
        let _ = (instance, channel);
        Err(self.unsupported("take_channel"))
    }

    // ── the fire path ───────────────────────────────────────────────────

    /// Run one forward pass over the submitted lanes.
    ///
    /// The hot verb. Everything it needs to decide — which windows run, which
    /// pages a lane writes, which graph is replayed — is a function of the
    /// lanes' words and row counts, and nothing on this path compiles,
    /// allocates or captures.
    ///
    /// # Errors
    ///
    /// [`DriverError::Invalid`] for a submission the contract does not
    /// describe, [`DriverError::Exhausted`] for one that does not fit right
    /// now, [`DriverError::Impossible`] for one past a baked ceiling,
    /// [`DriverError::Device`] for a launch the backend refused.
    fn fire(&mut self, submission: &FireSubmission) -> Result<FireTicket>;

    // ── state movement ──────────────────────────────────────────────────

    /// Move KV pages, within this device or across a domain boundary.
    ///
    /// # Errors
    ///
    /// [`DriverError::Unsupported`] from a driver that serves no copy
    /// direction, [`DriverError::Invalid`] for a malformed plan,
    /// [`DriverError::Device`] for the transfer.
    fn copy_kv(&mut self, copy: &KvCopy) -> Result<()> {
        let _ = copy;
        Err(self.unsupported("copy_kv"))
    }

    /// Move recurrent state between slots.
    ///
    /// # Errors
    ///
    /// As [`Driver::copy_kv`].
    fn copy_state(&mut self, copy: &StateCopy) -> Result<()> {
        let _ = copy;
        Err(self.unsupported("copy_state"))
    }

    /// Grow or shrink an elastic pool.
    ///
    /// # Errors
    ///
    /// [`DriverError::Unsupported`] from a driver whose pools are not virtual,
    /// [`DriverError::Exhausted`] when the budget will not cover the target.
    fn resize_pool(&mut self, resize: &PoolResize) -> Result<()> {
        let _ = resize;
        Err(self.unsupported("resize_pool"))
    }

    /// Encode non-text modalities into embedding rows.
    ///
    /// Takes `&mut` because the output rows are written back into the plan —
    /// the caller sized the buffer, so the encoder fills it rather than
    /// allocating a second one the caller then copies out of.
    ///
    /// # Errors
    ///
    /// [`DriverError::Unsupported`] from a load with no encoder,
    /// [`DriverError::Invalid`] for a payload with no anchor.
    fn encode(&mut self, plan: &mut MediaEncode) -> Result<()> {
        let _ = plan;
        Err(self.unsupported("encode"))
    }

    // ── lifetime ────────────────────────────────────────────────────────

    /// Tell the driver its caller is going away, with a reason for the log.
    ///
    /// Not a teardown — `Drop` is — but the hint a remote driver needs to stop
    /// reconnecting.
    fn disconnect(&self, message: &str) {
        let _ = message;
    }

    /// The refusal this driver answers `verb` with. A helper, not a verb.
    fn unsupported(&self, verb: &'static str) -> DriverError {
        DriverError::unsupported(self.kind(), verb)
    }
}

/// Object safety, checked here rather than at the first `Box<dyn Driver>`.
const _: () = {
    #[allow(dead_code)]
    fn object_safe(driver: &dyn Driver) -> &'static str {
        driver.kind()
    }
};
