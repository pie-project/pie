//! The seam to `driver-metal`.
//!
//! # A library call, not an ABI crossing
//!
//! The CUDA seam beside this one goes through the C ABI —
//! `pie_cuda_create`, `pie_cuda_launch`, a `*mut PieDriver` — because the
//! driver it talks to is C++. This one does not, because the driver it talks
//! to is Rust, and a `#[repr(C)]` boundary between two Rust crates is a second
//! spelling of a contract they already share.
//!
//! That is `metal.md`'s task 9 arriving early and from the other end: the C
//! boundary retires when its last C++ consumer does, and nothing here adds a
//! new one.
//!
//! # What this file is, and what it stopped being
//!
//! It is the DOOR, and only the door: it parses the boot config, brokers a
//! completion, calls one method on [`driver_metal::serve::Shell`], and
//! turns what comes back into the engine's own nouns. The work is next door,
//! in the crate that owns the device.
//!
//! It used to be 1,317 lines that assembled the seven-field `Machine` by
//! hand, computed KV write pages, staged nine fire tables and built the KV
//! page resolver — all of it the driver's work, done on this side of the
//! crate boundary, where every internal change to it was a breaking change.
//! `driver-cuda`'s seam beside this one is 456 lines of pure delegation and
//! always was, because a `PieDriver` is opaque and there was nothing to reach
//! into. This one had the same nouns and no wall
//! (`.wiki/driver/real-metal-north-star.md` §9, "one door").
//!
//! # What is servable today, and what is not
//!
//! The verbs split cleanly. `create`, `device_facts`, the registry four and
//! `close_*` are answered by machinery that is already ported and device
//! tested. `encode` refuses, as the CUDA side does — Metal media encode is
//! unsupported on both. `launch`, `copy_kv` and `resize_pool` are served: the
//! KV pool is ported, `launch` plans, lowers and runs a step against it,
//! `copy_kv` plans the movement and applies it, and `resize_pool` commits or
//! releases pages without moving a single address a fire has bound — the
//! pages are sparse, which is what makes that possible.
//!
//! `copy_state` is the one that refuses. A state copy moves recurrent state,
//! and no row this driver serves has any — the rows that do (`qwen3_5` and
//! its neighbours) refuse a Metal load at the row itself, so it is
//! unreachable rather than unfinished.
//!
//! It refuses by name rather than being absent. A backend that cannot be
//! selected teaches nothing; one that is selected and says exactly which verb
//! it cannot serve is a working seam with a stated hole.

use anyhow::{Result, anyhow, bail};

use ::driver_api::{
    BoundInstance, ChannelRegistrationPlan, CompletionBroker, Driver, FrameLaunchOutcome,
    FrameSubmission, InstanceBindingPlan, KvCopyPlan, MediaEncodePlan, PoolResizePlan,
    ProgramRegistration, RegisteredChannel, StateCopyPlan, SubmissionCompletion,
};

use super::settle_control;

/// The Metal shell, behind the seam's fourteen verbs.
///
/// Two fields, and the split between them is the boundary this file exists to
/// hold. The shell is the driver's state; the broker is the engine's, because
/// a completion is what a SCHEDULER waits on and a driver has no opinion about
/// it. `driver-cuda`'s seam holds exactly the same pair.
pub struct MetalDriver {
    shell: driver_metal::serve::Shell,
    broker: CompletionBroker,
}

impl MetalDriver {
    /// Open the default Metal 4 device.
    ///
    /// # Errors
    ///
    /// No Metal 4 device, or a device whose queue could not be created. Both
    /// are boot conditions, not runtime ones.
    pub fn create(config_bytes: &[u8]) -> Result<Self> {
        // ONE READER of the boot document, in `crate::driver::boot`. This
        // site used to rebuild the `from_utf8` -> `toml::Table` ->
        // `get("model")` chain itself, and so did the Vulkan and WebGPU seams
        // beside it -- three readers of one format, inside the crate whose
        // format it is, each quoting the same sentence about why a driver
        // must not be the second one.
        let boot = crate::driver::BootConfig::parse(config_bytes);
        let shell = driver_metal::serve::Shell::open(boot.config, boot.model_id)
            .map_err(|e| anyhow!("metal shell: {e}"))?;
        let facts = shell.device_facts().clone();
        let _ = facts;
        Ok(Self {
            shell,
            broker: CompletionBroker::new(),
        })
    }
}

impl Driver for MetalDriver {
    fn kind(&self) -> &'static str {
        "metal"
    }

    fn device_domain(&self) -> ::driver_api::DeviceDomain {
        ::driver_api::PIE_MEMORY_DOMAIN_METAL_PRIVATE
    }

    /// The device's stated facts.
    #[must_use]
    fn device_facts(&self) -> Option<&::driver_api::DeviceFacts> {
        Some(self.shell.device_facts())
    }

    /// Metal exports no KV handle: there is no cross-process sharing path.
    #[must_use]
    fn export_kv_handle(&self) -> Option<::driver_api::KvHandle> {
        self.shell.export_kv_handle()
    }

    /// Author the checkpoint's load plan, run it, and stage every tensor.
    ///
    /// # Errors
    ///
    /// More than one descriptor, a missing `config.json`, or a plan that will
    /// not compile or stage.
    fn load_model(
        &mut self,
        descs: Vec<::driver_api::ModelLoadDesc>,
    ) -> Result<::driver_api::DriverCapabilities> {
        self.shell.load_model(&descs).map_err(Into::into)
    }

    /// Register a PTIR program: its launch package and whatever kernels the
    /// host generated for it.
    ///
    /// # Errors
    ///
    /// A package the registry refuses (a channel whose shape it cannot serve,
    /// a stage it cannot read).
    fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64> {
        self.shell.register_program(desc).map_err(Into::into)
    }

    /// Register a channel and hand back where its ring lives.
    ///
    /// # Errors
    ///
    /// A shape or dtype the registry will not serve, or a duplicate id.
    fn register_channel(&mut self, desc: &ChannelRegistrationPlan) -> Result<RegisteredChannel> {
        let binding = self.shell.register_channel(desc)?;
        Ok(RegisteredChannel {
            driver_id: desc.driver_id,
            binding,
            reader_wait_id: desc.reader_wait_id,
            writer_wait_id: desc.writer_wait_id,
        })
    }

    /// Attach an instance of a registered program to its channels.
    ///
    /// # Errors
    ///
    /// A program id the registry does not hold, a channel an instance may not
    /// attach to, or a geometry class it does not serve.
    fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        let seeds: Vec<(u64, Vec<u8>)> = desc
            .seed_values
            .iter()
            .map(|v| (v.channel, v.bytes.clone()))
            .collect();
        let requested = (desc.requested_instance_id != 0).then_some(desc.requested_instance_id);
        let binding = self.shell.bind_instance(
            desc.program_id,
            requested,
            desc.geometry_class as u32,
            &desc.channel_ids,
            &seeds,
        )?;
        // The engine's own check, on the engine's side: the plan states which
        // instance it asked for and this is where that promise is held.
        desc.validate_binding(&binding)?;
        Ok(BoundInstance::new(
            desc.driver_id,
            desc.program_id,
            binding,
            desc.pacing_wait_id,
        ))
    }

    /// Post one sealed frame: admit it, then run its steps in order.
    ///
    /// # Errors
    ///
    /// A frame whose step tables do not describe its rows, an architecture no
    /// text serves, or a device failure. Admission is NOT an error: a frame
    /// that does not fit reports [`FrameLaunchOutcome::Exhausted`], which the
    /// engine re-posts, or `Impossible` when no eviction could ever make room.
    fn launch(&mut self, frame: &FrameSubmission) -> Result<FrameLaunchOutcome> {
        match self.shell.launch(frame)? {
            driver_metal::serve::Launched::Exhausted => Ok(FrameLaunchOutcome::Exhausted),
            driver_metal::serve::Launched::Impossible => Ok(FrameLaunchOutcome::Impossible),
            driver_metal::serve::Launched::Ran { faults } => {
                // Reported here rather than in the driver, because choosing a
                // logging backend is not a library's business. A fault kills
                // the one instance and does not fail the frame; the guest
                // behind it learns of it from the poison word
                // `driver::Registry::fire` publishes on the rings that
                // instance's host reads, not from this line.
                for (instance, why) in faults {
                    tracing::warn!(instance, %why, "metal: program faulted");
                }
                let (_raw, completion) = self.broker.launch_completion(1);
                Ok(FrameLaunchOutcome::Launched(completion))
            }
        }
    }

    /// # Errors
    ///
    /// Always. Media encode is unsupported on this backend, as it is on CUDA;
    /// both seams refuse rather than pretending.
    fn encode(&mut self, _plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        bail!("driver-metal: media encode is unsupported on this backend")
    }

    /// Move KV pages, and the rows inside them, within this pool.
    ///
    /// Settled on return: the move runs on the host, so nothing is in flight
    /// and a completion the caller waits on would wait for nothing.
    ///
    /// # Errors
    ///
    /// A call before `load_model`, a refusal from the planner, or a copy that
    /// leaves a layer's region.
    fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<SubmissionCompletion> {
        self.shell.copy_kv(desc)?;
        Ok(settle_control(&self.broker))
    }

    /// # Errors
    ///
    /// Always today: no model this backend serves has recurrent state.
    fn copy_state(&mut self, desc: &StateCopyPlan) -> Result<SubmissionCompletion> {
        self.shell.copy_state(desc)?;
        Ok(settle_control(&self.broker))
    }

    /// Commit or release KV pages so the pool holds `target_pages`.
    ///
    /// # Errors
    ///
    /// No pool loaded; a target past what the pool reserved address space
    /// for; or an arena without the memory to grow back into.
    fn resize_pool(&mut self, desc: &PoolResizePlan) -> Result<SubmissionCompletion> {
        self.shell.resize_pool(desc)?;
        // The WORK is settled already -- `Stepper::trim` waits for the GPU to
        // pass the unmap before it returns, and a growth is complete once the
        // memory is attached. The COMPLETION is a separate fact, and this line
        // used to get it wrong: it minted one and dropped the target, so the
        // engine waited on an op nobody would ever settle. `settle_control`
        // does both halves, in the order the engine's own check requires.
        Ok(settle_control(&self.broker))
    }

    /// # Errors
    ///
    /// Never today; the registry accepts a close of an id it does not hold,
    /// because a close is idempotent from the scheduler's side.
    fn close_instance(&mut self, id: u64) -> Result<()> {
        self.shell.close_instance(id);
        Ok(())
    }

    /// # Errors
    ///
    /// As [`Self::close_instance`].
    fn close_channel(&mut self, id: u64) -> Result<()> {
        self.shell.close_channel(id);
        Ok(())
    }
}
