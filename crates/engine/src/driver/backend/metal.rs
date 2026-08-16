//! The seam to `driver-metal`.
//!
//! Unlike the CUDA seam beside it, this one is a library call and not a C ABI
//! crossing: the driver it talks to is Rust, and a `#[repr(C)]` boundary
//! between two Rust crates is a second spelling of a contract they share.
//!
//! It is the DOOR, and only the door: it parses the boot config, brokers a
//! completion, calls one method on [`driver_metal::serve::Shell`], and turns
//! what comes back into the engine's own nouns. The work is next door, in the
//! crate that owns the device.
//!
//! `create`, `device_facts`, the registry four and `close_*` are answered by
//! machinery that is ported and device tested. `launch`, `copy_kv` and
//! `resize_pool` are served against the ported KV pool — `resize_pool` commits
//! or releases pages without moving an address a fire has bound, the pages
//! being sparse. `encode` refuses, as the CUDA side does: Metal media encode is
//! unsupported on both. `copy_state` refuses because no row this driver serves
//! has recurrent state — the rows that do refuse a Metal load at the row
//! itself. Refusing by name rather than being absent is what makes this a
//! working seam with a stated hole.

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
        // ONE READER of the boot document, in `crate::driver::boot`: the crate
        // that owns the format parses it, and no driver seam is a second.
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
                // `driver::Registry::fire` publishes, not from this line.
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
        // memory is attached. The COMPLETION is the separate fact, and
        // `settle_control` mints it and settles it in the order the engine's
        // own check requires.
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
