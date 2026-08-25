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
//! unsupported on both. `copy_state` is served against the ported recurrent
//! pool, which the hybrid rows allocate; a checkpoint with no linear-attention
//! layers refuses it by name, because a fork of such a row has nothing to move
//! that `copy_kv` has not already moved.

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
        // THE ONE KEY THIS DRIVER WANTS, parsed HERE. `Shell::open` takes
        // `[model] id` and nothing else, and says why in its own doc: "a boot
        // TOML is the engine's format, and a driver that read one would be the
        // second thing entitled to an opinion about its shape." So the seam —
        // which is engine code — reads it, and the driver stays out of the
        // format's business.
        //
        // `crate::driver::BootConfig` stood here and is gone: it tolerated
        // bytes "that are a PATH rather than a document", which is how a whole
        // startup TOML once fell back to defaults in silence. Nothing tolerates
        // that now; a document that does not parse states no id, which is the
        // ordinary case and not an error.
        let model_id = std::str::from_utf8(config_bytes)
            .ok()
            .and_then(|text| text.parse::<toml::Table>().ok())
            .and_then(|doc| {
                doc.get("model")?
                    .as_table()?
                    .get("id")?
                    .as_str()
                    .map(str::to_owned)
            });
        let shell =
            driver_metal::serve::Shell::open(model_id).map_err(|e| anyhow!("metal shell: {e}"))?;
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
    fn device_facts(&self) -> Option<&::driver_api::DeviceFacts> {
        Some(self.shell.device_facts())
    }

    /// Metal exports no KV handle: there is no cross-process sharing path.
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
            driver_metal::serve::Launched::Ran { faults, ran_steps } => {
                // Reported here rather than in the driver, because choosing a
                // logging backend is not a library's business. A fault kills
                // the one instance and does not fail the frame; the guest
                // behind it learns of it from the poison word
                // `driver::Registry::fire` publishes, not from this line.
                for (instance, why) in &faults {
                    tracing::warn!(instance, %why, "metal: program faulted");
                }
                // WHAT BECAME OF EACH REQUEST, which is a separate statement
                // from "the frame ran" below and is equally required.
                //
                // The scheduler hands every batch member a `TerminalCell` and
                // resolves that member's work item by READING it; an untouched
                // cell holds `Pending`, which `resolve_from_terminal` turns
                // into `work item completion terminal outcome is still
                // Pending` and the guest sees as `channel is poisoned:
                // pipeline: forward failed`. CUDA writes these from its stream
                // callback and `remote` from the executor's reply, so a
                // host-side driver has to write them here -- see the same
                // reasoning, at length, on `vulkan::publish_terminals`.
                publish_terminals(frame, &faults, ran_steps)?;
                let (raw, completion) = self.broker.launch_completion(1);
                // SETTLED ON RETURN, which is what `Ran` means on this
                // backend and is where it differs from CUDA's seam. There the
                // target is handed DOWN (`shell.launch(frame, target)`) and
                // the shell signals it when the GPU retires the work; here
                // `Shell::launch` waits for every step itself
                // (`stepper.wait_for` per in-flight fire) and runs the bound
                // programs over the read-out before it returns, so by this
                // line there is nothing left in flight to wait for.
                //
                // Minting the completion and dropping the target -- which is
                // what stood here -- is not a slow path but a permanent one:
                // nothing else in this crate holds the target, so no epoch was
                // ever published and the scheduler parked on the wait id for
                // good. `[pie-sched] driver 0 stalled ... in_flight_launches
                // (1) settled=false` every sixty seconds, on a frame whose GPU
                // work had finished before the message was first printed.
                self.broker.notify(completion.wait_id(), raw.target_epoch);
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
    /// A checkpoint with no recurrent stack, or a seat outside the pool's.
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

/// Tell every batch member's work item what became of it.
///
/// SUCCESS unless the instance behind the cell faulted this frame, or the
/// step it belongs to never fired, in which case FAILED. `Launched::Ran`
/// means every step up to `ran_steps` was submitted, waited for and read out,
/// so there is no third answer to give: a member that neither ran nor faulted
/// does not exist on this backend.
///
/// A fault is matched by INSTANCE and not by step, which is deliberate over
/// `vulkan`'s per-step precision: `driver-metal` reports faults for the frame
/// rather than per step, and an instance whose program faulted is poisoned
/// for the whole frame anyway, so failing its other steps' cells says what is
/// already true rather than losing information.
///
/// `ran_steps` is the OTHER half of that statement and is not an
/// optimisation. A step whose device-resolved geometry was not on its channel
/// yet comes back `Filled::Early`: the driver stops there, and every step
/// from it onwards has fired nothing. Publishing SUCCESS for those members --
/// which is what a per-frame answer does -- tells the scheduler a fire
/// happened, and the guest reads a cell the fire would have written and finds
/// the previous turn's value. FAILED is the true statement, and the guest
/// sees the error rather than a fluent wrong answer.
///
/// # Errors
///
/// A step that names a number of cells its roster does not, or a null cell:
/// both would have this write past what the scheduler owns, and a raw pointer
/// is not something a later layer can check.
fn publish_terminals(
    frame: &FrameSubmission,
    faults: &[(u64, String)],
    ran_steps: usize,
) -> Result<()> {
    for (index, step) in frame.steps.iter().enumerate() {
        if step.terminal_cells.is_empty() {
            // A step the scheduler is not waiting on -- the driver's own
            // tests build frames this way -- where writing nothing is the
            // whole of the right answer.
            continue;
        }
        if step.terminal_cells.len() != step.roster_rows.len() {
            bail!(
                "driver-metal: step {index} names {} terminal cells for {} members",
                step.terminal_cells.len(),
                step.roster_rows.len()
            );
        }
        for (&cell, &row) in step.terminal_cells.iter().zip(&step.roster_rows) {
            if cell.is_null() {
                bail!("driver-metal: a member of step {index} has a null terminal cell");
            }
            let id = *frame.instance_ids.get(row as usize).ok_or_else(|| {
                anyhow!("driver-metal: roster row {row} is outside the frame's instances")
            })?;
            let word = if index >= ran_steps || faults.iter().any(|(faulted, _)| *faulted == id) {
                ::driver_api::PIE_TERMINAL_OUTCOME_FAILED
            } else {
                ::driver_api::PIE_TERMINAL_OUTCOME_SUCCESS
            };
            // SAFETY: the cell is one the scheduler owns for the life of this
            // frame and handed down in the submission; `publish` is a release
            // store into an `AtomicU32` the engine only ever reads, so the
            // reader that observes the outcome also observes everything the
            // fire wrote before it.
            unsafe {
                (*cell).publish(word);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// The ports this seam claims are the ones it can actually resolve.
    ///
    /// The claim is not free: a driver that names `PIE_DECODE_ENVELOPE_PORTS`
    /// is handed decode envelopes whose geometry it must read off its own
    /// channels, which is what `driver_metal::envelope::fill` and the peek
    /// beside it are for. `serve/load.rs` claimed 0 for as long as neither
    /// existed, and the engine answered the 0 by folding the geometry on the
    /// host -- which cannot know `EmbedTokens` and said so, by name. So the
    /// pair is asserted together: the mask, and a call into the machinery
    /// that earns it.
    #[test]
    fn the_geometry_ports_this_seam_claims_are_ones_it_can_resolve() {
        // The number is a MEASUREMENT, not a constant this file also owns.
        // `forward.rs` gates the decode envelope on
        // `device_port_mask & required == required`, where `required` is
        // computed per envelope by `envelope_required_ports`; the envelope a
        // real `pie run` of qwen3-0.6b on THIS machine built asked for this,
        // and said so while the claim was still zero:
        //
        //     decode envelope on a driver without device geometry ports
        //     (mask 0x0, needs 0x25): falling back to host-evaluated
        //     serialized execution
        //
        // Quoting the observed requirement rather than the constant this
        // driver reports keeps the two sides independent: a
        // `PIE_DECODE_ENVELOPE_PORTS` that changed under us would fail here
        // rather than agree with itself.
        const MEASURED_REQUIREMENT: u32 = 0x25;
        // The gate itself, once, so the control below asks the same question
        // of a different mask rather than restating the arithmetic. Written
        // as a closure because `0 & MEASURED_REQUIREMENT` spelled inline is
        // `clippy::erasing_op` -- correct about the expression and wrong
        // about the intent, which is that the ZERO this driver used to report
        // does not pass a gate the real one does.
        let covers = |mask: u32| mask & MEASURED_REQUIREMENT == MEASURED_REQUIREMENT;
        let claimed = ::driver_api::PIE_DECODE_ENVELOPE_PORTS;
        assert!(
            covers(claimed),
            "the mask this seam reports must cover what a real decode envelope asked for"
        );
        // The control: the value this file used to report fails that same
        // gate, which is why the run took the host fallback and died on
        // `EmbedTokens is not host-derivable`.
        assert!(
            !covers(0),
            "a mask of zero must NOT satisfy the gate, or this test proves nothing"
        );
        // And what is NOT claimed is stated too, because this driver stops
        // strictly short of the wgpu seam beside it: the device-geometry set
        // names the pages, the CSR and the write descriptor, and
        // `serve::launch` derives every row's write target from its position
        // rather than reading one. `envelope::fill` refuses that class by
        // name for the same reason.
        assert_ne!(
            claimed & ::driver_api::PIE_DEVICE_GEOMETRY_PORTS,
            ::driver_api::PIE_DEVICE_GEOMETRY_PORTS,
            "this seam claims the decode envelope's ports, not every geometry port"
        );
        assert_eq!(
            claimed & ::driver_api::PIE_DEVICE_PORT_ATTN_MASK,
            0,
            "a dense device mask reaches the Metal text through the region table, not the \
             plan, so the port is not claimed"
        );
        // The machinery the claim promises, asked for an instance that is not
        // there: it must answer by NAME rather than by not existing.
        let registry = driver_metal::channel::Registry::new();
        let refused =
            driver_metal::envelope::geometry(&registry, 7, 16).expect_err("there is no instance 7");
        assert!(
            refused.to_string().contains('7'),
            "the refusal names the instance it could not find: {refused}"
        );
    }
}
