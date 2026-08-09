//! The seam to `driver-vulkan`.
//!
//! # A library call, like the Metal seam beside it
//!
//! No C ABI, no `*mut PieDriver`: the driver is Rust and a `#[repr(C)]`
//! boundary between two Rust crates is a second spelling of a contract they
//! already share. This file is the door and only the door.
//!
//! # Where loading lives, and why it is not next door
//!
//! `driver-metal` loads a checkpoint inside the driver. This one does not,
//! and the difference is deliberate rather than unfinished.
//!
//! `driver-vulkan` keeps `model` and `model-loader` as DEV-dependencies and
//! proves the closure with a test (`tests/pure.rs`), because a driver that
//! depended on a checkpoint FORMAT would be a driver that could not be handed
//! bytes. What it exposes instead is `Shell::hold(name, bytes)` -- a name,
//! some bytes, and no opinion about where they came from. So the conversion
//! from a publisher's tensor names to the ones a plan states runs HERE, on
//! the side that already depends on `model` and `model-compiler` for its own
//! reasons.
//!
//! That also means [`VulkanDriver::create`] cannot open a shell. A
//! `driver_vulkan::shell::Shell` is a device plus a model's two plans plus a
//! cache shaped for that model, and none of the last three exists until a
//! checkpoint has been read. So `create` opens a device far enough to state
//! its facts and then closes it; `load_model` opens the shell that serves.
//!
//! # What is servable, and what refuses by name
//!
//! `create`, `device_facts`, the registry four, `close_*`, `launch`,
//! `copy_kv` and `resize_pool` are served. `encode` and `copy_state` refuse,
//! as they do on both other backends -- there is no separate encode step in
//! this driver, and no model it serves holds a recurrent state.
//!
//! `export_kv_handle` answers `None`: there is no cross-process sharing path.
//!
//! # A verb that has already finished still has to say so
//!
//! `copy_kv` and `resize_pool` are host work -- this driver's buffers are
//! coherent, so the first is a `memmove` and the second an allocation, and
//! both are done by the time the verb returns. This seam took that to mean
//! there was nothing to settle, minted a completion and dropped its target.
//!
//! The engine then waited forever. Measured here, on `prefix-tree-kv-cache`,
//! which forks one prefill into two branches and each into two leaves:
//!
//! ```text
//! in_flight_control: KV copy pipeline Some(..) settled=false
//! [pie-sched] driver 0 stalled for 370.00s (no progress, work queued ...)
//! ```
//!
//! With [`settle_control`] the same run reaches its end in 11.8 s. It then
//! fails elsewhere, with `pipeline is closed` and no driver fault and no
//! stall, exactly as the wgpu seam does after the same repair.
//!
//! That second failure is not this driver's, and it is worth saying WHY
//! rather than only that. Backtracing every `PipelineScope::close` through
//! the run puts the call in the GUEST: `run_ahead` closes its pipeline the
//! instant the budget is spent, which is `Pipeline`'s own documented rule and
//! is worth +9.5% to +18.7% to a lane that owns its stream. This example does
//! not own its stream -- it builds four leaves on one `tree_pipeline` and
//! then generates from each, so the first leaf's `run_ahead` closes the
//! pipeline the other three still need.
//!
//! The obvious repair does not work either, which is what makes it a contract
//! problem rather than a typo. Giving each leaf its own pipeline trades one
//! engine refusal for the other:
//!
//! ```text
//! pipeline: KV working set is already scoped to pipeline 0000..0001
//! ```
//!
//! because a working set is scoped to the first pipeline that fires it and
//! the scope is reclaimable only once the old one has both closed AND
//! drained. So the example is caught between two rules, and no `run_ahead`
//! variant that leaves the stream open exists to escape them. Reported with
//! the mechanism, not fixed here: the fix is a decode loop's shape, and this
//! is a driver.
//!
//! `beam-search`, the other forking example, obeys both rules -- one
//! pipeline, one `run_ahead`, forks taken inside the callback -- and stops at
//! a third wall that is also not a backend's: its per-lane ancestry mask is a
//! channel-bound dense `AttnMask`, which `fire/geometry.rs` declines to the
//! pool-owned device-geometry class because "the envelope composes batched
//! lanes and has no per-lane mask state -- on any backend, not just this
//! one". So no forking example completes here yet, and none of the three
//! reasons is a Vulkan one. `Shell::fork` itself is measured, in
//! `tests/device.rs`: a fork answers as its source does, diverges when fed
//! different tokens, and refuses three ways.
//!
//! A `launch` here is two halves and this seam is where they meet: the shell
//! fires the model rows and hands back one `Step` per step, and then the
//! frame's PROGRAMS are fired over those steps -- each instance over its own
//! request's distribution -- so the answer reaches the channels the engine
//! reads. The registry lives on this side rather than in the shell (see
//! `programs` below), which is why the join is here rather than one layer
//! down.
//!
//! That join is now measured rather than asserted. `mod tests` proves the
//! registry answers -- a program, a channel and an instance, with no device
//! open at all -- but a registry that hands back well-formed ids for a
//! program nobody fires is indistinguishable from one that works, and what it
//! would hide is a channel that never fills, which is a guest waiting
//! forever. So `tests/gpu/tests/vulkan_programmable_sampler.rs` runs the
//! plane instead: `mirostat-v2-sampling` builds its own forward pass, reads
//! each step's logits back over a channel, decides the token host-side and
//! feeds a control value forward into the next step. Eight tokens in 4.2
//! seconds under the validation layer, with the guest's `mu` moving from the
//! 6.0 it was given to 10.41 -- and that movement is the round trip's
//! signature, since the only thing that can move it is a surprise computed
//! from a distribution that actually came back.
//!
//! Two more gates cover the plane's other directions. `vulkan_sampling_
//! primitives` runs a whole op set inside ONE epilogue and reads six channels
//! out of it, three of them a full 151936 wide. `vulkan_grammar_constrained`
//! runs the only direction where the host writes INTO the epilogue every
//! step: `json-schema-constrained-decoding` holds a grammar matcher host-side
//! and puts its allowed-token mask into a channel that `masked_argmax` reads
//! as an operand, then re-puts a different mask on the very next fire of the
//! same instance. A mask bound once and reused is still a legal decode, just
//! one against a stale grammar -- so that gate makes the model answer with an
//! object whose required keys are `zqx` and `wbn`, which no continuation of
//! its prompt would ever choose. Getting them back is only possible if the
//! buffer the host wrote is what the argmax ranged over.
//!
//! A served `launch` is not an unconditional one. A `LaunchPlan` can name
//! eight things this driver does not implement -- recurrent state, a user
//! mask, `max_layers`, `hook_page_mask`, `dense_device_mask`, images, audio
//! and pre-embedded rows -- and each is REFUSED at admission by its own
//! name, in `driver_vulkan::frames::unserved_in`, before anything is written
//! to a cache. Ignoring any of them produces no crash and no NaN: it
//! produces fluent, wrong text, which is the failure this crate is built
//! against.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};

use ::driver_api::{
    BoundInstance, ChannelRegistrationPlan, CompletionBroker, Driver, FrameLaunchOutcome,
    FrameSubmission, InstanceBindingPlan, KvCopyPlan, MediaEncodePlan, PoolResizePlan,
    ProgramRegistration, RegisteredChannel, StateCopyPlan, SubmissionCompletion,
};

use super::settle_control;

/// How many KV pages a shell is opened with.
///
/// A number the boot config can override (`[model] kv_pages`). It exists
/// because `driver-vulkan`'s pool is allocated at a page count rather than
/// reserved and committed: it can be resized -- `resize_pool` is served -- but
/// it has to start somewhere, and a scheduler that has not asked for anything
/// yet has no better number to offer.
const DEFAULT_KV_PAGES: u32 = 1024;

/// The Vulkan shell, behind the seam's fourteen verbs.
///
/// The shell is `Option` because it does not exist until a checkpoint has
/// been read -- see the module doc. `facts` is kept beside it because the
/// engine asks for them at `create`, before there is a shell to ask.
pub struct VulkanDriver {
    shell: Option<driver_vulkan::shell::Shell>,
    facts: ::driver_api::DeviceFacts,
    /// The compiled SPIR-V, by entrypoint. Read once at `create`, because a
    /// module set is a property of the build rather than of a model, and
    /// re-reading it per load would let two models on one device disagree
    /// about which kernels exist.
    modules: BTreeMap<String, Vec<u8>>,
    /// Where the boot config pointed, kept for the message a failed load
    /// gives: "no kernel named X" is unhelpful without saying where it looked.
    module_dir: std::path::PathBuf,
    kv_pages: u32,
    broker: CompletionBroker,
    /// The PTIR channel plane: programs, channels and their instances.
    ///
    /// Beside the shell rather than inside it, and alive from `create`
    /// rather than from `load_model`, because nothing in it is about a
    /// model. It is portable, deviceless host memory -- `driver-metal`'s
    /// `channel.rs` is `pub use driver::*` for the same reason -- and a
    /// registry that only existed once a checkpoint had been read would
    /// refuse a program the engine is entitled to register at any time,
    /// for a reason that has nothing to do with the program.
    programs: driver_vulkan::programs::Programs,
}

impl VulkanDriver {
    /// Open the default Vulkan device and read the kernel modules.
    ///
    /// # Errors
    ///
    /// No Vulkan device, or no readable module directory. Both are boot
    /// conditions rather than runtime ones, and both are worth failing at
    /// boot: a server that started without kernels would refuse its first
    /// request instead of its first configuration.
    pub fn create(config_bytes: &[u8]) -> Result<Self> {
        // The boot TOML is the ENGINE's format, read here for the same reason
        // the Metal seam reads it here: a driver that parsed it would be the
        // second thing entitled to an opinion about the file's shape.
        let Boot {
            module_dir,
            kv_pages,
        } = boot_of(config_bytes)?;
        let modules = read_modules(&module_dir)?;

        // Opened and dropped: the facts are all `create` owes, and holding a
        // device open until `load_model` would hold the whole GPU against a
        // model that might never arrive.
        let facts = {
            let device = driver_vulkan::device::Device::open()
                .map_err(|e| anyhow!("driver-vulkan: no device: {e}"))?;
            driver_vulkan::facts::of(&device)
        };
        Ok(Self {
            shell: None,
            facts,
            modules,
            module_dir,
            kv_pages,
            broker: CompletionBroker::new(),
            programs: driver_vulkan::programs::Programs::new(),
        })
    }

    /// The shell, or a message saying which verb was called before a load.
    fn shell(&mut self, what: &'static str) -> Result<&mut driver_vulkan::shell::Shell> {
        self.shell.as_mut().ok_or_else(|| {
            anyhow!(
                "driver-vulkan: {what} before load_model. A shell is a device plus a model's \
                 plans plus a cache shaped for that model, and none of the last three exists \
                 until a checkpoint has been read."
            )
        })
    }

    /// A driver with no device behind it, for the verbs that do not need one.
    ///
    /// This is the state a real driver is in between `create` and
    /// `load_model` -- no shell, an empty registry -- with two differences:
    /// its facts were never measured, so they are the specification's floor
    /// rather than a device's answer, and it holds no modules, because a
    /// module set is read from a directory and CI has no reason to have one.
    ///
    /// It exists because CI has no GPU, and this seam's seven existing tests
    /// all SHORT-CIRCUIT when `PIE_KERNELS_VULKAN_SPV_DIR` is unset or no
    /// device answers -- so on a machine without a card, not one verb of the
    /// fourteen was ever called. An `impl` that compiles proves only that a
    /// method exists; `todo!()`, `unimplemented!()` and a silent `Ok(())` all
    /// compile too, and the last would take a KV copy the scheduler then
    /// believes happened. Ten verbs never touch a device -- the five registry
    /// ones, the two by-name refusals, and the three "before `load_model`"
    /// errors -- so ten can be walked here, and are.
    #[cfg(test)]
    pub(super) fn without_adapter() -> Self {
        Self {
            shell: None,
            facts: driver_vulkan::facts::floor(),
            modules: BTreeMap::new(),
            module_dir: std::path::PathBuf::from("<none: this driver read no modules>"),
            kv_pages: 0,
            broker: CompletionBroker::new(),
            programs: driver_vulkan::programs::Programs::new(),
        }
    }
}

impl Driver for VulkanDriver {
    fn kind(&self) -> &'static str {
        "vulkan"
    }

    fn device_domain(&self) -> ::driver_api::DeviceDomain {
        ::driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE
    }

    /// The device's stated facts.
    fn device_facts(&self) -> Option<&::driver_api::DeviceFacts> {
        Some(&self.facts)
    }

    /// Vulkan exports no KV handle: there is no cross-process sharing path.
    fn export_kv_handle(&self) -> Option<::driver_api::KvHandle> {
        None
    }

    /// Identify the checkpoint, assemble its text, open a shell, and stage
    /// every weight the decode plan binds.
    ///
    /// # Errors
    ///
    /// More than one descriptor, a snapshot no catalog row matches, a plan
    /// that will not compile or execute, or a device that will not allocate.
    fn load_model(
        &mut self,
        descs: Vec<::driver_api::ModelLoadDesc>,
    ) -> Result<::driver_api::DriverCapabilities> {
        let [desc] = descs.as_slice() else {
            bail!(
                "driver-vulkan: {} model descriptors. This backend serves one model per \
                 device -- a shell holds one text and one cache shaped for it.",
                descs.len()
            );
        };
        let path = desc.snapshot_dir.as_path();
        let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(path)
            .map_err(|e| anyhow!("driver-vulkan: unreadable checkpoint at {path:?}: {e}"))?;
        // THE TENSORS decide which row this is, not a name in a config file:
        // `catalog::identify` is the same door `driver-metal`'s load uses, and
        // one identification means one answer.
        // `Override::None`: the tensors decide. A boot config that named a
        // model would be a second identification, and the two disagree in the
        // direction that matters -- a checkpoint served as the wrong row is
        // not refused, it is fluent and wrong.
        let row = model::catalog::identify(&meta, &model::catalog::Override::None)
            .map_err(|e| anyhow!("driver-vulkan: {path:?} matches no catalog row: {e}"))?;
        let (text, deployment) = text_of(row)?;
        // Read from the text BEFORE the shell takes it: what the decode plan
        // binds is what has to be staged, and the shell owns the plan after
        // `open`.
        let wanted = bound_names(&text);
        let mut shell = driver_vulkan::shell::Shell::open(
            text,
            driver_vulkan::shell::Deployment {
                pages: self.kv_pages,
                ..driver_vulkan::shell::Deployment::default()
            },
            self.modules.clone(),
        )
        .map_err(|e| {
            anyhow!(
                "driver-vulkan: the shell would not open with the modules in {:?}: {e}",
                self.module_dir
            )
        })?;

        for (name, bytes) in stage(path, &meta, row, &wanted)? {
            shell
                .hold(&name, &bytes)
                .map_err(|e| anyhow!("driver-vulkan: `{name}` would not stage: {e:?}"))?;
        }
        let shape = shell.shape();
        self.shell = Some(shell);
        Ok(::driver_api::DriverCapabilities {
            abi_version: ::driver_api::PIE_DRIVER_ABI_VERSION,
            total_pages: shape.pages,
            kv_page_size: shape.page_size,
            // No swap pool and no recurrent-state cache: this driver has
            // neither, and `copy_state` refuses by name for the same reason.
            swap_pool_size: 0,
            // Device to device, and only that. `Pool::copy_plan` moves whole
            // pages inside the one KV buffer -- which is what a prefix-cache
            // hit is -- and refuses any plan whose ends are not both
            // `PIE_MEMORY_DOMAIN_VULKAN_DEVICE`.
            //
            // Advertising it is new, and it only became true once the
            // scheduler stopped stamping `PIE_MEMORY_DOMAIN_CUDA_DEVICE` on
            // every plan regardless of backend. Host directions stay off:
            // there is no swap pool here, so `swap_pool_size` is 0 and a
            // device-to-host copy has nowhere to land.
            kv_copy_domain_mask: ::driver_api::KV_COPY_DEVICE_TO_DEVICE,
            // No recurrent-state cache. `driver-metal` answers this one with
            // `deployment.recurrent.is_some()`; this backend cannot serve a
            // hybrid stack at all, so the answer is a constant.
            //
            // A constant is only honest if a hybrid cannot arrive, and that
            // was a belief here until it was checked. It holds, but not for
            // the reason a guard here would give: every hybrid family in the
            // catalog is turned away one line above, by `row.trace`, because
            // none has a Metal text -- and the row says so far better than
            // this seam could, naming the backend that does serve it.
            // `a_model_that_holds_a_recurrent_state_is_refused_before_it_is_
            // staged` scans the whole catalog for that, so the day a hybrid
            // gains a Metal text these three lines stop being true and a test
            // says so.
            rs_cache_required: false,
            rs_cache_slots: 0,
            rs_cache_slot_bytes: 0,
            // Not elastic, and the reason is the COST rather than the
            // ability. `Shell::resize_pool` works: it preserves the pages
            // that survive, refuses by name a shrink that would strand a
            // seated conversation, and leaves the pool untouched when the
            // machine will not stage the new one. What it cannot be is
            // CHEAP. `Pool::resize` reads every layer's whole buffer to host
            // memory and writes the survivors into a fresh one, so the charge
            // is the pool's size twice over and not the delta's.
            //
            // Measured at 256 pages of qwen3-0.6b: handing back one page
            // takes 2.77 s and handing back a hundred and twenty-six takes
            // 0.74 s -- the deeper cut is nearly four times cheaper, because
            // it fills a smaller destination. The cheapest trim this pool
            // offers is the largest one, which is the opposite of what a trim
            // task is for. It also peaks at both sizes at once, so a shrink
            // asked for under memory pressure needs more memory than not
            // shrinking at all.
            //
            // So both numbers are zero together, which is the condition
            // `bootstrap` reads before it starts a trim task at all, and the
            // task never starts.
            //
            // This said "nothing can be given back page-wise" until it was
            // measured, and that was false -- a shrink does free the old
            // buffers, at any granularity asked for. Right answer, wrong
            // reason. `giving_back_one_page_costs_what_giving_back_half_the_
            // pool_costs` in `tests/device.rs` now pins the real one, and
            // goes red if this pool ever becomes page-wise.
            elastic_page_bytes: 0,
            elastic_budget_pages: 0,
            has_mtp_logits: false,
            has_mtp_drafts: false,
            has_value_head: false,
            // Sinks this backend cannot honour. Every one of them would bind
            // and then run as a silent no-op, which is worse than a refusal
            // at the door.
            has_kv_envelopes: false,
            has_attn_score: false,
            has_attn_page_mask: false,
            has_lora: false,
            model_site_summary: ::driver_api::ModelSiteSummary::default(),
            device_geometry_port_mask: ::driver_api::PIE_DECODE_ENVELOPE_PORTS,
            // The ceilings a batch is formed under, and they are the arena's:
            // `Shell::open` sizes one fire's scratch, and a fire wider than
            // this has nothing to run in.
            max_forward_tokens: 4096,
            max_forward_requests: 256,
            max_page_refs: shape.pages,
            // The row's answers, not a config's: the checkpoint was
            // identified once and these come from that identification.
            arch_name: deployment.advertised.arch.to_string(),
            model_id: row.id().to_string(),
            vocab_size: deployment.shape.vocab,
            max_model_len: deployment.advertised.max_model_len,
            activation_dtype: "bf16".to_string(),
            hidden_size: deployment.shape.hidden,
            // False about the BACKEND rather than about the row: there is no
            // encode entry point here at all, so a model with a vision tower
            // is served as its text half. `Shell::encode` refuses by name.
            supports_media_encode: false,
            snapshot_dir: path.display().to_string(),
            kv_handle: None,
            // The modules are read from disk already built; nothing upstream
            // generates a kernel for this driver.
            codegen_backend: String::new(),
        })
    }

    /// Register a PTIR program: its launch package and its emitted kernels.
    ///
    /// Served from `create`, before any model: see the `programs` field for
    /// why the plane does not wait on a checkpoint.
    ///
    /// # Errors
    ///
    /// A launch package the registry refuses -- no stages, a channel shape it
    /// cannot serve, a stage it cannot read.
    fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64> {
        self.programs
            .register_program(desc)
            .map_err(|e| anyhow!("driver-vulkan: {e}"))
    }

    /// Register a channel and hand back where its ring lives.
    ///
    /// The ring is HOST memory. Nothing about the channel plane is on the
    /// GPU: it is a different layer from the model forward, and a device
    /// buffer for it would be a round trip per cell for data no shader reads.
    ///
    /// # Errors
    ///
    /// A shape the registry will not serve, or a duplicate id.
    fn register_channel(&mut self, desc: &ChannelRegistrationPlan) -> Result<RegisteredChannel> {
        let binding = self
            .programs
            .register_channel(desc)
            .map_err(|e| anyhow!("driver-vulkan: {e}"))?;
        Ok(RegisteredChannel {
            driver_id: desc.driver_id,
            binding,
            reader_wait_id: desc.reader_wait_id,
            writer_wait_id: desc.writer_wait_id,
        })
    }

    /// Bind an instance of a registered program to registered channels.
    ///
    /// The binding is VALIDATED before it is returned, and a rejected one is
    /// closed on the way out: `plan.validate_binding` is what catches a
    /// driver that answered a different instance id than the one the engine
    /// requested, and an instance left open behind that error would be a leak
    /// nothing later has a handle to close.
    ///
    /// # Errors
    ///
    /// An unknown program or channel, a geometry class this build has no name
    /// for, or a binding that does not match what was asked.
    fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        // `requested_instance_id` is 0 for "any", which the registry spells
        // as `None`.
        let requested = (desc.requested_instance_id != 0).then_some(desc.requested_instance_id);
        let seeds: Vec<(u64, Vec<u8>)> = desc
            .seed_values
            .iter()
            .map(|v| (v.channel, v.bytes.clone()))
            .collect();
        let binding = self
            .programs
            .bind_instance(
                desc.program_id,
                requested,
                // The ABI's u32, which is what the registry's `Geometry::
                // from_wire` reads. `GeometryClass` is the engine's typed
                // spelling of the same three values -- `driver-api` static-
                // asserts the pairing.
                desc.geometry_class as u32,
                &desc.channel_ids,
                &seeds,
            )
            .map_err(|e| anyhow!("driver-vulkan: {e}"))?;
        if let Err(error) = desc.validate_binding(&binding) {
            self.programs.close_instance(binding.instance_id);
            return Err(error);
        }
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
    /// A frame whose step tables do not describe its rows, or a device
    /// failure. Admission is NOT an error: a frame that does not fit reports
    /// [`FrameLaunchOutcome::Exhausted`], which the engine re-posts, or
    /// `Impossible` when no growth could ever make room.
    fn launch(&mut self, frame: &FrameSubmission) -> Result<FrameLaunchOutcome> {
        let page = driver_vulkan::facts::PAGE_SIZE;
        match self.shell("launch")?.admit(frame) {
            Ok(Some(driver_vulkan::frames::Launched::Exhausted)) => {
                return Ok(FrameLaunchOutcome::Exhausted);
            }
            Ok(Some(driver_vulkan::frames::Launched::Impossible)) => {
                return Ok(FrameLaunchOutcome::Impossible);
            }
            Ok(Some(driver_vulkan::frames::Launched::Ran(_)) | None) => {}
            Err(e) => return Err(anyhow!("driver-vulkan: {e}")),
        }
        // A step at a time, because a DEVICE-RESOLVED step's tokens are what
        // the step before it PUT on a channel: they do not exist until that
        // step has both fired and had its program run. The stronger order --
        // convert every step before firing any -- is kept for a frame of
        // ordinary host-wire steps by `Shell::launch`, which this path does
        // not call; here the two halves are interleaved, and a step that
        // cannot be prepared has fired nothing of its own.
        let mut faults = Vec::new();
        for sub in &frame.steps {
            let filled = driver_vulkan::envelope::fill(&self.programs, frame, sub, page)
                .map_err(|e| anyhow!("driver-vulkan: {e}"))?;
            let plan = match filled {
                driver_vulkan::envelope::Filled::Ready(plan) => plan,
                // Nothing to fire and nothing wrong: the producer has not
                // run. Every member of this step is told to come back, which
                // is what the scheduler's re-post is for.
                driver_vulkan::envelope::Filled::Early { channel } => {
                    tracing::debug!(
                        channel,
                        "vulkan: a step's geometry channel is not filled yet"
                    );
                    tracing::warn!(
                        channel,
                        members = sub.roster_rows.len(),
                        "vulkan: a step's geometry channel is unfilled at fire time; \
                         v14 admission is supposed to make that impossible, so its \
                         members are failed rather than re-posted"
                    );
                    let early = vec![driver_vulkan::frames::Ran::Early; sub.roster_rows.len()];
                    publish_terminals(&sub.terminal_cells, &early)?;
                    break;
                }
            };
            let (requests, tokens) = self
                .shell("launch")?
                .prepare(&plan)
                .map_err(|e| anyhow!("driver-vulkan: {e}"))?;
            let step = self
                .shell("launch")?
                .serve(&requests, &tokens)
                .map_err(|e| anyhow!("driver-vulkan: {e}"))?;
            // The distributions do not come back through this return, and
            // that is the seam's shape rather than a loss: a step's answer is
            // read by the frame's own PROGRAMS, which put it on the channels
            // the engine reads. Firing them is this seam's job because the
            // registry is this seam's -- it is alive from `create`, before
            // there is a shell -- so the driver hands back the step and the
            // two halves are joined here.
            let ran = driver_vulkan::frames::run_programs(
                &mut self.programs,
                &frame.instance_ids,
                sub,
                &step,
                &mut faults,
            )
            .map_err(|e| anyhow!("driver-vulkan: {e}"))?;
            publish_terminals(&sub.terminal_cells, &ran)?;
        }
        // Logged and not returned, as Metal logs them: a fault kills the one
        // instance that faulted, and the requests batched with it ran. The
        // guest behind the dead one is not left waiting --
        // `driver::Registry::fire` publishes the fault on the rings that
        // instance's host READS, and the pipeline turns that poison word into
        // the guest's error -- so this line is an operator's record rather
        // than the only report.
        for (instance, why) in faults {
            tracing::warn!(instance, %why, "vulkan: program faulted");
        }
        // Settled here, because it is already settled. A completion is a
        // promise that the frame's work has finished, and the asynchronous
        // backends keep it by notifying from wherever the work actually lands
        // -- `remote` from its RPC task, CUDA from its stream. This driver has
        // nowhere to notify FROM: `Shell::serve` waits on the fence itself and
        // everything the frame asked for has happened by the time it returns.
        //
        // Handing back an unnotified completion is not a smaller version of
        // that. It is a promise nobody keeps: the scheduler parks the lane on
        // it and re-reports the same frame forever --
        //
        //     [pie-sched] driver 0 stalled for 1690s (no progress, work
        //     queued or in flight) ... batch of 1 (settled=false, age=1690s)
        //
        // -- which is what a real `pie serve` on this backend did, for every
        // turn, after a fire that had already computed the right answer in
        // 563 milliseconds.
        let (_raw, completion) = self.broker.launch_completion(1);
        self.broker.notify(completion.wait_id(), 1);
        Ok(FrameLaunchOutcome::Launched(completion))
    }

    /// # Errors
    ///
    /// Always. There is no separate encode step in this driver: a fire records
    /// and submits in one call, so there is no encoded frame to hand back.
    /// CUDA and Metal refuse the same verb.
    fn encode(&mut self, _plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        bail!("driver-vulkan: media encode is unsupported on this backend")
    }

    /// Move KV pages, and the rows inside them, within this pool.
    ///
    /// The move itself is finished on return: every buffer this driver
    /// allocates is host-visible and coherent, so it is a host `memmove` with
    /// no command buffer and nothing in flight. The COMPLETION is a separate
    /// fact, and this seam used to get it wrong -- it minted one and dropped
    /// the target, so the engine waited on an op nobody would ever settle.
    /// [`settle_control`] does both halves; see it for the order.
    ///
    /// # Errors
    ///
    /// A call before `load_model`, or a plan that leaves a layer's region.
    fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<SubmissionCompletion> {
        self.shell("copy_kv")?
            .copy_kv(desc)
            .map_err(|e| anyhow!("driver-vulkan: {e}"))?;
        Ok(settle_control(&self.broker))
    }

    /// # Errors
    ///
    /// Always: no model this backend serves has recurrent state.
    fn copy_state(&mut self, _desc: &StateCopyPlan) -> Result<SubmissionCompletion> {
        bail!("driver-vulkan: no model this driver serves holds a recurrent state")
    }

    /// Rebuild the KV pool at `target_pages`.
    ///
    /// # Errors
    ///
    /// A call before `load_model`, a shrink that would strand a conversation,
    /// or a device that will not allocate the new size.
    fn resize_pool(&mut self, desc: &PoolResizePlan) -> Result<SubmissionCompletion> {
        self.shell("resize_pool")?
            .resize_pool(desc)
            .map_err(|e| anyhow!("driver-vulkan: {e}"))?;
        // The resize returns with the new buffers allocated and the old ones
        // freed, so the only thing left is to say so.
        Ok(settle_control(&self.broker))
    }

    /// # Errors
    ///
    /// Never; the registry accepts a close of an id it does not hold, because
    /// a close is idempotent from the scheduler's side.
    fn close_instance(&mut self, id: u64) -> Result<()> {
        // A close BEFORE a load is answered rather than refused, and so is a
        // close of an id the registry does not hold: teardown races both ways
        // and a fault logged per conversation would be noise about a verb
        // that was right to be called.
        self.programs.close_instance(id);
        Ok(())
    }

    /// # Errors
    ///
    /// As [`Self::close_instance`].
    fn close_channel(&mut self, id: u64) -> Result<()> {
        self.programs.close_channel(id);
        Ok(())
    }
}

/// What the boot TOML says, and nothing that needs a device to find out.
///
/// Its own type and its own function so that the file's shape is testable
/// where no Vulkan device exists. This is a CONTRACT with the worker, which
/// writes the file (`write_vulkan_startup_toml`), and a contract with one
/// reader and no test is one that gets discovered broken by a server booting
/// with the wrong pool size and saying nothing about it.
struct Boot {
    /// Where the compiled SPIR-V modules are.
    module_dir: std::path::PathBuf,
    /// How many KV pages the shell allocates.
    kv_pages: u32,
}

/// Read the boot TOML.
///
/// # Errors
///
/// No module directory, from either the file or the environment.
fn boot_of(config_bytes: &[u8]) -> Result<Boot> {
    // The boot TOML is the ENGINE's format, read here for the same reason
    // the Metal seam reads it here: a driver that parsed it would be the
    // second thing entitled to an opinion about the file's shape.
    let boot = std::str::from_utf8(config_bytes)
        .ok()
        .and_then(|text| text.parse::<toml::Table>().ok());
    // The directory `kernels-vulkan`'s build script wrote, which reaches
    // this crate only if it is asked for. It is NOT a dependency of this
    // one: the engine linking a shader compiler to run a driver would be
    // the build-time equivalent of loading a checkpoint format, and a
    // driver consumes modules rather than producing them.
    let module_dir = boot
        .as_ref()
        .and_then(|v| Some(v.get("model")?.get("kernels")?.as_str()?.to_string()))
        .or_else(|| std::env::var("PIE_KERNELS_VULKAN_SPV_DIR").ok())
        .map(std::path::PathBuf::from)
        .ok_or_else(|| {
            anyhow!(
                "driver-vulkan: no SPIR-V module directory. Set `[model] kernels` in the \
                 boot config or PIE_KERNELS_VULKAN_SPV_DIR to the directory \
                 `kernels-vulkan` built."
            )
        })?;
    let kv_pages = boot
        .as_ref()
        .and_then(|v| v.get("model")?.get("kv_pages")?.as_integer())
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(DEFAULT_KV_PAGES);
    Ok(Boot {
        module_dir,
        kv_pages,
    })
}

/// Every `.spv` in `dir`, by file stem.
///
/// The stem is the entrypoint name, which is the key `Shell` looks a module up
/// under. A directory with none is an error and not an empty map: a shell
/// opened with no modules fails at its first fire with a message about a
/// missing kernel, which is a long way from the configuration that caused it.
fn read_modules(dir: &std::path::Path) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut modules = BTreeMap::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| anyhow!("driver-vulkan: cannot read the module directory {dir:?}: {e}"))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "spv")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            let bytes = std::fs::read(&path)
                .map_err(|e| anyhow!("driver-vulkan: cannot read {path:?}: {e}"))?;
            modules.insert(stem.to_string(), bytes);
        }
    }
    if modules.is_empty() {
        bail!(
            "driver-vulkan: no `.spv` modules in {dir:?}. A shell opened with none fails at \
             its first fire, a long way from the configuration that caused it."
        );
    }
    Ok(modules)
}

/// Every weight the decode plan binds, under the name that plan uses.
///
/// # Why the loader's own executor runs
///
/// The obvious shortcut is to read each tensor's source span out of the file
/// verbatim, and for one model it works: a `Binding::MLX_IN_PLACE` plan for
/// qwen3-0.6B is allocations, six bulk writes that tile the whole file, and
/// finalizes. qwen2.5 breaks it immediately -- its plan states 535 `TileMap`
/// transforms, which is what `fused_qkv: true` costs, and a verbatim read
/// hands the card three separate projections where the text binds one joined
/// weight. Not a fault on this backend; a wrong number.
///
/// `model_loader::executor::Execution` is a production path -- `pie model
/// convert` materialises artifacts through it -- so running the plan is both
/// less code here and the thing a real driver would do.
fn stage(
    path: &std::path::Path,
    meta: &model_loader::checkpoint::CheckpointMetadata,
    row: &'static dyn model::catalog::Variant,
    wanted: &[String],
) -> Result<Vec<(String, Vec<u8>)>> {
    // The declared encoding, out of the checkpoint's OWN metadata — the one
    // thing a catalog row genuinely cannot state, because a group size is not
    // an extent of any tensor. Read as `model/config` rather than off disk:
    // `one_normalizer::the_runtime_does_not_read_config_json` is the gate, and
    // the reason is that a second reader of that file is a second place for
    // what a model is made of to be decided. The other two drivers read it
    // exactly this way.
    let config =
        match model_loader::checkpoint::read::read_meta(meta, model::encoding::CONFIG_OBJECT) {
            Ok(Some(bytes)) => String::from_utf8(bytes).map_err(|e| {
                anyhow!(
                    "driver-vulkan: the embedded {} is not utf8: {e}",
                    model::encoding::CONFIG_OBJECT
                )
            })?,
            Ok(None) => bail!(
                "driver-vulkan: {} is not embedded in the checkpoint at {path:?}. Re-import it \
             with `pie model build`; one field is read out of it — the declared quantization \
             — and no kernel can be named without it.",
                model::encoding::CONFIG_OBJECT
            ),
            Err(e) => bail!("driver-vulkan: cannot read the embedded encoding: {e:?}"),
        };
    let encoding = model::encoding::Encoding::from_config_json(&config)
        .map_err(|e| anyhow!("driver-vulkan: unreadable encoding: {e}"))?;
    // `BackendKind::Vulkan`, not Metal's target borrowed. The two masks agree
    // today and `driver-vulkan/tests/checkpoint.rs` asserts they still do,
    // with a message saying where to re-measure if they stop.
    let target = model_loader::plan::StorageTarget::for_backend(
        model_loader::types::BackendKind::Vulkan,
        0,
        1,
    );
    let (plan, _) = model::boot::compile_load_plan_for(
        path,
        meta,
        &target,
        row,
        &encoding,
        model::boot::Binding::MLX_IN_PLACE,
    )
    .map_err(|e| {
        anyhow!(
            "driver-vulkan: `{}`'s load plan will not compile: {e}",
            row.id()
        )
    })?;
    let storage = model_loader::executor::Execution::new(&plan, path)
        .run()
        .map_err(|e| {
            anyhow!(
                "driver-vulkan: `{}`'s load plan will not run: {e}",
                row.id()
            )
        })?;

    // The conversion from a publisher's tensor names to the ones a plan
    // states. Measured in `driver-vulkan/tests/checkpoint.rs`: ZERO of 704
    // names agree before it and 704 of 704 after.
    let naming = driver_vulkan::names::Naming::mlx();
    let mut out = Vec::new();
    for traced in wanted {
        let bytes = naming
            .spellings(traced)
            .iter()
            .find_map(|s| storage.tensors.get(s.as_str()))
            .ok_or_else(|| {
                anyhow!(
                    "driver-vulkan: `{traced}` resolves to nothing `{}`'s load plan produced",
                    row.id()
                )
            })?;
        out.push((traced.clone(), bytes.clone()));
    }
    Ok(out)
}

/// Tell each member's work item what became of it.
///
/// # What a terminal cell is, and why a launched frame must write one
///
/// The scheduler hands every request in a frame a `PieTerminalCell` and
/// resolves that request's work item by READING it. Success commits, Failed
/// fails the one request, Retry re-posts it -- and `Pending`, which is what an
/// untouched cell holds, is none of those: `resolve_from_terminal` turns it
/// into `work item completion terminal outcome is still Pending`, which the
/// guest sees as `channel is poisoned: pipeline: forward failed`.
///
/// So this is not bookkeeping the asynchronous backends do for their own
/// reasons. It is the only channel a frame has for saying that it ran. CUDA
/// writes these cells from its stream callback and `remote` writes them from
/// the executor's reply; a host-side driver writes them here, because by the
/// time `Shell::launch` has returned every one of them is already decided.
///
/// # Errors
///
/// A frame that names fewer cells than the step had members, or a null one:
/// both would have this write past what the scheduler owns, and the pointer
/// is not something a later layer can check.
fn publish_terminals(
    cells: &[*mut ::driver_api::TerminalCell],
    ran: &[driver_vulkan::frames::Ran],
) -> Result<()> {
    use driver_vulkan::frames::Ran;

    if cells.is_empty() {
        // A step with no cells is one the scheduler is not waiting on -- the
        // driver's own tests build frames this way -- and writing nothing is
        // the whole of the right answer.
        return Ok(());
    }
    if cells.len() != ran.len() {
        bail!(
            "driver-vulkan: this frame names {} terminal cells for {} members",
            cells.len(),
            ran.len()
        );
    }
    for (&cell, outcome) in cells.iter().zip(ran) {
        if cell.is_null() {
            bail!("driver-vulkan: a member of this frame has a null terminal cell");
        }
        let word = match outcome {
            Ran::Fired => ::driver_api::PIE_TERMINAL_OUTCOME_SUCCESS,
            // FAILED, and it used to be RETRY. A member skipped for being
            // early has not run, so SUCCESS would answer a request whose
            // program never fired -- but RETRY is not an outcome this stack
            // still honours. The scheduler rejects it by name:
            //
            //     driver published RETRY at frame settle; retry is not a v14
            //     outcome (frame admission bounds every in-frame gate)
            //
            // and `worker`'s executor knows only SUCCESS and FAILED. So the
            // choice is between a legible failure and a rejection that blames
            // the driver in the abstract, and the reason is logged beside it.
            //
            // The v14 argument is also why nothing here tries to be cleverer:
            // if admission bounds every in-frame gate then a member cannot be
            // early, so reaching this arm is a broken invariant rather than a
            // slow producer, and a broken invariant should stop.
            Ran::Early => ::driver_api::PIE_TERMINAL_OUTCOME_FAILED,
            Ran::Faulted => ::driver_api::PIE_TERMINAL_OUTCOME_FAILED,
        };
        // `publish` releases, so the reader that observes the outcome also
        // observes everything the fire wrote before it.
        unsafe {
            (*cell).publish(word);
        }
    }
    Ok(())
}

/// Every weight name this model's decode plan binds.
///
/// The PLAN's list and not the checkpoint's: a checkpoint holds tensors no
/// fire reads, and staging those would cost the whole of them in device
/// memory for nothing. `scale.` names are dropped because they are the
/// lowering's own scalars rather than weights.
fn bound_names(text: &driver_vulkan::shell::Text) -> Vec<String> {
    use model_compiler::lower::{Arg, Fire, Row, lower};

    let Ok(low) = lower(
        &text.decode,
        &[Row::default()],
        Fire {
            captures_across_splits: false,
        },
    ) else {
        return Vec::new();
    };
    let names: std::collections::BTreeSet<String> = low
        .args
        .iter()
        .filter_map(|a| match a {
            Arg::Weight(n) if !n.starts_with("scale.") => Some(n.clone()),
            _ => None,
        })
        .collect();
    names.into_iter().collect()
}

/// The row's own two texts and the geometry they were traced for.
///
/// # Asked, not assembled
///
/// The first draft of this built a `LlamaLikeFacts` here and called
/// `llama_like_metal` on it. That is the third dispatch key
/// `driver-metal/src/model/binding.rs` exists to have deleted: a checkpoint's
/// identity is settled once by `catalog::identify` from the tensors, and
/// rebuilding a model's facts on this side would decide it a second time
/// from something else. So the row is asked, through the one door
/// `Variant::trace`, exactly as the Metal seam asks it.
///
/// # Why the METAL text
///
/// Because it is the text this driver was built against and measured on.
/// `driver-vulkan` reads `llama_like_metal`'s plans throughout -- its whole
/// device suite, its arena test and its two whole-distribution oracles -- and
/// the Metal shell is what every row goes through. The family is a naming of
/// the kernels a text names, and `driver-vulkan`'s modules answer to those
/// names; asking for a family this driver has no modules for would fail at
/// the first fire rather than here.
///
/// The gap that comes with it is stated in `driver_vulkan`'s crate doc and
/// is not this file's to close: the Metal text ignores `qkv_bias`, so a
/// Qwen2-family model is served without its attention biases.
///
/// # Errors
///
/// The ROW's refusal, carried unchanged: a model this build has no text for
/// says so in its own words rather than in a sentence this seam made up.
fn text_of(
    row: &'static dyn model::catalog::Variant,
) -> Result<(driver_vulkan::shell::Text, model::deployment::Deployment)> {
    use model::catalog::Deployed;
    use model_compiler::trace::FireClass;

    // The build's kernel capabilities, and nothing about the model. g64/b4 is
    // what `mlx-community` publishes and what every measurement in
    // `driver-vulkan` was taken against; `ANY_ENCODING` on the Metal side is
    // the same constant for the same reason.
    let binding = model::catalog::MetalBinding {
        quant_group: 64,
        quant_bits: 4,
        router_quant_group: 0,
        router_quant_bits: 0,
        moe_mxfp4: false,
        fuse_residual_gemv: true,
        paged_multi_batch: true,
        qmm_multi_batch: true,
        // The one line where this seam disagrees with `driver-metal`'s
        // constant: `binding.rs` resolves `Source::OutWidth`, which is where
        // `norm::add_bias` reads its row pitch, so this deployment can state
        // the Qwen-2 family's q/k/v projection biases. A backend that says
        // `false` here does not get an error -- it gets a text with no bias
        // in it, which is fluent and wrong, and is why the fact exists.
        add_bias: true,
    };
    let decode = row
        .trace(FireClass::Decode, Deployed::metal(&binding))
        .map_err(|e| anyhow!("driver-vulkan: `{}` has no decode text: {e}", row.id()))?;
    let prefill = row
        .trace(FireClass::Prefill, Deployed::metal(&binding))
        .map_err(|e| anyhow!("driver-vulkan: `{}` has no prefill text: {e}", row.id()))?;
    let deployment = row
        .deployment(Deployed::metal(&binding))
        .map_err(|e| anyhow!("driver-vulkan: `{}` projects no deployment: {e}", row.id()))?;
    let text = driver_vulkan::shell::Text {
        decode,
        prefill,
        geometry: driver_vulkan::dispatch::Geometry {
            q_heads: deployment.shape.q_heads,
            kv_heads: deployment.shape.kv_heads,
            // `head_dim_kernel`, not `head_dim`: phi-3's heads are 96 wide and
            // run on the 128-wide kernel, so a dispatch stating the
            // checkpoint's width addresses two thirds of what was allocated.
            // `driver-vulkan` refuses that model at `Shell::open` today; when
            // it stops doing so, this is already the right number.
            head_dim: deployment.shape.head_dim_kernel,
            rotary_dims: deployment.shape.head_dim_kernel,
            // Zero because they can only be zero: a routed mixture fired at
            // the wrong top-k routes each token to almost the right experts
            // and returns fluent nonsense, so `driver-vulkan` refuses one at
            // `Shell::open` rather than guessing.
            n_experts: 0,
            experts_per_token: 0,
        },
        layers: u16::try_from(deployment.layers).map_err(|_| {
            anyhow!(
                "driver-vulkan: `{}` has {} layers",
                row.id(),
                deployment.layers
            )
        })?,
    };
    Ok((text, deployment))
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_compiler::trace::FireClass;

    /// A member skipped for being early is FAILED, not RETRY.
    ///
    /// This mapping was RETRY, which reads as the kind answer and is the
    /// wrong one: `resolve_from_terminal` accepts the word, but the frame
    /// settle above it rejects the request that carried it --
    ///
    ///     driver published RETRY at frame settle; retry is not a v14
    ///     outcome (frame admission bounds every in-frame gate)
    ///
    /// -- and `worker`'s executor has only SUCCESS and FAILED in it at all.
    /// Nothing re-posts the member, so the RETRY bought a worse message and
    /// no second attempt.
    ///
    /// The arm is measured-unreachable in a real `pie serve`: the program
    /// that fills a channel runs in the same call that reads it. It is
    /// asserted here rather than deleted because the resolver can still
    /// return `NotReady`, and answering that with a fire would sample a
    /// distribution nobody computed.
    #[test]
    fn a_member_skipped_for_being_early_is_failed_rather_than_retried() {
        use driver_vulkan::frames::Ran;

        let cells: Vec<::driver_api::TerminalCell> = (0..3).map(|_| Default::default()).collect();
        let ptrs: Vec<*mut ::driver_api::TerminalCell> = cells
            .iter()
            .map(|c| std::ptr::from_ref(c).cast_mut())
            .collect();
        publish_terminals(&ptrs, &[Ran::Fired, Ran::Early, Ran::Faulted])
            .expect("three cells for three members");
        assert_eq!(
            cells[0].load(),
            ::driver_api::PIE_TERMINAL_OUTCOME_SUCCESS,
            "a member whose program fired is reported as having run"
        );
        assert_eq!(
            cells[1].load(),
            ::driver_api::PIE_TERMINAL_OUTCOME_FAILED,
            "an early member must be answered with a word this stack still \
             honours, and RETRY is not one"
        );
        assert_ne!(
            cells[1].load(),
            ::driver_api::PIE_TERMINAL_OUTCOME_RETRY,
            "RETRY at frame settle is rejected by the scheduler by name"
        );
        assert_eq!(
            cells[2].load(),
            ::driver_api::PIE_TERMINAL_OUTCOME_FAILED,
            "a member whose program faulted is reported as failed"
        );
    }

    /// A row's id and the fixture the driver's own numbers were taken from.
    type Measured = (
        &'static str,
        fn() -> model::shared::llama_like::forward::facts::LlamaLikeFacts,
    );

    /// The two rows `driver-vulkan`'s device suite serves for real, by the
    /// same ids that suite names them by.
    const MEASURED: &[Measured] = &[
        (
            "qwen3-0.6b",
            model::shared::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b,
        ),
        (
            "qwen2.5-1.5b",
            model::shared::llama_like::forward::facts::LlamaLikeFacts::qwen2_5_1_5b,
        ),
    ];

    /// The text this seam derives from a ROW is the text the driver was
    /// measured against.
    ///
    /// # What this pins
    ///
    /// Every number in `driver-vulkan/tests/device.rs` -- two whole logit
    /// distributions against a numpy oracle, the arena extents, the binding
    /// order -- was taken from `llama_like_metal(&LlamaLikeFacts::…,
    /// &synthetic(), class)`, a plan the test built by hand from a fixture.
    /// This seam builds its plan a different way, from the row the tensors
    /// identified. Two ways of getting a text is exactly the drift
    /// `catalog::identify` exists to prevent, and nothing else in the tree
    /// would notice them parting: a text that differed would still lower,
    /// still bind, still fire, and answer with slightly wrong logits.
    ///
    /// So the two are compared, for both classes of both measured rows. If
    /// this test ever fails, the driver's oracles no longer describe what the
    /// engine serves, and the oracles are the thing to re-take.
    #[test]
    fn the_seam_derives_the_text_the_driver_was_measured_against() {
        for (id, facts) in MEASURED {
            let row =
                model::catalog::find(id).unwrap_or_else(|| panic!("`{id}` is in the catalog"));
            let (text, _) = text_of(row).unwrap_or_else(|e| panic!("`{id}` has a text: {e}"));
            let fixture = facts();
            // `synthetic()` is documented as `driver-metal`'s answer sheet,
            // and this seam is not that driver: it states `add_bias`, which
            // `driver-vulkan`'s binder serves and `driver-metal`'s does not.
            // So the fixture side is spelled the way `driver-vulkan`'s device
            // suite spells it, which is the whole point of the comparison --
            // the oracles were taken under those facts.
            let metal = model::shared::llama_like::forward::facts::LlamaLikeMetalFacts {
                add_bias: true,
                ..model::shared::llama_like::forward::facts::LlamaLikeMetalFacts::synthetic()
            };
            for (class, ours) in [
                (FireClass::Decode, &text.decode),
                (FireClass::Prefill, &text.prefill),
            ] {
                let theirs =
                    model::shared::llama_like::forward::llama_like_metal(&fixture, &metal, class);
                assert_eq!(
                    format!("{ours:?}"),
                    format!("{theirs:?}"),
                    "`{id}`'s {class:?} text differs between the row and the fixture"
                );
            }
        }
    }

    /// The geometry the seam reads off the row is the one the fixture states.
    ///
    /// Separate from the text because it comes from a different door --
    /// `Variant::deployment` rather than `Variant::trace` -- and because it is
    /// what sizes the KV cache. A text that matched while the geometry did
    /// not would allocate a cache of the wrong width and read attention off
    /// the end of it, which this card answers with silent zeros.
    #[test]
    fn the_geometry_the_seam_reads_is_the_measured_one() {
        for (id, facts) in MEASURED {
            let row =
                model::catalog::find(id).unwrap_or_else(|| panic!("`{id}` is in the catalog"));
            let (text, _) = text_of(row).unwrap_or_else(|e| panic!("`{id}` has a text: {e}"));
            let fixture = facts();
            assert_eq!(text.geometry.q_heads, fixture.q_heads, "`{id}` q_heads");
            assert_eq!(text.geometry.kv_heads, fixture.kv_heads, "`{id}` kv_heads");
            assert_eq!(text.geometry.head_dim, fixture.head_dim, "`{id}` head_dim");
            assert_eq!(u32::from(text.layers), fixture.layers, "`{id}` layer count");
        }
    }

    /// The boot file the worker writes is the boot file this seam reads.
    ///
    /// Two crates, one file, and no shared type between them: `worker`'s
    /// `write_vulkan_startup_toml` puts `kernels` and `kv_pages` under
    /// `[model]`, and this reads them from there. Nothing in the compiler
    /// connects the two -- a key moved to `[batching]`, which is where the
    /// Metal writer keeps its page geometry and so the obvious thing to
    /// copy, would boot with this driver's own defaults and no complaint.
    /// An operator who asked for a bigger cache would get 1024 pages and no
    /// sign the number was ignored.
    ///
    /// The literal TOML here is that file's shape written out by hand on
    /// purpose. Calling the worker's writer would be the same mistake as
    /// sharing a type: the point is that the two agree, and a test that
    /// derives one from the other cannot say so.
    #[test]
    fn the_boot_file_the_worker_writes_is_the_one_this_seam_reads() {
        let boot = r#"
[model]
hf_path = "/tmp/snap"
kernels = "/tmp/spv"
kv_pages = 4096
"#;
        let read = super::boot_of(boot.as_bytes()).expect("the boot config reads");
        assert_eq!(read.module_dir, std::path::PathBuf::from("/tmp/spv"));
        assert_eq!(read.kv_pages, 4096);

        // And the default, which is the other half of the contract: the
        // worker omits `kernels` when the operator did not set one, and this
        // seam is then entitled to the environment. `kv_pages` is always
        // written, so an absent one is a file this driver did not get from
        // the worker at all -- a hand-written config, or an older one -- and
        // it gets the driver's own number rather than zero pages.
        let read = super::boot_of(
            br#"[model]
kernels = "/tmp/spv"
"#,
        )
        .expect("the boot config reads");
        assert_eq!(read.kv_pages, super::DEFAULT_KV_PAGES);
    }

    /// Every hybrid stack in the catalog is refused, and by the ROW.
    ///
    /// # The belief this checks
    ///
    /// Six comments across `driver-vulkan` say "no model this driver serves
    /// holds a recurrent state", and until this test not one of them was
    /// enforced or checked. It read as a prediction about which checkpoints
    /// would turn up, and the catalog looked like it falsified the
    /// prediction: `nemotron_h`, `kimi_k3` and `qwen_3_5` all project
    /// `recurrent: Some(..)`, and nothing here routes them away.
    ///
    /// # What was actually found, which is not what was expected
    ///
    /// A guard was written for this seam first -- refuse a row whose
    /// deployment states a recurrent shape -- and then measured, and it never
    /// fired. Every hybrid in the catalog is already refused one line
    /// earlier, by `row.trace`, because none of them has a Metal text: the
    /// qwen-3.5 family says its forward is `qwen3_5_hybrid_cuda` and
    /// "interleaves gated DeltaNet layers with attention"; nemotron-h says
    /// its forward is `nemotron_h_cuda` and that the one Metal text here
    /// "has no recurrent layer kind". Both then name the backend that does
    /// serve them.
    ///
    /// Those sentences are better than any this seam could write, and they
    /// arrive before a device is touched. So the guard was DELETED and this
    /// test kept. The belief is true; it is just not this crate that makes it
    /// true, and a guard that cannot fire is a claim that cannot be checked.
    ///
    /// # Why it scans the catalog instead of naming the families
    ///
    /// So the day a hybrid gains a Metal text, this goes red -- which is
    /// exactly the day the deleted guard would need to come back, and the
    /// only day it would have been worth anything. A test naming
    /// `nemotron-h` would pass forever while a fourth family walked in.
    ///
    /// # The controls
    ///
    /// 1. The scan must find hybrids, or it asserts nothing.
    /// 2. The two rows this backend serves must still be ACCEPTED, so this is
    ///    measuring a specific refusal and not a closed door.
    #[test]
    fn a_model_that_holds_a_recurrent_state_is_refused_before_it_is_staged() {
        use model::catalog::Deployed;
        let binding = model::catalog::MetalBinding {
            quant_group: 64,
            quant_bits: 4,
            router_quant_group: 0,
            router_quant_bits: 0,
            moe_mxfp4: false,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: true,
        };

        let mut hybrids = 0;
        for row in model::catalog::catalog() {
            let Ok(deployment) = row.deployment(Deployed::metal(&binding)) else {
                continue;
            };
            if deployment.recurrent.is_none() {
                continue;
            }
            hybrids += 1;
            let Err(said) = text_of(*row) else {
                panic!(
                    "`{}` holds a recurrent state and this driver allocates no slot for one, \
                     yet the seam took it: the guard deleted above needs to come back",
                    row.id()
                )
            };
            let said = said.to_string();
            assert!(
                said.contains(row.id()),
                "the refusal must name the row: {said}"
            );
            // The row's own sentence, carried out unchanged. If this stops
            // holding, the refusal has moved somewhere later and coarser.
            assert!(
                said.contains("no Metal text"),
                "a hybrid must be refused for having no text, before any device is touched: \
                 {said}"
            );
        }
        assert!(
            hybrids > 0,
            "no row in the catalog projects a recurrent shape, so this test asserted nothing -- \
             the qwen-3.5, nemotron-h and kimi-k3 families did when it was written"
        );

        // The control: a specific refusal, not a closed door.
        for (id, _) in MEASURED {
            let row =
                model::catalog::find(id).unwrap_or_else(|| panic!("`{id}` is in the catalog"));
            text_of(row).unwrap_or_else(|e| {
                panic!("`{id}` holds no recurrent state and must still be served: {e}")
            });
        }
    }

    /// A model this build has no text for is refused in the ROW's words.
    ///
    /// Phi-3-mini's heads are 96 wide and the Metal text names
    /// `sdpa_paged_decode_bfloat16_d_96`, a symbol no shader exports. The row
    /// says so, at length and with the alternatives it considered. This seam
    /// carries that sentence out unchanged rather than replacing it with one
    /// of its own: an operator who sees `head_dim 96` can act, and an
    /// operator who sees "driver-vulkan cannot load this model" cannot.
    ///
    /// It is also where the refusal HAPPENS that matters. `Shell::open`
    /// refuses this shape too, but that is one allocation and one module load
    /// later; the row turns it away before a device is touched.
    #[test]
    fn a_model_with_no_text_is_refused_in_the_rows_own_words() {
        let row = model::catalog::find("phi-3-mini-4k").expect("phi-3-mini-4k is in the catalog");
        assert_eq!(row.load_shape().head_dim, 96, "the checkpoint's own width");
        let Err(said) = text_of(row) else {
            panic!("phi-3-mini-4k has no Metal text")
        };
        let said = said.to_string();
        assert!(
            said.contains("phi-3-mini-4k") && said.contains("decode text"),
            "the seam says which row and which class: {said}"
        );
        assert!(
            said.contains("sdpa_paged"),
            "the row's own reason survives: {said}"
        );
    }

    /// What the seam will go looking for in a checkpoint is what the fire
    /// binds, and there is some of it.
    ///
    /// An empty answer is the failure that matters: `bound_names` swallows a
    /// lowering error into `Vec::new()`, and a `stage` that wanted nothing
    /// would load a model of zero weights and fire it, answering from
    /// whatever the pool happened to hold.
    #[test]
    fn the_seam_asks_for_the_weights_the_fire_binds() {
        for (id, _) in MEASURED {
            let row =
                model::catalog::find(id).unwrap_or_else(|| panic!("`{id}` is in the catalog"));
            let (text, _) = text_of(row).unwrap_or_else(|e| panic!("`{id}` has a text: {e}"));
            let names = bound_names(&text);
            assert!(
                names.len() > 10,
                "`{id}` binds only {} weights: {names:?}",
                names.len()
            );
            assert!(
                names.iter().any(|n| n.contains("embed")),
                "`{id}` binds no embedding: {names:?}"
            );
        }
    }

    /// The three registry verbs answer through the seam, and what they answer
    /// is what the engine's own plan shapes asked for.
    ///
    /// # What this is guarding
    ///
    /// These three used to `bail!` with "the channel plane is single-threaded
    /// by construction" -- true when `driver::ChannelState` held its cells in
    /// a `RefCell` behind an `Rc`, which made the whole `Registry` `!Send`
    /// and so unusable from a backend that lives in a `'static RwLock`. That
    /// is fixed, and this is the measurement that the refusal is gone AND
    /// that the fields cross correctly.
    ///
    /// The crossing is the part worth a test. `RegisteredChannel` takes its
    /// two wait ids and its driver id from the PLAN and only its binding from
    /// the driver, so a seam that read them off the wrong side would hand the
    /// scheduler a channel that signals nobody. And `bind_instance` maps a
    /// `requested_instance_id` of 0 to "any" -- a seam that passed the 0
    /// through would ask the registry for instance zero by name.
    ///
    /// No model is loaded and NO DEVICE IS OPEN: the plane is deviceless and
    /// modelless, and this running without either is part of what it
    /// measures. It used to go through `create`, which opens a device for its
    /// facts and then drops it -- so on a machine without a card the whole
    /// test skipped, and the deviceless claim was the one thing never checked
    /// where it is most worth checking.
    #[test]
    fn the_seam_registers_a_program_a_channel_and_an_instance() {
        let mut driver = super::VulkanDriver::without_adapter();

        let program = driver
            .register_program(&::driver_api::ProgramRegistration {
                program_hash: 0x51,
                launch: ::driver_api::plan::LaunchPackage {
                    channels: vec![::driver_api::plan::LaunchChannel {
                        id: 9,
                        capacity: 2,
                        dtype: ::driver_api::PIE_CHANNEL_DTYPE_F32,
                        flags: ::driver_api::PIE_CHANNEL_HOST_VISIBLE,
                        extern_dir: -1,
                        readiness: ::driver_api::PIE_READINESS_UNTOUCHED,
                        shape: vec![4],
                        extern_name: vec![],
                    }],
                    // `tensor_ir::registry::Stage::Epilogue`; a package with
                    // no stages is refused, and this is the emptiest legal
                    // one.
                    stages: vec![::driver_api::plan::LaunchStage {
                        kind: 3,
                        ..Default::default()
                    }],
                    plans: vec![::driver_api::plan::LaunchStagePlan::default()],
                    ..Default::default()
                },
                ..Default::default()
            })
            .expect("a one-stage program registers");

        let channel = driver
            .register_channel(&::driver_api::ChannelRegistrationPlan {
                driver_id: 4,
                channel_id: 9,
                shape: vec![4],
                dtype: ::driver_api::PIE_CHANNEL_DTYPE_F32,
                host_role: ::driver_api::PIE_CHANNEL_HOST_ROLE_WRITER,
                seeded: false,
                extern_dir: ::driver_api::PIE_CHANNEL_EXTERN_NONE,
                capacity: 2,
                reader_wait_id: 11,
                writer_wait_id: 12,
                extern_name: Vec::new(),
            })
            .expect("a channel matching the program's declaration");
        assert_eq!(
            (
                channel.driver_id,
                channel.reader_wait_id,
                channel.writer_wait_id
            ),
            (4, 11, 12),
            "the three fields the driver does not answer came from the wrong \
             side, so this channel signals nobody"
        );
        assert_eq!(channel.binding.channel_id, 9);

        let instance = driver
            .bind_instance(&::driver_api::instance::InstanceBindingPlan {
                driver_id: 4,
                program_id: program,
                // Zero is "any", not instance zero.
                requested_instance_id: 0,
                pacing_wait_id: 13,
                channel_ids: vec![9],
                seed_values: Vec::new(),
                geometry_class: ::driver_api::geometry::GeometryClass::Host,
            })
            .expect("an instance over the one channel");
        assert_eq!(instance.program_id, program);
        assert_eq!(instance.driver_id, 4);
        assert_eq!(instance.pacing_wait_id, 13);
        assert_eq!(
            instance.geometry_class,
            ::driver_api::geometry::GeometryClass::Host,
            "the class the plan asked for is not the class that came back"
        );

        driver.close_instance(instance.instance_id).expect("closes");
        driver.close_channel(9).expect("closes");
    }

    /// Every verb that needs no device is reachable, and says so in words.
    ///
    /// Ten of the fourteen. The five registry ones are the test above; these
    /// are the other five plus the four constants, and what they have in
    /// common is that a machine with no GPU can run them -- which is the
    /// machine this crate is compiled on most often, and the one where none
    /// of them was called at all until now.
    ///
    /// The claim is not that these verbs work. It is that each one is WIRED:
    /// that `encode` and `copy_state` refuse rather than silently succeed,
    /// and that the three device verbs called before `load_model` name
    /// themselves rather than panicking on an `unwrap` of `self.shell`. A
    /// silent `Ok` from `copy_kv` is the dangerous one -- the scheduler would
    /// take it as a promise that a conversation's pages had moved, and the
    /// next turn would read another request's tokens as its own.
    ///
    /// Each message is checked for the verb's own name, because "unsupported"
    /// alone in a log does not say which of fourteen calls produced it.
    #[test]
    fn every_verb_that_needs_no_device_answers_in_its_own_words() {
        let mut driver = super::VulkanDriver::without_adapter();

        assert_eq!(driver.kind(), "vulkan", "the name the engine matches on");
        assert_eq!(
            driver.device_domain(),
            ::driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE
        );
        let facts = driver.device_facts().expect("a driver states its facts");
        assert_eq!(facts.backend, driver_vulkan::facts::BACKEND);
        assert_eq!(facts.abi_version, ::driver_api::PIE_DRIVER_ABI_VERSION);
        assert_eq!(facts.page_size, driver_vulkan::facts::PAGE_SIZE);
        assert!(
            driver.export_kv_handle().is_none(),
            "this driver's pages are not exportable, and a handle here would \
             invite another process to map them"
        );

        // The two refusals. Both are unconditional, and both are a refusal
        // rather than a no-op because a no-op is indistinguishable from
        // success to the caller.
        let Err(encode) = driver.encode(&mut ::driver_api::MediaEncodePlan::default()) else {
            panic!("media encode is not implemented here, so it must not succeed")
        };
        assert!(
            encode.to_string().contains("encode"),
            "the refusal does not name the verb: {encode}"
        );
        let Err(state) = driver.copy_state(&::driver_api::StateCopyPlan::default()) else {
            panic!("no model here holds a recurrent state, so this must not succeed")
        };
        assert!(
            state.to_string().contains("recurrent state"),
            "the refusal does not say what is missing: {state}"
        );

        // The three that need a shell, called before there is one. Every
        // message has to name its own verb.
        let Err(launched) = driver.launch(&::driver_api::FrameSubmission::default()) else {
            panic!("a frame before a model has nothing to run on")
        };
        assert!(
            launched.to_string().contains("launch")
                && launched.to_string().contains("before load_model"),
            "a frame before a load did not say so: {launched}"
        );
        let Err(copied) = driver.copy_kv(&::driver_api::KvCopyPlan::default()) else {
            panic!("there is no pool to copy within, so this must not report a move")
        };
        assert!(
            copied.to_string().contains("copy_kv"),
            "a KV copy before a load did not name itself: {copied}"
        );
        let Err(resized) = driver.resize_pool(&::driver_api::PoolResizePlan::default()) else {
            panic!("there is no pool to resize")
        };
        assert!(
            resized.to_string().contains("resize_pool"),
            "a resize before a load did not name itself: {resized}"
        );
    }

    /// The seam's own staging is the driver's, measured by the answer.
    ///
    /// `driver-vulkan`'s device suite already serves these weights and gets
    /// the pattern right, but it stages them ITSELF: it reads a raw MLX
    /// snapshot and holds the bytes. This path is a different one end to end
    /// -- a `.zt` artifact `pie model build` authored, `catalog::identify`
    /// on already-lowered names, `compile_load_plan_for`, the streaming
    /// executor, and `Naming::mlx()` -- and every one of those steps could
    /// deliver a tensor that is transposed, half a row short, or another
    /// layer's. None of that is a crash on this card: `robustBufferAccess`
    /// returns zero past the end, so wrong staging is FLUENT.
    ///
    /// So the claim is the numbers. The prompt is five repeats of a
    /// six-token period plus two more, and four greedy steps must continue
    /// it exactly. The numpy CPU reference over the same checkpoint answers
    /// the same four, and a model that had lost a weight would still answer
    /// something.
    ///
    /// Skips without `PIE_VULKAN_ARTIFACT` and `PIE_KERNELS_VULKAN_SPV_DIR`.
    #[test]
    fn the_artifact_this_seam_stages_answers_what_the_checkpoint_answers() {
        let (Ok(modules), Ok(artifacts)) = (
            std::env::var("PIE_KERNELS_VULKAN_SPV_DIR"),
            std::env::var("PIE_VULKAN_ARTIFACT"),
        ) else {
            eprintln!("SKIP: PIE_KERNELS_VULKAN_SPV_DIR and PIE_VULKAN_ARTIFACT name the inputs");
            return;
        };
        let boot = format!("[model]\nkernels = \"{modules}\"\nkv_pages = 64\n");
        // Every artifact named, because one model cannot tell a conversion
        // from a table that happens to spell one model's names -- the same
        // reason `driver-vulkan/tests/checkpoint.rs` takes a list.
        let mut served = 0usize;
        for artifact in artifacts.split(':').filter(|a| !a.is_empty()) {
            let Ok(mut backend) = VulkanDriver::create(boot.as_bytes()) else {
                eprintln!("SKIP: no Vulkan device");
                return;
            };
            backend
                .load_model(vec![::driver_api::ModelLoadDesc {
                    snapshot_dir: std::path::PathBuf::from(artifact),
                    runtime_quant: String::new(),
                    mxfp4_moe: ::driver_api::Mxfp4MoeRequest::Auto,
                    component: ::driver_api::ModelComponent::Text,
                }])
                .unwrap_or_else(|e| panic!("{artifact} loads: {e}"));
            let shell = backend.shell.as_mut().expect("a shell after load_model");

            // The period `driver-vulkan/tests/device.rs` uses, and the same
            // reason: a repeating prompt makes a wrong answer visible without a
            // tokenizer, because the right one is the next term.
            const PERIOD: [u32; 6] = [15339, 1723, 88204, 6100, 41777, 2930];
            let mut prompt: Vec<u32> = Vec::new();
            for _ in 0..5 {
                prompt.extend_from_slice(&PERIOD);
            }
            prompt.push(PERIOD[0]);
            prompt.push(PERIOD[1]);
            // Thirty-two: the tiled GEMM takes whole 16-row tiles and a driver
            // may not pad a fire it did not author.
            assert_eq!(prompt.len() % 16, 0);

            let mut got = Vec::new();
            let mut turn = driver_vulkan::turns::Turn {
                who: 1,
                tokens: prompt,
            };
            for _ in 0..4 {
                let rows = turn.tokens.len();
                let step = shell
                    .step(std::slice::from_ref(&turn))
                    .unwrap_or_else(|e| panic!("the step ran: {e}"));
                assert_eq!(step.rows, rows, "the fire answered a different width");
                let vocab = step.logits.vocab;
                // The last row is the one that has seen the whole prompt.
                let at = (rows - 1) * vocab;
                let row = &step.logits.values[at..at + vocab];
                let next = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .expect("a non-empty distribution")
                    .0 as u32;
                got.push(next);
                turn.tokens = vec![next];
            }
            assert_eq!(
                got,
                vec![PERIOD[2], PERIOD[3], PERIOD[4], PERIOD[5]],
                "the artifact this seam staged from {artifact} does not continue \
             the pattern the same weights continue on the CPU"
            );
            served += 1;
        }
        assert!(served > 0, "PIE_VULKAN_ARTIFACT named nothing");
    }
}
