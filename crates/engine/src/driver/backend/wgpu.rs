//! The seam to `driver-wgpu`.
//!
//! # A library call, like the two seams beside it
//!
//! No C ABI and no `*mut PieDriver`: the driver is Rust, and a `#[repr(C)]`
//! boundary between two Rust crates is a second spelling of a contract they
//! already share. This file is the door and only the door.
//!
//! # What this backend has that neither sibling does
//!
//! `driver-cuda` needs a CUDA toolkit and an NVIDIA card. `driver-metal` needs
//! a Mac. `driver-vulkan` needs a loader and an ICD. **This one needs none of
//! them.** Its whole dependency closure is pure Rust -- `driver-wgpu`'s
//! `tests/pure.rs` asserts it with `native` ON -- it builds for
//! `wasm32-unknown-unknown` including the browser backend (`tests/browser.rs`),
//! and it reaches Vulkan, Metal, D3D12 and WebGPU through one binary. See the
//! `driver-wgpu` feature in `crates/engine/Cargo.toml` for what follows from
//! that and what deliberately does not.
//!
//! # No module directory, and that is the headline difference from Vulkan
//!
//! [`VulkanDriver::create`](super::VulkanDriver::create) fails at boot without
//! `PIE_KERNELS_VULKAN_SPV_DIR` or `[model] kernels`, because SPIR-V is a build
//! product that has to be found on disk. `kernels-wgpu` embeds every shader
//! source in its rlib and `naga` compiles it in this process, so there is no
//! directory to ship beside the binary, no `OUT_DIR` to relay, and no way for a
//! deployment to be given this driver and not its kernels. Everything the
//! Vulkan seam does with `modules`, `module_dir` and `read_modules` is absent
//! here rather than reimplemented.
//!
//! # Where loading lives, and why it is not next door
//!
//! `driver-wgpu` keeps `model` and `model-loader` as DEV-dependencies and
//! proves the closure with `tests/pure.rs`, because a driver that depended on a
//! checkpoint FORMAT would be a driver that could not be handed bytes. What it
//! exposes instead is `Shell::hold(name, bytes)`. So the conversion from a
//! publisher's tensor names to the ones a plan states runs HERE, on the side
//! that already depends on `model` and `model-compiler` for its own reasons --
//! exactly as it does for Vulkan.
//!
//! That also means [`WgpuDriver::create`] cannot open a shell: a
//! `driver_wgpu::shell::Shell` is a device plus a model's two plans plus a
//! cache shaped for that model, and none of the last three exists until a
//! checkpoint has been read. `create` opens the ADAPTER, states its facts and
//! refuses a machine that cannot bind the widest kernel; `load_model` builds
//! the shell ON that same device. The Vulkan seam drops its device instead and
//! opens another at load -- see the `device` field for why this one does not.
//!
//! # What is servable, and what refuses by name
//!
//! `create`, `device_facts`, the registry five, `launch`, `copy_kv` and
//! `resize_pool` are served. `encode` and `copy_state` refuse, as they do on
//! all three siblings. `export_kv_handle` answers `None`, and more firmly than
//! Vulkan does: WebGPU has no external-memory extension at all, where Vulkan
//! has one this driver declines.
//!
//! `launch` fires over the PHYSICAL pages the engine's scheduler chose, through
//! `driver_wgpu::frames` and `turns::Serving::over` -- the same two pieces the
//! Vulkan seam answers this verb with, and for the same reason. The driver's
//! own `pages::Book` is NOT consulted on that path: the scheduler already owns
//! eviction and prefix sharing, and two allocators over one pool does not fault
//! -- it answers one conversation out of another's keys, fluently.
//!
//! A plan naming something this driver does not implement -- a user mask,
//! `max_layers`, images, recurrent state -- is refused by that field's name at
//! admission rather than served without it, which is `frames::unserved_in`.

use anyhow::{Result, anyhow, bail};

use ::driver_api::{
    BoundInstance, ChannelRegistrationPlan, CompletionBroker, Driver, FrameLaunchOutcome,
    FrameSubmission, InstanceBindingPlan, KvCopyPlan, MediaEncodePlan, PoolResizePlan,
    ProgramRegistration, RegisteredChannel, StateCopyPlan, SubmissionCompletion,
};

use super::settle_control;

/// How many KV pages a shell is opened with.
///
/// A number the boot config can override (`[model] kv_pages`). `driver-wgpu`'s
/// own `Deployment::default` says 64, which is a desktop demo's number and not
/// a server's; the pool is resizable -- `resize_pool` is served -- but it has
/// to start somewhere, and a scheduler that has not asked for anything yet has
/// no better number to offer. 1024 is what the Vulkan seam starts at, and the
/// two pools are the same shape.
const DEFAULT_KV_PAGES: u32 = 1024;

/// The WebGPU shell, behind the seam's verbs.
///
/// The shell is `Option` because it does not exist until a checkpoint has been
/// read -- see the module doc. `facts` is kept beside it because the engine
/// asks for them at `create`, before there is a shell to ask.
pub struct WgpuDriver {
    shell: Option<driver_wgpu::shell::Shell>,
    facts: ::driver_api::DeviceFacts,
    /// The adapter `create` opened, HELD until `load_model` hands it to the
    /// shell.
    ///
    /// The Vulkan seam opens a device for its facts and drops it, with the
    /// argument that holding one keeps the whole GPU against a model that
    /// might never arrive. That argument does not survive here, and two things
    /// replace it. A `wgpu::Device` is an adapter and a queue and no
    /// allocations, so what is held is a handle rather than a card. And the
    /// facts this driver states are MEASURED off the device --
    /// `min_storage_buffer_offset_alignment` and whether the memory is unified
    /// -- so opening a second device at load and serving on that one would
    /// mean the engine plans against an adapter it is not running on. On a
    /// machine with one GPU those are the same adapter; on a laptop with an
    /// integrated and a discrete one they need not be.
    ///
    /// `driver_wgpu::shell::Shell::on` exists for exactly this caller: the
    /// crate's own doc says opening an instance and an adapter twice in one
    /// process is "legal and slow".
    device: Option<driver_wgpu::device::Device>,
    /// That adapter's name, kept for the message a failed load gives: "the
    /// shell would not open" is unhelpful without saying on what.
    adapter: String,
    kv_pages: u32,
    broker: CompletionBroker,
    /// The PTIR channel plane: programs, channels and their instances.
    ///
    /// Beside the shell rather than the one INSIDE it, and alive from `create`
    /// rather than from `load_model`. `driver_wgpu::shell::Shell` owns a
    /// `Programs` of its own and serves the five verbs off it, which is right
    /// for a server built on that crate alone and wrong here: a registry that
    /// only existed once a checkpoint had been read would refuse a program the
    /// engine is entitled to register at any time, for a reason that has
    /// nothing to do with the program. The plane is portable, deviceless host
    /// memory, so a second one costs nothing but the field. The shell's own
    /// copy stays empty and unused.
    programs: driver_wgpu::programs::Programs,
}

impl WgpuDriver {
    /// Open an adapter, state its facts, check it can bind this build's widest
    /// kernel, and keep it for the shell.
    ///
    /// # Errors
    ///
    /// No adapter, or an adapter whose storage-buffer limit cannot reach the
    /// attention kernels. Both are boot conditions rather than runtime ones,
    /// and the second is the reason this does more work than the Vulkan seam's
    /// `create`: WebGPU's guaranteed floor is 8 storage buffers per compute
    /// stage and `sdpa_paged_decode` binds eleven, so an adapter at the floor
    /// builds most of the tree and cannot build attention.
    /// `driver_wgpu::shell::Shell::on` refuses that too, one model load later;
    /// asking here means a deployment finds out from its configuration rather
    /// than from its first request.
    pub fn create(config_bytes: &[u8]) -> Result<Self> {
        let kv_pages = boot_kv_pages(config_bytes);

        let device = driver_wgpu::device::Device::open()
            .map_err(|e| anyhow!("driver-wgpu: no adapter: {e}"))?;
        let unreachable = device.unreachable();
        if !unreachable.is_empty() {
            bail!(
                "driver-wgpu: `{}` allows a compute stage {} storage buffers, and {} kernels \
                 need more -- starting with {:?}. This adapter can build most of the tree and \
                 not attention.",
                device.name(),
                device.limits().storage_buffers,
                unreachable.len(),
                unreachable.iter().take(4).collect::<Vec<_>>()
            );
        }
        // MEASURED off this device, not stated: the same build runs over
        // Vulkan, Metal, D3D12 and a browser, and the storage alignment is
        // whatever that implementation reports. It is the number the engine
        // lays its arena out to satisfy, so it has to be the number of the
        // adapter that will bind it -- which is why the device below is kept
        // rather than dropped.
        let facts = driver_wgpu::facts::of(
            u32::try_from(device.min_storage_offset()).unwrap_or(u32::MAX),
            device.unified(),
        );
        let adapter = device.name().to_owned();
        Ok(Self {
            shell: None,
            facts,
            device: Some(device),
            adapter,
            kv_pages,
            broker: CompletionBroker::new(),
            programs: driver_wgpu::programs::Programs::new(),
        })
    }

    /// A control completion for work that has already happened.
    ///
    /// # Why a control op needs TWO writes and a launch needs one
    ///
    /// `CompletionBroker::control_completion` mints a completion that carries
    /// a TERMINAL CELL, and the engine resolves the op by reading that cell --
    /// `launch_completion` deliberately carries none, because a frame answers
    /// per member instead. The asynchronous backends hand the whole
    /// `CompletionTarget` to whatever finishes the work: CUDA gives it to the
    /// shell, `remote` to its RPC task, and each publishes the cell and then
    /// notifies. A host-side driver has nobody to hand it to -- `Shell::copy_kv`
    /// has already copied by the time it returns -- so it does both here.
    ///
    /// **Both, and in this order.** Dropping the target and notifying nothing
    /// is a promise nobody keeps: a fork hung a real `pie run` for 850 seconds
    /// with the watchdog naming it exactly --
    ///
    /// ```text
    /// in_flight_control: KV copy pipeline Some(..) settled=false
    /// ```
    ///
    /// -- and notifying WITHOUT publishing is caught by the engine on the way
    /// out, `driver callback published before terminal outcome settled`, which
    /// is the assertion that made the ordering legible in the first place.
    ///
    /// Both halves live in [`settle_control`], beside the test that asserts
    /// the STATE they leave rather than that they were called.
    ///
    /// This seam's doc used to end "`driver-metal` and `driver-vulkan` both
    /// drop the target and notify nothing here, with a doc comment that says
    /// 'settled on return'. They are not." That was true when it was written
    /// and is not now: both siblings call the shared helper, and
    /// `backend/vulkan.rs` records the same hang in seconds -- 11.8 with it,
    /// against the 850 above. A sentence about what another crate gets wrong
    /// is one that goes quietly false the day it is fixed, which is why this
    /// one is kept as history rather than restated as fact.
    fn settled_control(&mut self) -> SubmissionCompletion {
        settle_control(&self.broker)
    }

    /// The shell, or a message saying which verb was called before a load.
    fn shell(&mut self, what: &'static str) -> Result<&mut driver_wgpu::shell::Shell> {
        self.shell.as_mut().ok_or_else(|| {
            anyhow!(
                "driver-wgpu: {what} before load_model. A shell is a device plus a model's \
                 plans plus a cache shaped for that model, and none of the last three exists \
                 until a checkpoint has been read."
            )
        })
    }

    /// A driver with no adapter behind it, for the verbs that do not need one.
    ///
    /// This is the state a real driver is in between `create` and
    /// `load_model` -- no shell, an empty registry -- with the one difference
    /// that its facts were never measured, so they are the specification's
    /// guaranteed floor rather than a device's answer.
    ///
    /// It exists because CI has no GPU and the thing worth checking there is
    /// not arithmetic: it is that every verb of the seam is REACHABLE. A match
    /// arm that compiles proves the arm exists; only calling it proves the arm
    /// does something other than panic. Ten of the fourteen verbs never touch
    /// an adapter -- the registry five, the two refusals, and the three
    /// "before load_model" errors -- so ten of them can be walked here.
    #[cfg(test)]
    pub(super) fn without_adapter() -> Self {
        Self {
            shell: None,
            facts: driver_wgpu::facts::of(driver_wgpu::facts::GUARANTEED_STORAGE_ALIGNMENT, false),
            device: None,
            adapter: "<none: this driver was built without opening one>".to_string(),
            kv_pages: DEFAULT_KV_PAGES,
            broker: CompletionBroker::new(),
            programs: driver_wgpu::programs::Programs::new(),
        }
    }
}

impl Driver for WgpuDriver {
    fn kind(&self) -> &'static str {
        "wgpu"
    }

    fn device_domain(&self) -> ::driver_api::DeviceDomain {
        ::driver_api::PIE_MEMORY_DOMAIN_WEBGPU_DEVICE
    }

    /// The device's stated facts.
    fn device_facts(&self) -> Option<&::driver_api::DeviceFacts> {
        Some(&self.facts)
    }

    /// WebGPU exports no KV handle. Vulkan answers `None` because it declines
    /// the external-memory extension; here there is no such extension to
    /// decline, in the API or in the specification.
    fn export_kv_handle(&self) -> Option<::driver_api::KvHandle> {
        None
    }

    /// Identify the checkpoint, assemble its text, open a shell, and stage
    /// every weight the decode plan binds.
    ///
    /// # Errors
    ///
    /// More than one descriptor, a snapshot no catalog row matches, a rope
    /// ladder this driver cannot build, a plan that will not compile or
    /// execute, or an adapter that will not allocate.
    fn load_model(
        &mut self,
        descs: Vec<::driver_api::ModelLoadDesc>,
    ) -> Result<::driver_api::DriverCapabilities> {
        let [desc] = descs.as_slice() else {
            bail!(
                "driver-wgpu: {} model descriptors. This backend serves one model per \
                 device -- a shell holds one text and one cache shaped for it.",
                descs.len()
            );
        };
        let path = desc.snapshot_dir.as_path();
        let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(path)
            .map_err(|e| anyhow!("driver-wgpu: unreadable checkpoint at {path:?}: {e}"))?;
        // THE TENSORS decide which row this is, not a name in a config file.
        // `Override::None`: a boot config that named a model would be a second
        // identification, and the two disagree in the direction that matters --
        // a checkpoint served as the wrong row is not refused, it is fluent and
        // wrong.
        let row = model::catalog::identify(&meta, &model::catalog::Override::None)
            .map_err(|e| anyhow!("driver-wgpu: {path:?} matches no catalog row: {e}"))?;
        let (text, deployment) = text_of(row)?;
        // Read from the text BEFORE the shell takes it: what the decode plan
        // binds is what has to be staged, and the shell owns the plan after
        // `open`.
        let wanted = bound_names(&text);
        // The widest fire this deployment can actually be given, measured the
        // same way and for the same reason: the text is the shell's after
        // `open`, and this is a question about the text.
        let widest = widest_fire(&text, 4096);
        // The device `create` opened and measured its facts from. A SECOND
        // call -- a re-load, not the several-descriptors case refused above --
        // opens a new adapter, because the first went into the shell this one
        // would replace. The facts the engine already believes are then a
        // different adapter's, which is a reason not to re-load rather than a
        // reason to keep a spare.
        let device = match self.device.take() {
            Some(device) => device,
            None => driver_wgpu::device::Device::open()
                .map_err(|e| anyhow!("driver-wgpu: no adapter: {e}"))?,
        };
        let mut shell = driver_wgpu::shell::Shell::on(
            device,
            text,
            driver_wgpu::shell::Deployment {
                pages: self.kv_pages,
                // The ROW's ladder, not the default. `Deployment::default`
                // states 1e6, which is right for qwen3 and wrong for
                // gpt-oss's 150_000 and llama-3's 500_000 -- and a rope base
                // that is wrong by a factor of six does not fault. It attends
                // at the wrong wavelengths and stays fluent, which is why this
                // is read off the deployment the row projected rather than
                // left to a constant.
                theta: theta_of(row, &deployment)?,
                rescale: rescale_of(row, &deployment)?,
                ..driver_wgpu::shell::Deployment::default()
            },
        )
        .map_err(|e| {
            anyhow!(
                "driver-wgpu: `{}` would not serve this model: {e}",
                self.adapter
            )
        })?;

        for (name, bytes) in stage(path, &meta, row, &wanted)? {
            shell
                .hold(&name, &bytes)
                .map_err(|e| anyhow!("driver-wgpu: `{name}` would not stage: {e}"))?;
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
            // pages and single rows inside the one KV buffer -- which is what a
            // prefix-cache hit is -- and refuses any plan whose ends are not
            // both this driver's own domain. Host directions stay off: there is
            // no swap pool, so a device-to-host copy has nowhere to land.
            kv_copy_domain_mask: ::driver_api::KV_COPY_DEVICE_TO_DEVICE,
            rs_cache_required: false,
            rs_cache_slots: 0,
            rs_cache_slot_bytes: 0,
            // Not elastic. `resize_pool` here reallocates the KV buffer whole,
            // so nothing can be given back page-wise, and both numbers are zero
            // together -- which is the condition `bootstrap` reads before it
            // starts a trim task at all. WebGPU has no sparse binding, so
            // unlike Vulkan this is not a declined optional feature: there is
            // nothing to decline.
            elastic_page_bytes: 0,
            elastic_budget_pages: 0,
            has_mtp_logits: false,
            has_mtp_drafts: false,
            has_value_head: false,
            // Sinks this backend cannot honour. Every one of them would bind
            // and then run as a silent no-op, which is worse than a refusal at
            // the door.
            has_kv_envelopes: false,
            has_attn_score: false,
            has_attn_page_mask: false,
            has_lora: false,
            model_site_summary: ::driver_api::ModelSiteSummary::default(),
            // The decode envelope's ports, claimed because `envelope::fill`
            // resolves them: `Programs::geometry` peeks each instance's
            // channels and `launch` drives the frame a step at a time so a
            // step's tokens can be what the step before it wrote. This field
            // read 0 for as long as that machinery was missing, and the 0 was
            // honest then -- the engine answered it by folding the geometry on
            // the host, which cannot know `EmbedTokens` and said so:
            //
            //     decode envelope on a driver without device geometry ports
            //     (mask 0x0, needs 0x25): falling back to host-evaluated
            //     serialized execution
            //     ... EmbedTokens is not host-derivable: channel 0 has no
            //     host-known value
            //
            // so the fix was to build the machinery, not to widen the claim.
            // Both device-resolved classes, because `envelope::fill` answers
            // both from one resolution: `DecodeEnvelope`'s three ports plus
            // the four `DeviceGeometry` adds -- the pages, the CSR and the
            // write descriptor -- which are READ off `driver::Geometry`
            // rather than derived from the positions.
            //
            // Claiming only the first three cost seven of the fifteen curated
            // failures on this backend: a program that traced its whole
            // geometry fell back to host evaluation, which cannot know
            // `EmbedTokens` and said so.
            // ATTN_MASK is in the claim because `envelope::fill` resolves it
            // too: a device-geometry program's dense per-lane mask is read off
            // `driver::Geometry::mask`, re-encoded as the runs every other
            // path in the driver reads, and staged as the rectangle
            // `Frame::of` packs. Without this bit the engine refuses such a
            // program at classification -- "a channel-bound dense AttnMask
            // belongs to the pool-owned device-geometry class" -- and sends it
            // down a host fallback that cannot derive `EmbedTokens`.
            device_geometry_port_mask: ::driver_api::PIE_DECODE_ENVELOPE_PORTS
                | ::driver_api::PIE_DEVICE_GEOMETRY_PORTS
                | ::driver_api::PIE_DEVICE_PORT_ATTN_MASK,
            // `launch` interleaves conversion and dispatch: it admits the
            // frame, then for each step in order calls `envelope::fill`,
            // `prepare`, `serve` and `run_programs` before touching the next.
            // So a slot chained behind an earlier slot of the same frame
            // reads a cell that slot's PROGRAM has already put -- and a cell
            // that is genuinely not there yet comes back `Filled::Early`,
            // which is a re-post rather than a fault.
            //
            // This said `false` with a comment claiming the opposite of what
            // the loop below does. The flag is a FACT about a driver, not a
            // preference: `validate_frame` reads it to decide whether a
            // device-geometry pass counts as host-resolved, and answering
            // `false` here made the engine refuse frames this backend can
            // execute.
            resolves_geometry_per_step: true,
            // The ceilings a batch is formed under. `Shell::open` sizes one
            // fire's scratch from `Deployment::seam`, and a fire wider than
            // this has nothing to run in.
            max_forward_tokens: widest,
            max_forward_requests: 256,
            max_page_refs: shape.pages,
            // The row's answers, not a config's: the checkpoint was identified
            // once and these come from that identification.
            arch_name: deployment.advertised.arch.to_string(),
            model_id: row.id().to_string(),
            vocab_size: deployment.shape.vocab,
            max_model_len: deployment.advertised.max_model_len,
            activation_dtype: "bf16".to_string(),
            hidden_size: deployment.shape.hidden,
            // False about the BACKEND rather than about the row: there is no
            // encode entry point here at all, so a model with a vision tower is
            // served as its text half.
            supports_media_encode: false,
            snapshot_dir: path.display().to_string(),
            kv_handle: None,
            // The shaders are in the rlib and `naga` compiles them; nothing
            // upstream generates a kernel for this driver.
            codegen_backend: String::new(),
        })
    }

    /// Register a PTIR program: its launch package and its emitted kernels.
    ///
    /// Served from `create`, before any model: see the `programs` field for why
    /// the plane does not wait on a checkpoint.
    ///
    /// # Errors
    ///
    /// A launch package the registry refuses -- no stages, a channel shape it
    /// cannot serve, a stage it cannot read.
    fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64> {
        self.programs
            .register_program(desc)
            .map_err(|e| anyhow!("driver-wgpu: {e}"))
    }

    /// Register a channel and hand back where its ring lives.
    ///
    /// The ring is HOST memory. Nothing about the channel plane is on the GPU,
    /// and the argument is stronger here than on either sibling: a WebGPU
    /// buffer is read back by mapping it and awaiting the queue, so a device
    /// ring would cost a submission per poll for data no shader reads.
    ///
    /// # Errors
    ///
    /// A shape the registry will not serve, or a duplicate id.
    fn register_channel(&mut self, desc: &ChannelRegistrationPlan) -> Result<RegisteredChannel> {
        let binding = self
            .programs
            .register_channel(desc)
            .map_err(|e| anyhow!("driver-wgpu: {e}"))?;
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
    /// closed on the way out: `plan.validate_binding` is what catches a driver
    /// that answered a different instance id than the one the engine requested,
    /// and an instance left open behind that error would be a leak nothing
    /// later has a handle to close.
    ///
    /// # Errors
    ///
    /// An unknown program or channel, a geometry class this build has no name
    /// for, or a binding that does not match what was asked.
    fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        // `requested_instance_id` is 0 for "any", which the registry spells as
        // `None`. Passing the 0 through would ask for instance zero by name.
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
                // spelling of the same three values -- `driver-api`
                // static-asserts the pairing.
                desc.geometry_class as u32,
                &desc.channel_ids,
                &seeds,
            )
            .map_err(|e| anyhow!("driver-wgpu: {e}"))?;
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
    /// A frame whose step tables do not describe its rows, a plan naming a
    /// field this driver does not implement, or a device failure. Admission is
    /// NOT an error: a frame that does not fit reports
    /// [`FrameLaunchOutcome::Exhausted`], which the engine re-posts, or
    /// `Impossible` when no growth could ever make room.
    fn launch(&mut self, frame: &FrameSubmission) -> Result<FrameLaunchOutcome> {
        let page = driver_wgpu::facts::PAGE_SIZE;
        match self.shell("launch")?.admit(frame) {
            Ok(Some(driver_wgpu::frames::Launched::Exhausted)) => {
                return Ok(FrameLaunchOutcome::Exhausted);
            }
            Ok(Some(driver_wgpu::frames::Launched::Impossible)) => {
                return Ok(FrameLaunchOutcome::Impossible);
            }
            Ok(Some(driver_wgpu::frames::Launched::Ran(_)) | None) => {}
            Err(e) => return Err(anyhow!("driver-wgpu: {e}")),
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
            let filled = driver_wgpu::envelope::fill(&self.programs, frame, sub, page)
                .map_err(|e| anyhow!("driver-wgpu: {e}"))?;
            let (plan, writes) = match filled {
                driver_wgpu::envelope::Filled::Ready { plan, writes } => (plan, writes),
                // Nothing to fire and nothing wrong: the producer has not
                // run. Every member of this step is told to come back, which
                // is what the scheduler's re-post is for.
                driver_wgpu::envelope::Filled::Early { channel } => {
                    tracing::debug!(channel, "wgpu: a step's geometry channel is not filled yet");
                    let early = vec![driver_wgpu::frames::Ran::Early; sub.roster_rows.len()];
                    publish_terminals(&sub.terminal_cells, &early)?;
                    break;
                }
            };
            let (requests, tokens) = self
                .shell("launch")?
                .prepare(&plan, &writes)
                .map_err(|e| anyhow!("driver-wgpu: {e}"))?;
            let step = self
                .shell("launch")?
                .serve(&requests, &tokens)
                .map_err(|e| anyhow!("driver-wgpu: {e}"))?;
            // The distributions do not come back through this return, and
            // that is the seam's shape rather than a loss: a step's answer is
            // read by the frame's own PROGRAMS, which put it on the channels
            // the engine reads. Firing them is this seam's job because the
            // registry is this seam's -- it is alive from `create`, before
            // there is a shell, and the shell's own copy stays empty -- so the
            // driver hands back the step and the two halves are joined here.
            let ran = driver_wgpu::frames::run_programs(
                &mut self.programs,
                &frame.instance_ids,
                sub,
                &step,
                &mut faults,
            )
            .map_err(|e| anyhow!("driver-wgpu: {e}"))?;
            publish_terminals(&sub.terminal_cells, &ran)?;
        }
        // Logged and not returned, as Metal and Vulkan log them: a fault kills
        // the one instance that faulted, and the requests batched with it ran.
        // The guest behind the dead one is not left waiting --
        // `driver::Registry::fire` publishes the fault on the rings that
        // instance's host READS, and the pipeline turns that poison word into
        // the guest's error -- so this line is an operator's record rather
        // than the only report.
        for (instance, why) in faults {
            tracing::warn!(instance, %why, "wgpu: program faulted");
        }
        // Settled here, because it is already settled. A completion is a
        // promise that the frame's work has finished, and the asynchronous
        // backends keep it by notifying from wherever the work actually lands
        // -- `remote` from its RPC task, CUDA from its stream. This driver has
        // nowhere to notify FROM: `Shell::serve` waits on the queue itself and
        // everything the frame asked for has happened by the time it returns.
        //
        // Handing back an unnotified completion is not a smaller version of
        // that. It is a promise nobody keeps: the scheduler parks the lane on
        // it and re-reports the same frame forever --
        //
        //     [pie-sched] driver 0 stalled for 370s (no progress, work queued
        //     or in flight) ... batch of 1 (posted(token=2), age=370s)
        //
        // -- which is what a real `pie run` on this backend did, after a fire
        // that had already computed its answer.
        let (_raw, completion) = self.broker.launch_completion(1);
        self.broker.notify(completion.wait_id(), 1);
        Ok(FrameLaunchOutcome::Launched(completion))
    }

    /// # Errors
    ///
    /// Always. There is no separate encode step in this driver: a fire records
    /// and submits in one call, so there is no encoded frame to hand back. CUDA,
    /// Metal and Vulkan refuse the same verb.
    fn encode(&mut self, _plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        bail!("driver-wgpu: media encode is unsupported on this backend")
    }

    /// Move KV pages, and the rows inside them, within this pool.
    ///
    /// Settled on return, and by a different route from Vulkan's. There the
    /// buffers are host-visible and coherent so a move is a `memmove` with no
    /// command buffer at all; here it is a real device copy -- one encoder for
    /// the whole plan, through `Device::shuffle`, which ends in a
    /// `PollType::Wait`. Either way nothing is in flight when this returns,
    /// which is what lets the completion be posted immediately.
    ///
    /// # Errors
    ///
    /// A call before `load_model`, a plan whose two ends are not one domain, or
    /// a page the pool does not have.
    fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<SubmissionCompletion> {
        self.shell("copy_kv")?
            .copy_kv(desc)
            .map_err(|e| anyhow!("driver-wgpu: {e}"))?;
        Ok(self.settled_control())
    }

    /// # Errors
    ///
    /// Always: no model this backend serves holds a recurrent state.
    fn copy_state(&mut self, _desc: &StateCopyPlan) -> Result<SubmissionCompletion> {
        bail!("driver-wgpu: no model this driver serves holds a recurrent state")
    }

    /// Rebuild the KV pool at `target_pages`.
    ///
    /// # Errors
    ///
    /// A call before `load_model`, a shrink that would strand a conversation,
    /// or an adapter that will not allocate the new size. A resize of a pool
    /// this driver does not have -- the trim task asks about three on every
    /// tick -- is answered rather than refused, inside the shell.
    fn resize_pool(&mut self, desc: &PoolResizePlan) -> Result<SubmissionCompletion> {
        self.shell("resize_pool")?
            .resize_pool(desc)
            .map_err(|e| anyhow!("driver-wgpu: {e}"))?;
        // Settled already: `Pool::resize` allocates the new buffers, copies the
        // kept prefix device-to-device through `Device::transfer` -- which
        // waits -- and drops the old ones.
        Ok(self.settled_control())
    }

    /// # Errors
    ///
    /// Never; the registry accepts a close of an id it does not hold, because a
    /// close is idempotent from the scheduler's side.
    fn close_instance(&mut self, id: u64) -> Result<()> {
        // A close BEFORE a load is answered rather than refused, and so is a
        // close of an id the registry does not hold: teardown races both ways,
        // and a fault logged per conversation would be noise about a verb that
        // was right to be called.
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

/// How many KV pages the boot TOML asks for.
///
/// Its own function, deviceless, for the reason the Vulkan seam's `boot_of`
/// gives beside it: this is a CONTRACT with the worker, which writes the
/// document (`embedded_driver::wgpu_startup_toml`), and a contract with one
/// reader and no test is one that gets discovered broken by a server booting
/// at a pool size nobody asked for.
///
/// One number where Vulkan reads two, because there is no module directory to
/// find -- the shaders are in the rlib. The BYTES are the document and not a
/// path to it; `create_driver_backend` hands over the text for that reason,
/// and `worker`'s `a_parsing_seam_is_handed_the_text_that_was_written` is the
/// other end of the same claim.
///
/// A document that does not parse, or one that states no `[model] kv_pages`,
/// gets [`DEFAULT_KV_PAGES`] rather than an error: a hand-written config is
/// entitled to leave the number out, and zero pages is not a cache.
fn boot_kv_pages(config_bytes: &[u8]) -> u32 {
    std::str::from_utf8(config_bytes)
        .ok()
        .and_then(|text| text.parse::<toml::Table>().ok())
        .and_then(|v| v.get("model")?.get("kv_pages")?.as_integer())
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_KV_PAGES)
}

/// The one rope base this shell can build a ladder at.
///
/// `driver_wgpu::shell::Deployment` carries a single `theta`, and a stack that
/// runs two bases -- gemma-3 and gemma-4 alternate a local and a global one --
/// has no single answer. Refused by name rather than served at the first
/// layer's base, because the layers that wanted the other one would attend at
/// the wrong wavelengths and stay fluent.
///
/// # Errors
///
/// A row whose `theta_by_layer` is non-empty, which is exactly the "they are
/// not all the same" case that method exists to report, and a row that states
/// no usable base at all.
fn theta_of(
    row: &'static dyn model::catalog::Variant,
    deployment: &model::deployment::Deployment,
) -> Result<f32> {
    if !deployment.theta_by_layer().is_empty() {
        bail!(
            "driver-wgpu: `{}` runs more than one rope base and this driver's deployment \
             carries one. Serving it at the first layer's base would rotate the rest at the \
             wrong wavelengths, which does not fault -- it degrades.",
            row.id()
        );
    }
    let theta = deployment.attention.first().map_or(0.0, |a| a.rope_theta);
    // Refused rather than clamped to something plausible. `rope::frequencies`
    // raises `theta.powf(-2i/d)`, so a base of zero gives a ladder of zeros and
    // a negative one gives NaNs; substituting a number here would be this seam
    // inventing a model fact, which is the one thing it exists not to do.
    if theta <= 0.0 || !theta.is_finite() {
        bail!(
            "driver-wgpu: `{}` states a rope base of {theta}, and a rotary ladder cannot be \
             raised from it",
            row.id()
        );
    }
    Ok(theta)
}

/// The rescaling the row asks for, in this driver's spelling.
///
/// `driver_wgpu::rope::Rescale` is the piecewise-in-wavelength one and only
/// that -- `rope.rs` builds llama-3's ladder and no other. A YaRN row is
/// therefore refused here rather than served unrescaled: an unrescaled YaRN
/// stack runs past its trained context with the wrong ladder, and that is the
/// same finite, plausible, never-faulting failure `RopeScaling`'s own doc
/// records for the factor-of-zero regression.
///
/// # Errors
///
/// A row asking for YaRN.
fn rescale_of(
    row: &'static dyn model::catalog::Variant,
    deployment: &model::deployment::Deployment,
) -> Result<Option<driver_wgpu::rope::Rescale>> {
    match deployment.rope_scaling {
        None => Ok(None),
        Some(model::deployment::RopeScaling::Piecewise {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        }) => Ok(Some(driver_wgpu::rope::Rescale {
            factor,
            low: low_freq_factor,
            high: high_freq_factor,
            original_max: original_max_position as f32,
        })),
        Some(model::deployment::RopeScaling::Yarn { .. }) => bail!(
            "driver-wgpu: `{}` asks for YaRN and `driver_wgpu::rope` builds only the \
             piecewise ladder. Serving it unrescaled would attend past its trained context \
             with the untouched frequencies, which degrades rather than fails.",
            row.id()
        ),
    }
}

/// Every weight the decode plan binds, under the name that plan uses.
///
/// # Why the loader's own executor runs
///
/// The obvious shortcut is to read each tensor's source span out of the file
/// verbatim, and for one model it works: a `Binding::MLX_IN_PLACE` plan for
/// qwen3-0.6B is allocations, six bulk writes that tile the whole file, and
/// finalizes. qwen2.5 breaks it immediately -- its plan states hundreds of
/// `TileMap` transforms, which is what `fused_qkv: true` costs, and a verbatim
/// read hands the card three separate projections where the text binds one
/// joined weight. Not a fault on this backend; a wrong number.
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
    // The declared encoding, out of the checkpoint's OWN metadata -- the one
    // thing a catalog row genuinely cannot state, because a group size is not
    // an extent of any tensor. Read as `model/config` rather than off disk:
    // a second reader of that file is a second place for what a model is made
    // of to be decided. All three other seams read it exactly this way.
    let config =
        match model_loader::checkpoint::read::read_meta(meta, model::encoding::CONFIG_OBJECT) {
            Ok(Some(bytes)) => String::from_utf8(bytes).map_err(|e| {
                anyhow!(
                    "driver-wgpu: the embedded {} is not utf8: {e}",
                    model::encoding::CONFIG_OBJECT
                )
            })?,
            Ok(None) => bail!(
                "driver-wgpu: {} is not embedded in the checkpoint at {path:?}. Re-import it \
                 with `pie model build`; one field is read out of it -- the declared \
                 quantization -- and no kernel can be named without it.",
                model::encoding::CONFIG_OBJECT
            ),
            Err(e) => bail!("driver-wgpu: cannot read the embedded encoding: {e:?}"),
        };
    let encoding = model::encoding::Encoding::from_config_json(&config)
        .map_err(|e| anyhow!("driver-wgpu: unreadable encoding: {e}"))?;
    // `BackendKind::Vulkan`, and this backend is not Vulkan.
    //
    // There is no `Wgpu` arm, deliberately: a load plan is compiled before an
    // adapter is asked, and this shell runs over whichever of Vulkan, Metal and
    // D3D12 answered. What a target decides is alignment, tile budget and which
    // load-time transforms a plan may CARRY; it does not decide what the
    // tensors are called. The Vulkan and Metal arms state the same tile-map
    // mask -- `driver-wgpu/tests/checkpoint.rs` asserts it, and so does the
    // test at the bottom of this file -- so the two adapters this shell can
    // actually sit on are handed one plan.
    //
    // What is NOT interchangeable is `BackendKind::Unknown`: its mask omits
    // `TILE_MAP_ENCODE`, so a quantised checkpoint compiled against it produces
    // a plan that silently skips the encode. The fallback arm is the one arm
    // this must not be.
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
            "driver-wgpu: `{}`'s load plan will not compile: {e}",
            row.id()
        )
    })?;
    let storage = model_loader::executor::Execution::new(&plan, path)
        .run()
        .map_err(|e| anyhow!("driver-wgpu: `{}`'s load plan will not run: {e}", row.id()))?;

    // The conversion from a publisher's tensor names to the ones a plan states.
    // Measured in `driver-wgpu/tests/checkpoint.rs` against real snapshots.
    let naming = driver_wgpu::names::Naming::mlx();
    let mut out = Vec::new();
    for traced in wanted {
        let bytes = naming
            .spellings(traced)
            .iter()
            .find_map(|s| storage.tensors.get(s.as_str()))
            .ok_or_else(|| {
                anyhow!(
                    "driver-wgpu: `{traced}` resolves to nothing `{}`'s load plan produced",
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
/// guest sees as `channel is poisoned: pipeline: forward failed`. That is the
/// exact wall a real `pie run` on this backend hit, one greedy decode after
/// the prefill had already computed the right row.
///
/// So this is not bookkeeping the asynchronous backends do for their own
/// reasons. It is the only channel a frame has for saying that it ran. CUDA
/// writes these cells from its stream callback and `remote` writes them from
/// the executor's reply; a host-side driver writes them here, because by the
/// time `Shell::serve` has returned every one of them is already decided.
///
/// # Errors
///
/// A frame that names fewer cells than the step had members, or a null one:
/// both would have this write past what the scheduler owns, and the pointer
/// is not something a later layer can check.
fn publish_terminals(
    cells: &[*mut ::driver_api::TerminalCell],
    ran: &[driver_wgpu::frames::Ran],
) -> Result<()> {
    use driver_wgpu::frames::Ran;

    if cells.is_empty() {
        // A step with no cells is one the scheduler is not waiting on -- the
        // driver's own tests build frames this way -- and writing nothing is
        // the whole of the right answer.
        return Ok(());
    }
    if cells.len() != ran.len() {
        bail!(
            "driver-wgpu: this frame names {} terminal cells for {} members",
            cells.len(),
            ran.len()
        );
    }
    for (&cell, outcome) in cells.iter().zip(ran) {
        if cell.is_null() {
            bail!("driver-wgpu: a member of this frame has a null terminal cell");
        }
        let word = match outcome {
            Ran::Fired => ::driver_api::PIE_TERMINAL_OUTCOME_SUCCESS,
            // Not a failure and not a success: the member was skipped without
            // being touched, and the scheduler's answer to that is to post it
            // again. Writing SUCCESS here would answer a request whose program
            // never ran.
            Ran::Early => ::driver_api::PIE_TERMINAL_OUTCOME_RETRY,
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

/// Does every launch of `text`'s prefill at `rows` fit the device's grid limit?
///
/// Split out of [`widest_fire`] so a test can ask it directly: the property
/// that matters is that the search returns a BOUNDARY, and checking a boundary
/// means asking this on both sides of it.
fn every_launch_fits(text: &driver_wgpu::shell::Text, rows: u32) -> bool {
    let Ok(low) = model_compiler::lower::lower(
        &text.prefill,
        &vec![
            model_compiler::lower::Row {
                samples: true,
                ..model_compiler::lower::Row::default()
            };
            rows as usize
        ],
        model_compiler::lower::Fire {
            captures_across_splits: false,
        },
    ) else {
        return false;
    };
    for launch in &low.launches {
        let symbol = &low.kernels[launch.kernel as usize];
        let Ok(rule) = driver_wgpu::dispatch::rule_of(driver_wgpu::KERNELS, symbol) else {
            continue;
        };
        let Some(sig) = driver_wgpu::sig_in(driver_wgpu::KERNELS, symbol) else {
            continue;
        };
        let Ok(declared) =
            driver_wgpu::reflect::entrypoint(symbol, driver_wgpu::Capability::Baseline)
        else {
            continue;
        };
        let module = driver_wgpu::geometry::Module::loaded(symbol, &declared);
        let dims = driver_wgpu::dispatch::dims_of(sig, &low, launch, text.geometry);
        match driver_wgpu::geometry::groups_within(
            rule,
            dims,
            module,
            driver_wgpu::geometry::MAX_WORKGROUPS_PER_DIMENSION,
        ) {
            Ok(_) => {}
            Err(driver_wgpu::geometry::Ungeometric::Unruled(_)) => {}
            Err(_) => return false,
        }
    }
    true
}

/// The widest fire this driver can actually dispatch for `text`, up to `most`.
///
/// # Why this is measured and not stated
///
/// `max_forward_tokens` is not a description. The scheduler FORMS BATCHES
/// under it, so a number here is a promise to take a fire that wide -- and
/// this seam stated `4096` as a literal, which it could not keep. At 4096
/// tokens qwen3-0.6b's `rms_single_row` wants 65,536 workgroups on one axis
/// and the device's limit is 65,535: over by exactly one, and the fire would
/// come back `PastDeviceLimit` from a batch the engine was told to form.
///
/// The ceiling is not a constant either. `Rule::Rms` launches
/// `width.div_ceil(axis) * rows` groups, so it moves with the model's hidden
/// width and the axis its row states; a literal that happened to fit one
/// checkpoint would be wrong for the next in the direction that refuses fires.
///
/// So it is searched, over the plan this deployment actually lowered, with the
/// same `groups_within` the fire path runs. A rule this backend does not serve
/// is skipped -- `frames::unserved_in` and `dispatch` refuse those by name,
/// and a row nothing can launch does not bound how wide a fire may be.
fn widest_fire(text: &driver_wgpu::shell::Text, most: u32) -> u32 {
    let fits = |rows: u32| every_launch_fits(text, rows);

    // A fire of one row is what a decode is, and a driver that could not take
    // it could not serve at all -- so the search starts from a floor it does
    // not test rather than from zero.
    let (mut lo, mut hi) = (1u32, most.max(1));
    if fits(hi) {
        return hi;
    }
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) { lo = mid } else { hi = mid }
    }
    lo
}

/// Every weight name this model's decode plan binds.
///
/// The PLAN's list and not the checkpoint's: a checkpoint holds tensors no fire
/// reads, and staging those would cost the whole of them in device memory for
/// nothing. `scale.` names are dropped because they are the lowering's own
/// scalars rather than weights.
fn bound_names(text: &driver_wgpu::shell::Text) -> Vec<String> {
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
/// The first draft of the Vulkan version of this built a `LlamaLikeFacts` by
/// hand and called `llama_like_metal` on it. That is a second dispatch key: a
/// checkpoint's identity is settled once by `catalog::identify` from the
/// tensors, and rebuilding a model's facts on this side would decide it again
/// from something else. So the row is asked, through the one door
/// `Variant::trace`.
///
/// # Why the METAL text
///
/// Because it is the text this driver was built against and measured on.
/// `driver-wgpu`'s `tests/arena.rs` walks `llama_like_metal`'s plans for six
/// rows at two fire classes, and its checkpoint suite lowers the same. The
/// family is a naming of the kernels a text names, and `kernels-wgpu`'s table
/// is `kernels-metal`'s row for row, so those names resolve; asking for a
/// family this build has no rows for would fail at the first fire rather than
/// here.
///
/// # Errors
///
/// The ROW's refusal, carried unchanged: a model this build has no text for
/// says so in its own words rather than in a sentence this seam made up.
fn text_of(
    row: &'static dyn model::catalog::Variant,
) -> Result<(driver_wgpu::shell::Text, model::deployment::Deployment)> {
    use model::catalog::Deployed;
    use model_ir::trace::FireClass;

    // The build's kernel capabilities, and nothing about the model. g64/b4 is
    // what `mlx-community` publishes and what every measurement in
    // `driver-wgpu` was taken against.
    let binding = model::catalog::MetalBinding {
        quant_group: 64,
        quant_bits: 4,
        router_quant_group: 0,
        router_quant_bits: 0,
        moe_mxfp4: false,
        fuse_residual_gemv: true,
        paged_multi_batch: true,
        qmm_multi_batch: true,
        // `true`, as `driver-vulkan`'s seam says it and `driver-metal`'s
        // constant does not: `binding::scalars`' `derived` closure answers
        // `Source::OutWidth`, which is where `norm::add_bias` reads its row
        // pitch, so this deployment can state the Qwen-2 family's q/k/v
        // projection biases. `driver-wgpu/tests/arena.rs` says the same thing
        // from the other side -- its `wgpu_facts()` is
        // `LlamaLikeMetalFacts::synthetic()` with this one line changed -- so
        // the plans this seam produces are the plans that file walks. A backend
        // that said `false` here does not get an error; it gets a text with no
        // bias in it, which is fluent and wrong.
        add_bias: true,
    };
    let decode = row
        .trace(FireClass::Decode, Deployed::metal(&binding))
        .map_err(|e| anyhow!("driver-wgpu: `{}` has no decode text: {e}", row.id()))?;
    let prefill = row
        .trace(FireClass::Prefill, Deployed::metal(&binding))
        .map_err(|e| anyhow!("driver-wgpu: `{}` has no prefill text: {e}", row.id()))?;
    let deployment = row
        .deployment(Deployed::metal(&binding))
        .map_err(|e| anyhow!("driver-wgpu: `{}` projects no deployment: {e}", row.id()))?;
    let text = driver_wgpu::shell::Text {
        decode,
        prefill,
        geometry: driver_wgpu::dispatch::Geometry {
            q_heads: deployment.shape.q_heads,
            kv_heads: deployment.shape.kv_heads,
            // `head_dim_kernel`, not `head_dim`: phi-3's heads are 96 wide and
            // run on the 128-wide kernel, so a dispatch stating the
            // checkpoint's width addresses two thirds of what was allocated.
            head_dim: deployment.shape.head_dim_kernel,
            rotary_dims: deployment.shape.head_dim_kernel,
            // THE ROW's numbers, where the Vulkan seam writes two zeros.
            //
            // That seam zeroes them because its shell serves no mixture. This
            // one's does: `driver_wgpu::geometry` carries the routed rules and
            // `tests/arena.rs` walks gpt-oss-20b's and qwen3-30b-a3b's plans
            // through them. Zeroing here would describe a routed model as
            // dense, `Text::servable` would accept it -- `0 <= 0` -- and the
            // fire would run with no router at all. Stating them means a
            // mixture is either served or refused by a check that can see it.
            n_experts: row.load_shape().n_experts,
            experts_per_token: deployment.shape.experts_per_token,
        },
        layers: u16::try_from(deployment.layers).map_err(|_| {
            anyhow!(
                "driver-wgpu: `{}` has {} layers",
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
    use model_ir::trace::FireClass;

    /// A row's id and the fixture `driver-wgpu`'s own numbers were taken from.
    type Measured = (
        &'static str,
        fn() -> model::shared::llama_like::forward::facts::LlamaLikeFacts,
    );

    /// The rows `driver-wgpu`'s suites serve for real, by the ids those suites
    /// name them by.
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
    /// Every extent in `driver-wgpu/tests/arena.rs` and every name in its
    /// checkpoint suite was taken from `llama_like_metal(&LlamaLikeFacts::…,
    /// &wgpu_facts(), class)`, a plan those files build by hand from a fixture.
    /// This seam builds its plan a different way, from the row the tensors
    /// identified. Two ways of getting a text is exactly the drift
    /// `catalog::identify` exists to prevent, and nothing else in the tree
    /// would notice them parting: a text that differed would still lower, still
    /// bind, still fire, and answer with slightly wrong logits.
    #[test]
    fn the_seam_derives_the_text_the_driver_was_measured_against() {
        for (id, facts) in MEASURED {
            let row =
                model::catalog::find(id).unwrap_or_else(|| panic!("`{id}` is in the catalog"));
            let (text, _) = text_of(row).unwrap_or_else(|e| panic!("`{id}` has a text: {e}"));
            let fixture = facts();
            // `synthetic()` is `driver-metal`'s answer sheet and this seam is
            // not that driver: it states `add_bias`, which this backend's
            // binder serves. `tests/arena.rs`'s `wgpu_facts()` is this exact
            // struct, which is what makes the comparison worth making.
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
    /// what sizes the KV cache. A text that matched while the geometry did not
    /// would allocate a cache of the wrong width and read attention off the end
    /// of it.
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

    /// A routed row reaches this seam as a routed one.
    ///
    /// The Vulkan seam writes two zeros here and can: its shell serves no
    /// mixture. This one's does, and a zeroed pair is not a refusal -- it is a
    /// text that says "dense" for a model with a router, which `Text::servable`
    /// accepts because zero experts picked out of zero is consistent. The fire
    /// would then run the dense path over a checkpoint whose FFN weights are
    /// expert banks.
    #[test]
    fn a_mixture_is_described_as_one() {
        let row = model::catalog::find("gpt-oss-20b").expect("gpt-oss-20b is in the catalog");
        let Ok((text, _)) = text_of(row) else {
            // Its text is `moe_mxfp4`-dependent; if this build has none, there
            // is nothing to check and nothing wrong.
            eprintln!("skipped: this build has no gpt-oss-20b text");
            return;
        };
        assert!(
            text.geometry.n_experts > 0 && text.geometry.experts_per_token > 0,
            "a routed row reached the shell as a dense one: {} experts, top-{}",
            text.geometry.n_experts,
            text.geometry.experts_per_token
        );
        assert!(
            text.geometry.experts_per_token <= text.geometry.n_experts,
            "top-{} out of {} experts is not servable and `Text::servable` would say so",
            text.geometry.experts_per_token,
            text.geometry.n_experts
        );
    }

    /// The rope base comes from the row, not from `Deployment::default`.
    ///
    /// The default is 1e6. gpt-oss is trained at 150_000 and llama-3 at
    /// 500_000, and a ladder built at the wrong base does not fault: it attends
    /// at the wrong wavelengths and degrades. This walks the catalog rather
    /// than naming a row, so a row added at a third base is covered the day it
    /// lands.
    #[test]
    fn the_rope_base_is_the_rows_own() {
        let mut checked = 0;
        for id in model::catalog::ids() {
            let row = model::catalog::find(id).expect("an id the catalog just listed");
            let Ok((_, deployment)) = text_of(row) else {
                continue;
            };
            let Ok(theta) = theta_of(row, &deployment) else {
                continue;
            };
            let stated = deployment.attention.first().map_or(0.0, |a| a.rope_theta);
            assert!(
                (theta - stated).abs() < f32::EPSILON * stated.max(1.0),
                "`{}` is deployed at a rope base of {theta} and states {stated}",
                row.id()
            );
            checked += 1;
        }
        assert!(checked > 0, "no row in this build has a text to check");
    }

    /// A model this build has no text for is refused in the ROW's words.
    ///
    /// Phi-3-mini's heads are 96 wide and the Metal text names
    /// `sdpa_paged_decode_bfloat16_d_96`, a symbol no shader exports. The row
    /// says so, at length and with the alternatives it considered. This seam
    /// carries that sentence out unchanged rather than replacing it with one of
    /// its own: an operator who sees `head_dim 96` can act, and one who sees
    /// "driver-wgpu cannot load this model" cannot.
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
    /// would load a model of zero weights and fire it, answering from whatever
    /// the pool happened to hold.
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

    /// The boot document the worker writes is the boot document this seam
    /// reads.
    ///
    /// Two crates, one document, and no shared type between them:
    /// `worker`'s `wgpu_startup_toml` puts `kv_pages` under `[model]`, and
    /// this reads it from there. Nothing in the compiler connects the two -- a
    /// key moved to `[batching]`, which is where the Metal writer keeps its
    /// page geometry and so the obvious thing to copy, would boot at this
    /// seam's own default with no complaint, and an operator who asked for a
    /// bigger cache would get 1024 pages and no sign the number was ignored.
    ///
    /// The literal TOML is that document's shape written out by hand on
    /// purpose. Calling the worker's writer would be the same mistake as
    /// sharing a type: the point is that the two agree, and a test that
    /// derives one from the other cannot say so.
    #[test]
    fn the_boot_document_the_worker_writes_is_the_one_this_seam_reads() {
        assert_eq!(
            boot_kv_pages(
                br#"[model]
kv_pages = 4096
"#
            ),
            4096
        );
        // The shapes that are NOT an error, each for its own reason: a
        // hand-written config that says nothing, and a document that does not
        // parse at all -- which is what a PATH looks like from in here, and
        // the reason the worker hands over the text.
        assert_eq!(boot_kv_pages(b"[model]\n"), DEFAULT_KV_PAGES);
        assert_eq!(
            boot_kv_pages(b"/home/someone/.pie/launch/0/driver.toml"),
            DEFAULT_KV_PAGES
        );
        // And zero, which parses and is not a cache.
        assert_eq!(boot_kv_pages(b"[model]\nkv_pages = 0\n"), DEFAULT_KV_PAGES);
    }

    /// The two targets this shell can sit on compile one load plan.
    ///
    /// `stage` asks `model-loader` for `BackendKind::Vulkan` and this backend
    /// is not Vulkan: it is whichever of Vulkan, Metal and D3D12 the adapter
    /// answered on, decided at runtime and after the plan is compiled. That is
    /// only sound while the two arms admit the same transforms, which is what
    /// this measures. `Unknown` is asserted DIFFERENT because it is the arm a
    /// missing backend falls into and its mask omits `TILE_MAP_ENCODE` -- a
    /// quantised checkpoint compiled against it loads and is wrong.
    #[test]
    fn the_target_this_seam_asks_for_is_the_one_either_adapter_needs() {
        use model_loader::plan::StorageTarget;
        use model_loader::types::BackendKind;

        let mask = |kind| StorageTarget::for_backend(kind, 0, 1).tile_map_mask;
        assert_eq!(
            mask(BackendKind::Vulkan),
            mask(BackendKind::Metal),
            "the Vulkan and Metal targets have parted, so this shell on a Metal adapter and \
             the same shell on a Vulkan one would be handed different plans -- and it picks \
             its backend at runtime"
        );
        assert_ne!(
            mask(BackendKind::Vulkan),
            mask(BackendKind::Unknown),
            "`stage` may not fall back to the unknown target: its mask omits the encode a \
             quantised checkpoint needs, and the plan skips it silently"
        );
    }

    /// A launched frame says so in every member's terminal cell.
    ///
    /// # Why this is a test and not a comment
    ///
    /// The cell is the ONLY channel a frame has for reporting that it ran, and
    /// `Pending` -- what an untouched cell holds -- is a failure rather than a
    /// silence. A `launch` that returned `Ok`, fired the right shaders and
    /// wrote nothing here produced this, on a real `pie run`, one greedy
    /// decode after a prefill that had already computed the right row:
    ///
    ///     direct launch terminal settlement failed
    ///     err=work item completion terminal outcome is still Pending
    ///
    /// Nothing else in this file's tests would have caught it: the seam
    /// answered every question it was asked correctly, and the omission was a
    /// question nobody asked.
    ///
    /// The three outcomes are asserted to be DIFFERENT from each other as well
    /// as from `Pending`, because a mapping that collapsed `Early` onto
    /// `Success` would pass a test that only checked "not pending" and would
    /// answer a request whose program never ran.
    #[test]
    fn a_launched_member_is_told_what_became_of_it() {
        use ::driver_api::TerminalCell;
        use driver_wgpu::frames::Ran;

        let cells = [
            TerminalCell::pending(),
            TerminalCell::pending(),
            TerminalCell::pending(),
        ];
        // Pending BEFORE, which is what makes the assertion below a change
        // rather than a coincidence.
        for cell in &cells {
            assert_eq!(cell.load(), ::driver_api::PIE_TERMINAL_OUTCOME_PENDING);
        }
        let ptrs: Vec<*mut TerminalCell> = cells
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
            ::driver_api::PIE_TERMINAL_OUTCOME_RETRY,
            "a member skipped for being early must be re-posted, not answered"
        );
        assert_eq!(
            cells[2].load(),
            ::driver_api::PIE_TERMINAL_OUTCOME_FAILED,
            "a member whose program faulted is reported as failed"
        );
    }

    /// A frame that names the wrong number of cells is refused, not written.
    ///
    /// The write is through a raw pointer the scheduler owns; a count this
    /// side guessed at would be a write past it.
    #[test]
    fn a_mismatched_roster_is_refused_before_anything_is_written() {
        use ::driver_api::TerminalCell;
        use driver_wgpu::frames::Ran;

        let cell = TerminalCell::pending();
        let ptrs: Vec<*mut TerminalCell> = vec![std::ptr::from_ref(&cell).cast_mut()];
        let refused = publish_terminals(&ptrs, &[Ran::Fired, Ran::Fired])
            .expect_err("one cell cannot answer for two members");
        assert!(
            refused.to_string().contains("terminal cells"),
            "the refusal names what disagreed: {refused}"
        );
        assert_eq!(
            cell.load(),
            ::driver_api::PIE_TERMINAL_OUTCOME_PENDING,
            "a refused publish writes nothing at all"
        );
        // And a null cell is refused rather than dereferenced.
        let nulls: Vec<*mut TerminalCell> = vec![std::ptr::null_mut()];
        assert!(publish_terminals(&nulls, &[Ran::Fired]).is_err());
        // A step the scheduler is not waiting on names no cells, and writing
        // nothing is the whole of the right answer.
        assert!(publish_terminals(&[], &[]).is_ok());
    }

    /// The ports this seam claims are the ones it can actually resolve.
    ///
    /// The claim is not free: a driver that names `PIE_DECODE_ENVELOPE_PORTS`
    /// is handed decode envelopes whose geometry it must read off its own
    /// channels, which is what `envelope::fill` and `Programs::geometry` are
    /// for. This file claimed 0 for as long as neither existed, and the engine
    /// answered the 0 by folding the geometry on the host -- which cannot know
    /// `EmbedTokens` and said so, by name. So the pair is asserted together:
    /// the mask, and a call into the machinery that earns it.
    #[test]
    fn the_geometry_ports_this_seam_claims_are_ones_it_can_resolve() {
        // The number is a MEASUREMENT, not a constant this file also owns.
        // `forward.rs` gates the decode envelope on
        // `device_port_mask & required == required`, where `required` is
        // computed per envelope by `envelope_required_ports`; the envelope a
        // real greedy `pie run` of qwen3-0.6b built asked for this, and said
        // so when the claim was still zero:
        //
        //     decode envelope on a driver without device geometry ports
        //     (mask 0x0, needs 0x25): falling back to host-evaluated
        //     serialized execution
        //
        // Quoting the observed requirement rather than the constant this file
        // reports keeps the two sides independent: a `PIE_DECODE_ENVELOPE_PORTS`
        // that changed under us would fail here rather than agree with itself.
        const MEASURED_REQUIREMENT: u32 = 0x25;
        // The gate itself, once, so the control below asks the same question
        // of a different mask rather than restating the arithmetic. Written as
        // a closure because `0 & MEASURED_REQUIREMENT` spelled inline is
        // `clippy::erasing_op` -- correct about the expression and wrong about
        // the intent, which is that the ZERO this driver used to report does
        // not pass a gate the real one does.
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
        // And what is NOT claimed is stated too: the full Track B set is
        // wider, and this driver does not answer for it.
        assert_ne!(
            claimed & ::driver_api::PIE_DEVICE_GEOMETRY_PORTS,
            ::driver_api::PIE_DEVICE_GEOMETRY_PORTS,
            "this seam claims the decode envelope's ports, not every geometry port"
        );
        // The machinery the claim promises, asked for an instance that is not
        // there: it must answer by NAME rather than by not existing.
        let programs = driver_wgpu::programs::Programs::default();
        let refused = programs
            .geometry(7, driver_wgpu::facts::PAGE_SIZE)
            .expect_err("there is no instance 7");
        assert!(
            refused.to_string().contains('7'),
            "the refusal names the instance it could not find: {refused}"
        );
    }

    /// The widest fire is the BOUNDARY, and both sides of it are asserted.
    ///
    /// # What this replaces
    ///
    /// `max_forward_tokens` was the literal `4096`, and the scheduler forms
    /// batches under it -- so it was a promise to take a fire that wide. At
    /// 4096 tokens qwen3-0.6b's `rms_single_row` wants 65,536 workgroups on
    /// one axis against a device limit of 65,535. Over by ONE, and the fire
    /// would have come back `PastDeviceLimit` from a batch the engine had been
    /// told to form.
    ///
    /// # Why a boundary and not a number
    ///
    /// Pinning the answer would pin this model's hidden width and this row's
    /// stated axis, and fail on the next checkpoint rather than on a defect.
    /// What is a property of the SEARCH, and true for every model, is that it
    /// returns the largest fitting count: `n` fits and `n + 1` does not. That
    /// is asserted by re-running the same check `widest_fire` runs, so a
    /// search that returned something conservative -- or something one too
    /// large, which is the bug it was written for -- fails here.
    #[test]
    fn the_widest_fire_is_the_largest_one_that_fits() {
        let row = model::catalog::find("qwen3-0.6b").expect("this build has qwen3-0.6b");
        let (text, _) = text_of(row).expect("the row has a text");
        let most = 4096;
        let n = widest_fire(&text, most);
        assert!(n >= 1, "a driver that cannot take one row cannot serve");
        assert!(n <= most, "the search may not exceed what it was asked for");

        let fits = |rows: u32| every_launch_fits(&text, rows);
        assert!(fits(n), "the answer must itself fit");
        if n < most {
            assert!(
                !fits(n + 1),
                "{n} was returned while {} also fits, so the search stops short",
                n + 1
            );
        }
        eprintln!("qwen3-0.6b takes a fire of {n} tokens, not the 4096 that was claimed");
        // And it is not the literal it replaced, which is the whole point.
        assert_ne!(
            n, 4096,
            "qwen3-0.6b cannot take a 4096-token fire: `rms_single_row` wants \
             65,536 workgroups on one axis and the limit is 65,535"
        );
    }
}
