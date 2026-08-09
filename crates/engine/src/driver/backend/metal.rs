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
//! # What is servable today, and what is not
//!
//! The verbs split cleanly. `create`, `device_facts`, the registry four and
//! `close_*` are answered by machinery that is already ported and device
//! tested. `encode` refuses, as the CUDA side does — Metal media encode is
//! unsupported on both. `launch`, `copy_kv`, `copy_state` and `resize_pool`
//! need the **KV pool**, which is the frame bridge's device half and the one
//! piece still missing.
//!
//! Those four refuse by name rather than being absent. A backend that cannot
//! be selected teaches nothing; one that is selected and says exactly which
//! verb it cannot serve is a working seam with a stated hole.

use anyhow::{Result, anyhow, bail};

use crate::driver::FrameLaunchOutcome;
use crate::driver::channel::RegisteredChannel;
use crate::driver::command::{
    ChannelRegistrationPlan, KvCopyPlan, MediaEncodePlan, PoolResizePlan, ProgramRegistration,
    StateCopyPlan,
};
use crate::driver::completion::{CompletionBroker, SubmissionCompletion};
use crate::driver::instance::{BoundInstance, InstanceBindingPlan};
use crate::driver::submission::FrameSubmission;
use driver_metal::Region;

/// The Metal shell, behind the seam's fourteen verbs.
pub struct MetalDriver {
    context: std::sync::Arc<driver_metal::metal::Context>,
    /// The command timeline, held ACROSS frames.
    ///
    /// This is what makes run-ahead run-ahead rather than within-frame
    /// pipelining. The timeline and the two-allocator ring live on the
    /// stepper, so a fresh one per frame has no previous value to compare
    /// against and no allocator to alternate: frame n+1 could not be queued
    /// while frame n ran, however the steps inside each were arranged.
    ///
    /// `Stepper::shared` rather than `Stepper::new` because a borrowing
    /// stepper beside the `Context` it borrows is a self-reference; sharing
    /// the context is what lets one outlive a call.
    stepper: driver_metal::metal::Stepper<'static>,
    /// Reusable fire regions, held ACROSS frames for the same reason the
    /// stepper is. A fresh region per fire leaks it into the residency set
    /// permanently -- nothing removes -- and moves an address that is one of
    /// only three things differing between two fires of one shape.
    scratch: driver_metal::metal::Scratch,
    /// Which buffer each address belongs to. A recorded command binds a
    /// BUFFER where this driver otherwise binds an address, so recording
    /// needs the inverse of what a fire computes.
    regions: driver_metal::metal::Regions,
    /// Fires already recorded, by what they are valid for. Replaying one
    /// costs 39.8 us where encoding the same fire costs 14.87 ms.
    recordings: driver_metal::metal::Recordings,
    registry: driver_metal::pipeline::Registry,
    device_facts: ::driver_api::DeviceFacts,
    /// The checkpoint, once one is loaded. Held because every address in its
    /// tensor map points into the region it owns.
    model: Option<driver_metal::model::load::Loaded>,
    /// What the checkpoint said it is — which text `model::text` looks up.
    arch: String,
    /// The paged KV pool, allocated at load.
    pool: Option<driver_metal::model::kv::Pool>,
    /// `[model] descriptor` from the boot TOML.
    ///
    /// The one key this seam reads out of `config_bytes`, and the same one
    /// `driver-cuda`'s shell reads. Model facts come from the
    /// `pie.model/1` descriptor the worker hands over — **not** from a
    /// checkpoint's `config.json`, which `model::config` normalizes ONCE and
    /// which `crates/model/tests/one_normalizer.rs` refuses to let the runtime
    /// read a second time.
    boot_descriptor: Option<std::path::PathBuf>,
    /// Whether the loaded checkpoint has GDN / linear-attention layers.
    ///
    /// A control-op capability rather than a shape: the recurrent state only
    /// exists if it does, so `copy_state` and `copy_kv` ask it before planning.
    has_linear_attn: bool,
    /// The deployment's facts, derived at load from the descriptor.
    ///
    /// Held rather than re-derived per fire: they come from a file, and a
    /// second reading is a second chance to disagree with the first.
    /// The rotary ladder, derived ONCE at load.
    ///
    /// A load-time derivation and not a per-fire one: a deployment that
    /// rescales its frequencies (llama-3, YaRN) states the rescaling in its
    /// config, and the config does not change between fires. Held as f32 bits
    /// because that is the channel it rides.
    inv_freq: Vec<u32>,
    deployment: Option<(
        model::families::llama_like::forward::facts::LlamaLikeFacts,
        model::families::llama_like::forward::facts::LlamaLikeMetalFacts,
    )>,
    /// The runtime shader compiler, and the pipelines a fire's symbols have
    /// compiled to. Held across fires: a model's symbol set is bounded by its
    /// text, so a driver that recompiled per fire would spend more time in the
    /// compiler than on the GPU.
    compiler: driver_metal::metal::Compiler,
    pipelines: driver_metal::model::encode::Pipelines,
    broker: CompletionBroker,
}

// The context holds Objective-C objects, which are not `Send` by declaration.
// The seam owns the driver exclusively and the scheduler drives it from one
// place, which is the same reason `DummyDriver` asserts this.
unsafe impl Send for MetalDriver {}
unsafe impl Sync for MetalDriver {}

impl MetalDriver {
    /// Open the default Metal 4 device.
    ///
    /// # Errors
    ///
    /// No Metal 4 device, or a device whose queue could not be created. Both
    /// are boot conditions, not runtime ones.
    pub fn create(config_bytes: &[u8]) -> Result<(Self, ::driver_api::DeviceFacts)> {
        let boot_descriptor = std::str::from_utf8(config_bytes)
            .ok()
            .and_then(|text| text.parse::<toml::Table>().ok())
            .and_then(|v| {
                v.get("model")?
                    .get("descriptor")?
                    .as_str()
                    .map(std::path::PathBuf::from)
            });
        let context = std::sync::Arc::new(
            driver_metal::metal::Context::new()
                .map_err(|e| anyhow!("metal context: {e:?}"))?,
        );
        let stepper = driver_metal::metal::Stepper::shared(context.clone())
            .map_err(|e| anyhow!("metal stepper: {e:?}"))?;
        let compiler = driver_metal::metal::Compiler::new(&context)
            .map_err(|e| anyhow!("metal compiler: {e:?}"))?;
        // The facts a scheduler reads, stated from what this backend IS
        // rather than parsed out of a config — a config that disagreed with
        // the hardware would simply be believed.
        //
        // `unified_memory` is the one that changes scheduling: on Apple
        // silicon the KV pool and the host share physical memory, so a
        // "device is full" question is a different question here.
        let device_facts = ::driver_api::DeviceFacts {
            abi_version: ::driver_api::PIE_DRIVER_ABI_VERSION,
            backend: "metal".to_string(),
            unified_memory: true,
            // Metal has no native fp8 path and no MXFP4 MoE kernel; the table
            // says which kernels exist and neither is among them.
            fp8_native: false,
            native_mxfp4_moe: false,
            storage_alignment: 256,
            storage_max_tile_bytes: 0,
            storage_tile_map_mask: 0,
            // The paged KV pool's rows per page, which every `kv_translation`
            // index is in units of.
            page_size: 16,
        };
        Ok((
            Self {
                context: context.clone(),
                stepper,
                scratch: driver_metal::metal::Scratch::new(),
                regions: driver_metal::metal::Regions::new(),
                recordings: driver_metal::metal::Recordings::new(),
                registry: driver_metal::pipeline::Registry::new(),
                device_facts: device_facts.clone(),
                model: None,
                arch: String::new(),
                pool: None,
                boot_descriptor,
                inv_freq: Vec::new(),
                deployment: None,
                has_linear_attn: false,
                compiler,
                pipelines: driver_metal::model::encode::Pipelines::new(shader_tree()),
                broker: CompletionBroker::new(),
            },
            device_facts,
        ))
    }

    /// The device's stated facts.
    #[must_use]
    pub fn device_facts(&self) -> &::driver_api::DeviceFacts {
        &self.device_facts
    }

    /// Metal exports no KV handle: there is no cross-process sharing path.
    #[must_use]
    pub fn export_kv_handle(&self) -> Option<::driver_api::KvHandle> {
        None
    }

    /// The device this driver runs on.
    #[must_use]
    pub fn context(&self) -> &driver_metal::metal::Context {
        &self.context
    }

    /// The program/instance/channel registry.
    #[must_use]
    pub fn registry(&self) -> &driver_metal::pipeline::Registry {
        &self.registry
    }

    /// Author the checkpoint's load plan, run it, and stage every tensor.
    ///
    /// One descriptor: this backend holds one model, which is the same shape
    /// the CUDA shell's `state.model` has and the reason a frame's instance
    /// roster is one family's.
    ///
    /// # Errors
    ///
    /// More than one descriptor, a missing `config.json`, or a plan that will
    /// not compile or stage.
    pub fn load_model(
        &mut self,
        descs: Vec<::driver_api::ModelLoadDesc>,
    ) -> Result<::driver_api::DriverCapabilities> {
        let [desc] = descs.as_slice() else {
            bail!(
                "driver-metal holds ONE model; got {} descriptors",
                descs.len()
            );
        };
        // The load plan is authored from the `pie.model/1` DESCRIPTOR, and
        // this seam does not make one. `model::config` normalizes a snapshot
        // exactly once, upstream, and `crates/model/tests/one_normalizer.rs`
        // refuses to let the runtime read a checkpoint's own config a second
        // time — two normalizers is how they come to disagree.
        let path = self.boot_descriptor.as_ref().ok_or_else(|| {
            anyhow!(
                "driver-metal: no `[model] descriptor` in the boot config. \
                 Model facts come from the descriptor the worker hands over, \
                 not from the checkpoint — see crates/model/tests/one_normalizer.rs."
            )
        })?;
        let descriptor =
            std::fs::read_to_string(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        let loaded =
            driver_metal::model::load::load(&self.context, &desc.snapshot_dir, &descriptor)
                .map_err(|e| anyhow!("metal load: {e:?}"))?;
        let facts = driver_metal::facts::ModelFacts::from_descriptor(&descriptor)
            .ok_or_else(|| anyhow!("the descriptor does not parse as model facts"))?;
        self.arch = facts.arch_name.clone();
        self.has_linear_attn = facts.has_linear_attn;
        if !driver_metal::model::text::serves(&self.arch) {
            bail!(
                "driver-metal has no Metal text for `{}`; it serves {:?}. \
                 The checkpoint loaded, but nothing states its forward pass.",
                self.arch,
                driver_metal::model::text::known()
            );
        }

        // The pool, at the geometry the checkpoint states. `PIE_METAL_KV_PAGES`
        // is the size knob: a pool is a fixed allocation on this backend, and
        // the number the engine would negotiate is the number it is told here.
        let pages: u32 = std::env::var("PIE_METAL_KV_PAGES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        // The geometry, DERIVED from the descriptor rather than guessed.
        //
        // `ModelFacts`'s `go_*` fields are gpt-oss's alone — their own docs say
        // a non-zero layer count marks "this config was read as gpt-oss" — so
        // reading them for a llama checkpoint allocates a pool of no layers.
        // `geometry_from_facts` is the general derivation and refuses rather
        // than defaulting. It answers a `DecodeGeometry`, which is on the
        // retirement list; when it goes, this reads the descriptor directly.
        // What is borrowed here is arithmetic over the config, not a model
        // definition.
        let geometry = driver_metal::batch::geometry_from_facts(&facts).map_err(|why| {
            anyhow!("the descriptor does not describe a servable family: {why:?}")
        })?;

        // The deployment's facts, from the geometry the descriptor states and
        // the tensors the checkpoint actually shipped. The three binding facts
        // — qk-norm, fused QKV, attention bias — ask the TENSORS, because a
        // config states an architecture and a tensor states a binding.
        // Two probes: which tensors the checkpoint shipped, and which of them
        // the load left in MXFP4. The second is what a MIXTURE needs -- a
        // checkpoint need not quantize uniformly, and reading an expert bank
        // with the dense format is NaNs rather than a near miss.
        self.deployment = Some(driver_metal::model::text::facts_from_with(
            &geometry,
            |name| loaded.tensors.contains_key(name),
            |name| loaded.mxfp4.contains(name),
        ));
        self.inv_freq = driver_metal::model::rope::frequencies(
            geometry.head_dim,
            geometry.rope_theta,
            (geometry.rope_freq_factor > 0.0).then_some(driver_metal::model::rope::Rescale {
                factor: geometry.rope_freq_factor,
                low: geometry.rope_low_freq_factor,
                high: geometry.rope_high_freq_factor,
                original_max: geometry.rope_original_max_position as f32,
            }),
        )
        .iter()
        .map(|f| f.to_bits())
        .collect();
        // Which buffer each weight address belongs to, so a fire can be
        // RECORDED. A model reload moves every address, so the old recordings
        // are invalid -- stated rather than left to the fingerprint, which
        // would also catch it but says nothing about why.
        self.recordings.clear();
        self.regions = driver_metal::metal::Regions::new();
        self.regions.add(&loaded.region);
        self.model = Some(loaded);
        let shape = driver_metal::model::kv::Shape {
            layers: geometry.n_layers,
            kv_heads: geometry.n_kv_heads,
            head_dim: geometry.head_dim,
            page_size: self.device_facts.page_size,
            pages,
            element_bytes: 2,
            // The FULL-attention layers' own shape, when the checkpoint
            // states a second one. Zero everywhere but gemma-4, and the pool
            // reads the zeros as "one shape for the whole stack".
            //
            // `full_attn_every` is the same rule `model::text` derives
            // `window_left` from, so the pool and the text agree about which
            // layers are full without a second list to keep in step.
            global_head_dim: geometry.global_head_dim,
            global_kv_heads: geometry.global_kv_heads,
            full_attn_every: geometry.full_attn_every,
        };
        let pool = driver_metal::model::kv::Pool::allocate(&self.context, shape)
            .map_err(|e| anyhow!("kv pool: {e:?}"))?;
        // Every layer's K and V, for the same reason as the weights.
        for l in 0..shape.layers {
            if let Some(layer) = pool.layer(l) {
                self.regions.add(&layer.k);
                self.regions.add(&layer.v);
            }
        }
        self.pool = Some(pool);

        // What the checkpoint states, and what the pool states.
        //
        // `total_pages` is the pool's own count now, so a scheduler admits
        // against what was actually allocated. It read zero while no pool
        // existed, which was the truth then and the reason nothing was
        // admitted.
        Ok(::driver_api::DriverCapabilities {
            abi_version: ::driver_api::PIE_DRIVER_ABI_VERSION,
            total_pages: pages,
            kv_page_size: self.device_facts.page_size,
            swap_pool_size: 0,
            kv_copy_domain_mask: 0,
            rs_cache_required: facts.has_linear_attn,
            rs_cache_slots: 0,
            rs_cache_slot_bytes: 0,
            // Zero because nothing here GROWS a pool, not because the
            // machinery is missing. `metal::elastic` is a complete
            // subsystem -- arena, budget, pressure, `create_elastic` -- with
            // 452 lines of tests and no production caller: `Pool::allocate`
            // above takes a fixed page count and never revisits it.
            //
            // Stated rather than left blank, because a zero that means "not
            // wired" and a zero that means "not supported" read identically
            // to a scheduler and only one of them is a TODO. Advertising a
            // non-zero budget before the pool can honour it would be the
            // worse error: the scheduler would admit against pages that
            // never arrive.
            elastic_page_bytes: 0,
            elastic_budget_pages: 0,
            has_mtp_logits: false,
            has_mtp_drafts: false,
            has_value_head: false,
            // Every one of these is a SINK this backend cannot honour, and the
            // `kernel!` rows say so: `sdpa_vector_decode` and
            // `sdpa_paged_decode` both declare `lacks = [Scores,
            // PageMaskSink]`. Advertising one would make a program bind and
            // then run as a silent no-op.
            has_kv_envelopes: false,
            has_attn_score: false,
            has_attn_page_mask: false,
            has_lora: false,
            model_site_summary: ::driver_api::ModelSiteSummary::default(),
            device_geometry_port_mask: 0,
            // The ceilings a scheduler batches under. Stated rather than
            // unbounded: a fire wider than this has no arena sized for it.
            max_forward_tokens: 4096,
            max_forward_requests: 256,
            max_page_refs: pages,
            arch_name: facts.arch_name.clone(),
            vocab_size: facts.vocab_size,
            max_model_len: facts.max_model_len,
            activation_dtype: "bf16".to_string(),
            hidden_size: geometry.hidden,
            supports_media_encode: false,
            snapshot_dir: desc.snapshot_dir.display().to_string(),
            kv_handle: None,
            // Metal compiles its shaders at run time from the tree; nothing
            // upstream needs to generate a kernel for it.
            codegen_backend: String::new(),
        })
    }

    /// The tensors the loaded checkpoint published, or `None` before a load.
    #[must_use]
    pub fn model(&self) -> Option<&driver_metal::model::load::Loaded> {
        self.model.as_ref()
    }

    /// Register a PTIR program: its launch package and whatever kernels the
    /// host generated for it.
    ///
    /// Memoised by hash inside the registry, so a re-registration costs a
    /// lookup — which is the engine's assumption and not an optimisation
    /// added here.
    ///
    /// # Errors
    ///
    /// A package the registry refuses (a channel whose shape it cannot serve,
    /// a stage it cannot read).
    pub fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64> {
        self.registry
            .register_program(
                desc.program_hash,
                desc.launch.clone(),
                // Field for field the same record under two names: the ABI's
                // and the registry's. Converted rather than aliased, because
                // the registry's is the one it validates against and a type
                // alias would let a future field diverge silently.
                desc.emitted_kernels
                    .iter()
                    .map(|k| driver_metal::pipeline::EmittedKernel {
                        kind: k.kind,
                        stage_index: k.stage_index,
                        region_index: k.region_index,
                        entry_name: k.entry_name.clone(),
                        source: k.source.clone(),
                        error: k.error.clone(),
                    })
                    .collect(),
            )
            .map_err(|e| anyhow!("metal register_program: {e:?}"))
    }

    /// # Errors
    ///
    /// As [`Self::register_program`].
    /// Register a channel and hand back where its ring lives.
    ///
    /// The ring is HOST memory on this backend, exactly as it is on the dummy
    /// driver: `ChannelState` holds the cells and four control words, and the
    /// binding is their addresses. Nothing about the channel plane is on the
    /// GPU — it is a different layer from the model forward and always has
    /// been (`.wiki/driver/progress-metal.md`).
    ///
    /// # Errors
    ///
    /// A shape or dtype the registry will not serve, or a duplicate id.
    pub fn register_channel(
        &mut self,
        desc: &ChannelRegistrationPlan,
    ) -> Result<RegisteredChannel> {
        let spec = driver_metal::pipeline::ChannelSpec {
            id: desc.channel_id,
            dtype: desc.dtype,
            shape: desc.shape.clone(),
            capacity: desc.capacity,
            role: driver_metal::pipeline::HostRole::from_wire(desc.host_role),
            seeded: desc.seeded,
            direction: driver_metal::pipeline::Direction::from_wire(desc.extern_dir),
            extern_name: desc.extern_name.clone(),
        };
        let endpoint = self
            .registry
            .register_channel(spec)
            .map_err(|e| anyhow!("metal register_channel: {e:?}"))?;
        Ok(RegisteredChannel {
            driver_id: desc.driver_id,
            binding: ::driver_api::PieChannelEndpointBinding {
                channel_id: endpoint.channel_id,
                mirror_base: endpoint.mirror_base,
                word_base: endpoint.word_base,
                mirror_bytes: endpoint.mirror_bytes as u64,
                word_bytes: endpoint.word_bytes as u64,
                cell_bytes: endpoint.cell_bytes,
                capacity: endpoint.capacity,
                // The ABI's order, and `ChannelState`'s: head, tail, poison,
                // closed. Stated here as constants because the two sides index
                // the same four words and neither can move without the other.
                head_word_index: 0,
                tail_word_index: 1,
                poison_word_index: 2,
                closed_word_index: 3,
            },
            reader_wait_id: desc.reader_wait_id,
            writer_wait_id: desc.writer_wait_id,
        })
    }

    /// # Errors
    ///
    /// As [`Self::register_program`].
    /// Attach an instance of a registered program to its channels.
    ///
    /// # Errors
    ///
    /// A program id the registry does not hold, a channel an instance may not
    /// attach to, or a geometry class it does not serve.
    pub fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        let geometry = driver_metal::pipeline::Geometry::from_wire(desc.geometry_class as u32)
            .map_err(|e| anyhow!("metal bind_instance: {e:?}"))?;
        let seeds: Vec<(u64, Vec<u8>)> = desc
            .seed_values
            .iter()
            .map(|v| (v.channel, v.bytes.clone()))
            .collect();
        let requested = (desc.requested_instance_id != 0).then_some(desc.requested_instance_id);
        let instance_id = self
            .registry
            .bind_instance(
                desc.program_id,
                requested,
                geometry,
                &desc.channel_ids,
                &seeds,
            )
            .map_err(|e| anyhow!("metal bind_instance: {e:?}"))?;
        let binding = ::driver_api::PieInstanceBinding {
            instance_id,
            geometry_class: desc.geometry_class as u32,
            reserved0: 0,
        };
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
    /// The whole body is the four calls the executor is made of, with
    /// admission in front. Nothing here decides what runs — the text states
    /// it, `lower` flattens it, and `run` walks the result.
    ///
    /// # Errors
    ///
    /// A frame whose step tables do not describe its rows, an architecture no
    /// text serves, or a device failure. Admission is NOT an error: a frame
    /// that does not fit reports [`FrameLaunchOutcome::Exhausted`], which the
    /// engine re-posts, or `Impossible` when no eviction could ever make room.
    pub fn launch(&mut self, frame: &FrameSubmission) -> Result<FrameLaunchOutcome> {
        let (Some(model), Some(pool)) = (self.model.as_ref(), self.pool.as_ref()) else {
            bail!("driver-metal: launch before load_model");
        };

        // ── Admission, against the frame-union demand. ──
        //
        // Before anything is encoded, and without side effects, which is what
        // lets the engine re-post: a frame that took an arena and then failed
        // to admit would have to be undone.
        if !pool.admits(frame.required_kv_pages) {
            // Impossible rather than Exhausted when no eviction could make
            // room — the demand exceeds the physical pool, so waiting is
            // waiting for something that cannot happen.
            return Ok(FrameLaunchOutcome::Impossible);
        }

        // ── The page translation, checked per lane. ──
        //
        // A page past the pool addresses another layer's memory and attention
        // would read it without complaint, so this is a refusal and not a
        // clamp.
        for lane in 0..frame.instance_ids.len() {
            driver_metal::model::kv::translate(
                pool,
                &frame.kv_translation,
                &frame.kv_translation_indptr,
                lane,
            )
            .map_err(|why| anyhow!("frame kv translation: {why:?}"))?;
        }

        let (facts, metal) = self
            .deployment
            .clone()
            .ok_or_else(|| anyhow!("driver-metal: launch before load_model"))?;
        let named = std::collections::HashMap::new();

        // ONE timeline for the whole frame, so a step is QUEUED while the
        // previous one runs rather than after it finishes.
        //
        // Every step used to build its own `Stepper` and end in a wait, which
        // made the frame N submissions and N full GPU stalls. `Stepper` is
        // bounded internally -- it waits for the step two back, because there
        // are two command allocators -- so this is one fire in flight while
        // one runs, which is the shape run-ahead wants
        // (`.wiki/new-driver/next.md`, priority 1).
        //
        // Command buffers committed to one queue execute in submission order,
        // which is what makes this SAFE for steps that depend on each other:
        // step n+1 reads the KV step n appended.
        //
        // Still per-FRAME rather than per-driver, and the reason is a
        // lifetime: `Stepper<'ctx>` borrows the `Context` this struct owns, so
        // holding one across `launch` calls is a self-reference. Making it own
        // an `Arc<Context>` is what across-frame run-ahead needs next.
        let mut in_flight: Vec<(&crate::driver::submission::StepSubmission, _)> = Vec::new();

        for step in &frame.steps {
            let s = driver_metal::model::frame::Step {
                token_ids: &step.plan.token_ids,
                qo_indptr: &step.plan.qo_indptr,
                region_row_indptr: &step.region_row_indptr,
                region_sig: &step.region_sig,
                region_k: &step.region_k,
                sampling_indices: &step.plan.sampling_indices,
            };
            let class = driver_metal::model::frame::fire_class(&s);
            let plan = driver_metal::model::text::plan_for(&self.arch, class, &facts, &metal)
                .map_err(|why| anyhow!("no text: {why:?}"))?;
            let lowered = driver_metal::model::frame::lower_step(&plan, &s)
                .map_err(|why| anyhow!("step did not lower: {why:?}"))?;

            let geometry = driver_metal::model::dispatch::Geometry {
                q_heads: facts.q_heads,
                kv_heads: facts.kv_heads,
                head_dim: facts.head_dim,
                rotary_dims: facts.head_dim,
                // The DEPLOYMENT's, not zero. `Rule::RouterLane`/`RouteRows`/
                // `RoutedQmv` read these off the dims the same way `Qmv` reads
                // `width`, so a mixture handed zeros launches a router over no
                // experts -- which is a fire that runs and routes nothing.
                n_experts: facts.n_experts,
                experts_per_token: facts.experts_per_token,
            };
            // The fire's own tables, staged into one device region. The row
            // names which a slot wants and this answers — the driver never
            // reads what a table MEANS, only where the frame put it.
            //
            // `i32` throughout: the shader reads some as `uint` and some as
            // `uchar`, and a `u32` written little-endian is the same first
            // byte. The narrowing is the kernel's and the width is the
            // frame's, which is the direction that is safe.
            // Every CSR invariant, checked BEFORE the pool is touched.
            //
            // The derivation below used `unwrap_or(0)` three times, and the
            // third one was the defect: a short or mis-sized CSR resolved a
            // token's physical KV page to **0**, which belongs to some other
            // request, and the fire wrote this request's keys over that
            // request's cache. Nothing faults and the damage lands on a
            // request that did nothing wrong.
            //
            // There is no safe fallback page, so the only correct answer is to
            // refuse the frame — and refusing has to happen here, before
            // anything is staged, which is the `decide, then move` rule
            // `store/control.rs` records the cost of breaking.
            step.plan
                .validate_geometry()
                .map_err(|e| anyhow!("this frame's geometry: {e}"))?;
            step.plan
                .validate_kv_writes(pool.shape().page_size)
                .map_err(|e| anyhow!("this frame's KV writes: {e}"))?;
            // Where the paged append writes each token: its physical page and
            // the row inside it. Driver arithmetic over a driver allocation --
            // the frame states a POSITION in a sequence and a page table, and
            // this normalizes the pair. `batch::fire_csr` computes the same
            // two the same way for the retiring path.
            //
            // Every lookup is infallible now: `validate_kv_writes` has already
            // proved each token's virtual page sits inside its own request's
            // span, so an `expect` here states a checked fact rather than
            // papering over an unchecked one.
            let (w_page, w_off) = {
                let page = pool.shape().page_size.max(1);
                let req = step.plan.req_of_token();
                let (mut pages, mut offs) = (Vec::new(), Vec::new());
                for (t, &pos) in step.plan.position_ids.iter().enumerate() {
                    let r = req[t] as usize;
                    let base = step.plan.kv_page_indptr[r] as usize;
                    let virt = base + (pos / page) as usize;
                    pages.push(step.plan.kv_page_indices[virt]);
                    offs.push(pos % page);
                }
                (pages, offs)
            };
            let req = step.plan.req_of_token();
            let staged = driver_metal::model::tables::stage(
                &self.context,
                driver_metal::model::tables::Frame {
                    token_ids: &step.plan.token_ids,
                    position_ids: &step.plan.position_ids,
                    req_of_token: &req,
                    kv_page_indices: &step.plan.kv_page_indices,
                    kv_page_indptr: &step.plan.kv_page_indptr,
                    kv_write_page: &w_page,
                    kv_write_offset: &w_off,
                    rope_frequencies: &self.inv_freq,
                    sampling_indices: &step.plan.sampling_indices,
                },
            )
            .map_err(|e| anyhow!("fire tables: {e:?}"))?;
            // The fire's tables, and the stand-in for an operand that
            // addresses NOTHING -- `dispatch::bind` answers an unfilled slot
            // with address zero, which `encode` binds happily and a recorded
            // command cannot. The tables region serves as that stand-in: it
            // is real, resident, and no statement writes through a slot it
            // did not fill.
            self.regions.add(&staged.region);
            self.regions.set_null(&staged.region);
            let tables = |which| staged.at(which);

            let names = driver_metal::model::resolve::Names::mlx();
            // The KV pages a statement's state reference resolves through. A
            // closure, because the map is portable and the pool is not.
            let pages = |layer: u16, values: bool| {
                pool.layer(u32::from(layer)).map(|l| {
                    let h = if values { &l.v } else { &l.k };
                    driver_metal::model::executor::Slice {
                        address: h.gpu_address(),
                        // THIS layer's, not the pool's: gemma-4's
                        // full-attention layers hold a different page size
                        // from its sliding ones, and a slice length that
                        // over-states the region is one an attention reads
                        // past the end of.
                        bytes: pool.shape().layer_bytes_at(u32::from(layer)),
                    }
                })
            };
            let mut store =
                driver_metal::model::resolve::Store::new(names, &model.tensors, &named)
                    .with_kv(&pages)
                    .with_fire(&tables)
                    // The shape the pool was allocated at, which is where the
                    // attention kernels' strides come from. A store without it
                    // answers zero, and a zero seq stride is every step of the
                    // scan reading the same token.
                    .with_pool(pool.shape());
            let mut machine = driver_metal::model::run::Machine {
                context: &self.context,
                compiler: &self.compiler,
                pipelines: &mut self.pipelines,
                stepper: &mut self.stepper,
                scratch: &self.scratch,
                regions: &mut self.regions,
                recordings: &mut self.recordings,
            };
            let fire = driver_metal::model::run::submit(
                &mut machine,
                &lowered,
                geometry,
                &mut store,
            )
            .map_err(|e| {
                // A fire that could not bind names them all, because a
                // checkpoint missing one tensor is usually missing a family of
                // them and stopping at the first costs a round trip each.
                let missed = store.missed();
                if missed.is_empty() {
                    anyhow!("metal fire: {e:?}")
                } else {
                    anyhow!("metal fire: {e:?}; unresolved names: {missed:?}")
                }
            })?;
            // Committed, not waited for. `lowered` is dropped at the end of
            // this iteration, so the read-out's shape is carried forward with
            // the fire rather than looked up again.
            in_flight.push((step, (fire, lowered.readout)));
        }

        // ── The read-outs, and the channel plane over them. ──
        //
        // After the whole frame is committed, in submission order. Reading an
        // arena before its fire retires is reading whatever the last fire left
        // there, which is a plausible tensor and the wrong one.
        for (step, (fire, readout)) in &in_flight {
            self.stepper
                .wait_for(fire.value)
                .map_err(|e| anyhow!("metal fire {}: {e:?}", fire.value))?;
            // What the fire COMPUTED, handed to the programs bound to this
            // frame. Until this landed the seam ran every launch and dropped
            // the arena, so a green frame and a frame that computed the wrong
            // thing were the same observation — `pipeline::step` had no
            // production caller at all, and the interpreter was exercised
            // only by tests that built their own inputs.
            let logits = read_logits(&fire.arena, *readout);
            Self::run_programs(&mut self.registry, &frame.instance_ids, step, logits.as_ref())?;
        }

        let (_raw, completion) = self.broker.launch_completion(1);
        Ok(FrameLaunchOutcome::Launched(completion))
    }

    /// # Errors
    ///
    /// Always. Media encode is unsupported on this backend, as it is on CUDA;
    /// both seams refuse rather than pretending.
    pub fn encode(&mut self, _plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        bail!("driver-metal: media encode is unsupported on this backend")
    }

    /// Move KV pages and rows within the pool.
    ///
    /// Two halves, both already written: `store::control::plan_kv_copy`
    /// decides what would move and refuses what cannot, and `Pool::apply`
    /// runs it — as a `memmove`, because the pages are `StorageModeShared`
    /// and therefore host addressable.
    ///
    /// **Page order is load-bearing**, and the plan says so: a chain like
    /// `{1→0, 2→1}` reads page 1 for the second pair *after* the first has
    /// overwritten it. Each pair is independent and the caller sequences; a
    /// true swap needs a scratch page or separate calls.
    ///
    /// # Errors
    ///
    /// A refusal from the planner (a foreign memory domain, a page the pool
    /// does not have), or a copy that leaves a layer's region.
    pub fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<SubmissionCompletion> {
        let pool = self
            .pool
            .as_ref()
            .ok_or_else(|| anyhow!("driver-metal: copy_kv before load_model"))?;
        let caps = driver_metal::store::Capabilities {
            has_linear_attn: self.has_linear_attn,
            kv_total_pages: pool.pages(),
            rs_slots: 0,
        };
        // ONE stride for the whole pool, or no copy at all.
        //
        // A move plan states byte offsets and applies them to every layer, so
        // it needs the pool to be page-major at one stride. gemma-4's is not:
        // its full-attention layers pack their pages at 4 heads x 512 where
        // its sliding ones use 16 x 256. Planning at either and applying to
        // both lands a page apart rather than obviously wrong.
        //
        // Refused by name rather than approximated. A KV copy is prefix
        // sharing and forking, which is a feature a deployment can be without
        // -- a corrupted cache is not.
        let (Some(grid), Some(page_bytes)) = (pool.shape().grid(), pool.shape().page_bytes())
        else {
            bail!(
                "driver-metal: copy_kv needs one page stride for the pool \
                 and this model has two -- its full-attention layers are \
                 {} kv heads x {} against {} x {} on the sliding ones. \
                 Prefix sharing is unavailable on this checkpoint",
                pool.shape().heads_at(0).0,
                pool.shape().heads_at(0).1,
                pool.shape().kv_heads,
                pool.shape().head_dim,
            );
        };
        let work = driver_metal::store::plan_kv_copy(desc, caps, grid)
            .map_err(|why| anyhow!("metal copy_kv: {why:?}"))?;

        // Whole-page moves first, as page pairs; then the row cells. Both run
        // over every layer's K and V, which the stride check above is what
        // makes true.
        let mut cells = Vec::new();
        for &(src, dst) in &work.pages {
            cells.push(driver_metal::store::CellCopy {
                src_off: u64::from(src) * page_bytes,
                dst_off: u64::from(dst) * page_bytes,
                bytes: page_bytes,
            });
        }
        if !cells.is_empty() {
            pool.apply(&driver_metal::store::CellMovePlan {
                copies: cells,
                pages_touched: work.pages_touched,
            })
            .map_err(|e| anyhow!("metal copy_kv: {e:?}"))?;
        }
        if let Some(plan) = work.cells.as_ref() {
            pool.apply(plan)
                .map_err(|e| anyhow!("metal copy_kv: {e:?}"))?;
        }

        // Settled already: the move ran on the host, so nothing is in flight
        // and a completion the caller waits on would wait for nothing.
        let (_raw, completion) = self.broker.pie_completion(1);
        Ok(completion)
    }

    /// # Errors
    ///
    /// As [`Self::copy_kv`].
    pub fn copy_state(&mut self, _desc: &StateCopyPlan) -> Result<SubmissionCompletion> {
        bail!(UNSERVED_MOVE)
    }

    /// # Errors
    ///
    /// As [`Self::copy_kv`].
    pub fn resize_pool(&mut self, _desc: &PoolResizePlan) -> Result<SubmissionCompletion> {
        bail!(UNSERVED_MOVE)
    }

    /// # Errors
    ///
    /// Never today; the registry accepts a close of an id it does not hold,
    /// because a close is idempotent from the scheduler's side.
    pub fn close_instance(&mut self, id: u64) -> Result<()> {
        // A close of an id the registry does not hold is not an error: the
        // scheduler closes on its own schedule and a double close is how a
        // teardown race reads from this side.
        let _ = self.registry.close_instance(id);
        Ok(())
    }

    /// # Errors
    ///
    /// As [`Self::close_instance`].
    pub fn close_channel(&mut self, id: u64) -> Result<()> {
        let _ = self.registry.close_channel(id);
        Ok(())
    }

    /// Run the channel-plane pass for every program batched into one step.
    ///
    /// One instance per roster row, in sub-batch order, each over ITS OWN
    /// rows of the read-out: the fire produced one distribution per request
    /// and the members of a batch are those requests, so member `p` reads
    /// `program_row_indptr[p]..[p+1]` and nothing else.
    ///
    /// A blocked pass is not an error. Readiness is the program's own gate and
    /// missing it means the fire did not happen for that member — the
    /// interpreter changed nothing, and the engine re-posts. A FAULT is also
    /// not an error here, for a different reason: it poisons the one instance
    /// that faulted, and failing the whole frame would take down every other
    /// request batched with it for a fault that is one program's.
    ///
    /// # Errors
    ///
    /// A roster row that names no bound instance — which is a frame the
    /// scheduler built against a registry it did not have.
    fn run_programs(
        registry: &mut driver_metal::pipeline::Registry,
        instance_ids: &[u64],
        step: &crate::driver::submission::StepSubmission,
        logits: Option<&(Vec<f32>, u32, u32)>,
    ) -> Result<()> {
        for (member, &row) in step.roster_rows.iter().enumerate() {
            let id = *instance_ids
                .get(row as usize)
                .ok_or_else(|| anyhow!("roster row {row} is outside the frame's instances"))?;
            // THIS member's rows of the read-out, and nothing else.
            //
            // Every instance in the frame used to be handed the whole logits
            // buffer, and `bind_intrinsic` reads it from `base_row = 0` — so
            // in an M>1 frame every request sampled the FIRST request's
            // distribution and returned its token. One fire, N requests, one
            // answer repeated. Nothing faults, and a single-request frame
            // (which is what most tests build) cannot tell the difference.
            //
            // `program_row_indptr` is the mapping and it was already here:
            // member `p` owns wire request rows `[indptr[p], indptr[p+1])`.
            // Slicing rather than passing an offset keeps `base_row = 0`
            // TRUE for each member instead of making it a parameter every
            // caller could forget — the interpreter's view is its own rows,
            // so there is no row it could reach that is not its.
            let inputs = match logits {
                None => driver_metal::pipeline::PassInputs::none(),
                Some((values, rows, vocab)) => {
                    let (start, end) = member_rows(&step.program_row_indptr, member, *rows);
                    let span = (end - start) as usize * *vocab as usize;
                    let from = start as usize * *vocab as usize;
                    if from + span > values.len() {
                        return Err(anyhow!(
                            "member {member} claims read-out rows {start}..{end} of {rows}, \
                             which is past the {} values this fire produced",
                            values.len()
                        ));
                    }
                    driver_metal::pipeline::PassInputs {
                        logits: Some(&values[from..from + span]),
                        rows: end - start,
                        vocab: *vocab,
                        mtp_draft_row: None,
                    }
                }
            };
            match registry.fire(id, &inputs) {
                Ok(driver_metal::pipeline::StepOutcome::Committed)
                | Ok(driver_metal::pipeline::StepOutcome::Blocked(_)) => {}
                Ok(driver_metal::pipeline::StepOutcome::Faulted(why)) => {
                    tracing::warn!(instance = id, %why, "metal: program faulted");
                }
                Err(e) => return Err(anyhow!("metal program {id}: {e:?}")),
            }
        }
        Ok(())
    }
}

/// This fire's logits, widened to `f32`, with the two extents beside them.
///
/// `None` when the text states no exit seam — a fire that computes something
/// other than a distribution, which is not an error.
///
/// # Why a copy, and why widening
///
/// The interpreter's `PassInputs` wants `&[f32]`, and the metal read-out is
/// **bf16**: `affine_qmv_fast` writes bf16 whatever the text declares, which
/// is a defect the reference gate found by reading a vocabulary that was
/// exactly half zeros. So the bytes have to be reinterpreted anyway, and a
/// widening reinterpretation is a copy. The alternative — teaching the
/// interpreter a dtype — buys nothing while there is one read-out format.
///
/// bf16 → f32 is exact: the low sixteen bits are zero and every bf16 is an
/// f32. Nothing is lost here, and nothing is gained either — the precision
/// was lost in the kernel.
fn read_logits(
    arena: &driver_metal::metal::Handle,
    readout: Option<model_compiler::lower::Readout>,
) -> Option<(Vec<f32>, u32, u32)> {
    let r = readout?;
    let span = r.rows as usize * r.vocab as usize * r.bytes as usize;
    if r.at + span > arena.len() as usize {
        return None;
    }
    // SAFETY: the arena is `StorageModeShared`, so its contents are host
    // addressable, and every launch encoded against it has completed —
    // `run_keeping_arena` waits before returning.
    let raw = unsafe {
        std::slice::from_raw_parts(
            arena
                .contents()
                .as_ptr()
                .cast_const()
                .cast::<u8>()
                .add(r.at),
            span,
        )
    };
    let values: Vec<f32> = if r.bytes == 4 {
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        // `batch::widen`, not a fourth hand-rolled shift-and-cast. The
        // conversion is one line either way; having ONE of it is how a change
        // to the rounding reaches every reader.
        let halves: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        driver_metal::batch::widen(&halves)
    };
    Some((values, r.rows, r.vocab))
}

/// The hole, named once so every verb that shares it reads the same.
///
/// The pool EXISTS now — `launch` admits against it and fires. What these
/// three still want is the MOVE: `store::control` decides what a copy or a
/// resize would do and `store::kv_move` plans the offsets, both portable and
/// both tested. What is missing is the encoder that runs the plan, and the
/// reallocation a resize implies for a pool that is a fixed allocation today.
const UNSERVED_MOVE: &str = "driver-metal: KV copy/resize is not wired to the seam yet. \
     The pool exists and `launch` fires against it; `store::control::plan_kv_copy` and \
     `store::kv_move::plan_cell_moves` already decide and plan the movement. What is \
     missing is the encoder that runs the plan.";

/// Where the Metal shader tree lives.
///
/// Metal compiles at run time from `(path, entry name)`, so a driver needs the
/// `.metal` sources on disk. `PIE_METAL_KERNELS` overrides; the default is the
/// checkout's own tree, which is what a development run wants and what every
/// device test already uses.
fn shader_tree() -> std::path::PathBuf {
    std::env::var_os("PIE_METAL_KERNELS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .map(|crates| crates.join("kernels-metal/kernels"))
                .unwrap_or_default()
        })
}


/// Which rows of a fire's read-out belong to batch member `member`.
///
/// `program_row_indptr` is the frame's own attribution CSR — member `p` owns
/// wire request rows `[indptr[p], indptr[p+1])` — and an empty one is the
/// single-member case, where the whole read-out is that member's.
///
/// Split out from `run_programs` so the M>1 case can be held to a number. It
/// was wrong in a way no single-instance test could see: every member was
/// handed the WHOLE buffer and `bind_intrinsic` reads from `base_row = 0`, so
/// each request in a batched frame sampled the first request's distribution.
fn member_rows(program_row_indptr: &[u32], member: usize, rows: u32) -> (u32, u32) {
    match (
        program_row_indptr.get(member),
        program_row_indptr.get(member + 1),
    ) {
        (Some(&s), Some(&e)) if e >= s => (s, e),
        _ => (0, rows),
    }
}

#[cfg(test)]
mod readout_rows {
    use super::member_rows;

    /// Three requests batched into one fire, one read-out row each.
    ///
    /// The defect this pins: every member used to get `(0, 3)`, so all three
    /// sampled row 0 and returned the same token. One fire, three requests,
    /// one answer repeated — and nothing faults.
    #[test]
    fn each_member_of_a_batched_frame_reads_its_own_row() {
        let indptr = [0, 1, 2, 3];
        assert_eq!(member_rows(&indptr, 0, 3), (0, 1));
        assert_eq!(member_rows(&indptr, 1, 3), (1, 2));
        assert_eq!(member_rows(&indptr, 2, 3), (2, 3));
    }

    /// A member may own several rows — a speculative fire reads out more than
    /// one row per request — and the span is the CSR's, not one row.
    #[test]
    fn a_member_that_owns_several_rows_gets_all_of_them() {
        let indptr = [0, 4, 5];
        assert_eq!(member_rows(&indptr, 0, 5), (0, 4));
        assert_eq!(member_rows(&indptr, 1, 5), (4, 5));
    }

    /// No attribution CSR is the single-member case, and the whole read-out
    /// is that member's — the behaviour every frame used to get.
    #[test]
    fn an_absent_csr_gives_the_whole_readout_to_the_one_member() {
        assert_eq!(member_rows(&[], 0, 7), (0, 7));
        // A CSR too short for this member is the same answer rather than a
        // panic: it is a frame the scheduler built inconsistently, and the
        // row-range check in `run_programs` is what refuses it.
        assert_eq!(member_rows(&[0, 1], 5, 7), (0, 7));
    }
}
