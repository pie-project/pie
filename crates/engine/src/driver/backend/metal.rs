//! The seam to `driver-metal-new`.
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

/// The Metal shell, behind the seam's fourteen verbs.
pub struct MetalDriver {
    context: driver_metal_new::metal::Context,
    registry: driver_metal_new::pipeline::Registry,
    device_facts: driver_abi::DeviceFacts,
    /// The checkpoint, once one is loaded. Held because every address in its
    /// tensor map points into the region it owns.
    model: Option<driver_metal_new::model::load::Loaded>,
    /// What the checkpoint said it is — which text `model::text` looks up.
    arch: String,
    /// The paged KV pool, allocated at load.
    pool: Option<driver_metal_new::model::kv::Pool>,
    /// `[model] descriptor` from the boot TOML.
    ///
    /// The one key this seam reads out of `config_bytes`, and the same one
    /// `driver-cuda-new`'s shell reads. Model facts come from the
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
    deployment: Option<(
        model::families::llama_like::forward::facts::LlamaLikeFacts,
        model::families::llama_like::forward::facts::LlamaLikeMetalFacts,
    )>,
    /// The runtime shader compiler, and the pipelines a fire's symbols have
    /// compiled to. Held across fires: a model's symbol set is bounded by its
    /// text, so a driver that recompiled per fire would spend more time in the
    /// compiler than on the GPU.
    compiler: driver_metal_new::metal::Compiler,
    pipelines: driver_metal_new::model::encode::Pipelines,
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
    pub fn create(config_bytes: &[u8]) -> Result<(Self, driver_abi::DeviceFacts)> {
        let boot_descriptor = std::str::from_utf8(config_bytes)
            .ok()
            .and_then(|text| text.parse::<toml::Table>().ok())
            .and_then(|v| {
                v.get("model")?
                    .get("descriptor")?
                    .as_str()
                    .map(std::path::PathBuf::from)
            });
        let context = driver_metal_new::metal::Context::new()
            .map_err(|e| anyhow!("metal context: {e:?}"))?;
        let compiler = driver_metal_new::metal::Compiler::new(&context)
            .map_err(|e| anyhow!("metal compiler: {e:?}"))?;
        // The facts a scheduler reads, stated from what this backend IS
        // rather than parsed out of a config — a config that disagreed with
        // the hardware would simply be believed.
        //
        // `unified_memory` is the one that changes scheduling: on Apple
        // silicon the KV pool and the host share physical memory, so a
        // "device is full" question is a different question here.
        let device_facts = driver_abi::DeviceFacts {
            abi_version: driver_abi::PIE_DRIVER_ABI_VERSION,
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
                context,
                registry: driver_metal_new::pipeline::Registry::new(),
                device_facts: device_facts.clone(),
                model: None,
                arch: String::new(),
                pool: None,
                boot_descriptor,
                deployment: None,
                has_linear_attn: false,
                compiler,
                pipelines: driver_metal_new::model::encode::Pipelines::new(shader_tree()),
                broker: CompletionBroker::new(),
            },
            device_facts,
        ))
    }

    /// The device's stated facts.
    #[must_use]
    pub fn device_facts(&self) -> &driver_abi::DeviceFacts {
        &self.device_facts
    }

    /// Metal exports no KV handle: there is no cross-process sharing path.
    #[must_use]
    pub fn export_kv_handle(&self) -> Option<driver_abi::KvHandle> {
        None
    }

    /// The device this driver runs on.
    #[must_use]
    pub fn context(&self) -> &driver_metal_new::metal::Context {
        &self.context
    }

    /// The program/instance/channel registry.
    #[must_use]
    pub fn registry(&self) -> &driver_metal_new::pipeline::Registry {
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
        descs: Vec<driver_abi::ModelLoadDesc>,
    ) -> Result<driver_abi::DriverCapabilities> {
        let [desc] = descs.as_slice() else {
            bail!(
                "driver-metal-new holds ONE model; got {} descriptors",
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
                "driver-metal-new: no `[model] descriptor` in the boot config. \
                 Model facts come from the descriptor the worker hands over, \
                 not from the checkpoint — see crates/model/tests/one_normalizer.rs."
            )
        })?;
        let descriptor = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("{}: {e}", path.display()))?;
        let loaded = driver_metal_new::model::load::load(
            &self.context,
            &desc.snapshot_dir,
            &descriptor,
        )
        .map_err(|e| anyhow!("metal load: {e:?}"))?;
        let facts = driver_metal_new::facts::ModelFacts::from_descriptor(&descriptor)
            .ok_or_else(|| anyhow!("the descriptor does not parse as model facts"))?;
        self.arch = facts.arch_name.clone();
        self.has_linear_attn = facts.has_linear_attn;
        if !driver_metal_new::model::text::serves(&self.arch) {
            bail!(
                "driver-metal-new has no Metal text for `{}`; it serves {:?}. \
                 The checkpoint loaded, but nothing states its forward pass.",
                self.arch,
                driver_metal_new::model::text::known()
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
        let geometry = driver_metal_new::batch::geometry_from_facts(&facts)
            .map_err(|why| anyhow!("the descriptor does not describe a servable family: {why:?}"))?;

        // The deployment's facts, from the geometry the descriptor states and
        // the tensors the checkpoint actually shipped. The three binding facts
        // — qk-norm, fused QKV, attention bias — ask the TENSORS, because a
        // config states an architecture and a tensor states a binding.
        self.deployment = Some(driver_metal_new::model::text::facts_from(&geometry, |name| {
            loaded.tensors.contains_key(name)
        }));
        self.model = Some(loaded);
        let shape = driver_metal_new::model::kv::Shape {
            layers: geometry.n_layers,
            kv_heads: geometry.n_kv_heads,
            head_dim: geometry.head_dim,
            page_size: self.device_facts.page_size,
            pages,
            element_bytes: 2,
        };
        self.pool = Some(
            driver_metal_new::model::kv::Pool::allocate(&self.context, shape)
                .map_err(|e| anyhow!("kv pool: {e:?}"))?,
        );


        // What the checkpoint states, and what the pool states.
        //
        // `total_pages` is the pool's own count now, so a scheduler admits
        // against what was actually allocated. It read zero while no pool
        // existed, which was the truth then and the reason nothing was
        // admitted.
        Ok(driver_abi::DriverCapabilities {
            abi_version: driver_abi::PIE_DRIVER_ABI_VERSION,
            total_pages: pages,
            kv_page_size: self.device_facts.page_size,
            swap_pool_size: 0,
            kv_copy_domain_mask: 0,
            rs_cache_required: facts.has_linear_attn,
            rs_cache_slots: 0,
            rs_cache_slot_bytes: 0,
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
            model_site_summary: driver_abi::ModelSiteSummary::default(),
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
    pub fn model(&self) -> Option<&driver_metal_new::model::load::Loaded> {
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
                    .map(|k| driver_metal_new::pipeline::EmittedKernel {
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
    /// been (`PARITY-INTERP.md`).
    ///
    /// # Errors
    ///
    /// A shape or dtype the registry will not serve, or a duplicate id.
    pub fn register_channel(&mut self, desc: &ChannelRegistrationPlan) -> Result<RegisteredChannel> {
        let spec = driver_metal_new::pipeline::ChannelSpec {
            id: desc.channel_id,
            dtype: desc.dtype,
            shape: desc.shape.clone(),
            capacity: desc.capacity,
            role: driver_metal_new::pipeline::HostRole::from_wire(desc.host_role),
            seeded: desc.seeded,
            direction: driver_metal_new::pipeline::Direction::from_wire(desc.extern_dir),
            extern_name: desc.extern_name.clone(),
        };
        let endpoint = self
            .registry
            .register_channel(spec)
            .map_err(|e| anyhow!("metal register_channel: {e:?}"))?;
        Ok(RegisteredChannel {
            driver_id: desc.driver_id,
            binding: driver_abi::PieChannelEndpointBinding {
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
        let geometry = driver_metal_new::pipeline::Geometry::from_wire(desc.geometry_class as u32)
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
        let binding = driver_abi::PieInstanceBinding {
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
            bail!("driver-metal-new: launch before load_model");
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
            driver_metal_new::model::kv::translate(
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
            .ok_or_else(|| anyhow!("driver-metal-new: launch before load_model"))?;
        let named = std::collections::HashMap::new();

        for step in &frame.steps {
            let s = driver_metal_new::model::frame::Step {
                token_ids: &step.plan.token_ids,
                qo_indptr: &step.plan.qo_indptr,
                region_row_indptr: &step.region_row_indptr,
                region_sig: &step.region_sig,
                region_k: &step.region_k,
                sampling_indices: &step.plan.sampling_indices,
            };
            let class = driver_metal_new::model::frame::fire_class(&s);
            let plan = driver_metal_new::model::text::plan_for(&self.arch, class, &facts, &metal)
                .map_err(|why| anyhow!("no text: {why:?}"))?;
            let lowered = driver_metal_new::model::frame::lower_step(&plan, &s)
                .map_err(|why| anyhow!("step did not lower: {why:?}"))?;

            let geometry = driver_metal_new::model::dispatch::Geometry {
                q_heads: facts.q_heads,
                kv_heads: facts.kv_heads,
                head_dim: facts.head_dim,
                rotary_dims: facts.head_dim,
                n_experts: 0,
                experts_per_token: 0,
            };
            // The fire's own tables, staged into one device region. The row
            // names which a slot wants and this answers — the driver never
            // reads what a table MEANS, only where the frame put it.
            //
            // `i32` throughout: the shader reads some as `uint` and some as
            // `uchar`, and a `u32` written little-endian is the same first
            // byte. The narrowing is the kernel's and the width is the
            // frame's, which is the direction that is safe.
            // Where the paged append writes each token: its physical page and
            // the row inside it. Driver arithmetic over a driver allocation --
            // the frame states a POSITION in a sequence and a page table, and
            // this normalizes the pair. `batch::fire_csr` computes the same
            // two the same way for the retiring path.
            let (w_page, w_off) = {
                let page = pool.shape().page_size.max(1);
                let (mut pages, mut offs) = (Vec::new(), Vec::new());
                for (t, &pos) in step.plan.position_ids.iter().enumerate() {
                    let r = req_of_token(&step.plan.qo_indptr)
                        .get(t)
                        .copied()
                        .unwrap_or(0) as usize;
                    let base = step.plan.kv_page_indptr.get(r).copied().unwrap_or(0) as usize;
                    let virt = base + (pos / page) as usize;
                    pages.push(step.plan.kv_page_indices.get(virt).copied().unwrap_or(0));
                    offs.push(pos % page);
                }
                (pages, offs)
            };
            let mut blob: Vec<u32> = Vec::new();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for table in [
                step.plan.token_ids.as_slice(),
                step.plan.position_ids.as_slice(),
                // Which request owns each token, expanded from the qo CSR:
                // the scheduler states the boundaries and the kernel wants it
                // per token.
                &req_of_token(&step.plan.qo_indptr),
                step.plan.kv_page_indices.as_slice(),
                step.plan.kv_page_indptr.as_slice(),
                &w_page,
                &w_off,
            ] {
                spans.push((blob.len(), table.len()));
                blob.extend_from_slice(table);
            }
            let staged = driver_metal_new::metal::allocate(
                &self.context,
                ((blob.len() * 4).max(4)) as u64,
                "fire tables",
            )
            .map_err(|e| anyhow!("fire tables: {e:?}"))?;
            // SAFETY: freshly allocated and not yet encoded against.
            unsafe {
                use driver_metal_new::region::Region as _;
                let raw = core::slice::from_raw_parts(blob.as_ptr().cast::<u8>(), blob.len() * 4);
                staged.write(0, raw).map_err(|e| anyhow!("fire tables: {e:?}"))?;
            }
            let tables = |which: driver_metal_new::model::executor::FireTable| {
                use driver_metal_new::model::executor::FireTable as F;
                let i = match which {
                    F::TokenIds => 0,
                    F::Positions => 1,
                    F::RequestOfToken => 2,
                    F::KvPageIndices => 3,
                    F::KvPageIndptr => 4,
                    F::KvWritePage => 5,
                    F::KvWriteOffset => 6,
                    // No custom mask on this path yet; a slot nobody fills is
                    // better than one filled with the wrong table.
                    F::AttentionMask | F::AttentionMaskEnabled => return None,
                    // Numbers, not addresses: answered by `pool`.
                    F::KvHeadStride | F::KvSeqStride | F::KvPageSize => return None,
                };
                let (at, len) = spans[i];
                (len > 0).then(|| driver_metal_new::model::executor::Slice {
                    address: staged.gpu_address() + (at * 4) as u64,
                    bytes: (len * 4) as u64,
                })
            };

            let names = driver_metal_new::model::resolve::Names::mlx();
            // The KV pages a statement's state reference resolves through. A
            // closure, because the map is portable and the pool is not.
            let pages = |layer: u16, values: bool| {
                pool.layer(u32::from(layer)).map(|l| {
                    let h = if values { &l.v } else { &l.k };
                    driver_metal_new::model::executor::Slice {
                        address: h.gpu_address(),
                        bytes: pool.shape().layer_bytes(),
                    }
                })
            };
            let mut store =
                driver_metal_new::model::resolve::Store::new(names, &model.tensors, &named)
                    .with_kv(&pages)
                    .with_fire(&tables)
                    // The shape the pool was allocated at, which is where the
                    // attention kernels' strides come from. A store without it
                    // answers zero, and a zero seq stride is every step of the
                    // scan reading the same token.
                    .with_pool(model.kv.shape());
            driver_metal_new::model::run::run(
                &self.context,
                &self.compiler,
                &mut self.pipelines,
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
        }

        let (_raw, completion) = self.broker.launch_completion(1);
        Ok(FrameLaunchOutcome::Launched(completion))
    }

    /// # Errors
    ///
    /// Always. Media encode is unsupported on this backend, as it is on CUDA;
    /// both seams refuse rather than pretending.
    pub fn encode(&mut self, _plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        bail!("driver-metal-new: media encode is unsupported on this backend")
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
            .ok_or_else(|| anyhow!("driver-metal-new: copy_kv before load_model"))?;
        let caps = driver_metal_new::store::Capabilities {
            has_linear_attn: self.has_linear_attn,
            kv_total_pages: pool.pages(),
            rs_slots: 0,
        };
        let work = driver_metal_new::store::plan_kv_copy(desc, caps, pool.shape().grid())
            .map_err(|why| anyhow!("metal copy_kv: {why:?}"))?;

        // Whole-page moves first, as page pairs; then the row cells. Both run
        // over every layer's K and V, because the pool is page-major at one
        // stride everywhere.
        let page_bytes = pool.shape().page_bytes();
        let mut cells = Vec::new();
        for &(src, dst) in &work.pages {
            cells.push(driver_metal_new::store::CellCopy {
                src_off: u64::from(src) * page_bytes,
                dst_off: u64::from(dst) * page_bytes,
                bytes: page_bytes,
            });
        }
        if !cells.is_empty() {
            pool.apply(&driver_metal_new::store::CellMovePlan {
                copies: cells,
                pages_touched: work.pages_touched,
            })
            .map_err(|e| anyhow!("metal copy_kv: {e:?}"))?;
        }
        if let Some(plan) = work.cells.as_ref() {
            pool.apply(plan).map_err(|e| anyhow!("metal copy_kv: {e:?}"))?;
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
}

/// The hole, named once so every verb that shares it reads the same.
///
/// The pool EXISTS now — `launch` admits against it and fires. What these
/// three still want is the MOVE: `store::control` decides what a copy or a
/// resize would do and `store::kv_move` plans the offsets, both portable and
/// both tested. What is missing is the encoder that runs the plan, and the
/// reallocation a resize implies for a pool that is a fixed allocation today.
const UNSERVED_MOVE: &str = "driver-metal-new: KV copy/resize is not wired to the seam yet. \
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

/// Which request owns each token, from the qo CSR.
///
/// The scheduler states the boundaries — request `r` owns rows
/// `[qo_indptr[r], qo_indptr[r+1])` — and `sdpa_paged_decode` wants the
/// inverse, one entry a token. Expanded here rather than asked of the
/// scheduler, because it is a restatement of what the CSR already says and a
/// second field would be a second chance to disagree with it.
fn req_of_token(qo_indptr: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for r in 0..qo_indptr.len().saturating_sub(1) {
        let (start, end) = (qo_indptr[r], qo_indptr[r + 1]);
        out.resize(out.len() + (end - start) as usize, r as u32);
    }
    out
}
