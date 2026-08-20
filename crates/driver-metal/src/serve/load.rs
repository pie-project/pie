//! Create, and the once-per-model work.
//!
//! One model per driver, the same shape `driver-cuda`'s `serve/state.rs` has
//! and the reason a frame's instance roster is one family's.

use crate::error::{Error, Result};
use crate::serve::state::{Shell, elastic_budget_bytes};

impl Shell {
    /// Identify the checkpoint, author its load plan, run it, and stage every
    /// tensor.
    ///
    /// The order is `driver-cuda`'s `load_impl` order: WHICH MODEL first, from
    /// the tensors, and everything after it a projection of the row that
    /// answered.
    ///
    /// # Errors
    ///
    /// More than one descriptor; a checkpoint no catalog row matches; a row
    /// whose architecture no Metal text states; a shape this build's kernels
    /// cannot be launched at; or a plan that will not stage.
    pub fn load_model(
        &mut self,
        descs: &[driver_api::ModelLoadDesc],
    ) -> Result<driver_api::DriverCapabilities> {
        let [desc] = descs else {
            return Err(Error::Unserved {
                what: "load_model",
                message: format!(
                    "this backend holds ONE model; got {} descriptors",
                    descs.len()
                ),
            });
        };
        // THE CHECKPOINT'S OWN `config.json`: embedded in the artifact, else
        // the boot TOML's path. ONE FIELD is read out of it — the declared
        // QUANTIZATION — because it is the one thing a catalog row genuinely
        // cannot state: the same model is published at four bits and at eight,
        // and a group size is not an extent of any tensor (g64/b8 and g128/b4
        // pack to identical shapes). Every other number is a row's.
        let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(&desc.snapshot_dir)
            .map_err(|e| Error::Unserved {
                what: "load_model",
                message: format!(
                    "{} did not read as a checkpoint: {e:?}",
                    desc.snapshot_dir.display()
                ),
            })?;
        let config_json = match model_loader::checkpoint::read::read_meta(
            &meta,
            model::encoding::CONFIG_OBJECT,
        ) {
            Ok(Some(bytes)) => String::from_utf8(bytes).map_err(|e| Error::Unserved {
                what: "load_model",
                message: format!(
                    "the embedded {} is not utf8: {e}",
                    model::encoding::CONFIG_OBJECT
                ),
            })?,
            Ok(None) => {
                let path = self.boot_config.as_ref().ok_or_else(|| Error::Unserved {
                    what: "load_model",
                    message: "no embedded model/config and no `[model] config` in the \
                              boot config. One field is read out of it — the declared \
                              quantization — and no metal kernel can be named without it"
                        .to_string(),
                })?;
                std::fs::read_to_string(path).map_err(|e| Error::Unserved {
                    what: "load_model",
                    message: format!("{}: {e}", path.display()),
                })?
            }
            Err(e) => {
                return Err(Error::Unserved {
                    what: "load_model",
                    message: format!("the artifact's metadata did not read: {e:?}"),
                });
            }
        };

        // WHICH MODEL THIS IS, asked of the TENSORS — unless pie wrote them.
        //
        // The config above no longer DECIDES anything. What this driver did
        // instead was read `architectures[0]` out of a descriptor, lowercase
        // it, strip its `ForCausalLM` tail, and use the result as a dispatch
        // key against a list — which is how `Qwen3MoeForCausalLM` and
        // `qwen3_moe` came to be two spellings of one architecture that two
        // gates answered differently, and how the load gate could report five
        // checkpoints healthy while the seam refused two of them.
        //
        // Identification and validation are the same operation here, and that
        // is the point: a config that lies about its geometry used to be
        // believed by the derivation and contradicted by an assertion several
        // frames later, if at all. A checkpoint is a known model or it is not.
        //
        // The exception is an artifact `pie model build` produced, whose
        // tensors are post-transform and match no manifest by construction; it
        // carries the row this same identification settled at build time. See
        // `catalog::identify_artifact`.
        let chosen = self
            .boot_model_id
            .as_ref()
            .map_or(model::catalog::Override::None, |id| {
                model::catalog::Override::Id(id.clone())
            });
        let attributes =
            model_loader::checkpoint::read::parse_checkpoint_attributes(&desc.snapshot_dir)
                .unwrap_or_default();
        let row = model::catalog::identify_artifact(&attributes, &meta, &chosen).map_err(|e| {
            Error::Unserved {
                what: "load_model",
                message: e.to_string(),
            }
        })?;
        let encoding = model::encoding::Encoding::from_config_json(&config_json).map_err(|e| {
            Error::Unserved {
                what: "load_model",
                message: format!("the checkpoint's config does not state its encoding: {e}"),
            }
        })?;

        // ONCE, at load, and never again. See `Shell::deployment`. A PROJECTION
        // of the matched row rather than a derivation from a parsed config.
        //
        // `Deployed::single()` because this driver serves one device: there is
        // no tensor-parallel split to state and no host scalars to hand over —
        // gemma-4's `layer_scalar` is the only row that takes any, and its two
        // attention shapes are refused by the geometry below.
        let deployment = row
            .deployment(model::catalog::Deployed::single())
            .map_err(Error::from)?;
        // Read off the projection BEFORE it is stored: both are `Copy` facts
        // the capability report publishes, and the shell keeps the projection.
        let arch = deployment.advertised.arch;
        let max_model_len = deployment.advertised.max_model_len;
        // Whether this build has a Metal text for the row, asked BEFORE a byte
        // is staged — and asked of the ROW.
        //
        // THE PLACEMENT IS DELIBERATE. On the 31B gemma, staging first means
        // 17 GB spent to reach a refusal identification had already settled.
        // Same rule as `weights::stage::fits_on_this_gpu`: asked before a byte
        // is read is the only moment it can be asked usefully.
        //
        // **The refusal must not depend on the binding facts**, and it does
        // not: `binding::serves` asks with `binding::ANY_ENCODING`, because a
        // row that refuses Metal refuses it for EVERY encoding. Whether a text
        // was written is a fact about this build's source, which is what lets
        // the question be asked here, where `moe_mxfp4` is not yet knowable.
        // `binding::a_row_is_served_the_same_way_at_every_encoding` holds the
        // whole catalog to that, so this placement stops being sound loudly.
        if let Err(refusal) = crate::model::binding::serves(row) {
            return Err(Error::Unserved {
                what: "load_model",
                message: format!(
                    "no Metal text for row `{}` (`{arch}`): {refusal}. The row exists \
                     and its author is written — what is missing is the forward pass, \
                     which `tests/catalog_coverage.rs` enumerates. Refused before \
                     staging: the answer is the row's, so reading the checkpoint could \
                     not have changed it.",
                    row.id()
                ),
            });
        }

        let loaded = crate::weights::load::load(&self.context, &desc.snapshot_dir, row, &encoding)?;
        self.id = row.id();
        self.has_linear_attn = deployment.recurrent.is_some();

        // The pool, at the geometry the checkpoint states. `PIE_METAL_KV_PAGES`
        // is the CEILING rather than the size: the pages are elastic, so this
        // is the count address space is reserved at and the most `resize_pool`
        // may grow back to. The pool starts committed to all of it.
        let pages: u32 = std::env::var("PIE_METAL_KV_PAGES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        // The Metal-side numbers, projected from the row's deployment.
        //
        // The affine point is passed separately because it is the CHECKPOINT's
        // and not the row's — and it is asked of the BYTES, not of
        // `config.json`, whose stated default per-tensor overrides may
        // supersede for every tensor in the file. `Loaded::affine_point` owns
        // the refusal when a checkpoint arrives at more than one, and
        // `geometry_from_deployment` refuses rather than defaulting.
        let quant = loaded.affine_point(row.id())?;
        let geometry = crate::batch::geometry_from_deployment(&deployment, row.load_shape(), quant)
            .map_err(Error::from)?;

        // The sandwich-norm/GELU pair is checked in
        // `model/tests/sandwich_norm_implies_gelu.rs`, where both halves are
        // stated and no checkpoint is needed, so every row is checked when it
        // is written rather than the ones a Metal load happens to reach. This
        // driver must not hold a second answer: `no_probe_decides_a_fact`
        // forbids the tensor question that guard asked.

        // THE ROW, and what this load OBSERVED that the row cannot state.
        //
        // No facts struct is rebuilt here from `has_tensor` probes. Deriving
        // qk-norm, fused QKV, attention bias, the router, the shared expert,
        // the sandwich norm, the attention sink, the per-layer scalar and the
        // norm variant from whether a name is in a safetensors index is how
        // the norm variant came to read `(1 + w)` for gemma-4 — a stack whose
        // gains are a plain multiplier — because it shipped the norm the probe
        // asked about. Every one of those is the row's answer.
        //
        // What is left is six values, none of them about the model: the affine
        // point the BYTES arrived in, whether the expert bank reached the
        // device still in MXFP4, and three capabilities of the kernels this
        // binary was BUILT with. `binding::observed`'s narrow signature is the
        // guarantee rather than a convenience — it cannot see the geometry, so
        // it cannot smuggle a model fact back in.
        //
        // `Loaded::mxfp4` decides an ENCODING, not a fact: a checkpoint need
        // not quantize uniformly, and reading an expert bank with the dense
        // format is NaNs rather than a near miss. MXFP4 banks take their own
        // kernel at their own group and are never read at an affine point, so
        // the ONE affine point `observed` takes from `geometry.quant` — the
        // point the kernels are built at and the scales were written at, one
        // value with one source — does not cover them.
        self.text_row = Some((
            row,
            crate::model::binding::observed(
                geometry.quant,
                |name| loaded.affine_point_of(name),
                |name| loaded.mxfp4.contains(name),
            ),
        ));
        self.inv_freq = crate::model::rope::table(&geometry)
            .iter()
            .map(|f| f.to_bits())
            .collect();
        // Which buffer each weight address belongs to, so a fire can be
        // RECORDED. A model reload moves every address, so the old recordings
        // are invalid — stated rather than left to the fingerprint.
        self.recordings.clear();
        // And the graphs, one step earlier: a lowering is the graph of the text
        // the OLD row named, and serving it over a new checkpoint's weights
        // would fire the previous architecture at whatever the new one staged.
        self.lowerings.clear();
        self.regions = crate::device::Regions::new();
        self.deployment = Some(deployment);
        for region in &loaded.regions {
            self.regions.add(region);
        }
        self.model = Some(loaded);
        let shape = crate::layout::kv::Shape {
            layers: geometry.n_layers,
            kv_heads: geometry.n_kv_heads,
            head_dim: geometry.head_dim,
            page_size: self.device_facts.page_size,
            pages,
            element_bytes: 2,
            // The FULL-attention layers' own shape, when the checkpoint states
            // a second one. Zero everywhere but gemma-4, and the pool reads
            // the zeros as "one shape for the whole stack". `full_attn_every`
            // is the rule the row's text derives `window_left` from, so pool
            // and text agree about which layers are full without a second list.
            global_head_dim: geometry.global_head_dim,
            global_kv_heads: geometry.global_kv_heads,
            full_attn_every: geometry.full_attn_every,
        };
        // The previous pool's memory goes back to the arena BEFORE the next one
        // asks for any. Elastic pages are charged against a budget, and holding
        // a whole model's KV while allocating a second one is how a reload gets
        // refused for memory that is about to be free.
        self.pool = None;
        // Elastic, so that `resize_pool` has something to resize: the pages sit
        // in placement heaps behind a sparse buffer, and giving memory back
        // moves no address a fire has already bound. Committed to full size
        // here, so an unresized pool behaves exactly as a fixed one.
        let pool = crate::pools::kv::Pool::allocate_elastic(
            &self.context,
            &mut self.stepper,
            &self.arena,
            shape,
        )?;
        // Every layer's K and V, for the same reason as the weights.
        for l in 0..shape.layers {
            if let Some(layer) = pool.layer(l) {
                layer.k.register(&mut self.regions);
                layer.v.register(&mut self.regions);
            }
        }
        self.pool = Some(pool);

        // The recurrent planes, for a hybrid that states any.
        //
        // Sized from the deployment's own `RecurrentShape` and NOT from its
        // `state_elem`: that number is the checkpoint's, and these planes are
        // the KERNEL's -- `gdn_core.metal` binds both as `device float*` with
        // no template parameter, so a build that believed the bf16 a CUDA
        // path uses would allocate half a plane and index past the end of it
        // on the first slot but zero. `layout::recurrent::ELEM_BYTES` is the
        // one place that says so.
        //
        // Slots are seats, not pages, and one is expensive -- 63 MB a seat on
        // qwen3.6-35B-A3B, 151 MB on the 27B -- so the count is a knob with a
        // small default rather than a fraction of memory. `PIE_METAL_RS_SLOTS`
        // is the ceiling and the allocation both: these are fixed, because
        // there is nothing to resize toward. A request holds its seat from
        // its first token to its last.
        self.recurrent = None;
        if let Some(rs) = self.deployment.as_ref().and_then(|d| d.recurrent.as_ref()) {
            let slots: u32 = std::env::var("PIE_METAL_RS_SLOTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8);
            let shape = crate::layout::recurrent::Shape {
                linear_layers: u32::try_from(rs.linear_layers.len()).unwrap_or(0),
                conv_dim: u32::try_from(rs.conv_dim).unwrap_or(0),
                conv_k: u32::try_from(rs.conv_k).unwrap_or(0),
                v_heads: u32::try_from(rs.v_h).unwrap_or(0),
                v_dim: u32::try_from(rs.v_d).unwrap_or(0),
                k_dim: u32::try_from(rs.k_d).unwrap_or(0),
                slots,
            };
            let pool = crate::pools::recurrent::Pool::allocate(&self.context, shape)?;
            pool.register(&mut self.regions);
            self.recurrent = Some(pool);
        }

        // What the checkpoint states, and what the pool states.
        //
        // `total_pages` is the pool's own count now, so a scheduler admits
        // against what was actually allocated. It read zero while no pool
        // existed, which was the truth then and the reason nothing was
        // admitted.
        Ok(driver_api::DriverCapabilities {
            abi_version: driver_api::PIE_DRIVER_ABI_VERSION,
            total_pages: pages,
            kv_page_size: self.device_facts.page_size,
            swap_pool_size: 0,
            kv_copy_domain_mask: 0,
            rs_cache_required: self.has_linear_attn,
            // What was allocated, not what was wished for. Zero for every
            // pure-attention checkpoint, and a scheduler reads the pair
            // together: slots with no bytes describes seats of no size.
            rs_cache_slots: self.recurrent.as_ref().map_or(0, |p| p.shape().slots),
            rs_cache_slot_bytes: self
                .recurrent
                .as_ref()
                .map_or(0, |p| p.shape().bytes_per_slot()),
            // What `resize_pool` can actually move, and the unit it moves it
            // in: the KV pool's pages are sparse, so memory can be given back
            // and taken again without any address moving.
            //
            // Both are non-zero together or not at all — `bootstrap` starts its
            // trim task only when both are.
            elastic_page_bytes: crate::device::PAGE,
            elastic_budget_pages: crate::device::pages_for_bytes(elastic_budget_bytes(
                &self.context,
            )),
            has_mtp_logits: false,
            has_mtp_drafts: false,
            has_value_head: false,
            // Every one of these is a SINK this backend cannot honour, and the
            // `kernel!` rows say so. Advertising one would make a program bind
            // and then run as a silent no-op.
            has_kv_envelopes: false,
            has_attn_score: false,
            has_attn_page_mask: false,
            has_lora: false,
            model_site_summary: driver_api::ModelSiteSummary::default(),
            // The DECODE ENVELOPE's ports, claimed because `envelope::fill`
            // resolves exactly these three: `EMBED_TOKENS` and `POSITIONS`
            // are read off `driver::resolve`, the same backend-neutral copier
            // wgpu and vulkan read theirs with, and `KV_LEN` is derived from
            // the positions and CHECKED against the stated value rather than
            // ignored.
            //
            // This read 0 for as long as that machinery was missing, and the
            // 0 was honest then -- the engine answered it by folding the
            // geometry on the host, which cannot know `EmbedTokens` and said
            // so by name:
            //
            //     decode envelope on a driver without device geometry ports
            //     (mask 0x0, needs 0x25): falling back to host-evaluated
            //     serialized execution
            //     ... EmbedTokens is not host-derivable: channel 0 has no
            //     host-known value
            //
            // so the fix was to build the machinery, not to widen the claim.
            //
            // And it is not widened FURTHER, deliberately.
            // `PIE_DEVICE_GEOMETRY_PORTS` names four more -- the pages, the
            // CSR and the two halves of the write descriptor -- and
            // `PIE_DEVICE_PORT_ATTN_MASK` a fifth. This driver has no
            // consumer for the last three: `serve::launch` derives every
            // row's write target from its position, and a custom attention
            // mask reaches the Metal text through the region table's `MASK`
            // bit rather than through `LaunchPlan::masks`. A claim is a
            // promise to READ, so `envelope::fill` refuses that class by name
            // instead, and the engine keeps sending it down a path that knows
            // it is not served here.
            device_geometry_port_mask: driver_api::PIE_DECODE_ENVELOPE_PORTS,
            // TRUE, and it is a FACT about how `launch` drives a frame rather
            // than a preference: a frame with any device-resolved member is
            // driven a step at a time -- fill, encode, wait, run the step's
            // programs -- before the next step is touched, because a decode
            // envelope's tokens are the cells the step before it PUT. So a
            // slot chained behind an earlier slot of the same frame reads a
            // cell that slot's program has already written, which is what
            // `pipeline::fire` reads this flag to decide.
            //
            // A frame of ordinary host-wire steps is still committed whole
            // and waited for afterwards, which is where this backend's
            // run-ahead comes from; the flag describes the harder case
            // because that is the one the engine is asking about.
            resolves_geometry_per_step: true,
            // The ceilings a scheduler batches under. Stated rather than
            // unbounded: a fire wider than this has no arena sized for it.
            max_forward_tokens: 4096,
            max_forward_requests: 256,
            max_page_refs: pages,
            // The three answers that are facts about the MODEL rather than the
            // device, and all three are the ROW's now — they were the last
            // thing keeping a parsed config resident for the life of a load.
            //
            // `arch_name` is a FAMILY label a guest program matches on and is
            // deliberately coarser than the id beside it. It is not a dispatch
            // key: nothing in this crate branches on it.
            arch_name: arch.to_string(),
            // WHICH ROW — the answer an operator and a boundary both want, and
            // which this driver could not give at all while what it had was a
            // `model_type` read off a config rather than an identity.
            model_id: self.id.to_string(),
            vocab_size: geometry.vocab,
            max_model_len,
            activation_dtype: "bf16".to_string(),
            hidden_size: geometry.hidden,
            // FALSE regardless of what the row ships, because this is a fact
            // about the BACKEND: there is no encode entry point here, so a row
            // with a vision or audio tower is served as its text half or
            // refused. `deployment.advertised.media_encode` is the row's own
            // answer, to read when that entry point exists.
            supports_media_encode: false,
            snapshot_dir: desc.snapshot_dir.display().to_string(),
            kv_handle: None,
            // Metal compiles its shaders at run time from the tree; nothing
            // upstream needs to generate a kernel for it.
            codegen_backend: String::new(),
        })
    }
}
