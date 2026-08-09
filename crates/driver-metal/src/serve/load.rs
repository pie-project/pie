//! Create, and the once-per-model work.
//!
//! One model per driver, which is the same shape `driver-cuda`'s
//! `serve/state.rs` has and the reason a frame's instance roster is one
//! family's.

use crate::error::{Error, Result};
use crate::serve::state::{Shell, elastic_budget_bytes};

impl Shell {
    /// Identify the checkpoint, author its load plan, run it, and stage
    /// every tensor.
    ///
    /// The order matters and it is the order `driver-cuda`'s `load_impl`
    /// uses: WHICH MODEL comes first, from the tensors, and everything after
    /// it is a projection of the row that answered. What this replaced asked
    /// the questions the other way round — it read a document, believed it,
    /// staged the weights, and found out at the first fire whether the two
    /// agreed.
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
        // the boot TOML's path.
        //
        // ONE FIELD IS READ OUT OF IT, and the shrinkage is the refactor.
        // This used to be a `pie.model/1` descriptor — ~40 numbers a
        // 845-line normalizer had projected out of a 136-field schema — that
        // this driver parsed back into a private facts struct and then read
        // the model out of: the architecture, the head counts, the rope base,
        // the expert widths, the KV sharing. Every one of those numbers is a
        // catalog row's now.
        //
        // What is left is the declared QUANTIZATION, and it stays because it
        // is the one thing a row genuinely cannot state: the same model is
        // published at four bits and at eight, `mlx-community` ships both,
        // and a group size is not an extent of any tensor — g64/b8 and
        // g128/b4 pack to identical shapes.
        let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(&desc.snapshot_dir)
            .map_err(|e| Error::Unserved {
                what: "load_model",
                message: format!("{} did not read as a checkpoint: {e:?}", desc.snapshot_dir.display()),
            })?;
        let config_json = match model_loader::checkpoint::read::read_meta(
            &meta,
            model::encoding::CONFIG_OBJECT,
        ) {
            Ok(Some(bytes)) => String::from_utf8(bytes).map_err(|e| Error::Unserved {
                what: "load_model",
                message: format!("the embedded {} is not utf8: {e}", model::encoding::CONFIG_OBJECT),
            })?,
            Ok(None) => {
                let path = self.boot_descriptor.as_ref().ok_or_else(|| Error::Unserved {
                    what: "load_model",
                    message: "no embedded model/config and no `[model] descriptor` in the \
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

        // WHICH MODEL THIS IS, asked of the TENSORS.
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
        let chosen = self
            .boot_model_id
            .as_ref()
            .map_or(model::catalog::Override::None, |id| {
                model::catalog::Override::Id(id.clone())
            });
        let row = model::catalog::identify(&meta, &chosen).map_err(|e| Error::Unserved {
            what: "load_model",
            message: e.to_string(),
        })?;
        let encoding = model::encoding::Encoding::from_config_json(&config_json).map_err(|e| {
            Error::Unserved {
                what: "load_model",
                message: format!("the checkpoint's config does not state its encoding: {e}"),
            }
        })?;

        // ONCE, at load, and never again. See `Shell::deployment`.
        //
        // A PROJECTION of the matched row rather than a derivation from a
        // parsed config. The eleven `*_facts_from_hf` functions and the four
        // family-prefixed blocks this replaces read the same numbers out of
        // the same checkpoint, one family at a time, keyed on a `model_type`
        // string that a second table keyed differently.
        //
        // `Deployed::single()` because this driver serves one device: there
        // is no tensor-parallel split to state and no host scalars to hand
        // over — gemma-4's `layer_scalar` is the only row that takes any, and
        // its two attention shapes are refused by the geometry below.
        let deployment = row
            .deployment(model::catalog::Deployed::single())
            .map_err(Error::from)?;
        // Read off the projection BEFORE it is stored, because both are
        // `Copy` facts about the model that the capability report publishes
        // and the shell keeps the projection itself.
        let arch = deployment.advertised.arch;
        let max_model_len = deployment.advertised.max_model_len;
        if !crate::model::text::serves(arch) {
            return Err(Error::Unserved {
                what: "load_model",
                message: format!(
                    "no Metal text for `{arch}` (row `{}`); this backend serves {:?}. The \
                     row exists and its author is written — what is missing is the forward \
                     pass, which is `tests/catalog_coverage.rs`'s list.",
                    row.id(),
                    crate::model::text::known()
                ),
            });
        }

        let loaded = crate::weights::load::load(&self.context, &desc.snapshot_dir, row, &encoding)?;
        self.id = row.id();
        self.has_linear_attn = deployment.recurrent.is_some();

        // The pool, at the geometry the checkpoint states. `PIE_METAL_KV_PAGES`
        // is the size knob, and it is the CEILING rather than the size: the
        // pages are elastic, so this is the count address space is reserved
        // at and the most `resize_pool` may grow back to. The pool starts
        // committed to all of it, so a deployment that never resizes sees the
        // number it asked for.
        let pages: u32 = std::env::var("PIE_METAL_KV_PAGES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        // The Metal-side numbers, projected from the row's deployment.
        //
        // What this is NOT any more: a ladder that asked which of four
        // family-prefixed blocks of a private facts struct had been filled
        // and merged them back into one shape. `go_*` was gpt-oss's alone —
        // its own doc said a non-zero layer count marked "this config was
        // read as gpt-oss" — so reading it for a llama checkpoint allocated a
        // pool of no layers.
        //
        // The affine point is passed separately because it is the
        // CHECKPOINT's and not the row's; everything else comes off the
        // projection. `geometry_from_deployment` refuses rather than
        // defaulting, which is what makes this arithmetic over a value rather
        // than a second model definition.
        let quant = crate::batch::AffineFormat {
            bits: encoding.bits,
            group: encoding.group_size,
        };
        let geometry = crate::batch::geometry_from_deployment(&deployment, row.load_shape(), quant)
            .map_err(Error::from)?;

        // TWO STATEMENTS FROM ONE CHECKPOINT THAT CANNOT BOTH BE TRUE.
        //
        // A stack that norms both ways round each sub-layer — a sandwich norm
        // — states its MLP gate as a GELU. `DecodeGeometry::gelu_gate` says
        // SiLU, and it says SiLU for every checkpoint that reaches this line,
        // because a `Deployment` STATES NO ACTIVATION AT ALL: the row's own
        // forward text names it, and a driver receives the shape rather than
        // the text. So the tensors are asked instead, and the answer they
        // give here contradicts the only answer this driver can project.
        //
        // Serving it anyway is a 2%-at-the-origin error that diverges from
        // there, produces finite plausible tokens and never faults — which is
        // exactly how every gemma checkpoint ran as a llama for as long as
        // the gate was inferred from a family flag. Refused, and the refusal
        // names what would lift it.
        if loaded
            .tensors
            .contains_key("layers.0.pre_feedforward_layernorm.weight")
            && !geometry.gelu_gate
        {
            return Err(Error::Unserved {
                what: "load_model",
                message: format!(
                    "`{}` ships a sandwich norm, whose MLP gate is a GELU, and this \
                     driver can only project a SiLU gate: `model::deployment::Deployment` \
                     states no activation, so the row's own text is the only thing that \
                     knows. Lifting this needs either an activation on `Deployment` or a \
                     `Variant::trace` that can be asked for a Metal text",
                    row.id()
                ),
            });
        }

        // The Metal text's facts, from the geometry the row projected and the
        // tensors the checkpoint actually shipped. The three binding facts —
        // qk-norm, fused QKV, attention bias — ask the TENSORS, because a row
        // states an architecture and a tensor states a binding.
        //
        // Two probes: which tensors the checkpoint shipped, and which of them
        // the load left in MXFP4. The second is what a MIXTURE needs -- a
        // checkpoint need not quantize uniformly, and reading an expert bank
        // with the dense format is NaNs rather than a near miss.
        self.text_facts = Some(crate::model::text::facts_from_with(
            &geometry,
            |name| loaded.tensors.contains_key(name),
            |name| loaded.mxfp4.contains(name),
        ));
        self.inv_freq = crate::model::rope::frequencies(
            geometry.head_dim,
            geometry.rope_theta,
            (geometry.rope_freq_factor > 0.0).then_some(crate::model::rope::Rescale {
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
        self.regions = crate::device::Regions::new();
        self.deployment = Some(deployment);
        self.regions.add(&loaded.region);
        self.model = Some(loaded);
        let shape = crate::layout::kv::Shape {
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
        // The previous pool's memory goes back to the arena BEFORE the next
        // one asks for any. Elastic pages are charged against a budget, and
        // holding a whole model's KV while allocating a second one is how a
        // reload gets refused for memory that is about to be free -- a
        // failure that would not happen on a fixed pool, and so would look
        // like elastic storage causing it.
        self.pool = None;
        // Elastic, so that `resize_pool` has something to resize: the pages
        // sit in placement heaps behind a sparse buffer, and giving memory
        // back does not move a single address a fire has already bound.
        // Committed to its full size here, so a pool that has never been
        // resized behaves exactly as a fixed one -- which is what
        // `device_real_weights.rs` compares it against, bit for bit.
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
            rs_cache_slots: 0,
            rs_cache_slot_bytes: 0,
            // What `resize_pool` can actually move, and the unit it moves it
            // in. The KV pool's pages are sparse -- `Pool::allocate_elastic`
            // -- so memory can be given back and taken again without any
            // address moving, which is the whole reason a scheduler is
            // allowed to ask.
            //
            // Both are non-zero together or not at all: `bootstrap` starts
            // its trim task only when both are, and a page size with no
            // budget describes a pool that can be measured but not resized.
            elastic_page_bytes: crate::device::PAGE,
            elastic_budget_pages: crate::device::pages_for_bytes(elastic_budget_bytes(&self.context)),
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
            model_site_summary: driver_api::ModelSiteSummary::default(),
            device_geometry_port_mask: 0,
            // The ceilings a scheduler batches under. Stated rather than
            // unbounded: a fire wider than this has no arena sized for it.
            max_forward_tokens: 4096,
            max_forward_requests: 256,
            max_page_refs: pages,
            // The three answers that are facts about the MODEL rather than
            // about the device, and all three are the ROW's now. They were
            // the last thing keeping a parsed config resident: a driver held
            // a whole normalized `config.json` for the life of a load in
            // order to answer `arch_name`, `max_model_len`, and whether a
            // media tower was present.
            //
            // `arch_name` is a FAMILY label a guest program matches on —
            // `engine`'s `model.arch_name()` is a host function inferlets
            // call — and deliberately coarser than the id beside it. It is
            // not a dispatch key: nothing in this crate branches on it except
            // `model::text`, which LOOKS UP a text rather than choosing one.
            arch_name: arch.to_string(),
            // WHICH ROW, which is the answer an operator and a boundary both
            // want and which this driver could not give at all: it published
            // an empty string, because the thing it had was a `model_type`
            // read off a config and not an identity.
            model_id: self.id.to_string(),
            vocab_size: geometry.vocab,
            max_model_len,
            activation_dtype: "bf16".to_string(),
            hidden_size: geometry.hidden,
            // FALSE regardless of what the row ships, because this is a
            // fact about the BACKEND: there is no encode entry point here at
            // all, so a row with a vision or audio tower is served as its
            // text half or refused, never encoded. `deployment.advertised
            // .media_encode` is the row's own answer and the one to read when
            // that entry point exists; advertising it before then would make
            // a program bind an encode it would never get.
            supports_media_encode: false,
            snapshot_dir: desc.snapshot_dir.display().to_string(),
            kv_handle: None,
            // Metal compiles its shaders at run time from the tree; nothing
            // upstream needs to generate a kernel for it.
            codegen_backend: String::new(),
        })
    }
}
