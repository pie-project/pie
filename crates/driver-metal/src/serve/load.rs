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
        // Whether this build has a Metal text for the row, asked BEFORE a
        // byte is staged — and asked of the ROW.
        //
        // THE PLACEMENT IS PRESERVED DELIBERATELY. The old order's own
        // message admitted what it costs to get this wrong: "the checkpoint
        // loaded, but nothing states its forward pass." On the 31B gemma that
        // sentence is 17 GB of staging spent to reach a refusal identification
        // had already settled. Same rule as `weights::stage::fits_on_this_gpu`,
        // whose doc states it: asked before a byte is read is the only moment
        // it can be asked usefully. The two refusals sit on the same side of
        // the load, and this one must stay on this side of it.
        //
        // WHAT CHANGED IS WHO ANSWERS. This was `text::serves(arch)` — a
        // membership test against an eleven-entry table of architecture
        // strings, which is the third dispatch key for an identity
        // `catalog::identify` had already settled from the tensors. Two keys
        // for one identity gave two answers: the table listed `"gemma4"`, so
        // this gate CLAIMED every gemma-4 and a second refusal ninety lines
        // below then rejected it on the sandwich norm — after the staging
        // this gate exists to avoid. The row answers now, so there is one
        // list and it is the one that traces.
        //
        // **The refusal must not depend on the binding facts**, and it does
        // not: `binding::serves` asks with `binding::ANY_ENCODING`, because a
        // row that refuses Metal refuses it for EVERY encoding. Whether a
        // text was written is a fact about this build's source, and no group
        // size, bit width or expert-bank format can move it — which is what
        // lets the question be asked here, where `moe_mxfp4` is not yet
        // knowable because the tensors are not yet on the device.
        // `binding::a_row_is_served_the_same_way_at_every_encoding` holds the
        // whole catalog to that, so this placement stops being sound the
        // moment it fails rather than silently.
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

        // The sandwich-norm/GELU pair used to be checked here, by asking
        // the staged tensors whether `pre_feedforward_layernorm` had
        // arrived and comparing that to the row's gate. It is now
        // `model/tests/sandwich_norm_implies_gelu.rs`, where both halves
        // are already stated and no checkpoint is needed: the manifest
        // declares the norm, the deployment states the gate, and every
        // row is checked when it is written rather than the ones a Metal
        // load happens to reach.
        //
        // The measurement that motivated it is recorded there. What does
        // not survive the move is the idea that this driver should hold a
        // second answer -- `no_probe_decides_a_fact` forbids exactly the
        // tensor question that guard asked, and it is right to: the norm
        // variant came to read `(1 + w)` for gemma-4 because a probe of
        // this shape asked whether it shipped the norm.

        // THE ROW, and what this load OBSERVED that the row cannot state.
        //
        // What stood here called `text::facts_from_with`, which rebuilt
        // twenty-nine `LlamaLikeFacts` out of the projected geometry and NINE
        // `has_tensor` probes: qk-norm, fused QKV, attention bias, the
        // router, the shared expert, the sandwich norm, the attention sink,
        // the per-layer scalar, and the norm variant. Every one of those is
        // stated by the row, was stated by the row while this code ran, and
        // was re-derived here anyway from whether a name happened to be in a
        // safetensors index. That is how the norm variant came to read
        // `(1 + w)` for gemma-4 — a stack whose gains are a plain
        // multiplier — because it shipped the norm the probe asked about.
        //
        // What is left is six values, and not one of them is about the model:
        // the affine point the BYTES arrived in, whether the expert bank
        // reached the device still in MXFP4, and three capabilities of the
        // kernels this binary was BUILT with. `binding::observed` takes the
        // affine format and one tensor question, and that narrow signature is
        // the guarantee rather than a convenience — it cannot see the
        // geometry, so it cannot smuggle a model fact back in.
        //
        // The remaining probe is `Loaded::mxfp4`: which names the load plan
        // left in the checkpoint's own format. It decides an ENCODING, not a
        // fact — a checkpoint need not quantize uniformly, and reading an
        // expert bank with the dense format is NaNs rather than a near miss.
        // ONE KERNEL SET, SO ONE AFFINE POINT — asked rather than assumed.
        //
        // `observed` below builds a single kernel set from `geometry.quant`,
        // which is the point the checkpoint's `config.json` states. The
        // TENSORS need not agree with it and need not agree with each other:
        // `mlx_lm` publishes a routed stack at 4 bits and its router gate at
        // 8, because the gate is small and the whole mixture inherits its
        // error. Every tensor is then dequantised at the stated width, the
        // gate included, and the failure is not a fault — it is every token
        // routed to almost the right experts, measured at cosine 0.84
        // against the reference logits.
        //
        // `DecodeGeometry::alt_quant` is the field that second point would
        // ride if this driver could instantiate two kernel sets. It cannot,
        // and inventing one here would put a guess where a fact belongs. So
        // the honest answer is the refusal, and what makes it possible is
        // that the LOAD PLAN knew all along: `QuantSpec` carries a
        // `group_size` and a `bits_per_element` per tensor, and nothing was
        // asking.
        //
        // MXFP4 banks are not in this set. They take their own kernel at
        // their own group and are never read at an affine point, which is
        // exactly the case `Loaded::mxfp4` already carries.
        if loaded.affine_points.len() > 1 {
            let points = loaded
                .affine_points
                .iter()
                .map(|(g, b)| format!("g{g}/b{b}"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(Error::Create {
                what: "checkpoint",
                message: format!(
                    "`{}` arrives at {} affine points ({points}) and this driver \
                     instantiates ONE kernel set, at the g{}/b{} its config \
                     states. Every tensor at another point would be dequantised \
                     at that width — scales read from the wrong offset, and for \
                     a router gate that is not a fault but a mixture routing to \
                     almost the right experts. Refused rather than served wrongly",
                    row.id(),
                    loaded.affine_points.len(),
                    geometry.quant.group,
                    geometry.quant.bits
                ),
            });
        }

        self.text_row = Some((
            row,
            crate::model::binding::observed(geometry.quant, |name| loaded.mxfp4.contains(name)),
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
            // The FULL-attention layers' own shape, when the checkpoint
            // states a second one. Zero everywhere but gemma-4, and the pool
            // reads the zeros as "one shape for the whole stack".
            //
            // `full_attn_every` is the same rule the row's text derives
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
            elastic_budget_pages: crate::device::pages_for_bytes(elastic_budget_bytes(
                &self.context,
            )),
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
            // not a dispatch key: nothing in this crate branches on it at
            // all any more, now that the text is the row's answer rather
            // than a table lookup on this string.
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
