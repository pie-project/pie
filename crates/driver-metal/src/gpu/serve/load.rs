//! Create, and the once-per-model work.
//!
//! One model per driver, which is the same shape `driver-cuda`'s
//! `serve/state.rs` has and the reason a frame's instance roster is one
//! family's.

use crate::error::{Error, Result};
use crate::gpu::serve::state::{Shell, elastic_budget_bytes};

impl Shell {
    /// Author the checkpoint's load plan, run it, and stage every tensor.
    ///
    /// # Errors
    ///
    /// More than one descriptor, a missing `[model] descriptor`, an
    /// architecture no Metal text states, or a plan that will not stage.
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
        // The load plan is authored from the `pie.model/1` DESCRIPTOR, and
        // this seam does not make one. `model::config` normalizes a snapshot
        // exactly once, upstream, and `crates/model/tests/one_normalizer.rs`
        // refuses to let the runtime read a checkpoint's own config a second
        // time — two normalizers is how they come to disagree.
        let path = self.boot_descriptor.as_ref().ok_or_else(|| Error::Unserved {
            what: "load_model",
            message: "no `[model] descriptor` in the boot config. Model facts come \
                      from the descriptor the worker hands over, not from the \
                      checkpoint — see crates/model/tests/one_normalizer.rs."
                .to_string(),
        })?;
        let descriptor = std::fs::read_to_string(path).map_err(|e| Error::Unserved {
            what: "load_model",
            message: format!("{}: {e}", path.display()),
        })?;
        let loaded = crate::gpu::weights::load::load(&self.context, &desc.snapshot_dir, &descriptor)?;
        let facts = crate::facts::ModelFacts::from_descriptor(&descriptor).ok_or_else(|| {
            Error::Unserved {
                what: "load_model",
                message: "the descriptor does not parse as model facts".to_string(),
            }
        })?;
        self.arch = facts.arch_name.clone();
        self.has_linear_attn = facts.has_linear_attn;
        if !crate::model::text::serves(&self.arch) {
            return Err(Error::Unserved {
                what: "load_model",
                message: format!(
                    "no Metal text for `{}`; this backend serves {:?}. The checkpoint \
                     loaded, but nothing states its forward pass.",
                    self.arch,
                    crate::model::text::known()
                ),
            });
        }

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
        let geometry = crate::batch::geometry_from_facts(&facts).map_err(|why| Error::Unserved {
            what: "load_model",
            message: format!("the descriptor does not describe a servable family: {why:?}"),
        })?;

        // The deployment's facts, from the geometry the descriptor states and
        // the tensors the checkpoint actually shipped. The three binding facts
        // — qk-norm, fused QKV, attention bias — ask the TENSORS, because a
        // config states an architecture and a tensor states a binding.
        // Two probes: which tensors the checkpoint shipped, and which of them
        // the load left in MXFP4. The second is what a MIXTURE needs -- a
        // checkpoint need not quantize uniformly, and reading an expert bank
        // with the dense format is NaNs rather than a near miss.
        self.deployment = Some(crate::model::text::facts_from_with(
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
        self.regions = crate::gpu::Regions::new();
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
        let pool = crate::gpu::pools::kv::Pool::allocate_elastic(
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
            rs_cache_required: facts.has_linear_attn,
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
            elastic_page_bytes: crate::gpu::PAGE,
            elastic_budget_pages: crate::gpu::pages_for_bytes(elastic_budget_bytes(&self.context)),
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
}
