//! Create, and the once-per-model work.
//!
//! One model per driver, the same shape `driver-cuda`'s `serve/load.rs` has
//! and the reason a frame's instance roster is one family's.
//!
//! # The order is the cuda sibling's, and every step of it moved
//!
//! `driver-cuda/src/serve/load.rs::load_impl` is the spec: WHICH SKU first,
//! asked of the checkpoint's own tensors; then the lane, traced for this
//! plane and bound; then the deployment, read off the SAME plan; then the
//! weights, produced from the SAME import table identification matched
//! against; then the pools, sized from the deployment.
//!
//! What that replaced here was five separate readings of one checkpoint: a
//! `config.json` parsed for its quantization, `catalog::identify_artifact`
//! matching tensor names against a hand-written manifest, an `Encoding`
//! projected from the config, a `MetalBinding` measured off the staged bytes
//! and handed back to the catalog to choose a text with, and a `LoadPlan`
//! authored from the row and the encoding together. Every one of them could
//! disagree with the others and two of them did.
//!
//! # A REFUSED LOAD AND NOT A WARNING
//!
//! Serving a checkpoint whose lane refuses would mean accepting `load_model`
//! and then failing every fire — the engine would have a model registered,
//! capabilities published and a scheduler admitting requests against a driver
//! that cannot answer one. So the eager resolve pass runs here
//! (`baker::Baked::unresolved`) and its whole report is the refusal's message.

use crate::baker::{Baked, Metal};
use crate::error::{Error, Result};
use crate::serve::state::{Shell, elastic_budget_bytes};
use crate::serve::weights::Weights;

impl Shell {
    /// Identify the checkpoint, trace its lane for this plane, produce its
    /// weights, and allocate the pools its deployment states.
    ///
    /// # Errors
    ///
    /// More than one descriptor; a snapshot no catalog row matches; a row
    /// whose lanes this plane cannot bind; a checkpoint this reader cannot
    /// produce from; or a pool the device declines.
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
        let unserved = |message: String| Error::Unserved {
            what: "load_model",
            message,
        };

        // ── 1. WHICH SKU THIS IS, asked of the checkpoint's own tensors. ──
        //
        // The same list `produce` is about to read, so identification and
        // loadability are one question — which is what replaced a
        // `config.json` deciding an architecture. `[model] id` outranks it,
        // which is how a row is proven before its checkpoint is one the
        // reader can tell apart.
        //
        // `Snapshot::at` AND NOT `Snapshot::open`: `open` resolves a cache-dir
        // NAME under `$HOME/.cache/huggingface/hub`, and a driver is handed
        // the snapshot directory itself.
        let sku = match self.boot_model_id.as_deref() {
            Some(id) => model::serve::row(id).map(|r| r.id).ok_or_else(|| {
                unserved(format!(
                    "`{id}` is not a row of `model::catalog()`; did you mean {}?",
                    model::serve::nearest_ids(id, 3).join(", "),
                ))
            })?,
            None => {
                let snap =
                    model::snapshot::Snapshot::at(desc.snapshot_dir.clone()).ok_or_else(|| {
                        unserved(format!(
                            "no safetensors snapshot at {}",
                            desc.snapshot_dir.display()
                        ))
                    })?;
                model::identify(&|name| snap.shape_of(name))
                    .map_err(|why| unserved(why.to_string()))?
            }
        };

        // ── 2. THE LANE, and the deployment read off the same plan. ───────
        //
        // Not beside anything — this IS the fire path. `Deployment::of` reads
        // the KV row, the recurrent slabs and every advertised width off the
        // trace the programs are built from, so a pool and the program that
        // indexes it cannot describe different models. That is what R3 bought,
        // and it is why cuda's `baker::Geometry::agrees_with` is gone.
        let baked = Baked::of::<Metal>(sku).map_err(unserved)?;
        let unresolved = baked.unresolved::<Metal>();
        if !unresolved.is_empty() {
            return Err(unserved(format!(
                "`{sku}` states {} point(s) this plane does not answer, so no fire \
                 could be served: {}",
                unresolved.len(),
                unresolved
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; "),
            )));
        }
        // `|l| l.is_ok()` AND NOT `Result::is_ok`: `Result` in this module is
        // `crate::error::Result`, whose error is fixed to `Error`, so the path
        // form names a function over a different type than the one `lanes`
        // holds (`model_compiler::program::Refusal`).
        if !baked.lanes.iter().any(|l| l.is_ok()) {
            return Err(unserved(format!(
                "`{sku}` binds no lane on this plane: {}",
                baked
                    .lanes
                    .iter()
                    .filter_map(|l| l.as_ref().err())
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" "),
            )));
        }
        let arch = baked.deployment.advertised.arch;
        let max_model_len = baked.deployment.advertised.max_model_len;
        let vocab = baked.deployment.shape.vocab;
        let hidden = baked.deployment.shape.hidden;

        // ── 3. THE WEIGHTS, through `produce` and not a load plan. ────────
        //
        // WHICH RANK, and this plane has exactly one honest answer.
        // `driver-cuda` reads `[driver] tp_rank` out of the boot TOML in its
        // `create_impl` and threads it to `baker::load`; there is no such key
        // on this path — `Shell::open` is handed `[model] config` and
        // `[model] id` and nothing else — and `kernels_metal::dist` states the
        // other half: this plane claims no `dist.all_reduce`, so there is no
        // collective for a world of more than one to sum through.
        //
        // So the rank is ZERO because the world is ONE, and a row that says
        // otherwise is REFUSED BY NAME rather than loaded at a rank nobody
        // chose. A `-tp2` row's `Param::shape` is a rank's share; loading it
        // whole would leave every sharded row of the join below reading
        // MISMATCH, which is the failure `model::produce`'s own note says a
        // driver-side cut was invented to avoid.
        //
        // The refusal reads the PLAN and not the SKU's spelling, which is the
        // rule `model::is_one_rank_of_a_world` states: `-tp2` is a name and
        // `dist.all_reduce` is the statement. `Baked::unresolved` above would
        // refuse the same row a step earlier, with the unanswered point named;
        // this says the same thing in the load's own vocabulary, so a rank
        // source arriving here has one place to change.
        const RANK: u32 = 0;
        if let Some(op) = baked
            .plan
            .ops
            .iter()
            .position(|op| op.kernel == "dist.all_reduce")
        {
            return Err(unserved(format!(
                "`{sku}` is ONE RANK of a tensor-parallel world — op {} states \
                 `dist.all_reduce`, so it holds a share of a weight some peer \
                 holds the rest of. This driver has no rank source (`[driver] \
                 tp_rank` is a cuda boot key and `Shell::open` reads none) and \
                 this plane claims no collective, so serving it would mean \
                 picking a rank on its behalf. Load the whole-model row \
                 instead.",
                op,
            )));
        }
        let weights = Weights::produce(&self.context, sku, &desc.snapshot_dir, &baked.plan, RANK)?;

        self.id = sku;
        self.has_linear_attn = baked.deployment.recurrent.is_some();
        // THE ROTARY LADDER IS NOT DERIVED HERE ANY MORE. A `Deployment` states
        // no `rope_theta`: a rotation is a statement of the trace
        // (`kernels::rope::{full,partial,partial_q,partial_last}`), its base
        // rides on that statement, and a rescaled ladder is a `Const` bank the
        // text names — which `baker::stage`'s own note gives as the reason
        // `FireTable::RopeFrequencies` left the table. `crate::model::rope`
        // is what a driver-side derivation would go through and nothing calls
        // it; see `tests/every_public_function_has_a_reader.rs`.

        // A model reload moves every address, so the old recordings are
        // invalid — stated rather than left to the fingerprint.
        self.recordings.clear();
        self.regions = crate::device::Regions::new();
        self.regions.add(weights.arena());

        // ── 4. THE POOLS, sized from the deployment. ──────────────────────
        //
        // `PIE_METAL_KV_PAGES` is the CEILING rather than the size: the pages
        // are elastic, so this is the count address space is reserved at and
        // the most `resize_pool` may grow back to.
        let pages: u32 = std::env::var("PIE_METAL_KV_PAGES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        // THE PER-LAYER WIDTHS ARE THE DEPLOYMENT'S, and G4 is why they have
        // to be: gemma-4's sliding layers attend 16 kv heads at 256 and its
        // global ones 4 at 512, in one tower. `Deployment::attention` is one
        // row per layer, read off the cache rows the statements NAME, so the
        // pool is laid out from the same reading the text fires against.
        //
        // `Shape::periodic` is the narrowing, and it REFUSES what it cannot
        // narrow. This read `Geometry::{global_head_dim, global_kv_heads,
        // full_attn_every}` — three fields G4 deleted, because what a driver
        // got from them was one number repeated `layers` times.
        let shape = crate::layout::kv::Shape::periodic(
            &baked.deployment.attention,
            self.device_facts.page_size,
            pages,
            2,
        )
        .map_err(|why| unserved(format!("`{sku}`'s KV pool cannot be laid out: {why}")))?;
        // The previous pool's memory goes back to the arena BEFORE the next
        // one asks for any: holding a whole model's KV while allocating a
        // second is how a reload gets refused for memory about to be free.
        self.pool = None;
        let pool = crate::pools::kv::Pool::allocate_elastic(
            &self.context,
            &mut self.stepper,
            &self.arena,
            shape,
        )?;
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
        // Slots are seats, not pages, and one is expensive, so the count is a
        // knob with a small default rather than a fraction of memory.
        self.recurrent = None;
        if let Some(rs) = baked.deployment.recurrent.as_ref() {
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

        self.deployment = Some(baked.deployment.clone());
        self.weights = Some(weights);
        self.baked = Some(baked);

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
            vocab_size: vocab,
            max_model_len,
            activation_dtype: "bf16".to_string(),
            hidden_size: hidden,
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
