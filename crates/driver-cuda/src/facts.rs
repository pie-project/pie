//! What a loaded checkpoint says about the model's shape, and which family
//! text serves it.
//!
//! One row per `model_type` ([`FACTS_ROWS`]), one derivation per family, and
//! the [`PlannedFamily`] trait that is the whole of what the serving path
//! asks a loaded model. The shell that used to hold all of this asked eleven
//! different questions of a three-armed enum at eleven sites; the questions
//! became the trait and the families became rows, so a new family is a row
//! and an impl rather than an edit in the shell.
//!
//! # Why this is its own module
//!
//! It answers a question about the MODEL, not about the fire. Nothing here
//! reads a fire's rows, its class, its peel window or its predicates —
//! `.wiki/driver/graph.md`'s subject matter does not appear in this file at
//! all. Keeping the two apart is what lets the supergraph work and the facts
//! work proceed without touching the same lines.
//!
//! `driver-metal/src/facts.rs` is the same module on the other backend,
//! and it states the destination this one is walking towards: *"portable, and
//! that is the point ... the one reading in the driver that decides a model's
//! geometry can be tested without a GPU."*
//!
//! # What still ties it to the shell
//!
//! [`Checkpoint`] — the borrowed view the shell hands in — and one status
//! code. Nothing else: this module names no CUDA symbol, allocates nothing
//! and touches no device.
//!
//! **That does not make it testable without one today, and the gate above
//! should not be read as claiming so.** This crate refuses to build at all
//! without `cuda-12` or `cuda-13` (`lib.rs`'s first `compile_error!`), so
//! `abi` alone is not a configuration anyone can compile. Narrowing the
//! gate states the dependency honestly; it buys nothing until that
//! `compile_error!` has an answer for a toolkit-free build.
//!
//! What it does buy is the move described below: code that names
//! `crate::loader` cannot go and live in the `model` crate, and this
//! module no longer does.
//!
//! It was gated on both for one day, and the reason is worth recording
//! because it is the failure mode this file is supposed to demonstrate
//! against. The first draft's view held `crate::loader`'s weight map
//! directly. Of the 244 field reads in this module, **228 are of the
//! descriptor** and sixteen are not; those sixteen — six weight lookups,
//! five alias lookups, three `tp_size`, two layer scalars, concentrated in
//! ONE derivation — dragged `loader`'s type in, and `loader` needs a device.
//! Eleven hundred lines of integer arithmetic over a JSON document sat
//! behind a CUDA feature gate to serve eleven call sites. [`Tensors`] is
//! the fix: the two questions asked of the load, as two methods.
//!
//! The `abi` gate that remains is `PIE_STATUS_UNSUPPORTED` — a refusing
//! derivation answers in the ABI's status vocabulary, and `driver-api` is
//! an optional dependency `abi` turns on. A local error enum would drop
//! that too; it is a smaller change than this one was and it is not this
//! commit's.
//!
//! # Where this goes next
//!
//! `.wiki/driver/north-star.md` §4: to the `model` crate, beside the
//! families it derives for and the load contract already there. That is
//! not tidiness. `findings.md` §2.5 measured the cost of the split — this
//! module's [`FACTS_ROWS`] and `model::contract::HF_ROWS` are two tables
//! keyed by the same `model_type`, and they have already disagreed once
//! (`qwen3_moe`: the contract authors it as a GDN mixture, the row here
//! derives it as dense). `tests/facts_registry.rs` holds them to each
//! other today; one table would need no holding. [`Checkpoint`] is the
//! surface that move has to preserve, which is the other reason to have
//! written it down.

use driver_api::local::PIE_STATUS_UNSUPPORTED;

/// What the LOAD answers, as opposed to what the descriptor states.
///
/// Two questions, and both are about bytes that reached the device rather
/// than about text in `config.json`: how big a tensor turned out to be, and
/// what a trace name was bound to. A derivation asks them when the
/// descriptor cannot answer — qwen3's per-head q-norm and olmo2's
/// whole-projection q-norm ship the same KEY and differ only in extent, and
/// whether a projection was fused is a decision the loader made.
///
/// A trait rather than the loader's map, and the reason is the module doc's:
/// `crate::loader` needs a device, these two answers do not, and taking the
/// map would put 1,100 lines of descriptor arithmetic behind a CUDA feature
/// gate to serve eleven call sites.
pub(crate) trait Tensors {
    /// How many bytes the tensor PLACED UNDER THIS NAME occupies, or
    /// `None` if the load placed none.
    ///
    /// Deliberately NOT alias-resolving, unlike the shell's `weight()`
    /// lookup. A derivation that wants the tensor behind a trace name says
    /// so in two steps — [`Self::alias`] then this — because the two
    /// questions have different answers and one call site needs to tell
    /// them apart: `fused_qkv` is true when EITHER a fused buffer was
    /// written under the trace name OR the trace name was aliased onto a
    /// checkpoint that already shipped one, and a lookup that silently
    /// followed aliases could not distinguish those.
    fn bytes(&self, name: &str) -> Option<usize>;

    /// The checkpoint name a trace name was bound to, or `None` when the
    /// load made no such binding.
    fn alias(&self, trace: &str) -> Option<&str>;
}

/// What a family's derivation may read about a loaded checkpoint.
///
/// The SUBJECT is `hf`: 228 of the 244 reads in this module are of the
/// descriptor, because a model's geometry is what its config states. The
/// other three fields are the exceptions, and they are here as exceptions —
/// each one is a question the descriptor cannot answer, asked by one or two
/// derivations.
///
/// It is a borrowed view and not the shell's `LoadedModel` because that
/// struct also holds the arena, the caps blob and the device buffers, none
/// of which a fact about a model's shape has any business reading. This type
/// is the enumeration of what is legitimate, and it is short enough to check
/// by eye.
pub(crate) struct Checkpoint<'a> {
    /// The normalized `pie.model/1` descriptor — the subject.
    pub(crate) hf: &'a crate::model::config::HfConfig,
    /// What the load placed, for the two questions the descriptor cannot
    /// answer. See [`Tensors`].
    pub(crate) tensors: &'a dyn Tensors,
    /// gemma-4's per-layer `layer_scalar` values, read to host once at load.
    /// Read by one `tables` implementation and nothing else.
    pub(crate) gemma_layer_scalars: &'a [f32],
    /// The tensor-parallel group this rank's weights were sharded for, so a
    /// family's facts and its load plan cannot disagree about how wide a
    /// rank is. Read by one derivation.
    pub(crate) tp_size: u32,
}

/// What the SHELL needs to ask a loaded model, and the whole of it.
///
/// `cuda.md` §5.B calls deleting `FamilyFacts` the real half of B's exit,
/// and the reason is this list: a shell that claims not to know which
/// families there are was matching a three-armed enum at eleven sites,
/// each asking a different question. The questions were never the
/// problem — every one of them is a legitimate thing a driver must know
/// before it can plan. Naming the families to answer them was.
///
/// So the questions became the trait, and the shape of the old matches
/// became the defaults: almost every site read `Gemma4(..) => …, _ => …`,
/// one family answering and the rest falling through. A fall-through IS a
/// default body, and writing it here means a family that never mentions
/// `head_dim_of` is *stating* that its layers agree about head dim rather
/// than being lumped in with everything else that never came up.
///
/// A new family implements this and appears in [`FACTS_ROWS`]. Nothing
/// else in the shell learns its name.
pub(crate) trait PlannedFamily {
    /// This family's text, traced and lowered for one fire class.
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan;

    /// Layers in the backbone — the length of every per-layer answer below.
    fn layers(&self) -> u32;

    /// Layer `l`'s head dim. Uniform unless a family says otherwise;
    /// gemma-4's two layer kinds disagree (256 vs 512), which is the only
    /// reason this is per-layer at all.
    fn head_dim_of(&self, _l: u32, uniform: u32) -> u32 {
        uniform
    }

    /// The layer whose KV pages `l` attends through, or `None` when `l`
    /// owns its own. A `Some` layer projects and writes nothing.
    fn kv_source(&self, _l: u32) -> Option<u32> {
        None
    }

    /// Layer `l`'s sliding window, `-1` for the whole context. An empty
    /// answer from [`Self::window_by_layer`] means the fire's single
    /// window applies to every layer.
    fn window_by_layer(&self, _sliding_window: i32) -> Vec<i32> {
        Vec::new()
    }

    /// The attention softmax scale. `1/sqrt(head_dim)` unless the
    /// family's q/k norms already carry it (gemma-4 runs 1.0).
    fn sm_scale(&self, head_dim: u32) -> f32 {
        1.0 / (head_dim as f32).sqrt()
    }

    /// Whether this family carries RECURRENT STATE. Such a fire is not
    /// replayable — a captured body bakes one instance's slots — so it
    /// stays eager, and it is the only family that may be handed an MTP
    /// service class.
    fn recurrent(&self) -> bool {
        false
    }

    /// The two head dims a family's layer kinds decode at, when it needs
    /// SEPARATE decode plans for them — `(sliding, full)`. `None` for a
    /// family whose layers agree, which is why one plan serves them: the
    /// planner bakes the head dim in.
    fn decode_plan_head_dims(&self) -> Option<(u32, u32)> {
        None
    }

    /// Whether the family's PREFILL plans internally, per fire, off the
    /// host CSR mirrors — so there is nothing to pre-plan and the mirrors
    /// must be uploaded.
    fn planless_prefill(&self) -> bool {
        false
    }

    /// Whether both attention forms state `[q, o]` as SSA args, so the
    /// guard-owned attention pins stay null. Only gemma-4 does.
    fn pins_attention_values(&self) -> bool {
        true
    }

    /// Per-layer rope tables, softcap, PLE width and named scalar
    /// constants — everything the prologue reads that is not a shape.
    /// Empty for a family whose rope is one theta and whose epilogue caps
    /// nothing.
    fn tables(&self, _model: &Checkpoint<'_>) -> FamilyTables {
        FamilyTables::default()
    }

    /// The family's recurrent geometry, when it has one.
    fn gdn_shape(&self) -> Option<GdnShape> {
        None
    }

    /// The KV STORE this family's attention reads, when it is not the
    /// paged k/v cache every other family shares.
    ///
    /// `None` means the standard store, which is what `kv_pools_for`
    /// builds. A `Some` names a store this shell does not build, and the
    /// point of asking is to refuse at LOAD rather than at the first fire:
    /// the facts row exists, so `facts_from_hf` succeeds, the checkpoint
    /// reports itself healthy, and the failure arrives as a
    /// `DispatchRefusal` from inside a walk — which is the shape of
    /// failure this crate refuses everywhere else.
    ///
    /// gpt-oss is the precedent and the reason this is not hypothetical:
    /// it was the only family with both a facts row and a Prefill arm, so
    /// a checkpoint loaded, said it was fine, and died at its first fire
    /// on `UnknownWeight("layer.0.router")`.
    fn unbuilt_kv_store(&self) -> Option<&'static str> {
        None
    }
}

impl PlannedFamily for model::gemma_2::forward::facts::Gemma2Facts {
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::gemma_2::forward::gemma2_cuda(self, class)
    }
    fn layers(&self) -> u32 {
        self.layers
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.attn.head_dim
    }
    fn window_by_layer(&self, _sliding_window: i32) -> Vec<i32> {
        self.window_left.clone()
    }
    fn tables(&self, _model: &Checkpoint<'_>) -> FamilyTables {
        FamilyTables {
            softcap: if self.final_logit_softcap { 30.0 } else { 0.0 },
            ..FamilyTables::default()
        }
    }
}

impl PlannedFamily
    for (
        model::gpt_oss::forward::facts::GptOssFacts,
        model::gpt_oss::forward::facts::GptOssCudaFacts,
    )
{
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::gpt_oss::forward::gpt_oss_cuda(&self.0, &self.1, class)
    }
    fn layers(&self) -> u32 {
        self.0.layers
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.0.head_dim
    }
    fn window_by_layer(&self, _sliding_window: i32) -> Vec<i32> {
        self.1.window_left.clone()
    }
}

impl PlannedFamily for model::glm5::forward::facts::Glm5Facts {
    fn unbuilt_kv_store(&self) -> Option<&'static str> {
        // Latent q/kv: the attention reads ckv and kpe planes, not the
        // paged k/v pair `kv_pools_for` builds, and no executor arm names
        // an MLA dispatch. `store::mla_cache` is ported and waiting; until
        // a forward path exists there is nothing to point it at.
        Some("an MLA latent cache")
    }
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::glm5::forward::glm5_cuda(self, class)
    }
    fn layers(&self) -> u32 {
        self.layers
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        // MLA's pages hold the LATENT, not a head-split key, and
        // `kv_a_width` is that row — the shared `MlaFacts` says it once
        // for every family in this lineage.
        self.attn.kv_a_width()
    }
}

impl PlannedFamily
    for (
        model::kimi_k2::forward::facts::KimiFacts,
        model::kimi_k2::forward::facts::KimiCudaFacts,
    )
{
    fn unbuilt_kv_store(&self) -> Option<&'static str> {
        // See the MLA note on `Glm5Facts`.
        Some("an MLA latent cache")
    }
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::kimi_k2::forward::kimi_cuda(&self.0, &self.1, class)
    }
    fn layers(&self) -> u32 {
        self.0.layers
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.0.attn.kv_a_width()
    }
}

impl PlannedFamily for model::kimi_k3::forward::facts::KimiK3Facts {
    fn unbuilt_kv_store(&self) -> Option<&'static str> {
        // Latent q/kv: the attention reads ckv and kpe planes, not the
        // paged k/v pair `kv_pools_for` builds, and no executor arm names
        // an MLA dispatch. `store::mla_cache` is ported and waiting; until
        // a forward path exists there is nothing to point it at.
        Some("an MLA latent cache")
    }
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::kimi_k3::forward::kimi_k3_cuda(self, class)
    }
    fn layers(&self) -> u32 {
        self.layers
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.attn.kv_a_width()
    }
    fn recurrent(&self) -> bool {
        // KDA carries per-request recurrent state, so a fire of this
        // family stays eager for the rule the hybrid states.
        true
    }
}

impl PlannedFamily for model::deepseek_v4::forward::facts::Dsv4Facts {
    fn unbuilt_kv_store(&self) -> Option<&'static str> {
        // Latent q/kv: the attention reads ckv and kpe planes, not the
        // paged k/v pair `kv_pools_for` builds, and no executor arm names
        // an MLA dispatch. `store::mla_cache` is ported and waiting; until
        // a forward path exists there is nothing to point it at.
        Some("the DSv4 compressed cache")
    }
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::deepseek_v4::forward::dsv4_cuda(self, class)
    }
    fn layers(&self) -> u32 {
        self.layers
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.attn.head_dim
    }
    fn window_by_layer(&self, _sliding_window: i32) -> Vec<i32> {
        let w = i32::try_from(self.attn.sliding_window).unwrap_or(0);
        (0..self.layers).map(|_| if w > 0 { w } else { -1 }).collect()
    }
}

impl PlannedFamily for model::nemotron_h::forward::facts::NemotronHFacts {
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::nemotron_h::forward::nemotron_h_cuda(self, class)
    }
    fn layers(&self) -> u32 {
        u32::try_from(self.layer_types.len()).unwrap_or(0)
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.attn.head_dim
    }
    fn window_by_layer(&self, _sliding_window: i32) -> Vec<i32> {
        self.window_left.clone()
    }
    fn recurrent(&self) -> bool {
        // The mamba layers' selective-scan state is per request.
        true
    }
}

impl PlannedFamily for model::gemma3n::forward::facts::Gemma3nFacts {
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::gemma3n::forward::gemma3n_cuda(self, class)
    }
    fn layers(&self) -> u32 {
        u32::try_from(self.per_layer_intermediate.len()).unwrap_or(0)
    }
    fn head_dim_of(&self, _l: u32, _uniform: u32) -> u32 {
        self.attn.head_dim
    }
    fn window_by_layer(&self, _sliding_window: i32) -> Vec<i32> {
        self.window_left.clone()
    }
}

/// The per-layer tables and named constants a family's prologue reads.
#[derive(Default)]
pub(crate) struct FamilyTables {
    /// Rope base per layer; empty means the one `rope_theta` applies.
    pub(crate) theta_by_layer: Vec<f32>,
    /// Rotary width per layer; empty means full rotation at head dim.
    pub(crate) rotary_by_layer: Vec<u32>,
    /// Final-logit softcap, 0 for none.
    pub(crate) softcap: f32,
    /// Per-layer-embedding width, 0 for a family without one.
    pub(crate) ple_dim: i32,
    /// Named scalar constants the trace refers to by name.
    pub(crate) scales: std::collections::BTreeMap<String, f32>,
}

/// A recurrent family's slab geometry — what the shell must allocate and
/// stride before it can hand the executor a `GdnCtx`.
pub(crate) struct GdnShape {
    pub(crate) layers: u32,
    pub(crate) linear_layers: Vec<u32>,
    pub(crate) conv_stride: usize,
    pub(crate) state_stride: usize,
    pub(crate) state_elem: usize,
    pub(crate) k_h: i32,
    pub(crate) v_h: i32,
    pub(crate) k_d: i32,
    pub(crate) v_d: i32,
    pub(crate) conv_dim: i32,
    pub(crate) conv_k: i32,
}

/// The three implementations. Each is the set of answers its family
/// PREVIOUSLY contributed to eleven scattered matches, gathered into one
/// place where the family's own name is the last time it is mentioned.

impl PlannedFamily
    for (
        model::families::llama_like::forward::facts::LlamaLikeFacts,
        model::families::llama_like::forward::facts::LlamaLikeCudaFacts,
    )
{
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::families::llama_like::forward::llama_like_cuda(&self.0, &self.1, class)
    }
    fn layers(&self) -> u32 {
        self.0.layers
    }
    // Every other answer is the default: uniform head dim, no KV sharing,
    // one window, the standard scale, no recurrence, one decode plan, no
    // per-layer tables. The lineage is the family the defaults were
    // written from.
}

impl PlannedFamily
    for (
        model::qwen_3_5::forward::facts::Qwen35HybridFacts,
        model::qwen_3_5::forward::facts::Qwen35CudaFacts,
    )
{
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::qwen_3_5::forward::qwen3_5_hybrid_cuda(&self.0, &self.1, class)
    }
    fn layers(&self) -> u32 {
        self.0.layers
    }
    fn recurrent(&self) -> bool {
        true
    }
    fn gdn_shape(&self) -> Option<GdnShape> {
        let g = &self.0.gdn;
        Some(GdnShape {
            layers: self.0.layers,
            linear_layers: (0..self.0.layers).filter(|&l| !self.0.is_full_attn(l)).collect(),
            conv_stride: (g.conv_kernel * g.conv_dim()) as usize,
            state_stride: (g.value_heads * g.key_head_dim * g.value_head_dim) as usize,
            state_elem: if self.1.state_bf16 { 2 } else { 4 },
            k_h: g.key_heads as i32,
            v_h: g.value_heads as i32,
            k_d: g.key_head_dim as i32,
            v_d: g.value_head_dim as i32,
            conv_dim: g.conv_dim() as i32,
            conv_k: g.conv_kernel as i32,
        })
    }
}

impl PlannedFamily
    for (
        model::gemma_4::forward::facts::Gemma4Facts,
        model::gemma_4::forward::facts::Gemma4CudaFacts,
    )
{
    fn trace(&self, class: model_compiler::trace::FireClass) -> model_compiler::trace::ForwardPlan {
        model::gemma_4::forward::gemma4_cuda(&self.0, &self.1, class)
    }
    fn layers(&self) -> u32 {
        self.0.layers
    }
    fn head_dim_of(&self, l: u32, _uniform: u32) -> u32 {
        self.0.head_dim_of(l)
    }
    fn kv_source(&self, l: u32) -> Option<u32> {
        self.0.kv_source(l)
    }
    fn window_by_layer(&self, sliding_window: i32) -> Vec<i32> {
        (0..self.0.layers)
            .map(|l| if self.0.is_full_attn(l) { -1 } else { sliding_window.max(0) })
            .collect()
    }
    fn sm_scale(&self, _head_dim: u32) -> f32 {
        // The q/k norms carry the scaling.
        1.0
    }
    fn decode_plan_head_dims(&self) -> Option<(u32, u32)> {
        Some((self.0.head_dim, self.0.global_head_dim))
    }
    fn planless_prefill(&self) -> bool {
        true
    }
    fn pins_attention_values(&self) -> bool {
        // Both attention forms state [q, o] as SSA args.
        false
    }
    fn tables(&self, model: &Checkpoint<'_>) -> FamilyTables {
        let (facts, hf) = (&self.0, model.hf);
        let theta: Vec<f32> = (0..facts.layers as usize)
            .map(|l| {
                hf.gemma_per_layer_rope_theta.get(l).copied().unwrap_or({
                    // The C++ parse fallback: full layers (and configs
                    // without a local base) ride `rope_theta`.
                    if facts.is_full_attn(l as u32) || hf.gemma3n_rope_local_base_freq <= 0.0 {
                        hf.rope_theta
                    } else {
                        hf.gemma3n_rope_local_base_freq
                    }
                })
            })
            .collect();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rotary: Vec<u32> = (0..facts.layers)
            .map(|l| {
                let f = hf
                    .gemma_per_layer_partial_rotary_factor
                    .get(l as usize)
                    .copied()
                    .unwrap_or(1.0);
                let d = facts.head_dim_of(l) as f32;
                2u32.max(2 * (0.5 * f * d) as u32)
            })
            .collect();
        let mut scales = std::collections::BTreeMap::new();
        let hidden = facts.hidden as f32;
        scales.insert("sqrt_hidden".into(), hidden.sqrt());
        scales.insert("sqrt_ple_dim".into(), (facts.ple_dim as f32).sqrt());
        scales.insert("rsqrt_hidden".into(), 1.0 / hidden.sqrt());
        scales.insert("rsqrt_2".into(), 1.0 / 2f32.sqrt());
        for (n, sc) in model.gemma_layer_scalars.iter().enumerate() {
            scales.insert(format!("layer.{n}.ple_norm"), *sc);
        }
        FamilyTables {
            theta_by_layer: theta,
            rotary_by_layer: rotary,
            softcap: facts.logit_softcap,
            ple_dim: facts.ple_dim as i32,
            scales,
        }
    }
}

/// gemma-4's facts off the checkpoint's config — the layer schedule
/// reduced to the interval (irregular arrays refuse, qwen3_5's rule),
/// the FULL layers' rotary width by the driver's derivation, the
/// double-wide-MLP and KV-shared counts as stated. The E2B anchor's
/// legs only: `k_eq_v` (26B-A4B's V-from-K mode) and the MoE block
/// refuse until a deployment anchors them.
fn gemma4_facts_from_hf(model: &Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::gemma_4::forward::facts::{Gemma4CudaFacts, Gemma4Facts};
    let hf = model.hf;
    let interval = u32::try_from(
        hf.layer_types.iter().position(|t| t == "full_attention").map_or(0, |i| i + 1),
    )
    .unwrap_or(0);
    let regular = interval > 0
        && hf.layer_types.iter().enumerate().all(|(l, t)| {
            (t == "full_attention") == (l as u32 % interval == interval - 1)
        });
    if !regular {
        eprintln!("[driver-cuda] launch: irregular gemma-4 layer_types refuse");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    if hf.gemma4_attention_k_eq_v || hf.gemma4_enable_moe {
        eprintln!("[driver-cuda] launch: gemma-4 k_eq_v/MoE legs await their anchor");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    let to_u32 = |v: i32| u32::try_from(v).unwrap_or(0);
    let global_d = to_u32(hf.gemma4_global_head_dim.max(hf.head_dim));
    // The FULL layers' partial factor: the per-layer table when the
    // config ships one, else full rotation — `rotary_of`'s input.
    let full_factor = (0..hf.layer_types.len())
        .find(|&l| hf.layer_types[l] == "full_attention")
        .and_then(|l| hf.gemma_per_layer_partial_rotary_factor.get(l).copied())
        .unwrap_or(1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let global_rotary = 2u32.max(2 * (0.5 * full_factor * global_d as f32) as u32);
    let facts = Gemma4Facts {
        hidden: to_u32(hf.hidden_size),
        layers: to_u32(hf.num_hidden_layers),
        full_attn_interval: interval,
        q_heads: to_u32(hf.num_attention_heads),
        kv_heads: to_u32(hf.num_key_value_heads),
        head_dim: to_u32(hf.head_dim),
        global_head_dim: global_d,
        global_rotary_dim: global_rotary,
        intermediate: to_u32(hf.intermediate_size),
        vocab: to_u32(hf.vocab_size),
        tied_embeddings: hf.tie_word_embeddings,
        kv_shared_layers: to_u32(hf.num_kv_shared_layers),
        ple_dim: to_u32(hf.gemma_hidden_size_per_layer_input),
        double_wide_shared: hf.gemma4_double_wide_mlp,
        logit_softcap: hf.gemma_final_logit_softcap,
    };
    // The LIVE binding: both banks fused (the load's joins built them),
    // native bf16 pages — the A/B's proven set.
    //
    // `window_left` is NOT empty here, and gemma-4 is the family that
    // makes the difference visible: full-attention layers see the whole
    // context and the rest attend a sliding window, on the family's own
    // interval. The shell already derived exactly this list for its
    // decode plans; now the declaration carries it too, and an empty list
    // would have the trace say "no window" while the plan applied one.
    let cuda = Gemma4CudaFacts {
        fused_qkv: true,
        gate_up_fused: true,
        kv_native_bf16: true,
        window_left: (0..facts.layers)
            .map(|l| if facts.is_full_attn(l) { -1 } else { hf.sliding_window.max(0) })
            .collect(),
    };
    Ok(Box::new((facts, cuda)))
}

/// The qwen3_5 hybrid's facts, read off the checkpoint's own config —
/// the layer schedule from `layer_types` (reduced to the interval, the
/// Metal driver's reduction; irregular arrays refuse), the GDN geometry
/// from the `linear_*` fields, the rotary width by the driver's
/// `max(2, 2·int(0.5·factor·head_dim))` derivation. Dense MLP only —
/// a MoE config refuses until a MoE deployment anchors that leg.
fn qwen35_facts_from_hf(model: &Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::qwen_3_5::forward::facts::{
        Qwen35CudaFacts, Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind,
        Qwen35MoeMlpFacts,
    };
    use model_compiler::trace::NormVariant;
    let hf = model.hf;
    let interval = u32::try_from(
        hf.layer_types.iter().position(|t| t == "full_attention").map_or(0, |i| i + 1),
    )
    .unwrap_or(0);
    let regular = interval > 0
        && hf.layer_types.iter().enumerate().all(|(l, t)| {
            (t == "full_attention") == (l as u32 % interval == interval - 1)
        });
    if !regular {
        eprintln!("[driver-cuda] launch: irregular qwen3_5 layer_types refuse");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    if hf.num_experts > 0 {
        eprintln!("[driver-cuda] launch: the qwen3_5 MoE leg awaits its anchor deployment");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    let to_u32 = |v: i32| u32::try_from(v).unwrap_or(0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let rotary =
        2u32.max(2 * (0.5 * hf.partial_rotary_factor * hf.head_dim as f32) as u32);
    let facts = Qwen35HybridFacts {
        layers: to_u32(hf.num_hidden_layers),
        full_attn_interval: interval,
        vocab: to_u32(hf.vocab_size),
        tied_embeddings: hf.tie_word_embeddings,
        norm_variant: NormVariant::Gemma,
        attn: Qwen35FullAttnFacts {
            hidden: to_u32(hf.hidden_size),
            q_heads: to_u32(hf.num_attention_heads),
            kv_heads: to_u32(hf.num_key_value_heads),
            head_dim: to_u32(hf.head_dim),
            rotary_dim: rotary,
            fused_qkv: false,
            norm_variant: NormVariant::Gemma,
        },
        gdn: Qwen35GdnFacts {
            hidden: to_u32(hf.hidden_size),
            key_heads: to_u32(hf.linear_num_key_heads),
            value_heads: to_u32(hf.linear_num_value_heads),
            key_head_dim: to_u32(hf.linear_key_head_dim),
            value_head_dim: to_u32(hf.linear_value_head_dim),
            conv_kernel: to_u32(hf.linear_conv_kernel_dim),
            fused_in_proj: false,
            norm_variant: NormVariant::Gemma,
        },
        // THE MLP KIND, off the config rather than assumed dense.
        //
        // `n_experts > 0` IS the mixture — the same reading
        // `LlamaLikeFacts::n_experts` documents, and the reason a routed
        // FFN is a fact and not a family: the attention is unchanged and
        // only the block between the two norms differs. The hybrid's own
        // text already branches on `Qwen35MlpKind`, so this derivation
        // was the only thing making every qwen3_5 deployment dense.
        //
        // Qwen3.5-35B-A3B is what it opens: 256 routed experts, top-k 8,
        // `moe_intermediate` 512 beside a shared expert of the same
        // width. Those numbers were PINNED as a fixture from the C++
        // driver's measured notes because no config was committed; the
        // checkpoint's own config agrees with the fixture on every one.
        mlp: if to_u32(hf.num_experts) > 0 {
            Qwen35MlpKind::Moe(Qwen35MoeMlpFacts {
                hidden: to_u32(hf.hidden_size),
                num_experts: to_u32(hf.num_experts),
                top_k: to_u32(hf.num_experts_per_tok),
                moe_intermediate: to_u32(hf.moe_intermediate_size),
                shared_expert_intermediate: to_u32(hf.shared_expert_intermediate_size),
                norm_variant: NormVariant::Gemma,
            })
        } else {
            Qwen35MlpKind::Dense { intermediate: to_u32(hf.intermediate_size) }
        },
    };
    // The LIVE L40S cuda set (`emissions.rs`): warp-tiled and the cached
    // prefill env-gated off, bf16 recurrent state, prefill-decode on.
    let cuda = Qwen35CudaFacts {
        state_bf16: true,
        warp_tiled: false,
        warp_tiled_max: 64,
        cached_max: 0,
        verify_stash: true,
        prefill_decode: true,
        moe_cutlass_max_rows: 0,
        moe_residual_fold: false,
        moe_shared_gate_dot: false,
        moe_streamed_experts: false,
        moe_force_general: false,
        gate_up_fused: true,
        // As llama_like's, and for the same reason.
        proj_repr: model_compiler::dsl::WeightRepr::Bf16,
        window_left: Vec::new(),
    };
    Ok(Box::new((facts, cuda)))
}

/// What a row dispatches to: this family's facts, off the checkpoint.
type FactsFrom = fn(&Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32>;

/// One row per `model_type` this shell can OPEN.
///
/// A table rather than the chain of weight-name sniffs this replaces, for
/// exactly the reason `model::contract::HF_ROWS` is a table: the supported
/// set becomes a VALUE something can iterate. The gap between what the
/// loader can author and what this shell can open is then a test with a
/// closed list (`tests/facts_registry.rs`) rather than a surprise at boot
/// — which is what §3.3's "eight families dispatch but cannot load" was.
///
/// Dispatch is on the model type because that is what the descriptor
/// SAYS. Sniffing a weight name infers the family from a consequence of
/// it, which is how `gemma3` used to be answered by the llama-like
/// derivation: it has `model.embed_tokens.weight` and a pre-norm, so the
/// sniff accepted it and transcribed the wrong facts. A model type with
/// no row now refuses by name, which is this plan's standing rule —
/// refuse what cannot be derived rather than guess it.
const FACTS_ROWS: &[(&str, FactsFrom)] = &[
    // ── llama lineage: dense/GQA decoders the llama_like text serves.
    ("qwen3", llama_like_facts_from_hf),
    ("qwen2", llama_like_facts_from_hf),
    ("llama", llama_like_facts_from_hf),
    ("llama3", llama_like_facts_from_hf),
    ("mistral", llama_like_facts_from_hf),
    ("mistral3", llama_like_facts_from_hf),
    ("ministral3", llama_like_facts_from_hf),
    ("olmo2", llama_like_facts_from_hf),
    ("olmo3", llama_like_facts_from_hf),
    ("phi3", llama_like_facts_from_hf),
    // Qwen3-VL binds the plain Qwen3 TEXT tower; the vision tower is a
    // service behind `pie_cuda_encode`, not part of this decode plan.
    ("qwen3_vl", llama_like_facts_from_hf),
    ("qwen3_vl_text", llama_like_facts_from_hf),
    // gemma-3 is a llama-lineage decoder with per-head qk-norm and an
    // alternating window; the derivation reads both off the checkpoint.
    ("gemma3", llama_like_facts_from_hf),
    ("gemma3_text", llama_like_facts_from_hf),
    // A ROUTED FFN is a fact, not a family: `n_experts > 0` selects the
    // mixture and the attention is unchanged, which is why these two
    // reach the same derivation as every dense deployment above.
    ("mixtral", llama_like_facts_from_hf),
    ("qwen3_moe", llama_like_facts_from_hf),
    // ── Gemma-4: nested decoder, PLE, two layer kinds.
    ("gemma4", gemma4_facts_from_hf),
    ("gemma4_text", gemma4_facts_from_hf),
    // ── Qwen3.5 hybrids: GDN linear attention beside full attention.
    ("qwen3_5", qwen35_facts_from_hf),
    ("qwen3_5_text", qwen35_facts_from_hf),
    ("qwen3_5_moe", qwen35_facts_from_hf),
    ("qwen3_5_moe_text", qwen35_facts_from_hf),
    // ── gemma-2: alternating local/global attention, softcapped twice.
    ("gemma2", gemma2_facts_from_hf),
    // ── gpt-oss: MXFP4 mixture, attention sinks, alternating window.
    ("gpt_oss", gpt_oss_facts_from_hf),
    // ── The MLA lineage: latent q/kv, a dense prefix, then the mixture.
    ("glm_moe_dsa", glm5_facts_from_hf),
    ("deepseek_v2", kimi_k2_facts_from_hf),
    ("deepseek_v3", kimi_k2_facts_from_hf),
    ("kimi_k2", kimi_k2_facts_from_hf),
    ("kimi_k3", kimi_k3_facts_from_hf),
    ("deepseek_v4", dsv4_facts_from_hf),
    // ── Hybrids and the per-layer-embedding gemma.
    ("nemotron_h", nemotron_h_facts_from_hf),
    ("gemma3n", gemma3n_facts_from_hf),
    ("gemma3n_text", gemma3n_facts_from_hf),
];

/// Every `model_type` this shell can open, in table order.
///
/// Public so that `tests/facts_registry.rs` can hold it against the
/// loader's own registry. The two lists answering "which model type is
/// supported" from opposite sides of the load is exactly the pairing
/// `model::contract`'s header describes, and the same failure it names
/// applies here: a family whose forward is declared but whose facts were
/// never written used to surface as a wrong answer rather than a refusal.
#[must_use]
pub fn openable_model_types() -> Vec<&'static str> {
    FACTS_ROWS.iter().map(|(k, _)| *k).collect()
}

/// gemma-2's facts off the checkpoint's config.
///
/// The window list is the family's own shape: gemma-2 ALTERNATES local
/// and global attention, odd layers seeing the whole context. `layer_types`
/// states it when the config ships one; the parity is the fallback the
/// C++ parse used.
fn gemma2_facts_from_hf(model: &Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::gemma_2::forward::facts::{Gemma2AttnFacts, Gemma2Facts};
    let hf = model.hf;
    let to_u32 = |v: i32| u32::try_from(v).unwrap_or(0);
    let layers = to_u32(hf.num_hidden_layers);
    let window_left: Vec<i32> = (0..layers)
        .map(|l| {
            let global = hf
                .layer_types
                .get(l as usize)
                .map_or(l % 2 == 1, |t| t == "full_attention");
            if global { -1 } else { hf.sliding_window.max(0) }
        })
        .collect();
    Ok(Box::new(Gemma2Facts {
        layers,
        vocab: to_u32(hf.vocab_size),
        hidden: to_u32(hf.hidden_size),
        intermediate: to_u32(hf.intermediate_size),
        tied_embeddings: hf.tie_word_embeddings,
        final_logit_softcap: hf.gemma_final_logit_softcap > 0.0,
        window_left,
        attn: Gemma2AttnFacts {
            heads: to_u32(hf.num_attention_heads),
            kv_heads: to_u32(hf.num_key_value_heads),
            head_dim: to_u32(hf.head_dim),
            qk_norm: false,
            query_pre_attn_scale: true,
            attn_logit_softcap: true,
        },
    }))
}

/// gpt-oss's facts. The sliding schedule is `layer_types`' when the
/// config ships one — gpt-oss alternates from layer 0 — and the fused
/// MXFP4 decode leg is the engine default this text states.
fn gpt_oss_facts_from_hf(model: &Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::gpt_oss::forward::facts::{GptOssCudaFacts, GptOssFacts};
    let hf = model.hf;
    let to_u32 = |v: i32| u32::try_from(v).unwrap_or(0);
    let layers = to_u32(hf.num_hidden_layers);
    let experts = to_u32(hf.num_experts);
    let facts = GptOssFacts {
        hidden: to_u32(hf.hidden_size),
        layers,
        q_heads: to_u32(hf.num_attention_heads),
        kv_heads: to_u32(hf.num_key_value_heads),
        head_dim: to_u32(hf.head_dim),
        intermediate: to_u32(hf.intermediate_size),
        experts,
        top_k: to_u32(hf.num_experts_per_tok),
        vocab: to_u32(hf.vocab_size),
        tied_embeddings: hf.tie_word_embeddings,
        swiglu_limit: 7.0,
        attention_bias: true,
        rope_yarn_original: true,
        attn_sinks: true,
    };
    let cuda = GptOssCudaFacts {
        mxfp4_decode_gemv: true,
        mxfp4_decode_max_routes: 32 * experts.max(1),
        streamed_experts: false,
        window_left: (0..layers)
            .map(|l| {
                let sliding = hf
                    .layer_types
                    .get(l as usize)
                    .map_or(l % 2 == 0, |t| t == "sliding_attention");
                if sliding { hf.sliding_window.max(0) } else { -1 }
            })
            .collect(),
    };
    Ok(Box::new((facts, cuda)))
}

/// The MLA lineage's shared reading: a dense PREFIX then the mixture,
/// latent q/kv projections, and the rope half carried beside the nope
/// half. `first_k_dense_replace` is the prefix length in every config
/// that ships one.
fn glm5_facts_from_hf(model: &Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::glm5::forward::facts::{Glm5DsaFacts, Glm5Facts, Glm5MlaFacts, Glm5MoeFacts};
    let hf = model.hf;
    let u = |v: i32| u32::try_from(v).unwrap_or(0);
    Ok(Box::new(Glm5Facts {
        layers: u(hf.num_hidden_layers),
        vocab: u(hf.vocab_size),
        hidden: u(hf.hidden_size),
        dense_intermediate: u(hf.intermediate_size),
        dense_layers: u(hf.first_k_dense_replace),
        attn: Glm5MlaFacts {
            hidden: u(hf.hidden_size),
            heads: u(hf.num_attention_heads),
            q_lora_rank: u(hf.q_lora_rank),
            kv_lora_rank: u(hf.kv_lora_rank),
            qk_nope_head_dim: u(hf.qk_nope_head_dim),
            qk_rope_head_dim: u(hf.qk_rope_head_dim),
            v_head_dim: u(hf.v_head_dim),
            // Only kimi-k3 gates the MLA output.
            output_gate: false,
        },
        dsa: Glm5DsaFacts {
            index_n_heads: u(hf.dsv4_index_n_heads),
            index_head_dim: u(hf.dsv4_index_head_dim),
            index_topk: u(hf.dsv4_index_topk),
        },
        moe: Glm5MoeFacts {
            hidden: u(hf.hidden_size),
            num_experts: u(hf.num_experts),
            top_k: u(hf.num_experts_per_tok),
            moe_intermediate: u(hf.moe_intermediate_size),
            shared_intermediate: u(hf.n_shared_experts) * u(hf.moe_intermediate_size),
            aligned_block: 16,
        },
    }))
}

/// kimi-k2: the same MLA reading as glm5, without the DSA indexer.
fn kimi_k2_facts_from_hf(model: &Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::kimi_k2::forward::facts::{KimiCudaFacts, KimiFacts, KimiMlaFacts, KimiMoeFacts};
    let hf = model.hf;
    let u = |v: i32| u32::try_from(v).unwrap_or(0);
    let facts = KimiFacts {
        layers: u(hf.num_hidden_layers),
        vocab: u(hf.vocab_size),
        hidden: u(hf.hidden_size),
        dense_intermediate: u(hf.intermediate_size),
        dense_layers: u(hf.first_k_dense_replace),
        attn: KimiMlaFacts {
            hidden: u(hf.hidden_size),
            heads: u(hf.num_attention_heads),
            q_lora_rank: u(hf.q_lora_rank),
            kv_lora_rank: u(hf.kv_lora_rank),
            qk_nope_head_dim: u(hf.qk_nope_head_dim),
            qk_rope_head_dim: u(hf.qk_rope_head_dim),
            v_head_dim: u(hf.v_head_dim),
            // Only kimi-k3 gates the MLA output.
            output_gate: false,
        },
        moe: KimiMoeFacts {
            num_experts: u(hf.num_experts),
            top_k: u(hf.num_experts_per_tok),
            moe_intermediate: u(hf.moe_intermediate_size),
            shared_intermediate: u(hf.n_shared_experts) * u(hf.moe_intermediate_size),
        },
    };
    // The BINDING facts: one fused q/kv latent GEMM when the load joined
    // them, and YaRN when the config asked for it.
    let cuda = KimiCudaFacts {
        q_kv_a_fused: model.tensors.alias("layer.0.q_kv_a_fused").is_some(),
        rope_yarn_original: matches!(
            hf.rope_scaling_kind,
            crate::model::config::RopeScaling::OriginalYarn
        ),
    };
    Ok(Box::new((facts, cuda)))
}

/// kimi-k3: MLA beside KDA linear attention, on the periodic schedule
/// `full_attn_at` states.
fn kimi_k3_facts_from_hf(model: &Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::kimi_k3::forward::facts::{
        KimiK3Facts, KimiK3KdaFacts, KimiK3MlaFacts, KimiK3MoeFacts,
    };
    let hf = model.hf;
    let u = |v: i32| u32::try_from(v).unwrap_or(0);
    let interval = u32::try_from(
        hf.layer_types.iter().position(|t| t == "full_attention").map_or(0, |i| i + 1),
    )
    .unwrap_or(0);
    Ok(Box::new(KimiK3Facts {
        layers: u(hf.num_hidden_layers),
        vocab: u(hf.vocab_size),
        hidden: u(hf.hidden_size),
        dense_intermediate: u(hf.intermediate_size),
        dense_layers: u(hf.first_k_dense_replace),
        full_attn_interval: interval,
        attn_res_block: 0,
        attn: KimiK3MlaFacts {
            hidden: u(hf.hidden_size),
            heads: u(hf.num_attention_heads),
            q_lora_rank: u(hf.q_lora_rank),
            kv_lora_rank: u(hf.kv_lora_rank),
            qk_nope_head_dim: u(hf.qk_nope_head_dim),
            qk_rope_head_dim: u(hf.qk_rope_head_dim),
            v_head_dim: u(hf.v_head_dim),
            output_gate: true,
        },
        kda: KimiK3KdaFacts {
            value_heads: u(hf.linear_num_value_heads),
            value_head_dim: u(hf.linear_value_head_dim),
            conv_kernel: u(hf.linear_conv_kernel_dim),
            gate_lower_bound_milli: 0,
        },
        moe: KimiK3MoeFacts {
            num_experts: u(hf.num_experts),
            top_k: u(hf.num_experts_per_tok),
            moe_intermediate: u(hf.moe_intermediate_size),
            shared_intermediate: u(hf.n_shared_experts) * u(hf.moe_intermediate_size),
        },
    }))
}

/// deepseek-v4: the DSA indexer, hyper-connections, and a routed MLP
/// whose activation clamps.
fn dsv4_facts_from_hf(model: &Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::deepseek_v4::forward::facts::{
        Dsv4AttnFacts, Dsv4Facts, Dsv4HcFacts, Dsv4MoeFacts,
    };
    let hf = model.hf;
    let u = |v: i32| u32::try_from(v).unwrap_or(0);
    Ok(Box::new(Dsv4Facts {
        layers: u(hf.num_hidden_layers),
        vocab: u(hf.vocab_size),
        hidden: u(hf.hidden_size),
        dense_intermediate: u(hf.intermediate_size),
        dense_layers: u(hf.first_k_dense_replace),
        attn: Dsv4AttnFacts {
            hidden: u(hf.hidden_size),
            heads: u(hf.num_attention_heads),
            head_dim: u(hf.head_dim),
            q_lora_rank: u(hf.q_lora_rank),
            qk_rope_head_dim: u(hf.qk_rope_head_dim),
            sliding_window: u(hf.sliding_window.max(0)),
            o_lora_rank: 0,
            o_groups: 1,
        },
        hc: Dsv4HcFacts { mult: 1 },
        moe: Dsv4MoeFacts {
            num_experts: u(hf.num_experts),
            top_k: u(hf.num_experts_per_tok),
            moe_intermediate: u(hf.moe_intermediate_size),
            swiglu_limit_milli: 0,
            hash_routed: false,
        },
    }))
}

/// nemotron-h: three layer kinds, and the schedule is the LIST rather
/// than an interval — the family has an MLP-only layer no period spells.
fn nemotron_h_facts_from_hf(model: &Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::nemotron_h::forward::facts::{
        NemotronAttnFacts, NemotronHFacts, NemotronLayerKind, NemotronMambaFacts,
        NemotronMoeFacts,
    };
    let hf = model.hf;
    let u = |v: i32| u32::try_from(v).unwrap_or(0);
    let layer_types: Vec<NemotronLayerKind> = hf
        .layer_types
        .iter()
        .map(|t| match t.as_str() {
            "attention" | "full_attention" => NemotronLayerKind::Attention,
            "mlp" => NemotronLayerKind::Mlp,
            _ => NemotronLayerKind::Mamba,
        })
        .collect();
    if layer_types.is_empty() {
        eprintln!("[driver-cuda] launch: nemotron-h states no layer_types");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    let window_left = vec![-1; layer_types.len()];
    Ok(Box::new(NemotronHFacts {
        vocab: u(hf.vocab_size),
        hidden: u(hf.hidden_size),
        layer_types,
        mamba: NemotronMambaFacts {
            num_heads: u(hf.mamba_num_heads),
            head_dim: u(hf.mamba_head_dim),
            state_size: u(hf.mamba_state_size),
            n_groups: u(hf.mamba_n_groups),
            conv_kernel: u(hf.mamba_conv_kernel),
        },
        attn: NemotronAttnFacts {
            heads: u(hf.num_attention_heads),
            kv_heads: u(hf.num_key_value_heads),
            head_dim: u(hf.head_dim),
        },
        moe: NemotronMoeFacts {
            num_experts: u(hf.num_experts),
            top_k: u(hf.num_experts_per_tok),
            moe_intermediate: u(hf.moe_intermediate_size),
            shared_intermediate: u(hf.shared_expert_intermediate_size),
        },
        window_left,
    }))
}

/// gemma-3n: altUp streams, laurel, per-layer embeddings and a per-layer
/// MLP width the config states as a list.
fn gemma3n_facts_from_hf(model: &Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::gemma3n::forward::facts::{Gemma3nAltUpFacts, Gemma3nAttnFacts, Gemma3nFacts};
    let hf = model.hf;
    let u = |v: i32| u32::try_from(v).unwrap_or(0);
    let layers = u(hf.num_hidden_layers) as usize;
    Ok(Box::new(Gemma3nFacts {
        vocab: u(hf.vocab_size),
        hidden: u(hf.hidden_size),
        per_layer_intermediate: vec![u(hf.intermediate_size); layers],
        laurel_rank: u(hf.laurel_rank),
        ple_width: u(hf.gemma_hidden_size_per_layer_input),
        sparsity_layers: u32::try_from(
            hf.gemma3n_activation_sparsity.iter().filter(|&&s| s > 0.0).count(),
        )
        .unwrap_or(0),
        altup: Gemma3nAltUpFacts {
            num_streams: u(hf.altup_num_inputs),
            active: u(hf.altup_active_idx),
        },
        attn: Gemma3nAttnFacts {
            heads: u(hf.num_attention_heads),
            kv_heads: u(hf.num_key_value_heads),
            head_dim: u(hf.head_dim),
        },
        window_left: (0..layers)
            .map(|l| {
                if hf.layer_types.get(l).is_some_and(|t| t == "full_attention") {
                    -1
                } else {
                    hf.sliding_window.max(0)
                }
            })
            .collect(),
    }))
}

/// THE GQA RATIO, refused at LOAD rather than discovered at launch.
///
/// FlashInfer's decode instantiates group sizes {1, 2, 3, 4, 8} and
/// reports anything else by THROWING. A throw crossing the C ABI is
/// undefined behaviour; the generated shim prints the message before it
/// dies, but printing is all it can do — the launcher signatures have
/// nowhere to put a failure. A load DOES: it returns a status code.
///
/// This lived inside the llama lineage's derivation, which made it a
/// property of that lineage rather than of the BUILD. It is the build's:
/// every family whose attention reaches the same dispatch is subject to
/// the same instantiation set, and the hybrid is the live proof —
/// Qwen3.6-27B declares `qwen3_5_text`, so it is already openable, and
/// its 24 query heads over 4 kv heads is a group size of six.
///
/// Qwen2.5-1.5B is the other live example, twelve over two.
fn refuse_unservable_gqa(hf: &crate::model::config::HfConfig) -> Result<(), i32> {
    let kv_heads = hf.num_key_value_heads.max(1);
    let group_size = hf.num_attention_heads / kv_heads;
    if hf.num_attention_heads % kv_heads != 0 || !matches!(group_size, 1 | 2 | 3 | 4 | 8) {
        eprintln!(
            "[driver-cuda] load: this build's decode does not instantiate \
             GQA group size {group_size} ({} q heads over {kv_heads} kv heads); \
             the supported set is 1, 2, 3, 4, 8",
            hf.num_attention_heads
        );
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    Ok(())
}

/// The facts for a loaded checkpoint, by the model type it declares.
pub(crate) fn facts_from_hf(model: &Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32> {
    refuse_unservable_gqa(model.hf)?;
    let model_type = model.hf.model_type.as_str();
    match FACTS_ROWS.iter().find(|(k, _)| *k == model_type) {
        Some((_, derive)) => derive(model),
        None => {
            eprintln!(
                "[driver-cuda] launch: no facts derivation for \
                 model_type='{model_type}'; the family declares a forward \
                 but nobody has written its facts"
            );
            Err(PIE_STATUS_UNSUPPORTED)
        }
    }
}

/// The llama lineage's facts, off the checkpoint's own config.
fn llama_like_facts_from_hf(model: &Checkpoint<'_>) -> Result<Box<dyn PlannedFamily>, i32> {
    use model::families::llama_like::forward::facts::{
        LlamaLikeCudaFacts, LlamaLikeFacts, NormPlacement, QkNorm,
    };
    use model_compiler::trace::{NormVariant, RopeKind};
    let hf = model.hf;
    if model.tensors.bytes("model.embed_tokens.weight").is_none() {
        eprintln!("[driver-cuda] launch: only HF llama-like checkpoints execute today");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    // NORM PLACEMENT, off the checkpoint. `input_layernorm`'s presence IS
    // the placement, which is the same fact `fuse_llama_like` already
    // binds on: pre-norm ships it, post-norm (olmo2) ships
    // `post_attention` + `post_feedforward` instead. The binder was
    // already correct for both; only this derivation refused.
    let pre_norm = model
        .tensors
        .alias("layer.0.attn_norm")
        .is_some_and(|t| t.ends_with("input_layernorm.weight"));

    // QK NORM, three ways, and the checkpoint distinguishes them by
    // SHAPE rather than by any config key. A deployment that norms q and
    // k ships `q_norm`/`k_norm`; whether it norms PER HEAD (qwen3, one
    // gamma of `head_dim`) or over the whole projection (olmo2, one gamma
    // of `q_heads * head_dim`) is the tensor's own extent. Reading the
    // extent is deriving from the checkpoint; assuming one is guessing,
    // and the two lower to different kernels.
    let elems_of = |trace: &str| -> Option<usize> {
        let ckpt = model.tensors.alias(trace)?;
        // bf16 gammas throughout this family.
        Some(model.tensors.bytes(ckpt)? / 2)
    };
    let qk_norm = match elems_of("layer.0.q_norm") {
        None => QkNorm::Off,
        Some(n) if n == usize::try_from(hf.head_dim).unwrap_or(0) => QkNorm::PerHead,
        Some(_) => QkNorm::Global,
    };

    // FUSED QKV is a fact about the LOAD, not about the checkpoint:
    // `fuse_llama_like` concatenates q/k/v when all three are present and
    // leaves them alone when they are not. So the honest source is
    // whether the fused name exists, which is what the trace will state.
    // Either spelling counts: `fuse` writes a concatenated buffer under
    // the trace name, while a checkpoint that already ships the fused
    // projection gets an alias to it instead.
    let fused_qkv = model.tensors.bytes("layer.0.qkv").is_some()
        || model.tensors.alias("layer.0.qkv").is_some();

    let to_u32 = |v: i32| u32::try_from(v).unwrap_or(0);
    let facts = LlamaLikeFacts {
        hidden: to_u32(hf.hidden_size),
        layers: to_u32(hf.num_hidden_layers),
        q_heads: to_u32(hf.num_attention_heads),
        kv_heads: to_u32(hf.num_key_value_heads),
        head_dim: to_u32(hf.head_dim),
        // A ROUTED FFN is a fact, not a family (the `LlamaLikeFacts` doc's
        // own argument), so these come off the checkpoint like every other
        // width. Zero throughout is a dense deployment, which is what the
        // fields mean rather than a stand-in for "unknown".
        n_experts: to_u32(hf.num_experts),
        experts_per_token: to_u32(hf.num_experts_per_tok),
        moe_intermediate: to_u32(hf.moe_intermediate_size),
        shared_intermediate: to_u32(hf.shared_expert_intermediate_size),
        intermediate: to_u32(hf.intermediate_size),
        vocab: to_u32(hf.vocab_size),
        rope: RopeKind::Standard,
        norm_variant: NormVariant::Plain,
        norm_placement: if pre_norm { NormPlacement::Pre } else { NormPlacement::Post },
        qk_norm,
        fused_qkv,
        tied_embeddings: hf.tie_word_embeddings,
        qkv_bias: hf.attention_bias,
    };
    let cuda = LlamaLikeCudaFacts {
        xqa_decode: false,
        decode_fused_post: false,
        rope_table: true,
        force_prefill_path: false,
        head_dim_padded: hf.head_dim != hf.head_dim_kernel,
        // The padded width itself, from the same place the flag reads.
        head_dim_kernel: to_u32(hf.head_dim_kernel),
        gate_up_fused: true,
        // The shell's own frame: one GPU, no collectives, bf16
        // checkpoints. `window_left` empty reads as "no window", which is
        // what this assembly meant before the declaration carried one —
        // the shell derives its own per-layer windows from
        // `hf.sliding_window` where a family has them, and that path is
        // unchanged.
        proj_repr: model_compiler::dsl::WeightRepr::Bf16,
        // The group the load sharded for. A rank whose weights are a band
        // of the real ones has to land its projections with a collective,
        // and this is where the trace learns that.
        tp_size: model.tp_size,
        window_left: Vec::new(),
        all_reduce_p2p_max_rows: 0,
    };
    Ok(Box::new((facts, cuda)))
}

