//! Trace names for the weights a checkpoint publishes.
//!
//! A family's forward asks for `layer.3.qkv`; its contract stages
//! `model.layers.3.self_attn.qkv_proj.fused.weight`. Both names are this
//! crate's — the DSL invents the first, the contract author the second — so
//! the map between them belongs here and not in a driver.
//!
//! It lived in `driver-cuda`'s shell, which meant a backend knew what a
//! gemma-4 per-layer projection is called. A second backend would have needed
//! the same 200 lines, and a family added here would have had to be added
//! there too.
//!
//! # Renames, not copies
//!
//! Almost every row is an ALIAS: two names for bytes that are already on the
//! device once. The exceptions are the fused banks, and even those are
//! renames now — `Projections::Fused` stages the join in the load plan, so
//! `layer.3.qkv` names the arena span the plan already laid out.
//!
//! [`Wiring::join`] is the one case that is neither: a checkpoint that ships
//! its projections pre-joined (Phi-3) has its contract SPLIT them, so there
//! is nothing for the fuser to join and the halves are merely adjacent. The
//! caller decides whether adjacent is good enough, because only it knows
//! where the bytes actually sit.
//!
//! # The second half: the MLX staging convention
//!
//! [`wire`] answers the HuggingFace side — trace name to the name a
//! `.safetensors` export wrote — and [`Names::mlx`] answers the MLX side,
//! which `driver-metal` stages and binds against. Both are maps between two
//! spellings this crate owns, and the rule above covers them equally.
//!
//! That half lived in `driver-metal/src/lowering/resolve.rs`, which meant
//! THAT backend knew what a gemma-4 per-layer projection, a gpt-oss attention
//! sink and a qwen3-moe expert bank are each called — the same 200 lines this
//! doc records `driver-cuda` having been relieved of. A third backend would
//! have needed a third copy.

use std::collections::HashMap;

use crate::catalog::LoadShape;

/// How a checkpoint spells what a text names.
///
/// Data rather than code: a family that spells its tensors differently is a
/// different spelling in this map, not a different resolver.
///
/// # Why a role has SEVERAL spellings
///
/// One role, one name was the earlier shape, and it forced a second map
/// (a second `Names` constructor) the moment a second convention appeared — and then
/// the driver had to CHOOSE which map, which is the driver choosing, which is
/// the one thing this crate may not do.
///
/// So a role names every spelling it has ever been seen under, and the
/// CHECKPOINT decides: the resolver takes the first candidate
/// the loaded tensor map actually publishes. Adding a convention is adding a
/// string, and no caller learns anything.
///
/// Three real disagreements this covers, all measured against checkpoints on
/// disk rather than guessed:
///
///   - `embed`/`lm_head`: `shared_embedding` when the deployment TIES them,
///     `embed_tokens` and `lm_head` when it does not (gpt-oss).
///   - the expert bank: `mlp.switch_mlp.*` (qwen3-moe),
///     `experts.switch_glu.*` (gemma4), `mlp.experts.*` (gpt-oss).
///   - the router: `mlp.gate` (qwen3-moe), `mlp.router` (gpt-oss).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Names {
    /// What a layer-scoped name gets in front of its index —
    /// `model.layers.` for a HuggingFace export.
    pub layer_prefix: String,
    /// The text's role name → the checkpoint's paths within a layer, in the
    /// order they are tried. `qkv` → `self_attn.qkv_proj`.
    pub roles: HashMap<String, Vec<String>>,
    /// Names with no layer at all — `embed`, `lm_head`, `final_norm`.
    pub globals: HashMap<String, Vec<String>>,
    /// What the checkpoint calls a packed weight's own tensor.
    ///
    /// `.weight` for a tensor that hangs under a module, and **the empty
    /// string** for one that IS the value: gpt-oss ships `self_attn.sinks`,
    /// not `self_attn.sinks.weight`, because a sink is a vector and not a
    /// linear layer.
    pub weight_suffix: Vec<String>,
    /// What it calls the zero point the text spells `.zeros`.
    ///
    /// `.biases`, which is MLX's affine quantisation and the only spelling
    /// any checkpoint here gives this plane.
    ///
    /// This used to also list `.bias`, "the same role one character apart"
    /// for the MXFP4 expert banks. It is not the same role. Measured on
    /// `mlx-community/gpt-oss-20b-MXFP4-Q4`: an expert bank publishes
    /// `weight`, `scales` and `bias` and no `biases`, and the bias is
    /// `[32, 2880]` — one value per output ROW, where the zero point beside
    /// `scales` would be `[32, 2880, 90]`, one per GROUP. It is the additive
    /// term `qmv_routed_bias` reads at `buffer(7)`, not the codec's plane at
    /// `buffer(2)`, and the MXFP4 codec genuinely has no plane at all.
    ///
    /// It was unreachable under this name, which is why the reading was never
    /// contradicted: `MatW::scale_names` emits `.zeros` only for an affine
    /// weight, and `.biases` answers every one of those first. See
    /// [`Self::bias_suffix`].
    pub zero_point_suffix: Vec<String>,
    /// What it calls the additive bias the text spells `.bias`.
    ///
    /// One per output row, beside a routed expert bank's packed weight. Only
    /// `dsl::metal::routed_qmv` names it, and only for the symbols that read
    /// one.
    pub bias_suffix: Vec<String>,
}

impl Names {
    /// The convention [`crate::llama_3::contract`] publishes, which is what
    /// `stage_plan_weights` keys its map by.
    ///
    /// **Read off the contract, not guessed.** The lowering maps
    /// `model.layers.{l}.{member}` to `layers.{l}.{member}`,
    /// `model.embed_tokens.*` to `shared_embedding.*` when the deployment ties
    /// its embeddings, and `model.norm.weight` to `final_norm.weight`. An
    /// earlier draft of this map assumed the HuggingFace spelling and was
    /// wrong on all three — it was self-consistent, and the test that held the
    /// text against it passed, because both sides were this file.
    ///
    /// # The two names with no tensor
    ///
    /// `qkv` and `gate_up` are FUSED handles, and **no Metal deployment has
    /// them**: `compile_load_plan` authors with `Projections::InPlace`, and
    /// `dense_fused_projection_joins` returns before doing anything under that
    /// policy. So the MLX path publishes the three and two projections
    /// separately, which is also what `weight_binds` binds.
    ///
    /// They are still mapped, to the spelling a JOINING plan would publish —
    /// `…qkv_proj.fused.weight`, with the loader's own `.fused` infix — so that
    /// a deployment which does join resolves. The text asks the tensors which
    /// it is (`LlamaLikeFacts::fused_qkv`, `LlamaLikeMetalFacts::gate_up_fused`)
    /// rather than either side assuming.
    #[must_use]
    pub fn mlx() -> Self {
        let roles = [
            // The fused pair — see the note above. The `.fused` infix is the
            // loader's own: `dense_fused_projection_joins` publishes
            // `…qkv_proj.fused.weight`, not `…qkv_proj.weight`.
            ("qkv", "self_attn.qkv_proj.fused"),
            ("gate_up", "mlp.gate_up_proj.fused"),
            // The projections as the checkpoint ships them.
            ("q_proj", "self_attn.q_proj"),
            ("k_proj", "self_attn.k_proj"),
            ("v_proj", "self_attn.v_proj"),
            ("o_proj", "self_attn.o_proj"),
            // The Qwen-2 family's projection biases, which are ADDITIVE
            // vectors and not a quantizer's zero points.
            //
            // These name the same MODULE as the three entries above, because
            // that is what every value in this table is: a module, to which
            // the reader appends the tensor it wants. The bias is that
            // module's `.bias` tensor, and the role's trailing `_bias` -- an
            // UNDERSCORE, which is what keeps it apart from the `.bias`
            // sidecar `decompose` strips -- is what tells the reader to ask
            // for it. Spelling `self_attn.q_proj.bias` here instead reads as
            // a module and gets `.weight` appended, for
            // `self_attn.q_proj.bias.weight`, which no checkpoint publishes.
            //
            // The CUDA `Wiring` below does spell the full tensor, and that is
            // consistent for IT: a wiring builds finished aliases eagerly and
            // appends nothing. Two readers, two conventions, one pair of
            // strings -- worth saying out loud, because copying the wiring's
            // spelling into this table is exactly the mistake that was made.
            ("q_bias", "self_attn.q_proj"),
            ("k_bias", "self_attn.k_proj"),
            ("v_bias", "self_attn.v_proj"),
            // gpt-oss's landing bias, on the same module as `o_proj` for the
            // reason the three above share theirs.
            ("o_bias", "self_attn.o_proj"),
            // The DSL's own handle names (`Layer::gate_proj` / `up_proj`),
            // which is what the text spells.
            ("gate_proj", "mlp.gate_proj"),
            ("up_proj", "mlp.up_proj"),
            ("down", "mlp.down_proj"),
            // The mixture. `mlp.gate` is MLX's name for the ROUTER -- an
            // unfortunate collision with `mlp.gate_proj`, which is an
            // expert's gate half, and worth spelling out here because the two
            // are one character apart and mean entirely different tensors.
            //
            // The expert banks carry no expert index: `switch_mlp` stores all
            // of them in one `[experts, out, in]` tensor and the routed kernel
            // indexes it by the slot it read.
            // Three conventions for one bank, and the checkpoint picks:
            // qwen3-moe's `switch_mlp`, gemma4's `switch_glu`, gpt-oss's
            // plain `experts`.
            // THREE conventions, and the third does not live under `mlp.` at
            // all: gemma-4's routed block is a sibling of the dense one, so
            // its router is `router.proj` where the bank is
            // `experts.switch_glu.*`. Measured on
            // `mlx-community/gemma-4-26b-a4b-it-4bit` -- every other name
            // that text states already resolved against that checkpoint, and
            // this role was the whole of the gap: 90 unpublished names, which
            // is thirty layers times the packed weight, its scales and its
            // zero point.
            ("router", "mlp.gate|mlp.router|router.proj"),
            // The router's own bias, under whichever of the two spellings
            // the checkpoint gave the router. One entry, both conventions,
            // because a checkpoint that renamed the module renamed its bias
            // with it.
            ("router_bias", "mlp.gate|mlp.router|router.proj"),
            (
                "expert_gate",
                "mlp.switch_mlp.gate_proj|experts.switch_glu.gate_proj|mlp.experts.gate_proj",
            ),
            (
                "expert_up",
                "mlp.switch_mlp.up_proj|experts.switch_glu.up_proj|mlp.experts.up_proj",
            ),
            (
                "expert_down",
                "mlp.switch_mlp.down_proj|experts.switch_glu.down_proj|mlp.experts.down_proj",
            ),
            ("shared_gate", "mlp.shared_expert.gate_proj"),
            ("shared_up", "mlp.shared_expert.up_proj"),
            ("shared_down", "mlp.shared_expert.down_proj"),
            ("shared_gate_proj", "mlp.shared_expert_gate"),
            // The norms.
            // gemma's per-layer embedding network: a second table, its
            // projection and norm, and the per-layer gate and output.
            ("ple_gate", "per_layer_gate"),
            ("ple_out", "per_layer_projection"),
            // `layer_scalar`, measured against
            // `mlx-community/gemma-4-26b-a4b-it-4bit`'s index and NOT the
            // `per_layer_scalar` the role is called. A role name and a
            // checkpoint name are two different things, which is what this map
            // is for.
            ("scalar", "layer_scalar"),
            // The attention sink, one learned logit per head.
            ("attn_sinks", "self_attn.sinks"),
            ("q_norm", "self_attn.q_norm"),
            ("k_norm", "self_attn.k_norm"),
            ("attn_norm", "input_layernorm"),
            // TWO spellings, and which one a checkpoint means is decided by
            // which one it ships. Under `NormPlacement::Pre` the pre-FFN norm
            // IS `post_attention_layernorm` — it sits after the attention and
            // before the MLP, and llama publishes nothing else. gemma splits
            // that position in two (`post_attention_layernorm` norms the
            // attention's OUTPUT, `pre_feedforward_layernorm` norms the MLP's
            // input), so it must take the second.
            //
            // Ordered gemma-first because the alternative resolves to the
            // first spelling the checkpoint HAS: a llama checkpoint ships no
            // `pre_feedforward_layernorm` and falls through, and a gemma one
            // would otherwise bind its attention-output norm as the MLP's
            // input norm and drop two norms entirely.
            (
                "mlp_norm",
                "pre_feedforward_layernorm|post_attention_layernorm",
            ),
            // The SANDWICH's output norms, which only a gemma text names.
            ("post_attn_norm", "post_attention_layernorm"),
            ("post_mlp_norm", "post_feedforward_layernorm"),
            // ── the MIXTURE layer's three EXTRA norms. ──
            //
            // gemma-4's MoE rows run two FFNs SIDE BY SIDE and norm each
            // leg's output before joining them, so a mixture layer ships
            // SEVEN norms where a dense one ships four. The suffixed pair
            // belongs to the legs -- `_1` the dense one, `_2` the routed one
            // -- and the unsuffixed `post_feedforward_layernorm` above norms
            // their SUM, which is why it keeps its name and its role.
            //
            // Measured off `mlx-community/gemma-4-26b-a4b-it-4bit`, whose
            // layer 0 publishes all seven. A dense gemma-4 ships none of
            // these three and resolves them to nothing, which is right: a
            // layer with one FFN has no leg to norm separately.
            ("post_mlp_norm_1", "post_feedforward_layernorm_1"),
            ("mlp_norm_2", "pre_feedforward_layernorm_2"),
            ("post_mlp_norm_2", "post_feedforward_layernorm_2"),
            // ── the ROUTER's two scales, which are not its quantisation. ──
            //
            // `router.scale` is an RMS-norm weight `[hidden]`: the router
            // norms its input before projecting, at its OWN scale rather
            // than the leg's. `router.per_expert_scale` is `[n_experts]` and
            // multiplies the post-softmax weights.
            //
            // Neither takes a `.weight` suffix in the checkpoint, which
            // costs nothing here because `weight_suffix` tries the bare name
            // too. Distinct from `router.proj.scales`, which IS quantisation
            // and reaches the map as the `router` role's own affine point.
            ("router_scale", "router.scale"),
            ("router_expert_scale", "router.per_expert_scale"),
        ]
        .into_iter()
        .map(|(a, b): (&str, &str)| (a.to_string(), b.split('|').map(str::to_string).collect()))
        .collect();
        let globals = [
            // Tied: one table serves both ends, which is why the readout and
            // the embedding answer to the same name.
            // Tied deployments publish ONE table under `shared_embedding`;
            // untied ones (gpt-oss) publish `embed_tokens` and `lm_head`
            // separately. Which is which is the checkpoint's to say.
            ("embed", "shared_embedding|embed_tokens"),
            // gemma's SECOND embedding table and its projection: layer-less,
            // gathered once per step, so they are globals rather than a
            // layer's.
            ("ple_embed", "per_layer_embedding"),
            ("ple_proj", "per_layer_input_projection"),
            ("ple_proj_norm", "per_layer_input_norm"),
            ("lm_head", "shared_embedding|lm_head"),
            ("final_norm", "final_norm"),
        ]
        .into_iter()
        .map(|(a, b): (&str, &str)| (a.to_string(), b.split('|').map(str::to_string).collect()))
        .collect();
        Self {
            layer_prefix: "layers.".to_string(),
            roles,
            globals,
            // `.weight` first, then the bare name: a role whose tensor IS
            // the value (`self_attn.sinks`) hangs under no module.
            weight_suffix: vec![".weight".to_string(), String::new()],
            // The text says `.zeros`, the checkpoint says `.biases` -- or
            // `.bias`, for the MXFP4 expert banks. Both are right on their own
            // side; the map is where they meet.
            zero_point_suffix: vec![".biases".to_string()],
            bias_suffix: vec![".bias".to_string()],
        }
    }
}

/// What a caller offers this module, and what it gets back.
pub struct Wiring<'a> {
    /// The names the load plan published. Asked, never modified.
    pub published: &'a dyn Fn(&str) -> bool,
    /// Trace name -> published name. The answer.
    pub aliases: Vec<(String, String)>,
    /// Trace name -> the published names whose bytes it spans, in order.
    /// A caller that can prove they are contiguous may name the span; one
    /// that cannot must skip the row. See the module doc.
    pub joins: Vec<(String, Vec<String>)>,
    /// Published names holding a load-time host scalar (gemma-4's
    /// `layer_scalar`), in layer order.
    pub scalars: Vec<String>,
    /// The row's shape.
    ///
    /// Two fields are read — the layer count and how many trailing
    /// layers share KV — and both used to come off a resident
    /// `HfConfig`, which is why this module took one. A `LoadShape` is
    /// the same two numbers stated by the ROW, so the names a trace asks
    /// for and the stack a driver fired come from one place.
    pub shape: LoadShape,
}

impl<'a> Wiring<'a> {
    /// A fresh wiring over the names a plan published.
    pub fn new(shape: LoadShape, published: &'a dyn Fn(&str) -> bool) -> Self {
        Self {
            published,
            aliases: Vec::new(),
            joins: Vec::new(),
            scalars: Vec::new(),
            shape,
        }
    }

    fn has(&self, name: &str) -> bool {
        (self.published)(name)
    }

    /// Record `trace` as another name for `published`, if it exists.
    fn alias(&mut self, trace: String, published: String) {
        if self.has(&published) {
            self.aliases.push((trace, published));
        }
    }

    /// Has `trace` already been answered?
    fn named(&self, trace: &str) -> bool {
        self.aliases.iter().any(|(t, _)| t == trace) || self.joins.iter().any(|(t, _)| t == trace)
    }

    /// Record `trace` as the concatenation of `parts`, if all exist.
    fn join(&mut self, trace: String, parts: &[String]) {
        if parts.iter().all(|p| self.has(p)) {
            self.joins.push((trace, parts.to_vec()));
        }
    }
}

/// Every trace name this checkpoint can answer, for whichever family it is.
///
/// The families are tried in turn and each recognises itself by a tensor only
/// it ships — gemma-4 by its per-layer embedding table, qwen3.5 by the VL
/// prefix without that table, llama-like by plain `model.embed_tokens`. A
/// checkpoint no family claims comes back empty, and a launch asking for a
/// trace name then gets the resolver's refusal, which is the honest state.
#[must_use]
pub fn wire<'a>(shape: LoadShape, published: &'a dyn Fn(&str) -> bool) -> Wiring<'a> {
    let mut w = Wiring::new(shape, published);
    llama_like(&mut w);
    gpt_oss(&mut w);
    gemma4(&mut w);
    qwen3_5(&mut w);
    w
}

fn llama_like(w: &mut Wiring<'_>) {
    if !w.has("model.embed_tokens.weight") {
        return; // not an HF llama-like naming scheme; leave raw
    }

    w.alias("embed".into(), "model.embed_tokens.weight".into());
    w.alias("final_norm".into(), "model.norm.weight".into());
    if w.has("lm_head.weight") {
        w.alias("lm_head".into(), "lm_head.weight".into());
    } else {
        // Tied embeddings: the trace's lm_head name IS "embed".
        w.alias("lm_head".into(), "model.embed_tokens.weight".into());
    }
    let layers = w.shape.layers as usize;
    for i in 0..layers {
        let n = |s: &str| format!("model.layers.{i}.{s}");
        // THE PLAN ALREADY JOINED THESE. `Projections::Fused` stages
        // `…qkv_proj.fused.weight` and `…gate_up_proj.fused.weight`
        // beside the split tensors, laid out in the arena in the order
        // the GEMM wants — so the trace name is a RENAME, where the
        // driver used to read three tensors back off the device and
        // upload their concatenation.
        w.alias(
            format!("layer.{i}.qkv"),
            n("self_attn.qkv_proj.fused.weight"),
        );
        w.alias(
            format!("layer.{i}.gate_up"),
            n("mlp.gate_up_proj.fused.weight"),
        );
        // Some checkpoints ship the fused projections ALREADY (phi3's
        // `qkv_proj` and `gate_up_proj`), in the same concatenation order
        // the fuse above builds. Those want an alias, not a copy -- and
        // `alias` is a no-op when the name is absent, so this costs the
        // deployments that split nothing.
        w.alias(format!("layer.{i}.qkv"), n("self_attn.qkv_proj.weight"));
        w.alias(format!("layer.{i}.gate_up"), n("mlp.gate_up_proj.weight"));
        // …and the third case: the parts abut in the arena but were never
        // given a joined name. See `name_contiguous_join`.
        if !w.named(&format!("layer.{i}.qkv")) {
            w.join(
                format!("layer.{i}.qkv"),
                &[
                    n("self_attn.q_proj.weight"),
                    n("self_attn.k_proj.weight"),
                    n("self_attn.v_proj.weight"),
                ],
            );
        }
        if !w.named(&format!("layer.{i}.gate_up")) {
            w.join(
                format!("layer.{i}.gate_up"),
                &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
            );
        }
        // THREE placements, and the discriminant is not what it looks
        // like. `input_layernorm`'s presence tells pre-norm from post-norm
        // and nothing else — a SANDWICH family publishes it too, so a
        // two-way branch on it sends gemma-2 and gemma3n down the pre-norm
        // arm and binds their `mlp_norm` to `post_attention_layernorm`
        // where the forward means `pre_feedforward_layernorm`.
        //
        // That is the failure `seam_names` cannot catch: a name that
        // resolves to the WRONG TENSOR rather than to nothing. Every
        // resolver answers, every shape matches, and the model is
        // slightly worse.
        //
        // So the sandwich is tested FIRST, by the tensor only it ships.
        // Its four norms are the four its forward names — before and after
        // both blocks — and none of them is a guess: the checkpoint either
        // publishes `pre_feedforward_layernorm` or it does not, and if it
        // does, the placement is unambiguous.
        //
        //   sandwich   attn=input   post_attn=post_attention
        //              mlp=pre_feedforward   post_mlp=post_feedforward
        //   pre-norm   attn=input   mlp=post_attention
        //   post-norm  attn=post_attention   mlp=post_feedforward  (olmo2,
        //              the bind_olmo3 convention the A/B verified)
        if w.has(&n("pre_feedforward_layernorm.weight")) {
            w.alias(format!("layer.{i}.attn_norm"), n("input_layernorm.weight"));
            w.alias(
                format!("layer.{i}.post_attn_norm"),
                n("post_attention_layernorm.weight"),
            );
            w.alias(
                format!("layer.{i}.mlp_norm"),
                n("pre_feedforward_layernorm.weight"),
            );
            w.alias(
                format!("layer.{i}.post_mlp_norm"),
                n("post_feedforward_layernorm.weight"),
            );
        } else if w.has(&n("input_layernorm.weight")) {
            w.alias(format!("layer.{i}.attn_norm"), n("input_layernorm.weight"));
            w.alias(
                format!("layer.{i}.mlp_norm"),
                n("post_attention_layernorm.weight"),
            );
        } else {
            w.alias(
                format!("layer.{i}.attn_norm"),
                n("post_attention_layernorm.weight"),
            );
            w.alias(
                format!("layer.{i}.mlp_norm"),
                n("post_feedforward_layernorm.weight"),
            );
        }
        for (trace, ckpt) in [
            ("q_norm", "self_attn.q_norm.weight"),
            ("k_norm", "self_attn.k_norm.weight"),
            ("o_proj", "self_attn.o_proj.weight"),
            ("down", "mlp.down_proj.weight"),
            ("q_proj", "self_attn.q_proj.weight"),
            ("k_proj", "self_attn.k_proj.weight"),
            ("v_proj", "self_attn.v_proj.weight"),
            ("q_bias", "self_attn.q_proj.bias"),
            ("k_bias", "self_attn.k_proj.bias"),
            ("v_bias", "self_attn.v_proj.bias"),
        ] {
            w.alias(format!("layer.{i}.{trace}"), n(ckpt));
        }
    }
}

/// Build the gpt-oss trace names that [`llama_like`] does not reach.
///
/// gpt-oss IS an HF llama-like checkpoint — `model.embed_tokens.weight`,
/// `input_layernorm`, `self_attn.{q,k,v,o}_proj` — so `llama_like` above
/// answers its backbone and this adds only the five names it does not:
/// the sigmoid router and its bias, the attention sinks, and the two MXFP4
/// expert banks.
///
/// **This is the family that BITES.** It is the only one with both a
/// `FACTS_ROWS` entry in the CUDA shell and a Prefill arm, so before this
/// builder a gpt-oss checkpoint loaded, reported itself healthy, and died
/// at its first fire on `UnknownWeight("layer.0.router")` — one name at a
/// time, at request time. `tests/seam_names.rs` is what turned that into
/// a list.
///
/// # The bank suffixes, and why they are not a convention invented here
///
/// `quant::mxfp4_moe_gate_up_decode_bf16` resolves `{bank}_scales`,
/// `{bank}_gate_bias` and `{bank}_up_bias` off the bank the statement
/// names; the down twin resolves `{bank}_scales` and `{bank}_bias`. Those
/// are the DRIVER's spellings and the trace never states them, so they
/// have to be answered here or the arm refuses.
///
/// The two GATE/UP biases are the interesting pair. The routed contract
/// publishes one fused `gate_up_proj.bias` — gate at even rows, up at odd
/// — which is a STRIDE and not a rename, and `wire()` can only alias or
/// join. The native path publishes the two halves separately, already
/// split. So both spellings are stated and `Wiring::alias`'s
/// record-only-if-published does the right thing without a branch: on the
/// native path they resolve, on the routed path they are absent and the
/// arm's `unwrap_or(null)` takes over, which is what it is written for.
/// That behaviour is the silent-failure hazard everywhere else in this
/// file and is exactly correct here.
fn gpt_oss(w: &mut Wiring<'_>) {
    // The sinks ARE the family's signature: no other llama-like
    // checkpoint ships a per-head attention sink beside its projections.
    if !w.has("model.layers.0.self_attn.sinks") {
        return;
    }
    let layers = w.shape.layers as usize;
    for i in 0..layers {
        let n = |s: &str| format!("model.layers.{i}.{s}");
        w.alias(format!("layer.{i}.router"), n("mlp.router.weight"));
        w.alias(format!("layer.{i}.router_bias"), n("mlp.router.bias"));
        w.alias(format!("layer.{i}.attn_sinks"), n("self_attn.sinks"));

        let gate_up = format!("layer.{i}.expert_gate_up_bank");
        let experts = n("mlp.experts");
        w.alias(gate_up.clone(), format!("{experts}.gate_up_proj.weight"));
        w.alias(
            format!("{gate_up}_scales"),
            format!("{experts}.gate_up_proj.weight_scale"),
        );
        w.alias(
            format!("{gate_up}_gate_bias"),
            format!("{experts}.gate_proj.bias"),
        );
        w.alias(
            format!("{gate_up}_up_bias"),
            format!("{experts}.up_proj.bias"),
        );

        let down = format!("layer.{i}.expert_down_bank");
        w.alias(down.clone(), format!("{experts}.down_proj.weight"));
        w.alias(
            format!("{down}_scales"),
            format!("{experts}.down_proj.weight_scale"),
        );
        w.alias(format!("{down}_bias"), format!("{experts}.down_proj.bias"));
    }
}

/// Build the qwen3_5 hybrid's trace names — the `real_hybrid` A/B's
/// binder vocabulary, promoted into the shell. The checkpoint naming is
/// the VL config's (`model.language_model.*`); the vision tower and the
/// MTP block stay untouched under their raw names.
/// Build the gemma-4 trace names beside the checkpoint names —
/// `gemma4.cpp`'s binder plus the engine's `dense_fused_projection_joins`
/// (q‖k‖v on the layers that project their own KV, gate‖up everywhere),
/// as the real-weight A/B proved them. Also reads the per-layer
/// `layer_scalar` [1] tensors to host — the load-time
/// `read_bf16_scalar_once`, stashed for the fire's `scales` map.
#[allow(clippy::too_many_lines)]
fn gemma4(w: &mut Wiring<'_>) {
    let p = "model.language_model";
    if !w.has(&format!("{p}.embed_tokens_per_layer.weight")) {
        return; // the PLE table IS the family's signature
    }
    w.alias("embed".into(), format!("{p}.embed_tokens.weight"));
    w.alias(
        "embed_per_layer".into(),
        format!("{p}.embed_tokens_per_layer.weight"),
    );
    w.alias(
        "ple_model_proj".into(),
        format!("{p}.per_layer_model_projection.weight"),
    );
    w.alias(
        "ple_model_norm".into(),
        format!("{p}.per_layer_projection_norm.weight"),
    );
    w.alias("final_norm".into(), format!("{p}.norm.weight"));
    let layers = w.shape.layers as usize;
    let first_shared = layers.saturating_sub(w.shape.kv_shared_layers as usize);
    for i in 0..layers {
        let n = |sfx: &str| format!("{p}.layers.{i}.{sfx}");
        w.alias(format!("layer.{i}.attn_norm"), n("input_layernorm.weight"));
        w.alias(
            format!("layer.{i}.post_attn_norm"),
            n("post_attention_layernorm.weight"),
        );
        w.alias(
            format!("layer.{i}.pre_ffw_norm"),
            n("pre_feedforward_layernorm.weight"),
        );
        w.alias(
            format!("layer.{i}.post_ffw_norm"),
            n("post_feedforward_layernorm.weight"),
        );
        w.alias(format!("layer.{i}.q_norm"), n("self_attn.q_norm.weight"));
        w.alias(format!("layer.{i}.o_proj"), n("self_attn.o_proj.weight"));
        w.alias(format!("layer.{i}.down"), n("mlp.down_proj.weight"));
        w.alias(
            format!("layer.{i}.ple_gate"),
            n("per_layer_input_gate.weight"),
        );
        w.alias(
            format!("layer.{i}.ple_proj"),
            n("per_layer_projection.weight"),
        );
        w.alias(
            format!("layer.{i}.ple_norm"),
            n("post_per_layer_input_norm.weight"),
        );
        if i >= first_shared {
            // A KV-shared layer states only the Q leg.
            w.alias(format!("layer.{i}.q_proj"), n("self_attn.q_proj.weight"));
        } else {
            w.alias(format!("layer.{i}.k_norm"), n("self_attn.k_norm.weight"));
            w.alias(
                format!("layer.{i}.qkv"),
                n("self_attn.qkv_proj.fused.weight"),
            );
        }
        w.alias(
            format!("layer.{i}.gate_up"),
            n("mlp.gate_up_proj.fused.weight"),
        );
        // The per-layer `layer_scalar` [1] tensors the fused sandwich norm
        // multiplies the whole stream by. NAMED here and READ by the caller:
        // a host read of a device tensor is the driver's business, and which
        // tensors carry the scalars is the family's.
        w.scalars.push(n("layer_scalar"));
    }
}

fn qwen3_5(w: &mut Wiring<'_>) {
    let p = "model.language_model";
    if !w.has(&format!("{p}.embed_tokens.weight")) {
        return; // not the qwen3_5 naming scheme; leave raw
    }
    if w.has(&format!("{p}.embed_tokens_per_layer.weight")) {
        return; // gemma-4 shares the prefix; its aliases are its own
    }
    w.alias("embed".into(), format!("{p}.embed_tokens.weight"));
    w.alias("final_norm".into(), format!("{p}.norm.weight"));
    let layers = w.shape.layers as usize;
    for i in 0..layers {
        let n = |sfx: &str| format!("{p}.layers.{i}.{sfx}");
        w.alias(format!("layer.{i}.attn_norm"), n("input_layernorm.weight"));
        w.alias(
            format!("layer.{i}.mlp_norm"),
            n("post_attention_layernorm.weight"),
        );
        w.alias(format!("layer.{i}.down"), n("mlp.down_proj.weight"));
        // FULL ATTENTION OR LINEAR, asked of the checkpoint.
        //
        // This read a `layer_types` list off the config — a per-layer
        // string array the derivation carried purely so this loop could
        // index it. A full-attention layer ships `self_attn.q_proj` and
        // a linear one does not, so the tensors already say which this
        // is, and saying it twice is how the two answers came apart.
        let full = w.has(&n("self_attn.q_proj.weight"));
        if full {
            for f in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
                w.alias(
                    format!("layer.{i}.{f}"),
                    n(&format!("self_attn.{f}.weight")),
                );
            }
        } else {
            for f in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"] {
                w.alias(
                    format!("layer.{i}.{f}"),
                    n(&format!("linear_attn.{f}.weight")),
                );
            }
            w.alias(format!("layer.{i}.conv"), n("linear_attn.conv1d.weight"));
            w.alias(format!("layer.{i}.a_log"), n("linear_attn.A_log"));
            w.alias(format!("layer.{i}.dt_bias"), n("linear_attn.dt_bias"));
            w.alias(format!("layer.{i}.gate_norm"), n("linear_attn.norm.weight"));
            w.alias(
                format!("layer.{i}.o_proj"),
                n("linear_attn.out_proj.weight"),
            );
        }
        // The fused gate‖up bank, gate first — the dense MLP's binding,
        // laid out by the plan.
        w.alias(
            format!("layer.{i}.gate_up"),
            n("mlp.gate_up_proj.fused.weight"),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Which checkpoint tensor a trace name resolves to.
    fn bound(published: &[&str], trace: &str) -> Option<String> {
        let shape = LoadShape::dense(1, 128, false);
        let set: std::collections::BTreeSet<String> =
            published.iter().map(|s| (*s).to_string()).collect();
        let has = |n: &str| set.contains(n);
        let w = wire(shape, &has);
        w.aliases
            .iter()
            .find(|(t, _)| t == trace)
            .map(|(_, c)| c.clone())
    }

    fn base() -> Vec<&'static str> {
        vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ]
    }

    /// THE FAILURE `seam_names` CANNOT SEE: a name that resolves to the
    /// WRONG TENSOR rather than to nothing.
    ///
    /// A sandwich family — gemma-2, gemma3n — norms before AND after both
    /// blocks, four per layer, and publishes `input_layernorm` like any
    /// pre-norm checkpoint does. A branch keyed on THAT sent it down the
    /// pre-norm arm, where `mlp_norm` binds `post_attention_layernorm`
    /// while the forward means `pre_feedforward_layernorm`. Every
    /// resolver answers, every shape matches, and the model is slightly
    /// worse — which is the one failure worth a test that reads the
    /// TARGET and not just the coverage.
    #[test]
    fn a_sandwich_checkpoint_binds_its_four_norms_to_four_tensors() {
        let mut p = base();
        p.extend([
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.pre_feedforward_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
        ]);
        let at = |t: &str| bound(&p, t);
        assert_eq!(
            at("layer.0.attn_norm").as_deref(),
            Some("model.layers.0.input_layernorm.weight")
        );
        assert_eq!(
            at("layer.0.post_attn_norm").as_deref(),
            Some("model.layers.0.post_attention_layernorm.weight")
        );
        assert_eq!(
            at("layer.0.mlp_norm").as_deref(),
            Some("model.layers.0.pre_feedforward_layernorm.weight"),
            "the MLP's PRE norm, not the attention's post — the defect this \
             test exists for"
        );
        assert_eq!(
            at("layer.0.post_mlp_norm").as_deref(),
            Some("model.layers.0.post_feedforward_layernorm.weight")
        );
    }

    /// And the two placements that already worked keep working — the
    /// sandwich branch is tested FIRST, so it has to decline for both.
    #[test]
    fn pre_norm_and_post_norm_are_unchanged() {
        let mut pre = base();
        pre.extend([
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ]);
        assert_eq!(
            bound(&pre, "layer.0.attn_norm").as_deref(),
            Some("model.layers.0.input_layernorm.weight")
        );
        assert_eq!(
            bound(&pre, "layer.0.mlp_norm").as_deref(),
            Some("model.layers.0.post_attention_layernorm.weight"),
            "a pre-norm checkpoint has no pre_feedforward norm to mean instead"
        );

        // olmo2: no `input_layernorm` at all.
        let mut post = base();
        post.extend([
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
        ]);
        assert_eq!(
            bound(&post, "layer.0.attn_norm").as_deref(),
            Some("model.layers.0.post_attention_layernorm.weight")
        );
        assert_eq!(
            bound(&post, "layer.0.mlp_norm").as_deref(),
            Some("model.layers.0.post_feedforward_layernorm.weight")
        );
    }

    /// Which trace names come back as a JOIN of several tensors.
    fn joined(published: &[&str], trace: &str) -> Option<Vec<String>> {
        let shape = LoadShape::dense(1, 128, false);
        let set: std::collections::BTreeSet<String> =
            published.iter().map(|s| (*s).to_string()).collect();
        let has = |n: &str| set.contains(n);
        let w = wire(shape, &has);
        w.joins
            .iter()
            .find(|(t, _)| t == trace)
            .map(|(_, parts)| parts.clone())
    }

    /// A join is recorded only when every part is there.
    ///
    /// The three legs abut in the arena, so `qkv` can be named as their
    /// concatenation without a copy -- but only if all three were
    /// published. A checkpoint missing one has no contiguous run to name,
    /// and recording the join anyway would hand the binder a name that
    /// resolves to two legs and whatever tensor the allocator put third.
    /// That is not a load error; it is an attention block reading someone
    /// else's weights.
    ///
    /// Nothing is put in its place, which is the honest state: the launch
    /// asking for `layer.0.qkv` gets the resolver's refusal, naming the
    /// tensor rather than producing numbers.
    #[test]
    fn a_join_needs_every_part_and_records_nothing_without_them() {
        assert_eq!(
            joined(&base(), "layer.0.qkv").as_deref(),
            Some(
                [
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.k_proj.weight",
                    "model.layers.0.self_attn.v_proj.weight",
                ]
                .map(String::from)
                .as_slice()
            ),
            "all three legs present names the run"
        );

        let no_v: Vec<&str> = base()
            .into_iter()
            .filter(|n| !n.ends_with("v_proj.weight"))
            .collect();
        assert_eq!(
            joined(&no_v, "layer.0.qkv"),
            None,
            "two of three legs names nothing"
        );
        assert_eq!(
            bound(&no_v, "layer.0.qkv"),
            None,
            "and no alias stands in for it either"
        );
        // The join that still has all its parts is unaffected.
        assert!(joined(&no_v, "layer.0.gate_up").is_some());
    }

    /// gemma-4's tail attends an earlier layer's KV, so it states one leg.
    ///
    /// `kv_shared_layers` counts back from the END. A shared layer has no
    /// K or V of its own -- no `k_norm`, and no q‖k‖v run to name -- so it
    /// binds `q_proj` alone. Wiring it like an ordinary layer asks for
    /// three tensors the checkpoint does not carry for those layers, and
    /// the load fails naming a weight gemma-4 is not supposed to have.
    ///
    /// The count is read off the row rather than sniffed, so the layers
    /// BELOW the cut must keep the ordinary wiring in the same
    /// checkpoint; a boundary read one layer out is the failure this
    /// pins.
    #[test]
    fn gemma4s_kv_shared_tail_states_only_its_q_leg() {
        const P: &str = "model.language_model";
        let mut published: Vec<String> = vec![
            format!("{P}.embed_tokens_per_layer.weight"),
            format!("{P}.embed_tokens.weight"),
        ];
        for i in 0..4 {
            published.push(format!("{P}.layers.{i}.self_attn.q_proj.weight"));
            published.push(format!("{P}.layers.{i}.self_attn.qkv_proj.fused.weight"));
            published.push(format!("{P}.layers.{i}.self_attn.k_norm.weight"));
        }
        let set: std::collections::BTreeSet<String> = published.into_iter().collect();
        let has = |n: &str| set.contains(n);

        let shape = LoadShape {
            layers: 4,
            kv_shared_layers: 2,
            ..LoadShape::dense(4, 128, false)
        };
        let w = wire(shape, &has);
        let at = |t: &str| {
            w.aliases
                .iter()
                .find(|(n, _)| n == t)
                .map(|(_, c)| c.clone())
        };

        for i in 0..4 {
            assert_eq!(
                at(&format!("layer.{i}.q_proj")).is_some(),
                i >= 2,
                "layer {i} states its q leg alone only in the shared tail"
            );
            assert_eq!(
                at(&format!("layer.{i}.k_norm")).is_some(),
                i < 2,
                "layer {i} has a k_norm only while it projects its own KV"
            );
            assert_eq!(
                at(&format!("layer.{i}.qkv")).is_some(),
                i < 2,
                "layer {i} names a q‖k‖v run only while it has one"
            );
        }
    }
}
