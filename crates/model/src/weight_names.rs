//! Trace names for the weights a checkpoint publishes.
//!
//! A family's forward asks for `layer.3.qkv`; its contract stages
//! `model.layers.3.self_attn.qkv_proj.fused.weight`. Both names are this
//! crate's — the DSL invents the first, the contract author the second — so
//! the map between them belongs here and not in a driver.
//!
//! It lived in `driver-cuda-new`'s shell, which meant a backend knew what a
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

use crate::config::HfConfig;

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
    /// The checkpoint's own shape.
    pub facts: &'a HfConfig,
}

impl<'a> Wiring<'a> {
    /// A fresh wiring over the names a plan published.
    pub fn new(facts: &'a HfConfig, published: &'a dyn Fn(&str) -> bool) -> Self {
        Self {
            published,
            aliases: Vec::new(),
            joins: Vec::new(),
            scalars: Vec::new(),
            facts,
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
        self.aliases.iter().any(|(t, _)| t == trace)
            || self.joins.iter().any(|(t, _)| t == trace)
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
pub fn wire<'a>(facts: &'a HfConfig, published: &'a dyn Fn(&str) -> bool) -> Wiring<'a> {
    let mut w = Wiring::new(facts, published);
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
    let layers = usize::try_from(w.facts.num_hidden_layers).unwrap_or(0);
    for i in 0..layers {
        let n = |s: &str| format!("model.layers.{i}.{s}");
        // THE PLAN ALREADY JOINED THESE. `Projections::Fused` stages
        // `…qkv_proj.fused.weight` and `…gate_up_proj.fused.weight`
        // beside the split tensors, laid out in the arena in the order
        // the GEMM wants — so the trace name is a RENAME, where the
        // driver used to read three tensors back off the device and
        // upload their concatenation.
        w.alias(format!("layer.{i}.qkv"), n("self_attn.qkv_proj.fused.weight"));
        w.alias(format!("layer.{i}.gate_up"), n("mlp.gate_up_proj.fused.weight"));
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
            w.join(format!("layer.{i}.qkv"), &[
                n("self_attn.q_proj.weight"),
                n("self_attn.k_proj.weight"),
                n("self_attn.v_proj.weight"),
            ]);
        }
        if !w.named(&format!("layer.{i}.gate_up")) {
            w.join(format!("layer.{i}.gate_up"), &[
                n("mlp.gate_proj.weight"),
                n("mlp.up_proj.weight"),
            ]);
        }
        // The norm placement decides the mapping, and `input_layernorm`'s
        // presence IS the placement: pre-norm has it (attn_norm=input,
        // mlp_norm=post_attention); post-norm (olmo2) lacks it
        // (attn_norm=post_attention, mlp_norm=post_feedforward) — the
        // bind_olmo3 convention the A/B verified.
        if w.has(&n("input_layernorm.weight")) {
            w.alias(format!("layer.{i}.attn_norm"), n("input_layernorm.weight"));
            w.alias(format!("layer.{i}.mlp_norm"), n("post_attention_layernorm.weight"));
        } else {
            w.alias(format!("layer.{i}.attn_norm"), n("post_attention_layernorm.weight"));
            w.alias(format!("layer.{i}.mlp_norm"), n("post_feedforward_layernorm.weight"));
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
    let layers = usize::try_from(w.facts.num_hidden_layers).unwrap_or(0);
    for i in 0..layers {
        let n = |s: &str| format!("model.layers.{i}.{s}");
        w.alias(format!("layer.{i}.router"), n("mlp.router.weight"));
        w.alias(format!("layer.{i}.router_bias"), n("mlp.router.bias"));
        w.alias(format!("layer.{i}.attn_sinks"), n("self_attn.sinks"));

        let gate_up = format!("layer.{i}.expert_gate_up_bank");
        let experts = n("mlp.experts");
        w.alias(gate_up.clone(), format!("{experts}.gate_up_proj.weight"));
        w.alias(format!("{gate_up}_scales"), format!("{experts}.gate_up_proj.weight_scale"));
        w.alias(format!("{gate_up}_gate_bias"), format!("{experts}.gate_proj.bias"));
        w.alias(format!("{gate_up}_up_bias"), format!("{experts}.up_proj.bias"));

        let down = format!("layer.{i}.expert_down_bank");
        w.alias(down.clone(), format!("{experts}.down_proj.weight"));
        w.alias(format!("{down}_scales"), format!("{experts}.down_proj.weight_scale"));
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
    w.alias("embed_per_layer".into(), format!("{p}.embed_tokens_per_layer.weight"));
    w.alias("ple_model_proj".into(), format!("{p}.per_layer_model_projection.weight"));
    w.alias("ple_model_norm".into(), format!("{p}.per_layer_projection_norm.weight"));
    w.alias("final_norm".into(), format!("{p}.norm.weight"));
    let layers = usize::try_from(w.facts.num_hidden_layers).unwrap_or(0);
    let first_shared =
        layers.saturating_sub(usize::try_from(w.facts.num_kv_shared_layers).unwrap_or(0));
    for i in 0..layers {
        let n = |sfx: &str| format!("{p}.layers.{i}.{sfx}");
        w.alias(format!("layer.{i}.attn_norm"), n("input_layernorm.weight"));
        w.alias(format!("layer.{i}.post_attn_norm"), n("post_attention_layernorm.weight"));
        w.alias(format!("layer.{i}.pre_ffw_norm"), n("pre_feedforward_layernorm.weight"));
        w.alias(format!("layer.{i}.post_ffw_norm"), n("post_feedforward_layernorm.weight"));
        w.alias(format!("layer.{i}.q_norm"), n("self_attn.q_norm.weight"));
        w.alias(format!("layer.{i}.o_proj"), n("self_attn.o_proj.weight"));
        w.alias(format!("layer.{i}.down"), n("mlp.down_proj.weight"));
        w.alias(format!("layer.{i}.ple_gate"), n("per_layer_input_gate.weight"));
        w.alias(format!("layer.{i}.ple_proj"), n("per_layer_projection.weight"));
        w.alias(format!("layer.{i}.ple_norm"), n("post_per_layer_input_norm.weight"));
        if i >= first_shared {
            // A KV-shared layer states only the Q leg.
            w.alias(format!("layer.{i}.q_proj"), n("self_attn.q_proj.weight"));
        } else {
            w.alias(format!("layer.{i}.k_norm"), n("self_attn.k_norm.weight"));
            w.alias(format!("layer.{i}.qkv"), n("self_attn.qkv_proj.fused.weight"));
        }
        w.alias(format!("layer.{i}.gate_up"), n("mlp.gate_up_proj.fused.weight"));
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
    let layers = usize::try_from(w.facts.num_hidden_layers).unwrap_or(0);
    for i in 0..layers {
        let n = |sfx: &str| format!("{p}.layers.{i}.{sfx}");
        w.alias(format!("layer.{i}.attn_norm"), n("input_layernorm.weight"));
        w.alias(format!("layer.{i}.mlp_norm"), n("post_attention_layernorm.weight"));
        w.alias(format!("layer.{i}.down"), n("mlp.down_proj.weight"));
        let full = w
            .facts
            .layer_types
            .get(i)
            .is_some_and(|t| t == "full_attention");
        if full {
            for f in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
                w.alias(format!("layer.{i}.{f}"), n(&format!("self_attn.{f}.weight")));
            }
        } else {
            for f in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"] {
                w.alias(format!("layer.{i}.{f}"), n(&format!("linear_attn.{f}.weight")));
            }
            w.alias(format!("layer.{i}.conv"), n("linear_attn.conv1d.weight"));
            w.alias(format!("layer.{i}.conv_bias"), n("linear_attn.conv1d.bias"));
            w.alias(format!("layer.{i}.a_log"), n("linear_attn.A_log"));
            w.alias(format!("layer.{i}.dt_bias"), n("linear_attn.dt_bias"));
            w.alias(format!("layer.{i}.gate_norm"), n("linear_attn.norm.weight"));
            w.alias(format!("layer.{i}.o_proj"), n("linear_attn.out_proj.weight"));
        }
        // The fused gate‖up bank, gate first — the dense MLP's binding,
        // laid out by the plan.
        w.alias(format!("layer.{i}.gate_up"), n("mlp.gate_up_proj.fused.weight"));
    }
}

