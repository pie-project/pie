use std::collections::HashMap;

use crate::catalog::LoadShape;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Names {
    pub layer_prefix: String,

    pub roles: HashMap<String, Vec<String>>,

    pub globals: HashMap<String, Vec<String>>,

    pub weight_suffix: Vec<String>,

    pub zero_point_suffix: Vec<String>,

    pub bias_suffix: Vec<String>,
}

impl Names {
    #[must_use]
    pub fn mlx() -> Self {
        let roles = [
            ("qkv", "self_attn.qkv_proj.fused"),
            ("gate_up", "mlp.gate_up_proj.fused"),
            ("q_proj", "self_attn.q_proj"),
            ("k_proj", "self_attn.k_proj"),
            ("v_proj", "self_attn.v_proj"),
            ("o_proj", "self_attn.o_proj|linear_attn.out_proj"),
            ("q_bias", "self_attn.q_proj"),
            ("k_bias", "self_attn.k_proj"),
            ("v_bias", "self_attn.v_proj"),
            ("o_bias", "self_attn.o_proj"),
            ("gate_proj", "mlp.gate_proj"),
            ("up_proj", "mlp.up_proj"),
            ("down", "mlp.down_proj"),
            ("router", "mlp.gate|mlp.router|router.proj"),
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
            ("ple_gate", "per_layer_gate"),
            ("ple_out", "per_layer_projection"),
            ("scalar", "layer_scalar"),
            ("attn_sinks", "self_attn.sinks"),
            ("q_norm", "self_attn.q_norm"),
            ("k_norm", "self_attn.k_norm"),
            ("in_proj_qkv", "linear_attn.in_proj_qkv"),
            ("in_proj_z", "linear_attn.in_proj_z"),
            ("in_proj_a", "linear_attn.in_proj_a"),
            ("in_proj_b", "linear_attn.in_proj_b"),
            ("conv_w", "linear_attn.conv1d"),
            ("conv_b", "linear_attn.conv1d.bias"),
            ("a_log", "linear_attn.A_log"),
            ("dt", "linear_attn.dt_bias"),
            ("gate_norm", "linear_attn.norm"),
            ("attn_norm", "input_layernorm"),
            (
                "mlp_norm",
                "pre_feedforward_layernorm|post_attention_layernorm",
            ),
            ("post_attn_norm", "post_attention_layernorm"),
            ("post_mlp_norm", "post_feedforward_layernorm"),
            ("post_mlp_norm_1", "post_feedforward_layernorm_1"),
            ("mlp_norm_2", "pre_feedforward_layernorm_2"),
            ("post_mlp_norm_2", "post_feedforward_layernorm_2"),
            ("router_scale", "router.scale"),
            ("router_expert_scale", "router.per_expert_scale"),
        ]
        .into_iter()
        .map(|(a, b): (&str, &str)| (a.to_string(), b.split('|').map(str::to_string).collect()))
        .collect();
        let globals = [
            ("embed", "shared_embedding|embed_tokens"),
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

            weight_suffix: vec![".weight".to_string(), String::new()],

            zero_point_suffix: vec![".biases".to_string()],
            bias_suffix: vec![".bias".to_string()],
        }
    }
}

pub struct Wiring<'a> {
    pub published: &'a dyn Fn(&str) -> bool,

    pub aliases: Vec<(String, String)>,

    pub joins: Vec<(String, Vec<String>)>,

    pub scalars: Vec<String>,

    pub shape: LoadShape,
}

impl<'a> Wiring<'a> {
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

    fn alias(&mut self, trace: String, published: String) {
        if self.has(&published) {
            self.aliases.push((trace, published));
        }
    }

    fn named(&self, trace: &str) -> bool {
        self.aliases.iter().any(|(t, _)| t == trace) || self.joins.iter().any(|(t, _)| t == trace)
    }

    fn join(&mut self, trace: String, parts: &[String]) {
        if parts.iter().all(|p| self.has(p)) {
            self.joins.push((trace, parts.to_vec()));
        }
    }
}

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
        return;
    }

    w.alias("embed".into(), "model.embed_tokens.weight".into());
    w.alias("final_norm".into(), "model.norm.weight".into());
    if w.has("lm_head.weight") {
        w.alias("lm_head".into(), "lm_head.weight".into());
    } else {
        w.alias("lm_head".into(), "model.embed_tokens.weight".into());
    }
    let layers = w.shape.layers as usize;
    for i in 0..layers {
        let n = |s: &str| format!("model.layers.{i}.{s}");

        w.alias(
            format!("layer.{i}.qkv"),
            n("self_attn.qkv_proj.fused.weight"),
        );
        w.alias(
            format!("layer.{i}.gate_up"),
            n("mlp.gate_up_proj.fused.weight"),
        );

        w.alias(format!("layer.{i}.qkv"), n("self_attn.qkv_proj.weight"));
        w.alias(format!("layer.{i}.gate_up"), n("mlp.gate_up_proj.weight"));

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
        w.alias(format!("layer.{i}.gate_proj"), n("mlp.gate_proj.weight"));
        w.alias(format!("layer.{i}.up_proj"), n("mlp.up_proj.weight"));
        w.alias(format!("layer.{i}.down_proj"), n("mlp.down_proj.weight"));

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
            ("o_bias", "self_attn.o_proj.bias"),
        ] {
            w.alias(format!("layer.{i}.{trace}"), n(ckpt));
        }
    }
}

fn gpt_oss(w: &mut Wiring<'_>) {
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

#[allow(clippy::too_many_lines)]
fn gemma4(w: &mut Wiring<'_>) {
    let p = "model.language_model";
    if !w.has(&format!("{p}.embed_tokens_per_layer.weight")) {
        return;
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

        w.scalars.push(n("layer_scalar"));
    }
}

fn qwen3_5(w: &mut Wiring<'_>) {
    let p = "model.language_model";
    if !w.has(&format!("{p}.embed_tokens.weight")) {
        return;
    }
    if w.has(&format!("{p}.embed_tokens_per_layer.weight")) {
        return;
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

        w.alias(
            format!("layer.{i}.gate_up"),
            n("mlp.gate_up_proj.fused.weight"),
        );
    }
}
