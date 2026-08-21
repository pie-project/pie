#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slot {

    Required(String),

    Optional(String),
}

impl Slot {

    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Required(n) | Self::Optional(n) => n,
        }
    }
}

fn clipped_linear(base: &str, out: &mut Vec<Slot>) {
    out.push(Slot::Required(format!("{base}.linear.weight")));
    for m in ["input_min", "input_max", "output_min", "output_max"] {
        out.push(Slot::Optional(format!("{base}.{m}")));
    }
}

#[must_use]
pub fn audio_head(prefix: &str, embed: &str) -> Vec<Slot> {
    [
        "subsample_conv_projection.layer0.conv.weight",
        "subsample_conv_projection.layer0.norm.weight",
        "subsample_conv_projection.layer1.conv.weight",
        "subsample_conv_projection.layer1.norm.weight",
        "subsample_conv_projection.input_proj_linear.weight",
        "output_proj.weight",
        "output_proj.bias",
    ]
    .into_iter()
    .map(|n| Slot::Required(format!("{prefix}.{n}")))
    .chain(std::iter::once(Slot::Required(embed.to_owned())))
    .collect()
}

#[must_use]
pub fn audio_layers(prefix: &str, layers: u32) -> Vec<Slot> {

    fn feed_forward(base: &str, out: &mut Vec<Slot>) {
        out.push(Slot::Required(format!("{base}.pre_layer_norm.weight")));
        out.push(Slot::Required(format!("{base}.post_layer_norm.weight")));
        clipped_linear(&format!("{base}.ffw_layer_1"), out);
        clipped_linear(&format!("{base}.ffw_layer_2"), out);
    }

    let mut out = Vec::with_capacity(layers as usize * AUDIO_SLOTS_PER_LAYER);
    for l in 0..layers {
        let lp = format!("{prefix}.layers.{l}");
        feed_forward(&format!("{lp}.feed_forward1"), &mut out);
        feed_forward(&format!("{lp}.feed_forward2"), &mut out);
        out.push(Slot::Required(format!("{lp}.norm_pre_attn.weight")));
        out.push(Slot::Required(format!("{lp}.norm_post_attn.weight")));
        for p in ["q_proj", "k_proj", "v_proj", "post"] {
            clipped_linear(&format!("{lp}.self_attn.{p}"), &mut out);
        }
        out.push(Slot::Required(format!(
            "{lp}.self_attn.relative_k_proj.weight"
        )));
        out.push(Slot::Required(format!("{lp}.self_attn.per_dim_scale")));
        out.push(Slot::Required(format!(
            "{lp}.lconv1d.pre_layer_norm.weight"
        )));
        out.push(Slot::Required(format!("{lp}.lconv1d.conv_norm.weight")));
        clipped_linear(&format!("{lp}.lconv1d.linear_start"), &mut out);
        clipped_linear(&format!("{lp}.lconv1d.linear_end"), &mut out);
        out.push(Slot::Required(format!(
            "{lp}.lconv1d.depthwise_conv1d.weight"
        )));
        out.push(Slot::Required(format!("{lp}.norm_out.weight")));
    }
    out
}

pub const AUDIO_SLOTS_PER_LAYER: usize = 62;

pub const VISION_SLOTS_PER_LAYER: usize = 41;

#[must_use]
pub fn vision_head(prefix: &str, embed: &str) -> Vec<Slot> {
    vec![
        Slot::Required(format!("{prefix}.patch_embedder.input_proj.weight")),
        Slot::Required(format!("{prefix}.patch_embedder.position_embedding_table")),
        Slot::Required(embed.to_owned()),
    ]
}

#[must_use]
pub fn vision_layers(prefix: &str, layers: u32) -> Vec<Slot> {
    let mut out = Vec::with_capacity(layers as usize * VISION_SLOTS_PER_LAYER);
    for l in 0..layers {
        let lp = format!("{prefix}.encoder.layers.{l}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ] {
            out.push(Slot::Required(format!("{lp}.{norm}.weight")));
        }
        for c in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            clipped_linear(&format!("{lp}.{c}"), &mut out);
        }
    }
    out
}
