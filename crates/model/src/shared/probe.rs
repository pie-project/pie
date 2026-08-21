use model_loader::types::Encoding;

pub fn hf_shard_axis(name: &str) -> Option<u8> {

    for suffix in [
        ".weight_scale_inv",
        ".weight_scale",
        ".weight_packed",
        ".scale",
    ] {
        if let Some(base) = name.strip_suffix(suffix) {
            return hf_shard_axis(base).or_else(|| hf_shard_axis(&format!("{base}.weight")));
        }
    }
    const ROW_PARALLEL: &[&str] = &[
        ".q_proj.weight",
        ".q_proj.bias",
        ".k_proj.weight",
        ".k_proj.bias",
        ".v_proj.weight",
        ".v_proj.bias",
        ".gate_proj.weight",
        ".up_proj.weight",
        ".sinks",
        ".w1.weight",
        ".w3.weight",
        ".w1.bias",
        ".w3.bias",
        ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight",
        ".linear_attn.in_proj_a.weight",
        ".linear_attn.dt_bias",
        ".linear_attn.A_log",
        ".self_attn.q_b_proj.weight",
        ".self_attn.kv_b_proj.weight",
    ];
    if ROW_PARALLEL.iter().any(|tail| name.ends_with(tail)) {
        return Some(0);
    }
    if [
        ".o_proj.weight",
        ".down_proj.weight",
        ".w2.weight",
        ".linear_attn.out_proj.weight",
    ]
    .iter()
    .any(|tail| name.ends_with(tail))
    {
        return Some(1);
    }

    if name.ends_with(".experts.down_proj") {
        return Some(2);
    }
    None
}

pub fn is_expert_projection(name: &str) -> bool {
    (name.contains(".mlp.experts.") || name.contains(".mlp.shared_experts."))
        && [".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"]
            .iter()
            .any(|tail| name.ends_with(tail))
}

pub fn is_companion_scale(name: &str) -> bool {
    [".weight_scale_inv", ".weight_scale", ".scale"]
        .iter()
        .any(|tail| name.ends_with(tail))
}

pub fn companion_weight_name(name: &str) -> Option<String> {
    for part in ["_scale_inv", "_scale"] {
        if let Some(base) = name.strip_suffix(part)
            && base.ends_with(".weight")
        {
            return Some(base.to_string());
        }
    }
    name.strip_suffix(".scale")
        .map(|base| format!("{base}.weight"))
}

pub fn is_tower_output(name: &str) -> bool {
    name.starts_with("model.vision_tower.")
        || name.starts_with("model.embed_vision.")
        || name.starts_with("model.audio_tower.")
        || name.starts_with("model.embed_audio.")
}

pub fn is_dense_addressable(encoding: &Encoding) -> bool {
    match encoding {
        Encoding::Raw(_) => true,
        Encoding::Quant(spec) => {
            let bits = if spec.bits_per_element != 0 {
                spec.bits_per_element
            } else {
                spec.scheme.default_bits()
            };
            bits % 8 == 0
        }
    }
}
