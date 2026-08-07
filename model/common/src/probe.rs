//! What the HF naming convention says about a tensor, read off its name.
//!
//! Ported from the CUDA driver's `model/contract.hpp` name-pattern policy,
//! and schema-side on purpose: every function here is a claim about *how
//! checkpoints are published* — which suffix denotes a column-parallel
//! projection, which tensor is a companion scale — and none is a claim about
//! any kernel. The kernel-side lists (which weights a runtime-quant request
//! may re-encode) live in [`builder`](crate::builder), beside the
//! lowering that reads them.

use pie_loader::types::Encoding;

/// The axis tensor parallelism splits a tensor on, read off its name.
///
/// Not any one family's rule, despite the family-flavoured tails in the list.
/// It is the HF naming convention's answer: a name ending `.q_proj.weight`
/// denotes a column-parallel projection whatever model ships it, and
/// `.sinks`, `.w1/.w2/.w3` and `.linear_attn.*` are conventions several
/// families adopted, not identities. A family whose checkpoint uses a
/// *different* convention for the same operator says so with
/// [`Builder::shard_axis_fn`](crate::builder::Builder::shard_axis_fn)
/// — DeepSeek-V4 is the one such family, and its rule lives in its own
/// module.
pub fn hf_shard_axis(name: &str) -> Option<u8> {
    // A companion scale splits exactly like the weight it scales, so ask
    // about the weight. Whether the scale has an axis to split at all is a
    // question about its shape, and `Builder::splittable_axis` answers that
    // one.
    for suffix in [
        ".weight_scale_inv",
        ".weight_scale",
        ".weight_packed",
        ".scale",
    ] {
        if let Some(base) = name.strip_suffix(suffix) {
            let of_weight = hf_shard_axis(&format!("{base}.weight"));
            return of_weight.or_else(|| hf_shard_axis(base));
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
    if name.ends_with(".experts.down_proj") || name.ends_with(".mlp.experts.down_proj") {
        return Some(2);
    }
    None
}

/// A routed or shared expert's projection, under the HF MoE naming convention.
pub fn is_expert_projection(name: &str) -> bool {
    (name.contains(".mlp.experts.") || name.contains(".mlp.shared_experts."))
        && [".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"]
            .iter()
            .any(|tail| name.ends_with(tail))
}

/// True for a tensor that only scales another one.
pub fn is_companion_scale(name: &str) -> bool {
    [".weight_scale_inv", ".weight_scale", ".scale"]
        .iter()
        .any(|tail| name.ends_with(tail))
}

/// The weight a companion scale belongs to, or `None` if `name` is not one.
///
/// `.weight_scale_inv` and `.weight_scale` hang off the weight's own name, so
/// only the scale part comes off; a bare `.scale` shares a base with the
/// weight, so `.weight` goes back on. All three end at `<base>.weight`.
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

/// A multimodal tower output, under the HF prefix convention.
pub fn is_tower_output(name: &str) -> bool {
    name.starts_with("model.vision_tower.")
        || name.starts_with("model.embed_vision.")
        || name.starts_with("model.audio_tower.")
        || name.starts_with("model.embed_audio.")
}

/// Whether byte-run addressing (a band, a stack, a strided window) is
/// meaningful over this encoding.
///
/// It is not when elements straddle byte boundaries. Checking it up front
/// turns that into a message naming the tensor rather than a wrong extent
/// later.
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
