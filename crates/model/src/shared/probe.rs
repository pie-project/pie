//! What the HF naming convention says about a tensor, read off its name.
//!
//! Ported from the CUDA driver's `model/contract.hpp` name-pattern policy,
//! and schema-side on purpose: every function here is a claim about *how
//! checkpoints are published* — which suffix denotes a column-parallel
//! projection, which tensor is a companion scale — and none is a claim about
//! any kernel. The kernel-side lists (which weights a runtime-quant request
//! may re-encode) live in [`builder`](crate::shared::builder), beside the
//! lowering that reads them.

use model_loader::types::Encoding;

/// The axis tensor parallelism splits a tensor on, read off its name.
///
/// Not any one family's rule, despite the family-flavoured tails in the list.
/// It is the HF naming convention's answer: a name ending `.q_proj.weight`
/// denotes a column-parallel projection whatever model ships it, and
/// `.sinks`, `.w1/.w2/.w3` and `.linear_attn.*` are conventions several
/// families adopted, not identities. A family whose checkpoint uses a
/// *different* convention for the same operator says so with
/// [`Builder::shard_axis_fn`](crate::shared::builder::Builder::shard_axis_fn)
/// — DeepSeek-V4 is the one such family, and its rule lives in its own
/// module.
pub fn hf_shard_axis(name: &str) -> Option<u8> {
    // A companion scale splits exactly like the weight it scales, so ask
    // about the weight. Whether the scale has an axis to split at all is a
    // question about its shape, and `Builder::splittable_axis` answers that
    // one.
    //
    // The BARE base is asked first and `<base>.weight` second, because the
    // bare base is the tensor's own name with the companion suffix taken
    // off, while `<base>.weight` is a guess at how the convention would
    // have spelled it. For a stacked expert bank the guess matches a
    // DIFFERENT rule: `model...mlp.experts.down_proj.weight` ends with
    // `.down_proj.weight` and answers 1, the axis an UNSTACKED down
    // projection splits on, while the bank's own name answers 2. Asking
    // the guess first returned 1 for every stacked bank's scale --
    // splitting the factor table on the expert-major axis, so each rank
    // dequantized its columns with another expert's factors. Both orders
    // agree on every unstacked name, because a bare `.q_proj` matches no
    // rule and falls through to the same second question.
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
    // A STACKED expert bank, whose leading axis is the expert: the split
    // is the intermediate, one axis further in than an unstacked
    // `down_proj`'s. Only the `.experts.` spelling is checked because
    // `.mlp.experts.down_proj` ends with it too — a second clause naming
    // the longer tail decides nothing and only reads as though it does.
    if name.ends_with(".experts.down_proj") {
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

#[cfg(test)]
mod tests {
    use super::*;
    use model_loader::types::{Axis, DType, QuantScheme, QuantSpec};

    const L: &str = "model.layers.3.";

    /// A companion scale splits like the weight it scales, and asking
    /// about the weight is how it finds out.
    ///
    /// Every one of the four spellings has to redirect, because a scale
    /// that answered `None` here would be REPLICATED while its weight was
    /// split: every rank would hold the whole scale table and index it
    /// with a local column, so rank 1's first column would be dequantized
    /// by rank 0's first factor. No shape disagrees — the scale is the
    /// right length for the tensor it is being read against — and the
    /// numbers are simply wrong, worse the further from rank 0.
    #[test]
    fn a_companion_scale_splits_on_the_axis_its_weight_splits_on() {
        for suffix in [
            ".weight_scale_inv",
            ".weight_scale",
            ".weight_packed",
            ".scale",
        ] {
            for (proj, axis) in [("q_proj", 0u8), ("o_proj", 1)] {
                let weight = format!("{L}self_attn.{proj}.weight");
                let scale = format!("{L}self_attn.{proj}{suffix}");
                assert_eq!(
                    hf_shard_axis(&scale),
                    Some(axis),
                    "{scale} must follow {weight}"
                );
                assert_eq!(hf_shard_axis(&weight), Some(axis));
            }
        }
    }

    /// The redirect asks the BARE base first, and the order is the whole
    /// answer for a stacked expert bank.
    ///
    /// A bank IS `...mlp.experts.down_proj`; there is no `.weight`
    /// spelling of it. But `...mlp.experts.down_proj.weight` matches the
    /// `.down_proj.weight` rule perfectly well, and that rule names the
    /// axis an UNSTACKED down projection splits on. Asking the guess
    /// first therefore answers 1 for a tensor whose leading axis is the
    /// expert index — one axis off, with the factor table split
    /// expert-major so each rank dequantizes its columns with another
    /// expert's factors.
    #[test]
    fn a_stacked_banks_scale_follows_the_bank_and_not_its_unstacked_namesake() {
        let bank = format!("{L}mlp.experts.down_proj");
        assert_eq!(hf_shard_axis(&bank), Some(2));
        assert_eq!(
            hf_shard_axis(&format!("{bank}.weight")),
            Some(1),
            "the `.weight` spelling is a DIFFERENT rule, which is why the \
             fallback exists"
        );
        assert_eq!(
            hf_shard_axis(&format!("{bank}.scale")),
            Some(2),
            "the scale of a stacked bank follows the bank, not the \
             unstacked projection whose name is a suffix of it"
        );
    }

    /// The three axes, one name each, read off the convention and not off
    /// any family.
    #[test]
    fn the_convention_answers_row_column_and_expert_stacked() {
        for name in [
            "q_proj.weight",
            "k_proj.bias",
            "gate_proj.weight",
            "w1.weight",
            "linear_attn.A_log",
            "self_attn.kv_b_proj.weight",
        ] {
            assert_eq!(hf_shard_axis(&format!("{L}{name}")), Some(0), "{name}");
        }
        for name in [
            "o_proj.weight",
            "down_proj.weight",
            "w2.weight",
            "linear_attn.out_proj.weight",
        ] {
            assert_eq!(hf_shard_axis(&format!("{L}{name}")), Some(1), "{name}");
        }
        assert_eq!(hf_shard_axis(&format!("{L}mlp.experts.down_proj")), Some(2));
        // Replicated: a norm is per-channel over the hidden width, which
        // every rank holds whole.
        for name in ["input_layernorm.weight", "mlp_norm.weight", "router.weight"] {
            assert_eq!(hf_shard_axis(&format!("{L}{name}")), None, "{name}");
        }
    }

    /// `.sinks` is row-parallel, and it is the one entry that is not a
    /// projection: gpt-oss's per-head attention sink is one value per
    /// query head, so it splits with the heads.
    #[test]
    fn the_attention_sink_splits_with_the_heads_it_belongs_to() {
        assert_eq!(hf_shard_axis(&format!("{L}self_attn.sinks")), Some(0));
    }

    #[test]
    fn an_expert_projection_is_one_under_either_expert_prefix() {
        for prefix in ["mlp.experts.0", "mlp.shared_experts"] {
            for member in ["gate_proj", "up_proj", "down_proj"] {
                assert!(is_expert_projection(&format!(
                    "{L}{prefix}.{member}.weight"
                )));
            }
        }
        // The prefix alone is not enough: a router lives under `.mlp.` and
        // is not an expert's, and an expert's SCALE is not its projection.
        assert!(!is_expert_projection(&format!("{L}mlp.gate.weight")));
        assert!(!is_expert_projection(&format!(
            "{L}mlp.experts.0.down_proj.weight_scale"
        )));
        assert!(!is_expert_projection(&format!(
            "{L}self_attn.q_proj.weight"
        )));
    }

    /// A packed weight is a WEIGHT. It shares the redirect list with the
    /// scales because it splits like one, and it is absent from this one
    /// because publishing it as a companion would drop the tensor that
    /// holds the model's numbers.
    #[test]
    fn a_packed_weight_is_not_a_companion_scale() {
        assert!(is_companion_scale(&format!(
            "{L}mlp.down_proj.weight_scale"
        )));
        assert!(is_companion_scale(&format!(
            "{L}mlp.down_proj.weight_scale_inv"
        )));
        assert!(is_companion_scale(&format!(
            "{L}mlp.experts.down_proj.scale"
        )));
        assert!(!is_companion_scale(&format!(
            "{L}mlp.down_proj.weight_packed"
        )));
        assert!(!is_companion_scale(&format!("{L}mlp.down_proj.weight")));
    }

    /// All three spellings end at `<base>.weight`, which is the point: the
    /// caller pairs a scale with its weight by name, and three answers of
    /// three shapes would be three pairings.
    #[test]
    fn every_spelling_of_a_scale_names_the_same_weight() {
        let want = format!("{L}mlp.down_proj.weight");
        for name in [
            format!("{L}mlp.down_proj.weight_scale"),
            format!("{L}mlp.down_proj.weight_scale_inv"),
            format!("{L}mlp.down_proj.scale"),
        ] {
            assert_eq!(
                companion_weight_name(&name).as_deref(),
                Some(&*want),
                "{name}"
            );
        }
        assert_eq!(
            companion_weight_name(&want),
            None,
            "a weight is not a scale"
        );
        // `_scale` only comes off when what is left is a weight: a tensor
        // whose own name happens to end in `_scale` is not a companion.
        assert_eq!(companion_weight_name(&format!("{L}mlp.router_scale")), None);
    }

    #[test]
    fn a_tower_output_is_one_under_all_four_published_prefixes() {
        for prefix in [
            "model.vision_tower.",
            "model.embed_vision.",
            "model.audio_tower.",
            "model.embed_audio.",
        ] {
            assert!(
                is_tower_output(&format!("{prefix}encoder.0.weight")),
                "{prefix}"
            );
        }
        assert!(!is_tower_output(&format!("{L}self_attn.q_proj.weight")));
        // The prefixes anchor at the START. A decoder tensor that merely
        // contains one is the decoder's.
        assert!(!is_tower_output("model.layers.0.model.vision_tower.weight"));
    }

    fn quant(scheme: QuantScheme, bits: u8) -> Encoding {
        Encoding::Quant(QuantSpec {
            scheme,
            logical_dtype: DType::BF16,
            bits_per_element: bits,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        })
    }

    /// A spec that states no width answers from its SCHEME, not from zero.
    ///
    /// `QuantSpec::bits_per_element` is `0` until `normalized()` runs, and
    /// nothing here runs it. Reading the zero would make `0 % 8 == 0`
    /// true, so every un-normalized quantized tensor would be called
    /// byte-addressable — including the 4-bit ones this function exists to
    /// refuse — and a band over one would be cut at a boundary that falls
    /// inside an element.
    #[test]
    fn a_spec_that_states_no_width_answers_from_its_scheme() {
        for scheme in [
            QuantScheme::AwqInt4,
            QuantScheme::GptqInt4,
            QuantScheme::Mxfp4E2M1E8M0,
            QuantScheme::MlxAffineU4,
            QuantScheme::Int4B8,
        ] {
            assert!(
                !is_dense_addressable(&quant(scheme, 0)),
                "{scheme:?} is 4 bits by default and 4 does not divide 8"
            );
            assert!(scheme.default_bits() % 8 != 0);
        }
        for scheme in [QuantScheme::Fp8E4M3, QuantScheme::Int8Symmetric] {
            assert!(
                is_dense_addressable(&quant(scheme, 0)),
                "{scheme:?} is 8 bits by default"
            );
        }
    }

    /// A STATED width wins over the scheme's, in both directions.
    #[test]
    fn a_stated_width_is_the_one_that_is_read() {
        // An int4 scheme carrying a byte width is addressable...
        assert!(is_dense_addressable(&quant(QuantScheme::AwqInt4, 8)));
        // ...and an 8-bit scheme carrying a sub-byte width is not.
        assert!(!is_dense_addressable(&quant(QuantScheme::Fp8E4M3, 5)));
        assert!(is_dense_addressable(&Encoding::Raw(DType::BF16)));
        assert!(is_dense_addressable(&Encoding::Raw(DType::F32)));
    }
}
