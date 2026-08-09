//! Which weights each dispatch binds: the one place a [`Kernel`] kind is
//! translated to checkpoint tensor names and bind slots.
//!
//! A kind IS a weight name. That is why the shared expert's projections are
//! separate kinds from the dense `QmvGate`/`QmvUp`/`QmvDown` — reusing the
//! dense kinds would have made the contract rename
//! `mlp.shared_expert.gate_proj` to `mlp.gate_proj`, true of the shape,
//! false of the model, and unreadable in any dump — and why a tied and an
//! untied embedding are two kinds over one kernel: what differs is only
//! which tensor is asked for.
//!
//! Names here are the MLX vocabulary the load plan publishes (the plan is
//! authored with `Naming::Mlx`); a name this table asks for that the plan
//! did not stage is a load failure naming the tensor, not a zero. That
//! cuts both ways: Qwen's experts carry no biases, so the llama-family
//! mixture kinds must NOT ask for `mlp.experts.gate_proj.bias` — gpt-oss's
//! must, at slot 7, where `.bias` (the Linear's additive vector) and
//! `.biases` (the affine triplet's zero point) differ by one character and
//! mean nothing alike.

use super::abi::Kernel;
use super::geometry::DecodeGeometry;

/// The bind-slot ordinals this table writes to. A subset of the C++
/// `bind::` namespace — the slots that carry WEIGHTS; the activation and
/// parameter slots arrive with the argument-table bind pass that reads
/// them.
pub mod slot {
    /// `bind::Qmv::W` — also `Qmm`'s and `GoQmv`'s: the ordinals are
    /// frozen so the weight binds are shared across GEMV and GEMM.
    pub const QMV_W: u8 = 0;
    /// `bind::Qmv::Scales`.
    pub const QMV_SCALES: u8 = 1;
    /// `bind::Qmv::Biases` — the affine zero point.
    pub const QMV_BIASES: u8 = 2;
    /// `bind::Rms::W`.
    pub const RMS_W: u8 = 1;
    /// `bind::GdnPrep::ConvW`.
    pub const GDN_PREP_CONV_W: u8 = 2;
    /// `bind::GdnPrep::ALog`.
    pub const GDN_PREP_A_LOG: u8 = 4;
    /// `bind::GdnPrep::DtBias`.
    pub const GDN_PREP_DT_BIAS: u8 = 5;
    /// `bind::GdnCore::ConvW`.
    pub const GDN_CORE_CONV_W: u8 = 4;
    /// `bind::GdnCore::ALog`.
    pub const GDN_CORE_A_LOG: u8 = 6;
    /// `bind::GdnCore::DtBias`.
    pub const GDN_CORE_DT_BIAS: u8 = 7;
    /// `bind::GdnCoreRecurrent::ConvW` — the slimmed core keeps the conv
    /// weight for the v convsilu it still owns.
    pub const GDN_CORE_RECURRENT_CONV_W: u8 = 4;
    /// `bind::GatedRms::W`.
    pub const GATED_RMS_W: u8 = 2;
    /// `bind::GoQmv::Bias` — the Linear's additive bias, NOT `Biases`.
    pub const GO_QMV_BIAS: u8 = 7;
    /// `bind::SdpaSink::Sinks` — gpt-oss's learned per-head scalar.
    pub const SDPA_SINK_SINKS: u8 = 14;
    /// `bind::SdpaPaged::Sinks` — the SAME tensor, a DIFFERENT slot: on
    /// the paged ABI index 14 is `AttnMaskEnabled`, and the C++ (one Kind,
    /// two ABIs) patched the collision with a bind-time remap — "a weight
    /// read as a mask, and a mask read as a weight." Two kinds, two
    /// constants; there is nothing to remap.
    pub const SDPA_PAGED_SINKS: u8 = 16;
    /// One past `bind::GoRouterTopK::Params`, which is what
    /// `router_topk_scaled` declares for gemma4's per-expert gain.
    pub const ROUTER_TOPK_SCALE: u8 = 4;
    /// `bind::LayerScalar::Scalar`.
    pub const LAYER_SCALAR: u8 = 1;
    /// `bind::RmsResidual::W` — keeps `bind::Rms`'s prefix so the norm
    /// weight lands at the same slot in the fused kind.
    pub const RMS_RESIDUAL_W: u8 = 1;
    /// `bind::RmsResidual::Scalar` — the learned gain the separate
    /// `LayerScalar` dispatch used to carry.
    pub const RMS_RESIDUAL_SCALAR: u8 = 5;
}

/// One weight a dispatch binds: the slot and the staged tensor's name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightBind {
    /// The `bind::<Kind>` slot.
    pub bind_index: u8,
    /// The runtime tensor name, as the load plan staged it.
    pub tensor: String,
}

/// `layers.<n>.` for a layered dispatch, nothing for a singleton.
#[must_use]
pub fn layer_prefix(layer: Option<u32>) -> String {
    match layer {
        Some(layer) => format!("layers.{layer}."),
        None => String::new(),
    }
}

fn bind(out: &mut Vec<WeightBind>, bind_index: u8, tensor: String) {
    out.push(WeightBind { bind_index, tensor });
}

/// The affine triplet: weight, scales, and the zero point.
fn push_quant(out: &mut Vec<WeightBind>, base: &str) {
    bind(out, slot::QMV_W, format!("{base}.weight"));
    bind(out, slot::QMV_SCALES, format!("{base}.scales"));
    bind(out, slot::QMV_BIASES, format!("{base}.biases"));
}

/// The MXFP4 flavour: a `.scales` of block exponents and no zero point.
/// The third slot is left unbound because the format has nothing to put in
/// it, and the kernel that reads these never loads from it.
fn push_mxfp4(out: &mut Vec<WeightBind>, base: &str) {
    bind(out, slot::QMV_W, format!("{base}.weight"));
    bind(out, slot::QMV_SCALES, format!("{base}.scales"));
}

/// The load-once weight tensors one dispatch binds; empty for a weightless
/// kind (the movers, sums and elementwise stages read activations only).
#[must_use]
#[allow(clippy::too_many_lines)] // one switch, one table: splitting it hides the mapping
pub fn weight_binds(
    kind: Kernel,
    layer: Option<u32>,
    g: &DecodeGeometry,
    gdn_prep: bool,
) -> Vec<WeightBind> {
    let mut out = Vec::new();
    let prefix = layer_prefix(layer);
    let push_expert = |out: &mut Vec<WeightBind>, base: &str| {
        if g.mxfp4_experts {
            push_mxfp4(out, base);
        } else {
            push_quant(out, base);
        }
    };
    match kind {
        // Tied: one table serves both ends of the model.
        Kernel::EmbedGather | Kernel::QmvLmHead => push_quant(&mut out, "shared_embedding"),
        Kernel::FinalRms => bind(&mut out, slot::RMS_W, "final_norm.weight".into()),
        Kernel::Rms => bind(
            &mut out,
            slot::RMS_W,
            format!("{prefix}input_layernorm.weight"),
        ),
        Kernel::FfnRms => bind(
            &mut out,
            slot::RMS_W,
            format!("{prefix}post_attention_layernorm.weight"),
        ),
        Kernel::QNorm => bind(
            &mut out,
            slot::RMS_W,
            format!("{prefix}self_attn.q_norm.weight"),
        ),
        Kernel::KNorm => bind(
            &mut out,
            slot::RMS_W,
            format!("{prefix}self_attn.k_norm.weight"),
        ),
        Kernel::QmvQ => push_quant(&mut out, &format!("{prefix}self_attn.q_proj")),
        Kernel::QmvK => push_quant(&mut out, &format!("{prefix}self_attn.k_proj")),
        Kernel::QmvV => push_quant(&mut out, &format!("{prefix}self_attn.v_proj")),
        Kernel::QmvO => push_quant(&mut out, &format!("{prefix}self_attn.o_proj")),
        Kernel::QmvIn => push_quant(&mut out, &format!("{prefix}linear_attn.in_proj_qkv")),
        Kernel::QmvInZ => push_quant(&mut out, &format!("{prefix}linear_attn.in_proj_z")),
        Kernel::QmvOut => push_quant(&mut out, &format!("{prefix}linear_attn.out_proj")),
        Kernel::GdnInA => push_quant(&mut out, &format!("{prefix}linear_attn.in_proj_a")),
        Kernel::GdnInB => push_quant(&mut out, &format!("{prefix}linear_attn.in_proj_b")),

        // ── Gemma 4: the norm sandwich (four per layer, so three need
        // their own kind), the layer scalar, and the PLE plumbing. ──
        Kernel::G4AttnPostNorm => bind(
            &mut out,
            slot::RMS_W,
            format!("{prefix}post_attention_layernorm.weight"),
        ),
        Kernel::G4FfnPreNorm => bind(
            &mut out,
            slot::RMS_W,
            format!("{prefix}pre_feedforward_layernorm.weight"),
        ),
        Kernel::G4FfnPostNorm => bind(
            &mut out,
            slot::RMS_W,
            format!("{prefix}post_feedforward_layernorm.weight"),
        ),
        Kernel::G4LayerScalar => {
            bind(
                &mut out,
                slot::LAYER_SCALAR,
                format!("{prefix}layer_scalar"),
            );
        }
        Kernel::G4PleNorm => bind(
            &mut out,
            slot::RMS_W,
            format!("{prefix}post_per_layer_input_norm.weight"),
        ),
        // The fused norm+residual kinds keep bind::Rms's prefix, so the
        // norm weight lands at the same slot; the scaled variant also
        // carries the learned gain the separate LayerScalar dispatch used
        // to.
        Kernel::G4AttnPostResidual => bind(
            &mut out,
            slot::RMS_RESIDUAL_W,
            format!("{prefix}post_attention_layernorm.weight"),
        ),
        Kernel::G4FfnPostResidual => bind(
            &mut out,
            slot::RMS_RESIDUAL_W,
            format!("{prefix}post_feedforward_layernorm.weight"),
        ),
        Kernel::G4PleResidualScaled => {
            bind(
                &mut out,
                slot::RMS_RESIDUAL_W,
                format!("{prefix}post_per_layer_input_norm.weight"),
            );
            bind(
                &mut out,
                slot::RMS_RESIDUAL_SCALAR,
                format!("{prefix}layer_scalar"),
            );
        }
        // Gemma 4's mixture sits BESIDE the dense FFN rather than replacing
        // it, so the layer carries five norms and both branches'
        // projections. mlx-lm's suffixes are `_1` for the dense branch's
        // closing norm and `_2` for the routed pair; they are numbered, not
        // named, so this match is the only place the mapping is written
        // down.
        Kernel::G4RouterNorm => {
            bind(&mut out, slot::RMS_W, format!("{prefix}router.scale"));
        }
        Kernel::G4MoeNorm => bind(
            &mut out,
            slot::RMS_W,
            format!("{prefix}pre_feedforward_layernorm_2.weight"),
        ),
        Kernel::G4DenseBranchNorm => bind(
            &mut out,
            slot::RMS_W,
            format!("{prefix}post_feedforward_layernorm_1.weight"),
        ),
        Kernel::G4MoeBranchNorm => bind(
            &mut out,
            slot::RMS_W,
            format!("{prefix}post_feedforward_layernorm_2.weight"),
        ),
        Kernel::G4Router => push_quant(&mut out, &format!("{prefix}router.proj")),
        // A WEIGHT on the top-k kernel, which is otherwise weightless:
        // gemma 4 rescales the selected softmax by a learned per-expert
        // gain, at the slot `router_topk_scaled` declares.
        Kernel::G4RouterTopK => bind(
            &mut out,
            slot::ROUTER_TOPK_SCALE,
            format!("{prefix}router.per_expert_scale"),
        ),
        // One stacked tensor per projection, indexed by the routing —
        // `switch_glu` is mlx-lm's name for the stack, not a third module.
        Kernel::G4ExpertGate => {
            push_expert(&mut out, &format!("{prefix}experts.switch_glu.gate_proj"));
        }
        Kernel::G4ExpertUp => {
            push_expert(&mut out, &format!("{prefix}experts.switch_glu.up_proj"));
        }
        Kernel::G4ExpertDown => {
            push_expert(&mut out, &format!("{prefix}experts.switch_glu.down_proj"));
        }
        Kernel::G4PleProjNorm => bind(
            &mut out,
            slot::RMS_W,
            "per_layer_projection_norm.weight".into(),
        ),
        // The PLE table is gathered exactly like the token embedding, and
        // the three PLE projections are ordinary quantized matvecs.
        Kernel::G4PleTokenGather => push_quant(&mut out, "embed_tokens_per_layer"),
        Kernel::G4PleProjGemv => push_quant(&mut out, "per_layer_model_projection"),
        Kernel::G4PleGateGemv => {
            push_quant(&mut out, &format!("{prefix}per_layer_input_gate"));
        }
        Kernel::G4PleProjLayerGemv => {
            push_quant(&mut out, &format!("{prefix}per_layer_projection"));
        }

        // ── GPT-OSS: untied ends, and every projection carries an additive
        // bias at slot 7 alongside its quantized triplet. ──
        Kernel::EmbedUntied => push_quant(&mut out, "embed_tokens"),
        Kernel::LmHeadUntied => push_quant(&mut out, "lm_head"),
        Kernel::GoQmvQ => {
            push_quant(&mut out, &format!("{prefix}self_attn.q_proj"));
            bind(
                &mut out,
                slot::GO_QMV_BIAS,
                format!("{prefix}self_attn.q_proj.bias"),
            );
        }
        Kernel::GoQmvK => {
            push_quant(&mut out, &format!("{prefix}self_attn.k_proj"));
            bind(
                &mut out,
                slot::GO_QMV_BIAS,
                format!("{prefix}self_attn.k_proj.bias"),
            );
        }
        Kernel::GoQmvV => {
            push_quant(&mut out, &format!("{prefix}self_attn.v_proj"));
            bind(
                &mut out,
                slot::GO_QMV_BIAS,
                format!("{prefix}self_attn.v_proj.bias"),
            );
        }
        Kernel::GoQmvO => {
            push_quant(&mut out, &format!("{prefix}self_attn.o_proj"));
            bind(
                &mut out,
                slot::GO_QMV_BIAS,
                format!("{prefix}self_attn.o_proj.bias"),
            );
        }
        // The paged form reads the SAME learned sinks at a DIFFERENT
        // index — see the slot constants: sharing 14 here is the C++'s
        // collision, verbatim.
        Kernel::GoSdpaSink => bind(
            &mut out,
            slot::SDPA_SINK_SINKS,
            format!("{prefix}self_attn.sinks"),
        ),
        Kernel::GoSdpaSinkPaged => bind(
            &mut out,
            slot::SDPA_PAGED_SINKS,
            format!("{prefix}self_attn.sinks"),
        ),
        Kernel::GoRouter => {
            push_quant(&mut out, &format!("{prefix}mlp.router"));
            bind(
                &mut out,
                slot::GO_QMV_BIAS,
                format!("{prefix}mlp.router.bias"),
            );
        }
        Kernel::GoExpertGate => {
            push_expert(&mut out, &format!("{prefix}mlp.experts.gate_proj"));
            bind(
                &mut out,
                slot::GO_QMV_BIAS,
                format!("{prefix}mlp.experts.gate_proj.bias"),
            );
        }
        Kernel::GoExpertUp => {
            push_expert(&mut out, &format!("{prefix}mlp.experts.up_proj"));
            bind(
                &mut out,
                slot::GO_QMV_BIAS,
                format!("{prefix}mlp.experts.up_proj.bias"),
            );
        }
        Kernel::GoExpertDown => {
            push_expert(&mut out, &format!("{prefix}mlp.experts.down_proj"));
            bind(
                &mut out,
                slot::GO_QMV_BIAS,
                format!("{prefix}mlp.experts.down_proj.bias"),
            );
        }

        // ── The llama family's routed FFN: the same tensors gpt-oss's
        // experts use, minus the biases — Qwen's experts carry none, and
        // asking for a tensor a checkpoint does not hold is a load
        // failure, not a zero. ──
        Kernel::LlRouter => push_quant(&mut out, &format!("{prefix}mlp.gate")),
        Kernel::LlExpertGate => {
            push_expert(&mut out, &format!("{prefix}mlp.experts.gate_proj"));
        }
        Kernel::LlExpertUp => push_expert(&mut out, &format!("{prefix}mlp.experts.up_proj")),
        Kernel::LlExpertDown => {
            push_expert(&mut out, &format!("{prefix}mlp.experts.down_proj"));
        }

        // The Qwen3.5 mixture's shared expert: a dense FFN under its own
        // names — see the module docs for why these are their own kinds.
        Kernel::LlSharedGate => {
            push_quant(&mut out, &format!("{prefix}mlp.shared_expert.gate_proj"));
        }
        Kernel::LlSharedUp => {
            push_quant(&mut out, &format!("{prefix}mlp.shared_expert.up_proj"));
        }
        Kernel::LlSharedDown => {
            push_quant(&mut out, &format!("{prefix}mlp.shared_expert.down_proj"));
        }
        Kernel::LlSharedGateProj => {
            push_quant(&mut out, &format!("{prefix}mlp.shared_expert_gate"));
        }

        // ── GDN ──
        Kernel::GdnPrep | Kernel::GdnPrepSlotted => {
            bind(
                &mut out,
                slot::GDN_PREP_CONV_W,
                format!("{prefix}linear_attn.conv1d.weight"),
            );
            bind(
                &mut out,
                slot::GDN_PREP_A_LOG,
                format!("{prefix}linear_attn.A_log"),
            );
            bind(
                &mut out,
                slot::GDN_PREP_DT_BIAS,
                format!("{prefix}linear_attn.dt_bias"),
            );
        }
        Kernel::GdnCore | Kernel::GdnCoreSlotted => {
            if gdn_prep || kind == Kernel::GdnCoreSlotted {
                // The slimmed recurrent core: prep owns the gating params;
                // the conv weight stays for the v convsilu.
                bind(
                    &mut out,
                    slot::GDN_CORE_RECURRENT_CONV_W,
                    format!("{prefix}linear_attn.conv1d.weight"),
                );
            } else {
                bind(
                    &mut out,
                    slot::GDN_CORE_CONV_W,
                    format!("{prefix}linear_attn.conv1d.weight"),
                );
                bind(
                    &mut out,
                    slot::GDN_CORE_A_LOG,
                    format!("{prefix}linear_attn.A_log"),
                );
                bind(
                    &mut out,
                    slot::GDN_CORE_DT_BIAS,
                    format!("{prefix}linear_attn.dt_bias"),
                );
            }
        }
        Kernel::GatedRms => bind(
            &mut out,
            slot::GATED_RMS_W,
            format!("{prefix}linear_attn.norm.weight"),
        ),
        Kernel::QmvGate => push_quant(&mut out, &format!("{prefix}mlp.gate_proj")),
        Kernel::QmvUp => push_quant(&mut out, &format!("{prefix}mlp.up_proj")),
        Kernel::QmvDown => push_quant(&mut out, &format!("{prefix}mlp.down_proj")),

        // Everything else is weightless: the movers, sums, elementwise
        // stages and the attention reads bind activations and state only.
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sinks_ride_a_different_slot_on_each_attention_abi() {
        // Index 14 is `SdpaSink::Sinks` on the ring ABI and
        // `AttnMaskEnabled` on the paged one; the C++ served both walks
        // from one Kind and patched the collision with a bind-time remap.
        // Two kinds answer for themselves, so a shared index here would
        // be the collision reintroduced — a weight read as a mask.
        let g = DecodeGeometry::default();
        let ring = weight_binds(Kernel::GoSdpaSink, Some(0), &g, false);
        let paged = weight_binds(Kernel::GoSdpaSinkPaged, Some(0), &g, false);
        assert_eq!(ring[0].tensor, paged[0].tensor, "the SAME learned tensor");
        assert_eq!(ring[0].bind_index, 14);
        assert_eq!(paged[0].bind_index, 16, "clear of AttnMaskEnabled at 14");
    }

    #[test]
    fn a_kind_is_a_weight_name_and_the_layer_prefixes_it() {
        let g = DecodeGeometry::default();
        let binds = weight_binds(Kernel::QmvQ, Some(3), &g, false);
        assert_eq!(binds[0].tensor, "layers.3.self_attn.q_proj.weight");
        assert_eq!(
            binds.iter().map(|b| b.bind_index).collect::<Vec<_>>(),
            [0, 1, 2],
            "the affine triplet's ordinals are frozen"
        );
        // Singletons carry no prefix; a tied head asks for the one table.
        assert_eq!(
            weight_binds(Kernel::QmvLmHead, None, &g, false)[0].tensor,
            "shared_embedding.weight"
        );
    }

    #[test]
    fn mxfp4_experts_leave_the_zero_point_unbound() {
        let mut g = DecodeGeometry {
            mxfp4_experts: true,
            ..DecodeGeometry::default()
        };
        let binds = weight_binds(Kernel::LlExpertGate, Some(0), &g, false);
        assert_eq!(binds.len(), 2, "block exponents, no zero point");
        g.mxfp4_experts = false;
        assert_eq!(
            weight_binds(Kernel::LlExpertGate, Some(0), &g, false).len(),
            3
        );
    }

    #[test]
    fn qwens_experts_ask_for_no_bias_and_gptoss_asks_at_slot_seven() {
        let g = DecodeGeometry::default();
        let qwen = weight_binds(Kernel::LlExpertGate, Some(0), &g, false);
        assert!(
            qwen.iter().all(|b| !b.tensor.ends_with(".bias")),
            "a tensor the checkpoint does not hold is a load failure, not a zero"
        );
        let gptoss = weight_binds(Kernel::GoExpertGate, Some(0), &g, false);
        let bias = gptoss.last().unwrap();
        assert_eq!(bias.bind_index, slot::GO_QMV_BIAS);
        assert!(bias.tensor.ends_with("gate_proj.bias"));
    }

    #[test]
    fn the_gdn_core_split_moves_the_gating_params_to_prep() {
        let g = DecodeGeometry::default();
        let fused = weight_binds(Kernel::GdnCore, Some(1), &g, false);
        assert_eq!(fused.len(), 3, "conv weight + A_log + dt_bias");
        let split = weight_binds(Kernel::GdnCore, Some(1), &g, true);
        assert_eq!(
            split.len(),
            1,
            "the slimmed core keeps only the conv weight"
        );
        let prep = weight_binds(Kernel::GdnPrep, Some(1), &g, true);
        assert_eq!(prep.len(), 3);
    }

    #[test]
    fn weightless_kinds_bind_nothing() {
        let g = DecodeGeometry::default();
        for kind in [
            Kernel::QSplit,
            Kernel::KvAppend,
            Kernel::Sdpa,
            Kernel::SiluMul,
            Kernel::Residual,
            Kernel::LayerOut,
            Kernel::LlMoeSort,
            Kernel::LlMoeCombine,
            Kernel::LlSharedCombine,
        ] {
            assert!(
                weight_binds(kind, Some(0), &g, false).is_empty(),
                "{kind:?}"
            );
        }
    }
}
