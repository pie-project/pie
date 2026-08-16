//! Semantic trace ops that lower without a stated kernel.

use super::*;


/// What a statement without a stated kernel lowers to.
pub(super) enum Semantic {
    Structural,
    Kernels(&'static [&'static str]),
    /// Host work the backend raises before the fire's launches run. Carries
    /// the kind rather than a symbol: a prep is not a kernel.
    Prep(model_ir::trace::PrepKind),
    Unlowered(&'static str),
}

/// These symbols are fixed executor behavior unless the trace states a launch.
pub(super) fn semantic(kind: &OpKind, peel_tail: bool) -> Semantic {
    use OpKind::*;
    match kind {
        HookSite { .. } => Semantic::Structural,
        Prep { prep } => Semantic::Prep(*prep),

        Embed { .. } => Semantic::Kernels(&["layout::embed_bf16"]),
        AddBias { .. } => Semantic::Kernels(&["norm::add_bias_bf16"]),
        ResidualAdd => Semantic::Kernels(&["norm::residual_add_bf16"]),

        GdnPrep { .. } => Semantic::Kernels(&["ssm::qwen_gdn_post_conv_prep_bf16"]),
        RmsnormGated { .. } => Semantic::Kernels(&["norm::rmsnorm_gated_fp32_in_bf16"]),
        SplitQGate { .. } => Semantic::Kernels(&["layout::split_q_gate_bf16"]),
        SigmoidGateMul => Semantic::Kernels(&["mlp::sigmoid_gate_inplace_bf16"]),

        // Gemma's `(1 + w)` variant uses a distinct symbol.
        Rmsnorm { variant, .. } | RmsnormPerHead { variant, .. } => {
            Semantic::Kernels(if variant.is_plain() {
                &["norm::rmsnorm_bf16"]
            } else {
                &["norm::rmsnorm_gemma_bf16"]
            })
        }

        SplitQkv { .. } => Semantic::Kernels(if peel_tail {
            &["attn::split_qkv_bf16_devwin"]
        } else {
            &["attn::split_qkv_bf16"]
        }),

        // Partial rope selects a distinct symbol; non-standard rope is residue.
        Rope { kind, partial } => {
            if !matches!(kind, model_ir::trace::RopeKind::Standard) {
                Semantic::Unlowered("only standard rope is emitted")
            } else if partial.is_some() {
                Semantic::Kernels(&["rope::rope_partial_bf16"])
            } else {
                Semantic::Kernels(&["rope::rope_bf16"])
            }
        }

        Matmul { selector, beta_one, .. } => {
            if selector.is_none() {
                // `beta_one` selects the accumulate symbol, which takes the residual operand.
                if *beta_one {
                    Semantic::Kernels(&["gemm::act_x_w_acc"])
                } else {
                    Semantic::Kernels(&["gemm::act_x_w"])
                }
            } else {
                Semantic::Kernels(&["moe::moe_grouped_gemm_bf16"])
            }
        }

        // Lowered traces state these as launches; semantic forms remain residue.
        KvAppend { .. } => Semantic::Unlowered("a lowered trace states the KV write as a launch"),
        Attention { .. } => {
            Semantic::Unlowered("a lowered trace states its attention kernel as a launch")
        }

        Swiglu { .. } => Semantic::Unlowered("the fused-gate_up binding fact is not in the facts"),

        TopK { .. } => Semantic::Kernels(&["moe::topk_softmax_bf16"]),
        // Token-batched combine; other forms are stated launches.
        WeightedSum { .. } => Semantic::Kernels(&["moe::token_batched_weighted_sum_bf16"]),
        SigmoidGateAdd => Semantic::Kernels(&["mlp::sigmoid_dot_scalar_gate_add_bf16"]),

        LmHead { .. } => Semantic::Structural,

        Select { .. } => Semantic::Structural,
        _ => Semantic::Unlowered("no lowering rule for this kind"),
    }
}

/// The kind's name, for a refusal a human reads.
pub(super) fn kind_name(kind: &OpKind) -> &'static str {
    use OpKind::*;
    match kind {
        Embed { .. } => "Embed",
        Matmul { .. } => "Matmul",
        Select { .. } => "Select",
        Rmsnorm { .. } => "Rmsnorm",
        AddBias { .. } => "AddBias",
        RmsnormPerHead { .. } => "RmsnormPerHead",
        SplitQkv { .. } => "SplitQkv",
        Rope { .. } => "Rope",
        KvAppend { .. } => "KvAppend",
        Attention { .. } => "Attention",
        Swiglu { .. } => "Swiglu",
        LmHead { .. } => "LmHead",
        ResidualAdd => "ResidualAdd",
        TopK { .. } => "TopK",
        WeightedSum { .. } => "WeightedSum",
        SigmoidGateAdd => "SigmoidGateAdd",
        SplitGdn { .. } => "SplitGdn",
        CausalConv1d { .. } => "CausalConv1d",
        GdnPrep { .. } => "GdnPrep",
        GatedDelta { .. } => "GatedDelta",
        RmsnormGated { .. } => "RmsnormGated",
        SplitQGate { .. } => "SplitQGate",
        SigmoidGateMul => "SigmoidGateMul",
        Launch { .. } => "Launch",
        Guard { .. } => "Guard",
        Prep { .. } => "Prep",
        HookSite { .. } => "HookSite",
        Peel { .. } => "Peel",
    }
}

/// Rows matching a lowered axis must be contiguous by seriation.
pub(super) fn contiguous(
    rows: &[Row],
    window: &Range<u32>,
    holds: fn(&Row) -> bool,
    axis: &'static str,
    at: usize,
) -> Result<Range<u32>, Uncovered> {
    let mut start = None;
    let mut end = window.start;
    for i in window.clone() {
        if holds(&rows[i as usize]) {
            if start.is_none() {
                start = Some(i);
            } else if end != i {
                return Err(Uncovered::Discontiguous { at_op: at, axis });
            }
            end = i + 1;
        }
    }
    Ok(match start {
        Some(s) => s..end,
        None => window.start..window.start,
    })
}

/// Subtracting an arm must leave one contiguous range.
pub(super) fn subtract(window: &Range<u32>, taken: &Range<u32>, at: usize) -> Result<Range<u32>, Uncovered> {
    if taken.start == window.start {
        Ok(taken.end..window.end)
    } else if taken.end == window.end {
        Ok(window.start..taken.start)
    } else {
        Err(Uncovered::Discontiguous {
            at_op: at,
            axis: "arm",
        })
    }
}
