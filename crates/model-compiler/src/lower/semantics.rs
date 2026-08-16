//! WHAT A STATEMENT WITHOUT A STATED KERNEL LOWERS TO.

use super::*;

// ── The semantic statements ────────────────────────────────────────────

/// What a statement that does NOT state its kernel lowers to.
pub(super) enum Semantic {
    /// Launches nothing from the kernel table: a structural marker.
    Structural,
    /// The kernels it launches, in order. Usually one — the kinds whose
    /// own doc comments say "one op because it is one launch".
    Kernels(&'static [&'static str]),
    /// It runs on the device, but the trace does not say what it runs.
    /// The payload is what the trace would have to state.
    Unlowered(&'static str),
}

/// The kernels a semantic statement launches, read off the executor that
/// launches them today (`crates/driver-cuda/csrc/src/model/llama_like/
/// declared_forward.cpp`), not guessed.
///
/// Most of these arms are the ones the doc's Amendment A diagnosed: they
/// branch, but on which BUFFER (`ws.norm_x` vs `ws.y` vs the value slot),
/// never on which kernel. Strip the buffer question — [`Buffers`] owns it
/// — and what is left is 1:1, which is why the flat list can carry them.
///
/// Where an arm genuinely picks a kernel, the pick is either a REGION
/// question the lowering already knows (`peel_tail`) or a fact the trace
/// does not carry, and the second is [`Semantic::Unlowered`] rather than
/// a guess.
pub(super) fn semantic(kind: &OpKind, peel_tail: bool) -> Semantic {
    use OpKind::*;
    match kind {
        // The sites are argument no-ops with nothing attached, and with
        // programs attached what they run is guest sideband plus the
        // bracket machinery (page-view reset, score staging) — never a
        // table kernel. Stating that bracket is what `seam!` is for.
        HookSite { .. } => Semantic::Structural,

        Embed { .. } => Semantic::Kernels(&["layout::embed_bf16"]),
        AddBias { .. } => Semantic::Kernels(&["norm::add_bias_bf16"]),
        ResidualAdd => Semantic::Kernels(&["norm::residual_add_bf16"]),

        // The GDN and full-attention kinds. Each is ONE kernel with no
        // branch — no fact to read, no variant to dispatch on, nothing
        // chosen per fire. They were residue only because the rule was
        // never written: the qwen3_5 executor walks, so nothing ever
        // asked the lowering what they were.
        //
        // Their operand plumbing (the per-layer `la.*` scratch, the fp32
        // parameter banks) is the EMITTER's, exactly as it is for the
        // kinds above — naming the symbol is what the lowering owes.
        GdnPrep { .. } => Semantic::Kernels(&["ssm::qwen_gdn_post_conv_prep_bf16"]),
        RmsnormGated { .. } => Semantic::Kernels(&["norm::rmsnorm_gated_fp32_in_bf16"]),
        SplitQGate { .. } => Semantic::Kernels(&["layout::split_q_gate_bf16"]),
        SigmoidGateMul => Semantic::Kernels(&["mlp::sigmoid_gate_inplace_bf16"]),

        // Gemma folds `(1 + w)` — different arithmetic, so a different
        // kernel, but the same signature and the same row space. The
        // variant is already on the wire (`param0`), so naming the
        // symbol is the whole of what the lowering owes; the executor
        // reads the same field to pick.
        //
        // The per-head kind differs only in its ROW COUNT (`N * heads`
        // rows of `head_dim` rather than `N` of `hidden`), which the
        // executor derives from the weight's geometry either way — so
        // both kinds fan onto the same pair.
        Rmsnorm { variant, .. } | RmsnormPerHead { variant, .. } => {
            Semantic::Kernels(if variant.is_plain() {
                &["norm::rmsnorm_bf16"]
            } else {
                &["norm::rmsnorm_gemma_bf16"]
            })
        }

        // Inside a peel's tail the split serves absolute row offsets in a
        // full-N buffer, which is a different kernel — and the REGION is
        // what asks for it, so the lowering states it rather than the
        // driver deriving it from a window pointer.
        SplitQkv { .. } => Semantic::Kernels(if peel_tail {
            &["attn::split_qkv_bf16_devwin"]
        } else {
            &["attn::split_qkv_bf16"]
        }),

        // Partial rope IS a different kernel, and the trace already says
        // which: the rotary width crosses as `param1`, zero for the full
        // rotation. So the lowering names the pair the same way it names
        // the norm's, and the width the executor needs is the width the
        // declaration already carried.
        Rope { kind, partial } => {
            if !matches!(kind, model_ir::trace::RopeKind::Standard) {
                Semantic::Unlowered("only standard rope is emitted")
            } else if partial.is_some() {
                Semantic::Kernels(&["rope::rope_partial_bf16"])
            } else {
                Semantic::Kernels(&["rope::rope_bf16"])
            }
        }

        // A selector makes the weight per-token, and grouped GEMM is that
        // op's lowering — a different call with a different argument
        // shape, chosen per fire.
        Matmul { selector, .. } => {
            if selector.is_none() {
                Semantic::Kernels(&["gemm::act_x_w"])
            } else {
                // A selector makes the weight per-token, and the grouped
                // GEMM is that op's lowering. It used to be a refusal
                // because no text stated the kernel; `moe_mlp_body_cuda`'s
                // general leg does now.
                Semantic::Kernels(&["moe::moe_grouped_gemm_bf16"])
            }
        }

        // Both of these THROW when a class trace reaches this executor
        // with them — a lowered trace states its KV write and its
        // attention as stated-kernel launches. So they cannot appear, and
        // if they do the honest answer is the same refusal.
        KvAppend { .. } => Semantic::Unlowered("a lowered trace states the KV write as a launch"),
        Attention { .. } => {
            Semantic::Unlowered("a lowered trace states its attention kernel as a launch")
        }

        // The packed-bank form when the checkpoint materialised a fused
        // gate_up. That is a BINDING fact, which the taxonomy puts in the
        // facts and erases at trace time — but no fact carries it today,
        // so the executor reads the workspace and picks. The trace has to
        // state it before this statement can be a rectangle.
        Swiglu { .. } => Semantic::Unlowered("the fused-gate_up binding fact is not in the facts"),

        // The MoE branch's three statements, each refused BY NAME until
        // `moe_mlp_body_cuda` states its kernel. They are grouped here
        // because they share one cause, and a residue ledger that says
        // "no lowering rule for this kind" three times would read as
        // three gaps instead of one missing text.
        // The router. One launch, and the semantic reading takes the
        // softmax form -- a text that wants the sigmoid or sqrt-softplus
        // router states it as a `Launch` instead.
        TopK { .. } => Semantic::Kernels(&["moe::topk_softmax_bf16"]),
        // The combine, in its TOKEN-BATCHED form. The two other forms --
        // the per-expert scatter-add and the fused +residual -- are what a
        // CUDA text states as launches when its binding takes them; this
        // is the reading a SEMANTIC trace gets, the same way `Swiglu`'s
        // unpacked form is.
        WeightedSum { .. } => Semantic::Kernels(&["moe::token_batched_weighted_sum_bf16"]),
        // The shared expert's landing: `sigmoid(x·g)` scaling the shared
        // output onto the routed sum, one launch.
        SigmoidGateAdd => Semantic::Kernels(&["mlp::sigmoid_dot_scalar_gate_add_bf16"]),

        // Handled by `Lowerer::epilogue`, which needs the row counts and
        // so cannot answer from the kind alone.
        LmHead { .. } => Semantic::Structural,

        // A window, not a launch: `Buffers` gives its value an offset
        // into its operand's, and there is no rectangle to emit. That is
        // the whole of what `Select` means.
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
        HookSite { .. } => "HookSite",
        Peel { .. } => "Peel",
    }
}

/// The rows of `window` satisfying `holds`, refusing a non-contiguous
/// answer — the seriation's guarantee, checked rather than assumed.
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

/// `window` minus `taken`, which must leave a contiguous remainder.
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
