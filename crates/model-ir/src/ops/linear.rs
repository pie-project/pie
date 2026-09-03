use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

/// Learned-weight channel mixing and its epilogues: plain matmuls, fused
/// MLP activations, MoE routing/grouped-matmul, and the low-rank correction
/// that adds a routed bank's `dW*x` to one of them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Linear {
    Matmul {
        act: ValueId,
        w: ValueId,
        y: ValueId,
    },
    LmHead {
        act: ValueId,
        w: ValueId,
        y: ValueId,
    },
    MlpSwiglu {
        packed: ValueId,
        intermediate: u32,
        y: ValueId,
    },
    MlpSwigluClamp {
        packed: ValueId,
        intermediate: u32,
        limit: f32,
        y: ValueId,
    },
    MlpSwigluClampAlpha {
        packed: ValueId,
        intermediate: u32,
        limit: f32,
        alpha: f32,
        y: ValueId,
    },
    /// [`MlpSwigluClamp`](Linear::MlpSwigluClamp)'s arithmetic over two
    /// separate tensors instead of one packed `2*inter` row — needed when a
    /// per-tensor quantization gives `gate_proj`/`up_proj` different scale
    /// planes that can't join into one bank.
    MlpSwigluClampSplit {
        gate: ValueId,
        up: ValueId,
        limit: f32,
        y: ValueId,
    },
    MlpGegluTanh {
        gate: ValueId,
        up: ValueId,
        y: ValueId,
    },
    /// The ungated GELU: `y = gelu_tanh(x)`, no `up` half. Every other gelu
    /// row here multiplies by an `up` half; this one avoids paying the GEMM
    /// and bank twice over to fake an all-ones `up`.
    MlpGeluTanh {
        x: ValueId,
        y: ValueId,
    },
    MlpGegluTanhPacked {
        packed: ValueId,
        intermediate: u32,
        y: ValueId,
    },
    MlpSitu {
        packed: ValueId,
        intermediate: u32,
        beta: f32,
        up_cap: Option<f32>,
        y: ValueId,
    },
    MoeTopkSoftmax {
        logits: ValueId,
        experts: u32,
        top_k: u32,
        routes: ValueId,
        weights: ValueId,
    },
    /// For Gemma 4's mixture: [`Self::MoeTopkSoftmax`]'s softmax, times a
    /// learned per-expert gain gathered by the routing vector this op wrote.
    /// A variant rather than an optional operand, since an unbound buffer is
    /// not a null pointer on either shell.
    MoeTopkSoftmaxScaled {
        logits: ValueId,
        /// `[experts]`, indexed by EXPERT and not by slot.
        scale: ValueId,
        experts: u32,
        top_k: u32,
        routes: ValueId,
        weights: ValueId,
    },
    MoeTopkSigmoid {
        logits: ValueId,
        /// Per-expert correction bias added for the choice only; the weights stay the sigmoid scores.
        bias: Option<ValueId>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: ValueId,
        weights: ValueId,
        /// A prediction of the next layer's routes, read by nobody but the
        /// streamed tier at this router's cut — see `MoeTopkSqrtSoftplus`.
        hint: Option<ValueId>,
    },
    /// Sigmoid routing with a per-expert bias; weights pass through sqrt-softplus.
    ///
    /// **`hint` IS A PREDICTION OF THE NEXT LAYER'S ROUTES, READ BY NOBODY.**
    /// A [`Self::MoePredictRoute`] output the text hands this router so that
    /// it stays LIVE across the router — the streamed tier reads it out of
    /// the arena at this router's segment cut, beside the real routes, and
    /// what it does with it (count, or prefetch) is the tier's. The kernel
    /// never sees it.
    MoeTopkSqrtSoftplus {
        logits: ValueId,
        bias: ValueId,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        hint: Option<ValueId>,
        routes: ValueId,
        weights: ValueId,
    },
    /// **A ROUTE PREDICTION**: the same sqrt-softplus-plus-bias ranking as
    /// [`Self::MoeTopkSqrtSoftplus`], over logits computed EARLY — a later
    /// layer's gate applied to the residual before this layer's experts
    /// land — so that a streamed load can learn which experts the later
    /// layer will want before its router runs. Not a router: no select reads
    /// its `routes`, no segment is cut after it, and its `weights` are
    /// unscaled scores nobody reads.
    MoePredictRoute {
        logits: ValueId,
        bias: ValueId,
        experts: u32,
        top_k: u32,
        routes: ValueId,
        weights: ValueId,
    },
    /// Routing by lookup, not by a gate. `tid2eid` is `[vocab, top_k]` I64:
    /// for every token id it names the `top_k` experts that id routes to, at
    /// uniform weight `1/top_k`. No router logits computed; lands the same
    /// `routes`/`weights` pair every gate above it lands.
    /// The weights are the gate's: sqrt-softplus scores at the table's picks,
    /// renormalized and scaled (the official `Gate.forward`), so `logits` is
    /// read too — only the ranking is skipped.
    MoeHashRoute {
        ids: ValueId,
        tid2eid: ValueId,
        logits: ValueId,
        vocab: u32,
        /// Not read by the kernel; read by every pass that divides a band by it.
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: ValueId,
        weights: ValueId,
    },
    /// **THE STATIC ROUTES OF A GROUPED PROJECTION**: `routes[n, g] = g` for
    /// every token row — the constant [`Self::MatmulGrouped`] walks.
    GroupRoutes {
        groups: u32,
        routes: ValueId,
    },
    /// **A BLOCK-DIAGONAL PROJECTION**: `y[n, g·N + j] = Σₖ x[n, g·K + k] ·
    /// w[g·N + j, k]` over `groups` even blocks of the row — one `[G·N, K]`
    /// plane read as `G` independent `[N, K]` projections of `G` slices of the
    /// input, which is a `[G, N, K]` bank walked by [`Self::GroupRoutes`].
    ///
    /// DeepSeek-V4-Flash's grouped o-projection is this: `wo_a` is `[o_groups
    /// · o_lora, heads · head_dim / o_groups]` and the official forward is
    /// `einsum("bsgd,grd->bsgr", o.view(b, s, groups, -1), wo_a.view(groups,
    /// o_lora, -1))`. Summing the `G` slices first and projecting once — the
    /// reading this tree fired before the reference was run against it —
    /// hands every group the sum of every other group's slice.
    MatmulGrouped {
        x: ValueId,
        w: ValueId,
        routes: ValueId,
        groups: u32,
        y: ValueId,
    },
    /// Grouped matmul: each routed row multiplies the expert `routes` selects from `bank`.
    MoeMatmulSelect {
        x: ValueId,
        bank: ValueId,
        routes: ValueId,
        y: ValueId,
    },
    MoeMatmulSelectBias {
        x: ValueId,
        bank: ValueId,
        bias: ValueId,
        routes: ValueId,
        y: ValueId,
    },
    /// The bias-free twin of `MoeMatmulSelectBias`: a split-plane quantized
    /// bank, nothing added inside the fold. The routed bias a rows-cut
    /// expert wants is said after the reduce, by `MoeBiasSum`.
    MoeMatmulSelectQuant {
        x: ValueId,
        bank: ValueId,
        routes: ValueId,
        y: ValueId,
    },
    /// Folds the top_k routed rows back to one row per token.
    MoeWeightedSum {
        routed: ValueId,
        weights: ValueId,
        y: ValueId,
    },
    /// The routed bias mixture, on an already-folded activation:
    /// `y[t] = x[t] + sum_k weights[t,k] * bias[routes[t,k]]`, applied after
    /// the reduce so it lands exactly once (not once per rank).
    MoeBiasSum {
        x: ValueId,
        bias: ValueId,
        routes: ValueId,
        weights: ValueId,
        y: ValueId,
    },
    MoeSigmoidGateAdd {
        routed: ValueId,
        shared: ValueId,
        gate: ValueId,
        y: ValueId,
    },
    /// The correction class: `y += B[a]*(A[a]*x)`, where `a = routes[row]`
    /// is the adapter this row's lane registered. In place (`y_out` aliases
    /// `y`): a class whose guard does not hold reads `y` unchanged.
    ///
    /// `bank_a` is `[adapters, rank, in]`, `bank_b` is `[adapters, out,
    /// rank]`, indexed by `routes`; the LoRA scale `alpha/r` is folded into
    /// `bank_b` at registration. An adapter shorter than its bank's rank is
    /// registered zero-padded.
    LoraCorrect {
        x: ValueId,
        bank_a: ValueId,
        bank_b: ValueId,
        routes: ValueId,
        y: ValueId,
        y_out: ValueId,
    },
}

impl Operands for Linear {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Matmul { act, w, .. } => sink.extend([*act, *w]),
            Self::LmHead { act, w, .. } => sink.extend([*act, *w]),
            Self::MlpSwiglu { packed, .. } => sink.push(*packed),
            Self::MlpSwigluClamp { packed, .. } => sink.push(*packed),
            Self::MlpSwigluClampAlpha { packed, .. } => sink.push(*packed),
            Self::MlpSwigluClampSplit { gate, up, .. } => sink.extend([*gate, *up]),
            Self::MlpGegluTanh { gate, up, .. } => sink.extend([*gate, *up]),
            Self::MlpGeluTanh { x, .. } => sink.push(*x),
            Self::MlpGegluTanhPacked { packed, .. } => sink.push(*packed),
            Self::MlpSitu { packed, .. } => sink.push(*packed),
            Self::MoeTopkSoftmax { logits, .. } => sink.push(*logits),
            Self::MoeTopkSoftmaxScaled { logits, scale, .. } => sink.extend([*logits, *scale]),
            Self::MoeTopkSigmoid { logits, bias, hint, .. } => {
                sink.push(*logits);
                sink.extend(*bias);
                sink.extend(*hint);
            }
            Self::MoeTopkSqrtSoftplus { logits, bias, hint, .. } => {
                sink.extend([*logits, *bias]);
                sink.extend(*hint);
            }
            Self::MoePredictRoute { logits, bias, .. } => sink.extend([*logits, *bias]),
            Self::MoeHashRoute { ids, tid2eid, logits, .. } => sink.extend([*ids, *tid2eid, *logits]),
            Self::GroupRoutes { .. } => {}
            Self::MatmulGrouped { x, w, routes, .. } => sink.extend([*x, *w, *routes]),
            Self::MoeMatmulSelect { x, bank, routes, .. } => sink.extend([*x, *bank, *routes]),
            Self::MoeMatmulSelectBias { x, bank, bias, routes, .. } => {
                sink.extend([*x, *bank, *bias, *routes]);
            }
            Self::MoeMatmulSelectQuant { x, bank, routes, .. } => sink.extend([*x, *bank, *routes]),
            Self::MoeWeightedSum { routed, weights, .. } => sink.extend([*routed, *weights]),
            Self::MoeBiasSum { x, bias, routes, weights, .. } => {
                sink.extend([*x, *bias, *routes, *weights]);
            }
            Self::MoeSigmoidGateAdd { routed, shared, gate, .. } => {
                sink.extend([*routed, *shared, *gate]);
            }
            Self::LoraCorrect { x, bank_a, bank_b, routes, y, .. } => {
                sink.extend([*x, *bank_a, *bank_b, *routes, *y]);
            }
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Matmul { y, .. } => sink.push(*y),
            Self::LmHead { y, .. } => sink.push(*y),
            Self::MlpSwiglu { y, .. } => sink.push(*y),
            Self::MlpSwigluClamp { y, .. } => sink.push(*y),
            Self::MlpSwigluClampAlpha { y, .. } => sink.push(*y),
            Self::MlpSwigluClampSplit { y, .. } => sink.push(*y),
            Self::MlpGegluTanh { y, .. } => sink.push(*y),
            Self::MlpGeluTanh { y, .. } => sink.push(*y),
            Self::MlpGegluTanhPacked { y, .. } => sink.push(*y),
            Self::MlpSitu { y, .. } => sink.push(*y),
            Self::MoeTopkSoftmax { routes, weights, .. } => sink.extend([*routes, *weights]),
            Self::MoeTopkSoftmaxScaled { routes, weights, .. } => sink.extend([*routes, *weights]),
            Self::MoeTopkSigmoid { routes, weights, .. } => sink.extend([*routes, *weights]),
            Self::MoeTopkSqrtSoftplus { routes, weights, .. } => sink.extend([*routes, *weights]),
            Self::MoePredictRoute { routes, weights, .. } => sink.extend([*routes, *weights]),
            Self::MoeHashRoute { routes, weights, .. } => sink.extend([*routes, *weights]),
            Self::GroupRoutes { routes, .. } => sink.push(*routes),
            Self::MatmulGrouped { y, .. } => sink.push(*y),
            Self::MoeMatmulSelect { y, .. } => sink.push(*y),
            Self::MoeMatmulSelectBias { y, .. } => sink.push(*y),
            Self::MoeMatmulSelectQuant { y, .. } => sink.push(*y),
            Self::MoeWeightedSum { y, .. } => sink.push(*y),
            Self::MoeBiasSum { y, .. } => sink.push(*y),
            Self::MoeSigmoidGateAdd { y, .. } => sink.push(*y),
            Self::LoraCorrect { y_out, .. } => sink.push(*y_out),
        }
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            // Writes through the output it corrects: one arena slot.
            Self::LoraCorrect { y, y_out, .. } => sink.push((*y_out, *y)),
            Self::Matmul { .. }
            | Self::LmHead { .. }
            | Self::MlpSwiglu { .. }
            | Self::MlpSwigluClamp { .. }
            | Self::MlpSwigluClampAlpha { .. }
            | Self::MlpSwigluClampSplit { .. }
            | Self::MlpGegluTanh { .. }
            | Self::MlpGeluTanh { .. }
            | Self::MlpGegluTanhPacked { .. }
            | Self::MlpSitu { .. }
            | Self::MoeTopkSoftmax { .. }
            | Self::MoeTopkSoftmaxScaled { .. }
            | Self::MoeTopkSigmoid { .. }
            | Self::MoeTopkSqrtSoftplus { .. }
            | Self::MoePredictRoute { .. }
            | Self::MoeHashRoute { .. }
            | Self::GroupRoutes { .. }
            | Self::MatmulGrouped { .. }
            | Self::MoeMatmulSelect { .. }
            | Self::MoeMatmulSelectBias { .. }
            | Self::MoeMatmulSelectQuant { .. }
            | Self::MoeWeightedSum { .. }
            | Self::MoeBiasSum { .. }
            | Self::MoeSigmoidGateAdd { .. } => {}
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Matmul { .. } => "linear.matmul",
            Self::LmHead { .. } => "linear.lm_head",
            Self::MlpSwiglu { .. } => "linear.mlp_swiglu",
            Self::MlpSwigluClamp { .. } => "linear.mlp_swiglu_clamp",
            Self::MlpSwigluClampAlpha { .. } => "linear.mlp_swiglu_clamp_alpha",
            Self::MlpSwigluClampSplit { .. } => "linear.mlp_swiglu_clamp_split",
            Self::MlpGegluTanh { .. } => "linear.mlp_geglu_tanh",
            Self::MlpGeluTanh { .. } => "linear.mlp_gelu_tanh",
            Self::MlpGegluTanhPacked { .. } => "linear.mlp_geglu_tanh_packed",
            Self::MlpSitu { .. } => "linear.mlp_situ",
            Self::MoeTopkSoftmax { .. } => "linear.moe_topk_softmax",
            Self::MoeTopkSoftmaxScaled { .. } => "linear.moe_topk_softmax_scaled",
            Self::MoeTopkSigmoid { .. } => "linear.moe_topk_sigmoid",
            Self::MoeTopkSqrtSoftplus { .. } => "linear.moe_topk_sqrt_softplus",
            Self::MoePredictRoute { .. } => "linear.moe_predict_route",
            Self::MoeHashRoute { .. } => "linear.moe_hash_route",
            Self::GroupRoutes { .. } => "linear.group_routes",
            Self::MatmulGrouped { .. } => "linear.matmul_grouped",
            Self::MoeMatmulSelect { .. } => "linear.moe_matmul_select",
            Self::MoeMatmulSelectBias { .. } => "linear.moe_matmul_select_bias",
            Self::MoeMatmulSelectQuant { .. } => "linear.moe_matmul_select_quant",
            Self::MoeWeightedSum { .. } => "linear.moe_weighted_sum",
            Self::MoeBiasSum { .. } => "linear.moe_bias_sum",
            Self::MoeSigmoidGateAdd { .. } => "linear.moe_sigmoid_gate_add",
            Self::LoraCorrect { .. } => "linear.lora_correct",
        }
    }
}
