//! Mixture paths: WNA16, MXFP4, nemotron_h dispatch, and aligned dispatch.
use super::*;

// ── kimi: WNA16 quantized MoE ─────────────────────────────────
// Int4-b8 weights with bf16 group scales; distinct from MXFP4/fp8 because
// checkpoint quantization is a declaration fact.
pub fn dequant_wna16_int4b8(
    t: &Trace,
    l: u32,
    w: &str,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
) -> Val {
    // `params[0]` is the group size, `[1..3]` the weight's own shape.
    record_with_params(
        t,
        Some(l),
        "quant::dequant_wna16_int4b8_to_bf16",
        vec![w.to_string()],
        None,
        vec![group_size, out_dim, in_dim],
        vec![],
        Some((
            Shape(vec![Dim::Const(out_dim), Dim::Const(in_dim)]),
            DType::BF16,
        )),
    )
    .expect("the dequant produces its value")
}

builder! {
    /// `quant::wna16_gate_up_decode_bf16`: decode gate/up from packed weights.
    /// `topk_idx` is `[N, K]` in token order, not route-major.
    /// Weight order: gate_packed, gate_scale, up_packed, up_scale.
    pub fn wna16_gate_up_decode(
        act: &Val,
        topk_idx: &Val,
        intermediate: u32,
        bank: &str,
    ) -> (Val, Val) {
        symbol: "quant::wna16_gate_up_decode_bf16",
        on: act,
        weights: [
            format!("{bank}.gate_packed"),
            format!("{bank}.gate_scale"),
            format!("{bank}.up_packed"),
            format!("{bank}.up_scale"),
        ],
        inputs: [act, topk_idx],
        outs: [
            [Dim::Tokens, Dim::Const(intermediate)] as BF16,
            [Dim::Tokens, Dim::Const(intermediate)] as BF16,
        ],
        made: "the projection states two outputs",
    }
}

builder! {
    /// `quant::wna16_down_decode_bf16`: decode down projection.
    pub fn wna16_down_decode(act: &Val, topk_idx: &Val, hidden: u32, bank: &str) -> Val {
        symbol: "quant::wna16_down_decode_bf16",
        on: act,
        weights: [format!("{bank}.down_packed"), format!("{bank}.down_scale")],
        inputs: [act, topk_idx],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the projection produces its value",
    }


    /// `norm::rmsnorm_strided_bf16`: norm a prefix of wider rows; stride keeps the real row pitch.
    pub fn rmsnorm_strided(x: &Val, weight: &str, hidden: u32) -> Val {
        symbol: "norm::rmsnorm_strided_bf16",
        on: x,
        weights: [weight],
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the norm produces its value",
    }
}

// ── mixtral / gpt-oss: MXFP4 MoE ──────────────────────────────
// Some statements are weight-shaped launches with no token extent.
builder! {
    /// `moe::add_moe_route_bias_bf16`: add route expert bias.
    /// `whole`: `topk_idx` is route-global.
    /// `params[0]` is the destination row pitch (`out_stride`), equal to `width` at this site.
    /// It is not `param_extents`: rows come from the lowered region, pitch from the allocation.
    /// The kernel accumulates into `x`; the signature row must mark output 0 in-place with input 0.
    pub fn add_moe_route_bias(x: &Val, topk_idx: &Val, bias: &str, width: u32) -> Val {
        symbol: "moe::add_moe_route_bias_bf16",
        on: x,
        weights: [bias],
        params: [width],
        inputs: [x, topk_idx],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the bias add produces its value",
    }
}

builder! {
    /// `norm::rmsnorm_bf16_with_fp16`: publish bf16 and fp16; MXFP4 grouped GEMM consumes fp16.
    pub fn rmsnorm_with_fp16(x: &Val, weight: &str, hidden: u32) -> (Val, Val) {
        symbol: "norm::rmsnorm_bf16_with_fp16",
        on: x,
        weights: [weight],
        inputs: [x],
        outs: [
            [Dim::Tokens, Dim::Const(hidden)] as BF16,
            [Dim::Tokens, Dim::Const(hidden)] as F16,
        ],
        made: "the norm states two outputs",
    }


    /// `rope::rope_write_kv_bf16`: rope q/k and commit k/v pages in one launch.
    pub fn rope_write_kv(q: &Val, k: &Val, v: &Val, l: u32, q_width: u32) -> Val {
        symbol: "rope::rope_write_kv_bf16",
        on: q,
        layer: Some(l),
        state: Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        inputs: [q, k, v],
        out: [Dim::Tokens, Dim::Const(q_width)] as BF16,
        made: "the fused rope+write produces its value",
    }
}

/// `quant::mxfp4_scales_to_marlin_e8m0`: repack E8M0 scales into Marlin order.
pub fn mxfp4_scales_to_marlin(t: &Trace, l: u32, w: &str, groups: u32, rows: u32) -> Val {
    record(
        t,
        Some(l),
        "quant::mxfp4_scales_to_marlin_e8m0",
        vec![w.to_string()],
        None,
        vec![],
        Some((
            Shape(vec![Dim::Const(groups), Dim::Const(rows)]),
            DType::I32,
        )),
    )
    .expect("the repack produces its value")
}

/// `moe::transpose_expert_scales_u8`: `[E, n, k/32]` to `[E, k/32, n]`.
/// No inputs or params: this is weight-loader work, not a fire-carried launch.
pub fn transpose_expert_scales(
    t: &Trace,
    l: u32,
    w: &str,
    experts: u32,
    k_groups: u32,
    n: u32,
) -> Val {
    record(
        t,
        Some(l),
        "moe::transpose_expert_scales_u8",
        vec![w.to_string()],
        None,
        vec![],
        Some((
            Shape(vec![
                Dim::Const(experts),
                Dim::Const(k_groups),
                Dim::Const(n),
            ]),
            DType::I32,
        )),
    )
    .expect("the transpose produces its value")
}

builder! {
    /// `moe::topk_sigmoid_bias_fp32`: fp32 router with per-expert correction bias.
    pub fn topk_sigmoid_bias(logits: &Val, bias: &str, top_k: u32) -> (Val, Val) {
        symbol: "moe::topk_sigmoid_bias_fp32",
        on: logits,
        weights: [bias],
        inputs: [logits],
        outs: [
            [Dim::Tokens, Dim::Const(top_k)] as I32,
            [Dim::Tokens, Dim::Const(top_k)] as F32,
        ],
        made: "the router states two outputs",
    }


    /// `moe::moe_bucket_exact`: unpadded bucket by expert.
    /// Returns `(sorted_route_ids, route_to_sorted_row, counts)` in kernel parameter order.
    /// All three outputs are required; the inverse map has no null guard.
    /// Shapes carry launcher extents: routes are `Tokens * top_k`, counts width is `num_experts`.
    pub fn moe_bucket_exact(topk_idx: &Val, num_experts: u32, top_k: u32) -> (Val, Val, Val) {
        symbol: "moe::moe_bucket_exact",
        on: topk_idx,
        inputs: [topk_idx],
        outs: [
            [Dim::Tokens, Dim::Const(top_k)] as I32,
            [Dim::Tokens, Dim::Const(top_k)] as I32,
            [Dim::Const(num_experts)] as I32,
        ],
        made: "the bucket states three outputs",
    }
}

/// `ssm::build_nemotron_moe_ptrs_aligned_bf16`: aligned batched-GEMM pointer arrays.
pub fn build_nemotron_moe_ptrs_aligned(expert_ids: &Val, aligned_in: &Val, l: u32) {
    record(
        &expert_ids.t,
        Some(l),
        "ssm::build_nemotron_moe_ptrs_aligned_bf16",
        vec![],
        None,
        vec![expert_ids.id, aligned_in.id],
        None,
    );
}

/// `ssm::build_nemotron_moe_ptrs_decode_batched_bf16`: decode pointer arrays.
pub fn build_nemotron_moe_ptrs_decode(topk_idx: &Val, topk_w: &Val, x: &Val, l: u32) {
    record(
        &topk_idx.t,
        Some(l),
        "ssm::build_nemotron_moe_ptrs_decode_batched_bf16",
        vec![],
        None,
        vec![topk_idx.id, topk_w.id, x.id],
        None,
    );
}

// ── MoE: aligned dispatch path ────────────────────────────────
// Routes are bucketed and padded globally; whole statements use route order from the full fire.
builder! {
    /// `moe::topk_sigmoid_bf16`: per-token sigmoid router.
    /// Returns `(topk_idx, topk_w)` and is not `whole`.
    pub fn topk_sigmoid(logits: &Val, top_k: u32) -> (Val, Val) {
        symbol: "moe::topk_sigmoid_bf16",
        on: logits,
        inputs: [logits],
        outs: [
            [Dim::Tokens, Dim::Const(top_k)] as I32,
            [Dim::Tokens, Dim::Const(top_k)] as F32,
        ],
        made: "the router states two outputs",
    }


    /// `moe::moe_align_decode`: bucket routes and pad buckets.
    /// `params[0]=num_experts`, `params[1]=block_size`, `params[2]=max_blocks`.
    /// Returns `(sorted_route_ids, expert_ids, route_to_aligned_row)`.
    pub fn moe_align(
        topk_idx: &Val,
        max_blocks: u32,
        block_size: u32,
        top_k: u32,
        num_experts: u32,
    ) -> (Val, Val, Val) {
        symbol: "moe::moe_align_decode",
        on: topk_idx,
        params: [num_experts, block_size, max_blocks],
        inputs: [topk_idx],
        outs: [
            [Dim::Const(max_blocks * block_size)] as I32,
            [Dim::Const(max_blocks)] as I32,
            [Dim::Tokens, Dim::Const(top_k)] as I32,
        ],
        made: "the align states three outputs",
    }


    /// `moe::gather_moe_aligned_inputs_bf16`: gather block-major operands.
    /// `params[0]=top_k`, needed to compute `num_routes`; no operand/result carries k.
    pub fn gather_moe_aligned_inputs(
        x: &Val,
        sorted_route_ids: &Val,
        aligned: Dim,
        hidden: u32,
        top_k: u32,
    ) -> Val {
        symbol: "moe::gather_moe_aligned_inputs_bf16",
        on: x,
        params: [top_k],
        inputs: [x, sorted_route_ids],
        out: [aligned, Dim::Const(hidden)] as BF16,
        made: "the gather produces its value",
    }
}

builder! {
    /// `moe::build_moe_ptrs_aligned_bf16`: declare staging buffers and build grouped-GEMM pointers.
    /// Outputs `(gate_up, act, out)` as `[aligned, 2I]`, `[aligned, I]`, `[aligned, H]`.
    /// Pointer arrays remain driver-owned; this vocabulary has no dtype for them.
    pub fn build_moe_ptrs_aligned(
        expert_ids: &Val,
        aligned_in: &Val,
        l: u32,
        gate_up_bank: &str,
        down_bank: &str,
        aligned: Dim,
        hidden: u32,
        moe_intermediate: u32,
    ) -> (Val, Val, Val) {
        symbol: "moe::build_moe_ptrs_aligned_bf16",
        on: expert_ids,
        weights: [gate_up_bank, down_bank],
        layer: Some(l),
        inputs: [expert_ids, aligned_in],
        outs: [
            [aligned, Dim::Const(2 * moe_intermediate)] as BF16,
            [aligned, Dim::Const(moe_intermediate)] as BF16,
            [aligned, Dim::Const(hidden)] as BF16,
        ],
        made: "the ptr build states three stages",
    }


    /// `moe::reorder_moe_aligned_output_bf16`: undo block permutation to route order.
    pub fn reorder_moe_aligned_output(
        aligned_out: &Val,
        sorted_route_ids: &Val,
        top_k: u32,
        hidden: u32,
    ) -> Val {
        symbol: "moe::reorder_moe_aligned_output_bf16",
        on: aligned_out,
        // `params[0]=top_k`; no operand here carries router width.
        params: [top_k],
        inputs: [aligned_out, sorted_route_ids],
        out: [Dim::Tokens, Dim::Const(top_k), Dim::Const(hidden)] as BF16,
        made: "the reorder produces its value",
    }
}
