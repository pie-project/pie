//! DeepSeek V4 CUDA DSL statements.
use super::*;

// ── deepseek_v4: hyper-connections ─────────────────────────────
// HC is a rank-K residual: layers read a weighted collapse and write back to all streams.
builder! {
    /// `norm::hc_rmsnorm_to_f32`: weightless norm of flattened multi-stream residual.
    pub fn hc_rmsnorm_to_f32(residual: &Val, width: u32) -> Val {
        symbol: "norm::hc_rmsnorm_to_f32",
        on: residual,
        inputs: [residual],
        out: [Dim::Tokens, Dim::Const(width)] as F32,
        made: "the norm produces its value",
    }


    /// `norm::hc_expand_bf16`: replicate embedding into K residual streams.
    pub fn hc_expand(x: &Val, hc_mult: u32, hidden: u32) -> Val {
        symbol: "norm::hc_expand_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(hc_mult), Dim::Const(hidden)] as BF16,
        made: "the expand produces its value",
    }


    /// `norm::hc_pre_postprocess_bf16`: returns `(post_mix, comb_mix, layer_input)`.
    /// `comb_mix` is a per-token `[K, K]` sinkhorn-normalized matrix.
    pub fn hc_pre(mixes: &Val, residual: &Val, hc_mult: u32, hidden: u32) -> (Val, Val, Val) {
        symbol: "norm::hc_pre_postprocess_bf16",
        on: mixes,
        inputs: [mixes, residual],
        outs: [
            [Dim::Tokens, Dim::Const(hc_mult)] as F32,
            [Dim::Tokens, Dim::Const(hc_mult), Dim::Const(hc_mult)] as F32,
            [Dim::Tokens, Dim::Const(hidden)] as BF16,
        ],
        made: "hc_pre states three outputs",
    }


    /// `norm::hc_post_bf16`: fold layer output back into all K streams.
    pub fn hc_post(
        x: &Val,
        residual: &Val,
        post_mix: &Val,
        comb_mix: &Val,
        hc_mult: u32,
        hidden: u32,
    ) -> Val {
        symbol: "norm::hc_post_bf16",
        on: x,
        inputs: [x, residual, post_mix, comb_mix],
        out: [Dim::Tokens, Dim::Const(hc_mult), Dim::Const(hidden)] as BF16,
        made: "the fold produces its value",
    }


    /// `norm::hc_head_postprocess_bf16`: collapse K streams for readout.
    pub fn hc_head(mixes: &Val, residual: &Val, hidden: u32) -> Val {
        symbol: "norm::hc_head_postprocess_bf16",
        on: mixes,
        inputs: [mixes, residual],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the collapse produces its value",
    }


    /// `norm::per_head_rmsnorm_bf16`: weightless RMS norm over head rows.
    pub fn per_head_rmsnorm(x: &Val, heads: u32, head_dim: u32) -> Val {
        symbol: "norm::per_head_rmsnorm_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)] as BF16,
        made: "the norm produces its value",
    }


    /// `norm::attn_sink_correction_bf16`: apply learned per-head sink correction.
    pub fn attn_sink_correction(o: &Val, lse: &Val, sink: &str, heads: u32, head_dim: u32) -> Val {
        symbol: "norm::attn_sink_correction_bf16",
        on: o,
        weights: [sink],
        inputs: [o, lse],
        out: [Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)] as BF16,
        made: "the correction produces its value",
    }
}

// ── deepseek_v4: compressed attention ──────────────────────────
// A compressed KV cache stores one entry per `ratio` tokens; LSE merge makes the split exact.
/// `attn::dsv4_boundary_meta_{decode,paged}`: boundary metadata for compression windows.
/// Returns `(pos, req, rope)`. Non-boundary tokens get `pos = -1`.
/// Outputs are per token on both classes; decode and prefill only choose different launchers.
pub fn dsv4_boundary_meta(positions: &Val, class: model_ir::trace::FireClass) -> (Val, Val, Val) {
    let kernel = match class {
        model_ir::trace::FireClass::Decode => "attn::dsv4_boundary_meta_decode",
        model_ir::trace::FireClass::Prefill => "attn::dsv4_boundary_meta_paged",
    };
    let outs = record_many(
        &positions.t,
        positions.layer,
        kernel,
        vec![],
        vec![positions.id],
        vec![
            (Shape(vec![Dim::Tokens]), DType::I32),
            (Shape(vec![Dim::Tokens]), DType::I32),
            (Shape(vec![Dim::Tokens]), DType::I32),
        ],
    );
    let mut it = outs.into_iter();
    let pos = it.next().expect("the meta states three outputs");
    let req = it.next().expect("the meta states three outputs");
    let rope = it.next().expect("the meta states three outputs");
    (pos, req, rope)
}

builder! {
    /// `attn::dsv4_compress_gather_paged_bf16`: build compressed entries at boundary tokens.
    /// `boundary_req` is [`dsv4_boundary_meta`](crate::cuda::dsv4_boundary_meta)'s second output.
    pub fn dsv4_compress_gather_paged(
        boundary_pos: &Val,
        boundary_req: &Val,
        l: u32,
        head_dim: u32,
    ) -> Val {
        symbol: "attn::dsv4_compress_gather_paged_bf16",
        on: boundary_pos,
        layer: Some(l),
        state: Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        inputs: [boundary_pos, boundary_req],
        out: [Dim::Tokens, Dim::Const(head_dim)] as BF16,
        made: "the gather produces its value",
    }
}

/// `attn::dsv4_store_comp_entries_bf16`: store compressed entries; `boundary_req` matches gather.
pub fn dsv4_store_comp_entries(entries: &Val, boundary_pos: &Val, boundary_req: &Val, l: u32) {
    record(
        &entries.t,
        Some(l),
        "attn::dsv4_store_comp_entries_bf16",
        vec![],
        Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        vec![entries.id, boundary_pos.id, boundary_req.id],
        None,
    );
}

builder! {
    /// `attn::attention_compressed_paged_bf16`: causal attention over compressed cache.
    /// Entry `c` is at absolute position `(c+1)·ratio - 1`.
    pub fn attention_compressed_paged(q: &Val, l: u32, heads: u32, head_dim: u32) -> (Val, Val) {
        symbol: "attn::attention_compressed_paged_bf16",
        on: q,
        layer: Some(l),
        inputs: [q],
        outs: [
            [Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)] as BF16,
            [Dim::Tokens, Dim::Const(heads)] as F32,
        ],
        made: "the attention states two outputs",
    }


    /// `attn::combine_attn_outputs_bf16`: exact LSE merge of fine and compressed attention.
    /// `params[0]=heads`, `params[1]=head_dim`.
    pub fn combine_attn_outputs(
        o1: &Val,
        lse1: &Val,
        o2: &Val,
        lse2: &Val,
        heads: u32,
        head_dim: u32,
    ) -> (Val, Val) {
        symbol: "attn::combine_attn_outputs_bf16",
        on: o1,
        params: [heads, head_dim],
        inputs: [o1, lse1, o2, lse2],
        outs: [
            [Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)] as BF16,
            [Dim::Tokens, Dim::Const(heads)] as F32,
        ],
        made: "the combine states two outputs",
    }


    /// `attn::lse_log2_to_ln`: rebase FlashInfer LSE from log2 to ln.
    pub fn lse_log2_to_ln(lse: &Val, heads: u32) -> Val {
        symbol: "attn::lse_log2_to_ln",
        on: lse,
        inputs: [lse],
        out: [Dim::Tokens, Dim::Const(heads)] as F32,
        made: "the rebase produces its value",
    }
}

// ── deepseek_v4: routing, activation, dequant ──────────────────
// Routers read token/order facts through operands; `on:` still tracks the activation trace.
builder! {
    /// `moe::topk_sqrtsoftplus_bf16`: router scored by `sqrt(softplus(x))`.
    pub fn topk_sqrtsoftplus(logits: &Val, bias: &str, top_k: u32) -> (Val, Val) {
        symbol: "moe::topk_sqrtsoftplus_bf16",
        on: logits,
        weights: [bias],
        inputs: [logits],
        outs: [
            [Dim::Tokens, Dim::Const(top_k)] as I32,
            [Dim::Tokens, Dim::Const(top_k)] as F32,
        ],
        made: "the router states two outputs",
    }


    /// `moe::hash_route_lookup`: expert indices from token-id hash; weights from logits.
    pub fn hash_route_lookup(token_ids: &Val, logits: &Val, table: &str, top_k: u32) -> (Val, Val) {
        symbol: "moe::hash_route_lookup",
        on: logits,
        weights: [table],
        inputs: [token_ids, logits],
        outs: [
            [Dim::Tokens, Dim::Const(top_k)] as I32,
            [Dim::Tokens, Dim::Const(top_k)] as F32,
        ],
        made: "the lookup states two outputs",
    }
}

builder! {
    /// `mlp::chunked_swiglu_clamp_bf16`: packed swiglu with clamped gate.
    /// One operand, packed; pair form is [`swiglu_clamp_pair`](crate::cuda::swiglu_clamp_pair).
    pub fn swiglu_clamp(x: &Val, intermediate: u32) -> Val {
        symbol: "mlp::chunked_swiglu_clamp_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(intermediate)] as BF16,
        made: "the activation produces its value",
    }


    /// `mlp::swiglu_clamp_bf16` pair form: gate and up are separate operands.
    pub fn swiglu_clamp_pair(gate: &Val, up: &Val, intermediate: u32) -> Val {
        symbol: "mlp::swiglu_clamp_bf16",
        on: gate,
        inputs: [gate, up],
        out: [Dim::Tokens, Dim::Const(intermediate)] as BF16,
        made: "the activation produces its value",
    }


    /// `rope::rope_partial_last_q_bf16`: rotate the last `rope_dim` channels of Q only.
    /// Separate symbol: operand/result count must not encode the operation.
    pub fn rope_partial_last(x: &Val, heads: u32, head_dim: u32) -> Val {
        symbol: "rope::rope_partial_last_q_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)] as BF16,
        made: "the rope produces its value",
    }
}

// Weight-shaped dequants stay hand-written: `dequant_fp8_e4m3` chooses a symbol by scale layout,
// and `dequant_mxfp4` has no input value for `builder!`'s `on:` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Scale {
    /// One scale for the whole tensor.
    PerTensor,
    /// One per output channel.
    PerChannel,
    /// One per group along the reduction axis.
    PerGroup,
}

/// `quant::dequant_fp8_e4m3_to_bf16[_per_channel|_per_group]`: widen fp8 to bf16.
pub fn dequant_fp8_e4m3(
    t: &Trace,
    l: u32,
    weight: &str,
    rows: u32,
    cols: u32,
    scale: Fp8Scale,
    per_tensor_scale: f32,
) -> Val {
    // THE FIRST SLOT IS A VALUE NOW, NOT A HOLE.
    //
    // It used to be `vec![0, rows, cols]` -- a literal zero held open so that
    // `rows` would land at index 1, with a comment above it because nothing in
    // the type system said so. The routine's `scale` was
    // `Env<f32, keys::DequantScale>`, a `fact!(stated ..)` key resolving
    // through `Source::Named` that NO DRIVER ANSWERS, so the launcher was
    // unreachable and the hole never showed.
    //
    // `Const<f32>` at parameter 3 puts the scale where it always belonged --
    // `params[0]`, read through the float channel, which is the reading
    // `Handles::param_f32` already gives that slot -- and the arity of the run
    // is declared by the signature, so `model-ir`'s `arity_problem` refuses a
    // statement that carries too few at PLAN time instead of binding a zero.
    //
    // The bits and not the number: the run is a `Vec<u32>` and the BITS are
    // the value. `1.0f32` rides as `0x3f80_0000`, and a conversion would hand
    // the kernel 1065353216.0.
    record_with_params(
        t,
        Some(l),
        match scale {
            Fp8Scale::PerTensor => "quant::dequant_fp8_e4m3_to_bf16",
            Fp8Scale::PerChannel => "quant::dequant_fp8_e4m3_to_bf16_per_channel",
            Fp8Scale::PerGroup => "quant::dequant_fp8_e4m3_to_bf16_per_group",
        },
        vec![weight.to_string()],
        None,
        vec![per_tensor_scale.to_bits(), rows, cols],
        vec![],
        Some((Shape(vec![Dim::Const(rows), Dim::Const(cols)]), DType::BF16)),
    )
    .expect("the dequant produces its value")
}

/// `quant::dequant_mxfp4_to_bf16`: widen MXFP4; scale is E8M0 per block of 32.
pub fn dequant_mxfp4(t: &Trace, l: u32, weight: &str, rows: u32, cols: u32) -> Val {
    record_with_params(
        t,
        Some(l),
        "quant::dequant_mxfp4_to_bf16",
        vec![weight.to_string()],
        None,
        vec![rows, cols],
        vec![],
        Some((Shape(vec![Dim::Const(rows), Dim::Const(cols)]), DType::BF16)),
    )
    .expect("the dequant produces its value")
}
