//! DeepSeek V4 CUDA DSL statements.
use super::*;

// ── deepseek_v4: hyper-connections ─────────────────────────────
// HC is a rank-K residual: layers read a weighted collapse and write back to all streams.
builder! {
    /// `norm::hc_rmsnorm_to_f32`: weightless norm of flattened multi-stream residual.
    pub fn hc_rmsnorm_to_f32(residual: &Val, width: u32, eps: f32) -> Val {
        symbol: "norm::hc_rmsnorm_to_f32",
        on: residual,
        params: [eps.to_bits()],
        inputs: [residual],
        out: [Dim::Tokens, Dim::Const(width)] as F32,
        made: "the norm produces its value",
    }


    /// `norm::hc_expand`: replicate embedding into K residual streams.
    pub fn hc_expand(x: &Val, hc_mult: u32, hidden: u32) -> Val {
        symbol: "norm::hc_expand",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(hc_mult), Dim::Const(hidden)] as BF16,
        made: "the expand produces its value",
    }


    /// `norm::hc_pre_postprocess`: returns `(post_mix, comb_mix, layer_input)`.
    /// `comb_mix` is a per-token `[K, K]` sinkhorn-normalized matrix.
    ///
    /// THE MIX IS AFFINE BEFORE IT IS A TRANSPORT PLAN, and `scale` and
    /// `base` are what makes it so. `norm/dsv4_hc.cuh` reads them as
    /// `[3]` and `[2M + M*M]` and its first act is
    /// `mixes · scale + base` -- three learned gains, one per band of the
    /// mix row, and a bias per entry of it. Two banks, dereferenced per
    /// token, and the statement placed neither: the arm bound whatever the
    /// two weight slots happened to hold.
    ///
    /// Trace names, like `attn_sink` and `router_bias` in the same family
    /// and for the reason `deepseek_v4/project.rs` gives about those two:
    /// no witnessed checkpoint spells them, and a manifest row for a name
    /// nobody has seen is a guess wearing a measurement's clothes.
    /// `[hc_eps, hc_post_alpha, sinkhorn_iters]` is the run, in the
    /// routine's order.
    pub fn hc_pre(
        mixes: &Val,
        residual: &Val,
        scale: &str,
        base: &str,
        hc_mult: u32,
        hidden: u32,
        eps: f32,
        post_alpha: f32,
        sinkhorn_iters: u32,
    ) -> (Val, Val, Val) {
        symbol: "norm::hc_pre_postprocess",
        on: mixes,
        weights: [scale, base],
        params: [eps.to_bits(), post_alpha.to_bits(), sinkhorn_iters],
        inputs: [mixes, residual],
        outs: [
            [Dim::Tokens, Dim::Const(hc_mult)] as F32,
            [Dim::Tokens, Dim::Const(hc_mult), Dim::Const(hc_mult)] as F32,
            [Dim::Tokens, Dim::Const(hidden)] as BF16,
        ],
        made: "hc_pre states three outputs",
    }


    /// `norm::hc_post`: fold layer output back into all K streams.
    pub fn hc_post(
        x: &Val,
        residual: &Val,
        post_mix: &Val,
        comb_mix: &Val,
        hc_mult: u32,
        hidden: u32,
    ) -> Val {
        symbol: "norm::hc_post",
        on: x,
        inputs: [x, residual, post_mix, comb_mix],
        out: [Dim::Tokens, Dim::Const(hc_mult), Dim::Const(hidden)] as BF16,
        made: "the fold produces its value",
    }


    /// `norm::hc_head_postprocess`: collapse K streams for readout.
    ///
    /// The same affine pair as [`hc_pre`], one band narrower: the head
    /// gates are `M` independent sigmoids rather than a transport plan, so
    /// `scale` is `[1]` and `base` is `[M]`. Both are still read per token
    /// and both are still weights.
    pub fn hc_head(
        mixes: &Val,
        residual: &Val,
        scale: &str,
        base: &str,
        hidden: u32,
        eps: f32,
    ) -> Val {
        symbol: "norm::hc_head_postprocess",
        on: mixes,
        weights: [scale, base],
        params: [eps.to_bits()],
        inputs: [mixes, residual],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the collapse produces its value",
    }


    /// `norm::per_head_rmsnorm`: weightless RMS norm over head rows.
    /// `[head_dim, eps]` is the run.
    pub fn per_head_rmsnorm(x: &Val, heads: u32, head_dim: u32, eps: f32) -> Val {
        symbol: "norm::per_head_rmsnorm",
        on: x,
        params: [head_dim, eps.to_bits()],
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)] as BF16,
        made: "the norm produces its value",
    }


    /// `norm::attn_sink_correction`: apply learned per-head sink correction.
    pub fn attn_sink_correction(o: &Val, lse: &Val, sink: &str, heads: u32, head_dim: u32) -> Val {
        symbol: "norm::attn_sink_correction",
        on: o,
        weights: [sink],
        params: [head_dim],
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
pub fn dsv4_boundary_meta(
    positions: &Val,
    class: model_ir::trace::FireClass,
    ratio: u32,
) -> (Val, Val, Val) {
    // Both classes read the row-validity mask after `positions`; the paged
    // form also walks the qo CSR, whose own row count is the request count.
    let row_valid = rt_tokens(&positions.t, "row_valid");
    let mut inputs = vec![positions.id, row_valid];
    let (kernel, params, extents) = match class {
        model_ir::trace::FireClass::Decode => (
            "attn::dsv4_boundary_meta_decode",
            vec![ratio],
            vec![],
        ),
        model_ir::trace::FireClass::Prefill => {
            inputs.push(rt_requests(&positions.t, "qo_indptr"));
            (
                "attn::dsv4_boundary_meta_paged",
                vec![ratio],
                vec![],
            )
        }
    };
    let outs = record_many_with_extents(
        &positions.t,
        positions.layer,
        kernel,
        vec![],
        None,
        params,
        extents,
        inputs,
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

/// `attn::dsv4_compress_gather_paged_bf16`: build compressed entries at boundary tokens.
/// `boundary_req` is [`dsv4_boundary_meta`](crate::cuda::dsv4_boundary_meta)'s second output.
/// The KV view and the three DSV4 residents (state halves and the APE
/// table) are operands; `[ratio, coff]` is the run — and `coff` is a PURE
/// FUNCTION of the ratio (the driver's `compressor_coff` rule: 4 pools 2,
/// else 1), derived HERE so no caller restates the driver's rule.
pub fn dsv4_compress_gather_paged(
    boundary_pos: &Val,
    boundary_req: &Val,
    l: u32,
    head_dim: u32,
    ratio: u32,
) -> Val {
    let coff = if ratio == 4 { 2 } else { 1 };
    let t = &boundary_pos.t;
    let inputs = vec![
        boundary_pos.id,
        boundary_req.id,
        rt_object(t, "kv_cache", Some(l)),
        rt_object(t, "dsv4.state_kv", Some(l)),
        rt_object(t, "dsv4.state_score", Some(l)),
        rt_object(t, "dsv4.ape", None),
    ];
    record_with_params(
        t,
        Some(l),
        "attn::dsv4_compress_gather_paged_bf16",
        vec![],
        Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        vec![ratio, coff],
        inputs,
        Some((Shape(vec![Dim::Tokens, Dim::Const(head_dim)]), DType::BF16)),
    )
    .expect("the gather produces its value")
}

/// `attn::dsv4_store_comp_entries_bf16`: store compressed entries; `boundary_req` matches gather.
pub fn dsv4_store_comp_entries(entries: &Val, boundary_pos: &Val, boundary_req: &Val, l: u32) {
    let kvc = rt_object(&entries.t, "kv_cache", Some(l));
    let comp_kv = rt_object(&entries.t, "dsv4.comp_kv_pages", Some(l));
    record(
        &entries.t,
        Some(l),
        "attn::dsv4_store_comp_entries_bf16",
        vec![],
        Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        vec![entries.id, boundary_pos.id, boundary_req.id, kvc, comp_kv],
        None,
    );
}

/// `attn::attention_compressed_paged_bf16`: causal attention over compressed cache.
/// Entry `c` is at absolute position `(c+1)·ratio - 1`. The KV view, the
/// positions/request-of-token streams and the compressed page pool are
/// operands; `[ratio, num_q_heads, head_dim, sm_scale]` the run.
pub fn attention_compressed_paged(
    q: &Val,
    l: u32,
    heads: u32,
    head_dim: u32,
    ratio: u32,
    sm_scale: f32,
) -> (Val, Val) {
    let t = &q.t;
    let inputs = vec![
        q.id,
        rt_object(t, "kv_cache", Some(l)),
        rt_tokens(t, "positions"),
        rt_tokens(t, "request_of_token"),
        rt_object(t, "dsv4.comp_kv_pages", Some(l)),
    ];
    let outs = record_many_with_params(
        t,
        Some(l),
        "attn::attention_compressed_paged_bf16",
        vec![],
        vec![ratio, heads, head_dim, sm_scale.to_bits()],
        inputs,
        vec![
            (
                Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]),
                DType::BF16,
            ),
            (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32),
        ],
    );
    let mut it = outs.into_iter();
    let o = it.next().expect("the attention states two outputs");
    let lse = it.next().expect("the attention states two outputs");
    (o, lse)
}

builder! {
    /// `attn::combine_attn_outputs`: exact LSE merge of fine and compressed attention.
    /// `params[0]=heads`, `params[1]=head_dim`.
    pub fn combine_attn_outputs(
        o1: &Val,
        lse1: &Val,
        o2: &Val,
        lse2: &Val,
        heads: u32,
        head_dim: u32,
    ) -> (Val, Val) {
        symbol: "attn::combine_attn_outputs",
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
    /// `moe::topk_sqrtsoftplus`: router scored by `sqrt(softplus(x))`.
    /// `[renormalize, routed_scaling_factor]` is the run.
    pub fn topk_sqrtsoftplus(
        logits: &Val,
        bias: &str,
        top_k: u32,
        renormalize: bool,
        routed_scaling: f32,
    ) -> (Val, Val) {
        symbol: "moe::topk_sqrtsoftplus",
        on: logits,
        weights: [bias],
        params: [u32::from(renormalize), routed_scaling.to_bits()],
        inputs: [logits],
        outs: [
            [Dim::Tokens, Dim::Const(top_k)] as I32,
            [Dim::Tokens, Dim::Const(top_k)] as F32,
        ],
        made: "the router states two outputs",
    }


    /// `moe::hash_route_lookup`: expert indices from token-id hash; weights from logits.
    /// `[vocab_size, renormalize, routed_scaling_factor]` is the run.
    pub fn hash_route_lookup(
        token_ids: &Val,
        logits: &Val,
        table: &str,
        top_k: u32,
        vocab: u32,
        renormalize: bool,
        routed_scaling: f32,
    ) -> (Val, Val) {
        symbol: "moe::hash_route_lookup",
        on: logits,
        weights: [table],
        params: [vocab, u32::from(renormalize), routed_scaling.to_bits()],
        inputs: [token_ids, logits],
        outs: [
            [Dim::Tokens, Dim::Const(top_k)] as I32,
            [Dim::Tokens, Dim::Const(top_k)] as F32,
        ],
        made: "the lookup states two outputs",
    }
}

builder! {
    /// `mlp::chunked_swiglu_clamp`: packed swiglu with clamped gate.
    /// One operand, packed; pair form is [`swiglu_clamp_pair`](crate::cuda::swiglu_clamp_pair).
    /// The clamp limit is the checkpoint's and rides the run.
    pub fn swiglu_clamp(x: &Val, intermediate: u32, limit: f32) -> Val {
        symbol: "mlp::chunked_swiglu_clamp",
        on: x,
        params: [limit.to_bits()],
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(intermediate)] as BF16,
        made: "the activation produces its value",
    }


    /// `mlp::swiglu_clamp` pair form: gate and up are separate operands.
    pub fn swiglu_clamp_pair(gate: &Val, up: &Val, intermediate: u32, limit: f32) -> Val {
        symbol: "mlp::swiglu_clamp",
        on: gate,
        params: [limit.to_bits()],
        inputs: [gate, up],
        out: [Dim::Tokens, Dim::Const(intermediate)] as BF16,
        made: "the activation produces its value",
    }


}

/// `rope::rope_partial_last_q_bf16`: rotate the last `rotary_dim` channels of Q only.
/// Separate symbol: operand/result count must not encode the operation.
/// The run, in the routine's order: `[head_dim, rotary_dim, theta,
/// interleaved, yarn_factor, yarn_beta_fast, yarn_beta_slow,
/// yarn_original_max_position]`; positions minted by name.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn rope_partial_last(
    x: &Val,
    heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    interleaved: bool,
    yarn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_original_max_position: u32,
) -> Val {
    let positions = rt_tokens(&x.t, "positions");
    record_with_params(
        &x.t,
        x.layer,
        "rope::rope_partial_last_q_bf16",
        vec![],
        None,
        vec![
            head_dim,
            rotary_dim,
            theta.to_bits(),
            u32::from(interleaved),
            yarn_factor.to_bits(),
            yarn_beta_fast.to_bits(),
            yarn_beta_slow.to_bits(),
            yarn_original_max_position,
        ],
        vec![x.id, positions],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]),
            DType::BF16,
        )),
    )
    .expect("the rope produces its value")
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

/// `quant::dequant_fp8_e4m3_to[_per_channel|_per_group]`: widen fp8 to bf16.
pub fn dequant_fp8_e4m3(
    t: &Trace,
    l: u32,
    weight: &str,
    rows: u32,
    cols: u32,
    scale: Fp8Scale,
    per_tensor_scale: f32,
    group_size: u32,
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
    // EACH VARIANT'S OWN RUN, per its swept signature: the per-tensor form
    // reads `[scale, rows, cols]`, the per-channel form `[rows, cols]` (its
    // scales are a plane, not a number), and the per-group form
    // `[group_size, rows]`.
    let (kernel, params) = match scale {
        Fp8Scale::PerTensor => (
            "quant::dequant_fp8_e4m3_to",
            vec![per_tensor_scale.to_bits(), rows, cols],
        ),
        Fp8Scale::PerChannel => (
            "quant::dequant_fp8_e4m3_to_bf16_per_channel",
            vec![rows, cols],
        ),
        Fp8Scale::PerGroup => (
            "quant::dequant_fp8_e4m3_to_bf16_per_group",
            vec![group_size, rows],
        ),
    };
    record_with_params(
        t,
        Some(l),
        kernel,
        vec![weight.to_string()],
        None,
        params,
        vec![],
        Some((Shape(vec![Dim::Const(rows), Dim::Const(cols)]), DType::BF16)),
    )
    .expect("the dequant produces its value")
}

/// `quant::dequant_mxfp4_to`: widen MXFP4; scale is E8M0 per block of 32.
pub fn dequant_mxfp4(t: &Trace, l: u32, weight: &str, rows: u32, cols: u32) -> Val {
    record_with_params(
        t,
        Some(l),
        "quant::dequant_mxfp4_to",
        vec![weight.to_string()],
        None,
        vec![rows, cols],
        vec![],
        Some((Shape(vec![Dim::Const(rows), Dim::Const(cols)]), DType::BF16)),
    )
    .expect("the dequant produces its value")
}
