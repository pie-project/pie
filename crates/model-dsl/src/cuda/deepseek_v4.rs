//! DeepSeek V4 CUDA DSL statements.
use super::*;

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
        model_ir::trace::FireClass::Decode => {
            ("attn::dsv4_boundary_meta_decode", vec![ratio], vec![])
        }
        model_ir::trace::FireClass::Prefill => {
            inputs.push(rt_requests(&positions.t, "qo_indptr"));
            ("attn::dsv4_boundary_meta_paged", vec![ratio], vec![])
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
