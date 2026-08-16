//! DEEPSEEK V4 — hyper-connections, compressed attention, and the
//! routing/activation/dequant statements those two need.

use super::*;

// ── deepseek_v4: hyper-connections ─────────────────────────────
//
// The SECOND rank-K residual scheme in this table, and it is not AltUp's.
// gemma-3n predicts each stream from a learned linear combination and
// corrects from one active stream; HC mixes with a per-token matrix that
// has been sinkhorn-normalized, and there is no active stream — every
// layer reads a weighted collapse of all of them and writes back to all
// of them. Two answers to "what if the residual had a rank", worth being
// able to state separately.

/// `kernels::norm::hc_rmsnorm_to_f32`: norm the flattened multi-stream
/// residual into the fp32 the mixing GEMM wants.
pub fn hc_rmsnorm_to_f32(residual: &Val, weight: &str, width: u32) -> Val {
    record(
        &residual.t,
        residual.layer,
        "norm::hc_rmsnorm_to_f32",
        vec![weight.to_string()],
        None,
        vec![residual.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::F32)),
    )
    .expect("the norm produces its value")
}

/// `kernels::norm::hc_expand_bf16`: replicate the embedding into K
/// streams, at the top of the stack.
///
/// Where a rank-K residual BEGINS. AltUp's equivalent is implicit in how
/// gemma-3n lays out its workspace; HC states it, which is the better of
/// the two and the one a declaration can read.
pub fn hc_expand(x: &Val, hc_mult: u32, hidden: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "norm::hc_expand_bf16",
        vec![],
        None,
        vec![x.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(hc_mult), Dim::Const(hidden)]),
            DType::BF16,
        )),
    )
    .expect("the expand produces its value")
}

/// `kernels::norm::hc_pre_postprocess_bf16`: turn the mixing GEMM's
/// output into `(post_mix, comb_mix, layer_input)`.
///
/// `comb_mix` is a `[hc_mult, hc_mult]` matrix PER TOKEN, sinkhorn-
/// normalized so the mixing is doubly stochastic. `layer_input` is the
/// single stream the layer body actually runs on.
pub fn hc_pre(mixes: &Val, residual: &Val, hc_mult: u32, hidden: u32) -> (Val, Val, Val) {
    let outs = record_many(
        &mixes.t,
        mixes.layer,
        "norm::hc_pre_postprocess_bf16",
        vec![],
        vec![mixes.id, residual.id],
        vec![
            (Shape(vec![Dim::Tokens, Dim::Const(hc_mult)]), DType::F32),
            (
                Shape(vec![Dim::Tokens, Dim::Const(hc_mult), Dim::Const(hc_mult)]),
                DType::F32,
            ),
            (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16),
        ],
    );
    let mut it = outs.into_iter();
    let post_mix = it.next().expect("hc_pre states three outputs");
    let comb_mix = it.next().expect("hc_pre states three outputs");
    let layer_input = it.next().expect("hc_pre states three outputs");
    (post_mix, comb_mix, layer_input)
}

/// `kernels::norm::hc_post_bf16`: fold the layer's output back into all
/// K streams — `new_residual_j = comb_mix_ij · residual_i + post_mix_j · x`.
pub fn hc_post(
    x: &Val,
    residual: &Val,
    post_mix: &Val,
    comb_mix: &Val,
    hc_mult: u32,
    hidden: u32,
) -> Val {
    record(
        &x.t,
        x.layer,
        "norm::hc_post_bf16",
        vec![],
        None,
        vec![x.id, residual.id, post_mix.id, comb_mix.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(hc_mult), Dim::Const(hidden)]),
            DType::BF16,
        )),
    )
    .expect("the fold produces its value")
}

/// `kernels::norm::hc_head_postprocess_bf16`: collapse the K streams to
/// one, for the readout.
pub fn hc_head(mixes: &Val, residual: &Val, hidden: u32) -> Val {
    record(
        &mixes.t,
        mixes.layer,
        "norm::hc_head_postprocess_bf16",
        vec![],
        None,
        vec![mixes.id, residual.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the collapse produces its value")
}

/// `kernels::norm::per_head_rmsnorm_bf16`: an RMS norm whose rows are
/// heads rather than the residual width.
pub fn per_head_rmsnorm(x: &Val, weight: &str, heads: u32, head_dim: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "norm::per_head_rmsnorm_bf16",
        vec![weight.to_string()],
        None,
        vec![x.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]),
            DType::BF16,
        )),
    )
    .expect("the norm produces its value")
}

/// `kernels::norm::attn_sink_correction_bf16`: the learned per-head sink
/// term, applied as a correction to the attention output.
pub fn attn_sink_correction(o: &Val, lse: &Val, sink: &str, heads: u32, head_dim: u32) -> Val {
    record(
        &o.t,
        o.layer,
        "norm::attn_sink_correction_bf16",
        vec![sink.to_string()],
        None,
        vec![o.id, lse.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]),
            DType::BF16,
        )),
    )
    .expect("the correction produces its value")
}

// ── deepseek_v4: compressed attention ──────────────────────────
//
// A SECOND KV cache beside the fine-grained one, holding a single entry
// per `ratio` tokens. Every query attends both, and the two outputs are
// merged by their log-sum-exps -- which is what makes it exact rather
// than an approximation: the merge is the same algebra flashinfer's own
// KV-split uses.

/// `kernels::attn::dsv4_boundary_meta_decode`: which decode tokens close
/// a compression window, CUDA-graph-safely.
///
/// Returns `(pos, req, rope)`. A token whose position is not a boundary
/// gets `pos = -1`, which the gather zero-fills and the store skips --
/// so the shape is fixed and the graph replays.
/// `kernels::attn::dsv4_boundary_meta_{decode,paged}`: which tokens close
/// a compression window, CUDA-graph-safely.
///
/// Returns `(pos, req, rope)`. A token whose position is not a boundary
/// gets `pos = -1`, which the gather zero-fills and the store skips —
/// so the shape is fixed and the graph replays.
///
/// **Per TOKEN, on both classes.** The outputs used to be stated as
/// `Dim::Requests`, which is true only on a pure-decode fire — the doc on
/// [`Dim::Tokens`] says the two coincide there — and that spelling was the
/// decode assumption leaking out of the kernel and into the shape. Whether
/// a position closes a window is a fact about that position, so the extent
/// is the fire's token rows and always was.
///
/// The class picks the launcher, and they differ in one line: decode may
/// shortcut the request index to the token index, and a prefill has to read
/// it out of `qo_indptr`.
pub fn dsv4_boundary_meta(
    positions: &Val,
    class: model_ir::trace::FireClass,
) -> (Val, Val, Val) {
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

/// `kernels::attn::dsv4_compress_gather_paged_bf16`: build one compressed
/// entry per boundary token, by a per-dimension softmax over the gate
/// scores of the window ending there.
pub fn dsv4_compress_gather_paged(boundary_pos: &Val, l: u32, head_dim: u32) -> Val {
    record(
        &boundary_pos.t,
        Some(l),
        "attn::dsv4_compress_gather_paged_bf16",
        vec![],
        Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        vec![boundary_pos.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(head_dim)]), DType::BF16)),
    )
    .expect("the gather produces its value")
}

/// `kernels::attn::dsv4_store_comp_entries_bf16`: commit those entries to
/// the compressed cache, at the boundary token's own slot.
pub fn dsv4_store_comp_entries(entries: &Val, boundary_pos: &Val, l: u32) {
    record(
        &entries.t,
        Some(l),
        "attn::dsv4_store_comp_entries_bf16",
        vec![],
        Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        vec![entries.id, boundary_pos.id],
        None,
    );
}

/// `kernels::attn::attention_compressed_paged_bf16`: attend the
/// compressed cache, causally.
///
/// Entry `c` lives at absolute position `(c+1)·ratio - 1`, and a query at
/// `p` may attend it iff that boundary is `<= p`.
pub fn attention_compressed_paged(q: &Val, l: u32, heads: u32, head_dim: u32) -> (Val, Val) {
    let outs = record_many(
        &q.t,
        Some(l),
        "attn::attention_compressed_paged_bf16",
        vec![],
        vec![q.id],
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

/// `kernels::attn::combine_attn_outputs_bf16`: merge two partial
/// attention results by their log-sum-exps.
///
/// Exact, not an approximation — the same algebra flashinfer's own
/// KV-split merge uses, which is why the fine and compressed halves can
/// be attended independently at all.
pub fn combine_attn_outputs(
    o1: &Val,
    lse1: &Val,
    o2: &Val,
    lse2: &Val,
    heads: u32,
    head_dim: u32,
) -> (Val, Val) {
    let outs = record_many_with_params(
        &o1.t,
        o1.layer,
        "attn::combine_attn_outputs_bf16",
        vec![],
        vec![heads, head_dim],
        vec![o1.id, lse1.id, o2.id, lse2.id],
        vec![
            (
                Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]),
                DType::BF16,
            ),
            (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32),
        ],
    );
    let mut it = outs.into_iter();
    let o = it.next().expect("the combine states two outputs");
    let lse = it.next().expect("the combine states two outputs");
    (o, lse)
}

/// `kernels::attn::lse_log2_to_ln`: rebase a log-sum-exp from log2 to
/// natural log.
///
/// FlashInfer publishes its LSE in log2; the combine above works in ln.
/// A unit conversion, and it is a launch, so the trace says so rather
/// than leaving a reader to wonder which base an LSE is in.
pub fn lse_log2_to_ln(lse: &Val, heads: u32) -> Val {
    record(
        &lse.t,
        lse.layer,
        "attn::lse_log2_to_ln",
        vec![],
        None,
        vec![lse.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32)),
    )
    .expect("the rebase produces its value")
}

// ── deepseek_v4: routing, activation, dequant ──────────────────

/// `kernels::moe::topk_sqrtsoftplus_bf16`: the router, scored by
/// `sqrt(softplus(·))`.
pub fn topk_sqrtsoftplus(logits: &Val, bias: &str, top_k: u32) -> (Val, Val) {
    let outs = record_many(
        &logits.t,
        logits.layer,
        "moe::topk_sqrtsoftplus_bf16",
        vec![bias.to_string()],
        vec![logits.id],
        vec![
            (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::I32),
            (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::F32),
        ],
    );
    let mut it = outs.into_iter();
    let idx = it.next().expect("the router states two outputs");
    let w = it.next().expect("the router states two outputs");
    (idx, w)
}

/// `kernels::moe::hash_route_lookup`: expert INDICES from a hash table
/// keyed by token id; weights still from the router logits.
///
/// A route that is a pure function of the token, not of its activations —
/// which is why it reads `token_ids` and a `tid2eid` table rather than
/// scoring anything. The weights are not hashed, so the logits GEMM above
/// it does not go away.
pub fn hash_route_lookup(token_ids: &Val, logits: &Val, table: &str, top_k: u32) -> (Val, Val) {
    let outs = record_many(
        &logits.t,
        logits.layer,
        "moe::hash_route_lookup",
        vec![table.to_string()],
        vec![token_ids.id, logits.id],
        vec![
            (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::I32),
            (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::F32),
        ],
    );
    let mut it = outs.into_iter();
    let idx = it.next().expect("the lookup states two outputs");
    let w = it.next().expect("the lookup states two outputs");
    (idx, w)
}

/// `kernels::mlp::swiglu_clamp_bf16` / `kernels::mlp::chunked_swiglu_clamp_bf16`:
/// swiglu with the gate clamped.
///
/// `packed` picks the chunked form, the binding choice [`swiglu`](crate::cuda::swiglu)
/// carries.
pub fn swiglu_clamp(x: &Val, intermediate: u32, packed: bool) -> Val {
    record(
        &x.t,
        x.layer,
        if packed {
            "mlp::chunked_swiglu_clamp_bf16"
        } else {
            "mlp::swiglu_clamp_bf16"
        },
        vec![],
        None,
        vec![x.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
            DType::BF16,
        )),
    )
    .expect("the activation produces its value")
}

/// `kernels::rope::rope_partial_last_bf16`: rope the LAST `rope_dim`
/// channels rather than the first.
///
/// A different statement from [`rope_partial_q_only`](crate::cuda::rope_partial_q_only), not a flag on
/// it: which end of the channel axis carries position is a property of
/// the checkpoint's layout.
pub fn rope_partial_last(x: &Val, heads: u32, head_dim: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "rope::rope_partial_last_bf16",
        vec![],
        None,
        vec![x.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]),
            DType::BF16,
        )),
    )
    .expect("the rope produces its value")
}

/// Which shape a quantized weight's SCALE has.
///
/// Three fp8 forms, and the difference is not a tuning knob: one scale
/// per tensor, one per output channel, or one per group of `group_size`
/// along K. It is a property of the checkpoint, so the declaration states
/// which — a driver that guessed would dequantize correctly on one
/// checkpoint and silently wrongly on another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Scale {
    /// One scale for the whole tensor.
    PerTensor,
    /// One per output channel.
    PerChannel,
    /// One per group along the reduction axis.
    PerGroup,
}

/// `kernels::quant::dequant_fp8_e4m3_to_bf16[_per_channel|_per_group]`:
/// widen an fp8 weight to bf16.
pub fn dequant_fp8_e4m3(
    t: &Trace,
    l: u32,
    weight: &str,
    rows: u32,
    cols: u32,
    scale: Fp8Scale,
) -> Val {
    record(
        t,
        Some(l),
        match scale {
            Fp8Scale::PerTensor => "quant::dequant_fp8_e4m3_to_bf16",
            Fp8Scale::PerChannel => "quant::dequant_fp8_e4m3_to_bf16_per_channel",
            Fp8Scale::PerGroup => "quant::dequant_fp8_e4m3_to_bf16_per_group",
        },
        vec![weight.to_string()],
        None,
        vec![],
        Some((Shape(vec![Dim::Const(rows), Dim::Const(cols)]), DType::BF16)),
    )
    .expect("the dequant produces its value")
}

/// `kernels::quant::dequant_mxfp4_to_bf16`: the same for MXFP4, whose
/// scale is an E8M0 exponent byte per block of 32.
pub fn dequant_mxfp4(t: &Trace, l: u32, weight: &str, rows: u32, cols: u32) -> Val {
    record(
        t,
        Some(l),
        "quant::dequant_mxfp4_to_bf16",
        vec![weight.to_string()],
        None,
        vec![],
        Some((Shape(vec![Dim::Const(rows), Dim::Const(cols)]), DType::BF16)),
    )
    .expect("the dequant produces its value")
}
