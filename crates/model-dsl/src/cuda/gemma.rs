//! GEMMA — 3n's AltUp, and gemma-4.

use super::*;

// ── gemma-3n: AltUp ────────────────────────────────────────────
//
// Gemma-3n carries K = `altup_num_inputs` PARALLEL residual streams
// instead of one. Each layer predicts the post-layer state of all K
// from a learned per-token combination of them, runs the real layer on
// one ACTIVE stream, and corrects the other K-1 from the difference.
// That is a residual stream with a rank, and it is why gemma-3n cannot
// be written as `llama_like` with different facts: `Dim::Tokens` rows
// are still rows, but the value under them is `[K, T, H]`.
//
// None of these carry a contract clause. Every one is row-shaped —
// token `t`'s output reads only token `t`'s inputs — so a peel may
// split them, no host plan is obligated, and there is no seam
// capability for one to refuse.

/// `kernels::norm::altup_predict_bf16`: the K streams' post-layer
/// state, predicted.
///
/// `predictions[k, t, h] = streams[k, t, h] + Σ_j coefs[t, j, k]·streams[j, t, h]`
///
/// `coefs` is fp32 and stays fp32: the K-summation accumulates
/// round-off that bf16 cannot absorb, which the kernel's own header
/// says is why it takes a float pointer.
pub fn altup_predict(streams: &Val, coefs: &Val, k: u32, hidden: u32) -> Val {
    record(
        &streams.t,
        streams.layer,
        "norm::altup_predict_bf16",
        vec![],
        None,
        vec![streams.id, coefs.id],
        Some((
            Shape(vec![Dim::Const(k), Dim::Tokens, Dim::Const(hidden)]),
            DType::BF16,
        )),
    )
    .expect("the prediction produces its value")
}

/// `kernels::norm::altup_correct_bf16`: the other K-1 streams,
/// corrected from what the active one actually computed.
///
/// `corrected[k] = predictions[k] + (activated - predictions[active])·(coefs[t,k] + 1)`
///
/// The `+1` is folded into the coefficient by
/// [`altup_unpack_correct_coefs`], not by this kernel.
pub fn altup_correct(
    predictions: &Val,
    activated: &Val,
    correction_coefs: &Val,
    k: u32,
    hidden: u32,
) -> Val {
    record(
        &predictions.t,
        predictions.layer,
        "norm::altup_correct_bf16",
        vec![],
        None,
        vec![predictions.id, activated.id, correction_coefs.id],
        Some((
            Shape(vec![Dim::Const(k), Dim::Tokens, Dim::Const(hidden)]),
            DType::BF16,
        )),
    )
    .expect("the correction produces its value")
}

/// `kernels::norm::altup_unpack_predict_coefs`: the router's bf16
/// `[T, K*K]` output as the fp32 `[T, K, K]` [`altup_predict`] reads.
///
/// Not a cast. It also applies the transpose HF spells
/// `.reshape(*, K, K).permute(0, 1, 3, 2)`, so the statement is a
/// distinct op rather than a dtype annotation on the matmul above it.
pub fn altup_unpack_predict_coefs(packed: &Val, k: u32) -> Val {
    record(
        &packed.t,
        packed.layer,
        "norm::altup_unpack_predict_coefs",
        vec![],
        None,
        vec![packed.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(k), Dim::Const(k)]),
            DType::F32,
        )),
    )
    .expect("the unpack produces its value")
}

/// `kernels::norm::altup_unpack_correct_coefs`: the same for the
/// correction's `[T, K]`, with HF's `+ 1.0` folded in.
pub fn altup_unpack_correct_coefs(packed: &Val, k: u32) -> Val {
    record(
        &packed.t,
        packed.layer,
        "norm::altup_unpack_correct_coefs",
        vec![],
        None,
        vec![packed.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::F32)),
    )
    .expect("the unpack produces its value")
}

/// `kernels::norm::mean_streams_bf16`: the K streams averaged into
/// one — `out[t, h] = (1/K) Σ_k streams[k, t, h]`.
///
/// How a rank-K residual stream is read by anything that expects one.
pub fn mean_streams(streams: &Val, hidden: u32) -> Val {
    record(
        &streams.t,
        streams.layer,
        "norm::mean_streams_bf16",
        vec![],
        None,
        vec![streams.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the mean produces its value")
}

/// `kernels::norm::compute_rms_bf16`: each row's RMS, as fp32.
///
/// A MEASUREMENT, not a normalization: it produces the target that
/// [`magnitude_rescale`] then holds another tensor to. The pair exists
/// because gemma-3n keeps a stream's magnitude fixed across a
/// projection rather than re-norming it.
pub fn compute_rms(x: &Val) -> Val {
    record(
        &x.t,
        x.layer,
        "norm::compute_rms_bf16",
        vec![],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens]), DType::F32)),
    )
    .expect("the measurement produces its value")
}

/// `kernels::norm::magnitude_rescale_bf16`: scale each row of `x` so
/// its RMS equals `target`'s.
///
/// In place in the kernel; a value here, because a trace records what
/// a statement produces and the reader should not have to know which
/// buffer it landed in.
pub fn magnitude_rescale(x: &Val, target_rms: &Val, hidden: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "norm::magnitude_rescale_bf16",
        vec![],
        None,
        vec![x.id, target_rms.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the rescale produces its value")
}

/// `kernels::norm::tanh_bf16` on AltUp's modality-router output.
///
/// HF computes this in fp32 and casts back; the kernel folds both, so
/// the trace states one op where the reference states three.
/// The result is the OPERAND's shape, read off the trace rather than
/// respelled: this kernel takes one pointer and rewrites it, so the
/// two are one buffer and a second spelling can only disagree.
///
/// It did. `[Tokens, width]` was the spelling, and gemma-3n's altup
/// coefficients run over a `Select`ed stream slice whose leading dim
/// is the STREAM count, not the fire's tokens — so the operand was
/// `[4, 4]` and the result claimed `[Tokens, 4]`. Nothing compared
/// them until the row said in place, at which point the arena put
/// one buffer where two shapes disagreed and
/// `an_alias_lands_inside_its_owner` refused it.
pub fn tanh(x: &Val) -> Val {
    let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
    record(
        &x.t,
        x.layer,
        "norm::tanh_bf16",
        vec![],
        None,
        vec![x.id],
        Some(out),
    )
    .expect("the activation produces its value")
}

/// `kernels::mlp::gaussian_topk_bf16`: gemma-3n's activation
/// sparsity — zero every element below `mean + std_multiplier·std` of
/// its own row.
///
/// A top-k by THRESHOLD rather than by count, which is what lets it be
/// row-shaped: no sort, no cross-row comparison, so a peel may split
/// it like any other elementwise statement.
pub fn gaussian_topk(x: &Val, width: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "mlp::gaussian_topk_bf16",
        vec![],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the sparsifier produces its value")
}

// ── gemma-4 ────────────────────────────────────────────────────
//
// The vocabulary the third family needs and the first two did not.
// Every one of these is a kernel the hand-written `gemma4.cpp`
// already fires; what is new is that a declaration can name it.

/// `kernels::launch_{chunked_,}geglu_tanh_bf16`: gemma-4's MLP
/// activation. `gelu_pytorch_tanh` on the gate, not SiLU — a
/// different function, so a different kernel, and NOT a variant of
/// [`swiglu`].
///
/// `packed` splits the same way swiglu's does: a bound gate‖up bank
/// lands one buffer and takes the chunked form. gemma-4 states the
/// binding as a fact for the same reason llama_like does.
pub fn geglu_tanh(x: &Val, intermediate: u32, packed: bool) -> Val {
    record(
        &x.t,
        x.layer,
        if packed {
            "mlp::chunked_geglu_tanh_bf16"
        } else {
            "mlp::geglu_tanh_bf16"
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

/// `kernels::mlp::geglu_tanh_bf16` in its PAIR form: the gate and
/// the up operand are two buffers, not one packed bank.
///
/// gemma-4's PLE epilogue needs it even on a checkpoint that bound a
/// packed MLP bank, because the "up" operand there is the layer's
/// slice of the per-layer table — a buffer that was never going to
/// be adjacent to the gate. Same kernel as [`geglu_tanh`]'s unpacked
/// arm; a different statement because the OPERANDS differ, which is
/// what a reader needs to see.
pub fn geglu_tanh_pair(gate: &Val, up: &Val, width: u32) -> Val {
    record(
        &gate.t,
        gate.layer,
        "mlp::geglu_tanh_bf16",
        vec![],
        None,
        vec![gate.id, up.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the activation produces its value")
}

/// `kernels::rope::rope_partial_bf16` rotating Q ALONE.
///
/// A KV-shared layer's K was rotated at its SOURCE layer, where it
/// was written to the cache, so rotating it again here would be
/// wrong twice over — the value is not even in this layer's
/// registers. The driver says that with `num_kv_heads = 0`; the
/// trace says it by the statement having ONE operand.
///
/// The semantic [`super::rope`] cannot: its shape is a (q, k) pair,
/// and a pair with an empty slot is a different statement, not a
/// degenerate one.
pub fn rope_partial_q_only(q: &Val, rotary_dim: u32) -> Val {
    let out = (q.t.inner.borrow().value_shape(q.id), DType::BF16);
    record_with_params(
        &q.t,
        q.layer,
        "rope::rope_partial_bf16",
        vec![],
        None,
        vec![rotary_dim],
        vec![q.id],
        Some(out),
    )
    .expect("the rotation produces its value")
}

/// [`qk_rmsnorm_rope_rounded`] with K absent — the SHARED sliding
/// layer's form.
///
/// Same symbol, and that is the point: the driver reaches this by
/// passing `k_norm = nullptr` and `num_kv_heads = 0` to the very
/// same launcher, so a declaration that spelled it as a rope plus a
/// separate norm would be naming a pair of kernels the pass never
/// fires. One operand, one weight, one launch.
pub fn qk_rmsnorm_rope_rounded_q_only(q: &Val, q_norm: &NormW) -> Val {
    let out = (q.t.inner.borrow().value_shape(q.id), DType::BF16);
    record(
        &q.t,
        q_norm.layer,
        "rope::qk_rmsnorm_rope_bf16_rounded",
        vec![q_norm.name.clone()],
        None,
        vec![q.id],
        Some(out),
    )
    .expect("the fused pair produces q")
}

/// `kernels::norm::rmsnorm_no_scale_bf16`: `v / rms(v)` per head,
/// with NO learnable weight — gemma-4's V-norm.
///
/// Weightless, so it takes no [`NormW`]: a norm handle contributes a
/// name and a layer, and this kernel reads neither. That is also why
/// it cannot be the semantic `Rmsnorm` with a variant — there is no
/// gamma for a variant to describe.
pub fn rmsnorm_no_scale(x: &Val) -> Val {
    let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
    record(
        &x.t,
        x.layer,
        "norm::rmsnorm_no_scale_bf16",
        vec![],
        None,
        vec![x.id],
        Some(out),
    )
    .expect("the norm produces its value")
}

/// `kernels::norm::rmsnorm_residual_add_scale_rmsnorm_bf16`: FOUR
/// statements in one launch — norm `x`, add it to the stream, scale
/// the result, then norm THAT with the next weight.
///
/// The last of those four is the next block's input norm, which is
/// why gemma-4's per-layer body appears to be missing one: the fused
/// kernel already produced it. A declaration that named the four
/// separately would be naming a shape the driver does not run.
///
/// Returns `(hidden, norm_out)` — the landed residual and the norm
/// the next block consumes.
pub fn norm_residual_scale_norm(
    x: &Val,
    y: &Val,
    w: &NormW,
    next: &NormW,
    hidden: u32,
) -> (Val, Val) {
    let shape = (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16);
    let ids = x.t.with(w.layer, |b| {
        b.launch(
            "norm::rmsnorm_residual_add_scale_rmsnorm_bf16",
            vec![w.name.clone(), next.name.clone()],
            None,
            // The STREAM is an operand. The kernel reads it and
            // accumulates into it, so a statement that named only `x`
            // left SSA with no edge from the old stream to the new
            // one -- and an executor binding buffers from the edges
            // then handed the launch a fresh buffer to land on.
            vec![x.id, y.id],
            vec![shape.clone(), shape],
        )
    });
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

/// `kernels::norm::rmsnorm_residual_add_bf16`: the two-statement
/// form — norm, then land on the stream. gemma-4's
/// post-feedforward norm, where no next-block norm follows to fuse.
pub fn norm_residual_add(x: &Val, y: &Val, w: &NormW, hidden: u32) -> Val {
    record(
        &x.t,
        w.layer,
        "norm::rmsnorm_residual_add_bf16",
        vec![w.name.clone()],
        None,
        // `y` is the residual stream this lands on: read, accumulated
        // into, and returned. Naming it is what gives the new stream
        // an SSA edge to the old one.
        vec![x.id, y.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the fused norm+residual produces its value")
}

/// `kernels::norm::scalar_mul_bf16`: multiply by a load-time
/// constant, NAMED.
///
/// gemma-4 fires this four times per fire with four different
/// constants — `sqrt(hidden)` on the embedding, then
/// `sqrt(ple_dim)`, `1/sqrt(hidden)` and `1/sqrt(2)` through the PLE
/// prologue. All four are derived from dims, so none is an operand;
/// but a statement that did not say WHICH would leave an executor
/// with four identical launches and no way to tell them apart. This
/// was written without the name first, and writing the arm is what
/// found it.
///
/// The name rides the weight slot because that is what a name slot
/// is: `scale.` marks it as a constant rather than a tensor, so a
/// binder never looks for it.
/// The NUMBER rides the param channel, in the bits an untyped `u32`
/// slot already has room for. The name stays because a reader wants
/// it; the driver used to need it, and that is the difference. It
/// held a name-to-arithmetic table — `sqrt(hidden)`, `sqrt(ple_dim)`,
/// `1/sqrt(hidden)`, `1/sqrt(2)` — recomputing on the device side
/// what the host had already derived from its own dims, and an
/// unrecognised name was a runtime refusal rather than a number.
/// `by` is OPTIONAL, and a `None` is a family saying its facts do
/// not carry the number yet — gemma-3n's altup and laurel scales and
/// gemma-2's query scale are per-layer constants nothing on the host
/// side has derived, and inventing one here would be worse than the
/// name it replaces. A statement without the param falls through the
/// generated branch's arity guard to whatever arm knows better,
/// which for those two families is the hand-written pass.
pub fn scalar_mul(x: &Val, scale: &str, by: Option<f32>) -> Val {
    let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
    record_with_params(
        &x.t,
        x.layer,
        "norm::scalar_mul_bf16",
        vec![format!("scale.{scale}")],
        None,
        by.map(f32::to_bits).into_iter().collect(),
        vec![x.id],
        Some(out),
    )
    .expect("the scale produces its value")
}

/// `kernels::attn::logit_softcap_bf16`: `cap * tanh(x / cap)` over
/// the logits. A load-time fact decides whether it runs at all
/// (`final_logit_softcapping`), so its presence is a trace-time
/// match, not a branch.
pub fn logit_softcap(x: &Val, vocab: u32) -> Val {
    record(
        &x.t,
        None,
        "attn::logit_softcap_bf16",
        vec![],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Requests, Dim::Const(vocab)]), DType::BF16)),
    )
    .expect("the softcap produces its value")
}

/// `kernels::attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16`:
/// gemma-4's decode post — split the packed projection, norm q and
/// k, rope them, norm v, and write k/v straight to the pages. One
/// launch, six statements, and the only value that survives it is q.
///
/// Its eligibility is a per-FIRE question in the hand-written pass
/// (`hooks == nullptr && !partial && !dump && native bf16 && a
/// decode path`), and the terms split cleanly: `partial` and the
/// cache format are load-time, hooks and the fire class are the
/// declaration's own class/guard vocabulary. So a class trace states
/// it or does not, and nothing reads a workspace to decide.
///
/// Writes through the KV pages, so it carries the layer's cache
/// state the way every write-side statement here does.
pub fn qkv_packed_post(
    packed: &Val,
    q_norm: &NormW,
    k_norm: &NormW,
    kv: &Kv,
    q_width: u32,
) -> Val {
    record(
        &packed.t,
        q_norm.layer,
        "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
        vec![q_norm.name.clone(), k_norm.name.clone()],
        kv_state(kv),
        vec![packed.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
    )
    .expect("the fused post produces q")
}

/// `kernels::rope::qk_rmsnorm_rope_bf16_rounded`: the per-head q/k
/// norm + rope pair, in the ROUNDED form.
///
/// gemma-4 rounds where qwen3_5 does not, and bf16 rounding is not
/// an implementation detail between two kernels that compute the
/// same function — it is which numbers come out. So the symbol is
/// the statement, and a family states the one its hand-written pass
/// fires. In place on q and k; SSA-wise two fresh values.
pub fn qk_rmsnorm_rope_rounded(q: &Val, k: &Val, q_norm: &NormW, k_norm: &NormW) -> (Val, Val) {
    let shapes = {
        let b = q.t.inner.borrow();
        vec![
            (b.value_shape(q.id), DType::BF16),
            (b.value_shape(k.id), DType::BF16),
        ]
    };
    let ids = q.t.with(q_norm.layer, |b| {
        b.launch(
            "rope::qk_rmsnorm_rope_bf16_rounded",
            vec![q_norm.name.clone(), k_norm.name.clone()],
            None,
            vec![q.id, k.id],
            shapes,
        )
    });
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q_norm.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

/// `kernels::layout::transpose_bf16_nld_to_lnd`: relay the PLE table
/// from `[N, L, D]` to `[L, N, D]` so each layer reads a CONTIGUOUS
/// slice.
///
/// The whole point of the statement is addressing, not arithmetic —
/// it replaces a per-layer slice-pack kernel with one relay per
/// fire, which is the driver's own comment at the call site. The
/// output's leading dim is the LAYER count, a load-time constant, so
/// the shape is `[Const(layers), Tokens, Const(dim)]`.
pub fn transpose_nld_to_lnd(x: &Val, layers: u32, dim: u32) -> Val {
    record(
        &x.t,
        None,
        "layout::transpose_bf16_nld_to_lnd",
        vec![],
        None,
        vec![x.id],
        Some((
            Shape(vec![Dim::Const(layers), Dim::Tokens, Dim::Const(dim)]),
            DType::BF16,
        )),
    )
    .expect("the relay produces its value")
}

/// `kernels::moe::topk_softmax_bf16`: the router's top-k + softmax +
/// renormalize, one launch, two results — expert indices
/// (`[Tokens, k]` i32, the `dyn` value every expert-indexed statement
/// consumes) and routing weights (`[Tokens, k]` f32).
///
/// The first statement of the MoE branch's CUDA text. The SEMANTIC
/// [`super::topk`] stays opaque; this one names the kernel, which is
/// what `lower()` needs before an expert-routed body can be a list of
/// rectangles rather than residue.
pub fn topk(logits: &Val, k: u32) -> (Val, Val) {
    let ids = logits.t.with(logits.layer, |b| {
        b.launch(
            "moe::topk_softmax_bf16",
            vec![],
            None,
            vec![logits.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::I32),
                (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::F32),
            ],
        )
    });
    let mk = |id| Val {
        t: logits.t.clone(),
        id,
        layer: logits.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

/// `kernels::moe::moe_gate_up_decode_gemv_bf16` /
/// `..._moe_down_decode_gemv_bf16`: the routed projections of the
/// decode GEMV leg, one launch each over the fire's `N * k` routes.
///
/// The expert axis is INSIDE the value, not outside it: one launch
/// reads `experts` and strides the stacked bank itself, so the
/// declaration stays a rectangle even though the arithmetic is
/// per-token-per-expert. That is why this leg is the one the CUDA
/// text can state — see [`matmul_per_token`](crate::matmul_per_token)'s other legs,
/// which reach the same numbers by *host* routing (the general path)
/// or by an aligned padding that gives the intermediate an extent no
/// [`Dim`] spells (the grouped-GEMM path).
///
/// Both projections carry the routed extent as a third dim: `k` is a
/// load-time constant, so `[Tokens, k, width]` is exactly the
/// `N * k`-row buffer the kernel writes, said without inventing a
/// row space.
pub fn moe_gate_up_gemv(x: &Val, w: &MatW, experts: &Val, top_k: u32) -> Val {
    moe_routed_gemv("moe::moe_gate_up_decode_gemv_bf16", x, w, experts, top_k)
}

pub fn moe_down_gemv(x: &Val, w: &MatW, experts: &Val, top_k: u32) -> Val {
    moe_routed_gemv("moe::moe_down_decode_gemv_bf16", x, w, experts, top_k)
}

fn moe_routed_gemv(kernel: &str, x: &Val, w: &MatW, experts: &Val, top_k: u32) -> Val {
    record(
        &x.t,
        w.layer,
        kernel,
        vec![w.name.clone()],
        None,
        vec![experts.id, x.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(top_k), Dim::Const(w.width)]),
            DType::BF16,
        )),
    )
    .expect("a routed projection produces its value")
}

/// `kernels::moe::flashinfer_cutlass_moe_bf16`: the whole routed block —
/// permute, both grouped GEMMs, the activation, and the weighted
/// finalize — as ONE call.
///
/// This is the leg the decode path actually takes, and it is stated
/// first because it is the only one that is a single rectangle. Its
/// `bool` return reads like a runtime fallthrough, but every false
/// it can produce is decided before the fire: null operands (a
/// binding question) and `workspace_bytes < needed`, where `needed`
/// is a pure function of the static dims and `num_rows`, and the
/// caller has already required `N <= cutlass_max_rows` — the row
/// count the workspace was sized for. So the leg is a FACT plus a
/// row bound, not a gamble, and fires above the bound decline rather
/// than the declaration guessing.
///
/// Consumes the router's two outputs and both expert banks; produces
/// the combined `[Tokens, hidden]` in one value, which is why the
/// text that names it has no separate WeightedSum.
pub fn moe_fused_cutlass(
    x: &Val,
    experts: &Val,
    weights: &Val,
    gate_up: &MatW,
    down: &MatW,
    hidden: u32,
) -> Val {
    record(
        &x.t,
        gate_up.layer,
        "moe::flashinfer_cutlass_moe_bf16",
        vec![gate_up.name.clone(), down.name.clone()],
        None,
        vec![x.id, experts.id, weights.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the fused MoE produces its value")
}

/// `kernels::norm::residual_add_bf16`: the explicit stream add, for
/// the legs whose producer wrote to scratch instead of folding.
pub fn residual_add(x: &Val, residual: &Val, hidden: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "norm::residual_add_bf16",
        vec![],
        None,
        vec![x.id, residual.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the residual add produces its value")
}

/// Row RMSNorm, STATING which fold it runs.
///
/// Gemma folds `(1 + w)` instead of `w` — different arithmetic, so a
/// different kernel — and the fold is a property of the WEIGHT,
/// which is why [`NormW`] carries it and why the caller passes no
/// variant.
///
/// The semantic [`super::rmsnorm`] carries the variant as a param
/// instead, and four drivers read it and pick; three had hard-coded
/// their own deployment's answer. A `*.cuda.*` text calls this one
/// and nothing downstream chooses.
///
/// PER-HEAD is not here yet: its row count is the operand's width
/// over the head dim, and `head_dim` has nowhere to ride on a
/// `Launch`. It moves when it states a kernel that takes it.
pub fn rmsnorm(x: &Val, w: &NormW) -> Val {
    let id = x.t.with(w.layer, |b| match w.per_head {
        // PER-HEAD falls through to the semantic kind, and the call
        // site does not have to know which it got: the handle
        // decides, and the same site is per-head on qwen3 and
        // row-wise on olmo2.
        Some(head_dim) => b.rmsnorm_per_head(x.id, &w.name, head_dim, w.variant),
        None => {
            let symbol = match w.variant {
                NormVariant::Gemma => "norm::rmsnorm_gemma_bf16",
                _ => "norm::rmsnorm_bf16",
            };
            let shape = b.value_shape(x.id);
            b.launch(
                symbol,
                vec![w.name.clone()],
                None,
                vec![x.id],
                vec![(shape, DType::BF16)],
            )[0]
        }
    });
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}
