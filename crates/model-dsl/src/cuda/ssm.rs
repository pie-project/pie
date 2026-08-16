//! THE RECURRENT PATHS — nemotron_h's mamba, Kimi Delta Attention,
//! and kimi_k3's SiTU with the fp32 widenings its recurrence needs.
//! Three generations, one shape of problem.

use super::*;

// ── nemotron_h: mamba ──────────────────────────────────────────
//
// The other linear-attention shape, and it is not GDN's or KDA's. Mamba
// carries a `[head_dim, state_size]` slab per head and advances it with a
// scalar `dA` derived from a per-token `dt` -- a selective scan, not a
// delta rule. The state is a different SHAPE, which is why nothing above
// stands in for it and why the todo lists it as its own missing algebra.

/// `kernels::ssm::nemotron_mamba_split_bf16`: split the fused input
/// projection into `(gate, conv_in, dt)`.
pub fn nemotron_mamba_split(
    projected: &Val,
    intermediate: u32,
    conv_dim: u32,
    heads: u32,
) -> (Val, Val, Val) {
    let outs = record_many(
        &projected.t,
        projected.layer,
        "ssm::nemotron_mamba_split_bf16",
        vec![],
        vec![projected.id],
        vec![
            (
                Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                DType::BF16,
            ),
            (Shape(vec![Dim::Tokens, Dim::Const(conv_dim)]), DType::BF16),
            (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::BF16),
        ],
    );
    let mut it = outs.into_iter();
    let gate = it.next().expect("the split states three outputs");
    let conv_in = it.next().expect("the split states three outputs");
    let dt = it.next().expect("the split states three outputs");
    (gate, conv_in, dt)
}

/// `kernels::ssm::nemotron_prepare_mamba_params`: widen `A_log`, `D`
/// and `dt_bias` to fp32, storing `A = -exp(A_log)`.
///
/// Per HEAD, with no token extent at all — it transforms weights, not
/// activations. Stated because it is a launch the fire performs, and a
/// reader following where `A` comes from should find it on the tape.
pub fn nemotron_prepare_mamba_params(
    t: &Trace,
    l: u32,
    a_log: &str,
    d: &str,
    dt_bias: &str,
    heads: u32,
) -> (Val, Val, Val) {
    let outs = record_many(
        t,
        Some(l),
        "ssm::nemotron_prepare_mamba_params",
        vec![a_log.to_string(), d.to_string(), dt_bias.to_string()],
        vec![],
        vec![
            (Shape(vec![Dim::Const(heads)]), DType::F32),
            (Shape(vec![Dim::Const(heads)]), DType::F32),
            (Shape(vec![Dim::Const(heads)]), DType::F32),
        ],
    );
    let mut it = outs.into_iter();
    let a = it.next().expect("the prepare states three outputs");
    let d_f32 = it.next().expect("the prepare states three outputs");
    let bias = it.next().expect("the prepare states three outputs");
    (a, d_f32, bias)
}

/// `kernels::ssm::nemotron_prepare_mamba_dt_da`: the per-token step
/// size and its decay, `(dt, dA)`.
pub fn nemotron_prepare_mamba_dt_da(dt_raw: &Val, a: &Val, heads: u32) -> (Val, Val) {
    let outs = record_many(
        &dt_raw.t,
        dt_raw.layer,
        "ssm::nemotron_prepare_mamba_dt_da",
        vec![],
        vec![dt_raw.id, a.id],
        vec![
            (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32),
            (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32),
        ],
    );
    let mut it = outs.into_iter();
    let dt = it.next().expect("the prepare states two outputs");
    let da = it.next().expect("the prepare states two outputs");
    (dt, da)
}

/// `kernels::ssm::nemotron_mamba_ssm_batched_bf16`: the selective scan.
///
/// `whole` for both reasons the table collects: it addresses through
/// `slot_ids` and `qo_indptr`, AND the scan carries state from token to
/// token, so a row window would resume from the wrong slab.
pub fn nemotron_mamba_ssm(conv_out: &Val, dt: &Val, l: u32, intermediate: u32) -> Val {
    record(
        &conv_out.t,
        Some(l),
        "ssm::nemotron_mamba_ssm_batched_bf16",
        vec![],
        Some(StateRef {
            store: StateStore::RecurrentState,
            layer: l,
        }),
        vec![conv_out.id, dt.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
            DType::BF16,
        )),
    )
    .expect("the scan produces its value")
}

/// `kernels::ssm::zamba_rmsnorm_gated_bf16`: the grouped, gated output
/// norm mamba's block ends with.
pub fn zamba_rmsnorm_gated(x: &Val, gate: &Val, weight: &str, hidden: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "ssm::zamba_rmsnorm_gated_bf16",
        vec![weight.to_string()],
        None,
        vec![x.id, gate.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the norm produces its value")
}

/// `kernels::mlp::relu2_bf16`: `relu(x)²`, nemotron_h's MLP activation.
pub fn relu2(x: &Val, width: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "mlp::relu2_bf16",
        vec![],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the activation produces its value")
}


// ── KDA: Kimi Delta Attention ──────────────────────────────────
//
// The linear-attention half of kimi_k3. Same gated delta rule qwen3_5
// runs, with one difference that changes every kernel: the decay is per
// KEY CHANNEL, not per head. Qwen3.5 multiplies the whole `[K_d, V_d]`
// state slab by one scalar `exp(g_h)`; KDA multiplies column `k` by
// `exp(gate[h, k])`. That is the "delta" in the name -- a fine-grained
// forget gate -- and it is why these are their own kernels rather than
// GDN's with a broadcast.
//
// All the arithmetic is fp32; bf16 operands are widened first, which is
// why the dtype casts below are statements rather than annotations.

/// `kernels::ssm::kda_gate_beta_bf16`: the forget gate and the write
/// strength, from their raw projections.
///
/// Returns `(gate, beta)`, both fp32. `A_log` is per head and `dt_bias`
/// per head-channel, so both are WEIGHTS the launch reads.
pub fn kda_gate_beta(
    raw_g: &Val,
    raw_beta: &Val,
    a_log: &str,
    dt_bias: &str,
    heads: u32,
    head_dim: u32,
) -> (Val, Val) {
    let outs = record_many_with_params(
        &raw_g.t,
        raw_g.layer,
        "ssm::kda_gate_beta_bf16",
        vec![a_log.to_string(), dt_bias.to_string()],
        // `head_dim`: the gate result is `[Tokens, heads * head_dim]`
        // and only the product is a shape, so the row reads `d` here.
        vec![head_dim],
        vec![raw_g.id, raw_beta.id],
        vec![
            (
                Shape(vec![Dim::Tokens, Dim::Const(heads * head_dim)]),
                DType::F32,
            ),
            (Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32),
        ],
    );
    let mut it = outs.into_iter();
    let gate = it.next().expect("the gate states two outputs");
    let beta = it.next().expect("the gate states two outputs");
    (gate, beta)
}

/// `kernels::ssm::kda_recurrent_step_batched`: one decode token per
/// request, advancing each request's state slot.
///
/// `whole`: `slot_ids` is indexed `0..R` against the fire's request
/// order, so a row window would advance the wrong slots.
pub fn kda_recurrent_step(
    q: &Val,
    k: &Val,
    v: &Val,
    gate: &Val,
    beta: &Val,
    l: u32,
    heads: u32,
    head_dim: u32,
) -> Val {
    record(
        &q.t,
        Some(l),
        "ssm::kda_recurrent_step_batched",
        vec![],
        Some(StateRef {
            store: StateStore::RecurrentState,
            layer: l,
        }),
        vec![q.id, k.id, v.id, gate.id, beta.id],
        Some((
            Shape(vec![Dim::Requests, Dim::Const(heads), Dim::Const(head_dim)]),
            DType::F32,
        )),
    )
    .expect("the recurrence produces its value")
}

/// `kernels::ssm::kda_prefill_batched`: the same recurrence over a
/// prefill window, one block per (request, head).
///
/// `whole` twice over: it walks windows out of `qo_indptr`, AND the
/// recurrence has a strict per-token state dependency -- the block walks
/// its window one token at a time because token `t`'s state is token
/// `t-1`'s output. A row window would start the scan from the wrong
/// state, which is a different answer rather than a misaddressed one.
pub fn kda_prefill(
    q: &Val,
    k: &Val,
    v: &Val,
    gate: &Val,
    beta: &Val,
    l: u32,
    heads: u32,
    head_dim: u32,
) -> Val {
    record(
        &q.t,
        Some(l),
        "ssm::kda_prefill_batched",
        vec![],
        Some(StateRef {
            store: StateStore::RecurrentState,
            layer: l,
        }),
        vec![q.id, k.id, v.id, gate.id, beta.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]),
            DType::F32,
        )),
    )
    .expect("the recurrence produces its value")
}

/// `kernels::ssm::kda_o_norm_gated_bf16`: the output norm and gate.
/// `heads` and `head_dim` ride the PARAM channel: the result is
/// `[Tokens, heads * head_dim]` and only their product is a shape.
pub fn kda_o_norm_gated(
    x: &Val,
    gate: &Val,
    weight: &str,
    width: u32,
    heads: u32,
    head_dim: u32,
) -> Val {
    record_with_params(
        &x.t,
        x.layer,
        "ssm::kda_o_norm_gated_bf16",
        vec![weight.to_string()],
        None,
        vec![heads, head_dim],
        vec![x.id, gate.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the norm produces its value")
}


// ── kimi_k3: SiTU, and the fp32 widenings its recurrence needs ─

/// `kernels::mlp::situ_bf16` / `kernels::mlp::chunked_situ_bf16`: Moonshot's
/// `SituAndMul`.
///
/// Not a swiglu variant. The tanh saturates far enough out (beta 4,
/// linear_beta 25 on K3) that a bf16 intermediate loses the distinction
/// the gate exists to make, so the kernel evaluates in fp32 and narrows
/// once. `packed` picks the chunked form, the same binding choice
/// [`swiglu`](crate::cuda::swiglu) carries.
pub fn situ(x: &Val, intermediate: u32, packed: bool) -> Val {
    record(
        &x.t,
        x.layer,
        if packed {
            "mlp::chunked_situ_bf16"
        } else {
            "mlp::situ_bf16"
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

/// `kernels::ssm::l2norm_scale_bf16_to_fp32`: l2-norm each row and
/// scale, widening to fp32.
///
/// `y[r,c] = x[r,c] / sqrt(Σ_c x[r,c]² + eps) · scale`. The recurrence
/// above wants q pre-scaled by `K_d^(-1/2)`; this is where that happens.
pub fn l2norm_scale_to_f32(x: &Val, width: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "ssm::l2norm_scale_bf16_to_fp32",
        vec![],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::F32)),
    )
    .expect("the norm produces its value")
}

/// `kernels::ssm::bf16_to_fp32`: widen.
///
/// A statement rather than a dtype annotation because it is a launch, and
/// the trace records launches. KDA's arithmetic is fp32 throughout, so an
/// operand that lives in bf16 in the workspace crosses here.
pub fn bf16_to_f32(x: &Val, width: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "ssm::bf16_to_fp32",
        vec![],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::F32)),
    )
    .expect("the cast produces its value")
}

/// `kernels::ssm::fp32_to_bf16`: narrow, on the way back out.
pub fn f32_to_bf16(x: &Val, width: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "ssm::fp32_to_bf16",
        vec![],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the cast produces its value")
}

/// `kernels::attn::attn_res_blend_bf16`: blend the prefix output with
/// the open blocks', weighted by a learned score.
///
/// With no open blocks the softmax is over a single row and the output IS
/// the prefix, which is what the tail blend of a model with no open
/// blocks means -- the kernel's own header says so, and it is why this
/// needs no guard around it.
pub fn attn_res_blend(
    prefix: &Val,
    blocks: &Val,
    norm_weight: &str,
    proj_weight: &str,
    width: u32,
) -> Val {
    record(
        &prefix.t,
        prefix.layer,
        "attn::attn_res_blend_bf16",
        vec![norm_weight.to_string(), proj_weight.to_string()],
        None,
        vec![prefix.id, blocks.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the blend produces its value")
}
