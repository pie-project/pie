//! Recurrent CUDA paths: mamba, Kimi Delta Attention, and SiTU support.

use super::*;


builder! {
    /// `kernels::ssm::nemotron_mamba_split_bf16`.
    pub fn nemotron_mamba_split(
        projected: &Val,
        intermediate: u32,
        conv_dim: u32,
        heads: u32,
    ) -> (Val, Val, Val) {
        symbol: "ssm::nemotron_mamba_split_bf16",
        on: projected,
        inputs: [projected],
        outs: [
            [Dim::Tokens, Dim::Const(intermediate)] as BF16,
            [Dim::Tokens, Dim::Const(conv_dim)] as BF16,
            [Dim::Tokens, Dim::Const(heads)] as BF16,
        ],
        made: "the split states three outputs",
    }
}

/// `kernels::ssm::nemotron_prepare_mamba_params`.
pub fn nemotron_prepare_mamba_params(
    t: &Trace,
    l: u32,
    a_log: &str,
    d: &str,
    dt_bias: &str,
    heads: u32,
) -> (Val, Val, Val) {
    let outs = record_many_with_params(
        t,
        Some(l),
        "ssm::nemotron_prepare_mamba_params",
        vec![a_log.to_string(), d.to_string(), dt_bias.to_string()],
        // The head count, which the shapes below already state. It reached the
        // routine as `keys::GdnVHeads`; the statement carries it now.
        vec![heads],
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

builder! {
    /// `kernels::ssm::nemotron_prepare_mamba_dt_da`.
    pub fn nemotron_prepare_mamba_dt_da(
        dt_raw: &Val,
        a: &Val,
        dt_bias: &Val,
        heads: u32,
    ) -> (Val, Val) {
        symbol: "ssm::nemotron_prepare_mamba_dt_da",
        on: dt_raw,
        inputs: [dt_raw, a, dt_bias],
        outs: [
            [Dim::Tokens, Dim::Const(heads)] as F32,
            [Dim::Tokens, Dim::Const(heads)] as F32,
        ],
        made: "the prepare states two outputs",
    }
}


/// `kernels::ssm::nemotron_mamba_ssm_batched_bf16`: the selective scan.
///
/// `[num_heads, head_dim, state_size, n_groups, conv_dim, r, rows]`, in the
/// order the routine's marks declare: five checkpoint numbers, then the
/// request and row counts the fire decides (spliced extents). The recurrent
/// view and the qo CSR are the operands after `da`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn nemotron_mamba_ssm(
    conv_out: &Val,
    dt: &Val,
    dt_raw: &Val,
    a: &Val,
    d: &Val,
    dt_bias: &Val,
    da: &Val,
    l: u32,
    intermediate: u32,
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    conv_dim: u32,
) -> Val {
    let rsv = rt_object(&conv_out.t, "recurrent_state", Some(l));
    let qo_indptr = rt_requests(&conv_out.t, "qo_indptr");
    record_with_extents(
        &conv_out.t,
        Some(l),
        "ssm::nemotron_mamba_ssm_batched_bf16",
        vec![],
        Some(StateRef {
            store: StateStore::RecurrentState,
            layer: l,
        }),
        vec![num_heads, head_dim, state_size, n_groups, conv_dim],
        vec![],
        vec![
            conv_out.id,
            dt.id,
            dt_raw.id,
            a.id,
            d.id,
            dt_bias.id,
            da.id,
            rsv,
            qo_indptr,
        ],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
            DType::BF16,
        )),
    )
    .expect("the scan produces its value")
}

builder! {
    /// `kernels::ssm::zamba_rmsnorm_gated`.
    ///
    /// `[n_groups, eps]` is the run: a grouped RMS norm reduces within a
    /// group, so the count is the geometry and not a fact of the fire, and
    /// the epsilon is the row's. Stating both is what `check_plan`'s params
    /// rule counts.
    pub fn zamba_rmsnorm_gated(
        x: &Val,
        gate: &Val,
        weight: &str,
        hidden: u32,
        n_groups: u32,
        eps: f32,
    ) -> Val {
        symbol: "ssm::zamba_rmsnorm_gated",
        on: x,
        weights: [weight],
        params: [n_groups, eps.to_bits()],
        inputs: [x, gate],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the norm produces its value",
    }


    /// `kernels::mlp::relu2`: `relu(x)²`, nemotron_h's MLP activation.
    pub fn relu2(x: &Val, width: u32) -> Val {
        symbol: "mlp::relu2",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the activation produces its value",
    }
}


builder! {
    /// `kernels::ssm::kda_gate_beta`.
    pub fn kda_gate_beta(
        raw_g: &Val,
        raw_beta: &Val,
        a_log: &str,
        dt_bias: &str,
        heads: u32,
        head_dim: u32,
    ) -> (Val, Val) {
        symbol: "ssm::kda_gate_beta",
        on: raw_g,
        weights: [a_log, dt_bias],
        // params[0] = head_dim; the shape only carries heads * head_dim.
        params: [head_dim],
        inputs: [raw_g, raw_beta],
        outs: [
            [Dim::Tokens, Dim::Const(heads * head_dim)] as F32,
            [Dim::Tokens, Dim::Const(heads)] as F32,
        ],
        made: "the gate states two outputs",
    }
}


/// `kernels::ssm::kda_recurrent_step_batched`.
/// `params = [heads, head_dim]` — the request count is the `[Requests, H, D]`
/// result's own row count. The recurrent view is the operand after `beta`.
#[allow(clippy::too_many_arguments)]
#[must_use]
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
    let rsv = rt_object(&q.t, "recurrent_state", Some(l));
    record_with_extents(
        &q.t,
        Some(l),
        "ssm::kda_recurrent_step_batched",
        vec![],
        Some(StateRef {
            store: StateStore::RecurrentState,
            layer: l,
        }),
        vec![heads, head_dim],
        vec![],
        vec![q.id, k.id, v.id, gate.id, beta.id, rsv],
        Some((
            Shape(vec![Dim::Requests, Dim::Const(heads), Dim::Const(head_dim)]),
            DType::F32,
        )),
    )
    .expect("the recurrence produces its value")
}

/// `kernels::ssm::kda_prefill_batched`: [`kda_recurrent_step`]'s run and
/// operands, plus the qo CSR the prefill walks.
#[allow(clippy::too_many_arguments)]
#[must_use]
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
    let rsv = rt_object(&q.t, "recurrent_state", Some(l));
    let qo_indptr = rt_requests(&q.t, "qo_indptr");
    record_with_extents(
        &q.t,
        Some(l),
        "ssm::kda_prefill_batched",
        vec![],
        Some(StateRef {
            store: StateStore::RecurrentState,
            layer: l,
        }),
        vec![heads, head_dim],
        vec![],
        vec![q.id, k.id, v.id, gate.id, beta.id, rsv, qo_indptr],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]),
            DType::F32,
        )),
    )
    .expect("the recurrence produces its value")
}

builder! {
    /// `kernels::ssm::kda_o_norm_gated`: the output norm and gate.
    pub fn kda_o_norm_gated(
        x: &Val,
        gate: &Val,
        weight: &str,
        width: u32,
        heads: u32,
        head_dim: u32,
        eps: f32,
    ) -> Val {
        symbol: "ssm::kda_o_norm_gated",
        on: x,
        weights: [weight],
        // params[0] = heads, params[1] = head_dim, params[2] = eps read as
        // its bits; grid and state stride use this order. The epsilon was
        // `Env<keys::RmsEps>` and is the ROW's number, so the statement
        // carries it -- an unstated one is a zero, and a zero epsilon makes
        // every all-zero row divide by nothing.
        params: [heads, head_dim, eps.to_bits()],
        inputs: [x, gate],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the norm produces its value",
    }
}


builder! {
    /// `kernels::mlp::situ` / `kernels::mlp::chunked_situ`: Moonshot's
    /// SiTU; `[beta, linear_beta]` are the checkpoint's and ride the run.
    pub fn situ(x: &Val, intermediate: u32, beta: f32, linear_beta: f32) -> Val {
        symbol: "mlp::chunked_situ",
        on: x,
        params: [beta.to_bits(), linear_beta.to_bits()],
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(intermediate)] as BF16,
        made: "the activation produces its value",
    }


    /// `kernels::mlp::situ` with gate and up as separate operands.
    pub fn situ_pair(gate: &Val, up: &Val, intermediate: u32, beta: f32, linear_beta: f32) -> Val {
        symbol: "mlp::situ",
        on: gate,
        params: [beta.to_bits(), linear_beta.to_bits()],
        inputs: [gate, up],
        out: [Dim::Tokens, Dim::Const(intermediate)] as BF16,
        made: "the activation produces its value",
    }


    /// `kernels::ssm::l2norm_scale_bf16_to_fp32`: l2-norm each row and
    /// scale; the epsilon rides the run.
    pub fn l2norm_scale_to_f32(x: &Val, width: u32, eps: f32) -> Val {
        symbol: "ssm::l2norm_scale_bf16_to_fp32",
        on: x,
        params: [eps.to_bits()],
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(width)] as F32,
        made: "the norm produces its value",
    }


    /// `kernels::ssm::bf16_to_fp32`: widen.
    pub fn bf16_to_f32(x: &Val, width: u32) -> Val {
        symbol: "ssm::bf16_to_fp32",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(width)] as F32,
        made: "the cast produces its value",
    }


    /// `kernels::ssm::fp32_to_bf16`: narrow, on the way back out.
    pub fn f32_to_bf16(x: &Val, width: u32) -> Val {
        symbol: "ssm::fp32_to_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the cast produces its value",
    }


    /// `kernels::attn::attn_res_blend`: blend the prefix output with
    pub fn attn_res_blend(
        prefix: &Val,
        blocks: &Val,
        norm_weight: &str,
        proj_weight: &str,
        width: u32,
    ) -> Val {
        symbol: "attn::attn_res_blend",
        on: prefix,
        weights: [norm_weight, proj_weight],
        inputs: [prefix, blocks],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the blend produces its value",
    }
}
