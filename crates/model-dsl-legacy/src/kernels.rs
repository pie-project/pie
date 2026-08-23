//! The stated-kernel surface. Tier-1 fns state role points and every plane
//! answers them; a plane's own symbols live under its module and appear
//! only behind that plane's gate.

use crate::axes::Dtype;
use crate::declare::{Norm, Tensor, Yarn};
use crate::forward::{Pages, State};
use crate::record::{Value, Windows};

pub fn embed<W: Dtype>(ids: &Value, table: &Tensor<W>) -> Value {
    ids.stmt("layout.embed").weight(table).done()
}

pub fn matmul<W: Dtype>(x: &Value, w: &Tensor<W>) -> Value {
    x.stmt("gemm.matmul").weight(w).done()
}

pub fn add(a: &Value, b: &Value) -> Value {
    a.stmt("norm.residual_add").value(b).done()
}

pub fn add_bias<W: Dtype>(x: &Value, b: &Tensor<W>) -> Value {
    x.stmt("norm.add_bias").weight(b).done()
}

pub fn scale<W: Dtype>(x: &Value, s: &Tensor<W>) -> Value {
    x.stmt("norm.scale").weight(s).done()
}

pub fn select(table: &Value, layer: u32) -> Value {
    table.stmt("layout.select").int(layer).done()
}

pub fn rmsnorm<W: Dtype>(x: &Value, n: &Norm<W>) -> Value {
    x.stmt("norm.rmsnorm").norm(n).done()
}

pub fn rmsnorm_per_head<W: Dtype>(x: &Value, n: &Norm<W>) -> Value {
    x.stmt("rmsnorm.per_head").norm(n).done()
}

pub fn rmsnorm_no_scale(x: &Value, head_dim: u32) -> Value {
    x.stmt("norm.rmsnorm_no_scale").int(head_dim).done()
}

pub fn rmsnorm_gated<W: Dtype>(x: &Value, n: &Norm<W>) -> Value {
    x.stmt("norm.rmsnorm_gated").norm(n).done()
}

pub fn rope(q: &Value, k: &Value, head_dim: u32, theta: f32, pos: &Value) -> (Value, Value) {
    q.stmt("rope.full").value(k).value(pos).int(head_dim).float(theta).pair()
}

pub fn rope_partial(
    q: &Value,
    k: &Value,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    pos: &Value,
) -> (Value, Value) {
    q.stmt("rope.partial").value(k).value(pos).int(rotary_dim).int(head_dim).float(theta).pair()
}

pub fn rope_partial_q(q: &Value, rotary_dim: u32, head_dim: u32, theta: f32, pos: &Value) -> Value {
    q.stmt("rope.partial_q").value(pos).int(rotary_dim).int(head_dim).float(theta).done()
}

pub fn rope_yarn(q: &Value, k: &Value, head_dim: u32, yarn: &Yarn, pos: &Value) -> (Value, Value) {
    q.stmt("rope.yarn")
        .value(k)
        .value(pos)
        .int(head_dim)
        .float(yarn.theta)
        .float(yarn.factor)
        .float(yarn.beta_fast)
        .float(yarn.beta_slow)
        .float(yarn.attention_factor)
        .int(yarn.original_max_position)
        .pair()
}

pub fn kv_append(k: &Value, v: &Value, pages: &Pages) {
    k.stmt("attention.kv_append").value(v).cache(&pages.name).effect();
}

pub fn split_qkv(x: &Value, q_width: u32, kv_width: u32) -> (Value, Value, Value) {
    x.stmt("layout.split_qkv").int(q_width).int(kv_width).triple()
}

pub fn split_q_gate(x: &Value, head_dim: u32) -> (Value, Value) {
    x.stmt("layout.split_q_gate").int(head_dim).pair()
}

pub fn split_rows(x: &Value, width: u32) -> (Value, Value) {
    x.stmt("layout.split_rows").int(width).pair()
}

pub fn query_windows(x: &Value) -> Windows {
    Windows {
        data: x.clone(),
        indptr: x.rec.runtime("qo_indptr"),
    }
}

pub fn attention_decode(q: &Value, pages: &Pages, window: Option<u32>, head_dim: u32, sm_scale: f32) -> Value {
    q.stmt("attention.decode").cache(&pages.name).window(window).int(head_dim).float(sm_scale).done()
}

pub fn attention_prefill(
    w: &Windows,
    pages: &Pages,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
) -> Value {
    w.data
        .stmt("attention.prefill")
        .value(&w.indptr)
        .cache(&pages.name)
        .window(window)
        .int(head_dim)
        .int(kv_heads)
        .float(sm_scale)
        .done()
}

pub fn attention_masked(w: &Windows, pages: &Pages, window: Option<u32>, head_dim: u32, sm_scale: f32) -> Value {
    w.data
        .stmt("attention.masked")
        .value(&w.indptr)
        .cache(&pages.name)
        .window(window)
        .int(head_dim)
        .float(sm_scale)
        .done()
}

pub fn attention_decode_lse(
    q: &Value,
    pages: &Pages,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
) -> (Value, Value) {
    q.stmt("attention.decode_lse")
        .cache(&pages.name)
        .window(window)
        .int(head_dim)
        .float(sm_scale)
        .pair()
}

pub fn attention_prefill_lse(
    w: &Windows,
    pages: &Pages,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
) -> (Value, Value) {
    w.data
        .stmt("attention.prefill_lse")
        .value(&w.indptr)
        .cache(&pages.name)
        .window(window)
        .int(head_dim)
        .int(kv_heads)
        .float(sm_scale)
        .pair()
}

pub fn attention_landing<W: Dtype>(a: &Value, o_proj: &Tensor<W>, layer: u32) -> Value {
    a.stmt("gemm.attention_landing").weight(o_proj).layer(layer).done()
}

pub fn all_reduce(x: &Value) -> Value {
    x.stmt("dist.all_reduce").done()
}

pub fn swiglu(x: &Value, intermediate: u32) -> Value {
    x.stmt("swiglu").int(intermediate).done()
}

pub fn swiglu_clamp_alpha(x: &Value, intermediate: u32, limit: f32, alpha: f32) -> Value {
    x.stmt("swiglu.clamp_alpha").int(intermediate).float(limit).float(alpha).done()
}

pub fn geglu_tanh(gate: &Value, up: &Value) -> Value {
    gate.stmt("swiglu.geglu_tanh").value(up).done()
}

pub fn geglu_tanh_packed(x: &Value, intermediate: u32) -> Value {
    x.stmt("swiglu.geglu_tanh_packed").int(intermediate).done()
}

pub fn logit_softcap(logits: &Value, cap: f32) -> Value {
    logits.stmt("attention.logit_softcap").float(cap).done()
}

pub fn lm_head<W: Dtype>(x: &Value, bank: &Tensor<W>) -> Value {
    x.stmt("gemm.lm_head").weight(bank).done()
}

pub fn topk_softmax(scores: &Value, experts: u32, top_k: u32) -> Value {
    scores.stmt("moe.topk_softmax").int(experts).int(top_k).done()
}

pub fn matmul_select<W: Dtype>(x: &Value, bank: &Tensor<W>, routes: &Value) -> Value {
    x.stmt("moe.matmul_select").value(routes).weight(bank).done()
}

pub fn matmul_select_bias<W: Dtype, B: Dtype>(
    x: &Value,
    bank: &Tensor<W>,
    bias: &Tensor<B>,
    routes: &Value,
) -> Value {
    x.stmt("moe.matmul_select_bias").value(routes).weight(bank).weight(bias).done()
}

pub fn weighted_sum(routed: &Value, routes: &Value) -> Value {
    routed.stmt("moe.weighted_sum").value(routes).done()
}

pub fn sigmoid_gate_add(routed: &Value, shared: &Value, gate: &Value) -> Value {
    routed.stmt("moe.sigmoid_gate_add").value(shared).value(gate).done()
}

pub fn sigmoid_gate_mul(x: &Value, gate: &Value) -> Value {
    x.stmt("gate.sigmoid_mul").value(gate).done()
}

pub fn causal_conv1d<W: Dtype>(x: &Value, conv: &Tensor<W>, state: &State) -> Value {
    x.stmt("ssm.causal_conv1d").weight(conv).cache(&state.name).done()
}

pub fn causal_conv1d_chunked<W: Dtype>(w: &Windows, conv: &Tensor<W>, state: &State) -> Value {
    w.data
        .stmt("ssm.causal_conv1d_chunked")
        .value(&w.indptr)
        .weight(conv)
        .cache(&state.name)
        .done()
}

pub fn gdn_prep<W: Dtype>(ba: &Value, dt_bias: &Tensor<W>, a_log: &Tensor<W>) -> Value {
    ba.stmt("ssm.gdn_prep").weight(dt_bias).weight(a_log).done()
}

pub fn gated_delta(
    qkv: &Value,
    z: &Value,
    gates: &Value,
    state: &State,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
) -> Value {
    qkv.stmt("ssm.gated_delta")
        .value(z)
        .value(gates)
        .cache(&state.name)
        .int(k_heads)
        .int(v_heads)
        .int(k_dim)
        .int(v_dim)
        .done()
}

pub fn gated_delta_chunked(
    w: &Windows,
    z: &Value,
    gates: &Value,
    state: &State,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
) -> Value {
    w.data
        .stmt("ssm.gated_delta_chunked")
        .value(&w.indptr)
        .value(z)
        .value(gates)
        .cache(&state.name)
        .int(k_heads)
        .int(v_heads)
        .int(k_dim)
        .int(v_dim)
        .done()
}

pub fn mla_latents<W: Dtype>(kv_a: &Value, norm: &Norm<W>, kv_lora_rank: u32) -> (Value, Value) {
    kv_a.stmt("mla.latents").norm(norm).int(kv_lora_rank).pair()
}

pub fn mla_latents_rope<W: Dtype>(
    kv_a: &Value,
    norm: &Norm<W>,
    kv_lora_rank: u32,
    rope_dim: u32,
    theta: f32,
    pos: &Value,
) -> (Value, Value) {
    kv_a.stmt("mla.latents_rope")
        .value(pos)
        .norm(norm)
        .int(kv_lora_rank)
        .int(rope_dim)
        .float(theta)
        .pair()
}

pub fn split_q_b(q_b: &Value, heads: u32, nope_dim: u32, rope_dim: u32) -> (Value, Value) {
    q_b.stmt("mla.split_q_b").int(heads).int(nope_dim).int(rope_dim).pair()
}

pub fn kv_append_mla(kv_c: &Value, k_pe: &Value, pages: &Pages) {
    kv_c.stmt("mla.kv_append").value(k_pe).cache(&pages.name).effect();
}

pub fn mla_absorb_q<W: Dtype>(
    q_nope: &Value,
    kv_b: &Tensor<W>,
    heads: u32,
    kv_lora_rank: u32,
    nope_dim: u32,
) -> Value {
    q_nope.stmt("mla.absorb_q").weight(kv_b).int(heads).int(kv_lora_rank).int(nope_dim).done()
}

pub fn mla_absorb_q_pe<W: Dtype>(
    q_nope: &Value,
    q_pe: &Value,
    kv_b: &Tensor<W>,
    heads: u32,
    kv_lora_rank: u32,
    nope_dim: u32,
) -> Value {
    q_nope
        .stmt("mla.absorb_q_pe")
        .value(q_pe)
        .weight(kv_b)
        .int(heads)
        .int(kv_lora_rank)
        .int(nope_dim)
        .done()
}

pub fn mla_absorb_out<W: Dtype>(
    latent: &Value,
    kv_b: &Tensor<W>,
    heads: u32,
    kv_lora_rank: u32,
    v_head_dim: u32,
) -> Value {
    latent.stmt("mla.absorb_out").weight(kv_b).int(heads).int(kv_lora_rank).int(v_head_dim).done()
}

pub fn mla_attention_decode(
    q: &Value,
    q_pe: &Value,
    pages: &Pages,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
) -> Value {
    q.stmt("mla.attention_decode")
        .value(q_pe)
        .cache(&pages.name)
        .int(heads)
        .int(kv_lora_rank)
        .float(sm_scale)
        .done()
}

pub fn mla_attention_prefill(
    w: &Windows,
    pe: &Windows,
    pages: &Pages,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
) -> Value {
    w.data
        .stmt("mla.attention_prefill")
        .value(&w.indptr)
        .value(&pe.data)
        .cache(&pages.name)
        .int(heads)
        .int(kv_lora_rank)
        .float(sm_scale)
        .done()
}

pub fn mla_attention_decode_selected(
    q: &Value,
    pages: &Pages,
    selection: &Value,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
) -> Value {
    q.stmt("mla.attention_decode_selected")
        .value(selection)
        .cache(&pages.name)
        .int(heads)
        .int(kv_lora_rank)
        .float(sm_scale)
        .done()
}

pub fn mla_attention_prefill_selected(
    w: &Windows,
    pages: &Pages,
    selection: &Value,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
) -> Value {
    w.data
        .stmt("mla.attention_prefill_selected")
        .value(&w.indptr)
        .value(selection)
        .cache(&pages.name)
        .int(heads)
        .int(kv_lora_rank)
        .float(sm_scale)
        .done()
}

pub fn index_layernorm_rope<W: Dtype>(
    k: &Value,
    norm: &Norm<W>,
    bias: &Tensor<W>,
    rope_dim: u32,
    theta: f32,
    pos: &Value,
) -> Value {
    k.stmt("index.layernorm_rope")
        .value(pos)
        .norm(norm)
        .weight(bias)
        .int(rope_dim)
        .float(theta)
        .done()
}

pub fn kv_append_index(k: &Value, keys: &Pages) {
    k.stmt("index.kv_append").cache(&keys.name).effect();
}

pub fn index_rope(q: &Value, heads: u32, head_dim: u32, rope_dim: u32, theta: f32, pos: &Value) -> Value {
    q.stmt("index.rope").value(pos).int(heads).int(head_dim).int(rope_dim).float(theta).done()
}

pub fn index_topk(
    q: &Value,
    weights: &Value,
    keys: &Pages,
    heads: u32,
    head_dim: u32,
    top_k: u32,
) -> Value {
    q.stmt("index.topk")
        .value(weights)
        .cache(&keys.name)
        .int(heads)
        .int(head_dim)
        .int(top_k)
        .done()
}

pub fn kda_step<W: Dtype>(
    mixed: &Value,
    f: &Value,
    b: &Value,
    dt_bias: &Tensor<W>,
    a_log: &Tensor<W>,
    delta_state: &State,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
) -> Value {
    mixed
        .stmt("ssm.kda_step")
        .value(f)
        .value(b)
        .weight(dt_bias)
        .weight(a_log)
        .cache(&delta_state.name)
        .int(heads)
        .int(head_dim)
        .float(norm_eps)
        .done()
}

pub fn kda_chunked<W: Dtype>(
    w: &Windows,
    f: &Value,
    b: &Value,
    dt_bias: &Tensor<W>,
    a_log: &Tensor<W>,
    delta_state: &State,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
) -> Value {
    w.data
        .stmt("ssm.kda_chunked")
        .value(&w.indptr)
        .value(f)
        .value(b)
        .weight(dt_bias)
        .weight(a_log)
        .cache(&delta_state.name)
        .int(heads)
        .int(head_dim)
        .float(norm_eps)
        .done()
}

pub fn rmsnorm_gated_by<W: Dtype>(x: &Value, gate: &Value, n: &Norm<W>) -> Value {
    x.stmt("norm.rmsnorm_gated_by").value(gate).norm(n).done()
}

pub fn res_blend<W: Dtype>(y: &Value, blocks: &[Value], n: &Norm<W>, proj: &Tensor<W>) -> Value {
    let mut s = y.stmt("norm.res_blend");
    for block in blocks {
        s = s.value(block);
    }
    s.norm(n).weight(proj).done()
}

pub fn situ(x: &Value, intermediate: u32, beta: f32, up_cap: Option<f32>) -> Value {
    x.stmt("situ").int(intermediate).float(beta).float(up_cap.unwrap_or(0.0)).done()
}

pub fn swiglu_clamp(x: &Value, intermediate: u32, limit: f32) -> Value {
    x.stmt("swiglu.clamp").int(intermediate).float(limit).done()
}

pub fn topk_sigmoid(
    scores: &Value,
    experts: u32,
    top_k: u32,
    norm_weights: bool,
    scaling: f32,
) -> Value {
    scores
        .stmt("moe.topk_sigmoid")
        .int(experts)
        .int(top_k)
        .int(u32::from(norm_weights))
        .float(scaling)
        .done()
}

pub fn topk_sqrt_softplus<W: Dtype>(
    scores: &Value,
    bias: &Tensor<W>,
    experts: u32,
    top_k: u32,
    renorm: bool,
    scaling: f32,
) -> Value {
    scores
        .stmt("moe.topk_sqrt_softplus")
        .weight(bias)
        .int(experts)
        .int(top_k)
        .int(u32::from(renorm))
        .float(scaling)
        .done()
}

pub fn rope_partial_last(x: &Value, rope_dim: u32, head_dim: u32, theta: f32, pos: &Value) -> Value {
    x.stmt("rope.partial_last").value(pos).int(rope_dim).int(head_dim).float(theta).done()
}

pub fn kv_append_shared(plane: &Value, pages: &Pages) {
    plane.stmt("attention.kv_append_shared").cache(&pages.name).effect();
}

pub fn lse_ln(lse: &Value) -> Value {
    lse.stmt("attention.lse_ln").done()
}

pub fn pool_boundary_decode(pos: &Value, ratio: u32) -> (Value, Value) {
    pos.stmt("pool.boundary_decode").int(ratio).pair()
}

pub fn pool_boundary_prefill(w: &Windows, ratio: u32) -> (Value, Value) {
    w.data.stmt("pool.boundary_prefill").value(&w.indptr).int(ratio).pair()
}

pub fn pool_gather(bpos: &Value, breq: &Value, pages: &Pages, head_dim: u32, ratio: u32) -> Value {
    bpos.stmt("pool.gather").value(breq).cache(&pages.name).int(head_dim).int(ratio).done()
}

pub fn kv_append_pool(pooled: &Value, bpos: &Value, breq: &Value, entries: &Pages) {
    pooled.stmt("pool.kv_append").value(bpos).value(breq).cache(&entries.name).effect();
}

pub fn attention_pooled_lse(
    q: &Value,
    entries: &Pages,
    ratio: u32,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
    pos: &Value,
) -> (Value, Value) {
    q.stmt("pool.attention_lse")
        .value(pos)
        .cache(&entries.name)
        .int(ratio)
        .int(heads)
        .int(head_dim)
        .float(sm_scale)
        .pair()
}

pub fn attention_merge_lse(
    o: &Value,
    lse: &Value,
    pooled: &Value,
    pooled_lse: &Value,
    heads: u32,
    head_dim: u32,
) -> (Value, Value) {
    o.stmt("attention.merge_lse")
        .value(lse)
        .value(pooled)
        .value(pooled_lse)
        .int(heads)
        .int(head_dim)
        .pair()
}

pub fn attention_sink<W: Dtype>(o: &Value, lse: &Value, sink: &Tensor<W>, head_dim: u32) -> Value {
    o.stmt("attention.sink").value(lse).weight(sink).int(head_dim).done()
}

pub fn hc_expand(x: &Value, streams: u32) -> Value {
    x.stmt("hc.expand").int(streams).done()
}

pub fn hc_rmsnorm_f32(streams: &Value, eps: f32) -> Value {
    streams.stmt("hc.rmsnorm_f32").float(eps).done()
}

pub fn hc_gates<W: Dtype>(
    normed: &Value,
    scale: &Tensor<W>,
    base: &Tensor<W>,
    streams: &Value,
    stream_count: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
) -> (Value, Value, Value) {
    normed
        .stmt("hc.gates")
        .value(streams)
        .weight(scale)
        .weight(base)
        .int(stream_count)
        .float(gate_eps)
        .float(alpha)
        .int(sinkhorn)
        .triple()
}

pub fn hc_fold(x: &Value, streams: &Value, post_mix: &Value, comb_mix: &Value) -> Value {
    x.stmt("hc.fold").value(streams).value(post_mix).value(comb_mix).done()
}

pub fn hc_collapse<W: Dtype>(
    streams: &Value,
    head_scale: &Tensor<W>,
    head_base: &Tensor<W>,
    stream_count: u32,
    gate_eps: f32,
) -> Value {
    streams
        .stmt("hc.collapse")
        .weight(head_scale)
        .weight(head_base)
        .int(stream_count)
        .float(gate_eps)
        .done()
}

pub mod cuda {
    use super::*;

    pub fn qkv_fused_qknorm_rope_vnorm_write<W: Dtype>(
        x: &Value,
        q_norm: &Norm<W>,
        k_norm: &Norm<W>,
        kv_heads: u32,
        head_dim: u32,
        pages: &Pages,
        theta: f32,
        pos: &Value,
    ) -> Value {
        x.stmt("cuda::qkv_fused_qknorm_rope_vnorm_write")
            .value(pos)
            .norm(q_norm)
            .norm(k_norm)
            .cache(&pages.name)
            .int(kv_heads)
            .int(head_dim)
            .float(theta)
            .done()
    }
}
