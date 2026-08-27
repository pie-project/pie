//! The `Attention` family: paged attention and its plans, the SSM/linear-attn
//! recurrences, MLA, the DSA index, and the pooled plane.

use super::*;

/// Builds the decode plan from kv space `space`'s declared geometry —
/// once per forward, shared visibly by every layer's `decode` (§6).
pub fn plan_decode(r: &Recorder, space: u32) -> Value {
    let kv_indptr = geometry(r, space, GeomKind::Indptr);
    let kv_indices = geometry(r, space, GeomKind::Indices);
    let last_page_len = geometry(r, space, GeomKind::LastPageLen);
    let kv_len = geometry(r, space, GeomKind::KvLen);
    let plan = r.fresh(Ty::Struct(StructKind::AttnDecodePlan));
    r.push(
        Attention::PlanDecode {
            kv_indptr: kv_indptr.id(),
            kv_indices: kv_indices.id(),
            last_page_len: last_page_len.id(),
            kv_len: kv_len.id(),
            plan: plan.id(),
        },
        &[&kv_indptr, &kv_indices, &last_page_len, &kv_len],
    );
    plan
}

/// Builds the prefill plan from kv space `space`'s declared geometry.
pub fn plan_prefill(r: &Recorder, space: u32) -> Value {
    let kv_indptr = geometry(r, space, GeomKind::Indptr);
    let kv_indices = geometry(r, space, GeomKind::Indices);
    let last_page_len = geometry(r, space, GeomKind::LastPageLen);
    let kv_len = geometry(r, space, GeomKind::KvLen);
    let plan = r.fresh(Ty::Struct(StructKind::AttnPrefillPlan));
    r.push(
        Attention::PlanPrefill {
            kv_indptr: kv_indptr.id(),
            kv_indices: kv_indices.id(),
            last_page_len: last_page_len.id(),
            kv_len: kv_len.id(),
            plan: plan.id(),
        },
        &[&kv_indptr, &kv_indices, &last_page_len, &kv_len],
    );
    plan
}

pub fn decode(
    q: &Value,
    plan: &Value,
    pages: ValueId,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
) -> Value {
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    r.push(
        Attention::Decode {
            q: q.id(),
            plan: plan.id(),
            cache: pages,
            window,
            head_dim,
            sm_scale,
            o: o.id(),
        },
        &[q, plan],
    );
    o
}

pub fn prefill(
    q: &Value,
    plan: &Value,
    pages: ValueId,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
) -> Value {
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    r.push(
        Attention::Prefill {
            q: q.id(),
            plan: plan.id(),
            cache: pages,
            window,
            head_dim,
            kv_heads,
            sm_scale,
            o: o.id(),
        },
        &[q, plan],
    );
    o
}

pub fn masked(
    q: &Value,
    plan: &Value,
    mask: &Value,
    pages: ValueId,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
) -> Value {
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    r.push(
        Attention::Masked {
            q: q.id(),
            plan: plan.id(),
            mask: mask.id(),
            cache: pages,
            window,
            head_dim,
            sm_scale,
            o: o.id(),
        },
        &[q, plan, mask],
    );
    o
}

pub fn decode_lse(
    q: &Value,
    plan: &Value,
    pages: ValueId,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
) -> (Value, Value) {
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    let lse = r.fresh(tensor(
        q.rows(),
        q.width() / u64::from(head_dim),
        Dtype::F32,
    ));
    r.push(
        Attention::DecodeLse {
            q: q.id(),
            plan: plan.id(),
            cache: pages,
            window,
            head_dim,
            sm_scale,
            o: o.id(),
            lse: lse.id(),
        },
        &[q, plan],
    );
    (o, lse)
}

pub fn prefill_lse(
    q: &Value,
    plan: &Value,
    pages: ValueId,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
) -> (Value, Value) {
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    let lse = r.fresh(tensor(
        q.rows(),
        q.width() / u64::from(head_dim),
        Dtype::F32,
    ));
    r.push(
        Attention::PrefillLse {
            q: q.id(),
            plan: plan.id(),
            cache: pages,
            window,
            head_dim,
            kv_heads,
            sm_scale,
            o: o.id(),
            lse: lse.id(),
        },
        &[q, plan],
    );
    (o, lse)
}

pub fn sink(o: &Value, lse: &Value, sink: &Weight, head_dim: u32) -> Value {
    let r = o.rec();
    let o_out = r.fresh(o.ty().clone());
    r.push(
        Attention::Sink {
            o: o.id(),
            lse: lse.id(),
            sink: r.weight(sink),
            head_dim,
            o_out: o_out.id(),
        },
        &[o, lse],
    );
    o_out
}

pub fn merge_lse(
    o1: &Value,
    lse1: &Value,
    o2: &Value,
    lse2: &Value,
    heads: u32,
    head_dim: u32,
) -> (Value, Value) {
    let r = o1.rec();
    let o = r.fresh(o1.ty().clone());
    let lse = r.fresh(lse1.ty().clone());
    r.push(
        Attention::MergeLse {
            o1: o1.id(),
            lse1: lse1.id(),
            o2: o2.id(),
            lse2: lse2.id(),
            heads,
            head_dim,
            o: o.id(),
            lse: lse.id(),
        },
        &[o1, lse1, o2, lse2],
    );
    (o, lse)
}

pub fn logit_softcap(x: &Value, cap: f32) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Attention::LogitSoftcap {
            x: x.id(),
            cap,
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

pub fn kv_append(k: &Value, v: &Value, pages: ValueId, write_page: &Value, write_offset: &Value) {
    let r = k.rec();
    r.push(
        Attention::KvAppend {
            k: k.id(),
            v: v.id(),
            cache: pages,
            write_page: write_page.id(),
            write_offset: write_offset.id(),
        },
        &[k, v, write_page, write_offset],
    );
}

pub fn kv_append_shared(plane: &Value, pages: ValueId, write_page: &Value, write_offset: &Value) {
    let r = plane.rec();
    r.push(
        Attention::KvAppendShared {
            plane: plane.id(),
            cache: pages,
            write_page: write_page.id(),
            write_offset: write_offset.id(),
        },
        &[plane, write_page, write_offset],
    );
}

pub fn ssm_causal_conv1d(x: &Value, weight: &Weight, state: ValueId, conv_width: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Attention::SsmCausalConv1d {
            x: x.id(),
            weight: r.weight(weight),
            state,
            conv_width,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn ssm_causal_conv1d_chunked(
    x: &Value,
    weight: &Weight,
    state: ValueId,
    conv_width: u32,
) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Attention::SsmCausalConv1dChunked {
            x: x.id(),
            weight: r.weight(weight),
            state,
            conv_width,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn ssm_gdn_prep(ba: &Value, dt_bias: &Weight, a_log: &Weight) -> Value {
    let r = ba.rec();
    let gates = r.fresh(tensor(ba.rows(), ba.width(), Dtype::F32));
    r.push(
        Attention::SsmGdnPrep {
            ba: ba.id(),
            dt_bias: r.weight(dt_bias),
            a_log: r.weight(a_log),
            gates: gates.id(),
        },
        &[ba],
    );
    gates
}

pub fn ssm_gated_delta(
    qkv: &Value,
    z: &Value,
    gates: &Value,
    state: ValueId,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
) -> Value {
    let r = qkv.rec();
    let width = u64::from(v_heads) * u64::from(v_dim);
    let y = r.fresh(tensor(qkv.rows(), width, Dtype::F32));
    r.push(
        Attention::SsmGatedDelta {
            qkv: qkv.id(),
            z: z.id(),
            gates: gates.id(),
            state,
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            y: y.id(),
        },
        &[qkv, z, gates],
    );
    y
}

pub fn ssm_gated_delta_chunked(
    qkv: &Value,
    z: &Value,
    gates: &Value,
    state: ValueId,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
) -> Value {
    let r = qkv.rec();
    let width = u64::from(v_heads) * u64::from(v_dim);
    let y = r.fresh(tensor(qkv.rows(), width, Dtype::F32));
    r.push(
        Attention::SsmGatedDeltaChunked {
            qkv: qkv.id(),
            z: z.id(),
            gates: gates.id(),
            state,
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            y: y.id(),
        },
        &[qkv, z, gates],
    );
    y
}

pub fn ssm_kda_step(
    mixed: &Value,
    f: &Value,
    b: &Value,
    dt_bias: &Weight,
    a_log: &Weight,
    state: ValueId,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
) -> Value {
    let r = mixed.rec();
    let width = u64::from(heads) * u64::from(head_dim);
    let y = r.fresh(tensor(mixed.rows(), width, Dtype::F32));
    r.push(
        Attention::SsmKdaStep {
            mixed: mixed.id(),
            f: f.id(),
            b: b.id(),
            dt_bias: r.weight(dt_bias),
            a_log: r.weight(a_log),
            state,
            heads,
            head_dim,
            norm_eps,
            y: y.id(),
        },
        &[mixed, f, b],
    );
    y
}

pub fn ssm_kda_chunked(
    mixed: &Value,
    f: &Value,
    b: &Value,
    dt_bias: &Weight,
    a_log: &Weight,
    state: ValueId,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
) -> Value {
    let r = mixed.rec();
    let width = u64::from(heads) * u64::from(head_dim);
    let y = r.fresh(tensor(mixed.rows(), width, Dtype::F32));
    r.push(
        Attention::SsmKdaChunked {
            mixed: mixed.id(),
            f: f.id(),
            b: b.id(),
            dt_bias: r.weight(dt_bias),
            a_log: r.weight(a_log),
            state,
            heads,
            head_dim,
            norm_eps,
            y: y.id(),
        },
        &[mixed, f, b],
    );
    y
}

/// Builds the one MLA plan — shared by decode and prefill — from kv
/// space `space`'s declared geometry.
pub fn mla_plan(r: &Recorder, space: u32) -> Value {
    let kv_indptr = geometry(r, space, GeomKind::Indptr);
    let kv_indices = geometry(r, space, GeomKind::Indices);
    let last_page_len = geometry(r, space, GeomKind::LastPageLen);
    let kv_len = geometry(r, space, GeomKind::KvLen);
    let plan = r.fresh(Ty::Struct(StructKind::MlaPlan));
    r.push(
        Attention::MlaPlan {
            kv_indptr: kv_indptr.id(),
            kv_indices: kv_indices.id(),
            last_page_len: last_page_len.id(),
            kv_len: kv_len.id(),
            plan: plan.id(),
        },
        &[&kv_indptr, &kv_indices, &last_page_len, &kv_len],
    );
    plan
}

pub fn mla_latents(kv_a: &Value, norm: &Weight, eps: f32, kv_lora_rank: u32) -> (Value, Value) {
    let r = kv_a.rec();
    let kv_c = r.fresh(tensor(kv_a.rows(), kv_lora_rank, kv_a.dtype()));
    let k_pe = r.fresh(tensor(
        kv_a.rows(),
        kv_a.width() - u64::from(kv_lora_rank),
        kv_a.dtype(),
    ));
    r.push(
        Attention::MlaLatents {
            kv_a: kv_a.id(),
            weight: r.weight(norm),
            eps,
            kv_lora_rank,
            kv_c: kv_c.id(),
            k_pe: k_pe.id(),
        },
        &[kv_a],
    );
    (kv_c, k_pe)
}

pub fn mla_latents_rope(
    kv_a: &Value,
    positions: &Value,
    norm: &Weight,
    eps: f32,
    kv_lora_rank: u32,
    rope_dim: u32,
    theta: f32,
) -> (Value, Value) {
    let r = kv_a.rec();
    let kv_c = r.fresh(tensor(kv_a.rows(), kv_lora_rank, kv_a.dtype()));
    let k_pe = r.fresh(tensor(
        kv_a.rows(),
        kv_a.width() - u64::from(kv_lora_rank),
        kv_a.dtype(),
    ));
    r.push(
        Attention::MlaLatentsRope {
            kv_a: kv_a.id(),
            positions: positions.id(),
            weight: r.weight(norm),
            eps,
            kv_lora_rank,
            rope_dim,
            theta,
            kv_c: kv_c.id(),
            k_pe: k_pe.id(),
        },
        &[kv_a, positions],
    );
    (kv_c, k_pe)
}

pub fn mla_split_q_b(q_b: &Value, heads: u32, nope_dim: u32, rope_dim: u32) -> (Value, Value) {
    let r = q_b.rec();
    let heads64 = u64::from(heads);
    let q_nope = r.fresh(tensor(
        q_b.rows(),
        heads64 * u64::from(nope_dim),
        q_b.dtype(),
    ));
    let q_pe = r.fresh(tensor(
        q_b.rows(),
        heads64 * u64::from(rope_dim),
        q_b.dtype(),
    ));
    r.push(
        Attention::MlaSplitQB {
            q_b: q_b.id(),
            heads,
            nope_dim,
            rope_dim,
            q_nope: q_nope.id(),
            q_pe: q_pe.id(),
        },
        &[q_b],
    );
    (q_nope, q_pe)
}

pub fn mla_absorb_q(
    q_nope: &Value,
    kv_b: &Weight,
    heads: u32,
    kv_lora_rank: u32,
    nope_dim: u32,
    v_head_dim: u32,
) -> Value {
    let r = q_nope.rec();
    let width = u64::from(heads) * u64::from(kv_lora_rank);
    let q_latent = r.fresh(tensor(q_nope.rows(), width, q_nope.dtype()));
    r.push(
        Attention::MlaAbsorbQ {
            q_nope: q_nope.id(),
            kv_b: r.weight(kv_b),
            heads,
            kv_lora_rank,
            nope_dim,
            v_head_dim,
            q_latent: q_latent.id(),
        },
        &[q_nope],
    );
    q_latent
}

/// The trailing pair reads `nope_dim, v_head_dim`, the same order
/// [`mla_absorb_q`] takes it in: the two calls sit a dozen lines apart in every
/// MLA text, take the same four numbers, and used to take the last two the
/// other way round — which no call site can see and no type can catch.
pub fn mla_absorb_out(
    latent: &Value,
    kv_b: &Weight,
    heads: u32,
    kv_lora_rank: u32,
    nope_dim: u32,
    v_head_dim: u32,
) -> Value {
    let r = latent.rec();
    let width = u64::from(heads) * u64::from(v_head_dim);
    let o = r.fresh(tensor(latent.rows(), width, latent.dtype()));
    r.push(
        Attention::MlaAbsorbOut {
            latent: latent.id(),
            kv_b: r.weight(kv_b),
            heads,
            kv_lora_rank,
            v_head_dim,
            nope_dim,
            o: o.id(),
        },
        &[latent],
    );
    o
}

pub fn mla_kv_append(
    kv_c: &Value,
    k_pe: &Value,
    pages: ValueId,
    write_page: &Value,
    write_offset: &Value,
) {
    let r = kv_c.rec();
    r.push(
        Attention::MlaKvAppend {
            kv_c: kv_c.id(),
            k_pe: k_pe.id(),
            cache: pages,
            write_page: write_page.id(),
            write_offset: write_offset.id(),
        },
        &[kv_c, k_pe, write_page, write_offset],
    );
}

pub fn mla_decode(
    q: &Value,
    plan: &Value,
    q_pe: &Value,
    pages: ValueId,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
) -> Value {
    let r = q.rec();
    let width = u64::from(heads) * u64::from(kv_lora_rank);
    let o = r.fresh(tensor(q.rows(), width, q.dtype()));
    r.push(
        Attention::MlaDecode {
            q: q.id(),
            plan: plan.id(),
            q_pe: q_pe.id(),
            cache: pages,
            heads,
            kv_lora_rank,
            sm_scale,
            o: o.id(),
        },
        &[q, plan, q_pe],
    );
    o
}

pub fn mla_prefill(
    q: &Value,
    plan: &Value,
    q_pe: &Value,
    pages: ValueId,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
) -> Value {
    let r = q.rec();
    let width = u64::from(heads) * u64::from(kv_lora_rank);
    let o = r.fresh(tensor(q.rows(), width, q.dtype()));
    r.push(
        Attention::MlaPrefill {
            q: q.id(),
            plan: plan.id(),
            q_pe: q_pe.id(),
            cache: pages,
            heads,
            kv_lora_rank,
            sm_scale,
            o: o.id(),
        },
        &[q, plan, q_pe],
    );
    o
}

pub fn mla_decode_selected(
    q: &Value,
    plan: &Value,
    q_pe: &Value,
    selection: &Value,
    pages: ValueId,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
) -> Value {
    let r = q.rec();
    let width = u64::from(heads) * u64::from(kv_lora_rank);
    let o = r.fresh(tensor(q.rows(), width, q.dtype()));
    r.push(
        Attention::MlaDecodeSelected {
            q: q.id(),
            plan: plan.id(),
            q_pe: q_pe.id(),
            selection: selection.id(),
            cache: pages,
            heads,
            kv_lora_rank,
            sm_scale,
            o: o.id(),
        },
        &[q, plan, q_pe, selection],
    );
    o
}

pub fn mla_prefill_selected(
    q: &Value,
    plan: &Value,
    q_pe: &Value,
    selection: &Value,
    pages: ValueId,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
) -> Value {
    let r = q.rec();
    let width = u64::from(heads) * u64::from(kv_lora_rank);
    let o = r.fresh(tensor(q.rows(), width, q.dtype()));
    r.push(
        Attention::MlaPrefillSelected {
            q: q.id(),
            plan: plan.id(),
            q_pe: q_pe.id(),
            selection: selection.id(),
            cache: pages,
            heads,
            kv_lora_rank,
            sm_scale,
            o: o.id(),
        },
        &[q, plan, q_pe, selection],
    );
    o
}

pub fn index_layernorm_rope(
    k: &Value,
    positions: &Value,
    norm: &Weight,
    eps: f32,
    bias: &Weight,
    rope_dim: u32,
    theta: f32,
) -> Value {
    let r = k.rec();
    let k_out = r.fresh(k.ty().clone());
    r.push(
        Attention::IndexLayernormRope {
            k: k.id(),
            positions: positions.id(),
            weight: r.weight(norm),
            bias: r.weight(bias),
            eps,
            rope_dim,
            theta,
            k_out: k_out.id(),
        },
        &[k, positions],
    );
    k_out
}

pub fn index_rope(
    q: &Value,
    positions: &Value,
    heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
) -> Value {
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    r.push(
        Attention::IndexRope {
            q: q.id(),
            positions: positions.id(),
            heads,
            head_dim,
            rope_dim,
            theta,
            q_out: q_out.id(),
        },
        &[q, positions],
    );
    q_out
}

pub fn index_topk(
    q: &Value,
    weights: &Value,
    keys: ValueId,
    heads: u32,
    head_dim: u32,
    top_k: u32,
) -> Value {
    let r = q.rec();
    let selection = r.fresh(tensor(q.rows(), top_k, Dtype::I32));
    r.push(
        Attention::IndexTopk {
            q: q.id(),
            weights: weights.id(),
            keys,
            heads,
            head_dim,
            top_k,
            selection: selection.id(),
        },
        &[q, weights],
    );
    selection
}

pub fn index_kv_append(k: &Value, keys: ValueId, write_page: &Value, write_offset: &Value) {
    let r = k.rec();
    r.push(
        Attention::IndexKvAppend {
            k: k.id(),
            keys,
            write_page: write_page.id(),
            write_offset: write_offset.id(),
        },
        &[k, write_page, write_offset],
    );
}

/// `row_valid` masks graph-padding rows out of the boundary math.
pub fn pool_boundary_decode(positions: &Value, row_valid: &Value, ratio: u32) -> (Value, Value) {
    let r = positions.rec();
    let boundary_pos = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    let boundary_req = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    r.push(
        Attention::PoolBoundaryDecode {
            positions: positions.id(),
            row_valid: row_valid.id(),
            ratio,
            boundary_pos: boundary_pos.id(),
            boundary_req: boundary_req.id(),
        },
        &[positions, row_valid],
    );
    (boundary_pos, boundary_req)
}

/// `row_valid` masks graph-padding rows out of the boundary math.
pub fn pool_boundary_prefill(positions: &Value, row_valid: &Value, ratio: u32) -> (Value, Value) {
    let r = positions.rec();
    let boundary_pos = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    let boundary_req = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    r.push(
        Attention::PoolBoundaryPrefill {
            positions: positions.id(),
            row_valid: row_valid.id(),
            ratio,
            boundary_pos: boundary_pos.id(),
            boundary_req: boundary_req.id(),
        },
        &[positions, row_valid],
    );
    (boundary_pos, boundary_req)
}

/// `dtype` is the pooled entries' element type — the wrapper has no data
/// input to inherit it from, so the model states its activation dtype
/// here, from its own declaration (`m.act`): no implicit driver state.
pub fn pool_gather(
    boundary_pos: &Value,
    boundary_req: &Value,
    pages: ValueId,
    head_dim: u32,
    ratio: u32,
    dtype: Dtype,
) -> Value {
    let r = boundary_pos.rec();
    let entries = r.fresh(tensor(Dim::Tokens, head_dim, dtype));
    r.push(
        Attention::PoolGather {
            boundary_pos: boundary_pos.id(),
            boundary_req: boundary_req.id(),
            pages,
            head_dim,
            ratio,
            entries: entries.id(),
        },
        &[boundary_pos, boundary_req],
    );
    entries
}

pub fn pool_kv_append(
    entries: &Value,
    boundary_pos: &Value,
    boundary_req: &Value,
    pool: ValueId,
    write_page: &Value,
    write_offset: &Value,
) {
    let r = entries.rec();
    r.push(
        Attention::PoolKvAppend {
            entries: entries.id(),
            boundary_pos: boundary_pos.id(),
            boundary_req: boundary_req.id(),
            pool,
            write_page: write_page.id(),
            write_offset: write_offset.id(),
        },
        &[
            entries,
            boundary_pos,
            boundary_req,
            write_page,
            write_offset,
        ],
    );
}

/// `request_of_token` maps each token row to its owning lane.
pub fn pool_lse(
    q: &Value,
    positions: &Value,
    request_of_token: &Value,
    entries: ValueId,
    ratio: u32,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
) -> (Value, Value) {
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    let lse = r.fresh(tensor(q.rows(), heads, Dtype::F32));
    r.push(
        Attention::PoolLse {
            q: q.id(),
            positions: positions.id(),
            request_of_token: request_of_token.id(),
            entries,
            ratio,
            heads,
            head_dim,
            sm_scale,
            o: o.id(),
            lse: lse.id(),
        },
        &[q, positions, request_of_token],
    );
    (o, lse)
}
