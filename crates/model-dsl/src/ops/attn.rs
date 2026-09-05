//! The `Attention` family: paged attention and its plans, the SSM/linear-attn
//! recurrences, MLA, the DSA index, and the pooled plane.

use super::*;
use crate::forward::Input;

/// Builds the decode plan off `inputs`' geometry and reading, once per
/// (reading × class) at the top of `forward`, shared by every layer's decode.
///
/// The plan is guarded by the arm it was built off (`Recorder::push` meets
/// that arm's conds into the plan node's guard), so a query from another arm
/// is refused. There is no interning or dedup: sharing a plan across layers
/// is a fact the text states by hoisting it once, not something a cache
/// inferred from matching numbers.
pub fn plan_decode<F>(
    inputs: &Input<F>,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    window: Option<u32>,
) -> Value {
    let kv_indptr = inputs.kv_indptr();
    let kv_indices = inputs.kv_indices();
    let last_page_len = inputs.last_page_len();
    let kv_len = inputs.kv_len();
    let r = kv_indptr.rec();
    let plan = r.fresh(Ty::Struct(StructKind::AttnDecodePlan));
    r.push(
        Attention::PlanDecode {
            kv_indptr: kv_indptr.id(),
            kv_indices: kv_indices.id(),
            last_page_len: last_page_len.id(),
            kv_len: kv_len.id(),
            q_heads,
            kv_heads,
            head_dim,
            window,
            plan: plan.id(),
        },
        &[&kv_indptr, &kv_indices, &last_page_len, &kv_len],
    );
    plan
}

/// Builds the prefill plan off `inputs`' geometry and its reading. Like
/// [`plan_decode`], the plan's guard is the arm it was built off, and its
/// reading is stated here rather than inferred from its readers.
pub fn plan_prefill<F>(
    inputs: &Input<F>,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    window: Option<u32>,
) -> Value {
    let kv_indptr = inputs.kv_indptr();
    let kv_indices = inputs.kv_indices();
    let last_page_len = inputs.last_page_len();
    let kv_len = inputs.kv_len();
    let r = kv_indptr.rec();
    let plan = r.fresh(Ty::Struct(StructKind::AttnPrefillPlan));
    r.push(
        Attention::PlanPrefill {
            kv_indptr: kv_indptr.id(),
            kv_indices: kv_indices.id(),
            last_page_len: last_page_len.id(),
            kv_len: kv_len.id(),
            q_heads,
            kv_heads,
            head_dim,
            window,
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
    kv_heads: u32,
    causal: bool,
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
            kv_heads,
            causal,
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
    ssm_causal_conv1d_dilated(x, weight, state, conv_width, 1)
}

/// Dilated form: tap `j` reads `dilation · j` positions back.
pub fn ssm_causal_conv1d_dilated(
    x: &Value,
    weight: &Weight,
    state: ValueId,
    conv_width: u32,
    dilation: u32,
) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Attention::SsmCausalConv1d {
            x: x.id(),
            weight: r.weight(weight),
            state,
            conv_width,
            dilation,
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
    ssm_causal_conv1d_chunked_dilated(x, weight, state, conv_width, 1)
}

pub fn ssm_causal_conv1d_chunked_dilated(
    x: &Value,
    weight: &Weight,
    state: ValueId,
    conv_width: u32,
    dilation: u32,
) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Attention::SsmCausalConv1dChunked {
            x: x.id(),
            weight: r.weight(weight),
            state,
            conv_width,
            dilation,
            y: y.id(),
        },
        &[x],
    );
    y
}

/// DFlash2's two-tap grouped dynamic convolution along each request's rows
/// (`Attention::BlockDynConv`): `side` 0 convolves a sublayer's input, 1 its
/// output, both with the coefficients `coeff` projected from that input;
/// `base` is the learned `[2·taps, channels]` kernel the projection corrects.
pub fn block_dyn_conv(
    x: &Value,
    coeff: &Value,
    base: &Weight,
    side: u32,
    taps: u32,
    group: u32,
) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Attention::BlockDynConv {
            x: x.id(),
            coeff: coeff.id(),
            base: r.weight(base),
            side,
            taps,
            group,
            y: y.id(),
        },
        &[x, coeff],
    );
    y
}

/// DFlash2's candidate selector, walked from each request's anchor
/// (`Attention::SelectorWalk`): the picked id at every slot row, `[rows, 1]`
/// i32 — a draft readout, planted where [`layout::argmax`](super::layout::argmax)'s
/// would be.
pub fn selector_walk(
    cand: &Value,
    unary: &Value,
    hp: Option<&Value>,
    tokens: &Value,
    pred: &Weight,
    succ: &Weight,
    first: u32,
) -> Value {
    let r = cand.rec();
    let picks = r.fresh(tensor(cand.rows(), 1u64, Dtype::I32));
    let mut deps: Vec<&Value> = vec![cand, unary];
    deps.extend(hp);
    deps.push(tokens);
    r.push(
        Attention::SelectorWalk {
            cand: cand.id(),
            unary: unary.id(),
            hp: hp.map(Value::id),
            tokens: tokens.id(),
            pred: r.weight(pred),
            succ: r.weight(succ),
            first,
            picks: picks.id(),
        },
        &deps,
    );
    picks
}

/// PLE n-gram hasher: `state` is the lane's trailing-token-id window; `mults`,
/// `primes`, `offsets` are seed-derived hash constants. Answer is `[rows, primes.len()]` `i32`.
pub fn ple_ngram_ids(
    ids: &Value,
    state: ValueId,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
) -> Value {
    let r = ids.rec();
    let ngram_ids = r.fresh(tensor(ids.rows(), primes.len() as u64, Dtype::I32));
    r.push(
        Attention::PleNgramIds {
            ids: ids.id(),
            state,
            eos,
            mults: mults.to_vec(),
            primes: primes.to_vec(),
            offsets: offsets.to_vec(),
            heads_per_ngram,
            ngram_ids: ngram_ids.id(),
        },
        &[ids],
    );
    ngram_ids
}

/// Prefill form of [`ple_ngram_ids`]: walks the fire's ambient request
/// boundaries, as the chunked convolution does.
pub fn ple_ngram_ids_chunked(
    ids: &Value,
    state: ValueId,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
) -> Value {
    let r = ids.rec();
    let ngram_ids = r.fresh(tensor(ids.rows(), primes.len() as u64, Dtype::I32));
    r.push(
        Attention::PleNgramIdsChunked {
            ids: ids.id(),
            state,
            eos,
            mults: mults.to_vec(),
            primes: primes.to_vec(),
            offsets: offsets.to_vec(),
            heads_per_ngram,
            ngram_ids: ngram_ids.id(),
        },
        &[ids],
    );
    ngram_ids
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
    gate_floor: f32,
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
            gate_floor,
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
    gate_floor: f32,
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
            gate_floor,
            y: y.id(),
        },
        &[mixed, f, b],
    );
    y
}

/// Builds the one MLA plan (serves both decode and prefill) off `inputs`'
/// geometry and reading. Guard and hoisting discipline as [`plan_decode`];
/// `heads` and `kv_lora_rank` are the absorbed reading the latent launches size against.
pub fn mla_plan<F>(inputs: &Input<F>, heads: u32, kv_lora_rank: u32) -> Value {
    let kv_indptr = inputs.kv_indptr();
    let kv_indices = inputs.kv_indices();
    let last_page_len = inputs.last_page_len();
    let kv_len = inputs.kv_len();
    let r = kv_indptr.rec();
    let plan = r.fresh(Ty::Struct(StructKind::MlaPlan));
    r.push(
        Attention::MlaPlan {
            kv_indptr: kv_indptr.id(),
            kv_indices: kv_indices.id(),
            last_page_len: last_page_len.id(),
            kv_len: kv_len.id(),
            heads,
            kv_lora_rank,
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

/// Trailing pair is `nope_dim, v_head_dim`, same order as [`mla_absorb_q`].
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

/// `ratio` is which cached rows are keys: `1` for a per-token key cache, the
/// compressor's own ratio for a per-block one. Published ids are positions at
/// `ratio == 1`, compressed-row indices otherwise.
#[allow(clippy::too_many_arguments)]
pub fn index_topk(
    q: &Value,
    weights: &Value,
    keys: ValueId,
    heads: u32,
    head_dim: u32,
    top_k: u32,
    ratio: u32,
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
            ratio,
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
/// Returns `(boundary_pos, boundary_req, boundary_rope)`: the cache cell,
/// its lane, and the compressed row's roped position (not the same as the cell).
pub fn pool_boundary_decode(
    positions: &Value,
    row_valid: &Value,
    ratio: u32,
) -> (Value, Value, Value) {
    let r = positions.rec();
    let boundary_pos = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    let boundary_req = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    let boundary_rope = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    r.push(
        Attention::PoolBoundaryDecode {
            positions: positions.id(),
            row_valid: row_valid.id(),
            ratio,
            boundary_pos: boundary_pos.id(),
            boundary_req: boundary_req.id(),
            boundary_rope: boundary_rope.id(),
        },
        &[positions, row_valid],
    );
    (boundary_pos, boundary_req, boundary_rope)
}

/// `row_valid` masks graph-padding rows out of the boundary math.
///
/// The prefill twin of [`pool_boundary_decode`], same three outputs.
pub fn pool_boundary_prefill(
    positions: &Value,
    row_valid: &Value,
    ratio: u32,
) -> (Value, Value, Value) {
    let r = positions.rec();
    let boundary_pos = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    let boundary_req = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    let boundary_rope = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    r.push(
        Attention::PoolBoundaryPrefill {
            positions: positions.id(),
            row_valid: row_valid.id(),
            ratio,
            boundary_pos: boundary_pos.id(),
            boundary_req: boundary_req.id(),
            boundary_rope: boundary_rope.id(),
        },
        &[positions, row_valid],
    );
    (boundary_pos, boundary_req, boundary_rope)
}

/// `dtype` is the pooled entries' element type; stated explicitly since the
/// wrapper has no data input to infer it from.
pub fn pool_gather(
    boundary_pos: &Value,
    boundary_req: &Value,
    pages: ValueId,
    ape: Option<&Weight>,
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
            ape: ape.map(|w| r.weight(w)),
            head_dim,
            ratio,
            entries: entries.id(),
        },
        &[boundary_pos, boundary_req],
    );
    entries
}

/// Compressor's rolling state, written at the source cache's own slot.
/// `kv` is `wkv · x`, `score` is `wgate · x`, both `[tokens, coff · head_dim]`;
/// [`pool_gather`] reads them back at `write_page`/`write_offset`'s cell.
pub fn pool_state_write(
    kv: &Value,
    score: &Value,
    pages: ValueId,
    write_page: &Value,
    write_offset: &Value,
    head_dim: u32,
    ratio: u32,
) {
    let r = kv.rec();
    r.push(
        Attention::PoolStateWrite {
            kv: kv.id(),
            score: score.id(),
            pages,
            write_page: write_page.id(),
            write_offset: write_offset.id(),
            head_dim,
            ratio,
        },
        &[kv, score, write_page, write_offset],
    );
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

/// [`pool_lse`] over the `selection` [`index_topk`] published (NSA fine branch).
/// `top_k` is the selection's own width.
#[allow(clippy::too_many_arguments)]
pub fn pool_lse_selected(
    q: &Value,
    positions: &Value,
    request_of_token: &Value,
    selection: &Value,
    entries: ValueId,
    ratio: u32,
    top_k: u32,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
) -> (Value, Value) {
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    let lse = r.fresh(tensor(q.rows(), heads, Dtype::F32));
    r.push(
        Attention::PoolLseSelected {
            q: q.id(),
            positions: positions.id(),
            request_of_token: request_of_token.id(),
            selection: selection.id(),
            entries,
            ratio,
            top_k,
            heads,
            head_dim,
            sm_scale,
            o: o.id(),
            lse: lse.id(),
        },
        &[q, positions, request_of_token, selection],
    );
    (o, lse)
}

/// Bidirectional attention over the patch window, block-diagonal per image.
/// `segments` is the patch axis's indptr: patch row `n` attends over the rows
/// of the image whose span contains it, both ways, and nothing else.
pub fn dense(
    q: &Value,
    k: &Value,
    v: &Value,
    segments: &Value,
    head_dim: u32,
    sm_scale: f32,
) -> Value {
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    r.push(
        Attention::Dense {
            q: q.id(),
            k: k.id(),
            v: v.id(),
            segments: segments.id(),
            head_dim,
            sm_scale,
            o: o.id(),
        },
        &[q, k, v, segments],
    );
    o
}
