use super::conv::{expect_grid, expect_voxels};
use super::*;
use crate::fact::Predicate;
use crate::forward::Input;
use poem_ir::{Blocks, Relative, VoxelSegment};

pub fn plan_decode(
    inputs: &Input,
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
        Attn::PlanDecode {
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

pub fn plan_prefill(
    inputs: &Input,
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
        Attn::PlanPrefill {
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

/// The blocks a selection reads, `ratio` keys each, where `selection` is
/// given; `per_head`, one run per kv head.
fn blocks(selection: Option<&Value>, ratio: Option<u32>, per_head: bool) -> Option<Blocks> {
    match (selection, ratio) {
        (Some(_), Some(ratio)) => Some(Blocks { ratio, per_head }),
        (None, None) => None,
        _ => panic!("a selection states its block ratio, and only a selection does"),
    }
}

/// A relative bias over `extent` positions, its distances past
/// `2^log_floor` in log buckets `log_alpha` wide where `log_scaling` says.
fn relative(
    bias: Option<&Value>,
    extent: Option<u32>,
    log_scaling: Option<(u32, f32)>,
) -> Option<Relative> {
    match (bias, extent) {
        (Some(_), Some(extent)) => {
            let (log_floor, log_alpha) = log_scaling.unwrap_or((0, 0.0));
            Some(Relative {
                extent,
                log_floor,
                log_alpha,
            })
        }
        (None, None) => None,
        _ => panic!("a relative bias states its extent, and only a bias does"),
    }
}

/// Each row's per-head log-sum-exp beside its output, where asked for.
fn lse_of(q: &Value, head_dim: u32, lse: bool) -> Option<Value> {
    lse.then(|| {
        q.rec().fresh(tensor(
            q.rows(),
            q.width() / u64::from(head_dim),
            Dtype::F32,
        ))
    })
}

/// Each row's query against its lane's paged keys, as one token's: under a
/// selection (`selection`, `ratio` keys a block, `per_head` one run per kv
/// head) only its blocks; under a relative bias (`bias` over `extent`
/// positions, `log_scaling` its log buckets) each key offset by its
/// distance. With `lse`, each row's per-head log-sum-exp beside its output.
#[allow(clippy::too_many_arguments)]
pub fn decode(
    q: &Value,
    plan: &Value,
    pages: ValueId,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    lse: bool,
    selection: Option<&Value>,
    ratio: Option<u32>,
    per_head: bool,
    bias: Option<&Value>,
    extent: Option<u32>,
    log_scaling: Option<(u32, f32)>,
) -> (Value, Option<Value>) {
    let selected = blocks(selection, ratio, per_head);
    let rel = relative(bias, extent, log_scaling);
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    let lse = lse_of(q, head_dim, lse);
    let mut ins = vec![q, plan];
    ins.extend(selection);
    ins.extend(bias);
    r.push(
        Attn::Decode {
            q: q.id(),
            plan: plan.id(),
            cache: pages,
            selection: selection.map(Value::id),
            bias: bias.map(Value::id),
            window,
            head_dim,
            sm_scale,
            selected,
            rel,
            o: o.id(),
            lse: lse.as_ref().map(Value::id),
        },
        &ins,
    );
    (o, lse)
}

/// Each row's query against its lane's paged keys, causal where `causal`, masked by `mask`; no
/// key below `floor`. A selection, a relative bias and `lse` as [`decode`]'s.
#[allow(clippy::too_many_arguments)]
pub fn prefill(
    q: &Value,
    plan: &Value,
    pages: ValueId,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    lse: bool,
    mask: Option<&Value>,
    causal: bool,
    floor: Option<&Value>,
    selection: Option<&Value>,
    ratio: Option<u32>,
    per_head: bool,
    bias: Option<&Value>,
    extent: Option<u32>,
    log_scaling: Option<(u32, f32)>,
) -> (Value, Option<Value>) {
    let selected = blocks(selection, ratio, per_head);
    let rel = relative(bias, extent, log_scaling);
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    let lse = lse_of(q, head_dim, lse);
    let mut ins = vec![q, plan];
    ins.extend(mask);
    ins.extend(selection);
    ins.extend(bias);
    ins.extend(floor);
    r.push(
        Attn::Prefill {
            q: q.id(),
            plan: plan.id(),
            cache: pages,
            mask: mask.map(Value::id),
            selection: selection.map(Value::id),
            bias: bias.map(Value::id),
            floor: floor.map(Value::id),
            window,
            head_dim,
            kv_heads,
            causal,
            sm_scale,
            selected,
            rel,
            o: o.id(),
            lse: lse.as_ref().map(Value::id),
        },
        &ins,
    );
    (o, lse)
}

/// Bounded replay's floor (`attn.replay_floor`): with `mark` the rows
/// of `positions` are ones a replaying layer skips, and their lane's state
/// takes the position past them; without, each row's floor from its lane's.
pub fn replay_floor(positions: &Value, state: ValueId, mark: bool) -> Value {
    let r = positions.rec();
    let floor = r.fresh(positions.ty().clone());
    r.push(
        Attn::ReplayFloor {
            positions: positions.id(),
            state,
            mark,
            floor: floor.id(),
        },
        &[positions],
    );
    floor
}

pub fn sink(o: &Value, lse: &Value, sink: &Weight, head_dim: u32) -> Value {
    let r = o.rec();
    let o_out = r.fresh(o.ty().clone());
    r.push(
        Attn::Sink {
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
        Attn::MergeLse {
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
        poem_ir::Elemwise::Softcap {
            x: x.id(),
            cap,
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

/// Each row's key and value written to `pages` at its page and offset;
/// without `v`, `k` is one plane the pool reads as both.
pub fn kv_append(
    k: &Value,
    v: Option<&Value>,
    pages: ValueId,
    write_page: &Value,
    write_offset: &Value,
) {
    let r = k.rec();
    let mut ins = vec![k];
    ins.extend(v);
    ins.extend([write_page, write_offset]);
    r.push(
        Attn::KvAppend {
            k: k.id(),
            v: v.map(Value::id),
            cache: pages,
            write_page: write_page.id(),
            write_offset: write_offset.id(),
        },
        &ins,
    );
}

/// Each row's lane's last key position in `pages` (`attn.kv_ends`).
pub fn kv_ends(request_of_token: &Value, pages: ValueId) -> Value {
    let r = request_of_token.rec();
    let ends = r.fresh(tensor(request_of_token.rows(), 1u64, Dtype::I32));
    r.push(
        Attn::KvEnds {
            request_of_token: request_of_token.id(),
            cache: pages,
            ends: ends.id(),
        },
        &[request_of_token],
    );
    ends
}

/// DeepSeek's window attention with its sinks, each row reading the
/// `window` positions that end at its `positions` (`attn.window_sink`).
#[allow(clippy::too_many_arguments)]
pub fn window_sink(
    q: &Value,
    positions: &Value,
    request_of_token: &Value,
    pages: ValueId,
    sink: &Weight,
    window: u32,
    head_dim: u32,
    sm_scale: f32,
) -> Value {
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    r.push(
        Attn::WindowSink {
            q: q.id(),
            positions: positions.id(),
            request_of_token: request_of_token.id(),
            cache: pages,
            sink: r.weight(sink),
            window,
            head_dim,
            sm_scale,
            o: o.id(),
        },
        &[q, positions, request_of_token],
    );
    o
}

pub fn mla_plan(inputs: &Input, heads: u32, kv_lora_rank: u32) -> Value {
    let kv_indptr = inputs.kv_indptr();
    let kv_indices = inputs.kv_indices();
    let last_page_len = inputs.last_page_len();
    let kv_len = inputs.kv_len();
    let r = kv_indptr.rec();
    let plan = r.fresh(Ty::Struct(StructKind::MlaPlan));
    r.push(
        Attn::MlaPlan {
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

/// The rope part of a `kv_a` row past its `kv_lora_rank` latent.
fn rope_width(kv_a: &Value, kv_lora_rank: u32) -> u64 {
    kv_a.width()
        .checked_sub(u64::from(kv_lora_rank))
        .unwrap_or_else(|| {
            panic!(
                "a {}-wide kv_a row holds no {kv_lora_rank}-wide latent",
                kv_a.width()
            )
        })
}

pub fn mla_latents(kv_a: &Value, norm: &Weight, eps: f32, kv_lora_rank: u32) -> (Value, Value) {
    let r = kv_a.rec();
    let kv_c = r.fresh(tensor(kv_a.rows(), kv_lora_rank, kv_a.dtype()));
    let k_pe = r.fresh(tensor(
        kv_a.rows(),
        rope_width(kv_a, kv_lora_rank),
        kv_a.dtype(),
    ));
    r.push(
        Attn::MlaLatents {
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
        rope_width(kv_a, kv_lora_rank),
        kv_a.dtype(),
    ));
    r.push(
        Attn::MlaLatentsRope {
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
        Attn::MlaSplitQB {
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
        Attn::MlaAbsorbQ {
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
        Attn::MlaAbsorbOut {
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
        Attn::MlaKvAppend {
            kv_c: kv_c.id(),
            k_pe: k_pe.id(),
            cache: pages,
            write_page: write_page.id(),
            write_offset: write_offset.id(),
        },
        &[kv_c, k_pe, write_page, write_offset],
    );
}

/// The scores of a latent attention, and with `with_lse` the base-2
/// log-sum-exp of each row's heads beside them.
#[allow(clippy::too_many_arguments)]
fn mla_scored(
    q: &Value,
    plan: &Value,
    q_pe: &Value,
    selection: Option<&Value>,
    pages: ValueId,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    decode: bool,
    with_lse: bool,
    window: Option<u32>,
) -> (Value, Option<Value>) {
    let r = q.rec();
    let width = u64::from(heads) * u64::from(kv_lora_rank);
    let o = r.fresh(tensor(q.rows(), width, q.dtype()));
    let lse = with_lse.then(|| r.fresh(tensor(q.rows(), heads, Dtype::F32)));
    let (q_id, plan_id, q_pe_id, o_id) = (q.id(), plan.id(), q_pe.id(), o.id());
    let lse_id = lse.as_ref().map(Value::id);
    let op = match (selection, decode) {
        (None, true) => Attn::MlaDecode {
            q: q_id,
            plan: plan_id,
            q_pe: q_pe_id,
            cache: pages,
            window,
            heads,
            kv_lora_rank,
            sm_scale,
            o: o_id,
            lse: lse_id,
            selection: None,
        },
        (None, false) => Attn::MlaPrefill {
            q: q_id,
            plan: plan_id,
            q_pe: q_pe_id,
            cache: pages,
            window,
            heads,
            kv_lora_rank,
            sm_scale,
            o: o_id,
            lse: lse_id,
            selection: None,
        },
        (Some(selection), true) => Attn::MlaDecode {
            q: q_id,
            plan: plan_id,
            q_pe: q_pe_id,
            selection: Some(selection.id()),
            cache: pages,
            heads,
            kv_lora_rank,
            sm_scale,
            o: o_id,
            lse: lse_id,
            window: None,
        },
        (Some(selection), false) => Attn::MlaPrefill {
            q: q_id,
            plan: plan_id,
            q_pe: q_pe_id,
            selection: Some(selection.id()),
            cache: pages,
            heads,
            kv_lora_rank,
            sm_scale,
            o: o_id,
            lse: lse_id,
            window: None,
        },
    };
    let mut deps = vec![q, plan, q_pe];
    deps.extend(selection);
    r.push(op, &deps);
    (o, lse)
}

/// One token's latent attention a row: its query (`q` absorbed, `q_pe` its
/// rotated part) against its lane's latent rows, or those `selection` picks;
/// with `lse`, each row's per-head log-sum-exp beside its output.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode(
    q: &Value,
    plan: &Value,
    q_pe: &Value,
    pages: ValueId,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    window: Option<u32>,
    lse: bool,
    selection: Option<&Value>,
) -> (Value, Option<Value>) {
    mla_scored(
        q,
        plan,
        q_pe,
        selection,
        pages,
        heads,
        kv_lora_rank,
        sm_scale,
        true,
        lse,
        window,
    )
}

/// A prompt's latent attention, as [`mla_decode`]'s causally over its rows.
#[allow(clippy::too_many_arguments)]
pub fn mla_prefill(
    q: &Value,
    plan: &Value,
    q_pe: &Value,
    pages: ValueId,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    window: Option<u32>,
    lse: bool,
    selection: Option<&Value>,
) -> (Value, Option<Value>) {
    mla_scored(
        q,
        plan,
        q_pe,
        selection,
        pages,
        heads,
        kv_lora_rank,
        sm_scale,
        false,
        lse,
        window,
    )
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
        Attn::IndexLayernormRope {
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
        Attn::IndexRope {
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

#[allow(clippy::too_many_arguments)]
pub fn index_topk(
    q: &Value,
    weights: Option<&Value>,
    keys: ValueId,
    heads: u32,
    head_dim: u32,
    top_k: u32,
    ratio: u32,
    score: IndexScore,
) -> Value {
    let r = q.rec();
    // A block-max ranking selects for each head on its own.
    let width = match score {
        IndexScore::Pooled => top_k,
        IndexScore::BlockMax => heads * top_k,
        IndexScore::PooledTokens => top_k * ratio.max(1) + ratio.max(1) - 1,
    };
    let selection = r.fresh(tensor(q.rows(), width, Dtype::I32));
    let mut deps = vec![q];
    deps.extend(weights);
    r.push(
        Attn::IndexTopk {
            q: q.id(),
            weights: weights.map(Value::id),
            keys,
            heads,
            head_dim,
            top_k,
            ratio,
            score,
            selection: selection.id(),
        },
        &deps,
    );
    selection
}

pub fn index_block_mean(
    boundary_pos: &Value,
    boundary_req: &Value,
    keys: ValueId,
    head_dim: u32,
    ratio: u32,
    dtype: Dtype,
) -> Value {
    let r = boundary_pos.rec();
    let entries = r.fresh(tensor(boundary_pos.rows(), head_dim, dtype));
    r.push(
        Attn::IndexBlockMean {
            boundary_pos: boundary_pos.id(),
            boundary_req: boundary_req.id(),
            keys,
            head_dim,
            ratio,
            entries: entries.id(),
        },
        &[boundary_pos, boundary_req],
    );
    entries
}

pub fn index_kv_append(k: &Value, keys: ValueId, write_page: &Value, write_offset: &Value) {
    let r = k.rec();
    r.push(
        Attn::IndexKvAppend {
            k: k.id(),
            keys,
            write_page: write_page.id(),
            write_offset: write_offset.id(),
        },
        &[k, write_page, write_offset],
    );
}

/// Where the rows close entries of a pool compressing `ratio` positions
/// each: one token's rows (`decode`) or a prompt's.
pub fn pool_boundary(
    positions: &Value,
    row_valid: &Value,
    ratio: u32,
    decode: bool,
) -> (Value, Value, Value) {
    let r = positions.rec();
    let boundary_pos = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    let boundary_req = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    let boundary_rope = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
    r.push(
        Attn::PoolBoundary {
            positions: positions.id(),
            row_valid: row_valid.id(),
            ratio,
            decode,
            boundary_pos: boundary_pos.id(),
            boundary_req: boundary_req.id(),
            boundary_rope: boundary_rope.id(),
        },
        &[positions, row_valid],
    );
    (boundary_pos, boundary_req, boundary_rope)
}

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
        Attn::PoolGather {
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
        Attn::PoolStateWrite {
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
        Attn::PoolKvAppend {
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

/// Each row's query against the closed entries of its lane's compressed
/// pool, or the `top_k` of them `selection` picks; its output and per-head
/// log-sum-exp.
#[allow(clippy::too_many_arguments)]
pub fn pool_lse(
    q: &Value,
    positions: &Value,
    request_of_token: &Value,
    entries: ValueId,
    ratio: u32,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
    selection: Option<&Value>,
    top_k: Option<u32>,
) -> (Value, Value) {
    assert_eq!(
        selection.is_some(),
        top_k.is_some(),
        "a selection states how many entries it picks, and only a selection does"
    );
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    let lse = r.fresh(tensor(q.rows(), heads, Dtype::F32));
    let mut ins = vec![q, positions, request_of_token];
    ins.extend(selection);
    r.push(
        Attn::PoolLse {
            q: q.id(),
            positions: positions.id(),
            request_of_token: request_of_token.id(),
            selection: selection.map(Value::id),
            entries,
            ratio,
            top_k,
            heads,
            head_dim,
            sm_scale,
            o: o.id(),
            lse: lse.id(),
        },
        &ins,
    );
    (o, lse)
}

pub fn ragged(
    q: &Value,
    k: &Value,
    v: &Value,
    q_indptr: &Value,
    kv_indptr: &Value,
    head_dim: u32,
    sm_scale: f32,
    mask: RaggedMask,
) -> Value {
    let r = q.rec();
    assert!(
        head_dim > 0 && q.width().is_multiple_of(u64::from(head_dim)),
        "q is {} wide, not a whole number of {head_dim}-wide heads",
        q.width()
    );
    assert!(
        k.width().is_multiple_of(u64::from(head_dim)) && k.width() == v.width(),
        "k ({}) and v ({}) are one shape of {head_dim}-wide heads",
        k.width(),
        v.width()
    );
    let heads = q.width() / u64::from(head_dim);
    let kv_heads = k.width() / u64::from(head_dim);
    assert!(
        heads.is_multiple_of(kv_heads),
        "{heads} query heads do not group over {kv_heads} kv heads"
    );
    assert_eq!(k.rows(), v.rows(), "k and v share a row space");
    let o = r.fresh(q.ty().clone());
    r.push(
        Attn::Ragged {
            q: q.id(),
            k: k.id(),
            v: v.id(),
            q_indptr: q_indptr.id(),
            kv_indptr: kv_indptr.id(),
            head_dim,
            kv_heads: kv_heads as u32,
            sm_scale,
            mask,
            o: o.id(),
        },
        &[q, k, v, q_indptr, kv_indptr],
    );
    o.under(q.cond())
}

#[must_use]
pub fn relative_bias(table: &Value, max_len: u32) -> RaggedMask {
    assert!(
        max_len > 0,
        "a relative bias over no positions has no width"
    );
    assert_eq!(
        table.dtype(),
        Dtype::F32,
        "the relative bias table is f32, not {:?}",
        table.dtype()
    );
    assert_eq!(
        table.width(),
        2 * u64::from(max_len) - 1,
        "the relative bias table is {} wide and max_len {max_len} reads 2 · max_len − 1",
        table.width()
    );
    assert!(
        matches!(table.rows(), Dim::Const(_)),
        "the relative bias table is a plan constant, `[heads, 2·max_len − 1]`, not {:?} rows",
        table.rows()
    );
    RaggedMask::RelativeBias {
        table: table.id(),
        max_len,
    }
}

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
        Attn::Dense {
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

#[must_use]
pub fn voxel(q: &Value, k: &Value, v: &Value, grid: &Value, sm_scale: f32) -> Value {
    voxel_over(q, k, v, grid, VoxelSegment::Clip, sm_scale)
}

#[must_use]
pub fn voxel_over(
    q: &Value,
    k: &Value,
    v: &Value,
    grid: &Value,
    segment: VoxelSegment,
    sm_scale: f32,
) -> Value {
    expect_voxels("`attn::voxel`'s query", q);
    expect_grid("`attn::voxel`'s grid", grid);
    assert!(
        q.ty() == k.ty() && q.ty() == v.ty(),
        "`attn::voxel` reads q, k and v at one type; got {:?}, {:?}, {:?}",
        q.ty(),
        k.ty(),
        v.ty()
    );
    assert_eq!(q.dtype(), Dtype::Bf16, "`attn::voxel` reads bf16 rows");
    assert!(
        segment != VoxelSegment::Frames(0),
        "`attn::voxel` over `Frames(0)` is a block with no rows in it"
    );
    let r = q.rec();
    let o = r.fresh(q.ty().clone());
    r.push(
        Attn::Voxel {
            q: q.id(),
            k: k.id(),
            v: v.id(),
            grid: grid.id(),
            segment,
            sm_scale,
            o: o.id(),
        },
        &[q, k, v, grid],
    );
    o
}

/// Attention of `q` over the kv row `kv`, each lane by its kind: a lane
/// `causal` does not hold for over every key it holds, a lane with its own
/// mask by that mask (when the model opens `mask`), a lane querying one
/// token by decode, and the rest by prefill. The kinds are taken in that order, each
/// lane by the first that holds. With `sinks`, every kind is attended with
/// its log-sum-exp and the sink logits folded in. With a `selection` (the
/// key blocks an indexer picked a row, `ratio` keys a block, one pick a head
/// when `per_head`), decode and prefill lanes read only those blocks; a
/// masked lane reads every key.
///
/// The host plans are the compiler's: each kind's is built once a trace,
/// over the lanes of that kind and outside any layer, and every layer asking
/// for the same one reads it.
#[allow(clippy::too_many_arguments)]
pub fn attend(
    q: &Value,
    inputs: &Input,
    kv: &str,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    sm_scale: f32,
    window: Option<u32>,
    sinks: Option<&Weight>,
    causal: Option<&Predicate>,
    mask: bool,
    selection: Option<&Value>,
    ratio: Option<u32>,
    per_head: bool,
) -> Value {
    let selected = selection.map(|selection| {
        assert!(
            sinks.is_none(),
            "`attn.attend` attends a selection without sinks; no kernel folds both"
        );
        let ratio = ratio.expect("`attn.attend` with a `selection` names its `ratio`");
        (selection, ratio)
    });
    use crate::fact;
    use crate::record::{FirstMatch, Refine};

    let wide = causal.map(|causal| !causal.clone());
    let pages = inputs.kv(kv);
    let rec = inputs.recorder();
    let plan = |decode: bool, lanes: &Input| {
        let kind = if decode { "decode" } else { "prefill" };
        let key = format!(
            "attn.plan_{kind} {:?} {q_heads} {kv_heads} {head_dim} {window:?}",
            lanes.rows()
        );
        rec.once(key, || {
            if decode {
                plan_decode(lanes, q_heads, kv_heads, head_dim, window)
            } else {
                plan_prefill(lanes, q_heads, kv_heads, head_dim, window)
            }
        })
    };
    // The lanes of the kind `kind` names, or of none of them.
    let lanes = |kind: Option<fact::Predicate>| {
        let mut first = FirstMatch::default();
        let mut cases = Vec::new();
        cases.extend(wide.clone());
        if mask {
            cases.push(fact::has(fact::Mask));
        }
        cases.push(fact::single_token());
        let mut picked = None;
        for case in &cases {
            let rows = first.case(rec.guard_of(case));
            if kind.as_ref() == Some(case) {
                picked = Some(rows);
            }
        }
        inputs.refined(picked.unwrap_or_else(|| first.rest()))
    };
    // An arm's output, the sinks folded in when the model has them.
    let sunk = |(o, lse): (Value, Value)| match sinks {
        Some(sinks) => sink(&o, &lse, sinks, head_dim),
        None => o,
    };

    // Each kind's rows of `q`: the first kind each lane is, within the rows
    // `q` is over. A kind no lane of `q` can be is not attended (`q` may
    // already be the lanes of some kinds alone). Every arm keeps `q`'s rows
    // spelled in its guard, so the merge reads as over them.
    let domain = q.cond();
    let rows_of = |rows: poem_ir::Guard| {
        let rows = poem_ir::Guard::and(domain.clone(), rows);
        possible(q, rows.clone()).then(|| q.refined(rows))
    };
    // A selection's rows of a kind, as `q`'s are.
    let picks = |rows: &poem_ir::Guard| {
        selected.map(|(selection, ratio)| {
            let over = poem_ir::Guard::and(selection.cond(), rows.clone());
            (selection.refined(over), ratio)
        })
    };
    let mut first = FirstMatch::default();
    let mut arms = Vec::new();
    if let Some(wide) = &wide
        && let Some(q) = rows_of(first.case(rec.guard_of(wide)))
    {
        let p = plan(false, &lanes(Some(wide.clone())));
        let (o, lse) = prefill(
            &q,
            &p,
            pages,
            window,
            head_dim,
            kv_heads,
            sm_scale,
            sinks.is_some(),
            None,
            false,
            None,
            None,
            None,
            false,
            None,
            None,
            None,
        );
        arms.push(match lse {
            Some(lse) => sunk((o, lse)),
            None => o,
        });
    }
    if mask && let Some(q) = rows_of(first.case(rec.guard_of(&fact::has(fact::Mask)))) {
        let masks = inputs.whole_mask();
        let p = plan(false, &lanes(Some(fact::has(fact::Mask))));
        let (o, lse) = prefill(
            &q,
            &p,
            pages,
            window,
            head_dim,
            kv_heads,
            sm_scale,
            sinks.is_some(),
            Some(&masks),
            true,
            None,
            None,
            None,
            false,
            None,
            None,
            None,
        );
        arms.push(match lse {
            Some(lse) => sunk((o, lse)),
            None => o,
        });
    }
    let single = first.case(rec.guard_of(&fact::single_token()));
    if let Some(q) = rows_of(single.clone()) {
        let p = plan(true, &lanes(Some(fact::single_token())));
        // A selection's blocks hold no sink; a dense arm's sinks fold in.
        let picked = picks(&single);
        let (sel, ratio) = picked.map_or((None, None), |(sel, ratio)| (Some(sel), Some(ratio)));
        let lse = sinks.is_some() && sel.is_none();
        let (o, lse) = decode(
            &q,
            &p,
            pages,
            window,
            head_dim,
            sm_scale,
            lse,
            sel.as_ref(),
            ratio,
            per_head && sel.is_some(),
            None,
            None,
            None,
        );
        arms.push(match lse {
            Some(lse) => sunk((o, lse)),
            None => o,
        });
    }
    let rest = first.rest();
    if let Some(q) = rows_of(rest.clone()) {
        let p = plan(false, &lanes(None));
        let picked = picks(&rest);
        let (sel, ratio) = picked.map_or((None, None), |(sel, ratio)| (Some(sel), Some(ratio)));
        let lse = sinks.is_some() && sel.is_none();
        let (o, lse) = prefill(
            &q,
            &p,
            pages,
            window,
            head_dim,
            kv_heads,
            sm_scale,
            lse,
            None,
            true,
            None,
            sel.as_ref(),
            ratio,
            per_head && sel.is_some(),
            None,
            None,
            None,
        );
        arms.push(match lse {
            Some(lse) => sunk((o, lse)),
            None => o,
        });
    }
    if arms.len() == 1 {
        return arms.remove(0);
    }
    Value::merge(arms)
}

/// The attention mass of each lane capturing scores (`fact::scores`) over
/// its first `kv_max` keys of `kv`, averaged over the lane's last `observe`
/// query rows of `q`: `[lanes, q_heads * kv_max]`, a `kv_max` run per query
/// head.
#[allow(clippy::too_many_arguments)]
pub fn score_capture(
    q: &Value,
    inputs: &Input,
    kv: &str,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    sm_scale: f32,
    observe: u32,
    kv_max: u32,
) -> Value {
    let rec = inputs.recorder();
    let rows = poem_ir::Guard::and(q.cond(), rec.guard_of(&crate::fact::scores()));
    let scoring = crate::record::Refine::refined(inputs, rows);
    let key = format!(
        "attn.plan_prefill {:?} {q_heads} {kv_heads} {head_dim} None",
        scoring.rows()
    );
    let plan = rec.once(key, || {
        plan_prefill(&scoring, q_heads, kv_heads, head_dim, None)
    });
    let q = crate::record::Refine::refined(q, scoring.rows().clone());
    let scores = rec.fresh(Ty::Tensor {
        shape: vec![
            Dim::Lanes,
            Dim::Const(u64::from(q_heads) * u64::from(kv_max)),
        ],
        dtype: Dtype::F32,
    });
    rec.push(
        Attn::ScoreCapture {
            q: q.id(),
            plan: plan.id(),
            cache: inputs.kv(kv),
            head_dim,
            kv_heads,
            sm_scale,
            observe,
            kv_max,
            scores: scores.id(),
        },
        &[&q, &plan],
    );
    scores
}

/// Whether any row of `x` can be one `rows` holds for.
fn possible(x: &Value, rows: poem_ir::Guard) -> bool {
    use poem_ir::Guard;
    !Guard::and(x.cond(), rows).implies(&!Guard::Always)
}

/// Latent attention of the absorbed queries `q` (and their rotated part
/// `q_pe`) over the latent kv row `kv`: a lane querying one token by decode,
/// the rest by prefill; with `whole`, every lane by prefill (rows that read
/// their keys whole, as a draft head's). With a `selection`, each lane reads
/// only the keys its indexer picked; `window` bounds what a lane without one
/// reads. With `sinks`, the arms' log-sum-exps are merged and the sink
/// logits folded in once. What comes back is still in the latent space, for
/// the model to absorb out.
///
/// The host plans are the compiler's, as [`attend`]'s are.
#[allow(clippy::too_many_arguments)]
pub fn attend_mla(
    q: &Value,
    q_pe: &Value,
    inputs: &Input,
    kv: &str,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    window: Option<u32>,
    selection: Option<&Value>,
    sinks: Option<&Weight>,
    whole: bool,
) -> Value {
    use crate::fact;
    use crate::record::Refine;
    use poem_ir::Guard;

    let pages = inputs.kv(kv);
    let rec = inputs.recorder();
    let plan = |lanes: &Input| {
        let key = format!("attn.mla_plan {:?} {heads} {kv_lora_rank}", lanes.rows());
        rec.once(key, || mla_plan(lanes, heads, kv_lora_rank))
    };
    // The arm over `rows` of the lanes (and the plan) `lanes` are.
    let arm = |decode: bool, rows: Guard, lanes: &Input| {
        let within = |x: &Value| x.refined(Guard::and(x.cond(), rows.clone()));
        let (q, q_pe) = (within(q), within(q_pe));
        let selection = selection.map(within);
        let p = plan(lanes);
        assert!(
            selection.is_some() || sinks.is_none(),
            "`attn.attend_mla` folds sinks into a selection's scores; no dense kernel returns \
             the log-sum-exp they need"
        );
        // A selection's rows are its picks, whatever the window.
        let window = if selection.is_some() { None } else { window };
        let score = if decode { mla_decode } else { mla_prefill };
        score(
            &q,
            &p,
            &q_pe,
            pages,
            heads,
            kv_lora_rank,
            sm_scale,
            window,
            sinks.is_some(),
            selection.as_ref(),
        )
    };
    let arms = if whole {
        vec![arm(false, Guard::Always, inputs)]
    } else {
        let one = rec.guard_of(&fact::single_token());
        let (decode, prefill) = (one.clone(), !one);
        vec![
            arm(true, decode.clone(), &inputs.refined(decode)),
            arm(false, prefill.clone(), &inputs.refined(prefill)),
        ]
    };
    let join = |mut parts: Vec<Value>| {
        if parts.len() == 1 {
            parts.remove(0)
        } else {
            Value::merge(parts)
        }
    };
    let (outs, lses): (Vec<Value>, Vec<Option<Value>>) = arms.into_iter().unzip();
    let scored = join(outs);
    match sinks {
        Some(sinks) => {
            let lse = join(
                lses.into_iter()
                    .map(|l| l.expect("an lse a sunk arm"))
                    .collect(),
            );
            sink(&scored, &lse, sinks, kv_lora_rank)
        }
        None => scored,
    }
}
