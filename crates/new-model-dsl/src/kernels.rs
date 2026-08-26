//! Kernel wrappers: the model author's surface over the op enums.
//!
//! Every wrapper computes its output shape in plain Rust (design §4), declares
//! the outputs with `Recorder::fresh`, pushes the typed op, and returns the
//! fresh [`Value`]s — the model author never writes a shape. In-place kernels
//! construct the SSA pair: the enum's `*_out` field names a fresh id and the
//! compiler folds the pair onto one arena slot, so the wrapper still returns a
//! fresh `Value` (§2). Raggedness is ambient (§5): prefill/chunked wrappers
//! take the fire-aligned tensor directly, with no indptr plumbing. Caches are
//! storage-only ids; geometry enters the graph through [`geometry`] as
//! declared runtime inputs, and plan ops are pure functions of them (§6, §7).

use crate::declare::{self, Weight};
use crate::record::{Recorder, Value};
use new_model_ir::{
    Attention, Cuda, Dim, Dist, Dtype, Gate, Gemm, GeomKind, Hc, Index, Layout, Mla, Mlp, Moe,
    Norm, Pool, Rope, RuntimeInput, Ssm, StructKind, Ty, ValueId,
};

/// A two-axis tensor type: the whole surviving shape algebra is `[rows, width]`.
fn tensor(rows: Dim, width: impl Into<u64>, dtype: Dtype) -> Ty {
    Ty::Tensor {
        shape: vec![rows, Dim::Const(width.into())],
        dtype,
    }
}

/// Declares one geometry vector of cache space `cache` as a runtime input
/// (§7): indptr-shaped vectors are `lanes + 1` long, the rest are per-lane.
/// The plan wrappers fetch their own; forwards call this for the write
/// geometry a `kv_append` takes.
pub fn geometry(r: &Recorder, cache: u32, kind: GeomKind) -> Value {
    let rows = match kind {
        GeomKind::Indptr => Dim::LanesPlus(1),
        GeomKind::Indices | GeomKind::SeqLens | GeomKind::LastPageLen => Dim::Lanes,
    };
    r.input(
        RuntimeInput::Geometry { cache, kind },
        Ty::Tensor {
            shape: vec![rows],
            dtype: Dtype::I32,
        },
    )
}

pub mod gemm {
    use super::*;

    pub fn matmul(act: &Value, w: &Weight) -> Value {
        let r = act.rec();
        let y = r.fresh(tensor(act.rows(), w.dim(0), act.dtype()));
        r.push(
            Gemm::Matmul {
                act: act.id(),
                w: r.weight(w),
                y: y.id(),
            },
            &[act],
        );
        y
    }

    pub fn lm_head(act: &Value, w: &Weight) -> Value {
        let r = act.rec();
        let y = r.fresh(tensor(act.rows(), w.dim(0), act.dtype()));
        r.push(
            Gemm::LmHead {
                act: act.id(),
                w: r.weight(w),
                y: y.id(),
            },
            &[act],
        );
        y
    }

    pub fn attention_landing(act: &Value, w: &Weight, layer: u32) -> Value {
        let r = act.rec();
        let y = r.fresh(tensor(act.rows(), w.dim(0), act.dtype()));
        r.push(
            Gemm::AttentionLanding {
                act: act.id(),
                w: r.weight(w),
                layer,
                y: y.id(),
            },
            &[act],
        );
        y
    }
}

pub mod dist {
    use super::*;

    pub fn all_reduce(buf: &Value) -> Value {
        let r = buf.rec();
        let buf_out = r.fresh(buf.ty().clone());
        r.push(
            Dist::AllReduce {
                buf: buf.id(),
                buf_out: buf_out.id(),
            },
            &[buf],
        );
        buf_out
    }

    /// Concatenates each rank's `width`-shard into the full tensor.
    pub fn all_gather(x: &Value, world: u32) -> Value {
        let r = x.rec();
        let y = r.fresh(tensor(x.rows(), x.width() * u64::from(world), x.dtype()));
        r.push(
            Dist::AllGather {
                x: x.id(),
                y: y.id(),
            },
            &[x],
        );
        y
    }

    /// Sums across ranks, leaving each rank its `width`-shard of the result.
    pub fn reduce_scatter(x: &Value, world: u32) -> Value {
        let world = u64::from(world);
        assert!(
            x.width().is_multiple_of(world),
            "a width of {} does not scatter {world} ways",
            x.width(),
        );
        let r = x.rec();
        let y = r.fresh(tensor(x.rows(), x.width() / world, x.dtype()));
        r.push(
            Dist::ReduceScatter {
                x: x.id(),
                y: y.id(),
            },
            &[x],
        );
        y
    }
}

pub mod norm {
    use super::*;

    pub fn rmsnorm(x: &Value, weight: &Weight, eps: f32) -> Value {
        let r = x.rec();
        let y = r.fresh(x.ty().clone());
        r.push(
            Norm::Rmsnorm {
                x: x.id(),
                weight: r.weight(weight),
                eps,
                y: y.id(),
            },
            &[x],
        );
        y
    }

    pub fn rmsnorm_per_head(x: &Value, weight: &Weight, head_dim: u32, eps: f32) -> Value {
        let r = x.rec();
        let y = r.fresh(x.ty().clone());
        r.push(
            Norm::RmsnormPerHead {
                x: x.id(),
                weight: r.weight(weight),
                head_dim,
                eps,
                y: y.id(),
            },
            &[x],
        );
        y
    }

    pub fn rmsnorm_plus_one(x: &Value, weight: &Weight, eps: f32) -> Value {
        let r = x.rec();
        let y = r.fresh(x.ty().clone());
        r.push(
            Norm::RmsnormPlusOne {
                x: x.id(),
                weight: r.weight(weight),
                eps,
                y: y.id(),
            },
            &[x],
        );
        y
    }

    pub fn rmsnorm_per_head_plus_one(x: &Value, weight: &Weight, head_dim: u32, eps: f32) -> Value {
        let r = x.rec();
        let y = r.fresh(x.ty().clone());
        r.push(
            Norm::RmsnormPerHeadPlusOne {
                x: x.id(),
                weight: r.weight(weight),
                head_dim,
                eps,
                y: y.id(),
            },
            &[x],
        );
        y
    }

    pub fn rmsnorm_no_scale(x: &Value, head_dim: u32, eps: f32) -> Value {
        let r = x.rec();
        let y = r.fresh(x.ty().clone());
        r.push(
            Norm::RmsnormNoScale {
                x: x.id(),
                head_dim,
                eps,
                y: y.id(),
            },
            &[x],
        );
        y
    }

    pub fn rmsnorm_gated(x: &Value, gate: &Value, weight: &Weight, head_dim: u32, eps: f32) -> Value {
        let r = x.rec();
        let y = r.fresh(gate.ty().clone());
        r.push(
            Norm::RmsnormGated {
                x: x.id(),
                gate: gate.id(),
                weight: r.weight(weight),
                head_dim,
                eps,
                y: y.id(),
            },
            &[x, gate],
        );
        y
    }

    pub fn rmsnorm_gated_by(x: &Value, gate: &Value, weight: &Weight, heads: u32, eps: f32) -> Value {
        let r = x.rec();
        let y = r.fresh(gate.ty().clone());
        r.push(
            Norm::RmsnormGatedBy {
                x: x.id(),
                gate: gate.id(),
                weight: r.weight(weight),
                heads,
                eps,
                y: y.id(),
            },
            &[x, gate],
        );
        y
    }

    pub fn residual_add(x: &Value, y: &Value) -> Value {
        let r = x.rec();
        let y_out = r.fresh(y.ty().clone());
        r.push(
            Norm::ResidualAdd {
                x: x.id(),
                y: y.id(),
                y_out: y_out.id(),
            },
            &[x, y],
        );
        y_out
    }

    pub fn add_bias(bias: &Weight, out: &Value) -> Value {
        let r = out.rec();
        let out_out = r.fresh(out.ty().clone());
        r.push(
            Norm::AddBias {
                bias: r.weight(bias),
                out: out.id(),
                out_out: out_out.id(),
            },
            &[out],
        );
        out_out
    }

    pub fn mul_scalar(s: f32, x: &Value) -> Value {
        let r = x.rec();
        let x_out = r.fresh(x.ty().clone());
        r.push(
            Norm::MulScalar {
                s,
                x: x.id(),
                x_out: x_out.id(),
            },
            &[x],
        );
        x_out
    }

    pub fn scale(s: &Weight, x: &Value) -> Value {
        let r = x.rec();
        let x_out = r.fresh(x.ty().clone());
        r.push(
            Norm::Scale {
                s: r.weight(s),
                x: x.id(),
                x_out: x_out.id(),
            },
            &[x],
        );
        x_out
    }

    pub fn res_blend(
        prefix: &Value,
        blocks: &[Value],
        norm: &declare::Norm,
        proj: &Weight,
    ) -> Value {
        let r = prefix.rec();
        let y = r.fresh(prefix.ty().clone());
        let mut ins: Vec<&Value> = Vec::with_capacity(1 + blocks.len());
        ins.push(prefix);
        ins.extend(blocks);
        r.push(
            Norm::ResBlend {
                prefix: prefix.id(),
                blocks: blocks.iter().map(Value::id).collect(),
                weight: r.weight(&norm.weight),
                eps: norm.eps,
                proj: r.weight(proj),
                y: y.id(),
            },
            &ins,
        );
        y
    }
}

pub mod mlp {
    use super::*;

    pub fn swiglu(packed: &Value, intermediate: u32) -> Value {
        let r = packed.rec();
        let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
        r.push(
            Mlp::Swiglu {
                packed: packed.id(),
                intermediate,
                y: y.id(),
            },
            &[packed],
        );
        y
    }

    pub fn swiglu_clamp(packed: &Value, intermediate: u32, limit: f32) -> Value {
        let r = packed.rec();
        let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
        r.push(
            Mlp::SwigluClamp {
                packed: packed.id(),
                intermediate,
                limit,
                y: y.id(),
            },
            &[packed],
        );
        y
    }

    pub fn swiglu_clamp_alpha(packed: &Value, intermediate: u32, limit: f32, alpha: f32) -> Value {
        let r = packed.rec();
        let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
        r.push(
            Mlp::SwigluClampAlpha {
                packed: packed.id(),
                intermediate,
                limit,
                alpha,
                y: y.id(),
            },
            &[packed],
        );
        y
    }

    pub fn geglu_tanh(gate: &Value, up: &Value) -> Value {
        let r = gate.rec();
        let y = r.fresh(gate.ty().clone());
        r.push(
            Mlp::GegluTanh {
                gate: gate.id(),
                up: up.id(),
                y: y.id(),
            },
            &[gate, up],
        );
        y
    }

    pub fn geglu_tanh_packed(packed: &Value, intermediate: u32) -> Value {
        let r = packed.rec();
        let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
        r.push(
            Mlp::GegluTanhPacked {
                packed: packed.id(),
                intermediate,
                y: y.id(),
            },
            &[packed],
        );
        y
    }

    pub fn situ(packed: &Value, intermediate: u32, beta: f32, up_cap: Option<f32>) -> Value {
        let r = packed.rec();
        let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
        r.push(
            Mlp::Situ {
                packed: packed.id(),
                intermediate,
                beta,
                up_cap,
                y: y.id(),
            },
            &[packed],
        );
        y
    }
}

pub mod rope {
    use super::*;

    pub fn full(
        q: &Value,
        k: &Value,
        positions: &Value,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> (Value, Value) {
        let r = q.rec();
        let q_out = r.fresh(q.ty().clone());
        let k_out = r.fresh(k.ty().clone());
        r.push(
            Rope::Full {
                q: q.id(),
                k: k.id(),
                positions: positions.id(),
                head_dim,
                theta,
                interleaved,
                q_out: q_out.id(),
                k_out: k_out.id(),
            },
            &[q, k, positions],
        );
        (q_out, k_out)
    }

    pub fn partial(
        q: &Value,
        k: &Value,
        positions: &Value,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> (Value, Value) {
        let r = q.rec();
        let q_out = r.fresh(q.ty().clone());
        let k_out = r.fresh(k.ty().clone());
        r.push(
            Rope::Partial {
                q: q.id(),
                k: k.id(),
                positions: positions.id(),
                rotary_dim,
                head_dim,
                theta,
                q_out: q_out.id(),
                k_out: k_out.id(),
            },
            &[q, k, positions],
        );
        (q_out, k_out)
    }

    pub fn partial_q(
        q: &Value,
        positions: &Value,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Value {
        let r = q.rec();
        let q_out = r.fresh(q.ty().clone());
        r.push(
            Rope::PartialQ {
                q: q.id(),
                positions: positions.id(),
                rotary_dim,
                head_dim,
                theta,
                q_out: q_out.id(),
            },
            &[q, positions],
        );
        q_out
    }

    pub fn partial_last(
        q: &Value,
        positions: &Value,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Value {
        let r = q.rec();
        let q_out = r.fresh(q.ty().clone());
        r.push(
            Rope::PartialLast {
                q: q.id(),
                positions: positions.id(),
                rotary_dim,
                head_dim,
                theta,
                interleaved,
                q_out: q_out.id(),
            },
            &[q, positions],
        );
        q_out
    }

    pub fn yarn(
        q: &Value,
        k: &Value,
        positions: &Value,
        head_dim: u32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
        original_max_position: u32,
        interleaved: bool,
    ) -> (Value, Value) {
        let r = q.rec();
        let q_out = r.fresh(q.ty().clone());
        let k_out = r.fresh(k.ty().clone());
        r.push(
            Rope::Yarn {
                q: q.id(),
                k: k.id(),
                positions: positions.id(),
                head_dim,
                theta,
                factor,
                beta_fast,
                beta_slow,
                attention_factor,
                original_max_position,
                interleaved,
                q_out: q_out.id(),
                k_out: k_out.id(),
            },
            &[q, k, positions],
        );
        (q_out, k_out)
    }
}

pub mod attention {
    use super::*;

    /// Builds the decode plan from cache space `cache`'s declared geometry —
    /// once per forward, shared visibly by every layer's `decode` (§6).
    pub fn plan_decode(r: &Recorder, cache: u32) -> Value {
        let kv_indptr = geometry(r, cache, GeomKind::Indptr);
        let kv_indices = geometry(r, cache, GeomKind::Indices);
        let last_page_len = geometry(r, cache, GeomKind::LastPageLen);
        let plan = r.fresh(Ty::Struct(StructKind::AttnDecodePlan));
        r.push(
            Attention::PlanDecode {
                kv_indptr: kv_indptr.id(),
                kv_indices: kv_indices.id(),
                last_page_len: last_page_len.id(),
                plan: plan.id(),
            },
            &[&kv_indptr, &kv_indices, &last_page_len],
        );
        plan
    }

    /// Builds the prefill plan from cache space `cache`'s declared geometry.
    pub fn plan_prefill(r: &Recorder, cache: u32) -> Value {
        let kv_indptr = geometry(r, cache, GeomKind::Indptr);
        let kv_indices = geometry(r, cache, GeomKind::Indices);
        let last_page_len = geometry(r, cache, GeomKind::LastPageLen);
        let plan = r.fresh(Ty::Struct(StructKind::AttnPrefillPlan));
        r.push(
            Attention::PlanPrefill {
                kv_indptr: kv_indptr.id(),
                kv_indices: kv_indices.id(),
                last_page_len: last_page_len.id(),
                plan: plan.id(),
            },
            &[&kv_indptr, &kv_indices, &last_page_len],
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
        let lse = r.fresh(tensor(q.rows(), q.width() / u64::from(head_dim), Dtype::F32));
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
        let lse = r.fresh(tensor(q.rows(), q.width() / u64::from(head_dim), Dtype::F32));
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

    pub fn kv_append(k: &Value, v: &Value, pages: ValueId, kv_indices: &Value, positions: &Value) {
        let r = k.rec();
        r.push(
            Attention::KvAppend {
                k: k.id(),
                v: v.id(),
                cache: pages,
                kv_indices: kv_indices.id(),
                positions: positions.id(),
            },
            &[k, v, kv_indices, positions],
        );
    }

    pub fn kv_append_shared(plane: &Value, pages: ValueId, kv_indices: &Value, positions: &Value) {
        let r = plane.rec();
        r.push(
            Attention::KvAppendShared {
                plane: plane.id(),
                cache: pages,
                kv_indices: kv_indices.id(),
                positions: positions.id(),
            },
            &[plane, kv_indices, positions],
        );
    }
}

pub mod moe {
    use super::*;

    pub fn topk_softmax(logits: &Value, experts: u32, top_k: u32) -> (Value, Value) {
        let r = logits.rec();
        let routes = r.fresh(tensor(Dim::Tokens, top_k, Dtype::I32));
        let weights = r.fresh(tensor(Dim::Tokens, top_k, Dtype::F32));
        r.push(
            Moe::TopkSoftmax {
                logits: logits.id(),
                experts,
                top_k,
                routes: routes.id(),
                weights: weights.id(),
            },
            &[logits],
        );
        (routes, weights)
    }

    pub fn topk_sigmoid(
        logits: &Value,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
    ) -> (Value, Value) {
        let r = logits.rec();
        let routes = r.fresh(tensor(Dim::Tokens, top_k, Dtype::I32));
        let weights = r.fresh(tensor(Dim::Tokens, top_k, Dtype::F32));
        r.push(
            Moe::TopkSigmoid {
                logits: logits.id(),
                experts,
                top_k,
                renormalize,
                scaling,
                routes: routes.id(),
                weights: weights.id(),
            },
            &[logits],
        );
        (routes, weights)
    }

    pub fn topk_sqrt_softplus(
        logits: &Value,
        bias: &Weight,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
    ) -> (Value, Value) {
        let r = logits.rec();
        let routes = r.fresh(tensor(Dim::Tokens, top_k, Dtype::I32));
        let weights = r.fresh(tensor(Dim::Tokens, top_k, Dtype::F32));
        r.push(
            Moe::TopkSqrtSoftplus {
                logits: logits.id(),
                bias: r.weight(bias),
                experts,
                top_k,
                renormalize,
                scaling,
                routes: routes.id(),
                weights: weights.id(),
            },
            &[logits],
        );
        (routes, weights)
    }

    /// The routed rows are `tokens × top_k` — the fold of the old
    /// `per(routes)` rule, so `top_k` rides along as a wrapper argument.
    pub fn matmul_select(x: &Value, bank: &Weight, routes: &Value, top_k: u32) -> Value {
        let r = x.rec();
        let y = r.fresh(tensor(Dim::TokensTimes(top_k), bank.dim(1), x.dtype()));
        r.push(
            Moe::MatmulSelect {
                x: x.id(),
                bank: r.weight(bank),
                routes: routes.id(),
                y: y.id(),
            },
            &[x, routes],
        );
        y
    }

    pub fn matmul_select_bias(
        x: &Value,
        bank: &Weight,
        bias: &Weight,
        routes: &Value,
        top_k: u32,
    ) -> Value {
        let r = x.rec();
        let y = r.fresh(tensor(Dim::TokensTimes(top_k), bank.dim(1), x.dtype()));
        r.push(
            Moe::MatmulSelectBias {
                x: x.id(),
                bank: r.weight(bank),
                bias: r.weight(bias),
                routes: routes.id(),
                y: y.id(),
            },
            &[x, routes],
        );
        y
    }

    pub fn weighted_sum(routed: &Value, weights: &Value) -> Value {
        let r = routed.rec();
        let y = r.fresh(tensor(Dim::Tokens, routed.width(), routed.dtype()));
        r.push(
            Moe::WeightedSum {
                routed: routed.id(),
                weights: weights.id(),
                y: y.id(),
            },
            &[routed, weights],
        );
        y
    }

    pub fn sigmoid_gate_add(routed: &Value, shared: &Value, gate: &Value) -> Value {
        let r = routed.rec();
        let y = r.fresh(routed.ty().clone());
        r.push(
            Moe::SigmoidGateAdd {
                routed: routed.id(),
                shared: shared.id(),
                gate: gate.id(),
                y: y.id(),
            },
            &[routed, shared, gate],
        );
        y
    }
}

pub mod gate {
    use super::*;

    pub fn sigmoid_mul(x: &Value, gate: &Value) -> Value {
        let r = x.rec();
        let x_out = r.fresh(x.ty().clone());
        r.push(
            Gate::SigmoidMul {
                x: x.id(),
                gate: gate.id(),
                x_out: x_out.id(),
            },
            &[x, gate],
        );
        x_out
    }
}

pub mod layout {
    use super::*;

    pub fn embed(ids: &Value, table: &Weight, vocab: u32) -> Value {
        let r = ids.rec();
        let y = r.fresh(tensor(Dim::Tokens, table.dim(1), table.dtype()));
        r.push(
            Layout::Embed {
                ids: ids.id(),
                table: r.weight(table),
                vocab,
                y: y.id(),
            },
            &[ids],
        );
        y
    }

    pub fn split_qkv(packed: &Value, q_width: u32, kv_width: u32) -> (Value, Value, Value) {
        let r = packed.rec();
        let q = r.fresh(tensor(packed.rows(), q_width, packed.dtype()));
        let k = r.fresh(tensor(packed.rows(), kv_width, packed.dtype()));
        let v = r.fresh(tensor(packed.rows(), kv_width, packed.dtype()));
        r.push(
            Layout::SplitQkv {
                packed: packed.id(),
                q_width,
                kv_width,
                q: q.id(),
                k: k.id(),
                v: v.id(),
            },
            &[packed],
        );
        (q, k, v)
    }

    pub fn split_q_gate(packed: &Value, head_dim: u32) -> (Value, Value) {
        let r = packed.rec();
        let head_dim64 = u64::from(head_dim);
        let half = packed.width() / (2 * head_dim64) * head_dim64;
        let q = r.fresh(tensor(packed.rows(), half, packed.dtype()));
        let gate = r.fresh(tensor(packed.rows(), half, packed.dtype()));
        r.push(
            Layout::SplitQGate {
                packed: packed.id(),
                head_dim,
                q: q.id(),
                gate: gate.id(),
            },
            &[packed],
        );
        (q, gate)
    }

    pub fn split_rows(x: &Value, width: u32) -> (Value, Value) {
        let r = x.rec();
        let left = r.fresh(tensor(x.rows(), width, x.dtype()));
        let right = r.fresh(tensor(x.rows(), x.width() - u64::from(width), x.dtype()));
        r.push(
            Layout::SplitRows {
                x: x.id(),
                width,
                left: left.id(),
                right: right.id(),
            },
            &[x],
        );
        (left, right)
    }

    pub fn select(table: &Value, layer: u32, width: u32) -> Value {
        let r = table.rec();
        let y = r.fresh(tensor(table.rows(), width, table.dtype()));
        r.push(
            Layout::Select {
                table: table.id(),
                layer,
                width,
                y: y.id(),
            },
            &[table],
        );
        y
    }
}

pub mod ssm {
    use super::*;

    pub fn causal_conv1d(x: &Value, weight: &Weight, state: ValueId, conv_width: u32) -> Value {
        let r = x.rec();
        let y = r.fresh(x.ty().clone());
        r.push(
            Ssm::CausalConv1d {
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

    pub fn causal_conv1d_chunked(
        x: &Value,
        weight: &Weight,
        state: ValueId,
        conv_width: u32,
    ) -> Value {
        let r = x.rec();
        let y = r.fresh(x.ty().clone());
        r.push(
            Ssm::CausalConv1dChunked {
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

    pub fn gdn_prep(ba: &Value, dt_bias: &Weight, a_log: &Weight) -> Value {
        let r = ba.rec();
        let gates = r.fresh(tensor(ba.rows(), ba.width(), Dtype::F32));
        r.push(
            Ssm::GdnPrep {
                ba: ba.id(),
                dt_bias: r.weight(dt_bias),
                a_log: r.weight(a_log),
                gates: gates.id(),
            },
            &[ba],
        );
        gates
    }

    pub fn gated_delta(
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
            Ssm::GatedDelta {
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

    pub fn gated_delta_chunked(
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
            Ssm::GatedDeltaChunked {
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

    pub fn kda_step(
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
            Ssm::KdaStep {
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

    pub fn kda_chunked(
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
            Ssm::KdaChunked {
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
}

pub mod cuda {
    use super::*;

    /// Splits packed qkv, head-norms q and k, ropes them, norms v, and appends
    /// k/v in one pass; `q` is the only tensor left over.
    pub fn qkv_fused_qknorm_rope_vnorm_write(
        packed: &Value,
        q_norm: &declare::Norm,
        k_norm: &declare::Norm,
        kv_heads: u32,
        head_dim: u32,
        pages: ValueId,
        kv_indices: &Value,
        theta: f32,
        positions: &Value,
    ) -> Value {
        let r = packed.rec();
        let q_width = packed.width() - 2 * u64::from(kv_heads) * u64::from(head_dim);
        let q = r.fresh(tensor(packed.rows(), q_width, packed.dtype()));
        r.push(
            Cuda::QkvFusedQknormRopeVnormWrite {
                packed: packed.id(),
                positions: positions.id(),
                q_norm_weight: r.weight(&q_norm.weight),
                q_norm_eps: q_norm.eps,
                k_norm_weight: r.weight(&k_norm.weight),
                k_norm_eps: k_norm.eps,
                cache: pages,
                kv_indices: kv_indices.id(),
                kv_heads,
                head_dim,
                theta,
                q: q.id(),
            },
            &[packed, positions, kv_indices],
        );
        q
    }
}

pub mod mla {
    use super::*;

    /// Builds the one MLA plan — shared by decode and prefill — from cache
    /// space `cache`'s declared geometry.
    pub fn plan(r: &Recorder, cache: u32) -> Value {
        let kv_indptr = geometry(r, cache, GeomKind::Indptr);
        let kv_indices = geometry(r, cache, GeomKind::Indices);
        let last_page_len = geometry(r, cache, GeomKind::LastPageLen);
        let plan = r.fresh(Ty::Struct(StructKind::MlaPlan));
        r.push(
            Mla::Plan {
                kv_indptr: kv_indptr.id(),
                kv_indices: kv_indices.id(),
                last_page_len: last_page_len.id(),
                plan: plan.id(),
            },
            &[&kv_indptr, &kv_indices, &last_page_len],
        );
        plan
    }

    pub fn latents(kv_a: &Value, norm: &declare::Norm, kv_lora_rank: u32) -> (Value, Value) {
        let r = kv_a.rec();
        let kv_c = r.fresh(tensor(kv_a.rows(), kv_lora_rank, kv_a.dtype()));
        let k_pe = r.fresh(tensor(
            kv_a.rows(),
            kv_a.width() - u64::from(kv_lora_rank),
            kv_a.dtype(),
        ));
        r.push(
            Mla::Latents {
                kv_a: kv_a.id(),
                weight: r.weight(&norm.weight),
                eps: norm.eps,
                kv_lora_rank,
                kv_c: kv_c.id(),
                k_pe: k_pe.id(),
            },
            &[kv_a],
        );
        (kv_c, k_pe)
    }

    pub fn latents_rope(
        kv_a: &Value,
        positions: &Value,
        norm: &declare::Norm,
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
            Mla::LatentsRope {
                kv_a: kv_a.id(),
                positions: positions.id(),
                weight: r.weight(&norm.weight),
                eps: norm.eps,
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

    pub fn split_q_b(q_b: &Value, heads: u32, nope_dim: u32, rope_dim: u32) -> (Value, Value) {
        let r = q_b.rec();
        let heads64 = u64::from(heads);
        let q_nope = r.fresh(tensor(q_b.rows(), heads64 * u64::from(nope_dim), q_b.dtype()));
        let q_pe = r.fresh(tensor(q_b.rows(), heads64 * u64::from(rope_dim), q_b.dtype()));
        r.push(
            Mla::SplitQB {
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

    pub fn absorb_q(
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
            Mla::AbsorbQ {
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

    pub fn absorb_out(
        latent: &Value,
        kv_b: &Weight,
        heads: u32,
        kv_lora_rank: u32,
        v_head_dim: u32,
        nope_dim: u32,
    ) -> Value {
        let r = latent.rec();
        let width = u64::from(heads) * u64::from(v_head_dim);
        let o = r.fresh(tensor(latent.rows(), width, latent.dtype()));
        r.push(
            Mla::AbsorbOut {
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

    pub fn kv_append(
        kv_c: &Value,
        k_pe: &Value,
        pages: ValueId,
        kv_indices: &Value,
        positions: &Value,
    ) {
        let r = kv_c.rec();
        r.push(
            Mla::KvAppend {
                kv_c: kv_c.id(),
                k_pe: k_pe.id(),
                cache: pages,
                kv_indices: kv_indices.id(),
                positions: positions.id(),
            },
            &[kv_c, k_pe, kv_indices, positions],
        );
    }

    pub fn attention_decode(
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
            Mla::AttentionDecode {
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

    pub fn attention_prefill(
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
            Mla::AttentionPrefill {
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

    pub fn attention_decode_selected(
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
            Mla::AttentionDecodeSelected {
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

    pub fn attention_prefill_selected(
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
            Mla::AttentionPrefillSelected {
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
}

pub mod index {
    use super::*;

    pub fn layernorm_rope(
        k: &Value,
        positions: &Value,
        norm: &declare::Norm,
        bias: &Weight,
        rope_dim: u32,
        theta: f32,
    ) -> Value {
        let r = k.rec();
        let k_out = r.fresh(k.ty().clone());
        r.push(
            Index::LayernormRope {
                k: k.id(),
                positions: positions.id(),
                weight: r.weight(&norm.weight),
                bias: r.weight(bias),
                eps: norm.eps,
                rope_dim,
                theta,
                k_out: k_out.id(),
            },
            &[k, positions],
        );
        k_out
    }

    pub fn rope(
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
            Index::Rope {
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

    pub fn topk(
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
            Index::Topk {
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

    pub fn kv_append(k: &Value, keys: ValueId, kv_indices: &Value, positions: &Value) {
        let r = k.rec();
        r.push(
            Index::KvAppend {
                k: k.id(),
                keys,
                kv_indices: kv_indices.id(),
                positions: positions.id(),
            },
            &[k, kv_indices, positions],
        );
    }
}

pub mod pool {
    use super::*;

    pub fn boundary_decode(positions: &Value, ratio: u32) -> (Value, Value) {
        let r = positions.rec();
        let boundary_pos = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
        let boundary_req = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
        r.push(
            Pool::BoundaryDecode {
                positions: positions.id(),
                ratio,
                boundary_pos: boundary_pos.id(),
                boundary_req: boundary_req.id(),
            },
            &[positions],
        );
        (boundary_pos, boundary_req)
    }

    pub fn boundary_prefill(positions: &Value, ratio: u32) -> (Value, Value) {
        let r = positions.rec();
        let boundary_pos = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
        let boundary_req = r.fresh(tensor(Dim::Tokens, 1u64, Dtype::I32));
        r.push(
            Pool::BoundaryPrefill {
                positions: positions.id(),
                ratio,
                boundary_pos: boundary_pos.id(),
                boundary_req: boundary_req.id(),
            },
            &[positions],
        );
        (boundary_pos, boundary_req)
    }

    /// `dtype` is the pooled entries' element type — the wrapper has no data
    /// input to inherit it from, so the model states its activation dtype.
    pub fn gather(
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
            Pool::Gather {
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

    pub fn kv_append(
        entries: &Value,
        boundary_pos: &Value,
        boundary_req: &Value,
        pool: ValueId,
        kv_indices: &Value,
    ) {
        let r = entries.rec();
        r.push(
            Pool::KvAppend {
                entries: entries.id(),
                boundary_pos: boundary_pos.id(),
                boundary_req: boundary_req.id(),
                pool,
                kv_indices: kv_indices.id(),
            },
            &[entries, boundary_pos, boundary_req, kv_indices],
        );
    }

    pub fn attention_lse(
        q: &Value,
        positions: &Value,
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
            Pool::AttentionLse {
                q: q.id(),
                positions: positions.id(),
                entries,
                ratio,
                heads,
                head_dim,
                sm_scale,
                o: o.id(),
                lse: lse.id(),
            },
            &[q, positions],
        );
        (o, lse)
    }
}

pub mod hc {
    use super::*;

    pub fn expand(x: &Value, streams: u32) -> Value {
        let r = x.rec();
        let y = r.fresh(tensor(x.rows(), x.width() * u64::from(streams), x.dtype()));
        r.push(
            Hc::Expand {
                x: x.id(),
                streams,
                y: y.id(),
            },
            &[x],
        );
        y
    }

    pub fn rmsnorm_f32(streams: &Value, eps: f32) -> Value {
        let r = streams.rec();
        let y = r.fresh(tensor(streams.rows(), streams.width(), Dtype::F32));
        r.push(
            Hc::RmsnormF32 {
                streams: streams.id(),
                eps,
                y: y.id(),
            },
            &[streams],
        );
        y
    }

    pub fn gates(
        normed: &Value,
        streams: &Value,
        scale: &Weight,
        base: &Weight,
        stream_count: u32,
        gate_eps: f32,
        alpha: f32,
        sinkhorn: u32,
    ) -> (Value, Value, Value) {
        let r = normed.rec();
        let count = u64::from(stream_count);
        let x = r.fresh(tensor(
            streams.rows(),
            streams.width() / count,
            streams.dtype(),
        ));
        let post_mix = r.fresh(tensor(streams.rows(), count, Dtype::F32));
        let comb_mix = r.fresh(tensor(streams.rows(), count * count, Dtype::F32));
        r.push(
            Hc::Gates {
                normed: normed.id(),
                streams: streams.id(),
                scale: r.weight(scale),
                base: r.weight(base),
                stream_count,
                gate_eps,
                alpha,
                sinkhorn,
                x: x.id(),
                post_mix: post_mix.id(),
                comb_mix: comb_mix.id(),
            },
            &[normed, streams],
        );
        (x, post_mix, comb_mix)
    }

    pub fn fold(x: &Value, streams: &Value, post_mix: &Value, comb_mix: &Value) -> Value {
        let r = x.rec();
        let y = r.fresh(streams.ty().clone());
        r.push(
            Hc::Fold {
                x: x.id(),
                streams: streams.id(),
                post_mix: post_mix.id(),
                comb_mix: comb_mix.id(),
                y: y.id(),
            },
            &[x, streams, post_mix, comb_mix],
        );
        y
    }

    pub fn collapse(
        streams: &Value,
        head_scale: &Weight,
        head_base: &Weight,
        stream_count: u32,
        gate_eps: f32,
    ) -> Value {
        let r = streams.rec();
        let y = r.fresh(tensor(
            streams.rows(),
            streams.width() / u64::from(stream_count),
            streams.dtype(),
        ));
        r.push(
            Hc::Collapse {
                streams: streams.id(),
                head_scale: r.weight(head_scale),
                head_base: r.weight(head_base),
                stream_count,
                gate_eps,
                y: y.id(),
            },
            &[streams],
        );
        y
    }
}
