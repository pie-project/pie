//! The stated-kernel surface. Tier-1 fns state role points and every plane
//! answers them; a plane's own symbols live under its module and appear
//! only behind that plane's gate.

use crate::axes::{Dtype, F32};
use crate::declare::{Norm, Tensor};
use crate::forward::{Pages, State};
use crate::record::{Value, Windows};

/// The `Gemm` family, one fn per method of `kernels::points::Gemm`: same
/// names, same parameter order, the point's path as the recorded op.
///
/// `attention_landing`'s `layer` is a stated scalar in the declaration and
/// the statement's own layer TAG here, not a param — that is where the
/// driver has always read it, and moving it into the params run would have
/// changed every plan for nothing. IT TAKES NO ARGUMENT, because a text
/// already said which layer it is on: `Recorder::at` carries the layer the
/// text's `inputs.layers(..)` loop opened, every statement is stamped with
/// it, and a builder that asked for it again would be asking the text to
/// spell the loop's own index a second time.
pub mod gemm {
    use super::*;

    pub fn matmul<W: Dtype>(act: &Value, w: &Tensor<W>) -> Value {
        act.stmt("gemm.matmul").weight(w).done()
    }

    pub fn lm_head<W: Dtype>(act: &Value, w: &Tensor<W>) -> Value {
        act.stmt("gemm.lm_head").weight(w).done()
    }

    pub fn attention_landing<W: Dtype>(act: &Value, w: &Tensor<W>) -> Value {
        act.stmt("gemm.attention_landing").weight(w).done()
    }
}

/// The `Dist` family, one fn per method of `kernels::points::Dist`, plus
/// [`dist::reduce`] — the `TP` fold over the one method, which is not a
/// statement and is exempted in `builders_are_the_points` as such.
pub mod dist {
    use super::*;

    pub fn all_reduce(buf: &Value) -> Value {
        buf.stmt("dist.all_reduce").done()
    }

    /// A rank's partial, as the world's answer: `all_reduce` at `TP > 1`,
    /// and the value ITSELF at `TP == 1`.
    ///
    /// WORLD 1 IS THE IDENTITY. That is Q6's ratified law and it already
    /// holds where the cut is made — `model::produce` derives a degree per
    /// row and writes tp1's bytes whole, with no `if tp` in it — but the
    /// texts broke it twelve times over, once per rank-cut result in each
    /// of six families, each spelling the same four lines. A verb states
    /// the law once and the arithmetic disappears from the texts: a reader
    /// of `qwen_3_5/forward.rs` no longer has to check that the `else` arm
    /// really is the identity, because there is no `else` arm to check.
    ///
    /// BY VALUE, so the fold is free: the hand form moved `buf` out of the
    /// `else` arm and this moves it out of the `if`. The recorded plan is
    /// byte-identical at both worlds.
    #[must_use]
    pub fn reduce<const TP: usize>(buf: Value) -> Value {
        if TP > 1 { all_reduce(&buf) } else { buf }
    }
}

/// The `Norm` family, one fn per method of `kernels::points::Norm`: same
/// names, same parameter order, the point's path as the recorded op.
pub mod norm {
    use super::*;

    pub fn rmsnorm<W: Dtype>(x: &Value, weight: &Tensor<W>, eps: f32) -> Value {
        x.stmt("norm.rmsnorm").weight(weight).float(eps).done()
    }

    pub fn rmsnorm_per_head<W: Dtype>(
        x: &Value,
        weight: &Tensor<W>,
        head_dim: u32,
        eps: f32,
    ) -> Value {
        x.stmt("norm.rmsnorm_per_head")
            .weight(weight)
            .int(head_dim)
            .float(eps)
            .done()
    }

    /// [`rmsnorm`] against a bank stored as an OFFSET (`1 + weight`). A
    /// separate fn because it is a separate point: the convention is the
    /// checkpoint's and a text states one for its whole life.
    pub fn rmsnorm_plus_one<W: Dtype>(x: &Value, weight: &Tensor<W>, eps: f32) -> Value {
        x.stmt("norm.rmsnorm_plus_one")
            .weight(weight)
            .float(eps)
            .done()
    }

    pub fn rmsnorm_per_head_plus_one<W: Dtype>(
        x: &Value,
        weight: &Tensor<W>,
        head_dim: u32,
        eps: f32,
    ) -> Value {
        x.stmt("norm.rmsnorm_per_head_plus_one")
            .weight(weight)
            .int(head_dim)
            .float(eps)
            .done()
    }

    pub fn rmsnorm_no_scale(x: &Value, head_dim: u32, eps: f32) -> Value {
        x.stmt("norm.rmsnorm_no_scale")
            .int(head_dim)
            .float(eps)
            .done()
    }

    pub fn rmsnorm_gated<W: Dtype>(
        x: &Value,
        gate: &Value,
        weight: &Tensor<W>,
        head_dim: u32,
        eps: f32,
    ) -> Value {
        x.stmt("norm.rmsnorm_gated")
            .value(gate)
            .weight(weight)
            .int(head_dim)
            .float(eps)
            .done()
    }

    pub fn rmsnorm_gated_by<W: Dtype>(
        x: &Value,
        gate: &Value,
        weight: &Tensor<W>,
        heads: u32,
        eps: f32,
    ) -> Value {
        x.stmt("norm.rmsnorm_gated_by")
            .value(gate)
            .weight(weight)
            .int(heads)
            .float(eps)
            .done()
    }

    pub fn residual_add(x: &Value, y: &Value) -> Value {
        x.stmt("norm.residual_add").value(y).done()
    }

    pub fn add_bias<W: Dtype>(bias: &Tensor<W>, out: &Value) -> Value {
        out.stmt("norm.add_bias").weight(bias).done()
    }

    pub fn mul_scalar(s: f32, x: &Value) -> Value {
        x.stmt("norm.mul_scalar").float(s).done()
    }

    pub fn scale<W: Dtype>(s: &Tensor<W>, x: &Value) -> Value {
        x.stmt("norm.scale").weight(s).done()
    }

    /// THE ONE VARIADIC BUILDER on this surface, and the open ledger item
    /// the declaration names: `blocks` grows by one every layer that blends,
    /// so this records one value per block where `norm.res_blend` states the
    /// single concatenated rectangle its routine takes. The recorded op is
    /// what it always was — receiver, then the blocks, then the norm's
    /// weight and eps, then the projection.
    pub fn res_blend<W: Dtype>(
        prefix: &Value,
        blocks: &[Value],
        norm: &Norm<W>,
        proj: &Tensor<W>,
    ) -> Value {
        let mut s = prefix.stmt("norm.res_blend");
        for block in blocks {
            s = s.value(block);
        }
        s.norm(norm).weight(proj).done()
    }
}

pub mod mlp {
    use super::*;

    pub fn swiglu(packed: &Value, intermediate: u32) -> Value {
        packed.stmt("mlp.swiglu").int(intermediate).done()
    }

    pub fn swiglu_clamp(packed: &Value, intermediate: u32, limit: f32) -> Value {
        packed
            .stmt("mlp.swiglu_clamp")
            .int(intermediate)
            .float(limit)
            .done()
    }

    pub fn swiglu_clamp_alpha(packed: &Value, intermediate: u32, limit: f32, alpha: f32) -> Value {
        packed
            .stmt("mlp.swiglu_clamp_alpha")
            .int(intermediate)
            .float(limit)
            .float(alpha)
            .done()
    }

    pub fn geglu_tanh(gate: &Value, up: &Value) -> Value {
        gate.stmt("mlp.geglu_tanh").value(up).done()
    }

    pub fn geglu_tanh_packed(packed: &Value, intermediate: u32) -> Value {
        packed
            .stmt("mlp.geglu_tanh_packed")
            .int(intermediate)
            .done()
    }

    /// `up_cap` is optional in the text and rides a `0.0` sentinel in the
    /// statement — the encoding the point declares, kept verbatim.
    pub fn situ(packed: &Value, intermediate: u32, beta: f32, up_cap: Option<f32>) -> Value {
        packed
            .stmt("mlp.situ")
            .int(intermediate)
            .float(beta)
            .float(up_cap.unwrap_or(0.0))
            .done()
    }
}

/// The `Rope` family, one fn per method of `kernels::points::Rope`: same
/// names, same parameter order, the point's path as the recorded op. Every
/// one rotates in place, so a statement's results are the operands it
/// rotated — `q` and `k` for the two-operand points, `q` alone for the
/// rest.
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
        q.stmt("rope.full")
            .value(k)
            .value(positions)
            .int(head_dim)
            .float(theta)
            .int(u32::from(interleaved))
            .pair()
    }

    pub fn partial(
        q: &Value,
        k: &Value,
        positions: &Value,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> (Value, Value) {
        q.stmt("rope.partial")
            .value(k)
            .value(positions)
            .int(rotary_dim)
            .int(head_dim)
            .float(theta)
            .pair()
    }

    pub fn partial_q(
        q: &Value,
        positions: &Value,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Value {
        q.stmt("rope.partial_q")
            .value(positions)
            .int(rotary_dim)
            .int(head_dim)
            .float(theta)
            .done()
    }

    pub fn partial_last(
        q: &Value,
        positions: &Value,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Value {
        q.stmt("rope.partial_last")
            .value(positions)
            .int(rotary_dim)
            .int(head_dim)
            .float(theta)
            .int(u32::from(interleaved))
            .done()
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
        q.stmt("rope.yarn")
            .value(k)
            .value(positions)
            .int(head_dim)
            .float(theta)
            .float(factor)
            .float(beta_fast)
            .float(beta_slow)
            .float(attention_factor)
            .int(original_max_position)
            .int(u32::from(interleaved))
            .pair()
    }
}

pub fn query_windows(x: &Value) -> Windows {
    Windows {
        data: x.clone(),
        indptr: x.rec.runtime("qo_indptr"),
    }
}

/// The `Attention` family, one fn per method of `kernels::points::Attention`:
/// same names, same parameter order, the point's path as the recorded op.
///
/// A `Cache<Self::Pages>` slot IS `.cache(&pages.name)` — the statement names
/// the KV pool row and the driver binds it — exactly as a recurrent slot is,
/// and a prefill point's `indptr` is the `Windows` half the ragged pairing
/// already carries, so a prefill fn takes the `&Windows` and unpacks it here
/// rather than asking a text for two values it holds as one.
///
/// `window` stays `Option<u32>` at this surface and the declaration states a
/// `u32`, because `.window()` is `int(w.unwrap_or(0))`: the text says what it
/// means and the plan carries what the driver reads, and neither spelling
/// moved when the family landed.
pub mod attention {
    use super::*;

    pub fn decode(
        q: &Value,
        pages: &Pages,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
    ) -> Value {
        q.stmt("attention.decode")
            .cache(&pages.name)
            .window(window)
            .int(head_dim)
            .float(sm_scale)
            .done()
    }

    pub fn prefill(
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

    pub fn masked(
        w: &Windows,
        pages: &Pages,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
    ) -> Value {
        w.data
            .stmt("attention.masked")
            .value(&w.indptr)
            .cache(&pages.name)
            .window(window)
            .int(head_dim)
            .float(sm_scale)
            .done()
    }

    pub fn decode_lse(
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

    pub fn prefill_lse(
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

    pub fn sink<W: Dtype>(o: &Value, lse: &Value, sink: &Tensor<W>, head_dim: u32) -> Value {
        o.stmt("attention.sink")
            .value(lse)
            .weight(sink)
            .int(head_dim)
            .done()
    }

    pub fn merge_lse(
        o1: &Value,
        lse1: &Value,
        o2: &Value,
        lse2: &Value,
        heads: u32,
        head_dim: u32,
    ) -> (Value, Value) {
        o1.stmt("attention.merge_lse")
            .value(lse1)
            .value(o2)
            .value(lse2)
            .int(heads)
            .int(head_dim)
            .pair()
    }

    pub fn logit_softcap(x: &Value, cap: f32) -> Value {
        x.stmt("attention.logit_softcap").float(cap).done()
    }

    pub fn kv_append(k: &Value, v: &Value, pages: &Pages) {
        k.stmt("attention.kv_append")
            .value(v)
            .cache(&pages.name)
            .effect();
    }

    pub fn kv_append_shared(plane: &Value, pages: &Pages) {
        plane
            .stmt("attention.kv_append_shared")
            .cache(&pages.name)
            .effect();
    }
}

/// The `Moe` family, one fn per method of `kernels::points::Moe`: same
/// names, same parameter order, the point's path as the recorded op. A
/// router states its two results as two values, which is what the
/// declaration's two `Out` slots are.
pub mod moe {
    use super::*;

    pub fn topk_softmax(logits: &Value, experts: u32, top_k: u32) -> (Value, Value) {
        logits
            .stmt("moe.topk_softmax")
            .int(experts)
            .int(top_k)
            .pair()
    }

    pub fn topk_sigmoid(
        logits: &Value,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
    ) -> (Value, Value) {
        logits
            .stmt("moe.topk_sigmoid")
            .int(experts)
            .int(top_k)
            .int(u32::from(renormalize))
            .float(scaling)
            .pair()
    }

    pub fn topk_sqrt_softplus<W: Dtype>(
        logits: &Value,
        bias: &Tensor<W>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
    ) -> (Value, Value) {
        logits
            .stmt("moe.topk_sqrt_softplus")
            .weight(bias)
            .int(experts)
            .int(top_k)
            .int(u32::from(renormalize))
            .float(scaling)
            .pair()
    }

    pub fn matmul_select<W: Dtype>(x: &Value, bank: &Tensor<W>, routes: &Value) -> Value {
        x.stmt("moe.matmul_select")
            .weight(bank)
            .value(routes)
            .done()
    }

    /// TWO REPR PARAMETERS AND NOW THEY MEAN DIFFERENT THINGS. `W` is the
    /// BANK's storage form and `B` the bias row's element; the declaration
    /// quantifies over both (`<T: Scalar, R: Repr>`) and gpt-oss instantiates
    /// them apart — an mxfp4 stack beside a bf16 bias. This used to be a
    /// documented exception to the builder mapping, because a single `W`
    /// could not spell it and the declaration had no second axis to hang the
    /// bank on.
    ///
    /// `.bank(..)` and not `.weight(..)` for the same reason: a bank at a
    /// quantised repr is as many parameters as its repr stores planes.
    pub fn matmul_select_bias<W: Dtype, B: Dtype>(
        x: &Value,
        bank: &Tensor<W>,
        bias: &Tensor<B>,
        routes: &Value,
    ) -> Value {
        x.stmt("moe.matmul_select_bias")
            .bank(bank)
            .weight(bias)
            .value(routes)
            .done()
    }

    pub fn weighted_sum(routed: &Value, weights: &Value) -> Value {
        routed.stmt("moe.weighted_sum").value(weights).done()
    }

    pub fn sigmoid_gate_add(routed: &Value, shared: &Value, gate: &Value) -> Value {
        routed
            .stmt("moe.sigmoid_gate_add")
            .value(shared)
            .value(gate)
            .done()
    }
}

/// The `Gate` family, one fn per method of `kernels::points::Gate`.
pub mod gate {
    use super::*;

    pub fn sigmoid_mul(x: &Value, gate: &Value) -> Value {
        x.stmt("gate.sigmoid_mul").value(gate).done()
    }
}

/// The `Layout` family, one fn per method of `kernels::points::Layout`:
/// same names, same parameter order, the point's path as the recorded op.
/// Every width here is a param on the statement AND the number its results
/// are allocated from — a cut's halves are sized here and nowhere else,
/// which is why a plane's claim may state one and never look at it.
pub mod layout {
    use super::*;

    /// One row of `table` per token id. The result is `ids`' rows by the
    /// TABLE's width, a shape no `out(..)` convention states: the statement
    /// allocates it and the plane reads the width back off it.
    ///
    /// `vocab` is the table's ROW count and sizes nothing here — it is the
    /// bound the gather clamps every id against, and the only slot in this
    /// module whose stated number guards a read.
    pub fn embed<W: Dtype>(ids: &Value, table: &Tensor<W>, vocab: u32) -> Value {
        ids.stmt("layout.embed").weight(table).int(vocab).done()
    }

    pub fn split_qkv(packed: &Value, q_width: u32, kv_width: u32) -> (Value, Value, Value) {
        packed
            .stmt("layout.split_qkv")
            .int(q_width)
            .int(kv_width)
            .triple()
    }

    pub fn split_q_gate(packed: &Value, head_dim: u32) -> (Value, Value) {
        packed.stmt("layout.split_q_gate").int(head_dim).pair()
    }

    pub fn split_rows(x: &Value, width: u32) -> (Value, Value) {
        x.stmt("layout.split_rows").int(width).pair()
    }

    /// One layer's slice of a relayed table: `[rows, layers * width]` in,
    /// `[rows, width]` out, at column `layer * width`.
    ///
    /// `width` is STATED because `layer` says which slice and never how
    /// many, so the packed row divides by a number no operand carries. The
    /// text holds it (gemma's `Ple::dim`) and the walk sizes the result from
    /// it — the `rmsnorm_per_head` law, applied to an operand instead of a
    /// weight.
    pub fn select(table: &Value, layer: u32, width: u32) -> Value {
        table.stmt("layout.select").int(layer).int(width).done()
    }
}

/// The `Ssm` family, one fn per method of `kernels::points::Ssm`: same
/// names, same parameter order, the point's path as the recorded op.
///
/// A `Cache<Self::Recurrent>` slot IS `.cache(&state.name)` — the statement
/// names the pool row and the driver binds it — and a `_chunked` point's
/// `indptr` is the `Windows` half the ragged pairing already carries, so a
/// chunked fn takes the `&Windows` and unpacks it here rather than asking a
/// text for two values it holds as one.
pub mod ssm {
    use super::*;

    pub fn causal_conv1d<W: Dtype>(
        x: &Value,
        weight: &Tensor<W>,
        state: &State,
        conv_width: u32,
    ) -> Value {
        x.stmt("ssm.causal_conv1d")
            .weight(weight)
            .cache(&state.name)
            .int(conv_width)
            .done()
    }

    pub fn causal_conv1d_chunked<W: Dtype>(
        w: &Windows,
        weight: &Tensor<W>,
        state: &State,
        conv_width: u32,
    ) -> Value {
        w.data
            .stmt("ssm.causal_conv1d_chunked")
            .value(&w.indptr)
            .weight(weight)
            .cache(&state.name)
            .int(conv_width)
            .done()
    }

    /// `a_log` is `Tensor<F32>` and not a second `W`, because
    /// `ssm.gdn_prep` declares that slot `Const<Self::Tensor<f32>>` while
    /// `dt_bias` rides `T`. The two are not one axis and a builder that let
    /// a text pass one dtype for both would be the place the lie could be
    /// told.
    pub fn gdn_prep<W: Dtype>(ba: &Value, dt_bias: &Tensor<W>, a_log: &Tensor<F32>) -> Value {
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

    /// `dt_bias` AND `a_log` are both `Tensor<F32>` and neither is a second
    /// `W`: `ssm.kda_step` declares both slots `Const<Self::Tensor<f32>>`,
    /// because `ssm/kda.cuh`'s `kda_gate_beta` takes both as `const
    /// float*`. The neighbouring [`gdn_prep`] splits them — that kernel
    /// reads `dt_bias` at the model's element — so the two builders differ
    /// here exactly where their kernels do.
    pub fn kda_step(
        mixed: &Value,
        f: &Value,
        b: &Value,
        dt_bias: &Tensor<F32>,
        a_log: &Tensor<F32>,
        state: &State,
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
            .cache(&state.name)
            .int(heads)
            .int(head_dim)
            .float(norm_eps)
            .done()
    }

    /// [`kda_step`] over a window; the same F32 decay pair, for the same
    /// reason.
    pub fn kda_chunked(
        w: &Windows,
        f: &Value,
        b: &Value,
        dt_bias: &Tensor<F32>,
        a_log: &Tensor<F32>,
        state: &State,
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
            .cache(&state.name)
            .int(heads)
            .int(head_dim)
            .float(norm_eps)
            .done()
    }
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

/// The `Mla` family, one fn per method of `kernels::points::Mla`: same
/// names, same parameter order, the point's path as the recorded op.
///
/// A `Cache<Self::Pages>` slot IS `.cache(&pages.name)` — the statement
/// names the pool row and the driver binds it — and the append that fills
/// that pool is a point of THIS family, because a cache write belongs to
/// whoever owns the cache.
///
/// THE TWO ABSORBS EACH STATE A WIDTH THEY DO NOT USE. `kv_b` is one bank
/// whose per-head pitch is `(nope_dim + v_head_dim) * kv_lora_rank`, a
/// `Const` carries an address and no rectangle, and so each absorb states
/// both halves — the one it multiplies by and the one it only steps over.
/// Both texts already hold both numbers.
pub mod mla {
    use super::*;

    pub fn latents<W: Dtype>(kv_a: &Value, norm: &Norm<W>, kv_lora_rank: u32) -> (Value, Value) {
        kv_a.stmt("mla.latents").norm(norm).int(kv_lora_rank).pair()
    }

    pub fn latents_rope<W: Dtype>(
        kv_a: &Value,
        positions: &Value,
        norm: &Norm<W>,
        kv_lora_rank: u32,
        rope_dim: u32,
        theta: f32,
    ) -> (Value, Value) {
        kv_a.stmt("mla.latents_rope")
            .value(positions)
            .norm(norm)
            .int(kv_lora_rank)
            .int(rope_dim)
            .float(theta)
            .pair()
    }

    pub fn split_q_b(q_b: &Value, heads: u32, nope_dim: u32, rope_dim: u32) -> (Value, Value) {
        q_b.stmt("mla.split_q_b")
            .int(heads)
            .int(nope_dim)
            .int(rope_dim)
            .pair()
    }

    pub fn absorb_q<W: Dtype>(
        q_nope: &Value,
        kv_b: &Tensor<W>,
        heads: u32,
        kv_lora_rank: u32,
        nope_dim: u32,
        v_head_dim: u32,
    ) -> Value {
        q_nope
            .stmt("mla.absorb_q")
            .weight(kv_b)
            .int(heads)
            .int(kv_lora_rank)
            .int(nope_dim)
            .int(v_head_dim)
            .done()
    }

    pub fn absorb_out<W: Dtype>(
        latent: &Value,
        kv_b: &Tensor<W>,
        heads: u32,
        kv_lora_rank: u32,
        v_head_dim: u32,
        nope_dim: u32,
    ) -> Value {
        latent
            .stmt("mla.absorb_out")
            .weight(kv_b)
            .int(heads)
            .int(kv_lora_rank)
            .int(v_head_dim)
            .int(nope_dim)
            .done()
    }

    pub fn kv_append(kv_c: &Value, k_pe: &Value, pages: &Pages) {
        kv_c.stmt("mla.kv_append")
            .value(k_pe)
            .cache(&pages.name)
            .effect();
    }

    pub fn attention_decode(
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

    /// ONE `&Windows` AND NOT TWO. The declaration states a single ragged
    /// pair — `q`, then the boundaries — and `q_pe` beside it as a plain
    /// operand. This fn used to take a second `&Windows` for it and read
    /// only its `.data`, which asked every text for an `indptr` the
    /// statement never recorded.
    pub fn attention_prefill(
        w: &Windows,
        q_pe: &Value,
        pages: &Pages,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
    ) -> Value {
        w.data
            .stmt("mla.attention_prefill")
            .value(&w.indptr)
            .value(q_pe)
            .cache(&pages.name)
            .int(heads)
            .int(kv_lora_rank)
            .float(sm_scale)
            .done()
    }

    /// [`attention_decode`] over the keys `selection` keeps. The query is
    /// the SAME separate pair — there is no fused query anywhere in this
    /// tree, at either end — and the selection is `index::topk`'s `i32`
    /// index list.
    pub fn attention_decode_selected(
        q: &Value,
        q_pe: &Value,
        selection: &Value,
        pages: &Pages,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
    ) -> Value {
        q.stmt("mla.attention_decode_selected")
            .value(q_pe)
            .value(selection)
            .cache(&pages.name)
            .int(heads)
            .int(kv_lora_rank)
            .float(sm_scale)
            .done()
    }

    pub fn attention_prefill_selected(
        w: &Windows,
        q_pe: &Value,
        selection: &Value,
        pages: &Pages,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
    ) -> Value {
        w.data
            .stmt("mla.attention_prefill_selected")
            .value(&w.indptr)
            .value(q_pe)
            .value(selection)
            .cache(&pages.name)
            .int(heads)
            .int(kv_lora_rank)
            .float(sm_scale)
            .done()
    }
}

/// The `Index` family, one fn per method of `kernels::points::Index`: same
/// names, same parameter order, the point's path as the recorded op.
///
/// Both rotations are in place, so a statement's result is the row it
/// rotated — the `rope` family's reading. The indexer's keys ride a pool of
/// their own and the append that fills it is a point here.
pub mod index {
    use super::*;

    pub fn layernorm_rope<W: Dtype>(
        k: &Value,
        positions: &Value,
        norm: &Norm<W>,
        bias: &Tensor<W>,
        rope_dim: u32,
        theta: f32,
    ) -> Value {
        k.stmt("index.layernorm_rope")
            .value(positions)
            .norm(norm)
            .weight(bias)
            .int(rope_dim)
            .float(theta)
            .done()
    }

    pub fn rope(
        q: &Value,
        positions: &Value,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
    ) -> Value {
        q.stmt("index.rope")
            .value(positions)
            .int(heads)
            .int(head_dim)
            .int(rope_dim)
            .float(theta)
            .done()
    }

    pub fn topk(
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

    pub fn kv_append(k: &Value, keys: &Pages) {
        k.stmt("index.kv_append").cache(&keys.name).effect();
    }
}

/// The `Pool` family, one fn per method of `kernels::points::Pool`: same
/// names, same parameter order, the point's path as the recorded op.
///
/// `boundary_decode` and `boundary_prefill` state TWO results where cuda's
/// routines write three — the third is a boundary rope plane no text reads,
/// and the statement is recorded as the texts state it. The append that
/// fills the entries pool and the attention that reads it are points here
/// rather than under `attention.`, because the pool is this family's.
pub mod pool {
    use super::*;

    pub fn boundary_decode(positions: &Value, ratio: u32) -> (Value, Value) {
        positions.stmt("pool.boundary_decode").int(ratio).pair()
    }

    pub fn boundary_prefill(w: &Windows, ratio: u32) -> (Value, Value) {
        w.data
            .stmt("pool.boundary_prefill")
            .value(&w.indptr)
            .int(ratio)
            .pair()
    }

    pub fn gather(
        boundary_pos: &Value,
        boundary_req: &Value,
        pages: &Pages,
        head_dim: u32,
        ratio: u32,
    ) -> Value {
        boundary_pos
            .stmt("pool.gather")
            .value(boundary_req)
            .cache(&pages.name)
            .int(head_dim)
            .int(ratio)
            .done()
    }

    pub fn kv_append(entries: &Value, boundary_pos: &Value, boundary_req: &Value, pool: &Pages) {
        entries
            .stmt("pool.kv_append")
            .value(boundary_pos)
            .value(boundary_req)
            .cache(&pool.name)
            .effect();
    }

    pub fn attention_lse(
        q: &Value,
        positions: &Value,
        entries: &Pages,
        ratio: u32,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
    ) -> (Value, Value) {
        q.stmt("pool.attention_lse")
            .value(positions)
            .cache(&entries.name)
            .int(ratio)
            .int(heads)
            .int(head_dim)
            .float(sm_scale)
            .pair()
    }
}

/// The `Hc` family, one fn per method of `kernels::points::Hc`: same names,
/// same parameter order, the point's path as the recorded op.
///
/// `gates` states THREE results and the first is the one the block consumes
/// — `(x, post_mix, comb_mix)`, which is the order every text reads them in
/// and the order the declaration's `Out` slots stand in. cuda's routine
/// writes them in a different order and the claim maps between the two,
/// named, once.
pub mod hc {
    use super::*;

    pub fn expand(x: &Value, streams: u32) -> Value {
        x.stmt("hc.expand").int(streams).done()
    }

    pub fn rmsnorm_f32(streams: &Value, eps: f32) -> Value {
        streams.stmt("hc.rmsnorm_f32").float(eps).done()
    }

    /// `scale` and `base` are `Tensor<F32>` and not a repr axis, for the
    /// reason [`ssm::kda_step`]'s pair is: `hc.gates` declares both slots
    /// `Const<Self::Tensor<f32>>` and the dispatch binds them as `float*`,
    /// so a bf16 bank here is an over-read the address carries no check
    /// against. `deepseek_v4::model::Mix` declares them that way.
    pub fn gates(
        normed: &Value,
        streams: &Value,
        scale: &Tensor<F32>,
        base: &Tensor<F32>,
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

    pub fn fold(x: &Value, streams: &Value, post_mix: &Value, comb_mix: &Value) -> Value {
        x.stmt("hc.fold")
            .value(streams)
            .value(post_mix)
            .value(comb_mix)
            .done()
    }

    /// `head_scale` and `head_base` are `Tensor<F32>` for [`gates`]'
    /// reason, one point over.
    pub fn collapse(
        streams: &Value,
        head_scale: &Tensor<F32>,
        head_base: &Tensor<F32>,
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
}
