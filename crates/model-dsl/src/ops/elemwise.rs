//! The `Elementwise` family: the norms and residual algebra, rope, the sigmoid
//! gate, and the hyper-connection stream ops.

use super::*;

pub use model_ir::ops::elemwise::Yarn;

pub fn rmsnorm(x: &Value, weight: &Weight, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elementwise::Rmsnorm {
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
        Elementwise::RmsnormPerHead {
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
        Elementwise::RmsnormPlusOne {
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
        Elementwise::RmsnormPerHeadPlusOne {
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
        Elementwise::RmsnormNoScale {
            x: x.id(),
            head_dim,
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

/// The centred norm: mean-subtract, then rms-normalize, no scale or bias.
/// The vision towers' `nn.LayerNorm` minus its two learned vectors, which
/// bake into the GEMM that reads the norm at import. Whole rows, no head
/// grouping.
pub fn layernorm_no_scale(x: &Value, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elementwise::LayernormNoScale {
            x: x.id(),
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

/// The whole `nn.LayerNorm` in one node: centred, scaled by `weight`, biased
/// by `bias`. `y = (x − mean(x)) · rsqrt(var(x) + eps) · w + b`. One node
/// rather than three, saving two device rectangles per norm.
///
/// [`layernorm_no_scale`] stays for the text whose scale genuinely bakes.
pub fn layernorm(x: &Value, weight: &Weight, bias: &Weight, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elementwise::Layernorm {
            x: x.id(),
            weight: r.weight(weight),
            bias: r.weight(bias),
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

/// The clipped linear's clamp: `min(max(x, lo), hi)`, in place, both bounds
/// trace constants. gemma4's `use_clipped_linears` publishes
/// `{input,output}_{min,max}` beside every vision projection.
/// The same clamp, with the bounds the checkpoint ships as `[1]` weight
/// planes rather than trace constants — gemma4's per-projection scalars are
/// too many and too checkpoint-specific to state as a `const`.
///
/// [`clamp`] stays for a bound the config states, like [`mul_scalar`]/[`scale`].
pub fn clamp_learned(x: &Value, lo: &Weight, hi: &Weight) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elementwise::ClampLearned {
            x: x.id(),
            lo: r.weight(lo),
            hi: r.weight(hi),
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

pub fn clamp(x: &Value, lo: f32, hi: f32) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elementwise::Clamp {
            x: x.id(),
            lo,
            hi,
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

pub fn rmsnorm_gated(
    x: &Value,
    gate: &Value,
    weight: &Weight,
    head_dim: u32,
    eps: f32,
    act: GateActivation,
) -> Value {
    let r = x.rec();
    let y = r.fresh(gate.ty().clone());
    r.push(
        Elementwise::RmsnormGated {
            x: x.id(),
            gate: gate.id(),
            weight: r.weight(weight),
            head_dim,
            eps,
            act,
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
        Elementwise::RmsnormGatedBy {
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

/// The hyper-connection norm: moments per `group`-wide slice, `weight + 1`
/// over the row's full width ([`Elementwise::RmsnormGroupedPlusOne`]).
pub fn rmsnorm_grouped_plus_one(x: &Value, weight: &Weight, group: u32, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elementwise::RmsnormGroupedPlusOne {
            x: x.id(),
            weight: r.weight(weight),
            group,
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn residual_add(x: &Value, y: &Value) -> Value {
    let r = x.rec();
    let y_out = r.fresh(y.ty().clone());
    r.push(
        Elementwise::ResidualAdd {
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
        Elementwise::AddBias {
            bias: r.weight(bias),
            out: out.id(),
            out_out: out_out.id(),
        },
        &[out],
    );
    out_out
}

/// The tower's output standardization (`vision_config.standardize`):
/// `y = (x − bias) · scale`, per column, in place. The last thing
/// `Gemma4VisionModel.forward` does to a pooled soft token before the
/// multimodal embedder projects it into trunk space.
pub fn standardize(x: &Value, bias: &Weight, scale: &Weight) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elementwise::Standardize {
            x: x.id(),
            bias: r.weight(bias),
            scale: r.weight(scale),
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

pub fn mul_scalar(s: f32, x: &Value) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elementwise::MulScalar {
            s,
            x: x.id(),
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

/// `silu(s · x)`, in place — the scalar sits inside the activation, which is
/// why this is not [`mul_scalar`] followed by anything.
pub fn silu_scaled(s: f32, x: &Value) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elementwise::SiluScaled {
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
        Elementwise::Scale {
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
    norm: &Weight,
    eps: f32,
    proj: &Weight,
) -> Value {
    let r = prefix.rec();
    let y = r.fresh(prefix.ty().clone());
    let mut ins: Vec<&Value> = Vec::with_capacity(1 + blocks.len());
    ins.push(prefix);
    ins.extend(blocks);
    r.push(
        Elementwise::ResBlend {
            prefix: prefix.id(),
            blocks: blocks.iter().map(Value::id).collect(),
            weight: r.weight(norm),
            eps,
            proj: r.weight(proj),
            y: y.id(),
        },
        &ins,
    );
    y
}

pub fn rope_full(
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
        Elementwise::RopeFull {
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

pub fn rope_partial(
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
        Elementwise::RopePartial {
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

pub fn rope_partial_q(
    q: &Value,
    positions: &Value,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Value {
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    r.push(
        Elementwise::RopePartialQ {
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

pub fn rope_partial_last(
    q: &Value,
    positions: &Value,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
) -> Value {
    rope_partial_last_yarn(q, positions, rotary_dim, head_dim, theta, interleaved, false, None)
}

/// [`rope_partial_last`] with the two facts a layer may state beside its
/// theta: `inverse` (un-rotate — the attention output of an MLA whose latent
/// is both key and value) and a YaRN ramp (`Some` on the layers whose
/// checkpoint states one).
#[allow(clippy::too_many_arguments)]
pub fn rope_partial_last_yarn(
    q: &Value,
    positions: &Value,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
    inverse: bool,
    yarn: Option<Yarn>,
) -> Value {
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    r.push(
        Elementwise::RopePartialLast {
            q: q.id(),
            positions: positions.id(),
            rotary_dim,
            head_dim,
            theta,
            interleaved,
            inverse,
            yarn,
            q_out: q_out.id(),
        },
        &[q, positions],
    );
    q_out
}

pub fn rope_yarn(
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
        Elementwise::RopeYarn {
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

pub fn gate_sigmoid_mul(x: &Value, gate: &Value) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elementwise::GateSigmoidMul {
            x: x.id(),
            gate: gate.id(),
            x_out: x_out.id(),
        },
        &[x, gate],
    );
    x_out
}

pub fn hc_expand(x: &Value, streams: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(tensor(x.rows(), x.width() * u64::from(streams), x.dtype()));
    r.push(
        Elementwise::HcExpand {
            x: x.id(),
            streams,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn hc_rmsnorm_f32(streams: &Value, eps: f32) -> Value {
    let r = streams.rec();
    let y = r.fresh(tensor(streams.rows(), streams.width(), Dtype::F32));
    r.push(
        Elementwise::HcRmsnormF32 {
            streams: streams.id(),
            eps,
            y: y.id(),
        },
        &[streams],
    );
    y
}

/// The per-token mix row: the normed stream row projected through the layer's
/// dynamic hyper plane (`{attn,ffn}_hc.fn`, `[2M + M², M·hidden]`) into the
/// `2M + M²` numbers [`hc_gates`] splits.
pub fn hc_project(normed: &Value, dynamic: &Weight, stream_count: u32) -> Value {
    let r = normed.rec();
    let count = u64::from(stream_count);
    let mix_hc = 2 * count + count * count;
    // The row is as wide as the plane says: a layer's plane lands the
    // `2M + M²` row `hc_gates` splits, the trunk's lands the `M` gates
    // `hc_collapse` folds under. Anything else is nobody's mixing function.
    assert!(
        dynamic.dim(0) == mix_hc || dynamic.dim(0) == count,
        "`{}` lands {} rows; a {stream_count}-stream mix row is {mix_hc} wide and a trunk \
         collapse row is {count}",
        dynamic.name,
        dynamic.dim(0),
    );
    let mixes = r.fresh(tensor(normed.rows(), dynamic.dim(0), Dtype::F32));
    r.push(
        Elementwise::HcProject {
            normed: normed.id(),
            weight: r.weight(dynamic),
            stream_count,
            mixes: mixes.id(),
        },
        &[normed],
    );
    mixes
}

pub fn hc_gates(
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
        Elementwise::HcGates {
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

/// **THE TRUNK COLLAPSE**: the `M` streams folded into the one `hidden`-wide
/// row the final norm reads, under the `M` sigmoid gates `mixes` (the
/// `[N, M]` row [`hc_project`] lands through `hc_head.fn`) state.
pub fn hc_collapse(
    mixes: &Value,
    streams: &Value,
    scale: &Weight,
    base: &Weight,
    stream_count: u32,
    hc_eps: f32,
) -> Value {
    let r = mixes.rec();
    let count = u64::from(stream_count);
    let y = r.fresh(tensor(
        streams.rows(),
        streams.width() / count,
        streams.dtype(),
    ));
    r.push(
        Elementwise::HcCollapse {
            mixes: mixes.id(),
            streams: streams.id(),
            scale: r.weight(scale),
            base: r.weight(base),
            stream_count,
            hc_eps,
            y: y.id(),
        },
        &[mixes, streams],
    );
    y
}

pub fn hc_fold(x: &Value, streams: &Value, post_mix: &Value, comb_mix: &Value) -> Value {
    let r = x.rec();
    let y = r.fresh(streams.ty().clone());
    r.push(
        Elementwise::HcFold {
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

/// The gated-residual mix ([`Elementwise::HcMix`]): one `hidden`-wide layer
/// input averaged out of `streams` normed residual streams under per-element
/// sigmoid gates. `gates` and `normed` are both `[rows, streams · hidden]`.
pub fn hc_mix(gates: &Value, normed: &Value, streams: u32) -> Value {
    let r = gates.rec();
    let y = r.fresh(tensor(
        normed.rows(),
        normed.width() / u64::from(streams),
        normed.dtype(),
    ));
    r.push(
        Elementwise::HcMix {
            gates: gates.id(),
            normed: normed.id(),
            streams,
            y: y.id(),
        },
        &[gates, normed],
    );
    y
}

/// The gated-residual injection ([`Elementwise::HcInject`]): `hyper[s·H+h] +=
/// 2·σ(gates[s]/streams)·o[h]`, in place on the wide residual. `gates` is
/// `[rows, streams]` of raw logits.
pub fn hc_inject(o: &Value, gates: &Value, streams: u32, hyper: &Value) -> Value {
    let r = o.rec();
    let hyper_out = r.fresh(hyper.ty().clone());
    r.push(
        Elementwise::HcInject {
            o: o.id(),
            gates: gates.id(),
            streams,
            hyper: hyper.id(),
            hyper_out: hyper_out.id(),
        },
        &[o, gates, hyper],
    );
    hyper_out
}

/// The PLE gate ([`Elementwise::PleGate`]): per stream, the n-gram key row
/// dotted with the normed residual stream, signed-square-root damped,
/// squashed, scaling the shared value row. `key` and `query` are
/// `[rows, streams · hidden]`, `value` is `[rows, hidden]`.
pub fn ple_gate(key: &Value, query: &Value, value: &Value, streams: u32) -> Value {
    let r = key.rec();
    let y = r.fresh(key.ty().clone());
    r.push(
        Elementwise::PleGate {
            key: key.id(),
            query: query.id(),
            value: value.id(),
            streams,
            y: y.id(),
        },
        &[key, query, value],
    );
    y
}

/// The multimodal rotary: [`rope_partial`] over a position that is a triple.
/// `positions` is `[rows, 3]` `i32`, one `(t, h, w)` per rotated row;
/// `sections` is the checkpoint's `mrope_section` trace constant. Rotates in
/// place, like every rope arm beside it.
///
/// `form` is the section layout: the trunk states [`MropeForm::Interleaved`],
/// the tower states [`MropeForm::Blocked`].
pub fn rope_mrope(
    q: &Value,
    k: &Value,
    positions: &Value,
    sections: [u32; 3],
    form: MropeForm,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> (Value, Value) {
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    let k_out = r.fresh(k.ty().clone());
    r.push(
        Elementwise::RopeMrope {
            q: q.id(),
            k: k.id(),
            positions: positions.id(),
            sections,
            form,
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
