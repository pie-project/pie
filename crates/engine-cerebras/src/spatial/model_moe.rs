//! A routed-MoE model's spec (gpt-oss): what [`ModelSpec::read`] keeps of a
//! trace with `linear.moe_*` ops, instead of refusing it.
//!
//! The decoder layer read here is gpt-oss's: pre-RMSNorm GQA attention with
//! q/k/v and o biases, no q/k norm, RoPE in neox halves (YaRN's frequencies
//! where stated), a softmax sink a query head, every layer full or over a
//! sliding window (the attention ops carry the window; the cache rows do
//! not); then a routed MoE: a biased router, top-k of the experts and a
//! softmax over them, per-expert gate/up (packed, biased) and down banks
//! (MXFP4 codes and E8M0 scales, or one dense dtype), gpt-oss's clamped
//! SwiGLU, the routed rows combined by weight and the down biases summed by
//! route (`MoeBiasSum`). Anything else is refused with the first op that
//! differs.

use poem_ir::{
    Attn, CacheRow, Def, Dtype, Elemwise, Gather, GluAct, Linear, NormWeight, Operands, Operation,
    RouteScore, Trace, Turn, ValueId, Yarn,
};

use super::model::{ModelSpec, Refused, refuse};

/// A routed MoE layer stack's facts, beyond [`ModelSpec`]'s.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeModel {
    pub experts: usize,
    pub top_k: usize,
    /// How the router scores and weighs its picks (gpt-oss: the top-k's
    /// softmax).
    pub routing: RouteScore,
    /// The gate's activation and the clamp on gate and up.
    pub act: GluAct,
    pub limit: Option<f32>,
    /// The experts' banks as stored (`Mxfp4`: codes and E8M0 scales).
    pub expert_dtype: Dtype,
    pub router_bias: bool,
    pub gate_up_bias: bool,
    /// Per-expert down biases, summed by route weight (`MoeBiasSum`).
    pub down_bias: bool,
    /// q/k/v projection biases and the o projection's.
    pub qkv_bias: bool,
    pub o_bias: bool,
    /// A softmax sink a query head (`Attn::Sink`).
    pub sinks: bool,
    /// Each layer's sliding window (`None`: full causal).
    pub windows: Vec<Option<u32>>,
    /// YaRN's frequency ramp and its cos/sin scale, where the rope states it.
    pub yarn: Option<(Yarn, f32)>,
}

impl MoeModel {
    /// Layers over a window, and the widest window.
    pub fn windowed(&self) -> (usize, u32) {
        let w: Vec<u32> = self.windows.iter().flatten().copied().collect();
        (w.len(), w.iter().copied().max().unwrap_or(0))
    }
}

#[derive(Default)]
struct LayerFacts {
    norms: usize,
    eps: Option<f32>,
    qkv: Option<(usize, usize)>,
    head_dim: Option<usize>,
    biases: usize,
    window: Option<Option<u32>>,
    sink: bool,
    route: Option<(usize, usize, RouteScore)>,
    glu: Option<(usize, GluAct, Option<f32>)>,
    selects: usize,
    combine: bool,
    bias_sum: bool,
    router_bias: bool,
    gate_up_bias: bool,
    expert_dtype: Option<Dtype>,
}

/// Reads a routed-MoE trace (the caller saw a `linear.moe_route`).
pub fn read(trace: &Trace) -> Result<ModelSpec, Refused> {
    let name = &trace.name;
    let param = |v: ValueId| match trace.values.get(v.0 as usize).map(|x| &x.def) {
        Some(Def::Weight(i)) => trace.params.get(*i as usize),
        _ => None,
    };
    let mut layers: Vec<LayerFacts> = Vec::new();
    let (mut vocab, mut hidden, mut theta, mut yarn, mut sm_scale) = (0, 0, 0.0f32, None, 0.0f32);
    let (mut weights, mut head_tied, mut final_norm, mut heads) = (None, None, 0, None);
    for node in &trace.nodes {
        let at = |why: &str| {
            let l = node.layer.map_or(String::new(), |l| format!("layer {l}: "));
            refuse::<()>(format!("{name}: {l}{}: {why}", node.op.name()))
        };
        let Some(l) = node.layer else {
            match &node.op {
                Operation::Gather(Gather::Embed {
                    table, vocab: v, ..
                }) => {
                    vocab = *v as usize;
                    if let Some(p) = param(*table) {
                        hidden = p.shape[1] as usize;
                        weights = Some(p.dtype);
                    }
                }
                Operation::Elemwise(Elemwise::Rmsnorm { .. }) => final_norm += 1,
                Operation::Gather(Gather::Rows { .. }) => {}
                Operation::Attn(Attn::PlanPrefill { .. } | Attn::PlanDecode { .. }) => {}
                Operation::Linear(Linear::Matmul {
                    w, logits: true, ..
                }) => {
                    head_tied = param(*w).map(|p| p.name == "embed");
                }
                _ => at("outside the decoder layers")?,
            }
            continue;
        };
        if layers.len() <= l as usize {
            layers.resize_with(l as usize + 1, LayerFacts::default);
        }
        let f = &mut layers[l as usize];
        match &node.op {
            Operation::Elemwise(Elemwise::Rmsnorm { weight, eps, .. }) => {
                if !matches!(weight, NormWeight::Scale(_)) {
                    at("a norm weight other than w")?;
                }
                if f.eps.is_some_and(|e| e != *eps) {
                    at("RMSNorm epsilons differ")?;
                }
                f.eps = Some(*eps);
                f.norms += 1;
            }
            Operation::Linear(Linear::Matmul { .. }) => {}
            Operation::Elemwise(Elemwise::AddBias { .. }) => f.biases += 1,
            Operation::Gather(Gather::SplitQkv {
                q_width, kv_width, ..
            }) => {
                f.qkv = Some((*q_width as usize, *kv_width as usize));
            }
            Operation::Elemwise(Elemwise::Rope {
                head_dim,
                theta: t,
                turn,
                freqs,
                ..
            }) => {
                if freqs.is_some() {
                    at("a stated frequency table")?;
                }
                match turn {
                    Turn::Full { interleaved: false } => {}
                    Turn::Yarn {
                        yarn: y,
                        attention_factor,
                        interleaved: false,
                    } => {
                        yarn = Some((*y, *attention_factor));
                    }
                    _ => at(&format!(
                        "a {turn:?} rotation (the kernels turn neox halves)"
                    ))?,
                }
                f.head_dim = Some(*head_dim as usize);
                theta = *t;
            }
            Operation::Attn(Attn::KvAppend { .. }) => {}
            Operation::Attn(Attn::Prefill {
                window,
                causal,
                sm_scale: s,
                kv_heads,
                bias,
                selection,
                ..
            }) => {
                if !causal || bias.is_some() || selection.is_some() {
                    at("non-causal, biased or selected attention")?;
                }
                if f.window.is_some_and(|w| w != *window) {
                    at("a layer's attention reads two windows")?;
                }
                f.window = Some(*window);
                sm_scale = *s;
                heads = Some(*kv_heads as usize);
            }
            Operation::Attn(Attn::Decode {
                window,
                bias,
                selection,
                ..
            }) => {
                if bias.is_some() || selection.is_some() {
                    at("biased or selected attention")?;
                }
                if f.window.is_some_and(|w| w != *window) {
                    at("a layer's attention reads two windows")?;
                }
                f.window = Some(*window);
            }
            Operation::Attn(Attn::Sink { .. }) => f.sink = true,
            // A full layer's scores are read out for score-reading inferlets
            // (the engine refuses those reads on its own).
            Operation::Attn(Attn::ScoreCapture { .. }) => {}
            Operation::Linear(Linear::LoraCorrect { .. }) => {}
            Operation::Elemwise(Elemwise::ResidualAdd { gate: None, .. }) => {}
            Operation::Linear(Linear::MoeRoute {
                ids: None,
                table: None,
                bias,
                scale: None,
                hint: None,
                experts,
                top_k,
                score,
                ..
            }) => {
                f.router_bias = f.biases >= 3 || bias.is_some();
                f.route = Some((*experts as usize, *top_k as usize, *score));
            }
            Operation::Linear(Linear::MoeSelect {
                bank,
                bias,
                input_scales: None,
                quant: None,
                ..
            }) => {
                if f.selects == 0 {
                    f.gate_up_bias = bias.is_some();
                }
                let dt = param(*bank).map(|p| p.dtype);
                if f.expert_dtype.is_some() && f.expert_dtype != dt {
                    at("the expert banks' dtypes differ")?;
                }
                f.expert_dtype = dt;
                f.selects += 1;
            }
            Operation::Linear(Linear::Glu {
                packed: Some(_),
                weights: None,
                intermediate,
                act,
                limit,
                ..
            }) => {
                f.glu = Some((*intermediate as usize, *act, *limit));
            }
            Operation::Linear(Linear::MoeCombine {
                weights: Some(_),
                shared: None,
                ..
            }) => f.combine = true,
            Operation::Linear(Linear::MoeBiasSum { .. }) => f.bias_sum = true,
            _ => at("an op the MoE layer reading does not take")?,
        }
    }
    if layers.is_empty() {
        return refuse(format!("{name}: no decoder layers"));
    }
    if final_norm != 1 || head_tied.is_none() {
        return refuse(format!("{name}: wanted one final norm and an lm_head"));
    }
    let first = &layers[0];
    let bad = |l: usize, why: &str| refuse::<ModelSpec>(format!("{name}: layer {l}: {why}"));
    for (l, f) in layers.iter().enumerate() {
        if f.norms != 2 || f.eps != first.eps {
            return bad(l, "wanted two RMSNorms of one epsilon");
        }
        if f.qkv.is_none()
            || f.qkv != first.qkv
            || f.head_dim != first.head_dim
            || f.window.is_none()
        {
            return bad(
                l,
                "wanted a packed qkv projection, RoPE and attention like layer 0's",
            );
        }
        if f.route.is_none()
            || f.route != first.route
            || f.glu != first.glu
            || f.selects != 2
            || !f.combine
        {
            return bad(
                l,
                "wanted a routed MoE like layer 0's (route, gate/up, down, combine)",
            );
        }
        if f.sink != first.sink || f.bias_sum != first.bias_sum || f.biases != first.biases {
            return bad(l, "sinks or biases differ from layer 0's");
        }
    }
    let (q_width, kv_width) = first.qkv.unwrap();
    let head_dim = first.head_dim.unwrap();
    let (experts, top_k, routing) = first.route.unwrap();
    let (inter, act, limit) = first.glu.unwrap();
    if !matches!(routing, RouteScore::Softmax) {
        return refuse(format!(
            "{name}: {routing:?} routing (the MoE unit takes the top-k's softmax)"
        ));
    }
    if !matches!(act, GluAct::SwigluAlpha { .. } | GluAct::Swiglu) {
        return refuse(format!("{name}: a {act:?} MoE (the unit runs SwiGLU)"));
    }
    // Biases a layer: qkv, o, the router's (gpt-oss: three AddBias ops).
    let qkv_bias = first.biases >= 2;
    let kv = trace.caches.iter().find_map(|c| match c {
        CacheRow::Kv { dtype, .. } => Some(*dtype),
        _ => None,
    });
    let Some(kv) = kv else {
        return refuse(format!("{name}: no KV cache"));
    };
    Ok(ModelSpec {
        name: name.clone(),
        layers: layers.len(),
        hidden,
        q_heads: q_width / head_dim,
        kv_heads: heads.unwrap_or(kv_width / head_dim),
        head_dim,
        inter,
        vocab,
        tied: head_tied == Some(true),
        qk_norm: false,
        rope_theta: theta,
        rope_table: None,
        rms_eps: first.eps.unwrap_or(0.0),
        sm_scale,
        weights: weights.unwrap_or(Dtype::Bf16),
        kv,
        norm_plus_one: false,
        hybrid: None,
        moe: Some(MoeModel {
            experts,
            top_k,
            routing,
            act,
            limit,
            expert_dtype: first.expert_dtype.unwrap_or(Dtype::Bf16),
            router_bias: first.router_bias,
            gate_up_bias: first.gate_up_bias,
            down_bias: first.bias_sum,
            qkv_bias,
            o_bias: qkv_bias,
            sinks: first.sink,
            windows: layers.iter().map(|f| f.window.unwrap()).collect(),
            yarn,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use poem_ir::Platform;

    fn spec(name: &str) -> Result<ModelSpec, Refused> {
        let t = poem_compiler::catalog::repository()
            .parse(name)
            .unwrap()
            .trace(Platform::Cerebras);
        ModelSpec::read(&t)
    }

    /// gpt-oss-20b reads as a routed MoE: 32 experts top 4 in MXFP4, biases,
    /// sinks, the even layers over a window of 128, YaRN.
    #[test]
    fn gptoss_20b_reads_as_a_moe() {
        let s = spec("gptoss-20b-bf16-mxfp4-kv-bf16").unwrap();
        assert_eq!(
            (s.layers, s.hidden, s.q_heads, s.kv_heads, s.head_dim),
            (24, 2880, 64, 8, 64)
        );
        assert_eq!(
            (s.inter, s.vocab, s.tied, s.qk_norm),
            (2880, 201088, false, false)
        );
        assert_eq!(
            (s.rms_eps, s.sm_scale, s.rope_theta),
            (1e-5, 0.125, 150000.0)
        );
        let m = s.moe.as_ref().unwrap();
        assert_eq!((m.experts, m.top_k, m.expert_dtype), (32, 4, Dtype::Mxfp4));
        assert_eq!(m.act, GluAct::SwigluAlpha { alpha: 1.702 });
        assert_eq!(m.limit, Some(7.0));
        assert!(m.router_bias && m.gate_up_bias && m.down_bias && m.qkv_bias && m.sinks);
        assert_eq!(m.windowed(), (12, 128));
        assert_eq!(m.windows[0], Some(128));
        assert_eq!(m.windows[1], None);
        let (y, af) = m.yarn.unwrap();
        assert_eq!(
            (y.factor, y.original_max_position, af),
            (32.0, 4096, 1.3465736)
        );
    }

    /// No program places a MoE yet: every program refuses it.
    #[test]
    fn a_moe_wants_its_program() {
        let s = spec("gptoss-20b-mini-bf16-mxfp4-kv-bf16").unwrap();
        assert_eq!((s.layers, s.moe.as_ref().unwrap().experts), (5, 16));
        let err = s
            .check_program(|k| (k == "HEADMODE").then_some(2))
            .unwrap_err();
        assert!(err.0.contains("mixture of experts"), "{err}");
    }
}
