//! The wafer's decode target: the model's [`Shape`] read off its trace ([`shape`]), the role
//! kernels its layers take ([`super::kernels::KernelSet`]), and its plan ([`plan`]): the search's
//! fastest partition that builds. [`plan_shape`] does the same for a shape stated outright.

use poem_ir::{
    Attn, Def, Elemwise, Gather, GluAct, Guard, Linear, NormAxis, NormWeight, Operation, RopeFreqs,
    RouteScore, Ssm, Trace, Turn, ValueId,
};

use super::compose::BiasOn;
use super::plan::{
    Act, AttnSpec, GdnSpec, HeadSpec, LayerKind, MlpSpec, MoeSpec, NormSpec, Plan, PleSpec, Preset,
    RopeSpec, Shape, YarnSpec,
};

type R<T> = Result<T, String>;

/// What one decoder layer's nodes state.
#[derive(Default)]
struct Layer {
    /// The KV cache the layer appends to (it owns its KV).
    owns: Option<ValueId>,
    /// The cache its attention reads, and the window it reads over.
    reads: Option<(ValueId, Option<u32>)>,
    /// q heads, KV heads and head dim of its attention.
    heads: Option<(u32, u32, u32)>,
    gdn: Option<GdnSpec>,
    glu: Option<(u32, GluAct)>,
    moe: Option<(u32, u32)>,
    row_norms: u32,
    head_norm: bool,
    plus_one: bool,
    gate: bool,
    sink: bool,
    /// Biases: q / k / v, the output projection, the router, the experts' legs.
    bias: bool,
    o_bias: bool,
    router_bias: bool,
    expert_bias: bool,
    /// The experts' banks quantized.
    quantized: bool,
    rope: Option<RopeSpec>,
}

/// Whether a node guarded by `g` runs on the wafer: every request it serves is one token at a
/// time (the token loop).
fn runs(trace: &Trace, g: &Guard) -> bool {
    poem_compiler::spatial::layer::single_token(trace, g)
}

/// The node that makes `v`, if an op does.
fn maker(trace: &Trace, v: ValueId) -> Option<&Operation> {
    match trace.values.get(v.0 as usize).map(|d| &d.def) {
        Some(Def::Op(n)) => trace.nodes.get(*n as usize).map(|n| &n.op),
        _ => None,
    }
}

pub(crate) fn act(a: &GluAct) -> R<Act> {
    match a {
        GluAct::Swiglu => Ok(Act::Silu),
        GluAct::GegluTanh => Ok(Act::GeluTanh),
        GluAct::SwigluAlpha { .. } => Ok(Act::SwigluOai),
        a => Err(format!("wafer: no kernel runs the MLP activation {a:?}")),
    }
}

/// A rotation as the CY's RoPE tables take it: the trace's frequencies kept only where they are
/// not `theta`'s own (a scaled rotation), a proportional rotation's span, YaRN's stretch; no
/// other cos / sin scale.
fn rope(theta: f32, head_dim: u32, turn: &Turn, freqs: Option<&RopeFreqs>) -> R<RopeSpec> {
    let (rotary_dims, span, yarn) = match turn {
        Turn::Partial { rotary_dim, span } => (
            *rotary_dim,
            (*span != rotary_dim / 2).then_some(*span),
            None,
        ),
        Turn::Yarn { yarn, .. } => (
            head_dim,
            None,
            Some(YarnSpec {
                factor: yarn.factor,
                beta_fast: yarn.beta_fast,
                beta_slow: yarn.beta_slow,
                original: yarn.original_max_position,
            }),
        ),
        _ => (head_dim, None, None),
    };
    let freqs = match freqs {
        None => None,
        Some(f) if f.mscale != 1.0 => {
            return Err(format!(
                "wafer: no kernel scales RoPE's cos / sin (mscale {})",
                f.mscale
            ));
        }
        Some(f) => {
            let own = (0..rotary_dims / 2)
                .map(|i| 1.0 / f64::from(theta).powf(f64::from(2 * i) / f64::from(rotary_dims)));
            let same = f.inv_freq.len() == (rotary_dims / 2) as usize
                && f.inv_freq
                    .iter()
                    .zip(own)
                    .all(|(&a, b)| (f64::from(a) - b).abs() <= 1e-6 * b);
            (!same).then(|| f.inv_freq.clone())
        }
    };
    Ok(RopeSpec {
        theta,
        rotary_dims,
        freqs,
        span,
        yarn,
        global: None,
    })
}

/// The model's [`Shape`], read off its trace: the layers (GDN, attention over the whole sequence
/// or a window, or over another layer's KV) and their dims, the MLP and its experts, the per-layer
/// inputs, the head, RoPE and the norms. A node that runs on the wafer and has no phase of the
/// layer program ([`poem_compiler::spatial::layer::lower`]) is refused by name, not dropped.
pub fn shape(trace: &Trace) -> R<Shape> {
    let analysis =
        poem_compiler::analysis::analyze(trace).map_err(|e| format!("{}: {e}", trace.name))?;
    let (mut vocab, mut table, mut hidden) = (0, None, 0);
    let (mut tied, mut softcap, mut eps, mut ple) = (false, None, None, None);
    let mut layers: Vec<Layer> = Vec::new();
    // Every node that runs has a phase, or the trace is refused by its name.
    poem_compiler::spatial::layer::lower(trace, &analysis, &|g| runs(trace, g))?;
    for (at, node) in trace.nodes.iter().enumerate() {
        let Some(l) = analysis.layers[at] else {
            match &node.op {
                // The token embedding: the last plain (unweighted) gather before the first layer.
                // A multimodal trace's encoders run before it and gather their own tables
                // (position embeddings, weighted by their grid taps).
                Operation::Gather(Gather::Embed {
                    table: t,
                    vocab: v,
                    weights: None,
                    ..
                }) if layers.is_empty() => {
                    (vocab, table) = (*v, Some(*t));
                    if let Some(Def::Weight(p)) = trace.values.get(t.0 as usize).map(|d| &d.def) {
                        hidden =
                            trace.params[*p as usize].shape.get(1).copied().unwrap_or(0) as u32;
                    }
                }
                Operation::Linear(Linear::Matmul {
                    w, logits: true, ..
                }) => {
                    tied = Some(*w) == table;
                }
                Operation::Elemwise(Elemwise::Softcap { cap, .. }) => softcap = Some(*cap),
                Operation::Gather(Gather::Select { width, .. }) => ple = Some(*width),
                _ => {}
            }
            continue;
        };
        let l = l as usize;
        if layers.len() <= l {
            layers.resize_with(l + 1, Layer::default);
        }
        let y = &mut layers[l];
        let refuse = |what: String| format!("{}: layer {l}: {what}", trace.name);
        match &node.op {
            Operation::Elemwise(Elemwise::AddBias { bias, .. }) => {
                let name = match trace.values.get(bias.0 as usize).map(|d| &d.def) {
                    Some(Def::Weight(p)) => trace.params[*p as usize].name.as_str(),
                    _ => "",
                };
                match crate::compile::compose::bias_on(name) {
                    Some(BiasOn::Qkv) => y.bias = true,
                    Some(BiasOn::O) => y.o_bias = true,
                    Some(BiasOn::Router) => y.router_bias = true,
                    None => {
                        return Err(refuse(format!(
                            "elemwise.add_bias of {name}: no kernel adds that bias"
                        )));
                    }
                }
            }
            Operation::Linear(Linear::MoeSelect {
                bias, quantized, ..
            }) => {
                y.expert_bias |= bias.is_some();
                y.quantized |= *quantized;
            }
            Operation::Linear(Linear::MoeBiasSum { .. }) => y.expert_bias = true,
            Operation::Linear(Linear::MoeCombine {
                shared: Some(_), ..
            }) => {
                return Err(refuse(
                    "linear.moe_combine with a shared expert: no kernel adds one".into(),
                ));
            }
            Operation::Linear(Linear::MoeRoute {
                ids,
                table,
                bias,
                scale,
                hint,
                score,
                ..
            }) if !matches!(score, RouteScore::Softmax)
                || ids.is_some()
                || table.is_some()
                || bias.is_some()
                || scale.is_some()
                || hint.is_some() =>
            {
                return Err(refuse(format!(
                    "linear.moe_route scored {score:?} (a pick bias, gain, hint or table: {}): \
                     the router kernel runs a softmax over the top k only",
                    bias.is_some() || scale.is_some() || hint.is_some() || table.is_some()
                )));
            }
            Operation::Attn(Attn::KvAppend { cache, .. }) => y.owns = Some(*cache),
            Operation::Attn(Attn::Decode {
                plan,
                cache,
                window,
                ..
            }) => {
                y.reads = Some((*cache, *window));
                if let Some(Operation::Attn(Attn::PlanDecode {
                    q_heads,
                    kv_heads,
                    head_dim,
                    ..
                })) = maker(trace, *plan)
                {
                    y.heads = Some((*q_heads, *kv_heads, *head_dim));
                }
            }
            Operation::Attn(Attn::Sink { .. }) => y.sink = true,
            Operation::Ssm(Ssm::GatedDelta {
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                ..
            }) => {
                y.gdn = Some(GdnSpec {
                    key_heads: *k_heads,
                    value_heads: *v_heads,
                    key_dim: *k_dim,
                    value_dim: *v_dim,
                });
            }
            Operation::Linear(Linear::Glu {
                intermediate, act, ..
            }) if y.glu.is_none() => y.glu = Some((*intermediate, *act)),
            Operation::Linear(Linear::MoeRoute { experts, top_k, .. }) => {
                y.moe = Some((*experts, *top_k));
            }
            Operation::Elemwise(Elemwise::Rmsnorm {
                weight,
                axis,
                eps: e,
                ..
            }) => {
                eps.get_or_insert(*e);
                match axis {
                    NormAxis::Row => {
                        y.row_norms += 1;
                        y.plus_one |= matches!(weight, NormWeight::PlusOne(_));
                    }
                    NormAxis::Head(_) => y.head_norm = true,
                    NormAxis::Group(_) => {}
                }
            }
            Operation::Elemwise(Elemwise::GateSigmoidMul { .. }) => y.gate = true,
            Operation::Elemwise(Elemwise::Rope {
                theta,
                turn,
                head_dim,
                freqs,
                ..
            }) if y.rope.is_none() => {
                y.rope = Some(rope(*theta, *head_dim, turn, freqs.as_ref())?);
            }
            Operation::Gather(Gather::Select { width, .. }) => ple = Some(*width),
            _ => {}
        }
    }
    if vocab == 0 || hidden == 0 || layers.is_empty() {
        return Err(format!("{}: no embedding or no layers", trace.name));
    }
    let owner = |c: ValueId| layers.iter().position(|y| y.owns == Some(c));
    let mut kinds = Vec::with_capacity(layers.len());
    for (l, y) in layers.iter().enumerate() {
        kinds.push(match (y.gdn, y.reads) {
            (Some(_), _) => LayerKind::Gdn,
            (None, Some((cache, window))) if y.owns == Some(cache) => match window {
                Some(window) => LayerKind::Sliding { window },
                None => LayerKind::Global,
            },
            (None, Some((cache, window))) => LayerKind::SharedKv {
                from: owner(cache)
                    .ok_or_else(|| format!("layer {l} reads a cache no layer writes"))?
                    as u32,
                sliding: window.is_some(),
            },
            (None, None) => return Err(format!("layer {l}: no mixer the wafer runs")),
        });
    }
    let attention = |sliding: bool| {
        layers.iter().zip(&kinds).find_map(|(y, k)| {
            let s = matches!(k, LayerKind::Sliding { .. });
            (matches!(k, LayerKind::Sliding { .. } | LayerKind::Global) && s == sliding)
                .then_some(y.heads)
                .flatten()
        })
    };
    let (q_heads, kv_heads, head_dim) = attention(true)
        .or(attention(false))
        .ok_or("wafer: no attention layer")?;
    let global_head_dim = attention(true)
        .and(attention(false))
        .map(|(_, _, hd)| hd)
        .filter(|&hd| hd != head_dim);
    let global_kv_heads = attention(true)
        .and(attention(false))
        .map(|(_, kv, _)| kv)
        .filter(|&kv| kv != kv_heads);
    let first = &layers[0];
    let (intermediate, glu) = first.glu.ok_or("wafer: layer 0 has no MLP")?;
    let moe = first.moe.map(|(experts, top_k)| MoeSpec {
        experts,
        top_k,
        expert_intermediate: intermediate,
        router_bias: layers.iter().any(|y| y.router_bias),
        expert_bias: layers.iter().any(|y| y.expert_bias),
        quantized: layers.iter().any(|y| y.quantized),
    });
    let mut rope = layers
        .iter()
        .find_map(|y| y.rope.clone())
        .unwrap_or(RopeSpec {
            theta: 0.0,
            rotary_dims: head_dim,
            freqs: None,
            span: None,
            yarn: None,
            global: None,
        });
    rope.global = layers
        .iter()
        .zip(&kinds)
        .find_map(|(y, k)| {
            matches!(k, LayerKind::Global)
                .then(|| y.rope.clone())
                .flatten()
        })
        .filter(|g| *g != rope)
        .map(Box::new);
    Ok(Shape {
        hidden,
        vocab,
        attn: AttnSpec {
            q_heads,
            kv_heads,
            head_dim,
            global_head_dim,
            global_kv_heads,
            output_gate: layers.iter().any(|y| y.gate),
            qk_norm: layers.iter().any(|y| y.head_norm),
            sinks: layers.iter().any(|y| y.sink),
            bias: layers.iter().any(|y| y.bias),
            o_bias: layers.iter().any(|y| y.o_bias),
        },
        mlp: MlpSpec {
            intermediate,
            act: act(&glu)?,
            layer_widths: layers
                .iter()
                .map(|y| y.glu.as_ref().map_or(intermediate, |g| g.0))
                .collect(),
        },
        moe,
        gdn: layers.iter().find_map(|y| y.gdn),
        ple: ple.map(|width| PleSpec { width }),
        head: HeadSpec {
            tied,
            final_softcap: softcap,
        },
        rope,
        norm: NormSpec {
            eps: eps.unwrap_or(1e-6),
            unit_offset: layers.iter().any(|y| y.plus_one),
            sandwich: layers.iter().any(|y| y.row_norms >= 4),
        },
        layers: kinds,
    })
}

/// The plan chosen, with the preset (overrides set) it came from.
#[derive(Clone, Debug)]
pub struct Planned {
    pub plan: Plan,
    pub preset: Preset,
}

/// The plan of `trace`: its shape, then the search's, with the overrides and positions `ask`
/// states.
pub fn plan(trace: &Trace, ask: &Ask) -> R<Planned> {
    plan_shape(shape(trace)?, ask)
}

/// The plan of a shape stated outright, chosen as [`plan`] chooses: the search's fastest
/// partition ([`crate::compile::search`]).
pub fn plan_shape(shape: Shape, ask: &Ask) -> R<Planned> {
    Preset::default().set(&ask.set)?;
    let asked = crate::compile::search::Asked {
        positions: ask.positions,
        set: ask.set.clone(),
    };
    crate::compile::search::search(&shape, &asked, 1).map(|mut f| f.remove(0).planned)
}

/// What a plan is asked for beyond the model: overrides of its keys (`K=V,...`) and the positions
/// it must serve.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ask {
    pub set: String,
    pub positions: Option<u32>,
}

/// The shipping models' deployments.
#[cfg(test)]
pub(crate) const SHIPPING: [&str; 9] = [
    "qwen35-d0.8b-bf16-kv-bf16",
    "qwen35-d4b-bf16-kv-bf16",
    "qwen35-d9b-bf16-kv-bf16",
    "qwen3-8b-bf16-kv-bf16",
    "llama31-8b-bf16-kv-bf16",
    "gemma4-e2b-bf16-kv-bf16",
    "gemma4-e4b-bf16-kv-bf16",
    "gemma4-12b-bf16-kv-bf16",
    "gptoss-20b-bf16-mxfp4-kv-bf16",
];

/// The search's plan of catalog deployment `name`, searched once a test run.
#[cfg(test)]
pub(crate) fn searched(name: &str) -> Planned {
    use std::sync::{Arc, Mutex, OnceLock};
    type Cells = Mutex<std::collections::BTreeMap<String, Arc<OnceLock<Planned>>>>;
    static CELLS: OnceLock<Cells> = OnceLock::new();
    let cell = CELLS
        .get_or_init(Cells::default)
        .lock()
        .unwrap()
        .entry(name.to_string())
        .or_default()
        .clone();
    cell.get_or_init(|| {
        let t = poem_compiler::catalog::repository()
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"))
            .trace(poem_ir::Platform::Cerebras);
        plan(&t, &Ask::default()).unwrap_or_else(|e| panic!("{name}: {e}"))
    })
    .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::kernels::KernelSet;
    use crate::compile::program::Program;

    fn trace(name: &str) -> Trace {
        poem_compiler::catalog::repository()
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"))
            .trace(poem_ir::Platform::Cerebras)
    }

    /// A multimodal deployment (what `pie serve` identifies from a checkpoint with its vision
    /// tower) plans the language model's program: its encoder's tables and dims are not the
    /// model's.
    #[test]
    fn a_vision_deployment_plans_its_language_model() {
        let want = searched("gemma4-e4b-bf16-kv-bf16").plan;
        let got = searched("gemma4-e4b-vision-bf16-kv-bf16").plan;
        assert_eq!(
            serde_json::to_value(&got).unwrap(),
            serde_json::to_value(&want).unwrap()
        );
    }

    /// Every shipping model plans from its trace, on its kernel set, its head holding the
    /// vocabulary.
    #[test]
    fn shipping_models_plan_from_their_traces() {
        for (name, set) in [
            ("qwen35-d9b-bf16-kv-bf16", KernelSet::Own),
            ("qwen35-d4b-bf16-kv-bf16", KernelSet::Own),
            ("qwen35-d0.8b-bf16-kv-bf16", KernelSet::Own),
            ("qwen3-8b-bf16-kv-bf16", KernelSet::Own),
            ("gemma4-e2b-bf16-kv-bf16", KernelSet::Shared),
            ("gemma4-e4b-bf16-kv-bf16", KernelSet::Shared),
            ("gemma4-12b-bf16-kv-bf16", KernelSet::Windows),
            ("gptoss-20b-bf16-mxfp4-kv-bf16", KernelSet::Experts),
            ("llama31-8b-bf16-kv-bf16", KernelSet::Own),
        ] {
            let p = searched(name).plan;
            assert_eq!(KernelSet::of_shape(&p.shape), set, "{name}");
            let h = &p.partition.head;
            assert!(
                u64::from(h.rows.max(h.cols)) * u64::from(h.vocab_rows) >= u64::from(p.shape.vocab),
                "{name}"
            );
            assert_eq!(p.contract.positions, 4096, "{name}");
        }
    }

    /// Llama 3.1 8B: dense attention with no q / k norm, its rotation Llama 3's (the long waves
    /// stretched: frequencies of their own), eps 1e-5.
    #[test]
    fn llama_plans_from_its_trace() {
        let p = searched("llama31-8b-bf16-kv-bf16").plan;
        let s = &p.shape;
        assert_eq!(KernelSet::of_shape(&p.shape), KernelSet::Own);
        assert!(!s.attn.qk_norm && !s.attn.output_gate && s.gdn.is_none());
        assert_eq!(
            (s.attn.q_heads, s.attn.kv_heads, s.attn.head_dim),
            (32, 8, 128)
        );
        assert_eq!(s.norm.eps, 1e-5);
        let f = s.rope.freqs.as_ref().expect("llama3's frequencies");
        assert_eq!(f.len(), 64);
        assert_eq!(f[0], 1.0);
        assert!(f[63] < 500_000f32.powf(-126.0 / 128.0) / 7.0);
    }

    /// The positions asked for bound the plan: the RoPE tables' levels follow them, and more than
    /// any partition holds find none.
    #[test]
    fn positions_bound_the_plan() {
        let p = searched("qwen35-d0.8b-bf16-kv-bf16").plan;
        assert_eq!((p.contract.positions, p.contract.rope_levels), (4096, 4));
        let ask = Ask {
            positions: Some(1 << 20),
            ..Ask::default()
        };
        let e = plan(&trace("qwen35-d0.8b-bf16-kv-bf16"), &ask).unwrap_err();
        assert!(e.contains("the search found no partition"), "{e}");
    }

    /// A GDN layer's state row k lives on weight row k: Qwen3.5-0.8B's search keeps a key head's
    /// 128 rows, and 64 rows (the chip's wrong tokens, 1603 / 1606) find no partition.
    #[test]
    fn a_gdn_unit_holds_every_key_row() {
        let p = searched("qwen35-d0.8b-bf16-kv-bf16").plan;
        assert!(p.partition.rows >= p.shape.gdn.unwrap().key_dim);
        let ask = Ask {
            set: "PY=64".into(),
            ..Ask::default()
        };
        let e = plan(&trace("qwen35-d0.8b-bf16-kv-bf16"), &ask).unwrap_err();
        assert!(e.contains("the search found no partition"), "{e}");
    }

    /// The shared kernels' weight image links to more code where its KV rows are not a power of
    /// two and its sliding ring masks the window: Gemma 4 E2B's 96 rows (1280 B of room by the
    /// tiles alone, 128 B as linked) find no partition, and its search keeps 128 rows.
    #[test]
    fn a_shared_weight_image_counts_its_code_paths() {
        let p = searched("gemma4-e2b-bf16-kv-bf16").plan;
        assert!(
            p.partition.kv_rows.is_power_of_two(),
            "{}",
            p.partition.kv_rows
        );
        let ask = Ask {
            set: "PY=96,NEG=43".into(),
            ..Ask::default()
        };
        let e = plan(&trace("gemma4-e2b-bf16-kv-bf16"), &ask).unwrap_err();
        assert!(e.contains("the search found no partition"), "{e}");
    }

    /// A head whose vocabulary groups hold fewer ids than the vocabulary is refused.
    #[test]
    fn a_head_short_of_the_vocabulary_is_refused() {
        let mut p = searched("qwen35-d0.8b-bf16-kv-bf16").plan;
        p.validate().unwrap();
        p.partition.head.rows = p.shape.vocab / p.partition.head.vocab_rows;
        let e = p.validate().unwrap_err();
        assert!(e.contains("holds less than the"), "{e}");
    }

    /// A partition is refused where it cannot link or serve: wider than the wafer, a head's dims
    /// not even over its columns; the search, moving the keys, finds none for an MLP no column
    /// count of the snake divides.
    #[test]
    fn fit_refuses_what_cannot_serve() {
        let t = trace("llama31-8b-bf16-kv-bf16");
        let s = shape(&t).unwrap();
        let at = searched("llama31-8b-bf16-kv-bf16").preset;
        let refused = |set: &str| {
            crate::compile::floorplan::plan(&s, &at.clone().set(set).unwrap())
                .and_then(|p| Program::build(p).map(|_| ()))
                .unwrap_err()
        };
        let e = refused("PX=320");
        assert!(e.contains("dims over 40 columns"), "{e}");
        let e = refused("PX=288");
        assert!(e.contains("dims over 36 columns"), "{e}");
        let e = refused("PX=512");
        assert!(e.contains("over the wafer"), "{e}");
        let mut odd = s.clone();
        odd.mlp.intermediate += 64;
        let e = plan_shape(odd, &Ask::default()).unwrap_err();
        assert!(e.contains("the search found no partition"), "{e}");
    }

    /// The memory model, anchored on its kernel set's linked program, predicts another of the
    /// same kernels' within the search's margin: Qwen3.5-4B's weight and head images.
    #[test]
    fn memory_follows_the_partition_within_the_margin() {
        use crate::compile::cost::memory;
        // the 4B's partition as it was linked
        let mut at = Preset::new(
            "4b",
            &[
                ("G2", 16.0),
                ("G2Y", 16.0),
                ("HR", 1153.0),
                ("K", 4.0),
                ("NE", 64.0),
                ("PX", 256.0),
                ("PY", 256.0),
                ("UNX", 2.0),
                ("UNY", 4.0),
                ("VR", 216.0),
                ("WL", 1.0),
            ],
        );
        at.levers = "NPX:1,DIAG:0,TSTAMP:0,XKEEP:0,HACC16:0,ATTL:2".into();
        let s = shape(&trace("qwen35-d4b-bf16-kv-bf16")).unwrap();
        let p = crate::compile::floorplan::plan(&s, &at).unwrap();
        // the 4B as linked: weight 34,848 B, head 27,968 B, every other image's room 21,568 B
        let want = (memory::PE_BYTES - 34848.0)
            .min(memory::PE_BYTES - 27968.0)
            .min(21568.0);
        let got = memory::room(&p, memory::anchor(&s)).min(21568.0);
        assert!(
            (got - want).abs() < crate::compile::search::MARGIN,
            "{got} vs {want}"
        );
    }
}
