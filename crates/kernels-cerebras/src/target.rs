//! The wafer as a target of poem's spatial pass ([`poem_compiler::spatial`]): the model's
//! [`Shape`] read off its trace ([`shape`]), the layout its layers take
//! ([`super::compose::Layout`]), the pinned plan named or made for the shape ([`super::pins`]) as
//! the partition to try, a plan that builds ([`Program::build`]) as the fit; a model with no
//! pinned plan that builds plans by the search ([`super::search`]). [`plan`] runs the pass over a
//! trace; [`plan_shape`] does the same for a shape stated outright (a checkpoint with no catalog
//! deployment, such as the tiny test checkpoints).

use poem_compiler::spatial::{BlockKind, SpatialGraph, SpatialTarget};
use poem_ir::{
    Attn, Def, Elemwise, Gather, GluAct, Guard, Linear, NormAxis, NormWeight, Operation, RopeFreqs,
    RouteScore, Ssm, Trace, Turn, ValueId,
};

use super::compose::{BiasOn, Layout};
use super::plan::{
    Act, AttnSpec, GdnSpec, HeadSpec, LayerKind, MlpSpec, MoeSpec, NormSpec, Plan, PleSpec, Preset,
    RopeSpec, Shape, YarnSpec,
};
use super::program::Program;

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
                match crate::compose::bias_on(name) {
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

/// The plan the pass chose, with the preset (overrides set) it came from.
#[derive(Clone, Debug)]
pub struct Planned {
    pub plan: Plan,
    pub preset: Preset,
}

/// A partition to try: a pinned plan's preset (the `order`-th tried), with the overrides set.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub layout: Layout,
    pub preset: Preset,
    pub order: usize,
}

/// The wafer target for one shape: the pinned plan asked for (else the one made for it) and the
/// overrides set over it.
pub struct Wafer {
    pub shape: Shape,
    pub preset: Option<String>,
    pub set: String,
    /// The positions the program must serve (the serving context); `None`: as many as the
    /// preset holds.
    pub positions: Option<u32>,
}

impl Wafer {
    /// The partitions to try: the pinned plan named (it must be of the shape's layout), else the
    /// one made for this shape ([`crate::pins`]); none for a model with no pinned plan (the
    /// search plans it).
    pub fn candidates(&self) -> Vec<Candidate> {
        let layout = Layout::of_shape(&self.shape);
        let sig = self.shape.signature();
        crate::pins::of(layout)
            .into_iter()
            .filter(|p| match &self.preset {
                Some(n) => *n == p.name,
                None => p.made_for == Some(sig),
            })
            .enumerate()
            .filter_map(|(order, p)| {
                let mut preset = p.set(&self.set).ok()?;
                if let Some(n) = self.positions {
                    preset.keys.insert("CTX".into(), f64::from(n));
                }
                Some(Candidate {
                    layout,
                    preset,
                    order,
                })
            })
            .collect()
    }

    /// The plan `c` makes of the shape, serving exactly the positions asked for (if any).
    pub fn plan_of(&self, c: &Candidate) -> R<Plan> {
        let mut plan = super::partition::plan(&self.shape, &c.preset)?;
        if let Some(n) = self.positions {
            let held = plan.contract.positions;
            if n > held {
                return Err(format!(
                    "preset {}: {n} positions, of at most {held}",
                    c.preset.name
                ));
            }
            plan.contract.positions = n;
        }
        // the head's rows (or its band's columns) hold every vocabulary row: the cut pads the
        // tables to what they hold, so a token past it would have no logit and no embedding
        let h = &plan.partition.head;
        let held = u64::from(h.rows.max(h.cols)) * u64::from(h.vocab_rows);
        if held < u64::from(self.shape.vocab) {
            return Err(format!(
                "preset {}: the head's {} rows of {} vocabulary rows hold {held} of {} tokens",
                c.preset.name,
                h.rows.max(h.cols),
                h.vocab_rows,
                self.shape.vocab
            ));
        }
        // the images leave the stack its room, as the memory model measures them
        let room = super::cost::memory::room(&plan, super::cost::memory::anchor(&self.shape));
        if room < f64::from(super::build::STACK_ROOM) {
            return Err(format!(
                "preset {}: the images leave {room:.0} B of stack room, under {} B",
                c.preset.name,
                super::build::STACK_ROOM
            ));
        }
        Ok(plan)
    }

    /// `c` with the plan it makes.
    fn planned(&self, c: Candidate) -> R<Planned> {
        Ok(Planned {
            plan: self.plan_of(&c)?,
            preset: c.preset,
        })
    }

    /// Whether `c`'s plan builds, or why not.
    pub fn fit(&self, c: &Candidate) -> R<()> {
        let plan = self.plan_of(c)?;
        Program::build(plan)
            .map(|_| ())
            .map_err(|e| format!("preset {}: {e}", c.preset.name))
    }

    /// The pinned plan that builds; with no preset named and none that builds (or none made for
    /// the shape), the search's fastest partition ([`crate::search`]).
    pub fn choose(&self) -> R<Planned> {
        let mut why = Vec::new();
        for c in self.candidates() {
            match self.fit(&c) {
                Ok(()) => return self.planned(c),
                Err(e) => why.push(e),
            }
        }
        self.searched(why)
    }

    /// With no preset named, the search's fastest partition ([`crate::search`]); else (or if the
    /// search finds none) the refusal of `why`.
    fn searched(&self, mut why: Vec<String>) -> R<Planned> {
        if self.preset.is_none() {
            let ask = crate::search::Asked {
                positions: self.positions,
                set: self.set.clone(),
            };
            match crate::search::search(&self.shape, &ask, 1) {
                Ok(mut found) => return Ok(found.remove(0).planned),
                Err(e) => why.push(e),
            }
        }
        Err(self.refused(why))
    }

    fn refused(&self, why: Vec<String>) -> String {
        match (&self.preset, why.is_empty()) {
            (Some(p), true) => match crate::pins::named(p) {
                Some((l, _)) => format!(
                    "the pinned plan {p} is a {} plan; this model's layers take the {} layout",
                    l.name(),
                    Layout::of_shape(&self.shape).name()
                ),
                None => format!("no pinned plan {p}"),
            },
            (_, true) => "no pinned plan to try".into(),
            _ => format!("no pinned plan builds: {}", why.join("; ")),
        }
    }
}

impl SpatialTarget for Wafer {
    type Partition = Candidate;

    /// Every node is the wafer's: the program lays out the whole model (its blocks are the
    /// layout's, chosen from the shape).
    fn classify(&self, _trace: &Trace, _node: u32) -> Option<BlockKind> {
        Some(BlockKind("wafer"))
    }

    fn candidates(&self, _trace: &Trace, _graph: &SpatialGraph) -> Vec<Candidate> {
        Wafer::candidates(self)
    }

    fn fit(&self, _graph: &SpatialGraph, c: &Candidate) -> R<()> {
        Wafer::fit(self, c)
    }

    fn cost(&self, _graph: &SpatialGraph, c: &Candidate) -> f64 {
        c.order as f64
    }
}

/// The plan of `trace` through poem's spatial pass: its shape, then the pinned plan `ask` names
/// (else the one made for its shape, else the search's) with its overrides and positions.
pub fn plan(trace: &Trace, ask: &Ask) -> R<Planned> {
    let wafer = ask.target(shape(trace)?)?;
    match poem_compiler::spatial::plan(trace, &wafer) {
        Ok(p) => wafer.planned(p.partition),
        Err(poem_compiler::spatial::SpatialError::Unfit(why)) => wafer.searched(why),
        Err(poem_compiler::spatial::SpatialError::NoCandidate) => wafer.searched(Vec::new()),
        Err(e) => Err(format!("{}: {e}", trace.name)),
    }
}

/// Every plan of `trace` that builds, in the order [`plan`] prefers them (its choice first): a
/// load links the next where one's program does not fit its PEs' memory.
pub fn plans(trace: &Trace, ask: &Ask) -> R<Vec<Planned>> {
    let wafer = ask.target(shape(trace)?)?;
    let mut why = Vec::new();
    let mut fit = Vec::new();
    for c in wafer.candidates() {
        match wafer.fit(&c) {
            Ok(()) => fit.push(wafer.planned(c)?),
            Err(e) => why.push(e),
        }
    }
    if fit.is_empty() {
        return wafer.searched(why).map(|p| vec![p]);
    }
    Ok(fit)
}

/// The plan of a shape stated outright, chosen as [`plan`] chooses.
pub fn plan_shape(shape: Shape, ask: &Ask) -> R<Planned> {
    ask.target(shape)?.choose()
}

/// What a plan is asked for beyond the model: a pinned plan by name (else the one made for the
/// shape, else the search's), overrides of its keys (`K=V,...`), and the positions it must serve.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ask {
    pub preset: Option<String>,
    pub set: String,
    pub positions: Option<u32>,
}

impl Ask {
    fn target(&self, shape: Shape) -> R<Wafer> {
        Preset::default().set(&self.set)?;
        super::compose::refuse(&shape)?;
        Ok(Wafer {
            shape,
            preset: self.preset.clone(),
            set: self.set.clone(),
            positions: self.positions,
        })
    }

    /// A named preset, nothing else.
    pub fn preset(name: &str) -> Ask {
        Ask {
            preset: Some(name.into()),
            ..Ask::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(name: &str) -> Trace {
        poem_compiler::catalog::repository()
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"))
            .trace(poem_ir::Platform::Cerebras)
    }

    /// An op no kernel runs is refused by name, not dropped: DeepSeek V4's attention.
    #[test]
    fn an_op_without_a_lowering_is_refused_by_name() {
        let e = shape(&trace("dsv4-flash-mini-bf16-kv-bf16")).err().unwrap();
        assert!(e.contains("has no spatial lowering"), "{e}");
        assert!(e.contains("elemwise.hc_expand"), "{e}");
    }

    /// A multimodal deployment (what `pie serve` identifies from a checkpoint with its vision
    /// tower) plans the language model's program: its encoder's tables and dims are not the
    /// model's.
    #[test]
    fn a_vision_deployment_plans_its_language_model() {
        let want = plan(&trace("gemma4-e4b-bf16-kv-bf16"), &Ask::preset("e4b"))
            .unwrap()
            .plan;
        let got = plan(
            &trace("gemma4-e4b-vision-bf16-kv-bf16"),
            &Ask::preset("e4b"),
        )
        .unwrap()
        .plan;
        assert_eq!(
            serde_json::to_value(&got).unwrap(),
            serde_json::to_value(&want).unwrap()
        );
    }

    /// Every shipping model plans from its trace to the cut its staged prep holds: (layout,
    /// units, layers a unit, columns, rows, a row's dims, vocabulary rows a head PE).
    #[test]
    fn shipping_models_plan_from_their_traces() {
        for (name, preset, want) in [
            (
                "qwen35-d9b-bf16-kv-bf16",
                "9b",
                (Layout::Snake, 8, 4, 256, 256, 16, 216),
            ),
            (
                "qwen35-d4b-bf16-kv-bf16",
                "4b",
                (Layout::Snake, 8, 4, 256, 192, 14, 216),
            ),
            (
                "qwen35-d0.8b-bf16-kv-bf16",
                "0.8b",
                (Layout::Snake, 6, 4, 256, 128, 8, 216),
            ),
            (
                "qwen3-8b-bf16-kv-bf16",
                "q3-8b",
                (Layout::Snake, 9, 4, 256, 228, 18, 132),
            ),
            (
                "gemma4-e2b-bf16-kv-bf16",
                "e2b",
                (Layout::Stacked, 7, 5, 256, 128, 12, 232),
            ),
            (
                "gemma4-e4b-bf16-kv-bf16",
                "e4b",
                (Layout::Stacked, 7, 6, 256, 160, 16, 232),
            ),
            (
                "gemma4-12b-bf16-kv-bf16",
                "g12",
                (Layout::Runs, 8, 6, 344, 283, 12, 232),
            ),
            (
                "gptoss-20b-bf16-mxfp4-kv-bf16",
                "full",
                (Layout::Split, 12, 2, 352, 160, 18, 272),
            ),
            (
                "llama31-8b-bf16-kv-bf16",
                "l3-8b",
                (Layout::Snake, 8, 4, 256, 256, 16, 112),
            ),
        ] {
            // with no preset named, each picks its shipping one
            let chosen = plan(&trace(name), &Ask::default()).unwrap();
            assert_eq!(chosen.preset.name, preset, "{name}");
            let p = plan(&trace(name), &Ask::preset(preset)).unwrap().plan;
            let q = &p.partition;
            let got = (
                Layout::of_shape(&p.shape),
                q.units,
                q.layers_per_unit,
                q.cols,
                q.rows,
                q.row_dims[0].dims,
                q.head.vocab_rows,
            );
            assert_eq!(got, want, "{name}");
        }
    }

    /// Llama 3.1 8B: dense attention with no q / k norm, its rotation Llama 3's (the long waves
    /// stretched: frequencies of their own), eps 1e-5; with no preset named, the first that fits.
    #[test]
    fn llama_plans_from_its_trace() {
        let p = plan(&trace("llama31-8b-bf16-kv-bf16"), &Ask::default())
            .unwrap()
            .plan;
        let s = &p.shape;
        assert_eq!(Layout::of_shape(&p.shape), Layout::Snake);
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
    /// the preset holds refuse it.
    #[test]
    fn positions_bound_the_plan() {
        let t = trace("qwen35-d9b-bf16-kv-bf16");
        let ask = |positions| Ask {
            preset: Some("9b".into()),
            positions,
            ..Ask::default()
        };
        let full = plan(&t, &ask(None)).unwrap().plan;
        assert_eq!(
            (full.contract.positions, full.contract.rope_levels),
            (16384, 5)
        );
        assert_eq!(full.levers, "NPX:1,DIAG:0,TSTAMP:0,XKEEP:0,HACC16:0,ATTL:2");
        let short = plan(&t, &ask(Some(4096))).unwrap().plan;
        assert_eq!(
            (short.contract.positions, short.contract.rope_levels),
            (4096, 4)
        );
        let e = plan(&t, &ask(Some(1 << 20))).unwrap_err();
        assert!(e.contains("of at most 16384"), "{e}");
    }

    /// Overrides that leave the head short of the vocabulary, or the images short of the stack's
    /// room, are refused by name (the chip ran 9B VR=144 with 166,032 of its 248,320 tokens, and
    /// 9B PY=224 failed at link); Llama's VR=144 holds its vocabulary and plans.
    #[test]
    fn overrides_that_do_not_hold_are_refused() {
        let t = trace("qwen35-d9b-bf16-kv-bf16");
        let set = |s: &str| Ask {
            preset: Some("9b".into()),
            set: s.into(),
            ..Ask::default()
        };
        let e = plan(&t, &set("VR=144,HR=1153")).unwrap_err();
        assert!(e.contains("hold 166032 of 248320 tokens"), "{e}");
        let e = plan(&t, &set("PY=224")).unwrap_err();
        assert!(e.contains("of stack room"), "{e}");
        let l = Ask {
            preset: Some("l3-8b".into()),
            set: "VR=144".into(),
            ..Ask::default()
        };
        plan(&trace("llama31-8b-bf16-kv-bf16"), &l).unwrap();
    }

    /// With no pinned plan made for the shape, the search plans it: Qwen3.5-0.8B with another
    /// MLP width (no shipping model's shape) plans from its layout's pinned plans and builds. A
    /// pinned plan of another layout is refused by name.
    #[test]
    fn a_shape_with_no_pinned_plan_is_searched() {
        let mut s = shape(&trace("qwen35-d0.8b-bf16-kv-bf16")).unwrap();
        s.mlp.intermediate += 256;
        let p = plan_shape(s, &Ask::default()).unwrap();
        Program::of(p.plan).unwrap();
        let e = plan(&trace("qwen35-d0.8b-bf16-kv-bf16"), &Ask::preset("e2b")).unwrap_err();
        assert!(e.contains("is a stacked plan"), "{e}");
    }

    /// The plans a load may fall back on start with the one `plan` chooses: a shipping model's
    /// pinned plan.
    #[test]
    fn the_fallback_plans_start_with_the_choice() {
        let t = trace("llama31-8b-bf16-kv-bf16");
        let all = plans(&t, &Ask::default()).unwrap();
        assert_eq!(all[0].preset.name, "l3-8b");
        assert_eq!(all[0].plan, plan(&t, &Ask::default()).unwrap().plan);
    }

    /// A partition is refused where it cannot link or serve: wider than the wafer, a head's dims
    /// not even over its columns; the search, moving the keys, finds none for an MLP no column
    /// count of the snake divides.
    #[test]
    fn fit_refuses_what_cannot_serve() {
        let t = trace("llama31-8b-bf16-kv-bf16");
        let ask = |set: &str| Ask {
            preset: Some("l3-8b".into()),
            set: set.into(),
            ..Ask::default()
        };
        let e = plan(&t, &ask("PX=320")).unwrap_err();
        assert!(e.contains("dims over 40 columns"), "{e}");
        let e = plan(&t, &ask("PX=288")).unwrap_err();
        assert!(e.contains("dims over 36 columns"), "{e}");
        let e = plan(&t, &ask("PX=512")).unwrap_err();
        assert!(e.contains("over the wafer"), "{e}");
        let mut odd = shape(&t).unwrap();
        odd.mlp.intermediate += 64;
        let e = plan_shape(odd, &Ask::default()).unwrap_err();
        assert!(e.contains("the search found no partition"), "{e}");
    }

    /// The cost model reads the chip's step of each measured model off its shipping plan, and
    /// the search, walking from the presets, finds nothing it predicts faster for the 9B than the
    /// preset made for it.
    #[test]
    fn the_search_keeps_the_9b_preset() {
        let t = trace("qwen35-d9b-bf16-kv-bf16");
        let s = shape(&t).unwrap();
        let shipped = plan(&t, &Ask::default()).unwrap().plan;
        let layer = crate::cost::time::layer(&s, Some(&shipped));
        let ms = crate::cost::time::step(&shipped, layer);
        assert!((ms - 0.5100).abs() < 1e-9, "{ms}");
        let ask = crate::search::Asked {
            positions: Some(shipped.contract.positions),
            ..Default::default()
        };
        let best = crate::search::search(&s, &ask, 1).unwrap().remove(0);
        assert!((best.ms - ms).abs() < 1e-9, "{} vs {ms}", best.ms);
    }

    /// A model no preset fits (Qwen3-30B-A3B mini: four layers, four KV heads, a 2,048 hidden,
    /// q / k head norms, 16-bit experts) is planned by the search, its plan cuts from the
    /// trace's params, and its program builds.
    #[test]
    fn a_model_no_preset_fits_is_planned_by_the_search() {
        let t = trace("qwen3-30b-a3b-mini-bf16-kv-bf16");
        assert!(plans(&t, &Ask::preset("full")).is_err());
        let p = plan(&t, &Ask::default()).unwrap();
        assert!(p.preset.name.starts_with("search("), "{}", p.preset.name);
        let shapes = |n: &str| {
            t.params
                .iter()
                .find(|q| q.name == n)
                .map(|q| q.shape.clone())
        };
        crate::tiles::spec_of(&p.plan, &shapes)
            .unwrap_or_else(|e| panic!("the search's plan cuts: {e}"));
        Program::of(p.plan).unwrap();
    }

    /// The search proposes only plans that cut: gpt-oss's pick (its moves of the attention
    /// columns and segments) cuts from the trace's params.
    #[test]
    fn the_searchs_pick_cuts() {
        let t = trace("gptoss-20b-bf16-mxfp4-kv-bf16");
        let s = shape(&t).unwrap();
        let shapes = |n: &str| {
            t.params
                .iter()
                .find(|q| q.name == n)
                .map(|q| q.shape.clone())
        };
        for found in crate::search::search(&s, &crate::search::Asked::default(), 3).unwrap() {
            crate::tiles::spec_of(found.plan(), &shapes)
                .unwrap_or_else(|e| panic!("{}: {e}", found.planned.preset.name));
        }
    }

    /// A combination no kernel runs is refused by name: q / k / v biases with head norms on the
    /// MoE halves.
    #[test]
    fn a_combination_the_kernels_do_not_run_is_refused_by_name() {
        let mut s = shape(&trace("qwen3-30b-a3b-mini-bf16-kv-bf16")).unwrap();
        s.attn.bias = true;
        let e = plan_shape(s, &Ask::default()).unwrap_err();
        assert!(e.contains("biases together with q / k head norms"), "{e}");
    }

    /// The memory model, anchored on one measured program, predicts another of the same kernels'
    /// within the search's margin: Qwen3.5-4B's weight and head images off the 0.8B's link.
    #[test]
    fn memory_follows_the_partition_within_the_margin() {
        use crate::cost::memory::{self, ANCHORS};
        let small = ANCHORS.iter().find(|a| a.preset == "0.8b").unwrap();
        // the 4B's program as linked then: its pin before the A4 move (PY=192, NE=86, G2=12, G2Y=12)
        let ask = Ask {
            preset: Some("4b".into()),
            set: "PY=256,NE=64,G2=16,G2Y=16".into(),
            ..Ask::default()
        };
        let p = plan(&trace("qwen35-d4b-bf16-kv-bf16"), &ask).unwrap().plan;
        // the 4B as linked: weight 34,848 B, head 27,968 B, every other image's room 21,568 B
        let want = (memory::PE_BYTES - 34848.0)
            .min(memory::PE_BYTES - 27968.0)
            .min(21568.0);
        let got = memory::room(&p, small).min(21568.0);
        assert!(
            (got - want).abs() < crate::search::MARGIN,
            "{got} vs {want}"
        );
    }
}
