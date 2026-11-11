//! The wafer as a target of poem's spatial pass ([`poem_compiler::spatial`]): the model's
//! [`Shape`] read off its trace ([`shape`]), the family the shape needs
//! ([`super::family::serving`]), that family's presets as the partitions to try (a named one, or
//! all of them in table order), a plan that builds ([`Program::build`]) as the fit, and the table's
//! order as the cost. [`plan`] runs the pass over a trace; [`plan_shape`] does the same for a shape
//! stated outright (a checkpoint with no catalog deployment, such as the tiny test checkpoints).

use poem_compiler::spatial::{BlockKind, SpatialGraph, SpatialTarget};
use poem_ir::{
    Attn, Def, Elemwise, Gather, GluAct, Guard, Linear, NormAxis, NormWeight, Operation, RopeFreqs,
    Ssm, Trace, Turn, ValueId,
};

use super::family;
use super::plan::{
    Act, AttnSpec, FamilyId, GdnSpec, HeadSpec, LayerKind, MlpSpec, MoeSpec, NormSpec, Plan,
    PleSpec, Preset, RopeSpec, Shape, YarnSpec,
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

fn act(a: &GluAct) -> R<Act> {
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
        match &node.op {
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

/// A partition to try: a family's preset (the `order`-th tried), with the overrides set.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub family: FamilyId,
    pub preset: Preset,
    pub order: usize,
}

/// The wafer target for one shape: the preset asked for (else every one of the family's) and the
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
    /// The partitions to try: the preset made for this shape first (its shipping program), then
    /// the family's table order.
    pub fn candidates(&self) -> Vec<Candidate> {
        let id = family::serving(&self.shape);
        let mut presets = family::of(id).presets();
        let sig = self.shape.signature();
        presets.sort_by_key(|p| p.made_for != Some(sig));
        presets
            .into_iter()
            .enumerate()
            .filter(|(_, p)| self.preset.as_ref().is_none_or(|n| *n == p.name))
            .filter_map(|(order, p)| {
                let mut preset = p.set(&self.set).ok()?;
                if let Some(n) = self.positions {
                    preset.keys.insert("CTX".into(), f64::from(n));
                }
                Some(Candidate {
                    family: id,
                    preset,
                    order,
                })
            })
            .collect()
    }

    /// The plan `c` makes of the shape, serving exactly the positions asked for (if any).
    pub fn plan_of(&self, c: &Candidate) -> R<Plan> {
        let mut plan = family::of(c.family).plan(&self.shape, &c.preset)?;
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
        Program::build(family::of(c.family), plan)
            .map(|_| ())
            .map_err(|e| format!("preset {}: {e}", c.preset.name))
    }

    /// The first candidate in table order that builds; with no preset named and none that
    /// builds, the search's fastest partition ([`crate::search`]).
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
            (Some(p), true) => format!(
                "the {} family has no preset {p}",
                family::serving(&self.shape).name()
            ),
            (_, true) => "no preset to try".into(),
            _ => format!("no preset builds: {}", why.join("; ")),
        }
    }
}

impl SpatialTarget for Wafer {
    type Partition = Candidate;

    /// Every node is the wafer's: the program lays out the whole model (its blocks are the
    /// family's, chosen from the shape).
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

/// The plan of `trace` through poem's spatial pass: its shape, then the preset `ask` names (else
/// the one made for its shape, else the first of the family's that builds, else the search's)
/// with its overrides and positions.
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

/// What a plan is asked for beyond the model: a preset by name (else the first of the family's
/// that builds), overrides of its keys (`K=V,...`), and the positions it must serve.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ask {
    pub preset: Option<String>,
    pub set: String,
    pub positions: Option<u32>,
}

impl Ask {
    fn target(&self, shape: Shape) -> R<Wafer> {
        Preset::default().set(&self.set)?;
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

    /// Every shipping model plans from its trace to the cut its staged prep holds: (family,
    /// units, layers a unit, columns, rows, a row's dims, vocabulary rows a head PE).
    #[test]
    fn shipping_models_plan_from_their_traces() {
        for (name, preset, want) in [
            (
                "qwen35-d9b-bf16-kv-bf16",
                "9b",
                (FamilyId::Dense, 8, 4, 256, 256, 16, 216),
            ),
            (
                "qwen35-d4b-bf16-kv-bf16",
                "4b",
                (FamilyId::Dense, 8, 4, 256, 256, 10, 216),
            ),
            (
                "qwen35-d0.8b-bf16-kv-bf16",
                "0.8b",
                (FamilyId::Dense, 6, 4, 256, 128, 8, 216),
            ),
            (
                "qwen3-8b-bf16-kv-bf16",
                "q3-8b",
                (FamilyId::Dense, 9, 4, 256, 228, 18, 132),
            ),
            (
                "gemma4-e2b-bf16-kv-bf16",
                "e2b",
                (FamilyId::Gemma, 7, 5, 256, 128, 12, 232),
            ),
            (
                "gemma4-e4b-bf16-kv-bf16",
                "e4b",
                (FamilyId::Gemma, 7, 6, 256, 160, 16, 232),
            ),
            (
                "gemma4-12b-bf16-kv-bf16",
                "g12",
                (FamilyId::Gemma12, 8, 6, 344, 283, 12, 232),
            ),
            (
                "gptoss-20b-bf16-mxfp4-kv-bf16",
                "full",
                (FamilyId::GptOss, 12, 2, 352, 160, 18, 272),
            ),
            (
                "llama31-8b-bf16-kv-bf16",
                "l3-8b",
                (FamilyId::Dense, 8, 4, 256, 256, 16, 216),
            ),
        ] {
            // with no preset named, each picks its shipping one
            let chosen = plan(&trace(name), &Ask::default()).unwrap();
            assert_eq!(chosen.preset.name, preset, "{name}");
            let p = plan(&trace(name), &Ask::preset(preset)).unwrap().plan;
            let q = &p.partition;
            let got = (
                p.family,
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
        assert_eq!(p.family, FamilyId::Dense);
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

    /// With no preset named and none made for the shape, the pass keeps the first of the
    /// family's that builds: Qwen3.5-0.8B's 24 layers with another MLP width leave the 9B's eight
    /// unit slots two short, so it takes the 0.8B's.
    #[test]
    fn the_first_preset_that_builds_is_kept() {
        let mut s = shape(&trace("qwen35-d0.8b-bf16-kv-bf16")).unwrap();
        s.mlp.intermediate += 256;
        let p = plan_shape(s, &Ask::default()).unwrap();
        assert_eq!(p.preset.name, "0.8b");
        assert!(plan(&trace("qwen35-d0.8b-bf16-kv-bf16"), &Ask::preset("e2b")).is_err());
    }

    /// The plans a load may fall back on start with the one `plan` chooses.
    #[test]
    fn the_fallback_plans_start_with_the_choice() {
        let t = trace("llama31-8b-bf16-kv-bf16");
        let all = plans(&t, &Ask::default()).unwrap();
        assert_eq!(all[0].preset.name, "l3-8b");
        assert_eq!(all[0].plan, plan(&t, &Ask::default()).unwrap().plan);
        assert!(all.len() > 1);
    }

    /// A partition is refused where it cannot link or serve: wider than the wafer, a head's dims
    /// not even over its columns, the MLP not even over the unit's (and the search, moving the
    /// keys not set, finds none for the wider MLP either).
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
        assert!(e.contains("the MLP's 14400 over 256 columns"), "{e}");
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
        assert!((ms - 0.5238).abs() < 1e-9, "{ms}");
        let ask = crate::search::Asked {
            positions: Some(shipped.contract.positions),
            ..Default::default()
        };
        let best = crate::search::search(&s, &ask, 1).unwrap().remove(0);
        assert!((best.ms - ms).abs() < 1e-9, "{} vs {ms}", best.ms);
    }

    /// A model no preset fits (Qwen3-30B-A3B mini: four layers, four KV heads, a 2,048 hidden)
    /// is planned by the search, and its program builds.
    #[test]
    fn a_model_no_preset_fits_is_planned_by_the_search() {
        let t = trace("qwen3-30b-a3b-mini-bf16-kv-bf16");
        assert!(plans(&t, &Ask::preset("full")).is_err());
        let p = plan(&t, &Ask::default()).unwrap();
        assert!(p.preset.name.starts_with("search("), "{}", p.preset.name);
        Program::of(p.plan).unwrap();
    }

    /// The memory model, anchored on one measured program, predicts another of the same kernels'
    /// within the search's margin: Qwen3.5-4B's weight and head images off the 0.8B's link.
    #[test]
    fn memory_follows_the_partition_within_the_margin() {
        use crate::cost::memory::{self, ANCHORS};
        let small = ANCHORS.iter().find(|a| a.preset == "0.8b").unwrap();
        let p = plan(&trace("qwen35-d4b-bf16-kv-bf16"), &Ask::default())
            .unwrap()
            .plan;
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
